//! Range-query specialisation: rRAPTOR scan
//! ([`raptor_range_rrap_arrival`]) plus its Pareto-front filter
//! ([`filter_range_pareto_front`]). Timetables mark stops whose trips
//! became catchable between successive departure times.

use crate::RangeJourney;
use crate::RaptorCache;
use crate::Timetable;
use crate::algorithm::boarding::extract_target_journeys;
use crate::algorithm::per_call::run_raptor_rounds;
use crate::endpoints::Endpoints;
use crate::label::Label;
use crate::time::SecondOfDay;

/// Pareto-profile filter for range-query output: keep entries that are
/// not weakly dominated by another on `(later depart, fewer transfers,
/// dominated label)`. Sort so the entries we'd prefer to keep come
/// first (later departure, fewer transfers, earlier arrival), then
/// sweep with mutual-domination removal.
pub(crate) fn filter_range_pareto_front<L: Label>(
    all: Vec<RangeJourney<L>>,
) -> Vec<RangeJourney<L>> {
    // Per-`(target_stop, target_walk)` slot Pareto filter. Each slot is
    // a distinct destination; range entries to different slots are not
    // directly compared. Within each slot the existing 3-D
    // (depart desc, plan_len asc, arrival asc) ordering applies.
    let mut by_slot: std::collections::BTreeMap<
        (crate::ids::StopIdx, crate::time::Duration),
        Vec<RangeJourney<L>>,
    > = std::collections::BTreeMap::new();
    for r in all {
        by_slot
            .entry((r.journey.target, r.journey.target_walk))
            .or_default()
            .push(r);
    }
    let mut front: Vec<RangeJourney<L>> = Vec::new();
    for (_, mut slot) in by_slot {
        slot.sort_by(|a, b| {
            b.depart
                .cmp(&a.depart)
                .then(a.journey.plan.len().cmp(&b.journey.plan.len()))
                .then(a.journey.arrival().cmp(&b.journey.arrival()))
        });
        let mut slot_front: Vec<RangeJourney<L>> = Vec::with_capacity(slot.len());
        'outer: for r in slot {
            for f in &slot_front {
                if f.depart >= r.depart
                    && f.journey.plan.len() <= r.journey.plan.len()
                    && f.journey.label.dominates(&r.journey.label)
                {
                    continue 'outer;
                }
            }
            slot_front.retain(|f| {
                !(r.depart >= f.depart
                    && r.journey.plan.len() <= f.journey.plan.len()
                    && r.journey.label.dominates(&f.journey.label))
            });
            slot_front.push(r);
        }
        front.extend(slot_front);
    }
    front
}

/// Range-query algorithm specialised for `Label = ArrivalTime`. Single
/// reverse-chronological scan that reuses labels across departure
/// events (rRAPTOR, paper §4). For non-`ArrivalTime` labels, the
/// parallel paths (`Query::run_par` / `Query::run_with_pool`)
/// still cover this case via the naïve batch.
///
/// `departures` must be sorted descending and deduped (the
/// `.depart_in_window(...)` builder normalises this).
///
/// **Why descending τ order:** for τ' < τ_prev, a journey valid at
/// τ_prev is also valid at τ' (the same trip with depart ≥ τ_prev ≥
/// τ' is still catchable). So descending order makes each new seed
/// monotonically improve the bag, and no improvement ever needs to
/// be undone.
///
/// **Why labels accumulate (cache reset only once):** the per-call
/// algorithm resets between queries; rRAPTOR resets only at the start
/// of the whole scan. Labels written at scan τ remain valid for any
/// τ' < τ, so persisting them across scans is correct, not stale —
/// and avoids re-deriving information already discovered.
///
/// **Why `best_arrival` accumulates too:** the per-round carry-forward
/// overwrites `labels[k][X]` at the start of each round k, but
/// `best_arrival` is never cleared. This gives the `pt_threshold`
/// pruning a tight bound across τ scans.
#[allow(clippy::too_many_arguments)]
pub(crate) fn raptor_range_rrap<T: Timetable + ?Sized, L: Label>(
    tt: &T,
    ctx: &L::Ctx,
    cache: &mut RaptorCache<L>,
    transfers: usize,
    departures: &[SecondOfDay],
    origins: Endpoints,
    targets: Endpoints,
    require_wheelchair_accessible: bool,
) -> Vec<RangeJourney<L>> {
    let origins = origins.as_slice();
    let targets = targets.as_slice();

    cache.reset_for_query(transfers, tt.n_stops() as u32, tt.n_routes() as u32);
    let RaptorCache {
        labels,
        best_arrival,
        board_detail,
        marked_stops,
        q_entry,
        q_routes,
        walked_buf,
        origin_set,
        relax_heap,
        ever_reached,
        ..
    } = cache;

    // origin_set is constant across τ scans; populate once.
    origin_set.clear();
    for &(source, _) in origins {
        origin_set.insert(source.idx());
    }

    let mut output: Vec<RangeJourney<L>> = Vec::new();
    let mut prev_tau: Option<SecondOfDay> = None;

    #[cfg(feature = "profiling")]
    eprintln!("range departures={} origins={} targets={} boardings={transfers}", departures.len(), origins.len(), targets.len());
    for &tau in departures {
        #[cfg(feature = "profiling")]
        eprintln!("range departure_seconds={}", tau.as_secs());
        // (a) Seed sources at τ + walk. `LabelBag::insert` returns
        // true iff the new label strictly improves the bag; for τ <
        // prev_tau the new label dominates the prior one, so
        // re-seeding tightens the source bag. For the first scan
        // there's nothing to dominate and the seed is added directly.
        marked_stops.clear();
        for &(source, walk) in origins {
            let seed = L::from_departure(ctx, tau + walk);
            if labels[0][source.idx()].insert(seed) {
                best_arrival[source.idx()].insert(seed);
                marked_stops.insert(source.idx());
                ever_reached.insert(source.idx());
            }
        }

        // (b) Mark stops with newly-catchable trips for this τ. The
        // window is half-open `[tau, prev_tau)` – trips departing at
        // exactly `tau` are first-catchable in this scan, while trips
        // at `prev_tau` were already covered by the previous scan.
        if let Some(prev) = prev_tau {
            tt.mark_newly_active_stops(tau, prev, marked_stops, require_wheelchair_accessible);
            // Round 1 boards only from round-0 labels. Events elsewhere
            // cannot add a boarding to this round and would queue unrelated
            // routes across the entire network.
            let inactive: Vec<_> = marked_stops
                .ones()
                .filter(|&stop| labels[0][stop].is_empty())
                .collect();
            for stop in inactive {
                marked_stops.set(stop, false);
            }
        }

        // (c) Round-0 footpath relax + rounds 1..=transfers, sharing
        // labels with previous scans.
        run_raptor_rounds(
            tt,
            ctx,
            labels,
            best_arrival,
            board_detail,
            marked_stops,
            q_entry,
            q_routes,
            walked_buf,
            relax_heap,
            ever_reached,
            transfers,
            require_wheelchair_accessible,
            targets,
        );

        // (d) Snapshot per-target journeys for this τ.
        #[cfg(feature = "profiling")]
        let reconstruction_started = std::time::Instant::now();
        let snapshot =
            extract_target_journeys(ctx, labels, board_detail, origin_set, targets, transfers);
        #[cfg(feature = "profiling")]
        eprintln!(
            "range reconstruction_ms={} journeys={}",
            reconstruction_started.elapsed().as_millis(),
            snapshot.len()
        );
        for journey in snapshot {
            output.push(RangeJourney {
                depart: tau,
                journey,
            });
        }

        prev_tau = Some(tau);
    }

    filter_range_pareto_front(output)
}
