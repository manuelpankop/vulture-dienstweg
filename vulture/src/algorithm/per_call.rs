//! Per-call RAPTOR driver: [`run_per_call_query`] is the entry point
//! invoked by `Query::run_with_cache`; [`run_raptor_rounds`] is the
//! shared rounds loop reused by the rRAPTOR scan.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::collections::HashSet;

use fixedbitset::FixedBitSet;
use smallvec::SmallVec;

use crate::RaptorCache;
use crate::Timetable;
use crate::algorithm::boarding::BoardingTree;
use crate::algorithm::boarding::Step;
use crate::algorithm::boarding::extract_target_journeys;
use crate::algorithm::footpaths::relax_footpaths_round;
use crate::algorithm::footpaths::relax_footpaths_round_closed;
use crate::algorithm::label_bag::LabelBag;
use crate::endpoints::IntoEndpoints;
use crate::ids::RouteIdx;
use crate::ids::StopIdx;
use crate::ids::TripIdx;
use crate::journey::Journey;
use crate::label::Label;
use crate::time::Duration;
use crate::time::SecondOfDay;

type RidingEntry<L> = (L, TripIdx, StopIdx, u32, Option<usize>);

struct RouteBag<L: Label> {
    items: SmallVec<[RidingEntry<L>; 8]>,
    index: Option<HashMap<u64, SmallVec<[usize; 1]>>>,
    boarded_trips: HashSet<TripIdx>,
}

impl<L: Label> RouteBag<L> {
    fn new() -> Self {
        Self { items: SmallVec::new(), index: None, boarded_trips: HashSet::new() }
    }
}

#[cfg(test)]
mod route_bag_tests {
    use super::*;
    use crate::label::TripContext;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct TestLabel {
        group: u32,
        arrival: SecondOfDay,
    }

    impl Label for TestLabel {
        type Ctx = ();
        const UNREACHED: Self = Self { group: 0, arrival: SecondOfDay::MAX };
        fn from_departure(_: &(), at: SecondOfDay) -> Self {
            Self { group: 0, arrival: at }
        }
        fn extend_by_trip(self, _: &(), trip: TripContext) -> Self {
            Self { arrival: trip.arrival, ..self }
        }
        fn extend_by_footpath(self, _: &(), _: StopIdx, _: StopIdx, walk: Duration) -> Self {
            Self { arrival: self.arrival + walk, ..self }
        }
        fn dominates(&self, other: &Self) -> bool {
            self.group == other.group && self.arrival <= other.arrival
        }
        fn dominance_partition(&self) -> Option<u64> {
            Some(u64::from(self.group % 13))
        }
        fn arrival(&self) -> SecondOfDay {
            self.arrival
        }
    }

    #[test]
    fn indexed_route_bag_matches_linear_algorithm() {
        let mut bag = RouteBag::new();
        let mut reference: Vec<RidingEntry<TestLabel>> = Vec::new();
        for i in 0..2000 {
            let label = TestLabel {
                group: (i * 37 + i / 7) % 103,
                arrival: SecondOfDay((i * 271 + i / 3) % 997),
            };
            let trip = TripIdx::new((i * 11) % 300);
            let entry = (label, trip, StopIdx::new(i % 11), i, None);
            let departure_of = |trip: TripIdx| SecondOfDay(trip.get());
            let departure = departure_of(trip);
            if !reference.iter().any(|(old, trip, _, _, _)| {
                old.dominates(&label) && departure_of(*trip) <= departure
            }) {
                reference.retain(|(old, trip, _, _, _)| {
                    !(label.dominates(old) && departure <= departure_of(*trip))
                });
                reference.push(entry);
            }
            bag.insert(entry, departure, departure_of);
            assert_eq!(bag.items.as_slice(), reference);
        }
        bag.clear();
        assert!(bag.items.is_empty());
    }

    #[derive(Debug, Clone, Copy)]
    struct SameTripLabel(u32);

    impl Label for SameTripLabel {
        type Ctx = ();
        const UNREACHED: Self = Self(u32::MAX);
        const SAME_TRIP_BOARDINGS_EQUIVALENT: bool = true;
        fn from_departure(_: &(), _: SecondOfDay) -> Self { Self(0) }
        fn extend_by_trip(self, _: &(), _: TripContext) -> Self { self }
        fn extend_by_footpath(self, _: &(), _: StopIdx, _: StopIdx, _: Duration) -> Self { self }
        fn arrival(&self) -> SecondOfDay { SecondOfDay(self.0) }
    }

    #[test]
    fn equivalent_boardings_merge_only_the_exact_same_trip() {
        let mut bag = RouteBag::new();
        let trip_a = TripIdx::new(1);
        let trip_b = TripIdx::new(2);
        let departure_of = |_| SecondOfDay(100);
        bag.insert((SameTripLabel(10), trip_a, StopIdx::new(1), 0, None), SecondOfDay(100), departure_of);
        bag.insert((SameTripLabel(20), trip_a, StopIdx::new(2), 1, None), SecondOfDay(100), departure_of);
        bag.insert((SameTripLabel(30), trip_b, StopIdx::new(2), 1, None), SecondOfDay(100), departure_of);
        assert_eq!(bag.items.len(), 2);
        assert_eq!(bag.items[0].0.0, 10);
        assert_eq!(bag.items[1].1, trip_b);
        bag.clear();
        bag.insert((SameTripLabel(40), trip_a, StopIdx::new(1), 0, None), SecondOfDay(100), departure_of);
        assert_eq!(bag.items.len(), 1);
    }
}

impl<L: Label> RouteBag<L> {
    fn clear(&mut self) {
        self.items.clear();
        self.index = None;
        self.boarded_trips.clear();
    }

    fn rebuild_index(&mut self) {
        let mut index: HashMap<u64, SmallVec<[usize; 1]>> = HashMap::new();
        for (i, entry) in self.items.iter().enumerate() {
            let key = entry.0.dominance_partition().expect("partitioned label must have a key");
            index.entry(key).or_default().push(i);
        }
        self.index = Some(index);
    }

    fn insert(
        &mut self,
        entry: RidingEntry<L>,
        departure: SecondOfDay,
        departure_of: impl Fn(TripIdx) -> SecondOfDay,
    ) {
        if L::SAME_TRIP_BOARDINGS_EQUIVALENT {
            if self.boarded_trips.insert(entry.1) {
                self.items.push(entry);
            }
            return;
        }
        let candidate = entry.0;
        if let (Some(key), Some(index)) = (candidate.dominance_partition(), &self.index) {
            let bucket = index.get(&key);
            if bucket.is_some_and(|indices| indices.iter().any(|&i| {
                let (existing, trip, _, _, _) = self.items[i];
                existing.dominates(&candidate) && departure_of(trip) <= departure
            })) {
                return;
            }
            let removes = bucket.is_some_and(|indices| indices.iter().any(|&i| {
                let (existing, trip, _, _, _) = self.items[i];
                candidate.dominates(&existing) && departure <= departure_of(trip)
            }));
            if removes {
                self.items.retain(|(existing, trip, _, _, _)| {
                    !(candidate.dominates(existing) && departure <= departure_of(*trip))
                });
                self.items.push(entry);
                self.rebuild_index();
            } else {
                self.index.as_mut().expect("index exists")
                    .entry(key).or_default().push(self.items.len());
                self.items.push(entry);
            }
            return;
        }

        if self.items.iter().any(|(existing, trip, _, _, _)| {
            existing.dominates(&candidate) && departure_of(*trip) <= departure
        }) {
            return;
        }
        self.items.retain(|(existing, trip, _, _, _)| {
            !(candidate.dominates(existing) && departure <= departure_of(*trip))
        });
        self.items.push(entry);
        if self.items.len() >= 32 && candidate.dominance_partition().is_some() {
            self.rebuild_index();
        }
    }
}

// `earliest_accessible_trip` lives on the `Timetable` trait now so
// adapters with direct access to per-route trip lists can walk by
// trip-index instead of by time, avoiding the simultaneous-departure
// leapfrog bug. See `Timetable::earliest_accessible_trip`.

/// Run round-0 footpath relaxation followed by rounds `1..=transfers`
/// against an already-seeded cache. Shared by the per-call algorithm
/// (which seeds a single τ) and the rRAPTOR scan (which re-seeds for
/// each τ in descending order); the only difference between callers
/// is the outer τ-loop that handles seeding.
///
/// **Caller invariants on entry:**
///   - `marked_stops` contains the round-0 starting set: at minimum
///     the sources, plus (for rRAPTOR) any stops where new trips just
///     became catchable at the current τ.
///   - `labels[0]` is seeded at every source with that source's
///     departure label; the corresponding `best_arrival` and
///     `ever_reached` bits are set.
///   - `best_arrival` and `ever_reached` are coherent with `labels`
///     (the helper reads and writes both).
///
/// On exit, `marked_stops` is empty (drained by the early-out check
/// or by the natural marked-stops migration through rounds).
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_raptor_rounds<T: Timetable + ?Sized, L: Label>(
    tt: &T,
    ctx: &L::Ctx,
    labels: &mut [Vec<LabelBag<L>>],
    best_arrival: &mut [LabelBag<L>],
    board_detail: &mut BoardingTree,
    marked_stops: &mut FixedBitSet,
    q_entry: &mut [Option<u32>],
    q_routes: &mut Vec<RouteIdx>,
    walked_buf: &mut Vec<StopIdx>,
    relax_heap: &mut BinaryHeap<Reverse<(SecondOfDay, u32)>>,
    ever_reached: &mut FixedBitSet,
    transfers: usize,
    require_wheelchair_accessible: bool,
    targets: &[(StopIdx, Duration)],
) {
    // Pruning bound: a candidate label is rejected by `insert_into_bag`
    // and the route-scan alight check when *every* target slot's
    // `best_arrival` bag already weakly dominates the candidate (see
    // `label_bag::all_targets_dominate`). With per-target output
    // semantics — where same-stop-different-walk targets and different-
    // stop targets are distinct destinations — this is the correct
    // analog of the older scalar cross-target `pt_threshold`: a label
    // can still produce a non-dominated journey at *some* target if
    // that target's bag does not dominate it, even when a different
    // target is reached faster. `target_walk` cancels out of the
    // dominance comparison so the check operates on raw labels.

    // Pick the per-round footpath relaxation strategy once. Closed
    // graphs use a single-pass O(E) walk; non-closed graphs need
    // multi-source Dijkstra to chain walks to a fixed point.
    let footpaths_closed = tt.footpaths_are_transitively_closed();

    if footpaths_closed {
        relax_footpaths_round_closed(
            tt,
            ctx,
            0,
            labels,
            best_arrival,
            board_detail,
            marked_stops,
            targets,
            walked_buf,
            ever_reached,
        );
    } else {
        relax_footpaths_round(
            tt,
            ctx,
            0,
            labels,
            best_arrival,
            board_detail,
            marked_stops,
            targets,
            walked_buf,
            relax_heap,
            ever_reached,
        );
    }
    for s in walked_buf.drain(..) {
        marked_stops.insert(s.idx());
    }

    for k in 1..=transfers {
        #[cfg(feature = "profiling")]
        let started = std::time::Instant::now();
        #[cfg(feature = "profiling")]
        let (mut routes_scanned, mut stops_scanned, mut candidates, mut alights, mut max_route_bag) = (0u64, 0u64, 0u64, 0u64, 0usize);
        #[cfg(feature = "profiling")]
        let (mut pruned_candidates, mut pruned_alights) = (0u64, 0u64);
        // Sparse carry-forward: only clone bags at stops ever
        // reached so far this query. For real feeds the reached
        // set is small (<10% of stops in typical city queries),
        // so this avoids the 50k-stop dense memcpy that would
        // otherwise dominate per-round overhead.
        let (prev_labels, this_labels) = labels.split_at_mut(k);
        let src = &prev_labels[k - 1];
        let dst = &mut this_labels[0];
        for bit in ever_reached.ones() {
            dst[bit] = src[bit].clone();
        }

        // Build the route queue for this round. Each entry pairs a
        // route with the earliest position on that route from which we
        // can board this round. Stored positions are folded with `min`
        // so multiple marked stops on the same route resolve to the
        // earliest one.
        for stop_bit in marked_stops.ones() {
            let marked_stop = StopIdx::new(stop_bit as u32);
            for &(route, pos) in tt.get_routes_serving_stop(marked_stop) {
                match q_entry[route.idx()] {
                    None => {
                        q_entry[route.idx()] = Some(pos);
                        q_routes.push(route);
                    }
                    Some(prev_pos) => {
                        if pos < prev_pos {
                            q_entry[route.idx()] = Some(pos);
                        }
                    }
                }
            }
        }

        marked_stops.clear();

        // Per-route Pareto bag of riding entries. Each entry =
        // (boarding_label, current_trip, boarding_stop, board_pos); a
        // single entry suffices for `ArrivalTime` (size-1 stop bags)
        // but multi-criterion impls may carry multiple Pareto-optimal
        // entries that boarded different trips. `board_pos` is
        // threaded through so labels with per-trip context (e.g. fares)
        // can know where the rider got on.
        let mut route_bag: RouteBag<L> = RouteBag::new();

        for &route in q_routes.iter() {
            #[cfg(feature = "profiling")]
            {
                routes_scanned += 1;
                if routes_scanned % 10_000 == 0 {
                    eprintln!("raptor_progress round={k} ms={} routes={routes_scanned} candidates={candidates} alights={alights} max_route_bag={max_route_bag}", started.elapsed().as_millis());
                }
            }
            let p_pos = q_entry[route.idx()].expect("route in q_routes must have an entry");
            route_bag.clear();

            for (offset, &pi) in tt.get_stops_after(route, p_pos).iter().enumerate() {
                #[cfg(feature = "profiling")]
                { stops_scanned += 1; alights += route_bag.items.len() as u64; }
                let pos = p_pos + offset as u32;

                // 1. Alight every active riding entry at pi.
                let stop_accessible_ok =
                    !require_wheelchair_accessible || tt.stop_wheelchair_accessible(pi);
                for &(boarding_label, trip, boarding_stop, board_pos, parent_node) in route_bag.items.iter() {
                    // Skip stops where this trip doesn't allow drop-off
                    // (GTFS drop_off_type = 1). The trip still passes
                    // through `pi`, but the rider cannot disembark, so
                    // we don't update arrival labels here.
                    if !tt.drop_off_allowed(trip, pos) {
                        continue;
                    }
                    // Same skip when the query requires wheelchair
                    // access and `pi` is marked inaccessible.
                    if !stop_accessible_ok {
                        continue;
                    }
                    let arr = tt.get_arrival_time(trip, pos);
                    let new_label = boarding_label.extend_by_trip(
                        ctx,
                        crate::label::TripContext {
                            trip,
                            route,
                            board_stop: boarding_stop,
                            board_pos,
                            alight_stop: pi,
                            alight_pos: pos,
                            arrival: arr,
                        },
                    );
                    if !new_label.can_reach_target(ctx, pi) {
                        #[cfg(feature = "profiling")]
                        { pruned_alights += 1; }
                        continue;
                    }
                    // Pareto-aware prune: skip when the bag at `pi`
                    // already weakly dominates `new_label`, or when every
                    // target slot's `best_arrival` bag does (the per-
                    // target equivalent of the older scalar
                    // `pt_threshold` cross-target check). For single-
                    // criterion `ArrivalTime` each bag has exactly one
                    // entry and the check reduces to a small number of
                    // `<=` comparisons; for multi-criterion impls it
                    // preserves Pareto-incomparable front entries.
                    if best_arrival[pi.idx()].dominates_label(&new_label)
                        || crate::algorithm::label_bag::all_targets_dominate(
                            best_arrival,
                            targets,
                            &new_label,
                            ctx,
                        )
                    {
                        continue;
                    }
                    if labels[k][pi.idx()].insert(new_label) {
                        best_arrival[pi.idx()].insert(new_label);
                        let node = board_detail.insert(
                            parent_node,
                            pi,
                            Step::Boarded {
                                from: boarding_stop,
                                route,
                                trip,
                                board_pos,
                                alight_pos: pos,
                            },
                        );
                        labels[k][pi.idx()].set_last_node(node);
                        marked_stops.insert(pi.idx());
                        ever_reached.insert(pi.idx());
                    }
                }

                // 2. Try to extend route_bag with labels from
                //    labels[k-1][pi] that can catch a trip on this
                //    route at pi. Iterate the source bag directly:
                //    nothing in this loop touches `labels`, so there
                //    is no aliasing to dodge. (Earlier code copied
                //    the bag into a SmallVec scratch buffer per pos
                //    × route, which the profile pinned at ~14 % of
                //    self-time across `SmallVec::clear / triple_mut
                //    / spilled` plus the `extend` traffic.)
                let cand_bag = &labels[k - 1][pi.idx()];
                for (candidate, parent_node) in cand_bag.iter_with_node() {
                    if !candidate.can_reach_target(ctx, pi) {
                        #[cfg(feature = "profiling")]
                        { pruned_candidates += 1; }
                        continue;
                    }
                    #[cfg(feature = "profiling")]
                    { candidates += 1; }
                    let cand_arr = candidate.arrival();
                    let trip = match tt.earliest_transfer_aware_trip(
                        route,
                        cand_arr,
                        pos,
                        require_wheelchair_accessible,
                        candidate.transfer_context(),
                    ) {
                        Some(t) => t,
                        None => continue,
                    };
                    if !candidate.can_board_trip(ctx, tt, route, trip, pos) {
                        continue;
                    }
                    let trip_dep = tt.get_departure_time(trip, pos);

                    route_bag.insert((candidate, trip, pi, pos, parent_node), trip_dep, |existing| {
                        tt.get_departure_time(existing, pos)
                    });
                    #[cfg(feature = "profiling")]
                    { max_route_bag = max_route_bag.max(route_bag.items.len()); }
                }
            }
        }

        // Sparse-set reset of the route queue.
        for r in q_routes.drain(..) {
            q_entry[r.idx()] = None;
        }

        #[cfg(feature = "profiling")]
        eprintln!("raptor_scan_complete round={k} ms={}", started.elapsed().as_millis());
        if footpaths_closed {
            relax_footpaths_round_closed(
                tt,
                ctx,
                k,
                labels,
                best_arrival,
                board_detail,
                marked_stops,
                targets,
                walked_buf,
                ever_reached,
            );
        } else {
            relax_footpaths_round(
                tt,
                ctx,
                k,
                labels,
                best_arrival,
                board_detail,
                marked_stops,
                targets,
                walked_buf,
                relax_heap,
                ever_reached,
            );
        }
        for s in walked_buf.drain(..) {
            marked_stops.insert(s.idx());
        }

        #[cfg(feature = "profiling")]
        {
            let (trips, rules) = crate::profiling::take_transfer_counts();
            eprintln!("raptor round={k} ms={} routes={routes_scanned} stops={stops_scanned} candidates={candidates} alights={alights} pruned_candidates={pruned_candidates} pruned_alights={pruned_alights} max_route_bag={max_route_bag} transfer_trips={trips} transfer_rules={rules} reached={}", started.elapsed().as_millis(), ever_reached.count_ones(..));
        }
        if marked_stops.is_clear() {
            break;
        }
    }
}

/// Per-call RAPTOR query: seed origins at `depart`, run rounds, then
/// reconstruct the Pareto front of journeys at the targets.
///
/// Invoked by `Query::run_with_cache` for single-departure queries and
/// by the parallel rRAPTOR path for each fan-out departure.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_per_call_query<T: Timetable + ?Sized, L: Label>(
    tt: &T,
    ctx: &L::Ctx,
    cache: &mut RaptorCache<L>,
    transfers: usize,
    depart: SecondOfDay,
    origins: impl IntoEndpoints,
    targets: impl IntoEndpoints,
    require_wheelchair_accessible: bool,
) -> Vec<Journey<L>> {
    let origins = origins.into_endpoints();
    let targets = targets.into_endpoints();
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

    // Clear and populate the origin set used by reconstruction.
    origin_set.clear();
    for &(o, _) in origins {
        origin_set.insert(o.idx());
    }

    // Seed labels for each origin at depart + its walk-time offset.
    // Reconstruction breaks the trace loop when it hits an origin
    // (origin_set bit is set), so origins don't need a Step entry.
    for &(o, walk) in origins {
        let t = depart + walk;
        let seed = L::from_departure(ctx, t);
        if labels[0][o.idx()].insert(seed) {
            best_arrival[o.idx()].insert(seed);
            marked_stops.insert(o.idx());
            ever_reached.insert(o.idx());
        }
    }

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

    // For each target stop, enumerate every Pareto-optimal label
    // in the target's bag at every round and reconstruct its plan.
    // Effective label = bag label extended by the target's walk
    // time offset.
    let journeys =
        extract_target_journeys(ctx, labels, board_detail, origin_set, targets, transfers);

    // Per-`(target, target_walk)` Pareto filter. Each `(target_stop,
    // target_walk)` slot in the query's targets list represents a
    // distinct user destination — same `target_stop` with different
    // `target_walk`s, or different stops, are not directly comparable.
    // Cross-target dominance is intentionally NOT applied: the
    // `pt_threshold` mechanism already drops trip-based journeys whose
    // raw arrival cannot beat *any* walk-only effective at insert time,
    // so surviving journeys are per-target Pareto-non-dominated and the
    // cross-target collapse the previous filter did would lose valid
    // options whenever two slots in the targets list differ on
    // `target_walk` (the multi-target same-stop-different-walks case)
    // or whenever a multi-criterion label is Pareto-incomparable across
    // destinations.
    //
    // Within each slot the comparison is on `(plan.len, label)`. For
    // single-criterion `ArrivalTime` this collapses to "strictly
    // decreasing arrival as trip count increases" within the slot; for
    // multi-criterion impls it preserves Pareto-incomparable journeys
    // (faster-but-more-walking vs. slower-but-less-walking) at this
    // slot.
    //
    // Journeys are sorted within each slot by (plan.len ascending,
    // `arrival()` ascending) so the iteration order is deterministic.
    let mut front: Vec<Journey<L>> = Vec::with_capacity(journeys.len());
    for &(slot_target, slot_walk) in targets {
        let mut slot_journeys: Vec<Journey<L>> = journeys
            .iter()
            .filter(|j| j.target == slot_target && j.target_walk == slot_walk)
            .cloned()
            .collect();
        slot_journeys.sort_by_key(|j| (j.plan.len(), j.arrival()));
        let mut slot_front: Vec<Journey<L>> = Vec::with_capacity(slot_journeys.len());
        'outer: for j in slot_journeys {
            for f in &slot_front {
                if f.plan.len() <= j.plan.len() && f.label.dominates(&j.label) {
                    continue 'outer;
                }
            }
            slot_front.retain(|f| !(j.plan.len() <= f.plan.len() && j.label.dominates(&f.label)));
            slot_front.push(j);
        }
        front.extend(slot_front);
    }
    front
}
