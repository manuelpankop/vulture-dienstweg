//! [`LabelBag`] – Pareto front of [`Label`]s at a single `(round, stop)`
//! cell – and the [`insert_into_bag`] free helper used by the per-round
//! footpath relaxation routines.

use fixedbitset::FixedBitSet;
use smallvec::SmallVec;
use std::collections::HashMap;

use crate::K;
use crate::algorithm::boarding::BoardingTree;
use crate::algorithm::boarding::Step;
use crate::ids::StopIdx;
use crate::label::Label;
use crate::time::SecondOfDay;

/// A Pareto front of [`Label`]s at a single `(round, stop)` cell.
/// Backed by `SmallVec<[L; 8]>` – for single-criterion `ArrivalTime`
/// the bag is always size 1 and stays inline; for multi-criterion
/// impls it grows up to 8 inline before spilling.
#[derive(Debug, Clone)]
pub(crate) struct LabelBag<L: Label> {
    // Two inline entries: most bags hold one or two labels, and large
    // multi-criterion labels would otherwise inflate every cell.
    items: SmallVec<[(L, Option<usize>); 2]>,
    index: Option<Box<HashMap<u64, SmallVec<[usize; 1]>>>>,
}

impl<L: Label> LabelBag<L> {
    pub(crate) fn new() -> Self {
        Self {
            items: SmallVec::new(),
            index: None,
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    #[cfg(test)]
    pub(crate) fn iter(&self) -> impl Iterator<Item = &L> {
        self.items.iter().map(|(label, _)| label)
    }

    pub(crate) fn iter_with_node(&self) -> impl Iterator<Item = (L, Option<usize>)> + '_ {
        self.items.iter().copied()
    }

    pub(crate) fn set_last_node(&mut self, node: usize) {
        self.items.last_mut().expect("a label was just inserted").1 = Some(node);
    }

    /// Try to insert `new`. Returns `true` if added (and removes any
    /// items it strictly dominates); returns `false` if some existing
    /// item weakly dominates `new` (no change).
    pub(crate) fn insert(&mut self, new: L) -> bool {
        if let (Some(key), Some(index)) = (new.dominance_partition(), &self.index) {
            let bucket = index.get(&key);
            if bucket.is_some_and(|indices| indices.iter().any(|&i| self.items[i].0.dominates(&new))) {
                return false;
            }
            let removes = bucket.is_some_and(|indices| indices.iter().any(|&i| new.dominates(&self.items[i].0)));
            if removes {
                self.items.retain(|(item, _)| !new.dominates(item));
                self.items.push((new, None));
                self.rebuild_index();
            } else {
                self.index.as_mut().expect("index exists")
                    .entry(key).or_default().push(self.items.len());
                self.items.push((new, None));
            }
            return true;
        }
        for (item, _) in &self.items {
            if item.dominates(&new) {
                return false;
            }
        }
        self.items.retain(|(item, _)| !new.dominates(item));
        self.items.push((new, None));
        if self.items.len() >= 32 && new.dominance_partition().is_some() {
            self.rebuild_index();
        }
        true
    }

    fn rebuild_index(&mut self) {
        let mut index: HashMap<u64, SmallVec<[usize; 1]>> = HashMap::new();
        for (i, (item, _)) in self.items.iter().enumerate() {
            let key = item.dominance_partition().expect("partitioned label must have a key");
            index.entry(key).or_default().push(i);
        }
        self.index = Some(Box::new(index));
    }

    /// Minimum `arrival()` across the bag, or `SecondOfDay::MAX` if empty.
    pub(crate) fn min_arrival(&self) -> SecondOfDay {
        self.items
            .iter()
            .map(|(l, _)| l.arrival())
            .min()
            .unwrap_or(SecondOfDay::MAX)
    }

    /// Returns `true` if any item in the bag weakly dominates `candidate`.
    /// Equivalent to "this candidate would be rejected by `insert` because
    /// the bag already has something at least as good". Used by the
    /// algorithm's pruning sites to skip exploring a label that is already
    /// dominated by a known better one, avoiding the cost of `extend_*` /
    /// boarding-tree insertion. For single-criterion `ArrivalTime` the bag
    /// is size 1 and this reduces to one `<=` comparison; for multi-
    /// criterion impls the comparison is Pareto-aware via `Label::dominates`.
    pub(crate) fn dominates_label(&self, candidate: &L) -> bool {
        self.dominates_label_when(candidate, |_| true)
    }

    fn dominates_label_when(&self, candidate: &L, eligible: impl Fn(&L) -> bool) -> bool {
        if let (Some(key), Some(index)) = (candidate.dominance_partition(), &self.index) {
            return index.get(&key).is_some_and(|indices| {
                indices.iter().any(|&i| eligible(&self.items[i].0) && self.items[i].0.dominates(candidate))
            });
        }
        self.items.iter().any(|(item, _)| eligible(item) && item.dominates(candidate))
    }

    pub(crate) fn clear(&mut self) {
        self.items.clear();
        self.index = None;
    }
}

impl<L: Label> Default for LabelBag<L> {
    fn default() -> Self {
        Self::new()
    }
}

/// Try to insert `(label, step)` into `labels[k][stop]` and the
/// boarding tree. Updates `best_arrival[stop]` and marks the stop in
/// `out`. Returns `true` if any insertion happened.
///
/// Labels are rejected before the bag insert when *every* target's
/// `best_arrival` bag already weakly dominates the candidate: such a
/// label cannot lead to a Pareto-optimal journey at any target slot,
/// so any boarding-tree entry it would produce is a ghost step that
/// reconstruction would later surface as a dominated journey.
///
/// Crucially the check uses an *all-targets* quantifier, not the older
/// scalar "min effective" cross-target threshold. With per-target output
/// semantics, a label that is dominated by one target's best is still
/// worth keeping if it could lead to a non-dominated journey at *some
/// other* target (e.g. a walking-footpath label at an intermediate stop
/// that ends up at a target the label-emitter couldn't reach via
/// transit). The `target_walk` extension cancels out of the dominance
/// comparison (extending both sides by the same `target_walk` preserves
/// the dominance relation), so the check operates on raw labels against
/// raw `best_arrival` bags.
#[allow(clippy::too_many_arguments)]
pub(crate) fn insert_into_bag<L: Label>(
    labels: &mut [Vec<LabelBag<L>>],
    best_arrival: &mut [LabelBag<L>],
    board_detail: &mut BoardingTree,
    out: &mut Vec<StopIdx>,
    ever_reached: &mut FixedBitSet,
    targets: &[(StopIdx, crate::time::Duration)],
    k: K,
    stop: StopIdx,
    label: L,
    parent_node: Option<usize>,
    step: Step,
    ctx: &L::Ctx,
) -> bool {
    if all_targets_dominate(best_arrival, targets, &label, ctx) {
        return false;
    }
    let added = labels[k][stop.idx()].insert(label);
    if !added {
        return false;
    }
    let node = board_detail.insert(parent_node, stop, step);
    labels[k][stop.idx()].set_last_node(node);
    best_arrival[stop.idx()].insert(label);
    ever_reached.insert(stop.idx());
    out.push(stop);
    true
}

/// Returns `true` iff every target slot's `best_arrival` bag weakly
/// dominates `label`. The per-target pruning predicate used by the
/// algorithm's insert sites — see [`insert_into_bag`] for context.
/// Empty `targets` returns `false` (no targets means no slots dominate,
/// so the label is not pruned).
pub(crate) fn all_targets_dominate<L: Label>(
    best_arrival: &[LabelBag<L>],
    targets: &[(StopIdx, crate::time::Duration)],
    label: &L,
    ctx: &L::Ctx,
) -> bool {
    if L::TARGETS_ARE_ALTERNATIVES {
        debug_assert!(targets.iter().all(|&(_, walk)| walk == crate::time::Duration::ZERO));
        return targets.iter().any(|&(t, _)| {
            best_arrival[t.idx()].items.iter().any(|(target, _)| {
                target.can_finish_at_target(ctx) && target.dominates_at_target(label)
            })
        });
    }
    !targets.is_empty()
        && targets
            .iter()
            .all(|&(t, _)| best_arrival[t.idx()]
                .dominates_label_when(label, |target| target.can_finish_at_target(ctx)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Duration, TripContext};

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct PartitionedLabel {
        group: u32,
        arrival: SecondOfDay,
    }

    impl Label for PartitionedLabel {
        type Ctx = ();
        const UNREACHED: Self = Self { group: 0, arrival: SecondOfDay::MAX };

        fn from_departure(_: &(), at: SecondOfDay) -> Self {
            Self { group: 0, arrival: at }
        }

        fn extend_by_trip(self, _: &(), leg: TripContext) -> Self {
            Self { arrival: leg.arrival, ..self }
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
    fn indexed_bag_matches_linear_dominance_with_collisions_and_removals() {
        let mut bag = LabelBag::new();
        let mut reference: Vec<PartitionedLabel> = Vec::new();
        for i in 0..2000 {
            let candidate = PartitionedLabel {
                group: (i * 37 + i / 7) % 103,
                arrival: SecondOfDay((i * 271 + i / 3) % 997),
            };
            let dominated = reference.iter().any(|item| item.dominates(&candidate));
            assert_eq!(bag.dominates_label(&candidate), dominated);
            assert_eq!(bag.insert(candidate), !dominated);
            if !dominated {
                reference.retain(|item| !candidate.dominates(item));
                reference.push(candidate);
            }
            assert_eq!(bag.iter().copied().collect::<Vec<_>>(), reference);
        }
        bag.clear();
        assert!(bag.is_empty());
        assert!(!bag.dominates_label(&PartitionedLabel::UNREACHED));
    }
}
