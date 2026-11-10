//! Immutable journey predecessor nodes and reconstruction.

use fixedbitset::FixedBitSet;

use crate::algorithm::label_bag::LabelBag;
use crate::ids::{RouteIdx, StopIdx, TripIdx};
use crate::journey::{Journey, JourneyStep};
use crate::label::Label;
use crate::time::{Duration, SecondOfDay};

#[derive(Debug, Clone, Copy)]
pub(crate) enum Step {
    Boarded {
        from: StopIdx,
        route: RouteIdx,
        trip: TripIdx,
        board_pos: u32,
        alight_pos: u32,
    },
    Walked {
        from: StopIdx,
        departure: SecondOfDay,
        arrival: SecondOfDay,
    },
}

#[derive(Debug, Clone, Copy)]
struct Node {
    parent: Option<usize>,
    stop: StopIdx,
    step: Step,
}

#[derive(Default)]
pub(crate) struct BoardingTree {
    nodes: Vec<Node>,
}

impl BoardingTree {
    pub(crate) fn clear(&mut self) {
        self.nodes.clear();
    }

    pub(crate) fn insert(&mut self, parent: Option<usize>, stop: StopIdx, step: Step) -> usize {
        let id = self.nodes.len();
        self.nodes.push(Node { parent, stop, step });
        id
    }
}

fn reconstruct_journey(
    tree: &BoardingTree,
    origins: &FixedBitSet,
    mut node_id: usize,
) -> Option<(StopIdx, Vec<JourneyStep>)> {
    let mut steps = Vec::new();
    let origin = loop {
        let node = tree.nodes.get(node_id)?;
        let previous = match node.step {
            Step::Boarded {
                from,
                route,
                trip,
                board_pos,
                alight_pos,
            } => {
                steps.push(JourneyStep::Transit {
                    route,
                    trip,
                    board: from,
                    alight: node.stop,
                    board_pos,
                    alight_pos,
                });
                from
            }
            Step::Walked {
                from,
                departure,
                arrival,
            } => {
                steps.push(JourneyStep::Walk {
                    from,
                    to: node.stop,
                    departure,
                    arrival,
                });
                from
            }
        };
        match node.parent {
            Some(parent) => node_id = parent,
            None if origins.contains(previous.idx()) => break previous,
            None => return None,
        }
    };
    steps.reverse();
    Some((origin, steps))
}

pub(crate) fn extract_target_journeys<L: Label>(
    ctx: &L::Ctx,
    labels: &[Vec<LabelBag<L>>],
    board_detail: &BoardingTree,
    origin_set: &FixedBitSet,
    targets: &[(StopIdx, Duration)],
    transfers: usize,
) -> Vec<Journey<L>> {
    let mut journeys = Vec::new();
    for &(target, walk) in targets {
        for bags in labels.iter().take(transfers + 1).skip(1) {
            for (raw_label, node) in bags[target.idx()].iter_with_node() {
                if raw_label.arrival() == SecondOfDay::MAX {
                    continue;
                }
                let Some((origin, steps)) =
                    node.and_then(|id| reconstruct_journey(board_detail, origin_set, id))
                else {
                    continue;
                };
                let plan = steps
                    .iter()
                    .filter_map(|step| match step {
                        JourneyStep::Transit { route, alight, .. } => Some((*route, *alight)),
                        _ => None,
                    })
                    .collect();
                let trips = steps
                    .iter()
                    .filter_map(|step| match step {
                        JourneyStep::Transit { trip, .. } => Some(*trip),
                        _ => None,
                    })
                    .collect();
                journeys.push(Journey {
                    origin,
                    target,
                    target_walk: walk,
                    plan,
                    trips,
                    steps,
                    label: raw_label.extend_by_footpath(ctx, target, target, walk),
                });
            }
        }
    }
    journeys
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predecessor_nodes_remain_stable_for_identical_arrivals() {
        let origin = StopIdx::new(0);
        let target = StopIdx::new(1);
        let mut origins = FixedBitSet::with_capacity(2);
        origins.insert(origin.idx());
        let mut tree = BoardingTree::default();
        let node = |trip| Step::Boarded {
            from: origin,
            route: RouteIdx::new(trip),
            trip: TripIdx::new(trip),
            board_pos: 0,
            alight_pos: 1,
        };
        let first = tree.insert(None, target, node(1));
        let second = tree.insert(None, target, node(2));
        let (_, steps) = reconstruct_journey(&tree, &origins, first).unwrap();
        assert!(matches!(steps[0], JourneyStep::Transit { trip, .. } if trip == TripIdx::new(1)));
        let (_, steps) = reconstruct_journey(&tree, &origins, second).unwrap();
        assert!(matches!(steps[0], JourneyStep::Transit { trip, .. } if trip == TripIdx::new(2)));
    }
}
