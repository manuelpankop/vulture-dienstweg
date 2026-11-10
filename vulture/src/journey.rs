//! [`Journey`] – what the algorithm returns: a sequence of `(route, alight stop)`
//! steps plus a label at the target. Search-generated journeys retain exact
//! boarding positions and footpaths. [`Journey::with_timing`] checks those
//! steps against the timetable; failures surface as [`TimingError`].

use crate::Timetable;
use crate::ids::RouteIdx;
use crate::ids::StopIdx;
use crate::ids::TripIdx;
use crate::label::ArrivalTime;
use crate::label::Label;
use crate::time::Duration;
use crate::time::SecondOfDay;

/// Exact steps retained from the search, including intermediate walks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JourneyStep {
    /// One directed walking edge.
    Walk {
        /// Start stop.
        from: StopIdx,
        /// End stop.
        to: StopIdx,
        /// Start time.
        departure: SecondOfDay,
        /// End time.
        arrival: SecondOfDay,
    },
    /// One selected vehicle and its exact positions in the route.
    Transit {
        /// Route ridden.
        route: RouteIdx,
        /// Trip ridden.
        trip: TripIdx,
        /// Boarding stop.
        board: StopIdx,
        /// Alighting stop.
        alight: StopIdx,
        /// Boarding position on the route.
        board_pos: u32,
        /// Alighting position on the route.
        alight_pos: u32,
    },
}

/// A journey found by the RAPTOR algorithm.
///
/// A sequence of (route, alight stop) steps plus a final label. A single
/// query may return multiple journeys, each a Pareto trade-off between
/// fewer transfers and earlier arrival.
///
/// `origin` and `target` are the stops the algorithm actually picked from
/// the user-supplied origin / target sets (relevant for multi-source /
/// multi-target queries such as parent-station expansion). `target_walk`
/// is the walk offset that was paired with `target` in the query — for a
/// query like `.to([(stop_1, 0_s), (stop_1, 30_s)])` the same `target`
/// can appear with different `target_walk`s, and journeys to those two
/// `(target, target_walk)` slots are kept separate by the output Pareto
/// filter.
///
/// `L` defaults to [`ArrivalTime`] for single-criterion routing.
#[derive(Debug, Clone)]
pub struct Journey<L: Label = ArrivalTime> {
    /// The origin stop this journey starts from, picked from the
    /// user-supplied origin set.
    pub origin: StopIdx,
    /// The target stop this journey ends at, picked from the user-supplied
    /// target set.
    pub target: StopIdx,
    /// The walk-offset for the `target` slot in the query's targets list.
    /// The walk has already been folded into `label` (via the label's
    /// [`Label::extend_by_footpath`] / equivalent), so this field is
    /// recovered metadata rather than something callers need to apply
    /// themselves.
    pub target_walk: Duration,
    /// Sequence of steps, each a (route, alight stop) pair.
    ///
    /// The origin stop is implicit – it is not part of the plan. Each entry
    /// means "take this route until this stop". The first step boards at the
    /// origin stop, and each subsequent step boards at the stop where the
    /// previous step got off (possibly via an intermediate footpath).
    pub plan: Vec<(RouteIdx, StopIdx)>,
    /// Exact trip selected for each corresponding entry in `plan`.
    pub trips: Vec<TripIdx>,
    /// Exact search steps. An empty value permits legacy manually-built plans.
    pub steps: Vec<JourneyStep>,
    /// The label at the target stop, with the target's walk-time offset
    /// already folded in. For [`ArrivalTime`], `label.arrival()` is the
    /// effective arrival time in seconds since midnight.
    pub label: L,
}

impl<L: Label> Journey<L> {
    /// Convenience accessor for `self.label.arrival()`: the effective
    /// arrival at the chosen target, with the target's walk-time offset
    /// already applied.
    pub fn arrival(&self) -> SecondOfDay {
        self.label.arrival()
    }

    /// Walk the plan against `tt` to recover the specific trip ridden and
    /// the per-leg depart / arrive times for each leg. `depart` is the
    /// original query departure time; `origin_walk` is the walk offset for
    /// `self.origin` from the original origins slice ([`Duration::ZERO`]
    /// for single-stop queries).
    ///
    /// Each [`TimedLeg`] reports the route, boarding stop, alighting stop,
    /// [`TripIdx`], and times. Exact walks are available in [`Journey::steps`];
    /// they are checked here but do not produce separate `TimedLeg`s.
    /// For legacy manually constructed journeys with empty `steps`, this
    /// instead infers at most one walking hop between successive legs.
    ///
    /// # Errors
    ///
    /// Returns a [`TimingError`] when the steps or legacy plan cannot be
    /// matched against `tt` or the requested departure.
    pub fn with_timing<T: Timetable>(
        &self,
        tt: &T,
        depart: SecondOfDay,
        origin_walk: Duration,
    ) -> Result<Vec<TimedLeg>, TimingError> {
        if !self.steps.is_empty() {
            let mut legs = Vec::with_capacity(self.plan.len());
            let mut time = depart + origin_walk;
            let mut stop = self.origin;
            for (step_index, step) in self.steps.iter().enumerate() {
                match *step {
                    JourneyStep::Walk { from, to, departure, arrival } => {
                        if from != stop || departure < time || arrival < departure
                            || !tt.get_footpaths_from(from).contains(&to)
                            || departure + tt.get_transfer_time(from, to) != arrival
                        {
                            return Err(TimingError::InvalidStep { step: step_index });
                        }
                        time = arrival;
                        stop = to;
                    }
                    JourneyStep::Transit { route, trip, board, alight, board_pos, alight_pos } => {
                        let route_stops = tt.get_stops_after(route, 0);
                        if board != stop || alight_pos <= board_pos
                            || route_stops.get(board_pos as usize) != Some(&board)
                            || route_stops.get(alight_pos as usize) != Some(&alight)
                        {
                            return Err(TimingError::InvalidStep { step: step_index });
                        }
                        let departure = tt.get_departure_time(trip, board_pos);
                        if departure < time {
                            return Err(TimingError::NoCatchableTrip {
                                leg: legs.len(), route, board_pos, at: time,
                            });
                        }
                        let arrive = tt.get_arrival_time(trip, alight_pos);
                        legs.push(TimedLeg { route, board, alight, trip, depart: departure, arrive });
                        stop = alight;
                        time = arrive;
                    }
                }
            }
            if stop != self.target {
                return Err(TimingError::InvalidStep { step: self.steps.len() });
            }
            return Ok(legs);
        }
        let mut legs = Vec::with_capacity(self.plan.len());
        let mut current_time = depart + origin_walk;
        let mut current_stop = self.origin;

        for (leg, &(route, alight)) in self.plan.iter().enumerate() {
            // Find a stop on `route` reachable from current_stop —
            // either current_stop itself or, failing that, a one-hop
            // footpath neighbour that the route serves. Pick the
            // first matching neighbour in iteration order; for the
            // typical case of one neighbour on the route this is
            // unambiguous.
            let serving_here = tt.get_routes_serving_stop(current_stop);
            // A boarding stop is only valid when the alighting stop still follows it on the
            // route; a neighbour that is itself the alighting stop would give a zero-length leg.
            let alights_after = |pos: u32| {
                tt.get_stops_after(route, pos)
                    .iter()
                    .skip(1)
                    .any(|&s| s == alight)
            };
            let (board, board_pos, walk_time) = if let Some(&(_, pos)) = serving_here
                .iter()
                .find(|(r, p)| *r == route && alights_after(*p))
            {
                (current_stop, pos, Duration::ZERO)
            } else {
                let mut found = None;
                for &neighbour in tt.get_footpaths_from(current_stop) {
                    if let Some(&(_, pos)) = tt
                        .get_routes_serving_stop(neighbour)
                        .iter()
                        .find(|(r, p)| *r == route && alights_after(*p))
                    {
                        let walk = tt.get_transfer_time(current_stop, neighbour);
                        found = Some((neighbour, pos, walk));
                        break;
                    }
                }
                found.ok_or(TimingError::NoBoardingStop {
                    leg,
                    route,
                    from_stop: current_stop,
                })?
            };

            current_time = current_time + walk_time;

            let trip = self
                .trips
                .get(leg)
                .copied()
                .ok_or(TimingError::MissingSelectedTrip { leg })?;
            let depart = tt.get_departure_time(trip, board_pos);
            if depart < current_time {
                return Err(TimingError::NoCatchableTrip {
                    leg,
                    route,
                    board_pos,
                    at: current_time,
                });
            }

            // Find the alight position by scanning forward from board_pos.
            let stops_ahead = tt.get_stops_after(route, board_pos);
            let alight_offset = stops_ahead
                .iter()
                .skip(1)
                .position(|&s| s == alight)
                .map(|offset| offset + 1)
                .ok_or(TimingError::UnreachableAlight {
                    leg,
                    route,
                    board_pos,
                    alight,
                })?;
            let alight_pos = board_pos + alight_offset as u32;
            let arrive = tt.get_arrival_time(trip, alight_pos);

            legs.push(TimedLeg {
                route,
                board,
                alight,
                trip,
                depart,
                arrive,
            });

            current_time = arrive;
            current_stop = alight;
        }

        Ok(legs)
    }
}

/// One transit leg of a [`Journey`] with reconstructed timing.
/// Produced by [`Journey::with_timing`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimedLeg {
    /// The route boarded.
    pub route: RouteIdx,
    /// Stop where the rider boards.
    pub board: StopIdx,
    /// Stop where the rider alights.
    pub alight: StopIdx,
    /// The specific trip ridden – the earliest one departing at or
    /// after the rider's available time at `board`.
    pub trip: TripIdx,
    /// Departure time at `board`, in seconds since midnight.
    pub depart: SecondOfDay,
    /// Arrival time at `alight`, in seconds since midnight.
    pub arrive: SecondOfDay,
}

/// Failure modes for [`Journey::with_timing`].
///
/// Each variant carries the leg index where reconstruction stopped (zero-based)
/// and enough context to identify what went wrong, including inconsistent
/// exact steps, legacy plans and mismatched timetables.
#[non_exhaustive]
#[derive(Debug, Clone, thiserror::Error)]
pub enum TimingError {
    /// A reconstructed step is not connected or contradicts the timetable.
    #[error("journey step {step} is inconsistent with its predecessor or timetable")]
    InvalidStep {
        /// Zero-based index of the invalid step.
        step: usize,
    },
    /// The reconstructed plan has no selected trip at this leg position.
    #[error("leg {leg}: journey has no selected trip")]
    MissingSelectedTrip {
        /// Zero-based plan index with missing trip metadata.
        leg: usize,
    },
    /// No stop in the plan-reconstruction frontier (the previous leg's
    /// alight stop or any of its one-hop footpath neighbours) is served
    /// by the route this leg wants to board. Multi-hop walk chains are
    /// not reconstructed – if the original journey actually walked
    /// through more than one intermediate stop, that walk can't be
    /// recovered from `Journey.plan` alone today.
    #[error(
        "leg {leg}: no boarding stop for route {route} reachable from {from_stop} \
         (or any one-hop walk neighbour)"
    )]
    NoBoardingStop {
        /// Zero-based plan index where reconstruction stopped.
        leg: usize,
        /// The route this leg wanted to board.
        route: RouteIdx,
        /// The stop the rider was at when reconstruction stopped (the
        /// previous leg's alight stop, or `self.origin` for `leg == 0`).
        from_stop: StopIdx,
    },
    /// No trip on the boarded route departs at or after the rider's
    /// available time. For a `Journey` produced by the same timetable
    /// this should not happen – surface as a programmer-error escape
    /// hatch.
    #[error(
        "leg {leg}: no trip on route {route} departs at or after {at} from position {board_pos}"
    )]
    NoCatchableTrip {
        /// Zero-based plan index where reconstruction stopped.
        leg: usize,
        /// The route this leg boarded.
        route: RouteIdx,
        /// The position within `route` where the rider boarded.
        board_pos: u32,
        /// The time at which the rider was ready to board.
        at: SecondOfDay,
    },
    /// The boarded route's stop sequence (from the boarding position
    /// onwards) does not include the claimed alighting stop. For a
    /// `Journey` produced by the same timetable this should not
    /// happen – surface as a programmer-error escape hatch.
    #[error("leg {leg}: route {route} does not reach stop {alight} from position {board_pos}")]
    UnreachableAlight {
        /// Zero-based plan index where reconstruction stopped.
        leg: usize,
        /// The route this leg boarded.
        route: RouteIdx,
        /// The position within `route` where the rider boarded.
        board_pos: u32,
        /// The stop the plan claimed the rider alighted at.
        alight: StopIdx,
    },
}
