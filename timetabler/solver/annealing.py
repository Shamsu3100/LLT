"""Simulated annealing (build plan Phase 3.3).

Three move types, as specified: move one course to a new room and slot, swap
two courses' placements, change one course's room. Cooling is geometric, and
the initial temperature is calibrated from the instance by sampling worsening
moves, so the schedule does not have to be retuned by hand for every
department.

The search is seeded and deterministic: the same seed and settings reproduce
a run exactly, which the paper needs (§2).
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from .cost import CostBreakdown, CostModel, Move
from .problem import ProblemIndex
from .timetable import Assignment, Timetable

MoveProposer = Callable[[Timetable, ProblemIndex, random.Random], Move | None]


# -- move generators ------------------------------------------------------


def _rooms_for(index: ProblemIndex, course_id: str) -> tuple[str, ...]:
    return index.rooms_fitting(course_id) or index.room_ids


def propose_move_course(
    tt: Timetable, index: ProblemIndex, rng: random.Random
) -> Move | None:
    """Move one course to a different room and slot."""
    course_id = rng.choice(index.course_ids)
    rooms = _rooms_for(index, course_id)
    slots = index.starts_for(course_id)
    if not rooms or not slots:
        return None
    candidate = Assignment(room=rng.choice(rooms), slot=rng.choice(slots))
    if tt.get(course_id) == candidate:
        return None
    return Move({course_id: candidate})


def propose_change_room(
    tt: Timetable, index: ProblemIndex, rng: random.Random
) -> Move | None:
    """Keep the slot, change the room."""
    course_id = rng.choice(index.course_ids)
    current = tt.get(course_id)
    if current is None:
        return None
    rooms = [r for r in _rooms_for(index, course_id) if r != current.room]
    if not rooms:
        return None
    return Move({course_id: Assignment(room=rng.choice(rooms), slot=current.slot)})


def propose_swap(
    tt: Timetable, index: ProblemIndex, rng: random.Random
) -> Move | None:
    """Exchange two courses' placements.

    Returns None when the exchange would put a course where its block does not
    fit the day, which happens whenever the two durations differ. Rejecting
    here keeps every reachable assignment inside the grid, so no overflow
    penalty is needed.
    """
    if len(index.course_ids) < 2:
        return None
    a, b = rng.sample(index.course_ids, 2)
    pa, pb = tt.get(a), tt.get(b)
    if pa is None or pb is None or pa == pb:
        return None
    if pb.slot not in index.starts_for(a) or pa.slot not in index.starts_for(b):
        return None
    return Move({a: pb, b: pa})


DEFAULT_MOVES: tuple[tuple[MoveProposer, float], ...] = (
    (propose_move_course, 0.5),
    (propose_swap, 0.3),
    (propose_change_room, 0.2),
)


# -- results --------------------------------------------------------------


@dataclass
class SolverResult:
    timetable: Timetable
    cost: float
    breakdown: CostBreakdown
    #: Iterations actually executed, which is fewer than the configured budget
    #: when the search reaches cost 0 and stops.
    iterations: int
    accepted: int
    improvements: int
    proposals_rejected: int
    seed: int
    elapsed_seconds: float
    initial_cost: float
    initial_temperature: float
    history: list[tuple[int, float, float]] = field(default_factory=list)

    @property
    def feasible(self) -> bool:
        return self.breakdown.feasible

    @property
    def acceptance_rate(self) -> float | None:
        return self.accepted / self.iterations if self.iterations else None

    def summary(self) -> str:
        return (
            f"{self.breakdown.summary()}\n"
            f"  {self.iterations} iterations in {self.elapsed_seconds:.2f}s, "
            f"seed {self.seed}, T0 {self.initial_temperature:.1f}\n"
            f"  cost {self.initial_cost:.0f} -> {self.cost:.0f}, "
            f"{self.improvements} improvements, "
            f"acceptance {(self.acceptance_rate or 0) * 100:.1f}%"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "cost": self.cost,
            "feasible": self.feasible,
            "iterations": self.iterations,
            "accepted": self.accepted,
            "improvements": self.improvements,
            "seed": self.seed,
            "elapsed_seconds": self.elapsed_seconds,
            "initial_cost": self.initial_cost,
            "initial_temperature": self.initial_temperature,
            "acceptance_rate": self.acceptance_rate,
            "breakdown": self.breakdown.as_dict(),
            "timetable": self.timetable.as_dict(),
        }


# -- the annealer ---------------------------------------------------------


class SimulatedAnnealer:
    def __init__(
        self,
        model: CostModel,
        *,
        iterations: int = 20_000,
        initial_temperature: float | None = None,
        final_temperature: float = 0.1,
        target_initial_acceptance: float = 0.8,
        moves: Sequence[tuple[MoveProposer, float]] = DEFAULT_MOVES,
        seed: int = 0,
        history_every: int = 0,
        verify_every: int = 0,
    ) -> None:
        if iterations < 1:
            raise ValueError("iterations must be at least 1")
        if not 0.0 < target_initial_acceptance < 1.0:
            raise ValueError("target_initial_acceptance must lie in (0, 1)")
        self.model = model
        self.index = model.index
        self.iterations = iterations
        self.initial_temperature = initial_temperature
        self.final_temperature = max(final_temperature, 1e-9)
        self.target_initial_acceptance = target_initial_acceptance
        self.proposers = [p for p, _ in moves]
        self.move_weights = [w for _, w in moves]
        self.seed = seed
        self.history_every = history_every
        #: When set, re-derives the cost from scratch every N accepted moves
        #: and raises on disagreement with the delta-tracked value. Used by
        #: the tests to hold partial evaluation honest.
        self.verify_every = verify_every

    # -- temperature ------------------------------------------------------

    def calibrate_temperature(
        self, tt: Timetable, rng: random.Random, samples: int = 200
    ) -> float:
        """Set T0 so that a worsening move is accepted at the target rate.

        Sampling the instance beats a hard-coded constant: the cost scale
        depends on how many hard constraints an instance carries, and a
        temperature tuned on one department would be wrong for the next.
        """
        deltas: list[float] = []
        for _ in range(samples):
            move = self._propose(tt, rng)
            if move is None:
                continue
            delta, _ = self.model.delta(tt, move)
            if delta > 0:
                deltas.append(delta)
        if not deltas:
            return 1.0
        mean_worsening = sum(deltas) / len(deltas)
        return max(mean_worsening / -math.log(self.target_initial_acceptance), 1e-6)

    def _propose(self, tt: Timetable, rng: random.Random) -> Move | None:
        proposer = rng.choices(self.proposers, weights=self.move_weights, k=1)[0]
        return proposer(tt, self.index, rng)

    # -- search -----------------------------------------------------------

    def solve(self, start: Timetable | None = None) -> SolverResult:
        rng = random.Random(self.seed)
        current = (start.copy() if start is not None else
                   Timetable.random(self.index, rng))

        started = time.perf_counter()
        current_cost = self.model.total(current)
        initial_cost = current_cost

        t0 = (self.initial_temperature if self.initial_temperature is not None
              else self.calibrate_temperature(current, random.Random(self.seed + 1)))
        cooling = (self.final_temperature / t0) ** (1.0 / max(self.iterations - 1, 1))

        best = current.copy()
        best_cost = current_cost
        accepted = improvements = rejected_proposals = 0
        history: list[tuple[int, float, float]] = []
        temperature = t0
        steps = 0

        for step in range(self.iterations):
            steps = step + 1
            move = self._propose(current, rng)
            if move is None:
                rejected_proposals += 1
                temperature *= cooling
                continue

            closure, indices = self.model.affected(move, current)
            before = self.model.partial(current, closure, indices)
            previous = move.apply(current)
            after = self.model.partial(current, closure, indices)
            delta = after - before

            if delta <= 0 or rng.random() < math.exp(-delta / temperature):
                current_cost += delta
                accepted += 1
                if current_cost < best_cost - 1e-9:
                    best_cost = current_cost
                    best = current.copy()
                    improvements += 1
                if self.verify_every and accepted % self.verify_every == 0:
                    self._verify(current, current_cost, step)
            else:
                Move.revert(current, previous)

            if self.history_every and step % self.history_every == 0:
                history.append((step, current_cost, temperature))

            temperature *= cooling
            if best_cost <= 0:
                break  # nothing left to improve

        return SolverResult(
            timetable=best,
            cost=best_cost,
            breakdown=self.model.breakdown(best),
            iterations=steps,
            accepted=accepted,
            improvements=improvements,
            proposals_rejected=rejected_proposals,
            seed=self.seed,
            elapsed_seconds=time.perf_counter() - started,
            initial_cost=initial_cost,
            initial_temperature=t0,
            history=history,
        )

    def _verify(self, tt: Timetable, tracked: float, step: int) -> None:
        exact = self.model.total(tt)
        if abs(exact - tracked) > 1e-6:
            raise AssertionError(
                f"delta evaluation drifted at step {step}: tracked {tracked}, "
                f"recomputed {exact}. The scope calculation in "
                f"CostModel.affected is under-reporting."
            )


def solve(
    index: ProblemIndex,
    constraints: Sequence[Any] = (),
    *,
    iterations: int = 20_000,
    seed: int = 0,
    start: Timetable | None = None,
    **kwargs: Any,
) -> SolverResult:
    """Convenience entry point: build the model and run one search."""
    model = CostModel(index, constraints)
    annealer = SimulatedAnnealer(model, iterations=iterations, seed=seed, **kwargs)
    return annealer.solve(start)


def solve_best_of(
    index: ProblemIndex,
    constraints: Sequence[Any] = (),
    *,
    restarts: int = 5,
    iterations: int = 20_000,
    seed: int = 0,
    **kwargs: Any,
) -> SolverResult:
    """Independent restarts from different seeds, keeping the best run.

    Annealing on a timetabling instance is prone to settling in a local
    optimum that one schedule cannot escape, and restarts are cheaper to
    reason about than a reheating schedule.
    """
    model = CostModel(index, constraints)
    best: SolverResult | None = None
    for r in range(restarts):
        annealer = SimulatedAnnealer(
            model, iterations=iterations, seed=seed + r, **kwargs
        )
        result = annealer.solve()
        if best is None or result.cost < best.cost:
            best = result
        if best.cost <= 0:
            break
    assert best is not None
    return best


__all__ = [
    "DEFAULT_MOVES",
    "MoveProposer",
    "SimulatedAnnealer",
    "SolverResult",
    "propose_change_room",
    "propose_move_course",
    "propose_swap",
    "solve",
    "solve_best_of",
]
