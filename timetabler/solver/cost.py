"""Cost function (build plan Phase 3.2, 3.6).

    cost = HARD_PENALTY * (structural + hard elicited violations)
         + sum(weight * soft elicited violations)

Soft weights run 1-10 and hard constraints are charged at 1000, so no
accumulation of soft penalties can buy a single hard breach.

`CostModel` also supports **partial** evaluation, which is what makes the
annealer usable beyond toy instances: a move changes only the constraints
whose scope it meets, and only the structural clashes among courses sharing a
lecturer, group or room with it. Recomputing the whole cost each iteration is
O(courses x constraints) and does not survive an ITC-sized instance.

Partial evaluation is only as sound as its scope calculation, so
`test_solver.py` walks a random sequence of moves and asserts the
delta-tracked cost still equals a full recompute.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from ..schema.constraints import HARD_PENALTY, SOFT_WEIGHT_MAX
from .problem import ProblemIndex
from .timetable import Assignment, Timetable
from .violations import constraint_scope, structural_counts, structural_total, violations

#: Applied when a soft constraint carries no weight. Mid-scale: the parser is
#: told to use 5 when the speaker gives no sense of strength, and an unweighted
#: constraint reaching here means nobody chose otherwise.
DEFAULT_SOFT_WEIGHT = 5


def weight_of(constraint: Any, default: int = DEFAULT_SOFT_WEIGHT) -> int:
    if constraint.strength == "hard":
        return HARD_PENALTY
    weight = getattr(constraint, "weight", None)
    if weight is None:
        return default
    return min(int(weight), SOFT_WEIGHT_MAX)


@dataclass
class CostBreakdown:
    """Where the cost came from. Drives the UI's violated-constraint list."""

    total: float = 0.0
    hard_total: float = 0.0
    soft_total: float = 0.0
    structural: dict[str, int] = field(default_factory=dict)
    per_constraint: list[dict[str, Any]] = field(default_factory=list)

    @property
    def structural_violations(self) -> int:
        return sum(self.structural.values())

    @property
    def hard_violations(self) -> int:
        return self.structural_violations + sum(
            r["count"] for r in self.per_constraint if r["strength"] == "hard"
        )

    @property
    def feasible(self) -> bool:
        """No structural breach and no hard elicited breach."""
        return self.hard_violations == 0

    @property
    def violated_soft(self) -> list[dict[str, Any]]:
        return [
            r for r in self.per_constraint
            if r["strength"] == "soft" and r["count"] > 0
        ]

    @property
    def violated_hard(self) -> list[dict[str, Any]]:
        """Breached hard constraints, which the UI must report back.

        A non-empty list on a finished search means the elicited set cannot be
        satisfied together, or cannot be satisfied alongside the structural
        constraints. Phase 5.3 asks staff to relax something, so naming the
        specific constraints is what the report needs.
        """
        return [
            r for r in self.per_constraint
            if r["strength"] == "hard" and r["count"] > 0
        ]

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "hard_total": self.hard_total,
            "soft_total": self.soft_total,
            "feasible": self.feasible,
            "hard_violations": self.hard_violations,
            "structural": dict(self.structural),
            "per_constraint": list(self.per_constraint),
        }

    def summary(self) -> str:
        state = "feasible" if self.feasible else "INFEASIBLE"
        breached = ", ".join(
            f"{k}={v}" for k, v in sorted(self.structural.items()) if v
        ) or "none"
        return (
            f"cost {self.total:.0f} ({state}): hard {self.hard_total:.0f}, "
            f"soft {self.soft_total:.0f}; structural {breached}; "
            f"{len(self.violated_hard)} hard and {len(self.violated_soft)} soft "
            f"constraint(s) breached"
        )


@dataclass
class Move:
    """New placements for one or more courses.

    All three move types in Phase 3.3 reduce to this: moving a course changes
    its room and slot, changing a room keeps the slot, and swapping two
    courses exchanges both placements.
    """

    placements: dict[str, Assignment]

    @property
    def courses(self) -> tuple[str, ...]:
        return tuple(self.placements)

    @property
    def target_rooms(self) -> frozenset[str]:
        return frozenset(a.room for a in self.placements.values())

    def apply(self, tt: Timetable) -> dict[str, Assignment | None]:
        """Apply in place, returning what was there so it can be restored."""
        previous = {c: tt.get(c) for c in self.placements}
        for course_id, assignment in self.placements.items():
            tt.place(course_id, assignment.room, assignment.slot)
        return previous

    @staticmethod
    def revert(tt: Timetable, previous: dict[str, Assignment | None]) -> None:
        for course_id, assignment in previous.items():
            if assignment is None:
                tt.remove(course_id)
            else:
                tt.place(course_id, assignment.room, assignment.slot)


class CostModel:
    """Cost of a timetable under a fixed set of elicited constraints."""

    def __init__(
        self,
        index: ProblemIndex,
        constraints: Sequence[Any] = (),
        *,
        hard_penalty: int = HARD_PENALTY,
        default_soft_weight: int = DEFAULT_SOFT_WEIGHT,
    ) -> None:
        self.index = index
        self.constraints = list(constraints)
        self.hard_penalty = hard_penalty
        self.default_soft_weight = default_soft_weight
        self.weights = [
            hard_penalty if c.strength == "hard" else weight_of(c, default_soft_weight)
            for c in self.constraints
        ]
        self.scopes = [constraint_scope(c, index) for c in self.constraints]
        self._by_course: dict[str, list[int]] = {}
        for i, scope in enumerate(self.scopes):
            for course_id in scope:
                self._by_course.setdefault(course_id, []).append(i)

    # -- full evaluation --------------------------------------------------

    def total(self, tt: Timetable) -> float:
        cost = self.hard_penalty * structural_total(tt, self.index)
        for constraint, weight in zip(self.constraints, self.weights):
            count = violations(tt, constraint, self.index)
            if count:
                cost += weight * count
        return float(cost)

    def breakdown(self, tt: Timetable) -> CostBreakdown:
        out = CostBreakdown()
        out.structural = structural_counts(tt, self.index)
        out.hard_total = float(self.hard_penalty * sum(out.structural.values()))

        for constraint, weight in zip(self.constraints, self.weights):
            count = violations(tt, constraint, self.index)
            contribution = float(weight * count)
            out.per_constraint.append({
                "type": constraint.type,
                "strength": constraint.strength,
                "weight": weight,
                "count": count,
                "contribution": contribution,
                "source_text": constraint.source_text,
            })
            if constraint.strength == "hard":
                out.hard_total += contribution
            else:
                out.soft_total += contribution

        out.total = out.hard_total + out.soft_total
        return out

    # -- partial evaluation ----------------------------------------------

    def affected(
        self, move: Move, tt: Timetable
    ) -> tuple[frozenset[str], tuple[int, ...]]:
        """The course closure and constraint indices a move can change.

        The closure is every course sharing a lecturer, group, or room with a
        moved course, taking rooms from the current state **and** the move's
        targets. Closing over owners is what makes the restricted structural
        count exact: a clash needs two occupants, and restricting cells to a
        set that excludes one of them would miss it.
        """
        moved = move.courses
        lecturers = {self.index.courses[c].lecturer for c in moved}
        groups = {self.index.courses[c].group for c in moved}
        rooms = set(move.target_rooms)
        for course_id in moved:
            room = tt.room_of(course_id)
            if room is not None:
                rooms.add(room)

        closure = set(moved)
        for lecturer in lecturers:
            closure.update(self.index.courses_by_lecturer.get(lecturer, ()))
        for group in groups:
            closure.update(self.index.courses_by_group.get(group, ()))
        if rooms:
            closure.update(
                c for c in self.index.course_ids if tt.room_of(c) in rooms
            )

        indices: set[int] = set()
        for course_id in moved:
            indices.update(self._by_course.get(course_id, ()))
        return frozenset(closure), tuple(sorted(indices))

    def partial(
        self,
        tt: Timetable,
        closure: Iterable[str],
        constraint_indices: Iterable[int],
    ) -> float:
        """Cost restricted to a closed course set and given constraints."""
        cost = float(self.hard_penalty * structural_total(tt, self.index, closure))
        for i in constraint_indices:
            count = violations(tt, self.constraints[i], self.index)
            if count:
                cost += self.weights[i] * count
        return cost

    def delta(self, tt: Timetable, move: Move) -> tuple[float, dict[str, Any]]:
        """Cost change if `move` were applied. Leaves `tt` as it was found."""
        closure, indices = self.affected(move, tt)
        before = self.partial(tt, closure, indices)
        previous = move.apply(tt)
        after = self.partial(tt, closure, indices)
        Move.revert(tt, previous)
        return after - before, {"closure": closure, "indices": indices}


__all__ = [
    "DEFAULT_SOFT_WEIGHT",
    "CostBreakdown",
    "CostModel",
    "Move",
    "weight_of",
]
