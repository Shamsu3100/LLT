"""Timetable representation (build plan Phase 3.1).

`course -> (room, slot)`, where the slot is the START of a block covering the
course's duration.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, Iterator, NamedTuple

from .problem import ProblemIndex


class Assignment(NamedTuple):
    room: str
    slot: str


@dataclass
class Timetable:
    """A complete or partial assignment of courses to rooms and slots."""

    index: ProblemIndex
    assignments: dict[str, Assignment]

    # -- construction -----------------------------------------------------

    @classmethod
    def empty(cls, index: ProblemIndex) -> "Timetable":
        return cls(index=index, assignments={})

    @classmethod
    def random(
        cls,
        index: ProblemIndex,
        rng: random.Random | None = None,
        *,
        prefer_fitting_rooms: bool = True,
    ) -> "Timetable":
        """A random complete assignment.

        `prefer_fitting_rooms` restricts the draw to rooms of sufficient
        capacity where any exist, which starts the search inside the
        capacity-feasible region instead of spending early iterations
        climbing out of it.
        """
        r = rng or random.Random()
        assignments: dict[str, Assignment] = {}
        for course_id in index.course_ids:
            rooms = (
                index.rooms_fitting(course_id) if prefer_fitting_rooms else ()
            ) or index.room_ids
            assignments[course_id] = Assignment(
                room=r.choice(list(rooms)),
                slot=r.choice(list(index.starts_for(course_id))),
            )
        return cls(index=index, assignments=assignments)

    def copy(self) -> "Timetable":
        return Timetable(index=self.index, assignments=dict(self.assignments))

    # -- access -----------------------------------------------------------

    def __len__(self) -> int:
        return len(self.assignments)

    def __contains__(self, course_id: object) -> bool:
        return course_id in self.assignments

    def get(self, course_id: str) -> Assignment | None:
        return self.assignments.get(course_id)

    def room_of(self, course_id: str) -> str | None:
        a = self.assignments.get(course_id)
        return a.room if a else None

    def slot_of(self, course_id: str) -> str | None:
        a = self.assignments.get(course_id)
        return a.slot if a else None

    @property
    def complete(self) -> bool:
        return len(self.assignments) == len(self.index.course_ids)

    @property
    def unassigned(self) -> tuple[str, ...]:
        return tuple(c for c in self.index.course_ids if c not in self.assignments)

    def occupied(self, course_id: str) -> tuple[tuple[str, int], ...]:
        a = self.assignments.get(course_id)
        if a is None:
            return ()
        return self.index.occupied(course_id, a.slot)

    def day_of(self, course_id: str) -> str | None:
        a = self.assignments.get(course_id)
        return self.index.slots[a.slot].day if a else None

    def periods_on_day(self, course_ids: Iterable[str], day: str) -> list[int]:
        """Periods occupied on one day by any of the given courses."""
        out: list[int] = []
        for cid in course_ids:
            out.extend(p for d, p in self.occupied(cid) if d == day)
        return sorted(out)

    def cells(self, course_ids: Iterable[str] | None = None) -> Iterator[
        tuple[str, str, int]
    ]:
        """Yield (course_id, day, period) for every occupied cell."""
        ids = self.index.course_ids if course_ids is None else course_ids
        for cid in ids:
            for day, period in self.occupied(cid):
                yield cid, day, period

    # -- mutation ---------------------------------------------------------

    def place(self, course_id: str, room: str, slot: str) -> None:
        self.assignments[course_id] = Assignment(room=room, slot=slot)

    def remove(self, course_id: str) -> None:
        self.assignments.pop(course_id, None)

    # -- presentation -----------------------------------------------------

    def grid(self) -> dict[str, dict[int, list[str]]]:
        """day -> period -> courses, for the UI and for debugging."""
        out: dict[str, dict[int, list[str]]] = {
            day: {p: [] for p in periods}
            for day, periods in self.index.periods_by_day.items()
        }
        for cid, day, period in self.cells():
            out[day][period].append(cid)
        return out

    def render(self) -> str:
        """A fixed-width text grid. Rooms shown beside each course."""
        grid = self.grid()
        days = [d for d in self.index.department.days if d in grid]
        periods = self.index.department.periods
        width = 22

        lines = ["     " + "".join(f"{d:<{width}}" for d in days)]
        for p in periods:
            cells: list[str] = []
            for d in days:
                here = grid.get(d, {}).get(p, [])
                label = ", ".join(
                    f"{cid}@{self.room_of(cid)}" for cid in sorted(here)
                )
                cells.append(f"{label[:width - 1]:<{width}}")
            lines.append(f"  {p}  " + "".join(cells))
        return "\n".join(lines)

    def as_dict(self) -> dict[str, dict[str, str]]:
        return {
            cid: {"room": a.room, "slot": a.slot}
            for cid, a in sorted(self.assignments.items())
        }


__all__ = ["Assignment", "Timetable"]
