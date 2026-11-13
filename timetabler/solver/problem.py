"""Precomputed lookups over a `Department` (build plan Phase 3).

Built once and shared by the cost function and the annealer. Everything here
is derived from the department and never changes during a search, so it is
computed up front and read from hot loops.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

from ..schema.entities import Course, Department, Room, Slot


@dataclass(frozen=True)
class ProblemIndex:
    department: Department

    # -- entity maps ------------------------------------------------------

    @cached_property
    def courses(self) -> dict[str, Course]:
        return {c.id: c for c in self.department.courses}

    @cached_property
    def rooms(self) -> dict[str, Room]:
        return {r.id: r for r in self.department.rooms}

    @cached_property
    def slots(self) -> dict[str, Slot]:
        return {s.id: s for s in self.department.slots}

    @cached_property
    def course_ids(self) -> tuple[str, ...]:
        return tuple(c.id for c in self.department.courses)

    @cached_property
    def room_ids(self) -> tuple[str, ...]:
        return tuple(r.id for r in self.department.rooms)

    # -- slot grid --------------------------------------------------------

    @cached_property
    def periods_by_day(self) -> dict[str, tuple[int, ...]]:
        out: dict[str, list[int]] = {}
        for s in self.department.slots:
            out.setdefault(s.day, []).append(s.period)
        return {day: tuple(sorted(ps)) for day, ps in out.items()}

    @cached_property
    def slot_by_day_period(self) -> dict[tuple[str, int], str]:
        return {(s.day, s.period): s.id for s in self.department.slots}

    @cached_property
    def valid_starts(self) -> dict[int, tuple[str, ...]]:
        """Start slots by course duration, so a course always fits its day.

        A course of duration d may start at period p on a day only when
        p, p+1, ... p+d-1 all exist on that day. Generating moves from this
        removes the need for an overflow penalty: an assignment the search can
        reach is always within the grid.
        """
        durations = {c.duration for c in self.department.courses}
        out: dict[int, tuple[str, ...]] = {}
        for d in durations:
            ok: list[str] = []
            for slot in self.department.slots:
                periods = set(self.periods_by_day[slot.day])
                if all(slot.period + k in periods for k in range(d)):
                    ok.append(slot.id)
            out[d] = tuple(ok)
        return out

    def starts_for(self, course_id: str) -> tuple[str, ...]:
        return self.valid_starts[self.courses[course_id].duration]

    @cached_property
    def valid_start_pairs(self) -> frozenset[tuple[str, str]]:
        """Every (course, start slot) whose block fits its day.

        Membership is an O(1) replacement for re-deriving the block and
        checking each period against the day's grid.
        """
        return frozenset(
            (course_id, slot_id)
            for course_id in self.course_ids
            for slot_id in self.starts_for(course_id)
        )

    @cached_property
    def _occupied(self) -> dict[tuple[str, str], tuple[tuple[str, int], ...]]:
        """Precomputed blocks for every (course, slot) pair.

        `occupied` is the hottest call in the search: a profile of 8,000
        annealing iterations showed 430,000 calls rebuilding the same tuples.
        The table is |courses| x |slots| entries, so it is cheaper to hold
        than to recompute.
        """
        out: dict[tuple[str, str], tuple[tuple[str, int], ...]] = {}
        for course in self.department.courses:
            for slot in self.department.slots:
                out[(course.id, slot.id)] = tuple(
                    (slot.day, slot.period + k) for k in range(course.duration)
                )
        return out

    def occupied(self, course_id: str, start_slot_id: str) -> tuple[tuple[str, int], ...]:
        """The (day, period) cells a course covers from a start slot."""
        return self._occupied[(course_id, start_slot_id)]

    def fits(self, course_id: str, start_slot_id: str) -> bool:
        return (course_id, start_slot_id) in self.valid_start_pairs

    # -- groupings used by constraint scopes ------------------------------

    @cached_property
    def courses_by_lecturer(self) -> dict[str, tuple[str, ...]]:
        out: dict[str, list[str]] = {}
        for c in self.department.courses:
            out.setdefault(c.lecturer, []).append(c.id)
        return {k: tuple(v) for k, v in out.items()}

    @cached_property
    def courses_by_group(self) -> dict[str, tuple[str, ...]]:
        out: dict[str, list[str]] = {}
        for c in self.department.courses:
            out.setdefault(c.group, []).append(c.id)
        return {k: tuple(v) for k, v in out.items()}

    @cached_property
    def rooms_by_type(self) -> dict[str, tuple[str, ...]]:
        out: dict[str, list[str]] = {}
        for r in self.department.rooms:
            out.setdefault(r.type, []).append(r.id)
        return {k: tuple(v) for k, v in out.items()}

    def rooms_fitting(self, course_id: str) -> tuple[str, ...]:
        """Rooms large enough for a course, by capacity alone."""
        size = self.courses[course_id].size
        return tuple(r.id for r in self.department.rooms if r.capacity >= size)


__all__ = ["ProblemIndex"]
