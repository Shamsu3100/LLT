"""Violation counting (build plan Phase 3.2, 3.4).

Two families, kept apart because they come from different places:

* **Structural constraints** (Phase 1.4) hold for every instance and are never
  elicited: no lecturer, room or group may be double-booked, and a room must
  hold its course. They are always hard.
* **Elicited constraints** arrive from staff through the parser. Each frozen
  type has one function `(timetable, constraint, index) -> int`, registered in
  `VIOLATION_FUNCTIONS`, so adding a type later means adding a function.

Entity fields must already hold canonical ids: run `materialise` on parser
output first.

Each function returns a **count**, not a boolean, so that a timetable
breaching one constraint twice is scored worse than one breaching it once.
Where a count is a matter of convention, the docstring says which was chosen.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Iterable

from ..schema.constraints import OPEN_TYPES
from .problem import ProblemIndex
from .timetable import Timetable

ViolationFn = Callable[[Timetable, Any, ProblemIndex], int]


# -- structural -----------------------------------------------------------


STRUCTURAL_NAMES: tuple[str, ...] = (
    "lecturer_clash",
    "room_clash",
    "group_clash",
    "room_capacity",
    "block_fits_day",
    "assigned",
)


def structural_counts(
    tt: Timetable, index: ProblemIndex, courses: Iterable[str] | None = None
) -> dict[str, int]:
    """All structural violations in one pass over the occupied cells.

    Computed together because they read the same data: a profile showed three
    separate clash passes costing more than everything else in the search put
    together. The individual functions below read from this one, so they
    cannot drift from what the cost model uses.

    A clash is counted as occupants minus one per (owner, day, period), so a
    cell with three occupants counts two.
    """
    ids = index.course_ids if courses is None else tuple(courses)
    # Plain dicts, not Counters: a Counter resolves every new key through
    # __missing__, which a profile showed firing half a million times per
    # 8,000 iterations. Overlaps accumulate as they are found, so each
    # occupant after the first adds one, which is the same total as
    # sum(n - 1 for n > 1) without a second pass.
    lecturer_cells: dict[tuple[str, tuple[str, int]], int] = {}
    room_cells: dict[tuple[str, tuple[str, int]], int] = {}
    group_cells: dict[tuple[str, tuple[str, int]], int] = {}
    lecturer_clash = room_clash = group_clash = 0
    capacity = fits = unassigned = 0

    course_map = index.courses
    room_map = index.rooms
    valid_pairs = index.valid_start_pairs
    occupied = index._occupied
    assignments = tt.assignments

    for course_id in ids:
        assignment = assignments.get(course_id)
        if assignment is None:
            unassigned += 1
            continue
        course = course_map[course_id]
        room = assignment.room
        if room_map[room].capacity < course.size:
            capacity += 1
        if (course_id, assignment.slot) not in valid_pairs:
            fits += 1

        lecturer = course.lecturer
        group = course.group
        for cell in occupied[(course_id, assignment.slot)]:
            key = (lecturer, cell)
            n = lecturer_cells.get(key)
            if n is None:
                lecturer_cells[key] = 1
            else:
                lecturer_cells[key] = n + 1
                lecturer_clash += 1

            key = (room, cell)
            n = room_cells.get(key)
            if n is None:
                room_cells[key] = 1
            else:
                room_cells[key] = n + 1
                room_clash += 1

            key = (group, cell)
            n = group_cells.get(key)
            if n is None:
                group_cells[key] = 1
            else:
                group_cells[key] = n + 1
                group_clash += 1

    return {
        "lecturer_clash": lecturer_clash,
        "room_clash": room_clash,
        "group_clash": group_clash,
        "room_capacity": capacity,
        "block_fits_day": fits,
        "assigned": unassigned,
    }


def structural_total(
    tt: Timetable, index: ProblemIndex, courses: Iterable[str] | None = None
) -> int:
    return sum(structural_counts(tt, index, courses).values())


def _structural(name: str) -> Callable[..., int]:
    def fn(
        tt: Timetable, index: ProblemIndex, courses: Iterable[str] | None = None
    ) -> int:
        return structural_counts(tt, index, courses)[name]

    fn.__name__ = name
    return fn


#: One lecturer, room or group may hold only one course per cell; a room must
#: hold its course; a block must fit its day; every course must be placed.
lecturer_clashes = _structural("lecturer_clash")
room_clashes = _structural("room_clash")
group_clashes = _structural("group_clash")
capacity_violations = _structural("room_capacity")
fit_violations = _structural("block_fits_day")
unassigned_count = _structural("assigned")

STRUCTURAL_FUNCTIONS: dict[str, Callable[..., int]] = {
    "lecturer_clash": lecturer_clashes,
    "room_clash": room_clashes,
    "group_clash": group_clashes,
    "room_capacity": capacity_violations,
    "block_fits_day": fit_violations,
    "assigned": unassigned_count,
}


# -- elicited: one function per frozen type ------------------------------


def _matches_window(
    day: str, period: int, days: list[str] | None, periods: list[int] | None
) -> bool:
    """Whether a cell falls inside a (days, periods) window. Null means all."""
    return (days is None or day in days) and (periods is None or period in periods)


def v_lecturer_unavailable(tt: Timetable, c: Any, index: ProblemIndex) -> int:
    """One per occupied cell inside the stated unavailable window."""
    total = 0
    for course_id in index.courses_by_lecturer.get(c.lecturer, ()):
        for day, period in tt.occupied(course_id):
            if _matches_window(day, period, c.days, c.periods):
                total += 1
    return total


def v_lecturer_prefers_time(tt: Timetable, c: Any, index: ProblemIndex) -> int:
    """One per occupied cell OUTSIDE the preferred window.

    The listed times are the preferred ones, so a violation is a cell that
    falls outside them.
    """
    total = 0
    for course_id in index.courses_by_lecturer.get(c.lecturer, ()):
        for day, period in tt.occupied(course_id):
            if not _matches_window(day, period, c.days, c.periods):
                total += 1
    return total


def v_course_requires_room(tt: Timetable, c: Any, index: ProblemIndex) -> int:
    """One if the course sits in any other room."""
    room = tt.room_of(c.course)
    return 0 if room is None or room == c.room else 1


def v_course_requires_room_type(tt: Timetable, c: Any, index: ProblemIndex) -> int:
    room = tt.room_of(c.course)
    if room is None:
        return 0
    return 0 if index.rooms[room].type == c.room_type else 1


def _runs(periods: list[int]) -> list[int]:
    """Lengths of maximal consecutive runs in a sorted period list."""
    if not periods:
        return []
    lengths: list[int] = []
    run = 1
    for prev, cur in zip(periods, periods[1:]):
        if cur == prev + 1:
            run += 1
        elif cur == prev:
            continue  # a clash elsewhere; do not inflate the run
        else:
            lengths.append(run)
            run = 1
    lengths.append(run)
    return lengths


def v_max_consecutive_hours(tt: Timetable, c: Any, index: ProblemIndex) -> int:
    """Excess periods per over-long run, summed over days.

    A run of 6 against a cap of 3 counts 3, so the search is pushed toward
    shorter runs instead of treating every breach alike.
    """
    if c.lecturer is not None:
        courses = index.courses_by_lecturer.get(c.lecturer, ())
    else:
        courses = index.courses_by_group.get(c.group, ())
    total = 0
    for day in index.periods_by_day:
        for length in _runs(tt.periods_on_day(courses, day)):
            if length > c.max_hours:
                total += length - c.max_hours
    return total


def _scope_courses(c: Any, index: ProblemIndex) -> tuple[str, ...]:
    if getattr(c, "lecturer", None) is not None:
        return index.courses_by_lecturer.get(c.lecturer, ())
    if getattr(c, "group", None) is not None:
        return index.courses_by_group.get(c.group, ())
    return index.course_ids


def v_no_early_start(tt: Timetable, c: Any, index: ProblemIndex) -> int:
    """One per occupied cell before the earliest acceptable period."""
    total = 0
    for course_id in _scope_courses(c, index):
        for _, period in tt.occupied(course_id):
            if period < c.earliest_period:
                total += 1
    return total


def v_courses_not_same_time(tt: Timetable, c: Any, index: ProblemIndex) -> int:
    """One per pair of the named courses that share any cell.

    Counted per pair, not per shared cell: the constraint is about the pair,
    and a long double-booked block is one fault to fix, not several.
    """
    occupied = {cid: set(tt.occupied(cid)) for cid in c.courses}
    ids = [cid for cid in c.courses if occupied.get(cid)]
    total = 0
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            if occupied[a] & occupied[b]:
                total += 1
    return total


def v_courses_same_day(tt: Timetable, c: Any, index: ProblemIndex) -> int:
    """Courses away from the most-used day among the named set.

    The target day is whichever already holds the most of them, so a set
    already together scores 0 wherever it sits, and the count is the number of
    courses that would have to move.
    """
    days = [tt.day_of(cid) for cid in c.courses]
    present = [d for d in days if d is not None]
    if len(present) <= 1:
        return 0
    counts = Counter(present)
    best = max(counts.values())
    return len(present) - best


VIOLATION_FUNCTIONS: dict[str, ViolationFn] = {
    "lecturer_unavailable": v_lecturer_unavailable,
    "lecturer_prefers_time": v_lecturer_prefers_time,
    "course_requires_room": v_course_requires_room,
    "course_requires_room_type": v_course_requires_room_type,
    "max_consecutive_hours": v_max_consecutive_hours,
    "no_early_start": v_no_early_start,
    "courses_not_same_time": v_courses_not_same_time,
    "courses_same_day": v_courses_same_day,
}


def violations(tt: Timetable, constraint: Any, index: ProblemIndex) -> int:
    """Violation count for one elicited constraint."""
    fn = VIOLATION_FUNCTIONS.get(constraint.type)
    if fn is None:
        if constraint.type in OPEN_TYPES:
            raise ValueError(
                f"{constraint.type} is not solvable; it is for review, not search"
            )
        raise ValueError(f"no violation function registered for {constraint.type!r}")
    return fn(tt, constraint, index)


def constraint_scope(constraint: Any, index: ProblemIndex) -> frozenset[str]:
    """Courses whose placement can change this constraint's count.

    The annealer re-evaluates only the constraints whose scope meets a move,
    so this must never under-report. A type absent from the map falls back to
    every course, which is slow and correct; the alternative would be a silent
    wrong delta.
    """
    t = constraint.type
    if t in ("lecturer_unavailable", "lecturer_prefers_time"):
        return frozenset(index.courses_by_lecturer.get(constraint.lecturer, ()))
    if t in ("course_requires_room", "course_requires_room_type"):
        return frozenset({constraint.course})
    if t in ("courses_not_same_time", "courses_same_day"):
        return frozenset(constraint.courses)
    if t in ("max_consecutive_hours", "no_early_start"):
        return frozenset(_scope_courses(constraint, index))
    return frozenset(index.course_ids)


__all__ = [
    "STRUCTURAL_FUNCTIONS",
    "STRUCTURAL_NAMES",
    "VIOLATION_FUNCTIONS",
    "ViolationFn",
    "capacity_violations",
    "constraint_scope",
    "fit_violations",
    "group_clashes",
    "lecturer_clashes",
    "room_clashes",
    "structural_counts",
    "structural_total",
    "unassigned_count",
    "violations",
]
