"""Constraints as readable English (build plan Phase 5).

The verification step asks staff to confirm a reading of their own sentence,
so the reading has to be legible: "Dr. Sarah Lee cannot teach on Friday:
HARD", never the JSON. A participant who cannot read the restatement is not
verifying anything, and the residual-error measure in Phase 6C would be
measuring the interface instead of the parser.

Pure functions over a constraint and a `Department`, so the wording is
testable without running the app.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from ..schema.entities import DAYS, Department

DAY_NAMES: dict[str, str] = {
    "Mon": "Monday",
    "Tue": "Tuesday",
    "Wed": "Wednesday",
    "Thu": "Thursday",
    "Fri": "Friday",
    "Sat": "Saturday",
    "Sun": "Sunday",
}

ROOM_TYPE_NAMES: dict[str, str] = {
    "lecture": "a lecture theatre",
    "lab": "a laboratory",
    "computer_lab": "a computer lab",
    "seminar": "a seminar room",
}

STRENGTH_LABELS: dict[str, str] = {"hard": "HARD", "soft": "SOFT"}


def join(parts: Sequence[str], conjunction: str = "and") -> str:
    """"a", "a and b", "a, b and c"."""
    items = [p for p in parts if p]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} {conjunction} {items[1]}"
    return f"{', '.join(items[:-1])} {conjunction} {items[-1]}"


def day_phrase(days: Iterable[str] | None) -> str:
    """"on Monday and Friday", or "" when every day is meant."""
    if days is None:
        return ""
    ordered = [d for d in DAYS if d in set(days)]
    names = [DAY_NAMES.get(d, d) for d in ordered]
    return f"on {join(names)}" if names else ""


def period_phrase(periods: Sequence[int] | None) -> str:
    """"during period 1", "during periods 1 to 3", "during periods 1, 3 and 5".

    A contiguous run is written as a range, which is how staff describe a
    block. Scattered periods are listed.
    """
    if not periods:
        return ""
    ordered = sorted(set(periods))
    if len(ordered) == 1:
        return f"during period {ordered[0]}"
    contiguous = ordered == list(range(ordered[0], ordered[-1] + 1))
    if contiguous:
        return f"during periods {ordered[0]} to {ordered[-1]}"
    return f"during periods {join([str(p) for p in ordered])}"


class Namer:
    """Turns ids or staff wording into display labels.

    Handles both directions, because the verification UI shows a reading
    before resolution (staff wording) and the result page shows one after it
    (canonical ids). An unknown value is returned unchanged, so a hallucinated
    name still appears and can be challenged.
    """

    def __init__(self, department: Department | None = None) -> None:
        self.department = department
        self._labels: dict[str, str] = {}
        if department is None:
            return
        for lecturer in department.lecturers:
            self._labels[lecturer.id] = lecturer.name
        for group in department.groups:
            self._labels[group.id] = group.name
        for room in department.rooms:
            self._labels[room.id] = room.name or room.id
        for course in department.courses:
            self._labels[course.id] = course.name or course.id

    def __call__(self, value: str | None) -> str:
        if value is None:
            return ""
        return self._labels.get(value, value)

    def all(self, values: Iterable[str]) -> list[str]:
        return [self(v) for v in values]

    def room_type(self, value: str) -> str:
        return ROOM_TYPE_NAMES.get(value, f"a {value.replace('_', ' ')}")


# -- per-type wording ----------------------------------------------------


def _subject_of(c: Any, name: Namer) -> tuple[str, bool]:
    """The thing a constraint is about, and whether it is a person."""
    if getattr(c, "lecturer", None) is not None:
        return name(c.lecturer), True
    if getattr(c, "group", None) is not None:
        return name(c.group), False
    return "", False


def _lecturer_unavailable(c: Any, name: Namer) -> str:
    when = " ".join(p for p in (day_phrase(c.days), period_phrase(c.periods)) if p)
    if c.days is None and c.periods:
        when = f"{period_phrase(c.periods)} on any day"
    verb = "cannot teach" if c.strength == "hard" else "prefers not to teach"
    return f"{name(c.lecturer)} {verb} {when}".strip()


def _lecturer_prefers_time(c: Any, name: Namer) -> str:
    when = " ".join(p for p in (day_phrase(c.days), period_phrase(c.periods)) if p)
    return f"{name(c.lecturer)} prefers to teach {when}".strip()


def _course_requires_room(c: Any, name: Namer) -> str:
    modal = "must be" if c.strength == "hard" else "should be"
    return f"{name(c.course)} {modal} scheduled in {name(c.room)}"


def _course_requires_room_type(c: Any, name: Namer) -> str:
    modal = "must be" if c.strength == "hard" else "should be"
    return f"{name(c.course)} {modal} scheduled in {name.room_type(c.room_type)}"


def _max_consecutive_hours(c: Any, name: Namer) -> str:
    subject, is_person = _subject_of(c, name)
    modal = "must not" if c.strength == "hard" else "should not"
    activity = "teach" if is_person else "be taught for"
    periods = "period" if c.max_hours == 1 else "periods"
    if is_person:
        return (
            f"{subject} {modal} teach more than {c.max_hours} consecutive {periods}"
        )
    return (
        f"{subject} {modal} have more than {c.max_hours} consecutive {periods}"
    )


def _no_early_start(c: Any, name: Namer) -> str:
    subject, _ = _subject_of(c, name)
    if not subject:
        # "No class" already carries the negation, so the modal must not
        # repeat it: "No class must not start before 2" says the opposite.
        modal = "may" if c.strength == "hard" else "should"
        return f"No class {modal} start before period {c.earliest_period}"
    modal = "must not" if c.strength == "hard" else "should not"
    return f"{subject} {modal} start before period {c.earliest_period}"


def _courses_not_same_time(c: Any, name: Namer) -> str:
    modal = "must not" if c.strength == "hard" else "should not"
    return (
        f"{join(name.all(c.courses))} {modal} be scheduled at the same time"
    )


def _courses_same_day(c: Any, name: Namer) -> str:
    modal = "must" if c.strength == "hard" else "should"
    return f"{join(name.all(c.courses))} {modal} be scheduled on the same day"


def _unsupported(c: Any, name: Namer) -> str:
    return f"Not expressible in the current constraint types. {c.note}"


def _needs_clarification(c: Any, name: Namer) -> str:
    return f"Needs clarification. {c.question}"


WORDING: dict[str, Any] = {
    "lecturer_unavailable": _lecturer_unavailable,
    "lecturer_prefers_time": _lecturer_prefers_time,
    "course_requires_room": _course_requires_room,
    "course_requires_room_type": _course_requires_room_type,
    "max_consecutive_hours": _max_consecutive_hours,
    "no_early_start": _no_early_start,
    "courses_not_same_time": _courses_not_same_time,
    "courses_same_day": _courses_same_day,
    "unsupported": _unsupported,
    "needs_clarification": _needs_clarification,
}


def verbalise(
    constraint: Any,
    department: Department | None = None,
    *,
    include_strength: bool = True,
) -> str:
    """One constraint as a sentence, with its strength appended.

    The strength tag stays even though the modal already carries it: the tag
    is what a participant checks when deciding whether to use the hard/soft
    toggle, and the plan's worked example shows it.
    """
    namer = department if isinstance(department, Namer) else Namer(department)
    writer = WORDING.get(constraint.type)
    if writer is None:
        raise ValueError(f"no wording registered for {constraint.type!r}")
    sentence = " ".join(writer(constraint, namer).split())

    if not include_strength or constraint.type in ("unsupported", "needs_clarification"):
        return sentence

    label = STRENGTH_LABELS.get(constraint.strength, constraint.strength.upper())
    weight = getattr(constraint, "weight", None)
    if constraint.strength == "soft" and weight is not None:
        return f"{sentence}: {label} (weight {weight})"
    return f"{sentence}: {label}"


def verbalise_all(
    constraints: Iterable[Any],
    department: Department | None = None,
    **kwargs: Any,
) -> list[str]:
    namer = Namer(department)
    return [verbalise(c, namer, **kwargs) for c in constraints]


__all__ = [
    "DAY_NAMES",
    "ROOM_TYPE_NAMES",
    "STRENGTH_LABELS",
    "Namer",
    "day_phrase",
    "join",
    "period_phrase",
    "verbalise",
    "verbalise_all",
]
