"""Problem-model entities (build plan Phase 1.3).

Entity identifiers are the only names the solver ever sees. Staff text is
resolved to these ids by `parser.resolve`; see Phase 4.4 and 4.6.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

RoomType = Literal["lecture", "lab", "seminar", "computer_lab"]
CourseType = Literal["lecture", "lab", "seminar", "tutorial"]
Day = Literal["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

DAYS: tuple[str, ...] = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


class Lecturer(BaseModel):
    """A member of teaching staff.

    `aliases` carries the forms staff actually write: surname only, title plus
    surname, a nickname. Phase 4.4 resolves against these before failing.
    """

    id: str
    name: str
    aliases: list[str] = Field(default_factory=list)


class Group(BaseModel):
    """A student group that must not be double-booked."""

    id: str
    name: str
    size: int = Field(gt=0)
    aliases: list[str] = Field(default_factory=list)


class Room(BaseModel):
    id: str
    capacity: int = Field(gt=0)
    type: RoomType
    name: str | None = None
    aliases: list[str] = Field(default_factory=list)


class Course(BaseModel):
    id: str
    lecturer: str
    group: str
    duration: int = Field(gt=0, description="Consecutive periods required.")
    type: CourseType
    size: int = Field(gt=0)
    name: str | None = None
    aliases: list[str] = Field(default_factory=list)


class Slot(BaseModel):
    id: str
    day: Day
    period: int = Field(ge=1)


class Department(BaseModel):
    """The full problem instance: entities plus the slot grid.

    Loaded from JSON by `data.load_department`. Held in memory and passed to
    both the prompt builder and the resolver, so the two never disagree about
    which entities exist.
    """

    name: str
    lecturers: list[Lecturer]
    groups: list[Group]
    rooms: list[Room]
    courses: list[Course]
    slots: list[Slot]

    @field_validator("lecturers", "groups", "rooms", "courses", "slots")
    @classmethod
    def _unique_ids(cls, v: list) -> list:
        ids = [item.id for item in v]
        duplicates = {i for i in ids if ids.count(i) > 1}
        if duplicates:
            raise ValueError(f"duplicate ids: {sorted(duplicates)}")
        return v

    def model_post_init(self, _context) -> None:
        lecturer_ids = {x.id for x in self.lecturers}
        group_ids = {x.id for x in self.groups}
        problems: list[str] = []
        for c in self.courses:
            if c.lecturer not in lecturer_ids:
                problems.append(f"course {c.id} references unknown lecturer {c.lecturer}")
            if c.group not in group_ids:
                problems.append(f"course {c.id} references unknown group {c.group}")
        if problems:
            raise ValueError("; ".join(problems))

    # -- lookups ---------------------------------------------------------

    @property
    def periods(self) -> list[int]:
        return sorted({s.period for s in self.slots})

    @property
    def days(self) -> list[str]:
        seen = {s.day for s in self.slots}
        return [d for d in DAYS if d in seen]

    def lecturer_ids(self) -> list[str]:
        return [x.id for x in self.lecturers]

    def course_ids(self) -> list[str]:
        return [x.id for x in self.courses]

    def room_ids(self) -> list[str]:
        return [x.id for x in self.rooms]

    def group_ids(self) -> list[str]:
        return [x.id for x in self.groups]

    def room_types(self) -> list[str]:
        return sorted({x.type for x in self.rooms})
