"""Closed constraint vocabulary (build plan Phase 2).

The type set is FROZEN after Week 3. `unsupported` and `needs_clarification`
stay open: every instance is logged and categorised, because that is where the
elicitation findings are (Phase 2, Phase 6A).

Entity-valued fields hold the string the staff member wrote. Resolution to
`Department` ids happens afterwards in `parser.resolve`, which keeps the LLM
contract simple and keeps resolution failures measurable on their own
(Phase 4.6).
"""

from __future__ import annotations

from typing import Annotated, Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Strength = Literal["hard", "soft"]
EntityKind = Literal["lecturer", "course", "room", "group", "room_type"]

#: Soft weights use a 1-10 scale; hard constraints are 1000+ (Phase 3.6).
SOFT_WEIGHT_MIN = 1
SOFT_WEIGHT_MAX = 10
HARD_PENALTY = 1000


class BaseConstraint(BaseModel):
    """Fields every constraint carries (Phase 2)."""

    model_config = ConfigDict(extra="forbid")

    strength: Strength
    source_text: str = Field(min_length=1, description="The staff sentence this came from.")
    weight: int | None = Field(
        default=None,
        description="Soft weight on a 1-10 scale. None for hard constraints.",
    )
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "LLM self-reported confidence. NOT requested from the model and NOT "
            "used for flagging under the default configuration: the plan's "
            "option (a) flags by rule instead, because self-reported confidence "
            "is uncalibrated. Kept so option (b), a calibration study, is a "
            "config change and not a schema change."
        ),
    )

    #: field name -> kind of entity it names, consumed by parser.resolve
    ENTITY_FIELDS: ClassVar[dict[str, EntityKind]] = {}

    @model_validator(mode="after")
    def _weight_matches_strength(self) -> "BaseConstraint":
        if self.strength == "hard" and self.weight is not None:
            raise ValueError(
                "hard constraints take no weight; the solver applies HARD_PENALTY"
            )
        if self.strength == "soft" and self.weight is not None:
            if not SOFT_WEIGHT_MIN <= self.weight <= SOFT_WEIGHT_MAX:
                raise ValueError(
                    f"soft weight must be {SOFT_WEIGHT_MIN}-{SOFT_WEIGHT_MAX}, got {self.weight}"
                )
        return self

    def entity_references(self) -> list[tuple[str, EntityKind, str]]:
        """Yield (field_path, kind, raw_value) for every entity named here."""
        out: list[tuple[str, EntityKind, str]] = []
        for field, kind in self.ENTITY_FIELDS.items():
            value = getattr(self, field, None)
            if value is None:
                continue
            if isinstance(value, list):
                for i, item in enumerate(value):
                    out.append((f"{field}[{i}]", kind, item))
            else:
                out.append((field, kind, value))
        return out


def _validate_days(v: list[str] | None) -> list[str] | None:
    if v is None:
        return None
    from .entities import DAYS

    bad = [d for d in v if d not in DAYS]
    if bad:
        raise ValueError(f"unknown day(s) {bad}; expected one of {list(DAYS)}")
    if not v:
        raise ValueError("days must be omitted (null) or non-empty")
    return v


def _validate_periods(v: list[int] | None) -> list[int] | None:
    if v is None:
        return None
    if not v:
        raise ValueError("periods must be omitted (null) or non-empty")
    if any(p < 1 for p in v):
        raise ValueError("periods are 1-indexed")
    return v


# -- the frozen vocabulary ------------------------------------------------


class LecturerUnavailable(BaseConstraint):
    """"I can't teach Fridays." Null days or periods means "all of them"."""

    type: Literal["lecturer_unavailable"] = "lecturer_unavailable"
    lecturer: str
    days: list[str] | None = None
    periods: list[int] | None = None

    ENTITY_FIELDS: ClassVar[dict[str, EntityKind]] = {"lecturer": "lecturer"}

    _days = field_validator("days")(_validate_days)
    _periods = field_validator("periods")(_validate_periods)

    @model_validator(mode="after")
    def _not_both_null(self) -> "LecturerUnavailable":
        if self.days is None and self.periods is None:
            raise ValueError(
                "lecturer_unavailable with both days and periods null would make the "
                "lecturer unavailable for every slot; state at least one"
            )
        return self


class LecturerPrefersTime(BaseConstraint):
    """"I'd rather teach mornings." The listed times are the PREFERRED ones."""

    type: Literal["lecturer_prefers_time"] = "lecturer_prefers_time"
    lecturer: str
    days: list[str] | None = None
    periods: list[int] | None = None

    ENTITY_FIELDS: ClassVar[dict[str, EntityKind]] = {"lecturer": "lecturer"}

    _days = field_validator("days")(_validate_days)
    _periods = field_validator("periods")(_validate_periods)

    @model_validator(mode="after")
    def _not_both_null(self) -> "LecturerPrefersTime":
        if self.days is None and self.periods is None:
            raise ValueError("state a preferred day or period")
        return self


class CourseRequiresRoom(BaseConstraint):
    """A named course pinned to one named room."""

    type: Literal["course_requires_room"] = "course_requires_room"
    course: str
    room: str

    ENTITY_FIELDS: ClassVar[dict[str, EntityKind]] = {"course": "course", "room": "room"}


class CourseRequiresRoomType(BaseConstraint):
    """"The practical needs a lab." """

    type: Literal["course_requires_room_type"] = "course_requires_room_type"
    course: str
    room_type: str

    ENTITY_FIELDS: ClassVar[dict[str, EntityKind]] = {
        "course": "course",
        "room_type": "room_type",
    }


class MaxConsecutiveHours(BaseConstraint):
    """Cap on back-to-back periods for a lecturer or a student group."""

    type: Literal["max_consecutive_hours"] = "max_consecutive_hours"
    max_hours: int = Field(ge=1)
    lecturer: str | None = None
    group: str | None = None

    ENTITY_FIELDS: ClassVar[dict[str, EntityKind]] = {
        "lecturer": "lecturer",
        "group": "group",
    }

    @model_validator(mode="after")
    def _exactly_one_subject(self) -> "MaxConsecutiveHours":
        if (self.lecturer is None) == (self.group is None):
            raise ValueError("set exactly one of lecturer or group")
        return self


class NoEarlyStart(BaseConstraint):
    """"Nothing before 10." Applies department-wide when both subjects are null."""

    type: Literal["no_early_start"] = "no_early_start"
    earliest_period: int = Field(ge=1)
    lecturer: str | None = None
    group: str | None = None

    ENTITY_FIELDS: ClassVar[dict[str, EntityKind]] = {
        "lecturer": "lecturer",
        "group": "group",
    }

    @model_validator(mode="after")
    def _at_most_one_subject(self) -> "NoEarlyStart":
        if self.lecturer is not None and self.group is not None:
            raise ValueError("set at most one of lecturer or group")
        return self


class CoursesNotSameTime(BaseConstraint):
    """Two or more courses that must not overlap."""

    type: Literal["courses_not_same_time"] = "courses_not_same_time"
    courses: list[str] = Field(min_length=2)

    ENTITY_FIELDS: ClassVar[dict[str, EntityKind]] = {"courses": "course"}

    @field_validator("courses")
    @classmethod
    def _distinct(cls, v: list[str]) -> list[str]:
        if len({x.strip().lower() for x in v}) != len(v):
            raise ValueError("courses must be distinct")
        return v


class CoursesSameDay(BaseConstraint):
    """Two or more courses to be scheduled on one day."""

    type: Literal["courses_same_day"] = "courses_same_day"
    courses: list[str] = Field(min_length=2)

    ENTITY_FIELDS: ClassVar[dict[str, EntityKind]] = {"courses": "course"}

    @field_validator("courses")
    @classmethod
    def _distinct(cls, v: list[str]) -> list[str]:
        if len({x.strip().lower() for x in v}) != len(v):
            raise ValueError("courses must be distinct")
        return v


# -- the two open categories ---------------------------------------------


class Unsupported(BaseConstraint):
    """No frozen type fits. Logged and categorised; never sent to the solver.

    Phase 2 keeps this category open deliberately: the distribution of what
    staff ask for and the schema cannot express is a finding, not a defect.
    """

    type: Literal["unsupported"] = "unsupported"
    note: str = Field(min_length=1, description="Why no type fits. Categorised later.")
    strength: Strength = "soft"


class NeedsClarification(BaseConstraint):
    """Ambiguous input. Carries the question to put to the user (Phase 4.3).

    `candidate_types` records what the parser was choosing between, so the
    ambiguity can be categorised without re-reading the source text.
    """

    type: Literal["needs_clarification"] = "needs_clarification"
    question: str = Field(min_length=1)
    candidate_types: list[str] = Field(default_factory=list)
    note: str | None = None
    strength: Strength = "soft"


Constraint = Annotated[
    LecturerUnavailable
    | LecturerPrefersTime
    | CourseRequiresRoom
    | CourseRequiresRoomType
    | MaxConsecutiveHours
    | NoEarlyStart
    | CoursesNotSameTime
    | CoursesSameDay
    | Unsupported
    | NeedsClarification,
    Field(discriminator="type"),
]

#: Types the solver can act on. The two open categories are excluded.
SOLVABLE_TYPES: tuple[str, ...] = (
    "lecturer_unavailable",
    "lecturer_prefers_time",
    "course_requires_room",
    "course_requires_room_type",
    "max_consecutive_hours",
    "no_early_start",
    "courses_not_same_time",
    "courses_same_day",
)

OPEN_TYPES: tuple[str, ...] = ("unsupported", "needs_clarification")

ALL_TYPES: tuple[str, ...] = SOLVABLE_TYPES + OPEN_TYPES


class ConstraintSet(BaseModel):
    """What the parser returns for one staff submission."""

    model_config = ConfigDict(extra="forbid")

    constraints: list[Constraint] = Field(default_factory=list)

    def by_type(self, type_name: str) -> list[Any]:
        return [c for c in self.constraints if c.type == type_name]

    @property
    def solvable(self) -> list[Any]:
        return [c for c in self.constraints if c.type in SOLVABLE_TYPES]

    @property
    def needs_attention(self) -> list[Any]:
        return [c for c in self.constraints if c.type in OPEN_TYPES]
