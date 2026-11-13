"""The frozen vocabulary enforces its own rules (build plan Phase 2)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from timetabler.schema import (
    ALL_TYPES,
    OPEN_TYPES,
    SOLVABLE_TYPES,
    ConstraintSet,
    LecturerUnavailable,
    MaxConsecutiveHours,
    NoEarlyStart,
)


def test_vocabulary_is_closed_and_complete():
    assert len(SOLVABLE_TYPES) == 8
    assert len(OPEN_TYPES) == 2
    assert len(ALL_TYPES) == 10
    assert not set(SOLVABLE_TYPES) & set(OPEN_TYPES)


def test_hard_constraint_takes_no_weight():
    with pytest.raises(ValidationError, match="hard constraints take no weight"):
        LecturerUnavailable(
            lecturer="L001", days=["Fri"], strength="hard", weight=5,
            source_text="x",
        )


@pytest.mark.parametrize("weight", [0, 11, 1000])
def test_soft_weight_stays_on_the_documented_scale(weight):
    with pytest.raises(ValidationError, match="soft weight must be 1-10"):
        LecturerUnavailable(
            lecturer="L001", days=["Fri"], strength="soft", weight=weight,
            source_text="x",
        )


def test_soft_weight_may_be_omitted():
    c = LecturerUnavailable(
        lecturer="L001", days=["Fri"], strength="soft", source_text="x"
    )
    assert c.weight is None


def test_unavailable_needs_a_day_or_a_period():
    with pytest.raises(ValidationError, match="unavailable for every slot"):
        LecturerUnavailable(lecturer="L001", strength="hard", source_text="x")


def test_unknown_day_rejected():
    with pytest.raises(ValidationError, match="unknown day"):
        LecturerUnavailable(
            lecturer="L001", days=["Friday"], strength="hard", source_text="x"
        )


def test_max_consecutive_needs_exactly_one_subject():
    with pytest.raises(ValidationError, match="exactly one of lecturer or group"):
        MaxConsecutiveHours(max_hours=3, strength="hard", source_text="x")
    with pytest.raises(ValidationError, match="exactly one of lecturer or group"):
        MaxConsecutiveHours(
            max_hours=3, lecturer="L001", group="G1A", strength="hard",
            source_text="x",
        )


def test_no_early_start_allows_department_wide():
    c = NoEarlyStart(earliest_period=2, strength="hard", source_text="x")
    assert c.lecturer is None and c.group is None


def test_extra_fields_rejected():
    with pytest.raises(ValidationError):
        LecturerUnavailable(
            lecturer="L001", days=["Fri"], strength="hard", source_text="x",
            invented_field=1,
        )


def test_discriminated_union_routes_by_type():
    payload = {"constraints": [
        {"type": "no_early_start", "earliest_period": 3, "strength": "hard",
         "source_text": "no early starts"},
        {"type": "unsupported", "note": "cap on teaching days",
         "strength": "soft", "source_text": "two days a week"},
    ]}
    cs = ConstraintSet.model_validate(payload)
    assert [c.type for c in cs.constraints] == ["no_early_start", "unsupported"]
    assert len(cs.solvable) == 1
    assert len(cs.needs_attention) == 1


def test_entity_references_reports_paths_and_kinds():
    c = LecturerUnavailable(
        lecturer="Dr Lee", days=["Fri"], strength="hard", source_text="x"
    )
    assert c.entity_references() == [("lecturer", "lecturer", "Dr Lee")]


def test_list_entity_references_are_indexed():
    from timetabler.schema import CoursesNotSameTime

    c = CoursesNotSameTime(
        courses=["CS301", "CS302"], strength="hard", source_text="x"
    )
    paths = [p for p, _, _ in c.entity_references()]
    assert paths == ["courses[0]", "courses[1]"]


def test_duplicate_courses_rejected():
    from timetabler.schema import CoursesSameDay

    with pytest.raises(ValidationError, match="distinct"):
        CoursesSameDay(
            courses=["CS301", "cs301"], strength="soft", source_text="x"
        )
