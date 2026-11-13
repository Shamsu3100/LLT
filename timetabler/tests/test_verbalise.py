"""Readable sentences for the verification step (build plan Phase 5)."""

from __future__ import annotations

import pytest

from timetabler.app.verbalise import (
    Namer,
    day_phrase,
    join,
    period_phrase,
    verbalise,
    verbalise_all,
)
from timetabler.data import load_department
from timetabler.eval.dataset import load_items
from timetabler.schema import (
    ALL_TYPES,
    CourseRequiresRoom,
    CourseRequiresRoomType,
    CoursesNotSameTime,
    CoursesSameDay,
    LecturerPrefersTime,
    LecturerUnavailable,
    MaxConsecutiveHours,
    NeedsClarification,
    NoEarlyStart,
    Unsupported,
)


@pytest.fixture(scope="module")
def department():
    return load_department()


# -- helpers -------------------------------------------------------------


@pytest.mark.parametrize("parts,expected", [
    ([], ""),
    (["a"], "a"),
    (["a", "b"], "a and b"),
    (["a", "b", "c"], "a, b and c"),
])
def test_join(parts, expected):
    assert join(parts) == expected


def test_day_phrase_uses_full_names_in_week_order():
    assert day_phrase(["Fri", "Mon"]) == "on Monday and Friday"


def test_day_phrase_empty_when_every_day_is_meant():
    assert day_phrase(None) == ""


@pytest.mark.parametrize("periods,expected", [
    ([1], "during period 1"),
    ([1, 2, 3], "during periods 1 to 3"),
    ([3, 2, 1], "during periods 1 to 3"),
    ([1, 3, 5], "during periods 1, 3 and 5"),
    (None, ""),
])
def test_period_phrase(periods, expected):
    assert period_phrase(periods) == expected


def test_contiguous_run_reads_as_a_range():
    assert period_phrase([4, 5, 6]) == "during periods 4 to 6"


# -- naming --------------------------------------------------------------


def test_namer_resolves_ids_to_display_names(department):
    namer = Namer(department)
    assert namer("L001") == "Dr. Sarah Lee"
    assert namer("CS301") == "Machine Learning"
    assert namer("R101") == "Lecture Theatre 1"
    assert namer("G1A") == "Year 1 Group A"


def test_namer_passes_unknown_values_through(department):
    """A hallucinated name must still appear so it can be challenged."""
    assert Namer(department)("Prof Zhang") == "Prof Zhang"


def test_namer_works_without_a_department():
    assert Namer()("L001") == "L001"


# -- wording per type ----------------------------------------------------


def test_the_plan_worked_example(department):
    """Phase 5: "Dr. Lee cannot teach on Friday: HARD"."""
    c = LecturerUnavailable(
        lecturer="L001", days=["Fri"], strength="hard",
        source_text="I can't teach Fridays",
    )
    assert verbalise(c, department) == "Dr. Sarah Lee cannot teach on Friday: HARD"


def test_soft_unavailable_uses_a_softer_verb(department):
    c = LecturerUnavailable(
        lecturer="L001", days=["Wed"], periods=[4, 5, 6], strength="soft",
        weight=4, source_text="x",
    )
    assert verbalise(c, department) == (
        "Dr. Sarah Lee prefers not to teach on Wednesday during periods 4 to 6: "
        "SOFT (weight 4)"
    )


def test_periods_without_days_say_any_day(department):
    c = LecturerUnavailable(
        lecturer="L002", periods=[1], strength="hard", source_text="x"
    )
    assert "during period 1 on any day" in verbalise(c, department)


def test_prefers_time(department):
    c = LecturerPrefersTime(
        lecturer="L003", periods=[1, 2, 3], strength="soft", weight=5,
        source_text="x",
    )
    assert verbalise(c, department) == (
        "Dr. Mei Tan prefers to teach during periods 1 to 3: SOFT (weight 5)"
    )


def test_requires_room(department):
    c = CourseRequiresRoom(
        course="CS301", room="R101", strength="hard", source_text="x"
    )
    assert verbalise(c, department) == (
        "Machine Learning must be scheduled in Lecture Theatre 1: HARD"
    )


def test_requires_room_type_gets_an_article(department):
    c = CourseRequiresRoomType(
        course="CS202L", room_type="computer_lab", strength="hard", source_text="x"
    )
    assert "in a computer lab" in verbalise(c, department)


def test_max_consecutive_for_a_lecturer_says_teach(department):
    c = MaxConsecutiveHours(
        max_hours=3, lecturer="L004", strength="hard", source_text="x"
    )
    assert verbalise(c, department) == (
        "Dr. James Okafor must not teach more than 3 consecutive periods: HARD"
    )


def test_max_consecutive_for_a_group_says_have(department):
    c = MaxConsecutiveHours(
        max_hours=4, group="G2A", strength="soft", weight=4, source_text="x"
    )
    assert "Year 2 Group A should not have more than 4 consecutive periods" in (
        verbalise(c, department)
    )


def test_singular_period_when_the_cap_is_one(department):
    c = MaxConsecutiveHours(
        max_hours=1, lecturer="L004", strength="hard", source_text="x"
    )
    assert "1 consecutive period:" in verbalise(c, department)


def test_no_early_start_for_a_group(department):
    c = NoEarlyStart(
        earliest_period=2, group="G1A", strength="soft", weight=5, source_text="x"
    )
    assert "Year 1 Group A should not start before period 2" in verbalise(c, department)


def test_department_wide_no_early_start_avoids_a_double_negative(department):
    """Regression: "No class must not start before 2" says the opposite."""
    hard = NoEarlyStart(earliest_period=2, strength="hard", source_text="x")
    soft = NoEarlyStart(
        earliest_period=2, strength="soft", weight=5, source_text="x"
    )
    assert verbalise(hard, department) == "No class may start before period 2: HARD"
    assert "No class should start before period 2" in verbalise(soft, department)
    for c in (hard, soft):
        assert "not start" not in verbalise(c, department)


def test_not_same_time(department):
    c = CoursesNotSameTime(
        courses=["CS201", "CS202"], strength="hard", source_text="x"
    )
    assert verbalise(c, department) == (
        "Data Structures and Computer Architecture must not be scheduled at the "
        "same time: HARD"
    )


def test_same_day_lists_three_courses(department):
    c = CoursesSameDay(
        courses=["CS101", "CS102", "CS201"], strength="soft", weight=3,
        source_text="x",
    )
    assert (
        "Introduction to Programming, Discrete Mathematics and Data Structures"
        in verbalise(c, department)
    )


def test_open_categories_carry_no_strength_tag(department):
    """Neither is a constraint, so HARD or SOFT would mislead."""
    unsupported = Unsupported(note="Caps teaching days.", source_text="x")
    clarify = NeedsClarification(question="Which room did you mean?", source_text="x")
    for c in (unsupported, clarify):
        rendered = verbalise(c, department)
        assert "HARD" not in rendered and "SOFT" not in rendered
    assert "Caps teaching days." in verbalise(unsupported, department)
    assert "Which room did you mean?" in verbalise(clarify, department)


# -- invariants ----------------------------------------------------------


def test_every_frozen_type_has_wording():
    from timetabler.app.verbalise import WORDING

    assert set(WORDING) == set(ALL_TYPES)


def test_the_whole_seed_set_verbalises(department):
    """No gold constraint may render as JSON, blank, or a leftover id."""
    items = load_items()
    rendered = verbalise_all(
        [c for item in items for c in item.gold], department
    )
    assert len(rendered) == 25
    for line in rendered:
        assert line and not line.startswith("{")
        assert "  " not in line, f"double space in: {line}"
        assert line[0].isupper(), f"does not start with a capital: {line}"


def test_unresolved_entities_render_as_the_staff_wording(department):
    """Verification happens before resolution, so staff words must show."""
    c = LecturerUnavailable(
        lecturer="the big boss", days=["Fri"], strength="hard", source_text="x"
    )
    assert "the big boss cannot teach on Friday" in verbalise(c, department)


def test_strength_can_be_suppressed(department):
    c = LecturerUnavailable(
        lecturer="L001", days=["Fri"], strength="hard", source_text="x"
    )
    assert verbalise(c, department, include_strength=False) == (
        "Dr. Sarah Lee cannot teach on Friday"
    )


def test_unregistered_type_is_an_error(department):
    class Fake:
        type = "invented"
        strength = "hard"

    with pytest.raises(ValueError, match="no wording registered"):
        verbalise(Fake(), department)
