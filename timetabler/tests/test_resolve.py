"""Entity resolution and rule-based flags (build plan Phase 4.3, 4.4, 4.6)."""

from __future__ import annotations

import pytest

from timetabler.data import load_department
from timetabler.parser.resolve import (
    EntityResolver,
    Outcome,
    Severity,
    is_descriptive,
    normalise,
)
from timetabler.schema import (
    CourseRequiresRoom,
    CourseRequiresRoomType,
    LecturerUnavailable,
    MaxConsecutiveHours,
    NoEarlyStart,
    Unsupported,
)


@pytest.fixture(scope="module")
def resolver():
    return EntityResolver(load_department())


def codes(resolved):
    return {f.code for f in resolved.flags}


# -- normalisation --------------------------------------------------------


@pytest.mark.parametrize("raw,expected", [
    ("Dr. Lee", "lee"),
    ("  PROF   Rahman ", "rahman"),
    ("Ms. Priya Nair", "priya nair"),
    ("Mei Tán", "mei tan"),
])
def test_normalise_strips_titles_accents_and_spacing(raw, expected):
    assert normalise(raw) == expected


@pytest.mark.parametrize("raw", ["the big lab", "my Tuesday group", "All rooms"])
def test_descriptive_detected(raw):
    assert is_descriptive(raw)


@pytest.mark.parametrize("raw", ["Lee", "CS301", "Lecture Theatre 1"])
def test_named_references_are_not_descriptive(raw):
    assert not is_descriptive(raw)


# -- resolution paths ----------------------------------------------------


@pytest.mark.parametrize("raw,kind,expected_id,expected_outcome", [
    ("L001", "lecturer", "L001", Outcome.ID),
    ("Dr. Sarah Lee", "lecturer", "L001", Outcome.NAME),
    ("Lee", "lecturer", "L001", Outcome.ALIAS),
    ("CS301", "course", "CS301", Outcome.ID),
    ("Machine Learning", "course", "CS301", Outcome.NAME),
    ("the big lab", "room", "R201", Outcome.ALIAS),
    ("the first years", "group", "G1A", Outcome.ALIAS),
])
def test_resolution_paths(resolver, raw, kind, expected_id, expected_outcome):
    ref = resolver.resolve_reference("f", kind, raw)
    assert (ref.resolved_id, ref.outcome) == (expected_id, expected_outcome)
    assert ref.outcome.resolved


@pytest.mark.parametrize("raw,expected", [
    ("the first years", "first years"),
    ("The Big Lab", "big lab"),
    ("Dr. Lee", "lee"),
    ("the", "the"),
])
def test_leading_determiners_dropped_from_the_lookup_key(raw, expected):
    assert normalise(raw) == expected


def test_determiner_stripping_leaves_descriptiveness_intact(resolver):
    """The lookup key loses 'the'; the Phase 4.6 signal does not."""
    ref = resolver.resolve_reference("f", "group", "the first years")
    assert ref.outcome is Outcome.ALIAS
    assert ref.descriptive


def test_genuine_partial_match_still_reports_partial(resolver):
    ref = resolver.resolve_reference("f", "course", "Machine Learning Methods")
    assert ref.outcome is Outcome.PARTIAL
    assert ref.resolved_id == "CS301"
    assert ref.outcome.needs_confirmation


def test_partial_match_raises_a_review_flag(resolver):
    c = LecturerUnavailable(
        lecturer="Sarah Lee Jones", days=["Fri"], strength="hard",
        source_text="Sarah Lee Jones cannot teach Fridays",
    )
    resolved = resolver.resolve(c)
    assert "weak_entity_match" in codes(resolved)
    assert not resolved.blocked


def test_unknown_name_is_unresolved_and_offers_candidates(resolver):
    ref = resolver.resolve_reference("f", "lecturer", "Prof Zhang")
    assert ref.outcome is Outcome.UNRESOLVED
    assert ref.resolved_id is None
    assert ref.candidates, "an unresolved reference must offer options (Phase 4.6)"


def test_descriptive_reference_is_flagged_even_when_resolved(resolver):
    c = CourseRequiresRoom(
        course="Machine Learning", room="the big lab", strength="hard",
        source_text="ML must be in the big lab",
    )
    resolved = resolver.resolve(c)
    assert resolved.resolved_ids["room"] == "R201"
    assert "descriptive_reference" in codes(resolved)
    assert not resolved.blocked, "a resolved description is reviewable, not blocking"


def test_room_type_word_covering_two_configured_types_is_ambiguous(resolver):
    ref = resolver.resolve_reference("f", "room_type", "lab")
    assert ref.outcome is Outcome.AMBIGUOUS
    assert set(ref.candidates) == {"lab", "computer_lab"}


def test_specific_room_type_resolves(resolver):
    assert resolver.resolve_reference("f", "room_type", "computer lab").resolved_id == (
        "computer_lab"
    )
    assert resolver.resolve_reference("f", "room_type", "lecture theatre").resolved_id == (
        "lecture"
    )


# -- flags ---------------------------------------------------------------


def test_hallucinated_lecturer_blocks(resolver):
    c = LecturerUnavailable(
        lecturer="Zhang", days=["Mon"], strength="hard",
        source_text="Zhang cannot teach Monday",
    )
    resolved = resolver.resolve(c)
    assert resolved.blocked
    assert "unknown_entity" in codes(resolved)


def test_ambiguous_room_type_blocks(resolver):
    c = CourseRequiresRoomType(
        course="CS202L", room_type="lab", strength="hard",
        source_text="the practical must be in a lab",
    )
    resolved = resolver.resolve(c)
    assert resolved.blocked
    assert "ambiguous_entity" in codes(resolved)


def test_open_category_never_reaches_the_solver(resolver):
    c = Unsupported(note="cap on teaching days", source_text="two days a week")
    resolved = resolver.resolve(c)
    assert resolved.blocked
    assert "open_category" in codes(resolved)


def test_both_cue_kinds_present_is_flagged(resolver):
    c = LecturerUnavailable(
        lecturer="Lee", days=["Wed"], strength="hard",
        source_text="I can't do Wednesdays, though ideally I'd keep Thursday free",
    )
    assert "strength_ambiguous" in codes(resolver.resolve(c))


def test_no_cue_is_flagged(resolver):
    c = CourseRequiresRoom(
        course="CS301", room="R101", strength="hard",
        source_text="Machine Learning goes in Lecture Theatre 1",
    )
    assert "strength_uncued" in codes(resolver.resolve(c))


def test_cue_contradicting_the_parsed_strength_is_flagged(resolver):
    c = LecturerUnavailable(
        lecturer="Lee", days=["Fri"], strength="soft", weight=5,
        source_text="I cannot teach Fridays",
    )
    assert "strength_cue_mismatch" in codes(resolver.resolve(c))


def test_clean_hard_cue_raises_no_strength_flag(resolver):
    c = LecturerUnavailable(
        lecturer="Lee", days=["Fri"], strength="hard",
        source_text="I cannot teach Fridays",
    )
    assert not {c for c in codes(resolver.resolve(c)) if c.startswith("strength")}


# -- semantic checks against the department data -------------------------


def test_room_too_small_for_the_course_is_flagged(resolver):
    c = CourseRequiresRoom(
        course="CS202L", room="R203", strength="hard",
        source_text="the architecture practical must use the hardware lab",
    )
    resolved = resolver.resolve(c)
    assert "capacity_conflict" in codes(resolved)
    assert not resolved.blocked, "infeasibility is for the user to resolve, not a parse error"


def test_period_beyond_the_grid_blocks(resolver):
    c = NoEarlyStart(
        earliest_period=9, strength="hard", source_text="nothing before period 9"
    )
    resolved = resolver.resolve(c)
    assert "out_of_range" in codes(resolved)
    assert resolved.blocked


def test_untaught_day_is_flagged(resolver):
    c = LecturerUnavailable(
        lecturer="Lee", days=["Sun"], strength="hard",
        source_text="Lee cannot teach Sundays",
    )
    assert "out_of_range" in codes(resolver.resolve(c))


def test_cap_below_the_longest_course_is_flagged(resolver):
    c = MaxConsecutiveHours(
        max_hours=1, lecturer="L004", strength="hard",
        source_text="Okafor must never teach more than 1 hour in a row",
    )
    assert "capacity_conflict" in codes(resolver.resolve(c))


def test_flag_severities_are_only_block_or_review(resolver):
    c = LecturerUnavailable(
        lecturer="Zhang", days=["Sun"], strength="soft", weight=3,
        source_text="Zhang must not teach Sundays",
    )
    for f in resolver.resolve(c).flags:
        assert f.severity in (Severity.BLOCK, Severity.REVIEW)
