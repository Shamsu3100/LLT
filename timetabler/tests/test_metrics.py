"""Accuracy metrics and inter-rater agreement (build plan Phase 6A)."""

from __future__ import annotations

import pytest

from timetabler.data import load_department
from timetabler.eval.dataset import load_items, paired_labels
from timetabler.eval.metrics import (
    EntityError,
    aggregate,
    agreement_report,
    align,
    canonical,
    cohen_kappa,
    score_sentence,
)
from timetabler.parser.resolve import EntityResolver
from timetabler.schema import CourseRequiresRoom, LecturerUnavailable, Unsupported


@pytest.fixture(scope="module")
def resolver():
    return EntityResolver(load_department())


def unavail(lecturer, days=("Fri",), strength="hard", weight=None, text="x"):
    return LecturerUnavailable(
        lecturer=lecturer, days=list(days), strength=strength, weight=weight,
        source_text=text,
    )


# -- canonical form ------------------------------------------------------


def test_source_text_does_not_affect_equality():
    a = unavail("L001", text="I can't teach Fridays")
    b = unavail("L001", text="not available Fridays")
    assert canonical(a) == canonical(b)


def test_weight_excluded_by_default_and_scored_on_request():
    a = unavail("L001", strength="soft", weight=3)
    b = unavail("L001", strength="soft", weight=8)
    assert canonical(a) == canonical(b)
    assert canonical(a, include_weight=True) != canonical(b, include_weight=True)


def test_strength_always_affects_equality():
    assert canonical(unavail("L001")) != canonical(
        unavail("L001", strength="soft", weight=5)
    )


def test_day_order_does_not_affect_equality():
    assert canonical(unavail("L001", days=("Fri", "Mon"))) == canonical(
        unavail("L001", days=("Mon", "Fri"))
    )


def test_gold_ids_and_resolved_ids_compare_equal(resolver):
    """Regression: a case mismatch made every entity-bearing constraint fail."""
    gold = unavail("L001")
    pred = unavail("Dr. Lee")
    resolved = resolver.resolve(pred)
    assert canonical(pred, resolved.resolved_ids) == canonical(gold)


# -- alignment -----------------------------------------------------------


def test_alignment_is_one_to_one():
    preds = [unavail("L001"), unavail("L002")]
    golds = [unavail("L002"), unavail("L001")]
    pairs = align(preds, golds)
    assert all(p.matched for p in pairs)
    assert len(pairs) == 2


def test_unmatched_prediction_is_spurious():
    pairs = align([unavail("L001"), unavail("L002")], [unavail("L001")])
    assert sum(1 for p in pairs if p.gold is None) == 1


def test_unmatched_gold_is_missed():
    pairs = align([unavail("L001")], [unavail("L001"), unavail("L002")])
    assert sum(1 for p in pairs if p.pred is None) == 1


# -- sentence scoring ----------------------------------------------------


def test_correct_prediction_scores_perfectly(resolver):
    gold = [unavail("L001")]
    pred = [unavail("Lee")]
    resolved = [resolver.resolve(c) for c in pred]
    s = score_sentence(
        "t1", "I can't teach Fridays", pred, gold,
        references=[r.references for r in resolved],
        resolved_ids=[r.resolved_ids for r in resolved],
    )
    assert s.exact_match
    assert s.entities.accuracy == 1.0


def test_unresolved_entity_counted_as_unknown_name(resolver):
    gold = [unavail("L001")]
    pred = [unavail("Zhang")]
    resolved = [resolver.resolve(c) for c in pred]
    s = score_sentence(
        "t2", "Zhang can't teach Fridays", pred, gold,
        references=[r.references for r in resolved],
        resolved_ids=[r.resolved_ids for r in resolved],
    )
    assert not s.exact_match
    assert s.entities.counts[EntityError.UNKNOWN_NAME.value] == 1
    assert s.entities.counts[EntityError.MISRESOLVED.value] == 0


def test_wrong_entity_counted_as_misresolved(resolver):
    gold = [unavail("L001")]
    pred = [unavail("Rahman")]
    resolved = [resolver.resolve(c) for c in pred]
    s = score_sentence(
        "t3", "x", pred, gold,
        references=[r.references for r in resolved],
        resolved_ids=[r.resolved_ids for r in resolved],
    )
    assert s.entities.counts[EntityError.MISRESOLVED.value] == 1
    assert s.entities.counts[EntityError.UNKNOWN_NAME.value] == 0


def test_descriptive_references_are_counted_separately(resolver):
    """Phase 4.6: a description read wrongly is its own failure class."""
    gold = [CourseRequiresRoom(
        course="CS301", room="R101", strength="hard", source_text="x"
    )]
    pred = [CourseRequiresRoom(
        course="Machine Learning", room="the big lab", strength="hard",
        source_text="x",
    )]
    resolved = [resolver.resolve(c) for c in pred]
    s = score_sentence(
        "t4", "ML goes in the big lab", pred, gold,
        references=[r.references for r in resolved],
        resolved_ids=[r.resolved_ids for r in resolved],
    )
    assert s.entities.descriptive_total == 1, "'the big lab' is descriptive"
    assert s.entities.descriptive_accuracy == 0.0, "it resolved to R201, gold is R101"
    assert s.entities.counts[EntityError.CORRECT.value] == 1, "the course was right"


def test_type_error_recorded(resolver):
    gold = [unavail("L001")]
    pred = [Unsupported(note="n", source_text="x")]
    s = score_sentence("t5", "x", pred, gold)
    assert not s.exact_match
    assert s.matched_pairs == []
    assert sum(1 for p in s.pairs if p.gold is None) == 1


def test_parse_failure_cannot_be_an_exact_match():
    s = score_sentence("t6", "x", [], [unavail("L001")], parse_failed=True)
    assert not s.exact_match


# -- aggregation ---------------------------------------------------------


def test_aggregate_reports_every_specified_measure(resolver):
    gold = [unavail("L001")]
    pred = [unavail("Lee")]
    resolved = [resolver.resolve(c) for c in pred]
    s = score_sentence(
        "t7", "x", pred, gold,
        references=[r.references for r in resolved],
        resolved_ids=[r.resolved_ids for r in resolved],
    )
    summary = aggregate([s])
    for key in ("exact_match", "type_accuracy", "strength_accuracy",
                "field_accuracy", "entity", "confusion"):
        assert key in summary
    assert summary["exact_match"] == 1.0
    assert summary["type_accuracy"] == 1.0


def test_aggregate_on_empty_input_returns_none_not_a_crash():
    summary = aggregate([])
    assert summary["n_sentences"] == 0
    assert summary["exact_match"] is None


# -- agreement -----------------------------------------------------------


def test_identical_labellers_give_kappa_one():
    assert cohen_kappa(["a", "b", "a"], ["a", "b", "a"]) == pytest.approx(1.0)


def test_kappa_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same items"):
        cohen_kappa(["a"], ["a", "b"])


def test_agreement_report_on_the_seed_set():
    items = load_items()
    a, b, covered = paired_labels(items, "A", "B")
    report = agreement_report(a, b)
    assert len(covered) == len(items)
    assert 0.0 < report["type_kappa"] < 1.0, "the seed set has deliberate disagreements"
    assert report["set_exact_agreement"] < 1.0
    assert report["unit_of_analysis"]


def test_seed_set_covers_the_whole_vocabulary():
    from timetabler.schema import ALL_TYPES

    items = load_items()
    covered = {c.type for i in items for c in i.gold}
    assert covered == set(ALL_TYPES), f"uncovered: {set(ALL_TYPES) - covered}"


def test_seed_set_is_marked_synthetic():
    """Guard: the Phase 6A claim must not rest on these items."""
    items = load_items()
    assert all(i.source == "synthetic" for i in items)
