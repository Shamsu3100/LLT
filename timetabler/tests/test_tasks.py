"""Task keys, residual-error scoring, and the synthetic generator (Phase 6C)."""

from __future__ import annotations

import pytest

from timetabler.app.store import StudyStore
from timetabler.data import load_department
from timetabler.eval.export_study import session_rows
from timetabler.eval.synth import generate
from timetabler.eval.tasks import (
    Task,
    load_tasks,
    score_session,
    validate_against_department,
)
from timetabler.schema import ALL_TYPES, LecturerUnavailable


@pytest.fixture(scope="module")
def department():
    return load_department()


@pytest.fixture(scope="module")
def tasks():
    return load_tasks()


@pytest.fixture(scope="module")
def t1(tasks):
    return tasks.by_id("T1")


def altered(constraint, **fields):
    data = constraint.model_dump()
    data.update(fields)
    return type(constraint).model_validate(data)


# -- the task set --------------------------------------------------------


def test_sample_tasks_load(tasks):
    assert tasks.ids == ["T1", "T2", "T3", "T4"]


def test_tasks_are_large_enough_for_the_margin(tasks):
    """A task of 1-3 constraints makes the error rate coarser than the margin.

    With 3 intended constraints a participant's error rate can only land on
    0, 0.33, 0.67 or 1.0 - a granularity larger than the 5-percentage-point
    non-inferiority margin the study has to test. Measured across 8 seeds,
    enlarging the tasks cut the across-seed spread of the mean from
    0.000-0.178 to 0.058-0.116.
    """
    for task in tasks.tasks:
        assert task.size >= 6, f"{task.id} has only {task.size} constraints"


def test_task_keys_resolve_against_the_department(tasks, department):
    """A key naming an unknown entity scores every participant wrong."""
    assert validate_against_department(tasks, department) == []


def test_validation_catches_a_bad_key(department):
    from timetabler.eval.tasks import TaskSet

    bad = TaskSet(tasks=[Task(
        id="X", title="bad", brief=["a fact"],
        intended=[LecturerUnavailable(
            lecturer="Dr Lee", days=["Fri"], strength="hard", source_text="x"
        )],
    )])
    problems = validate_against_department(bad, department)
    assert problems and "not a canonical" in problems[0]


def test_briefs_are_facts_not_the_sentences_to_produce(tasks):
    """A brief that states the target sentence makes it a typing race.

    Weak but useful guard: no brief line may be a constraint's source_text.
    """
    for task in tasks.tasks:
        sources = {c.source_text.casefold() for c in task.intended}
        for line in task.brief:
            assert line.casefold() not in sources, f"{task.id}: {line}"


def test_the_task_set_exercises_an_open_category(tasks):
    """T4's correct answer is that the vocabulary cannot express it."""
    types = {c.type for t in tasks.tasks for c in t.intended}
    assert "unsupported" in types
    assert types <= set(ALL_TYPES)


# -- scoring -------------------------------------------------------------


def test_perfect_submission_scores_clean(t1, department):
    r = score_session(t1.intended, t1, department)
    assert (r.correct, r.missed, r.spurious) == (t1.size, 0, 0)
    # Regression: with no pred_ids supplied to align(), score_entities found
    # no ids to compare and counted every reference as unresolved, so a
    # perfect submission reported wrong_entity equal to its reference count.
    assert r.wrong_entity == 0
    assert r.error_rate == 0.0 and r.accuracy == 1.0


def test_missing_constraint_counted_as_missed(t1, department):
    r = score_session(t1.intended[:-1], t1, department)
    assert r.missed == 1 and r.correct == t1.size - 1
    assert r.error_rate == pytest.approx(1 / t1.size)


def test_wrong_strength_is_its_own_category(t1, department):
    submitted = list(t1.intended)
    soft_at = next(i for i, c in enumerate(submitted) if c.strength == "soft")
    submitted[soft_at] = altered(submitted[soft_at], strength="hard", weight=None)
    r = score_session(submitted, t1, department)
    assert r.wrong_strength == 1 and r.wrong_entity == 0
    assert r.correct == t1.size - 1


def test_wrong_entity_is_its_own_category(t1, department):
    submitted = list(t1.intended)
    submitted[0] = altered(submitted[0], lecturer="L003")
    r = score_session(submitted, t1, department)
    assert r.wrong_entity == 1 and r.wrong_strength == 0


def test_invented_constraint_counted_as_spurious(t1, department):
    submitted = list(t1.intended) + [LecturerUnavailable(
        lecturer="L002", days=["Mon"], strength="hard", source_text="invented"
    )]
    r = score_session(submitted, t1, department)
    assert r.spurious == 1 and r.correct == t1.size


def test_unresolvable_entity_counted_and_not_crashing(t1, department):
    submitted = list(t1.intended[:-1]) + [LecturerUnavailable(
        lecturer="Prof Zhang", days=["Mon"], strength="hard", source_text="x"
    )]
    r = score_session(submitted, t1, department)
    assert r.unresolved == 1
    assert r.missed == 1, "the dropped intended constraint is still absent"


def test_empty_submission_scores_everything_missed(t1, department):
    r = score_session([], t1, department)
    assert r.correct == 0 and r.missed == t1.size
    assert r.error_rate == 1.0


def test_scoring_accepts_stored_dicts(t1, department):
    """The export passes rows straight from SQLite."""
    as_dicts = [c.model_dump(mode="json") for c in t1.intended]
    assert score_session(as_dicts, t1, department).correct == t1.size


# -- the synthetic generator --------------------------------------------


def test_generate_builds_a_within_subject_study(tmp_path, department, tasks):
    db = tmp_path / "s.sqlite"
    stats = generate(db, participants=6, seed=3, abandon_rate=0.0)
    assert stats["sessions"] == 12, "two conditions per participant"
    with StudyStore(db) as store:
        rows = session_rows(store, {}, 180.0, tasks, department)
    assert {r["condition"] for r in rows} == {"form", "llm"}
    assert len({r["participant"] for r in rows}) == 6


def test_order_is_counterbalanced(tmp_path):
    db = tmp_path / "s.sqlite"
    generate(db, participants=6, seed=3, abandon_rate=0.0)
    with StudyStore(db) as store:
        first = {
            s["participant"]: s["condition"]
            for s in store.sessions() if s["order_index"] == 1
        }
    assert set(first.values()) == {"form", "llm"}, "both arms go first for someone"


def test_requested_time_reduction_appears(tmp_path, department, tasks):
    db = tmp_path / "s.sqlite"
    generate(db, participants=30, seed=11, time_reduction=0.30, abandon_rate=0.0)
    with StudyStore(db) as store:
        rows = session_rows(store, {}, 180.0, tasks, department)
    form = [r["total_seconds"] for r in rows if r["condition"] == "form"]
    llm = [r["total_seconds"] for r in rows if r["condition"] == "llm"]
    observed = 1 - (sum(llm) / len(llm)) / (sum(form) / len(form))
    assert 0.18 < observed < 0.42, f"asked for 30%, got {observed:.0%}"


def test_null_effect_produces_no_difference(tmp_path, department, tasks):
    """The check that matters: a pipeline finding an effect in null data is broken."""
    db = tmp_path / "s.sqlite"
    generate(db, participants=30, seed=11, time_reduction=0.0, abandon_rate=0.0)
    with StudyStore(db) as store:
        rows = session_rows(store, {}, 180.0, tasks, department)
    form = [r["total_seconds"] for r in rows if r["condition"] == "form"]
    llm = [r["total_seconds"] for r in rows if r["condition"] == "llm"]
    observed = 1 - (sum(llm) / len(llm)) / (sum(form) / len(form))
    assert abs(observed) < 0.12, f"no effect was injected, found {observed:.0%}"


def test_verification_time_only_in_the_llm_arm(tmp_path, department, tasks):
    db = tmp_path / "s.sqlite"
    generate(db, participants=8, seed=4, abandon_rate=0.0)
    with StudyStore(db) as store:
        rows = session_rows(store, {}, 180.0, tasks, department)
    for row in rows:
        if row["condition"] == "form":
            assert row["verification_seconds"] == 0.0
        else:
            assert row["verification_seconds"] > 0.0


def test_abandoned_sessions_are_produced_when_asked(tmp_path):
    db = tmp_path / "s.sqlite"
    stats = generate(db, participants=40, seed=2, abandon_rate=0.5)
    assert stats["abandoned"] > 0, "the exclusion logic needs something to exclude"


def test_residual_error_is_scored_for_every_session(tmp_path, department, tasks):
    db = tmp_path / "s.sqlite"
    generate(db, participants=8, seed=5, error_rate=0.2, abandon_rate=0.0)
    with StudyStore(db) as store:
        rows = session_rows(store, {}, 180.0, tasks, department)
    assert all(r["task_intended"] != "" for r in rows)
    assert all(r["correct_constraints"] != "" for r in rows)
    assert all(r["throughput_per_minute"] != "" for r in rows)


def test_without_a_task_key_the_columns_stay_blank(tmp_path):
    db = tmp_path / "s.sqlite"
    generate(db, participants=4, seed=5, abandon_rate=0.0)
    with StudyStore(db) as store:
        rows = session_rows(store, {}, 180.0, None, None)
    assert all(r["residual_error_rate"] == "" for r in rows)
    assert all(r["correct_constraints"] == "" for r in rows)


def test_generate_refuses_to_overwrite_an_existing_log(tmp_path):
    """A real study log must never be clobbered by a synthetic run."""
    db = tmp_path / "s.sqlite"
    generate(db, participants=2, seed=1, abandon_rate=0.0)
    with pytest.raises(FileExistsError, match="never be overwritten"):
        generate(db, participants=2, seed=1)


def test_generate_is_deterministic(tmp_path, department, tasks):
    def totals(name, seed):
        db = tmp_path / name
        generate(db, participants=5, seed=seed, abandon_rate=0.0)
        with StudyStore(db) as store:
            return [r["total_seconds"] for r in session_rows(store, {}, 180.0, tasks, department)]

    assert totals("a.sqlite", 9) == totals("b.sqlite", 9)
    assert totals("c.sqlite", 9) != totals("d.sqlite", 10)
