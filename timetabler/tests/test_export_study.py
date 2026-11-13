"""Study export to tidy CSV (build plan Phase 6C)."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from timetabler.app.recorder import StudyRecorder
from timetabler.app.store import Condition, Phase, StudyStore
from timetabler.data import load_department
from timetabler.eval.export_study import (
    CONSTRAINT_COLUMNS,
    SESSION_COLUMNS,
    is_analysable,
    load_correctness,
    main,
    summarise,
)
from timetabler.schema import CourseRequiresRoom, LecturerUnavailable, Unsupported


@pytest.fixture(scope="module")
def department():
    return load_department()


def unavail(text="I can't teach Fridays", strength="hard"):
    return LecturerUnavailable(
        lecturer="Lee", days=["Fri"], strength=strength,
        weight=None if strength == "hard" else 5, source_text=text,
    )


@pytest.fixture
def study(tmp_path, department):
    """One form session, one LLM session, one abandoned session."""
    db = tmp_path / "study.sqlite"
    with StudyStore(db) as store:
        form = StudyRecorder(store, "P01", Condition.FORM, department=department,
                             order_index=1)
        form.enter(Phase.ENTRY, at=0.0)
        form.record_form_add(CourseRequiresRoom(
            course="CS301", room="R101", strength="hard", source_text="[form] a"
        ), at=30.0)
        form.complete(at=100.0)

        llm = StudyRecorder(store, "P01", Condition.LLM, department=department,
                            order_index=2, model="claude-sonnet-5-5")
        llm.enter(Phase.ENTRY, at=0.0)
        llm.record_text_submission("I can't teach Fridays", at=20.0)
        cs = [unavail(), unavail("soft one", "soft"),
              Unsupported(note="n", source_text="two days a week")]
        ids = llm.record_parse("x", cs, at=25.0)
        llm.enter(Phase.VERIFICATION, at=30.0)
        llm.accept(ids[0], cs[0], at=40.0)
        llm.edit(ids[1], unavail("soft one"), changed=["strength"], at=50.0)
        llm.reject(ids[2], reason="unsupported", at=55.0)
        llm.complete(at=80.0)

        abandoned = StudyRecorder(store, "P02", Condition.LLM,
                                  department=department)
        abandoned.enter(Phase.ENTRY, at=0.0)
        abandoned.record_text_submission("walked away", at=10.0)
        # never completed: the ENTRY phase stays open

    return db


def read(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))


def run_export(study, tmp_path, *extra):
    out = tmp_path / "exports"
    assert main(["--db", str(study), "--out", str(out), *extra]) == 0
    return out


# -- the three tables ----------------------------------------------------


def test_all_three_tables_are_written(study, tmp_path):
    out = run_export(study, tmp_path)
    for name in ("sessions.csv", "constraints.csv", "events.csv"):
        assert (out / name).exists(), name


def test_one_session_row_per_participant_condition(study, tmp_path):
    rows = read(run_export(study, tmp_path) / "sessions.csv")
    assert len(rows) == 3
    assert {(r["participant"], r["condition"]) for r in rows} == {
        ("P01", "form"), ("P01", "llm"), ("P02", "llm")
    }


def test_session_columns_are_stable(study, tmp_path):
    rows = read(run_export(study, tmp_path) / "sessions.csv")
    assert list(rows[0]) == SESSION_COLUMNS


def test_constraint_columns_are_stable(study, tmp_path):
    rows = read(run_export(study, tmp_path) / "constraints.csv")
    assert list(rows[0]) == CONSTRAINT_COLUMNS


# -- the measurement that matters ---------------------------------------


def test_entry_and_verification_are_separate_columns(study, tmp_path):
    """The whole point of Phase 5.4: one total cannot answer the objection."""
    rows = read(run_export(study, tmp_path) / "sessions.csv")
    llm = next(r for r in rows if r["condition"] == "llm" and r["completed"] == "True")
    assert float(llm["entry_seconds"]) == 30.0
    assert float(llm["verification_seconds"]) == 50.0
    assert float(llm["total_seconds"]) == 80.0
    assert float(llm["verification_share"]) == pytest.approx(0.625)


def test_form_condition_records_no_verification_time(study, tmp_path):
    rows = read(run_export(study, tmp_path) / "sessions.csv")
    form = next(r for r in rows if r["condition"] == "form")
    assert float(form["verification_seconds"]) == 0.0
    assert float(form["total_seconds"]) == 100.0


def test_decision_counts_land_in_their_own_columns(study, tmp_path):
    rows = read(run_export(study, tmp_path) / "sessions.csv")
    llm = next(r for r in rows if r["condition"] == "llm" and r["completed"] == "True")
    assert (llm["accepted"], llm["edited"], llm["rejected"]) == ("1", "1", "1")
    assert llm["constraints_proposed"] == "3"
    assert llm["constraints_kept"] == "2", "the rejected one is not kept"


def test_strength_edit_is_visible_per_constraint(study, tmp_path):
    """An edit to strength is a hard/soft failure, and must be attributable."""
    rows = read(run_export(study, tmp_path) / "constraints.csv")
    edited = next(r for r in rows if r["status"] == "edited")
    assert edited["strength_before"] == "soft"
    assert edited["strength_after"] == "hard"
    assert edited["strength_changed"] == "True"
    assert edited["edit_count"] == "1"


# -- abandoned sessions --------------------------------------------------


def test_abandoned_session_is_flagged_not_dropped(study, tmp_path):
    """Its row stays in the CSV; the researcher decides what to do with it."""
    rows = read(run_export(study, tmp_path) / "sessions.csv")
    abandoned = next(r for r in rows if r["participant"] == "P02")
    assert abandoned["completed"] == "False"
    assert abandoned["unclosed_phases"] == "entry"


def test_abandoned_session_is_excluded_from_the_means():
    """Regression: an unclosed phase closes at the last event seen.

    A session left open overnight produces a span of tens of thousands of
    seconds, which swamped a condition's mean before this was fixed.
    """
    good = {
        "condition": "llm", "completed": True, "unclosed_phases": "",
        "participant": "P01", "session_id": "a" * 12,
        "entry_seconds": 30.0, "verification_seconds": 50.0,
        "total_seconds": 80.0, "idle_spans": 0, "correct_constraints": "",
    }
    ruinous = {**good, "completed": False, "unclosed_phases": "entry",
               "participant": "P02", "session_id": "b" * 12,
               "entry_seconds": 256_373_074.0, "total_seconds": 256_373_074.0}

    text = summarise([good, ruinous])
    assert "80.0" in text
    assert "256373074" not in text, "an abandoned session must not reach the mean"
    assert "excluded from the means" in text
    assert "P02" in text, "and it must be named"


@pytest.mark.parametrize("row,expected", [
    ({"completed": True, "unclosed_phases": ""}, True),
    ({"completed": False, "unclosed_phases": ""}, False),
    ({"completed": True, "unclosed_phases": "entry"}, False),
])
def test_analysable_definition(row, expected):
    assert is_analysable(row) is expected


# -- correctness and throughput -----------------------------------------


def test_throughput_blank_until_correctness_is_supplied(study, tmp_path):
    rows = read(run_export(study, tmp_path) / "sessions.csv")
    assert all(r["throughput_per_minute"] == "" for r in rows)
    assert all(r["correct_constraints"] == "" for r in rows)


def test_throughput_computed_from_a_correctness_file(study, tmp_path):
    out = tmp_path / "exports"
    main(["--db", str(study), "--out", str(out)])
    sessions = read(out / "sessions.csv")
    llm = next(r for r in sessions
               if r["condition"] == "llm" and r["completed"] == "True")

    correct = tmp_path / "correct.csv"
    correct.write_text(
        f"session_id,correct_constraints\n{llm['session_id']},2\n", encoding="utf-8"
    )
    out2 = run_export(study, tmp_path / "second", "--correct", str(correct))
    scored = next(r for r in read(out2 / "sessions.csv")
                  if r["session_id"] == llm["session_id"])
    assert scored["correct_constraints"] == "2"
    # 2 constraints over 80 seconds = 1.5 per minute
    assert float(scored["throughput_per_minute"]) == pytest.approx(1.5)


def test_correctness_file_tolerates_blank_rows(tmp_path):
    path = tmp_path / "c.csv"
    path.write_text(
        "session_id,correct_constraints\nabc,3\n,\ndef,\n", encoding="utf-8"
    )
    assert load_correctness(str(path)) == {"abc": 3}


def test_no_correctness_file_is_fine():
    assert load_correctness(None) == {}


# -- safety --------------------------------------------------------------


def test_export_carries_no_credential(study, tmp_path):
    """Every written file is scrubbed and then checked."""
    from timetabler.app.credentials import assert_no_credentials

    out = run_export(study, tmp_path)
    for name in ("sessions.csv", "constraints.csv", "events.csv"):
        assert_no_credentials((out / name).read_text(encoding="utf-8"), where=name)


def test_missing_database_exits_with_guidance(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(["--db", str(tmp_path / "nope.sqlite"), "--out", str(tmp_path)])
    assert "TIMETABLER_DB" in str(exc.value)


def test_json_export_is_optional(study, tmp_path):
    out = run_export(study, tmp_path)
    assert not (out / "study.json").exists()
    out2 = run_export(study, tmp_path / "with-json", "--json")
    assert (out2 / "study.json").exists()


def test_empty_study_summarises_without_crashing():
    assert "no sessions recorded" in summarise([])
