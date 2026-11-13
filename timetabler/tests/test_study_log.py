"""Study logging and phase timing (build plan Phase 5.4).

These tests carry the measurement claim: if entry and verification time are
not separated correctly, the Phase 6C primary outcome is wrong whatever the
interface looks like. Timestamps are injected throughout, so nothing here
depends on wall-clock timing.
"""

from __future__ import annotations

import pytest

from timetabler.app.recorder import StudyRecorder
from timetabler.app.store import (
    Condition,
    Decision,
    EventKind,
    Phase,
    StudyStore,
)
from timetabler.app.timing import (
    DEFAULT_IDLE_THRESHOLD,
    build_report,
    compare_conditions,
    decision_counts,
    phase_spans,
    throughput,
)
from timetabler.data import load_department
from timetabler.schema import CourseRequiresRoom, LecturerUnavailable, Unsupported


@pytest.fixture
def store():
    with StudyStore(":memory:") as s:
        yield s


@pytest.fixture(scope="module")
def department():
    return load_department()


def unavail(lecturer="Lee", text="I can't teach Fridays", strength="hard"):
    return LecturerUnavailable(
        lecturer=lecturer, days=["Fri"], strength=strength,
        weight=None if strength == "hard" else 5, source_text=text,
    )


# -- store ---------------------------------------------------------------


def test_session_round_trips(store):
    sid = store.create_session("P01", Condition.LLM.value, at=100.0)
    session = store.session(sid)
    assert session["participant"] == "P01"
    assert session["condition"] == "llm"
    assert session["completed_at"] is None
    store.complete_session(sid, at=200.0)
    assert store.session(sid)["completed_at"] == 200.0


def test_session_start_is_logged(store):
    sid = store.create_session("P01", Condition.LLM.value, at=100.0)
    assert [e.kind for e in store.events(sid)] == [EventKind.SESSION_START.value]


def test_sessions_filter_by_participant_and_condition(store):
    store.create_session("P01", Condition.FORM.value, at=1.0)
    store.create_session("P01", Condition.LLM.value, at=2.0)
    store.create_session("P02", Condition.LLM.value, at=3.0)
    assert len(store.sessions(participant="P01")) == 2
    assert len(store.sessions(condition="llm")) == 2


def test_constraint_decision_recorded(store):
    sid = store.create_session("P01", Condition.LLM.value, at=100.0)
    cid = store.add_constraint(
        sid, 1, "I can't teach Fridays", unavail().model_dump(mode="json"), at=101.0
    )
    final = unavail(strength="soft").model_dump(mode="json")
    store.decide(cid, Decision.EDITED, final=final, increment_edits=True, at=110.0)

    record = store.constraints(sid)[0]
    assert record["status"] == "edited"
    assert record["strength_before"] == "hard"
    assert record["strength_after"] == "soft"
    assert record["edit_count"] == 1
    assert record["decided_at"] == 110.0


def test_accepted_constraints_exclude_rejections(store):
    sid = store.create_session("P01", Condition.LLM.value, at=1.0)
    keep = store.add_constraint(sid, 1, "a", unavail(text="a").model_dump(mode="json"))
    drop = store.add_constraint(sid, 2, "b", unavail(text="b").model_dump(mode="json"))
    store.decide(keep, Decision.ACCEPTED, final=unavail(text="a").model_dump(mode="json"))
    store.decide(drop, Decision.REJECTED)
    final = store.accepted_constraints(sid)
    assert len(final) == 1
    assert final[0]["source_text"] == "a"


def test_store_is_usable_from_another_thread(store):
    """Regression: Streamlit reruns the script on a different thread each time.

    A default sqlite3 connection is bound to its creating thread, so the store
    built on first load raised on the first logged action of a real session.
    """
    import threading

    sid = store.create_session("P01", Condition.LLM.value, at=1.0)
    errors: list[Exception] = []

    def work() -> None:
        try:
            cid = store.add_constraint(
                sid, 1, "x", unavail().model_dump(mode="json"), at=2.0
            )
            store.decide(cid, Decision.ACCEPTED,
                         final=unavail().model_dump(mode="json"), at=3.0)
            store.log(sid, Phase.ENTRY.value, EventKind.TEXT_SUBMIT.value, at=4.0)
        except Exception as exc:  # pragma: no cover - the failure being guarded
            errors.append(exc)

    thread = threading.Thread(target=work)
    thread.start()
    thread.join()

    assert not errors, f"store failed off-thread: {errors}"
    assert len(store.constraints(sid)) == 1


def test_concurrent_writers_do_not_lose_events(store):
    import threading

    sid = store.create_session("P01", Condition.LLM.value, at=0.0)

    def write(n: int) -> None:
        for i in range(20):
            store.log(sid, Phase.ENTRY.value, EventKind.NOTE.value,
                      payload={"w": n, "i": i}, at=float(n * 100 + i))

    threads = [threading.Thread(target=write, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    notes = [e for e in store.events(sid) if e.kind == EventKind.NOTE.value]
    assert len(notes) == 80


def test_flags_survive_the_round_trip(store):
    sid = store.create_session("P01", Condition.LLM.value, at=1.0)
    cid = store.add_constraint(
        sid, 1, "x", unavail().model_dump(mode="json"),
        flags=[{"code": "unknown_entity", "severity": "block"}],
    )
    assert store.constraints(sid)[0]["flags"][0]["code"] == "unknown_entity"


# -- phase spans ---------------------------------------------------------


def make_events(store, sid, script):
    for at, phase, kind in script:
        store.log(sid, phase, kind, at=at)
    return store.events(sid)


def test_entry_and_verification_measured_separately(store):
    """The point of Phase 5.4: one total cannot answer the objection."""
    sid = store.create_session("P01", Condition.LLM.value, at=0.0)
    events = make_events(store, sid, [
        (10.0, Phase.ENTRY.value, EventKind.PHASE_ENTER.value),
        (40.0, Phase.ENTRY.value, EventKind.PHASE_EXIT.value),
        (41.0, Phase.VERIFICATION.value, EventKind.PHASE_ENTER.value),
        (91.0, Phase.VERIFICATION.value, EventKind.PHASE_EXIT.value),
    ])
    report = build_report(events)
    assert report.entry_seconds == 30.0
    assert report.verification_seconds == 50.0
    assert report.total_seconds == 80.0
    assert report.verification_share == pytest.approx(50.0 / 80.0)


def test_revisiting_a_phase_accumulates(store):
    sid = store.create_session("P01", Condition.LLM.value, at=0.0)
    events = make_events(store, sid, [
        (10.0, Phase.ENTRY.value, EventKind.PHASE_ENTER.value),
        (20.0, Phase.ENTRY.value, EventKind.PHASE_EXIT.value),
        (30.0, Phase.VERIFICATION.value, EventKind.PHASE_ENTER.value),
        (35.0, Phase.VERIFICATION.value, EventKind.PHASE_EXIT.value),
        (40.0, Phase.ENTRY.value, EventKind.PHASE_ENTER.value),
        (55.0, Phase.ENTRY.value, EventKind.PHASE_EXIT.value),
    ])
    report = build_report(events)
    assert report.entry_seconds == 25.0, "10s then 15s"
    assert report.verification_seconds == 5.0


def test_result_and_setup_time_excluded_from_elicitation(store):
    sid = store.create_session("P01", Condition.LLM.value, at=0.0)
    events = make_events(store, sid, [
        (10.0, Phase.ENTRY.value, EventKind.PHASE_ENTER.value),
        (20.0, Phase.ENTRY.value, EventKind.PHASE_EXIT.value),
        (20.0, Phase.RESULT.value, EventKind.PHASE_ENTER.value),
        (900.0, Phase.RESULT.value, EventKind.PHASE_EXIT.value),
    ])
    report = build_report(events)
    assert report.total_seconds == 10.0
    assert report.raw[Phase.RESULT.value] == 880.0


def test_unclosed_phase_is_closed_and_named(store):
    sid = store.create_session("P01", Condition.LLM.value, at=0.0)
    events = make_events(store, sid, [
        (10.0, Phase.ENTRY.value, EventKind.PHASE_ENTER.value),
        (30.0, Phase.ENTRY.value, EventKind.TEXT_SUBMIT.value),
    ])
    report = build_report(events)
    assert report.unclosed_phases == [Phase.ENTRY.value]
    assert report.entry_seconds == 20.0, "closed at the last event seen"


def test_reenter_without_exit_closes_the_previous_span(store):
    sid = store.create_session("P01", Condition.LLM.value, at=0.0)
    events = make_events(store, sid, [
        (10.0, Phase.ENTRY.value, EventKind.PHASE_ENTER.value),
        (20.0, Phase.ENTRY.value, EventKind.PHASE_ENTER.value),
        (30.0, Phase.ENTRY.value, EventKind.PHASE_EXIT.value),
    ])
    report = build_report(events)
    assert report.entry_seconds == 20.0, "10-20 then 20-30, not double-counted"


def test_spans_are_never_negative(store):
    sid = store.create_session("P01", Condition.LLM.value, at=0.0)
    spans, _ = phase_spans(make_events(store, sid, [
        (50.0, Phase.ENTRY.value, EventKind.PHASE_ENTER.value),
        (40.0, Phase.ENTRY.value, EventKind.PHASE_EXIT.value),
    ]))
    assert all(s.duration >= 0 for s in spans)


# -- idle handling -------------------------------------------------------


def test_long_span_reported_raw_and_capped(store):
    sid = store.create_session("P01", Condition.LLM.value, at=0.0)
    events = make_events(store, sid, [
        (0.0, Phase.ENTRY.value, EventKind.PHASE_ENTER.value),
        (600.0, Phase.ENTRY.value, EventKind.PHASE_EXIT.value),
    ])
    report = build_report(events, idle_threshold=180.0)
    assert report.entry_seconds == 600.0, "the raw figure is never altered"
    assert report.capped[Phase.ENTRY.value] == 180.0
    assert report.idle_spans == 1


def test_short_spans_are_not_capped(store):
    sid = store.create_session("P01", Condition.LLM.value, at=0.0)
    events = make_events(store, sid, [
        (0.0, Phase.ENTRY.value, EventKind.PHASE_ENTER.value),
        (30.0, Phase.ENTRY.value, EventKind.PHASE_EXIT.value),
    ])
    report = build_report(events)
    assert report.capped[Phase.ENTRY.value] == 30.0
    assert report.idle_spans == 0
    assert report.idle_threshold == DEFAULT_IDLE_THRESHOLD


# -- recorder ------------------------------------------------------------


def test_recorder_refuses_two_open_phases(store, department):
    """Entry and verification time must not both run and double-count."""
    rec = StudyRecorder(store, "P01", Condition.LLM, department=department)
    rec.enter(Phase.ENTRY, at=10.0)
    rec.enter(Phase.VERIFICATION, at=20.0)
    rec.leave(at=35.0)
    report = rec.timing()
    assert report.entry_seconds == 10.0
    assert report.verification_seconds == 15.0
    assert report.unclosed_phases == []


def test_reentering_the_same_phase_is_a_no_op(store, department):
    rec = StudyRecorder(store, "P01", Condition.LLM, department=department)
    rec.enter(Phase.ENTRY, at=10.0)
    rec.enter(Phase.ENTRY, at=15.0)
    rec.leave(at=30.0)
    assert rec.timing().entry_seconds == 20.0


def test_recorder_logs_a_full_llm_pass(store, department):
    rec = StudyRecorder(store, "P01", Condition.LLM, department=department)
    rec.enter(Phase.ENTRY, at=0.0)
    rec.record_text_submission("I can't teach Fridays", at=5.0)
    ids = rec.record_parse("I can't teach Fridays", [unavail()], at=6.0)
    rec.enter(Phase.VERIFICATION, at=7.0)
    rec.accept(ids[0], unavail(), at=12.0)
    rec.complete(at=20.0)

    report = rec.report(correct_constraints=1)
    assert report["constraints_proposed"] == 1
    assert report["constraints_kept"] == 1
    assert report["decisions"]["accepted"] == 1
    assert report["timing"]["entry_seconds"] == 7.0
    assert report["timing"]["verification_seconds"] == 13.0
    assert report["throughput_per_minute"] == pytest.approx(1 / (20 / 60))


def test_edit_records_which_fields_changed(store, department):
    """Phase 6A builds its error categories from these field names."""
    rec = StudyRecorder(store, "P01", Condition.LLM, department=department)
    rec.enter(Phase.VERIFICATION, at=0.0)
    ids = rec.record_parse("x", [unavail()], at=1.0)
    rec.edit(ids[0], unavail(strength="soft"), changed=["strength"], at=5.0)

    record = store.constraints(rec.session_id)[0]
    assert record["status"] == "edited"
    assert record["edit_count"] == 1
    edits = [e for e in store.events(rec.session_id)
             if e.kind == EventKind.CONSTRAINT_EDIT.value]
    assert edits[0].payload["changed"] == ["strength"]


def test_rejected_constraint_is_not_in_the_final_set(store, department):
    rec = StudyRecorder(store, "P01", Condition.LLM, department=department)
    ids = rec.record_parse("x", [unavail(text="a"), unavail(text="b")], at=1.0)
    rec.accept(ids[0], unavail(text="a"), at=2.0)
    rec.reject(ids[1], reason="misread", at=3.0)
    assert len(rec.final_constraints()) == 1
    assert rec.report()["decisions"]["rejected"] == 1


def test_parse_failure_records_no_constraints(store, department):
    rec = StudyRecorder(store, "P01", Condition.LLM, department=department)
    rec.enter(Phase.ENTRY, at=0.0)
    assert rec.record_parse("gibberish", [], error="no JSON object", at=1.0) == []
    kinds = [e.kind for e in store.events(rec.session_id)]
    assert EventKind.PARSE_FAILED.value in kinds
    assert EventKind.PARSE_COMPLETE.value not in kinds


def test_clarification_is_logged_and_the_constraint_dropped(store, department):
    rec = StudyRecorder(store, "P01", Condition.LLM, department=department)
    rec.enter(Phase.VERIFICATION, at=0.0)
    ids = rec.record_parse(
        "x", [Unsupported(note="n", source_text="x")], at=1.0
    )
    rec.answer_clarification(ids[0], "I meant Lab A", at=4.0)
    rec.reject(ids[0], reason="clarified", at=5.0)
    assert rec.final_constraints() == []
    answers = [e for e in store.events(rec.session_id)
               if e.kind == EventKind.CLARIFICATION_ANSWER.value]
    assert answers[0].payload["characters"] == len("I meant Lab A")


# -- the form baseline ---------------------------------------------------


def test_form_condition_has_zero_verification_time(store, department):
    """Condition 1 composes constraints field by field, so nothing is verified."""
    rec = StudyRecorder(store, "P01", Condition.FORM, department=department)
    rec.enter(Phase.ENTRY, at=0.0)
    rec.record_form_add(
        CourseRequiresRoom(
            course="CS301", room="R101", strength="hard", source_text="[form] x"
        ),
        at=30.0,
    )
    rec.complete(at=60.0)
    report = rec.report()
    assert report["timing"]["entry_seconds"] == 60.0
    assert report["timing"]["verification_seconds"] == 0.0
    assert report["timing"]["total_seconds"] == 60.0
    assert report["decisions"]["accepted"] == 1


def test_form_removal_drops_the_constraint(store, department):
    rec = StudyRecorder(store, "P01", Condition.FORM, department=department)
    cid = rec.record_form_add(
        CourseRequiresRoom(
            course="CS301", room="R101", strength="hard", source_text="[form] x"
        ),
        at=1.0,
    )
    rec.record_form_remove(cid, at=2.0)
    assert rec.final_constraints() == []


# -- aggregation ---------------------------------------------------------


def test_throughput_is_per_minute():
    assert throughput(6, 120.0) == pytest.approx(3.0)
    assert throughput(1, 0.0) is None


def test_decision_counts_sum_edits():
    records = [
        {"status": "accepted", "edit_count": 0},
        {"status": "edited", "edit_count": 2},
        {"status": "rejected", "edit_count": 0},
    ]
    counts = decision_counts(records)
    assert counts["accepted"] == 1 and counts["edited"] == 1
    assert counts["edit_events"] == 2


def test_compare_conditions_reports_both_components(store, department):
    rec_a = StudyRecorder(store, "P01", Condition.FORM, department=department)
    rec_a.enter(Phase.ENTRY, at=0.0)
    rec_a.complete(at=100.0)

    rec_b = StudyRecorder(store, "P01", Condition.LLM, department=department)
    rec_b.enter(Phase.ENTRY, at=0.0)
    rec_b.enter(Phase.VERIFICATION, at=40.0)
    rec_b.complete(at=70.0)

    out = compare_conditions({"form": [rec_a.timing()], "llm": [rec_b.timing()]})
    assert out["form"]["mean_total_seconds"] == 100.0
    assert out["form"]["mean_verification_seconds"] == 0.0
    assert out["llm"]["mean_entry_seconds"] == 40.0
    assert out["llm"]["mean_verification_seconds"] == 30.0


def test_compare_conditions_handles_an_empty_arm():
    assert compare_conditions({"llm": []})["llm"] is None


def test_export_carries_session_timing_and_constraints(store, department):
    rec = StudyRecorder(store, "P01", Condition.LLM, department=department)
    rec.enter(Phase.ENTRY, at=0.0)
    ids = rec.record_parse("x", [unavail()], at=1.0)
    rec.accept(ids[0], unavail(), at=2.0)
    rec.complete(at=10.0)

    export = store.export(rec.session_id)
    assert export["session"]["participant"] == "P01"
    assert export["timing"]["total_seconds"] == 10.0
    assert len(export["constraints"]) == 1
    assert len(export["events"]) > 3
    assert len(store.export_all()) == 1
