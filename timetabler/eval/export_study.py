"""Export a study log to tidy CSV (build plan Phase 6C).

    python -m timetabler.eval.export_study --out exports/

Writes three tables:

* ``sessions.csv`` — one row per participant-condition. Entry time,
  verification time and their total are separate columns, because the whole
  point of the split is that a single total cannot answer the objection that
  time saved on entry reappears as verification time.
* ``constraints.csv`` — one row per constraint, with what the parser proposed,
  what the participant decided, and what the set finally held. This is the
  raw material for the residual-error measure.
* ``events.csv`` — the full timestamped log, for any re-analysis the two
  tables above do not support.

Everything written passes through the credential scrubber and is then checked,
so an export can be shared or archived without carrying a key.

**The residual-error measure needs something this repository does not yet
define.** Participants write their own sentences, so the Phase 6A labelled set
is not the comparison: it labels a different corpus. Residual error is the gap
between the set a participant finally accepted and the set the *task* asked
them to express, which means each task needs a key of intended constraints.
The build plan does not specify one. Until it does, supply correctness
yourself with ``--correct``, a CSV of ``session_id,correct_constraints``, and
this tool fills the throughput column from it.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from ..app.credentials import assert_no_credentials, redact
from ..app.store import Decision, EventKind, StudyStore, default_db_path
from ..app.timing import build_report, decision_counts, throughput

SESSION_COLUMNS = [
    "session_id", "participant", "condition", "task", "order_index",
    "started_at", "completed_at", "completed",
    "model", "prompt_hash", "department",
    "entry_seconds", "verification_seconds", "total_seconds",
    "verification_share", "capped_total_seconds", "idle_spans",
    "unclosed_phases",
    "constraints_proposed", "constraints_kept",
    "accepted", "edited", "rejected", "pending", "edit_events",
    "text_submits", "parses_ok", "parses_failed", "strength_toggles",
    "clarifications", "form_adds", "form_removes", "solves",
    "correct_constraints", "throughput_per_minute",
]

CONSTRAINT_COLUMNS = [
    "session_id", "participant", "condition", "ordinal", "constraint_id",
    "proposed_type", "final_type", "type_changed",
    "status", "edit_count", "strength_before", "strength_after",
    "strength_changed",
    "n_flags", "n_block_flags", "n_review_flags", "flag_codes",
    "source_text", "proposed_json", "final_json",
]

EVENT_COLUMNS = [
    "session_id", "participant", "condition", "at", "phase", "kind",
    "constraint_id", "payload_json",
]

#: Event kinds counted into their own session columns.
_COUNT_MAP = {
    "text_submits": EventKind.TEXT_SUBMIT.value,
    "parses_ok": EventKind.PARSE_COMPLETE.value,
    "parses_failed": EventKind.PARSE_FAILED.value,
    "strength_toggles": EventKind.STRENGTH_TOGGLE.value,
    "clarifications": EventKind.CLARIFICATION_ANSWER.value,
    "form_adds": EventKind.FORM_ADD.value,
    "form_removes": EventKind.FORM_REMOVE.value,
    "solves": EventKind.SOLVE_COMPLETE.value,
}


def load_correctness(path: str | None) -> dict[str, int]:
    """Read `session_id,correct_constraints` supplied by the researcher."""
    if not path:
        return {}
    out: dict[str, int] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row.get("session_id") or "").strip()
            value = (row.get("correct_constraints") or "").strip()
            if key and value:
                out[key] = int(value)
    return out


def session_rows(
    store: StudyStore, correctness: dict[str, int], idle_threshold: float
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for session in store.sessions():
        sid = session["id"]
        timing = build_report(store.events(sid), idle_threshold=idle_threshold)
        records = store.constraints(sid)
        decisions = decision_counts(records)
        kinds = Counter(e.kind for e in store.events(sid))

        correct = correctness.get(sid)
        row: dict[str, Any] = {
            "session_id": sid,
            "participant": session["participant"],
            "condition": session["condition"],
            "task": session["task"],
            "order_index": session["order_index"],
            "started_at": session["started_at"],
            "completed_at": session["completed_at"],
            "completed": session["completed_at"] is not None,
            "model": session["model"],
            "prompt_hash": session["prompt_hash"],
            "department": session["department"],
            "entry_seconds": round(timing.entry_seconds, 3),
            "verification_seconds": round(timing.verification_seconds, 3),
            "total_seconds": round(timing.total_seconds, 3),
            "verification_share": (
                round(timing.verification_share, 4)
                if timing.verification_share is not None else ""
            ),
            "capped_total_seconds": round(timing.capped_total_seconds, 3),
            "idle_spans": timing.idle_spans,
            "unclosed_phases": ";".join(timing.unclosed_phases),
            "constraints_proposed": len(records),
            "constraints_kept": len(store.accepted_constraints(sid)),
            "accepted": decisions.get(Decision.ACCEPTED.value, 0),
            "edited": decisions.get(Decision.EDITED.value, 0),
            "rejected": decisions.get(Decision.REJECTED.value, 0),
            "pending": decisions.get(Decision.PENDING.value, 0),
            "edit_events": decisions.get("edit_events", 0),
            "correct_constraints": correct if correct is not None else "",
            "throughput_per_minute": (
                round(throughput(correct, timing.total_seconds) or 0.0, 4)
                if correct is not None and timing.total_seconds > 0 else ""
            ),
        }
        for column, kind in _COUNT_MAP.items():
            row[column] = kinds.get(kind, 0)
        rows.append(row)
    return rows


def constraint_rows(store: StudyStore) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for session in store.sessions():
        sid = session["id"]
        for record in store.constraints(sid):
            proposed = record["proposed"] or {}
            final = record["final"] or {}
            flags = record["flags"] or []
            rows.append({
                "session_id": sid,
                "participant": session["participant"],
                "condition": session["condition"],
                "ordinal": record["ordinal"],
                "constraint_id": record["id"],
                "proposed_type": proposed.get("type", ""),
                "final_type": final.get("type", ""),
                "type_changed": bool(final) and (
                    final.get("type") != proposed.get("type")
                ),
                "status": record["status"],
                "edit_count": record["edit_count"],
                "strength_before": record["strength_before"] or "",
                "strength_after": record["strength_after"] or "",
                "strength_changed": bool(record["strength_after"]) and (
                    record["strength_after"] != record["strength_before"]
                ),
                "n_flags": len(flags),
                "n_block_flags": sum(
                    1 for f in flags if f.get("severity") == "block"
                ),
                "n_review_flags": sum(
                    1 for f in flags if f.get("severity") == "review"
                ),
                "flag_codes": ";".join(sorted({f.get("code", "") for f in flags})),
                "source_text": record["source_text"],
                "proposed_json": json.dumps(proposed, ensure_ascii=False),
                "final_json": json.dumps(final, ensure_ascii=False) if final else "",
            })
    return rows


def event_rows(store: StudyStore) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for session in store.sessions():
        sid = session["id"]
        for event in store.events(sid):
            rows.append({
                "session_id": sid,
                "participant": session["participant"],
                "condition": session["condition"],
                "at": event.at,
                "phase": event.phase,
                "kind": event.kind,
                "constraint_id": event.constraint_id or "",
                "payload_json": (
                    json.dumps(event.payload, ensure_ascii=False)
                    if event.payload else ""
                ),
            })
    return rows


def write_csv(path: Path, columns: list[str], rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(redact(row))
            count += 1
    assert_no_credentials(path.read_text(encoding="utf-8"), where=str(path))
    return count


def is_analysable(row: dict[str, Any]) -> bool:
    """Whether a session's timing can be used.

    A session that was never finished has an unclosed phase, and an unclosed
    phase is closed at the last event seen. Leave the app open overnight and
    that span runs to tens of thousands of seconds, so one abandoned session
    can swamp a condition's mean. Every row still goes into the CSV; only the
    summary and any mean exclude these, and the count is reported.
    """
    return bool(row["completed"]) and not row["unclosed_phases"]


def summarise(rows: list[dict[str, Any]]) -> str:
    """A short per-condition read-out, so a mistake is visible immediately."""
    if not rows:
        return "  no sessions recorded"

    usable = [r for r in rows if is_analysable(r)]
    excluded = [r for r in rows if not is_analysable(r)]

    lines: list[str] = []
    if usable:
        by_condition: dict[str, list[dict[str, Any]]] = {}
        for row in usable:
            by_condition.setdefault(row["condition"], []).append(row)

        lines += [
            f"  {'condition':<16}{'n':>4}{'entry':>10}{'verify':>10}"
            f"{'total':>10}{'verify%':>9}",
            f"  {'-' * 16}{'-' * 4}{'-' * 10}{'-' * 10}{'-' * 10}{'-' * 9}",
        ]
        for condition, items in sorted(by_condition.items()):
            n = len(items)
            entry = sum(r["entry_seconds"] for r in items) / n
            verify = sum(r["verification_seconds"] for r in items) / n
            total = sum(r["total_seconds"] for r in items) / n
            share = (verify / total * 100) if total else 0.0
            lines.append(
                f"  {condition:<16}{n:>4}{entry:>10.1f}{verify:>10.1f}"
                f"{total:>10.1f}{share:>8.0f}%"
            )
        lines.append("  seconds, complete sessions only")
    else:
        lines.append("  no complete sessions to summarise")

    if excluded:
        lines.append(
            f"\n  {len(excluded)} session(s) excluded from the means "
            f"(never finished, or a phase left open):"
        )
        for row in excluded:
            lines.append(
                f"    {row['participant']} / {row['condition']} "
                f"({row['session_id'][:8]}) - "
                + ("unclosed: " + row["unclosed_phases"]
                   if row["unclosed_phases"] else "not completed")
            )
        lines.append("    Their rows are still in sessions.csv. Decide whether "
                     "to drop or repair them before analysis.")

    idle = sum(r["idle_spans"] for r in usable)
    if idle:
        lines.append(
            f"\n  {idle} span(s) over the idle threshold among complete "
            f"sessions. Compare total_seconds with capped_total_seconds."
        )
    if not any(r["correct_constraints"] != "" for r in rows):
        lines.append(
            "\n  correct_constraints is empty, so throughput is not computed. "
            "Supply --correct once residual error is scored."
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Export a study log to tidy CSV for analysis."
    )
    ap.add_argument("--db", default=None,
                    help=f"Study log (default: $TIMETABLER_DB or "
                         f"{default_db_path()})")
    ap.add_argument("--out", default="exports",
                    help="Directory for the CSV files (default: exports)")
    ap.add_argument("--correct", default=None,
                    help="CSV of session_id,correct_constraints to fill the "
                         "throughput column.")
    ap.add_argument("--idle-threshold", type=float, default=180.0,
                    help="Seconds above which a phase span is also reported "
                         "capped (default: 180).")
    ap.add_argument("--json", action="store_true",
                    help="Also write the full nested export as JSON.")
    args = ap.parse_args(argv)

    db_path = args.db or default_db_path()
    if db_path != ":memory:" and not Path(db_path).exists():
        sys.exit(
            f"\nNo study log at {db_path}. Set --db, or export TIMETABLER_DB "
            f"to where the app wrote it.\n"
        )

    out = Path(args.out)
    correctness = load_correctness(args.correct)

    with StudyStore(db_path) as store:
        sessions = session_rows(store, correctness, args.idle_threshold)
        n_sessions = write_csv(out / "sessions.csv", SESSION_COLUMNS, sessions)
        n_constraints = write_csv(
            out / "constraints.csv", CONSTRAINT_COLUMNS, constraint_rows(store)
        )
        n_events = write_csv(out / "events.csv", EVENT_COLUMNS, event_rows(store))

        if args.json:
            payload = redact(store.export_all())
            assert_no_credentials(payload, where="json export")
            (out / "study.json").write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )

    print(f"\nExported from {db_path}")
    print(f"  {out / 'sessions.csv'}      {n_sessions} rows")
    print(f"  {out / 'constraints.csv'}   {n_constraints} rows")
    print(f"  {out / 'events.csv'}        {n_events} rows")
    if args.json:
        print(f"  {out / 'study.json'}")
    print()
    print(summarise(sessions))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
