"""SQLite event store for the study (build plan Phase 5.4, §2).

Event-sourced on purpose. Durations are derived from timestamped phase spans
instead of a running stopwatch, which means the raw log can be re-analysed
after the fact under a different idle rule, and a participant who navigates
back and forth is still measured correctly.

Three tables:

* ``sessions`` — one row per participant per condition.
* ``events`` — every action, timestamped, with its phase.
* ``constraints`` — what the parser proposed, what the participant decided,
  and what the set finally contained. Phase 6C compares the final set against
  gold to get residual error after verification, so the before and after both
  have to survive.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Sequence

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id            TEXT PRIMARY KEY,
    participant   TEXT NOT NULL,
    condition     TEXT NOT NULL,
    task          TEXT,
    started_at    REAL NOT NULL,
    completed_at  REAL,
    department    TEXT,
    model         TEXT,
    prompt_hash   TEXT,
    order_index   INTEGER,
    notes         TEXT
);

CREATE TABLE IF NOT EXISTS events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT NOT NULL REFERENCES sessions(id),
    at            REAL NOT NULL,
    phase         TEXT NOT NULL,
    kind          TEXT NOT NULL,
    constraint_id TEXT,
    payload       TEXT
);

CREATE TABLE IF NOT EXISTS constraints (
    id            TEXT PRIMARY KEY,
    session_id    TEXT NOT NULL REFERENCES sessions(id),
    ordinal       INTEGER NOT NULL,
    source_text   TEXT NOT NULL,
    proposed      TEXT NOT NULL,
    final         TEXT,
    status        TEXT NOT NULL,
    strength_before TEXT,
    strength_after  TEXT,
    edit_count    INTEGER NOT NULL DEFAULT 0,
    flags         TEXT,
    created_at    REAL NOT NULL,
    decided_at    REAL
);

CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id, at);
CREATE INDEX IF NOT EXISTS idx_constraints_session ON constraints(session_id, ordinal);
"""


class Phase(str, Enum):
    """Phases whose durations are reported separately (Phase 5.4).

    ENTRY and VERIFICATION are kept apart because the standing objection to
    any time-saving claim is that time saved on entry reappears as
    verification time. A single total cannot answer it.
    """

    SETUP = "setup"
    ENTRY = "entry"
    VERIFICATION = "verification"
    RESULT = "result"
    DEBRIEF = "debrief"


class EventKind(str, Enum):
    SESSION_START = "session_start"
    PHASE_ENTER = "phase_enter"
    PHASE_EXIT = "phase_exit"
    TEXT_SUBMIT = "text_submit"
    PARSE_COMPLETE = "parse_complete"
    PARSE_FAILED = "parse_failed"
    CONSTRAINT_ACCEPT = "constraint_accept"
    CONSTRAINT_EDIT = "constraint_edit"
    CONSTRAINT_REJECT = "constraint_reject"
    STRENGTH_TOGGLE = "strength_toggle"
    CLARIFICATION_ANSWER = "clarification_answer"
    FORM_ADD = "form_add"
    FORM_REMOVE = "form_remove"
    SOLVE_START = "solve_start"
    SOLVE_COMPLETE = "solve_complete"
    SESSION_COMPLETE = "session_complete"
    NOTE = "note"


class Decision(str, Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    EDITED = "edited"
    REJECTED = "rejected"


class Condition(str, Enum):
    """Study arms (build plan Phase 6C)."""

    FORM = "form"  # Condition 1: structured form baseline
    LLM = "llm"  # Condition 2: LLM-assisted entry
    LLM_NO_VERIFY = "llm_no_verify"  # Condition 3 (stretch): no review step


@dataclass(frozen=True)
class Event:
    at: float
    phase: str
    kind: str
    constraint_id: str | None = None
    payload: dict[str, Any] | None = None


def _now() -> float:
    return time.time()


#: Where the study log lives when no path is given. Set `TIMETABLER_DB` to
#: move it — required on any host whose default filesystem is ephemeral, where
#: a restart or redeploy would otherwise discard the sessions collected so far.
DEFAULT_DB_PATH = "logs/study.sqlite"


def default_db_path() -> str:
    return os.environ.get("TIMETABLER_DB") or DEFAULT_DB_PATH


class StudyStore:
    """Thin SQLite wrapper. One file per study, or `:memory:` for tests."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = str(path if path is not None else default_db_path())
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        # Streamlit reruns the script on a different thread each time, and a
        # sqlite3 connection is bound to its creating thread by default. The
        # store is built once and used from every rerun, so the check is
        # disabled and a lock serialises access instead. Without this the
        # first logged action of a real session raises.
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self.conn.executescript(SCHEMA)
            self.conn.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
            self.conn.commit()

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    def _write(self, sql: str, params: Sequence[Any] = ()) -> None:
        with self._lock:
            self.conn.execute(sql, params)
            self.conn.commit()

    def _read(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self.conn.execute(sql, params).fetchall()

    def __enter__(self) -> "StudyStore":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- sessions ---------------------------------------------------------

    def create_session(
        self,
        participant: str,
        condition: str,
        *,
        task: str | None = None,
        department: str | None = None,
        model: str | None = None,
        prompt_hash: str | None = None,
        order_index: int | None = None,
        session_id: str | None = None,
        at: float | None = None,
    ) -> str:
        sid = session_id or uuid.uuid4().hex
        started = at if at is not None else _now()
        self._write(
            "INSERT INTO sessions(id, participant, condition, task, started_at, "
            "department, model, prompt_hash, order_index) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (sid, participant, condition, task, started, department, model,
             prompt_hash, order_index),
        )
        self.log(sid, Phase.SETUP, EventKind.SESSION_START, at=started)
        return sid

    def complete_session(self, session_id: str, at: float | None = None) -> None:
        when = at if at is not None else _now()
        self._write(
            "UPDATE sessions SET completed_at = ? WHERE id = ?", (when, session_id)
        )
        self.log(session_id, Phase.DEBRIEF, EventKind.SESSION_COMPLETE, at=when)

    def session(self, session_id: str) -> dict[str, Any] | None:
        rows = self._read("SELECT * FROM sessions WHERE id = ?", (session_id,))
        return dict(rows[0]) if rows else None

    def sessions(
        self, participant: str | None = None, condition: str | None = None
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM sessions"
        clauses: list[str] = []
        params: list[Any] = []
        if participant is not None:
            clauses.append("participant = ?")
            params.append(participant)
        if condition is not None:
            clauses.append("condition = ?")
            params.append(condition)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY started_at"
        return [dict(r) for r in self._read(sql, params)]

    # -- events -----------------------------------------------------------

    def log(
        self,
        session_id: str,
        phase: str,
        kind: str,
        *,
        constraint_id: str | None = None,
        payload: dict[str, Any] | None = None,
        at: float | None = None,
    ) -> float:
        when = at if at is not None else _now()
        self._write(
            "INSERT INTO events(session_id, at, phase, kind, constraint_id, payload) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, when, str(getattr(phase, "value", phase)),
             str(getattr(kind, "value", kind)), constraint_id,
             json.dumps(payload, ensure_ascii=False) if payload else None),
        )
        return when

    def events(self, session_id: str) -> list[Event]:
        rows = self._read(
            "SELECT at, phase, kind, constraint_id, payload FROM events "
            "WHERE session_id = ? ORDER BY at, id",
            (session_id,),
        )
        return [
            Event(
                at=r["at"],
                phase=r["phase"],
                kind=r["kind"],
                constraint_id=r["constraint_id"],
                payload=json.loads(r["payload"]) if r["payload"] else None,
            )
            for r in rows
        ]

    # -- constraints ------------------------------------------------------

    def add_constraint(
        self,
        session_id: str,
        ordinal: int,
        source_text: str,
        proposed: dict[str, Any],
        *,
        flags: Sequence[dict[str, Any]] = (),
        constraint_id: str | None = None,
        at: float | None = None,
    ) -> str:
        cid = constraint_id or uuid.uuid4().hex
        self._write(
            "INSERT INTO constraints(id, session_id, ordinal, source_text, "
            "proposed, status, strength_before, edit_count, flags, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?)",
            (cid, session_id, ordinal, source_text,
             json.dumps(proposed, ensure_ascii=False), Decision.PENDING.value,
             proposed.get("strength"),
             json.dumps(list(flags), ensure_ascii=False) if flags else None,
             at if at is not None else _now()),
        )
        return cid

    def decide(
        self,
        constraint_id: str,
        status: str,
        *,
        final: dict[str, Any] | None = None,
        increment_edits: bool = False,
        at: float | None = None,
    ) -> None:
        when = at if at is not None else _now()
        self._write(
            "UPDATE constraints SET status = ?, final = ?, strength_after = ?, "
            "decided_at = ?, edit_count = edit_count + ? WHERE id = ?",
            (str(getattr(status, "value", status)),
             json.dumps(final, ensure_ascii=False) if final else None,
             (final or {}).get("strength"),
             when, 1 if increment_edits else 0, constraint_id),
        )

    def constraints(self, session_id: str) -> list[dict[str, Any]]:
        rows = self._read(
            "SELECT * FROM constraints WHERE session_id = ? ORDER BY ordinal",
            (session_id,),
        )
        out: list[dict[str, Any]] = []
        for row in rows:
            record = dict(row)
            record["proposed"] = json.loads(record["proposed"])
            record["final"] = json.loads(record["final"]) if record["final"] else None
            record["flags"] = json.loads(record["flags"]) if record["flags"] else []
            out.append(record)
        return out

    def accepted_constraints(self, session_id: str) -> list[dict[str, Any]]:
        """The set as the participant left it, for comparison against gold."""
        return [
            (c["final"] or c["proposed"])
            for c in self.constraints(session_id)
            if c["status"] in (Decision.ACCEPTED.value, Decision.EDITED.value)
        ]

    # -- export -----------------------------------------------------------

    def export(self, session_id: str) -> dict[str, Any]:
        from .timing import timing_report

        return {
            "session": self.session(session_id),
            "timing": timing_report(self.events(session_id)),
            "constraints": self.constraints(session_id),
            "events": [
                {
                    "at": e.at, "phase": e.phase, "kind": e.kind,
                    "constraint_id": e.constraint_id, "payload": e.payload,
                }
                for e in self.events(session_id)
            ],
        }

    def export_all(self) -> list[dict[str, Any]]:
        return [self.export(s["id"]) for s in self.sessions()]


__all__ = [
    "DEFAULT_DB_PATH",
    "SCHEMA",
    "SCHEMA_VERSION",
    "Condition",
    "Decision",
    "Event",
    "EventKind",
    "Phase",
    "StudyStore",
    "default_db_path",
]
