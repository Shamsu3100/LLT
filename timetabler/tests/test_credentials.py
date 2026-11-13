"""Credential containment (build plan §5, Privacy).

The guarantee under test: a credential typed into the UI never reaches the
study log, a session export, the environment, or disk. A study whose logs are
released must not carry a key, so these are the tests that have to hold.
"""

from __future__ import annotations

import json
import os

import pytest

from timetabler.app.credentials import (
    REDACTED,
    CredentialStore,
    assert_no_credentials,
    looks_like_credential,
    redact,
)
from timetabler.app.recorder import StudyRecorder
from timetabler.app.store import Condition, Phase, StudyStore
from timetabler.data import load_department
from timetabler.schema import LecturerUnavailable

#: Shapes only. None of these is a real credential.
FAKE_KEYS = [
    "sk-ant-api03-" + "A" * 40,
    "sk-proj-" + "B" * 40,
    "sk-or-v1-" + "c" * 40,
    "gsk_" + "D" * 40,
    "xai-" + "E" * 40,
    "sk-" + "F" * 40,
]


@pytest.fixture
def store():
    with StudyStore(":memory:") as s:
        yield s


@pytest.fixture(scope="module")
def department():
    return load_department()


# -- detection -----------------------------------------------------------


@pytest.mark.parametrize("key", FAKE_KEYS)
def test_every_known_key_shape_is_detected(key):
    assert looks_like_credential(key)


def test_bearer_tokens_are_detected():
    assert looks_like_credential("Authorization: Bearer " + "z" * 40)


@pytest.mark.parametrize("text", [
    "I can't teach Fridays",
    "Dr. Sarah Lee cannot teach on Friday: HARD",
    "CS101L must be scheduled in R201",
    "",
])
def test_ordinary_study_text_is_not_flagged(text):
    assert not looks_like_credential(text)


# -- redaction -----------------------------------------------------------


@pytest.mark.parametrize("key", FAKE_KEYS)
def test_redact_removes_the_key(key):
    out = redact(f"auth failed for {key} at provider")
    assert key not in out and REDACTED in out


def test_redact_walks_nested_structures():
    payload = {
        "error": f"bad key sk-ant-api03-{'A' * 40}",
        "items": [{"detail": f"gsk_{'D' * 40}"}],
    }
    cleaned = redact(payload)
    assert_no_credentials(cleaned, where="cleaned payload")
    assert cleaned["items"][0]["detail"] == REDACTED


def test_redact_leaves_other_values_alone():
    assert redact({"n": 3, "ok": True, "t": "Friday"}) == {
        "n": 3, "ok": True, "t": "Friday"
    }


def test_assert_no_credentials_raises_on_a_leak():
    with pytest.raises(AssertionError, match="credential-shaped"):
        assert_no_credentials({"note": "sk-ant-api03-" + "A" * 40})


def test_assert_no_credentials_passes_on_clean_text():
    assert_no_credentials({"note": "Dr Lee cannot teach Friday"})


# -- the store -----------------------------------------------------------


def test_credential_is_held_and_reported_without_its_value():
    creds = CredentialStore()
    key = FAKE_KEYS[0]
    creds.set("anthropic", key)
    described = creds.describe("anthropic")
    assert creds.has("anthropic")
    assert key not in described
    assert "set in this session" in described


def test_unset_provider_reports_not_set():
    assert CredentialStore().describe("openai") == "not set"


def test_credential_only_leaves_as_client_options():
    creds = CredentialStore()
    creds.set("openai", FAKE_KEYS[1])
    assert creds.options_for("openai") == {"api_key": FAKE_KEYS[1]}
    assert creds.options_for("deepseek") == {}, "no key, no override"


def test_blank_input_clears_rather_than_storing_empty():
    creds = CredentialStore()
    creds.set("groq", FAKE_KEYS[3])
    creds.set("groq", "   ")
    assert not creds.has("groq")


def test_forget_removes_one_provider():
    creds = CredentialStore()
    creds.set("anthropic", FAKE_KEYS[0])
    creds.set("openai", FAKE_KEYS[1])
    creds.clear("anthropic")
    assert creds.providers() == ["openai"]


def test_store_never_touches_the_environment():
    """A key in os.environ can reach a subprocess or a crash dump."""
    before = dict(os.environ)
    creds = CredentialStore()
    creds.set("anthropic", FAKE_KEYS[0])
    creds.options_for("anthropic")
    assert dict(os.environ) == before


def test_store_exposes_no_bulk_reader():
    """There must be nowhere for a credential to go but a constructor."""
    creds = CredentialStore()
    for forbidden in ("get_all", "save", "load", "dump", "to_json"):
        assert not hasattr(creds, forbidden), forbidden


# -- containment in the study log ---------------------------------------


def test_session_export_carries_no_credential(store, department):
    """The headline guarantee, end to end."""
    recorder = StudyRecorder(store, "P01", Condition.LLM, department=department)
    recorder.enter(Phase.ENTRY, at=0.0)
    constraint = LecturerUnavailable(
        lecturer="Lee", days=["Fri"], strength="hard",
        source_text="I can't teach Fridays",
    )
    ids = recorder.record_parse("I can't teach Fridays", [constraint], at=1.0)
    recorder.accept(ids[0], constraint, at=2.0)
    recorder.complete(at=10.0)

    export = store.export(recorder.session_id)
    assert_no_credentials(export, where="session export")
    assert_no_credentials(json.dumps(store.export_all(), default=str), where="full export")


def test_a_credential_in_a_note_is_caught_by_the_guard(store, department):
    """If anything ever does carry a key through, the guard must fail loudly.

    This is the negative control: it proves the assertion is not vacuous.
    """
    recorder = StudyRecorder(store, "P01", Condition.LLM, department=department)
    recorder.note("pasted key " + FAKE_KEYS[0], at=1.0)
    with pytest.raises(AssertionError):
        assert_no_credentials(store.export(recorder.session_id), where="export")


def test_sqlite_file_on_disk_carries_no_credential(tmp_path, department):
    path = tmp_path / "study.sqlite"
    with StudyStore(path) as disk_store:
        recorder = StudyRecorder(
            disk_store, "P01", Condition.LLM, department=department
        )
        recorder.enter(Phase.ENTRY, at=0.0)
        recorder.record_text_submission("I can't teach Fridays", at=1.0)
        recorder.complete(at=5.0)

    raw = path.read_bytes().decode("utf-8", errors="ignore")
    assert_no_credentials(raw, where="sqlite file")


def test_recorder_logs_text_length_not_text(store, department):
    """Submissions are logged as counts, so staff prose stays out of events."""
    recorder = StudyRecorder(store, "P01", Condition.LLM, department=department)
    recorder.enter(Phase.ENTRY, at=0.0)
    recorder.record_text_submission("secret sentence about Dr Lee", at=1.0)
    events = [e for e in store.events(recorder.session_id) if e.payload]
    submissions = [e for e in events if e.kind == "text_submit"]
    assert submissions[0].payload == {"characters": 28, "words": 5}
    assert "secret sentence" not in json.dumps(
        [e.payload for e in events], default=str
    )
