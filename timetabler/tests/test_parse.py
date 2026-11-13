"""Parse orchestration: extraction, validation, the single retry (Phase 4.2)."""

from __future__ import annotations

import json

import pytest

from timetabler.data import load_department
from timetabler.parser.client import Completion, RunMetadata, StubClient
from timetabler.parser.parse import (
    ConstraintParser,
    JSONExtractionError,
    coerce_payload,
    extract_json,
)

GOOD = {"constraints": [{
    "type": "lecturer_unavailable", "lecturer": "Lee", "days": ["Fri"],
    "periods": None, "strength": "hard", "weight": None,
    "source_text": "I can't teach Fridays",
}]}


@pytest.fixture(scope="module")
def department():
    return load_department()


class SequenceClient:
    """Returns prepared replies in order, ignoring the prompt.

    Needed for the retry test: the retry message embeds the previous reply and
    the validation error, so it cannot be keyed by text.
    """

    backend = "sequence"
    model = "sequence"
    temperature = 0.0

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls = 0

    def complete(self, system: str, user: str, *, prompt_hash: str = "") -> Completion:
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        self.last_user = user
        return Completion(
            text=reply,
            metadata=RunMetadata("sequence", 0.0, prompt_hash, "now", "sequence"),
        )


# -- JSON extraction -----------------------------------------------------


def test_extract_plain_object():
    assert extract_json('{"constraints": []}') == {"constraints": []}


def test_extract_from_code_fence():
    text = 'Sure:\n```json\n{"constraints": []}\n```'
    assert extract_json(text) == {"constraints": []}


def test_extract_from_unlabelled_fence():
    assert extract_json('```\n{"constraints": []}\n```') == {"constraints": []}


def test_extract_ignores_surrounding_prose():
    text = 'Here is the result: {"constraints": [], "note": "x"} Hope that helps.'
    assert extract_json(text) == {"constraints": [], "note": "x"}


def test_extract_handles_braces_inside_strings():
    text = '{"constraints": [], "note": "a } brace"}'
    assert extract_json(text)["note"] == "a } brace"


def test_extract_rejects_empty():
    with pytest.raises(JSONExtractionError, match="empty response"):
        extract_json("   ")


def test_extract_rejects_prose_only():
    with pytest.raises(JSONExtractionError, match="no JSON object"):
        extract_json("I am not able to answer that.")


# -- payload shapes ------------------------------------------------------


def test_bare_list_accepted():
    assert coerce_payload([{"type": "x"}]) == {"constraints": [{"type": "x"}]}


def test_single_constraint_object_accepted():
    assert coerce_payload({"type": "unsupported"})["constraints"][0]["type"] == (
        "unsupported"
    )


def test_unrecognised_shape_rejected():
    with pytest.raises(JSONExtractionError):
        coerce_payload("a string")


# -- the pipeline --------------------------------------------------------


def test_happy_path(department):
    client = StubClient().add("I can't teach Fridays", GOOD)
    result = ConstraintParser(department, client).parse("I can't teach Fridays")
    assert result.ok and not result.retried
    assert len(result.constraints) == 1
    assert result.constraints[0].resolved_ids["lecturer"] == "L001"
    assert len(result.ready_for_solver) == 1


def test_retry_once_recovers_from_invalid_output(department):
    bad = json.dumps({"constraints": [{
        "type": "lecturer_unavailable", "lecturer": "Lee", "days": ["Friday"],
        "strength": "hard", "source_text": "I can't teach Fridays",
    }]})
    client = SequenceClient([bad, json.dumps(GOOD)])
    result = ConstraintParser(department, client).parse("I can't teach Fridays")
    assert client.calls == 2
    assert result.ok and result.retried
    assert result.attempts[0].error and "unknown day" in result.attempts[0].error
    assert result.attempts[1].error is None


def test_retry_message_carries_the_problem(department):
    bad = json.dumps({"constraints": [{"type": "nonexistent_type"}]})
    client = SequenceClient([bad, json.dumps(GOOD)])
    ConstraintParser(department, client).parse("I can't teach Fridays")
    assert "Problem:" in client.last_user
    assert "JSON only" in client.last_user


def test_gives_up_after_one_retry(department):
    bad = json.dumps({"constraints": [{"type": "nonexistent_type"}]})
    client = SequenceClient([bad, bad, json.dumps(GOOD)])
    result = ConstraintParser(department, client).parse("x")
    assert client.calls == 2, "exactly one retry, per Phase 4.2"
    assert not result.ok
    assert len(result.attempts) == 2


def test_prose_only_reply_is_a_parse_failure(department):
    client = SequenceClient(["I cannot help with that."])
    result = ConstraintParser(department, client).parse("x")
    assert not result.ok
    assert "no JSON object" in result.error


def test_hallucinated_entity_blocks_the_solver_path(department):
    payload = {"constraints": [{
        "type": "lecturer_unavailable", "lecturer": "Zhang", "days": ["Mon"],
        "periods": None, "strength": "hard", "weight": None,
        "source_text": "Zhang cannot teach Monday",
    }]}
    client = StubClient().add("Zhang cannot teach Monday", payload)
    result = ConstraintParser(department, client).parse("Zhang cannot teach Monday")
    assert result.ok, "a resolvable-looking reply still parses"
    assert result.ready_for_solver == [], "but it must not reach the solver"
    assert "unknown_entity" in result.flag_codes()


def test_several_constraints_from_one_sentence(department):
    payload = {"constraints": [
        {"type": "lecturer_unavailable", "lecturer": "Nair", "days": ["Thu"],
         "periods": None, "strength": "hard", "weight": None,
         "source_text": "Nair can't do Thursdays"},
        {"type": "course_requires_room", "course": "Machine Learning",
         "room": "Lecture Theatre 1", "strength": "hard", "weight": None,
         "source_text": "ML must be in Lecture Theatre 1"},
    ]}
    text = "Nair can't do Thursdays and ML must be in Lecture Theatre 1"
    client = StubClient().add(text, payload)
    result = ConstraintParser(department, client).parse(text)
    assert len(result.constraints) == 2
    assert len(result.ready_for_solver) == 2


def test_stub_miss_is_an_error_by_default():
    client = StubClient()
    with pytest.raises(KeyError, match="no response"):
        client.complete("sys", "unregistered")


def test_run_metadata_is_recorded(department):
    client = StubClient().add("I can't teach Fridays", GOOD)
    parser = ConstraintParser(department, client)
    result = parser.parse("I can't teach Fridays")
    meta = result.attempts[0].metadata
    assert meta.prompt_hash == parser.prompt_hash
    assert meta.timestamp and meta.backend == "stub"


def test_prompt_hash_tracks_the_prompt(department):
    from timetabler.parser.prompt import FEW_SHOT

    full = ConstraintParser(department, StubClient()).prompt_hash
    fewer = ConstraintParser(department, StubClient(), examples=FEW_SHOT[:4]).prompt_hash
    assert full != fewer, "the sensitivity sweep must be distinguishable in the log"
