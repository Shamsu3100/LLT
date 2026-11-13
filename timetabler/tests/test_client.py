"""Live-client behaviour (build plan Phase 4.2).

The API is never called here. These tests pin the handling around it: which
failures are recorded as an item's outcome, which stop reasons must not be
accepted as a result, and how cost is accounted.
"""

from __future__ import annotations

import json

import pytest

from timetabler.data import load_department
from timetabler.parser.client import (
    DEFAULT_EFFORT,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    PRICING,
    AnthropicClient,
    ClientError,
    Completion,
    RunMetadata,
    StubClient,
    estimate_cost,
)
from timetabler.parser.parse import ConstraintParser

GOOD = {"constraints": [{
    "type": "lecturer_unavailable", "lecturer": "Lee", "days": ["Fri"],
    "periods": None, "strength": "hard", "weight": None,
    "source_text": "I can't teach Fridays",
}]}


@pytest.fixture(scope="module")
def department():
    return load_department()


def meta(**kwargs):
    base = dict(
        model="claude-sonnet-5-5", prompt_hash="h", timestamp="t",
        backend="anthropic", effort="medium",
    )
    base.update(kwargs)
    return RunMetadata(**base)


class ScriptedClient:
    """Returns prepared completions, or raises a prepared error."""

    backend = "scripted"
    model = "scripted"
    effort = "medium"

    def __init__(self, items):
        self.items = list(items)
        self.calls = 0

    def complete(self, system, user, *, prompt_hash=""):
        item = self.items[min(self.calls, len(self.items) - 1)]
        self.calls += 1
        if isinstance(item, Exception):
            raise item
        return item


# -- defaults ------------------------------------------------------------

def test_defaults_match_the_build_plan():
    assert DEFAULT_MODEL == "claude-sonnet-5-5", "the plan's §2 choice"
    assert DEFAULT_EFFORT in ("low", "medium", "high", "xhigh", "max")
    assert DEFAULT_MAX_TOKENS >= 8000, "a long submission must not truncate"


def test_no_sampling_parameters_are_accepted():
    """Regression: the SDK removed `temperature` entirely.

    The old client passed `temperature=0.0`, which raises a TypeError before
    the request is even built.
    """
    import inspect

    assert "temperature" not in inspect.signature(AnthropicClient.__init__).parameters


# -- cost ----------------------------------------------------------------

def test_cost_uses_the_published_rates():
    rates = PRICING["claude-sonnet-5-5"]
    assert rates["input"] == 2.00 and rates["output"] == 10.00
    cost = estimate_cost("claude-sonnet-5-5", 1_000_000, 1_000_000)
    assert cost == pytest.approx(12.00)


def test_cost_counts_cache_reads_at_the_cheaper_rate():
    plain = estimate_cost("claude-sonnet-5-5", 1_000_000, 0)
    cached = estimate_cost("claude-sonnet-5-5", 0, 0, 1_000_000)
    assert cached < plain


def test_unknown_model_has_no_cost_estimate():
    assert estimate_cost("some-future-model", 100, 100) is None


def test_retries_are_charged_to_the_item(department):
    """A retry is a second billed call, and the report must show it."""
    bad = Completion("not json", meta(input_tokens=100, output_tokens=10, cost_usd=0.5))
    good = Completion(json.dumps(GOOD), meta(input_tokens=100, output_tokens=20, cost_usd=0.7))
    parser = ConstraintParser(department, ScriptedClient([bad, good]))
    result = parser.parse("I can't teach Fridays")
    assert result.ok and result.retried
    assert result.cost_usd == pytest.approx(1.2)
    assert result.tokens["output"] == 30


# -- stop reasons --------------------------------------------------------

def test_refusal_is_recorded_and_not_retried(department):
    """A refusal will not be fixed by sending the same text again."""
    refused = Completion(
        "", meta(stop_reason="refusal", refusal_category="cyber")
    )
    client = ScriptedClient([refused, Completion(json.dumps(GOOD), meta())])
    result = ConstraintParser(department, client).parse("x")
    assert client.calls == 1, "no retry on a refusal"
    assert not result.ok
    assert "declined" in result.error and "cyber" in result.error


def test_truncated_response_is_never_accepted(department):
    """A partial constraint set must not be read as a complete one."""
    partial = json.dumps(GOOD)[:40]
    client = ScriptedClient([Completion(partial, meta(stop_reason="max_tokens"))])
    result = ConstraintParser(department, client).parse("x")
    assert not result.ok
    assert "truncated" in result.error
    assert result.constraints == []


def test_normal_stop_reason_parses(department):
    client = ScriptedClient([Completion(json.dumps(GOOD), meta(stop_reason="end_turn"))])
    result = ConstraintParser(department, client).parse("I can't teach Fridays")
    assert result.ok and len(result.constraints) == 1


def test_completion_flags():
    assert Completion("", meta(stop_reason="refusal")).refused
    assert Completion("", meta(stop_reason="max_tokens")).truncated
    assert not Completion("", meta(stop_reason="end_turn")).refused


# -- API failures --------------------------------------------------------

def test_api_failure_is_this_items_outcome_not_the_runs(department):
    """A 300-item batch must not die because one call failed."""
    client = ScriptedClient([ClientError("rate limited", retryable=True)])
    result = ConstraintParser(department, client).parse("x")
    assert not result.ok
    assert "api call failed" in result.error
    assert result.attempts == [], "nothing came back to record"


def test_client_error_carries_retryability():
    assert ClientError("overloaded", retryable=True).retryable
    assert not ClientError("bad model").retryable


def test_missing_credential_is_a_clean_error(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(ClientError, match="No credential found"):
        AnthropicClient()


def test_system_prompt_is_marked_cacheable(monkeypatch):
    """The system prompt is identical on every item of a run.

    Resending ~2,700 tokens of schema, register and few-shot examples for each
    of 300 sentences is the bulk of a run's input cost.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-a-real-credential")
    param = AnthropicClient()._system_param("SYSTEM")
    assert isinstance(param, list)
    assert param[0]["cache_control"] == {"type": "ephemeral"}
    assert param[0]["text"] == "SYSTEM"


def test_caching_can_be_turned_off(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-a-real-credential")
    client = AnthropicClient(cache_system_prompt=False)
    assert client._system_param("SYSTEM") == "SYSTEM"


def test_credential_from_environment_is_accepted(monkeypatch):
    """Construction must not call the API."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-a-real-credential")
    client = AnthropicClient(model="claude-sonnet-5-5", effort="low")
    assert client.model == "claude-sonnet-5-5"
    assert client.effort == "low"


# -- stub parity ---------------------------------------------------------

def test_stub_and_live_clients_share_the_contract():
    from timetabler.parser.client import LLMClient

    assert isinstance(StubClient(), LLMClient)


def test_stub_reports_a_normal_stop_reason():
    completion = StubClient(on_miss="empty").complete("sys", "anything")
    assert not completion.refused and not completion.truncated
