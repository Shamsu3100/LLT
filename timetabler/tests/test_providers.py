"""Multi-provider clients and the factory (build plan Phase 4.2, 6A).

No network call is made. These tests pin the spec grammar, the credential
handling, and — most importantly — that every provider is driven through the
same instrument, since an asymmetric one would measure the harness instead of
the model.
"""

from __future__ import annotations

import json

import pytest

from timetabler.data import load_department
from timetabler.parser.base_client import PRICING, register_pricing
from timetabler.parser.client import (
    ENDPOINTS,
    PROVIDERS,
    AnthropicClient,
    ClientError,
    Completion,
    LLMClient,
    OpenAICompatibleClient,
    RunMetadata,
    StubClient,
    describe_providers,
    estimate_cost,
    make_client,
    parse_spec,
)
from timetabler.parser.parse import ConstraintParser

GOOD = {"constraints": [{
    "type": "lecturer_unavailable", "lecturer": "Lee", "days": ["Fri"],
    "periods": None, "strength": "hard", "weight": None,
    "source_text": "I can't teach Fridays",
}]}

ALL_KEYS = [
    "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE",
    *(e.env_var for e in ENDPOINTS.values()),
]


@pytest.fixture
def no_credentials(monkeypatch):
    for var in ALL_KEYS:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(scope="module")
def department():
    return load_department()


# -- spec grammar --------------------------------------------------------


@pytest.mark.parametrize("spec,expected", [
    ("anthropic:claude-sonnet-5-5", ("anthropic", "claude-sonnet-5-5")),
    ("openai:some-model", ("openai", "some-model")),
    ("deepseek:some-model", ("deepseek", "some-model")),
    ("anthropic", ("anthropic", "")),
    ("  Anthropic  ", ("anthropic", "")),
    ("stub", ("stub", "")),
])
def test_parse_spec(spec, expected):
    assert parse_spec(spec) == expected


def test_only_the_first_colon_separates():
    """Model ids contain colons: Ollama tags, OpenRouter vendor paths."""
    assert parse_spec("ollama:llama3.1:8b") == ("ollama", "llama3.1:8b")
    assert parse_spec("openrouter:vendor/model:free") == (
        "openrouter", "vendor/model:free"
    )


@pytest.mark.parametrize("spec", ["", "   ", "gpt4", "claude", "nope:model"])
def test_bad_spec_is_rejected_with_the_known_list(spec):
    with pytest.raises(ClientError):
        parse_spec(spec)


def test_every_provider_is_listed_in_the_help():
    text = describe_providers()
    for provider in PROVIDERS:
        assert provider in text


def test_every_endpoint_names_a_credential_variable():
    for name, endpoint in ENDPOINTS.items():
        assert endpoint.env_var.endswith("_API_KEY"), name
        assert endpoint.base_url is None or endpoint.base_url.startswith("http")


# -- factory -------------------------------------------------------------


def test_factory_builds_a_stub():
    client = make_client("stub")
    assert isinstance(client, LLMClient)
    assert json.loads(client.complete("s", "u").text) == {"constraints": []}


def test_factory_builds_anthropic(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-a-real-credential")
    client = make_client("anthropic")
    assert isinstance(client, AnthropicClient)
    assert client.model == "claude-sonnet-5-5", "the plan's §2 default"


def test_factory_builds_an_openai_compatible_provider(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "placeholder")
    client = make_client("deepseek:some-model")
    assert isinstance(client, OpenAICompatibleClient)
    assert client.provider == "deepseek"
    assert str(client._client.base_url).startswith("https://api.deepseek.com")


def test_missing_credential_names_the_variable_to_set(no_credentials):
    with pytest.raises(ClientError, match="OPENAI_API_KEY"):
        make_client("openai:some-model")
    with pytest.raises(ClientError, match="ANTHROPIC_API_KEY"):
        make_client("anthropic")


def test_local_providers_need_no_credential(no_credentials):
    """A local server has no key, but the SDK insists on a non-empty string."""
    for spec in ("ollama:some-model", "local:some-model"):
        assert isinstance(make_client(spec), OpenAICompatibleClient)


def test_model_id_is_required_for_non_anthropic_providers(monkeypatch):
    """Ids change and vary by account, so none is guessed."""
    monkeypatch.setenv("OPENAI_API_KEY", "placeholder")
    with pytest.raises(ClientError, match="needs an explicit model id"):
        make_client("openai")


def test_base_url_can_be_overridden(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "placeholder")
    client = make_client("openai:m", base_url="http://127.0.0.1:9999/v1")
    assert "127.0.0.1:9999" in str(client._client.base_url)


# -- instrument parity ---------------------------------------------------


def test_json_mode_is_off_by_default(monkeypatch):
    """The Claude arm does not use it; an asymmetric instrument confounds."""
    monkeypatch.setenv("OPENAI_API_KEY", "placeholder")
    assert make_client("openai:m").json_mode is False


def test_temperature_is_not_sent_by_default(monkeypatch):
    """Current Claude models reject it, so a default would break parity."""
    monkeypatch.setenv("OPENAI_API_KEY", "placeholder")
    assert make_client("openai:m").temperature is None


def test_optional_features_can_be_turned_on_explicitly(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "placeholder")
    client = make_client("openai:m", json_mode=True, temperature=0.0)
    assert client.json_mode and client.temperature == 0.0


def test_anthropic_client_ignores_options_meant_for_others(monkeypatch):
    """One call site serves every provider, so extras must not raise."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-a-real-credential")
    client = make_client("anthropic", json_mode=True, temperature=0.5, base_url="x")
    assert client.temperature is None, "Claude cannot accept a temperature"
    assert client.json_mode is False


def test_every_client_satisfies_the_contract(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-a-real-credential")
    monkeypatch.setenv("OPENAI_API_KEY", "placeholder")
    for spec in ("stub", "anthropic", "openai:m"):
        assert isinstance(make_client(spec), LLMClient), spec


# -- metadata and cost ---------------------------------------------------


def test_metadata_records_the_provider():
    """A result has to say which provider produced it."""
    assert make_client("stub").complete("s", "u").metadata.provider == "stub"


def test_unpriced_model_reports_no_cost_instead_of_guessing():
    assert estimate_cost("some-provider-model-id", 1000, 1000) is None


def test_pricing_can_be_registered_for_any_model():
    register_pricing("test-only-model", 1.0, 2.0)
    try:
        assert estimate_cost("test-only-model", 1_000_000, 1_000_000) == pytest.approx(3.0)
    finally:
        PRICING.pop("test-only-model", None)


def test_only_verifiable_rates_are_shipped():
    """A wrong hardcoded rate would be worse than no estimate."""
    assert all(m.startswith("claude-") for m in PRICING), (
        "non-Anthropic rates must be supplied by the user, not assumed"
    )


# -- stop-reason normalisation ------------------------------------------


def meta(**kwargs):
    base = dict(model="m", prompt_hash="h", timestamp="t", backend="b")
    base.update(kwargs)
    return RunMetadata(**base)


def test_length_and_max_tokens_both_count_as_truncation():
    """Providers spell the same outcome differently."""
    assert Completion("", meta(stop_reason="max_tokens")).truncated
    assert Completion("", meta(stop_reason="length")).truncated
    assert not Completion("", meta(stop_reason="stop")).truncated


def test_content_filter_is_treated_as_a_refusal(department):
    class Refusing:
        backend = provider = "fake"
        model = "m"

        def complete(self, system, user, *, prompt_hash=""):
            return Completion("", meta(
                stop_reason="refusal", refusal_category="content_filter"
            ))

    result = ConstraintParser(department, Refusing()).parse("x")
    assert not result.ok
    assert "declined" in result.error and "content_filter" in result.error
