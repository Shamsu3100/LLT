"""Client surface and provider factory (build plan Phase 4.2, 6A).

Public entry point for every backend. The shared contract lives in
`base_client`, the concrete clients in `providers`; this module re-exports
both and adds `make_client`, which turns a `provider:model` string into a
ready client.

    make_client("anthropic:claude-sonnet-5-5")
    make_client("openai:<model-id>")
    make_client("deepseek:<model-id>")
    make_client("ollama:<model-id>")      # local, no credential needed
    make_client("stub")                   # offline

Phase 6A treats model choice as a sensitivity axis, so a run is identified by
its full spec, and the spec is recorded with the result.
"""

from __future__ import annotations

from typing import Any

from .base_client import (
    PRICING,
    ClientError,
    Completion,
    LLMClient,
    RunMetadata,
    StubClient,
    estimate_cost,
    now_iso,
    register_pricing,
)
from .providers import (
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_EFFORT,
    DEFAULT_MAX_TOKENS,
    ENDPOINTS,
    AnthropicClient,
    Endpoint,
    OpenAICompatibleClient,
)

#: Kept for callers that predate multi-provider support.
DEFAULT_MODEL = DEFAULT_ANTHROPIC_MODEL

#: Every provider `make_client` understands.
PROVIDERS: tuple[str, ...] = ("anthropic", "stub", *sorted(ENDPOINTS))


def parse_spec(spec: str) -> tuple[str, str]:
    """Split `"provider:model"`. A bare word is a provider with no model.

    A model id may itself contain colons (OpenRouter writes `vendor/name`,
    Ollama writes `name:tag`), so only the first colon separates.
    """
    text = (spec or "").strip()
    if not text:
        raise ClientError("empty client spec")
    provider, _, model = text.partition(":")
    provider = provider.strip().casefold()
    if provider not in PROVIDERS:
        known = ", ".join(PROVIDERS)
        raise ClientError(f"Unknown provider {provider!r}. Known: {known}")
    return provider, model.strip()


def make_client(spec: str, **options: Any) -> LLMClient:
    """Build a client from a `provider:model` spec.

    Options are passed through; each client ignores what does not apply to
    it, so one call site can serve every provider.
    """
    provider, model = parse_spec(spec)

    if provider == "stub":
        return StubClient(on_miss=options.get("on_miss", "empty"))
    if provider == "anthropic":
        return AnthropicClient(model=model or DEFAULT_ANTHROPIC_MODEL, **options)
    return OpenAICompatibleClient(model=model, provider=provider, **options)


def describe_providers() -> str:
    """A help block listing providers and the variable each one reads."""
    lines = [
        "  anthropic   ANTHROPIC_API_KEY      (default model: "
        f"{DEFAULT_ANTHROPIC_MODEL})",
        "  stub        -                      offline, returns nothing",
    ]
    for name in sorted(ENDPOINTS):
        endpoint = ENDPOINTS[name]
        lines.append(f"  {name:<11} {endpoint.env_var:<22} {endpoint.docs}")
    return "\n".join(lines)


__all__ = [
    "DEFAULT_ANTHROPIC_MODEL",
    "DEFAULT_EFFORT",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_MODEL",
    "ENDPOINTS",
    "PRICING",
    "PROVIDERS",
    "AnthropicClient",
    "ClientError",
    "Completion",
    "Endpoint",
    "LLMClient",
    "OpenAICompatibleClient",
    "RunMetadata",
    "StubClient",
    "describe_providers",
    "estimate_cost",
    "make_client",
    "now_iso",
    "parse_spec",
    "register_pricing",
]
