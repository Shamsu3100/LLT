"""Shared client contract across providers (build plan Phase 4.2, 6A).

Phase 6A calls for testing sensitivity to model choice. With several
providers behind one contract that becomes a real axis: the same prompt, the
same schema and the same scoring, run against Claude, GPT, DeepSeek or a
local model, with the differences attributable to the model.

Every client returns a `Completion` carrying the metadata the paper needs:
provider, model, prompt digest, timestamp, token usage and — where rates are
known — estimated cost.

**Keeping the comparison fair.** The instrument has to be identical across
providers, or a difference measures the harness instead of the model. So the
same system prompt, the same JSON-in-the-prompt contract, the same Pydantic
validation and the same single retry run everywhere. Two provider features
are therefore off by default and flagged when used:

* Native JSON mode (`response_format`), which Anthropic's path does not use.
* `temperature`, which the current Claude models reject outright, so a run
  that sets it is not comparable to a Claude run.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, runtime_checkable

#: USD per million tokens. Only rates that can be verified are listed; an
#: unlisted model reports no cost instead of a guess. Add yours with
#: `register_pricing` before quoting a figure in the paper.
PRICING: dict[str, dict[str, float]] = {
    "claude-opus-5-5": {"input": 4.00, "output": 20.00, "cache_read": 0.20},
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00, "cache_read": 0.10},
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50, "cache_read": 0.01},
}


def register_pricing(
    model: str, input_rate: float, output_rate: float, cache_read: float = 0.0
) -> None:
    """Record USD-per-million-token rates for a model.

    Provider rates change and are not all public in a machine-readable form,
    so they are supplied, not assumed. A model with no entry reports
    `cost_usd = None`, which is honest; a wrong hardcoded rate would not be.
    """
    PRICING[model] = {
        "input": input_rate, "output": output_rate, "cache_read": cache_read
    }


def estimate_cost(
    model: str, input_tokens: int, output_tokens: int, cache_read_tokens: int = 0
) -> float | None:
    rates = PRICING.get(model)
    if rates is None:
        return None
    return (
        input_tokens * rates["input"]
        + output_tokens * rates["output"]
        + cache_read_tokens * rates["cache_read"]
    ) / 1_000_000


@dataclass(frozen=True)
class RunMetadata:
    """Everything needed to reproduce and price one call."""

    model: str
    prompt_hash: str
    timestamp: str
    backend: str
    provider: str | None = None
    effort: str | None = None
    temperature: float | None = None
    json_mode: bool = False
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    #: Tokens written to the cache. Billed above the base input rate on the
    #: first call of a run; zero thereafter while the entry stays warm.
    cache_write_tokens: int | None = None
    #: Reasoning tokens, where the provider reports them separately. Billed
    #: as output on every provider that charges for them.
    reasoning_tokens: int | None = None
    response_id: str | None = None
    stop_reason: str | None = None
    refusal_category: str | None = None
    cost_usd: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Completion:
    text: str
    metadata: RunMetadata
    raw: Any = field(default=None, repr=False)

    @property
    def refused(self) -> bool:
        return self.metadata.stop_reason == "refusal"

    @property
    def truncated(self) -> bool:
        """Output hit the ceiling. Provider spellings differ."""
        return self.metadata.stop_reason in ("max_tokens", "length")


class ClientError(RuntimeError):
    """A call that failed for a reason the caller should see, not a crash."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


@runtime_checkable
class LLMClient(Protocol):
    """Minimal contract the parser depends on."""

    def complete(self, system: str, user: str, *, prompt_hash: str = "") -> Completion:
        ...


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


class StubClient:
    """Offline client returning canned responses, keyed by input text.

    Used by the test suite and by anyone running the pipeline without a
    credential. A miss is an error by default: a stub that silently invented
    output would make a passing test meaningless.
    """

    backend = "stub"
    provider = "stub"

    def __init__(
        self,
        responses: dict[str, Any] | None = None,
        *,
        model: str = "stub",
        on_miss: str = "raise",
        default: Any = None,
    ) -> None:
        if on_miss not in ("raise", "default", "empty"):
            raise ValueError("on_miss must be 'raise', 'default' or 'empty'")
        self.responses = dict(responses or {})
        self.model = model
        self.effort = None
        self.on_miss = on_miss
        self.default = default
        self.calls: list[tuple[str, str]] = []

    @staticmethod
    def _key(text: str) -> str:
        return " ".join(text.split()).strip().casefold()

    def add(self, text: str, payload: Any) -> "StubClient":
        """Register a response. `payload` may be a JSON string or an object."""
        self.responses[self._key(text)] = payload
        return self

    def complete(self, system: str, user: str, *, prompt_hash: str = "") -> Completion:
        self.calls.append((system, user))
        key = self._key(user)
        if key in self.responses:
            payload = self.responses[key]
        elif self.on_miss == "default" and self.default is not None:
            payload = self.default
        elif self.on_miss == "empty":
            payload = {"constraints": []}
        else:
            raise KeyError(
                f"StubClient has no response for {user[:80]!r}. Register one with "
                f".add(), or set on_miss='empty'."
            )

        text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
        return Completion(
            text=text,
            metadata=RunMetadata(
                model=self.model,
                prompt_hash=prompt_hash,
                timestamp=now_iso(),
                backend=self.backend,
                provider=self.provider,
                stop_reason="end_turn",
            ),
        )


__all__ = [
    "PRICING",
    "ClientError",
    "Completion",
    "LLMClient",
    "RunMetadata",
    "StubClient",
    "estimate_cost",
    "now_iso",
    "register_pricing",
]
