"""Concrete provider clients (build plan Phase 4.2, 6A model-choice axis).

Two implementations cover the field:

* `AnthropicClient` for Claude, through the `anthropic` SDK.
* `OpenAICompatibleClient` for everything that speaks the OpenAI
  chat-completions API — OpenAI itself, DeepSeek, OpenRouter, Groq, Together,
  Mistral, xAI, and a local Ollama or vLLM server. Only the base URL, the
  credential and the model id change, so one class serves them all.

**Model ids are not hardcoded for non-Anthropic providers.** They change
often and vary by account, so a guessed default would fail at the first call
or, worse, silently run a different model than the paper claims. Pass the id
explicitly, or call `list_models()` to see what the credential can reach.
"""

from __future__ import annotations

import os
from typing import Any, Iterable

from .base_client import (
    ClientError,
    Completion,
    RunMetadata,
    estimate_cost,
    now_iso,
)

#: The build plan's §2 choice.
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5-5"
DEFAULT_MAX_TOKENS = 16000
DEFAULT_EFFORT = "medium"


# -- provider endpoints --------------------------------------------------


class Endpoint:
    """One OpenAI-compatible provider: where it lives and how it authenticates."""

    def __init__(
        self,
        name: str,
        base_url: str | None,
        env_var: str,
        *,
        docs: str = "",
        supports_json_mode: bool = True,
    ) -> None:
        self.name = name
        self.base_url = base_url
        self.env_var = env_var
        self.docs = docs
        self.supports_json_mode = supports_json_mode


#: Base URLs are stable published endpoints; model ids deliberately are not
#: listed, for the reason in the module docstring.
ENDPOINTS: dict[str, Endpoint] = {
    "openai": Endpoint(
        "openai", None, "OPENAI_API_KEY",
        docs="https://platform.openai.com/docs/models",
    ),
    "deepseek": Endpoint(
        "deepseek", "https://api.deepseek.com", "DEEPSEEK_API_KEY",
        docs="https://api-docs.deepseek.com",
    ),
    "openrouter": Endpoint(
        "openrouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY",
        docs="https://openrouter.ai/models",
    ),
    "groq": Endpoint(
        "groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY",
        docs="https://console.groq.com/docs/models",
    ),
    "together": Endpoint(
        "together", "https://api.together.xyz/v1", "TOGETHER_API_KEY",
        docs="https://docs.together.ai/docs/inference-models",
    ),
    "mistral": Endpoint(
        "mistral", "https://api.mistral.ai/v1", "MISTRAL_API_KEY",
        docs="https://docs.mistral.ai/getting-started/models/",
    ),
    "xai": Endpoint(
        "xai", "https://api.x.ai/v1", "XAI_API_KEY",
        docs="https://docs.x.ai/docs/models",
    ),
    "ollama": Endpoint(
        "ollama", "http://localhost:11434/v1", "OLLAMA_API_KEY",
        docs="https://ollama.com/library",
    ),
    "local": Endpoint(
        "local", "http://localhost:8000/v1", "LOCAL_API_KEY",
        docs="any OpenAI-compatible server (vLLM, llama.cpp, LM Studio)",
    ),
}


# -- Claude ---------------------------------------------------------------


class AnthropicClient:
    """Claude, through the `anthropic` SDK.

    Before any staff text is sent through this, the provider's data-retention
    policy must be held in writing and stated in the ethics application and
    the paper (§5, Privacy). The same applies to every provider below.

    No server-side refusal fallback is configured. A fallback would silently
    answer some items on a different model, and a measurement attributed to
    one model has to come from that model. A refusal is surfaced as a failed
    item instead, with its category, and counted.
    """

    backend = "anthropic"
    provider = "anthropic"

    def __init__(
        self,
        model: str = DEFAULT_ANTHROPIC_MODEL,
        *,
        effort: str = DEFAULT_EFFORT,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        api_key: str | None = None,
        max_retries: int = 3,
        timeout: float = 120.0,
        cache_system_prompt: bool = True,
        **ignored: Any,
    ) -> None:
        try:
            import anthropic
        except ModuleNotFoundError as exc:  # pragma: no cover - env dependent
            raise ClientError(
                "The anthropic package is not installed. Run "
                "`pip install anthropic`, or use StubClient for offline work."
            ) from exc

        self._anthropic = anthropic
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        token = os.environ.get("ANTHROPIC_AUTH_TOKEN")
        profile = os.environ.get("ANTHROPIC_PROFILE")
        if not (key or token or profile):
            raise ClientError(
                "No credential found for anthropic. Set ANTHROPIC_API_KEY in the "
                "environment, or sign in with `ant auth login`, then re-run."
            )

        self._client = anthropic.Anthropic(
            api_key=key, max_retries=max_retries, timeout=timeout
        )
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.cache_system_prompt = cache_system_prompt
        self.temperature = None
        self.json_mode = False

    def _system_param(self, system: str) -> Any:
        """The system prompt, marked cacheable.

        It is the same ~2,700 tokens on every item of a run — schema, the
        department register, and 16 few-shot examples — while the user turn is
        one short sentence. Caching it turns most of the input cost of a
        300-item run into cache reads at a twentieth of the rate. Verify with
        `usage.cache_read_input_tokens`: a run reporting zero cache reads
        throughout is paying full price.
        """
        if not self.cache_system_prompt:
            return system
        return [{
            "type": "text",
            "text": system,
            "cache_control": {"type": "ephemeral"},
        }]

    def complete(self, system: str, user: str, *, prompt_hash: str = "") -> Completion:
        anthropic = self._anthropic
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self._system_param(system),
                output_config={"effort": self.effort},
                messages=[{"role": "user", "content": user}],
            )
        except anthropic.NotFoundError as exc:
            raise ClientError(
                f"Model {self.model!r} is unknown or unavailable to this "
                f"organisation: {exc}"
            ) from exc
        except anthropic.RateLimitError as exc:
            raise ClientError(
                f"Rate limited after the SDK's own retries: {exc}", retryable=True
            ) from exc
        except anthropic.APITimeoutError as exc:
            raise ClientError(f"Request timed out: {exc}", retryable=True) from exc
        except anthropic.APIStatusError as exc:
            retryable = getattr(exc, "type", None) in ("overloaded_error", "api_error")
            raise ClientError(
                f"API returned {exc.status_code} ({getattr(exc, 'type', '?')}): "
                f"{getattr(exc, 'message', exc)}",
                retryable=retryable,
            ) from exc
        except anthropic.APIConnectionError as exc:
            raise ClientError(f"Could not reach the API: {exc}", retryable=True) from exc

        return self._to_completion(response, prompt_hash)

    def _to_completion(self, response: Any, prompt_hash: str) -> Completion:
        text = "".join(
            block.text
            for block in response.content
            if getattr(block, "type", "") == "text"
        )
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", None) or 0
        output_tokens = getattr(usage, "output_tokens", None) or 0
        cache_read = getattr(usage, "cache_read_input_tokens", None) or 0
        cache_write = getattr(usage, "cache_creation_input_tokens", None) or 0

        # stop_details is populated only on a refusal; guard before reading.
        details = getattr(response, "stop_details", None)
        category = getattr(details, "category", None) if details else None

        return Completion(
            text=text,
            metadata=RunMetadata(
                model=self.model,
                prompt_hash=prompt_hash,
                timestamp=now_iso(),
                backend=self.backend,
                provider=self.provider,
                effort=self.effort,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cache_read,
                cache_write_tokens=cache_write,
                response_id=getattr(response, "id", None),
                stop_reason=getattr(response, "stop_reason", None),
                refusal_category=category,
                cost_usd=estimate_cost(
                    self.model, input_tokens, output_tokens, cache_read
                ),
            ),
            raw=response,
        )

    def ping(self) -> RunMetadata:
        """One minimal call, to prove the credential and model work."""
        return self.complete(
            "Reply with the single word: ok", "ping", prompt_hash="ping"
        ).metadata

    def list_models(self) -> list[str]:
        try:
            return [m.id for m in self._client.models.list()]
        except Exception as exc:  # pragma: no cover - network dependent
            raise ClientError(f"Could not list models: {exc}") from exc


# -- everything that speaks chat-completions -----------------------------


class OpenAICompatibleClient:
    """OpenAI, DeepSeek, OpenRouter, Groq, Together, Mistral, xAI, or local.

    One class because they implement the same chat-completions contract; the
    provider name selects a base URL and a credential variable.

    `temperature` and native JSON mode are both off unless asked for: the
    Claude arm cannot accept the first and does not use the second, and an
    asymmetric instrument would measure the harness instead of the model.
    """

    backend = "openai_compatible"

    def __init__(
        self,
        model: str,
        *,
        provider: str = "openai",
        base_url: str | None = None,
        api_key: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float | None = None,
        json_mode: bool = False,
        effort: str | None = None,
        max_retries: int = 3,
        timeout: float = 120.0,
        **ignored: Any,
    ) -> None:
        try:
            import openai
        except ModuleNotFoundError as exc:  # pragma: no cover - env dependent
            raise ClientError(
                "The openai package is not installed. Run `pip install openai`, "
                "or use StubClient for offline work."
            ) from exc

        self._openai = openai
        endpoint = ENDPOINTS.get(provider)
        if endpoint is None:
            known = ", ".join(sorted(ENDPOINTS))
            raise ClientError(f"Unknown provider {provider!r}. Known: {known}")
        if not model:
            raise ClientError(
                f"{provider} needs an explicit model id (see {endpoint.docs}). "
                f"Model ids change and vary by account, so none is assumed."
            )

        url = base_url or endpoint.base_url
        key = api_key or os.environ.get(endpoint.env_var)
        if not key:
            # A local server usually needs no real key, but the SDK insists
            # on a non-empty string.
            if provider in ("ollama", "local"):
                key = "not-needed"
            else:
                raise ClientError(
                    f"No credential found for {provider}. Set {endpoint.env_var} "
                    f"in the environment, then re-run."
                )

        self._client = openai.OpenAI(
            api_key=key, base_url=url, max_retries=max_retries, timeout=timeout
        )
        self.provider = provider
        self.endpoint = endpoint
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.json_mode = json_mode
        self.effort = effort

    def complete(self, system: str, user: str, *, prompt_hash: str = "") -> Completion:
        openai = self._openai
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_completion_tokens": self.max_tokens,
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.json_mode and self.endpoint.supports_json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if self.effort:
            kwargs["reasoning_effort"] = self.effort

        try:
            response = self._client.chat.completions.create(**kwargs)
        except openai.NotFoundError as exc:
            raise ClientError(
                f"Model {self.model!r} is unknown to {self.provider} or "
                f"unavailable to this credential: {exc}. See {self.endpoint.docs}"
            ) from exc
        except openai.RateLimitError as exc:
            raise ClientError(
                f"Rate limited after the SDK's own retries: {exc}", retryable=True
            ) from exc
        except openai.APITimeoutError as exc:
            raise ClientError(f"Request timed out: {exc}", retryable=True) from exc
        except openai.BadRequestError as exc:
            raise ClientError(
                f"{self.provider} rejected the request: {exc}. Some providers "
                f"do not accept every parameter; try --no-json-mode or drop "
                f"--effort."
            ) from exc
        except openai.APIStatusError as exc:
            raise ClientError(
                f"{self.provider} returned {exc.status_code}: {exc}",
                retryable=exc.status_code >= 500,
            ) from exc
        except openai.APIConnectionError as exc:
            raise ClientError(
                f"Could not reach {self.provider} at {self._client.base_url}: {exc}",
                retryable=True,
            ) from exc

        return self._to_completion(response, prompt_hash)

    def _to_completion(self, response: Any, prompt_hash: str) -> Completion:
        choice = response.choices[0] if response.choices else None
        message = getattr(choice, "message", None)
        text = (getattr(message, "content", None) or "") if message else ""
        finish = getattr(choice, "finish_reason", None) if choice else None

        # Some providers signal a content-policy stop in finish_reason.
        refusal = getattr(message, "refusal", None) if message else None
        if refusal or finish == "content_filter":
            finish = "refusal"

        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", None) or 0
        output_tokens = getattr(usage, "completion_tokens", None) or 0

        cached = 0
        details = getattr(usage, "prompt_tokens_details", None)
        if details is not None:
            cached = getattr(details, "cached_tokens", None) or 0

        reasoning = None
        out_details = getattr(usage, "completion_tokens_details", None)
        if out_details is not None:
            reasoning = getattr(out_details, "reasoning_tokens", None)

        return Completion(
            text=text,
            metadata=RunMetadata(
                model=self.model,
                prompt_hash=prompt_hash,
                timestamp=now_iso(),
                backend=self.backend,
                provider=self.provider,
                effort=self.effort,
                temperature=self.temperature,
                json_mode=self.json_mode,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cached,
                reasoning_tokens=reasoning,
                response_id=getattr(response, "id", None),
                stop_reason=finish,
                refusal_category="content_filter" if finish == "refusal" else None,
                cost_usd=estimate_cost(
                    self.model, input_tokens, output_tokens, cached
                ),
            ),
            raw=response,
        )

    def ping(self) -> RunMetadata:
        return self.complete(
            "Reply with the single word: ok", "ping", prompt_hash="ping"
        ).metadata

    def list_models(self) -> list[str]:
        try:
            return sorted(m.id for m in self._client.models.list())
        except Exception as exc:  # pragma: no cover - network dependent
            raise ClientError(
                f"Could not list {self.provider} models: {exc}"
            ) from exc


__all__ = [
    "DEFAULT_ANTHROPIC_MODEL",
    "DEFAULT_EFFORT",
    "DEFAULT_MAX_TOKENS",
    "ENDPOINTS",
    "AnthropicClient",
    "Endpoint",
    "OpenAICompatibleClient",
]
