"""In-session credential handling for the UI (build plan §5, Privacy).

A researcher can type a provider credential into the app instead of exporting
a shell variable. The constraint that shapes this module: **a credential must
never reach the study log, an export, or the disk.**

How that is guaranteed:

* Credentials are held in Streamlit session state only, for the life of the
  process. Nothing here writes to the SQLite store or to a file.
* They are passed to the client as an `api_key` argument, so they are never
  put into `os.environ` either, where a later subprocess or a crash dump could
  pick them up.
* `redact` scrubs anything key-shaped out of text before it is shown or
  stored, and `assert_no_credentials` is the test hook that proves an export
  is clean.
* Nothing in this module returns a credential to a caller that wants to
  display it. `describe` reports only whether one is set and how long it is.

Participants never see this page. Key entry belongs to the researcher setting
the study up, not to the person being measured.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping

#: Credential shapes worth catching in free text. Deliberately broad: a false
#: positive redacts a harmless string, a false negative leaks a key.
_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}"),       # Anthropic
    re.compile(r"sk-proj-[A-Za-z0-9_\-]{8,}"),      # OpenAI project keys
    re.compile(r"sk-or-v1-[A-Za-z0-9_\-]{8,}"),     # OpenRouter
    re.compile(r"\bgsk_[A-Za-z0-9]{20,}"),          # Groq
    re.compile(r"\bxai-[A-Za-z0-9]{20,}"),          # xAI
    re.compile(r"\bsk-[A-Za-z0-9]{20,}"),           # generic sk- keys
    re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{20,}", re.IGNORECASE),
)

REDACTED = "[redacted]"


def looks_like_credential(text: str) -> bool:
    return any(p.search(text) for p in _PATTERNS)


def redact(value: Any) -> Any:
    """Replace anything key-shaped, walking into dicts and lists.

    Applied to text on its way to the screen or the log. It is a backstop,
    not the primary defence: the primary defence is that credentials are
    never handed to the store in the first place.
    """
    if isinstance(value, str):
        out = value
        for pattern in _PATTERNS:
            out = pattern.sub(REDACTED, out)
        return out
    if isinstance(value, Mapping):
        return {k: redact(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(redact(v) for v in value)
    return value


def assert_no_credentials(payload: Any, *, where: str = "payload") -> None:
    """Raise if anything key-shaped survives into `payload`.

    The test hook behind the guarantee. Run it over a session export and over
    the event log; a study whose logs are shared must not carry a key.
    """
    import json

    text = payload if isinstance(payload, str) else json.dumps(payload, default=str)
    for pattern in _PATTERNS:
        match = pattern.search(text)
        if match:
            raise AssertionError(
                f"{where} contains something credential-shaped at offset "
                f"{match.start()}. It must never be stored or exported."
            )


class CredentialStore:
    """Provider credentials for this process only.

    Backed by a plain dict, which the UI keeps in Streamlit session state.
    Deliberately has no save, no load and no `get_all`: there is nowhere for
    a credential to go except into a client constructor.
    """

    def __init__(self, backing: dict[str, str] | None = None) -> None:
        self._keys: dict[str, str] = backing if backing is not None else {}

    def set(self, provider: str, key: str) -> None:
        cleaned = (key or "").strip()
        if cleaned:
            self._keys[provider] = cleaned
        else:
            self._keys.pop(provider, None)

    def clear(self, provider: str | None = None) -> None:
        if provider is None:
            self._keys.clear()
        else:
            self._keys.pop(provider, None)

    def has(self, provider: str) -> bool:
        return bool(self._keys.get(provider))

    def providers(self) -> list[str]:
        return sorted(self._keys)

    def options_for(self, provider: str) -> dict[str, str]:
        """Client kwargs for this provider, or `{}` to fall back to the env.

        The only way a credential leaves this object, and it goes straight
        into a client constructor.
        """
        key = self._keys.get(provider)
        return {"api_key": key} if key else {}

    def describe(self, provider: str) -> str:
        """Whether a credential is set, never any part of its value."""
        key = self._keys.get(provider)
        if not key:
            return "not set"
        return f"set in this session ({len(key)} characters)"


__all__ = [
    "REDACTED",
    "CredentialStore",
    "assert_no_credentials",
    "looks_like_credential",
    "redact",
]
