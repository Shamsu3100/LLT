"""Parse orchestration (build plan Phase 4.2-4.6).

One staff submission in, a validated and resolved constraint set out:

    text -> LLM -> JSON extraction -> Pydantic -> (retry once) -> resolve -> flags

The LLM never sees a timetable and never produces one (Phase 4.5). Its only
output is constraints, and every one of them passes through the verification
UI before the solver is called.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from ..schema.constraints import ConstraintSet
from ..schema.entities import Department
from .client import ClientError, Completion, LLMClient, RunMetadata
from .prompt import build_system_prompt, prompt_hash
from .resolve import EntityResolver, Flag, ResolvedConstraint, Severity

MAX_ATTEMPTS = 2  # one call, one retry on validation failure (Phase 4.2)


class JSONExtractionError(ValueError):
    """The response held no parsable JSON object."""


def extract_json(text: str) -> Any:
    """Pull a JSON value out of a model response.

    Tolerates code fences and surrounding prose, since a model that otherwise
    produced correct constraints should not be scored as a parse failure over
    a stray ```json. Anything else is an error, deliberately: silently
    repairing malformed output would hide a real failure mode from Phase 6A.
    """
    s = text.strip()
    if not s:
        raise JSONExtractionError("empty response")

    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else ""
        if "```" in s:
            s = s.rsplit("```", 1)[0]
        s = s.strip()

    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    for opener, closer in (("{", "}"), ("[", "]")):
        start = s.find(opener)
        if start == -1:
            continue
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(s)):
            ch = s[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(s[start : i + 1])
                    except json.JSONDecodeError:
                        break
    raise JSONExtractionError(f"no JSON object found in response: {text[:200]!r}")


def coerce_payload(value: Any) -> dict[str, Any]:
    """Accept the contracted shape, and the two near-misses models produce."""
    if isinstance(value, list):
        return {"constraints": value}
    if isinstance(value, dict):
        if "constraints" in value:
            return value
        if "type" in value:  # a bare single constraint
            return {"constraints": [value]}
    raise JSONExtractionError(
        f"expected {{'constraints': [...]}} or a list, got {type(value).__name__}"
    )


@dataclass
class Attempt:
    """One call to the model, kept for the run log."""

    index: int
    response_text: str
    metadata: RunMetadata
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "response_text": self.response_text,
            "metadata": self.metadata.as_dict(),
            "error": self.error,
        }


@dataclass
class ParseResult:
    """Everything one submission produced."""

    source_text: str
    constraints: list[ResolvedConstraint] = field(default_factory=list)
    attempts: list[Attempt] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def retried(self) -> bool:
        return len(self.attempts) > 1

    @property
    def blocked(self) -> list[ResolvedConstraint]:
        return [c for c in self.constraints if c.blocked]

    @property
    def flags(self) -> list[Flag]:
        return [f for c in self.constraints for f in c.flags]

    def flag_codes(self) -> list[str]:
        return [f.code for f in self.flags]

    @property
    def tokens(self) -> dict[str, int]:
        """Token usage summed over every attempt, retries included."""
        return {
            "input": sum(a.metadata.input_tokens or 0 for a in self.attempts),
            "output": sum(a.metadata.output_tokens or 0 for a in self.attempts),
            "cache_read": sum(a.metadata.cache_read_tokens or 0 for a in self.attempts),
        }

    @property
    def cost_usd(self) -> float:
        """Cost of this item, retries included. A retry is not free."""
        return sum(a.metadata.cost_usd or 0.0 for a in self.attempts)

    @property
    def ready_for_solver(self) -> list[Any]:
        """Constraints with no blocking flag. Verification still comes first."""
        from ..schema.constraints import SOLVABLE_TYPES

        return [
            c.constraint
            for c in self.constraints
            if not c.blocked and c.constraint.type in SOLVABLE_TYPES
        ]

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_text": self.source_text,
            "ok": self.ok,
            "error": self.error,
            "retried": self.retried,
            "tokens": self.tokens,
            "cost_usd": self.cost_usd,
            "constraints": [c.as_dict() for c in self.constraints],
            "attempts": [a.as_dict() for a in self.attempts],
        }


class ConstraintParser:
    """Turns staff text into resolved, flagged constraints."""

    def __init__(
        self,
        department: Department,
        client: LLMClient,
        *,
        examples: list[tuple[str, list[dict[str, Any]]]] | None = None,
        max_attempts: int = MAX_ATTEMPTS,
    ) -> None:
        self.department = department
        self.client = client
        self.resolver = EntityResolver(department)
        self.system_prompt = build_system_prompt(department, examples)
        self.prompt_hash = prompt_hash(self.system_prompt)
        self.max_attempts = max_attempts

    # -- single submission ------------------------------------------------

    def parse(self, text: str) -> ParseResult:
        result = ParseResult(source_text=text)
        user = text
        last_error: str | None = None

        for index in range(1, self.max_attempts + 1):
            try:
                completion = self._call(user)
            except ClientError as exc:
                # A failed call is this item's outcome, not the run's. A
                # 300-item batch must not die on one rate limit or outage.
                result.error = f"api call failed -> {exc}"
                return result

            attempt = Attempt(index, completion.text, completion.metadata)
            result.attempts.append(attempt)

            if completion.refused:
                category = completion.metadata.refusal_category or "unspecified"
                attempt.error = f"model declined the request (category: {category})"
                result.error = attempt.error
                return result  # a retry of the same text will decline again

            if completion.truncated:
                attempt.error = (
                    "response hit max_tokens and is truncated; a partial "
                    "constraint set must not be accepted"
                )
                result.error = attempt.error
                return result

            try:
                payload = coerce_payload(extract_json(completion.text))
                parsed = ConstraintSet.model_validate(payload)
            except (JSONExtractionError, ValidationError) as exc:
                last_error = _format_error(exc)
                attempt.error = last_error
                if index < self.max_attempts:
                    user = self._retry_message(text, completion.text, last_error)
                    continue
                result.error = last_error
                return result

            result.constraints = self.resolver.resolve_all(parsed.constraints)
            result.error = None
            return result

        result.error = last_error
        return result

    def parse_many(self, texts: list[str]) -> list[ParseResult]:
        return [self.parse(t) for t in texts]

    # -- internals --------------------------------------------------------

    def _call(self, user: str) -> Completion:
        return self.client.complete(
            self.system_prompt, user, prompt_hash=self.prompt_hash
        )

    @staticmethod
    def _retry_message(original: str, bad_response: str, error: str) -> str:
        return (
            f"{original}\n\n"
            f"---\n"
            f"Your previous reply could not be used.\n\n"
            f"Reply was:\n{bad_response[:1500]}\n\n"
            f"Problem:\n{error}\n\n"
            f"Send the corrected JSON only. No prose, no code fences. Keep the "
            f"same readings; fix only what the problem names."
        )


def _format_error(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        parts = []
        for err in exc.errors()[:6]:
            loc = ".".join(str(p) for p in err["loc"])
            parts.append(f"{loc}: {err['msg']}")
        return "schema validation failed -> " + "; ".join(parts)
    return f"{type(exc).__name__}: {exc}"


__all__ = [
    "MAX_ATTEMPTS",
    "Attempt",
    "ConstraintParser",
    "JSONExtractionError",
    "ParseResult",
    "Severity",
    "coerce_payload",
    "extract_json",
]
