"""Entity resolution and rule-based flagging (build plan Phase 4.3, 4.4, 4.6).

Two jobs, kept apart from the LLM call so that each is measurable on its own:

* **Resolution.** Map the string a staff member wrote onto a `Department` id.
  Phase 4.4 covers names; Phase 4.6 covers descriptions that pick out an
  entity without naming it ("the big lab", "my Tuesday group"). The resolver
  records *how* each reference resolved, so Phase 6A can report entity
  accuracy split by resolution path. A description resolving to the wrong room
  is a different failure from a hallucinated room, and the two are counted
  separately.

* **Flagging.** The plan's option (a): flag by rule, since LLM self-reported
  confidence is uncalibrated. Nothing here consults the `confidence` field.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

from ..schema.constraints import OPEN_TYPES, EntityKind
from ..schema.entities import Department

# -- normalisation --------------------------------------------------------

_TITLES = ("dr", "prof", "professor", "mr", "mrs", "ms", "miss", "sir", "madam")
_DETERMINERS = ("the", "my", "our", "a", "an", "this", "that", "these", "those", "all")
_PUNCT = re.compile(r"[^\w\s]+", re.UNICODE)
_WS = re.compile(r"\s+")


def normalise(text: str) -> str:
    """Casefold, strip accents and punctuation, drop leading titles and
    determiners, collapse spaces.

    Leading determiners go because they carry no reference content: "the first
    years" and "first years" name the same group. Keeping them would push
    every such mention onto the partial-match path and raise a confirm-this
    flag on input that resolved unambiguously. Descriptiveness is read from
    the original text by `is_descriptive`, so a stripped determiner still
    leaves the Phase 4.6 signal intact.
    """
    s = unicodedata.normalize("NFKD", text)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = _PUNCT.sub(" ", s).casefold()
    tokens = [t for t in _WS.sub(" ", s).strip().split(" ") if t]
    while len(tokens) > 1 and tokens[0] in _TITLES + _DETERMINERS:
        tokens.pop(0)
    return " ".join(tokens)


def is_descriptive(text: str) -> bool:
    """True when the reference describes an entity instead of naming it.

    The signal is a leading determiner ("the big lab", "my Tuesday group").
    Phase 4.6 counts these separately, because they are the class LlmAcq found
    staff actually produce: a reference that names no identifier at all.
    """
    s = _PUNCT.sub(" ", text).casefold().strip()
    first = s.split(" ")[0] if s else ""
    return first in _DETERMINERS


# -- outcomes -------------------------------------------------------------


class Outcome(str, Enum):
    ID = "id"  # matched an identifier
    NAME = "name"  # matched the full entity name
    ALIAS = "alias"  # matched a recorded alias
    PARTIAL = "partial"  # matched on token containment, single candidate
    AMBIGUOUS = "ambiguous"  # several candidates, none decisive
    UNRESOLVED = "unresolved"  # no candidate

    @property
    def resolved(self) -> bool:
        return self in (Outcome.ID, Outcome.NAME, Outcome.ALIAS, Outcome.PARTIAL)

    @property
    def needs_confirmation(self) -> bool:
        """Resolved, but by a route weak enough that the user should confirm."""
        return self in (Outcome.PARTIAL,)


@dataclass
class Reference:
    """One entity mention and what became of it."""

    field_path: str
    kind: EntityKind
    raw: str
    outcome: Outcome
    resolved_id: str | None = None
    candidates: list[str] = field(default_factory=list)
    descriptive: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "field": self.field_path,
            "kind": self.kind,
            "raw": self.raw,
            "outcome": self.outcome.value,
            "resolved_id": self.resolved_id,
            "candidates": list(self.candidates),
            "descriptive": self.descriptive,
        }


class Severity(str, Enum):
    BLOCK = "block"  # must not reach the solver until resolved
    REVIEW = "review"  # surfaced prominently in the verification UI


@dataclass
class Flag:
    code: str
    severity: Severity
    message: str
    field_path: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity.value,
            "message": self.message,
            "field": self.field_path,
        }


@dataclass
class ResolvedConstraint:
    """A parsed constraint plus its resolution report and flags."""

    constraint: Any
    references: list[Reference] = field(default_factory=list)
    flags: list[Flag] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return any(f.severity is Severity.BLOCK for f in self.flags)

    @property
    def resolved_ids(self) -> dict[str, str | None]:
        return {r.field_path: r.resolved_id for r in self.references}

    def as_dict(self) -> dict[str, Any]:
        return {
            "constraint": self.constraint.model_dump(mode="json"),
            "references": [r.as_dict() for r in self.references],
            "flags": [f.as_dict() for f in self.flags],
            "blocked": self.blocked,
        }


# -- strength cues (Phase 4.3) -------------------------------------------

HARD_CUES: tuple[str, ...] = (
    "can't", "cannot", "can not", "must", "never", "unable", "impossible",
    "not available", "unavailable", "won't", "will not", "has to", "have to",
    "need to", "required", "requires", "only", "no exceptions", "strictly",
)

SOFT_CUES: tuple[str, ...] = (
    "prefer", "prefers", "preferably", "rather", "if possible", "ideally",
    "would like", "'d like", "better", "hopefully", "try to", "where possible",
    "nice to", "happy to", "don't mind", "avoid if", "when possible",
)

#: Room-type words staff use that map to more than one configured type.
ROOM_TYPE_SYNONYMS: dict[str, tuple[str, ...]] = {
    "lab": ("lab", "computer_lab"),
    "laboratory": ("lab", "computer_lab"),
    "practical room": ("lab", "computer_lab"),
    "computer lab": ("computer_lab",),
    "computer room": ("computer_lab",),
    "hardware lab": ("lab",),
    "lecture hall": ("lecture",),
    "lecture theatre": ("lecture",),
    "lecture theater": ("lecture",),
    "theatre": ("lecture",),
    "seminar room": ("seminar",),
    "tutorial room": ("seminar",),
}


def _cues_present(text: str, cues: Iterable[str]) -> list[str]:
    low = text.casefold()
    return [c for c in cues if c in low]


# -- the resolver ---------------------------------------------------------


class EntityResolver:
    """Resolves staff-written entity references against a `Department`."""

    #: Cap on options offered with an unresolved reference.
    MAX_CANDIDATES = 8

    def __init__(self, department: Department) -> None:
        self.department = department
        self._index: dict[EntityKind, dict[str, list[tuple[str, Outcome]]]] = {}
        self._build_index()

    def _all_ids(self, kind: EntityKind) -> list[str]:
        return {
            "lecturer": self.department.lecturer_ids,
            "course": self.department.course_ids,
            "room": self.department.room_ids,
            "group": self.department.group_ids,
        }[kind]()

    def _build_index(self) -> None:
        sources: dict[EntityKind, list[Any]] = {
            "lecturer": list(self.department.lecturers),
            "course": list(self.department.courses),
            "room": list(self.department.rooms),
            "group": list(self.department.groups),
        }
        for kind, items in sources.items():
            table: dict[str, list[tuple[str, Outcome]]] = {}
            for item in items:
                self._add(table, normalise(item.id), item.id, Outcome.ID)
                name = getattr(item, "name", None)
                if name:
                    self._add(table, normalise(name), item.id, Outcome.NAME)
                for alias in getattr(item, "aliases", []) or []:
                    self._add(table, normalise(alias), item.id, Outcome.ALIAS)
            self._index[kind] = table

    @staticmethod
    def _add(
        table: dict[str, list[tuple[str, Outcome]]],
        key: str,
        entity_id: str,
        outcome: Outcome,
    ) -> None:
        if not key:
            return
        bucket = table.setdefault(key, [])
        if not any(eid == entity_id for eid, _ in bucket):
            bucket.append((entity_id, outcome))

    # -- room types are resolved against a configured set, not an index ---

    def _resolve_room_type(self, field_path: str, raw: str) -> Reference:
        available = set(self.department.room_types())
        key = normalise(raw).replace("_", " ")
        descriptive = is_descriptive(raw)

        # The synonym map is consulted before the direct match, because a word
        # can name one configured type and still be ambiguous in context:
        # "lab" is itself a type here, and also the word staff use for
        # "computer_lab". Surfacing that beats silently taking the literal
        # reading, which would be a misresolution no flag would catch.
        mapped = ROOM_TYPE_SYNONYMS.get(key)
        if mapped:
            hits = [m for m in mapped if m in available]
            if len(hits) > 1:
                return Reference(field_path, "room_type", raw, Outcome.AMBIGUOUS,
                                 candidates=sorted(hits), descriptive=descriptive)
            if len(hits) == 1:
                outcome = Outcome.ID if hits[0] == key.replace(" ", "_") else Outcome.ALIAS
                return Reference(field_path, "room_type", raw, outcome, hits[0],
                                 descriptive=descriptive)

        direct = key.replace(" ", "_")
        if direct in available:
            return Reference(field_path, "room_type", raw, Outcome.ID, direct,
                             descriptive=descriptive)

        partial = sorted(t for t in available if key and key in t.replace("_", " "))
        if len(partial) == 1:
            return Reference(field_path, "room_type", raw, Outcome.PARTIAL, partial[0],
                             descriptive=descriptive)
        if len(partial) > 1:
            return Reference(field_path, "room_type", raw, Outcome.AMBIGUOUS,
                             candidates=partial, descriptive=descriptive)
        return Reference(field_path, "room_type", raw, Outcome.UNRESOLVED,
                         candidates=sorted(available), descriptive=descriptive)

    def resolve_reference(self, field_path: str, kind: EntityKind, raw: str) -> Reference:
        if kind == "room_type":
            return self._resolve_room_type(field_path, raw)

        table = self._index.get(kind, {})
        key = normalise(raw)
        descriptive = is_descriptive(raw)

        exact = table.get(key, [])
        if len(exact) == 1:
            entity_id, outcome = exact[0]
            return Reference(field_path, kind, raw, outcome, entity_id,
                             descriptive=descriptive)
        if len(exact) > 1:
            return Reference(field_path, kind, raw, Outcome.AMBIGUOUS,
                             candidates=sorted(eid for eid, _ in exact),
                             descriptive=descriptive)

        # Token containment, both directions: "machine learning" vs
        # "machine learning 2", and "programming lab" vs "the programming lab".
        tokens = set(key.split()) if key else set()
        hits: set[str] = set()
        if tokens:
            for candidate_key, bucket in table.items():
                cand_tokens = set(candidate_key.split())
                if tokens <= cand_tokens or cand_tokens <= tokens:
                    hits.update(eid for eid, _ in bucket)
        if len(hits) == 1:
            return Reference(field_path, kind, raw, Outcome.PARTIAL, hits.pop(),
                             descriptive=descriptive)
        if len(hits) > 1:
            return Reference(field_path, kind, raw, Outcome.AMBIGUOUS,
                             candidates=sorted(hits), descriptive=descriptive)
        # Offer the full set of this kind so the clarification question can
        # list options (Phase 4.6), capped to keep the prompt and UI readable.
        return Reference(field_path, kind, raw, Outcome.UNRESOLVED,
                         candidates=self._all_ids(kind)[:self.MAX_CANDIDATES],
                         descriptive=descriptive)

    # -- whole-constraint pass -------------------------------------------

    def resolve(self, constraint: Any) -> ResolvedConstraint:
        refs = [
            self.resolve_reference(path, kind, raw)
            for path, kind, raw in constraint.entity_references()
        ]
        out = ResolvedConstraint(constraint=constraint, references=refs)
        out.flags = self._flags(constraint, refs)
        return out

    def resolve_all(self, constraints: Iterable[Any]) -> list[ResolvedConstraint]:
        return [self.resolve(c) for c in constraints]

    # -- rule-based flags (Phase 4.3, option (a)) ------------------------

    def _flags(self, constraint: Any, refs: list[Reference]) -> list[Flag]:
        flags: list[Flag] = []

        for ref in refs:
            if ref.outcome is Outcome.UNRESOLVED:
                flags.append(Flag(
                    "unknown_entity", Severity.BLOCK,
                    f"No {ref.kind} matches {ref.raw!r}.", ref.field_path,
                ))
            elif ref.outcome is Outcome.AMBIGUOUS:
                shown = ", ".join(ref.candidates[:5]) or "several entities"
                flags.append(Flag(
                    "ambiguous_entity", Severity.BLOCK,
                    f"{ref.raw!r} could be: {shown}. Ask which.", ref.field_path,
                ))
            elif ref.outcome.needs_confirmation:
                flags.append(Flag(
                    "weak_entity_match", Severity.REVIEW,
                    f"{ref.raw!r} was matched to {ref.resolved_id} on a partial "
                    f"match. Confirm before use.", ref.field_path,
                ))
            if ref.descriptive and ref.outcome.resolved:
                flags.append(Flag(
                    "descriptive_reference", Severity.REVIEW,
                    f"{ref.raw!r} describes a {ref.kind} without naming it; "
                    f"read as {ref.resolved_id}. Confirm.", ref.field_path,
                ))

        if constraint.type in OPEN_TYPES:
            flags.append(Flag(
                "open_category", Severity.BLOCK,
                f"Type {constraint.type!r} never reaches the solver. Log and categorise.",
            ))

        flags.extend(self._strength_flags(constraint))
        flags.extend(self._semantic_flags(constraint, refs))
        return flags

    def _strength_flags(self, constraint: Any) -> list[Flag]:
        if constraint.type in OPEN_TYPES:
            return []
        text = constraint.source_text
        hard = _cues_present(text, HARD_CUES)
        soft = _cues_present(text, SOFT_CUES)

        if hard and soft:
            return [Flag(
                "strength_ambiguous", Severity.REVIEW,
                f"Source text carries both hard cues ({', '.join(hard[:3])}) and "
                f"soft cues ({', '.join(soft[:3])}). Read as {constraint.strength}; "
                f"confirm with the user.",
                "strength",
            )]
        if not hard and not soft:
            return [Flag(
                "strength_uncued", Severity.REVIEW,
                f"No hard or soft cue in the source text. Read as "
                f"{constraint.strength} by default; confirm with the user.",
                "strength",
            )]
        if hard and constraint.strength == "soft":
            return [Flag(
                "strength_cue_mismatch", Severity.REVIEW,
                f"Hard cue(s) present ({', '.join(hard[:3])}) but parsed as soft.",
                "strength",
            )]
        if soft and constraint.strength == "hard":
            return [Flag(
                "strength_cue_mismatch", Severity.REVIEW,
                f"Soft cue(s) present ({', '.join(soft[:3])}) but parsed as hard.",
                "strength",
            )]
        return []

    def _semantic_flags(self, constraint: Any, refs: list[Reference]) -> list[Flag]:
        """Checks that need the department data, not only the schema."""
        flags: list[Flag] = []
        ids = {r.field_path: r.resolved_id for r in refs}

        if constraint.type == "course_requires_room":
            course_id, room_id = ids.get("course"), ids.get("room")
            course = next((c for c in self.department.courses if c.id == course_id), None)
            room = next((r for r in self.department.rooms if r.id == room_id), None)
            if course and room and room.capacity < course.size:
                flags.append(Flag(
                    "capacity_conflict", Severity.REVIEW,
                    f"{room.id} holds {room.capacity}; {course.id} has {course.size} "
                    f"students. Infeasible as stated.", "room",
                ))

        if constraint.type == "no_early_start":
            available = self.department.periods
            if available and constraint.earliest_period > max(available):
                flags.append(Flag(
                    "out_of_range", Severity.BLOCK,
                    f"earliest_period {constraint.earliest_period} exceeds the last "
                    f"period of the day ({max(available)}).", "earliest_period",
                ))

        if constraint.type in ("lecturer_unavailable", "lecturer_prefers_time"):
            available_days = set(self.department.days)
            stated = set(constraint.days or [])
            unknown = sorted(stated - available_days) if stated else []
            if unknown:
                flags.append(Flag(
                    "out_of_range", Severity.REVIEW,
                    f"Day(s) {unknown} are not taught in this department "
                    f"({', '.join(self.department.days)}).", "days",
                ))
            periods = set(self.department.periods)
            bad_periods = sorted(set(constraint.periods or []) - periods) if periods else []
            if bad_periods:
                flags.append(Flag(
                    "out_of_range", Severity.REVIEW,
                    f"Period(s) {bad_periods} are outside the configured grid "
                    f"(1-{max(periods)}).", "periods",
                ))

        if constraint.type == "max_consecutive_hours":
            longest = max((c.duration for c in self.department.courses), default=0)
            if constraint.max_hours < longest:
                flags.append(Flag(
                    "capacity_conflict", Severity.REVIEW,
                    f"max_hours {constraint.max_hours} is below the longest single "
                    f"course duration ({longest} periods), which no timetable can "
                    f"satisfy.", "max_hours",
                ))

        return flags


_INDEXED_FIELD = re.compile(r"^(\w+)\[(\d+)\]$")


class UnresolvedReferenceError(ValueError):
    """A constraint still naming something the register does not contain."""


def materialise(resolved: ResolvedConstraint) -> Any:
    """Rewrite a constraint's entity fields as canonical department ids.

    The parser emits the words staff wrote; the solver works in ids. This is
    the one conversion between them, and it refuses to guess: an unresolved
    or ambiguous reference raises. Those constraints carry a blocking flag
    already, so reaching this function with one is a caller error.
    """
    constraint = resolved.constraint
    data = constraint.model_dump()
    for ref in resolved.references:
        if ref.resolved_id is None:
            raise UnresolvedReferenceError(
                f"{constraint.type}.{ref.field_path} is {ref.outcome.value} "
                f"for {ref.raw!r}; resolve it with the user before solving"
            )
        match = _INDEXED_FIELD.match(ref.field_path)
        if match:
            data[match.group(1)][int(match.group(2))] = ref.resolved_id
        else:
            data[ref.field_path] = ref.resolved_id
    return type(constraint).model_validate(data)


def materialise_all(
    resolved: Iterable[ResolvedConstraint], *, skip_blocked: bool = True
) -> list[Any]:
    """Materialise every constraint fit to solve, dropping the rest.

    With `skip_blocked`, anything carrying a blocking flag is left out, which
    is what the verification step is for. Without it, a blocked constraint
    raises, so a caller cannot route unverified input to the solver by
    accident.
    """
    out: list[Any] = []
    for item in resolved:
        if skip_blocked and item.blocked:
            continue
        if item.constraint.type in OPEN_TYPES:
            if skip_blocked:
                continue
            raise UnresolvedReferenceError(
                f"{item.constraint.type} never reaches the solver"
            )
        out.append(materialise(item))
    return out


__all__ = [
    "EntityResolver",
    "Flag",
    "HARD_CUES",
    "Outcome",
    "Reference",
    "ResolvedConstraint",
    "ROOM_TYPE_SYNONYMS",
    "SOFT_CUES",
    "Severity",
    "UnresolvedReferenceError",
    "is_descriptive",
    "materialise",
    "materialise_all",
    "normalise",
]
