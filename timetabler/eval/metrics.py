"""Translation-accuracy metrics (build plan Phase 6A).

Four measures, as specified: exact match, type accuracy, hard/soft accuracy,
entity accuracy. Entity errors are split into unknown-name and misresolved
cases, and descriptive references are counted separately, because Phase 4.6
treats a description read as the wrong room as a different failure from a
hallucinated room.

Comparison conventions, all stated because each is a methodological choice:

* **Gold uses canonical ids.** A gold entity field holds `L001`, not "Lee".
  Predictions hold what the model wrote, and are compared through the
  resolver's output. Translation accuracy therefore scores the parser and the
  resolver as one pipeline, which is what the staff-facing system is.
* **`source_text` is excluded.** It is a span quote, and two labellers will
  disagree about span boundaries without disagreeing about the reading.
* **`weight` is excluded by default.** A soft weight is a scale choice, not a
  reading of the sentence. Pass `include_weight=True` to score it.
* **Alignment is one-to-one and greedy on a match score.** One sentence may
  yield several constraints, so predictions are matched to gold before
  per-pair measures are taken. Unmatched predictions count as spurious and
  unmatched gold as missed.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Sequence

from ..schema.constraints import ALL_TYPES, OPEN_TYPES

EXCLUDED_FIELDS = frozenset({"source_text", "confidence", "type"})


# -- canonical forms -----------------------------------------------------


def _norm_scalar(value: Any) -> Any:
    if isinstance(value, str):
        return " ".join(value.split()).casefold()
    return value


def _norm_value(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return tuple(sorted(_norm_scalar(v) for v in value))
    return _norm_scalar(value)


def canonical(
    constraint: Any,
    resolved_ids: dict[str, str | None] | None = None,
    *,
    include_weight: bool = False,
) -> tuple:
    """A comparable form of one constraint.

    `resolved_ids` maps field paths to resolver output and is supplied for
    predictions. Gold constraints already hold ids, so they need none.
    """
    entity_fields = set(type(constraint).ENTITY_FIELDS)
    skip = set(EXCLUDED_FIELDS)
    if not include_weight:
        skip.add("weight")

    parts: list[tuple[str, Any]] = []
    for name in sorted(type(constraint).model_fields):
        if name in skip:
            continue
        value = getattr(constraint, name, None)
        if name in entity_fields and value is not None:
            if isinstance(value, list):
                ids = tuple(sorted(
                    _resolved_or_raw(resolved_ids, f"{name}[{i}]", item)
                    for i, item in enumerate(value)
                ))
                parts.append((name, ids))
            else:
                parts.append((name, _resolved_or_raw(resolved_ids, name, value)))
        else:
            parts.append((name, _norm_value(value)))
    return (constraint.type, tuple(parts))


def _resolved_or_raw(
    resolved_ids: dict[str, str | None] | None, path: str, raw: str
) -> str | None:
    """Gold ids and resolver ids must pass through the same normalisation.

    Without this, a gold `L001` (casefolded here) never equals a resolved
    `L001` (not casefolded), and every constraint carrying an entity fails
    exact match while type and entity accuracy both read 100%.
    """
    if resolved_ids is None:
        return _norm_scalar(raw)
    got = resolved_ids.get(path)
    return _norm_scalar(got) if got is not None else None


# -- alignment -----------------------------------------------------------


def _match_score(pred: Any, gold: Any, pred_ids: dict[str, str | None] | None) -> float:
    """How well one prediction fits one gold constraint. Higher is better."""
    if pred.type != gold.type:
        return 0.0
    p = dict(canonical(pred, pred_ids)[1])
    g = dict(canonical(gold)[1])
    keys = set(p) | set(g)
    if not keys:
        return 1.0
    agree = sum(1 for k in keys if p.get(k) == g.get(k))
    return 1.0 + agree / len(keys)  # type match is worth more than any field


@dataclass
class Pair:
    pred: Any | None
    gold: Any | None
    pred_ids: dict[str, str | None] | None = None

    @property
    def matched(self) -> bool:
        return self.pred is not None and self.gold is not None

    @property
    def type_correct(self) -> bool:
        return self.matched and self.pred.type == self.gold.type

    @property
    def strength_correct(self) -> bool:
        return self.matched and self.pred.strength == self.gold.strength

    @property
    def fields_correct(self) -> bool:
        return (
            self.matched
            and canonical(self.pred, self.pred_ids) == canonical(self.gold)
        )


def align(
    preds: Sequence[Any],
    golds: Sequence[Any],
    pred_ids: Sequence[dict[str, str | None] | None] | None = None,
) -> list[Pair]:
    """Greedy one-to-one alignment, best score first."""
    ids_list = list(pred_ids) if pred_ids is not None else [None] * len(preds)
    scored: list[tuple[float, int, int]] = []
    for pi, p in enumerate(preds):
        for gi, g in enumerate(golds):
            s = _match_score(p, g, ids_list[pi])
            if s > 0:
                scored.append((s, pi, gi))
    scored.sort(key=lambda t: (-t[0], t[1], t[2]))

    used_p: set[int] = set()
    used_g: set[int] = set()
    pairs: list[Pair] = []
    for _, pi, gi in scored:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        pairs.append(Pair(preds[pi], golds[gi], ids_list[pi]))

    for pi, p in enumerate(preds):
        if pi not in used_p:
            pairs.append(Pair(p, None, ids_list[pi]))
    for gi, g in enumerate(golds):
        if gi not in used_g:
            pairs.append(Pair(None, g, None))
    return pairs


# -- entity accounting ---------------------------------------------------


class EntityError(str, Enum):
    CORRECT = "correct"
    UNKNOWN_NAME = "unknown_name"  # predicted nothing: unresolved reference
    MISRESOLVED = "misresolved"  # resolved, to the wrong entity
    MISSED = "missed"  # gold reference absent from the prediction
    SPURIOUS = "spurious"  # predicted reference with no gold counterpart


@dataclass
class EntityTally:
    """Entity outcomes, split as Phase 4.6 requires."""

    counts: Counter = field(default_factory=Counter)
    descriptive: Counter = field(default_factory=Counter)

    def add(self, outcome: EntityError, *, descriptive: bool) -> None:
        self.counts[outcome.value] += 1
        if descriptive:
            self.descriptive[outcome.value] += 1

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    @property
    def accuracy(self) -> float | None:
        return (
            self.counts[EntityError.CORRECT.value] / self.total
            if self.total
            else None
        )

    @property
    def descriptive_total(self) -> int:
        return sum(self.descriptive.values())

    @property
    def descriptive_accuracy(self) -> float | None:
        return (
            self.descriptive[EntityError.CORRECT.value] / self.descriptive_total
            if self.descriptive_total
            else None
        )

    def merge(self, other: "EntityTally") -> None:
        self.counts.update(other.counts)
        self.descriptive.update(other.descriptive)

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "accuracy": self.accuracy,
            "counts": dict(self.counts),
            "descriptive_total": self.descriptive_total,
            "descriptive_accuracy": self.descriptive_accuracy,
            "descriptive_counts": dict(self.descriptive),
        }


def _descriptive_map(references: Iterable[Any]) -> dict[str, bool]:
    return {r.field_path: r.descriptive for r in references}


def score_entities(pair: Pair, references: Iterable[Any] | None = None) -> EntityTally:
    tally = EntityTally()
    desc = _descriptive_map(references or [])

    if pair.gold is None:
        for path, _, _ in (pair.pred.entity_references() if pair.pred else []):
            tally.add(EntityError.SPURIOUS, descriptive=desc.get(path, False))
        return tally
    if pair.pred is None:
        for _ in pair.gold.entity_references():
            tally.add(EntityError.MISSED, descriptive=False)
        return tally

    pred_paths = {path for path, _, _ in pair.pred.entity_references()}
    gold_refs = list(pair.gold.entity_references())

    for path, _, gold_id in gold_refs:
        descriptive = desc.get(path, False)
        if path not in pred_paths:
            tally.add(EntityError.MISSED, descriptive=descriptive)
            continue
        got = (pair.pred_ids or {}).get(path)
        if got is None:
            tally.add(EntityError.UNKNOWN_NAME, descriptive=descriptive)
        elif _norm_scalar(got) == _norm_scalar(gold_id):
            tally.add(EntityError.CORRECT, descriptive=descriptive)
        else:
            tally.add(EntityError.MISRESOLVED, descriptive=descriptive)

    gold_paths = {path for path, _, _ in gold_refs}
    for path in pred_paths - gold_paths:
        tally.add(EntityError.SPURIOUS, descriptive=desc.get(path, False))
    return tally


# -- per-sentence and aggregate -----------------------------------------


@dataclass
class SentenceScore:
    item_id: str
    text: str
    exact_match: bool
    n_gold: int
    n_pred: int
    pairs: list[Pair] = field(default_factory=list)
    entities: EntityTally = field(default_factory=EntityTally)
    parse_failed: bool = False
    retried: bool = False
    error: str | None = None

    @property
    def matched_pairs(self) -> list[Pair]:
        return [p for p in self.pairs if p.matched]

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.item_id,
            "text": self.text,
            "exact_match": self.exact_match,
            "n_gold": self.n_gold,
            "n_pred": self.n_pred,
            "parse_failed": self.parse_failed,
            "retried": self.retried,
            "error": self.error,
            "type_correct": sum(p.type_correct for p in self.pairs),
            "strength_correct": sum(p.strength_correct for p in self.pairs),
            "fields_correct": sum(p.fields_correct for p in self.pairs),
            "missed": sum(1 for p in self.pairs if p.pred is None),
            "spurious": sum(1 for p in self.pairs if p.gold is None),
            "entities": self.entities.as_dict(),
        }


def score_sentence(
    item_id: str,
    text: str,
    predicted: Sequence[Any],
    gold: Sequence[Any],
    *,
    references: Sequence[Iterable[Any]] | None = None,
    resolved_ids: Sequence[dict[str, str | None] | None] | None = None,
    include_weight: bool = False,
    parse_failed: bool = False,
    retried: bool = False,
    error: str | None = None,
) -> SentenceScore:
    pairs = align(predicted, gold, resolved_ids)

    pred_canon = Counter(
        canonical(p, (resolved_ids or [None] * len(predicted))[i],
                  include_weight=include_weight)
        for i, p in enumerate(predicted)
    )
    gold_canon = Counter(canonical(g, include_weight=include_weight) for g in gold)
    exact = pred_canon == gold_canon and not parse_failed

    entities = EntityTally()
    ref_by_index = list(references) if references is not None else None
    for pair in pairs:
        refs: Iterable[Any] = []
        if pair.pred is not None and ref_by_index is not None:
            idx = next((i for i, p in enumerate(predicted) if p is pair.pred), None)
            if idx is not None:
                refs = ref_by_index[idx]
        entities.merge(score_entities(pair, refs))

    return SentenceScore(
        item_id=item_id,
        text=text,
        exact_match=exact,
        n_gold=len(gold),
        n_pred=len(predicted),
        pairs=pairs,
        entities=entities,
        parse_failed=parse_failed,
        retried=retried,
        error=error,
    )


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def aggregate(scores: Sequence[SentenceScore]) -> dict[str, Any]:
    """Headline table for Phase 6A."""
    n = len(scores)
    matched = [p for s in scores for p in s.pairs if p.matched]
    all_pairs = [p for s in scores for p in s.pairs]

    entities = EntityTally()
    for s in scores:
        entities.merge(s.entities)

    type_den = sum(s.n_gold for s in scores)
    type_num = sum(p.type_correct for p in all_pairs)
    strength_num = sum(1 for p in matched if p.type_correct and p.strength_correct)
    strength_den = sum(1 for p in matched if p.type_correct)

    open_pred = sum(
        1 for s in scores for p in s.pairs
        if p.pred is not None and p.pred.type in OPEN_TYPES
    )
    open_gold = sum(
        1 for s in scores for p in s.pairs
        if p.gold is not None and p.gold.type in OPEN_TYPES
    )

    return {
        "n_sentences": n,
        "n_gold_constraints": type_den,
        "n_predicted_constraints": sum(s.n_pred for s in scores),
        "parse_failures": sum(s.parse_failed for s in scores),
        "retries": sum(s.retried for s in scores),
        "exact_match": _ratio(sum(s.exact_match for s in scores), n),
        "type_accuracy": _ratio(type_num, type_den),
        "strength_accuracy": _ratio(strength_num, strength_den),
        "field_accuracy": _ratio(sum(p.fields_correct for p in all_pairs), type_den),
        "missed_constraints": sum(1 for p in all_pairs if p.pred is None),
        "spurious_constraints": sum(1 for p in all_pairs if p.gold is None),
        "entity": entities.as_dict(),
        "open_category_predicted": open_pred,
        "open_category_gold": open_gold,
        "confusion": type_confusion(scores),
    }


def type_confusion(scores: Sequence[SentenceScore]) -> dict[str, dict[str, int]]:
    """Gold type -> predicted type counts, for the error-category breakdown."""
    table: dict[str, dict[str, int]] = {}
    for s in scores:
        for p in s.pairs:
            gold_t = p.gold.type if p.gold is not None else "<none>"
            pred_t = p.pred.type if p.pred is not None else "<none>"
            table.setdefault(gold_t, {}).setdefault(pred_t, 0)
            table[gold_t][pred_t] += 1
    return table


# -- inter-rater agreement (Phase 6A) ------------------------------------


def cohen_kappa(labels_a: Sequence[str], labels_b: Sequence[str]) -> float:
    """Cohen's kappa for two labellers over the same items.

    Falls back to a direct computation when scikit-learn is absent, so the
    harness has no hard dependency on it.
    """
    if len(labels_a) != len(labels_b):
        raise ValueError("labellers must cover the same items")
    if not labels_a:
        raise ValueError("no items to compare")
    try:
        from sklearn.metrics import cohen_kappa_score

        return float(cohen_kappa_score(list(labels_a), list(labels_b)))
    except ModuleNotFoundError:  # pragma: no cover - env dependent
        pass

    n = len(labels_a)
    observed = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n
    ca, cb = Counter(labels_a), Counter(labels_b)
    expected = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1 - expected)


def agreement_report(
    labels_a: Sequence[Sequence[Any]],
    labels_b: Sequence[Sequence[Any]],
    *,
    include_weight: bool = False,
) -> dict[str, Any]:
    """Inter-rater agreement over two labellers' constraint sets.

    Unit of analysis, stated because kappa needs one category per item: the
    **primary** (first) constraint of each sentence supplies the type and
    strength labels. `multi_constraint_sentences` reports how often that
    discards a label, so the reader can judge the cost. Set-level exact
    agreement is reported alongside and uses every constraint.
    """
    if len(labels_a) != len(labels_b):
        raise ValueError("labellers must cover the same items")

    def primary_type(cs: Sequence[Any]) -> str:
        return cs[0].type if cs else "<none>"

    def primary_strength(cs: Sequence[Any]) -> str:
        return cs[0].strength if cs else "<none>"

    types_a = [primary_type(c) for c in labels_a]
    types_b = [primary_type(c) for c in labels_b]
    str_a = [primary_strength(c) for c in labels_a]
    str_b = [primary_strength(c) for c in labels_b]

    exact = sum(
        Counter(canonical(c, include_weight=include_weight) for c in a)
        == Counter(canonical(c, include_weight=include_weight) for c in b)
        for a, b in zip(labels_a, labels_b)
    )
    multi = sum(1 for a, b in zip(labels_a, labels_b) if len(a) > 1 or len(b) > 1)

    return {
        "n_items": len(labels_a),
        "type_kappa": cohen_kappa(types_a, types_b),
        "strength_kappa": cohen_kappa(str_a, str_b),
        "type_raw_agreement": _ratio(
            sum(1 for a, b in zip(types_a, types_b) if a == b), len(types_a)
        ),
        "strength_raw_agreement": _ratio(
            sum(1 for a, b in zip(str_a, str_b) if a == b), len(str_a)
        ),
        "set_exact_agreement": _ratio(exact, len(labels_a)),
        "multi_constraint_sentences": multi,
        "unit_of_analysis": "primary (first) constraint per sentence",
    }


__all__ = [
    "ALL_TYPES",
    "EXCLUDED_FIELDS",
    "EntityError",
    "EntityTally",
    "Pair",
    "SentenceScore",
    "aggregate",
    "agreement_report",
    "align",
    "canonical",
    "cohen_kappa",
    "score_entities",
    "score_sentence",
    "type_confusion",
]
