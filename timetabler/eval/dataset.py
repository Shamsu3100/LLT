"""Labelled-set handling (build plan Phase 6A).

One JSONL record per staff sentence. `gold` is the adjudicated reading;
`labels` holds each labeller's independent pass, which is what inter-rater
agreement is computed from. Entity fields in both hold canonical department
ids, so a label does not depend on how the resolver happens to behave.

`source` separates provenance, and the distinction is load-bearing: the
quantitative claim in Phase 6A rests on `staff_authored` items. Synthetic
items exist for development and for testing this harness.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..schema.constraints import Constraint

LABELLED_DIR = Path(__file__).parent / "labelled"
SEED_PATH = LABELLED_DIR / "seed.jsonl"

Provenance = Literal["staff_authored", "staff_derived", "synthetic"]


class LabelledItem(BaseModel):
    """One sentence and its labels."""

    model_config = ConfigDict(extra="forbid")

    id: str
    text: str
    gold: list[Constraint] = Field(default_factory=list)
    labels: dict[str, list[Constraint]] = Field(default_factory=dict)
    source: Provenance = "synthetic"
    notes: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)

    @property
    def labeller_ids(self) -> list[str]:
        return sorted(self.labels)


def load_items(path: str | Path | None = None) -> list[LabelledItem]:
    """Read a JSONL labelled set, skipping blank lines and `//` comments."""
    target = Path(path) if path is not None else SEED_PATH
    items: list[LabelledItem] = []
    for lineno, line in enumerate(target.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        try:
            items.append(LabelledItem.model_validate_json(stripped))
        except Exception as exc:
            raise ValueError(f"{target.name}:{lineno}: {exc}") from exc

    seen: set[str] = set()
    for item in items:
        if item.id in seen:
            raise ValueError(f"{target.name}: duplicate item id {item.id!r}")
        seen.add(item.id)
    return items


def save_items(items: list[LabelledItem], path: str | Path) -> None:
    lines = [
        json.dumps(item.model_dump(mode="json", exclude_none=False), ensure_ascii=False)
        for item in items
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def iter_texts(items: list[LabelledItem]) -> Iterator[str]:
    for item in items:
        yield item.text


def paired_labels(
    items: list[LabelledItem], a: str, b: str
) -> tuple[list[list[Any]], list[list[Any]], list[str]]:
    """Labels from two labellers over the items both of them covered."""
    la: list[list[Any]] = []
    lb: list[list[Any]] = []
    covered: list[str] = []
    for item in items:
        if a in item.labels and b in item.labels:
            la.append(item.labels[a])
            lb.append(item.labels[b])
            covered.append(item.id)
    return la, lb, covered


def describe(items: list[LabelledItem]) -> dict[str, Any]:
    from collections import Counter

    by_source = Counter(i.source for i in items)
    by_type = Counter(c.type for i in items for c in i.gold)
    return {
        "n_items": len(items),
        "n_gold_constraints": sum(len(i.gold) for i in items),
        "multi_constraint_items": sum(1 for i in items if len(i.gold) > 1),
        "by_source": dict(by_source),
        "by_gold_type": dict(by_type),
        "labellers": sorted({k for i in items for k in i.labels}),
        "staff_authored": by_source.get("staff_authored", 0),
    }


__all__ = [
    "LABELLED_DIR",
    "SEED_PATH",
    "LabelledItem",
    "Provenance",
    "describe",
    "iter_texts",
    "load_items",
    "paired_labels",
    "save_items",
]
