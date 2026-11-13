"""Department data loading (build plan Phase 1)."""

from __future__ import annotations

import json
from pathlib import Path

from ..schema.entities import Department

DATA_DIR = Path(__file__).parent
SAMPLE_PATH = DATA_DIR / "sample_department.json"


def load_department(path: str | Path | None = None) -> Department:
    """Load and validate a department instance.

    Defaults to the synthetic fixture, which exists for development and tests.
    Phase 1.1 replaces it with real or benchmark data.
    """
    target = Path(path) if path is not None else SAMPLE_PATH
    raw = json.loads(target.read_text(encoding="utf-8"))
    raw.pop("_note", None)
    return Department.model_validate(raw)


__all__ = ["DATA_DIR", "SAMPLE_PATH", "load_department"]
