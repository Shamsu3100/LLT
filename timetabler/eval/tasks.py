"""Task keys and residual-error scoring (build plan Phase 6C).

A **task** is what a participant is asked to express, and a **task key** is the
set of constraints that expresses it. Residual error is the gap between the
set a participant finally accepted and that key. It is the pre-registered
secondary outcome, and the whole non-inferiority check rests on it, so the key
has to exist before the pilot locks the margin.

The Phase 6A labelled set cannot serve here: it labels sentences nobody in the
study wrote. This is a different comparison, against what the task asked for.

**The brief must not be written as the sentences you want back.** If the task
says "You cannot teach on Fridays", a participant in the free-text arm copies
that sentence and a participant in the form arm transcribes it, and the study
measures transcription speed instead of elicitation. Each `brief` line is
therefore a fact, a cell in a grid, or a diary note — something a person has
to put into their own words either way. `presentation` records how the brief
reaches the participant, because a brief read aloud and a brief shown as a
calendar are different instruments.

Scoring reuses the Phase 6A alignment, so a constraint is "correct" under the
same definition in both studies. Form-arm constraints already hold canonical
ids; free-text-arm constraints hold the words staff wrote, so they are
resolved first.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field

from ..parser.resolve import EntityResolver, UnresolvedReferenceError, materialise
from ..schema.constraints import Constraint, ConstraintSet
from ..schema.entities import Department
from .metrics import EntityError, align, canonical, score_entities

TASKS_DIR = Path(__file__).parent / "labelled"
SAMPLE_TASKS_PATH = TASKS_DIR / "sample_tasks.json"

Presentation = Literal["grid", "bullet_facts", "diary", "spoken", "other"]


class Task(BaseModel):
    """One scenario a participant is asked to express."""

    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    #: What the participant is shown. Facts, not sentences to copy.
    brief: list[str] = Field(min_length=1)
    #: How the brief reaches them. Part of the instrument, so it is recorded.
    presentation: Presentation = "bullet_facts"
    #: The constraints that express the brief, with canonical department ids.
    intended: list[Constraint] = Field(min_length=1)
    notes: str | None = None

    @property
    def size(self) -> int:
        return len(self.intended)


class TaskSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    department: str | None = None
    tasks: list[Task]

    def by_id(self, task_id: str) -> Task | None:
        return next((t for t in self.tasks if t.id == task_id), None)

    @property
    def ids(self) -> list[str]:
        return [t.id for t in self.tasks]


def load_tasks(path: str | Path | None = None) -> TaskSet:
    target = Path(path) if path is not None else SAMPLE_TASKS_PATH
    raw = json.loads(target.read_text(encoding="utf-8"))
    raw.pop("_note", None)
    return TaskSet.model_validate(raw)


# -- scoring -------------------------------------------------------------


class ResidualError(BaseModel):
    """What survived verification, against what the task asked for."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    intended: int
    submitted: int
    correct: int
    missed: int
    spurious: int
    wrong_type: int
    wrong_strength: int
    wrong_entity: int
    unresolved: int

    @property
    def error_rate(self) -> float | None:
        """Errors per intended constraint.

        The denominator is what the task asked for, so omitting a constraint
        and inventing one both count. A rate above 1 is possible and means
        exactly what it says.
        """
        if not self.intended:
            return None
        return (self.intended - self.correct + self.spurious) / self.intended

    @property
    def accuracy(self) -> float | None:
        return self.correct / self.intended if self.intended else None

    def as_dict(self) -> dict[str, Any]:
        out = self.model_dump()
        out["error_rate"] = self.error_rate
        out["accuracy"] = self.accuracy
        return out


def _resolved(
    constraints: Iterable[Any], resolver: EntityResolver
) -> tuple[list[Any], int]:
    """Rewrite entity fields as canonical ids, counting what cannot resolve.

    The form arm already submits ids, so resolution is a no-op there. The
    free-text arm submits the words staff wrote. An unresolvable reference is
    an error in its own right, not a crash: the participant accepted a
    constraint naming something the register does not contain.
    """
    out: list[Any] = []
    unresolved = 0
    for constraint in constraints:
        try:
            out.append(materialise(resolver.resolve(constraint)))
        except UnresolvedReferenceError:
            unresolved += 1
    return out, unresolved


def score_session(
    submitted: Sequence[dict[str, Any]] | Sequence[Any],
    task: Task,
    department: Department,
) -> ResidualError:
    """Residual error for one session against its task key."""
    if submitted and isinstance(submitted[0], dict):
        parsed = ConstraintSet.model_validate(
            {"constraints": list(submitted)}
        ).constraints
    else:
        parsed = list(submitted)

    resolver = EntityResolver(department)
    resolved, unresolved = _resolved(parsed, resolver)

    # Both sides now hold canonical ids, so each prediction's own fields are
    # its resolution report. Without this, `score_entities` finds no ids to
    # compare and counts every reference as unresolved, which showed up as
    # wrong_entity equal to the reference count on a perfect submission.
    pred_ids = [
        {path: raw for path, _, raw in c.entity_references()} for c in resolved
    ]
    pairs = align(resolved, task.intended, pred_ids)
    correct = sum(1 for p in pairs if p.fields_correct)
    missed = sum(1 for p in pairs if p.pred is None)
    spurious = sum(1 for p in pairs if p.gold is None)
    wrong_type = sum(
        1 for p in pairs if p.matched and not p.type_correct
    )
    wrong_strength = sum(
        1 for p in pairs
        if p.matched and p.type_correct and not p.strength_correct
    )

    wrong_entity = 0
    for pair in pairs:
        if not pair.matched:
            continue
        tally = score_entities(pair)
        wrong_entity += (
            tally.counts[EntityError.MISRESOLVED.value]
            + tally.counts[EntityError.UNKNOWN_NAME.value]
        )

    return ResidualError(
        task_id=task.id,
        intended=task.size,
        submitted=len(parsed),
        correct=correct,
        missed=missed,
        spurious=spurious,
        wrong_type=wrong_type,
        wrong_strength=wrong_strength,
        wrong_entity=wrong_entity,
        unresolved=unresolved,
    )


def validate_against_department(
    tasks: TaskSet, department: Department
) -> list[str]:
    """Problems that would make a task unscoreable. Empty means usable.

    Run this before the pilot: a task key naming an entity the department does
    not contain scores every participant wrong for a reason that is the
    researcher's, not theirs.
    """
    resolver = EntityResolver(department)
    problems: list[str] = []
    for task in tasks.tasks:
        if not task.intended:
            problems.append(f"{task.id}: no intended constraints")
        for i, constraint in enumerate(task.intended):
            for path, kind, raw in constraint.entity_references():
                ref = resolver.resolve_reference(path, kind, raw)
                if ref.resolved_id != raw:
                    problems.append(
                        f"{task.id}[{i}].{path}: {raw!r} is not a canonical "
                        f"{kind} id in {department.name} "
                        f"(resolved to {ref.resolved_id!r})"
                    )
    return problems


__all__ = [
    "SAMPLE_TASKS_PATH",
    "Presentation",
    "ResidualError",
    "Task",
    "TaskSet",
    "load_tasks",
    "score_session",
    "validate_against_department",
]
