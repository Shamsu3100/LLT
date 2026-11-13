"""Synthetic study logs (build plan Phase 6C).

    python -m timetabler.eval.synth --participants 30 --out logs/synthetic.sqlite

**What this is for, and what it is not for.**

A synthetic log exercises the whole chain — recorder, store, export, scoring —
at the real shape of the study, so the analysis plan can be dry-run before a
single participant is recruited. That is worth doing: the pre-registration
commits to an analysis, and an analysis that has never been executed is a
guess.

It tells you **nothing about the result**. The effect is whatever
`--time-reduction` and `--error-rate` put there, so finding it again is
circular. Three uses are legitimate:

1. **Pipeline validation.** The columns are right, the exclusion logic fires,
   the export is clean, the scoring aligns.
2. **A null check.** Generate with `--time-reduction 0` and confirm the
   analysis does *not* report a difference. A pipeline that finds an effect in
   null data is broken, and this is the cheapest way to catch it.
3. **A power sanity check.** Generate at the target effect and confirm the
   analysis detects it at the planned sample size. Crude, since the
   between-participant variance is invented, but it catches a plan that cannot
   detect its own target.

What it cannot reproduce: real between-participant variance, which is what
actually determines power; which constraints humans misread, and why; learning
across conditions; and the correlation between a participant's speed and their
accuracy. Treat every number it produces as a shape, not an estimate.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

from ..app.recorder import StudyRecorder
from ..app.store import Condition, Phase, StudyStore
from ..data import load_department
from ..schema.constraints import ConstraintSet
from .tasks import Task, load_tasks

#: Plausible shapes, not estimates. Each is a knob the caller should move.
#:
#: Rates are **per intended constraint**, so a task of eight takes longer than
#: a task of three. An absolute per-session total would make the generated
#: effect depend on which task a participant drew, which is not how the study
#: behaves.
DEFAULTS = {
    "form_seconds_per_constraint": 35.0,       # choose type, fill fields, add
    "llm_entry_seconds_per_constraint": 12.0,  # write a sentence
    "llm_verify_seconds_per_constraint": 10.0, # read it back, accept or edit
    "participant_spread": 0.35,   # multiplicative, per participant
    "task_spread": 0.15,          # multiplicative, per session
}


def _jitter(rng: random.Random, value: float, spread: float) -> float:
    return max(value * (1.0 + rng.gauss(0.0, spread)), 1.0)


def _degrade(
    intended: list[Any], rng: random.Random, error_rate: float
) -> list[Any]:
    """Return a submission with roughly `error_rate` of the key wrong.

    Four failure modes, drawn uniformly: drop one, flip a strength, point an
    entity somewhere else, or invent one. These are shapes taken from the
    error categories the scorer reports, not measured frequencies.
    """
    out = [c.model_copy(deep=True) for c in intended]
    for i in range(len(out)):
        if rng.random() >= error_rate:
            continue
        mode = rng.choice(["drop", "strength", "entity", "extra"])
        data = out[i].model_dump()

        if mode == "drop":
            out[i] = None
        elif mode == "strength" and data.get("strength") == "soft":
            data["strength"] = "hard"
            data["weight"] = None
            out[i] = type(intended[i]).model_validate(data)
        elif mode == "strength":
            data["strength"] = "soft"
            data["weight"] = 5
            out[i] = type(intended[i]).model_validate(data)
        elif mode == "entity":
            for field in ("lecturer", "course", "room", "group"):
                if data.get(field):
                    data[field] = "L999" if field == "lecturer" else data[field]
                    break
            try:
                out[i] = type(intended[i]).model_validate(data)
            except Exception:
                pass
        elif mode == "extra":
            out.append(out[i].model_copy(deep=True))
    return [c for c in out if c is not None]


def generate(
    db_path: str | Path,
    *,
    participants: int = 30,
    seed: int = 0,
    time_reduction: float = 0.30,
    error_rate: float = 0.10,
    abandon_rate: float = 0.03,
    tasks_path: str | None = None,
    department_path: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write a synthetic within-subject study to `db_path`."""
    path = Path(db_path)
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"{path} exists. Pass overwrite=True, or choose another path; "
                f"a real study log must never be overwritten."
            )
        path.unlink()

    rng = random.Random(seed)
    department = load_department(department_path)
    task_set = load_tasks(tasks_path)
    if not task_set.tasks:
        raise ValueError("task set is empty")

    form_rate = DEFAULTS["form_seconds_per_constraint"]
    entry_share = DEFAULTS["llm_entry_seconds_per_constraint"] / (
        DEFAULTS["llm_entry_seconds_per_constraint"]
        + DEFAULTS["llm_verify_seconds_per_constraint"]
    )

    stats: dict[str, Any] = {
        "sessions": 0, "abandoned": 0, "participants": participants,
        "tasks": len(task_set.tasks), "intended_total": 0,
    }

    with StudyStore(path) as store, store.batch():
        for index in range(1, participants + 1):
            pid = f"P{index:02d}"
            speed = 1.0 + rng.gauss(0.0, DEFAULTS["participant_spread"])
            speed = max(speed, 0.4)
            task = task_set.tasks[(index - 1) % len(task_set.tasks)]
            # Counterbalanced: odd participants see the form first.
            order = ["form", "llm"] if index % 2 else ["llm", "form"]

            for position, arm in enumerate(order, start=1):
                if rng.random() < abandon_rate:
                    _abandon(store, pid, arm, task, department, position, rng)
                    stats["abandoned"] += 1
                    stats["sessions"] += 1
                    continue
                # Both arms are sized from the same task, so the effect
                # is the requested one whichever task was drawn.
                form_total = form_rate * task.size
                if arm == "form":
                    _form_session(
                        store, pid, task, department, position, rng, speed,
                        form_total, error_rate,
                    )
                else:
                    _llm_session(
                        store, pid, task, department, position, rng, speed,
                        form_total * (1.0 - time_reduction), entry_share,
                        error_rate,
                    )
                stats["sessions"] += 1
                stats["intended_total"] += task.size

    stats["path"] = str(path)
    return stats


def _submission(task: Task, rng: random.Random, error_rate: float) -> list[Any]:
    return _degrade(list(task.intended), rng, error_rate)


def _form_session(
    store, pid, task, department, position, rng, speed, base, error_rate
) -> None:
    rec = StudyRecorder(
        store, pid, Condition.FORM, task=task.id, department=department,
        order_index=position,
    )
    duration = _jitter(rng, base * speed, DEFAULTS["task_spread"])
    rec.enter(Phase.ENTRY, at=0.0)
    submitted = _submission(task, rng, error_rate)
    step = duration / max(len(submitted), 1)
    for i, constraint in enumerate(submitted, start=1):
        rec.record_form_add(constraint, at=min(step * i, duration - 1))
    rec.complete(at=duration)


def _llm_session(
    store, pid, task, department, position, rng, speed, total, entry_share,
    error_rate,
) -> None:
    rec = StudyRecorder(
        store, pid, Condition.LLM, task=task.id, department=department,
        model="claude-sonnet-5-5", order_index=position,
    )
    duration = _jitter(rng, total * speed, DEFAULTS["task_spread"])
    entry_end = duration * entry_share

    rec.enter(Phase.ENTRY, at=0.0)
    rec.record_text_submission(
        " ".join(task.brief)[:200], at=max(entry_end - 5.0, 1.0)
    )
    proposed = _submission(task, rng, error_rate)
    ids = rec.record_parse("synthetic", proposed, at=entry_end)

    rec.enter(Phase.VERIFICATION, at=entry_end)
    step = (duration - entry_end) / max(len(ids), 1)
    for i, (cid, constraint) in enumerate(zip(ids, proposed), start=1):
        at = entry_end + step * i
        roll = rng.random()
        if roll < 0.12:
            rec.reject(cid, reason="misread", at=at)
        elif roll < 0.30:
            rec.edit(cid, constraint, changed=["strength"], at=at)
        else:
            rec.accept(cid, constraint, at=at)
    rec.complete(at=duration)


def _abandon(store, pid, arm, task, department, position, rng) -> None:
    """A session that was started and never finished.

    Included deliberately: the export's exclusion logic only earns trust if
    something in the data exercises it.
    """
    condition = Condition.FORM if arm == "form" else Condition.LLM
    rec = StudyRecorder(
        store, pid, condition, task=task.id, department=department,
        order_index=position,
    )
    rec.enter(Phase.ENTRY, at=0.0)
    rec.record_text_submission("started and stopped", at=rng.uniform(5.0, 40.0))
    # No complete(): the entry phase stays open, as it would in the field.


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Generate a synthetic study log for dry-running the analysis."
    )
    ap.add_argument("--out", default="logs/synthetic.sqlite")
    ap.add_argument("--participants", type=int, default=30,
                    help="Build plan Phase 6C targets 25-30.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--time-reduction", type=float, default=0.30,
                    help="Fraction by which the LLM arm is faster. The primary "
                         "outcome's target is 0.30. Use 0 for a null check.")
    ap.add_argument("--error-rate", type=float, default=0.10,
                    help="Fraction of each task key submitted wrongly.")
    ap.add_argument("--abandon-rate", type=float, default=0.03,
                    help="Fraction of sessions left unfinished.")
    ap.add_argument("--tasks", default=None)
    ap.add_argument("--department", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    try:
        stats = generate(
            args.out,
            participants=args.participants,
            seed=args.seed,
            time_reduction=args.time_reduction,
            error_rate=args.error_rate,
            abandon_rate=args.abandon_rate,
            tasks_path=args.tasks,
            department_path=args.department,
            overwrite=args.overwrite,
        )
    except FileExistsError as exc:
        sys.exit(f"\n{exc}\n")

    print(f"\nSynthetic study written to {stats['path']}")
    print(f"  {stats['participants']} participants, {stats['sessions']} sessions, "
          f"{stats['abandoned']} abandoned")
    print(f"  seed {args.seed}, time reduction {args.time_reduction:.0%}, "
          f"error rate {args.error_rate:.0%}")
    print(f"  {stats['intended_total']} intended constraints scored across "
          f"{stats['tasks']} task(s)")
    print("\n  This is shape, not evidence: the effect is the one you asked")
    print("  for, so recovering it proves the pipeline runs, nothing more.")
    print(f"\n  python -m timetabler.eval.export_study --db {stats['path']} \\")
    print("      --tasks timetabler/eval/labelled/sample_tasks.json --out exports/\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
