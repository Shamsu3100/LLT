"""Run the solver from the command line (build plan Phase 3).

    python -m timetabler.solver.run
    python -m timetabler.solver.run --iterations 60000 --restarts 3
    python -m timetabler.solver.run --constraints none

Constraints come from the gold labels of a labelled set by default, which
exercises the whole chain: labelled sentence -> resolver -> materialise ->
cost model. Anything carrying a blocking flag is dropped, exactly as the
verification step would drop it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..data import load_department
from ..eval.dataset import load_items
from ..parser.resolve import EntityResolver, materialise_all
from ..schema.constraints import SOLVABLE_TYPES
from .annealing import solve, solve_best_of
from .problem import ProblemIndex


def gather_constraints(department: Any, source: str, path: str | None) -> list[Any]:
    if source == "none":
        return []
    resolver = EntityResolver(department)
    items = load_items(path)
    gold = [c for item in items for c in item.gold if c.type in SOLVABLE_TYPES]
    return materialise_all(resolver.resolve(c) for c in gold)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Solve a timetable.")
    ap.add_argument("--department", help="Department JSON (default: sample)")
    ap.add_argument("--constraints", default="labelled",
                    choices=["labelled", "none"],
                    help="Elicited constraints to load (default: labelled)")
    ap.add_argument("--labelled", help="Labelled set JSONL (default: seed.jsonl)")
    ap.add_argument("--iterations", type=int, default=40_000)
    ap.add_argument("--restarts", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", help="Write the result as JSON here")
    args = ap.parse_args(argv)

    department = load_department(args.department)
    index = ProblemIndex(department)
    constraints = gather_constraints(department, args.constraints, args.labelled)

    hard = sum(1 for c in constraints if c.strength == "hard")
    print(f"\n{department.name}")
    print(f"  {len(index.course_ids)} courses, {len(index.room_ids)} rooms, "
          f"{len(department.slots)} slots")
    print(f"  {len(constraints)} elicited constraints "
          f"({hard} hard, {len(constraints) - hard} soft)")

    if args.restarts > 1:
        result = solve_best_of(
            index, constraints, restarts=args.restarts,
            iterations=args.iterations, seed=args.seed,
        )
    else:
        result = solve(
            index, constraints, iterations=args.iterations, seed=args.seed
        )

    print(f"\n{result.summary()}\n")
    print(result.timetable.render())

    if result.breakdown.violated_hard:
        print("\n  Hard constraints that could not be satisfied together.")
        print("  Phase 5.3: report these and ask staff to relax one.")
        for row in result.breakdown.violated_hard:
            print(f"    x{row['count']:<3}{row['type']:<28}{row['source_text'][:46]}")

    if result.breakdown.violated_soft:
        print("\n  Breached soft constraints:")
        for row in sorted(result.breakdown.violated_soft,
                          key=lambda r: -r["contribution"]):
            print(f"    w{row['weight']:<3}x{row['count']:<3}"
                  f"= {row['contribution']:<6.0f}{row['source_text'][:44]}")
    print()

    if args.out:
        Path(args.out).write_text(
            json.dumps(result.as_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"result written to {args.out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
