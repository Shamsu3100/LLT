"""Offline accuracy run (build plan Phase 6A).

    python -m timetabler.eval.run_accuracy --client gold
    python -m timetabler.eval.run_accuracy --client anthropic --out report.json

Backends:

* ``gold`` replays the gold labels as predictions. A harness self-test: every
  accuracy figure should come back at 1.0, and anything less is a defect in
  the metrics or the resolver, not in a model.
* ``recorded`` replays a JSON map of sentence -> raw response, so a scored run
  is reproducible without re-calling the API.
* ``anthropic`` calls the live API. Needs ``pip install anthropic`` and
  ``ANTHROPIC_API_KEY``, and the provider's data-retention policy must be held
  in writing before any staff text is sent (§5, Privacy).

Every run records model, effort, prompt digest, timestamps, token usage and
estimated cost. ``--save-responses`` writes the raw replies so a later
re-score needs no further calls, which is what reproducibility rests on now
that the API accepts no sampling parameters to pin.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ..data import load_department
from ..parser.client import (
    ClientError,
    StubClient,
    describe_providers,
    make_client,
    parse_spec,
)
from ..parser.parse import ConstraintParser
from .dataset import LabelledItem, describe, load_items, paired_labels
from .metrics import aggregate, agreement_report, score_sentence


def build_client(kind: str, items: list[LabelledItem], args: argparse.Namespace):
    if kind == "gold":
        stub = StubClient(model="replay-gold", on_miss="empty")
        for item in items:
            stub.add(item.text, {
                "constraints": [c.model_dump(mode="json") for c in item.gold]
            })
        return stub
    if kind == "recorded":
        if not args.responses:
            sys.exit("--client recorded needs --responses FILE")
        recorded = json.loads(Path(args.responses).read_text(encoding="utf-8"))
        stub = StubClient(model=recorded.get("_model", "recorded"), on_miss="empty")
        for text, response in recorded.get("responses", {}).items():
            stub.add(text, response)
        return stub
    # Anything else is a live provider spec: "anthropic", "openai:<model>",
    # "deepseek:<model>", "ollama:<model>", ...
    options: dict[str, Any] = {}
    if args.effort:
        options["effort"] = args.effort
    if args.temperature is not None:
        options["temperature"] = args.temperature
    if args.json_mode:
        options["json_mode"] = True
    if args.base_url:
        options["base_url"] = args.base_url

    client = make_client(kind, **options)
    if not args.no_ping:
        # One cheap call first: a bad model id or a billing problem should
        # surface here, not a third of the way through a 300-item run.
        meta = client.ping()
        print(f"  connected: {meta.provider}/{meta.model}"
              + (f", effort {meta.effort}" if meta.effort else "")
              + f", {meta.input_tokens}+{meta.output_tokens} tokens")
    return client


def run(args: argparse.Namespace) -> dict[str, Any]:
    department = load_department(args.department)
    items = load_items(args.labelled)
    if args.only_staff:
        items = [i for i in items if i.source == "staff_authored"]
        if not items:
            sys.exit("no staff_authored items in the labelled set")
    if args.limit:
        items = items[: args.limit]

    client = build_client(args.client, items, args)
    parser = ConstraintParser(department, client)

    scores = []
    responses: dict[str, str] = {}
    spend = {"input": 0, "output": 0, "cache_read": 0, "cost_usd": 0.0}
    for item in items:
        result = parser.parse(item.text)
        for key, value in result.tokens.items():
            spend[key] += value
        spend["cost_usd"] += result.cost_usd
        if result.attempts:
            responses[item.text] = result.attempts[-1].response_text

        predicted = [rc.constraint for rc in result.constraints]
        scores.append(score_sentence(
            item.id,
            item.text,
            predicted,
            item.gold,
            references=[rc.references for rc in result.constraints],
            resolved_ids=[rc.resolved_ids for rc in result.constraints],
            include_weight=args.include_weight,
            parse_failed=not result.ok,
            retried=result.retried,
            error=result.error,
        ))

    summary = aggregate(scores)

    agreement = None
    labellers = sorted({k for i in items for k in i.labels})
    if len(labellers) >= 2:
        a, b, covered = paired_labels(items, labellers[0], labellers[1])
        if a:
            agreement = agreement_report(a, b, include_weight=args.include_weight)
            agreement["labellers"] = [labellers[0], labellers[1]]
            agreement["items_covered"] = len(covered)

    report = {
        "run": {
            "client": args.client,
            "provider": getattr(client, "provider", None),
            "model": getattr(client, "model", "?"),
            "effort": getattr(client, "effort", None),
            "temperature": getattr(client, "temperature", None),
            "json_mode": getattr(client, "json_mode", False),
            "prompt_hash": parser.prompt_hash,
            "include_weight": args.include_weight,
            "department": department.name,
            "labelled_set": str(args.labelled or "seed.jsonl"),
            "spec": args.client,
        },
        "dataset": describe(items),
        "spend": spend,
        "accuracy": summary,
        "agreement": agreement,
        "per_sentence": [s.as_dict() for s in scores],
    }

    if args.save_responses:
        Path(args.save_responses).write_text(
            json.dumps(
                {"_model": getattr(client, "model", "?"), "responses": responses},
                indent=2, ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    return report


def _pct(value: float | None) -> str:
    return "   n/a" if value is None else f"{value * 100:5.1f}%"


def print_report(report: dict[str, Any]) -> None:
    run_info = report["run"]
    data = report["dataset"]
    acc = report["accuracy"]
    ent = acc["entity"]

    print(f"\nPhase 6A translation accuracy")
    provider = run_info.get("provider") or run_info["client"]
    print(f"  {provider}/{run_info['model']}  prompt {run_info['prompt_hash']}"
          + (f"  effort {run_info['effort']}" if run_info.get("effort") else "")
          + ("  json-mode" if run_info.get("json_mode") else "")
          + (f"  temp {run_info['temperature']}"
             if run_info.get("temperature") is not None else ""))
    print(f"  {data['n_items']} sentences, {data['n_gold_constraints']} gold "
          f"constraints, {data['staff_authored']} staff-authored")
    if data["staff_authored"] == 0:
        print("  NOTE: no staff-authored items. Phase 6A's quantitative claim "
              "needs human-authored text.")

    print(f"\n  {'measure':<26}{'score':>8}   detail")
    print(f"  {'-' * 26}{'-' * 8}   {'-' * 34}")
    rows = [
        ("exact match (sentence)", acc["exact_match"],
         f"{acc['n_sentences']} sentences"),
        ("type accuracy", acc["type_accuracy"],
         f"over {acc['n_gold_constraints']} gold constraints"),
        ("hard/soft accuracy", acc["strength_accuracy"],
         "where the type is correct"),
        ("all fields correct", acc["field_accuracy"],
         f"over {acc['n_gold_constraints']} gold constraints"),
        ("entity accuracy", ent["accuracy"], f"{ent['total']} references"),
        ("  of descriptive refs", ent["descriptive_accuracy"],
         f"{ent['descriptive_total']} references"),
    ]
    for label, value, detail in rows:
        print(f"  {label:<26}{_pct(value):>8}   {detail}")

    print(f"\n  parse failures      {acc['parse_failures']}")
    print(f"  retries             {acc['retries']}")
    print(f"  missed constraints  {acc['missed_constraints']}")
    print(f"  spurious            {acc['spurious_constraints']}")
    print(f"  open categories     predicted {acc['open_category_predicted']}, "
          f"gold {acc['open_category_gold']}")

    if ent["total"]:
        breakdown = ", ".join(
            f"{k}={v}" for k, v in sorted(ent["counts"].items()) if k != "correct"
        )
        print(f"  entity errors       {breakdown or 'none'}")

    ag = report.get("agreement")
    if ag:
        print(f"\n  Inter-rater agreement ({'/'.join(ag['labellers'])}, "
              f"{ag['items_covered']} items)")
        print(f"    type kappa        {ag['type_kappa']:.3f}  "
              f"(raw {_pct(ag['type_raw_agreement']).strip()})")
        print(f"    strength kappa    {ag['strength_kappa']:.3f}  "
              f"(raw {_pct(ag['strength_raw_agreement']).strip()})")
        print(f"    set exact agree   {_pct(ag['set_exact_agreement']).strip()}")
        print(f"    unit: {ag['unit_of_analysis']}")

    confusion = {
        g: {p: n for p, n in preds.items() if p != g}
        for g, preds in acc["confusion"].items()
    }
    confusion = {g: v for g, v in confusion.items() if v}
    if confusion:
        print("\n  Type confusion (gold -> predicted)")
        for gold_type, preds in sorted(confusion.items()):
            for pred_type, n in sorted(preds.items(), key=lambda kv: -kv[1]):
                print(f"    {gold_type:<28} -> {pred_type:<28} {n}")
    print()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Score LLM constraint translation against a labelled set."
    )
    ap.add_argument("--client", default="gold", metavar="SPEC",
                    help="'gold', 'recorded', or a provider spec such as "
                         "anthropic:claude-sonnet-5-5 / openai:<model> / "
                         "deepseek:<model>. See --list-providers.")
    ap.add_argument("--list-providers", action="store_true",
                    help="Show every provider and the variable it reads.")
    ap.add_argument("--list-models", action="store_true",
                    help="Ask the provider which models this credential can "
                         "reach, then exit.")
    ap.add_argument("--labelled", help="JSONL labelled set (default: seed.jsonl)")
    ap.add_argument("--department", help="Department JSON (default: sample)")
    ap.add_argument("--effort", default=None,
                    help="Reasoning depth, where the provider has one "
                         "(Claude: low|medium|high|xhigh|max). A Phase 6A "
                         "sweep axis.")
    ap.add_argument("--temperature", type=float, default=None,
                    help="Sampling temperature. NOT accepted by current Claude "
                         "models, so a run that sets it is not comparable to a "
                         "Claude run.")
    ap.add_argument("--json-mode", action="store_true",
                    help="Use the provider's native JSON mode. Off by default: "
                         "the Claude arm does not use it, and an asymmetric "
                         "instrument measures the harness, not the model.")
    ap.add_argument("--base-url", help="Override the provider endpoint "
                                       "(self-hosted or proxy).")
    ap.add_argument("--no-ping", action="store_true",
                    help="Skip the connectivity check before a live run.")
    ap.add_argument("--responses", help="Recorded responses for --client recorded")
    ap.add_argument("--save-responses", help="Write raw replies here")
    ap.add_argument("--out", help="Write the full JSON report here")
    ap.add_argument("--limit", type=int, help="Score only the first N items")
    ap.add_argument("--only-staff", action="store_true",
                    help="Score only staff_authored items")
    ap.add_argument("--include-weight", action="store_true",
                    help="Score soft weights as part of a correct reading")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    if args.list_providers:
        print("\nProviders (spec is provider:model)\n")
        print(describe_providers())
        print()
        return 0

    if args.list_models:
        try:
            client = make_client(args.client, base_url=args.base_url)
            for name in client.list_models():
                print(name)
        except ClientError as exc:
            sys.exit(f"\n{exc}\n")
        return 0

    try:
        report = run(args)
    except ClientError as exc:
        # A credential, model or connectivity problem is the operator's to
        # fix; a traceback buries the one line that says how.
        sys.exit(f"\n{exc}\n")
    if not args.quiet:
        print_report(report)
    if args.out:
        Path(args.out).write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"report written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
