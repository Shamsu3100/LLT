"""Deriving durations from the event log (build plan Phase 5.4).

Entry time and verification time are reported **separately** and alongside the
total. The nearest timed study of a natural-language interface bundled
re-prompting and checking into one completion time and could not separate
them, which left its 31% speed-up open to the standing objection that the time
saved on entry reappears as verification time. A single total cannot answer
that; two components can.

Durations come from paired `phase_enter` / `phase_exit` events, so a
participant who moves back and forth between entry and review is measured
correctly, and a re-analysis under a different idle rule needs no new data.

**Idle handling.** A span longer than `idle_threshold` is reported both raw
and capped. Neither is privileged: the capped figure is stated as a sensitivity
analysis, and which one the paper leads with belongs in the pre-registration,
not in this module.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .store import Decision, Event, EventKind, Phase

#: A span beyond this is treated as a possible walk-away in the capped
#: variant. 180s is long for a single interaction and short for a break;
#: the value is a reporting parameter, not a finding.
DEFAULT_IDLE_THRESHOLD = 180.0


@dataclass
class PhaseSpan:
    phase: str
    started: float
    ended: float | None = None

    @property
    def closed(self) -> bool:
        return self.ended is not None

    @property
    def duration(self) -> float:
        return 0.0 if self.ended is None else max(self.ended - self.started, 0.0)


@dataclass
class TimingReport:
    """Per-phase durations plus the action counts Phase 5.4 asks for."""

    raw: dict[str, float] = field(default_factory=dict)
    capped: dict[str, float] = field(default_factory=dict)
    spans: list[PhaseSpan] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    unclosed_phases: list[str] = field(default_factory=list)
    idle_threshold: float = DEFAULT_IDLE_THRESHOLD
    idle_spans: int = 0

    # -- the two components and the total --------------------------------

    @property
    def entry_seconds(self) -> float:
        return self.raw.get(Phase.ENTRY.value, 0.0)

    @property
    def verification_seconds(self) -> float:
        return self.raw.get(Phase.VERIFICATION.value, 0.0)

    @property
    def total_seconds(self) -> float:
        """Entry plus verification: the elicitation time Phase 6C compares.

        Setup, result-viewing and debrief are excluded, since they are not
        part of entering constraints under either condition.
        """
        return self.entry_seconds + self.verification_seconds

    @property
    def capped_total_seconds(self) -> float:
        return (
            self.capped.get(Phase.ENTRY.value, 0.0)
            + self.capped.get(Phase.VERIFICATION.value, 0.0)
        )

    @property
    def verification_share(self) -> float | None:
        """Fraction of elicitation time spent checking, not entering."""
        total = self.total_seconds
        return self.verification_seconds / total if total > 0 else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "entry_seconds": self.entry_seconds,
            "verification_seconds": self.verification_seconds,
            "total_seconds": self.total_seconds,
            "verification_share": self.verification_share,
            "capped_total_seconds": self.capped_total_seconds,
            "idle_threshold": self.idle_threshold,
            "idle_spans": self.idle_spans,
            "raw_by_phase": dict(self.raw),
            "capped_by_phase": dict(self.capped),
            "counts": dict(self.counts),
            "unclosed_phases": list(self.unclosed_phases),
        }

    def summary(self) -> str:
        share = self.verification_share
        share_text = "n/a" if share is None else f"{share * 100:.0f}%"
        return (
            f"entry {self.entry_seconds:.1f}s + verification "
            f"{self.verification_seconds:.1f}s = {self.total_seconds:.1f}s "
            f"({share_text} spent verifying)"
        )


#: Event kinds counted per session, for the edit/accept/reject figures.
COUNTED_KINDS: tuple[str, ...] = (
    EventKind.TEXT_SUBMIT.value,
    EventKind.PARSE_COMPLETE.value,
    EventKind.PARSE_FAILED.value,
    EventKind.CONSTRAINT_ACCEPT.value,
    EventKind.CONSTRAINT_EDIT.value,
    EventKind.CONSTRAINT_REJECT.value,
    EventKind.STRENGTH_TOGGLE.value,
    EventKind.CLARIFICATION_ANSWER.value,
    EventKind.FORM_ADD.value,
    EventKind.FORM_REMOVE.value,
    EventKind.SOLVE_COMPLETE.value,
)


def phase_spans(events: Sequence[Event]) -> tuple[list[PhaseSpan], list[str]]:
    """Pair phase_enter with phase_exit, tolerating a missing exit.

    An unclosed span is closed at the last event seen and named in the
    report, so an abandoned session is visible as such instead of silently
    contributing zero.
    """
    spans: list[PhaseSpan] = []
    open_spans: dict[str, PhaseSpan] = {}
    unclosed: list[str] = []
    last_at = events[-1].at if events else 0.0

    for event in events:
        if event.kind == EventKind.PHASE_ENTER.value:
            if event.phase in open_spans:
                # A re-enter without an exit: close the previous one here.
                open_spans[event.phase].ended = event.at
            span = PhaseSpan(phase=event.phase, started=event.at)
            open_spans[event.phase] = span
            spans.append(span)
        elif event.kind == EventKind.PHASE_EXIT.value:
            span = open_spans.pop(event.phase, None)
            if span is not None:
                span.ended = event.at

    for phase, span in open_spans.items():
        span.ended = max(last_at, span.started)
        unclosed.append(phase)
    return spans, unclosed


def timing_report(
    events: Iterable[Event], *, idle_threshold: float = DEFAULT_IDLE_THRESHOLD
) -> dict[str, Any]:
    return build_report(events, idle_threshold=idle_threshold).as_dict()


def build_report(
    events: Iterable[Event], *, idle_threshold: float = DEFAULT_IDLE_THRESHOLD
) -> TimingReport:
    ordered = sorted(events, key=lambda e: e.at)
    spans, unclosed = phase_spans(ordered)

    raw: dict[str, float] = defaultdict(float)
    capped: dict[str, float] = defaultdict(float)
    idle_spans = 0
    for span in spans:
        raw[span.phase] += span.duration
        if span.duration > idle_threshold:
            idle_spans += 1
            capped[span.phase] += idle_threshold
        else:
            capped[span.phase] += span.duration

    counts: dict[str, int] = {kind: 0 for kind in COUNTED_KINDS}
    for event in ordered:
        if event.kind in counts:
            counts[event.kind] += 1

    return TimingReport(
        raw=dict(raw),
        capped=dict(capped),
        spans=spans,
        counts=counts,
        unclosed_phases=unclosed,
        idle_threshold=idle_threshold,
        idle_spans=idle_spans,
    )


# -- study-level aggregation ---------------------------------------------


def throughput(
    correct_constraints: int, total_seconds: float
) -> float | None:
    """Correct constraints per minute of staff time (build plan, outcomes).

    Added in v3 after a close analogue found a 31% time reduction alongside an
    18-point accuracy loss, leaving correct output per hour unchanged. Neither
    the primary nor the secondary outcome alone reveals that cancellation.
    """
    if total_seconds <= 0:
        return None
    return correct_constraints / (total_seconds / 60.0)


def decision_counts(constraints: Sequence[dict[str, Any]]) -> dict[str, int]:
    """Accepted / edited / rejected / pending, plus total edits."""
    out = {d.value: 0 for d in Decision}
    edits = 0
    for record in constraints:
        out[record.get("status", Decision.PENDING.value)] = out.get(
            record.get("status", Decision.PENDING.value), 0
        ) + 1
        edits += int(record.get("edit_count") or 0)
    out["edit_events"] = edits
    return out


def compare_conditions(
    reports: dict[str, Sequence[TimingReport]]
) -> dict[str, Any]:
    """Mean entry, verification and total per condition.

    Descriptive only. The paired tests, effect sizes and confidence intervals
    belong in the analysis scripts under `eval/`, against the pre-registered
    plan.
    """
    out: dict[str, Any] = {}
    for condition, items in reports.items():
        if not items:
            out[condition] = None
            continue
        n = len(items)
        out[condition] = {
            "n": n,
            "mean_entry_seconds": sum(r.entry_seconds for r in items) / n,
            "mean_verification_seconds": sum(r.verification_seconds for r in items) / n,
            "mean_total_seconds": sum(r.total_seconds for r in items) / n,
            "mean_verification_share": (
                sum(r.verification_share or 0.0 for r in items) / n
            ),
        }
    return out


__all__ = [
    "COUNTED_KINDS",
    "DEFAULT_IDLE_THRESHOLD",
    "PhaseSpan",
    "TimingReport",
    "build_report",
    "compare_conditions",
    "decision_counts",
    "phase_spans",
    "throughput",
    "timing_report",
]
