"""The recording API the UI drives (build plan Phase 5.4).

Everything the study measures goes through here, and nothing here imports
Streamlit. The interface is a view over this object, so the measurement logic
is tested without a browser and a change of front end cannot quietly alter
what gets logged.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from ..schema.entities import Department
from .store import Condition, Decision, EventKind, Phase, StudyStore
from .timing import TimingReport, build_report, decision_counts, throughput


class StudyRecorder:
    """One participant working one condition.

    Phase transitions are explicit: `enter(Phase.ENTRY)` and `leave()`. The
    recorder refuses to open a second phase without closing the first, so
    entry and verification time cannot both be running and double-count the
    same seconds.
    """

    def __init__(
        self,
        store: StudyStore,
        participant: str,
        condition: str,
        *,
        task: str | None = None,
        department: Department | None = None,
        model: str | None = None,
        prompt_hash: str | None = None,
        order_index: int | None = None,
    ) -> None:
        self.store = store
        self.participant = participant
        self.condition = str(getattr(condition, "value", condition))
        #: Which scenario the participant was asked to express. Residual error
        #: is scored against this task's key (`eval/tasks.py`).
        self.task_id = task
        self.session_id = store.create_session(
            participant,
            self.condition,
            task=task,
            department=department.name if department else None,
            model=model,
            prompt_hash=prompt_hash,
            order_index=order_index,
        )
        self._phase: str | None = None
        self._ordinal = 0

    # -- phases -----------------------------------------------------------

    @property
    def phase(self) -> str | None:
        return self._phase

    def enter(self, phase: str, at: float | None = None) -> None:
        name = str(getattr(phase, "value", phase))
        if self._phase == name:
            return
        if self._phase is not None:
            self.leave(at=at)
        self._phase = name
        self.store.log(self.session_id, name, EventKind.PHASE_ENTER, at=at)

    def leave(self, at: float | None = None) -> None:
        if self._phase is None:
            return
        self.store.log(self.session_id, self._phase, EventKind.PHASE_EXIT, at=at)
        self._phase = None

    def _log(self, kind: str, **kwargs: Any) -> float:
        return self.store.log(
            self.session_id, self._phase or Phase.SETUP.value, kind, **kwargs
        )

    # -- entry ------------------------------------------------------------

    def record_text_submission(self, text: str, at: float | None = None) -> None:
        self._log(
            EventKind.TEXT_SUBMIT,
            payload={"characters": len(text), "words": len(text.split())},
            at=at,
        )

    def record_parse(
        self,
        text: str,
        constraints: Sequence[Any],
        *,
        flags_by_index: Sequence[Sequence[dict[str, Any]]] = (),
        retried: bool = False,
        error: str | None = None,
        at: float | None = None,
    ) -> list[str]:
        """Store what the parser proposed, and return one id per constraint."""
        if error is not None:
            self._log(
                EventKind.PARSE_FAILED,
                payload={"error": error, "retried": retried}, at=at,
            )
            return []

        ids: list[str] = []
        for i, constraint in enumerate(constraints):
            self._ordinal += 1
            flags = list(flags_by_index[i]) if i < len(flags_by_index) else []
            ids.append(self.store.add_constraint(
                self.session_id,
                self._ordinal,
                constraint.source_text,
                constraint.model_dump(mode="json"),
                flags=flags,
                at=at,
            ))
        self._log(
            EventKind.PARSE_COMPLETE,
            payload={
                "n_constraints": len(constraints),
                "retried": retried,
                "types": [c.type for c in constraints],
            },
            at=at,
        )
        return ids

    # -- verification -----------------------------------------------------

    def accept(self, constraint_id: str, final: Any, at: float | None = None) -> None:
        self.store.decide(
            constraint_id, Decision.ACCEPTED,
            final=final.model_dump(mode="json"), at=at,
        )
        self._log(EventKind.CONSTRAINT_ACCEPT, constraint_id=constraint_id, at=at)

    def edit(
        self,
        constraint_id: str,
        final: Any,
        *,
        changed: Iterable[str] = (),
        at: float | None = None,
    ) -> None:
        """An accepted constraint the participant altered first.

        `changed` names the fields, which is what the Phase 6A error
        categories are built from: an edit to `strength` is a hard/soft
        failure, an edit to an entity field is a resolution failure.
        """
        self.store.decide(
            constraint_id, Decision.EDITED,
            final=final.model_dump(mode="json"), increment_edits=True, at=at,
        )
        self._log(
            EventKind.CONSTRAINT_EDIT,
            constraint_id=constraint_id,
            payload={"changed": sorted(changed)},
            at=at,
        )

    def reject(
        self, constraint_id: str, *, reason: str | None = None, at: float | None = None
    ) -> None:
        self.store.decide(constraint_id, Decision.REJECTED, at=at)
        self._log(
            EventKind.CONSTRAINT_REJECT,
            constraint_id=constraint_id,
            payload={"reason": reason} if reason else None,
            at=at,
        )

    def toggle_strength(
        self, constraint_id: str, before: str, after: str, at: float | None = None
    ) -> None:
        self._log(
            EventKind.STRENGTH_TOGGLE,
            constraint_id=constraint_id,
            payload={"from": before, "to": after},
            at=at,
        )

    def answer_clarification(
        self, constraint_id: str, answer: str, at: float | None = None
    ) -> None:
        self._log(
            EventKind.CLARIFICATION_ANSWER,
            constraint_id=constraint_id,
            payload={"characters": len(answer)},
            at=at,
        )

    # -- the structured-form baseline (Condition 1) ----------------------

    def record_form_add(self, constraint: Any, at: float | None = None) -> str:
        """A constraint built from dropdowns, with no parser involved.

        Logged into the same tables as the LLM arm and marked ACCEPTED on
        creation, because the participant composed it field by field and there
        is no proposal to verify. Verification time in this arm is therefore
        genuinely zero, and that is the comparison the plan intends.
        """
        self._ordinal += 1
        cid = self.store.add_constraint(
            self.session_id, self._ordinal, constraint.source_text,
            constraint.model_dump(mode="json"), at=at,
        )
        self.store.decide(
            cid, Decision.ACCEPTED, final=constraint.model_dump(mode="json"), at=at
        )
        self._log(
            EventKind.FORM_ADD, constraint_id=cid,
            payload={"type": constraint.type}, at=at,
        )
        return cid

    def record_form_remove(self, constraint_id: str, at: float | None = None) -> None:
        self.store.decide(constraint_id, Decision.REJECTED, at=at)
        self._log(EventKind.FORM_REMOVE, constraint_id=constraint_id, at=at)

    # -- solving ----------------------------------------------------------

    def record_solve(
        self, result: Any, *, at: float | None = None
    ) -> None:
        self._log(
            EventKind.SOLVE_COMPLETE,
            payload={
                "cost": result.cost,
                "feasible": result.feasible,
                "iterations": result.iterations,
                "seconds": result.elapsed_seconds,
                "hard_breaches": len(result.breakdown.violated_hard),
                "soft_breaches": len(result.breakdown.violated_soft),
            },
            at=at,
        )

    def note(self, text: str, at: float | None = None) -> None:
        self._log(EventKind.NOTE, payload={"text": text}, at=at)

    # -- finishing --------------------------------------------------------

    def complete(self, at: float | None = None) -> None:
        self.leave(at=at)
        self.store.complete_session(self.session_id, at=at)

    # -- reporting --------------------------------------------------------

    def timing(self, **kwargs: Any) -> TimingReport:
        return build_report(self.store.events(self.session_id), **kwargs)

    def final_constraints(self) -> list[dict[str, Any]]:
        return self.store.accepted_constraints(self.session_id)

    def report(self, correct_constraints: int | None = None) -> dict[str, Any]:
        """Per-session figures. Phase 6C aggregates these across participants."""
        timing = self.timing()
        records = self.store.constraints(self.session_id)
        decisions = decision_counts(records)
        kept = len(self.final_constraints())
        out: dict[str, Any] = {
            "session_id": self.session_id,
            "participant": self.participant,
            "condition": self.condition,
            "timing": timing.as_dict(),
            "decisions": decisions,
            "constraints_proposed": len(records),
            "constraints_kept": kept,
        }
        if correct_constraints is not None:
            out["correct_constraints"] = correct_constraints
            out["throughput_per_minute"] = throughput(
                correct_constraints, timing.total_seconds
            )
        return out


__all__ = ["Condition", "Decision", "Phase", "StudyRecorder"]
