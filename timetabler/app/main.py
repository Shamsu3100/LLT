"""Streamlit verification UI (build plan Phase 5).

    streamlit run timetabler/app/main.py

Four pages: Input, Review, Result, and a Log view for the researcher. The
condition picked in the sidebar decides whether entry runs through the parser
(Condition 2), a structured form (Condition 1, the baseline), or the parser
with the review step skipped (Condition 3, stretch).

This module is a view. Every measurement goes through `StudyRecorder`, so what
the study records does not depend on Streamlit's rerun behaviour.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import streamlit as st

# `streamlit run` executes this file as a script, so it has no parent package
# and relative imports fail. Put the project root on the path and import
# absolutely. Everything else in the package keeps its relative imports.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from timetabler.app.credentials import CredentialStore, redact
from timetabler.app.recorder import Condition, StudyRecorder
from timetabler.app.store import Decision, Phase, StudyStore
from timetabler.app.verbalise import Namer, verbalise
from timetabler.data import load_department
from timetabler.parser.client import (
    DEFAULT_MODEL,
    ENDPOINTS,
    PROVIDERS,
    ClientError,
    StubClient,
    make_client,
)
from timetabler.eval.tasks import load_tasks
from timetabler.parser.parse import ConstraintParser
from timetabler.parser.resolve import EntityResolver, materialise_all
from timetabler.schema.constraints import OPEN_TYPES, SOFT_WEIGHT_MAX, SOFT_WEIGHT_MIN
from timetabler.schema.entities import DAYS
from timetabler.solver import ProblemIndex, solve

PAGES = ("Input", "Review", "Result", "Log", "Setup")

CONDITION_LABELS = {
    Condition.FORM.value: "1 — Structured form (baseline)",
    Condition.LLM.value: "2 — LLM-assisted entry",
    Condition.LLM_NO_VERIFY.value: "3 — LLM, no verification (stretch)",
}


# -- shared state ---------------------------------------------------------


def boot() -> None:
    st.set_page_config(page_title="Timetable constraints", layout="wide")
    if "department" not in st.session_state:
        st.session_state.department = load_department()
        st.session_state.index = ProblemIndex(st.session_state.department)
        st.session_state.resolver = EntityResolver(st.session_state.department)
        st.session_state.namer = Namer(st.session_state.department)
        st.session_state.store = StudyStore()
        st.session_state.recorder = None
        st.session_state.parsed = []
        st.session_state.decisions = {}
        st.session_state.result = None
        st.session_state.page = "Input"
        # Held for this process only; never written to the store or disk.
        st.session_state.api_keys = {}
        try:
            st.session_state.tasks = load_tasks()
        except Exception:
            st.session_state.tasks = None
    st.session_state.credentials = CredentialStore(st.session_state.api_keys)


def department() -> Any:
    return st.session_state.department


def recorder() -> StudyRecorder | None:
    return st.session_state.recorder


def build_parser(spec: str | None = None) -> ConstraintParser:
    """A live client when a credential is configured, a stub otherwise.

    `spec` is a `provider:model` string, so a study can be run against any
    configured provider (Phase 6A's model-choice axis). The stub keeps the
    interface walkable without a credential and is labelled in the sidebar: a
    session recorded against it is a UI rehearsal, not study data.
    """
    wanted = spec or st.session_state.get("client_spec") or "anthropic"
    if st.session_state.get("parser") is not None and (
        st.session_state.get("parser_spec") == wanted
    ):
        return st.session_state.parser

    provider = wanted.split(":", 1)[0].strip().casefold()
    options = st.session_state.credentials.options_for(provider)
    try:
        client: Any = make_client(wanted, **options)
        st.session_state.live = True
        st.session_state.client_error = None
    except ClientError as exc:
        client = StubClient(on_miss="empty", model=f"stub ({wanted} unavailable)")
        st.session_state.live = False
        st.session_state.client_error = str(exc)

    st.session_state.parser = ConstraintParser(department(), client)
    st.session_state.parser_spec = wanted
    return st.session_state.parser


# -- sidebar --------------------------------------------------------------


def sidebar() -> None:
    st.sidebar.header("Session")
    active = recorder()

    if active is None:
        participant = st.sidebar.text_input("Participant code", value="P01")
        condition = st.sidebar.selectbox(
            "Condition",
            list(CONDITION_LABELS),
            format_func=lambda c: CONDITION_LABELS[c],
        )
        order = st.sidebar.number_input(
            "Order position", min_value=1, max_value=4, value=1,
            help="Which condition this participant saw first. Counterbalanced "
                 "across participants (Phase 6C).",
        )
        tasks = st.session_state.get("tasks")
        task_id = None
        if tasks is not None:
            task_id = st.sidebar.selectbox(
                "Task", tasks.ids,
                format_func=lambda t: f"{t} - {tasks.by_id(t).title}",
                help="Which scenario this participant is asked to express. "
                     "Recorded on the session so residual error can be scored "
                     "against the task key.",
            )
        st.session_state.task_id = task_id
        st.session_state.client_spec = st.sidebar.text_input(
            "Model", value=st.session_state.get("client_spec", "anthropic"),
            help="provider:model — anthropic:claude-sonnet-5-5, openai:<model>, "
                 "deepseek:<model>, ollama:<model>. Providers: "
                 + ", ".join(PROVIDERS),
        )
        if st.sidebar.button("Start session", type="primary"):
            parser = build_parser(st.session_state.client_spec)
            st.session_state.recorder = StudyRecorder(
                st.session_state.store, participant, condition,
                task=st.session_state.get("task_id"),
                department=department(),
                model=getattr(parser.client, "model", None),
                prompt_hash=parser.prompt_hash,
                order_index=int(order),
            )
            st.session_state.parsed = []
            st.session_state.decisions = {}
            st.session_state.result = None
            st.rerun()
        st.sidebar.caption("Start a session before entering constraints.")
    else:
        st.sidebar.success(
            f"{active.participant} — {CONDITION_LABELS.get(active.condition, active.condition)}"
        )
        timing = active.timing()
        st.sidebar.metric("Entry", f"{timing.entry_seconds:.0f}s")
        st.sidebar.metric("Verification", f"{timing.verification_seconds:.0f}s")
        st.sidebar.metric("Elicitation total", f"{timing.total_seconds:.0f}s")
        if st.sidebar.button("Finish session"):
            active.complete()
            st.session_state.recorder = None
            st.rerun()

    st.sidebar.divider()
    st.session_state.page = st.sidebar.radio("Page", PAGES, key="nav")
    if st.session_state.get("live") is False:
        st.sidebar.warning(
            "The parser is stubbed and returns nothing, so sessions recorded "
            "now are rehearsals, not study data.\n\n"
            + (st.session_state.get("client_error") or "")
        )


# -- page: input ----------------------------------------------------------


def show_brief(active: StudyRecorder) -> None:
    """The scenario the participant has to express.

    Shown identically in every condition, and written as facts rather
    than as the sentences we want back: a brief that states the target
    sentence turns the study into a transcription race.
    """
    tasks = st.session_state.get("tasks")
    task = tasks.by_id(active.task_id) if (tasks and active.task_id) else None
    if task is None:
        return
    with st.expander(f"Task {task.id} - {task.title}", expanded=True):
        st.caption(f"presented as: {task.presentation}")
        for line in task.brief:
            st.write(f"- {line}")


def page_input() -> None:
    st.title("Your timetabling constraints")
    active = recorder()
    if active is None:
        st.info("Start a session in the sidebar.")
        return
    show_brief(active)

    if active.condition == Condition.FORM.value:
        form_entry(active)
        return

    active.enter(Phase.ENTRY)
    st.write(
        "Write your constraints in your own words, as you would to a "
        "colleague. One per line is fine, and several in one sentence is fine."
    )
    text = st.text_area("Constraints", height=200, key="entry_text")

    if st.button("Submit", type="primary", disabled=not text.strip()):
        active.record_text_submission(text)
        parser = build_parser()
        with st.spinner("Reading your constraints..."):
            result = parser.parse(text)

        resolved = [rc for rc in result.constraints]
        ids = active.record_parse(
            text,
            [rc.constraint for rc in resolved],
            flags_by_index=[[f.as_dict() for f in rc.flags] for rc in resolved],
            retried=result.retried,
            error=result.error,
        )
        st.session_state.parsed = list(zip(ids, resolved))
        st.session_state.decisions = {}

        if result.error:
            st.error(f"That could not be read as constraints: {result.error}")
            return
        if active.condition == Condition.LLM_NO_VERIFY.value:
            # Condition 3 skips review: accept everything as proposed.
            for cid, rc in st.session_state.parsed:
                active.accept(cid, rc.constraint)
            st.success(f"{len(ids)} constraint(s) recorded. See Result.")
        else:
            st.success(f"{len(ids)} constraint(s) read. Check them on Review.")


def form_entry(active: StudyRecorder) -> None:
    """Condition 1: the structured-form baseline.

    Identical for every participant, as Phase 6C requires. Dropdowns only, so
    nothing here can be misread and there is nothing to verify.
    """
    active.enter(Phase.ENTRY)
    st.caption("Baseline condition: build each constraint from the fields below.")
    dept = department()
    namer = st.session_state.namer

    kind = st.selectbox("Constraint type", [
        "Lecturer unavailable",
        "Lecturer prefers a time",
        "Course needs a specific room",
        "Course needs a type of room",
        "Maximum consecutive periods",
        "No early start",
        "Two courses must not clash",
        "Two courses on the same day",
    ])
    strength = st.radio("Strength", ["hard", "soft"], horizontal=True)
    weight = None
    if strength == "soft":
        weight = st.slider(
            "Weight", SOFT_WEIGHT_MIN, SOFT_WEIGHT_MAX, 5,
            help=f"{SOFT_WEIGHT_MAX} is the strongest preference.",
        )

    lecturers = {x.name: x.id for x in dept.lecturers}
    groups = {x.name: x.id for x in dept.groups}
    rooms = {(x.name or x.id): x.id for x in dept.rooms}
    courses = {(x.name or x.id): x.id for x in dept.courses}
    periods = dept.periods

    built = None
    source = ""
    from timetabler.schema import (
        CourseRequiresRoom, CourseRequiresRoomType, CoursesNotSameTime,
        CoursesSameDay, LecturerPrefersTime, LecturerUnavailable,
        MaxConsecutiveHours, NoEarlyStart,
    )
    common = {"strength": strength, "weight": weight}

    if kind in ("Lecturer unavailable", "Lecturer prefers a time"):
        who = st.selectbox("Lecturer", list(lecturers))
        days = st.multiselect("Days", [d for d in DAYS if d in dept.days])
        picked = st.multiselect("Periods", periods)
        model = LecturerUnavailable if kind == "Lecturer unavailable" else LecturerPrefersTime
        source = f"[form] {kind}: {who}, days={days or 'any'}, periods={picked or 'any'}"
        if days or picked:
            built = model(
                lecturer=lecturers[who], days=days or None,
                periods=picked or None, source_text=source, **common,
            )
    elif kind == "Course needs a specific room":
        course = st.selectbox("Course", list(courses))
        room = st.selectbox("Room", list(rooms))
        source = f"[form] {courses[course]} requires room {rooms[room]}"
        built = CourseRequiresRoom(
            course=courses[course], room=rooms[room], source_text=source, **common
        )
    elif kind == "Course needs a type of room":
        course = st.selectbox("Course", list(courses))
        room_type = st.selectbox("Room type", dept.room_types())
        source = f"[form] {courses[course]} requires a {room_type}"
        built = CourseRequiresRoomType(
            course=courses[course], room_type=room_type, source_text=source, **common
        )
    elif kind == "Maximum consecutive periods":
        scope = st.radio("Applies to", ["Lecturer", "Student group"], horizontal=True)
        who = st.selectbox("Who", list(lecturers if scope == "Lecturer" else groups))
        hours = st.number_input("Maximum", 1, max(periods), 3)
        source = f"[form] max {hours} consecutive periods for {who}"
        built = MaxConsecutiveHours(
            max_hours=int(hours),
            lecturer=lecturers[who] if scope == "Lecturer" else None,
            group=groups[who] if scope == "Student group" else None,
            source_text=source, **common,
        )
    elif kind == "No early start":
        scope = st.radio(
            "Applies to", ["Everyone", "Lecturer", "Student group"], horizontal=True
        )
        who = None
        if scope != "Everyone":
            who = st.selectbox("Who", list(lecturers if scope == "Lecturer" else groups))
        earliest = st.selectbox("Earliest period", periods, index=min(1, len(periods) - 1))
        source = f"[form] no start before period {earliest} for {who or 'everyone'}"
        built = NoEarlyStart(
            earliest_period=int(earliest),
            lecturer=lecturers[who] if scope == "Lecturer" else None,
            group=groups[who] if scope == "Student group" else None,
            source_text=source, **common,
        )
    else:
        picked = st.multiselect("Courses", list(courses), max_selections=4)
        if len(picked) >= 2:
            ids = [courses[p] for p in picked]
            model = CoursesNotSameTime if "not clash" in kind else CoursesSameDay
            source = f"[form] {kind}: {', '.join(ids)}"
            built = model(courses=ids, source_text=source, **common)

    if built is not None:
        st.info(verbalise(built, namer))
    if st.button("Add constraint", type="primary", disabled=built is None):
        active.record_form_add(built)
        st.success("Added.")
        st.rerun()

    existing = st.session_state.store.constraints(active.session_id)
    kept = [c for c in existing if c["status"] != Decision.REJECTED.value]
    if kept:
        st.subheader(f"Your constraints ({len(kept)})")
        for record in kept:
            cols = st.columns([8, 1])
            cols[0].write(record["source_text"])
            if cols[1].button("Remove", key=f"rm_{record['id']}"):
                active.record_form_remove(record["id"])
                st.rerun()


# -- page: review ---------------------------------------------------------


def page_review() -> None:
    st.title("Check what we understood")
    active = recorder()
    if active is None:
        st.info("Start a session in the sidebar.")
        return
    if active.condition == Condition.FORM.value:
        st.info("The baseline condition has nothing to verify: you built each "
                "constraint from the fields yourself.")
        return
    if active.condition == Condition.LLM_NO_VERIFY.value:
        st.warning("This condition skips verification by design. See Result.")
        return
    if not st.session_state.parsed:
        st.info("Nothing to review yet. Enter constraints on the Input page.")
        return

    active.enter(Phase.VERIFICATION)
    namer = st.session_state.namer
    st.caption(
        "Each line is how we read one of your sentences. Accept it, change it, "
        "or reject it."
    )

    for cid, rc in st.session_state.parsed:
        constraint = rc.constraint
        decided = st.session_state.decisions.get(cid)
        with st.container(border=True):
            st.markdown(f"**{verbalise(constraint, namer)}**")
            st.caption(f"from your words: “{constraint.source_text}”")

            for flag in rc.flags:
                (st.error if flag.severity.value == "block" else st.warning)(
                    flag.message
                )

            if decided:
                st.success(f"Marked {decided}.")
                continue

            if constraint.type in OPEN_TYPES:
                answer = st.text_input(
                    "Your answer", key=f"clar_{cid}",
                    placeholder="Rephrase the constraint, or leave it out.",
                )
                if st.button("Send answer", key=f"send_{cid}", disabled=not answer):
                    active.answer_clarification(cid, answer)
                    active.reject(cid, reason="clarified")
                    st.session_state.decisions[cid] = "clarified"
                    st.rerun()
                continue

            edited = constraint
            new_strength = st.radio(
                "Strength", ["hard", "soft"],
                index=0 if constraint.strength == "hard" else 1,
                horizontal=True, key=f"str_{cid}",
            )
            if new_strength != constraint.strength:
                payload = constraint.model_dump()
                payload["strength"] = new_strength
                payload["weight"] = None if new_strength == "hard" else 5
                edited = type(constraint).model_validate(payload)
                st.info(f"Would become: {verbalise(edited, namer)}")

            cols = st.columns(3)
            if cols[0].button("Accept", key=f"acc_{cid}", type="primary"):
                if new_strength != constraint.strength:
                    active.toggle_strength(cid, constraint.strength, new_strength)
                    active.edit(cid, edited, changed=["strength"])
                    st.session_state.decisions[cid] = "edited"
                else:
                    active.accept(cid, constraint)
                    st.session_state.decisions[cid] = "accepted"
                st.rerun()
            if cols[1].button("Reject", key=f"rej_{cid}"):
                active.reject(cid)
                st.session_state.decisions[cid] = "rejected"
                st.rerun()
            cols[2].caption("Reject anything we read wrongly.")

    remaining = len(st.session_state.parsed) - len(st.session_state.decisions)
    if remaining == 0:
        st.success("All checked. Go to Result to build the timetable.")
    else:
        st.caption(f"{remaining} still to check.")


# -- page: result ---------------------------------------------------------


def page_result() -> None:
    st.title("Timetable")
    active = recorder()
    if active is None:
        st.info("Start a session in the sidebar.")
        return

    active.enter(Phase.RESULT)
    index = st.session_state.index
    resolver = st.session_state.resolver
    namer = st.session_state.namer

    final = active.final_constraints()
    if not final:
        st.info("No constraints accepted yet.")
        return

    st.caption(f"{len(final)} constraint(s) accepted.")
    iterations = st.select_slider(
        "Search effort", [5_000, 20_000, 40_000, 80_000], value=20_000
    )

    if st.button("Build timetable", type="primary"):
        from timetabler.schema.constraints import ConstraintSet

        rebuilt = ConstraintSet.model_validate({"constraints": final}).constraints
        solvable = materialise_all(resolver.resolve(c) for c in rebuilt)
        with st.spinner("Searching..."):
            result = solve(index, solvable, iterations=int(iterations), seed=0)
        active.record_solve(result)
        st.session_state.result = result

    result = st.session_state.result
    if result is None:
        return

    if result.feasible:
        st.success(result.breakdown.summary())
    else:
        st.error(result.breakdown.summary())

    st.subheader("Grid")
    st.code(result.timetable.render(), language=None)

    if result.breakdown.violated_hard:
        st.subheader("Constraints that cannot all hold")
        st.caption("These conflict with each other or with room capacity. "
                   "One of them has to be relaxed.")
        for row in result.breakdown.violated_hard:
            st.error(f"“{row['source_text']}” — breached {row['count']}x")

    if result.breakdown.violated_soft:
        st.subheader("Preferences not met")
        for row in sorted(result.breakdown.violated_soft,
                          key=lambda r: -r["contribution"]):
            st.warning(
                f"“{row['source_text']}” — breached {row['count']}x "
                f"(weight {row['weight']})"
            )

    st.subheader("Per person")
    dept = department()
    who = st.selectbox(
        "Lecturer", [x.id for x in dept.lecturers], format_func=namer
    )
    mine = [
        (cid, result.timetable.get(cid))
        for cid in index.courses_by_lecturer.get(who, ())
    ]
    if not mine:
        st.caption("Nothing scheduled.")
    for course_id, assignment in mine:
        if assignment is None:
            continue
        slot = index.slots[assignment.slot]
        duration = index.courses[course_id].duration
        last = slot.period + duration - 1
        span = f"period {slot.period}" if duration == 1 else f"periods {slot.period}-{last}"
        st.write(f"- {namer(course_id)} — {slot.day}, {span}, in {namer(assignment.room)}")


# -- page: log ------------------------------------------------------------


def page_log() -> None:
    st.title("Study log")
    st.caption("For the researcher. Participants do not need this page.")
    store = st.session_state.store
    sessions = store.sessions()
    if not sessions:
        st.info("No sessions recorded yet.")
        return

    st.dataframe([
        {
            "participant": s["participant"],
            "condition": s["condition"],
            "order": s["order_index"],
            "model": s["model"],
            "complete": s["completed_at"] is not None,
        }
        for s in sessions
    ], use_container_width=True)

    chosen = st.selectbox(
        "Session", [s["id"] for s in sessions],
        format_func=lambda sid: next(
            f"{s['participant']} / {s['condition']}" for s in sessions if s["id"] == sid
        ),
    )
    export = store.export(chosen)
    timing = export["timing"]

    cols = st.columns(4)
    cols[0].metric("Entry", f"{timing['entry_seconds']:.0f}s")
    cols[1].metric("Verification", f"{timing['verification_seconds']:.0f}s")
    cols[2].metric("Total", f"{timing['total_seconds']:.0f}s")
    share = timing["verification_share"]
    cols[3].metric("Verifying", "n/a" if share is None else f"{share * 100:.0f}%")

    if timing["idle_spans"]:
        st.caption(
            f"{timing['idle_spans']} span(s) over {timing['idle_threshold']:.0f}s. "
            f"Capped total: {timing['capped_total_seconds']:.0f}s."
        )
    if timing["unclosed_phases"]:
        st.warning(f"Unclosed phases: {', '.join(timing['unclosed_phases'])}")

    st.subheader("Decisions")
    st.json(timing["counts"])
    st.subheader("Constraints")
    st.dataframe([
        {
            "ordinal": c["ordinal"],
            "type": c["proposed"].get("type"),
            "status": c["status"],
            "edits": c["edit_count"],
            "strength before": c["strength_before"],
            "strength after": c["strength_after"],
            "source": c["source_text"][:60],
        }
        for c in export["constraints"]
    ], use_container_width=True)

    st.download_button(
        "Download session JSON",
        json.dumps(export, indent=2, ensure_ascii=False),
        file_name=f"session_{chosen[:8]}.json",
        mime="application/json",
    )


# -- entry point ----------------------------------------------------------


def page_setup() -> None:
    """Researcher-only credential entry (build plan §5, Privacy).

    Keys live in session state for the life of the process. They are passed
    to the client as an argument, never written to the study log, never
    exported, never placed in the environment, and never read back to the
    screen. Participants do not see this page.
    """
    st.title("Setup")
    store = st.session_state.credentials

    st.info(
        "Credentials entered here are held in memory for this session only. "
        "They are never written to the study log, the session export, or disk, "
        "and they are never shown back on screen. Close the app and they are "
        "gone. For an unattended study, export the variable in the shell "
        "instead.",
        icon=None,
    )
    st.warning(
        "Before sending any staff text to a provider, hold its data-retention "
        "policy in writing and state it in the ethics application and the "
        "paper (build plan §5, Privacy)."
    )

    provider = st.selectbox(
        "Provider", [p for p in PROVIDERS if p != "stub"],
        help="Each provider reads its own credential.",
    )
    endpoint = ENDPOINTS.get(provider)
    if provider == "anthropic":
        st.caption("Reads ANTHROPIC_API_KEY from the environment when unset here.")
    elif endpoint is not None:
        st.caption(
            f"Reads {endpoint.env_var} from the environment when unset here. "
            f"Model ids: {endpoint.docs}"
        )

    st.write(f"**Status:** {store.describe(provider)}")

    entered = st.text_input(
        f"{provider} credential", type="password", key=f"key_{provider}",
        placeholder="paste here; it is not echoed or stored on disk",
    )

    cols = st.columns(3)
    if cols[0].button("Save for this session", type="primary", disabled=not entered):
        store.set(provider, entered)
        st.session_state.parser = None  # rebuild the client with the new key
        st.success(f"Credential held for {provider} in this session only.")
        st.rerun()

    if cols[1].button("Forget", disabled=not store.has(provider)):
        store.clear(provider)
        st.session_state.parser = None
        st.rerun()

    if cols[2].button("Test connection"):
        model = st.session_state.get("client_spec") or provider
        if not model.startswith(provider):
            model = provider
        try:
            client = make_client(model, **store.options_for(provider))
            meta = client.ping()
            st.success(
                f"Reached {meta.provider}/{meta.model}: "
                f"{meta.input_tokens}+{meta.output_tokens} tokens"
                + (f", ${meta.cost_usd:.5f}" if meta.cost_usd else "")
            )
        except ClientError as exc:
            # redact in case a provider echoes the credential in its error
            st.error(redact(str(exc)))

    if store.providers():
        st.divider()
        st.caption("Credentials held this session: " + ", ".join(store.providers()))


def main() -> None:
    boot()
    sidebar()
    page = st.session_state.get("page", "Input")
    if page == "Input":
        page_input()
    elif page == "Review":
        page_review()
    elif page == "Result":
        page_result()
    elif page == "Setup":
        page_setup()
    else:
        page_log()


main()
