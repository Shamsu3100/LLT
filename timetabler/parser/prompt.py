"""System prompt construction (build plan Phase 4.1).

The prompt carries three things: the frozen schema, the department's entities
so names resolve correctly, and the few-shot set below.

`FEW_SHOT` doubles as a development set. It is NOT part of the Phase 6A
labelled corpus: these examples are in the prompt, so scoring against them
would measure recall of the prompt. Phase 6A uses held-out human-authored
sentences.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from ..schema.constraints import ALL_TYPES, SOFT_WEIGHT_MAX, SOFT_WEIGHT_MIN
from ..schema.entities import Department

# -- few-shot set (Phase 4.1 asks for 10-20) -----------------------------

FEW_SHOT: list[tuple[str, list[dict[str, Any]]]] = [
    (
        "I can't teach on Fridays.",
        [{"type": "lecturer_unavailable", "lecturer": "Lee", "days": ["Fri"],
          "periods": None, "strength": "hard", "weight": None,
          "source_text": "I can't teach on Fridays."}],
    ),
    (
        "Prof Rahman would prefer mornings if possible.",
        [{"type": "lecturer_prefers_time", "lecturer": "Rahman", "days": None,
          "periods": [1, 2, 3], "strength": "soft", "weight": 5,
          "source_text": "Prof Rahman would prefer mornings if possible."}],
    ),
    (
        "Data Structures has to be in Lecture Theatre 1.",
        [{"type": "course_requires_room", "course": "Data Structures",
          "room": "Lecture Theatre 1", "strength": "hard", "weight": None,
          "source_text": "Data Structures has to be in Lecture Theatre 1."}],
    ),
    (
        "The architecture practical needs a computer lab.",
        [{"type": "course_requires_room_type", "course": "the architecture practical",
          "room_type": "computer lab", "strength": "hard", "weight": None,
          "source_text": "The architecture practical needs a computer lab."}],
    ),
    (
        "Dr Tan must never teach more than 3 hours back to back.",
        [{"type": "max_consecutive_hours", "lecturer": "Tan", "group": None,
          "max_hours": 3, "strength": "hard", "weight": None,
          "source_text": "Dr Tan must never teach more than 3 hours back to back."}],
    ),
    (
        "Ideally the second years shouldn't sit through more than four straight periods.",
        [{"type": "max_consecutive_hours", "lecturer": None, "group": "second years",
          "max_hours": 4, "strength": "soft", "weight": 4,
          "source_text": "Ideally the second years shouldn't sit through more than "
                         "four straight periods."}],
    ),
    (
        "Nothing before period 2 for the first years, please.",
        [{"type": "no_early_start", "lecturer": None, "group": "the first years",
          "earliest_period": 2, "strength": "soft", "weight": 5,
          "source_text": "Nothing before period 2 for the first years, please."}],
    ),
    (
        "Machine Learning and Computer Architecture cannot clash.",
        [{"type": "courses_not_same_time",
          "courses": ["Machine Learning", "Computer Architecture"],
          "strength": "hard", "weight": None,
          "source_text": "Machine Learning and Computer Architecture cannot clash."}],
    ),
    (
        "Put Introduction to Programming and the programming lab on the same day.",
        [{"type": "courses_same_day",
          "courses": ["Introduction to Programming", "the programming lab"],
          "strength": "soft", "weight": 5,
          "source_text": "Put Introduction to Programming and the programming lab "
                         "on the same day."}],
    ),
    (
        "I'm not around Wednesday afternoons and I'd rather not start before 10.",
        [{"type": "lecturer_unavailable", "lecturer": "Lee", "days": ["Wed"],
          "periods": [4, 5, 6], "strength": "hard", "weight": None,
          "source_text": "I'm not around Wednesday afternoons"},
         {"type": "no_early_start", "lecturer": "Lee", "group": None,
          "earliest_period": 3, "strength": "soft", "weight": 4,
          "source_text": "I'd rather not start before 10"}],
    ),
    (
        "Discrete Maths should be somewhere sensible.",
        [{"type": "needs_clarification",
          "question": "Which constraint did you mean for Discrete Mathematics? For "
                      "example a particular room, a day, or a time of day.",
          "candidate_types": ["course_requires_room", "lecturer_prefers_time"],
          "note": "No schedulable property stated.", "strength": "soft",
          "weight": None,
          "source_text": "Discrete Maths should be somewhere sensible."}],
    ),
    (
        "Keep my teaching to two days a week.",
        [{"type": "unsupported",
          "note": "Caps the number of distinct teaching days per lecturer. No "
                  "frozen type expresses a count of days.",
          "strength": "soft", "weight": None,
          "source_text": "Keep my teaching to two days a week."}],
    ),
    (
        "The FYP seminar goes in the small lab.",
        [{"type": "course_requires_room", "course": "FYP seminar",
          "room": "the small lab", "strength": "hard", "weight": None,
          "source_text": "The FYP seminar goes in the small lab."}],
    ),
    (
        "Tan is unavailable Monday period 1.",
        [{"type": "lecturer_unavailable", "lecturer": "Tan", "days": ["Mon"],
          "periods": [1], "strength": "hard", "weight": None,
          "source_text": "Tan is unavailable Monday period 1."}],
    ),
    (
        "It would be good to avoid clashing the two year-one lectures, but it's not critical.",
        [{"type": "courses_not_same_time",
          "courses": ["Introduction to Programming", "Discrete Mathematics"],
          "strength": "soft", "weight": 2,
          "source_text": "It would be good to avoid clashing the two year-one "
                         "lectures, but it's not critical."}],
    ),
    (
        "Nair can't do Thursdays and Machine Learning must be in the big lecture theatre.",
        [{"type": "lecturer_unavailable", "lecturer": "Nair", "days": ["Thu"],
          "periods": None, "strength": "hard", "weight": None,
          "source_text": "Nair can't do Thursdays"},
         {"type": "course_requires_room", "course": "Machine Learning",
          "room": "the big lecture theatre", "strength": "hard", "weight": None,
          "source_text": "Machine Learning must be in the big lecture theatre"}],
    ),
]

SCHEMA_RULES = f"""\
Output contract
- Reply with JSON only. No prose, no code fences.
- Shape: {{"constraints": [ ... ]}}
- One object per constraint. One sentence may yield several constraints; split
  them and give each its own `source_text` quoting only its own span.
- Every object carries: `type`, `strength`, `source_text`, `weight`.
- `strength` is "hard" or "soft".
- `weight` is null for hard constraints, and {SOFT_WEIGHT_MIN}-{SOFT_WEIGHT_MAX}
  for soft ones ({SOFT_WEIGHT_MAX} = strongest preference). Use 5 when the
  speaker gives no sense of strength.
- Emit no field that is not listed for the type. Use null for an absent
  optional field.

Strength rules
- "can't", "cannot", "must", "never", "unable", "not available", "has to",
  "required", "only" -> "hard".
- "prefer", "rather", "if possible", "ideally", "would like", "better",
  "hopefully", "try to" -> "soft".
- Both kinds of wording present, or neither, and the reading changes the
  timetable -> emit `needs_clarification` with the question to ask.

Entity names
- Copy the words the speaker used into the entity fields. Do not invent an
  identifier and do not correct a name. Descriptions are fine: "the big lab",
  "the first years", "the programming lab" are all acceptable values.
- Resolution against the department register happens after you reply, so a
  description that you cannot match is still worth passing through verbatim.
- Never name an entity the speaker did not mention.

Times
- Days are "Mon" "Tue" "Wed" "Thu" "Fri" "Sat" "Sun".
- Periods are integers from 1. Read "mornings" as the first half of the
  configured periods and "afternoons" as the second half.
- `days: null` means every day; `periods: null` means every period.

When no type fits
- A schedulable request that no listed type expresses -> `unsupported`, with a
  `note` naming what it asks for.
- A request too vague to schedule -> `needs_clarification`, with a `question`.
- Guessing is worse than either. Both categories are logged and reviewed.
"""

TYPE_SPECS = """\
Constraint types (this list is closed; emit no other `type` value)

1. lecturer_unavailable — lecturer, days (list|null), periods (list[int]|null)
   A time the lecturer is not able to teach. At least one of days/periods must
   be given.
2. lecturer_prefers_time — lecturer, days (list|null), periods (list[int]|null)
   The times listed are the PREFERRED ones. At least one must be given.
3. course_requires_room — course, room
   One named course pinned to one named room.
4. course_requires_room_type — course, room_type
   A kind of room, when no specific room is named.
5. max_consecutive_hours — max_hours (int), lecturer (|null), group (|null)
   Exactly one of lecturer or group.
6. no_early_start — earliest_period (int), lecturer (|null), group (|null)
   Earliest acceptable period. At most one of lecturer or group; both null
   means department-wide.
7. courses_not_same_time — courses (list, 2 or more)
   These courses must not overlap.
8. courses_same_day — courses (list, 2 or more)
   These courses should fall on one day.
9. unsupported — note (string)
   No type above fits.
10. needs_clarification — question (string), candidate_types (list[str]),
    note (string|null)
    Ambiguous input. `question` is put to the user verbatim.
"""


def render_entities(department: Department, max_aliases: int = 3) -> str:
    """Render the department register for the prompt (Phase 4.1)."""
    lines: list[str] = []

    def block(title: str, rows: list[str]) -> None:
        lines.append(title)
        lines.extend(f"  {r}" for r in rows)
        lines.append("")

    def aliases(item: Any) -> str:
        al = (getattr(item, "aliases", None) or [])[:max_aliases]
        return f"  [also: {', '.join(al)}]" if al else ""

    block("Lecturers", [
        f"{x.id}  {x.name}{aliases(x)}" for x in department.lecturers
    ])
    block("Student groups", [
        f"{x.id}  {x.name} (size {x.size}){aliases(x)}" for x in department.groups
    ])
    block("Rooms", [
        f"{x.id}  {x.name or x.id} — {x.type}, holds {x.capacity}{aliases(x)}"
        for x in department.rooms
    ])
    block("Courses", [
        f"{x.id}  {x.name or x.id} — {x.type}, {x.duration} periods, "
        f"{x.size} students, lecturer {x.lecturer}, group {x.group}{aliases(x)}"
        for x in department.courses
    ])
    block("Room types in use", [", ".join(department.room_types())])
    block("Teaching grid", [
        f"days: {', '.join(department.days)}",
        f"periods: {min(department.periods)}-{max(department.periods)}",
    ])
    return "\n".join(lines).rstrip()


def render_few_shot(examples: list[tuple[str, list[dict[str, Any]]]] | None = None) -> str:
    rows = FEW_SHOT if examples is None else examples
    out: list[str] = ["Examples"]
    for text, constraints in rows:
        payload = json.dumps({"constraints": constraints}, ensure_ascii=False)
        out.append(f"\nInput: {text}\nOutput: {payload}")
    return "\n".join(out)


def build_system_prompt(
    department: Department,
    examples: list[tuple[str, list[dict[str, Any]]]] | None = None,
) -> str:
    """Assemble the full system prompt."""
    return "\n\n".join([
        "You convert timetabling constraints written in plain English by "
        "teaching staff into JSON objects drawn from a fixed schema. You do "
        "not build or alter timetables, and you do not decide which "
        "constraints matter.",
        TYPE_SPECS,
        SCHEMA_RULES,
        "Department register\n\n" + render_entities(department),
        render_few_shot(examples),
    ])


def prompt_hash(system_prompt: str) -> str:
    """Short digest recorded with every run for reproducibility (§2)."""
    return hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()[:16]


def _assert_examples_valid() -> None:
    """Guard: every few-shot output must satisfy the frozen schema."""
    from ..schema.constraints import ConstraintSet

    for text, constraints in FEW_SHOT:
        for c in constraints:
            if c["type"] not in ALL_TYPES:
                raise ValueError(f"few-shot uses unknown type {c['type']!r}")
        ConstraintSet.model_validate({"constraints": constraints})


__all__ = [
    "FEW_SHOT",
    "SCHEMA_RULES",
    "TYPE_SPECS",
    "build_system_prompt",
    "prompt_hash",
    "render_entities",
    "render_few_shot",
]
