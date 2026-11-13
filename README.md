# LLT — LLM-Assisted Constraint Elicitation for Timetabling

Staff write timetabling constraints in plain English, an LLM converts them to
formal constraints, staff verify each one, and a solver builds the timetable.
The system exists to measure three things: how much staff time the assisted
path saves, how many constraint errors survive verification, and whether staff
trust the result.

Research code. `timetabler/README.md` documents the design decisions and the
measurement conventions in detail.

## Run it locally

Python 3.11 or newer.

```bash
pip install -r requirements.txt
```

The verification UI:

```bash
python -m streamlit run timetabler/app/main.py
```

Use `python -m streamlit`, not the bare `streamlit` command: on a machine with
more than one interpreter the console script may belong to an installation
that has no `pydantic`.

The offline accuracy harness, which needs no credential:

```bash
python -m timetabler.eval.run_accuracy --client gold
```

The solver:

```bash
python -m timetabler.solver.run --iterations 40000 --restarts 3
```

## Collecting study data

Point the log somewhere fixed, so it does not depend on the launch directory:

```bash
export TIMETABLER_DB=~/llt-study/study.sqlite
```

Run one session per participant per condition. Set the participant code,
condition and order position in the sidebar, click **Start session**, work
through Input, Review and Result, then **Finish session**. Every action is
timestamped into that one file as it happens, so a crash loses at most the
current phase. Counterbalancing is yours to fix in advance: the app records
the order position you enter, it does not assign one.

Back up after each participant - it is a single file:

```bash
cp ~/llt-study/study.sqlite ~/llt-study/backups/after-P01.sqlite
```

Export to tidy CSV for analysis:

```bash
python -m timetabler.eval.export_study --out exports/
```

That writes `sessions.csv` (one row per participant-condition, with entry
time, verification time and total as separate columns), `constraints.csv`
(one row per constraint, with what was proposed, what was decided, and what
changed), and `events.csv` (the full log). It prints a per-condition summary
and names any session that was abandoned or left with an open phase.

Throughput needs correctness, which this repository cannot score: participants
write their own sentences, so the Phase 6A labelled set is a different corpus.
Supply `--correct session_id,correct_constraints` once residual error is
scored, and the column is filled.

**Abandoned sessions are excluded from the printed means but kept in the CSV.**
An unclosed phase is closed at the last event seen, so a session left open
overnight would otherwise swamp its condition's mean.

### Task keys

Residual error is the gap between the set a participant finally accepted and
the set the **task** asked them to express, so each task needs a key of
intended constraints. `timetabler/eval/labelled/sample_tasks.json` holds four
starter tasks for the sample department.

```bash
python -m timetabler.eval.export_study --tasks timetabler/eval/labelled/sample_tasks.json --out exports/
```

That fills `task_correct`, `task_missed`, `task_spurious`,
`task_wrong_type`, `task_wrong_strength`, `task_wrong_entity`,
`task_unresolved` and `residual_error_rate`, and derives throughput from them.
An explicit `--correct` file still wins, so a scoring decision can be
overridden without editing the log.

**A brief must be written as facts, not as the sentences you want back.** If a
task says "You cannot teach on Fridays", the free-text arm copies that sentence
and the form arm transcribes it, and the study measures typing speed instead of
elicitation. The starter briefs are shaded grids, inventory extracts and diary
notes; `presentation` records which, because a brief read aloud and a brief
shown as a calendar are different instruments. A test asserts no brief line is
ever a constraint's `source_text`.

Run `validate_against_department` before the pilot. A key naming an entity the
department does not contain scores every participant wrong for the
researcher's reason, not theirs — the export refuses to run in that case.

### Synthetic studies

```bash
python -m timetabler.eval.synth --participants 30 --out logs/synthetic.sqlite
```

Generates a full within-subject study at the real shape, so the analysis can
be dry-run before anyone is recruited. **It says nothing about the result** —
the effect is whatever `--time-reduction` put there, so recovering it is
circular. Three uses are legitimate:

1. **Pipeline validation** — the columns are right, the exclusion logic fires,
   the scoring aligns, the export is clean.
2. **A null check** — `--time-reduction 0`, then confirm the analysis reports
   no difference. A pipeline that finds an effect in null data is broken.
3. **A power sanity check** — generate at the target effect and confirm the
   planned sample can detect it.

It cannot reproduce real between-participant variance (which is what actually
determines power), which constraints humans misread and why, learning across
conditions, or the correlation between a participant's speed and accuracy.

It refuses to overwrite an existing log without `--overwrite`, so a real study
cannot be clobbered by a synthetic run.

The tests:

```bash
python -m pytest timetabler/tests -q
```

## Credentials

Nothing reads a credential from a file in this repository. Either export the
variable:

```bash
export ANTHROPIC_API_KEY=...
```

or type it into the app's **Setup** page, where it is masked, held in memory
for that process only, and never written to the study log, an export, or
disk.

`python -m timetabler.eval.run_accuracy --list-providers` lists every
supported provider and the variable it reads. Ten are supported: Anthropic,
OpenAI, DeepSeek, OpenRouter, Groq, Together, Mistral, xAI, plus a local
Ollama or any OpenAI-compatible server.

## Deploying

`render.yaml` is a Render blueprint. **Read its header before collecting study
data.** The study log is SQLite on the local filesystem, so on a host with an
ephemeral disk a restart or redeploy discards every session recorded so far.
The blueprint therefore asks for a persistent disk and points `TIMETABLER_DB`
at it.

Set `TIMETABLER_DB` on any host to move the study log somewhere durable.

A free instance that sleeps after idle will also hand a participant a cold
start mid-task, which corrupts the time measurement that is the study's
primary outcome. Free instances are for demos; real sessions need a persistent
instance.

Participant data on a third-party host has to be named in the ethics
application, together with the model provider's data-retention terms.

## Layout

| Path | Contents |
|---|---|
| `timetabler/schema/` | Problem model and the frozen constraint vocabulary |
| `timetabler/data/` | Department instance and loader |
| `timetabler/parser/` | Prompt, provider clients, validation, entity resolution |
| `timetabler/solver/` | Cost model and simulated annealing |
| `timetabler/app/` | Streamlit UI, study logging, credential handling |
| `timetabler/eval/` | Labelled set, accuracy metrics, Cohen's kappa |
| `timetabler/tests/` | 301 tests |

## Status

The schema, solver, parser, verification UI and evaluation harness are built
and tested. Not done: the labelled corpus is 24 synthetic items for exercising
the harness, not the 150–300 human-authored sentences the study needs; no
accuracy figure has been produced against a live model; and solver validation
against ITC benchmark instances needs those instance files.
