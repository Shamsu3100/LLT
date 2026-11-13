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
