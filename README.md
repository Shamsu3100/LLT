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
