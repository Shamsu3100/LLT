# LLT — LLM-Assisted Constraint Elicitation for Timetabling

Staff write timetabling constraints in plain English, an LLM converts them to
formal constraints, staff verify each one, and a solver builds the timetable.
The system exists to measure three things: how much staff time the assisted
path saves, how many constraint errors survive verification, and whether staff
trust the result.

Research code. `timetabler/README.md` documents the design decisions and the
measurement conventions in detail; `paper/README.md` covers the two drafts.

---

## Quick start

```bash
pip install -r requirements.txt
```

```bash
python -m pytest timetabler/tests -q
```

345 tests, about two minutes. If they pass, everything below will run.

---

## Running everything

Python 3.11 or newer. Every command is run from the repository root.

### 1. The verification UI

```bash
python -m streamlit run timetabler/app/main.py
```

Use `python -m streamlit`, not the bare `streamlit` command: on a machine with
more than one interpreter the console script may belong to an installation that
has no `pydantic`.

Five pages. **Setup** takes a provider credential, masked and held in memory
only. **Input** collects constraints, as free text or through the baseline form
depending on the condition. **Review** shows each parsed constraint as a
sentence to accept, edit or reject. **Result** builds and shows the timetable.
**Log** is the researcher's view of recorded sessions.

### 2. The offline accuracy harness

No credential needed for the self-test:

```bash
python -m timetabler.eval.run_accuracy --client gold
```

`--client gold` replays the gold labels as predictions, so every figure should
read 100%. Anything less is a defect in the metrics or the resolver. It caught
one during development.

Against a live model, saving the raw replies:

```bash
python -m timetabler.eval.run_accuracy --client anthropic --save-responses runs/r1.json --out runs/report1.json
```

Re-score without calling the API again:

```bash
python -m timetabler.eval.run_accuracy --client recorded --responses runs/r1.json
```

### 3. The solver

```bash
python -m timetabler.solver.run --iterations 40000 --restarts 3
```

Constraints come from the labelled set's gold labels, so this exercises the
whole chain: sentence, resolver, materialise, cost model. On the sample
department it settles at one unsatisfiable hard constraint and names it.

### 4. A synthetic study

```bash
python -m timetabler.eval.synth --participants 30 --out logs/synthetic.sqlite
```

### 5. Exporting study data

```bash
python -m timetabler.eval.export_study --db logs/synthetic.sqlite --tasks timetabler/eval/labelled/sample_tasks.json --out exports/
```

### 6. The papers

```bash
cd paper && latexmk main.tex
```

```bash
cd paper && latexmk workshop.tex
```

---

## Credentials

Nothing reads a credential from a file in this repository. Either export the
variable:

```bash
export ANTHROPIC_API_KEY=...
```

or type it into the app's **Setup** page, where it is masked, held in memory
for that process only, and never written to the study log, an export, or disk.

```bash
python -m timetabler.eval.run_accuracy --list-providers
```

| Provider | Credential | Notes |
|---|---|---|
| `anthropic` | `ANTHROPIC_API_KEY` | Default model `claude-sonnet-5-5` |
| `openai` | `OPENAI_API_KEY` | |
| `deepseek` | `DEEPSEEK_API_KEY` | |
| `openrouter` | `OPENROUTER_API_KEY` | Many vendors through one key |
| `groq`, `together`, `mistral`, `xai` | `<NAME>_API_KEY` | |
| `ollama`, `local` | none needed | A local Ollama, vLLM or llama.cpp server |
| `stub` | none | Offline; returns nothing |

Everything except Claude speaks the OpenAI chat-completions API, so one client
covers them all. A run is named by a `provider:model` spec:

```bash
python -m timetabler.eval.run_accuracy --client deepseek:<model> --list-models
```

Model ids are not hardcoded for non-Anthropic providers: they change often and
vary by account, and a stale default would fail at the first call or, worse,
silently run a model the paper does not name.

---

## Collecting study data

Point the log somewhere fixed, so it does not depend on the launch directory:

```bash
export TIMETABLER_DB=~/llt-study/study.sqlite
```

Run one session per participant per condition. Set the participant code,
condition, task and order position in the sidebar, click **Start session**,
work through Input, Review and Result, then **Finish session**. Every action is
timestamped into that one file as it happens, so a crash loses at most the
current phase. Counterbalancing is yours to fix in advance: the app records the
order position you enter, it does not assign one.

Back up after each participant — it is a single file:

```bash
cp ~/llt-study/study.sqlite ~/llt-study/backups/after-P01.sqlite
```

Then export:

```bash
python -m timetabler.eval.export_study --tasks timetabler/eval/labelled/sample_tasks.json --out exports/
```

That writes `sessions.csv` (one row per participant-condition, with entry time,
verification time and total as separate columns, plus the residual-error
breakdown), `constraints.csv` (one row per constraint, with what was proposed,
what was decided, and what changed), and `events.csv` (the full log). It prints
a per-condition summary and names any session that was abandoned or left with
an open phase.

**Abandoned sessions are excluded from the printed means but kept in the CSV.**
An unclosed phase is closed at the last event seen, so a session left open
overnight would otherwise swamp its condition's mean.

### Task keys

Residual error is the gap between the set a participant finally accepted and
the set the **task** asked them to express, so each task needs a key of
intended constraints. `timetabler/eval/labelled/sample_tasks.json` holds four
starter tasks of 7–8 constraints each.

**A brief must be written as facts, not as the sentences you want back.** If a
task says "You cannot teach on Fridays", the free-text arm copies that sentence
and the form arm transcribes it, and the study measures typing speed instead of
elicitation. The starter briefs are shaded grids, inventory extracts, complaint
logs and email threads.

**Tasks must carry at least six constraints.** With three, a participant's
error rate can only land on 0, 0.33, 0.67 or 1.0 — coarser than the
5-percentage-point non-inferiority margin the study has to test. Measured
across eight synthetic seeds at a fixed injected error rate, enlarging the
tasks cut the across-seed spread of the mean from 0.000–0.178 to 0.058–0.116.

Run `validate_against_department` before the pilot. A key naming an entity the
department does not contain scores every participant wrong for the researcher's
reason, not theirs — the export refuses to run in that case.

### Synthetic studies

Generates a full within-subject study at the real shape, so the analysis can be
dry-run before anyone is recruited. **It says nothing about the result** — the
effect is whatever `--time-reduction` put there, so recovering it is circular.
Three uses are legitimate:

1. **Pipeline validation** — the columns are right, the exclusion logic fires,
   the scoring aligns, the export is clean.
2. **A null check** — `--time-reduction 0`, then confirm the analysis reports
   no difference. A pipeline that finds an effect in null data is broken.
3. **A power sanity check** — generate at the target effect and confirm the
   planned sample can detect it.

It cannot reproduce real between-participant variance, which is what actually
determines power, nor which constraints humans misread and why, learning across
conditions, or the correlation between a participant's speed and accuracy.

It refuses to overwrite an existing log without `--overwrite`, so a real study
cannot be clobbered by a synthetic run.

---

## Deploying

`render.yaml` is a Render blueprint. **Read its header before collecting study
data.** The study log is SQLite on the local filesystem, so on a host with an
ephemeral disk a restart or redeploy discards every session recorded so far.
The blueprint asks for a persistent disk and points `TIMETABLER_DB` at it.

For the study itself, running locally is the better instrument, not a
compromise: the primary outcome is a duration, and remote hosting puts network
latency and its variance directly into the dependent variable.

Participant data on a third-party host has to be named in the ethics
application, together with the model provider's data-retention terms.

---

## Layout

| Path | Contents |
|---|---|
| `timetabler/schema/` | Problem model and the frozen constraint vocabulary |
| `timetabler/data/` | Department instance and loader |
| `timetabler/parser/` | Prompt, provider clients, validation, entity resolution |
| `timetabler/solver/` | Cost model and simulated annealing |
| `timetabler/app/` | Streamlit UI, study logging, credential handling |
| `timetabler/eval/` | Labelled set, task keys, metrics, export, synthetic studies |
| `timetabler/tests/` | 345 tests |
| `paper/` | Both article drafts |

---

## Status

Built and tested: the schema, solver, parser, verification UI, study logging,
task keys, export, and the evaluation harness.

Not done:

- The labelled corpus is 24 synthetic items for exercising the harness, not the
  150–300 human-authored sentences the study needs.
- No accuracy figure has been produced against a live model.
- Solver validation against ITC benchmark instances needs those instance files.
- Ethics approval is pending, so no study session has been run.
- The papers' results, discussion and conclusion sections are headers only.
