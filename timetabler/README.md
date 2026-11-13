# timetabler

LLM-assisted constraint elicitation for timetabling. Implements
`nl_timetabling_build_plan_v3.md`: the frozen constraint schema (Phase 2), the
solver (Phase 3), the LLM parser with entity resolution and rule-based
flagging (Phase 4), the verification UI and study logging (Phase 5), and the
offline translation-accuracy harness (Phase 6A).

Not done: benchmark validation of the solver (Phase 3.5) needs the ITC
instance files, which are not in the repository, and no run has yet been made
against the live API.

## Install

```bash
pip install -r requirements.txt
```

`anthropic` is needed only for live API runs. Everything else, the test suite
included, runs offline.

## Run the accuracy harness

```bash
python -m timetabler.eval.run_accuracy --client gold
```

`--client gold` replays the gold labels as predictions. It is a self-test of
the harness: every figure should read 100%, and anything less is a defect in
the metrics or the resolver. It caught one such defect during development, a
case mismatch between gold ids and resolver output that left type and entity
accuracy at 100% while exact match collapsed to 16.7%.

Live run. Set the credential in your own shell — none is read from a file in
this repository:

```bash
export ANTHROPIC_API_KEY=...
```

```bash
python -m timetabler.eval.run_accuracy --client anthropic --save-responses runs/r1.json --out runs/r1_report.json
```

The same command against any other provider, which is how the Phase 6A
model-choice sweep is run:

```bash
python -m timetabler.eval.run_accuracy --client deepseek:<model> --out runs/deepseek.json
```

The run opens with one cheap `ping` call, so a wrong model id or a billing
problem surfaces immediately instead of a third of the way through 300 items
(`--no-ping` skips it). Token usage and estimated cost are printed and written
into the report.

Re-score without further API calls:

```bash
python -m timetabler.eval.run_accuracy --client recorded --responses runs/r1.json
```

**Before any live run on staff text:** hold the provider's data-retention
policy in writing, and state it in the ethics application and the paper (build
plan §5, Privacy).

## Run the solver

```bash
python -m timetabler.solver.run --iterations 40000 --restarts 3
```

Constraints come from the labelled set's gold labels, so the run exercises the
whole chain: sentence, resolver, `materialise`, cost model. Anything carrying a
blocking flag is dropped, as the verification step would drop it.

On the sample department this settles at one unsatisfiable hard constraint:
`CS202L` is asked for `R203`, which holds 30 against the course's 38 students.
The solver keeps the structural constraint and reports the elicited one by its
source sentence, which is what Phase 5.3 needs in order to ask staff to relax
something.

## Run the verification UI

```bash
python -m streamlit run timetabler/app/main.py
```

Use `python -m streamlit`, not the bare `streamlit` command: on a machine with
more than one interpreter the console script may belong to an installation
that has no `pydantic`.

Pick a participant code and a condition in the sidebar, then work through
Input, Review and Result. The Log page is for the researcher and exports one
session as JSON.

Three conditions, matching Phase 6C:

| Condition | Entry | Verification |
|---|---|---|
| 1 `form` | Dropdowns only, identical for every participant | Nothing to verify |
| 2 `llm` | Free text through the parser | Accept / edit / reject per constraint |
| 3 `llm_no_verify` | Free text through the parser | Skipped by design (stretch arm) |

Without an API key the parser is stubbed and the sidebar says so. A session
recorded against the stub is a rehearsal, not study data.

## Tests

```bash
python -m pytest timetabler/tests -q
```

## Layout

| Path | Phase | Contents |
|---|---|---|
| `schema/entities.py` | 1 | `Course`, `Room`, `Slot`, `Lecturer`, `Group`, `Department` |
| `schema/constraints.py` | 2 | The frozen vocabulary: 8 solvable types, 2 open categories |
| `data/` | 1 | `sample_department.json` and its loader |
| `parser/prompt.py` | 4.1 | System prompt, department register, 16 few-shot examples |
| `parser/client.py` | 4.2 | `AnthropicClient`, `StubClient`, per-call run metadata |
| `parser/parse.py` | 4.2 | JSON extraction, validation, the single retry |
| `parser/resolve.py` | 4.3–4.6 | Entity resolution, rule-based flags |
| `solver/problem.py` | 3 | Precomputed lookups: valid starts, blocks, groupings |
| `solver/timetable.py` | 3.1 | `course -> (room, slot)`, text grid rendering |
| `solver/violations.py` | 3.2, 3.4 | One function per type, plus structural checks |
| `solver/cost.py` | 3.2, 3.6 | Cost model, breakdown, partial evaluation |
| `solver/annealing.py` | 3.3 | Simulated annealing, three move types |
| `solver/run.py` | 3 | CLI runner |
| `app/verbalise.py` | 5 | Constraints as readable English |
| `app/store.py` | 5.4 | SQLite event log: sessions, events, constraints |
| `app/timing.py` | 5.4 | Entry and verification durations, throughput |
| `app/recorder.py` | 5.4 | The recording API the UI drives |
| `app/main.py` | 5 | Streamlit pages: Input, Review, Result, Log |
| `eval/dataset.py` | 6A | Labelled-set loading, labeller pairing |
| `eval/metrics.py` | 6A | Accuracy measures, entity breakdown, Cohen's kappa |
| `eval/run_accuracy.py` | 6A | CLI runner and report |

## The constraint vocabulary

Frozen after Week 3 (build plan Phase 2). Eight solvable types:
`lecturer_unavailable`, `lecturer_prefers_time`, `course_requires_room`,
`course_requires_room_type`, `max_consecutive_hours`, `no_early_start`,
`courses_not_same_time`, `courses_same_day`.

Two categories stay open, and neither reaches the solver:

- `unsupported` — a schedulable request no frozen type expresses.
- `needs_clarification` — input too ambiguous to formalise, carrying the
  question to put to the user.

Both are logged and categorised. The build plan treats their distribution as a
finding, not a defect.

## Design decisions worth knowing

**The LLM outputs constraints and nothing else** (Phase 4.5). It never sees or
produces a timetable. SCHEDBench's 2026 results support the split: 13 models
averaged 15.2% feasibility on natural-language scheduling, around 3.3% on the
ITC-2007 timetabling family.

**Entity resolution is separate from the LLM call.** The model copies the words
staff used; resolution against the department register happens afterwards in
`parser/resolve.py`. This keeps the output contract simple and makes resolution
failures measurable on their own.

**Descriptive references are tracked as their own class** (Phase 4.6). Staff
write "the big lab" and "the first years", not `R201` and `G1A`. LlmAcq (JAIR
2026) found the same pattern in free-text constraint feedback: users lean on
implicit structural knowledge and may name no identifier at all. A description
read as the wrong room is counted separately from a hallucinated room, because
they are different failures with different fixes.

**Flagging is by rule, never by self-reported confidence** (Phase 2, option
(a)). The `confidence` field exists on the schema and is unused: it is not
requested from the model and nothing reads it. Option (b), a calibration study,
is therefore a config change and not a schema change.

**Flags are `block` or `review`.** A blocking flag keeps a constraint away from
the solver: an unresolved entity, an ambiguous entity, an open category, a
period outside the grid. A review flag surfaces it prominently and lets the
user decide: a descriptive reference, a weak partial match, a strength cue
contradicting the parsed strength, a room too small for the course. Human
verification is mandatory either way (Phase 5.2).

## The solver

**Structural constraints are separate from elicited ones.** No lecturer, room
or group may be double-booked, a room must hold its course, and a block must
fit its day. These hold for every instance, are never elicited, and are always
hard. They are computed in one pass over the occupied cells; the named
functions in `violations.py` read from that pass, so they cannot drift from
what the cost model charges.

**Hard constraints are charged at 1000 and soft weights run 1–10**, so no
accumulation of soft penalties can buy a single hard breach. An unweighted soft
constraint gets 5, matching what the prompt tells the model to use when a
speaker gives no sense of strength.

**Violation functions return counts, not booleans**, so two breaches of one
constraint score worse than one. Where the count is a matter of convention the
docstring says which was chosen: `max_consecutive_hours` charges the excess
periods per over-long run, `courses_not_same_time` charges per overlapping
pair, and `courses_same_day` charges the number of courses that would have to
move to join the day already holding most of them.

**Move generation cannot leave the grid.** `ProblemIndex.valid_starts` holds
only the starts where a course's whole block fits its day, and a swap that
would place a course badly is refused. No overflow penalty is needed, and
`block_fits_day` exists to catch a timetable built outside the search.

**The cost model evaluates moves partially.** A move changes only the
constraints whose scope it meets, and only the structural clashes among courses
sharing a lecturer, group or room with it. Recomputing the whole cost per
iteration is O(courses x constraints) and will not survive an ITC-sized
instance.

Partial evaluation is only as good as its scope calculation, and an
under-reported scope would make the search optimise a cost that silently drifts
from the real one — with every individual violation function still correct. Two
things guard it: `test_delta_matches_full_recompute_over_a_random_walk` walks
600 random moves asserting the tracked cost still equals a full recompute, and
`SimulatedAnnealer(verify_every=N)` re-derives the cost during a live search
and raises on disagreement.

Measured effect of the two optimisations against the first working version:
570 to 200 microseconds per iteration on the sample department. Precomputing
the `(course, slot)` block table removed 430,000 tuple rebuilds per 8,000
iterations; collapsing three clash passes into one and dropping `Counter` for
plain dicts removed the rest.

**Annealing is seeded and deterministic**, so a seed and settings reproduce a
run exactly (build plan §2). The initial temperature is calibrated per instance
by sampling worsening moves for a target acceptance rate, since the cost scale
depends on how many hard constraints an instance carries and a temperature
tuned on one department would be wrong for the next. `solve_best_of` runs
independent restarts, which are easier to reason about than reheating.

## Providers

Phase 6A treats model choice as a sensitivity axis, so every backend sits
behind one contract and a run is identified by a `provider:model` spec.

```bash
python -m timetabler.eval.run_accuracy --list-providers
python -m timetabler.eval.run_accuracy --client openai:<model> --list-models
```

| Provider | Credential | Notes |
|---|---|---|
| `anthropic` | `ANTHROPIC_API_KEY` | Default model `claude-sonnet-5-5` (build plan §2) |
| `openai` | `OPENAI_API_KEY` | |
| `deepseek` | `DEEPSEEK_API_KEY` | |
| `openrouter` | `OPENROUTER_API_KEY` | Reaches many vendors through one key |
| `groq`, `together`, `mistral`, `xai` | `<NAME>_API_KEY` | |
| `ollama`, `local` | none needed | A local Ollama, vLLM or llama.cpp server |
| `stub` | none | Offline; returns nothing |

Everything except Claude speaks the OpenAI chat-completions API, so one client
covers them all; only the base URL, the credential and the model id change.
`--base-url` points any of them at a self-hosted endpoint or proxy.

**Model ids are not hardcoded for non-Anthropic providers.** They change often
and vary by account, so a guessed default would fail at the first call or,
worse, silently run a model the paper does not name. Pass the id, or use
`--list-models` to see what a credential can reach.

**Costs are only reported for rates that can be verified.** The shipped table
holds Anthropic rates only; any other model reports `cost_usd: null` instead
of a guess. Supply yours with `register_pricing(model, input, output)` before
quoting a figure in the paper.

### Keeping the comparison fair

A cross-provider difference is only about the model if the instrument is
identical, so the same system prompt, the same JSON-in-the-prompt contract,
the same Pydantic validation and the same single retry run everywhere. Two
provider features are therefore **off by default**:

- **Native JSON mode** (`--json-mode`). The Claude path does not use it, so
  turning it on for one arm alone would confound that arm.
- **`temperature`** (`--temperature`). Current Claude models reject it
  outright, so a run that sets it is not comparable to a Claude run.

`--effort` maps to Claude's `output_config.effort` and to `reasoning_effort`
where a provider has one. Semantics differ between providers, so treat it as a
per-provider setting, not a shared scale.

## The live Claude path

**No sampling parameters.** `temperature`, `top_p` and `top_k` were removed
from the Messages API and are not accepted by the SDK at all — passing
`temperature` raises a `TypeError` before a request is built. Reproducibility
rests instead on pinning the model id, the prompt digest and the effort level,
and on `--save-responses`, which records every raw reply so a scored run can be
re-scored without calling the API again.

**Effort replaces the temperature knob.** `--effort low|medium|high|xhigh|max`
sets reasoning depth. The default is `medium`: an extraction task that still
has to judge hard/soft ambiguity. It is a cost lever and a sweep axis, so
Phase 6A's sensitivity analysis should vary it alongside prompt wording and
model choice.

**No server-side refusal fallback.** A fallback would silently answer some
items on a different model, and a measurement attributed to one model has to
come from that model. A refusal is recorded as a failed item with its category
instead, and counted.

**The system prompt is cached.** It is the same ~2,700 tokens on every item —
schema, department register, 16 few-shot examples — while the user turn is one
short sentence, so a 300-item run would otherwise resend it 300 times. With
`cache_control` on it, most of that input bills as cache reads at a twentieth
of the rate. Check `usage.cache_read_input_tokens` in the report: a run showing
zero cache reads throughout is paying full price, which means the prefix fell
below the model's minimum cacheable length.

**Three outcomes are never accepted as a result:**

| Outcome | Handling |
|---|---|
| `stop_reason: "refusal"` | Recorded with its category. Not retried — the same text declines again. |
| `stop_reason: "max_tokens"` | Recorded as truncated. A partial constraint set must not be read as a complete one. |
| An API error (429, 5xx, network) | Recorded as that item's outcome, so a 300-item batch does not die on one failure. |

Only a schema-validation failure triggers the single retry, which is what
Phase 4.2 specifies. Retries are billed, so `cost_usd` sums every attempt.

## The verification UI and study log

**Constraints are shown as sentences, never as JSON.** "Dr. Sarah Lee cannot
teach on Wednesday and Friday: HARD". A participant who cannot read the
restatement is not verifying anything, and the residual-error measure would
end up measuring the interface instead of the parser. The wording lives in
`verbalise.py` as pure functions, so it is tested without a browser.

**Entry time and verification time are logged separately**, and both are
reported alongside the total. The nearest timed study of a natural-language
interface bundled re-prompting and checking into one completion time and could
not separate them, leaving its 31% speed-up open to the objection that time
saved on entry reappears as verification time. One total cannot answer that.

**Durations are derived from timestamped phase spans**, not from a running
stopwatch. A participant who moves between Input and Review is measured
correctly, and the raw log can be re-analysed later under a different idle
rule without collecting new data. A span longer than `idle_threshold` is
reported raw *and* capped; neither is privileged, and which one the paper
leads with belongs in the pre-registration.

**The recorder refuses to hold two phases open at once**, so entry and
verification can never both be running and count the same seconds twice.

**Nothing in `recorder.py`, `store.py`, `timing.py` or `verbalise.py` imports
Streamlit.** The interface is a view over the recorder, so what the study
records does not depend on Streamlit's rerun behaviour, and the measurement
logic is tested directly.

Two bugs here were found by running the app, not by reading it: `streamlit
run` executes the entry point as a script so its relative imports failed, and
a default SQLite connection is bound to its creating thread while Streamlit
reruns on a new thread each time — which raised on the first logged action of
a real session. Both are pinned by tests.

## Metric conventions

Each one a choice, not a default:

- Gold labels hold canonical ids, so predictions are scored through the
  resolver. Translation accuracy measures the parser and resolver as one
  pipeline, which is what staff actually use.
- `source_text` is excluded from comparison. Two labellers will disagree about
  span boundaries without disagreeing about the reading.
- `weight` is excluded by default. A soft weight is a scale choice, not a
  reading. `--include-weight` scores it.
- Alignment is one-to-one and greedy, since one sentence may yield several
  constraints. Unmatched predictions count as spurious, unmatched gold as
  missed.
- Cohen's kappa needs one category per item, so the **primary (first)**
  constraint of each sentence supplies the type and strength labels. The report
  states this and counts how many sentences held more than one constraint, so
  the cost of the convention is visible. Set-level exact agreement is reported
  alongside and uses every constraint.

## The seed labelled set is not the Phase 6A corpus

`eval/labelled/seed.jsonl` holds 24 synthetic items covering all ten types,
with two labellers who disagree on five of them (type kappa 0.81, strength
kappa 0.91). It exists to develop and test this harness.

Phase 6A needs **150–300 human-authored sentences**, labelled independently by
two people. Every item carries `source`, and `--only-staff` restricts a run to
`staff_authored` items; the runner prints a warning when a set contains none.
The distinction is load-bearing: existing scheduling benchmarks render their
natural language from templates, and SCHEDBench's own limitations section names
human-authored phrasing as what it lacks. The human corpus is the contribution.
