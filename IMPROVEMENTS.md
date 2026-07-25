# yardstick — review and improvement plan

A review of the whole rig: setup, experiment definition, running, analysis, the
dashboard, and the breadth of supported models and coding tools. Findings are
split into bug fixes and new features, then sequenced into milestones at the end.

Findings marked **[verified]** were reproduced against the code in this repo.
Findings marked **[by inspection]** are read from the source and look certain but
were not executed (usually because they need a live proxy and a real agent).

## Verdict

The architecture is sound and the code is unusually well-commented about its own
compromises. The measurement model — sit in the request path as a LiteLLM proxy,
hash message histories, derive metrics offline from SQLite — is the right shape
for this problem, and the CLI/dashboard split through `ys/runs.py` is clean.

Three things stand between it and being trustworthy:

1. **It doesn't run.** `ys/render.py` was a `SyntaxError` on every Python below
   3.12, so `ys compare`, `ys report`, and the entire dashboard were dead on the
   project's own declared minimum of 3.10. Fixed on this branch; the cause was
   the absence of any CI that runs the tests.
2. **The core loop doesn't connect end to end.** `ys harness point` never tells
   the agent which model to ask for, so a real Claude Code session requests a
   model name the proxy has never heard of.
3. **Some headline numbers are wrong on real runs**, not just imprecise —
   background and subagent requests fabricate compaction events and cost is
   silently reported as $0 for models LiteLLM can't price.

Everything else is comparatively routine.

---

## P0 — broken right now

### 1. `render.py` did not parse on Python < 3.12 [verified] — fixed on this branch

```python
f"{' <span class=\"warn\">UNCONTROLLED</span>' if a.fingerprint_drifted else ''}</th>"
```

Backslashes inside f-string expressions are only legal from Python 3.12 (PEP 701).
`pyproject.toml` declares `requires-python = ">=3.10"`. On 3.10 and 3.11 this is a
hard `SyntaxError`, so `import ys.render` fails — taking out `ys compare`,
`ys report`, and `ys/web/app.py`, which imports `render` at module scope.

`tests/test_render.py` and `tests/test_web.py` could not even be collected. The
suite reported "62 passed" while 18 tests were silently never running. With the
fix applied it is 80 passed.

### 2. There is no CI running the tests [verified]

`.github/workflows/` contains only `pages.yml`, which deploys the docs site.
Nothing runs `pytest`, which is exactly why a syntax error in a core module
shipped to `main`.

**Fix:** add a test workflow on push/PR running the suite on 3.10, 3.11, 3.12 and
3.13. The version matrix is the point — a 3.12-only CI would not have caught
finding 1. Add a lint pass (`ruff`) in the same workflow.

### 3. `ys harness point` never sets the model, so real runs fail [by inspection]

`harness.point()` writes only `ANTHROPIC_BASE_URL` and `ANTHROPIC_API_KEY`. But
`proxy.generate_config()` registers models under the experiment's *factor value*
(`probe-claude-mock`, `claude-sonnet-5`), and a Claude Code session left to itself
requests its own default model id. That name is not in `model_list`, so LiteLLM
rejects it and every request in the run fails.

The mock smoke test in `experiments/example.yaml` cannot work as documented
either: nothing will ever ask for `probe-claude-mock`.

There is a second half to this. Claude Code issues background requests against a
small/fast model for things like title generation. Even with the main model
mapped, those requests carry a *different* model id and will 400 against the same
`model_list`.

**Fix, in order of value:**

- Make `point` arm-aware: `ys harness point claude-code --exp E --arm A`, writing
  `ANTHROPIC_MODEL` (and `ANTHROPIC_SMALL_FAST_MODEL` /
  `ANTHROPIC_DEFAULT_HAIKU_MODEL`) to the arm's model key.
- Emit a catch-all entry in the generated proxy config so unmapped model ids are
  passed through and *recorded* rather than rejected — an unmeasured request is
  better than a broken run.
- Have `ys start` verify the running proxy actually serves the arm's model, and
  fail loudly if not.
- Fix the README quick start ordering, which currently points the harness before
  the arm is known.

### 4. Background and subagent traffic corrupts conversation metrics [verified]

`_classify_transition` compares each request's message-hash list against the
single previous request in the same run. Claude Code interleaves background calls
and `Task` subagent conversations into the same run — each has its own unrelated,
much shorter history. Replaying that pattern through the real function:

```
main turn 1     msgs=2  -> None
main turn 2     msgs=4  -> continuation
background req  msgs=2  -> compaction     <- fabricated
main turn 3     msgs=6  -> reset          <- main thread chain destroyed
```

One background request invents a compaction event and breaks the continuation
chain for everything after it. Every metric derived from `transition` is
unreliable on real runs: `compaction_events`, `tokens_dropped`,
`turns_to_recompaction`, `post_compaction_regrowth`.

The same root cause poisons the run fingerprint. `_fill_fingerprint_if_missing`
records `model`, `toolset_hash` and `system_prompt_hash` from the *first
successful* request — if that happens to be a background call, the run is
fingerprinted against the wrong conversation, and `render._fingerprint_drifted`
then flags arms `UNCONTROLLED` essentially at random.

It also inflates `turns` (a background title generation is not a turn) and
distorts `overhead_tokens_per_turn`, which extrapolates from request 0 alone.

**Fix:** add a `thread_key` column to `requests`, derived from the system prompt
hash plus the hash of the first message. Classify transitions within a thread,
compute conversation metrics over the main thread (the one with the most
requests), and report background traffic as its own line item — "N background
requests, M tokens" is itself a useful harness-efficiency number.

### 5. `ys proxy down` silently orphans the proxy [verified]

```
stop() returned after 5.0s: 'stopped process (pid 11395)'
process actually still alive?  True
pidfile still on disk?         False
```

`procutil.stop` sends one `SIGTERM`, waits 5s, then removes the pidfile and
reports success regardless of whether the process died. There is no `SIGKILL`
escalation. Because `launch_detached` uses `start_new_session=True`, the signal
also never reaches any worker the proxy forked.

The user is left with a process still holding port 4000 and no pidfile to find it
by. The next `ys proxy up` then fails after a 15-second readiness timeout with a
message that points at a log file rather than at the real cause.

**Fix:** signal the process group (`os.killpg`), escalate to `SIGKILL` after a
grace period, only remove the pidfile once the process is confirmed gone, and
report honestly when it isn't. Add `ys proxy down --force` and have `proxy up`
detect a port that is bound but unowned.

---

## P1 — data integrity and robustness

### 6. SQLite is configured for a single writer [by inspection]

`db.connect()` sets `foreign_keys` and nothing else. The collector writes from
inside the proxy process while the CLI and the dashboard read and write the same
file. Without WAL mode and a busy timeout, "database is locked" under concurrent
access is a matter of when, not if — and the collector's only failure handling is
`traceback.print_exc()` into the proxy log, so measurements would be lost
silently mid-run.

**Fix:** `PRAGMA journal_mode=WAL`, `PRAGMA busy_timeout=5000`,
`PRAGMA synchronous=NORMAL`. Retry the collector write, and count dropped records
so a lossy run is visible rather than quietly short.

### 7. `seq` assignment races [by inspection]

`_next_seq` does `SELECT COALESCE(MAX(seq),0)+1` outside any explicit
transaction, and the writes are dispatched through a thread pool executor. Two
concurrent requests — routine with parallel tool use or a subagent — can read the
same maximum and both write it. There is no `UNIQUE(run_id, seq)` constraint to
catch it, and `_last_msg_hashes` orders by `seq DESC`, so a collision also
corrupts the transition chain.

**Fix:** add `UNIQUE(run_id, seq)`, allocate inside a `BEGIN IMMEDIATE`
transaction, and retry on conflict. Consider ordering by the autoincrement `id`
and treating `seq` as a derived presentation value.

### 8. No schema versioning [by inspection]

The schema is applied with `CREATE TABLE IF NOT EXISTS` and nothing else. Every
fix in this document that adds a column (`thread_key`, `cost_source`,
`task_snapshot`) will silently not apply to any existing database, and queries
will fail against tables that were created by an older version.

**Fix:** `PRAGMA user_version` plus an ordered migration list applied in
`init_db()`, before anything else here lands.

### 9. Cost is silently $0 for models LiteLLM cannot price [by inspection]

`response_cost` comes straight from LiteLLM's cost map. For a model id not in that
map — which includes `claude-sonnet-5` as configured in
`experiments/interactive-sonnet.yaml`, and anything served via a custom deployment
name — LiteLLM returns 0.0. `cost_usd`, `cost_per_success` and every delta
computed from them then read as zero with no indication anything is wrong.

For a tool whose headline output is a cost comparison, silently reporting $0 is
the worst available failure mode.

**Fix:** record a `cost_source` per request (`litellm` / `declared` / `unknown`).
Flag any request with non-zero tokens and zero cost. Let an experiment declare
per-model input/output/cache prices and compute cost from tokens when LiteLLM
can't. Surface "cost unavailable for model X" prominently in `compare` and
`report` rather than printing a confident 0.0000.

### 10. `billable_tokens` hardcodes one vendor's cache weights [by inspection]

`input + cache_creation + output + cache_read * 0.1` bakes in Anthropic's
cache-read discount, and treats cache *writes* at 1.0 when Anthropic prices them
at roughly 1.25x. It is also meaningless across providers with different cache
economics — which the rig otherwise wants to compare.

**Fix:** make the weights per-model configuration with the current values as the
Anthropic default, and document that `billable_tokens` is a pricing-weighted
proxy, not a token count.

### 11. `ys end` drops in-flight requests [by inspection]

`finish_run` clears the active-run state file, and `_resolve_run_id` falls back to
that file for any harness that can't set `x-ys-run`. A response that lands after
`ys end` is attributed to `unattributed` — so the tail of every run, including the
final and often largest turn, can go missing.

**Fix:** keep the state file until a short drain window has passed, or record
`ended_at` and attribute by timestamp window rather than by file presence.

### 12. Unattributed traffic is invisible [by inspection]

Requests that can't be attributed land in a synthetic `unattributed` run. Nothing
in the CLI or the dashboard ever surfaces it. A user who misconfigures the harness
sees a run with zero requests and no explanation anywhere — which is precisely the
situation finding 3 puts everyone in on their first real run.

**Fix:** surface the unattributed count in `ys status`, `ys end` and the dashboard
banner: "42 requests since 14:02 could not be attributed to a run". This is the
single highest-value diagnostic in the app.

### 13. `--force` leaves the previous run orphaned forever [by inspection]

`state.set_active(force=True)` overwrites the active slot, but nothing closes the
run it displaced. That run keeps `ended_at`, `task_success` and `wall_clock_s`
NULL permanently. `aggregate_run_metrics` counts every run in `n_runs` regardless
of whether it finished, so each forced start permanently depresses the arm's
success rate.

**Fix:** mark the displaced run abandoned and exclude unfinished runs from
aggregation, reporting them separately.

### 14. `compare` mixes runs from different versions of the experiment [by inspection]

`render.py`'s docstring already admits this: `experiments.config_yaml` and
`task_json` are overwritten on every `ys start`, so there is no per-run record of
what a given run actually executed. `_run_ids_for_arm` then aggregates *every* run
ever recorded for that arm id — including runs from before you changed the task,
the success check, or the model.

The guardrail that is supposed to refuse mismatched tasks only compares today's
YAML against the single stored row, which was itself overwritten by the most
recent start.

**Fix:** snapshot `task_json` and a hash of the config onto each run row at
`begin_run`. Group by that hash in `compare`, use only the current group by
default, and warn when an arm's history spans more than one.

### 15–18. Declared configuration that does nothing [by inspection]

The YAML schema advertises more than the code implements. Each of these is either
a feature to build or a field to delete, but shipping them as inert config is
worse than either:

- **`metrics.gate` / `primary` / `secondary` / `derived`** — `render.py` hardcodes
  its own metric lists and `aggregate_run_metrics` hardcodes the gate. Both
  example experiments carefully declare a `metrics:` block that has no effect.
  `derived: [tokens_per_turn]` names a metric that is never computed at all.
- **`factors`** — the declared factor space is never validated against the arms.
  Nothing catches an arm referencing a model with no `models:` entry or a factor
  key that doesn't exist, and there is no generate-the-cartesian-product helper,
  which is the obvious reason to declare it.
- **`task.repo` / `task.ref` / `task.prompt_file`** — parsed, stored, never read.
  `prompt_file` in particular is the hook for reproducible runs (see feature 1).
- **`ttft_ms`** — a schema column, hardcoded `None` in `extract_record`. Time to
  first token is one of the more interesting harness-level metrics and LiteLLM
  exposes it for streaming responses.

Also: `repeats` is advisory. Nothing warns when one arm has 7 runs and another has
1, which makes the comparison invalid in a way the table does not show.

---

## P2 — the dashboard

All six of these were reproduced against `ys/web/app.py`.

| # | Defect | Behaviour |
|---|---|---|
| 19 | Invalid experiment name in a URL | `GET /experiments/foo.bar` → **HTTP 500**. `store.experiment_path` raises `InvalidExperimentName`, which no route catches. |
| 20 | Non-numeric `timeout_s` or `repeats` | **HTTP 500**. `int(form.get(...))` is unguarded. |
| 21 | Creating an experiment that already exists | Silently overwrites the YAML — verified changing `task.id` on an experiment that already had runs, with no warning. The old runs stay attached to the same experiment id and are then aggregated together with the new ones. |
| 22 | Validation failure | Raw pydantic error text is URL-encoded into a query string (`?error=1%20validation%20error%20for%20Experiment...`) and every field the user typed is discarded by the redirect. |
| 23 | Starting the proxy for a nonexistent experiment | No existence check; reports whatever the proxy layer complains about first (`LITELLM_MASTER_KEY is not set`) and redirects to a page that then 500s. |
| 24 | Two baseline arms | The form uses checkboxes with a JS-synchronised `value`, so two can be checked; the failure surfaces only as a raw pydantic string via defect 22. Should be radio buttons. |

Beyond the defects, the dashboard is thin in ways that matter:

- **Invalid HTML**: `<a href="..."><button></button></a>` in `index.html` and
  `experiment.html`. Interactive elements cannot be nested; this breaks keyboard
  and assistive-technology behaviour. Use a styled link or a form button.
- **It can't see the repo's own experiments.** `store.list_experiments()` reads
  only `~/.yardstick/experiments`, so `experiments/example.yaml` — the file the
  README tells you to use — never appears. The docs and the UI disagree about
  where experiments live.
- **No edit, no YAML view, no delete.** An experiment is write-once through the
  form; any change means hand-editing a file the UI won't show you.
- **Nothing updates during a live run.** No auto-refresh, no running token/cost
  counter, no request feed. During the one phase where the user is watching, the
  dashboard is a static page.
- **The comparison view escapes the app.** `/experiments/{name}/compare` returns
  the standalone report document with no shell and no way back.
- **Run detail omits the useful parts**: no `success_output` (the check's own
  output, the first thing you want when a run fails), no notes, no factors, no
  per-turn chart.
- **Only the `model` factor is expressible** in the new-experiment form, though
  `Arm.factors` is an arbitrary dict — so the harness-vs-harness comparison the
  tool is named for can't be set up from the UI at all.
- The mock model id hardcoded in the form (`claude-3-5-sonnet-20241022`) is stale.

And on the CLI side: there is no `ys runs list`. Runs can be deleted by id but
never enumerated, so the only way to find an id is the dashboard or raw SQL.

---

## Features

### 1. Unattended runs — the highest-value addition

Today every repeat is hand-driven: point the harness, start, drive the agent
yourself, end. That is the biggest threat to the rig's own validity, because the
human in the loop is an uncontrolled variable across repeats and arms, and
`repeats: 3` means doing it three times by hand.

`task.prompt_file` already exists in the schema for this. Build
`ys run --exp E --arm A --repeats N` that invokes the agent non-interactively
(`claude -p`, `opencode run`, `codex exec`), loops the repeats, and scores each
one. This turns the tool from a logger into an experiment runner and makes
overnight matrices possible.

### 2. Workspace isolation per run

`success_check` runs via `shell=True` in whatever directory `ys end` happened to
be invoked from. There is no working directory, no setup, no teardown, and no
reset between repeats — so repeat 2 starts from whatever state repeat 1's agent
left on disk. `task.repo` and `task.ref` are declared but unused.

Add `task.workdir`, `task.setup`, `task.teardown`, and a per-run git
worktree/clone from `repo`@`ref` so every repeat starts from an identical tree.
Without this, repeats measure a moving target and the error bars are not what they
appear to be.

### 3. Statistics worth the name

`aggregate_run_metrics` reports a mean and a population standard deviation over
n=3. `render` shows `± spread` and a percentage delta. Nothing indicates whether a
difference between arms is real, which is the only question the user actually has.

Add bootstrap confidence intervals on each metric, a permutation or Mann-Whitney
test on the primary metric, a Wilson interval on success rate, and an explicit
verdict line: *"arm-b is 18% cheaper, but with n=3 the difference is not
distinguishable from noise; ~12 repeats needed."* Also add a minimum-detectable-
effect helper so users can choose `repeats` deliberately.

### 4. `ys doctor`

There are many moving parts — home directory, schema version, proxy process,
generated config, harness config, two API keys, active-run state — and the failure
modes are mostly silent. A single preflight command that checks all of them,
verifies the running proxy serves the current experiment's models, and reports
unattributed request counts would prevent most of the wasted runs this review
found paths to.

### 5. Provider and harness breadth

**Models.** Everything assumes Anthropic: the fallback is `anthropic/<value>`, the
harness writes `ANTHROPIC_*` variables, the web form defaults to a Claude id, and
`billable_tokens` bakes in Anthropic cache weights. LiteLLM already brokers
OpenAI, Gemini, Bedrock, Vertex and local models — and cross-model comparison is
the tool's entire premise. Make the model config provider-agnostic and map harness
environment variables per provider.

**Coding tools.** Two are supported: Claude Code and opencode. Worth adding, in
rough order of demand: Codex CLI, Gemini CLI, Aider, Cursor CLI, Copilot CLI,
Cline/Roo. Also support project-level `.claude/settings.json`, not just `~`.

**Harness config safety.** `point()` writes the API key in plaintext into the
user's real `~/.claude/settings.json` and leaves it there if anything dies before
`reset`. Add `ys harness point --env-only`, which prints exports and never touches
the user's config — safer, and it works for agents with no config file at all.
Also register an automatic reset on `ys end`. The known `.jsonc` comment-stripping
loss on opencode configs is documented but is still destructive to a real user
file; the `--env-only` path sidesteps it entirely.

### 6. Smaller additions

- `ys export --csv` / `--json` so results can leave the tool.
- Budget guard: `ys start --budget 5.00` warns or aborts when a run's recorded
  cost crosses a threshold.
- Cross-experiment comparison / leaderboard across tasks.
- Per-turn drill-down charts in the dashboard, not only in the static report.

---

## Housekeeping

- `output/joes-shears-*/` — 12 files of generated website artifacts from some past
  experiment run are committed to the repo. These are run outputs, not source;
  gitignore them.
- `explore/` — ad-hoc probe scripts committed at the root. Valuable as provenance
  for the collector's field paths (the collector docstring cites them), so keep
  them, but move them under `tools/` or `docs/provenance/` and say so in the
  README.
- `pyproject.toml` has no dev-dependency group; `pytest` isn't declared anywhere
  despite the README's `pytest` instruction.

---

## Suggested sequencing

**Milestone 1 — make it run.** Findings 1 (done), 2, 3, 5. Add the test matrix CI
first so the rest is defended. After this, a first-time user can complete the
README quick start with a real agent.

**Milestone 2 — make the numbers trustworthy.** Findings 8 (migrations, first),
then 4, 6, 7, 9, 11, 12, 13, 14. This is the batch that decides whether the tool's
output can be believed; nothing above it matters if `compaction_events` and
`cost_usd` are wrong.

**Milestone 3 — make it usable.** The dashboard defect table (19–24), the HTML and
experiment-discovery fixes, `ys doctor`, `ys runs list`, and the unattributed
surface.

**Milestone 4 — make it a lab.** Unattended runs, workspace isolation, real
statistics. Resolve the dead-config items (15–18) here, since most of them are the
schema hooks these features need.

**Milestone 5 — breadth.** Additional providers and coding tools, export, budget
guards.

The ordering matters more than usual: features 1 and 3 from Milestone 4 are what
make the tool genuinely useful, but building them on top of unreliable transition
classification and silently-zero costs would produce confident, wrong answers
faster.
