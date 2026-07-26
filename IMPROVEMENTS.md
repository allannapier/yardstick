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
   project's own declared minimum of 3.10. Fixed; the cause was the absence of
   any CI that runs the tests.
2. **The core loop doesn't connect end to end.** `ys harness point` never tells
   the agent which model to ask for, so a real Claude Code session requests a
   model name the proxy has never heard of.
3. **Some headline numbers are wrong on real runs**, not just imprecise —
   background and subagent requests fabricate compaction events and cost is
   silently reported as $0 for models LiteLLM can't price.

Everything else is comparatively routine.

---

## P0 — broken right now

### 1. `render.py` did not parse on Python < 3.12 [verified] — fixed

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

### 2. There is no CI running the tests [verified] — fixed

`.github/workflows/` contained only `pages.yml`, which deploys the docs site.
Nothing ran `pytest`, which is exactly why a syntax error in a core module
shipped to `main`.

**Fix:** added `.github/workflows/tests.yml`, a test workflow on push/PR running
the suite on 3.10, 3.11, 3.12 and 3.13. The version matrix is the point — a
3.12-only CI would not have caught finding 1. It also runs `ruff check .` as a
lint pass in a separate job. `pyproject.toml` now declares a `dev` extra
(`pytest`, `ruff`) so `pip install -e ".[dev]"` is enough to reproduce CI
locally; the handful of real lint findings (unused imports, an unused local, a
lambda-assignment, an f-string without placeholders) are fixed alongside it.

### 3. `ys harness point` never sets the model, so real runs fail [verified] — fixed

`harness.point()` wrote only `ANTHROPIC_BASE_URL` and `ANTHROPIC_API_KEY`. But
`proxy.generate_config()` registers models under the experiment's *factor value*
(`probe-claude-mock`, `claude-sonnet-5`), and a Claude Code session left to itself
requests its own default model id. That name is not in `model_list`, so LiteLLM
rejects it and every request in the run fails.

The mock smoke test in `experiments/example.yaml` couldn't work as documented
either: nothing would ever ask for `probe-claude-mock`.

There is a second half to this. Claude Code issues background requests against a
small/fast model for things like title generation. Even with the main model
mapped, those requests carry a *different* model id and would 400 against the
same `model_list`.

**Fix:**

- `point()` is now arm-aware: `ys harness point claude-code --exp E --arm A`
  resolves the arm's `factors.model` and writes `ANTHROPIC_MODEL`,
  `ANTHROPIC_SMALL_FAST_MODEL` and `ANTHROPIC_DEFAULT_HAIKU_MODEL` (opencode gets
  a best-effort top-level `model: anthropic/<value>`) so background requests
  land on the same registered `model_name` as the main turn.
- `proxy.generate_config()` now always emits a `model_name: "*"` catch-all entry
  (LiteLLM's provider-wildcard routing) so any model id the experiment didn't
  declare is passed through to Anthropic and still recorded, rather than
  rejected — an unmeasured request is better than a broken run.
- `ys start` now queries the running proxy's `/v1/models` and warns loudly if the
  arm's model has no explicit `model_list` entry (it would only work via the
  catch-all, silently skipping any `mock_response`/params declared for it) or if
  the proxy can't be reached at all.
- The README quick start now points the harness with `--exp`/`--arm` so the model
  is pinned before the run starts, instead of pointing blind.

### 4. Background and subagent traffic corrupts conversation metrics [verified] — fixed

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

**Fix:** added a `thread_key` column to `requests`, assigned in
`collector._resolve_thread` by chain-following: a request joins the thread
whose most recent request shares its system prompt *and* is a plausible
parent per `_classify_transition` (continuation, compaction, or branch —
anything but "reset"); no match starts a new thread. This was originally a
static hash of the system prompt plus the first non-system message, which
broke on the exact case the fix exists for — a harness-side compaction that
summarizes/rewrites that first message would have looked like the start of a
brand new thread instead of a continuation of the real one. Chain-following
doesn't have that failure mode: a compaction is explicitly not a "reset", so
it still resolves to the same thread. Background/subagent calls, which get a
different system prompt, task instruction, or both, still correctly land in
their own thread even when interleaved mid-run — an interleaved background
call can no longer fabricate a compaction/reset event or break the main
thread's continuation chain.

`ys/metrics.py` computes turns, overhead, tool-call, redundancy, compaction,
and context-growth/cache-reuse metrics over the run's largest thread only
(`_main_requests`/`_main_tool_calls`, picked by request count via
`_main_thread_key`); `cost_usd`/`billable_tokens` still total across all
traffic, since that's real spend regardless of thread. Background/subagent
traffic is reported separately as `background_requests`/`background_tokens`
(surfaced in `ys end`'s headline metrics and in `compare`/`report`).

The run fingerprint (`model`/`toolset_hash`/`system_prompt_hash`) is corrected
at `finish_run` from `metrics.main_thread_fingerprint` — the first successful
request of the largest thread — since the eager per-request fill in the
collector can't yet know, mid-run, which thread will end up being the main
one. `ys/db.py`'s migration list gained a second entry (`thread_key` plus
per-request `toolset_hash`/`system_prompt_hash`, needed to recompute the
fingerprint); since a plain `ALTER TABLE ADD COLUMN` isn't safe to replay,
non-`CREATE ... IF NOT EXISTS` migrations are now Python callables that guard
their own column additions (`db._add_column_if_missing`).

### 5. `ys proxy down` silently orphans the proxy [verified] — fixed

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

**Fix:**

- `procutil.stop` now signals the whole process group (`os.killpg`, falling
  back to `os.kill` if the group lookup fails) so a worker `launch_detached`
  spawned under `start_new_session=True` is reachable too, not just the
  leader pid.
- After the grace period, a still-alive process is reported honestly — the
  pidfile is left in place (so it can still be found) instead of being
  removed regardless of outcome — and the message points at the new
  `ys proxy down --force` / `ys web down --force`, which escalates to
  `SIGKILL` and only clears the pidfile once the process is confirmed dead.
- `proxy up` and `web up` now probe the target port before starting; a port
  that's bound but has no live pidfile behind it (the exact state the old
  `stop` could leave behind) fails fast with an explanation instead of
  burning the 15s readiness timeout and blaming a log file.

---

## P1 — data integrity and robustness

### 6. SQLite is configured for a single writer [by inspection] — fixed

`db.connect()` set `foreign_keys` and nothing else. The collector writes from
inside the proxy process while the CLI and the dashboard read and write the same
file. Without WAL mode and a busy timeout, "database is locked" under concurrent
access is a matter of when, not if — and the collector's only failure handling was
`traceback.print_exc()` into the proxy log, so measurements would be lost
silently mid-run.

**Fix:**

- `db.connect()` now sets `PRAGMA journal_mode=WAL`, `PRAGMA busy_timeout=5000`
  and `PRAGMA synchronous=NORMAL` (plus a 5s `sqlite3.connect` timeout), so
  readers and the writer no longer block each other and a conflicting writer
  is waited out instead of failing immediately.
- `YardstickLogger._handle` retries a write up to 3 times with a short backoff
  on `sqlite3.OperationalError`, covering contention that outlasts the
  busy timeout.
- A request that still can't be written — lock contention exhausted, or any
  other failure — is now counted via `ys/dropped.py` (an append-only log
  under `YARDSTICK_HOME`) instead of only being printed to the proxy log.
  `ys status` and `ys end` surface the count so a lossy run is visible rather
  than quietly short.

### 7. `seq` assignment races [verified] — fixed

`_next_seq` did `SELECT COALESCE(MAX(seq),0)+1` outside any explicit
transaction, and writes are dispatched through a thread pool executor. Two
concurrent requests — routine with parallel tool use or a subagent — could read
the same maximum and both write it. There was no `UNIQUE(run_id, seq)`
constraint to catch it, and thread resolution (`_resolve_thread`, finding 4)
picks each thread's most recent request by `MAX(seq)`, so a collision also
corrupts the transition chain.

Reproduced directly: 20 threads calling `_write` concurrently against the same
run, with the fix below reverted, reliably raised `sqlite3.IntegrityError` from
the new unique index well before all 20 landed.

**Fix:**

- `collector._write` now allocates seq inside an explicit `BEGIN IMMEDIATE`
  transaction (previously the read and the insert were separate statements
  under SQLite's default deferred-transaction behavior, with no lock held
  between them). `BEGIN IMMEDIATE` takes the write lock up front, so a
  concurrent `_write` on the same database file blocks (honoring
  `busy_timeout`) until this one commits, instead of racing it.
- A migration adds `CREATE UNIQUE INDEX ... ON requests(run_id, seq)` as a hard
  backstop, so a collision that somehow still occurred would raise instead of
  silently corrupting the transition chain, and drops the now-redundant plain
  index from migration 1 (same leftmost columns, so keeping both would only be
  write-amplification). Since a database written before the `BEGIN IMMEDIATE`
  fix may already have real duplicate `(run_id, seq)` rows on disk — which
  would make `CREATE UNIQUE INDEX` fail outright — the migration checks for
  duplicates first and, only if any exist, renumbers every run's requests
  densely in `(seq, id)` order (`id`, the autoincrement rowid, preserves
  actual write order) before indexing; a database with no duplicates — every
  one created after this fix — skips that full-table rewrite entirely.
- `YardstickLogger._handle`'s existing write-retry loop now also retries on
  `sqlite3.IntegrityError`, not just `OperationalError`, so a request that
  still lost the unique-index race self-heals on the next attempt (fresh
  transaction, fresh seq) instead of being dropped.

### 8. No schema versioning [verified] — fixed

The schema was applied with `CREATE TABLE IF NOT EXISTS` and nothing else. Every
fix in this document that adds a column (`thread_key`, `cost_source`,
`task_snapshot`) would have silently not applied to any existing database, and
queries would fail against tables that were created by an older version.

**Fix:** `ys/db.py` now tracks `PRAGMA user_version` against an ordered
`MIGRATIONS` list (currently one entry: the original schema). `init_db()`
applies only the migrations above the database's current version, in order,
committing after each one, so a fresh install and an old database both
converge to the same state. Future schema changes are appended as new list
entries rather than edits to migration 1. A database created before this
change has every table but `user_version = 0`; replaying migration 1's
`CREATE TABLE IF NOT EXISTS` / `CREATE INDEX IF NOT EXISTS` against it is a
no-op that still advances the version, which `tests/test_db.py` covers
directly.

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

## P1 — residual gaps left by the fixes above

Findings 25–29 came out of reviewing the merged fixes for findings 3, 4 and 6
rather than the original pass over the code. None of them reopen the finding
they belong to — each fix does what it claims — but each is a place where the
fix rests on an assumption that isn't enforced, or is narrower than the finding
it closed.

### 25. Thread separation depends entirely on the system prompt hash [by inspection]

`_resolve_thread` (finding 4) only considers a candidate thread whose most
recent request shares this request's `system_prompt_hash`. That check is doing
*all* of the work, because `_classify_transition` tests `len(cur) < len(prev)`
before it tests the shared prefix: any shorter history is classified
"compaction", and `_resolve_thread` accepts "compaction" as a plausible parent.

So a subagent or background call that happened to share the main thread's
system prompt would be pulled into the main thread as a fabricated compaction
event — the exact failure finding 4 exists to prevent. It doesn't happen today
because Claude Code's `Task` subagents and its title-generation calls both
carry a different system prompt, but that is a property of one harness at one
version, not of the algorithm.

The ambiguity is real and not obviously resolvable from message hashes alone: a
genuine harness-side compaction rewrites the early history into a summary, so
its shared prefix with the pre-compaction history is also 0 — indistinguishable
from an unrelated shorter conversation on hashes alone.

**Fix:** at minimum, state the assumption in `_resolve_thread`'s docstring and
add a test that pins the behaviour when the system prompts *do* match, so a
future change to the transition classifier can't silently widen the hole. A
stronger version needs a second signal — a large drop in message count with a
zero shared prefix looks much more like a new thread than like a compaction,
and compaction candidates could require the request to be within some ratio of
the parent's size.

### 26. The "main" thread is whichever thread is largest [by inspection]

`metrics._main_thread_key` picks the `thread_key` with the most requests, ties
broken by earliest `MIN(seq)`. A long-running `Task` subagent can plausibly
issue more requests than the conversation that spawned it, and when it does it
becomes "the main thread": `turns`, the compaction metrics, and the run's
corrected fingerprint (`main_thread_fingerprint`, written at `finish_run`) all
come from the subagent instead of the driving conversation.

**Fix:** prefer the thread containing the run's first request — the driving
conversation is the one that starts the run — or keep the largest-thread rule
but warn when the largest thread doesn't contain `seq = 1`, which is the case
where the two rules disagree.

### 27. Pinning the small/fast model routes background traffic through the arm's model [by inspection]

Finding 3's fix writes `ANTHROPIC_SMALL_FAST_MODEL` and
`ANTHROPIC_DEFAULT_HAIKU_MODEL` to the arm's model alongside `ANTHROPIC_MODEL`.
Background traffic that a real session would send to a small, cheap model
therefore runs on the arm's model, and `cost_usd`/`billable_tokens` — which are
deliberately run-wide, since that spend is real — are inflated relative to an
unmeasured session. `background_requests`/`background_tokens` (finding 4) make
the traffic *visible* but don't remove it from the cost totals.

The same PR added the `model_name: "*"` catch-all, so the requests would now
route and record correctly *without* being pinned. Pinning is still the right
default for a mock experiment — unpinned background traffic would bypass
`mock_response` and hit the real API during what is supposed to be a dry smoke
test — so this is a trade-off to make explicit rather than a bug to reverse.

**Fix:** document the effect where `point()` writes the variables, and consider
`ys harness point --no-pin-background` (or pinning only when the arm's model
declares a `mock_response`) so a real cost comparison can opt out. Overlaps
findings 9 and 10, which have to price this traffic correctly either way.

### 28. Only the collector retries a locked write [by inspection]

Finding 6 gave `YardstickLogger._handle` a retry loop, but every other writer —
`ys start`, `ys end`, `ys runs delete`, the dashboard — goes through a bare
`db.cursor()`. WAL plus `busy_timeout=5000` makes contention unlikely rather
than impossible, and a write that outlasts the busy timeout still surfaces as
an unhandled `database is locked` traceback from the CLI, with `ys end` (which
races the tail of the proxy's in-flight writes) the most exposed.

**Fix:** move the retry into a helper alongside `db.cursor()` and use it for the
CLI/dashboard write paths too, rather than duplicating the loop per call site.

### 29. `ys start` silently skips its own model check without a master key [by inspection]

The `model_available` warning added for finding 3 is guarded by
`if model and master_key`, where `master_key` is read from the `ys start`
process's environment. Run `ys proxy up` in one terminal and `ys start` in
another without re-exporting `LITELLM_MASTER_KEY` and the check doesn't run,
doesn't warn that it didn't run, and the user proceeds believing a verified
proxy is serving their model — a first-run scenario, and the one finding 3 is
about.

**Fix:** print a "couldn't verify — `LITELLM_MASTER_KEY` not set in this shell"
line in that branch. Belongs to the same diagnostic surface as findings 12
and `ys doctor` (feature 4).

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

- ~~`output/joes-shears-*/` — 12 files of generated website artifacts from some past
  experiment run are committed to the repo. These are run outputs, not source;
  gitignore them.~~ Untracked and gitignored (whole `output/` directory); the
  files themselves were left on disk since they're the user's data, not the repo's.
- ~~`explore/` — ad-hoc probe scripts committed at the root. Valuable as provenance
  for the collector's field paths (the collector docstring cites them), so keep
  them, but move them under `tools/` or `docs/provenance/` and say so in the
  README.~~ Moved to `docs/provenance/` with `git mv`; README now says so, and
  every reference to the old `explore/` path (the collector docstring,
  `.gitignore`, and the relocated scripts' own hardcoded paths) was updated.
- ~~`pyproject.toml` has no dev-dependency group; `pytest` isn't declared
  anywhere despite the README's `pytest` instruction.~~ Fixed alongside
  finding 2: a `dev` extra now declares `pytest` and `ruff`.

---

## Suggested sequencing

**Milestone 1 — make it run.** Findings 1 (done), 2 (done), 3 (done), 5 (done). Add
the test matrix CI first so the rest is defended. After this, a first-time user
can complete the README quick start with a real agent.

**Milestone 2 — make the numbers trustworthy.** Finding 8 (migrations, done —
this is what the rest of the milestone builds its schema changes on), then 4
(done), 6 (done), 7 (done), 9, 11, 12, 13, 14, plus the residual gaps those
fixes left: 25 and 26 (both decide which requests the headline metrics are
computed over), 27 (alongside 9 and 10, since all three are about cost being
right), and 28. This is the batch that decides whether the tool's output can be
believed; nothing above it matters if `compaction_events` and `cost_usd` are
wrong.

**Milestone 3 — make it usable.** The dashboard defect table (19–24), the HTML and
experiment-discovery fixes, `ys doctor`, `ys runs list`, and the unattributed
surface — with finding 29 folded into the same diagnostic pass.

**Milestone 4 — make it a lab.** Unattended runs, workspace isolation, real
statistics. Resolve the dead-config items (15–18) here, since most of them are the
schema hooks these features need.

**Milestone 5 — breadth.** Additional providers and coding tools, export, budget
guards.

The ordering matters more than usual: features 1 and 3 from Milestone 4 are what
make the tool genuinely useful, but building them on top of unreliable transition
classification and silently-zero costs would produce confident, wrong answers
faster.
