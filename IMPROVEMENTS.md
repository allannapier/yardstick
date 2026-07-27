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

### 9. Cost is silently $0 for models LiteLLM cannot price [by inspection] — fixed

`response_cost` comes straight from LiteLLM's cost map. For a model id not in that
map — which includes `claude-sonnet-5` as configured in
`experiments/interactive-sonnet.yaml`, and anything served via a custom deployment
name — LiteLLM returns 0.0. `cost_usd`, `cost_per_success` and every delta
computed from them then read as zero with no indication anything is wrong.

For a tool whose headline output is a cost comparison, silently reporting $0 is
the worst available failure mode.

**Fix:**

- `requests` gained a `cost_source` column (migration 4, `ys/db.py`, following
  migration 2's `_add_column_if_missing` pattern) recording how
  `ys/collector.py`'s new `_resolve_cost` obtained a request's cost: `litellm`
  when LiteLLM's own number is nonzero; `declared` when it was zero but an
  experiment's `pricing:` block could price the tokens instead; `unknown` when
  neither could — i.e. exactly the silent-$0 case this finding is about, now
  tagged instead of trusted.
- `Experiment` gained a `pricing: dict[str, ModelPricing]` field
  (`ys/experiment.py`), keyed by `factors.model` value exactly like the
  existing `models:` block rather than a parallel mechanism — declaring
  optional `input_per_mtok` / `output_per_mtok` / `cache_write_per_mtok` /
  `cache_read_per_mtok` prices. `experiments/interactive-sonnet.yaml` (the
  file this finding was reproduced against) now declares them for
  `claude-sonnet-5`.
- The collector runs inside the separate proxy process, not the CLI process
  that loaded the experiment YAML, so it recovers a run's declared pricing
  from `experiments.config_yaml` — already stored verbatim at `ys start`
  (`ys/runs.py`'s `begin_run`) and the same field `finish_run` reads for
  `task.success_check` — rather than needing new plumbing between the two.
  `resolve_model_key` (`ys/experiment.py`) bridges a provider-prefixed
  recorded model id (`anthropic/claude-sonnet-5`) against the bare
  `factors.model` key `pricing:` is declared under.
- `ys compare`/`ys report` (`ys/render.py`) now surface any arm with an
  `unknown`-cost request the way `UNCONTROLLED` surfaces fingerprint drift: a
  `[COST UNKNOWN]` marker on the arm's header *and* on its `cost_usd`/
  `cost_per_success` cells specifically (in the HTML report, a `<span
  class="warn">COST UNKNOWN</span>` header badge plus a `?` marker on the
  same cells), plus an explicit, unmissable line per (arm, model) — e.g.
  "cost unavailable for model 'claude-sonnet-5' in arm 'arm-sonnet' (12
  request(s)) — LiteLLM has no price for it and the experiment declares no
  `pricing:` override for it. cost_usd and cost_per_success for this arm are
  undercounted, not merely imprecise." — printed after the table by `ys
  compare` and rendered as a bordered banner (not a footnote) right under the
  `<h1>` in the HTML report.

### 10. `billable_tokens` hardcodes one vendor's cache weights [by inspection] — fixed

`input + cache_creation + output + cache_read * 0.1` bakes in Anthropic's
cache-read discount, and treats cache *writes* at 1.0 when Anthropic prices them
at roughly 1.25x. It is also meaningless across providers with different cache
economics — which the rig otherwise wants to compare.

**Fix:**

- `Experiment` gained a `billable_weights: dict[str, BillableWeights]` field
  (`ys/experiment.py`), keyed by `factors.model` value like `models:`/
  `pricing:`. `BillableWeights` has four fields (`input`, `output`,
  `cache_creation`, `cache_read`) defaulting to Anthropic's shape — and the
  cache-write default is now `1.25`, not the old `1.0`, which was simply
  wrong (Anthropic bills a 5-minute cache write at a premium over a plain
  input token, not at parity).
- `ys/metrics.py`'s `token_metrics` computes `billable_tokens` per request,
  weighted by *that request's own* `model` (a run can in principle mix
  models — background/catch-all traffic), via a new
  `billable_weights_by_model` parameter threaded through
  `compute_run_metrics`/`aggregate_run_metrics`. A model with no declared
  override falls back to the Anthropic-shaped `DEFAULT_BILLABLE_WEIGHTS`. `ys
  compare`/`ys report` (`ys/render.py`) pass the experiment's declared
  weights explicitly; `ys end`'s immediate single-run summary (`ys/runs.py`)
  uses the default, since it has no `Experiment` object at hand — a
  deliberate scope cut (see the PR description) rather than an oversight.
- The module comment above `token_metrics`, and `BillableWeights`'s own
  docstring, now say explicitly that `billable_tokens` is a
  *pricing-weighted proxy* for spend, not a token count, and are meaningless
  as a cross-provider constant.

### 11. `ys end` drops in-flight requests [by inspection] — fixed

`finish_run` clears the active-run state file, and `_resolve_run_id` falls back to
that file for any harness that can't set `x-ys-run`. A response that lands after
`ys end` is attributed to `unattributed` — so the tail of every run, including the
final and often largest turn, can go missing.

**Fix:** chose the timestamp-window approach (record `ended_at`, attribute by
elapsed time) over literally keeping `active.json` alive longer, for two reasons
named directly in the finding: `ys end` is a synchronous CLI command the user is
waiting on, so a fix that makes it sleep before returning has a real UX cost and
was ruled out; and `active.json`'s *presence* is also what `state.set_active`/
`ys status`/the next `ys start` treat as "a run is in progress" — stretching its
lifetime would make a fast `ys start` right after `ys end` see a phantom active
run.

- `ys/runs.py`'s `finish_run` now calls a new `state.mark_ended(run_id,
  experiment, arm, ended_at)` immediately before its existing
  `state.clear_active()` (unchanged, still unconditional and immediate — `ys
  status`/the next `ys start` see the slot free right away, exactly as before).
  `mark_ended` writes a *separate* file, `ys/paths.py`'s new
  `LAST_ENDED_RUN_PATH`, recording the run id plus an absolute
  `time.time() + state.DRAIN_WINDOW_S` (60s) deadline.
- `ys/collector.py`'s `_resolve_run_id` — running in the separate proxy process,
  hence still file-based cross-process state rather than an in-memory handoff —
  gained a third fallback after the `x-ys-run` header and `active.json`:
  `state.get_draining_run()`, which returns the most recently ended run's id only
  if `time.time()` hasn't yet passed its recorded deadline, else `None` (falling
  through to `unattributed` as before). This directly attributes by elapsed time
  rather than by a file's mere existence, per the finding's second suggested fix.
- What this does *not* solve, deliberately: `finish_run` still computes
  `summary_metrics` (and the corrected fingerprint) from the rows in the database
  at the moment `ys end` runs. A request that lands during the drain window is
  now attributed to the correct run in the database — `ys compare`/`ys report`/a
  later re-query will see it — but it will not retroactively appear in the
  `ys end` command's own printed summary for that invocation. The finding
  explicitly separates "attributed to the wrong run" (fixed here) from "landed
  after metrics were computed" (a different problem, out of scope for this fix).
- No schema/migration change: both `LAST_ENDED_RUN_PATH` and the drain-window
  fallback are file-based, like the existing `active.json` mechanism they sit
  next to, and never touch the `runs` table.
- Regression tests in `tests/test_state.py`, `tests/test_collector.py` and
  `tests/test_runs.py` pin: `finish_run` leaves a draining record behind
  (`test_finish_run_leaves_a_draining_record_behind_for_stragglers`);
  `_resolve_run_id` falls back to it after `ys end` has already cleared
  `active.json` (`test_resolve_run_id_falls_back_to_the_draining_run_after_ys_end`);
  a request resolved that way lands in the *correct* run's rows in the database,
  not `unattributed`
  (`test_write_attributes_a_post_end_request_to_the_run_it_belongs_to` — the
  literal "still attributed to the run it belongs to" case); and the fallback
  expires once the window elapses, so a request arriving long after an unrelated
  run ended is not misattributed to it forever
  (`test_resolve_run_id_does_not_use_a_draining_run_past_its_window`). All four
  fail with the fix reverted (missing attributes / wrong run_id in the database)
  and pass with it restored.

### 12. Unattributed traffic is invisible [by inspection] — fixed

Requests that can't be attributed land in a synthetic `unattributed` run. Nothing
in the CLI or the dashboard ever surfaces it. A user who misconfigures the harness
sees a run with zero requests and no explanation anywhere — which is precisely the
situation finding 3 puts everyone in on their first real run.

**Fix:**

- `ys/runs.py` gained `unattributed_summary()`, querying
  `COUNT(*)`/`MIN(ts)` over `requests WHERE run_id = 'unattributed'` and
  returning a count plus the earliest request's `HH:MM` (the same `ts` format
  `_now()` already writes, sliced rather than reparsed).
- `ys/cli.py`'s `ys status` and `ys end` both print it via a shared
  `_print_unattributed_notice()`, right alongside the existing dropped-request
  count (finding 6) — its sibling diagnostic: dropped requests never made it
  into the database at all, unattributed ones did, just not under the run they
  belonged to. Phrasing follows the finding's own example: "N request(s) since
  HH:MM UTC could not be attributed to a run", plus a pointer at the likely
  cause (harness not pointed at the proxy / not sending `x-ys-run` / running
  outside an active run).
- The dashboard banner is deliberately **not** included in this fix — left as a
  scoped-out follow-up (`ys/web/**` was off limits for this change to avoid
  colliding with concurrent work already in flight there). `ys status`/`ys end`
  now cover the CLI half of the finding; the dashboard half is still open.
- Regression tests in `tests/test_runs.py` (`unattributed_summary` counts and
  reports the earliest timestamp) and `tests/test_cli.py` (`ys status`/`ys end`
  print the notice when unattributed requests exist, and stay silent when there
  are none) fail with the fix reverted and pass with it restored.

### 13. `--force` leaves the previous run orphaned forever [by inspection] — fixed

`state.set_active(force=True)` overwrites the active slot, but nothing closes the
run it displaced. That run keeps `ended_at`, `task_success` and `wall_clock_s`
NULL permanently. `aggregate_run_metrics` counts every run in `n_runs` regardless
of whether it finished, so each forced start permanently depresses the arm's
success rate.

**Fix:**

- `state.set_active` now closes the displaced run out before overwriting the
  active slot: a new `_abandon_displaced_run` (`ys/state.py`) stamps `ended_at`/
  `wall_clock_s` (the same way a normal `ys end` would) and sets a new
  `abandoned` column (migration 5, `ys/db.py`) to 1, guarded by
  `WHERE ended_at IS NULL` so a run that already finished by the time `--force`
  runs (e.g. `ys end` won a race) is left untouched. `task_success` is
  deliberately left NULL — the task was never actually scored, which isn't the
  same as failing it, so a forced start still shouldn't count as a recorded
  failure either.
- `aggregate_run_metrics` (`ys/metrics.py`) now excludes any run whose
  `task_success` is still NULL — covering both an `abandoned` run and one that
  simply never got an `ys end` for any other reason — from `n_runs`/`n_success`/
  `success_rate` entirely, instead of silently counting an unscored run against
  the arm forever. The excluded count is reported separately as the new
  `n_unfinished`, surfaced as an "unfinished (excluded)" row in `ys compare`'s
  table and the HTML report (`ys/render.py`) whenever any arm has one.
- Regression tests: `tests/test_state.py::test_force_marks_the_displaced_run_abandoned_and_closes_it_out`
  pins the abandon behavior (and
  `test_force_does_not_touch_a_run_that_was_already_closed` pins the
  `ended_at IS NULL` guard); `tests/test_metrics.py::test_aggregate_run_metrics_excludes_unfinished_runs_from_n_runs`
  proves a forced/unfinished run no longer depresses the arm's success rate
  (reverting either fix makes its respective test fail).

### 14. `compare` mixes runs from different versions of the experiment [by inspection] — fixed

`render.py`'s docstring already admits this: `experiments.config_yaml` and
`task_json` are overwritten on every `ys start`, so there is no per-run record of
what a given run actually executed. `_run_ids_for_arm` then aggregates *every* run
ever recorded for that arm id — including runs from before you changed the task,
the success check, or the model.

The guardrail that is supposed to refuse mismatched tasks only compares today's
YAML against the single stored row, which was itself overwritten by the most
recent start.

**Fix:**

- `begin_run` (`ys/runs.py`) now snapshots `task_json` onto the run row itself
  (`task_json_snapshot`, independent of the `experiments` row, which keeps being
  overwritten) plus a `config_hash` (migration 5, `ys/db.py`) computed by the new
  `runs.config_hash_for_arm`. The hash covers exactly `task` (id/repo/ref/
  prompt_file/success_check/timeout_s) and the specific arm's own `factors`
  (where `model` lives) — deliberately *not* the raw YAML text (a comment edit or
  a `question:` tweak would then spuriously split an arm's history for no real
  reason) and deliberately *not* `metrics:`/`pricing:`/`billable_weights:`/other
  arms' `factors` (those change how a run's numbers are displayed or priced
  after the fact, not what the agent was asked to do or how it was judged). A
  changed `task.success_check` or a changed `factors.model` under the same arm
  id — the two cases the finding calls out by name — always produce a different
  hash.
- `ys/render.py`'s `compare_experiment` groups each arm's run history by
  `config_hash` and aggregates only the group matching today's YAML by default.
  A run recorded under any other hash is excluded and named in the new
  `Comparison.config_warnings` (`render.config_warnings()`, printed by `ys
  compare` and rendered as a banner in the HTML report the same way finding 9's
  cost-unavailable banner is) — including how many runs were excluded and
  whether that's because the config actually changed or because the run
  predates this fix.
- Runs written before this fix have no `config_hash` at all (`NULL`). That group
  is never trusted as "current" by default, even when it's the only history an
  arm has — silently treating unverifiable old data as matching today's config
  is exactly the failure mode this finding is about, so the arm is instead
  reported as having no comparable data (with an explicit warning naming the
  excluded run count) until it's run again under the new schema.
- The previous "refuse mismatched task.id" guardrail (comparing today's YAML
  against the single, constantly-overwritten `experiments.task_json` row) is
  superseded by the per-arm, per-run-hash check above and has been removed.
- Regression tests: `tests/test_render.py::test_compare_experiment_excludes_runs_from_a_different_config_version`
  proves two config versions of the same arm no longer aggregate together, and
  `test_compare_experiment_never_trusts_a_pre_snapshot_run_as_current` /
  `test_compare_experiment_excludes_arm_entirely_when_no_runs_match_current_config`
  cover the no-snapshot and all-stale edge cases (reverting the grouping fix in
  `compare_experiment` makes all three fail); `tests/test_runs.py` pins what the
  hash covers directly (`test_config_hash_for_arm_changes_when_success_check_changes`,
  `test_config_hash_for_arm_changes_when_arm_model_factor_changes`,
  `test_config_hash_for_arm_unaffected_by_question_or_task_id_field`) and that
  `begin_run` actually snapshots both columns
  (`test_begin_run_snapshots_config_hash_and_task_json`).

### 15–18. Declared configuration that does nothing [by inspection] — fixed

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

**Fix:** built, not deleted, in every case — each field turned out to have a
real consumer worth writing rather than being genuinely dead:

- `metrics.gate`/`primary`/`secondary`/`derived`: **built.** `ys/experiment.py`'s
  `Metrics` now validates `gate` against `VALID_GATE_NAMES` and
  `primary`/`secondary`/`derived` against `VALID_METRIC_NAMES` at YAML-load
  time — an unknown or misspelled metric name (untrusted user input) now fails
  loudly, naming the valid options, instead of silently displaying nothing.
  `ys/metrics.py`'s new `resolve_gate` turns the validated `gate` string into
  the predicate `aggregate_run_metrics` gates on (previously hardcoded inline);
  `ys/render.py`'s `compare_experiment` passes it through and resolves
  `Comparison.primary_metrics`/`.secondary_metrics`/`.derived_metrics` from the
  experiment's declared lists (via `_resolve_metric_list`, which uses
  pydantic's `model_fields_set` so an explicit `secondary: []` — "show
  nothing here" — is distinguished from "not declared at all", which still
  falls back to the old hardcoded defaults for backward compatibility).
  `build_table`/`render_html` now iterate `Comparison.display_metrics()`
  instead of the old module-level `PRIMARY_METRICS`/`SECONDARY_METRICS`
  constants. `derived: [tokens_per_turn]` is no longer a metric name that
  goes nowhere: `ys/metrics.py` now actually computes it
  (`billable_tokens / turns` per run, added to `_EFFICIENCY_METRICS` so it's
  aggregated like every other metric).
- `factors`: **built.** `Experiment` gained a `_arms_within_declared_factor_space`
  validator (`ys/experiment.py`) that rejects an arm referencing a factor key
  not declared under `factors:`, or a value not in that key's declared list —
  both previously silent failures that only surfaced far downstream (a
  never-matching `models:`/`pricing:`/`billable_weights:` entry, or the
  proxy's catch-all quietly serving an unregistered model). Skipped entirely
  when `factors:` itself isn't declared, so an experiment not using the
  factor-space feature is unaffected. The missing "generate the cartesian
  product" helper the finding calls out now exists too:
  `expand_factors()`/`Experiment.factor_combinations()`.
- `task.repo`/`task.ref`/`task.prompt_file`: **kept, not deleted, and now
  validated.** These are the designated hooks for features 1 (unattended
  runs, `prompt_file`) and 2 (workspace isolation, `repo`/`ref`), both queued
  follow-up work — deleting them would just mean re-adding the same fields
  later. `Task` gained a schema-only `_ref_requires_repo` validator (a `ref`
  with no `repo` can never mean anything), and `ys/cli.py`'s `start()` now
  calls the new `validate_task_paths()` (`ys/experiment.py`) before claiming
  the active-run slot: a `prompt_file` that doesn't exist on disk fails
  loudly right there, per the finding's own suggested fix, instead of
  silently once feature 1 finally reads it. A `repo` that looks like a local
  filesystem path (no `://`, no `user@host:` syntax) is checked the same way;
  one that looks like a remote git URL is left unvalidated, since confirming
  it's reachable needs a network call that belongs to feature 2's own
  implementation. Both fields' docstrings in `ys/experiment.py` now say
  outright that they're declared-but-unconsumed pending those features.
- `ttft_ms`: **built.** `ys/collector.py`'s `extract_record` now computes it
  via a new `_ttft_ms` helper instead of hardcoding `None`. Checked against
  the installed `litellm` package rather than assumed: LiteLLM's own
  `standard_logging_object` already carries `startTime`/`completionStartTime`
  (both epoch-second floats), and for a **streaming** response LiteLLM's
  streaming handler stamps `completion_start_time` the moment the first
  chunk arrives, so `completionStartTime - startTime` is a real
  time-to-first-token. For a **non-streaming** response, LiteLLM never sets
  `completion_start_time` at all — `get_standard_logging_object_payload`
  defaults it to `end_time` — so `ttft_ms` collapses to the full round-trip
  latency for those requests. That's documented as correct, not a bug: a
  non-streaming response arrives as a single event, so "time to first token"
  and "time to last token" are genuinely the same moment. Returns `None`
  when either timestamp is missing (e.g. a failed request with no
  `standard_logging_object`) rather than fabricating a number.
- `repeats` advisory: **built.** `ys/render.py`'s new `repeat_count_warnings`
  flags exactly the case the finding describes — arms in the same comparison
  with unequal recorded run counts — printed by `ys compare` (yellow, since
  unlike an unpriced-cost warning this doesn't mean a number is wrong, just
  that comparing it isn't apples-to-apples) and as an amber banner in the
  HTML report. Not raised when every arm is short by the same amount (e.g.
  nobody's past repeat 1 yet), only when the arms disagree with each other.

---

## P1 — residual gaps left by the fixes above

Findings 25–29 came out of reviewing the merged fixes for findings 3, 4 and 6
rather than the original pass over the code. None of them reopen the finding
they belong to — each fix does what it claims — but each is a place where the
fix rests on an assumption that isn't enforced, or is narrower than the finding
it closed.

### 25. Thread separation depends entirely on the system prompt hash [by inspection] — fixed

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

**Fix:**

- `_resolve_thread`'s docstring now states the assumption directly: matching
  `system_prompt_hash` is treated as strong evidence of "same conversation",
  and that's only safe because every harness this rig currently drives gives
  background/subagent traffic a different system prompt from the main
  conversation — a property of those harnesses today, not a guarantee.
- The stronger fix: `_resolve_thread` no longer accepts every "compaction"
  transition as a plausible parent. A new `_plausible_compaction` check
  requires the candidate's message count to retain at least a third
  (`_MIN_COMPACTION_RATIO = 1/3`) of the parent's — a low bar for a real
  compaction (lossy, but not annihilative: the conversation keeps going) but
  usually well out of reach for an unrelated exchange that only coincidentally
  shares a system prompt, since a subagent or background call restarts from a
  small, roughly constant handful of messages regardless of how long the
  thread it's mistaken for has grown. This isn't a precise boundary — the
  ambiguity above is real and hashes alone can't fully resolve it — just a
  conservative floor against the worst case: a tiny unrelated exchange
  landing right after a long thread and being read as "compaction" on
  message-count alone. A "compaction" transition that fails the ratio falls
  through to starting its own thread instead of joining the parent's.
- Two new tests in `tests/test_collector.py` pin both directions:
  `test_write_rejects_a_same_system_prompt_short_unrelated_call_as_compaction`
  (a same-system-prompt exchange collapsing from 12 messages to 2 must start
  its own thread — this fails with a bare `AssertionError` against the
  pre-fix code, since the old code merges it as "compaction") and
  `test_write_still_follows_a_plausible_large_compaction` (a real compaction
  retaining exactly a third of the pre-compaction message count — the ratio's
  inclusive boundary — must still resolve to the same thread, which is the
  case the fix must not regress).

### 26. The "main" thread is whichever thread is largest [by inspection] — fixed

`metrics._main_thread_key` picks the `thread_key` with the most requests, ties
broken by earliest `MIN(seq)`. A long-running `Task` subagent can plausibly
issue more requests than the conversation that spawned it, and when it does it
becomes "the main thread": `turns`, the compaction metrics, and the run's
corrected fingerprint (`main_thread_fingerprint`, written at `finish_run`) all
come from the subagent instead of the driving conversation.

**Fix:** kept the largest-thread rule rather than overriding it to "whichever
thread contains the run's first request", and added a warning signal instead.
The override was tempting but isn't a strict improvement: it has its own
failure mode, already pinned by
`test_main_thread_fingerprint_prefers_largest_thread_over_first_request`
(finding 4) — a background call (e.g. title generation) that happens to be
logged as the run's very first request, ahead of the real conversation's first
turn, is a singleton that must not win the fingerprint just for being first.
Neither signal (request count, chronology) dominates the other, so overriding
to chronology would only trade one mis-attribution for the other rather than
fixing it.

Instead, `metrics._main_thread_started_run` reports whether the thread
`_main_thread_key` picked also contains `seq = 1` — false is exactly the case
where the two rules disagree, i.e. where a secondary thread has out-issued the
conversation that started the run. It's surfaced as `main_thread_started_run`
in `compute_run_metrics`'s output (via the new `main_thread_metrics`), a
boolean finding excluded from `_EFFICIENCY_METRICS` the same way
`overhead_drift` is — not a magnitude to average across repeats, and,
crucially, actually visible: `ys end` now prints a `[yellow]warning: ...[/yellow]`
line when it's false, rather than leaving the flag sitting unread in the
metrics dict (`compare`/`report`/the dashboard don't surface it yet — folded
into the later `ys doctor`/diagnostics pass instead, alongside findings 12 and
29, rather than reaching into `render.py`/the dashboard here). `cost_usd`/
`billable_tokens` are unaffected either way (still run-wide); `turns`, the
compaction metrics, and the corrected fingerprint still come from the
largest thread, now with a way to tell when that choice is questionable.
Two new tests in `tests/test_metrics.py` cover both the ordinary case and the
finding-26 scenario (a 3-request `subagent` thread outnumbering a 2-request
`main` thread that holds `seq = 1`).

### 27. Pinning the small/fast model routes background traffic through the arm's model [by inspection] — fixed

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

**Fix:**

- `harness.point()`'s docstring now spells out the trade-off in full where
  the two background-model variables are written: pinning them is what keeps
  a `mock_response` experiment a dry smoke test, at the cost of inflating
  run-wide `cost_usd`/`billable_tokens` relative to an unmeasured session,
  since Claude Code's background traffic then runs on — and is billed/
  weighted as — the arm's own model instead of a real session's cheap
  small/fast model.
- `point()` gained a `pin_background: bool = True` parameter gating the two
  `_deep_set` calls that write `ANTHROPIC_SMALL_FAST_MODEL`/
  `ANTHROPIC_DEFAULT_HAIKU_MODEL`; `ANTHROPIC_MODEL` itself is always set
  when a model is given, since the main turn still needs to be pinned either
  way. Exposed as `ys harness point --pin-background/--no-pin-background`
  (default: on, unchanged behaviour), per the plan's suggested opt-out; `ys
  start`'s own model-check warning is untouched. Passing `--no-pin-background`
  with a model also prints an explicit warning that background requests will
  use the harness's own default model instead of the arm's.
- Left as a documented trade-off, not reversed: the default is still to pin,
  because `experiments/example.yaml`'s mock smoke test — and any
  `mock_response`-based experiment — genuinely needs it to stay a dry run.
  Overlaps findings 9 and 10, which now price/weight this traffic correctly
  either way it's routed.

### 28. Only the collector retries a locked write [by inspection] — fixed

Finding 6 gave `YardstickLogger._handle` a retry loop, but every other writer —
`ys start`, `ys end`, `ys runs delete`, the dashboard — goes through a bare
`db.cursor()`. WAL plus `busy_timeout=5000` makes contention unlikely rather
than impossible, and a write that outlasts the busy timeout still surfaces as
an unhandled `database is locked` traceback from the CLI, with `ys end` (which
races the tail of the proxy's in-flight writes) the most exposed.

**Fix:**

- `db.call_with_retry(fn, *args, **kwargs)` is the one retry policy the
  collector and the CLI/dashboard now share, living next to `db.cursor()`.
  It retries `fn(*args, **kwargs)` up to `db.MAX_WRITE_ATTEMPTS` (3) times
  with a short backoff on `sqlite3.OperationalError`/`IntegrityError` — the
  same two exceptions and the same policy `_handle` had inline — and lets
  anything else propagate on the first attempt. `fn` is expected to open
  (and commit) its own `db.cursor()` per call, so a retried attempt starts
  from a clean transaction instead of replaying inside one that already
  failed partway through.
- `ys/collector.py`'s `YardstickLogger._handle` now calls
  `db.call_with_retry(_write, run_id, rec)` instead of its own inline loop;
  the existing tests that pin its retry behaviour (3 attempts, backoff,
  drop-and-record via `ys/dropped.py` once exhausted) pass unchanged, which
  is the evidence the refactor didn't change the policy.
- `ys/runs.py`'s `begin_run`, `finish_run` and `delete_run` — the functions
  `ys start`, `ys end`, `ys runs delete`, and the dashboard's equivalent
  routes all funnel through — now wrap each write in `db.call_with_retry`
  via a small `_with_cursor` helper, instead of a bare `with db.cursor()`.
  `ys end`'s `ended_at`/`task_success` write, the one this finding calls out
  as most exposed, is included.
- `ys/cli.py` and `ys/web/app.py` catch
  `(sqlite3.OperationalError, sqlite3.IntegrityError)` around the `runs.*`
  calls in `start`/`end`/`delete` (CLI) and the equivalent dashboard routes,
  and turn an exhausted retry into a readable message instead of an
  unhandled traceback (CLI) or a 500 (dashboard).

### 29. `ys start` silently skips its own model check without a master key [by inspection] — fixed

The `model_available` warning added for finding 3 is guarded by
`if model and master_key`, where `master_key` is read from the `ys start`
process's environment. Run `ys proxy up` in one terminal and `ys start` in
another without re-exporting `LITELLM_MASTER_KEY` and the check doesn't run,
doesn't warn that it didn't run, and the user proceeds believing a verified
proxy is serving their model — a first-run scenario, and the one finding 3 is
about.

**Fix:** `ys start` now has an `elif model and not master_key` branch that
prints "couldn't verify model '<model>' is registered on the proxy —
LITELLM_MASTER_KEY not set in this shell" instead of silently doing nothing.
The message itself lives in `proxy.model_check_skipped_message()` rather than
inlined at the one call site, so `ys doctor` (feature 4) and finding 12's
diagnostic surface can print the same wording later instead of inventing
their own.

---

## P2 — the dashboard

All six of these were reproduced against `ys/web/app.py`. **Fixed** (findings
19–24, plus the invalid-HTML bullet below the table).

| # | Defect | Behaviour |
|---|---|---|
| 19 | Invalid experiment name in a URL — fixed | `GET /experiments/foo.bar` → **HTTP 500**. `store.experiment_path` raises `InvalidExperimentName`, which no route catches. |
| 20 | Non-numeric `timeout_s` or `repeats` — fixed | **HTTP 500**. `int(form.get(...))` is unguarded. |
| 21 | Creating an experiment that already exists — fixed | Silently overwrites the YAML — verified changing `task.id` on an experiment that already had runs, with no warning. The old runs stay attached to the same experiment id and are then aggregated together with the new ones. |
| 22 | Validation failure — fixed | Raw pydantic error text is URL-encoded into a query string (`?error=1%20validation%20error%20for%20Experiment...`) and every field the user typed is discarded by the redirect. |
| 23 | Starting the proxy for a nonexistent experiment — fixed | No existence check; reports whatever the proxy layer complains about first (`LITELLM_MASTER_KEY is not set`) and redirects to a page that then 500s. |
| 24 | Two baseline arms — fixed | The form uses checkboxes with a JS-synchronised `value`, so two can be checked; the failure surfaces only as a raw pydantic string via defect 22. Should be radio buttons. |

**Fix:**

- 19 and 23 shared one root cause: `store.experiment_path`'s
  `InvalidExperimentName` and a plain missing-file check were each handled in
  some routes and not others. `app._load_experiment_or_404` centralizes both
  into a real `404` response rendered from a new `error.html` page (extends
  `base.html`, so it keeps the dashboard's own chrome instead of a bare
  traceback or FastAPI's default error body), and every route keyed by
  experiment name (`experiment_detail`, `experiment_compare`, `start_proxy`,
  `start_run`) now goes through it.
- 20 and 22 are the same shape of problem: a bad value used to either crash
  (`int(...)` on non-numeric input) or get caught only late, as a pydantic
  `ValidationError` whose raw text was URL-encoded into `?error=...` while the
  redirect threw away everything else the user had typed. `create_experiment`
  now guards the int conversions itself (`app._parse_int_field`) and
  translates pydantic's error list into short, readable per-field messages
  (`app._split_validation_errors` — errors that map to one input, like
  `task.id`, are shown next to that field; errors like "duplicate arm ids"
  that don't correspond to a single input are shown above the form). On any
  failure the form re-renders (HTTP 400) with the submitted values intact
  (`app._form_snapshot`) instead of redirecting to a query string.
- 21: `store.save_experiment` still writes `<name>.yaml` unconditionally, but
  `create_experiment` now refuses to call it when the file already exists,
  unless a `confirm_overwrite` checkbox (off by default, never implied) was
  submitted — and the refusal message names how many runs are already
  recorded against that experiment id, since those are exactly what would end
  up aggregated with whatever the new definition produces. This closes the
  silent-corruption path without removing the ability to deliberately
  replace a definition.
- 24: the baseline checkbox is now a radio button. Every row's radio shares
  the same `name="arm_baseline"`, so the browser's native radio-group
  behaviour makes "two arms marked baseline" unreachable from the form,
  closing the defect at the source rather than downstream in validation.
  Each row also gets a stable `arm_seq` assigned once at creation, and the
  server matches the single submitted `arm_baseline` value against each
  row's `arm_seq` — this is what let the old JS value-syncing hack (keeping
  the checkbox's `value` equal to the arm id text as the user typed it) be
  deleted entirely; the seq never needs to change after creation.

Beyond the defects, the dashboard is thin in ways that matter:

- ~~**Invalid HTML**: `<a href="..."><button></button></a>` in `index.html`
  and `experiment.html`. Interactive elements cannot be nested; this breaks
  keyboard and assistive-technology behaviour. Use a styled link or a form
  button.~~ **Fixed** — both replaced with a styled `<a class="btn">` (see
  `base.html`'s `a.btn` rules), so there's exactly one interactive element
  where there was a `<button>` nested inside an `<a>`.
- ~~**It can't see the repo's own experiments.** `store.list_experiments()` reads
  only `~/.yardstick/experiments`, so `experiments/example.yaml` — the file the
  README tells you to use — never appears. The docs and the UI disagree about
  where experiments live.~~ **Fixed** — `store.discovery_dirs()` now also
  searches an `experiments/` directory next to the process's current working
  directory (read-only), indexed by the parsed `experiment:` field rather than
  filename (a discovered file's name need not match, e.g. `example.yaml`'s
  `experiment:` is `mock-smoke-01`). `EXPERIMENTS_DIR` stays the only directory
  the dashboard writes to and always wins a name collision;
  `store.experiment_path` (and hence `save_experiment`) is unchanged, so every
  route already keyed off it keeps working — `store.find_experiment` is the new
  read-side resolver every name-keyed route goes through instead, and it still
  runs `validate_name` first, so a URL-supplied name can't escape into an
  arbitrary path just because more directories are searched. An experiment
  discovered outside `EXPERIMENTS_DIR` is shown as **read-only** (view/run/
  compare all work; edit/delete are refused with an explanation) so the
  dashboard can never rewrite or remove a file that's part of a git checkout.
- ~~**No edit, no YAML view, no delete.** An experiment is write-once through the
  form; any change means hand-editing a file the UI won't show you.~~ **Fixed**
  — `/experiments/{name}/yaml` renders `store.read_raw` (previously unused);
  `/experiments/{name}/edit` reuses the same form defects 20/22/24 reworked,
  pre-filled from the current definition; `/experiments/{name}/delete` removes
  the YAML after a confirmation that names the recorded run count (defect 21's
  house style) and explains that those runs are *not* deleted, just unreachable
  from the dashboard until a same-named experiment exists again. All three are
  refused for a discovered, non-`EXPERIMENTS_DIR` experiment (see above).
- ~~**Nothing updates during a live run.** No auto-refresh, no running token/cost
  counter, no request feed. During the one phase where the user is watching, the
  dashboard is a static page.~~ **Fixed** — a small `/runs/{run_id}/live` JSON
  endpoint (request count, turns, cost, billable tokens, per-request rows) is
  polled every few seconds: from the active-run banner on every page (running
  request/token/cost counters), and from `run_detail.html` itself while that
  run hasn't ended (the request table refreshes in place; the page reloads
  once on the ended transition to pick up the fields `/live` doesn't carry).
  Dependency-free — small inline JS, no build step, no client library, matching
  the rest of the dashboard.
- ~~**The comparison view escapes the app.** `/experiments/{name}/compare` returns
  the standalone report document with no shell and no way back.~~ **Fixed** —
  embedded inside the app shell (`compare.html`, with a back-to-experiment link)
  instead of forking the renderer. `render.render_html` gained a minimal
  `standalone=False` seam (splits the existing body markup from its wrapping
  `<title>`/`<style>`, default unchanged for `ys report --html`) so the
  dashboard can embed just the fragment rather than nesting one HTML document's
  `<title>`/`<style>` inside another's `<body>`.
- ~~**Run detail omits the useful parts**: no `success_output` (the check's own
  output, the first thing you want when a run fails), no notes, no factors, no
  per-turn chart.~~ **Fixed** — all four now render on `/runs/{run_id}`:
  `success_output` and `notes` straight from the `runs` row, `factors` from the
  arm's `factors_json`, and a per-turn context-tokens chart as plain inline SVG
  (hand-rolled, the same weight as `render.py`'s own `_sparkline_svg` — no
  charting library) once a run has at least two main-thread requests recorded.
- ~~**Only the `model` factor is expressible** in the new-experiment form, though
  `Arm.factors` is an arbitrary dict — so the harness-vs-harness comparison the
  tool is named for can't be set up from the UI at all.~~ **Fixed** — each arm
  row in the form can now carry arbitrary extra factor key/value pairs (e.g.
  `harness=claude-code` vs `harness=opencode`) alongside the dedicated `model`
  field, joined back to their arm by the same stable `arm_seq` the baseline
  radio group (defect 24) already uses. The extra-factor rows round-trip
  through a validation failure the same way the rest of the form does.
- ~~The mock model id hardcoded in the form (`claude-3-5-sonnet-20241022`) is
  stale.~~ **Fixed** — updated to a current model id (it never reaches
  Anthropic either way, since `mock_response` short-circuits it, but the form
  shouldn't teach a dead id by example).

And on the CLI side: there was no `ys runs list` — runs could be deleted by id
but never enumerated, so the only way to find an id was the dashboard or raw
SQL — **fixed**: `ys runs list` (`ys/runs.py`'s `list_runs`) prints run id,
experiment, arm, repeat, start time, status (`finished` / `unfinished` /
`abandoned` — finding 13's three-way split, named per row instead of only
summarized in `ys compare`), success, and — with `--exp` — whether each run's
stored `config_hash` still matches today's YAML (finding 14's "which of these
runs still count" question, the one that sends people to raw SQL in the first
place; a run predating finding 14, or whose arm the current YAML no longer
declares, is reported as not current rather than silently trusted).
`--exp`/`--arm` filter, `--limit` caps the row count, most recent first.

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

### 3. Statistics worth the name -- fixed

`aggregate_run_metrics` reports a mean and a population standard deviation over
n=3. `render` shows `± spread` and a percentage delta. Nothing indicates whether a
difference between arms is real, which is the only question the user actually has.

Add bootstrap confidence intervals on each metric, a permutation or Mann-Whitney
test on the primary metric, a Wilson interval on success rate, and an explicit
verdict line: *"arm-b is 18% cheaper, but with n=3 the difference is not
distinguishable from noise; ~12 repeats needed."* Also add a minimum-detectable-
effect helper so users can choose `repeats` deliberately.

**Fix:**

- New `ys/statistics.py`, kept deliberately pure -- plain lists of floats in,
  dataclasses out, no sqlite/`Experiment`/rich/HTML in sight, so it's testable
  against hand-computed known-answer cases without a database or a
  `Comparison` in the loop. `ys/render.py` is the only consumer and owns all
  the wording (arm labels, "cheaper" vs "higher", banner styling).
- **No new dependency.** `scipy` is not declared in `pyproject.toml` and
  still isn't: `bootstrap_ci` is a resampling loop over `random.Random`,
  `wilson_interval` is closed-form `math`, and the significance test is an
  **exact two-sided permutation test** (`itertools.combinations` enumerates
  every relabeling when `C(n_a+n_b, n_a)` is small enough -- always true at
  this rig's default `repeats: 3`, where it's a mere 20), falling back to a
  seeded Monte Carlo sample only past a size no default-`repeats` run would
  reach. Chosen over Mann-Whitney specifically because exact enumeration at
  tiny n needs no distributional assumption and, unlike a U-statistic's
  asymptotic p-value, is exact by construction rather than approximate.
  `_norm_ppf` (Peter Acklam's rational approximation to the normal quantile
  function, public domain) is the one piece of "real" numerics, needed only
  by the minimum-detectable-effect helper's z-score lookup.
- **Deterministic by construction.** Every source of randomness (the
  bootstrap resampler, the permutation test's Monte Carlo fallback) is
  driven by a `random.Random(seed)` instance built fresh inside the call
  from a fixed module-level `DEFAULT_SEED`, never the global `random`
  module -- so `ys compare` on unchanged data prints the exact same verdict
  every time, not just a statistically-similar one. `tests/test_statistics.py`
  pins this directly (`test_permutation_test_is_deterministic_across_repeated_calls`,
  `test_bootstrap_ci_is_deterministic_across_repeated_calls`,
  `test_metric_verdict_is_deterministic_across_repeated_calls`), and
  `tests/test_render.py::test_significance_verdicts_is_deterministic_across_repeated_calls`
  covers the same property through `compare_experiment`/`significance_verdicts`.
- **The n=3 problem is answered head-on, not hidden.** `min_two_sided_p(3, 3)
  == 0.1`: with the schema's own default `repeats: 3`, no possible dataset
  can produce a two-sided exact permutation p-value below 0.05 (`C(6,3) = 20`
  relabelings; only 2 -- the true split and its mirror -- can ever be the
  most extreme). `min_n_for_exact_significance()` computes the first n where
  that stops being true (4, per side). The verdict line leads with this fact
  in plain language ("the smallest possible p-value an exact test could
  report at this n is 0.1, so no result here could reach significance")
  rather than printing `p=0.1` and leaving the user to know what that means
  at n=3 -- exactly the "don't imply more precision than exists" instruction
  this feature was built under. `tests/test_statistics.py::test_metric_verdict_at_n3_never_claims_significance`
  and `tests/test_render.py::test_significance_verdicts_names_direction_effect_size_and_repeats_needed`
  pin the n=3 case end to end.
- **`metrics.primary` drives the significance test**, per the plan's own
  instruction and finding 15-18's precedent -- no separate mechanism.
  `render.significance_verdicts(comparison)` runs `stats.metric_verdict`
  once per (non-baseline arm, `comparison.primary_metrics` entry) using the
  *exact same gate-passing, config-hash-matched run population*
  `aggregate_run_metrics` already computes `mean`/`n`/`spread` from (finding
  13/14) -- `aggregate_run_metrics`'s per-metric dict now also carries the
  raw `values` list the mean/spread were computed from, so the significance
  test can never silently operate over a different n than the table beside
  it claims.
- **The verdict line**, matching the plan's own wording: e.g. *"arm-b is 18%
  cheaper than arm-a on cost_usd, but with n=3 the smallest possible p-value
  an exact test could report at n=3 is 0.1, so no result here could reach
  significance; ~4 repeats needed to tell a difference this size from
  noise."* Always states direction (a small per-metric word table gives
  `cost_usd`/`cost_per_success` "cheaper"/"more expensive", `wall_clock_s`/
  `active_s` "faster"/"slower", count-like metrics "fewer"/"more", and
  anything else falls back to generic "lower"/"higher"), effect size,
  whether it's distinguishable from noise, and -- when it isn't -- a
  required-repeats estimate from `required_repeats_per_arm` (a standard
  two-sample z-test sample-size formula, floored at
  `min_n_for_exact_significance` so a very large effect with very low noise
  can never suggest fewer repeats than the permutation test could ever act
  on). When the test *is* significant, the line says so plainly instead
  (`"... (n=8, permutation p=0.000155) -- distinguishable from noise."`).
- **Surfaced the same way the three existing warning families are** (`cost_
  warnings`/`config_warnings`/`repeat_count_warnings`), not a fourth
  invented style: `render.significance_verdicts(comparison) -> list[str]`,
  printed by `ys compare` under an "is the difference real?" heading and
  rendered as a bordered banner (blue, distinct from the red/amber warning
  banners since this is the answer, not a caveat) right above the table in
  the HTML report.
- **Bootstrap CIs on every displayed metric** (`stats.bootstrap_ci`, a
  percentile bootstrap over the same raw `values`), shown in the HTML report
  next to each metric's existing `± spread` cell (`CI95[low, high]`) -- the
  CLI table stays terse, matching the existing precedent that spread itself
  is HTML-only. Omitted (not a degenerate `CI95[x, x]`) when a metric has
  fewer than 2 observations to resample.
- **Wilson interval on success rate**: `stats.wilson_interval`, appended to
  the existing `n_success/n_runs` cell in both `ys compare` and the HTML
  report (`3/3 (95% CI 44-100%)`) -- chosen over the normal approximation
  specifically because the latter collapses to a zero-width interval at 0%
  or 100% observed, which is the unremarkable common case at n=3.
- Tests: `tests/test_statistics.py` (new) has 29 known-answer cases --
  hand-computed Wilson intervals, a permutation test on datasets small
  enough to enumerate by hand (n=2 vs n=2, n=3 vs n=3), a bootstrap CI
  pinned against values traced directly from the seeded RNG, and the
  determinism/n=3-never-significant properties above.
  `tests/test_metrics.py::test_aggregate_run_metrics_carries_raw_values_for_bootstrap_and_permutation_tests`
  pins that the new `values` list matches exactly the gate-passing
  population `mean`/`n` already describe. `tests/test_render.py` and
  `tests/test_cli.py::test_compare_prints_significance_verdict_for_primary_metric`
  cover the verdict banner end to end through `compare_experiment`/
  `ys compare`/`ys report --html`, including the no-baseline and
  no-primary-metrics empty cases.

### 4. `ys doctor` — fixed

There are many moving parts — home directory, schema version, proxy process,
generated config, harness config, two API keys, active-run state — and the failure
modes are mostly silent. A single preflight command that checks all of them,
verifies the running proxy serves the current experiment's models, and reports
unattributed request counts would prevent most of the wasted runs this review
found paths to.

**Fix:**

- `ys/doctor.py` (new module, kept out of `ys/cli.py` so the check logic is
  directly testable) runs a fixed list of checks, each returning pass / warn
  / fail plus one specific, actionable message — the value of a preflight
  command is entirely in the wording, not the verdict:
  - **yardstick home** — the directory exists and is writable.
  - **schema version** — `PRAGMA user_version` against `db.MIGRATIONS`
    (finding 8), via a plain read connection so the check itself never
    creates the database or applies a migration.
  - **proxy process** — pidfile/port sanity via `procutil`/`proxy.proxy_status`
    (finding 5): running; a stale pidfile with the port now free (run `ys
    proxy up`); or, finding 5's exact silent-orphan scenario, a stale pidfile
    with the port *still* bound by something else (fail — run `ys proxy down
    --force`).
  - **generated proxy config** — whether `ys proxy up`'s last generated
    `model_list` parses, and, with `--exp`/`--arm`, whether it has an
    explicit entry for that arm's model. The static, read-from-disk sibling
    of the live check below — still useful with the proxy not running.
  - **task paths** — with `--exp`, reuses `experiment.validate_task_paths`
    (findings 15-18) verbatim, so a `task.prompt_file`/`task.repo` typo shows
    up in `ys doctor` before you even get to `ys start`.
  - **harness config**, one row per agent — reuses `harness.status` to report
    whether each agent's config is currently pointed at a proxy, and, if so,
    whether a proxy is actually running there (fail if not: real requests
    would fail until `ys proxy up` again or `ys harness reset`).
  - **API keys** — `LITELLM_MASTER_KEY` and `ANTHROPIC_API_KEY` presence in
    this shell's environment (the two the plan calls out by name).
  - **active-run state** — cross-checks `active.json` against the `runs`
    table, catching the exact mismatch `ys end` raises
    `ActiveRunMissingDbRow` for, before `ys end` hits it.
  - **unattributed requests** — reuses `runs.unattributed_summary()`
    (finding 12) verbatim — "the single highest-value diagnostic in the
    app" per this plan.
  - **dropped requests** — reuses `dropped.count()` (finding 6) verbatim.
  - **proxy serves model** — with `--exp`/`--arm` both given, reuses
    `proxy.model_available`/`proxy.model_check_skipped_message` (findings
    3/29) verbatim against the *running* proxy; without them, a `skip` row
    says so instead of silently omitting the check.
- Strictly read-only, unlike every neighbouring command: no `db.init_db()`/
  migration, no `paths.ensure_home()`, no starting/stopping the proxy or
  dashboard, no writing to a harness config. Where a reused helper would
  normally have a side effect (`db.connect()`'s `paths.ensure_home()`,
  `db.cursor()` on a database that doesn't exist yet), `ys doctor` works
  around it — a plain `sqlite3.connect` guarded by an existence check —
  rather than accepting the side effect as a shortcut. A user running `ys
  doctor` to find out why something is broken must never have the act of
  asking change the answer.
- `ys doctor` exits non-zero if any check fails (not on warnings), so it's
  usable as a script gate.
- `ys runs list` (below) reuses the same instinct — surface a diagnostic
  instead of sending the user to raw SQL — for enumerating runs specifically.

### 5. Provider and harness breadth — fixed

**Models.** Everything assumed Anthropic: the fallback was `anthropic/<value>`, the
harness wrote `ANTHROPIC_*` variables, the web form defaulted to a Claude id, and
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

**Fix:**

- **Models are provider-agnostic.** `ys/proxy.py`'s `_fallback_params` (the
  convention used when a `factors.model` value has no explicit `models:` entry)
  used to always prepend `anthropic/`, which double-prefixed an
  already-provider-prefixed value (`openai/gpt-4o` → `anthropic/openai/gpt-4o`,
  nonsense). It now uses an already-prefixed value as-is and only falls back to
  the `anthropic/<value>` convention for an unprefixed one, unchanged for every
  experiment written before this. The provider is read straight off that
  `<provider>/<id>` prefix — LiteLLM's own routing convention, which the
  experiment author already has to get right for LiteLLM to reach the intended
  backend at all — rather than sniffed from the model name's text, or declared
  in a second, independently-maintained `provider:` field that could drift out
  of sync with it. A small `_SIMPLE_API_KEY_ENV_VAR` map (anthropic/openai/gemini)
  supplies the right `api_key` env var for providers with a plain bearer-token
  credential; providers with fundamentally different auth (Bedrock's AWS SigV4,
  Vertex's GCP service account) deliberately get no fabricated `api_key` field —
  LiteLLM already reads their own standard env vars on its own. The catch-all
  entry (finding 3) is no longer hardcoded to `anthropic/*` either: a new
  `_catch_all_params` infers a single provider from the experiment's declared
  models when they all agree, and only falls back to the original
  `anthropic/*` default when they don't (a genuine cross-provider comparison) —
  otherwise a Codex CLI/Aider run's own OpenAI-shaped background traffic would
  route through an Anthropic catch-all and hit the wrong API entirely.
  `experiments/cross-provider-example.yaml` is a new, mocked (free to run)
  experiment demonstrating an Anthropic arm and an OpenAI arm side by side.
- **Where "provider" is determined from, and why per-tool, not per-model.**
  Every coding tool this rig drives speaks exactly one wire protocol to
  whatever endpoint it's pointed at: Claude Code and opencode speak
  Anthropic's Messages format; Codex CLI and Aider speak OpenAI's Chat
  Completions format. That's independent of which real backend the arm's
  model resolves to — LiteLLM's proxy is what bridges any backend to
  whichever of those endpoints the client calls. So "map harness
  environment variables per provider" turned out to mean *per coding tool*
  (which of those wire protocols it natively expects), not per the
  experiment's declared backend model — the model's own provider prefix
  already routes correctly through `ys/proxy.py` regardless of which agent
  is asking. `ys/harness.py`'s module docstring states this explicitly, since
  it's the one design decision in this feature most likely to look wrong at a
  glance.
- **Coding tools added:** Codex CLI and Aider, on top of the existing Claude
  Code and opencode. Both are **best-effort / unverified against a live
  install** — implemented from public documentation, not from running the
  tool, and flagged as such in `ys/harness.py`'s own docstrings:
  - **Codex CLI** writes `~/.codex/config.toml` (`model_provider`/`model` plus
    a `[model_providers.yardstick]` table with `base_url`/`env_key`/
    `wire_api`). This repo has no TOML parser/writer dependency (adding one
    is out of scope: `pyproject.toml` isn't in this change's file set), so
    unlike the JSON agents, it **only ever creates a fresh file** — any
    existing config.toml with real content is refused outright rather than
    risk silently mangling it via a hand-rolled parser. `--env-only` isn't
    supported for it: Codex's `env_key` only names which env var holds the
    API key, the base URL itself is config.toml-only.
  - **Aider** embeds LiteLLM directly and takes its provider config purely as
    environment variables (`OPENAI_API_KEY`, `AIDER_MODEL=openai/<value>`),
    so it's naturally `--env-only`-*only* (`AgentSpec.env_only = True` — there
    is no config file for `point`/`reset` to manage at all). Its base-URL
    variable name has two real candidates in the wild (`OPENAI_API_BASE`,
    Aider's own docs; `OPENAI_BASE_URL`, current openai-python) and we could
    not verify which its installed version reads without running it, so both
    are exported — an unused extra env var, not a wrong value that routes
    anywhere, and not a fabricated name (both are real, documented names).
  - **Deliberately left out**, each for a stated reason rather than a guess
    (`ys/harness.py`'s `EXCLUDED_TOOLS` comment): **Gemini CLI** (no
    environment-variable-only mechanism for a custom endpoint could be
    confirmed), **Cursor CLI** (its agent traffic is proxied through Cursor's
    own backend by design, no confirmed custom-endpoint override at all),
    **GitHub Copilot CLI** (structurally out of scope, not just unverified —
    bound to GitHub's own backend), **Cline/Roo** (config lives in a VS Code
    extension's own settings storage, not a stable on-disk path+schema).
  - **Project-level Claude Code settings**: `ys harness point claude-code
    --scope project` (and `harness reset`/`harness status --scope project`)
    write `./.claude/settings.json` instead of `~/.claude/settings.json`.
    Only claude-code's project path is implemented — `AgentSpec.resolve_path`
    raises for any other agent rather than guess at one. Each scope gets its
    own backup manifest (`{agent}.json` for user, `{agent}-project.json` for
    project), so pointing both scopes for the same agent can't clobber each
    other's backup.
- **Harness config safety — the priority half of this feature:**
  - `ys harness point <agent> --env-only` (`harness.env_exports`) prints
    `export` statements for exactly the env vars `point()` would otherwise
    write into a config file, and never reads or writes that file at all —
    verified directly:
    `test_env_only_leaves_the_config_file_completely_untouched` and
    `test_env_only_does_not_touch_a_preexisting_config_either`
    (`tests/test_harness.py`) plus the CLI-level
    `test_harness_point_env_only_never_writes_the_config_file`
    (`tests/test_cli.py`). Supported for claude-code (its `ANTHROPIC_*`
    variables are exactly what Claude Code itself documents reading) and
    aider (its only mechanism); raises `HarnessError` for opencode and
    codex-cli, whose base URL is only confirmed to work via their config
    file, rather than guess at an env var for them too.
  - `ys end` now calls a new `cli._auto_reset_pointed_harnesses()` by
    default: it checks every agent (skipping env-only ones, which never
    wrote a file) across every scope it could plausibly have been pointed at
    (`harness.scopes_for_agent`), and resets whichever ones
    `harness.status(...).pointed_at_proxy` reports as still pointed at the
    proxy. This closes the exact gap the finding named — nothing used to
    reset `point()`'s plaintext key automatically at all, so a crash between
    `ys start` and a manual `ys harness reset`, or simply forgetting that
    step, left it there indefinitely. `ys end --keep-harness-pointed` opts
    out for a multi-repeat workflow that wants to stay pointed across
    several `ys start`/`ys end` cycles without re-pointing before each one —
    the same plaintext-exposure trade-off the flag otherwise closes, made
    explicit rather than silently reintroduced.
  - The `.jsonc` comment-stripping loss on opencode configs is unchanged and
    still real (`--env-only` isn't supported for opencode, so it doesn't
    sidestep it there) — documented in both `ys/harness.py`'s module
    docstring and the README's "Harness config safety" section.
- **Verification.** New regression tests in `tests/test_harness.py`,
  `tests/test_proxy.py` and `tests/test_cli.py` (21, 4, and 6 new tests
  respectively — 316 passed overall, up from this branch's 286-test baseline
  after rebasing onto the findings-13/14 and 15-18 PRs that merged to `main`
  in the meantime) cover: the fallback double-prefix fix and catch-all provider
  inference; `--env-only` for every agent that supports it and the explicit
  refusal for the two that don't; codex-cli's create-fresh-only behavior
  (including refusing to touch a populated file, and treating a
  whitespace-only file as absent); project-vs-user scope isolation
  (independent backups, `--scope project` unsupported for opencode); aider
  being env-only end to end; and, at the CLI layer, `--env-only` never
  touching the fake claude-code settings file and `ys end` actually
  resetting (and, with `--keep-harness-pointed`, actually *not* resetting) a
  pointed harness. Five of these — the auto-reset on `ys end`, `--env-only`
  never writing the file, the fallback double-prefix fix, the catch-all
  provider inference, and codex-cli's refuse-on-existing-content guard — were
  each explicitly reverted and confirmed to fail before restoring the fix, as
  the most safety-critical and/or novel mechanisms in this change; the
  remainder follow the same patterns and pass as part of the same 316-test
  suite. `tests/conftest.py` gained a second autouse fixture,
  `isolated_harness_agents`, isolating claude-code/opencode/codex-cli to fake
  per-test paths for *every* test in the suite (not just this file's) — `ys
  end`'s new auto-reset walks every entry in `harness.AGENTS` on every `ys
  end` call, so without this, any existing test anywhere that invokes `ys
  end` would have silently read (and, if it ever looked pointed, overwritten)
  the real files on whichever machine runs the suite.
- **Left for later:** the web dashboard's new-experiment form still only
  offers a bare model id field (`ys/web/**` was off-limits for this change);
  Gemini CLI/Cursor CLI/Copilot CLI/Cline/Roo remain unimplemented for the
  stated reasons above, pending someone who can verify the real mechanism
  against a live install.

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
  README.~~ Moved to `tools/provenance/` with `git mv` (not `docs/`, since
  `docs/` is published verbatim to GitHub Pages); README now says so, and
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
