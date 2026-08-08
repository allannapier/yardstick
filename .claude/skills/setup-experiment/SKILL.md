---
name: setup-experiment
description: Set up and run a yardstick experiment comparing models, harnesses, or providers. Use when asked to create/configure an experiment YAML, add an arm, pick models for a comparison, wire up a success_check, or drive `ys run` end to end. Encodes the failure modes that produced silently wrong or empty results in practice.
---

# Setting up a yardstick experiment

The rig measures cost/tokens/turns per arm. Its worst failure mode is not
crashing — it is **reporting confident numbers that are wrong**, or scoring a
run that never happened. Most of this file exists to catch that before it
costs money.

Golden rule: **prove each layer with one cheap request before launching a
paid multi-repeat run.** Every expensive lesson below came from skipping that.

## 1. Resolve models against the live API, never from memory

Model ids drift and marketing names don't map cleanly ("Gemini 3.6" is
`gemini-3.6-flash`; there is no 3.6 Pro).

```bash
curl -s "https://generativelanguage.googleapis.com/v1beta/models?key=$GEMINI_API_KEY&pageSize=200" \
  | grep -o '"name": "models/[^"]*"'
curl -s https://openrouter.ai/api/v1/models | python -c "import sys,json;[print(m['id'],m['pricing']) for m in json.load(sys.stdin)['data']]"
curl -s https://api.anthropic.com/v1/models -H "x-api-key: $ANTHROPIC_API_KEY" -H "anthropic-version: 2023-06-01"
```

OpenRouter ids can carry a leading `~` (`~deepseek/deepseek-v4-flash-latest`)
— that is part of the id, not a typo.

## 2. Verify keys work, don't just check they exist

`ys doctor` checks **presence only**, and only for `ANTHROPIC_API_KEY` — it
knows nothing about `GEMINI_API_KEY` or `OPENROUTER_API_KEY`. It will report
PASS on an experiment that cannot run. Actually spend one token:

```bash
curl -s https://api.anthropic.com/v1/messages -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" -H "content-type: application/json" \
  -d '{"model":"claude-sonnet-5","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}'
```

**Watch for free-tier quotas.** A free Gemini key allows ~20 requests/day for
`gemini-3.6-flash` — one agent repeat exhausts it. The 429 then arrives
*mid-stream*, after `content_block_start`, so the agent receives a truncated
stream plus an untyped `{"error":...}` object and dies with a schema error
naming neither quota nor HTTP status (opencode reports a zod
`No matching discriminator` on `path: ["type"]`). If an arm fails with an
opaque parse error, `curl` the proxy directly with `"stream":true` before
believing the error message. Routing that model through OpenRouter instead
sidesteps the cap at identical list price — and putting every arm behind one
gateway removes provider-side confounds from the comparison.

## 3. Pricing: declare it, and key it the way lookup actually works

LiteLLM silently returns `0.0` for any model its cost map lacks. A confident
`$0.0000` is the worst output a cost-comparison tool can produce, so
`ys/collector.py` falls back to the experiment's `pricing:` block whenever
LiteLLM's number is zero *and* tokens were spent (`cost_source` becomes
`declared`; `unknown` means nobody could price it — investigate).

`resolve_model_key` (`ys/experiment.py`) tries the exact recorded model string,
then **only the part after the first `/`**. Two-segment provider ids therefore
never match a bare factor-value key:

```
openrouter/google/gemini-3.6-flash  ->  google/gemini-3.6-flash   (not "gemini-3.6-flash")
```

Declare **both** keys — the factor value (the documented convention) and the
two-segment form (what actually matches):

```yaml
pricing:
  gemini-3.6-flash:        {input_per_mtok: 1.50, output_per_mtok: 7.50, cache_read_per_mtok: 0.15}
  google/gemini-3.6-flash: {input_per_mtok: 1.50, output_per_mtok: 7.50, cache_read_per_mtok: 0.15}
```

Verify before running:

```bash
python -c "
from ys.experiment import load_experiment
from ys.collector import _declared_cost
e=load_experiment('experiments/X.yaml')
t={k:v.__dict__ for k,v in e.pricing.items()}
print(_declared_cost('openrouter/google/gemini-3.6-flash',1_000_000,0,0,100_000,t))"
```

`openrouter` is not in `ys/proxy.py`'s `_SIMPLE_API_KEY_ENV_VAR`, so an
undeclared `openrouter/...` factor value registers with **no api_key**. Always
give OpenRouter models an explicit `models:` entry.

## 4. Design a gate that fails on an empty workspace

`success_check` must distinguish "did the task" from "did nothing". A bare
`pytest -q` on a fresh clone passes trivially, so every arm scores 100%
regardless of behaviour.

- Put the check **outside** the workspace (e.g. `experiments/X.check.sh`,
  invoked by absolute path) so the agent can't read the gate and target it.
- Assert one clause per prompt requirement, print reasons to stderr.
- **Test it before running:** empty dir → FAIL, correct output → PASS, plus
  the near-miss cases the prompt forbids.

## 5. Workspaces

- Greenfield task → clone a **blank scaffold repo**, not the project you're
  grading in:
  ```bash
  S=~/.yardstick/scaffolds/blank-web
  mkdir -p $S && cd $S && git init -q -b main && touch .gitkeep && git add -A && git commit -qm scaffold
  ```
- `task.repo` + `ref` is the only mode that re-isolates between repeats.
  `task.workdir` alone does not, and no `repo` falls back to the invoking
  directory.
- **Workspaces are deleted after scoring.** If the artifact matters, copy it
  out in `task.teardown` using `$YS_WORKDIR`:
  ```yaml
  teardown: 'd="$HOME/out/$(date +%Y%m%d-%H%M%S)-$$"; mkdir -p "$d"; cp -R "$YS_WORKDIR"/. "$d"/'
  ```

## 6. opencode specifics

opencode is the most failure-prone supported agent. All of these were live
defects; the fixes are in the tree, but check the behaviour still holds.

- **It reads `PWD`, not the subprocess `cwd`.** `subprocess.run(cwd=…)` leaves
  `PWD` pointing at wherever `ys run` was launched — so repeats ran against
  the user's real checkout while workspace clones sat empty, and the agent
  wrote files into a live repo. `ys/runner.py` now sets `env["PWD"]` and
  passes `--dir`. Confirm isolation held:
  ```bash
  grep "message=created" ~/.local/share/opencode/log/opencode.log | tail -1
  ```
  The `directory=` must be the workspace, never the project.
- **The arm's model must be declared**, not just selected. opencode resolves
  ids against models.dev; an arm's `model_name` is by definition absent, and
  it aborts with `UnknownError: Unexpected server error` before issuing any
  request. Needs `provider.anthropic.models.<model>` as well as `model`.
- **Pin `small_model`.** Otherwise title-generation goes out as a Claude id
  the experiment never declared, which the proxy's `*` catch-all routes
  straight to the real Anthropic API — off-arm spend, invisible to metrics.
- **`--env-only` is unsupported**, so its real config file gets rewritten.
  Back it up yourself; `ys harness reset` is not sufficient (next point).
- **MCP servers inflate every measurement.** A single MCP server added ~9k
  input tokens per request. Constant across arms, so comparisons still hold,
  but absolute cost and `fixed_overhead_tokens` won't represent a bare agent.
  Strip them for clean numbers, and restore afterwards.

## 7. `ys harness reset` restores a stale snapshot

`_snapshot_if_absent` never refreshes: the **first snapshot ever taken** is
what every later reset restores. A config edit made after that is silently
reverted — mid-run, between arms. To run with a modified config, patch the
`raw` field of `~/.yardstick/harness_backups/<agent>.json` too, and restore
both from your own backup at the end.

## 8. Confirm requests are actually recorded

A `200 OK` in the proxy log does **not** mean a row was written — a raising
callback is swallowed per-request, so successes vanish while failures record.
After one invocation, reconcile both sides:

```bash
grep -c "POST /v1/messages" ~/.yardstick/proxy.log
python -c "
import sqlite3,os
c=sqlite3.connect(os.path.expanduser('~/.yardstick/yardstick.db'))
for r in c.execute('select run_id,model,input_tokens,output_tokens,cost_source from requests order by rowid desc limit 5'): print(r)"
```

Red flags: rows only for failed requests; `run_id='unattributed'`; a model id
you didn't declare (traffic escaping the arm); `cost_source='unknown'`.

## 9. Before `ys run`

- Clear any stale active run (`ys status`) — it steals attribution *and*
  prices new requests against the wrong experiment's `pricing:` block.
- `ys proxy up` re-reads the YAML: **restart it after editing models**.
- The proxy resolves `os.environ/*` in **its own** process — export keys
  before starting it, not after.
- `ys doctor --exp … --arm …`, remembering §2's limits.

## 10. Interpreting results honestly

- Delete runs from broken setups before comparing; they skew the aggregate.
- `config_hash` is **per-arm** (task + that arm's own factors, excluding
  `models:`/`pricing:`). Arms differing from each other is expected, not drift.
- n=3 cannot reach significance — the smallest possible exact p-value is 0.1.
  `ys compare` states the repeats needed; quote that alongside any delta.
- Metrics don't capture output quality. Inspect the preserved artifacts before
  concluding "cheaper is as good".
