# yardstick

Agent / harness / model efficiency measurement rig. Yardstick sits between a
coding agent (Claude Code, opencode, Codex CLI, Aider, ...) and the LLM API as
a LiteLLM proxy, recording every request/response so you can compare arms of
an experiment (e.g. same task, different model, provider, or harness config)
on cost, tokens, turns, tool-call efficiency, and task success.

## How it works

1. `ys proxy up` starts a LiteLLM proxy configured from your experiment
   YAML(s). Models can be served by any LiteLLM-supported provider (Anthropic,
   OpenAI, Gemini, Bedrock, Vertex, a local model, ...) -- see "Models and
   providers" below.
2. `ys harness point <agent> --exp ... --arm ...` repoints an agent's config
   (e.g. Claude Code's `~/.claude/settings.json`) at that proxy and pins it to
   the arm's model, backing up the original config first. Pass `--env-only`
   to print `export` statements instead and never touch any file at all --
   see "Harness config safety" below.
3. `ys start --exp ... --arm ...` marks a run active; you then drive the
   agent through its task as normal.
4. `ys end` finishes the run, runs the arm's `success_check`, prints headline
   metrics, and resets any harness config `ys harness point` touched (see
   below).
5. `ys compare` / `ys report` aggregate recorded runs per arm into a table or
   a self-contained HTML report.

Every proxied request is logged to a local SQLite database
(`~/.yardstick/yardstick.db`), from which per-run metrics (tokens, cost,
cache-read ratio, tool-call counts/errors, redundant calls, compaction
events, context high-water mark, etc.) are derived.

## Install

```bash
pip install -e .
```

Requires Python >= 3.10. Installs the `ys` CLI (`ys.cli:app`, via Typer) with
`litellm[proxy]`, `fastapi`/`uvicorn` (dashboard), `pydantic`, and friends.

## Quick start

```bash
ys init                                    # create ~/.yardstick and the db

export LITELLM_MASTER_KEY=sk-...           # proxy auth key you choose
ys proxy up --exp experiments/example.yaml # start the measurement proxy

# point Claude Code at the proxy, pinned to arm-a's model
ys harness point claude-code --exp experiments/example.yaml --arm arm-a
ys start --exp experiments/example.yaml --arm arm-a

# ... drive the agent through the task ...

ys end                                     # score + print headline metrics,
                                            # then reset claude-code's config
ys proxy down
```

`ys end` restores the agent config `ys harness point` backed up automatically
(pass `--keep-harness-pointed` to stay pointed across several repeats without
re-running `ys harness point` before each one). To avoid touching any config
file at all -- e.g. scripting the agent in the same shell, or pointing an
agent with no config file (aider) -- pass `--env-only` instead and export the
printed variables yourself:

```bash
eval "$(ys harness point claude-code --exp experiments/example.yaml --arm arm-a --env-only | grep ^export)"
```

Compare recorded runs for an experiment:

```bash
ys compare --exp experiments/example.yaml
ys report  --exp experiments/example.yaml --html report.html
```

Enumerate recorded runs (id, experiment, arm, status, success — the
config column flags a run whose config_hash doesn't match today's YAML):

```bash
ys runs list --exp experiments/example.yaml
```

Preflight everything at once — home directory, schema version, proxy
process, generated config, harness config, both API keys, active-run
state, and unattributed/dropped request counts, plus (with `--exp`/`--arm`)
whether the running proxy actually serves that arm's model. Read-only;
exits non-zero if anything fails:

```bash
ys doctor --exp experiments/example.yaml --arm arm-a
```

Optional dashboard for setting up experiments and browsing runs in a browser:

```bash
ys web up
```

## Unattended runs

Steps 2-4 above are hand-driven: point the harness, start, drive the agent
yourself, end -- so a human is an uncontrolled variable across repeats and
arms, and `repeats: 3` means doing it three times by hand. `ys run` instead
invokes the agent non-interactively and loops the repeats itself:

```bash
export LITELLM_MASTER_KEY=sk-...
ys proxy up --exp experiments/unattended-example.yaml
ys run --exp experiments/unattended-example.yaml --arm arm-a --repeats 3
```

It requires `task.prompt_file` (there's no human to type a prompt) and an
`--agent` (or the arm's own `factors.agent`) whose CLI is on `PATH` --
`claude -p`, `opencode run`, and `codex exec` are invoked as named in
IMPROVEMENTS.md's feature 1; `aider --message ... --yes-always
--no-auto-commits` is a best-effort fourth, reconstructed from its docs and
not verified against a live install (see `ys/runner.py`'s
`build_agent_command` -- the one place all four are constructed, so a wrong
invocation form is a one-line fix). Every one of these checks, plus the
proxy being up and `LITELLM_MASTER_KEY` being set, is verified *before* the
first repeat starts, not discovered mid-loop.

Two protections against an unattended loop burning real money: each agent
invocation is bounded by `task.timeout_s`, and the whole run hard-stops
after `--max-consecutive-failures` (default 3) agent invocations fail in a
row -- a repeat that ran fine but simply failed its own `success_check`
doesn't count, only the sign something upstream is broken (the proxy went
down, the agent can't authenticate, ...) does. `--settle-s` (default 2s)
pauses briefly between repeats: `ys end` starts a short window (see
"Attribution" below) during which a header-less straggling response is
still credited to the run that just ended, and a fast automated loop
starting the next repeat immediately would flip that attribution target
before the window closes.

Pointing prefers `--env-only` (see "Harness config safety" below): the
agent subprocess gets its own environment directly and your real config
file is never touched. It falls back to `ys harness point`/`reset` (once,
around the whole loop) only for an agent `--env-only` doesn't support yet
(opencode, codex-cli).

## Workspace isolation

`task.success_check` used to run via `shell=True` wherever `ys end` was
invoked from -- no working directory, no setup, no teardown, no reset
between repeats, so repeat 2 started from whatever repeat 1's agent left on
disk. `ys run` now gives every repeat its own workspace:

- `task.repo` + `task.ref`: a fresh `git clone` + `git checkout` per repeat,
  so every repeat starts from an identical tree. `task.workdir` alongside
  `repo` names a subdirectory within that clone to actually run in (e.g. a
  monorepo package).
- `task.workdir` alone (no `repo`): an existing directory you manage --
  typically a real project checkout -- used as-is. Repeats are *not*
  isolated in this mode; there's no ref to reset to.
- Neither set: falls back to the invoking process's own directory, matching
  pre-feature-2 behaviour exactly.
- `task.setup` / `task.teardown`: shell strings run once per repeat, before/
  after the agent, in that workspace -- the same trust model
  `success_check` already had (`shell=True` isn't new exposure). The
  workspace path is only ever passed via `cwd`/the `$YS_WORKDIR` env var,
  never interpolated into the command string itself.

**Safety rule:** only a directory yardstick itself created (a `repo` clone
under `~/.yardstick/workspaces/<run_id>/`) is ever deleted between/after
repeats. A `task.workdir` pointing at your real project -- or the
cwd fallback -- is never created or removed by yardstick, no matter what.
See `ys/workspace.py`'s module docstring for the two independent checks
`cleanup_workspace` makes before removing anything.

[`experiments/unattended-example.yaml`](experiments/unattended-example.yaml)
demonstrates both features together against a mocked model (free to run,
though `ys run` itself still needs a real agent CLI installed to actually
invoke).

## Experiment YAML

An experiment defines a task, one or more models, and a set of arms
(factor combinations to compare), plus which metrics to report. See
[`experiments/example.yaml`](experiments/example.yaml) for a minimal
end-to-end example using a mocked model response,
[`experiments/interactive-sonnet.yaml`](experiments/interactive-sonnet.yaml)
for a real-model interactive run,
[`experiments/cross-provider-example.yaml`](experiments/cross-provider-example.yaml)
for an Anthropic-vs-OpenAI comparison (mocked, so it costs nothing to run),
and [`experiments/unattended-example.yaml`](experiments/unattended-example.yaml)
for `ys run` plus workspace isolation (also mocked).

## Models and providers

`models:` maps a `factors.model` value to the `litellm_params` the proxy
registers it under -- these can point at any LiteLLM-supported provider
(Anthropic, OpenAI, Gemini, Bedrock, Vertex, a local model, ...), not just
Anthropic. Which provider a model belongs to is read off the `model` field's
own `<provider>/<id>` prefix (LiteLLM's own routing convention, e.g.
`openai/gpt-4o`) rather than a separate, independently-maintained
`provider:` field that could drift out of sync with it. A `factors.model`
value with no `models:` entry at all still falls back to the
`anthropic/<value>` convention if it isn't already prefixed -- unprefixed
values from before this was provider-agnostic keep working unchanged.

## Coding tools

Supported today: **Claude Code**, **opencode**, **Codex CLI**, and **Aider**.
Every one of these speaks exactly one wire protocol to whatever endpoint
it's pointed at (Claude Code and opencode: Anthropic Messages format; Codex
CLI and Aider: OpenAI Chat Completions format) -- the proxy, not the coding
tool, is what bridges that to whichever real provider an arm's model
actually resolves to. Codex CLI and Aider are best-effort/unverified against
a live install (see `ys/harness.py`'s module docstring and IMPROVEMENTS.md
feature 5 for exactly what's verified vs. reconstructed from public docs).
Cursor CLI, Copilot CLI, Gemini CLI, and Cline/Roo were evaluated and
deliberately left out -- `ys/harness.py`'s `EXCLUDED_TOOLS` comment states
why for each, rather than guessing at an env var/config key that could
silently route a run past the proxy to the real API.

Claude Code also supports project-level config: `ys harness point claude-code
--scope project` writes `./.claude/settings.json` (relative to the current
directory) instead of `~/.claude/settings.json`.

## Harness config safety

`ys harness point` writes an API key in plaintext into the agent's real
config file, backing up the original first. Two things limit the exposure:

- `ys harness point <agent> --env-only` prints `export` statements and never
  touches any config file at all -- safer when you don't need the config
  file itself changed, and the only way to point an agent with no config
  file (Aider).
- `ys end` automatically resets any agent config `ys harness point` touched,
  right after the run it was pointed for finishes. Pass
  `--keep-harness-pointed` to stay pointed across several repeats of the
  same arm without re-running `ys harness point` before each one -- the
  trade-off is the same plaintext-key exposure this flag otherwise closes.

opencode's config file is `.jsonc`, which may contain `//` comments;
`ys harness point`/`reset` only round-trip strict JSON, so comments would be
lost on `point` (though `reset` always restores the exact original bytes
regardless -- see `ys/harness.py`'s module docstring). `--env-only` isn't
supported for opencode (no verified environment-variable-only mechanism for
it), so this one is still a real exposure for opencode specifically -- back
up a `.jsonc` config with comments yourself before pointing it if that
matters to you.

## Attribution

Requests are attributed to the active run via the `x-ys-run` header when the
harness supports custom headers, or otherwise via the active-run state file
`ys start` writes — which is correct as long as only one run is active at a
time. A response landing just after `ys end` still attributes to the run
that just finished for a short drain window (60s), so the tail of a run
isn't dropped into `unattributed` just because it arrived a moment late.

## Export, budget guard, and leaderboard

```bash
ys export --exp experiments/example.yaml --csv runs.csv --json runs.json
```

Writes one row per recorded run of the experiment (every arm, every repeat —
including unfinished/abandoned runs and ones recorded under an older config,
each tagged with its own `status` and `config_matches_current` so the file is
interpretable without the database beside it). `ys compare`/`ys report`
narrow to the gate-passing, current-config population on purpose (see
`aggregate_run_metrics`); `ys export` hands over everything so you can narrow
it yourself.

```bash
ys start --exp experiments/example.yaml --arm arm-a --budget 5.00
```

`--budget` checks the arm's own already-*recorded* run history against the
threshold — `ys start` returns before the harness sends a single request, so
it cannot know this run's own cost yet. If the arm's finished runs already
total at or above the budget, `ys start` refuses to begin another repeat
(exit 1); if any of that history has a request neither LiteLLM nor a
declared `pricing:` override could price (`cost_source='unknown'`, see
finding 9), it says so explicitly rather than reporting "under budget" on
spend it can't actually verify. The run you're about to start still gets its
own real `cost_usd` only once it finishes — that's `ys end`'s printed summary,
unchanged.

```bash
ys leaderboard --exp exp-a.yaml --exp exp-b.yaml --metric cost_usd
```

Ranks each experiment's arms on one metric, side by side. Rank is scoped
*within* each experiment — two experiments are two different tasks, so a
mean on one isn't comparable in magnitude to a mean on the other — and every
non-baseline row reuses feature 3's significance test against its own
experiment's baseline, so an arm that merely looks best isn't presented as a
settled win when it isn't distinguishable from noise.

## Development

```bash
pytest
```

`YARDSTICK_HOME` (default `~/.yardstick`) holds the database, proxy/web
pid/port/log files, and generated proxy config.

## Provenance

`tools/provenance/` holds the ad-hoc probe scripts used to capture a real
LiteLLM callback payload against a live proxied request. `ys/collector.py`'s
field paths are verified against that captured shape rather than guessed
from LiteLLM's docs; see its module docstring. The scripts were originally
kept at the repo root as `explore/` and were moved here as documentation,
not source. They live under `tools/` rather than `docs/` specifically
because `docs/` is published verbatim to GitHub Pages
(`.github/workflows/pages.yml` uploads the whole directory on every push
that touches it) — putting probe scripts and captured payloads there would
publish them to the public site and trigger a docs redeploy on every edit.
