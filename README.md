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

## Experiment YAML

An experiment defines a task, one or more models, and a set of arms
(factor combinations to compare), plus which metrics to report. See
[`experiments/example.yaml`](experiments/example.yaml) for a minimal
end-to-end example using a mocked model response,
[`experiments/interactive-sonnet.yaml`](experiments/interactive-sonnet.yaml)
for a real-model interactive run, and
[`experiments/cross-provider-example.yaml`](experiments/cross-provider-example.yaml)
for an Anthropic-vs-OpenAI comparison (mocked, so it costs nothing to run).

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
time.

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
