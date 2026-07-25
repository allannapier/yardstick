# yardstick

Agent / harness / model efficiency measurement rig. Yardstick sits between a
coding agent (Claude Code, opencode, ...) and the LLM API as a LiteLLM proxy,
recording every request/response so you can compare arms of an experiment
(e.g. same task, different model or harness config) on cost, tokens, turns,
tool-call efficiency, and task success.

## How it works

1. `ys proxy up` starts a LiteLLM proxy configured from your experiment YAML(s).
2. `ys harness point` repoints an agent's config (e.g. Claude Code's
   `~/.claude/settings.json`) at that proxy, backing up the original first.
3. `ys start --exp ... --arm ...` marks a run active; you then drive the
   agent through its task as normal.
4. `ys end` finishes the run, runs the arm's `success_check`, and prints
   headline metrics.
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

ys harness point claude-code               # point Claude Code at the proxy
ys start --exp experiments/example.yaml --arm arm-a

# ... drive the agent through the task ...

ys end                                     # score + print headline metrics
ys harness reset claude-code               # restore the agent's real config
ys proxy down
```

Compare recorded runs for an experiment:

```bash
ys compare --exp experiments/example.yaml
ys report  --exp experiments/example.yaml --html report.html
```

Optional dashboard for setting up experiments and browsing runs in a browser:

```bash
ys web up
```

## Experiment YAML

An experiment defines a task, one or more models, and a set of arms
(factor combinations to compare), plus which metrics to report. See
[`experiments/example.yaml`](experiments/example.yaml) for a minimal
end-to-end example using a mocked model response, and
[`experiments/interactive-sonnet.yaml`](experiments/interactive-sonnet.yaml)
for a real-model interactive run.

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
