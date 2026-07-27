"""Point a coding agent's real config at the yardstick proxy, and reset it
back.

Agents supported today, in roughly the plan's order of demand
(IMPROVEMENTS.md feature 5):

  - claude-code  -- ~/.claude/settings.json, or a project-level
                    ./.claude/settings.json (--scope project)
  - opencode     -- ~/.config/opencode/opencode.jsonc (or .json)
  - codex-cli    -- ~/.codex/config.toml, **create-fresh-only** (see
                    `_point_codex_cli`'s docstring -- this repo has no TOML
                    dependency to safely round-trip an existing file)
  - aider        -- environment variables only, no config file at all
                    (`--env-only` is the *only* way to point it)

Four more named in the plan were evaluated and deliberately left out rather
than guessed at -- see the EXCLUDED_TOOLS comment below for why each one.

Why "provider" isn't a per-agent branch here: every one of these tools
speaks exactly one wire protocol to whatever endpoint it's pointed at,
independent of which real backend (Anthropic, OpenAI, Gemini, Bedrock, a
local model, ...) that endpoint ends up calling -- that translation is
LiteLLM's job (`ys/proxy.py`'s model_list), not the coding tool's. Claude
Code always speaks Anthropic's Messages format; Codex CLI and Aider always
speak OpenAI's Chat Completions format. So "map harness environment
variables per provider" (feature 5) means *per coding tool*, keyed by which
of those wire protocols it natively expects -- not per the experiment's
declared backend model, which `ys/proxy.py` already routes correctly
regardless of which agent is asking. A model factor value like
`openai/gpt-4o` changes what `ys/proxy.py` puts in `litellm_params`; it does
not change which env vars/config keys `ys harness point` writes for a given
agent.

Safety model: the first time `point()` touches a given agent/scope's config,
it snapshots the file's exact original bytes (or the fact that it didn't
exist at all) into a manifest under ~/.yardstick/harness_backups/. `reset()`
always restores from that manifest, never from whatever `point()` last
wrote -- so repeated point/reset cycles can't drift the "original" away from
what was really there before yardstick touched anything.

Known limitation: opencode's config file is .jsonc, which may contain //
comments. We only do a strict json.loads/json.dump round trip (no comment
support) -- comments would be lost on `point`, though `reset` always restores
the true original bytes regardless. If the live file isn't strict JSON, we
raise rather than guess at a lossy comment-stripping parse.

`--env-only` (`env_exports()`) sidesteps all of the above for the agents it
supports: it never reads or writes any file, so there's nothing to lose,
nothing to back up, and nothing that can be left mid-run if the process
dies -- it just returns the environment variables the caller should export
before launching the agent themselves. It's the only way to point an agent
that has no config file at all (aider), and the safer way to point one that
does, when you don't need `ys end`'s automatic reset (see cli.py's
`_auto_reset_pointed_harnesses`) to matter -- e.g. scripting the agent
directly in the same shell.
"""
import json
import os
from dataclasses import dataclass, field
from typing import Optional

from ys import paths


class HarnessError(Exception):
    pass


@dataclass
class AgentSpec:
    name: str
    config_path_candidates: list = field(default_factory=list)
    # Relative to the current working directory, e.g. ".claude/settings.json".
    # Only claude-code's project-level settings file is implemented (feature
    # 5 asked for it by name); other agents may well have an equivalent, but
    # we haven't verified one, so `scope="project"` raises for them instead
    # of guessing a path.
    project_relpath: Optional[str] = None
    # True for an agent with no config file yardstick manages at all --
    # every setting is an environment variable, so `point`/`reset`/`status`
    # (which are all about a file) don't apply; only `env_exports` does.
    env_only: bool = False

    def resolve_path(self, scope: str = "user") -> str:
        if scope == "project":
            if not self.project_relpath:
                raise HarnessError(
                    f"'{self.name}' has no known project-level config path -- "
                    "only claude-code's is verified (./.claude/settings.json). "
                    "Use --scope user (the default), or point it manually."
                )
            return os.path.join(os.getcwd(), self.project_relpath)
        for p in self.config_path_candidates:
            if os.path.exists(p):
                return p
        return self.config_path_candidates[0]


AGENTS = {
    "claude-code": AgentSpec(
        "claude-code",
        [os.path.expanduser("~/.claude/settings.json")],
        project_relpath=os.path.join(".claude", "settings.json"),
    ),
    "opencode": AgentSpec(
        "opencode",
        [
            os.path.expanduser("~/.config/opencode/opencode.jsonc"),
            os.path.expanduser("~/.config/opencode/opencode.json"),
        ],
    ),
    "codex-cli": AgentSpec(
        "codex-cli",
        [os.path.expanduser("~/.codex/config.toml")],
    ),
    "aider": AgentSpec("aider", env_only=True),
}

# Tools named in IMPROVEMENTS.md feature 5, in the plan's order, evaluated
# and deliberately left unimplemented -- each for a specific, stated reason,
# not merely "didn't get to it". Adding a wrong env var/config key silently
# routes an agent straight past the proxy to the real API (worse than no
# support at all), so these stay out until someone can verify the real
# mechanism against a live install.
#
#   - Gemini CLI: no environment-variable-only mechanism for redirecting it
#     at an arbitrary local endpoint could be confirmed from what's known
#     here (its documented auth paths are an API key, Vertex AI, or Google
#     OAuth login -- not a generic custom-base-URL override). Guessing a
#     plausible variable name is exactly what this feature's instructions
#     rule out.
#   - Cursor CLI: Cursor's agent traffic is proxied through Cursor's own
#     backend by design; there's no confirmed way to redirect it at an
#     arbitrary local endpoint at all, unlike a CLI that natively speaks an
#     open wire protocol (Anthropic Messages / OpenAI Chat Completions).
#   - GitHub Copilot CLI: excluded structurally, not just unverified.
#     Copilot CLI authenticates against GitHub's own Copilot backend and has
#     no supported custom-endpoint override, so there is nothing to point.
#   - Cline / Roo: both are VS Code extensions whose provider config lives
#     in the extension's own settings storage (VS Code global/workspace
#     state), not a single well-known on-disk path+schema this module could
#     confidently read and write like a plain JSON/TOML file.
EXCLUDED_TOOLS = ("gemini-cli", "cursor-cli", "copilot-cli", "cline", "roo")


def scopes_for_agent(agent_name: str) -> list:
    """Which scopes `agent_name` can plausibly be pointed at -- used by
    `ys end`'s automatic reset (cli.py) to check every scope an agent might
    have been pointed under, not just the default "user" one."""
    if agent_name not in AGENTS:
        raise HarnessError(f"unknown agent '{agent_name}'. Choose from: {', '.join(AGENTS)}")
    spec = AGENTS[agent_name]
    if spec.env_only:
        return []
    return ["user", "project"] if spec.project_relpath else ["user"]


def _backup_dir(create: bool = False) -> str:
    """`create=False` (the default) only computes the path -- no
    `paths.ensure_home()`, no `os.makedirs` -- so every read-only caller
    (`_load_manifest`, and hence `status()`/`reset()`'s existence check)
    can compute where a manifest *would* live without bringing
    YARDSTICK_HOME into existence just by asking. Only `_snapshot_if_absent`
    (the one path that's about to write a manifest) passes `create=True`.
    `ys doctor`'s hard "must not mutate anything" rule depends on this split:
    `check_harness_config` calls `harness.status()` in a loop (one call per
    agent), and before this split every one of those calls created
    ~/.yardstick, ~/.yardstick/experiments and ~/.yardstick/harness_backups
    as a side effect of merely asking whether a backup existed."""
    d = os.path.join(paths.YARDSTICK_HOME, "harness_backups")
    if create:
        paths.ensure_home()
        os.makedirs(d, exist_ok=True)
    return d


def _manifest_path(agent_name: str, scope: str = "user", create: bool = False) -> str:
    # "user" keeps the original, unsuffixed filename -- backward compatible
    # with any manifest a pre-feature-5 install already wrote, and the
    # common case stays exactly as before. Project scope gets its own
    # manifest file so pointing both scopes for the same agent can't clobber
    # each other's backup.
    suffix = "" if scope == "user" else f"-{scope}"
    return os.path.join(_backup_dir(create=create), f"{agent_name}{suffix}.json")


def _load_manifest(agent_name: str, scope: str = "user") -> Optional[dict]:
    """Read-only: always resolves the manifest path with `create=False`.
    Called from `status()`/`reset()`'s existence check as well as
    `_snapshot_if_absent`'s own "does one already exist" check below --
    none of those should ever create ~/.yardstick, and `os.path.exists`
    against a path inside a directory that doesn't exist simply (and
    correctly) returns False, no error."""
    path = _manifest_path(agent_name, scope)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _snapshot_if_absent(agent_name: str, config_path: str, scope: str = "user"):
    """Capture the true original state exactly once. No-op on later calls,
    even if the live file has since been modified by `point` -- the whole
    point is that this snapshot never moves. This is the one path allowed
    to create the backup directory (`create=True`) -- it's only ever called
    from `point()`, a command whose entire job is to write files."""
    if _load_manifest(agent_name, scope) is not None:
        return
    existed = os.path.exists(config_path)
    raw = None
    if existed:
        with open(config_path) as f:
            raw = f.read()
    manifest = {"config_path": config_path, "existed": existed, "raw": raw}
    with open(_manifest_path(agent_name, scope, create=True), "w") as f:
        json.dump(manifest, f, indent=2)


def _load_json_strict(config_path: str) -> dict:
    if not os.path.exists(config_path):
        return {}
    with open(config_path) as f:
        text = f.read().strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise HarnessError(
            f"{config_path} is not strict JSON (comments/trailing commas aren't "
            f"supported by yardstick's merge) -- {e}. Edit it by hand instead, or "
            f"remove comments temporarily."
        )


def _deep_set(d: dict, path: list, value):
    cur = d
    for key in path[:-1]:
        cur = cur.setdefault(key, {})
    cur[path[-1]] = value


def _claude_code_env_vars(port: int, api_key: str, model: Optional[str], pin_background: bool) -> dict:
    """The exact env vars claude-code needs, as a flat dict -- the one
    source of truth `point()` (writing them into settings.json's `env`
    block) and `env_exports()` (printing them as `export` statements,
    `--env-only`) both read from, so the two paths can't drift apart.

    `pin_background` (default True) additionally pins
    `ANTHROPIC_SMALL_FAST_MODEL`/`ANTHROPIC_DEFAULT_HAIKU_MODEL` -- the
    model ids Claude Code sends its background requests (title generation,
    etc.) to -- at the same registered model_name as the main arm. This is
    a real trade-off, not a free safety net (finding 27 in IMPROVEMENTS.md):
    background traffic that a real session would send to a small, cheap
    model instead runs on -- and is billed/weighted as -- the arm's own
    model, inflating cost_usd/billable_tokens (which are deliberately
    run-wide, since that spend is real; finding 4) relative to an
    unmeasured session. It's still the right *default*, because
    `ys proxy up` only registers a `mock_response`/params for the exact
    model_name the experiment declared (see ys/proxy.py's
    generate_config) -- unpinned background traffic would carry its own
    (harness-default) model id, which the proxy's `model_name: "*"`
    catch-all still routes, but straight through to the real API, silently
    defeating a mock experiment's entire point as a dry smoke test.
    """
    env = {
        "ANTHROPIC_BASE_URL": f"http://localhost:{port}",
        "ANTHROPIC_API_KEY": api_key,
    }
    if model:
        env["ANTHROPIC_MODEL"] = model
        if pin_background:
            env["ANTHROPIC_SMALL_FAST_MODEL"] = model
            env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = model
    return env


def _aider_env_vars(port: int, api_key: str, model: Optional[str]) -> dict:
    """Aider embeds LiteLLM directly rather than shipping its own config
    file for this, so it's naturally an env-only agent (`AgentSpec.env_only
    = True` -- there's no file for `point`/`reset` to manage at all).

    Two candidate env var names exist in the wild for overriding an
    OpenAI-compatible client's base URL: `OPENAI_API_BASE` (the name
    Aider's own docs use for pointing it at a custom/local OpenAI-compatible
    endpoint) and `OPENAI_BASE_URL` (the current openai-python SDK's own
    name for the same thing). We were not able to verify which one Aider's
    installed version actually reads without running it, so -- rather than
    pick one and risk it being silently ignored -- both are exported; an
    unrecognized one is simply an unused extra env var, not a wrong value
    that routes anywhere. This is a hedge across two real, documented names,
    not an invented one.

    `model`, when given, is exported as `AIDER_MODEL=openai/<value>` --
    Aider takes `--model`/`AIDER_MODEL` in `<provider>/<name>` form like any
    other LiteLLM-fronted tool, and the `openai/` prefix selects Aider's
    OpenAI-compatible client regardless of which real backend the proxy's
    model_list ultimately routes `<value>` to.
    """
    base_url = f"http://localhost:{port}/v1"
    env = {
        "OPENAI_API_KEY": api_key,
        "OPENAI_API_BASE": base_url,
        "OPENAI_BASE_URL": base_url,
    }
    if model:
        env["AIDER_MODEL"] = f"openai/{model}"
    return env


def env_exports(
    agent_name: str,
    port: int,
    api_key: str,
    model: Optional[str] = None,
    pin_background: bool = True,
) -> dict:
    """`--env-only`: the environment variables that would point `agent_name`
    at the proxy, without ever touching a file. See the module docstring for
    why this is the higher-priority half of feature 5 -- it never risks the
    user's real config, and it's the only mechanism at all for an agent with
    no config file (aider).

    Raises HarnessError for an agent with no verified environment-variable
    path (opencode, codex-cli) -- both are confirmed to need their config
    file for the base URL/API key specifically, so pointing them still goes
    through `point()`.
    """
    if agent_name not in AGENTS:
        raise HarnessError(f"unknown agent '{agent_name}'. Choose from: {', '.join(AGENTS)}")

    if agent_name == "claude-code":
        return _claude_code_env_vars(port, api_key, model, pin_background)
    if agent_name == "aider":
        return _aider_env_vars(port, api_key, model)
    if agent_name == "opencode":
        raise HarnessError(
            "opencode's base URL/API key are only confirmed to work via its JSON "
            "config (provider.anthropic.options.*) -- yardstick has no verified "
            "environment-variable-only mechanism for opencode, so --env-only isn't "
            "supported for it. Use `ys harness point opencode` instead."
        )
    if agent_name == "codex-cli":
        raise HarnessError(
            "Codex CLI's base URL is only configurable via model_providers entries "
            "in ~/.codex/config.toml (its env_key just names which env var holds the "
            "API key, not the base URL itself) -- --env-only isn't supported for it. "
            "Use `ys harness point codex-cli` instead."
        )
    raise HarnessError(f"--env-only is not supported for '{agent_name}'.")


def point(
    agent_name: str,
    port: int,
    api_key: str,
    model: Optional[str] = None,
    pin_background: bool = True,
    scope: str = "user",
) -> str:
    """`model` should be the arm's `factors.model` value -- i.e. exactly the
    `model_name` `ys proxy up` registered it under, not the underlying
    provider model id. Without it, the agent requests whatever model id it
    defaults to, which the proxy has never heard of and rejects (finding 3);
    the request would only survive via the proxy's catch-all passthrough,
    which skips any mock_response/params the experiment declared for the
    real model_name.

    Written into each agent's config as that agent expects it: verbatim env
    vars for Claude Code, provider-prefixed (`anthropic/<model>`) for
    opencode's `model` key, `model_provider`/`model` for codex-cli's
    config.toml.

    `scope` is "user" (default, `~/.claude/settings.json`) or "project"
    (`./.claude/settings.json`, feature 5) -- only claude-code has a
    verified project-level path; any other agent raises HarnessError for
    `scope="project"` rather than guess at one.

    `pin_background` is documented on `_claude_code_env_vars` above (claude-
    code only; ignored otherwise, since it's a Claude-Code-specific
    background-model concept -- see finding 27 in IMPROVEMENTS.md).
    """
    if agent_name not in AGENTS:
        raise HarnessError(f"unknown agent '{agent_name}'. Choose from: {', '.join(AGENTS)}")

    spec = AGENTS[agent_name]
    if spec.env_only:
        raise HarnessError(
            f"'{agent_name}' has no config file yardstick manages -- use "
            f"`ys harness point {agent_name} --env-only` to print the environment "
            "variables to export instead."
        )

    if agent_name == "codex-cli":
        return _point_codex_cli(spec, port, api_key, model)

    config_path = spec.resolve_path(scope)
    _snapshot_if_absent(agent_name, config_path, scope)

    config = _load_json_strict(config_path)
    base_url = f"http://localhost:{port}"

    if agent_name == "claude-code":
        for key, value in _claude_code_env_vars(port, api_key, model, pin_background).items():
            _deep_set(config, ["env", key], value)
    elif agent_name == "opencode":
        # opencode's anthropic provider POSTs to `{baseURL}/messages` without
        # adding a version prefix itself, but LiteLLM's Anthropic-compatible
        # route lives at `/v1/messages` -- so baseURL must include `/v1`
        # (unlike claude-code, which already sends `/v1/messages` itself).
        _deep_set(config, ["provider", "anthropic", "options", "baseURL"], f"{base_url}/v1")
        _deep_set(config, ["provider", "anthropic", "options", "apiKey"], api_key)
        if model:
            _deep_set(config, ["model"], f"anthropic/{model}")

    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")

    return config_path


def _point_codex_cli(spec: AgentSpec, port: int, api_key: str, model: Optional[str]) -> str:
    """Codex CLI's config lives at ~/.codex/config.toml, a `[model_providers.X]`
    table with `base_url`/`env_key`/`wire_api` plus top-level `model_provider`/
    `model` -- reconstructed from Codex CLI's published docs, not verified
    against a live install (see IMPROVEMENTS.md feature 5 and the PR body).

    Unlike the JSON agents above, this never merges into an existing file:
    this repo has no TOML parser/writer dependency (adding one is out of
    scope -- pyproject.toml isn't in this change's file set), so there's no
    safe way to round-trip arbitrary existing TOML content the way
    `_load_json_strict` does for JSON. Rather than hand-roll a parser and
    risk silently mangling a real config, a config.toml with any existing
    content is refused outright; this only ever creates a fresh file. The
    api_key itself is never written into the file -- `env_key` just names
    which environment variable Codex should read it from at runtime, so the
    caller still needs to export that variable themselves (same as
    `ys start`'s own printed instructions for ANTHROPIC_API_KEY).
    """
    config_path = spec.resolve_path("user")
    existing = ""
    if os.path.exists(config_path):
        with open(config_path) as f:
            existing = f.read().strip()
    if existing:
        raise HarnessError(
            f"{config_path} already has content -- yardstick doesn't parse/round-trip "
            "TOML (no dependency for it in this repo), so it can only create a fresh "
            "config.toml, never edit an existing one, to avoid silently destroying real "
            "content. Back up/remove it yourself and re-run, or add a "
            "[model_providers.yardstick] entry by hand (base_url="
            f"\"http://localhost:{port}/v1\", env_key=\"OPENAI_API_KEY\", "
            "wire_api=\"chat\")."
        )

    _snapshot_if_absent("codex-cli", config_path)

    lines = [
        "# Generated by `ys harness point codex-cli` -- see ys/harness.py's",
        "# _point_codex_cli docstring: this file is only ever created fresh,",
        "# never merged into existing content.",
        'model_provider = "yardstick"',
    ]
    if model:
        lines.append(f'model = "{model}"')
    lines += [
        "",
        "[model_providers.yardstick]",
        'name = "yardstick"',
        f'base_url = "http://localhost:{port}/v1"',
        'env_key = "OPENAI_API_KEY"',
        'wire_api = "chat"',
    ]

    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    return config_path


def reset(agent_name: str, scope: str = "user") -> str:
    if agent_name not in AGENTS:
        raise HarnessError(f"unknown agent '{agent_name}'. Choose from: {', '.join(AGENTS)}")
    if AGENTS[agent_name].env_only:
        raise HarnessError(
            f"'{agent_name}' has no config file yardstick manages -- there's nothing "
            "to reset (--env-only never wrote anything)."
        )

    manifest = _load_manifest(agent_name, scope)
    if manifest is None:
        raise HarnessError(
            f"no backup found for '{agent_name}' ({scope} scope) -- `ys harness point` "
            "was never run for it (or the backup dir was cleared), so there's nothing "
            "to restore."
        )

    config_path = manifest["config_path"]
    if manifest["existed"]:
        with open(config_path, "w") as f:
            f.write(manifest["raw"])
    elif os.path.exists(config_path):
        os.remove(config_path)

    return config_path


@dataclass
class AgentStatus:
    agent: str
    config_path: str
    config_exists: bool
    pointed_at_proxy: bool
    has_backup: bool
    scope: str = "user"
    env_only: bool = False


def status(agent_name: str, scope: str = "user") -> AgentStatus:
    if agent_name not in AGENTS:
        raise HarnessError(f"unknown agent '{agent_name}'. Choose from: {', '.join(AGENTS)}")

    spec = AGENTS[agent_name]
    if spec.env_only:
        return AgentStatus(
            agent=agent_name,
            config_path="",
            config_exists=False,
            pointed_at_proxy=False,
            has_backup=False,
            scope=scope,
            env_only=True,
        )

    config_path = spec.resolve_path(scope)
    config_exists = os.path.exists(config_path)

    pointed = False
    if config_exists:
        if agent_name == "codex-cli":
            # Not JSON -- a plain substring check on the raw TOML is enough
            # to answer "does this look pointed at us", without needing a
            # TOML parser (see _point_codex_cli's docstring for why we don't
            # have one).
            with open(config_path) as f:
                raw = f.read()
            pointed = "localhost" in raw or "127.0.0.1" in raw
        else:
            try:
                config = _load_json_strict(config_path)
                if agent_name == "claude-code":
                    url = config.get("env", {}).get("ANTHROPIC_BASE_URL", "")
                else:
                    url = config.get("provider", {}).get("anthropic", {}).get("options", {}).get("baseURL", "")
                pointed = "localhost" in url or "127.0.0.1" in url
            except HarnessError:
                pass

    return AgentStatus(
        agent=agent_name,
        config_path=config_path,
        config_exists=config_exists,
        pointed_at_proxy=pointed,
        has_backup=_load_manifest(agent_name, scope) is not None,
        scope=scope,
    )
