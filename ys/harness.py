"""Point a coding agent's real config at the yardstick proxy, and reset it
back. Two agents are supported today: Claude Code (~/.claude/settings.json)
and opencode (~/.config/opencode/opencode.jsonc).

Safety model: the first time `point()` touches a given agent's config, it
snapshots the file's exact original bytes (or the fact that it didn't exist
at all) into a manifest under ~/.yardstick/harness_backups/. `reset()` always
restores from that manifest, never from whatever `point()` last wrote -- so
repeated point/reset cycles can't drift the "original" away from what was
really there before yardstick touched anything.

Known limitation: opencode's config file is .jsonc, which may contain //
comments. We only do a strict json.loads/json.dump round trip (no comment
support) -- comments would be lost on `point`, though `reset` always restores
the true original bytes regardless. If the live file isn't strict JSON, we
raise rather than guess at a lossy comment-stripping parse.
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

    def resolve_path(self) -> str:
        for p in self.config_path_candidates:
            if os.path.exists(p):
                return p
        return self.config_path_candidates[0]


AGENTS = {
    "claude-code": AgentSpec("claude-code", [os.path.expanduser("~/.claude/settings.json")]),
    "opencode": AgentSpec(
        "opencode",
        [
            os.path.expanduser("~/.config/opencode/opencode.jsonc"),
            os.path.expanduser("~/.config/opencode/opencode.json"),
        ],
    ),
}


def _backup_dir() -> str:
    paths.ensure_home()
    d = os.path.join(paths.YARDSTICK_HOME, "harness_backups")
    os.makedirs(d, exist_ok=True)
    return d


def _manifest_path(agent_name: str) -> str:
    return os.path.join(_backup_dir(), f"{agent_name}.json")


def _load_manifest(agent_name: str) -> Optional[dict]:
    path = _manifest_path(agent_name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _snapshot_if_absent(agent_name: str, config_path: str):
    """Capture the true original state exactly once. No-op on later calls,
    even if the live file has since been modified by `point` -- the whole
    point is that this snapshot never moves."""
    if _load_manifest(agent_name) is not None:
        return
    existed = os.path.exists(config_path)
    raw = None
    if existed:
        with open(config_path) as f:
            raw = f.read()
    manifest = {"config_path": config_path, "existed": existed, "raw": raw}
    with open(_manifest_path(agent_name), "w") as f:
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


def point(agent_name: str, port: int, api_key: str, model: Optional[str] = None) -> str:
    """`model` should be the arm's `factors.model` value -- i.e. exactly the
    `model_name` `ys proxy up` registered it under, not the underlying
    provider model id. Without it, the agent requests whatever model id it
    defaults to, which the proxy has never heard of and rejects (finding 3);
    the request would only survive via the proxy's catch-all passthrough,
    which skips any mock_response/params the experiment declared for the
    real model_name.

    Written into each agent's config as that agent expects it: verbatim for
    Claude Code's `ANTHROPIC_MODEL` env vars, provider-prefixed
    (`anthropic/<model>`) for opencode's `model` key."""
    if agent_name not in AGENTS:
        raise HarnessError(f"unknown agent '{agent_name}'. Choose from: {', '.join(AGENTS)}")

    spec = AGENTS[agent_name]
    config_path = spec.resolve_path()
    _snapshot_if_absent(agent_name, config_path)

    config = _load_json_strict(config_path)
    base_url = f"http://localhost:{port}"

    if agent_name == "claude-code":
        _deep_set(config, ["env", "ANTHROPIC_BASE_URL"], base_url)
        _deep_set(config, ["env", "ANTHROPIC_API_KEY"], api_key)
        if model:
            # Claude Code also issues background requests (title generation,
            # etc.) against a separate "small/fast" model id -- pin those to
            # the same registered model_name too, or they 400 against the
            # same model_list.
            _deep_set(config, ["env", "ANTHROPIC_MODEL"], model)
            _deep_set(config, ["env", "ANTHROPIC_SMALL_FAST_MODEL"], model)
            _deep_set(config, ["env", "ANTHROPIC_DEFAULT_HAIKU_MODEL"], model)
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


def reset(agent_name: str) -> str:
    if agent_name not in AGENTS:
        raise HarnessError(f"unknown agent '{agent_name}'. Choose from: {', '.join(AGENTS)}")

    manifest = _load_manifest(agent_name)
    if manifest is None:
        raise HarnessError(
            f"no backup found for '{agent_name}' -- `ys harness point` was never run for it "
            "(or the backup dir was cleared), so there's nothing to restore."
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


def status(agent_name: str) -> AgentStatus:
    if agent_name not in AGENTS:
        raise HarnessError(f"unknown agent '{agent_name}'. Choose from: {', '.join(AGENTS)}")

    spec = AGENTS[agent_name]
    config_path = spec.resolve_path()
    config_exists = os.path.exists(config_path)

    pointed = False
    if config_exists:
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
        has_backup=_load_manifest(agent_name) is not None,
    )
