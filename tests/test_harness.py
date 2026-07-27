import json
import os

import pytest

from ys import harness


@pytest.fixture(autouse=True)
def fake_agents(monkeypatch, tmp_path):
    """Never touch the real ~/.claude, ~/.config/opencode or ~/.codex during
    tests. claude-code keeps a project_relpath like the real AgentSpec, so
    scope="project" tests exercise the real resolution logic against a fake
    "user" path instead of guessing at a wholly separate code path."""
    claude_path = str(tmp_path / "claude_settings.json")
    opencode_path = str(tmp_path / "opencode.jsonc")
    codex_path = str(tmp_path / "codex_config.toml")
    monkeypatch.setitem(
        harness.AGENTS,
        "claude-code",
        harness.AgentSpec(
            "claude-code", [claude_path], project_relpath=os.path.join(".claude", "settings.json")
        ),
    )
    monkeypatch.setitem(harness.AGENTS, "opencode", harness.AgentSpec("opencode", [opencode_path]))
    monkeypatch.setitem(harness.AGENTS, "codex-cli", harness.AgentSpec("codex-cli", [codex_path]))
    return {"claude-code": claude_path, "opencode": opencode_path, "codex-cli": codex_path}


def test_point_unknown_agent_raises():
    with pytest.raises(harness.HarnessError):
        harness.point("not-a-real-agent", 4000, "key")


def test_point_creates_config_when_none_existed(fake_agents):
    path = harness.point("claude-code", 4000, "sk-test")
    assert path == fake_agents["claude-code"]
    with open(path) as f:
        config = json.load(f)
    assert config["env"]["ANTHROPIC_BASE_URL"] == "http://localhost:4000"
    assert config["env"]["ANTHROPIC_API_KEY"] == "sk-test"


def test_point_preserves_unrelated_claude_settings(fake_agents):
    with open(fake_agents["claude-code"], "w") as f:
        json.dump({"model": "sonnet", "env": {"OTHER_VAR": "keep-me"}}, f)

    harness.point("claude-code", 4000, "sk-test")

    with open(fake_agents["claude-code"]) as f:
        config = json.load(f)
    assert config["model"] == "sonnet"
    assert config["env"]["OTHER_VAR"] == "keep-me"
    assert config["env"]["ANTHROPIC_BASE_URL"] == "http://localhost:4000"


def test_point_with_model_sets_claude_code_model_env_vars(fake_agents):
    path = harness.point("claude-code", 4000, "sk-test", model="claude-sonnet-5")
    with open(path) as f:
        config = json.load(f)
    assert config["env"]["ANTHROPIC_MODEL"] == "claude-sonnet-5"
    assert config["env"]["ANTHROPIC_SMALL_FAST_MODEL"] == "claude-sonnet-5"
    assert config["env"]["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "claude-sonnet-5"


def test_point_without_model_does_not_set_model_env_vars(fake_agents):
    path = harness.point("claude-code", 4000, "sk-test")
    with open(path) as f:
        config = json.load(f)
    assert "ANTHROPIC_MODEL" not in config.get("env", {})


# --- finding 27: --no-pin-background opt-out --------------------------------


def test_point_pins_background_small_fast_model_by_default(fake_agents):
    """Default behaviour (pin_background=True) is unchanged from before
    finding 27 -- this is the same assertion as
    test_point_with_model_sets_claude_code_model_env_vars, pinned here too
    so a future change to the default can't silently widen finding 27's
    hole without a test noticing."""
    path = harness.point("claude-code", 4000, "sk-test", model="claude-sonnet-5")
    with open(path) as f:
        config = json.load(f)
    assert config["env"]["ANTHROPIC_SMALL_FAST_MODEL"] == "claude-sonnet-5"
    assert config["env"]["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "claude-sonnet-5"


def test_point_no_pin_background_leaves_small_fast_model_env_vars_unset(fake_agents):
    """Regression test for finding 27: with pin_background=False, background
    (title-generation) traffic is left to request the harness's own default
    small/fast model instead of being routed through the arm's model --
    ANTHROPIC_MODEL still gets set (the main turn still needs to be pinned),
    but the two background-model env vars must not be touched."""
    path = harness.point(
        "claude-code", 4000, "sk-test", model="claude-sonnet-5", pin_background=False
    )
    with open(path) as f:
        config = json.load(f)
    assert config["env"]["ANTHROPIC_MODEL"] == "claude-sonnet-5"
    assert "ANTHROPIC_SMALL_FAST_MODEL" not in config["env"]
    assert "ANTHROPIC_DEFAULT_HAIKU_MODEL" not in config["env"]


def test_point_no_pin_background_is_a_no_op_without_a_model(fake_agents):
    """pin_background only matters once a model is being pinned at all --
    with no model, there's nothing to pin the background traffic to
    either."""
    path = harness.point("claude-code", 4000, "sk-test", pin_background=False)
    with open(path) as f:
        config = json.load(f)
    assert "ANTHROPIC_MODEL" not in config.get("env", {})
    assert "ANTHROPIC_SMALL_FAST_MODEL" not in config.get("env", {})


def test_point_with_model_sets_opencode_model(fake_agents):
    path = harness.point("opencode", 4010, "sk-test", model="claude-sonnet-5")
    with open(path) as f:
        config = json.load(f)
    assert config["model"] == "anthropic/claude-sonnet-5"


def test_point_preserves_unrelated_opencode_settings(fake_agents):
    with open(fake_agents["opencode"], "w") as f:
        json.dump({"$schema": "https://opencode.ai/config.json", "mcp": {"foo": {"type": "local"}}}, f)

    harness.point("opencode", 4010, "sk-test")

    with open(fake_agents["opencode"]) as f:
        config = json.load(f)
    assert config["mcp"]["foo"]["type"] == "local"
    assert config["provider"]["anthropic"]["options"]["baseURL"] == "http://localhost:4010/v1"
    assert config["provider"]["anthropic"]["options"]["apiKey"] == "sk-test"


def test_point_rejects_non_strict_json(fake_agents):
    with open(fake_agents["opencode"], "w") as f:
        f.write('{\n  // a comment\n  "mcp": {}\n}')

    with pytest.raises(harness.HarnessError, match="not strict JSON"):
        harness.point("opencode", 4000, "sk-test")


def test_reset_without_prior_point_raises(fake_agents):
    with pytest.raises(harness.HarnessError, match="no backup found"):
        harness.reset("claude-code")


def test_reset_restores_original_bytes_exactly(fake_agents):
    original = '{\n  "model": "opus",\n  "customField": [1, 2, 3]\n}'
    with open(fake_agents["claude-code"], "w") as f:
        f.write(original)

    harness.point("claude-code", 4000, "sk-test")
    restored_path = harness.reset("claude-code")

    assert restored_path == fake_agents["claude-code"]
    with open(fake_agents["claude-code"]) as f:
        assert f.read() == original


def test_reset_deletes_file_that_never_existed(fake_agents):
    assert not __import__("os").path.exists(fake_agents["claude-code"])
    harness.point("claude-code", 4000, "sk-test")
    assert __import__("os").path.exists(fake_agents["claude-code"])

    harness.reset("claude-code")
    assert not __import__("os").path.exists(fake_agents["claude-code"])


def test_repeated_point_does_not_move_the_backup_baseline(fake_agents):
    original = '{"model": "sonnet"}'
    with open(fake_agents["claude-code"], "w") as f:
        f.write(original)

    harness.point("claude-code", 4000, "sk-test")
    harness.point("claude-code", 4001, "sk-test-2")  # point again, e.g. different port

    harness.reset("claude-code")
    with open(fake_agents["claude-code"]) as f:
        assert f.read() == original


def test_status_reports_pointed_and_backup_state(fake_agents):
    before = harness.status("claude-code")
    assert before.config_exists is False
    assert before.pointed_at_proxy is False
    assert before.has_backup is False

    harness.point("claude-code", 4000, "sk-test")
    after = harness.status("claude-code")
    assert after.config_exists is True
    assert after.pointed_at_proxy is True
    assert after.has_backup is True

    harness.reset("claude-code")
    final = harness.status("claude-code")
    assert final.config_exists is False
    assert final.has_backup is True  # backup manifest itself is retained


# --- feature 5: --env-only ---------------------------------------------------


def test_env_only_leaves_the_config_file_completely_untouched(fake_agents):
    """The headline regression test for feature 5's harness-safety half:
    env_exports must never read or write the agent's config file at all --
    not even to check whether it exists. Reverting to a point()-only
    implementation (no env_exports) would fail this by construction, since
    there'd be no way to get the export statements without writing the file
    point() always writes."""
    assert not os.path.exists(fake_agents["claude-code"])
    exports = harness.env_exports("claude-code", 4000, "sk-test", model="claude-sonnet-5")
    assert not os.path.exists(fake_agents["claude-code"])
    assert exports["ANTHROPIC_BASE_URL"] == "http://localhost:4000"
    assert exports["ANTHROPIC_API_KEY"] == "sk-test"
    assert exports["ANTHROPIC_MODEL"] == "claude-sonnet-5"
    assert exports["ANTHROPIC_SMALL_FAST_MODEL"] == "claude-sonnet-5"


def test_env_only_does_not_touch_a_preexisting_config_either(fake_agents):
    """Same as above, but the file already existed with real content --
    env_exports must leave it byte-for-byte alone, unlike point() (which
    would rewrite it, backup or not)."""
    original = '{"model": "opus", "env": {"OTHER_VAR": "keep-me"}}'
    with open(fake_agents["claude-code"], "w") as f:
        f.write(original)

    harness.env_exports("claude-code", 4000, "sk-test")

    with open(fake_agents["claude-code"]) as f:
        assert f.read() == original


def test_env_exports_without_model_omits_model_vars(fake_agents):
    exports = harness.env_exports("claude-code", 4000, "sk-test")
    assert "ANTHROPIC_MODEL" not in exports
    assert "ANTHROPIC_SMALL_FAST_MODEL" not in exports


def test_env_exports_no_pin_background_omits_background_vars(fake_agents):
    exports = harness.env_exports(
        "claude-code", 4000, "sk-test", model="claude-sonnet-5", pin_background=False
    )
    assert exports["ANTHROPIC_MODEL"] == "claude-sonnet-5"
    assert "ANTHROPIC_SMALL_FAST_MODEL" not in exports
    assert "ANTHROPIC_DEFAULT_HAIKU_MODEL" not in exports


def test_env_exports_aider_returns_openai_shaped_vars(fake_agents):
    exports = harness.env_exports("aider", 4000, "sk-test", model="claude-sonnet-5")
    assert exports["OPENAI_API_KEY"] == "sk-test"
    assert exports["OPENAI_API_BASE"] == "http://localhost:4000/v1"
    assert exports["OPENAI_BASE_URL"] == "http://localhost:4000/v1"
    assert exports["AIDER_MODEL"] == "openai/claude-sonnet-5"


def test_env_exports_opencode_unsupported(fake_agents):
    with pytest.raises(harness.HarnessError, match="no verified environment-variable-only"):
        harness.env_exports("opencode", 4000, "sk-test")


def test_env_exports_codex_cli_unsupported(fake_agents):
    with pytest.raises(harness.HarnessError, match="base URL is only configurable"):
        harness.env_exports("codex-cli", 4000, "sk-test")


def test_env_exports_unknown_agent_raises(fake_agents):
    with pytest.raises(harness.HarnessError, match="unknown agent"):
        harness.env_exports("not-a-real-agent", 4000, "sk-test")


# --- feature 5: aider is env-only, point()/reset() refuse it ----------------


def test_point_refuses_env_only_agent(fake_agents):
    with pytest.raises(harness.HarnessError, match="no config file yardstick manages"):
        harness.point("aider", 4000, "sk-test")


def test_reset_refuses_env_only_agent(fake_agents):
    with pytest.raises(harness.HarnessError, match="no config file yardstick manages"):
        harness.reset("aider")


def test_status_reports_env_only_agent_without_touching_anything(fake_agents):
    s = harness.status("aider")
    assert s.env_only is True
    assert s.config_exists is False
    assert s.pointed_at_proxy is False
    assert s.has_backup is False


def test_scopes_for_agent(fake_agents):
    assert harness.scopes_for_agent("claude-code") == ["user", "project"]
    assert harness.scopes_for_agent("opencode") == ["user"]
    assert harness.scopes_for_agent("aider") == []


# --- feature 5: project-level claude-code settings --------------------------


def test_point_project_scope_writes_to_project_path_not_home(fake_agents, tmp_path, monkeypatch):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    monkeypatch.chdir(project_dir)

    path = harness.point("claude-code", 4000, "sk-test", model="claude-sonnet-5", scope="project")

    assert path == str(project_dir / ".claude" / "settings.json")
    assert os.path.exists(path)
    assert not os.path.exists(fake_agents["claude-code"])  # user-scope path untouched
    with open(path) as f:
        config = json.load(f)
    assert config["env"]["ANTHROPIC_MODEL"] == "claude-sonnet-5"


def test_reset_project_scope_restores_project_path(fake_agents, tmp_path, monkeypatch):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    monkeypatch.chdir(project_dir)

    project_settings = project_dir / ".claude" / "settings.json"
    project_settings.parent.mkdir(parents=True, exist_ok=True)
    original = '{"model": "opus"}'
    project_settings.write_text(original)

    harness.point("claude-code", 4000, "sk-test", scope="project")
    restored_path = harness.reset("claude-code", scope="project")

    assert restored_path == str(project_settings)
    assert project_settings.read_text() == original


def test_project_and_user_scope_backups_are_independent(fake_agents, tmp_path, monkeypatch):
    """Pointing both scopes for the same agent must not let one scope's
    backup manifest clobber the other's."""
    with open(fake_agents["claude-code"], "w") as f:
        f.write('{"scope": "user-original"}')

    project_dir = tmp_path / "project"
    project_dir.mkdir()
    monkeypatch.chdir(project_dir)

    harness.point("claude-code", 4000, "sk-test", scope="user")
    harness.point("claude-code", 4001, "sk-test", scope="project")

    harness.reset("claude-code", scope="user")
    harness.reset("claude-code", scope="project")

    with open(fake_agents["claude-code"]) as f:
        assert json.load(f) == {"scope": "user-original"}
    assert not os.path.exists(project_dir / ".claude" / "settings.json")


def test_project_scope_unsupported_for_opencode(fake_agents):
    with pytest.raises(harness.HarnessError, match="no known project-level config path"):
        harness.point("opencode", 4000, "sk-test", scope="project")


# --- feature 5: codex-cli -----------------------------------------------------


def test_point_codex_cli_creates_fresh_config(fake_agents):
    path = harness.point("codex-cli", 4000, "sk-test", model="claude-sonnet-5")
    assert path == fake_agents["codex-cli"]
    with open(path) as f:
        content = f.read()
    assert 'model_provider = "yardstick"' in content
    assert 'model = "claude-sonnet-5"' in content
    assert 'base_url = "http://localhost:4000/v1"' in content
    assert 'env_key = "OPENAI_API_KEY"' in content


def test_point_codex_cli_refuses_to_touch_existing_populated_file(fake_agents):
    with open(fake_agents["codex-cli"], "w") as f:
        f.write('model_provider = "openai"\n')

    with pytest.raises(harness.HarnessError, match="already has content"):
        harness.point("codex-cli", 4000, "sk-test")

    with open(fake_agents["codex-cli"]) as f:
        assert f.read() == 'model_provider = "openai"\n'


def test_point_codex_cli_treats_empty_file_as_absent(fake_agents):
    with open(fake_agents["codex-cli"], "w") as f:
        f.write("   \n")

    path = harness.point("codex-cli", 4000, "sk-test")
    with open(path) as f:
        assert "yardstick" in f.read()


def test_reset_codex_cli_removes_file_that_never_existed(fake_agents):
    assert not os.path.exists(fake_agents["codex-cli"])
    harness.point("codex-cli", 4000, "sk-test")
    assert os.path.exists(fake_agents["codex-cli"])

    harness.reset("codex-cli")
    assert not os.path.exists(fake_agents["codex-cli"])


def test_status_codex_cli_detects_pointed_via_substring(fake_agents):
    before = harness.status("codex-cli")
    assert before.pointed_at_proxy is False

    harness.point("codex-cli", 4000, "sk-test")
    after = harness.status("codex-cli")
    assert after.pointed_at_proxy is True
