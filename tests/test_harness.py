import json

import pytest

from ys import harness


@pytest.fixture(autouse=True)
def fake_agents(monkeypatch, tmp_path):
    """Never touch the real ~/.claude or ~/.config/opencode during tests."""
    claude_path = str(tmp_path / "claude_settings.json")
    opencode_path = str(tmp_path / "opencode.jsonc")
    monkeypatch.setitem(harness.AGENTS, "claude-code", harness.AgentSpec("claude-code", [claude_path]))
    monkeypatch.setitem(harness.AGENTS, "opencode", harness.AgentSpec("opencode", [opencode_path]))
    return {"claude-code": claude_path, "opencode": opencode_path}


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
