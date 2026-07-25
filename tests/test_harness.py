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
