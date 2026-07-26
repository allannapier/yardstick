import json
import re

import pytest
from typer.testing import CliRunner

from ys import db, harness, proxy
from ys.cli import app

runner = CliRunner()

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def plain(text: str) -> str:
    """Strip Rich's ANSI styling so substring assertions aren't broken by
    color codes landing mid-word (e.g. Rich highlighting 'only' inside
    'only-arm' splits the literal string with a reset code)."""
    return _ANSI.sub("", text)


def unwrapped(text: str) -> str:
    """`plain` plus collapsing Rich's line-wrapping, for substring assertions
    against sentences long enough that Rich may wrap them at the terminal
    width mid-phrase."""
    return " ".join(plain(text).split())

EXPERIMENT_YAML = """
experiment: cli-test-exp
task:
  id: t0
  success_check: "{check}"
  timeout_s: 5
arms:
  - id: only-arm
    factors: {{}}
    baseline: true
repeats: 1
"""


def _write_exp(tmp_path, check="true"):
    path = tmp_path / "exp.yaml"
    path.write_text(EXPERIMENT_YAML.format(check=check))
    return str(path)


MODEL_EXPERIMENT_YAML = """
experiment: cli-model-exp
task:
  id: t0
  success_check: "true"
  timeout_s: 5
arms:
  - id: model-arm
    factors: {model: claude-sonnet-5}
    baseline: true
repeats: 1
"""


def _write_model_exp(tmp_path):
    path = tmp_path / "model_exp.yaml"
    path.write_text(MODEL_EXPERIMENT_YAML)
    return str(path)


@pytest.fixture
def fake_claude_agent(monkeypatch, tmp_path):
    claude_path = str(tmp_path / "claude_settings.json")
    monkeypatch.setitem(harness.AGENTS, "claude-code", harness.AgentSpec("claude-code", [claude_path]))
    monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-test")
    return claude_path


def test_init_creates_db():
    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0
    db.connect().close()  # would raise if init_db never ran


def test_status_reports_no_active_run_initially():
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "no active run" in result.stdout


def test_start_then_status_shows_active_run(tmp_path):
    exp = _write_exp(tmp_path)
    result = runner.invoke(app, ["start", "--exp", exp, "--arm", "only-arm"])
    assert result.exit_code == 0, result.stdout
    assert "only-arm" in plain(result.stdout)

    status = runner.invoke(app, ["status"])
    assert "only-arm" in plain(status.stdout)

    runner.invoke(app, ["end"])  # clean up active state for other tests


def test_start_unknown_arm_lists_valid_arms(tmp_path):
    exp = _write_exp(tmp_path)
    result = runner.invoke(app, ["start", "--exp", exp, "--arm", "nope"])
    assert result.exit_code != 0
    assert "only-arm" in result.stdout


def test_double_start_without_force_fails(tmp_path):
    exp = _write_exp(tmp_path)
    first = runner.invoke(app, ["start", "--exp", exp, "--arm", "only-arm"])
    assert first.exit_code == 0
    second = runner.invoke(app, ["start", "--exp", exp, "--arm", "only-arm"])
    assert second.exit_code != 0
    runner.invoke(app, ["end"])


def test_end_with_no_active_run_fails():
    result = runner.invoke(app, ["end"])
    assert result.exit_code != 0
    assert "no active run" in result.stdout


def test_full_lifecycle_success_check_true(tmp_path):
    exp = _write_exp(tmp_path, check="true")
    start = runner.invoke(app, ["start", "--exp", exp, "--arm", "only-arm"])
    assert start.exit_code == 0, start.stdout

    end = runner.invoke(app, ["end"])
    assert end.exit_code == 0, end.stdout
    assert "SUCCESS" in end.stdout

    status = runner.invoke(app, ["status"])
    assert "no active run" in status.stdout


def test_full_lifecycle_success_check_false(tmp_path):
    exp = _write_exp(tmp_path, check="false")
    runner.invoke(app, ["start", "--exp", exp, "--arm", "only-arm"])
    end = runner.invoke(app, ["end"])
    assert end.exit_code == 0
    assert "FAIL" in end.stdout


def test_manual_score_skips_success_check(tmp_path):
    exp = _write_exp(tmp_path, check="exit 1")  # would fail if actually run
    runner.invoke(app, ["start", "--exp", exp, "--arm", "only-arm"])
    end = runner.invoke(app, ["end", "--manual-score", "1"])
    assert end.exit_code == 0
    assert "SUCCESS" in end.stdout


def test_delete_unknown_run_fails():
    result = runner.invoke(app, ["delete", "no-such-run", "--yes"])
    assert result.exit_code != 0
    assert "no such run" in plain(result.stdout)


def test_delete_refuses_active_run(tmp_path):
    exp = _write_exp(tmp_path)
    runner.invoke(app, ["start", "--exp", exp, "--arm", "only-arm"])
    status = runner.invoke(app, ["status"])
    run_id = re.search(r'"run_id": "([^"]+)"', status.stdout).group(1)

    result = runner.invoke(app, ["delete", run_id, "--yes"])
    assert result.exit_code != 0
    assert "active" in plain(result.stdout)

    runner.invoke(app, ["end"])


def test_delete_removes_finished_run(tmp_path):
    exp = _write_exp(tmp_path)
    runner.invoke(app, ["start", "--exp", exp, "--arm", "only-arm"])
    end = runner.invoke(app, ["end"])
    run_id = re.search(r"run (\S+)", plain(end.stdout)).group(1)

    result = runner.invoke(app, ["delete", run_id, "--yes"])
    assert result.exit_code == 0, result.stdout
    assert run_id in plain(result.stdout)

    with db.cursor() as cur:
        row = cur.execute("SELECT id FROM runs WHERE id = ?", (run_id,)).fetchone()
    assert row is None


def test_delete_without_yes_prompts_and_can_be_declined(tmp_path):
    exp = _write_exp(tmp_path)
    runner.invoke(app, ["start", "--exp", exp, "--arm", "only-arm"])
    end = runner.invoke(app, ["end"])
    run_id = re.search(r"run (\S+)", plain(end.stdout)).group(1)

    result = runner.invoke(app, ["delete", run_id], input="n\n")
    assert result.exit_code == 0

    with db.cursor() as cur:
        row = cur.execute("SELECT id FROM runs WHERE id = ?", (run_id,)).fetchone()
    assert row is not None


def test_start_refuses_orphan_row_on_failed_claim(tmp_path):
    """If set_active refuses (already active), no run row should be left behind
    inflating the next repeat_idx."""
    exp = _write_exp(tmp_path)
    runner.invoke(app, ["start", "--exp", exp, "--arm", "only-arm"])
    blocked = runner.invoke(app, ["start", "--exp", exp, "--arm", "only-arm"])
    assert blocked.exit_code != 0

    runner.invoke(app, ["end"])

    with db.cursor() as cur:
        count = cur.execute(
            "SELECT COUNT(*) AS c FROM runs WHERE arm_id = 'cli-test-exp::only-arm'"
        ).fetchone()["c"]
    assert count == 1


def test_harness_point_requires_exp_and_arm_together(fake_claude_agent):
    result = runner.invoke(app, ["harness", "point", "claude-code", "--exp", "some.yaml"])
    assert result.exit_code != 0
    assert "--exp and --arm must be given together" in plain(result.stdout)


def test_harness_point_with_exp_arm_pins_model(tmp_path, fake_claude_agent):
    exp = _write_model_exp(tmp_path)
    result = runner.invoke(app, ["harness", "point", "claude-code", "--exp", exp, "--arm", "model-arm"])
    assert result.exit_code == 0, result.stdout
    assert "model=claude-sonnet-5" in plain(result.stdout)

    with open(fake_claude_agent) as f:
        config = json.load(f)
    assert config["env"]["ANTHROPIC_MODEL"] == "claude-sonnet-5"


def test_harness_point_without_exp_arm_warns(fake_claude_agent):
    result = runner.invoke(app, ["harness", "point", "claude-code"])
    assert result.exit_code == 0, result.stdout
    assert "no --exp/--arm given" in plain(result.stdout)


def test_end_warns_when_largest_thread_is_not_the_driving_conversation(tmp_path):
    """Regression test for finding 26: a Task subagent that issues more
    requests than the conversation that started the run becomes the
    "main thread" (finding 4's largest-thread rule, kept deliberately --
    see IMPROVEMENTS.md). `ys end` must surface that as a visible warning
    instead of silently reporting subagent-derived turns/compaction metrics
    as if they were the driving conversation's."""
    exp = _write_exp(tmp_path)
    start = runner.invoke(app, ["start", "--exp", exp, "--arm", "only-arm"])
    assert start.exit_code == 0, start.stdout
    run_id = re.search(r"run (\S+)", plain(start.stdout)).group(1)

    with db.cursor() as cur:
        # the driving conversation: 2 requests, holds seq=1
        for seq in (1, 2):
            cur.execute(
                "INSERT INTO requests (run_id, seq, ts, thread_key) VALUES (?,?,?,?)",
                (run_id, seq, "2026-01-01T00:00:00Z", "main"),
            )
        # a Task subagent that out-issues it
        for seq in (3, 4, 5):
            cur.execute(
                "INSERT INTO requests (run_id, seq, ts, thread_key) VALUES (?,?,?,?)",
                (run_id, seq, "2026-01-01T00:00:00Z", "subagent"),
            )

    end = runner.invoke(app, ["end"])
    assert end.exit_code == 0, end.stdout
    assert "doesn't contain its first request" in unwrapped(end.stdout)


def test_end_does_not_warn_when_largest_thread_holds_seq1(tmp_path):
    exp = _write_exp(tmp_path)
    start = runner.invoke(app, ["start", "--exp", exp, "--arm", "only-arm"])
    assert start.exit_code == 0, start.stdout
    run_id = re.search(r"run (\S+)", plain(start.stdout)).group(1)

    with db.cursor() as cur:
        for seq in (1, 2, 3):
            cur.execute(
                "INSERT INTO requests (run_id, seq, ts, thread_key) VALUES (?,?,?,?)",
                (run_id, seq, "2026-01-01T00:00:00Z", "main"),
            )

    end = runner.invoke(app, ["end"])
    assert end.exit_code == 0, end.stdout
    assert "doesn't contain its first request" not in unwrapped(end.stdout)


def test_start_warns_when_proxy_missing_arm_model(tmp_path, monkeypatch):
    monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-test")
    monkeypatch.setattr(proxy, "model_available", lambda model, port, key: False)

    exp = _write_model_exp(tmp_path)
    result = runner.invoke(app, ["start", "--exp", exp, "--arm", "model-arm"])
    assert result.exit_code == 0, result.stdout
    assert "no explicit entry for model 'claude-sonnet-5'" in unwrapped(result.stdout)

    runner.invoke(app, ["end"])


def test_start_warns_when_proxy_unreachable(tmp_path, monkeypatch):
    monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-test")
    monkeypatch.setattr(proxy, "model_available", lambda model, port, key: None)

    exp = _write_model_exp(tmp_path)
    result = runner.invoke(app, ["start", "--exp", exp, "--arm", "model-arm"])
    assert result.exit_code == 0, result.stdout
    assert "could not reach the proxy" in unwrapped(result.stdout)

    runner.invoke(app, ["end"])


def test_start_warns_when_master_key_unset_skips_model_check(tmp_path, monkeypatch):
    """Regression test for finding 29: without LITELLM_MASTER_KEY in `ys
    start`'s own environment (e.g. `ys proxy up` ran in a different shell),
    the model_available check added for finding 3 can't run at all -- it
    must say so instead of silently doing nothing, which left the user
    believing a verified proxy was serving their model."""
    monkeypatch.delenv("LITELLM_MASTER_KEY", raising=False)

    def _boom(model, port, key):
        raise AssertionError("model_available must not run without a master key")

    monkeypatch.setattr(proxy, "model_available", _boom)

    exp = _write_model_exp(tmp_path)
    result = runner.invoke(app, ["start", "--exp", exp, "--arm", "model-arm"])
    assert result.exit_code == 0, result.stdout
    assert "couldn't verify model 'claude-sonnet-5'" in unwrapped(result.stdout)
    assert "LITELLM_MASTER_KEY not set in this shell" in unwrapped(result.stdout)

    runner.invoke(app, ["end"])
