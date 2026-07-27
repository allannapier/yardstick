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


PROMPT_FILE_EXPERIMENT_YAML = """
experiment: cli-prompt-file-exp
task:
  id: t0
  success_check: "true"
  timeout_s: 5
  prompt_file: {prompt_file}
arms:
  - id: only-arm
    factors: {{}}
    baseline: true
repeats: 1
"""


def _write_exp_with_prompt_file(tmp_path, prompt_file):
    path = tmp_path / "prompt_file_exp.yaml"
    path.write_text(PROMPT_FILE_EXPERIMENT_YAML.format(prompt_file=prompt_file))
    return str(path)


@pytest.fixture
def fake_claude_agent(monkeypatch, tmp_path):
    """Isolates every agent harness.py knows about, not just claude-code --
    `ys end`'s automatic reset (feature 5) walks every entry in
    harness.AGENTS, so a test exercising it must never let that walk fall
    through to a real ~/.config/opencode or ~/.codex/config.toml."""
    claude_path = str(tmp_path / "claude_settings.json")
    monkeypatch.setitem(harness.AGENTS, "claude-code", harness.AgentSpec("claude-code", [claude_path]))
    monkeypatch.setitem(
        harness.AGENTS, "opencode", harness.AgentSpec("opencode", [str(tmp_path / "opencode.jsonc")])
    )
    monkeypatch.setitem(
        harness.AGENTS, "codex-cli", harness.AgentSpec("codex-cli", [str(tmp_path / "codex_config.toml")])
    )
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


# --- unattributed traffic surfaced in status/end (finding 12) --------------


def _insert_unattributed_request(ts="2026-01-01T14:02:33Z"):
    with db.cursor() as cur:
        cur.execute(
            "INSERT OR IGNORE INTO experiments (id, name, question, task_json, config_yaml, created_at) "
            "VALUES ('unattributed', 'unattributed', NULL, '{}', '', ?)",
            (ts,),
        )
        cur.execute(
            "INSERT OR IGNORE INTO arms (id, experiment_id, label, factors_json, is_baseline) "
            "VALUES ('unattributed', 'unattributed', 'unattributed', '{}', 0)"
        )
        cur.execute(
            "INSERT OR IGNORE INTO runs (id, experiment_id, arm_id, repeat_idx, started_at) "
            "VALUES ('unattributed', 'unattributed', 'unattributed', 0, ?)",
            (ts,),
        )
        cur.execute(
            "INSERT INTO requests (run_id, seq, ts, status_code) VALUES ('unattributed', 1, ?, 200)",
            (ts,),
        )


def test_status_reports_unattributed_requests():
    """Regression test for finding 12: before this, a request that landed in
    the synthetic 'unattributed' run was invisible in every CLI command --
    a misconfigured harness produced a run with zero requests and no
    explanation anywhere. Reverting the `_print_unattributed_notice` call in
    `ys status` makes this fail."""
    _insert_unattributed_request()
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "1 request(s) since 14:02" in unwrapped(result.stdout)
    assert "could not be attributed to a run" in unwrapped(result.stdout)


def test_status_silent_about_unattributed_when_there_are_none():
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "could not be attributed" not in result.stdout


def test_end_reports_unattributed_requests(tmp_path):
    exp = _write_exp(tmp_path)
    runner.invoke(app, ["start", "--exp", exp, "--arm", "only-arm"])
    _insert_unattributed_request()

    result = runner.invoke(app, ["end"])
    assert result.exit_code == 0, result.stdout
    assert "1 request(s) since 14:02" in unwrapped(result.stdout)
    assert "could not be attributed to a run" in unwrapped(result.stdout)


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


def test_start_fails_loudly_when_prompt_file_missing(tmp_path):
    """Finding 15-18: task.prompt_file is a declared-but-unconsumed hook for
    feature 1 (unattended runs) -- but a typo'd path must fail loudly at
    `ys start`, not silently once the feature that eventually reads it
    ships. Must not leave an active run behind either."""
    missing = str(tmp_path / "does-not-exist.txt")
    exp = _write_exp_with_prompt_file(tmp_path, missing)
    result = runner.invoke(app, ["start", "--exp", exp, "--arm", "only-arm"])
    assert result.exit_code != 0
    assert "does not exist" in plain(result.stdout)

    # the rejected start must not have claimed the active-run slot
    status = runner.invoke(app, ["status"])
    assert "no active run" in plain(status.stdout)


def test_start_succeeds_when_prompt_file_exists(tmp_path):
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("do the task")
    exp = _write_exp_with_prompt_file(tmp_path, str(prompt_file))
    result = runner.invoke(app, ["start", "--exp", exp, "--arm", "only-arm"])
    assert result.exit_code == 0, result.stdout
    runner.invoke(app, ["end"])


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


def test_harness_point_no_pin_background_leaves_small_fast_model_unset(tmp_path, fake_claude_agent):
    """Regression test for finding 27's CLI wiring: --no-pin-background
    must reach ys.harness.point and actually leave the background model
    env vars unset, and warn that it's doing so."""
    exp = _write_model_exp(tmp_path)
    result = runner.invoke(
        app,
        ["harness", "point", "claude-code", "--exp", exp, "--arm", "model-arm", "--no-pin-background"],
    )
    assert result.exit_code == 0, result.stdout
    assert "--no-pin-background" in unwrapped(result.stdout)

    with open(fake_claude_agent) as f:
        config = json.load(f)
    assert config["env"]["ANTHROPIC_MODEL"] == "claude-sonnet-5"
    assert "ANTHROPIC_SMALL_FAST_MODEL" not in config["env"]
    assert "ANTHROPIC_DEFAULT_HAIKU_MODEL" not in config["env"]


def test_harness_point_pins_background_by_default(tmp_path, fake_claude_agent):
    exp = _write_model_exp(tmp_path)
    result = runner.invoke(app, ["harness", "point", "claude-code", "--exp", exp, "--arm", "model-arm"])
    assert result.exit_code == 0, result.stdout

    with open(fake_claude_agent) as f:
        config = json.load(f)
    assert config["env"]["ANTHROPIC_SMALL_FAST_MODEL"] == "claude-sonnet-5"


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


def test_compare_prints_cost_unknown_warning_for_unpriced_model(tmp_path):
    """Regression test for finding 9: `ys compare` must surface a request
    LiteLLM couldn't price (cost_source='unknown') as a visible warning,
    not fold it silently into cost_usd."""
    exp = _write_exp(tmp_path)
    start = runner.invoke(app, ["start", "--exp", exp, "--arm", "only-arm"])
    assert start.exit_code == 0, start.stdout
    run_id = re.search(r"run (\S+)", plain(start.stdout)).group(1)

    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO requests (run_id, seq, ts, model, input_tokens, output_tokens, "
            "response_cost, cost_source) VALUES (?,1,?,?,?,?,?,?)",
            (run_id, "2026-01-01T00:00:00Z", "claude-sonnet-5", 100, 20, 0.0, "unknown"),
        )

    runner.invoke(app, ["end"])

    result = runner.invoke(app, ["compare", "--exp", exp])
    assert result.exit_code == 0, result.stdout
    output = unwrapped(result.stdout)
    assert "cost unavailable for model 'claude-sonnet-5'" in output
    assert "COST UNKNOWN" in output


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


# --- feature 5: --env-only never touches the config file --------------------


def test_harness_point_env_only_never_writes_the_config_file(tmp_path, fake_claude_agent):
    """The CLI-level regression test for feature 5's harness-safety half:
    `ys harness point claude-code --env-only` must print export statements
    and never create/modify the (fake, isolated) claude-code settings file
    at all. Reverting --env-only's wiring in cli.py -- falling through to
    the normal harness.point() call -- makes this fail by writing the file."""
    exp = _write_model_exp(tmp_path)
    assert not __import__("os").path.exists(fake_claude_agent)

    result = runner.invoke(
        app,
        ["harness", "point", "claude-code", "--exp", exp, "--arm", "model-arm", "--env-only"],
    )
    assert result.exit_code == 0, result.stdout
    assert "export ANTHROPIC_BASE_URL=" in plain(result.stdout)
    assert "export ANTHROPIC_MODEL=claude-sonnet-5" in plain(result.stdout)
    assert not __import__("os").path.exists(fake_claude_agent)


def test_harness_point_env_only_unsupported_agent_reports_error(fake_claude_agent):
    result = runner.invoke(app, ["harness", "point", "opencode", "--env-only"])
    assert result.exit_code != 0
    assert "not supported for opencode" in plain(result.stdout) or "opencode" in plain(result.stdout)


# --- feature 5: automatic harness reset on `ys end` -------------------------


def test_end_automatically_resets_a_pointed_harness(tmp_path, fake_claude_agent):
    """Regression test for feature 5's other harness-safety half: before
    this, nothing ever reset a harness `ys harness point` had pointed --
    only a manual `ys harness reset` did, and a crash (or simply forgetting)
    left the plaintext API key sitting in the (fake, isolated) settings file
    indefinitely. `ys end` must reset it automatically by default. Reverting
    the `_auto_reset_pointed_harnesses()` call in cli.py's `end()` makes this
    fail: the file would still report ANTHROPIC_BASE_URL pointed at the
    proxy after `ys end` returns."""
    exp = _write_model_exp(tmp_path)
    runner.invoke(app, ["harness", "point", "claude-code", "--exp", exp, "--arm", "model-arm"])
    with open(fake_claude_agent) as f:
        assert json.load(f)["env"]["ANTHROPIC_BASE_URL"]  # sanity: really pointed

    task_exp = _write_exp(tmp_path)
    runner.invoke(app, ["start", "--exp", task_exp, "--arm", "only-arm"])
    end = runner.invoke(app, ["end"])
    assert end.exit_code == 0, end.stdout

    assert not __import__("os").path.exists(fake_claude_agent)  # restored: never existed before


def test_end_keep_harness_pointed_leaves_it_pointed(tmp_path, fake_claude_agent):
    """--keep-harness-pointed is the opt-out for a multi-repeat workflow --
    `ys end` must leave the harness's config exactly as `ys harness point`
    left it."""
    exp = _write_model_exp(tmp_path)
    runner.invoke(app, ["harness", "point", "claude-code", "--exp", exp, "--arm", "model-arm"])

    task_exp = _write_exp(tmp_path)
    runner.invoke(app, ["start", "--exp", task_exp, "--arm", "only-arm"])
    end = runner.invoke(app, ["end", "--keep-harness-pointed"])
    assert end.exit_code == 0, end.stdout
    assert "--keep-harness-pointed" in unwrapped(end.stdout)

    with open(fake_claude_agent) as f:
        config = json.load(f)
    assert config["env"]["ANTHROPIC_BASE_URL"] == "http://localhost:4000"

    runner.invoke(app, ["harness", "reset", "claude-code"])  # clean up for other tests


def test_end_reset_harness_is_a_no_op_when_nothing_was_pointed(tmp_path, fake_claude_agent):
    """`ys end` must not error out (or print anything alarming) when no
    harness was ever pointed -- the common case for most of this test file's
    other runs."""
    exp = _write_exp(tmp_path)
    runner.invoke(app, ["start", "--exp", exp, "--arm", "only-arm"])
    end = runner.invoke(app, ["end"])
    assert end.exit_code == 0, end.stdout
