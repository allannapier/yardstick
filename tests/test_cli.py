import re

from typer.testing import CliRunner

from ys import db
from ys.cli import app

runner = CliRunner()

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def plain(text: str) -> str:
    """Strip Rich's ANSI styling so substring assertions aren't broken by
    color codes landing mid-word (e.g. Rich highlighting 'only' inside
    'only-arm' splits the literal string with a reset code)."""
    return _ANSI.sub("", text)

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
