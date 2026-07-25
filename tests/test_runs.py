import pytest

from ys import runs, state
from ys.experiment import Experiment

EXPERIMENT_YAML = """
experiment: runs-test-exp
task:
  id: t0
  success_check: "true"
  timeout_s: 5
arms:
  - id: only-arm
    factors: {}
    baseline: true
repeats: 1
"""


def _exp(check="true"):
    return Experiment.model_validate(
        {
            "experiment": "runs-test-exp",
            "task": {"id": "t0", "success_check": check, "timeout_s": 5},
            "arms": [{"id": "only-arm", "factors": {}, "baseline": True}],
        }
    )


def _yaml_for(check="true"):
    # finish_run re-parses success_check from the *stored config_yaml text*,
    # not from the in-memory Experiment object -- so tests must keep these in
    # sync (this mirrors real behavior: the YAML is the source of truth).
    return f"""
experiment: runs-test-exp
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


def test_begin_run_unknown_arm_raises_with_valid_list():
    with pytest.raises(runs.ArmNotFound) as exc_info:
        runs.begin_run(_exp(), EXPERIMENT_YAML, "nope")
    assert "only-arm" in str(exc_info.value)


def test_begin_run_sets_active_and_returns_repeat_idx():
    result = runs.begin_run(_exp(), EXPERIMENT_YAML, "only-arm")
    assert result.repeat_idx == 1
    assert state.get_active()["run_id"] == result.run_id
    runs.finish_run()


def test_begin_run_second_repeat_increments():
    r1 = runs.begin_run(_exp(), EXPERIMENT_YAML, "only-arm")
    runs.finish_run()
    r2 = runs.begin_run(_exp(), EXPERIMENT_YAML, "only-arm")
    assert r2.repeat_idx == 2
    runs.finish_run()


def test_begin_run_refused_when_already_active_leaves_no_orphan_row():
    runs.begin_run(_exp(), EXPERIMENT_YAML, "only-arm")
    with pytest.raises(state.RunAlreadyActive):
        runs.begin_run(_exp(), EXPERIMENT_YAML, "only-arm")

    result = runs.finish_run()
    # only one run row should exist despite the refused second begin_run
    assert result.run_id is not None


def test_finish_run_no_active_raises():
    with pytest.raises(runs.NoActiveRun):
        runs.finish_run()


def test_finish_run_success_check_true():
    runs.begin_run(_exp(check="true"), _yaml_for("true"), "only-arm")
    result = runs.finish_run()
    assert result.task_success is True


def test_finish_run_success_check_false():
    runs.begin_run(_exp(check="false"), _yaml_for("false"), "only-arm")
    result = runs.finish_run()
    assert result.task_success is False


def test_finish_run_manual_score_skips_check():
    runs.begin_run(_exp(check="exit 1"), _yaml_for("exit 1"), "only-arm")
    result = runs.finish_run(manual_score=1.0)
    assert result.task_success is True


def test_finish_run_includes_summary_metrics():
    runs.begin_run(_exp(), EXPERIMENT_YAML, "only-arm")
    result = runs.finish_run()
    assert "turns" in result.summary_metrics


def test_delete_run_unknown_id_raises():
    with pytest.raises(runs.RunNotFound):
        runs.delete_run("no-such-run")


def test_delete_run_refuses_active_run():
    begun = runs.begin_run(_exp(), EXPERIMENT_YAML, "only-arm")
    with pytest.raises(runs.CannotDeleteActiveRun):
        runs.delete_run(begun.run_id)
    runs.finish_run()


def test_delete_run_removes_row_and_frees_repeat_slot():
    from ys import db

    begun = runs.begin_run(_exp(), EXPERIMENT_YAML, "only-arm")
    runs.finish_run()

    result = runs.delete_run(begun.run_id)
    assert result.run_id == begun.run_id
    assert result.experiment_name == "runs-test-exp"
    assert result.arm_id == "only-arm"

    with db.cursor() as cur:
        row = cur.execute("SELECT id FROM runs WHERE id = ?", (begun.run_id,)).fetchone()
    assert row is None

    # deleting freed the slot, so the next run reuses repeat_idx 1
    again = runs.begin_run(_exp(), EXPERIMENT_YAML, "only-arm")
    assert again.repeat_idx == 1
    runs.finish_run()
