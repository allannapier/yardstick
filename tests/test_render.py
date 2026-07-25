import pytest

from ys import db, render
from ys.experiment import Experiment


def _make_experiment(arms):
    return Experiment.model_validate(
        {
            "experiment": "e1",
            "task": {"id": "t0", "success_check": "true"},
            "arms": arms,
        }
    )


def _seed_run(cur, exp_id, arm_row_id, run_id, repeat_idx, cost, turns, task_success=1):
    cur.execute(
        "INSERT OR IGNORE INTO experiments (id, name, question, task_json, config_yaml, created_at) "
        "VALUES (?, ?, NULL, ?, '', '2026-01-01')",
        (exp_id, exp_id, db.dumps({"id": "t0"})),
    )
    cur.execute(
        "INSERT OR IGNORE INTO arms (id, experiment_id, label, factors_json, is_baseline) "
        "VALUES (?, ?, ?, '{}', 0)",
        (arm_row_id, exp_id, arm_row_id),
    )
    cur.execute(
        "INSERT INTO runs (id, experiment_id, arm_id, repeat_idx, started_at, ended_at, wall_clock_s, task_success, model) "
        "VALUES (?, ?, ?, ?, '2026-01-01', '2026-01-01', 1.0, ?, 'test-model')",
        (run_id, exp_id, arm_row_id, repeat_idx, task_success),
    )
    for seq in range(1, turns + 1):
        cur.execute(
            "INSERT INTO requests (run_id, seq, ts, input_tokens, cache_creation, cache_read, output_tokens, "
            "response_cost, system_tokens, tools_tokens) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (run_id, seq, "2026-01-01", 10, 0, 0, 5, cost / turns, 3, 2),
        )


def test_compare_experiment_orders_baseline_first():
    db.init_db()
    exp = _make_experiment(
        [
            {"id": "a", "factors": {}, "baseline": False},
            {"id": "b", "factors": {}, "baseline": True},
        ]
    )
    with db.cursor() as cur:
        _seed_run(cur, "e1", "e1::a", "ra1", 1, 0.01, 2)
        _seed_run(cur, "e1", "e1::b", "rb1", 1, 0.02, 3)

        comparison = render.compare_experiment(cur, exp)

    assert comparison.arms[0].label == "b"
    assert comparison.arms[0].is_baseline is True
    assert comparison.arms[1].label == "a"


def test_compare_experiment_raises_when_no_runs():
    db.init_db()
    exp = _make_experiment([{"id": "a", "factors": {}}])
    with db.cursor() as cur:
        with pytest.raises(render.CompareError):
            render.compare_experiment(cur, exp)


def test_compare_experiment_refuses_mismatched_task_id():
    db.init_db()
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO experiments (id, name, question, task_json, config_yaml, created_at) "
            "VALUES ('e1','e1',NULL,?,'','2026-01-01')",
            (db.dumps({"id": "old-task"}),),
        )
        _seed_run(cur, "e1", "e1::a", "ra1", 1, 0.01, 1)

        exp = _make_experiment([{"id": "a", "factors": {}}])
        with pytest.raises(render.CompareError):
            render.compare_experiment(cur, exp)


def test_fingerprint_drift_detected_across_repeats():
    db.init_db()
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO experiments (id, name, question, task_json, config_yaml, created_at) "
            "VALUES ('e1','e1',NULL,'{}','','2026-01-01')"
        )
        cur.execute(
            "INSERT INTO arms (id, experiment_id, label, factors_json, is_baseline) "
            "VALUES ('e1::a','e1','a','{}',0)"
        )
        cur.execute(
            "INSERT INTO runs (id, experiment_id, arm_id, repeat_idx, started_at, model, toolset_hash) "
            "VALUES ('r1','e1','e1::a',1,'2026-01-01','model-A','hashA')"
        )
        cur.execute(
            "INSERT INTO runs (id, experiment_id, arm_id, repeat_idx, started_at, model, toolset_hash) "
            "VALUES ('r2','e1','e1::a',2,'2026-01-01','model-B','hashA')"
        )
        assert render._fingerprint_drifted(cur, ["r1", "r2"]) is True


def test_fingerprint_no_drift_when_consistent():
    db.init_db()
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO experiments (id, name, question, task_json, config_yaml, created_at) "
            "VALUES ('e1','e1',NULL,'{}','','2026-01-01')"
        )
        cur.execute(
            "INSERT INTO arms (id, experiment_id, label, factors_json, is_baseline) "
            "VALUES ('e1::a','e1','a','{}',0)"
        )
        cur.execute(
            "INSERT INTO runs (id, experiment_id, arm_id, repeat_idx, started_at, model, toolset_hash) "
            "VALUES ('r1','e1','e1::a',1,'2026-01-01','model-A','hashA')"
        )
        cur.execute(
            "INSERT INTO runs (id, experiment_id, arm_id, repeat_idx, started_at, model, toolset_hash) "
            "VALUES ('r2','e1','e1::a',2,'2026-01-01','model-A','hashA')"
        )
        assert render._fingerprint_drifted(cur, ["r1", "r2"]) is False


def test_render_html_is_well_formed():
    db.init_db()
    exp = _make_experiment([{"id": "a", "factors": {}, "baseline": True}])
    with db.cursor() as cur:
        _seed_run(cur, "e1", "e1::a", "ra1", 1, 0.01, 2)
        comparison = render.compare_experiment(cur, exp)
        content = render.render_html(comparison, cur)

    assert "<table>" in content
    assert "e1" in content
    assert content.count("<tr>") == content.count("</tr>")
