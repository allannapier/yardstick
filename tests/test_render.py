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


def _seed_run(cur, exp_id, arm_row_id, run_id, repeat_idx, cost, turns, task_success=1,
               model=None, cost_source=None):
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
            "INSERT INTO requests (run_id, seq, ts, model, input_tokens, cache_creation, cache_read, "
            "output_tokens, response_cost, system_tokens, tools_tokens, cost_source) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, seq, "2026-01-01", model, 10, 0, 0, 5, cost / turns, 3, 2, cost_source),
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


# ---------------------------------------------------------------------------
# finding 9: a request LiteLLM couldn't price (and no declared `pricing:`
# override could price either) must be flagged prominently in both the CLI
# table and the HTML report, not folded silently into cost_usd.
# ---------------------------------------------------------------------------

def test_compare_experiment_collects_unpriced_models_per_arm():
    db.init_db()
    exp = _make_experiment([{"id": "a", "factors": {}, "baseline": True}])
    with db.cursor() as cur:
        _seed_run(cur, "e1", "e1::a", "ra1", 1, 0.0, 2, model="claude-sonnet-5", cost_source="unknown")
        comparison = render.compare_experiment(cur, exp)

    assert comparison.arms[0].unpriced_models == [{"model": "claude-sonnet-5", "count": 2}]


def test_compare_experiment_no_unpriced_models_when_all_priced():
    db.init_db()
    exp = _make_experiment([{"id": "a", "factors": {}, "baseline": True}])
    with db.cursor() as cur:
        _seed_run(cur, "e1", "e1::a", "ra1", 1, 0.01, 2, model="claude-sonnet-5", cost_source="litellm")
        comparison = render.compare_experiment(cur, exp)

    assert comparison.arms[0].unpriced_models == []


def test_cost_warnings_names_the_model_arm_and_count():
    db.init_db()
    exp = _make_experiment([{"id": "a", "factors": {}, "baseline": True}])
    with db.cursor() as cur:
        _seed_run(cur, "e1", "e1::a", "ra1", 1, 0.0, 3, model="claude-sonnet-5", cost_source="unknown")
        comparison = render.compare_experiment(cur, exp)

    warnings = render.cost_warnings(comparison)
    assert len(warnings) == 1
    assert "claude-sonnet-5" in warnings[0]
    assert "'a'" in warnings[0]
    assert "3 request(s)" in warnings[0]


def test_cost_warnings_empty_when_nothing_unpriced():
    db.init_db()
    exp = _make_experiment([{"id": "a", "factors": {}, "baseline": True}])
    with db.cursor() as cur:
        _seed_run(cur, "e1", "e1::a", "ra1", 1, 0.01, 2, model="claude-sonnet-5", cost_source="litellm")
        comparison = render.compare_experiment(cur, exp)

    assert render.cost_warnings(comparison) == []


def test_build_table_marks_cost_cells_and_header_for_unpriced_arm():
    db.init_db()
    exp = _make_experiment([{"id": "a", "factors": {}, "baseline": True}])
    with db.cursor() as cur:
        _seed_run(cur, "e1", "e1::a", "ra1", 1, 0.0, 2, model="claude-sonnet-5", cost_source="unknown")
        comparison = render.compare_experiment(cur, exp)

    table = render.build_table(comparison)
    from io import StringIO

    from rich.console import Console

    buf = StringIO()
    Console(file=buf, width=200).print(table)
    rendered = buf.getvalue()

    assert "COST UNKNOWN" in rendered  # header marker, precedent: UNCONTROLLED


def test_render_html_shows_cost_unavailable_banner_and_markers():
    db.init_db()
    exp = _make_experiment([{"id": "a", "factors": {}, "baseline": True}])
    with db.cursor() as cur:
        _seed_run(cur, "e1", "e1::a", "ra1", 1, 0.0, 2, model="claude-sonnet-5", cost_source="unknown")
        comparison = render.compare_experiment(cur, exp)
        content = render.render_html(comparison, cur)

    assert "Cost unavailable" in content
    assert "claude-sonnet-5" in content
    assert "COST UNKNOWN" in content
    assert content.count("<tr>") == content.count("</tr>")


def test_billable_weights_declared_on_experiment_flow_into_comparison():
    """finding 10 end to end through compare_experiment: a declared
    `billable_weights` override changes the aggregated billable_tokens, not
    just token_metrics in isolation."""
    db.init_db()
    exp = Experiment.model_validate(
        {
            "experiment": "e1",
            "task": {"id": "t0", "success_check": "true"},
            "arms": [{"id": "a", "factors": {}, "baseline": True}],
            "billable_weights": {"claude-sonnet-5": {"input": 10.0}},
        }
    )
    with db.cursor() as cur:
        _seed_run(cur, "e1", "e1::a", "ra1", 1, 0.02, 2, model="claude-sonnet-5")
        comparison = render.compare_experiment(cur, exp)

    # 2 requests, 10 input tokens each @ weight 10.0 (declared) + 5 output
    # tokens each @ weight 1.0 (undeclared -- Anthropic-shaped default):
    # (10*10 + 5*1) * 2 = 210.0
    assert comparison.arms[0].aggregate["metrics"]["billable_tokens"]["mean"] == pytest.approx(210.0)
