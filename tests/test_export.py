import csv
import io
import json

from ys import db, export, runs
from ys.experiment import Experiment


def _make_experiment(arms):
    return Experiment.model_validate(
        {
            "experiment": "export-exp",
            "task": {"id": "t0", "success_check": "true"},
            "arms": arms,
        }
    )


def _seed_experiment_and_arm(cur, experiment, arm_id):
    exp_id = experiment.experiment
    row_id = runs.arm_row_id(exp_id, arm_id)
    cur.execute(
        "INSERT OR IGNORE INTO experiments (id, name, question, task_json, config_yaml, created_at) "
        "VALUES (?, ?, NULL, ?, '', '2026-01-01')",
        (exp_id, exp_id, db.dumps({"id": "t0"})),
    )
    cur.execute(
        "INSERT OR IGNORE INTO arms (id, experiment_id, label, factors_json, is_baseline) "
        "VALUES (?, ?, ?, '{}', 0)",
        (row_id, exp_id, row_id),
    )
    return row_id


def _seed_run(
    cur,
    experiment,
    arm_id,
    run_id,
    repeat_idx,
    task_success=1,
    abandoned=0,
    ended_at="2026-01-02",
    wall_clock_s=1.0,
    manual_score=None,
    model=None,
    config_hash="__current__",
    cost=0.05,
    cost_source="litellm",
):
    row_id = _seed_experiment_and_arm(cur, experiment, arm_id)
    if config_hash == "__current__":
        config_hash = runs.config_hash_for_arm(experiment, experiment.get_arm(arm_id))
    cur.execute(
        "INSERT INTO runs (id, experiment_id, arm_id, repeat_idx, started_at, ended_at, "
        "wall_clock_s, task_success, manual_score, model, config_hash, abandoned) "
        "VALUES (?, ?, ?, ?, '2026-01-01', ?, ?, ?, ?, ?, ?, ?)",
        (
            run_id,
            experiment.experiment,
            row_id,
            repeat_idx,
            ended_at,
            wall_clock_s,
            task_success,
            manual_score,
            model,
            config_hash,
            abandoned,
        ),
    )
    cur.execute(
        "INSERT INTO requests (run_id, seq, ts, model, input_tokens, cache_creation, "
        "cache_read, output_tokens, response_cost, system_tokens, tools_tokens, cost_source) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (run_id, 1, "2026-01-01", model, 10, 0, 0, 5, cost, 3, 2, cost_source),
    )


def test_export_rows_includes_identifying_context():
    """The whole point of an export is that a reader doesn't need the
    database beside the file -- experiment/arm/run id/config group/status
    must all be on the row itself."""
    db.init_db()
    exp = _make_experiment([{"id": "a", "factors": {}, "baseline": True}])
    with db.cursor() as cur:
        _seed_run(cur, exp, "a", "r1", 1)
        rows = export.export_rows(cur, exp)

    assert len(rows) == 1
    row = rows[0]
    assert row["experiment"] == "export-exp"
    assert row["arm"] == "a"
    assert row["run_id"] == "r1"
    assert row["config_matches_current"] is True
    assert row["status"] == "success"


def test_export_rows_status_covers_success_fail_unfinished_abandoned():
    """Finding 13: task_success IS NULL means "never scored", which is a
    different state from "failed" -- and abandoned (a --force-displaced
    run) is different again. An export that only showed a bare
    task_success column would blur all three."""
    db.init_db()
    exp = _make_experiment([{"id": "a", "factors": {}, "baseline": True}])
    with db.cursor() as cur:
        _seed_run(cur, exp, "a", "r-success", 1, task_success=1)
        _seed_run(cur, exp, "a", "r-fail", 2, task_success=0)
        _seed_run(cur, exp, "a", "r-unfinished", 3, task_success=None, ended_at=None, wall_clock_s=None)
        _seed_run(cur, exp, "a", "r-abandoned", 4, task_success=None, abandoned=1, ended_at="2026-01-02")
        rows = export.export_rows(cur, exp)

    status_by_run = {r["run_id"]: r["status"] for r in rows}
    assert status_by_run["r-success"] == "success"
    assert status_by_run["r-fail"] == "fail"
    assert status_by_run["r-unfinished"] == "unfinished"
    assert status_by_run["r-abandoned"] == "abandoned"


def test_export_rows_flags_config_mismatch():
    db.init_db()
    exp = _make_experiment([{"id": "a", "factors": {}, "baseline": True}])
    with db.cursor() as cur:
        _seed_run(cur, exp, "a", "r-current", 1, config_hash="__current__")
        _seed_run(cur, exp, "a", "r-stale", 2, config_hash="some-old-hash")
        rows = export.export_rows(cur, exp)

    matches_by_run = {r["run_id"]: r["config_matches_current"] for r in rows}
    assert matches_by_run["r-current"] is True
    assert matches_by_run["r-stale"] is False


def test_export_rows_flags_cost_unknown_from_cost_source():
    """Finding 9: cost_source='unknown' means neither LiteLLM nor a
    declared pricing override could price this run's traffic -- cost_usd
    reading as a confident 0 there would be the exact silent failure mode
    this rig's cost fix exists to prevent."""
    db.init_db()
    exp = _make_experiment([{"id": "a", "factors": {}, "baseline": True}])
    with db.cursor() as cur:
        _seed_run(cur, exp, "a", "r-priced", 1, cost=0.05, cost_source="litellm")
        _seed_run(cur, exp, "a", "r-unpriced", 2, cost=0.0, cost_source="unknown", model="mystery-model")
        rows = export.export_rows(cur, exp)

    unknown_by_run = {r["run_id"]: r["cost_unknown"] for r in rows}
    assert unknown_by_run["r-priced"] is False
    assert unknown_by_run["r-unpriced"] is True


def test_export_rows_can_be_limited_to_one_arm():
    db.init_db()
    exp = _make_experiment(
        [{"id": "a", "factors": {}, "baseline": True}, {"id": "b", "factors": {}}]
    )
    with db.cursor() as cur:
        _seed_run(cur, exp, "a", "ra1", 1)
        _seed_run(cur, exp, "b", "rb1", 1)
        rows = export.export_rows(cur, exp, arm_id="b")

    assert len(rows) == 1
    assert rows[0]["arm"] == "b"


def test_to_csv_quotes_a_value_containing_a_comma():
    """csv.DictWriter's QUOTE_MINIMAL default should quote a field
    containing the delimiter -- verified by parsing the output back with
    csv.reader rather than just eyeballing the raw text for a literal
    quote character."""
    db.init_db()
    exp = _make_experiment([{"id": "a", "factors": {}, "baseline": True}])
    with db.cursor() as cur:
        _seed_run(cur, exp, "a", "r1", 1, model="claude-x, custom-fork")
        rows = export.export_rows(cur, exp)

    csv_text = export.to_csv(rows)
    parsed = list(csv.DictReader(io.StringIO(csv_text)))
    assert len(parsed) == 1
    assert parsed[0]["model"] == "claude-x, custom-fork"
    # The raw text must actually be quoted, not just parse correctly by
    # accident -- a naive split on ',' on the unquoted text would produce
    # the wrong number of fields.
    assert '"claude-x, custom-fork"' in csv_text


def test_to_csv_writes_none_as_an_empty_cell_not_the_string_none():
    db.init_db()
    exp = _make_experiment([{"id": "a", "factors": {}, "baseline": True}])
    with db.cursor() as cur:
        _seed_run(cur, exp, "a", "r1", 1, task_success=None, ended_at=None, wall_clock_s=None, manual_score=None)
        rows = export.export_rows(cur, exp)

    csv_text = export.to_csv(rows)
    parsed = list(csv.DictReader(io.StringIO(csv_text)))
    assert parsed[0]["ended_at"] == ""
    assert parsed[0]["manual_score"] == ""
    assert "None" not in csv_text


def test_to_json_round_trips_none_as_null():
    db.init_db()
    exp = _make_experiment([{"id": "a", "factors": {}, "baseline": True}])
    with db.cursor() as cur:
        _seed_run(cur, exp, "a", "r1", 1, task_success=None, ended_at=None, wall_clock_s=None, manual_score=None)
        rows = export.export_rows(cur, exp)

    parsed = json.loads(export.to_json(rows))
    assert len(parsed) == 1
    assert parsed[0]["ended_at"] is None
    assert parsed[0]["manual_score"] is None
    assert parsed[0]["run_id"] == "r1"


def test_export_rows_empty_when_no_runs_recorded():
    db.init_db()
    exp = _make_experiment([{"id": "a", "factors": {}, "baseline": True}])
    with db.cursor() as cur:
        rows = export.export_rows(cur, exp)
    assert rows == []
    assert export.to_csv(rows).splitlines() == [",".join(export.COLUMNS)]
    assert json.loads(export.to_json(rows)) == []
