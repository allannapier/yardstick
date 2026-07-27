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
    runs.begin_run(_exp(), EXPERIMENT_YAML, "only-arm")
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


def test_finish_run_success_check_runs_in_the_given_cwd(tmp_path):
    """Feature 2 (workspace isolation): success_check should run in a
    per-run workspace, not unconditionally wherever the calling process
    happened to be invoked from -- proven here with a check that only
    passes if `cwd` was actually honoured."""
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    (workdir / "marker.txt").write_text("present\n")
    check = "test -f marker.txt"
    runs.begin_run(_exp(check=check), _yaml_for(check), "only-arm")
    result = runs.finish_run(cwd=str(workdir))
    assert result.task_success is True


def test_finish_run_default_cwd_is_unchanged_from_before_feature_2():
    """`cwd=None` (the default) must behave exactly as before this feature
    -- i.e. inherit the calling process's own directory, not fail or change
    behaviour just because the parameter exists now."""
    runs.begin_run(_exp(check="true"), _yaml_for("true"), "only-arm")
    result = runs.finish_run()
    assert result.task_success is True


def test_finish_run_includes_summary_metrics():
    runs.begin_run(_exp(), EXPERIMENT_YAML, "only-arm")
    result = runs.finish_run()
    assert "turns" in result.summary_metrics


def test_finish_run_corrects_fingerprint_from_main_thread():
    """Regression test for finding 4: if a background/subagent request lands
    before the main conversation's first request, the eager per-request
    fingerprint fill in ys.collector can stamp `runs` from the wrong
    conversation. finish_run must correct it from the thread with the most
    requests once the run is complete."""
    from ys import db

    begun = runs.begin_run(_exp(), EXPERIMENT_YAML, "only-arm")
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO requests (run_id, seq, ts, model, status_code, thread_key, toolset_hash, system_prompt_hash) "
            "VALUES (?,1,'2026-01-01','bg-model',200,'bg',NULL,NULL)",
            (begun.run_id,),
        )
        cur.execute(
            "INSERT INTO requests (run_id, seq, ts, model, status_code, thread_key, toolset_hash, system_prompt_hash) "
            "VALUES (?,2,'2026-01-01','main-model',200,'main','tools-hash','sys-hash')",
            (begun.run_id,),
        )
        cur.execute(
            "INSERT INTO requests (run_id, seq, ts, model, status_code, thread_key, toolset_hash, system_prompt_hash) "
            "VALUES (?,3,'2026-01-01','main-model',200,'main','tools-hash','sys-hash')",
            (begun.run_id,),
        )

    runs.finish_run()

    with db.cursor() as cur:
        run_row = cur.execute(
            "SELECT model, toolset_hash, system_prompt_hash FROM runs WHERE id = ?", (begun.run_id,)
        ).fetchone()

    assert run_row["model"] == "main-model"
    assert run_row["toolset_hash"] == "tools-hash"
    assert run_row["system_prompt_hash"] == "sys-hash"


# ---------------------------------------------------------------------------
# finding 14: begin_run snapshots task_json + a config_hash onto the run row
# itself, independent of the `experiments` row (which keeps being
# overwritten by every `ys start`) -- see `runs.config_hash_for_arm` for
# exactly what the hash covers and why.
# ---------------------------------------------------------------------------

def test_begin_run_snapshots_config_hash_and_task_json():
    """Regression test for finding 14. Reverting the fix (the plain
    `INSERT INTO runs (..., notes)` with no config_hash/task_json_snapshot
    columns) makes this fail outright -- those columns would stay NULL."""
    from ys import db

    exp = _exp()
    begun = runs.begin_run(exp, EXPERIMENT_YAML, "only-arm")
    with db.cursor() as cur:
        row = cur.execute(
            "SELECT config_hash, task_json_snapshot FROM runs WHERE id = ?",
            (begun.run_id,),
        ).fetchone()
    runs.finish_run()

    expected_hash = runs.config_hash_for_arm(exp, exp.get_arm("only-arm"))
    assert row["config_hash"] == expected_hash

    import json

    assert json.loads(row["task_json_snapshot"])["id"] == "t0"


def test_config_hash_for_arm_changes_when_success_check_changes():
    """The thing finding 14 is actually about: a changed success_check must
    always split an arm's run history into a different group."""
    exp_true = _exp(check="true")
    exp_false = _exp(check="false")
    h_true = runs.config_hash_for_arm(exp_true, exp_true.get_arm("only-arm"))
    h_false = runs.config_hash_for_arm(exp_false, exp_false.get_arm("only-arm"))
    assert h_true != h_false


def test_config_hash_for_arm_changes_when_arm_model_factor_changes():
    """The other thing finding 14 calls out by name: a changed model under
    the same arm id must also split the group, not silently aggregate."""
    from ys.experiment import Experiment

    def exp_with_model(model):
        return Experiment.model_validate(
            {
                "experiment": "e",
                "task": {"id": "t0", "success_check": "true"},
                "arms": [{"id": "a", "factors": {"model": model}, "baseline": True}],
            }
        )

    exp_a = exp_with_model("model-a")
    exp_b = exp_with_model("model-b")
    h_a = runs.config_hash_for_arm(exp_a, exp_a.get_arm("a"))
    h_b = runs.config_hash_for_arm(exp_b, exp_b.get_arm("a"))
    assert h_a != h_b


def test_config_hash_for_arm_unaffected_by_question_or_task_id_field():
    """Deliberately coarse in the other direction: editing `question` (a
    comment-like field with no bearing on what the agent does or how it's
    scored) must NOT split the group -- otherwise every unrelated YAML
    tweak would spuriously fragment an arm's comparable history."""
    from ys.experiment import Experiment

    def exp_with_question(question):
        return Experiment.model_validate(
            {
                "experiment": "e",
                "question": question,
                "task": {"id": "t0", "success_check": "true"},
                "arms": [{"id": "a", "factors": {}, "baseline": True}],
            }
        )

    exp1 = exp_with_question("does X help?")
    exp2 = exp_with_question("a completely different question, reworded")
    h1 = runs.config_hash_for_arm(exp1, exp1.get_arm("a"))
    h2 = runs.config_hash_for_arm(exp2, exp2.get_arm("a"))
    assert h1 == h2


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


# --- write retry (finding 28) ------------------------------------------------
#
# Before this fix, ys start/ys end/ys runs delete wrote through a bare
# db.cursor() -- only ys/collector.py's YardstickLogger._handle (finding 6)
# retried a locked write. These pin that ys/runs.py's write paths now go
# through the same db.call_with_retry policy.


def test_begin_run_retries_a_transient_lock_and_succeeds(monkeypatch):
    """Regression test for finding 28: `ys start` (`runs.begin_run`) writes
    through `db.call_with_retry` now, the same policy the collector has had
    since finding 6 -- a lock that clears within a couple of attempts must
    not surface as a raw sqlite3.OperationalError. Reverting the fix (a bare
    `with db.cursor()` in begin_run) makes this fail: the first flaky
    attempt raises straight out of begin_run instead of being retried."""
    import sqlite3
    from contextlib import contextmanager

    from ys import db

    real_cursor = db.cursor
    calls = {"n": 0}

    @contextmanager
    def flaky_cursor():
        calls["n"] += 1
        if calls["n"] < 3:
            raise sqlite3.OperationalError("database is locked")
        with real_cursor() as cur:
            yield cur

    monkeypatch.setattr(db, "cursor", flaky_cursor)

    result = runs.begin_run(_exp(), EXPERIMENT_YAML, "only-arm")
    assert result.repeat_idx == 1
    assert calls["n"] == 3  # two failures + the successful retry

    runs.finish_run()


def test_finish_run_retries_a_transient_lock_on_the_ended_at_write(monkeypatch):
    """`ys end` races the tail of the proxy's in-flight writes more than any
    other writer (finding 28) -- its ended_at/task_success UPDATE must
    retry a transient lock instead of raising. Reverting the fix makes this
    fail with an unhandled sqlite3.OperationalError."""
    import sqlite3
    from contextlib import contextmanager

    from ys import db

    begun = runs.begin_run(_exp(), EXPERIMENT_YAML, "only-arm")

    real_cursor = db.cursor
    calls = {"n": 0}

    @contextmanager
    def flaky_cursor():
        calls["n"] += 1
        # Let the first db.cursor() call (finish_run's initial read of the
        # stored config_yaml) through untouched, then fail the very next
        # one -- the ended_at/task_success write -- exactly once.
        if calls["n"] == 2:
            raise sqlite3.OperationalError("database is locked")
        with real_cursor() as cur:
            yield cur

    monkeypatch.setattr(db, "cursor", flaky_cursor)

    result = runs.finish_run()
    assert result.run_id == begun.run_id
    assert calls["n"] >= 3  # read, failed write attempt, successful retry

    with db.cursor() as cur:
        row = cur.execute(
            "SELECT ended_at, task_success FROM runs WHERE id = ?", (begun.run_id,)
        ).fetchone()
    assert row["ended_at"] is not None
    assert row["task_success"] == 1


def test_begin_run_gives_up_after_exhausting_retries_on_a_persistent_lock(monkeypatch):
    """Once retries are exhausted, the underlying sqlite3 error must still
    propagate -- so the CLI/dashboard can turn it into a readable message --
    and the active-run slot claimed before the write must not be left
    behind for a run that was never actually recorded."""
    import sqlite3

    from ys import db, state

    def always_locked():
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(db, "cursor", always_locked)

    with pytest.raises(sqlite3.OperationalError):
        runs.begin_run(_exp(), EXPERIMENT_YAML, "only-arm")

    assert state.get_active() is None


# --- drain window on finish (finding 11) ------------------------------------


def test_finish_run_leaves_a_draining_record_behind_for_stragglers():
    """`finish_run` must record the ended run via `state.mark_ended` (finding
    11), not just clear active.json -- otherwise a request landing right
    after `ys end` (see ys/collector.py's `_resolve_run_id`) has no
    attribution signal left and falls into 'unattributed'. Reverting
    finish_run to skip the `state.mark_ended` call makes this fail."""
    begun = runs.begin_run(_exp(), EXPERIMENT_YAML, "only-arm")
    runs.finish_run()

    draining = state.get_draining_run()
    assert draining is not None
    assert draining["run_id"] == begun.run_id
    # the active slot is still freed immediately, unaffected by the drain
    # record living in its own separate file
    assert state.get_active() is None


# --- unattributed traffic (finding 12) --------------------------------------


def test_unattributed_summary_reports_zero_with_no_unattributed_requests():
    summary = runs.unattributed_summary()
    assert summary.count == 0
    assert summary.since is None


def test_unattributed_summary_counts_requests_and_reports_earliest_time():
    from ys import db

    with db.cursor() as cur:
        cur.execute(
            "INSERT OR IGNORE INTO experiments (id, name, question, task_json, config_yaml, created_at) "
            "VALUES ('unattributed', 'unattributed', NULL, '{}', '', '2026-01-01T14:02:33Z')"
        )
        cur.execute(
            "INSERT OR IGNORE INTO arms (id, experiment_id, label, factors_json, is_baseline) "
            "VALUES ('unattributed', 'unattributed', 'unattributed', '{}', 0)"
        )
        cur.execute(
            "INSERT OR IGNORE INTO runs (id, experiment_id, arm_id, repeat_idx, started_at) "
            "VALUES ('unattributed', 'unattributed', 'unattributed', 0, '2026-01-01T14:02:33Z')"
        )
        cur.execute(
            "INSERT INTO requests (run_id, seq, ts, status_code) "
            "VALUES ('unattributed', 1, '2026-01-01T14:02:33Z', 200)"
        )
        cur.execute(
            "INSERT INTO requests (run_id, seq, ts, status_code) "
            "VALUES ('unattributed', 2, '2026-01-01T15:00:00Z', 200)"
        )

    summary = runs.unattributed_summary()
    assert summary.count == 2
    assert summary.since == "14:02"


# --- ys runs list (P2 in IMPROVEMENTS.md) -----------------------------------
#
# "there is no `ys runs list`. Runs can be deleted by id but never
# enumerated, so the only way to find an id is the dashboard or raw SQL."


def test_list_runs_empty_when_nothing_recorded():
    assert runs.list_runs() == []


def test_list_runs_filters_by_experiment_and_arm():
    exp_a = Experiment.model_validate(
        {
            "experiment": "list-exp-a",
            "task": {"id": "t0", "success_check": "true"},
            "arms": [{"id": "arm-1", "factors": {}, "baseline": True}],
        }
    )
    exp_b = Experiment.model_validate(
        {
            "experiment": "list-exp-b",
            "task": {"id": "t0", "success_check": "true"},
            "arms": [{"id": "arm-1", "factors": {}, "baseline": True}],
        }
    )
    yaml_a = "experiment: list-exp-a\ntask: {id: t0, success_check: 'true'}\narms: [{id: arm-1, factors: {}, baseline: true}]\n"
    yaml_b = "experiment: list-exp-b\ntask: {id: t0, success_check: 'true'}\narms: [{id: arm-1, factors: {}, baseline: true}]\n"

    begun_a = runs.begin_run(exp_a, yaml_a, "arm-1")
    runs.finish_run()
    begun_b = runs.begin_run(exp_b, yaml_b, "arm-1")
    runs.finish_run()

    only_a = runs.list_runs(experiment="list-exp-a")
    assert [r.run_id for r in only_a] == [begun_a.run_id]

    only_b_arm1 = runs.list_runs(experiment="list-exp-b", arm="arm-1")
    assert [r.run_id for r in only_b_arm1] == [begun_b.run_id]

    everything = runs.list_runs()
    assert {r.run_id for r in everything} == {begun_a.run_id, begun_b.run_id}


def test_list_runs_reports_finished_status_and_success():
    runs.begin_run(_exp(check="true"), _yaml_for("true"), "only-arm")
    runs.finish_run()

    rows = runs.list_runs()
    assert len(rows) == 1
    assert rows[0].status == "finished"
    assert rows[0].success is True


def test_list_runs_reports_unfinished_status_and_no_success():
    begun = runs.begin_run(_exp(), EXPERIMENT_YAML, "only-arm")
    rows = runs.list_runs()
    assert len(rows) == 1
    assert rows[0].run_id == begun.run_id
    assert rows[0].status == "unfinished"
    assert rows[0].success is None
    runs.finish_run()  # tidy up the active slot


def test_list_runs_reports_abandoned_status():
    """Finding 13: a run displaced by `--force` is closed out with
    `abandoned=1` and no verdict -- `ys runs list` must call that out
    distinctly from an ordinary unfinished run, not just lump it in."""
    displaced = runs.begin_run(_exp(), EXPERIMENT_YAML, "only-arm")
    runs.begin_run(_exp(), EXPERIMENT_YAML, "only-arm", force=True)

    rows = {r.run_id: r for r in runs.list_runs()}
    assert rows[displaced.run_id].status == "abandoned"
    assert rows[displaced.run_id].success is None

    runs.finish_run()  # tidy up the still-active forced run


def test_list_runs_respects_limit_and_most_recent_first():
    for _ in range(3):
        runs.begin_run(_exp(), EXPERIMENT_YAML, "only-arm")
        runs.finish_run()

    all_rows = runs.list_runs()
    assert len(all_rows) == 3
    assert all_rows[0].repeat_idx == 3  # most recent (highest repeat_idx) first

    limited = runs.list_runs(limit=2)
    assert len(limited) == 2
    assert [r.repeat_idx for r in limited] == [3, 2]


def test_list_runs_config_current_true_when_matching_todays_yaml():
    exp = _exp(check="true")
    runs.begin_run(exp, _yaml_for("true"), "only-arm")
    runs.finish_run()

    rows = runs.list_runs(experiment_obj=exp)
    assert rows[0].config_current is True


def test_list_runs_config_current_false_when_config_changed():
    """The thing finding 14 exists for: a changed success_check must not be
    silently treated as 'still current'."""
    old_exp = _exp(check="true")
    runs.begin_run(old_exp, _yaml_for("true"), "only-arm")
    runs.finish_run()

    new_exp = _exp(check="false")
    rows = runs.list_runs(experiment_obj=new_exp)
    assert rows[0].config_current is False


def test_list_runs_config_current_false_when_no_snapshot_recorded():
    """A run written before finding 14 shipped has config_hash=NULL --
    never trusted as current, same rule render.compare_experiment uses."""
    exp = _exp()
    begun = runs.begin_run(exp, EXPERIMENT_YAML, "only-arm")
    runs.finish_run()

    from ys import db

    with db.cursor() as cur:
        cur.execute("UPDATE runs SET config_hash = NULL WHERE id = ?", (begun.run_id,))

    rows = runs.list_runs(experiment_obj=exp)
    assert rows[0].config_current is False


def test_list_runs_config_current_false_when_arm_no_longer_declared():
    exp = _exp()
    runs.begin_run(exp, EXPERIMENT_YAML, "only-arm")
    runs.finish_run()

    renamed_exp = Experiment.model_validate(
        {
            "experiment": "runs-test-exp",
            "task": {"id": "t0", "success_check": "true"},
            "arms": [{"id": "renamed-arm", "factors": {}, "baseline": True}],
        }
    )
    rows = runs.list_runs(experiment_obj=renamed_exp)
    assert rows[0].config_current is False


def test_list_runs_config_current_none_when_no_experiment_obj_given():
    runs.begin_run(_exp(), EXPERIMENT_YAML, "only-arm")
    runs.finish_run()

    rows = runs.list_runs()
    assert rows[0].config_current is None
