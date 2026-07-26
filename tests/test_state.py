import pytest

from ys import db, state


def test_no_active_run_initially():
    assert state.get_active() is None


def test_set_and_get_active():
    state.set_active("run1", "exp1", "arm1", "2026-01-01T00:00:00Z")
    active = state.get_active()
    assert active == {
        "run_id": "run1",
        "experiment": "exp1",
        "arm": "arm1",
        "started_at": "2026-01-01T00:00:00Z",
    }


def test_double_start_refused_without_force():
    state.set_active("run1", "exp1", "arm1", "2026-01-01T00:00:00Z")
    with pytest.raises(state.RunAlreadyActive):
        state.set_active("run2", "exp1", "arm2", "2026-01-01T00:01:00Z")
    # original run is untouched
    assert state.get_active()["run_id"] == "run1"


def test_force_overrides_active_run():
    state.set_active("run1", "exp1", "arm1", "2026-01-01T00:00:00Z")
    state.set_active("run2", "exp1", "arm2", "2026-01-01T00:01:00Z", force=True)
    assert state.get_active()["run_id"] == "run2"


def test_clear_active():
    state.set_active("run1", "exp1", "arm1", "2026-01-01T00:00:00Z")
    state.clear_active()
    assert state.get_active() is None
    # clearing twice must not raise
    state.clear_active()


# ---------------------------------------------------------------------------
# finding 13: `--force` must close out the run it displaces instead of
# leaving it dangling forever with ended_at/task_success/wall_clock_s NULL.
# ---------------------------------------------------------------------------

def _seed_run_row(run_id="run1", started_at="2026-01-01T00:00:00Z"):
    with db.cursor() as cur:
        cur.execute(
            "INSERT OR IGNORE INTO experiments (id, name, question, task_json, config_yaml, created_at) "
            "VALUES ('exp1','exp1',NULL,'{}','','2026-01-01T00:00:00Z')"
        )
        cur.execute(
            "INSERT OR IGNORE INTO arms (id, experiment_id, label, factors_json, is_baseline) "
            "VALUES ('arm1','exp1','arm1','{}',0)"
        )
        cur.execute(
            "INSERT INTO runs (id, experiment_id, arm_id, repeat_idx, started_at) "
            "VALUES (?, 'exp1', 'arm1', 1, ?)",
            (run_id, started_at),
        )


def test_force_marks_the_displaced_run_abandoned_and_closes_it_out():
    """Regression test for finding 13: before the fix, `--force` only ever
    overwrote the active-run file -- the run row it displaced kept
    ended_at/task_success/wall_clock_s NULL forever. Reverting the fix (a
    plain overwrite with no `_abandon_displaced_run` call) makes this fail:
    the row stays untouched (ended_at/wall_clock_s/abandoned all still
    NULL/0)."""
    _seed_run_row("run1", "2026-01-01T00:00:00Z")

    state.set_active("run1", "exp1", "arm1", "2026-01-01T00:00:00Z")
    state.set_active("run2", "exp1", "arm1", "2026-01-01T00:05:00Z", force=True)

    with db.cursor() as cur:
        row = cur.execute(
            "SELECT ended_at, task_success, wall_clock_s, abandoned FROM runs WHERE id = 'run1'"
        ).fetchone()

    assert row["abandoned"] == 1
    assert row["ended_at"] is not None
    # never actually scored -- abandoned is not the same thing as failed.
    assert row["task_success"] is None
    # `_abandon_displaced_run` stamps `ended_at` with the real wall-clock
    # time (like `runs.finish_run` does), not the synthetic `started_at`
    # this test seeded -- so just check it's a real, non-negative number,
    # not an exact value.
    assert row["wall_clock_s"] is not None
    assert row["wall_clock_s"] >= 0


def test_force_does_not_touch_a_run_that_was_already_closed():
    """A displaced run that (somehow) already has ended_at set -- e.g. `ys
    end` won a race against the force -- must not be re-stamped abandoned;
    the guard is `WHERE ended_at IS NULL`."""
    _seed_run_row("run1", "2026-01-01T00:00:00Z")
    with db.cursor() as cur:
        cur.execute(
            "UPDATE runs SET ended_at='2026-01-01T00:02:00Z', task_success=1, wall_clock_s=120.0 "
            "WHERE id='run1'"
        )

    state.set_active("run1", "exp1", "arm1", "2026-01-01T00:00:00Z")
    state.set_active("run2", "exp1", "arm1", "2026-01-01T00:05:00Z", force=True)

    with db.cursor() as cur:
        row = cur.execute(
            "SELECT ended_at, task_success, wall_clock_s, abandoned FROM runs WHERE id = 'run1'"
        ).fetchone()

    assert row["ended_at"] == "2026-01-01T00:02:00Z"
    assert row["task_success"] == 1
    assert row["wall_clock_s"] == pytest.approx(120.0)
    assert row["abandoned"] == 0
