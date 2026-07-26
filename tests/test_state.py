import pytest

from ys import state


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


# --- drain window (finding 11) ----------------------------------------------


def test_get_draining_run_is_none_before_any_run_has_ended():
    assert state.get_draining_run() is None


def test_mark_ended_then_get_draining_run_returns_it_within_the_window():
    state.mark_ended("run1", "exp1", "arm1", "2026-01-01T00:01:00Z")
    draining = state.get_draining_run()
    assert draining is not None
    assert draining["run_id"] == "run1"
    assert draining["experiment"] == "exp1"
    assert draining["arm"] == "arm1"


def test_get_draining_run_returns_none_once_the_window_has_elapsed():
    """Regression test for finding 11's drain window actually expiring: a
    request arriving long after a run ended must not be misattributed to it
    forever. Reverting `get_draining_run`'s deadline check (e.g. always
    returning the record) makes this fail. Rewrites the persisted deadline
    into the past directly, rather than monkeypatching `time.time` (which
    `state.mark_ended`/`get_draining_run` would both observe, defeating the
    point of the test)."""
    import json
    import time

    from ys import paths

    state.mark_ended("run1", "exp1", "arm1", "2026-01-01T00:01:00Z")
    with open(paths.LAST_ENDED_RUN_PATH) as f:
        record = json.load(f)
    record["drain_until"] = time.time() - 1
    with open(paths.LAST_ENDED_RUN_PATH, "w") as f:
        json.dump(record, f)

    assert state.get_draining_run() is None


def test_get_draining_run_survives_a_corrupt_file():
    from ys import paths

    with open(paths.LAST_ENDED_RUN_PATH, "w") as f:
        f.write("not json")
    assert state.get_draining_run() is None
