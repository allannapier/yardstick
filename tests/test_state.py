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
