import os
import re
import sqlite3
from contextlib import contextmanager
from urllib.parse import unquote

import pytest
from fastapi.testclient import TestClient

from ys import db, state
from ys.web import store
from ys.web.app import app

client = TestClient(app)


def _create_experiment(name: str):
    client.post(
        "/experiments",
        data={
            "name": name,
            "task_id": "t0",
            "success_check": "true",
            "timeout_s": "60",
            "repeats": "1",
            "model_key": ["m1"],
            "model_kind": ["mock"],
            "model_value": ["hi"],
            "arm_id": ["only-arm"],
            "arm_model": ["m1"],
            "arm_seq": ["0"],
            "arm_baseline": ["0"],
            "arm_notes": [""],
        },
    )


def _factor_exp_payload(name="factor-exp", **overrides):
    """A one-arm mock experiment whose arm carries an extra, non-`model`
    factor (`harness=claude-code`) -- the shape item 6's arbitrary-factor
    fields (factor_arm_seq/factor_key/factor_value) need to round-trip
    through create/edit/re-render-on-error."""
    payload = {
        "name": name,
        "task_id": "t0",
        "success_check": "true",
        "timeout_s": "60",
        "repeats": "1",
        "model_key": ["m1"],
        "model_kind": ["mock"],
        "model_value": ["hi"],
        "arm_id": ["a"],
        "arm_model": ["m1"],
        "arm_seq": ["0"],
        "arm_baseline": ["0"],
        "arm_notes": ["a note"],
        "factor_arm_seq": ["0"],
        "factor_key": ["harness"],
        "factor_value": ["claude-code"],
    }
    payload.update(overrides)
    return payload


def test_health():
    assert client.get("/health").text == "ok"


def test_index_empty(monkeypatch, tmp_path):
    # Isolate cwd from wherever pytest happens to be invoked from -- since
    # store.discovery_dirs() now also looks at ./experiments relative to
    # the process cwd (see the discovery tests below), running this suite
    # from this repo's own root would otherwise make the "no experiments
    # yet" empty state unreachable (this repo's experiments/ directory
    # would always be discovered).
    monkeypatch.chdir(tmp_path)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "no experiments yet" in resp.text


def test_create_experiment_via_form():
    resp = client.post(
        "/experiments",
        data={
            "name": "web-test-exp",
            "question": "does the dashboard work?",
            "task_id": "t0",
            "success_check": "true",
            "timeout_s": "60",
            "repeats": "2",
            "model_key": ["probe-mock"],
            "model_kind": ["mock"],
            "model_value": ["hello"],
            "arm_id": ["arm-a"],
            "arm_model": ["probe-mock"],
            "arm_seq": ["0"],
            "arm_baseline": ["0"],
            "arm_notes": [""],
        },
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert resp.headers["location"].startswith("/experiments/web-test-exp")

    detail = client.get("/experiments/web-test-exp")
    assert detail.status_code == 200
    assert "arm-a" in detail.text
    assert "baseline" in detail.text


def test_create_experiment_rejects_duplicate_arm_ids():
    # Defect 22: a pydantic validation failure used to redirect to
    # /experiments/new?error=<raw pydantic text>, discarding everything the
    # user typed. It should instead re-render the form (so the name, task
    # fields etc. the user already typed are still there) with a readable
    # message, and not a redirect at all.
    resp = client.post(
        "/experiments",
        data={
            "name": "bad-exp",
            "task_id": "t0",
            "success_check": "true",
            "timeout_s": "60",
            "repeats": "1",
            "model_key": ["m1"],
            "model_kind": ["mock"],
            "model_value": ["hi"],
            "arm_id": ["dup", "dup"],
            "arm_model": ["m1", "m1"],
            "arm_seq": ["0", "1"],
            "arm_baseline": [],
            "arm_notes": ["", ""],
        },
        follow_redirects=False,
    )
    assert resp.status_code == 400
    assert "duplicate arm ids" in resp.text
    # the raw pydantic dump ("1 validation error for Experiment", a link to
    # errors.pydantic.dev) must not leak into the page
    assert "validation error for Experiment" not in resp.text
    # the user's other input is preserved, not discarded by a redirect
    assert 'value="bad-exp"' in resp.text
    assert 'value="t0"' in resp.text


def test_create_experiment_non_numeric_timeout_reprompts_without_500():
    # Defect 20: int(form.get("timeout_s")) on a non-numeric value used to
    # raise straight through to a 500. It should become a field-level error
    # and re-render the form with everything else intact instead.
    resp = client.post(
        "/experiments",
        data={
            "name": "bad-timeout-exp",
            "question": "keep me",
            "task_id": "t0",
            "success_check": "true",
            "timeout_s": "soon",
            "repeats": "2",
            "model_key": ["m1"],
            "model_kind": ["mock"],
            "model_value": ["hi"],
            "arm_id": ["a"],
            "arm_model": ["m1"],
            "arm_seq": ["0"],
            "arm_baseline": [],
            "arm_notes": [""],
        },
        follow_redirects=False,
    )
    assert resp.status_code == 400
    assert "must be a whole number" in resp.text
    assert 'value="bad-timeout-exp"' in resp.text
    assert not os.path.exists(store.experiment_path("bad-timeout-exp"))


def test_create_experiment_non_numeric_repeats_reprompts_without_500():
    resp = client.post(
        "/experiments",
        data={
            "name": "bad-repeats-exp",
            "task_id": "t0",
            "success_check": "true",
            "timeout_s": "60",
            "repeats": "lots",
            "model_key": ["m1"],
            "model_kind": ["mock"],
            "model_value": ["hi"],
            "arm_id": ["a"],
            "arm_model": ["m1"],
            "arm_seq": ["0"],
            "arm_baseline": [],
            "arm_notes": [""],
        },
        follow_redirects=False,
    )
    assert resp.status_code == 400
    assert "must be a whole number" in resp.text


def test_experiment_detail_404_for_unknown_name():
    resp = client.get("/experiments/does-not-exist", follow_redirects=False)
    assert resp.status_code == 404
    assert "no experiment named" in resp.text


def test_experiment_detail_invalid_name_returns_404_not_500():
    # Defect 19: GET /experiments/foo.bar raised store.InvalidExperimentName
    # straight through to a 500 -- no route caught it.
    resp = client.get("/experiments/foo.bar", follow_redirects=False)
    assert resp.status_code == 404
    assert "not a valid experiment name" in resp.text


def test_start_proxy_for_nonexistent_experiment_returns_404():
    # Defect 23: starting the proxy for a nonexistent experiment had no
    # existence check at all -- it surfaced whatever the proxy layer
    # complained about first (e.g. a missing LITELLM_MASTER_KEY) and
    # redirected to a detail page that then 500s on the same call site as
    # defect 19.
    resp = client.post("/experiments/does-not-exist-proxy/proxy/up", follow_redirects=False)
    assert resp.status_code == 404


def test_create_experiment_refuses_to_silently_overwrite_existing():
    # Defect 21: re-submitting the form with a name that already has a
    # YAML file silently overwrote it -- verified in the review by changing
    # task.id on an experiment that already had runs recorded against it;
    # those runs stay attached to the same experiment id and get
    # aggregated with whatever the new definition produces. Refuse by
    # default; only an explicit confirm_overwrite=on gets through.
    payload = {
        "name": "overwrite-exp",
        "task_id": "t0",
        "success_check": "true",
        "timeout_s": "60",
        "repeats": "1",
        "model_key": ["m1"],
        "model_kind": ["mock"],
        "model_value": ["hi"],
        "arm_id": ["a"],
        "arm_model": ["m1"],
        "arm_seq": ["0"],
        "arm_baseline": ["0"],
        "arm_notes": [""],
    }
    created = client.post("/experiments", data=payload, follow_redirects=False)
    assert created.status_code == 303
    original = store.read_raw("overwrite-exp")

    changed = dict(payload)
    changed["task_id"] = "different-task"
    resp = client.post("/experiments", data=changed, follow_redirects=False)
    assert resp.status_code == 400
    assert "already exists" in resp.text
    assert store.read_raw("overwrite-exp") == original  # not clobbered

    changed["confirm_overwrite"] = "on"
    resp2 = client.post("/experiments", data=changed, follow_redirects=False)
    assert resp2.status_code == 303
    assert "different-task" in store.read_raw("overwrite-exp")


def test_new_experiment_form_uses_radio_buttons_for_baseline():
    # Defect 24: the baseline control was a checkbox with a JS-synchronised
    # `value`, so two arms could both be checked -- radios sharing one
    # `name` make that unreachable natively.
    resp = client.get("/experiments/new")
    assert resp.status_code == 200
    assert 'type="radio"' in resp.text
    assert 'name="arm_baseline"' in resp.text
    assert 'type="checkbox" name="arm_baseline"' not in resp.text


def test_two_submitted_baselines_cannot_create_two_baseline_arms():
    # Even a client that bypasses the radio group in the browser (e.g. a
    # raw POST) can't create two baseline arms: the server keys off a
    # single scalar `arm_baseline` value, not a set of checked values.
    resp = client.post(
        "/experiments",
        data={
            "name": "two-baseline-exp",
            "task_id": "t0",
            "success_check": "true",
            "timeout_s": "60",
            "repeats": "1",
            "model_key": ["m1"],
            "model_kind": ["mock"],
            "model_value": ["hi"],
            "arm_id": ["a", "b"],
            "arm_model": ["m1", "m1"],
            "arm_seq": ["0", "1"],
            "arm_baseline": ["0", "1"],
            "arm_notes": ["", ""],
        },
        follow_redirects=False,
    )
    assert resp.status_code == 303
    detail = client.get("/experiments/two-baseline-exp")
    assert detail.text.count("baseline</span>") <= 1


def test_no_nested_interactive_elements_in_index_and_experiment_pages():
    # Invalid HTML bullet: <a href="..."><button></button></a> nests two
    # interactive elements, which is invalid and breaks keyboard/AT
    # behavior. Neither page should do it any more.
    index_html = client.get("/").text
    assert re.search(r"<a[^>]*>\s*<button", index_html) is None

    client.post(
        "/experiments",
        data={
            "name": "html-check-exp",
            "task_id": "t0",
            "success_check": "true",
            "timeout_s": "60",
            "repeats": "1",
            "model_key": ["m1"],
            "model_kind": ["mock"],
            "model_value": ["hi"],
            "arm_id": ["a"],
            "arm_model": ["m1"],
            "arm_seq": ["0"],
            "arm_baseline": ["0"],
            "arm_notes": [""],
        },
    )
    detail_html = client.get("/experiments/html-check-exp").text
    assert re.search(r"<a[^>]*>\s*<button", detail_html) is None


def test_full_run_lifecycle_via_web():
    client.post(
        "/experiments",
        data={
            "name": "lifecycle-exp",
            "task_id": "t0",
            "success_check": "true",
            "timeout_s": "60",
            "repeats": "1",
            "model_key": ["m1"],
            "model_kind": ["mock"],
            "model_value": ["hi"],
            "arm_id": ["only-arm"],
            "arm_model": ["m1"],
            "arm_seq": ["0"],
            "arm_baseline": ["0"],
            "arm_notes": [""],
        },
    )

    start = client.post(
        "/experiments/lifecycle-exp/runs/start",
        data={"arm_id": "only-arm"},
        follow_redirects=False,
    )
    assert start.status_code == 303
    assert "error=" not in start.headers["location"]
    assert state.get_active() is not None
    assert state.get_active()["experiment"] == "lifecycle-exp"

    # a second start while one is active should be refused, not silently
    # override -- and must not leave an orphan run row
    blocked = client.post(
        "/experiments/lifecycle-exp/runs/start",
        data={"arm_id": "only-arm"},
        follow_redirects=False,
    )
    assert "error=" in blocked.headers["location"]

    end = client.post("/runs/end", data={}, follow_redirects=False)
    assert end.status_code == 303
    assert state.get_active() is None

    with db.cursor() as cur:
        count = cur.execute(
            "SELECT COUNT(*) AS c FROM runs WHERE arm_id = 'lifecycle-exp::only-arm'"
        ).fetchone()["c"]
    assert count == 1

    detail = client.get("/experiments/lifecycle-exp")
    assert "success" in detail.text.lower()

    run_id_row = client.get("/experiments/lifecycle-exp")
    assert run_id_row.status_code == 200


def test_compare_view_after_a_run():
    client.post(
        "/experiments",
        data={
            "name": "compare-exp",
            "task_id": "t0",
            "success_check": "true",
            "timeout_s": "60",
            "repeats": "1",
            "model_key": ["m1"],
            "model_kind": ["mock"],
            "model_value": ["hi"],
            "arm_id": ["only-arm"],
            "arm_model": ["m1"],
            "arm_seq": ["0"],
            "arm_baseline": ["0"],
            "arm_notes": [""],
        },
    )
    client.post("/experiments/compare-exp/runs/start", data={"arm_id": "only-arm"})
    client.post("/runs/end", data={})

    resp = client.get("/experiments/compare-exp/compare")
    assert resp.status_code == 200
    assert "compare-exp" in resp.text
    assert "<table>" in resp.text


def test_compare_before_any_run_redirects_with_error():
    client.post(
        "/experiments",
        data={
            "name": "no-runs-exp",
            "task_id": "t0",
            "success_check": "true",
            "timeout_s": "60",
            "repeats": "1",
            "model_key": ["m1"],
            "model_kind": ["mock"],
            "model_value": ["hi"],
            "arm_id": ["only-arm"],
            "arm_model": ["m1"],
            "arm_seq": ["0"],
            "arm_baseline": ["0"],
            "arm_notes": [""],
        },
    )
    resp = client.get("/experiments/no-runs-exp/compare", follow_redirects=False)
    assert resp.status_code == 303
    assert "error=" in resp.headers["location"]


def test_delete_run_removes_it_and_redirects_to_experiment():
    client.post(
        "/experiments",
        data={
            "name": "delete-exp",
            "task_id": "t0",
            "success_check": "true",
            "timeout_s": "60",
            "repeats": "1",
            "model_key": ["m1"],
            "model_kind": ["mock"],
            "model_value": ["hi"],
            "arm_id": ["only-arm"],
            "arm_model": ["m1"],
            "arm_seq": ["0"],
            "arm_baseline": ["0"],
            "arm_notes": [""],
        },
    )
    client.post("/experiments/delete-exp/runs/start", data={"arm_id": "only-arm"})
    with db.cursor() as cur:
        run_id = cur.execute(
            "SELECT id FROM runs WHERE arm_id = 'delete-exp::only-arm'"
        ).fetchone()["id"]
    client.post("/runs/end", data={})

    resp = client.post(f"/runs/{run_id}/delete", follow_redirects=False)
    assert resp.status_code == 303
    assert resp.headers["location"].startswith("/experiments/delete-exp")
    assert "error=" not in resp.headers["location"]

    with db.cursor() as cur:
        row = cur.execute("SELECT id FROM runs WHERE id = ?", (run_id,)).fetchone()
    assert row is None

    detail = client.get("/experiments/delete-exp")
    assert run_id not in detail.text


def test_delete_unknown_run_redirects_home_with_error():
    resp = client.post("/runs/no-such-run/delete", follow_redirects=False)
    assert resp.status_code == 303
    assert resp.headers["location"].startswith("/?")
    assert "error=" in resp.headers["location"]


def test_delete_active_run_is_refused():
    client.post(
        "/experiments",
        data={
            "name": "delete-active-exp",
            "task_id": "t0",
            "success_check": "true",
            "timeout_s": "60",
            "repeats": "1",
            "model_key": ["m1"],
            "model_kind": ["mock"],
            "model_value": ["hi"],
            "arm_id": ["only-arm"],
            "arm_model": ["m1"],
            "arm_seq": ["0"],
            "arm_baseline": ["0"],
            "arm_notes": [""],
        },
    )
    client.post("/experiments/delete-active-exp/runs/start", data={"arm_id": "only-arm"})
    run_id = state.get_active()["run_id"]

    resp = client.post(f"/runs/{run_id}/delete", follow_redirects=False)
    assert resp.status_code == 303
    assert "error=" in resp.headers["location"]

    with db.cursor() as cur:
        row = cur.execute("SELECT id FROM runs WHERE id = ?", (run_id,)).fetchone()
    assert row is not None

    client.post("/runs/end", data={})  # clean up active state for other tests


def test_run_detail_page():
    client.post(
        "/experiments",
        data={
            "name": "detail-exp",
            "task_id": "t0",
            "success_check": "true",
            "timeout_s": "60",
            "repeats": "1",
            "model_key": ["m1"],
            "model_kind": ["mock"],
            "model_value": ["hi"],
            "arm_id": ["only-arm"],
            "arm_model": ["m1"],
            "arm_seq": ["0"],
            "arm_baseline": ["0"],
            "arm_notes": [""],
        },
    )
    client.post("/experiments/detail-exp/runs/start", data={"arm_id": "only-arm"})
    with db.cursor() as cur:
        run_id = cur.execute(
            "SELECT id FROM runs WHERE arm_id = 'detail-exp::only-arm'"
        ).fetchone()["id"]
    client.post("/runs/end", data={})

    resp = client.get(f"/runs/{run_id}")
    assert resp.status_code == 200
    assert run_id in resp.text


# --- write retry (finding 28) ------------------------------------------------


def test_start_run_survives_contention_that_previously_raised(monkeypatch):
    """Regression test for finding 28: before this fix, every dashboard
    write went through a bare db.cursor() -- only the collector retried a
    locked write. `runs.begin_run` now writes through `db.call_with_retry`,
    so a lock that clears within a couple of attempts must not surface as a
    500. Reverting the fix makes this 500 instead of redirecting cleanly."""
    _create_experiment("retry-web-exp")

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

    resp = client.post(
        "/experiments/retry-web-exp/runs/start",
        data={"arm_id": "only-arm"},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert "error=" not in resp.headers["location"]
    assert calls["n"] == 3
    assert state.get_active() is not None

    client.post("/runs/end", data={})


def test_start_run_reports_a_readable_error_after_exhausting_retries(monkeypatch):
    """Once retries are exhausted the write still fails, but the dashboard
    must redirect with a plain-English error instead of a raw 500
    traceback from an unhandled sqlite3.OperationalError."""
    _create_experiment("stuck-web-exp")

    def always_locked():
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(db, "cursor", always_locked)

    resp = client.post(
        "/experiments/stuck-web-exp/runs/start",
        data={"arm_id": "only-arm"},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert "error=" in resp.headers["location"]
    assert "could not write to the database" in unquote(resp.headers["location"])
    assert state.get_active() is None


# ---------------------------------------------------------------------------
# Experiment discovery beyond EXPERIMENTS_DIR (dashboard bullet: "it can't
# see the repo's own experiments")
# ---------------------------------------------------------------------------


def test_index_discovers_experiments_outside_experiments_dir(monkeypatch, tmp_path):
    # store.list_experiments() used to only read EXPERIMENTS_DIR, so
    # experiments/example.yaml -- the file the README's quick start points
    # at -- never appeared. Reproduce that shape without depending on this
    # repo's actual experiments/ directory: a fake cwd with its own
    # experiments/ folder, and a filename that deliberately doesn't match
    # the experiment id inside (mirroring example.yaml's filename
    # "example.yaml" vs its `experiment:` field "mock-smoke-01").
    exp_dir = tmp_path / "experiments"
    exp_dir.mkdir()
    (exp_dir / "some-file.yaml").write_text(
        "experiment: discovered-exp\n"
        'task:\n  id: t0\n  success_check: "true"\n'
        "models:\n  m1: {mock_response: hi}\n"
        "arms:\n  - id: a\n    factors: {model: m1}\n    baseline: true\n"
    )
    monkeypatch.chdir(tmp_path)

    resp = client.get("/")
    assert resp.status_code == 200
    assert "discovered-exp" in resp.text
    assert "read-only" in resp.text  # not dashboard-managed

    detail = client.get("/experiments/discovered-exp")
    assert detail.status_code == 200
    assert "outside the dashboard's own" in detail.text

    # discovery is read-only: the dashboard's own writable location is untouched
    assert not os.path.exists(store.experiment_path("discovered-exp"))


def test_dashboard_managed_experiment_wins_name_collision_over_discovered(monkeypatch, tmp_path):
    exp_dir = tmp_path / "experiments"
    exp_dir.mkdir()
    (exp_dir / "collide.yaml").write_text(
        "experiment: collide-exp\n"
        'task:\n  id: from-disk\n  success_check: "true"\n'
        "models:\n  m1: {mock_response: hi}\n"
        "arms:\n  - id: a\n    factors: {model: m1}\n    baseline: true\n"
    )
    monkeypatch.chdir(tmp_path)

    client.post("/experiments", data=_factor_exp_payload(name="collide-exp", task_id="from-dashboard"))

    detail = client.get("/experiments/collide-exp")
    assert "from-dashboard" in detail.text
    assert "from-disk" not in detail.text


def test_find_experiment_still_rejects_invalid_names_across_discovery_dirs():
    # Widening discovery to more than one directory must not widen what a
    # URL-supplied name can resolve to -- validate_name still runs first
    # regardless of how many directories find_experiment searches.
    with pytest.raises(store.InvalidExperimentName):
        store.find_experiment("../escape")


# ---------------------------------------------------------------------------
# View YAML / edit / delete an experiment definition (dashboard bullet:
# "no edit, no YAML view, no delete")
# ---------------------------------------------------------------------------


def test_view_yaml_route_shows_raw_definition():
    client.post("/experiments", data=_factor_exp_payload())
    resp = client.get("/experiments/factor-exp/yaml")
    assert resp.status_code == 200
    assert "success_check" in resp.text
    assert "harness" in resp.text
    assert "claude-code" in resp.text


def test_view_yaml_works_for_a_discovered_read_only_experiment(monkeypatch, tmp_path):
    exp_dir = tmp_path / "experiments"
    exp_dir.mkdir()
    (exp_dir / "ro.yaml").write_text(
        "experiment: readonly-yaml-exp\n"
        'task:\n  id: t0\n  success_check: "true"\n'
        "models:\n  m1: {mock_response: hi}\n"
        "arms:\n  - id: a\n    factors: {model: m1}\n    baseline: true\n"
    )
    monkeypatch.chdir(tmp_path)

    resp = client.get("/experiments/readonly-yaml-exp/yaml")
    assert resp.status_code == 200
    assert "readonly-yaml-exp" in resp.text


def test_edit_form_is_prefilled_with_current_definition():
    client.post("/experiments", data=_factor_exp_payload())
    resp = client.get("/experiments/factor-exp/edit")
    assert resp.status_code == 200
    assert 'value="t0"' in resp.text
    assert 'value="harness"' in resp.text
    assert 'value="claude-code"' in resp.text
    assert "readonly" in resp.text  # the name field can't be changed via edit


def test_edit_saves_changes_without_needing_confirm_overwrite():
    client.post("/experiments", data=_factor_exp_payload())
    changed = _factor_exp_payload(success_check="echo changed")
    resp = client.post("/experiments/factor-exp/edit", data=changed, follow_redirects=False)
    assert resp.status_code == 303
    assert resp.headers["location"].startswith("/experiments/factor-exp")
    assert "changed" in store.read_raw("factor-exp")


def test_edit_reports_field_errors_without_discarding_input():
    client.post("/experiments", data=_factor_exp_payload())
    bad = _factor_exp_payload(timeout_s="soon")
    resp = client.post("/experiments/factor-exp/edit", data=bad, follow_redirects=False)
    assert resp.status_code == 400
    assert "must be a whole number" in resp.text
    assert 'value="t0"' in resp.text  # other fields preserved, not discarded


def test_experiment_detail_names_run_count_in_delete_confirmation():
    # Mirrors defect 21's house style (IMPROVEMENTS.md): name the run count
    # in the confirmation so the consequence is visible before the user
    # commits to deleting the definition.
    client.post("/experiments", data=_factor_exp_payload())
    client.post("/experiments/factor-exp/runs/start", data={"arm_id": "a"})
    client.post("/runs/end", data={})

    detail = client.get("/experiments/factor-exp")
    assert "1 recorded run(s)" in detail.text


def test_delete_experiment_removes_definition_but_keeps_runs():
    client.post("/experiments", data=_factor_exp_payload())
    client.post("/experiments/factor-exp/runs/start", data={"arm_id": "a"})
    with db.cursor() as cur:
        run_id = cur.execute(
            "SELECT id FROM runs WHERE arm_id = 'factor-exp::a'"
        ).fetchone()["id"]
    client.post("/runs/end", data={})

    resp = client.post("/experiments/factor-exp/delete", follow_redirects=False)
    assert resp.status_code == 303
    assert resp.headers["location"].startswith("/?")
    assert not os.path.exists(store.experiment_path("factor-exp"))
    assert client.get("/experiments/factor-exp").status_code == 404

    with db.cursor() as cur:
        row = cur.execute("SELECT id FROM runs WHERE id = ?", (run_id,)).fetchone()
    assert row is not None  # runs are not deleted, just orphaned


def test_edit_and_delete_are_refused_for_discovered_non_managed_experiments(monkeypatch, tmp_path):
    exp_dir = tmp_path / "experiments"
    exp_dir.mkdir()
    (exp_dir / "ro.yaml").write_text(
        "experiment: readonly-exp\n"
        'task:\n  id: t0\n  success_check: "true"\n'
        "models:\n  m1: {mock_response: hi}\n"
        "arms:\n  - id: a\n    factors: {model: m1}\n    baseline: true\n"
    )
    monkeypatch.chdir(tmp_path)

    edit_resp = client.get("/experiments/readonly-exp/edit")
    assert edit_resp.status_code == 400
    # `_error_page`'s message renders through Jinja's auto-escaping (unlike
    # experiment.html's own static copy), so the apostrophe comes back as
    # `&#39;` -- assert on a stretch of the message that doesn't include one.
    assert "outside the dashboard" in edit_resp.text

    delete_resp = client.post("/experiments/readonly-exp/delete", follow_redirects=False)
    assert delete_resp.status_code == 303
    assert "error=" in delete_resp.headers["location"]
    assert (exp_dir / "ro.yaml").exists()


# ---------------------------------------------------------------------------
# Live updates during an active run (dashboard bullet: "nothing updates
# during a live run")
# ---------------------------------------------------------------------------


def test_run_live_endpoint_reports_running_totals():
    client.post("/experiments", data=_factor_exp_payload(name="live-exp"))
    client.post("/experiments/live-exp/runs/start", data={"arm_id": "a"})
    run_id = state.get_active()["run_id"]

    live = client.get(f"/runs/{run_id}/live")
    assert live.status_code == 200
    body = live.json()
    assert body["ended"] is False
    assert body["request_count"] == 0

    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO requests (run_id, seq, ts, model, input_tokens, response_cost, status_code) "
            "VALUES (?, 1, '2026-01-01T00:00:00Z', 'm1', 100, 0.01, 200)",
            (run_id,),
        )

    live2 = client.get(f"/runs/{run_id}/live").json()
    assert live2["request_count"] == 1
    assert live2["cost_usd"] == pytest.approx(0.01)

    client.post("/runs/end", data={})
    live3 = client.get(f"/runs/{run_id}/live").json()
    assert live3["ended"] is True


def test_run_live_unknown_run_returns_404():
    resp = client.get("/runs/no-such-run/live")
    assert resp.status_code == 404


def test_active_run_banner_includes_live_polling_markup():
    client.post("/experiments", data=_factor_exp_payload(name="live-banner-exp"))
    client.post("/experiments/live-banner-exp/runs/start", data={"arm_id": "a"})

    resp = client.get("/")
    assert "live-request-count" in resp.text
    assert "/live" in resp.text

    client.post("/runs/end", data={})  # clean up active-run state


# ---------------------------------------------------------------------------
# The comparison view stays inside the app shell (dashboard bullet: "the
# comparison view escapes the app")
# ---------------------------------------------------------------------------


def test_compare_view_is_embedded_in_app_shell_with_navigation():
    client.post("/experiments", data=_factor_exp_payload(name="shell-compare-exp"))
    client.post("/experiments/shell-compare-exp/runs/start", data={"arm_id": "a"})
    client.post("/runs/end", data={})

    resp = client.get("/experiments/shell-compare-exp/compare")
    assert resp.status_code == 200
    # the standalone report used to be served raw with no way back to the
    # app -- it should now carry the same header/nav every other page has.
    assert '<a href="/">yardstick</a>' in resp.text
    assert '/experiments/shell-compare-exp"' in resp.text
    assert "back to experiment" in resp.text


# ---------------------------------------------------------------------------
# Run detail: notes, factors, success_output, per-turn chart (dashboard
# bullet: "run detail omits the useful parts")
# ---------------------------------------------------------------------------


def test_run_detail_shows_notes_factors_success_output_and_chart():
    client.post(
        "/experiments",
        data=_factor_exp_payload(
            name="rich-detail-exp",
            success_check="echo detail-output-marker",
            arm_notes=["a distinctive note"],
        ),
    )
    client.post("/experiments/rich-detail-exp/runs/start", data={"arm_id": "a"})
    with db.cursor() as cur:
        run_id = cur.execute(
            "SELECT id FROM runs WHERE arm_id = 'rich-detail-exp::a'"
        ).fetchone()["id"]
    client.post("/runs/end", data={})

    resp = client.get(f"/runs/{run_id}")
    assert resp.status_code == 200
    assert "a distinctive note" in resp.text
    assert "harness=claude-code" in resp.text
    assert "detail-output-marker" in resp.text
    assert "<svg" not in resp.text  # fewer than 2 requests recorded -- no chart yet

    with db.cursor() as cur:
        for seq, tokens in [(1, 100), (2, 500)]:
            cur.execute(
                "INSERT INTO requests (run_id, seq, ts, model, input_tokens, status_code) "
                "VALUES (?, ?, '2026-01-01T00:00:00Z', 'm1', ?, 200)",
                (run_id, seq, tokens),
            )

    resp2 = client.get(f"/runs/{run_id}")
    assert "<svg" in resp2.text


# ---------------------------------------------------------------------------
# Arbitrary per-arm factors, not just `model` (dashboard bullet: "only the
# model factor is expressible")
# ---------------------------------------------------------------------------


def test_arbitrary_arm_factors_are_saved_to_the_yaml():
    resp = client.post("/experiments", data=_factor_exp_payload(), follow_redirects=False)
    assert resp.status_code == 303
    raw = store.read_raw("factor-exp")
    assert "harness: claude-code" in raw


def test_arbitrary_arm_factors_round_trip_through_a_validation_failure():
    bad = _factor_exp_payload(name="factor-roundtrip-exp", timeout_s="not-a-number")
    resp = client.post("/experiments", data=bad, follow_redirects=False)
    assert resp.status_code == 400
    # the extra factor row must survive the re-render, not just the base fields
    assert 'value="harness"' in resp.text
    assert 'value="claude-code"' in resp.text


# ---------------------------------------------------------------------------
# Stale mock model id (dashboard bullet: "the mock model id hardcoded in
# the form is stale")
# ---------------------------------------------------------------------------


def test_mock_model_id_in_form_is_not_the_stale_2024_sonnet():
    resp = client.post(
        "/experiments", data=_factor_exp_payload(name="mock-id-exp"), follow_redirects=False
    )
    assert resp.status_code == 303
    raw = store.read_raw("mock-id-exp")
    assert "claude-3-5-sonnet-20241022" not in raw
