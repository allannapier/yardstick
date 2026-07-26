import os
import re
import sqlite3
from contextlib import contextmanager
from urllib.parse import unquote

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


def test_health():
    assert client.get("/health").text == "ok"


def test_index_empty():
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
