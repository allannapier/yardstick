from fastapi.testclient import TestClient

from ys import db, state
from ys.web.app import app

client = TestClient(app)


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
            "arm_baseline": ["arm-a"],
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
            "arm_baseline": [],
            "arm_notes": ["", ""],
        },
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert "/experiments/new" in resp.headers["location"]
    assert "error=" in resp.headers["location"]


def test_experiment_detail_404_for_unknown_name():
    resp = client.get("/experiments/does-not-exist", follow_redirects=False)
    assert resp.status_code == 303
    assert "error=" in resp.headers["location"]


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
            "arm_baseline": ["only-arm"],
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
            "arm_baseline": ["only-arm"],
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
            "arm_baseline": ["only-arm"],
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
            "arm_baseline": ["only-arm"],
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
            "arm_baseline": ["only-arm"],
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
            "arm_baseline": ["only-arm"],
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
