import json

from ys import portkey_app


# --- _check_local_auth -------------------------------------------------------


def test_check_local_auth_accepts_bearer_form():
    assert portkey_app._check_local_auth({"authorization": "Bearer sk-local"}, "sk-local") is True


def test_check_local_auth_accepts_x_api_key_form():
    assert portkey_app._check_local_auth({"x-api-key": "sk-local"}, "sk-local") is True


def test_check_local_auth_rejects_wrong_key():
    assert portkey_app._check_local_auth({"authorization": "Bearer wrong"}, "sk-local") is False
    assert portkey_app._check_local_auth({}, "sk-local") is False


def test_check_local_auth_passes_when_no_key_configured():
    assert portkey_app._check_local_auth({}, None) is True
    assert portkey_app._check_local_auth({}, "") is True


# --- _active_run_id -----------------------------------------------------------


def test_active_run_id_prefers_explicit_header():
    assert portkey_app._active_run_id({"x-ys-run": "run-from-header"}) == "run-from-header"


def test_active_run_id_falls_back_to_active_run_file(tmp_path, monkeypatch):
    from ys import paths

    active_path = tmp_path / "active.json"
    active_path.write_text(json.dumps({"run_id": "run-from-file"}))
    monkeypatch.setattr(paths, "ACTIVE_RUN_PATH", str(active_path))

    assert portkey_app._active_run_id({}) == "run-from-file"


def test_active_run_id_none_when_nothing_available(tmp_path, monkeypatch):
    from ys import paths

    monkeypatch.setattr(paths, "ACTIVE_RUN_PATH", str(tmp_path / "does_not_exist.json"))
    assert portkey_app._active_run_id({}) is None


# --- _build_upstream_headers --------------------------------------------------


def test_build_upstream_headers_carries_portkey_auth(monkeypatch):
    monkeypatch.setenv("PORTKEY_API_KEY", "pk-real")
    monkeypatch.setenv("PORTKEY_VIRTUAL_KEY", "vk-anthropic")

    headers = portkey_app._build_upstream_headers({}, run_id=None)

    assert headers["authorization"] == "Bearer pk-real"
    assert headers["x-portkey-virtual-key"] == "vk-anthropic"
    assert "x-portkey-metadata" not in headers


def test_build_upstream_headers_tags_run_id_as_metadata(monkeypatch):
    monkeypatch.setenv("PORTKEY_API_KEY", "pk-real")
    monkeypatch.delenv("PORTKEY_VIRTUAL_KEY", raising=False)

    headers = portkey_app._build_upstream_headers({}, run_id="abc123")

    assert json.loads(headers["x-portkey-metadata"]) == {"ys_run_id": "abc123"}
    assert "x-portkey-virtual-key" not in headers


def test_build_upstream_headers_carries_anthropic_version_and_accept(monkeypatch):
    monkeypatch.setenv("PORTKEY_API_KEY", "pk-real")

    headers = portkey_app._build_upstream_headers(
        {"anthropic-version": "2023-06-01", "accept": "text/event-stream"}, run_id=None
    )

    assert headers["anthropic-version"] == "2023-06-01"
    assert headers["accept"] == "text/event-stream"


# --- end-to-end route behavior (auth gate) via FastAPI's TestClient ----------


def test_route_rejects_missing_local_auth(monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-local")
    client = TestClient(portkey_app.app)

    resp = client.post("/v1/messages", json={"model": "claude-sonnet-5"})

    assert resp.status_code == 401


def test_readiness_endpoint_ok():
    from fastapi.testclient import TestClient

    client = TestClient(portkey_app.app)
    resp = client.get("/health/readiness")
    assert resp.status_code == 200
