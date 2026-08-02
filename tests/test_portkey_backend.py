from ys import portkey_backend, proxy


def _clear_portkey_env(monkeypatch):
    for var in ("LITELLM_MASTER_KEY", "PORTKEY_API_KEY", "PORTKEY_VIRTUAL_KEY"):
        monkeypatch.delenv(var, raising=False)


def test_proxy_up_requires_litellm_master_key(monkeypatch):
    _clear_portkey_env(monkeypatch)
    try:
        portkey_backend.proxy_up(port=4321)
        assert False, "expected ProxyError"
    except proxy.ProxyError as e:
        assert "LITELLM_MASTER_KEY" in str(e)


def test_proxy_up_requires_portkey_api_key(monkeypatch):
    _clear_portkey_env(monkeypatch)
    monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-local")
    try:
        portkey_backend.proxy_up(port=4321)
        assert False, "expected ProxyError"
    except proxy.ProxyError as e:
        assert "PORTKEY_API_KEY" in str(e)


def test_proxy_up_requires_portkey_virtual_key(monkeypatch):
    _clear_portkey_env(monkeypatch)
    monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-local")
    monkeypatch.setenv("PORTKEY_API_KEY", "pk-real")
    try:
        portkey_backend.proxy_up(port=4321)
        assert False, "expected ProxyError"
    except proxy.ProxyError as e:
        assert "PORTKEY_VIRTUAL_KEY" in str(e)


def test_proxy_up_launches_uvicorn_and_records_port(monkeypatch):
    _clear_portkey_env(monkeypatch)
    monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-local")
    monkeypatch.setenv("PORTKEY_API_KEY", "pk-real")
    monkeypatch.setenv("PORTKEY_VIRTUAL_KEY", "vk-anthropic")

    launched = {}

    def fake_launch_detached(cmd, log_path, pid_path, port_path, port):
        launched["cmd"] = cmd
        with open(pid_path, "w") as f:
            f.write("123")
        with open(port_path, "w") as f:
            f.write(str(port))
        return 123

    monkeypatch.setattr(portkey_backend.procutil, "launch_detached", fake_launch_detached)
    monkeypatch.setattr(portkey_backend.procutil, "wait_ready", lambda url, timeout_s=15.0: True)

    url = portkey_backend.proxy_up(port=4321)

    assert url == "http://localhost:4321"
    assert launched["cmd"][:2] == ["uvicorn", "ys.portkey_app:app"]
    assert "4321" in launched["cmd"]


def test_proxy_up_refuses_when_already_running(monkeypatch):
    _clear_portkey_env(monkeypatch)
    monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-local")
    monkeypatch.setenv("PORTKEY_API_KEY", "pk-real")
    monkeypatch.setenv("PORTKEY_VIRTUAL_KEY", "vk-anthropic")

    monkeypatch.setattr(portkey_backend.procutil, "read_pid", lambda pid_path: 999)
    monkeypatch.setattr(portkey_backend.procutil, "alive", lambda pid: True)

    try:
        portkey_backend.proxy_up(port=4321)
        assert False, "expected ProxyError"
    except proxy.ProxyError as e:
        assert "already running" in str(e)


# --- ys/proxy.py dispatch + backend bookkeeping -----------------------------


def test_read_backend_defaults_to_litellm_with_no_marker():
    assert proxy.read_backend() == proxy.BACKEND_LITELLM


def test_proxy_up_dispatches_to_portkey_backend_and_records_it(monkeypatch, tmp_path):
    monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-local")
    monkeypatch.setenv("PORTKEY_API_KEY", "pk-real")
    monkeypatch.setenv("PORTKEY_VIRTUAL_KEY", "vk-anthropic")

    def fake_portkey_proxy_up(port):
        return f"http://localhost:{port}"

    from ys import portkey_backend as pb

    monkeypatch.setattr(pb, "proxy_up", fake_portkey_proxy_up)

    url = proxy.proxy_up([], port=4322, backend=proxy.BACKEND_PORTKEY)

    assert url == "http://localhost:4322"
    assert proxy.read_backend() == proxy.BACKEND_PORTKEY


def test_proxy_up_rejects_unknown_backend():
    try:
        proxy.proxy_up([], port=4323, backend="not-a-real-backend")
        assert False, "expected ProxyError"
    except proxy.ProxyError as e:
        assert "unknown backend" in str(e)


def test_proxy_down_clears_backend_marker(monkeypatch):
    import os

    from ys import paths

    os.makedirs(os.path.dirname(paths.PROXY_BACKEND_PATH), exist_ok=True)
    with open(paths.PROXY_BACKEND_PATH, "w") as f:
        f.write(proxy.BACKEND_PORTKEY)

    assert proxy.read_backend() == proxy.BACKEND_PORTKEY
    proxy.proxy_down()
    assert proxy.read_backend() == proxy.BACKEND_LITELLM
