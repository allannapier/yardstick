import os

import pytest


@pytest.fixture(autouse=True)
def isolated_yardstick_home(monkeypatch, tmp_path):
    """Every test gets its own YARDSTICK_HOME so tests never touch a real
    ~/.yardstick or collide with each other. Every ys module reads paths via
    `paths.X` (not `from ys.paths import X`), so patching ys.paths' own
    attributes here is sufficient -- no per-module patching needed."""
    home = tmp_path / "yardstick_home"
    monkeypatch.setenv("YARDSTICK_HOME", str(home))

    import ys.paths as paths

    monkeypatch.setattr(paths, "YARDSTICK_HOME", str(home))
    monkeypatch.setattr(paths, "DB_PATH", os.path.join(str(home), "yardstick.db"))
    monkeypatch.setattr(
        paths, "DROPPED_LOG_PATH", os.path.join(str(home), "dropped_requests.jsonl")
    )
    monkeypatch.setattr(paths, "ACTIVE_RUN_PATH", os.path.join(str(home), "active.json"))
    monkeypatch.setattr(paths, "LAST_ENDED_RUN_PATH", os.path.join(str(home), "last_ended.json"))
    monkeypatch.setattr(paths, "PROXY_CONFIG_PATH", os.path.join(str(home), "proxy_config.yaml"))
    monkeypatch.setattr(paths, "PROXY_PID_PATH", os.path.join(str(home), "proxy.pid"))
    monkeypatch.setattr(paths, "PROXY_PORT_PATH", os.path.join(str(home), "proxy.port"))
    monkeypatch.setattr(paths, "PROXY_LOG_PATH", os.path.join(str(home), "proxy.log"))
    monkeypatch.setattr(paths, "EXPERIMENTS_DIR", os.path.join(str(home), "experiments"))
    monkeypatch.setattr(paths, "WEB_PID_PATH", os.path.join(str(home), "web.pid"))
    monkeypatch.setattr(paths, "WEB_PORT_PATH", os.path.join(str(home), "web.port"))
    monkeypatch.setattr(paths, "WEB_LOG_PATH", os.path.join(str(home), "web.log"))

    from ys import db

    db.init_db()

    yield home
