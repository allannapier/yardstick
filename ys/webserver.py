import sys

from ys import db, paths, procutil

DEFAULT_PORT = 8501


class WebServerError(Exception):
    pass


def web_up(port: int = DEFAULT_PORT, host: str = "127.0.0.1") -> str:
    paths.ensure_home()
    db.init_db()  # dashboard routes assume the schema exists; `ys init` is easy to skip

    existing = procutil.read_pid(paths.WEB_PID_PATH)
    if existing and procutil.alive(existing):
        raise WebServerError(
            f"dashboard already running (pid {existing}). Run `ys web down` first."
        )

    pid = procutil.launch_detached(
        [sys.executable, "-m", "uvicorn", "ys.web.app:app", "--host", host, "--port", str(port)],
        paths.WEB_LOG_PATH,
        paths.WEB_PID_PATH,
        paths.WEB_PORT_PATH,
        port,
    )

    if not procutil.wait_ready(f"http://{host}:{port}/health"):
        raise WebServerError(
            f"dashboard (pid {pid}) did not become ready within 15s. Check the log at {paths.WEB_LOG_PATH}"
        )

    return f"http://{host}:{port}"


def read_port(default: int = DEFAULT_PORT) -> int:
    return procutil.read_port(paths.WEB_PORT_PATH, default)


def web_down() -> str:
    return procutil.stop(paths.WEB_PID_PATH, paths.WEB_PORT_PATH)


def web_status() -> tuple[bool, int | None]:
    return procutil.status(paths.WEB_PID_PATH)
