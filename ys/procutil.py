"""Shared pidfile-backed background process management for ys/proxy.py
(the LiteLLM proxy) and ys/webserver.py (the dashboard) -- both are "start a
detached subprocess, remember its pid/port in a file, tear it down later"
and there's no reason for that bookkeeping to live twice.
"""
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request


def read_pid(pid_path: str) -> int | None:
    if not os.path.exists(pid_path):
        return None
    try:
        with open(pid_path) as f:
            return int(f.read().strip())
    except (ValueError, OSError):
        return None


def read_port(port_path: str, default: int) -> int:
    if not os.path.exists(port_path):
        return default
    try:
        with open(port_path) as f:
            return int(f.read().strip())
    except (ValueError, OSError):
        return default


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_ready(url: str, timeout_s: float = 15.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(0.5)
    return False


def remove_if_exists(path: str):
    if os.path.exists(path):
        os.remove(path)


def launch_detached(cmd: list[str], log_path: str, pid_path: str, port_path: str, port: int) -> int:
    log = open(log_path, "ab")
    try:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    finally:
        log.close()

    with open(pid_path, "w") as f:
        f.write(str(proc.pid))
    with open(port_path, "w") as f:
        f.write(str(port))

    return proc.pid


def stop(pid_path: str, port_path: str) -> str:
    pid = read_pid(pid_path)
    if pid is None:
        return "no pidfile found; nothing to stop"

    if not alive(pid):
        remove_if_exists(pid_path)
        remove_if_exists(port_path)
        return f"process (pid {pid}) was already gone; removed stale pidfile"

    os.kill(pid, signal.SIGTERM)
    for _ in range(20):
        if not alive(pid):
            break
        time.sleep(0.25)
    remove_if_exists(pid_path)
    remove_if_exists(port_path)
    return f"stopped process (pid {pid})"


def status(pid_path: str) -> tuple[bool, int | None]:
    pid = read_pid(pid_path)
    if pid is None:
        return False, None
    return alive(pid), pid
