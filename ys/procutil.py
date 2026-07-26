"""Shared pidfile-backed background process management for ys/proxy.py
(the LiteLLM proxy) and ys/webserver.py (the dashboard) -- both are "start a
detached subprocess, remember its pid/port in a file, tear it down later"
and there's no reason for that bookkeeping to live twice.
"""
import os
import signal
import socket
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


def port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Whether something is accepting connections on `port`, regardless of
    whether it's a process ys knows about. Used to tell a genuinely free port
    apart from one held by a process that outlived (or was never recorded in)
    our pidfile."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


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


def _signal_process_group(pid: int, sig: int) -> None:
    """Signal the whole process group `launch_detached` created (it starts
    the child with start_new_session=True, so a plain os.kill on the leader
    pid never reaches anything it forked). Falls back to signalling just the
    pid if the group lookup fails for any reason."""
    try:
        os.killpg(os.getpgid(pid), sig)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass


def _wait_for_death(pid: int, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if not alive(pid):
            return True
        time.sleep(0.25)
    return not alive(pid)


def stop(pid_path: str, port_path: str, force: bool = False, grace_s: float = 5.0) -> str:
    pid = read_pid(pid_path)
    if pid is None:
        return "no pidfile found; nothing to stop"

    if not alive(pid):
        remove_if_exists(pid_path)
        remove_if_exists(port_path)
        return f"process (pid {pid}) was already gone; removed stale pidfile"

    _signal_process_group(pid, signal.SIGTERM)
    if _wait_for_death(pid, grace_s):
        remove_if_exists(pid_path)
        remove_if_exists(port_path)
        return f"stopped process (pid {pid})"

    if not force:
        return (
            f"process (pid {pid}) did not stop within {grace_s:.0f}s of SIGTERM; it is "
            "still running and the pidfile was left in place so it can still be found. "
            "Re-run with `--force` to send SIGKILL."
        )

    _signal_process_group(pid, signal.SIGKILL)
    if _wait_for_death(pid, grace_s):
        remove_if_exists(pid_path)
        remove_if_exists(port_path)
        return f"process (pid {pid}) did not respond to SIGTERM; killed it with SIGKILL"

    return (
        f"process (pid {pid}) is still alive after SIGKILL (likely a zombie or stuck in "
        "uninterruptible sleep). The pidfile was left in place -- inspect the process manually."
    )


def status(pid_path: str) -> tuple[bool, int | None]:
    pid = read_pid(pid_path)
    if pid is None:
        return False, None
    return alive(pid), pid
