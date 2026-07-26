import os
import socket
import subprocess
import sys
import threading
import time

from ys import procutil


def _pid_path(tmp_path):
    return str(tmp_path / "proc.pid"), str(tmp_path / "proc.port")


def _spawn(cmd, **kwargs):
    """Popen a process and reap it in the background as soon as it exits.

    In production the pidfile-tracked process is an orphan of a long-gone CLI
    invocation, so init reaps it the instant it dies. In this test process we
    *are* the parent, so without an active waiter a killed child sits as a
    zombie -- for which `kill(pid, 0)` still succeeds -- and `alive()` would
    (correctly, for a zombie) keep reporting it as present.
    """
    proc = subprocess.Popen(cmd, start_new_session=True, **kwargs)
    threading.Thread(target=proc.wait, daemon=True).start()
    return proc


def test_stop_no_pidfile(tmp_path):
    pid_path, port_path = _pid_path(tmp_path)
    assert procutil.stop(pid_path, port_path) == "no pidfile found; nothing to stop"


def test_stop_removes_stale_pidfile(tmp_path):
    pid_path, port_path = _pid_path(tmp_path)
    # Spawn and reap a process so its pid is guaranteed dead, then pretend we
    # still have a pidfile pointing at it.
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    with open(pid_path, "w") as f:
        f.write(str(proc.pid))
    with open(port_path, "w") as f:
        f.write("4000")

    result = procutil.stop(pid_path, port_path)

    assert "already gone" in result
    assert not os.path.exists(pid_path)
    assert not os.path.exists(port_path)


def test_stop_kills_live_process(tmp_path):
    pid_path, port_path = _pid_path(tmp_path)
    proc = _spawn([sys.executable, "-c", "import time; time.sleep(60)"])
    with open(pid_path, "w") as f:
        f.write(str(proc.pid))
    open(port_path, "w").close()

    try:
        result = procutil.stop(pid_path, port_path)
    finally:
        if proc.poll() is None:
            proc.kill()

    assert result == f"stopped process (pid {proc.pid})"
    assert not os.path.exists(pid_path)
    assert not os.path.exists(port_path)
    assert not procutil.alive(proc.pid)


# A leader that spawns a child in its own session/process group (Popen
# without start_new_session leaves the child in the leader's group), then
# publishes the child's pid atomically -- write-then-rename, so the waiting
# test can never read a half-written file. Kept in Python rather than a shell
# one-liner so the test doesn't depend on an external `bash`.
_LEADER_SPAWNS_CHILD = (
    "import pathlib, subprocess, sys, time;"
    "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']);"
    "tmp = pathlib.Path('child.pid.tmp');"
    "tmp.write_text(str(child.pid));"
    "tmp.rename('child.pid');"
    "time.sleep(60)"
)


def test_stop_reaches_process_group_children(tmp_path):
    pid_path, port_path = _pid_path(tmp_path)
    # A leader that spawns a child in the same session/process group --
    # mirrors launch_detached's start_new_session=True, where a plain
    # os.kill(leader_pid) never reaches anything the leader spawned.
    proc = _spawn([sys.executable, "-c", _LEADER_SPAWNS_CHILD], cwd=tmp_path)
    with open(pid_path, "w") as f:
        f.write(str(proc.pid))
    open(port_path, "w").close()

    child_pid_file = tmp_path / "child.pid"
    deadline = time.time() + 3
    while not child_pid_file.exists() and time.time() < deadline:
        time.sleep(0.05)
    child_pid = int(child_pid_file.read_text().strip())

    try:
        procutil.stop(pid_path, port_path)
        deadline = time.time() + 3
        while procutil.alive(child_pid) and time.time() < deadline:
            time.sleep(0.05)
        assert not procutil.alive(child_pid)
    finally:
        if procutil.alive(child_pid):
            os.kill(child_pid, 9)
        if proc.poll() is None:
            proc.kill()


_IGNORE_SIGTERM_AND_SLEEP = (
    "import signal, time, sys;"
    "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
    "print('ready', flush=True);"
    "time.sleep(60)"
)


def _spawn_ignoring_sigterm():
    """Spawn the sleeper above and block until its SIGTERM handler is
    actually installed, so stop() can't race the signal against startup."""
    proc = _spawn([sys.executable, "-u", "-c", _IGNORE_SIGTERM_AND_SLEEP], stdout=subprocess.PIPE, text=True)
    proc.stdout.readline()
    return proc


def test_stop_without_force_leaves_pidfile_when_process_survives_sigterm(tmp_path):
    pid_path, port_path = _pid_path(tmp_path)
    proc = _spawn_ignoring_sigterm()
    with open(pid_path, "w") as f:
        f.write(str(proc.pid))
    open(port_path, "w").close()

    try:
        result = procutil.stop(pid_path, port_path, grace_s=1.0)

        assert "did not stop" in result
        assert "--force" in result
        assert os.path.exists(pid_path)
        assert procutil.alive(proc.pid)
    finally:
        proc.kill()


def test_stop_with_force_escalates_to_sigkill(tmp_path):
    pid_path, port_path = _pid_path(tmp_path)
    proc = _spawn_ignoring_sigterm()
    with open(pid_path, "w") as f:
        f.write(str(proc.pid))
    open(port_path, "w").close()

    try:
        result = procutil.stop(pid_path, port_path, force=True, grace_s=1.0)
    finally:
        if proc.poll() is None:
            proc.kill()

    assert "SIGKILL" in result
    assert not os.path.exists(pid_path)
    assert not os.path.exists(port_path)


def test_port_in_use_false_when_nothing_listening():
    # Bind an ephemeral port and immediately release it: the kernel handed it
    # out because it was free, which is a far safer bet on a shared CI runner
    # than asserting some hardcoded port number is idle.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    assert procutil.port_in_use(port) is False


def test_port_in_use_true_when_something_listening():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    try:
        assert procutil.port_in_use(port) is True
    finally:
        server.close()
