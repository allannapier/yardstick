"""Per-run workspace isolation (IMPROVEMENTS.md feature 2).

Before this, `success_check` ran via `shell=True` in whatever directory `ys
end` happened to be invoked from -- no working directory, no setup, no
teardown, and no reset between repeats, so repeat 2 started from whatever
state repeat 1's agent left on disk. This module gives each run its own
directory -- optionally checked out fresh from `task.repo`@`task.ref` -- and
runs `task.setup`/`task.teardown` in it. `ys/runner.py` (feature 1) is the
main caller; `ys/runs.py`'s `finish_run` also takes an optional `cwd` so
`success_check` runs in the same workspace.

SAFETY RULE -- read this before touching `cleanup_workspace`:

    Only a directory yardstick itself created, inside workspaces_root()
    (~/.yardstick/workspaces/<run_id>/), may ever be deleted by this module.

`Workspace.managed` records whether the directory was created here (`repo`
given -- a fresh clone) or supplied by the caller (`workdir` with no `repo`,
or the plain-cwd fallback with neither). `cleanup_workspace` refuses to
delete anything unless *both* `managed` is True *and* the resolved
`delete_root` is a real child of `workspaces_root()` -- two independent checks,
not just the flag, so a future bug that mislabels a caller-supplied
directory as managed still can't delete it. A `task.workdir` pointing at the
user's real project is therefore never touched by cleanup, no matter what
its path looks like: it is never `managed` and never has a `delete_root`.

Shell exposure: `task.setup`/`task.teardown` are shell strings from a config
file, exactly like `task.success_check` already was before this feature --
running them here isn't new exposure. What *would* be new is interpolating a
yardstick-constructed value (the workspace path, the run id, ...) into that
string. Nothing in this module does that: every path is passed via `cwd`,
never string-formatted into the command, and `git clone`/`git checkout`
pass `task.repo`/`task.ref` as argv list elements, never through a shell at
all.
"""
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional

from ys import paths


class WorkspaceError(Exception):
    pass


def workspaces_root() -> str:
    """Every managed workspace lives under here. Lazily computed from
    `paths.YARDSTICK_HOME` on every call -- deliberately *not* a
    module-level constant, which would freeze in whatever
    `paths.YARDSTICK_HOME` was at import time and silently stop honouring
    `YARDSTICK_HOME` changed afterwards (the tests monkeypatch
    `paths.YARDSTICK_HOME` per test; a frozen constant here would make every
    test in this module race the real `~/.yardstick`). Same lazy pattern as
    `ys/harness.py`'s `_backup_dir()`, for the same reason."""
    return os.path.join(paths.YARDSTICK_HOME, "workspaces")


@dataclass
class Workspace:
    path: str  # cwd for setup/the agent/success_check/teardown
    managed: bool  # True iff this workspace owns a directory it may delete
    # The exact directory `cleanup_workspace` may `rmtree` -- not
    # necessarily `path` itself (task.workdir can name a subdirectory of a
    # clone), and never set at all for a non-managed workspace.
    delete_root: Optional[str] = None


@dataclass
class ShellResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False


def _run_shell(command: str, cwd: str, timeout_s: int, extra_env: Optional[dict] = None) -> ShellResult:
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    try:
        proc = subprocess.run(
            command, shell=True, cwd=cwd, env=env, timeout=timeout_s,
            capture_output=True, text=True,
        )
        return ShellResult(proc.returncode, proc.stdout or "", proc.stderr or "")
    except subprocess.TimeoutExpired as e:
        return ShellResult(-1, e.stdout or "", e.stderr or "", timed_out=True)


def run_setup(task, ws: Workspace) -> ShellResult:
    """Runs once per repeat, before the agent. A no-op (returncode 0) when
    `task.setup` isn't declared, so callers can call this unconditionally."""
    if not task.setup:
        return ShellResult(0, "", "")
    return _run_shell(task.setup, cwd=ws.path, timeout_s=task.timeout_s, extra_env={"YS_WORKDIR": ws.path})


def run_teardown(task, ws: Workspace) -> ShellResult:
    """Runs once per repeat, after scoring -- best-effort by design (the
    caller logs a non-zero exit but a failed teardown must not stop
    `cleanup_workspace` from still running, or abort the loop). A no-op
    when `task.teardown` isn't declared."""
    if not task.teardown:
        return ShellResult(0, "", "")
    return _run_shell(task.teardown, cwd=ws.path, timeout_s=task.timeout_s, extra_env={"YS_WORKDIR": ws.path})


def _git(args: list, cwd: Optional[str] = None, timeout_s: int = 300) -> subprocess.CompletedProcess:
    """Every git invocation in this module takes its arguments as a list,
    never a shell string -- `task.repo`/`task.ref` are config-file values
    and this keeps them out of shell interpolation entirely (no `shell=True`
    at all here, stronger than the cwd-not-interpolation rule the rest of
    this module follows for `setup`/`teardown`)."""
    return subprocess.run(["git", *args], cwd=cwd, timeout=timeout_s, capture_output=True, text=True)


def _clone_repo(repo: str, ref: Optional[str], dest: str, timeout_s: int = 300):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    proc = _git(["clone", "--quiet", repo, dest], timeout_s=timeout_s)
    if proc.returncode != 0:
        raise WorkspaceError(f"git clone of '{repo}' failed: {proc.stderr.strip()}")
    if ref:
        proc = _git(["checkout", "--quiet", ref], cwd=dest, timeout_s=timeout_s)
        if proc.returncode != 0:
            raise WorkspaceError(
                f"git checkout of ref '{ref}' (repo '{repo}') failed: {proc.stderr.strip()}"
            )


def prepare_workspace(task, run_id: str) -> Workspace:
    """One workspace per run, keyed by `run_id` (a fresh uuid4 from
    `runs.begin_run`) so concurrent/back-to-back repeats never share a
    directory, and so `cleanup_workspace` never has to guess which run a
    given managed directory belonged to.

    - `task.repo` set: a fresh `git clone` (+ `git checkout <ref>` if given)
      into `workspaces_root()/<run_id>` -- the actual "every repeat starts
      from an identical tree" mechanism feature 2 asks for. `task.workdir`,
      if also given, is a relative subdirectory *within* that clone (e.g. a
      monorepo package) used as the returned `path`; the whole clone --
      `delete_root` -- is still what `cleanup_workspace` may remove.
    - `task.repo` not set, `task.workdir` set: an existing directory the
      caller manages, typically the user's real project checkout. Used
      as-is -- never created, never deleted (`managed=False`). Repeats are
      *not* isolated in this mode: there's no ref to reset to, only
      whatever `task.setup`/`task.teardown` do on their own.
    - Neither set: falls back to the current working directory, matching
      `success_check`'s pre-feature-2 behaviour exactly. Also
      `managed=False`.
    """
    if task.repo:
        dest = os.path.join(workspaces_root(), run_id)
        if os.path.exists(dest):
            # run_id is a fresh uuid4 per begin_run -- this should be
            # unreachable. Refuse rather than clone into (and later rm -rf)
            # a directory this call didn't just create itself.
            raise WorkspaceError(f"workspace directory already exists: {dest}")
        _clone_repo(task.repo, task.ref, dest)
        path = os.path.join(dest, task.workdir) if task.workdir else dest
        if not os.path.isdir(path):
            raise WorkspaceError(
                f"task.workdir '{task.workdir}' does not exist inside the checkout of "
                f"'{task.repo}'"
            )
        return Workspace(path=path, managed=True, delete_root=dest)

    if task.workdir:
        if not os.path.isdir(task.workdir):
            raise WorkspaceError(
                f"task.workdir '{task.workdir}' does not exist (and no task.repo to "
                "create it from)"
            )
        return Workspace(path=os.path.abspath(task.workdir), managed=False)

    return Workspace(path=os.getcwd(), managed=False)


def cleanup_workspace(ws: Workspace):
    """Delete a managed workspace's directory. Refuses unless *both* (1)
    `ws.managed` is True and (2) the resolved `delete_root` is a real child
    of `workspaces_root()` -- see the module docstring's safety rule. A
    caller-supplied `task.workdir` (`managed=False`) is never touched,
    regardless of its path; `ignore_errors=True` on the actual removal
    reflects that a failed cleanup (e.g. a file the agent left read-only)
    should not crash the repeat loop over disk hygiene."""
    if not ws.managed:
        return
    if not ws.delete_root:
        raise WorkspaceError("managed workspace has no delete_root -- refusing to guess what to delete")

    root = os.path.realpath(workspaces_root())
    target = os.path.realpath(ws.delete_root)
    if not target.startswith(root + os.sep):
        # Should be unreachable -- delete_root is only ever set to a path
        # this module itself built under workspaces_root(). Treated as a hard
        # safety check anyway, not a normal error path: never silently
        # widen what may be deleted just because a flag says so.
        raise WorkspaceError(
            f"refusing to delete '{ws.delete_root}' -- it is not inside the "
            f"yardstick-managed workspaces directory ({workspaces_root()})"
        )
    shutil.rmtree(target, ignore_errors=True)
