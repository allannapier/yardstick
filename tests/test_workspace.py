import os
import subprocess

import pytest

from ys import paths, workspace
from ys.experiment import Task


def _task(**overrides):
    defaults = {"id": "t0", "success_check": "true"}
    defaults.update(overrides)
    return Task.model_validate(defaults)


def _init_local_repo(tmp_path, name="upstream"):
    """A real, local-only git repo -- cloning from a filesystem path never
    touches the network, so this stays within the "never touch the network"
    test rule while still exercising the real `git clone`/`git checkout`
    subprocess calls."""
    repo_dir = tmp_path / name
    repo_dir.mkdir()
    subprocess.run(["git", "init", "--quiet"], cwd=repo_dir, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_dir, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo_dir, check=True)
    (repo_dir / "README.md").write_text("hello\n")
    subprocess.run(["git", "add", "."], cwd=repo_dir, check=True)
    subprocess.run(["git", "commit", "--quiet", "-m", "initial"], cwd=repo_dir, check=True)
    subprocess.run(["git", "branch", "--quiet", "a-branch"], cwd=repo_dir, check=True)
    return str(repo_dir)


# --- prepare_workspace ------------------------------------------------------


def test_prepare_workspace_with_no_repo_or_workdir_falls_back_to_cwd():
    ws = workspace.prepare_workspace(_task(), "run-1")
    assert ws.path == os.getcwd()
    assert ws.managed is False
    assert ws.delete_root is None


def test_prepare_workspace_with_workdir_no_repo_uses_existing_directory(tmp_path):
    d = tmp_path / "existing-project"
    d.mkdir()
    ws = workspace.prepare_workspace(_task(workdir=str(d)), "run-1")
    assert ws.path == str(d)
    assert ws.managed is False
    assert ws.delete_root is None


def test_prepare_workspace_workdir_without_repo_must_exist(tmp_path):
    missing = tmp_path / "does-not-exist"
    with pytest.raises(workspace.WorkspaceError):
        workspace.prepare_workspace(_task(workdir=str(missing)), "run-1")


def test_prepare_workspace_clones_repo_into_managed_directory(tmp_path):
    repo = _init_local_repo(tmp_path)
    ws = workspace.prepare_workspace(_task(repo=repo), "run-clone-1")
    assert ws.managed is True
    assert ws.delete_root == os.path.join(workspace.workspaces_root(), "run-clone-1")
    assert ws.path == ws.delete_root
    assert os.path.isfile(os.path.join(ws.path, "README.md"))


def test_prepare_workspace_checks_out_ref(tmp_path):
    repo = _init_local_repo(tmp_path)
    subprocess.run(
        ["git", "checkout", "--quiet", "-b", "feature-x"], cwd=repo, check=True,
    )
    with open(os.path.join(repo, "feature.txt"), "w") as f:
        f.write("feature branch file\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "--quiet", "-m", "feature"], cwd=repo, check=True)

    ws = workspace.prepare_workspace(_task(repo=repo, ref="feature-x"), "run-clone-ref")
    assert os.path.isfile(os.path.join(ws.path, "feature.txt"))


def test_prepare_workspace_bad_ref_raises(tmp_path):
    repo = _init_local_repo(tmp_path)
    with pytest.raises(workspace.WorkspaceError):
        workspace.prepare_workspace(_task(repo=repo, ref="no-such-ref"), "run-bad-ref")


def test_prepare_workspace_bad_repo_raises(tmp_path):
    with pytest.raises(workspace.WorkspaceError):
        workspace.prepare_workspace(_task(repo=str(tmp_path / "nope")), "run-bad-repo")


def test_prepare_workspace_repo_plus_workdir_uses_subdirectory(tmp_path):
    repo = _init_local_repo(tmp_path)
    os.makedirs(os.path.join(repo, "packages", "app"))
    with open(os.path.join(repo, "packages", "app", "marker.txt"), "w") as f:
        f.write("x\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "--quiet", "-m", "add subdir"], cwd=repo, check=True)

    ws = workspace.prepare_workspace(_task(repo=repo, workdir="packages/app"), "run-subdir")
    assert ws.path.endswith(os.path.join("packages", "app"))
    assert os.path.isfile(os.path.join(ws.path, "marker.txt"))
    # the *whole clone*, not the subdirectory, is what cleanup may delete
    assert ws.delete_root == os.path.join(workspace.workspaces_root(), "run-subdir")


def test_prepare_workspace_repo_plus_missing_workdir_raises(tmp_path):
    repo = _init_local_repo(tmp_path)
    with pytest.raises(workspace.WorkspaceError):
        workspace.prepare_workspace(_task(repo=repo, workdir="no/such/dir"), "run-bad-subdir")


def test_prepare_workspace_refuses_to_clone_into_an_existing_directory(tmp_path):
    repo = _init_local_repo(tmp_path)
    dest = os.path.join(workspace.workspaces_root(), "run-collide")
    os.makedirs(dest)
    with pytest.raises(workspace.WorkspaceError):
        workspace.prepare_workspace(_task(repo=repo), "run-collide")


# --- cleanup_workspace: the safety rule --------------------------------------


def test_cleanup_workspace_removes_a_managed_clone(tmp_path):
    repo = _init_local_repo(tmp_path)
    ws = workspace.prepare_workspace(_task(repo=repo), "run-cleanup-1")
    assert os.path.isdir(ws.path)
    workspace.cleanup_workspace(ws)
    assert not os.path.exists(ws.path)


def test_cleanup_workspace_never_deletes_an_unmanaged_workdir(tmp_path):
    """The critical safety property: a task.workdir pointing at a real
    project (no task.repo) must survive cleanup untouched, no matter how
    many repeats run against it."""
    d = tmp_path / "real-users-project"
    d.mkdir()
    (d / "important.txt").write_text("do not delete me\n")
    ws = workspace.prepare_workspace(_task(workdir=str(d)), "run-1")

    workspace.cleanup_workspace(ws)

    assert os.path.isdir(d)
    assert (d / "important.txt").read_text() == "do not delete me\n"


def test_cleanup_workspace_never_deletes_the_cwd_fallback(tmp_path, monkeypatch):
    marker = tmp_path / "cwd-marker.txt"
    marker.write_text("still here\n")
    monkeypatch.chdir(tmp_path)

    ws = workspace.prepare_workspace(_task(), "run-1")
    workspace.cleanup_workspace(ws)

    assert marker.exists()


def test_cleanup_workspace_refuses_a_managed_path_outside_the_root(tmp_path):
    """Defense in depth: even if `managed` were somehow set incorrectly, the
    path-containment check must independently refuse to delete anything
    outside WORKSPACES_ROOT. This should be unreachable through the public
    prepare_workspace API -- constructed directly here to prove the second
    check really is independent of the first."""
    outside = tmp_path / "definitely-not-managed"
    outside.mkdir()
    (outside / "keepme.txt").write_text("x\n")
    ws = workspace.Workspace(path=str(outside), managed=True, delete_root=str(outside))

    with pytest.raises(workspace.WorkspaceError):
        workspace.cleanup_workspace(ws)

    assert os.path.isdir(outside)


def test_cleanup_workspace_is_a_noop_for_unmanaged_even_without_delete_root():
    ws = workspace.Workspace(path="/some/path", managed=False, delete_root=None)
    workspace.cleanup_workspace(ws)  # must not raise


def test_managed_workspaces_live_under_workspaces_root():
    assert workspace.workspaces_root() == os.path.join(paths.YARDSTICK_HOME, "workspaces")


# --- run_setup / run_teardown ------------------------------------------------


def test_run_setup_is_a_noop_when_not_declared(tmp_path):
    ws = workspace.Workspace(path=str(tmp_path), managed=False)
    result = workspace.run_setup(_task(), ws)
    assert result.returncode == 0


def test_run_setup_runs_in_the_workspace_cwd(tmp_path):
    ws = workspace.Workspace(path=str(tmp_path), managed=False)
    result = workspace.run_setup(_task(setup="pwd"), ws)
    assert result.returncode == 0
    assert str(tmp_path) in result.stdout


def test_run_setup_does_not_interpolate_the_path_into_the_command_string(tmp_path):
    """The safety rule from IMPROVEMENTS.md: no shell string is ever built
    by formatting a yardstick-constructed value into it -- the path is
    passed via cwd/env only. Proven here by using a workspace path containing
    a shell metacharacter that would break (or worse, execute something) if
    it were ever string-interpolated into the command."""
    d = tmp_path / "weird; touch pwned"
    d.mkdir()
    ws = workspace.Workspace(path=str(d), managed=False)
    result = workspace.run_setup(_task(setup="echo ok"), ws)
    assert result.returncode == 0
    assert result.stdout.strip() == "ok"
    assert not (tmp_path / "pwned").exists()


def test_run_setup_exposes_workdir_env_var(tmp_path):
    ws = workspace.Workspace(path=str(tmp_path), managed=False)
    result = workspace.run_setup(_task(setup="echo $YS_WORKDIR"), ws)
    assert result.stdout.strip() == str(tmp_path)


def test_run_teardown_is_a_noop_when_not_declared(tmp_path):
    ws = workspace.Workspace(path=str(tmp_path), managed=False)
    result = workspace.run_teardown(_task(), ws)
    assert result.returncode == 0


def test_run_teardown_runs_in_the_workspace_cwd(tmp_path):
    ws = workspace.Workspace(path=str(tmp_path), managed=False)
    result = workspace.run_teardown(_task(teardown="touch teardown-marker"), ws)
    assert result.returncode == 0
    assert (tmp_path / "teardown-marker").exists()


def test_run_setup_times_out(tmp_path):
    ws = workspace.Workspace(path=str(tmp_path), managed=False)
    task = _task(setup="sleep 5", timeout_s=1)
    result = workspace.run_setup(task, ws)
    assert result.timed_out is True
    assert result.returncode != 0
