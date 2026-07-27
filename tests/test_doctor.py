"""Feature 4 (IMPROVEMENTS.md): unit tests for each `ys/doctor.py` check --
one pass/warn/fail (and skip, where applicable) path per check, plus a
handful of `run_checks` composition tests. `tests/conftest.py`'s autouse
`isolated_yardstick_home` fixture already gives every test its own
YARDSTICK_HOME and a freshly-migrated database (via `db.init_db()`), so most
"pass" paths are the default state and most "warn"/"fail" paths need one
targeted change.
"""
import os
import sqlite3

import pytest

from ys import db, doctor, dropped, harness, paths, proxy, runs, state
from ys.experiment import Experiment


@pytest.fixture(autouse=True)
def fake_agents(monkeypatch, tmp_path):
    """Never touch the real ~/.claude or ~/.config/opencode while testing
    doctor's harness-config check (same fixture shape as tests/test_harness.py).
    claude-code keeps a `project_relpath` like the real AgentSpec (and like
    tests/test_harness.py's own fixture) so `check_harness_config`'s
    project-scope branch exercises the real `scopes_for_agent`/`resolve_path`
    logic instead of silently only ever seeing "user" scope -- safe without a
    `monkeypatch.chdir` here because it's read-only and this worktree has no
    `.claude/settings.json` at its root to accidentally read."""
    claude_path = str(tmp_path / "claude_settings.json")
    opencode_path = str(tmp_path / "opencode.jsonc")
    monkeypatch.setitem(
        harness.AGENTS,
        "claude-code",
        harness.AgentSpec("claude-code", [claude_path], project_relpath=os.path.join(".claude", "settings.json")),
    )
    monkeypatch.setitem(harness.AGENTS, "opencode", harness.AgentSpec("opencode", [opencode_path]))
    return {"claude-code": claude_path, "opencode": opencode_path}


def _exp(**task_overrides):
    task = {"id": "t0", "success_check": "true"}
    task.update(task_overrides)
    return Experiment.model_validate(
        {
            "experiment": "doctor-test-exp",
            "task": task,
            "arms": [{"id": "arm-a", "factors": {"model": "claude-sonnet-5"}, "baseline": True}],
        }
    )


# --- check_home_directory ---------------------------------------------------


def test_check_home_directory_passes_when_it_exists_and_is_writable():
    result = doctor.check_home_directory()
    assert result.status == doctor.PASS
    assert paths.YARDSTICK_HOME in result.message


def test_check_home_directory_warns_when_missing(monkeypatch, tmp_path):
    missing = str(tmp_path / "does-not-exist-yet")
    monkeypatch.setattr(paths, "YARDSTICK_HOME", missing)
    result = doctor.check_home_directory()
    assert result.status == doctor.WARN
    assert "ys init" in result.message


def test_check_home_directory_fails_when_not_writable(monkeypatch):
    monkeypatch.setattr(os, "access", lambda path, mode: False)
    result = doctor.check_home_directory()
    assert result.status == doctor.FAIL
    assert "not writable" in result.message


# --- check_schema_version ---------------------------------------------------


def test_check_schema_version_passes_when_current():
    result = doctor.check_schema_version()
    assert result.status == doctor.PASS
    assert f"version {len(db.MIGRATIONS)}" in result.message


def test_check_schema_version_warns_when_no_database(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "DB_PATH", str(tmp_path / "nope.db"))
    result = doctor.check_schema_version()
    assert result.status == doctor.WARN
    assert "ys init" in result.message


def test_check_schema_version_warns_when_behind():
    conn = sqlite3.connect(paths.DB_PATH)
    conn.execute(f"PRAGMA user_version = {len(db.MIGRATIONS) - 1}")
    conn.close()
    result = doctor.check_schema_version()
    assert result.status == doctor.WARN
    assert "code expects" in result.message


def test_check_schema_version_warns_when_ahead():
    conn = sqlite3.connect(paths.DB_PATH)
    conn.execute(f"PRAGMA user_version = {len(db.MIGRATIONS) + 1}")
    conn.close()
    result = doctor.check_schema_version()
    assert result.status == doctor.WARN
    assert "newer than this code" in result.message


def test_check_schema_version_never_creates_the_db_file(tmp_path, monkeypatch):
    """Regression guard: a naive db.cursor()/db.connect() call would create
    an empty database file as a side effect of merely checking on it."""
    missing = str(tmp_path / "nope.db")
    monkeypatch.setattr(paths, "DB_PATH", missing)
    doctor.check_schema_version()
    assert not os.path.exists(missing)


# --- check_proxy_process -----------------------------------------------------


def test_check_proxy_process_passes_when_running(monkeypatch):
    monkeypatch.setattr(proxy, "proxy_status", lambda: (True, 4242))
    result = doctor.check_proxy_process(port=4000)
    assert result.status == doctor.PASS
    assert "4242" in result.message


def test_check_proxy_process_warns_on_stale_pidfile_with_free_port(monkeypatch):
    monkeypatch.setattr(proxy, "proxy_status", lambda: (False, 4242))
    monkeypatch.setattr(doctor.procutil, "port_in_use", lambda port: False)
    result = doctor.check_proxy_process(port=4000)
    assert result.status == doctor.WARN
    assert "stale pidfile" in result.message


def test_check_proxy_process_fails_on_stale_pidfile_with_bound_port(monkeypatch):
    """Finding 5's exact silent-orphan scenario: the pidfile is stale but
    something is still bound to the port."""
    monkeypatch.setattr(proxy, "proxy_status", lambda: (False, 4242))
    monkeypatch.setattr(doctor.procutil, "port_in_use", lambda port: True)
    result = doctor.check_proxy_process(port=4000)
    assert result.status == doctor.FAIL
    assert "proxy down --force" in result.message


def test_check_proxy_process_warns_when_not_running_and_port_free(monkeypatch):
    monkeypatch.setattr(proxy, "proxy_status", lambda: (False, None))
    monkeypatch.setattr(doctor.procutil, "port_in_use", lambda port: False)
    result = doctor.check_proxy_process(port=4000)
    assert result.status == doctor.WARN
    assert "ys proxy up" in result.message


def test_check_proxy_process_warns_when_port_held_by_unknown_process(monkeypatch):
    monkeypatch.setattr(proxy, "proxy_status", lambda: (False, None))
    monkeypatch.setattr(doctor.procutil, "port_in_use", lambda port: True)
    result = doctor.check_proxy_process(port=4000)
    assert result.status == doctor.WARN
    assert "already bound" in result.message


# --- check_generated_config --------------------------------------------------


def test_check_generated_config_warns_when_missing():
    result = doctor.check_generated_config()
    assert result.status == doctor.WARN
    assert "ys proxy up" in result.message


def test_check_generated_config_fails_on_unparseable_yaml():
    paths.ensure_home()
    with open(paths.PROXY_CONFIG_PATH, "w") as f:
        f.write("model_list: [this is not: valid: yaml")
    result = doctor.check_generated_config()
    assert result.status == doctor.FAIL
    assert "does not parse" in result.message


def _write_generated_config(model_names):
    paths.ensure_home()
    import yaml as yaml_module

    config = {"model_list": [{"model_name": n, "litellm_params": {}} for n in model_names]}
    with open(paths.PROXY_CONFIG_PATH, "w") as f:
        yaml_module.safe_dump(config, f)


def test_check_generated_config_passes_with_no_experiment_scope():
    _write_generated_config(["claude-sonnet-5", "*"])
    result = doctor.check_generated_config()
    assert result.status == doctor.PASS
    assert "2 model(s)" in result.message


def test_check_generated_config_passes_when_arm_model_registered():
    _write_generated_config(["claude-sonnet-5", "*"])
    result = doctor.check_generated_config(_exp(), "arm-a")
    assert result.status == doctor.PASS
    assert "claude-sonnet-5" in result.message


def test_check_generated_config_warns_when_arm_model_missing():
    _write_generated_config(["some-other-model", "*"])
    result = doctor.check_generated_config(_exp(), "arm-a")
    assert result.status == doctor.WARN
    assert "no explicit entry for model 'claude-sonnet-5'" in result.message


def test_check_generated_config_fails_for_unknown_arm():
    _write_generated_config(["*"])
    result = doctor.check_generated_config(_exp(), "no-such-arm")
    assert result.status == doctor.FAIL


def test_check_generated_config_skips_arm_with_no_model_factor():
    _write_generated_config(["*"])
    exp = Experiment.model_validate(
        {
            "experiment": "e",
            "task": {"id": "t0", "success_check": "true"},
            "arms": [{"id": "a", "factors": {}, "baseline": True}],
        }
    )
    result = doctor.check_generated_config(exp, "a")
    assert result.status == doctor.SKIP


# --- check_task_paths (findings 15-18) --------------------------------------


def test_check_task_paths_skips_without_an_experiment():
    result = doctor.check_task_paths(None)
    assert result.status == doctor.SKIP


def test_check_task_paths_passes_when_nothing_declared():
    result = doctor.check_task_paths(_exp())
    assert result.status == doctor.PASS
    assert "nothing to check" in result.message


def test_check_task_paths_passes_when_prompt_file_exists(tmp_path):
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("do the thing")
    result = doctor.check_task_paths(_exp(prompt_file=str(prompt)))
    assert result.status == doctor.PASS
    assert "check out" in result.message


def test_check_task_paths_fails_when_prompt_file_missing(tmp_path):
    result = doctor.check_task_paths(_exp(prompt_file=str(tmp_path / "nope.txt")))
    assert result.status == doctor.FAIL
    assert "does not exist" in result.message
    assert "ys start" in result.message


# --- check_model_available (findings 3/29) -----------------------------------


def test_check_model_available_warns_and_reuses_skip_message_without_master_key():
    result = doctor.check_model_available("claude-sonnet-5", 4000, None)
    assert result.status == doctor.WARN
    assert result.message == proxy.model_check_skipped_message("claude-sonnet-5")


def test_check_model_available_passes_when_available(monkeypatch):
    monkeypatch.setattr(proxy, "model_available", lambda model, port, key: True)
    result = doctor.check_model_available("claude-sonnet-5", 4000, "sk-test")
    assert result.status == doctor.PASS


def test_check_model_available_fails_when_not_registered(monkeypatch):
    monkeypatch.setattr(proxy, "model_available", lambda model, port, key: False)
    result = doctor.check_model_available("claude-sonnet-5", 4000, "sk-test")
    assert result.status == doctor.FAIL
    assert "no explicit entry" in result.message


def test_check_model_available_warns_when_proxy_unreachable(monkeypatch):
    monkeypatch.setattr(proxy, "model_available", lambda model, port, key: None)
    result = doctor.check_model_available("claude-sonnet-5", 4000, "sk-test")
    assert result.status == doctor.WARN
    assert "could not reach the proxy" in result.message


# --- check_harness_config (finding 5's config-side sibling) -----------------


def test_check_harness_config_passes_when_config_does_not_exist():
    result = doctor.check_harness_config("claude-code")
    assert result.status == doctor.PASS
    assert "doesn't exist yet" in result.message


def test_check_harness_config_passes_when_not_pointed(fake_agents):
    with open(fake_agents["claude-code"], "w") as f:
        f.write("{}")
    result = doctor.check_harness_config("claude-code")
    assert result.status == doctor.PASS
    assert "not pointed" in result.message


def test_check_harness_config_passes_when_pointed_and_proxy_running(fake_agents, monkeypatch):
    harness.point("claude-code", 4000, "sk-test")
    monkeypatch.setattr(proxy, "proxy_status", lambda: (True, 123))
    result = doctor.check_harness_config("claude-code")
    assert result.status == doctor.PASS
    assert "proxy is running" in result.message


def test_check_harness_config_fails_when_pointed_and_proxy_not_running(fake_agents, monkeypatch):
    harness.point("claude-code", 4000, "sk-test")
    monkeypatch.setattr(proxy, "proxy_status", lambda: (False, None))
    result = doctor.check_harness_config("claude-code")
    assert result.status == doctor.FAIL
    assert "ys harness reset" in result.message


def test_check_harness_config_fails_for_unknown_agent():
    result = doctor.check_harness_config("not-a-real-agent")
    assert result.status == doctor.FAIL


def test_check_harness_config_passes_for_env_only_agent():
    """Feature 5 added agents with no config file at all (env_only=True,
    e.g. aider) -- harness.status() already reports config_exists=False for
    them rather than raising; this pins the nicer, dedicated message over
    the generic 'config doesn't exist yet at ' (with an empty path)."""
    result = doctor.check_harness_config("aider")
    assert result.status == doctor.PASS
    assert "env-only" in result.message


# --- project scope (feature 5) ----------------------------------------------


def test_check_harness_config_default_scope_name_omits_scope_suffix():
    """A single-scope agent's (or the default "user" scope's) check name is
    unchanged from before project scope existed."""
    result = doctor.check_harness_config("opencode")
    assert result.name == "harness config (opencode)"


def test_check_harness_config_project_scope_name_includes_scope():
    result = doctor.check_harness_config("claude-code", "project")
    assert result.name == "harness config (claude-code, project)"
    # this worktree has no .claude/settings.json at its root, so the
    # project-scope config simply doesn't exist -- still a PASS, not a FAIL.
    assert result.status == doctor.PASS


def test_check_harness_config_project_scope_fails_when_pointed_and_proxy_down(
    fake_agents, monkeypatch, tmp_path
):
    project_dir = tmp_path / "some-project"
    project_dir.mkdir()
    monkeypatch.chdir(project_dir)
    harness.point("claude-code", 4000, "sk-test", scope="project")
    monkeypatch.setattr(proxy, "proxy_status", lambda: (False, None))

    result = doctor.check_harness_config("claude-code", "project")
    assert result.status == doctor.FAIL
    assert "--scope project" in result.message


def test_run_checks_covers_both_scopes_for_claude_code_and_one_for_opencode():
    """Regression test: `run_checks` must drive `check_harness_config` from
    `harness.scopes_for_agent` (["user", "project"] for claude-code,
    ["user"] for opencode, [] -- checked once, scope-less -- for an
    env_only agent like aider), not hardcode a single "user" check per
    agent and silently miss project-scope drift."""
    results = doctor.run_checks()
    names = {r.name for r in results}
    assert "harness config (claude-code)" in names
    assert "harness config (claude-code, project)" in names
    assert "harness config (opencode)" in names
    assert "harness config (opencode, project)" not in names
    assert "harness config (aider)" in names


# --- check_api_keys ----------------------------------------------------------


def test_check_api_keys_pass_when_both_set(monkeypatch):
    monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-master")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-anthropic")
    results = doctor.check_api_keys()
    assert all(r.status == doctor.PASS for r in results)


def test_check_api_keys_warn_when_both_missing(monkeypatch):
    monkeypatch.delenv("LITELLM_MASTER_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    results = doctor.check_api_keys()
    assert all(r.status == doctor.WARN for r in results)
    names = {r.name for r in results}
    assert names == {"LITELLM_MASTER_KEY", "ANTHROPIC_API_KEY"}


# --- check_active_run --------------------------------------------------------


def test_check_active_run_passes_with_no_active_run():
    result = doctor.check_active_run()
    assert result.status == doctor.PASS
    assert "no active run" in result.message


def _exp_for_runs():
    return Experiment.model_validate(
        {
            "experiment": "doctor-runs-exp",
            "task": {"id": "t0", "success_check": "true"},
            "arms": [{"id": "only-arm", "factors": {}, "baseline": True}],
        }
    )


_RUNS_EXP_YAML = """
experiment: doctor-runs-exp
task:
  id: t0
  success_check: "true"
arms:
  - id: only-arm
    factors: {}
    baseline: true
"""


def test_check_active_run_passes_when_active_run_has_a_db_row():
    begun = runs.begin_run(_exp_for_runs(), _RUNS_EXP_YAML, "only-arm")
    result = doctor.check_active_run()
    assert result.status == doctor.PASS
    assert begun.run_id in result.message
    runs.finish_run()


def test_check_active_run_fails_when_db_row_missing():
    """The exact mismatch `ys end` raises ActiveRunMissingDbRow for --
    active.json points at a run id, but no such row exists."""
    state.set_active("ghost-run", "doctor-runs-exp", "only-arm", runs.now())
    result = doctor.check_active_run()
    assert result.status == doctor.FAIL
    assert "ActiveRunMissingDbRow" in result.message
    state.clear_active()


def test_check_active_run_fails_when_database_missing_entirely():
    state.set_active("ghost-run", "doctor-runs-exp", "only-arm", runs.now())
    os.remove(paths.DB_PATH)
    result = doctor.check_active_run()
    assert result.status == doctor.FAIL
    assert "no database exists" in result.message
    state.clear_active()


# --- check_unattributed (finding 12) -----------------------------------------


def test_check_unattributed_passes_when_none_recorded():
    result = doctor.check_unattributed()
    assert result.status == doctor.PASS


def test_check_unattributed_warns_when_present():
    with db.cursor() as cur:
        cur.execute(
            "INSERT OR IGNORE INTO experiments (id, name, question, task_json, config_yaml, created_at) "
            "VALUES ('unattributed', 'unattributed', NULL, '{}', '', '2026-01-01T14:02:33Z')"
        )
        cur.execute(
            "INSERT OR IGNORE INTO arms (id, experiment_id, label, factors_json, is_baseline) "
            "VALUES ('unattributed', 'unattributed', 'unattributed', '{}', 0)"
        )
        cur.execute(
            "INSERT OR IGNORE INTO runs (id, experiment_id, arm_id, repeat_idx, started_at) "
            "VALUES ('unattributed', 'unattributed', 'unattributed', 0, '2026-01-01T14:02:33Z')"
        )
        cur.execute(
            "INSERT INTO requests (run_id, seq, ts, status_code) VALUES ('unattributed', 1, "
            "'2026-01-01T14:02:33Z', 200)"
        )
    result = doctor.check_unattributed()
    assert result.status == doctor.WARN
    assert "14:02" in result.message


def test_check_unattributed_skips_when_no_database():
    os.remove(paths.DB_PATH)
    result = doctor.check_unattributed()
    assert result.status == doctor.SKIP


# --- check_dropped (finding 6) -----------------------------------------------


def test_check_dropped_passes_when_none_recorded():
    result = doctor.check_dropped()
    assert result.status == doctor.PASS


def test_check_dropped_warns_when_present():
    dropped.record("some-run", "database is locked")
    result = doctor.check_dropped()
    assert result.status == doctor.WARN
    assert "1 request(s)" in result.message


# --- run_checks composition ---------------------------------------------------


def test_run_checks_without_exp_arm_skips_model_availability():
    results = doctor.run_checks()
    model_check = next(r for r in results if r.name == "proxy serves model")
    assert model_check.status == doctor.SKIP
    assert "--exp/--arm" in model_check.message


def test_run_checks_requires_exp_and_arm_together_for_model_check():
    results = doctor.run_checks(exp="experiments/example.yaml")
    model_check = next(r for r in results if r.name == "proxy serves model")
    assert model_check.status == doctor.SKIP
    assert "must both be given" in model_check.message


def test_run_checks_with_exp_arm_includes_model_and_task_path_checks(monkeypatch):
    monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-test")
    monkeypatch.setattr(proxy, "model_available", lambda model, port, key: True)
    results = doctor.run_checks(exp="experiments/example.yaml", arm="arm-a")
    names = {r.name for r in results}
    assert "proxy serves model" in names
    assert "task paths" in names
    model_check = next(r for r in results if r.name == "proxy serves model")
    assert model_check.status == doctor.PASS


def test_run_checks_reports_bad_experiment_yaml():
    results = doctor.run_checks(exp="/no/such/file.yaml", arm="whatever")
    exp_check = next(r for r in results if r.name == "experiment YAML")
    assert exp_check.status == doctor.FAIL


def test_run_checks_all_pass_on_a_clean_slate_yields_zero_failures():
    results = doctor.run_checks()
    assert all(r.status != doctor.FAIL for r in results)


# --- read-only guarantee: ys doctor must never create YARDSTICK_HOME -------
#
# The module docstring's hard rule: "A user running `ys doctor` to find out
# *why* something is broken must never have the act of asking change the
# answer." tests/conftest.py's autouse isolated_yardstick_home fixture always
# calls db.init_db() before a test body runs, so YARDSTICK_HOME already
# exists by the time any other test starts -- these two point YARDSTICK_HOME
# at a directory that has deliberately never been created, so the assertion
# is meaningful. Before ys/harness.py's `_backup_dir()` gained its
# `create: bool = False` default, `check_harness_config` -> `harness.status`
# -> `_load_manifest` -> `_manifest_path` -> `_backup_dir` called
# `paths.ensure_home()` unconditionally on every call, so these fail against
# that code (asserting the directory is absent, when it would in fact have
# been created) and pass with the fix.


def _point_paths_at_a_never_created_home(monkeypatch, fresh_home):
    monkeypatch.setattr(paths, "YARDSTICK_HOME", str(fresh_home))
    monkeypatch.setattr(paths, "DB_PATH", str(fresh_home / "yardstick.db"))
    monkeypatch.setattr(paths, "DROPPED_LOG_PATH", str(fresh_home / "dropped_requests.jsonl"))
    monkeypatch.setattr(paths, "ACTIVE_RUN_PATH", str(fresh_home / "active.json"))
    monkeypatch.setattr(paths, "LAST_ENDED_RUN_PATH", str(fresh_home / "last_ended.json"))
    monkeypatch.setattr(paths, "PROXY_CONFIG_PATH", str(fresh_home / "proxy_config.yaml"))
    monkeypatch.setattr(paths, "PROXY_PID_PATH", str(fresh_home / "proxy.pid"))
    monkeypatch.setattr(paths, "PROXY_PORT_PATH", str(fresh_home / "proxy.port"))
    monkeypatch.setattr(paths, "PROXY_LOG_PATH", str(fresh_home / "proxy.log"))
    monkeypatch.setattr(paths, "EXPERIMENTS_DIR", str(fresh_home / "experiments"))


def test_run_checks_never_creates_yardstick_home_without_exp_arm(monkeypatch, tmp_path):
    fresh_home = tmp_path / "never-created-yardstick-home"
    assert not fresh_home.exists()
    _point_paths_at_a_never_created_home(monkeypatch, fresh_home)

    doctor.run_checks()

    assert not fresh_home.exists(), (
        "ys doctor created YARDSTICK_HOME just by running -- the diagnostic "
        "altered the thing it diagnoses"
    )


def test_run_checks_never_creates_yardstick_home_with_exp_arm(monkeypatch, tmp_path):
    """The harness-config loop (the leaky call path) runs regardless of
    whether --exp/--arm are given, so this pins the --exp/--arm branch too."""
    fresh_home = tmp_path / "never-created-yardstick-home"
    assert not fresh_home.exists()
    _point_paths_at_a_never_created_home(monkeypatch, fresh_home)
    monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-test")
    monkeypatch.setattr(proxy, "model_available", lambda model, port, key: True)

    doctor.run_checks(exp="experiments/example.yaml", arm="arm-a")

    assert not fresh_home.exists(), (
        "ys doctor --exp/--arm created YARDSTICK_HOME just by running -- the "
        "diagnostic altered the thing it diagnoses"
    )


def test_check_harness_config_alone_never_creates_yardstick_home(monkeypatch, tmp_path):
    """Narrower unit-level pin of the exact leak reported: harness.status()
    (called in a loop by check_harness_config, once per agent) used to
    create ~/.yardstick/harness_backups as a side effect of only checking
    whether a backup manifest existed."""
    fresh_home = tmp_path / "never-created-yardstick-home"
    assert not fresh_home.exists()
    _point_paths_at_a_never_created_home(monkeypatch, fresh_home)

    doctor.check_harness_config("claude-code")
    doctor.check_harness_config("opencode")

    assert not fresh_home.exists()
