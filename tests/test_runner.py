import subprocess

import pytest

from ys import db, runner, state
from ys.experiment import Experiment


def _experiment(prompt_file, success_check="true", agent="claude-code", model=None, **task_overrides):
    factors = {}
    if agent:
        factors["agent"] = agent
    if model:
        factors["model"] = model
    task = {
        "id": "t0",
        "success_check": success_check,
        "timeout_s": 5,
        "prompt_file": str(prompt_file),
    }
    task.update(task_overrides)
    return Experiment.model_validate(
        {
            "experiment": "runner-test-exp",
            "task": task,
            "arms": [{"id": "only-arm", "factors": factors, "baseline": True}],
            "repeats": 1,
        }
    )


def _yaml_for(prompt_file, success_check="true", agent="claude-code", model=None, extra_task=""):
    model_line = f"    model: {model}\n" if model else ""
    agent_line = f"    agent: {agent}\n" if agent else ""
    return f"""
experiment: runner-test-exp
task:
  id: t0
  success_check: "{success_check}"
  timeout_s: 5
  prompt_file: {prompt_file}
{extra_task}
arms:
  - id: only-arm
    factors:
{agent_line}{model_line}
    baseline: true
repeats: 1
"""


@pytest.fixture
def prompt_file(tmp_path):
    p = tmp_path / "prompt.md"
    p.write_text("do the task\n")
    return p


@pytest.fixture(autouse=True)
def _ok_environment(monkeypatch):
    """Every test in this module gets a passing preflight (binary found,
    proxy alive) by default -- individual tests override one piece to prove
    the guard it's testing for. Keeps this separate from the isolated_harness_agents
    fixture in conftest.py, which only fakes the config *paths*, not proxy
    reachability or PATH lookups."""
    monkeypatch.setattr(runner, "check_agent_binary", lambda name: "/usr/bin/fake-" + name)
    monkeypatch.setattr(runner.proxy, "proxy_status", lambda: (True, 4242))


_REAL_SUBPROCESS_RUN = subprocess.run


def _agent_only(fake):
    """Wrap a fake so it only intercepts the actual agent-CLI invocation
    (`cmd[0]` in {claude, opencode, codex, aider}) and defers everything
    else -- `task.success_check` (ys/runs.py), `task.setup`/`task.teardown`,
    and `git clone`/`git checkout` (ys/workspace.py) -- to the real
    `subprocess.run`. Needed because `runner.subprocess` *is* the global
    `subprocess` module (not a copy), so monkeypatching `runner.subprocess.run`
    would otherwise also intercept those real, harmless local calls that
    other tests in this file rely on actually running."""
    def _run(cmd, *args, **kwargs):
        if isinstance(cmd, list) and cmd and cmd[0] in runner.AGENT_BINARIES.values():
            return fake(cmd, *args, **kwargs)
        return _REAL_SUBPROCESS_RUN(cmd, *args, **kwargs)
    return _run


def _fake_agent_run(returncode=0, stdout="ok", stderr=""):
    return _agent_only(
        lambda cmd, **k: subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr=stderr)
    )


# --- build_agent_command ------------------------------------------------


def test_build_agent_command_claude_code():
    assert runner.build_agent_command("claude-code", "do it") == ["claude", "-p", "do it"]


def test_build_agent_command_opencode():
    assert runner.build_agent_command("opencode", "do it") == ["opencode", "run", "do it"]


def test_build_agent_command_codex_cli():
    assert runner.build_agent_command("codex-cli", "do it") == ["codex", "exec", "do it"]


def test_build_agent_command_aider():
    cmd = runner.build_agent_command("aider", "do it")
    assert cmd[:2] == ["aider", "--message"]
    assert "do it" in cmd


def test_build_agent_command_unknown_agent_raises():
    with pytest.raises(runner.RunnerError):
        runner.build_agent_command("not-a-real-agent", "do it")


# --- preflight ------------------------------------------------------------


def test_preflight_passes_with_everything_in_place(prompt_file):
    exp = _experiment(prompt_file)
    problems = runner.preflight(exp.task, "claude-code", "sk-test")
    assert problems == []


def test_preflight_flags_missing_prompt_file(tmp_path):
    exp = _experiment(tmp_path / "does-not-exist.md")
    problems = runner.preflight(exp.task, "claude-code", "sk-test")
    assert any("prompt_file" in p for p in problems)


def test_preflight_flags_no_prompt_file_at_all():
    exp = Experiment.model_validate(
        {
            "experiment": "e",
            "task": {"id": "t0", "success_check": "true"},
            "arms": [{"id": "a", "factors": {}}],
        }
    )
    problems = runner.preflight(exp.task, "claude-code", "sk-test")
    assert any("prompt_file is required" in p for p in problems)


def test_preflight_flags_missing_binary(monkeypatch, prompt_file):
    monkeypatch.setattr(runner, "check_agent_binary", lambda name: None)
    exp = _experiment(prompt_file)
    problems = runner.preflight(exp.task, "claude-code", "sk-test")
    assert any("not on PATH" in p for p in problems)


def test_preflight_flags_proxy_down(monkeypatch, prompt_file):
    monkeypatch.setattr(runner.proxy, "proxy_status", lambda: (False, None))
    exp = _experiment(prompt_file)
    problems = runner.preflight(exp.task, "claude-code", "sk-test")
    assert any("proxy is not running" in p for p in problems)


def test_preflight_flags_missing_master_key(prompt_file):
    exp = _experiment(prompt_file)
    problems = runner.preflight(exp.task, "claude-code", "")
    assert any("LITELLM_MASTER_KEY" in p for p in problems)


def test_preflight_flags_unknown_agent(prompt_file):
    exp = _experiment(prompt_file)
    problems = runner.preflight(exp.task, "not-a-real-agent", "sk-test")
    assert any("unknown agent" in p for p in problems)


# --- run_experiment: preflight gates the whole loop ------------------------


def test_run_experiment_raises_before_any_repeat_when_preflight_fails(monkeypatch, prompt_file):
    monkeypatch.setattr(runner, "check_agent_binary", lambda name: None)
    exp = _experiment(prompt_file)
    called = []
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: called.append(1))

    with pytest.raises(runner.RunnerError):
        runner.run_experiment(
            exp, _yaml_for(prompt_file), "only-arm", agent_name="claude-code",
            repeats=3, port=4000, master_key="sk-test", settle_s=0,
        )

    assert called == []
    assert state.get_active() is None


def test_run_experiment_unknown_arm_raises(prompt_file):
    exp = _experiment(prompt_file)
    with pytest.raises(runner.RunnerError):
        runner.run_experiment(
            exp, _yaml_for(prompt_file), "no-such-arm", agent_name="claude-code",
            repeats=1, port=4000, master_key="sk-test", settle_s=0,
        )


# --- run_experiment: the repeat loop ---------------------------------------


def test_run_experiment_completes_all_repeats_on_success(monkeypatch, prompt_file):
    monkeypatch.setattr(runner.subprocess, "run", _fake_agent_run(returncode=0))
    exp = _experiment(prompt_file, success_check="true")

    summary = runner.run_experiment(
        exp, _yaml_for(prompt_file), "only-arm", agent_name="claude-code",
        repeats=3, port=4000, master_key="sk-test", settle_s=0,
    )

    assert summary.repeats_completed == 3
    assert summary.aborted_reason is None
    assert [o.task_success for o in summary.outcomes] == [True, True, True]
    assert all(o.invocation_ok for o in summary.outcomes)
    run_ids = {o.run_id for o in summary.outcomes}
    assert len(run_ids) == 3  # every repeat gets its own run
    assert state.get_active() is None  # loop always ends the last run


def test_run_experiment_records_task_failure_without_invocation_failure(monkeypatch, prompt_file):
    monkeypatch.setattr(runner.subprocess, "run", _fake_agent_run(returncode=0))
    exp = _experiment(prompt_file, success_check="false")

    summary = runner.run_experiment(
        exp, _yaml_for(prompt_file, success_check="false"), "only-arm", agent_name="claude-code",
        repeats=1, port=4000, master_key="sk-test", settle_s=0,
    )

    outcome = summary.outcomes[0]
    assert outcome.invocation_ok is True
    assert outcome.task_success is False


def test_run_experiment_nonzero_exit_is_an_invocation_failure_but_still_scored(monkeypatch, prompt_file):
    monkeypatch.setattr(runner.subprocess, "run", _fake_agent_run(returncode=1, stderr="boom"))
    exp = _experiment(prompt_file, success_check="true")

    summary = runner.run_experiment(
        exp, _yaml_for(prompt_file), "only-arm", agent_name="claude-code",
        repeats=1, port=4000, master_key="sk-test", settle_s=0,
        max_consecutive_failures=99,
    )

    outcome = summary.outcomes[0]
    assert outcome.invocation_ok is False
    assert "boom" in outcome.error
    # success_check still ran and still recorded a real result
    assert outcome.task_success is True


def test_run_experiment_timeout_skips_success_check_and_scores_failed(monkeypatch, prompt_file):
    def _timeout(cmd, **k):
        raise subprocess.TimeoutExpired(cmd, k.get("timeout"))

    monkeypatch.setattr(runner.subprocess, "run", _agent_only(_timeout))
    exp = _experiment(prompt_file, success_check="true")

    summary = runner.run_experiment(
        exp, _yaml_for(prompt_file), "only-arm", agent_name="claude-code",
        repeats=1, port=4000, master_key="sk-test", settle_s=0,
        max_consecutive_failures=99,
    )

    outcome = summary.outcomes[0]
    assert outcome.invocation_ok is False
    assert outcome.task_success is False
    assert "did not finish" in outcome.error


def test_run_experiment_missing_binary_mid_invocation_is_an_invocation_failure(monkeypatch, prompt_file):
    def _missing(cmd, **k):
        raise FileNotFoundError("no such file: claude")

    monkeypatch.setattr(runner.subprocess, "run", _agent_only(_missing))
    exp = _experiment(prompt_file, success_check="true")

    summary = runner.run_experiment(
        exp, _yaml_for(prompt_file), "only-arm", agent_name="claude-code",
        repeats=1, port=4000, master_key="sk-test", settle_s=0,
        max_consecutive_failures=99,
    )

    outcome = summary.outcomes[0]
    assert outcome.invocation_ok is False
    assert "could not launch" in outcome.error


# --- consecutive-failure guard ----------------------------------------------


def test_run_experiment_stops_after_max_consecutive_failures(monkeypatch, prompt_file):
    monkeypatch.setattr(runner.subprocess, "run", _fake_agent_run(returncode=1, stderr="down"))
    exp = _experiment(prompt_file, success_check="true")

    summary = runner.run_experiment(
        exp, _yaml_for(prompt_file), "only-arm", agent_name="claude-code",
        repeats=5, port=4000, master_key="sk-test", settle_s=0,
        max_consecutive_failures=2,
    )

    assert summary.repeats_completed == 2
    assert summary.aborted_reason is not None
    assert "2 consecutive" in summary.aborted_reason
    assert state.get_active() is None


def test_run_experiment_a_success_resets_the_consecutive_failure_counter(monkeypatch, prompt_file):
    calls = {"n": 0}

    def _flaky(cmd, **k):
        calls["n"] += 1
        # fail, succeed, fail, succeed, fail -- never two in a row, so a
        # max_consecutive_failures=2 guard must never trip.
        returncode = 1 if calls["n"] % 2 == 1 else 0
        return subprocess.CompletedProcess(cmd, returncode, stdout="", stderr="err")

    monkeypatch.setattr(runner.subprocess, "run", _agent_only(_flaky))
    exp = _experiment(prompt_file, success_check="true")

    summary = runner.run_experiment(
        exp, _yaml_for(prompt_file), "only-arm", agent_name="claude-code",
        repeats=5, port=4000, master_key="sk-test", settle_s=0,
        max_consecutive_failures=2,
    )

    assert summary.repeats_completed == 5
    assert summary.aborted_reason is None


# --- settle between repeats (finding 11 drain-window race) -----------------


def test_run_experiment_sleeps_settle_s_between_but_not_after_the_last_repeat(monkeypatch, prompt_file):
    monkeypatch.setattr(runner.subprocess, "run", _fake_agent_run(returncode=0))
    sleeps = []
    monkeypatch.setattr(runner.time, "sleep", lambda s: sleeps.append(s))
    exp = _experiment(prompt_file, success_check="true")

    runner.run_experiment(
        exp, _yaml_for(prompt_file), "only-arm", agent_name="claude-code",
        repeats=3, port=4000, master_key="sk-test", settle_s=1.5,
    )

    assert sleeps == [1.5, 1.5]  # between repeat 1->2 and 2->3, not after the 3rd


def test_run_experiment_does_not_sleep_after_an_aborting_failure(monkeypatch, prompt_file):
    monkeypatch.setattr(runner.subprocess, "run", _fake_agent_run(returncode=1, stderr="down"))
    sleeps = []
    monkeypatch.setattr(runner.time, "sleep", lambda s: sleeps.append(s))
    exp = _experiment(prompt_file, success_check="true")

    runner.run_experiment(
        exp, _yaml_for(prompt_file), "only-arm", agent_name="claude-code",
        repeats=5, port=4000, master_key="sk-test", settle_s=1.5,
        max_consecutive_failures=1,
    )

    assert sleeps == []  # aborted on repeat 1 -- no "next repeat" to settle before


# --- workspace failures feed the same consecutive-failure guard ------------


def test_run_experiment_workspace_failure_counts_as_invocation_failure(monkeypatch, prompt_file):
    called = []
    monkeypatch.setattr(runner.subprocess, "run", _agent_only(lambda cmd, **k: called.append(1)))
    exp = _experiment(prompt_file, success_check="true", repo="/no/such/repo/on/disk")

    summary = runner.run_experiment(
        exp, _yaml_for(prompt_file, extra_task="  repo: /no/such/repo/on/disk\n"),
        "only-arm", agent_name="claude-code",
        repeats=1, port=4000, master_key="sk-test", settle_s=0,
    )

    outcome = summary.outcomes[0]
    assert outcome.invocation_ok is False
    assert outcome.task_success is False
    assert "workspace" in outcome.error
    assert called == []  # never got as far as invoking the agent


# --- harness pointing: env-only preferred, file-based fallback reset -------


def test_run_experiment_uses_env_only_pointing_for_claude_code(monkeypatch, prompt_file):
    """claude-code supports --env-only -- run_experiment must prefer it and
    never call harness.point()/reset() (nothing to reset, nothing written
    to disk) for it."""
    from ys import harness

    point_calls = []
    reset_calls = []
    monkeypatch.setattr(harness, "point", lambda *a, **k: point_calls.append((a, k)))
    monkeypatch.setattr(harness, "reset", lambda *a, **k: reset_calls.append((a, k)))

    seen_env = {}

    def _capture(cmd, env=None, **k):
        seen_env.update(env or {})
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr(runner.subprocess, "run", _agent_only(_capture))
    exp = _experiment(prompt_file, success_check="true", model="claude-sonnet-5")

    runner.run_experiment(
        exp, _yaml_for(prompt_file, model="claude-sonnet-5"), "only-arm", agent_name="claude-code",
        repeats=1, port=4321, master_key="sk-test", settle_s=0,
    )

    assert point_calls == []
    assert reset_calls == []
    assert seen_env.get("ANTHROPIC_BASE_URL") == "http://localhost:4321"
    assert seen_env.get("ANTHROPIC_API_KEY") == "sk-test"
    assert seen_env.get("ANTHROPIC_MODEL") == "claude-sonnet-5"


def test_run_experiment_falls_back_to_point_and_reset_for_opencode(monkeypatch, prompt_file):
    """opencode has no verified --env-only mechanism (ys/harness.py raises
    HarnessError for it) -- run_experiment must fall back to point() once
    before the loop and reset() once after, and the agent subprocess must
    not have been handed a synthesized env (point() already wrote the
    config file instead)."""
    monkeypatch.setattr(runner.subprocess, "run", _fake_agent_run(returncode=0))
    exp = _experiment(prompt_file, success_check="true", agent="opencode")

    from ys import harness

    runner.run_experiment(
        exp, _yaml_for(prompt_file, agent="opencode"), "only-arm", agent_name="opencode",
        repeats=1, port=4000, master_key="sk-test", settle_s=0,
    )

    status = harness.status("opencode")
    # reset() restored the pre-point state -- the fake config never existed
    # before this test, so a successful reset means it's gone again.
    assert status.config_exists is False


def test_run_experiment_resets_file_based_harness_even_when_the_loop_aborts(monkeypatch, prompt_file):
    monkeypatch.setattr(runner.subprocess, "run", _fake_agent_run(returncode=1, stderr="down"))
    exp = _experiment(prompt_file, success_check="true", agent="opencode")

    from ys import harness

    runner.run_experiment(
        exp, _yaml_for(prompt_file, agent="opencode"), "only-arm", agent_name="opencode",
        repeats=5, port=4000, master_key="sk-test", settle_s=0,
        max_consecutive_failures=1,
    )

    status = harness.status("opencode")
    assert status.config_exists is False


# --- run_experiment: the budget guard (feature 6) --------------------------


def _fake_agent_spending(cost, cost_source="litellm"):
    """A fake agent invocation that also records what it 'spent'. The
    runner puts YS_RUN_ID in the subprocess environment, so writing one
    priced request row against it is exactly the shape of what a real
    agent's traffic leaves behind through the proxy's collector -- which is
    what makes the between-repeats check have anything real to total."""
    def _run(cmd, **kwargs):
        run_id = kwargs["env"]["YS_RUN_ID"]
        with db.cursor() as cur:
            cur.execute(
                "INSERT INTO requests (run_id, seq, ts, model, input_tokens, output_tokens, "
                "response_cost, cost_source) VALUES (?,1,?,?,?,?,?,?)",
                (run_id, "2026-01-01T00:00:00Z", "test-model", 10, 5, cost, cost_source),
            )
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    return _agent_only(_run)


def _events(sink):
    return "\n".join(e.message for e in sink)


def test_run_experiment_stops_between_repeats_once_the_budget_is_spent(monkeypatch, prompt_file):
    """Feature 6's budget guard at the place money actually burns: each
    repeat costs $1.00 against a $1.50 budget, so the loop must stop after
    repeat 2 rather than driving the agent a third time. Reverting the
    between-repeats check in run_experiment lets all 3 repeats run -- this
    test fails without it."""
    monkeypatch.setattr(runner.subprocess, "run", _fake_agent_spending(1.00))
    exp = _experiment(prompt_file)
    sink = []

    summary = runner.run_experiment(
        exp, _yaml_for(prompt_file), "only-arm", agent_name="claude-code",
        repeats=3, port=4000, master_key="sk-test", settle_s=0,
        budget=1.50, on_event=sink.append,
    )

    assert summary.repeats_completed == 2
    assert summary.aborted_reason is not None
    assert "stopping before repeat 3/3" in summary.aborted_reason
    assert "$2.00" in summary.aborted_reason and "$1.50" in summary.aborted_reason


def test_run_experiment_reports_the_running_total_after_every_repeat(monkeypatch, prompt_file):
    monkeypatch.setattr(runner.subprocess, "run", _fake_agent_spending(0.25))
    exp = _experiment(prompt_file)
    sink = []

    summary = runner.run_experiment(
        exp, _yaml_for(prompt_file), "only-arm", agent_name="claude-code",
        repeats=3, port=4000, master_key="sk-test", settle_s=0,
        budget=5.00, on_event=sink.append,
    )

    assert summary.repeats_completed == 3
    assert summary.aborted_reason is None
    output = _events(sink)
    assert "$0.25 of $5.00" in output
    assert "$0.50 of $5.00" in output
    assert "$0.75 of $5.00" in output


def test_run_experiment_refuses_to_start_a_single_repeat_when_already_over_budget(monkeypatch, prompt_file):
    """The pre-flight leg: an arm whose recorded history already meets the
    budget must not invoke the agent even once."""
    invoked = []

    def _spy(cmd, **kwargs):
        invoked.append(cmd)
        return _fake_agent_spending(3.00)(cmd, **kwargs)

    monkeypatch.setattr(runner.subprocess, "run", _agent_only(_spy))
    exp = _experiment(prompt_file)

    # One prior repeat spends $3.00 of a $2.00 budget...
    runner.run_experiment(
        exp, _yaml_for(prompt_file), "only-arm", agent_name="claude-code",
        repeats=1, port=4000, master_key="sk-test", settle_s=0,
    )
    assert len(invoked) == 1
    invoked.clear()

    # ...so a second `ys run` against the same arm never starts at all.
    summary = runner.run_experiment(
        exp, _yaml_for(prompt_file), "only-arm", agent_name="claude-code",
        repeats=2, port=4000, master_key="sk-test", settle_s=0, budget=2.00,
    )

    assert invoked == []
    assert summary.repeats_completed == 0
    assert "refusing to start any repeat" in summary.aborted_reason


def test_run_experiment_treats_an_unpriced_repeat_as_a_floor_not_a_measurement(monkeypatch, prompt_file):
    """Finding 9's honesty rule carried into the loop: a repeat whose
    requests couldn't be priced makes the running total a floor, so the
    guard must not report the arm as being under budget on the strength of
    an understated number."""
    monkeypatch.setattr(runner.subprocess, "run", _fake_agent_spending(0.0, cost_source="unknown"))
    exp = _experiment(prompt_file)
    sink = []

    summary = runner.run_experiment(
        exp, _yaml_for(prompt_file), "only-arm", agent_name="claude-code",
        repeats=2, port=4000, master_key="sk-test", settle_s=0,
        budget=5.00, on_event=sink.append,
    )

    output = _events(sink)
    assert summary.repeats_completed == 2  # unpriceable != unrunnable
    assert "FLOOR" in output
    assert "Cannot confirm this arm is under budget" in output
    assert [e for e in sink if e.level == "warning" and "budget guard" in e.message]


def test_run_experiment_over_budget_on_the_final_repeat_is_reported_not_an_abort(monkeypatch, prompt_file):
    """Nothing is left to stop once the last repeat has run, so blowing the
    budget there is reported without being dressed up as an early abort."""
    monkeypatch.setattr(runner.subprocess, "run", _fake_agent_spending(2.00))
    exp = _experiment(prompt_file)
    sink = []

    summary = runner.run_experiment(
        exp, _yaml_for(prompt_file), "only-arm", agent_name="claude-code",
        repeats=1, port=4000, master_key="sk-test", settle_s=0,
        budget=0.50, on_event=sink.append,
    )

    assert summary.repeats_completed == 1
    assert summary.aborted_reason is None
    assert "at or over the $0.50 budget" in _events(sink)


def test_run_experiment_without_a_budget_never_checks_one(monkeypatch, prompt_file):
    monkeypatch.setattr(runner.subprocess, "run", _fake_agent_spending(9.99))
    exp = _experiment(prompt_file)
    sink = []

    summary = runner.run_experiment(
        exp, _yaml_for(prompt_file), "only-arm", agent_name="claude-code",
        repeats=2, port=4000, master_key="sk-test", settle_s=0, on_event=sink.append,
    )

    assert summary.repeats_completed == 2
    assert "budget guard" not in _events(sink)
