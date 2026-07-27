"""Unattended runs (IMPROVEMENTS.md feature 1): `ys run --exp E --arm A
--repeats N` invokes a coding agent non-interactively, loops the repeats,
and scores each one -- turning yardstick from a logger a human drives by
hand into an experiment runner. `ys/cli.py`'s `run` command is a thin
wrapper around `run_experiment` below.

Command construction is deliberately isolated in one function,
`build_agent_command` -- none of the agent CLIs this module can invoke
(`claude`, `opencode`, `codex`, `aider`) are installed in this repo's test/
dev environment, so the exact invocation form for each is unverified
against a live binary. If one of these forms turns out to be wrong, that
function is the one place to fix, not a hunt through the runner.

Money-burning protections, all named directly in IMPROVEMENTS.md's feature 1
section:

  - `preflight` checks the agent's binary is on PATH, `task.prompt_file`
    exists, the proxy is reachable, and LITELLM_MASTER_KEY is set --
    *before* the first repeat starts, not discovered mid-loop on repeat 2.
  - `task.timeout_s` bounds every single agent invocation (`subprocess.run`
    timeout) exactly like it already bounds `success_check`.
  - `max_consecutive_failures` hard-stops the whole loop after that many
    agent-invocation failures *in a row* (nonzero exit, exception, or
    timeout) -- e.g. the proxy went down mid-run, or the agent can't
    authenticate. A task that runs fine but fails its own success_check is
    NOT a failure for this counter: that's a real experimental outcome, not
    a sign something is broken and retrying blindly.
  - `budget` (feature 6) totals the arm's real recorded spend after every
    repeat and stops before starting one that would go past the threshold
    -- see `check_budget` for why the guard is enforced here rather than
    only in `ys start`, and for how it stays honest about runs LiteLLM
    couldn't price.
  - `on_event` gives the caller (ys/cli.py) a line per repeat/phase so a
    human watching an overnight matrix can tell what's happening, not stare
    at a silent terminal for hours.

Harness pointing: prefers `harness.env_exports()` (feature 5's --env-only)
so the agent subprocess gets its own environment directly and the user's
real config file is never touched. Only falls back to `harness.point()`/
`harness.reset()` (once, around the whole loop -- not per repeat) for an
agent env_exports doesn't support yet (opencode, codex-cli).
"""
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from ys import db, harness, proxy, runs, state, workspace

# Binary each agent's CLI is invoked as -- keyed the same as ys/harness.py's
# AGENTS dict, so an --agent value that's valid for `ys harness point` is
# also valid here.
AGENT_BINARIES = {
    "claude-code": "claude",
    "opencode": "opencode",
    "codex-cli": "codex",
    "aider": "aider",
}

DEFAULT_MAX_CONSECUTIVE_FAILURES = 3
# Not the full 60s state.DRAIN_WINDOW_S -- see the comment at the sleep call
# site in run_experiment for why a short settle is enough here specifically.
DEFAULT_SETTLE_S = 2.0


class RunnerError(Exception):
    pass


def build_agent_command(agent_name: str, prompt: str) -> list:
    """The exact non-interactive invocation for each supported coding tool
    -- the single place in the codebase that constructs one (see the module
    docstring for why, and the PR body for what's verified vs. not).

      - claude-code: `claude -p <prompt>`    -- Claude Code's documented
        non-interactive "print mode": run one turn, print the result, exit.
        Named explicitly in IMPROVEMENTS.md's feature 1 text.
      - opencode:    `opencode run <prompt>` -- opencode's documented
        non-interactive single-shot command. Also named explicitly.
      - codex-cli:   `codex exec <prompt>`   -- Codex CLI's documented
        non-interactive "exec" subcommand. Also named explicitly.
      - aider:       `aider --message <prompt> --yes-always --no-auto-commits`
        -- NOT one of IMPROVEMENTS.md's three named examples; reconstructed
        from aider's public docs the same way ys/harness.py's codex-cli/
        aider support was, and equally unverified against a live install.
        `--yes-always` is required for a non-interactive loop (aider
        otherwise prompts before applying edits); `--no-auto-commits` keeps
        it from committing to git in the per-run workspace clone.
    """
    if agent_name == "claude-code":
        return ["claude", "-p", prompt]
    if agent_name == "opencode":
        return ["opencode", "run", prompt]
    if agent_name == "codex-cli":
        return ["codex", "exec", prompt]
    if agent_name == "aider":
        return ["aider", "--message", prompt, "--yes-always", "--no-auto-commits"]
    raise RunnerError(f"no non-interactive command form known for agent '{agent_name}'")


def check_agent_binary(agent_name: str) -> Optional[str]:
    """The resolved path if `agent_name`'s CLI binary is on PATH, else
    None. A real `shutil.which` call, not a fake -- tests monkeypatch this
    function itself (or `shutil.which`) rather than requiring any binary to
    actually be installed."""
    binary = AGENT_BINARIES.get(agent_name)
    if not binary:
        return None
    return shutil.which(binary)


def preflight(task, agent_name: str, master_key: str) -> list:
    """Checks that must pass before spending a single paid request. Returns
    a list of human-readable problems (empty if none), mirroring
    `ys/experiment.py`'s `validate_task_paths` -- return problems, let the
    caller decide how loudly to fail, rather than raising eagerly."""
    problems = []
    if agent_name not in AGENT_BINARIES:
        problems.append(f"unknown agent '{agent_name}'. Choose from: {', '.join(AGENT_BINARIES)}")
        return problems  # nothing else below is meaningful without a known agent

    if not task.prompt_file:
        problems.append(
            "task.prompt_file is required for `ys run` -- there's no human to type a "
            "prompt in an unattended run."
        )
    elif not os.path.isfile(task.prompt_file):
        problems.append(f"task.prompt_file '{task.prompt_file}' does not exist")

    if check_agent_binary(agent_name) is None:
        binary = AGENT_BINARIES[agent_name]
        problems.append(
            f"'{binary}' is not on PATH -- install the {agent_name} CLI before running "
            f"`ys run --agent {agent_name}`"
        )

    alive, _ = proxy.proxy_status()
    if not alive:
        problems.append("the proxy is not running -- start it with `ys proxy up` before `ys run`")

    if not master_key:
        problems.append(
            "LITELLM_MASTER_KEY is not set -- export the same key you started "
            "`ys proxy up` with"
        )

    return problems


@dataclass
class RunEvent:
    level: str  # "info" | "warning" | "error" | "success"
    message: str


@dataclass
class BudgetCheck:
    """One evaluation of `--budget` against what the arm has actually spent
    so far. `total_cost_usd` is a *floor* whenever `has_unknown_cost` is
    True -- see `check_budget` below."""
    total_cost_usd: float
    n_runs: int
    has_unknown_cost: bool
    over: bool
    message: str


def check_budget(experiment_name: str, arm_id: str, budget: float, spent_before_loop: float = 0.0) -> BudgetCheck:
    """Feature 6's budget guard, at the one place in this codebase where
    money is spent with nobody watching.

    `ys start --budget` can only check the arm's *already-finished*
    history, because `ys start` returns before the harness sends a single
    request (see `runs.arm_cost_summary`). `ys run` doesn't have that
    limitation for the repeats it drives itself: once a repeat's `ys end`
    equivalent has run, that repeat's real `cost_usd` is recorded, so the
    loop can total the arm's spend and stop *before* starting a repeat that
    would take it past the threshold. That's the version of the guard worth
    having, and it's why `--budget` is enforced between repeats here rather
    than only once up front.

    The budget is measured against the arm's whole recorded history, not
    just this invocation's repeats -- the same unit `ys start --budget`
    uses, so the two flags can't disagree about what "$5 for this arm"
    means. `spent_before_loop` is subtracted only to *report* how much of
    that total this loop is responsible for; it never changes the verdict.

    Honesty about unpriced runs (finding 9): `has_unknown_cost` means at
    least one counted run has a request neither LiteLLM nor a declared
    `pricing:` override could price, so `total_cost_usd` is a floor rather
    than a measurement. That still supports one sound conclusion -- a floor
    at or over the budget is definitely over the budget, so `over` is
    computed the same way either way -- but it can never support the
    opposite one. Under-budget with unknown cost present is reported as
    unverifiable rather than as "under budget", and the loop continues:
    refusing to run at all would make `--budget` unusable for exactly the
    models finding 9 is about (an unpriced model is the common case, and
    the fix for it is a `pricing:` block, not abandoning the run).
    """
    with db.cursor() as cur:
        summary = runs.arm_cost_summary(cur, experiment_name, arm_id)

    total = summary.total_cost_usd
    this_loop = total - spent_before_loop
    over = total >= budget
    floor_note = (
        " -- and that total is a FLOOR, not a measurement: at least one counted run has a "
        "request neither LiteLLM nor a declared `pricing:` override could price "
        "(cost_source='unknown', finding 9)"
    )

    if over:
        message = (
            f"budget guard: arm '{arm_id}' has recorded ${total:.2f} across "
            f"{summary.n_runs} finished run(s) (${this_loop:.2f} of it in this loop), at "
            f"or over the ${budget:.2f} budget"
        )
        if summary.has_unknown_cost:
            message += floor_note + ", so real spend is at least this much"
    elif summary.has_unknown_cost:
        message = (
            f"budget guard: arm '{arm_id}' has recorded ${total:.2f} of a ${budget:.2f} "
            f"budget across {summary.n_runs} finished run(s) (${this_loop:.2f} of it in "
            f"this loop){floor_note}. Cannot confirm this arm is under budget -- declare "
            "a `pricing:` block for the arm's model to make this guard enforceable"
        )
    else:
        message = (
            f"budget guard: ${total:.2f} of ${budget:.2f} spent on arm '{arm_id}' across "
            f"{summary.n_runs} finished run(s) (${this_loop:.2f} of it in this loop)"
        )

    return BudgetCheck(
        total_cost_usd=total,
        n_runs=summary.n_runs,
        has_unknown_cost=summary.has_unknown_cost,
        over=over,
        message=message,
    )


@dataclass
class RepeatOutcome:
    repeat: int
    run_id: Optional[str]
    invocation_ok: bool  # False = agent process itself failed (exit/timeout/exception)
    task_success: Optional[bool]  # None when no score was ever recorded for this repeat
    error: Optional[str] = None


@dataclass
class RunSummary:
    experiment_name: str
    arm_id: str
    repeats_requested: int
    outcomes: list = field(default_factory=list)
    aborted_reason: Optional[str] = None

    @property
    def repeats_completed(self) -> int:
        return len(self.outcomes)


def _finish_repeat(task, ws, run_id, repeat_num, invocation_ok, error, manual_score, on_event) -> RepeatOutcome:
    """Scores the repeat -- `success_check` in the workspace via
    `runs.finish_run(cwd=...)`, or a direct `manual_score` for a repeat
    that never got far enough to make success_check meaningful (workspace
    setup failed, task.setup failed, or the agent timed out) -- then always
    attempts teardown and cleanup regardless of the score. A failed repeat
    still left a workspace behind that needs the same handling as a
    successful one; teardown always runs before cleanup, since a teardown
    script operating on a directory that's already been rm -rf'd would be
    pointless."""
    task_success = None
    try:
        finish = runs.finish_run(manual_score=manual_score, cwd=(ws.path if ws else None))
        task_success = finish.task_success
    except runs.NoSuccessCheck as e:
        on_event(RunEvent("warning", f"repeat {repeat_num}: {e}"))
    except runs.NoActiveRun:
        pass

    if ws is not None:
        if task.teardown:
            try:
                teardown_result = workspace.run_teardown(task, ws)
                if teardown_result.returncode != 0:
                    on_event(RunEvent(
                        "warning",
                        f"repeat {repeat_num}: task.teardown exited "
                        f"{teardown_result.returncode} (non-fatal): "
                        f"{teardown_result.stderr[-300:]}",
                    ))
            except Exception as e:  # teardown must never abort the loop
                on_event(RunEvent("warning", f"repeat {repeat_num}: task.teardown raised: {e}"))
        try:
            workspace.cleanup_workspace(ws)
        except workspace.WorkspaceError as e:
            on_event(RunEvent("warning", f"repeat {repeat_num}: workspace cleanup refused: {e}"))

    on_event(RunEvent(
        "success" if task_success else "warning",
        f"repeat {repeat_num}: {'PASS' if task_success else 'FAIL'}",
    ))
    return RepeatOutcome(
        repeat=repeat_num, run_id=run_id, invocation_ok=invocation_ok,
        task_success=task_success, error=error,
    )


def _run_one_repeat(experiment, config_yaml, arm_id, task, agent_name, agent_env, prompt, repeat_num, on_event) -> RepeatOutcome:
    try:
        result = runs.begin_run(experiment, config_yaml, arm_id, force=False)
    except (runs.ArmNotFound, state.RunAlreadyActive) as e:
        # Shouldn't happen -- the loop always finishes one repeat's run
        # before starting the next -- but if it does, this repeat can't be
        # attributed to anything real; count it as an invocation failure so
        # the consecutive-failure guard still sees it.
        return RepeatOutcome(repeat=repeat_num, run_id=None, invocation_ok=False, task_success=None, error=str(e))

    run_id = result.run_id

    try:
        ws = workspace.prepare_workspace(task, run_id)
    except workspace.WorkspaceError as e:
        on_event(RunEvent("error", f"repeat {repeat_num}: workspace setup failed: {e}"))
        return _finish_repeat(
            task, None, run_id, repeat_num, invocation_ok=False,
            error=f"workspace: {e}", manual_score=0, on_event=on_event,
        )

    if task.setup:
        setup_result = workspace.run_setup(task, ws)
        if setup_result.returncode != 0:
            err = f"task.setup failed (exit {setup_result.returncode}): {setup_result.stderr[-500:]}"
            on_event(RunEvent("error", f"repeat {repeat_num}: {err}"))
            return _finish_repeat(
                task, ws, run_id, repeat_num, invocation_ok=False,
                error=err, manual_score=0, on_event=on_event,
            )

    cmd = build_agent_command(agent_name, prompt)
    on_event(RunEvent("info", f"repeat {repeat_num}: invoking {cmd[0]} ..."))
    env = dict(os.environ)
    env.update(agent_env)
    env["YS_RUN_ID"] = run_id

    invocation_ok = True
    error = None
    timed_out = False
    try:
        proc = subprocess.run(
            cmd, cwd=ws.path, env=env, timeout=task.timeout_s, capture_output=True, text=True,
        )
        if proc.returncode != 0:
            invocation_ok = False
            error = f"{cmd[0]} exited {proc.returncode}: {(proc.stderr or proc.stdout or '')[-500:]}"
    except FileNotFoundError as e:
        invocation_ok = False
        error = f"could not launch '{cmd[0]}': {e}"
    except subprocess.TimeoutExpired:
        invocation_ok = False
        timed_out = True
        error = f"{cmd[0]} did not finish within task.timeout_s={task.timeout_s}s"

    if not invocation_ok:
        on_event(RunEvent("error", f"repeat {repeat_num}: {error}"))

    # A timeout means the agent may still be actively mutating the
    # workspace when it was killed -- don't spend more time running
    # success_check against a tree in an unknown state, just score it
    # failed directly. A plain nonzero exit still gets scored normally: it
    # doesn't necessarily mean the task wasn't completed (or wasn't already
    # failed on its own merits before the agent process exited badly), and
    # the DB row is more informative with a real success_check result.
    manual_score = 0 if timed_out else None
    return _finish_repeat(
        task, ws, run_id, repeat_num, invocation_ok=invocation_ok,
        error=error, manual_score=manual_score, on_event=on_event,
    )


def run_experiment(
    experiment,
    config_yaml: str,
    arm_id: str,
    agent_name: str,
    repeats: int,
    port: int,
    master_key: str,
    max_consecutive_failures: int = DEFAULT_MAX_CONSECUTIVE_FAILURES,
    settle_s: float = DEFAULT_SETTLE_S,
    budget: Optional[float] = None,
    on_event: Callable[[RunEvent], None] = lambda evt: None,
) -> RunSummary:
    try:
        arm = experiment.get_arm(arm_id)
    except KeyError as e:
        raise RunnerError(str(e))

    task = experiment.task
    problems = preflight(task, agent_name, master_key)
    if problems:
        raise RunnerError("; ".join(problems))

    with open(task.prompt_file) as f:
        prompt = f.read()

    summary = RunSummary(experiment_name=experiment.experiment, arm_id=arm_id, repeats_requested=repeats)

    # The budget guard's pre-flight leg: checked before the harness is
    # pointed and before the first repeat, so an arm that's already over
    # its budget costs nothing at all -- not even a mutated config file to
    # reset afterwards.
    spent_before_loop = 0.0
    if budget is not None:
        pre = check_budget(experiment.experiment, arm_id, budget)
        spent_before_loop = pre.total_cost_usd
        if pre.over:
            summary.aborted_reason = pre.message + " -- refusing to start any repeat."
            on_event(RunEvent("error", summary.aborted_reason))
            return summary
        on_event(RunEvent("warning" if pre.has_unknown_cost else "info", pre.message))

    model = arm.factors.get("model")

    # Prefer --env-only pointing (feature 5): the agent subprocess gets its
    # own environment directly, and the user's real config file is never
    # touched -- exactly what an unattended loop spawning subprocesses
    # wants. Only falls back to point()/reset() for an agent env_exports
    # doesn't support yet (opencode, codex-cli); point() runs once before
    # the loop starts, not per repeat -- the same "stay pointed across
    # repeats" trade-off `ys end --keep-harness-pointed` makes explicit for
    # the manual flow.
    pointed_file_scope = None
    try:
        agent_env = harness.env_exports(agent_name, port, master_key, model=model)
    except harness.HarnessError:
        try:
            harness.point(agent_name, port, master_key, model=model, scope="user")
        except harness.HarnessError as e:
            raise RunnerError(f"could not point {agent_name} at the proxy: {e}")
        pointed_file_scope = "user"
        agent_env = {}

    consecutive_failures = 0

    try:
        for i in range(1, repeats + 1):
            on_event(RunEvent("info", f"repeat {i}/{repeats}: starting"))
            outcome = _run_one_repeat(
                experiment, config_yaml, arm_id, task, agent_name, agent_env, prompt, i, on_event,
            )
            summary.outcomes.append(outcome)

            if outcome.invocation_ok:
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                on_event(RunEvent(
                    "error",
                    f"repeat {i}/{repeats}: agent invocation failed -- "
                    f"{consecutive_failures}/{max_consecutive_failures} consecutive failures",
                ))
                if consecutive_failures >= max_consecutive_failures:
                    summary.aborted_reason = (
                        f"stopped after {consecutive_failures} consecutive agent-invocation "
                        f"failures (repeat {i}/{repeats}) -- see the last error above. This "
                        "is the guard against burning paid requests on a broken setup (e.g. "
                        "the proxy went down, or the agent can't authenticate)."
                    )
                    on_event(RunEvent("error", summary.aborted_reason))
                    break

            # The repeat just finished, so its own cost_usd is recorded and
            # the arm's running total is real -- check it before committing
            # to another repeat. On the final repeat there's nothing left to
            # stop, so an overage is reported but isn't an early abort.
            if budget is not None:
                check = check_budget(experiment.experiment, arm_id, budget, spent_before_loop)
                if check.over and i < repeats:
                    summary.aborted_reason = (
                        check.message + f" -- stopping before repeat {i + 1}/{repeats}."
                    )
                    on_event(RunEvent("error", summary.aborted_reason))
                    break
                on_event(RunEvent(
                    "warning" if (check.over or check.has_unknown_cost) else "info",
                    check.message,
                ))

            if i < repeats:
                # finding 11's drain window, raced by an automated loop:
                # `finish_run` (inside _run_one_repeat, just above) started
                # a state.DRAIN_WINDOW_S-second fallback for this repeat's
                # run, but the next line through the loop is about to
                # overwrite active.json for the *next* repeat. From that
                # moment on, any late, header-less response resolves to the
                # new run instead of falling through to the old one's drain
                # window -- silently *contaminating* repeat i+1 with repeat
                # i's tail, which is worse than finding 11's original
                # "unattributed" failure mode. The agent subprocess already
                # blocked until exit above, so anything it was itself still
                # waiting on has already landed; what's left is the proxy's
                # own async collector write (finding 6) lagging the HTTP
                # response by a moment. This short settle absorbs that
                # specific gap without paying the full drain window on
                # every single repeat.
                time.sleep(settle_s)
    finally:
        if pointed_file_scope:
            try:
                harness.reset(agent_name, scope=pointed_file_scope)
                on_event(RunEvent("info", f"{agent_name}: config reset"))
            except harness.HarnessError as e:
                on_event(RunEvent("warning", f"could not reset {agent_name}'s config: {e}"))

    return summary
