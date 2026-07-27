import json
import os
import sqlite3
from typing import Optional

import typer
from rich.console import Console

from ys import db, dropped, harness, paths, proxy, runs, state, webserver
from ys.experiment import load_experiment, validate_task_paths

app = typer.Typer(help="yardstick -- measure agent/harness/model efficiency")
proxy_app = typer.Typer(help="manage the LiteLLM measurement proxy")
app.add_typer(proxy_app, name="proxy")
web_app = typer.Typer(help="manage the yardstick dashboard")
app.add_typer(web_app, name="web")
harness_app = typer.Typer(help="point/reset a coding agent's config at the proxy")
app.add_typer(harness_app, name="harness")

console = Console()


def _print_unattributed_notice():
    """Finding 12: requests the collector couldn't attribute to a real run
    land in the synthetic 'unattributed' run and, before this, were never
    surfaced anywhere -- a misconfigured harness produced a run with zero
    requests and no explanation in sight. Printed by both `ys status` and
    `ys end`, right alongside the `ys dropped` count (finding 6), which this
    is the sibling diagnostic of: dropped requests never made it into the
    database at all, unattributed ones did, just not under the run they
    belonged to."""
    summary = runs.unattributed_summary()
    if summary.count:
        console.print(
            f"\n[yellow]{summary.count} request(s) since {summary.since} UTC could not be "
            "attributed to a run (cumulative across every run ever recorded, not just "
            "this one) -- check the harness is pointed at the proxy and either sending "
            "x-ys-run or running while `ys start` has the active-run slot claimed.[/yellow]"
        )


def _auto_reset_pointed_harnesses():
    """Feature 5 in IMPROVEMENTS.md: `harness.point()` writes an API key in
    plaintext into a real config file, and before this nothing ever reset it
    automatically -- a crash between `ys start` and a manual `ys harness
    reset`, or simply forgetting that last step, left it there indefinitely.
    `ys end` now closes that window itself: for every agent with a config
    file at all (env-only agents like aider never wrote one), check every
    scope it could plausibly have been pointed at
    (`harness.scopes_for_agent`) and reset whichever ones currently look
    pointed at the proxy (`harness.status(...).pointed_at_proxy`) -- rather
    than requiring the caller to track which agent/scope combination it
    used. `end --keep-harness-pointed` opts out for a multi-repeat workflow
    that wants to stay pointed across several `ys start`/`ys end` cycles
    without re-running `ys harness point` before each one."""
    for name, spec in harness.AGENTS.items():
        if spec.env_only:
            continue
        for scope in harness.scopes_for_agent(name):
            try:
                s = harness.status(name, scope=scope)
            except harness.HarnessError:
                continue
            if not s.pointed_at_proxy:
                continue
            try:
                path = harness.reset(name, scope=scope)
            except harness.HarnessError:
                continue
            console.print(f"[dim]harness: reset {name} ({scope}) -- {path}[/dim]")


def _report_write_failed(e: Exception):
    """`runs.begin_run`/`finish_run`/`delete_run` write through
    `db.call_with_retry` (finding 28), which already retries a locked
    database `db.MAX_WRITE_ATTEMPTS` times -- if it still raises, the lock
    genuinely outlasted that, so surface a plain message instead of an
    unhandled sqlite3 traceback."""
    console.print(
        f"[red]could not write to the database after {db.MAX_WRITE_ATTEMPTS} "
        f"attempts ({e}) -- is another yardstick process (the proxy, another "
        "`ys` command) holding a long write against the same database "
        f"file?[/red]"
    )
    raise typer.Exit(1)


@app.command()
def init():
    """Create YARDSTICK_HOME and initialise the database."""
    db.init_db()
    console.print(f"yardstick home: [bold]{paths.YARDSTICK_HOME}[/bold]")
    console.print("database initialised")


@proxy_app.command("up")
def proxy_up_cmd(
    exp: list[str] = typer.Option(
        ..., "--exp", help="experiment YAML whose models to serve (repeatable)"
    ),
    port: int = typer.Option(proxy.DEFAULT_PORT, "--port"),
):
    """Generate a proxy config from experiments and start the proxy."""
    try:
        url = proxy.proxy_up(exp, port=port)
    except proxy.ProxyError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    console.print(f"proxy ready at [bold]{url}[/bold]")


@proxy_app.command("down")
def proxy_down_cmd(
    force: bool = typer.Option(
        False, "--force", help="send SIGKILL if the process doesn't stop from SIGTERM"
    ),
):
    """Stop the proxy."""
    console.print(proxy.proxy_down(force=force))


@proxy_app.command("status")
def proxy_status_cmd():
    """Report whether the proxy is running."""
    alive, pid = proxy.proxy_status()
    if alive:
        console.print(f"proxy running (pid {pid})")
    elif pid is not None:
        console.print(f"proxy not running (stale pidfile, pid {pid})")
    else:
        console.print("proxy not running")


def _agent_names(agent: str, skip_env_only: bool = False) -> list:
    if agent == "all":
        names = list(harness.AGENTS)
        if skip_env_only:
            for name in names:
                if harness.AGENTS[name].env_only:
                    console.print(
                        f"[yellow]{name}: skipped (env-only agent, no config file to "
                        f"point/reset -- run `ys harness point {name} --env-only` "
                        "directly)[/yellow]"
                    )
            names = [n for n in names if not harness.AGENTS[n].env_only]
        return names
    if agent not in harness.AGENTS:
        console.print(f"[red]unknown agent '{agent}'. Choose from: {', '.join(harness.AGENTS)}, all[/red]")
        raise typer.Exit(1)
    return [agent]


@harness_app.command("point")
def harness_point_cmd(
    agent: str = typer.Argument(..., help="claude-code | opencode | codex-cli | aider | all"),
    port: int = typer.Option(proxy.DEFAULT_PORT, "--port"),
    exp: Optional[str] = typer.Option(
        None, "--exp", help="experiment YAML -- combine with --arm to pin the agent's model"
    ),
    arm: Optional[str] = typer.Option(
        None, "--arm", help="arm id whose model factor to point the agent at"
    ),
    pin_background: bool = typer.Option(
        True,
        "--pin-background/--no-pin-background",
        help=(
            "also pin Claude Code's background small/fast-model env vars to the arm's "
            "model (default: on). This is what keeps a mock_response experiment a dry "
            "smoke test, but it also inflates cost_usd/billable_tokens relative to an "
            "unmeasured session -- pass --no-pin-background for a real cost comparison "
            "once the arm's model doesn't need mock_response to stay safe. See finding "
            "27 in IMPROVEMENTS.md."
        ),
    ),
    scope: str = typer.Option(
        "user",
        "--scope",
        help=(
            "'user' (default, e.g. ~/.claude/settings.json) or 'project' (e.g. "
            "./.claude/settings.json) -- project scope is only verified for claude-code; "
            "other agents raise an error rather than guess a path (feature 5 in "
            "IMPROVEMENTS.md)."
        ),
    ),
    env_only: bool = typer.Option(
        False,
        "--env-only",
        help=(
            "print `export` statements for the environment variables that would point "
            "this agent at the proxy, and never touch any config file at all -- safer "
            "than point()'s file-editing default, and the only way to point an agent "
            "with no config file (aider). See IMPROVEMENTS.md feature 5."
        ),
    ),
):
    """Point an agent's real config at the yardstick proxy (backs up the original first)."""
    api_key = os.environ.get("LITELLM_MASTER_KEY")
    if not api_key:
        console.print(
            "[red]LITELLM_MASTER_KEY is not set -- export the same key you started "
            "`ys proxy up` with.[/red]"
        )
        raise typer.Exit(1)

    if bool(exp) != bool(arm):
        console.print("[red]--exp and --arm must be given together.[/red]")
        raise typer.Exit(1)

    model = None
    if exp and arm:
        experiment = load_experiment(exp)
        try:
            arm_obj = experiment.get_arm(arm)
        except KeyError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1)
        model = arm_obj.factors.get("model")
        if not model:
            console.print(
                f"[yellow]warning: arm '{arm}' has no 'model' factor -- the agent's "
                "own default model id will be used, which the proxy likely hasn't "
                "registered.[/yellow]"
            )
    else:
        console.print(
            "[yellow]no --exp/--arm given -- the agent will request its own default "
            "model id, which the proxy may not have registered. Pass --exp and --arm "
            "to pin it to the arm's model.[/yellow]"
        )

    if model and not pin_background:
        console.print(
            "[yellow]--no-pin-background: background (small/fast model) requests will "
            "use the agent's own default, not the arm's model -- only safe if that "
            "model doesn't rely on mock_response to avoid hitting the real API "
            "(finding 27 in IMPROVEMENTS.md).[/yellow]"
        )

    for name in _agent_names(agent, skip_env_only=not env_only):
        if env_only:
            try:
                exports = harness.env_exports(name, port, api_key, model=model, pin_background=pin_background)
            except harness.HarnessError as e:
                if agent == "all":
                    # Don't let one agent with no verified env-only mechanism
                    # (opencode, codex-cli) abort the whole sweep -- report
                    # and move on, same as the skip_env_only note above does
                    # for point/reset.
                    console.print(f"[yellow]{name}: skipped ({e})[/yellow]")
                    continue
                console.print(f"[red]{name}: {e}[/red]")
                raise typer.Exit(1)
            console.print(f"# {name}: nothing written to disk -- export these yourself")
            for key, value in exports.items():
                console.print(f"export {key}={value}", highlight=False)
            continue
        try:
            path = harness.point(name, port, api_key, model=model, pin_background=pin_background, scope=scope)
        except harness.HarnessError as e:
            console.print(f"[red]{name}: {e}[/red]")
            raise typer.Exit(1)
        model_str = f", model={model}" if model else ""
        console.print(f"{name}: pointed at http://localhost:{port}{model_str} ({path})")


@harness_app.command("reset")
def harness_reset_cmd(
    agent: str = typer.Argument(..., help="claude-code | opencode | codex-cli | aider | all"),
    scope: str = typer.Option("user", "--scope", help="'user' or 'project' -- see `harness point --help`"),
):
    """Restore an agent's config to what it was before `ys harness point`."""
    for name in _agent_names(agent, skip_env_only=True):
        try:
            path = harness.reset(name, scope=scope)
        except harness.HarnessError as e:
            console.print(f"[red]{name}: {e}[/red]")
            raise typer.Exit(1)
        console.print(f"{name}: restored ({path})")


@harness_app.command("status")
def harness_status_cmd(
    agent: str = typer.Argument("all", help="claude-code | opencode | codex-cli | aider | all"),
    scope: str = typer.Option("user", "--scope", help="'user' or 'project' -- see `harness point --help`"),
):
    """Show whether each agent is currently pointed at the proxy."""
    for name in _agent_names(agent):
        if harness.AGENTS[name].env_only:
            console.print(
                f"{name}: env-only agent, nothing persisted -- run "
                f"`ys harness point {name} --env-only` for the export statements"
            )
            continue
        s = harness.status(name, scope=scope)
        state_str = "[green]pointed at proxy[/green]" if s.pointed_at_proxy else "not pointed"
        exists_str = "" if s.config_exists else " [yellow](config file doesn't exist)[/yellow]"
        backup_str = "backup available" if s.has_backup else "no backup yet"
        console.print(f"{s.agent} ({s.scope}): {state_str}{exists_str}  ({backup_str})  -- {s.config_path}")


@web_app.command("up")
def web_up_cmd(
    port: int = typer.Option(webserver.DEFAULT_PORT, "--port"),
):
    """Start the dashboard (experiment setup, run start/stop, results browsing)."""
    try:
        url = webserver.web_up(port=port)
    except webserver.WebServerError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    console.print(f"dashboard ready at [bold]{url}[/bold]")


@web_app.command("down")
def web_down_cmd(
    force: bool = typer.Option(
        False, "--force", help="send SIGKILL if the process doesn't stop from SIGTERM"
    ),
):
    """Stop the dashboard."""
    console.print(webserver.web_down(force=force))


@web_app.command("status")
def web_status_cmd():
    """Report whether the dashboard is running."""
    alive, pid = webserver.web_status()
    if alive:
        console.print(f"dashboard running (pid {pid})")
    elif pid is not None:
        console.print(f"dashboard not running (stale pidfile, pid {pid})")
    else:
        console.print("dashboard not running")


HEADLINE_METRICS = [
    "turns",
    "billable_tokens",
    "cost_usd",
    "cache_read_ratio",
    "context_high_water",
    "tool_calls",
    "tool_error_rate",
    "redundant_tool_calls",
    "compaction_events",
    "background_requests",
    "background_tokens",
    "active_s",
]


@app.command()
def start(
    exp: str = typer.Option(..., "--exp", help="path to the experiment YAML"),
    arm: str = typer.Option(..., "--arm", help="arm id to run"),
    force: bool = typer.Option(False, "--force", help="override an already-active run"),
    budget: Optional[float] = typer.Option(
        None,
        "--budget",
        help=(
            "USD budget guard for this arm (feature 6 in IMPROVEMENTS.md). `ys start` "
            "returns before the harness sends a single request, so this can't check the "
            "run about to happen -- it checks the arm's own recorded history instead: "
            "refuses to start (exit 1) if that history alone already meets or exceeds "
            "the budget, and warns rather than claiming 'under budget' if any of that "
            "history couldn't be priced at all (cost_source='unknown', finding 9). The "
            "run's own actual cost is still only known once it finishes -- see cost_usd "
            "in `ys end`'s printed summary, unchanged by this flag."
        ),
    ),
):
    """Begin a run of one arm and mark it active."""
    experiment = load_experiment(exp)
    with open(exp) as f:
        config_yaml = f.read()

    # Finding 15-18: task.prompt_file/repo are declared-but-unconsumed hooks
    # for features 1/2 (unattended runs, workspace isolation) -- but a
    # typo'd path should fail loudly right here, not silently once a future
    # feature finally reads it. Checked before begin_run claims the active
    # slot, so a rejected start doesn't leave anything to clean up.
    path_problems = validate_task_paths(experiment.task)
    if path_problems:
        for problem in path_problems:
            console.print(f"[red]{problem}[/red]")
        raise typer.Exit(1)

    # Feature 6: the budget guard. See the --budget help text above for why
    # this checks the arm's history rather than the run about to start --
    # a run that predicted the future would be a different, much bigger
    # feature (live enforcement inside the proxy/collector, which is out of
    # scope here). Checked before begin_run claims the active slot, same as
    # the path-problems check above, so a refusal doesn't leave anything to
    # clean up.
    if budget is not None:
        with db.cursor() as cur:
            cost_summary = runs.arm_cost_summary(cur, experiment.experiment, arm)
        if cost_summary.n_runs == 0:
            console.print(
                f"[dim]budget guard: no finished runs recorded yet for arm '{arm}' -- "
                f"nothing to check against the ${budget:.2f} budget until at least one "
                "finishes.[/dim]"
            )
        elif cost_summary.has_unknown_cost:
            console.print(
                f"[yellow]budget guard: arm '{arm}' has {cost_summary.n_runs} recorded "
                f"run(s) totalling ${cost_summary.total_cost_usd:.2f} against a "
                f"${budget:.2f} budget -- but at least one of those runs has a request "
                "neither LiteLLM nor a declared `pricing:` override could price "
                "(cost_source='unknown', finding 9), so that total is a floor, not a "
                "real total. Treating this as unknown, not 'under budget'.[/yellow]"
            )
        elif cost_summary.total_cost_usd >= budget:
            console.print(
                f"[red]budget guard: arm '{arm}' has already recorded "
                f"${cost_summary.total_cost_usd:.2f} across {cost_summary.n_runs} "
                f"finished run(s), at or over the ${budget:.2f} budget -- refusing to "
                "start another repeat.[/red]"
            )
            raise typer.Exit(1)
        else:
            console.print(
                f"[dim]budget guard: arm '{arm}' has recorded "
                f"${cost_summary.total_cost_usd:.2f} of a ${budget:.2f} budget across "
                f"{cost_summary.n_runs} finished run(s).[/dim]"
            )

    try:
        result = runs.begin_run(experiment, config_yaml, arm, force=force)
    except runs.ArmNotFound as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    except state.RunAlreadyActive as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    except (sqlite3.OperationalError, sqlite3.IntegrityError) as e:
        _report_write_failed(e)

    master_key = os.environ.get("LITELLM_MASTER_KEY")
    key_line = (
        f"export ANTHROPIC_API_KEY={master_key}"
        if master_key
        else "export ANTHROPIC_API_KEY=<the LITELLM_MASTER_KEY you exported for `ys proxy up`>"
    )

    console.print(
        f"\nrun [bold]{result.run_id}[/bold]  exp={result.experiment_name}  "
        f"arm={result.arm_id}  repeat={result.repeat_idx}\n"
    )

    model = experiment.get_arm(arm).factors.get("model")
    if model and master_key:
        port = proxy.read_port()
        available = proxy.model_available(model, port, master_key)
        if available is False:
            console.print(
                f"[red]warning: the proxy on port {port} has no explicit entry for "
                f"model '{model}' -- it will only work via the catch-all passthrough "
                "(mock_response/params declared for it won't apply). Run "
                f"`ys proxy up --exp {exp}` to register it.[/red]\n"
            )
        elif available is None:
            console.print(
                f"[yellow]warning: could not reach the proxy on port {port} to verify "
                f"it serves model '{model}' -- is `ys proxy up` running?[/yellow]\n"
            )
    elif model and not master_key:
        # Same check as above, just gated on this process's own environment
        # rather than the arm's model -- `ys proxy up` and `ys start` are
        # commonly run in separate shells, and without this the check
        # silently never runs at all (finding 29).
        console.print(f"[yellow]warning: {proxy.model_check_skipped_message(model)}[/yellow]\n")

    proxy_url = f"http://localhost:{proxy.read_port()}"
    console.print("point your harness at the proxy:\n")
    console.print(f"  export ANTHROPIC_BASE_URL={proxy_url}", highlight=False)
    console.print(f"  {key_line}", highlight=False)
    console.print(f"\noptional request header:  x-ys-run: {result.run_id}", highlight=False)
    console.print(
        "the header is the precise way to attribute traffic, but harnesses like "
        "Claude Code may not let you set arbitrary headers. Without it, requests "
        "are attributed via the active-run file this command just wrote, which is "
        "correct as long as only one run is active at a time.\n"
    )
    console.print("when the task is finished, run `ys end`")


@app.command()
def end(
    manual_score: Optional[float] = typer.Option(
        None, "--manual-score", help="record a score instead of running success_check"
    ),
    reset_harness: bool = typer.Option(
        True,
        "--reset-harness/--keep-harness-pointed",
        help=(
            "restore any agent config `ys harness point` touched, right after this run "
            "ends (default: on). This is what stops point()'s plaintext API key from "
            "lingering in a real ~/.claude/settings.json (or its project-level "
            "equivalent) any longer than one run, closing the gap finding 5's original "
            "harness-safety concern named: previously nothing reset it automatically at "
            "all, so a crash between `ys start` and a manual `ys harness reset` left it "
            "there indefinitely. Pass --keep-harness-pointed to stay pointed across "
            "repeat runs of the same arm without re-running `ys harness point` before "
            "each one -- the trade-off is the same plaintext-key exposure this flag "
            "otherwise closes (feature 5 in IMPROVEMENTS.md). --env-only agents have "
            "nothing to reset either way, since nothing was ever written to disk."
        ),
    ),
):
    """Finish the active run, score it, and print a summary."""
    try:
        if manual_score is None:
            console.print("running success check...")
        result = runs.finish_run(manual_score=manual_score)
    except runs.NoActiveRun:
        console.print("[red]no active run. Start one with `ys start`.[/red]")
        raise typer.Exit(1)
    except runs.ActiveRunMissingDbRow as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    except runs.NoSuccessCheck as e:
        console.print(f"[red]{e} (re-run with --manual-score).[/red]")
        raise typer.Exit(1)
    except (sqlite3.OperationalError, sqlite3.IntegrityError) as e:
        _report_write_failed(e)

    verdict = "[green]SUCCESS[/green]" if result.task_success else "[red]FAIL[/red]"
    console.print(
        f"\nrun [bold]{result.run_id}[/bold]  exp={result.experiment_name}  arm={result.arm_id}"
    )
    console.print(f"result: {verdict}   wall clock: {result.wall_clock_s:.1f}s")
    for key in HEADLINE_METRICS:
        value = result.summary_metrics.get(key)
        if value is None:
            continue
        if isinstance(value, float):
            value = f"{value:.4g}"
        console.print(f"  {key}: {value}")

    # finding 26: the largest thread (what turns/compaction/the fingerprint
    # are computed over) isn't always the conversation that started the run
    # -- a long-running Task subagent can out-issue it. A boolean doesn't
    # belong in the metrics table above, so it's a separate line.
    if result.summary_metrics.get("main_thread_started_run") is False:
        console.print(
            "\n[yellow]warning: the run's largest thread doesn't contain its "
            "first request -- turns, compaction metrics and the fingerprint "
            "were computed over a secondary thread (likely a Task subagent), "
            "not the conversation that started the run[/yellow]"
        )

    dropped_count = dropped.count()
    if dropped_count:
        console.print(
            f"\n[red]{dropped_count} request(s) could not be written to the database "
            f"and were dropped (cumulative across all runs recorded in "
            f"{paths.DROPPED_LOG_PATH}, not just this one)[/red]"
        )
    _print_unattributed_notice()

    if reset_harness:
        _auto_reset_pointed_harnesses()
    else:
        console.print(
            "\n[yellow]--keep-harness-pointed: any agent config `ys harness point` "
            "touched is left as-is -- remember `ys harness reset` before the plaintext "
            "key in it goes stale or gets left behind.[/yellow]"
        )


@app.command("run")
def run_cmd(
    exp: str = typer.Option(..., "--exp", help="path to the experiment YAML"),
    arm: str = typer.Option(..., "--arm", help="arm id to run"),
    repeats: Optional[int] = typer.Option(
        None, "--repeats", help="number of repeats (default: the experiment's own `repeats:`)"
    ),
    agent: Optional[str] = typer.Option(
        None,
        "--agent",
        help=(
            "claude-code | opencode | codex-cli | aider -- which non-interactive CLI to "
            "invoke each repeat with (default: the arm's factors.agent, if it has one)"
        ),
    ),
    port: int = typer.Option(proxy.DEFAULT_PORT, "--port"),
    max_consecutive_failures: Optional[int] = typer.Option(
        None,
        "--max-consecutive-failures",
        help=(
            "hard-stop after this many consecutive agent-invocation failures (not task "
            "failures -- a repeat that ran fine but failed its own success_check doesn't "
            "count). The guard against burning paid requests on a broken setup, e.g. the "
            "proxy going down mid-run. Default: 3."
        ),
    ),
    settle_s: Optional[float] = typer.Option(
        None,
        "--settle-s",
        help=(
            "pause between repeats so a straggling response from one repeat's agent can't "
            "be misattributed to the next repeat's run (finding 11's drain window, raced "
            "by a fast automated loop -- see IMPROVEMENTS.md feature 1). Default: 2.0."
        ),
    ),
    budget: Optional[float] = typer.Option(
        None,
        "--budget",
        help=(
            "USD budget guard for this arm (feature 6 in IMPROVEMENTS.md). Unlike "
            "`ys start --budget`, this loop sees the cost of the repeats it drives: it "
            "totals the arm's spend after every repeat and stops before starting one that "
            "would go past the budget. Measured against the arm's whole recorded history, "
            "not just this invocation, so it means the same thing as `ys start --budget`. "
            "If any counted run couldn't be priced (cost_source='unknown', finding 9) the "
            "total is reported as a floor and the guard says so rather than claiming "
            "'under budget'."
        ),
    ),
):
    """Drive an agent non-interactively through --repeats repeats of an arm's
    task, scoring each one (IMPROVEMENTS.md feature 1: unattended runs)."""
    from ys import runner

    experiment = load_experiment(exp)
    with open(exp) as f:
        config_yaml = f.read()

    # Finding 15-18 / feature 1&2: same filesystem checks `ys start` already
    # runs before claiming the active slot -- a typo'd prompt_file/repo
    # should fail loudly here too, before the loop even starts, not
    # mid-repeat. runner.preflight (below) additionally checks the agent
    # binary and the proxy, since run_experiment can in principle be called
    # without going through this CLI path first.
    path_problems = validate_task_paths(experiment.task)
    if path_problems:
        for problem in path_problems:
            console.print(f"[red]{problem}[/red]")
        raise typer.Exit(1)

    try:
        arm_obj = experiment.get_arm(arm)
    except KeyError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    agent_name = agent or arm_obj.factors.get("agent")
    if not agent_name:
        console.print(
            "[red]no --agent given and arm has no 'agent' factor -- pass --agent "
            "explicitly (claude-code | opencode | codex-cli | aider).[/red]"
        )
        raise typer.Exit(1)

    master_key = os.environ.get("LITELLM_MASTER_KEY") or ""
    repeats_n = repeats if repeats is not None else experiment.repeats
    kwargs = {}
    if max_consecutive_failures is not None:
        kwargs["max_consecutive_failures"] = max_consecutive_failures
    if settle_s is not None:
        kwargs["settle_s"] = settle_s
    if budget is not None:
        kwargs["budget"] = budget

    def _on_event(evt):
        style = {"info": "dim", "warning": "yellow", "error": "red", "success": "green"}.get(evt.level)
        console.print(f"[{style}]{evt.message}[/{style}]" if style else evt.message)

    try:
        summary = runner.run_experiment(
            experiment,
            config_yaml,
            arm,
            agent_name=agent_name,
            repeats=repeats_n,
            port=port,
            master_key=master_key,
            on_event=_on_event,
            **kwargs,
        )
    except runner.RunnerError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    n_success = sum(1 for o in summary.outcomes if o.task_success)
    n_fail = sum(1 for o in summary.outcomes if o.task_success is False)
    console.print(
        f"\n[bold]{summary.repeats_completed}/{summary.repeats_requested}[/bold] repeat(s) "
        f"attempted -- {n_success} succeeded, {n_fail} failed"
    )
    if summary.aborted_reason:
        console.print(f"[red]stopped early: {summary.aborted_reason}[/red]")
        raise typer.Exit(1)


@app.command()
def delete(
    run_id: str = typer.Argument(..., help="run id to delete"),
    yes: bool = typer.Option(False, "--yes", "-y", help="skip the confirmation prompt"),
):
    """Delete a run and its recorded requests/tool calls."""
    if not yes and not typer.confirm(f"delete run {run_id}? this cannot be undone"):
        raise typer.Exit(0)

    try:
        result = runs.delete_run(run_id)
    except runs.RunNotFound as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    except runs.CannotDeleteActiveRun as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    except (sqlite3.OperationalError, sqlite3.IntegrityError) as e:
        _report_write_failed(e)

    console.print(
        f"deleted run [bold]{result.run_id}[/bold]  "
        f"exp={result.experiment_name}  arm={result.arm_id}"
    )


@app.command()
def status():
    """Show the currently active run, if any."""
    active = state.get_active()
    if active is None:
        console.print("no active run")
    else:
        console.print(json.dumps(active, indent=2))

    dropped_count = dropped.count()
    if dropped_count:
        console.print(
            f"\n[red]{dropped_count} request(s) could not be written to the database "
            f"and were dropped (cumulative across all runs recorded in "
            f"{paths.DROPPED_LOG_PATH}, not just this one)[/red]"
        )
    _print_unattributed_notice()


@app.command()
def compare(
    exp: str = typer.Option(..., "--exp", help="path to the experiment YAML"),
):
    """Compare an experiment's arms from already-recorded runs."""
    from ys import render

    experiment = load_experiment(exp)
    with db.cursor() as cur:
        try:
            comparison = render.compare_experiment(cur, experiment)
        except render.CompareError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1)
        table = render.build_table(comparison)
    console.print(table)
    for warning in render.cost_warnings(comparison):
        console.print(f"[red]{warning}[/red]")
    for warning in render.config_warnings(comparison):
        console.print(f"[yellow]{warning}[/yellow]")
    # finding 15-18: `repeats:` is advisory -- unequal sample sizes across
    # arms invalidate the comparison in a way the table doesn't show on its
    # own. Yellow, not red: unlike a cost-unknown request, this doesn't mean
    # a number above is wrong, just that comparing it isn't apples-to-apples.
    for warning in render.repeat_count_warnings(comparison):
        console.print(f"[yellow]warning: {warning}[/yellow]")
    # Feature 3: the question a comparison table alone can't answer -- is a
    # difference real, or noise. One sentence per (arm, primary metric),
    # printed after the other banners since it's the verdict that follows
    # from the table, not a caveat about the table itself.
    verdicts = render.significance_verdicts(comparison)
    if verdicts:
        console.print("\n[bold]is the difference real?[/bold]")
        for verdict in verdicts:
            console.print(f"  {verdict}")


@app.command()
def report(
    exp: str = typer.Option(..., "--exp", help="path to the experiment YAML"),
    html: str = typer.Option(..., "--html", help="output path for the self-contained HTML report"),
):
    """Write a static self-contained HTML comparison report."""
    from ys import render

    experiment = load_experiment(exp)
    with db.cursor() as cur:
        try:
            comparison = render.compare_experiment(cur, experiment)
        except render.CompareError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1)
        content = render.render_html(comparison, cur)

    with open(html, "w") as f:
        f.write(content)
    console.print(f"wrote [bold]{html}[/bold]")


runs_app = typer.Typer(help="enumerate recorded runs")
app.add_typer(runs_app, name="runs")


@runs_app.command("list")
def runs_list_cmd(
    exp: Optional[str] = typer.Option(
        None,
        "--exp",
        help="path to the experiment YAML -- filters to this experiment and checks "
        "each run's config_hash against today's YAML (finding 14)",
    ),
    arm: Optional[str] = typer.Option(None, "--arm", help="arm id to filter to"),
    limit: Optional[int] = typer.Option(
        None, "--limit", help="show at most this many runs (most recent first)"
    ),
):
    """List recorded runs -- id, experiment, arm, start time, status and
    success -- so a run id can be found without the dashboard or raw SQL
    (P2 in IMPROVEMENTS.md)."""
    from rich.table import Table

    experiment_obj = None
    experiment_name = None
    if exp:
        experiment_obj = load_experiment(exp)
        experiment_name = experiment_obj.experiment

    rows = runs.list_runs(experiment=experiment_name, arm=arm, limit=limit, experiment_obj=experiment_obj)

    if not rows:
        console.print("no runs recorded")
        return

    table = Table()
    # overflow="fold" (wrap the cell onto another line) rather than Rich
    # Table's default ellipsis-truncation -- a run id or experiment name is
    # exactly the information this command exists to show; silently
    # truncating it on a narrow terminal would defeat the point.
    table.add_column("run id", overflow="fold")
    table.add_column("experiment", overflow="fold")
    table.add_column("arm", overflow="fold")
    table.add_column("repeat")
    table.add_column("started (UTC)", overflow="fold")
    table.add_column("status")
    table.add_column("success")
    table.add_column("config")

    status_style = {
        "finished": "finished",
        "unfinished": "[yellow]unfinished[/yellow]",
        "abandoned": "[red]abandoned[/red]",
    }
    for r in rows:
        success_str = "-" if r.success is None else ("yes" if r.success else "no")
        if r.config_current is None:
            config_str = "-"
        elif r.config_current:
            config_str = "[green]current[/green]"
        else:
            config_str = "[yellow]stale[/yellow]"
        table.add_row(
            r.run_id,
            r.experiment_id,
            r.arm_id,
            str(r.repeat_idx),
            r.started_at,
            status_style[r.status],
            success_str,
            config_str,
        )
    # A run id, an experiment name and a UTC timestamp per row don't fit
    # Rich's default 80-column fallback width without truncating the exact
    # information this command exists to show. `console.print(..., width=N)`
    # can only ever *narrow* Rich's own render (it clamps to
    # min(N, console.width), never widens it), so a real terminal still gets
    # its own actual width either way -- only when there's no terminal to
    # detect at all (piped/redirected output, or this file's own tests) does
    # a dedicated wider Console kick in, since 80 columns in that case is an
    # arbitrary fallback, not a real constraint worth truncating data for.
    render_console = console if console.is_terminal else Console(width=200)
    render_console.print(table)


@app.command()
def doctor(
    exp: Optional[str] = typer.Option(
        None,
        "--exp",
        help="experiment YAML -- combine with --arm to also verify the running "
        "proxy serves the arm's model",
    ),
    arm: Optional[str] = typer.Option(
        None, "--arm", help="arm id whose model factor to verify against the running proxy"
    ),
    port: Optional[int] = typer.Option(
        None,
        "--port",
        help="port to check the proxy on (default: the last `ys proxy up`/`ys proxy "
        "down` port, or 4000)",
    ),
):
    """Read-only preflight over every moving part -- home directory, schema
    version, proxy process, generated proxy config, task.prompt_file/repo,
    each harness's config, both API keys, active-run state, and
    unattributed/dropped request counts -- plus, with --exp/--arm, whether
    the running proxy serves that arm's model. Never writes to the
    database, starts/stops a process, runs a migration, or touches a
    harness config file. Exits non-zero if any check fails (Feature 4 in
    IMPROVEMENTS.md)."""
    from ys import doctor as doctor_mod

    if bool(exp) != bool(arm):
        console.print("[red]--exp and --arm must be given together.[/red]")
        raise typer.Exit(1)

    results = doctor_mod.run_checks(exp, arm, port)

    style = {
        doctor_mod.PASS: "[green]PASS[/green]",
        doctor_mod.WARN: "[yellow]WARN[/yellow]",
        doctor_mod.FAIL: "[red]FAIL[/red]",
        doctor_mod.SKIP: "[dim]SKIP[/dim]",
    }
    for r in results:
        console.print(f"{style[r.status]}  [bold]{r.name}[/bold]: {r.message}")

    n_fail = sum(1 for r in results if r.status == doctor_mod.FAIL)
    n_warn = sum(1 for r in results if r.status == doctor_mod.WARN)
    console.print(f"\n{len(results)} check(s): {n_fail} failed, {n_warn} warning(s)")
    if n_fail:
        raise typer.Exit(1)


@app.command(name="export")
def export_cmd(
    exp: str = typer.Option(..., "--exp", help="path to the experiment YAML"),
    arm: Optional[str] = typer.Option(None, "--arm", help="limit to one arm id (default: every arm)"),
    csv_path: Optional[str] = typer.Option(None, "--csv", help="write CSV to this path"),
    json_path: Optional[str] = typer.Option(None, "--json", help="write JSON to this path"),
):
    """Export every recorded run of an experiment (all repeats/arms,
    including unfinished/abandoned ones and runs from an older config) as
    CSV and/or JSON -- see ys/export.py for why the row is one run, not one
    arm-aggregate or one request."""
    from ys import export as export_mod

    if not csv_path and not json_path:
        console.print("[red]give at least one of --csv or --json.[/red]")
        raise typer.Exit(1)

    experiment = load_experiment(exp)
    with db.cursor() as cur:
        rows = export_mod.export_rows(cur, experiment, arm_id=arm)

    if not rows:
        console.print(
            f"[yellow]no recorded runs for experiment '{experiment.experiment}'"
            + (f" arm '{arm}'" if arm else "")
            + " -- writing an empty file.[/yellow]"
        )

    if csv_path:
        with open(csv_path, "w", newline="") as f:
            f.write(export_mod.to_csv(rows))
        console.print(f"wrote [bold]{csv_path}[/bold] ({len(rows)} row(s))")
    if json_path:
        with open(json_path, "w") as f:
            f.write(export_mod.to_json(rows))
        console.print(f"wrote [bold]{json_path}[/bold] ({len(rows)} row(s))")


@app.command()
def leaderboard(
    exp: list[str] = typer.Option(
        ..., "--exp", help="experiment YAML whose arms to include (repeatable across experiments/tasks)"
    ),
    metric: str = typer.Option("cost_usd", "--metric", help="metric to rank arms by, within each experiment"),
):
    """Rank arms across multiple experiments (different tasks) on one
    metric. Ranking is scoped within each experiment -- two experiments are
    two different tasks, so a mean on one isn't comparable in magnitude to
    a mean on the other -- and reuses feature 3's significance test so an
    arm that merely *looks* best isn't presented as a settled win when it
    isn't distinguishable from its own baseline's noise."""
    from ys import render

    comparisons = []
    errors = []
    for path in exp:
        experiment = load_experiment(path)
        with db.cursor() as cur:
            try:
                comparisons.append(render.compare_experiment(cur, experiment))
            except render.CompareError as e:
                errors.append(str(e))

    if not comparisons:
        for e in errors:
            console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    rows = render.build_leaderboard(comparisons, metric=metric)
    table = render.build_leaderboard_table(rows, metric)
    console.print(table)
    for e in errors:
        console.print(f"[yellow]{e}[/yellow]")
    notes = render.leaderboard_notes(rows)
    if notes:
        console.print("\n[bold]is the ranking real?[/bold]")
        for note in notes:
            console.print(f"  {note}")


if __name__ == "__main__":
    app()
