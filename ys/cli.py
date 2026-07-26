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


def _agent_names(agent: str) -> list:
    if agent == "all":
        return list(harness.AGENTS)
    if agent not in harness.AGENTS:
        console.print(f"[red]unknown agent '{agent}'. Choose from: {', '.join(harness.AGENTS)}, all[/red]")
        raise typer.Exit(1)
    return [agent]


@harness_app.command("point")
def harness_point_cmd(
    agent: str = typer.Argument(..., help="claude-code | opencode | all"),
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

    for name in _agent_names(agent):
        try:
            path = harness.point(name, port, api_key, model=model, pin_background=pin_background)
        except harness.HarnessError as e:
            console.print(f"[red]{name}: {e}[/red]")
            raise typer.Exit(1)
        model_str = f", model={model}" if model else ""
        console.print(f"{name}: pointed at http://localhost:{port}{model_str} ({path})")


@harness_app.command("reset")
def harness_reset_cmd(
    agent: str = typer.Argument(..., help="claude-code | opencode | all"),
):
    """Restore an agent's config to what it was before `ys harness point`."""
    for name in _agent_names(agent):
        try:
            path = harness.reset(name)
        except harness.HarnessError as e:
            console.print(f"[red]{name}: {e}[/red]")
            raise typer.Exit(1)
        console.print(f"{name}: restored ({path})")


@harness_app.command("status")
def harness_status_cmd(
    agent: str = typer.Argument("all", help="claude-code | opencode | all"),
):
    """Show whether each agent is currently pointed at the proxy."""
    for name in _agent_names(agent):
        s = harness.status(name)
        state_str = "[green]pointed at proxy[/green]" if s.pointed_at_proxy else "not pointed"
        exists_str = "" if s.config_exists else " [yellow](config file doesn't exist)[/yellow]"
        backup_str = "backup available" if s.has_backup else "no backup yet"
        console.print(f"{s.agent}: {state_str}{exists_str}  ({backup_str})  -- {s.config_path}")


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


if __name__ == "__main__":
    app()
