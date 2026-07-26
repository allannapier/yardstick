import json
import os
from typing import Optional

import typer
from rich.console import Console

from ys import db, harness, paths, proxy, runs, state, webserver
from ys.experiment import load_experiment

app = typer.Typer(help="yardstick -- measure agent/harness/model efficiency")
proxy_app = typer.Typer(help="manage the LiteLLM measurement proxy")
app.add_typer(proxy_app, name="proxy")
web_app = typer.Typer(help="manage the yardstick dashboard")
app.add_typer(web_app, name="web")
harness_app = typer.Typer(help="point/reset a coding agent's config at the proxy")
app.add_typer(harness_app, name="harness")

console = Console()


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
def proxy_down_cmd():
    """Stop the proxy."""
    console.print(proxy.proxy_down())


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

    for name in _agent_names(agent):
        try:
            path = harness.point(name, port, api_key, model=model)
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
def web_down_cmd():
    """Stop the dashboard."""
    console.print(webserver.web_down())


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

    try:
        result = runs.begin_run(experiment, config_yaml, arm, force=force)
    except runs.ArmNotFound as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    except state.RunAlreadyActive as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

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
        return
    console.print(json.dumps(active, indent=2))


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
