"""ys compare / ys report --html: aggregate an experiment's runs by arm and
render a comparison, per spec section 8.

Known limitation: `experiments.config_yaml`/`task_json` are overwritten on
every `ys start` (see ys/cli.py), so there is no per-run historical snapshot
of which task.id/ref a given run actually executed. The "refuse to compare
runs with different task.id/ref" guardrail in the spec is therefore only
checked against the *current* experiment YAML passed to `ys compare`, not
against what each historical run actually saw. A full fix needs a schema
change (snapshot task_json onto the run row) that's out of scope here.
"""
import html
import statistics
from dataclasses import dataclass, field
from typing import Optional

from ys import metrics
from ys.experiment import Experiment

PRIMARY_METRICS = ["cost_usd", "billable_tokens", "turns", "wall_clock_s"]
SECONDARY_METRICS = [
    "tool_calls",
    "redundant_tool_calls",
    "tool_error_rate",
    "cache_read_ratio",
    "fixed_overhead_tokens",
    "context_high_water",
    "compaction_events",
]


class CompareError(Exception):
    pass


@dataclass
class ArmResult:
    arm_id: str
    label: str
    is_baseline: bool
    aggregate: dict
    fingerprint_drifted: bool
    run_ids: list = field(default_factory=list)


@dataclass
class Comparison:
    experiment_name: str
    task_id: str
    repeats: int
    arms: list  # list[ArmResult], baseline (if any) first


def _arm_row_id(experiment_name: str, arm_id: str) -> str:
    return f"{experiment_name}::{arm_id}"


def _run_ids_for_arm(cur, arm_row_id: str) -> list:
    rows = cur.execute(
        "SELECT id FROM runs WHERE arm_id = ? ORDER BY repeat_idx", (arm_row_id,)
    ).fetchall()
    return [r["id"] for r in rows]


def _fingerprint_drifted(cur, run_ids: list) -> bool:
    """True if model / toolset_hash / system_prompt_hash differ across this
    arm's repeats -- a broken control per spec section 4's guardrail."""
    if len(run_ids) < 2:
        return False
    rows = cur.execute(
        f"SELECT model, toolset_hash, system_prompt_hash FROM runs "
        f"WHERE id IN ({','.join('?' * len(run_ids))})",
        run_ids,
    ).fetchall()
    fingerprints = {
        (r["model"], r["toolset_hash"], r["system_prompt_hash"])
        for r in rows
        if r["model"] is not None  # unset = no requests logged yet, not a drift signal
    }
    return len(fingerprints) > 1


def compare_experiment(cur, experiment: Experiment) -> Comparison:
    """Aggregate every arm of `experiment` from already-recorded runs.

    Raises CompareError if no runs exist for any arm, or if the experiment
    row in the db (if present) was created for a different task.id -- the
    spec's "refuse mismatched task.id/ref" guardrail, checked against
    whatever's live in the YAML you're comparing (see module docstring for
    the historical-drift limitation this doesn't cover).
    """
    stored_task = cur.execute(
        "SELECT task_json FROM experiments WHERE id = ?", (experiment.experiment,)
    ).fetchone()
    if stored_task:
        import json

        stored = json.loads(stored_task["task_json"])
        if stored.get("id") and stored["id"] != experiment.task.id:
            raise CompareError(
                f"stored experiment '{experiment.experiment}' was last run with "
                f"task.id='{stored['id']}', but the YAML you're comparing now has "
                f"task.id='{experiment.task.id}'. Refusing to compare -- these are "
                f"not the same fixed task."
            )

    results = []
    for arm in experiment.arms:
        arm_row_id = _arm_row_id(experiment.experiment, arm.id)
        run_ids = _run_ids_for_arm(cur, arm_row_id)
        if not run_ids:
            continue
        aggregate = metrics.aggregate_run_metrics(cur, run_ids)
        results.append(
            ArmResult(
                arm_id=arm.id,
                label=arm.id,
                is_baseline=arm.baseline,
                aggregate=aggregate,
                fingerprint_drifted=_fingerprint_drifted(cur, run_ids),
                run_ids=run_ids,
            )
        )

    if not results:
        raise CompareError(
            f"no runs found for any arm of experiment '{experiment.experiment}'. "
            "Run `ys start` / `ys end` at least once per arm first."
        )

    results.sort(key=lambda r: (not r.is_baseline, r.label))
    return Comparison(
        experiment_name=experiment.experiment,
        task_id=experiment.task.id,
        repeats=experiment.repeats,
        arms=results,
    )


# ---------------------------------------------------------------------------
# Text rendering (ys compare)
# ---------------------------------------------------------------------------


def _fmt(value, spec: str = "") -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return format(value, spec or ".4g")
    return str(value)


def _delta_str(value: Optional[float], baseline: Optional[float]) -> str:
    if value is None or baseline is None or baseline == 0:
        return ""
    pct = (value - baseline) / abs(baseline) * 100
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.0f}%"


def build_table(comparison: Comparison):
    from rich.table import Table

    table = Table(title=f"{comparison.experiment_name}   task: {comparison.task_id}   repeats: {comparison.repeats}")
    table.add_column("metric")
    baseline = next((a for a in comparison.arms if a.is_baseline), None)
    for arm in comparison.arms:
        header = f"{arm.label}*" if arm.is_baseline else arm.label
        if arm.fingerprint_drifted:
            header += " [UNCONTROLLED]"
        table.add_column(header, justify="right")

    n_runs = {a.label: a.aggregate["n_runs"] for a in comparison.arms}
    n_success = {a.label: a.aggregate["n_success"] for a in comparison.arms}
    table.add_row(
        "success rate",
        *[f"{n_success[a.label]}/{n_runs[a.label]}" for a in comparison.arms],
    )
    table.add_row(
        "cost_per_success",
        *[_fmt(a.aggregate["cost_per_success"], ".4f") for a in comparison.arms],
    )

    for key in PRIMARY_METRICS + SECONDARY_METRICS:
        row = [key]
        base_mean = baseline.aggregate["metrics"].get(key, {}).get("mean") if baseline else None
        for arm in comparison.arms:
            stat = arm.aggregate["metrics"].get(key, {})
            mean = stat.get("mean")
            cell = _fmt(mean)
            if not arm.is_baseline:
                delta = _delta_str(mean, base_mean)
                if delta:
                    cell += f"  {delta}"
            row.append(cell)
        table.add_row(*row)

    return table


# ---------------------------------------------------------------------------
# HTML report (ys report --html)
# ---------------------------------------------------------------------------


def _sparkline_svg(series: list, width: int = 240, height: int = 40, color: str = "#4f7cff") -> str:
    if len(series) < 2:
        return "<em>not enough data</em>"
    lo, hi = min(series), max(series)
    span = (hi - lo) or 1
    step = width / (len(series) - 1)
    points = " ".join(
        f"{i * step:.1f},{height - ((v - lo) / span) * height:.1f}" for i, v in enumerate(series)
    )
    return (
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'role="img" aria-label="context tokens over turns">'
        f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{points}" /></svg>'
    )


def _context_series(cur, run_id: str) -> list:
    rows = cur.execute(
        "SELECT input_tokens, cache_creation, cache_read FROM requests WHERE run_id = ? ORDER BY seq",
        (run_id,),
    ).fetchall()
    return [
        (r["input_tokens"] or 0) + (r["cache_creation"] or 0) + (r["cache_read"] or 0) for r in rows
    ]


def _compaction_timeline_svg(cur, run_id: str, width: int = 240, height: int = 24) -> str:
    rows = cur.execute(
        "SELECT seq, transition FROM requests WHERE run_id = ? ORDER BY seq", (run_id,)
    ).fetchall()
    if not rows:
        return "<em>no requests</em>"
    n = len(rows)
    step = width / max(n, 1)
    colors = {
        "compaction": "#e0555f",
        "branch": "#e0a83f",
        "reset": "#9b59b6",
        "continuation": "#3fae5a",
    }
    marks = []
    for i, r in enumerate(rows):
        color = colors.get(r["transition"], "#c8c8c8")
        marks.append(f'<rect x="{i * step:.1f}" y="0" width="{max(step - 1, 1):.1f}" height="{height}" fill="{color}" />')
    return f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}">' + "".join(marks) + "</svg>"


def render_html(comparison: Comparison, cur) -> str:
    rows_html = []
    baseline = next((a for a in comparison.arms if a.is_baseline), None)

    uncontrolled = ' <span class="warn">UNCONTROLLED</span>'
    header_cells = "".join(
        f"<th>{html.escape(a.label)}{' *' if a.is_baseline else ''}"
        f"{uncontrolled if a.fingerprint_drifted else ''}</th>"
        for a in comparison.arms
    )

    def metric_row(key, fmt_spec=""):
        base_mean = baseline.aggregate["metrics"].get(key, {}).get("mean") if baseline else None
        cells = []
        for a in comparison.arms:
            stat = a.aggregate["metrics"].get(key, {})
            mean = stat.get("mean")
            n = stat.get("n")
            spread = stat.get("spread")
            cell = _fmt(mean, fmt_spec)
            if spread is not None and n and n > 1:
                cell += f' <span class="spread">± {_fmt(spread, fmt_spec)} (n={n})</span>'
            if not a.is_baseline:
                delta = _delta_str(mean, base_mean)
                if delta:
                    cell += f' <span class="delta">{delta}</span>'
            cells.append(f"<td>{cell}</td>")
        return f"<tr><th>{html.escape(key)}</th>{''.join(cells)}</tr>"

    rows_html.append(
        f"<tr><th>success rate</th>"
        + "".join(
            f"<td>{a.aggregate['n_success']}/{a.aggregate['n_runs']}</td>" for a in comparison.arms
        )
        + "</tr>"
    )
    rows_html.append(
        "<tr><th>cost_per_success</th>"
        + "".join(f"<td>{_fmt(a.aggregate['cost_per_success'], '.4f')}</td>" for a in comparison.arms)
        + "</tr>"
    )
    for key in PRIMARY_METRICS:
        rows_html.append(metric_row(key))
    for key in SECONDARY_METRICS:
        rows_html.append(metric_row(key))

    charts_html = []
    for a in comparison.arms:
        for run_id in a.run_ids:
            series = _context_series(cur, run_id)
            timeline = _compaction_timeline_svg(cur, run_id)
            spark = _sparkline_svg(series)
            charts_html.append(
                f'<div class="chart-card"><h3>{html.escape(a.label)} · {html.escape(run_id)}</h3>'
                f'<div class="chart-label">context tokens per turn</div>{spark}'
                f'<div class="chart-label">transition timeline</div>{timeline}</div>'
            )

    return f"""<title>{html.escape(comparison.experiment_name)} -- yardstick report</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: -apple-system, Segoe UI, sans-serif; max-width: 960px; margin: 2rem auto; padding: 0 1rem; }}
  h1 {{ font-size: 1.3rem; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
  th, td {{ text-align: right; padding: 0.35rem 0.6rem; border-bottom: 1px solid #8883; }}
  th:first-child, td:first-child {{ text-align: left; }}
  .warn {{ color: #e0555f; font-weight: 600; font-size: 0.75em; }}
  .delta {{ color: #888; font-size: 0.85em; }}
  .spread {{ color: #888; font-size: 0.8em; }}
  .chart-grid {{ display: flex; flex-wrap: wrap; gap: 1rem; }}
  .chart-card {{ border: 1px solid #8883; border-radius: 8px; padding: 0.75rem; }}
  .chart-label {{ font-size: 0.75rem; color: #888; margin-top: 0.5rem; }}
</style>
<h1>{html.escape(comparison.experiment_name)}</h1>
<p>task: {html.escape(comparison.task_id)} &middot; repeats: {comparison.repeats} &middot; (*) baseline</p>
<table><tr><th></th>{header_cells}</tr>{''.join(rows_html)}</table>
<h2>per-run detail</h2>
<div class="chart-grid">{''.join(charts_html)}</div>
"""
