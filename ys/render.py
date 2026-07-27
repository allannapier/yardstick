"""ys compare / ys report --html: aggregate an experiment's runs by arm and
render a comparison, per spec section 8.

Finding 14 (fixed): `experiments.config_yaml`/`task_json` are overwritten on
every `ys start`, so the *experiment* row can never be a per-run record of
what a given run actually executed -- only the run row itself can be. Every
run now carries its own `config_hash` (`ys.runs.config_hash_for_arm`,
snapshotted at `begin_run`), and `compare_experiment` groups an arm's run
history by that hash instead of aggregating every run ever attributed to the
arm id. Only the group matching today's YAML is used by default; runs under
any other hash -- a different task/success_check/model, or (for a run
recorded before this fix shipped) no hash at all -- are excluded and
reported via `Comparison.config_warnings`/`config_warnings()` instead of
being silently folded in or silently dropped.
"""
import html
from dataclasses import dataclass, field
from typing import Optional

from ys import metrics
from ys import statistics as stats
from ys.experiment import Experiment
from ys.runs import config_hash_for_arm

# Fallbacks used when an experiment's `metrics:` block doesn't declare
# `primary`/`secondary`/`derived` (the default, empty lists) -- finding
# 15-18: these used to be the *only* lists this module consulted, so a
# carefully hand-written `metrics:` block in the YAML had no effect at all.
# `compare_experiment` now prefers the experiment's declared lists and only
# falls back to these, so existing YAMLs that already spell out the same
# names (both files under experiments/) see no change in what's displayed,
# but a YAML that declares something different now actually changes the
# table/report. There is no DEFAULT_DERIVED_METRICS -- before finding
# 15-18, nothing was ever displayed as "derived", so an experiment that
# doesn't declare `metrics.derived` keeps seeing exactly that (empty).
DEFAULT_PRIMARY_METRICS = ["cost_usd", "billable_tokens", "turns", "wall_clock_s"]
DEFAULT_SECONDARY_METRICS = [
    "tool_calls",
    "redundant_tool_calls",
    "tool_error_rate",
    "cache_read_ratio",
    "fixed_overhead_tokens",
    "context_high_water",
    "compaction_events",
    "background_requests",
    "background_tokens",
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
    # {"model": ..., "count": ...} entries with at least one request whose
    # cost LiteLLM couldn't price and no declared `pricing:` override could
    # price either -- finding 9. Non-empty means this arm's cost_usd/
    # cost_per_success are undercounted, not merely imprecise.
    unpriced_models: list = field(default_factory=list)


@dataclass
class Comparison:
    experiment_name: str
    task_id: str
    repeats: int
    arms: list  # list[ArmResult], baseline (if any) first
    # One line per arm with run history excluded from the table below
    # because it doesn't share today's config_hash -- finding 14. Printed
    # by `ys compare`/rendered in the HTML report the same way
    # `cost_warnings` is: prominently, not as a footnote.
    config_warnings: list = field(default_factory=list)
    # Resolved from `experiment.metrics.primary`/`.secondary`/`.derived`
    # (falling back to the DEFAULT_* lists above when undeclared) -- finding
    # 15-18. `build_table`/`render_html` iterate `display_metrics()` instead
    # of a hardcoded list, so a declared `metrics:` block actually changes
    # what gets displayed.
    primary_metrics: list = field(default_factory=lambda: list(DEFAULT_PRIMARY_METRICS))
    secondary_metrics: list = field(default_factory=lambda: list(DEFAULT_SECONDARY_METRICS))
    derived_metrics: list = field(default_factory=list)

    def display_metrics(self) -> list:
        """primary + secondary + derived, de-duplicated in that order and
        with `cost_per_success` dropped -- it's a valid metric name (an
        experiment can legitimately list it, see experiments/*.yaml's
        `derived:` block) but it's an aggregate-level ratio already
        rendered as its own fixed row above, not a per-run mean/spread, so
        repeating it in the generic metric-row loop would just print a
        bare '-' for it."""
        seen = set()
        out = []
        for key in self.primary_metrics + self.secondary_metrics + self.derived_metrics:
            if key == "cost_per_success" or key in seen:
                continue
            seen.add(key)
            out.append(key)
        return out


def _resolve_metric_list(declared_metrics, field_name: str, default: list) -> list:
    """`experiment.metrics.<field_name>` if the YAML actually declared that
    field, else `default`. A plain `value or default` fallback can't tell
    "the user wrote `secondary: []` on purpose, to show nothing there"
    apart from "the user never wrote a `secondary:` key at all" -- both
    parse to the same empty list. `model_fields_set` (populated by pydantic
    from the raw dict this sub-model was built from, independent of the
    parent) is what actually distinguishes them, so an explicit empty list
    is honoured instead of silently falling back to the default."""
    if field_name in declared_metrics.model_fields_set:
        return list(getattr(declared_metrics, field_name))
    return list(default)


def _arm_row_id(experiment_name: str, arm_id: str) -> str:
    return f"{experiment_name}::{arm_id}"


def _run_groups_for_arm(cur, arm_row_id: str) -> dict:
    """Every run ever recorded against this arm row id, grouped by
    `config_hash` (finding 14) and ordered by `repeat_idx` within each
    group. `None` is its own group: runs written before this fix shipped
    have no snapshot at all, so they can never be *verified* to match
    today's config -- see `compare_experiment` for how that group is
    treated (never trusted as "current", even if it's the only data an arm
    has)."""
    rows = cur.execute(
        "SELECT id, config_hash FROM runs WHERE arm_id = ? ORDER BY repeat_idx",
        (arm_row_id,),
    ).fetchall()
    groups: dict = {}
    for r in rows:
        groups.setdefault(r["config_hash"], []).append(r["id"])
    return groups


def _config_warning_for_arm(arm_id: str, groups: dict, current_hash: str, run_ids: list) -> Optional[str]:
    """finding 14: describe (if any) the runs excluded from `run_ids` --
    this arm's group matching `current_hash` -- because they were recorded
    under a different config_hash. Two reasons get named separately, since
    they call for different user action: the config actually changed
    (task/success_check/model), vs. the run simply predates this fix and
    was never given a hash to compare at all. Returns None when every run
    this arm has matches today's config."""
    excluded = sum(len(ids) for h, ids in groups.items() if h != current_hash)
    if excluded == 0:
        return None
    predates_snapshot = len(groups.get(None, []))
    changed_config = excluded - predates_snapshot
    reasons = []
    if changed_config:
        reasons.append(f"{changed_config} run(s) recorded under a different config (task/success_check/model changed since)")
    if predates_snapshot:
        reasons.append(f"{predates_snapshot} run(s) recorded before this config-tracking fix existed and can't be verified against today's config")
    reason_str = "; ".join(reasons)
    if run_ids:
        return (
            f"arm '{arm_id}': {reason_str} -- excluded below. Only the "
            f"{len(run_ids)} run(s) matching today's config are aggregated."
        )
    return (
        f"arm '{arm_id}': {reason_str} -- excluded below, and none of this "
        "arm's runs match today's config, so it has no comparable data. Run "
        "`ys start`/`ys end` for this arm again."
    )


def config_warnings(comparison: Comparison) -> list:
    """Same shape/precedent as `cost_warnings` -- a flat list of strings
    meant to be printed prominently alongside the table/report."""
    return comparison.config_warnings


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


def _unpriced_models_for_arm(cur, run_ids: list) -> list:
    """Merge `metrics.unpriced_models` across every run of an arm into one
    {"model", "count"} list -- finding 9. Counts are summed across repeats
    so a model unpriced in every run reads as clearly worse than one hit
    once."""
    counts: dict = {}
    for run_id in run_ids:
        for entry in metrics.unpriced_models(cur, run_id):
            counts[entry["model"]] = counts.get(entry["model"], 0) + entry["count"]
    return [{"model": model, "count": n} for model, n in sorted(counts.items())]


def compare_experiment(cur, experiment: Experiment) -> Comparison:
    """Aggregate every arm of `experiment` from already-recorded runs.

    Finding 14: each arm's run history is grouped by `config_hash`
    (`ys.runs.config_hash_for_arm`, snapshotted per run at `begin_run`) and
    only the group matching today's YAML is aggregated -- a run recorded
    before the task/success_check/model changed (or before this fix
    existed to snapshot anything at all) is excluded rather than silently
    pooled in with runs of a different config. `Comparison.config_warnings`
    names exactly what got excluded and why, per arm.

    Raises CompareError if no arm ends up with any runs matching its
    current config (this also covers "no runs recorded at all", the
    previous condition for this error).
    """
    # Plain-dict-ified once so `metrics.py` (which knows nothing about
    # pydantic) can look weights up per request's own model. See finding 10.
    billable_weights_by_model = {
        key: weights.model_dump() for key, weights in experiment.billable_weights.items()
    }

    # Finding 15-18: previously `aggregate_run_metrics` hardcoded the
    # task_success gate regardless of `metrics.gate`. `Experiment` already
    # validated the name against `VALID_GATE_NAMES` at YAML-load time, so
    # this can't raise on a well-formed `Experiment` -- but `resolve_gate`
    # still checks, rather than trusting that every caller went through
    # pydantic first.
    gate = metrics.resolve_gate(experiment.metrics.gate)

    results = []
    warnings = []
    for arm in experiment.arms:
        arm_row_id = _arm_row_id(experiment.experiment, arm.id)
        groups = _run_groups_for_arm(cur, arm_row_id)
        if not groups:
            continue  # arm has never been run at all -- nothing to warn about

        current_hash = config_hash_for_arm(experiment, arm)
        run_ids = groups.get(current_hash, [])

        warning = _config_warning_for_arm(arm.id, groups, current_hash, run_ids)
        if warning:
            warnings.append(warning)
        if not run_ids:
            continue

        aggregate = metrics.aggregate_run_metrics(
            cur, run_ids, gate=gate, billable_weights_by_model=billable_weights_by_model
        )
        results.append(
            ArmResult(
                arm_id=arm.id,
                label=arm.id,
                is_baseline=arm.baseline,
                aggregate=aggregate,
                fingerprint_drifted=_fingerprint_drifted(cur, run_ids),
                run_ids=run_ids,
                unpriced_models=_unpriced_models_for_arm(cur, run_ids),
            )
        )

    if not results:
        raise CompareError(
            f"no runs found for any arm of experiment '{experiment.experiment}' "
            "matching today's config. Run `ys start` / `ys end` at least once "
            "per arm first."
        )

    results.sort(key=lambda r: (not r.is_baseline, r.label))
    return Comparison(
        experiment_name=experiment.experiment,
        task_id=experiment.task.id,
        repeats=experiment.repeats,
        arms=results,
        config_warnings=warnings,
        primary_metrics=_resolve_metric_list(experiment.metrics, "primary", DEFAULT_PRIMARY_METRICS),
        secondary_metrics=_resolve_metric_list(experiment.metrics, "secondary", DEFAULT_SECONDARY_METRICS),
        derived_metrics=_resolve_metric_list(experiment.metrics, "derived", []),
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


# Metrics whose value is directly undercounted (not merely imprecise) when an
# arm has any unpriced request -- see finding 9. Both are derived straight
# from response_cost.
_COST_DERIVED_METRICS = ("cost_usd", "cost_per_success")
_COST_UNKNOWN_MARKER = " [COST UNKNOWN]"


def cost_warnings(comparison: Comparison) -> list:
    """One line per (arm, model) with at least one request LiteLLM couldn't
    price and that no declared `pricing:` override could price either
    (finding 9). Meant to be printed prominently alongside the table/report
    -- a request like this makes cost_usd/cost_per_success for that arm
    silently *wrong*, not just imprecise, so this is not a footnote."""
    lines = []
    for arm in comparison.arms:
        for entry in arm.unpriced_models:
            lines.append(
                f"cost unavailable for model '{entry['model']}' in arm '{arm.label}' "
                f"({entry['count']} request(s)) -- LiteLLM has no price for it and the "
                "experiment declares no `pricing:` override for it. cost_usd and "
                "cost_per_success for this arm are undercounted, not merely imprecise."
            )
    return lines


def repeat_count_warnings(comparison: Comparison) -> list:
    """Finding 15-18: `repeats:` is advisory -- nothing checked that every
    arm actually ended up with the same number of recorded runs. An arm
    with 7 runs and another with 1 still print as an ordinary mean +/-
    spread pair, with nothing in the table itself signalling that the two
    numbers don't rest on comparable sample sizes. Returns one line per arm
    only when the arms *disagree* with each other -- if every arm is short
    by the same amount (e.g. nobody's gotten past repeat 1 yet), that's a
    less misleading state than one arm racing ahead of the rest, so it's
    not flagged."""
    counts = {arm.label: arm.aggregate["n_runs"] for arm in comparison.arms}
    if len(set(counts.values())) <= 1:
        return []
    return [
        f"arm '{label}' has {n} recorded run(s) (experiment declares repeats: "
        f"{comparison.repeats}) -- arms in this comparison don't have equal sample "
        "sizes, so the means/spreads above are not an apples-to-apples comparison."
        for label, n in counts.items()
    ]


# ---------------------------------------------------------------------------
# Feature 3 -- statistics worth the name. `ys/statistics.py` does the actual
# math (bootstrap CIs, Wilson intervals, the exact permutation test, the
# minimum-detectable-effect helper); everything here is presentation only --
# turning those numbers into the sentence a user actually reads.
# ---------------------------------------------------------------------------

# Metric-specific direction words for the verdict line ("18% cheaper", not
# "18% lower", when the metric is a cost). The sentence always ends with
# "on {metric}" (e.g. "...on cost_usd"), so these are deliberately relational
# words rather than nouns that would repeat the metric name in the same
# sentence ("fewer turns ... on turns" reads worse than "fewer ... on
# turns"). Falls back to generic lower/higher for anything not listed --
# most of _EFFICIENCY_METRICS (redundancy rates, context high-water, etc.)
# reads fine that way and isn't worth a bespoke word.
_METRIC_DIRECTION_WORDS = {
    "cost_usd": ("cheaper", "more expensive"),
    "cost_per_success": ("cheaper", "more expensive"),
    "wall_clock_s": ("faster", "slower"),
    "active_s": ("faster", "slower"),
    "billable_tokens": ("fewer", "more"),
    "tokens_per_turn": ("fewer", "more"),
    "turns": ("fewer", "more"),
    "tool_calls": ("fewer", "more"),
}
_DEFAULT_DIRECTION_WORDS = ("lower", "higher")

# "fewer"/"more" are quantity words that need the thing being counted next to
# them -- "arm-b is 30% fewer than arm-a on turns" isn't a sentence. So the
# count metrics take a different clause shape ("arm-b uses 30% fewer turns
# than arm-a") that names the metric inline instead of trailing it with
# "on {metric}". Everything else reads correctly in the relational form
# ("18% cheaper than ... on cost_usd") and keeps it.
_COUNT_METRICS = frozenset({"billable_tokens", "tokens_per_turn", "turns", "tool_calls"})


def _direction_word(metric: str, relative_effect: Optional[float]) -> str:
    lower_word, higher_word = _METRIC_DIRECTION_WORDS.get(metric, _DEFAULT_DIRECTION_WORDS)
    if relative_effect is None or relative_effect >= 0:
        return higher_word
    return lower_word


def _effect_clause(metric: str, arm_label: str, baseline_label: str, direction: str, pct_str: str) -> str:
    """The "arm-b is 18% cheaper than arm-a on cost_usd" opening of a verdict
    line, in whichever of the two shapes the metric's direction word needs."""
    if metric in _COUNT_METRICS:
        return f"{arm_label} uses {pct_str} {direction} {metric} than {baseline_label}"
    return f"{arm_label} is {pct_str} {direction} than {baseline_label} on {metric}"


def _format_metric_verdict(metric: str, arm_label: str, baseline_label: str, v: stats.MetricVerdict) -> str:
    """The plan's own example, in the plan's own words: "arm-b is 18%
    cheaper, but with n=3 the difference is not distinguishable from noise;
    ~12 repeats needed." Always names direction, effect size, whether it's
    distinguishable from noise, and (when not) how many repeats would be
    needed -- never a bare p-value standing in for that judgment, and never
    silent about *why* a difference this size isn't claimed as real."""
    if v.relative_effect is None:
        pct_str = "an unmeasured amount (baseline mean is 0)"
    else:
        pct_str = f"{abs(v.relative_effect) * 100:.0f}%"
    direction = _direction_word(metric, v.relative_effect)
    n_desc = f"n={v.n_baseline}" if v.n_baseline == v.n_arm else f"n={v.n_baseline} vs {v.n_arm}"
    perm = v.permutation

    effect = _effect_clause(metric, arm_label, baseline_label, direction, pct_str)

    if v.significant:
        return (
            f"{effect} ({n_desc}, permutation p={perm.p_value:.3g}) "
            "-- distinguishable from noise."
        )

    if perm.exact and perm.min_possible_p is not None and perm.min_possible_p >= stats.DEFAULT_ALPHA:
        # The enclosing sentence already opens with "but with {n_desc}", so
        # this clause deliberately doesn't repeat the n ("but with n=3 the
        # smallest p-value ... at n=3 is 0.1" said it twice).
        noise_clause = (
            f"the smallest possible p-value an exact test could report is "
            f"{perm.min_possible_p:.2g}, so no result here could reach significance"
        )
    else:
        noise_clause = f"the difference is not distinguishable from noise (permutation p={perm.p_value:.3g})"

    if v.required_repeats is not None:
        repeats_clause = f"~{v.required_repeats} repeats needed to tell a difference this size from noise"
    else:
        repeats_clause = "not enough repeats yet to estimate how many would be needed"

    return f"{effect}, but with {n_desc} {noise_clause}; {repeats_clause}."


def significance_verdicts(comparison: Comparison) -> list:
    """One verdict sentence per (non-baseline arm, primary metric) pair --
    the feature-3 deliverable. Uses `comparison.primary_metrics`
    (`metrics.primary` in the YAML, see finding 15-18) to decide which
    metric(s) to run the significance test on, exactly as the plan
    specifies, rather than inventing a separate mechanism. Empty when there
    is no baseline arm to compare against, or the experiment declares no
    primary metrics at all."""
    baseline = next((a for a in comparison.arms if a.is_baseline), None)
    if baseline is None or not comparison.primary_metrics:
        return []
    lines = []
    for metric in comparison.primary_metrics:
        baseline_values = baseline.aggregate["metrics"].get(metric, {}).get("values") or []
        for arm in comparison.arms:
            if arm.is_baseline:
                continue
            arm_values = arm.aggregate["metrics"].get(metric, {}).get("values") or []
            verdict = stats.metric_verdict(baseline_values, arm_values)
            if verdict is None:
                continue
            lines.append(_format_metric_verdict(metric, arm.label, baseline.label, verdict))
    return lines


def _success_rate_cell(n_success: int, n_runs: int) -> str:
    """success rate cell text with a Wilson 95% CI appended once there's at
    least one run to bound -- the plan's third deliverable. Wilson, not the
    normal approximation, specifically because a 3/3 or 0/3 run (entirely
    unremarkable at n=3) would otherwise report a zero-width interval that
    reads as far more certain than three observations can support."""
    base = f"{n_success}/{n_runs}"
    if n_runs <= 0:
        return base
    w = stats.wilson_interval(n_success, n_runs)
    if w is None:
        return base
    return f"{base} (95% CI {w.low * 100:.0f}-{w.high * 100:.0f}%)"


def _ci_suffix(values: list, fmt_spec: str = "") -> str:
    """`ys.statistics.bootstrap_ci`'s interval, formatted to append after a
    metric cell's mean/spread -- the plan's first deliverable. Empty string
    (not "-") when there aren't enough observations to bootstrap, so it
    silently doesn't clutter a `repeats: 1` cell instead of printing a
    degenerate interval."""
    ci = stats.bootstrap_ci(values)
    if ci is None:
        return ""
    return f" CI95[{_fmt(ci.low, fmt_spec)}, {_fmt(ci.high, fmt_spec)}]"


def build_table(comparison: Comparison):
    from rich.table import Table

    table = Table(title=f"{comparison.experiment_name}   task: {comparison.task_id}   repeats: {comparison.repeats}")
    table.add_column("metric")
    baseline = next((a for a in comparison.arms if a.is_baseline), None)
    for arm in comparison.arms:
        header = f"{arm.label}*" if arm.is_baseline else arm.label
        if arm.fingerprint_drifted:
            header += " [UNCONTROLLED]"
        if arm.unpriced_models:
            header += _COST_UNKNOWN_MARKER
        table.add_column(header, justify="right")

    n_runs = {a.label: a.aggregate["n_runs"] for a in comparison.arms}
    n_success = {a.label: a.aggregate["n_success"] for a in comparison.arms}
    n_unfinished = {a.label: a.aggregate.get("n_unfinished", 0) for a in comparison.arms}
    table.add_row(
        "success rate",
        *[_success_rate_cell(n_success[a.label], n_runs[a.label]) for a in comparison.arms],
    )
    if any(n_unfinished.values()):
        # finding 13: runs displaced by `--force` (or otherwise never
        # `ys end`ed) are excluded from n_runs/success rate above rather
        # than silently counting against the arm -- but still reported, so
        # "excluded" doesn't mean "invisible".
        table.add_row(
            "unfinished (excluded)",
            *[str(n_unfinished[a.label]) for a in comparison.arms],
        )
    table.add_row(
        "cost_per_success",
        *[
            _fmt(a.aggregate["cost_per_success"], ".4f")
            + (_COST_UNKNOWN_MARKER if a.unpriced_models else "")
            for a in comparison.arms
        ],
    )

    for key in comparison.display_metrics():
        row = [key]
        base_mean = baseline.aggregate["metrics"].get(key, {}).get("mean") if baseline else None
        for arm in comparison.arms:
            stat = arm.aggregate["metrics"].get(key, {})
            mean = stat.get("mean")
            cell = _fmt(mean)
            if key in _COST_DERIVED_METRICS and arm.unpriced_models:
                cell += _COST_UNKNOWN_MARKER
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
    # Main thread only, matching token_metrics -- interleaved background/
    # subagent requests have their own short, unrelated context and would
    # otherwise show up as spurious drops in the conversation's chart.
    return [metrics.context_tokens(r) for r in metrics._main_requests(cur, run_id)]


def _compaction_timeline_svg(cur, run_id: str, width: int = 240, height: int = 24) -> str:
    rows = metrics._main_requests(cur, run_id)
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


def render_html(comparison: Comparison, cur, *, standalone: bool = True) -> str:
    """Render `comparison` as HTML. `standalone=True` (the default, used by
    `ys report --html`) wraps the table/charts in their own `<title>` and
    `<style>` so the result is a complete document. The dashboard's
    `/experiments/{name}/compare` route (ys/web/app.py) embeds this inside
    its own page shell instead of forking the renderer, so it passes
    `standalone=False` to get just the body fragment -- avoiding a second,
    conflicting `body { ... }` rule landing in the middle of that page.
    """
    rows_html = []
    baseline = next((a for a in comparison.arms if a.is_baseline), None)

    uncontrolled = ' <span class="warn">UNCONTROLLED</span>'
    cost_unknown_header = ' <span class="warn">COST UNKNOWN</span>'
    header_cells = "".join(
        f"<th>{html.escape(a.label)}{' *' if a.is_baseline else ''}"
        f"{uncontrolled if a.fingerprint_drifted else ''}"
        f"{cost_unknown_header if a.unpriced_models else ''}</th>"
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
            ci_suffix = _ci_suffix(stat.get("values") or [], fmt_spec)
            if ci_suffix:
                cell += f' <span class="spread">{ci_suffix.strip()}</span>'
            if key in _COST_DERIVED_METRICS and a.unpriced_models:
                cell += ' <span class="warn">?</span>'
            if not a.is_baseline:
                delta = _delta_str(mean, base_mean)
                if delta:
                    cell += f' <span class="delta">{delta}</span>'
            cells.append(f"<td>{cell}</td>")
        return f"<tr><th>{html.escape(key)}</th>{''.join(cells)}</tr>"

    rows_html.append(
        "<tr><th>success rate</th>"
        + "".join(
            f"<td>{html.escape(_success_rate_cell(a.aggregate['n_success'], a.aggregate['n_runs']))}</td>"
            for a in comparison.arms
        )
        + "</tr>"
    )
    if any(a.aggregate.get("n_unfinished", 0) for a in comparison.arms):
        # finding 13: shown separately from success rate above -- these
        # runs (displaced by `--force`, or never `ys end`ed) were never
        # scored, so they're excluded from n_runs rather than silently
        # dragging the success rate down.
        rows_html.append(
            "<tr><th>unfinished (excluded)</th>"
            + "".join(f"<td>{a.aggregate.get('n_unfinished', 0)}</td>" for a in comparison.arms)
            + "</tr>"
        )
    rows_html.append(
        "<tr><th>cost_per_success</th>"
        + "".join(
            f"<td>{_fmt(a.aggregate['cost_per_success'], '.4f')}"
            + (' <span class="warn">?</span>' if a.unpriced_models else "")
            + "</td>"
            for a in comparison.arms
        )
        + "</tr>"
    )
    for key in comparison.display_metrics():
        rows_html.append(metric_row(key))

    warnings = cost_warnings(comparison)
    warnings_html = ""
    if warnings:
        items = "".join(f"<li>{html.escape(w)}</li>" for w in warnings)
        warnings_html = f'<div class="cost-warning"><strong>Cost unavailable</strong><ul>{items}</ul></div>'

    cfg_warnings = config_warnings(comparison)
    config_warnings_html = ""
    if cfg_warnings:
        items = "".join(f"<li>{html.escape(w)}</li>" for w in cfg_warnings)
        config_warnings_html = (
            f'<div class="cost-warning"><strong>Some runs excluded (config changed)</strong>'
            f"<ul>{items}</ul></div>"
        )

    repeat_warnings = repeat_count_warnings(comparison)
    repeat_warnings_html = ""
    if repeat_warnings:
        items = "".join(f"<li>{html.escape(w)}</li>" for w in repeat_warnings)
        repeat_warnings_html = (
            f'<div class="repeat-warning"><strong>Unequal repeats</strong><ul>{items}</ul></div>'
        )

    verdicts = significance_verdicts(comparison)
    verdicts_html = ""
    if verdicts:
        items = "".join(f"<li>{html.escape(v)}</li>" for v in verdicts)
        verdicts_html = (
            f'<div class="verdict-banner"><strong>Is the difference real?</strong><ul>{items}</ul></div>'
        )

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

    body = f"""<h1>{html.escape(comparison.experiment_name)}</h1>
<p>task: {html.escape(comparison.task_id)} &middot; repeats: {comparison.repeats} &middot; (*) baseline</p>
{warnings_html}
{config_warnings_html}
{repeat_warnings_html}
{verdicts_html}
<table><tr><th></th>{header_cells}</tr>{''.join(rows_html)}</table>
<h2>per-run detail</h2>
<div class="chart-grid">{''.join(charts_html)}</div>
"""
    if not standalone:
        return body

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
  .cost-warning {{ border: 2px solid #e0555f; border-radius: 8px; padding: 0.75rem 1rem; margin: 1rem 0; color: #e0555f; }}
  .cost-warning ul {{ margin: 0.4rem 0 0; padding-left: 1.2rem; }}
  .repeat-warning {{ border: 2px solid #d9922a; border-radius: 8px; padding: 0.75rem 1rem; margin: 1rem 0; color: #d9922a; }}
  .repeat-warning ul {{ margin: 0.4rem 0 0; padding-left: 1.2rem; }}
  .verdict-banner {{ border: 2px solid #4f7cff; border-radius: 8px; padding: 0.75rem 1rem; margin: 1rem 0; color: #4f7cff; }}
  .verdict-banner ul {{ margin: 0.4rem 0 0; padding-left: 1.2rem; }}
</style>
{body}"""
