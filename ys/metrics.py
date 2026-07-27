"""Efficiency metric computations for yardstick runs and arms.

Every function here takes a sqlite3 cursor (as yielded by ys.db.cursor()) and
a run_id, and does read-only queries against the requests/tool_calls/runs
tables written by ys.collector. No writes, no network, no litellm calls --
these are pure functions over already-collected data, which is what makes
them testable with synthetic rows inserted directly through ys.db.

See spec section 5 for the metric definitions this module implements.
"""
import statistics
from typing import Callable, Optional

from ys.experiment import resolve_model_key

# Anthropic-shaped default weights for `billable_tokens` (finding 10): a
# pricing-*weighted proxy* for spend, not a token count. A 5-minute cache
# write costs ~1.25x a plain input token under Anthropic's pricing (the
# previous hardcoded formula used 1.0 for this -- simply wrong); a cache
# read costs ~0.1x (a ~90% discount). Meaningless for a provider with
# different cache economics -- an experiment can override these per model
# via `Experiment.billable_weights` (ys/experiment.py); a model with no
# override falls back to this default.
DEFAULT_BILLABLE_WEIGHTS = {
    "input": 1.0,
    "output": 1.0,
    "cache_creation": 1.25,
    "cache_read": 0.1,
}


# ---------------------------------------------------------------------------
# Row loading helpers
# ---------------------------------------------------------------------------

def _requests(cur, run_id: str) -> list[dict]:
    rows = cur.execute(
        "SELECT * FROM requests WHERE run_id = ? ORDER BY seq", (run_id,)
    ).fetchall()
    return [dict(r) for r in rows]


def _tool_calls(cur, run_id: str) -> list[dict]:
    rows = cur.execute("SELECT * FROM tool_calls WHERE run_id = ?", (run_id,)).fetchall()
    return [dict(r) for r in rows]


def _run_row(cur, run_id: str) -> Optional[dict]:
    row = cur.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    return dict(row) if row else None


def _main_thread_key(cur, run_id: str):
    """The thread_key with the most requests in this run -- the run's
    actual driving conversation, as opposed to interleaved background or
    Task-subagent traffic. Ties (e.g. a run with no thread_key data at all,
    grouped under NULL) broken toward whichever thread started first. See
    finding 4 in IMPROVEMENTS.md.

    Assumption (finding 26): "most requests" and "the conversation that
    started the run" are usually the same thread, but not always -- a
    long-running Task subagent can plausibly issue more requests than the
    conversation that spawned it, and this rule would then pick the
    subagent as "main". The rule is kept as-is rather than overridden to
    "whichever thread contains seq=1", because that alternative has its own
    failure mode already pinned by
    test_main_thread_fingerprint_prefers_largest_thread_over_first_request:
    a background call (e.g. title generation) that happens to be logged as
    the run's very first request, ahead of the real conversation's first
    turn, is a singleton that must not win just for being first. Neither
    signal (size, chronology) dominates the other, so overriding to
    chronology would only trade one mis-attribution for the other rather
    than fixing it. Instead, `_main_thread_started_run` below surfaces the
    disagreement so a run where the two signals conflict is visible rather
    than silently resolved one way. `cost_usd`/`billable_tokens` stay
    run-wide regardless of which thread this function picks, since that
    spend is real either way; only the conversation-shaped metrics
    (turns, compaction, overhead, the fingerprint) depend on this choice.
    """
    row = cur.execute(
        "SELECT thread_key FROM requests WHERE run_id = ? "
        "GROUP BY thread_key ORDER BY COUNT(*) DESC, MIN(seq) ASC LIMIT 1",
        (run_id,),
    ).fetchone()
    return row["thread_key"] if row else None


def _main_thread_started_run(cur, run_id: str, main_key=None) -> bool:
    """True if the thread `_main_thread_key` picked also contains the run's
    first request (seq=1). False is the finding-26 warning signal: the
    largest-thread rule and "whichever thread started the run" disagree,
    which happens when a Task subagent or other secondary thread racks up
    more requests than the conversation that spawned it. Doesn't change
    which thread is treated as main -- see `_main_thread_key`'s docstring
    for why overriding to chronology isn't a strict improvement -- just
    flags the case so it's visible instead of silently resolved."""
    if main_key is None:
        main_key = _main_thread_key(cur, run_id)
    row = cur.execute(
        "SELECT thread_key FROM requests WHERE run_id = ? ORDER BY seq LIMIT 1",
        (run_id,),
    ).fetchone()
    if row is None:
        return True
    return row["thread_key"] == main_key


def _main_requests(cur, run_id: str) -> list[dict]:
    main_key = _main_thread_key(cur, run_id)
    rows = cur.execute(
        "SELECT * FROM requests WHERE run_id = ? AND thread_key IS ? ORDER BY seq",
        (run_id, main_key),
    ).fetchall()
    return [dict(r) for r in rows]


def _main_tool_calls(cur, run_id: str) -> list[dict]:
    main_key = _main_thread_key(cur, run_id)
    rows = cur.execute(
        "SELECT tc.* FROM tool_calls tc JOIN requests r ON r.id = tc.request_id "
        "WHERE tc.run_id = ? AND r.thread_key IS ?",
        (run_id, main_key),
    ).fetchall()
    return [dict(r) for r in rows]


def background_metrics(cur, run_id: str) -> dict:
    """Traffic outside the run's main conversation thread: background
    (harness title-generation) requests and Task-subagent conversations.
    Reported as its own line item instead of being folded into -- and
    corrupting -- the main thread's conversation metrics. See finding 4."""
    reqs = _requests(cur, run_id)
    main_key = _main_thread_key(cur, run_id)
    background = [r for r in reqs if r.get("thread_key") != main_key]
    return {
        "background_requests": len(background),
        "background_tokens": sum(context_tokens(r) for r in background),
    }


def main_thread_metrics(cur, run_id: str) -> dict:
    """Finding 26 diagnostic: whether the thread `_main_thread_key` picked
    (most requests) is also the thread that contains the run's first
    request (seq=1). These agree in the overwhelmingly common case; they
    disagree exactly when a Task subagent or other secondary thread issues
    more requests than the conversation that actually started the run, in
    which case every conversation-shaped metric (turns, compaction,
    overhead, the corrected fingerprint) is being computed over the
    subagent instead. `main_thread_started_run` is a boolean finding, not a
    magnitude -- like `overhead_drift`, it's deliberately excluded from
    `_EFFICIENCY_METRICS` below rather than averaged across repeats."""
    return {"main_thread_started_run": _main_thread_started_run(cur, run_id)}


def main_thread_fingerprint(cur, run_id: str) -> Optional[dict]:
    """model/toolset_hash/system_prompt_hash of the first successful request
    in the run's main thread. Used to correct `runs`' fingerprint columns
    once a run finishes, in case the eager per-request fill in
    ys.collector (which can't yet know which thread will end up largest)
    stamped them from a background or subagent request instead.

    Inherits `_main_thread_key`'s finding-26 assumption: on the rare run
    where a Task subagent out-issues the conversation that spawned it, this
    fingerprint is the subagent's, not the driving conversation's. Check
    `main_thread_metrics`'s `main_thread_started_run` alongside this to know
    whether that's happened."""
    main_key = _main_thread_key(cur, run_id)
    row = cur.execute(
        "SELECT model, toolset_hash, system_prompt_hash FROM requests "
        "WHERE run_id = ? AND thread_key IS ? AND status_code = 200 "
        "ORDER BY seq LIMIT 1",
        (run_id, main_key),
    ).fetchone()
    return dict(row) if row else None


def _billable_weights_for(model, billable_weights_by_model: Optional[dict]) -> dict:
    """The BillableWeights (as a plain dict) declared for `model`, or the
    Anthropic-shaped default if none was declared / none matches. See
    finding 10 in IMPROVEMENTS.md."""
    if billable_weights_by_model:
        key = resolve_model_key(model, billable_weights_by_model)
        if key is not None:
            return billable_weights_by_model[key]
    return DEFAULT_BILLABLE_WEIGHTS


def _request_billable_tokens(req: dict, billable_weights_by_model: Optional[dict]) -> float:
    w = _billable_weights_for(req.get("model"), billable_weights_by_model)
    return (
        (req.get("input_tokens") or 0) * w.get("input", 1.0)
        + (req.get("cache_creation") or 0) * w.get("cache_creation", 1.25)
        + (req.get("output_tokens") or 0) * w.get("output", 1.0)
        + (req.get("cache_read") or 0) * w.get("cache_read", 0.1)
    )


def unpriced_models(cur, run_id: str) -> list[dict]:
    """Models with at least one request in this run whose cost LiteLLM
    couldn't price and no declared `pricing:` override could price either
    (`cost_source == 'unknown'`, ys/collector.py's `_resolve_cost`) --
    finding 9's silent-$0 case. Returned as {"model", "count"} so callers
    can report how many requests are affected, not just which model."""
    rows = cur.execute(
        "SELECT model, COUNT(*) AS n FROM requests "
        "WHERE run_id = ? AND cost_source = 'unknown' GROUP BY model",
        (run_id,),
    ).fetchall()
    return [{"model": r["model"], "count": r["n"]} for r in rows if r["model"]]


def context_tokens(req: dict) -> int:
    """Full prompt size for one request: new + cache-write + cache-read tokens."""
    return (req.get("input_tokens") or 0) + (req.get("cache_creation") or 0) + (req.get("cache_read") or 0)


def _linear_slope(xs: list[float], ys: list[float]) -> Optional[float]:
    """Least-squares slope of ys vs xs. None if fewer than 2 distinct x values."""
    n = len(xs)
    if n < 2 or len(set(xs)) < 2:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den = sum((x - mean_x) ** 2 for x in xs)
    if den == 0:
        return None
    return num / den


# ---------------------------------------------------------------------------
# 5.1 / 5.2 -- token accounting and cache efficiency
# ---------------------------------------------------------------------------

def _tokens_per_turn(billable_tokens: float, turns: int) -> Optional[float]:
    """Finding 15-18: `derived: [tokens_per_turn]` was named in both example
    experiments' `metrics:` block but never computed anywhere -- the
    obvious, cheap derived metric (billable_tokens per conversational turn)
    that field was presumably meant to enable. None for a run with no main-
    thread turns rather than a division error."""
    return (billable_tokens / turns) if turns else None


def token_metrics(cur, run_id: str, billable_weights_by_model: Optional[dict] = None) -> dict:
    reqs = _requests(cur, run_id)

    # billable_tokens is a pricing-*weighted proxy* for spend, not a token
    # count (finding 10) -- weighted per request by that request's own
    # model, since a run can in principle mix models (background/catch-all
    # traffic). `billable_weights_by_model` is the experiment's declared
    # `Experiment.billable_weights` (ys/experiment.py), plain-dict-ified by
    # the caller; a model with no entry there falls back to
    # DEFAULT_BILLABLE_WEIGHTS (Anthropic's cache economics).
    billable_tokens = sum(_request_billable_tokens(r, billable_weights_by_model) for r in reqs)
    cost_usd = sum(r.get("response_cost") or 0.0 for r in reqs)

    # Context growth and cache reuse describe the shape of one conversation,
    # not the run's total traffic -- computed over the main thread only so
    # interleaved background/subagent requests (their own short, low-cache
    # histories) don't distort them. Token/cost totals above stay run-wide:
    # that traffic is still real spend. See finding 4 in IMPROVEMENTS.md.
    main_reqs = _main_requests(cur, run_id)
    ctx = [context_tokens(r) for r in main_reqs]
    context_high_water = max(ctx) if ctx else 0
    context_growth_rate = _linear_slope([float(r["seq"]) for r in main_reqs], [float(c) for c in ctx])

    main_input_sum = sum(r.get("input_tokens") or 0 for r in main_reqs)
    main_cache_creation_sum = sum(r.get("cache_creation") or 0 for r in main_reqs)
    main_cache_read_sum = sum(r.get("cache_read") or 0 for r in main_reqs)
    cache_denom = main_cache_read_sum + main_input_sum + main_cache_creation_sum
    cache_read_ratio = (main_cache_read_sum / cache_denom) if cache_denom else 0.0

    return {
        "billable_tokens": billable_tokens,
        "cost_usd": cost_usd,
        "context_high_water": context_high_water,
        "context_growth_rate": context_growth_rate,
        "cache_read_ratio": cache_read_ratio,
    }


# ---------------------------------------------------------------------------
# 5.3 -- fixed overhead
# ---------------------------------------------------------------------------

def overhead_metrics(cur, run_id: str) -> dict:
    reqs = _main_requests(cur, run_id)
    turns = len(reqs)
    if not reqs:
        return {
            "overhead_tokens_per_turn": 0,
            "fixed_overhead_tokens": 0,
            "overhead_share": 0.0,
            "overhead_drift": False,
        }

    first = reqs[0]
    overhead_tokens_per_turn = (first.get("system_tokens") or 0) + (first.get("tools_tokens") or 0)
    fixed_overhead_tokens = overhead_tokens_per_turn * turns

    ctx_sum = sum(context_tokens(r) for r in reqs)
    overhead_share = (fixed_overhead_tokens / ctx_sum) if ctx_sum else 0.0

    # Spec 5.2: a harness that mutates its system prompt or tool defs every
    # turn is itself a metric-worthy finding, distinct from the point-in-time
    # overhead_tokens_per_turn number (which only looks at the first request).
    overhead_drift = any(
        (r.get("system_tokens") or 0) != (first.get("system_tokens") or 0)
        or (r.get("tools_tokens") or 0) != (first.get("tools_tokens") or 0)
        for r in reqs[1:]
    )

    return {
        "overhead_tokens_per_turn": overhead_tokens_per_turn,
        "fixed_overhead_tokens": fixed_overhead_tokens,
        "overhead_share": overhead_share,
        "overhead_drift": overhead_drift,
    }


# ---------------------------------------------------------------------------
# 5.4 -- turns and tool calls
# ---------------------------------------------------------------------------

def turn_metrics(cur, run_id: str) -> dict:
    # Main thread only -- a background title-generation request is not a turn.
    return {"turns": len(_main_requests(cur, run_id))}


def tool_call_metrics(cur, run_id: str) -> dict:
    turns = turn_metrics(cur, run_id)["turns"]
    calls = _main_tool_calls(cur, run_id)
    n_calls = len(calls)
    n_errors = sum(1 for c in calls if c.get("is_error"))

    return {
        "tool_calls": n_calls,
        "tool_calls_per_turn": (n_calls / turns) if turns else 0.0,
        # Denominator is all logged tool_calls, not just ones with a matching
        # tool_result block -- the schema doesn't distinguish "no result seen
        # yet" from "resolved, no error" (both leave is_error=0), so a call
        # missing its result is treated as non-error rather than excluded.
        "tool_error_rate": (n_errors / n_calls) if n_calls else 0.0,
    }


# ---------------------------------------------------------------------------
# 5.5 -- redundancy
# ---------------------------------------------------------------------------

_READ_LIKE_SUBSTRINGS = ("read", "grep", "glob")


def redundancy_metrics(cur, run_id: str) -> dict:
    calls = _main_tool_calls(cur, run_id)
    n_calls = len(calls)

    hashes = {c["input_hash"] for c in calls}
    unique_tool_calls = len(hashes)
    redundant_tool_calls = n_calls - unique_tool_calls
    redundancy_rate = (redundant_tool_calls / n_calls) if n_calls else 0.0

    # Best-effort: the schema stores only a hash of (name, canonicalized
    # input), not the raw file path, so we can't decode "which file" from
    # tool_calls alone. Group by (name, input_hash) instead -- repeated
    # identical calls to the same read/grep/glob-class tool are the closest
    # available proxy for "reads of the same file", since identical input
    # (including the path arg) hashes identically. This undercounts true
    # amplification when a harness reads the same file with slightly
    # different args (e.g. different line ranges), which would hash
    # differently and be treated as distinct "files".
    read_like = [c for c in calls if any(s in (c.get("name") or "").lower() for s in _READ_LIKE_SUBSTRINGS)]
    if read_like:
        groups: dict[tuple, int] = {}
        for c in read_like:
            key = (c["name"], c["input_hash"])
            groups[key] = groups.get(key, 0) + 1
        read_amplification = sum(groups.values()) / len(groups)
    else:
        read_amplification = None

    return {
        "unique_tool_calls": unique_tool_calls,
        "redundant_tool_calls": redundant_tool_calls,
        "redundancy_rate": redundancy_rate,
        "read_amplification": read_amplification,
    }


# ---------------------------------------------------------------------------
# 5.6 -- compaction and branching
# ---------------------------------------------------------------------------

def compaction_metrics(cur, run_id: str, regrowth_window: int = 5) -> dict:
    reqs = _main_requests(cur, run_id)
    compaction_idxs = [i for i, r in enumerate(reqs) if r.get("transition") == "compaction"]
    compaction_events = len(compaction_idxs)

    tokens_dropped = 0
    for i in compaction_idxs:
        if i == 0:
            continue
        drop = context_tokens(reqs[i - 1]) - context_tokens(reqs[i])
        if drop > 0:
            tokens_dropped += drop

    if compaction_events >= 2:
        seqs = [reqs[i]["seq"] for i in compaction_idxs]
        gaps = [b - a for a, b in zip(seqs, seqs[1:])]
        turns_to_recompaction = sum(gaps) / len(gaps)
    else:
        turns_to_recompaction = None

    # Best-effort: fit a simple linear slope of context_tokens over the
    # (up to regrowth_window) requests immediately following each compaction
    # event, then average the per-event slopes. Windows can run into a
    # subsequent compaction event (not excluded) -- a real drop inside the
    # window is itself part of "what happens after compaction" and folding
    # it in is preferred over arbitrarily truncating the window.
    slopes = []
    for i in compaction_idxs:
        window = reqs[i + 1 : i + 1 + regrowth_window]
        if len(window) < 2:
            continue
        slope = _linear_slope([float(r["seq"]) for r in window], [float(context_tokens(r)) for r in window])
        if slope is not None:
            slopes.append(slope)
    post_compaction_regrowth = (sum(slopes) / len(slopes)) if slopes else None

    return {
        "compaction_events": compaction_events,
        "tokens_dropped": tokens_dropped,
        "turns_to_recompaction": turns_to_recompaction,
        "post_compaction_regrowth": post_compaction_regrowth,
    }


# ---------------------------------------------------------------------------
# 5.7 -- outcome
# ---------------------------------------------------------------------------

def outcome_metrics(cur, run_id: str) -> dict:
    run = _run_row(cur, run_id)
    reqs = _requests(cur, run_id)
    active_s = sum(r.get("latency_ms") or 0 for r in reqs) / 1000.0

    task_success = None
    wall_clock_s = None
    if run:
        ts = run.get("task_success")
        task_success = bool(ts) if ts is not None else None
        wall_clock_s = run.get("wall_clock_s")

    return {
        "task_success": task_success,
        "wall_clock_s": wall_clock_s,
        "active_s": active_s,
    }


# ---------------------------------------------------------------------------
# Per-run rollup
# ---------------------------------------------------------------------------

def compute_run_metrics(cur, run_id: str, billable_weights_by_model: Optional[dict] = None) -> dict:
    """All per-run metrics for one run_id, merged into a single flat dict.

    `billable_weights_by_model` (see `token_metrics`) is optional -- callers
    that don't have an `Experiment` handy (e.g. `ys end`'s immediate,
    single-run summary) get `billable_tokens` under the Anthropic-shaped
    default; `ys compare`/`ys report` (ys/render.py) pass the experiment's
    declared `billable_weights` explicitly."""
    metrics: dict = {}
    metrics.update(token_metrics(cur, run_id, billable_weights_by_model))
    metrics.update(overhead_metrics(cur, run_id))
    metrics.update(turn_metrics(cur, run_id))
    metrics.update(tool_call_metrics(cur, run_id))
    metrics.update(redundancy_metrics(cur, run_id))
    metrics.update(compaction_metrics(cur, run_id))
    metrics.update(background_metrics(cur, run_id))
    metrics.update(main_thread_metrics(cur, run_id))
    metrics.update(outcome_metrics(cur, run_id))
    # Derived from two metrics already computed above -- must run after both
    # token_metrics (billable_tokens) and turn_metrics (turns). See finding
    # 15-18 and `_tokens_per_turn`'s docstring.
    metrics["tokens_per_turn"] = _tokens_per_turn(metrics["billable_tokens"], metrics["turns"])
    return metrics


# Numeric metrics that get mean/spread aggregated across the repeats of an
# arm. Excludes task_success (the gate itself) and overhead_drift (boolean
# finding, not a magnitude).
_EFFICIENCY_METRICS = [
    "billable_tokens",
    "cost_usd",
    "context_high_water",
    "context_growth_rate",
    "cache_read_ratio",
    "overhead_tokens_per_turn",
    "fixed_overhead_tokens",
    "overhead_share",
    "turns",
    "tool_calls",
    "tool_calls_per_turn",
    "tool_error_rate",
    "unique_tool_calls",
    "redundant_tool_calls",
    "redundancy_rate",
    "read_amplification",
    "compaction_events",
    "tokens_dropped",
    "turns_to_recompaction",
    "post_compaction_regrowth",
    "background_requests",
    "background_tokens",
    "wall_clock_s",
    "active_s",
    "tokens_per_turn",
]


# `Experiment.metrics.gate` (ys/experiment.py) is a string in the YAML,
# validated there against `VALID_GATE_NAMES`; this is the other half --
# turning that validated string into the predicate `aggregate_run_metrics`
# actually gates on. Finding 15-18: previously `aggregate_run_metrics`
# simply hardcoded the task_success predicate inline and ignored the
# schema field entirely. Only one gate exists today (task_success is the
# only boolean pass/fail signal this rig computes), but callers now go
# through this registry instead of the hardcoded default, so a second gate
# is a one-line addition here rather than a second hardcoded branch.
_GATES: dict[str, Callable[[dict], bool]] = {
    "task_success": lambda m: bool(m.get("task_success")),
}


def resolve_gate(name: str) -> Callable[[dict], bool]:
    """Map a `metrics.gate` name to the predicate `aggregate_run_metrics`
    uses to decide whether a run counts as a success. Raises ValueError
    naming the valid options on an unknown name -- in practice unreachable
    from a YAML that's already passed `Experiment` validation (which checks
    the same registry), but this is also callable directly, so it
    shouldn't trust its input either."""
    try:
        return _GATES[name]
    except KeyError:
        raise ValueError(f"unknown metrics.gate '{name}' -- valid options: {', '.join(sorted(_GATES))}")


def aggregate_run_metrics(
    cur,
    run_ids: list[str],
    gate: Optional[Callable[[dict], bool]] = None,
    billable_weights_by_model: Optional[dict] = None,
) -> dict:
    """Aggregate per-run metrics across the repeats of one arm.

    `gate` is a predicate over a single run's metrics dict deciding whether
    that run counts as a "success"; it defaults to the run's task_success
    flag. Every *finished* run_id contributes to n_runs and the success
    count/rate regardless of gate outcome. Only gate-passing runs
    contribute to the efficiency metric means/spreads -- an arm that's
    cheap but fails half the time should not look cheap. cost_per_success
    sums cost_usd over ALL runs of the arm, finished or not (a run that
    never finished, e.g. one displaced by `--force`, may still have spent
    real money before it was cut off) divided by the count of successful
    runs, per spec 5.7.

    Finding 13: a run whose `task_success` is still NULL never got an
    `ys end` -- either it's still genuinely active, or it was displaced by
    `--force` and `state.py` flagged it `abandoned` without ever assigning
    it a verdict. Either way it was never scored, which isn't the same as
    failing the task, so it's excluded from n_runs/n_success/success_rate
    entirely instead of silently counting against the arm forever -- and
    reported separately via `n_unfinished` so it stays visible rather than
    just vanishing from the totals.

    `billable_weights_by_model` is forwarded to `compute_run_metrics` for
    every run (see `token_metrics`) -- finding 10.

    Spread is population stdev (statistics.pstdev): 0.0 for a single
    observation rather than undefined, which keeps `repeats: 1` arms usable.

    Each metric's entry also carries the raw per-run `values` list that
    `mean`/`spread` were computed from (same gate-passing population, same
    order as `run_ids`) -- feature 3's bootstrap/permutation statistics
    (`ys/statistics.py`) need the actual observations, not just their
    summary, and this is the one place that already assembles them per
    metric. `render.py` is the only consumer; nothing here changes for
    existing callers that only read `mean`/`n`/`spread`.
    """
    if gate is None:

        def gate(m):
            return bool(m.get("task_success"))

    per_run = {rid: compute_run_metrics(cur, rid, billable_weights_by_model) for rid in run_ids}
    finished = {rid: m for rid, m in per_run.items() if m.get("task_success") is not None}
    n_runs = len(finished)
    n_unfinished = len(run_ids) - n_runs
    passing = {rid: m for rid, m in finished.items() if gate(m)}
    n_success = len(passing)

    metrics_out = {}
    for key in _EFFICIENCY_METRICS:
        values = [m[key] for m in passing.values() if m.get(key) is not None]
        if values:
            mean = sum(values) / len(values)
            spread = statistics.pstdev(values) if len(values) > 1 else 0.0
        else:
            mean = None
            spread = None
        metrics_out[key] = {"mean": mean, "n": len(values), "spread": spread, "values": values}

    total_cost = sum(m["cost_usd"] for m in per_run.values())
    cost_per_success = (total_cost / n_success) if n_success else None

    return {
        "n_runs": n_runs,
        "n_unfinished": n_unfinished,
        "n_success": n_success,
        "success_rate": (n_success / n_runs) if n_runs else None,
        "cost_per_success": cost_per_success,
        "metrics": metrics_out,
    }
