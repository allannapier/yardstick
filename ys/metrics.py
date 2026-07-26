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
    finding 4 in IMPROVEMENTS.md."""
    row = cur.execute(
        "SELECT thread_key FROM requests WHERE run_id = ? "
        "GROUP BY thread_key ORDER BY COUNT(*) DESC, MIN(seq) ASC LIMIT 1",
        (run_id,),
    ).fetchone()
    return row["thread_key"] if row else None


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


def main_thread_fingerprint(cur, run_id: str) -> Optional[dict]:
    """model/toolset_hash/system_prompt_hash of the first successful request
    in the run's main thread. Used to correct `runs`' fingerprint columns
    once a run finishes, in case the eager per-request fill in
    ys.collector (which can't yet know which thread will end up largest)
    stamped them from a background or subagent request instead."""
    main_key = _main_thread_key(cur, run_id)
    row = cur.execute(
        "SELECT model, toolset_hash, system_prompt_hash FROM requests "
        "WHERE run_id = ? AND thread_key IS ? AND status_code = 200 "
        "ORDER BY seq LIMIT 1",
        (run_id, main_key),
    ).fetchone()
    return dict(row) if row else None


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

def token_metrics(cur, run_id: str) -> dict:
    reqs = _requests(cur, run_id)

    input_sum = sum(r.get("input_tokens") or 0 for r in reqs)
    cache_creation_sum = sum(r.get("cache_creation") or 0 for r in reqs)
    cache_read_sum = sum(r.get("cache_read") or 0 for r in reqs)
    output_sum = sum(r.get("output_tokens") or 0 for r in reqs)

    billable_tokens = input_sum + cache_creation_sum + output_sum + cache_read_sum * 0.1
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

def compute_run_metrics(cur, run_id: str) -> dict:
    """All per-run metrics for one run_id, merged into a single flat dict."""
    metrics: dict = {}
    metrics.update(token_metrics(cur, run_id))
    metrics.update(overhead_metrics(cur, run_id))
    metrics.update(turn_metrics(cur, run_id))
    metrics.update(tool_call_metrics(cur, run_id))
    metrics.update(redundancy_metrics(cur, run_id))
    metrics.update(compaction_metrics(cur, run_id))
    metrics.update(background_metrics(cur, run_id))
    metrics.update(outcome_metrics(cur, run_id))
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
]


def aggregate_run_metrics(
    cur, run_ids: list[str], gate: Optional[Callable[[dict], bool]] = None
) -> dict:
    """Aggregate per-run metrics across the repeats of one arm.

    `gate` is a predicate over a single run's metrics dict deciding whether
    that run counts as a "success"; it defaults to the run's task_success
    flag. Every run_id contributes to n_runs and the success count/rate
    regardless of gate outcome. Only gate-passing runs contribute to the
    efficiency metric means/spreads -- an arm that's cheap but fails half
    the time should not look cheap. cost_per_success sums cost_usd over
    ALL runs of the arm (successful or not -- failed runs still spent
    money) divided by the count of successful runs, per spec 5.7.

    Spread is population stdev (statistics.pstdev): 0.0 for a single
    observation rather than undefined, which keeps `repeats: 1` arms usable.
    """
    if gate is None:

        def gate(m):
            return bool(m.get("task_success"))

    per_run = {rid: compute_run_metrics(cur, rid) for rid in run_ids}
    n_runs = len(run_ids)
    passing = {rid: m for rid, m in per_run.items() if gate(m)}
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
        metrics_out[key] = {"mean": mean, "n": len(values), "spread": spread}

    total_cost = sum(m["cost_usd"] for m in per_run.values())
    cost_per_success = (total_cost / n_success) if n_success else None

    return {
        "n_runs": n_runs,
        "n_success": n_success,
        "success_rate": (n_success / n_runs) if n_runs else None,
        "cost_per_success": cost_per_success,
        "metrics": metrics_out,
    }
