import pytest

from ys import db, metrics


# ---------------------------------------------------------------------------
# Synthetic fixture helpers -- direct inserts through ys.db, no proxy/LLM.
# ---------------------------------------------------------------------------

def _mk_arm(cur, experiment_id="exp1", arm_id="arm1"):
    cur.execute(
        "INSERT OR IGNORE INTO experiments (id, name, question, task_json, config_yaml, created_at) "
        "VALUES (?,?,?,?,?,?)",
        (experiment_id, experiment_id, None, "{}", "", "2026-01-01T00:00:00Z"),
    )
    cur.execute(
        "INSERT OR IGNORE INTO arms (id, experiment_id, label, factors_json, is_baseline) VALUES (?,?,?,?,0)",
        (arm_id, experiment_id, arm_id, "{}"),
    )


def _mk_run(cur, run_id, arm_id="arm1", experiment_id="exp1", task_success=1, wall_clock_s=10.0):
    _mk_arm(cur, experiment_id, arm_id)
    cur.execute(
        "INSERT INTO runs (id, experiment_id, arm_id, repeat_idx, started_at, wall_clock_s, task_success) "
        "VALUES (?,?,?,0,?,?,?)",
        (run_id, experiment_id, arm_id, "2026-01-01T00:00:00Z", wall_clock_s, task_success),
    )


def _mk_request(cur, run_id, seq, **overrides):
    defaults = dict(
        ts="2026-01-01T00:00:00Z",
        provider="anthropic",
        model="claude-x",
        stream=0,
        input_tokens=100,
        cache_creation=0,
        cache_read=0,
        output_tokens=50,
        response_cost=0.01,
        latency_ms=500.0,
        ttft_ms=None,
        status_code=200,
        error=None,
        msg_count=1,
        msg_hashes_json="[]",
        system_tokens=200,
        tools_tokens=100,
        transition=None,
        # None (not e.g. "main") so existing tests -- which never set this --
        # keep landing in one group and stay unaffected by the main-thread
        # scoping added for finding 4.
        thread_key=None,
        toolset_hash=None,
        system_prompt_hash=None,
        # None (not e.g. "litellm") so existing tests -- which never set
        # this -- stay unaffected by the cost_source column added for
        # finding 9.
        cost_source=None,
    )
    defaults.update(overrides)
    cur.execute(
        """INSERT INTO requests
           (run_id, seq, ts, provider, model, stream, input_tokens, cache_creation,
            cache_read, output_tokens, response_cost, latency_ms, ttft_ms, status_code,
            error, msg_count, msg_hashes_json, system_tokens, tools_tokens, transition,
            thread_key, toolset_hash, system_prompt_hash, cost_source)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            run_id, seq, defaults["ts"], defaults["provider"], defaults["model"], defaults["stream"],
            defaults["input_tokens"], defaults["cache_creation"], defaults["cache_read"],
            defaults["output_tokens"], defaults["response_cost"], defaults["latency_ms"],
            defaults["ttft_ms"], defaults["status_code"], defaults["error"], defaults["msg_count"],
            defaults["msg_hashes_json"], defaults["system_tokens"], defaults["tools_tokens"],
            defaults["transition"], defaults["thread_key"], defaults["toolset_hash"],
            defaults["system_prompt_hash"], defaults["cost_source"],
        ),
    )
    return cur.lastrowid


def _mk_tool_call(cur, request_id, run_id, name, input_hash, is_error=0):
    cur.execute(
        "INSERT INTO tool_calls (request_id, run_id, name, input_hash, input_bytes, is_error, result_tokens) "
        "VALUES (?,?,?,?,?,?,10)",
        (request_id, run_id, name, input_hash, 20, is_error),
    )


# ---------------------------------------------------------------------------
# Normal run: multi-turn, mixed cache/read tokens, some redundant tool calls
# ---------------------------------------------------------------------------

def test_normal_run():
    db.init_db()
    with db.cursor() as cur:
        _mk_run(cur, "r1")
        r1 = _mk_request(cur, "r1", 1, input_tokens=1000, cache_creation=200, cache_read=0,
                          output_tokens=300, response_cost=0.05, system_tokens=500, tools_tokens=300,
                          latency_ms=1000)
        r2 = _mk_request(cur, "r1", 2, input_tokens=50, cache_creation=0, cache_read=1000,
                          output_tokens=200, response_cost=0.03, system_tokens=500, tools_tokens=300,
                          latency_ms=1200)
        r3 = _mk_request(cur, "r1", 3, input_tokens=50, cache_creation=0, cache_read=1200,
                          output_tokens=250, response_cost=0.04, system_tokens=500, tools_tokens=300,
                          latency_ms=1300)

        _mk_tool_call(cur, r1, "r1", "Read", "hashA")
        _mk_tool_call(cur, r1, "r1", "Read", "hashA")  # redundant, same file
        _mk_tool_call(cur, r1, "r1", "Bash", "hashB")
        _mk_tool_call(cur, r2, "r1", "Grep", "hashC")
        _mk_tool_call(cur, r3, "r1", "Read", "hashA")  # re-read same file, later turn

    with db.cursor() as cur:
        m = metrics.compute_run_metrics(cur, "r1")

    assert m["turns"] == 3
    assert m["cost_usd"] == pytest.approx(0.12)
    # billable = sum(input)*1.0 + sum(cache_creation)*1.25 + sum(output)*1.0 + sum(cache_read)*0.1
    #          = 1100 + 200*1.25 + 750 + 2200*0.1
    # cache_creation weight is 1.25 (finding 10's fixed Anthropic default --
    # a cache write costs ~1.25x a plain input token, not 1.0x).
    assert m["billable_tokens"] == pytest.approx(2320.0)
    # context_tokens per request: 1200, 1050, 1250
    assert m["context_high_water"] == 1250
    assert m["context_growth_rate"] == pytest.approx(25.0)
    # cache_read_ratio = 2200 / (2200 + 1100 + 200)
    assert m["cache_read_ratio"] == pytest.approx(2200 / 3500)

    assert m["overhead_tokens_per_turn"] == 800
    assert m["fixed_overhead_tokens"] == 2400
    assert m["overhead_share"] == pytest.approx(2400 / 3500)
    assert m["overhead_drift"] is False

    assert m["tool_calls"] == 5
    assert m["tool_calls_per_turn"] == pytest.approx(5 / 3)
    assert m["tool_error_rate"] == pytest.approx(0.0)

    assert m["unique_tool_calls"] == 3
    assert m["redundant_tool_calls"] == 2
    assert m["redundancy_rate"] == pytest.approx(0.4)
    # read-like calls: Read x3 (all hashA -> 1 group of 3), Grep x1 (1 group of 1)
    # mean group size = 4 calls / 2 groups
    assert m["read_amplification"] == pytest.approx(2.0)

    assert m["compaction_events"] == 0
    assert m["tokens_dropped"] == 0
    assert m["turns_to_recompaction"] is None
    assert m["post_compaction_regrowth"] is None

    assert m["task_success"] is True
    assert m["wall_clock_s"] == pytest.approx(10.0)
    assert m["active_s"] == pytest.approx((1000 + 1200 + 1300) / 1000.0)


# ---------------------------------------------------------------------------
# Zero tool calls
# ---------------------------------------------------------------------------

def test_run_with_zero_tool_calls():
    db.init_db()
    with db.cursor() as cur:
        _mk_run(cur, "r_notools")
        _mk_request(cur, "r_notools", 1)
        _mk_request(cur, "r_notools", 2)

    with db.cursor() as cur:
        m = metrics.compute_run_metrics(cur, "r_notools")

    assert m["turns"] == 2
    assert m["tool_calls"] == 0
    assert m["tool_calls_per_turn"] == 0.0
    assert m["tool_error_rate"] == 0.0
    assert m["unique_tool_calls"] == 0
    assert m["redundant_tool_calls"] == 0
    assert m["redundancy_rate"] == 0.0
    assert m["read_amplification"] is None


# ---------------------------------------------------------------------------
# 100% tool error rate
# ---------------------------------------------------------------------------

def test_run_with_full_tool_error_rate():
    db.init_db()
    with db.cursor() as cur:
        _mk_run(cur, "r_err")
        r1 = _mk_request(cur, "r_err", 1)
        _mk_tool_call(cur, r1, "r_err", "Bash", "hashA", is_error=1)
        _mk_tool_call(cur, r1, "r_err", "Bash", "hashB", is_error=1)

    with db.cursor() as cur:
        m = metrics.compute_run_metrics(cur, "r_err")

    assert m["tool_calls"] == 2
    assert m["tool_error_rate"] == pytest.approx(1.0)
    assert m["unique_tool_calls"] == 2
    assert m["redundancy_rate"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Compaction event, single occurrence, with post-compaction regrowth
# ---------------------------------------------------------------------------

def test_run_with_single_compaction_event():
    db.init_db()
    with db.cursor() as cur:
        _mk_run(cur, "r_compact")
        # context_tokens = input_tokens here (cache fields left at 0) for simplicity
        _mk_request(cur, "r_compact", 1, input_tokens=1000, transition=None)
        _mk_request(cur, "r_compact", 2, input_tokens=1400, transition="continuation")
        _mk_request(cur, "r_compact", 3, input_tokens=300, transition="compaction")
        _mk_request(cur, "r_compact", 4, input_tokens=500, transition="continuation")
        _mk_request(cur, "r_compact", 5, input_tokens=700, transition="continuation")

    with db.cursor() as cur:
        m = metrics.compute_run_metrics(cur, "r_compact")

    assert m["compaction_events"] == 1
    assert m["tokens_dropped"] == 1100  # 1400 -> 300
    assert m["turns_to_recompaction"] is None  # only one event, no gap to measure
    # regrowth window after seq=3: seq 4,5 -> context 500,700 -> slope 200/turn
    assert m["post_compaction_regrowth"] == pytest.approx(200.0)


# ---------------------------------------------------------------------------
# Compaction event, two occurrences (exercises turns_to_recompaction)
# ---------------------------------------------------------------------------

def test_run_with_multiple_compaction_events():
    db.init_db()
    with db.cursor() as cur:
        _mk_run(cur, "r_compact2")
        _mk_request(cur, "r_compact2", 1, input_tokens=1000, transition=None)
        _mk_request(cur, "r_compact2", 2, input_tokens=1400, transition="continuation")
        _mk_request(cur, "r_compact2", 3, input_tokens=300, transition="compaction")
        _mk_request(cur, "r_compact2", 4, input_tokens=500, transition="continuation")
        _mk_request(cur, "r_compact2", 5, input_tokens=700, transition="continuation")
        _mk_request(cur, "r_compact2", 6, input_tokens=100, transition="compaction")
        _mk_request(cur, "r_compact2", 7, input_tokens=150, transition="continuation")

    with db.cursor() as cur:
        m = metrics.compute_run_metrics(cur, "r_compact2")

    assert m["compaction_events"] == 2
    assert m["tokens_dropped"] == 1100 + 600  # (1400->300) + (700->100)
    assert m["turns_to_recompaction"] == pytest.approx(3.0)  # seq 3 -> seq 6


# ---------------------------------------------------------------------------
# Cross-run aggregation: gate-filtering excludes failed runs from efficiency
# means but includes them in n / success-rate / cost_per_success denominator.
# ---------------------------------------------------------------------------

def test_aggregate_run_metrics_excludes_failed_runs_from_efficiency_but_not_n():
    db.init_db()
    with db.cursor() as cur:
        _mk_run(cur, "r_ok1", task_success=1)
        _mk_request(cur, "r_ok1", 1, input_tokens=100, output_tokens=50, response_cost=1.0)

        _mk_run(cur, "r_ok2", task_success=1)
        _mk_request(cur, "r_ok2", 1, input_tokens=200, output_tokens=100, response_cost=2.0)

        _mk_run(cur, "r_fail", task_success=0)
        _mk_request(cur, "r_fail", 1, input_tokens=1000, output_tokens=500, response_cost=5.0)

    with db.cursor() as cur:
        agg = metrics.aggregate_run_metrics(cur, ["r_ok1", "r_ok2", "r_fail"])

    assert agg["n_runs"] == 3
    assert agg["n_success"] == 2
    assert agg["success_rate"] == pytest.approx(2 / 3)
    # cost_per_success = sum(cost_usd over ALL runs) / n_success = (1+2+5)/2
    assert agg["cost_per_success"] == pytest.approx(4.0)

    billable = agg["metrics"]["billable_tokens"]
    # only r_ok1 (150) and r_ok2 (300) contribute; r_fail's 1500 is excluded
    assert billable["n"] == 2
    assert billable["mean"] == pytest.approx((150 + 300) / 2)
    assert billable["spread"] == pytest.approx(75.0)  # pstdev([150, 300])

    cost = agg["metrics"]["cost_usd"]
    assert cost["n"] == 2
    assert cost["mean"] == pytest.approx((1.0 + 2.0) / 2)


# ---------------------------------------------------------------------------
# finding 13: a run that never got an `ys end` (task_success still NULL --
# e.g. one displaced by `--force`, which ys/state.py flags `abandoned`
# without ever assigning it a verdict) must not count toward n_runs/
# success_rate at all, since it was never scored -- not the same thing as
# having failed the task.
# ---------------------------------------------------------------------------

def test_aggregate_run_metrics_excludes_unfinished_runs_from_n_runs():
    """Regression test for finding 13. Before the fix, n_runs was simply
    len(run_ids) regardless of whether a run ever finished, so a run
    displaced by `--force` (task_success left NULL forever) permanently
    dragged the arm's success rate down: with the old code this would
    compute n_runs == 3 and success_rate == 2/3 instead of n_runs == 2 and
    success_rate == 1.0."""
    db.init_db()
    with db.cursor() as cur:
        _mk_run(cur, "r_ok1", task_success=1)
        _mk_request(cur, "r_ok1", 1, response_cost=1.0)

        _mk_run(cur, "r_ok2", task_success=1)
        _mk_request(cur, "r_ok2", 1, response_cost=2.0)

        _mk_run(cur, "r_unfinished", task_success=None)
        _mk_request(cur, "r_unfinished", 1, response_cost=3.0)

    with db.cursor() as cur:
        agg = metrics.aggregate_run_metrics(cur, ["r_ok1", "r_ok2", "r_unfinished"])

    assert agg["n_runs"] == 2
    assert agg["n_unfinished"] == 1
    assert agg["n_success"] == 2
    assert agg["success_rate"] == pytest.approx(1.0)
    # cost_per_success still totals cost_usd over every run, finished or
    # not -- an unfinished run may still have spent real money before it
    # was cut off, and spec 5.7's cost_per_success is unconditional on that.
    assert agg["cost_per_success"] == pytest.approx((1.0 + 2.0 + 3.0) / 2)


def test_aggregate_run_metrics_all_unfinished_yields_no_success_rate():
    db.init_db()
    with db.cursor() as cur:
        _mk_run(cur, "r1", task_success=None)
        _mk_request(cur, "r1", 1)

    with db.cursor() as cur:
        agg = metrics.aggregate_run_metrics(cur, ["r1"])

    assert agg["n_runs"] == 0
    assert agg["n_unfinished"] == 1
    assert agg["success_rate"] is None


def test_aggregate_run_metrics_custom_gate():
    db.init_db()
    with db.cursor() as cur:
        _mk_run(cur, "r1", task_success=1)
        _mk_request(cur, "r1", 1, response_cost=1.0)
        _mk_run(cur, "r2", task_success=1)
        _mk_request(cur, "r2", 1, response_cost=2.0)

    with db.cursor() as cur:
        agg = metrics.aggregate_run_metrics(cur, ["r1", "r2"], gate=lambda m: False)

    assert agg["n_runs"] == 2
    assert agg["n_success"] == 0
    assert agg["cost_per_success"] is None
    assert agg["metrics"]["cost_usd"]["mean"] is None
    assert agg["metrics"]["cost_usd"]["n"] == 0


# ---------------------------------------------------------------------------
# finding 4: interleaved background/subagent traffic must not corrupt the
# main conversation's metrics, and should be reported as its own line item.
# ---------------------------------------------------------------------------

def test_background_traffic_excluded_from_conversation_metrics_but_counted_separately():
    db.init_db()
    with db.cursor() as cur:
        _mk_run(cur, "r_bg")
        _mk_request(cur, "r_bg", 1, thread_key="main", input_tokens=100, output_tokens=50,
                     response_cost=0.01)
        _mk_request(cur, "r_bg", 2, thread_key="main", input_tokens=100, output_tokens=50,
                     response_cost=0.01)
        # a harness title-generation call landing between two main turns
        _mk_request(cur, "r_bg", 3, thread_key="bg-title-gen", input_tokens=20, output_tokens=5,
                     response_cost=0.001, system_tokens=10, tools_tokens=0)

    with db.cursor() as cur:
        m = metrics.compute_run_metrics(cur, "r_bg")

    assert m["turns"] == 2  # the background request is not a conversational turn
    assert m["background_requests"] == 1
    assert m["background_tokens"] == 20
    # totals still include the background request's real spend/tokens
    assert m["cost_usd"] == pytest.approx(0.021)
    assert m["billable_tokens"] == pytest.approx(325.0)


def test_main_thread_fingerprint_prefers_largest_thread_over_first_request():
    """A background/subagent call landing before the main conversation's
    first request must not win the run's fingerprint -- see finding 4."""
    db.init_db()
    with db.cursor() as cur:
        _mk_run(cur, "r_fp")
        _mk_request(cur, "r_fp", 1, thread_key="bg", model="bg-model")
        _mk_request(cur, "r_fp", 2, thread_key="main", model="main-model")
        _mk_request(cur, "r_fp", 3, thread_key="main", model="main-model")

    with db.cursor() as cur:
        fp = metrics.main_thread_fingerprint(cur, "r_fp")

    assert fp["model"] == "main-model"


# ---------------------------------------------------------------------------
# finding 26: the largest thread isn't always the one that started the run.
# ---------------------------------------------------------------------------

def test_main_thread_started_run_true_when_largest_thread_contains_seq1():
    db.init_db()
    with db.cursor() as cur:
        _mk_run(cur, "r_started")
        _mk_request(cur, "r_started", 1, thread_key="main")
        _mk_request(cur, "r_started", 2, thread_key="main")
        _mk_request(cur, "r_started", 3, thread_key="bg")

    with db.cursor() as cur:
        m = metrics.compute_run_metrics(cur, "r_started")

    assert m["main_thread_started_run"] is True


def test_main_thread_started_run_false_when_subagent_outnumbers_driving_conversation():
    """Regression test for finding 26: a long-running Task subagent can
    issue more requests than the conversation that spawned it, so the
    largest-thread rule (_main_thread_key, finding 4) picks the subagent as
    "main" instead of the conversation that actually started the run (seq
    1). _main_thread_key keeps its size-based rule -- see its docstring for
    why overriding to "whichever thread has seq=1" isn't a strict
    improvement, given test_main_thread_fingerprint_prefers_largest_thread_over_first_request
    above pins the opposite failure mode -- but main_thread_started_run must
    surface the disagreement instead of resolving it silently."""
    db.init_db()
    with db.cursor() as cur:
        _mk_run(cur, "r_sub")
        _mk_request(cur, "r_sub", 1, thread_key="main")
        _mk_request(cur, "r_sub", 2, thread_key="main")
        _mk_request(cur, "r_sub", 3, thread_key="subagent")
        _mk_request(cur, "r_sub", 4, thread_key="subagent")
        _mk_request(cur, "r_sub", 5, thread_key="subagent")

    with db.cursor() as cur:
        main_key = metrics._main_thread_key(cur, "r_sub")
        m = metrics.compute_run_metrics(cur, "r_sub")

    assert main_key == "subagent"  # largest thread still wins -- unchanged from finding 4
    assert m["main_thread_started_run"] is False


# ---------------------------------------------------------------------------
# finding 10: billable_tokens is a per-model-weighted proxy, not a flat
# formula -- an experiment's declared `billable_weights` (ys/experiment.py)
# must override the Anthropic-shaped default per request's own model.
# ---------------------------------------------------------------------------

def test_billable_tokens_uses_declared_per_model_weights():
    db.init_db()
    with db.cursor() as cur:
        _mk_run(cur, "r_weights")
        _mk_request(cur, "r_weights", 1, model="model-a", input_tokens=100, output_tokens=0,
                     cache_creation=0, cache_read=0)
        _mk_request(cur, "r_weights", 2, model="model-b", input_tokens=100, output_tokens=0,
                     cache_creation=0, cache_read=0)

    weights = {
        "model-a": {"input": 2.0, "output": 1.0, "cache_creation": 1.25, "cache_read": 0.1},
        "model-b": {"input": 5.0, "output": 1.0, "cache_creation": 1.25, "cache_read": 0.1},
    }
    with db.cursor() as cur:
        m = metrics.compute_run_metrics(cur, "r_weights", billable_weights_by_model=weights)

    # model-a: 100 * 2.0 = 200; model-b: 100 * 5.0 = 500
    assert m["billable_tokens"] == pytest.approx(700.0)


def test_billable_tokens_falls_back_to_anthropic_default_for_undeclared_model():
    db.init_db()
    with db.cursor() as cur:
        _mk_run(cur, "r_fallback")
        _mk_request(cur, "r_fallback", 1, model="unknown-model", input_tokens=100,
                     output_tokens=0, cache_creation=100, cache_read=0)

    # billable_weights_by_model declares weights for a *different* model --
    # "unknown-model" must fall back to DEFAULT_BILLABLE_WEIGHTS, not to 0.
    weights = {"some-other-model": {"input": 9.0, "output": 9.0, "cache_creation": 9.0, "cache_read": 9.0}}
    with db.cursor() as cur:
        m = metrics.compute_run_metrics(cur, "r_fallback", billable_weights_by_model=weights)

    # 100 * 1.0 (input) + 100 * 1.25 (cache_creation, Anthropic default)
    assert m["billable_tokens"] == pytest.approx(225.0)


def test_billable_tokens_resolves_provider_prefixed_model_against_bare_weight_key():
    db.init_db()
    with db.cursor() as cur:
        _mk_run(cur, "r_prefix")
        _mk_request(cur, "r_prefix", 1, model="anthropic/claude-sonnet-5", input_tokens=100,
                     output_tokens=0, cache_creation=0, cache_read=0)

    weights = {"claude-sonnet-5": {"input": 3.0, "output": 1.0, "cache_creation": 1.25, "cache_read": 0.1}}
    with db.cursor() as cur:
        m = metrics.compute_run_metrics(cur, "r_prefix", billable_weights_by_model=weights)

    assert m["billable_tokens"] == pytest.approx(300.0)


# ---------------------------------------------------------------------------
# finding 9: requests LiteLLM couldn't price (and no `pricing:` override
# could price either) must be surfaced, not silently folded into a total
# that reads as complete.
# ---------------------------------------------------------------------------

def test_unpriced_models_reports_model_and_count():
    db.init_db()
    with db.cursor() as cur:
        _mk_run(cur, "r_unpriced")
        _mk_request(cur, "r_unpriced", 1, model="claude-sonnet-5", response_cost=0.0,
                     input_tokens=100, output_tokens=20, cost_source="unknown")
        _mk_request(cur, "r_unpriced", 2, model="claude-sonnet-5", response_cost=0.0,
                     input_tokens=100, output_tokens=20, cost_source="unknown")
        _mk_request(cur, "r_unpriced", 3, model="claude-sonnet-5", response_cost=0.01,
                     input_tokens=100, output_tokens=20, cost_source="litellm")

    with db.cursor() as cur:
        unpriced = metrics.unpriced_models(cur, "r_unpriced")

    assert unpriced == [{"model": "claude-sonnet-5", "count": 2}]


def test_unpriced_models_empty_when_all_requests_priced():
    db.init_db()
    with db.cursor() as cur:
        _mk_run(cur, "r_priced")
        _mk_request(cur, "r_priced", 1, cost_source="litellm")

    with db.cursor() as cur:
        assert metrics.unpriced_models(cur, "r_priced") == []
