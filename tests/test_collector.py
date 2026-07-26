import asyncio
import concurrent.futures
import datetime
import sqlite3

from ys import db, dropped
from ys.collector import (
    YardstickLogger,
    _classify_transition,
    _extract_tool_calls,
    _extract_tool_results,
    _msg_hashes,
    _redact,
    _write,
    extract_record,
)


# --- redaction -------------------------------------------------------------


def test_redact_flat_key():
    assert _redact({"api_key": "sk-ant-secret"}) == {"api_key": "<redacted>"}


def test_redact_nested_key_any_depth():
    obj = {"litellm_params": {"metadata": {"user_api_key_auth": {"api_key": "sk-ant-secret"}}}}
    redacted = _redact(obj)
    assert redacted["litellm_params"]["metadata"]["user_api_key_auth"]["api_key"] == "<redacted>"


def test_redact_case_insensitive_authorization_header():
    obj = {"headers": {"Authorization": "Bearer sk-ant-secret"}}
    assert _redact(obj)["headers"]["Authorization"] == "<redacted>"


def test_redact_leaves_non_secret_fields_alone():
    obj = {"model": "claude-haiku-4-5-20251001", "nested": {"x": 1}}
    assert _redact(obj) == obj


# --- message hashing / transition classification ---------------------------


def test_msg_hashes_stable_and_order_sensitive():
    a = [{"role": "user", "content": "hi"}]
    b = [{"role": "user", "content": "hi"}]
    c = [{"role": "assistant", "content": "hi"}]
    assert _msg_hashes(a) == _msg_hashes(b)
    assert _msg_hashes(a) != _msg_hashes(c)


def test_transition_first_request_is_none():
    assert _classify_transition([], ["h1"]) is None


def test_transition_continuation_on_append():
    prev = ["h1", "h2"]
    cur = ["h1", "h2", "h3"]
    assert _classify_transition(prev, cur) == "continuation"


def test_transition_compaction_on_shrink():
    # Regression test for the live-probe finding: a shorter history counts
    # as compaction even when message[0] (the stable system prompt) is
    # unchanged -- the spec's literal "cur[0] != prev[0]" check misses this.
    prev = ["sys", "h1", "h2", "h3"]
    cur = ["sys", "summary", "h4"]
    assert _classify_transition(prev, cur) == "compaction"


def test_transition_branch_on_same_length_divergence():
    prev = ["sys", "h1", "h2"]
    cur = ["sys", "h1", "h2different"]
    assert _classify_transition(prev, cur) == "branch"


def test_transition_reset_on_no_shared_prefix_same_or_growing_length():
    prev = ["h1", "h2"]
    cur = ["totally", "different", "conversation"]
    assert _classify_transition(prev, cur) == "reset"


# --- tool call / tool result extraction -------------------------------------


class FakeMessage:
    def __init__(self, tool_calls):
        self.tool_calls = tool_calls


class FakeChoice:
    def __init__(self, message):
        self.message = message


class FakeResponse:
    def __init__(self, choices):
        self.choices = choices


def test_extract_tool_calls_from_dict_shaped_response():
    response = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "toolu_1",
                            "function": {"name": "list_files", "arguments": '{"path": "."}'},
                        }
                    ]
                }
            }
        ]
    }
    calls = _extract_tool_calls(response)
    assert len(calls) == 1
    assert calls[0]["name"] == "list_files"
    assert calls[0]["provider_call_id"] == "toolu_1"


def test_extract_tool_calls_identical_args_produce_identical_hash():
    def make(args):
        return {
            "choices": [
                {"message": {"tool_calls": [{"id": "x", "function": {"name": "read", "arguments": args}}]}}
            ]
        }

    h1 = _extract_tool_calls(make('{"path": "a.py"}'))[0]["input_hash"]
    h2 = _extract_tool_calls(make('{"path": "a.py"}'))[0]["input_hash"]
    h3 = _extract_tool_calls(make('{"path": "b.py"}'))[0]["input_hash"]
    assert h1 == h2
    assert h1 != h3


def test_extract_tool_calls_no_tool_calls_returns_empty():
    assert _extract_tool_calls({"choices": [{"message": {"content": "no tools here"}}]}) == []


def test_extract_tool_results_from_tool_result_block():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "is_error": True, "content": "boom"}
            ],
        }
    ]
    results = _extract_tool_results(messages)
    assert len(results) == 1
    assert results[0]["tool_use_id"] == "toolu_1"
    assert results[0]["is_error"] is True


# --- end-to-end extract_record + _write against the schema -----------------


def _fake_kwargs(system="You are a coding agent.", tools=None, messages=None, model="claude-haiku-4-5-20251001", status="success"):
    tools = tools if tools is not None else []
    messages = messages if messages is not None else [{"role": "user", "content": "hi"}]
    return {
        "standard_logging_object": {
            "model": f"anthropic/{model}",
            "custom_llm_provider": "anthropic",
            "response_cost": 0.001,
            "status": status,
            "error_str": None,
            "messages": [{"role": "system", "content": system}] + messages,
            "metadata": {
                "usage_object": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "cache_creation_input_tokens": 5,
                    "cache_read_input_tokens": 50,
                },
                "requester_custom_headers": {},
            },
        },
        "additional_args": {"complete_input_dict": {"system": system, "tools": tools}},
        "system": system,
        "tools": tools,
        "model": model,
        "messages": messages,
        "stream": False,
        "litellm_params": {"proxy_server_request": {"headers": {"user-agent": "pytest"}}},
    }


def test_extract_record_pulls_usage_and_overhead_fields():
    kwargs = _fake_kwargs(tools=[{"type": "function", "function": {"name": "list_files"}}])
    start = datetime.datetime(2026, 1, 1, 0, 0, 0)
    end = datetime.datetime(2026, 1, 1, 0, 0, 1)
    rec = extract_record(kwargs, FakeResponse([]), start, end)

    assert rec["input_tokens"] == 100
    assert rec["output_tokens"] == 20
    assert rec["cache_creation"] == 5
    assert rec["cache_read"] == 50
    assert rec["response_cost"] == 0.001
    assert rec["tool_count"] == 1
    assert rec["system_tokens"] > 0
    assert rec["tools_tokens"] > 0
    assert rec["latency_ms"] == 1000.0


def test_write_falls_back_to_unattributed_for_unknown_run_id():
    db.init_db()
    rec = extract_record(_fake_kwargs(), FakeResponse([]), None, None)
    _write("some-run-that-was-never-started", rec)

    with db.cursor() as cur:
        row = cur.execute(
            "SELECT run_id FROM requests WHERE run_id = 'unattributed'"
        ).fetchone()
    assert row is not None


def test_write_attributes_to_existing_run_and_backfills_tool_error():
    db.init_db()
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO experiments (id, name, question, task_json, config_yaml, created_at) "
            "VALUES ('e','e',NULL,'{}','','2026-01-01')"
        )
        cur.execute(
            "INSERT INTO arms (id, experiment_id, label, factors_json, is_baseline) "
            "VALUES ('a','e','a','{}',0)"
        )
        cur.execute(
            "INSERT INTO runs (id, experiment_id, arm_id, repeat_idx, started_at) "
            "VALUES ('r','e','a',0,'2026-01-01')"
        )

    # turn 1: model calls a tool
    turn1_response = FakeResponse(
        [
            {
                "message": {
                    "tool_calls": [
                        {"id": "toolu_1", "function": {"name": "run_tests", "arguments": "{}"}}
                    ]
                }
            }
        ]
    )
    rec1 = extract_record(_fake_kwargs(messages=[{"role": "user", "content": "go"}]), turn1_response, None, None)
    _write("r", rec1)

    # turn 2: harness reports that tool call failed
    turn2_messages = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_1", "name": "run_tests"}]},
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "is_error": True, "content": "FAILED"}],
        },
    ]
    rec2 = extract_record(_fake_kwargs(messages=turn2_messages), FakeResponse([]), None, None)
    _write("r", rec2)

    with db.cursor() as cur:
        tc = cur.execute("SELECT is_error FROM tool_calls WHERE run_id='r' AND name='run_tests'").fetchone()
        run_row = cur.execute("SELECT model, tool_count FROM runs WHERE id='r'").fetchone()
        transitions = [
            r["transition"]
            for r in cur.execute("SELECT transition FROM requests WHERE run_id='r' ORDER BY seq")
        ]

    assert tc["is_error"] == 1
    assert run_row["model"] is not None  # fingerprint filled from first request
    assert transitions[0] is None
    assert transitions[1] == "continuation"


def test_write_scopes_transitions_within_a_thread_not_across_interleaved_traffic():
    """Regression test for finding 4 (IMPROVEMENTS.md): Claude Code
    interleaves background (harness title-generation) requests between main
    conversation turns. A background request must not be classified against
    the main thread's history (which would fabricate a compaction/reset
    event), and must not become what the *next* main-thread turn is
    classified against either."""
    db.init_db()
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO experiments (id, name, question, task_json, config_yaml, created_at) "
            "VALUES ('e','e',NULL,'{}','','2026-01-01')"
        )
        cur.execute(
            "INSERT INTO arms (id, experiment_id, label, factors_json, is_baseline) "
            "VALUES ('a','e','a','{}',0)"
        )
        cur.execute(
            "INSERT INTO runs (id, experiment_id, arm_id, repeat_idx, started_at) "
            "VALUES ('r','e','a',0,'2026-01-01')"
        )

    main_system = "You are a coding agent."
    bg_system = "Summarize this conversation in five words."

    main_turn1 = [{"role": "user", "content": "please fix the bug"}]
    main_turn2 = main_turn1 + [
        {"role": "assistant", "content": "looking into it"},
        {"role": "user", "content": "any luck?"},
    ]
    main_turn3 = main_turn2 + [
        {"role": "assistant", "content": "found it"},
        {"role": "user", "content": "great, ship it"},
    ]
    bg_turn = [{"role": "user", "content": "conversation so far: ..."}]

    def emit(system, messages):
        rec = extract_record(_fake_kwargs(system=system, messages=messages), FakeResponse([]), None, None)
        _write("r", rec)

    emit(main_system, main_turn1)
    emit(main_system, main_turn2)
    emit(bg_system, bg_turn)  # interleaved background call
    emit(main_system, main_turn3)

    with db.cursor() as cur:
        rows = cur.execute(
            "SELECT seq, transition, thread_key FROM requests WHERE run_id='r' ORDER BY seq"
        ).fetchall()

    assert [r["transition"] for r in rows] == [None, "continuation", None, "continuation"]
    assert rows[0]["thread_key"] == rows[1]["thread_key"] == rows[3]["thread_key"]
    assert rows[2]["thread_key"] != rows[0]["thread_key"]


def test_write_thread_survives_a_compaction_that_rewrites_the_anchor_message():
    """A harness-side compaction summarizes/rewrites the early history --
    including whatever message an anchor-based thread key would have
    pinned on. thread_key must be assigned by chain-following (does this
    request's history plausibly extend the thread's last request?), not by
    re-hashing a fixed anchor, or a legitimate compaction would be
    misclassified as the start of a brand new thread and its metrics
    would silently fall out of the main conversation."""
    db.init_db()
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO experiments (id, name, question, task_json, config_yaml, created_at) "
            "VALUES ('e','e',NULL,'{}','','2026-01-01')"
        )
        cur.execute(
            "INSERT INTO arms (id, experiment_id, label, factors_json, is_baseline) "
            "VALUES ('a','e','a','{}',0)"
        )
        cur.execute(
            "INSERT INTO runs (id, experiment_id, arm_id, repeat_idx, started_at) "
            "VALUES ('r','e','a',0,'2026-01-01')"
        )

    system = "You are a coding agent."
    turn1 = [{"role": "user", "content": "please fix the bug"}]
    turn2 = turn1 + [
        {"role": "assistant", "content": "looking into it"},
        {"role": "user", "content": "any luck?"},
    ]
    # compaction: the original first message is gone, replaced by a summary
    # -- an anchor-hash-based thread_key would see this as unrelated to
    # turn1/turn2's anchor and start a new thread.
    turn3_compacted = [
        {"role": "user", "content": "summary: fixing a bug, made progress"},
        {"role": "user", "content": "keep going"},
    ]

    def emit(messages):
        rec = extract_record(_fake_kwargs(system=system, messages=messages), FakeResponse([]), None, None)
        _write("r", rec)

    emit(turn1)
    emit(turn2)
    emit(turn3_compacted)

    with db.cursor() as cur:
        rows = cur.execute(
            "SELECT seq, transition, thread_key FROM requests WHERE run_id='r' ORDER BY seq"
        ).fetchall()

    assert rows[2]["transition"] == "compaction"
    assert rows[2]["thread_key"] == rows[0]["thread_key"] == rows[1]["thread_key"]


def _messages(n, prefix):
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"{prefix}-{i}"}
        for i in range(n)
    ]


def test_write_rejects_a_same_system_prompt_short_unrelated_call_as_compaction():
    """Regression test for finding 25 (IMPROVEMENTS.md): _resolve_thread's
    system_prompt_hash match is currently the *only* thing keeping an
    unrelated conversation from being absorbed into the main thread, since
    _classify_transition calls any shorter history "compaction" regardless
    of shared prefix. If a subagent or background call ever reused the
    main thread's system prompt, it must not be pulled in just because it's
    shorter -- a message count collapsing from 12 to 2 (dropping to a
    sixth) is far more consistent with "unrelated short conversation" than
    "compaction", and must start its own thread instead."""
    db.init_db()
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO experiments (id, name, question, task_json, config_yaml, created_at) "
            "VALUES ('e','e',NULL,'{}','','2026-01-01')"
        )
        cur.execute(
            "INSERT INTO arms (id, experiment_id, label, factors_json, is_baseline) "
            "VALUES ('a','e','a','{}',0)"
        )
        cur.execute(
            "INSERT INTO runs (id, experiment_id, arm_id, repeat_idx, started_at) "
            "VALUES ('r','e','a',0,'2026-01-01')"
        )

    system = "You are a coding agent."
    main_history = _messages(12, "main")
    # An unrelated exchange that happens to carry the *same* system prompt
    # (the coincidence finding 25 is about) and shares no prefix with the
    # main history -- exactly the shape a same-system-prompt subagent or
    # background call would have.
    unrelated_short = _messages(2, "unrelated")

    def emit(messages):
        rec = extract_record(_fake_kwargs(system=system, messages=messages), FakeResponse([]), None, None)
        _write("r", rec)

    emit(main_history)
    emit(unrelated_short)

    with db.cursor() as cur:
        rows = cur.execute(
            "SELECT seq, transition, thread_key FROM requests WHERE run_id='r' ORDER BY seq"
        ).fetchall()

    assert rows[1]["transition"] is None  # not absorbed as a fabricated compaction
    assert rows[1]["thread_key"] != rows[0]["thread_key"]


def test_write_still_follows_a_plausible_large_compaction():
    """The fix for finding 25 must not make real compactions look like new
    threads (that's an explicit non-goal in IMPROVEMENTS.md). A compaction
    that drops a long history to a third of its length -- summary plus a
    handful of recent turns, the shape a real harness-side compaction
    takes -- must still resolve to the same thread."""
    db.init_db()
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO experiments (id, name, question, task_json, config_yaml, created_at) "
            "VALUES ('e','e',NULL,'{}','','2026-01-01')"
        )
        cur.execute(
            "INSERT INTO arms (id, experiment_id, label, factors_json, is_baseline) "
            "VALUES ('a','e','a','{}',0)"
        )
        cur.execute(
            "INSERT INTO runs (id, experiment_id, arm_id, repeat_idx, started_at) "
            "VALUES ('r','e','a',0,'2026-01-01')"
        )

    system = "You are a coding agent."
    main_history = _messages(12, "main")
    # Summary + a handful of recent turns -- retains exactly a third of the
    # pre-compaction message count, the ratio's inclusive boundary.
    compacted = _messages(4, "summary")

    def emit(messages):
        rec = extract_record(_fake_kwargs(system=system, messages=messages), FakeResponse([]), None, None)
        _write("r", rec)

    emit(main_history)
    emit(compacted)

    with db.cursor() as cur:
        rows = cur.execute(
            "SELECT seq, transition, thread_key FROM requests WHERE run_id='r' ORDER BY seq"
        ).fetchall()

    assert rows[1]["transition"] == "compaction"
    assert rows[1]["thread_key"] == rows[0]["thread_key"]


def test_write_does_not_stamp_fingerprint_from_a_failed_request():
    """A harness's rejected/throwaway first call (e.g. opencode's title-gen
    ping hitting a model the proxy doesn't recognise) must not permanently
    stamp the run's model/toolset fingerprint -- the next successful request
    should win instead."""
    db.init_db()
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO experiments (id, name, question, task_json, config_yaml, created_at) "
            "VALUES ('e','e',NULL,'{}','','2026-01-01')"
        )
        cur.execute(
            "INSERT INTO arms (id, experiment_id, label, factors_json, is_baseline) "
            "VALUES ('a','e','a','{}',0)"
        )
        cur.execute(
            "INSERT INTO runs (id, experiment_id, arm_id, repeat_idx, started_at) "
            "VALUES ('r','e','a',0,'2026-01-01')"
        )

    failed_rec = extract_record(
        _fake_kwargs(model="claude-haiku-4-5-20251001", status="failure"), FakeResponse([]), None, None
    )
    _write("r", failed_rec)

    ok_rec = extract_record(
        _fake_kwargs(model="claude-sonnet-5", status="success"), FakeResponse([]), None, None
    )
    _write("r", ok_rec)

    with db.cursor() as cur:
        run_row = cur.execute("SELECT model FROM runs WHERE id='r'").fetchone()

    assert run_row["model"] == "anthropic/claude-sonnet-5"


def test_write_seq_allocation_is_race_free_under_concurrent_writers():
    """Regression test for finding 7: `_next_seq`'s read-then-insert used to
    run with no lock held, so concurrent writers to the same run (parallel
    tool use, a subagent) could read the same MAX(seq) and both insert it.
    `_write` now allocates seq inside a `BEGIN IMMEDIATE` transaction, which
    serializes writers against this database file -- 20 threads hammering
    the same run must still land 20 distinct, gapless seq values.

    Uses a real run id, not 'unattributed': `_ensure_run_exists` only issues
    writes (which incidentally also acquire the write lock, even without
    the BEGIN IMMEDIATE fix) for the 'unattributed' bucket, so that id
    doesn't exercise the race an ordinary run hits. Verified against a
    build of this test with the BEGIN IMMEDIATE line reverted: it reliably
    raised `sqlite3.IntegrityError` from the UNIQUE(run_id, seq) backstop
    well under 20 concurrent writers."""
    db.init_db()
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO experiments (id, name, question, task_json, config_yaml, created_at) "
            "VALUES ('e','e',NULL,'{}','','2026-01-01')"
        )
        cur.execute(
            "INSERT INTO arms (id, experiment_id, label, factors_json, is_baseline) "
            "VALUES ('a','e','a','{}',0)"
        )
        cur.execute(
            "INSERT INTO runs (id, experiment_id, arm_id, repeat_idx, started_at) "
            "VALUES ('r','e','a',0,'2026-01-01')"
        )

    rec = extract_record(_fake_kwargs(), FakeResponse([]), None, None)
    n = 20

    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
        list(pool.map(lambda _: _write("r", rec), range(n)))

    with db.cursor() as cur:
        seqs = [
            r["seq"]
            for r in cur.execute("SELECT seq FROM requests WHERE run_id = 'r' ORDER BY seq")
        ]
    assert seqs == list(range(1, n + 1))


# --- write retry / drop accounting ------------------------------------------


def test_handle_retries_a_locked_database_and_still_writes(monkeypatch):
    """A transient `sqlite3.OperationalError` (e.g. a losing race against
    another writer) must not drop the record outright -- it should be
    retried and succeed once the lock clears."""
    db.init_db()
    calls = {"n": 0}
    real_write = _write

    def flaky_write(run_id, rec):
        calls["n"] += 1
        if calls["n"] < 3:
            raise sqlite3.OperationalError("database is locked")
        real_write(run_id, rec)

    monkeypatch.setattr("ys.collector._write", flaky_write)

    logger = YardstickLogger()
    kwargs = _fake_kwargs()
    asyncio.run(logger._handle(kwargs, FakeResponse([]), None, None))

    assert calls["n"] == 3
    assert dropped.count() == 0
    with db.cursor() as cur:
        row = cur.execute("SELECT run_id FROM requests WHERE run_id = 'unattributed'").fetchone()
    assert row is not None


def test_handle_drops_and_records_after_exhausting_retries(monkeypatch):
    """Once retries are exhausted the request is genuinely lost -- it must
    be counted so a lossy run is visible (`ys status`), not just logged to
    a file nobody watches mid-run."""
    db.init_db()

    def always_locked(run_id, rec):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr("ys.collector._write", always_locked)

    logger = YardstickLogger()
    kwargs = _fake_kwargs()
    asyncio.run(logger._handle(kwargs, FakeResponse([]), None, None))

    assert dropped.count() == 1


def test_handle_counts_non_operational_errors_as_dropped_too(monkeypatch):
    """Any failure that keeps a request from landing in the database is a
    dropped request, not only lock contention."""
    db.init_db()

    def boom(run_id, rec):
        raise ValueError("boom")

    monkeypatch.setattr("ys.collector._write", boom)

    logger = YardstickLogger()
    kwargs = _fake_kwargs()
    asyncio.run(logger._handle(kwargs, FakeResponse([]), None, None))

    assert dropped.count() == 1
