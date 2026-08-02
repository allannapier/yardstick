import json

from ys import db, portkey_collector, runs
from ys.experiment import Arm, Experiment, Task


def _anthropic_entry(**overrides):
    entry = {
        "id": "log-1",
        "created_at": "2026-08-02T10:00:00Z",
        "ai_provider": "anthropic",
        "ai_model": "claude-sonnet-5",
        "request_tokens": 100,
        "response_tokens": 20,
        "cost": 0.0042,
        "response_time": 850,
        "status_code": 200,
        "metadata": json.dumps({"ys_run_id": "run-1"}),
        "request": json.dumps(
            {
                "model": "claude-sonnet-5",
                "system": "you are a helpful agent",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"name": "bash", "description": "run a shell command"}],
            }
        ),
        "response": json.dumps(
            {
                "content": [
                    {"type": "text", "text": "sure"},
                    {"type": "tool_use", "id": "call_1", "name": "bash", "input": {"cmd": "ls"}},
                ]
            }
        ),
    }
    entry.update(overrides)
    return entry


# --- extract_record_from_portkey (pure) -------------------------------------


def test_extract_record_maps_core_fields():
    rec = portkey_collector.extract_record_from_portkey(_anthropic_entry())
    assert rec["provider"] == "anthropic"
    assert rec["model"] == "claude-sonnet-5"
    assert rec["input_tokens"] == 100
    assert rec["output_tokens"] == 20
    assert rec["response_cost"] == 0.0042
    assert rec["latency_ms"] == 850
    assert rec["status_code"] == 200
    assert rec["msg_count"] == 1
    assert rec["system_prompt_hash"] is not None
    assert rec["tool_count"] == 1


def test_extract_record_finds_anthropic_native_tool_use_blocks():
    rec = portkey_collector.extract_record_from_portkey(_anthropic_entry())
    assert len(rec["tool_calls"]) == 1
    assert rec["tool_calls"][0]["name"] == "bash"
    assert rec["tool_calls"][0]["provider_call_id"] == "call_1"


def test_extract_record_falls_back_to_openai_shaped_tool_calls():
    entry = _anthropic_entry(
        response=json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "call_2",
                                    "function": {"name": "grep", "arguments": '{"pattern": "foo"}'},
                                }
                            ]
                        }
                    }
                ]
            }
        )
    )
    rec = portkey_collector.extract_record_from_portkey(entry)
    assert len(rec["tool_calls"]) == 1
    assert rec["tool_calls"][0]["name"] == "grep"


def test_extract_record_handles_openai_shaped_system_message():
    entry = _anthropic_entry(
        request=json.dumps(
            {
                "model": "gpt-4o",
                "messages": [
                    {"role": "system", "content": "you are a helpful agent"},
                    {"role": "user", "content": "hi"},
                ],
            }
        )
    )
    rec = portkey_collector.extract_record_from_portkey(entry)
    assert rec["msg_count"] == 1  # system message stripped out of the conversation proper
    assert rec["system_prompt_hash"] is not None


def test_extract_record_marks_error_for_non_200_status():
    entry = _anthropic_entry(status_code=500, response=json.dumps({"error": "boom"}))
    rec = portkey_collector.extract_record_from_portkey(entry)
    assert rec["status_code"] == 500
    assert rec["error"] and "boom" in rec["error"]


# --- ingest() -- network calls mocked ----------------------------------------


def _seed_run(monkeypatch, run_id="run-1"):
    experiment = Experiment(
        experiment="e1",
        task=Task(id="t0", success_check="true"),
        arms=[Arm(id="a", factors={"model": "claude-sonnet-5"})],
    )
    runs.begin_run(experiment, "experiment: e1\ntask: {}\narms: []\n", "a")
    # begin_run generates its own uuid run_id; overwrite active state's id isn't
    # practical here, so instead grab whatever id it actually assigned.
    with db.cursor() as cur:
        row = cur.execute("SELECT id FROM runs ORDER BY rowid DESC LIMIT 1").fetchone()
    return row["id"]


def test_ingest_writes_matching_records_and_skips_other_runs(monkeypatch):
    run_id = _seed_run(monkeypatch)

    monkeypatch.setenv("PORTKEY_ADMIN_API_KEY", "admin-key")
    monkeypatch.setattr(portkey_collector, "_create_export", lambda *a, **k: "export-1")
    monkeypatch.setattr(portkey_collector, "_start_export", lambda *a, **k: None)
    monkeypatch.setattr(portkey_collector, "_poll_until_complete", lambda *a, **k: {"status": "complete"})

    mine = _anthropic_entry(metadata=json.dumps({"ys_run_id": run_id}))
    other = _anthropic_entry(metadata=json.dumps({"ys_run_id": "some-other-run"}))
    monkeypatch.setattr(portkey_collector, "_download", lambda export_id: [mine, other])

    written = portkey_collector.ingest(run_id)

    assert written == 1
    with db.cursor() as cur:
        rows = cur.execute("SELECT * FROM requests WHERE run_id = ?", (run_id,)).fetchall()
    assert len(rows) == 1
    assert rows[0]["model"] == "claude-sonnet-5"


def test_ingest_is_idempotent_on_retry(monkeypatch):
    run_id = _seed_run(monkeypatch)

    monkeypatch.setenv("PORTKEY_ADMIN_API_KEY", "admin-key")
    monkeypatch.setattr(portkey_collector, "_create_export", lambda *a, **k: "export-1")
    monkeypatch.setattr(portkey_collector, "_start_export", lambda *a, **k: None)
    monkeypatch.setattr(portkey_collector, "_poll_until_complete", lambda *a, **k: {"status": "complete"})
    mine = _anthropic_entry(metadata=json.dumps({"ys_run_id": run_id}))
    monkeypatch.setattr(portkey_collector, "_download", lambda export_id: [mine])

    portkey_collector.ingest(run_id)
    written_second_time = portkey_collector.ingest(run_id)

    assert written_second_time == 1
    with db.cursor() as cur:
        rows = cur.execute("SELECT * FROM requests WHERE run_id = ?", (run_id,)).fetchall()
    assert len(rows) == 1  # replaced, not duplicated


def test_ingest_raises_for_unknown_run():
    try:
        portkey_collector.ingest("no-such-run")
        assert False, "expected PortkeyCollectorError"
    except portkey_collector.PortkeyCollectorError as e:
        assert "no such run" in str(e)


def test_admin_headers_requires_admin_api_key(monkeypatch):
    monkeypatch.delenv("PORTKEY_ADMIN_API_KEY", raising=False)
    try:
        portkey_collector._admin_headers()
        assert False, "expected PortkeyCollectorError"
    except portkey_collector.PortkeyCollectorError as e:
        assert "PORTKEY_ADMIN_API_KEY" in str(e)
