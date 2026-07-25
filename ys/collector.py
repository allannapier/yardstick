"""Production LiteLLM CustomLogger for yardstick.

Field paths here are verified against a real proxied Claude Code-shaped
request (see explore/dumps/), not guessed from LiteLLM's docs. See
explore/ for the probe that produced the ground truth.
"""
import asyncio
import hashlib
import json
import time
import traceback

from litellm.integrations.custom_logger import CustomLogger

from ys import db, paths

_SECRET_KEYS = {"api_key", "authorization", "x-api-key", "api-key"}


def _redact(obj, key=None):
    if key is not None and str(key).lower() in _SECRET_KEYS:
        return "<redacted>"
    if isinstance(obj, dict):
        return {k: _redact(v, key=k) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_redact(v) for v in obj]
    return obj


def _canonical(obj) -> str:
    return json.dumps(_redact(obj), sort_keys=True, separators=(",", ":"), default=str)


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _msg_hashes(messages) -> list:
    if not messages:
        return []
    out = []
    for m in messages:
        if isinstance(m, dict):
            out.append(_sha256(_canonical(m)))
    return out


def _classify_transition(prev: list, cur: list) -> str:
    """Classify how this request's message history relates to the previous
    request's, for detecting harness-side context management.

    Deviates from the spec's literal pseudocode (`cur[0] != prev[0]` as the
    compaction signal): message[0] is almost always the stable system
    prompt, so a mid-conversation summarization -- which shortens the
    history but leaves the system prompt at position 0 untouched -- would
    never trip that check and would be misclassified as a branch. Verified
    against a live probe (see explore/): a shrink-and-diverge-after-index-0
    case landed as 'branch' under the literal spec logic. Keying off
    "did it get shorter" instead of "did position 0 change" is what the
    metric is actually trying to detect.
    """
    if not prev:
        return None
    if cur[: len(prev)] == prev:
        return "continuation"

    k = 0
    for a, b in zip(prev, cur):
        if a == b:
            k += 1
        else:
            break

    if len(cur) < len(prev):
        return "compaction"
    if 0 < k < len(prev):
        return "branch"
    return "reset"


def _safe_token_count(model: str, text: str) -> int:
    if not text:
        return 0
    try:
        import litellm

        return litellm.token_counter(model=model or "gpt-4", text=text)
    except Exception:
        return max(1, len(text) // 4)


def _resolve_run_id(kwargs: dict) -> str:
    try:
        headers = (
            kwargs.get("litellm_params", {})
            .get("proxy_server_request", {})
            .get("headers", {})
            or {}
        )
        run_id = headers.get("x-ys-run")
        if run_id:
            return run_id
    except Exception:
        pass

    try:
        import os

        if os.path.exists(paths.ACTIVE_RUN_PATH):
            with open(paths.ACTIVE_RUN_PATH) as f:
                active = json.load(f)
            return active["run_id"]
    except Exception:
        pass

    return "unattributed"


def _next_seq(cur, run_id: str) -> int:
    row = cur.execute(
        "SELECT COALESCE(MAX(seq), 0) AS m FROM requests WHERE run_id = ?", (run_id,)
    ).fetchone()
    return (row["m"] or 0) + 1


def _last_msg_hashes(cur, run_id: str):
    row = cur.execute(
        "SELECT msg_hashes_json FROM requests WHERE run_id = ? ORDER BY seq DESC LIMIT 1",
        (run_id,),
    ).fetchone()
    if not row or not row["msg_hashes_json"]:
        return []
    return json.loads(row["msg_hashes_json"])


def _ensure_run_exists(cur, run_id: str):
    if run_id == "unattributed":
        cur.execute(
            "INSERT OR IGNORE INTO experiments (id, name, question, task_json, config_yaml, created_at) "
            "VALUES ('unattributed', 'unattributed', NULL, '{}', '', ?)",
            (_now(),),
        )
        cur.execute(
            "INSERT OR IGNORE INTO arms (id, experiment_id, label, factors_json, is_baseline) "
            "VALUES ('unattributed', 'unattributed', 'unattributed', '{}', 0)"
        )
        cur.execute(
            "INSERT OR IGNORE INTO runs (id, experiment_id, arm_id, repeat_idx, started_at) "
            "VALUES ('unattributed', 'unattributed', 'unattributed', 0, ?)",
            (_now(),),
        )


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _fill_fingerprint_if_missing(cur, run_id, model, tool_count, toolset_hash, system_hash, user_agent):
    row = cur.execute("SELECT model FROM runs WHERE id = ?", (run_id,)).fetchone()
    if row is None or row["model"] is not None:
        return
    cur.execute(
        "UPDATE runs SET model=?, tool_count=?, toolset_hash=?, system_prompt_hash=?, harness_user_agent=? "
        "WHERE id=?",
        (model, tool_count, toolset_hash, system_hash, user_agent, run_id),
    )


def extract_record(kwargs: dict, response_obj, start_time, end_time) -> dict:
    """Pure extraction, no I/O. Kept separate from the DB write for testability."""
    slo = kwargs.get("standard_logging_object") or {}
    metadata = slo.get("metadata") or {}
    usage = metadata.get("usage_object") or {}

    complete_input = (kwargs.get("additional_args") or {}).get("complete_input_dict") or {}
    system_text = complete_input.get("system") or kwargs.get("system") or ""
    if isinstance(system_text, list):
        system_text = " ".join(
            b.get("text", "") for b in system_text if isinstance(b, dict)
        )
    tools = complete_input.get("tools") or kwargs.get("tools") or []
    model = slo.get("model") or kwargs.get("model")

    messages = slo.get("messages") or kwargs.get("messages") or []
    msg_hashes = _msg_hashes(messages)

    headers = (
        (kwargs.get("litellm_params") or {}).get("proxy_server_request", {}).get("headers", {})
        or {}
    )

    latency_ms = None
    try:
        latency_ms = (end_time - start_time).total_seconds() * 1000
    except Exception:
        pass

    return {
        "ts": _now(),
        "provider": slo.get("custom_llm_provider") or kwargs.get("custom_llm_provider"),
        "model": model,
        "stream": bool(kwargs.get("stream")),
        "input_tokens": usage.get("prompt_tokens"),
        "cache_creation": usage.get("cache_creation_input_tokens", 0),
        "cache_read": usage.get("cache_read_input_tokens", 0),
        "output_tokens": usage.get("completion_tokens"),
        "response_cost": slo.get("response_cost") or kwargs.get("response_cost") or 0.0,
        "latency_ms": latency_ms,
        "ttft_ms": None,
        "status_code": 200 if slo.get("status") == "success" else 500,
        "error": slo.get("error_str"),
        "msg_count": len(messages),
        "msg_hashes": msg_hashes,
        "system_tokens": _safe_token_count(model, system_text),
        "tools_tokens": _safe_token_count(model, _canonical(tools)) if tools else 0,
        "tool_calls": _extract_tool_calls(response_obj),
        "tool_results": _extract_tool_results(messages),
        "user_agent": headers.get("user-agent"),
        "toolset_hash": _sha256(_canonical(tools)) if tools else None,
        "tool_count": len(tools) if isinstance(tools, list) else 0,
        "system_prompt_hash": _sha256(system_text) if system_text else None,
    }


def _extract_tool_calls(response_obj) -> list:
    out = []
    try:
        choices = getattr(response_obj, "choices", None)
        if choices is None and isinstance(response_obj, dict):
            choices = response_obj.get("choices")
        if not choices:
            return out
        message = choices[0].get("message") if isinstance(choices[0], dict) else getattr(choices[0], "message", None)
        tool_calls = (message or {}).get("tool_calls") if isinstance(message, dict) else getattr(message, "tool_calls", None)
        for tc in tool_calls or []:
            fn = tc.get("function") if isinstance(tc, dict) else getattr(tc, "function", None)
            name = fn.get("name") if isinstance(fn, dict) else getattr(fn, "name", None)
            args_raw = fn.get("arguments") if isinstance(fn, dict) else getattr(fn, "arguments", None)
            call_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
            try:
                args_parsed = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            except Exception:
                args_parsed = args_raw
            input_hash = _sha256(f"{name}:{_canonical(args_parsed)}")
            out.append(
                {
                    "name": name,
                    "input_hash": input_hash,
                    "input_bytes": len(args_raw) if isinstance(args_raw, str) else len(_canonical(args_parsed)),
                    "provider_call_id": call_id,
                }
            )
    except Exception:
        pass
    return out


def _extract_tool_results(messages) -> list:
    """Scan this request's own message history for tool_result blocks, so we
    can backfill is_error/result_tokens on tool_calls logged in prior turns."""
    out = []
    try:
        for m in messages or []:
            content = m.get("content") if isinstance(m, dict) else None
            if not isinstance(content, list):
                continue
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    text = block.get("content")
                    if isinstance(text, list):
                        text = " ".join(
                            b.get("text", "") for b in text if isinstance(b, dict)
                        )
                    out.append(
                        {
                            "tool_use_id": block.get("tool_use_id") or block.get("tool_call_id"),
                            "is_error": bool(block.get("is_error")),
                            "result_tokens": _safe_token_count(None, text or ""),
                        }
                    )
    except Exception:
        pass
    return out


def _run_row_exists(cur, run_id: str) -> bool:
    return cur.execute("SELECT 1 FROM runs WHERE id = ?", (run_id,)).fetchone() is not None


def _write(run_id: str, rec: dict):
    with db.cursor() as cur:
        if run_id != "unattributed" and not _run_row_exists(cur, run_id):
            # A run_id was claimed (header or stale active.json) that doesn't
            # exist in the DB -- e.g. ys start crashed, or a request arrived
            # after ys end cleared the state file. Never drop the data.
            run_id = "unattributed"
        _ensure_run_exists(cur, run_id)
        seq = _next_seq(cur, run_id)
        prev_hashes = _last_msg_hashes(cur, run_id)
        transition = _classify_transition(prev_hashes, rec["msg_hashes"])

        cur.execute(
            """INSERT INTO requests
               (run_id, seq, ts, provider, model, stream, input_tokens, cache_creation,
                cache_read, output_tokens, response_cost, latency_ms, ttft_ms, status_code,
                error, msg_count, msg_hashes_json, system_tokens, tools_tokens, transition)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                run_id,
                seq,
                rec["ts"],
                rec["provider"],
                rec["model"],
                int(rec["stream"]),
                rec["input_tokens"],
                rec["cache_creation"],
                rec["cache_read"],
                rec["output_tokens"],
                rec["response_cost"],
                rec["latency_ms"],
                rec["ttft_ms"],
                rec["status_code"],
                rec["error"],
                rec["msg_count"],
                json.dumps(rec["msg_hashes"]),
                rec["system_tokens"],
                rec["tools_tokens"],
                transition,
            ),
        )
        request_id = cur.lastrowid

        for tc in rec["tool_calls"]:
            cur.execute(
                """INSERT INTO tool_calls (request_id, run_id, name, input_hash, input_bytes, provider_call_id)
                   VALUES (?,?,?,?,?,?)""",
                (request_id, run_id, tc["name"], tc["input_hash"], tc["input_bytes"], tc["provider_call_id"]),
            )

        for tr in rec["tool_results"]:
            if not tr["tool_use_id"]:
                continue
            cur.execute(
                """UPDATE tool_calls SET is_error=?, result_tokens=?
                   WHERE run_id=? AND provider_call_id=?""",
                (int(tr["is_error"]), tr["result_tokens"], run_id, tr["tool_use_id"]),
            )

        if run_id != "unattributed" and rec["status_code"] == 200:
            _fill_fingerprint_if_missing(
                cur,
                run_id,
                rec["model"],
                rec["tool_count"],
                rec["toolset_hash"],
                rec["system_prompt_hash"],
                rec["user_agent"],
            )


class YardstickLogger(CustomLogger):
    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        await self._handle(kwargs, response_obj, start_time, end_time)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        await self._handle(kwargs, response_obj, start_time, end_time)

    async def _handle(self, kwargs, response_obj, start_time, end_time):
        try:
            run_id = _resolve_run_id(kwargs)
            rec = extract_record(kwargs, response_obj, start_time, end_time)
            await asyncio.get_event_loop().run_in_executor(None, _write, run_id, rec)
        except Exception:
            traceback.print_exc()


yardstick_logger = YardstickLogger()
