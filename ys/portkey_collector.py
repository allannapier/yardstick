"""Post-run ingestion of Portkey Cloud's own request logs into yardstick's
database -- the Portkey-backend equivalent of ys/collector.py's LiteLLM
CustomLogger, run *after* a run ends instead of inline during it. See
ys/portkey_backend.py's module docstring for why: Portkey Cloud is a remote
service, so ys/portkey_app.py's local passthrough has no in-process Python
hook to capture request/response data as it happens the way LiteLLM's proxy
provides one. This module closes that gap by pulling the same data back out
of Portkey afterward, via Portkey's Logs Export API, and writing it through
ys/collector.py's own `_write` -- so it lands in exactly the same
requests/tool_calls tables, in the same shape, that `ys compare`/`ys
report`/`ys export` already know how to read.

`ys end` calls `ingest()` automatically once a run finishes (best-effort --
see ys/cli.py); `ys proxy pull-logs <run_id>` calls it manually, for a retry
if Portkey's own log pipeline hadn't finished indexing the run yet.

CAVEAT -- unverified against a live account: the endpoint paths, request/
response shapes, and field names below are transcribed from Portkey's
published docs (the Logs Export API is beta there), not confirmed against a
real call. Two things worth checking with whoever admins your work Portkey
account before trusting this for a real comparison:

  1. That your account's plan actually exposes the Logs Export API at all
     (it may be gated to certain tiers) -- and that PORTKEY_ADMIN_API_KEY,
     an Admin-scoped key distinct from PORTKEY_API_KEY (the gateway key
     ys/portkey_app.py forwards requests with), is the credential it wants.
  2. Whether `filters.metadata` on the export actually narrows by the
     x-portkey-metadata tag ys/portkey_app.py attaches to each request, or
     is silently ignored -- `ingest()` re-filters by that same tag
     client-side either way (see its docstring), so a wrong assumption here
     costs extra downloaded records, not wrong data.

The workflow (Portkey's own multi-step export flow, not a single call):

  1. POST /v1/logs/exports            -- create an export job, filtered by
     time_of_generation_min/max (the run's [started_at, ended_at] window).
  2. POST /v1/logs/exports/{id}/start -- kick the created job off.
  3. GET  /v1/logs/exports/{id}       -- poll until its status is complete.
  4. GET  /v1/logs/exports/{id}/download -- fetch the JSONL payload.
"""
import json
import time
import urllib.error
import urllib.request
from typing import Optional

from ys import db
from ys.collector import (
    _extract_tool_calls,
    _extract_tool_results,
    _msg_hashes,
    _now,
    _safe_token_count,
    _sha256,
    _write,
)

PORTKEY_BASE_URL = "https://api.portkey.ai"
POLL_INTERVAL_S = 2.0
POLL_TIMEOUT_S = 120.0
_COMPLETE_STATUSES = {"complete", "completed", "success"}
_FAILED_STATUSES = {"failed", "error"}


class PortkeyCollectorError(Exception):
    pass


def _admin_headers() -> dict:
    import os

    key = os.environ.get("PORTKEY_ADMIN_API_KEY")
    if not key:
        raise PortkeyCollectorError(
            "PORTKEY_ADMIN_API_KEY is not set -- the Logs Export API needs an "
            "Admin-scoped Portkey key, distinct from PORTKEY_API_KEY (the gateway key "
            "ys/portkey_app.py forwards requests with). Ask whoever admins your work "
            "Portkey account for one, and confirm your account's plan exposes this API "
            "at all -- see ys/portkey_collector.py's module docstring."
        )
    return {"x-portkey-api-key": key, "content-type": "application/json"}


def _request(method: str, path: str, body: Optional[dict] = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"{PORTKEY_BASE_URL}{path}", data=data, headers=_admin_headers(), method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        raise PortkeyCollectorError(f"Portkey API {method} {path} failed: {e.code} {e.read().decode()[:500]}")
    except urllib.error.URLError as e:
        raise PortkeyCollectorError(f"Portkey API {method} {path} unreachable: {e}")


def _create_export(started_at: str, ended_at: str, run_id: str) -> str:
    body = {
        "filters": {
            "time_of_generation_min": started_at,
            "time_of_generation_max": ended_at,
            "metadata": {"ys_run_id": run_id},
        },
        "requested_data": [
            "id",
            "trace_id",
            "created_at",
            "request",
            "response",
            "ai_provider",
            "ai_model",
            "request_tokens",
            "response_tokens",
            "total_tokens",
            "cost",
            "response_time",
            "metadata",
            "status_code",
        ],
    }
    resp = _request("POST", "/v1/logs/exports", body)
    export_id = resp.get("id")
    if not export_id:
        raise PortkeyCollectorError(f"Portkey did not return an export id: {resp}")
    return export_id


def _start_export(export_id: str):
    _request("POST", f"/v1/logs/exports/{export_id}/start")


def _poll_until_complete(export_id: str) -> dict:
    deadline = time.time() + POLL_TIMEOUT_S
    while True:
        resp = _request("GET", f"/v1/logs/exports/{export_id}")
        status = resp.get("status")
        if status in _COMPLETE_STATUSES:
            return resp
        if status in _FAILED_STATUSES:
            raise PortkeyCollectorError(f"Portkey export {export_id} failed: {resp}")
        if time.time() >= deadline:
            raise PortkeyCollectorError(
                f"Portkey export {export_id} did not complete within {POLL_TIMEOUT_S:.0f}s -- "
                "logs may still be indexing; run `ys proxy pull-logs` again shortly."
            )
        time.sleep(POLL_INTERVAL_S)


def _download(export_id: str) -> list:
    req = urllib.request.Request(
        f"{PORTKEY_BASE_URL}/v1/logs/exports/{export_id}/download", headers=_admin_headers()
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode()
    except urllib.error.HTTPError as e:
        raise PortkeyCollectorError(f"Portkey export download failed: {e.code} {e.read().decode()[:500]}")
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def _parse_json_field(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return {}
    return value or {}


def _messages_and_system(request_body: dict) -> tuple:
    """`request_body` is either Anthropic-Messages-shaped or OpenAI-Chat-
    Completions-shaped, whichever wire protocol the harness spoke through
    ys/portkey_app.py's passthrough. Both carry the conversation under
    `messages`; Anthropic keeps the system prompt in its own top-level
    `system` field (string or content-block list -- same shape
    ys/collector.py's `extract_record` already handles for the LiteLLM
    path), OpenAI folds it into messages[0] with role=system instead."""
    messages = request_body.get("messages") or []
    system_text = request_body.get("system") or ""
    if isinstance(system_text, list):
        system_text = " ".join(b.get("text", "") for b in system_text if isinstance(b, dict))
    if not system_text and messages and isinstance(messages[0], dict) and messages[0].get("role") == "system":
        content = messages[0].get("content")
        system_text = content if isinstance(content, str) else ""
        messages = messages[1:]
    return messages, system_text


def _extract_tool_calls_from_response(response_body) -> list:
    """Handles either wire shape Portkey's export may have logged for the
    response, since which one it actually stores for an Anthropic-Messages-
    shaped call -- native content-block `tool_use` vs. normalized OpenAI-
    style `choices[].message.tool_calls` -- is unconfirmed without a live
    account to check (see this module's docstring). Anthropic's native
    shape is tried first, since Claude Code -- the harness this rig mainly
    drives -- speaks it directly through ys/portkey_app.py's passthrough;
    falls back to ys/collector.py's OpenAI-shaped `_extract_tool_calls` if
    no `tool_use` blocks are found."""
    content = response_body.get("content") if isinstance(response_body, dict) else None
    if isinstance(content, list):
        out = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                args = block.get("input") or {}
                args_json = json.dumps(args, sort_keys=True, default=str)
                out.append(
                    {
                        "name": block.get("name"),
                        "input_hash": _sha256(f"{block.get('name')}:{args_json}"),
                        "input_bytes": len(json.dumps(args, default=str)),
                        "provider_call_id": block.get("id"),
                    }
                )
        if out:
            return out
    return _extract_tool_calls(response_body)


def extract_record_from_portkey(entry: dict) -> dict:
    """Build the same `rec` shape ys/collector.py's `extract_record`
    produces for the LiteLLM path, from one row of a Portkey Logs Export.
    Pure -- no I/O -- so it's testable without a live export."""
    request_body = _parse_json_field(entry.get("request"))
    response_body = _parse_json_field(entry.get("response"))

    messages, system_text = _messages_and_system(request_body)
    tools = request_body.get("tools") or []
    model = entry.get("ai_model") or request_body.get("model")
    msg_hashes = _msg_hashes(messages)
    system_prompt_hash = _sha256(system_text) if system_text else None
    status_code = entry.get("status_code") or 200
    tools_json = json.dumps(tools, sort_keys=True, default=str) if tools else None

    return {
        "ts": entry.get("created_at") or _now(),
        "provider": entry.get("ai_provider"),
        "model": model,
        "stream": bool(request_body.get("stream")),
        "input_tokens": entry.get("request_tokens"),
        # Portkey's export fields (per its docs) don't break cache tokens
        # out from input_tokens the way LiteLLM's standard_logging_object
        # does -- left at 0 rather than guessed, same as any other field
        # this prototype can't source; billable_tokens/cache_read_ratio for
        # Portkey-backed runs will read low relative to a LiteLLM-backed
        # run of the same model until this is confirmed and fixed.
        "cache_creation": 0,
        "cache_read": 0,
        "output_tokens": entry.get("response_tokens"),
        "response_cost": entry.get("cost") or 0.0,
        "latency_ms": entry.get("response_time"),
        # Portkey's export has no separate first-token timestamp field in
        # what's documented -- unlike ys/collector.py's `_ttft_ms`, which
        # falls back to full latency for a non-streaming request, there's
        # nothing to fall back to here at all.
        "ttft_ms": None,
        "status_code": status_code,
        "error": None if status_code == 200 else json.dumps(response_body, default=str)[:2000],
        "msg_count": len(messages),
        "msg_hashes": msg_hashes,
        "system_tokens": _safe_token_count(model, system_text),
        "tools_tokens": _safe_token_count(model, tools_json) if tools_json else 0,
        "tool_calls": _extract_tool_calls_from_response(response_body),
        "tool_results": _extract_tool_results(messages),
        "user_agent": None,
        "toolset_hash": _sha256(tools_json) if tools_json else None,
        "tool_count": len(tools) if isinstance(tools, list) else 0,
        "system_prompt_hash": system_prompt_hash,
    }


def _run_window(run_id: str) -> tuple:
    with db.cursor() as cur:
        row = cur.execute("SELECT started_at, ended_at FROM runs WHERE id = ?", (run_id,)).fetchone()
    if row is None:
        raise PortkeyCollectorError(f"no such run '{run_id}'")
    return row["started_at"], row["ended_at"] or _now()


def ingest(run_id: str) -> int:
    """Pull this run's requests back from Portkey's Logs Export API and
    write them into the same requests/tool_calls tables ys/collector.py's
    LiteLLM path writes to. Returns the number of records written.

    Idempotent/re-runnable: any rows already recorded for `run_id` (e.g.
    from an earlier `ys proxy pull-logs` attempt) are deleted first, so a
    retry replaces rather than duplicates -- important since Portkey's log
    pipeline may not have finished indexing a request by the time `ys end`
    makes its own best-effort attempt (see ys/cli.py).
    """
    started_at, ended_at = _run_window(run_id)

    export_id = _create_export(started_at, ended_at, run_id)
    _start_export(export_id)
    _poll_until_complete(export_id)
    records = _download(export_id)

    def _clear(cur):
        cur.execute("DELETE FROM tool_calls WHERE run_id = ?", (run_id,))
        cur.execute("DELETE FROM requests WHERE run_id = ?", (run_id,))

    with db.cursor() as cur:
        _clear(cur)

    written = 0
    for entry in records:
        metadata = _parse_json_field(entry.get("metadata"))
        # Client-side narrowing to this run, in case `filters.metadata` in
        # `_create_export` isn't actually honored by the export API (see
        # this module's docstring) -- a record with no ys_run_id tag at all
        # predates ys/portkey_app.py's tagging (or came from other traffic
        # on the same account) and is skipped rather than misattributed.
        if metadata.get("ys_run_id") != run_id:
            continue
        rec = extract_record_from_portkey(entry)
        db.call_with_retry(_write, run_id, rec)
        written += 1
    return written
