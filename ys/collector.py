"""Production LiteLLM CustomLogger for yardstick.

Field paths here are verified against a real proxied Claude Code-shaped
request (see tools/provenance/dumps/), not guessed from LiteLLM's docs. See
tools/provenance/ for the probe that produced the ground truth.
"""
import asyncio
import hashlib
import json
import time
import traceback
import uuid
from typing import Optional

import yaml
from litellm.integrations.custom_logger import CustomLogger

from ys import db, dropped, paths, state
from ys.experiment import resolve_model_key

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
    against a live probe (see tools/provenance/): a shrink-and-diverge-after-index-0
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


def _shared_prefix_len(prev: list, cur: list) -> int:
    k = 0
    for a, b in zip(prev, cur):
        if a == b:
            k += 1
        else:
            break
    return k


# Finding 25: `_classify_transition` calls *any* shorter history a
# "compaction" as long as it isn't a byte-for-byte continuation -- that's
# deliberate (a real compaction rewrites message[0], which is the same
# thing an unrelated conversation does, so the classifier can't use
# position-0 equality to tell them apart either; see its docstring). That
# leaves `_resolve_thread`'s system_prompt_hash match as the *only* thing
# standing between a same-system-prompt subagent/background call and being
# absorbed into the main thread as a fabricated compaction event -- exactly
# what finding 4 exists to prevent.
#
# A second, coarser signal: a genuine compaction is lossy but not
# annihilative -- it summarizes old turns while keeping the conversation
# going, so a meaningful fraction of the message count survives. An
# unrelated conversation restarts from a small, roughly constant handful of
# messages (a subagent's task instructions, a title-gen prompt) regardless
# of how long the thread it's mistaken for has grown -- the longer the real
# conversation gets, the more implausible it is that an unrelated exchange
# would happen to retain a third of its message count. Requiring at least
# that fraction to survive is a low bar for a real compaction to clear and,
# per finding 25's own caveat, not a precise boundary (hashes alone can't
# fully disambiguate a same-system-prompt impostor from a real compaction)
# -- just a deliberately conservative floor against the worst case: a tiny
# unrelated exchange landing right after a long thread and being read as
# "compaction" on message-count alone.
_MIN_COMPACTION_RATIO = 1 / 3


def _plausible_compaction(prev: list, cur: list) -> bool:
    if not prev:
        return True
    return len(cur) >= len(prev) * _MIN_COMPACTION_RATIO


def _resolve_thread(cur, run_id: str, system_prompt_hash, msg_hashes: list) -> tuple:
    """Assigns a request to a thread and classifies its transition within
    that thread in one pass, so transition classification and conversation
    metrics aren't computed across unrelated interleaved traffic (finding
    4): Claude Code interleaves the main conversation with background
    (title-generation) requests and Task-subagent conversations in the same
    run, each its own unrelated, much shorter history.

    A request joins the thread whose most recent request (a) shares this
    request's system prompt and (b) is a plausible parent of this request's
    history per _classify_transition -- continuation, compaction, or
    branch, anything but "reset". Chain-following like this, rather than
    hashing a fixed anchor message (an earlier version of this function),
    is what lets a genuine harness-side compaction -- which rewrites/
    summarizes the early history, including whatever message a fixed
    anchor would have pinned on -- still resolve to the same thread instead
    of registering as a new one. Ties go to the larger shared-message-
    prefix match. No candidate (or an empty run) starts a new thread.

    Assumption (finding 25): matching system_prompt_hash is treated as
    strong evidence of "same conversation", and a "compaction" transition
    is accepted as a plausible parent on top of that. That combination is
    only safe because, in every harness this rig currently drives, a
    subagent or background call carries a different system prompt from the
    main conversation -- a property of those harnesses at their current
    versions, not a guarantee. If a harness ever reused the main system
    prompt for background/subagent traffic, `_plausible_compaction` above
    is the second signal that keeps that traffic from being absorbed into
    the main thread as a fabricated compaction event: a candidate whose
    message count collapsed far more than a real compaction plausibly would
    is rejected here and falls through to starting its own thread instead.
    """
    rows = cur.execute(
        """
        SELECT r.thread_key, r.system_prompt_hash, r.msg_hashes_json
        FROM requests r
        JOIN (
            SELECT thread_key, MAX(seq) AS max_seq FROM requests
            WHERE run_id = ? GROUP BY thread_key
        ) latest ON r.thread_key IS latest.thread_key AND r.seq = latest.max_seq
        WHERE r.run_id = ?
        """,
        (run_id, run_id),
    ).fetchall()

    best = None  # (overlap, thread_key, transition)
    for row in rows:
        if row["system_prompt_hash"] != system_prompt_hash:
            continue
        prev_hashes = json.loads(row["msg_hashes_json"]) if row["msg_hashes_json"] else []
        transition = _classify_transition(prev_hashes, msg_hashes)
        if transition is None or transition == "reset":
            continue
        if transition == "compaction" and not _plausible_compaction(prev_hashes, msg_hashes):
            continue
        overlap = _shared_prefix_len(prev_hashes, msg_hashes)
        if best is None or overlap > best[0]:
            best = (overlap, row["thread_key"], transition)

    if best is not None:
        return best[1], best[2]

    return _sha256(f"{system_prompt_hash or ''}|{uuid.uuid4().hex}"), None


def _safe_token_count(model: str, text: str) -> int:
    if not text:
        return 0
    try:
        import litellm

        return litellm.token_counter(model=model or "gpt-4", text=text)
    except Exception:
        return max(1, len(text) // 4)


def _resolve_run_id(kwargs: dict) -> str:
    """Finding 11: a response that lands after `ys end` has no `x-ys-run`
    header (harnesses like Claude Code can't set arbitrary headers) and no
    active.json to fall back to either, since `finish_run` clears it right
    away -- without the drain-window fallback below, that would land in
    `unattributed` even though it's plainly the tail of the run that just
    finished (often the final and largest turn). See `state.mark_ended`/
    `state.get_draining_run`."""
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

    try:
        draining = state.get_draining_run()
        if draining:
            return draining["run_id"]
    except Exception:
        pass

    return "unattributed"


def _next_seq(cur, run_id: str) -> int:
    """Must run inside a transaction that already holds the write lock (see
    `_write`'s `BEGIN IMMEDIATE`) -- read-then-insert with no lock held is
    exactly the race finding 7 in IMPROVEMENTS.md describes: two concurrent
    requests (parallel tool use, a subagent) read the same MAX(seq) and both
    insert it, corrupting the seq-ordered transition chain with no
    constraint to catch it."""
    row = cur.execute(
        "SELECT COALESCE(MAX(seq), 0) AS m FROM requests WHERE run_id = ?", (run_id,)
    ).fetchone()
    return (row["m"] or 0) + 1


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


def _ttft_ms(slo: dict) -> Optional[float]:
    """Time to first token, in ms, from LiteLLM's own
    `standard_logging_object` timestamps -- `startTime`/`completionStartTime`
    (both epoch-second floats; camelCase is LiteLLM's own naming, not this
    codebase's -- see `StandardLoggingPayload` in
    litellm/types/utils.py). Finding 15-18: this was a schema column
    hardcoded to None; verified against the installed litellm package
    (not guessed) that both fields are actually there to use:

    - Streaming: LiteLLM's streaming handler
      (litellm_core_utils/streaming_handler.py) stamps
      `completion_start_time` on the first chunk it receives, so
      `completionStartTime - startTime` is a real time-to-first-token.
    - Non-streaming: LiteLLM never sets `completion_start_time` at all;
      `get_standard_logging_object_payload` (litellm_core_utils/
      litellm_logging.py) defaults it to `end_time` when building the
      payload. So for a non-streaming request this collapses to the full
      round-trip latency (`completionStartTime == endTime`) -- which isn't
      a bug in this function, it's the correct answer: a non-streaming
      response arrives as a single event, so "time to first token" and
      "time to last token" are the same moment.

    Returns None if either timestamp is missing (e.g. a failed request with
    no standard_logging_object at all) rather than fabricating a number.
    """
    start = slo.get("startTime")
    completion_start = slo.get("completionStartTime")
    if start is None or completion_start is None:
        return None
    try:
        return (completion_start - start) * 1000
    except TypeError:
        return None


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
    system_prompt_hash = _sha256(system_text) if system_text else None

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
        "ttft_ms": _ttft_ms(slo),
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
        "system_prompt_hash": system_prompt_hash,
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


def _pricing_table_for_run(cur, run_id: str) -> dict:
    """The `pricing:` block (ys/experiment.py) of the experiment YAML behind
    this run, keyed by `factors.model` value -- used to price a request
    LiteLLM's own cost map can't (finding 9). Read from
    `experiments.config_yaml`, which is stored verbatim at `ys start`
    (ys/runs.py's begin_run) and is the same field `finish_run` already
    reads for `task.success_check` -- reusing it here means the collector
    (running inside the separate proxy process) needs no new plumbing to
    the CLI/experiment-loading side to find an experiment's declared
    prices. Best-effort: a missing run/experiment row, unparseable YAML, or
    no `pricing:` block at all all just mean "nothing declared", not an
    error -- the caller falls back to reporting the cost as unknown."""
    row = cur.execute(
        "SELECT e.config_yaml FROM runs r JOIN experiments e ON e.id = r.experiment_id "
        "WHERE r.id = ?",
        (run_id,),
    ).fetchone()
    if not row or not row["config_yaml"]:
        return {}
    try:
        cfg = yaml.safe_load(row["config_yaml"]) or {}
    except yaml.YAMLError:
        return {}
    pricing = cfg.get("pricing") if isinstance(cfg, dict) else None
    return pricing if isinstance(pricing, dict) else {}


def _declared_cost(model, input_tokens, cache_creation, cache_read, output_tokens, pricing_table: dict):
    """Compute cost from tokens using a declared `pricing:` entry (USD per
    million tokens), or return None if no entry matches this model. Pure
    function -- `pricing_table` is passed in rather than loaded here, so
    this is testable without a database."""
    key = resolve_model_key(model, pricing_table)
    if key is None:
        return None
    price = pricing_table.get(key) or {}
    if not isinstance(price, dict):
        return None

    def per_tok(field):
        rate = price.get(field)
        return (rate or 0) / 1_000_000

    return (
        (input_tokens or 0) * per_tok("input_per_mtok")
        + (output_tokens or 0) * per_tok("output_per_mtok")
        + (cache_creation or 0) * per_tok("cache_write_per_mtok")
        + (cache_read or 0) * per_tok("cache_read_per_mtok")
    )


def _resolve_cost(rec: dict, pricing_table: dict) -> tuple:
    """Decide this request's cost and how it was obtained (finding 9).
    LiteLLM's own cost map silently returns 0.0 for any model id it has no
    price for -- verified true for `claude-sonnet-5` as configured in
    experiments/interactive-sonnet.yaml, and true in general for any
    custom/self-hosted deployment name. A confident $0.0000 in `ys
    compare`/`ys report` is the worst available failure mode for a tool
    whose headline output is a cost comparison, so:

    - LiteLLM's own number wins whenever it's nonzero ("litellm").
    - If it's zero and this request actually spent tokens, an experiment's
      declared `pricing:` block is used instead if one matches this model
      ("declared").
    - If neither can price it, cost stays 0.0 but is tagged "unknown" so
      the zero can be flagged instead of silently trusted.
    - Zero cost with zero tokens (e.g. a failed request with no usage) is
      not the failure mode this guards against, so it's left as "litellm"
      rather than flagged.
    """
    cost = rec.get("response_cost") or 0.0
    if cost:
        return cost, "litellm"

    total_tokens = (
        (rec.get("input_tokens") or 0)
        + (rec.get("output_tokens") or 0)
        + (rec.get("cache_creation") or 0)
        + (rec.get("cache_read") or 0)
    )
    if not total_tokens:
        return cost, "litellm"

    declared = _declared_cost(
        rec.get("model"),
        rec.get("input_tokens"),
        rec.get("cache_creation"),
        rec.get("cache_read"),
        rec.get("output_tokens"),
        pricing_table,
    )
    if declared is not None:
        return declared, "declared"
    return 0.0, "unknown"


def _write(run_id: str, rec: dict):
    with db.cursor() as cur:
        # BEGIN IMMEDIATE grabs the write lock up front instead of at the
        # first DML statement, so the seq read in `_next_seq` and the insert
        # that uses it are atomic with respect to every other writer on this
        # database file -- a concurrent `_write` blocks (honoring
        # `busy_timeout`) until this transaction commits, rather than racing
        # it. See finding 7 in IMPROVEMENTS.md.
        cur.execute("BEGIN IMMEDIATE")
        if run_id != "unattributed" and not _run_row_exists(cur, run_id):
            # A run_id was claimed (header or stale active.json) that doesn't
            # exist in the DB -- e.g. ys start crashed, or a request arrived
            # after ys end cleared the state file. Never drop the data.
            run_id = "unattributed"
        _ensure_run_exists(cur, run_id)
        seq = _next_seq(cur, run_id)
        thread_key, transition = _resolve_thread(
            cur, run_id, rec["system_prompt_hash"], rec["msg_hashes"]
        )
        # See finding 9 in IMPROVEMENTS.md: LiteLLM's own cost map silently
        # returns 0.0 for a model id it can't price, which would otherwise
        # read as a confident (and wrong) $0 in every downstream total.
        response_cost, cost_source = _resolve_cost(rec, _pricing_table_for_run(cur, run_id))

        cur.execute(
            """INSERT INTO requests
               (run_id, seq, ts, provider, model, stream, input_tokens, cache_creation,
                cache_read, output_tokens, response_cost, latency_ms, ttft_ms, status_code,
                error, msg_count, msg_hashes_json, system_tokens, tools_tokens, transition,
                thread_key, toolset_hash, system_prompt_hash, cost_source)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
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
                response_cost,
                rec["latency_ms"],
                rec["ttft_ms"],
                rec["status_code"],
                rec["error"],
                rec["msg_count"],
                json.dumps(rec["msg_hashes"]),
                rec["system_tokens"],
                rec["tools_tokens"],
                transition,
                thread_key,
                rec["toolset_hash"],
                rec["system_prompt_hash"],
                cost_source,
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
        run_id = "unknown"
        try:
            run_id = _resolve_run_id(kwargs)
            rec = extract_record(kwargs, response_obj, start_time, end_time)
            # `_write` already opens its own `db.cursor()` transaction per
            # call, so `db.call_with_retry` can just call it again on a
            # retryable failure -- see finding 28 in IMPROVEMENTS.md, and
            # `db.call_with_retry`'s docstring for why this is now the one
            # retry policy shared with the CLI/dashboard write paths
            # (ys/runs.py) instead of collector-only inline retry logic.
            await asyncio.get_running_loop().run_in_executor(
                None, db.call_with_retry, _write, run_id, rec
            )
        except Exception as e:
            # A request whose write never lands has no other record of its
            # existence -- count it so a lossy run is visible instead of
            # quietly short (see `ys status`).
            dropped.record(run_id, repr(e))
            traceback.print_exc()


yardstick_logger = YardstickLogger()
