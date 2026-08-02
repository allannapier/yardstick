"""The local reverse-proxy process for the Portkey backend
(ys/portkey_backend.py) -- run under uvicorn as its own subprocess, the same
way ys/proxy.py runs `litellm` as a subprocess. Point a harness at
`http://localhost:<port>` exactly as you would for the LiteLLM backend
(ys/harness.py doesn't need to know or care which backend it's talking to).

Speaks whatever wire protocol the harness sends -- Claude Code's native
Anthropic Messages format at `/v1/messages`, or OpenAI Chat Completions at
`/v1/chat/completions` (Codex CLI/Aider/opencode) -- and forwards it
byte-for-byte to the same path on Portkey Cloud, so this app never needs to
know which shape it's carrying (mirrors ys/harness.py's own "provider isn't
a per-agent branch" reasoning). Two things are rewritten in transit:

  1. Auth: the harness's local key (LITELLM_MASTER_KEY, checked by
     `_check_local_auth`) is swapped for real Portkey auth
     (PORTKEY_API_KEY + PORTKEY_VIRTUAL_KEY, see ys/portkey_backend.py's
     module docstring for what each is).
  2. Tagging: an x-portkey-metadata header carrying the active ys run id is
     attached, so ys/portkey_collector.py can find this request again in
     Portkey's logs after the run ends -- Claude Code can't be told to send
     arbitrary headers itself (same limitation ys/collector.py's
     `_resolve_run_id` works around for the LiteLLM path), so this is
     resolved locally from ACTIVE_RUN_PATH instead, same fallback source.

Deliberately does not parse or persist request/response bodies -- seeing
the traffic pass through here would be enough to log it the way
ys/collector.py does for LiteLLM, but doing so would just be a second,
divergent measurement of the same requests Portkey's own Logs Export API
already records with its own cost/token accounting. See
ys/portkey_collector.py's module docstring for the actual recording path.
"""
import json
import os
from typing import Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ys import paths

app = FastAPI()

PORTKEY_BASE_URL = "https://api.portkey.ai"

# Hop-by-hop headers that must never be copied straight through from
# Portkey's response -- httpx has already undone any transfer/content
# encoding by the time we see the body, and letting a stale
# content-length/content-encoding value ride along on a response we're
# about to re-chunk ourselves produces a response real HTTP clients
# (correctly) refuse to parse.
_STRIP_RESPONSE_HEADERS = {"content-encoding", "transfer-encoding", "content-length", "connection"}


def _check_local_auth(headers, expected_key: Optional[str]) -> bool:
    """True if `headers` (case-insensitive dict-like: starlette's
    `Request.headers`, or a plain dict in tests) carries the local proxy's
    own shared secret, either as `Authorization: Bearer <key>` (the shape
    Codex CLI/Aider/opencode send) or `x-api-key: <key>` (Anthropic's own
    convention, what Claude Code sends). No key configured at all
    (`expected_key` falsy) is treated as "auth disabled", same as LiteLLM's
    proxy without a master_key -- not this prototype's problem to fix."""
    if not expected_key:
        return True
    auth = headers.get("authorization", "")
    if auth.removeprefix("Bearer ").strip() == expected_key:
        return True
    return headers.get("x-api-key", "") == expected_key


def _active_run_id(headers) -> Optional[str]:
    """The x-ys-run header if the harness sent one (most don't -- see
    ys/harness.py's module docstring), else whatever `ys start` last wrote
    to ACTIVE_RUN_PATH. Unlike ys/collector.py's `_resolve_run_id`, this has
    no "unattributed"/drain-window fallback to reach for: it isn't writing
    to the database itself, just tagging outgoing metadata, so a request
    with no discoverable run id is simply tagged with none rather than a
    synthetic run row that would never mean anything on the Portkey side."""
    run_id = headers.get("x-ys-run")
    if run_id:
        return run_id
    if os.path.exists(paths.ACTIVE_RUN_PATH):
        try:
            with open(paths.ACTIVE_RUN_PATH) as f:
                return json.load(f).get("run_id")
        except (OSError, ValueError, KeyError):
            return None
    return None


def _build_upstream_headers(incoming_headers, run_id: Optional[str]) -> dict:
    """Pure function (no I/O, no env reads beyond what's passed in) so the
    header-rewrite logic is unit-testable without a running server -- reads
    PORTKEY_API_KEY/PORTKEY_VIRTUAL_KEY from the environment directly since
    those are real secrets that should never round-trip through a test
    fixture's argument list by convention elsewhere in this codebase (see
    ys/proxy.py's `os.environ/<VAR>` indirection for the same instinct)."""
    headers = {
        "content-type": incoming_headers.get("content-type", "application/json"),
        "authorization": f"Bearer {os.environ.get('PORTKEY_API_KEY', '')}",
    }
    virtual_key = os.environ.get("PORTKEY_VIRTUAL_KEY")
    if virtual_key:
        headers["x-portkey-virtual-key"] = virtual_key
    if run_id:
        headers["x-portkey-metadata"] = json.dumps({"ys_run_id": run_id})
    accept = incoming_headers.get("accept")
    if accept:
        headers["accept"] = accept
    anthropic_version = incoming_headers.get("anthropic-version")
    if anthropic_version:
        headers["anthropic-version"] = anthropic_version
    return headers


@app.get("/health/readiness")
async def readiness():
    return {"status": "ok"}


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def proxy(path: str, request: Request):
    if not _check_local_auth(request.headers, os.environ.get("LITELLM_MASTER_KEY")):
        return JSONResponse({"error": {"message": "invalid local proxy key"}}, status_code=401)

    body = await request.body()
    run_id = _active_run_id(request.headers)
    upstream_headers = _build_upstream_headers(request.headers, run_id)
    upstream_url = f"{PORTKEY_BASE_URL}/{path}"

    client = httpx.AsyncClient(timeout=600.0)
    try:
        upstream_req = client.build_request(
            request.method,
            upstream_url,
            params=request.query_params,
            headers=upstream_headers,
            content=body,
        )
        upstream_resp = await client.send(upstream_req, stream=True)
    except httpx.HTTPError as e:
        await client.aclose()
        return JSONResponse(
            {"error": {"message": f"could not reach Portkey Cloud: {e}"}}, status_code=502
        )

    async def _stream():
        try:
            async for chunk in upstream_resp.aiter_raw():
                yield chunk
        finally:
            await upstream_resp.aclose()
            await client.aclose()

    response_headers = {
        k: v for k, v in upstream_resp.headers.items() if k.lower() not in _STRIP_RESPONSE_HEADERS
    }
    return StreamingResponse(
        _stream(), status_code=upstream_resp.status_code, headers=response_headers
    )
