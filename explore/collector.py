"""Phase 0 probe: dump the real kwargs/response shape LiteLLM's CustomLogger
callback receives, so the yardstick extractor is built against verified
field paths instead of guessed ones.
"""
import json
import os
from datetime import datetime, timezone

from litellm.integrations.custom_logger import CustomLogger

DUMP_DIR = os.path.join(os.path.dirname(__file__), "dumps")
os.makedirs(DUMP_DIR, exist_ok=True)

# kwargs from litellm's logging callback carries the raw provider credential
# (kwargs["api_key"], and again nested in litellm_params) plus auth headers
# forwarded from the client. None of that may ever reach disk.
_SECRET_KEYS = {"api_key", "authorization", "x-api-key", "api-key"}


def _redact_secrets(obj, key=None):
    if key is not None and str(key).lower() in _SECRET_KEYS:
        return "<redacted>"
    if isinstance(obj, dict):
        return {k: _redact_secrets(v, key=k) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_redact_secrets(v) for v in obj]
    return obj


def _safe(obj):
    """Best-effort JSON-safe conversion for arbitrary litellm objects."""
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        pass
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass
    if isinstance(obj, dict):
        return {k: _safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe(v) for v in obj]
    return repr(obj)


class YardstickLogger(CustomLogger):
    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        await self._dump("success", kwargs, response_obj, start_time, end_time)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        await self._dump("failure", kwargs, response_obj, start_time, end_time)

    async def _dump(self, kind, kwargs, response_obj, start_time, end_time):
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        path = os.path.join(DUMP_DIR, f"{ts}_{kind}.json")

        record = {
            "kind": kind,
            "kwargs_top_level_keys": sorted(list(kwargs.keys())),
            "kwargs": _redact_secrets({k: _safe(v) for k, v in kwargs.items()}),
            "response_obj": _safe(response_obj),
            "response_obj_type": str(type(response_obj)),
            "start_time": str(start_time),
            "end_time": str(end_time),
        }
        with open(path, "w") as f:
            json.dump(record, f, indent=2, default=str)
        print(f"[yardstick-probe] wrote {path}")


yardstick_logger = YardstickLogger()
