"""Tracks requests the collector could not persist -- e.g. a write that lost
the race against `db.connect()`'s busy_timeout under sustained concurrent
access. Without this, a lossy run looks identical to a clean one: the only
prior failure handling was `traceback.print_exc()` into the proxy log, which
nobody watches mid-run. See finding 6 in IMPROVEMENTS.md."""
import json
import os
from datetime import datetime, timezone

from ys import paths


def record(run_id: str, error: str):
    paths.ensure_home()
    entry = {
        "run_id": run_id,
        "ts": datetime.now(timezone.utc).isoformat(),
        "error": error,
    }
    with open(paths.DROPPED_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def count() -> int:
    if not os.path.exists(paths.DROPPED_LOG_PATH):
        return 0
    with open(paths.DROPPED_LOG_PATH, encoding="utf-8") as f:
        return sum(1 for _ in f)
