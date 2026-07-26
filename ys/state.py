import json
import os
import time
from typing import Optional

from ys import paths

# Finding 11: `finish_run` clears active.json unconditionally so `ys status`
# and the next `ys start` see the slot free right away -- but a harness that
# can't set `x-ys-run` has no attribution signal left once that file is gone,
# and a response that lands after `ys end` (the tail of the run, often the
# largest turn) would fall into `unattributed`. DRAIN_WINDOW_S is how long
# after a run ends a request with no other signal is still credited to it
# instead. `ys end` is a synchronous CLI command the user is waiting on, so
# the fix can't be "sleep until stragglers land" -- see `mark_ended`/
# `get_draining_run` below for the timestamp-window approach used instead.
DRAIN_WINDOW_S = 60


class RunAlreadyActive(Exception):
    pass


def get_active() -> Optional[dict]:
    if not os.path.exists(paths.ACTIVE_RUN_PATH):
        return None
    with open(paths.ACTIVE_RUN_PATH) as f:
        return json.load(f)


def set_active(run_id: str, experiment: str, arm: str, started_at: str, force: bool = False):
    paths.ensure_home()
    existing = get_active()
    if existing and not force:
        raise RunAlreadyActive(
            f"run {existing['run_id']} (exp={existing['experiment']}, arm={existing['arm']}) "
            "is already active. Use --force to override, or `ys end` it first."
        )
    with open(paths.ACTIVE_RUN_PATH, "w") as f:
        json.dump(
            {
                "run_id": run_id,
                "experiment": experiment,
                "arm": arm,
                "started_at": started_at,
            },
            f,
            indent=2,
        )


def clear_active():
    if os.path.exists(paths.ACTIVE_RUN_PATH):
        os.remove(paths.ACTIVE_RUN_PATH)


def mark_ended(run_id: str, experiment: str, arm: str, ended_at: str):
    """Record that `run_id` just finished, for `get_draining_run`'s drain
    window (finding 11). Deliberately a *separate* file from active.json,
    not an `ended_at` field left on it: active.json's mere presence is what
    `set_active`/`ys status`/a following `ys start` treat as "a run is in
    progress", and a straggling request landing seconds after `ys end`
    closed the run out cleanly must not resurrect that. `finish_run` calls
    this before `clear_active()` so a request racing the two writes still
    has a signal to fall back to either way."""
    paths.ensure_home()
    with open(paths.LAST_ENDED_RUN_PATH, "w") as f:
        json.dump(
            {
                "run_id": run_id,
                "experiment": experiment,
                "arm": arm,
                "ended_at": ended_at,
                # Absolute deadline, computed once at write time -- so
                # get_draining_run only needs a cheap `time.time()` compare,
                # not to re-parse/re-derive it (and not to re-import
                # ys/runs.py's TS_FORMAT parsing, which would create a
                # state<->runs import cycle: ys/runs.py already imports
                # ys/state.py).
                "drain_until": time.time() + DRAIN_WINDOW_S,
            },
            f,
            indent=2,
        )


def get_draining_run() -> Optional[dict]:
    """The most recently `ys end`-ed run, if it ended within DRAIN_WINDOW_S
    of now -- otherwise None, so a request arriving well after the run
    closed still falls through to `unattributed` (finding 12 makes that
    visible) rather than being silently misattributed to a stale run.
    Read from the proxy process via `ys/collector.py`'s `_resolve_run_id`,
    which is why this is file-based cross-process state like active.json,
    not an in-memory value the CLI process could hand off directly."""
    if not os.path.exists(paths.LAST_ENDED_RUN_PATH):
        return None
    try:
        with open(paths.LAST_ENDED_RUN_PATH) as f:
            record = json.load(f)
        if time.time() > record["drain_until"]:
            return None
    except (json.JSONDecodeError, KeyError, OSError):
        return None
    return record
