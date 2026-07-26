import json
import os
import time
from typing import Optional

from ys import db, paths

# Duplicated from ys/runs.py's TS_FORMAT/now()/_parse_ts rather than
# imported: ys/runs.py imports this module (for set_active/get_active/
# clear_active), so importing runs.py back from here would be a cycle.
TS_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

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


def _now() -> str:
    return time.strftime(TS_FORMAT, time.gmtime())


def _parse_ts(ts: str) -> float:
    return time.mktime(time.strptime(ts, TS_FORMAT)) - time.timezone


def _elapsed_s(started_at: str, ended_at: str) -> float:
    return _parse_ts(ended_at) - _parse_ts(started_at)


def _abandon_displaced_run(existing: dict):
    """Finding 13: `--force` overwrote the active slot without closing the
    run it displaced, leaving `ended_at`/`task_success`/`wall_clock_s` NULL
    on that row forever -- and `aggregate_run_metrics` counted it toward
    `n_runs` regardless, so every forced start permanently depressed the
    arm's success rate. Close the displaced run's row out here instead:
    record when it stopped and how long it ran, and flag it `abandoned` so
    aggregation can exclude it and report it separately. `task_success` is
    deliberately left NULL -- the task was never actually scored, which is
    a different thing from failing it.

    Guarded by `ended_at IS NULL` so this is a no-op if the row was somehow
    already closed (e.g. a `ys end` that raced the force and won) --
    finding 11 owns exactly when the active-run file itself gets cleared/
    drained; this only ever touches the run row of whatever was active at
    the moment `--force` is used.
    """
    ended_at = _now()
    wall_clock_s = _elapsed_s(existing["started_at"], ended_at)

    def _write():
        with db.cursor() as cur:
            cur.execute(
                "UPDATE runs SET ended_at=?, wall_clock_s=?, abandoned=1 "
                "WHERE id=? AND ended_at IS NULL",
                (ended_at, wall_clock_s, existing["run_id"]),
            )

    db.call_with_retry(_write)


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
    if existing and force:
        # finding 13: don't just overwrite the slot -- close out the run
        # being displaced so it doesn't dangle forever as an unscored row
        # that still counts toward the arm's n_runs.
        _abandon_displaced_run(existing)
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
