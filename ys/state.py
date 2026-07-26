import json
import os
import time
from typing import Optional

from ys import db, paths

# Duplicated from ys/runs.py's TS_FORMAT/now()/_parse_ts rather than
# imported: ys/runs.py imports this module (for set_active/get_active/
# clear_active), so importing runs.py back from here would be a cycle.
TS_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


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
