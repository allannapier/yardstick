"""Run lifecycle business logic, shared by the CLI (ys/cli.py) and the web
dashboard (ys/web/app.py) so there is exactly one implementation of the
active-slot-before-db-write ordering, config_yaml refresh, and orphan-row
avoidance that ys/cli.py originally had inline. Both callers catch the
exceptions here and format them for their own presentation layer (Rich
console vs JSON/HTML).
"""
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import yaml

from ys import db, state
from ys.experiment import Experiment

TS_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def _with_cursor(fn):
    """Run `fn(cur)` inside a fresh `db.cursor()` transaction. Exists so a
    write can be handed to `db.call_with_retry` -- which re-invokes its
    callable from scratch on a retryable failure -- without each call site
    re-deriving the "open a fresh cursor per attempt" boilerplate. See
    finding 28 in IMPROVEMENTS.md."""
    with db.cursor() as cur:
        return fn(cur)


def now() -> str:
    return time.strftime(TS_FORMAT, time.gmtime())


def _parse_ts(ts: str) -> float:
    return time.mktime(time.strptime(ts, TS_FORMAT)) - time.timezone


def arm_row_id(experiment: str, arm_id: str) -> str:
    return f"{experiment}::{arm_id}"


class ArmNotFound(Exception):
    def __init__(self, arm_id: str, experiment_name: str, valid_arms: list):
        self.arm_id = arm_id
        self.experiment_name = experiment_name
        self.valid_arms = valid_arms
        super().__init__(
            f"no such arm '{arm_id}' in experiment '{experiment_name}'. "
            f"Valid arms: {', '.join(valid_arms)}"
        )


class NoActiveRun(Exception):
    pass


class ActiveRunMissingDbRow(Exception):
    def __init__(self, run_id: str):
        self.run_id = run_id
        super().__init__(f"run {run_id} is active but has no database row")


class NoSuccessCheck(Exception):
    pass


class RunNotFound(Exception):
    def __init__(self, run_id: str):
        self.run_id = run_id
        super().__init__(f"no such run '{run_id}'")


class CannotDeleteActiveRun(Exception):
    def __init__(self, run_id: str):
        self.run_id = run_id
        super().__init__(
            f"run {run_id} is the active run -- end it with `ys end` before deleting it"
        )


@dataclass
class BeginResult:
    run_id: str
    experiment_name: str
    arm_id: str
    repeat_idx: int
    started_at: str


def begin_run(experiment: Experiment, config_yaml: str, arm_id: str, force: bool = False) -> BeginResult:
    """`config_yaml` is the raw YAML text backing `experiment` -- stored
    verbatim so `finish_run` re-reads exactly what was run, not a
    reconstruction from the parsed model (which wouldn't round-trip
    comments/formatting and has no reason to diverge from the source file)."""
    try:
        arm_obj = experiment.get_arm(arm_id)
    except KeyError:
        raise ArmNotFound(arm_id, experiment.experiment, [a.id for a in experiment.arms])

    run_id = str(uuid.uuid4())[:8]
    started_at = now()
    exp_id = experiment.experiment
    a_id = arm_row_id(exp_id, arm_id)

    # Claim the active slot before writing any rows, so a refused start does
    # not leave an orphan run row inflating the arm's repeat count.
    state.set_active(run_id, exp_id, arm_id, started_at, force=force)  # may raise RunAlreadyActive

    def _insert(cur):
        cur.execute(
            "INSERT OR IGNORE INTO experiments (id, name, question, task_json, config_yaml, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                exp_id,
                exp_id,
                experiment.question,
                db.dumps(experiment.task.model_dump()),
                config_yaml,
                started_at,
            ),
        )
        # The YAML is the source of truth; refresh it so `finish_run` re-reads
        # the success_check being run today, not a stale first version.
        cur.execute(
            "UPDATE experiments SET config_yaml = ?, task_json = ? WHERE id = ?",
            (config_yaml, db.dumps(experiment.task.model_dump()), exp_id),
        )
        cur.execute(
            "INSERT OR IGNORE INTO arms (id, experiment_id, label, factors_json, is_baseline) "
            "VALUES (?, ?, ?, ?, ?)",
            (a_id, exp_id, arm_id, db.dumps(arm_obj.factors), int(arm_obj.baseline)),
        )
        repeat_idx = (
            cur.execute(
                "SELECT COUNT(*) AS c FROM runs WHERE arm_id = ?", (a_id,)
            ).fetchone()["c"]
            + 1
        )
        cur.execute(
            "INSERT INTO runs (id, experiment_id, arm_id, repeat_idx, started_at, notes) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, exp_id, a_id, repeat_idx, started_at, arm_obj.notes),
        )
        return repeat_idx

    try:
        # `ys start` writes from the CLI/dashboard process while the proxy
        # may be writing requests for a still-active previous run at the
        # same moment -- retry a lock that outlasts busy_timeout instead of
        # surfacing "database is locked" as a raw traceback (finding 28).
        repeat_idx = db.call_with_retry(_with_cursor, _insert)
    except Exception:
        state.clear_active()
        raise

    return BeginResult(
        run_id=run_id,
        experiment_name=exp_id,
        arm_id=arm_id,
        repeat_idx=repeat_idx,
        started_at=started_at,
    )


@dataclass
class FinishResult:
    run_id: str
    experiment_name: str
    arm_id: str
    task_success: bool
    wall_clock_s: float
    success_output: Optional[str]
    summary_metrics: dict = field(default_factory=dict)


def finish_run(manual_score: Optional[float] = None) -> FinishResult:
    active = state.get_active()
    if active is None:
        raise NoActiveRun("no active run")

    run_id = active["run_id"]
    ended_at = now()
    wall_clock_s = _parse_ts(ended_at) - _parse_ts(active["started_at"])

    with db.cursor() as cur:
        row = cur.execute(
            "SELECT r.experiment_id, e.config_yaml FROM runs r "
            "JOIN experiments e ON e.id = r.experiment_id WHERE r.id = ?",
            (run_id,),
        ).fetchone()
    if row is None:
        raise ActiveRunMissingDbRow(run_id)

    success_output = None
    if manual_score is not None:
        task_success = manual_score > 0
    else:
        cfg = yaml.safe_load(row["config_yaml"]) or {}
        task = cfg.get("task") or {}
        check = task.get("success_check")
        timeout_s = task.get("timeout_s", 1800)
        if not check:
            raise NoSuccessCheck(
                "experiment has no task.success_check; finish with a manual score instead"
            )
        try:
            proc = subprocess.run(
                check, shell=True, timeout=timeout_s, capture_output=True, text=True
            )
            task_success = proc.returncode == 0
            success_output = ((proc.stdout or "") + (proc.stderr or ""))[-4000:]
        except subprocess.TimeoutExpired:
            task_success = False
            success_output = f"success_check timed out after {timeout_s}s"

    def _write_ended(cur):
        cur.execute(
            "UPDATE runs SET ended_at=?, wall_clock_s=?, task_success=?, "
            "success_output=?, manual_score=? WHERE id=?",
            (ended_at, wall_clock_s, int(task_success), success_output, manual_score, run_id),
        )

    # `ys end` races the tail of the proxy's in-flight writes for this same
    # run more than any other writer -- retry a lock that outlasts
    # busy_timeout instead of surfacing "database is locked" as a raw
    # traceback (finding 28).
    db.call_with_retry(_with_cursor, _write_ended)

    state.clear_active()

    summary_metrics = {}
    try:
        from ys import metrics

        def _correct_fingerprint(cur):
            # The per-request fingerprint fill in ys.collector fires eagerly
            # on the first successful request, which can't yet know which
            # thread will end up being the run's main conversation -- if
            # that request happened to be a background or subagent call,
            # the run got fingerprinted against the wrong conversation (see
            # finding 4). Now that the run is finished, correct it from the
            # actual main thread.
            fingerprint = metrics.main_thread_fingerprint(cur, run_id)
            if fingerprint:
                cur.execute(
                    "UPDATE runs SET model=?, toolset_hash=?, system_prompt_hash=? WHERE id=?",
                    (
                        fingerprint["model"],
                        fingerprint["toolset_hash"],
                        fingerprint["system_prompt_hash"],
                        run_id,
                    ),
                )
            return metrics.compute_run_metrics(cur, run_id)

        summary_metrics = db.call_with_retry(_with_cursor, _correct_fingerprint)
    except ImportError:
        pass

    return FinishResult(
        run_id=run_id,
        experiment_name=active["experiment"],
        arm_id=active["arm"],
        task_success=task_success,
        wall_clock_s=wall_clock_s,
        success_output=success_output,
        summary_metrics=summary_metrics,
    )


@dataclass
class DeleteResult:
    run_id: str
    experiment_name: str
    arm_id: str


def delete_run(run_id: str) -> DeleteResult:
    """Deletes a run and its dependent requests/tool_calls rows. Refuses to
    delete the currently active run so the active-run state file never
    points at a run that no longer exists in the db."""
    active = state.get_active()
    if active is not None and active["run_id"] == run_id:
        raise CannotDeleteActiveRun(run_id)

    def _delete(cur):
        row = cur.execute(
            "SELECT experiment_id, arm_id FROM runs WHERE id = ?", (run_id,)
        ).fetchone()
        if row is None:
            raise RunNotFound(run_id)
        experiment_name = row["experiment_id"]
        arm_row = row["arm_id"]

        cur.execute("DELETE FROM tool_calls WHERE run_id = ?", (run_id,))
        cur.execute("DELETE FROM requests WHERE run_id = ?", (run_id,))
        cur.execute("DELETE FROM runs WHERE id = ?", (run_id,))
        return experiment_name, arm_row

    # RunNotFound isn't a lock/uniqueness race, so it propagates on the
    # first attempt -- only sqlite3.OperationalError/IntegrityError are
    # retried (finding 28).
    experiment_name, arm_row = db.call_with_retry(_with_cursor, _delete)

    arm_id = arm_row.split("::", 1)[1] if "::" in arm_row else arm_row
    return DeleteResult(run_id=run_id, experiment_name=experiment_name, arm_id=arm_id)
