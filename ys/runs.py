"""Run lifecycle business logic, shared by the CLI (ys/cli.py) and the web
dashboard (ys/web/app.py) so there is exactly one implementation of the
active-slot-before-db-write ordering, config_yaml refresh, and orphan-row
avoidance that ys/cli.py originally had inline. Both callers catch the
exceptions here and format them for their own presentation layer (Rich
console vs JSON/HTML).
"""
import hashlib
import json
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import yaml

from ys import db, state
from ys.experiment import Arm, Experiment

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


def config_hash_for_arm(experiment: Experiment, arm: Arm) -> str:
    """Finding 14: a per-run fingerprint of exactly the parts of an
    experiment's config that determine what a run of `arm` actually
    executes -- `task` (id/repo/ref/prompt_file/success_check/timeout_s)
    and this arm's own `factors` (which is where `model` lives). Snapshotted
    onto each run row at `begin_run` so `ys/render.py`'s `compare_experiment`
    can group an arm's run history by "which version of the config produced
    this run" instead of silently aggregating every run ever attributed to
    the arm id, including ones from before the task, success_check, or
    model changed.

    Deliberately NOT the raw YAML text and NOT the whole parsed Experiment:
    hashing the entire file would split an arm's history over an edited
    comment or a `question:` tweak that changed nothing about what ran, and
    hashing only e.g. `task.id` would let a changed `success_check` or a
    changed `factors.model` (a different model under the same arm id)
    silently keep aggregating with runs that were scored, or ran, under
    different terms -- exactly the failure mode finding 14 is about.
    Also deliberately excludes `experiment.metrics`/`.pricing`/
    `.billable_weights`/other arms' `factors`: those change how a run's
    numbers are displayed or priced after the fact, not what the agent was
    asked to do or how its result was judged, so a change to them alone
    should not fragment comparability.
    """
    payload = {
        "task": experiment.task.model_dump(),
        "factors": arm.factors,
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


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
    # Snapshotted onto the run row itself, independent of `experiments`
    # (which keeps being overwritten below) -- see finding 14 and
    # `config_hash_for_arm`'s docstring for what the hash covers.
    task_json_snapshot = db.dumps(experiment.task.model_dump())
    config_hash = config_hash_for_arm(experiment, arm_obj)

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
            "INSERT INTO runs (id, experiment_id, arm_id, repeat_idx, started_at, notes, "
            "task_json_snapshot, config_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                exp_id,
                a_id,
                repeat_idx,
                started_at,
                arm_obj.notes,
                task_json_snapshot,
                config_hash,
            ),
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

    # Finding 11: record the drain-window fallback before clearing the
    # active-run slot, so a request racing these two writes -- or arriving
    # in the (short) gap between them -- still has `state.get_draining_run`
    # to fall back to; `clear_active` still runs unconditionally right after
    # so `ys status`/the next `ys start` see the slot free immediately.
    state.mark_ended(run_id, active["experiment"], active["arm"], ended_at)
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


@dataclass
class UnattributedSummary:
    count: int
    since: Optional[str]  # "HH:MM" (UTC) of the earliest unattributed request, or None if count == 0


def unattributed_summary() -> UnattributedSummary:
    """A request the collector can't attribute to a real run -- no
    `x-ys-run` header, no active-run file, and (finding 11) outside the
    drain window of the run that most recently ended -- lands in the
    synthetic 'unattributed' run (`ys/collector.py`'s `_resolve_run_id`/
    `_ensure_run_exists`). Before this, nothing surfaced that anywhere: a
    misconfigured harness produced a run with zero requests and no
    explanation in the CLI or the dashboard -- exactly the situation
    finding 3 puts everyone in on their first real run. `ys status`/`ys end`
    print this so it's visible instead of silently invisible (finding 12).
    """
    with db.cursor() as cur:
        row = cur.execute(
            "SELECT COUNT(*) AS c, MIN(ts) AS earliest FROM requests WHERE run_id = 'unattributed'"
        ).fetchone()
    count = row["c"] or 0
    since = row["earliest"][11:16] if count and row["earliest"] else None
    return UnattributedSummary(count=count, since=since)


@dataclass
class RunListRow:
    run_id: str
    experiment_id: str
    arm_id: str
    repeat_idx: int
    started_at: str
    status: str  # "finished" | "unfinished" | "abandoned" -- see finding 13
    success: Optional[bool]  # None if the run has no verdict yet
    config_current: Optional[bool]  # None if not checked (no experiment_obj given)


def list_runs(
    experiment: Optional[str] = None,
    arm: Optional[str] = None,
    limit: Optional[int] = None,
    experiment_obj: Optional[Experiment] = None,
) -> list:
    """P2 (IMPROVEMENTS.md): "there is no `ys runs list`. Runs can be
    deleted by id but never enumerated, so the only way to find an id is the
    dashboard or raw SQL." Joins `arms` to report the bare arm id
    (`arms.label`, e.g. `arm-a`) instead of the internal `experiment::arm`
    row id `runs.arm_id` is actually stored as.

    `status` surfaces finding 13's `abandoned` column and plain
    never-`ys end`-ed runs (`ended_at IS NULL`) as distinct from a normally
    `finished` run -- the same three-way split `aggregate_run_metrics`
    already uses to decide what counts toward `n_runs`/`n_unfinished`,
    just named per-row instead of only summarized in `ys compare`.

    `config_current` answers finding 14's "which of these runs still count"
    question, the one the P2 section says sends people to raw SQL in the
    first place: given `experiment_obj` (the experiment's *current* parsed
    YAML), each row's stored `config_hash` (see `config_hash_for_arm`) is
    compared against what that arm's config hashes to today. A NULL stored
    hash (a run recorded before finding 14 shipped) is never treated as
    current, same as `render.compare_experiment`'s rule -- silently trusting
    unverifiable old data is exactly the failure mode finding 14 is about.
    An arm the current YAML no longer declares at all is also reported as
    stale rather than raising, since "the arm was renamed/removed" is itself
    exactly the kind of drift this is meant to surface. Left `None`
    (not computed) when `experiment_obj` isn't given, e.g. an unscoped
    `ys runs list` with no `--exp`.
    """
    query = (
        "SELECT r.id, r.experiment_id, a.label AS arm_label, r.repeat_idx, "
        "r.started_at, r.ended_at, r.task_success, r.abandoned, r.config_hash "
        "FROM runs r JOIN arms a ON a.id = r.arm_id "
        "WHERE (? IS NULL OR r.experiment_id = ?) AND (? IS NULL OR a.label = ?) "
        # started_at has only second resolution, so two runs begun within the
        # same second (routine in a fast test suite, and not impossible for a
        # human either) would otherwise tie -- runs.id is a UUID, not an
        # insertion-ordered key, so the tiebreaker is SQLite's implicit
        # `rowid` (monotonically increasing on insert for an ordinary
        # rowid table, which this is -- no WITHOUT ROWID), breaking the tie
        # in favour of "most recent first" instead of SQLite's unspecified
        # order among ties.
        "ORDER BY r.started_at DESC, r.rowid DESC"
    )
    params: tuple = (experiment, experiment, arm, arm)
    if limit is not None:
        query += " LIMIT ?"
        params = params + (limit,)

    with db.cursor() as cur:
        rows = cur.execute(query, params).fetchall()

    result = []
    for row in rows:
        if row["abandoned"]:
            status = "abandoned"
        elif row["ended_at"] is None:
            status = "unfinished"
        else:
            status = "finished"

        success = bool(row["task_success"]) if row["task_success"] is not None else None

        config_current = None
        if experiment_obj is not None:
            try:
                arm_obj = experiment_obj.get_arm(row["arm_label"])
            except KeyError:
                config_current = False
            else:
                if row["config_hash"] is None:
                    config_current = False
                else:
                    config_current = row["config_hash"] == config_hash_for_arm(experiment_obj, arm_obj)

        result.append(
            RunListRow(
                run_id=row["id"],
                experiment_id=row["experiment_id"],
                arm_id=row["arm_label"],
                repeat_idx=row["repeat_idx"],
                started_at=row["started_at"],
                status=status,
                success=success,
                config_current=config_current,
            )
        )
    return result
