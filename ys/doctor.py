"""Feature 4 in IMPROVEMENTS.md: `ys doctor`, a single read-only preflight
over every moving part findings 3, 5, 6, 8, 12 and 29 already diagnose one
at a time -- the yardstick home directory, the database schema version, the
proxy process, the generated proxy config, each agent's harness config, the
two API keys, active-run state, and the unattributed/dropped counters.

Each check returns a `CheckResult` (pass/warn/fail/skip plus one specific,
actionable message) instead of a bare bool -- the whole value of this
command is in the wording, not the verdict. `run_checks` composes them in a
fixed order and is the one function `ys/cli.py`'s `doctor` command calls.

Hard rule, unlike every other command in this file's neighbourhood: nothing
here may mutate anything. No `db.init_db()`/migration, no
`paths.ensure_home()`, no starting or stopping the proxy/dashboard, no
writing to a harness config. A user running `ys doctor` to find out *why*
something is broken must never have the act of asking change the answer.
Where a helper this module calls would normally have a side effect (e.g.
`db.connect()` calling `paths.ensure_home()`), this module works around it
(a plain `sqlite3.connect` here, guarded by an existence check) rather than
accepting the side effect as a shortcut.

Reuses rather than re-derives: `proxy.model_available` /
`proxy.model_check_skipped_message` (findings 3/29), `harness.status`
(finding 5's sibling concern -- is the agent still pointed at a proxy that
might not be there), `runs.unattributed_summary` (finding 12, "the single
highest-value diagnostic in the app" per the plan), `dropped.count`
(finding 6), and `experiment.validate_task_paths` (findings 15-18 -- the
same `task.prompt_file`/`task.repo` filesystem check `ys start` already
runs before claiming the active slot). None of their logic or wording is
reimplemented here.
"""
import os
import sqlite3
from dataclasses import dataclass
from typing import Optional

import yaml

from ys import db, dropped, harness, paths, procutil, proxy, runs, state
from ys.experiment import Experiment, load_experiment, validate_task_paths

PASS = "pass"
WARN = "warn"
FAIL = "fail"
SKIP = "skip"


@dataclass
class CheckResult:
    name: str
    status: str  # PASS | WARN | FAIL | SKIP
    message: str


def check_home_directory() -> CheckResult:
    """Deliberately `os.path.isdir`, not `paths.ensure_home()` -- the latter
    creates the directory tree as a side effect, which would make merely
    running `ys doctor` before `ys init` bring `~/.yardstick` into
    existence. A missing home directory is reported, not fixed."""
    home = paths.YARDSTICK_HOME
    if not os.path.isdir(home):
        return CheckResult(
            "yardstick home",
            WARN,
            f"{home} does not exist yet -- run `ys init` to create it.",
        )
    if not os.access(home, os.W_OK):
        return CheckResult(
            "yardstick home",
            FAIL,
            f"{home} exists but is not writable by this user -- every run/proxy/"
            "harness command writes here; fix its permissions.",
        )
    return CheckResult("yardstick home", PASS, f"{home} exists and is writable")


def check_schema_version() -> CheckResult:
    """Finding 8: `PRAGMA user_version` against `db.MIGRATIONS`. Opens a
    plain `sqlite3.connect` rather than `db.connect()`/`db.init_db()` -- the
    former sets WAL/busy_timeout pragmas and the latter would apply any
    pending migration, neither of which a read-only check should trigger.
    The existence check first means this never creates the file either
    (sqlite3.connect on a nonexistent path would)."""
    if not os.path.exists(paths.DB_PATH):
        return CheckResult(
            "schema version",
            WARN,
            f"no database at {paths.DB_PATH} yet -- run `ys init` to create it.",
        )
    conn = sqlite3.connect(paths.DB_PATH)
    try:
        current = conn.execute("PRAGMA user_version").fetchone()[0]
    finally:
        conn.close()
    latest = len(db.MIGRATIONS)
    if current < latest:
        return CheckResult(
            "schema version",
            WARN,
            f"database is at schema version {current}, code expects {latest} -- run "
            "any yardstick command that calls db.init_db() (e.g. `ys init`, `ys proxy "
            "up`) to migrate it.",
        )
    if current > latest:
        return CheckResult(
            "schema version",
            WARN,
            f"database is at schema version {current}, newer than this code's "
            f"{latest} migrations -- you're likely running an older yardstick "
            "install against a database a newer one already migrated.",
        )
    return CheckResult("schema version", PASS, f"schema is current (version {current})")


def check_proxy_process(port: Optional[int] = None) -> CheckResult:
    """Finding 5: the exact pidfile/port bookkeeping `ys proxy down`'s
    SIGKILL escalation relies on -- a stale pidfile plus a still-bound port
    is precisely the state that fix's docstring calls out as the old
    silent-orphan failure mode."""
    alive, pid = proxy.proxy_status()
    effective_port = port if port is not None else proxy.read_port()
    if alive:
        return CheckResult("proxy process", PASS, f"running (pid {pid}, port {effective_port})")
    if pid is not None:
        if procutil.port_in_use(effective_port):
            return CheckResult(
                "proxy process",
                FAIL,
                f"pidfile points at pid {pid}, which is not running, but port "
                f"{effective_port} is still bound by some other process -- run `ys "
                "proxy down --force` to clear the stale pidfile (finding 5), then "
                "find out what's holding the port before starting a new proxy there.",
            )
        return CheckResult(
            "proxy process",
            WARN,
            f"not running (stale pidfile for pid {pid}) -- run `ys proxy up --exp "
            "<experiment>` to start it.",
        )
    if procutil.port_in_use(effective_port):
        return CheckResult(
            "proxy process",
            WARN,
            f"no pidfile, but port {effective_port} is already bound by some other "
            "process -- `ys proxy up` will fail there until you free the port or "
            "pass a different --port.",
        )
    return CheckResult(
        "proxy process",
        WARN,
        "not running -- run `ys proxy up --exp <experiment>` before pointing a "
        "harness at it.",
    )


def check_generated_config(
    experiment: Optional[Experiment] = None, arm_id: Optional[str] = None
) -> CheckResult:
    """Whether `ys proxy up`'s generated `model_list` (`proxy.generate_config`)
    has an explicit entry for the given arm's model. This is the static,
    read-from-disk sibling of `check_model_available` below -- it still says
    something useful when the proxy isn't currently running at all, since it
    doesn't need a live server to answer."""
    if not os.path.exists(paths.PROXY_CONFIG_PATH):
        return CheckResult(
            "generated proxy config",
            WARN,
            f"no generated config at {paths.PROXY_CONFIG_PATH} yet -- run `ys proxy "
            "up --exp <experiment>` to generate one.",
        )
    try:
        with open(paths.PROXY_CONFIG_PATH) as f:
            config = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        return CheckResult(
            "generated proxy config",
            FAIL,
            f"{paths.PROXY_CONFIG_PATH} does not parse as YAML ({e}) -- regenerate it "
            "with `ys proxy up --exp <experiment>`.",
        )
    names = {entry.get("model_name") for entry in config.get("model_list", [])}
    if experiment is None or arm_id is None:
        return CheckResult(
            "generated proxy config",
            PASS,
            f"{paths.PROXY_CONFIG_PATH} parses ({len(names)} model(s) registered)",
        )
    try:
        arm_obj = experiment.get_arm(arm_id)
    except KeyError as e:
        return CheckResult("generated proxy config", FAIL, str(e))
    model = arm_obj.factors.get("model")
    if not model:
        return CheckResult(
            "generated proxy config",
            SKIP,
            f"arm '{arm_id}' has no 'model' factor -- nothing to check it against.",
        )
    if model in names:
        return CheckResult(
            "generated proxy config", PASS, f"model '{model}' has an explicit model_list entry"
        )
    return CheckResult(
        "generated proxy config",
        WARN,
        f"generated config has no explicit entry for model '{model}' -- it would only "
        "work via the '*' catch-all passthrough (mock_response/params declared for it "
        "won't apply). Run `ys proxy up --exp <experiment>` to register it.",
    )


def check_task_paths(experiment: Optional[Experiment]) -> CheckResult:
    """Findings 15-18: `task.prompt_file`/`task.repo` are the declared hooks
    features 1/2 (unattended runs, workspace isolation) will read -- `ys
    start` already refuses to begin a run when they don't check out
    (`validate_task_paths`, called before the active-run slot is claimed).
    Reuses that exact function, not its own copy of the path/scheme
    heuristics, so `ys doctor` catches the same typo before you even get to
    `ys start`."""
    if experiment is None:
        return CheckResult(
            "task paths",
            SKIP,
            "pass --exp to check task.prompt_file/task.repo exist (findings 15-18).",
        )
    problems = validate_task_paths(experiment.task)
    if problems:
        return CheckResult(
            "task paths",
            FAIL,
            "; ".join(problems) + " -- `ys start` will refuse to begin a run until "
            "this is fixed.",
        )
    if experiment.task.prompt_file or experiment.task.repo:
        return CheckResult("task paths", PASS, "task.prompt_file/task.repo check out")
    return CheckResult("task paths", PASS, "task declares no prompt_file/repo -- nothing to check")


def check_model_available(model: str, port: int, master_key: Optional[str]) -> CheckResult:
    """Finding 3's live check and finding 29's fix for silently skipping it
    -- same helpers `ys start` calls (`proxy.model_available`,
    `proxy.model_check_skipped_message`), same three-way outcome, reused
    verbatim rather than re-worded."""
    if not master_key:
        return CheckResult("proxy serves model", WARN, proxy.model_check_skipped_message(model))
    available = proxy.model_available(model, port, master_key)
    if available is True:
        return CheckResult(
            "proxy serves model", PASS, f"the proxy on port {port} has an explicit entry for model '{model}'"
        )
    if available is False:
        return CheckResult(
            "proxy serves model",
            FAIL,
            f"the proxy on port {port} has no explicit entry for model '{model}' -- it "
            "will only work via the catch-all passthrough (mock_response/params "
            "declared for it won't apply). Run `ys proxy up --exp <experiment>` to "
            "register it.",
        )
    return CheckResult(
        "proxy serves model",
        WARN,
        f"could not reach the proxy on port {port} to verify it serves model "
        f"'{model}' -- is `ys proxy up` running?",
    )


def check_harness_config(agent_name: str) -> CheckResult:
    """The other half of finding 5's process concern: not just "is the
    proxy up", but "does this agent's config still point at one". Reuses
    `harness.status` instead of re-reading/re-parsing the agent's config
    file a second way.

    Feature 5 added agents with no config file at all (`env_only=True`,
    e.g. aider) -- `harness.status` already reports those with
    `config_exists=False`/`config_path=""` rather than raising, so this
    only needs a nicer message for that specific case instead of new
    branching logic."""
    name = f"harness config ({agent_name})"
    try:
        s = harness.status(agent_name)
    except harness.HarnessError as e:
        return CheckResult(name, FAIL, str(e))
    if s.env_only:
        return CheckResult(
            name, PASS, "env-only (no config file to check) -- see `ys harness point --env-only`"
        )
    if not s.pointed_at_proxy:
        if s.config_exists:
            return CheckResult(name, PASS, f"not pointed at a proxy ({s.config_path})")
        return CheckResult(name, PASS, f"config doesn't exist yet at {s.config_path} (nothing to check)")
    alive, pid = proxy.proxy_status()
    if alive:
        return CheckResult(
            name, PASS, f"pointed at the proxy, and a proxy is running (pid {pid}) ({s.config_path})"
        )
    return CheckResult(
        name,
        FAIL,
        f"{s.config_path} points at a proxy on localhost, but no proxy is running -- "
        f"real requests will fail until `ys proxy up` again, or run `ys harness reset "
        f"{agent_name}` to restore the backed-up config.",
    )


def check_api_keys() -> list:
    """The two API keys the plan calls out by name: `LITELLM_MASTER_KEY`
    (auth to the proxy itself -- what the harness sends, and what finding
    29's check needs to query `/v1/models`) and `ANTHROPIC_API_KEY` (the
    real provider key `proxy.generate_config`'s `os.environ/ANTHROPIC_API_KEY`
    params substitute at request time). Both are read from *this* shell's
    environment, same as `ys start`'s existing `LITELLM_MASTER_KEY` check --
    `ANTHROPIC_API_KEY` is really needed wherever the proxy process runs,
    which is worth saying explicitly since that's often a different shell."""
    results = []
    if os.environ.get("LITELLM_MASTER_KEY"):
        results.append(CheckResult("LITELLM_MASTER_KEY", PASS, "set in this shell"))
    else:
        results.append(
            CheckResult(
                "LITELLM_MASTER_KEY",
                WARN,
                "not set in this shell -- export the same key you started (or will "
                "start) `ys proxy up` with; `ys start`'s own model-availability check "
                "needs it too (finding 29 in IMPROVEMENTS.md).",
            )
        )
    if os.environ.get("ANTHROPIC_API_KEY"):
        results.append(CheckResult("ANTHROPIC_API_KEY", PASS, "set in this shell"))
    else:
        results.append(
            CheckResult(
                "ANTHROPIC_API_KEY",
                WARN,
                "not set in this shell -- the proxy substitutes this for "
                "os.environ/ANTHROPIC_API_KEY when it forwards a request to Anthropic "
                "(ys/proxy.py's model_list/catch-all params). Export it wherever `ys "
                "proxy up` actually runs, not necessarily in this shell.",
            )
        )
    return results


def check_active_run() -> CheckResult:
    """Cross-checks active.json against the `runs` table -- the exact
    mismatch `ys end` raises `ActiveRunMissingDbRow` for, surfaced here
    before you get to `ys end` and hit it."""
    active = state.get_active()
    if active is None:
        return CheckResult("active-run state", PASS, "no active run")
    if not os.path.exists(paths.DB_PATH):
        # db.cursor() would both create an empty, schema-less database file
        # (a mutation this command must never cause) and then fail with
        # "no such table" querying it -- report the inconsistency directly
        # instead. check_schema_version already flags the missing database
        # on its own line.
        return CheckResult(
            "active-run state",
            FAIL,
            f"active.json claims run {active['run_id']} is active (exp="
            f"{active['experiment']} arm={active['arm']}), but no database exists at "
            f"{paths.DB_PATH} at all -- see the schema version check above.",
        )
    with db.cursor() as cur:
        row = cur.execute("SELECT id FROM runs WHERE id = ?", (active["run_id"],)).fetchone()
    if row is None:
        return CheckResult(
            "active-run state",
            FAIL,
            f"active.json points at run {active['run_id']} (exp={active['experiment']} "
            f"arm={active['arm']}), but no such row exists in the database -- `ys end` "
            f"would fail with ActiveRunMissingDbRow. Remove {paths.ACTIVE_RUN_PATH} by "
            "hand, or `ys start --force` a new run.",
        )
    return CheckResult(
        "active-run state",
        PASS,
        f"run {active['run_id']} active (exp={active['experiment']} arm={active['arm']}, "
        f"started {active['started_at']})",
    )


def check_unattributed() -> CheckResult:
    """Finding 12 -- "the single highest-value diagnostic in the app" per
    the plan. Reuses `runs.unattributed_summary()` verbatim, the same query
    `ys status`/`ys end`'s `_print_unattributed_notice` already run."""
    if not os.path.exists(paths.DB_PATH):
        # Same reasoning as check_active_run: don't let a plain query create
        # an empty database file as a side effect on a system that never
        # ran `ys init`.
        return CheckResult(
            "unattributed requests", SKIP, "no database yet -- see the schema version check above."
        )
    summary = runs.unattributed_summary()
    if summary.count:
        return CheckResult(
            "unattributed requests",
            WARN,
            f"{summary.count} request(s) since {summary.since} UTC could not be "
            "attributed to a run (cumulative across every run ever recorded) -- check "
            "the harness is pointed at the proxy and either sending x-ys-run or "
            "running while `ys start` has the active-run slot claimed.",
        )
    return CheckResult("unattributed requests", PASS, "none recorded")


def check_dropped() -> CheckResult:
    """Finding 6 -- reuses `dropped.count()` verbatim, the same count `ys
    status`/`ys end` already print."""
    count = dropped.count()
    if count:
        return CheckResult(
            "dropped requests",
            WARN,
            f"{count} request(s) could not be written to the database and were "
            f"dropped (cumulative, recorded in {paths.DROPPED_LOG_PATH}).",
        )
    return CheckResult("dropped requests", PASS, "none recorded")


def run_checks(
    exp: Optional[str] = None, arm: Optional[str] = None, port: Optional[int] = None
) -> list:
    """Runs every check and returns them in a fixed, readable order. Nothing
    here mutates state -- see the module docstring. `exp`/`arm` are both
    optional; without them, every check that doesn't need a concrete
    experiment/arm still runs, and the proxy-serves-model / generated-config
    checks degrade to their experiment-agnostic form (or a SKIP row saying
    what's missing) instead of erroring."""
    results = [check_home_directory(), check_schema_version(), check_proxy_process(port)]

    experiment: Optional[Experiment] = None
    if exp:
        try:
            experiment = load_experiment(exp)
        except Exception as e:
            results.append(CheckResult("experiment YAML", FAIL, f"could not load {exp}: {e}"))

    results.append(check_generated_config(experiment, arm))
    results.append(check_task_paths(experiment))

    for agent_name in harness.AGENTS:
        results.append(check_harness_config(agent_name))

    results.extend(check_api_keys())
    results.append(check_active_run())
    results.append(check_unattributed())
    results.append(check_dropped())

    if experiment is not None and arm:
        try:
            arm_obj = experiment.get_arm(arm)
        except KeyError as e:
            results.append(CheckResult("proxy serves model", FAIL, str(e)))
        else:
            model = arm_obj.factors.get("model")
            if not model:
                results.append(
                    CheckResult(
                        "proxy serves model",
                        SKIP,
                        f"arm '{arm}' has no 'model' factor -- nothing to verify.",
                    )
                )
            else:
                effective_port = port if port is not None else proxy.read_port()
                master_key = os.environ.get("LITELLM_MASTER_KEY")
                results.append(check_model_available(model, effective_port, master_key))
    elif exp or arm:
        results.append(
            CheckResult(
                "proxy serves model",
                SKIP,
                "--exp and --arm must both be given to verify the proxy serves a "
                "specific arm's model.",
            )
        )
    else:
        results.append(
            CheckResult(
                "proxy serves model",
                SKIP,
                "pass --exp/--arm to verify the running proxy serves a specific arm's "
                "model.",
            )
        )

    return results
