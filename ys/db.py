import json
import sqlite3
import time
import warnings
from contextlib import contextmanager

from ys import paths

# db.connect()'s busy_timeout already makes SQLite itself wait out a
# conflicting writer, but a write that outlasts the busy timeout -- or loses
# the UNIQUE(run_id, seq) race from finding 7 -- still needs a caller-level
# retry. A few attempts with a short backoff covers that residual
# contention without masking a persistently broken database. See
# `call_with_retry` below and finding 28 in IMPROVEMENTS.md.
MAX_WRITE_ATTEMPTS = 3
RETRY_BACKOFF_S = 0.2


def _add_column_if_missing(conn: sqlite3.Connection, table: str, column: str, decl: str):
    """SQLite has no `ADD COLUMN IF NOT EXISTS`, and a plain `ALTER TABLE`
    isn't safe to replay -- needed because a migration must converge cleanly
    even when run against a database that already has the column (see
    test_pre_migration_database_converges_without_error, which replays every
    migration from user_version 0 against an already-current schema)."""
    cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")


def _migration_2(conn: sqlite3.Connection):
    """thread_key + per-request toolset_hash/system_prompt_hash. Claude Code
    interleaves the main conversation with background (harness
    title-generation) requests and Task-subagent conversations in the same
    run, each its own much shorter, unrelated message history. Without a way
    to tell them apart, transition classification, turn counts, and the
    run's fingerprint were all computed against whichever request happened
    to land next in seq order. thread_key (assigned in ys/collector.py by
    chain-following each request to the most recent same-system-prompt
    request it plausibly continues) lets metrics scope themselves to the
    run's actual driving conversation. See finding 4 in IMPROVEMENTS.md."""
    _add_column_if_missing(conn, "requests", "thread_key", "TEXT")
    _add_column_if_missing(conn, "requests", "toolset_hash", "TEXT")
    _add_column_if_missing(conn, "requests", "system_prompt_hash", "TEXT")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_requests_run_thread_seq "
        "ON requests(run_id, thread_key, seq)"
    )


def _migration_3(conn: sqlite3.Connection):
    """UNIQUE(run_id, seq) backstop for finding 7 -- `_next_seq` in
    ys/collector.py read MAX(seq) and inserted with no constraint to catch a
    collision, so two concurrent writers (parallel tool use, a subagent)
    could read the same max and both write it, corrupting the transition
    chain that `_resolve_thread` orders by seq. The collector now allocates
    seq inside a `BEGIN IMMEDIATE` transaction, which serializes writers
    against this same database file and should make a collision impossible
    going forward; the index is a hard backstop in case some future write
    path doesn't go through that transaction.

    A database written before the collector fix may already have real
    (run_id, seq) duplicates on disk, which would make `CREATE UNIQUE INDEX`
    fail outright -- if any exist, every run's requests are renumbered
    densely in (seq, id) order first (id, the autoincrement rowid, reflects
    actual write order, so this preserves relative order). Checked first and
    skipped when absent, since this is a full-table rewrite and every
    database created after the collector fix will never have a duplicate to
    fix. `idx_requests_run_seq` (migration 1) is dropped in favour of the
    new unique index, which covers the same leftmost columns and makes the
    old one redundant -- keeping both would only add write-amplification."""
    has_dupes = conn.execute(
        "SELECT 1 FROM requests GROUP BY run_id, seq HAVING COUNT(*) > 1 LIMIT 1"
    ).fetchone()
    if has_dupes:
        conn.execute(
            """
            WITH ranked AS (
                SELECT id, ROW_NUMBER() OVER (PARTITION BY run_id ORDER BY seq, id) AS rn
                FROM requests
            )
            UPDATE requests SET seq = (SELECT rn FROM ranked WHERE ranked.id = requests.id)
            """
        )
    conn.execute("DROP INDEX IF EXISTS idx_requests_run_seq")
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_requests_run_seq_unique ON requests(run_id, seq)"
    )


def _migration_4(conn: sqlite3.Connection):
    """`cost_source` per request -- finding 9. LiteLLM's own cost map
    returns 0.0, silently, for any model id it has no price for (verified
    true for `claude-sonnet-5` as configured in
    experiments/interactive-sonnet.yaml). `ys/collector.py` now records
    which of `litellm` / `declared` / `unknown` produced a request's
    `response_cost`: `litellm` when LiteLLM's own number is nonzero,
    `declared` when it was zero but an experiment's `pricing:` block
    (ys/experiment.py) could price the tokens instead, `unknown` when
    neither could -- so `ys compare`/`ys report` can flag a request whose
    cost is a genuine unknown rather than silently folding a confident $0
    into the total."""
    _add_column_if_missing(conn, "requests", "cost_source", "TEXT")


# Ordered migrations, applied in `init_db()` against `PRAGMA user_version`.
# Each entry is gated by user_version, so it runs at most once per database
# file in the normal case -- but every entry must still converge cleanly if
# replayed from user_version 0 against a database that already has its
# changes applied (a database created before this file existed has every
# table but `user_version = 0`; see test_pre_migration_database_converges_
# without_error). A `CREATE TABLE IF NOT EXISTS` / `CREATE INDEX IF NOT
# EXISTS` string is naturally replay-safe; anything else (e.g. `ALTER TABLE
# ADD COLUMN`, which SQLite has no `IF NOT EXISTS` form for) must be a
# callable that guards itself, like `_migration_2` above. A schema change is
# a new entry appended to this list; earlier entries are never edited after
# release.
MIGRATIONS: list = [
    # 1: initial schema
    """
CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    question TEXT,
    task_json TEXT NOT NULL,
    config_yaml TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS arms (
    id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL REFERENCES experiments(id),
    label TEXT NOT NULL,
    factors_json TEXT NOT NULL,
    is_baseline INTEGER NOT NULL DEFAULT 0,
    UNIQUE(experiment_id, label)
);

CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL REFERENCES experiments(id),
    arm_id TEXT NOT NULL REFERENCES arms(id),
    repeat_idx INTEGER NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    wall_clock_s REAL,
    task_success INTEGER,
    success_output TEXT,
    manual_score REAL,
    harness_user_agent TEXT,
    model TEXT,
    toolset_hash TEXT,
    tool_count INTEGER,
    system_prompt_hash TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(id),
    seq INTEGER NOT NULL,
    ts TEXT NOT NULL,
    provider TEXT,
    model TEXT,
    stream INTEGER,
    input_tokens INTEGER,
    cache_creation INTEGER,
    cache_read INTEGER,
    output_tokens INTEGER,
    response_cost REAL,
    latency_ms REAL,
    ttft_ms REAL,
    status_code INTEGER,
    error TEXT,
    msg_count INTEGER,
    msg_hashes_json TEXT,
    system_tokens INTEGER,
    tools_tokens INTEGER,
    transition TEXT
);

CREATE INDEX IF NOT EXISTS idx_requests_run_seq ON requests(run_id, seq);

CREATE TABLE IF NOT EXISTS tool_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id INTEGER NOT NULL REFERENCES requests(id),
    run_id TEXT NOT NULL REFERENCES runs(id),
    name TEXT NOT NULL,
    input_hash TEXT NOT NULL,
    input_bytes INTEGER,
    is_error INTEGER NOT NULL DEFAULT 0,
    result_tokens INTEGER,
    provider_call_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_tool_calls_request ON tool_calls(request_id);
CREATE INDEX IF NOT EXISTS idx_tool_calls_provider_call_id ON tool_calls(run_id, provider_call_id);
""",
    # 2: see _migration_2 above.
    _migration_2,
    # 3: see _migration_3 above.
    _migration_3,
    # 4: see _migration_4 above.
    _migration_4,
]


def connect() -> sqlite3.Connection:
    """The collector writes from inside the proxy process while the CLI and
    the dashboard read/write the same file -- WAL lets readers and the
    writer proceed concurrently instead of blocking on each other, and
    busy_timeout makes a writer wait out a conflicting writer instead of
    failing immediately with "database is locked". synchronous=NORMAL is
    the pairing WAL's own docs recommend: still durable across an OS crash,
    just not fsync-per-commit. See finding 6 in IMPROVEMENTS.md."""
    paths.ensure_home()
    conn = sqlite3.connect(paths.DB_PATH, timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    journal_mode = conn.execute("PRAGMA journal_mode = WAL").fetchone()[0]
    if journal_mode.lower() != "wal":
        # SQLite silently falls back to another journal mode when WAL isn't
        # supported (e.g. a network filesystem) -- surface that instead of
        # letting this whole fix quietly be a no-op.
        warnings.warn(
            f"could not enable SQLite WAL mode (got journal_mode={journal_mode!r}); "
            "concurrent readers/writer will contend more than expected",
            RuntimeWarning,
            stacklevel=2,
        )
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA synchronous = NORMAL")
    return conn


def schema_version(conn: sqlite3.Connection) -> int:
    return conn.execute("PRAGMA user_version").fetchone()[0]


def init_db():
    conn = connect()
    conn.isolation_level = None  # manual transaction control: each migration's schema change and version bump commit as one atomic unit
    try:
        current = schema_version(conn)
        for version, migration in enumerate(MIGRATIONS, start=1):
            if version <= current:
                continue
            try:
                if callable(migration):
                    conn.execute("BEGIN")
                    migration(conn)
                    conn.execute(f"PRAGMA user_version = {version}")
                    conn.execute("COMMIT")
                else:
                    conn.executescript(
                        f"BEGIN;\n{migration}\nPRAGMA user_version = {version};\nCOMMIT;"
                    )
            except Exception:
                try:
                    conn.execute("ROLLBACK")
                except sqlite3.OperationalError:
                    # No transaction to roll back (e.g. the failure was the
                    # BEGIN itself, or executescript already unwound it).
                    # Never let cleanup replace the real migration error.
                    pass
                raise
    finally:
        conn.close()


@contextmanager
def cursor():
    conn = connect()
    try:
        cur = conn.cursor()
        yield cur
        conn.commit()
    finally:
        conn.close()


def call_with_retry(fn, *args, **kwargs):
    """Call `fn(*args, **kwargs)`, retrying up to `MAX_WRITE_ATTEMPTS` times
    with a short linear backoff on `sqlite3.OperationalError` (a lock that
    outlasted `connect()`'s busy_timeout) or `sqlite3.IntegrityError` (the
    `UNIQUE(run_id, seq)` backstop from finding 7 -- a fresh attempt gets a
    fresh transaction and a fresh seq, so it self-heals). Any other
    exception propagates immediately; only lock/uniqueness races are worth
    retrying.

    Originally this loop lived only in `ys/collector.py`'s `YardstickLogger.
    _handle` (finding 6); every other writer -- `ys start`, `ys end`,
    `ys runs delete`, the dashboard -- went through a bare `db.cursor()` and
    surfaced "database is locked" as an unhandled traceback instead (finding
    28). This is the one retry policy both paths now share.

    `fn` must open (and commit) its own `db.cursor()` per call, so a retried
    attempt starts from a clean connection/transaction rather than replaying
    inside one that already failed partway through -- see `ys/collector.py`'s
    `_write` and `ys/runs.py`'s write helpers for the two shapes of caller.
    """
    for attempt in range(1, MAX_WRITE_ATTEMPTS + 1):
        try:
            return fn(*args, **kwargs)
        except (sqlite3.OperationalError, sqlite3.IntegrityError):
            if attempt == MAX_WRITE_ATTEMPTS:
                raise
            time.sleep(RETRY_BACKOFF_S * attempt)


def dumps(obj) -> str:
    return json.dumps(obj, default=str)
