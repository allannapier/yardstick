import json
import sqlite3
from contextlib import contextmanager

from ys import paths


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
    to land next in seq order. thread_key (derived in ys/collector.py from
    the system prompt hash plus a hash of the first non-system message) lets
    metrics scope themselves to the run's actual driving conversation. See
    finding 4 in IMPROVEMENTS.md."""
    _add_column_if_missing(conn, "requests", "thread_key", "TEXT")
    _add_column_if_missing(conn, "requests", "toolset_hash", "TEXT")
    _add_column_if_missing(conn, "requests", "system_prompt_hash", "TEXT")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_requests_run_thread_seq "
        "ON requests(run_id, thread_key, seq)"
    )


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
]


def connect() -> sqlite3.Connection:
    paths.ensure_home()
    conn = sqlite3.connect(paths.DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
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
                conn.execute("ROLLBACK")
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


def dumps(obj) -> str:
    return json.dumps(obj, default=str)
