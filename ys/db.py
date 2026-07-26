import json
import sqlite3
from contextlib import contextmanager

from ys import paths

# Ordered migrations, applied in `init_db()` against `PRAGMA user_version`.
# Each entry is gated by user_version, so it runs at most once per database
# file — new entries do not need to be replay-safe. Migration 1 is the one
# exception: a database created before this file existed already has every
# one of these tables but `user_version = 0`, so its `CREATE TABLE IF NOT
# EXISTS` / `CREATE INDEX IF NOT EXISTS` guards are what let it converge to
# the same state as a fresh database instead of erroring. A schema change
# (new table, new column) is a new entry appended to this list; earlier
# entries are never edited after release.
MIGRATIONS: list[str] = [
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
        for version, script in enumerate(MIGRATIONS, start=1):
            if version <= current:
                continue
            try:
                conn.executescript(f"BEGIN;\n{script}\nPRAGMA user_version = {version};\nCOMMIT;")
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
