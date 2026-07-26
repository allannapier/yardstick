import sqlite3

import pytest

from ys import db


def test_connect_enables_wal_and_busy_timeout():
    conn = db.connect()
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
        assert conn.execute("PRAGMA synchronous").fetchone()[0] == 1  # NORMAL
    finally:
        conn.close()


def test_connect_warns_when_wal_mode_unavailable(monkeypatch):
    """SQLite silently falls back to another journal mode when WAL isn't
    supported (e.g. a network filesystem) -- connect() must surface that
    instead of letting the fix quietly be a no-op. sqlite3.Connection is a
    C type and can't be monkeypatched on the instance, so we route through
    a subclass via the `factory` argument instead."""

    class _FakeWALConnection(sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):
            if sql.strip().upper().startswith("PRAGMA JOURNAL_MODE"):
                class _FakeCursor:
                    def fetchone(self_inner):
                        return ("truncate",)

                return _FakeCursor()
            return super().execute(sql, *args, **kwargs)

    real_connect = sqlite3.connect

    def fake_connect(*args, **kwargs):
        kwargs["factory"] = _FakeWALConnection
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(db.sqlite3, "connect", fake_connect)

    with pytest.warns(RuntimeWarning, match="WAL"):
        conn = db.connect()
    conn.close()


def test_init_db_creates_tables():
    db.init_db()
    with db.cursor() as cur:
        tables = {
            r["name"]
            for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    assert {"experiments", "arms", "runs", "requests", "tool_calls"} <= tables


def test_migration_3_drops_the_superseded_plain_index():
    """idx_requests_run_seq (migration 1) covers the same leftmost columns
    as the new idx_requests_run_seq_unique -- keeping both is pure
    write-amplification, so migration 3 drops the old one."""
    db.init_db()
    with db.cursor() as cur:
        indexes = {
            r["name"]
            for r in cur.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='requests'"
            )
        }
    assert "idx_requests_run_seq" not in indexes
    assert "idx_requests_run_seq_unique" in indexes


def test_migration_4_adds_cost_source_column():
    """finding 9: `cost_source` (litellm/declared/unknown, ys/collector.py's
    `_resolve_cost`) is a new `requests` column, added via
    `_add_column_if_missing` like `thread_key` (migration 2) before it."""
    db.init_db()
    with db.cursor() as cur:
        cols = {r["name"] for r in cur.execute("PRAGMA table_info(requests)")}
    assert "cost_source" in cols


def test_foreign_key_enforcement():
    db.init_db()
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO experiments (id, name, question, task_json, config_yaml, created_at) "
            "VALUES ('e1','e1',NULL,'{}','','2026-01-01')"
        )
        cur.execute(
            "INSERT INTO arms (id, experiment_id, label, factors_json, is_baseline) "
            "VALUES ('a1','e1','a1','{}',0)"
        )
        cur.execute(
            "INSERT INTO runs (id, experiment_id, arm_id, repeat_idx, started_at) "
            "VALUES ('r1','e1','a1',0,'2026-01-01')"
        )

    with pytest.raises(sqlite3.IntegrityError):
        with db.cursor() as cur:
            cur.execute(
                "INSERT INTO runs (id, experiment_id, arm_id, repeat_idx, started_at) "
                "VALUES ('r2','nonexistent-experiment','a1',0,'2026-01-01')"
            )


def test_init_db_is_idempotent():
    db.init_db()
    db.init_db()  # must not raise on re-run


def test_init_db_sets_user_version_to_latest_migration():
    db.init_db()
    conn = db.connect()
    try:
        assert db.schema_version(conn) == len(db.MIGRATIONS)
    finally:
        conn.close()


def test_pre_migration_database_converges_without_error():
    """A database created before user_version tracking existed has every
    table from migration 1 but is stuck at user_version 0. Replaying
    migration 1's `CREATE TABLE IF NOT EXISTS` against it must be a no-op,
    not an error, and must still advance the version."""
    db.init_db()
    conn = db.connect()
    try:
        conn.execute("PRAGMA user_version = 0")
        conn.commit()
    finally:
        conn.close()

    db.init_db()  # must not raise despite tables already existing

    conn = db.connect()
    try:
        assert db.schema_version(conn) == len(db.MIGRATIONS)
    finally:
        conn.close()


def test_migrations_apply_incrementally_and_only_once(monkeypatch):
    db.init_db()

    monkeypatch.setattr(
        db,
        "MIGRATIONS",
        db.MIGRATIONS + ["ALTER TABLE runs ADD COLUMN extra_col TEXT;"],
    )
    db.init_db()

    conn = db.connect()
    try:
        assert db.schema_version(conn) == len(db.MIGRATIONS)
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(runs)")}
        assert "extra_col" in cols
    finally:
        conn.close()

    # already at the latest version -- must not attempt to re-run the ALTER
    # TABLE, which would raise "duplicate column name"
    db.init_db()


def _insert_experiment_arm_run(conn_or_cur, exp="e1", arm="a1", run="r1"):
    conn_or_cur.execute(
        "INSERT INTO experiments (id, name, question, task_json, config_yaml, created_at) "
        "VALUES (?,?,NULL,'{}','','2026-01-01')",
        (exp, exp),
    )
    conn_or_cur.execute(
        "INSERT INTO arms (id, experiment_id, label, factors_json, is_baseline) VALUES (?,?,?,'{}',0)",
        (arm, exp, arm),
    )
    conn_or_cur.execute(
        "INSERT INTO runs (id, experiment_id, arm_id, repeat_idx, started_at) "
        "VALUES (?,?,?,0,'2026-01-01')",
        (run, exp, arm),
    )


def test_unique_index_rejects_duplicate_seq():
    """Backstop for finding 7 -- even if some future write path allocates
    seq without holding the write lock, a collision must be rejected rather
    than silently corrupting the seq-ordered transition chain."""
    db.init_db()
    with db.cursor() as cur:
        _insert_experiment_arm_run(cur)
        cur.execute("INSERT INTO requests (run_id, seq, ts) VALUES ('r1', 1, 't1')")

    with pytest.raises(sqlite3.IntegrityError):
        with db.cursor() as cur:
            cur.execute("INSERT INTO requests (run_id, seq, ts) VALUES ('r1', 1, 't2')")


def test_migration_3_renumbers_pre_existing_seq_duplicates_before_indexing():
    """A database written before the collector's BEGIN IMMEDIATE fix could
    already have real (run_id, seq) duplicates on disk -- migration 3 must
    renumber those (preserving relative write order via the autoincrement
    id) instead of failing to add the UNIQUE index."""
    db.init_db()
    conn = db.connect()
    try:
        conn.execute("PRAGMA user_version = 2")  # pretend migration 3 never ran
        conn.execute("DROP INDEX IF EXISTS idx_requests_run_seq_unique")  # ...including its index
        _insert_experiment_arm_run(conn)
        # two requests that raced to the same seq under the old code, plus a
        # normal non-colliding one
        conn.execute("INSERT INTO requests (run_id, seq, ts) VALUES ('r1', 1, 't1')")
        conn.execute("INSERT INTO requests (run_id, seq, ts) VALUES ('r1', 1, 't2')")
        conn.execute("INSERT INTO requests (run_id, seq, ts) VALUES ('r1', 2, 't3')")
        conn.commit()
    finally:
        conn.close()

    db.init_db()  # must not raise despite the pre-existing duplicate

    with db.cursor() as cur:
        rows = cur.execute(
            "SELECT ts, seq FROM requests WHERE run_id = 'r1' ORDER BY id"
        ).fetchall()
    seqs = [r["seq"] for r in rows]
    assert len(seqs) == len(set(seqs))  # de-duplicated
    assert seqs == sorted(seqs)  # relative write order preserved

    conn = db.connect()
    try:
        assert db.schema_version(conn) == len(db.MIGRATIONS)
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO requests (run_id, seq, ts) VALUES ('r1', ?, 't4')", (seqs[0],)
            )
    finally:
        conn.close()


def test_failed_migration_rolls_back_schema_change_and_version(monkeypatch):
    """A migration that fails partway through must not leave the schema
    change applied with the version bump missing (or vice versa) -- either
    both land or neither does, so a retry sees a clean starting point."""
    db.init_db()

    monkeypatch.setattr(
        db,
        "MIGRATIONS",
        db.MIGRATIONS
        + ["ALTER TABLE runs ADD COLUMN broken_col TEXT; SELECT this_is_not_a_real_column;"],
    )

    with pytest.raises(sqlite3.OperationalError):
        db.init_db()

    conn = db.connect()
    try:
        assert db.schema_version(conn) == len(db.MIGRATIONS) - 1
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(runs)")}
        assert "broken_col" not in cols
    finally:
        conn.close()


# --- call_with_retry (finding 28) --------------------------------------------
#
# Finding 6 gave ys/collector.py's YardstickLogger._handle a retry loop, but
# every other writer (ys start, ys end, ys runs delete, the dashboard) went
# through a bare db.cursor(). call_with_retry is the one retry policy both
# now share -- these pin its behavior directly, independent of any one
# caller.


def test_call_with_retry_recovers_from_transient_operational_error(monkeypatch):
    monkeypatch.setattr(db.time, "sleep", lambda s: None)  # skip real backoff
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise sqlite3.OperationalError("database is locked")
        return "ok"

    assert db.call_with_retry(flaky) == "ok"
    assert calls["n"] == 3


def test_call_with_retry_recovers_from_integrity_error(monkeypatch):
    """finding 7's self-heal: a write that lost the UNIQUE(run_id, seq)
    race gets a fresh attempt (fresh transaction, fresh seq) instead of
    being dropped outright."""
    monkeypatch.setattr(db.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            raise sqlite3.IntegrityError("UNIQUE constraint failed")
        return "ok"

    assert db.call_with_retry(flaky) == "ok"
    assert calls["n"] == 2


def test_call_with_retry_gives_up_after_max_attempts(monkeypatch):
    monkeypatch.setattr(db.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def always_locked():
        calls["n"] += 1
        raise sqlite3.OperationalError("database is locked")

    with pytest.raises(sqlite3.OperationalError):
        db.call_with_retry(always_locked)
    assert calls["n"] == db.MAX_WRITE_ATTEMPTS


def test_call_with_retry_does_not_retry_unrelated_exceptions():
    """Only lock/uniqueness races are worth retrying -- anything else (a
    programming error, a business-rule exception like runs.RunNotFound)
    must propagate on the first attempt, not be masked by three retries."""
    calls = {"n": 0}

    def boom():
        calls["n"] += 1
        raise ValueError("boom")

    with pytest.raises(ValueError):
        db.call_with_retry(boom)
    assert calls["n"] == 1


def test_call_with_retry_passes_through_args_and_kwargs():
    def add(a, b, c=0):
        return a + b + c

    assert db.call_with_retry(add, 1, 2, c=3) == 6
