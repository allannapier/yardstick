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
