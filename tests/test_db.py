from ys import db


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

    import pytest
    import sqlite3

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
