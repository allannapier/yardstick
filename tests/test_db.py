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
