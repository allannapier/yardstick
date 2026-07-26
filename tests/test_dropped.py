from ys import dropped


def test_count_is_zero_when_no_log_exists():
    assert dropped.count() == 0


def test_record_then_count():
    dropped.record("run-1", "database is locked")
    dropped.record("run-2", "boom")
    assert dropped.count() == 2
