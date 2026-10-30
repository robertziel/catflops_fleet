"""S1 — the toolchain: the engine is importable in the image and bootstraps sqlite."""
import os
import tempfile


def test_engine_bootstraps_a_throwaway_sqlite_db(monkeypatch):
    from tests.conftest import needs_broker
    needs_broker()
    monkeypatch.setenv("LB_TEST_DB", os.path.join(tempfile.mkdtemp(), "b.db"))
    import queue_workflows as qw
    from queue_workflows import db

    qw.configure(db_backend="sqlite", db_url_env="LB_TEST_DB")
    db.bootstrap()
    assert qw.__version__.startswith("1.")
