"""The lease declares the context window it needs (broker_parrot migration 0028) — when it can.

Three engine/DB states exist during a rollout, and the lease must be created in all three:
  * engine WITHOUT `enqueue_node_job(context_length=)` (1.4.0 — what the fleet runs today)
    → unsized lease, status says why;
  * engine WITH it and the column present → sized lease; the 0028 claim gate then admits it
    only on a box serving at least that window;
  * engine WITH it but the DB not migrated yet → the sized INSERT fails; retry unsized, remember.
The suite runs against both engines; each test skips on the engine it does not apply to.
"""
from __future__ import annotations

import inspect

import pytest

from tests.conftest import NO_ENGINE
from vllm_lb.config import Config
from vllm_lb.leases import BrokerLeases

# Every mark below asks the engine which migrations it has: without it there is nothing to ask.
pytest.importorskip("queue_workflows", reason=NO_ENGINE)


def _engine_has_context() -> bool:
    from queue_workflows import node_queue
    return "context_length" in inspect.signature(node_queue.enqueue_node_job).parameters


needs_ctx = pytest.mark.skipif("not _engine_has_context()", reason="engine predates migration 0028")
needs_no_ctx = pytest.mark.skipif("_engine_has_context()", reason="engine has migration 0028")

#: The window a lease declares is the LEASE MODEL's catalog context (`Config.lease_context_length`): the classic
#: default row's `limits.max_model_len` — never an LB-wide number (LB_CONTEXT_LENGTH is retired, 2026-10-29).
CFG = Config()
WINDOW = 262144


def _row(job_id):
    from queue_workflows import node_queue
    return node_queue.get_node_job(job_id)


def _claim(label, serving_context="unset"):
    from queue_workflows import node_queue
    kw = {} if serving_context == "unset" else {"serving_context": serving_context}
    return node_queue.claim_next_gpu_job(0, None, host=label, served_types=["ollama", "vllm"],
                                         require_model=False, pool_modules=(), **kw)


def test_the_window_is_the_lease_models_catalog_context():
    """One place owns a model's context: its catalog row. The lease declares THAT row's window (the row whose
    served name the lease names), so the broker's 0028 gate and the LB's own routing read one figure."""
    from vllm_lb.catalog import CLASSIC, ModelCatalog
    row = ModelCatalog.load().default(CLASSIC)
    assert Config.from_env({"LB_ALLOWED_HOSTS": "loopback"}).lease_context_length == row.limits.max_model_len == WINDOW
    assert Config(lease_model=row.served_name).lease_context_length == WINDOW


def test_a_lease_model_no_catalog_row_serves_has_no_window_and_says_so():
    """No row, no window — never an unsized lease by default (which the 0028 gate would admit anywhere)."""
    with pytest.raises(ValueError, match=r"no-such:latest.*catalog"):
        Config(lease_model="no-such:latest").lease_context_length


def test_LB_CONTEXT_LENGTH_is_retired_naming_the_catalog():
    """Setting it — to anything, blank too — refuses the boot and names where the window lives now."""
    for value in ("131072", "0", ""):
        with pytest.raises(ValueError, match=r"LB_CONTEXT_LENGTH.*catalog") as e:
            Config.from_env({"LB_ALLOWED_HOSTS": "loopback", "LB_CONTEXT_LENGTH": value})
        assert "limits.max_model_len" in str(e.value)
    assert not hasattr(Config(), "context_length"), "the LB-wide window is gone, not merely unread"


@needs_no_ctx
def test_an_engine_without_0028_gets_an_unsized_lease_and_says_why(broker_db):
    leases = BrokerLeases(CFG)
    lease = leases.create(avoid_boxes=())
    assert lease is not None and "context_length" not in _row(lease.job_id)
    assert leases.context_state == "unsized: engine predates migration 0028"


@needs_ctx
def test_the_lease_declares_its_window_on_an_0028_engine(broker_db):
    leases = BrokerLeases(CFG)
    lease = leases.create(avoid_boxes=())
    assert _row(lease.job_id)["context_length"] == 262144
    assert leases.context_state == "sized: 262144"


@needs_ctx
def test_the_0028_gate_admits_the_lease_only_where_the_window_fits(broker_db):
    lease = BrokerLeases(CFG).create(avoid_boxes=())
    assert _claim("small-gpu", serving_context=131072) is None
    assert _claim("blind-gpu", serving_context=None) is None, \
        "a lane that cannot state its window never claims a sized row (rollout dependency)"
    assert _claim("box-b-gpu", serving_context=262144)["id"] == lease.job_id


@needs_ctx
def test_a_lease_model_without_a_catalog_row_is_refused_not_leased_unsized(broker_db):
    """`create` never raises (its verbs fence the DB): no lease, and `last_error` names the model and the field."""
    leases = BrokerLeases(Config(lease_model="no-such:latest"))
    assert leases.create(avoid_boxes=()) is None
    assert "no-such:latest" in leases.last_error and "limits.max_model_len" in leases.last_error


@needs_ctx
def test_engine_ahead_of_the_db_is_reported_as_a_migration_gap_not_a_mystery(broker_db):
    """An 0028 engine writes the column on EVERY enqueue (node_queue's INSERT column list),
    so against a pre-0028 database no lease is possible at all — sized or not. The LB cannot
    fix that; it must name it precisely (the operator applies the migration) and keep serving
    from the fallback, which it does because the proxy never waits on the broker."""
    from queue_workflows import db
    with db.connection() as conn, conn.cursor() as cur:
        cur.execute("ALTER TABLE workflow_node_jobs DROP COLUMN context_length")
        conn.commit()
    leases = BrokerLeases(CFG)
    assert leases.create(avoid_boxes=()) is None
    assert leases.db_ok is False
    assert "migration 0028" in leases.context_state and "apply" in leases.context_state
    assert leases.create(avoid_boxes=()) is None, "still honest on the next attempt"


@needs_ctx
def test_the_window_comes_back_the_moment_the_migration_lands(broker_db):
    """The twin of the 0030 recovery test — same inherited shape, found by the same audit.
    Remembering "the DB lacks 0028" across a successful enqueue would silently downgrade every
    later lease to unsized while the status line still blamed the migration."""
    from queue_workflows import db
    leases = BrokerLeases(CFG)
    with db.connection() as conn, conn.cursor() as cur:
        cur.execute("ALTER TABLE workflow_node_jobs DROP COLUMN context_length")
        conn.commit()
    assert leases.create(avoid_boxes=()) is None and "BLOCKED" in leases.context_state

    with db.connection() as conn, conn.cursor() as cur:
        cur.execute("ALTER TABLE workflow_node_jobs ADD COLUMN context_length INTEGER")
        conn.commit()
    lease = leases.create(avoid_boxes=())
    assert lease is not None and _row(lease.job_id)["context_length"] == 262144
    assert leases.context_state == "sized: 262144"
