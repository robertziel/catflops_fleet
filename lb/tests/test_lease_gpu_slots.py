"""The lease takes the WHOLE box (broker_parrot migration 0030, `gpu_slots='all'`).

Measured 2026-09-12: the fleet co-scheduled `wait=4 run=1` of its own `llm_chat` into the leased
box-b vLLM while the gateway's turns queued behind it. A lease weighed 1 against a PAR-2 box,
so the box-slot arbiter left a slot open. The operator: *"implement in our broker_parrot
integration node_job to occupy ALL."*

Three engine/DB states exist during a rollout — the same three `context_length` (0028) has —
and the lease must be created in all three:
  * engine WITHOUT `enqueue_node_job(gpu_slots=)` → an unweighted lease, status says why;
  * engine WITH it and the column present → the row says `all`, and NOTHING else is claimable
    on that box while we hold it;
  * engine WITH it but the DB not migrated → the INSERT fails; name the gap, keep serving.
"""
from __future__ import annotations

import http.client
import inspect
import json
import uuid

import pytest

from tests.conftest import NO_ENGINE
from vllm_lb.config import Config
from vllm_lb.leases import BrokerLeases

# Every mark below asks the engine which migrations it has: without it there is nothing to ask.
pytest.importorskip("queue_workflows", reason=NO_ENGINE)


def _engine_has_slots() -> bool:
    from queue_workflows import node_queue
    return "gpu_slots" in inspect.signature(node_queue.enqueue_node_job).parameters


needs_slots = pytest.mark.skipif("not _engine_has_slots()", reason="engine predates migration 0030")
needs_no_slots = pytest.mark.skipif("_engine_has_slots()", reason="engine has migration 0030")


def _row(job_id):
    from queue_workflows import node_queue
    return node_queue.get_node_job(job_id)


def _claim(label, box, par=2):
    """Claim as a GB10 lane does: its own PAR as `max_running` (the box-slot arbiter's budget),
    `kind_caps={}` so only THAT arbiter decides here, and the window it serves — the lease is
    SIZED (0028), and a lane that cannot state its window never claims a sized row."""
    from queue_workflows import node_queue
    kw = {}
    if "serving_context" in inspect.signature(node_queue.claim_next_gpu_job).parameters:
        kw["serving_context"] = 262144
    return node_queue.claim_next_gpu_job(0, None, host=label, box=box, max_running=par,
                                         kind_caps={}, served_types=["ollama", "vllm"],
                                         require_model=False, pool_modules=(), **kw)


def _ordinary_gpu_job():
    """What the fleet enqueues all day: an `llm_chat`-shaped row that weighs 1."""
    from queue_workflows import node_queue, run_store
    run_id = str(uuid.uuid4())
    run_store.insert_run(run_id=run_id, workflow_name="fleet", out_dir=None, status="running",
                         mode="node")
    return node_queue.enqueue_node_job(run_id=run_id, node_id="chat", node_module="x",
                                       queue="gpu", priority=100)


# ── the row says it takes the whole box ────────────────────────────────────────

@needs_slots
def test_the_vllm_lease_takes_the_whole_box(broker_db):
    leases = BrokerLeases(Config())
    lease = leases.create(avoid_boxes=())
    assert _row(lease.job_id)["gpu_slots"] == "all"
    assert leases.gpu_slots_state == "all: the lease holds the whole box"


# ── and the arbiter honours it: nothing else fits beside our lease ─────────────

@needs_slots
def test_nothing_else_is_claimable_on_the_box_our_lease_holds(broker_db):
    """THE POINT. A PAR-2 GB10 box claims our lease; the fleet's next ordinary row must NOT land
    there while we hold it — and must still land on the other GB10 box (the weight is per box)."""
    leases = BrokerLeases(Config())
    lease = leases.create(avoid_boxes=())
    got = _claim("box-a-gpu", box="box-a")
    assert got is not None and got["id"] == lease.job_id
    chat = _ordinary_gpu_job()
    assert _claim("box-a-gpu", box="box-a") is None, \
        "the leased box has no slot left: 'all' weighs the whole budget"
    other = _claim("box-b-gpu", box="box-b")
    assert other is not None and other["id"] == chat


@needs_slots
def test_our_lease_waits_behind_fleet_work_instead_of_squeezing_in(broker_db):
    """The other direction of the same rule, and the KILL line of the engine frame: a box
    already running one ordinary job cannot take an `all` row — the lease WAITS (it is not
    stranded: it claims the moment the box drains)."""
    from queue_workflows import node_queue
    chat = _ordinary_gpu_job()
    assert _claim("box-a-gpu", box="box-a")["id"] == chat
    lease = BrokerLeases(Config()).create(avoid_boxes=())
    assert _claim("box-a-gpu", box="box-a") is None, "PAR 2: one running + a whole-box row > 2"
    node_queue.mark_completed(chat, context_delta={}, seconds=1.0)
    assert _claim("box-a-gpu", box="box-a")["id"] == lease.job_id


# ── rollout: the other two states ──────────────────────────────────────────────

@needs_no_slots
def test_an_engine_without_0030_still_gets_a_lease_and_says_why(broker_db):
    leases = BrokerLeases(Config())
    lease = leases.create(avoid_boxes=())
    assert lease is not None and "gpu_slots" not in _row(lease.job_id)
    assert leases.gpu_slots_state == "unweighted: engine predates migration 0030"


@needs_slots
def test_engine_ahead_of_the_db_is_reported_as_a_migration_gap_not_a_mystery(broker_db):
    """A 0030 engine writes the column on EVERY enqueue, so against a pre-0030 database no
    lease is possible at all. Name it precisely; the orchestrator's next boot applies it."""
    from queue_workflows import db
    with db.connection() as conn, conn.cursor() as cur:
        cur.execute("ALTER TABLE workflow_node_jobs DROP COLUMN gpu_slots")
        conn.commit()
    leases = BrokerLeases(Config())
    assert leases.create(avoid_boxes=()) is None
    assert leases.db_ok is False
    assert "migration 0030" in leases.gpu_slots_state and "apply" in leases.gpu_slots_state


@needs_slots
def test_the_weight_comes_back_the_moment_the_migration_lands(broker_db):
    """The "DB lacks 0030" memory must not outlive the gap it describes.

    Independent audit, 2026-09-12 (finding 1): the flag was set once and never cleared, so the
    FIRST lease after the operator applied the migration was created UNWEIGHTED — the exact bug
    this feature exists to fix, silently, while `/lb/status` still said BLOCKED. There is nothing
    to remember: the 0030 engine names `gpu_slots` in every INSERT, so a rejected enqueue re-states
    the gap and an accepted one PROVES the column is there."""
    from queue_workflows import db
    leases = BrokerLeases(Config())
    with db.connection() as conn, conn.cursor() as cur:
        cur.execute("ALTER TABLE workflow_node_jobs DROP COLUMN gpu_slots")
        conn.commit()
    assert leases.create(avoid_boxes=()) is None and "BLOCKED" in leases.gpu_slots_state

    with db.connection() as conn, conn.cursor() as cur:          # the operator applies it
        cur.execute("ALTER TABLE workflow_node_jobs ADD COLUMN gpu_slots text")
        conn.commit()
    lease = leases.create(avoid_boxes=())
    assert lease is not None, "the migration landed — a lease is possible again"
    assert _row(lease.job_id)["gpu_slots"] == "all", "and it must WEIGH again, not just exist"
    assert leases.gpu_slots_state == "all: the lease holds the whole box"


# ── it is visible ──────────────────────────────────────────────────────────────

def test_lb_status_shows_the_weight_state():
    from vllm_lb.hub import Hub
    from vllm_lb.proxy import Proxy
    cfg = Config(broker_enabled=True, listen_host="127.0.0.1", listen_port=0)
    hub = Hub(cfg)
    hub.set_gpu_slots_state("all: the lease holds the whole box")
    proxy = Proxy(cfg, hub)
    proxy.start()
    try:
        c = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=10)
        c.request("GET", "/lb/status")
        body = json.loads(c.getresponse().read())
    finally:
        proxy.stop()
    assert body["gpu_slots"] == "all: the lease holds the whole box"
