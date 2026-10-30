"""S5 — the control loop: probe → refresh leases → pressure → plan → apply, never raising.

The broker side is the REAL adapter on a throwaway sqlite engine; "the engine" claiming a
lease is the same `claim_next_gpu_job` call a GB10 box's lane makes; probing is a table.
"""
from __future__ import annotations

import pytest

from tests.fakes import FakeProber
from vllm_lb import router
from vllm_lb.config import Config, VllmBox
from vllm_lb.control import Controller
from vllm_lb.hub import Hub
from vllm_lb.leases import BrokerLeases
from vllm_lb.state import LEASE_QUEUED, LEASE_RUNNING

def vllm_leases(hub):
    """The vLLM leases only (the fallback's lease is retired with the fallback, 2026-10-17)."""
    from vllm_lb.state import VLLM
    return [l for l in hub.leases() if l.kind == VLLM]


A, B = "http://10.0.0.1:8000", "http://10.0.0.2:8000"
CFG = Config(broker_enabled=True,
             vllm_boxes=(VllmBox("box-a-gpu", "box-a", A), VllmBox("box-b-gpu", "box-b", B)),
             idle_release_s=600, max_leases=2, scale_up_waiting=2,
             scale_up_sustain_s=30, probe_interval_s=0, lease_backoff_max_s=60)
LIVE_V = {"live": True, "served_models": ("qwen3.8:latest",), "max_model_len": 4096,
          "waiting": 0, "running": 0, "last_error": ""}


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture
def world(broker_db):
    clock = Clock()
    hub = Hub(CFG, clock=clock)
    leases = BrokerLeases(CFG, clock=clock)
    prober = FakeProber({A: {"live": False, "last_error": "refused"},
                         B: {"live": False, "last_error": "refused"}})
    ctl = Controller(CFG, hub, leases, prober=prober, clock=clock)
    return ctl, hub, leases, prober, clock


def _claim_as(label):
    from tests.fakes import claim_gpu
    return claim_gpu(label)


def _pending():
    from queue_workflows import node_queue
    return node_queue.count_pending_for_server(None, "vllm", any_model=True)


def _run_status(run_id):
    from queue_workflows import run_store
    return run_store.get_run(run_id)["status"]


def test_no_request_means_no_lease(world):
    ctl, hub, *_ = world
    ctl.tick()
    assert vllm_leases(hub) == [] and _pending() == 0


def test_first_request_creates_one_lease_on_the_next_tick(world):
    ctl, hub, *_ = world
    hub.note_request()
    ctl.tick()
    (lease,) = vllm_leases(hub)
    assert lease.status == LEASE_QUEUED and _pending() == 1
    ctl.tick()
    assert len(vllm_leases(hub)) == 1, "no duplicate while the first is still queued"


def test_probes_land_in_the_hub_every_tick(world):
    ctl, hub, _, prober, _ = world
    ctl.tick()
    assert hub.backend_for(A).live is False and hub.backend_for(B).live is False
    assert hub.backend_for(A).last_error == "refused"
    assert set(prober.calls) == {A, B}


def test_claim_is_learned_and_routing_prefers_the_leased_vllm(world):
    ctl, hub, _, prober, _ = world
    hub.note_request()
    ctl.tick()
    _claim_as("box-a-gpu")
    prober.table[A] = LIVE_V
    ctl.tick()
    (lease,) = vllm_leases(hub)
    assert lease.status == LEASE_RUNNING and lease.box == "box-a"
    assert router.choose(hub.snapshot(), CFG).url == A


def test_idle_releases_the_lease_and_cancels_its_run(world):
    ctl, hub, _, prober, clock = world
    hub.note_request()
    ctl.tick()
    _claim_as("box-a-gpu")
    prober.table[A] = LIVE_V
    ctl.tick()
    (lease,) = vllm_leases(hub)
    clock.t += 601
    ctl.tick()
    assert vllm_leases(hub) == []
    assert _run_status(lease.lease_id) == "cancelled"


def test_sustained_pressure_adds_a_second_lease_with_no_box_constraint(world):
    from queue_workflows import node_queue
    ctl, hub, _, prober, clock = world
    hub.note_request()
    ctl.tick()
    _claim_as("box-a-gpu")
    prober.table[A] = {**LIVE_V, "waiting": 5}
    ctl.tick()                      # pressure starts
    assert len(vllm_leases(hub)) == 1, "not before it is sustained"
    clock.t += 30
    hub.note_request()
    ctl.tick()                      # sustained ⇒ second lease
    leases = sorted(vllm_leases(hub), key=lambda l: l.created_at)
    assert len(leases) == 2 and leases[1].status == LEASE_QUEUED
    # Pass 16: avoid_box returns, but only while another candidate box exists. In this
    # fixture the prober knows two boxes, so the 2nd lease steers off the one we hold.
    assert node_queue.get_node_job(leases[1].job_id)["avoid_box"] in (None, ["box-a"]), \
        "either constraint is correct here; what must never happen is excluding the LAST box"


def test_pressure_gone_releases_the_unclaimed_lease_first(world):
    ctl, hub, _, prober, clock = world
    hub.note_request()
    ctl.tick()
    _claim_as("box-a-gpu")
    prober.table[A] = {**LIVE_V, "waiting": 5}
    ctl.tick()
    clock.t += 30
    hub.note_request()
    ctl.tick()
    assert len(vllm_leases(hub)) == 2
    prober.table[A] = LIVE_V
    clock.t += 601                  # pressure memory expires …
    hub.note_request()              # … but demand persists
    ctl.tick()
    (left,) = vllm_leases(hub)
    assert left.status == LEASE_RUNNING and left.box == "box-a"


def test_a_lost_lease_is_replaced_while_demand_persists(world):
    from queue_workflows import db
    ctl, hub, *_ = world
    hub.note_request()
    ctl.tick()
    (lease,) = vllm_leases(hub)
    with db.connection() as conn, conn.cursor() as cur:
        cur.execute("UPDATE workflow_node_jobs SET status = 'failed' WHERE id = %(id)s",
                    {"id": lease.job_id})
        conn.commit()
    ctl.tick()
    fresh = vllm_leases(hub)
    assert len(fresh) == 1 and fresh[0].lease_id != lease.lease_id
    assert _run_status(lease.lease_id) == "cancelled", "the lost run is closed, not left open"


def test_create_failure_backs_off_and_recovers(world, monkeypatch):
    import queue_workflows as qw
    from queue_workflows import db
    ctl, hub, leases, _, clock = world
    hub.note_request()
    db.close_pool()
    monkeypatch.setenv("LB_TEST_DB_DOWN", "/nonexistent/dir/broker.db")
    qw.configure(db_backend="sqlite", db_url_env="LB_TEST_DB_DOWN")
    ctl.tick()
    assert vllm_leases(hub) == [] and hub.snapshot().db_ok is False
    assert ctl.backoff_until > clock.t
    qw.configure(db_backend="sqlite", db_url_env="LB_TEST_DB")
    ctl.tick()
    assert vllm_leases(hub) == [], "still backing off"
    clock.t = ctl.backoff_until + 1
    ctl.tick()
    assert len(vllm_leases(hub)) == 1 and hub.snapshot().db_ok is True


def test_a_raising_prober_does_not_kill_the_tick(world):
    ctl, hub, _, prober, _ = world
    prober.table[A] = RuntimeError("boom")
    hub.note_request()
    ctl.tick()
    assert hub.backend_for(A).live is False and "boom" in hub.backend_for(A).last_error
    assert hub.backend_for(B).last_error == "refused", "the other server is still probed"
    assert len(vllm_leases(hub)) == 1
    assert hub.snapshot().extra["last_tick_error"] == ""


def test_engine_setup_is_retried_each_tick_and_broker_phases_wait_for_it(world):
    ctl, hub, leases, prober, _ = world
    answers = ["ModuleNotFoundError: queue_workflows", "ModuleNotFoundError: queue_workflows", ""]
    ctl.ensure_engine = lambda: answers.pop(0)
    ctl.engine_ready = False
    made = leases.create(avoid_boxes=())        # an open lease from a previous life
    hub.note_request()
    ctl.tick()
    assert hub.backend_for(B).last_error == "refused", "probing does not wait for the engine"
    assert vllm_leases(hub) == [] and hub.snapshot().db_ok is False
    assert "queue_workflows" in hub.snapshot().db_error
    assert hub.snapshot().extra["last_tick_error"] == ""
    ctl.tick()
    assert vllm_leases(hub) == []
    ctl.tick()                                   # the engine appears: recover, then plan
    assert ctl.engine_ready is True
    assert [l.lease_id for l in vllm_leases(hub)] == [made.lease_id]
    assert hub.snapshot().db_ok is True


def test_recover_adopts_open_leases_before_the_first_tick(world):
    ctl, hub, leases, _, _ = world
    made = leases.create(avoid_boxes=())
    _claim_as("box-b-gpu")
    ctl.recover()
    (lease,) = vllm_leases(hub)
    assert lease.lease_id == made.lease_id and lease.box == "box-b"
