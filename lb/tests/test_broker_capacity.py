"""The lease cap comes from the BROKER, not from our config file.

`desired_leases` used to cap at `len(LB_VLLM_BOXES)` — a static guess. Asked live on
2026-09-12 the fleet disagreed:

    box-c-gpu  llm_servers_available=['ollama']
    box-a-gpu    llm_servers_available=['vllm']      <- the only one
    box-b-gpu   llm_servers_available=['ollama']    <- its `vllm` sidecar Exited(0)

The LB believed two, so under pressure it asked for a second lease nothing could claim — the
row that sat queued 38 minutes wearing a run-next flag (worklog/vllm-lb-never-avoid.md).

`worker_heartbeats.llm_servers_available` is the engine's OWN source of truth: the same column
its `unassignable_reason = 'no live worker serves any of: …'` sweep reads.
"""
from __future__ import annotations

import pytest

from vllm_lb.config import Config, parse_boxes
from vllm_lb.leases import BrokerLeases
from vllm_lb.scaler import desired_leases

from tests.test_scaler import CFG, NOW, snap

CFGL = Config(lease_workflow="vllm_lb_lease", lease_node_module="system.vllm_lease",
             vllm_boxes=parse_boxes("box-a-gpu=box-a=http://192.0.2.10:8000,"
                                   "box-b-gpu=box-b=http://192.0.2.11:8000"))


class Clock:
    def __init__(self, t=100.0):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture
def leases(broker_db):
    return BrokerLeases(CFGL, clock=Clock())


def test_capacity_counts_boxes_that_could_TAKE_a_lease(leases, broker_db):
    """REVERSED 2026-09-12 by pass 16 (worklog/vllm-lb-capacity-deadlock.md). This used to
    count only workers already ADVERTISING vllm. That deadlocks: a box advertises vllm only
    once its sidecar runs, and the sidecar runs only once a lease is claimed, so a fleet with
    every vLLM stopped reported capacity 0 and never asked for one. Observed live."""
    from queue_workflows import node_queue
    node_queue.upsert_worker_heartbeat(host_label="box-a-gpu", queue="gpu",
                                       concurrency=1, llm_servers_available=["vllm"])
    node_queue.upsert_worker_heartbeat(host_label="box-b-gpu", queue="gpu",
                                       concurrency=1, llm_servers_available=["ollama"])
    node_queue.upsert_worker_heartbeat(host_label="box-c-gpu", queue="gpu",
                                       concurrency=1, llm_servers_available=["ollama"])
    n, detail = leases.vllm_capacity()
    assert n == 2, f"both configured boxes could take a lease; got {n} ({detail})"
    assert "box-a-gpu" in detail and "box-b-gpu" in detail


def test_a_fleet_whose_only_live_worker_is_unroutable_caps_at_zero(leases, broker_db):
    """Still zero, but for the right reason: not "serves no vllm" (pass 16) — "is not a box the
    LB has a URL for"."""
    from queue_workflows import node_queue
    node_queue.upsert_worker_heartbeat(host_label="box-c-gpu", queue="gpu",
                                       concurrency=1, llm_servers_available=["ollama"])
    n, _ = leases.vllm_capacity()
    assert n == 0, "box-c-gpu is not in LB_VLLM_BOXES"


def test_an_unreadable_control_plane_fails_OPEN(leases, monkeypatch):
    """A DB blip must not stop leasing: fall back to the configured box count."""
    import queue_workflows.db as db
    # Imported BEFORE the patch: node_queue binds `from queue_workflows.db import connection` at import, so a first
    # import under the patch (vllm_capacity's lazy one) kept `boom` for every later test in this worker — "db down"
    # in whichever broker test xdist scheduled next, reproducible on 759860c by running this test then any other.
    import queue_workflows.node_queue  # noqa: F401

    def boom(*a, **k):
        raise RuntimeError("db down")

    monkeypatch.setattr(db, "connection", boom)
    n, detail = leases.vllm_capacity()
    assert n == len(CFGL.vllm_boxes), "fail open to today's behaviour"
    assert "unknown" in detail.lower()


def test_the_scaler_never_wants_more_leases_than_the_broker_can_serve():
    s = snap(last_request_at=NOW, last_pressure_at=NOW, vllm_capacity=1)
    assert desired_leases(s, CFG) <= 1, "one vLLM box ⇒ at most one lease"


def test_capacity_zero_means_no_lease_is_requested_at_all():
    s = snap(last_request_at=NOW, last_pressure_at=NOW, vllm_capacity=0)
    assert desired_leases(s, CFG) == 0


def test_no_live_gpu_worker_at_all_is_UNKNOWN_not_zero(leases, broker_db):
    """The distinction the suite forced into the design: an empty heartbeat table means "we
    cannot tell", not "nothing serves vllm". Capping at 0 there would stop the fleet leasing
    whenever heartbeats were briefly absent. The engine draws the same line — its unassignable
    sweep only says "no live worker serves …" when at least one live gpu worker EXISTS."""
    n, detail = leases.vllm_capacity()
    assert n == len(CFGL.vllm_boxes), "fall back to the configured count"
    assert "unknown" in detail.lower() and "heartbeat" in detail.lower()


def test_a_worker_defaults_to_ollama_so_it_does_NOT_count_as_vllm_capable(leases, broker_db):
    """Reality check, from the engine: `upsert_worker_heartbeat` DEFAULTS
    llm_servers_available to ['ollama'] when the caller does not pass it — an
    "un-advertised" gpu worker does not exist in practice. So a box that never announced a
    vLLM is correctly NOT counted, which is exactly why the cap reads 1 on this fleet while
    LB_VLLM_BOXES claims 2."""
    from queue_workflows import node_queue
    node_queue.upsert_worker_heartbeat(host_label="fresh-gpu", queue="gpu", concurrency=1)
    n, detail = leases.vllm_capacity()
    assert n == 0, f"defaulted to ollama, so it cannot serve a vLLM lease ({detail})"


def test_an_explicitly_empty_server_list_counts_as_able(leases, broker_db):
    """The engine's own claim path allows a worker with an EMPTY list so a cold worker cannot
    wedge the queue (`lg_empty` in its unassignable sweep). The cap makes the same allowance."""
    from queue_workflows import node_queue
    # A ROUTABLE box (in LB_VLLM_BOXES): since pass 14 the cap also intersects with the boxes
    # the LB has a URL for, because one it cannot dial can hold a lease but never serve it.
    node_queue.upsert_worker_heartbeat(host_label="box-a-gpu", queue="gpu", concurrency=1,
                                       llm_servers_available=[])
    n, detail = leases.vllm_capacity()
    assert n >= 1, f"an empty list is 'not announced yet', not 'no' ({detail})"
