"""A lease is a REQUEST for a vLLM, so the cap on leases cannot require one to already exist.

Live, after the pass-15 restart — and reproduced by the operator's own `vllmstart`:

    vllm   http://192.0.2.2:8000   down — ConnectionRefusedError
    vllm   http://192.0.2.12:8000   down — ConnectionRefusedError
    ollama http://host.docker.internal:11434   LIVE
    desired leases: 0

    no vLLM running -> no worker advertises vllm -> capacity 0 -> desired 0
                    -> no lease requested -> no vLLM started -> (loop)

"Which boxes serve vLLM right now" is a ROUTING fact. "Which boxes could TAKE a vLLM lease" is
the CAPACITY fact, and only the second bounds how many leases to ask for.
"""
from __future__ import annotations

import pytest

from vllm_lb.config import Config, parse_boxes
from vllm_lb.leases import BrokerLeases
from vllm_lb.scaler import Create, plan_actions

CFGL = Config(lease_workflow="vllm_lb_lease", lease_node_module="system.vllm_lease",
             vllm_boxes=parse_boxes("box-a-gpu=box-a=http://192.0.2.10:8000,"
                                   "box-b-gpu=box-b=http://192.0.2.11:8000"))


@pytest.fixture
def leases(broker_db):
    return BrokerLeases(CFGL, clock=lambda: 100.0)


def test_a_fleet_with_NO_vllm_running_can_still_be_asked_for_one(leases, broker_db):
    """THE DEADLOCK. Both boxes advertise ollama only, because their sidecars are stopped —
    which is precisely when a lease is needed."""
    from queue_workflows import node_queue
    for label in ("box-a-gpu", "box-b-gpu"):
        node_queue.upsert_worker_heartbeat(host_label=label, queue="gpu", concurrency=2,
                                           llm_servers_available=["ollama"])
    n, detail = leases.vllm_capacity()
    assert n == 2, f"both boxes COULD take a lease ({detail})"


def test_a_parked_box_cannot_take_a_lease(leases, broker_db):
    from queue_workflows import node_queue, worker_control
    for label in ("box-a-gpu", "box-b-gpu"):
        node_queue.upsert_worker_heartbeat(host_label=label, queue="gpu", concurrency=2,
                                           llm_servers_available=["ollama"])
    worker_control.set_worker_control(host_label="box-b-gpu", queue="gpu", desired_state="off")
    n, detail = leases.vllm_capacity()
    assert n == 1 and "box-a-gpu" in detail, detail


def test_a_box_not_in_LB_VLLM_BOXES_still_does_not_count(leases, broker_db):
    from queue_workflows import node_queue
    node_queue.upsert_worker_heartbeat(host_label="box-c-gpu", queue="gpu", concurrency=2,
                                       llm_servers_available=["vllm"])
    n, _ = leases.vllm_capacity()
    assert n == 0, "the LB has no URL for it, so it can hold a lease but never serve"


# ── the operator's 'exclude box' ───────────────────────────────────────────────

def test_a_second_lease_avoids_the_box_we_already_hold_when_another_exists():
    """box-b's dashboard showed TWO system.vllm_lease jobs on one box. A gpu lane is two wide,
    so they take both lanes and buy nothing."""
    from tests.test_scaler import CFG, NOW, lease, snap
    s = snap(last_request_at=NOW, last_pressure_at=NOW, vllm_capacity=2,
             leases=[lease(1, box="box-b", host_label="box-b-gpu")])
    assert plan_actions(s, CFG) == [Create(avoid_boxes=("box-b",))]


def test_it_sends_NO_avoid_box_when_the_held_box_is_the_only_candidate():
    """The KILL line, and pass 11's whole lesson: a filter that excludes the last candidate
    strands the row. With capacity 1 there is nowhere else, so nothing is excluded."""
    from tests.test_scaler import CFG, NOW, lease, snap
    s = snap(last_request_at=NOW, last_pressure_at=NOW, vllm_capacity=1,
             leases=[lease(1, box="box-a", host_label="box-a-gpu")])
    for a in plan_actions(s, CFG):
        if isinstance(a, Create):
            assert a.avoid_boxes == (), "never exclude the last candidate"
