"""`priority=0` is the BAND. `is_priority` is "run next".

The GPU claim sorts `is_priority DESC, affinity, warm_affinity, priority ASC, host_dir`
(queue_workflows/node_queue.py:1044), so two things outrank the band we set — and the engine's
own docstring says a warm-model job "preempts a colder one even from a worse priority band",
with "only the operator 'run next' is_priority flag" outranking affinity.

That is live on this fleet: both GB10 boxes hold qwen3.8:latest resident in broker-ollama, so every
queued qwen3.8 job has warm affinity and sorts ahead of a priority=0 lease.

The flag is an OPTIMISATION, never a precondition: a lease that could not be flagged is still a
lease, because the alternative is a daemon that cannot get a backend at all.
"""
from __future__ import annotations

import pytest

from vllm_lb.config import Config
from vllm_lb.leases import BrokerLeases
from vllm_lb.state import LEASE_QUEUED

CFG = Config(lease_max_hold_s=1234, lease_workflow="vllm_lb_lease",
             lease_node_module="system.vllm_lease")


class Clock:
    def __init__(self, t=100.0):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture
def leases(broker_db):
    return BrokerLeases(CFG, clock=Clock())


def _job(job_id):
    from queue_workflows import node_queue
    return node_queue.get_node_job(job_id)


def test_a_vllm_lease_is_flagged_run_next(leases):
    lease = leases.create(avoid_boxes=())
    assert lease is not None
    job = _job(lease.job_id)
    assert job["priority"] == 0, "still the top band"
    assert job["is_priority"] is True, "and ahead of warm-model affinity — 'run next'"


def test_a_lease_survives_an_engine_with_no_run_next_flag(leases, monkeypatch):
    """Pre-0016 engines have no prioritize_node_job. Unflagged is worse than flagged; it is
    very much better than no lease."""
    from queue_workflows import node_queue
    monkeypatch.delattr(node_queue, "prioritize_node_job", raising=False)
    lease = leases.create(avoid_boxes=())
    assert lease is not None and lease.status == LEASE_QUEUED and leases.db_ok


def test_a_lease_survives_the_flag_raising(leases, monkeypatch):
    from queue_workflows import node_queue

    def boom(job_id):
        raise RuntimeError("deadlock detected")

    monkeypatch.setattr(node_queue, "prioritize_node_job", boom)
    lease = leases.create(avoid_boxes=())
    assert lease is not None and lease.status == LEASE_QUEUED
    assert leases.db_ok, "a failed OPTIMISATION must not mark the whole adapter unhealthy"


def test_a_flag_that_did_not_take_is_not_an_error(leases, monkeypatch):
    """`prioritize_node_job` returns None when the row was not queued — overwhelmingly because
    a worker claimed it in the microseconds since the INSERT, which is the outcome we wanted."""
    from queue_workflows import node_queue
    monkeypatch.setattr(node_queue, "prioritize_node_job", lambda job_id: None)
    lease = leases.create(avoid_boxes=())
    assert lease is not None and leases.db_ok
    assert "run next" in leases.run_next_state.lower() or leases.run_next_state
