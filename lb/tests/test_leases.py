"""S3 — the lease adapter speaks broker_parrot's EXISTING job API, and only that.

A lease is one `workflow_runs` row we own (status running, mode node) plus one gpu node-job
with `server_types=['vllm']` and NO required_model (pool lane, no catalog coupling). Release =
cancel the run (the body's `cancel_event`) and, while still queued, the job itself. Every DB
failure is reported through `db_ok` / `last_error`, never raised into the daemon.
"""
from __future__ import annotations

import pytest

from vllm_lb.config import Config, VllmBox
from vllm_lb.leases import BrokerLeases
from vllm_lb.state import LEASE_LOST, LEASE_QUEUED, LEASE_RUNNING

CFG = Config(lease_max_hold_s=1234,
             vllm_boxes=(VllmBox("box-a-gpu", "box-a", "http://192.0.2.10:8000"),
                         VllmBox("box-b-gpu", "box-b", "http://192.0.2.11:8000")),
             lease_workflow="vllm_lb_lease",
             lease_node_module="system.vllm_lease")


class Clock:
    def __init__(self, t=100.0):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture
def leases(broker_db):
    return BrokerLeases(CFG, clock=Clock())


def _claim_as(host_label: str):
    """What a GB10 box's gpu lane does: claim the row as a box whose vLLM answers (and, on an
    0028 engine, states the context window it serves)."""
    from tests.fakes import claim_gpu
    return claim_gpu(host_label)


def _job(job_id):
    from queue_workflows import node_queue
    return node_queue.get_node_job(job_id)


def _run(run_id):
    from queue_workflows import run_store
    return run_store.get_run(run_id)


def test_create_writes_exactly_the_row_the_inquiry_proved(leases):
    lease = leases.create(avoid_boxes=())
    assert lease is not None and lease.status == LEASE_QUEUED and leases.db_ok
    job, run = _job(lease.job_id), _run(lease.lease_id)
    assert run["workflow_name"] == "vllm_lb_lease" and run["status"] == "running"
    assert run["mode"] == "node" and run["out_dir"] is None
    assert job["queue"] == "gpu" and job["status"] == "queued"
    assert job["server_types"] == ["vllm"] and job["required_model"] is None
    assert job["node_module"] == "system.vllm_lease"
    assert job["priority"] == 0, "operator: highest priority in the broker"
    assert job["avoid_box"] is None
    assert job["inputs"]["lease_id"] == lease.lease_id
    assert job["inputs"]["max_hold_s"] == 1234
    assert lease.created_at == 100.0


def test_create_is_the_demand_signal_the_engine_starts_vllm_on(leases):
    from queue_workflows import node_queue
    leases.create(avoid_boxes=())
    assert node_queue.count_pending_for_server(None, "vllm", any_model=True) == 1


def test_create_with_avoid_boxes_pins_the_placement(leases):
    lease = leases.create(avoid_boxes=("box-a",))
    assert _job(lease.job_id)["avoid_box"] == ["box-a"]


def test_refresh_after_a_claim_learns_the_box_from_claimed_by(leases):
    lease = leases.create(avoid_boxes=())
    assert _claim_as("box-a-gpu")["id"] == lease.job_id
    leases.clock.t = 160.0
    (updated,) = leases.refresh([lease])
    assert updated.status == LEASE_RUNNING
    assert updated.host_label == "box-a-gpu" and updated.box == "box-a"
    assert updated.claimed_at == 160.0


def test_a_claim_by_an_unknown_label_is_running_but_unmapped(leases):
    lease = leases.create(avoid_boxes=())
    _claim_as("mystery-gpu")
    (updated,) = leases.refresh([lease])
    assert updated.status == LEASE_RUNNING and updated.host_label == "mystery-gpu"
    assert updated.box is None and "not in LB_VLLM_BOXES" in updated.detail


@pytest.mark.parametrize("terminal", ["failed", "cancelled", "completed"])
def test_a_row_that_ended_without_us_is_lost(leases, terminal):
    from queue_workflows import db
    lease = leases.create(avoid_boxes=())
    with db.connection() as conn, conn.cursor() as cur:
        cur.execute("UPDATE workflow_node_jobs SET status = %(s)s WHERE id = %(id)s",
                    {"s": terminal, "id": lease.job_id})
        conn.commit()
    (updated,) = leases.refresh([lease])
    assert updated.status == LEASE_LOST and terminal in updated.detail


def test_a_vanished_row_is_lost(leases):
    from queue_workflows import db
    lease = leases.create(avoid_boxes=())
    with db.connection() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM workflow_node_jobs WHERE id = %(id)s", {"id": lease.job_id})
        conn.commit()
    (updated,) = leases.refresh([lease])
    assert updated.status == LEASE_LOST


def test_release_of_a_queued_lease_cancels_job_and_run(leases):
    lease = leases.create(avoid_boxes=())
    assert leases.release(lease) is True
    assert _job(lease.job_id)["status"] == "cancelled"
    assert _run(lease.lease_id)["status"] == "cancelled"
    assert _run(lease.lease_id)["finished_at"] is not None


def test_release_of_a_running_lease_cancels_the_run_and_leaves_the_body_to_return(leases):
    from queue_workflows import node_queue
    lease = leases.create(avoid_boxes=())
    _claim_as("box-a-gpu")
    (lease,) = leases.refresh([lease])
    assert leases.release(lease) is True
    assert _run(lease.lease_id)["status"] == "cancelled"
    assert _job(lease.job_id)["status"] == "running", "the body sees cancel_event and returns"
    assert node_queue.count_pending_for_server(None, "vllm", any_model=True) == 0


def test_recover_adopts_our_open_leases_and_only_ours(leases, broker_db):
    from queue_workflows import node_queue, run_store
    a = leases.create(avoid_boxes=())
    b = leases.create(avoid_boxes=("box-a",))
    _claim_as("box-b-gpu")            # claims a (oldest first)
    # somebody else's run + job must never be adopted
    run_store.insert_run(run_id="other", workflow_name="llm_chat", status="running", mode="node")
    node_queue.enqueue_node_job(run_id="other", node_id="n", node_module="llm_chat",
                                queue="gpu", server_types=["vllm"])
    fresh = BrokerLeases(CFG, clock=Clock(500.0))
    got = {l.lease_id: l for l in fresh.recover()}
    assert set(got) == {a.lease_id, b.lease_id}
    assert got[a.lease_id].status == LEASE_RUNNING and got[a.lease_id].box == "box-b"
    assert got[b.lease_id].status == LEASE_QUEUED


def test_recover_closes_a_run_whose_job_already_ended(leases):
    from queue_workflows import db
    a = leases.create(avoid_boxes=())
    with db.connection() as conn, conn.cursor() as cur:
        cur.execute("UPDATE workflow_node_jobs SET status = 'completed' WHERE id = %(id)s",
                    {"id": a.job_id})
        conn.commit()
    assert BrokerLeases(CFG, clock=Clock()).recover() == []
    assert _run(a.lease_id)["status"] == "cancelled"


def test_db_failure_is_reported_never_raised(leases, monkeypatch):
    """A REAL failure, not a mocked one: the engine is re-pointed at a database file that
    cannot be opened, exactly what a broker outage looks like from the LB's seat."""
    import queue_workflows as qw
    from queue_workflows import db

    lease = leases.create(avoid_boxes=())
    db.close_pool()
    monkeypatch.setenv("LB_TEST_DB_DOWN", "/nonexistent/dir/broker.db")
    qw.configure(db_backend="sqlite", db_url_env="LB_TEST_DB_DOWN")

    assert leases.create(avoid_boxes=()) is None
    assert leases.db_ok is False and leases.last_error.startswith("create:")
    assert leases.refresh([lease]) == [lease], "unchanged when unreadable"
    assert leases.release(lease) is False
    assert leases.recover() == []
