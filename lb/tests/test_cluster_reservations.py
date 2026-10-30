"""The cluster's reservation of a GB10 box is ONE broker_parrot node_job lease — the contract, on the real engine.

The sqlite engine is the same `node_queue` the GB10 boxes' lanes claim from, so a green test here is the
row a lane will see. The contract (provisional, 2026-09-15; worklog/vllm-gateway-program.md pass 2) is
made in ONE place, `BrokerLeases._cluster_reservation_kwargs`. The LB reads parks and never writes one.
"""
from __future__ import annotations

import inspect

import pytest

from tests.fakes import claim_gpu
from vllm_lb.cluster import Reservation
from vllm_lb.config import Config, VllmBox
from vllm_lb.leases import BrokerLeases, cluster_reservation_id
from vllm_lb.state import CLUSTER, Lease

CFG = Config(vllm_boxes=(VllmBox("box-a-gpu", "box-a", "http://box-a:8000"),
                         VllmBox("box-b-gpu", "box-b", "http://box-b:8000")),
             lease_workflow="vllm_lb_lease", lease_node_module="system.vllm_lease")
BOXES = ("box-a", "box-b")
LABELS = ("box-a-gpu", "box-b-gpu")
START = "0f5e2c1d9b8a4e7f"
#: What the generation's ModelChoice serves — the catalog's cluster default's served name.
MODEL = "Qwen/Qwen3.8-Flash-Next"


def claim(label: str, box: str, served=("cluster",)):
    """What a GB10 box's gpu lane does once it serves the `cluster` kind: claim, naming its physical box."""
    return claim_gpu(label, box=box, served_types=served)


def job(job_id):
    from queue_workflows import node_queue
    return node_queue.get_node_job(job_id)


def run_row(run_id):
    from queue_workflows import run_store
    return run_store.get_run(run_id)


def seen(leases, box="box-a", gen=4, start=START):
    return leases.cluster_reservations(BOXES, gen, start)[box]


def set_job_status(job_id, status):
    from queue_workflows import db
    with db.connection() as conn, conn.cursor() as cur:
        cur.execute("UPDATE workflow_node_jobs SET status = %(s)s WHERE id = %(id)s", {"s": status, "id": job_id})
        conn.commit()


def test_a_reservation_is_exactly_the_contract_row(broker_db):
    from queue_workflows import node_queue
    leases = BrokerLeases(CFG)
    job_id = leases.reserve_for_cluster("box-a", 4, START, model=MODEL)
    row = job(job_id)
    run = run_row(row["run_id"])
    assert run["id"] == cluster_reservation_id(4, START, "box-a")
    assert (run["workflow_name"], run["status"], run["mode"], run["out_dir"]) == ("vllm_lb_lease", "running", "node", None)
    assert (row["queue"], row["status"], row["node_module"], row["node_id"]) == ("gpu", "queued", "system.vllm_lease",
                                                                                "cluster-box-a")
    assert row["server_types"] == [node_queue.CLUSTER_SERVER_KIND] == ["cluster"]
    assert row["force_box"] == ["box-a"] and row["avoid_box"] is None
    assert row["required_model"] is None and row["priority"] == 0 and row["is_priority"] is True
    assert row["inputs"] == {"lease_id": run["id"], "purpose": "vllm-lb/cluster#4",
                             "model": "Qwen/Qwen3.8-Flash-Next", "max_hold_s": CFG.cluster_reservation_max_hold_s}
    assert CFG.cluster_reservation_max_hold_s >= 24 * 3600, "the node's own 6 h default would end a serving pair"
    assert leases.cluster_support() == ""
    assert row.get("context_length") is None, "a sized row is unclaimable by a lane that starts no server"
    if "gpu_slots" in inspect.signature(node_queue.enqueue_node_job).parameters:
        assert row["gpu_slots"] == "all"
    assert leases.db_ok


def test_the_hold_ceiling_is_the_operators_and_the_declared_model_is_the_model_choice(broker_db):
    """ONE model setting (pass 4, DRY): a reservation declares what its generation's ModelChoice serves. There
    is no second knob — the old LB_CLUSTER_MODEL — that could declare something else."""
    cfg = Config(vllm_boxes=CFG.vllm_boxes, cluster_reservation_max_hold_s=86400.0)
    row = job(BrokerLeases(cfg).reserve_for_cluster("box-b", 4, START, model="Qwen/Qwen3.8-27B"))
    assert (row["inputs"]["max_hold_s"], row["inputs"]["model"], row["node_id"]) == (86400.0, "Qwen/Qwen3.8-27B",
                                                                                   "cluster-box-b")
    env = Config.from_env({"LB_ALLOWED_HOSTS": "loopback", "LB_CLUSTER_RESERVATION_MAX_HOLD_S": "172800", "LB_CLUSTER_MODEL": "Qwen/Env"})
    assert env.cluster_reservation_max_hold_s == 172800.0
    assert not hasattr(env, "cluster_model"), "the ModelChoice is the one model setting"
    with pytest.raises(TypeError):
        BrokerLeases(CFG).reserve_for_cluster("box-a", 4, START)      # a reservation always names its model


def test_an_engine_that_cannot_claim_a_reservation_refuses_to_enqueue_one(broker_db, monkeypatch):
    """A row nothing can claim is worse than no row: it is a Start that silently never starts."""
    from queue_workflows import db, node_queue
    monkeypatch.delattr(node_queue, "CLUSTER_SERVER_KIND")
    leases = BrokerLeases(CFG)
    assert "CLUSTER_SERVER_KIND" in leases.cluster_support()
    assert leases.reserve_for_cluster("box-a", 4, START, model=MODEL) is None and "CLUSTER_SERVER_KIND" in leases.last_error
    with db.connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) AS n FROM workflow_runs")
        assert cur.fetchone()["n"] == 0


def test_asking_again_or_from_a_restarted_lb_never_makes_a_second_row(broker_db):
    from queue_workflows import db
    first = BrokerLeases(CFG).reserve_for_cluster("box-a", 4, START, model=MODEL)
    assert BrokerLeases(CFG).reserve_for_cluster("box-a", 4, START, model=MODEL) == first
    with db.connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) AS n FROM workflow_node_jobs WHERE node_module = 'system.vllm_lease'")
        assert cur.fetchone()["n"] == 1


def test_only_the_pinned_box_can_claim_it(broker_db):
    """WHICH lane serving kinds may claim it is broker_parrot's side of the contract (its claim is being
    changed to exempt a reservation from the `served_types` gate: it asks for the box, not a server), so
    this asserts only what the LB relies on — `force_box` — and claims as a lane that serves `cluster`."""
    job_id = BrokerLeases(CFG).reserve_for_cluster("box-a", 4, START, model=MODEL)
    assert claim("box-b-gpu", "box-b") is None, "force_box pins it to box-a"
    assert claim("box-a-gpu", "box-a")["id"] == job_id


def test_the_observation_follows_the_row_through_claim_requeue_and_reclaim(broker_db):
    from queue_workflows import node_queue
    leases = BrokerLeases(CFG)
    job_id = leases.reserve_for_cluster("box-a", 4, START, model=MODEL)
    run_id = cluster_reservation_id(4, START, "box-a")
    assert seen(leases) == (Reservation(run_id, job_id, "queued", None, True), ())
    assert seen(leases, "box-b") == (None, ())
    claim("box-a-gpu", "box-a")
    assert seen(leases) == (Reservation(run_id, job_id, "running", "box-a-gpu", True), ())
    assert node_queue.requeue_running_for_worker("box-a-gpu", "gpu") == 1          # a lane restart
    assert seen(leases) == (Reservation(run_id, job_id, "queued", None, True), ())
    claim("box-a-gpu", "box-a")
    assert seen(leases)[0].status == "running"


@pytest.mark.parametrize("status", ["failed", "completed", "cancelled"])
def test_an_ended_reservation_stays_visible_as_ended(broker_db, status):
    leases = BrokerLeases(CFG)
    job_id = leases.reserve_for_cluster("box-a", 4, START, model=MODEL)
    set_job_status(job_id, status)
    res, _ = seen(leases)
    assert (res.job_id, res.status, res.run_open) == (job_id, status, True)


def test_a_reservation_whose_run_someone_closed_is_still_found_so_it_is_never_asked_for_again(broker_db):
    from queue_workflows import node_queue, run_store
    leases = BrokerLeases(CFG)
    job_id = leases.reserve_for_cluster("box-a", 4, START, model=MODEL)
    run_id = cluster_reservation_id(4, START, "box-a")
    run_store.update_run(run_id, status="cancelled")
    node_queue.cancel_queued_jobs_for_run(run_id)
    fresh = BrokerLeases(CFG)                                   # and after an LB restart
    assert seen(fresh) == (Reservation(run_id, job_id, "cancelled", None, False), ())


def test_release_is_the_classic_release(broker_db):
    leases = BrokerLeases(CFG)
    job_id = leases.reserve_for_cluster("box-a", 4, START, model=MODEL)
    claim("box-a-gpu", "box-a")
    res, _ = seen(leases)
    assert leases.release(Lease(lease_id=res.reservation_id, job_id=res.job_id, created_at=0.0, kind=CLUSTER))
    assert run_row(res.reservation_id)["status"] == "cancelled"
    assert job(job_id)["status"] == "running", "the body sees cancel_event and returns"
    assert seen(leases)[0].run_open is False


def test_an_older_generations_open_reservation_is_found_by_its_purpose_tag(broker_db):
    leases = BrokerLeases(CFG)
    old = leases.reserve_for_cluster("box-a", 3, "a-start-whose-file-is-gone", model=MODEL)
    res, stale = seen(leases)
    assert res is None and [s.job_id for s in stale] == [old]


def test_a_current_reservation_named_by_another_start_id_is_adopted_by_its_purpose_tag(broker_db):
    leases = BrokerLeases(CFG)
    made = leases.reserve_for_cluster("box-a", 4, "the-file-was-rewritten", model=MODEL)
    res, stale = seen(leases)
    assert res is not None and res.job_id == made and stale == ()


def test_the_classic_recover_never_adopts_or_closes_a_reservation(broker_db):
    leases = BrokerLeases(CFG)
    leases.reserve_for_cluster("box-a", 4, START, model=MODEL)
    ended = leases.reserve_for_cluster("box-b", 4, START, model=MODEL)
    set_job_status(ended, "failed")
    assert BrokerLeases(CFG).recover() == []
    assert run_row(cluster_reservation_id(4, START, "box-b"))["status"] == "running", \
        "an ended reservation's run is the cluster loop's to close, on Stop"


def test_running_gpu_jobs_leave_our_reservations_out(broker_db):
    from queue_workflows import node_queue, run_store
    leases = BrokerLeases(CFG)
    job_id = leases.reserve_for_cluster("box-a", 4, START, model=MODEL)
    claim("box-a-gpu", "box-a")
    run_store.insert_run(run_id="fleet", workflow_name="llm_chat", status="running", mode="node")
    node_queue.enqueue_node_job(run_id="fleet", node_id="n", node_module="llm_chat", queue="gpu")
    claim_gpu("box-b-gpu", box="box-b")
    assert leases.running_gpu_jobs(LABELS, exclude=(job_id,)) == {"box-a-gpu": 0, "box-b-gpu": 1}


def test_a_park_is_read_for_the_page_and_the_adapter_has_no_verb_to_write_one(broker_db):
    from queue_workflows import worker_control
    worker_control.set_worker_control("box-b-gpu", "gpu", desired_state="off", stop_policy="drain",
                                      requested_by="cluster-panel")        # the OPERATOR
    leases = BrokerLeases(CFG)
    assert leases.parked_by(LABELS) == {"box-a-gpu": None, "box-b-gpu": "cluster-panel"}
    assert not [name for name in ("park", "unpark", "cluster_parks") if hasattr(leases, name)]


def test_an_unreadable_broker_is_none_never_empty(broker_db, monkeypatch):
    from queue_workflows import db, run_store, worker_control
    leases = BrokerLeases(CFG)

    def boom(*_a, **_k):
        raise RuntimeError("database did not answer")

    monkeypatch.setattr(worker_control, "get_worker_control", boom)
    monkeypatch.setattr(run_store, "get_run", boom)
    monkeypatch.setattr(db, "connection", boom)
    assert leases.cluster_reservations(BOXES, 4, START) is None
    assert leases.parked_by(LABELS) is None
    assert leases.running_gpu_jobs(LABELS) is None
    assert leases.reserve_for_cluster("box-a", 4, START, model=MODEL) is None and leases.db_ok is False
