"""A LOCAL NODE HOLDS ITS BOX THROUGH THE PAIR'S RESERVATION CONTRACT (operator, 2026-10-17: "ensure box-c is also
correctly reserved in broker_parrot"; worklog/parallel-clusters.md, "The local node's reservation").

A node the LB requests runs on a box broker_parrot's own gpu lane also serves (box-c's lane runs llm_chat,
compose_region_report and the gateway's own fallback leases on the same iGPU — worklog/chat-capabilities-research.md).
Without a reservation the node's vLLM and the broker's work would load onto one card, neither side knowing. So, with the
broker on and a control whose data says it reserves (`fleet.Control.reserves`):

  1. an `up` wish asks broker_parrot for the box's reservation FIRST — the pair's row, word for word
     (`BrokerLeases._cluster_reservation_kwargs`), pinned with `force_box`, the whole box, of the engine's cluster kind;
  2. the `up` intent is released to the host only once the reservation is CLAIMED (the lane holds the box: its own work
     drained, its ollama unloaded) — waiting, or draining while the lane still runs jobs, until then;
  3. idempotent: the run id is derived from the wish's id and the box; a restarted LB adopts what is open;
  4. on `down` the reservation is returned only once the node is off — never under a running engine;
  5. broker off: as today, and the view says the box is unreserved.

The sqlite engine is the same `node_queue` the lanes claim from (tests/conftest.py `broker_db`).
"""
from __future__ import annotations

from tests.fakes import FakeProber, claim_gpu
from tests.test_cluster_local import FLEET, Local
from tests.test_cluster_one_host import DOWN, LIVE, NODE, Clock
from vllm_lb import node_control as nc
from vllm_lb.catalog import ModelCatalog
from vllm_lb.cluster import ClusterHost
from vllm_lb.cluster_run import build_cluster_loop
from vllm_lb.config import Config
from vllm_lb.control import Controller
from vllm_lb.fleet import parse_fleet
from vllm_lb.hub import Hub
from vllm_lb.leases import BrokerLeases, cluster_reservation_id


class Reserved(Local):
    """`Local` with the broker ON: the same node, its loop holding a `BrokerLeases` over the sqlite engine."""

    def __init__(self, tmp_path, answer: dict, *, listening: bool = True) -> None:  # noqa: D107 — see the class
        self.perf_dir = str(tmp_path / "perf")
        self.cfg = Config(cluster_hosts=(ClusterHost("node-a-gpu", "node-a"),), cluster_url=NODE,
                          cluster_control="local", fleet=parse_fleet(FLEET, "fleet.yml"), probe_interval_s=0,
                          cluster_state_path=str(tmp_path / "cluster.json"), listen_host="127.0.0.1", listen_port=0,
                          perf_dir=self.perf_dir, broker_enabled=True, lease_workflow="vllm_lb_lease",
                          lease_node_module="system.vllm_lease")
        self.dir = tmp_path / nc.NODE_DIR
        self.dir.mkdir(exist_ok=True)
        self.clock = Clock()
        self.hub = Hub(self.cfg, clock=self.clock)
        self.leases = BrokerLeases(self.cfg)
        self.loop = build_cluster_loop(self.cfg, self.hub, self.leases, None, catalog=ModelCatalog.load())
        self.loop.clock = self.clock
        self.prober = FakeProber({NODE: answer})
        self.recorder = None
        self.ctl = Controller(self.cfg, self.hub, self.leases, prober=self.prober, clock=self.clock,
                              cluster_loop=self.loop)
        if listening:
            self.beat()

    def wish(self, state: str) -> dict:
        return nc.IntentStore(self.cfg.cluster_state_path).write(state, {}, now=self.clock.t)


def _rows():
    from queue_workflows import db
    with db.connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT j.id, j.status, j.force_box, j.server_types, j.inputs, j.run_id, r.status AS run_status "
                    "FROM workflow_node_jobs j JOIN workflow_runs r ON r.id = j.run_id ORDER BY j.created_at")
        return [dict(r) for r in cur.fetchall()]


def _claim():
    return claim_gpu("node-a-gpu", box="node-a", served_types=("cluster",))


def test_an_up_wish_reserves_the_box_first_and_releases_nothing_until_the_lane_holds_it(tmp_path, broker_db):
    node = Reserved(tmp_path, DOWN)
    wish = node.wish(nc.UP)
    view = node.tick()
    rows = _rows()
    assert len(rows) == 1, rows
    row = rows[0]
    assert row["force_box"] == ["node-a"] and row["server_types"] == ["cluster"] and row["status"] == "queued"
    import json
    start = json.loads((node.dir / nc.HOLD).read_text())
    assert start["wish"] == wish["id"] and len(start["id"]) == 16, start
    assert row["run_id"] == cluster_reservation_id(wish["id"], f"node:{start['id']}", "node-a"), \
        "derived from this START's identity (minted at the up, not the wish id alone) + the box"
    assert row["inputs"]["purpose"].startswith("vllm-lb/cluster#") and row["inputs"]["cluster"] == "node"
    assert node.released() is None, "no `up` reaches the host before broker_parrot holds the box"
    assert view["state"] in ("reserving", "waiting") and "node-a" in view["detail"], view["detail"]
    assert view["boxes"]["node-a"]["reservation"]["status"] == "queued"
    assert {"reserve", "drain"} <= set(view["topology"]["steps"]), "the stepper shows reserve and drain"
    view = node.tick()
    assert len(_rows()) == 1, "idempotent: a second tick asks for nothing new"
    assert view["state"] == "waiting" and node.released() is None
    assert _claim() is not None, "the box's lane claims it as the pair's lanes do"
    view = node.tick()
    assert node.released() is not None and node.released()["state"] == nc.UP, "claimed: the up intent is released"
    assert view["state"] == "starting" and view["boxes"]["node-a"]["reservation"]["status"] == "running"
    assert view["reservation"]["state"] == "held"


def test_while_the_lane_still_runs_jobs_the_node_waits_for_the_drain(tmp_path, broker_db):
    from queue_workflows import node_queue, run_store
    run_store.insert_run(run_id="other", workflow_name="someone", out_dir=None, status="running", mode="node")
    node_queue.enqueue_node_job(run_id="other", node_id="chat", node_module="llm_chat", queue="gpu",
                                force_box=["node-a"])
    claim_gpu("node-a-gpu", box="node-a")
    node = Reserved(tmp_path, DOWN)
    node.wish(nc.UP)
    node.tick()
    view = node.tick()
    assert view["state"] == "draining" and "1" in view["detail"], view["detail"]
    assert node.released() is None


def test_a_restarted_lb_adopts_the_open_reservation(tmp_path, broker_db):
    node = Reserved(tmp_path, DOWN)
    node.wish(nc.UP)
    node.tick()
    again = Reserved(tmp_path, DOWN)
    again.tick()
    assert len(_rows()) == 1, "found, not asked again"


def test_a_down_returns_the_reservation_only_once_the_node_is_off(tmp_path, broker_db):
    node = Reserved(tmp_path, DOWN)
    node.wish(nc.UP)
    node.tick()
    _claim()
    node.tick()
    node.beat(applied=node.applied())
    node.prober.table[NODE] = LIVE
    assert node.tick()["state"] == "running"
    node.wish(nc.DOWN)
    view = node.tick()
    assert view["state"] in ("unloading", "starting") and _rows()[0]["run_status"] == "running", "still held"
    node.beat(applied=node.applied())
    node.prober.table[NODE] = DOWN
    view = node.tick()
    view = node.tick()
    assert _rows()[0]["run_status"] == "cancelled", "returned once the node is off"
    assert node.tick()["state"] == "off"


def test_with_the_broker_off_nothing_is_reserved_and_the_view_says_so(tmp_path):
    node = Local(tmp_path, DOWN)
    nc.IntentStore(node.cfg.cluster_state_path).write(nc.UP, {}, now=node.clock.t)
    view = node.tick()
    assert node.released() is not None, "released at once, as today"
    assert view["reservation"] == {"state": "unreserved", "detail": "the broker is off (LB_BROKER_ENABLED): nothing "
                                                                    "reserves node-a in broker_parrot"}
    assert "reserve" not in view["topology"]["steps"]


def test_whether_a_cluster_reserves_is_its_controls_data_never_its_name():
    from vllm_lb.fleet import CONTROLS
    assert {n: c.reserves for n, c in CONTROLS.items()} == {"ssh": True, "local": True, "none": False}


# ── the broker engine claims the whole box beside a lease already on it ───────────────────────────────────────────

def test_the_engine_claims_a_reservation_beside_a_running_ollama_lease_on_its_box(tmp_path, broker_db):
    """AUDIT-FINAL-3 P2-6, ported (P6): the sqlite engine claims the whole-box reservation beside a running `force_box`
    ollama lease — so no LB-side lease rule is needed for the claim (the earlier scaler change is dropped). (Keep loaded,
    whose kept lease this row once was, is retired, 2026-10-17: recover releases one.)"""
    from queue_workflows import node_queue, run_store
    run_store.insert_run(run_id="kept", workflow_name="vllm_lb_lease", out_dir=None, status="running", mode="node")
    node_queue.enqueue_node_job(run_id="kept", node_id="lease", node_module="system.vllm_lease", queue="gpu",
                                server_types=["ollama"], force_box=["node-a"],
                                inputs={"lease_id": "kept", "purpose": "vllm-lb/keep", "max_hold_s": 21600})
    assert claim_gpu("node-a-gpu", box="node-a") is not None
    node = Reserved(tmp_path, DOWN)
    node.wish(nc.UP)
    node.tick()
    assert _claim() is not None


# ── the reservation backs the node, or the node comes down (AUDIT-FINAL-3, probes ported) ───────────────────────────

def _up_running(tmp_path):
    node = Reserved(tmp_path, DOWN)
    node.wish(nc.UP)
    node.tick()
    assert node.released() is None
    assert _claim() is not None
    node.tick()
    node.beat(applied=node.applied())
    node.prober.table[NODE] = LIVE
    assert node.tick()["state"] == "running"
    return node


def _comes_down_and_stays_failed(node, word):
    view = node.tick()
    assert view["state"] == "failed" and word in view["detail"], view["detail"]
    released = node.released()
    assert released["state"] == nc.DOWN and released["force"] is True, "the node is taken down at once"
    node.beat(applied=node.applied())
    node.prober.table[NODE] = DOWN
    for _ in range(3):
        view = node.tick()
    assert view["state"] == "failed", "sticky: never quietly re-reserved"
    assert len(_rows()) == 1, "no second row under the same start"


def test_a_reservation_that_ends_under_a_running_node_takes_the_node_down(tmp_path, broker_db):
    """P1-1 (probe P4): the job ends (a hold ceiling, the lane's cancel) while the node serves — the node does not keep
    the card the lane is giving back to its own work: a DOWN intent is released, and the plan is failed."""
    from queue_workflows import node_queue
    node = _up_running(tmp_path)
    node_queue.mark_failed(_rows()[0]["id"], error="hold ceiling")
    _comes_down_and_stays_failed(node, "ended the reservation")


def test_a_run_closed_from_outside_is_a_sticky_failure_never_a_quiet_re_ask(tmp_path, broker_db):
    """P1-2 (probe P5): the run is cancelled outside the LB — read whatever its status, it is ENDED: failed, down."""
    from queue_workflows import run_store
    node = _up_running(tmp_path)
    run_store.update_run(_rows()[0]["run_id"], status="cancelled")
    _comes_down_and_stays_failed(node, "its run was closed")


def test_a_reservation_claimed_by_another_lane_does_not_back_the_node(tmp_path, broker_db):
    """P2-3: the pair's `claimed_by` check — only the box's OWN lane holding it backs the node."""
    node = Reserved(tmp_path, DOWN)
    node.wish(nc.UP)
    node.tick()
    assert claim_gpu("node-a-gpu", box="node-a", served_types=("cluster",)) is not None
    from queue_workflows import db
    with db.connection() as conn, conn.cursor() as cur:
        cur.execute("UPDATE workflow_node_jobs SET claimed_by = 'elsewhere-gpu'")
        conn.commit()
    view = node.tick()
    assert view["state"] == "failed" and "elsewhere-gpu" in view["detail"], view["detail"]
    assert node.released() is None, "never released up on another lane's claim"


def test_after_a_stop_the_next_start_is_a_new_start_even_if_the_wish_file_was_lost(tmp_path, broker_db):
    """P2-2: the run id names THIS start (`hold.json`), not the wish id alone — which restarts at 1 when the wish file
    is lost and would name the old, closed run."""
    import os
    node = _up_running(tmp_path)
    first = _rows()[0]["run_id"]
    node.wish(nc.DOWN)
    node.tick()
    node.beat(applied=node.applied())
    node.prober.table[NODE] = DOWN
    for _ in range(3):
        node.tick()
    assert _rows()[0]["run_status"] == "cancelled" and not (node.dir / nc.HOLD).exists()
    os.remove(node.cfg.cluster_state_path)                     # the wish file is lost: its next id is 1 again
    assert node.wish(nc.UP)["id"] == 1
    view = node.tick()
    assert view["state"] in ("reserving", "waiting"), view["detail"]
    assert {r["run_id"] for r in _rows()} == {first, _rows()[-1]["run_id"]} and len(_rows()) == 2


def test_a_control_that_does_not_reserve_reserves_nothing_with_the_broker_on(tmp_path, broker_db, monkeypatch):
    """P2-1: whether a cluster reserves is READ off its control (`Control.reserves`): a local control patched to
    reserve nothing, the broker on, asks for no row, draws no reserve/drain step and says unreserved."""
    from dataclasses import replace
    from types import MappingProxyType

    from vllm_lb import fleet
    monkeypatch.setattr(fleet, "CONTROLS", MappingProxyType({**fleet.CONTROLS,
                                                             "local": replace(fleet.CONTROLS["local"], reserves=False)}))
    node = Reserved(tmp_path, DOWN)
    node.wish(nc.UP)
    view = node.tick()
    assert _rows() == [] and node.released() is not None
    assert view["reservation"]["state"] == "unreserved" and "reserve" not in view["topology"]["steps"]


# ── AUDIT-FINAL-4 (REJECT of 5748da3): probes ported ────────────────────────────────────────────────────────────────

def _db(sql, args=None):
    from queue_workflows import db
    with db.connection() as conn, conn.cursor() as cur:
        cur.execute(sql, args or {})
        conn.commit()


def test_a_lost_start_file_never_orphans_the_reservation_it_held(tmp_path, broker_db):
    """Q2 (P1): `hold.json` lost under a held, running node. The node comes down (nothing backs it for a start it
    knows); once it is off, this cluster's OTHER open reservations on its box — found by `inputs.cluster`, as the pair
    finds its stale ones — are returned, so the box is never held for up to `max_hold_s` by a row nobody owns."""
    import os
    node = _up_running(tmp_path)
    os.remove(node.dir / nc.HOLD)
    assert node.tick()["state"] == "failed" and node.released()["state"] == nc.DOWN
    node.beat(applied=node.applied())
    node.prober.table[NODE] = DOWN
    for _ in range(3):
        node.tick()
    rows = _rows()
    assert rows[0]["run_status"] == "cancelled", "the orphan is returned once the node is off"
    assert len(rows) == 2 and rows[1]["status"] == "queued", "the wish is still up: a new start asks again (not sticky)"
    node.wish(nc.DOWN)
    for _ in range(3):
        node.tick()
    assert {r["run_status"] for r in _rows()} == {"cancelled"}, "a Stop returns everything this cluster holds there"


def test_a_reservation_requeued_under_a_running_node_is_granted_the_pairs_grace_then_the_node_comes_down(
        tmp_path, broker_db):
    """Q3: back in the queue under the running node (a lane restart): kept for the pair's REQUEUE_GRACE_S, then
    NodeDown + failed — never kept without a bound."""
    from vllm_lb.cluster import REQUEUE_GRACE_S
    node = _up_running(tmp_path)
    _db("UPDATE workflow_node_jobs SET status = 'queued', claimed_by = NULL")
    view = node.tick(seconds=REQUEUE_GRACE_S / 2)
    assert view["state"] == "waiting" and node.released()["state"] == nc.UP
    node.tick(seconds=REQUEUE_GRACE_S)
    view = node.tick()
    assert view["state"] == "failed" and node.released()["state"] == nc.DOWN, view["detail"]


def test_an_open_run_with_no_job_yet_is_asked_again_never_a_claim_on_no_lane(tmp_path, broker_db):
    """Q4: the enqueue failed after the run row was written (or the LB died between the two): the pair's reading —
    not asked yet — and the next tick enqueues the job under the same run (`reserve_for_cluster`'s own retry)."""
    from vllm_lb.leases import BrokerLeases
    node = Reserved(tmp_path, DOWN)
    node.wish(nc.UP)
    orig = BrokerLeases._cluster_reservation_kwargs

    def boom(self, **kw):
        raise RuntimeError("enqueue failed")
    BrokerLeases._cluster_reservation_kwargs = boom
    try:
        node.tick()
    finally:
        BrokerLeases._cluster_reservation_kwargs = orig
    assert _rows() == []
    view = node.tick()
    assert len(_rows()) == 1 and view["state"] in ("reserving", "waiting"), view["detail"]


def test_an_unreadable_broker_under_a_running_node_keeps_the_node(tmp_path, broker_db, monkeypatch):
    """A9: the pair's rule — an unreadable broker is not an ended reservation: `unknown`, the `up` stays."""
    node = _up_running(tmp_path)
    monkeypatch.setattr(node.leases, "node_reservation", lambda run_id: (False, None))
    view = node.tick()
    assert view["state"] == "unknown" and node.released()["state"] == nc.UP


def test_a_closed_run_is_refused_by_reserve_for_cluster_never_handed_back_as_asked(tmp_path, broker_db):
    """A5: the closed-run guard in `reserve_for_cluster` itself (the sticky failure `cluster_reservation_id` promises)."""
    from queue_workflows import run_store
    node = Reserved(tmp_path, DOWN)
    assert node.leases.reserve_for_cluster("node-a", 1, "x", model="m", cluster="node") is not None
    run_store.update_run(_rows()[0]["run_id"], status="cancelled")
    assert node.leases.reserve_for_cluster("node-a", 1, "x", model="m", cluster="node") is None
    assert "was closed" in node.leases.last_error


def test_a_node_told_up_but_not_answering_yet_is_up_for_the_hold(tmp_path, broker_db):
    """A6: while vLLM loads (minutes) the node answers nothing, yet its host was told `up`: an ended reservation then
    takes it down at once, not only once it answers."""
    from queue_workflows import node_queue
    node = Reserved(tmp_path, DOWN)
    node.wish(nc.UP)
    node.tick()
    _claim()
    node.tick()
    assert node.released()["state"] == nc.UP
    node_queue.mark_failed(_rows()[0]["id"], error="hold ceiling")
    node.tick()
    assert node.released()["state"] == nc.DOWN


def test_the_forced_down_is_written_once_not_every_tick(tmp_path, broker_db):
    """Q5: the node still answers while its host applies the down: the forced DOWN intent is not rewritten."""
    from queue_workflows import node_queue
    node = _up_running(tmp_path)
    node_queue.mark_failed(_rows()[0]["id"], error="hold ceiling")
    ids = []
    for _ in range(4):
        node.tick()
        ids.append(node.released()["id"])
    assert len(set(ids)) == 1, ids


# ── AUDIT-FINAL-5 pins: the stale scan's own read (D5), the sweep's node-off gate (D6), past the grace (P3-1) ─────

def test_an_unreadable_stale_scan_alone_under_a_running_node_keeps_the_node(tmp_path, broker_db, monkeypatch):
    """D5 (auditor's R2): the start's own row reads, but the stale scan (`node_reservations`) cannot — `unknown`, the
    `up` stays: an unreadable broker is never an ended reservation, whichever of the two reads failed."""
    node = _up_running(tmp_path)
    monkeypatch.setattr(node.leases, "node_reservations", lambda box, cluster: None)
    view = node.tick()
    assert view["state"] == "unknown" and node.released()["state"] == nc.UP, view["detail"]


def test_a_stray_reservation_is_returned_only_once_the_node_is_off():
    """D6: a stray (a lost start file's row) is never returned under a node still up — the box would go back to the
    lane while the engine is on it; once the node is off, it is."""
    from vllm_lb.cluster import Plan, ReleaseReservation, Reservation, plan_node_hold
    held = Reservation("run-now", "job-now", "running", claimed_by="node-a-gpu")
    stray = Reservation("run-old", "job-old", "running", claimed_by="node-a-gpu")

    def plan(**kw):
        return plan_node_hold(up=True, start="1", current=held, readable=True, strays=(stray,), support="",
                              box="node-a", label="node-a-gpu", jobs=None, local=Plan([], "running"), **kw)
    actions, _ = plan(node_up=True, node_off=False)
    assert not any(isinstance(a, ReleaseReservation) for a in actions), actions
    actions, _ = plan(node_up=False, node_off=True)
    assert actions == [ReleaseReservation("node-a", "run-old", "job-old")]


def test_past_the_requeue_grace_the_node_stays_down_until_its_lane_claims_the_box_again(tmp_path, broker_db):
    """P3-1 (auditor's R1): past REQUEUE_GRACE_S the node is taken down and the plan failed — NOT sticky (the
    reservation never ended, it is still this start's): with the node off and the row still queued it waits, and once
    the box's own lane claims it again the node starts with no Stop/Start — only ever on a box broker_parrot holds."""
    from vllm_lb.cluster import REQUEUE_GRACE_S
    node = _up_running(tmp_path)
    _db("UPDATE workflow_node_jobs SET status = 'queued', claimed_by = NULL")
    node.tick(seconds=REQUEUE_GRACE_S / 2)
    node.tick(seconds=REQUEUE_GRACE_S)
    view = node.tick()
    assert view["state"] == "failed" and node.released()["state"] == nc.DOWN, view["detail"]
    assert "Stop, then Start" not in view["detail"], "the text says what happens: it is not sticky"
    node.beat(applied=node.applied())
    node.prober.table[NODE] = DOWN
    for _ in range(3):
        view = node.tick()
    assert view["state"] == "waiting" and node.released()["state"] == nc.DOWN, view["detail"]
    _db("UPDATE workflow_node_jobs SET status = 'running', claimed_by = 'node-a-gpu'")
    node.tick()
    assert node.released()["state"] == nc.UP, "held again by its own lane: the node starts again"
    assert len(_rows()) == 1, "the same reservation, never a second row"
