"""Only DEMAND creates a lease: a restart's adoption is RETENTION, and classic demand starts at the Stop.

2026-09-15 (worklog/vllm-gateway-program.md, "Frame — pass 7"). The LB restarted at 20:10:42 and adopted
a running FALLBACK lease. `Hub.note_recovered` wrote that adoption into `last_request_at`, the one clock
`desired_leases` reads as demand. At 20:19:40 Stop unparked the GB10 boxes; at 20:19:42 the scaler created
vLLM lease 9ec8a27b with no client request since 20:09. broker_parrot started vLLM on BOTH GB10 boxes for it,
and stopped both, unused, at 20:28:40.

Two concepts shared that clock. They are split here, by name:

* DEMAND — a client request ARRIVED (`Hub.note_request` → `last_request_at`). Only demand creates a
  lease. For a classic vLLM it counts from the cluster's last Stop (`cluster_stopped_at`,
  `scaler.classic_demand_at`): a turn the cluster served is no reason to start a classic vLLM.
* RETENTION — what adopting a lease grants (`Hub.note_recovered` → `adopted_at`, `adopted_lease_ids`;
  `scaler.retained_leases`). The adopted lease is not released as idle until its window has passed
  from the adoption, and nothing new is created on its account.
"""
from __future__ import annotations

import http.client
import json
import threading
import time

import pytest

from tests.fakes import FakeBackend, FakeProber
from tests.test_boxes_on import _control as park
from tests.test_control import A, B, CFG, LIVE_V, Clock, _pending, vllm_leases, world  # noqa: F401  (fixture)
from tests.test_restart import _held_by_a_previous_process
from vllm_lb import scaler
from vllm_lb.cluster import parse_hosts
from vllm_lb.config import Config, VllmBox
from vllm_lb.control import Controller
from vllm_lb.hub import Hub
from vllm_lb.leases import RETIRED_FALLBACK_KIND, BrokerLeases
from vllm_lb.proxy import Proxy
from vllm_lb.scaler import Release, desired_leases, plan_actions
from vllm_lb.state import CLUSTER, LEASE_QUEUED, LEASE_RUNNING, VLLM, Backend, Lease, Snapshot


def _publish(hub, mode, *, held=False):
    """What the cluster loop publishes, as far as this rule reads it: the operator's desired mode."""
    hub.set_cluster({"configured": True, "desired": {"mode": mode}, "held": held}, "pair")


# ── the pure half: the rules live in the scaler ──────────────────────────────

NOW = 10_000.0
RULES = Config(broker_enabled=True, idle_release_s=600.0, max_leases=2)


def _box_a(label):
    return Backend(kind=VLLM, url=f"http://{label}:8000", host_label=label, box=label.removesuffix("-gpu"))


def _running(lease_id):
    """A vLLM lease held since long ago: past every window on its own clocks."""
    return Lease(lease_id=lease_id, job_id=f"j-{lease_id}", kind=VLLM, status=LEASE_RUNNING,
                 host_label="box-a-gpu", box="box-a", created_at=NOW - 5000, claimed_at=NOW - 5000)


def _snap(*, leases=(), last_request_at=None, adopted_at=None, adopted=(), cluster_stopped_at=None,
          last_pressure_at=None, cluster_inflight=0, vllm_inflight=0):
    """The deployed gateway's own shape: two GB10 box sidecars and the TP=2 head (the fallback is retired).

    `*_inflight` is what THIS LB is relaying to that backend right now — the fact the Stop of
    2026-09-15 turned into a classic lease.
    """
    head = Backend(kind=CLUSTER, url="http://192.0.2.2:8100", live=True,
                   served_models=("Qwen/Qwen3.8-Flash-Next",), inflight=cluster_inflight)
    box_a = _box_a("box-a-gpu")
    box_a.inflight = vllm_inflight
    return Snapshot(now=NOW, backends=[box_a, _box_a("box-b-gpu"), head],
                    leases=list(leases), last_request_at=last_request_at,
                    last_pressure_at=last_pressure_at,
                    db_ok=True, adopted_at=adopted_at, adopted_lease_ids=frozenset(adopted),
                    cluster_stopped_at=cluster_stopped_at)


def test_classic_demand_counts_only_from_the_cluster_stop():
    before = _snap(last_request_at=NOW - 60, cluster_stopped_at=NOW - 30)
    after = _snap(last_request_at=NOW - 10, cluster_stopped_at=NOW - 30)
    assert scaler.classic_demand_at(before) is None and desired_leases(before, RULES) == 0, \
        "a turn served before the Stop is not demand for a classic vLLM"
    assert scaler.classic_demand_at(after) == NOW - 10 and desired_leases(after, RULES) == 1
    assert desired_leases(_snap(last_request_at=NOW - 60), RULES) == 1, "no Stop on record: every request counts"


def test_retention_keeps_an_adopted_lease_through_its_window_then_lets_it_go():
    held = _running("r1")
    inside = _snap(leases=[held], adopted_at=NOW - 600, adopted={"r1"})
    assert scaler.retained_leases(inside, VLLM, RULES.idle_release_s) == [held]
    assert desired_leases(inside, RULES) == 1 and plan_actions(inside, RULES) == [], \
        "a restart is not idleness: neither the fleet rule nor the per-box rule releases it"
    past = _snap(leases=[held], adopted_at=NOW - 601, adopted={"r1"})
    assert plan_actions(past, RULES) == [Release(lease_id="r1")]


def test_retention_asks_for_nothing_new():
    """An adopted lease that is no longer held is not replaced on retention's account."""
    gone = _snap(adopted_at=NOW - 5, adopted={"r1"})
    assert desired_leases(gone, RULES) == 0 and plan_actions(gone, RULES) == []


# ── the hub records the facts; it decides nothing ────────────────────────────


def test_adoption_is_recorded_as_retention_never_as_demand():
    hub = Hub(CFG, clock=Clock(1000.0))
    hub.note_recovered([Lease(lease_id="r1", job_id="j1", created_at=1.0, status=LEASE_RUNNING,
                              host_label="box-a-gpu", box="box-a"),
                        Lease(lease_id="q1", job_id="j2", created_at=1.0, status=LEASE_QUEUED)])
    s = hub.snapshot()
    assert s.last_request_at is None, "adopting a lease is not a client request"
    assert (s.adopted_at, s.adopted_lease_ids) == (1000.0, frozenset({"r1"}))
    assert hub.backend_for(A).last_used_at is None, "nor is it a use of the box"


def test_the_stop_is_the_desired_mode_leaving_cluster_for_classic():
    clock = Clock(1000.0)
    hub = Hub(CFG, clock=clock)

    def publish_at(t, mode):
        clock.t = t
        _publish(hub, mode)
        return hub.snapshot().cluster_stopped_at

    assert publish_at(1000, "classic") is None, "booting into classic is not a Stop"
    assert publish_at(1010, "cluster") is None, "a Start is not a Stop"
    assert publish_at(1020, "cluster") is None
    assert publish_at(1030, "classic") == 1030
    assert publish_at(1040, "classic") == 1030, "staying classic moves nothing"
    publish_at(1050, "cluster")
    assert publish_at(1060, "classic") == 1060, "the LAST Stop is the one that counts"


# ── the loop: the real controller over a sqlite broker ───────────────────────


def test_D1_a_fresh_lb_that_adopts_nothing_and_hears_nothing_creates_no_lease(world):
    ctl, hub, _, _, clock = world
    ctl.recover()
    start = clock.t
    for offset in (0, 2, 300, 600, 602):
        clock.t = start + offset
        ctl.tick()
        assert hub.leases() == [] and _pending() == 0, f"a lease at +{offset}s with no request"


def test_D1b_a_fallback_lease_an_older_lb_left_is_released_and_asks_for_no_vllm(world):
    """The fallback tier is retired (2026-10-17): a fallback lease a previous process held is nobody's now —
    `recover` releases it on sight (no plan would ever let it go), and it is no reason for a vLLM either."""
    from queue_workflows import run_store
    ctl, hub, leases, _, clock = world
    held = _held_by_a_previous_process(leases, kind=RETIRED_FALLBACK_KIND)
    adopted = clock.t
    ctl.recover()                                    # the NEW process releases it
    ctl.tick()
    assert hub.leases() == [] and _pending() == 0, "released, and adoption is not demand"
    assert run_store.get_run(held.lease_id)["status"] == "cancelled"
    clock.t = adopted + ctl.cfg.idle_release_s       # and no vLLM across the whole vLLM window
    ctl.tick()
    assert vllm_leases(hub) == []

    hub.note_request()                               # this world CAN lease a vLLM: demand was all it lacked
    ctl.tick()
    assert len(vllm_leases(hub)) == 1


def test_D2_turns_before_the_stop_are_not_classic_demand_but_the_next_request_is(world):
    ctl, hub, _, _, clock = world
    _publish(hub, "cluster")
    hub.note_request()                               # a turn while the operator wants the cluster
    clock.t += 60
    _publish(hub, "classic")                         # Stop
    for _ in range(3):
        ctl.tick()
        assert vllm_leases(hub) == [] and _pending() == 0, "the last request is older than the Stop"
        clock.t += 2

    hub.note_request()                               # the first classic turn
    ctl.tick()
    (lease,) = vllm_leases(hub)
    assert lease.status == LEASE_QUEUED, "a request after the Stop creates one on the next tick"


def test_retention_never_replaces_an_adopted_lease_the_broker_ended(world):
    from queue_workflows import db
    ctl, hub, leases, prober, _ = world
    held = _held_by_a_previous_process(leases)
    prober.table[A] = LIVE_V
    ctl.recover()
    ctl.tick()
    assert [l.lease_id for l in vllm_leases(hub)] == [held.lease_id]

    with db.connection() as conn, conn.cursor() as cur:
        cur.execute("UPDATE workflow_node_jobs SET status = 'failed' WHERE id = %(id)s", {"id": held.job_id})
        conn.commit()
    ctl.tick()                                       # refresh finds it lost
    ctl.tick()
    assert vllm_leases(hub) == [] and _pending() == 0, \
        "retention keeps what was adopted; with no request it never asks for another"


def test_the_2026_09_15_chain_starts_no_vllm(world):
    """Restart finds a (now retired) fallback lease while the cluster holds the GB10 boxes — and releases it; nine minutes
    later Stop unparks them. The old seed was 540 s old at 20:19:42, inside the 600 s window."""
    ctl, hub, leases, _, clock = world
    park("box-a-gpu", "off")
    park("box-b-gpu", "off")
    _publish(hub, "cluster")
    _held_by_a_previous_process(leases, kind=RETIRED_FALLBACK_KIND)
    ctl.recover()                                    # 20:10:42
    ctl.tick()

    clock.t += 538                                   # 20:19:40: Stop, and the GB10 boxes come back
    _publish(hub, "classic")
    park("box-a-gpu", "on")
    park("box-b-gpu", "on")
    ctl.tick()
    clock.t += 2                                     # 20:19:42
    ctl.tick()
    assert vllm_leases(hub) == [] and _pending() == 0


# ── the delta round: D4 failed LIVE on 2026-09-15 ────────────────────────────
#
# The operator stopped generation 5 at 23:23:52,804 with turns still in flight on the pair
# (`2 seqs`), and the broker's job table shows a classic vLLM lease created at **23:23:54** and a
# second at **23:24:03**. D4 forbids both for three minutes.
#
# No client turn is needed to explain them. `desired_leases` read `inflight` on ANY backend as
# classic demand — including the TP=2 HEAD, which at 23:23:54 was carrying the seven turns the
# Stop was about to kill (they died 2 s later, at 23:23:56, all with `IncompleteRead(0 bytes
# read)`). And the second ask rested on `last_pressure_at`, built before the Stop by those same
# turns overflowing a pair the router judged full — a clock the Stop does not reset either.
#
# The `inflight` leak of `81f4b2f` made those counts permanent (the panel still read 7 at 23:45),
# but it is not the cause and fixing it is not enough: at 23:23:54 the seven were genuinely in
# flight. The rule pass 7 wrote for arrivals holds for every input — nothing that predates the
# Stop is classic demand — and in-flight work belongs to the backend that is serving it.


def test_D4_the_turns_in_flight_when_the_cluster_stops_are_not_classic_demand():
    """The 23:23:54 row. Seven turns in flight on the HEAD, every one of them arrived before the
    Stop: a classic vLLM would serve none of them, and asking for one costs both GB10 boxes a cold start."""
    s = _snap(last_request_at=NOW - 30, cluster_stopped_at=NOW - 2, cluster_inflight=7)
    assert scaler.classic_demand_at(s) is None, "every one of them arrived before the Stop"
    assert desired_leases(s, RULES) == 0, "the cluster's own turns are not a classic vLLM's demand"
    assert plan_actions(s, RULES) == []


def test_pressure_from_before_the_stop_does_not_double_the_first_classic_ask():
    """The 23:24:03 row. Pressure is demand's own signal, so it counts from the Stop like demand:
    turns that queued while the cluster served are no evidence that ONE classic box is too few."""
    stale = _snap(last_request_at=NOW - 1, last_pressure_at=NOW - 30, cluster_stopped_at=NOW - 2)
    assert desired_leases(stale, RULES) == 1, "a classic turn arrived: one box, not two"
    fresh = _snap(last_request_at=NOW - 1, last_pressure_at=NOW - 1, cluster_stopped_at=NOW - 2)
    assert desired_leases(fresh, RULES) == 2, "pressure measured AFTER the Stop still asks for the second"


def test_a_stream_on_a_vllm_we_hold_still_keeps_its_lease_past_the_window():
    """THE KILL LINE for this delta (`tests/test_scaler.py` since `e4fadf4`): the idle clock starts
    at ARRIVAL, so a generation streaming longer than `idle_release_s` must not lose its lease
    mid-stream. In flight on a box we hold KEEPS — it is retention, like adoption; it never asks."""
    held = _running("r1")                            # box-a-gpu
    s = _snap(leases=[held], last_request_at=NOW - 900, vllm_inflight=1)
    assert desired_leases(s, RULES) == 1
    assert plan_actions(s, RULES) == [], "neither the fleet rule nor the per-box rule releases it"


# ── the same Stop, through the real proxy and the real controller ────────────


def _send_a_turn(proxy) -> None:
    """One inference turn, as a client sends it. It never completes: the head answers a status
    line and then says nothing, which is what a pair being torn down looks like from here."""
    try:
        conn = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=30)
        conn.request("POST", "/v1/messages",
                     body=json.dumps({"model": "Qwen/Qwen3.8-Flash-Next"}).encode(),
                     headers={"Content-Type": "application/json"})
        conn.getresponse().read()
    except OSError:
        pass


def _wait_for_inflight(hub, url: str, timeout_s: float = 5.0) -> None:
    until = time.monotonic() + timeout_s
    while time.monotonic() < until:
        if hub.backend_for(url).inflight > 0:
            return
        time.sleep(0.02)
    raise AssertionError(f"no request reached {url} within {timeout_s}s")


@pytest.fixture
def gateway(broker_db):
    """The deployed shape, hermetically: the real Proxy, Hub, BrokerLeases and Controller over a
    fake TP=2 head. No cluster loop — the loop's only input to this rule is the view it publishes,
    and the test publishes that itself, so a Stop here is exactly what a Stop is there."""
    head = FakeBackend(served=("Qwen/Qwen3.8-Flash-Next",), max_model_len=262144)
    head.stream_chunks = []
    head.stall_after_chunks = 0                      # a status line, then silence: still in flight
    clock = Clock()
    cfg = Config(broker_enabled=True,
                 vllm_boxes=(VllmBox("box-a-gpu", "box-a", A), VllmBox("box-b-gpu", "box-b", B)),
                 listen_host="127.0.0.1", listen_port=0,
                 cluster_name="pair", cluster_hosts=parse_hosts("box-a-gpu=box-a=user@10.0.0.1=10.9.200.1,"
                                           "box-b-gpu=box-b=user@10.0.0.2=10.9.200.2"),
                 cluster_url=head.url, probe_interval_s=0, idle_release_s=600, max_leases=2,
                 connect_timeout_s=2.0, first_byte_timeout_s=3.0, stall_timeout_s=1.0)
    hub = Hub(cfg, clock=clock)
    prober = FakeProber({A: {"live": False, "last_error": "refused"},
                         B: {"live": False, "last_error": "refused"},
                         head.url: {"live": True, "served_models": ("Qwen/Qwen3.8-Flash-Next",),
                                    "max_model_len": 262144, "waiting": 0, "running": 0,
                                    "last_error": ""}})
    ctl = Controller(cfg, hub, BrokerLeases(cfg, clock=clock), prober=prober, clock=clock)
    proxy = Proxy(cfg, hub)
    proxy.start()
    try:
        yield ctl, hub, proxy, clock, head
    finally:
        proxy.stop()
        head.close()


def test_D4_a_stop_with_turns_still_in_flight_on_the_pair_creates_no_vllm_lease(gateway):
    """2026-09-15, 23:23:52,804 → 23:23:54, end to end. A real turn goes to the pair through the
    real proxy and is still in flight when the operator presses Stop; three ticks of the D4 window
    follow, at the deployed tick interval. No client turn arrives after the Stop, so any lease here
    is one the Stop manufactured."""
    ctl, hub, proxy, clock, head = gateway
    _publish(hub, "cluster", held=True)
    ctl.tick()
    assert vllm_leases(hub) == [], "nothing is wanted while the cluster serves"

    turn = threading.Thread(target=_send_a_turn, args=(proxy,), daemon=True)
    turn.start()
    _wait_for_inflight(hub, head.url)
    assert hub.snapshot().last_request_at == clock.t, "the turn ARRIVED before the Stop"

    clock.t += 30
    _publish(hub, "classic")                         # the operator's Stop, from the page
    for _ in range(3):
        clock.t += 2                                 # the deployed tick interval
        ctl.tick()
        assert vllm_leases(hub) == [] and _pending() == 0, \
            "a Stop must not, by itself, manufacture classic demand — the turns it kills are not it"
    assert hub.backend_for(head.url).inflight == 1, "and the turn really was in flight throughout"
    turn.join(timeout=20)
