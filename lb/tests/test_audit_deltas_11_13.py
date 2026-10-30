"""Deltas an input-isolated auditor found in passes 11-13. Each one is a real defect it
demonstrated, not a style note.

The headline: pass 13 was reported as a success, and five minutes after its deploy the running
container logged a turn that got a 200 status line and then ZERO body bytes for 300 s with no
failover — a silent empty stream, worse than the 504 it replaced.
"""
from __future__ import annotations

import http.client
import json

import pytest

from tests.fakes import FakeBackend
from vllm_lb.config import Config, VllmBox, parse_boxes
from vllm_lb.hub import Hub
from vllm_lb.probe import probe_backend
from vllm_lb.proxy import Proxy
from vllm_lb.router import choose
from vllm_lb.state import LEASE_RUNNING, VLLM, Backend, Lease, Snapshot

NOW = 1000.0


def v(url, *, waiting=0, running=0, inflight=0, label="box-a-gpu", box="box-a"):
    b = Backend(kind=VLLM, url=url, host_label=label, box=box)
    b.live, b.served_models = True, ("qwen3.8:latest",)
    b.waiting, b.running, b.inflight = waiting, running, inflight
    return b


def idle2():
    """A second leased vLLM, idle with its depth known: the one these tiering tests must prefer."""
    return v("http://box-b:8000", label="box-b-gpu", box="box-b")


def snap(backends, leased=("box-a-gpu", "box-b-gpu")):
    return Snapshot(now=NOW, backends=list(backends),
                    leases=[Lease(lease_id=f"r{i}", job_id=f"j{i}", created_at=NOW,
                                  status=LEASE_RUNNING, host_label=l, box=l.split("-")[0])
                            for i, l in enumerate(leased)],
                    last_request_at=NOW, last_pressure_at=None, db_ok=True)


# ── B: unknown pressure is BUSY, never idle ────────────────────────────────────

def test_a_metrics_failure_does_not_read_as_an_idle_backend():
    """probe_backend seeded waiting/running to 0 and swallowed /metrics errors as 'optional'.
    Pass 13 promoted them to the PRIMARY routing key, so a loaded vLLM whose /metrics times
    out became the shallowest backend alive and won every route — and a loaded box is exactly
    when /metrics is slow."""
    def get(url, timeout):
        if url.endswith("/health"):
            return 200, b""
        if url.endswith("/v1/models"):
            return 200, json.dumps({"data": [{"id": "qwen3.8:latest", "max_model_len": 4096}]}).encode()
        raise TimeoutError("metrics timed out")

    f = probe_backend(Backend(kind=VLLM, url="http://box-a:8000"), Config(), now=NOW, get=get)
    assert f["live"] is True, "the engine is up; only its pressure is unknown"
    assert f.get("depth_known") is False, "the LB must know it does not know"
    assert "metrics" in f["last_error"].lower(), f"must be visible in /lb/status: {f['last_error']!r}"


def test_a_backend_with_unknown_depth_is_not_preferred_over_a_known_idle_one():
    unknown = v("http://box-a:8000")
    unknown.depth_known = False
    s = snap([unknown, idle2()])
    assert choose(s, Config(vllm_max_depth=1)).url == "http://box-b:8000"


# ── C: concurrent admissions inside one probe interval ─────────────────────────

def test_requests_already_sent_count_towards_depth():
    """depth is up to probe_interval_s stale and is the primary sort key, so eight concurrent
    turns all picked the same 2-slot server before the next probe."""
    s = snap([v("http://box-a:8000", running=1, inflight=3), idle2()])
    assert choose(s, Config(vllm_max_depth=1)).url == "http://box-b:8000", \
        "three already in flight means the free slot is long gone"


# ── D: `deep` must not reverse use_unleased_vllm ───────────────────────────────

def test_an_unleased_vllm_stays_excluded_even_when_it_is_the_deep_tier():
    rogue = v("http://rogue:8000", waiting=9, running=2, label="rogue-gpu", box="rogue")
    s = snap([rogue], leased=())
    assert choose(s, Config(broker_enabled=True, use_unleased_vllm=False, vllm_max_depth=1)) is None, \
        "policy must not be escaped by being overloaded"


# ── A: a status line is not a commitment until a body byte arrives ─────────────

@pytest.fixture
def stack():
    made = []

    def make(vllm: FakeBackend, other: FakeBackend | None = None, **over):
        """vllm on box-a; `other` (if given) a second leased vLLM on box-b."""
        boxes = ((VllmBox("box-a-gpu", "box-a", vllm.url),)
                 + ((VllmBox("box-b-gpu", "box-b", other.url),) if other else ()))
        cfg = Config(vllm_boxes=boxes, listen_host="127.0.0.1", listen_port=0, connect_timeout_s=2.0, **over)
        hub = Hub(cfg)
        for b in (x for x in (vllm, other) if x):
            back = hub.backend_for(b.url)
            back.live, back.served_models = True, ("qwen3.8:latest",)
        hub.set_leases([Lease(lease_id=f"r-{box}", job_id=f"j-{box}", created_at=0, status=LEASE_RUNNING,
                              host_label=f"{box}-gpu", box=box) for box in ("box-a", "box-b")[:len(boxes)]])
        p = Proxy(cfg, hub)
        p.start()
        made.append(p)
        return p

    yield make
    for p in made:
        p.stop()


def test_headers_then_silence_is_an_honest_504_and_is_never_re_sent(stack):
    """THE regression, from the deployed container's own log:

        went silent for 300s after 0 bytes (timed out); not retried
        POST /v1/messages?beta=true -> 192.0.2.12:8000 200 0B 300658ms

    REVERSED 2026-09-12 by pass 15 (worklog/vllm-lb-honest-first-byte.md). The silent empty
    200 is still wrong and is still fixed — but the remedy is an honest 504, NOT failover. A
    fresh auditor proved the failover re-sent a completion the backend may already have been
    generating, which was this pass's own KILL line: "primary received the full request body:
    1 time(s) / fallback ALSO received the same completion: 1 time(s)". One clear error beats
    two generations on a shared fleet."""
    silent = FakeBackend()
    silent.stream_chunks = []
    silent.stall_after_chunks = 0          # headers, then nothing, forever
    good = FakeBackend()
    good.reply_json = {"from": "another leased server"}
    proxy = stack(silent, good, stall_timeout_s=1.0, first_byte_timeout_s=20.0)
    proxy.hub.backend_for(good.url).running = 1     # busier, so the silent one is chosen first
    conn = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=60)
    conn.request("POST", "/v1/messages", body=json.dumps({"model": "qwen3.8:latest"}).encode(),
                 headers={"Content-Type": "application/json"})
    r = conn.getresponse()
    body = r.read()
    assert r.status == 504, f"honest error, got {r.status}"
    assert [q for q in good.requests if q["method"] == "POST"] == [], \
        "the same completion must never reach two backends"
    silent.close()
    good.close()


# ── E: the capacity query must be the ENGINE's query ───────────────────────────

CFGL = Config(lease_workflow="vllm_lb_lease", lease_node_module="system.vllm_lease",
             vllm_boxes=parse_boxes("box-a-gpu=box-a=http://192.0.2.10:8000,"
                                   "box-b-gpu=box-b=http://192.0.2.11:8000"))


@pytest.fixture
def leases(broker_db):
    from vllm_lb.leases import BrokerLeases
    return BrokerLeases(CFGL, clock=lambda: NOW)


def test_capacity_counts_one_physical_box_once(leases, broker_db):
    """worker_heartbeats is keyed (host_label, queue, project) and every engine query filters
    on project — including the unassignable sweep leases.py claims to mirror. Ours did not, and
    collected into a list, so one box with two tenant tags read as capacity 2."""
    from queue_workflows import node_queue
    for proj in ("", "other-tenant"):
        node_queue.upsert_worker_heartbeat(host_label="box-a-gpu", queue="gpu", concurrency=1,
                                           llm_servers_available=["vllm"], project=proj)
    n, detail = leases.vllm_capacity()
    assert n == 1, f"one physical box is one box ({detail})"


def test_capacity_ignores_boxes_the_lb_cannot_route_to(leases, broker_db):
    """A lease claimed by a box absent from LB_VLLM_BOXES is held but unusable — the LB has no
    URL for it. Counting it inflates the cap that pass 12 exists to make honest."""
    from queue_workflows import node_queue
    node_queue.upsert_worker_heartbeat(host_label="box-a-gpu", queue="gpu", concurrency=1,
                                       llm_servers_available=["vllm"])
    node_queue.upsert_worker_heartbeat(host_label="box-c-gpu", queue="gpu", concurrency=1,
                                       llm_servers_available=["vllm"])
    n, detail = leases.vllm_capacity()
    assert n == 1, f"box-c-gpu is not in LB_VLLM_BOXES ({detail})"


# ── F: one vLLM lease per box ──────────────────────────────────────────────────

def test_a_duplicate_lease_on_a_box_we_already_hold_is_released():
    """A gpu lane is TWO wide (worker_heartbeats.concurrency = 2). Two lease jobs on one box
    occupy BOTH lanes for up to lease_max_hold_s = 6 h, during which no llm_chat or
    compose_region_report can be claimed there — and the second buys nothing, because that
    box's single vLLM already serves us.

    Removing `avoid_box` was right (a hard claim filter with no fallback strands the row), but
    it means we cannot STEER the second lease. So the guard is reconciliation, not admission:
    when two leases land on the same box, let one go. That can never make a row unclaimable.
    """
    from tests.test_scaler import CFG, NOW as SNOW, lease, snap as ssnap
    from vllm_lb.scaler import Release, plan_actions
    s = ssnap(last_request_at=SNOW, last_pressure_at=SNOW, vllm_capacity=2,
              leases=[lease(1, box="box-a", host_label="box-a-gpu"),
                      lease(2, box="box-a", host_label="box-a-gpu")])
    rel = [a for a in plan_actions(s, CFG) if isinstance(a, Release)]
    assert len(rel) == 1, "exactly one of the two duplicates goes back"


def test_two_leases_on_DIFFERENT_boxes_are_both_kept():
    from tests.test_scaler import CFG, NOW as SNOW, lease, snap as ssnap
    from vllm_lb.scaler import Release, plan_actions
    s = ssnap(last_request_at=SNOW, last_pressure_at=SNOW, vllm_capacity=2,
              leases=[lease(1, box="box-a", host_label="box-a-gpu"),
                      lease(2, box="box-b", host_label="box-b-gpu")])
    assert [a for a in plan_actions(s, CFG) if isinstance(a, Release)] == []
