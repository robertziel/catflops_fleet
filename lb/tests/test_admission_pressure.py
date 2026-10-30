"""Our own queueing must become a lease request.

Live after deploying leased-only routing: `last_pressure_at: None` — pressure had never fired —
while box-b sat at depth 2 (the demotion threshold) and box-a, a capable box, had NO SERVER.
The only pressure source was `waiting` on a leased vLLM, and the router (then) demoted a vLLM to
the fallback before its `waiting` ever reached the threshold. Our queueing showed up as fallback
traffic, never as pressure (the fallback tier is retired since 2026-10-17). The signal that should say "we need another server" was exactly the
event that hid it.
"""
from __future__ import annotations

import http.client
import json

import pytest

from tests.fakes import FakeBackend
from vllm_lb.config import Config, VllmBox
from vllm_lb.hub import Hub
from vllm_lb.proxy import Proxy
from vllm_lb.scaler import desired_leases, pressure_sample
from vllm_lb.state import LEASE_RUNNING, VLLM, Backend, Lease, Snapshot

NOW = 5000.0


# ── the hub counts starved admissions inside a window ──────────────────────────

def test_the_hub_counts_starved_admissions_inside_the_window():
    h = Hub(Config(), clock=lambda: NOW)
    h.note_starved(at=NOW - 45)
    h.note_starved(at=NOW - 10)
    h.note_starved(at=NOW - 2)
    assert h.starved_in(30.0, now=NOW) == 2, "the 45 s-old one has aged out of a 30 s window"
    assert h.snapshot(NOW).extra["starved_recent"] == 2


# ── the sample the tracker sees folds them in ─────────────────────────────────

def _snap(*, waiting=0, starved=0, capacity=2, leases=None):
    v = Backend(kind=VLLM, url="http://box-b:8000", host_label="box-b-gpu", box="box-b")
    v.live, v.waiting = True, waiting
    if leases is None:                      # `waiting` only counts on a LEASED vLLM
        leases = [Lease(lease_id="r", job_id="j", created_at=NOW, status=LEASE_RUNNING,
                        host_label="box-b-gpu", box="box-b", kind=VLLM)]
    return Snapshot(now=NOW, backends=[v], leases=list(leases), last_request_at=NOW,
                    last_pressure_at=None, db_ok=True, vllm_capacity=capacity,
                    extra={"starved_recent": starved})


def test_starved_admissions_count_like_queue_depth():
    cfg = Config(scale_up_waiting=2)
    assert pressure_sample(_snap(waiting=0, starved=2), cfg) == 2


def test_real_queue_depth_still_counts_on_its_own():
    cfg = Config(scale_up_waiting=2)
    assert pressure_sample(_snap(waiting=3, starved=0), cfg) == 3


# ── pressure never exceeds what the broker can start ──────────────────────────

def test_pressure_cannot_ask_for_more_boxes_than_the_broker_reports():
    """THE KILL LINE: the stranded-lease bug of pass 12 must not return through a new door."""
    lease = Lease(lease_id="r", job_id="j", created_at=NOW, status=LEASE_RUNNING,
                  host_label="box-b-gpu", box="box-b", kind=VLLM)
    s = _snap(starved=5, capacity=1, leases=[lease])
    s = Snapshot(**{**s.__dict__, "last_pressure_at": NOW})
    assert desired_leases(s, Config(max_leases=2)) == 1


# ── the proxy decides "starved" where it already knows ────────────────────────

@pytest.fixture
def stack():
    made = []

    def make(vllm, *, leased_vllm=True, capacity=2, **over):
        cfg = Config(broker_enabled=True, vllm_boxes=(VllmBox("box-b-gpu", "box-b", vllm.url),),
                     listen_host="127.0.0.1", listen_port=0, connect_timeout_s=2.0, **over)
        hub = Hub(cfg)
        back = hub.backend_for(vllm.url)
        back.live, back.served_models = True, ("qwen3.8:latest",)
        hub.set_leases([Lease(lease_id="rv", job_id="jv", created_at=0, status=LEASE_RUNNING,
                              host_label="box-b-gpu", box="box-b", kind=VLLM)] if leased_vllm else [])
        hub.set_vllm_capacity(capacity, "test")
        p = Proxy(cfg, hub)
        p.start()
        made.append(p)
        return p, hub

    yield make
    for p in made:
        p.stop()


def _post(proxy):
    c = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=60)
    c.request("POST", "/v1/messages", body=json.dumps({"model": "qwen3.8:latest"}).encode(),
              headers={"Content-Type": "application/json"})
    r = c.getresponse()
    r.read()
    return r.status


def test_a_fast_answer_from_a_leased_vllm_is_not_starved(stack):
    v = FakeBackend()
    proxy, hub = stack(vllm=v)
    assert _post(proxy) == 200
    assert hub.starved_in(60.0) == 0
    v.close()


def test_nothing_leased_at_all_is_starved(stack):
    """Shape 1: the 503 is the strongest demand signal there is."""
    v = FakeBackend()
    proxy, hub = stack(vllm=v, leased_vllm=False, capacity=2, admission_wait_s=0)
    assert _post(proxy) == 503
    assert hub.starved_in(60.0) == 1
    v.close()


def test_a_slow_first_byte_on_a_leased_vllm_is_starved(stack):
    """Shape 3: the operator's own words — waited more than N seconds to even reach inference."""
    v = FakeBackend()
    v.first_byte_delay_s = 1.5
    proxy, hub = stack(vllm=v, starve_after_s=0.5, first_byte_timeout_s=30.0)
    assert _post(proxy) == 200, "it still answered; it just took too long to start"
    assert hub.starved_in(60.0) == 1
    v.close()


def test_the_shipped_trigger_is_depth_1_for_15_seconds():
    """Operator, 2026-09-12: 'Queue depth at least 1 for 15 seconds then ask for support
    additional server broker_parrot.'"""
    c = Config()
    assert c.scale_up_waiting == 1
    assert c.scale_up_sustain_s == 15.0


def test_one_queued_request_for_15_seconds_becomes_pressure():
    from vllm_lb.scaler import PressureTracker
    t = PressureTracker(threshold=Config().scale_up_waiting, sustain_s=Config().scale_up_sustain_s)
    assert t.observe(now=0.0, waiting=1) is None, "not yet — it has to hold"
    assert t.observe(now=14.0, waiting=1) is None
    assert t.observe(now=15.0, waiting=1) == 15.0, "held for the window ⇒ pressure"
    assert t.observe(now=16.0, waiting=0) == 15.0, "and the last time it counted is remembered"
