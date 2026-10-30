"""Pass 15: the routing signal must be true, and failure must be honest and singular.

Production, six attempts, zero output:

    scope (retry 5) — Failed · qwen3.8:latest · attempt 6 (stalled)
    0 tok · 22m 44s ... stalled — no progress for 180000ms
"""
from __future__ import annotations

import http.client
import json
import time

import pytest

from tests.fakes import FakeBackend
from vllm_lb.config import Config, VllmBox
from vllm_lb.hub import Hub
from vllm_lb.probe import parse_vllm_metrics
from vllm_lb.proxy import Proxy
from vllm_lb.router import choose
from vllm_lb.state import LEASE_RUNNING, VLLM, Backend, Lease, Snapshot

NOW = 1000.0

LIVE_METRICS = """\
vllm:num_requests_running{engine="0",model_name="qwen3.8:latest"} 2.0
vllm:num_requests_waiting{engine="0",model_name="qwen3.8:latest"} 5.0
vllm:num_requests_waiting_by_reason{engine="0",model_name="qwen3.8:latest",reason="capacity"} 5.0
vllm:num_requests_waiting_by_reason{engine="0",model_name="qwen3.8:latest",reason="deferred"} 0.0
"""


def test_waiting_is_the_exact_metric_not_a_prefix():
    """Captured verbatim from box-a. `startswith` also matched `_by_reason`, so the LB has
    reported wait=10 where vLLM says 5 — every depth figure in passes 13 and 14 is ~2x off."""
    got = parse_vllm_metrics(LIVE_METRICS)
    assert got["waiting"] == 5 and got["running"] == 2


def v(url, *, waiting=0, running=0, inflight=0, label="box-a-gpu", box="box-a"):
    b = Backend(kind=VLLM, url=url, host_label=label, box=box)
    b.live, b.served_models = True, ("qwen3.8:latest",)
    b.waiting, b.running, b.inflight = waiting, running, inflight
    return b


BOX_A, BOX_B = "http://box-a:8000", "http://box-b:8000"


def snap(backends):
    return Snapshot(now=NOW, backends=list(backends),
                    leases=[Lease(lease_id="r", job_id="j", created_at=NOW,
                                  status=LEASE_RUNNING, host_label="box-a-gpu", box="box-a"),
                            Lease(lease_id="r2", job_id="j2", created_at=NOW,
                                  status=LEASE_RUNNING, host_label="box-b-gpu", box="box-b")],
                    # Both are leased: since 2026-09-12 the router serves ONLY from backends the LB holds
                    # (worklog/vllm-lb-leased-only.md), and these tests are about TIERING, not the lease gate.
                    last_request_at=NOW, last_pressure_at=None, db_ok=True)


def deep():
    """box-b, queued past the limit (2 > 1): DEMOTED — chosen only when nothing has room."""
    return v(BOX_B, running=2, label="box-b-gpu", box="box-b")


def test_one_of_our_own_requests_does_not_demote_an_idle_two_slot_vllm():
    """vLLM's `running` ALREADY counts the request we sent; adding `inflight` counted it twice
    and capped effective concurrency at 1 per box — worse than before depth routing existed.
    (Counted twice, box-a would be as deep as the demoted box-b and lose the tie on `inflight`.)"""
    c = Config(vllm_max_depth=1)
    assert choose(snap([v(BOX_A, running=1, inflight=1), deep()]), c).url == BOX_A
    # and in the stale window, before the probe has caught up, `inflight` still carries it
    assert choose(snap([v(BOX_A, running=0, inflight=1), deep()]), c).url == BOX_A


def test_two_of_our_requests_on_a_two_slot_box_DO_demote_it():
    c = Config(vllm_max_depth=1)
    idle = v(BOX_B, label="box-b-gpu", box="box-b")
    assert choose(snap([v(BOX_A, running=2, inflight=2), idle]), c).url == BOX_B


@pytest.fixture
def stack():
    made = []

    def make(vllm: FakeBackend, others=(), **over):
        """`vllm` on box-a; `others` on box-1, box-2 … — every one leased and live."""
        boxes = [VllmBox("box-a-gpu", "box-a", vllm.url)] + [
            VllmBox(f"box-{i}-gpu", f"box-{i}", o.url) for i, o in enumerate(others, 1)]
        cfg = Config(vllm_boxes=tuple(boxes), listen_host="127.0.0.1", listen_port=0, connect_timeout_s=2.0, **over)
        hub = Hub(cfg)
        for b in (vllm, *others):
            back = hub.backend_for(b.url)
            back.live, back.served_models = True, ("qwen3.8:latest",)
        hub.set_leases([Lease(lease_id=f"r{i}", job_id=f"j{i}", created_at=0, status=LEASE_RUNNING,
                              host_label=b.host_label, box=b.box) for i, b in enumerate(boxes)])
        p = Proxy(cfg, hub)
        p.start()
        made.append((p, hub))
        return p, hub

    yield make
    for p, _ in made:
        p.stop()


def _post(proxy, timeout=60):
    conn = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=timeout)
    conn.request("POST", "/v1/messages", body=json.dumps({"model": "qwen3.8:latest"}).encode(),
                 headers={"Content-Type": "application/json"})
    return conn.getresponse()


def test_a_slow_but_valid_first_token_is_delivered_not_discarded(stack):
    """vLLM sends its status line at HTTP ADMISSION, before engine queueing — so the real wait
    for a first token happens AFTER the headers, where the socket had dropped to the 300 s
    stall budget. The first BODY byte must get the first-byte budget instead."""
    b = FakeBackend()
    b.first_byte_delay_s = 0.0
    b.stream_chunks = [b"data: hello\n\n"]
    b.stream_delay_s = 1.5                      # first token well past a short stall budget
    proxy, _ = stack(b, stall_timeout_s=0.5, first_byte_timeout_s=30.0)
    r = _post(proxy)
    assert r.status == 200
    assert b"hello" in r.read(), "a healthy slow first token must survive"
    b.close()


def test_a_backend_that_never_bodies_is_an_honest_504_sent_exactly_once(stack):
    """Pass 14 failed this over and thereby re-sent a completion a backend may already have
    been generating — its own KILL line. One clear error beats two generations."""
    silent = FakeBackend()
    silent.stream_chunks = []
    silent.stall_after_chunks = 0               # headers, then nothing, forever
    fb = FakeBackend()                          # a second live, leased server it must NOT be re-sent to
    proxy, hub = stack(silent, others=(fb,), stall_timeout_s=0.5, first_byte_timeout_s=2.0)
    other = hub.backend_for(fb.url)
    other.running, other.depth_known = 5, True  # demoted: `silent` is the first choice, fb only a failover
    r = _post(proxy)
    body = r.read()
    assert r.status == 504, f"honest error, got {r.status} {body[:80]!r}"
    assert [q for q in fb.requests if q["method"] == "POST"] == [], \
        "THE KILL LINE: the same completion must never reach two backends"
    silent.close()
    fb.close()


def test_the_honest_504_ends_the_request_it_started_instead_of_leaking_inflight(stack):
    """The 504 above is honest to the CLIENT but not to the fleet: `note_request_start` covers TTFT, and
    this path returns between it and the relay's only `note_request_end`. A backend that answers a status
    line and then produces nothing stayed counted as busy forever, so the router kept steering around a
    depth that no longer existed and the card kept showing it. Found by pass 5's agent reading the relay,
    not by a failure in the wild (worklog/vllm-gateway-program.md, follow-ups)."""
    silent = FakeBackend()
    silent.stream_chunks = []
    silent.stall_after_chunks = 0               # headers, then nothing, forever
    proxy, hub = stack(silent, stall_timeout_s=0.5, first_byte_timeout_s=2.0)
    assert hub.backend_for(silent.url).inflight == 0, "nothing in flight before the turn"
    r = _post(proxy)
    r.read()
    assert r.status == 504
    assert hub.backend_for(silent.url).inflight == 0, \
        "the 504 returned without ending the request it started: this backend is busy forever"
    silent.close()


def test_inflight_covers_the_time_to_first_token(stack):
    """Pass 14 moved note_request_start below the blocking first-body read, so the window
    `inflight` was added to close was the one window it no longer covered."""
    b = FakeBackend()
    b.stream_chunks = [b"data: x\n\n"]
    b.stream_delay_s = 1.0
    proxy, hub = stack(b, stall_timeout_s=5.0, first_byte_timeout_s=30.0)
    import threading
    seen = []

    def watch():
        time.sleep(0.4)                          # inside TTFT, before any body byte
        seen.append(hub.snapshot().backends[0].inflight)

    t = threading.Thread(target=watch)
    t.start()
    _post(proxy).read()
    t.join()
    assert seen and seen[0] >= 1, f"inflight during TTFT was {seen}"
    b.close()
