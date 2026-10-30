"""A backend that answers instantly and always fails is not a backend.

2026-09-12: 30 consecutive 500s in 4-14 ms each, and the LB kept choosing that backend while a
healthy ollama (the fallback, retired since 2026-10-17) sat live — an 18 GB model loaded, a lease held, a second model contending for
the same card, GPU at 0%. The readiness gate cannot see this: /health is 200 because the engine
IS up; it is the REQUEST it refuses.
"""
from __future__ import annotations

import http.client
import json

import pytest

from tests.fakes import FakeBackend
from vllm_lb.config import Config, VllmBox
from vllm_lb.hub import Hub
from vllm_lb.proxy import Proxy
from vllm_lb.state import LEASE_RUNNING, Lease


@pytest.fixture
def stack():
    made = []

    def make(vllm: FakeBackend, other: FakeBackend | None = None, **over):
        """box-a serving at `vllm` and, given, a second leased server (box-b) one request deeper — so box-a is
        chosen while it is not penalised."""
        boxes = (VllmBox("box-a-gpu", "box-a", vllm.url),) + ((VllmBox("box-b-gpu", "box-b", other.url),)
                                                              if other else ())
        cfg = Config(vllm_boxes=boxes, listen_host="127.0.0.1", listen_port=0, connect_timeout_s=2.0,
                     **({"cluster_max_num_batched_tokens": 8192} if "cluster_hosts" in over else {}), **over)
        hub = Hub(cfg)
        for b in (vllm, *([other] if other else [])):
            back = hub.backend_for(b.url)
            back.live, back.served_models = True, ("qwen3.8:latest",)
        if other:
            hub.backend_for(other.url).waiting = 1
        hub.set_leases([Lease(lease_id="r", job_id="j", created_at=0, status=LEASE_RUNNING,
                              host_label="box-a-gpu", box="box-a"),
                        Lease(lease_id="r2", job_id="j2", created_at=0, status=LEASE_RUNNING,
                              host_label="box-b-gpu", box="box-b")])
        proxy = Proxy(cfg, hub)
        proxy.start()
        made.append((proxy, hub))
        return proxy, hub

    yield make
    for p, _ in made:
        p.stop()


def _post(proxy):
    conn = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=30)
    conn.request("POST", "/v1/messages", body=json.dumps({"model": "qwen3.8:latest"}).encode(),
                 headers={"Content-Type": "application/json"})
    r = conn.getresponse()
    r.read()
    return r.status


def test_a_backend_that_only_5xxes_is_taken_out_of_rotation(stack):
    bad = FakeBackend()
    bad.reply_status = 500
    good = FakeBackend()
    good.reply_json = {"from": "the other server"}
    proxy, hub = stack(bad, good, failure_streak_max=3)

    for _ in range(3):
        assert _post(proxy) == 500, "the 5xx is relayed, never invented"
    assert _post(proxy) == 200, "after the streak the turn must go somewhere that works"
    assert [q for q in good.requests if q["method"] == "POST"], "the other server served it"
    bad.close()
    good.close()


def test_one_success_resets_the_streak(stack):
    b = FakeBackend()
    proxy, hub = stack(b, failure_streak_max=2)
    b.reply_status = 500
    _post(proxy)
    b.reply_status = 200
    _post(proxy)
    b.reply_status = 500
    _post(proxy)
    assert not hub.snapshot().backends[0].penalty_until, "a success in between is not a streak"
    b.close()


def test_a_4xx_is_the_clients_fault_and_never_penalises_the_backend(stack):
    b = FakeBackend()
    b.reply_status = 400
    proxy, hub = stack(b, failure_streak_max=2)
    for _ in range(4):
        assert _post(proxy) == 400
    assert not hub.snapshot().backends[0].penalty_until, "a bad request is not a bad backend"
    b.close()


def test_the_last_live_backend_still_serves_even_while_penalised(stack):
    """The frame's KILL condition: an answered-badly turn beats an unanswerable one."""
    b = FakeBackend()
    b.reply_status = 500
    proxy, hub = stack(b, failure_streak_max=2)
    for _ in range(4):
        assert _post(proxy) == 500, "with nowhere else to go it must still be tried"
    b.close()


def test_the_streak_is_visible_in_status(stack):
    b = FakeBackend()
    b.reply_status = 500
    proxy, hub = stack(b, failure_streak_max=5)
    _post(proxy)
    row = [x for x in proxy.status()["backends"] if x["url"] == b.url][0]
    assert row["failure_streak"] == 1
    b.close()
