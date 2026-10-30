"""Every answer says which broker_parrot BOX and SERVER produced it.

The operator: "include details which broker_parrot server, box runs request so we know what's
happening under the hood". The proxy already stamps X-LB-Backend with a URL, but a URL is not
an answer to "which box" — 192.0.2.12 means box-b only if you have the topology memorised.
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

    def make(*vllms, **over):
        """Each FakeBackend becomes a leased classic vLLM: the first on box-b, the next on box-a."""
        names = ("box-b", "box-a")
        boxes = tuple(VllmBox(f"{n}-gpu", n, b.url) for n, b in zip(names, vllms))
        cfg = Config(vllm_boxes=boxes, listen_host="127.0.0.1", listen_port=0, connect_timeout_s=2.0, **over)
        hub = Hub(cfg)
        for b in vllms:
            back = hub.backend_for(b.url)
            back.live, back.served_models = True, ("qwen3.8:latest",)
        hub.set_leases([Lease(lease_id=f"r-{n}", job_id=f"job-{n}", created_at=0, status=LEASE_RUNNING,
                              host_label=f"{n}-gpu", box=n) for n, _ in zip(names, vllms)])
        p = Proxy(cfg, hub)
        p.start()
        made.append(p)
        return p

    yield make
    for p in made:
        p.stop()


def _post(proxy):
    c = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=30)
    c.request("POST", "/v1/messages", body=json.dumps({"model": "qwen3.8:latest"}).encode(),
              headers={"Content-Type": "application/json"})
    r = c.getresponse()
    r.read()
    return r


def test_the_response_says_which_box_served_it(stack):
    b = FakeBackend()
    proxy = stack(b)
    r = _post(proxy)
    assert r.getheader("X-LB-Backend") == b.url
    assert r.getheader("X-LB-Box") == "box-b", "the physical box broker_parrot placed it on"
    assert r.getheader("X-LB-Host-Label") == "box-b-gpu", "what the claim stamps"
    assert r.getheader("X-LB-Kind") == "vllm"
    b.close()


def test_status_reports_a_box_for_every_backend(stack):
    v, v2 = FakeBackend(), FakeBackend()
    proxy = stack(v, v2)
    c = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=10)
    c.request("GET", "/lb/status")
    s = json.loads(c.getresponse().read())
    boxes = {b["url"]: (b["kind"], b["box"]) for b in s["backends"]}
    assert boxes == {v.url: ("vllm", "box-b"), v2.url: ("vllm", "box-a")}, boxes
    v.close()
    v2.close()


@pytest.mark.parametrize("key", ["LB_FALLBACK_BOX_LABEL", "LB_FALLBACK_BOXES"])
def test_the_fallback_box_knobs_are_refused(key):
    """There is no fallback box any more ("If can't run then can't run, no fallback box"): naming
    one refuses the start, never quietly ignored."""
    with pytest.raises(ValueError, match=f"{key} is retired"):
        Config.from_env({"LB_ALLOWED_HOSTS": "loopback", key: "box-b"})
