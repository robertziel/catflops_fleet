"""A box with no vLLM running on it is not a box that is DOWN.

Operator: "now thinks box-a is down, it's not." Verified at that moment — box-a had been up
2h57m with nine containers and a load of 0.51, while its `vllm` sidecar showed
`Exited (0) 53 minutes ago` because broker_parrot had released the lease. The LB probed
192.0.2.2:8000, got ConnectionRefused, and the card said DOWN.

That is the NORMAL resting state of this fleet: the broker only runs a vLLM on demand. The two
states need different words, and the LB already has what it takes to tell them apart — the same
`worker_heartbeats` read that produces `vllm_capacity`.
"""
from __future__ import annotations

from vllm_lb.config import Config, VllmBox
from vllm_lb.hub import Hub
from vllm_lb.state import VLLM


# The fleet this file was written against, at the documentation address range (RFC 5737).
BOX_URL = "http://192.0.2.10:8000"
FLEET = (VllmBox("box-a-gpu", "box-a", BOX_URL), VllmBox("box-b-gpu", "box-b", "http://192.0.2.11:8000"))


def _hub():
    return Hub(Config(vllm_boxes=FLEET))


def test_a_backend_starts_with_its_box_liveness_unknown():
    """None is not False: before any heartbeat read we must not claim the box is down."""
    h = _hub()
    assert all(b.box_alive is None for b in h.snapshot().backends)


def test_a_heartbeating_worker_marks_its_box_alive_even_with_the_server_down():
    h = _hub()
    url = BOX_URL
    h.backend_for(url).live = False                  # the sidecar is not running
    h.set_live_boxes({"box-a-gpu"})
    b = [x for x in h.snapshot().backends if x.url == url][0]
    assert b.live is False and b.box_alive is True, \
        "the server is down; the BOX is heartbeating"


def test_a_box_whose_worker_stopped_heartbeating_is_not_alive():
    h = _hub()
    url = BOX_URL
    h.set_live_boxes({"box-b-gpu"})
    assert [x for x in h.snapshot().backends if x.url == url][0].box_alive is False


def test_only_the_lbs_own_servers_are_backends_no_fallback():
    """The fallback tier is retired (2026-10-17): a heartbeat on another lane's label adds no backend and no box."""
    h = _hub()
    h.set_live_boxes({"box-c-gpu"})
    assert all(x.kind == VLLM for x in h.snapshot().backends) and len(h.snapshot().backends) == 2


def test_status_reports_box_liveness_per_backend():
    from vllm_lb.proxy import Proxy
    cfg = Config(vllm_boxes=FLEET, listen_host="127.0.0.1", listen_port=0)
    h = Hub(cfg)
    h.set_live_boxes({"box-a-gpu"})
    rows = {r["box"]: r for r in Proxy(cfg, h).status()["backends"]}
    assert rows["box-a"]["box_alive"] is True
    assert rows["box-b"]["box_alive"] is False
