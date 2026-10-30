"""PARKED is not DOWN, and hardware is proof of life.

Operator, looking at box-a's card while box-a's GPU sat at 93% with a 4.7 s-old sample:

    "Box down or just parked? That's a big difference, we may need different label for that
     and color"

Three different facts had been collapsed into one red `BOX DOWN`:

  * the operator PARKED that lane (`worker_controls.desired_state = 'off'`) — deliberate, the
    broker will start nothing there, and nothing is wrong;
  * the machine is up but no gpu worker is heartbeating — something to look at;
  * the machine is not answering at all — actually down.

Since the LB now hears every box's hardware broadcast, the third case can be told from the
first two by evidence rather than by inference: a box sending a fresh sample is NOT down,
whatever the broker's heartbeat table says.
"""
from __future__ import annotations

import http.client
import json

import pytest

from vllm_lb.config import Config, VllmBox
from vllm_lb.hub import Hub
from vllm_lb.hw import HwCollector
from vllm_lb.proxy import Proxy
from vllm_lb.state import VLLM

URL = "http://box-a:8000"
CFG = Config(vllm_boxes=(VllmBox("box-a-gpu", "box-a", URL),), 
             hw_hosts={"box-a-host": "box-a"}, listen_host="127.0.0.1", listen_port=0)


def _status(hub, hw=None):
    proxy = Proxy(CFG, hub, hw=hw)
    proxy.start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=10)
        conn.request("GET", "/lb/status")
        return json.loads(conn.getresponse().read())
    finally:
        proxy.stop()


def _backend(body):
    return [b for b in body["backends"] if b["kind"] == VLLM][0]


def test_a_parked_box_says_so_instead_of_looking_dead():
    hub = Hub(CFG)
    hub.set_parked_boxes({"box-a-gpu"})
    assert _backend(_status(hub))["parked"] is True


def test_an_unparked_box_is_not_flagged():
    hub = Hub(CFG)
    hub.set_parked_boxes(set())
    assert _backend(_status(hub))["parked"] is False


def test_park_state_is_unknown_before_the_control_plane_is_read():
    """None, not False. Claiming "not parked" before looking is the same class of lie as
    claiming a box is down before looking — `box_alive` already draws that line."""
    assert _backend(_status(Hub(CFG)))["parked"] is None


def test_a_box_broadcasting_hardware_is_alive_however_the_heartbeat_table_looks():
    """The evidence that settles DOWN vs PARKED. box-a was rendered BOX DOWN while its GPU ran
    at 93% and its sample was 4.7 s old."""
    hub = Hub(CFG)
    hub.set_live_boxes(set())                     # no gpu worker heartbeating → box_alive False
    hw = HwCollector(CFG)
    hw.ingest(json.dumps({"host": "box-a-host", "cpu_percent": 8.0,
                          "gpus": [{"id": 0, "use_pct": 93}]}))
    body = _status(hub, hw=hw)
    assert _backend(body)["box_alive"] is False, "the broker's view is unchanged and still shown"
    assert body["hardware"]["box-a"]["stale"] is False, "…but the machine is demonstrably alive"
    assert body["hardware"]["box-a"]["gpu_pct"] == 93
