"""S5 — hermetic end to end: real proxy, real controller, real prober, real lease adapter
on a throwaway sqlite broker; a fake vLLM over real HTTP. The only fakes are the
backends and the clock.

The story the frame requires: first request → refused AND a lease queued (no fallback serves it);
"the engine" claims and the vLLM appears → traffic moves to it; idle past the release window
→ the run is cancelled; the broker DB goes away → proxying continues.
"""
from __future__ import annotations

import http.client
import json

import pytest

from tests.fakes import FakeBackend, free_port
from vllm_lb.config import Config, VllmBox
from vllm_lb.control import Controller
from vllm_lb.hub import Hub
from vllm_lb.leases import BrokerLeases
from vllm_lb.probe import probe_backend
from vllm_lb.proxy import Proxy


class Clock:
    def __init__(self, t=10_000.0):
        self.t = t

    def __call__(self):
        return self.t


def _post(port, body):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request("POST", "/v1/messages", body=json.dumps(body).encode(),
                 headers={"Content-Type": "application/json"})
    r = conn.getresponse()
    return r.status, json.loads(r.read() or b"{}"), r.getheader("X-LB-Backend")


def _status(port):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request("GET", "/lb/status")
    return json.loads(conn.getresponse().read())


@pytest.fixture
def e2e(broker_db):
    vllm_port = free_port()
    cfg = Config(broker_enabled=True, vllm_boxes=(VllmBox("box-a-gpu", "box-a", f"http://127.0.0.1:{vllm_port}"),),
                 listen_host="127.0.0.1", listen_port=0,
                 idle_release_s=600, probe_interval_s=0, probe_timeout_s=0.5,
                 connect_timeout_s=1.0, first_byte_timeout_s=30.0,
                 # this story claims leases BY HAND between turns: a turn nothing holds is refused at once
                 admission_wait_s=0)
    clock = Clock()
    hub = Hub(cfg, clock=clock)
    leases = BrokerLeases(cfg, clock=clock)
    ctl = Controller(cfg, hub, leases, prober=probe_backend, clock=clock)
    proxy = Proxy(cfg, hub)
    proxy.start()
    yield cfg, ctl, hub, proxy, clock, vllm_port
    proxy.stop()


def test_the_whole_story(e2e):
    from queue_workflows import node_queue, run_store
    cfg, ctl, hub, proxy, clock, vllm_port = e2e
    ctl.recover()
    ctl.tick()                                      # nothing asked yet

    # 1. first request: NOTHING is leased yet, so it is REFUSED — and that refusal is itself
    #    the demand signal (worklog/vllm-lb-leased-only.md: no lease, no traffic). The LB now
    #    wants a vLLM — and only that: there is no fallback box to lease.
    from tests.fakes import claim_gpu
    status, body, via = _post(proxy.port, {"model": "qwen3.6", "messages": []})
    assert status == 503, f"unleased servers get no traffic, got {status}"
    ctl.tick()
    st = _status(proxy.port)
    vl = [l for l in st["leases"] if l["kind"] == "vllm"]
    assert st["desired_leases"] == 1 and len(vl) == 1 and vl[0]["status"] == "queued"
    assert [l["kind"] for l in st["leases"]] == ["vllm"], st["leases"]
    assert node_queue.count_pending_for_server(None, "vllm", any_model=True) == 1

    # 2. "the engine": a GB10 box claims the lease and its vLLM comes up
    got = claim_gpu("box-a-gpu")
    assert got["id"] == vl[0]["job_id"]
    vllm = FakeBackend(served=("qwen3.8:latest",), max_model_len=131072, port=vllm_port)
    vllm.reply_json = {"served_by": "vllm"}
    ctl.tick()
    st = _status(proxy.port)
    vl = [l for l in st["leases"] if l["kind"] == "vllm"]
    assert vl[0]["status"] == "running" and vl[0]["box"] == "box-a"
    assert [b for b in st["backends"] if b["kind"] == "vllm"][0]["live"] is True

    # 3. traffic moves to the leased vLLM, model rewritten to what it serves
    status, body, via = _post(proxy.port, {"model": "qwen3.6", "messages": []})
    assert status == 200 and body == {"served_by": "vllm"} and via == vllm.url
    assert json.loads(vllm.requests[-1]["body"])["model"] == "qwen3.8:latest"
    conn = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=5)
    conn.request("GET", "/v1/models")
    models = json.loads(conn.getresponse().read())["data"]
    assert models[0]["id"] == "qwen3.8:latest" and models[0]["max_model_len"] == 131072

    # 4. idle past the release window: the lease is released, its run cancelled
    lease_id = vl[0]["lease_id"]
    clock.t += cfg.idle_release_s + 1
    ctl.tick()
    assert [l for l in _status(proxy.port)["leases"] if l["kind"] == "vllm"] == []
    assert run_store.get_run(lease_id)["status"] == "cancelled"
    assert node_queue.count_pending_for_server(None, "vllm", any_model=True) == 0

    # 5. the broker DB disappears: the tick survives and reports it, rather than crashing.
    #
    #    CHANGED 2026-09-12 by operator instruction (worklog/vllm-lb-leased-only.md). This step
    #    used to assert "vLLM is still live — nothing to do with the DB" and expect a 200. Step
    #    4 released that lease, and an unleased server now gets no traffic however healthy it
    #    looks, so the honest answer is a refusal. Leases we ALREADY hold do survive a DB blip
    #    (`refresh` returns the same list when the DB is unreadable) — what cannot happen while
    #    the broker is unreachable is acquiring a new one.
    import queue_workflows as qw
    from queue_workflows import db
    db.close_pool()
    import os
    os.environ["LB_TEST_DB_DOWN"] = "/nonexistent/dir/broker.db"
    qw.configure(db_backend="sqlite", db_url_env="LB_TEST_DB_DOWN")
    status, body, via = _post(proxy.port, {"model": "x", "messages": []})
    assert status == 503, "the vLLM is live but unleased, and the broker cannot be asked"
    ctl.tick()
    st = _status(proxy.port)
    assert st["db_ok"] is False and st["db_error"]
    assert st["last_tick_error"] == "", "a DB outage is reported, not a crash"
    vllm.down()
