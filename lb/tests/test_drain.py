"""A REDEPLOY DRAINS: SIGTERM lets the turns in flight finish instead of cutting them.

Evidence (transcript audit, 2026-10-01/02): clusters of 2-7 Claude Code sessions lost their stream —
"Connection lost mid-response" — in the same second, 1-2 s after an LB redeploy (`docker compose up -d`
recreates the container): 29 visible, 17 more inside WebFetch. SIGTERM set the control loop's stop,
`proxy.stop()` closed the listener and the process returned, and every handler thread — each one a
client's stream — died with it as a daemon.

Pinned here: a turn in flight streams to its end; a turn that ARRIVES during the drain is held (no
byte sent) and answered 503 `draining` with a `Retry-After` the moment the drain is over, so it costs
the client one retry instead of its whole retry budget; the drain is bounded and names what it waits
on; a second signal ends it at once; compose's `stop_grace_period` outlasts it.
"""
from __future__ import annotations

import http.client
import json
import logging
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from tests.fleet_files import one_box
from tests.fakes import FakeBackend, free_port
from vllm_lb.cluster import parse_hosts
from vllm_lb.config import Config, VllmBox
from vllm_lb.hub import Hub
from vllm_lb.proxy import Proxy
from vllm_lb.state import LEASE_RUNNING, Lease

LB = Path(__file__).resolve().parent.parent
HOSTS = "box-a-gpu=box-a=user@10.0.0.1=10.9.200.1,box-b-gpu=box-b=user@10.0.0.2=10.9.200.2"
CHUNKS = [f"event: delta\ndata: {i}\n\n".encode() for i in range(8)]


@pytest.fixture
def stack():
    created = []

    def make(vllm: FakeBackend, **over):
        cfg = Config(vllm_boxes=(VllmBox("box-a-gpu", "box-a", vllm.url),), 
                     listen_host="127.0.0.1", listen_port=0, connect_timeout_s=2.0,
                     first_byte_timeout_s=30.0, **over)
        hub = Hub(cfg)
        b = hub.backend_for(vllm.url)
        b.live, b.served_models = True, tuple(vllm.served)
        hub.set_leases([Lease(lease_id="r", job_id="j", created_at=0, status=LEASE_RUNNING,
                              host_label="box-a-gpu", box="box-a")])
        proxy = Proxy(cfg, hub)
        proxy.start()
        created.append(proxy)
        return proxy, hub

    yield make
    for p in created:
        p.stop()


def _post(port: int, out: dict, key: str, timeout: float = 30) -> None:
    """One POST in its own thread: status, headers, the whole body, and WHEN the last byte arrived."""
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
        conn.request("POST", "/v1/messages", body=json.dumps({"model": "x", "stream": True}).encode(),
                     headers={"Content-Type": "application/json"})
        r = conn.getresponse()
        body = r.read()
        out[key] = {"status": r.status, "body": body, "retry_after": r.getheader("Retry-After"),
                    "done": time.monotonic()}
    except Exception as exc:                                    # noqa: BLE001 — the test reads it
        out[key] = {"error": f"{type(exc).__name__}: {exc}", "done": time.monotonic()}


def _wait(cond, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.02)


# ── the drain, in process ─────────────────────────────────────────────────────

def test_a_stream_in_flight_finishes_and_a_turn_arriving_meanwhile_is_answered_draining_once_it_has(stack):
    v = FakeBackend()
    v.stream_chunks, v.stream_delay_s = CHUNKS, 0.25            # ~2 s of stream
    proxy, _ = stack(v, drain_retry_after_s=5)
    out: dict = {}
    threading.Thread(target=_post, args=(proxy.port, out, "streaming"), daemon=True).start()
    _wait(lambda: len(v.requests) == 1)
    drained: list = []
    threading.Thread(target=lambda: drained.append(proxy.drain(20.0)), daemon=True).start()
    time.sleep(0.2)
    threading.Thread(target=_post, args=(proxy.port, out, "late"), daemon=True).start()
    _wait(lambda: drained and "late" in out and "streaming" in out)

    assert out["streaming"]["status"] == 200
    assert out["streaming"]["body"] == b"".join(CHUNKS), "the stream in flight must reach its end"
    late = out["late"]
    assert late["status"] == 503 and json.loads(late["body"])["error"]["type"] == "draining"
    assert late["retry_after"] == "5"
    # HELD, not bounced: answered only once the stream it waited behind was over.
    assert late["done"] >= out["streaming"]["done"] - 0.05
    assert len(v.requests) == 1, "a turn that arrived during the drain never reaches a backend"
    assert drained == [0], "nothing was cut"


def test_the_drain_is_bounded_by_its_window_and_names_what_it_cut(stack, caplog):
    v = FakeBackend()
    v.stream_chunks, v.stall_after_chunks = CHUNKS, 1           # one chunk, then silence
    proxy, _ = stack(v)
    out: dict = {}
    threading.Thread(target=_post, args=(proxy.port, out, "stuck", 30), daemon=True).start()
    _wait(lambda: len(v.requests) == 1)
    time.sleep(0.2)
    caplog.set_level(logging.INFO, logger="vllm_lb.proxy")
    t0 = time.monotonic()
    cut = proxy.drain(0.8)
    assert cut == 1
    assert 0.7 <= time.monotonic() - t0 < 3.0
    text = caplog.text
    assert "POST /v1/messages" in text and v.url in text, "the drain must say what it waits on"


def test_with_nothing_in_flight_the_drain_ends_at_once(stack):
    proxy, _ = stack(FakeBackend())
    t0 = time.monotonic()
    assert proxy.drain(30.0) == 0
    assert time.monotonic() - t0 < 1.0


def test_a_turn_held_for_the_cluster_when_the_drain_begins_is_answered_draining_not_after_its_hold(tmp_path):
    head = FakeBackend(served=("Qwen/Qwen3.8-Flash-Next",))
    cfg = Config(vllm_boxes=(), listen_host="127.0.0.1", listen_port=0,
                 cluster_name="pair", cluster_hosts=parse_hosts(HOSTS), cluster_url=head.url,
                 cluster_state_path=str(tmp_path / "cluster.json"), cluster_hold_s=60.0, drain_retry_after_s=5)
    hub = Hub(cfg)
    hub.set_cluster({"configured": True, "state": "starting", "detail": "launching", "held": False,
                     "desired": {"mode": "cluster"}, "max_depth": 8})
    proxy = Proxy(cfg, hub)
    proxy.start()
    try:
        out: dict = {}
        threading.Thread(target=_post, args=(proxy.port, out, "held"), daemon=True).start()
        time.sleep(0.5)
        assert "held" not in out, "the pair is coming: the turn waits for it"
        t0 = time.monotonic()
        assert proxy.drain(10.0) == 0, "a held turn has no backend: nothing is in flight"
        _wait(lambda: "held" in out, timeout=5)
        assert out["held"]["status"] == 503 and out["held"]["retry_after"] == "5"
        assert json.loads(out["held"]["body"])["error"]["type"] == "draining"
        assert out["held"]["done"] - t0 < 3.0
        assert not head.requests
    finally:
        proxy.stop()


def test_a_turn_that_reaches_the_cluster_hold_just_as_the_drain_begins_does_not_stall_the_drain(tmp_path, monkeypatch):
    """The race: admitted before the drain (so it counts as in flight), it asks to be held after it began. It
    must leave the drain's count as it starts waiting for the drain's end — or each waits for the other until
    the window runs out."""
    head = FakeBackend(served=("Qwen/Qwen3.8-Flash-Next",))
    cfg = Config(vllm_boxes=(), listen_host="127.0.0.1", listen_port=0,
                 cluster_name="pair", cluster_hosts=parse_hosts(HOSTS), cluster_url=head.url,
                 cluster_state_path=str(tmp_path / "cluster.json"))
    hub = Hub(cfg)
    hub.set_cluster({"configured": True, "state": "starting", "detail": "launching", "held": False,
                     "desired": {"mode": "cluster"}, "max_depth": 8})
    proxy = Proxy(cfg, hub)
    in_claim, go_on = threading.Event(), threading.Event()

    def slow_claim(snap, excluded, turn, route=None):   # `turn`: a kill ends a claim wait (test_request_kill.py)
        in_claim.set()
        go_on.wait(10)
        return None

    monkeypatch.setattr(proxy, "_await_claim", slow_claim)
    proxy.start()
    try:
        out: dict = {}
        threading.Thread(target=_post, args=(proxy.port, out, "late"), daemon=True).start()
        assert in_claim.wait(5)
        drained: list = []
        threading.Thread(target=lambda: drained.append(proxy.drain(10.0)), daemon=True).start()
        time.sleep(0.3)
        t0 = time.monotonic()
        go_on.set()
        _wait(lambda: drained and "late" in out, timeout=12)
        assert drained == [0] and time.monotonic() - t0 < 3.0
        assert out["late"]["status"] == 503 and json.loads(out["late"]["body"])["error"]["type"] == "draining"
    finally:
        go_on.set()
        proxy.stop()


# ── the real process: SIGTERM ─────────────────────────────────────────────────

def _boot(fake: FakeBackend, **env_over) -> tuple[subprocess.Popen, int]:
    port = free_port()
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": "/app",                       # no broker engine: the probe phase alone runs
        "LB_LISTEN": f"127.0.0.1:{port}",
        "LB_ALLOWED_HOSTS": "loopback",
        "LB_FLEET": one_box(fake.url),
        "LB_USE_UNLEASED_VLLM": "1",                # no broker, so no lease: route to it unleased
        "LB_PROBE_INTERVAL_S": "0.2",
        "LB_PROBE_TIMEOUT_S": "1",
        "LB_TICK_S": "0.1",
        "LB_PERF": "0",
        "LB_HW_PORT": "0",
        "LB_DB_BACKEND": "sqlite",
        "LB_DB_URL_ENV": "LB_SQLITE",
        "LB_SQLITE": f"/tmp/drain-test-{port}.db",
        **env_over,
    }
    proc = subprocess.Popen([sys.executable, "-m", "vllm_lb"], env=env, cwd=str(LB),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    deadline = time.monotonic() + 10
    while True:
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
            conn.request("GET", "/lb/status")
            st = json.loads(conn.getresponse().read())
            if any(b.get("live") for b in st.get("backends", [])):
                return proc, port
        except OSError:
            pass
        if time.monotonic() > deadline:
            proc.kill()
            raise AssertionError("the LB never saw its backend live:\n"
                                 + proc.communicate(timeout=5)[0].decode(errors="replace"))
        time.sleep(0.1)


def test_sigterm_lets_a_stream_in_flight_finish_and_then_the_process_exits():
    v = FakeBackend()
    v.stream_chunks, v.stream_delay_s = CHUNKS, 0.3             # ~2.4 s of stream
    proc, port = _boot(v)
    try:
        out: dict = {}
        threading.Thread(target=_post, args=(port, out, "streaming"), daemon=True).start()
        _wait(lambda: len([r for r in v.requests if r["method"] == "POST"]) == 1)
        time.sleep(0.3)
        proc.send_signal(signal.SIGTERM)
        time.sleep(0.4)
        threading.Thread(target=_post, args=(port, out, "late"), daemon=True).start()
        code = proc.wait(timeout=20)
        _wait(lambda: "streaming" in out and "late" in out, timeout=5)
    finally:
        if proc.poll() is None:
            proc.kill()
        log = proc.communicate(timeout=5)[0].decode(errors="replace")
    assert out["streaming"].get("body") == b"".join(CHUNKS), (out["streaming"], log)
    assert out["late"].get("status") == 503, (out["late"], log)
    assert json.loads(out["late"]["body"])["error"]["type"] == "draining"
    assert out["late"]["retry_after"] and int(out["late"]["retry_after"]) > 0
    assert code == 0, log
    assert "draining" in log and "Traceback" not in log, log


def test_a_second_signal_ends_the_drain_at_once():
    v = FakeBackend()
    v.stream_chunks, v.stall_after_chunks = CHUNKS, 1           # a turn that will not end on its own
    proc, port = _boot(v)
    try:
        out: dict = {}
        threading.Thread(target=_post, args=(port, out, "stuck"), daemon=True).start()
        _wait(lambda: len([r for r in v.requests if r["method"] == "POST"]) == 1)
        time.sleep(0.3)
        proc.send_signal(signal.SIGTERM)
        time.sleep(1.0)
        assert proc.poll() is None, "the first signal drains: a turn is in flight"
        proc.send_signal(signal.SIGTERM)
        t0 = time.monotonic()
        proc.wait(timeout=10)
        assert time.monotonic() - t0 < 5.0
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.communicate(timeout=5)


# ── the window, and compose ───────────────────────────────────────────────────

def test_the_drain_window_comes_from_the_environment():
    cfg = Config.from_env({"LB_ALLOWED_HOSTS": "loopback", "LB_DRAIN_S": "120", "LB_DRAIN_RETRY_AFTER_S": "3"})
    assert cfg.drain_s == 120.0 and cfg.drain_retry_after_s == 3.0
    assert Config.from_env({"LB_ALLOWED_HOSTS": "loopback"}).drain_s == Config().drain_s == 900.0


def _seconds(text: str) -> float:
    m = re.fullmatch(r"(?:(\d+)m)?(?:(\d+)s)?", text.strip())
    assert m and any(m.groups()), text
    return int(m.group(1) or 0) * 60 + int(m.group(2) or 0)


def test_the_compose_grace_period_outlasts_the_drain():
    """Docker SIGKILLs a container `stop_grace_period` after the SIGTERM — 10 s by default. Without a
    longer one the drain is decoration: every redeploy still cuts every stream at 10 s."""
    compose = (LB / "docker-compose.yml").read_text()
    service = compose.split("\n  searxng:")[0]
    grace = re.search(r"^\s+stop_grace_period:\s*(\S+)", service, re.M)
    drain = re.search(r"LB_DRAIN_S=\$\{LB_DRAIN_S:-(\d+)\}", service)
    assert grace, "no stop_grace_period: Docker's 10 s default SIGKILLs the drain"
    assert drain, "LB_DRAIN_S is not set in compose: the grace period has nothing to outlast by contract"
    assert int(drain.group(1)) == Config().drain_s
    assert _seconds(grace.group(1)) >= int(drain.group(1)) + 30, (grace.group(1), drain.group(1))
