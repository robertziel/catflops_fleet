"""Timeouts measure PROGRESS, not wall time.

A turn on the shared fallback can legitimately sit for a long time before its first byte —
queued behind a broker_parrot job (ollama serialises per model), waiting for an 18 GB model to
load, or prefilling 200k tokens. None of that is a fault, so the FIRST-BYTE budget is generous.

Once bytes are flowing, silence means something is actually wrong, so the between-chunk
(STALL) budget is short. That is the difference between "waiting" and "stuck", and only the
second one deserves a quick failure.
"""
from __future__ import annotations

import http.client
import json
import threading
import time

import pytest

from tests.fakes import FakeBackend
from vllm_lb.config import Config, VllmBox
from vllm_lb.hub import Hub
from vllm_lb.proxy import Proxy
from vllm_lb.state import LEASE_RUNNING, Lease


@pytest.fixture
def stack():
    made = []

    def make(backend: FakeBackend, **over):
        cfg = Config(vllm_boxes=(VllmBox("box-a-gpu", "box-a", backend.url),), 
                     listen_host="127.0.0.1", listen_port=0, connect_timeout_s=2.0, **over)
        hub = Hub(cfg)
        b = hub.backend_for(backend.url)
        b.live, b.served_models = True, ("qwen3.8:latest",)
        hub.set_leases([Lease(lease_id="r", job_id="j", created_at=0, status=LEASE_RUNNING,
                              host_label="box-a-gpu", box="box-a")])
        proxy = Proxy(cfg, hub)
        proxy.start()
        made.append(proxy)
        return proxy

    yield make
    for p in made:
        p.stop()


def _post(proxy, body, timeout=60):
    conn = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=timeout)
    conn.request("POST", "/v1/messages", body=json.dumps(body).encode(),
                 headers={"Content-Type": "application/json"})
    return conn.getresponse()


def test_config_defaults_and_env():
    c = Config()
    # 1800, not 300, since 2026-10-03. Five minutes of silence was a fault when a turn was a question
    # and an answer; an agentic turn is silent for longer than that while perfectly healthy — a long
    # tool call, a follow-up turn's prefill, an extended reasoning block — and cutting it produced
    # "the response stopped arriving" on a backend that was fine. It is deliberately the same 30 min
    # as the client's own stream-idle watchdog, so neither side gives up before the other.
    assert c.stall_timeout_s == 1800.0, "30 minutes of silence mid-stream is a fault"
    assert c.first_byte_timeout_s >= 1800.0, "queueing + load + prefill must not be a fault"
    e = Config.from_env({"LB_ALLOWED_HOSTS": "loopback", "LB_STALL_TIMEOUT_S": "7", "LB_FIRST_BYTE_TIMEOUT_S": "99"})
    assert e.stall_timeout_s == 7.0 and e.first_byte_timeout_s == 99.0


def test_a_slow_but_progressing_stream_is_never_cut(stack):
    """Chunks slower than the stall budget would be, in total, far past it — but each gap is
    inside it, so the turn completes. This is the long-generation case."""
    b = FakeBackend()
    b.stream_chunks = [f"data: {i}\n\n".encode() for i in range(6)]
    b.stream_delay_s = 0.25
    proxy = stack(b, stall_timeout_s=1.0, first_byte_timeout_s=30.0)
    r = _post(proxy, {"model": "qwen3.8:latest", "stream": True})
    got = b""
    while True:
        chunk = r.read1(4096)
        if not chunk:
            break
        got += chunk
    assert b"data: 5" in got, "a steadily progressing stream must not be cut"


def test_a_stream_that_goes_silent_is_cut_at_the_stall_budget(stack):
    b = FakeBackend()
    b.stream_chunks = [b"data: 1\n\n"]
    b.stall_after_chunks = 1          # headers + one chunk, then silence forever
    proxy = stack(b, stall_timeout_s=1.0, first_byte_timeout_s=30.0)
    t0 = time.monotonic()
    r = _post(proxy, {"model": "qwen3.8:latest", "stream": True})
    got = b""
    try:
        while True:
            chunk = r.read1(4096)
            if not chunk:
                break
            got += chunk
    except (http.client.IncompleteRead, ConnectionError, OSError):
        pass
    took = time.monotonic() - t0
    assert got.startswith(b"data: 1"), "what arrived before the stall is kept"
    assert took < 10, f"a silent stream must end at the stall budget, took {took:.1f}s"


def test_waiting_for_the_first_byte_is_not_a_stall(stack):
    """The backend takes longer than the stall budget to say anything — queued, loading or
    prefilling. That is patience, not a fault: the answer still arrives."""
    b = FakeBackend()
    b.reply_json = {"served": "after a long wait"}
    b.first_byte_delay_s = 2.0
    proxy = stack(b, stall_timeout_s=0.5, first_byte_timeout_s=30.0)
    r = _post(proxy, {"model": "qwen3.8:latest"})
    assert r.status == 200 and json.loads(r.read()) == {"served": "after a long wait"}


def test_a_backend_that_never_answers_ends_at_the_first_byte_budget(stack):
    b = FakeBackend()
    b.first_byte_delay_s = 60.0
    proxy = stack(b, stall_timeout_s=0.5, first_byte_timeout_s=2.0)
    t0 = time.monotonic()
    r = _post(proxy, {"model": "qwen3.8:latest"})
    took = time.monotonic() - t0
    assert r.status == 504
    assert json.loads(r.read())["error"]["type"] == "backend_timeout"
    assert took < 20, f"must end at the first-byte budget, took {took:.1f}s"
