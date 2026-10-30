"""The relay counts what it relays: one UsageTap per inference response, fed each chunk after it was
written, and exactly one `hub.note_usage` per response — without changing a single relayed byte.

worklog/vllm-gateway-throughput.md, "Proxy token accounting", edge rows 31-34 and the KILL line. A
real Proxy relays a FakeBackend whose stream is the one CAPTURED through the deployed LB from ollama
(fixtures/throughput/anthropic-stream), cut at the captured read boundaries.
"""
from __future__ import annotations

import http.client
import json
import random
import threading
import time
from pathlib import Path

import pytest

from tests.fakes import FakeBackend
from vllm_lb.config import Config, VllmBox
from vllm_lb.hub import Hub
from vllm_lb.proxy import Proxy
from vllm_lb.state import LEASE_RUNNING, Lease
from vllm_lb.usage_tap import Usage, UsageTap

FIXTURES = Path(__file__).parent / "fixtures" / "throughput"
REQUEST = json.dumps({"model": "qwen3.8:latest", "max_tokens": 64, "stream": True,
                      "messages": [{"role": "user", "content": "count"}]}).encode()


def captured_chunks(name: str = "anthropic-stream") -> list[bytes]:
    body = (FIXTURES / f"{name}.body").read_bytes()
    reads = json.loads((FIXTURES / f"{name}.json").read_text())["reads"]
    chunks, offset = [], 0
    for _, size in reads:
        chunks.append(body[offset:offset + size])
        offset += size
    return chunks


@pytest.fixture
def served():
    """A proxy holding a leased, live vLLM backend (a FakeBackend), and its hub."""
    made = []

    def make() -> tuple[Proxy, Hub, FakeBackend]:
        fake = FakeBackend()
        cfg = Config(vllm_boxes=(VllmBox("box-a-gpu", "box-a", fake.url),), listen_host="127.0.0.1",
                     listen_port=0, connect_timeout_s=2.0, first_byte_timeout_s=20.0,
                     stall_timeout_s=10.0, admission_wait_s=0)
        hub = Hub(cfg)
        backend = hub.backend_for(fake.url)
        backend.live, backend.served_models = True, ("qwen3.8:latest",)
        hub.set_leases([Lease(lease_id="r", job_id="j", created_at=0, status=LEASE_RUNNING,
                              host_label="box-a-gpu", box="box-a")])
        proxy = Proxy(cfg, hub)
        proxy.start()
        made.append((proxy, fake))
        return proxy, hub, fake

    yield make
    for proxy, fake in made:
        proxy.stop()
        fake.close()


def send(proxy: Proxy, path: str = "/v1/messages?beta=true") -> tuple[int, bytes]:
    conn = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=20)
    try:
        conn.request("POST", path, body=REQUEST, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        return resp.status, resp.read()
    finally:
        conn.close()


def ours(proxy: Proxy) -> dict:
    (backend,) = proxy.status()["backends"]
    return backend["throughput"]


def eventually(check, timeout_s: float = 10.0):
    """The client can see the last byte a moment before the relay's `finally` has run."""
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            return check()
        except AssertionError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.02)


def test_a_counted_stream_lands_once_with_its_usage(served):
    proxy, _, fake = served()
    fake.stream_chunks = captured_chunks()
    status, body = send(proxy)
    assert status == 200 and body == b"".join(fake.stream_chunks)

    def counted():
        throughput = ours(proxy)
        assert throughput["ours"]["counted"] == 1, throughput["ours"]
        last = throughput["last_request"]
        assert (last["path"], last["prompt_tokens"], last["output_tokens"]) == ("/v1/messages", 30, 64)
        assert last["decode_basis"] in ("LB stream timing", None)
        assert sum(throughput["ours"]["uncounted"].values()) == 0
        # the totals come from the engine's /metrics alone — a relayed stream never adds to them
        assert throughput["totals"]["generated"] == 0 and throughput["totals"]["prompt"] == 0
    eventually(counted)


@pytest.mark.parametrize("seed", range(4))
def test_relayed_bytes_identical_under_fuzz(served, monkeypatch, seed):
    """Row 34 and the KILL line: random garbage spliced into a real stream, cut at random sizes — the
    client receives exactly the backend's bytes. Then again with the tap's parser raising on every
    chunk: the same bytes, and that request is uncounted as a parser error."""
    rng = random.Random(seed)
    real = b"".join(captured_chunks())
    stream, i = b"", 0
    while i < len(real):
        step = rng.randint(1, 900)
        stream += real[i:i + step]
        if rng.random() < 0.3:
            stream += rng.randbytes(rng.randint(1, 400))
        i += step
    chunks, j = [], 0
    while j < len(stream):
        size = rng.choice([1, 7, 100, 1500, 9000])
        chunks.append(stream[j:j + size])
        j += size

    proxy, _, fake = served()
    fake.stream_chunks = chunks
    status, body = send(proxy)
    assert status == 200 and body == stream

    def broken(self, chunk):
        raise RuntimeError("the parser fell over")
    monkeypatch.setattr(UsageTap, "_feed_sse", broken)
    status, body = send(proxy)
    assert status == 200 and body == stream
    eventually(lambda: _assert_uncounted(proxy, "parser_error", 1))


def _assert_uncounted(proxy: Proxy, reason: str, n: int) -> None:
    got = ours(proxy)["ours"]["uncounted"]
    assert got.get(reason) == n, got


def test_client_gone_counts_nothing(served):
    """Row 31. The client hangs up after its first bytes: nothing is counted, and the reason is
    recorded. (vLLM's own /metrics would still count those tokens.)"""
    proxy, _, fake = served()
    fake.stream_chunks = captured_chunks()
    fake.stream_delay_s = 0.05
    conn = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=20)
    conn.request("POST", "/v1/messages", body=REQUEST, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    assert resp.read1(64)
    resp.close()
    conn.close()
    eventually(lambda: _assert_uncounted(proxy, "client_gone", 1))
    assert ours(proxy)["ours"]["counted"] == 0 and ours(proxy)["last_request"] is None


def test_5xx_counts_nothing(served):
    """Row 32. A status other than 200 on an accounted path counts nothing."""
    proxy, _, fake = served()
    fake.reply_status = 500
    fake.reply_json = {"type": "error", "error": {"type": "api_error", "message": "boom"}}
    status, _ = send(proxy, "/v1/messages")
    assert status == 500
    eventually(lambda: _assert_uncounted(proxy, "error_status", 1))
    assert ours(proxy)["ours"]["counted"] == 0


def test_a_backend_that_dies_mid_stream_counts_nothing(served):
    """Row 32. The backend is gone after three chunks: no terminal event, nothing counted."""
    proxy, _, fake = served()
    fake.stream_chunks = captured_chunks()
    fake.drop_after_chunks = 3
    send(proxy)
    eventually(lambda: _assert_uncounted(proxy, "no_terminal_event", 1))
    assert ours(proxy)["ours"]["counted"] == 0


def test_two_interleaved_streams(served):
    """Row 33. Two streams relayed at once: each counted exactly once, by its own tap."""
    proxy, _, fake = served()
    fake.stream_chunks = captured_chunks()
    fake.stream_delay_s = 0.01
    bodies = []
    threads = [threading.Thread(target=lambda: bodies.append(send(proxy)[1])) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    assert bodies == [b"".join(fake.stream_chunks)] * 2

    def both():
        throughput = ours(proxy)
        assert throughput["ours"]["counted"] == 2 and throughput["totals"]["generated"] == 0
    eventually(both)


def test_an_unaccounted_path_is_not_counted_and_not_an_error(served):
    """Row 30 through the relay: count_tokens answers JSON with no usage of a turn."""
    proxy, _, fake = served()
    fake.reply_json = {"input_tokens": 12}
    status, _ = send(proxy, "/v1/messages/count_tokens")
    assert status == 200
    time.sleep(0.2)
    throughput = ours(proxy)
    assert throughput["ours"]["counted"] == 0 and sum(throughput["ours"]["uncounted"].values()) == 0


def test_note_usage_lands_exactly_once_per_response(served, monkeypatch):
    proxy, hub, fake = served()
    verdicts = []
    lock = threading.Lock()

    def note_usage(url, verdict):
        with lock:
            verdicts.append(verdict)
    monkeypatch.setattr(hub, "note_usage", note_usage)
    fake.stream_chunks = captured_chunks()
    send(proxy)                                          # counted
    fake.stream_chunks = None
    send(proxy, "/v1/messages/count_tokens")             # not an accounted shape
    fake.reply_status = 500
    send(proxy, "/v1/messages")                          # an error status
    eventually(lambda: _assert_len(verdicts, 3))
    kinds = sorted(type(v).__name__ if v is not None else "None" for v in verdicts)
    assert kinds == ["None", "Usage", "str"] and "error_status" in verdicts


def _assert_len(items: list, n: int) -> None:
    assert len(items) == n, items
