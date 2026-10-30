"""SILENCE ON A LIVE STREAM IS FILLED, AND A BACKEND THAT VANISHED IS NOTICED.

Measured against Claude Code 2.1.288: once a 200 `text/event-stream` is flowing, 20 s with no BYTES shows
"Waiting for API response · will retry in 29m", and the client's byte-level watchdog aborts after
CLAUDE_STREAM_IDLE_TIMEOUT_MS (≤ 30 min; 180 s for a session without that variable). The silences are
real and healthy: a long prefill before SGLang's adapter emits `message_start`, tool arguments the parser
holds until they are whole. An SSE comment line resets that watchdog and is ignored by every parser.

Pinned here: on `/v1/messages` only, `: keepalive` after `sse_keepalive_s` of backend silence, ONLY at an event boundary — never
inside a half-relayed event — repeated while the silence lasts, never fed to the usage tap, and never
mistaken for progress by the stall budget, which still measures the BACKEND. The upstream socket probes
its peer (TCP keepalive), so a backend that vanished without a FIN — power loss, a pulled cable — is
noticed in about two minutes instead of `stall_timeout_s`; and a backend that fails mid-stream at an
event boundary ends the stream with an `overloaded_error` event, which the client retries.
"""
from __future__ import annotations

import http.client
import json
import socket
import time

import pytest

import vllm_lb.proxy as proxy_mod
from tests.fakes import FakeBackend
from vllm_lb.config import Config, VllmBox
from vllm_lb.hub import Hub
from vllm_lb.proxy import Proxy
from vllm_lb.state import LEASE_RUNNING, Lease

KEEPALIVE = b": keepalive\n\n"


@pytest.fixture
def stack():
    created = []

    def make(v: FakeBackend, **over):
        cfg = Config(vllm_boxes=(VllmBox("box-a-gpu", "box-a", v.url),), 
                     listen_host="127.0.0.1", listen_port=0, connect_timeout_s=2.0,
                     first_byte_timeout_s=30.0, **over)
        hub = Hub(cfg)
        b = hub.backend_for(v.url)
        b.live, b.served_models = True, tuple(v.served)
        hub.set_leases([Lease(lease_id="r", job_id="j", created_at=0, status=LEASE_RUNNING,
                              host_label="box-a-gpu", box="box-a")])
        proxy = Proxy(cfg, hub)
        proxy.start()
        created.append(proxy)
        return proxy

    yield make
    for p in created:
        p.stop()


def stream(proxy: Proxy, path: str = "/v1/messages", timeout: float = 30) -> bytes:
    conn = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=timeout)
    conn.request("POST", path, body=json.dumps({"model": "x", "stream": True}).encode(),
                 headers={"Content-Type": "application/json"})
    r = conn.getresponse()
    got = b""
    try:
        while chunk := r.read1(65536):
            got += chunk
    except (http.client.IncompleteRead, ConnectionError, OSError):
        pass
    return got


def test_a_silent_stream_gets_keepalive_comments_between_events(stack):
    v = FakeBackend()
    v.stream_chunks = [b"event: message_start\ndata: {}\n\n", b"event: ping\ndata: {}\n\n"]
    v.stream_delay_s = 1.0                                      # a second of silence after each event
    got = stream(stack(v, sse_keepalive_s=0.3))
    assert KEEPALIVE in got, "silence on a live stream must be filled"
    assert got.count(KEEPALIVE) >= 2, "and filled again while it lasts"
    assert got.replace(KEEPALIVE, b"") == b"".join(v.stream_chunks), "every backend byte, in order"
    assert got.startswith(v.stream_chunks[0]), "never before the first event is whole"


def test_an_openai_stream_keeps_its_silence_for_its_own_client_to_judge(stack):
    """Only Claude Code's `/v1/messages` gets keepalives: the banner and the watchdog they answer are its.
    vllm-chat streams `/v1/chat/completions` and reads 300 s of silence as a stuck backend; filling that
    silence would leave it to the LB's 1800 s stall budget instead."""
    v = FakeBackend()
    v.stream_chunks = [b"data: {}\n\n", b"data: [DONE]\n\n"]
    v.stream_delay_s = 1.0
    got = stream(stack(v, sse_keepalive_s=0.3), path="/v1/chat/completions")
    assert got == b"".join(v.stream_chunks), got


def test_no_keepalive_is_ever_put_inside_a_half_relayed_event(stack):
    v = FakeBackend()
    v.stream_chunks = [b'event: content_block_delta\ndata: {"a":', b'1}\n\n']   # one event, split
    v.stream_delay_s = 1.0
    got = stream(stack(v, sse_keepalive_s=0.3))
    assert b'event: content_block_delta\ndata: {"a":1}\n\n' in got, got


def test_keepalives_are_not_usage(stack, monkeypatch):
    fed: list[bytes] = []

    class Recording(proxy_mod.UsageTap):
        def feed(self, chunk: bytes) -> None:
            fed.append(chunk)
            super().feed(chunk)

    monkeypatch.setattr(proxy_mod, "UsageTap", Recording)
    v = FakeBackend()
    v.stream_chunks = [b"event: message_start\ndata: {}\n\n", b"event: ping\ndata: {}\n\n"]
    v.stream_delay_s = 1.0
    got = stream(stack(v, sse_keepalive_s=0.3))
    assert KEEPALIVE in got
    assert KEEPALIVE not in b"".join(fed) and b"".join(fed) == b"".join(v.stream_chunks)


def test_keepalives_do_not_hide_a_backend_that_stopped_the_stall_budget_still_ends_it(stack):
    v = FakeBackend()
    v.stream_chunks = [b"event: message_start\ndata: {}\n\n", b"never"]
    v.stall_after_chunks = 1                                    # one event, then silence forever
    t0 = time.monotonic()
    got = stream(stack(v, sse_keepalive_s=0.3, stall_timeout_s=1.5))
    took = time.monotonic() - t0
    assert KEEPALIVE in got
    assert took < 8, f"the stall budget measures the BACKEND's silence, not our writes — took {took:.1f}s"


def test_a_json_answer_is_never_given_a_keepalive(stack):
    v = FakeBackend()
    v.reply_json = {"type": "message", "content": []}
    got = stream(stack(v, sse_keepalive_s=0.1))
    assert json.loads(got) == {"type": "message", "content": []}


def test_a_backend_that_fails_at_an_event_boundary_ends_the_stream_with_an_error_the_client_retries(stack):
    v = FakeBackend()
    v.stream_chunks = [b"event: message_start\ndata: {}\n\n", b"event: ping\ndata: {}\n\n", b"x"]
    v.drop_after_chunks = 2
    got = stream(stack(v))
    assert got.startswith(b"".join(v.stream_chunks[:2]))
    tail = got[len(b"".join(v.stream_chunks[:2])):]
    assert tail.startswith(b"event: error\ndata: ") and tail.endswith(b"\n\n"), tail
    event = json.loads(tail[len(b"event: error\ndata: "):].strip())
    assert event["type"] == "error" and event["error"]["type"] == "overloaded_error"


def test_a_backend_that_fails_inside_an_event_is_cut_with_no_event_glued_onto_it(stack):
    v = FakeBackend()
    v.stream_chunks = [b"event: message_start\ndata: {}\n\n", b'event: content_block_delta\ndata: {"a":', b"1}\n\n"]
    v.drop_after_chunks = 2
    got = stream(stack(v))
    assert got == b"".join(v.stream_chunks[:2]), got


def test_a_stream_that_already_ended_gets_no_error_after_its_last_event(stack):
    """A backend that hangs up after `message_stop` but before the chunked terminator has delivered the whole
    answer. An error event after it would make the client throw a finished turn away and run it again."""
    v = FakeBackend()
    v.stream_chunks = [b"event: message_start\ndata: {}\n\n",
                       b'event: message_stop\ndata: {"type":"message_stop"}\n\n', b"x"]
    v.drop_after_chunks = 2
    got = stream(stack(v))
    assert got == b"".join(v.stream_chunks[:2]), got


def test_the_upstream_socket_probes_a_backend_that_vanished_without_a_word(stack, monkeypatch):
    seen: list[dict] = []
    real = proxy_mod.upstream_keepalive

    def recording(sock):
        real(sock)
        seen.append({"on": sock.getsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE),
                     "idle": sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE),
                     "interval": sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL),
                     "count": sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT)})

    monkeypatch.setattr(proxy_mod, "upstream_keepalive", recording)
    v = FakeBackend()
    stream(stack(v))
    assert seen == [{"on": 1, "idle": 60, "interval": 10, "count": 6}]
    # ~2 min from the last byte to a dead peer reported: far inside the 30-min stall budget, and a
    # live-but-silent backend still ACKs every probe, so a long prefill is never cut by it.
    assert 60 + 10 * 6 <= 180 < Config().stall_timeout_s


def test_the_keepalive_interval_comes_from_the_environment():
    assert Config.from_env({"LB_ALLOWED_HOSTS": "loopback", "LB_SSE_KEEPALIVE_S": "5"}).sse_keepalive_s == 5.0
    assert Config.from_env({"LB_ALLOWED_HOSTS": "loopback", "LB_SSE_KEEPALIVE_S": "0"}).sse_keepalive_s == 0.0
    assert Config().sse_keepalive_s == 15.0, "inside the client's 20 s before its banner"
