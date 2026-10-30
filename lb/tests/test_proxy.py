"""S4 — the wire: an always-listening HTTP/1.1 proxy over the router's choice.

Pinned here: passthrough of status/headers/body; SSE streamed as it arrives (not buffered);
failover BEFORE the response headers only (a backend that refuses is penalised and the next
candidate serves); no retry after headers; the model field rewritten on inference paths;
request accounting (inference paths only); the LB's own endpoints; 503 with a JSON body when
nothing is live; 502 when every candidate refused.
"""
from __future__ import annotations

import http.client
import json
import logging
import threading
import time

import pytest

from tests.fakes import FakeBackend
from vllm_lb.config import Config, VllmBox
from vllm_lb.hub import Hub
from vllm_lb.proxy import Proxy
from vllm_lb.state import LEASE_RUNNING, Backend, Lease


def _cfg(vllm: FakeBackend | None, other: FakeBackend | None, **over) -> Config:
    boxes = ((VllmBox("box-a-gpu", "box-a", vllm.url),) if vllm else ()) + (
        (VllmBox("box-b-gpu", "box-b", other.url),) if other else ())
    return Config(broker_enabled=True, vllm_boxes=boxes, listen_host="127.0.0.1", listen_port=0,
                  connect_timeout_s=2.0, first_byte_timeout_s=30.0, penalty_s=30.0, **over)


def _lease_other(hub):
    """box-b's lease (the second server's), as a claim gives it."""
    hub.set_leases([*hub.leases(), Lease(lease_id="r2", job_id="j2", created_at=0, status=LEASE_RUNNING,
                                         host_label="box-b-gpu", box="box-b")])


@pytest.fixture
def stack():
    """A live proxy in a thread, with fakes; the test sets the hub's view of the world."""
    created = []

    def make(*, vllm: FakeBackend | None = None, other: FakeBackend | None = None,
             leased=True, **over):
        """box-a at `vllm` and, given, a second server box-b at `other` — live, one request deeper so box-a is
        tried first, and leased only beside a leased box-a (alone it is live but unheld)."""
        cfg = _cfg(vllm, other, **over)
        hub = Hub(cfg)
        if vllm:
            b = hub.backend_for(vllm.url)
            b.live, b.served_models, b.max_model_len = True, tuple(vllm.served), vllm.max_model_len
            if leased:
                hub.set_leases([Lease(lease_id="r", job_id="j", created_at=0, status=LEASE_RUNNING,
                                      host_label="box-a-gpu", box="box-a")])
                if other:
                    _lease_other(hub)
        if other:
            o = hub.backend_for(other.url)
            o.live, o.served_models, o.waiting = True, tuple(other.served), 1
        proxy = Proxy(cfg, hub)
        proxy.start()
        created.append(proxy)
        return proxy, hub

    yield make
    for p in created:
        p.stop()


def _post(proxy, path, body: dict | bytes, headers=None, timeout=10):
    conn = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=timeout)
    payload = body if isinstance(body, bytes) else json.dumps(body).encode()
    h = {"Content-Type": "application/json", "X-Client": "t"}
    h.update(headers or {})
    conn.request("POST", path, body=payload, headers=h)
    return conn.getresponse()


def _get(proxy, path, timeout=10):
    conn = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=timeout)
    conn.request("GET", path)
    return conn.getresponse()


def _wait_inflight_zero(hub, url, timeout=2.0):
    """The client can see the last byte before the proxy's own bookkeeping runs."""
    deadline = time.time() + timeout
    while hub.backend_for(url).inflight and time.time() < deadline:
        time.sleep(0.02)


# ── passthrough ──────────────────────────────────────────────────────────────

def test_a_request_is_forwarded_and_the_reply_comes_back_intact(stack):
    v = FakeBackend()
    v.reply_json = {"hello": "world"}
    v.extra_headers = {"X-Backend-Trace": "abc"}
    proxy, _ = stack(vllm=v)
    r = _post(proxy, "/v1/messages", {"model": "qwen3.8:latest", "max_tokens": 1})
    assert r.status == 200 and json.loads(r.read()) == {"hello": "world"}
    assert r.getheader("X-Backend-Trace") == "abc"
    assert r.getheader("Content-Type") == "application/json"
    seen = v.requests[-1]
    assert seen["path"] == "/v1/messages" and seen["headers"]["x-client"] == "t"
    assert seen["headers"]["host"] == f"127.0.0.1:{v.port}", "Host is the backend's"
    assert "x-forwarded-for" in seen["headers"]


def test_non_200_status_passes_through(stack):
    v = FakeBackend()
    v.reply_status, v.reply_json = 429, {"error": "slow down"}
    proxy, _ = stack(vllm=v)
    r = _post(proxy, "/v1/chat/completions", {"model": "qwen3.8:latest"})
    assert r.status == 429 and json.loads(r.read()) == {"error": "slow down"}


def _proxy_log_lines():
    """A handler on the proxy's own logger (caplog does not cross the proxy's worker thread via
    root here) and a poller — the final per-request line fires just after the body is sent."""
    records: list[logging.LogRecord] = []
    handler = logging.Handler(level=logging.INFO)
    handler.emit = records.append
    logger = logging.getLogger("vllm_lb.proxy")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)   # the per-request line is INFO; the harness root is WARNING

    def until(substr, timeout=3.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            hit = [r.getMessage() for r in records if substr in r.getMessage()]
            if hit:
                return hit
            time.sleep(0.02)
        return []

    return records, handler, until


def test_a_5xx_upstream_body_is_logged_so_the_failure_is_trackable(stack):
    """The per-request line for an upstream 5xx carries the backend's error body, so a 500 is
    diagnosable from `docker logs vllm-lb` without SSHing the engine box. The 79-byte
    `Internal server error` an SGLang template raise produced — which Claude Code retried 10x
    (`attempt 6/10`) — logged as a bare `500 79B` before this, naming no reason."""
    v = FakeBackend()
    v.reply_status = 500
    v.reply_json = {"type": "error", "error": {"type": "api_error", "message": "Internal server error"}}
    proxy, _ = stack(vllm=v)
    records, handler, until = _proxy_log_lines()
    try:
        r = _post(proxy, "/v1/chat/completions", {"model": "qwen3.8:latest", "max_tokens": 1})
        assert r.status == 500
        r.read()
        hit = until("/v1/chat/completions")
    finally:
        logging.getLogger("vllm_lb.proxy").removeHandler(handler)
    assert hit, [r.getMessage() for r in records]
    assert "500" in hit[-1] and "Internal server error" in hit[-1], hit[-1]


def test_a_2xx_response_body_is_not_logged(stack):
    """Only failing bodies are surfaced — a healthy answer is never copied into the log."""
    v = FakeBackend()
    v.reply_json = {"content": [{"type": "text", "text": "a secret answer"}]}
    proxy, _ = stack(vllm=v)
    records, handler, until = _proxy_log_lines()
    try:
        _post(proxy, "/v1/messages", {"model": "qwen3.8:latest", "max_tokens": 1}).read()
        until("/v1/messages")
    finally:
        logging.getLogger("vllm_lb.proxy").removeHandler(handler)
    assert not any("a secret answer" in r.getMessage() for r in records), [r.getMessage() for r in records]


def test_the_model_is_rewritten_to_what_the_backend_serves(stack):
    v = FakeBackend(served=("qwen3.8:latest",))
    proxy, _ = stack(vllm=v)
    _post(proxy, "/v1/messages", {"model": "qwen3.6", "messages": []}).read()
    sent = json.loads(v.requests[-1]["body"])
    assert sent["model"] == "qwen3.8:latest" and sent["messages"] == []
    assert v.requests[-1]["headers"]["content-length"] == str(len(v.requests[-1]["body"]))


def test_a_non_json_body_is_forwarded_byte_for_byte(stack):
    v = FakeBackend()
    proxy, _ = stack(vllm=v)
    _post(proxy, "/v1/messages", b"not json {", headers={"Content-Type": "text/plain"}).read()
    assert v.requests[-1]["body"] == b"not json {"


def test_count_tokens_is_proxied_like_any_other_path(stack):
    v = FakeBackend()
    v.reply_json = {"input_tokens": 3}
    proxy, _ = stack(vllm=v)
    r = _post(proxy, "/v1/messages/count_tokens", {"model": "x", "messages": []})
    assert json.loads(r.read()) == {"input_tokens": 3}


# ── streaming ────────────────────────────────────────────────────────────────

def test_sse_chunks_arrive_as_the_backend_sends_them(stack):
    v = FakeBackend()
    v.stream_chunks = [b"event: a\ndata: 1\n\n", b"event: b\ndata: 2\n\n"]
    v.stream_delay_s = 1.5
    proxy, _ = stack(vllm=v)
    t0 = time.monotonic()
    r = _post(proxy, "/v1/messages", {"model": "qwen3.8:latest", "stream": True})
    first = r.read1(4096)
    t_first = time.monotonic() - t0
    rest = b""
    while True:
        chunk = r.read1(4096)
        if not chunk:
            break
        rest += chunk
    assert first.startswith(b"event: a"), first
    assert t_first < 1.0, f"first chunk took {t_first:.2f}s — the proxy buffered the stream"
    assert b"event: b" in rest
    assert r.getheader("Content-Type") == "text/event-stream"


def test_a_backend_that_dies_mid_stream_is_not_retried(stack):
    v = FakeBackend()
    v.stream_chunks = [b"data: 1\n\n", b"data: 2\n\n", b"data: 3\n\n"]
    v.drop_after_chunks = 1
    o = FakeBackend()
    proxy, _ = stack(vllm=v, other=o)
    r = _post(proxy, "/v1/messages", {"model": "qwen3.8:latest", "stream": True})
    got = b""
    try:
        while True:
            chunk = r.read1(4096)
            if not chunk:
                break
            got += chunk
    except (http.client.IncompleteRead, ConnectionError, OSError):
        pass
    assert got.startswith(b"data: 1")
    assert not o.requests, "after headers there is no failover — the client retries"


# ── failover ─────────────────────────────────────────────────────────────────

def test_a_refused_backend_fails_over_before_headers_and_is_penalised(stack):
    v = FakeBackend()
    v.down()
    o = FakeBackend()
    o.reply_json = {"from": "box-b"}
    proxy, hub = stack(vllm=v, other=o)
    r = _post(proxy, "/v1/messages", {"model": "qwen3.8:latest"})
    assert json.loads(r.read()) == {"from": "box-b"}
    assert json.loads(o.requests[-1]["body"])["model"] == "qwen3.8:latest"
    b = hub.backend_for(v.url)
    assert b.penalty_until > time.time() and "refused" in b.last_error.lower() or b.last_error


def test_a_backend_that_took_the_body_then_hung_up_is_NOT_retried_elsewhere(stack):
    """After the body is delivered the backend may already be generating; sending the same
    completion to a second backend would run it twice. 502, and the client decides."""
    v = FakeBackend()
    v.drop_before_headers = True
    o = FakeBackend()
    proxy, _ = stack(vllm=v, other=o)
    r = _post(proxy, "/v1/messages", {"model": "qwen3.8:latest"}, timeout=10)
    assert r.status == 502
    body = json.loads(r.read())
    assert body["error"]["type"] == "backend_dropped_request"
    assert len(v.requests) == 1 and not o.requests


def test_nothing_live_answers_503_json_not_a_hang(stack):
    proxy, _ = stack()
    r = _post(proxy, "/v1/messages", {"model": "x"}, timeout=5)
    assert r.status == 503
    body = json.loads(r.read())
    assert body["error"]["type"] == "no_backend"


# ── a claim on its way is waited for, not bounced ────────────────────────────
# After a lease idled out, the next turn found a server live but unheld and got a 503 while broker_parrot
# claimed a fresh lease in seconds (worklog/vllm-fleet-restore.md, "503s after two quiet minutes" — the
# fallback's then, retired 2026-10-17; the same wait for a vLLM's claim). A client that does not retry saw an
# error for a server seconds away.


def test_a_turn_waits_for_the_claim_on_its_way_instead_of_a_503(stack):
    o = FakeBackend()
    proxy, hub = stack(other=o, admission_wait_s=10.0)          # live, not held
    threading.Timer(0.8, _lease_other, args=(hub,)).start()
    t0 = time.monotonic()
    r = _post(proxy, "/v1/messages", {"model": "x"}, timeout=15)
    r.read()
    assert r.status == 200 and len(o.requests) == 1
    assert 0.6 <= time.monotonic() - t0 < 5.0


def test_a_claim_that_does_not_come_is_still_a_503_once_the_wait_is_over(stack):
    o = FakeBackend()
    proxy, _ = stack(other=o, admission_wait_s=0.6)
    t0 = time.monotonic()
    r = _post(proxy, "/v1/messages", {"model": "x"}, timeout=10)
    assert r.status == 503 and json.loads(r.read())["error"]["type"] == "no_backend"
    assert time.monotonic() - t0 >= 0.5 and not o.requests


def test_nothing_a_claim_could_serve_is_refused_at_once(stack):
    proxy, _ = stack(admission_wait_s=30.0)                        # nothing live anywhere
    t0 = time.monotonic()
    r = _post(proxy, "/v1/messages", {"model": "x"}, timeout=10)
    assert r.status == 503 and time.monotonic() - t0 < 2.0


def test_a_broker_we_cannot_reach_is_not_waited_on(stack):
    o = FakeBackend()
    proxy, hub = stack(other=o, admission_wait_s=30.0)
    hub.set_db_state(False, "engine unavailable")
    t0 = time.monotonic()
    r = _post(proxy, "/v1/messages", {"model": "x"}, timeout=10)
    assert r.status == 503 and time.monotonic() - t0 < 2.0


def test_every_candidate_refusing_answers_502_with_who_was_tried(stack):
    v = FakeBackend()
    v.down()
    o = FakeBackend()
    o.down()
    proxy, _ = stack(vllm=v, other=o)
    r = _post(proxy, "/v1/messages", {"model": "x"}, timeout=10)
    assert r.status == 502
    body = json.loads(r.read())
    assert body["error"]["type"] == "all_backends_failed"
    assert set(body["error"]["tried"]) == {v.url, o.url}


# ── accounting + the LB's own endpoints ──────────────────────────────────────

def test_inference_requests_are_counted_and_lb_endpoints_are_not(stack):
    v = FakeBackend()
    proxy, hub = stack(vllm=v)
    assert hub.snapshot().last_request_at is None
    _get(proxy, "/healthz").read()
    _get(proxy, "/v1/models").read()
    _get(proxy, "/lb/status").read()
    assert hub.snapshot().last_request_at is None
    _post(proxy, "/v1/messages", {"model": "qwen3.8:latest"}).read()
    assert hub.snapshot().last_request_at is not None
    _wait_inflight_zero(hub, v.url)
    assert hub.backend_for(v.url).inflight == 0, "decremented after the reply"
    assert hub.snapshot().extra["requests_total"] == 1


def test_inflight_is_held_for_the_duration_of_a_stream(stack):
    v = FakeBackend()
    v.stream_chunks = [b"data: 1\n\n", b"data: 2\n\n"]
    v.stream_delay_s = 0.8
    proxy, hub = stack(vllm=v)
    r = _post(proxy, "/v1/messages", {"model": "qwen3.8:latest", "stream": True})
    r.read1(64)
    assert hub.backend_for(v.url).inflight == 1
    while r.read1(4096):
        pass
    _wait_inflight_zero(hub, v.url)
    assert hub.backend_for(v.url).inflight == 0


def test_healthz_is_always_200_even_with_nothing_live(stack):
    proxy, _ = stack()
    r = _get(proxy, "/healthz")
    assert r.status == 200 and json.loads(r.read())["ok"] is True


def test_v1_models_is_synthesized_from_the_routing_view(stack):
    v = FakeBackend(served=("qwen3.8:latest",), max_model_len=20000)
    proxy, _ = stack(vllm=v)
    data = json.loads(_get(proxy, "/v1/models").read())["data"]
    assert data[0]["id"] == "qwen3.8:latest" and data[0]["max_model_len"] == 20000
    assert not [q for q in v.requests if q["path"] == "/v1/models"], "not proxied"


def test_lb_status_reports_backends_leases_and_clocks(stack):
    v = FakeBackend()
    proxy, hub = stack(vllm=v)
    _post(proxy, "/v1/messages", {"model": "qwen3.8:latest"}).read()
    st = json.loads(_get(proxy, "/lb/status").read())
    assert st["backends"][0]["url"] == v.url and st["backends"][0]["live"] is True
    assert st["leases"][0]["status"] == "running" and st["leases"][0]["box"] == "box-a"
    assert st["last_request_at"] is not None and st["db_ok"] is True
    assert st["desired_leases"] == 1


def test_concurrent_requests_are_served_in_parallel(stack):
    v = FakeBackend()
    v.stream_chunks = [b"data: 1\n\n"]
    v.stream_delay_s = 1.0
    proxy, _ = stack(vllm=v)
    t0 = time.monotonic()
    results = []

    def one():
        r = _post(proxy, "/v1/messages", {"model": "qwen3.8:latest", "stream": True})
        while r.read1(4096):
            pass
        results.append(r.status)

    threads = [threading.Thread(target=one) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert results == [200] * 4
    assert time.monotonic() - t0 < 3.0, "four 1 s streams must not serialise"
