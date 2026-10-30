"""Failover is decided by whether the request was DELIVERED — never by the exception type.

`socket.timeout` IS `TimeoutError` on Python 3.10+, so a branch that keys on the exception type
cannot tell "the SYN was dropped, nothing was sent" from "the backend took the whole request and
never answered". Those need opposite handling: the first is safe to retry elsewhere, the second
is not (the backend may already be generating, and re-sending would run the completion twice).

The existing failover test uses a CLOSED port, which refuses instantly — the fast path. A box
that DROPS instead of refusing (firewall, powered-off machine on a blackholing switch, an
unpublished Docker port) is the case that regressed, so these tests use a real blackhole.
"""
from __future__ import annotations

import http.client
import json
import socket
import threading

import pytest

from tests.fakes import FakeBackend
from vllm_lb.config import Config, VllmBox
from vllm_lb.hub import Hub
from vllm_lb.proxy import Proxy
from vllm_lb.state import LEASE_RUNNING, Lease


class Blackhole:
    """A listener whose accept queue is deliberately full, so further SYNs are DROPPED rather
    than refused and `connect()` hangs until the caller's own timeout. This is what a firewalled
    or switched-off box looks like on the wire; a closed port (ConnectionRefused) is not."""

    def __init__(self) -> None:
        self.srv = socket.socket()
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen(1)                      # never accept()ed
        self.port = self.srv.getsockname()[1]
        self._fill: list[socket.socket] = []
        for _ in range(6):                      # overflow the queue so the next SYN is dropped
            s = socket.socket()
            s.settimeout(0.3)
            try:
                s.connect(("127.0.0.1", self.port))
                self._fill.append(s)
            except OSError:
                s.close()
                break

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def close(self) -> None:
        for s in self._fill:
            s.close()
        self.srv.close()


@pytest.fixture
def blackhole():
    b = Blackhole()
    yield b
    b.close()


@pytest.fixture
def stack():
    made = []

    def make(vllm_url: str, other: str | None = None, **over):
        """box-a at `vllm_url` and, given, a second leased server (box-b at `other`) one request deeper, so box-a
        is tried first and box-b is where a failover lands."""
        boxes = (VllmBox("box-a-gpu", "box-a", vllm_url),) + ((VllmBox("box-b-gpu", "box-b", other),) if other else ())
        cfg = Config(vllm_boxes=boxes, listen_host="127.0.0.1", listen_port=0, connect_timeout_s=1.0, **over)
        hub = Hub(cfg)
        for url in [vllm_url] + ([other] if other else []):
            b = hub.backend_for(url)
            b.live, b.served_models = True, ("qwen3.8:latest",)
        if other:
            hub.backend_for(other).waiting = 1
        hub.set_leases([Lease(lease_id="r", job_id="j", created_at=0, status=LEASE_RUNNING,
                              host_label="box-a-gpu", box="box-a"),
                        Lease(lease_id="r2", job_id="j2", created_at=0, status=LEASE_RUNNING,
                              host_label="box-b-gpu", box="box-b")])
        proxy = Proxy(cfg, hub)
        proxy.start()
        made.append(proxy)
        return proxy

    yield make
    for p in made:
        p.stop()


def _post(proxy, timeout=60):
    conn = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=timeout)
    conn.request("POST", "/v1/messages",
                 body=json.dumps({"model": "qwen3.8:latest"}).encode(),
                 headers={"Content-Type": "application/json"})
    return conn.getresponse()


def test_a_blackholed_backend_fails_over_to_the_next_server(blackhole, stack):
    """THE regression: nothing was sent, so the next backend must get the turn. Before this
    fix the client got a 504 while a healthy server sat live and unused."""
    fb = FakeBackend()
    fb.reply_json = {"from": "the other server"}
    proxy = stack(blackhole.url, fb.url, first_byte_timeout_s=30.0)
    r = _post(proxy)
    assert r.status == 200, f"expected failover, got {r.status}"
    assert json.loads(r.read()) == {"from": "the other server"}
    assert len([q for q in fb.requests if q["method"] == "POST"]) == 1
    fb.close()


def test_a_connect_timeout_with_nowhere_left_names_the_CONNECT_budget(blackhole, stack):
    """With nowhere left the client still gets an error — but it must not claim an hour-long
    wait that never happened. The connect budget is the one that expired."""
    proxy = stack(blackhole.url, first_byte_timeout_s=3600.0)
    r = _post(proxy)
    body = json.loads(r.read())
    msg = json.dumps(body)
    assert "3600" not in msg, f"reported a budget that never ran: {msg}"
    assert r.status in (502, 504)


def test_a_delivered_request_that_never_answers_is_a_504_naming_the_first_byte_budget(stack):
    """The other half of the same branch: the backend TOOK the request, so this one is not
    retried elsewhere and the first-byte budget is genuinely the one that expired."""
    b = FakeBackend()
    b.first_byte_delay_s = 60.0
    proxy = stack(b.url, first_byte_timeout_s=2.0)
    r = _post(proxy)
    body = json.loads(r.read())
    assert r.status == 504 and body["error"]["type"] == "backend_timeout"
    assert "2s" in json.dumps(body) or "2 s" in json.dumps(body), json.dumps(body)
    b.close()


def test_a_delivered_request_is_never_re_sent_even_when_another_server_is_free(stack):
    """The KILL condition of this pass: 'not delivered ⇒ fail over' must not reopen the
    re-send hazard fixed in pass 3."""
    b = FakeBackend()
    b.drop_before_headers = True
    fb = FakeBackend()
    proxy = stack(b.url, fb.url, first_byte_timeout_s=30.0)
    r = _post(proxy)
    assert r.status == 502
    assert json.loads(r.read())["error"]["type"] == "backend_dropped_request"
    assert [q for q in fb.requests if q["method"] == "POST"] == [], "it must NOT be re-sent"
    b.close()
    fb.close()
