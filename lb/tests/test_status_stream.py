"""/lb/stream: the page's status as Server-Sent Events — the document whole once, then after every control
tick only what changed (vllm_lb/status_delta.py). What this pins down:

  * the reply is an event stream a proxy will not buffer, and its first event is exactly what /lb/status answers;
  * a finished tick sends a `patch`, and that patch applied to the first document is the status after the tick;
  * with no tick the stream still speaks within two tick periods (`tick_s`), so a page can tell a hung loop from a dead stream;
  * a member's stream is the member's narrowed view, never the admin's;
  * a drain ends every stream, so the page reconnects to the next container instead of to nothing.
"""
from __future__ import annotations

import http.client
import json
import threading
import time

import pytest

from tests.fakes import FakeBackend
from vllm_lb import status_delta
from vllm_lb.config import Config, VllmBox
from vllm_lb.hub import Hub
from vllm_lb.proxy import Proxy
from vllm_lb.state import LEASE_RUNNING, Lease

SERVED = "qwen3.8:latest"
MEMBER = {"X-Vllm-Role": "member", "X-Vllm-Models": SERVED}


@pytest.fixture
def lb():
    made = []

    def make(**over):
        engine = FakeBackend(served=(SERVED,))
        cfg = Config(vllm_boxes=(VllmBox("box-a-gpu", "box-a", engine.url),), listen_host="127.0.0.1", listen_port=0,
                     connect_timeout_s=2.0, first_byte_timeout_s=30.0, admission_wait_s=0, **over)
        hub = Hub(cfg)
        b = hub.backend_for(engine.url)
        b.live, b.served_models, b.depth_known, b.waiting = True, (SERVED,), True, 1
        hub.set_leases([Lease(lease_id="r", job_id="j", created_at=0, status=LEASE_RUNNING,
                              host_label="box-a-gpu", box="box-a")])
        proxy = Proxy(cfg, hub)
        proxy.start()
        made.append((proxy, engine))
        return proxy, hub

    yield make
    for proxy, engine in made:
        proxy.stop()
        engine.close()


class Stream:
    """One /lb/stream reader: `next()` is the next event as (name, data), parsed the way EventSource does."""

    def __init__(self, port: int, headers: dict | None = None) -> None:
        self.conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        self.conn.request("GET", "/lb/stream", headers=headers or {})
        self.r = self.conn.getresponse()
        self.retry = None

    def next(self):
        name, data = "message", []
        while True:
            line = self.r.fp.readline()
            if not line:
                return None                                   # the stream ended
            line = line.decode().rstrip("\n")
            if line == "":
                if data:
                    return name, json.loads("\n".join(data))
                continue
            field, _, value = line.partition(":")
            value = value[1:] if value.startswith(" ") else value
            if field == "event":
                name = value
            elif field == "data":
                data.append(value)
            elif field == "retry":
                self.retry = int(value)

    def close(self) -> None:
        self.r.close()                                       # the response holds the socket too
        self.conn.close()


def _get(port: int, path: str, headers: dict | None = None) -> dict:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request("GET", path, headers=headers or {})
    return json.loads(conn.getresponse().read())


def _steady(doc: dict) -> dict:
    """A status without the keys that move with the clock alone — two reads a moment apart agree on the rest."""
    return {k: v for k, v in doc.items() if k not in ("now", "uptime_s", "last_tick_at")}


def test_the_stream_is_an_unbuffered_event_stream_that_opens_with_the_whole_status(lb):
    proxy, _ = lb()
    s = Stream(proxy.port)
    try:
        assert s.r.status == 200
        assert s.r.getheader("Content-Type").startswith("text/event-stream")
        assert s.r.getheader("Cache-Control") == "no-cache"
        assert s.r.getheader("X-Accel-Buffering") == "no"     # nginx in front must pass each event at once
        name, doc = s.next()
        assert name == "status"
        assert s.retry is not None and s.retry > 0          # how soon EventSource comes back after a drop
        assert _steady(doc) == _steady(_get(proxy.port, "/lb/status"))
    finally:
        s.close()


def test_a_tick_sends_the_patch_that_makes_the_next_status(lb):
    proxy, hub = lb(tick_s=15.0)
    s = Stream(proxy.port)
    try:
        _, first = s.next()
        b = hub.backend_for(hub.backends()[0].url)
        with hub._lock:
            b.waiting = 7
        hub.note_tick()
        name, patch = s.next()
        assert name == "patch"
        after = status_delta.apply(patch, first)
        assert after["backends"][0]["waiting"] == 7
        assert _steady(after) == _steady(_get(proxy.port, "/lb/status"))
    finally:
        s.close()


def test_with_no_tick_the_stream_still_speaks_within_its_idle_bound(lb):
    proxy, _ = lb(tick_s=0.15)
    s = Stream(proxy.port)
    try:
        s.next()
        t0 = time.monotonic()
        name, _ = s.next()
        assert name == "patch" and time.monotonic() - t0 < 3.0
    finally:
        s.close()


def test_a_members_stream_is_the_members_view(lb):
    proxy, _ = lb()
    s = Stream(proxy.port, MEMBER)
    try:
        _, doc = s.next()
        assert doc["role"] == "member" and doc["granted_models"] == [SERVED]
        assert _steady(doc) == _steady(_get(proxy.port, "/lb/status", MEMBER))
    finally:
        s.close()


def test_a_drain_ends_every_stream(lb):
    proxy, _ = lb(tick_s=15.0)
    s = Stream(proxy.port)
    try:
        s.next()
        ended = []
        reader = threading.Thread(target=lambda: ended.append(s.next()), daemon=True)
        reader.start()
        proxy.drain(0.0)
        reader.join(5.0)
        assert ended == [None], "the stream must end with the drain, not with the process"
    finally:
        s.close()


def test_streams_share_one_status_per_tick(lb):
    """Each tick's status is BUILT ONCE however many tabs follow it: a build per stream per tick, ~7 ms each
    on a busy gateway, would be multiplied by every open tab."""
    proxy, hub = lb(tick_s=15.0)
    built = []
    real = proxy.status
    proxy.status = lambda: built.append(1) or real()
    streams = [Stream(proxy.port) for _ in range(4)]
    try:
        for s in streams:
            s.next()
        built.clear()
        hub.note_tick()
        patches = [s.next() for s in streams]
        assert all(name == "patch" for name, _ in patches)
        assert len(built) == 1, f"the status was built {len(built)} times for one tick"
    finally:
        for s in streams:
            s.close()


def test_past_the_cap_a_stream_is_refused_loudly(lb):
    """Streams hold a thread each; past STREAM_MAX the next is refused 503 with the reason, never queued or left
    hanging."""
    from vllm_lb import proxy as proxy_mod
    proxy, hub = lb(tick_s=15.0)
    held = [Stream(proxy.port) for _ in range(proxy_mod.STREAM_MAX)]
    try:
        for s in held:
            assert s.next()[0] == "status"
        conn = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=10)
        conn.request("GET", "/lb/stream")
        r = conn.getresponse()
        body = json.loads(r.read())
        assert r.status == 503 and body["error"]["type"] == "too_many_streams"
        assert str(proxy_mod.STREAM_MAX) in body["error"]["message"]
    finally:
        for s in held:
            s.close()
    # a closed tab gives its place back at its stream's next write — the next tick (one a second in a deployment)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        hub.note_tick()
        s = Stream(proxy.port)
        ok = s.r.status == 200
        s.close()
        if ok:
            break
        time.sleep(0.1)
    assert ok


def test_a_stale_reader_never_puts_an_older_frame_back(lb):
    """A stream that read the tick count just before a tick, and takes the frame lock after another stream built the
    newer frame, must get that newer frame — never rebuild and store an older one over it. Otherwise the next tick's
    shared patch is diffed against a document no stream was sent, and the page applies it to the wrong base."""
    proxy, hub = lb(tick_s=15.0)
    hub.note_tick()
    hub.note_tick()
    newer = proxy._tick_frame()
    real = hub.tick_count
    hub.tick_count = lambda: newer[0] - 1                   # the stale reader's view of the count
    try:
        assert proxy._tick_frame() is newer
    finally:
        hub.tick_count = real
    assert proxy._frame is newer
    hub.note_tick()
    seq, doc, patch = proxy._tick_frame()
    assert seq == newer[0] + 1 and patch is not None
    name, _, body = patch.decode().partition("data: ")
    assert status_delta.apply(json.loads(body), json.loads(json.dumps(newer[1]))) == json.loads(json.dumps(doc))


def test_a_drain_before_the_stream_waits_is_never_slept_through(lb):
    """THE LOST WAKE-UP: a drain whose wake-up lands while a stream is still sending its event — before it waits for
    the next tick — must not leave that stream asleep for its idle bound (2 × tick_s). The end flag is checked under
    the same lock the wake-up takes, so a stream that arrives late sees it and does not wait at all."""
    proxy, hub = lb(tick_s=15.0)
    proxy._end_streams()                         # the drain's flag + wake-up, BEFORE anyone waits
    t0 = time.monotonic()
    hub.wait_for_tick(hub.tick_count(), 30.0, stop=proxy._streams_end)
    assert time.monotonic() - t0 < 1.0, "a stream arriving after the drain's wake-up slept through it"
