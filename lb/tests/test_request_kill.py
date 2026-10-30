"""KILL ONE REQUEST: the operator ends a turn from the REQUESTS card — a 1 s hold on its red ×, which posts
`POST /lb/requests/<id>/kill` (JSON, like every LB write route).

Pinned here, through ONE mechanism for every engine path the LB proxies (a classic box and the cluster
head — a pair, a local node and an observed node are all that one cluster backend to the relay):

  * a turn the LB HOLDS (nothing sent to any engine) is answered at once and never sent;
  * a turn SENT and not answered yet (queued or prefilling at the engine) and a STREAMING one: the LB closes its
    connection to the engine — the engine sees its client go, which is what aborts a request in vLLM and SGLang
    (vllm_lb/request_kill.py cites both) — and a stand-in engine here RECORDS that disconnect;
  * the client is told, in its API's own shape, with a status no SDK retries and `x-should-retry: false`: before
    any byte an HTTP error; mid-stream an Anthropic `event: error`, or an OpenAI error chunk + `[DONE]` — never a
    silent hang, never a fake normal finish;
  * an unknown id or one already finished is 404 with why; a second kill is 200 "already killed";
  * the route takes JSON only (the CSRF rule of the other LB write routes); every kill is logged with the id,
    client, state, age, what was produced so far and WHO (the gate's `X-Vllm-User`); the finished row's outcome is
    `killed`; a kill never penalises the backend; and the gate's user header never reaches an engine.
"""
from __future__ import annotations

import http.client
import json
import logging
import select
import socket
import threading
import time

import pytest

from tests.fakes import FakeBackend
from vllm_lb.cluster import parse_hosts
from vllm_lb.config import Config, VllmBox
from vllm_lb.hub import Hub
from vllm_lb.proxy import Proxy
from vllm_lb.state import LEASE_RUNNING, Lease

UA = "claude-cli/2.1.296 (external, cli)"
USER = "someone@example.com"
HOSTS = "box-a-gpu=box-a=user@10.0.0.1=10.9.200.1,box-b-gpu=box-b=user@10.0.0.2=10.9.200.2"
SERVED = "qwen3.8:latest"


def rk():
    """The module under test, imported per test: a RED run fails test by test, not at collection."""
    from vllm_lb import request_kill
    return request_kill


def sse(name: str, obj: dict) -> bytes:
    return f"event: {name}\ndata: {json.dumps(obj)}\n\n".encode()


ANTHROPIC_HEAD = [sse("message_start", {"type": "message_start", "message": {
                      "id": "msg_1", "type": "message", "role": "assistant", "model": SERVED, "content": [],
                      "usage": {"input_tokens": 900, "output_tokens": 0}}}),
                  sse("content_block_start", {"type": "content_block_start", "index": 0,
                                              "content_block": {"type": "thinking", "thinking": ""}})]
ANTHROPIC_DELTA = sse("content_block_delta", {"type": "content_block_delta", "index": 0,
                                              "delta": {"type": "thinking_delta", "thinking": " t"}})
OPENAI_HEAD = [b'data: {"id":"c1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant"}}]}\n\n']
OPENAI_DELTA = b'data: {"id":"c1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"w"}}]}\n\n'


class Engine:
    """A stand-in ENGINE that records its client's disconnect. It reads one request, then either HOLDS — no byte
    back, the way a queued or prefilling request looks from outside — or STREAMS SSE (`head`, then `delta` every
    `every` s, forever), and in both cases watches its socket: the moment its client closes the connection,
    `gone` is set. Anything else it is asked (GET) is a 404."""

    def __init__(self, mode: str, head: list[bytes] = (), delta: bytes = b"", every: float = 0.05):
        self.mode, self.head, self.delta, self.every = mode, list(head), delta, every
        self.requests: list[dict] = []
        self.gone = threading.Event()
        self.gone_at: float | None = None
        self._sock = socket.socket()
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(16)
        self.port = self._sock.getsockname()[1]
        self.url = f"http://127.0.0.1:{self.port}"
        self.served = (SERVED,)
        self._closed = False
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self) -> None:
        while not self._closed:
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _read_request(self, conn) -> dict | None:
        buf = b""
        while b"\r\n\r\n" not in buf:
            part = conn.recv(65536)
            if not part:
                return None
            buf += part
        head, body = buf.split(b"\r\n\r\n", 1)
        lines = head.decode("latin-1").split("\r\n")
        method, path, _ = lines[0].split(" ", 2)
        headers = {k.strip().lower(): v.strip() for k, v in (ln.split(":", 1) for ln in lines[1:] if ":" in ln)}
        n = int(headers.get("content-length") or 0)
        while len(body) < n:
            body += conn.recv(65536)
        return {"method": method, "path": path, "headers": headers, "body": body}

    def _went(self) -> None:
        if self.gone_at is None:
            self.gone_at = time.monotonic()
        self.gone.set()

    def _serve(self, conn) -> None:
        try:
            req = self._read_request(conn)
            if req is None:
                return
            self.requests.append(req)
            if req["method"] != "POST":
                conn.sendall(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                return
            if self.mode == "hold":
                if conn.recv(1) == b"":                  # nothing is ever sent: the only thing to hear is EOF
                    self._went()
                return
            conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n")
            chunks = iter(self.head)
            while True:
                chunk = next(chunks, self.delta)
                conn.sendall(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                ready, _, _ = select.select([conn], [], [], self.every)
                if ready and conn.recv(1) == b"":
                    self._went()
                    return
        except OSError:
            self._went()
        finally:
            conn.close()

    def posts(self) -> list[dict]:
        return [r for r in self.requests if r["method"] == "POST"]

    def close(self) -> None:
        self._closed = True
        self._sock.close()


# ── the stack: one proxy in front of one engine, on each kind of backend the LB routes to ─────────────────

@pytest.fixture
def lb(tmp_path):
    made = []

    def make(engine, *, kind: str = "box", **over):
        base = dict(listen_host="127.0.0.1", listen_port=0, connect_timeout_s=2.0, first_byte_timeout_s=30.0,
                    admission_wait_s=0)
        if kind == "box":
            cfg = Config(vllm_boxes=(VllmBox("box-a-gpu", "box-a", engine.url),), **base, **over)
        else:
            cfg = Config(vllm_boxes=(), cluster_name="pair", cluster_hosts=parse_hosts(HOSTS), cluster_url=engine.url,
                         cluster_state_path=str(tmp_path / "cluster.json"), **base, **over)
        hub = Hub(cfg)
        if kind == "cluster":
            hub.set_cluster({"configured": True, "state": "running", "detail": "", "held": True,
                             "desired": {"mode": "cluster"}, "max_depth": 8})
            hub.update_backend(engine.url, live=True, served_models=engine.served, max_model_len=262144)
        else:
            b = hub.backend_for(engine.url)
            b.live, b.served_models, b.depth_known, b.waiting = True, tuple(engine.served), True, 1
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


def ask(port: int, path: str, body: dict, out: dict, *, headers: dict | None = None) -> threading.Thread:
    """A client turn in a thread: `out` gets its status, headers and whole body (the LB closes after an answer)."""
    def run():
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
            conn.request("POST", path, body=json.dumps(body).encode(),
                         headers={"Content-Type": "application/json", "User-Agent": UA, **(headers or {})})
            r = conn.getresponse()
            out.update(status=r.status, headers={k.lower(): v for k, v in r.getheaders()}, body=r.read())
        except Exception as exc:                                # noqa: BLE001 — the test reads it
            out.update(error=f"{type(exc).__name__}: {exc}")
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


def kill(port: int, rid: str, *, body: bytes = b"{}", ctype: str | None = "application/json",
         headers: dict | None = None, method: str = "POST") -> tuple[int, dict]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    h = {**({"Content-Type": ctype} if ctype else {}), **(headers or {})}
    conn.request(method, f"/lb/requests/{rid}/kill", body=body, headers=h)
    r = conn.getresponse()
    raw = r.read()
    try:
        return r.status, json.loads(raw or b"{}")
    except ValueError:
        return r.status, {"raw": raw.decode("latin-1")}


def wait(cond, timeout: float = 10.0):
    deadline = time.monotonic() + timeout
    while True:
        got = cond()
        if got:
            return got
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.02)


def live_row(proxy: Proxy, state: str | None = None) -> dict:
    return wait(lambda: next((r for r in proxy.queue_view()["inflight"] if state in (None, r["state"])), None))


def finished(proxy: Proxy) -> dict:
    return wait(lambda: (x := proxy.queue_view()["recent"]) and x[0])


def anthropic_body(stream: bool = True) -> dict:
    return {"model": SERVED, "max_tokens": 64, "stream": stream,
            "messages": [{"role": "user", "content": "say hi"}]}


def openai_body(stream: bool = True) -> dict:
    return {"model": SERVED, "stream": stream, "messages": [{"role": "user", "content": "say hi"}]}


def assert_told_not_to_retry(out: dict, api: str) -> dict:
    """The client got the kill as an HTTP error: the chosen status, `x-should-retry: false`, its API's error shape."""
    assert "error" not in out, out
    assert out["status"] == rk().KILL_STATUS, out
    assert out["headers"].get("x-should-retry") == "false", out["headers"]
    body = json.loads(out["body"])
    if api == "anthropic":
        assert body["type"] == "error" and set(body["error"]) == {"type", "message", "details"}, body
        assert body["error"]["type"] == "policy_blocked" and body["error"]["details"] == {"error_code": "dlp_request_denied"}
    else:
        assert set(body["error"]) == {"message", "type", "param", "code"}, body
    assert "killed by the operator" in body["error"]["message"]
    return body


# ── 1. what the client is told, per API: a status no SDK retries ──────────────────────────────────────────

def test_the_kill_status_is_one_no_sdk_and_no_claude_code_retries():
    """anthropic/openai python + TS SDKs retry 408, 409, 429 and every 5xx (`_should_retry` / `shouldRetry`, cited in
    request_kill.py) and obey `x-should-retry`; Claude Code 2.1.296 also retries 401 (and 403 under some auth),
    re-sends a 404 without streaming, and re-runs any other refusal on a fallback model — except a 400 whose type is
    `policy_blocked`, its gateway-refusal contract. Mid-stream it re-sends unless `details.error_code` is
    `dlp_request_denied`."""
    m = rk()
    assert m.KILL_STATUS == 400 and m.NO_RETRY == ("x-should-retry", "false")
    assert m.ANTHROPIC_TYPE == "policy_blocked" and m.NO_RESEND_CODE == "dlp_request_denied"
    assert m.ANTHROPIC_TYPE != "overloaded_error", "the one type every Claude Code path retries"


def test_the_error_is_in_its_apis_own_shape():
    m = rk()
    a = json.loads(m.error_body("/v1/messages", "1a"))
    assert a["type"] == "error" and a["error"]["type"] == m.ANTHROPIC_TYPE and a["request_id"] == "1a"
    assert a["error"]["details"] == {"error_code": m.NO_RESEND_CODE}
    assert "killed by the operator" in a["error"]["message"] and "1a" in a["error"]["message"]
    o = json.loads(m.error_body("/v1/chat/completions", "1a"))
    assert o == {"error": {"message": a["error"]["message"], "type": m.OPENAI_TYPE, "param": None,
                           "code": m.OPENAI_CODE}}
    ev = m.stream_error("/v1/messages", "1a")
    assert ev.startswith(b"event: error\ndata: ") and ev.endswith(b"\n\n")
    assert json.loads(ev.split(b"data: ", 1)[1]) == a
    chunk = m.stream_error("/v1/chat/completions", "1a")
    first, done = chunk.split(b"\n\n")[:2]
    assert json.loads(first[len(b"data: "):]) == o and done == b"data: [DONE]"
    assert chunk.endswith(b"data: [DONE]\n\n")


# ── 2. a turn the LB holds: answered at once, never sent ─────────────────────────────────────────────────────

def test_a_turn_held_for_the_cluster_is_killed_never_sent_and_told_not_to_retry(lb, tmp_path):
    engine = Engine("hold")
    proxy, hub = lb(engine, kind="cluster", cluster_hold_s=60)
    hub.update_backend(engine.url, live=False)
    hub.set_cluster({"configured": True, "state": "starting", "detail": "the pair is starting", "held": False,
                     "desired": {"mode": "cluster"}, "max_depth": 8})
    out: dict = {}
    t = ask(proxy.port, "/v1/messages", anthropic_body(), out)
    turn = wait(lambda: next((x for x in list(proxy._turns) if x.holding), None))
    assert proxy.queue_view()["counts"]["held"] == 1
    t0 = time.monotonic()
    status, got = kill(proxy.port, turn.id, headers={"X-Vllm-User": USER})
    assert status == 200 and got["killed"]["state"] == "held" and got["killed"]["outcome"] == "killed", got
    t.join(5)
    assert time.monotonic() - t0 < 3.0, "a held turn is answered at once, not when its hold runs out"
    assert_told_not_to_retry(out, "anthropic")
    assert engine.posts() == [], "a held turn is never sent to an engine"
    row = finished(proxy)
    assert row["outcome"] == "killed" and row["id"] == turn.id


def test_a_turn_held_by_a_drain_is_killed_at_once_not_answered_draining(lb):
    engine = Engine("sse", ANTHROPIC_HEAD, ANTHROPIC_DELTA)
    proxy, _ = lb(engine)
    first: dict = {}
    t1 = ask(proxy.port, "/v1/messages", anthropic_body(), first)
    r1 = wait(lambda: live_row(proxy, "streaming"))
    threading.Thread(target=proxy.drain, args=(20.0,), daemon=True).start()
    late: dict = {}
    t2 = ask(proxy.port, "/v1/messages", anthropic_body(), late)
    turn = wait(lambda: next((x for x in list(proxy._turns) if x.holding), None))
    t0 = time.monotonic()
    status, got = kill(proxy.port, turn.id)
    assert status == 200 and got["killed"]["state"] == "held", got
    t2.join(5)
    assert time.monotonic() - t0 < 3.0, "answered at once, not when the drain ends"
    assert_told_not_to_retry(late, "anthropic")
    assert len(engine.posts()) == 1, "the held turn was never sent"
    assert kill(proxy.port, r1["id"])[0] == 200, "and the one in flight ends the drain"
    t1.join(5)


# ── 3. a turn at the engine, no byte back: the connection is closed and the engine sees it go ───────────────

@pytest.mark.parametrize("kind", ["box", "cluster"])
def test_a_turn_queued_at_the_engine_is_killed_by_closing_its_connection(lb, kind, caplog):
    caplog.set_level(logging.INFO, logger="vllm_lb.proxy")
    engine = Engine("hold")
    proxy, hub = lb(engine, kind=kind)
    out: dict = {}
    t = ask(proxy.port, "/v1/messages", anthropic_body(), out, headers={"X-Claude-Code-Session-Id": "s" * 36})
    r = live_row(proxy)
    assert r["state"] in ("queued", "prefill", "sent") and len(engine.posts()) == 1
    time.sleep(0.3)
    status, got = kill(proxy.port, r["id"], headers={"X-Vllm-User": USER})
    assert status == 200, got
    k = got["killed"]
    assert k["id"] == r["id"] and k["outcome"] == "killed" and k["state"] == r["state"] and k["by"] == USER
    assert engine.gone.wait(3), "the engine's connection was not closed"
    t.join(5)
    assert_told_not_to_retry(out, "anthropic")
    row = finished(proxy)
    assert (row["outcome"], row["status"]) == ("killed", rk().KILL_STATUS)
    assert USER in row["why"] and r["state"] in row["why"]
    assert len(engine.posts()) == 1, "never re-sent elsewhere"
    b = hub.backend_for(engine.url)
    assert b.penalty_until == 0.0 and b.last_error == "", f"a kill is the operator's, not the backend's failure: {b.last_error}"


def test_a_queued_openai_turn_gets_the_openai_error_shape(lb):
    engine = Engine("hold")
    proxy, _ = lb(engine)
    out: dict = {}
    t = ask(proxy.port, "/v1/chat/completions", openai_body(stream=False), out)
    r = live_row(proxy)
    assert kill(proxy.port, r["id"])[0] == 200
    t.join(5)
    body = assert_told_not_to_retry(out, "openai")
    assert body["error"]["type"] == rk().OPENAI_TYPE and body["error"]["code"] == rk().OPENAI_CODE


# ── 4. a streaming turn: the engine sees its client go, the client gets its stream's own error ──────────────

def test_a_streaming_anthropic_turn_ends_with_an_error_event_never_a_fake_finish(lb):
    engine = Engine("sse", ANTHROPIC_HEAD, ANTHROPIC_DELTA)
    proxy, _ = lb(engine)
    out: dict = {}
    t = ask(proxy.port, "/v1/messages", anthropic_body(), out)
    r = wait(lambda: (x := live_row(proxy, "streaming")) and x["deltas"] >= 3 and x)
    status, got = kill(proxy.port, r["id"])
    assert status == 200 and got["killed"]["state"] == "streaming" and got["killed"]["deltas"] >= 3, got
    assert engine.gone.wait(3), "the engine's connection was not closed"
    t.join(5)
    assert out["status"] == 200, "the stream had begun: the error is IN it"
    events = out["body"].split(b"\n\n")
    assert events[-1] == b"", "the stream ends on a whole event"
    last = events[-2]
    assert last.startswith(b"event: error\ndata: "), last
    err = json.loads(last.split(b"data: ", 1)[1])
    assert err["type"] == "error" and err["error"]["type"] == rk().ANTHROPIC_TYPE
    assert "killed by the operator" in err["error"]["message"]
    assert b"message_stop" not in out["body"] and b"message_delta" not in out["body"], "no fake normal finish"
    row = finished(proxy)
    assert row["outcome"] == "killed" and row["deltas"] >= 3


def test_a_streaming_openai_turn_ends_with_an_error_chunk_then_done(lb):
    engine = Engine("sse", OPENAI_HEAD, OPENAI_DELTA)
    proxy, _ = lb(engine)
    out: dict = {}
    t = ask(proxy.port, "/v1/chat/completions", openai_body(), out)
    r = wait(lambda: (x := live_row(proxy, "streaming")) and x["bytes"] > 400 and x)
    assert kill(proxy.port, r["id"])[0] == 200
    assert engine.gone.wait(3)
    t.join(5)
    assert out["status"] == 200
    events = [e for e in out["body"].split(b"\n\n") if e]
    assert events[-1] == b"data: [DONE]", events[-3:]
    err = json.loads(events[-2][len(b"data: "):])
    assert err["error"]["code"] == rk().OPENAI_CODE and "killed by the operator" in err["error"]["message"]
    assert finished(proxy)["outcome"] == "killed"


# ── 5. ids: unknown, finished, killed twice ──────────────────────────────────────────────────────────────────

def test_an_unknown_id_is_404_with_why(lb):
    proxy, _ = lb(Engine("hold"))
    status, got = kill(proxy.port, "ffff")
    assert status == 404 and got["error"]["type"] == "unknown_request", got
    assert "ffff" in got["error"]["message"]
    status, got = kill(proxy.port, "not-an-id")
    assert status == 404 and got["error"]["type"] == "unknown_request", got


def test_a_finished_id_is_404_naming_how_it_ended(lb):
    v = FakeBackend()
    proxy, _ = lb(v)
    out: dict = {}
    ask(proxy.port, "/v1/messages", anthropic_body(stream=False), out).join(5)
    done = finished(proxy)
    status, got = kill(proxy.port, done["id"])
    assert status == 404 and got["error"]["type"] == "request_finished", got
    assert done["id"] in got["error"]["message"] and "done" in got["error"]["message"]


def test_a_second_kill_is_200_already_killed(lb):
    engine = Engine("hold")
    proxy, _ = lb(engine)
    out: dict = {}
    t = ask(proxy.port, "/v1/messages", anthropic_body(), out)
    r = live_row(proxy)
    assert kill(proxy.port, r["id"])[0] == 200
    t.join(5)
    status, got = kill(proxy.port, r["id"])
    assert status == 200 and got["killed"]["already"] is True and got["killed"]["outcome"] == "killed", got
    assert len(engine.posts()) == 1


# ── 6. the route: JSON only (the CSRF rule), POST only ─────────────────────────────────────────────────────

@pytest.mark.parametrize("ctype", [None, "text/plain", "application/x-www-form-urlencoded", "multipart/form-data",
                                   "text/plain; x=json"])   # still text/plain to a browser: no preflight
def test_a_kill_that_is_not_json_is_refused_and_kills_nothing(lb, ctype):
    engine = Engine("hold")
    proxy, _ = lb(engine)
    out: dict = {}
    ask(proxy.port, "/v1/messages", anthropic_body(), out)
    r = live_row(proxy)
    status, got = kill(proxy.port, r["id"], ctype=ctype)
    assert status == 415 and got["error"]["type"] == "unsupported_media_type", got
    status, got = kill(proxy.port, r["id"], body=b"[1]")
    assert status == 400 and got["error"]["type"] == "invalid_request", got
    assert kill(proxy.port, r["id"], method="GET", ctype=None)[0] in (404, 405)
    time.sleep(0.3)
    assert not engine.gone.is_set() and live_row(proxy)["id"] == r["id"], "nothing was killed"


# ── 7. the log line, and who ─────────────────────────────────────────────────────────────────────────────────

def test_every_kill_is_logged_with_id_client_state_age_progress_and_who(lb, caplog):
    caplog.set_level(logging.INFO, logger="vllm_lb.proxy")
    engine = Engine("sse", ANTHROPIC_HEAD, ANTHROPIC_DELTA)
    proxy, _ = lb(engine)
    out: dict = {}
    t = ask(proxy.port, "/v1/messages", anthropic_body(), out, headers={"X-Claude-Code-Session-Id": "abcdef0123456789"})
    r = wait(lambda: (x := live_row(proxy, "streaming")) and x["deltas"] >= 2 and x)
    assert kill(proxy.port, r["id"], headers={"X-Vllm-User": USER})[0] == 200
    t.join(5)
    (line,) = [rec.getMessage() for rec in caplog.records if "[proxy] kill" in rec.getMessage()]
    for part in (f"request {r['id']}", "streaming", "claude", "session 23456789", "events", f"by {USER}"):
        assert part in line, f"{part!r} not in {line!r}"
    assert " s," in line or " s " in line, f"its age: {line}"

    caplog.clear()
    out2: dict = {}
    t = ask(proxy.port, "/v1/messages", anthropic_body(), out2)
    r2 = wait(lambda: (x := live_row(proxy, "streaming")) and x["id"] != r["id"] and x)
    assert kill(proxy.port, r2["id"])[0] == 200
    t.join(5)
    (line,) = [rec.getMessage() for rec in caplog.records if "[proxy] kill" in rec.getMessage()]
    assert "no X-Vllm-User" in line, f"an unnamed caller is said to be one: {line}"


def test_the_gates_user_header_never_reaches_an_engine(lb):
    v = FakeBackend()
    proxy, _ = lb(v)
    out: dict = {}
    ask(proxy.port, "/v1/messages", anthropic_body(stream=False), out, headers={"X-Vllm-User": USER}).join(5)
    assert out["status"] == 200, out
    (sent,) = [r for r in v.requests if r["method"] == "POST"]
    assert "x-vllm-user" not in sent["headers"] and USER.encode() not in json.dumps(sent["headers"]).encode()
