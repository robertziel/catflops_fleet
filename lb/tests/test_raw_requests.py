"""Replies to request lines no browser sends are still replies the LB writes.

`http.server` reads a request line without a valid HTTP/1.x version as HTTP/0.9, and in that state
`send_response`, `send_header` and `end_headers` write NOTHING: a 400/505 error page, or the gateway page
itself, went out with no status line and none of the LB's own headers — the second independent audit
of worklog/vllm-gateway-deep-audit.md (G5). http.client cannot send these, so a raw socket does.
"""
from __future__ import annotations

import socket

import pytest

from tests.test_static_ui import OWN_HEADERS
from tests.test_static_ui import proxy  # noqa: F401 — the same in-process gateway fixture


def raw(proxy, request: bytes) -> bytes:
    with socket.create_connection(("127.0.0.1", proxy.port), timeout=10) as s:
        s.sendall(request)
        s.shutdown(socket.SHUT_WR)
        chunks = []
        while True:
            chunk = s.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
    return b"".join(chunks)


def host(proxy) -> bytes:
    """The Host line of a request to the in-process gateway: its own loopback name at its bound port — the one name
    it answers to (vllm_lb/host_guard.py). A request without it is answered 421, which would test only that."""
    return b"Host: 127.0.0.1:%d\r\n" % proxy.port


def head_of(reply: bytes) -> tuple[str, dict[str, str]]:
    head = reply.split(b"\r\n\r\n", 1)[0].decode("iso-8859-1")
    status, *lines = head.split("\r\n")
    headers = dict(line.split(": ", 1) for line in lines if ": " in line)
    return status, headers


@pytest.mark.parametrize("request_line", [
    b"GARBAGE\r\n\r\n",                     # no version, not a method → 400
    b"GET / HTTP/1.1 x\r\n\r\n",            # four words → 400
    b"GET / HTTP/x.y\r\n\r\n",              # a version that is not a number → 400
    b"POST /\r\n\r\n",                      # HTTP/0.9 has only GET → 400
    b"GET / HTTP/2.0\r\n\r\n",              # 505
    b"GET /\r\n\r\n",                       # HTTP/0.9 GET: the page itself
])
def test_a_request_line_no_browser_sends_still_gets_a_status_line_and_the_lbs_headers(proxy, request_line):
    status, headers = head_of(raw(proxy, request_line[:-2] + host(proxy) + b"\r\n"))
    assert status.startswith("HTTP/1."), f"{request_line!r} → no status line: {status[:60]!r}"
    assert {name: headers.get(name) for name in OWN_HEADERS} == OWN_HEADERS, f"{request_line!r} → {status}"


#: Every request-line version a client can send, and every path class the LB writes itself for. The third
#: audit found `HTTP/0.9` given EXPLICITLY — below 2.0, so http.server stores it as given, and then writes no
#: status line and no header for any reply. The matrix makes the next version sub-case a failing row here,
#: not a fourth audit.
VERSIONS = (b" HTTP/0.9", b" HTTP/1.0", b" HTTP/1.1", b" HTTP/1.9", b"")
LB_WRITTEN = (b"GET /", b"HEAD /", b"GET /healthz", b"GET /lb/status", b"GET /no-such-file", b"BREW /",
              b"GET /" + b"a" * 300)


@pytest.mark.parametrize("version", VERSIONS, ids=lambda v: v.decode().strip() or "none")
@pytest.mark.parametrize("line", LB_WRITTEN, ids=lambda l: l.decode()[:20])
def test_every_version_and_every_reply_the_lb_writes_carries_a_status_line_and_its_headers(proxy, line, version):
    if version == b"" and not line.startswith(b"GET "):
        pytest.skip("a version-less line is HTTP/0.9's own form, which only has GET")
    request = line + version + b"\r\n" + host(proxy) + b"\r\n"
    status, headers = head_of(raw(proxy, request))
    assert status.startswith("HTTP/1."), f"{request!r} → no status line: {status[:60]!r}"
    assert {name: headers.get(name) for name in OWN_HEADERS} == OWN_HEADERS, f"{request!r} → {status}"


@pytest.mark.parametrize("version", VERSIONS, ids=lambda v: v.decode().strip() or "none")
def test_a_flood_of_header_lines_gets_its_431_with_our_headers_whatever_the_version(proxy, version):
    """Over 100 header lines are refused by `handle_one_request` itself — a writer with no handler of ours
    in front of it, and the third audit's other named case."""
    request = b"GET /" + version + b"\r\n" + b"".join(b"X-Pad-%d: 1\r\n" % i for i in range(101)) + b"\r\n"
    status, headers = head_of(raw(proxy, request))
    assert status.startswith("HTTP/1.1 431"), f"101 headers over {version!r} → {status[:60]!r}"
    assert {name: headers.get(name) for name in OWN_HEADERS} == OWN_HEADERS, f"{version!r} → {status}"


#: The LB's OWN surface: every path that is not the inference API. The gateway page lives at `/`, and a
#: browser's CORS preflight is an `OPTIONS` to that very path — so it is a request the LB must answer ITSELF.
#: Until the fourth audit, `handle` answered only GET and HEAD there and every other method fell through to
#: `forward()`: proxied to the operator's cluster and written back under `relaying`, carrying the backend's
#: `access-control-allow-origin: *` and none of OWN_HEADERS, on a port published through a tunnel.
#:
#: The fixture must HOLD A LEASE. With nothing held the router refuses and every method gets the LB's own 503
#: with all four headers — which is exactly why three earlier rounds, and their version × path matrix, went
#: green while the deployment leaked.
OWN_SURFACE = ("/", "/healthz", "/lb/status", "/gateway.js", "/no-such-file")
NOT_READ_METHODS = ("OPTIONS", "DELETE", "PUT", "PATCH", "POST")


@pytest.fixture
def leased(proxy):
    """`proxy`, with one running vLLM lease so the router HAS somewhere to send a forwarded request."""
    from vllm_lb.state import LEASE_RUNNING, Lease

    box = proxy.cfg.vllm_boxes[0]
    proxy.hub.backend_for(box.url).live = True
    proxy.hub.set_leases([Lease(lease_id="own", job_id="j", created_at=0, status=LEASE_RUNNING,
                                host_label=box.host_label, box=box.box)])
    return proxy


@pytest.mark.parametrize("path", OWN_SURFACE)
@pytest.mark.parametrize("method", NOT_READ_METHODS)
#: With a cross-site Origin a WRITE is now refused 403 before any route (host_guard.py): without one, the same request
#: still reaches the route this test is about.
@pytest.mark.parametrize("origin", ["Origin: https://evil.example\r\n", ""], ids=["cross-site", "no-origin"])
def test_no_method_on_the_lbs_own_surface_is_proxied_to_a_backend(leased, method, path, origin):
    request = (f"{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{leased.port}\r\n{origin}"
               "Access-Control-Request-Method: GET\r\n"
               "Content-Length: 0\r\nConnection: close\r\n\r\n").encode()
    status, headers = head_of(raw(leased, request))
    lower = {name.lower() for name in headers}
    assert status.startswith("HTTP/1."), f"{method} {path} → no status line: {status[:60]!r}"
    assert "x-lb-backend" not in lower, f"{method} {path} was PROXIED to a backend: {status}"
    assert "access-control-allow-origin" not in lower, \
        f"{method} {path} handed a cross-origin caller a backend's CORS grant: {status}"
    assert {name: headers.get(name) for name in OWN_HEADERS} == OWN_HEADERS, f"{method} {path} → {status}"


@pytest.mark.parametrize("blank", [b"\r\n", b"\n", b"\r\n\r\n"], ids=["crlf", "lf", "two"])
def test_a_blank_line_before_the_request_line_is_ignored_rather_than_answered_with_silence(proxy, blank):
    """RFC 7230 §3.5: a server SHOULD ignore at least one empty line received before the request line —
    they are the remains of a previous message's terminator. `http.server` reads the blank AS the request
    line, `parse_request` returns False on a zero-word line, and NOTHING is written back: 0 bytes, no
    status line at all. Found live and hermetically by the fourth independent audit."""
    reply = raw(proxy, blank + b"GET / HTTP/1.1\r\n" + host(proxy) + b"Connection: close\r\n\r\n")
    assert reply, f"{blank!r} before the request line got NO reply at all"
    status, headers = head_of(reply)
    assert status.startswith("HTTP/1.1 200"), f"{blank!r} → {status[:60]!r}"
    assert {name: headers.get(name) for name in OWN_HEADERS} == OWN_HEADERS, f"{blank!r} → {status}"


def test_a_100_continue_is_never_written_to_a_client_that_spoke_http_0_9(proxy):
    """Round 3 normalised the STORED request version, which incidentally made `request_version >=
    "HTTP/1.1"` true for a request line that said `HTTP/0.9` — so `parse_request` began answering
    `Expect: 100-continue` with a `100 Continue` a 0.9 client cannot parse. The WIRE version decides this,
    not the stored one. The fourth audit's second unrecorded side effect."""
    body = b"{}"
    reply = raw(proxy, b"POST /no-such-file HTTP/0.9\r\n" + host(proxy) + b"Expect: 100-continue\r\n"
                       b"Content-Type: application/json\r\nContent-Length: " + str(len(body)).encode()
                       + b"\r\nConnection: close\r\n\r\n" + body)
    assert b"100 Continue" not in reply, "wrote a 100 Continue to a client that spoke HTTP/0.9"
    status, headers = head_of(reply)
    assert status.startswith("HTTP/1.1 405"), f"the request itself must still be answered: {status[:60]!r}"
    assert {name: headers.get(name) for name in OWN_HEADERS} == OWN_HEADERS, status


#: A browser never preflights `/` — the page is same-origin. What a CROSS-ORIGIN page preflights is the
#: API itself, and `OPTIONS /v1/messages` fell on the far side of round 4's `not path.startswith("/v1/")`:
#: proxied to the operator's cluster and relayed with the backend's `access-control-allow-origin: *`.
#: `forward()` calls it not-inference itself (`inference = h.command == "POST"`), so it is neither half of
#: the frozen sentence — the fourth audit's own reasoning, one path prefix over (fifth audit).
API_PATHS = ("/v1/messages", "/v1/models", "/v1/chat/completions", "/v1/embeddings")


#: INFERENCE is the only thing relayed. `forward()` says so itself — `inference = h.command == "POST"` —
#: so every other method under `/v1/` is a reply the LB WRITES, and the frozen sentence covers it. Round 5
#: closed `OPTIONS` alone; GET and HEAD are *simple* cross-origin requests that a browser sends with NO
#: preflight, so a 405 on OPTIONS is never consulted for them (sixth audit).
API_METHODS = ("OPTIONS", "GET", "HEAD", "DELETE", "PUT", "PATCH")


@pytest.mark.parametrize("path", API_PATHS)
@pytest.mark.parametrize("method", API_METHODS)
#: With a cross-site Origin a WRITE is now refused 403 before any route (host_guard.py): without one, the same request
#: still reaches the route this test is about.
@pytest.mark.parametrize("origin", ["Origin: https://evil.example\r\n", ""], ids=["cross-site", "no-origin"])
def test_no_method_but_inference_is_proxied_from_the_api_paths(leased, method, path, origin):
    request = (f"{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{leased.port}\r\n{origin}"
               "Access-Control-Request-Method: POST\r\n"
               "Access-Control-Request-Headers: content-type\r\n"
               "Content-Length: 0\r\nConnection: close\r\n\r\n").encode()
    status, headers = head_of(raw(leased, request))
    lower = {name.lower() for name in headers}
    assert status.startswith("HTTP/1."), f"{method} {path} → no status line: {status[:60]!r}"
    assert "x-lb-backend" not in lower, f"{method} {path} was PROXIED to a backend: {status}"
    assert not [h for h in lower if h.startswith("access-control-")], \
        f"{method} {path} handed a cross-origin caller the backend's CORS grant: {sorted(lower)}"
    assert {name: headers.get(name) for name in OWN_HEADERS} == OWN_HEADERS, f"{method} {path} → {status}"


def test_a_100_continue_is_not_written_to_a_version_less_request_line_either(proxy):
    """The OTHER spelling of HTTP/0.9. `proxy.py` says in its own comment that this condition has two
    shapes, and that fixing one of them is what cost three audit rounds — and then the Expect gate was
    written against only the explicit one (fifth audit).

    It must be a GET. A version-less request line is HTTP/0.9's own form, and the stdlib refuses
    `POST /x` as a "Bad HTTP/0.9 request type" with a 400 BEFORE it reads a single header — which is what
    made the first version of this test vacuous: it never reached the Expect path, passed against the
    unfixed tree, and would have certified a fix that does not exist."""
    reply = raw(proxy, b"GET /\r\n" + host(proxy) + b"Expect: 100-continue\r\n\r\n")
    assert b"100 Continue" not in reply, "wrote a 100 Continue to a version-less (HTTP/0.9) request line"
    status, headers = head_of(reply)
    assert status.startswith("HTTP/1.1 200"), f"the request itself must still be answered: {status[:60]!r}"
    assert {name: headers.get(name) for name in OWN_HEADERS} == OWN_HEADERS, status


def test_a_leading_crlf_split_across_tcp_segments_still_gets_a_reply(proxy):
    """`peek(2)` returns whatever is buffered — one byte when the segment boundary falls between the CR
    and the LF — so the blank line was not recognised and the reply was 0 bytes again (fifth audit)."""
    import time

    with socket.create_connection(("127.0.0.1", proxy.port), timeout=10) as s:
        # TCP_NODELAY and a real pause, or the two writes coalesce into one segment and the split never
        # happens — which is precisely why the first version of this test passed against a tree that has
        # no lone-CR branch at all, and certified a fix that was never written (sixth audit).
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        s.sendall(b"\r")                       # the CR alone, in its own segment
        time.sleep(0.05)
        s.sendall(b"\nGET / HTTP/1.1\r\n" + host(proxy) + b"Connection: close\r\n\r\n")
        s.shutdown(socket.SHUT_WR)
        chunks = []
        while True:
            chunk = s.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
    reply = b"".join(chunks)
    assert reply, "a CRLF split across segments got NO reply at all"
    status, headers = head_of(reply)
    assert status.startswith("HTTP/1.1 200"), f"split CRLF → {status[:60]!r}"
    assert {name: headers.get(name) for name in OWN_HEADERS} == OWN_HEADERS, status


def test_an_explicit_http_0_9_relay_gets_its_status_line_and_the_backends_headers_not_ours(proxy):
    """The relay keeps the backend's own headers — but a reply with no status line is no reply at all."""
    from vllm_lb.state import LEASE_RUNNING, Lease

    box = proxy.cfg.vllm_boxes[0]
    proxy.hub.backend_for(box.url).live = True
    proxy.hub.set_leases([Lease(lease_id="r", job_id="j", created_at=0, status=LEASE_RUNNING,
                                host_label=box.host_label, box=box.box)])
    body = b'{"model": "qwen3.8:latest", "max_tokens": 1}'
    request = (b"POST /v1/messages HTTP/0.9\r\n" + host(proxy) + b"Content-Type: application/json\r\nContent-Length: "
               + str(len(body)).encode() + b"\r\n\r\n" + body)
    status, headers = head_of(raw(proxy, request))
    assert status.startswith("HTTP/1.1 200"), f"relay over HTTP/0.9 → {status[:60]!r}"
    assert headers.get("X-LB-Backend") == box.url, "this did not go through the relay"
    assert not {name for name in OWN_HEADERS if name in headers}, "a relayed answer keeps the backend's headers"
