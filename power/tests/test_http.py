"""The HTTP wrapper around PowerApp: a real server on an ephemeral port inside the test container."""
from __future__ import annotations

import http.client
import threading
from http.server import ThreadingHTTPServer

import pytest

from fleet_power.__main__ import make_handler
from tests.fakes import GATE, ORIGIN, make_app


@pytest.fixture
def server(tmp_path):
    app = make_app(tmp_path)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    app.listening_on("127.0.0.1", srv.server_address[1])
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_address[1]
    srv.shutdown()


def raw(port, request: bytes) -> bytes:
    import socket
    with socket.create_connection(("127.0.0.1", port), timeout=5) as s:
        s.sendall(request)
        s.shutdown(socket.SHUT_WR)
        out = b""
        while chunk := s.recv(65536):
            out += chunk
    return out


HDRS = (f"X-Power-Gate: {GATE}\r\nX-Power-User: operator\r\nOrigin: {ORIGIN}\r\n"
        "Content-Type: application/json\r\n").encode()


def test_state_over_http(server):
    c = http.client.HTTPConnection("127.0.0.1", server, timeout=5)
    c.request("GET", "/power/api/state", headers={"X-Power-Gate": GATE, "X-Power-User": "operator"})
    r = c.getresponse()
    assert r.status == 200 and r.getheader("X-Content-Type-Options") == "nosniff"


@pytest.mark.parametrize("cl", [b"abc", b"-5", b"99999999"])
def test_a_bad_content_length_is_a_clean_4xx(server, cl):
    out = raw(server, f"POST /power/api/confirm HTTP/1.1\r\nHost: 127.0.0.1:{server}\r\n".encode() + HDRS
              + b"Content-Length: " + cl + b"\r\n\r\n{}")
    status = int(out.split(b" ", 2)[1])
    assert status in (400, 413), out[:200]
