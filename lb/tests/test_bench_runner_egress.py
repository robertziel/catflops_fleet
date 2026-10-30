"""The sealed runner's egress server (lb/bench_runner/egress.py): the proxy and the two fixed relays.

The policy is tested on its own (test_bench_runner_policy.py). Here it is the SERVER: that a refused
CONNECT is answered 403 and never dialled, that only CONNECT is spoken, that an allowed tunnel carries
bytes both ways to exactly the address the decision checked, and that a relay reaches its one fixed
target and nothing else. Everything runs on loopback against a local echo server, with a stub decision,
so nothing here leaves the machine.
"""
from __future__ import annotations

import asyncio
import socket
import threading
import time

import pytest

from bench_runner import egress, policy


class Echo:
    """A TCP server that echoes what it receives and counts who connected."""

    def __init__(self) -> None:
        self.connections = 0
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(16)
        self.port = self.sock.getsockname()[1]
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self) -> None:
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            self.connections += 1
            threading.Thread(target=self._echo, args=(conn,), daemon=True).start()

    @staticmethod
    def _echo(conn) -> None:
        with conn:
            while True:
                data = conn.recv(65536)
                if not data:
                    return
                conn.sendall(data)

    def close(self) -> None:
        self.sock.close()


class Running:
    """An Egress on its own loop in a thread, bound to loopback on ephemeral ports."""

    def __init__(self, decide, relays=(), **kw) -> None:
        self.loop = asyncio.new_event_loop()
        self.lines: list[str] = []
        self.egress = egress.Egress(decide=decide, log=self.lines.append, **kw)
        ready = threading.Event()

        def run():
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self.egress.start("127.0.0.1", 0, list(relays)))
            ready.set()
            self.loop.run_forever()

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()
        assert ready.wait(10)

    @property
    def proxy_port(self) -> int:
        return self.egress.ports["proxy"]

    def relay_port(self, listen: int) -> int:
        return self.egress.ports[listen]

    def close(self) -> None:
        """Stop listening, end every connection the egress still holds, and close its loop."""
        async def shutdown():
            for server in self.egress.servers:
                server.close()
            current = asyncio.current_task()
            tasks = [t for t in asyncio.all_tasks() if t is not current]
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

        asyncio.run_coroutine_threadsafe(shutdown(), self.loop).result(10)
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(10)
        self.loop.close()


@pytest.fixture
def echo():
    e = Echo()
    yield e
    e.close()


def allow_to_loopback(host, port):
    """A stub decision: `public.test` is allowed and pinned to 127.0.0.1; everything else is refused."""
    if host == "public.test":
        return policy.Decision(True, "public.test is a test host", ("127.0.0.1",))
    return policy.Decision(False, f"{host} is refused by the test policy")


def talk(port: int, payload: bytes, *, read_until_close=False, timeout=5.0) -> bytes:
    with socket.create_connection(("127.0.0.1", port), timeout=timeout) as s:
        s.sendall(payload)
        if read_until_close:
            chunks = []
            while True:
                data = s.recv(65536)
                if not data:
                    break
                chunks.append(data)
            return b"".join(chunks)
        return s.recv(65536)


def test_a_refused_connect_is_a_403_with_the_reason_and_nothing_is_dialled(echo):
    e = Running(allow_to_loopback)
    try:
        answer = talk(e.proxy_port, f"CONNECT api.anthropic.com:{echo.port} HTTP/1.1\r\n"
                                    f"Host: api.anthropic.com\r\n\r\n".encode(), read_until_close=True)
        assert answer.startswith(b"HTTP/1.1 403 "), answer
        assert b"refused by the test policy" in answer
        assert echo.connections == 0, "a refused destination must never be dialled"
        assert any(line.startswith("DENY") and "api.anthropic.com" in line for line in e.lines), e.lines
    finally:
        e.close()


def test_an_allowed_tunnel_carries_bytes_both_ways_to_the_checked_address(echo):
    e = Running(allow_to_loopback)
    try:
        with socket.create_connection(("127.0.0.1", e.proxy_port), timeout=5) as s:
            s.sendall(f"CONNECT public.test:{echo.port} HTTP/1.1\r\nHost: public.test\r\n\r\n".encode())
            head = b""
            while b"\r\n\r\n" not in head:
                head += s.recv(1)
            assert head.startswith(b"HTTP/1.1 200 "), head
            s.sendall(b"hello through the tunnel")
            got = b""
            while len(got) < len(b"hello through the tunnel"):
                got += s.recv(65536)
            assert got == b"hello through the tunnel"
        assert echo.connections == 1
        assert any(line.startswith("ALLOW") and "127.0.0.1" in line for line in e.lines), e.lines
    finally:
        e.close()


def test_bytes_sent_with_the_connect_reach_the_upstream(echo):
    """A client may send its first bytes (a TLS ClientHello) right behind the CONNECT head."""
    e = Running(allow_to_loopback)
    try:
        with socket.create_connection(("127.0.0.1", e.proxy_port), timeout=5) as s:
            s.sendall(f"CONNECT public.test:{echo.port} HTTP/1.1\r\n\r\nEARLY".encode())
            got = b""
            deadline = time.time() + 5
            while b"EARLY" not in got and time.time() < deadline:
                got += s.recv(65536)
            assert got.startswith(b"HTTP/1.1 200 ") and got.endswith(b"EARLY"), got
    finally:
        e.close()


@pytest.mark.parametrize("request_head, status", [
    (b"GET http://example.com/ HTTP/1.1\r\nHost: example.com\r\n\r\n", b"405"),
    (b"POST http://lb:8010/v1/messages HTTP/1.1\r\n\r\n", b"405"),
    (b"CONNECT example.com HTTP/1.1\r\n\r\n", b"400"),
    (b"CONNECT example.com:https HTTP/1.1\r\n\r\n", b"400"),
    (b"not a request at all\r\n\r\n", b"400"),
])
def test_only_a_well_formed_connect_is_spoken(echo, request_head, status):
    e = Running(allow_to_loopback)
    try:
        answer = talk(e.proxy_port, request_head, read_until_close=True)
        assert answer.startswith(b"HTTP/1.1 " + status), answer
        assert echo.connections == 0
    finally:
        e.close()


def test_an_allowed_destination_that_does_not_answer_is_a_502(echo):
    spare = socket.socket()
    spare.bind(("127.0.0.1", 0))
    port = spare.getsockname()[1]
    spare.close()                     # a port nothing listens on
    e = Running(allow_to_loopback)
    try:
        answer = talk(e.proxy_port, f"CONNECT public.test:{port} HTTP/1.1\r\n\r\n".encode(),
                      read_until_close=True)
        assert answer.startswith(b"HTTP/1.1 502 "), answer
    finally:
        e.close()


def test_the_tunnel_cap_answers_503_rather_than_queueing(echo):
    e = Running(allow_to_loopback, max_tunnels=1)
    try:
        with socket.create_connection(("127.0.0.1", e.proxy_port), timeout=5) as first:
            first.sendall(f"CONNECT public.test:{echo.port} HTTP/1.1\r\n\r\n".encode())
            assert first.recv(64).startswith(b"HTTP/1.1 200 ")
            answer = talk(e.proxy_port, f"CONNECT public.test:{echo.port} HTTP/1.1\r\n\r\n".encode(),
                          read_until_close=True)
            assert answer.startswith(b"HTTP/1.1 503 "), answer
    finally:
        e.close()


def test_a_relay_reaches_its_one_fixed_private_target(echo):
    e = Running(allow_to_loopback, relays=[(0, "127.0.0.1", echo.port)])
    try:
        port = e.relay_port(0)
        with socket.create_connection(("127.0.0.1", port), timeout=5) as s:
            s.sendall(b"GET /healthz HTTP/1.1\r\n\r\n")
            got = b""
            while len(got) < 25:
                got += s.recv(65536)
            assert got == b"GET /healthz HTTP/1.1\r\n\r\n"
        assert echo.connections == 1
    finally:
        e.close()


def test_a_relay_whose_target_resolves_public_dials_nothing(echo):
    """A relay is for the LB and the engine on private networks; one pointed at a public address — at
    Anthropic's above all — is refused per connection, before a byte is sent."""
    e = Running(allow_to_loopback, relays=[(0, "160.79.104.10", 443)])
    try:
        answer = talk(e.relay_port(0), b"anything", read_until_close=True)
        assert answer == b""
        assert any("REFUSE relay" in line and "Anthropic" in line for line in e.lines), e.lines
    finally:
        e.close()


@pytest.mark.parametrize("spec, expected", [("8010=vllm-lb:8010", (8010, "vllm-lb", 8010)),
                                            ("8000=node-test:8000", (8000, "node-test", 8000)),
                                            ("8000=192.0.2.2:8100", (8000, "192.0.2.2", 8100))])
def test_a_relay_spec_is_listen_equals_host_colon_port(spec, expected):
    assert egress.parse_relay(spec) == expected


@pytest.mark.parametrize("spec", ["8010", "8010=", "=vllm-lb:8010", "x=vllm-lb:8010", "8010=vllm-lb",
                                  "8010=vllm-lb:port", "70000=vllm-lb:8010"])
def test_a_bad_relay_spec_is_refused_at_start(spec):
    with pytest.raises(ValueError):
        egress.parse_relay(spec)