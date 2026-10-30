"""THE HOST ALLOWLIST (vllm_lb/host_guard.py) on the gateway — the DNS-rebinding hole of the 1.0.0-beta.2 audit.

A page the operator opens can rebind its own hostname to 127.0.0.1 and then, SAME-ORIGIN, read /lb/status and POST JSON
to every control route: the JSON-only content-type rule stops a cross-origin form, not that. What such a page cannot
forge is its own name, which the browser sends as `Host` (and `Origin` on a write). So the gateway answers only to its
listener's loopback names and the names the deployment states in LB_ALLOWED_HOSTS — REQUIRED, no default — and refuses
a write from any other origin.
"""
from __future__ import annotations

import http.client
import json
import logging
import os
import socket
import subprocess
import sys

import pytest

from tests.fakes import FakeBackend, free_port
from vllm_lb import host_guard
from vllm_lb.config import Config, VllmBox
from vllm_lb.host_guard import Authority, HostGuard, parse_allowed, parse_authority
from vllm_lb.hub import Hub
from vllm_lb.proxy import Proxy

# ── the variable: required, one deliberate spelling for "loopback only" ──────────────────────────────────────


def test_the_env_is_required_and_its_absence_refuses_the_start_by_name():
    with pytest.raises(ValueError, match="LB_ALLOWED_HOSTS is required"):
        Config.from_env({})


@pytest.mark.parametrize("raw", ["", "   "])
def test_an_empty_value_is_refused_never_read_as_loopback(raw):
    with pytest.raises(ValueError, match="LB_ALLOWED_HOSTS is empty.*loopback"):
        Config.from_env({"LB_ALLOWED_HOSTS": raw})


@pytest.mark.parametrize("raw", ["loopback", " LOOPBACK "])
def test_loopback_is_the_stated_choice_of_no_other_name(raw):
    assert Config.from_env({"LB_ALLOWED_HOSTS": raw}).allowed_hosts == ()


def test_the_names_are_parsed_lowercased_and_deduplicated():
    cfg = Config.from_env({"LB_ALLOWED_HOSTS": " vllm.example.com , VLLM-LB:8010,lb:8010,[::1]:9000,vllm-lb:8010"})
    assert cfg.allowed_hosts == (Authority("vllm.example.com", None), Authority("vllm-lb", 8010),
                                 Authority("lb", 8010), Authority("[::1]", 9000))


@pytest.mark.parametrize("raw", ["https://vllm.example.com", "vllm.example.com/x", "*.example.com", "a,,b",
                                 "loopback,vllm-lb:8010", "vllm-lb:0", "vllm-lb:70000", "vllm-lb:abc", "user@host",
                                 "[::1", "[nope]:1", "a b"])
def test_anything_but_host_and_port_refuses_the_start_by_name(raw):
    with pytest.raises(ValueError, match="LB_ALLOWED_HOSTS"):
        Config.from_env({"LB_ALLOWED_HOSTS": raw})


def test_the_class_default_is_the_strictest_set_loopback_only():
    assert Config().allowed_hosts == ()


# ── the rules, without a socket ──────────────────────────────────────────────────────────────────────────────


def _guard(allowed: str = "loopback", listen: str = "0.0.0.0", port: int = 8010, said=None) -> HostGuard:
    return HostGuard("LB_ALLOWED_HOSTS", parse_allowed("LB_ALLOWED_HOSTS", allowed), listen, port,
                     (said if said is not None else []).append)


@pytest.mark.parametrize("host", ["127.0.0.1:8010", "localhost:8010", "LOCALHOST:8010", "[::1]:8010"])
def test_the_listeners_loopback_names_at_its_port_are_allowed(host):
    assert _guard().refusal("GET", host, None) is None


@pytest.mark.parametrize("host", ["127.0.0.1:8011", "localhost:80", "[::1]:18010", "127.0.0.1", "localhost",
                                  "evil.example:8010", "evil.example", "127.0.0.2:8010", "", None, "bad host:1"])
def test_anything_else_is_misdirected(host):
    status, body = _guard().refusal("GET", host, None)
    assert status == 421 and body["error"]["type"] == "misdirected_host"
    assert "LB_ALLOWED_HOSTS" in body["error"]["message"]


def test_a_specific_bind_address_is_allowed_too_a_wildcard_names_nothing():
    assert _guard(listen="192.0.2.5").refusal("GET", "192.0.2.5:8010", None) is None
    assert _guard(listen="0.0.0.0").refusal("GET", "0.0.0.0:8010", None)[0] == 421


@pytest.mark.parametrize("host, ok", [("vllm.example.com", True), ("vllm.example.com:443", True),
                                      ("vllm.example.com:80", True), ("VLLM.example.com", True),
                                      ("vllm.example.com:8443", False), ("vllm.example.com.evil", False)])
def test_a_portless_name_is_a_proxy_on_its_default_port(host, ok):
    assert (_guard("vllm.example.com").refusal("GET", host, None) is None) is ok


@pytest.mark.parametrize("host, ok", [("vllm-lb:8010", True), ("vllm-lb", False), ("vllm-lb:8011", False)])
def test_a_name_with_a_port_matches_that_port_only(host, ok):
    assert (_guard("vllm-lb:8010").refusal("GET", host, None) is None) is ok


@pytest.mark.parametrize("origin", ["https://evil.example", "http://evil.example:8010", "null", "file://",
                                    "https://vllm.example.com:8443", "http://127.0.0.1:9999"])
@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE"])
def test_a_write_from_a_foreign_origin_is_refused(method, origin):
    status, body = _guard("vllm.example.com").refusal(method, "127.0.0.1:8010", origin)
    assert status == 403 and body["error"]["type"] == "cross_origin" and body["error"]["origin"] == origin


@pytest.mark.parametrize("origin", ["https://vllm.example.com", "http://vllm.example.com", "http://127.0.0.1:8010",
                                    "http://localhost:8010", "http://[::1]:8010", None])
def test_a_write_from_an_allowed_origin_or_none_passes(origin):
    assert _guard("vllm.example.com").refusal("POST", "vllm.example.com", origin) is None


def test_a_read_is_judged_by_its_host_alone():
    assert _guard().refusal("GET", "127.0.0.1:8010", "https://evil.example") is None


def test_each_host_is_logged_once_and_the_log_is_bounded():
    said: list[str] = []
    g = _guard("vllm-lb:8010", said=said)
    for _ in range(3):
        g.refusal("GET", "evil.example", None)
        g.refusal("GET", "vllm-lb:8010", None)
    assert len(said) == 2
    assert "'evil.example'" in said[0] and "REFUSED" in said[0] and "LB_ALLOWED_HOSTS" in said[0]
    assert "'vllm-lb:8010'" in said[1] and "allowed" in said[1]
    for i in range(host_guard.SEEN_CAP + 10):
        g.refusal("GET", f"h{i}.example", None)
    assert len(said) == host_guard.SEEN_CAP + 1, "capped, and the cap said once"


def test_the_refusal_never_lists_the_allowed_names():
    status, body = _guard("secret-name.example").refusal("GET", "evil.example", None)
    assert "secret-name" not in json.dumps(body)


@pytest.mark.parametrize("text", ["a:b:c", "[::1]x", "[::1]:", ":8010", "-a", "a-"])
def test_parse_authority_refuses_malformed(text):
    with pytest.raises(ValueError):
        parse_authority(text)


# ── the gateway, on a real socket ────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def gateway():
    b = FakeBackend()
    cfg = Config(vllm_boxes=(VllmBox("box-a-gpu", "box-a", b.url),), listen_host="127.0.0.1", listen_port=0,
                 allowed_hosts=parse_allowed("LB_ALLOWED_HOSTS", "vllm.example.com,vllm-lb:8010,lb:8010"))
    p = Proxy(cfg, Hub(cfg))
    p.start()
    yield p
    p.stop()
    b.close()


def send(proxy, method: str, path: str, host: str | None, headers: dict | None = None, body: bytes = b""):
    """One request with exactly the Host given (None = no Host header at all)."""
    lines = [f"{method} {path} HTTP/1.1"]
    if host is not None:
        lines.append(f"Host: {host}")
    for k, v in (headers or {}).items():
        lines.append(f"{k}: {v}")
    lines += [f"Content-Length: {len(body)}", "Connection: close", "", ""]
    with socket.create_connection(("127.0.0.1", proxy.port), timeout=10) as s:
        s.sendall("\r\n".join(lines).encode() + body)
        s.shutdown(socket.SHUT_WR)
        reply = b""
        while chunk := s.recv(65536):
            reply += chunk
    head, _, payload = reply.partition(b"\r\n\r\n")
    status = int(head.split(b" ", 2)[1])
    return status, payload


@pytest.mark.parametrize("name", ["127.0.0.1", "localhost", "[::1]"])
def test_the_gateway_answers_its_loopback_names_at_its_bound_port(gateway, name):
    status, body = send(gateway, "GET", "/lb/status", f"{name}:{gateway.port}")
    assert status == 200 and "backends" in json.loads(body)


def test_the_gateway_refuses_a_port_mismatch(gateway):
    status, body = send(gateway, "GET", "/healthz", f"127.0.0.1:{gateway.port + 1}")
    assert status == 421 and "LB_ALLOWED_HOSTS" in json.loads(body)["error"]["message"]


def test_a_rebound_name_cannot_read_the_status(gateway):
    status, body = send(gateway, "GET", "/lb/status", f"attacker.example:{gateway.port}")
    assert status == 421, "a DNS-rebound page read /lb/status"
    assert json.loads(body)["error"] == {
        "type": "misdirected_host", "host": f"attacker.example:{gateway.port}",
        "message": json.loads(body)["error"]["message"]}
    assert b"backends" not in body


def test_a_rebound_name_cannot_kill_or_switch(gateway):
    for path in ("/lb/cluster", "/lb/requests/x/kill", "/lb/swarm/start", "/lb/clusters/pair/activate"):
        status, _ = send(gateway, "POST", path, f"attacker.example:{gateway.port}",
                         {"Content-Type": "application/json", "Origin": f"http://attacker.example:{gateway.port}"},
                         b"{}")
        assert status == 421, f"POST {path} from a rebound name → {status}"


def test_a_missing_host_is_refused(gateway):
    assert send(gateway, "GET", "/healthz", None)[0] == 421


@pytest.mark.parametrize("host", ["vllm.example.com", "vllm.example.com:443", "vllm-lb:8010", "lb:8010"])
def test_the_proxy_name_and_the_docker_names_are_served(gateway, host):
    status, body = send(gateway, "GET", "/healthz", host)
    assert (status, json.loads(body)) == (200, {"ok": True})


def test_an_inference_post_through_the_docker_name_is_not_refused_by_the_guard(gateway):
    status, body = send(gateway, "POST", "/v1/messages", "vllm-lb:8010", {"Content-Type": "application/json"},
                        b'{"model": "x", "max_tokens": 1, "messages": []}')
    assert status not in (421, 403), body


def test_a_cross_site_post_with_an_allowed_host_is_refused(gateway):
    status, body = send(gateway, "POST", "/lb/requests/x/kill", f"127.0.0.1:{gateway.port}",
                        {"Content-Type": "application/json", "Origin": "https://evil.example"}, b"{}")
    assert status == 403 and json.loads(body)["error"]["type"] == "cross_origin"


def test_a_same_origin_post_from_the_proxy_name_reaches_its_route(gateway):
    status, body = send(gateway, "POST", "/lb/requests/no-such-turn/kill", "vllm.example.com",
                        {"Content-Type": "application/json", "Origin": "https://vllm.example.com"}, b"{}")
    assert status not in (421, 403), body


def test_the_gateway_logs_each_host_once(gateway, caplog):
    with caplog.at_level(logging.WARNING, logger="vllm_lb.proxy"):
        for _ in range(3):
            send(gateway, "GET", "/healthz", "rebound.example")
    lines = [r.getMessage() for r in caplog.records if "rebound.example" in r.getMessage()]
    assert len(lines) == 1 and "LB_ALLOWED_HOSTS" in lines[0]


def test_the_entrypoint_refuses_to_boot_without_the_variable():
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": "/app", "LB_LISTEN": f"127.0.0.1:{free_port()}"}
    proc = subprocess.run([sys.executable, "-m", "vllm_lb"], env=env, cwd="/app", capture_output=True, text=True,
                          timeout=60)
    assert proc.returncode == 78, proc.stdout + proc.stderr
    assert "LB_ALLOWED_HOSTS is required" in proc.stdout + proc.stderr
