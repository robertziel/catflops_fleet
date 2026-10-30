"""The LB serves the gateway UI itself — one image, one origin, no relay.

The dashboard was a second container and a second runtime whose whole job was forwarding
/api/lb -> /lb/status and /api/chat -> /v1/messages to a process that already serves both.
"""
from __future__ import annotations

import http.client
import json

import pytest

from tests.fakes import FakeBackend
from vllm_lb.config import Config, VllmBox
from vllm_lb.hub import Hub
from vllm_lb.proxy import Proxy
from vllm_lb.state import LEASE_RUNNING, Lease


@pytest.fixture
def proxy():
    b = FakeBackend()
    cfg = Config(vllm_boxes=(VllmBox("box-a-gpu", "box-a", b.url),), 
                 listen_host="127.0.0.1", listen_port=0)
    p = Proxy(cfg, Hub(cfg))
    p.start()
    yield p
    p.stop()
    b.close()


def get(proxy, path):
    c = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=10)
    c.request("GET", path)
    r = c.getresponse()
    return r.status, r.getheader("Content-Type") or "", r.read()


def test_the_root_serves_the_gateway_page(proxy):
    status, ctype, body = get(proxy, "/")
    assert status == 200
    assert "text/html" in ctype
    # The fleet's brand (operator 2026-10-28): catflops' card-flop mark beside the name, in the top bar and the tab.
    assert b"<title>CatFlops Fleet</title>" in body
    header = body.split(b'<header class="top">', 1)[1].split(b"</header>", 1)[0]
    assert b'<svg class="mark"' in header and b'<span class="wordmark"><span class="wm-cat">Cat</span><span class="wm-flops">Flops</span> <span class="wm-fleet">Fleet</span></span>' in header
    assert b'rel="icon" href="data:image/svg+xml' in body
    assert b"vLLM Gateway" not in body


def test_the_mark_is_silver_on_the_dark_panel(proxy):
    # Operator 2026-10-28: "Use silver logo on dark mode", "similar colors to CPU svg icon IDLE with little sparks on
    # some CPUs" -- the panel is dark-only.
    css = get(proxy, "/gateway.css")[2]
    assert b"--mark-cat: #B1B8C0;" in css and b"--mark-idle: #8B949E;" in css
    body = get(proxy, "/")[2]
    assert body.count(b'class="mark-core-idle"') == 3 and b'class="mark-box-a-cyan"' in body
    # "Make Cat Flops Fleet each in different color but keep them equal": one hue each, one OKLCH lightness (0.86).
    for word in (b".wm-cat { color: #CBD2D9; }", b".wm-flops { color: #64EBDB; }", b".wm-fleet { color: #7BEE98; }"):
        assert word in css


def test_assets_are_served_with_their_own_content_type(proxy):
    for path, want in (("/gateway.js", "javascript"), ("/gateway.css", "css")):
        status, ctype, body = get(proxy, path)
        assert status == 200, f"{path} -> {status}"
        assert want in ctype, f"{path} -> {ctype}"
        assert body, f"{path} is empty"


@pytest.mark.parametrize("path", [
    "/../config.py",
    "/../../etc/passwd",
    "/%2e%2e/%2e%2e/etc/passwd",
    "/....//config.py",
    "//etc/passwd",
])
def test_no_request_can_read_outside_the_static_directory(proxy, path):
    """THE KILL LINE. This process holds broker credentials and is reachable from the internet
    through the tunnel, so containment is verified after resolution — never by trusting the
    request string."""
    status, _, body = get(proxy, path)
    assert status in (400, 403, 404), f"{path} -> {status}"
    assert b"root:" not in body and b"LB_DB_URL_ENV" not in body


def test_the_api_surface_is_unchanged(proxy):
    assert get(proxy, "/healthz")[0] == 200
    status, ctype, body = get(proxy, "/lb/status")
    assert status == 200 and "json" in ctype
    assert "backends" in json.loads(body)


def test_an_unknown_path_is_a_404_not_a_proxied_request(proxy):
    status, _, _ = get(proxy, "/definitely-not-a-file")
    assert status == 404


@pytest.mark.parametrize("path", [
    "/" + "a" * 300,                    # one segment past NAME_MAX -> OSError(36)
    "/" + ("b" * 250 + "/") * 20,       # many long segments
    "/index.html\x00.txt",              # a REAL NUL, not the literal four chars
])
def test_a_name_the_filesystem_refuses_is_a_404_not_a_500(proxy, path):
    """An unauthenticated 500 on an internet-facing endpoint, and it wrote the whole
    attacker-controlled path to the log TWICE per request: a 60 KB path grew the log by
    120 KB. `is_file()` sat outside the guard, so OSError(36) 'File name too long' escaped."""
    try:
        status, _, _ = get(proxy, path)
    except http.client.InvalidURL:
        return                           # the client refused to send it; nothing reached us
    assert status == 404, f"{len(path)}-char path -> {status}"


def test_every_asset_the_page_links_actually_resolves(proxy):
    """D1: the migration dropped style.css and linked only the overlay. Nothing failed — a
    missing <link> makes no request and logs no console error — so the suite stayed green and
    the page rendered as unstyled black-on-white with invisible headings. Assert the page's
    OWN references resolve, rather than a hand-written list that drifts from the markup."""
    import re
    _, _, page = get(proxy, "/")
    html = page.decode()
    refs = re.findall(r'<link[^>]+href="(/[^"]+)"', html) + re.findall(r'<script[^>]+src="(/[^"]+)"', html)
    assert refs, "the page links nothing — the test would pass vacuously"
    for ref in refs:
        status, _, body = get(proxy, ref)
        assert status == 200 and body, f"{ref} -> {status} ({len(body)} bytes)"


def test_the_page_has_a_cluster_panel(proxy):
    """The two-box TP=2 cluster is switched from this page. Byte-level only: what it renders
    in each state is checked in a real browser (worklog/gateway_ui_states.py)."""
    _, _, page = get(proxy, "/")
    assert b'id="cluster-panel"' in page


def test_the_script_drives_the_cluster_switch(proxy):
    _, _, js = get(proxy, "/gateway.js")
    assert b"/lb/cluster" in js, "the page never posts the operator's switch"
    assert b"cluster-toggle" in js
    assert b"load-pass" in js and b"startup-phase" in js, "a cold start shows no bar per LoadPass, no StartupPhase"


def test_keep_loaded_and_the_legacy_sections_are_gone_from_the_page(proxy):
    """Keep loaded and the panel's Other servers / Fleet / Chat sections are retired (operator, 2026-10-17)."""
    _, _, js = get(proxy, "/gateway.js")
    _, _, css = get(proxy, "/gateway.css")
    for gone in (b"/keep", b"keep-toggle", b"chat-form", b"renderKpis"):
        assert gone not in js and gone not in css, gone


def test_the_keep_route_is_gone_and_status_names_no_keep(proxy):
    c = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=10)
    c.request("POST", "/lb/boxes/box-a/keep", body=b'{"enabled": true}', headers={"Content-Type": "application/json"})
    r = c.getresponse()
    assert r.status == 405 and json.loads(r.read())["error"]["type"] == "method_not_allowed"
    _, _, body = get(proxy, "/lb/status")
    status = json.loads(body)
    assert "keep" not in status and "keep_error" not in status


# ── what the LB writes itself: never framed, never sniffed ─────────────────────

#: The page carries a Start/Stop switch for a cluster that takes most of an hour to start, so no
#: other origin may frame it and steer a click; and nothing the LB serves may be sniffed into a
#: type it was not sent as.
OWN_HEADERS = {"X-Frame-Options": "DENY", "Content-Security-Policy": "frame-ancestors 'none'",
               "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer"}


def answer(proxy, method, path, body=None):
    c = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=10)
    c.request(method, path, body=body,
              headers={"Content-Type": "application/json"} if body is not None else {})
    r = c.getresponse()
    r.read()
    return r


@pytest.mark.parametrize("method, path, body", [
    ("GET", "/", None),
    ("HEAD", "/", None),
    ("GET", "/gateway.js", None),
    ("GET", "/gateway.css", None),
    ("GET", "/lb/status", None),
    ("POST", "/lb/cluster", b'{"enabled": true}'),     # refused here (no cluster): still the LB's JSON
    ("GET", "/no-such-file", None),
    ("BREW", "/", None),                               # no do_BREW: Python's own error page — still ours
])
def test_what_the_lb_writes_itself_cannot_be_framed_or_sniffed(proxy, method, path, body):
    """Exactly `frame-ancestors 'none'` for CSP: a script-src would be a second policy this page never
    agreed to keep."""
    r = answer(proxy, method, path, body)
    assert {name: r.getheader(name) for name in OWN_HEADERS} == OWN_HEADERS, f"{method} {path} -> {r.status}"


def test_a_relayed_inference_answer_keeps_the_backends_headers(proxy):
    """The policy is for what the LB writes ITSELF. A relayed answer is passed through as the backend
    sent it — the streaming relay is not where a page's framing rules belong."""
    box = proxy.cfg.vllm_boxes[0]
    proxy.hub.backend_for(box.url).live = True
    proxy.hub.set_leases([Lease(lease_id="r", job_id="j", created_at=0, status=LEASE_RUNNING,
                                host_label=box.host_label, box=box.box)])
    r = answer(proxy, "POST", "/v1/messages", b'{"model": "qwen3.8:latest", "max_tokens": 1}')
    assert r.status == 200 and r.getheader("X-LB-Backend") == box.url, "this did not go through the relay"
    assert {name: r.getheader(name) for name in OWN_HEADERS} == dict.fromkeys(OWN_HEADERS)
