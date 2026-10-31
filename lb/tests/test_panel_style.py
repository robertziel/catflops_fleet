"""THE PANEL'S DEFAULT STYLE IS THE DEPLOYMENT'S (operator, 2026-10-31: "allow setup default in yml config").

  THE FLEET KEY  `panel: {style: <key>}` — REQUIRED, like every value of the fleet file (no defaults): a fleet that
                 names none, or names a key styles/index.json does not list, refuses the boot naming the known keys.
  SERVED         the LB stamps the key (and its sheet's href) onto <html> of every page it serves —
                 `data-style-default` / `data-style-default-href` — so styles.js applies it BEFORE FIRST PAINT, with
                 no request of its own. The source pages carry both attributes empty: an LB with no fleet stamps
                 nothing and the page says so (tests/styles.node.js).
  STATELESS      the server keeps only the deployment's default; a browser's own choice is still its localStorage.
"""
from __future__ import annotations

import http.client
import json
import pathlib
import re

import pytest

from tests.fakes import FakeBackend
from tests.fleet_files import ONE_HOST, PAIR
from vllm_lb import panel_styles
from vllm_lb.config import Config, VllmBox
from vllm_lb.fleet import parse_fleet
from vllm_lb.hub import Hub
from vllm_lb.proxy import Proxy

LB = pathlib.Path(__file__).resolve().parent.parent
STATIC = LB / "vllm_lb" / "static"
PAGES = ("index.html", "benchmarks.html", "sessions.html")
KEYS = [s["key"] for s in json.loads((STATIC / "styles" / "index.json").read_text())["styles"]]


def with_style(text: str, style: str) -> str:
    return text.replace("panel:\n  style: vibe\n", f"panel:\n  style: {style}\n")


# ── the fleet key ───────────────────────────────────────────────────────────────────────────────────

def test_the_registry_keys_are_the_ones_the_fleet_accepts():
    assert panel_styles.keys() == tuple(KEYS) == ("vibe", "clean", "mono", "glass", "terminal")


@pytest.mark.parametrize("key", KEYS)
def test_a_fleet_names_its_panels_default_style(key):
    assert parse_fleet(with_style(PAIR, key), "fleet.yml").panel_style == key


def test_a_fleet_that_names_no_panel_style_is_refused_naming_the_known_ones():
    bare = PAIR.replace("panel:\n  style: vibe\n", "")
    assert bare != PAIR
    with pytest.raises(ValueError) as e:
        parse_fleet(bare, "fleet.yml")
    msg = str(e.value)
    assert "fleet.yml names no panel.style" in msg and "one of vibe, clean, mono, glass, terminal" in msg, msg
    assert "panel:\n  style: glass" in msg, "the refusal carries the line to add: the default the examples ship"
    assert "vibe keeps the original look" in msg, msg


@pytest.mark.parametrize("bad, says", [
    ("panel:\n  style: neon\n", "panel.style 'neon' is no style of the panel — one of vibe, clean, mono, glass, terminal"),
    ("panel:\n  style: Clean\n", "panel.style 'Clean' is no style"),
    ("panel:\n  style: 3\n", "panel.style 3 is no style"),
    ("panel:\n  style: vibe\n  font: x\n", "unknown font"),
    ("panel: {}\n", "missing style"),
    ("panel: clean\n", "panel must be a mapping"),
])
def test_an_unknown_or_malformed_panel_block_is_refused(bad, says):
    with pytest.raises(ValueError) as e:
        parse_fleet(PAIR.replace("panel:\n  style: vibe\n", bad), "fleet.yml")
    assert says in str(e.value) and "fleet.yml" in str(e.value), str(e.value)


def test_the_one_host_shape_names_it_too():
    assert parse_fleet(ONE_HOST, "fleet.yml").panel_style == "vibe"


# ── served: stamped onto every page, before first paint ─────────────────────────────────────────────

@pytest.mark.parametrize("page", PAGES)
def test_every_page_carries_the_empty_default_the_lb_fills(page):
    html = (STATIC / page).read_text()
    tag = html[html.index("<html"):html.index(">", html.index("<html")) + 1]
    assert tag == '<html lang="en" data-style-default="" data-style-default-href="">', tag
    assert html.count('data-style-default=""') == 1 and html.count('data-style-default-href=""') == 1


@pytest.mark.parametrize("key", KEYS)
def test_stamp_fills_the_key_and_its_sheet(key):
    html = (STATIC / "index.html").read_bytes()
    out = panel_styles.stamp(html, key).decode()
    entry = next(s for s in json.loads((STATIC / "styles" / "index.json").read_text())["styles"] if s["key"] == key)
    assert f'<html lang="en" data-style-default="{key}" data-style-default-href="{entry["file"] or ""}">' in out
    assert out.replace(f'data-style-default="{key}"', 'data-style-default=""').replace(
        f'data-style-default-href="{entry["file"] or ""}"', 'data-style-default-href=""') == html.decode()


def test_stamp_refuses_a_key_the_registry_does_not_list():
    with pytest.raises(ValueError, match="neon"):
        panel_styles.stamp((STATIC / "index.html").read_bytes(), "neon")


def test_stamp_reads_the_built_registry_so_the_href_carries_the_builds_hash(tmp_path, monkeypatch):
    built = tmp_path / "index.json"
    built.write_text(json.dumps({"styles": [{"key": "vibe", "label": "Vibe coded", "file": None},
                                            {"key": "glass", "label": "Glass", "file": "/styles/glass.css?v=0123456789ab"}]}))
    monkeypatch.setattr(panel_styles, "INDEX", built)
    panel_styles.registry.cache_clear()
    try:
        out = panel_styles.stamp(b'<html lang="en" data-style-default="" data-style-default-href="">', "glass")
        assert out == b'<html lang="en" data-style-default="glass" data-style-default-href="/styles/glass.css?v=0123456789ab">'
    finally:
        panel_styles.registry.cache_clear()


def serve(fleet):
    b = FakeBackend()
    cfg = Config(vllm_boxes=(VllmBox("box-a-gpu", "box-a", b.url),), listen_host="127.0.0.1", listen_port=0,
                 fleet=fleet)
    p = Proxy(cfg, Hub(cfg))
    p.start()
    return p, b


@pytest.mark.parametrize("path", ["/", "/index.html", "/benchmarks", "/sessions"])
def test_the_lb_serves_every_page_with_the_fleets_default(path):
    p, b = serve(parse_fleet(with_style(PAIR, "glass"), "fleet.yml"))
    try:
        c = http.client.HTTPConnection("127.0.0.1", p.port, timeout=10)
        c.request("GET", path)
        r = c.getresponse()
        body = r.read().decode()
        assert r.status == 200
        assert 'data-style-default="glass" data-style-default-href="/styles/glass.css' in body, body[:300]
    finally:
        p.stop()
        b.close()


def test_an_lb_without_a_fleet_stamps_nothing_and_the_page_says_so():
    p, b = serve(None)
    try:
        c = http.client.HTTPConnection("127.0.0.1", p.port, timeout=10)
        c.request("GET", "/")
        body = c.getresponse().read().decode()
        assert '<html lang="en" data-style-default="" data-style-default-href="">' in body
    finally:
        p.stop()
        b.close()


# ── every shipped fleet names it ────────────────────────────────────────────────────────────────────

def test_every_shipped_fleet_example_names_its_panel_style():
    for p in sorted(LB.glob("fleet*.example.yml")):
        assert re.search(r"^panel:.*\n  style: [a-z]", p.read_text(), re.M), f"{p.name} names no panel.style"
