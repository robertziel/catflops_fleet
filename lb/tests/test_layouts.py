"""THE PANEL'S LAYOUT TEMPLATES (operator, 2026-10-30: "implement custom UX UI, you can choose template in config, you
can arrange elements in the view, move around, make it all stateless, per browser that keeps state data").

What is pinned here, and why:

  THE REGISTRY   every movable element of the panel is a host in index.html carrying `data-ly="<id>"` and
                 `data-ly-label`: the ONE place its id lives. The templates name those ids and layout.js reads them
                 from the DOM at boot — a second list would drift from the markup.
  THE TEMPLATES  `vllm_lb/static/layouts/<name>.json`, one file per template; `layouts/index.json` names them in
                 menu order (the LB serves no directory listing) and the FIRST is what a fresh browser starts with —
                 "the template in config" is this committed file, not an env key: nothing about a layout is a
                 deployment's. A template is COMPLETE (every registry id placed or hidden) and names no id the
                 registry lacks: the engine shows an unknown id as a visible notice, never drops it, and a shipped
                 file must never trigger that.
  THE DEFAULT    reproduces the panel as it was before layouts existed: every element shown, in document order.
  STATELESS      the server stores nothing: no endpoint, no file, no env name for a user's layout. The state is the
                 browser's (localStorage, `lb.layout`) — tests/layout.node.js covers it under node.
  THE BUILD      lb/Dockerfile stamps every static ref with a content hash and FAILS on a miss: the new files are in
                 the loop and the page links them exactly as the others.
"""
from __future__ import annotations

import http.client
import json
import pathlib
import re
import shutil
import subprocess

import pytest

from tests.fakes import FakeBackend
from vllm_lb.config import Config, VllmBox
from vllm_lb.hub import Hub
from vllm_lb.proxy import Proxy

HERE = pathlib.Path(__file__).resolve().parent
LB = HERE.parent
STATIC = LB / "vllm_lb" / "static"
LAYOUTS = STATIC / "layouts"

#: layout.js `LY_WIDTHS` / `LY_SIZES`: a column's width token, an item's span. Pinned here so a template cannot
#: carry a token the engine refuses at runtime.
WIDTHS = {"side", "1", "2", "3"}
SIZES = {"full", "half"}
MAX_COLS = 3


def registry() -> list[str]:
    """The movable elements, in document order — index.html is the registry."""
    html = (STATIC / "index.html").read_text()
    ids = re.findall(r'data-ly="([^"]+)"', html)
    assert ids, "index.html names no movable element (data-ly)"
    return ids


def template_files() -> list[pathlib.Path]:
    return sorted(p for p in LAYOUTS.glob("*.json") if p.name != "index.json")


def items_of(template: dict) -> list[str]:
    out = []
    for row in template["rows"]:
        for col in row["cols"]:
            for it in col["items"]:
                out.append(it if isinstance(it, str) else it["id"])
    return out


# ── the registry ────────────────────────────────────────────────────────────────────────────────────

def test_every_movable_element_is_a_host_with_a_stable_id_and_a_label():
    html = (STATIC / "index.html").read_text()
    ids = registry()
    assert len(ids) == len(set(ids)), f"a layout id is given twice: {ids}"
    for id_ in ids:
        assert re.search(rf'data-ly="{id_}" data-ly-label="[^"]+"', html), f"{id_} has no label"
    # The seven parts of the panel (operator 2026-10-17/18: the queued leases, the Cluster of boxes, the server
    # tabs, the selected server's card, its engine's throughput, its requests, its saturation).
    assert ids == ["queued", "cluster", "tabs", "server", "engine", "requests", "saturation"]
    main = re.search(r'<main id="layout" class="layout"[^>]*>(.*?)</main>', html, re.S)
    assert main, "the movable elements sit in <main id=layout>"
    assert re.findall(r'data-ly="([^"]+)"', main.group(1)) == ids, "every host is a direct part of the layout"
    assert 'id="layout-hidden"' in html, "the tray a hidden element is parked in (its bindings keep running)"


# ── the templates ───────────────────────────────────────────────────────────────────────────────────

def test_the_index_names_every_shipped_template_and_the_default_first():
    index = json.loads((LAYOUTS / "index.json").read_text())
    names = index["templates"]
    assert isinstance(names, list) and names and all(isinstance(n, str) for n in names)
    assert sorted(names) == sorted(p.stem for p in template_files()), "index.json and the files on disk disagree"
    assert names[0] == "default", "a fresh browser starts with the first template, which is `default`"
    assert len(names) >= 3, "the library is a choice: default plus at least two others"


@pytest.mark.parametrize("path", template_files(), ids=lambda p: p.stem)
def test_a_shipped_template_is_complete_and_names_only_registry_ids(path):
    t = json.loads(path.read_text())
    assert t["schema"] == 1 and t["name"] == path.stem
    assert isinstance(t["label"], str) and t["label"] and isinstance(t["what"], str) and t["what"]
    ids = set(registry())
    placed = items_of(t)
    hidden = t.get("hidden", [])
    assert isinstance(hidden, list)
    named = placed + hidden
    assert len(named) == len(set(named)), f"{path.name} names an element twice: {named}"
    assert set(named) - ids == set(), f"{path.name} names an element the registry lacks: {set(named) - ids}"
    assert ids - set(named) == set(), f"{path.name} leaves an element unnamed (neither placed nor hidden): {ids - set(named)}"
    assert t["rows"], f"{path.name} has no row"
    for row in t["rows"]:
        assert set(row) == {"cols"}, f"{path.name}: a row holds only `cols`: {sorted(row)}"
        assert 1 <= len(row["cols"]) <= MAX_COLS
        for col in row["cols"]:
            assert set(col) == {"width", "items"}, f"{path.name}: a column holds `width` and `items`: {sorted(col)}"
            assert col["width"] in WIDTHS, f"{path.name}: width {col['width']!r} is not one of {sorted(WIDTHS)}"
            for it in col["items"]:
                if not isinstance(it, str):
                    assert set(it) <= {"id", "size"} and "id" in it, f"{path.name}: item keys {sorted(it)}"
                    assert it["size"] in SIZES


def test_the_default_template_is_the_panel_as_it_was():
    """Every element shown, in document order: the queued leases, the Cluster, the tabs on top; the server card in
    the side column; throughput, requests and saturation beside it (gateway.css `.fleet`, before 2026-10-30)."""
    t = json.loads((LAYOUTS / "default.json").read_text())
    assert t.get("hidden", []) == []
    assert items_of(t) == registry()
    assert [[c["width"] for c in r["cols"]] for r in t["rows"]] == [["1"], ["side", "1"]]
    assert t["rows"][1]["cols"][0]["items"] == ["server"]
    assert all(isinstance(it, str) for r in t["rows"] for c in r["cols"] for it in c["items"]), "nothing is half-width"


# ── stateless: served, never stored ─────────────────────────────────────────────────────────────────

@pytest.fixture
def proxy():
    b = FakeBackend()
    cfg = Config(vllm_boxes=(VllmBox("box-a-gpu", "box-a", b.url),), listen_host="127.0.0.1", listen_port=0)
    p = Proxy(cfg, Hub(cfg))
    p.start()
    yield p
    p.stop()
    b.close()


def test_the_templates_are_served_as_json_and_a_layout_is_never_posted(proxy):
    c = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=10)
    c.request("GET", "/layouts/index.json")
    r = c.getresponse()
    assert r.status == 200 and "json" in (r.getheader("Content-Type") or "")
    assert json.loads(r.read())["templates"][0] == "default"
    c.request("GET", "/layouts/default.json")
    r = c.getresponse()
    assert r.status == 200 and json.loads(r.read())["name"] == "default"
    # Nothing to write to: the LB's own surface takes GET/HEAD only outside /v1/ (proxy.py), and no handler reads a
    # layout — the state is the browser's.
    c.request("POST", "/layouts/default.json", body="{}", headers={"Content-Type": "application/json"})
    r = c.getresponse()
    assert r.status == 405
    r.read()


def test_the_server_keeps_no_layout_and_reads_no_layout_env():
    for p in (LB / "vllm_lb").rglob("*.py"):
        text = p.read_text()
        assert "LB_LAYOUT" not in text and "layouts/" not in text, f"{p.name} knows about layouts: the server must not"
    assert "LAYOUT" not in (LB / ".env.example").read_text()
    assert "layout" not in (LB / "docker-compose.yml").read_text().lower()


# ── the page and the build ──────────────────────────────────────────────────────────────────────────

def test_the_layout_module_is_wired_into_the_page_and_the_build_stamps_it():
    html = (STATIC / "index.html").read_text()
    assert '<script src="/layout.js"></script>' in html and '<link rel="stylesheet" href="/layout.css">' in html
    assert html.index('"/gateway.js"') < html.index('"/layout.js"'), "the engine moves what gateway.js draws: loaded after it"
    dockerfile = (LB / "Dockerfile").read_text()
    loop = re.search(r"for f in ([^;]+); do", dockerfile).group(1).split()
    assert "layout.js" in loop and "layout.css" in loop
    assert "grep -q 'layout.js?v=' index.html" in dockerfile and "grep -q 'layout.css?v=' index.html" in dockerfile
    assert 'id="layout-button"' in html and 'id="layout-menu"' in html, "the header's Layout control and its menu"
    assert 'id="layout-notice"' in html, "where an unknown element, a lost store or a failed template is said"
    assert re.search(r'id="cluster-panel"[^>]*role="tabpanel"', html), "a server tab controls the selected server's card"
    assert 'aria-controls="cluster-panel"' in (STATIC / "clusters.js").read_text()
    gateway = (STATIC / "gateway.js").read_text()
    assert "no-cluster" not in gateway, "an empty column collapses by the layout's own rule, not the card's parent"


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node is not in the LB image: run tests/layout.node.js under node:22-slim")
def test_the_layout_engine_under_node():
    r = subprocess.run(["node", str(HERE / "layout.node.js")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ok" in r.stdout
