"""The server tabs and the page's two sections (worklog/parallel-clusters.md, "Servers, not clusters"): tests/server_tabs.node.js
runs under node when it is on the path (the LB image ships none —
`docker run --rm -v "$PWD/lb":/lb:ro -w /lb node:22-slim node tests/server_tabs.node.js`); the markup is pinned here."""
from __future__ import annotations

import pathlib
import re
import shutil
import subprocess

import pytest

HERE = pathlib.Path(__file__).resolve().parent
LB = HERE.parent
STATIC = LB / "vllm_lb" / "static"


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node is not in the LB image: run tests/server_tabs.node.js under node:22-slim")
def test_one_tab_per_server_selects_it_on_the_page():
    r = subprocess.run(["node", str(HERE / "server_tabs.node.js")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ok" in r.stdout


def test_the_tabs_module_is_wired_into_the_page_and_the_build_stamps_it():
    """lb/Dockerfile rewrites every static ref with a content hash and FAILS on a miss (AGENTS.md pitfalls)."""
    html = (STATIC / "index.html").read_text()
    assert '<script src="/clusters.js"></script>' in html
    assert html.index('"/clusters.js"') < html.index('"/gateway.js"'), "the tabs' module is loaded before the page's"
    dockerfile = (LB / "Dockerfile").read_text()
    loop = re.search(r"for f in ([^;]+); do", dockerfile).group(1)
    assert "clusters.js" in loop.split()
    assert "grep -q 'clusters.js?v=' index.html" in dockerfile
    gateway = (STATIC / "gateway.js").read_text()
    assert "renderClusters(s)" in gateway, "the poll hands every status to the tabs"
    assert "function renderStatus(s)" in gateway, "the one entry a tab's selection redraws the page through"


def test_the_page_is_two_sections_the_selected_server_and_the_cluster_of_boxes():
    """Operator (2026-10-17): the upper section is the SELECTED server alone — its card, its engine, its requests;
    the machines are a separate section titled "Cluster" (the hardware), never switched by the tab; "Server" is an engine."""
    html = (STATIC / "index.html").read_text()
    tablist = re.search(r'<div id="tablist"[^>]*>(.*?)</div>', html, re.S)
    assert tablist and 'aria-label="servers"' in tablist.group(0)
    assert "tab-btn-fleet" not in html and "tab-btn-clusters" not in html and 'id="tab-clusters"' not in html, "no Fleet or Clusters tab"
    assert "clusters-panel" not in html and "cluster-pick" not in html, "the Clusters cards and the card's chips are gone: the tabs select"
    # Since 2026-10-30 WHERE the parts stand is the browser's layout (layout.js): a server tab controls the selected
    # server's card, and the card, its engine, its requests and its saturation are four tiles of the registry — the
    # machines a fifth, never the server's. The `default` template keeps the card beside the three (tests/test_layouts.py).
    assert re.search(r'<section id="cluster-panel" data-ly="server"[^>]*role="tabpanel"', html), "a tab controls the server's card"
    assert "tab-server" not in html and 'aria-controls="cluster-panel"' in (STATIC / "clusters.js").read_text()
    for own, ly in (("cluster-panel", "server"), ("lb-servers", "engine"), ("lb-requests", "requests"), ("sat-section", "saturation")):
        assert re.search(rf'id="{own}" data-ly="{ly}"', html), f"{own} is the selected server's tile {ly}"
    assert re.search(r'id="lb-backends"(?! data-ly)', html) and 'id="cluster-section" data-ly="cluster"' in html, \
        "the machines are the Cluster tile, not the server's"
    cluster = re.search(r'<details id="cluster-section" data-ly="cluster" data-ly-label="Cluster" class="section" open>\s*<summary><h2>Cluster</h2>.*?</details>', html, re.S)
    assert cluster and 'id="lb-backends" class="machines"' in cluster.group(0), "the boxes are the Cluster section"
    assert html.index('id="cluster-section"') < html.index('<nav class="tabs">'), \
        "operator 2026-10-18: the Cluster section sits over the server tabs (the default template's order is the markup's)"
    assert "<h2>Servers not serving</h2>" not in html and "<h2>Other servers</h2>" not in html, \
        "Other servers is retired (2026-10-17): the tabs are the run's servers"
    assert 'id="lb-summary"' not in html and 'id="chat"' not in html, "the Fleet and Chat sections are retired"
    assert 'aria-label="declared servers"' in html and "no server declared" in html
    assert "Start server" in html and "Start cluster" not in html
    server = html[html.index('id="cluster-panel"'):html.index('id="lb-servers"')]
    assert 'id="cluster-links"' in server, "the server's own pages: performance, sessions, connect"
    top = html.split('<header class="top">')[1].split("</header>")[0]
    assert "connect-open" not in top and "/benchmarks" not in top and "/sessions" not in top, "nothing fleet-wide has a page: no header links"
    assert "<summary>Boxes</summary>" not in html and "Planner &middot; per rank" in html, \
        "the card's disclosure is the planner's per-rank reading, the Cluster section the hardware — named apart"


def test_the_operator_reads_server_never_cluster_in_the_panel_words():
    """Code names, API fields and ids keep `cluster`; what is SHOWN says server. The API's own sentences (a 409's
    message) are the API's words and are out of scope here (worklog)."""
    html = (STATIC / "index.html").read_text()
    body = re.sub(r"<!--.*?-->", "", html, flags=re.S)
    shown = re.findall(r'>([^<]*?)<|(?:aria-label|title|placeholder)="([^"]*)"', body)
    said = [a or b for a, b in shown if re.search(r"cluster", a or b, re.I)]
    assert said == ["Cluster"], f"the only visible 'cluster' is the hardware section's title: {said}"
    js = (STATIC / "gateway.js").read_text()
    for label in ("'Stop cluster'", "'Start cluster'", "'Stop node'", "'Start node'", "cluster on'", "rank of cluster"):
        assert label not in js, label
    assert "'Stop server'" in js and "'Start server'" in js
    css = (STATIC / "gateway.css").read_text()
    assert "cl-chip" not in css and ".cl-open" not in css and ".cl-engine" not in css, "the Clusters cards' styles are gone"
    phone = re.search(r"@media \(max-width: 600px\) \{\s*/\* 44 px touch targets[^}]*\}[^}]*\}", css)
    assert phone and '[role="tab"]' in phone.group(0) and ".cl-act" in phone.group(0) and "min-height: 44px" in phone.group(0)
    assert "auto-fit, minmax(340px" in css


def test_the_servers_pages_take_a_server_and_say_what_they_cannot_split():
    """performance / sessions `?server=<name>`: the title names the server; the perf log carries no server name
    (control._record feeds none while several run), so the figures are replaced by the reason, never another server's."""
    for page, js in (("benchmarks", "benchmarks.js"), ("sessions", "sessions.js")):
        html = (STATIC / f"{page}.html").read_text()
        src = (STATIC / js).read_text()
        assert "serverScope(" in src, f"{js} reads ?server="
        assert 'id="server-only"' in html, f"{page}.html has the slot the reason is written to"
    gateway = (STATIC / "gateway.js").read_text()
    assert "?server=${encodeURIComponent(c.name)}" in gateway and "`/benchmarks${q}`" in gateway and "`/sessions${q}`" in gateway
    assert "function scopeQueue(" in gateway, "the Requests card is the selected server's rows, counts and decode"
