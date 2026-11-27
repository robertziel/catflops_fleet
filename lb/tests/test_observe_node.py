"""The observability views (check, history, daily) are built in the browser (static/observe.js), so
they are checked in the browser's language: tests/observe.node.js. This runs it when node is on the path, and says
where to run it when not (the LB image ships no node —
`docker run --rm -v "$PWD/lb":/app -w /app node:22-slim node tests/observe.node.js`)."""
from __future__ import annotations

import pathlib
import re
import shutil
import subprocess

import pytest

HERE = pathlib.Path(__file__).resolve().parent
STATIC = HERE.parent / "vllm_lb" / "static"


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node is not in the LB image: run tests/observe.node.js under node:22-slim")
def test_the_views_draw_figures_and_escape_the_servers_words():
    r = subprocess.run(["node", str(HERE / "observe.node.js")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ok" in r.stdout


def test_the_page_loads_observe_after_gateway_and_before_layout():
    html = (STATIC / "index.html").read_text()
    order = [html.index(f'src="/{name}') for name in ("gateway.js", "observe.js", "layout.js")]
    assert order == sorted(order), "observe.js uses gateway.js's helpers; layout.js boots last"
    assert 'href="/observe.css' in html
    assert "observePaint" in (STATIC / "gateway.js").read_text()


def test_the_build_stamps_the_observe_files():
    docker = (HERE.parent / "Dockerfile").read_text()
    loop = docker.split("for f in", 1)[1].split(";", 1)[0]
    for name in ("observe.js", "observe.css"):
        assert name in loop, f"{name} is in the stamp loop"
        assert f"grep -q '{name}?v=' index.html" in docker, f"the build fails if {name}'s stamp missed"


def _phone_rule(css: str, selector: str) -> str:
    """The declarations `selector` gets inside the stylesheet's `@media (max-width: 600px)` block."""
    block = css.split("@media (max-width: 600px)", 1)[1]
    hit = re.search(r"(?m)^\s*" + re.escape(selector) + r"\s*\{([^}]*)\}", block)
    assert hit, f"{selector} has a phone rule"
    return hit.group(1)


def test_a_phone_tap_target_is_44px_each_way():
    """The UX audit: the card buttons (check / history / daily), the history range toggle and the Activity kind
    chips were 32 px on a phone — under the 44 px a finger needs."""
    observe = (STATIC / "observe.css").read_text()
    health = (STATIC / "health.css").read_text()
    for css, sel in ((observe, ".obs-btn"), (observe, ".obs-tier"), (health, ".ev-chip")):
        rule = _phone_rule(css, sel)
        assert "min-height: 44px" in rule and "min-width: 44px" in rule, (sel, rule)


def test_the_card_buttons_share_one_neutral_style():
    """check · history · daily are one look — `.obs-btn`, the ink-dim pill — never connect's accent `.cn-open`."""
    js = (STATIC / "observe.js").read_text()
    assert "btn.className = 'obs-btn obs-daily'" in js
    assert "cn-open obs-daily" not in js
    css = (STATIC / "observe.css").read_text()
    rule = re.search(r"(?m)^\.obs-btn \{([^}]*)\}", css).group(1)
    assert "color: var(--ink-dim)" in rule and "background: transparent" in rule
