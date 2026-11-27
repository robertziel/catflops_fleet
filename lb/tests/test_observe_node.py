"""The observability views (check, history, daily) are built in the browser (static/observe.js), so
they are checked in the browser's language: tests/observe.node.js. This runs it when node is on the path, and says
where to run it when not (the LB image ships no node —
`docker run --rm -v "$PWD/lb":/app -w /app node:22-slim node tests/observe.node.js`)."""
from __future__ import annotations

import pathlib
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
