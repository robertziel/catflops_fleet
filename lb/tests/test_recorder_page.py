"""The flight recorder's strip, open view and expand view are built in the browser (static/recorder.js), so they are
checked in the browser's language: tests/recorder.node.js — one line of figures, n/a with why for what a box does not
send (never 0), containers with "+k more", the charts as the panel's own tiles, the open state kept per browser. This
runs it when node is on the path, and says where to run it when not (the LB image ships no node —
`docker run --rm -v "$PWD/lb":/app -w /app node:22-slim node tests/recorder.node.js`)."""
from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

HERE = pathlib.Path(__file__).resolve().parent
STATIC = HERE.parent / "vllm_lb" / "static"


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node is not in the LB image: run tests/recorder.node.js under node:22-slim")
def test_the_recorder_draws_figures_only_and_says_why_one_is_missing():
    r = subprocess.run(["node", str(HERE / "recorder.node.js")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ok" in r.stdout


def test_the_page_loads_the_recorder_after_gateway_and_before_layout():
    html = (STATIC / "index.html").read_text()
    order = [html.index(f'src="/{name}') for name in ("gateway.js", "recorder.js", "layout.js")]
    assert order == sorted(order), "recorder.js uses gateway.js's helpers; layout.js boots last"
    assert 'href="/recorder.css' in html


def test_the_build_stamps_the_recorder_files():
    docker = (HERE.parent / "Dockerfile").read_text()
    loop = docker.split("for f in", 1)[1].split(";", 1)[0]
    for name in ("recorder.js", "recorder.css"):
        assert name in loop, f"{name} is in the stamp loop"
        assert f"grep -q '{name}?v=' index.html" in docker, f"the build fails if {name}'s stamp missed"


def test_a_machine_card_and_its_recorder_share_one_slot_so_the_card_never_grows():
    """The strip is the card's sibling in a two-row subgrid (gateway.css `.machine-slot`): a closed strip adds no
    height to the card, an open one grows only the strip's row. The power control still finds
    `#lb-backends article.card` (power/fleet_power/static/button.js)."""
    js = (STATIC / "gateway.js").read_text()
    assert "machine-slot" in js and "recorderPaint" in js
    css = (STATIC / "recorder.css").read_text()
    assert "grid-template-rows: subgrid" in css
