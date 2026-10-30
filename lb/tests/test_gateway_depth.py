"""The server card's queue tile is built in the browser (gateway.js `queueTile`), so what it claims
is checked in the browser's language. tests/gateway_depth.node.js asserts that the tile's headline is
the sum of the parts printed under it — the operator's own reading of a card that headed 25 over
"0 waiting · 6 running · 25 ours" — and that the ROUTER's depth is untouched by the display fix.
This runs it when node is on the path, and says where to run it when not (the LB image ships no node
— `docker run --rm -v "$PWD/lb":/app -w /app node:22-slim node tests/gateway_depth.node.js`)."""
from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

HERE = pathlib.Path(__file__).resolve().parent


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node is not in the LB image: run tests/gateway_depth.node.js under node:22-slim")
def test_the_queue_tiles_headline_is_the_sum_of_the_parts_it_prints():
    r = subprocess.run(["node", str(HERE / "gateway_depth.node.js")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ok" in r.stdout
