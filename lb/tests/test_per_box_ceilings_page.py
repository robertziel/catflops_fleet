"""COMPUTE and MEMORY BANDWIDTH are drawn PER BOX USED (gateway.js), checked in the browser's language:
tests/per_box_ceilings.node.js — one tile and one meter per serving box, each its own rate against its own card's
peak, "derived" wherever a box's rate is a share of the engine's one rate.

The LB image ships no node: `docker run --rm -v "$PWD":/lb:ro -w /lb node:22-slim node tests/per_box_ceilings.node.js`."""
from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

HERE = pathlib.Path(__file__).resolve().parent


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node is not in the LB image: run tests/per_box_ceilings.node.js under node:22-slim")
def test_compute_and_bandwidth_are_drawn_per_box_used():
    r = subprocess.run(["node", str(HERE / "per_box_ceilings.node.js")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ok" in r.stdout
