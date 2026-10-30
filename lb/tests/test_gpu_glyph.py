"""A box's GPU rows are drawn as graphics cards in the browser (gateway.js `gpuColumn`), so the card's shape, its fan
(the utilisation's ring), its VRAM chips (only where the row carries VRAM, lit by used / total) and its "not reported"
state are checked in the browser's language: tests/gpu_glyph.node.js. This runs it when node is on the path, and says
where to run it when not (the LB image ships no node —
`docker run --rm -v "$PWD/lb":/lb:ro -w /lb node:22-slim node tests/gpu_glyph.node.js`)."""
from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

HERE = pathlib.Path(__file__).resolve().parent


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node is not in the LB image: run tests/gpu_glyph.node.js under node:22-slim")
def test_each_gpu_row_is_a_graphics_card():
    r = subprocess.run(["node", str(HERE / "gpu_glyph.node.js")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ok" in r.stdout
