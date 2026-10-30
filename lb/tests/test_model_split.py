"""/benchmarks shows ONE model at a time: which model is on show (the viewer's pick, else the one with the
most recent minute), that nothing of another model reaches a chart, a hatch rank or the context table, and
that Flash-Next's 28 GB KV pin is never applied to another model's minutes — all computed in the browser
(benchmarks.js `resolveModel`, `ofModel`, `recomputeDomains`, `contextRows`). tests/model_split.node.js
checks it under node; this runs it when node is on the path, and says where to run it when not (the LB image
ships no node — `docker run --rm -v "$PWD/lb":/app -w /app node:22-slim node tests/model_split.node.js`).
The drawn result is the Chromium audit's job (worklog/benchmarks-redesign/audit.py --models split)."""
from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

HERE = pathlib.Path(__file__).resolve().parent


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node is not in the LB image: run tests/model_split.node.js under node:22-slim")
def test_the_page_shows_one_model_at_a_time():
    r = subprocess.run(["node", str(HERE / "model_split.node.js")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ok" in r.stdout
