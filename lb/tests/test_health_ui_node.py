"""The health band and the Activity feed in the browser's language: tests/health_ui.node.js. Runs when node
is on the path; the LB image ships none — `docker run --rm -v "$PWD/lb":/app:ro -w /app node:22-slim node
tests/health_ui.node.js`."""
from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

HERE = pathlib.Path(__file__).resolve().parent


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node is not in the LB image: run tests/health_ui.node.js under node:22-slim")
def test_the_band_and_the_feed_say_figures_escaped():
    r = subprocess.run(["node", str(HERE / "health_ui.node.js")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ok" in r.stdout
