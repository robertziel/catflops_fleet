"""Box health on a machine card is drawn in the browser (static/hwextra.js, called by gateway.js
`machineBody`), so it is checked in the browser's language: tests/hwextra.node.js. This runs it when node is on the
path, and says where to run it when not (the LB image ships no node —
`docker run --rm -v "$PWD/lb":/app -w /app node:22-slim node tests/hwextra.node.js`)."""
from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

HERE = pathlib.Path(__file__).resolve().parent


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node is not in the LB image: run tests/hwextra.node.js under node:22-slim")
def test_box_health_on_the_machine_card():
    r = subprocess.run(["node", str(HERE / "hwextra.node.js")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ok" in r.stdout
