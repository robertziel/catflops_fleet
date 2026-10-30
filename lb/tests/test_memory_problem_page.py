"""Not enough RAM said before the Start: tests/memory_problem.node.js checks the line the page refuses a Start
with when a box at rest cannot hold the selection. This runs it
when node is on the path, and says where to run it when not (the LB image ships no node —
`docker run --rm -v "$PWD/lb":/app -w /app node:22-bookworm-slim node tests/memory_problem.node.js`)."""
from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

HERE = pathlib.Path(__file__).resolve().parent


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node is not in the LB image: run tests/memory_problem.node.js under node:22-bookworm-slim")
def test_a_start_a_box_at_rest_cannot_hold_is_refused_on_the_page():
    r = subprocess.run(["node", str(HERE / "memory_problem.node.js")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ok" in r.stdout
