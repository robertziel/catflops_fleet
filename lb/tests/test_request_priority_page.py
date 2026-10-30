"""The REQUESTS card's priority square, on the page: tests/request_priority.node.js checks the square, its green state,
the one POST per click and the lock, where gateway.js builds them. This runs it when node is on the path, and says where
to run it when not (the LB image ships no node —
`docker run --rm -v "$PWD/lb":/lb:ro -w /lb node:22-slim node tests/request_priority.node.js`)."""
from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

HERE = pathlib.Path(__file__).resolve().parent


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node is not in the LB image: run tests/request_priority.node.js under node:22-slim")
def test_the_priority_square_toggles_with_one_post_and_shows_when_it_is_on_or_locked():
    r = subprocess.run(["node", str(HERE / "request_priority.node.js")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ok" in r.stdout
