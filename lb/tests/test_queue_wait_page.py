"""The REQUESTS card's WAIT column and the queue tile's breakdown, on the page: tests/queue_wait.node.js checks them
where gateway.js builds them. This runs it when node is on the path, and says where to run it when not (the LB image
ships no node — `docker run --rm -v "$PWD/lb":/lb:ro -w /lb node:22-slim node tests/queue_wait.node.js`)."""
from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

HERE = pathlib.Path(__file__).resolve().parent


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node is not in the LB image: run tests/queue_wait.node.js under node:22-slim")
def test_the_requests_card_says_why_a_turn_waits_and_the_queue_tile_its_parts():
    r = subprocess.run(["node", str(HERE / "queue_wait.node.js")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ok" in r.stdout
