"""The REQUESTS card, on the page: tests/request_queue.node.js checks the rows, badges, escaping and the empty
state where gateway.js builds them, in the browser's own language. This runs it when node is on the path, and
says where to run it when not (the LB image ships no node —
`docker run --rm -v "$PWD/lb":/app -w /app node:22-bookworm-slim node tests/request_queue.node.js`)."""
from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

HERE = pathlib.Path(__file__).resolve().parent


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node is not in the LB image: run tests/request_queue.node.js under node:22-bookworm-slim")
def test_the_requests_card_draws_each_turn_escaped_and_in_place():
    r = subprocess.run(["node", str(HERE / "request_queue.node.js")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ok" in r.stdout
