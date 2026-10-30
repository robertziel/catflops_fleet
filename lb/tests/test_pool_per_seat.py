"""A KV pool reserved one window a seat (TensorFold): tests/pool_per_seat.node.js checks the page sizes it as seats x
context at the model's bytes a token — never the gateway's pin — and draws the KV tile's share of that pool. This runs
it when node is on the path, and says where to run it when not (the LB image ships no node —
`docker run --rm -v "$PWD/lb":/app -w /app node:22-bookworm-slim node tests/pool_per_seat.node.js`)."""
from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

HERE = pathlib.Path(__file__).resolve().parent


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node is not in the LB image: run tests/pool_per_seat.node.js under node:22-bookworm-slim")
def test_a_pool_reserved_per_seat_is_seats_times_context():
    r = subprocess.run(["node", str(HERE / "pool_per_seat.node.js")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ok" in r.stdout
