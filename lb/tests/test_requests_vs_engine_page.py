"""The REQUESTS card's two counts, on the page: tests/requests_vs_engine.node.js checks the gateway's count beside
the engine's, their difference and its why, the QUEUE tile saying whose count it is, and the just-ended rows kept
in the live list, where gateway.js builds them. This runs it when node is on the path, and says where to run it when
not (the LB image ships no node — `docker run --rm -v "$PWD/lb":/app -w /app node:22-slim node
tests/requests_vs_engine.node.js`)."""
from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

HERE = pathlib.Path(__file__).resolve().parent


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node is not in the LB image: run tests/requests_vs_engine.node.js under node:22-slim")
def test_the_gateway_count_stands_beside_the_engine_count():
    r = subprocess.run(["node", str(HERE / "requests_vs_engine.node.js")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ok" in r.stdout
