"""How long the pair has been up, on the page: tests/cluster_uptime.node.js checks the stats row's uptime
(`cluster.up_since`), in the browser's own language. This runs it when node is on the path,
and says where to run it when not (the LB image ships no node —
`docker run --rm -v "$PWD/lb":/app -w /app node:22-bookworm-slim node tests/cluster_uptime.node.js`)."""
from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

HERE = pathlib.Path(__file__).resolve().parent


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node is not in the LB image: run tests/cluster_uptime.node.js under node:22-bookworm-slim")
def test_the_stats_row_says_how_long_the_pair_has_been_up():
    r = subprocess.run(["node", str(HERE / "cluster_uptime.node.js")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ok" in r.stdout
