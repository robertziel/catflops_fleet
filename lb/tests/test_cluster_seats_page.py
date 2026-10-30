"""The page's DEMOTED reads a cluster's own seats (`cluster.max_depth`), never the classic boxes' limit:
tests/cluster_seats.node.js. This runs it when node is on the path, and says where to run it when not (the LB image
ships no node — `docker run --rm -v "$PWD/lb":/lb:ro -w /lb node:22-slim node tests/cluster_seats.node.js`)."""
from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

HERE = pathlib.Path(__file__).resolve().parent


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node is not in the LB image: run tests/cluster_seats.node.js under node:22-slim")
def test_the_page_demotes_a_cluster_by_its_own_seats():
    r = subprocess.run(["node", str(HERE / "cluster_seats.node.js")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ok" in r.stdout
