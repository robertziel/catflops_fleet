"""The cluster card names what the cluster IS (`cluster.topology`): tests/cluster_topology.node.js checks the pair
renders as it always did — with the field and without it — and one observed node is named by its box, TP=1, with only
the start steps that apply and no Start/Stop. This runs it when node is on the path, and says where to run it when not
(the LB image ships no node — `docker run --rm -v "$PWD/lb":/lb:ro -w /lb node:22-slim node tests/cluster_topology.node.js`)."""
from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

HERE = pathlib.Path(__file__).resolve().parent


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node is not in the LB image: run tests/cluster_topology.node.js under node:22-slim")
def test_the_cluster_card_names_the_topology_it_is_given():
    r = subprocess.run(["node", str(HERE / "cluster_topology.node.js")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ok" in r.stdout
