"""Each cluster's server card reads ITS OWN view, never the selected one's (worklog/parallel-clusters.md, AUDIT-02):
tests/parallel_cards.node.js. This runs it when node is on the path, and says where to run it when not (the LB image
ships no node — `docker run --rm -v "$PWD/lb":/lb:ro -w /lb node:22-slim node tests/parallel_cards.node.js`)."""
from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

HERE = pathlib.Path(__file__).resolve().parent


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node is not in the LB image: run tests/parallel_cards.node.js under node:22-slim")
def test_every_cluster_card_reads_its_own_view():
    r = subprocess.run(["node", str(HERE / "parallel_cards.node.js")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ok" in r.stdout
