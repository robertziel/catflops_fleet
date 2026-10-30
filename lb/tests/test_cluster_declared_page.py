"""What the fleet declares a node runs, on the page: tests/cluster_declared.node.js checks the declared catalog row is
the model (the node's own engine, no drafter or size from the row's pair profile), the declared seats with their
source, agreement marked where it is checked, the fleet-only keys on one line, and a disagreement drawn loudly with
both values — never resolved. This runs it when node is on the path, and says where to run it when not (the LB image
ships no node — `docker run --rm -v "$PWD/lb":/lb:ro -w /lb node:22-slim node tests/cluster_declared.node.js`)."""
from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

HERE = pathlib.Path(__file__).resolve().parent


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node is not in the LB image: run tests/cluster_declared.node.js under node:22-slim")
def test_the_fleets_declaration_is_drawn_beside_the_nodes_own_report():
    r = subprocess.run(["node", str(HERE / "cluster_declared.node.js")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ok" in r.stdout
