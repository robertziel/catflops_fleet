"""A node the LB requests on the page — load, unload, select the model, set its parameters, the fit in the fleet's
numbers, the applier's own words: tests/cluster_local.node.js. This runs it when node is on the path, and says where
to run it when not (the LB image ships no node —
`docker run --rm -v "$PWD/lb":/lb:ro -w /lb node:22-slim node tests/cluster_local.node.js`)."""
from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

HERE = pathlib.Path(__file__).resolve().parent


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node is not in the LB image: run tests/cluster_local.node.js under node:22-slim")
def test_a_local_node_is_loaded_unloaded_and_set_from_the_page():
    r = subprocess.run(["node", str(HERE / "cluster_local.node.js")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ok" in r.stdout
