"""The settings of a node this LB only observes: tests/cluster_observed_settings.node.js checks its fields show what
the node SAYS it runs with, locked, each note its source (`node` / `not exposed by the node`), the KV pool line from
the engine's own pool, the facts line, and the node's own engine word on its badge and card. This runs it when node is
on the path, and says where to run it when not (the LB image ships no node —
`docker run --rm -v "$PWD/lb":/lb:ro -w /lb node:22-slim node tests/cluster_observed_settings.node.js`)."""
from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

HERE = pathlib.Path(__file__).resolve().parent


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node is not in the LB image: run tests/cluster_observed_settings.node.js under node:22-slim")
def test_an_observed_nodes_settings_are_its_own_reports_locked():
    r = subprocess.run(["node", str(HERE / "cluster_observed_settings.node.js")], capture_output=True, text=True,
                       timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ok" in r.stdout
