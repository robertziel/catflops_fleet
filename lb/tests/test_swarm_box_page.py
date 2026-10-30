"""The swarm control names the box its runner reports, never a literal one (tests/swarm_box.node.js).

The LB image has no node, so this runs the node test where node exists and SKIPS — naming the command —
where it does not:

    docker run --rm -v "$PWD/lb":/lb:ro -w /lb node:22-slim node tests/swarm_box.node.js
"""
from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

HERE = pathlib.Path(__file__).resolve().parent


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node is not in the LB image: run tests/swarm_box.node.js under node:22-slim")
def test_the_swarm_control_names_the_runners_box_from_its_report():
    r = subprocess.run(["node", str(HERE / "swarm_box.node.js")], capture_output=True, text=True,
                       timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "swarm_box: ok" in r.stdout
