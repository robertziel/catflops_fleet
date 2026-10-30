"""Red is an error, never a clock, on the page: tests/cluster_tone.node.js checks that the gateway's cluster panel
paints `failed` alone red, `stalled` (a hung start the LB halted) and `unknown` (a box it could not observe) amber
with the planner's reason quoted, in the browser's own language. This runs it when node is on the path, and says
where to run it when not (the LB image ships no node —
`docker run --rm -v "$PWD/lb":/lb:ro -w /lb node:22-slim node tests/cluster_tone.node.js`)."""
from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

HERE = pathlib.Path(__file__).resolve().parent


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node is not in the LB image: run tests/cluster_tone.node.js under node:22-slim")
def test_only_a_real_failure_is_painted_red_on_the_cluster_panel():
    r = subprocess.run(["node", str(HERE / "cluster_tone.node.js")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ok" in r.stdout
