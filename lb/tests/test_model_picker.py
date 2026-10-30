"""WHAT A MODEL PICKER ROW SAYS — the operator (2026-10-09): "make select more detailed, include tok/s targets
if set". tests/model_picker.node.js asserts it in the browser's language against the SHIPPED catalog: every row
names its checkpoint and what a Start of it launches (`cluster.setups`, the same resolve as `cluster.defaults`)
in the cluster card's words, its rate to aim for (`setup_targets`) only when a minute was recorded, the twin's
own and never the original's, and renderModelChoice writes one option per checkpoint. This runs it when node is
on the path, and says where to run it when not (the LB image ships no node —
`docker run --rm -v "$PWD/lb":/app -w /app node:22-slim node tests/model_picker.node.js`)."""
from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

HERE = pathlib.Path(__file__).resolve().parent


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node is not in the LB image: run tests/model_picker.node.js under node:22-slim")
def test_each_picker_row_says_what_it_launches_and_the_rate_it_ran():
    r = subprocess.run(["node", str(HERE / "model_picker.node.js")],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ok" in r.stdout
