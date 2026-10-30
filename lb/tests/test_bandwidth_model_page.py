"""The modeled MEMORY BANDWIDTH is labelled a model on the page (gateway.js), checked in the browser's
language: tests/bandwidth_model.node.js — 'modeled' under the label, its parts as figures in the hover,
'≥' on a floor, and an engine's own estimate drawn exactly as before.

The LB image ships no node: `docker run --rm -v "$PWD":/lb:ro -w /lb node:22-slim node tests/bandwidth_model.node.js`."""
from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

HERE = pathlib.Path(__file__).resolve().parent


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node is not in the LB image: run tests/bandwidth_model.node.js under node:22-slim")
def test_the_modeled_bandwidth_is_labelled_a_model_and_an_engine_estimate_is_not():
    r = subprocess.run(["node", str(HERE / "bandwidth_model.node.js")], capture_output=True, text=True,
                       timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ok" in r.stdout
