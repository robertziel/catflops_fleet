"""The measured-best start input ON THE PAGE: tests/tuned_defaults.node.js checks that picking another model moves
the concurrency, context, prefill budget and drafter depth to THAT model's own measured figures, says they were
measured and names the run, and leaves a model with nothing measured at the LB's published defaults. This runs it
when node is on the path, and says where to run it when not (the LB image ships no node —
`docker run --rm -v "$PWD/lb":/app -w /app node:22-bookworm-slim node tests/tuned_defaults.node.js`)."""
from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

HERE = pathlib.Path(__file__).resolve().parent


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node is not in the LB image: run tests/tuned_defaults.node.js under node:22-bookworm-slim")
def test_switching_the_model_fills_the_start_input_with_that_models_measured_best():
    r = subprocess.run(["node", str(HERE / "tuned_defaults.node.js")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ok" in r.stdout
