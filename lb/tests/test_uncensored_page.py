"""The checkpoint control ON THE PAGE: tests/uncensored_knob.node.js checks that a model with an uncensored twin
offers it (the twin as the default, its dir and "not calibration-verified" in the note), that the pick travels as
`uncensored: true/false` and a reset as null, that a model with no twin is locked at the original, and that the
picker marks a model by the checkpoint its default launches. Runs under node when it is on the path, and says
where to run it when not (the LB image ships no node —
`docker run --rm -v "$PWD/lb":/app -w /app node:22-bookworm-slim node tests/uncensored_knob.node.js`)."""
from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

HERE = pathlib.Path(__file__).resolve().parent


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node is not in the LB image: run tests/uncensored_knob.node.js under node:22-bookworm-slim")
def test_the_checkpoint_control_offers_the_twin_where_the_catalog_names_one():
    r = subprocess.run(["node", str(HERE / "uncensored_knob.node.js")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ok" in r.stdout
