"""The model's tags, on the page: tests/model_tags.node.js checks that the picker's trigger and rows, the cluster card's
model line and a server card's badge wear ONE colour-coded tag family from one facts function — the engine in two
hues, UNCENSORED loud when the generation shown launched the twin (read from the launch record, never guessed), the
drafter with its depth, the figures as neutral chips. This runs it when node is on the path, and says where to run it
when not (the LB image ships no node — `docker run --rm -v "$PWD/lb":/lb:ro -w /lb node:22-slim node tests/model_tags.node.js`)."""
from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

HERE = pathlib.Path(__file__).resolve().parent


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node is not in the LB image: run tests/model_tags.node.js under node:22-slim")
def test_every_place_a_model_is_shown_wears_the_same_tags():
    r = subprocess.run(["node", str(HERE / "model_tags.node.js")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ok" in r.stdout
