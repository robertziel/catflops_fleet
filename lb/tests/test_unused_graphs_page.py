"""The graphs a cluster card leaves out under its setup, named in a quiet row of tags under the card's graphs:
tests/unused_graphs.node.js checks, in the browser's own language, that a one-rank card names `network · TP=1`, a card
with no KV tier in RAM names the tier's two graphs, a setup that draws every graph names none, and an LB that does not
say whether there is a tier names none for it. This runs it when node is on the path, and says where to run it when
not (the LB image ships no node — `docker run --rm -v "$PWD/lb":/lb:ro -w /lb node:22-slim node
tests/unused_graphs.node.js`)."""
from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

HERE = pathlib.Path(__file__).resolve().parent


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node is not in the LB image: run tests/unused_graphs.node.js under node:22-slim")
def test_a_card_names_the_graphs_its_setup_does_not_collect():
    r = subprocess.run(["node", str(HERE / "unused_graphs.node.js")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ok" in r.stdout
