"""The NETWORK tile on the engine card only where there is a link between ranks: tests/network_tile.node.js checks, in
the browser's own language, that a one-rank cluster (`cluster.topology.tp` 1) draws no network tile, no placeholder
and no note, its remaining ceiling tiles on a grid of their own count (no empty column), while the TP=2 pair keeps the
tile exactly where and as it was. This runs it when node is on the path, and says where to run it when not (the LB
image ships no node — `docker run --rm -v "$PWD/lb":/lb:ro -w /lb node:22-slim node tests/network_tile.node.js`)."""
from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

HERE = pathlib.Path(__file__).resolve().parent


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node is not in the LB image: run tests/network_tile.node.js under node:22-slim")
def test_a_one_rank_card_draws_no_network_tile_and_the_pair_keeps_its_own():
    r = subprocess.run(["node", str(HERE / "network_tile.node.js")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ok" in r.stdout
