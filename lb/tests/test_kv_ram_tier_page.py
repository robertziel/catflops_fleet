"""The KV tier in the host's RAM on the engine card: tests/kv_ram_tier.node.js checks, in the browser's own language,
that the card draws "kv · ram" and "reload · store" beside the KV tile from `throughput.kv_offload` and its rings,
nothing for a server without the tier, "—" with its reason for a figure not measured yet, and a grid of seven that
wraps on a phone. This runs it when node is on the path, and says where to run it when not (the LB image ships no node —
`docker run --rm -v "$PWD/lb":/lb:ro -w /lb node:22-slim node tests/kv_ram_tier.node.js`)."""
from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

HERE = pathlib.Path(__file__).resolve().parent


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node is not in the LB image: run tests/kv_ram_tier.node.js under node:22-slim")
def test_the_engine_card_draws_the_ram_tier_beside_the_kv_tile_and_nothing_without_one():
    r = subprocess.run(["node", str(HERE / "kv_ram_tier.node.js")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ok" in r.stdout
