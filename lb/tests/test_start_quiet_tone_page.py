"""A quiet start is a timer, not an error, on the page: tests/start_quiet_tone.node.js checks the trail's tones past the
LB's own thresholds (amber, then bold amber — never red) and that the GiB a load has taken (`loading.taken_gib`) moves
the page's clock and shows on the row, in the browser's own language. This runs it when node is on the path, and says
where to run it when not (the LB image ships no node —
`docker run --rm -v "$PWD/lb":/lb:ro -w /lb node:22-slim node tests/start_quiet_tone.node.js`)."""
from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

HERE = pathlib.Path(__file__).resolve().parent


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node is not in the LB image: run tests/start_quiet_tone.node.js under node:22-slim")
def test_a_quiet_start_is_amber_never_red_and_a_loads_memory_moves_the_pages_clock():
    r = subprocess.run(["node", str(HERE / "start_quiet_tone.node.js")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ok" in r.stdout
