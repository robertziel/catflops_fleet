"""Each model chooser row says whether its checkpoint starts: tests/start_rates.node.js checks the figures the page
prints from `/lb/status.start_rates` ("1/6 · 17 %" in the verdict's tone), the sparkline of the days (a gap on a day
with no launch, never 0 %), "no starts in 7 d" for a row nothing launched, starts with no verdict named apart, an
unreadable boot log as "starts unknown", no block at all from an LB without the field, and the page's own tokens for
every colour. This runs it when node is on the path, and says where to run it when not (the LB image ships no node —
`docker run --rm -v "$PWD/lb":/lb:ro -w /lb node:22-slim node tests/start_rates.node.js`)."""
from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

HERE = pathlib.Path(__file__).resolve().parent


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node is not in the LB image: run tests/start_rates.node.js under node:22-slim")
def test_each_chooser_row_says_whether_it_starts():
    r = subprocess.run(["node", str(HERE / "start_rates.node.js")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ok" in r.stdout
