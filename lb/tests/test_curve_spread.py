"""The batch-size curve's p10 / p90 and distribution are computed in the browser (benchmarks.js
`levelFigures`), off the per-level TTFT histograms perflog records. tests/curve_spread.node.js checks
that arithmetic under node; this runs it when node is on the path, and says where to run it when not
(the LB image ships no node — `docker run --rm -v "$PWD/lb":/app -w /app node:22-slim node
tests/curve_spread.node.js`). The DOM side — the whisker, the tooltip, the matrix selection scoping
the charts — has no seam a test can reach without a browser; that is the Chromium audit's job."""
from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

HERE = pathlib.Path(__file__).resolve().parent


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node is not in the LB image: run tests/curve_spread.node.js under node:22-slim")
def test_the_curves_spread_is_the_levels_own_histogram_read_exactly():
    r = subprocess.run(["node", str(HERE / "curve_spread.node.js")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ok" in r.stdout
