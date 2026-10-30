"""A real-traffic configuration's one encoding on /benchmarks — its cap's ramp step, a hatch for a second
configuration at the same cap, the dashes the time charts use, which four measures lead — is computed in
the browser (benchmarks.js `cfgStyle`, `recomputeDomains`, `CURVE`). tests/level_style.node.js checks it
under node; this runs it when node is on the path, and says where to run it when not (the LB image ships
no node — `docker run --rm -v "$PWD/lb":/app -w /app node:22-slim node tests/level_style.node.js`). The
drawn result — grouped bars, ghost bars, the clipped whisker — is the Chromium audit's job
(worklog/benchmarks-redesign/audit.py)."""
from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

HERE = pathlib.Path(__file__).resolve().parent


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node is not in the LB image: run tests/level_style.node.js under node:22-slim")
def test_a_configuration_is_drawn_one_way_everywhere():
    r = subprocess.run(["node", str(HERE / "level_style.node.js")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ok" in r.stdout
