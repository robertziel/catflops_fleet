"""The swarm's DURATION as the browser computes it — run under node, because that is where it is computed.

Until 2026-10-05 the cap was a fixed fifteen minutes everywhere, the page's label included ("Test swarm
running · 0.3 of 15 min"). The LB's half and the host runner's half are pinned from Python
(tests/test_bench_swarm.py, tests/test_bench_runner.py); tests/swarm_duration.node.js holds the two claims
only the page can answer — the label names the chosen duration (the chooser's while none runs, THAT
session's while one does) and the track along the button scales to it, so a two-hour run crawls where a
quarter-hour one fills. It also holds the refusal: a number outside the range /lb/status publishes greys
the button before the three-second hold, never after it.

This runs it when node is on the path, and says where to run it when not (the LB image ships no node —
`docker run --rm -v "$PWD/lb":/app -w /app node:22-slim node tests/swarm_duration.node.js`)."""
from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

HERE = pathlib.Path(__file__).resolve().parent


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node is not in the LB image: run tests/swarm_duration.node.js under node:22-slim")
def test_the_label_and_the_progress_bar_are_the_duration_that_was_chosen():
    r = subprocess.run(["node", str(HERE / "swarm_duration.node.js")],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ok" in r.stdout
