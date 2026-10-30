"""The per-request line on a machine card's rate graphs: tests/per_request_line.node.js checks that it
is the arithmetic it claims (the rate ÷ the requests the engine was running), that it shares the
headline's axis, that a sample with nothing running is a gap rather than a zero, and that it stays
subordinate — thinner, grey, half-transparent, never glowing. This runs it when node is on the path,
and says where to run it when not (the LB image ships no node —
`docker run --rm -v "$PWD/lb":/app -w /app node:22-bookworm-slim node tests/per_request_line.node.js`)."""
from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

HERE = pathlib.Path(__file__).resolve().parent


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node is not in the LB image: run tests/per_request_line.node.js under node:22-bookworm-slim")
def test_the_per_request_line_is_the_rate_over_the_running_requests():
    r = subprocess.run(["node", str(HERE / "per_request_line.node.js")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ok" in r.stdout
