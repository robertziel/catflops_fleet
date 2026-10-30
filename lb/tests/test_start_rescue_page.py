"""The rescue of a hung start, on the page: tests/start_rescue.node.js checks the line the trail carries while a
relaunch is on its way (`cluster.rescue`), in the browser's own language. This runs it when node is on the path,
and says where to run it when not (the LB image ships no node —
`docker run --rm -v "$PWD/lb":/app -w /app node:22-bookworm-slim node tests/start_rescue.node.js`)."""
from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

HERE = pathlib.Path(__file__).resolve().parent


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node is not in the LB image: run tests/start_rescue.node.js under node:22-bookworm-slim")
def test_a_relaunch_after_a_hang_is_said_on_the_page():
    r = subprocess.run(["node", str(HERE / "start_rescue.node.js")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ok" in r.stdout
