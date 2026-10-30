"""The REQUESTS card's kill ×, on the page: tests/request_kill.node.js checks the button, the 1 s hold, the one POST and
the row's "killing…" / "killed" / refusal where gateway.js builds them. This runs it when node is on the path, and says
where to run it when not (the LB image ships no node —
`docker run --rm -v "$PWD/lb":/lb:ro -w /lb node:22-slim node tests/request_kill.node.js`)."""
from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

HERE = pathlib.Path(__file__).resolve().parent


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node is not in the LB image: run tests/request_kill.node.js under node:22-slim")
def test_the_kill_button_posts_once_after_a_1_s_hold_and_the_row_says_what_happened():
    r = subprocess.run(["node", str(HERE / "request_kill.node.js")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ok" in r.stdout
