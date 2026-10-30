"""The session history ON THE PAGE: tests/sessions_page.node.js checks that /sessions draws the totals,
one strip per model on ONE token scale, the cumulative line and the table — and that a figure nobody
measured is a dash there, never a zero. This runs it when node is on the path, and says where to run it
when not (the LB image ships no node —
`docker run --rm -v "$PWD/lb":/app -w /app node:22-bookworm-slim node tests/sessions_page.node.js`)."""
from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

HERE = pathlib.Path(__file__).resolve().parent


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node is not in the LB image: run tests/sessions_page.node.js under node:22-bookworm-slim")
def test_the_session_history_page_draws_what_the_fold_gives_it():
    r = subprocess.run(["node", str(HERE / "sessions_page.node.js")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ok" in r.stdout
