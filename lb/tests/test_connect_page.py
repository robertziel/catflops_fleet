"""The top bar's Connect dialog, on the page: tests/connect.node.js checks the dialog where gateway.js builds it — the
base from `location.origin`, the model from /v1/models, the door from /auth/session, the routes from /lb/status
`connect`, the snippets, Esc, the focus trap and copy. This runs it when node is on the path, and says where to run it
when not (the LB image ships no node — `docker run --rm -v "$PWD/lb":/lb:ro -w /lb node:22-slim node tests/connect.node.js`)."""
from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

HERE = pathlib.Path(__file__).resolve().parent


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node is not in the LB image: run tests/connect.node.js under node:22-slim")
def test_the_connect_dialog_builds_working_snippets_for_this_origin_and_says_where_none_work():
    r = subprocess.run(["node", str(HERE / "connect.node.js")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ok" in r.stdout
