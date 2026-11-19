"""The panel's face of roles: tests/access.node.js checks vllm_lb/static/access.js — the admin's Access
dialog over the gate's /auth/access and a member's line in the bar. This runs it when node is on the path, and says
where to run it when not (the LB image ships no node —
`docker run --rm -v "$PWD/lb":/lb:ro -w /lb node:22-slim node tests/access.node.js`)."""
from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

HERE = pathlib.Path(__file__).resolve().parent


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node is not in the LB image: run tests/access.node.js under node:22-slim")
def test_the_access_dialog_is_the_admins_and_a_member_is_told_their_models():
    r = subprocess.run(["node", str(HERE / "access.node.js")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ok" in r.stdout
