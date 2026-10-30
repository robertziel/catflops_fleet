"""The gateway page with broker_parrot OFF is drawn in the browser (gateway.js), so it is checked there:
tests/broker_off.node.js asserts a held idle server reads IDLE with no lease and no db_ok raises no broker
alarm — and that with the broker ON every word is what it was. This runs it when node is on the path, and says where to run it when not (the LB image
ships no node — `docker run --rm -v "$PWD/lb":/lb:ro -w /lb node:22-slim node tests/broker_off.node.js`)."""
from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

HERE = pathlib.Path(__file__).resolve().parent


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node is not in the LB image: run tests/broker_off.node.js under node:22-slim")
def test_the_page_reads_a_gateway_without_the_broker_as_one():
    r = subprocess.run(["node", str(HERE / "broker_off.node.js")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ok" in r.stdout
