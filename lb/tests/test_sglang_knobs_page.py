"""The knobs an engine does not take, on the page: tests/sglang_knobs.node.js checks that the panel locks them at what
the engine runs, says why, and lists a stored value as the reset it needs, in the browser's own language. This runs it
when node is on the path, and says where to run it when not (the LB image ships no node —
`docker run --rm -v "$PWD/lb":/app -w /app node:22-bookworm-slim node tests/sglang_knobs.node.js`)."""
from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

HERE = pathlib.Path(__file__).resolve().parent


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node is not in the LB image: run tests/sglang_knobs.node.js under node:22-bookworm-slim")
def test_a_knob_the_engine_does_not_take_is_locked_and_said_on_the_page():
    r = subprocess.run(["node", str(HERE / "sglang_knobs.node.js")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ok" in r.stdout
