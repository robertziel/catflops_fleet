"""Library strings (an engine key used as a chip's class, a card type's peak precision) reach the panel escaped, in
text and in attributes: tests/escape_sinks.node.js checks gateway.js `tag` and `ceilingMeter` in the browser's
language. This runs it when node is on the path, and says where to run it when not (the LB image ships no node —
`docker run --rm -v "$PWD/lb":/lb:ro -w /lb node:22-slim node tests/escape_sinks.node.js`)."""
from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

HERE = pathlib.Path(__file__).resolve().parent


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node is not in the LB image: run tests/escape_sinks.node.js under node:22-slim")
def test_library_strings_cannot_open_an_element_or_leave_their_attribute():
    r = subprocess.run(["node", str(HERE / "escape_sinks.node.js")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ok" in r.stdout
