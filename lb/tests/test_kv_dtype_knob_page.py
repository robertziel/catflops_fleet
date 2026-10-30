"""The KV type row on the page: tests/kv_dtype_knob.node.js checks the pair's and a local node's KV dtype choices — the
library's own, refused ones disabled with their words — and that a pick is listed and sent. This runs it
when node is on the path, and says where to run it when not (the LB image ships no node —
`docker run --rm -v "$PWD/lb":/app -w /app node:22-bookworm-slim node tests/kv_dtype_knob.node.js`)."""
from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

HERE = pathlib.Path(__file__).resolve().parent


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node is not in the LB image: run tests/kv_dtype_knob.node.js under node:22-bookworm-slim")
def test_the_kv_type_row_offers_the_published_dtypes_and_sends_the_pick():
    r = subprocess.run(["node", str(HERE / "kv_dtype_knob.node.js")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ok" in r.stdout
