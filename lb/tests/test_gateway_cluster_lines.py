"""Three lines the LB publishes and the page never showed — all built in the browser, so they are checked
in the browser's language. tests/gateway_cluster_lines.node.js asserts that a swarm sub-line's start tag
parses (its `class="` was never closed, so with no title the browser swallowed the line's own text), that
the pair's card names the park from the field the wire carries (`park`, not `parked_by`) and never claims
a park is the cluster's, and that `throughput.ours.uncounted` reaches the page for a `metrics` backend —
the cluster — whose tiles never draw the proxy-only tile that was its one call site. This runs it when
node is on the path, and says where to run it when not (the LB image ships no node —
`docker run --rm -v "$PWD/lb":/app -w /app node:22-slim node tests/gateway_cluster_lines.node.js`)."""
from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

HERE = pathlib.Path(__file__).resolve().parent


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node is not in the LB image: run tests/gateway_cluster_lines.node.js under node:22-slim")
def test_the_swarm_line_the_park_and_the_uncounted_turns_all_reach_the_page():
    r = subprocess.run(["node", str(HERE / "gateway_cluster_lines.node.js")],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ok" in r.stdout
