"""WHICH MODEL the cluster card names — the operator read three cards that all said "Qwen3.8 27B" and
could not tell which was loading. The name the badge prints is the label's first segment, and
vllm_lb/library gives three cluster models the same one; the segment that told them apart
(`mb-build`) is hidden at a 600 px container while the cluster panel is 440 px at a 1440 viewport. So
the badge prints the served id for a name the catalog shares. tests/model_badge.node.js asserts that in
the browser's language against the SHIPPED catalog — one distinct name per cluster model serving and
pending, the full identity still in the `title`, the two CSS rules that keep it at every width — and
that no launch fact names a drafter method: catalog.json states `method: dflash` for both DFlash2 builds
and `mtp` for the other two, while `LaunchProfile.view()` publishes neither, so "MTP 7" over a dflash
profile was the page's own invention. This runs it when node is on the path, and says where to run it
when not (the LB image ships no node —
`docker run --rm -v "$PWD/lb":/app -w /app node:22-slim node tests/model_badge.node.js`)."""
from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

HERE = pathlib.Path(__file__).resolve().parent


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node is not in the LB image: run tests/model_badge.node.js under node:22-slim")
def test_the_badge_names_one_model_and_no_drafter_method_it_was_never_told():
    r = subprocess.run(["node", str(HERE / "model_badge.node.js")],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ok" in r.stdout
