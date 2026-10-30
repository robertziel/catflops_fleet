"""The cluster card's token rows are built in the browser (gateway.js `renderSaturation`), so what
they claim is checked in the browser's language. tests/cluster_tokens.node.js asserts the operator's
ask — generated and prefill tok/s as two rows BETWEEN the SM-fill pair and the bandwidth/compute pair
— and the two honesty rules that decide what may go in them: the prefill rate shown is the one the
engine COMPUTED (the billed rate is ~10x it on this fleet, the prefix cache's doing) and an unmeasured
rate is "—" with the LB's reason, never a 0.

This runs it when node is on the path, and says where to run it when not. The LB image ships no node,
and the captured document lives outside the `lb/` mount, so the repo ROOT is what goes in:
`docker run --rm -v "$PWD":/repo:ro -w /repo/lb node:22-slim node tests/cluster_tokens.node.js`."""
from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

HERE = pathlib.Path(__file__).resolve().parent
#: A REAL captured /lb/status, so the quantities asserted are the ones the page actually meets (addresses,
#: hostnames and model roots replaced by documentation values; the figures are as captured).
FIXTURE = HERE / "fixtures" / "lb-status-live-pre-knobs.json"


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node is not in the LB image: run tests/cluster_tokens.node.js under node:22-slim")
@pytest.mark.skipif(not FIXTURE.exists(),
                    reason=f"the captured /lb/status is outside the mount: {FIXTURE}")
def test_the_cluster_cards_token_rows_sit_between_the_pairs_and_show_the_computed_prefill():
    r = subprocess.run(["node", str(HERE / "cluster_tokens.node.js")],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ok" in r.stdout
