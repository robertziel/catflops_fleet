"""/benchmarks shows EVERYTHING RECORDED when it loads: the range picker opens on `all` (its own option,
labelled "everything recorded"), and the matrix — the page's inventory — carries every configuration the log
holds, of every model, not only the model whose minutes the charts compare. What must not change in the
widening: a group is still one model's records, the shade is still ranked per model, and the hatch rank is
still taken among the model on show's own configurations (benchmarks.js `perf.days`, `groupsOf`, `ofModel`,
`perfScales`, `recomputeDomains`). tests/default_view.node.js checks it under node; this runs it when node is
on the path, and says where to run it when not (the LB image ships no node — `docker run --rm -v
"$PWD/lb":/app -w /app node:22-slim node tests/default_view.node.js`). The drawn result is the Chromium
audit's job (worklog/benchmarks-redesign/audit.py --models live)."""
from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

HERE = pathlib.Path(__file__).resolve().parent


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node is not in the LB image: run tests/default_view.node.js under node:22-slim")
def test_the_page_loads_everything_recorded():
    r = subprocess.run(["node", str(HERE / "default_view.node.js")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ok" in r.stdout
