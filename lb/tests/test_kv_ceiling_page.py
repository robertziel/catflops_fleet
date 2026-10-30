"""The KV pool chooser never offers a pin the profile cannot take: tests/kv_ceiling.node.js checks, in the browser's own
language, that the page greys every pin past `launch.cluster.kv_cache_memory_max` naming the maximum, says the bound under
the field, and refuses a stored pin past it before the Start (generation 187, 2026-10-10). This runs it when node is on
the path, and says where to run it when not (the LB image ships no node —
`docker run --rm -v "$PWD/lb":/lb:ro -w /lb node:22-slim node tests/kv_ceiling.node.js`)."""
from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

HERE = pathlib.Path(__file__).resolve().parent


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node is not in the LB image: run tests/kv_ceiling.node.js under node:22-slim")
def test_the_kv_chooser_greys_the_pins_past_the_profiles_bound_and_refuses_a_stored_one_by_name():
    r = subprocess.run(["node", str(HERE / "kv_ceiling.node.js")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ok" in r.stdout
