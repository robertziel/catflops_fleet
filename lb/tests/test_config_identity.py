"""What a configuration IS on /benchmarks and how it is said: the GPU memory fraction, the KV pin and the KV
tier in the host's RAM are part of its identity; absent ("not recorded"), null (the engine's own default) and a
value are three different things; a model that only booted has no grid; a minute's context comes only from the
pool it recorded — all computed in the browser (benchmarks.js `settingsKey`, `cfgText`, `inventory`,
`contextRows`). tests/config_identity.node.js checks it under node; this runs it when node is on the path, and
says where to run it when not (the LB image ships no node —
`docker run --rm -v "$PWD/lb":/app -w /app node:22-slim node tests/config_identity.node.js`)."""
from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

HERE = pathlib.Path(__file__).resolve().parent


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node is not in the LB image: run tests/config_identity.node.js under node:22-slim")
def test_a_configuration_is_every_setting_that_differs_said_without_a_question_mark():
    r = subprocess.run(["node", str(HERE / "config_identity.node.js")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ok" in r.stdout
