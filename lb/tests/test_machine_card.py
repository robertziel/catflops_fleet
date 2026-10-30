"""A machine card's rings, headlines and graphs are built in the browser (gateway.js `machineBody`), so
the rule that they agree — one value per figure, one colour — is checked in the browser's language.
tests/machine_card.node.js asserts that every figure's graph ends at its ring's value, in its ring's
colour, under a headline in that colour: box-c's MEM graph was drawn from the GPU's memory (95 % of an
iGPU's 512 MB) beneath a ring showing the box's RAM (20 %). This runs it when node is on the path, and
says where to run it when not (the LB image ships no node —
`docker run --rm -v "$PWD/lb":/app -w /app node:22-slim node tests/machine_card.node.js`)."""
from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

HERE = pathlib.Path(__file__).resolve().parent


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node is not in the LB image: run tests/machine_card.node.js under node:22-slim")
def test_every_machine_figure_reads_one_value_in_one_colour():
    r = subprocess.run(["node", str(HERE / "machine_card.node.js")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ok" in r.stdout
