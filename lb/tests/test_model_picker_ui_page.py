"""The model picker as a combobox, on the page: tests/model_picker_ui.node.js checks the rows, groups, badges and
chips the page draws from one status, the ✓ and running marks, a row missing on a box disabled with the box named,
the keyboard (arrows over enabled rows, Enter commits through the hidden <select>'s change event, Escape closes
without a change), a disabled row that cannot be chosen, and the Start body still carrying model_choice +
uncensored after a keyboard pick. This runs it when node is on the path, and says where to run it when not (the
LB image ships no node — `docker run --rm -v "$PWD/lb":/lb:ro -w /lb node:22-slim node tests/model_picker_ui.node.js`)."""
from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

HERE = pathlib.Path(__file__).resolve().parent


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node is not in the LB image: run tests/model_picker_ui.node.js under node:22-slim")
def test_the_model_picker_is_a_combobox_over_the_hidden_select():
    r = subprocess.run(["node", str(HERE / "model_picker_ui.node.js")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ok" in r.stdout
