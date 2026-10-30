"""A transcription server's card is drawn from its declared tile groups (tests/speech_card.node.js). This runs it when
node is on the path; the LB image ships none — `docker run --rm -v "$PWD/lb":/app -w /app node:22-slim node
tests/speech_card.node.js`."""
from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

HERE = pathlib.Path(__file__).resolve().parent


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node is not in the LB image: run tests/speech_card.node.js under node:22-slim")
def test_a_transcription_servers_card_draws_its_speech_tiles_and_no_llm_ones():
    r = subprocess.run(["node", str(HERE / "speech_card.node.js")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ok" in r.stdout
