"""The fidelity section is drawn in the browser (static/fidelity.js), so it is checked in the
browser's language: tests/fidelity.node.js. This runs it when node is on the path, and says where to run it when not
(the LB image ships no node — `docker run --rm -v "$PWD/lb":/app -w /app node:22-slim node tests/fidelity.node.js`)."""
from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

HERE = pathlib.Path(__file__).resolve().parent
STATIC = HERE.parent / "vllm_lb" / "static"


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node is not in the LB image: run tests/fidelity.node.js under node:22-slim")
def test_the_section_draws_figures_and_escapes_the_servers_words():
    r = subprocess.run(["node", str(HERE / "fidelity.node.js")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ok" in r.stdout


def test_the_benchmarks_page_loads_fidelity_after_benchmarks_and_has_its_section():
    html = (STATIC / "benchmarks.html").read_text()
    assert html.index('src="/benchmarks.js') < html.index('src="/fidelity.js')
    assert 'href="/fidelity.css' in html and 'id="fid-root"' in html


def test_the_build_stamps_the_fidelity_files():
    docker = (HERE.parent / "Dockerfile").read_text()
    loop = docker.split("for f in", 1)[1].split(";", 1)[0]
    for name in ("fidelity.js", "fidelity.css"):
        assert name in loop, f"{name} is in the stamp loop"
        assert f"grep -q '{name}?v=' benchmarks.html" in docker, f"the build fails if {name}'s stamp missed"


def test_the_page_posts_only_from_the_buttons_click():
    js = (STATIC / "fidelity.js").read_text()
    assert js.count("method: 'POST'") == 1
    post_fn = js.split("async function fidStart", 1)[1].split("\nfunction ", 1)[0]
    assert "method: 'POST'" in post_fn
    assert js.count("fidStart();") == 1, "fidStart is called from the click handler only"
