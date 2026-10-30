"""The swarm control names what it would load from the FLEET, never a topology or a box written in the code.

`/lb/status.swarm.model_detail` said "the two-box cluster is serving qwen3.8-27b-uncensored" on site-b's gateway —
one RTX 5090 node (2026-10-11 20:47Z): `proxy._model_loaded` named every cluster backend the box-c pair. The words
now come from the fleet file the LB runs (`topology.named`: the cluster's name, its TP and its boxes), and a guard
reads every string the swarm's code paths execute for a deployment's or a box's name (AGENTS.md, open-source ready).
"""
from __future__ import annotations

import ast
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.fleet_files import ONE_HOST, PAIR, write
from vllm_lb import topology
from vllm_lb.config import Config
from vllm_lb.proxy import Proxy
from vllm_lb.state import CLUSTER, VLLM
from tests.publishable_words import CANARY, offenders

LB = Path(__file__).resolve().parent.parent


def loaded(cfg, *backends):
    """`Proxy._model_loaded` on a gateway whose hub holds exactly `backends`."""
    snapshot = SimpleNamespace(backends=list(backends), extra={}, cluster_views=dict)
    return Proxy._model_loaded(SimpleNamespace(cfg=cfg, hub=SimpleNamespace(snapshot=lambda: snapshot)))


def backend(kind, served, label=None, url="http://192.0.2.10:8000"):
    return SimpleNamespace(live=True, kind=kind, served_models=list(served), host_label=label, url=url)


def test_a_one_host_node_is_named_by_its_fleet_cluster_and_its_box(tmp_path):
    cfg = Config.from_env({"LB_ALLOWED_HOSTS": "loopback", "LB_FLEET": write(tmp_path, ONE_HOST)})
    ok, words = loaded(cfg, backend(CLUSTER, ["model-x"]))
    assert ok is True
    assert words == "cluster node (TP=1 on solo) is serving model-x"


def test_the_pair_is_named_by_its_fleet_cluster_and_both_boxes(tmp_path):
    cfg = Config.from_env({"LB_ALLOWED_HOSTS": "loopback", "LB_FLEET": write(tmp_path, PAIR)})
    ok, words = loaded(cfg, backend(CLUSTER, ["model-y"]))
    assert words == "cluster pair (TP=2 on node-1, node-2) is serving model-y"
    assert topology.named(cfg) == "cluster pair (TP=2 on node-1, node-2)"


def test_a_classic_server_is_named_by_its_label_as_before(tmp_path):
    cfg = Config.from_env({"LB_ALLOWED_HOSTS": "loopback", "LB_FLEET": write(tmp_path, PAIR)})
    ok, words = loaded(cfg, backend(VLLM, ["model-z"], label="node-1-gpu"))
    assert (ok, words) == (True, "node-1-gpu is serving model-z")


def test_a_cluster_no_fleet_names_is_named_by_its_configured_boxes_and_nothing_else():
    """A Config built by hand (no fleet file) still says only what it was configured with."""
    from vllm_lb.cluster import parse_hosts
    cfg = Config(cluster_name="pair", cluster_hosts=parse_hosts("a-gpu=box-a=op@192.0.2.10=198.51.100.1,"
                                           "b-gpu=box-b=op@192.0.2.11=198.51.100.2"),
                 cluster_url="http://192.0.2.10:8100")
    assert topology.named(cfg) == "the cluster (TP=2 on box-a, box-b)"


# ── the guard: no deployment or box name in a string the swarm's code executes ──────────────────────

#: A deployment's or a box's name, or a topology written as one: tests/publishable_words.py's digests, plus the
#: pair written as "two-<box>" — matched as words, case-insensitively.
def _names_a_deployment(text: str) -> bool:
    return bool(offenders(text)) or bool(re.search(r"(?i)(?<![a-z0-9])two-box(?![a-z0-9])", text))
#: The swarm's code paths: the LB's half, the runner, the session, the sealed runner — and, in proxy.py, the
#: two functions the control runs through.
SWARM_FILES = (LB / "vllm_lb" / "benchswarm.py", LB / "bench_queue_runner.py", LB / "bench_session.py",
               *sorted((LB / "bench_runner").glob("*.py")))
SWARM_PROXY_FUNCTIONS = ("_model_loaded", "swarm_start")


def executed_strings(tree: ast.AST, only: tuple[str, ...] = ()) -> list[tuple[int, str]]:
    """(line, text) of every string constant the code EXECUTES — docstrings excluded — optionally only inside the
    named functions."""
    docs = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and body \
                and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                and isinstance(body[0].value.value, str):
            docs.add(id(body[0].value))
    roots = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in only] \
        if only else [tree]
    return [(n.lineno, n.value) for root in roots for n in ast.walk(root)
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs]


def deployment_names(tree, only=()) -> list[tuple[int, str]]:
    return [(line, text) for line, text in executed_strings(tree, only) if _names_a_deployment(text)]


def test_no_string_the_swarm_executes_names_a_deployment_or_a_box():
    found = []
    for path in SWARM_FILES:
        found += [(path.name, *hit) for hit in deployment_names(ast.parse(path.read_text()))]
    proxy = ast.parse((LB / "vllm_lb" / "proxy.py").read_text())
    found += [("proxy.py", *hit) for hit in deployment_names(proxy, SWARM_PROXY_FUNCTIONS)]
    assert found == []
    assert {n.name for n in ast.walk(proxy) if isinstance(n, ast.FunctionDef)} >= set(SWARM_PROXY_FUNCTIONS), \
        "the guard reads functions that exist"


@pytest.mark.parametrize("planted, caught", [
    ('def f():\n    return "the two-box cluster"\n', True),
    (f'def f():\n    """Runs on {CANARY}."""\n    return 1\n', False),     # prose explains; it is not executed
    (f'# {CANARY}\nX = "box-a"\n', False),                            # a comment is not executed
    (f'def f(b):\n    return f"runner on {{b}}" if b else "runner on {CANARY}"\n', True),
    (f'def f():\n    return "{CANARY} is down"\n', True),
    ('def f():\n    return "sparkline"\n', False),
])
def test_the_guard_catches_a_plant_and_ignores_prose(planted, caught):
    assert bool(deployment_names(ast.parse(planted))) is caught

def test_a_transcription_node_is_no_model_for_a_swarm_and_the_chat_cluster_is_named_among_several(tmp_path):
    """2026-10-24: with the pair loading and only box-b's Parakeet node up, the swarm said 'the cluster (TP=0) is
    serving /models/parakeet-…' — a deep-research session cannot talk to a speech-to-text model."""
    from tests.fleet_files import LOCAL_ON_PAIR_BOX
    cfg = Config.from_env({"LB_ALLOWED_HOSTS": "loopback", "LB_FLEET": write(tmp_path, LOCAL_ON_PAIR_BOX)})
    speech = SimpleNamespace(**vars(backend(CLUSTER, ["/models/parakeet-tdt-0.6b-v3"])), cluster="speech")
    pair = SimpleNamespace(**vars(backend(CLUSTER, ["model-y"])), cluster="pair")
    ok, words = loaded(cfg, speech)
    assert ok is False and "parakeet" not in words
    assert loaded(cfg, speech, pair) == (True, "cluster pair (TP=2 on node-1, node-2) is serving model-y")
