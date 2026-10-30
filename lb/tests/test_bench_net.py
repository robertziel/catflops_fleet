"""The LB's bench-only network (lb/docker-compose.bench-net.yml) — for a deployment whose LB has no network
of its own that the sealed swarm runner's egress may join.

WHY. The sealed runner (lb/bench_runner) reaches the LB through its egress container, which must sit on a
network the LB is on. On a standalone server that is the LB's own node network (deploy/standalone: the LB,
the node and the egress share EDGE_NODE_NETWORK). With broker_parrot on, the LB's ONLY network is the
broker's — another project's production network (docker-compose.broker.yml) — and the operator chose
(2026-10-11) not to put the egress there. This override gives the LB one more network, for the bench
alone: named by the deployment (LB_BENCH_NETWORK, no default), EXTERNAL, and joined by nothing else here.

WHY EXTERNAL. Two compose projects share it — the LB's and the runner's — so neither owns it: a `down` in
either leaves it alone, and the deployment creates it once, on a subnet it names (a host whose docker
pools are exhausted cannot have one picked for it). An LB started with this file before the network
exists stops at once, naming it, before any container is touched.

WHY ONLY WITH AN OVERRIDE THAT NAMES THE LB'S NETWORKS. A service that names networks loses compose's
implicit `default`; the broker override names broker, so the merge is {broker, bench}. Used alone,
this file would take the LB off the network its searxng sidecar is on — so .env.example selects it only
after docker-compose.broker.yml.
"""
from __future__ import annotations

import pathlib
import re

import yaml
from tests.publishable_words import offenders

LB = pathlib.Path(__file__).resolve().parent.parent
OVERRIDE = LB / "docker-compose.bench-net.yml"


def _code(path: pathlib.Path) -> str:
    return "\n".join(ln for ln in path.read_text().splitlines() if not ln.lstrip().startswith("#"))


def test_the_override_adds_one_external_network_to_the_lb_and_nothing_else():
    over = yaml.safe_load(OVERRIDE.read_text())
    assert set(over) == {"services", "networks"}
    assert over["services"] == {"vllm-lb": {"networks": ["bench"]}}, "only the LB joins it, and only it"
    assert over["networks"] == {"bench": {
        "external": True,
        "name": "${LB_BENCH_NETWORK:?set in lb/.env when docker-compose.bench-net.yml is in COMPOSE_FILE}"}}


def test_the_network_is_named_by_the_deployment_never_by_this_repo():
    used = re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*)([^}]*)\}", _code(OVERRIDE))
    assert [v for v, _ in used] == ["LB_BENCH_NETWORK"]
    assert all(rest.startswith(":?") for _, rest in used), "no default: an unnamed network refuses the start"
    code = _code(OVERRIDE).lower()
    assert "bench-net_" not in code and offenders(code) == [], "the override names a deployment"


def test_the_env_example_selects_it_only_after_the_override_that_names_the_lbs_networks():
    text = (LB / ".env.example").read_text()
    assert re.search(r"^# LB_BENCH_NETWORK=$", text, re.M), "documented, commented: opt-in"
    assert re.search(r"^# COMPOSE_FILE=docker-compose\.yml:docker-compose\.broker\.yml:"
                     r"docker-compose\.bench-net\.yml$", text, re.M)
    assert not re.search(r"COMPOSE_FILE=docker-compose\.yml:docker-compose\.bench-net\.yml", text), \
        "alone it would take the LB off its implicit default network"


def test_the_sealed_runners_example_says_which_network_to_name_upstream():
    text = (LB / "bench_runner" / ".env.example").read_text()
    assert "docker-compose.bench-net.yml" in text and "LB_BENCH_NETWORK" in text