"""The LB's network to a one-host node it requests on its OWN box (lb/docker-compose.node.yml) — e.g. the gui node on
the LB host's iGPU, beside a pair the LB drives over SSH.

WHY. The node publishes on loopback only (deploy/standalone up.sh refuses anything else: it serves without sign-in),
and a bridge container cannot dial its host's loopback — so the LB reaches the node by its container name on a network
both join: the node's EDGE_NODE_NETWORK (its server file; up.sh creates it when missing), named here LB_NODE_NETWORK
(lb/.env, no default). The fleet file's `clusters.<name>.url` is then `http://<EDGE_NODE_NAME>:8000`.
A standalone server (deploy/standalone/lb.override.yml) does the same from its one server file; this is for an LB
brought up from lb/ with its own lb/.env.

WHY EXTERNAL, AND ONLY AFTER AN OVERRIDE THAT NAMES THE LB'S NETWORKS: as docker-compose.bench-net.yml
(tests/test_bench_net.py) — two compose projects share it, and a service that names networks loses compose's implicit
`default`, so used alone it would take the LB off its searxng sidecar's network.
"""
from __future__ import annotations

import pathlib
import re

import yaml
from tests.publishable_words import offenders

LB = pathlib.Path(__file__).resolve().parent.parent
OVERRIDE = LB / "docker-compose.node.yml"


def _code(path: pathlib.Path) -> str:
    return "\n".join(ln for ln in path.read_text().splitlines() if not ln.lstrip().startswith("#"))


def test_the_override_adds_one_external_network_to_the_lb_and_nothing_else():
    over = yaml.safe_load(OVERRIDE.read_text())
    assert set(over) == {"services", "networks"}
    assert over["services"] == {"vllm-lb": {"networks": ["node"]}}, "only the LB joins it, and only it"
    assert over["networks"] == {"node": {
        "external": True,
        "name": "${LB_NODE_NETWORK:?set in lb/.env when docker-compose.node.yml is in COMPOSE_FILE}"}}


def test_the_network_is_named_by_the_deployment_never_by_this_repo():
    used = re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*)([^}]*)\}", _code(OVERRIDE))
    assert [v for v, _ in used] == ["LB_NODE_NETWORK"]
    assert all(rest.startswith(":?") for _, rest in used), "no default: an unnamed network refuses the start"
    code = _code(OVERRIDE).lower()
    assert "gui" not in code and offenders(code) == [], "the override names a deployment"


def test_the_env_example_selects_it_after_the_override_that_names_the_lbs_networks():
    text = (LB / ".env.example").read_text()
    assert re.search(r"^# LB_NODE_NETWORK=$", text, re.M), "documented, commented: opt-in"
    assert re.search(r"^# COMPOSE_FILE=docker-compose\.yml:docker-compose\.broker\.yml:(\S+:)?"
                     r"docker-compose\.node\.yml$", text, re.M)
    assert not re.search(r"COMPOSE_FILE=docker-compose\.yml:docker-compose\.node\.yml", text), \
        "alone it would take the LB off its implicit default network"
    assert "EDGE_NODE_NETWORK" in text and "EDGE_NODE_NAME" in text, "says which value it must equal, and the url"


def test_the_fleet_example_dials_the_node_by_its_container_name():
    text = (LB / "fleet.amd-node.example.yml").read_text()
    assert "docker-compose.node.yml" in text and "EDGE_NODE_NAME" in text
