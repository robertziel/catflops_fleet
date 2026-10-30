"""The LB's client network (lb/docker-compose.clients.yml) — for another compose project on this host that calls
the LB from a container: e.g. a browser agent's visual grounding (→ a GUI-grounding model on the gui server).

WHY. The LB publishes on 127.0.0.1 only, which a container cannot dial, and widening that bind is not on the table
(deploy/ routes the public hop). The node and bench networks each carry one purpose and their own peers (a model
node, an egress proxy); a client joining either could reach those too. This override gives the LB one more
network, for clients alone: named by the deployment (LB_CLIENT_NETWORK, no default), EXTERNAL (two projects share
it, neither owns it), selected only after an override that names the LB's networks — as bench-net, and for the
same reason (a service naming networks loses compose's implicit default).
"""
from __future__ import annotations

import pathlib
import re

import yaml
from tests.publishable_words import offenders

LB = pathlib.Path(__file__).resolve().parent.parent
OVERRIDE = LB / "docker-compose.clients.yml"


def _code(path: pathlib.Path) -> str:
    return "\n".join(ln for ln in path.read_text().splitlines() if not ln.lstrip().startswith("#"))


def test_the_override_adds_one_external_network_to_the_lb_and_nothing_else():
    over = yaml.safe_load(OVERRIDE.read_text())
    assert set(over) == {"services", "networks"}
    assert over["services"] == {"vllm-lb": {"networks": ["clients"]}}, "only the LB joins it here"
    assert over["networks"] == {"clients": {
        "external": True,
        "name": "${LB_CLIENT_NETWORK:?set in lb/.env when docker-compose.clients.yml is in COMPOSE_FILE}"}}


def test_the_network_is_named_by_the_deployment_never_by_this_repo():
    used = re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*)([^}]*)\}", _code(OVERRIDE))
    assert [v for v, _ in used] == ["LB_CLIENT_NETWORK"]
    assert all(rest.startswith(":?") for _, rest in used), "no default: an unnamed network refuses the start"
    code = _code(OVERRIDE).lower()
    assert "chromium" not in code and offenders(code) == [], "the override names a deployment"


def test_the_env_example_selects_it_only_after_the_override_that_names_the_lbs_networks():
    text = (LB / ".env.example").read_text()
    assert re.search(r"^# LB_CLIENT_NETWORK=$", text, re.M), "documented, commented: opt-in"
    assert re.search(r"^# COMPOSE_FILE=docker-compose\.yml:docker-compose\.broker\.yml:\S*"
                     r"docker-compose\.clients\.yml$", text, re.M)
    assert not re.search(r"COMPOSE_FILE=docker-compose\.yml:docker-compose\.clients\.yml", text), \
        "alone it would take the LB off its implicit default network"
