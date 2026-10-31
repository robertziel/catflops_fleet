"""The shipped defaults name NO machine.

vllm-lb is published as source (PolyForm Noncommercial), and a stranger's install must not dial the operator's
GB10 boxes, their LAN addresses, their hardware hostnames or an ollama on `host.docker.internal`
because the code said so. Every such value now comes from the environment (lb/.env in the
operator's deployment, documented in lb/.env.example) and the shipped default is EMPTY — which
means "off", never "a stand-in": no classic vLLM box, no fallback URL, no hardware mapping.

Two things are pinned here: the code defaults, and the compose file that passes the operator's
values through — the compose file must not carry a literal default for them either.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from vllm_lb.config import Config
from tests.publishable_words import offenders

COMPOSE = Path(__file__).resolve().parent.parent / "docker-compose.yml"


def test_the_shipped_defaults_carry_no_operator_hardware():
    c = Config()
    assert c.vllm_boxes == ()
    assert c.hw_hosts == {}


def test_the_operator_values_arrive_through_the_fleet_file(tmp_path):
    """The operator's machines are the deployment's fleet file (vllm_lb/fleet.py, LB_FLEET) — never a code default,
    never a compose default, and no longer the environment (config.py RETIRED_KEYS)."""
    from tests.fleet_files import write
    c = Config.from_env({"LB_ALLOWED_HOSTS": "loopback", "LB_FLEET": write(tmp_path, """
panel:
  style: vibe
boxes:
  box-a: {hostname: host-a, label: box-a-gpu, platform: linux/arm64, cards: [{index: 0, type: gb10, serves: true}]}
  box-b: {hostname: host-b, label: box-b-gpu, platform: linux/arm64, cards: [{index: 0, type: gb10, serves: true}]}
clusters: {}
backends:
  box-a: {url: "http://192.0.2.10:8000"}
  box-b: {url: "http://192.0.2.11:8000"}
""")})
    assert [(b.host_label, b.box, b.url) for b in c.vllm_boxes] == [
        ("box-a-gpu", "box-a", "http://192.0.2.10:8000"),
        ("box-b-gpu", "box-b", "http://192.0.2.11:8000"),
    ]
    assert c.hw_hosts == {"host-a": "box-a", "host-b": "box-b"}


def test_no_environment_means_no_fallback_host_is_dialled():
    """Outside compose the LB used to silently dial ollama on host.docker.internal. The fallback tier is retired
    (2026-10-17): no setting for one exists, and the request path answers 503 rather than a stand-in."""
    c = Config.from_env({"LB_ALLOWED_HOSTS": "loopback"})
    assert c.vllm_boxes == ()
    assert not [f for f in Config.__dataclass_fields__ if f.startswith("fallback_")]


def _compose_env() -> dict[str, str]:
    doc = yaml.safe_load(COMPOSE.read_text())
    out: dict[str, str] = {}
    for item in doc["services"]["vllm-lb"]["environment"]:
        key, _, value = item.partition("=")
        out[key] = value
    return out


def test_compose_passes_the_operator_values_through_with_no_literal_default():
    env = _compose_env()
    assert not [k for k in env if k.startswith("LB_FALLBACK_")], "the fallback tier is retired: compose passes none"
    # The machines are the fleet file's, mounted read-only from lb/fleet.yml — which must exist: a missing file
    # fails `docker compose up` rather than becoming an empty directory.
    assert env["LB_FLEET"] == "/etc/vllm-lb/fleet.yml"
    for key in ("LB_VLLM_BOXES", "LB_HW_HOSTS", "LB_CLUSTER_HOSTS", "LB_CLUSTER_URL"):
        assert key not in env, f"{key} is retired into the fleet file"
    mount = next(v for v in yaml.safe_load(COMPOSE.read_text())["services"]["vllm-lb"]["volumes"]
                 if isinstance(v, dict) and v.get("target") == "/etc/vllm-lb/fleet.yml")
    assert (mount["type"], mount["source"], mount["read_only"], mount["bind"]["create_host_path"]) == \
        ("bind", "./fleet.yml", True, False)


def test_compose_carries_no_literal_lan_address_or_hostname():
    text = COMPOSE.read_text()
    for literal in ("192.168.", "host.docker.internal:11434"):
        assert literal not in text, f"{literal!r} is an operator literal in the shipped compose"
    assert offenders(text) == [], "a deployment's host or address is in the shipped compose (tests/publishable_words.py)"


# ── hardware peaks are claims about a GPU, and a shipped default claims none ─────────────────────

def test_the_shipped_code_claims_no_peak_of_its_own():
    """A GPU's peaks are its card's hardware type's, named per box by a fleet file — the code carries no LB-wide
    peak at all, and an old LB_GPU_PEAK_* line refuses the start (config.py RETIRED_KEYS)."""
    c = Config()
    assert not any(hasattr(c, k) for k in ("gpu_peak_gbps", "gpu_peak_tflops", "gpu_peak_tflops_precision"))


def test_compose_passes_no_lb_wide_peak():
    env = _compose_env()
    assert not any(k.startswith("LB_GPU_PEAK_") for k in env), "a GPU's peaks are its card's, per box"
