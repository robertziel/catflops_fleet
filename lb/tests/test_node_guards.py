"""THE GUARDS A NODE ON A SHARED BOX RUNS UNDER — AUDIT-FINAL-5 P2-3, P2-4, P2-5 (2026-10-17).

  * THE MFU BYTE-SIZE PATCH is a fact of the card's type (`mfu_patch_loads`, {value, source}): on the AMD iGPU its
    sitecustomize recursed under ROCm's platform probe until the container's memcg OOM-killed it (measured on this
    fleet's box, 2026-10-17). A launch whose checkpoint NEEDS the patch (its quant method in fleet.MFU_PATCHED_QUANTS) on
    a type that cannot load it — or does not say — is REFUSED by the fit check (a choice: `unlaunchable`; an intent:
    ValueError), never rendered with the patch on.
  * THE CONTAINER'S LIMITS are the box's (`node_limits`: memory_gib, cpus, pids, nice — no defaults): a box that hosts a
    node its host's applier runs (control local) states them, or the fleet refuses to load; node-env renders them for
    deploy/standalone's compose (mem_limit = memswap_limit, cpus, pids_limit, nice), the measured run's guards.
  * EVERY CATALOG ROW STATES ITS QUANT METHOD ({value, source}; null = unquantised, said so): an absent one is refused,
    never read as "no patch" — the FFN would silently fall out of the node's FLOP/s and GB/s.
"""
from __future__ import annotations

import json

import pytest

from tests.fleet_files import ONE_HOST
from vllm_lb.catalog import document, ModelCatalog
from vllm_lb.fleet import node_env, parse_fleet, validate_launch
from vllm_lb.hardware.profiles import types

DIGEST = "rocm/vllm@sha256:" + "2" * 64
LIMITS = "    node_limits: {memory_gib: 24, cpus: 16, pids: 4096, nice: 10}\n"
APU = f"""
panel:
  style: vibe
boxes:
  apu:
    hostname: apu-host
    label: apu-gpu
    platform: linux/amd64
    gpu_ram_max_gib: 64
{LIMITS}    cards:
      - {{index: 0, type: radeon8060s, serves: true}}
clusters:
  gui:
    control: local
    run_at_boot: false
    model: mai-ui-2b
    url: http://gui-node:8000
    launch:
      image: {DIGEST}
      max_model_len: 32768
      max_num_seqs: 4
      gpu_memory_utilization: 0.40
      tool_call_parser: hermes
      choices: [mai-ui-2b, qwen3.8-27b-nvidia-nvfp4]
    ranks:
      - {{box: apu}}
backends: {{}}
"""


# ── P2-3: the MFU patch, as the card type's fact ───────────────────────────────────────────────────────────────────

def test_every_type_states_whether_the_mfu_patch_loads_with_its_source():
    for key, t in types().items():
        assert t.mfu_patch_loads.source.strip(), key
    assert types()["radeon8060s"].mfu_patch_loads.value is False, "measured: it recursed until OOM under ROCm"
    assert types()["rtx5090"].mfu_patch_loads.value is True, "site-b's 27B node runs with it"


def test_a_choice_that_needs_the_patch_on_a_type_that_cannot_load_it_is_unlaunchable():
    gui = parse_fleet(APU, "fleet.yml").clusters["gui"]
    assert "qwen3.8-27b-nvidia-nvfp4" in gui.unlaunchable and "mfu_patch_loads" in gui.unlaunchable, gui.unlaunchable


def test_an_intent_for_it_is_refused_and_never_rendered_with_the_patch_on():
    one = APU.replace("choices: [mai-ui-2b, qwen3.8-27b-nvidia-nvfp4]", "choices: [mai-ui-2b]")
    fleet = parse_fleet(APU, "fleet.yml")
    with pytest.raises(ValueError, match="mfu_patch_loads"):
        validate_launch(fleet.clusters["gui"], {"model": "qwen3.8-27b-nvidia-nvfp4"})
    with pytest.raises(ValueError, match="mfu_patch_loads"):
        node_env(fleet, "gui", {"model": "qwen3.8-27b-nvidia-nvfp4"})
    lines = node_env(parse_fleet(one, "fleet.yml"), "gui", {})
    assert "EDGE_NODE_MFU_PATCH=off" in lines, "a BF16 row needs none, on any type"


# ── P2-4: the container's limits, the box's data ───────────────────────────────────────────────────────────────────

def test_a_box_hosting_a_local_node_states_its_limits_or_the_fleet_refuses():
    with pytest.raises(ValueError, match="node_limits"):
        parse_fleet(APU.replace(LIMITS, ""), "fleet.yml")
    # a box that hosts none needs none (the pair's boxes, an observed node's)
    parse_fleet(ONE_HOST.replace(LIMITS, ""), "fleet.yml")


@pytest.mark.parametrize("bad", ["{memory_gib: 24, cpus: 16, pids: 4096}", "{memory_gib: 0, cpus: 16, pids: 4096, nice: 10}",
                                 "{memory_gib: 24, cpus: 16, pids: 4096, nice: 20}", "{memory_gib: 24, cpus: 0, pids: 4096, nice: 10}",
                                 "{memory_gib: 24, cpus: 16, pids: 4096, nice: 10, swap: 1}", "24"])
def test_limits_are_whole_and_in_range_never_defaulted(bad):
    with pytest.raises(ValueError, match="node_limits"):
        parse_fleet(APU.replace(LIMITS, f"    node_limits: {bad}\n"), "fleet.yml")


def test_node_env_renders_the_boxes_limits():
    lines = node_env(parse_fleet(APU.replace(", qwen3.8-27b-nvidia-nvfp4]", "]"), "fleet.yml"), "gui", {})
    for line in ("EDGE_NODE_MEMORY=24g", "EDGE_NODE_CPUS=16", "EDGE_NODE_PIDS_LIMIT=4096", "EDGE_NODE_NICE=10"):
        assert line in lines, lines


# ── P2-5: every row states its quant method ────────────────────────────────────────────────────────────────────────

def test_every_shipped_row_states_its_quant_method_with_a_source():
    doc = document()
    for row in doc["models"]:
        fact = row.get("quant_method")
        assert isinstance(fact, dict) and fact.get("source", "").strip(), row["id"]


def test_every_shipped_row_states_whether_a_chat_may_use_it():
    """`chat` (operator 2026-10-18): the chat lists only servers whose model it can talk to — a GUI-grounding model
    answers click(x,y), not prose. REQUIRED, so a new row cannot slip into (or out of) the chat by omission."""
    doc = document()
    for row in doc["models"]:
        fact = row.get("chat")
        assert isinstance(fact, dict) and isinstance(fact.get("value"), bool) and fact.get("source", "").strip(), row["id"]
    cat = ModelCatalog.from_dict(doc)
    assert cat.get("mai-ui-2b").chat is False and cat.get("qwen3.8-27b-sglang-dflash2").chat is True
    del doc["models"][0]["chat"]
    with pytest.raises(ValueError, match="chat"):
        ModelCatalog.from_dict(doc)


def test_a_row_without_one_is_refused_never_read_as_unquantised():
    doc = document()
    del doc["models"][0]["quant_method"]
    with pytest.raises(ValueError, match="quant_method"):
        ModelCatalog.from_dict(doc)
