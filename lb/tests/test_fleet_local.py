"""A ONE-HOST NODE THE LB REQUESTS AND ITS HOST APPLIES (`control: local`), AND EVERY BOX'S PLATFORM — never inferred.

Operator decision, 2026-10-11: a standalone server's LB controls its one node (load/unload, parameters) with NO SSH.
The LB writes an INTENT into its state dir; a systemd applier on the host validates it and runs `up.sh node`. In the
fleet file that is a third control:

  ssh    the tp-rank pair — unchanged: 2 ranks x 1 card, each box's ssh and interconnect, no model or launch here
  local  ONE rank on ONE card; its `model` and its `launch` (the BOOT DEFAULT, which an intent moves within the
         model's catalog limits) — never an `ssh` on its box (nothing reaches it), never an interconnect (no peer).
         Its launch lists the models the panel may SELECT (`launch.choices`, the boot default among them): each a
         catalog row whose weights FIT the serving card at the launch's memory fraction. A choice's served name and
         checkpoint folder are its catalog row's (never typed): node-env renders EDGE_NODE_MODEL from the chosen one.
  none   a node the LB only observes — unchanged

The applier validates an intent with the SAME code as the launch block: `fleet.validate_launch(cluster, overrides)`,
also behind `python -m vllm_lb.fleet node-env <fleet> <cluster> <intent.json>` for an applier that runs this image.
An intent moves the model (one of the choices) and the figures the catalog bounds (`MOVABLE`); the image and parsers
are the fleet's. Every launch, the boot default and each intent, is FIT-checked by the same code.

And the image a node runs must be built for its box's CPU — the 5090 node needs an x86 image, the catalog's are
arm64 — so every box declares its `platform` (linux/amd64 | linux/arm64). A card type whose chip includes the CPU
states it too (the GB10 superchip is Arm), and the two must agree. `node-env` renders the platform for compose to pin.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys

import pytest

from tests.fleet_files import ONE_HOST, PAIR, write
from vllm_lb.config import topology_of
from vllm_lb.fleet import CONTROLS, MOVABLE, launch_fit, node_env, parse_fleet, validate_launch

DIGEST = "vllm/vllm-openai@sha256:" + "1" * 64
LOCAL = f"""
panel:
  style: vibe
boxes:
  solo:
    hostname: solo-host
    label: solo-gpu
    platform: linux/amd64
    node_limits: {{memory_gib: 24, cpus: 16, pids: 4096, nice: 10}}
    cards:
      - {{index: 0, type: rtx3090, serves: false}}
      - {{index: 1, type: rtx5090, serves: true}}
clusters:
  node:
    control: local
    model: qwen3.8-27b-nvidia-nvfp4@uncensored
    url: http://vllm-node:8000
    launch:
      image: {DIGEST}
      max_model_len: 131072
      max_num_seqs: 8
      gpu_memory_utilization: 0.9
      reasoning_parser: qwen3
      tool_call_parser: qwen3_coder
      choices: [qwen3.8-27b-nvidia-nvfp4@uncensored, qwen3.8-27b-nvidia-nvfp4]
    ranks:
      - {{box: solo}}
backends: {{}}
"""
#: The catalog's own words for the 27B row (the model library, vllm_lb/library): what a choice of it is served as and loaded from.
SERVED, TWIN_DIR, ORIGINAL_DIR = "Qwen/Qwen3.8-27B", "Qwen3.8-27B-NVFP4_uncensored", "Qwen3.8-27B-NVFP4"


def _fleet(text: str):
    return parse_fleet(text, "fleet.yml")


def _node():
    return _fleet(LOCAL).clusters["node"]


# ── control: local ──────────────────────────────────────────────────────────────────────────────────

def test_a_node_the_lb_requests_is_one_rank_on_one_card_with_no_ssh():
    node = _node()
    assert (node.control, node.tp, node.model.ref, node.launch.image) == \
        ("local", 1, "qwen3.8-27b-nvidia-nvfp4@uncensored", DIGEST)
    assert [(c.index, c.type.key) for c in node.serving_cards] == [(1, "rtx5090")]
    c = CONTROLS["local"]
    assert (c.ranks, c.cards_per_rank, c.ssh, c.interconnect, c.model, c.launch, c.drives, c.selects) == \
        (1, 1, "forbidden", False, "required", "required", True, True)


def test_ssh_and_none_are_unchanged():
    assert [(c.name, c.ranks, c.cards_per_rank, c.ssh, c.interconnect, c.model, c.launch, c.drives, c.selects)
            for c in (CONTROLS["ssh"], CONTROLS["none"])] == [
        ("ssh", 2, 1, "required", True, "forbidden", "forbidden", True, False),
        ("none", None, None, "optional", False, "required", "optional", False, False)]


@pytest.mark.parametrize("old, new, says", [
    ("    platform: linux/amd64\n", "    platform: linux/amd64\n    ssh: op@192.0.2.20\n", "ssh"),   # nothing uses it
    ("      - {box: solo}", "      - {box: solo, interconnect: 198.51.100.1}", "interconnect"),   # no peer
    ("    model: qwen3.8-27b-nvidia-nvfp4@uncensored\n", "", "model"),
    ("    launch:\n", "    boot:\n", "launch"),                                             # the boot default
    ("{index: 0, type: rtx3090, serves: false}", "{index: 0, type: rtx5090, serves: true}", "1 card"),
    ("{index: 1, type: rtx5090, serves: true}", "{index: 1, type: rtx3090, serves: true}", "rtx3090"),
])
def test_a_local_node_missing_what_it_needs_is_refused_naming_it(old, new, says):
    assert old in LOCAL
    with pytest.raises(ValueError) as e:
        _fleet(LOCAL.replace(old, new))
    assert says in str(e.value)


def test_the_lb_drives_the_pair_and_a_local_node_in_parallel():
    """Parallel clusters (worklog/parallel-clusters.md): a pair the LB drives over SSH and a node it requests, each on
    its own card, each keyed by its name."""
    two = PAIR.replace("    control: ssh\n", "    control: ssh\n    run_at_boot: true\n    default: true\n").replace("clusters:", """  solo:
    hostname: solo-host
    label: solo-gpu
    platform: linux/amd64
    node_limits: {memory_gib: 24, cpus: 16, pids: 4096, nice: 10}
    cards:
      - {index: 0, type: rtx5090, serves: true}
clusters:""").replace("backends:", """  node:
    control: local
    run_at_boot: false
    model: qwen3.8-27b-nvidia-nvfp4
    url: http://vllm-node:8000
    launch:
      image: vllm/vllm-openai@sha256:""" + "1" * 64 + """
      max_model_len: 1024
      max_num_seqs: 1
      gpu_memory_utilization: 0.9
      reasoning_parser: qwen3
      tool_call_parser: qwen3_coder
      choices: [qwen3.8-27b-nvidia-nvfp4]
    ranks:
      - {box: solo}
backends:""")
    f = _fleet(two)
    assert [(c.name, c.control) for c in f.clusters.values()] == [("pair", "ssh"), ("node", "local")]


def test_the_config_reaches_a_local_node_over_nothing():
    (node,) = topology_of(_fleet(LOCAL))["clusters"]
    (host,) = node.hosts
    assert (node.name, node.control, host.label, host.box, host.ssh_target, host.interconnect) == \
        ("node", "local", "solo-gpu", "solo", "", "")
    assert node.url == "http://vllm-node:8000" and topology_of(_fleet(LOCAL))["vllm_boxes"] == ()


def test_a_local_node_and_a_classic_server_on_one_box_would_share_its_name():
    with pytest.raises(ValueError, match="both would be named"):
        topology_of(_fleet(LOCAL.replace("backends: {}", "backends:\n  solo: {url: http://192.0.2.20:8001}")))


def test_node_env_renders_the_boot_default_and_the_boxes_platform():
    lines = node_env(_fleet(LOCAL), "node")
    assert lines[:4] == [f"EDGE_NODE_IMAGE={DIGEST}", "EDGE_NODE_GPU=1", "EDGE_NODE_GPU_RUNTIME=nvidia", f"EDGE_NODE_MODEL={TWIN_DIR}"]
    assert f"EDGE_NODE_SERVED_NAME={SERVED}" in lines and "EDGE_NODE_PLATFORM=linux/amd64" in lines


# ── the models the panel may select: named once, each checked against the catalog and the card ─────

def test_a_local_node_lists_its_choices_and_boots_its_model():
    node = _node()
    assert [m.ref for m in node.choices] == ["qwen3.8-27b-nvidia-nvfp4@uncensored", "qwen3.8-27b-nvidia-nvfp4"]
    assert (node.launch.model, node.launch.checkpoint, node.launch.served_name) == \
        ("qwen3.8-27b-nvidia-nvfp4@uncensored", TWIN_DIR, SERVED)
    assert node.launch.served_name == node.model.spec.served_name, "the catalog row's, never typed"


@pytest.mark.parametrize("old, new, says", [
    ("choices: [qwen3.8-27b-nvidia-nvfp4@uncensored, qwen3.8-27b-nvidia-nvfp4]",
     "choices: [qwen3.8-27b-nvidia-nvfp4]", "boot default"),                          # the boot model not among them
    ("qwen3.8-27b-nvidia-nvfp4@uncensored, qwen3.8-27b-nvidia-nvfp4]",
     "qwen3.8-27b-nvidia-nvfp4@uncensored, qwen9-unknown]", "qwen9-unknown"),          # no such catalog row
    ("qwen3.8-27b-nvidia-nvfp4@uncensored, qwen3.8-27b-nvidia-nvfp4]",
     "qwen3.8-27b-nvidia-nvfp4@uncensored, qwen3.8-27b-nvidia-nvfp4@uncensored]", "twice"),
    ("      choices: [qwen3.8-27b-nvidia-nvfp4@uncensored, qwen3.8-27b-nvidia-nvfp4]\n", "", "choices"),
    ("choices: [qwen3.8-27b-nvidia-nvfp4@uncensored, qwen3.8-27b-nvidia-nvfp4]", "choices: []", "choices"),
    ("      max_model_len: 131072\n", "      served_name: mine\n      max_model_len: 131072\n", "served_name"),
])
def test_wrong_choices_are_refused_naming_them(old, new, says):
    assert old in LOCAL
    with pytest.raises(ValueError) as e:
        _fleet(LOCAL.replace(old, new))
    assert says in str(e.value)


def test_a_node_the_lb_only_observes_lists_no_choices():
    observed = LOCAL.replace("control: local", "control: none").replace(
        "      max_model_len: 131072\n", "      served_name: mine\n      max_model_len: 131072\n")
    with pytest.raises(ValueError, match="choices"):
        _fleet(observed)


# ── FIT: the chosen model's weights within the launch's share of the serving card ──────────────────

def test_a_choice_that_cannot_fit_the_card_is_refused_at_load_with_the_numbers():
    big = LOCAL.replace("qwen3.8-27b-nvidia-nvfp4@uncensored, qwen3.8-27b-nvidia-nvfp4]",
                        "qwen3.8-27b-nvidia-nvfp4@uncensored, qwen3.8-flash-next-hibrid48]")
    with pytest.raises(ValueError) as e:
        _fleet(big)
    assert "qwen3.8-flash-next-hibrid48: 98.1 GiB of weights > 0.90 × 32 GiB on rtx5090" in str(e.value)


def test_the_boot_default_is_fit_checked_too():
    with pytest.raises(ValueError) as e:
        _fleet(LOCAL.replace("gpu_memory_utilization: 0.9", "gpu_memory_utilization: 0.5"))
    assert "20.4 GiB of weights > 0.50 × 32 GiB on rtx5090" in str(e.value)


def test_the_live_5090_launch_fits_with_one_cards_kv_floor():
    """site-b serves THIS launch today (the twin, 131,072, 4 seats, 0.90): weights 20.4 GiB + the engine's own overhead
    on one card 3.56 GiB + one request's KV at the 5090's own block rule (90 blocks of 1,568 tokens, 4.31 GiB) = 28.27
    ≤ 28.8 GiB. The pair's per-rank rule scaled to one card (×2) would refuse it."""
    node = _node()
    fit = launch_fit(node.model, node.serving_cards[0].type, 0.9, 131072)
    assert (fit.weights_gib, fit.overhead_gib, round(fit.kv_gib, 2), fit.budget_gib) == (20.4, 3.559, 4.31, 28.8)
    assert fit.weights_gib + fit.overhead_gib + fit.kv_gib <= fit.budget_gib
    assert validate_launch(_node(), {"max_num_seqs": 4}).max_num_seqs == 4


def test_the_full_window_cannot_start_on_the_5090_at_any_fraction():
    """vLLM's boot line on the 5090: the whole card leaves 8.04 GiB of KV — 168 blocks, ≤ 254,016 tokens a request. So
    262,144 starts at NO fraction (1.0 vLLM refuses before loading: `test_a_fraction_vllm_refuses_at_start…`); 254,016
    is the largest window the card holds — its whole KV, PINNED (the fit's pin ceiling), the fraction one vLLM starts."""
    with pytest.raises(ValueError) as e:
        validate_launch(_node(), {"max_model_len": 262144, "gpu_memory_utilization": 0.98})
    assert ("20.4 GiB of weights + 3.6 GiB of engine overhead + 8.3 GiB of KV for one 262,144-token request > "
            "0.98 × 32 GiB on rtx5090") in str(e.value)
    with pytest.raises(ValueError, match="vLLM refuses before loading"):
        validate_launch(_node(), {"max_model_len": 262144, "gpu_memory_utilization": 1.0})
    from vllm_lb.fleet import pin_ceiling
    whole = {"gpu_memory_utilization": 0.95, "kv_cache_memory": pin_ceiling(_node().model,
                                                                             _node().serving_cards[0].type)}
    assert validate_launch(_node(), {**whole, "max_model_len": 254016}).max_model_len == 254016
    with pytest.raises(ValueError, match="169 blocks"):
        validate_launch(_node(), {**whole, "max_model_len": 254017})


def test_a_launch_whose_weights_fit_but_not_one_request_is_refused_with_the_kv_in_the_words():
    with pytest.raises(ValueError) as e:
        validate_launch(_node(), {"gpu_memory_utilization": 0.85})
    assert ("qwen3.8-27b-nvidia-nvfp4@uncensored: 20.4 GiB of weights + 3.6 GiB of engine overhead + 4.3 GiB of KV "
            "for one 131,072-token request > 0.85 × 32 GiB on rtx5090") in str(e.value)
    # A shorter window frees the KV it needs: the same fraction fits at 65,536.
    assert validate_launch(_node(), {"gpu_memory_utilization": 0.85, "max_model_len": 65536}).max_model_len == 65536


def test_a_choice_whose_row_states_no_single_card_block_rule_is_refused_not_scaled():
    with pytest.raises(ValueError) as e:
        _fleet(LOCAL.replace("qwen3.8-27b-nvidia-nvfp4@uncensored, qwen3.8-27b-nvidia-nvfp4]",
                             "qwen3.8-27b-nvidia-nvfp4@uncensored, qwen3.8-27b-dflash2]"))
    assert "qwen3.8-27b-dflash2" in str(e.value) and "single_card_kv_blocks" in str(e.value)


# ── vLLM's START CHECK: the fraction of the card's visible memory within its free memory at start ──────────────────
# site-b, 2026-10-11 23:21Z: the panel took GPU memory 0.99 (0.99 × 32 GiB priced fine), and the node restart-looped on
# vLLM's own refusal — "Free memory on device cuda:0 (30.85/31.36 GiB) on startup is less than desired GPU memory
# utilization (0.99, 31.04 GiB)". `request_memory` (v1/worker/utils.py:716-770 @2eb491e; the check at :761) runs at
# gpu_worker.py:521 for EVERY launch, a pinned pool too, before anything loads: the fraction × the card's VISIBLE memory
# must be within its FREE memory at start. Both are the card type's measured facts (rtx5090.yml), as vLLM printed them —
# to 0.01 GiB, so the product here is of the printed figures (0.99 × 31.36 = 31.05; vLLM's own, of the bytes, 31.04).

def _start_refusal(util: float, need: str) -> str:
    return (f"GPU memory {util:.2f} × 31.36 GiB = {need} GiB > 30.85 GiB free on an idle rtx5090 at start: vLLM "
            f"refuses before loading (v1/worker/utils.py:761 @2eb491e)")


@pytest.mark.parametrize("util, need", [(0.99, "31.05"), (1.0, "31.36")])
def test_a_fraction_vllm_refuses_at_start_is_refused_in_its_numbers(util, need):
    with pytest.raises(ValueError) as e:
        validate_launch(_node(), {"gpu_memory_utilization": util})
    assert _start_refusal(util, need) in str(e.value), str(e.value)
    assert "gpu_memory_utilization" in str(e.value), "the key it names"


@pytest.mark.parametrize("util", [0.95, 0.98])
def test_a_fraction_vllm_starts_still_fits(util):
    """0.98 × 31.36 = 30.73 ≤ 30.85: the largest preset vLLM starts on this card."""
    assert validate_launch(_node(), {"gpu_memory_utilization": util}).gpu_memory_utilization == util


def test_a_pinned_pool_does_not_skip_the_start_check():
    """--kv-cache-memory-bytes skips vLLM's profile-based sizing, never its start check on the fraction: the relaunch
    (16a198f) starts at 0.95; the same pin at 0.99 is refused before loading."""
    assert validate_launch(_rammed(), RELAUNCH).kv_cache_memory == WHOLE_CARD
    with pytest.raises(ValueError) as e:
        validate_launch(_rammed(), {**RELAUNCH, "gpu_memory_utilization": 0.99})
    assert _start_refusal(0.99, "31.05") in str(e.value), str(e.value)


def test_a_launch_block_at_a_fraction_vllm_refuses_is_refused_at_load_in_the_same_words():
    with pytest.raises(ValueError) as e:
        _fleet(LOCAL.replace("gpu_memory_utilization: 0.9", "gpu_memory_utilization: 0.99"))
    assert _start_refusal(0.99, "31.05") in str(e.value), str(e.value)


def test_a_card_type_that_states_no_start_memory_cannot_be_start_checked_and_says_which(tmp_path):
    """No measured visible or free memory → vLLM's start check cannot be judged: refused naming the missing facts and
    the type's own reason — never a default, never the spec sheet's 32 GB standing in."""
    from vllm_lb.hardware import TYPES_DIR, load_types
    for f in TYPES_DIR.glob("*.yml"):
        text = f.read_text()
        if f.stem == "rtx5090":
            text = re.sub(r"  visible_gib: \{.*\}\n", '  visible_gib: {value: null, source: "not measured on this card"}\n',
                          text)
            text = re.sub(r"  free_at_start_gib: \{.*\}\n",
                          '  free_at_start_gib: {value: null, source: "not measured on this card"}\n', text)
        (tmp_path / f.name).write_text(text)
    library = load_types(tmp_path)
    assert library["rtx5090"].free_at_start_gib.value is None, "the plant took"
    with pytest.raises(ValueError) as e:
        parse_fleet(LOCAL, "fleet.yml", library)
    said = str(e.value)
    for want in ("visible_gib", "free_at_start_gib", "not measured on this card", "rtx5090", "cannot be checked",
                 "v1/worker/utils.py:761"):
        assert want in said, (want, said)


def test_an_intent_is_fit_checked_by_the_same_words():
    with pytest.raises(ValueError) as by_intent:
        validate_launch(_node(), {"gpu_memory_utilization": 0.5})
    with pytest.raises(ValueError) as by_block:
        _fleet(LOCAL.replace("gpu_memory_utilization: 0.9", "gpu_memory_utilization: 0.5"))
    said = "qwen3.8-27b-nvidia-nvfp4@uncensored: 20.4 GiB of weights > 0.50 × 32 GiB on rtx5090"
    assert said in str(by_intent.value) and said in str(by_block.value)


# ── an intent: the boot default, moved within the same bounds, by the same validator ────────────────

def test_an_empty_intent_is_the_boot_default():
    assert validate_launch(_node(), {}) == _node().launch


def test_an_intent_moves_the_figures_and_nothing_else():
    moved = validate_launch(_node(), {"max_model_len": 65536, "max_num_seqs": 12, "gpu_memory_utilization": 0.85})
    assert (moved.max_model_len, moved.max_num_seqs, moved.gpu_memory_utilization) == (65536, 12, 0.85)
    assert (moved.image, moved.served_name, moved.reasoning_parser, moved.tool_call_parser, moved.model) == \
        (DIGEST, SERVED, "qwen3", "qwen3_coder", "qwen3.8-27b-nvidia-nvfp4@uncensored")
    assert MOVABLE == ("model", "max_model_len", "max_num_seqs", "gpu_memory_utilization", "max_num_batched_tokens",
                       "kv_cache_memory", "kv_offload_ram_gib", "kv_cache_dtype", "max_queue", "compute_type")


def test_an_intent_selects_one_of_the_choices_and_node_env_loads_its_folder():
    moved = validate_launch(_node(), {"model": "qwen3.8-27b-nvidia-nvfp4"})
    assert (moved.model, moved.checkpoint, moved.served_name) == ("qwen3.8-27b-nvidia-nvfp4", ORIGINAL_DIR, SERVED)
    lines = node_env(_fleet(LOCAL), "node", {"model": "qwen3.8-27b-nvidia-nvfp4"})
    assert f"EDGE_NODE_MODEL={ORIGINAL_DIR}" in lines


@pytest.mark.parametrize("model, says", [
    ("qwen3.8-27b-dflash2", "not among"),             # a real catalog row the fleet does not offer
    ("qwen9-unknown", "not among"),
    (7, "not among"),
])
def test_an_intent_naming_a_model_the_fleet_does_not_offer_is_refused(model, says):
    with pytest.raises(ValueError) as e:
        validate_launch(_node(), {"model": model})
    assert says in str(e.value) and "qwen3.8-27b-nvidia-nvfp4@uncensored" in str(e.value)


@pytest.mark.parametrize("key, value", [
    ("max_model_len", 10_000_000),       # past the model's window
    ("max_model_len", 0),
    ("max_num_seqs", 999),               # past the model's seats
    ("max_num_seqs", True),
    ("gpu_memory_utilization", 1.5),
    ("gpu_memory_utilization", "0.9"),
])
def test_an_intent_out_of_bounds_is_refused_with_the_launch_blocks_own_words(key, value):
    """The SAME validator: the refusal an intent gets is the one the fleet file gets for that value in its block."""
    with pytest.raises(ValueError) as by_intent:
        validate_launch(_node(), {key: value})
    block = LOCAL.replace(f"      {key}: {getattr(_node().launch, key):g}\n", f"      {key}: {json.dumps(value)}\n")
    assert block != LOCAL
    with pytest.raises(ValueError) as by_block:
        _fleet(block)
    said = str(by_block.value).split("launch.", 1)[1]
    assert str(by_intent.value).endswith("launch." + said), (str(by_intent.value), said)


@pytest.mark.parametrize("overrides, says", [
    ({"image": "evil/image@sha256:" + "2" * 64}, "image"),            # the fleet's alone: nothing the LB writes picks it
    ({"served_name": "other"}, "served_name"),
    ({"checkpoint": "elsewhere"}, "checkpoint"),
    ({"choices": ["qwen3.8-27b-dflash2"]}, "choices"),
    ({"reasoning_parser": "none"}, "reasoning_parser"),
    ({"tool_call_parser": "none"}, "tool_call_parser"),
    ({"tensor_parallel_size": 2}, "tensor_parallel_size"),           # no launch key
    (["max_num_seqs", 4], "mapping"),
])
def test_an_intent_naming_what_it_cannot_move_is_refused_naming_it(overrides, says):
    with pytest.raises(ValueError) as e:
        validate_launch(_node(), overrides)
    assert says in str(e.value) and "clusters.node" in str(e.value)


#: The same node, observed: its own deployment launches it, so its launch block names its served name and no choices.
OBSERVED = LOCAL.replace("control: local", "control: none").replace(
    "      choices: [qwen3.8-27b-nvidia-nvfp4@uncensored, qwen3.8-27b-nvidia-nvfp4]\n", "").replace(
    "      max_model_len: 131072\n", f"      served_name: {SERVED}\n      max_model_len: 131072\n")


def test_an_intent_for_a_node_the_lb_only_observes_is_refused():
    observed = _fleet(OBSERVED).clusters["node"]
    with pytest.raises(ValueError, match="only observes"):
        validate_launch(observed, {"max_num_seqs": 4})
    # Its node-env is unchanged: no EDGE_NODE_MODEL (its folder stays its own compose wiring), the rest the same.
    assert node_env(_fleet(OBSERVED), "node") == [ln for ln in node_env(_fleet(LOCAL), "node")
                                                  if not ln.startswith("EDGE_NODE_MODEL=")]


# ── the PREFILL CHUNK (vLLM's --max-num-batched-tokens): optional, movable, within vLLM's own rules ───────────────
# A KV lever on one card: vLLM sizes its peak activation on the chunk (2.18 GiB at its default on the 5090), so a
# smaller chunk leaves the KV pool more of the card, at the cost of prefill speed. ABSENT is vLLM's own default — no
# number of ours stands in for it.

#: The same node, booted with an explicit chunk.
CHUNKED = LOCAL.replace("      max_num_seqs: 8\n", "      max_num_seqs: 8\n      max_num_batched_tokens: 1024\n")
#: The chunk the catalog's single-card overhead for the 27B was measured at: vLLM's own default on the 5090.
MEASURED_CHUNK = 2048


def test_the_prefill_chunk_is_optional_and_absent_is_vllms_own_default():
    assert CHUNKED != LOCAL
    assert _node().launch.max_num_batched_tokens is None, "absent: vLLM's own default, never a number of ours"
    assert _fleet(CHUNKED).clusters["node"].launch.max_num_batched_tokens == 1024
    # Rendered either way: EMPTY when absent, so a server file's own value can never stand in for vLLM's default.
    assert "EDGE_NODE_MAX_NUM_BATCHED_TOKENS=" in node_env(_fleet(LOCAL), "node")
    assert "EDGE_NODE_MAX_NUM_BATCHED_TOKENS=1024" in node_env(_fleet(CHUNKED), "node")


def test_an_intent_moves_the_prefill_chunk_and_null_gives_it_back_to_vllm():
    assert validate_launch(_node(), {"max_num_batched_tokens": 1024}).max_num_batched_tokens == 1024
    chunked = _fleet(CHUNKED).clusters["node"]
    assert validate_launch(chunked, {}).max_num_batched_tokens == 1024, "the boot default's, unmoved"
    assert validate_launch(chunked, {"max_num_batched_tokens": None}).max_num_batched_tokens is None
    assert "EDGE_NODE_MAX_NUM_BATCHED_TOKENS=" in node_env(_fleet(CHUNKED), "node", {"max_num_batched_tokens": None})


@pytest.mark.parametrize("value, says", [
    (0, "from 1"), (-512, "from 1"), (True, "from 1"), ("1024", "from 1"), (1024.5, "from 1"),
    # vLLM v0.31.1rc1.dev273 vllm/config/scheduler.py:328: "must be greater than or equal to max_num_seqs" (8 here)
    (4, "under max_num_seqs 8"),
])
def test_a_chunk_vllm_would_refuse_is_refused_in_the_same_words_for_block_and_intent(value, says):
    with pytest.raises(ValueError) as by_intent:
        validate_launch(_node(), {"max_num_batched_tokens": value})
    with pytest.raises(ValueError) as by_block:
        _fleet(CHUNKED.replace("max_num_batched_tokens: 1024", f"max_num_batched_tokens: {json.dumps(value)}"))
    assert says in str(by_intent.value), str(by_intent.value)
    said = str(by_block.value).split("launch.", 1)[1]
    assert str(by_intent.value).endswith("launch." + said), (str(by_intent.value), said)


def test_the_chunk_is_judged_against_the_seats_of_the_same_launch():
    with pytest.raises(ValueError, match="max_num_batched_tokens 16 is under max_num_seqs 24"):
        validate_launch(_fleet(CHUNKED).clusters["node"], {"max_num_batched_tokens": 16, "max_num_seqs": 24})
    assert validate_launch(_node(), {"max_num_batched_tokens": 24, "max_num_seqs": 24}).max_num_batched_tokens == 24


def test_a_chunk_up_to_the_measured_one_is_priced_at_the_measured_overhead_and_says_so():
    """The overhead was MEASURED at vLLM's default chunk. A smaller one only lowers the activation, so it is charged
    the measured figure — conservative, never a credit scaled by a formula nobody measured — and the fit says so."""
    node = _node()
    card = node.serving_cards[0].type
    default = launch_fit(node.model, card, 0.9, 131072)
    small = launch_fit(node.model, card, 0.9, 131072, 1024)
    assert small.overhead_gib == default.overhead_gib == 3.559
    assert (default.overhead_chunk, small.overhead_chunk) == (MEASURED_CHUNK, MEASURED_CHUNK)
    assert default.note == "" and launch_fit(node.model, card, 0.9, 131072, MEASURED_CHUNK).note == ""
    assert "2,048" in small.note and "1,024" in small.note and "not credited" in small.note, small.note
    assert small.why == "", "a smaller chunk is checked, not refused"


def test_a_chunk_larger_than_the_measured_one_is_refused_not_guessed():
    with pytest.raises(ValueError) as e:
        validate_launch(_node(), {"max_num_batched_tokens": 4096})
    said = str(e.value)
    assert "4,096" in said and "2,048" in said and "cannot be checked" in said, said


def test_a_smaller_chunk_never_makes_the_fit_check_stricter():
    """Every launch the fleet takes at vLLM's default chunk it takes at any smaller one, with a need never larger."""
    node = _node()
    card = node.serving_cards[0].type
    need = lambda f: f.weights_gib + f.overhead_gib + f.kv_gib
    taken = 0
    for util in (0.85, 0.88, 0.9, 0.92, 0.95, 0.97, 1.0):
        for length in (32768, 65536, 131072, 196608, 254016, 262144):
            base = {"gpu_memory_utilization": util, "max_model_len": length}
            try:
                validate_launch(node, base)
            except ValueError:
                continue
            taken += 1
            for chunk in (8, 512, 1024, 1568, MEASURED_CHUNK):
                assert validate_launch(node, {**base, "max_num_batched_tokens": chunk}).max_num_batched_tokens == chunk
                assert need(launch_fit(node.model, card, util, length, chunk)) <= \
                    need(launch_fit(node.model, card, util, length))
    assert taken >= 10, "the grid exercises launches the fleet takes"


def test_an_overhead_measured_at_no_stated_chunk_refuses_an_explicit_one():
    """Without the chunk the overhead was measured at, no explicit chunk can be judged against it: refused, while
    vLLM's own default — what the figure was taken at — is checked as before."""
    import copy
    from vllm_lb.catalog import document, ModelCatalog
    doc = document()
    row = next(m for m in doc["models"] if m["id"] == "qwen3.8-27b-nvidia-nvfp4")
    del row["memory"]["single_card_overhead"]["chunk"]
    node = parse_fleet(LOCAL, "fleet.yml", catalog=ModelCatalog.from_dict(doc)).clusters["node"]
    assert validate_launch(node, {}).max_num_batched_tokens is None
    with pytest.raises(ValueError, match="no prefill chunk"):
        validate_launch(node, {"max_num_batched_tokens": 1024})


def test_node_env_renders_an_intent(tmp_path):
    lines = node_env(_fleet(LOCAL), "node", {"max_num_seqs": 4})
    assert "EDGE_NODE_MAX_NUM_SEQS=4" in lines and "EDGE_NODE_MAX_MODEL_LEN=131072" in lines


def test_the_cli_applies_an_intent_file_or_refuses_it_with_exit_78(tmp_path):
    fleet = write(tmp_path, LOCAL)
    good, bad, junk = tmp_path / "good.json", tmp_path / "bad.json", tmp_path / "junk.json"
    good.write_text(json.dumps({"max_model_len": 32768}))
    bad.write_text(json.dumps({"image": "evil/image@sha256:" + "2" * 64}))
    junk.write_text("{not json")

    def run(*args):
        return subprocess.run([sys.executable, "-m", "vllm_lb.fleet", "node-env", fleet, "node", *args],
                              capture_output=True, text=True, timeout=60)

    ok = run(str(good))
    assert ok.returncode == 0, ok.stderr
    assert "EDGE_NODE_MAX_MODEL_LEN=32768" in ok.stdout.splitlines()
    for path, says in ((bad, "image"), (junk, "junk.json")):
        r = run(str(path))
        assert r.returncode == 78 and says in r.stderr and r.stdout == "", (r.returncode, r.stderr)


# ── every box's platform ────────────────────────────────────────────────────────────────────────────

def test_every_box_names_its_platform_never_inferred():
    with pytest.raises(ValueError, match="platform"):
        _fleet(LOCAL.replace("    platform: linux/amd64\n", ""))
    with pytest.raises(ValueError, match="platform"):
        _fleet(LOCAL.replace("platform: linux/amd64", "platform: x86"))


def test_a_gb10_box_is_arm_because_the_superchip_is():
    assert _fleet(PAIR).boxes["node-1"].platform == "linux/arm64"
    wrong = PAIR.replace("    label: node-1-gpu\n    platform: linux/arm64\n",
                         "    label: node-1-gpu\n    platform: linux/amd64\n")
    assert wrong != PAIR
    with pytest.raises(ValueError) as e:
        _fleet(wrong)
    assert "gb10" in str(e.value) and "linux/arm64" in str(e.value)


def test_a_discrete_cards_host_platform_is_its_boxes():
    from vllm_lb.hardware import profile
    assert profile("gb10").host_platform.value == "linux/arm64"
    assert profile("rtx5090").host_platform.value is None, "a discrete card runs in any host; the box says which"
    assert profile("radeon8060s").host_platform.value == "linux/amd64", "the Ryzen AI Max+ 395 APU is x86"


def test_the_one_host_shape_still_parses():
    assert _fleet(ONE_HOST).clusters["node"].control == "none"


# ── the KV POOL PINNED (vLLM --kv-cache-memory-bytes) and a KV TIER IN THE HOST'S RAM (--kv-offloading-size) ─────
# Operator, 2026-10-12: "offload to RAM to fix KV issues … on site-b we can enable it", and the card's "0.99". Both are
# optional MOVABLE launch keys; absent is vLLM's own (the pool sized from gpu_memory_utilization; no tier), and
# node-env renders each line EMPTY then — byte for byte today's launch. vLLM at 2eb491e, read in the image the node runs:
#   * --kv-cache-memory-bytes (engine/arg_utils.py:1352) sizes the pool and SKIPS the profile-based sizing
#     (v1/worker/gpu_worker.py:635-660, "This does not respect the gpu_memory_utilization config"), while the start
#     check on gpu_memory_utilization still runs (v1/worker/utils.py:716-770, called unconditionally at
#     gpu_worker.py:521) — so a pinned launch is priced on the WHOLE card, the fraction kept for that check alone;
#   * --kv-offloading-size N (GiB; native backend = OffloadingConnector, config/vllm.py:1115-1122) maps a tier of N GiB
#     in the host's /dev/shm (v1/kv_offload/cpu/shared_offload_region.py:110), pinned whole at boot, refused by vLLM
#     when /dev/shm has less free (shm_broadcast.py:249-258). It parks a preempted request's PROMPT KV and copies it
#     back into VRAM blocks before the request runs again; decode never reads host RAM.

#: The box with the fleet's bound on its RAM for a KV tier.
RAMMED = LOCAL.replace("    platform: linux/amd64\n", "    platform: linux/amd64\n    kv_offload_ram_max_gib: 32\n")
#: The live node's "fully utilize" figure (gpu_worker.py:1020 on site-b, 2026-10-11 22:05:42, at 0.95 and 16 seats).
WHOLE_CARD = 8_501_810_176
#: The relaunch the coordinator posts (2026-10-12): the live window and seats, the card's whole KV, a 32 GiB RAM tier.
RELAUNCH = {"max_model_len": 131072, "max_num_seqs": 16, "gpu_memory_utilization": 0.95,
            "kv_cache_memory": WHOLE_CARD, "kv_offload_ram_gib": 32}


def _rammed():
    return _fleet(RAMMED).clusters["node"]


def test_both_are_movable_and_absent_is_vllms_own_launch():
    assert "kv_cache_memory" in MOVABLE and "kv_offload_ram_gib" in MOVABLE
    launch = _node().launch
    assert (launch.kv_cache_memory, launch.kv_offload_ram_gib) == (None, None)
    lines = node_env(_fleet(LOCAL), "node")
    assert "EDGE_NODE_KV_CACHE_MEMORY=" in lines and "EDGE_NODE_KV_OFFLOAD_RAM_GIB=" in lines, \
        "rendered EMPTY: a server file's own value can never stand in for vLLM's own"
    assert _fleet(RAMMED).boxes["solo"].kv_offload_ram_max_gib == 32
    assert _fleet(LOCAL).boxes["solo"].kv_offload_ram_max_gib is None, "absent: no bound, so no tier — never assumed"


def test_the_relaunch_is_taken_and_rendered():
    launch = validate_launch(_rammed(), RELAUNCH)
    assert (launch.kv_cache_memory, launch.kv_offload_ram_gib, launch.max_num_seqs) == (WHOLE_CARD, 32, 16)
    lines = node_env(_fleet(RAMMED), "node", RELAUNCH)
    assert f"EDGE_NODE_KV_CACHE_MEMORY={WHOLE_CARD}" in lines and "EDGE_NODE_KV_OFFLOAD_RAM_GIB=32" in lines
    back = validate_launch(_rammed(), {**RELAUNCH, "kv_cache_memory": None, "kv_offload_ram_gib": None})
    assert (back.kv_cache_memory, back.kv_offload_ram_gib) == (None, None), "null gives each back to vLLM"


def test_a_launch_block_may_name_them_too():
    boot = RAMMED.replace("      max_num_seqs: 8\n", "      max_num_seqs: 8\n      kv_offload_ram_gib: 16\n"
                                                       f"      kv_cache_memory: {WHOLE_CARD}\n")
    launch = _fleet(boot).clusters["node"].launch
    assert (launch.kv_offload_ram_gib, launch.kv_cache_memory) == (16, WHOLE_CARD)
    assert validate_launch(_fleet(boot).clusters["node"], {}).kv_offload_ram_gib == 16, "the boot default's"
    assert "EDGE_NODE_KV_OFFLOAD_RAM_GIB=16" in node_env(_fleet(boot), "node")


# — the RAM tier ——————————————————————————————————————————————————————————————————————————————

def test_a_ram_tier_past_the_boxes_bound_is_refused_naming_the_bound_and_dev_shm():
    assert validate_launch(_rammed(), {"kv_offload_ram_gib": 32}).kv_offload_ram_gib == 32
    with pytest.raises(ValueError) as e:
        validate_launch(_rammed(), {"kv_offload_ram_gib": 33})
    said = str(e.value)
    for want in ("33 GiB", "kv_offload_ram_max_gib 32", "solo", "/dev/shm", "shm_broadcast.py:249-258"):
        assert want in said, (want, said)


def test_a_ram_tier_on_a_box_that_states_no_bound_is_refused_never_assumed():
    with pytest.raises(ValueError) as e:
        validate_launch(_node(), {"kv_offload_ram_gib": 8})
    assert "states no kv_offload_ram_max_gib" in str(e.value) and "solo" in str(e.value), str(e.value)
    with pytest.raises(ValueError, match="no kv_offload_ram_max_gib"):
        _fleet(LOCAL.replace("      max_num_seqs: 8\n", "      max_num_seqs: 8\n      kv_offload_ram_gib: 8\n"))


@pytest.mark.parametrize("value", [0, -8, True, "32", 8.5])
def test_a_ram_tier_vllm_would_refuse_is_refused(value):
    """0 is no tier to vLLM either: its spec raises without cpu_bytes_to_use (v1/kv_offload/cpu/spec.py:96-100) — off
    is null, never 0."""
    with pytest.raises(ValueError) as e:
        validate_launch(_rammed(), {"kv_offload_ram_gib": value})
    assert "kv_offload_ram_gib must be a whole number of GiB from 1" in str(e.value), str(e.value)


@pytest.mark.parametrize("value", [0, -1, "x", 1.5])
def test_the_boxes_bound_is_a_whole_number_of_gib(value):
    with pytest.raises(ValueError, match="kv_offload_ram_max_gib must be a whole number of GiB from 1"):
        _fleet(RAMMED.replace("kv_offload_ram_max_gib: 32", f"kv_offload_ram_max_gib: {json.dumps(value)}"))


def test_a_ram_tier_on_unified_memory_is_refused_in_the_familys_words():
    """On a GB10 the host's RAM IS the GPU's pool: a RAM tier takes the memory the KV pool lives in. The family says so
    (hardware/memory.py), whatever the box's bound — the fleet never asks which kind of card it is."""
    unified = f"""
panel:
  style: vibe
boxes:
  box-a:
    hostname: box-a-host
    label: box-a-gpu
    platform: linux/arm64
    kv_offload_ram_max_gib: 32
    cards: [{{index: 0, type: gb10, serves: true}}]
clusters:
  node:
    control: none
    model: qwen3.8-27b-nvidia-nvfp4
    url: http://vllm-node:8000
    launch:
      image: {DIGEST}
      served_name: Qwen/Qwen3.8-27B
      max_model_len: 131072
      max_num_seqs: 8
      gpu_memory_utilization: 0.9
      kv_offload_ram_gib: 8
      reasoning_parser: qwen3
      tool_call_parser: qwen3_coder
    ranks:
      - {{box: box-a}}
backends: {{}}
"""
    with pytest.raises(ValueError) as e:
        _fleet(unified)
    said = str(e.value)
    assert "unified memory: the host RAM is the GPU's own pool" in said and "gb10" in said, said
    assert _fleet(unified.replace("      kv_offload_ram_gib: 8\n", "")).clusters["node"].launch.kv_offload_ram_gib is None


def test_the_ram_tier_is_on_the_fit_line_and_never_in_the_cards_sum():
    node = _rammed()
    card = node.serving_cards[0].type
    plain, tiered = launch_fit(node.model, card, 0.95, 131072), launch_fit(node.model, card, 0.95, 131072,
                                                                           kv_offload_ram_gib=32)
    assert (plain.ram_gib, tiered.ram_gib) == (None, 32)
    assert (tiered.weights_gib, tiered.overhead_gib, tiered.kv_gib, tiered.budget_gib) == \
        (plain.weights_gib, plain.overhead_gib, plain.kv_gib, plain.budget_gib), "host RAM is not the card's"


# — the KV pool pinned ———————————————————————————————————————————————————————————————————————

def test_a_pinned_pool_is_priced_on_the_whole_card():
    """The catalog's single-card overhead was taken so that the fit at the whole card reproduces vLLM's own "fully
    utilize" figure (catalog.json, single_card_overhead's source): weights + engine + the pinned pool ≤ the card."""
    node = _node()
    f = launch_fit(node.model, node.serving_cards[0].type, 0.95, 131072, kv_cache_memory=WHOLE_CARD)
    assert f.budget_gib == 32, "the whole card: vLLM sizes nothing from the fraction once the pool is pinned"
    assert f.pool_gib == pytest.approx(WHOLE_CARD / 2**30)
    assert f.weights_gib + f.overhead_gib + f.pool_gib <= f.budget_gib
    assert launch_fit(node.model, node.serving_cards[0].type, 0.95, 131072).pool_gib is None


def test_a_pin_the_card_cannot_hold_is_refused_with_the_numbers():
    with pytest.raises(ValueError) as e:
        validate_launch(_node(), {"kv_cache_memory": 9_000_000_000})
    said = str(e.value)
    for want in ("20.4 GiB of weights", "3.6 GiB of engine overhead", "8.38 GiB of pinned KV pool", "32 GiB",
                 "rtx5090"):
        assert want in said, (want, said)


def test_a_pin_under_one_request_is_refused_as_vllm_would():
    """vLLM does not start a pool that cannot hold one request of max_model_len (v1/core/kv_cache_utils.py:1031, "To
    serve at least one request with the model's max seq len"): one 131,072-token request takes 84 + 6 blocks."""
    with pytest.raises(ValueError) as e:
        validate_launch(_node(), {"kv_cache_memory": 4_000_000_000})
    said = str(e.value)
    assert "one 131,072-token request" in said and "90 blocks" in said, said
    assert validate_launch(_node(), {"kv_cache_memory": 4_000_000_000, "max_model_len": 32768}).kv_cache_memory


@pytest.mark.parametrize("value", [0, -1, True, "8501810176", 8.5e9])
def test_a_pin_that_is_no_whole_number_of_bytes_is_refused(value):
    with pytest.raises(ValueError) as e:
        validate_launch(_node(), {"kv_cache_memory": value})
    assert "kv_cache_memory must be a whole number of bytes" in str(e.value), str(e.value)
