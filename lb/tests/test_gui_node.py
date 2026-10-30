"""THE GUI NODE ON THE AMD iGPU IS LAUNCHABLE (worklog/parallel-clusters.md, "The gui node"): measured 2026-10-17 —
MAI-UI-2B served by vLLM 0.19.1 (AMD's gfx1151 ROCm build) on a Radeon 8060S at gpu_memory_utilization 0.10, 32,768
tokens, 4 seats (rocm-measure/MEASURED.md). What the fleet does with it, all as data:

  1. the card type states its measured start memory, so a launch on it PARSES — bounded by the BOX: a unified-memory
     pool is the box's own RAM and vLLM's start check does not guard it (its free figure overstates the box's free RAM
     by ~29 GiB), so the box states `gpu_ram_max_gib` (no default) and a pool above it is refused;
  2. the catalog row states the model's measured memory; a reasoning parser it has none of is ABSENT (no flag), and the
     MFU byte-size patch is rendered on only for a checkpoint whose quant method needs it — off here;
  3. `node-env` renders a launch identical in effect to the measured run.
"""
from __future__ import annotations

import pytest

from vllm_lb.fleet import node_env, parse_fleet

IMAGE = "rocm/vllm@sha256:394194d36edcf9b36bcb563e143b21b80e64e7d04f33a447b448c0c0c00c04a8"

GUI = f"""
boxes:
  amd-box:
    hostname: amd-box-host
    label: amd-box-gpu
    platform: linux/amd64
    node_limits: {{memory_gib: 24, cpus: 16, pids: 4096, nice: 10}}
    gpu_ram_max_gib: 32
    cards:
      - {{index: 0, type: radeon8060s, serves: true}}
clusters:
  gui:
    control: local
    model: mai-ui-2b
    url: http://192.0.2.30:8000
    launch:
      image: {IMAGE}
      max_model_len: 32768
      max_num_seqs: 4
      gpu_memory_utilization: 0.10
      tool_call_parser: hermes
      choices: [mai-ui-2b]
    ranks:
      - {{box: amd-box}}
backends: {{}}
"""


def test_the_measured_launch_parses_and_renders_the_run_that_served():
    """`node-env` for the measured launch: the same image, card, checkpoint, served name, window, seats, fraction and
    tool parser as rocm-measure/run.sh; no reasoning parser (the run's `--reasoning-parser qwen3` emptied every answer's
    content on this model — MEASURED.md); the ROCm attachment; the MFU patch OFF (a BF16 checkpoint: the patch would
    only add a byte size for `modelopt_mixed`, and its sitecustomize recursed under ROCm's platform probe)."""
    lines = node_env(parse_fleet(GUI, "fleet.yml"), "gui")
    env = dict(line.split("=", 1) for line in lines)
    assert {k: env[k] for k in ("EDGE_NODE_IMAGE", "EDGE_NODE_GPU", "EDGE_NODE_GPU_RUNTIME", "EDGE_NODE_MODEL",
                                "EDGE_NODE_SERVED_NAME", "EDGE_NODE_MAX_MODEL_LEN", "EDGE_NODE_MAX_NUM_SEQS",
                                "EDGE_NODE_GPU_MEM_UTIL", "EDGE_NODE_REASONING_PARSER", "EDGE_NODE_TOOL_CALL_PARSER",
                                "EDGE_NODE_MFU_PATCH", "EDGE_NODE_PLATFORM", "EDGE_NODE_MAX_NUM_BATCHED_TOKENS")} == {
        "EDGE_NODE_IMAGE": IMAGE, "EDGE_NODE_GPU": "0", "EDGE_NODE_GPU_RUNTIME": "rocm",
        "EDGE_NODE_MODEL": "MAI-UI-2B", "EDGE_NODE_SERVED_NAME": "Tongyi-MAI/MAI-UI-2B",
        "EDGE_NODE_MAX_MODEL_LEN": "32768", "EDGE_NODE_MAX_NUM_SEQS": "4", "EDGE_NODE_GPU_MEM_UTIL": "0.1",
        "EDGE_NODE_REASONING_PARSER": "", "EDGE_NODE_TOOL_CALL_PARSER": "hermes", "EDGE_NODE_MFU_PATCH": "off",
        "EDGE_NODE_PLATFORM": "linux/amd64", "EDGE_NODE_MAX_NUM_BATCHED_TOKENS": ""}


def test_the_fleets_fit_prices_the_measured_launch_on_the_measured_pool():
    """The LB's own fit at the measured launch: 0.10 × the 124.0 GiB pool = 12.40 GiB budget; 3.963 GiB of weights +
    3.087 GiB of one-card overhead (DERIVED from the same boot, not fit-checked against a second one) + one 32,768-token
    request's KV (2,048 blocks × 1,835,008 B = 3.5 GiB) = 10.55 GiB — it fits."""
    from vllm_lb.fleet import launch_fit
    from vllm_lb.hardware import profile
    gui = parse_fleet(GUI, "fleet.yml").clusters["gui"]
    fit = launch_fit(gui.model, profile("radeon8060s"), 0.10, 32768)
    assert (round(fit.budget_gib, 2), fit.weights_gib, fit.overhead_gib, fit.kv_gib) == (12.4, 3.963, 3.087, 3.5)
    assert fit.why == "" and fit.weights_gib + fit.overhead_gib + fit.kv_gib <= fit.budget_gib


def test_a_unified_box_that_states_no_bound_for_its_pool_runs_no_launch():
    with pytest.raises(ValueError) as refused:
        parse_fleet(GUI.replace("    gpu_ram_max_gib: 32\n", ""), "fleet.yml")
    assert "gpu_ram_max_gib" in str(refused.value) and "amd-box" in str(refused.value)


def test_a_pool_above_the_boxs_bound_is_refused_in_its_numbers_though_vllms_start_check_would_pass_it():
    """0.90 × 124.0 = 111.6 GiB ≤ 119.26 free: vLLM's start check passes it — and the box, whose RAM it is, cannot."""
    with pytest.raises(ValueError, match=r"111\.60 GiB GPU pool > box amd-box's gpu_ram_max_gib 32 GiB"):
        parse_fleet(GUI.replace("gpu_memory_utilization: 0.10", "gpu_memory_utilization: 0.90"), "fleet.yml")


def test_the_bound_is_the_unified_familys_a_discrete_card_ignores_it():
    from vllm_lb.hardware import profile
    assert profile("rtx5090").memory.pool_problem(object(), 1e9) == ""


def test_a_reasoning_parser_is_passed_only_where_the_launch_names_one():
    with_one = GUI.replace("      tool_call_parser: hermes\n", "      tool_call_parser: hermes\n      reasoning_parser: qwen3\n")
    assert "EDGE_NODE_REASONING_PARSER=qwen3" in node_env(parse_fleet(with_one, "fleet.yml"), "gui")


def test_the_mfu_patch_is_on_only_for_a_checkpoint_whose_quant_method_needs_it():
    """The patch adds vLLM's MFU model a byte size for `modelopt_mixed` (deploy/standalone/vllm-mfu): selected by the
    catalog row's `quant_method`, never always on."""
    from tests.test_cluster_local import FLEET
    env = dict(line.split("=", 1) for line in node_env(parse_fleet(FLEET, "fleet.yml"), "node"))
    assert env["EDGE_NODE_MFU_PATCH"] == "on", "the 27B NVFP4 checkpoint reports modelopt_mixed"


def test_the_committed_amd_node_example_is_a_valid_parallel_fleet():
    from pathlib import Path
    text = (Path(__file__).resolve().parents[1] / "fleet.amd-node.example.yml").read_text()
    f = parse_fleet(text, "fleet.amd-node.example.yml")
    assert [(c.name, c.control, c.default, c.run_at_boot) for c in f.clusters.values()] == [
        ("pair", "ssh", True, True), ("gui", "local", False, False), ("stt", "local", True, True)]
    # stt is the fleet's one transcription cluster: its task's default (a loose `whisper-1` goes there).
    assert f.clusters["gui"].serving_cards[0].type.key == "radeon8060s" and f.clusters["gui"].unlaunchable == ""
    # the speech node beside it: transcription, run at boot, both its engines' choices fit the box
    assert (f.clusters["stt"].task, f.clusters["stt"].unlaunchable, sorted(f.clusters["stt"].images)) == \
        ("transcription", "", ["faster-whisper", "vllm"])
