"""WHAT A NODE THE LB ONLY OBSERVES SERVES, AND HOW IT IS LAUNCHED — named once, in the fleet file.

The one-place rule (AGENTS.md, operator 2026-10-11): a standalone server's node (control: none) is started by its own
compose project, so the LB cannot read its launch off a launch record of its own. The fleet file names it:

  clusters.<name>.model    `<catalog id>[@<variant>]` — validated against the model catalog (lb/vllm_lb/library):
                           an unknown id or variant is refused, naming it. Required for control: none; refused for
                           control: ssh, whose model is the operator's Start on the panel (a second definition).
  clusters.<name>.launch   for control: none only — image (pinned by digest), served_name, max_model_len,
                           max_num_seqs, gpu_memory_utilization, reasoning_parser, tool_call_parser; every key when
                           the block is present, no other.

`python -m vllm_lb.fleet node-env <fleet file> <cluster>` renders that launch for the standalone node's compose as
KEY=VALUE lines — its GPU the cluster's serving card's index, so deploy/standalone's server file need not say it again.
"""
from __future__ import annotations

import subprocess
import sys

import pytest

from tests.fleet_files import ONE_HOST, PAIR, write
from vllm_lb.fleet import node_env, parse_fleet

DIGEST = "vllm/vllm-openai@sha256:" + "0" * 64
LAUNCH = f"""    launch:
      image: {DIGEST}
      served_name: qwen3.8-27b-uncensored
      max_model_len: 131072
      max_num_seqs: 8
      gpu_memory_utilization: 0.9
      reasoning_parser: qwen3
      tool_call_parser: qwen3_coder
"""
NODE = ONE_HOST.replace("    url: http://vllm-node:8000\n", "    url: http://vllm-node:8000\n" + LAUNCH)


def _fleet(text: str):
    return parse_fleet(text, "fleet.yml")


# ── the model ───────────────────────────────────────────────────────────────────────────────────────

def test_a_node_names_its_catalog_row_and_variant():
    model = _fleet(ONE_HOST).clusters["node"].model
    assert (model.spec.id, model.variant, model.ref) == \
        ("qwen3.8-27b-nvidia-nvfp4", "uncensored", "qwen3.8-27b-nvidia-nvfp4@uncensored")


def test_a_model_with_no_variant_is_the_original_checkpoint():
    f = _fleet(ONE_HOST.replace("qwen3.8-27b-nvidia-nvfp4@uncensored", "qwen3.8-27b-nvidia-nvfp4"))
    assert (f.clusters["node"].model.variant, f.clusters["node"].model.ref) == (None, "qwen3.8-27b-nvidia-nvfp4")


@pytest.mark.parametrize("ref, says", [
    ("qwen9-unknown", "qwen9-unknown"),                                  # no such catalog row
    ("qwen3.8-27b-nvidia-nvfp4@spicy", "spicy"),                         # no such variant
    ("qwen3.8-27b-dflash2@uncensored", "qwen3.8-27b-dflash2"),           # this row has no twin
    ("qwen3.8-27b-nvidia-nvfp4@", "model"),
    ("@uncensored", "model"),
])
def test_an_unknown_model_or_variant_is_refused_naming_it(ref, says):
    with pytest.raises(ValueError) as e:
        _fleet(ONE_HOST.replace("qwen3.8-27b-nvidia-nvfp4@uncensored", ref))
    assert says in str(e.value)


def test_a_node_the_lb_observes_must_name_its_model():
    with pytest.raises(ValueError, match="model"):
        _fleet(ONE_HOST.replace("    model: qwen3.8-27b-nvidia-nvfp4@uncensored\n", ""))


def test_a_pair_the_lb_drives_names_no_model_its_start_does():
    with pytest.raises(ValueError, match="model"):
        _fleet(PAIR.replace("    control: ssh\n", "    control: ssh\n    model: qwen3.8-27b-nvidia-nvfp4\n"))


# ── the launch ──────────────────────────────────────────────────────────────────────────────────────

def test_the_launch_block_is_read_whole():
    launch = _fleet(NODE).clusters["node"].launch
    assert (launch.image, launch.served_name, launch.max_model_len, launch.max_num_seqs,
            launch.gpu_memory_utilization, launch.reasoning_parser, launch.tool_call_parser) == \
        (DIGEST, "qwen3.8-27b-uncensored", 131072, 8, 0.9, "qwen3", "qwen3_coder")
    assert _fleet(ONE_HOST).clusters["node"].launch is None, "no block: no launch named here"


@pytest.mark.parametrize("old, new, says", [
    (f"image: {DIGEST}", "image: vllm/vllm-openai:latest", "digest"),               # a moving tag
    (f"image: {DIGEST}", "image: vllm/vllm-openai@sha256:abc", "digest"),
    ("max_model_len: 131072", "max_model_len: 0", "max_model_len"),
    ("max_model_len: 131072", "max_model_len: 524288", "max_model_len"),          # past the model's window
    ("max_num_seqs: 8", "max_num_seqs: '8'", "max_num_seqs"),
    ("max_num_seqs: 8", "max_num_seqs: 64", "max_num_seqs"),                      # past the model's limit
    ("gpu_memory_utilization: 0.9", "gpu_memory_utilization: 1.5", "gpu_memory_utilization"),
    ("reasoning_parser: qwen3", "reasoning_parser: ''", "reasoning_parser"),
    ("tool_call_parser: qwen3_coder", "tool_call_parser: qwen3 coder", "tool_call_parser"),   # not one token
    ("      served_name: qwen3.8-27b-uncensored\n", "", "served_name"),           # a key missing
    ("      tool_call_parser: qwen3_coder\n", "      tool_call_parser: qwen3_coder\n      dtype: auto\n", "dtype"),
])
def test_a_wrong_launch_is_refused_naming_the_key(old, new, says):
    with pytest.raises(ValueError) as e:
        _fleet(NODE.replace(old, new))
    assert says in str(e.value)


def test_a_pair_the_lb_drives_takes_no_launch_block_tp_rank_launches_it():
    pair = PAIR.replace("    control: ssh\n", "    control: ssh\n" + LAUNCH)
    with pytest.raises(ValueError, match="launch"):
        _fleet(pair)


# ── the node's compose values, rendered from the one place ──────────────────────────────────────────

def test_node_env_renders_the_launch_and_takes_the_gpu_from_the_serving_card():
    assert node_env(_fleet(NODE), "node") == [
        f"EDGE_NODE_IMAGE={DIGEST}", "EDGE_NODE_GPU=1", "EDGE_NODE_GPU_RUNTIME=nvidia", "EDGE_NODE_SERVED_NAME=qwen3.8-27b-uncensored",
        "EDGE_NODE_MAX_MODEL_LEN=131072", "EDGE_NODE_MAX_NUM_SEQS=8",
        "EDGE_NODE_MAX_NUM_BATCHED_TOKENS=",            # not set: vLLM's own default chunk, empty so nothing stands in
        "EDGE_NODE_KV_CACHE_MEMORY=",                   # not set: the pool sized from the fraction
        "EDGE_NODE_KV_OFFLOAD_RAM_GIB=",                # not set: no KV tier in the host's RAM
        "EDGE_NODE_KV_CACHE_DTYPE=",                    # not set: vLLM's own KV dtype (auto), no flag
        "EDGE_NODE_GPU_MEM_UTIL=0.9",
        "EDGE_NODE_REASONING_PARSER=qwen3", "EDGE_NODE_TOOL_CALL_PARSER=qwen3_coder",
        "EDGE_NODE_MFU_PATCH=on",                       # on: its row's quant_method is modelopt_mixed
        # the container's limits, the box's node_limits (tests/test_node_guards.py)
        "EDGE_NODE_MEMORY=24g", "EDGE_NODE_CPUS=16", "EDGE_NODE_PIDS_LIMIT=4096", "EDGE_NODE_NICE=10",
        "EDGE_NODE_PLATFORM=linux/amd64",
        "EDGE_NODE_ENGINE=vllm"]                        # its row's node_engine: node-engine.vllm.yml


@pytest.mark.parametrize("text, cluster, says", [
    (NODE, "nope", "nope"),                                  # no such cluster
    (ONE_HOST, "node", "launch"),                            # no launch block to render
    (PAIR, "pair", "launch"),                                # tp-rank launches the pair, not compose
])
def test_node_env_refuses_what_it_cannot_render(text, cluster, says):
    with pytest.raises(ValueError, match=says):
        node_env(_fleet(text), cluster)


def test_the_cli_prints_the_lines_or_refuses_with_exit_78(tmp_path):
    path = write(tmp_path, NODE)
    ok = subprocess.run([sys.executable, "-m", "vllm_lb.fleet", "node-env", path, "node"], capture_output=True,
                        text=True, timeout=60)
    assert ok.returncode == 0, ok.stderr
    assert ok.stdout.splitlines()[1] == "EDGE_NODE_GPU=1"
    bad = subprocess.run([sys.executable, "-m", "vllm_lb.fleet", "node-env", path, "nope"], capture_output=True,
                         text=True, timeout=60)
    assert bad.returncode == 78 and "nope" in bad.stderr and bad.stdout == ""
    usage = subprocess.run([sys.executable, "-m", "vllm_lb.fleet"], capture_output=True, text=True, timeout=60)
    assert usage.returncode == 64 and "node-env" in usage.stderr
