"""SGLANG ON A ONE-HOST NODE (operator 2026-10-26: "can we switch on site-b to sglang?" — "proceed implementation").

The pair has run SGLang through tp-rank since 2026-10-02; a standalone node ran vLLM, faster-whisper and nemo only. The
27B's SGLang row loads the SAME checkpoint folder as its vLLM row (catalog `path` / `uncensored.path`), so a node that
holds those weights selects it as one more choice; its engine is `node_engine: sglang` (fleet.py SglangEngine), its
image one more entry of the launch block's per-engine `image` map, its compose override node-engine.sglang.yml.

SGLang takes a STATIC FRACTION of the card at start — weights, KV pool, GDN state slots — by its own rule for the card,
so its fit is the WHOLE ENGINE'S MEASURED MEMORY (catalog `engine_memory`), as faster-whisper's and nemo's: refused
where none is measured, never scaled from the pair's per-rank figures. vLLM's pinned pool, RAM tier and prefill chunk
are not SGLang's: a switch NAMING one is refused, and the boot block's own (a vLLM boot launch on site-b pins both) are
left behind with the engine that reads them — what the LB's switch already does with a stored wish (proxy `_read_by`).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from vllm_lb.catalog import document, ModelCatalog
from vllm_lb.fleet import ENGINES, node_env, parse_fleet, validate_launch
from vllm_lb.probe import ENGINE_SGLANG

SGLANG_ROW = "qwen3.8-27b-sglang-dflash2"
SGL_REF = f"{SGLANG_ROW}@uncensored"
VLLM_IMAGE = "vllm/vllm-openai@sha256:" + "1" * 64
SGL_IMAGE = "sha256:" + "2" * 64
#: A site-b-shaped node: the 5090 serves, the vLLM boot launch PINS its KV pool and parks KV in a RAM tier.
NODE = f"""
panel:
  style: vibe
boxes:
  solo:
    hostname: solo-host
    label: solo-gpu
    platform: linux/amd64
    kv_offload_ram_max_gib: 32
    node_limits: {{memory_gib: 64, cpus: 16, pids: 4096, nice: 10}}
    cards:
      - {{index: 0, type: rtx3090, serves: false}}
      - {{index: 1, type: rtx5090, serves: true}}
clusters:
  node:
    control: local
    model: qwen3.8-27b-nvidia-nvfp4@uncensored
    url: http://vllm-node:8000
    launch:
      image: {{vllm: "{VLLM_IMAGE}", sglang: "{SGL_IMAGE}"}}
      max_model_len: 98304
      max_num_seqs: 16
      gpu_memory_utilization: 0.98
      kv_cache_memory: 8501810176
      kv_offload_ram_gib: 32
      reasoning_parser: qwen3
      tool_call_parser: qwen3_coder
      choices: [qwen3.8-27b-nvidia-nvfp4@uncensored, {SGL_REF}]
    ranks:
      - {{box: solo}}
backends: {{}}
"""


def _catalog(gib: float, seats: int) -> ModelCatalog:
    """The shipped catalog with the SGLang row's whole-engine memory stated (`gib` at `seats` requests)."""
    raw = document()
    row = next(r for r in raw["models"] if r["id"] == SGLANG_ROW)
    row["engine_memory"] = {"gib": gib, "max_num_seqs": seats, "source": "test: a measured whole-engine figure"}
    return ModelCatalog.from_dict(raw)


def _node(gib: float = 30.5, seats: int = 32):
    return parse_fleet(NODE, "fleet.yml", catalog=_catalog(gib, seats))


def _env(lines):
    return dict(line.split("=", 1) for line in lines)


def test_the_27b_sglang_row_runs_on_the_sglang_node_engine():
    assert ModelCatalog.load().get(SGLANG_ROW).node_engine == "sglang"
    sgl = ENGINES["sglang"]
    assert sgl.vocabulary == ENGINE_SGLANG, "the probe reads its /metrics as the pair's SGLang"
    assert [s.key for s in sgl.settings("chat")] == ["model", "max_num_seqs", "max_model_len", "kv_cache_dtype"]


def test_an_sglang_choice_with_no_measured_engine_memory_is_refused_never_guessed():
    shipped = ModelCatalog.load().get(SGLANG_ROW)
    if shipped.engine_memory is not None:
        pytest.skip("the shipped row is measured now; the refusal is pinned with a catalog that states none")
    with pytest.raises(ValueError, match="states no engine_memory"):
        parse_fleet(NODE, "fleet.yml")


def test_a_switch_to_sglang_renders_its_own_settings_and_leaves_vllms_behind():
    env = _env(node_env(_node(), "node", {"model": SGL_REF, "max_num_seqs": 8}))
    assert (env["EDGE_NODE_ENGINE"], env["EDGE_NODE_IMAGE"]) == ("sglang", SGL_IMAGE)
    assert env["EDGE_NODE_MODEL"] == "Qwen3.8-27B-NVFP4_uncensored", "the same folder the vLLM row loads"
    assert env["EDGE_NODE_SERVED_NAME"] == "Qwen/Qwen3.8-27B-SGLang"
    assert (env["EDGE_NODE_MAX_MODEL_LEN"], env["EDGE_NODE_MAX_NUM_SEQS"]) == ("98304", "8")
    assert (env["EDGE_NODE_REASONING_PARSER"], env["EDGE_NODE_TOOL_CALL_PARSER"]) == ("qwen3", "qwen3_coder")
    assert env["EDGE_NODE_KV_CACHE_DTYPE"] == ""
    for vllm_only in ("EDGE_NODE_GPU_MEM_UTIL", "EDGE_NODE_KV_CACHE_MEMORY", "EDGE_NODE_KV_OFFLOAD_RAM_GIB",
                      "EDGE_NODE_MAX_NUM_BATCHED_TOKENS"):
        assert vllm_only not in env
    assert env["EDGE_NODE_MFU_PATCH"] == "off", "the MFU byte-size patch is vLLM's sitecustomize"


def test_the_boot_blocks_vllm_settings_stay_with_vllm():
    """The boot launch pins a KV pool and a RAM tier — vLLM's. A switch to SGLang naming neither is SGLang's launch,
    not a refusal of settings it never had; the vLLM boot launch keeps both."""
    node = _node().clusters["node"]
    moved = validate_launch(node, {"model": SGL_REF})
    assert (moved.engine, moved.kv_cache_memory, moved.kv_offload_ram_gib) == ("sglang", None, None)
    boot = validate_launch(node, {})
    assert (boot.engine, boot.kv_cache_memory, boot.kv_offload_ram_gib) == ("vllm", 8501810176, 32)
    assert _env(node_env(_node(), "node", {}))["EDGE_NODE_MFU_PATCH"] == "on"


@pytest.mark.parametrize("key, value", [("kv_cache_memory", 8501810176), ("kv_offload_ram_gib", 32),
                                        ("max_num_batched_tokens", 2048)])
def test_a_vllm_setting_named_on_an_sglang_switch_is_refused(key, value):
    with pytest.raises(ValueError, match=f"runs on sglang, which takes no {key}"):
        validate_launch(_node().clusters["node"], {"model": SGL_REF, key: value})


def test_the_sglang_fit_is_its_measured_engine_memory():
    node = _node(gib=30.5, seats=16).clusters["node"]
    validate_launch(node, {"model": SGL_REF, "max_num_seqs": 16})
    with pytest.raises(ValueError, match="more is not measured"):
        validate_launch(node, {"model": SGL_REF, "max_num_seqs": 17})
    with pytest.raises(ValueError, match="GiB measured > 32 GiB on rtx5090"):
        parse_fleet(NODE, "fleet.yml", catalog=_catalog(33.0, 16))
