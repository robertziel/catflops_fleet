"""A vLLM PROFILE'S ENGINE FLAGS (operator 2026-10-20: "prepare engine that will run GLM-5.3-Flash on our cluster").

GLM-5.3-Flash on GB10 needs five vLLM settings no Qwen profile ever named — each measured necessary on 2x DGX Spark by
the published recipes (himorishige/glm53-flash-2x-dgx-spark-recipe docs/pitfalls.md; Classmethod 2026-09-02):

  kv_cache_dtype       fp8_e4m3   bf16 KV cannot hold one 262K request beside the weights
  block_size           2304       the page the fp8 paged MQA expects; without it the output corrupts with no error
  moe_backend          marlin     the auto-picked FLASHINFER_CUTLASS NVFP4 MoE repeats tokens on sm_121 silently
  enforce_eager        true       CUDA graphs not measured on the pair
  language_model_only  true       the vision tower's warmup profile exhausts the unified pool

and two that serve its images and video instead (MiaAI-Lab/GLM-5.3-Flash-NVFP4-Dual-DGX-Spark @aed98a1, operator
2026-10-21 "Proceed implementation based on" it):

  skip_mm_profiling    true       no max-size image/video dummy forward at init — that one OOMs a rank
  limit_mm_per_prompt  {image: 4, video: 1}   the bound that profile would have measured, stated instead

They are the profile's own fields (vLLM's settings are never a free-form engine_args), ONE table in the catalog
(`VLLM_FLAGS`), sent by `launch_env` as one CLUSTER_* setting each and turned into flags by tp-rank. A profile that
names none sends none — every launch before them is byte for byte what it was."""
from __future__ import annotations

import copy
import json

import pytest

from tests.test_tp_rank import FLASH, MODEL, lay_out_checkpoint, last_run, box_a  # noqa: F401 (fixture)
from vllm_lb.catalog import document, CLUSTER_MODE, ModelCatalog
from vllm_lb.cluster_exec import launch_env

# The cluster vLLM images' own `CacheDType` (vllm/config/cache.py), verbatim: myllmbox/glm53-flash-cluster-vllm:v2
# (vLLM 0.1.dev20051+g487ecf187) L19-37 and myllmbox/qwen38-flash-next-cluster-vllm:v5 (vLLM 0.29.0) L39-57 carry the
# same list, read 2026-10-22 (lb/vllm_lb/kv_cache_dtypes.json). OLDER_IMAGE stands for any image whose
# list lacks a value.
IMAGE_CACHE_DTYPES = '''CacheDType = Literal[
    "auto",
    "float16",
    "bfloat16",
    "fp8",
    "fp8_e4m3",
    "fp8_e5m2",
    "fp8_inc",
    "fp8_ds_mla",
    "turboquant_k8v4",
    "turboquant_4bit_nc",
    "turboquant_k3v4_nc",
    "turboquant_3bit_nc",
    "int4_per_token_head",
    "int8_per_token_head",
    "fp8_per_token_head",
    "nvfp4",
    "nvfp4_4over6",
]
'''
OLDER_IMAGE = 'CacheDType = Literal["auto", "bfloat16", "fp8", "fp8_e4m3", "fp8_e5m2", "fp8_inc"]\n'


def image_accepts(box_a, cache_dtype_source, image="myllmbox/qwen38-flash-next-cluster-vllm:v5"):
    """Lay out the image's vllm package as the fake docker shows it to `up`: config/cache.py holding
    `cache_dtype_source` (None: no such file). Its __init__ refuses to run, and so does cache.py's own torch import:
    reading the list must never import vLLM."""
    site = box_a.root / "image-site" / image.replace("/", "_").replace(":", "_")
    (site / "vllm" / "config").mkdir(parents=True)
    (site / "vllm" / "__init__.py").write_text("raise SystemExit('vllm was imported')\n")
    if cache_dtype_source is not None:
        (site / "vllm" / "config" / "cache.py").write_text(
            "from typing import Literal\n\nimport torch_that_is_not_here\n\n" + cache_dtype_source)
    box_a.set(vllm_site={image: str(site)})

FLAGS = {"kv_cache_dtype": "fp8_e4m3", "block_size": 2304, "moe_backend": "marlin", "enforce_eager": True,
         "language_model_only": True, "skip_mm_profiling": True, "limit_mm_per_prompt": {"image": 4, "video": 1}}
ENV = {"CLUSTER_KV_CACHE_DTYPE": "fp8_e4m3", "CLUSTER_BLOCK_SIZE": "2304", "CLUSTER_MOE_BACKEND": "marlin",
       "CLUSTER_ENFORCE_EAGER": "1", "CLUSTER_LANGUAGE_MODEL_ONLY": "1", "CLUSTER_SKIP_MM_PROFILING": "1",
       "CLUSTER_LIMIT_MM_PER_PROMPT": '{"image":4,"video":1}'}
# DeepSeek-V4-Flash on GB10's B12X stack (operator 2026-10-23; eugr/spark-vllm-docker @0bdc9c94
# recipes/deepseek-v4-flash-0731.yaml): the tokenizer, loader and kernel families, the graph capture, and three config
# objects — each one argv element of JSON, flat (a list of scalars is fine, a nested object is not).
DSV4_FLAGS = {"trust_remote_code": True, "tokenizer_mode": "deepseek_v4", "load_format": "b12x",
              "linear_backend": "b12x", "attention_backend": "B12X", "max_cudagraph_capture_size": 48,
              "compilation_config": {"cudagraph_mode": "FULL_AND_PIECEWISE", "custom_ops": ["all"]},
              "reasoning_config": {"reasoning_parser": "deepseek_v4", "reasoning_start_str": "",
                                   "reasoning_end_str": ""},
              "default_chat_template_kwargs": {"thinking": True, "reasoning_effort": "high"}}
DSV4_ENV = {"CLUSTER_TRUST_REMOTE_CODE": "1", "CLUSTER_TOKENIZER_MODE": "deepseek_v4", "CLUSTER_LOAD_FORMAT": "b12x",
            "CLUSTER_LINEAR_BACKEND": "b12x", "CLUSTER_ATTENTION_BACKEND": "B12X",
            "CLUSTER_MAX_CUDAGRAPH_CAPTURE_SIZE": "48",
            "CLUSTER_COMPILATION_CONFIG": '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}',
            "CLUSTER_REASONING_CONFIG": '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"","reasoning_end_str":""}',
            "CLUSTER_DEFAULT_CHAT_TEMPLATE_KWARGS": '{"thinking":true,"reasoning_effort":"high"}'}


def _catalog(**profile_over):
    raw = document()
    row = copy.deepcopy(next(r for r in raw["models"] if r["id"] == FLASH))
    # The row WITHOUT its own engine flags (its KV dtype since 2026-10-26), so each test states exactly the ones it means.
    row["launch"]["cluster"].pop("kv_cache_dtype", None)
    row["launch"]["cluster"].update(profile_over)
    raw["models"] = [row]
    return ModelCatalog.from_dict(raw), row["id"]


def test_a_vllm_profile_states_its_engine_flags():
    cat, mid = _catalog(**FLAGS)
    assert cat.get(mid).profile(CLUSTER_MODE).vllm_flags == FLAGS


@pytest.mark.parametrize("key, bad", [
    ("kv_cache_dtype", "fp8 --x"), ("kv_cache_dtype", 8), ("block_size", "2304"), ("block_size", 0),
    ("block_size", True), ("moe_backend", ""), ("enforce_eager", "yes"), ("language_model_only", 1),
    ("limit_mm_per_prompt", '{"image":4}'), ("limit_mm_per_prompt", {}), ("limit_mm_per_prompt", {"image": 0}),
    ("limit_mm_per_prompt", {"image": True}), ("limit_mm_per_prompt", {"Image x": 1})])
def test_a_flag_of_the_wrong_kind_is_refused_naming_it(key, bad):
    with pytest.raises(ValueError, match=key):
        _catalog(**{key: bad})


def test_a_vllm_profile_states_the_deepseek_v4_flags():
    cat, mid = _catalog(**DSV4_FLAGS)
    assert cat.get(mid).profile(CLUSTER_MODE).vllm_flags == DSV4_FLAGS
    assert {k: launch_env(cat.get(mid))[k] for k in DSV4_ENV} == DSV4_ENV


@pytest.mark.parametrize("key, bad", [
    ("trust_remote_code", 1), ("tokenizer_mode", "deepseek v4"), ("load_format", ""), ("attention_backend", 7),
    ("max_cudagraph_capture_size", 0), ("max_cudagraph_capture_size", "48"),
    ("compilation_config", {}), ("compilation_config", '{"cudagraph_mode":"FULL"}'),
    ("compilation_config", {"pass_config": {"fuse_act_quant": True}}), ("compilation_config", {"x": [{"a": 1}]}),
    ("reasoning_config", {"bad key": "x"}), ("default_chat_template_kwargs", {"thinking": "{"})])
def test_a_deepseek_v4_flag_of_the_wrong_kind_is_refused_naming_it(key, bad):
    with pytest.raises(ValueError, match=key):
        _catalog(**{key: bad})


def test_tp_rank_passes_each_deepseek_v4_flag_as_one_argv_element(box_a):  # noqa: F811
    lay_out_checkpoint(box_a.models / MODEL)
    proc = box_a.run("up", "0", "7", "8", "131072", FLASH, config={"CLUSTER_HOST_IP": "192.0.2.201", **DSV4_ENV})
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    run = last_run(box_a)
    assert "--trust-remote-code" in run
    for flag, env in (("--tokenizer-mode", "CLUSTER_TOKENIZER_MODE"), ("--load-format", "CLUSTER_LOAD_FORMAT"),
                      ("--linear-backend", "CLUSTER_LINEAR_BACKEND"), ("--attention-backend", "CLUSTER_ATTENTION_BACKEND"),
                      ("--max-cudagraph-capture-size", "CLUSTER_MAX_CUDAGRAPH_CAPTURE_SIZE"),
                      ("--compilation-config", "CLUSTER_COMPILATION_CONFIG"),
                      ("--reasoning-config", "CLUSTER_REASONING_CONFIG"),
                      ("--default-chat-template-kwargs", "CLUSTER_DEFAULT_CHAT_TEMPLATE_KWARGS")):
        assert run[run.index(flag) + 1] == DSV4_ENV[env], flag


def test_tp_rank_passes_no_deepseek_v4_flag_unasked(box_a):  # noqa: F811
    lay_out_checkpoint(box_a.models / MODEL)
    proc = box_a.run("up", "0", "7", "16", "262144", FLASH, config={"CLUSTER_HOST_IP": "192.0.2.201"})
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    run = last_run(box_a)
    for flag in ("--trust-remote-code", "--tokenizer-mode", "--load-format", "--linear-backend", "--attention-backend",
                 "--max-cudagraph-capture-size", "--compilation-config", "--reasoning-config",
                 "--default-chat-template-kwargs"):
        assert flag not in run, flag


@pytest.mark.parametrize("key, bad", [("CLUSTER_TOKENIZER_MODE", "a;b"), ("CLUSTER_ATTENTION_BACKEND", "B12X x"),
                                      ("CLUSTER_MAX_CUDAGRAPH_CAPTURE_SIZE", "4x"), ("CLUSTER_TRUST_REMOTE_CODE", "y"),
                                      ("CLUSTER_COMPILATION_CONFIG", '{"a":{"b":1}}'),
                                      ("CLUSTER_REASONING_CONFIG", "reasoning"),
                                      ("CLUSTER_DEFAULT_CHAT_TEMPLATE_KWARGS", '{"thinking":true};x')])
def test_tp_rank_refuses_a_malformed_deepseek_v4_flag_before_touching_docker(box_a, key, bad):  # noqa: F811
    lay_out_checkpoint(box_a.models / MODEL)
    proc = box_a.run("up", "0", "7", "16", "262144", FLASH, config={"CLUSTER_HOST_IP": "192.0.2.201", key: bad})
    assert proc.returncode != 0 and key in (proc.stdout + proc.stderr)
    assert not [c for c in box_a.calls() if c[0] == "run"]


def test_an_sglang_profile_refuses_vllm_flags():
    raw = document()
    row = copy.deepcopy(next(r for r in raw["models"]
                             if r["launch"].get("cluster", {}).get("engine") == "sglang"))
    row["launch"]["cluster"]["moe_backend"] = "marlin"
    raw["models"] = [row]
    with pytest.raises(ValueError, match="moe_backend"):
        ModelCatalog.from_dict(raw)


def test_launch_env_sends_each_flag_once_and_a_profile_without_sends_none():
    cat, mid = _catalog(**FLAGS)
    env = launch_env(cat.get(mid))
    assert {k: env[k] for k in ENV} == ENV
    off, _ = _catalog(enforce_eager=False)
    assert launch_env(off.get(mid))["CLUSTER_ENFORCE_EAGER"] == "0"
    plain, _ = _catalog()
    assert not set(ENV) & set(launch_env(plain.get(mid))), "no flag named, no setting sent"


def test_tp_rank_passes_each_flag_to_vllm(box_a):  # noqa: F811
    lay_out_checkpoint(box_a.models / MODEL)
    image_accepts(box_a, IMAGE_CACHE_DTYPES)
    proc = box_a.run("up", "0", "7", "16", "262144", FLASH, config={"CLUSTER_HOST_IP": "192.0.2.201", **ENV})
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    run = last_run(box_a)
    assert run[run.index("--kv-cache-dtype") + 1] == "fp8_e4m3"
    assert run[run.index("--block-size") + 1] == "2304"
    assert run[run.index("--moe-backend") + 1] == "marlin"
    assert "--enforce-eager" in run and "--language-model-only" in run and "--skip-mm-profiling" in run
    assert run[run.index("--limit-mm-per-prompt") + 1] == '{"image":4,"video":1}', "one argv element, never re-parsed"


def test_tp_rank_passes_none_unasked_and_no_bool_flag_at_zero(box_a):  # noqa: F811
    lay_out_checkpoint(box_a.models / MODEL)
    proc = box_a.run("up", "0", "7", "16", "262144", FLASH,
                     config={"CLUSTER_HOST_IP": "192.0.2.201", "CLUSTER_ENFORCE_EAGER": "0"})
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    run = last_run(box_a)
    for flag in ("--kv-cache-dtype", "--block-size", "--moe-backend", "--enforce-eager", "--language-model-only",
                 "--skip-mm-profiling", "--limit-mm-per-prompt"):
        assert flag not in run, flag
    assert not [c for c in box_a.calls() if "--entrypoint" in c], "no KV dtype asked, the image is not read"


@pytest.mark.parametrize("key, bad", [("CLUSTER_BLOCK_SIZE", "23x"), ("CLUSTER_KV_CACHE_DTYPE", "fp8;rm"),
                                      ("CLUSTER_ENFORCE_EAGER", "yes"), ("CLUSTER_MOE_BACKEND", "a b"),
                                      ("CLUSTER_LIMIT_MM_PER_PROMPT", '{"image":4};x'),
                                      ("CLUSTER_LIMIT_MM_PER_PROMPT", '{"image":{"count":4}}'),
                                      ("CLUSTER_SKIP_MM_PROFILING", "2")])
def test_tp_rank_refuses_a_malformed_flag_before_touching_docker(box_a, key, bad):  # noqa: F811
    lay_out_checkpoint(box_a.models / MODEL)
    proc = box_a.run("up", "0", "7", "16", "262144", FLASH, config={"CLUSTER_HOST_IP": "192.0.2.201", key: bad})
    assert proc.returncode != 0 and key in (proc.stdout + proc.stderr)
    assert not [c for c in box_a.calls() if c[0] == "run"]


@pytest.mark.parametrize("dtype", ["nvfp4", "nvfp4_4over6", "fp8_e4m3", "auto", "int4_per_token_head"])
def test_tp_rank_passes_a_kv_dtype_the_image_accepts(box_a, dtype):  # noqa: F811
    lay_out_checkpoint(box_a.models / MODEL)
    image_accepts(box_a, IMAGE_CACHE_DTYPES)
    proc = box_a.run("up", "0", "7", "16", "262144", FLASH,
                     config={"CLUSTER_HOST_IP": "192.0.2.201", "CLUSTER_KV_CACHE_DTYPE": dtype})
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    run = last_run(box_a)
    assert run[run.index("--kv-cache-dtype") + 1] == dtype


@pytest.mark.parametrize("source, dtype", [(IMAGE_CACHE_DTYPES, "fp4"), (IMAGE_CACHE_DTYPES, "int8"),
                                           (OLDER_IMAGE, "nvfp4")], ids=["fp4", "int8", "older-image-nvfp4"])
def test_tp_rank_refuses_a_kv_dtype_the_image_does_not_accept_naming_its_list(box_a, source, dtype):  # noqa: F811
    lay_out_checkpoint(box_a.models / MODEL)
    image_accepts(box_a, source)
    proc = box_a.run("up", "0", "7", "16", "262144", FLASH,
                     config={"CLUSTER_HOST_IP": "192.0.2.201", "CLUSTER_KV_CACHE_DTYPE": dtype})
    out = proc.stdout + proc.stderr
    assert proc.returncode != 0 and "CLUSTER_KV_CACHE_DTYPE" in out and f"'{dtype}'" in out, out
    assert "auto, bfloat16, fp8, fp8_e4m3" in out or "auto, float16, bfloat16, fp8" in out, "the image's own list"
    assert not [c for c in box_a.calls() if c[:2] == ["run", "-d"]], "no launch"


@pytest.mark.parametrize("source", [None, "CacheDType = str\n", ""],
                         ids=["no-cache-py", "not-a-literal", "no-cachedtype"])
def test_tp_rank_fails_loudly_when_the_image_states_no_kv_dtypes(box_a, source):  # noqa: F811
    """No cache.py, no Literal, no CacheDType: refused naming why — never "pass it and see"."""
    lay_out_checkpoint(box_a.models / MODEL)
    image_accepts(box_a, source)
    proc = box_a.run("up", "0", "7", "16", "262144", FLASH,
                     config={"CLUSTER_HOST_IP": "192.0.2.201", "CLUSTER_KV_CACHE_DTYPE": "nvfp4"})
    out = proc.stdout + proc.stderr
    assert proc.returncode != 0 and "CacheDType" in out and "myllmbox/qwen38-flash-next-cluster-vllm:v5" in out, out
    assert not [c for c in box_a.calls() if c[:2] == ["run", "-d"]]


def test_tp_rank_fails_loudly_when_the_image_is_not_on_the_box(box_a):  # noqa: F811
    box_a.set(vllm_site={})                             # the fake box holds the image by default: take it away
    lay_out_checkpoint(box_a.models / MODEL)
    proc = box_a.run("up", "0", "7", "16", "262144", FLASH,
                     config={"CLUSTER_HOST_IP": "192.0.2.201", "CLUSTER_KV_CACHE_DTYPE": "nvfp4"})
    out = proc.stdout + proc.stderr
    assert proc.returncode != 0 and "No such image" in out, out
    assert not [c for c in box_a.calls() if c[:2] == ["run", "-d"]]
