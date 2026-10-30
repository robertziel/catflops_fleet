"""WHAT AN ENGINE SAYS IT RUNS WITH, read off its own HTTP API — the settings a node this LB only observes can be shown
by (worklog/standalone-cluster-panel.md, slice 3): nothing here is assumed, and a figure the engine does not publish
stays null.

  * `vllm:cache_config_info` carries, beside the pool the probe already reads, the GPU memory fraction the engine was
    started with (`gpu_memory_utilization`) and two WORDS: its KV cache dtype and whether prefix caching is on —
    carried verbatim, never turned into a figure;
  * the scrape names the vocabulary it was read in — `vllm` or `sglang` — which is the engine, as the node itself
    says it, not as a launch profile intended;
  * `/v1/models` names the checkpoint each served model was loaded from (`root`): the folder the engine really holds.
"""
from __future__ import annotations

import json

from vllm_lb.config import Config
from vllm_lb.probe import parse_metrics, parse_sglang_metrics, parse_vllm_metrics, probe_backend
from vllm_lb.state import CLUSTER, Backend

CFG = Config(probe_timeout_s=0.5)

#: The shape a vLLM nightly publishes (2026-10-11, a one-GPU node: fp8 KV cache, a hybrid model's block size).
CACHE_CONFIG = (
    'vllm:cache_config_info{_block_size_resolved="True",block_size="1568",cache_dtype="fp8_e4m3",'
    'enable_prefix_caching="True",engine="0",gpu_memory_utilization="0.9",kv_cache_max_concurrency="1.28888",'
    'kv_cache_memory_bytes="None",kv_cache_size_tokens="168937",num_gpu_blocks="116"} 1.0\n'
    'vllm:num_requests_running{engine="0",model_name="model-a"} 0.0\n'
    'vllm:num_requests_waiting{engine="0",model_name="model-a"} 0.0\n'
)


def test_the_memory_fraction_and_the_cache_words_are_read_off_the_engines_own_cache_config():
    got = parse_vllm_metrics(CACHE_CONFIG)
    assert got["gpu_memory_utilization"] == 0.9
    assert got["kv_cache_dtype"] == "fp8_e4m3", "the engine's own word, verbatim"
    assert got["prefix_caching"] is True
    assert got["kv_pool_bytes"] is None, "`None` is the engine saying it was not pinned — never a figure"
    assert got["kv_pool_tokens"] == 168937 and got["kv_pool_blocks"] == 116 and got["kv_block_tokens"] == 1568


def test_an_engine_that_says_nothing_of_its_cache_says_nothing_here():
    got = parse_vllm_metrics('vllm:num_requests_running{engine="0"} 1.0\n')
    assert (got["gpu_memory_utilization"], got["kv_cache_dtype"], got["prefix_caching"]) == (None, None, None)
    off = parse_vllm_metrics('vllm:cache_config_info{enable_prefix_caching="False"} 1.0\n')
    assert off["prefix_caching"] is False
    odd = parse_vllm_metrics('vllm:cache_config_info{enable_prefix_caching="maybe"} 1.0\n')
    assert odd["prefix_caching"] is None, "a word that is neither True nor False is not read as either"


def test_the_scrape_names_the_engine_that_wrote_it():
    assert parse_metrics(CACHE_CONFIG)["engine"] == "vllm"
    assert parse_sglang_metrics('sglang:num_running_reqs{tp_rank="0"} 2.0\n')["engine"] == "sglang"
    assert parse_metrics("")["engine"] is None, "an empty exposition is no engine at all"
    assert parse_metrics('process_start_time_seconds 1.0\n')["engine"] is None


def test_the_throughput_block_carries_what_the_engine_says_it_runs_with():
    """`throughput.engine_config`: the engine's own words for how it runs, off the last scrape — null per word it does
    not say, every one null while /metrics is not answering."""
    from vllm_lb.throughput import Throughput

    t = Throughput(CLUSTER, interval_s=5.0)
    b = Backend(kind=CLUSTER, url="http://node-a:8000", live=True)
    assert t.view(b, 10.0, 0.0)["engine_config"] == {"engine": None, "gpu_memory_utilization": None,
                                                     "kv_cache_dtype": None, "prefix_caching": None}
    t.observe(live=True, scrape=parse_metrics(CACHE_CONFIG), mono_at=1.0, at=1.0)
    assert t.view(b, 10.0, 0.0)["engine_config"] == {"engine": "vllm", "gpu_memory_utilization": 0.9,
                                                     "kv_cache_dtype": "fp8_e4m3", "prefix_caching": True}


def test_the_checkpoint_each_model_was_loaded_from_is_read_off_v1_models():
    models = {"data": [{"id": "model-a", "root": "/models/Model-A-NVFP4", "max_model_len": 131072}]}

    def get(url, timeout):
        if url.endswith("/v1/models"):
            return 200, json.dumps(models).encode()
        if url.endswith("/metrics"):
            return 200, CACHE_CONFIG.encode()
        return 200, b""

    out = probe_backend(Backend(kind=CLUSTER, url="http://node-a:8000"), CFG, now=1.0, get=get)
    assert out["live"] is True and out["model_roots"] == ("/models/Model-A-NVFP4",)
    models["data"][0].pop("root")
    out = probe_backend(Backend(kind=CLUSTER, url="http://node-a:8000"), CFG, now=1.0, get=get)
    assert out["model_roots"] == (), "an engine that names no root names no checkpoint"
