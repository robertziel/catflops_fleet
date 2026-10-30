"""S5 — the prober turns a backend URL into Backend fields, bounded by a timeout."""
from __future__ import annotations

import json
import time

import pytest

from tests.fakes import FakeBackend, free_port
from vllm_lb.config import Config
from vllm_lb.probe import probe_backend
from vllm_lb.state import VLLM, Backend

CFG = Config(probe_timeout_s=0.5)


def test_a_live_vllm_reports_ids_window_and_queue_pressure():
    f = FakeBackend(served=("qwen3.8:latest",), max_model_len=32768)
    f.waiting, f.running = 3, 2
    out = probe_backend(Backend(kind=VLLM, url=f.url), CFG, now=7.0)
    assert out["live"] is True and out["served_models"] == ("qwen3.8:latest",)
    assert out["max_model_len"] == 32768 and out["waiting"] == 3 and out["running"] == 2
    assert out["last_probe_at"] == 7.0 and out["last_error"] == ""


def test_a_vllm_that_does_not_answer_is_not_live():
    out = probe_backend(Backend(kind=VLLM, url=f"http://127.0.0.1:{free_port()}"), CFG, now=1.0)
    assert out["live"] is False and out["served_models"] == () and out["last_error"]


def test_a_hanging_backend_is_bounded_by_the_probe_timeout():
    """A server that takes the connection and never answers: the probe gives up at its timeout."""
    import socket
    silent = socket.socket()
    silent.bind(("127.0.0.1", 0))
    silent.listen(8)                     # accepts into the backlog, never answers a byte
    try:
        t0 = time.monotonic()
        out = probe_backend(Backend(kind=VLLM, url=f"http://127.0.0.1:{silent.getsockname()[1]}"), CFG, now=1.0)
        assert time.monotonic() - t0 < 2.0
        assert out["live"] is False and "timed out" in out["last_error"].lower()
    finally:
        silent.close()


def test_metrics_failure_leaves_a_live_vllm_live_with_zero_pressure():
    f = FakeBackend()
    f.waiting = 9
    # /metrics answers only on the fake's real path; a vLLM without metrics is still a vLLM
    out = probe_backend(Backend(kind=VLLM, url=f.url), CFG, now=1.0)
    assert out["live"] is True and out["waiting"] == 9


# ── the sample's time is when its body was READ (worklog/vllm-gateway-throughput.md, row 12) ──

MODELS = json.dumps({"data": [{"id": "qwen3.8:latest", "max_model_len": 4096}]}).encode()


def test_sample_is_stamped_after_the_body_is_read():
    """/metrics is the third of three sequential GETs, and on a loaded box the slow one. A rate's dt
    runs from the moment the body was read, never from the tick that started the probe."""
    slow_s = 0.3

    def get(url, timeout):
        if url.endswith("/health"):
            return 200, b""
        if url.endswith("/v1/models"):
            return 200, MODELS
        if url.endswith("/metrics"):
            time.sleep(slow_s)
            return 200, b'vllm:generation_tokens_total{engine="0",model_name="qwen3.8:latest"} 10.0\n'
        return 404, b""

    before = time.monotonic()
    out = probe_backend(Backend(kind=VLLM, url="http://box-a:8000"), CFG, now=7.0, get=get)
    assert out["mono_at"] >= before + slow_s
    assert out["scrape"]["gen_tokens_total"] == 10
    assert out["last_probe_at"] == 7.0, "the tick's wall time still says when it was probed"


def test_a_probe_that_found_no_server_is_stamped_too():
    out = probe_backend(Backend(kind=VLLM, url=f"http://127.0.0.1:{free_port()}"), CFG, now=1.0)
    assert out["live"] is False and isinstance(out["mono_at"], float)
    assert out["scrape"] is None


def test_metrics_without_the_queue_families_leave_depth_unknown():
    """Row 16: a scrape that does not expose the depth families cannot say the server is idle."""
    def get(url, timeout):
        if url.endswith("/v1/models"):
            return 200, MODELS
        if url.endswith("/metrics"):
            return 200, b'vllm:generation_tokens_total{engine="0",model_name="m"} 10.0\n'
        return 200, b""

    out = probe_backend(Backend(kind=VLLM, url="http://box-a:8000"), CFG, now=1.0, get=get)
    assert out["live"] is True and out["depth_known"] is False
    assert "depth" in out["last_error"]


# ── the KV pool, read off the engine rather than assumed (probe.SCRAPE_INFO_LABELS) ──────────────────

#: Captured verbatim from the head rank on 2026-10-01, generation 98. The figures are in the LABELS and
#: the sample's value is a constant 1.0 — the shape SCRAPE_FAMILIES cannot read, because there a label
#: selects WHICH sample to take and the value is the reading.
LIVE_CACHE_CONFIG = (
    'vllm:cache_config_info{block_size="8",cache_dtype="auto",enable_prefix_caching="True",engine="0",'
    'gpu_memory_utilization="0.7",kv_cache_max_concurrency="6.51595744680851",'
    'kv_cache_memory_bytes="28000000000",kv_cache_size_tokens="1708119",num_gpu_blocks="1225"} 1.0\n'
)


def test_the_kv_pool_is_read_off_the_engines_own_cache_config():
    """The pin is an operator knob now, so nothing may assume 28 GB. The engine publishes what it really
    allocated; this is where the LB picks it up."""
    from vllm_lb.probe import parse_vllm_metrics

    got = parse_vllm_metrics(LIVE_CACHE_CONFIG)
    assert got["kv_pool_tokens"] == 1_708_119
    assert got["kv_pool_bytes"] == 28_000_000_000
    assert got["kv_max_concurrency"] == pytest.approx(6.5159, abs=1e-4)


def test_an_engine_that_publishes_no_cache_config_reports_null_never_zero():
    """A pool of 0 would read as "no KV at all" and divide every derived figure by nothing. Absent is
    absent — the same rule every other family here follows."""
    from vllm_lb.probe import parse_vllm_metrics

    got = parse_vllm_metrics('vllm:num_requests_running{engine="0"} 1.0\n')
    assert got["kv_pool_tokens"] is None and got["kv_pool_bytes"] is None
    assert got["kv_max_concurrency"] is None


def test_a_garbled_pool_label_costs_its_own_key_and_nothing_else():
    """One unreadable label must not cost the scrape, nor its siblings — the discipline the counters keep."""
    from vllm_lb.probe import parse_vllm_metrics

    got = parse_vllm_metrics('vllm:cache_config_info{kv_cache_size_tokens="many",'
                             'kv_cache_memory_bytes="28000000000"} 1.0\n')
    assert got["kv_pool_tokens"] is None, "unreadable"
    assert got["kv_pool_bytes"] == 28_000_000_000, "its sibling survives"


def test_the_engines_block_count_and_block_size_are_read_off_the_same_cache_config():
    """The seats a pool holds are blocks over blocks-per-request (cluster_defaults.seats_for): the engine
    says how many blocks it really has, and — once resolved — how many tokens one holds."""
    from vllm_lb.probe import parse_vllm_metrics

    got = parse_vllm_metrics(LIVE_CACHE_CONFIG)
    assert got["kv_pool_blocks"] == 1225 and got["kv_block_tokens"] == 8
    dense = ('vllm:cache_config_info{block_size="816",kv_cache_max_concurrency="4.9676470588235295",'
             'kv_cache_memory_bytes="48000000000",kv_cache_size_tokens="1302238",num_gpu_blocks="1689"} 1.0\n')
    got = parse_vllm_metrics(dense)
    assert got["kv_pool_blocks"] == 1689 and got["kv_block_tokens"] == 816
    assert parse_vllm_metrics('vllm:num_requests_running{engine="0"} 1.0\n')["kv_pool_blocks"] is None
