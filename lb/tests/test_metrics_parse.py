"""vLLM's `/metrics`, read exactly: whole sample names, None for what a scrape did not say, and one
bad line costs one family — never the whole scrape.

worklog/vllm-gateway-throughput.md Q2 and edge rows 14, 15, 16 and 21. The exposition fixtures are
SYNTHETIC (no vLLM answered during pass 5) and say so in their headers; they are built from the
family names Q2 cites, and their depth and token values are the ones captured from box-a in
test_engine_stats.py.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from vllm_lb.probe import MODEL_INPUT_KEYS, POOL_INPUT_KEYS, SPEECH_KEYS, parse_vllm_metrics

FIXTURES = Path(__file__).parent / "fixtures" / "throughput"


def exposition(name: str) -> str:
    return (FIXTURES / name).read_text()


def test_a_fresh_idle_vllm_reads_zero_not_absent():
    """Every lease start is a fresh process whose counters start at 0 — a real, readable 0."""
    got = parse_vllm_metrics(exposition("vllm-v0.23.0-idle.metrics"))
    assert got["gen_tokens_total"] == 0 and got["prompt_tokens_total"] == 0
    assert got["waiting"] == 0 and got["running"] == 0
    assert got["prompt_computed_total"] == 0 and got["prompt_cached_total"] == 0
    assert got["kv_cache_usage"] == 0.0


def test_a_busy_classic_sidecar():
    got = parse_vllm_metrics(exposition("vllm-v0.23.0-busy.metrics"))
    assert got["waiting"] == 5, "waiting_by_reason repeats the same figure per reason"
    assert got["running"] == 2
    assert got["gen_tokens_total"] == 92438 and got["prompt_tokens_total"] == 95027
    assert got["prompt_computed_total"] == 48000, "only source=local_compute is prefill work"
    assert got["prompt_cached_total"] == 47027
    assert got["kv_cache_usage"] == pytest.approx(0.1234)
    assert got["prefill_computed_tokens_sum"] == 48000
    assert got["prefill_time_sum"] == pytest.approx(56.4)
    assert got["prefill_requests_total"] == 31
    assert got["process_start_time"] == pytest.approx(1789504123.45)


def test_iteration_tokens_histogram_is_not_a_counter():
    """Row 15. `vllm:iteration_tokens_total` is a HISTOGRAM whose name ends in `_total`; its `_sum`
    and every counter's `_created` gauge must never land in a counter."""
    text = exposition("vllm-v0.23.0-busy.metrics")
    assert "vllm:iteration_tokens_total_sum" in text and "vllm:generation_tokens_created" in text
    got = parse_vllm_metrics(text)
    assert got["gen_tokens_total"] == 92438 and got["prompt_tokens_total"] == 95027


def test_absent_family_is_none_not_zero():
    """Row 16. `--disable-log-stats`, or a renamed metric: what the scrape did not say is unknown,
    and the old parser's 0.0 default read it as idle."""
    got = parse_vllm_metrics('vllm:num_requests_running{engine="0",model_name="m"} 1.0\n'
                             'vllm:num_requests_waiting{engine="0",model_name="m"} 0.0\n')
    assert got["running"] == 1 and got["waiting"] == 0
    for key in ("gen_tokens_total", "prompt_tokens_total", "prompt_computed_total",
                "prompt_cached_total", "kv_cache_usage", "prefill_time_sum", "spec_drafts_total",
                "process_start_time"):
        assert got[key] is None, key
    assert all(value is None for value in parse_vllm_metrics("").values())


def test_nan_poisons_one_series_only():
    """Row 14. One NaN gauge used to fail the whole parse — depth included — at the final int()."""
    text = exposition("vllm-v0.23.0-busy.metrics")
    kv = 'vllm:kv_cache_usage_perc{engine="0",model_name="qwen3.8:latest"} '
    prompt = 'vllm:prompt_tokens_total{engine="0",model_name="qwen3.8:latest"} '
    text = text.replace(kv + "0.1234", kv + "NaN").replace(prompt + "95027.0", prompt + "+Inf")
    text += 'vllm:prompt_tokens_cached_total{engine="1",model_name="qwen3.8:latest"} garbage\n'
    text += "this line is not exposition at all\n"
    got = parse_vllm_metrics(text)
    assert got["kv_cache_usage"] is None
    assert got["prompt_tokens_total"] is None
    assert got["prompt_cached_total"] is None, "one bad engine line makes the family's sum unknown"
    assert got["waiting"] == 5 and got["running"] == 2 and got["gen_tokens_total"] == 92438


def test_a_negative_counter_is_not_a_count():
    got = parse_vllm_metrics('vllm:generation_tokens_total{engine="0",model_name="m"} -5.0\n')
    assert got["gen_tokens_total"] is None


def test_a_kv_gauge_outside_its_range_is_kept_raw():
    """The page shows `?` with the raw value; the parser does not clamp or invent."""
    got = parse_vllm_metrics('vllm:kv_cache_usage_perc{engine="0",model_name="m"} 1.7\n')
    assert got["kv_cache_usage"] == 1.7


def test_mtp_counters():
    """Row 21. On the cluster head, generation counts EMITTED tokens; drafts are counted apart, and
    the per-position counter is never added into the accepted total."""
    got = parse_vllm_metrics(exposition("vllm-v0.29.0-cluster-mtp.metrics"))
    assert got["gen_tokens_total"] == 58400
    assert got["spec_drafts_total"] == 12000
    assert got["spec_draft_tokens_total"] == 48000
    assert got["spec_accepted_total"] == 27840


def test_no_spec_counters_where_speculative_decoding_is_off():
    got = parse_vllm_metrics(exposition("vllm-v0.23.0-busy.metrics"))
    assert got["spec_drafts_total"] is None and got["spec_draft_tokens_total"] is None
    assert got["spec_accepted_total"] is None


def test_process_start_time_is_optional():
    text = "\n".join(line for line in exposition("vllm-v0.23.0-busy.metrics").splitlines()
                     if not line.startswith("process_start_time_seconds"))
    got = parse_vllm_metrics(text)
    assert got["process_start_time"] is None and got["gen_tokens_total"] == 92438


def test_label_sets_are_summed_and_only_the_named_source_is_computed():
    text = ('vllm:generation_tokens_total{engine="0",model_name="a"} 100.0\n'
            'vllm:generation_tokens_total{engine="1",model_name="a"} 50.0\n'
            'vllm:generation_tokens_total{engine="0",model_name="b"} 7.0\n'
            'vllm:num_requests_waiting{engine="0",model_name="a"} 2.0\n'
            'vllm:num_requests_waiting_by_reason{engine="0",model_name="a",reason="capacity"} 2.0\n'
            'vllm:prompt_tokens_by_source_total{engine="0",model_name="a",source="local_compute"} 30.0\n'
            'vllm:prompt_tokens_by_source_total{engine="0",model_name="a",source="local_cache_hit"} 70.0\n')
    got = parse_vllm_metrics(text)
    assert got["gen_tokens_total"] == 157 and got["waiting"] == 2
    assert got["prompt_computed_total"] == 30


def test_the_real_cluster_head_exposes_every_family_the_lb_reads():
    """CAPTURED, not built: the TP=2 cluster head (vLLM v0.29.0 plus six local patches, MTP), read by
    `capture.py --metrics` on 2026-09-15 while it served. Every family the parser reads is there under
    its exact name, next to the `_created` gauges it must ignore."""
    text = exposition("vllm-v0.29.0-cluster-head-1.metrics")
    assert "vllm:generation_tokens_created" in text and "vllm:iteration_tokens_total_sum" in text
    got = parse_vllm_metrics(text)
    # Every key but the three only SGLang feeds — the modeled bandwidth's inputs, null on vLLM by design — and the
    # five only a KV tier in the host's RAM feeds: this pair runs none (`kv_offloading_size="None"`) — and the speech
    # figures only a transcription service publishes — and the input of a pool reserved per seat (TensorFold's).
    sglang_only = {key for key, _ in (*MODEL_INPUT_KEYS, *POOL_INPUT_KEYS, *SPEECH_KEYS)}
    ram_tier = {"kv_offload_ram_gib", "kv_offload_load_bytes_total", "kv_offload_load_seconds_total",
                "kv_offload_store_bytes_total", "kv_offload_store_seconds_total"}
    assert [key for key, value in got.items() if value is None and key not in sglang_only | ram_tier] == []
    assert all(got[key] is None for key in sglang_only | ram_tier)
    assert got["running"] == 1 and got["waiting"] == 13
    assert got["gen_tokens_total"] == 23656
    assert got["spec_drafts_total"] == 7667 and got["spec_accepted_total"] == 16042


def test_label_values_can_hold_braces_spaces_quotes_and_inf():
    text = ('vllm:generation_tokens_total{engine="0",model_name="odd } name \\"quoted\\" +Inf"} 42.0\n'
            'vllm:request_prefill_time_seconds_bucket{engine="0",model_name="m",le="+Inf"} 3.0\n'
            'vllm:request_prefill_time_seconds_count{engine="0",model_name="m"} 3.0 1789500000000\n')
    got = parse_vllm_metrics(text)
    assert got["gen_tokens_total"] == 42
    assert got["prefill_requests_total"] == 3, "a sample may carry a timestamp after its value"
