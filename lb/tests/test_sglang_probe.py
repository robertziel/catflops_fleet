"""The probe reads an SGLang backend in the LB's own vocabulary.

Every figure asserted here was READ OFF a real scrape of the SGLang + DFlash2 TP=2 pair that served
`qwen38-27b-sglang-tp2` on the two GB10 boxes at 16 seats on 2026-10-02 (box-a:8020), captured before the
pair was torn down:

  * `sglang-metrics-2026-10-02-busy-a.txt` — 16 running, mid-generation
  * `sglang-metrics-2026-10-02-busy-b.txt` — the same pair 62 s later, so the COUNTERS have a real delta
  * `sglang-metrics-2026-10-02-idle.txt`   — the same engine with nothing running

Every prefix-cache mode in those three is 0 (that benchmark sent distinct prompts), which is exactly
the blind spot an UNVERIFIED assumption about `mode="input"` survived in, so two more were taken off
the SERVING head of generation 123 (`qwen3.8-27b-sglang-dflash2`, 192.0.2.2:8100) 83 s apart on
2026-10-03 while real traffic ran through the LB and the cache was hitting at 95-99 %:

  * `sglang-metrics-2026-10-03-cached-a.txt` / `-cached-b.txt` — a real delta on all three
    `realtime_tokens_total` modes, including the `prefill_cache` mode the 2026-10-02 pair never
    exposed at all

A hand-written imitation of that exposition would only prove the adapter reads its author's fiction,
so these tests parse the captured bytes themselves.
"""
from __future__ import annotations

import pathlib

import pytest

from vllm_lb.config import Config
from vllm_lb.rates import NOT_EXPOSED
from vllm_lb.probe import (ITL_PER_STEP, ITL_PER_TOKEN, SCRAPE_KEYS, SGLANG_BEHIND_A_FLAG,
                           SGLANG_FAMILIES, SGLANG_UNAVAILABLE, _HIT_MODES, _PROMPT_MODES,
                           parse_metrics, parse_sglang_metrics, parse_vllm_metrics, probe_backend)
from vllm_lb.state import CLUSTER, Backend
from vllm_lb.throughput import Throughput

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
BUSY_A = (FIXTURES / "sglang-metrics-2026-10-02-busy-a.txt").read_text()
BUSY_B = (FIXTURES / "sglang-metrics-2026-10-02-busy-b.txt").read_text()
IDLE = (FIXTURES / "sglang-metrics-2026-10-02-idle.txt").read_text()

#: What that pair's `/v1/models` answered, captured in the same minute as the metrics above.
MODELS_BODY = (b'{"object":"list","data":[{"id":"qwen38-27b-sglang-tp2","object":"model",'
               b'"created":1790977509,"owned_by":"sglang","root":"qwen38-27b-sglang-tp2",'
               b'"parent":null,"max_model_len":262144}]}')

CFG = Config(probe_timeout_s=0.5)


def _sglang_getter(metrics: str, *, health: int = 200):
    """A `Getter` that answers exactly what that SGLang pair answered, from the captured bytes."""
    def get(url: str, timeout: float):
        if url.endswith("/health"):
            return health, b""
        if url.endswith("/v1/models"):
            return 200, MODELS_BODY
        if url.endswith("/metrics"):
            return 200, metrics.encode()
        return 404, b""
    return get


# ───────────────────────── the defect, in the probe's own terms ─────────────────────────

def test_the_vllm_only_parser_finds_nothing_in_a_real_sglang_scrape():
    """WHY THE ADAPTER EXISTS. The LB's canonical reader is `vllm:*`-only, so a real SGLang scrape
    gives it not one figure — every key None. Everything downstream (the page, the saturation logic,
    the CUDA CORES tile, the performance log) therefore draws nothing for an SGLang generation."""
    out = parse_vllm_metrics(BUSY_B)
    assert set(out) == {key for key, _ in SCRAPE_KEYS}
    assert all(value is None for value in out.values()), \
        "the vLLM parser must not invent a figure from SGLang's vocabulary"


def test_the_probe_publishes_the_canonical_figures_for_a_real_sglang_scrape():
    """THE HEADLINE: the real captured SGLang sample in, the LB's own figures out — through
    `probe_backend`, the same path the cluster backend is probed by every tick."""
    out = probe_backend(Backend(kind=CLUSTER, url="http://sglang.invalid"), CFG,
                        now=11.0, get=_sglang_getter(BUSY_B))
    assert out["live"] is True
    assert out["served_models"] == ("qwen38-27b-sglang-tp2",)
    assert out["max_model_len"] == 262144
    # The queue depth the router and the saturation logic read. 0 waiting is a READING, not a gap.
    assert out["running"] == 16 and out["waiting"] == 0 and out["depth_known"] is True
    assert out["last_error"] == ""
    assert out["scrape"]["gen_tokens_total"] == 53343
    assert out["gen_tokens_total"] == 53343     # the Backend alias the page still reads


# ───────────────────────── the mapping, figure by figure ─────────────────────────

def test_scheduler_depth():
    busy, idle = parse_metrics(BUSY_B), parse_metrics(IDLE)
    assert (busy["running"], busy["waiting"]) == (16, 0)     # sglang:num_running_reqs / num_queue_reqs
    assert (idle["running"], idle["waiting"]) == (0, 0)      # an idle engine READS zero


def test_generation_and_prompt_tokens_come_from_the_live_per_step_counters():
    """`sglang:generation_tokens_total` is incremented when a REQUEST FINISHES: across busy-a and
    busy-b — 62 s in which this engine emitted 19 138 tokens — it does not move at all, because the
    16 in-flight requests had not returned. `sglang:realtime_tokens_total` is the scheduler's own
    per-step counter and does. Throughput must read the one that moves while work is happening."""
    a, b = parse_metrics(BUSY_A), parse_metrics(BUSY_B)
    assert (a["gen_tokens_total"], b["gen_tokens_total"]) == (34205, 53343)
    assert b["gen_tokens_total"] - a["gen_tokens_total"] == 19138
    assert (a["prompt_tokens_total"], b["prompt_tokens_total"]) == (5489, 5489)
    assert b["prompt_computed_total"] == 5489


def test_the_token_rate_agrees_with_the_engines_own_throughput_gauge():
    """THE MAPPING'S OWN ORACLE. SGLang also publishes `sglang:gen_throughput`, its own decode rate —
    a gauge the LB does not read, which makes it an independent check on the counter the LB DOES read.

    The two captures are 62 s apart and differ by 19 138 tokens on `realtime_tokens_total`, so the
    rate taken the way throughput.py takes it is 308.7 tok/s; the engine said 316.6 at busy-a and
    306.3 at busy-b. The LB's figure lands between the engine's own two readings, which it could not
    do if this were the wrong counter."""
    a, b = parse_metrics(BUSY_A), parse_metrics(BUSY_B)
    rate = (b["gen_tokens_total"] - a["gen_tokens_total"]) / 62.0
    assert rate == pytest.approx(308.7, abs=0.1)
    said_a = float(BUSY_A.split("sglang:gen_throughput{")[1].split("} ")[1].split("\n")[0])
    said_b = float(BUSY_B.split("sglang:gen_throughput{")[1].split("} ")[1].split("\n")[0])
    assert (said_b, said_a) == pytest.approx((306.29523481203245, 316.6429991809716))
    assert min(said_a, said_b) * 0.95 <= rate <= max(said_a, said_b) * 1.05


def test_kv_usage_is_the_kv_pool_not_the_draft_pool():
    """`sglang:token_usage` is the WORST of the engine's pools, and DFlash2's draft state sits in a
    tiny mamba pool that reads 0.8 while the KV cache is 1.3 % full. Reading it would put the KV tile
    at 80 % on an almost empty cache. `sglang:full_token_usage` is the KV pool's own fraction, and in
    this capture it equals kv_used_tokens / max_total_num_tokens exactly."""
    assert parse_metrics(BUSY_B)["kv_cache_usage"] == pytest.approx(0.013131126263854879)
    assert 45072 / 3432455 == pytest.approx(0.013131126263854879)   # the identity it satisfies
    assert "0.8" in BUSY_B.split("sglang:token_usage{")[1].split("\n")[0]   # what we did NOT read
    assert parse_metrics(IDLE)["kv_cache_usage"] == 0.0


def test_engine_steps_and_their_tokens_feed_the_cuda_cores_tile():
    """`probe.py` SCRAPE_FAMILIES' `steps_total` / `step_tokens_sum` are what cores.py splits the
    GPU's time with. SGLang counts one forward pass per scheduler iteration, labelled by whether a
    CUDA graph was used, and its tokens in the same per-step counter as generation.

    The delta proves the pairing is coherent: 477 passes carried 19 138 decode tokens over 16 running
    sequences = 2.51 tokens per sequence per pass, which is this engine's own `spec_accept_length`
    (2.56 at busy-b) — so one pass is one verify step, not one draft forward."""
    a, b = parse_metrics(BUSY_A), parse_metrics(BUSY_B)
    assert (a["steps_total"], b["steps_total"]) == (975, 1452)
    assert (a["step_tokens_sum"], b["step_tokens_sum"]) == (39694, 58832)
    steps, tokens = b["steps_total"] - a["steps_total"], b["step_tokens_sum"] - a["step_tokens_sum"]
    assert (steps, tokens) == (477, 19138)
    assert tokens / steps / 16 == pytest.approx(2.51, abs=0.02)


def test_latency_histograms_carry_their_sum_count_and_buckets():
    b = parse_metrics(BUSY_B)
    assert b["ttft_count"] == 41 and b["ttft_sum"] == pytest.approx(39.91641765400004)
    assert b["itl_count"] == 53245 and b["itl_sum"] == pytest.approx(2756.847991787013)
    # Buckets ascending, `+Inf` last, and its cumulative count is the histogram's own count.
    for key, count in (("ttft_buckets", 41), ("itl_buckets", 53245)):
        edges = b[key]
        assert edges is not None and len(edges) > 5
        assert [le for le, _ in edges] == sorted(le for le, _ in edges)
        assert edges[-1][0] == float("inf") and edges[-1][1] == count


def test_the_kv_pool_the_engine_really_allocated():
    """vLLM publishes the pool in `vllm:cache_config_info`'s labels; SGLang publishes it as gauges.
    `kv_cache_memory_usage_gb` is GiB: 52.3751220703125 GiB over 3 432 455 tokens is 16 384.00
    bytes a token — exactly 2**14 — which a 10**9 reading would not give."""
    b = parse_metrics(BUSY_B)
    assert b["kv_pool_tokens"] == 3432455
    assert b["kv_pool_blocks"] == 3432455 and b["kv_block_tokens"] == 1
    assert b["kv_pool_bytes"] == 56237359104
    assert b["kv_pool_bytes"] / b["kv_pool_tokens"] == pytest.approx(16384, abs=0.01)


def test_prefill_speed_and_finished_requests():
    b = parse_metrics(BUSY_B)
    assert b["prefill_time_sum"] == pytest.approx(44.00701040500212)   # stage="prefill_forward"
    assert b["prefill_requests_total"] == 43
    assert b["prefill_computed_tokens_sum"] == 5489
    assert b["requests_finished_total"] == 25        # e2e_request_latency_seconds_count, both streams


def test_speculative_decoding_is_read_off_the_completion_side_counters():
    """SGLang's own arithmetic, confirmed to the last digit in all three captures:

        spec_accept_rate == (spec_accept_length - 1) / (spec_num_draft_tokens - 1)

    so a verify step proposes `num_draft_tokens - 1` = 3 DRAFT tokens and always emits 1 bonus token.
    `spec_verify_calls_total` counts verify steps and `sglang:generation_tokens_total` the tokens they
    produced — both per FINISHED REQUEST, so the triple is internally consistent."""
    b = parse_metrics(BUSY_B)
    assert b["spec_drafts_total"] == 3730                          # sglang:spec_verify_calls_total
    assert b["spec_draft_tokens_total"] == 11190                   # 3730 drafts x 3 draft tokens
    assert b["spec_accepted_total"] == 5878                         # 9608 tokens - 3730 bonus tokens
    # What the consumers compute from it (throughput.py `_spec_decode`).
    assert b["spec_accepted_total"] / b["spec_draft_tokens_total"] == pytest.approx(0.525, abs=0.001)
    assert 1 + b["spec_accepted_total"] / b["spec_drafts_total"] == pytest.approx(2.576, abs=0.001)


def test_the_list_of_honest_gaps_is_exactly_what_comes_back_null():
    """THE RULE, not one instance of it: between them, `SGLANG_UNAVAILABLE` and `SGLANG_BEHIND_A_FLAG`
    document every canonical figure this capture does not carry, and together they are EXACTLY the set
    that really comes back null off the real scrape. A mapping added without declaring its key in one
    of the two — or a key struck from both without a mapping — fails here, so neither list can drift
    into a lie about what the page will draw.

    They are not interchangeable, and that is the point of there being two. `SGLANG_UNAVAILABLE` says
    NO SOURCE EXISTS: the page can never show this figure for an SGLang generation, whatever the engine
    was started with. `SGLANG_BEHIND_A_FLAG` says a source exists and this capture was taken without
    the switch that publishes it. Until the pinned image was read, the MFU estimators sat in the first
    list claiming the first thing while the second was true."""
    scrape = parse_metrics(BUSY_B)
    declared = {**SGLANG_UNAVAILABLE, **SGLANG_BEHIND_A_FLAG}
    assert sorted(k for k, v in scrape.items() if v is None) == sorted(declared)
    assert not set(SGLANG_UNAVAILABLE) & set(SGLANG_BEHIND_A_FLAG), \
        "a gap is one kind or the other: an engine that cannot, or a start that did not ask"
    assert set(declared) <= {key for key, _ in SCRAPE_KEYS}
    assert all(reason for reason in declared.values()), "each gap states its reason"
    assert all(flag.startswith("--") for flag in SGLANG_BEHIND_A_FLAG.values()), \
        "a flag-gated gap names the switch that ends it, so the fix is in the list"


@pytest.mark.parametrize("key", ["process_start_time", "preemptions_total", "kv_max_concurrency"])
def test_what_sglang_does_not_publish_at_all_is_null_never_zero(key):
    """THE BUG CLASS THIS AVOIDS: worklog/gateway-figures-audit.md finding 14 — a figure published as
    a fabricated 0.0 rather than null, which the page then draws as a real reading.

    * `process_start_time` is the restart fingerprint — SGLang exposes no epoch gauge, only
      per-phase startup DURATIONS, so an SGLang restart cannot be detected from its metrics
    * `preemptions_total` — `sglang:num_retracted_reqs` is a GAUGE of what is retracted right now,
      not a cumulative counter, so a delta of it would be meaningless
    * `kv_max_concurrency` is only in `/get_server_info`, not in `/metrics`
    """
    assert key in SGLANG_UNAVAILABLE
    assert parse_metrics(BUSY_B)[key] is None
    assert parse_metrics(IDLE)[key] is None


@pytest.mark.parametrize("key", ["est_flops_total", "est_read_bytes_total", "est_write_bytes_total"])
def test_the_mfu_estimators_are_null_because_the_switch_was_OFF_not_because_sglang_lacks_them(key):
    """THE DEFECT THE OPERATOR ASKED ABOUT: the COMPUTE and MEMORY BANDWIDTH tiles read "not exposed by
    this server" on an SGLang generation. The adapter's note said SGLang had no counterpart to vLLM's
    MFU estimator at all. The pinned image refutes that — it publishes the same three counters under
    its own prefix (`SGLANG_FAMILIES`) — so the tiles were blank for a DIFFERENT reason: the engine was
    never asked for them.

    The capture's own evidence that this is the reason, not absence: the exposition carries no
    `estimated_` family at all (not a family at 0, as vLLM's would be), and the start that the LB made
    on this engine logged SGLang's own `server_args` with the switch off.

    Null, never 0, either way: a 0 TFLOP/s on a pair serving 16 requests would be the same invention
    as a 0 GB VRAM tile."""
    assert key in SGLANG_BEHIND_A_FLAG and key not in SGLANG_UNAVAILABLE
    assert SGLANG_BEHIND_A_FLAG[key] == "--enable-mfu-metrics"
    assert parse_metrics(BUSY_B)[key] is None
    assert parse_metrics(IDLE)[key] is None
    # The families are ABSENT from the scrape, which is why null is the honest reading.
    assert "estimated_" not in BUSY_B and "estimated_" not in IDLE
    # SGLang's own server_args, as the LB's own SGLang start logged them.
    started = (FIXTURES / "sglang-lb-start-2026-10-03-rank0.log").read_text()
    assert "'enable_mfu_metrics': False" in started, "the switch the captures were taken without"


# ──────────── the two tiles, once the switch is on — UNVERIFIED AGAINST A LIVE SCRAPE ────────────

#: SGLANG'S ESTIMATOR COUNTERS AS A HEAD WOULD RENDER THEM. **DERIVED, NOT CAPTURED**: no scrape taken
#: with `--enable-mfu-metrics` on exists yet, because the switch only takes effect on the next start and
#: no pair has been started with it. So of this exposition:
#:
#:   * the three NAMES are the pinned image's own, read off its source (cited at `SGLANG_FAMILIES`);
#:   * the LABEL SET is lifted verbatim from a real sample in this very capture — and the image's own
#:     `SchedulerMetricsCollector` labels these three with exactly that set (model_name, engine_type,
#:     tp_rank, pp_rank, moe_ep_rank), so the rank rule below is exercised against the real labels;
#:   * the VALUES are chosen. What these tests prove is the PLUMBING and the UNITS — that a scrape
#:     carrying these names fills the two tiles instead of reading "not exposed" — never a reading.
#:
#: What would replace this with a capture: start an SGLang pair from the LB, then on the head's box
#:   curl -s localhost:8100/metrics | grep -c 'sglang:estimated_.*_per_gpu_total'   # expect 3
def _with_mfu(text: str, flops: float, read_bytes: float, written: float) -> str:
    labels = text.split("sglang:num_running_reqs{")[1].split("}")[0]
    return text + "".join(
        f"sglang:estimated_{what}_per_gpu_total{{{labels}}} {value}\n"
        for what, value in (("flops", flops), ("read_bytes", read_bytes), ("write_bytes", written)))


def test_the_estimator_counters_land_on_the_canonical_keys_the_two_tiles_read():
    """The mapping itself: SGLang's three names onto the LB's own `est_*` keys, in the engine's units.
    Raw FLOPs and raw BYTES, already per GPU — the same quantities vLLM's identically-named counters
    carry, which is why they need no conversion on the way to `throughput.py`'s own divisors."""
    out = parse_metrics(_with_mfu(BUSY_B, 1.0e14, 2.0e12, 1.0e11))
    assert out["est_flops_total"] == 100_000_000_000_000
    # THE TWO BYTE COUNTERS ARE WITHHELD ON PURPOSE, and this asserts the decision rather than the
    # mapping. SGLang charges the whole per-rank weight read for every generated token at bf16 itemsize,
    # so at the pair's own 278.69 tok/s its read counter works out to 10,847 GB/s a GPU against the
    # 273 GB/s the hardware has — 39.7x a hard physical limit, which would drive the saturation verdict
    # to BANDWIDTH_BOUND off a number that cannot be true. The page says "not exposed by this server"
    # instead, which is the truth about what this gateway is willing to read, not about what SGLang emits.
    assert out["est_read_bytes_total"] is None and out["est_write_bytes_total"] is None
    # And nothing else in the scrape moved: the one line is additive, not a reinterpretation.
    before = parse_metrics(BUSY_B)
    changed = {k for k in before if before[k] != out[k]}
    assert changed == {"est_flops_total"}


def test_an_estimator_counter_is_read_once_per_pair_like_every_other_rank_labelled_family():
    """THE TP=2 HAZARD again, on the one family the tiles depend on: these are PER-GPU figures, so a
    head exposing both ranks must not add them into a per-GPU figure twice the truth. SGLang really does
    label them `tp_rank` (the image's own collector gives them the scheduler label set), and in practice
    only the head rank is scrapeable at all — so this is the same safety net the other families have."""
    one = _with_mfu(BUSY_B, 1.0e14, 2.0e12, 1.0e11)
    both = one + "".join(line.replace('tp_rank="0"', 'tp_rank="1"') + "\n"
                         for line in one.splitlines() if "sglang:estimated_" in line)
    assert parse_metrics(both)["est_flops_total"] == 100_000_000_000_000
    assert parse_metrics(both) == parse_metrics(one)


def test_with_the_switch_off_the_two_tiles_say_not_exposed_rather_than_a_fabricated_zero():
    """WHAT THE OPERATOR SAW, reproduced from the real capture and through the real consumer. Absent is
    not "registered and still 0": on vLLM a flat 0 counter in a process that produced tokens proves the
    switch is off (`MFU_NOT_MEASURED`), and SGLang cannot say that because it registers nothing — so
    the honest verdict is NOT_EXPOSED and the evidence about the switch stays unclaimed."""
    t = Throughput(CLUSTER, interval_s=5.0)
    for at in (0.0, 5.0):
        t.observe(live=True, scrape=parse_metrics(BUSY_B), mono_at=at, at=at)
    perf = t.engine_perf()
    assert perf["tflops_per_gpu"] is None and perf["flops_why"] == NOT_EXPOSED
    assert perf["gbps_per_gpu"] is None and perf["bytes_why"] == NOT_EXPOSED
    assert perf["mfu_metrics"] is None, "an absent family proves nothing about the switch"


def test_the_compute_and_bandwidth_tiles_draw_a_number_once_the_switch_is_on():
    """THE HEADLINE OF THE FIX, end to end: the captured SGLang scrape plus the counters in, TFLOP/s
    per GPU out — through `Throughput.engine_perf`, the one caller the tiles read. The bandwidth half is
    withheld on purpose and this pins that too, so restoring it has to be a deliberate edit here.

    The divisors are the engine's own arithmetic, as the vLLM path already pins: `rate(flops)/1e12` is
    TFLOP/s and `(rate(read) + rate(write))/1e9` GB/s. 60 TFLOP over 5 s is 12 TFLOP/s; 1.1 TB over
    5 s is 220 GB/s."""
    t = Throughput(CLUSTER, interval_s=5.0)
    for at, flops, read_bytes, written in ((0.0, 0.0, 0.0, 0.0), (5.0, 6e13, 1.0e12, 1.0e11)):
        t.observe(live=True, scrape=parse_metrics(_with_mfu(BUSY_B, flops, read_bytes, written)),
                  mono_at=at, at=at)
    perf = t.engine_perf()
    assert perf["tflops_per_gpu"] == pytest.approx(12.0) and perf["flops_why"] is None
    # COMPUTE draws; BANDWIDTH does not, and says why rather than drawing SGLang's own byte estimate —
    # which charges a whole per-rank weight read per generated token and lands 39.7x over the 273 GB/s
    # this hardware has. A tile that reads "not exposed by this server" is the honest end of a figure
    # nobody can act on; one reading 10,847 GB/s would send the saturation verdict to BANDWIDTH_BOUND.
    assert perf["gbps_per_gpu"] is None and perf["bytes_why"]
    assert perf["window_s"] == pytest.approx(5.0)
    assert perf["mfu_metrics"] == "on", "a counter that left 0 is the switch proving itself"


# ───────────────────────── which vocabulary, and the per-rank hazard ─────────────────────────

def test_the_scrape_text_declares_its_own_vocabulary():
    """The probe does not need to be told which engine it reached: the exposition names it. A real
    SGLang scrape carries `sglang:` families and not one `vllm:` sample."""
    assert "vllm" not in BUSY_B
    assert parse_metrics(BUSY_B) == parse_sglang_metrics(BUSY_B)
    vllm_text = ("vllm:num_requests_running{model_name=\"q\"} 4.0\n"
                 "vllm:num_requests_waiting{model_name=\"q\"} 1.0\n")
    assert parse_metrics(vllm_text) == parse_vllm_metrics(vllm_text)
    assert parse_metrics(vllm_text)["running"] == 4


def test_an_unreadable_scrape_stays_vllm_shaped_and_empty():
    for text in ("", "# nothing but a comment\n", "garbage\n"):
        out = parse_metrics(text)
        assert set(out) == {key for key, _ in SCRAPE_KEYS}
        assert all(value is None for value in out.values())


def test_both_parsers_publish_exactly_the_same_keys():
    """One shape, so no consumer can tell which engine it is reading from the keys it gets."""
    assert set(parse_sglang_metrics(BUSY_B)) == set(parse_vllm_metrics(BUSY_B))


def test_a_per_rank_series_is_read_once_not_summed_over_the_pair():
    """THE TP=2 HAZARD. SGLang tags its scheduler families with `tp_rank`/`pp_rank`/`moe_ep_rank`,
    and each rank publishes THE SAME figure. The captured pair's HTTP head exposes rank 0 only, so
    summing happened to be harmless there — but a head that exposed both ranks would have reported
    32 running on a 16-seat engine and double every token counter.

    So the adapter keeps the LOWEST rank it sees and sums only across the labels that really do
    partition the work. Here the rank-1 lines are the rank-0 lines relabelled."""
    assert 'tp_rank="1"' not in BUSY_B              # what the real capture contains
    doubled = BUSY_B + "".join(line.replace('tp_rank="0"', 'tp_rank="1"') + "\n"
                               for line in BUSY_B.splitlines()
                               if 'tp_rank="0"' in line and not line.startswith("#"))
    assert 'tp_rank="1"' in doubled
    both, one = parse_metrics(doubled), parse_metrics(BUSY_B)
    assert both["running"] == 16, "a two-rank head must not read 32 seats on a 16-seat engine"
    assert both["gen_tokens_total"] == one["gen_tokens_total"] == 53343
    assert both == one


def test_labels_that_really_partition_the_work_are_summed():
    """The other half of the rule: `tp_rank` replicates, `is_streaming` and `mode` partition.
    `ttft_count` is 41 = 40 streaming + 1 non-streaming, and `steps_total` 1452 = 1443 decode passes
    that used a CUDA graph + 9 prefill passes that did not."""
    b = parse_metrics(BUSY_B)
    assert b["ttft_count"] == 40 + 1
    assert b["steps_total"] == 1443 + 9


# ───────────── the prefix cache, on a pair that was really hitting it (C1, C2, C9) ─────────────
#
# The three 2026-10-02 captures above have EVERY prefix-cache mode at 0 — the benchmark that produced
# them sent distinct prompts — so they could not settle what `mode="input"` counts, and the adapter
# carried an UNVERIFIED assumption that it was the billed prompt, hits included. The two captures
# below are the same adapter's blind spot filled in: read off the serving head's own `/metrics`
# (http://192.0.2.2:8100, generation 123, `qwen3.8-27b-sglang-dflash2`, TP=2, 16 seats) 83 s apart
# on 2026-10-03 while a swarm session drove real traffic through the LB, so the prefix cache was at
# 95-99 % by the engine's own `sglang:cache_hit_rate` and all three token counters have a real delta.

CACHED_A = (FIXTURES / "sglang-metrics-2026-10-03-cached-a.txt").read_text()
CACHED_B = (FIXTURES / "sglang-metrics-2026-10-03-cached-b.txt").read_text()

#: What a real vLLM TP=2 head published, for the one question that needs both vocabularies at once
#: (its ITL convention). The same two reads test_engine_stats.py replays.
VLLM_HEAD = (FIXTURES / "throughput" / "vllm-v0.29.0-cluster-head-1.metrics").read_text()
VLLM_HEAD_2 = (FIXTURES / "throughput" / "vllm-v0.29.0-cluster-head-2.metrics").read_text()


def _series(text: str, name: str, **labels) -> float:
    """One exposed series' value, straight out of the capture's bytes — so a figure asserted about
    the ENGINE is read from the engine's own text and not through the adapter under test."""
    for line in text.splitlines():
        if line.startswith(name + "{") and all(f'{k}="{v}"' in line for k, v in labels.items()):
            return float(line.rsplit("} ", 1)[1])
    raise AssertionError(f"{name}{labels} is not in this capture")


@pytest.mark.parametrize("text", [CACHED_A, CACHED_B], ids=["cached-a", "cached-b"])
def test_mode_input_is_the_computed_prompt_not_the_billed_one(text):
    """THE MEASUREMENT THAT SETTLES C1, from the engine's own two families rather than from the
    adapter: `prefill_effective_tokens_total` partitions the prompt the same way
    `realtime_tokens_total` partitions the work, mode for mode and to the last token.

        prefill_effective{mode="input"}       ==  realtime{mode="prefill_compute"}
        prefill_effective{mode="device_hit"}  ==  realtime{mode="prefill_cache"}

    So `input` EXCLUDES the hits: it is what the layers computed. Reading it as the billed prompt —
    which is what the adapter did while this was UNVERIFIED — makes the prefix hit rate hits over
    COMPUTED, a ratio with no ceiling, and leaves the billed-minus-hits computed figure at 0."""
    assert _series(text, "sglang:prefill_effective_tokens_total", mode="input") \
        == _series(text, "sglang:realtime_tokens_total", mode="prefill_compute")
    assert _series(text, "sglang:prefill_effective_tokens_total", mode="device_hit") \
        == _series(text, "sglang:realtime_tokens_total", mode="prefill_cache")
    # The hits are the bulk of the prompt here, which is why the error is worth a factor of ten and
    # not a rounding: 7.48M reused against 0.80M computed.
    assert _series(text, "sglang:prefill_effective_tokens_total", mode="device_hit") \
        > 8 * _series(text, "sglang:prefill_effective_tokens_total", mode="input")
    # And the third `realtime` mode is why C2 is a separate defect: the 2026-10-02 captures have two.
    assert text.count("sglang:realtime_tokens_total{") == 3
    assert BUSY_B.count("sglang:realtime_tokens_total{") == 2
    # THE ENGINE SAYS SO IN WORDS TOO, in the family's own HELP line — which names the denominator
    # this adapter now uses, and which was in every capture all along:
    assert ("Windowed prefix cache hit rate = rate(sum of *_hit) / rate(sum of all modes)") in text
    assert "mode: prefill_compute, prefill_cache, decode." in text    # the three, named by the engine


def test_the_prompt_tokens_split_into_what_was_computed_and_what_the_cache_served():
    """Each canonical key off the mode it is really built from: billed = everything the requests
    brought (all four modes), computed = what a layer processed (`input`), cached/hits = what the
    cache served (the three hit modes). The three then satisfy the identity the page and the
    performance log both assume — `computed + cached == billed` — which they could not while billed
    was `input` alone."""
    a, b = parse_metrics(CACHED_A), parse_metrics(CACHED_B)
    assert (a["prompt_computed_total"], b["prompt_computed_total"]) == (801500, 815282)
    assert (a["prompt_cached_total"], b["prompt_cached_total"]) == (7481424, 7701912)
    assert (a["prompt_tokens_total"], b["prompt_tokens_total"]) == (8282924, 8517194)
    for s in (a, b):
        assert s["prompt_computed_total"] + s["prompt_cached_total"] == s["prompt_tokens_total"]
        # The prefix-cache pair is the same split under the names perflog.py reads.
        assert s["prefix_queries_total"] == s["prompt_tokens_total"]
        assert s["prefix_hits_total"] == s["prompt_cached_total"]


def test_the_prefix_hit_rate_is_a_share_of_the_prompt_and_not_a_multiple_of_it():
    """THE FIGURE THE OPERATOR READS. `perflog.py` records `prefix_hit_rate = hits / queries` per
    window and /benchmarks draws it as a percent against a 100 % axis; `throughput.py` divides the
    cached rate by the prompt rate for the card's `cached` field.

    Over the 83 s between these two captures the old mapping gives 220 488 / 13 782 = 15.998 — a
    1 599.8 % "hit rate", and 933.4 % cumulative — because the denominator was the COMPUTED prompt.
    Against the billed prompt it is 94.1 %, which is a share."""
    a, b = parse_metrics(CACHED_A), parse_metrics(CACHED_B)
    hits = b["prefix_hits_total"] - a["prefix_hits_total"]
    queries = b["prefix_queries_total"] - a["prefix_queries_total"]
    assert (hits, queries) == (220488, 234270)
    assert hits / queries == pytest.approx(0.94117, abs=0.00001)
    assert a["prefix_hits_total"] / a["prefix_queries_total"] == pytest.approx(0.90323, abs=0.00001)
    # THE ENGINE'S OWN ORACLE: `sglang:cache_hit_rate`, a gauge the adapter does not read, so it is
    # an independent reading of the same quantity. It says 98.9 % and 95.4 % — a share, like ours,
    # and nowhere near 16x.
    said = [_series(t, "sglang:cache_hit_rate") for t in (CACHED_A, CACHED_B)]
    assert said == pytest.approx([0.988512811634349, 0.9538256940601174])
    assert hits / queries == pytest.approx(said[1], abs=0.02)


def test_what_perflog_derives_from_the_billed_prompt_is_no_longer_a_hard_zero():
    """WHY 31 OF 50 RECORDED SGLANG WINDOWS SHOW `prompt_computed_tokens` 0. `perflog.py:574` takes
    `computed = max(0, billed - hits)`; with billed read as the computed prompt and hits larger than
    it, that is 0 every time a prefix cache is working — so /benchmarks' `computed (billed minus
    prefix-cache hits)` series sits pinned at 0 under a billed line running 400-2 200 tok/s.

    With billed = everything the requests brought, that subtraction lands exactly on the counter the
    engine publishes for the same thing, which is the check that the two readings now agree."""
    a, b = parse_metrics(CACHED_A), parse_metrics(CACHED_B)
    billed = b["prompt_tokens_total"] - a["prompt_tokens_total"]
    hits = b["prefix_hits_total"] - a["prefix_hits_total"]
    computed = b["prompt_computed_total"] - a["prompt_computed_total"]
    assert max(0, billed - hits) == computed == 13782
    assert max(0, billed - hits) > 0


@pytest.mark.parametrize("text", [CACHED_A, CACHED_B, BUSY_A, BUSY_B, IDLE],
                         ids=["cached-a", "cached-b", "busy-a", "busy-b", "idle"])
def test_a_prefix_hit_rate_outside_0_1_is_impossible_whatever_the_capture(text):
    """THE INVARIANT THAT WOULD HAVE CAUGHT C1 ON ITS FIRST DAY, and the rule rather than this
    instance of it: a cache cannot serve more tokens than were asked of it. It holds on every SGLang
    capture in the repo — the ones whose hit modes are 0 as much as the ones whose cache is at 94 % —
    and it is STRUCTURAL, not lucky: `prefix_hits_total` reads `_HIT_MODES`, its denominator reads
    `_PROMPT_MODES`, and the first is a subset of the second."""
    s = parse_metrics(text)
    assert _HIT_MODES < _PROMPT_MODES, "the hits must be part of what was queried, not beside it"
    assert 0 <= s["prefix_hits_total"] <= s["prefix_queries_total"]
    assert 0 <= s["prefix_hits_total"] / s["prefix_queries_total"] <= 1
    assert 0 <= s["prompt_cached_total"] <= s["prompt_tokens_total"]
    assert 0 <= s["prompt_computed_total"] <= s["prompt_tokens_total"]


def test_the_mapping_itself_cannot_put_a_hit_outside_the_queries_it_answered():
    """The same invariant read off `SGLANG_FAMILIES` instead of off a capture, so a future edit that
    narrows the denominator again fails here even with no sample that exposes it."""
    family = dict((key, want) for key, want, _ in
                  SGLANG_FAMILIES["sglang:prefill_effective_tokens_total"])
    assert family["prefix_hits_total"] == family["prompt_cached_total"] == ("mode", _HIT_MODES)
    assert family["prefix_queries_total"] == family["prompt_tokens_total"] == ("mode", _PROMPT_MODES)
    assert _HIT_MODES < _PROMPT_MODES
    # Every mode either of the two captures exposes is accounted for by one of the two sets, so the
    # billed prompt is the whole family and not a part of it that happens to be the part we know.
    exposed = {line.split('mode="')[1].split('"')[0] for line in CACHED_B.splitlines()
               if line.startswith("sglang:prefill_effective_tokens_total{")}
    assert exposed == _PROMPT_MODES


def test_the_cuda_cores_tile_counts_only_the_tokens_a_forward_pass_processed():
    """C2. `step_tokens_sum` is cores.py's numerator — `flops_s = steps_s * (decode_flops + prefill *
    per_token)` — so a token in it is a claim that the layers ran over it. `mode="prefill_cache"` is
    reused KV pages: no FLOPs, no weight read, nothing processed.

    The filter was `None`, every mode, and these two captures are the first to expose the third one.
    655.5 tok/step was what the live tile divided by at 16 seats; over this quieter 83 s window the
    same error reads 327.7 against 26.1 really processed — a x12.56 numerator, and the tile's
    `86.5 TFLOP/s real` against a measured nvfp4 peak of 323.2."""
    a, b = parse_metrics(CACHED_A), parse_metrics(CACHED_B)
    assert (a["steps_total"], b["steps_total"]) == (11869, 12600)
    assert (a["step_tokens_sum"], b["step_tokens_sum"]) == (966105, 985171)
    for s, text in ((a, CACHED_A), (b, CACHED_B)):
        # Exactly the two modes a pass processed, and the cache mode is excluded though it is exposed.
        assert s["step_tokens_sum"] == (_series(text, "sglang:realtime_tokens_total", mode="decode")
                                        + _series(text, "sglang:realtime_tokens_total",
                                                  mode="prefill_compute"))
        assert _series(text, "sglang:realtime_tokens_total", mode="prefill_cache") > 0
        # And it is the two token counters' own split of the same prompt: computed, not billed.
        assert s["step_tokens_sum"] == s["gen_tokens_total"] + s["prompt_computed_total"]
    steps = b["steps_total"] - a["steps_total"]
    tokens = b["step_tokens_sum"] - a["step_tokens_sum"]
    assert (steps, tokens) == (731, 19066)
    assert tokens / steps == pytest.approx(26.08, abs=0.01)
    # What the tile divided by before: every mode, 86 % of it reused pages.
    every_mode = sum(_series(t, "sglang:realtime_tokens_total", mode=m)
                     for t in (CACHED_B,) for m in ("decode", "prefill_compute", "prefill_cache")) \
        - sum(_series(t, "sglang:realtime_tokens_total", mode=m)
              for t in (CACHED_A,) for m in ("decode", "prefill_compute", "prefill_cache"))
    assert every_mode / steps == pytest.approx(327.71, abs=0.01)
    assert every_mode / tokens == pytest.approx(12.56, abs=0.01)


def test_the_two_token_counters_still_agree_with_each_other_after_the_split():
    """A guard on the widened mapping: `prompt_computed_total` is taken off `realtime_tokens_total`
    and `prompt_tokens_total` off `prefill_effective_tokens_total`, two different families, and the
    computed part of the second must be the whole of the first. On the 2026-10-02 captures, whose hit
    modes are 0, that collapses to the old reading — which is why those tests still hold."""
    for text in (CACHED_A, CACHED_B, BUSY_A, BUSY_B, IDLE):
        s = parse_metrics(text)
        assert s["prompt_computed_total"] == s["prompt_tokens_total"] - s["prompt_cached_total"]
    assert parse_metrics(BUSY_B)["prompt_tokens_total"] == 5489        # unchanged: no hits in it
    assert parse_metrics(CACHED_B)["prompt_tokens_total"] == 8517194   # the pair that was hitting


# ───────────────────── which convention produced the inter-token latency ─────────────────────

def test_the_itl_convention_travels_with_the_itl_value():
    """C9. Both engines publish `inter_token_latency_seconds`, and they do not mean the same thing:

      * vLLM records one gap per ENGINE STEP. In the captured TP=2 head `itl_count` is EXACTLY
        `spec_decode_num_drafts_total` — 7 667 == 7 667, and 7 864 == 7 864 in the second read —
        which is 0.324 of its generated tokens, i.e. 1/accept_length.
      * SGLang records one gap per EMITTED TOKEN: 169 554 against 169 889 decode tokens (0.998),
        and 3.02x its 56 122 verify steps.

    A mean ITL pooled across the two is therefore wrong by the accept length in opposite directions
    — the SGLang pair reads ~3x the faster per gap at the same tok/s. The value stays as the engine
    published it, because the BUCKETS behind the p90 cannot be rescaled after the fact; what the
    scrape now carries is which convention produced it."""
    sg, vl = parse_metrics(CACHED_B), parse_metrics(VLLM_HEAD)
    assert sg["itl_basis"] == ITL_PER_TOKEN == "token"
    assert vl["itl_basis"] == ITL_PER_STEP == "step"
    assert sg["itl_basis"] != vl["itl_basis"], "nothing may pool the two without seeing this"
    # The measurement behind each, from the captures' own counters.
    assert vl["itl_count"] == _series(VLLM_HEAD, "vllm:spec_decode_num_drafts_total") == 7667
    assert parse_metrics(VLLM_HEAD_2)["itl_count"] == 7864 \
        == _series(VLLM_HEAD_2, "vllm:spec_decode_num_drafts_total")
    assert vl["itl_count"] / vl["gen_tokens_total"] == pytest.approx(0.324, abs=0.001)
    assert sg["itl_count"] / sg["gen_tokens_total"] == pytest.approx(0.998, abs=0.001)
    assert sg["itl_count"] / sg["spec_drafts_total"] == pytest.approx(3.021, abs=0.001)
    # So the two means differ by the tokens behind one vLLM gap — 3.085, which is that engine's own
    # accept length (1 + 16 042/7 667 = 3.092) to within the first token of each request.
    assert (vl["itl_sum"] / vl["itl_count"]) / (vl["itl_sum"] / vl["gen_tokens_total"]) \
        == pytest.approx(3.085, abs=0.001)
    assert 1 + _series(VLLM_HEAD, "vllm:spec_decode_num_accepted_tokens_total") / vl["itl_count"] \
        == pytest.approx(3.092, abs=0.001)
    # AND NOTHING ON THE WIRE SAYS WHICH IS WHICH: both engines describe the family with the same
    # sentence, word for word, which is why the basis has to be carried rather than inferred.
    help_line = "Histogram of inter-token latency in seconds."
    assert f"# HELP sglang:inter_token_latency_seconds {help_line}" in CACHED_B
    assert f"# HELP vllm:inter_token_latency_seconds {help_line}" in VLLM_HEAD


def test_a_scrape_with_no_inter_token_latency_claims_no_basis():
    """The basis describes a value; with no value there is nothing to describe, and a non-null key on
    an empty scrape would be the invented-figure bug class in a new place."""
    for text in ("", "# nothing but a comment\n", "garbage\n",
                 'vllm:num_requests_running{engine="0"} 1.0\n'):
        out = parse_metrics(text)
        assert out["itl_count"] is None and out["itl_sum"] is None and out["itl_buckets"] is None
        assert out["itl_basis"] is None
    assert parse_metrics(IDLE)["itl_basis"] == ITL_PER_TOKEN      # an idle SGLang still has gaps
    assert parse_metrics(IDLE)["itl_count"] == 9576


def test_the_basis_is_one_key_of_the_canonical_scrape_like_every_other():
    """It travels on the same document as the figure it qualifies, in both vocabularies, so a
    consumer cannot read `itl_sum` without it being available — and it is a string, not a reading,
    which is why it is declared with its own kind rather than as a counter."""
    assert ("itl_basis", "basis") in SCRAPE_KEYS
    assert "itl_basis" in parse_sglang_metrics(CACHED_B)
    assert "itl_basis" in parse_vllm_metrics(VLLM_HEAD)
    assert {kind for key, kind in SCRAPE_KEYS if key == "itl_basis"} == {"basis"}
    assert "itl_basis" not in SGLANG_UNAVAILABLE and "itl_basis" not in SGLANG_BEHIND_A_FLAG
