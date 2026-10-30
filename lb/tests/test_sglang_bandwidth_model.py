"""MEMORY BANDWIDTH for an SGLang generation: a figure the LB MODELS from SGLang's own counters — and the
CUDA CORES tile pricing a decode pass with the same model, so the two tiles cannot disagree.

WHY SGLANG'S OWN FIGURE IS NOT DRAWN (probe.py `SGLANG_FAMILIES`): `sglang:estimated_read_bytes_per_gpu_
total` charges the whole per-rank weight read for every GENERATED TOKEN, at bf16 itemsize, and a KV term
8x the real fp8 one (metrics_reporter.py:476-620 in the pinned image). The two live captures below are
the first ones taken with `--enable-mfu-metrics` on, and over their 5.03 s its read counter moved
3.70e14 bytes — 73,600 GB/s a GPU on a 273 GB/s part. It stays unmapped.

WHAT IS DRAWN INSTEAD, per GPU (one rank of TP=2), over the same two-probe window as every rate:

    bytes/s = decode passes/s  x (W_target + W_draft + context x KV_B + min(context, seqs x 2048) x DKV_B
                                  + seqs x S x (D + 3))
            + prefill passes/s x W_target

  passes      `sglang:cuda_graph_passes_total`, +1 per decode batch (batch_result_processor.py:929-932)
              and +1 per prefill batch (metrics_reporter.py:726-729): ONE FORWARD each, spec on or off.
              NOT `sglang:spec_verify_calls_total`, which is +`req.spec_verify_ct` when a REQUEST
              FINISHES (metrics_collector.py:1803-1804; the per-request count is bumped per verify at
              batch_result_processor.py:773) — per request and completion-side, so it does not move
              while the work happens.
  W_target    8,804,567,176 B: the checkpoint's 64 layers (16,891,248,640 B of >=2-D tensors, halved
              by TP, + 1,351,808 B of 1-D ones every rank holds) + half its NVFP4 lm_head. Shard
              headers of /srv/models/Qwen3.8-27B-NVFP4. The 1.27e9 B embedding shard and the
              0.92e9 B vision tower are resident and never streamed.
  W_draft     2,353,031,688 B: DFlash2's attention + MLP halved by TP (QKV/MergedColumn/RowParallel in
              srt/models/dflash.py), its replicated fc + grouped convs + norms, + the TARGET's lm_head
              shard, which its candidate selector projects onto every pass (dflash.py:1097-1124). The
              selector's two 248,320 x 256 codebooks are gathered by candidate id, never streamed.
  KV_B        2 x 16 full-attention layers x 4/2 KV heads x 256 x 1 B (fp8_e4m3) = 16,384 B a token.
  DKV_B       the drafter's own KV: 2 x 5 layers x 8/2 KV heads x 128 x 1 B = 5,120 B a token, read over
              its 2,048-token sliding window only.
  S           48 GDN layers x 48/2 value heads x 128 x 128 x 2 B (--mamba-ssm-dtype bfloat16) =
              37,748,736 B a sequence. A verify reads it once, writes D snapshots
              (gdn_triton.py:201-241, `disable_state_update=True`, `cache_steps=draft_token_num`), and
              the commit copies the accepted one back (read + write): (D + 3) S. Without a drafter a
              decode pass reads and writes it once: 2 S.
  context     `sglang:decode_sum_seq_lens`, seqs `sglang:num_running_reqs`, D `sglang:spec_num_draft_
              tokens` (4: SGLang resolves DFlash2's block to it — 'block_size=4' and
              'num_tokens_per_req=4' in its own start log) — the engine's own gauges.

A window that held a prefill pass is a FLOOR: a prefill pass is charged its weights only, not the
prefix KV its attention reads. The constants live in the catalog (`traffic`), derived from config.json;
the engine's own allocations are the oracle each derivation is checked against below.
"""
from __future__ import annotations

import pathlib
import re

import pytest

from vllm_lb import cores
from vllm_lb.catalog import CLUSTER_MODE, SGLANG, ModelCatalog
from vllm_lb.probe import SGLANG_BEHIND_A_FLAG, SGLANG_UNAVAILABLE, parse_metrics
from vllm_lb.rates import FIRST_SAMPLE, NOT_EXPOSED, PASS_IN_FLIGHT, WHY
from vllm_lb.saturation import BANDWIDTH_BOUND, BOUND_BANDWIDTH_PCT, BUSY_UNMEASURED, gpu_view, judge
from vllm_lb.state import CLUSTER
from vllm_lb.throughput import PREFILL_FLOOR, Throughput
from tests.fleet import GB10_PEAK_GBPS

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
BUSY_A = (FIXTURES / "sglang-metrics-2026-10-02-busy-a.txt").read_text()
BUSY_B = (FIXTURES / "sglang-metrics-2026-10-02-busy-b.txt").read_text()
IDLE = (FIXTURES / "sglang-metrics-2026-10-02-idle.txt").read_text()
CACHED_B = (FIXTURES / "sglang-metrics-2026-10-03-cached-b.txt").read_text()
#: THE LIVE HEAD of the pair serving `qwen3.8-27b-sglang-dflash2` (GET http://192.0.2.2:8100/metrics,
#: read-only), 2026-10-03 07:02:18.396Z and 07:02:23.423Z: 5.027 s, 11 decode + 2 prefill passes, 29 -> 30
#: running, 21-23 queued, 1,502,161 context tokens in the decode batch. The first captures taken with
#: `--enable-mfu-metrics` on.
MFU_A = (FIXTURES / "sglang-metrics-2026-10-03-mfu-a.txt").read_text()
MFU_B = (FIXTURES / "sglang-metrics-2026-10-03-mfu-b.txt").read_text()
MFU_DT = 5.027
#: The same head 07:08:17.109Z and 07:08:22.109Z (5.005 s): 0 decode passes, 1 prefill pass, 28 running.
#: Over the minute these opened, 13 prefill passes ran and 0 decode passes for 55 s.
PREFILL_A = (FIXTURES / "sglang-metrics-2026-10-03-prefill-a.txt").read_text()
PREFILL_B = (FIXTURES / "sglang-metrics-2026-10-03-prefill-b.txt").read_text()
PREFILL_DT = 5.005
START_LOG = (FIXTURES / "sglang-lb-start-2026-10-03-rank0.log").read_text()

SPEC_ID = "qwen3.8-27b-sglang-dflash2"
CATALOG = ModelCatalog.load()
W_T, W_D, KV_B, DKV_B, S = 8_804_567_176, 2_353_031_688, 16_384, 5_120, 37_748_736
GIB = 1 << 30


def _series(text: str, name: str, **labels) -> float:
    """One exposed series' value straight out of the capture's bytes, not through the adapter."""
    for line in text.splitlines():
        if line.startswith(name + "{") and all(f'{k}="{v}"' in line for k, v in labels.items()):
            return float(line.rsplit("} ", 1)[1])
    raise AssertionError(f"{name}{labels} is not in this capture")


def _traffic():
    return CATALOG.get(SPEC_ID).traffic


def _pass(context: int, seqs: int, width: int = 4) -> int:
    """One decode pass, written out term by term, independently of the code under test."""
    drafting = width >= 2
    return (W_T + (W_D if drafting else 0) + context * KV_B
            + (min(context, seqs * 2048) * DKV_B if drafting else 0)
            + seqs * S * ((width + 3) if drafting else 2))


def _fed(*reads: tuple[float, str | dict], interval_s: float = 5.0) -> Throughput:
    """A Throughput fed these reads. The 2026-10-02 pair is 62 s apart, so it is read at a 30 s interval:
    at 5 s a 62 s hole is a probe gap, and no rate is taken across one."""
    t = Throughput(CLUSTER, interval_s=interval_s)
    for at, text in reads:
        scrape = parse_metrics(text) if isinstance(text, str) else text
        t.observe(live=True, scrape=scrape, mono_at=at, at=at)
    return t


# ───────────────────────── which counter is a forward pass ─────────────────────────

def test_spec_verify_calls_is_per_finished_request_and_cannot_be_a_pass_rate():
    """The counter first proposed. Across busy-a -> busy-b (62 s) 477 decode passes ran and it did not
    move (3730 in all three 2026-10-02 captures), because it is added to only when a request FINISHES.
    Live, it moved 172 over 11 decode passes — 34 'forwards' a second against 2.2 real ones."""
    for text in (BUSY_A, BUSY_B, IDLE):
        assert _series(text, "sglang:spec_verify_calls_total") == 3730
    passes = (_series(BUSY_B, "sglang:cuda_graph_passes_total", mode="decode_cuda_graph")
              - _series(BUSY_A, "sglang:cuda_graph_passes_total", mode="decode_cuda_graph"))
    assert passes == 477
    assert _series(MFU_B, "sglang:spec_verify_calls_total") - _series(MFU_A, "sglang:spec_verify_calls_total") == 172
    assert _series(MFU_B, "sglang:cuda_graph_passes_total", mode="decode_cuda_graph") \
        - _series(MFU_A, "sglang:cuda_graph_passes_total", mode="decode_cuda_graph") == 11


@pytest.mark.parametrize("text,decode,steps", [(BUSY_A, 966, 975), (BUSY_B, 1443, 1452), (MFU_A, 3845, 4094),
                                               (MFU_B, 3856, 4107), (PREFILL_B, 4368, 4678)],
                         ids=["busy-a", "busy-b", "mfu-a", "mfu-b", "prefill-b"])
def test_decode_passes_are_the_decode_modes_of_the_pass_counter(text, decode, steps):
    s = parse_metrics(text)
    assert s["decode_steps_total"] == decode
    assert s["steps_total"] == steps, "every mode, prefill included, as cores.py already reads it"


def test_the_gauges_the_model_reads_are_the_engines_own():
    b, live = parse_metrics(BUSY_B), parse_metrics(MFU_B)
    assert (b["decode_context_tokens"], b["running"], b["spec_verify_width"]) == (45654, 16, 4)
    assert (live["decode_context_tokens"], live["running"], live["spec_verify_width"]) == (1502161, 30, 4)
    for key in ("decode_steps_total", "decode_context_tokens", "spec_verify_width"):
        assert key not in SGLANG_UNAVAILABLE and key not in SGLANG_BEHIND_A_FLAG


def test_the_verify_width_is_the_block_sglang_resolved_not_one_plus_the_catalogs_k():
    """DFlash2's checkpoint says block_size 8; SGLang overrides it with --speculative-num-draft-tokens and
    verifies that many tokens a request, bonus included — its own start log says so."""
    assert "using speculative_num_draft_tokens=4 but draft config block_size=8" in START_LOG
    assert "model=DFlash2DraftModel, block_size=4" in START_LOG
    assert "Capture target verify CUDA graph begin. backend=full, num_tokens_per_req=4" in START_LOG
    assert parse_metrics(MFU_B)["spec_verify_width"] == 4


def test_a_vllm_scrape_feeds_none_of_the_three():
    """vLLM generations keep their own path: these keys are SGLang's, and null on a vLLM head."""
    head = (FIXTURES / "throughput" / "vllm-v0.29.0-cluster-head-1.metrics").read_text()
    s = parse_metrics(head)
    assert s["steps_total"] is not None
    assert s["decode_steps_total"] is None and s["decode_context_tokens"] is None
    assert s["spec_verify_width"] is None


def test_sglangs_own_read_counter_is_still_not_drawn_and_the_live_capture_shows_why():
    """The first live captures WITH the switch on: SGLang's own read counter moved 3.70e14 B in 5.027 s —
    73,600 GB/s a GPU, 270x the bus. The mapping stays withheld."""
    moved = (_series(MFU_B, "sglang:estimated_read_bytes_per_gpu_total")
             - _series(MFU_A, "sglang:estimated_read_bytes_per_gpu_total"))
    assert moved / MFU_DT / 1e9 == pytest.approx(73_640, rel=0.01)
    assert parse_metrics(MFU_B)["est_read_bytes_total"] is None
    assert parse_metrics(MFU_B)["est_flops_total"] is not None


# ───────────────────────── the constants, against the engine's own allocations ─────────────────────────

def test_the_sglang_spec_carries_its_traffic_and_no_vllm_spec_does():
    spec = CATALOG.get(SPEC_ID)
    assert spec.profile(CLUSTER_MODE).engine == SGLANG
    assert (spec.traffic.target_read_bytes, spec.traffic.draft_read_bytes) == (W_T, W_D)
    # the one other row with a traffic is TensorFold's routed-expert Flash-Next (test_tensorfold_bandwidth_model.py)
    assert {s.id for s in CATALOG.specs if s.traffic is not None} == {SPEC_ID, "qwen3.8-flash-next-tensorfold-nvfp4"}


@pytest.mark.parametrize("text", [BUSY_B, CACHED_B, MFU_B], ids=["busy-b", "cached-b", "mfu-b"])
def test_kv_bytes_a_token_is_the_pool_the_engine_allocated_over_its_tokens(text):
    """Derived from config.json, checked against three different pools: 52.375 GiB / 3,432,455 tokens,
    52.112 GiB / 3,415,240 and 48.521 GiB / 3,179,854 are each 16,384 B a token, exactly."""
    s = parse_metrics(text)
    assert _traffic().kv_bytes_per_token == KV_B == 2 * 16 * (4 // 2) * 256 * 1
    assert s["kv_pool_bytes"] / s["kv_pool_tokens"] == pytest.approx(KV_B, abs=0.05)


def test_the_drafters_kv_a_token_is_its_own_pool_over_the_same_tokens():
    """SGLang's start log allocates the drafter a second pool beside the target's: 'K size: 8.16 GB,
    V size: 8.16 GB' over the same 3,420,910 tokens = 5,122 B, the 5,120 the drafter's config derives
    to at the log's two-decimal rounding."""
    pools = re.findall(r"KV Cache is allocated\. dtype: torch\.float8_e4m3fn, #tokens: (\d+), K size: ([\d.]+) GB",
                       START_LOG)
    assert [(int(t), float(k)) for t, k in pools] == [(3420910, 26.10), (3420910, 8.16)]
    assert _traffic().draft_kv_bytes_per_token == DKV_B == 2 * 5 * (8 // 2) * 128 * 1
    assert 2 * 8.16 * GIB / 3420910 == pytest.approx(DKV_B, rel=0.001)
    assert 2 * 26.10 * GIB / 3420910 == pytest.approx(KV_B, rel=0.001)


def test_the_gdn_state_a_sequence_is_the_pool_the_engine_allocated_over_its_slots():
    """SGLang's own start line: 'max_mamba_cache_size: 80, ... ssm_state size: 2.85GB intermediate_ssm_
    state_cache size: 2.39GB' — GiB, over 80 + 1 slots, and 16 + 1 verify slots x 4 snapshots."""
    line = next(l for l in START_LOG.splitlines() if "Mamba Cache is allocated" in l)
    ssm = float(re.search(r"ssm_state size: ([\d.]+)GB", line).group(1))
    snapshots = float(re.search(r"intermediate_ssm_state_cache size: ([\d.]+)GB", line).group(1))
    assert _traffic().linear_state_bytes == S == 48 * (48 // 2) * 128 * 128 * 2
    assert ssm * GIB / 81 == pytest.approx(S, rel=0.002)
    assert snapshots * GIB / (17 * 4) == pytest.approx(S, rel=0.003)


def test_the_weights_a_pass_streams_are_under_what_the_engine_holds():
    """Read per pass <= resident: 'Load weight end ... mem usage=11.18 GB' (target) and '2.25 GB'
    (DFlash2, SHARDED: the whole file is 3.58 GiB). The catalog's resident `weights_gib` 10.28 +
    `mtp_gib` 3.58 is 1.33x the read: it holds the gathered embedding and the drafter whole."""
    loads = [float(x) for x in re.findall(r"Load weight end\..*?mem usage=([\d.]+) GB", START_LOG)]
    assert loads == [11.18, 2.25]
    assert W_T < loads[0] * GIB
    assert W_D - 357_580_808 <= loads[1] * GIB, "the drafter's own share, the target head it borrows aside"
    mem = CATALOG.get(SPEC_ID).memory
    assert (mem.weights_gib + mem.mtp_gib) * GIB > (W_T + W_D) * 1.3


def test_a_decode_pass_and_its_three_terms():
    """At the brief's sample — 22 running, 1,154,302 context tokens, D 4 — a decode pass moves
    36.11e9 B a rank: 11.16e9 of weights, 19.14e9 of KV (18.91 target + 0.23 drafter), 5.81e9 of GDN state."""
    weights, kv, state = _traffic().decode_pass_bytes(context=1_154_302, seqs=22, verify_width=4)
    assert (weights, kv, state) == (W_T + W_D, 1_154_302 * KV_B + 22 * 2048 * DKV_B, 22 * 7 * S)
    assert weights + kv + state == _pass(1_154_302, 22) == 36_113_674_896
    # Speculative decoding off: no drafter, and the state is read and written once.
    assert _traffic().decode_pass_bytes(context=1_154_302, seqs=22, verify_width=0) == \
        (W_T, 1_154_302 * KV_B, 22 * 2 * S)
    # A drafter window longer than the context it has is the context.
    assert _traffic().decode_pass_bytes(context=3_000, seqs=4, verify_width=4)[1] == 3_000 * (KV_B + DKV_B)
    assert _traffic().prefill_bytes(8192) == _traffic().prefill_bytes(None) == W_T


# ───────────────────────── the rate, through throughput.py ─────────────────────────

def test_the_live_pair_models_100_gbps_a_gpu_under_the_273_the_bus_has():
    """THE SANITY CHECK, on the live head: 11 decode passes x 44.01e9 B + 2 prefill passes x 8.80e9 B over
    5.027 s = 99.8 GB/s a GPU — 37 % of 273. A floor: two prefill passes ran in the window."""
    m = _fed((0.0, MFU_A), (MFU_DT, MFU_B)).bandwidth_model(_traffic())
    assert _pass(1_502_161, 30) == 44_010_812_048
    assert m["gbps_per_gpu"] == pytest.approx((11 * _pass(1_502_161, 30) + 2 * W_T) / MFU_DT / 1e9)
    assert m["gbps_per_gpu"] == pytest.approx(99.81, abs=0.01)
    assert m["gbps_per_gpu"] <= 273.0
    assert m["why"] is None and m["window_s"] == pytest.approx(MFU_DT)
    assert m["floor"] == PREFILL_FLOOR
    assert (m["decode_passes_s"], m["prefill_passes_s"]) == (pytest.approx(11 / MFU_DT), pytest.approx(2 / MFU_DT))
    assert (m["context_tokens"], m["seqs"], m["verify_width"]) == (1_502_161, 30, 4)
    assert m["weights_gbps"] + m["kv_gbps"] + m["state_gbps"] == pytest.approx(m["gbps_per_gpu"])
    assert (m["weights_gbps"], m["kv_gbps"], m["state_gbps"]) == (
        pytest.approx(27.92, abs=0.01), pytest.approx(54.54, abs=0.01), pytest.approx(17.35, abs=0.01))


def test_a_decode_only_window_is_the_model_not_a_floor():
    """busy-a -> busy-b: 477 decode passes, no prefill, 16 seats, 45,654 context tokens, 62 s: 125.4 GB/s."""
    m = _fed((0.0, BUSY_A), (62.0, BUSY_B), interval_s=30.0).bandwidth_model(_traffic())
    assert m["gbps_per_gpu"] == pytest.approx(477 * _pass(45_654, 16) / 62 / 1e9)
    assert m["gbps_per_gpu"] == pytest.approx(125.41, abs=0.01)
    assert m["floor"] is None and m["prefill_passes_s"] == 0.0


def test_a_prefill_only_window_is_a_floor_of_its_weights():
    """The live minute with no decode pass: one prefill pass in 5.005 s is 1.76 GB/s of weights — a floor,
    since the prefix KV that pass's attention read is not modeled."""
    m = _fed((0.0, PREFILL_A), (PREFILL_DT, PREFILL_B)).bandwidth_model(_traffic())
    assert m["gbps_per_gpu"] == pytest.approx(W_T / PREFILL_DT / 1e9)
    assert m["floor"] == PREFILL_FLOOR and m["decode_passes_s"] == 0.0


def test_idle_is_a_measured_zero():
    m = _fed((0.0, IDLE), (5.0, IDLE)).bandwidth_model(_traffic())
    assert m["gbps_per_gpu"] == 0.0 and m["why"] is None and m["floor"] is None


def test_a_pass_still_running_is_not_a_zero_and_widens_to_the_last_one_that_completed():
    """Requests running, no pass completed since the last probe: a pass in flight. With an earlier
    completed pass in the kept history the window widens back to it — a real count, and a floor."""
    alone = _fed((0.0, BUSY_B), (5.0, BUSY_B)).bandwidth_model(_traffic())
    assert alone["gbps_per_gpu"] is None and alone["why"] == PASS_IN_FLIGHT
    assert PASS_IN_FLIGHT in WHY
    widened = _fed((0.0, BUSY_A), (5.0, BUSY_B), (10.0, BUSY_B)).bandwidth_model(_traffic())
    assert widened["window_s"] == pytest.approx(10.0)
    assert widened["gbps_per_gpu"] == pytest.approx(477 * _pass(45_654, 16) / 10 / 1e9)
    assert widened["floor"]


def test_without_two_reads_there_is_no_rate_and_it_says_why():
    m = _fed((0.0, MFU_A)).bandwidth_model(_traffic())
    assert m["gbps_per_gpu"] is None and m["why"] == FIRST_SAMPLE


def test_a_scrape_without_the_gauges_is_not_exposed():
    scrape = {**parse_metrics(BUSY_B), "decode_context_tokens": None}
    m = _fed((0.0, BUSY_A), (5.0, scrape)).bandwidth_model(_traffic())
    assert m["gbps_per_gpu"] is None and m["why"] == NOT_EXPOSED


def test_spec_off_charges_no_drafter_and_one_state_round_trip():
    off = {**parse_metrics(BUSY_B), "spec_verify_width": 0}
    m = _fed((0.0, BUSY_A), (62.0, off), interval_s=30.0).bandwidth_model(_traffic())
    assert m["gbps_per_gpu"] == pytest.approx(477 * (W_T + 45_654 * KV_B + 16 * 2 * S) / 62 / 1e9)


def test_the_throughput_view_carries_it_only_when_given_a_traffic():
    from vllm_lb.state import Backend
    t = _fed((0.0, MFU_A), (MFU_DT, MFU_B))
    b = Backend(kind=CLUSTER, url="http://h:8100", live=True)
    assert t.view(b, MFU_DT, 0.0)["bandwidth_model"] is None
    assert t.view(b, MFU_DT, 0.0, traffic=_traffic())["bandwidth_model"]["gbps_per_gpu"] == \
        pytest.approx(99.81, abs=0.01)
    # The engine's own estimate is unchanged by it: SGLang's bytes stay withheld.
    assert t.engine_perf()["gbps_per_gpu"] is None


# ───────────────────────── the CUDA CORES tile prices a pass the same way ─────────────────────────

#: The bottleneck audit's decode sample (worklog/sglang-bottleneck-2026-10-03.md): 29 seats, 1.354 M
#: context tokens summed, a 393 ms decode step — the step the tile's 744 memory cores were drawn at.
AUDIT = dict(steps_s=1 / 0.393, tokens_step=91.0, running=29, speculative=4, budget=8192,
             peak_tflops=323.2, peak_gbps=273.0, fit=(0.393, 0.0), context=1_354_000 / 29, cuda_cores=6144)


def test_the_cores_tiles_memory_is_the_whole_modeled_pass_not_its_weights():
    """WEIGHTS ONLY was 13.0 GB a step (744 cores at 393 ms); the pass moves 41.31e9 B: 2,366 cores."""
    c = cores.estimate(SPEC_ID, traffic=_traffic(), verify_width=4, **AUDIT)
    weighed = cores.estimate(SPEC_ID, **AUDIT)
    assert weighed["cases"]["decode"]["memory"] == round(6144 * (8.80e9 + 4.21e9) / 0.393 / 273e9) == 745
    assert _pass(1_354_000, 29) == 41_308_615_312
    assert c["cases"]["decode"]["memory"] == round(6144 * _pass(1_354_000, 29) / 0.393 / 273e9) == 2366
    assert c["bytes"] == "modeled" and weighed["bytes"] == "weights"


def test_the_cores_tile_verifies_four_tokens_a_seat_and_drafts_a_block_of_four():
    """SGLang verifies D = 4 a seat (not 1 + K = 5) and its drafter runs one block of D positions (not 8):
    116 verified tokens at 46,690 context each, + 4 x 29 drafted positions."""
    c = cores.estimate(SPEC_ID, traffic=_traffic(), verify_width=4, **AUDIT)
    ctx = 1_354_000 / 29
    flops = 116 * (24.32e9 + ctx * 196_608.0 + 1.271e9) + 4 * 29 * 5.12e9
    assert c["cases"]["decode"]["multiplying"] == round(6144 * flops / 0.393 / 323.2e12) == 224


def test_lb_status_cores_and_bandwidth_price_a_pass_with_one_model():
    s = _status(SPEC_ID, [(0.0, parse_metrics(MFU_A)), (MFU_DT, parse_metrics(MFU_B))])
    c = s["saturation"]["cores"]
    assert c["bytes"] == "modeled"
    t = c["cases"]["decode"]["step_ms"] / 1e3
    assert c["cases"]["decode"]["memory"] == pytest.approx(
        min(6144 - c["cases"]["decode"]["multiplying"], 6144 * _pass(1_502_161, 30) / t / 273e9), abs=40)


# ───────────────────────── the verdict ─────────────────────────

BOX_A = {"count": 1, "sm_pct": 96, "mem_duty_pct": 0, "pwr_w": 57, "gtemp_c": 71, "pclk_mhz": 2424,
         "pclk_max_mhz": 3003}


def _judged(gbps: float, floor: str | None) -> dict:
    model = {"gbps_per_gpu": gbps, "window_s": 5.0, "why": None, "floor": floor, "weights_gbps": 1.0,
             "kv_gbps": 1.0, "state_gbps": 1.0, "decode_passes_s": 1.0, "prefill_passes_s": 0.0,
             "context_tokens": 1, "seqs": 1, "verify_width": 4}
    return judge(gpus=[dict(gpu_view(BOX_A), box="box-a")], tflops_per_gpu=None, gbps_per_gpu=gbps,
                 peak_tflops=None, peak_gbps=273.0, gpu_count=2, engine="sglang", bytes_model=model)


def test_a_modeled_figure_judges_bandwidth_bound_at_the_same_threshold():
    v = _judged(0.75 * 273.0, None)
    assert BOUND_BANDWIDTH_PCT == 70.0
    assert v["verdict"] == BANDWIDTH_BOUND and v["headroom"] is False
    assert "modeled" in v["because"]
    assert v["bandwidth_model"]["gbps_per_gpu"] == pytest.approx(204.75, abs=0.05)
    assert "modeled" in v["rates_means"]
    assert _judged(0.69 * 273.0, None)["verdict"] == BUSY_UNMEASURED


def test_a_floor_over_the_threshold_still_proves_it_and_says_at_least():
    v = _judged(0.8 * 273.0, PREFILL_FLOOR)
    assert v["verdict"] == BANDWIDTH_BOUND and "at least" in v["because"]


def test_a_floor_under_the_threshold_claims_no_headroom():
    v = _judged(99.8, PREFILL_FLOOR)
    assert v["verdict"] == BUSY_UNMEASURED and v["headroom"] is None
    assert PREFILL_FLOOR in v["because"]
    assert "more concurrency should still buy throughput" not in v["because"]
    assert _judged(99.8, None)["headroom"] is True


def test_an_engine_estimated_bandwidth_carries_no_model():
    v = judge(gpus=[dict(gpu_view(BOX_A), box="box-a")], tflops_per_gpu=12.0, gbps_per_gpu=220.0,
              peak_tflops=None, peak_gbps=273.0, gpu_count=2)
    assert v["bandwidth_model"] is None
    assert "modeled" not in v["rates_means"] and "modeled" not in v["because"]


# ───────────────────────── end to end, over the real Proxy.status() ─────────────────────────

HEAD = "http://192.0.2.2:8100"
HOSTS = "box-a-gpu=box-a=user@10.0.0.1=10.9.200.1,box-b-gpu=box-b=user@10.0.0.2=10.9.200.2"


def _status(choice: str, reads: list[tuple[float, dict]]) -> dict:
    from vllm_lb.cluster import parse_hosts
    from vllm_lb.config import Config
    from vllm_lb.hub import Hub
    from vllm_lb.proxy import Proxy
    # The nvfp4 peak the deployed LB is configured with (LB_GPU_PEAK_TFLOPS): without one there is no
    # CUDA CORES tile at all.
    from tests.fleet_files import gb10_pair
    # The pair's cards: GB10s, whose type states 273 GB/s and 323.2 TFLOP/s nvfp4 — without a peak there is no
    # CUDA CORES tile at all.
    cfg = Config(vllm_boxes=(), 
                 cluster_hosts=parse_hosts(HOSTS), cluster_url=HEAD, fleet=gb10_pair(),
                 listen_host="127.0.0.1", listen_port=0)
    hub = Hub(cfg)
    hub.set_cluster({"configured": True, "state": "running",
                     "effective": {"model_choice": choice, "num_speculative_tokens": 4,
                                   "max_num_batched_tokens": 8192},
                     "boxes": {"box-a": {"gpu": gpu_view(BOX_A)}, "box-b": {"gpu": gpu_view(BOX_A)}}})
    for at, scrape in reads:
        hub.update_backend(HEAD, live=True, last_probe_at=at, mono_at=at, depth_known=True,
                           waiting=scrape.get("waiting") or 0, running=scrape.get("running") or 0,
                           served_models=("Qwen/Qwen3.8-27B-SGLang",), max_model_len=262144, scrape=scrape)
    return Proxy(cfg, hub).status()


def test_lb_status_draws_the_modeled_bandwidth_for_an_sglang_generation():
    s = _status(SPEC_ID, [(0.0, parse_metrics(MFU_A)), (MFU_DT, parse_metrics(MFU_B))])
    sat = s["saturation"]
    assert sat["bandwidth"]["rate"] == pytest.approx(99.8, abs=0.05)
    assert sat["bandwidth"]["pct"] == pytest.approx(36.6, abs=0.05)
    assert sat["bandwidth_model"]["floor"] == PREFILL_FLOOR
    assert sat["verdict"] == BUSY_UNMEASURED and sat["headroom"] is None
    assert sat["cluster_gbps"] == pytest.approx(199.6, abs=0.1)
    head = next(b for b in s["backends"] if b["kind"] == CLUSTER)
    assert head["throughput"]["bandwidth_model"]["gbps_per_gpu"] == pytest.approx(99.81, abs=0.01)


def test_a_vllm_generation_keeps_the_engines_own_estimate_and_no_model():
    vllm_head = [(at, parse_metrics((FIXTURES / "throughput" / f"vllm-v0.29.0-cluster-head-{i}.metrics")
                                    .read_text())) for at, i in ((0.0, 1), (5.0, 2))]
    s = _status("qwen3.8-27b-dflash2", vllm_head)
    assert s["saturation"]["bandwidth_model"] is None
    head = next(b for b in s["backends"] if b["kind"] == CLUSTER)
    assert head["throughput"]["bandwidth_model"] is None
    assert (s["saturation"]["cores"] or {}).get("bytes") in (None, "weights")
