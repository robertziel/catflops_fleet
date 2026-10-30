"""COMPUTE for an SGLang generation: the LB's OWN FLOP model — the one the CUDA CORES tile draws — and not
SGLang's `sglang:estimated_flops_per_gpu_total`, which read 236.8 TFLOP/s (73 % of 323.2) beside a CUDA
CORES tile showing 742 of 6,144 cores multiplying during prefill. One model, one window, one number.

WHY SGLANG'S COUNTER IS NOT THE FIGURE. Its estimator (metrics_reporter.py:476-620, pinned image
myllmbox/sglang-27b-tp2:708f51e-fleet.2) prices a token as if every one of the checkpoint's 64 layers were
a dense GQA attention layer with an UNSHARDED MLP:

    linear     (2·h·d·(q + 2kv) + 2·h·d·q + 6·h·I) x 64      = 38.92e9 FLOP a token, against the real
               24.32e9 (the 64 layers' matmuls, halved by TP: cores.py `_DENSE_27B`)    -> x1.60
                 MLP          6·h·I charged whole on each rank: 34.23e9 against 17.11e9     -> x2.00
                 projections  64 GQA layers without the output gate: 4.70e9 against 7.21e9 (16 gated
                              attention + 48 GDN in/out projections)                       -> x0.65
    attention  4·q·d x 64 = 786,432 FLOP a query-key pair, against 196,608 (16 full-attention layers:
               2 FLOP x 12 heads x 256, QK and AV)                                        -> x4.00
    left out   the drafter's context projection (0.31e9 a prompt token, 1.3 %), the lm_head

so a prefill chunk is overcounted x1.66 cold, x2.4 at 65k deep, x2.6 at 86k, and toward x4 as depth grows.

WHAT IS DRAWN INSTEAD, per GPU, over the CUDA CORES tile's own window (the last ~1 min of probes):

    FLOP/s = [ T_p x (24.32e9 + 0.31e9) + P x 196,608  +  n_d x F_verify ] / max(wall, busy)

    T_p      prompt tokens the engine COMPUTED (`realtime_tokens_total{prefill_compute}`)
    P        the causal query-key pairs of those tokens — each against its prompt's whole prefix, so a
             chunk deep in a 130k prompt attends to ~130k positions. SGLang's own counter carries them
             EXACTLY (`_prefill_attention_pairs`: Σ c·p + c(c+1)/2 over the batch's extend and prefix
             lengths), so they are read back out of it with its own constants:
                 P = (ΔE − 38.92e9 x (T_p + T_decode) − 786,432 x n_d x Σctx) / 786,432
             Without the counter: T_p x (cached prefix a pass + chunk/2), an approximation that misses
             how deep a continuation chunk sits.
    F_verify a decode pass: seqs x D x (24.32e9 + lm_head) + D x Σctx x 196,608 + D x seqs x drafter
    busy     SGLang's own `scheduler_stage_seconds_total{category="run_batch"}`: the wall clock its
             scheduler spent in forward passes. It is credited when a pass returns, as the pass counter
             is, so a window shorter than the passes it counted is widened to their time.

and the busy time is split between the two kinds of pass by what was MEASURED: all of it to the one kind
when only one ran, else a least-squares fit over the window's probe intervals (busy = t_dec x decode passes
+ prefill FLOPs / prefill FLOP rate), else the step-time prior's ratio scaled to the measured total.
"""
from __future__ import annotations

import pathlib

import pytest

from vllm_lb import cores
from vllm_lb.probe import parse_metrics
from vllm_lb.rates import FIRST_SAMPLE, PASS_IN_FLIGHT
from vllm_lb.saturation import gpu_view
from vllm_lb.state import CLUSTER
from tests.fleet import GB10_PEAK_GBPS

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
SPEC_ID = "qwen3.8-27b-sglang-dflash2"
PEAK = 323.2
N = 6144
#: The real FLOPs, per GPU: the 27B's layers a token, the drafter's context projection a prompt token
#: (its replicated 5 x 5,120 -> 5,120 `fc` + its five layers' K and V, sharded), a query-key pair.
LIN = 24.32e9
DRAFT_PREFILL = 2 * 5 * 5120 * 5120 + 2 * (5 * 5120 * 1024 * 2) / 2
PAIR = 2 * 2 * 12 * 256 * 16
#: SGLang's estimator constants, written out from its own formula and the checkpoint's config.json.
H, D, Q, KV, I, L = 5120, 256, 24 // 2, 4 // 2, 17408, 64
L_SG = (2 * H * D * (Q + 2 * KV) + 2 * H * D * Q + 6 * H * I) * L
A_SG = 4 * Q * D * L


def _text(name: str) -> str:
    return (FIXTURES / f"sglang-metrics-2026-10-03-{name}.txt").read_text()


#: Two reads 5.005 s apart, 07:08:17Z: ONE 8,192-token prefill pass, no decode pass, run_batch +6.507 s.
PREFILL = [(0.0, parse_metrics(_text("prefill-a"))), (5.005, parse_metrics(_text("prefill-b")))]
#: 07:02:18Z, 5.027 s: 11 decode passes and 2 prefill passes (8,454 tokens), run_batch +6.320 s.
MFU = [(0.0, parse_metrics(_text("mfu-a"))), (5.027, parse_metrics(_text("mfu-b")))]


def _live(est, busy, prefill_passes, computed):
    """One read of the live head (GET :8100/metrics, read-only, 2026-10-03 ~10:26Z) while a cold prompt was
    being chunked: nothing but prefill passes, 28 running, 1,116,301 context tokens in the decode batch."""
    return {"steps_total": prefill_passes + 1738, "decode_steps_total": 1738, "prompt_computed_total": computed,
            "prompt_cached_total": 3_838_851, "gen_tokens_total": 50_096, "est_flops_total": est,
            "busy_seconds_total": busy, "decode_context_tokens": 1_116_301, "running": 28, "waiting": 20,
            "spec_verify_width": 4, "step_tokens_sum": computed + 50_096, "kv_cache_usage": 0.1}


#: Seven consecutive reads ~6 s apart: five 8,192-token chunks of ONE prompt, and one interval in which the
#: chunk running had not returned yet (no pass, no busy time credited).
CHUNKS = [
    (0.0, _live(1.1831077179804877e17, 1073.7513276000009, 173, 1_258_731)),
    (6.024494590, _live(1.1939489348426138e17, 1082.9106960180009, 174, 1_266_923)),
    (12.050949649, _live(1.1939489348426138e17, 1082.9106960180009, 174, 1_266_923)),
    (18.071114733, _live(1.2053179172860723e17, 1090.1188043080008, 175, 1_275_115)),
    (24.092073792, _live(1.2172146653108634e17, 1097.5647550170008, 176, 1_283_307)),
    (30.113079571, _live(1.2296391789169869e17, 1105.2576594310008, 177, 1_291_499)),
    (36.133896782, _live(1.2425914581044429e17, 1113.5088662010007, 178, 1_299_691)),
]


def _model(reads, **kw):
    return cores.modeled_flops(SPEC_ID, reads, peak_tflops=PEAK, **kw)


# ───────────────────────── SGLang's counter, and what it overcounts ─────────────────────────

def test_sglangs_estimator_constants_are_its_own_formula_on_this_checkpoint():
    shape = cores.SHAPES[SPEC_ID]
    assert shape["engine_linear_flops"] == L_SG == 38_923_141_120
    assert shape["engine_attn_flops"] == A_SG == 786_432
    assert shape["draft_prefill_flops"] == pytest.approx(DRAFT_PREFILL) == pytest.approx(0.3146e9, abs=1e6)
    assert L_SG / shape["flops_per_token"] == pytest.approx(1.60, abs=0.005)
    assert A_SG / shape["attn_flops_per_ctx_token"] == 4.0


def test_a_chunks_depth_is_read_back_exactly_out_of_sglangs_own_counter():
    """Five chunks of one cold prompt: each chunk's mean depth is (k + ½) x 8,192 + ½ — chunk k of a prompt
    that started at position 0 — to the token. Only SGLang's real constants give that."""
    depths = [_model(CHUNKS[i:i + 2])["prefill_depth"] for i in (0, 2, 3, 4, 5)]
    assert depths == pytest.approx([(k + 0.5) * 8192 + 0.5 for k in (14, 15, 16, 17, 18)], abs=0.01)
    assert _model(CHUNKS[:2])["depth_from"] == "engine"


def test_without_sglangs_counter_the_depth_is_the_cached_prefix_plus_half_a_pass_and_says_so():
    reads = [(at, {**s, "est_flops_total": None}) for at, s in MFU]
    m = _model(reads)
    cached = MFU[1][1]["prompt_cached_total"] - MFU[0][1]["prompt_cached_total"]
    tokens = MFU[1][1]["prompt_computed_total"] - MFU[0][1]["prompt_computed_total"]
    assert m["depth_from"] == "cached prefix"
    assert m["prefill_depth"] == pytest.approx(cached / 2 + tokens / 2 / 2)


# ───────────────────────── the figure, on the committed captures and the live chunks ─────────────────

def test_one_prefill_chunk_is_its_flops_over_the_time_sglang_spent_in_it():
    """The prefill fixture: chunk 11 of a cold prompt (mean depth 86,016.5), 3.40e14 FLOP in the 6.507 s
    run_batch credited for it — 52.3 TFLOP/s, 16 % of 323.2. SGLang's own counter says 134.2 over the same
    6.507 s: x2.57."""
    m = _model(PREFILL)
    pairs = 86_016.5 * 8192
    flops = 8192 * (LIN + DRAFT_PREFILL) + pairs * PAIR
    busy = 1770.048247809002 - 1763.541419974002
    assert m["prefill_depth"] == pytest.approx(86_016.5, abs=0.01)
    assert m["window_s"] == pytest.approx(busy) and m["busy_s"] == pytest.approx(busy)
    assert m["tflops_per_gpu"] == pytest.approx(flops / busy / 1e12) == pytest.approx(52.31, abs=0.01)
    assert m["prefill_s"] == pytest.approx(busy) and m["decode_s"] == 0.0 and m["split"] == "one kind"
    assert m["prefill_tflops"] == pytest.approx(m["tflops_per_gpu"])
    assert m["engine_tflops"] == pytest.approx(134.17, abs=0.01)
    assert 0.15 < m["tflops_per_gpu"] / PEAK < 0.17


def test_a_mixed_window_adds_its_decode_passes_at_their_own_cost():
    """11 verify passes x 4 positions at 1,502,161 context, beside 8,454 prompt tokens at depth 38,280. The
    seats are the two reads' mean, 29 -> 30: 29.5 a pass."""
    m = _model(MFU)
    verify = 29.5 * 4 * (LIN + 1.271e9) + 4 * 1_502_161 * PAIR + 4 * 29.5 * 5.12e9
    pairs = 38_280.498 * 8454
    flops = 8454 * (LIN + DRAFT_PREFILL) + pairs * PAIR + 11 * verify
    busy = 1420.342328014002 - 1414.022616080002
    assert m["prefill_depth"] == pytest.approx(38_280.5, abs=0.5)
    assert m["tflops_per_gpu"] == pytest.approx(flops / busy / 1e12, rel=1e-4) == pytest.approx(51.39, abs=0.02)
    assert m["decode_passes"] == 11 and m["prefill_passes"] == 2
    assert m["prefill_s"] + m["decode_s"] == pytest.approx(busy)
    # One interval cannot separate the two kinds: the prior's ratio, scaled to what was measured.
    assert m["split"] == "prior"
    assert m["decode_tflops"] < m["prefill_tflops"] / 2
    assert m["engine_tflops"] == pytest.approx(100.56, abs=0.01)


def test_the_live_chunks_run_at_17_percent_of_peak_while_sglang_says_more_than_double():
    m = _model(CHUNKS)
    busy = 1113.5088662010007 - 1073.7513276000009
    assert m["window_s"] == pytest.approx(busy), "five ~8 s passes counted in 36 s: widened to their time"
    assert m["prefill_passes"] == 5 and m["prefill_s"] / 5 == pytest.approx(7.95, abs=0.01)
    assert m["tflops_per_gpu"] == pytest.approx(52.76, abs=0.01)
    assert m["engine_tflops"] > 2 * m["tflops_per_gpu"]


def _synthetic(rows, t_dec=0.30, rate=70e12):
    """Reads of an engine whose decode passes take `t_dec` and whose prefill runs at `rate` FLOP/s, one interval
    per (decode passes, prefill passes, prompt tokens) row. No SGLang counter: depth = half a pass."""
    reads, c = [], dict(steps=0, dec=0, tok=0, busy=0.0)
    for i, (nd, npf, tok) in enumerate([(0, 0, 0)] + rows):
        flops_p = tok * (LIN + DRAFT_PREFILL) + tok * (tok / max(1, npf) / 2) * PAIR
        c["steps"] += nd + npf; c["dec"] += nd; c["tok"] += tok
        c["busy"] += nd * t_dec + flops_p / rate
        reads.append((6.0 * i, {"steps_total": c["steps"], "decode_steps_total": c["dec"],
                                "prompt_computed_total": c["tok"], "prompt_cached_total": 0,
                                "gen_tokens_total": 0, "est_flops_total": None,
                                "busy_seconds_total": c["busy"], "decode_context_tokens": 1_000_000,
                                "running": 30, "spec_verify_width": 4}))
    return reads


VARIED = [(10, 0, 0), (0, 1, 8192), (5, 1, 8192), (12, 0, 0), (2, 2, 12000), (8, 1, 4000)]


def test_the_busy_split_is_fitted_over_the_windows_intervals_when_they_vary():
    """Decode passes of 0.30 s and prefill at 70 TFLOP/s, in intervals of different mixes: the fit returns both."""
    m = _model(_synthetic(VARIED))
    assert m["split"] == "fit"
    assert m["decode_s"] / m["decode_passes"] == pytest.approx(0.30, rel=1e-6)
    assert m["prefill_tflops"] == pytest.approx(70.0, rel=1e-6)


def test_a_window_too_uniform_to_fit_borrows_the_fit_of_the_reads_before_it():
    """The last interval alone cannot separate the two kinds; the reads the hub kept before it can."""
    reads = _synthetic(VARIED + [(6, 1, 8192)])
    assert _model(reads[-2:])["split"] == "prior"
    m = _model(reads[-2:], history=reads)
    assert m["split"] == "fit"
    assert m["decode_s"] == pytest.approx(6 * 0.30, rel=1e-6)
    assert m["prefill_tflops"] == pytest.approx(70.0, rel=1e-6)


def _decode_read(passes, busy):
    return {"steps_total": passes, "decode_steps_total": passes, "prompt_computed_total": 0, "prompt_cached_total": 0,
            "gen_tokens_total": 0, "est_flops_total": None, "busy_seconds_total": busy,
            "decode_context_tokens": 1_502_161, "running": 30, "spec_verify_width": 4}


def test_a_decode_pass_takes_at_least_its_bytes_over_the_bus():
    """A clock that credited 11 decode passes 0.4 s would have moved 41 GB a pass at 1,100 GB/s: each is given
    at least what the MEMORY BANDWIDTH tile's model says it moves, over 273 GB/s."""
    from vllm_lb.catalog import ModelCatalog
    traffic = ModelCatalog.load().get(SPEC_ID).traffic
    m = _model([(0.0, _decode_read(0, 0.0)), (6.0, _decode_read(11, 0.4))], traffic=traffic, peak_gbps=273.0)
    per_pass = sum(traffic.decode_pass_bytes(context=1_502_161, seqs=30, verify_width=4))
    assert m["decode_s"] == pytest.approx(11 * per_pass / 273e9)
    assert _model([(0.0, _decode_read(0, 0.0)), (6.0, _decode_read(11, 0.4))])["decode_s"] == pytest.approx(0.4)


def test_never_above_peak():
    """A chunk credited 0.1 s of busy time in a 0.5 s window would be 3,400 TFLOP/s: its time is at least
    what its FLOPs need at peak, and the window at least that."""
    busy = CHUNKS[0][1]["busy_seconds_total"] + 0.1
    m = _model([(0.0, CHUNKS[0][1]), (0.5, {**CHUNKS[1][1], "busy_seconds_total": busy})])
    assert m["prefill_tflops"] == pytest.approx(PEAK) and m["tflops_per_gpu"] == pytest.approx(PEAK)


def test_a_pass_still_running_is_not_a_zero():
    m = _model(CHUNKS[1:3])
    assert m["tflops_per_gpu"] is None and m["why"] == PASS_IN_FLIGHT


def test_too_few_reads_is_no_figure_and_says_why():
    m = _model(PREFILL[:1])
    assert m["tflops_per_gpu"] is None and m["why"] == FIRST_SAMPLE


def test_an_idle_engine_is_a_measured_zero():
    idle = {**CHUNKS[0][1], "running": 0}
    assert _model([(0.0, idle), (6.0, idle)])["tflops_per_gpu"] == 0.0


# ───────────────────────── the CUDA CORES tile prices its cases with the same window ─────────────────

def test_the_prefill_case_is_priced_at_the_measured_pass_time_not_the_fit():
    """The fit's prior drew a 5,716 ms prefill step; the engine spent 7.95 s in each of these five chunks."""
    c = _status(SPEC_ID, CHUNKS)["saturation"]["cores"]
    p = c["cases"]["prefill"]
    assert p["step_ms"] == pytest.approx(7951, abs=2)
    assert p["time_share"] == 1.0 and c["cases"]["decode"]["time_share"] == 0.0
    per_pass = 8192 * (LIN + DRAFT_PREFILL) + PAIR * 8192 * sum((k + 0.5) * 8192 + 0.5 for k in range(14, 19)) / 5
    t = (1113.5088662010007 - 1073.7513276000009) / 5
    assert p["multiplying"] == round(N * per_pass / t / (PEAK * 1e12))


def test_the_chunk_is_the_engines_own_not_the_panels_budget():
    """The panel's budget said 2,048 while the engine ran 8,192-token passes: the tile prices what ran."""
    at_2k = _status(SPEC_ID, CHUNKS, budget=2048)["saturation"]["cores"]
    at_8k = _status(SPEC_ID, CHUNKS, budget=8192)["saturation"]["cores"]
    assert at_2k["cases"] == at_8k["cases"] and at_2k["tflops"] == at_8k["tflops"]
    assert at_2k["tokens_step"] == pytest.approx(8192.0)


# ───────────────────────── /lb/status: COMPUTE and CUDA CORES, one number ─────────────────────────

@pytest.mark.parametrize("reads", [PREFILL, MFU, CHUNKS], ids=["prefill", "mfu", "chunks"])
def test_compute_and_cuda_cores_read_one_flop_rate_for_one_window(reads):
    sat = _status(SPEC_ID, reads)["saturation"]
    rate, c = sat["compute"]["rate"], sat["cores"]
    assert rate == pytest.approx(c["tflops"], abs=0.05)
    assert c["multiplying"] == round(N * c["tflops"] / PEAK)
    assert sat["compute"]["pct"] == pytest.approx(100 * c["multiplying"] / N, abs=0.1)
    assert 0 < rate <= PEAK
    assert sat["compute_model"]["tflops_per_gpu"] == pytest.approx(rate, abs=0.05)


def test_compute_is_labelled_a_model_and_sglangs_own_figure_stays_beside_it():
    s = _status(SPEC_ID, PREFILL)
    sat = s["saturation"]
    assert sat["compute"]["rate"] == pytest.approx(52.3, abs=0.05)
    assert sat["compute_model"]["engine_tflops"] == pytest.approx(134.17, abs=0.01)
    assert "FLOP" in sat["rates_means"] and "modeled" in sat["rates_means"]
    assert "SGLang's analytical FLOP estimate" not in sat["rates_means"]
    head = next(b for b in s["backends"] if b["kind"] == CLUSTER)
    # SGLang's own counter is still read, and still in the server's engine_perf.
    assert head["throughput"]["engine_perf"]["tflops_per_gpu"] == pytest.approx(174.43, abs=0.01)


def test_a_vllm_generation_keeps_its_engines_compute_and_no_model():
    vllm_head = [(at, parse_metrics((FIXTURES / "throughput" / f"vllm-v0.29.0-cluster-head-{i}.metrics")
                                    .read_text())) for at, i in ((0.0, 1), (5.0, 2))]
    s = _status("qwen3.8-27b-dflash2", vllm_head)
    assert s["saturation"]["compute_model"] is None
    head = next(b for b in s["backends"] if b["kind"] == CLUSTER)
    assert s["saturation"]["compute"]["rate"] == pytest.approx(
        head["throughput"]["engine_perf"]["tflops_per_gpu"], abs=0.05)


# ───────────────────────── end to end, over the real Proxy.status() ─────────────────────────

HEAD = "http://192.0.2.2:8100"
HOSTS = "box-a-gpu=box-a=user@10.0.0.1=10.9.200.1,box-b-gpu=box-b=user@10.0.0.2=10.9.200.2"
BOX_A = {"count": 1, "sm_pct": 96, "mem_duty_pct": 0, "pwr_w": 57, "gtemp_c": 71, "pclk_mhz": 2424,
         "pclk_max_mhz": 3003}


def _status(choice: str, reads: list[tuple[float, dict]], budget: int = 8192) -> dict:
    from vllm_lb.cluster import parse_hosts
    from vllm_lb.config import Config
    from vllm_lb.hub import Hub
    from vllm_lb.proxy import Proxy
    from tests.fleet_files import gb10_pair
    cfg = Config(vllm_boxes=(), 
                 cluster_hosts=parse_hosts(HOSTS), cluster_url=HEAD, fleet=gb10_pair(),
                 listen_host="127.0.0.1", listen_port=0)
    hub = Hub(cfg)
    hub.set_cluster({"configured": True, "state": "running",
                     "effective": {"model_choice": choice, "num_speculative_tokens": 4,
                                   "max_num_batched_tokens": budget},
                     "boxes": {"box-a": {"gpu": gpu_view(BOX_A)}, "box-b": {"gpu": gpu_view(BOX_A)}}})
    for at, scrape in reads:
        hub.update_backend(HEAD, live=True, last_probe_at=at, mono_at=at, depth_known=True,
                           waiting=scrape.get("waiting") or 0, running=scrape.get("running") or 0,
                           served_models=("Qwen/Qwen3.8-27B-SGLang",), max_model_len=262144, scrape=scrape)
    return Proxy(cfg, hub).status()
