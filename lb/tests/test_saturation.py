"""IS THERE HEADROOM — and the three ways this panel is allowed to be wrong, each of which has a test.

The operator asked for "real GPU usage, GPU cores fill rate, teraflops/s rate in real time, so we can
indicate if any process can still be optimised or is fully saturated already", after a 96 % GPU tile
had already misled them into reading a memory-bound decode as a full card. So the failures this file
exists to catch are not crashes:

  1. calling an SM fill a share of peak FLOP/s (`sm_fill_means`, and no COMPUTE_BOUND without a
     measured compute ceiling);
  2. printing 0 TFLOP/s when the engine simply is not measuring — vLLM REGISTERS the MFU counters
     whatever it was started with and feeds them only under `--enable-mfu-metrics`, so a counter still
     at 0 after the engine produced tokens is a switch, not a measurement (and, the other way round, a
     counter flat for one window while a long step runs is NOT the switch:
     tests/test_lumpy_perf_counters.py);
  3. quoting `nvidia-smi`'s `mem` as memory bandwidth. On a GB10 it reads 0 at `sm 96 %` (measured on
     box-a and box-b, 2026-09-29), so it is carried as `mem_duty_pct` and judges nothing.

Nothing here needs SSH, Docker, a GPU or a broker.
"""
from __future__ import annotations

import json

import pytest

from vllm_lb.probe import SCRAPE_FAMILIES, parse_vllm_metrics
from vllm_lb.rates import MFU_NOT_MEASURED, NOT_EXPOSED, WHY
from vllm_lb.saturation import (BANDWIDTH_BOUND, BUSY_UNMEASURED, COMPUTE_BOUND, IDLE, MFU_OFF,
                               UNKNOWN, VERDICTS, gpu_view, judge)
from vllm_lb.state import CLUSTER, VLLM
from vllm_lb.throughput import Throughput
from tests.fleet import GB10_PEAK_GBPS

#: One GB10 under a serving pair, exactly as `tp-rank status` reported box-a at 2026-09-29 19:0x:
#: SM pinned, the memory duty-cycle flat at 0, the clock 81 % of its own ceiling.
BOX_A_GPU = {"count": 1, "sm_pct": 96, "mem_duty_pct": 0, "pwr_w": 57, "gtemp_c": 71,
             "pclk_mhz": 2424, "pclk_max_mhz": 3003}
BOX_B_GPU = {"count": 1, "sm_pct": 91, "mem_duty_pct": 0, "pwr_w": 52, "gtemp_c": 72,
              "pclk_mhz": 2470, "pclk_max_mhz": 3003}
#: The GB10's published LPDDR5X peak, per GPU — config's default.
PEAK_GBPS = 273.0


def boxes(*raw) -> list[dict]:
    return [dict(gpu_view(g), box=f"box-a{i or ''}") for i, g in enumerate(raw)]


# ── one box's GPU, as the wire hands it over ───────────────────────────────────

def test_a_gpu_reading_keeps_only_numbers_and_adds_the_clocks_own_share():
    g = gpu_view(BOX_A_GPU)
    assert g["sm_pct"] == 96.0 and g["pwr_w"] == 57.0 and g["gtemp_c"] == 71.0
    assert g["mem_duty_pct"] == 0.0, "carried as measured — and never renamed bandwidth"
    # 2424 / 3003: the number that makes 2424 MHz readable as normal rather than as throttling.
    assert g["clock_pct"] == 80.7


def test_a_box_that_reported_no_gpu_is_none_never_a_zeroed_reading():
    """The `hw.py` rule: a GB10 broadcasting `vram_used_mb: 0` must not become a "0 GB VRAM" tile. An
    unobserved GPU is the same case — no nvidia-smi, telemetry off, a driver that did not answer."""
    assert gpu_view(None) is None
    assert gpu_view({}) is None
    assert gpu_view({"pwr_w": 57}) is None, "a GPU object with no SM fill in it observed nothing"


def test_a_non_number_is_dropped_rather_than_coerced():
    g = gpu_view({"sm_pct": 96, "pwr_w": "[N/A]", "gtemp_c": True, "pclk_mhz": None})
    assert g["sm_pct"] == 96.0
    assert g["pwr_w"] is None and g["gtemp_c"] is None, "a bool is not a temperature"
    assert g["clock_pct"] is None, "no clock, so no share of its ceiling"


# ── the verdict ───────────────────────────────────────────────────────────────

def judged(**over) -> dict:
    kwargs = {"gpus": boxes(BOX_A_GPU, BOX_B_GPU), "tflops_per_gpu": None, "gbps_per_gpu": None,
              "peak_tflops": None, "peak_gbps": PEAK_GBPS, "gpu_count": 2}
    return judge(**{**kwargs, **over})


def test_every_verdict_is_from_the_closed_set_and_the_block_is_json():
    for over in ({}, {"gbps_per_gpu": 220.0}, {"gpus": []},
                 {"gpus": boxes({**BOX_A_GPU, "sm_pct": 3}, {**BOX_B_GPU, "sm_pct": 2})},
                 {"tflops_per_gpu": 40.0, "peak_tflops": 50.0}):
        v = judged(**over)
        assert v["verdict"] in VERDICTS
        assert json.loads(json.dumps(v)) == v, "the whole block is what the page is sent"


def test_a_96_percent_sm_fill_alone_is_never_called_compute_saturated():
    """THE DEFECT THIS PANEL REPLACES. Both GB10 boxes are pinned and the engine measures no ceiling — its
    switch PROVEN off (tokens out, counter still 0): the only honest verdict names the missing
    measurement, and the page must be able to print it. (A counter merely flat for one window is not
    that proof: tests/test_lumpy_perf_counters.py.)"""
    v = judged(flops_why=MFU_NOT_MEASURED, bytes_why=MFU_NOT_MEASURED)
    assert v["verdict"] == BUSY_UNMEASURED
    assert v["verdict"] != COMPUTE_BOUND
    assert MFU_OFF in v["because"] and "--enable-mfu-metrics" in v["because"]
    assert v["headroom"] is None, "not knowing is not the same as 'no headroom'"
    assert v["sm_fill_pct"] == 96.0, "the BUSIER box: a pair is as saturated as its busier half"
    assert "not a share of peak FLOP/s" in v["sm_fill_means"]


def test_bandwidth_near_the_boxs_peak_is_the_verdict_and_names_the_compute_left_over():
    """The measured shape of this fleet: memory-bandwidth-bound with compute to spare. The verdict has
    to SAY that, or the operator cuts concurrency looking for compute they already have."""
    v = judged(gbps_per_gpu=220.0, tflops_per_gpu=12.0, peak_tflops=100.0)
    assert v["verdict"] == BANDWIDTH_BOUND and v["headroom"] is False
    assert "81 % of the 273 GB/s" in v["because"]
    assert "compute headroom exists" in v["because"]
    assert v["bandwidth"] == {"rate": 220.0, "peak": 273.0, "pct": 80.6, "why": None}
    assert v["cluster_gbps"] == 440.0 and v["cluster_tflops"] == 24.0


def test_bandwidth_bound_still_reports_the_compute_rate_when_no_compute_peak_is_configured():
    """Half the answer is useless: "bandwidth bound" alone leaves the operator guessing what compute is
    doing. The FLOP/s goes in the same sentence — as a bare rate, with the share explicitly NOT claimed."""
    v = judged(gbps_per_gpu=220.0, tflops_per_gpu=12.0, peak_tflops=None)
    assert v["verdict"] == BANDWIDTH_BOUND
    assert "12.0 TFLOP/s per GPU" in v["because"]
    assert "no compute peak is configured" in v["because"]
    assert "compute headroom exists" not in v["because"], "not claimed without a peak to claim it from"


def test_compute_is_only_called_saturated_against_a_peak_that_was_configured():
    """`gpu_peak_tflops` defaults to 0 on purpose: no dense peak for this checkpoint has been measured
    on this fleet, and a guessed one would recreate the very claim this panel replaces."""
    without = judged(tflops_per_gpu=95.0, peak_tflops=None)
    assert without["verdict"] != COMPUTE_BOUND
    assert without["compute"]["pct"] is None and without["compute"]["rate"] == 95.0
    assert "no peak configured" in without["because"]
    with_peak = judged(tflops_per_gpu=95.0, peak_tflops=100.0)
    assert with_peak["verdict"] == COMPUTE_BOUND and with_peak["headroom"] is False


def test_neither_ceiling_reached_reads_as_headroom_in_so_many_words():
    v = judged(gbps_per_gpu=90.0, tflops_per_gpu=10.0, peak_tflops=100.0)
    assert v["verdict"] == BUSY_UNMEASURED and v["headroom"] is True
    assert "more concurrency should still buy throughput" in v["because"]


def test_an_idle_pair_says_so_and_claims_nothing_about_a_ceiling():
    v = judged(gpus=boxes({**BOX_A_GPU, "sm_pct": 4}, {**BOX_B_GPU, "sm_pct": 0}))
    assert v["verdict"] == IDLE and v["headroom"] is True and v["sm_fill_pct"] == 4.0


def test_no_box_reporting_a_gpu_is_not_observed_and_not_idle():
    v = judged(gpus=[])
    assert v["verdict"] == UNKNOWN and v["headroom"] is None and v["sm_fill_pct"] is None


def test_the_memory_duty_cycle_never_becomes_the_bandwidth_verdict():
    """On a GB10 `nvidia-smi`'s `mem` reads 0 at `sm 96 %` — the memory is unified with the host's.
    Even at 100 it decides nothing: only the engine's own byte counters can say GB/s."""
    pinned = boxes({**BOX_A_GPU, "mem_duty_pct": 100}, {**BOX_B_GPU, "mem_duty_pct": 100})
    assert judged(gpus=pinned)["verdict"] != BANDWIDTH_BOUND
    assert judged(gpus=pinned)["bandwidth"]["rate"] is None


# ── the rates, off the engine's own counters ───────────────────────────────────

FAMILIES = {key for key, _, _ in SCRAPE_FAMILIES.values()}
#: What the head rank really exposes. Read off http://192.0.2.2:8100/metrics, 2026-09-29: the three
#: counters ARE registered and ARE flat, because the engine was started without --enable-mfu-metrics.
REAL_METRICS = """\
# HELP vllm:estimated_flops_per_gpu_total Estimated number of floating point operations per GPU.
# TYPE vllm:estimated_flops_per_gpu_total counter
vllm:estimated_flops_per_gpu_total{engine="0",model_name="Qwen/Qwen3.8-Flash-Next"} 0.0
vllm:estimated_read_bytes_per_gpu_total{engine="0",model_name="Qwen/Qwen3.8-Flash-Next"} 0.0
vllm:estimated_write_bytes_per_gpu_total{engine="0",model_name="Qwen/Qwen3.8-Flash-Next"} 0.0
vllm:num_requests_running{engine="0",model_name="Qwen/Qwen3.8-Flash-Next"} 3.0
"""


def test_the_three_engine_counters_are_scraped_off_the_one_scrape_the_router_already_takes():
    """No new call anywhere: the families ride the same /metrics read that feeds queue depth."""
    assert {"est_flops_total", "est_read_bytes_total", "est_write_bytes_total"} <= FAMILIES
    got = parse_vllm_metrics(REAL_METRICS)
    assert got["est_flops_total"] == 0.0 and got["est_read_bytes_total"] == 0.0
    assert got["est_write_bytes_total"] == 0.0 and got["running"] == 3.0


def feed(t: Throughput, at: float, **families) -> None:
    scrape = dict.fromkeys(FAMILIES)
    scrape.update({"running": 0, **families})
    t.observe(live=True, scrape=scrape, mono_at=at, at=at)


def test_a_flat_counter_while_the_engine_has_work_is_the_switch_being_off_not_a_measured_zero():
    """REGRESSION for the one number this panel could most easily invent. The counters exist and stay
    at 0 without `--enable-mfu-metrics`; a rate of 0 there would read as "the GPUs did no work" while
    three requests were being served. The proof is the engine's own: it generated 300 tokens — steps
    completed — and its FLOP counter never left 0."""
    t = Throughput(CLUSTER, interval_s=5.0)
    feed(t, 0.0, running=3, est_flops_total=0.0, est_read_bytes_total=0.0, est_write_bytes_total=0.0,
         gen_tokens_total=100)
    feed(t, 5.0, running=3, est_flops_total=0.0, est_read_bytes_total=0.0, est_write_bytes_total=0.0,
         gen_tokens_total=400)
    perf = t.engine_perf()
    assert perf["tflops_per_gpu"] is None and perf["flops_why"] == MFU_NOT_MEASURED
    assert perf["gbps_per_gpu"] is None and perf["bytes_why"] == MFU_NOT_MEASURED
    assert MFU_NOT_MEASURED in WHY, "the vocabulary is closed, and this is in it"


def test_a_switch_proven_off_is_said_at_rest_too_never_a_measured_zero():
    """site-b, 2026-10-11: a vLLM started without --enable-mfu-metrics that HAD served (7,895 tokens out, every
    estimator counter still exactly 0.0) read `compute / GPU 0.0 TFLOPS` whenever it was idle — a "measured" zero
    from a counter that was never counting. The evidence that proves the switch off under load proves it at rest."""
    t = Throughput(CLUSTER, interval_s=5.0)
    for at in (0.0, 5.0):
        feed(t, at, running=0, est_flops_total=0.0, est_read_bytes_total=0.0, est_write_bytes_total=0.0,
             gen_tokens_total=7895, prompt_tokens_total=48504)
    perf = t.engine_perf()
    assert perf["mfu_metrics"] == "off"
    assert perf["tflops_per_gpu"] is None and perf["flops_why"] == MFU_NOT_MEASURED
    assert perf["gbps_per_gpu"] is None and perf["bytes_why"] == MFU_NOT_MEASURED


def test_a_flat_counter_with_nothing_running_is_a_measured_zero_and_stays_one():
    """Idle is a fact. Only a flat counter WHILE the engine had work is a switch being off."""
    t = Throughput(CLUSTER, interval_s=5.0)
    feed(t, 0.0, running=0, est_flops_total=0.0, est_read_bytes_total=0.0, est_write_bytes_total=0.0)
    feed(t, 5.0, running=0, est_flops_total=0.0, est_read_bytes_total=0.0, est_write_bytes_total=0.0)
    perf = t.engine_perf()
    assert perf["tflops_per_gpu"] == 0.0 and perf["flops_why"] is None
    assert perf["gbps_per_gpu"] == 0.0 and perf["bytes_why"] is None


def test_the_rates_are_vllms_own_arithmetic_per_gpu():
    """vLLM's own PromQL for these counters (v1/metrics/perf.py): `rate(flops)/1e12` TFLOP/s, and
    `(rate(read) + rate(write))/1e9` GB/s. The divisors are the engine's, not ours."""
    t = Throughput(CLUSTER, interval_s=5.0)
    feed(t, 0.0, running=4, est_flops_total=0.0, est_read_bytes_total=0.0, est_write_bytes_total=0.0)
    feed(t, 5.0, running=4, est_flops_total=6e13,          # 60 TFLOP over 5 s = 12 TFLOP/s
         est_read_bytes_total=1.0e12, est_write_bytes_total=1.0e11)   # 1.1 TB over 5 s = 220 GB/s
    perf = t.engine_perf()
    assert perf["tflops_per_gpu"] == pytest.approx(12.0)
    assert perf["gbps_per_gpu"] == pytest.approx(220.0)
    assert perf["window_s"] == pytest.approx(5.0)


def test_a_server_that_does_not_expose_the_counters_says_so_rather_than_nothing():
    """`--disable-log-stats`, or an engine older than the metric: not exposed is its own reason, and a
    different one from "not measuring"."""
    t = Throughput(CLUSTER, interval_s=5.0)
    feed(t, 0.0, running=2)
    feed(t, 5.0, running=2)
    perf = t.engine_perf()
    assert perf["tflops_per_gpu"] is None and perf["flops_why"] == NOT_EXPOSED
    assert perf["gbps_per_gpu"] is None and perf["bytes_why"] == NOT_EXPOSED


# ── the wiring: /lb/status carries it, off what is already read every tick ─────

HEAD = "http://192.0.2.2:8100"
HOSTS = "box-a-gpu=box-a=user@10.0.0.1=10.9.200.1,box-b-gpu=box-b=user@10.0.0.2=10.9.200.2"


def fleet(**over):
    """A real Hub and a real Proxy, fed the way the loops feed them."""
    from vllm_lb.cluster import parse_hosts
    from vllm_lb.config import Config
    from vllm_lb.hub import Hub
    from vllm_lb.proxy import Proxy
    from tests.fleet_files import gb10_pair
    # The pair's cards are GB10s (the fleet names them): their type states the 273 GB/s bus.
    cfg = Config(vllm_boxes=(), fleet=gb10_pair(),
                 cluster_hosts=parse_hosts(HOSTS), cluster_url=HEAD, listen_host="127.0.0.1", listen_port=0, **over)
    hub = Hub(cfg)
    return cfg, hub, Proxy(cfg, hub)


def published(hub, box_a_gpu, box_b_gpu) -> None:
    hub.set_cluster({"configured": True, "state": "running",
                     "boxes": {"box-a": {"gpu": box_a_gpu}, "box-b": {"gpu": box_b_gpu}}})


def probe(hub, at: float, **families) -> None:
    scrape = dict.fromkeys(FAMILIES)
    scrape.update({"waiting": 0, "running": 0, **families})
    hub.update_backend(HEAD, live=True, last_probe_at=at, mono_at=at, depth_known=True,
                       waiting=0, running=scrape["running"] or 0,
                       served_models=("Qwen/Qwen3.8-Flash-Next",), max_model_len=262144,
                       scrape=scrape)


def test_lb_status_carries_the_verdict_built_from_the_boxes_and_the_engines_own_counters():
    """END TO END over the real Proxy.status(): the per-box GPU comes from `tp-rank status` through the
    cluster loop's published view, and the rates from the SAME /metrics scrape the router takes."""
    cfg, hub, proxy = fleet()
    published(hub, gpu_view(BOX_A_GPU), gpu_view(BOX_B_GPU))
    probe(hub, 0.0, running=4, est_flops_total=0.0, est_read_bytes_total=0.0,
          est_write_bytes_total=0.0)
    probe(hub, 5.0, running=4, est_flops_total=6e13, est_read_bytes_total=1.0e12,
          est_write_bytes_total=1.0e11)
    sat = proxy.status()["saturation"]
    assert sat["verdict"] == BANDWIDTH_BOUND
    assert sat["bandwidth"]["rate"] == pytest.approx(220.0, abs=0.1)
    assert sat["bandwidth"]["peak"] == PEAK_GBPS, "the GB10's published peak, from config"
    assert sat["compute"]["rate"] == pytest.approx(12.0, abs=0.1)
    assert sat["compute"]["pct"] is None, "no dense peak has been measured on this fleet"
    assert sat["sm_fill_pct"] == 96.0
    assert [b["box"] for b in sat["boxes"]] == ["box-a", "box-b"]
    assert sat["gpu_count"] == 2 and sat["cluster_gbps"] == pytest.approx(440.0, abs=0.2)


def test_an_lb_with_no_cluster_has_no_saturation_block_at_all():
    """There is nothing whose saturation this would be — and an empty verdict would read as one."""
    from vllm_lb.config import Config
    from vllm_lb.hub import Hub
    from vllm_lb.proxy import Proxy
    cfg = Config(vllm_boxes=(), listen_host="127.0.0.1", listen_port=0)
    assert Proxy(cfg, Hub(cfg)).status()["saturation"] is None


def test_the_live_shape_of_this_fleet_today_says_the_switch_is_off_not_that_the_gpus_are_free():
    """What the panel showed on the cluster as it ran on 2026-09-29: both GB10 boxes pinned at ~96 % SM,
    serving — tokens coming out — and the three MFU counters registered and still 0 because the engine
    was started without --enable-mfu-metrics (read off http://192.0.2.2:8100/metrics). The verdict
    must name the missing switch."""
    cfg, hub, proxy = fleet()
    published(hub, gpu_view(BOX_A_GPU), gpu_view(BOX_B_GPU))
    for at, generated in ((0.0, 1200), (5.0, 1500)):
        probe(hub, at, running=3, est_flops_total=0.0, est_read_bytes_total=0.0,
              est_write_bytes_total=0.0, gen_tokens_total=generated)
    sat = proxy.status()["saturation"]
    assert sat["verdict"] == BUSY_UNMEASURED and sat["headroom"] is None
    assert sat["compute"]["why"] == MFU_NOT_MEASURED
    assert sat["bandwidth"]["why"] == MFU_NOT_MEASURED
    assert "--enable-mfu-metrics" in sat["because"]
    assert json.loads(json.dumps(sat)) == sat


# ── the engine's byte estimate can be INCOMPLETE ──────────────────────────────
# vLLM drops a component whose quantisation it cannot size. This checkpoint is
# `modelopt_mixed`, which its map has no entry for, so the MoE FFN is excluded and the
# reported byte rate is ~1 % of peak while the SMs are 88 % busy (measured 2026-09-29).

def test_a_busy_pair_with_a_near_zero_byte_rate_claims_no_bandwidth_share():
    """The pre-fix symptom, as it actually read: BOTH rates near zero at busy SMs (2.6 GB/s and
    ~0.1 TFLOP/s at 88 % fill). A low byte rate with real FLOP/s is ordinary prefill - see
    test_ordinary_prefill_with_a_low_byte_rate_is_not_called_incomplete."""
    out = judge(gpus=[{"sm_pct": 88.0}], tflops_per_gpu=0.1, gbps_per_gpu=2.6,
                           peak_tflops=None, peak_gbps=273.0, gpu_count=2)
    assert out["bandwidth"]["rate"] == 2.6, "the raw rate stays visible"
    assert out["bandwidth"]["pct"] is None, "1 % of peak must NOT be presented as a real share"
    assert "quantised weights" in (out["bandwidth"]["why"] or "")
    assert out["verdict"] != BANDWIDTH_BOUND
    assert out["headroom"] is not False, "must not claim saturation from an incomplete figure"


def test_the_incomplete_texts_name_no_architecture():
    """The same sentences are shown for EVERY model in the catalog: the dense 27B has no MoE weights, and
    what vLLM's MFU model drops is a quantisation it has no entry for — whatever the layer is."""
    from vllm_lb.saturation import BYTES_INCOMPLETE, FLOPS_INCOMPLETE
    for text in (BYTES_INCOMPLETE, FLOPS_INCOMPLETE):
        assert "MoE" not in text and "this model's quantised weights" in text


def test_a_believable_bandwidth_share_is_still_reported():
    out = judge(gpus=[{"sm_pct": 88.0}], tflops_per_gpu=None, gbps_per_gpu=220.0,
                           peak_tflops=None, peak_gbps=273.0, gpu_count=2)
    assert out["bandwidth"]["pct"] == 80.6
    assert out["verdict"] == BANDWIDTH_BOUND



def test_the_sentence_never_blames_a_missing_peak_when_one_is_configured():
    """Screenshot 2026-09-30 showed '14 GB/s ... no peak configured to judge it' while
    LB_GPU_PEAK_GBPS=273 WAS configured. The share was missing because the LB withheld it."""
    out = judge(gpus=[{"sm_pct": 95.0}], tflops_per_gpu=0.0, gbps_per_gpu=14.0,
                peak_tflops=None, peak_gbps=273.0, gpu_count=2)
    assert out["bandwidth"]["peak"] == 273.0, "the peak IS configured"
    assert out["bandwidth"]["pct"] is None
    assert "no peak configured" not in out["because"], out["because"]
    assert "quantised weights" in out["because"]


def test_a_busy_pair_computing_nothing_is_also_flagged_incomplete():
    out = judge(gpus=[{"sm_pct": 95.0}], tflops_per_gpu=0.0, gbps_per_gpu=14.0,
                peak_tflops=None, peak_gbps=273.0, gpu_count=2)
    assert out["compute"]["rate"] == 0.0, "the raw rate stays visible"
    assert "quantised weights" in (out["compute"]["why"] or ""), out["compute"]
    assert "floor" in (out["compute"]["why"] or "")


def test_a_genuinely_unconfigured_peak_still_says_so():
    out = judge(gpus=[{"sm_pct": 95.0}], tflops_per_gpu=12.0, gbps_per_gpu=220.0,
                peak_tflops=None, peak_gbps=273.0, gpu_count=2)
    assert out["compute"]["why"] is None
    assert "no compute peak is configured" in out["because"], out["because"]


# ── a few minutes of history, so the panel shows a trend not an instant ───────

def test_history_keeps_a_window_and_drops_the_oldest():
    from vllm_lb.saturation import History
    h = History(keep=3, min_interval_s=0.0)
    for i in range(5):
        v = h.observe(100.0 + i, float(i), float(i * 10), float(i * 2))
    assert v["at"] == [102.0, 103.0, 104.0], v["at"]
    assert v["sm_fill_pct"] == [2.0, 3.0, 4.0]
    assert v["gbps"] == [20.0, 30.0, 40.0]
    assert v["tflops"] == [4.0, 6.0, 8.0]


def test_history_does_not_sample_faster_than_its_interval():
    from vllm_lb.saturation import History
    h = History(keep=60, min_interval_s=4.0)
    h.observe(100.0, 90.0, 200.0, 10.0)
    h.observe(101.0, 91.0, 201.0, 11.0)   # too soon — ignored
    v = h.observe(105.0, 92.0, 202.0, 12.0)
    assert v["at"] == [100.0, 105.0], "a fast poller must not inflate the series"
    assert v["sm_fill_pct"] == [90.0, 92.0]


def test_history_keeps_an_unmeasured_sample_as_a_gap_not_a_zero():
    from vllm_lb.saturation import History
    h = History(keep=10, min_interval_s=0.0)
    h.observe(1.0, 90.0, 200.0, 10.0)
    v = h.observe(2.0, None, None, None)
    assert v["gbps"] == [200.0, None], "an unmeasured rate must not become 0"
    assert v["tflops"] == [10.0, None]



def test_ordinary_prefill_with_a_low_byte_rate_is_not_called_incomplete():
    """Prefill is compute-heavy and light on bytes: measured 2026-09-30 at ~27 GB/s (9.9 % of 273)
    beside ~50 TFLOP/s per GPU at 96 % SM fill. The old guard read the low byte rate alone as a
    dropped MoE component and withheld the share - a false claim once the quantisation fix was in."""
    out = judge(gpus=[{"sm_pct": 96.0}], tflops_per_gpu=50.0, gbps_per_gpu=27.0,
                peak_tflops=323.2, peak_gbps=273.0, gpu_count=2)
    assert out["bandwidth"]["pct"] == 9.9, out["bandwidth"]
    assert "quantised weights" not in (out["bandwidth"]["why"] or "")
    assert "quantised weights" not in (out["compute"]["why"] or "")
    assert out["compute"]["pct"] == 15.5
