"""COMPUTE AND MEMORY BANDWIDTH, PER BOX USED — each serving box's rate against ITS OWN card's peak.

Operator, 2026-10-11: GPU peaks belong to each BOX and its card(s), never to the LB as a whole — one LB-wide
`LB_GPU_PEAK_*` claim cannot describe site-b (a 3090 beside the 5090 that serves) or a GB10 box pair next to a 5090.
So the ceilings are the serving cards' hardware types' (vllm_lb/hardware/types, named by the fleet file), and the
TP=2 pair shows box-a and box-b SEPARATELY, each against its own box's peak — never one combined "% of 323".

THE DERIVATION, never presented as a measurement: the engine reports ONE rate for the whole pair — vLLM's
`estimated_*_per_gpu` off the head rank (rank 1 runs headless, with no /metrics), the LB's own model for SGLang —
and no per-rank counter. Tensor parallelism splits every layer's work evenly over the ranks, so a box's share is the
per-GPU rate times the serving cards it holds. The per-box rates add back up to the engine-wide rate (pinned below).
"""
from __future__ import annotations

import pytest

from tests.fleet_files import ENV_OF_PAIR, gb10_pair, write
from vllm_lb.config import RETIRED_KEYS, Config
from vllm_lb.saturation import DERIVED_PER_BOX, gpu_view, per_box

GB10_TFLOPS, GB10_GBPS = 323.2, 273      # vllm_lb/hardware/types/gb10.yml: nvfp4 (measured), LPDDR5X (spec)


# ── the split, as arithmetic ────────────────────────────────────────────────────────────────────────

def test_the_pair_is_two_boxes_each_against_its_own_gb10():
    boxes = per_box(12.0, [("box-a", GB10_TFLOPS, None), ("box-b", GB10_TFLOPS, None)])
    assert [(b["box"], b["cards"], b["rate"], b["peak"], b["pct"]) for b in boxes] == \
        [("box-a", 1, 12.0, 323.2, 3.7), ("box-b", 1, 12.0, 323.2, 3.7)]
    assert all(b["derived"] == DERIVED_PER_BOX for b in boxes), "a box's share of one engine-wide rate is derived"


def test_the_box_rates_add_back_up_to_the_engine_wide_rate():
    per_gpu, cards = 12.34, [("box-a", GB10_TFLOPS, None), ("box-b", GB10_TFLOPS, None)]
    assert sum(b["rate"] for b in per_box(per_gpu, cards)) == pytest.approx(per_gpu * len(cards), abs=0.11)


def test_two_different_cards_are_two_different_shares():
    boxes = per_box(100.0, [("a", GB10_GBPS, None), ("b", 1792, None)])
    assert [b["pct"] for b in boxes] == [36.6, 5.6], "the same rate is 37 % of a GB10's bus and 6 % of a 5090's"


def test_a_box_serving_on_two_cards_holds_two_shares_against_both_peaks():
    (box,) = per_box(10.0, [("solo", 209.5, None), ("solo", 209.5, None)])
    assert (box["cards"], box["rate"], box["peak"], box["pct"]) == (2, 20.0, 419.0, 4.8)


def test_one_card_alone_is_the_engines_own_figure_not_a_derived_share():
    (box,) = per_box(12.0, [("solo", 1676, None)])
    assert box["derived"] is None and box["rate"] == 12.0


def test_a_card_whose_type_states_no_peak_claims_no_share_and_says_why():
    (box,) = per_box(5.0, [("solo", None, "NVIDIA lists N/A")])
    assert (box["rate"], box["peak"], box["pct"], box["why"]) == (5.0, None, None, "NVIDIA lists N/A")


def test_no_rate_is_no_share():
    (box,) = per_box(None, [("solo", 1676, None)])
    assert (box["rate"], box["pct"]) == (None, None)


def test_a_share_the_lb_withheld_stays_withheld_on_every_box():
    boxes = per_box(2.0, [("box-a", GB10_GBPS, None), ("box-b", GB10_GBPS, None)], withheld="incomplete bytes")
    assert all(b["pct"] is None and b["why"] == "incomplete bytes" for b in boxes)


# ── end to end: /lb/status off the real Proxy ──────────────────────────────────────────────────────

HEAD = "http://192.0.2.10:8100"
GPU = {"count": 1, "sm_pct": 96, "mem_duty_pct": 0, "pwr_w": 44, "gtemp_c": 71, "pclk_mhz": 2437,
       "pclk_max_mhz": 3003}


def _sat(**effective) -> dict:
    from vllm_lb.cluster import parse_hosts
    from vllm_lb.hub import Hub
    from vllm_lb.probe import SCRAPE_FAMILIES
    from vllm_lb.proxy import Proxy
    cfg = Config(vllm_boxes=(), fleet=gb10_pair(), cluster_url=HEAD,
                 cluster_hosts=parse_hosts("box-a-gpu=box-a=op@192.0.2.10=198.51.100.1,"
                                           "box-b-gpu=box-b=op@192.0.2.11=198.51.100.2"),
                 listen_host="127.0.0.1", listen_port=0)
    hub = Hub(cfg)
    hub.set_cluster({"configured": True, "state": "running", "effective": effective or None,
                     "boxes": {"box-a": {"gpu": gpu_view(GPU)}, "box-b": {"gpu": gpu_view(GPU)}}})
    for at, flops, read in ((0.0, 0.0, 0.0), (5.0, 6e13, 1.0e12)):
        scrape = dict.fromkeys(key for key, _, _ in SCRAPE_FAMILIES.values())
        scrape.update({"waiting": 0, "running": 4, "est_flops_total": flops, "est_read_bytes_total": read,
                       "est_write_bytes_total": read / 10})
        hub.update_backend(HEAD, live=True, last_probe_at=at, mono_at=at, depth_known=True, waiting=0, running=4,
                           served_models=("Qwen/Qwen3.8-Flash-Next",), max_model_len=262144, scrape=scrape)
    return Proxy(cfg, hub).status()["saturation"]


def test_the_pairs_compute_and_bandwidth_are_split_per_box_each_against_its_own_peak():
    sat = _sat(model_choice="qwen3.8-flash-next-hibrid48")
    compute, bandwidth = sat["compute"], sat["bandwidth"]
    assert [(b["box"], b["rate"], b["peak"]) for b in compute["boxes"]] == \
        [("box-a", 12.0, GB10_TFLOPS), ("box-b", 12.0, GB10_TFLOPS)]
    assert [(b["box"], b["peak"]) for b in bandwidth["boxes"]] == [("box-a", GB10_GBPS), ("box-b", GB10_GBPS)]
    assert [b["pct"] for b in compute["boxes"]] == [3.7, 3.7]
    assert compute["peak_precision"] == "nvfp4", "the model's precision (cores.py), not an LB-wide setting"
    assert all(b["derived"] == DERIVED_PER_BOX for b in compute["boxes"] + bandwidth["boxes"])
    # The consistency check: the boxes' rates are the engine-wide rate, split — nothing added, nothing lost.
    assert sum(b["rate"] for b in compute["boxes"]) == pytest.approx(sat["cluster_tflops"], abs=0.11)
    assert sum(b["rate"] for b in bandwidth["boxes"]) == pytest.approx(sat["cluster_gbps"], abs=0.11)


def test_a_model_with_no_recorded_precision_claims_no_compute_share_and_says_why():
    sat = _sat()                                       # nothing launched: no model, so no precision
    assert all(b["peak"] is None and b["pct"] is None and b["why"] for b in sat["compute"]["boxes"])
    assert [b["peak"] for b in sat["bandwidth"]["boxes"]] == [GB10_GBPS, GB10_GBPS], "a bus needs no precision"


# ── the LB-wide peaks are retired ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("key, value", [("LB_GPU_PEAK_GBPS", "273"), ("LB_GPU_PEAK_TFLOPS", "323.2"),
                                        ("LB_GPU_PEAK_TFLOPS_PRECISION", "nvfp4")])
def test_an_lb_wide_peak_is_refused_naming_where_peaks_live_now(tmp_path, key, value):
    from tests.fleet_files import PAIR
    with pytest.raises(ValueError) as e:
        Config.from_env({"LB_ALLOWED_HOSTS": "loopback", "LB_FLEET": write(tmp_path, PAIR), key: value})
    assert key in str(e.value) and "hardware type" in RETIRED_KEYS[key]


def test_the_config_carries_no_lb_wide_peak():
    c = Config()
    for gone in ("gpu_peak_gbps", "gpu_peak_tflops", "gpu_peak_tflops_precision"):
        assert not hasattr(c, gone), gone
    assert set(ENV_OF_PAIR) < set(RETIRED_KEYS)


# ── a node the LB only observes: its model is the fleet's, so its compute peak is its card's at that precision ─────

def test_the_observed_nodes_ceilings_are_its_5090s_at_its_named_models_precision(tmp_path):
    """site-b's shape: the node runs on the 5090 (card 1), the 3090 beside it serves nothing; the fleet names the model
    (qwen3.8-27b-nvidia-nvfp4@uncensored, nvfp4 GEMMs), so COMPUTE reads the 5090's dense FP4 peak and MEMORY
    BANDWIDTH its GDDR7 — one box, the engine's own figure, nothing derived."""
    from tests.fleet_files import ONE_HOST
    from vllm_lb.hub import Hub
    from vllm_lb.probe import SCRAPE_FAMILIES
    from vllm_lb.proxy import Proxy
    cfg = Config.from_env({"LB_ALLOWED_HOSTS": "loopback", "LB_FLEET": write(tmp_path, ONE_HOST), "LB_LISTEN": "127.0.0.1:0"})
    hub = Hub(cfg)
    hub.set_cluster({"configured": True, "state": "running", "effective": None, "boxes": {"solo": {"gpu": None}}})
    for at, flops, read in ((0.0, 0.0, 0.0), (5.0, 6e13, 1.0e12)):
        scrape = dict.fromkeys(key for key, _, _ in SCRAPE_FAMILIES.values())
        scrape.update({"waiting": 0, "running": 4, "est_flops_total": flops, "est_read_bytes_total": read,
                       "est_write_bytes_total": read / 10})
        hub.update_backend(cfg.cluster_url, live=True, last_probe_at=at, mono_at=at, depth_known=True, waiting=0,
                           running=4, served_models=("qwen3.8-27b-uncensored",), max_model_len=131072, scrape=scrape)
    sat = Proxy(cfg, hub).status()["saturation"]
    (compute,), (bus,) = sat["compute"]["boxes"], sat["bandwidth"]["boxes"]
    assert (compute["box"], compute["peak"], compute["derived"]) == ("solo", 1676, None)
    assert sat["compute"]["peak_precision"] == "nvfp4"
    assert (bus["box"], bus["peak"], bus["derived"]) == ("solo", 1792, None)
