"""Two boxes, two readings — and the two readings that are NOT per box.

The operator, 2026-09-30: "since it's two boxes cluster, we see only one as consolidated, not two
separated". The panel drew ONE SM-fill meter, the busier box's, so an imbalance between the ranks —
one rank idle while the other works — could not be seen at all.

What each box MEASURES on its own (nvidia-smi dmon via `tp-rank status`, one ~1 s sample per box, both
asked in parallel every cluster tick): SM fill, power, SM clock against its own ceiling, temperature.
Those are shown per rank, each with its own few-minute history.

What is NOT per box: the TFLOP/s and GB/s come from `vllm:estimated_*_per_gpu_total`, which only the
head rank exposes (rank 1 runs --headless, with no API server and no /metrics) and which vLLM computes
ANALYTICALLY — the model's FLOPs and bytes for each step divided by the TP size (v1/metrics/perf.py,
`per_gpu` divides by tp_size). They are the same for both ranks by construction, and the block says so
instead of inventing a per-box figure.
"""
from __future__ import annotations

import json

import pytest

from vllm_lb.probe import SCRAPE_FAMILIES
from vllm_lb.saturation import History, gpu_view, judge

BOX_A_GPU = {"count": 1, "sm_pct": 96, "mem_duty_pct": 0, "pwr_w": 44, "gtemp_c": 71,
             "pclk_mhz": 2437, "pclk_max_mhz": 3003}
BOX_B_GPU = {"count": 1, "sm_pct": 81, "mem_duty_pct": 0, "pwr_w": 36, "gtemp_c": 66,
              "pclk_mhz": 2476, "pclk_max_mhz": 3003}
RANKS = [{"box": "box-a", "rank": 0, "role": "head"}, {"box": "box-b", "rank": 1, "role": "worker"}]


def per_box_judge(gpus, **over) -> dict:
    kwargs = {"gpus": gpus, "tflops_per_gpu": 4.0, "gbps_per_gpu": 200.0, "peak_tflops": 323.2,
              "peak_gbps": 273.0, "gpu_count": 2, "ranks": RANKS}
    return judge(**{**kwargs, **over})


# ── the block ──────────────────────────────────────────────────────────────────

def test_each_rank_is_its_own_entry_in_rank_order_with_what_its_box_measured():
    gpus = [dict(gpu_view(BOX_B_GPU), box="box-b"), dict(gpu_view(BOX_A_GPU), box="box-a")]
    v = per_box_judge(gpus)
    assert [(b["box"], b["rank"], b["role"]) for b in v["boxes"]] == [
        ("box-a", 0, "head"), ("box-b", 1, "worker")], "rank 0 first, whatever order they arrived in"
    head, worker = v["boxes"]
    assert (head["sm_pct"], head["pwr_w"], head["gtemp_c"], head["clock_pct"]) == (96.0, 44.0, 71.0, 81.2)
    assert (worker["sm_pct"], worker["pwr_w"], worker["gtemp_c"]) == (81.0, 36.0, 66.0)
    assert json.loads(json.dumps(v)) == v


def test_the_gap_between_the_ranks_is_stated_signed_rank0_minus_rank1():
    v = per_box_judge([dict(gpu_view(BOX_A_GPU), box="box-a"), dict(gpu_view(BOX_B_GPU), box="box-b")])
    assert v["sm_fill_gap_pts"] == 15.0, "96 − 81: rank 0 had work 15 points more of the time"
    flipped = per_box_judge([dict(gpu_view({**BOX_A_GPU, "sm_pct": 70}), box="box-a"),
                             dict(gpu_view(BOX_B_GPU), box="box-b")])
    assert flipped["sm_fill_gap_pts"] == -11.0


def test_a_rank_whose_box_reported_no_gpu_keeps_its_place_as_a_gap_never_a_zero():
    v = per_box_judge([dict(gpu_view(BOX_A_GPU), box="box-a")])
    assert [b["box"] for b in v["boxes"]] == ["box-a", "box-b"]
    worker = v["boxes"][1]
    assert worker["sm_pct"] is None and worker["why"], worker
    assert v["sm_fill_gap_pts"] is None, "a gap against an unobserved rank is not a number"
    assert v["sm_fill_pct"] == 96.0, "the verdict still reads the rank that was observed"


def test_the_engine_rates_are_labelled_as_one_analytical_per_gpu_figure_not_a_per_box_one():
    """No per-box FLOP/s or GB/s exists, so none may appear: the per-rank entries carry only what
    nvidia-smi measured on that box, and the block names where the rates come from and what they are."""
    v = per_box_judge([dict(gpu_view(BOX_A_GPU), box="box-a"), dict(gpu_view(BOX_B_GPU), box="box-b")],
                      rates_from="box-a")
    for entry in v["boxes"]:
        assert not [k for k in entry if any(w in k.lower() for w in ("flop", "gbps", "bandwidth"))], entry
    assert v["rates_from"] == "box-a"
    means = v["rates_means"].lower()
    assert "analytical" in means and "both ranks" in means and "tp" in means, v["rates_means"]


def test_the_rates_name_the_engine_that_published_them():
    """The sentence beside the two rates said "vLLM's" unconditionally. That was true of every
    generation this gateway ever ran until the pair first served SGLang on 2026-10-03 — and then it
    named the wrong engine on the one figure whose purpose is to say where the number came from
    (worklog/sglang-ui-audit.md C6). The rates are the engine's own estimate, so the sentence has to
    be the engine's own name."""
    from vllm_lb.saturation import rates_mean
    gpus = [dict(gpu_view(BOX_A_GPU), box="box-a"), dict(gpu_view(BOX_B_GPU), box="box-b")]
    assert per_box_judge(gpus, rates_from="box-a")["rates_means"].startswith("vLLM's"), "the default"
    sglang = per_box_judge(gpus, rates_from="box-a", engine="sglang")["rates_means"]
    assert sglang.startswith("SGLang's"), sglang
    assert "vLLM" not in sglang, "the other engine's name must not survive in it"
    # Everything else about the sentence is unchanged — it is one word, not a rewrite.
    assert sglang.split("'s", 1)[1] == rates_mean("vllm").split("'s", 1)[1]
    # An engine nobody has taught it is spelled as it arrives rather than guessed or dropped.
    assert rates_mean("tensorrt").startswith("tensorrt's")


# ── the history ────────────────────────────────────────────────────────────────

def test_history_keeps_each_boxs_sm_fill_aligned_with_the_shared_clock():
    h = History(keep=3, min_interval_s=0.0)
    h.observe(1.0, 96.0, 200.0, 10.0, boxes={"box-a": 96.0, "box-b": 93.0})
    h.observe(2.0, 90.0, 190.0, 9.0, boxes={"box-a": 90.0})                    # box-b not read
    h.observe(3.0, 88.0, 180.0, 8.0, boxes={"box-a": 88.0, "box-b": 80.0, "box-d": 50.0})
    v = h.observe(4.0, 70.0, 170.0, 7.0, boxes={"box-a": 70.0, "box-b": 71.0, "box-d": 52.0})
    assert v["at"] == [2.0, 3.0, 4.0]
    assert v["boxes"]["box-a"] == [90.0, 88.0, 70.0]
    assert v["boxes"]["box-b"] == [None, 80.0, 71.0], "not read is a gap, never a zero"
    assert v["boxes"]["box-d"] == [None, 50.0, 52.0], "a box that appears late has no past"
    assert v["sm_fill_pct"] == [90.0, 88.0, 70.0], "the pair's series is kept as before"


def test_history_per_box_obeys_the_same_minimum_interval():
    h = History(keep=60, min_interval_s=4.0)
    h.observe(100.0, 90.0, 200.0, 10.0, boxes={"box-a": 90.0, "box-b": 85.0})
    v = h.observe(101.0, 91.0, 201.0, 11.0, boxes={"box-a": 91.0, "box-b": 86.0})   # too soon
    assert v["boxes"] == {"box-a": [90.0], "box-b": [85.0]}


def test_a_box_that_stopped_reporting_is_dropped_once_its_samples_have_aged_out():
    """A box no longer passed at all (the pair was reconfigured) must not keep a row of gaps forever;
    a configured box whose GPU simply was not read IS passed, as None, and keeps its row."""
    h = History(keep=2, min_interval_s=0.0)
    h.observe(1.0, 90.0, None, None, boxes={"old": 40.0, "box-a": 90.0})
    h.observe(2.0, 91.0, None, None, boxes={"box-a": 91.0})
    v = h.observe(3.0, 92.0, None, None, boxes={"box-a": 92.0})
    assert "old" not in v["boxes"], "a box that no longer reports must not keep a row of gaps forever"


# ── end to end: /lb/status ────────────────────────────────────────────────────

#: The head's box name sorts AFTER the worker's on purpose: rank order is the configured host order
#: (head first, `cluster_hosts`), never the alphabet.
HOSTS = "zeta-gpu=zeta=user@10.0.0.1=10.9.200.1,alpha-gpu=alpha=user@10.0.0.2=10.9.200.2"
HEAD = "http://192.0.2.2:8100"


def fleet():
    from vllm_lb.cluster import parse_hosts
    from vllm_lb.config import Config
    from vllm_lb.hub import Hub
    from vllm_lb.proxy import Proxy
    cfg = Config(vllm_boxes=(), cluster_name="pair", cluster_hosts=parse_hosts(HOSTS),
                 cluster_url=HEAD, listen_host="127.0.0.1", listen_port=0)
    hub = Hub(cfg)
    return hub, Proxy(cfg, hub)


def probe(hub, at: float, **families) -> None:
    scrape = dict.fromkeys(key for key, _, _ in SCRAPE_FAMILIES.values())
    scrape.update({"waiting": 0, "running": 4, **families})
    hub.update_backend(HEAD, live=True, last_probe_at=at, mono_at=at, depth_known=True, waiting=0,
                       running=4, served_models=("Qwen/Qwen3.8-Flash-Next",), max_model_len=262144,
                       scrape=scrape)


def test_lb_status_shows_both_ranks_head_first_each_with_its_own_history():
    hub, proxy = fleet()
    hub.set_cluster({"configured": True, "state": "running",
                     "boxes": {"alpha": {"gpu": gpu_view(BOX_B_GPU)}, "zeta": {"gpu": gpu_view(BOX_A_GPU)}}})
    probe(hub, 0.0, est_flops_total=1e15, est_read_bytes_total=1e13, est_write_bytes_total=1e11,
          gen_tokens_total=10)
    probe(hub, 5.0, est_flops_total=1.02e15, est_read_bytes_total=1e13 + 1e12,
          est_write_bytes_total=1.1e11, gen_tokens_total=500)
    sat = proxy.status()["saturation"]
    assert [(b["box"], b["rank"], b["role"]) for b in sat["boxes"]] == [("zeta", 0, "head"),
                                                                        ("alpha", 1, "worker")]
    assert sat["sm_fill_gap_pts"] == 15.0
    assert sat["history"]["boxes"] == {"zeta": [96.0], "alpha": [81.0]}
    assert sat["rates_from"] == "zeta", "the rates are read off the HEAD rank's /metrics"
    assert json.loads(json.dumps(sat)) == sat


def test_lb_status_keeps_an_unobserved_rank_in_its_place():
    hub, proxy = fleet()
    hub.set_cluster({"configured": True, "state": "running",
                     "boxes": {"alpha": {"gpu": None}, "zeta": {"gpu": gpu_view(BOX_A_GPU)}}})
    sat = proxy.status()["saturation"]
    assert [b["box"] for b in sat["boxes"]] == ["zeta", "alpha"]
    assert sat["boxes"][1]["sm_pct"] is None
    assert sat["history"]["boxes"]["alpha"] == [None]
