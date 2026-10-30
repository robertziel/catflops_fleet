"""The real-traffic performance log (perflog.py): minutes the cluster served real requests, recorded
by the LB off the probe it already takes — what a window is, what is kept, what is never written, and
that nothing about it can stop the LB.

The reads here are shaped like the head engine's real ones: probe.parse_vllm_metrics keys, including
the TTFT / inter-token histograms as ((le, cumulative), ...) with +Inf last, and a cluster view as the
cluster loop publishes it (`configured`, `desired`, `boxes[box].gpu`)."""
from __future__ import annotations

import http.client
import json
import math
import os

import pytest

from vllm_lb import perflog
from vllm_lb.config import Config, VllmBox, parse_hosts
from vllm_lb.control import Controller
from vllm_lb.hub import Hub
from vllm_lb.perflog import (CONFIG_KEYS, EFFECTIVE, FALLBACK, JsonlWriter, PerfLog, PerfRecorder,
                             WindowAccumulator, _plain, effective_config, quantile, throughput_target,
                             top_decile)
from vllm_lb.probe import HISTOGRAM_BUCKETS, SCRAPE_FAMILIES, parse_vllm_metrics
from vllm_lb.proxy import Proxy
from vllm_lb.state import CLUSTER

TTFT_LE = (0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, math.inf)
ITL_LE = (0.025, 0.05, 0.075, 0.1, 0.2, math.inf)
T0 = 1_790_740_020.0          # 2026-09-30 05:47:00 UTC — a minute boundary (T0 % 60 == 0)


def desired(gen=52, seqs=8):
    return {"mode": "cluster", "generation": gen, "max_num_seqs": seqs, "max_model_len": 262144,
            "model_choice": "qwen3.8-flash-next-hibrid48"}


def cluster(gen=52, seqs=8, effective=None, sm=(88.0, 86.0), pwr=(61.0, 58.0)):
    view = {"configured": True, "desired": desired(gen, seqs), "state": "running",
            "boxes": {"box-a": {"gpu": {"sm_pct": sm[0], "pwr_w": pwr[0]}},
                      "box-b": {"gpu": {"sm_pct": sm[1], "pwr_w": pwr[1]}}}}
    if effective is not None:
        view["effective"] = effective
    return view


class Engine:
    """Counters that only go up, read the way probe.parse_vllm_metrics returns them."""

    def __init__(self, start=1_790_739_346.72):
        self.start = start
        self.c = dict.fromkeys(("gen_tokens_total", "prompt_tokens_total", "prefix_hits_total",
                                "prefix_queries_total", "requests_finished_total", "preemptions_total",
                                "spec_drafts_total", "spec_draft_tokens_total", "spec_accepted_total",
                                "est_flops_total", "est_read_bytes_total", "est_write_bytes_total",
                                "ttft_count", "itl_count"), 0)
        self.sums = {"ttft_sum": 0.0, "itl_sum": 0.0}
        self.ttft = dict.fromkeys(TTFT_LE, 0.0)
        self.itl = dict.fromkeys(ITL_LE, 0.0)
        self.running, self.waiting, self.kv = 0, 0, 0.0

    def work(self, *, gen=0, prompt=0, hits=0, finished=0, ttfts=(), itls=(), drafts=0, accepted=0,
             flops=0.0, read=0.0, write=0.0, preempt=0):
        c = self.c
        c["gen_tokens_total"] += gen
        c["prompt_tokens_total"] += prompt
        c["prefix_queries_total"] += prompt
        c["prefix_hits_total"] += hits
        c["requests_finished_total"] += finished
        c["spec_draft_tokens_total"] += drafts
        c["spec_accepted_total"] += accepted
        c["spec_drafts_total"] += drafts // 4
        c["est_flops_total"] += flops
        c["est_read_bytes_total"] += read
        c["est_write_bytes_total"] += write
        c["preemptions_total"] += preempt
        for v in ttfts:
            self._observe(self.ttft, v)
            c["ttft_count"] += 1
            self.sums["ttft_sum"] += v
        for v in itls:
            self._observe(self.itl, v)
            c["itl_count"] += 1
            self.sums["itl_sum"] += v

    @staticmethod
    def _observe(hist, v):
        for le in hist:
            if v <= le:
                hist[le] += 1

    def scrape(self):
        out = dict.fromkeys(key for key, _, _ in SCRAPE_FAMILIES.values())
        out.update(self.c, **self.sums, running=self.running, waiting=self.waiting,
                   kv_cache_usage=self.kv, process_start_time=self.start)
        out["ttft_buckets"] = tuple(sorted(self.ttft.items()))
        out["itl_buckets"] = tuple(sorted(self.itl.items()))
        return out


def feed(acc, engine, at, mono=None, *, live=True, view=None, scrape=None):
    return acc.observe(at=at, mono=at if mono is None else mono, live=live,
                       scrape=engine.scrape() if scrape is None else scrape,
                       cluster=cluster() if view is None else view)


# ── the pieces ──────────────────────────────────────────────────────────────────

def test_quantile_is_prometheus_histogram_quantile():
    cum = [(1.0, 10.0), (2.0, 20.0), (math.inf, 20.0)]
    assert quantile(cum, 0.5) == 1.0
    assert quantile(cum, 0.9) == pytest.approx(1.8)
    # In the open-ended bucket the answer is its lower bound: nothing says how far the tail went.
    assert quantile([(1.0, 1.0), (math.inf, 10.0)], 0.9) == 1.0
    assert quantile([(1.0, 0.0), (math.inf, 0.0)], 0.9) is None and quantile([], 0.5) is None


def test_the_config_is_the_effective_block_when_the_lb_publishes_one():
    eff = {"model_choice": "m", "generation": 53, "max_num_seqs": 16,
           "max_num_batched_tokens": 16384, "num_speculative_tokens": 6, "ple_replicate": False,
           "kv_cache_memory": 36_000_000_000, "kv_offload_gib": 0, "max_model_len": 131072,
           "long_prefill_token_threshold": 2048, "uncensored": True, "kv_cache_dtype": "nvfp4"}
    config, source = effective_config(cluster(effective=eff))
    assert source == EFFECTIVE and config == eff


def test_until_then_it_is_desired_plus_the_launch_defaults_and_says_so():
    config, source = effective_config(cluster(gen=52, seqs=8))
    assert source == FALLBACK
    assert config == {"model_choice": "qwen3.8-flash-next-hibrid48", "generation": 52, "max_num_seqs": 8,
                      "max_num_batched_tokens": 8192, "num_speculative_tokens": 4, "ple_replicate": None, "kv_cache_memory": None, "kv_offload_gib": None,
                      "max_model_len": 262144, "long_prefill_token_threshold": None,
                      "uncensored": None, "kv_cache_dtype": None}, "the fallback cannot see the pool knobs: null, never guessed"
    assert effective_config({"configured": False}) == (None, "")
    assert effective_config({"configured": True, "desired": None}) == (None, "")


# ── windows ─────────────────────────────────────────────────────────────────────

def busy_minute(acc, e, t0, reads=10, step=6.0):
    """A minute of real work: every read, 2 requests finish, tokens flow, the pair is busy."""
    closed = []
    for i in range(reads):
        e.running, e.waiting, e.kv = 6 + i % 3, i % 2, 0.30 + 0.01 * i
        e.work(gen=600, prompt=2000, hits=1500, finished=2, ttfts=(0.4, 3.0), itls=(0.04,) * 10 + (0.09,) * 2,
               drafts=400, accepted=176, flops=50e12 * step, read=180e9 * step, write=2e9 * step)
        closed += feed(acc, e, t0 + i * step)
    return closed


def test_a_busy_minute_is_one_record_with_every_rate_over_its_own_span():
    acc, e = WindowAccumulator(60), Engine()
    feed(acc, e, T0 - 6.0)                                   # the read before the minute
    assert busy_minute(acc, e, T0) == []                     # reads at T0 .. T0+54: nothing closes
    (rec,) = feed(acc, e, T0 + 60.0)                         # the first read of the next minute closes it
    # Pairs belong to the window of their LATER read, so this one starts at the read before T0.
    assert rec["t_start"] == T0 - 6.0 and rec["t_end"] == T0 + 54.0
    assert rec["span_s"] == 60.0 and rec["pairs"] == 10 and rec["reads"] == 10
    assert rec["config_source"] == FALLBACK and rec["config"]["max_num_seqs"] == 8
    assert rec["gen_tokens"] == 6000 and rec["gen_tok_s"] == pytest.approx(100.0)
    assert rec["prompt_tok_s"] == pytest.approx(20000 / 60)
    # Computed = billed - prefix-cache hits; the hit rate is hits / queries, both in tokens.
    assert rec["prompt_computed_tokens"] == 5000 and rec["prefix_hit_rate"] == 0.75
    assert rec["requests_finished"] == 20
    assert rec["ttft"]["n"] == 20 and rec["ttft"]["mean"] == pytest.approx(1.7)
    # Rank 18 of 20 falls in vLLM's (2.5, 5] bucket, 8 of its 10: interpolated to 4.5.
    assert rec["ttft"]["p90"] == pytest.approx(4.5)
    assert rec["ttft"]["buckets"] == [[0.5, 10], [5.0, 10]]
    assert rec["itl"]["n"] == 120 and rec["itl"]["mean"] == pytest.approx((0.4 + 0.18) / 12, abs=1e-4)
    assert rec["spec"] == {"draft_tokens": 4000, "accepted": 1760, "acceptance": 0.44}
    assert rec["running"] == {"mean": 6.9, "max": 8.0}
    assert rec["waiting"]["max"] == 1.0 and rec["kv"]["max"] == pytest.approx(0.39)
    assert rec["mfu"] == "measured"
    assert rec["tflops_per_gpu"] == pytest.approx(50.0) and rec["gbps_per_gpu"] == pytest.approx(182.0)
    assert rec["boxes"]["box-a"] == {"sm_mean": 88.0, "sm_max": 88.0, "pwr_mean": 61.0, "pwr_max": 61.0,
                                     "reads": 10}
    assert rec["preemptions"] == 0 and rec["engine_restarts"] == 0
    json.dumps(rec, allow_nan=False)                         # what is written is plain JSON


def test_an_idle_minute_writes_nothing():
    acc, e = WindowAccumulator(60), Engine()
    for i in range(12):
        assert feed(acc, e, T0 + i * 6.0) == []
    assert feed(acc, e, T0 + 72.0) == []
    assert acc.skipped_idle == 1


def test_a_request_running_without_a_token_yet_is_activity():
    """A long prefill: running for a minute, no token out yet — real work, and it is kept."""
    acc, e = WindowAccumulator(60), Engine()
    e.running = 1
    for i in range(10):
        feed(acc, e, T0 + i * 6.0)
    (rec,) = feed(acc, e, T0 + 60.0)
    assert rec["gen_tokens"] == 0 and rec["running"]["max"] == 1.0


def test_a_config_change_cuts_the_window_and_no_pair_spans_it():
    acc, e = WindowAccumulator(60), Engine()
    e.running = 4
    feed(acc, e, T0)
    e.work(gen=300)
    feed(acc, e, T0 + 6)
    e.work(gen=300)
    (first,) = feed(acc, e, T0 + 12, view=cluster(gen=53, seqs=16))
    assert first["config"]["generation"] == 52 and first["gen_tokens"] == 300
    e.work(gen=500)
    feed(acc, e, T0 + 18, view=cluster(gen=53, seqs=16))
    (second,) = acc.close()
    # The 300 tokens between the two configurations are nobody's: they are in neither window.
    assert second["config"]["max_num_seqs"] == 16 and second["gen_tokens"] == 500
    assert second["t_start"] == T0 + 12


def test_an_engine_restart_counts_what_the_new_engine_counted():
    acc, e = WindowAccumulator(60), Engine()
    e.running = 2
    e.work(gen=10_000)
    feed(acc, e, T0)
    fresh = Engine(start=e.start + 500)                      # a new process, counters from zero
    fresh.running = 2
    fresh.work(gen=120, ttfts=(0.3,))
    feed(acc, fresh, T0 + 6)
    (rec,) = acc.close()
    assert rec["gen_tokens"] == 120 and rec["engine_restarts"] == 1
    assert rec["ttft"]["n"] == 1 and rec["ttft"]["buckets"] == [[0.5, 1]]


def test_a_counter_that_went_down_is_a_reset_even_with_the_same_start_time():
    acc, e = WindowAccumulator(60), Engine()
    e.running = 1
    e.work(gen=5000)
    feed(acc, e, T0)
    e.c["gen_tokens_total"] = 40
    feed(acc, e, T0 + 6)
    (rec,) = acc.close()
    assert rec["gen_tokens"] == 40


def test_a_probe_gap_and_a_dead_head_are_gaps_not_pairs():
    acc, e = WindowAccumulator(60), Engine()
    e.running = 3
    feed(acc, e, T0)
    e.work(gen=900)
    feed(acc, e, T0 + 45)                                    # 45 s apart: a probe gap
    e.work(gen=100)
    feed(acc, e, T0 + 50, live=False)                        # the head did not answer
    e.work(gen=100)
    feed(acc, e, T0 + 56)
    # Not one pair in this minute: nothing it did can be put over a known span, so nothing is written.
    assert feed(acc, e, T0 + 61) == []
    # The first read of the next minute pairs with the last one of this minute; the 900 tokens across
    # the gap and the 100 across the dead read are in no window.
    e.work(gen=60)
    feed(acc, e, T0 + 67)
    (nxt,) = acc.close()
    assert nxt["gen_tokens"] == 60 and nxt["span_s"] == 11.0 and nxt["t_start"] == T0 + 56


def test_a_flat_flop_counter_while_working_is_not_measured_rather_than_zero():
    acc, e = WindowAccumulator(60), Engine()
    e.running = 2
    feed(acc, e, T0)
    e.work(gen=500)
    feed(acc, e, T0 + 6)
    (rec,) = acc.close()
    assert rec["mfu"] == "flat while working"
    assert rec["tflops_per_gpu"] is None and rec["gbps_per_gpu"] is None


def test_a_family_the_engine_does_not_expose_is_null_not_zero():
    acc, e = WindowAccumulator(60), Engine()
    e.running = 2
    blank = lambda: {**e.scrape(), "ttft_count": None, "ttft_sum": None, "ttft_buckets": None,
                     "spec_draft_tokens_total": None, "spec_accepted_total": None,
                     "est_flops_total": None}
    feed(acc, e, T0, scrape=blank())
    e.work(gen=500)
    feed(acc, e, T0 + 6, scrape=blank())
    (rec,) = acc.close()
    assert rec["ttft"] is None and rec["spec"] is None and rec["mfu"] == "not exposed"
    assert rec["itl"] == {"n": 0}                            # exposed, nothing completed in the window


def test_no_cluster_records_nothing():
    acc, e = WindowAccumulator(60), Engine()
    e.running = 5
    for i in range(12):
        assert feed(acc, e, T0 + i * 6, view={"configured": False}) == []
    assert acc.close() == []


def test_the_published_defaults_complete_desired_when_effective_is_missing():
    view = cluster()
    view["defaults"] = {"max_num_seqs": 8, "max_num_batched_tokens": 16384, "num_speculative_tokens": 6,
                        "max_model_len": 262144}
    config, source = effective_config(view)
    assert source == FALLBACK
    assert (config["max_num_batched_tokens"], config["num_speculative_tokens"]) == (16384, 6)
    assert config["max_num_seqs"] == 8                       # desired's own value, not the default's


# ── parallelism, kept exact ─────────────────────────────────────────────────────

def test_a_minute_that_mixed_two_levels_keeps_them_apart():
    """2 running for half the minute, 12 for the other half: one window, two buckets, and each bucket's
    sums are exactly its own pairs' — the per-request speed of each level survives the minute."""
    acc, e = WindowAccumulator(60), Engine()
    e.running = 2
    feed(acc, e, T0)
    for i in range(1, 6):                                    # 5 pairs at 2 running: 30 tok/s each
        e.work(gen=2 * 30 * 6, ttfts=(0.3,))
        feed(acc, e, T0 + 6 * i)
    e.running = 12
    e.work(gen=12 * 15 * 6)                                  # the pair from 2 to 12 running: level 7
    feed(acc, e, T0 + 36)
    for i in range(7, 10):                                   # 3 pairs at 12 running: 15 tok/s each
        e.work(gen=12 * 15 * 6, ttfts=(4.0,))
        feed(acc, e, T0 + 6 * i)
    (rec,) = acc.close()
    by = rec["by_running"]
    assert list(by) == ["2", "5-8", "9-12"]
    two, twelve = by["2"], by["9-12"]
    assert (two["pairs"], two["span_s"], two["req_s"]) == (5, 30.0, 60.0)
    assert two["gen_tokens"] / two["req_s"] == pytest.approx(30.0)      # per-request tok/s at 2
    assert twelve["gen_tokens"] / twelve["req_s"] == pytest.approx(15.0)  # ... and at 12
    assert twelve["gen_tokens"] / twelve["span_s"] == pytest.approx(180.0)  # the aggregate at 12
    assert (two["ttft_n"], twelve["ttft_n"]) == (5, 3)
    assert two["ttft_buckets"] == [[0.5, 5]] and twelve["ttft_buckets"] == [[5.0, 3]]
    assert by["5-8"]["pairs"] == 1 and by["5-8"]["req_s"] == 42.0      # mean of 2 and 12, over 6 s
    # Every pair did work, so the buckets add up to the window.
    assert sum(b["gen_tokens"] for b in by.values()) == rec["gen_tokens"]
    assert sum(b["span_s"] for b in by.values()) == rec["span_s"]
    assert rec["reads_running"] == [2] * 6 + [12] * 4
    assert by["2"]["boxes"]["box-a"] == {"sm_s": 88.0 * 30, "sm_t": 30.0, "pwr_s": 61.0 * 30, "pwr_t": 30.0}


def test_an_idle_pair_is_in_no_bucket_but_a_busy_minute_still_counts_its_time():
    acc, e = WindowAccumulator(60), Engine()
    feed(acc, e, T0)
    feed(acc, e, T0 + 6)                                     # idle pair: nothing running, no token
    e.running = 1
    e.work(gen=90)
    feed(acc, e, T0 + 12)
    (rec,) = acc.close()
    assert list(rec["by_running"]) == ["1"] and rec["by_running"]["1"]["pairs"] == 1
    assert rec["span_s"] == 12.0 and rec["by_running"]["1"]["span_s"] == 6.0


def test_short_requests_between_two_empty_reads_still_count_at_level_one():
    acc, e = WindowAccumulator(60), Engine()
    feed(acc, e, T0)
    e.work(gen=40, finished=1)                               # started and finished between two reads
    feed(acc, e, T0 + 6)
    (rec,) = acc.close()
    assert rec["by_running"]["1"]["gen_tokens"] == 40 and rec["by_running"]["1"]["req_s"] == 6.0


# ── the axis is the ENGINE's, never ours ─────────────────────────────────────

# "ensure benchmark aggregation is based on running not so called stupid ours" (operator,
# 2026-09-30). Two numbers on this fleet could both be called "how many at once", and they differ by
# a lot: the ENGINE's `vllm:num_requests_running`, and the LB's own `inflight` — what it has
# forwarded and is awaiting, which includes requests vLLM's API layer has not handed to the
# scheduler yet (measured: 35 ours over 14 running). Only the first is a parallelism level; the
# second would inflate every bucket by work no GPU was doing. The benchmarks page's whole
# batch-size curve hangs off this axis, so these lock it where it is decided.


def test_the_parallel_axis_is_fed_by_the_engines_own_running_gauge_and_nothing_else():
    """Provenance, at the one hop where a scrape key is named: `running` comes from
    vllm:num_requests_running, and no second family may quietly also write it."""
    assert SCRAPE_FAMILIES["vllm:num_requests_running"][0] == "running"
    assert [m for m, (key, *_) in SCRAPE_FAMILIES.items() if key == "running"] == [
        "vllm:num_requests_running"]


def test_a_bucket_is_chosen_by_what_the_engine_ran_not_by_what_the_lb_forwarded():
    """6 running while 25 are ours in flight: the minute belongs to level 5-8, not to "25+".

    The counts travel together in one scrape dict, so a future edit that reaches for the wrong key
    is a one-word change — and silently re-labels every benchmark window it touches. The `ours`
    names are planted here precisely so that edit cannot pass."""
    acc, e = WindowAccumulator(60), Engine()
    e.running, e.waiting = 6, 0
    ours = {"inflight": 25, "in_flight": 25, "ours": 25, "outstanding": 25}
    feed(acc, e, T0, scrape={**e.scrape(), **ours})
    for i in range(1, 6):                                    # 5 pairs, 6 s each, 30 tok/s per request
        e.work(gen=6 * 30 * 6, ttfts=(0.3,))
        feed(acc, e, T0 + 6 * i, scrape={**e.scrape(), **ours})
    (rec,) = acc.close()
    assert list(rec["by_running"]) == ["5-8"]                # bucket_of(6), never bucket_of(25)
    assert rec["reads_running"] == [6] * 6
    # req_s is engine-running x seconds: 6 x 6 s x 5 pairs. Off `inflight` it would read 750.
    assert rec["by_running"]["5-8"]["req_s"] == 180.0
    assert rec["by_running"]["5-8"]["gen_tokens"] / rec["by_running"]["5-8"]["req_s"] == pytest.approx(30.0)


def test_a_pair_with_nothing_running_is_in_no_bucket_however_many_are_waiting():
    """"ensure benchmark when assigning to queue size reads running only not waiting" (operator,
    2026-09-30). A pair at 0 running with 3 waiting and no token is an engine doing NOTHING — a
    stall, a prefill that has not started, a scheduler holding back. It used to land in bucket "1"
    (`bucket_of(0)` floors at 1) on the strength of its waiting count alone, and every such pair
    stretched level 1's span with seconds no request ran, dragging its tok/s down. The level is the
    running count and nothing else: no running, no work, no bucket. The window itself is still kept
    — waiting is real activity — so the stall stays visible in `waiting` and `reads_waiting`."""
    acc, e = WindowAccumulator(60), Engine()
    e.running, e.waiting = 0, 3
    feed(acc, e, T0)
    for i in range(1, 6):                                    # 5 pairs, 6 s each, nothing moves
        feed(acc, e, T0 + 6 * i)
    (rec,) = acc.close()
    assert rec["by_running"] == {}                           # not {"1": ...}
    assert rec["reads_waiting"] == [3] * 6 and rec["waiting"]["max"] == 3.0
    assert rec["reads_running"] == [0] * 6

    # And when ONE request runs with three queued behind it, the level is 1 — the waiting count is
    # no part of the choice, so 1 running + 3 waiting is bucket "1", never "3-4".
    acc, e = WindowAccumulator(60), Engine()
    e.running, e.waiting = 1, 3
    feed(acc, e, T0)
    for i in range(1, 6):
        e.work(gen=30 * 6, ttfts=(0.3,))
        feed(acc, e, T0 + 6 * i)
    (rec2,) = acc.close()
    assert list(rec2["by_running"]) == ["1"]
    assert rec2["by_running"]["1"]["pairs"] == 5 and rec2["by_running"]["1"]["req_s"] == 30.0


def test_neither_the_recorder_nor_the_probe_has_any_notion_of_our_inflight():
    """The strongest form of the rule: the two modules that decide a benchmark's parallel level
    cannot use the LB's own count, because they do not mention it. `inflight` has never appeared in
    perflog.py's history; if it ever does, this is where that shows up."""
    here = os.path.dirname(perflog.__file__)
    for name in ("perflog.py", "probe.py"):
        with open(os.path.join(here, name), encoding="utf-8") as fh:
            src = fh.read()
        for word in ("inflight", "in_flight"):
            assert word not in src, f"{name} now reads the LB's own {word}; the axis is the engine's"


# ── the disk ────────────────────────────────────────────────────────────────────

def rec_at(t, **over):
    return {"v": 1, "t_start": t, "t_end": t + 60, "span_s": 60, "gen_tok_s": 10.0, **over}


def test_records_are_appended_one_line_each_to_their_utc_day(tmp_path):
    w = JsonlWriter(str(tmp_path), max_total_bytes=1 << 20, max_file_bytes=1 << 20)
    w.write(rec_at(T0))
    w.write(rec_at(T0 + 60))
    w.write(rec_at(T0 + 86400))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["perf-2026-09-30.jsonl", "perf-2026-10-01.jsonl"]
    lines = (tmp_path / "perf-2026-09-30.jsonl").read_text().splitlines()
    assert [json.loads(line)["t_start"] for line in lines] == [T0, T0 + 60]
    assert w.status()["written"] == 3 and w.status()["errors"] == 0


def test_a_line_cut_short_by_a_crash_does_not_swallow_the_next_one(tmp_path):
    day = tmp_path / "perf-2026-09-30.jsonl"
    day.write_text(json.dumps(rec_at(T0)) + "\n" + '{"v":1,"t_start":17907')     # killed mid-write
    JsonlWriter(str(tmp_path), max_total_bytes=1 << 20, max_file_bytes=1 << 20).write(rec_at(T0 + 120))
    idx = PerfLog(str(tmp_path)).index(days=None, now=T0 + 200)
    assert [w["t_start"] for w in idx["windows"]] == [T0, T0 + 120]
    assert idx["malformed_total"] == 1 and idx["malformed"][0]["line"] == 2


def test_a_thirty_minute_swarms_minutes_are_all_drawn_and_counted_once(tmp_path):
    """A swarm's duration became the operator's on 2026-10-05, so a run can now be 30 minutes or 240.
    This log has no notion of a cap — one record per BUSY MINUTE, and MAX_WINDOWS = 20160 of them, which
    is fourteen days — so the only thing worth pinning is that nothing here is sized in quarter-hours:
    thirty consecutive minutes come back as thirty windows, in order, spanning 1800 s and not 3600."""
    w = JsonlWriter(str(tmp_path), max_total_bytes=1 << 20, max_file_bytes=1 << 20)
    for i in range(30):
        w.write(rec_at(T0 + i * 60))
    idx = PerfLog(str(tmp_path)).index(days=None, now=T0 + 1900)
    assert len(idx["windows"]) == 30 and idx["truncated"] == 0, "a 30-minute run was cut short"
    assert [win["t_start"] for win in idx["windows"]] == [T0 + i * 60 for i in range(30)]
    assert sum(win["span_s"] for win in idx["windows"]) == 1800, "a minute was counted twice"
    assert idx["malformed_total"] == 0


def test_a_days_file_stops_at_its_cap_and_says_so(tmp_path):
    w = JsonlWriter(str(tmp_path), max_total_bytes=1 << 20, max_file_bytes=200)
    w.write(rec_at(T0))
    w.write(rec_at(T0 + 60, pad="x" * 200))
    s = w.status()
    assert s["written"] == 1 and s["capped"] == 1 and "cap" in s["last_error"]


def test_the_oldest_days_go_first_once_the_directory_is_over_budget(tmp_path):
    w = JsonlWriter(str(tmp_path), max_total_bytes=600, max_file_bytes=10_000)
    for d in range(4):
        w.write(rec_at(T0 + d * 86400, pad="x" * 150))
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names[-1] == "perf-2026-10-03.jsonl" and "perf-2026-09-30.jsonl" not in names
    assert sum(p.stat().st_size for p in tmp_path.iterdir()) <= 600
    assert w.status()["deleted_days"] >= 1


def test_the_day_being_written_is_never_deleted_even_over_budget(tmp_path):
    w = JsonlWriter(str(tmp_path), max_total_bytes=10, max_file_bytes=10_000)
    w.write(rec_at(T0))
    assert [p.name for p in tmp_path.iterdir()] == ["perf-2026-09-30.jsonl"]


def test_a_disk_error_is_counted_and_shown_never_raised(tmp_path):
    blocker = tmp_path / "perf"
    blocker.write_text("a FILE where the directory should be")
    w = JsonlWriter(str(blocker), max_total_bytes=1 << 20, max_file_bytes=1 << 20)
    w.write(rec_at(T0))                                      # must not raise
    s = w.status()
    assert s["errors"] == 1 and s["written"] == 0 and s["last_error"]


def test_submit_never_blocks_a_full_queue_drops_and_counts(tmp_path):
    w = JsonlWriter(str(tmp_path), max_total_bytes=1 << 20, max_file_bytes=1 << 20, queue_size=1)
    assert w.submit(rec_at(T0)) is True
    assert w.submit(rec_at(T0 + 60)) is False                # no thread draining: full, dropped
    assert w.status()["dropped"] == 1
    w.drain()
    assert w.status()["written"] == 1


def test_the_recorder_survives_a_surprise_and_a_clean_stop_keeps_the_minute_in_progress(tmp_path):
    rec = PerfRecorder(str(tmp_path), window_s=60)
    rec.observe(at=T0, mono=T0, live=True, scrape={"running": object()}, cluster=cluster())
    assert rec.status()["internal_errors"] == 0              # an unreadable value is simply not a number
    rec.observe(at=T0, mono=T0, live=True, scrape="not a dict", cluster=cluster())
    e = Engine()
    e.running = 2
    rec.observe(at=T0 + 6, mono=T0 + 6, live=True, scrape=e.scrape(), cluster=cluster())
    e.work(gen=420)
    rec.observe(at=T0 + 12, mono=T0 + 12, live=True, scrape=e.scrape(), cluster=cluster())
    rec.close()                                              # no thread was started: stop() is a no-op
    rec._writer.drain()
    (line,) = (tmp_path / "perf-2026-09-30.jsonl").read_text().splitlines()
    assert json.loads(line)["gen_tokens"] == 420


def test_a_started_recorder_writes_from_its_own_thread(tmp_path):
    rec = PerfRecorder(str(tmp_path), window_s=60)
    rec.start()
    e = Engine()
    e.running = 1
    rec.observe(at=T0, mono=T0, live=True, scrape=e.scrape(), cluster=cluster())
    e.work(gen=60)
    rec.observe(at=T0 + 6, mono=T0 + 6, live=True, scrape=e.scrape(), cluster=cluster())
    rec.observe(at=T0 + 61, mono=T0 + 61, live=True, scrape=e.scrape(), cluster=cluster())
    rec.close()
    assert rec.status()["written"] == 1
    assert json.loads((tmp_path / "perf-2026-09-30.jsonl").read_text())["gen_tokens"] == 60


def test_from_config_is_none_when_switched_off_and_never_raises(tmp_path):
    assert PerfRecorder.from_config(Config(perf_enabled=False)) is None
    started = PerfRecorder.from_config(Config(perf_dir=str(tmp_path / "p")))
    assert started is not None
    started.close()


# ── reading it back ─────────────────────────────────────────────────────────────

def test_the_reader_lists_what_it_cannot_read_line_by_line(tmp_path):
    day = tmp_path / "perf-2026-09-30.jsonl"
    day.write_text("\n".join([json.dumps(rec_at(T0)), "[1, 2]", '{"v":1,"t_start":NaN,"t_end":1}',
                              json.dumps({"v": 1, "gen_tok_s": 3}), "", json.dumps(rec_at(T0 + 60))]) + "\n")
    (tmp_path / "notes.txt").write_text("not a day file")
    idx = PerfLog(str(tmp_path)).index(days=None, now=T0 + 3600)
    assert [w["t_start"] for w in idx["windows"]] == [T0, T0 + 60]
    assert [(m["line"], m["error"][:12]) for m in idx["malformed"]] == [
        (2, "not a window"), (3, "not JSON: no"), (4, "not a window")]
    assert idx["files"] == [{"file": "perf-2026-09-30.jsonl", "windows": 2, "malformed": 3, "error": ""}]


def test_the_range_keeps_windows_that_ended_inside_it(tmp_path):
    w = JsonlWriter(str(tmp_path), max_total_bytes=1 << 20, max_file_bytes=1 << 20)
    for t in (T0 - 3 * 86400, T0 - 86400 + 30, T0):
        w.write(rec_at(t))
    log = PerfLog(str(tmp_path))
    assert [x["t_start"] for x in log.index(days=1, now=T0 + 60)["windows"]] == [T0 - 86400 + 30, T0]
    assert len(log.index(days=None, now=T0 + 60)["windows"]) == 3


def test_an_appended_window_is_seen_on_the_next_read(tmp_path):
    w = JsonlWriter(str(tmp_path), max_total_bytes=1 << 20, max_file_bytes=1 << 20)
    log = PerfLog(str(tmp_path))
    w.write(rec_at(T0))
    assert len(log.index(days=None, now=T0)["windows"]) == 1
    w.write(rec_at(T0 + 60))
    assert len(log.index(days=None, now=T0)["windows"]) == 2


def test_no_directory_yet_is_said_not_raised(tmp_path):
    idx = PerfLog(str(tmp_path / "missing")).index()
    assert idx["windows"] == [] and "nothing has been recorded" in idx["error"]


def test_the_newest_windows_are_kept_when_there_are_too_many(tmp_path, monkeypatch):
    monkeypatch.setattr(perflog, "MAX_WINDOWS", 2)
    w = JsonlWriter(str(tmp_path), max_total_bytes=1 << 20, max_file_bytes=1 << 20)
    for i in range(5):
        w.write(rec_at(T0 + 60 * i))
    idx = PerfLog(str(tmp_path)).index(days=None, now=T0)
    assert [x["t_start"] for x in idx["windows"]] == [T0 + 180, T0 + 240] and idx["truncated"] == 3


def test_the_cut_minutes_are_counted_per_model(tmp_path, monkeypatch):
    """/benchmarks shows one model at a time: "the N oldest not shown" on a model's page is that model's
    count, so the index says whose minutes it cut — and "" for a minute that recorded no model."""
    monkeypatch.setattr(perflog, "MAX_WINDOWS", 2)
    w = JsonlWriter(str(tmp_path), max_total_bytes=1 << 20, max_file_bytes=1 << 20)
    for i, model in enumerate(["flash", "dense", "flash", None, "dense", "dense"]):
        w.write(rec_at(T0 + 60 * i, **({"config": {"model_choice": model}} if model else {})))
    idx = PerfLog(str(tmp_path)).index(days=None, now=T0)
    assert idx["truncated"] == 4 and idx["truncated_by_model"] == {"flash": 2, "dense": 1, "": 1}
    assert [x["config"]["model_choice"] for x in idx["windows"]] == ["dense", "dense"]
    monkeypatch.setattr(perflog, "MAX_WINDOWS", 20)
    assert PerfLog(str(tmp_path)).index(days=None, now=T0)["truncated_by_model"] == {}


# ── the probe reads what the log needs, off the same scrape ─────────────────────

LIVE = """\
vllm:request_success_total{engine="0",finished_reason="stop",model_name="m"} 7.0
vllm:request_success_total{engine="0",finished_reason="length",model_name="m"} 2.0
vllm:num_preemptions_total{engine="0",model_name="m"} 1.0
vllm:prefix_cache_queries_total{engine="0",model_name="m"} 5000.0
vllm:prefix_cache_hits_total{engine="0",model_name="m"} 3750.0
vllm:time_to_first_token_seconds_bucket{engine="0",le="0.5",model_name="m"} 3.0
vllm:time_to_first_token_seconds_bucket{engine="0",le="5.0",model_name="m"} 9.0
vllm:time_to_first_token_seconds_bucket{engine="0",le="+Inf",model_name="m"} 9.0
vllm:time_to_first_token_seconds_count{engine="0",model_name="m"} 9.0
vllm:time_to_first_token_seconds_sum{engine="0",model_name="m"} 12.5
vllm:inter_token_latency_seconds_bucket{engine="0",le="0.05",model_name="m"} 80.0
vllm:inter_token_latency_seconds_bucket{engine="0",le="+Inf",model_name="m"} 100.0
vllm:inter_token_latency_seconds_count{engine="0",model_name="m"} 100.0
vllm:inter_token_latency_seconds_sum{engine="0",model_name="m"} 4.2
"""


def test_the_probe_parses_the_new_families_and_both_histograms():
    got = parse_vllm_metrics(LIVE)
    assert got["requests_finished_total"] == 9 and got["preemptions_total"] == 1
    assert (got["prefix_queries_total"], got["prefix_hits_total"]) == (5000, 3750)
    assert (got["ttft_count"], got["ttft_sum"]) == (9, 12.5)
    assert got["ttft_buckets"] == ((0.5, 3.0), (5.0, 9.0), (math.inf, 9.0))
    assert got["itl_buckets"] == ((0.05, 80.0), (math.inf, 100.0))
    assert set(HISTOGRAM_BUCKETS.values()) <= set(got)


def test_one_bad_bucket_line_costs_only_its_histogram():
    got = parse_vllm_metrics(LIVE + 'vllm:inter_token_latency_seconds_bucket{engine="0",le="0.1"} NaN\n')
    assert got["itl_buckets"] is None and got["ttft_buckets"] is not None and got["itl_count"] == 100


def test_an_engine_without_the_histograms_reads_as_not_exposed():
    got = parse_vllm_metrics('vllm:num_requests_running{engine="0"} 2.0\n')
    assert got["ttft_buckets"] is None and got["itl_buckets"] is None and got["ttft_count"] is None


# ── wired into the LB ───────────────────────────────────────────────────────────

HOSTS = "box-a-gpu=box-a=user@10.0.0.1=10.9.200.1,box-b-gpu=box-b=user@10.0.0.2=10.9.200.2"
HEAD = "http://10.0.0.1:8100"


class Recorded:
    def __init__(self, boom=False):
        self.calls, self.boom = [], boom

    def observe(self, **kw):
        self.calls.append(kw)
        if self.boom:
            raise RuntimeError("the diary broke")


def test_the_control_loop_shows_the_recorder_the_heads_probe_and_nothing_else():
    cfg = Config(vllm_boxes=(VllmBox("box-a-gpu", "box-a", "http://10.0.0.9:8000"),),
                 cluster_name="pair", cluster_hosts=parse_hosts(HOSTS), cluster_url=HEAD, probe_interval_s=0)
    scrape = Engine().scrape()

    def prober(backend, cfg, *, now):
        return {"last_probe_at": now, "live": True, "mono_at": 42.0,
                **({"scrape": scrape} if backend.kind == CLUSTER else {})}

    for boom in (False, True):
        hub = Hub(cfg)
        hub.set_cluster(cluster())
        rec = Recorded(boom=boom)
        ctl = Controller(cfg, hub, leases=object(), prober=prober, recorder=rec)
        ctl._probe(T0)                                       # a recorder that raises costs nothing
        (call,) = rec.calls
        assert call["scrape"] is scrape and call["mono"] == 42.0 and call["at"] == T0
        assert call["live"] is True and call["cluster"]["desired"]["max_num_seqs"] == 8
        assert hub.backend_for(HEAD).live is True


@pytest.fixture
def served(tmp_path):
    started = []

    def make(recorder=None):
        cfg = Config(vllm_boxes=(VllmBox("box-a-gpu", "box-a", "http://127.0.0.1:9"),),
                     listen_host="127.0.0.1", listen_port=0, bench_dir=str(tmp_path / "bench"),
                     perf_dir=str(tmp_path / "perf"))
        proxy = Proxy(cfg, Hub(cfg), perf_recorder=recorder)
        proxy.start()
        started.append(proxy)
        return proxy, tmp_path / "perf"

    yield make
    for p in started:
        p.stop()


def get(proxy, path, method="GET"):
    conn = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=10)
    conn.request(method, path)
    resp = conn.getresponse()
    return resp.status, resp.read()


def test_the_route_answers_the_windows_and_the_recorders_state(served):
    rec = PerfRecorder("/nonexistent-never-written", window_s=60)
    proxy, perf = served(recorder=rec)
    JsonlWriter(str(perf), max_total_bytes=1 << 20, max_file_bytes=1 << 20).write(rec_at(T0))
    status, body = get(proxy, "/lb/perf?days=all")
    out = json.loads(body)
    assert status == 200 and [w["t_start"] for w in out["windows"]] == [T0]
    assert out["recorder"]["enabled"] is True and out["recorder"]["window_s"] == 60
    assert json.loads(get(proxy, "/lb/perf")[1])["days"] == 7.0


def test_the_route_says_when_nothing_is_recorded(served):
    proxy, _ = served(recorder=None)
    out = json.loads(get(proxy, "/lb/perf")[1])
    assert out["recorder"] == {"enabled": False, "dir": out["dir"]}
    assert out["windows"] == [] and "nothing has been recorded" in out["error"]


def test_a_nonsense_range_is_the_callers_mistake(served):
    proxy, _ = served()
    for q in ("abc", "-3", "nan"):
        status, body = get(proxy, f"/lb/perf?days={q}")
        assert status == 400 and "`days`" in json.loads(body)["error"]["message"], q


def test_the_log_route_is_read_only(served):
    proxy, _ = served()
    for method in ("POST", "PUT", "DELETE"):
        assert get(proxy, "/lb/perf", method)[0] == 405


# ── the throughput target: the best recorded minute at this setup ────────────────────────────────

def _setup(seqs=16, batched=8192, spec=4, model="qwen3.8-flash-next-hibrid48"):
    return {"model_choice": model, "generation": 80, "max_num_seqs": seqs,
            "max_num_batched_tokens": batched, "num_speculative_tokens": spec, "max_model_len": 262144}


def _minute(gen, prompt, span=60.0, **setup):
    return {"t_start": T0, "t_end": T0 + span, "span_s": span, "config": _setup(**setup),
            "gen_tok_s": gen, "prompt_tok_s": prompt}


def test_the_target_is_the_best_minute_recorded_at_the_same_setup():
    windows = [_minute(120.0, 2000.0), _minute(151.4, 1800.0), _minute(300.0, 9000.0, seqs=24)]
    t = throughput_target(windows, _setup())
    assert (t["gen_tok_s"], t["prompt_tok_s"], t["windows"], t["basis"]) == (151.4, 2000.0, 2, "setup")
    assert t["config"]["max_num_seqs"] == 16 and "generation" not in t["config"]


def test_the_target_carries_the_computed_prompt_rate_for_the_prefill_tile():
    """PREFILL's dashed line: the best tenth of the minutes' COMPUTED prompt rate — its own quantity,
    never the billed one. A minute that recorded none sets none."""
    windows = [dict(_minute(120.0, 20000.0), prompt_computed_tok_s=1500.0),
               dict(_minute(150.0, 18000.0), prompt_computed_tok_s=2600.0), _minute(90.0, 9000.0)]
    t = throughput_target(windows, _setup())
    assert (t["prompt_tok_s"], t["computed_tok_s"], t["windows"]) == (20000.0, 2600.0, 3)
    assert throughput_target([_minute(120.0, 2000.0)], _setup())["computed_tok_s"] is None


def test_a_restart_with_the_same_settings_is_the_same_setup():
    w = _minute(140.0, 2500.0)
    w["config"]["generation"] = 12
    assert throughput_target([w], _setup())["basis"] == "setup"


def test_a_cut_minute_sets_no_target():
    assert throughput_target([_minute(999.0, 99999.0, span=12.0)], _setup()) is None


def test_with_no_minute_at_this_setup_the_models_best_stands_in_and_says_so():
    t = throughput_target([_minute(90.0, 1500.0, seqs=8), _minute(130.0, 1200.0, seqs=24)], _setup())
    assert (t["gen_tok_s"], t["prompt_tok_s"], t["basis"]) == (130.0, 1500.0, "model")
    assert t["config"] == {"model_choice": "qwen3.8-flash-next-hibrid48", "uncensored": False}


def _twin(minute, uncensored):
    minute["config"]["uncensored"] = uncensored
    return minute


def test_the_uncensored_checkpoint_has_its_own_target():
    """A twin is other weights (an abliterated checkpoint; a drafter accepts its tokens at its own rate), so its
    minutes are not the original's. Blended, the twin row in the picker would print a rate the twin never ran."""
    windows = [_twin(_minute(150.0, 2000.0), False), _twin(_minute(120.0, 1800.0), True)]
    plain, twin = throughput_target(windows, _setup()), throughput_target(windows, dict(_setup(), uncensored=True))
    assert (plain["gen_tok_s"], plain["windows"], plain["basis"]) == (150.0, 1, "setup")
    assert (twin["gen_tok_s"], twin["windows"], twin["basis"]) == (120.0, 1, "setup")
    assert twin["config"]["uncensored"] is True and plain["config"]["uncensored"] is False


def test_a_minute_from_before_the_twin_existed_is_the_originals():
    """Minutes recorded before 2026-10-07 carry no `uncensored`: only the original existed, and a launch with
    None runs the original (cluster_exec.resolved_launch). They count for it, never for the twin."""
    legacy = _minute(140.0, 2100.0)
    assert "uncensored" not in legacy["config"]
    assert throughput_target([legacy], _setup())["gen_tok_s"] == 140.0
    assert throughput_target([legacy], dict(_setup(), uncensored=False))["gen_tok_s"] == 140.0
    assert throughput_target([legacy], dict(_setup(), uncensored=True)) is None


def test_the_models_best_never_crosses_to_the_other_checkpoint():
    """The `model` basis is the same CHECKPOINT at another setup — the original's best is no stand-in for a twin."""
    windows = [_twin(_minute(130.0, 1200.0, seqs=24), False)]
    assert throughput_target(windows, dict(_setup(), uncensored=True)) is None
    t = throughput_target(windows + [_twin(_minute(95.0, 900.0, seqs=8), True)], dict(_setup(), uncensored=True))
    assert (t["gen_tok_s"], t["basis"], t["config"]) == (
        95.0, "model", {"model_choice": "qwen3.8-flash-next-hibrid48", "uncensored": True})


def test_status_answers_a_target_for_every_picker_row(served):
    """`setup_targets`: the rate to aim for at what a Start of each picker row launches (`cluster.setups`, keyed
    `<id>` / `<id>@uncensored`) — the twin's from the twin's minutes, a row nothing has ever run null."""
    import time
    proxy, perf = served()
    now = time.time()
    launch = {"model_choice": "m-a", "max_num_seqs": 16, "max_num_batched_tokens": 4096, "num_speculative_tokens": 4,
              "max_model_len": 262144, "kv_cache_memory": 1, "ple_replicate": None, "kv_offload_gib": 0,
              "long_prefill_token_threshold": 0}

    def minute(gen, uncensored):
        return {"t_start": now - 120, "t_end": now - 60, "span_s": 60.0, "gen_tok_s": gen, "prompt_tok_s": 1000.0,
                "config": dict(launch, uncensored=uncensored, generation=3)}

    writer = JsonlWriter(str(perf), max_total_bytes=1 << 20, max_file_bytes=1 << 20)
    writer.write(minute(150.0, False))
    writer.write(minute(120.0, True))
    setups = {"m-a": dict(launch, uncensored=False), "m-a@uncensored": dict(launch, uncensored=True),
              "m-b": dict(launch, model_choice="m-b", uncensored=False)}
    proxy.hub.set_cluster(dict(cluster(), setups=setups), "pair")
    out = json.loads(get(proxy, "/lb/status")[1])["setup_targets"]
    assert set(out) == set(setups)
    assert (out["m-a"]["gen_tok_s"], out["m-a"]["basis"]) == (150.0, "setup")
    assert (out["m-a@uncensored"]["gen_tok_s"], out["m-a@uncensored"]["basis"]) == (120.0, "setup")
    assert out["m-b"] is None
    # A view that publishes no rows (an LB loop from before `setups`): an empty map, never an error.
    proxy.hub.set_cluster(cluster())
    assert json.loads(get(proxy, "/lb/status")[1])["setup_targets"] == {}


def test_another_models_minutes_and_no_config_set_no_target():
    assert throughput_target([_minute(90.0, 1500.0, model="other")], _setup()) is None
    assert throughput_target([_minute(90.0, 1500.0)], None) is None
    assert throughput_target([], _setup()) is None


def test_the_target_is_the_floor_of_the_top_tenth_of_minutes_not_one_freak_minute():
    # 20 minutes: nineteen between 100 and 118, and one burst at 400. The burst is not the bar.
    windows = [_minute(100.0 + i, 2000.0 + 10 * i) for i in range(19)] + [_minute(400.0, 9000.0)]
    t = throughput_target(windows, _setup())
    assert (t["gen_tok_s"], t["prompt_tok_s"], t["windows"]) == (118.0, 2180.0, 20)
    assert top_decile([5.0, 1.0, 3.0]) == 5.0 and top_decile([]) is None
    assert top_decile([float(i) for i in range(1, 31)]) == 28.0


# ── the pool a minute ran under, recorded with it ────────────────────────────────────────────────────

def test_a_recorded_minute_carries_the_kv_pool_the_engine_reported():
    """SELF-DESCRIBING, and that is the whole point. The pin is an operator knob now, so a reader that
    divides by a constant silently re-interprets every minute recorded under a different pool — and the
    first thing anyone does with the knob is an A/B across it, which is exactly what that would ruin."""
    acc, e = WindowAccumulator(60), Engine()
    pool = {"kv_pool_tokens": 1_708_119.0, "kv_pool_bytes": 28e9, "kv_max_concurrency": 6.516}
    e.running = 2
    for i in range(10):
        feed(acc, e, T0 + i * 6.0, scrape={**e.scrape(), **pool})
        e.work(gen=10, prompt=100, hits=80)
    (rec,) = feed(acc, e, T0 + 60.0, scrape={**e.scrape(), **pool})
    assert rec["kv_pool_tokens"] == 1_708_119
    assert rec["kv_pool_bytes"] == 28_000_000_000
    assert rec["kv_max_concurrency"] == pytest.approx(6.516, abs=1e-3)


def test_a_minute_whose_engine_said_nothing_about_the_pool_records_null_not_a_guess():
    """Older engines and `--disable-log-stats` publish no cache_config_info. Null says 'not known', which
    the page can fall back from honestly; a 28 GB written in anyway would be a fabrication."""
    acc, e = WindowAccumulator(60), Engine()
    e.running = 2
    for i in range(10):
        feed(acc, e, T0 + i * 6.0)
        e.work(gen=10, prompt=100, hits=80)
    (rec,) = feed(acc, e, T0 + 60.0)
    assert rec["kv_pool_tokens"] is None and rec["kv_pool_bytes"] is None


def test_the_pool_is_part_of_the_configuration_a_window_is_keyed_by():
    """Two minutes under different pools are not the same configuration: pooling them would average across
    the very change an A/B is measuring."""
    assert "kv_cache_memory" in CONFIG_KEYS and "ple_replicate" in CONFIG_KEYS


def test_the_long_prefill_threshold_is_part_of_the_configuration_a_window_is_keyed_by():
    """A minute at threshold 2048 and one with it off share every other setting: only this key tells the arms of
    that A/B apart, so it is carried from `cluster.effective` as launched — and a pair launched before the knob
    existed (no key in its record) is null, never a guessed 0."""
    assert "long_prefill_token_threshold" in CONFIG_KEYS
    base = {"model_choice": "m", "generation": 101, "max_num_seqs": 16, "max_num_batched_tokens": 8192,
            "num_speculative_tokens": 4, "ple_replicate": False, "kv_cache_memory": 31_000_000_000,
            "kv_offload_gib": 0, "max_model_len": 262144}
    on, _ = effective_config(cluster(effective={**base, "long_prefill_token_threshold": 2048}))
    off, _ = effective_config(cluster(effective={**base, "long_prefill_token_threshold": 0}))
    before, _ = effective_config(cluster(effective=base))
    assert (on["long_prefill_token_threshold"], off["long_prefill_token_threshold"],
            before["long_prefill_token_threshold"]) == (2048, 0, None)
    assert {k: v for k, v in on.items() if k != "long_prefill_token_threshold"} == \
        {k: v for k, v in off.items() if k != "long_prefill_token_threshold"}


def test_a_true_false_setting_survives_the_config_writer():
    """`isinstance(True, int)` is true in Python, so the numeric path would write the PLE placement as 0/1
    and no reader would ever get `false` back."""
    assert _plain(False) is False and _plain(True) is True
    assert _plain(None) is None and _plain(3.0) == 3
