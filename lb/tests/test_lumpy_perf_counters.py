"""A flat FLOP counter for one probe window is NOT the switch being off.

2026-09-30, the B1 variant (`--max-num-batched-tokens 16384`) of the tuning series, during its pure
prefill probe (16 requests of ~37k tokens, max_tokens=1): the saturation panel said "busy — no ceiling
measured · ... it was started without --enable-mfu-metrics". The flag WAS on (engine argv), the
/mfu-patch was mounted, and `vllm:estimated_flops_per_gpu_total` read 4.36e15 and rising.

What vLLM 0.29 actually does (read in the image, not assumed):

* the FLOPs and bytes are computed per engine STEP from the SchedulerOutput, prefill chunks included
  (v1/metrics/perf.py:1336 `get_step_perf_stats_per_gpu`: new requests and chunked-prefill requests
  are both added to the ExecutionContext);
* they ride on SchedulerStats, which the scheduler returns for EVERY step, output or not
  (v1/core/sched/scheduler.py:2219 "We must return the stats even if there are no request outputs
  this step"), which the API server records for every EngineCoreOutputs it receives
  (v1/engine/async_llm.py:806-816, v1/metrics/loggers.py:1150);
* but `update_from_output` runs only AFTER the step's forward pass returns (v1/engine/core.py:615-624),
  so the counters move in one lump per COMPLETED step.

(Only IterationStats — `iteration_tokens_total`, the token counters — is gated on output:
async_llm.py:768-770. That is why the prefill probe saw exactly 16 "iterations"; it is not what feeds
the FLOP counter.)

A 16384-token prefill step on this pair takes ~7 s (16384 tok / 2348 tok/s, B1's own probe) and the LB
probes every 5 s: a window can close with a step still running. So a flat window while requests run
says "a step is in flight" — never, by itself, "the flag is off". The flag is only called off on
evidence: the engine has produced tokens and its FLOP counter is still exactly 0 in the same process.

Seen on the live engine, read-only, 2026-09-30 03:12 (variant C8-s4, batched 8192, a /metrics GET once a
second during its prefill probe): the FLOP counter jumped +116..+206 TFLOP every ~2.7 s, each jump on
the same second as a KV-usage change, while generation_tokens and iteration_tokens_total_count did NOT
move — steps with no output were counted — and it stood still in between.

`vllm:num_requests_running` rides the SAME SchedulerStats, so it too is only updated when a step
completes: during the first step of new work the engine still reports what it reported last. The
sequences below model that.

Every sequence here is synthetic, built from those numbers; nothing needs SSH, Docker or a GPU.
"""
from __future__ import annotations

import pytest

from vllm_lb.probe import SCRAPE_FAMILIES
from vllm_lb.rates import FIRST_SAMPLE, MFU_NOT_MEASURED
from vllm_lb.saturation import BUSY_UNMEASURED, gpu_view, judge
from vllm_lb.state import CLUSTER
from vllm_lb.throughput import Throughput

FAMILIES = {key for key, _, _ in SCRAPE_FAMILIES.values()}
INTERVAL = 5.0
#: One `--max-num-batched-tokens 16384` prefill step on this pair: 16384 tok at B1's 2348 tok/s.
STEP_S = 7.0
#: What one such step adds to the three counters, PER GPU: B1's own averages (44.83 TFLOP/s,
#: 24.9 GB/s) times the step.
FLOPS_PER_STEP = 3.1e14
READ_PER_STEP = 1.7e11
WRITE_PER_STEP = 4.0e9
#: What the counter already holds from earlier work in this same engine process: the flag is ON.
WARM = 4.36e15
PROMPT_TOKENS = 37_085                  # one realprefill.json request
REQUESTS = 16
BOX_A_GPU = {"count": 1, "sm_pct": 96, "mem_duty_pct": 0, "pwr_w": 44, "gtemp_c": 71,
             "pclk_mhz": 2437, "pclk_max_mhz": 3003}
BOX_B_GPU = {"count": 1, "sm_pct": 93, "mem_duty_pct": 0, "pwr_w": 41, "gtemp_c": 67,
              "pclk_mhz": 2476, "pclk_max_mhz": 3003}


def scrape(**families) -> dict:
    out = dict.fromkeys(FAMILIES)
    out.update({"waiting": 0, "running": 0, "process_start_time": 1_790_000_000.0, **families})
    return out


def lumpy_prefill(end: float = 60.0, arrive: float = 0.5, warm: float = WARM,
                  gen_before: int = 5_000) -> list[tuple[float, dict]]:
    """(t, scrape) at every 5 s probe. The 16 requests arrive just after t=0; one 7 s step completes
    at 7.5, 14.5, 21.5 ... and each completion adds ONE step's FLOPs and bytes — the only moment vLLM
    increments these counters. A request emits its single token when its last chunk completes. The
    running gauge is what the LAST COMPLETED step reported: idle (0) until the first one completes."""
    out, t = [], 0.0
    while t <= end:
        busy = t >= arrive
        steps = int((t - arrive) // STEP_S) if busy else 0
        done = min(REQUESTS, steps * 16384 // PROMPT_TOKENS)
        out.append((t, scrape(
            running=REQUESTS - done if steps else 0,
            est_flops_total=warm + steps * FLOPS_PER_STEP,
            est_read_bytes_total=(warm / 20) + steps * READ_PER_STEP,
            est_write_bytes_total=(warm / 800) + steps * WRITE_PER_STEP,
            gen_tokens_total=gen_before + done,
            prompt_tokens_total=40_000 + done * PROMPT_TOKENS,
            kv_cache_usage=0.02 + steps * 0.003)))
        t += INTERVAL
    return out


def fed(sequence) -> list[tuple[float, dict]]:
    """Feed a Throughput the whole sequence; the engine_perf block after each probe."""
    t = Throughput(CLUSTER, interval_s=INTERVAL)
    out = []
    for at, s in sequence:
        t.observe(live=True, scrape=s, mono_at=at, at=at)
        out.append((at, t.engine_perf()))
    return out


def at(perfs, when: float) -> dict:
    return next(p for t, p in perfs if t == when)


# ── the defect ─────────────────────────────────────────────────────────────────

def test_a_long_prefill_step_is_never_reported_as_the_switch_being_off():
    """THE BUG. With 7 s steps and 5 s probes, the windows ending at 20, 35 and 55 s hold no step
    completion. The engine's counter is 4.36e15 and rising — it IS measuring."""
    sequence = lumpy_prefill()
    flat = [t for (_, a), (t, b) in zip(sequence, sequence[1:])
            if a["running"] and b["running"] and a["est_flops_total"] == b["est_flops_total"]]
    assert flat == [20.0, 35.0, 55.0], "the synthetic sequence must actually contain flat windows"
    for when, perf in fed(sequence):
        assert perf["flops_why"] != MFU_NOT_MEASURED, (when, perf)
        assert perf["bytes_why"] != MFU_NOT_MEASURED, (when, perf)


def test_a_flat_window_after_a_completed_step_is_widened_to_that_step_and_stays_a_measurement():
    """The window 15 s → 20 s is flat. The step that completed at 14.5 s first shows at the 15 s probe,
    so the narrowest window ending now that holds a completed step starts at the 10 s probe: over
    10 s → 20 s the engine counted exactly one step. That is a real count over a stated window."""
    perf = at(fed(lumpy_prefill()), 20.0)
    assert perf["tflops_per_gpu"] == pytest.approx(FLOPS_PER_STEP / 10.0 / 1e12)       # 31.0
    assert perf["gbps_per_gpu"] == pytest.approx((READ_PER_STEP + WRITE_PER_STEP) / 10.0 / 1e9)
    assert perf["window_s"] == pytest.approx(10.0)
    assert perf["flops_why"] is None and perf["bytes_why"] is None
    assert "prefill in progress" in (perf.get("widened") or ""), perf


def test_the_widened_window_says_prefill_even_when_one_request_finished_inside_it():
    """At 55 s the step that completed at 49.5 s finished a request, so ONE token was generated in
    the widened window — while 13 requests were still being read. Fewer tokens than running requests
    over a completed step means requests are still in prefill."""
    perf = at(fed(lumpy_prefill()), 55.0)
    assert perf["tflops_per_gpu"] is not None and perf["flops_why"] is None
    assert "prefill in progress" in (perf.get("widened") or ""), perf


def test_a_window_that_moved_is_reported_as_before_and_not_marked_widened():
    perf = at(fed(lumpy_prefill()), 15.0)
    assert perf["tflops_per_gpu"] == pytest.approx(FLOPS_PER_STEP / 5.0 / 1e12)
    assert perf["window_s"] == pytest.approx(5.0)
    assert perf.get("widened") is None


def test_an_lb_that_starts_reading_during_a_long_step_says_the_step_is_in_flight():
    """The LB restarted mid-prefill (every recreate between benchmark variants does this): its first
    read is at 15 s and the next, at 20 s, is flat. The counter is far from 0, so the switch is on —
    but the LB has seen no completed step to widen to. The honest reading is "a step is in flight"."""
    perf = fed([s for s in lumpy_prefill() if s[0] >= 15.0])[1][1]
    assert perf["tflops_per_gpu"] is None and perf["gbps_per_gpu"] is None
    assert perf["flops_why"] not in (None, MFU_NOT_MEASURED), perf
    assert "step" in perf["flops_why"] and perf["mfu_metrics"] == "on", perf


def test_the_first_window_of_new_work_reports_what_the_engine_reports():
    """At 5 s the requests arrived 4.5 s ago and no step has completed, so the engine's own gauges
    still say nothing runs and its counter has not moved: a 0 over that window is what it reported.
    What must NOT appear is the switch."""
    perf = at(fed(lumpy_prefill()), 5.0)
    assert perf["tflops_per_gpu"] == 0.0 and perf["flops_why"] is None
    assert perf["mfu_metrics"] == "on"


def test_widening_never_reaches_back_across_idle_time():
    """Defensive: should the engine ever report requests running with a flat counter straight after an
    idle read (a step that scheduled no tokens adds no FLOPs), the window must not be widened back
    across that idle read and averaged with an earlier burst."""
    s = [(0.0, scrape(running=8, est_flops_total=1e15, est_read_bytes_total=1e13,
                      est_write_bytes_total=1e11, gen_tokens_total=100)),
         (5.0, scrape(running=8, est_flops_total=2e15, est_read_bytes_total=2e13,
                      est_write_bytes_total=2e11, gen_tokens_total=900)),
         (10.0, scrape(running=0, est_flops_total=2e15, est_read_bytes_total=2e13,
                       est_write_bytes_total=2e11, gen_tokens_total=1000)),
         (15.0, scrape(running=16, est_flops_total=2e15, est_read_bytes_total=2e13,
                       est_write_bytes_total=2e11, gen_tokens_total=1000))]
    perf = fed(s)[-1][1]
    assert perf["tflops_per_gpu"] is None, "a rate spanning the idle gap would be an invented average"
    assert perf["flops_why"] not in (None, MFU_NOT_MEASURED)


def test_a_decode_step_that_stalls_past_the_window_is_not_the_switch_either():
    """The UNEXPLAINED bimodal decode stall on this cluster: 24 requests decoding, the counter moving
    every window, then one step that takes longer than a probe window. The note must not say
    prefill — every running request was emitting tokens."""
    s, flops, gen = [], WARM, 50_000
    for i in range(6):
        s.append((i * 5.0, scrape(running=24, est_flops_total=flops, est_read_bytes_total=flops / 20,
                                   est_write_bytes_total=flops / 800, gen_tokens_total=gen)))
        if i < 4:
            flops, gen = flops + 1.5e13, gen + 24 * 40      # ~40 decode steps per window
    perfs = fed(s)
    stalled = perfs[-1][1]                                   # 20 → 25 s: nothing completed
    assert stalled["flops_why"] != MFU_NOT_MEASURED
    assert stalled["tflops_per_gpu"] is not None, "widened to the last completed step"
    note = stalled.get("widened") or ""
    assert note and "prefill" not in note, stalled


# ── only EVIDENCE calls the switch off ────────────────────────────────────────

def test_tokens_produced_while_the_counter_stays_at_zero_is_the_switch_being_off():
    """The evidence rule, kept: this process generated tokens — so steps completed — and its FLOP
    counter never left 0.0. That only happens when the engine does not count FLOPs."""
    s = [(0.0, scrape(running=3, est_flops_total=0.0, est_read_bytes_total=0.0,
                      est_write_bytes_total=0.0, gen_tokens_total=100)),
         (5.0, scrape(running=3, est_flops_total=0.0, est_read_bytes_total=0.0,
                      est_write_bytes_total=0.0, gen_tokens_total=400))]
    perf = fed(s)[-1][1]
    assert perf["tflops_per_gpu"] is None and perf["flops_why"] == MFU_NOT_MEASURED
    assert perf["gbps_per_gpu"] is None and perf["bytes_why"] == MFU_NOT_MEASURED


def test_a_fresh_engine_whose_first_step_is_still_running_is_not_known_yet():
    """A new process, its first long prefill in flight: the counter is 0 and no token is out yet. A
    first step still running and a disabled switch look the same here — so neither is claimed."""
    s = [(0.0, scrape(running=16, est_flops_total=0.0, est_read_bytes_total=0.0,
                      est_write_bytes_total=0.0, gen_tokens_total=0, prompt_tokens_total=0)),
         (5.0, scrape(running=16, est_flops_total=0.0, est_read_bytes_total=0.0,
                      est_write_bytes_total=0.0, gen_tokens_total=0, prompt_tokens_total=0))]
    perf = fed(s)[-1][1]
    assert perf["tflops_per_gpu"] is None
    assert perf["flops_why"] not in (None, MFU_NOT_MEASURED), perf
    assert "not known yet" in perf["flops_why"], perf


def test_the_evidence_is_carried_in_the_block():
    """What the LB knows about the switch, stated: on (the counter has moved in this process), off
    (tokens out, counter still 0), or null (not known yet)."""
    on = at(fed(lumpy_prefill()), 20.0)
    assert on.get("mfu_metrics") == "on"
    off = fed([(0.0, scrape(running=3, est_flops_total=0.0, est_read_bytes_total=0.0,
                            est_write_bytes_total=0.0, gen_tokens_total=10)),
               (5.0, scrape(running=3, est_flops_total=0.0, est_read_bytes_total=0.0,
                            est_write_bytes_total=0.0, gen_tokens_total=90))])[-1][1]
    assert off.get("mfu_metrics") == "off"
    unknown = fed([(0.0, scrape(running=16, est_flops_total=0.0, est_read_bytes_total=0.0,
                                est_write_bytes_total=0.0, gen_tokens_total=0))])[-1][1]
    assert "mfu_metrics" in unknown and unknown["mfu_metrics"] is None


def test_a_restarted_engine_is_judged_on_its_own_counters_not_the_last_ones():
    """A new process_start_time is a new engine: its counter starts at 0 again, and the previous
    process having counted FLOPs proves nothing about this one."""
    s = [(0.0, scrape(running=4, est_flops_total=WARM, est_read_bytes_total=WARM / 20,
                      est_write_bytes_total=WARM / 800, gen_tokens_total=900)),
         (5.0, scrape(running=16, est_flops_total=0.0, est_read_bytes_total=0.0,
                      est_write_bytes_total=0.0, gen_tokens_total=0, prompt_tokens_total=0,
                      process_start_time=1_790_000_999.0)),
         (10.0, scrape(running=16, est_flops_total=0.0, est_read_bytes_total=0.0,
                       est_write_bytes_total=0.0, gen_tokens_total=0, prompt_tokens_total=0,
                       process_start_time=1_790_000_999.0))]
    perf = fed(s)[-1][1]
    assert perf["tflops_per_gpu"] is None and perf["flops_why"] != MFU_NOT_MEASURED
    assert perf.get("mfu_metrics") is None


# ── the verdict never blames the switch without that evidence ──────────────────

def busy_judge(**over) -> dict:
    gpus = [dict(gpu_view(BOX_A_GPU), box="box-a"), dict(gpu_view(BOX_B_GPU), box="box-b")]
    kwargs = {"gpus": gpus, "tflops_per_gpu": None, "gbps_per_gpu": None, "peak_tflops": 323.2,
              "peak_gbps": 273.0, "gpu_count": 2}
    return judge(**{**kwargs, **over})


@pytest.mark.parametrize("why", [FIRST_SAMPLE, "counter reset", "probe gap", "scrape failed",
                                 "not exposed by this server"])
def test_a_missing_rate_for_any_other_reason_does_not_blame_the_flag(why):
    """An LB that restarted a moment ago has a FIRST SAMPLE for every rate. "It was started without
    --enable-mfu-metrics" there is the same false claim as during a prefill."""
    v = busy_judge(flops_why=why, bytes_why=why)
    assert v["verdict"] == BUSY_UNMEASURED and v["headroom"] is None
    assert "--enable-mfu-metrics" not in v["because"], v["because"]
    assert why in v["because"], "the page says the real reason instead"


def test_a_step_in_flight_is_named_in_the_verdict():
    perf = fed([s for s in lumpy_prefill() if s[0] >= 15.0])[1][1]
    assert perf["flops_why"], "the LB-restarted-mid-step case carries a why"
    v = busy_judge(flops_why=perf["flops_why"], bytes_why=perf["bytes_why"])
    assert "--enable-mfu-metrics" not in v["because"], v["because"]
    assert perf["flops_why"] in v["because"]


def test_the_proven_off_switch_is_still_named_in_the_verdict():
    v = busy_judge(flops_why=MFU_NOT_MEASURED, bytes_why=MFU_NOT_MEASURED)
    assert "--enable-mfu-metrics" in v["because"]


def test_a_widened_rate_is_a_floor_so_it_proves_neither_a_missing_component_nor_headroom():
    """Over a widened window the step still in flight has done work the counter does not hold yet, so
    the rate is a FLOOR. A low floor is expected, not a sign the estimate lacks the MoE (the rule that
    catches that symptom must not fire on it), and a floor under a ceiling does not prove the ceiling
    is far: no headroom is claimed from it."""
    perf = at(fed(lumpy_prefill()), 20.0)
    v = busy_judge(tflops_per_gpu=perf["tflops_per_gpu"], gbps_per_gpu=perf["gbps_per_gpu"],
                   window_s=perf["window_s"], window_widened=perf["widened"])
    assert v["bandwidth"]["why"] is None and v["compute"]["why"] is None, v
    assert v["bandwidth"]["pct"] == pytest.approx(6.4, abs=0.05), "the share over the stated window"
    assert v["headroom"] is None, v["because"]
    assert "floor" in v["because"], v["because"]


def test_a_widened_floor_above_a_ceiling_still_proves_the_bound():
    """The other direction holds: if even the floor is at the bandwidth ceiling, the pair is bound."""
    v = busy_judge(tflops_per_gpu=3.0, gbps_per_gpu=210.0, window_s=10.0,
                   window_widened="an engine step outlasted the probe window")
    assert v["verdict"] == "memory-bandwidth bound" and v["headroom"] is False


# ── end to end: /lb/status through a lumpy prefill ────────────────────────────

HEAD = "http://192.0.2.2:8100"
HOSTS = "box-a-gpu=box-a=user@10.0.0.1=10.9.200.1,box-b-gpu=box-b=user@10.0.0.2=10.9.200.2"


def test_lb_status_never_says_the_switch_is_off_through_a_lumpy_prefill():
    """The whole path the page reads: the Hub fed each probe as the control loop feeds it, and the
    real Proxy.status() after every one."""
    from vllm_lb.cluster import parse_hosts
    from vllm_lb.config import Config
    from vllm_lb.hub import Hub
    from vllm_lb.proxy import Proxy
    cfg = Config(vllm_boxes=(), cluster_name="pair", cluster_hosts=parse_hosts(HOSTS),
                 cluster_url=HEAD, listen_host="127.0.0.1", listen_port=0)
    hub, seen = Hub(cfg), {}
    proxy = Proxy(cfg, hub)
    hub.set_cluster({"configured": True, "state": "running",
                     "boxes": {"box-a": {"gpu": gpu_view(BOX_A_GPU)},
                               "box-b": {"gpu": gpu_view(BOX_B_GPU)}}})
    for when, s in lumpy_prefill():
        hub.update_backend(HEAD, live=True, last_probe_at=when, mono_at=when, depth_known=True,
                           waiting=0, running=s["running"],
                           served_models=("Qwen/Qwen3.8-Flash-Next",), max_model_len=262144,
                           scrape=s)
        sat = proxy.status()["saturation"]
        assert "--enable-mfu-metrics" not in sat["because"], (when, sat["because"])
        assert sat["compute"]["why"] != MFU_NOT_MEASURED, (when, sat["compute"])
        assert sat["bandwidth"]["why"] != MFU_NOT_MEASURED, (when, sat["bandwidth"])
        seen[when] = sat
    widened = seen[20.0]
    assert widened["compute"]["rate"] == pytest.approx(31.0, abs=0.05)
    assert widened["window_s"] == pytest.approx(10.0)
    assert "prefill in progress" in (widened.get("window_widened") or ""), widened
    assert "MoE" not in (widened["bandwidth"]["why"] or ""), "a floor is not a missing component"
    assert widened["headroom"] is None, widened["because"]
