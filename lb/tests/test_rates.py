"""RateMeter: a rate is ONE real delta between two real samples, or it is nothing, with a reason.

worklog/vllm-gateway-throughput.md, Design 2 and edge rows 1, 3-10 and 23. The hub replay in that
file's Ground showed the old gate reading an idle fresh vLLM as "not measured", doubling a rate
across a failed scrape, keeping a dead server's last rate and bridging a restart. The rule behind
every test here: count nothing rather than something wrong.
"""
from __future__ import annotations

import random
import time

import pytest

from vllm_lb.rates import (COUNTER_RESET, FIRST_SAMPLE, PROBE_GAP, SCRAPE_FAILED, SERVER_DOWN,
                           WHY, RateMeter, Reading)

INTERVAL = 5.0


def feed(meter: RateMeter, *samples):
    """(value, mono_at[, instance]) -> each Reading (None = the sample was not taken, rule 4)."""
    return [meter.sample(*s) for s in samples]


def test_a_zero_counter_is_a_measurement():
    """Row 1. A fresh vLLM starts at 0 and stays there while idle: that is 0 tok/s, measured."""
    m = RateMeter(INTERVAL)
    first, second = feed(m, (0, 1005.0), (0, 1010.0))
    assert first == Reading(None, FIRST_SAMPLE, None)
    assert second.tok_s == 0.0 and second.why is None and second.window_s == 5.0


def test_first_sample_has_no_rate():
    """Row 3. One sample is not a rate; the next good probe is."""
    m = RateMeter(INTERVAL)
    first, second = feed(m, (92438, 1000.0), (92588, 1005.0))
    assert first.tok_s is None and first.why == FIRST_SAMPLE
    assert second.tok_s == 30.0


def test_down_then_up_starts_fresh():
    """Row 4. The probe that found the server down clears the base, so the new process's first
    sample is a first sample and the old process's rate never reappears."""
    m = RateMeter(INTERVAL)
    got = feed(m, (100, 1000.0), (175, 1005.0))
    assert got[1].tok_s == 15.0
    down = m.sample(None, 1010.0, why_none=SERVER_DOWN)
    assert down == Reading(None, SERVER_DOWN, None)
    back, next_ = feed(m, (1000, 1015.0), (1075, 1020.0))
    assert back == Reading(None, FIRST_SAMPLE, None), "a number here would bridge the outage"
    assert next_.tok_s == 15.0


def test_a_scrape_that_failed_says_so():
    m = RateMeter(INTERVAL)
    feed(m, (100, 1000.0))
    assert m.sample(None, 1005.0) == Reading(None, SCRAPE_FAILED, None)


def test_backwards_is_reset_not_negative():
    """Row 5. A counter below its base is a restart: the sample becomes the new base."""
    m = RateMeter(INTERVAL)
    got = feed(m, (900, 1000.0), (950, 1005.0), (20, 1010.0), (70, 1015.0))
    assert [r.tok_s for r in got] == [None, 10.0, None, 10.0]
    assert got[2].why == COUNTER_RESET


def test_new_instance_is_reset_even_when_higher():
    """Row 6. `process_start_time_seconds` changed: a restart, even though 250 > 100."""
    m = RateMeter(INTERVAL)
    got = feed(m, (50, 1000.0, 1.0), (100, 1005.0, 1.0), (250, 1010.0, 2.0), (300, 1015.0, 2.0))
    assert [r.tok_s for r in got] == [None, 10.0, None, 10.0]
    assert got[2].why == COUNTER_RESET


def test_an_instance_that_is_not_exposed_is_not_a_restart():
    """The fingerprint is optional: appearing or vanishing says nothing about a restart."""
    m = RateMeter(INTERVAL)
    got = feed(m, (50, 1000.0, None), (100, 1005.0, 7.0), (150, 1010.0, None))
    assert [r.tok_s for r in got] == [None, 10.0, 10.0]


def test_restart_seen_only_through_failed_probes():
    """Row 6, without the fingerprint: 50 -> 100 -> (cold start, probes fail) -> 250 read 30.0
    before; the failed probes in between must end the chain."""
    m = RateMeter(INTERVAL)
    feed(m, (50, 1000.0), (100, 1005.0))
    m.sample(None, 1010.0, why_none=SERVER_DOWN)
    m.sample(None, 1015.0, why_none=SERVER_DOWN)
    got = feed(m, (250, 1020.0), (300, 1025.0))
    assert got[0] == Reading(None, FIRST_SAMPLE, None)
    assert got[1].tok_s == 10.0


def test_long_gap_is_not_bridged():
    """Row 8. The tick stalled for 20 s (> 3 x 5 s): no rate across it, and the sample is the base."""
    m = RateMeter(INTERVAL)
    got = feed(m, (100, 1000.0), (150, 1005.0), (300, 1025.0), (350, 1030.0))
    assert [r.tok_s for r in got] == [None, 10.0, None, 10.0]
    assert got[2].why == PROBE_GAP


def test_too_close_changes_nothing():
    """Row 9. A sample 1 s after the base is not a measurement: neither the base nor the result moves."""
    m = RateMeter(INTERVAL)
    feed(m, (100, 1000.0), (150, 1005.0))
    assert m.sample(160, 1006.0) is None
    after = m.sample(200, 1010.0)
    assert after.tok_s == 10.0 and after.window_s == 5.0, "the delta runs from the untouched base"


def test_full_precision_is_kept():
    m = RateMeter(INTERVAL)
    got = feed(m, (0, 0.0), (1, 3.0))
    assert got[1].tok_s == pytest.approx(1 / 3) and got[1].tok_s != 0.3


@pytest.mark.parametrize("seed", range(40))
def test_every_rate_is_one_real_delta(seed):
    """Row 10, as a property: every emitted rate is Δvalue/Δt between this sample and the last real
    sample the meter took, of the same instance, within [0.5, 3] x the interval; after a None,
    nothing is emitted until two real samples exist; every reason is in the vocabulary."""
    rng = random.Random(seed)
    m = RateMeter(INTERVAL)
    t, value, instance = 1000.0, 0, 1.0
    last = None                 # (t, value, instance) of the last real sample taken
    real_since_none = 0
    for step in range(300):
        t += rng.choice([0.4, 1.0, 4.9, 5.0, 5.0, 5.3, 7.5, 14.9, 16.0, 40.0])
        roll = rng.random()
        if roll < 0.08:
            reading = m.sample(None, t, why_none=rng.choice([SERVER_DOWN, SCRAPE_FAILED]))
            assert reading.tok_s is None and reading.why in WHY
            last, real_since_none = None, 0
            continue
        if roll < 0.12:
            value = rng.randint(0, 40)                   # a restart, lower
        elif roll < 0.15:
            instance = rng.random()                      # a restart, fingerprinted
            value += rng.randint(0, 900)
        else:
            value += rng.choice([0, 0, rng.randint(1, 600)])
        reading = m.sample(value, t, instance)
        if reading is None:                              # rule 4: not taken
            assert last is not None and t - last[0] < 0.5 * INTERVAL
            continue
        real_since_none += 1
        if reading.tok_s is None:
            assert reading.why in WHY
        else:
            assert reading.why is None
            assert real_since_none >= 2, "a rate after a None needs two real samples"
            assert last is not None and last[2] == instance and value >= last[1]
            dt = t - last[0]
            assert 0.5 * INTERVAL <= dt <= 3 * INTERVAL
            assert reading.tok_s == pytest.approx((value - last[1]) / dt)
            assert reading.tok_s >= 0
        last = (t, value, instance)


def test_prompt_window_smooths_a_credit_spike():
    """Row 23. vLLM credits a prompt's tokens at its first output token: a 250,000-token prompt lands
    as ONE step. Over a 5 s probe that reads 50,000 tok/s; over the 60 s window it reads what was
    really loaded in that minute, and says over how long."""
    plain, window = RateMeter(INTERVAL), RateMeter(INTERVAL, window_s=60.0)
    t, total = 1000.0, 10_000
    for _ in range(12):
        plain.sample(total, t)
        window.sample(total, t)
        t += 5.0
    total += 250_000                                     # the first token of the long prompt
    spike, smoothed = plain.sample(total, t), window.sample(total, t)
    assert spike.tok_s == 50_000.0
    assert smoothed.tok_s == pytest.approx(250_000 / 60.0) and smoothed.window_s == 60.0


def test_a_window_reports_the_span_it_really_covers():
    """A window is never pretended full: two samples 10 s apart are a 10 s window, and a reset
    starts the run again."""
    m = RateMeter(INTERVAL, window_s=60.0)
    got = feed(m, (0, 1000.0), (50, 1005.0), (100, 1010.0))
    assert got[2].tok_s == 10.0 and got[2].window_s == 10.0
    reset = m.sample(5, 1015.0)
    assert reset.why == COUNTER_RESET
    after = m.sample(55, 1020.0)
    assert after.tok_s == 10.0 and after.window_s == 5.0


def test_a_window_drops_samples_older_than_itself():
    m = RateMeter(INTERVAL, window_s=20.0)
    readings = [m.sample(i * 10, 1000.0 + i * 5) for i in range(10)]
    assert readings[-1].window_s == 20.0 and readings[-1].tok_s == 2.0


def test_wall_clock_jump_is_ignored(monkeypatch):
    """Row 13, at the hub: dt runs on the monotonic stamps the probe takes when it reads, never on
    the wall clock. An NTP step of -1 h and then +2 h between probes — `time.time` itself jumping —
    changes nothing. The old hub took dt from `last_probe_at` and read [None, None, 0.0]."""
    from vllm_lb.config import Config, VllmBox
    from vllm_lb.hub import Hub
    from vllm_lb.probe import SCRAPE_FAMILIES

    url = "http://box-a:8000"
    hub = Hub(Config(vllm_boxes=(VllmBox("box-a-gpu", "box-a", url),)))
    rates = []
    for wall, mono, gen in ((1000.0, 50.0, 1000), (1000.0 - 3600, 55.0, 1075),
                            (1000.0 + 7200, 60.0, 1150)):
        monkeypatch.setattr(time, "time", lambda wall=wall: wall)
        scrape = dict.fromkeys(key for key, _, _ in SCRAPE_FAMILIES.values())
        scrape.update(waiting=0, running=0, gen_tokens_total=gen)
        hub.update_backend(url, live=True, scrape=scrape, last_probe_at=wall, mono_at=mono)
        rates.append(hub.backend_for(url).tokens_per_s)
    assert rates == [None, 15.0, 15.0]


def test_the_vocabulary_is_closed_and_spelled_once():
    assert WHY == {"server down", "first sample", "counter reset", "probe gap", "scrape failed",
                   "not exposed by this server", "no /metrics on ollama",
                   "no request through the LB yet", "credited at first token",
                   "too short to time", "window not reached",
                   "the engine is not measuring it (no --enable-mfu-metrics)",
                   "an engine step is still running: vLLM counts a step's FLOPs and bytes only when "
                   "the step completes",
                   "not known yet: the engine has produced no token and its FLOP counter is still 0 — "
                   "a first step still running and a missing --enable-mfu-metrics look alike until a "
                   "token is out",
                   # The LB's modeled SGLang bandwidth (throughput.py `bandwidth_model`).
                   "a forward pass is still running: SGLang counts a pass only when it completes"}


# ── a counter that moves in lumps: keep a little more, to find the last lump ───
# vLLM adds an engine step's FLOPs when the STEP completes, so a 7 s prefill step shows as a flat
# 5 s window followed by a jump. `keep_s` lets the meter look back to that jump without changing what
# its own two-probe reading says (throughput.Throughput.engine_perf is the one caller).

def test_keeping_more_of_the_run_does_not_change_the_reading():
    plain, kept = RateMeter(INTERVAL), RateMeter(INTERVAL, keep_s=60.0)
    for i, value in enumerate((0, 10, 10, 40, 40, 40, 90)):
        a, b = plain.sample(value, 1000.0 + 5 * i), kept.sample(value, 1000.0 + 5 * i)
        assert a == b, (i, a, b)


def test_rise_base_is_the_sample_just_before_the_last_increase():
    m = RateMeter(INTERVAL, keep_s=60.0)
    feed(m, (0, 1000.0), (10, 1005.0), (10, 1010.0), (40, 1015.0), (40, 1020.0), (40, 1025.0))
    assert m.rise_base() == 1010.0, "40 first appeared at 1015; the sample before it is the base"
    assert m.rate_since(1010.0) == Reading(30 / 15, None, 15.0)


def test_rise_base_never_looks_before_its_floor():
    m = RateMeter(INTERVAL, keep_s=60.0)
    feed(m, (0, 1000.0), (10, 1005.0), (10, 1010.0), (10, 1015.0))
    assert m.rise_base() == 1000.0
    assert m.rise_base(floor_at=1005.0) is None, "the rise before the floor is not this work's"


def test_rise_base_and_rate_since_see_only_the_kept_unbroken_run():
    m = RateMeter(INTERVAL, keep_s=10.0)
    feed(m, (0, 1000.0), (10, 1005.0), (10, 1010.0), (10, 1015.0), (10, 1020.0))
    assert m.rise_base() is None, "the rise at 1005 aged out of a 10 s keep"
    assert m.rate_since(1000.0) is None
    feed(m, (20, 1025.0))
    m.sample(None, 1030.0)                                   # a scrape that read nothing
    assert m.rise_base() is None and m.rate_since(1025.0) is None, "never across a broken run"
