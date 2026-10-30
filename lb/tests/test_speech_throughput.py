"""The SPEECH figures (`Throughput.speech`, /lb/status `throughput.speech`): audio transcribed per second, how many
times faster than real time (audio seconds ÷ processing seconds), latency per request, and answers per minute by
outcome — each a delta of the engine's own counters over one window, or null with its why. Counters from the real
transcription capture (tests/fixtures/transcription-metrics-2026-10-23-idle.txt) moved by hand between two probes."""
from __future__ import annotations

import pathlib

import pytest

from vllm_lb.probe import parse_metrics
from vllm_lb.rates import FIRST_SAMPLE, NOT_EXPOSED, WINDOW_NOT_REACHED
from vllm_lb.throughput import SPEECH_WINDOW_S, Throughput

IDLE = parse_metrics((pathlib.Path(__file__).parent / "fixtures" / "transcription-metrics-2026-10-23-idle.txt")
                     .read_text())


def moved(**by) -> dict:
    out = dict(IDLE)
    for key, delta in by.items():
        out[key] = IDLE[key] + delta
    return out


def meter(*scrapes, every=2.0) -> Throughput:
    t = Throughput("cluster", every)
    for i, s in enumerate(scrapes):
        t.observe(live=True, scrape=s, mono_at=100.0 + i * every)
    return t


def test_first_sample_has_no_rate():
    s = meter(IDLE).speech()
    assert s["audio_s_per_s"] is None and s["why"] == FIRST_SAMPLE


def test_audio_rate_speed_and_latency_over_one_window():
    # 2 s apart: 2 requests answered, 274 s of audio in 8 s of processing.
    s = meter(IDLE, moved(audio_seconds_total=274.0, request_seconds_sum=8.0, request_seconds_count=2,
                          requests_finished_total=2)).speech()
    assert s["audio_s_per_s"] == pytest.approx(137.0)
    assert s["speed_x"] == pytest.approx(274.0 / 8.0)
    assert s["latency_s"] == pytest.approx(4.0)
    assert s["per_min"]["finished"] == pytest.approx(60.0)
    assert s["per_min"]["failed"] == 0 and s["per_min"]["rejected"] == 0
    assert s["window_s"] == 2.0 and s["why"] is None


def test_no_answer_in_the_window_is_a_zero_rate_and_no_speed():
    s = meter(IDLE, IDLE).speech()
    assert s["audio_s_per_s"] == 0
    assert s["speed_x"] is None and s["latency_s"] is None and s["speed_why"] == WINDOW_NOT_REACHED


def test_gauges_are_the_last_scrape():
    s = meter(IDLE).speech()
    assert (s["seats"], s["queue_capacity"], s["running"], s["waiting"]) == (4, 10, 0, 0)
    assert s["gpu_memory_bytes"] == IDLE["gpu_memory_bytes"]
    assert s["totals"]["audio_seconds"] == pytest.approx(3464.604) and s["totals"]["finished"] == 12


def test_an_engine_with_no_audio_counter_says_not_exposed_but_keeps_latency():
    vllm = dict.fromkeys(IDLE)
    vllm.update(request_seconds_sum=10.0, request_seconds_count=4, requests_finished_total=4, running=0, waiting=0)
    later = {**vllm, "request_seconds_sum": 13.0, "request_seconds_count": 5, "requests_finished_total": 5}
    s = meter(vllm, later).speech()
    assert s["audio_s_per_s"] is None and s["audio_why"] == NOT_EXPOSED
    assert s["speed_x"] is None and s["latency_s"] == pytest.approx(3.0)
    assert s["per_min"]["finished"] == pytest.approx(30.0) and s["per_min"]["silence"] is None


def test_the_window_is_minutes_long():
    assert SPEECH_WINDOW_S >= 60
