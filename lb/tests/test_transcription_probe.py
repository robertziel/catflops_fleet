"""The probe reads a TRANSCRIPTION backend (a faster-whisper transcription server, `transcription_*`) in the LB's own vocabulary.

Captured off a throwaway node of the service's own image (vexa-stt-rocm, faster-whisper large-v3-turbo FP16 on a
Radeon 8060S, MAX_ACTIVE_REQUESTS=4) on 2026-10-23, after real transcriptions through it:

  * `transcription-metrics-2026-10-23-busy.txt` — 4 running (six 137 s readings sent at once: four seated, two 503)
  * `transcription-metrics-2026-10-23-idle.txt` — the same process after they finished and a silent clip

Where the service publishes no equivalent, the canonical key stays null — never a zero.
"""
from __future__ import annotations

import pathlib

import pytest

from vllm_lb.config import Config
from vllm_lb.probe import (ENGINE_TRANSCRIPTION, SCRAPE_KEYS, SGLANG_UNAVAILABLE, TRANSCRIPTION_UNAVAILABLE,
                           parse_metrics, probe_backend)
from vllm_lb.state import CLUSTER, Backend

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
BUSY = (FIXTURES / "transcription-metrics-2026-10-23-busy.txt").read_text()
IDLE = (FIXTURES / "transcription-metrics-2026-10-23-idle.txt").read_text()
MODELS = b'{"object":"list","data":[{"id":"/models/faster-whisper-large-v3-turbo","object":"model"}]}'


def test_the_queue_depth_is_read_so_the_router_knows_it():
    busy = parse_metrics(BUSY)
    assert (busy["running"], busy["waiting"], busy["seats"], busy["queue_capacity"]) == (4, 0, 4, 10)
    assert busy["engine"] == ENGINE_TRANSCRIPTION == "faster-whisper"


def test_outcomes_audio_and_processing_seconds():
    idle = parse_metrics(IDLE)
    assert idle["requests_finished_total"] == 12, "ok 11 + silence 1: both answered"
    assert idle["requests_silence_total"] == 1
    assert idle["requests_failed_total"] == 0
    assert idle["requests_rejected_total"] == 2, "busy + queue full + deferred"
    assert idle["audio_seconds_total"] == pytest.approx(3464.604)
    assert idle["request_seconds_sum"] == pytest.approx(123.37377262115479)
    assert idle["request_seconds_count"] == 12


def test_gpu_memory_sums_the_regions():
    assert parse_metrics(IDLE)["gpu_memory_bytes"] == 2828525568 + 1417216


def test_every_other_key_is_null_and_declared():
    out = parse_metrics(IDLE)
    assert set(out) == {k for k, _ in SCRAPE_KEYS}
    assert sorted(k for k, v in out.items() if v is None) == sorted(TRANSCRIPTION_UNAVAILABLE)
    assert all(TRANSCRIPTION_UNAVAILABLE.values())


def test_a_service_without_gpu_memory_publishes_none():
    text = "\n".join(l for l in IDLE.splitlines() if "gpu_memory" not in l)
    assert parse_metrics(text)["gpu_memory_bytes"] is None


def test_vllm_and_sglang_carry_request_latency_and_no_speech_figures():
    vllm = parse_metrics((FIXTURES / "throughput" / "vllm-v0.29.0-cluster-head-1.metrics").read_text())
    assert vllm["request_seconds_count"] is not None and vllm["request_seconds_sum"] is not None
    assert vllm["audio_seconds_total"] is None and vllm["seats"] is None
    sglang = parse_metrics((FIXTURES / "sglang-metrics-2026-10-02-busy-b.txt").read_text())
    assert sglang["request_seconds_count"] == 25
    assert "audio_seconds_total" in SGLANG_UNAVAILABLE


def test_the_probe_sets_depth_known_off_a_transcription_scrape():
    def get(url, timeout):
        return {"/health": (200, b"{}"), "/v1/models": (200, MODELS), "/metrics": (200, BUSY.encode())}[
            url[url.index("/", 8):]]
    fields = probe_backend(Backend(url="http://node:8000", kind=CLUSTER), Config(probe_timeout_s=0.5), now=0.0, get=get)
    assert fields["depth_known"] is True and fields["running"] == 4 and fields["last_error"] == ""
