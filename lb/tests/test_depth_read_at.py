"""WHEN the engine's running / waiting were read: `depth_read_at` on every /lb/status backend.

Operator, 2026-10-30: "why shows only one request if queue tells 2?". The server card's QUEUE tile is the
ENGINE's own gauges off its last /metrics scrape; the REQUESTS card lists this gateway's relays NOW. Short turns
(~1 s each) make the two disagree by one or two, and the panel says so with the scrape's age ("engine read 3 s
ago"). `last_probe_at` is the tick's wall clock, taken BEFORE /health and /v1/models — under load seconds before
/metrics answers — so it is not the read. `depth_read_at` is: the tick's wall clock plus the monotonic time to the
moment the /metrics body was read, and null whenever no running / waiting were read (never a stale or guessed time).
"""
from __future__ import annotations

from tests.fakes import FakeBackend, free_port
from tests.test_throughput_status import T0, Fleet, HEAD
from vllm_lb.config import Config
from vllm_lb.probe import probe_backend
from vllm_lb.state import VLLM, Backend

CFG = Config(probe_timeout_s=0.5)


def test_a_probe_that_read_the_depth_says_when():
    f = FakeBackend(served=("qwen3.8:latest",), max_model_len=32768)
    f.waiting, f.running = 0, 2
    out = probe_backend(Backend(kind=VLLM, url=f.url), CFG, now=7.0)
    assert out["depth_known"] is True and out["running"] == 2
    assert 7.0 <= out["depth_read_at"] < 9.0, out["depth_read_at"]


def test_a_probe_that_read_nothing_has_no_read_time():
    out = probe_backend(Backend(kind=VLLM, url=f"http://127.0.0.1:{free_port()}"), CFG, now=1.0)
    assert out["depth_known"] is False and out["depth_read_at"] is None


def test_status_carries_the_read_time_beside_the_gauges():
    fleet = Fleet()
    fleet.probe(HEAD, T0, running=2)
    fleet.hub.update_backend(HEAD, depth_read_at=T0 + 0.4)
    b = fleet.status()[HEAD]
    assert b["running"] == 2 and b["depth_read_at"] == T0 + 0.4
