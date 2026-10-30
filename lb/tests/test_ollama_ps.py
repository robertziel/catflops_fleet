"""ollama's loaded model, from `GET /api/ps` and that response's own `Date` (`probe.parse_ollama_ps`): what a box's
OWN ollama holds — broker_parrot's, read for the cluster's RAM gate (cluster_loop `_holders`) — with an unload countdown
that is never negative and never read on a skewed clock. (The LB's own ollama FALLBACK, whose loaded/serving states
these bodies also drew, is retired, 2026-10-17: its tests went with it.)

worklog/vllm-gateway-throughput.md Q3-Q4, edge rows 37-38. The two /api/ps bodies are CAPTURED from box-c's ollama
0.33.2 (fixtures/throughput/ollama-ps-{before,after}.json).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from vllm_lb.probe import parse_ollama_ps

FIXTURES = Path(__file__).parent / "fixtures" / "throughput"
#: What the captured `expires_at` (21:28:31.023366889Z) says past its `Date` (21:23:31 GMT), to µs.
CAPTURED_COUNTDOWN_S = 300.023366


def captured(label: str) -> tuple[bytes, str]:
    meta = json.loads((FIXTURES / f"ollama-ps-{label}.json").read_text())
    return meta["body"].encode(), dict(meta["headers"])["Date"]


def resident(**model) -> bytes:
    """The captured resident body with its one model's fields changed."""
    body = json.loads(captured("after")[0])
    body["models"][0].update(model)
    return json.dumps(body).encode()


def test_the_captured_bodies_parse():
    ps = parse_ollama_ps(*captured("after"))
    (model,) = ps["models"]
    assert (model["name"], model["context"]) == ("qwen3.8:latest", 262144)
    assert model["size_bytes"] == model["gpu_bytes"] == 18_714_627_931 and model["offloaded"] is False
    assert model["expires_in_s"] == pytest.approx(CAPTURED_COUNTDOWN_S, abs=1e-6)
    assert model["expiry"] == "counting down"
    assert parse_ollama_ps(*captured("before"))["models"] == []


@pytest.mark.parametrize("expires_at, expiry", [
    ("0001-01-01T00:00:00Z", "in use"),          # Go's zero time: not set yet, still loading
    ("2026-09-15T21:20:00Z", "unloading"),       # before the server's own Date
    ("not a time", "unknown"),
])
def test_expiry_edge_values(expires_at, expiry):
    """Row 37. Never a negative countdown: a zero or past expiry says what it means instead."""
    model = parse_ollama_ps(resident(expires_at=expires_at), captured("after")[1])["models"][0]
    assert model["expires_in_s"] is None and model["expiry"] == expiry


def test_skewed_server_clock():
    """Row 38. ollama's clock an hour ahead of the LB's: the countdown is read against ollama's own
    Date, so the skew changes nothing. Without a Date there is no countdown at all."""
    body = resident(expires_at="2026-09-15T22:27:31Z")
    model = parse_ollama_ps(body, "Tue, 15 Sep 2026 22:23:31 GMT")["models"][0]
    assert model["expires_in_s"] == 240.0 and model["expiry"] == "counting down"
    model = parse_ollama_ps(body, None)["models"][0]
    assert model["expires_in_s"] is None and model["expiry"] == "unknown"


def test_partial_gpu_offload_is_flagged():
    """Q4: `size_vram < size` means part of the model runs on the CPU (broker_parrot's own rule)."""
    model = parse_ollama_ps(resident(size_vram=12_000_000_000), captured("after")[1])["models"][0]
    assert model["offloaded"] is True and model["gpu_bytes"] == 12_000_000_000
