"""Engine stats the gateway can show: measured throughput, and the lease→backend join.

Every diagnosis in this project came from sampling `vllm:generation_tokens_total` by hand —
it is how "slow" was told apart from "wedged" (box-a frozen at 92438.0 for 80 s while still
reporting running=2), and where the 15 tok/s behind the concurrency benchmark came from. The
prober fetches /metrics every tick and throws those counters away.

The hub's rules are worklog/vllm-gateway-throughput.md, Design 2 (RateMeter). Its Ground replayed
the old gate against the real Hub: a fresh idle vLLM read "not measured", a failed scrape doubled a
true 15 tok/s to 30.0, and a server that went down kept its last rate. The first version of this
file pinned that last one as a feature ("keep the last measurement"); rows 7 and 11 of the frozen
edge table reverse it — a probe that read nothing is a gap, never a repeat.
"""
from __future__ import annotations

from vllm_lb.config import Config, VllmBox
from vllm_lb.hub import Hub
from vllm_lb.probe import SCRAPE_FAMILIES, parse_vllm_metrics

METRICS = """\
vllm:num_requests_running{engine="0",model_name="qwen3.8:latest"} 2.0
vllm:num_requests_waiting{engine="0",model_name="qwen3.8:latest"} 5.0
vllm:num_requests_waiting_by_reason{engine="0",model_name="qwen3.8:latest",reason="capacity"} 5.0
vllm:generation_tokens_total{engine="0",model_name="qwen3.8:latest"} 92438.0
vllm:prompt_tokens_total{engine="0",model_name="qwen3.8:latest"} 95027.0
"""


def test_the_parser_keeps_the_token_counters():
    got = parse_vllm_metrics(METRICS)
    assert got["waiting"] == 5 and got["running"] == 2
    assert got["gen_tokens_total"] == 92438
    assert got["prompt_tokens_total"] == 95027


URL = "http://box-a:8000"
CFG = Config(vllm_boxes=(VllmBox("box-a-gpu", "box-a", URL),))


def scraped(gen: int | None, **families) -> dict:
    """A scrape as parse_vllm_metrics returns it: every family None unless given."""
    scrape = dict.fromkeys(key for key, _, _ in SCRAPE_FAMILIES.values())
    scrape.update(waiting=0, running=0, gen_tokens_total=gen, **families)
    return scrape


def probe(hub: Hub, at: float, gen: int | None = None, *, live: bool = True,
          read: bool = True) -> float | None:
    """One probe result as probe_backend returns it, and the rate the hub then shows. `at` is both
    the tick's wall time and the monotonic read stamp, so each test reads as one timeline."""
    hub.update_backend(URL, live=live, last_probe_at=at, mono_at=at, depth_known=live and read,
                       scrape=scraped(gen) if live and read else None)
    return hub.backend_for(URL).tokens_per_s


def test_the_first_sample_has_no_rate_yet():
    """None is not zero: zero means measured-and-idle."""
    assert probe(Hub(CFG), 1000.0, 1000) is None


def test_the_rate_is_the_delta_between_two_probes():
    h = Hub(CFG)
    probe(h, 1000.0, 1000)
    assert probe(h, 1005.0, 1075) == 15.0                     # +75 in 5 s


def test_a_counter_that_went_BACKWARDS_is_a_restart_not_negative_throughput():
    """vLLM resets its counters when the sidecar restarts — which on this fleet happens every
    time a lease is released and re-taken."""
    h = Hub(CFG)
    probe(h, 1000.0, 90000)
    assert probe(h, 1005.0, 12) is None, "a reset is unknown, never negative"


def test_idle_fresh_vllm_reads_zero():
    """Row 1. Every lease start is a fresh vLLM whose counters start at 0. `hub.py:216` gated the
    rate on a truthy previous total, so an idle fresh server was never measured, and its first
    working interval was dropped too (Ground, replay A: None, None, None, then 30.0)."""
    h = Hub(CFG)
    probe(h, 1000.0, live=False)
    assert [probe(h, 1005.0, 0), probe(h, 1010.0, 0), probe(h, 1015.0, 150)] == [None, 0.0, 30.0]


def test_failed_scrape_never_doubles_the_rate():
    """Row 7. A true 15 tok/s read 30.0 after one failed /metrics: dt ran from the failed probe's
    stamp (Ground, replay B). The failed probe is a gap, and the next good sample a first sample."""
    h = Hub(CFG)
    probe(h, 1000.0, 1000)
    assert probe(h, 1005.0, 1075) == 15.0
    assert probe(h, 1010.0, read=False) is None, "a probe that read nothing never repeats a rate"
    assert probe(h, 1015.0, 1225) is None, "and a delta across it would double the true rate"
    assert probe(h, 1020.0, 1300) == 15.0


def test_two_real_head_scrapes_make_the_hand_sampled_rate():
    """Two CAPTURED reads of the cluster head's /metrics, 10.0 s apart (capture.py --metrics): the hub
    shows exactly the Δgeneration/Δt a hand sample of those two reads gives — SUCCESS 2's arithmetic,
    on real bytes rather than on numbers typed into a test."""
    import json
    from pathlib import Path

    import pytest

    fixtures = Path(__file__).parent / "fixtures" / "throughput"
    reads = json.loads((fixtures / "vllm-v0.29.0-cluster-head.json").read_text())["samples"]
    h = Hub(CFG)
    scrapes = []
    for read in reads:
        scrape = parse_vllm_metrics((fixtures / read["file"]).read_text())
        h.update_backend(URL, live=True, scrape=scrape, depth_known=True,
                         last_probe_at=1000.0 + read["read_at_s"], mono_at=read["read_at_s"])
        scrapes.append(scrape)
    by_hand = ((scrapes[1]["gen_tokens_total"] - scrapes[0]["gen_tokens_total"])
               / (reads[1]["read_at_s"] - reads[0]["read_at_s"]))
    assert h.backend_for(URL).tokens_per_s == pytest.approx(by_hand, rel=1e-12)
    assert by_hand == pytest.approx(69.0, abs=0.1)


def test_down_server_has_no_rate():
    """Row 11. A server that went DOWN kept its last rate, and the page drew it whatever `live`
    said (Ground, defect C)."""
    h = Hub(CFG)
    probe(h, 1000.0, 1000)
    assert probe(h, 1005.0, 1150) == 30.0
    assert probe(h, 1010.0, live=False) is None
