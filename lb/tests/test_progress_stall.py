"""WEDGED means nothing MOVES while work waits — a long prefill generates no tokens and is not a fault.

`stalled_s` (in /lb/status) is the seconds since nothing moved while the backend had work; null when
it has none. MOVED means `vllm:generation_tokens_total` or `vllm:prompt_tokens_total` advanced, or
`vllm:kv_cache_usage_perc` changed. The clock starts when work ARRIVES, never at the last token of
some earlier turn. (Audit 2026-09-15: an 84 KB replay in prefill painted box-a red WEDGED eight
seconds before it answered.)

vLLM credits a prompt's tokens once, at the request's FIRST output token, and emits nothing for a
partial chunked-prefill step (worklog/vllm-gateway-throughput.md, Q1b), so both counters stay flat
through a long prefill. What moves is KV usage, which the engine sets every step. This file's first
version fed the hub a mid-prefill jump in the prompt counter: a shape vLLM never produces.
"""
from __future__ import annotations

import pytest

from vllm_lb.config import Config, VllmBox
from vllm_lb.hub import Hub
from vllm_lb.probe import SCRAPE_FAMILIES
from vllm_lb.proxy import Proxy

URL = "http://s1:8000"
CFG = Config(vllm_boxes=(VllmBox("box-a-gpu", "box-a", URL),), 
             listen_host="127.0.0.1", listen_port=0)
WEDGED_AFTER_S = 120      # gateway.js draws the line here


class Clock:
    def __init__(self, t: float) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def stalled(hub: Hub):
    (backend,) = [b for b in Proxy(CFG, hub).status()["backends"] if b["url"] == URL]
    return backend["stalled_s"]


def probe(hub: Hub, clock: Clock, *, running: int, gen: int, prompt: int,
          kv: float | None = None) -> None:
    """A probe result as probe_backend returns it: the scrape, its alias counters, both stamps."""
    scrape = dict.fromkeys(key for key, _, _ in SCRAPE_FAMILIES.values())
    scrape.update(waiting=0, running=running, gen_tokens_total=gen, prompt_tokens_total=prompt,
                  kv_cache_usage=kv)
    hub.update_backend(URL, live=True, waiting=0, running=running, depth_known=True,
                       scrape=scrape, gen_tokens_total=gen, prompt_tokens_total=prompt,
                       last_probe_at=clock.t, mono_at=clock.t)


def test_no_work_is_not_a_stall():
    clock = Clock(100.0)
    hub = Hub(CFG, clock=clock)
    probe(hub, clock, running=0, gen=10, prompt=10)
    clock.t = 400.0
    assert stalled(hub) is None


def test_the_clock_starts_when_work_arrives_not_at_the_last_token():
    clock = Clock(100.0)
    hub = Hub(CFG, clock=clock)
    probe(hub, clock, running=0, gen=10, prompt=10)
    clock.t = 1000.0
    probe(hub, clock, running=1, gen=10, prompt=10)
    clock.t = 1008.0
    assert stalled(hub) == 8.0


def test_our_own_request_starts_the_clock_between_probes():
    clock = Clock(100.0)
    hub = Hub(CFG, clock=clock)
    probe(hub, clock, running=0, gen=10, prompt=10)
    clock.t = 1000.0
    hub.note_request_start(URL)
    clock.t = 1005.0
    assert stalled(hub) == 5.0


def test_the_first_token_credits_the_prompt_and_resets_the_clock():
    """The counter shape vLLM really exposes: both flat through the prefill, then the prompt credited
    in one lump together with the first generated token."""
    clock = Clock(1000.0)
    hub = Hub(CFG, clock=clock)
    probe(hub, clock, running=1, gen=10, prompt=10)
    clock.t = 1060.0
    probe(hub, clock, running=1, gen=11, prompt=5010)
    clock.t = 1065.0
    assert stalled(hub) == 5.0


def test_kv_growth_is_prefill_progress():
    """Row 22. A lone 251,509-token prompt at ~851 tok/s prefills for ~295 s with both counters flat,
    and read WEDGED at 120 s. KV usage rising at every probe is that prefill moving."""
    clock = Clock(1000.0)
    hub = Hub(CFG, clock=clock)
    kv = 0.12
    probe(hub, clock, running=1, gen=10, prompt=10, kv=kv)
    longest = 0.0
    for _ in range(60):                            # five minutes of 5 s probes
        clock.t += 2.5
        longest = max(longest, stalled(hub))       # between probes, too
        clock.t += 2.5
        kv = round(kv + 0.003, 4)
        probe(hub, clock, running=1, gen=10, prompt=10, kv=kv)
    clock.t += 3.0
    assert stalled(hub) == 3.0 and longest == 2.5, "never WEDGED while the prefill moves"


@pytest.mark.parametrize("kv", [None, 0.2], ids=["kv not exposed", "kv flat"])
def test_nothing_moving_under_work_accumulates(kv):
    """Flat counters and a flat (or unexposed) KV gauge: nothing moved, so the seconds add up."""
    clock = Clock(1000.0)
    hub = Hub(CFG, clock=clock)
    probe(hub, clock, running=1, gen=10, prompt=10, kv=kv)
    clock.t = 1200.0
    probe(hub, clock, running=1, gen=10, prompt=10, kv=kv)
    clock.t = 1300.0
    assert stalled(hub) == 300.0 > WEDGED_AFTER_S
