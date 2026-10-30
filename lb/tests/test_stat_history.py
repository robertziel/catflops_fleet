"""Every stat has a past, and the LB owns it — not the page.

`gateway.js` kept `const history = new Map()` and pushed one tok/s per poll, so the sparkline was
born empty on every reload and told the operator nothing about the minute before they opened the
tab. The operator's tile leads with history, so the history has to outlive the page.

The ring is appended from PROBE results only. `update_backend` is also how the proxy adjusts
`inflight` around a request; counting those would sample the same instant many times and make a
busy backend's history denser than an idle one's — a graph of our own traffic shape, not of the
backend.

Each sample is TIME-STAMPED (`history.at`, the LB's clock) so the page can place it in a fixed time
bucket, and a probe that read nothing is a gap, never a repeat of the last rate
(worklog/vllm-gateway-throughput.md, Design 3 and edge row 7).
"""
from __future__ import annotations

import http.client
import json

import pytest

from vllm_lb.config import Config, VllmBox
from vllm_lb.hub import Hub
from vllm_lb.probe import SCRAPE_FAMILIES
from vllm_lb.proxy import Proxy
from vllm_lb.state import VLLM

URL = "http://box-a:8000"
CFG = Config(vllm_boxes=(VllmBox("box-a-gpu", "box-a", URL),))


def _probe(hub, *, at, running=0, waiting=0, gen=None, live=True, read=True):
    """A probe result as probe_backend returns it; `at` is both its wall and its monotonic stamp."""
    scrape = None
    if live and read:
        scrape = dict.fromkeys(key for key, _, _ in SCRAPE_FAMILIES.values())
        scrape.update(waiting=waiting, running=running, gen_tokens_total=gen)
    hub.update_backend(URL, live=live, running=running, waiting=waiting, scrape=scrape,
                       depth_known=scrape is not None, last_probe_at=at, mono_at=at)


def test_the_hub_keeps_a_bounded_ring_per_backend():
    hub = Hub(Config(**{**CFG.__dict__, "stat_history": 4}))
    for i in range(10):
        _probe(hub, at=1000.0 + 5 * i, running=i, gen=10 * i)
    h = hub.history(URL)
    assert h["depth"] == [6, 7, 8, 9], "oldest first, newest last, bounded"
    assert h["gen_tok_s"] == [2.0, 2.0, 2.0, 2.0]
    assert h["at"] == [1030.0, 1035.0, 1040.0, 1045.0]
    assert h["tok_s"] == h["gen_tok_s"], "the old key stays, as an alias, until the page migrates"


def test_depth_is_the_same_number_the_router_demotes_on():
    hub = Hub(CFG)
    _probe(hub, at=1000.0, running=2, waiting=3)
    assert hub.history(URL)["depth"] == [5], "waiting + running, as depthOf() means it"


def test_an_unknown_rate_is_a_gap_not_a_zero():
    """tok/s is None when the counter went backwards (the sidecar restarted). Drawing that as 0
    would say "measured, and idle" about a backend we simply could not measure."""
    hub = Hub(CFG)
    _probe(hub, at=1000.0, gen=1000)
    _probe(hub, at=1005.0, gen=1060)
    _probe(hub, at=1010.0, gen=10)
    assert hub.history(URL)["gen_tok_s"] == [None, 12.0, None]


def test_failed_probe_is_a_gap_not_a_repeat():
    """Row 7. The failed probe's sample repeated the stale rate: a sample that measured nothing,
    drawn as a measurement (Ground, replay B: [None, 15.0, 15.0, 30.0])."""
    hub = Hub(CFG)
    _probe(hub, at=1000.0, gen=1000)
    _probe(hub, at=1005.0, gen=1075)
    _probe(hub, at=1010.0, read=False)
    _probe(hub, at=1015.0, gen=1225)
    assert hub.history(URL)["gen_tok_s"] == [None, 15.0, None, None]


def test_the_ring_says_when_and_whether_the_server_was_live():
    """The page buckets samples by time, and a bucket where a LIVE server has no rate is a gap in
    the fleet's line — so each sample carries its LB time and liveness."""
    hub = Hub(CFG)
    _probe(hub, at=1000.0, gen=0)
    _probe(hub, at=1007.5, live=False)
    h = hub.history(URL)
    assert h["at"] == [1000.0, 1007.5] and h["live"] == [True, False]


def test_a_probe_too_close_to_the_last_is_not_a_sample():
    """RateMeter rule 4 at the hub: a probe 1 s after the last one taken changes neither the rate
    nor the ring."""
    hub = Hub(CFG)
    _probe(hub, at=1000.0, gen=1000)
    _probe(hub, at=1005.0, gen=1075)
    _probe(hub, at=1006.0, gen=1080)
    assert hub.history(URL)["at"] == [1000.0, 1005.0]
    assert hub.backend_for(URL).tokens_per_s == 15.0


def test_adjusting_inflight_does_not_sample_the_ring():
    hub = Hub(CFG)
    _probe(hub, at=1000.0, running=1)
    hub.update_backend(URL, inflight=1)
    hub.update_backend(URL, inflight=0)
    assert hub.history(URL)["depth"] == [1], "only probes sample; our own traffic does not"


def test_lb_status_ships_the_ring_with_each_backend():
    hub = Hub(CFG)
    _probe(hub, at=1000.0, running=1, waiting=1, gen=1000)
    _probe(hub, at=1005.0, running=1, waiting=1, gen=1075)
    cfg = Config(**{**CFG.__dict__, "listen_host": "127.0.0.1", "listen_port": 0})
    proxy = Proxy(cfg, hub)
    proxy.start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=10)
        conn.request("GET", "/lb/status")
        body = json.loads(conn.getresponse().read())
    finally:
        proxy.stop()
    (backend,) = [b for b in body["backends"] if b["kind"] == VLLM]
    history = backend["history"]
    assert history["depth"] == [2, 2] and history["at"] == [1000.0, 1005.0]
    assert history["gen_tok_s"] == [None, 15.0] and history["tok_s"] == [None, 15.0]
    assert history["prompt_tok_s"] == [None, None], "prompt was not exposed: unknown, never 0"
    assert history["prefill_tok_s"] == [None, None], "prefill was not exposed: unknown, never 0"
    assert history["computed_tok_s"] == [None, None], "computed prompt was not exposed: unknown, never 0"


def test_the_prefill_ring_is_the_prefill_speed_the_saturation_row_heads():
    """PREFILL on the saturation row headed "1,170 tok/s · 5 requests over 5m 1s" over an empty slot,
    while GENERATED beside it drew its ring: `prefill_speed` was a scalar with no history behind it. The
    ring is that same figure, sampled on every probe — a gap while no request has finished its prefill
    in the window, then a step each time one does — and never the billed prompt rate, which counts the
    prefix cache's hits as if they were read."""
    hub = Hub(CFG)
    headed = []
    # (at, prefill tokens computed, prefill seconds, requests finished, prompt tokens billed)
    for at, tokens, seconds, finished, billed in ((1000.0, 0, 0.0, 0, 0),
                                                  (1005.0, 0, 0.0, 0, 0),
                                                  (1010.0, 8000, 4.0, 1, 80000),
                                                  (1015.0, 8000, 4.0, 1, 80000),
                                                  (1020.0, 11000, 6.0, 2, 110000)):
        scrape = dict.fromkeys(key for key, _, _ in SCRAPE_FAMILIES.values())
        scrape.update(waiting=0, running=1, gen_tokens_total=int(at), prompt_tokens_total=billed,
                      prefill_computed_tokens_sum=tokens, prefill_time_sum=seconds,
                      prefill_requests_total=finished)
        hub.update_backend(URL, live=True, running=1, waiting=0, scrape=scrape, depth_known=True,
                           last_probe_at=at, mono_at=at)
        headed.append(hub.throughput(hub.backend_for(URL), at)["prefill_speed"]["tok_s"])
    ring = hub.history(URL)["prefill_tok_s"]
    assert ring == headed, "the dots under PREFILL are the number PREFILL heads, probe by probe"
    assert ring[:2] == [None, None], "no request has finished its prefill: a gap, never 0 tok/s"
    assert ring[2:] == pytest.approx([2000.0, 2000.0, 11000 / 6.0]), "a staircase: it moves when one finishes"
    assert hub.history(URL)["prompt_tok_s"][2:] != pytest.approx(ring[2:]), "the billed rate is not this line"


def test_the_computed_ring_is_the_prefill_tile_s_own_past_never_the_billed_rate():
    """PREFILL on the card heads `prompt.computed_tok_s` — the prompt tokens the engine computed per
    second, PROMPT's window with the prefix cache's hits left out. Its ring is that figure probe by
    probe: a gap until it is measured, and never the billed rate PROMPT draws."""
    hub = Hub(CFG)
    headed = []
    # (at, prompt tokens billed, of which computed, of which cached)
    for at, billed, computed, cached in ((1000.0, 0, 0, 0), (1005.0, 50000, 5000, 45000),
                                         (1010.0, 100000, 10000, 90000), (1015.0, 150000, 15000, 135000)):
        scrape = dict.fromkeys(key for key, _, _ in SCRAPE_FAMILIES.values())
        scrape.update(waiting=0, running=1, gen_tokens_total=int(at), prompt_tokens_total=billed,
                      prompt_computed_total=computed, prompt_cached_total=cached)
        hub.update_backend(URL, live=True, running=1, waiting=0, scrape=scrape, depth_known=True,
                           last_probe_at=at, mono_at=at)
        headed.append(hub.throughput(hub.backend_for(URL), at)["prompt"]["computed_tok_s"])
    ring = hub.history(URL)["computed_tok_s"]
    assert ring == headed, "the dots under PREFILL are the number PREFILL heads, probe by probe"
    assert ring[0] is None, "one read is no rate: a gap, never 0 tok/s"
    assert ring[-1] is not None and ring[-1] > 0, ring
    billed_ring = hub.history(URL)["prompt_tok_s"]
    assert ring[-1] == pytest.approx(billed_ring[-1] / 10), "a tenth was computed: the billed rate is not this line"
    # …and the engine's own totals since it loaded ride the same block, as last read.
    totals = hub.throughput(hub.backend_for(URL), 1015.0)["engine_totals"]
    assert totals == {"generated": 1015, "prompt": 150000, "computed": 15000, "started_at": None}, totals


def test_the_engines_own_kv_pool_rides_the_throughput_block():
    """Audit #1: the page divided the pin by one model's bytes per token. A serving engine says what it really
    allocated (`vllm:cache_config_info`); `throughput.kv_pool` carries it — each figure null when the engine
    does not expose it, the block null when it exposes neither or /metrics is not answering."""
    hub = Hub(CFG)

    def pool_after(at, **found):
        scrape = dict.fromkeys(key for key, _, _ in SCRAPE_FAMILIES.values())
        scrape.update(waiting=0, running=1, gen_tokens_total=int(at), **found)
        hub.update_backend(URL, live=True, running=1, waiting=0, scrape=scrape, depth_known=True,
                           last_probe_at=at, mono_at=at)
        return hub.throughput(hub.backend_for(URL), at)["kv_pool"]

    assert hub.throughput(hub.backend_for(URL), 999.0)["kv_pool"] is None, "nothing read yet"
    assert pool_after(1000.0) is None, "an engine that exposes no cache_config_info"
    assert pool_after(1005.0, kv_pool_tokens=1_302_238, kv_pool_bytes=48_000_000_000) == {
        "tokens": 1_302_238, "bytes": 48_000_000_000, "blocks": None, "block_tokens": None}
    assert pool_after(1010.0, kv_pool_tokens=1_302_238) == {"tokens": 1_302_238, "bytes": None, "blocks": None,
                                                           "block_tokens": None}
    # …and the pool in BLOCKS, with the tokens one holds, as the engine resolved them (generation 105).
    assert pool_after(1015.0, kv_pool_blocks=1689, kv_block_tokens=816) == {
        "tokens": None, "bytes": None, "blocks": 1689, "block_tokens": 816}
    hub.update_backend(URL, live=True, running=0, waiting=0, scrape=None, depth_known=False,
                       last_probe_at=1020.0, mono_at=1020.0)
    assert hub.throughput(hub.backend_for(URL), 1020.0)["kv_pool"] is None, "/metrics did not answer"


def test_a_queue_nobody_read_is_a_gap_and_forwarded_requests_are_not_in_it():
    """QUEUE heads the engine's running + waiting, and "—" when /metrics did not answer. Its ring drew
    probe.py's placeholder zeros for a server that was down — a flat line under "—" on every idle card,
    a queue "measured" empty on a server that was not there — and still added the requests this LB had
    forwarded and the engine had not admitted, a number the card stopped heading in 5660b68."""
    hub = Hub(CFG)
    _probe(hub, at=1000.0, running=14, waiting=2, gen=0)
    hub.update_backend(URL, inflight=35)          # forwarded by us: routing counts them, the card does not
    _probe(hub, at=1005.0, running=14, waiting=2, gen=75)
    _probe(hub, at=1010.0, read=False)            # up, but /metrics did not answer
    hub.update_backend(URL, inflight=0)
    _probe(hub, at=1015.0, live=False)            # down: probe.py's 0 running, 0 waiting
    assert hub.history(URL)["depth"] == [16, 16, None, None]
