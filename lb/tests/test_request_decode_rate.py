"""THE REQUESTS CARD's DECODE column ADDS UP to the engine's own GENERATED tok/s — operator, 2026-10-16: "requests
tok/s doesn't add up to official total" (rows Δ… · 8.6/s summing to ~44/s beside ~180 tok/s).

Three causes, each pinned here:
  1. the row's Δ counts SSE content EVENTS, not tokens — no relayed stream carries a per-chunk token count (every
     captured body, fixtures/throughput/*.body: usage rides `message_start` and the final `message_delta` /
     usage chunk ONLY), and speculative decoding packs several tokens per event (the captured cluster stream: 32
     tokens in 9 events);
  2. its rate was a lifetime average since `message_start` while the total is the engine's last two-probe window;
  3. the turns in prefill / queued contributed nothing, invisibly.

So a row's `tok_s` is DERIVED and labelled so: the engine's GENERATED tok/s over ITS window, apportioned across the
turns relayed to that server by their content events in THAT SAME window. Turns that finished (or are finishing)
inside the window keep their share as `ended_tok_s`, rows past the 64-row cap as `unlisted_tok_s`, so

    Σ rows' tok_s + ended_tok_s + unlisted_tok_s == the engine's tok/s     (± 0.05 per rounded figure)

and with no figure for the total, NO row gets a number: each says why (the engine's own `why`).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.fakes import FakeBackend
from tests.test_request_queue import Clock, _H, anthropic_stream, claude_body, in_thread, sse, stack, wait  # noqa: F401
from vllm_lb import request_queue as rq
from vllm_lb.config import Config, VllmBox
from vllm_lb.hub import Hub
from vllm_lb.probe import SCRAPE_FAMILIES
from vllm_lb.proxy import _Turn
from vllm_lb.usage_tap import UsageTap

URL = "http://head"


def delta(i: int = 0) -> bytes:
    return sse("content_block_delta", {"type": "content_block_delta", "index": i,
                                       "delta": {"type": "text_delta", "text": " w"}})


def _tolerance(n_figures: int) -> float:
    """Each figure is rounded to 0.1: the sum of n of them is off by at most 0.05 each."""
    return 0.05 * n_figures + 1e-9


# ── the tap: content events on its own clock, countable over any window ──────────────────────────────

def test_the_tap_counts_its_content_events_inside_a_window():
    clock = Clock()
    tap = UsageTap("/v1/messages", 200, "text/event-stream", clock=clock)
    stream = anthropic_stream(n=6)
    clock.t = 1.0
    tap.feed(stream[0] + stream[1])                       # message_start, the thinking block's start
    for k, at in enumerate((2.0, 3.0, 4.0, 5.0, 6.0, 7.0)):
        clock.t = at
        tap.feed(stream[2 + k])
    assert tap.events_between(0.0, 10.0) == 6
    assert tap.events_between(2.0, 5.0) == 3, "(start, end]: 3.0, 4.0, 5.0"
    assert tap.events_between(7.0, 9.0) == 0
    assert tap.progress()["deltas"] == 6, "the lifetime count is unchanged"


def test_an_openai_streams_role_and_usage_chunks_are_not_content_events():
    clock = Clock(1.0)
    tap = UsageTap("/v1/chat/completions", 200, "text/event-stream", clock=clock)
    tap.feed(b'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}\n\n')
    clock.t = 2.0
    tap.feed(b'data: {"choices":[{"index":0,"delta":{"content":"hi"}}]}\n\n')
    clock.t = 3.0
    tap.feed(b'data: {"choices":[{"index":0,"delta":{"reasoning":"hm"}}]}\n\n')
    clock.t = 4.0
    tap.feed(b'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}\n\n')
    assert tap.events_between(0.0, 10.0) == 2


# ── the engine's window: the probe instant it ended at, on the relay's monotonic clock ────────────────

def test_the_hub_says_the_generated_rate_and_the_window_it_was_taken_over():
    cfg = Config(vllm_boxes=(VllmBox("g", "box-a", URL),), listen_host="127.0.0.1", listen_port=0)
    hub = Hub(cfg)

    def probe(at: float, gen: int) -> None:
        scrape = dict.fromkeys(key for key, _, _ in SCRAPE_FAMILIES.values())
        scrape.update({"waiting": 0, "running": 1, "gen_tokens_total": gen})
        hub.update_backend(URL, live=True, last_probe_at=1e9 + at, mono_at=at, depth_known=True, waiting=0,
                           running=1, served_models=("m",), scrape=scrape)

    probe(100.0, 1000)
    first = hub.generated_window(URL)
    assert first["tok_s"] is None and first["why"] == "first sample"
    probe(105.0, 1900)
    w = hub.generated_window(URL)
    assert (w["tok_s"], w["window_s"], w["end_at"], w["why"]) == (pytest.approx(180.0), pytest.approx(5.0), 105.0, None)
    assert hub.generated_window("http://nobody") is None


# ── the apportioning ─────────────────────────────────────────────────────────────────────────────────

def _turn(started: float, events_at: list[float], *, prompt: str, first_at: float | None = None, stop=False):
    clock = Clock(started)
    t = _Turn(_H(), body=claude_body(prompt))
    t.started, t.holding, t.backend, t.sent_at = started, False, URL, started
    stream = anthropic_stream(n=max(1, len(events_at)))
    if events_at or first_at is not None:
        t.tap = UsageTap("/v1/messages", 200, "text/event-stream", clock=clock)
        clock.t = first_at if first_at is not None else events_at[0]
        t.tap.feed(stream[0] + stream[1])
        t.first_at = clock.t
        for at in events_at:
            clock.t = at
            t.tap.feed(delta())
        if stop:
            t.tap.feed(stream[-2])                        # message_delta: its stop reason — finishing
    return t


def _head():
    return SimpleNamespace(url=URL, live=True, depth_known=True, waiting=0, running=3)


def _window(tok_s, why=None, end_at=50.0, window_s=5.0):
    return {URL: {"tok_s": tok_s, "why": why, "window_s": window_s, "end_at": end_at}}


def test_the_rows_tok_s_add_up_to_the_engines_generated_rate_over_its_window():
    """180 tok/s over (45, 50]: a turn with 30 events in it, one with 10 before AND 10 inside it (its lifetime
    average would mislead), one prefilling (none: 0 tok/s, said), and one that FINISHED inside the window."""
    a = _turn(1.0, [45.0 + 0.1 * (k + 1) for k in range(30)], prompt="a")
    b = _turn(2.0, [10.0 + k for k in range(10)] + [46.0 + 0.3 * k for k in range(10)], prompt="b")
    prefill = _turn(40.0, [], prompt="p")
    gone = _turn(3.0, [44.0, 45.5, 46.5, 47.5, 48.5], prompt="g")
    q = rq.queue_view([a, b, prefill], [], [_head()], now=51.0, rates=_window(180.0), ended=[gone])
    rows = {r["prompt"].split()[0]: r for r in q["inflight"]}
    assert rows["a"]["tok_s"] == pytest.approx(180.0 * 30 / 44, abs=0.05)
    assert rows["b"]["tok_s"] == pytest.approx(180.0 * 10 / 44, abs=0.05)
    assert rows["p"]["tok_s"] == 0.0 and rows["p"]["window_events"] == 0, "a prefilling turn: 0, said"
    assert (rows["a"]["window_events"], rows["b"]["window_events"]) == (30, 10)
    d = q["decode"]
    per = d["backends"][URL]
    assert per["engine_tok_s"] == 180.0 and per["window_s"] == 5.0 and per["events"] == 44
    assert per["ended_tok_s"] == pytest.approx(180.0 * 4 / 44, abs=0.05)
    added = sum(r["tok_s"] for r in q["inflight"]) + per["ended_tok_s"] + per["unlisted_tok_s"]
    assert abs(added - 180.0) <= _tolerance(len(q["inflight"]) + 2), f"{added} != 180.0"
    assert per["rows_tok_s"] == pytest.approx(180.0 * 40 / 44, abs=0.05)
    assert d["engine_tok_s"] == 180.0 and d["rows_tok_s"] == per["rows_tok_s"] and d["unmeasured"] == 0
    assert d["derived"] == rq.DERIVED_DECODE and "derived" in rq.DERIVED_DECODE and "events" in rq.DERIVED_DECODE
    assert all(r["tok_s_why"] is None for r in q["inflight"])


def test_a_finishing_turn_is_no_row_but_keeps_its_share_of_the_window():
    a = _turn(1.0, [46.0, 47.0, 48.0], prompt="a")
    fin = _turn(2.0, [46.5], prompt="f", stop=True)
    q = rq.queue_view([a, fin], [], [_head()], now=51.0, rates=_window(100.0))
    assert [r["prompt"].split()[0] for r in q["inflight"]] == ["a"]
    per = q["decode"]["backends"][URL]
    assert q["inflight"][0]["tok_s"] == 75.0 and per["ended_tok_s"] == 25.0


def test_rows_past_the_cap_keep_their_share_as_unlisted():
    turns = [_turn(float(i), [46.0 + 0.01 * i], prompt=f"t{i}") for i in range(rq.INFLIGHT_ROWS + 6)]
    q = rq.queue_view(turns, [], [_head()], now=51.0, rates=_window(70.0))
    per = q["decode"]["backends"][URL]
    assert per["unlisted_tok_s"] == pytest.approx(6.0, abs=0.05)
    added = sum(r["tok_s"] for r in q["inflight"]) + per["unlisted_tok_s"] + per["ended_tok_s"]
    assert abs(added - 70.0) <= _tolerance(len(q["inflight"]) + 2)


def test_no_total_no_number_each_row_says_why():
    a = _turn(1.0, [46.0, 47.0], prompt="a")
    q = rq.queue_view([a], [], [_head()], now=51.0, rates=_window(None, why="first sample", window_s=None))
    r = q["inflight"][0]
    assert r["tok_s"] is None and r["tok_s_why"] == "first sample"
    per = q["decode"]["backends"][URL]
    assert per["engine_tok_s"] is None and per["rows_tok_s"] is None and per["why"] == "first sample"
    assert q["decode"]["engine_tok_s"] is None and q["decode"]["unmeasured"] == 1


def test_a_server_never_probed_is_no_number_either():
    a = _turn(1.0, [46.0], prompt="a")
    q = rq.queue_view([a], [], [_head()], now=51.0, rates={})
    assert q["inflight"][0]["tok_s"] is None and q["inflight"][0]["tok_s_why"]


def test_tokens_counted_but_no_event_relayed_in_the_window_is_not_apportioned():
    """The engine generated, but no turn relayed here sent a content event in its window (another client, or the
    events not out yet): there is nothing to apportion by, so no row is given a number."""
    a = _turn(1.0, [10.0, 11.0], prompt="a")
    q = rq.queue_view([a], [], [_head()], now=51.0, rates=_window(50.0))
    assert q["inflight"][0]["tok_s"] is None and "no content event" in q["inflight"][0]["tok_s_why"]
    zero = rq.queue_view([a], [], [_head()], now=51.0, rates=_window(0.0))
    assert zero["inflight"][0]["tok_s"] == 0.0, "the engine measured 0: every row's share of it is 0"


def test_a_window_older_than_the_events_kept_is_refused():
    a = _turn(1.0, [46.0], prompt="a")
    q = rq.queue_view([a], [], [_head()], now=50.0 + rq.EVENTS_KEPT_S + 1, rates=_window(10.0))
    assert q["inflight"][0]["tok_s"] is None and "kept" in q["inflight"][0]["tok_s_why"]


def test_through_the_proxy_two_live_streams_add_up_to_the_engines_figure(stack):
    """End to end: two real turns streaming through the proxy, the hub probed twice with the engine's counter on the
    relay's monotonic clock — /lb/status `queue` gives each row a share, and they add up to the engine's figure."""
    import time as _time
    v = FakeBackend()
    v.stream_chunks, v.stream_delay_s = anthropic_stream(n=60), 0.05
    proxy, hub = stack(v, probe_interval_s=0.5)

    def probe(gen: int) -> None:
        scrape = dict.fromkeys(key for key, _, _ in SCRAPE_FAMILIES.values())
        scrape.update({"waiting": 0, "running": 2, "gen_tokens_total": gen})
        hub.update_backend(v.url, live=True, last_probe_at=_time.time(), mono_at=_time.monotonic(),
                           depth_known=True, waiting=0, running=2, served_models=tuple(v.served), scrape=scrape)

    in_thread(proxy.port, claude_body("one"))
    in_thread(proxy.port, claude_body("two"))
    wait(lambda: len([r for r in proxy.queue_view()["inflight"] if r["deltas"] >= 3]) == 2)
    probe(1000)
    _time.sleep(0.8)
    probe(1144)                                           # 144 tokens over ~0.8 s
    q = proxy.queue_view()
    per = q["decode"]["backends"][v.url]
    assert per["why"] is None and per["engine_tok_s"] > 0 and per["events"] > 0, per
    rows = [r for r in q["inflight"] if r["state"] == rq.STREAMING]
    assert rows and all(isinstance(r["tok_s"], float) and r["window_events"] > 0 for r in rows), rows
    added = sum(r["tok_s"] for r in q["inflight"]) + per["ended_tok_s"] + per["unlisted_tok_s"]
    assert abs(added - per["engine_tok_s"]) <= _tolerance(len(q["inflight"]) + 3)
    assert q["decode"]["engine_tok_s"] == per["engine_tok_s"]
    assert per["engine_tok_s"] == pytest.approx(hub.generated_window(v.url)["tok_s"], abs=0.05)


def test_a_turn_that_ended_inside_the_window_keeps_its_share(stack):
    import time as _time
    v = FakeBackend()
    v.stream_chunks, v.stream_delay_s = anthropic_stream(n=8), 0.02
    proxy, hub = stack(v, probe_interval_s=0.5)

    def probe(gen: int) -> None:
        scrape = dict.fromkeys(key for key, _, _ in SCRAPE_FAMILIES.values())
        scrape.update({"waiting": 0, "running": 0, "gen_tokens_total": gen})
        hub.update_backend(v.url, live=True, last_probe_at=_time.time(), mono_at=_time.monotonic(),
                           depth_known=True, waiting=0, running=0, served_models=tuple(v.served), scrape=scrape)

    probe(0)
    out: dict = {}
    in_thread(proxy.port, claude_body("gone"), out)
    wait(lambda: "r" in out and proxy.queue_view()["recent"])
    _time.sleep(0.6)
    probe(17)
    q = proxy.queue_view()
    per = q["decode"]["backends"][v.url]
    assert q["inflight"] == [] and per["why"] is None
    assert per["ended_tok_s"] == pytest.approx(per["engine_tok_s"], abs=0.05) and per["rows_tok_s"] == 0.0
