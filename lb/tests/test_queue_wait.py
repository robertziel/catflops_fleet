"""WHY A TURN WAITS at the engine, and the stream facts the REQUESTS card shows for an OpenAI-API client.

The operator (2026-10-12, a one-GPU node at 16 seats): "why only one is streaming, missing context space?". vLLM
admits a waiting request only when a seat is free AND its blocks fit in the KV pool (v1 scheduler: the waiting loop
breaks at `max_num_running_reqs`, and at the head whose `allocate_slots` returns None — FCFS, head of line). The
captured read (fixtures/throughput/vllm-2eb491e-kv-full-queue.metrics, the node under that load): 3 running of 16
seats, 14 waiting (10 `capacity`, 4 `deferred`), KV 0.9878048780487805 = 162 / 164 usable blocks held (num_gpu_blocks
165 less vLLM's null block) — 2 free, and a request takes at least its 6 fixed blocks + 1 (the engine's own figures:
165 / kv_cache_max_concurrency 1.25 = 132 a full-window request = ceil(196608 / 1568) 126 + 6). So nothing more fits:
"kv full", from the engine's own numbers and nothing else.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.fakes import FakeBackend
from vllm_lb.config import Config, VllmBox
from vllm_lb.hub import Hub
from vllm_lb.probe import parse_metrics
from vllm_lb.proxy import Proxy
from vllm_lb import request_queue as rq
from vllm_lb.usage_tap import UsageTap

FIXTURES = Path(__file__).parent / "fixtures"
KV_FULL = FIXTURES / "throughput" / "vllm-2eb491e-kv-full-queue.metrics"


# ── the engine's own reasons, off its /metrics ────────────────────────────────────────────────────────

def test_the_probe_reads_the_engines_waiting_reasons_apart():
    s = parse_metrics(KV_FULL.read_text())
    assert (s["running"], s["waiting"]) == (3, 14)
    assert (s["waiting_capacity"], s["waiting_deferred"]) == (10, 4), "the two reasons vLLM labels"
    assert s["waiting"] == 14, "the exact family, never the by-reason one summed into it"
    assert (s["kv_pool_blocks"], s["kv_block_tokens"], s["kv_max_concurrency"]) == (165, 1568, 1.25)


def test_an_engine_that_publishes_no_reasons_reads_none_not_zero():
    old = "\n".join(line for line in KV_FULL.read_text().splitlines() if "waiting_by_reason" not in line)
    s = parse_metrics(old)
    assert s["waiting"] == 14
    assert s["waiting_capacity"] is None and s["waiting_deferred"] is None
    sglang = parse_metrics((FIXTURES / "sglang-metrics-2026-10-02-busy-a.txt").read_text())
    assert sglang["waiting_capacity"] is None and sglang["waiting_deferred"] is None


# ── the verdict: seats, KV, the engine's deferral, or not known ───────────────────────────────────────

def _facts(**over):
    s = parse_metrics(KV_FULL.read_text())
    base = {"running": s["running"], "waiting": s["waiting"], "waiting_capacity": s["waiting_capacity"],
            "waiting_deferred": s["waiting_deferred"], "kv_cache_usage": s["kv_cache_usage"],
            "kv_pool_blocks": s["kv_pool_blocks"], "kv_block_tokens": s["kv_block_tokens"],
            "kv_max_concurrency": s["kv_max_concurrency"], "max_model_len": 196608, "seats": 16}
    return {**base, **over}


def test_the_captured_queue_waits_for_kv_space_not_for_a_seat():
    a = rq.admission(_facts())
    assert a["wait"] == rq.WAIT_KV
    assert (a["kv_usable_blocks"], a["kv_free_blocks"], a["kv_fixed_blocks"], a["kv_least_blocks"]) == (164, 2, 6, 7)
    assert a["kv_free_tokens"] == 2 * 1568
    assert (a["running"], a["seats"], a["waiting"], a["waiting_capacity"], a["waiting_deferred"]) == (3, 16, 14, 10, 4)
    assert a["kv_pct"] == pytest.approx(98.78, abs=0.01)
    # the running requests' mean context, their fixed blocks left out: (162 / 3 − 6) × 1568
    assert a["mean_context"] == (162 // 3 - 6) * 1568
    for word in ("2 of 164", "7", "3 of 16"):
        assert word in a["why"], f"{word!r} not in the rule: {a['why']}"


def test_every_seat_taken_is_a_wait_for_a_seat_whatever_the_kv():
    a = rq.admission(_facts(running=16, kv_cache_usage=0.5))
    assert a["wait"] == rq.WAIT_SEATS and "16 of 16" in a["why"]


def test_free_blocks_and_a_free_seat_say_not_known_with_the_figures():
    """2026-10-12, 10 s later on the same node: 2 running, 16 waiting, KV 0.6829 (52 free) — neither full by the
    engine's figures; the head's own size (a prefix reloaded from the RAM tier needs its blocks) is not published."""
    a = rq.admission(_facts(running=2, waiting=16, waiting_capacity=16, waiting_deferred=0,
                            kv_cache_usage=0.6829268292682926))
    assert a["wait"] is None and a["kv_free_blocks"] == 52
    assert "52 of 164" in a["why"] and "2 of 16" in a["why"]


def test_only_deferred_requests_wait_on_the_engines_transfer():
    a = rq.admission(_facts(running=2, waiting=3, waiting_capacity=0, waiting_deferred=3, kv_cache_usage=0.5))
    assert a["wait"] == rq.WAIT_DEFERRED


def test_without_the_seats_or_the_pool_the_rule_that_needs_them_is_not_applied():
    no_seats = rq.admission(_facts(seats=None))
    assert no_seats["wait"] == rq.WAIT_KV, "KV full needs no seat count"
    assert no_seats["seats"] is None
    blind = rq.admission(_facts(seats=None, kv_cache_usage=None))
    assert blind["wait"] is None and "KV" in blind["why"]
    no_conc = rq.admission(_facts(kv_max_concurrency=None))
    assert no_conc["kv_fixed_blocks"] is None and no_conc["kv_least_blocks"] == 1, "a request takes ≥ 1 block"
    assert no_conc["wait"] is None, "2 free blocks with the request's fixed blocks unknown prove nothing"
    zero = rq.admission(_facts(kv_max_concurrency=None, kv_cache_usage=1.0))
    assert zero["wait"] == rq.WAIT_KV and zero["kv_free_blocks"] == 0
    idle = rq.admission(_facts(waiting=0, waiting_capacity=0, waiting_deferred=0))
    assert idle["wait"] is None
    gone = rq.admission(None)
    assert gone["wait"] is None and gone["why"]


# ── the queue view carries it per server, with its rows per state ─────────────────────────────────────

@pytest.fixture
def stack():
    created = []

    def make(v):
        cfg = Config(vllm_boxes=(VllmBox("gpu-a", "box-a", v.url),), 
                     listen_host="127.0.0.1", listen_port=0, connect_timeout_s=2.0,
                     first_byte_timeout_s=30.0, admission_wait_s=0)
        hub = Hub(cfg)
        proxy = Proxy(cfg, hub)
        proxy.start()
        created.append(proxy)
        return proxy, hub

    yield make
    for p in created:
        p.stop()


def test_the_status_queue_says_per_server_why_its_turns_wait(stack):
    v = FakeBackend()
    proxy, hub = stack(v)
    scrape = parse_metrics(KV_FULL.read_text())
    hub.update_backend(v.url, live=True, depth_known=True, running=3, waiting=14, max_model_len=196608,
                       last_probe_at=1.0, mono_at=1.0, scrape=scrape)
    q = proxy.queue_view()
    eng = q["backends"][v.url]
    assert eng["wait"] == rq.WAIT_KV and eng["kv_free_blocks"] == 2
    assert eng["seats"] is None, "a classic vLLM publishes no max_num_seqs: its seat rule is not applied"
    assert eng["rows"] == {"streaming": 0, "prefill": 0, "queued": 0, "sent": 0}
    assert q["engine"]["waiting_capacity"] == 10 and q["engine"]["waiting_deferred"] == 4
    json.dumps(q)


# ── an OpenAI-API stream: its thinking, text and tool calls, and its finish reason ────────────────────

def _chunk(delta: dict | None = None, finish: str | None = None, choices=True, usage=None) -> bytes:
    obj = {"id": "chatcmpl-1", "object": "chat.completion.chunk", "model": "m",
           "choices": ([{"index": 0, "delta": delta or {}, "logprobs": None, "finish_reason": finish}]
                       if choices else [])}
    if usage is not None:
        obj["usage"] = usage
    return b"data: " + json.dumps(obj).encode() + b"\n\n"


def test_the_captured_openai_stream_splits_its_reasoning_and_says_its_finish_reason():
    body = (FIXTURES / "throughput" / "cluster-chat-stream-usage.body").read_bytes()
    tap = UsageTap("/v1/chat/completions", 200, "text/event-stream")
    tap.feed(body)
    p = tap.progress()
    reasoning = body.count(b'"delta":{"reasoning"')
    assert reasoning > 0 and p["thinking"] == reasoning, "vLLM's `reasoning` deltas are thinking"
    assert p["text"] == 0 and p["tool"] == 0 and p["tool_uses"] == 0
    assert p["stop_reason"] == "length", "the stream's own finish_reason"
    assert p["output_tokens"] == 32


def test_an_openai_stream_counts_content_and_each_tool_call_once():
    tap = UsageTap("/v1/chat/completions", 200, "text/event-stream")
    chunks = [_chunk({"role": "assistant", "content": ""}),
              _chunk({"reasoning": "think"}), _chunk({"reasoning_content": "more"}),
              _chunk({"content": "hi"}), _chunk({"content": " there"}),
              _chunk({"tool_calls": [{"id": "call_1", "type": "function", "index": 0,
                                      "function": {"name": "Bash", "arguments": ""}}]}),
              _chunk({"tool_calls": [{"index": 0, "function": {"arguments": "{\"a\""}}]}),
              _chunk({"tool_calls": [{"id": "call_2", "type": "function", "index": 1,
                                      "function": {"name": "Read", "arguments": "{}"}}]})]
    for c in chunks:
        tap.feed(c)
    p = tap.progress()
    assert (p["thinking"], p["text"], p["tool"], p["tool_uses"]) == (2, 2, 3, 2)
    assert p["stop_reason"] is None, "no finish_reason yet: still streaming"
    tap.feed(_chunk({}, finish="tool_calls"))
    assert tap.progress()["stop_reason"] == "tool_calls"
    tap.feed(_chunk(choices=False, usage={"prompt_tokens": 10, "completion_tokens": 9, "total_tokens": 19}))
    tap.feed(b"data: [DONE]\n\n")
    assert tap.finish().output_tokens == 9


def test_an_openai_turn_that_said_its_finish_reason_is_finishing_not_a_row():
    from tests.test_request_queue import _stub, _head
    t = _stub(1.0, first_at=2.0)
    t.tap = UsageTap("/v1/chat/completions", 200, "text/event-stream")
    t.tap.feed(_chunk({"content": "x"}))
    assert len(rq.queue_view([t], [], [_head(0)], now=5.0)["inflight"]) == 1
    t.tap.feed(_chunk({}, finish="stop"))
    assert rq.queue_view([t], [], [_head(0)], now=5.0)["inflight"] == []
