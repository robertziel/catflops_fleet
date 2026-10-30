"""THE REQUEST QUEUE: the turns the ENGINE is working on — prefilling, queued or streaming — one row each with the
LB's own measurements off the relayed stream, and the last finished ones; /lb/status `queue`, the REQUESTS card.

The wire these tests speak is CAPTURED, not remembered (2026-10-03, Claude Code 2.1.288 `claude -p` against
a local listener, scratchpad capture/wire.jsonl): `User-Agent: claude-cli/2.1.288 (external, sdk-cli)`,
`X-Claude-Code-Session-Id: <uuid>`, `metadata.user_id` a JSON STRING carrying the same `session_id`, a
`system` of three text blocks the FIRST of which is `x-anthropic-billing-header: …`, 23 tools with full
`input_schema`s, and a conversation whose LAST message is `role: system` (mid-conversation-system beta) —
so "the last user turn" is the last `role: user` message, never `messages[-1]`. Its user turn is a
`<system-reminder>` text block, then the typed text. The stream figures replay the captured cluster stream
(fixtures/throughput/cluster-anthropic-stream.body) at its recorded read times.

Pinned: HELD turns (nothing sent) and finished ones are counts, not rows; the no-byte turns split FCFS — the
oldest N − W of N are PREFILL, the rest QUEUED, W the engine's own waiting gauge — and say "sent" when there is
no gauge; rows ordered streaming, prefill, queued, oldest first in each; prompt tokens and cached share from
`message_start`, PREFILL = send → `message_start`, DECODE = content events since (thinking / text / tool, NOT
tokens) until the engine's count; content BOUNDED (the last user turn ≤ 4,000 chars, a 160-char snippet once
finished, never an earlier message, a tool's schema or description, the system prompt); 64 rows, 30 finished.
"""
from __future__ import annotations

import http.client
import json
import socket
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.fakes import FakeBackend
from vllm_lb.config import Config, VllmBox
from vllm_lb.hub import Hub
from vllm_lb.proxy import Proxy, _Turn
from vllm_lb import request_queue as rq
from vllm_lb.state import LEASE_RUNNING, Lease
from vllm_lb.usage_tap import UsageTap, Usage

UA = "claude-cli/2.1.288 (external, sdk-cli)"
SESSION = "2b32f8ff-4164-4249-b369-c35e5e00eaf8"
BILLING = "x-anthropic-billing-header: cc_version=2.1.288.f22; cc_entrypoint=sdk-cli;"
FIXTURES = Path(__file__).parent / "fixtures" / "throughput"


def sse(name: str, obj: dict) -> bytes:
    return f"event: {name}\ndata: {json.dumps(obj)}\n\n".encode()


def anthropic_stream(n: int = 10, prompt: int = 1234, cached: int = 1000) -> list[bytes]:
    """The cluster head's Anthropic stream (fixtures/throughput/cluster-anthropic-stream.body), with a thinking
    block of `n` deltas, a text block of 2 and a tool_use block of 2 `input_json_delta`s; `cached` of the prompt
    read from the prefix cache (reported apart, as vLLM 0.29 does)."""
    def block(i, kind, deltas):
        start = {"thinking": {"type": "thinking", "thinking": ""}, "text": {"type": "text", "text": ""},
                 "tool_use": {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {}}}[kind]
        delta = {"thinking": lambda k: {"type": "thinking_delta", "thinking": f" t{k}"},
                 "text": lambda k: {"type": "text_delta", "text": f" w{k}"},
                 "tool_use": lambda k: {"type": "input_json_delta", "partial_json": "{\"a\": 1}"[:k + 1]}}[kind]
        return ([sse("content_block_start", {"type": "content_block_start", "index": i, "content_block": start})]
                + [sse("content_block_delta", {"type": "content_block_delta", "index": i, "delta": delta(k)})
                   for k in range(deltas)]
                + [sse("content_block_stop", {"type": "content_block_stop", "index": i})])
    usage = {"input_tokens": prompt - cached, "cache_read_input_tokens": cached}
    return ([sse("message_start", {"type": "message_start", "message": {
                "id": "msg_1", "type": "message", "role": "assistant", "model": "qwen3.8:latest", "content": [],
                "usage": {**usage, "output_tokens": 0}}})]
            + block(0, "thinking", n) + block(1, "text", 2) + block(2, "tool_use", 2)
            + [sse("message_delta", {"type": "message_delta", "delta": {"stop_reason": "tool_use"},
                                     "usage": {**usage, "output_tokens": n + 9}}),
               sse("message_stop", {"type": "message_stop"})])


def claude_body(last_text: str = "say hi", *, results: tuple[str, ...] = (), stream: bool = True) -> dict:
    """A Claude Code request in the captured shape, with a marked secret wherever content must NOT leak."""
    last: list[dict] = [{"type": "tool_result", "tool_use_id": f"t{i}", "content": [{"type": "text", "text": r}]}
                        for i, r in enumerate(results)]
    last += [{"type": "text", "text": "<system-reminder>\nREMINDER-SECRET attribution lines\n</system-reminder>"},
             {"type": "text", "text": last_text}]
    return {
        "model": "qwen3.8:latest", "max_tokens": 32000, "stream": stream,
        "metadata": {"user_id": json.dumps({"device_id": "d" * 64, "account_uuid": "", "session_id": SESSION})},
        "system": [{"type": "text", "text": BILLING},
                   {"type": "text", "text": "SYSTEM-SECRET You are a Claude agent. " * 5},
                   {"type": "text", "text": "SYSTEM-LATER-SECRET " * 50}],
        "tools": [{"name": name, "description": f"TOOLDESC-SECRET {name}",
                   "input_schema": {"type": "object", "properties": {"SCHEMA-SECRET": {"type": "string"}}}}
                  for name in ("Agent", "Bash", "Read", "Write")],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "EARLIER-SECRET first prompt"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "EARLIER-SECRET answer"},
                                              {"type": "tool_use", "id": "t0", "name": "Bash",
                                               "input": {"command": "EARLIER-SECRET ls"}}]},
            {"role": "user", "content": last},
            {"role": "system", "content": [{"type": "text", "text": "TRAILING-SYSTEM-SECRET # Environment"}]},
        ],
    }


SECRETS = ("EARLIER-SECRET", "SCHEMA-SECRET", "TOOLDESC-SECRET", "SYSTEM-SECRET", "SYSTEM-LATER-SECRET",
           "TRAILING-SYSTEM-SECRET", "REMINDER-SECRET", "input_schema", "\"Bash\"")


# ── the body: bounded content, nothing else ─────────────────────────────────────────────────────────

def test_the_facts_are_the_last_user_turn_and_counts_and_nothing_else():
    facts = rq.request_facts(claude_body("say hi", results=("R" * 300, "second result")))
    assert facts["model"] == "qwen3.8:latest"
    assert facts["messages"] == 4 and facts["tools"] == 4
    # tool_result blocks summarised with the first 120 chars of their text; the reminder is counted, not quoted.
    assert facts["prompt"].startswith("[tool_result ×2] " + "R" * 120)
    assert "R" * 121 not in facts["prompt"]
    assert "say hi" in facts["prompt"] and "[system-reminder ×1]" in facts["prompt"]
    assert set(facts) == {"model", "messages", "tools", "prompt", "prompt_chars"}
    text = json.dumps(facts)
    for secret in SECRETS:
        assert secret not in text, f"{secret} leaked into the view"


def test_the_prompt_is_capped_at_4000_chars_and_says_how_long_it_was():
    facts = rq.request_facts(claude_body("x" * 10_000))
    assert facts["prompt_chars"] > 10_000
    assert len(facts["prompt"]) <= rq.PROMPT_CAP + 32
    assert facts["prompt"].endswith(f"… ({facts['prompt_chars']:,} chars)")
    short = rq.request_facts(claude_body("hello"))
    assert "…" not in short["prompt"] and short["prompt_chars"] == len(short["prompt"])


def test_an_openai_chat_body_and_a_plain_string_content_read_the_same_way():
    facts = rq.request_facts({"model": "qwen3.8", "messages": [
        {"role": "system", "content": "SYSTEM-SECRET " * 40},
        {"role": "user", "content": "EARLIER-SECRET"}, {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "what is <script>alert(1)</script>?"}],
        "tools": [{"type": "function", "function": {"name": "search", "parameters": {"SCHEMA-SECRET": 1}}}]})
    assert facts["prompt"] == "what is <script>alert(1)</script>?"   # raw: escaping is the page's job
    assert facts["tools"] == 1 and facts["messages"] == 4
    for secret in ("EARLIER-SECRET", "SCHEMA-SECRET", "SYSTEM-SECRET"):
        assert secret not in json.dumps(facts)


def test_no_body_or_a_body_of_another_shape_is_no_facts_not_an_error():
    assert rq.request_facts(None)["prompt"] is None
    odd = rq.request_facts({"messages": "nope", "tools": 3, "system": 7, "model": ["x"]})
    assert odd["prompt"] is None and odd["messages"] is None and odd["tools"] is None and odd["model"] is None


def test_the_client_is_named_by_its_user_agent_and_nothing_else():
    assert rq.client_of(UA) == ("claude", "sdk-cli")
    assert rq.client_of("claude-cli/2.1.288 (external, cli)") == ("claude", "cli")
    assert rq.client_of("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36")[0] == "browser"
    assert rq.client_of(None) == ("no ua", None)
    assert rq.client_of("python-requests/2.32.3") == ("python-requests", None)
    assert len(rq.client_of("x" * 500)[0]) <= 24


def test_the_session_is_the_header_else_metadata_user_id_and_only_its_tail_is_shown():
    assert rq.session_of({"X-Claude-Code-Session-Id": SESSION}, None) == SESSION[-8:]
    assert rq.session_of({}, claude_body()) == SESSION[-8:]
    legacy = {"metadata": {"user_id": "user_abc_account__session_11111111-2222-3333-4444-555566667777"}}
    assert rq.session_of({}, legacy) == "66667777"
    assert rq.session_of({}, {"metadata": {"user_id": "not json"}}) is None
    assert rq.session_of({}, None) is None


# ── the tap: what a stream has said so far ──────────────────────────────────────────────────────────

class Clock:
    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def test_the_tap_reports_progress_mid_stream_without_changing_its_verdict():
    chunks = anthropic_stream(n=5, prompt=72, cached=40)
    clock = Clock()
    tap = UsageTap("/v1/messages", 200, "text/event-stream", clock=clock)
    p0 = tap.progress()
    assert (p0["deltas"], p0["prompt_tokens"], p0["start_at"], p0["output_tokens"]) == (0, None, None, None)
    clock.t = 3.5
    for c in chunks[:5]:                                  # message_start, thinking start, three thinking deltas
        tap.feed(c)
    p = tap.progress()
    assert (p["deltas"], p["thinking"], p["text"], p["tool"], p["tool_uses"]) == (3, 3, 0, 0, 0)
    assert (p["prompt_tokens"], p["cached_tokens"], p["prompt_from"], p["start_at"]) == (72, 40, "message_start", 3.5)
    assert p["output_tokens"] is None and p["stop_reason"] is None
    clock.t = 9.0
    for c in chunks[5:]:
        tap.feed(c)
    p = tap.progress()
    assert (p["deltas"], p["thinking"], p["text"], p["tool"], p["tool_uses"]) == (9, 5, 2, 2, 1)
    assert (p["prompt_tokens"], p["cached_tokens"], p["prompt_from"]) == (72, 40, "message_delta")
    assert (p["output_tokens"], p["stop_reason"], p["start_at"]) == (14, "tool_use", 3.5)
    assert isinstance(tap.verdict, Usage) and tap.verdict.output_tokens == 14


def test_a_malformed_message_start_costs_the_progress_figure_not_the_count():
    bad = [b"event: message_start\ndata: {not json\n\n"] + anthropic_stream(n=3)[1:]
    tap = UsageTap("/v1/messages", 200, "text/event-stream")
    for c in bad:
        tap.feed(c)
    got = tap.finish()
    assert isinstance(got, Usage) and got.output_tokens == 12
    assert tap.progress()["prompt_tokens"] == 1234 and tap.progress()["prompt_from"] == "message_delta"


# ── the figures, off the captured stream at its recorded pace ─────────────────────────────────────────

class _H:
    """A handler as `_Turn` reads one."""
    command, path, client_address = "POST", "/v1/messages", ("127.0.0.1", 1)

    def __init__(self, ua=UA):
        self.headers = {"User-Agent": ua}


def test_prefill_and_decode_figures_from_the_captured_cluster_stream():
    """Sent at t = 0 (capture.py's t0 is the send); `message_start` rode the first read. PREFILL = send →
    `message_start`, its tok/s = the prompt it reported ÷ that; DECODE = content events since, which are NOT the
    engine's 32 tokens (9 events carried them); once over, the engine's count and the tap's own decode rate."""
    body = (FIXTURES / "cluster-anthropic-stream.body").read_bytes()
    meta = json.loads((FIXTURES / "cluster-anthropic-stream.json").read_text())
    clock = Clock()
    tap = UsageTap("/v1/messages", 200, "text/event-stream", clock=clock)
    turn = _Turn(_H(), body=meta["request"]["body"])
    turn.backend, turn.box, turn.sent_at, turn.tap, turn.anthropic_sse = "http://head", "box-a + box-b", 0.0, tap, True
    offset, start_read = 0, None
    for i, (at, size) in enumerate(meta["reads"][:4]):
        clock.t = at
        tap.feed(body[offset:offset + size])
        offset += size
        if start_read is None and b"event: message_start" in body[:offset]:
            start_read = at
        turn.first_at = turn.first_at or at
        turn.last_at = at
    now = clock.t + 1.0
    row = rq.live_row(turn, now, rq.STREAMING)
    seen = body[:offset].count(b"event: content_block_delta")
    assert row["prompt_tokens"] == 72 and row["cached_share"] == 0.0
    assert row["prefill_s"] == pytest.approx(start_read, abs=1e-3)
    assert row["prefill_tok_s"] == pytest.approx(72 / start_read, rel=1e-3)
    assert (row["deltas"], row["thinking"], row["text"]) == (seen, seen, 0)
    assert row["decode_s"] == pytest.approx(now - start_read, abs=0.05)
    assert row["delta_rate"] == pytest.approx(seen / (now - start_read), abs=0.05)
    assert row["output_tokens"] is None, "no count until the engine says one"
    for at, size in meta["reads"][4:]:
        clock.t = at
        tap.feed(body[offset:offset + size])
        offset += size
    usage = tap.finish()
    turn.status, turn.ended, turn.sent = 200, True, offset
    done = rq.finished_row(turn, clock.t, 1.8e9)
    assert (done["outcome"], done["output_tokens"], done["stop_reason"]) == (rq.DONE, 32, "max_tokens")
    assert done["deltas"] == 9 and done["thinking"] == 9
    assert done["decode_tok_s"] == pytest.approx(usage.decode_tok_s, abs=0.05)
    assert done["prefill_tok_s"] == pytest.approx(72 / start_read, rel=1e-3)


# ── queued or prefilling: the engine's FCFS, inferred ─────────────────────────────────────────────────

def _stub(started: float, *, backend: str | None = "http://head", sent_at: float | None = None,
          first_at: float | None = None, holding: bool = False, prompt: str = "p"):
    t = _Turn(_H(), body=claude_body(prompt))
    t.started, t.holding, t.backend = started, holding, backend
    t.sent_at = sent_at if sent_at is not None else (None if backend is None else started)
    t.first_at = first_at
    return t


def test_a_finished_row_names_the_backend_it_went_to():
    """The panel scopes its Requests card to the SELECTED server (worklog/parallel-clusters.md, "Servers, not clusters"):
    a live row always carried `backend`; a finished one did not, so the recent fold could not be scoped."""
    t = _stub(0.0, first_at=0.5)
    t.status, t.ended = 200, True
    assert rq.finished_row(t, 2.0, 1.8e9)["backend"] == "http://head"


def _head(waiting: int | None, running: int = 0):
    return SimpleNamespace(url="http://head", live=True, depth_known=waiting is not None,
                           waiting=waiting or 0, running=running)


def test_of_n_no_byte_turns_the_oldest_n_minus_waiting_are_prefill_and_the_rest_queued():
    """SGLang serves FCFS and chunk-prefills one request at a time; the request being prefilled is in NEITHER gauge
    (worklog/sglang-bottleneck-2026-10-03.md: 30 running + 6 waiting + 1 = 37 LB in flight)."""
    turns = [_stub(float(i), prompt=f"n{i}") for i in range(5)]                 # n0 oldest
    streaming = [_stub(10.0 + i, first_at=20.0 + i, prompt=f"s{i}") for i in range(2)]
    held = [_stub(30.0, backend=None, holding=True, prompt="h")]
    q = rq.queue_view(turns + streaming + held, [], [_head(waiting=3)], now=40.0)
    order = [(r["prompt"].split()[0], r["state"]) for r in q["inflight"]]
    assert order == [("s0", "streaming"), ("s1", "streaming"), ("n0", "prefill"), ("n1", "prefill"),
                     ("n2", "queued"), ("n3", "queued"), ("n4", "queued")]
    assert q["counts"] == {"streaming": 2, "prefill": 2, "queued": 3, "sent": 0, "held": 1, "recent": 0}
    assert q["inflight"][2]["inferred"] == {"no_bytes": 5, "waiting": 3}
    assert all(r["state"] != rq.HELD for r in q["inflight"]), "a held turn is a count, never a row"

    all_queued = rq.queue_view(turns, [], [_head(waiting=9)], now=40.0)
    assert {r["state"] for r in all_queued["inflight"]} == {"queued"}
    unknown = rq.queue_view(turns, [], [_head(waiting=None)], now=40.0)
    assert {r["state"] for r in unknown["inflight"]} == {"sent"}, "no gauge: no bytes yet, split unknown"
    assert unknown["inflight"][0]["inferred"] is None


def test_a_turn_whose_stream_said_stop_is_finishing_not_a_row():
    t = _stub(1.0, first_at=2.0)
    t.tap = UsageTap("/v1/messages", 200, "text/event-stream")
    for c in anthropic_stream(n=2)[:-1]:                 # up to message_delta: its stop reason is in
        t.tap.feed(c)
    assert rq.queue_view([t], [], [_head(0)], now=5.0)["inflight"] == []


# ── the proxy: the turns as the LB sees them ────────────────────────────────────────────────────────

@pytest.fixture
def stack():
    created = []

    def make(v: FakeBackend, *, live: bool = True, depth_known: bool = True, waiting: int = 0, **over):
        cfg = Config(vllm_boxes=(VllmBox("box-a-gpu", "box-a", v.url),), 
                     listen_host="127.0.0.1", listen_port=0, connect_timeout_s=2.0,
                     first_byte_timeout_s=30.0, admission_wait_s=0, **over)
        hub = Hub(cfg)
        b = hub.backend_for(v.url)
        b.live, b.served_models, b.depth_known, b.waiting = live, tuple(v.served), depth_known, waiting
        hub.set_leases([Lease(lease_id="r", job_id="j", created_at=0, status=LEASE_RUNNING,
                              host_label="box-a-gpu", box="box-a")])
        proxy = Proxy(cfg, hub)
        proxy.start()
        created.append(proxy)
        return proxy, hub

    yield make
    for p in created:
        p.stop()


def post(port: int, body: dict, out: dict | None = None, key: str = "r", *, headers: dict | None = None,
         timeout: float = 30) -> None:
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
        conn.request("POST", "/v1/messages?beta=true", body=json.dumps(body).encode(),
                     headers={"Content-Type": "application/json", "User-Agent": UA,
                              "X-Claude-Code-Session-Id": SESSION, **(headers or {})})
        r = conn.getresponse()
        data = r.read()
        if out is not None:
            out[key] = {"status": r.status, "body": data}
    except Exception as exc:                                     # noqa: BLE001 — the test reads it
        if out is not None:
            out[key] = {"error": f"{type(exc).__name__}: {exc}"}


def in_thread(*args, **kw) -> threading.Thread:
    t = threading.Thread(target=post, args=args, kwargs=kw, daemon=True)
    t.start()
    return t


def wait(cond, timeout: float = 10.0):
    deadline = time.monotonic() + timeout
    while True:
        got = cond()
        if got:
            return got
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.02)


def finished(proxy: Proxy, n: int = 1) -> list[dict]:
    """The finished rows once `n` are there: a client has its last byte a moment before `handle` unbooks the turn."""
    return wait(lambda: (x := proxy.queue_view()["recent"]) and len(x) >= n and x)


def row(proxy: Proxy, state: str | None = None):
    rows = proxy.queue_view()["inflight"]
    return next((r for r in rows if state is None or r["state"] == state), None)


def test_a_streaming_turn_is_one_row_with_its_identity_figures_and_bounded_content(stack):
    v = FakeBackend()
    v.stream_chunks, v.stream_delay_s = anthropic_stream(n=12), 0.1
    proxy, _ = stack(v)
    out: dict = {}
    in_thread(proxy.port, claude_body("summarise <script>alert(1)</script> " + "y" * 5000), out)
    r = wait(lambda: (x := row(proxy, rq.STREAMING)) and x["deltas"] >= 3 and x)
    assert r["client"] == "claude" and r["client_detail"] == "sdk-cli"
    assert r["session"] == SESSION[-8:]
    assert r["model"] == "qwen3.8:latest" and r["messages"] == 4 and r["tools"] == 4
    assert r["box"] == "box-a" and r["backend"] == v.url
    assert r["prompt"].startswith("summarise <script>") and len(r["prompt"]) <= rq.PROMPT_CAP + 32
    assert r["prompt_chars"] > rq.PROMPT_CAP
    assert r["prompt_tokens"] == 1234 and r["cached_share"] == pytest.approx(1000 / 1234, abs=1e-3)
    assert r["prefill_s"] is not None and 0 <= r["prefill_s"] < 5 and r["prefill_tok_s"] > 0
    assert r["thinking"] >= 2 and r["text"] == 0 and r["tool_uses"] == 0
    assert r["output_tokens"] is None, "no count until the engine says one: deltas are not tokens"
    assert r["bytes"] > 0 and r["last_byte_s"] is not None and r["last_byte_s"] < 5
    assert r["age_s"] >= r["phase_s"] >= 0 and isinstance(r["started_at"], float)
    first_id = r["id"]
    assert row(proxy, rq.STREAMING)["id"] == first_id, "an id is stable for the life of its turn"
    text = json.dumps(proxy.status()["queue"])
    for secret in SECRETS:
        assert secret not in text, f"{secret} leaked into /lb/status"

    wait(lambda: "r" in out)
    done = finished(proxy)[0]
    assert proxy.queue_view()["inflight"] == [], "a finished turn is not a row"
    assert done["id"] == first_id and done["outcome"] == rq.DONE and done["status"] == 200
    assert done["output_tokens"] == 21 and done["prompt_tokens"] == 1234 and done["stop_reason"] == "tool_use"
    assert (done["thinking"], done["text"], done["tool"], done["tool_uses"]) == (12, 2, 2, 1)
    assert done["duration_s"] > 1.0 and done["decode_tok_s"] is not None and done["decode_tok_s"] > 0
    assert done["prefill_tok_s"] is not None
    assert len(done["prompt"]) <= rq.SNIPPET_CHARS + 1 and done["prompt"].startswith("summarise")


@pytest.mark.parametrize("depth_known,waiting,state", [(False, 0, "sent"), (True, 0, "prefill"), (True, 1, "queued")])
def test_a_turn_sent_with_no_byte_back_yet_is_prefill_queued_or_sent(stack, depth_known, waiting, state):
    v = FakeBackend()
    v.stream_chunks, v.first_byte_delay_s = anthropic_stream(n=2), 1.2
    proxy, _ = stack(v, depth_known=depth_known, waiting=waiting)
    in_thread(proxy.port, claude_body())
    wait(lambda: row(proxy, state))
    time.sleep(0.4)
    r = row(proxy, state)
    assert r is not None and r["bytes"] == 0 and r["last_byte_s"] is None and r["deltas"] == 0
    assert r["phase_s"] >= 0.3 and r["prefill_s"] is None
    assert r["prompt_tokens"] is None, "before message_start the engine has said nothing"


def test_a_turn_held_by_the_drain_is_a_count_and_its_503_a_failure_that_says_draining(stack):
    v = FakeBackend()
    v.stream_chunks, v.stream_delay_s = anthropic_stream(n=8), 0.15
    proxy, _ = stack(v, drain_retry_after_s=5)
    out: dict = {}
    in_thread(proxy.port, claude_body(), out, "first")
    wait(lambda: row(proxy, rq.STREAMING))
    threading.Thread(target=proxy.drain, args=(20.0,), daemon=True).start()
    time.sleep(0.2)
    in_thread(proxy.port, claude_body("late"), out, "late")
    q = wait(lambda: (x := proxy.queue_view()) and x["counts"]["held"] == 1 and x)
    assert [r["state"] for r in q["inflight"]] == ["streaming"], "a held turn is never a row"
    wait(lambda: "late" in out and "first" in out)
    recent = {r["prompt"].split()[0]: r for r in finished(proxy, 2)}
    assert recent["late"]["outcome"] == rq.FAILED and recent["late"]["why"] == "draining"
    assert recent["late"]["status"] == 503


def test_an_error_status_a_refusal_and_a_cut_stream_are_failures_with_their_why(stack):
    v = FakeBackend()
    v.reply_status = 500
    proxy, _ = stack(v)
    post(proxy.port, claude_body(stream=False))
    got = finished(proxy)[0]
    assert (got["outcome"], got["why"], got["status"]) == (rq.FAILED, "HTTP 500", 500)

    dead, _ = stack(FakeBackend(), live=False)
    post(dead.port, claude_body())
    got = finished(dead)[0]
    assert (got["outcome"], got["why"], got["status"]) == (rq.FAILED, "no_backend", 503)

    cut = FakeBackend()
    cut.stream_chunks, cut.drop_after_chunks = anthropic_stream(n=6), 4
    p3, _ = stack(cut)
    post(p3.port, claude_body())
    got = finished(p3)[0]
    assert got["outcome"] == rq.FAILED and got["why"] in ("backend failed mid-stream", "no message_stop")


def test_a_client_that_hangs_up_mid_stream_is_client_gone(stack):
    v = FakeBackend()
    v.stream_chunks, v.stream_delay_s = anthropic_stream(n=40), 0.05
    proxy, _ = stack(v)
    body = json.dumps(claude_body()).encode()
    s = socket.create_connection(("127.0.0.1", proxy.port))
    s.sendall(f"POST /v1/messages HTTP/1.1\r\nHost: 127.0.0.1:{proxy.port}\r\n".encode() + b"Content-Type: application/json\r\n"
              + f"User-Agent: {UA}\r\nContent-Length: {len(body)}\r\n\r\n".encode() + body)
    s.recv(256)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, b"\x01\x00\x00\x00\x00\x00\x00\x00")
    s.close()
    assert finished(proxy)[0]["outcome"] == rq.CLIENT_GONE


def test_a_finished_json_answer_is_done_with_the_engines_count(stack):
    v = FakeBackend()
    v.reply_json = {"id": "m", "type": "message", "content": [{"type": "text", "text": "OK"}],
                    "usage": {"input_tokens": 50, "output_tokens": 7}}
    proxy, _ = stack(v)
    post(proxy.port, claude_body(stream=False))
    got = finished(proxy)[0]
    assert (got["outcome"], got["status"], got["output_tokens"], got["prompt_tokens"]) == (rq.DONE, 200, 7, 50)


def test_recent_keeps_the_last_30_newest_first(stack):
    v = FakeBackend()
    proxy, _ = stack(v)
    for i in range(rq.RECENT_ROWS + 5):
        post(proxy.port, claude_body(f"n{i}", stream=False))
        finished(proxy, min(i + 1, rq.RECENT_ROWS))
    recent = wait(lambda: (x := finished(proxy, rq.RECENT_ROWS))
                  and x[0]["prompt"].startswith(f"n{rq.RECENT_ROWS + 4}") and x)
    assert len(recent) == rq.RECENT_ROWS and recent[-1]["prompt"].startswith("n5")
    assert proxy.queue_view()["counts"]["recent"] == rq.RECENT_ROWS


def test_the_rows_are_capped_at_64_and_say_how_many_more(stack):
    v = FakeBackend()
    proxy, hub = stack(v)
    turns = [_stub(float(i), first_at=float(i) + 0.5, backend=v.url, prompt=f"t{i}") for i in range(70)]
    with proxy._turns_cv:
        proxy._turns.update(turns)
    try:
        q = proxy.queue_view()
        assert len(q["inflight"]) == rq.INFLIGHT_ROWS and q["inflight_total"] == 70
        assert q["inflight"][0]["prompt"].startswith("t0"), "oldest first within a group"
        assert len({r["id"] for r in q["inflight"]}) == rq.INFLIGHT_ROWS
        assert len(json.dumps(q)) < rq.INFLIGHT_ROWS * (rq.PROMPT_CAP + 2048) + rq.RECENT_ROWS * 1024
    finally:
        with proxy._turns_cv:
            proxy._turns.difference_update(turns)


def test_the_engines_own_aggregates_stand_beside_the_lbs_turns(stack):
    v = FakeBackend()
    proxy, hub = stack(v)
    b = hub.backend_for(v.url)
    b.running, b.waiting, b.depth_known = 30, 6, True
    q = proxy.status()["queue"]
    assert q["engine"]["running"] == 30 and q["engine"]["waiting"] == 6
    assert q["inflight"] == [] and q["counts"] == {"streaming": 0, "prefill": 0, "queued": 0, "sent": 0,
                                                   "held": 0, "recent": 0}
    b.live = False
    assert proxy.queue_view()["engine"] is None, "no live engine reports a depth: none is shown"


def test_a_web_search_answered_by_the_lb_itself_is_not_a_row():
    """`answer_web_search`: SearXNG, no engine — nothing the GPU is doing."""
    search = _stub(1.0, backend=rq.WEBSEARCH)
    q = rq.queue_view([search], [], [_head(0)], now=3.0)
    assert q["inflight"] == [] and sum(v for k, v in q["counts"].items() if k != "recent") == 0
