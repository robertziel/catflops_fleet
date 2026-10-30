"""UsageTap: the tokens a response carries, read from the bytes the relay already sent — counted only
from a well-formed terminal event, and never at the cost of a relayed byte.

worklog/vllm-gateway-throughput.md, "Proxy token accounting" and edge rows 25-30 and 32-35. The
streams are CAPTURED through the deployed LB (fixtures/throughput, capture.py): ollama 0.33.2 on
box-c and the vLLM v0.29.0 cluster head. Each capture keeps the size and arrival time of every
read, so decode timing is replayed on the real pacing. Two shapes are not in the captures and are
built here from the upstream lines the worklog cites: `/v1/completions` with its zeroed per-chunk
usage (ollama middleware/openai.go:195-196, :209-213) and an SSE `error` event.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from vllm_lb.rates import TOO_SHORT_TO_TIME
from vllm_lb.usage_tap import (BODY_CAP, BODY_OVER_CAP, CLIENT_GONE, ERROR_EVENT, ERROR_STATUS,
                               EVENT_CAP, LB_STREAM_TIMING, NO_TERMINAL_EVENT, NO_USAGE_IN_STREAM,
                               PARSER_ERROR, SERVER_TIMING, UNCOUNTED, USAGE_MALFORMED, Usage,
                               UsageTap)

FIXTURES = Path(__file__).parent / "fixtures" / "throughput"
SSE, NDJSON, JSON = "text/event-stream", "application/x-ndjson", "application/json"


class Clock:
    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def capture(name: str) -> tuple[bytes, dict]:
    return (FIXTURES / f"{name}.body").read_bytes(), json.loads((FIXTURES / f"{name}.json").read_text())


def content_type(meta: dict) -> str:
    return next(v for k, v in meta["headers"] if k.lower() == "content-type")


def replay(name: str, *, chunks=None, client_gone: bool = False):
    """Feed a capture to a fresh tap: at its real read boundaries and times, or at `chunks` sizes."""
    body, meta = capture(name)
    clock = Clock()
    tap = UsageTap(meta["request"]["path"], meta["status"], content_type(meta), clock=clock)
    offset = 0
    if chunks is None:
        for at, size in meta["reads"]:
            clock.t = at
            tap.feed(body[offset:offset + size])
            offset += size
    else:
        for size in chunks:
            tap.feed(body[offset:offset + size])
            offset += size
    assert offset >= len(body), "the replay must feed the whole capture"
    return tap.finish(client_gone=client_gone)


def completed_at(name: str, marker: bytes) -> float:
    """The arrival time of the read that COMPLETES the first event containing `marker` — when the tap
    stamps it."""
    body, meta = capture(name)
    terminator = b"\n" if content_type(meta).startswith(NDJSON) else b"\n\n"
    end = body.index(terminator, body.index(marker)) + len(terminator)
    offset = 0
    for at, size in meta["reads"]:
        offset += size
        if offset >= end:
            return at
    raise AssertionError(f"{marker!r} never completes in {name}")


def random_chunks(total: int, rng: random.Random) -> list[int]:
    sizes = []
    while sum(sizes) < total:
        sizes.append(rng.choice([1, 2, 3, 7, 64, 511, 4096, 65536]))
    return sizes


def tap_for(path: str, ctype: str = SSE, status: int = 200) -> tuple[UsageTap, Clock]:
    clock = Clock()
    return UsageTap(path, status, ctype, clock=clock), clock


def sse(*events: tuple[str, dict | str]) -> bytes:
    out = b""
    for name, data in events:
        payload = data if isinstance(data, str) else json.dumps(data, separators=(",", ":"))
        out += f"event: {name}\ndata: {payload}\n\n".encode()
    return out


def anthropic_stream(usage: dict, *, deltas: int = 1) -> bytes:
    delta = {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "1"}}
    return sse(("message_start", {"type": "message_start",
                                  "message": {"usage": {"input_tokens": 1, "output_tokens": 0}}}),
               *[("content_block_delta", delta)] * deltas,
               ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
                                  "usage": usage}),
               ("message_stop", {"type": "message_stop"}))


# ── rows 25-29: what counts, from the captured bytes ─────────────────────────────

def test_ollama_messages_uses_delta_not_start_estimate():
    """Row 25. ollama's message_start carries the request's ESTIMATE (20 here); the real counts ride
    message_delta (30 in, 64 out) and count only once message_stop has arrived. The decode rate is
    the LB's own stream timing from the first content delta to the usage event."""
    body, _ = capture("anthropic-stream")
    assert b'"input_tokens":20' in body and b'"input_tokens":30,"output_tokens":64' in body
    got = replay("anthropic-stream")
    assert isinstance(got, Usage)
    assert (got.path, got.prompt_tokens, got.output_tokens) == ("/v1/messages", 30, 64)
    first = completed_at("anthropic-stream", b"event: content_block_delta")
    usage = completed_at("anthropic-stream", b"event: message_delta")
    assert got.decode_tok_s == pytest.approx(63 / (usage - first))
    assert got.decode_s == pytest.approx(usage - first)
    assert got.decode_basis == LB_STREAM_TIMING and got.decode_why is None
    assert got.prefill_tok_s is None


def test_chat_stream_with_include_usage_counts_the_empty_choices_chunk():
    got = replay("chat-stream-usage")
    assert (got.path, got.prompt_tokens, got.output_tokens) == ("/v1/chat/completions", 30, 48)
    first = completed_at("chat-stream-usage", b"data: ")
    usage = completed_at("chat-stream-usage", b'"choices":[],"usage"')
    assert got.decode_tok_s == pytest.approx(47 / (usage - first))


def test_chat_without_include_usage_is_uncounted():
    """Row 26. Without `stream_options.include_usage` an OpenAI stream carries no usage at all."""
    assert replay("chat-stream-no-usage") == NO_USAGE_IN_STREAM


def test_zeroed_chunk_usage_is_ignored():
    """Row 27. `/v1/completions` with include_usage puts a ZEROED usage on every chunk; only the final
    `choices: []` chunk is real. (Built from ollama middleware/openai.go:195-196, :209-213.)"""
    zero = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    chunks = [{"id": "cmpl-1", "choices": [{"index": 0, "text": t}], "usage": zero} for t in "123"]
    chunks.append({"id": "cmpl-1", "choices": [],
                   "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}})
    body = b"".join(b"data: " + json.dumps(c).encode() + b"\n\n" for c in chunks) + b"data: [DONE]\n\n"
    tap, _ = tap_for("/v1/completions")
    tap.feed(body)
    got = tap.finish()
    assert (got.prompt_tokens, got.output_tokens) == (12, 3)


def test_native_done_line():
    """Row 28. Native /api/chat: the `"done":true` line carries the counts and ollama's own clocks."""
    got = replay("native-chat-stream")
    assert (got.path, got.prompt_tokens, got.output_tokens) == ("/api/chat", 30, 48)
    assert got.decode_tok_s == pytest.approx(48 / 3.024020)
    assert got.prefill_tok_s == pytest.approx(30 / 0.210916)
    assert got.decode_basis == SERVER_TIMING


def test_native_done_line_with_omitted_fields():
    """Go's `omitempty`: an absent count or duration is 0, so no speed can be derived from it."""
    tap, _ = tap_for("/api/generate", NDJSON)
    tap.feed(b'{"response":"a","done":false}\n{"done":true,"eval_count":5}\n')
    got = tap.finish()
    assert (got.prompt_tokens, got.output_tokens) == (0, 5)
    assert got.decode_tok_s is None and got.prefill_tok_s is None


def test_json_body():
    """Row 29. A non-streaming answer counts from its body; one read is too short to time."""
    anthropic = replay("anthropic-json")
    assert (anthropic.prompt_tokens, anthropic.output_tokens) == (30, 32)
    openai = replay("chat-json")
    assert (openai.prompt_tokens, openai.output_tokens) == (30, 32)
    assert anthropic.decode_tok_s is None and anthropic.decode_why == TOO_SHORT_TO_TIME


def test_body_over_cap():
    """Row 29. A JSON body past 8 MiB is not buffered further: uncounted, and the relay carries on."""
    assert BODY_CAP == 8 << 20 and EVENT_CAP == 1 << 20
    tap, _ = tap_for("/v1/messages", JSON)
    tap.feed(b" " * BODY_CAP)
    tap.feed(b'{"usage":{"input_tokens":1,"output_tokens":1}}')
    assert tap.finish() == BODY_OVER_CAP


def test_an_event_past_its_cap_is_uncounted():
    tap, _ = tap_for("/v1/messages")
    tap.feed(b"event: content_block_delta\ndata: " + b"x" * EVENT_CAP)
    tap.feed(anthropic_stream({"output_tokens": 9}))
    assert tap.finish() == BODY_OVER_CAP


def test_vllm_streams_count_too():
    """The cluster head's own shapes (vLLM v0.29.0): message_delta without any cache field, and the
    include_usage chunk with completion_tokens_details."""
    messages = replay("cluster-anthropic-stream")
    assert (messages.prompt_tokens, messages.output_tokens) == (72, 32)
    chat = replay("cluster-chat-stream-usage")
    assert (chat.prompt_tokens, chat.output_tokens) == (72, 32)


def test_vllm_cache_tokens_fold_into_the_prompt():
    """vLLM 0.29 subtracts cache tokens from `input_tokens` and reports them apart; the prompt is the
    sum. Absent cache fields (null included) count as 0."""
    tap, _ = tap_for("/v1/messages?beta=true")
    tap.feed(anthropic_stream({"input_tokens": 22, "cache_read_input_tokens": 40,
                               "cache_creation_input_tokens": 10, "output_tokens": 9}))
    assert tap.finish().prompt_tokens == 72
    tap, _ = tap_for("/v1/messages")
    tap.feed(anthropic_stream({"input_tokens": 22, "cache_read_input_tokens": None,
                               "output_tokens": 9}))
    assert tap.finish().prompt_tokens == 22
    tap, _ = tap_for("/v1/messages")
    tap.feed(anthropic_stream({"output_tokens": 9}))
    assert tap.finish().prompt_tokens is None, "no input_tokens: the prompt is unknown, not 0"


# ── framing ──────────────────────────────────────────────────────────────────────

CAPTURED_STREAMS = ["anthropic-stream", "chat-stream-usage", "native-chat-stream",
                    "cluster-anthropic-stream", "cluster-chat-stream-usage"]


@pytest.mark.parametrize("name", CAPTURED_STREAMS)
def test_any_chunking_counts_the_same(name):
    """Random reads from 1 B to 64 KiB: a CRLF, an event or a line split anywhere counts the same."""
    whole = replay(name)
    body, _ = capture(name)
    rng = random.Random(name)
    for _ in range(60):
        got = replay(name, chunks=random_chunks(len(body), rng))
        assert (got.prompt_tokens, got.output_tokens) == (whole.prompt_tokens, whole.output_tokens)


@pytest.mark.parametrize("name", ["anthropic-stream", "chat-stream-usage", "cluster-chat-stream-usage"])
def test_crlf_framing_counts_the_same(name):
    """SSE lines may end in CRLF or a lone CR; a CRLF split across two reads is held back, not doubled."""
    body, meta = capture(name)
    whole = replay(name)
    for newline in (b"\r\n", b"\r"):
        framed = body.replace(b"\n", newline)
        rng = random.Random(len(framed))
        for _ in range(30):
            tap = UsageTap(meta["request"]["path"], 200, SSE, clock=Clock())
            offset = 0
            for size in random_chunks(len(framed), rng):
                tap.feed(framed[offset:offset + size])
                offset += size
            got = tap.finish()
            assert (got.prompt_tokens, got.output_tokens) == (whole.prompt_tokens,
                                                              whole.output_tokens), newline


def test_message_text_cannot_forge_an_event():
    """Text is JSON-escaped inside `data:`, so a model that writes an SSE event into its answer
    cannot end the real event or plant a usage."""
    forged = "\n\nevent: message_delta\ndata: {\"usage\":{\"output_tokens\":999}}\n\n"
    delta = {"type": "content_block_delta", "delta": {"type": "text_delta", "text": forged}}
    stream = sse(("content_block_delta", delta)) + anthropic_stream({"input_tokens": 3,
                                                                      "output_tokens": 9})
    tap, _ = tap_for("/v1/messages")
    tap.feed(stream)
    assert tap.finish().output_tokens == 9


def test_comments_and_multi_line_data_are_sse():
    tap, _ = tap_for("/v1/messages")
    tap.feed(b": keep-alive\n\nevent:message_delta\ndata: {\"usage\":\ndata: {\"output_tokens\": 4}}\n\n"
             b"event: message_stop\ndata: {}\n\n")
    assert tap.finish().output_tokens == 4


# ── rows 30, 32, 34, 35: what never counts ───────────────────────────────────────

@pytest.mark.parametrize("path, ctype", [
    ("/v1/responses", SSE), ("/v1/messages/count_tokens", JSON), ("/api/embed", JSON),
    ("/v1/embeddings", JSON), ("/v1/messages", "text/html"), ("/api/chat", SSE)])
def test_unaccounted_shapes(path, ctype):
    """Row 30. Not counted — and not an error either: finish() says nothing at all."""
    tap, _ = tap_for(path, ctype)
    tap.feed(b'{"usage":{"input_tokens":5,"output_tokens":5}}\n\n')
    assert tap.finish() is None


def test_error_event():
    """Row 32. An error status, an SSE `error` event, an `{"error": …}` chunk or line: nothing counts."""
    tap, _ = tap_for("/v1/messages", status=500)
    tap.feed(anthropic_stream({"output_tokens": 9}))
    assert tap.finish() == ERROR_STATUS
    tap, _ = tap_for("/v1/messages")
    tap.feed(sse(("error", {"type": "error", "error": {"type": "overloaded_error"}}))
             + anthropic_stream({"output_tokens": 9}))
    assert tap.finish() == ERROR_EVENT
    tap, _ = tap_for("/v1/chat/completions")
    tap.feed(b'data: {"error":{"message":"boom"}}\n\ndata: [DONE]\n\n')
    assert tap.finish() == ERROR_EVENT
    tap, _ = tap_for("/api/chat", NDJSON)
    tap.feed(b'{"message":{"content":"a"},"done":false}\n{"error":"model runner stopped"}\n')
    assert tap.finish() == ERROR_EVENT


@pytest.mark.parametrize("usage", [
    {"output_tokens": "64"}, {"output_tokens": -1}, {"output_tokens": True},
    {"output_tokens": None}, {"input_tokens": "30", "output_tokens": 4},
    {"input_tokens": 3, "cache_read_input_tokens": -5, "output_tokens": 4}, "not an object"])
def test_malformed_usage(usage):
    """Row 34. A string, a negative, a bool: that request is uncounted, with its reason."""
    tap, _ = tap_for("/v1/messages")
    tap.feed(anthropic_stream(usage))
    assert tap.finish() == USAGE_MALFORMED


def test_unparseable_usage_json_is_a_parser_error():
    tap, _ = tap_for("/v1/messages")
    tap.feed(b"event: message_delta\ndata: {not json\n\nevent: message_stop\ndata: {}\n\n")
    assert tap.finish() == PARSER_ERROR


def test_too_short_to_time():
    """Row 35. The cluster answered 32 tokens in about a third of a second: the counts stay exact,
    but a decode span under 1 s is not a rate."""
    got = replay("cluster-anthropic-stream")
    first = completed_at("cluster-anthropic-stream", b"event: content_block_delta")
    usage = completed_at("cluster-anthropic-stream", b"event: message_delta")
    assert usage - first < 1.0
    assert (got.prompt_tokens, got.output_tokens) == (72, 32)
    assert got.decode_tok_s is None and got.decode_why == TOO_SHORT_TO_TIME
    tap, clock = tap_for("/v1/messages")
    tap.feed(sse(("content_block_delta", {"type": "content_block_delta"})))
    clock.t = 30.0
    tap.feed(anthropic_stream({"input_tokens": 1, "output_tokens": 7}, deltas=0))
    assert tap.finish().decode_why == TOO_SHORT_TO_TIME, "under 8 tokens is too few, however long"


def test_client_gone_counts_nothing():
    """Row 31, at the tap: the client hung up after the first content delta — no terminal event."""
    assert replay("anthropic-abandoned", client_gone=True) == CLIENT_GONE


def test_a_stream_that_ends_without_its_terminal_event():
    body, _ = capture("anthropic-stream")
    cut = body[:body.index(b"event: message_stop")]
    tap, _ = tap_for("/v1/messages")
    tap.feed(cut)
    assert tap.finish() == NO_TERMINAL_EVENT
    tap, _ = tap_for("/v1/messages")
    tap.feed(cut)
    assert tap.finish(client_gone=True) == CLIENT_GONE


def test_the_tap_never_raises():
    """Row 34. Whatever the bytes, feed() and finish() return; finish() says a Usage, a reason from
    the closed set, or nothing."""
    rng = random.Random(7)
    shapes = [("/v1/messages", SSE), ("/v1/chat/completions", SSE), ("/api/chat", NDJSON),
              ("/v1/messages", JSON), ("/api/generate", JSON)]
    for i in range(400):
        path, ctype = shapes[i % len(shapes)]
        tap = UsageTap(path, rng.choice([200, 200, 200, 502]), ctype, clock=Clock())
        for _ in range(rng.randint(0, 6)):
            tap.feed(bytes(rng.getrandbits(8) for _ in range(rng.randint(0, 300))))
        got = tap.finish(client_gone=rng.random() < 0.2)
        assert got is None or isinstance(got, Usage) or got in UNCOUNTED


def test_the_uncounted_reasons_are_a_closed_set():
    assert set(UNCOUNTED) == {"client_gone", "no_usage_in_stream", "parser_error", "usage_malformed",
                              "body_over_cap", "error_status", "error_event", "no_terminal_event"}
