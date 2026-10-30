"""UsageTap: the tokens a response carries, read from the bytes the relay has already sent.

One tap per inference response (worklog/vllm-gateway-throughput.md, "Proxy token accounting"). The
relay feeds it each chunk AFTER that chunk was written to the client, and asks it for exactly one
verdict in its `finally`. The tap does no I/O, takes no lock and never raises: anything unexpected
leaves that request uncounted (`parser_error`), and it never touches a relayed byte.

A request COUNTS only when its terminal event was parsed carrying well-formed usage. Every other
response is either a shape this LB does not account — `finish()` returns None, which is not an
error — or uncounted, with its reason from the closed set `UNCOUNTED`.

MID-STREAM (`progress`, the page's REQUESTS card, request_queue.py): WHEN `message_start` arrived and the
prompt (and cache-read) tokens it reported; the content events relayed since, by the kind of block they belong
to (thinking, text, tool) — events, NOT tokens: the captured cluster stream (fixtures/throughput/
cluster-anthropic-stream.body) carried 32 output tokens in 9 deltas, and SGLang's DFlash2 accepts ~3 tokens a
step (worklog/sglang-bottleneck-2026-10-03.md) — the tool_use blocks started, the stop reason, and the engine's
output count once its final usage arrives. An OpenAI stream says the same by its chunks: each delta's first key
(`reasoning` / `reasoning_content` thinking, `content` text, `tool_calls` tool), each tool call once by its index, and
its `finish_reason` the stop reason; its prompt only in the final usage chunk. All of it best-effort: an event that does not parse costs its
figure, never the request's count.

AND WHEN each content event arrived (`events_between`), so the REQUESTS card can give a turn its share of the
engine's GENERATED tok/s over THAT rate's own window (request_queue.py `DERIVED_DECODE`): no relayed stream carries a
per-chunk token count — every captured body (fixtures/throughput/*.body) reports usage at `message_start` and in the
final `message_delta` / usage chunk only — so a turn's tokens per second mid-stream can only be derived. An OpenAI
chunk of kind OTHER (the role chunk, the usage chunk) carries no generated text and is not stamped. The stamps are
the relay's monotonic clock, kept EVENTS_KEPT_S back from the newest and at most EVENTS_KEPT: one float per event.
"""
from __future__ import annotations

import bisect
import json
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable
from urllib.parse import urlsplit

from vllm_lb.rates import TOO_SHORT_TO_TIME

# ── why a request was not counted: the closed set ──────────────────────────────
CLIENT_GONE = "client_gone"
NO_USAGE_IN_STREAM = "no_usage_in_stream"
USAGE_MALFORMED = "usage_malformed"
BODY_OVER_CAP = "body_over_cap"
ERROR_STATUS = "error_status"
ERROR_EVENT = "error_event"
NO_TERMINAL_EVENT = "no_terminal_event"
PARSER_ERROR = "parser_error"
UNCOUNTED = (CLIENT_GONE, NO_USAGE_IN_STREAM, USAGE_MALFORMED, BODY_OVER_CAP, ERROR_STATUS,
             ERROR_EVENT, NO_TERMINAL_EVENT, PARSER_ERROR)

#: How a request's decode rate was timed: by the LB, from the stream's own events, or by the server.
LB_STREAM_TIMING = "LB stream timing"
SERVER_TIMING = "server timing"

#: One SSE event or one NDJSON line / one JSON body. Past a cap the request is uncounted and the
#: buffer dropped; the relay carries on regardless.
EVENT_CAP = 1 << 20
BODY_CAP = 8 << 20
#: The LB times a decode only over at least this span and this many tokens.
MIN_TIMED_SPAN_S = 1.0
MIN_TIMED_TOKENS = 8
#: How far back a tap keeps its content events' stamps (`events_between`): well past the engine's two-probe window
#: (a probe gap past 3 intervals breaks the rate, rates.py rule 5), and a bound on what a long stream holds.
EVENTS_KEPT_S = 60.0
EVENTS_KEPT = 1 << 14

ANTHROPIC, OPENAI, NATIVE = "anthropic", "openai", "native"
#: `progress`: the kind of block a content event belongs to, from its `content_block_start`.
THINKING, TEXT, TOOL, OTHER = "thinking", "text", "tool", "other"
_BLOCK_KINDS = {"thinking": THINKING, "redacted_thinking": THINKING, "text": TEXT,
                "tool_use": TOOL, "server_tool_use": TOOL}
SSE, NDJSON, JSON_BODY = "sse", "ndjson", "json"

#: The accounted shapes: request path (query stripped — Claude Code sends `/v1/messages?beta=true`)
#: to protocol, and each protocol's response content types to their framing.
PATHS = {"/v1/messages": ANTHROPIC, "/v1/chat/completions": OPENAI, "/v1/completions": OPENAI,
         "/api/chat": NATIVE, "/api/generate": NATIVE}
FRAMINGS = {ANTHROPIC: {"text/event-stream": SSE, "application/json": JSON_BODY},
            OPENAI: {"text/event-stream": SSE, "application/json": JSON_BODY},
            NATIVE: {"application/x-ndjson": NDJSON, "application/json": JSON_BODY}}

_CONTENT_DELTA = b"event: content_block_delta\n"
_DATA = b"data: "
_DONE = b"data: [DONE]"
#: AN OPENAI CHUNK'S KIND, by the first key of its `delta` (vLLM's chunks, captured in
#: fixtures/throughput/cluster-chat-stream-usage.body: `{"reasoning": …}`, `{"content": …}`, `{"tool_calls": […]}`;
#: `reasoning_content` is the older spelling). The role chunk that opens a stream, and anything else, is OTHER.
_OPENAI_DELTA = b'"delta":'
_OPENAI_KINDS = {b"reasoning": THINKING, b"reasoning_content": THINKING, b"content": TEXT, b"tool_calls": TOOL}
_FINISH = b'"finish_reason":'


@dataclass(frozen=True)
class Usage:
    """One counted request."""
    path: str
    prompt_tokens: int | None           # None: the response did not say
    output_tokens: int
    decode_tok_s: float | None
    decode_basis: str | None            # LB_STREAM_TIMING | SERVER_TIMING
    decode_why: str | None              # TOO_SHORT_TO_TIME when there is no rate
    decode_s: float | None              # the timed span, when there is one
    prefill_tok_s: float | None         # only the server's own clock can time a prefill


def classify(path: str, content_type: str | None) -> tuple[str, str] | None:
    """(protocol, framing) for an accounted shape; None for anything else."""
    protocol = PATHS.get(urlsplit(path).path)
    mime = (content_type or "").split(";", 1)[0].strip().lower()
    framing = FRAMINGS[protocol].get(mime) if protocol else None
    return (protocol, framing) if framing else None


class _Malformed(Exception):
    """A usage field that is not a non-negative integer."""


def _count(obj: dict, key: str, absent: int | None = None, *, required: bool = False) -> int | None:
    """A token count or duration as a server sends it: a non-negative int, never a bool. An absent
    (or null) optional field is `absent`; anything else present is malformed."""
    value = obj.get(key)
    if value is None and not required:
        return absent
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise _Malformed(key)
    return value


def _fields(raw: bytes) -> tuple[bytes | None, bytes]:
    """An SSE event's name and its data lines joined with `\\n`; comments ignored."""
    name, data = None, []
    for line in raw.split(b"\n"):
        if not line or line[:1] == b":":
            continue
        field, _, value = line.partition(b":")
        if value[:1] == b" ":
            value = value[1:]
        if field == b"event":
            name = value
        elif field == b"data":
            data.append(value)
    return name, b"\n".join(data)


class UsageTap:
    def __init__(self, path: str, status: int, content_type: str | None, *,
                 clock: Callable[[], float] = time.monotonic, event_cap: int = EVENT_CAP,
                 body_cap: int = BODY_CAP) -> None:
        self.path = urlsplit(path).path
        self._clock = clock
        self._event_cap, self._body_cap = event_cap, body_cap
        shape = classify(path, content_type)
        self._protocol, self._framing = shape or (None, None)
        #: The verdict once decided: a Usage, or an uncounted reason. Decided means stop parsing.
        self._verdict: Usage | str | None = ERROR_STATUS if shape and status != 200 else None
        self._buf = bytearray()
        self._held_cr = False           # a chunk ended in `\r`: its `\n` may open the next one
        self._first_at: float | None = None
        self._usage: tuple[int | None, int] | None = None     # (prompt, output), awaiting the end
        self._usage_at: float | None = None
        #: `progress`: content events by the kind of block they are in, tool_use blocks started, what
        #: `message_start` reported and WHEN (at, prompt, cache-read), the final cache-read, the stop reason.
        self._deltas = {THINKING: 0, TEXT: 0, TOOL: 0, OTHER: 0}
        self._block = OTHER
        self._tool_uses = 0
        self._tool_calls: set = set()           # an OpenAI stream's tool calls seen, by (choice, index)
        self._start: tuple[float, int | None, int] | None = None
        self._final_cached: int | None = None
        self._usage_from = "message_delta"          # the event the final usage rode: Anthropic's, or "usage"
        self._stop_reason: str | None = None
        #: When each content event arrived, oldest first, on `_clock` (`events_between`).
        self._event_at: deque[float] = deque(maxlen=EVENTS_KEPT)

    # ── the relay's two calls ──────────────────────────────────────────────

    def feed(self, chunk: bytes) -> None:
        """One chunk the client has already been sent. Never raises."""
        if self._framing is None or self._verdict is not None or not chunk:
            return
        try:
            if self._framing == SSE:
                self._feed_sse(chunk)
            elif self._framing == NDJSON:
                self._feed_ndjson(chunk)
            elif len(self._buf) + len(chunk) > self._body_cap:
                self._decide(BODY_OVER_CAP)
            else:
                self._buf += chunk
        except Exception:
            self._decide(PARSER_ERROR)

    def finish(self, *, client_gone: bool = False) -> Usage | str | None:
        """This response's one verdict: a Usage, an uncounted reason, or None when the shape is not
        accounted at all. Never raises."""
        if self._framing is None:
            return None
        if self._verdict is None:
            try:
                if self._held_cr:
                    # A stream that ENDS in a lone `\r`: no `\n` is coming, so that `\r` was a line end.
                    # Every byte fed here reached the client, so its last event counts as delivered.
                    self._held_cr = False
                    self._feed_sse(b"\n")
            except Exception:
                self._decide(PARSER_ERROR)
        if self._verdict is None:
            try:
                if client_gone:
                    self._decide(CLIENT_GONE)
                elif self._framing == JSON_BODY:
                    self._decide(self._count_body())
                else:
                    self._decide(NO_TERMINAL_EVENT)
            except Exception:
                self._decide(PARSER_ERROR)
        return self._verdict

    def _decide(self, verdict: Usage | str) -> None:
        self._verdict = verdict
        self._buf = bytearray()

    @property
    def verdict(self) -> Usage | str | None:
        """The verdict once decided (a Usage or an uncounted reason), else None. Reading it decides nothing."""
        return self._verdict

    def progress(self) -> dict:
        """What the stream has said SO FAR — read by another thread while the relay feeds this one, so it only
        reads. `prompt_from` names the event the prompt figures came from; `start_at` is on this tap's clock."""
        usage, start, deltas = self._usage, self._start, dict(self._deltas)
        if usage is not None and usage[0] is not None:
            prompt, cached, source = usage[0], self._final_cached, self._usage_from
        elif start is not None and start[1] is not None:
            prompt, cached, source = start[1], start[2], "message_start"
        else:
            prompt, cached, source = None, None, None
        return {"deltas": sum(deltas.values()), THINKING: deltas[THINKING], TEXT: deltas[TEXT], TOOL: deltas[TOOL],
                "tool_uses": self._tool_uses, "prompt_tokens": prompt, "cached_tokens": cached,
                "prompt_from": source, "output_tokens": usage[1] if usage is not None else None,
                "start_at": start[0] if start is not None else None, "stop_reason": self._stop_reason}

    def events_between(self, start: float, end: float) -> int:
        """How many content events arrived in (`start`, `end`] on this tap's clock — read by another thread while the
        relay feeds this one: a copy of the stamps, never the deque itself."""
        stamps = list(self._event_at)
        return max(0, bisect.bisect_right(stamps, end) - bisect.bisect_right(stamps, start))

    def _stamp(self) -> float:
        """One content event, now: its stamp kept, the ones older than EVENTS_KEPT_S dropped."""
        at = self._clock()
        stamps = self._event_at
        stamps.append(at)
        while stamps[0] < at - EVENTS_KEPT_S:
            stamps.popleft()
        return at

    # ── SSE (Anthropic and OpenAI streams) ─────────────────────────────────

    def _feed_sse(self, chunk: bytes) -> None:
        if self._held_cr:
            chunk, self._held_cr = b"\r" + chunk, False
        if chunk[-1:] == b"\r":
            chunk, self._held_cr = chunk[:-1], True
        if b"\r" in chunk:
            chunk = chunk.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        buf = self._buf
        scan = max(0, len(buf) - 1)      # a blank line may straddle the old tail and this chunk
        buf += chunk
        start = 0
        while self._verdict is None:
            end = buf.find(b"\n\n", max(start, scan))
            if end == -1:
                break
            self._sse_event(buf, start, end)
            start = end + 2
        if self._verdict is None:
            del buf[:start]
            if len(buf) > self._event_cap:
                self._decide(BODY_OVER_CAP)

    def _sse_event(self, buf: bytearray, start: int, end: int) -> None:
        if self._protocol == ANTHROPIC:
            if buf.startswith(_CONTENT_DELTA, start):          # nearly every event: one stamp
                self._deltas[self._block] += 1
                at = self._stamp()
                if self._first_at is None:
                    self._first_at = at
                return
            name, data = _fields(bytes(buf[start:end]))
            if name == b"content_block_delta":
                self._deltas[self._block] += 1
                at = self._stamp()
                if self._first_at is None:
                    self._first_at = at
            elif name == b"content_block_start":
                self._block_start(data)
            elif name == b"message_start":
                self._message_start(data)
            elif name == b"message_delta":
                self._anthropic_usage(json.loads(data))
            elif name == b"message_stop":
                self._count_stream()
            elif name == b"error":
                self._decide(ERROR_EVENT)
            return
        if buf.startswith(_DATA, start) and buf.find(b"\n", start, end) == -1:
            data_at = start + len(_DATA)              # one `data:` line: no copy unless it matters
        else:
            data = _fields(bytes(buf[start:end]))[1]
            if not data:
                return
            buf, data_at, end = bytearray(data), 0, len(data)
        if self._first_at is None:
            self._first_at = self._clock()
        if end - data_at == len(_DONE) - len(_DATA) and buf.startswith(_DONE[len(_DATA):], data_at):
            self._count_stream()
            return
        kind = self._openai_kind(buf, data_at, end)
        self._deltas[kind] += 1
        if kind != OTHER:
            self._stamp()
        finish = buf.find(_FINISH, data_at, end)
        said_finish = finish != -1 and buf[finish + len(_FINISH):end].lstrip()[:1] == b'"'
        if (kind == TOOL or said_finish or buf.find(b'"usage"', data_at, end) != -1
                or buf.find(b'"error"', data_at, end) != -1):
            self._openai_chunk(json.loads(bytes(buf[data_at:end])))

    @staticmethod
    def _openai_kind(buf: bytearray, start: int, end: int) -> str:
        """The kind of one OpenAI chunk, by its delta's FIRST key — read in place, no parse (nearly every event)."""
        at = buf.find(_OPENAI_DELTA, start, end)
        if at == -1:
            return OTHER
        i = at + len(_OPENAI_DELTA)
        while i < end and buf[i] in b" \t":
            i += 1
        if i >= end or buf[i] != ord("{"):
            return OTHER
        i += 1
        while i < end and buf[i] in b" \t":
            i += 1
        if i >= end or buf[i] != ord('"'):
            return OTHER
        close = buf.find(b'"', i + 1, end)
        return OTHER if close == -1 else _OPENAI_KINDS.get(bytes(buf[i + 1:close]), OTHER)

    def _message_start(self, data: bytes) -> None:
        """WHEN `message_start` arrived, and the prompt it reports — input + cache read + cache creation (vLLM 0.29
        reports the cached apart, as `_anthropic_usage` folds them) — best-effort, for `progress` only: whatever goes
        wrong here costs those figures and nothing else (the count rides `message_delta`)."""
        at = self._clock()
        try:
            usage = json.loads(data)["message"]["usage"]
            given = _count(usage, "input_tokens")
            read = _count(usage, "cache_read_input_tokens", 0)
            total = None if given is None else given + read + _count(usage, "cache_creation_input_tokens", 0)
            self._start = (at, total, read)
        except Exception:
            self._start = (at, None, 0)

    def _block_start(self, data: bytes) -> None:
        """The kind of block the deltas that follow belong to — Anthropic blocks never interleave — and one more
        tool_use when it is one. Best-effort, like `_message_start`."""
        try:
            kind = json.loads(data)["content_block"]["type"]
        except Exception:
            kind = None
        self._block = _BLOCK_KINDS.get(kind, OTHER)
        if self._block == TOOL:
            self._tool_uses += 1

    def _anthropic_usage(self, event: dict) -> None:
        delta = event.get("delta") if isinstance(event, dict) else None
        if isinstance(delta, dict) and isinstance(delta.get("stop_reason"), str):
            self._stop_reason = delta["stop_reason"][:32]
        usage = event.get("usage") if isinstance(event, dict) else None
        try:
            if not isinstance(usage, dict):
                raise _Malformed("usage")
            output = _count(usage, "output_tokens", required=True)
            given = _count(usage, "input_tokens")
            # vLLM 0.29 subtracts cache tokens from `input_tokens` and reports them apart.
            read = _count(usage, "cache_read_input_tokens", 0)
            cached = read + _count(usage, "cache_creation_input_tokens", 0)
        except _Malformed:
            return self._decide(USAGE_MALFORMED)
        self._final_cached = read
        self._usage = (None if given is None else given + cached, output)
        self._usage_at = self._clock()

    def _openai_chunk(self, chunk) -> None:
        if not isinstance(chunk, dict):
            return
        if chunk.get("error") is not None:
            return self._decide(ERROR_EVENT)
        # `progress`: each tool call ONCE, by its (choice, index) — its later chunks carry the arguments under the
        # same index — and the stream's own finish_reason, its stop reason. Best-effort: a shape it does not know
        # costs these figures, never the count.
        for choice in chunk.get("choices") or ():
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            calls = delta.get("tool_calls") if isinstance(delta, dict) else None
            for call in calls if isinstance(calls, list) else ():
                key = (choice.get("index"), call.get("index") if isinstance(call, dict) else None)
                if key not in self._tool_calls:
                    self._tool_calls.add(key)
                    self._tool_uses += 1
            if isinstance(choice.get("finish_reason"), str):
                self._stop_reason = choice["finish_reason"][:32]
        # Only the `choices: []` chunk carries the real usage; `/v1/completions` puts a ZEROED usage
        # on every other chunk (ollama middleware/openai.go:195-196).
        if chunk.get("choices") != [] or "usage" not in chunk:
            return
        usage = chunk["usage"]
        try:
            if not isinstance(usage, dict):
                raise _Malformed("usage")
            output = _count(usage, "completion_tokens", required=True)
            prompt = _count(usage, "prompt_tokens", required=True)
        except _Malformed:
            return self._decide(USAGE_MALFORMED)
        self._usage_from = "usage"                  # OpenAI's `choices: []` chunk
        self._usage = (prompt, output)
        self._usage_at = self._clock()

    def _count_stream(self) -> None:
        """The terminal event of a stream the LB times itself: (output − 1) over the span from the first
        content event to the usage event, when that span and that output are long enough."""
        if self._usage is None:
            return self._decide(NO_USAGE_IN_STREAM)
        prompt, output = self._usage
        span = (self._usage_at - self._first_at
                if self._first_at is not None and self._usage_at is not None else None)
        timed = span is not None and span >= MIN_TIMED_SPAN_S and output >= MIN_TIMED_TOKENS
        self._decide(Usage(self.path, prompt, output,
                           decode_tok_s=(output - 1) / span if timed else None,
                           decode_basis=LB_STREAM_TIMING if timed else None,
                           decode_why=None if timed else TOO_SHORT_TO_TIME,
                           decode_s=span, prefill_tok_s=None))

    # ── NDJSON (native /api/chat, /api/generate) ───────────────────────────

    def _feed_ndjson(self, chunk: bytes) -> None:
        buf = self._buf
        buf += chunk
        start = 0
        while self._verdict is None:
            end = buf.find(b"\n", start)
            if end == -1:
                break
            if end - start > self._event_cap:
                return self._decide(BODY_OVER_CAP)
            # Go's encoder writes no space in `"done":true`, and a quote inside text is escaped, so
            # message text cannot forge either match.
            if buf.find(b'"done":true', start, end) != -1 or buf.find(b'"error"', start, end) != -1:
                self._native_line(json.loads(bytes(buf[start:end])))
            start = end + 1
        if self._verdict is None:
            del buf[:start]
            if len(buf) > self._event_cap:
                self._decide(BODY_OVER_CAP)

    def _native_line(self, line) -> None:
        if not isinstance(line, dict):
            return
        if line.get("error") is not None:
            return self._decide(ERROR_EVENT)
        if line.get("done") is True:
            self._decide(self._count_native(line))

    @staticmethod
    def _count_native_fields(line: dict) -> tuple[int, int, int, int]:
        # `omitempty` (ollama api/types.go:557-563): an absent count or duration is 0.
        return (_count(line, "prompt_eval_count", 0), _count(line, "eval_count", 0),
                _count(line, "prompt_eval_duration", 0), _count(line, "eval_duration", 0))

    def _count_native(self, line: dict) -> Usage | str:
        """ollama's own counts and clocks (nanoseconds)."""
        try:
            prompt, output, prompt_ns, eval_ns = self._count_native_fields(line)
        except _Malformed:
            return USAGE_MALFORMED
        timed = output > 0 and eval_ns > 0
        return Usage(self.path, prompt, output,
                     decode_tok_s=output / (eval_ns / 1e9) if timed else None,
                     decode_basis=SERVER_TIMING if timed else None,
                     decode_why=None if timed else TOO_SHORT_TO_TIME,
                     decode_s=eval_ns / 1e9 if timed else None,
                     prefill_tok_s=prompt / (prompt_ns / 1e9) if prompt > 0 and prompt_ns > 0 else None)

    # ── one JSON body ──────────────────────────────────────────────────────

    def _count_body(self) -> Usage | str:
        body = json.loads(bytes(self._buf))
        if not isinstance(body, dict):
            return NO_USAGE_IN_STREAM
        if body.get("error") is not None or body.get("type") == "error":
            return ERROR_EVENT
        if self._protocol == NATIVE:
            return self._count_native(body)
        usage = body.get("usage")
        try:
            if not isinstance(usage, dict):
                return NO_USAGE_IN_STREAM
            if self._protocol == ANTHROPIC:
                output = _count(usage, "output_tokens", required=True)
                given = _count(usage, "input_tokens")
                cached = _count(usage, "cache_read_input_tokens", 0) + _count(usage, "cache_creation_input_tokens", 0)
                prompt = None if given is None else given + cached
            else:
                output = _count(usage, "completion_tokens", required=True)
                prompt = _count(usage, "prompt_tokens", required=True)
        except _Malformed:
            return USAGE_MALFORMED
        return Usage(self.path, prompt, output, decode_tok_s=None, decode_basis=None,
                     decode_why=TOO_SHORT_TO_TIME, decode_s=None, prefill_tok_s=None)
