"""The wire: an HTTP/1.1 proxy that is ALWAYS listening, whatever the backends are doing.

Rules it keeps:
  * every inference request is counted on ARRIVAL (demand is what leases are made of, and
    a request that found nothing live is the strongest demand signal there is);
  * the backend is chosen per request by the pure router, the `model` field is rewritten to
    what that backend serves, and the body is forwarded byte-for-byte otherwise;
  * failover happens ONLY before the response headers (a connect/handshake failure
    penalises the backend and the next candidate is tried); once headers are through, the
    stream is passed as it arrives (`read1`, never a buffered `read(n)`) and a mid-stream
    failure ends the response — the client retries, the proxy never duplicates a request;
  * the LB's own endpoints (`/healthz`, `/lb/status`, `/v1/models`) never touch a backend;
  * nothing held ⇒ a bounded wait while a claim is all that is missing, then 503 JSON; every
    candidate refused ⇒ 502 JSON naming who was tried;
  * nothing live while the operator WANTS the cluster and it is on its way (a restart, a model switch)
    ⇒ the turn is HELD, no byte sent, until the pair serves — bounded by `cluster_hold_s` and
    `hold_max`; every 503 carries `Retry-After` (≤ 60 s), and `x-should-retry: false` when nothing is
    on its way;
  * SIGTERM DRAINS (`drain`): turns in flight stream to their end, a turn arriving meanwhile is held and
    answered 503 `draining` the moment the drain is over;
  * a relayed `/v1/messages` stream silent for `sse_keepalive_s` gets `: keepalive` at an event boundary, and one
    whose backend fails at an event boundary ends with an `overloaded_error` event; the upstream socket
    probes its peer, so a backend that vanished without a FIN is noticed in ~2 min.
"""
from __future__ import annotations

import collections
import http.client
import itertools
import json
import logging
import math
import queue
import socket
import threading
import time
import types
import pathlib
import re
import urllib.parse
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from vllm_lb import multipart, prompt_count, router, topology, websearch
from vllm_lb.bench import BenchStore
from vllm_lb.benchswarm import CAP_MIN_RANGE, BenchSwarm, SwarmRefused
from vllm_lb.catalog import SGLANG as SGLANG_ENGINE
from vllm_lb.catalog import VLLM as VLLM_ENGINE
from vllm_lb.catalog import (CLASSIC, CLUSTER_MODE, ENGINE_MODELED_PERF, MAX_MODEL_LEN_CAP, RANK_ROLES, ModelCatalog,
                             UnknownModelChoice)
from vllm_lb.cluster import (KV_AUTO, KV_CACHE_MEMORY_RANGE, KV_OFFLOAD_GIB_RANGE, LONG_PREFILL_TOKEN_THRESHOLD_RANGE,
                            MAX_NUM_BATCHED_TOKENS_RANGE, NUM_SPECULATIVE_TOKENS_RANGE)
from vllm_lb.cluster_defaults import box_total, kv_auto, kv_auto_resolver, memory_admission, restart_basis
from vllm_lb.config import Config
from vllm_lb.fleet import BLUEPRINTS_MOVED
from vllm_lb.host_guard import HostGuard
from vllm_lb.hub import Hub
from vllm_lb.memory_estimate import CLASSIC_MEMORY, estimate_memory
from vllm_lb.models_view import models_payload
from vllm_lb.connect import connect_view
from vllm_lb import cores
from vllm_lb import request_kill
from vllm_lb import request_queue
from vllm_lb import pricing as _pricing
from vllm_lb.perflog import (BOOT_PREFIX, PerfLog, effective_config, engine_start, of_cluster, sessions,
                             throughput_target)
from vllm_lb.start_rates import start_rates
from vllm_lb.saturation import ENGINE_NAMES, History, gpu_view, judge, per_box
from vllm_lb.state import CLUSTER, LEASE_RUNNING, VLLM, Backend
from vllm_lb.scaler import desired_leases
from vllm_lb.usage_tap import UsageTap

#: `_perf_days`'s "that is not a number of days": a value only the endpoints may turn into words.
_BAD_DAYS = object()
_BAD_SERVER = object()                     # `?server=` named no declared server: the 404 is already sent

#: How often a turn waiting for a claim looks again (`Config.admission_wait_s`).
_CLAIM_POLL_S = 0.25

#: THE MOST A `Retry-After` MAY SAY. Measured against Claude Code 2.1.288: on a 503 it reads only this
#: header, in seconds, and one over 60 makes it give up after a single request
#: (`api_request_retry_after_too_long`, a 60,000 ms cap) — so a longer wait is never asked for; at 60 its ten
#: default retries span ~10 minutes.
RETRY_AFTER_MAX_S = 60

#: Cluster states that do not end on their own: a `failed` pair waits for Stop, then Start, and so does a
#: `stalled` one (a hung start halted after its relaunches — cluster.plan); a `blocked` one for another model.
#: A held turn is released at once on any of them, and told not to retry (`x-should-retry`).
CLUSTER_NEEDS_OPERATOR = ("failed", "blocked", "stalled")

#: How often a held turn looks again with no change published. The hub broadcasts every change that can
#: make a backend routable, so this is only the floor under one that does not (a penalty running out).
_HOLD_RECHECK_S = 5.0

#: How often a drain says what it is still waiting on, and how long the turns it held get to receive their
#: 503 once it is over, before the process goes.
_DRAIN_LOG_EVERY_S = 30.0
_DRAIN_ANSWER_S = 5.0

#: An SSE comment: every parser skips it, and it is a BYTE — what the client's idle watchdog counts.
KEEPALIVE_COMMENT = b": keepalive\n\n"

#: TCP keepalive on the UPSTREAM socket: probe a peer silent for 60 s every 10 s, give up after 6 unanswered —
#: a backend that vanished without a FIN or RST (power loss, a pulled cable; box-a lost power on
#: 2026-09-19) is reported ~2 min after its last byte instead of after `stall_timeout_s` (30 min). A
#: live-but-silent backend ACKs every probe from its kernel, so a long prefill is never cut by it.
UPSTREAM_KEEPALIVE = (60, 10, 6)            # idle s, interval s, count


def upstream_keepalive(sock: socket.socket) -> None:
    """Turn on the probes above (`UPSTREAM_KEEPALIVE`). Linux names all three; elsewhere the kernel's own
    timings stand."""
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    for name, value in zip(("TCP_KEEPIDLE", "TCP_KEEPINTVL", "TCP_KEEPCNT"), UPSTREAM_KEEPALIVE):
        opt = getattr(socket, name, None)
        if opt is not None:
            sock.setsockopt(socket.IPPROTO_TCP, opt, value)


def retry_after_header(seconds: float) -> str:
    """A `Retry-After` value: whole seconds, at least 1, never past `RETRY_AFTER_MAX_S`."""
    return str(max(1, min(RETRY_AFTER_MAX_S, math.ceil(seconds))))


def cluster_coming(view: dict, stored_mode: str | None = None) -> bool:
    """Is the TP=2 pair ON ITS WAY — worth holding a turn for?

    Yes while the operator WANTS it (`desired.mode == cluster`) and it is in any state but one that needs the
    operator: a switch passes through restarting, checking, reserving, waiting, draining, freeing and
    starting (cluster.plan), and `unknown` — a tick that could not observe a GB10 box — usually clears on the
    next. No when it is desired off (unloading, releasing, off: the GB10 boxes go back to the fleet), not
    configured, or `failed`/`blocked`/`stalled`.

    `stored_mode` is the switch on disk, for a view the cluster loop has not published yet: the first
    seconds after an LB redeploy, which is exactly where a client retrying onto the new container lands."""
    if not view.get("configured"):
        return False
    mode = (view.get("desired") or {}).get("mode") or stored_mode
    return mode == CLUSTER_MODE and view.get("state") not in CLUSTER_NEEDS_OPERATOR


def _at_boundary(tail: bytes) -> bool:
    """Has the client been given whole SSE events only — nothing, or bytes ending in a blank line? Only
    there may the relay add a byte of its own: a comment inside a half-relayed event would corrupt it."""
    return not tail or tail.endswith(b"\n\n") or tail.endswith(b"\r\n\r\n")


def _sse_error(message: str) -> bytes:
    """The Anthropic stream's own error event. `overloaded_error` is the type the client RETRIES (with
    backoff) — a stream that just stops reads as "Connection lost mid-response"."""
    return (b"event: error\ndata: "
            + json.dumps({"type": "error", "error": {"type": "overloaded_error", "message": message}}).encode()
            + b"\n\n")


#: `_Pump.next()`: the backend said nothing for a keepalive interval.
_SILENT = object()
#: `_hold_for_cluster`: the drain began — the turn waits for it to end, then is answered `draining`.
_DRAINING = object()
#: `_hold_for_cluster`: the operator killed the turn while it was held — it is answered at once, never sent.
_KILLED = object()

#: WHO IS SIGNED IN, as the gate names them: vllm-gate answers nginx's `auth_request` with it
#: (the sign-in gate's decision — a service outside this repository), and nginx REPLACES any client-sent copy with
#: that answer on the way to the LB (the deployment's edge vhost). Read for one thing — who killed a turn,
#: in its log line and its row — and never forwarded to an engine.
GATE_USER_HEADER = "X-Vllm-User"
_GATE_USER = GATE_USER_HEADER.lower()
#: An id as `_TURN_IDS` writes one: lowercase hex. Anything else names no turn.
_TURN_ID = re.compile(r"^[0-9a-f]{1,16}$")


class _Pump:
    """Reads the BACKEND in a thread of its own, so the relay can tell silence from progress without
    timing the socket out.

    A socket read that times out poisons the file object (`cannot read from timed out object`), so the
    relay cannot simply wait 15 s and read again. The reader keeps the socket's own timeout —
    `stall_timeout_s`, the BACKEND's silence, unchanged — and hands each chunk over a short queue; the
    relay waits on the queue for `every` seconds and hears `_SILENT` when nothing came. The queue is
    bounded, so the backend is still read no faster than the client takes it."""

    def __init__(self, resp, every: float) -> None:
        self._resp = resp
        self._every = every
        self._q: queue.Queue = queue.Queue(maxsize=8)
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="vllm-lb-pump", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            while True:
                chunk = self._resp.read1(_CHUNK)
                if not self._put(chunk) or not chunk:
                    return
        except BaseException as exc:          # noqa: BLE001 — handed to the relay, which decides
            self._put(exc)

    def _put(self, item) -> bool:
        while not self._closed:
            try:
                self._q.put(item, timeout=1.0)
                return True
            except queue.Full:
                continue
        return False

    def next(self):
        """The next chunk (b"" at the end), `_SILENT` after `every` seconds of nothing, or the read's error."""
        try:
            item = self._q.get(timeout=self._every)
        except queue.Empty:
            return _SILENT
        if isinstance(item, BaseException):
            raise item
        return item

    def close(self, sock) -> None:
        """Stop the reader. Shutting the socket down is what wakes a read blocked in the kernel — closing
        the file alone leaves it parked until the stall budget."""
        self._closed = True
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        self._thread.join(2.0)


#: A turn's id on the REQUESTS card: short, and never reused while this process lives.
_TURN_IDS = itertools.count(1)


def _shut(sock) -> None:
    """Close a connection to an engine in BOTH directions, from any thread: the engine reads EOF (its client went —
    request_kill.py), and a read the relay or a `_Pump` is blocked in returns at once."""
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass


class _Turn:
    """One inference request this process has admitted, as the drain sees it. `holding` = waiting for a
    backend it does not have (the cluster hold, or the end of a drain): the drain does not wait for those —
    they have nothing in flight — and `hold_max` counts them.

    And as the REQUESTS card sees it (request_queue.py): who sent it (User-Agent, session), the bounded facts of
    its body — read off the body `handle` already parsed — and the relay's clocks: `sent_at` when a backend was
    chosen, `first_at`/`last_at` the first and latest body byte to the client, `keepalives`, the usage `tap`,
    and how it ended (`status` as written, `error`, `client_gone`, `ended` = the stream's own `message_stop`).

    And THE OPERATOR'S KILL (`kill`, POST /lb/requests/<id>/kill): `killed` — who and why, set once — and
    `kill_answered`, set where the relay or a hold acted on it (the client told, the engine's connection closed). A
    kill that lands after the turn ended on its own leaves `kill_answered` False: its outcome stays its own."""
    __slots__ = ("method", "path", "client", "started", "backend", "sent", "holding",
                 "id", "started_at", "client_kind", "client_detail", "session", "facts", "box", "sent_at",
                 "first_at", "last_at", "keepalives", "tap", "status", "error", "client_gone", "ended",
                 "anthropic_sse", "killed", "kill_answered", "counted", "priority", "requeue", "committed",
                 "_upstream", "_lock")

    def __init__(self, h, *, holding: bool = False, body: dict | None = None) -> None:
        self.method, self.path = h.command, urllib.parse.urlsplit(h.path).path
        self.client = h.client_address[0] if h.client_address else "?"
        self.started = time.monotonic()
        self.backend: str | None = None
        self.sent = 0
        self.holding = holding
        self.id = f"{next(_TURN_IDS):x}"
        self.started_at = time.time()
        headers = getattr(h, "headers", None)
        self.client_kind, self.client_detail = request_queue.client_of(
            headers.get("User-Agent") if headers is not None else None)
        self.session = request_queue.session_of(headers, body)
        self.facts = request_queue.request_facts(body)
        self.box: str | None = None
        self.sent_at: float | None = None
        self.first_at: float | None = None
        self.last_at: float | None = None
        self.keepalives = 0
        self.tap: UsageTap | None = None
        self.status: int | None = None
        self.error: str | None = None
        self.client_gone = False
        self.ended = False
        self.anthropic_sse = False
        self.killed: dict | None = None
        self.kill_answered = False
        self.counted: dict | None = None     # the engine's prompt count (prompt_count.py), set by its own thread
        # THE CALLER'S PRIORITY (router.PRIORITY_HEADER, the page's toggle): what each send's body carries; `requeue`:
        # the toggle moved it after a send — the relay sends it again; `committed`: the client has its first byte.
        self.priority: int | None = None
        self.requeue = False
        self.committed = False
        self._upstream: socket.socket | None = None
        self._lock = threading.Lock()

    # ── the kill: ONE mechanism for every engine path — the turn's own connection to its engine ──────────

    def attach(self, sock: socket.socket) -> None:
        """The connection to the engine this turn is on now: what a kill closes. A kill that came first closes it
        at once, so a kill landing while the relay connects is never lost."""
        with self._lock:
            self._upstream = sock
            cut = self.killed is not None or self.requeue
        if cut:
            _shut(sock)

    def detach(self) -> None:
        with self._lock:
            self._upstream = None

    def kill(self, why: dict) -> bool:
        """The operator's kill, recorded ONCE (False for a second). The engine connection, when there is one, is
        closed here — the engine sees its client go, and the relay's blocked read returns; what the client is told
        is the relay's (`Proxy._answer_killed`, the relay's own error event). A turn with no connection (held, or
        between backends) is woken by the caller and answers it where it waits."""
        with self._lock:
            if self.killed is not None:
                return False
            self.killed = why
            sock = self._upstream
        if sock is not None:
            _shut(sock)
        return True

    # ── the priority toggle: the same connection, closed so the turn is SENT AGAIN ─────────────────────────────

    def reprioritize(self, priority: int | None) -> str | None:
        """The page's toggle: `priority` from the next send on. A turn already sent (`sent_at`) has its engine
        connection closed and is sent again by the relay ("requeued") — the client has had nothing of it — one not
        yet sent just carries it ("set"). None: the client has its first byte (`committed`), nothing moves."""
        with self._lock:
            if self.committed:
                return None
            self.priority = priority
            if self.sent_at is None:
                return "set"
            self.requeue = True
            sock = self._upstream
        if sock is not None:
            _shut(sock)
        return "requeued"

    def take_requeue(self) -> bool:
        """Was the turn's connection closed for a re-send? Consumed: the relay sends it once more."""
        with self._lock:
            requeue, self.requeue = self.requeue, False
        return requeue

    def commit(self) -> bool:
        """The relay is about to give the client the answer's first byte: from here the priority is fixed — False
        (consumed) when a toggle closed the connection first, so this answer is dropped and the turn sent again."""
        with self._lock:
            if self.requeue:
                self.requeue = False
                return False
            self.committed = True
            return True

    def describe(self, now: float) -> str:
        return (f"{self.method} {self.path} → {self.backend or 'no backend yet'} "
                f"({now - self.started:.0f} s, {self.sent} B, from {self.client})")


@dataclass(frozen=True)
class _Route:
    """What a request's routing is asked with: the `model` it names (None: none) and whether that name is a catalog
    model only a NON-default cluster of this fleet runs (`exact`, `router.exact_models`) — served as named or not at all (`router.serving`).
    `task`: what the request's path asks of a model (`router.task_of_path`) — only backends serving it are candidates."""
    model: str | None = None
    exact: bool = False
    task: str = router.CHAT


class _Refusal:
    """A 503 `no_backend`, decided: what it says, how long the client should wait, whether to retry at all."""
    __slots__ = ("message", "retry_after_s", "retry")

    def __init__(self, message: str, retry_after_s: float, retry: bool = True) -> None:
        self.message, self.retry_after_s, self.retry = message, retry_after_s, retry

#: The gateway UI, served by the LB itself. One directory, no templating, no directory listing.
STATIC_DIR = (pathlib.Path(__file__).resolve().parent / "static")
_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
    ".json": "application/json",
}


def static_file(url_path: str) -> tuple[pathlib.Path, str] | None:
    """The file this URL names inside STATIC_DIR, or None.

    Containment is verified AFTER resolution, never by inspecting the request string: this
    process holds broker credentials and is reachable from the internet through the tunnel, so
    `..`, encoded `..`, doubled slashes and symlinks all have to end at the same check. A name
    that resolves outside the directory is simply not found.
    """
    name = urllib.parse.unquote(url_path).lstrip("/") or "index.html"
    if "\x00" in name:
        return None
    # EVERY filesystem touch is inside the guard, `is_file()` included. A single segment of
    # 256+ bytes makes it raise OSError(36) — which escaped, became an unauthenticated 500 on
    # an internet-facing endpoint, and wrote the whole attacker-controlled path to the log
    # TWICE per request (a 60 KB path grew the log by 120 KB). A symlink loop inside static/
    # raises RuntimeError and did the same. Anything that cannot be resolved to a real file in
    # this directory is simply not found.
    try:
        target = (STATIC_DIR / name).resolve()
        target.relative_to(STATIC_DIR.resolve())
        if not target.is_file():
            return None
    except (ValueError, OSError, RuntimeError):
        return None
    return target, _CONTENT_TYPES.get(target.suffix, "application/octet-stream")

log = logging.getLogger(__name__)

HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
              "te", "trailer", "transfer-encoding", "upgrade"}

#: On every response the LB writes itself (`_Handler.send_own`). The page has a Start/Stop switch
#: for a cluster that takes most of an hour to start, so no other origin may frame it and steer a
#: click, and nothing served here may be sniffed into a type it was not sent as. CSP is
#: `frame-ancestors` and nothing else: a script-src would be a second policy nobody asked for. A
#: relayed inference answer keeps the backend's own headers.
OWN_HEADERS = (
    ("X-Frame-Options", "DENY"),
    ("Content-Security-Policy", "frame-ancestors 'none'"),
    ("X-Content-Type-Options", "nosniff"),
    ("Referrer-Policy", "no-referrer"),
)
_CHUNK = 64 * 1024
#: On a 5xx the upstream's own error envelope is appended to the per-request log line so a
#: failure is diagnosable from `docker logs vllm-lb` without SSHing the engine box — a bare
#: `500 79B` named no reason, which is what left an SGLang template-render 500 (`Unexpected
#: reasoning effort max`) invisible while Claude Code retried it ten times. Capped: an error
#: envelope is small, and only failing bodies are logged — never a healthy answer.
_ERR_BODY_MAX = 400
_UNPARSED = object()


def _as_json(body: bytes, content_type: str | None) -> dict | None:
    """The request body as a JSON object, or None (not JSON, or not an object)."""
    if not body or "json" not in (content_type or "").lower():
        return None
    try:
        parsed = json.loads(body)
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _is_json_type(content_type: str | None) -> bool:
    """THE CSRF RULE of every LB write route: the media type is `application/json`, exactly (parameters such as
    `charset` aside). A browser sends a "simple" POST — no type, a form, or text/plain WITH ANY PARAMETERS — to any
    origin without a preflight, so a substring test ("json" anywhere) let `text/plain; x=json` through unasked.
    application/json makes the browser ask first, and this gateway grants no cross-origin access."""
    return (content_type or "").split(";", 1)[0].strip().lower() == "application/json"


def _or_default(value) -> str:
    """An engine knob as the switch's log line says it: its number, or `default` for null."""
    return "default" if value is None else str(value)


def _read_switch(h: _Handler) -> dict | None:
    """An operator's switch as the page posts it, a JSON object with a boolean `enabled`; or None once it
    has been refused with a 400. One rule and one wording for every switch the page has: the cluster's
    (`/lb/cluster`, `/lb/clusters/<name>`)."""
    ctype = h.headers.get("Content-Type")
    body = _as_json(h.read_body(), ctype) if _is_json_type(ctype) else None
    if not isinstance(body, dict) or not isinstance(body.get("enabled"), bool):
        h.send_json(400, {"error": {"type": "invalid_request",
                                    "message": "a JSON body with a boolean `enabled` is required"}})
        return None
    return body


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    proxy: "Proxy"


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "vllm-lb/1.0"

    #: The request's stored version is the ONE gate on whether `send_response`, `send_header` and
    #: `end_headers` write anything at all: at HTTP/0.9 they write nothing, so the gateway page, a 400, a
    #: 501 and /lb/status all went out with no status line and none of OWN_HEADERS. The stdlib stores
    #: HTTP/0.9 both for a request line that omits the version (`default_request_version`) and for one that
    #: says `HTTP/0.9` outright — two audit rounds each fixed only one of those, so the fix belongs where
    #: the version is stored and covers every writer at once (worklog/vllm-gateway-deep-audit.md, G5).
    #: The reply's own version is `protocol_version`, so nothing else about the exchange changes.
    default_request_version = "HTTP/1.1"
    _request_version = "HTTP/1.1"
    #: What the client actually PUT ON THE WIRE, before the normalisation above. Only one decision needs it:
    #: whether a `100 Continue` may be written back (see `handle_expect_100`).
    _wire_version = "HTTP/1.1"

    @property
    def request_version(self) -> str:
        return self._request_version

    @request_version.setter
    def request_version(self, version: str) -> None:
        self._wire_version = version
        self._request_version = "HTTP/1.1" if version == "HTTP/0.9" else version

    #: How many empty lines may precede a request line. RFC 7230 §3.5: a server SHOULD ignore at least one —
    #: they are the remains of a previous message's terminator, and a client that sends them is not at fault.
    _MAX_LEADING_BLANKS = 2

    def handle_one_request(self) -> None:
        """Skip blank lines before the request line, then hand over to the stdlib.

        `http.server` reads the blank AS the request line; `parse_request` returns False on a zero-word line
        and writes NOTHING back — 0 bytes, no status line, which is not a reply at all
        (worklog/vllm-gateway-deep-audit.md, fourth audit). Peeking rather than reading a line keeps a real
        request line untouched in the buffer for the stdlib to parse exactly as before."""
        for _ in range(self._MAX_LEADING_BLANKS):
            try:
                ahead = self.rfile.peek(2)[:2]
            except (AttributeError, OSError, ValueError):
                break
            if ahead.startswith(b"\r\n"):
                self.rfile.read(2)
            elif ahead.startswith(b"\n"):
                self.rfile.read(1)
            elif ahead == b"\r":
                # Only the CR is buffered, because the CRLF was split across TCP segments and `peek` does
                # not block for the rest. Take the CR, then look again — `peek` waits for the next segment.
                # Without this the blank line goes unrecognised and the reply is 0 bytes; the test that was
                # supposed to cover it passed because its two `sendall`s coalesced into one segment, so the
                # split never happened (worklog/vllm-gateway-deep-audit.md, sixth audit).
                self.rfile.read(1)
                try:
                    if self.rfile.peek(1)[:1] != b"\n":
                        break
                except (AttributeError, OSError, ValueError):
                    break
                self.rfile.read(1)
            else:
                break
        super().handle_one_request()

    def handle_expect_100(self) -> bool:
        """Never write `100 Continue` to a client that spoke HTTP/0.9 — it cannot parse one.

        The stdlib gates this on `request_version >= "HTTP/1.1"`, which the normalisation above makes true
        for both HTTP/0.9 request lines. Returning True carries on and reads the body, writing nothing: the
        client gets its real answer instead of an interim it cannot read.

        HTTP/0.9 has TWO spellings and this gate must know both — a line that SAYS `HTTP/0.9`, which the
        setter keeps in `_wire_version`, and a line with NO version, which the stdlib silently replaces
        with `default_request_version` so that only the request line itself still remembers. Gating on one
        spelling is the mistake that cost audit rounds 2 and 3; it was then made a fourth time here, and
        the version-less half was caught by the fifth audit — after a test that looked green proved vacuous
        (a version-less line may only be GET, so its `POST` never reached this method at all)."""
        if self._wire_version == "HTTP/0.9" or len(self.requestline.split()) < 3:
            return True
        return super().handle_expect_100()

    def log_message(self, fmt, *args):  # noqa: D102 — one structured line per request instead
        log.debug("%s " + fmt, self.client_address[0], *args)

    def do_GET(self):     # noqa: N802
        self._dispatch()

    def do_POST(self):    # noqa: N802
        self._dispatch()

    def do_PUT(self):     # noqa: N802
        self._dispatch()

    def do_PATCH(self):   # noqa: N802
        self._dispatch()

    def do_DELETE(self):  # noqa: N802
        self._dispatch()

    def do_HEAD(self):    # noqa: N802
        self._dispatch()

    def do_OPTIONS(self):  # noqa: N802
        self._dispatch()

    def _dispatch(self) -> None:
        try:
            self.server.proxy.handle(self)
        except Exception:                      # the listener must survive every request
            log.exception("[proxy] unhandled error serving %s %s", self.command, self.path)
            try:
                self.send_json(500, {"error": {"type": "proxy_error", "message": "internal"}})
            except Exception:
                pass
        finally:
            self.close_connection = True

    # ── helpers ────────────────────────────────────────────────────────────

    def read_body(self) -> bytes:
        te = (self.headers.get("Transfer-Encoding") or "").lower()
        if "chunked" in te:
            out = bytearray()
            while True:
                line = self.rfile.readline().strip()
                size = int(line.split(b";")[0] or b"0", 16)
                if size == 0:
                    while self.rfile.readline().strip():   # trailers
                        pass
                    return bytes(out)
                out += self.rfile.read(size)
                self.rfile.readline()                       # CRLF after the chunk
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n > 0 else b""

    #: True only while a backend's answer is being relayed: its headers are the backend's, not ours.
    relaying = False
    #: THE STATUS THE CLIENT GOT, whoever wrote it — a relayed answer, `send_own`, the stdlib's `send_error` —
    #: recorded where every status line is written (`send_response`), and the LB's own error `type` when the LB
    #: wrote the refusal (`send_json`, `Proxy._refuse`). The REQUESTS card's outcome reads both.
    sent_status: int | None = None
    own_error: str | None = None

    def send_response(self, code, message=None) -> None:
        self.sent_status = code
        super().send_response(code, message)

    def end_headers(self) -> None:
        """EVERY response this process writes passes through here — `send_own`, the streamed web_search
        answer, Python's own `send_error` page, whatever is added next — so OWN_HEADERS are added in
        exactly one place. A helper-by-helper fix missed the SSE writer (worklog/vllm-gateway-deep-audit.md,
        G5). Only a relayed backend answer keeps the backend's headers as sent."""
        if not self.relaying:
            for name, value in OWN_HEADERS:
                self.send_header(name, value)
        super().end_headers()

    def send_own(self, status: int, payload: bytes, content_type: str,
                 headers: tuple[tuple[str, str], ...] = ()) -> None:
        """A response the LB writes ITSELF — the page, an asset, a JSON answer — whole. OWN_HEADERS come
        from `end_headers`, like every other response the LB writes."""
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        for name, value in headers:
            self.send_header(name, value)
        self.send_header("Connection", "close")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(payload)
        self.wfile.flush()

    def send_json(self, status: int, obj: dict) -> None:
        error = obj.get("error") if isinstance(obj, dict) else None
        if isinstance(error, dict) and isinstance(error.get("type"), str):
            self.own_error = error["type"]
        self.send_own(status, json.dumps(obj).encode(), "application/json")


def effective_engine(view: dict, catalog) -> str | None:
    """The ENGINE a cluster view's EFFECTIVE launch runs (its catalog profile's `engine`) — None while nothing is
    launched or the catalog does not know the choice: no engine is named that the view does not say."""
    choice = ((view or {}).get("effective") or {}).get("model_choice")
    spec = catalog.get(choice) if choice and catalog is not None else None
    profile = spec.profile(CLUSTER_MODE) if spec is not None else None
    return profile.engine if profile is not None else None


def engine_of_backend(snap, backend: Backend, catalog) -> str | None:
    """Which engine answers at `backend`: a cluster head's, from its view's effective launch; None for a classic
    server (broker_parrot's vLLM, which every launch schedules by priority) or a head with nothing launched."""
    if backend.kind != CLUSTER:
        return None
    return effective_engine(snap.cluster_view_of(backend), catalog)


def _common(cards: list[tuple]) -> float | None:
    """The ONE peak every card in `cards` ((box, peak, why), `Proxy._card_peaks`) states — None when any states none
    or they differ: the pair's verdict reads one ceiling, and two different ones are not one."""
    peaks = {peak for _, peak, _ in cards}
    one = next(iter(peaks)) if len(peaks) == 1 else None
    return one if isinstance(one, (int, float)) and not isinstance(one, bool) and one > 0 else None


#: MEMORY BANDWIDTH for an engine whose bytes the LB models (catalog `ENGINE_MODELED_PERF`) on a model the catalog
#: records no `traffic` for (`StepTraffic` / `RoutedStepTraffic`): the bytes one forward moves are not modeled (the
#: TensorFold MLX row, whose Python engine counts no pass to price one with), so there is no figure — and the server is
#: not the one withholding it.
NO_STEP_TRAFFIC = "no per-pass byte model (StepTraffic) is recorded for this model"

class Proxy:
    #: Choosing the CLASSIC model needs broker_parrot to launch from a named artifact (M7, pass 4's
    #: PHASE C). Until that engine change ships the page shows which model classic vLLM serves, but must
    #: not offer a control that would do nothing.
    CLASSIC_NOT_SELECTABLE = ("broker_parrot chooses the model classic vLLM serves; choosing it here "
                              "needs its engine to launch from a named artifact")

    def __init__(self, cfg: Config, hub: Hub, searcher=None, hw=None, cluster_switch=None,
                 catalog: ModelCatalog | None = None, perf_recorder=None, run=None) -> None:
        self.cfg = cfg
        self.hub = hub
        #: THE RUN (vllm_lb/cluster_run.py): which declared clusters run now — their switches by name, activation and
        #: deactivation, and the one answer to "is this box taken". None in tests that build the proxy alone.
        self.run = run
        #: THE ModelCatalog behind both pickers. The switch validates a ModelChoice against this same
        #: instance, so the page can never be offered a model the switch would refuse.
        self.catalog = catalog if catalog is not None else (
            cluster_switch.catalog if cluster_switch is not None else ModelCatalog.load())
        #: The LAN hardware listener (`hw.HwCollector`), or None when it never started. The
        #: proxy only ever READS it, so a collector that dies costs tiles, not the gateway.
        self.hw = hw
        #: `query -> websearch.SearchOutcome`. Injected in tests; SearXNG in production.
        self.search = searcher or (lambda q: websearch.searxng_search(q, cfg))
        #: A few minutes of SM fill (the pair's and each box's) / GB/s / TFLOP/s, so the panel shows a
        #: trend not an instant.
        #: Each server's saturation history, by its fleet name (`_saturation`): one engine's rates never another's.
        self._sat_histories: dict[str, History] = {}
        #: The cluster switch (`cluster.DesiredStore`): the one thing POST /lb/cluster writes. The
        #: cluster loop reads it on its own tick. None when the cluster is not configured.
        self.cluster_switch = cluster_switch
        #: Saved benchmark runs (`bench.BenchStore`), read from the state volume for the page at
        #: /benchmarks. READ-ONLY: the LB never writes a result — the harness on the host does.
        self.bench = BenchStore(cfg.bench_dir)
        #: THE SWARM CONTROL (`benchswarm.BenchSwarm`): the one request the page may leave for the
        #: host runner — ONE deep-research session against the model already loaded — and that
        #: runner's report read back. It writes one small file in the LB's OWN state directory and
        #: starts nothing; `loaded` is how it knows whether there is a model to put under load,
        #: because a swarm never loads one.
        self.swarm = BenchSwarm(cfg.swarm_state, loaded=self._model_loaded)
        # ── the real-traffic performance log (perflog.py) — this block and `perf_index` only ──
        #: What the recorder wrote (read-only here), and the recorder itself for its status line.
        #: None = this LB records nothing (LB_PERF=0), which the page then says.
        #: Published vendor price lists, read ONCE: what the sessions' tokens would have cost (pricing.py).
        self.pricing = _pricing.Pricing.load()
        self.perf = PerfLog(cfg.perf_dir)
        self.boots = PerfLog(cfg.perf_dir, prefix=BOOT_PREFIX)     # the cluster boot log, same dir, same range
        self.perf_recorder = perf_recorder
        #: The names routed EXACTLY (`router.exact_models`): served as named or refused, never moved.
        self._exact_models = {t: router.exact_models(cfg, self.catalog, t) for t in (router.CHAT, router.TRANSCRIPTION)}
        self._server: _Server | None = None
        #: THE HOST ALLOWLIST (host_guard.py), built by `start` once the listener's port is known: the loopback names
        #: of the address it is bound to, plus cfg.allowed_hosts (LB_ALLOWED_HOSTS). Every request passes it first.
        self.host_guard: HostGuard | None = None
        self._stop_lock = threading.Lock()     # stop(): one taker of the server, however many callers
        self._thread: threading.Thread | None = None
        self.port: int | None = None
        #: THE DRAIN's books: every admitted inference request (`_Turn`), and the two flags, under ONE lock —
        #: so a turn is either refused or waited for, never neither.
        self._turns: set[_Turn] = set()
        self._turns_cv = threading.Condition()
        self._draining = False
        self._drain_done = False
        #: THE REQUESTS CARD's finished rows (request_queue.finished_row), oldest first, under `_turns_cv`.
        self._recent: collections.deque = collections.deque(maxlen=request_queue.RECENT_ROWS)
        #: THE TURNS THAT ENDED in the last EVENTS_KEPT_S, as (monotonic end, backend + tap) under `_turns_cv`: a turn
        #: that finished inside the engine's GENERATED window is part of its count, so it keeps its share of it
        #: (request_queue `ended_tok_s`) — only its backend and tap are kept, never the handler.
        self._ended: collections.deque = collections.deque(maxlen=512)

    # ── lifecycle ──────────────────────────────────────────────────────────

    def start(self) -> None:
        self._server = _Server((self.cfg.listen_host, self.cfg.listen_port), _Handler)
        self._server.proxy = self
        self.port = self._server.server_address[1]
        self.host_guard = HostGuard("LB_ALLOWED_HOSTS", self.cfg.allowed_hosts, self.cfg.listen_host, self.port,
                                    log.warning)
        log.info("[proxy] answering to Host %s", ", ".join(str(a) for a in self.host_guard.allowed))
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        name="vllm-lb-http", daemon=True)
        self._thread.start()
        log.info("[proxy] listening on %s:%s", self.cfg.listen_host, self.port)

    def stop(self) -> None:
        """Stop listening. Idempotent, and safe from two threads at once: drain() calls it from its own
        thread while SIGTERM's path or a test teardown may call it too. It used to check `_server` and
        then use it three times, so a second caller finishing in between left the first calling
        `server_close()` on None (tests/test_proxy_stop.py). The server is taken out under a lock and
        only its taker shuts it down."""
        with self._stop_lock:
            server, self._server = self._server, None
        if server is not None:
            server.shutdown()
            server.server_close()

    def drain(self, window_s: float) -> int:
        """SIGTERM's answer (`__main__`): let the turns in flight finish, up to `window_s`, then stop listening.

        Until 2026-10-03 a stop closed the listener and returned, and every relay thread — each one a
        client's stream — died with the process: 2-7 Claude Code sessions at a time lost their stream 1-2 s
        after each redeploy. Now a turn in flight streams to its end; a turn that ARRIVES meanwhile is held,
        no byte sent, and answered 503 `draining` once the drain is over — one retry onto the new container,
        `drain_retry_after_s` later, instead of the client's whole retry budget spent against this one. A turn
        held for the cluster stops waiting for it at once (this process will not serve the pair) and is
        answered the same way. Returns how many turns were still in flight when the window ran out — they end
        with the process."""
        t0 = time.monotonic()
        deadline = t0 + max(0.0, window_s)
        with self._turns_cv:
            self._draining = True
            busy = self._in_flight()
        self.hub.notify_change()                  # a turn held for the cluster: stop waiting for the pair
        if busy:
            log.info("[proxy] draining: waiting up to %.0f s for %d request(s) in flight: %s",
                     window_s, len(busy), "; ".join(t.describe(t0) for t in busy))
        else:
            log.info("[proxy] draining: nothing in flight")
        try:
            with self._turns_cv:
                next_log = t0 + _DRAIN_LOG_EVERY_S
                while (busy := self._in_flight()) and (now := time.monotonic()) < deadline:
                    if now >= next_log:
                        log.info("[proxy] draining: %d request(s) still in flight after %.0f s of %.0f s: %s",
                                 len(busy), now - t0, window_s, "; ".join(t.describe(now) for t in busy))
                        next_log = now + _DRAIN_LOG_EVERY_S
                    self._turns_cv.wait(min(deadline, next_log) - now)
                cut = self._in_flight()
                held = sum(1 for t in self._turns if t.holding)
        finally:
            with self._turns_cv:
                self._drain_done = True
                self._turns_cv.notify_all()
        # Nothing may arrive after the last `draining` answer: stop listening, then answer the held.
        self.stop()
        with self._turns_cv:
            answer_by = time.monotonic() + _DRAIN_ANSWER_S
            while any(t.holding for t in self._turns) and (left := answer_by - time.monotonic()) > 0:
                self._turns_cv.wait(left)
        now = time.monotonic()
        if cut:
            log.warning("[proxy] drain window (%.0f s) is over with %d request(s) still in flight — they end with "
                        "this process: %s", window_s, len(cut), "; ".join(t.describe(now) for t in cut))
        else:
            log.info("[proxy] drained in %.1f s; %d request(s) that arrived meanwhile were answered 503 "
                     "draining (Retry-After %s s)", now - t0, held,
                     retry_after_header(self.cfg.drain_retry_after_s))
        return len(cut)

    # ── the drain's books ──────────────────────────────────────────────────

    def _in_flight(self) -> list[_Turn]:
        """Turns with a backend or on their way to one — what a drain waits for. Under `_turns_cv`."""
        return sorted((t for t in self._turns if not t.holding), key=lambda t: t.started)

    def _admit(self, h: _Handler, body: dict | None = None) -> _Turn | None:
        """Book an inference request. During a drain it is booked as HELD — it will be answered `draining` once
        the drain is over — unless `hold_max` are already waiting: then it is refused now (None). `body` is the
        request as `handle` parsed it: the REQUESTS card's facts are read off it once, here."""
        with self._turns_cv:
            if not self._draining or sum(1 for t in self._turns if t.holding) < self.cfg.hold_max:
                turn = _Turn(h, holding=self._draining, body=body)
                self._turns.add(turn)
                return turn
        self._refuse(h, "draining", f"the gateway is restarting and {self.cfg.hold_max} requests are already "
                                    f"waiting for it; this one was not sent to any backend", self.cfg.retry_after_s)
        return None

    def _retire(self, turn: _Turn, h: _Handler | None = None) -> None:
        """Unbook a turn, and keep it as a finished row: the status the client got (`h.sent_status`) and, when the
        LB wrote the answer itself, its error type."""
        if h is not None:
            turn.status = h.sent_status
            if h.own_error and not turn.error and (h.sent_status or 0) >= 400:
                turn.error = h.own_error
        try:
            done = request_queue.finished_row(turn, time.monotonic(), time.time())
        except Exception:                      # a view is never allowed to break a turn
            log.exception("[proxy] the finished row for turn %s could not be built", turn.id)
            done = None
        gone_at = time.monotonic()
        with self._turns_cv:
            self._turns.discard(turn)
            if done is not None:
                self._recent.append(done)
            if turn.tap is not None and turn.backend not in (None, request_queue.WEBSEARCH):
                self._ended.append((gone_at, types.SimpleNamespace(backend=turn.backend, tap=turn.tap)))
            while self._ended and self._ended[0][0] < gone_at - request_queue.EVENTS_KEPT_S:
                self._ended.popleft()
            self._turns_cv.notify_all()

    def queue_view(self, snap=None) -> dict:
        """/lb/status `queue` (request_queue.queue_view): the turns in flight, the last finished, and the engine's
        own aggregates beside them."""
        snap = snap if snap is not None else self.hub.snapshot()
        backends = snap.backends
        # WHY ITS QUEUED WAIT (request_queue.admission): each live server's own figures off its last scrape, its
        # window, and its seats where this LB knows them — the cluster's APPLIED max_num_seqs (`effective`, what the
        # engine runs), each cluster head's OWN cluster's (`Snapshot.cluster_view_of`); a classic vLLM publishes none,
        # so its seat rule is not applied.
        engines = {}
        for b in backends:
            gauges = self.hub.engine_gauges(b.url, request_queue.ADMISSION_KEYS) if b.live else None
            if gauges is not None:
                engines[b.url] = {**gauges, "max_model_len": b.max_model_len,
                                  "seats": ((snap.cluster_view_of(b).get("effective") or {}).get("max_num_seqs")
                                            if b.kind == CLUSTER else None)}
        # THE DECODE SHARES (request_queue DERIVED_DECODE): each live server's GENERATED and the window it ends at.
        rates = {b.url: self.hub.generated_window(b.url) for b in backends if b.live}
        now = time.monotonic()
        with self._turns_cv:
            turns, recent = list(self._turns), list(self._recent)
            ended = [t for at, t in self._ended if at >= now - request_queue.EVENTS_KEPT_S]
        return request_queue.queue_view(turns, recent, backends, now, engines, rates=rates, ended=ended)

    def _wait_out_drain(self, h: _Handler, turn: _Turn) -> None:
        """A turn this process will not serve: hold it — no byte sent, so the client just waits — until the
        drain is over, then answer 503 `draining`. `drain()` always ends it, whatever happens to the window; the
        operator's kill ends it at once (`kill_request` wakes `_turns_cv`)."""
        with self._turns_cv:
            while not self._drain_done and turn.killed is None:
                self._turns_cv.wait()
        if turn.killed is not None:
            return self._answer_killed(h, turn)
        self._refuse(h, "draining", "the gateway is restarting; this request was not sent to any backend — the "
                                    "new process takes it in a moment", self.cfg.drain_retry_after_s)

    def _refuse(self, h: _Handler, kind: str, message: str, retry_after_s: float, *, retry: bool = True) -> None:
        """Every 503 this proxy writes: the JSON error, a `Retry-After` the client will honour, and — when nothing
        is on its way — `x-should-retry: false`, so the client shows the error at once instead of retrying ten
        times into the same answer."""
        headers = [("Retry-After", retry_after_header(retry_after_s))]
        if not retry:
            headers.append(("x-should-retry", "false"))
        h.own_error = kind
        h.send_own(503, json.dumps({"error": {"type": kind, "message": message}}).encode(), "application/json",
                   tuple(headers))

    def _answer_killed(self, h: _Handler, turn: _Turn) -> None:
        """A turn the operator killed BEFORE any byte of its answer reached its client — held, or sent and not
        answered yet: its API's own error, with a status no client retries and `x-should-retry: false`
        (request_kill.py)."""
        turn.kill_answered = True
        h.own_error = request_queue.KILLED
        try:
            h.send_own(request_kill.KILL_STATUS, request_kill.error_body(turn.path, turn.id), "application/json",
                       (request_kill.NO_RETRY,))
        except OSError:
            turn.client_gone = True            # it had hung up too: nothing left to tell

    @staticmethod
    def _killed_mid_stream(h: _Handler, turn: _Turn, *, sse: bool, sent: int, tail: bytes) -> None:
        """A turn killed while its answer was being relayed — its status line is the client's already, so the kill is
        said IN the stream, at an event boundary: an Anthropic `event: error`, or an OpenAI error chunk and then
        `[DONE]` (request_kill.stream_error) — never a silent stop and never a fake normal finish. Inside an event
        nothing may be added (a byte there corrupts it), and a body that is not SSE has no place for words: those are
        cut, as a backend failure is."""
        turn.kill_answered = True
        if sse and sent and _at_boundary(tail):
            try:
                h.wfile.write(request_kill.stream_error(turn.path, turn.id))
                h.wfile.flush()
            except OSError:
                turn.client_gone = True

    # ── the LB's own view ──────────────────────────────────────────────────

    @staticmethod
    def _held_backend(lease, backends) -> str | None:
        """The URL of the server a lease holds, by the router's rule (`router.choose`/held): a vLLM
        lease by the label that claimed it."""
        for b in backends:
            if b.kind == VLLM and b.host_label and b.host_label == lease.host_label:
                return b.url
        return None

    def _cluster_engine(self, cluster: dict) -> str:
        """Which server the pair is running, for the words the page puts beside the engine's own
        figures. The profile of whatever the cluster EFFECTIVELY launched, not what is stored — a
        selection that has not started yet describes nothing. Falls back to vLLM, which is what every
        profile was before 2026-10-03 and what the page said unconditionally until then."""
        # catalog.VLLM, not state.VLLM: the two spell the same string today and mean different things
        # — a backend KIND against an ENGINE — and this is the engine.
        return effective_engine(cluster, self.catalog) or VLLM_ENGINE

    def _cluster_traffic(self, cluster: dict):
        """The catalog's `traffic` (`StepTraffic`, `RoutedStepTraffic`) for what the pair EFFECTIVELY launched, at the
        KV dtype it launched with (`ModelSpec.at_kv_cache_dtype`), when the LB models its memory bandwidth — an SGLang
        or TensorFold profile that records one. None for every vLLM generation, whose figures stay the engine's own
        estimate, and for a cluster with nothing launched."""
        effective = cluster.get("effective") or {}
        choice = effective.get("model_choice")
        spec = self.catalog.get(choice) if choice else None
        if spec is not None:
            spec = spec.at_kv_cache_dtype(effective.get("kv_cache_dtype"))
        profile = spec.profile(CLUSTER_MODE) if spec is not None else None
        return spec.traffic if profile is not None and profile.engine in ENGINE_MODELED_PERF else None

    def _lb_models_perf(self, cluster: dict) -> bool:
        """Whether the LB models this generation's COMPUTE from the engine's pass counters (catalog
        `ENGINE_MODELED_PERF`) rather than drawing the engine's own estimate."""
        return effective_engine(cluster, self.catalog) in ENGINE_MODELED_PERF

    def _saturation(self, snap, name: str) -> dict | None:
        """IS THERE HEADROOM — the one question a 96 % GPU tile could not answer (saturation.py).

        Built from what is already read every tick: each GB10 box's GPU off `tp-rank status`, and the
        cluster engine's own FLOP/s and byte/s off the same `/metrics` scrape the router uses. No poll
        of its own, no SSH call of its own, nothing installed anywhere.

        SERVER `name`'s (operator 2026-10-20, "Fix all"): its own config (`hub.cluster_cfgs[name]` — its ranks, its
        serving cards, its control), its own view and head, its own history. None for a server not in the run or not
        configured — there is then nothing whose saturation this would be."""
        cfg = self.hub.cluster_cfgs.get(name)
        cluster = snap.cluster_views().get(name) or {}
        if cfg is None or not cluster.get("configured"):
            return None
        boxes = cluster.get("boxes") or {}
        # RANK order is the configured host order — rank 0, the head that serves the API, first — never
        # the alphabet: the page draws one SM-fill meter per rank, and "which rank" is the point.
        order = [host.box for host in cfg.cluster_hosts]
        ranks = [{"box": box, "rank": i, "role": RANK_ROLES[min(i, len(RANK_ROLES) - 1)]}
                 for i, box in enumerate(order)]
        if cfg.cluster_control == topology.NONE:
            gpus = self._observed_gpus(cfg, order, snap.now)
        else:
            gpus = [dict(boxes[box]["gpu"], box=box)
                    for box in order + sorted(b for b in boxes if b not in order)
                    if isinstance(boxes.get(box), dict) and isinstance(boxes[box].get("gpu"), dict)]
        # THE CEILINGS ARE THE SERVING CARDS' (the fleet names them, their hardware types state the peaks): per card,
        # head first, the dense peak at the precision the MODEL's GEMMs run in (cores.py) and the card's bus.
        compute_cards, bus_cards, precision = self._card_peaks(cfg, order, cluster)
        peak_tflops, peak_gbps = _common(compute_cards), _common(bus_cards)
        head = next((b for b in snap.backends if b.kind == CLUSTER and b.cluster == name), None)
        traffic = self._cluster_traffic(cluster)
        measured = self.hub.throughput(head, snap.now, traffic=traffic) if head else {}
        perf = measured.get("engine_perf") or {}
        # AN SGLANG OR TENSORFOLD GENERATION'S GB/s IS THE LB'S MODEL (throughput.py `bandwidth_model`): SGLang's own
        # byte counter is withheld at the probe and TensorFold publishes none, so the model stands in for it, reason
        # and all. vLLM: unchanged.
        model = measured.get("bandwidth_model")
        modeled = self._lb_models_perf(cluster)
        gbps, bytes_why = ((model["gbps_per_gpu"], model["why"]) if model is not None
                           # an engine whose bytes the LB models, on a model the catalog has no per-pass bytes for
                           else (None, NO_STEP_TRAFFIC) if modeled and traffic is None
                           else (perf.get("gbps_per_gpu"), perf.get("bytes_why")))
        # …AND ITS FLOP/s IS THE LB'S MODEL TOO (cores.py `modeled_flops`), over the CUDA CORES tile's own window,
        # which prices its cores with the same figure: SGLang's counter charges the 27B's 64 layers as dense
        # attention with an unsharded MLP and read 236.8 TFLOP/s beside 742 cores multiplying. It stays read
        # (`engine_perf`), and the model carries it as `engine_tflops` for the hover. vLLM: unchanged.
        flops = self._modeled_flops(cfg, head, cluster, traffic) if modeled and head is not None else None
        tflops, flops_why = ((flops["tflops_per_gpu"], flops["why"]) if flops is not None
                             else (perf.get("tflops_per_gpu"), perf.get("flops_why")))
        out = judge(gpus=gpus,
                     tflops_per_gpu=tflops,
                     gbps_per_gpu=gbps,
                     # The pair's verdict reads ONE ceiling: the cards' own, where every serving card states the
                     # same one; the per-box figures below carry each box's own.
                     peak_tflops=peak_tflops,
                     peak_tflops_precision=precision or "",
                     peak_gbps=peak_gbps,
                     flops_why=flops_why, bytes_why=bytes_why, bytes_model=model, flops_model=flops,
                     # The modeled window is never widened: it is the passes the engine counted, over their time.
                     window_s=flops["window_s"] if flops is not None else perf.get("window_s"),
                     window_widened=None if flops is not None else perf.get("widened"),
                     # The serving cards (TP); the rank count where no fleet names them.
                     gpu_count=len(compute_cards) if cfg.fleet is not None else max(1, len(boxes)),
                     # The rates are the HEAD's: only rank 0 serves /metrics (rank 1 is --headless).
                     ranks=ranks, rates_from=order[0] if order else None,
                     # WHOSE estimate it is. The sentence beside these rates named vLLM unconditionally,
                     # which was true of every generation until the pair first served SGLang.
                     engine=self._cluster_engine(cluster))
        # PER BOX USED (operator, 2026-10-11): each serving box's rate against ITS OWN cards' peak — the engine-wide
        # rate split by the TP layout, DERIVED and labelled so (saturation.per_box), adding back up to it.
        for key, cards in (("compute", compute_cards), ("bandwidth", bus_cards)):
            c = out[key]
            withheld = c.get("why") if c.get("pct") is None and c.get("peak") is not None and c.get("rate") is not None \
                else None
            c["boxes"] = per_box(c.get("rate"), cards, withheld=withheld)
        # THE TP LINK: each box's RDMA rate (cluster_loop._net_rate). The meter is the busiest direction
        # on the busiest box against one direction of the link — the all-reduce is symmetric, so tx ≈ rx.
        # One rank has no link: the tile says so (`topology.NO_LINK`) rather than wait for a reading forever.
        nets = {box: (boxes.get(box) or {}).get("network") for box in order}
        rates = [max(n["tx_gbps"], n["rx_gbps"]) for n in nets.values() if isinstance(n, dict)]
        peak = next((n["link_gbps"] for n in nets.values() if isinstance(n, dict) and n.get("link_gbps")), None)
        rate = max(rates) if rates else None
        out["network"] = ({"rate": None, "peak": None, "pct": None, "why": topology.NO_LINK, "boxes": {}}
                          if len(order) < 2 else
                          {"rate": rate, "peak": peak,
                           "pct": None if rate is None or not peak else round(rate / peak * 100, 1),
                           "why": None if rates else "awaiting 2nd RDMA reading",
                           "boxes": nets})
        # THE CORES: what one GPU's cores were doing over the last probe interval (cores.estimate).
        out["cores"] = self._cores(cfg, head, cluster, traffic=traffic, model=model, flops=flops)
        history = self._sat_histories.get(name) or self._sat_histories.setdefault(
            name, History(keep=self.cfg.stat_history))
        out["history"] = history.observe(
            time.time(), out.get("sm_fill_pct"),
            (out.get("bandwidth") or {}).get("rate"), (out.get("compute") or {}).get("rate"),
            boxes={b["box"]: b.get("sm_pct") for b in out["boxes"] if b.get("box")}, net=rate)
        return out

    def _observed_gpus(self, cfg: Config, order: list[str], now: float) -> list[dict]:
        """A node this LB only observes: its card's SM fill as `gpu_view` readings. The card is the one it RUNS ON, as
        the hardware broadcast read it, asked through the ONE adapter (`topology.serving_card`). No answer is no
        reading — a gap with its reason, never the box's busiest card. `nvidia-smi`'s utilization.gpu is the same
        quantity tp-rank reads as SM fill: the share of the sample a kernel was resident."""
        hardware = self.hw.snapshot(now) if self.hw else {}
        gpus = []
        for box in order:
            card = topology.serving_card(cfg, hardware, box)
            reading = gpu_view({"sm_pct": card.get("use_pct")}) if isinstance(card, dict) else None
            if reading is not None:
                gpus.append(dict(reading, box=box))
        return gpus

    def _card_peaks(self, cfg: Config, order: list[str], cluster: dict) -> tuple[list[tuple], list[tuple], str | None]:
        """Each serving card's two ceilings, HEAD FIRST, as `saturation.per_box` takes them — (box, peak, why) for
        compute and for memory bandwidth — and the precision the compute peak is at. A card's peaks are its hardware
        type's (vllm_lb/hardware/types, the fleet names the card); compute's is at the precision the cluster's MODEL
        runs in (cores.py `precision_of`), so without a model there is no compute peak, and it says so. Where no fleet
        names the cards, each rank's box is one card with no peak and the reason."""
        cards = topology.serving_cards(cfg)
        # WHAT THE CLUSTER SERVES: the pair's Start says it (its launch record); a node this LB only observes is
        # named by the fleet file (`topology.cluster_model`) — one source per topology, never one standing in.
        if cfg.cluster_control == topology.NONE:
            named = topology.cluster_model(cfg)
            model = named.spec.id if named is not None else None
        else:
            model = (cluster.get("effective") or {}).get("model_choice")
        precision = cores.precision_of(model)
        if not cards:
            why = "the fleet file names no serving card for this box"
            return [(box, None, why) for box in order], [(box, None, why) for box in order], precision
        no_precision = (f"no compute precision: {model} has no shape recorded (cores.py)" if model
                        else "no compute precision: no model is launched on the cluster")
        compute = []
        for c in cards:
            fact = c.type.peak_tflops_for(precision) if precision else None
            compute.append((c.box, fact.value if fact else None,
                            (None if fact and fact.known else (fact.source if fact else no_precision))))
        bus = [(c.box, c.type.peak_gbps.value, None if c.type.peak_gbps.known else c.type.peak_gbps.source)
               for c in cards]
        return compute, bus, precision

    #: Probes the cores estimate averages over (~1 min at the 6 s probe interval).
    CORES_WINDOW = 10

    def _modeled_flops(self, cfg: Config, head, cluster: dict, traffic) -> dict | None:
        """The LB's FLOP model of the pair over the CUDA CORES tile's window (`CORES_WINDOW` probe intervals):
        COMPUTE's figure, and the cores tile's (cores.py `modeled_flops`). Every read the hub keeps is the
        history its busy-time fit falls back to. None for a model with no shape."""
        hist = self.hub.history(head.url)
        reads = self.hub.engine_reads(head.url)
        try:
            compute_cards, bus_cards, _ = self._card_peaks(cfg, [h.box for h in cfg.cluster_hosts], cluster)
            return cores.modeled_flops((cluster.get("effective") or {}).get("model_choice"),
                                       reads[-(self.CORES_WINDOW + 1):], history=reads, traffic=traffic,
                                       peak_tflops=_common(compute_cards), peak_gbps=_common(bus_cards),
                                       fit=cores.fit_step_time(zip(hist.get("steps_s") or [],
                                                                   hist.get("tokens_step") or [])))
        except Exception:                      # a model is never allowed to break /lb/status
            return None

    def _cores(self, cfg: Config, head, cluster: dict, traffic=None, model: dict | None = None,
               flops: dict | None = None) -> dict | None:
        """`traffic`, `model`: an SGLang generation's StepTraffic and modeled bandwidth (`_saturation`). With
        them the tile prices a decode pass with the SAME bytes the MEMORY BANDWIDTH tile draws, at the context
        the engine itself sums (`decode_sum_seq_lens` / running) and the width it verifies. `flops`: its modeled
        FLOPs (`_modeled_flops`) — the tile is then drawn from that window, so it and COMPUTE are one figure."""
        if head is None:
            return None
        hist = self.hub.history(head.url)
        # OVER THE LAST MINUTE, not the last probe: one 6 s interval often holds no prefill step at all,
        # and the prefill grid then blinked in and out with every poll. Steps are summed across the
        # window, so tokens a step is the window's own mean.
        pairs = [(s_, t_) for s_, t_ in zip(hist.get("steps_s") or [], hist.get("tokens_step") or [])
                 if s_ is not None and t_ is not None][-self.CORES_WINDOW:]
        steps_s = sum(s_ for s_, _ in pairs) / len(pairs) if pairs else None
        tokens_step = (sum(s_ * t_ for s_, t_ in pairs) / sum(s_ for s_, _ in pairs)) if pairs else None
        eff = cluster.get("effective") or {}
        compute_cards, bus_cards, _ = self._card_peaks(cfg, [h.box for h in cfg.cluster_hosts], cluster)
        try:
            # THE CONTEXT A RUNNING REQUEST HOLDS, for a model whose attention reads all of it: the engine's
            # KV gauge counts the blocks its RUNNING requests reference, so gauge x pool / running is their
            # mean context. The pool in tokens is the launch's pin over the model's bytes per token.
            spec = self.catalog.get(eff.get("model_choice"))
            # …at the KV dtype it was launched with (`auto`: the profile's own, as measured).
            if spec is not None and eff.get("kv_cache_dtype") not in (None, "auto"):
                spec = spec.at_kv_cache_dtype(eff["kv_cache_dtype"])
            kv = next((v for v in reversed(hist.get("kv_pct") or []) if v is not None), None)
            pin, per_token = eff.get("kv_cache_memory"), (spec.memory.kv_bytes_per_token
                                                          if spec is not None and spec.memory else None)
            blocks = spec.memory.kv_blocks if spec is not None and spec.memory is not None else None
            if blocks is not None and kv is not None and pin:
                # BY BLOCKS, the fixed ones per request left out: the engine's count while it serves.
                pool = (self.hub.throughput(head, time.time()).get("kv_pool") or {}).get("blocks")
                context = cores.mean_context(kv / 100.0, running=head.running,
                                             pool_blocks=pool if isinstance(pool, int) and pool > 0
                                             else blocks.blocks_in(int(pin)), blocks=blocks)
            else:
                context = (kv / 100.0 * pin / per_token / max(1, head.running)
                           if kv is not None and pin and per_token else None)
            verify = None
            if traffic is not None and model is not None:
                # NOT the KV gauge: prefix sharing makes that the DEDUPLICATED context — 16k a request live
                # against the 46k each request's attention reads (worklog/sglang-bottleneck-2026-10-03.md).
                summed, seqs = model.get("context_tokens"), model.get("seqs")
                context = summed / seqs if summed is not None and seqs else None
                verify = model.get("verify_width")
            return cores.estimate(eff.get("model_choice"), steps_s=steps_s,
                                  tokens_step=tokens_step, running=head.running,
                                  speculative=eff.get("num_speculative_tokens"),
                                  budget=eff.get("max_num_batched_tokens"),
                                  peak_tflops=_common(compute_cards) or 0, peak_gbps=_common(bus_cards) or 0,
                                  # ONE GPU's cores: the serving cards' hardware type's (the fleet names them).
                                  cuda_cores=topology.serving_cuda_cores(cfg),
                                  fit=cores.fit_step_time(zip(hist.get("steps_s") or [],
                                                              hist.get("tokens_step") or [])),
                                  context=context, traffic=traffic, verify_width=verify, flops=flops)
        except Exception:                      # an estimate is never allowed to break /lb/status
            return None

    #: The target is read off the perf log, which grows by a line a minute: recomputing it more often
    #: than this would re-sort a week of windows on every poll of the page.
    TARGET_TTL_S = 30.0
    #: How far back a minute still counts towards the target: long enough to keep a benchmark matrix,
    #: short enough that an engine or model build from another season does not set today's bar.
    TARGET_DAYS = 30.0

    def _owners(self) -> dict:
        """{server: the model names it may run} for every declared server (router.cluster_models) — what says whose a
        perf or boot line from before lines named their server is (perflog.of_cluster)."""
        hit = getattr(self, "_owners_cache", None)
        if hit is None:
            hit = self._owners_cache = {c.name: router.cluster_models(self.cfg, self.catalog, c.name)
                                        for c in self.cfg.clusters}
        return hit

    def _windows_of(self, name: str) -> list[dict]:
        """Server `name`'s recorded minutes over the targets' range (perflog.of_cluster)."""
        return self.perf.index(days=self.TARGET_DAYS, keep=self._keep(name))["windows"]

    def _keep(self, server: str | None):
        """PerfLog.index's `keep` for server `server` (perflog.of_cluster) — None, unscoped, for no server."""
        # An LB whose config declares no fleet cluster (built by hand) has one server, the one asked about.
        owners = self._owners() or {server: frozenset()}
        return None if not server else (lambda lines: of_cluster(lines, server, owners))

    def _throughput_target(self, snap, name: str) -> dict | None:
        """SERVER `name`'s rate to aim for: the best of ITS minutes at the setup it runs now."""
        config, _ = effective_config(snap.cluster_views().get(name))
        key = (name, json.dumps(config, sort_keys=True, default=str))
        cache = self.__dict__.setdefault("_target_cache", {})
        hit = cache.get(name)
        if hit and hit[0] == key and snap.now - hit[1] < self.TARGET_TTL_S:
            return hit[2]
        try:
            target = throughput_target(self._windows_of(name), config)
        except Exception:                      # the log is never allowed to break /lb/status
            target = None
        cache[name] = (key, snap.now, target)
        return target

    def _setup_targets(self, snap, name: str) -> dict:
        """`setup_targets`: SERVER `name`'s PICKER ROWS' RATES — `throughput_target` at what a Start of each row
        launches (`cluster.setups`, keyed `<id>` / `<id>@uncensored`), on its own minutes and the tile's clock.
        A row nothing has ever run is null; a view that publishes no rows is an empty map."""
        cluster = snap.cluster_views().get(name)
        setups = cluster.get("setups") if isinstance(cluster, dict) else None
        if not isinstance(setups, dict) or not setups:
            return {}
        key = json.dumps(setups, sort_keys=True, default=str)
        cache = self.__dict__.setdefault("_setup_targets_cache", {})
        hit = cache.get(name)
        if hit and hit[0] == key and snap.now - hit[1] < self.TARGET_TTL_S:
            return hit[2]
        try:
            windows = self._windows_of(name)
            targets = {row: throughput_target(windows, setup) for row, setup in setups.items()}
        except Exception:                      # the log is never allowed to break /lb/status
            targets = {row: None for row in setups}
        cache[name] = (key, snap.now, targets)
        return targets

    def _server_figures(self, snap) -> dict:
        """EACH SERVER'S OWN FIGURES, by name: its saturation, throughput target and picker rows' rates. Computed once
        per status (the saturation history observes each call)."""
        return {n: {"saturation": self._saturation(snap, n), "throughput_target": self._throughput_target(snap, n),
                    "setup_targets": self._setup_targets(snap, n)} for n in snap.cluster_views()}

    #: The rows' start rates are read off the boot log, which grows by a line a START: re-reading every line it keeps
    #: on every poll of every open page buys nothing that a verdict half a minute old does not already say.
    START_RATES_TTL_S = 30.0

    def _start_rates(self, snap) -> dict:
        """`start_rates`: THE CHOOSER ROWS' START SUCCESS RATES — per row (`<id>` / `<id>@uncensored`), of the LB's
        launches of that checkpoint over the last 7 UTC days, how many reached serving, per day and in all
        (vllm_lb/start_rates.py, over every line the boot log keeps, through the reader /lb/perf uses). `malformed`:
        lines the reader could not parse. `error`: why the rates are NOT KNOWN — the recorder is off (LB_PERF=0), or the
        log cannot be read — and then there are no rows, since an empty map reads "no starts in 7 d" on every row. A
        boot log directory that does not exist is not an error: the recorder makes it with its first line, so nothing
        has been launched here."""
        hit = getattr(self, "_start_rates_cache", None)
        if hit and snap.now - hit[0] < self.START_RATES_TTL_S:
            return hit[1]
        out = {**start_rates((), now=snap.now), "malformed": 0, "error": ""}
        if not self.cfg.perf_enabled:
            out["error"] = "the boot log is off (LB_PERF=0): no start is recorded"
        elif self.boots.dir.exists():
            try:
                index = self.boots.index(days=None, now=snap.now)
                error = index["error"] or "; ".join(f"{f['file']}: {f['error']}" for f in index["files"] if f["error"])
                if error:
                    out["error"] = error
                else:
                    out = {**start_rates(index["windows"], now=snap.now), "malformed": index["malformed_total"],
                           "error": ""}
            except Exception as exc:           # the log is never allowed to break /lb/status — said, never hidden
                out["error"] = f"{type(exc).__name__}: {exc}"
        self._start_rates_cache = (snap.now, out)
        return out

    #: What /lb/status says ONLY with the broker on: its DB's state and what the LB asked it for. With the broker
    #: off there is none of it to report — never a db_ok that was not measured, nor a "0 leases of N wanted".
    BROKER_STATUS_KEYS = ("db_ok", "db_error", "desired_leases", "vllm_capacity", "vllm_capacity_detail",
                          "vllm_boxes_on", "vllm_boxes_detail", "context", "run_next", "gpu_slots")

    def status(self) -> dict:
        snap = self.hub.snapshot()
        out = self._status(snap)
        # THE SWITCH, stated plainly (LB_BROKER_ENABLED): the page draws the broker's chips only when it is on.
        out["broker"] = {"enabled": self.cfg.broker_enabled}
        # THE CONNECT DIALOG (connect.py): the routes only the deployment can state, and the budgets a client must
        # out-wait. Read once per dialog; never anything secret.
        out["connect"] = connect_view(self.cfg)
        if not self.cfg.broker_enabled:
            for key in self.BROKER_STATUS_KEYS:
                out.pop(key, None)
        return out

    def _lease_window(self) -> int | None:
        """The window the next lease declares (`Config.lease_context_length`), or None — with the reason kept beside it
        in `context.state` by the lease loop — when no catalog row serves the lease model."""
        try:
            return self.cfg.lease_context_length
        except ValueError:
            return None

    def _status(self, snap) -> dict:
        figures = self._server_figures(snap)
        # The top-level figures are the ONE server's alias, as `cluster` is: with several, none is "the" one.
        one = next(iter(figures.values())) if len(figures) == 1 else {}
        return {
            "now": snap.now,
            "uptime_s": round(snap.now - snap.extra.get("started_at", snap.now), 1),
            "backends": [{
                "kind": b.kind, "url": b.url, "host_label": b.host_label, "box": b.box,
                # A cluster head's fleet NAME (its view is `clusters[<it>]`); null for a classic server or the fallback.
                "cluster": b.cluster,
                "live": b.live, "served_models": list(b.served_models),
                "max_model_len": b.max_model_len, "waiting": b.waiting, "running": b.running,
                "inflight": b.inflight, "penalised": b.penalty_until > snap.now,
                # Seconds nothing MOVED (no generated, no prefilled token) while work waited; null
                # when there is no work. A long prefill is progress, not a wedge.
                "stalled_s": (round(snap.now - b.progress_at, 1)
                              if b.progress_at is not None and b.waiting + max(b.running, b.inflight) > 0
                              else None),
                "failure_streak": b.failure_streak,
                # False = /metrics did not answer, so depth is a guess and the
                # router treats this backend as busy rather than idle.
                "depth_known": b.depth_known,
                # The BOX, not the server: an absent vLLM on a live box is this
                # fleet's resting state, not a fault.
                "box_alive": b.box_alive,
                # Parked by the operator (`worker_controls.desired_state='off'`): deliberate
                # and healthy, and NOT the same thing as a box that is not answering.
                "parked": b.parked,
                # MEASURED across probes, not reported by the engine. null = not yet known.
                "tokens_per_s": b.tokens_per_s,
                "gen_tokens_total": b.gen_tokens_total,
                "prompt_tokens_total": b.prompt_tokens_total,
                "last_used_at": b.last_used_at,
                "last_error": b.last_error, "last_probe_at": b.last_probe_at,
                # WHEN running / waiting were read: the REQUESTS card ages the engine's count against its own by it.
                "depth_read_at": b.depth_read_at,
                # Oldest first. Kept HERE so a page reload does not erase every stat's past.
                "history": self.hub.history(b.url),
                # What the server generates and loads, and the model it holds — MEASURED, or null
                # with its `why` (worklog/vllm-gateway-throughput.md, Design 3).
                "throughput": self.hub.throughput(
                    b, snap.now, traffic=(self._cluster_traffic(snap.cluster_view_of(b))
                                          if b.kind == CLUSTER else None)),
                "loaded": self.hub.loaded(b, snap.now),
            } for b in snap.backends],
            "leases": [{
                "lease_id": l.lease_id, "job_id": l.job_id, "status": l.status,
                # WHICH KIND of server this node_job holds — a vLLM the fleet started for us,
                # or the fallback box we pinned. They have different planners and different
                # release clocks, and the UI must not show them as the same thing.
                "kind": l.kind,
                "host_label": l.host_label, "box": l.box, "created_at": l.created_at,
                "claimed_at": l.claimed_at, "released_at": l.released_at, "detail": l.detail,
                # The server THIS node_job holds, by the router's own rule (see _held_backend).
                "backend": self._held_backend(l, snap.backends),
            } for l in snap.leases],
            "last_request_at": snap.last_request_at,
            "last_pressure_at": snap.last_pressure_at,
            "desired_leases": desired_leases(snap, self.cfg),
            "db_ok": snap.db_ok, "db_error": snap.db_error,
            "vllm_boxes_on": snap.boxes_on,
            # What the BROKER says can serve a vLLM lease right now — the ceiling on leases,
            # asked every tick from worker_heartbeats.llm_servers_available.
            "vllm_capacity": snap.vllm_capacity,
            "vllm_capacity_detail": snap.extra.get("vllm_capacity_detail", ""),
            "vllm_boxes_detail": snap.extra.get("boxes_detail", ""),
            # The window the lease declares: the lease model's catalog context (`Config.lease_context_length`); the
            # status says "none" with the reason when no row serves that name, rather than failing the whole page.
            "context": {"requested": self._lease_window(),
                        "state": snap.extra.get("context_state", "")},
            # priority=0 wins the BAND; is_priority (0016) jumps the whole claim ORDER BY,
            # warm-model affinity included. Surfaced so a lost flag is visible, not silent.
            "run_next": snap.extra.get("run_next_state", ""),
            # What each BOX is actually doing, off broker_parrot's LAN hardware broadcast.
            # Keyed by box, so a card can show the machine behind the server. `stale` says the
            # sample stopped arriving; its numbers go blank rather than freezing on screen.
            "hardware": self.hw.snapshot(snap.now) if self.hw else {},
            "hardware_state": self.hw.status() if self.hw else {"state": "not configured"},
            # gpu_slots='all' (0030): the lease weighs the whole box, so the fleet cannot
            # co-schedule into the server we hold. Surfaced so an unweighted lease is visible.
            "gpu_slots": snap.extra.get("gpu_slots_state", ""),
            # Our own queueing, made visible: admissions in the sustain window that could
            # not get an unsaturated leased vLLM while a capable box was idle.
            "starved_recent": snap.extra.get("starved_recent", 0),
            "requests_total": snap.extra.get("requests_total", 0),
            "last_tick_at": snap.extra.get("last_tick_at"),
            "last_tick_error": snap.extra.get("last_tick_error", ""),
            # THE one cluster as its loop last published it; {"configured": false} when there is none — or several
            # (then it lists their names). `clusters`: EVERY cluster by name (worklog/parallel-clusters.md).
            "cluster": snap.extra.get("cluster") or {"configured": False},
            "clusters": {n: {**v, "serves": self._serves(v), **figures.get(n, {}), **self._engine_block(n, v)}
                         for n, v in (snap.extra.get("clusters") or {}).items()},
            # EVERY cluster the fleet DECLARES — in the run or not, and whether it may join now (cluster_run.py).
            "declared": self.run.declared() if self.run is not None else [],
            # THE SWARM CONTROL (benchswarm.py), on the poll the page already makes rather than a
            # channel of its own: whether a deep-research session is running and how far into its
            # cap, what the host runner last answered, and whether a model is loaded to put under
            # load at all. {"configured": false} where the control is off.
            "swarm": self.swarm.view(),
            # THE REQUESTS CARD (request_queue.py): one row per turn the ENGINE is working on — streaming, then the
            # no-byte turns split FCFS into prefill and queued against its own waiting gauge — with the LB's figures
            # off the relayed stream and bounded content; held turns and the last finished as counts and a list;
            # and the engine's running/waiting beside them. The engine publishes no per-request state.
            "queue": self.queue_view(snap),
            # THE RATE TO AIM FOR at the setup the cluster runs now: the best generated / prompt tok/s any
            # recorded minute reached under the same settings (perflog.throughput_target). null when no
            # minute was ever recorded for this model; the tiles then scale against their own recent peak.
            "throughput_target": one.get("throughput_target"),
            "setup_targets": one.get("setup_targets") or {},
            # DOES EACH CHOOSER ROW START: of the LB's launches of that checkpoint over the last 7 UTC days, how many
            # reached serving, per day (null = no launch that day) — off the boot log (start_rates.py).
            "start_rates": self._start_rates(snap),
            # IS THERE HEADROOM LEFT, and against WHICH ceiling — compute or memory bandwidth. null
            # when there is no cluster. Every figure in it is measured or null-with-a-reason; the one
            # thing it will not do is turn a 96 % SM fill into "the GPU is used up".
            "saturation": one.get("saturation"),
            # Every model this gateway knows how to launch, per ServingMode — the ONE catalog behind the
            # cluster's picker and classic vLLM's.
            "models": self._models_block(),
            "config": {"idle_release_s": self.cfg.idle_release_s,
                       "vllm_max_depth": self.cfg.vllm_max_depth,
                       "max_leases": self.cfg.max_leases,
                       "use_unleased_vllm": self.cfg.use_unleased_vllm,
                       # The page places each backend's history in time buckets this wide.
                       "probe_interval_s": self.cfg.probe_interval_s},
        }

    def recorder_request(self, h: _Handler, box: str) -> None:
        """GET /lb/hardware/<box>/recorder: the box's recorder hour. 503 with the reason when the hardware listener is
        not running here; 404 naming the box when no sample of it has been heard — never an empty hour."""
        if self.hw is None:
            return h.send_json(503, {"error": "the hardware listener is not running in this gateway: no recorder"})
        out = self.hw.recorder_series(box)
        if out is None:
            return h.send_json(404, {"error": f"no hardware sample heard from box {box!r}"})
        return h.send_json(200, out)

    def _engine_block(self, name: str, view: dict) -> dict:
        """WHAT THE CLUSTER'S SERVER IS (fleet.engine_block): its task, the engine of the model it is set to serve with
        that engine's settings and tiles, every engine its choices run, its card. {} for a cluster the fleet does not
        declare."""
        from vllm_lb.fleet import engine_block
        cluster = self.cfg.fleet.clusters.get(name) if self.cfg.fleet is not None else None
        return {} if cluster is None else engine_block(cluster, (view.get("desired") or {}).get("model"))

    def _serves(self, view: dict) -> dict | None:
        """The catalog model a server is SET to serve — what it loaded, else the choice it was started with, so one
        still coming up is known — and whether a chat may use it (the chat's server picker); None: the catalog
        knows neither."""
        loaded = view.get("loaded_model") or {}
        spec = self.catalog.get(loaded.get("spec")) or self.catalog.get((view.get("desired") or {}).get("model_choice"))
        return None if spec is None else {"spec": spec.id, "served_name": spec.served_name, "label": spec.label,
                                          "chat": spec.chat}

    def _models_block(self) -> dict:
        """What the page may choose from, in each ServingMode. `error` says why there is nothing to
        choose when the catalog itself could not be read — an empty picker with no reason is worse."""
        classic_default = self.catalog.default(CLASSIC)
        return {
            "cluster": [s.view() for s in self.catalog.for_mode(CLUSTER_MODE)],
            "classic": [s.view() for s in self.catalog.for_mode(CLASSIC)],
            "classic_choice": classic_default.id if classic_default is not None else None,
            "classic_selectable": False,
            "classic_detail": self.CLASSIC_NOT_SELECTABLE,
            # Classic vLLM's memory is broker_parrot's to size; it is never estimated here.
            "classic_memory": CLASSIC_MEMORY,
            "error": self.catalog.error,
        }

    # ── one request ────────────────────────────────────────────────────────

    def handle(self, h: _Handler) -> None:
        # THE HOST ALLOWLIST FIRST (host_guard.py): before any route — the page, /lb/status, a control POST, a relayed
        # turn — a request must name this gateway by a name it is really reached by, and a write must come from such
        # an origin. A DNS-rebound page names itself; it is answered here and nowhere else.
        refused = self.host_guard.refusal(h.command, h.headers.get("Host"), h.headers.get("Origin"))
        if refused is not None:
            h.read_body()     # drain what was sent, so the reply is not raced by the rest of its body
            return h.send_json(*refused)
        path = urllib.parse.urlsplit(h.path).path
        if path == "/healthz":
            return h.send_json(200, {"ok": True})
        if path == "/lb/status":
            return h.send_json(200, self.status())
        if path == "/lb/cluster" and h.command == "POST":
            return self.cluster_switch_request(h)
        # PER-CLUSTER (worklog/parallel-clusters.md): /lb/clusters/<name> (its switch), /activate, /deactivate (the run).
        if h.command == "POST" and path.startswith("/lb/clusters/"):
            return self.clusters_request(h, path)
        # Before the static branch below, which answers every other GET: this is an API call, not a page.
        if path == "/lb/cluster/estimate" and h.command in ("GET", "HEAD"):
            return self.cluster_estimate(h, urllib.parse.urlsplit(h.path).query)
        if path.startswith("/lb/clusters/") and path.endswith("/estimate") and h.command in ("GET", "HEAD"):
            return self.clusters_estimate(h, path[len("/lb/clusters/"):-len("/estimate")],
                                          urllib.parse.urlsplit(h.path).query)
        # THE OPERATOR'S KILL (request_kill.py): one turn, by its id on the REQUESTS card. Every POST shaped like one
        # is answered here, an id that names nothing included.
        if h.command == "POST" and path.startswith("/lb/requests/") and path.endswith("/kill"):
            return self.kill_request(h, urllib.parse.unquote(path[len("/lb/requests/"):-len("/kill")]))
        if h.command == "POST" and path.startswith("/lb/requests/") and path.endswith("/priority"):
            return self.priority_request(h, urllib.parse.unquote(path[len("/lb/requests/"):-len("/priority")]))
        # THE SWARM (benchswarm.py): one deep-research session, asked of the host runner. POST only —
        # a GET of it would be a link, and a link that changes state is one a prefetch or a crawler
        # follows. Its state is read from /lb/status like everything else the page shows. There is no
        # endpoint for the thirty-run matrix: it is started at the keyboard, never from the page.
        if path == "/lb/swarm/start" and h.command == "POST":
            return self.swarm_start(h)
        # THE FLIGHT RECORDER (recorder.py): one box's last hour, every figure, for the charts the panel opens on demand
        # — /lb/status carries only its summary, so a poll every few seconds does not carry the hour. GET/HEAD only.
        if path.startswith("/lb/hardware/") and path.endswith("/recorder") and h.command in ("GET", "HEAD"):
            return self.recorder_request(h, urllib.parse.unquote(path[len("/lb/hardware/"):-len("/recorder")]))
        # SAVED BENCHMARK RUNS (bench.py). Read-only, GET/HEAD only, answered by the LB itself; the
        # page is /benchmarks (an alias for the static benchmarks.html, so the operator's URL has no
        # extension). Nothing here can reach a backend or write anything.
        if path in ("/lb/bench", "/lb/bench/") and h.command in ("GET", "HEAD"):
            return h.send_json(200, self.bench.index())
        if path == "/lb/bench/run" and h.command in ("GET", "HEAD"):
            return self.bench_run(h, urllib.parse.urlsplit(h.path).query)
        # REAL-TRAFFIC PERFORMANCE LOG (perflog.py): the minutes the cluster served real requests, as
        # this LB recorded them. Read-only, GET/HEAD only; the page is /benchmarks.
        if path in ("/lb/perf", "/lb/perf/") and h.command in ("GET", "HEAD"):
            return self.perf_index(h, urllib.parse.urlsplit(h.path).query)
        # THE SAME LOG, FOLDED INTO SESSIONS (perflog.sessions): one row per generation of the pair —
        # the model that was loaded, what it served, what it cost to start and what it drew. Folded HERE
        # rather than in the browser because this page is the history: it must not thin out as the log
        # grows past what /lb/perf will hand over (MAX_WINDOWS).
        if path in ("/lb/sessions", "/lb/sessions/") and h.command in ("GET", "HEAD"):
            return self.sessions_index(h, urllib.parse.urlsplit(h.path).query)
        if path in ("/sessions", "/sessions/") and h.command in ("GET", "HEAD"):
            path = "/sessions.html"
        if path in ("/benchmarks", "/benchmarks/") and h.command in ("GET", "HEAD"):
            path = "/benchmarks.html"
        if path == "/v1/models" and h.command in ("GET", "HEAD"):
            return h.send_json(200, models_payload(self.hub.snapshot(), self.cfg))
        # NOTHING BUT INFERENCE IS RELAYED FROM THE API. `forward()` opens with
        # `inference = h.command == "POST"`, so every OTHER method under /v1/ is a reply the LB writes, and
        # the frozen rule covers it. Fixing this one method at a time is what cost rounds 4, 5 and 6: round
        # 4 answered the own surface, round 5 added `OPTIONS` — and GET and HEAD are SIMPLE cross-origin
        # requests, which a browser sends with no preflight at all, so a 405 on OPTIONS was never consulted
        # for them. They were still proxied to the operator's cluster and relayed with its
        # `access-control-allow-origin: *` (worklog/vllm-gateway-deep-audit.md, G5, sixth audit).
        # `/v1/models` GET/HEAD is answered by the LB above and stays a 200.
        if path.startswith("/v1/") and h.command != "POST":
            h.read_body()
            return h.send_own(405, json.dumps({"error": {
                "type": "method_not_allowed", "method": h.command, "path": path,
                "message": "this gateway relays inference only, which is POST; it answers no cross-origin "
                           "preflight and grants no cross-origin access"}}).encode(),
                "application/json", (("Allow", "POST"),))
        # The gateway UI, and the whole of the LB's OWN surface: everything outside /v1/. EVERY method ends
        # here, not just GET/HEAD. While only GET/HEAD were answered, an `OPTIONS /` — the CORS preflight
        # every browser sends, to the page's own path — fell through to `forward()`, was proxied to the
        # operator's cluster, and came back under `relaying` carrying the backend's
        # `access-control-allow-origin: *` and none of OWN_HEADERS, on a port published through a tunnel.
        # DELETE, PUT and PATCH did the same (worklog/vllm-gateway-deep-audit.md, G5, fourth audit).
        # The LB's own POST endpoints — /lb/cluster, /lb/swarm/start and
        # /lb/requests/<id>/kill — are answered above, so this branch never sees them.
        if not path.startswith("/v1/"):
            if h.command in ("GET", "HEAD"):
                found = static_file(path)
                if found is not None:
                    target, ctype = found
                    return h.send_own(200, target.read_bytes(), ctype, (("Cache-Control", "no-cache"),))
                # No file of that name. A browse-shaped GET is NOT an API call, so it ends here
                # rather than being forwarded to an inference backend — which is also what stops a
                # traversal attempt turning into a proxied request.
                return h.send_json(404, {"error": {"type": "not_found", "path": path}})
            h.read_body()     # drain what was sent, so the reply is not raced by the rest of its body
            return h.send_own(405, json.dumps({"error": {
                "type": "method_not_allowed", "method": h.command, "path": path,
                "message": "this path is served by the gateway itself and is never proxied; it answers "
                           "GET and HEAD only, and grants no cross-origin access"}}).encode(),
                "application/json", (("Allow", "GET, HEAD"),))
        body = h.read_body()
        parsed = _as_json(body, h.headers.get("Content-Type"))
        # Everything from here is a POST under /v1/ — a turn. It is booked for the drain, and one that
        # arrives DURING a drain is held until it is over and answered `draining` (`drain`).
        turn = self._admit(h, parsed)
        if turn is None:
            return None
        try:
            if turn.holding:
                return self._wait_out_drain(h, turn)
            if (h.command == "POST" and self.cfg.search_url
                    and websearch.is_web_search_request(h.path, parsed)):
                # No backend is involved: the query is in the prompt and the answer is the two
                # blocks the client harvests. It is not inference, so it is not lease demand.
                turn.backend = turn.box = request_queue.WEBSEARCH
                turn.sent_at = time.monotonic()
                return self.answer_web_search(h, parsed)
            self.forward(h, h.path, body, parsed, turn=turn)
        except BaseException:
            turn.error = turn.error or "proxy error"
            raise
        finally:
            self._retire(turn, h)

    # ── the server-side web_search tool, performed here ────────────────────

    @staticmethod
    def _asked_by(h: _Handler) -> str:
        """Who sent a switch. Behind the tunnel the address is frpc's 127.0.0.1; the User-Agent and the edge's
        X-Forwarded-For are what tell a browser from a script."""
        return (f"asked by {h.client_address[0]} ua={h.headers.get('User-Agent')!r} "
                f"xff={h.headers.get('X-Forwarded-For')!r}")

    def _observed_only(self) -> bool:
        """Is the cluster a node this LB only observes (control `none`, vllm_lb/topology.py)?"""
        return self.cfg.cluster_configured and self.cfg.cluster_control == topology.NONE

    def _refuse_observed(self, h: _Handler) -> None:
        """A switch or an estimate for a node this LB cannot start, stop or reconfigure: 409, in the one sentence the
        page shows for it (`topology.OBSERVED_WHY`) — never accepted and dropped."""
        return h.send_json(409, {"error": {"type": "cluster_observed_only", "message": topology.OBSERVED_WHY}})

    #: What a switch for a node the LB requests may name besides `enabled`: `force` and the figures an intent moves.
    LOCAL_SWITCH_KEYS = ("enabled", "force")

    def _local_estimate(self, h: _Handler, query: str, cfg: Config | None = None) -> None:
        """GET /lb/cluster/estimate?model=&max_model_len=&max_num_seqs=&gpu_memory_utilization=&max_num_batched_tokens=
        &kv_cache_memory=&kv_offload_ram_gib=&kv_cache_dtype= for a node the LB requests: what that selection needs on the card and the
        fleet's verdict on it (`topology.local_fit`), asked as the operator picks, so a refusal is read while it can
        still be changed — 200 either way. Left out, a figure is the stored wish's (else the boot default's); an
        optional key (fleet.OPTIONAL: the prefill chunk, the pinned pool, the RAM tier) asked BLANK is vLLM's own (as
        vLLM reads an empty value); a figure that is not a number is a 400 naming it."""
        from vllm_lb import node_control as nc
        from vllm_lb.fleet import CHUNK, COMPUTE, KV_DTYPE, OPTIONAL, PIN, QUEUE, RAM_TIER
        cfg = self.cfg if cfg is None else cfg
        asked = {k: v[-1] for k, v in urllib.parse.parse_qs(query, keep_blank_values=True).items()}
        args = {k: v for k, v in asked.items() if v != ""}
        try:
            current = nc.IntentStore(cfg.cluster_state_path).read()
        except ValueError:
            current = None
        overrides = self._read_by(cfg, dict(current["launch"]) if current else {}, args.get("model"))
        for key in OPTIONAL:
            if asked.get(key) == "":
                overrides[key] = None
        for key, kind in (("model", str), ("max_model_len", int), ("max_num_seqs", int),
                          ("gpu_memory_utilization", float), (CHUNK, int), (PIN, int), (RAM_TIER, int),
                          (KV_DTYPE, str), (QUEUE, int), (COMPUTE, str)):
            if key not in args:
                continue
            try:
                overrides[key] = kind(args[key])
            except ValueError:
                return h.send_json(400, {"error": {"type": "invalid_request", "message": f"`{key}` must be a "
                                                   f"{'whole number' if kind is int else 'number'}"}})
        return h.send_json(200, topology.local_fit(cfg, overrides))

    def _local_switch(self, h: _Handler, body: dict, cfg: Config | None = None) -> None:
        """POST /lb/cluster for a node the LB REQUESTS and its host APPLIES (control `local`, node_control.py):
        `{enabled, force?, <fleet.MOVABLE key>?…}` becomes the node's WISH — the stored one's figures, moved by what
        the body names — after the fleet's ONE validator took it (`fleet.validate_launch`: the launch block's own rules,
        the card's fit); the cluster loop releases it to the host once the node has finished what it serves (`force`:
        at once). Refused, in the validator's words and with nothing written: a key no intent may move or no launch
        key at all (400), a value out of bounds (400) — and the whole switch while no applier is listening on the
        node's host (503, the report's own line), because nothing would ever apply it."""
        from vllm_lb import node_control as nc
        from vllm_lb.fleet import MOVABLE, validate_launch
        cfg = self.cfg if cfg is None else cfg
        now = self.hub.clock()                         # the clock the loop and the report are read by
        report = nc.read_report(nc.node_dir(cfg.cluster_state_path), now)
        if not report["listening"]:
            return h.send_json(503, {"error": {"type": "node_applier_not_listening", "message": report["error"]}})
        force = body.get("force", False)
        if not isinstance(force, bool):
            return h.send_json(400, {"error": {"type": "invalid_request", "message": "`force` must be true or false"}})
        store = nc.IntentStore(cfg.cluster_state_path)
        launch = self._local_launch(cfg, body)
        try:
            validate_launch(topology.fleet_cluster(cfg), launch)
        except ValueError as exc:
            return h.send_json(400, {"error": {"type": "invalid_launch", "message": str(exc),
                                               "movable": list(MOVABLE)}})
        wish = store.write(nc.UP if body["enabled"] else nc.DOWN, launch, now=now, force=force)
        log.info("cluster: node wish #%d %s %s%s", wish["id"], wish["state"], wish["launch"],
                 " (force)" if force else "")
        return h.send_json(200, {"desired": wish})

    def _local_launch(self, cfg: Config, body: dict) -> dict:
        """The launch keys a local node's switch `body` writes: the stored wish's, moved by what the body names."""
        from vllm_lb import node_control as nc
        try:
            current = nc.IntentStore(cfg.cluster_state_path).read()
        except ValueError:
            current = None                             # an unreadable wish is replaced by this one
        launch = dict(current["launch"]) if current else {}
        asked = {k: v for k, v in body.items() if k not in self.LOCAL_SWITCH_KEYS}
        launch = self._read_by(cfg, launch, asked.get("model"))
        launch.update(asked)
        return launch

    @staticmethod
    def _read_by(cfg: Config, launch: dict, ref: str | None) -> dict:
        """The stored `launch` keys a node moved to model `ref` still has: ANOTHER MODEL, MAYBE ANOTHER ENGINE — what the
        stored wish set for the old one is no setting of the new one unless its engine reads it (fleet.NodeEngine.
        settings). A key the caller names itself is judged as asked, never dropped."""
        from vllm_lb.fleet import ENGINES, MOVABLE
        cluster = topology.fleet_cluster(cfg)
        choice = next((m for m in cluster.choices if m.ref == ref), None) if ref is not None else None
        if choice is None or not choice.spec.node_engine:
            return launch
        reads = {st.key for st in ENGINES[choice.spec.node_engine].settings(cluster.task)}
        return {k: v for k, v in launch.items() if k not in MOVABLE or k in reads}

    def clusters_estimate(self, h: _Handler, name: str, query: str) -> None:
        """GET /lb/clusters/<name>/estimate — `cluster_estimate` for THAT cluster (the page asks `${switch}/estimate`):
        a local node's fit on its card, a pair's per-box estimate. 404 for a name the fleet does not declare; 409 for
        one not in this run (it has no view to price against)."""
        names = [c.name for c in self.cfg.clusters]
        if name not in names:
            return h.send_json(404, {"error": {"type": "cluster_unknown",
                                               "message": f"no cluster {name!r}: the fleet declares {', '.join(names)}"}})
        if name not in self.hub.cluster_cfgs:
            return h.send_json(409, {"error": {"type": "cluster_inactive", "message": (
                f"cluster {name!r} is not in this run: POST /lb/clusters/{name}/activate first")}})
        return self.cluster_estimate(h, query, name=name)

    def clusters_request(self, h: _Handler, path: str) -> None:
        """POST /lb/clusters/<name> — that cluster's switch (`cluster_switch_request` for it); /lb/clusters/<name>/activate
        and /deactivate — the run's (cluster_run.ClusterRun), answered in its words and status. 404 for a name the fleet
        does not declare; a switch for a cluster not in the run is a 409 saying to activate it first."""
        name, _, action = path[len("/lb/clusters/"):].partition("/")
        names = [c.name for c in self.cfg.clusters]
        if name not in names:
            return h.send_json(404, {"error": {"type": "cluster_unknown",
                                               "message": f"no cluster {name!r}: the fleet declares {', '.join(names)}"}})
        if action in ("activate", "deactivate"):
            from vllm_lb.cluster_run import RunRefused
            h.read_body()
            if not _is_json_type(h.headers.get("Content-Type")):
                return h.send_json(415, {"error": {
                    "type": "unsupported_media_type",
                    "message": f"an {action} is a POST with `Content-Type: application/json` (a body of {{}} or none); "
                               f"any other type is one a cross-site page could send"}})
            if self.run is None:
                return h.send_json(503, {"error": {"type": "no_run", "message": "this LB keeps no run of clusters"}})
            try:
                return h.send_json(200, getattr(self.run, action)(name))
            except RunRefused as exc:
                return h.send_json(exc.status, {"error": {"type": exc.kind, "message": str(exc)}})
        if action:
            return h.send_json(404, {"error": {"type": "not_found", "message": f"no action {action!r}: activate, deactivate, "
                                                                                f"or no action for the switch"}})
        if name not in self.hub.cluster_cfgs:
            return h.send_json(409, {"error": {"type": "cluster_inactive",
                                               "message": f"cluster {name!r} is not in this run: POST /lb/clusters/{name}/activate first"}})
        return self.cluster_switch_request(h, name=name)

    def cluster_switch_request(self, h: _Handler, name: str | None = None) -> None:
        """POST /lb/cluster `{enabled, restart?, model_choice?, max_num_seqs?, max_model_len?,
        max_num_batched_tokens?, num_speculative_tokens?, ple_replicate?, kv_cache_memory?,
        kv_offload_gib?, long_prefill_token_threshold?}` — writes the operator's intent and nothing else. The
        cluster loop acts on its next tick; the page watches /lb/status. The ModelChoice is validated by the ONE
        validator, `ModelCatalog.choice`.

        `restart: true` with `enabled: true` is "reload the pair at these settings": the generation is bumped
        while the mode stays `cluster`, so the boxes are never given back between generations
        (worklog/vllm-cluster-handover.md H1). Without it, changing the selection under a running cluster is
        still refused with a 409.

        The ENGINE KNOBS — `max_num_batched_tokens` (the prefill budget, 2048..65536) and
        `num_speculative_tokens` (the drafter's depth, 0..8, 0 = off) — are left out to keep what is stored, and
        take `null` as the default (the box's CLUSTER_MAX_NUM_BATCHED_TOKENS; the model's catalog LaunchProfile),
        so what `desired` says can be posted back as it is. A value of the wrong type or out of range is a 400
        naming the field and its range; a depth for a model with no drafter is the switch's refusal, a 409.
        `long_prefill_token_threshold` (0..65536, the most prompt tokens one request may take in one step) is
        taken the same way; 0 and null are vLLM's own default, off.

        `kv_cache_memory` also takes "auto": every Start while it is the choice sizes the pool to what the boxes
        have left (`cluster_defaults.kv_auto`, the figure GET /lb/cluster/estimate shows as `auto`) and stores
        that number beside `kv_cache_auto: true`; a pool it cannot size is the switch's 409, in the gate's words."""
        body = _read_switch(h)
        if body is None:
            return None
        if name is None and len(self.cfg.clusters) > 1:
            # Several clusters: "the cluster" names none of them (worklog/parallel-clusters.md). Refused naming each
            # one's own route — never applied to the first.
            return h.send_json(409, {"error": {"type": "cluster_unnamed", "message": (
                f"{len(self.cfg.clusters)} clusters: name the one to switch — "
                + ", ".join(f"POST /lb/clusters/{c.name}" for c in self.cfg.clusters))}})
        # THIS cluster's config and switch: the named one's (`clusters_request`), else the one cluster's.
        cfg = self.hub.cluster_cfgs[name] if name is not None else self.cfg
        switch = self._switch_for(name) if name is not None else self.cluster_switch
        # A BLUEPRINT IS NO SWITCH KEY (operator, 2026-10-30: "Blueprints meant to be only for agent to build their own
        # fleet settings, not something user chooses"): the reference setups are the setup wizard's, prefilled into a
        # deployment's own answers. A body still naming one is refused, never accepted with the key dropped.
        if "blueprint" in body:
            return h.send_json(400, {"error": {"type": "blueprint_retired", "message": (
                f"`blueprint` is no switch key: {BLUEPRINTS_MOVED} — post the settings themselves")}})
        if body.get("enabled") is True and self.run is not None and cfg.cluster_name:
            # A START NEVER LANDS ON A TAKEN BOX it may not share (cluster_run.py `blockers`: the one definition, the +
            # tab's too; a box is shared only within its memory pool) — and
            # its check and its write are ONE step under the run's lock (`start_guard`): two Starts on shared boxes
            # with no tick between them both passed a check made apart from the write (AUDIT-FINAL P1-1).
            # A local node's Start is judged at the launch it would write (`_local_launch`): its memory on a shared box.
            launch = self._local_launch(cfg, body) if cfg.cluster_control == topology.LOCAL else None
            with self.run.start_guard(cfg.cluster_name, launch) as blocked:
                if blocked:
                    return h.send_json(409, {"error": {"type": "cluster_blocked", "message": (
                        f"cluster {cfg.cluster_name!r} cannot start: " + "; ".join(b["reason"] for b in blocked))}})
                return self._switch_apply(h, body, cfg, switch)
        return self._switch_apply(h, body, cfg, switch)

    def _switch_apply(self, h: _Handler, body: dict, cfg: Config, switch) -> None:
        """`cluster_switch_request` past the box check: the switch for THIS cluster (`cfg`, `switch`), written — for a
        Start, while the caller holds the run's lock."""
        if cfg.cluster_configured and cfg.cluster_control == topology.NONE:
            return self._refuse_observed(h)
        if cfg.cluster_configured and cfg.cluster_control == topology.LOCAL:
            return self._local_switch(h, body, cfg)
        settings = {}
        for key in ("max_num_seqs", "max_model_len"):
            value = body.get(key)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                return h.send_json(400, {"error": {"type": "invalid_request",
                                                   "message": f"`{key}` must be an integer"}})
            settings[key] = value
        for key, (lo, hi) in (("max_num_batched_tokens", MAX_NUM_BATCHED_TOKENS_RANGE),
                              ("num_speculative_tokens", NUM_SPECULATIVE_TOKENS_RANGE),
                              ("kv_cache_memory", KV_CACHE_MEMORY_RANGE),
                              ("kv_offload_gib", KV_OFFLOAD_GIB_RANGE),
                              ("long_prefill_token_threshold", LONG_PREFILL_TOKEN_THRESHOLD_RANGE)):
            if key not in body:
                continue                                  # left out: the stored value stays
            value = body[key]
            if key == "kv_cache_memory" and value == KV_AUTO:
                settings[key] = KV_AUTO                   # sized by this Start, from what the boxes report
                continue
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi):
                return h.send_json(400, {"error": {"type": "invalid_request",
                                                   "message": f"`{key}` must be an integer from {lo} to {hi}"
                                                              + (" (0 turns speculation off)"
                                                                 if key == "num_speculative_tokens" else "")
                                                              + (" (0 turns the disk tier off)"
                                                                 if key == "kv_offload_gib" else "")
                                                              + (" (0 turns it off)"
                                                                 if key == "long_prefill_token_threshold" else "")
                                                              + (f", \"{KV_AUTO}\" to size it to what the boxes "
                                                                 f"have left" if key == "kv_cache_memory" else "")
                                                              + ", or null for the default"}})
            settings[key] = value
        # WHERE THE PLE TABLE LIVES — true/false, or null for the model's own placement. Not a range, so it
        # is checked for its type here and for nothing else: what it COSTS is the start gate's business.
        if "ple_replicate" in body:
            value = body["ple_replicate"]
            if value is not None and not isinstance(value, bool):
                return h.send_json(400, {"error": {"type": "invalid_request",
                                                   "message": "`ple_replicate` must be true, false, "
                                                              "or null for the model's default"}})
            settings["ple_replicate"] = value
        # THE KV CACHE DTYPE — a dtype name, or null for the profile's own. Whether the model's engine image runs it on
        # these cards is the switch's refusal (409, the image's own words); here only its type.
        if "kv_cache_dtype" in body:
            value = body["kv_cache_dtype"]
            if value is not None and not isinstance(value, str):
                return h.send_json(400, {"error": {"type": "invalid_request",
                                                   "message": "`kv_cache_dtype` must be a dtype name (e.g. "
                                                              "fp8_e4m3, nvfp4), or null for the profile's own"}})
            settings["kv_cache_dtype"] = value
        # WHICH CHECKPOINT — true (the model's uncensored twin), false (the original), or null for the model's
        # default. Whether the model HAS a twin is the switch's refusal (409), like a depth for a model with
        # no drafter.
        if "uncensored" in body:
            value = body["uncensored"]
            if value is not None and not isinstance(value, bool):
                return h.send_json(400, {"error": {"type": "invalid_request",
                                                   "message": "`uncensored` must be true, false, "
                                                              "or null for the model's default checkpoint"}})
            settings["uncensored"] = value
        # THE ESTIMATE, THE OPERATOR'S EXPLICIT CHOICE (docs/design/engines.md): true starts a model never measured on
        # these cards sized by memory_estimate's computation from its config.json — labelled "estimated" everywhere.
        # Whether the model IS unmeasured here, and can be estimated, is the switch's refusal (409); here only its type.
        if "estimated" in body:
            value = body["estimated"]
            if not isinstance(value, bool):
                return h.send_json(400, {"error": {"type": "invalid_request",
                                                   "message": "`estimated` must be true or false"}})
            settings["estimated"] = value
        choice = body.get("model_choice")
        if choice is not None:
            if not isinstance(choice, str) or not choice.strip():
                return h.send_json(400, {"error": {"type": "invalid_request",
                                                   "message": "`model_choice` must be a model id"}})
            settings["model_choice"] = choice
        # A RESTART: reload the pair at these settings without passing through classic, which is what hands
        # the GB10 boxes back to the fleet (worklog/vllm-cluster-handover.md D1). Absent = False, so every
        # existing client keeps the old refusal and no stray Start can replace a serving pair.
        restart = body.get("restart", False)
        if not isinstance(restart, bool):
            return h.send_json(400, {"error": {"type": "invalid_request",
                                               "message": "`restart` must be true or false"}})
        if switch is None or not cfg.cluster_configured:
            return h.send_json(409, {"error": {"type": "cluster_rejected",
                                               "message": "the cluster is not configured on this gateway"}})
        # AUTO is sized by the Start that stores it, from what the cluster loop last published — the same figures
        # GET /lb/cluster/estimate showed the operator — and refused, with the gate's reason, when they do not fit.
        view_of = lambda: self.hub.snapshot().cluster_views().get(cfg.cluster_name) or {}  # noqa: E731
        resolve = kv_auto_resolver(cfg, self.catalog, view_of)
        try:
            desired = switch.set(enabled=body["enabled"], restart=restart, resolve_kv_auto=resolve,
                                 admit=memory_admission(cfg, self.catalog, view_of),
                                              **settings)
        except UnknownModelChoice as exc:
            # The caller named something this gateway cannot launch: their mistake, and the message
            # says what it does have.
            return h.send_json(400, {"error": {"type": "unknown_model_choice", "message": str(exc)}})
        except ValueError as exc:
            return h.send_json(409, {"error": {"type": "cluster_rejected", "message": str(exc)}})
        # The most consequential write this LB takes — it takes both GB10 boxes from the fleet — so its line
        # names the caller (`_asked_by`), and WHICH MODEL it just committed them to, at which settings.
        log.info("[proxy] cluster %s: %s %s%s (generation %d, %d seqs, %d tokens, prefill budget %s, "
                 "speculative tokens %s, PLE %s, KV pool %s, KV dtype %s, disk offload %s, long-prefill threshold %s) — %s",
                 "restart" if restart and body["enabled"] else "switch",
                 desired.mode, desired.model_choice,
                 " uncensored" if desired.uncensored else ("" if desired.uncensored is None else " original"),
                 desired.generation, desired.max_num_seqs,
                 desired.max_model_len, _or_default(desired.max_num_batched_tokens),
                 _or_default(desired.num_speculative_tokens),
                 "default" if desired.ple_replicate is None else
                 ("replicated" if desired.ple_replicate else "split"),
                 _or_default(desired.kv_cache_memory) + (f" ({KV_AUTO})" if desired.kv_cache_auto else ""),
                 _or_default(desired.kv_cache_dtype), _or_default(desired.kv_offload_gib),
                 _or_default(desired.long_prefill_token_threshold),
                 self._asked_by(h))
        return h.send_json(202, {"desired": asdict(desired)})

    def bench_run(self, h: _Handler, query: str) -> None:
        """GET /lb/bench/run?id=<run> — one saved run with its time series and per-request records.

        A run id is a file name, so it is resolved and contained by `BenchStore.path_of` before
        anything is opened; an id that names nothing is a 404 and never a 500. No id at all is the
        caller's mistake, and the message says so rather than returning the first run it finds.
        """
        ids = urllib.parse.parse_qs(query).get("id") or []
        run_id = ids[-1].strip() if ids else ""
        if not run_id:
            return h.send_json(400, {"error": {"type": "invalid_request",
                                               "message": "`id` names the run to read"}})
        found = self.bench.run(run_id)
        if found is None:
            return h.send_json(404, {"error": {"type": "not_found", "id": run_id,
                                               "message": "no saved run by that name"}})
        return h.send_json(200, {"run": found})

    #: The widest range /lb/perf and /lb/sessions read, in days: ten years is "all of it" for this log.
    PERF_MAX_DAYS = 3650.0

    def _perf_days(self, query: str, *, default: float | None = 7.0):
        """`days` as both log endpoints read it: a positive number of days, `all` (or 0) for every one,
        absent for `default`. `_BAD_DAYS` for anything else — the caller's mistake, which each endpoint
        answers in its own words rather than silently meaning the default."""
        values = urllib.parse.parse_qs(query).get("days") or []
        raw = values[-1].strip().lower() if values else ""
        if raw == "":
            return default
        if raw in ("all", "0"):
            return None
        try:
            days = float(raw)
        except ValueError:
            return _BAD_DAYS
        if not (days > 0) or days != days:
            return _BAD_DAYS
        return min(days, self.PERF_MAX_DAYS)

    def _describe_model(self, model_choice: str) -> dict:
        """What the catalog calls a RECORDED model id and which engine serves it — the log stores the id, and
        an id is not a name. One the catalog does not know keeps its raw id as its label and says so. `engine_name`
        is how the engine is written for a reader (saturation.ENGINE_NAMES): the page says a launch that left a
        setting unset ran "vLLM's default", and only the catalog knows which engine that was."""
        spec = self.catalog.get(model_choice)
        if spec is None:
            return {"label": model_choice, "engine": None, "engine_name": None, "known": False}
        profile = spec.profile(CLUSTER_MODE)
        engine = profile.engine if profile is not None else None
        return {"label": spec.label, "engine": engine,
                "engine_name": ENGINE_NAMES.get(engine, engine) if engine is not None else None,
                "served_name": spec.served_name, "known": True}

    def _server_param(self, h: _Handler, query: str):
        """`?server=<name>` on /lb/perf and /lb/sessions: a server the fleet DECLARES (in the run or not — its minutes
        outlive its run), None when absent; else the 404 is sent, naming the declared ones, and `_BAD_SERVER` returned."""
        values = urllib.parse.parse_qs(query).get("server") or []
        name = values[-1].strip() if values else ""
        if not name:
            return None
        names = [c.name for c in self.cfg.clusters]
        if name in names:
            return name
        h.send_json(404, {"error": {"type": "server_unknown",
                                    "message": f"no server {name!r}: the fleet declares {', '.join(names) or 'none'}"}})
        return _BAD_SERVER

    def perf_index(self, h: _Handler, query: str) -> None:
        """GET /lb/perf?days=N — the recorded windows that ended inside the last N days (default 7;
        `days=all` or 0 for every one), what could not be read, and the recorder's own state. A `days`
        that is not a number is the caller's mistake and says so, rather than silently meaning 7."""
        days = self._perf_days(query)
        if days is _BAD_DAYS:
            return h.send_json(400, {"error": {"type": "invalid_request",
                                               "message": "`days` is a positive number of days, or `all`"}})
        server = self._server_param(h, query)
        if server is _BAD_SERVER:
            return None
        out = self.perf.index(days=days, keep=self._keep(server))
        out["server"] = server
        # THE MODELS THIS DEPLOYMENT RECORDED in the range (perflog.recorded_models) — the selector's list,
        # never the catalog's: the catalog only NAMES a model that has minutes; one it does not know keeps its id.
        out["models"] = [{**m, **self._describe_model(m["id"])} for m in out["models"]]
        # Every cluster start in the same range (perflog.BootRecorder), for the matrix's boot block.
        boots = self.boots.index(days=days, keep=self._keep(server))
        out["boots"], out["boots_malformed"] = boots["windows"], boots["malformed_total"]
        out["recorder"] = (self.perf_recorder.status() if self.perf_recorder is not None
                           else {"enabled": False, "dir": str(self.perf.dir)})
        out["cluster_configured"] = self.cfg.cluster_configured
        return h.send_json(200, out)

    def sessions_index(self, h: _Handler, query: str) -> None:
        """GET /lb/sessions?days=N — every LOADED-MODEL SESSION in range, newest first, with the totals
        and the per-model rollup the page's tiles read (`perflog.sessions`). `days` is the same word it
        is on /lb/perf, and the DEFAULT IS ALL OF IT: this is the history page, not a live view.

        The catalog says what a model id is called and which engine serves it — the log stores the id,
        and an id is not a name. A model the catalog no longer has keeps its id and says so.
        """
        days = self._perf_days(query, default=None)
        if days is _BAD_DAYS:
            return h.send_json(400, {"error": {"type": "invalid_request",
                                               "message": "`days` is a positive number of days, or `all`"}})
        server = self._server_param(h, query)
        if server is _BAD_SERVER:
            return None
        index = self.perf.index(days=days, keep=self._keep(server))
        # EVERY boot line, the range cut inside the fold (`since`): a line before the range makes no session, but it
        # is what says a line in the range is a start read again (perflog.read_again) rather than a 7 s start.
        boots = self.boots.index(days=None, now=index["now"], keep=self._keep(server))
        out = sessions(index["windows"], boots["windows"], describe=self._describe_model, since=index["cutoff"])
        out["server"] = server
        # WHAT THE TOKENS WOULD HAVE COST: a price list's dollars on every session, rollup and total —
        # None where nothing is published, never 0 and never another model's figure (pricing.py).
        _pricing.apply(out, self.pricing)
        # THE SESSION RUNNING NOW (operator, 2026-10-06: "on top show current session running model
        # costs and Opus, below total all models"). The pair that SERVES tags every log minute with
        # exactly this model id and generation — the effective launch record, never the current wish —
        # so the page can find the one row this pair owns. No pair serving, or one that has launched
        # without an effective record: null, and the page draws no tile for it — never a stand-in model. WHICH pair:
        # the named server's; unscoped, the one server's when one runs (with several, none is "the" one).
        views = self.hub.snapshot().cluster_views()
        cluster = (views.get(server) if server else
                   next(iter(views.values())) if len(views) == 1 else None) or {}
        eff = cluster.get("effective")
        eff = eff if isinstance(eff, dict) else {}
        cur_model, cur_gen = eff.get("model_choice"), eff.get("generation")
        # AN ENGINE THE LB DID NOT START (a node it only observes) has no generation: its session is the engine
        # process serving, by the same figure its minutes carry (`up_since`, the engine's process start) — and with
        # neither, nothing names a running session.
        engine = None if cur_gen is not None else engine_start(cluster.get("up_since"))
        out["current"] = (
            {"model_choice": cur_model, "generation": cur_gen, "engine_started_at": engine,
             "state": cluster.get("state"), "label": self._describe_model(cur_model).get("label")}
            if isinstance(cur_model, str) and cur_model and (cur_gen is not None or engine is not None) else None)
        out.update({"now": index["now"], "days": days, "dir": index["dir"], "error": index["error"],
                    # The windows the log would not hand over (MAX_WINDOWS) are minutes these sessions
                    # really had: a total that silently leaves them out is a wrong total, so it is said.
                    "windows_truncated": index["truncated"], "malformed": index["malformed_total"],
                    "boots_malformed": boots["malformed_total"],
                    "recorder": (self.perf_recorder.status() if self.perf_recorder is not None
                                 else {"enabled": False, "dir": str(self.perf.dir)}),
                    "cluster_configured": self.cfg.cluster_configured})
        return h.send_json(200, out)

    def cluster_estimate(self, h: _Handler, query: str, name: str | None = None) -> None:
        """GET /lb/cluster/estimate?model_choice=&max_num_seqs=&max_model_len=&max_num_batched_tokens=
        &num_speculative_tokens=&kv_cache_memory=&ple_replicate= — what a selection would need on each box, BEFORE
        anything is started, and (`auto`) the KV pool a Start at `kv_cache_memory: "auto"` would pin now. The page
        asks as the operator picks, so the cost of a choice is visible while it can still be changed (M8).
        `num_speculative_tokens=0` is a launch without the drafter, whose weights it then does not count; left
        out, the model's own. `max_num_batched_tokens` left out is the budget a launch would really resolve to —
        tp-rank's own default — never "no budget": this endpoint is where a refusal has to be READABLE while the
        knob can still be turned, instead of afterwards from two hung boxes (2026-09-30)."""
        if name is None and len(self.cfg.clusters) > 1:
            # Several clusters: "the cluster" names none — refused naming each one's own route, never the first's.
            return h.send_json(409, {"error": {"type": "cluster_unnamed", "message": (
                f"{len(self.cfg.clusters)} clusters: name the one to estimate — "
                + ", ".join(f"GET /lb/clusters/{c.name}/estimate" for c in self.cfg.clusters))}})
        # THIS cluster's one-cluster config and view (`clusters_estimate` named it; else the one cluster's).
        cfg = self.hub.cluster_cfgs[name] if name is not None else self.cfg
        if cfg.cluster_configured and cfg.cluster_control == topology.NONE:
            return self._refuse_observed(h)
        if cfg.cluster_configured and cfg.cluster_control == topology.LOCAL:
            return self._local_estimate(h, query, cfg)
        args = urllib.parse.parse_qs(query)

        def one(name: str) -> str:
            values = args.get(name) or []
            return values[-1].strip() if values else ""

        def refuse(message: str, kind: str = "invalid_request"):
            return h.send_json(400, {"error": {"type": kind, "message": message}})

        try:
            spec = self.catalog.choice(CLUSTER_MODE, one("model_choice"))
        except UnknownModelChoice as exc:
            return refuse(str(exc), "unknown_model_choice")
        except ValueError as exc:                          # a catalog that could not be read, or unsupported
            return refuse(str(exc), "unknown_model_choice")
        lo_seqs, hi_seqs = spec.limits.max_num_seqs
        try:
            # Nothing given is this spec's own floor and its whole window: the cheapest honest answer.
            seqs = int(one("max_num_seqs")) if one("max_num_seqs") else lo_seqs
            ctx = int(one("max_model_len")) if one("max_model_len") else spec.limits.max_model_len
        except ValueError:
            return refuse("`max_num_seqs` and `max_model_len` must be whole numbers")
        if not lo_seqs <= seqs <= hi_seqs:
            return refuse(f"max_num_seqs must be between {lo_seqs} and {hi_seqs} for {spec.label}")
        if not MAX_MODEL_LEN_CAP[0] <= ctx <= spec.limits.max_model_len:
            return refuse(f"max_model_len must be between {MAX_MODEL_LEN_CAP[0]} and "
                          f"{spec.limits.max_model_len} for {spec.label}")
        budget = one("max_num_batched_tokens")
        lo_budget, hi_budget = MAX_NUM_BATCHED_TOKENS_RANGE
        if budget and not (budget.isdigit() and lo_budget <= int(budget) <= hi_budget):
            return refuse(f"max_num_batched_tokens must be a whole number between {lo_budget} and {hi_budget}")
        # Nothing given: what the launch would resolve — the model's own budget, else the fleet's (the cluster's
        # `max_num_batched_tokens`); None for an engine that chunks for itself, and the estimate charges no budget.
        profile = spec.profile(CLUSTER_MODE)
        batched = (int(budget) if budget else profile.prefill_budget if profile.prefill_budget is not None
                   else None if "max_num_batched_tokens" in profile.refuses else cfg.cluster_max_num_batched_tokens)
        drafts = one("num_speculative_tokens")
        lo_drafts, hi_drafts = NUM_SPECULATIVE_TOKENS_RANGE
        if drafts and not (drafts.isdigit() and lo_drafts <= int(drafts) <= hi_drafts):
            return refuse(f"num_speculative_tokens must be a whole number between {lo_drafts} and {hi_drafts}")
        pin_raw = one("kv_cache_memory")
        lo_pin, hi_pin = KV_CACHE_MEMORY_RANGE
        if pin_raw and pin_raw != KV_AUTO and not (pin_raw.isdigit() and lo_pin <= int(pin_raw) <= hi_pin):
            return refuse(f"kv_cache_memory must be a whole number of bytes between {lo_pin} and {hi_pin}, "
                          f"or {KV_AUTO}")
        ple_raw = one("ple_replicate")
        if ple_raw not in (None, "", "true", "false"):
            return refuse("ple_replicate must be true or false")
        # The KV dtype the selection would run (empty: the profile's own) — refused in the switch's words.
        dtype = one("kv_cache_dtype") or None
        if dtype is not None:
            problem = spec.kv_cache_dtype_problem(dtype, self._switch_archs(cfg), int(drafts) if drafts else None)
            if problem:
                return refuse(f"kv_cache_dtype: {spec.label}: {problem}", "cluster_rejected")
        view = self.hub.snapshot().cluster_views().get(cfg.cluster_name) or {}
        # WHAT `auto` WOULD PIN NOW for this selection — the number a Start at auto stores, from the same
        # published figures — and, asked for as the pool, the size every box below is estimated at.
        auto = kv_auto(cfg, self.catalog, view, spec, max_num_seqs=seqs, max_model_len=ctx,
                       max_num_batched_tokens=batched, num_speculative_tokens=int(drafts) if drafts else None,
                       ple_replicate=None if not ple_raw else ple_raw == "true")
        pin = auto.kv_cache_memory if pin_raw == KV_AUTO else (int(pin_raw) if pin_raw else None)
        # A PIN THE PROFILE CANNOT TAKE is refused here in the launch's words, with the maximum — the same refusal the
        # switch gives (`cluster._check_pin_ceiling`), while the knob can still be turned. `auto` is sized under it.
        if pin is not None and pin_raw != KV_AUTO:
            problem = spec.profile(CLUSTER_MODE).kv_pin_problem(pin)
            if problem:
                return refuse(f"kv_cache_memory: {spec.label}: {problem}", "cluster_rejected")
        # THE ENGINE'S OWN BLOCK COUNT, while the pair serves THIS model at THIS pin: its seats, not ours.
        engine_blocks = self._engine_blocks(cfg, view, spec, pin)
        try:                                             # what each box keeps free: the fleet's reserve needs MemTotal
            policies = {host.box: cfg.memory_policy(box_total(view, host.box)) for host in cfg.cluster_hosts}
        except ValueError as exc:
            return refuse(str(exc), "cluster_rejected")
        boxes = []
        for i, host in enumerate(cfg.cluster_hosts):
            role = RANK_ROLES[i] if i < len(RANK_ROLES) else RANK_ROLES[-1]
            # A RESTART REPLACES THE PAIR: while our rank serves on this box, the memory for the selection is
            # what the box had at the launch — else free now plus what the rank holds — never free now alone,
            # which refused the running selection itself (`available_basis`, `credited_gib` say which).
            free, basis, credited = restart_basis(cfg, self.catalog, view, host.box, role)
            boxes.append({**estimate_memory(
                spec, box=host.box, role=role,
                max_num_seqs=seqs, max_model_len=ctx, tp=len(cfg.cluster_hosts),
                available_gib=free,
                max_num_batched_tokens=batched,
                **policies[host.box],
                kv_cache_memory=pin, engine_blocks=engine_blocks,
                ple_replicate=(None if not ple_raw else ple_raw == "true"),
                speculative=int(drafts) > 0 if drafts else None, kv_cache_dtype=dtype).view(),
                "available_basis": basis, "credited_gib": credited,
                # Nothing the LB can wait for holds its RAM: the page refuses a Start this does not fit (`admit`).
                "at_rest": ((view.get("boxes") or {}).get(host.box) or {}).get("at_rest") is True})
        return h.send_json(200, {"model_choice": spec.id, "label": spec.label, "max_num_seqs": seqs,
                                 "max_model_len": ctx, "max_num_batched_tokens": batched,
                                 "kv_cache_memory": pin, "kv_cache_auto": pin_raw == KV_AUTO,
                                 "ple_replicate": (None if not ple_raw else ple_raw == "true"),
                                 "kv_cache_dtype": dtype, "auto": auto.view(), "boxes": boxes})

    @staticmethod
    def _switch_archs(cfg: Config) -> tuple:
        """The serving cards' arch facts of `cfg`'s cluster — what its switch checks a KV dtype against."""
        return tuple(c.type.arch.value for c in topology.serving_cards(cfg))

    def _engine_blocks(self, cfg: Config, view: dict, spec, pin) -> int | None:
        """`num_gpu_blocks` as the serving head reports it — only for the model and pin it runs, else None."""
        eff = view.get("effective") if isinstance(view.get("effective"), dict) else None
        if not (bool(view.get("held")) and eff is not None and eff.get("model_choice") == spec.id):
            return None
        profile = spec.profile(CLUSTER_MODE)
        want = pin if pin is not None else (profile.kv_cache_memory if profile is not None else None)
        if want is None or eff.get("kv_cache_memory") != want:
            return None
        snap = self.hub.snapshot()
        head = next((b for b in snap.backends if b.url == cfg.cluster_url), None)
        if head is None or not head.live:
            return None
        blocks = (self.hub.throughput(head, snap.now).get("kv_pool") or {}).get("blocks")
        return blocks if isinstance(blocks, int) and blocks > 0 else None

    #: How long a kill waits for its turn to END before it answers: the relay acts in milliseconds (the engine's
    #: connection shut wakes its read), so the page is told "killed" as a fact, not a hope. A turn still ending
    #: after this is a 202, and the page re-reads /lb/status.
    KILL_WAIT_S = 5.0

    def kill_request(self, h: _Handler, rid: str) -> None:
        """POST /lb/requests/<id>/kill `{}` — the operator's kill of ONE turn, the REQUESTS card's 1 s hold.

        ONE mechanism for every engine path (a classic box, the fallback, the cluster head — a pair, a local node,
        an observed node): the turn's own connection to its engine is closed (`_Turn.kill`), which is what makes
        vLLM and SGLang abort it and what wakes the relay; a turn with no connection — held by the LB, nothing sent
        — is woken where it waits. The client is told by the relay, in its API's shape (request_kill.py).

        Answers, in the LB's own `{"error": {type, message}}` for a refusal:
          * 200 `{"killed": {...}}` once the turn has ENDED as killed (≤ KILL_WAIT_S): its id, the state it was
            in, its age, client, session, what it had produced (content events, the engine's output count where
            given, prompt tokens, bytes), `by`, `outcome: "killed"`;
          * 200 with `already: true` for a turn killed before — IDEMPOTENT: a repeated POST (a retry, a second tab)
            is told the truth, not an error;
          * 202 `ending: true` when it has not ended within KILL_WAIT_S;
          * 404 `unknown_request` — no such id in flight or among the last RECENT_ROWS finished — and 404
            `request_finished` naming how it ended; 409 `not_killed` for one that finished on its own before the
            kill reached it, `not_killable` for a web_search the LB answers itself;
          * 415 for anything but `Content-Type: application/json` (the CSRF rule of every LB write route,
            `swarm_start`) and 400 for JSON that is not an object.

        Every kill is logged: the id, client and session, state and age, what was produced, and WHO — the gate's
        `X-Vllm-User` (GATE_USER_HEADER), or "an unnamed caller" without one — beside the address and User-Agent."""
        raw = h.read_body()
        ctype = h.headers.get("Content-Type")
        if not _is_json_type(ctype):
            return h.send_json(415, {"error": {
                "type": "unsupported_media_type",
                "message": "a kill is a POST with `Content-Type: application/json` and a body of {}; any other type "
                           "is one a cross-site page could send"}})
        if raw.strip() and not isinstance(_as_json(raw, ctype), dict):
            return h.send_json(400, {"error": {"type": "invalid_request",
                                               "message": "a kill's body is a JSON object ({}), or empty"}})
        who = (h.headers.get(GATE_USER_HEADER) or "").strip()[:request_kill.WHO_CHARS] or None
        with self._turns_cv:
            turn = next((t for t in self._turns if t.id == rid), None) if _TURN_ID.match(rid) else None
            done = (next((r for r in reversed(self._recent) if r.get("id") == rid), None)
                    if turn is None and _TURN_ID.match(rid) else None)
        if turn is None:
            if done is None:
                return h.send_json(404, {"error": {
                    "type": "unknown_request",
                    "message": f"no request {rid!r} is in flight here or among the last {request_queue.RECENT_ROWS} "
                               f"finished"}})
            if done.get("outcome") == request_queue.KILLED:
                return h.send_json(200, {"killed": {**request_kill.facts_of(done), "outcome": request_queue.KILLED,
                                                    "why": done.get("why"), "already": True,
                                                    "message": "already killed"}})
            why = f" ({done['why']})" if done.get("why") else ""
            status = f", HTTP {done['status']}" if done.get("status") is not None else ""
            return h.send_json(404, {"error": {
                "type": "request_finished",
                "message": f"request {rid} already finished: {done.get('outcome') or '?'}{why}{status}"}})
        if turn.backend == request_queue.WEBSEARCH:
            return h.send_json(409, {"error": {
                "type": "not_killable",
                "message": f"request {rid} is a web_search the gateway answers itself; no engine is working on it"}})
        state = self._state_of(turn)
        facts = request_kill.facts_of(request_queue.live_row(turn, time.monotonic(), state))
        if not turn.kill({"by": who, "why": request_kill.why(who, state, facts.get("age_s"))}):
            return h.send_json(200, {"killed": {**facts, "by": turn.killed.get("by"), "outcome": None,
                                                "already": True, "message": "already killed; still ending"}})
        log.warning("[proxy] kill: %s — %s", request_kill.describe(facts), request_kill.by_line(who, self._asked_by(h)))
        # A turn with no connection waits where it was held: wake it — the cluster hold on the hub's change, a claim
        # wait on its own poll, the drain's held on `_turns_cv`.
        self.hub.notify_change()
        with self._turns_cv:
            self._turns_cv.notify_all()
            deadline = time.monotonic() + self.KILL_WAIT_S
            while turn in self._turns and (left := deadline - time.monotonic()) > 0:
                self._turns_cv.wait(left)
            ending = turn in self._turns
            row = None if ending else next((r for r in reversed(self._recent) if r.get("id") == rid), None)
        if ending:
            return h.send_json(202, {"killed": {**facts, "by": who, "outcome": None, "already": False, "ending": True,
                                                "message": f"kill sent; still ending after {self.KILL_WAIT_S:.0f} s"}})
        outcome = (row or {}).get("outcome")
        if outcome != request_queue.KILLED:
            log.info("[proxy] kill: request %s had finished on its own before the kill reached it (%s)", rid, outcome)
            return h.send_json(409, {"error": {
                "type": "not_killed",
                "message": f"request {rid} finished on its own before the kill reached it: {outcome or '?'}"}})
        return h.send_json(200, {"killed": {**facts, "by": who, "outcome": request_queue.KILLED, "already": False,
                                            "message": "killed"}})

    def priority_request(self, h: _Handler, rid: str) -> None:
        """POST /lb/requests/<id>/priority `{"active": bool}` — the REQUESTS card's square: on = router.PRIORITY_GUI,
        off = none (the engine's default). Neither engine moves a request it holds, so a turn the client has had no
        byte of is cut from its engine and sent again (`_Turn.reprioritize`); answered:
          * 200 `{"priority": {id, active, value, applied}}` — `applied` "requeued" (sent again) or "set" (not sent yet);
          * 404 `unknown_request` — no such turn in flight; 409 `priority_locked` with why (router.priority_lock);
          * 415 for anything but `Content-Type: application/json` (the CSRF rule of every LB write route) and 400 for
            a body that is not `{"active": true|false}`."""
        raw = h.read_body()
        ctype = h.headers.get("Content-Type")
        if not _is_json_type(ctype):
            return h.send_json(415, {"error": {
                "type": "unsupported_media_type",
                "message": "a priority toggle is a POST with `Content-Type: application/json`; any other type is one a "
                           "cross-site page could send"}})
        body = _as_json(raw, ctype)
        if not isinstance(body, dict) or set(body) != {"active"} or not isinstance(body["active"], bool):
            return h.send_json(400, {"error": {"type": "invalid_request",
                                               "message": 'a priority toggle\'s body is {"active": true} or {"active": false}'}})
        with self._turns_cv:
            turn = next((t for t in self._turns if t.id == rid), None) if _TURN_ID.match(rid) else None
        if turn is None:
            return h.send_json(404, {"error": {"type": "unknown_request",
                                               "message": f"no request {rid!r} is in flight here"}})
        value = router.PRIORITY_GUI if body["active"] else None
        locked = router.priority_lock(turn.path, turn.committed)
        applied = None if locked else turn.reprioritize(value)
        if applied is None:
            return h.send_json(409, {"error": {"type": "priority_locked", "message": (
                f"request {rid}: " + (locked or router.priority_lock(turn.path, True)))}})
        who = (h.headers.get(GATE_USER_HEADER) or "").strip()[:request_kill.WHO_CHARS] or None
        log.info("[proxy] priority: request %s %s (%s) by %s", rid, "on" if body["active"] else "off", applied,
                 who or "an unnamed caller")
        return h.send_json(200, {"priority": {"id": rid, "active": body["active"], "value": value, "applied": applied}})

    @staticmethod
    def _log_requeue(h: _Handler, path: str, url: str, turn: _Turn) -> None:
        log.info("[proxy] %s %s → %s re-sent at priority %s (the operator's toggle); nothing had reached the client",
                 h.command, path, url, turn.priority)

    def _state_of(self, turn: _Turn) -> str:
        """The state a turn is in as the REQUESTS card draws it — its row's (the FCFS split included) — or, for one
        with no row, `held` (nothing sent), `finishing` (its stream has said its stop reason) or the relay's own."""
        row = next((r for r in self.queue_view()["inflight"] if r["id"] == turn.id), None)
        if row is not None:
            return row["state"]
        if turn.holding or turn.backend is None:
            return request_queue.HELD
        if turn.tap is not None and turn.tap.progress()["stop_reason"] is not None:
            return request_queue.FINISHING
        return request_queue.STREAMING if turn.first_at is not None else request_queue.SENT

    def _model_loaded(self) -> tuple[bool | None, str]:
        """Is there a model serving RIGHT NOW, for a swarm to put under load?

        The operator's rule, 2026-10-01: "benchmark run should not unload model, should use already
        loaded model". So this control never loads one, and with nothing serving the honest answer is
        to refuse rather than to start a session that would have nothing to talk to. A live cluster
        backend is the pair the swarm is about; a live classic vLLM counts too, because the LB would
        route the session's traffic there, and the words say which it is."""
        snap = self.hub.snapshot()
        # A cluster that answers another task (a speech-to-text node) is nothing a research session can talk to.
        tasks = {c.name: c.task for c in self.cfg.clusters}
        live = [b for b in snap.backends if b.live and (b.kind == VLLM or (
            b.kind == CLUSTER and tasks.get(getattr(b, "cluster", None), "chat") == "chat"))]
        if not live:
            # Several clusters: each one's state, by name; one: its state, as ever.
            views = snap.cluster_views()
            state = (", ".join(f"{n} {v.get('state')}" for n, v in views.items()) if len(views) > 1
                     else (next(iter(views.values())).get("state") if views else None))
            return False, (f"no vLLM is serving on this gateway"
                           f"{f' (the cluster is {state})' if state else ''}")
        pick = next((b for b in live if b.kind == CLUSTER), live[0])
        served = ", ".join(sorted(pick.served_models)) or "a model it did not name"
        # Named from the fleet (`topology.named`), never a topology written here: "the two-box cluster" was said
        # of a one-GPU node on site-b's gateway (2026-10-11).
        one = self.cfg.for_cluster(pick.cluster) if len(self.cfg.clusters) > 1 and getattr(pick, "cluster", None) \
            else self.cfg
        where = topology.named(one) if pick.kind == CLUSTER else f"{pick.host_label or pick.url}"
        return True, f"{where} is serving {served}"

    def swarm_start(self, h: _Handler) -> None:
        """POST /lb/swarm/start `{cap_min?}` — leave a request for ONE deep-research swarm.

        This starts nothing, and it changes nothing about the cluster: no restart, no setting, no
        model loaded or unloaded. The LB has no way to run a process on box-c, so it writes one
        small file in its own state directory and `lb/bench_queue_runner.py` performs it — that
        runner refuses while a swarm or the matrix is already up, and launches `lb/bench_session.py`,
        which is guarded four ways against ever running on the operator's own subscription. The 202
        carries `request_pending: true`, and the page says "requested" until the runner names this
        request in its own report.

        `cap_min` is HOW LONG, in minutes, and it is the one argument this control takes (the operator,
        2026-10-05: keep 15 the default, allow 30 or more). Left out or null it is the host runner's own
        default — 15 min, and then the request file is byte for byte the one this endpoint has always
        written — and out of range it is a 400 naming the field and its range, the same way /lb/cluster
        refuses a knob. The range is argued from measurements in `bench_session.CAP_MIN_RANGE`.

        There is no endpoint for the thirty-run matrix. It tears the pair down and rebuilds it per
        configuration for thirteen hours — on 2026-10-01 a click did exactly that to a working
        default — so it lives at the keyboard: `bench_queue_runner.py --start-matrix`."""
        raw = h.read_body()
        # A START IS JSON, OR IT IS NOTHING (CSRF). A browser sends a "simple" POST — no type, text/plain, a
        # form — to any origin without asking first, with the gateway's cached basic-auth credentials
        # attached, so any of those shapes is one a page on another site can fire here. `application/json`
        # makes the browser ask first (a preflight), and this gateway grants no cross-origin access. The
        # page has always sent JSON; the cluster switches already demand it (`_read_switch`).
        if not _is_json_type(h.headers.get("Content-Type")):
            return h.send_json(415, {"error": {
                "type": "unsupported_media_type",
                "message": "a swarm start is a POST with `Content-Type: application/json` (an empty body, or "
                           "{\"cap_min\": <minutes>}); any other type is one a cross-site page could send"}})
        body = _as_json(raw, h.headers.get("Content-Type"))
        # AN EMPTY JSON START IS STILL A START. Until 2026-10-05 this endpoint took no arguments, and the
        # page sends an empty object for the host runner's own default; a 400 for it would be this change
        # breaking the button it extends.
        cap = body.get("cap_min") if isinstance(body, dict) else None
        lo, hi = CAP_MIN_RANGE
        if cap is not None and (isinstance(cap, bool) or not isinstance(cap, (int, float))
                                or not lo <= float(cap) <= hi):
            return h.send_json(400, {"error": {
                "type": "invalid_request",
                "message": f"`cap_min` must be a number of minutes from {lo:g} to {hi:g}, or null for "
                           f"the host runner's own default"}})
        if not self.swarm.configured:
            return h.send_json(409, {"error": {
                "type": "swarm_rejected",
                "message": "no swarm state directory is configured on this gateway"}})
        try:
            view = self.swarm.request_session(cap)
        # Two at once would make both measurements meaningless, and a swarm with nothing loaded has
        # nothing to load: the caller is told which, in the words the page shows.
        except SwarmRefused as exc:
            return h.send_json(409, {"error": {"type": "swarm_rejected", "message": str(exc)}})
        # `benchswarm.check_cap` again, which this method has already made: unreachable while the two
        # agree, and a 400 rather than a 500 on the day one of them is changed alone.
        except ValueError as exc:
            return h.send_json(400, {"error": {"type": "invalid_request", "message": str(exc)}})
        # `SwarmUnavailable` is an OSError: a state directory that cannot be written is this
        # deployment's fault, not the caller's, and the reason names the path so it can be fixed.
        except OSError as exc:
            log.error("[proxy] the swarm request was not written: %s", exc)
            return h.send_json(500, {"error": {
                "type": "swarm_not_written",
                "message": f"the swarm request was not written to {self.cfg.swarm_state}: {exc}"}})
        # It puts the pair under a measured stretch of real load on the operator's own hardware, so like
        # the cluster switch its line names the caller — and the minutes, because since 2026-10-05 a
        # click can ask for four hours of it and the log is where that is read back afterwards.
        log.info("[proxy] swarm: one deep-research session requested, cap %s — %s",
                 f"{float(cap):g} min" if cap is not None else f"the runner's own ({view.get('cap_min')})",
                 self._asked_by(h))
        return h.send_json(202, {"swarm": view})

    def answer_web_search(self, h: _Handler, body: dict) -> None:
        t0 = time.monotonic()
        query = websearch.extract_query(body)
        model = body.get("model") or ""                 # the request's own word, echoed; none stands in
        if not query:
            outcome = websearch.SearchOutcome.failure(websearch.INVALID_TOOL_INPUT)
        else:
            try:
                outcome = self.search(query)
            except Exception as exc:                      # any failure → the client's own error
                log.warning("[websearch] %r failed (%s: %s)", query, type(exc).__name__, exc)
                outcome = websearch.SearchOutcome.failure(websearch.UNAVAILABLE)
        msg = websearch.build_message(query or "", outcome, model=model)
        # `→ 0 results` on its own hid a four-day outage: while nothing comes back, the line
        # names the engines that did not answer, in SearXNG's own words.
        log.info("[websearch] %r → %s %.0fms", query, outcome.detail(),
                 (time.monotonic() - t0) * 1000)
        if not body.get("stream"):
            return h.send_json(200, msg)
        payload = websearch.sse_body(msg)
        h.send_response(200)
        h.send_header("Content-Type", "text/event-stream; charset=utf-8")
        h.send_header("Cache-Control", "no-cache")
        h.send_header("Content-Length", str(len(payload)))
        h.send_header("Connection", "close")
        h.send_header("X-LB-Backend", "websearch")
        h.end_headers()
        if h.command != "HEAD":
            h.wfile.write(payload)
        h.wfile.flush()

    def _account(self, url: str, tap: UsageTap | None, *, client_gone: bool) -> None:
        """Exactly one `hub.note_usage` per relayed inference response, after its last byte went out.
        The tap never raises by contract; this guard is for everything after it, because an exception
        escaping the relay's `finally` would reach `_dispatch`, which writes its own 500 JSON after bytes
        the client already has — a changed response, the one thing accounting must never cause."""
        if tap is None:
            return
        try:
            self.hub.note_usage(url, tap.finish(client_gone=client_gone))
        except Exception:
            log.exception("[proxy] usage accounting failed for %s; the response was relayed intact", url)

    def _end_inference(self, url: str, tap: UsageTap | None, *, client_gone: bool) -> None:
        """The other half of `note_request_start`, for EVERY path that started a turn — the relay's `finally`
        and the first-byte-timeout 504 alike. The 504 used to return between the two and leave the backend
        counted as busy for the life of the process, which biases the router and the card against a backend
        that is in fact idle. `_account` ignores a tap that was never made."""
        self.hub.note_request_end(url)
        self._account(url, tap, client_gone=client_gone)

    def _await_claim(self, snap, excluded: set[str], turn: _Turn, route: "_Route" = None):
        """A turn nothing holds WAITS for the claim on its way, up to `admission_wait_s` — only
        while a claim is all that is missing: something we could route to is live (the router's
        counterfactual) and the broker is reachable. Its refusal was already counted as demand,
        which is what gets the lease asked for; this only stops that refusal reaching the client.
        A kill ends the wait on its next look (the hold after it answers it).

        Never with the broker off: no claim is ever on its way."""
        route = route or _Route()
        if (not self.cfg.broker_enabled or self.cfg.admission_wait_s <= 0 or not snap.db_ok
                or router.choose(snap, self.cfg, require_lease=False, model=route.model, exact=route.exact,
                                         task=route.task) is None):
            return None
        deadline = time.monotonic() + self.cfg.admission_wait_s
        while (left := deadline - time.monotonic()) > 0 and turn.killed is None:
            time.sleep(min(_CLAIM_POLL_S, left))
            snap = self.hub.snapshot()
            backend = router.choose(snap, self.cfg, exclude=frozenset(excluded), model=route.model, exact=route.exact,
                                    task=route.task)
            if backend is not None or not snap.db_ok:
                return backend
        return None

    # ── the hold: a turn waits out a planned cluster switch ────────────────

    def _cluster_boxes(self, backend) -> str | None:
        """What X-LB-Box says for a cluster head: it has no single box, so it names ITS cluster's boxes, head first
        (`hub.cluster_cfgs[backend.cluster]`). None for any other backend."""
        if backend.kind != CLUSTER:
            return None
        cfg = self.hub.cluster_cfgs.get(backend.cluster)
        return " + ".join(h.box for h in cfg.cluster_hosts) if cfg is not None else None

    def _switch_for(self, name: str | None):
        """Cluster `name`'s switch (`cluster.DesiredStore`) — the run's, by name, when this proxy has a run; else the
        one switch it was built with (tests that build a proxy alone). None for a cluster with no switch."""
        if self.run is not None:
            return self.run.switches.get(name) if name is not None else None
        return self.cluster_switch

    def _stored_mode(self, view: dict) -> str | None:
        """The switch on disk, but only for a view with no `desired` yet — the seconds before the cluster loop's
        first tick. `DesiredStore.load` never raises. The view's OWN cluster's switch (`name`)."""
        switch = self._switch_for(view.get("name"))
        if view.get("desired") or switch is None:
            return None
        return switch.load().mode

    def _hold_view(self, snap, route: "_Route") -> dict:
        """The cluster view a turn waits on: of the clusters that could serve it — those serving its model
        (`router.serving`), or, for a model none serves yet, every active cluster not already serving another (any
        of those may be the one bringing it up) — one on its way (`cluster_coming`), else one that needs the
        operator, whose words the 503 carries, else any, else `{}`. Each backend's OWN view; asked afresh on every wake, so a second cluster still coming is
        waited for after the first fails. Never the singular alias, which names no cluster once two run."""
        route = route or _Route()
        heads = [b for b in snap.backends if b.kind == CLUSTER and not (b.live and b.served_models)]
        views = [snap.cluster_view_of(b) for b in
                 ([b for b in router.serving(snap, self.cfg, route.model, exact=route.exact, task=route.task) if b.kind == CLUSTER]
                  or heads)]
        coming = [v for v in views if cluster_coming(v, self._stored_mode(v))]
        return (coming or [v for v in views if v.get("state") in CLUSTER_NEEDS_OPERATOR] or views or [{}])[0]

    def _nothing_coming(self, snap, view: dict) -> _Refusal:
        """The 503 for a turn nothing is on its way for. Not worth a retry storm — `x-should-retry: false` — when
        the cluster needs the operator, or when no backend is live at all; worth one when something live only
        lacks its claim (a lease asked for, seconds away)."""
        desired = (view.get("desired") or {}).get("mode") or self._stored_mode(view)
        state = view.get("state")
        if view.get("configured") and desired == CLUSTER_MODE and state in CLUSTER_NEEDS_OPERATOR:
            return _Refusal(f"the cluster is {state}: {view.get('detail') or 'no detail'} — it needs the operator "
                            f"(Stop, then Start), and nothing else is live", self.cfg.retry_after_s, retry=False)
        return _Refusal(self._none_live(), self.cfg.retry_after_s, retry=any(b.live for b in snap.backends))

    def _none_live(self) -> str:
        """The 503's words when nothing is live: with the broker on, the lease that refusal has asked for; with it
        off there is no lease, and nothing is said to be on its way."""
        if self.cfg.broker_enabled:
            return ("no vLLM or ollama backend is live right now; a lease has been requested from "
                    "broker_parrot — retry shortly")
        return "no vLLM or ollama backend is live right now — retry shortly"

    def _hold_for_cluster(self, h: _Handler, turn: _Turn, excluded: set[str], route: "_Route" = None):
        """A turn that found nothing live WAITS for the pair while the operator wants it and it is on its way
        (`cluster_coming`) — no byte sent, so the client simply waits (Claude Code: up to API_TIMEOUT_MS for
        headers, with no banner) — on the hub's change condition, not a poll. Returns `(backend, None)` the
        moment the router has one, `(None, _DRAINING)` if a drain began, else `(None, _Refusal)`: the start
        failed or was blocked, the cluster was switched off, `cluster_hold_s` ran out, or `hold_max` turns were
        already waiting.

        158 Claude Code errors on 10-01/10-02 ("no vLLM or ollama backend is live") were all inside a restart or
        a switch — 2-10 min that end on their own — answered 503 at once, with no Retry-After."""
        cfg = self.cfg
        route = route or _Route()
        snap = self.hub.snapshot()
        view = self._hold_view(snap, route)
        if not cluster_coming(view, self._stored_mode(view)):
            return None, self._nothing_coming(snap, view)
        state = view.get("state") or "unknown"
        if cfg.cluster_hold_s <= 0:
            return None, _Refusal(f"the cluster is {state} ({view.get('detail') or 'no detail'}); this gateway "
                                  f"holds no request for it (LB_CLUSTER_HOLD_S=0)", cfg.retry_after_s)
        with self._turns_cv:
            if self._draining:
                # Admitted before the drain, so counted in flight: leave the count as it starts waiting for the
                # drain's end — or the two wait for each other until the window runs out.
                turn.holding = True
                self._turns_cv.notify_all()
                return None, _DRAINING
            full = sum(1 for t in self._turns if t.holding) >= cfg.hold_max
            if not full:
                turn.holding = True
        if full:
            log.warning("[proxy] %s %s: %d request(s) are already held for the cluster (%s) — refused",
                        h.command, turn.path, cfg.hold_max, state)
            return None, _Refusal(f"{cfg.hold_max} requests are already waiting for the cluster, which is {state} "
                                  f"({view.get('detail') or 'no detail'}); retry shortly", cfg.retry_after_s)
        t0 = time.monotonic()
        deadline = t0 + cfg.cluster_hold_s
        log.info("[proxy] %s %s: nothing is live while the cluster is %s (%s) — holding it up to %.0f s for the pair",
                 h.command, turn.path, state, view.get("detail") or "no detail", cfg.cluster_hold_s)
        while True:
            seen = self.hub.change_version()      # BEFORE the snapshot: a change in between is not slept through
            snap = self.hub.snapshot()
            view = self._hold_view(snap, route)
            if turn.killed is not None:           # read after `seen`: the kill's own change wakes the wait below
                log.info("[proxy] %s %s: held %.0f s — killed by the operator, never sent", h.command, turn.path,
                         time.monotonic() - t0)
                return None, _KILLED
            with self._turns_cv:
                if self._draining:
                    return None, _DRAINING        # still holding: the drain does not wait for it
                backend = router.choose(snap, cfg, exclude=frozenset(excluded), model=route.model, exact=route.exact,
                                    task=route.task)
                if backend is not None:
                    turn.holding = False          # in flight from here on: a drain waits for it
            if backend is not None:
                log.info("[proxy] %s %s: held %.0f s — the cluster serves, sent to %s",
                         h.command, turn.path, time.monotonic() - t0, backend.url)
                return backend, None
            if not cluster_coming(view, self._stored_mode(view)):
                log.warning("[proxy] %s %s: held %.0f s, released — the cluster is %s (%s)", h.command, turn.path,
                            time.monotonic() - t0, view.get("state"), view.get("detail") or "no detail")
                return None, self._nothing_coming(snap, view)
            left = deadline - time.monotonic()
            if left <= 0:
                state = view.get("state") or "unknown"
                log.warning("[proxy] %s %s: held %.0f s, the hold limit — the cluster is still %s (%s)",
                            h.command, turn.path, cfg.cluster_hold_s, state, view.get("detail") or "no detail")
                return None, _Refusal(f"waited {cfg.cluster_hold_s:.0f} s for the cluster, which is still {state} "
                                      f"({view.get('detail') or 'no detail'}); retry shortly", cfg.retry_after_s)
            self.hub.wait_for_change(seen, min(left, _HOLD_RECHECK_S))

    @staticmethod
    def _count_prompt(turn: _Turn, url: str, body: dict) -> None:
        """An OpenAI chat turn's prompt tokens, counted by the engine it goes to while it queues (prompt_count.py)."""
        request = prompt_count.request_of(body)
        if request is not None:
            threading.Thread(target=lambda: setattr(turn, "counted", prompt_count.count(url, request)),
                             name=f"prompt-count-{turn.id}", daemon=True).start()

    def forward(self, h: _Handler, path: str, body: bytes, parsed=_UNPARSED, turn: _Turn | None = None) -> None:
        t0 = time.monotonic()
        inference = h.command == "POST"
        if inference:
            self.hub.note_request()
        if turn is None:
            turn = _Turn(h)                      # called directly: tracked for its own hold, booked nowhere
        refusal: _Refusal | None = None

        # Who is signed in is the gate's word to THIS process, never an engine's business (GATE_USER_HEADER).
        headers = {k: v for k, v in h.headers.items()
                   if k.lower() not in HOP_BY_HOP and k.lower() not in ("host", "content-length", _GATE_USER)}
        headers["X-Forwarded-For"] = h.client_address[0]
        headers["X-Forwarded-Proto"] = "http"
        if parsed is _UNPARSED:
            parsed = _as_json(body, h.headers.get("Content-Type"))
        # THE CALLER'S PRIORITY (router.PRIORITY_HEADER): the engine's own `priority` field, or refused before any
        # backend is chosen where the engine would drop it — never forwarded to be ignored.
        if inference:
            try:
                turn.priority = router.priority_of(h.headers.get(router.PRIORITY_HEADER),
                                                   urllib.parse.urlsplit(path).path, parsed)
            except ValueError as exc:
                return h.send_json(400, {"error": {"type": "invalid_priority", "message": str(exc)}})

        tried: list[str] = []
        excluded: set[str] = set()
        # THE MODEL THE REQUEST NAMES decides which clusters may answer it (router.serving): every routing decision of
        # this request — the first choice, the claim wait, the hold — is asked with it, and a catalog model no cluster
        # serves is refused here, before any engine is chosen (router.eligible), never rewritten onto another's.
        model = parsed.get("model") if inference and isinstance(parsed, dict) else None
        task = router.task_of_path(urllib.parse.urlsplit(path).path) if inference else router.CHAT
        if task != router.CHAT:
            # AN UPLOAD: its model is the form's field — routed by as a chat body's is (`router.serving`), then set to
            # what the chosen backend serves (`multipart.set_field`) — so a body that is no form is refused here.
            try:
                model = multipart.get_field(body, h.headers.get("Content-Type"), "model")
            except ValueError as exc:
                return h.send_json(400, {"error": {"type": "invalid_request", "message": (
                    f"{urllib.parse.urlsplit(path).path} takes a multipart/form-data upload: {exc}")}})
        route = _Route(model if isinstance(model, str) else None, model in self._exact_models[task], task)
        while True:
            # KILLED with no connection open — before the first backend, or between two: answered here, never sent.
            if turn.killed is not None:
                return self._answer_killed(h, turn)
            snap = self.hub.snapshot()
            backend = router.choose(snap, self.cfg, exclude=frozenset(excluded), model=route.model, exact=route.exact,
                                    task=route.task)
            if inference and not tried:
                # STARVED ADMISSION, decided on the first choice for this request: nothing
                # leased (shape 1), or settling for a demoted vLLM while the broker reports a box
                # that could take a lease we do not hold (shape 2). Our own queueing is invisible
                # everywhere else, so this is where it is counted, and the scaler turns it into a
                # lease request.
                held = sum(1 for l in snap.leases
                           if l.status == LEASE_RUNNING and l.kind == VLLM)
                capacity = (snap.vllm_capacity if snap.vllm_capacity is not None
                            else len(snap.vllm_backends()))
                if backend is None:
                    self.hub.note_starved()
                    backend = self._await_claim(snap, excluded, turn, route)
                    if backend is None:
                        backend, refusal = self._hold_for_cluster(h, turn, excluded, route)
                        if refusal is _DRAINING:
                            return self._wait_out_drain(h, turn)
                        if refusal is _KILLED or turn.killed is not None:
                            return self._answer_killed(h, turn)
                elif capacity > held and (
                        backend.waiting + max(backend.running, backend.inflight) > self.cfg.vllm_max_depth):
                    self.hub.note_starved()
            if backend is None:
                break
            # …and refused where the CHOSEN engine would drop it (TensorFold reads only "background"): never sent.
            dropped = (router.priority_refusal(engine_of_backend(snap, backend, self.catalog))
                       if turn.priority is not None else "")
            if dropped:
                return h.send_json(400, {"error": {"type": "invalid_priority", "message": dropped}})
            turn.backend = backend.url
            turn.box = backend.box or self._cluster_boxes(backend)
            turn.sent_at = time.monotonic()
            out_body = body
            if parsed is not None:
                rewritten = router.rewrite_effort(parsed, backend, self.cfg)
                rewritten = router.rewrite_model(rewritten, backend, self.cfg)
                rewritten = router.rewrite_priority(rewritten, turn.priority)
                if rewritten is not parsed:
                    out_body = json.dumps(rewritten).encode()
                if inference and urllib.parse.urlsplit(path).path == "/v1/chat/completions":
                    self._count_prompt(turn, backend.url, rewritten)
            elif route.task != router.CHAT and router.served_by(backend, self.cfg):
                out_body = multipart.set_field(body, h.headers.get("Content-Type"), "model",
                                               router.served_by(backend, self.cfg)[0])
            netloc = urllib.parse.urlsplit(backend.url).netloc
            conn = http.client.HTTPConnection(
                urllib.parse.urlsplit(backend.url).hostname,
                urllib.parse.urlsplit(backend.url).port or 80,
                timeout=self.cfg.connect_timeout_s)
            body_delivered = False
            upstream: socket.socket | None = None
            try:
                conn.connect()
                upstream = conn.sock
                turn.attach(upstream)             # what a kill closes from here on — at once, if one came first
                try:
                    upstream_keepalive(upstream)  # never a reason to fail a request
                except OSError:
                    pass
                # Waiting for the first byte covers queueing, model load and prefill.
                conn.sock.settimeout(self.cfg.first_byte_timeout_s)
                conn.putrequest(h.command, path, skip_host=True, skip_accept_encoding=True)
                conn.putheader("Host", netloc)
                for k, v in headers.items():
                    conn.putheader(k, v)
                if out_body or h.command in ("POST", "PUT", "PATCH"):
                    conn.putheader("Content-Length", str(len(out_body)))
                conn.endheaders(out_body or None)
                body_delivered = True
                resp = conn.getresponse()
            except (OSError, http.client.HTTPException) as exc:
                conn.close()
                turn.detach()
                # THE OPERATOR closed it (`_Turn.kill`), not the backend: no penalty, no failover — the turn is
                # answered killed, and the engine has seen its client go.
                if turn.killed is not None:
                    log.info("[proxy] %s %s → %s killed by the operator before an answer began (%s); the engine's "
                             "connection is closed", h.command, path, backend.url, type(exc).__name__)
                    return self._answer_killed(h, turn)
                if turn.take_requeue():
                    self._log_requeue(h, path, backend.url, turn)
                    continue
                err = f"{type(exc).__name__}: {exc}"
                self.hub.penalise(backend.url, self.cfg.penalty_s, err)
                # WHETHER THE REQUEST WAS DELIVERED decides this, never the exception type.
                # `socket.timeout` IS `TimeoutError` on 3.10+, so keying on the type cannot
                # tell a dropped SYN (nothing sent — a firewalled box, a blackholing switch)
                # from a backend that took the whole request and went quiet. They need
                # opposite handling, and getting it wrong strands the client on a 504 while a
                # healthy server sits unused.
                if body_delivered:
                    if isinstance(exc, TimeoutError):
                        log.error("[proxy] %s %s → %s took the request and produced nothing in "
                                  "%.0fs; giving up", h.command, path, backend.url,
                                  self.cfg.first_byte_timeout_s)
                        return h.send_json(504, {"error": {
                            "type": "backend_timeout", "backend": backend.url,
                            "message": f"no response started within "
                                       f"{self.cfg.first_byte_timeout_s:.0f}s (queueing, model "
                                       f"load and prefill are included in that budget)"}})
                    # The backend took the whole request and then hung up: it may already
                    # be generating. Sending the same completion elsewhere could run it
                    # twice, so this is the client's decision, not ours.
                    log.warning("[proxy] %s %s → %s dropped the request after taking it (%s); "
                                "not re-sent", h.command, path, backend.url, err)
                    return h.send_json(502, {"error": {
                        "type": "backend_dropped_request", "backend": backend.url,
                        "message": "the backend accepted the request and hung up before "
                                   "answering; it was not re-sent elsewhere — retry if safe"}})
                # Nothing — or only part — of the request reached it. A partial request starts
                # no inference, so trying the next backend cannot run a completion twice.
                log.warning("[proxy] %s %s → %s failed before the request was delivered (%s); "
                            "failing over", h.command, path, backend.url, err)
                tried.append(backend.url)
                excluded.add(backend.url)
                continue

            # The verdict is the STATUS LINE, so record it here rather than after the relay:
            # a backend that answers instantly and always 5xxes is broken, not busy, and left
            # unchecked it keeps winning the route while a loaded model sits at 0% GPU and a
            # healthy server goes unused — the shape the operator caught on box-a. Penalising
            # is enough: the router prefers anyone else but will still pick a penalised backend
            # when it is all there is, and the penalty lapses so recovery needs no intervention.
            streak = self.hub.note_response(backend.url, resp.status)
            if streak >= self.cfg.failure_streak_max:
                log.error("[proxy] %s has failed %d requests in a row (last %s) — penalising it "
                          "for %.0fs so traffic goes elsewhere", backend.url, streak,
                          resp.status, self.cfg.penalty_s)
                self.hub.penalise(backend.url, self.cfg.penalty_s,
                                  f"{streak} consecutive {resp.status}s")

            # A STATUS LINE IS NOT A COMMITMENT — but the wait that follows it is the real
            # time-to-first-token. vLLM emits its status line at HTTP ADMISSION, before engine
            # queueing, so the queueing/prefill that `first_byte_timeout_s` was written for
            # happens HERE, after the headers. Budget the first BODY byte with it; every read
            # after that is an inactivity budget and gets `stall_timeout_s`.
            if inference:
                self.hub.note_request_start(backend.url)   # covers TTFT, not just the stream

            first = b""
            try:
                if h.command != "HEAD" and resp.status not in (204, 304):
                    try:
                        conn.sock.settimeout(self.cfg.first_byte_timeout_s)
                    except (AttributeError, OSError):
                        pass
                    try:
                        first = resp.read1(_CHUNK)
                        if (inference and backend.kind == VLLM
                                and time.monotonic() - t0 > self.cfg.starve_after_s):
                            self.hub.note_starved()   # shape 3: waited too long to reach inference
                    except (OSError, http.client.HTTPException) as exc:
                        # KILLED while queued or prefilling at the engine — its status line in, no body yet, and
                        # nothing relayed: the client gets the kill as an HTTP error; nothing is penalised.
                        if turn.killed is not None:
                            conn.close()
                            turn.detach()
                            log.info("[proxy] %s %s → %s killed by the operator before its first byte (%s); the "
                                     "engine's connection is closed", h.command, path, backend.url, type(exc).__name__)
                            if inference:
                                self._end_inference(backend.url, None, client_gone=False)
                            return self._answer_killed(h, turn)
                        if turn.take_requeue():
                            conn.close()
                            turn.detach()
                            if inference:
                                self._end_inference(backend.url, None, client_gone=False)
                            self._log_requeue(h, path, backend.url, turn)
                            continue
                        # NEVER RE-SENT. The backend has the whole request and may already be
                        # generating from it; handing the same completion to a second backend
                        # would run it twice on a shared fleet. One honest error beats that, and
                        # it is the same policy as the hang-up case below — one rule, not two.
                        err = f"{type(exc).__name__}: {exc}"
                        self.hub.penalise(backend.url, self.cfg.penalty_s, err)
                        conn.close()
                        log.error("[proxy] %s %s → %s answered %s then produced no body in "
                                  "%.0fs (%s); not re-sent", h.command, path, backend.url,
                                  resp.status, self.cfg.first_byte_timeout_s, err)
                        if inference:
                            self._end_inference(backend.url, None, client_gone=False)
                        return h.send_json(504, {"error": {
                            "type": "backend_timeout", "backend": backend.url,
                            "message": f"the backend accepted the request and answered "
                                       f"{resp.status}, then produced no output within "
                                       f"{self.cfg.first_byte_timeout_s:.0f}s; it was NOT "
                                       f"re-sent elsewhere (it may already be generating) — "
                                       f"retry if safe"}})
                    if not first and turn.killed is not None:
                        # A body with no length, ended by the kill's own close: the same kill, not an empty answer.
                        conn.close()
                        turn.detach()
                        if inference:
                            self._end_inference(backend.url, None, client_gone=False)
                        return self._answer_killed(h, turn)
                try:
                    conn.sock.settimeout(self.cfg.stall_timeout_s)
                except (AttributeError, OSError):
                    pass
            finally:
                pass
            # THE COMMIT: the client is given the answer's first byte next. A toggle that closed the connection
            # before it — the read ended, or raced in after it — means nothing of this answer reaches the client:
            # the turn is sent again at its new priority.
            if not turn.commit():
                conn.close()
                turn.detach()
                if inference:
                    self._end_inference(backend.url, None, client_gone=False)
                self._log_requeue(h, path, backend.url, turn)
                continue

            sent = 0
            # ACCOUNTING (worklog/vllm-gateway-throughput.md, "Proxy token accounting"): one UsageTap per
            # inference response, fed each chunk only AFTER the client has it, asked once in `finally`.
            tap = UsageTap(path, resp.status, resp.getheader("Content-Type")) if inference else None
            writing = False     # a write to the CLIENT is in flight: if it fails, the client is gone
            # AN SSE STREAM is the one shape the relay may add bytes to — only at an event boundary (`tail`):
            # a `: keepalive` comment while the backend is silent, an `overloaded_error` event if it fails.
            sse = (inference and resp.status == 200
                   and "text/event-stream" in (resp.getheader("Content-Type") or "").lower())
            anthropic_sse = sse and urllib.parse.urlsplit(path).path == "/v1/messages"
            turn.tap, turn.anthropic_sse = tap, anthropic_sse
            pump: _Pump | None = None
            tail = b""          # the last bytes the client was given
            ended = False       # the stream's own `message_stop` went out: the answer is whole
            done_sent = False   # an OpenAI stream's own `[DONE]` went out: so did its whole answer
            keepalives = 0
            try:
                h.relaying = True                 # the backend's headers, not the LB's own
                writing = True
                h.send_response(resp.status, resp.reason)
                for k, v in resp.getheaders():
                    if k.lower() not in HOP_BY_HOP:
                        h.send_header(k, v)
                h.send_header("Connection", "close")
                h.send_header("X-LB-Backend", backend.url)
                # WHICH BOX, not just which URL: 192.0.2.12 means box-b only to someone
                # holding the topology in their head. The UI stamps every answer with this.
                h.send_header("X-LB-Kind", backend.kind)
                box = backend.box or self._cluster_boxes(backend)
                if box:
                    h.send_header("X-LB-Box", box)
                if backend.host_label:
                    h.send_header("X-LB-Host-Label", backend.host_label)
                h.end_headers()
                writing = False
                if h.command != "HEAD":
                    # Claude Code's stream only: its banner and byte watchdog are what silence trips, while
                    # a web chat client reads 300 s of silence on /v1/chat/completions as a stuck backend.
                    if anthropic_sse and first and self.cfg.sse_keepalive_s > 0:
                        pump = _Pump(resp, self.cfg.sse_keepalive_s)
                    chunk = first
                    while chunk:
                        writing = True
                        h.wfile.write(chunk)
                        h.wfile.flush()
                        writing = False
                        sent += len(chunk)
                        turn.sent = sent
                        turn.last_at = time.monotonic()
                        if turn.first_at is None:
                            turn.first_at = turn.last_at
                        ended = ended or (anthropic_sse and b"message_stop" in tail + chunk)
                        done_sent = done_sent or (sse and b"data: [DONE]" in tail + chunk)
                        tail = (tail + chunk)[-16:]
                        if tap is not None:
                            tap.feed(chunk)       # after the client has it: never delays a byte
                        if pump is None:
                            chunk = resp.read1(_CHUNK)
                            continue
                        # The backend's silence is filled for the CLIENT only: never fed to the tap, never
                        # counted in `sent`, and never progress — the pump's read still times out at
                        # `stall_timeout_s` of the BACKEND's silence, whatever was written meanwhile.
                        while (chunk := pump.next()) is _SILENT:
                            if _at_boundary(tail):
                                writing = True
                                h.wfile.write(KEEPALIVE_COMMENT)
                                h.wfile.flush()
                                writing = False
                                keepalives += 1
                                turn.keepalives = keepalives
                    # A body with no length reads its end as EOF, and so does a kill's close: only the stream's own
                    # end marker tells a whole answer from a killed one.
                    if turn.killed is not None and not (ended or done_sent):
                        self._killed_mid_stream(h, turn, sse=sse, sent=sent, tail=tail)
            except (OSError, http.client.HTTPException) as exc:
                stalled = isinstance(exc, TimeoutError)
                if not writing and turn.killed is not None:
                    # THE OPERATOR closed the engine's connection (`_Turn.kill`): not the backend's failure, and
                    # never the `overloaded_error` below — that one the client retries.
                    log.info("[proxy] %s %s → %s killed by the operator after %d bytes; the engine's connection "
                             "is closed", h.command, path, backend.url, sent)
                    self._killed_mid_stream(h, turn, sse=sse, sent=sent, tail=tail)
                else:
                    if writing:
                        turn.client_gone = True
                    else:
                        turn.error = "backend went silent" if stalled else "backend failed mid-stream"
                    log.warning("[proxy] %s %s → %s %s after %d bytes (%s); not retried",
                                h.command, path, backend.url,
                                f"went silent for {self.cfg.stall_timeout_s:.0f}s" if stalled
                                else "failed mid-stream", sent, exc)
                    # THE BACKEND failed (not the client: `writing` is False) at an event boundary of an Anthropic
                    # stream: say so in the stream's own words. `overloaded_error` is what the client retries; a
                    # stream that just stops is "Connection lost mid-response". Inside an event nothing may be
                    # added — the stream is cut as before; and after `message_stop` the answer is whole, so an
                    # error there would make the client throw a finished turn away and run it again.
                    if anthropic_sse and not writing and sent and not ended and _at_boundary(tail):
                        try:
                            h.wfile.write(_sse_error(
                                f"the backend {'went silent' if stalled else 'failed'} mid-stream after {sent} "
                                f"bytes ({type(exc).__name__}); this answer is incomplete — retry"))
                            h.wfile.flush()
                        except OSError:
                            pass
            finally:
                turn.detach()
                turn.ended = ended
                h.relaying = False
                if pump is not None:
                    pump.close(upstream)          # wakes a read still parked on the backend
                conn.close()
                if inference:
                    self._end_inference(backend.url, tap, client_gone=writing)
            upstream_err = ""
            if resp.status >= 500:
                snippet = first[:_ERR_BODY_MAX].decode("utf-8", "replace").replace("\n", " ").strip()
                if snippet:
                    upstream_err = f" upstream={snippet}"
            log.info("[proxy] %s %s → %s %s %dB %.0fms%s%s", h.command, path, backend.url,
                     resp.status, sent, (time.monotonic() - t0) * 1000,
                     f" ({keepalives} keepalive(s) through the backend's silence)" if keepalives else "",
                     upstream_err)
            return

        if tried:
            log.error("[proxy] %s %s: every candidate failed: %s", h.command, path, tried)
            return h.send_json(502, {"error": {
                "type": "all_backends_failed", "tried": tried,
                "message": "no live backend could be reached (refused, unreachable or timed "
                           "out while connecting); none of them was sent the request"}})
        if refusal is None:
            refusal = _Refusal(self._none_live(), self.cfg.retry_after_s)
        snap = self.hub.snapshot()
        other = router.elsewhere(snap, self.cfg, route.model, exact=route.exact, task=route.task)
        if route.task != router.CHAT:
            # NOTHING OF THIS TASK IS LIVE: never a cluster of another (a chat model answers no upload) — said so.
            serving = sorted(c.name for c in (self.cfg.fleet.clusters.values() if self.cfg.fleet is not None else ())
                             if c.task == route.task)
            refusal = _Refusal(f"no {route.task} backend is live: {urllib.parse.urlsplit(path).path} goes only to a "
                               f"cluster whose model's task is {route.task} "
                               f"({', '.join(serving) if serving else 'this fleet declares none'}), never to a chat "
                               f"model — {refusal.message}", refusal.retry_after_s, retry=refusal.retry)
        elif other:
            # Live on ANOTHER cluster, never offered to this request (no model, a loose name: the default cluster's
            # alone): the 503 says the default's state and what IS live — never "nothing is live" — so the client can
            # name that model; it is never moved onto it.
            name = self.cfg.default_cluster
            view = snap.cluster_views().get(name) or {}
            state = view.get("state") or "not in this run"
            detail = f" ({view['detail']})" if view.get("detail") else ""
            refusal = _Refusal(f"this request names no model a cluster serves, so it goes to the default cluster, and "
                               f"the default cluster {name} is {state}{detail}; live elsewhere: {', '.join(other)} "
                               f"(name that model to use it)", refusal.retry_after_s, retry=refusal.retry)
        verdict = router.eligible(self.hub.snapshot(), self.cfg, model=route.model, catalog=self.catalog)
        if verdict.why:
            # A CATALOG MODEL NO CLUSTER SERVES, and nothing on its way brought it: refused naming what is live —
            # never rewritten onto another cluster's model (worklog/parallel-clusters.md, decision 4).
            log.warning("[proxy] %s %s: 503 model_not_served: %s", h.command, path, verdict.why)
            return self._refuse(h, "model_not_served", verdict.why, refusal.retry_after_s, retry=refusal.retry)
        log.warning("[proxy] %s %s: no backend is live — 503, Retry-After %s s%s: %s", h.command, path,
                    retry_after_header(refusal.retry_after_s), "" if refusal.retry else ", no retry",
                    refusal.message)
        self._refuse(h, "no_backend", refusal.message, refusal.retry_after_s, retry=refusal.retry)
