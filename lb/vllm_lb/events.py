"""THE ACTIVITY EVENT LOG — one typed, append-only feed of what HAPPENED to the fleet.

WHY. Before it, what happened was spread over three places, each with its own shape: the cluster boot log
(`boots-*.jsonl`, perflog.BootRecorder — one line per START, written when the start ends), the power panel's audit log
(another service), and request kills (only a log line). "What happened at 03:12?" took three readers. This is the one
feed: every line a typed event — its `kind`, the box or server it is about (`subject`), one human line (`summary`) and
the kind's own fields — and GET /lb/events reads it back newest first, filtered by time and kind.

THE KINDS (`KINDS`, closed: an unknown kind is a programming error, refused at the call):
  cluster.state     a cluster's state changed (cluster_loop `_note_state`): state, previous, detail
  request.killed    the operator killed a turn (proxy `kill_request`): id, by, and what it had produced
  finding.firing    a health finding held for its hold time (health.py): box, kind, value, threshold
  finding.cleared   a firing finding stopped holding (health.py)
  alert.sent        an outbound alert reached a channel (alerts.py): channel, about
  alert.failed      an outbound alert did NOT reach a channel (alerts.py): channel, why — never retried elsewhere
  power.action      a power action the LB itself sees (none today: the power panel is its own service and does not
                    tell the LB; the kind is reserved so its line has one shape when it does)

WHERE. In the perf log's directory (`Config.perf_dir`, the state volume), one file per UTC day,
`events-YYYY-MM-DD.jsonl` — and off with the perf log (LB_PERF=0): the same switch for the same diary directory. The
writing is the perf log's own `JsonlWriter` (one thread, a bounded queue, a day's file capped, the oldest days pruned
past the budget) — never a second implementation of the pruning. The budget is `MAX_TOTAL_BYTES`, the boot log's: events
are a few hundred bytes each, a few an hour; a day of a flapping finding is still far inside it.

NEVER IN THE WAY. `record` builds a dict, hands it to the writer's queue (`put_nowait`) and calls the listeners — the
alert sender (alerts.py) is one, and it only enqueues. A full queue drops the line and counts it; a disk error is a
count and a sentence in the status; nothing here raises into the control loop. With no log installed (LB_PERF=0, or a
test that builds a loop alone) the module's `record` writes nothing and returns None: the feed is OFF, which
/lb/status says (`events.enabled`), never a quiet partial feed.
"""
from __future__ import annotations

import json
import logging
import pathlib
import threading
import time
from typing import Any, Callable

from vllm_lb.perflog import EXCERPT, JsonlWriter, _finite_float, _reject_constant, day_of, file_re

log = logging.getLogger(__name__)

PREFIX = "events"
RECORD_VERSION = 1
KINDS = frozenset(("cluster.state", "request.killed", "finding.firing", "finding.cleared",
                   "alert.sent", "alert.failed", "power.action"))
#: The directory's budget for event files: the boot log's (perflog.BootRecorder). Past it the oldest days go.
MAX_TOTAL_BYTES = 16 << 20
#: The most events one read returns (newest first): a page, not the whole history.
READ_LIMIT = 500
#: A summary longer than this is cut: one line, not a log dump.
SUMMARY_CHARS = 300


class EventLog:
    """The feed: `record` appends, `read` reads back. Thread-safe."""

    def __init__(self, directory: str, *, max_total_bytes: int = MAX_TOTAL_BYTES,
                 clock: Callable[[], float] = time.time) -> None:
        self.dir = str(directory)
        self.clock = clock
        self._writer = JsonlWriter(directory, prefix=PREFIX, stamp="t", max_total_bytes=max_total_bytes,
                                   max_file_bytes=max(1 << 16, max_total_bytes // 8))
        self._file_re = file_re(PREFIX)
        self._listeners: list[Callable[[dict], None]] = []
        self._lock = threading.Lock()
        self.listener_errors = 0

    @classmethod
    def from_config(cls, cfg) -> "EventLog | None":
        """The process's feed, its writer started — in the perf log's directory, off with it (LB_PERF=0)."""
        if not getattr(cfg, "perf_enabled", False):
            return None
        lg = cls(cfg.perf_dir)
        lg.start()
        return lg

    def start(self) -> None:
        self._writer.start()

    def close(self, timeout: float = 5.0) -> None:
        self._writer.stop(timeout)

    def drain(self) -> None:
        """Write what is queued, in this thread (tests)."""
        self._writer.drain()

    def subscribe(self, fn: Callable[[dict], None]) -> None:
        """`fn(event)` on every record, in the recording thread: it must not block (alerts.py only enqueues)."""
        with self._lock:
            self._listeners.append(fn)

    def record(self, kind: str, subject: str | None, summary: str, **fields: Any) -> dict:
        if kind not in KINDS:
            raise ValueError(f"event kind {kind!r} is not one of {', '.join(sorted(KINDS))}")
        rec = {"v": RECORD_VERSION, "t": round(self.clock(), 3), "kind": kind,
               "subject": subject if subject is None else str(subject), "summary": str(summary)[:SUMMARY_CHARS],
               **fields}
        self._writer.submit(rec)
        with self._lock:
            listeners = list(self._listeners)
        for fn in listeners:
            try:
                fn(rec)
            except Exception:                      # noqa: BLE001 — a listener's bug must not cost the recorder
                with self._lock:
                    self.listener_errors += 1
                log.exception("[events] a listener failed on %s", kind)
        return rec

    # ── read ───────────────────────────────────────────────────────────────

    def read(self, since: float | None = None, kinds: set[str] | None = None, limit: int = READ_LIMIT) -> dict:
        """Newest first: every event at or after `since` (None = any time) of one of `kinds` (None = every kind), at
        most `limit`; plus every line that could not be read (file, line, the parser's words) and why there is
        nothing, when there is nothing to read."""
        d = pathlib.Path(self.dir)
        out: dict[str, Any] = {"events": [], "bad": [], "why": "", "kinds": sorted(KINDS)}
        try:
            files = sorted((p for p in d.iterdir() if p.is_file() and self._file_re.match(p.name)), reverse=True)
        except FileNotFoundError:
            out["why"] = f"no event log at {self.dir} yet — nothing has been recorded there"
            return out
        except OSError as exc:
            out["why"] = f"cannot read {self.dir}: {exc}"
            return out
        first_day = day_of(since) if since is not None else None
        for p in files:
            if first_day is not None and p.name < f"{PREFIX}-{first_day}.jsonl":
                break
            try:
                text = p.read_bytes().decode("utf-8", "replace")
            except OSError as exc:
                out["bad"].append({"file": p.name, "line": None, "error": f"cannot read: {exc}"[:200]})
                continue
            day: list[dict] = []
            for i, line in enumerate(text.splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line, parse_float=_finite_float, parse_constant=_reject_constant)
                except ValueError as exc:
                    out["bad"].append({"file": p.name, "line": i, "error": f"not JSON: {exc}"[:200],
                                       "excerpt": line[:EXCERPT]})
                    continue
                t = rec.get("t") if isinstance(rec, dict) else None
                if not isinstance(t, (int, float)) or isinstance(t, bool) or rec.get("kind") not in KINDS:
                    out["bad"].append({"file": p.name, "line": i, "error": "not an event (no t, or an unknown kind)",
                                       "excerpt": line[:EXCERPT]})
                    continue
                if (since is None or t >= since) and (kinds is None or rec["kind"] in kinds):
                    day.append(rec)
            day.reverse()                           # the later line first where two share a stamp
            day.sort(key=lambda e: e["t"], reverse=True)
            out["events"] += day
            if len(out["events"]) >= limit:
                break
        out["events"] = out["events"][:limit]
        return out

    def status(self) -> dict:
        return {"enabled": True, "dir": self.dir, "listener_errors": self.listener_errors, **self._writer.status()}


# ── the process's feed ────────────────────────────────────────────────────

_installed: EventLog | None = None


def install(lg: EventLog | None) -> None:
    """Make `lg` the feed `record` writes to (main, at boot); None takes it away."""
    global _installed
    _installed = lg


def installed() -> EventLog | None:
    return _installed


#: The process's listeners (`subscribe`): told every event `record` makes, whether or not a feed is installed — the
#: alert sender is one, so switching the DIARY off (LB_PERF=0) never switches the alerts off with it.
_listeners: list[Callable[[dict], None]] = []


def subscribe(fn: Callable[[dict], None]) -> None:
    """`fn(event)` on every module-level `record` (main: the alert sender's `hear`). It must not block."""
    _listeners.append(fn)


def unsubscribe(fn: Callable[[dict], None]) -> None:
    if fn in _listeners:
        _listeners.remove(fn)


def record(kind: str, subject: str | None, summary: str, **fields: Any) -> dict | None:
    """One event: written to the process's feed and told to every listener. Returns the event as written — None when
    no feed is installed (LB_PERF=0): nothing is written, the status says the feed is off, and the listeners are
    still told. An unknown kind raises whether or not a feed is installed."""
    if kind not in KINDS:
        raise ValueError(f"event kind {kind!r} is not one of {', '.join(sorted(KINDS))}")
    lg = _installed
    written = lg.record(kind, subject, summary, **fields) if lg is not None else None
    rec = written or {"v": RECORD_VERSION, "t": round(time.time(), 3), "kind": kind, "subject": subject,
                      "summary": str(summary)[:SUMMARY_CHARS], **fields}
    for fn in list(_listeners):
        try:
            fn(rec)
        except Exception:                          # noqa: BLE001 — a listener's bug must not cost the caller
            log.exception("[events] a listener failed on %s", kind)
    return written


def status() -> dict:
    lg = _installed
    return lg.status() if lg is not None else {"enabled": False,
                                               "why": "no event log: the perf log's directory is off (LB_PERF=0)"}


def answer(query: str) -> tuple[int, dict]:
    """GET /lb/events?since=<epoch s>&kind=<kind>[,<kind>…] — (status, body): the feed newest first (`read`), the
    writer's state, and why it is empty when it is. A `since` that is not a number, or a kind not in KINDS, is the
    caller's mistake and a 400 naming it — never read as "any time" or "every kind". An admin's route (roles.py: every
    /lb/ route a member's allowlist does not name is refused before it gets here)."""
    import math
    import urllib.parse
    q = urllib.parse.parse_qs(query or "", keep_blank_values=True)
    since = None
    if q.get("since"):
        try:
            since = float(q["since"][-1])
            if not math.isfinite(since):
                raise ValueError
        except ValueError:
            return 400, {"error": {"type": "invalid_request",
                                   "message": f"`since` is an epoch time in seconds, got {q['since'][-1]!r}"}}
    kinds = None
    if q.get("kind"):
        kinds = {k.strip() for k in ",".join(q["kind"]).split(",") if k.strip()}
        unknown = sorted(kinds - KINDS)
        if unknown:
            return 400, {"error": {"type": "invalid_request",
                                   "message": f"kind {', '.join(unknown)} is not one of {', '.join(sorted(KINDS))}"}}
    lg = _installed
    if lg is None:
        st = status()
        return 200, {"events": [], "bad": [], "why": st["why"], "kinds": sorted(KINDS), "status": st}
    return 200, {**lg.read(since=since, kinds=kinds), "status": lg.status()}


# ── POST /lb/events: what another service of the fleet tells the feed ─────

#: The kinds a POST may record: only what the LB cannot see itself — the power panel's executed actions (fleet-power
#: posts each one, power/fleet_power/app.py). Every other kind is the LB's own and is refused.
POSTABLE = frozenset(("power.action",))
#: A posted event's body, at most: one action's facts, not a log.
POST_MAX_BYTES = 4096
#: A posted event's fields beyond kind / subject / summary — each a plain scalar.
POST_FIELDS = frozenset(("action", "result", "user", "dry_run", "rc"))


def post_answer(content_type: str | None, content_length: str | None, read: Callable[[], bytes]) -> tuple[int, dict]:
    """POST /lb/events {kind, subject, summary, action?, result?, user?, dry_run?, rc?} — (status, body). Inside the
    trust boundary only: a member is refused before this (roles.py: not in MEMBER_POSTS), and a write needs
    `Content-Type: application/json` (the LB's CSRF rule). A body over POST_MAX_BYTES (or with no Content-Length) is
    refused unread; a kind not in POSTABLE, a missing or non-text subject / summary, an unknown or non-scalar field
    is 400 naming it. With the feed off (LB_PERF=0) there is nowhere to keep it: 503, never a quiet drop."""
    if (content_type or "").split(";", 1)[0].strip().lower() != "application/json":
        return 415, {"error": {"type": "unsupported_media_type",
                               "message": "an event is a POST with `Content-Type: application/json`"}}
    try:
        n = int(content_length) if content_length is not None else -1
    except ValueError:
        n = -1
    if n < 0:
        return 411, {"error": {"type": "length_required", "message": "an event POST states its Content-Length"}}
    if n > POST_MAX_BYTES:
        return 413, {"error": {"type": "too_large", "message": f"an event body is at most {POST_MAX_BYTES} bytes"}}
    try:
        body = json.loads(read() or b"null", parse_constant=_reject_constant)
    except ValueError as exc:
        return 400, {"error": {"type": "invalid_request", "message": f"not JSON: {exc}"[:200]}}
    if not isinstance(body, dict):
        return 400, {"error": {"type": "invalid_request", "message": "an event is a JSON object"}}
    kind = body.get("kind")
    if kind not in POSTABLE:
        return 400, {"error": {"type": "invalid_request",
                               "message": f"kind {kind!r} may not be posted: only {', '.join(sorted(POSTABLE))} "
                                          f"(every other kind is the LB's own)"}}
    for k in ("subject", "summary"):
        if not isinstance(body.get(k), str) or not body[k].strip():
            return 400, {"error": {"type": "invalid_request", "message": f"`{k}` must be a non-empty string"}}
    extra = {k: v for k, v in body.items() if k not in ("kind", "subject", "summary")}
    unknown = sorted(set(extra) - POST_FIELDS)
    if unknown:
        return 400, {"error": {"type": "invalid_request",
                               "message": f"unknown field {', '.join(unknown)}: a {kind} carries "
                                          f"{', '.join(sorted(POST_FIELDS))}"}}
    bad = [k for k, v in extra.items() if not (v is None or isinstance(v, (str, bool, int)))]
    if bad:
        return 400, {"error": {"type": "invalid_request", "message": f"{', '.join(sorted(bad))} must be a plain value"}}
    if _installed is None:
        return 503, {"error": {"type": "events_off", "message": status()["why"]}}
    extra = {k: v[:300] if isinstance(v, str) else v for k, v in extra.items()}
    rec = record(kind, body["subject"].strip()[:80], body["summary"].strip(), **extra)
    return 201, {"recorded": rec}
