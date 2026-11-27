"""OUTBOUND ALERTS — the transitions that matter, said to the operator's phone.

WHY. A health finding (health.py) or a failed cluster is drawn on the panel, and nobody is looking at the panel when
it happens. This sends it — to ntfy, to Gotify, to any webhook — once per transition: a finding FIRING, a finding
CLEARED, a cluster whose state became `failed` or `stalled`. A server that stopped answering is a finding
(`backend_down`, health.py) and is alerted as one.

FED BY THE ACTIVITY FEED. The sender is a listener of events.py (`hear`): every transition is already exactly one event
there (health.py records a firing once, cluster_loop records a state change once), so de-duplication is the feed's — one
event, one message per channel. `hear` only enqueues (bounded; a full queue drops the alert and counts it): it runs
in whatever thread recorded the event — the control loop, the health monitor — and must never block it. One thread
of its own does the HTTP.

THE CHANNELS are env (`channels_from_env`; lb/.env.example documents each name; every one optional, each VALIDATED
when set — half a channel or a URL that is not http(s) refuses the boot naming the variable, never a channel quietly
dropped):
  ntfy     LB_ALERT_NTFY_URL + LB_ALERT_NTFY_TOPIC     POST <url>/<topic>, the summary as the body; Title, Priority
  gotify   LB_ALERT_GOTIFY_URL + LB_ALERT_GOTIFY_TOKEN POST <url>/message, X-Gotify-Key, JSON {title, message, priority}
  webhook  LB_ALERT_WEBHOOK_URL                        POST <url>, the event itself as JSON
  slack    LB_ALERT_SLACK_URL                          POST <incoming webhook>, {"text": "<title>\n<summary>"}
  discord  LB_ALERT_DISCORD_URL                        POST <channel webhook>, {"content": "<title>\n<summary>"}

EVERY CHANNEL GETS EVERY ALERT, independently. A send that fails (an HTTP error, a refused connection, a timeout) is
an `alert.failed` event with the channel and why, and a line in /lb/status `health.alerts.recent_failures` the panel
shows. It is NEVER retried, and never sent to a different channel instead: a substitute route would make a dead
channel look alive (AGENTS.md, no fallbacks).
"""
from __future__ import annotations

import json
import logging
import queue
import re
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass
from typing import Mapping

from vllm_lb import events

log = logging.getLogger(__name__)

#: One send, end to end: a channel slower than this is a failure.
TIMEOUT_S = 10.0
#: Alerts waiting to be sent: past this, an alert is dropped and counted (a channel hung for minutes).
QUEUE_SIZE = 256
#: The failures the status keeps for the panel, newest first.
FAILURES_KEPT = 5
#: The cluster states that are an alert.
CLUSTER_STATES = frozenset(("failed", "stalled"))
NONE_CONFIGURED = ("no alert channel is configured: set LB_ALERT_NTFY_URL + LB_ALERT_NTFY_TOPIC, LB_ALERT_GOTIFY_URL + "
                   "LB_ALERT_GOTIFY_TOKEN, LB_ALERT_WEBHOOK_URL, LB_ALERT_SLACK_URL or LB_ALERT_DISCORD_URL (lb/.env.example)")
_URL = re.compile(r"https?://[^\s/]+(?:/\S*)?")
_TOPIC = re.compile(r"[A-Za-z0-9_-]{1,64}")


@dataclass(frozen=True)
class Channel:
    name: str                  # ntfy | gotify | webhook | slack | discord
    url: str                   # where the POST goes (ntfy: <url>/<topic>; gotify: <url>/message)
    token: str | None = None   # gotify's application token

    def view(self) -> dict:
        """What /lb/status says of it: its kind only — the URL can carry a secret (a webhook's path, a topic)."""
        return {"name": self.name}


def channels_from_env(env: Mapping[str, str]) -> tuple[Channel, ...]:
    """The channels the env configures, in a fixed order — or ValueError naming the variable that makes one unusable."""
    def get(k: str) -> str:
        return (env.get(k) or "").strip()

    def url(k: str) -> str:
        v = get(k)
        if not _URL.fullmatch(v):
            raise ValueError(f"{k} must be an http:// or https:// URL, got {v!r}")
        return v.rstrip("/")

    def pair(a: str, b: str) -> bool:
        if bool(get(a)) != bool(get(b)):
            have, lack = (a, b) if get(a) else (b, a)
            raise ValueError(f"{have} is set but {lack} is not: the channel needs both (lb/.env.example)")
        return bool(get(a))

    out = []
    if pair("LB_ALERT_NTFY_URL", "LB_ALERT_NTFY_TOPIC"):
        topic = get("LB_ALERT_NTFY_TOPIC")
        if not _TOPIC.fullmatch(topic):
            raise ValueError(f"LB_ALERT_NTFY_TOPIC must be 1-64 letters, digits, - or _, got {topic!r}")
        out.append(Channel("ntfy", f"{url('LB_ALERT_NTFY_URL')}/{topic}"))
    if pair("LB_ALERT_GOTIFY_URL", "LB_ALERT_GOTIFY_TOKEN"):
        out.append(Channel("gotify", f"{url('LB_ALERT_GOTIFY_URL')}/message", token=get("LB_ALERT_GOTIFY_TOKEN")))
    if get("LB_ALERT_WEBHOOK_URL"):
        out.append(Channel("webhook", url("LB_ALERT_WEBHOOK_URL")))
    if get("LB_ALERT_SLACK_URL"):
        out.append(Channel("slack", url("LB_ALERT_SLACK_URL")))
    if get("LB_ALERT_DISCORD_URL"):
        out.append(Channel("discord", url("LB_ALERT_DISCORD_URL")))
    return tuple(out)


def is_alert(e: dict) -> bool:
    """Whether an event of the feed is one the operator is told."""
    k = e.get("kind")
    return k in ("finding.firing", "finding.cleared") or (k == "cluster.state" and e.get("state") in CLUSTER_STATES)


def title_of(e: dict) -> str:
    k, subject = e.get("kind"), e.get("subject") or "?"
    if k == "finding.firing":
        return f"FIRING {e.get('finding')} on {subject}"
    if k == "finding.cleared":
        return f"CLEARED {e.get('finding')} on {subject}"
    return f"CLUSTER {subject} {e.get('state')}"


def _ascii(s: str) -> str:
    """A header value: HTTP/1.1 headers are latin-1; a name with anything else is said with `?`."""
    return s.encode("ascii", "replace").decode("ascii")


_STOP = object()


class AlertSender:
    """Hears the feed, sends on its own thread, records what happened. Thread-safe."""

    def __init__(self, channels: tuple[Channel, ...], *, timeout_s: float = TIMEOUT_S) -> None:
        self.channels = tuple(channels)
        self.timeout_s = timeout_s
        self._q: queue.Queue = queue.Queue(maxsize=QUEUE_SIZE)
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.sent = 0
        self.failed = 0
        self.dropped = 0
        self._failures: deque = deque(maxlen=FAILURES_KEPT)

    # ── intake: in the recording thread ────────────────────────────────────

    def hear(self, e: dict) -> None:
        if not self.channels or not is_alert(e):
            return
        try:
            self._q.put_nowait(dict(e))
        except queue.Full:
            with self._lock:
                self.dropped += 1
            log.warning("[alerts] queue full: %s not sent", title_of(e))

    # ── the sending thread ─────────────────────────────────────────────────

    def start(self) -> None:
        if self._thread is None and self.channels:
            self._thread = threading.Thread(target=self._run, name="alerts", daemon=True)
            self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        if self._thread is None:
            return
        try:
            self._q.put(_STOP, timeout=timeout)
        except queue.Full:
            pass
        self._thread.join(timeout)
        self._thread = None

    def _run(self) -> None:
        while True:
            e = self._q.get()
            if e is _STOP:
                return
            for ch in self.channels:
                self._send(ch, e)

    def _request(self, ch: Channel, e: dict) -> urllib.request.Request:
        title, summary = title_of(e), str(e.get("summary") or "")
        urgent = e.get("kind") != "finding.cleared"
        if ch.name == "ntfy":
            return urllib.request.Request(ch.url, data=summary.encode("utf-8"), method="POST", headers={
                "Title": _ascii(title), "Priority": "high" if urgent else "default",
                "Tags": "rotating_light" if urgent else "white_check_mark",
                "Content-Type": "text/plain; charset=utf-8"})
        if ch.name == "gotify":
            body = {"title": title, "message": summary, "priority": 8 if urgent else 4}
            return urllib.request.Request(ch.url, data=json.dumps(body).encode(), method="POST", headers={
                "X-Gotify-Key": ch.token or "", "Content-Type": "application/json"})
        if ch.name in ("slack", "discord"):
            # Slack's incoming webhook takes `text`, Discord's channel webhook `content` (at most 2000 characters).
            key, text = ("text", f"{title}\n{summary}") if ch.name == "slack" else ("content", f"{title}\n{summary}"[:2000])
            return urllib.request.Request(ch.url, data=json.dumps({key: text}).encode(), method="POST",
                                          headers={"Content-Type": "application/json"})
        return urllib.request.Request(ch.url, data=json.dumps({**e, "title": title}).encode(), method="POST",
                                      headers={"Content-Type": "application/json"})

    def _send(self, ch: Channel, e: dict) -> None:
        about = e.get("id") or e.get("subject")
        try:
            with urllib.request.urlopen(self._request(ch, e), timeout=self.timeout_s) as r:
                r.read(4096)
            with self._lock:
                self.sent += 1
            events.record("alert.sent", e.get("subject"), f"{ch.name}: {title_of(e)}", channel=ch.name,
                          about=about, alerted=e.get("kind"))
        except Exception as exc:                   # noqa: BLE001 — every failure is a recorded failure, never raised
            why = (f"HTTP {exc.code} {exc.reason}" if isinstance(exc, urllib.error.HTTPError)
                   else f"{type(exc).__name__}: {getattr(exc, 'reason', None) or exc}")[:300]
            with self._lock:
                self.failed += 1
                self._failures.appendleft({"at": round(time.time(), 3), "channel": ch.name, "title": title_of(e),
                                           "why": why})
            log.warning("[alerts] %s not sent to %s: %s", title_of(e), ch.name, why)
            events.record("alert.failed", e.get("subject"), f"{ch.name} failed: {title_of(e)} — {why}",
                          channel=ch.name, about=about, alerted=e.get("kind"), why=why)

    def status(self) -> dict:
        with self._lock:
            out = {"channels": [c.view() for c in self.channels], "sent": self.sent, "failed": self.failed,
                   "dropped": self.dropped, "recent_failures": list(self._failures)}
        if not self.channels:
            out["why"] = NONE_CONFIGURED
        return out
