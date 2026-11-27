"""TELL THE GATEWAY'S ACTIVITY FEED WHAT WAS DONE: one `power.action` event per executed action.

WHY. The gateway's activity feed (lb/vllm_lb/events.py, GET /lb/events) is the fleet's one record of what happened —
cluster states, request kills, findings, alerts. A poweroff, a Wake-on-LAN or a clock cap is done HERE, and the LB
cannot see it; so after each executed action this service POSTs it to the LB's `POST /lb/events` (which accepts that
kind only, inside its trust boundary). The audit log (POWER_AUDIT) stays this service's own full record.

WHERE. POWER_LB_EVENTS — REQUIRED: the LB's /lb/events URL as this container reaches it, or the word `none` to switch
the feed's power lines off (a deployment without the gateway). No default: absent, the service refuses to start.
The power container runs with `network_mode: host` and the LB publishes 127.0.0.1:8010 (lb/docker-compose.yml), so on
one host it is http://127.0.0.1:8010/lb/events — the same door POWER_LB_STATUS already reads; the LB's host guard
allows its own loopback names, and a POST with no Origin and no X-Vllm-Role is a caller inside its trust boundary.

NEVER IN THE WAY. `notify` hands the post to a daemon thread and returns: a slow or dead LB never delays, fails or
reverses a power action. A post that fails is printed (the container's log) and not retried.
"""
from __future__ import annotations

import json
import re
import threading
import urllib.error
import urllib.request
from typing import Mapping

TIMEOUT_S = 5.0
OFF = "none"
_URL = re.compile(r"https?://[^\s/]+(?:/\S*)?")


def lb_events_url_from_env(env: Mapping[str, str]) -> str | None:
    """POWER_LB_EVENTS: the LB's /lb/events URL, or None for `none` — ValueError when absent, blank or not http(s)."""
    raw = (env.get("POWER_LB_EVENTS") or "").strip()
    if not raw:
        raise ValueError("POWER_LB_EVENTS is not set: the gateway's /lb/events URL as this container reaches it "
                         "(e.g. http://127.0.0.1:8010/lb/events), or `none` to tell the activity feed nothing "
                         "(power/.env.example)")
    if raw == OFF:
        return None
    if not _URL.fullmatch(raw):
        raise ValueError(f"POWER_LB_EVENTS must be an http:// or https:// URL, or `none`, got {raw!r}")
    return raw


class LbNotifier:
    """Posts one event to the LB; `notify` does it off the caller's thread."""

    def __init__(self, url: str, *, timeout_s: float = TIMEOUT_S) -> None:
        self.url, self.timeout_s = url, timeout_s

    def send(self, event: dict) -> bool:
        """POST `event` as JSON. True when the LB recorded it; False, printed with why, otherwise. Never raises."""
        req = urllib.request.Request(self.url, data=json.dumps(event).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                r.read(4096)
            return True
        except Exception as exc:                   # noqa: BLE001 — a failed post is a log line, never the action's
            why = (f"HTTP {exc.code} {exc.reason}" if isinstance(exc, urllib.error.HTTPError)
                   else f"{type(exc).__name__}: {getattr(exc, 'reason', None) or exc}")
            print(f"lb-events: {event.get('action')} {event.get('subject')} not told to the LB ({self.url}): {why}",
                  flush=True)
            return False

    def notify(self, event: dict) -> None:
        threading.Thread(target=self.send, args=(dict(event),), name="lb-events", daemon=True).start()
