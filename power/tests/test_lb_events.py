"""EVERY EXECUTED POWER ACTION IS TOLD TO THE GATEWAY'S ACTIVITY FEED.

The LB cannot see a poweroff, a Wake-on-LAN or a clock cap: this service does them. So after each executed action it
posts one `power.action` event (box, action, result, user, dry run, rc) to the LB URL in POWER_LB_EVENTS — required,
or the word `none` to switch it off. The post runs on its own thread: a slow or dead LB never delays or fails the
power action, and a failed post is logged here, never retried. A refused request is not an action and posts nothing."""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from fleet_power.app import HOLD_MS, HOLD_S
from fleet_power.lb_events import LbNotifier, lb_events_url_from_env
from tests.fakes import USER, FakeRunner, make_app, post


def confirm(app, box="box-b", action="poweroff"):
    return post(app, "/power/api/confirm", {"box": box, "action": action})


def execute(app, token):
    app.clock.t += HOLD_S
    return post(app, "/power/api/execute", {"token": token, "held_ms": HOLD_MS})


# ── POWER_LB_EVENTS ─────────────────────────────────────────────────────────────────────────────────────────────────

def test_the_lb_url_is_required_or_explicitly_none():
    with pytest.raises(ValueError, match="POWER_LB_EVENTS"):
        lb_events_url_from_env({})
    with pytest.raises(ValueError, match="POWER_LB_EVENTS"):
        lb_events_url_from_env({"POWER_LB_EVENTS": "  "})
    assert lb_events_url_from_env({"POWER_LB_EVENTS": "none"}) is None
    assert lb_events_url_from_env({"POWER_LB_EVENTS": "http://127.0.0.1:8010/lb/events"}) == \
        "http://127.0.0.1:8010/lb/events"


@pytest.mark.parametrize("bad", ["127.0.0.1:8010/lb/events", "ftp://x/lb/events", "http://x/ y"])
def test_a_url_that_is_not_http_is_refused_naming_the_variable(bad):
    with pytest.raises(ValueError, match="POWER_LB_EVENTS"):
        lb_events_url_from_env({"POWER_LB_EVENTS": bad})


# ── what the app tells it ───────────────────────────────────────────────────────────────────────────────────────────

def test_an_executed_action_is_one_power_action_event(tmp_path):
    told = []
    app = make_app(tmp_path, dry_run=True, notify=told.append)
    status, out = execute(app, confirm(app)[1]["token"])
    assert status == 200 and out["result"] == "sent"
    assert told == [{"kind": "power.action", "subject": "box-b",
                     "summary": f"poweroff box-b: sent (dry run) by {USER}", "action": "poweroff", "result": "sent",
                     "user": USER, "dry_run": True, "rc": 0}]


def test_a_failed_action_is_told_too(tmp_path):
    told = []
    app = make_app(tmp_path, runner=FakeRunner((255, "", "Connection refused")), notify=told.append)
    status, _ = execute(app, confirm(app)[1]["token"])
    assert status == 502 and told[0]["result"] == "failed" and told[0]["rc"] == 255 and told[0]["dry_run"] is False


def test_a_refused_request_is_no_action_and_tells_nothing(tmp_path):
    told = []
    app = make_app(tmp_path, notify=told.append)
    assert post(app, "/power/api/execute", {"token": "nope", "held_ms": HOLD_MS})[0] >= 400
    assert told == []


def test_a_notifier_that_raises_never_fails_the_action(tmp_path):
    def boom(event):
        raise RuntimeError("the LB is down")
    app = make_app(tmp_path, notify=boom)
    status, out = execute(app, confirm(app)[1]["token"])
    assert status == 200 and out["result"] == "sent"


# ── the post itself ─────────────────────────────────────────────────────────────────────────────────────────────────

class FakeLb:
    def __init__(self, status=201, delay_s=0.0):
        self.got, outer = [], self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):                     # noqa: N802
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                outer.got.append({"path": self.path, "ctype": self.headers.get("Content-Type"), "body": body})
                time.sleep(delay_s)
                self.send_response(status)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")

            def log_message(self, *a):
                pass

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}/lb/events"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()


EVENT = {"kind": "power.action", "subject": "box-b", "summary": "poweroff box-b: sent by operator",
         "action": "poweroff", "result": "sent", "user": "operator", "dry_run": False, "rc": 0}


def test_the_event_is_posted_as_json_to_the_lb():
    lb = FakeLb()
    try:
        assert LbNotifier(lb.url).send(EVENT) is True
    finally:
        lb.close()
    assert lb.got[0]["path"] == "/lb/events" and lb.got[0]["ctype"] == "application/json"
    assert json.loads(lb.got[0]["body"]) == EVENT


def test_a_refused_or_unreachable_post_is_logged_and_returns(capsys):
    lb = FakeLb(status=503)
    try:
        assert LbNotifier(lb.url).send(EVENT) is False
    finally:
        lb.close()
    assert LbNotifier("http://127.0.0.1:9/lb/events").send(EVENT) is False
    err = capsys.readouterr().out
    assert "503" in err and "not told to the LB" in err


def test_notify_returns_at_once_even_when_the_lb_is_slow():
    lb = FakeLb(delay_s=1.5)
    try:
        t0 = time.monotonic()
        LbNotifier(lb.url).notify(EVENT)
        assert time.monotonic() - t0 < 0.3
    finally:
        time.sleep(0.2)
        lb.close()
