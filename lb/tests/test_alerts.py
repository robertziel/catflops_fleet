"""OUTBOUND ALERTS (vllm_lb/alerts.py): a health finding firing or clearing, a cluster that failed,
said to the channels the env configures — ntfy, Gotify, a generic webhook — once per transition, off the control
loop's thread. A failed send is an `alert.failed` event and a line in the panel, never retried on another channel.
Tested against a fake HTTP receiver: what each channel is sent, byte for byte where it matters."""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from vllm_lb import events
from vllm_lb.alerts import AlertSender, Channel, channels_from_env
from vllm_lb.config import Config
from vllm_lb.events import EventLog


class Receiver:
    """A fake ntfy / Gotify / webhook endpoint: records every POST; answers `status` (per path prefix)."""

    def __init__(self, status: int = 200, delay_s: float = 0.0):
        self.got: list[dict] = []
        self.status, self.delay_s = status, delay_s
        outer = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):                     # noqa: N802
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                outer.got.append({"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()},
                                  "body": body})
                time.sleep(outer.delay_s)
                self.send_response(outer.status)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")

            def log_message(self, *a):
                pass

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()


@pytest.fixture
def rx():
    made = []

    def make(**kw):
        r = Receiver(**kw)
        made.append(r)
        return r

    yield make
    for r in made:
        r.close()


@pytest.fixture
def feed(tmp_path):
    lg = EventLog(str(tmp_path))
    events.install(lg)
    yield lg
    events.install(None)


def wait(cond, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


FIRING = dict(id="box-a:gpu_temp", finding="gpu_temp", value=88.0, threshold=85.0, unit="°C")


# ── which channels: env, validated when set ─────────────────────────────────────────────────────────────────────────

def test_no_alert_env_is_no_channel():
    assert channels_from_env({}) == ()


def test_each_channel_is_built_from_its_env_names():
    got = channels_from_env({"LB_ALERT_NTFY_URL": "https://ntfy.example.org", "LB_ALERT_NTFY_TOPIC": "fleet",
                             "LB_ALERT_GOTIFY_URL": "http://gotify.example.org", "LB_ALERT_GOTIFY_TOKEN": "tok",
                             "LB_ALERT_WEBHOOK_URL": "https://hooks.example.org/x",
                             "LB_ALERT_SLACK_URL": "https://hooks.example.org/slack",
                             "LB_ALERT_DISCORD_URL": "https://hooks.example.org/discord"})
    assert [c.name for c in got] == ["ntfy", "gotify", "webhook", "slack", "discord"]


@pytest.mark.parametrize("env,says", [
    ({"LB_ALERT_NTFY_URL": "https://ntfy.example.org"}, "LB_ALERT_NTFY_TOPIC"),            # half a channel
    ({"LB_ALERT_NTFY_TOPIC": "fleet"}, "LB_ALERT_NTFY_URL"),
    ({"LB_ALERT_GOTIFY_URL": "http://g.example.org"}, "LB_ALERT_GOTIFY_TOKEN"),
    ({"LB_ALERT_WEBHOOK_URL": "ftp://x.example.org"}, "LB_ALERT_WEBHOOK_URL"),               # not http(s)
    ({"LB_ALERT_SLACK_URL": "hooks.example.org/x"}, "LB_ALERT_SLACK_URL"),
    ({"LB_ALERT_DISCORD_URL": "not a url"}, "LB_ALERT_DISCORD_URL"),
    ({"LB_ALERT_NTFY_URL": "https://ntfy.example.org", "LB_ALERT_NTFY_TOPIC": "a b"}, "LB_ALERT_NTFY_TOPIC"),
])
def test_a_channel_set_but_unusable_refuses_the_boot_naming_the_variable(env, says):
    with pytest.raises(ValueError, match=says):
        channels_from_env(env)


def test_the_config_refuses_an_unusable_channel_at_boot(tmp_path):
    from tests.fleet_files import PAIR, write
    env = {"LB_ALLOWED_HOSTS": "loopback", "LB_FLEET": write(tmp_path, PAIR),
           "LB_CLUSTER_STATE": str(tmp_path / "c.json"), "LB_ALERT_WEBHOOK_URL": "not a url"}
    with pytest.raises(ValueError, match="LB_ALERT_WEBHOOK_URL"):
        Config.from_env(env)
    env["LB_ALERT_WEBHOOK_URL"] = "https://hooks.example.org/x"
    assert [c.name for c in Config.from_env(env).alert_channels] == ["webhook"]


# ── what is sent ────────────────────────────────────────────────────────────────────────────────────────────────────

def test_a_finding_firing_reaches_every_channel_in_its_own_shape_and_is_recorded_sent(rx, feed):
    r = rx()
    sender = AlertSender((Channel("ntfy", f"{r.url}/fleet"), Channel("gotify", f"{r.url}/message", token="tok"),
                          Channel("webhook", f"{r.url}/hook")))
    feed.subscribe(sender.hear)
    sender.start()
    try:
        feed.record("finding.firing", "box-a", "GPU TEMP on box-a: hottest GPU 88 °C ≥ 85 °C", **FIRING)
        assert wait(lambda: len(r.got) == 3)
        assert wait(lambda: sender.status()["sent"] == 3)
    finally:
        sender.stop()
    by = {g["path"]: g for g in r.got}
    ntfy = by["/fleet"]
    assert ntfy["body"].decode() == "GPU TEMP on box-a: hottest GPU 88 °C ≥ 85 °C"
    assert ntfy["headers"]["title"] == "FIRING gpu_temp on box-a" and ntfy["headers"]["priority"] == "high"
    gotify = by["/message"]
    assert gotify["headers"]["x-gotify-key"] == "tok"
    assert json.loads(gotify["body"])["title"] == "FIRING gpu_temp on box-a"
    hook = json.loads(by["/hook"]["body"])
    assert hook["kind"] == "finding.firing" and hook["subject"] == "box-a" and hook["value"] == 88.0
    feed.drain()
    assert sorted(e["channel"] for e in feed.read(kinds={"alert.sent"})["events"]) == ["gotify", "ntfy", "webhook"]


def test_a_cleared_finding_and_a_failed_cluster_are_alerts_and_other_events_are_not(rx, feed):
    r = rx()
    sender = AlertSender((Channel("webhook", f"{r.url}/hook"),))
    feed.subscribe(sender.hear)
    sender.start()
    try:
        feed.record("finding.cleared", "box-a", "GPU TEMP cleared on box-a", **FIRING)
        feed.record("cluster.state", "pair", "starting → failed: rank exited", state="failed", previous="starting")
        feed.record("cluster.state", "pair", "starting → running", state="running", previous="starting")
        feed.record("request.killed", "box-a", "request x killed")
        assert wait(lambda: sender.status()["sent"] == 2)
        time.sleep(0.2)
    finally:
        sender.stop()
    assert [json.loads(g["body"])["kind"] for g in r.got] == ["finding.cleared", "cluster.state"]


def test_a_cluster_is_alerted_on_failed_and_on_stalled_and_on_no_other_state():
    """A halted hung start is `stalled` (cluster_loop: the cause kept over the 137 that follows it): as much a dead
    cluster as `failed`, so both are told — each pinned on its own, so dropping either fails here."""
    from vllm_lb.alerts import is_alert, title_of
    for state in ("failed", "stalled"):
        e = {"kind": "cluster.state", "subject": "pair", "state": state}
        assert is_alert(e), state
        assert title_of(e) == f"CLUSTER pair {state}"
    for state in ("running", "starting", "off", "draining", "unknown", "blocked"):
        assert not is_alert({"kind": "cluster.state", "subject": "pair", "state": state}), state


def test_a_failed_send_is_recorded_shown_and_never_retried_on_another_channel(rx, feed):
    bad, good = rx(status=500), rx()
    sender = AlertSender((Channel("ntfy", f"{bad.url}/fleet"), Channel("webhook", f"{good.url}/hook")))
    feed.subscribe(sender.hear)
    sender.start()
    try:
        feed.record("finding.firing", "box-a", "hot", **FIRING)
        assert wait(lambda: sender.status()["failed"] == 1 and sender.status()["sent"] == 1)
    finally:
        sender.stop()
    assert len(bad.got) == 1 and len(good.got) == 1, "each channel once: no retry, no reroute"
    st = sender.status()
    assert st["recent_failures"][0]["channel"] == "ntfy" and "500" in st["recent_failures"][0]["why"]
    feed.drain()
    failed = feed.read(kinds={"alert.failed"})["events"]
    assert len(failed) == 1 and failed[0]["channel"] == "ntfy" and "500" in failed[0]["why"]


def test_an_unreachable_channel_is_a_failure_with_why(feed):
    sender = AlertSender((Channel("webhook", "http://127.0.0.1:9/hook"),))
    feed.subscribe(sender.hear)
    sender.start()
    try:
        feed.record("finding.firing", "box-a", "hot", **FIRING)
        assert wait(lambda: sender.status()["failed"] == 1)
    finally:
        sender.stop()
    assert sender.status()["recent_failures"][0]["why"]


def test_hearing_never_blocks_the_recording_thread_even_on_a_slow_channel(rx, feed):
    r = rx(delay_s=1.0)
    sender = AlertSender((Channel("webhook", f"{r.url}/hook"),))
    feed.subscribe(sender.hear)
    sender.start()
    try:
        t0 = time.monotonic()
        for _ in range(3):
            feed.record("finding.firing", "box-a", "hot", **FIRING)
        assert time.monotonic() - t0 < 0.5
    finally:
        sender.stop(timeout=0.1)


def test_with_no_channel_an_alert_is_nothing_sent_and_the_status_says_none_configured(feed):
    sender = AlertSender(())
    feed.subscribe(sender.hear)
    feed.record("finding.firing", "box-a", "hot", **FIRING)
    st = sender.status()
    assert st["channels"] == [] and st["sent"] == 0 and "LB_ALERT_" in st["why"]


def test_slack_gets_text_and_discord_gets_content_title_then_summary(rx, feed):
    """Slack's incoming webhook reads `text`, Discord's channel webhook `content`: the body is pinned, byte for byte."""
    r = rx()
    sender = AlertSender((Channel("slack", f"{r.url}/slack"), Channel("discord", f"{r.url}/discord")))
    feed.subscribe(sender.hear)
    sender.start()
    try:
        feed.record("finding.firing", "box-a", "GPU TEMP on box-a: GPU 3 °C below its own throttle limit", **FIRING)
        assert wait(lambda: sender.status()["sent"] == 2)
    finally:
        sender.stop()
    by = {g["path"]: g for g in r.got}
    want = "FIRING gpu_temp on box-a\nGPU TEMP on box-a: GPU 3 °C below its own throttle limit"
    assert json.loads(by["/slack"]["body"]) == {"text": want}
    assert json.loads(by["/discord"]["body"]) == {"content": want}
    assert by["/slack"]["headers"]["content-type"] == "application/json"
