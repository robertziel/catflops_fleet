"""THE ACTIVITY EVENT LOG (vllm_lb/events.py): one typed, append-only feed — cluster state changes,
request kills, health findings, outbound alerts — in `events-YYYY-MM-DD.jsonl` beside the perf log, under a byte
budget pruned by the perf log's own writer; read back newest first by GET /lb/events (an admin's route)."""
from __future__ import annotations

import json
import pathlib

import pytest

from vllm_lb import events
from vllm_lb.events import EventLog, KINDS


@pytest.fixture
def log(tmp_path):
    t = [1_800_000_000.0]
    lg = EventLog(str(tmp_path), max_total_bytes=1 << 20, clock=lambda: t[0])
    lg.tick = t                                   # the test's clock hand
    return lg


def _lines(directory: pathlib.Path) -> list[dict]:
    out = []
    for p in sorted(directory.glob("events-*.jsonl")):
        out += [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    return out


def test_a_record_is_one_json_line_in_the_days_file_with_kind_subject_summary_and_fields(log, tmp_path):
    rec = log.record("cluster.state", "pair", "starting → running", state="running", previous="starting")
    log.drain()
    assert rec == {"v": 1, "t": 1_800_000_000.0, "kind": "cluster.state", "subject": "pair",
                   "summary": "starting → running", "state": "running", "previous": "starting"}
    files = list(tmp_path.glob("events-*.jsonl"))
    assert [f.name for f in files] == ["events-2027-01-15.jsonl"]
    assert _lines(tmp_path) == [rec]


def test_an_unknown_kind_is_refused_naming_the_kinds_there_are(log):
    with pytest.raises(ValueError, match="cluster.state"):
        log.record("cluster.exploded", None, "x")


def test_the_feed_reads_newest_first_and_filters_by_since_and_kind(log):
    for i, kind in enumerate(("cluster.state", "request.killed", "finding.firing", "cluster.state")):
        log.tick[0] = 1_800_000_000.0 + i * 10
        log.record(kind, "pair", f"#{i}")
    log.drain()
    got = log.read()
    assert [e["summary"] for e in got["events"]] == ["#3", "#2", "#1", "#0"]
    assert [e["summary"] for e in log.read(since=1_800_000_015.0)["events"]] == ["#3", "#2"]
    assert [e["summary"] for e in log.read(kinds={"cluster.state"})["events"]] == ["#3", "#0"]
    assert got["kinds"] == sorted(KINDS)


def test_the_feed_spans_days_and_lists_a_line_it_cannot_read_never_drops_it(log, tmp_path):
    log.record("cluster.state", "pair", "day one")
    log.tick[0] += 86400
    log.record("cluster.state", "pair", "day two")
    log.drain()
    first = sorted(tmp_path.glob("events-*.jsonl"))[0]
    first.write_text(first.read_text() + "{not json\n")
    got = log.read()
    assert [e["summary"] for e in got["events"]] == ["day two", "day one"]
    assert got["bad"] and got["bad"][0]["file"] == first.name and got["bad"][0]["line"] == 2


def test_the_directory_stays_inside_its_budget_oldest_days_first(tmp_path):
    t = [1_800_000_000.0]
    lg = EventLog(str(tmp_path), max_total_bytes=600, clock=lambda: t[0])
    for day in range(5):
        t[0] = 1_800_000_000.0 + day * 86400
        lg.record("cluster.state", "pair", "x" * 200)
        lg.drain()
    names = sorted(p.name for p in tmp_path.glob("events-*.jsonl"))
    assert len(names) < 5 and names[-1] == "events-2027-01-19.jsonl"
    assert lg.status()["deleted_days"] >= 1


def test_a_listener_hears_every_record_and_a_listener_that_raises_costs_nothing(log):
    heard = []
    log.subscribe(lambda e: heard.append(e["kind"]))
    log.subscribe(lambda e: 1 / 0)
    log.record("finding.firing", "box-a", "hot")
    assert heard == ["finding.firing"]


def test_the_module_record_goes_to_the_installed_log_and_is_none_without_one(log):
    events.install(None)
    assert events.record("cluster.state", "pair", "nobody listening") is None
    events.install(log)
    try:
        assert events.record("cluster.state", "pair", "heard")["summary"] == "heard"
    finally:
        events.install(None)


def test_a_record_that_cannot_be_json_is_a_count_not_an_exception(log):
    log.record("cluster.state", "pair", "nan", value=float("nan"))
    log.drain()
    assert log.status()["errors"] == 1


def test_a_missing_directory_reads_as_no_events_with_why(tmp_path):
    lg = EventLog(str(tmp_path / "never"), max_total_bytes=1 << 20)
    got = lg.read()
    assert got["events"] == [] and "nothing has been recorded" in got["why"]


def test_from_config_lives_in_the_perf_dir_and_is_off_with_the_perf_log(tmp_path):
    from vllm_lb.config import Config
    on = EventLog.from_config(Config(perf_dir=str(tmp_path), perf_enabled=True))
    assert on is not None and on.dir == str(tmp_path)
    on.close()
    assert EventLog.from_config(Config(perf_dir=str(tmp_path), perf_enabled=False)) is None


# ── the hooks that feed it ──────────────────────────────────────────────────────────────────────────────────────────

def test_a_cluster_state_change_is_an_event_naming_the_cluster_and_both_states(log):
    from types import SimpleNamespace
    from vllm_lb.cluster_loop import ClusterLoop
    loop = SimpleNamespace(_state="starting", _since=0.0, cfg=SimpleNamespace(cluster_name="pair"))
    events.install(log)
    try:
        ClusterLoop._note_state(loop, SimpleNamespace(state="failed", detail="rank on box-a exited (137)"), 10.0)
        ClusterLoop._note_state(loop, SimpleNamespace(state="failed", detail="still failed"), 11.0)   # no change
    finally:
        events.install(None)
    log.drain()
    got = log.read(kinds={"cluster.state"})["events"]
    assert len(got) == 1
    e = got[0]
    assert (e["subject"], e["state"], e["previous"], e["detail"]) == ("pair", "failed", "starting",
                                                                      "rank on box-a exited (137)")
    assert e["summary"] == "starting → failed: rank on box-a exited (137)"


# ── GET /lb/events ──────────────────────────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def lb(tmp_path):
    import http.client
    from vllm_lb.config import Config, VllmBox
    from vllm_lb.hub import Hub
    from vllm_lb.proxy import Proxy
    from tests.fakes import FakeBackend
    engine = FakeBackend()
    cfg = Config(vllm_boxes=(VllmBox("box-a-gpu", "box-a", engine.url),), listen_host="127.0.0.1", listen_port=0,
                 perf_dir=str(tmp_path))
    lg = EventLog(str(tmp_path), max_total_bytes=1 << 20)
    events.install(lg)
    proxy = Proxy(cfg, Hub(cfg))
    proxy.start()

    def get(path, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=10)
        conn.request("GET", path, headers=headers or {})
        r = conn.getresponse()
        return r.status, json.loads(r.read())

    get.port = proxy.port
    yield lg, get
    proxy.stop()
    engine.close()
    events.install(None)


def test_get_events_answers_newest_first_filtered_by_kind_and_since(lb):
    lg, get = lb
    lg.record("cluster.state", "pair", "a")
    lg.record("request.killed", "box-a-gpu", "b")
    lg.drain()
    status, body = get("/lb/events")
    assert status == 200 and [e["summary"] for e in body["events"]] == ["b", "a"]
    assert body["status"]["enabled"] is True
    status, body = get("/lb/events?kind=cluster.state,finding.firing")
    assert [e["summary"] for e in body["events"]] == ["a"]
    status, body = get("/lb/events?since=4102444800")                       # 2100: nothing yet
    assert status == 200 and body["events"] == []


@pytest.mark.parametrize("query,says", [("since=yesterday", "since"), ("kind=cluster.exploded", "cluster.state")])
def test_a_query_the_feed_cannot_mean_is_400_naming_what(lb, query, says):
    _, get = lb
    status, body = get(f"/lb/events?{query}")
    assert status == 400 and says in body["error"]["message"]


def test_a_member_is_refused_the_feed(lb):
    _, get = lb
    status, body = get("/lb/events", {"X-Vllm-Role": "member", "X-Vllm-Models": "m"})
    assert status == 403 and body["error"]["type"] == "admin_only"


def test_with_the_feed_off_the_route_says_so(lb):
    _, get = lb
    events.install(None)
    status, body = get("/lb/events")
    assert status == 200 and body["events"] == [] and body["status"]["enabled"] is False
    assert "LB_PERF=0" in body["why"]


def test_a_process_listener_is_told_even_with_the_diary_off():
    heard = []
    events.install(None)
    events.subscribe(heard.append)
    try:
        assert events.record("finding.firing", "box-a", "hot", finding="gpu_temp") is None   # nothing written…
    finally:
        events.unsubscribe(heard.append)
    assert [e["kind"] for e in heard] == ["finding.firing"]                                 # …and still heard


# ── POST /lb/events: fleet-power's executed actions ─────────────────────────────────────────────────────────────────

def _post(port, body, headers=None, raw=None):
    import http.client
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    data = raw if raw is not None else json.dumps(body).encode()
    conn.request("POST", "/lb/events", body=data, headers={"Content-Type": "application/json", **(headers or {})})
    r = conn.getresponse()
    return r.status, json.loads(r.read() or b"null")


@pytest.fixture
def lb_port(lb):
    lg, get = lb
    return lg, get, get.port


POWER = {"kind": "power.action", "subject": "box-a", "summary": "poweroff box-a: sent (dry run) by op",
         "action": "poweroff", "result": "sent", "user": "op", "dry_run": True, "rc": 0}


def test_a_power_action_posted_is_recorded_on_the_feed(lb_port):
    lg, get, port = lb_port
    status, body = _post(port, POWER)
    assert status == 201 and body["recorded"]["kind"] == "power.action"
    lg.drain()
    e = get("/lb/events?kind=power.action")[1]["events"][0]
    assert (e["subject"], e["action"], e["result"], e["user"], e["dry_run"]) == ("box-a", "poweroff", "sent", "op", True)


@pytest.mark.parametrize("body,says", [
    ({**POWER, "kind": "cluster.state"}, "only power.action"),                # the LB's own kinds are not postable
    ({**POWER, "kind": "finding.firing"}, "only power.action"),
    ({**POWER, "subject": ""}, "subject"),
    ({k: v for k, v in POWER.items() if k != "summary"}, "summary"),
    ({**POWER, "colour": "red"}, "colour"),
    ({**POWER, "user": {"name": "op"}}, "plain value"),
    ([POWER], "JSON object"),
])
def test_a_post_that_is_not_a_well_formed_power_action_is_400_naming_why(lb_port, body, says):
    _, _, port = lb_port
    status, got = _post(port, body)
    assert status == 400 and says in got["error"]["message"]


def test_a_post_must_be_json_and_small(lb_port):
    _, _, port = lb_port
    assert _post(port, POWER, headers={"Content-Type": "text/plain"})[0] == 415
    big = {**POWER, "summary": "x" * 5000}
    status, got = _post(port, big)
    assert status == 413 and "4096" in got["error"]["message"]


def test_a_member_may_not_post_an_event(lb_port):
    _, _, port = lb_port
    status, got = _post(port, POWER, headers={"X-Vllm-Role": "member", "X-Vllm-Models": "m"})
    assert status == 403 and got["error"]["type"] == "admin_only"


def test_with_the_feed_off_a_post_is_503_never_a_quiet_drop(lb_port):
    _, _, port = lb_port
    events.install(None)
    status, got = _post(port, POWER)
    assert status == 503 and "LB_PERF=0" in got["error"]["message"]
