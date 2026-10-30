"""The swarm control on the gateway: what the page may ASK FOR, and what it may only READ.

WHAT THE BUTTON MEANS, SINCE 2026-10-01. One click asks the host for ONE `fullclaudevllm`
deep-research session against the model ALREADY LOADED. The operator's words:

    "I should be able to click run benchmark any time, but that only starts deep research swarm,
     nothing more, we assume benchmark data are collected all the way even on normal requests"
    "benchmark run should not unload model, should use already loaded model, the multi settings
     benchmark should not be part of the button run on web nor mobile"

Until that morning the same button started `worklog/bench/fullclaudevllm_queue.sh`: thirty runs,
thirteen hours, the TP=2 pair torn down and rebuilt per configuration. At 05:53 the operator pressed
it and it did exactly that to their working 16/8192/4 default. So these tests hold two lines:

  * the gateway asks for a SESSION and for nothing else, and there is no endpoint, no body and no
    code path here by which the matrix can be started from a page;
  * it refuses, in words, the only two things that make a swarm wrong — one already running (or the
    matrix running), and no model loaded to put under load, because a swarm never loads one.

A start is a REQUEST, left in the LB's own state directory for `lb/bench_queue_runner.py`. The LB
records it and says so; whether a session starts is the runner's to decide and to report.
"""
from __future__ import annotations

import http.client
import json
import time

import pytest

from tests.fakes import FakeBackend
import bench_session
from vllm_lb.benchswarm import (ACTION, CAP_MIN, CAP_MIN_RANGE, GATEWAY_MARK, REQUEST, RUNNER,
                                RUNNER_STALE_S, BenchSwarm, SwarmRefused, SwarmUnavailable)
from vllm_lb.config import Config, VllmBox
from vllm_lb.hub import Hub
from vllm_lb.proxy import Proxy

LOADED = (True, "the two-box cluster is serving qwen3.8:latest")
NOTHING = (False, "no vLLM is serving on this gateway (the cluster is stopped)")


def swarm(tmp_path, loaded=LOADED):
    return BenchSwarm(str(tmp_path / "benchq"), loaded=lambda: loaded)


def reports(tmp_path, **report):
    """What the host runner last left behind (lb/bench_queue_runner.py writes exactly this)."""
    state = tmp_path / "benchq"
    state.mkdir(parents=True, exist_ok=True)
    body = {"version": 2, "at": time.time(), "pid": 4242, "cap_min": 15.0,
            "session": {"alive": False, "pid": None, "since": None, "label": None, "phase": None,
                        "cap_min": None, "claude_pid": None, "started_at": None, "ended_at": None,
                        "rc": None, "detail": "", "guards": None, "at": None},
            "matrix": {"running": False, "pid": None, "since": None, "script": None},
            "last": None}
    for key, value in report.items():
        if key in ("session", "matrix") and isinstance(value, dict):
            body[key] = {**body[key], **value}
        else:
            body[key] = value
    (state / RUNNER).write_text(json.dumps(body))
    return body


# ── the view: what the control is allowed to sit beside ───────────────────────


def test_a_gateway_with_no_state_directory_reports_none():
    """No state directory = the control is off and nothing renders — the same shape the cluster
    reports when it is not configured."""
    assert BenchSwarm("").view() == {"configured": False}
    assert BenchSwarm("").configured is False


def test_the_view_says_what_is_loaded_and_what_the_cap_is(tmp_path):
    q = swarm(tmp_path)
    reports(tmp_path, cap_min=7.5)
    v = q.view()
    assert v["configured"] is True and v["model_loaded"] is True
    assert "serving qwen3.8" in v["model_detail"]
    assert v["cap_min"] == 7.5, "the page names the cap the host runner will actually use"


def test_a_gateway_that_cannot_tell_what_is_loaded_says_so_rather_than_no(tmp_path):
    """Not knowing is its own answer: `None` must never be drawn, or acted on, as "nothing is
    loaded" — that would refuse a swarm the operator can perfectly well have."""
    q = BenchSwarm(str(tmp_path / "benchq"))
    assert q.view()["model_loaded"] is None
    q.request_session()                       # and it does not refuse over it
    assert q.view()["request_pending"] is True


def test_a_probe_that_raises_costs_the_answer_not_the_status_poll(tmp_path):
    def boom():
        raise RuntimeError("the hub went away")
    q = BenchSwarm(str(tmp_path / "benchq"), loaded=boom)
    v = q.view()
    assert v["model_loaded"] is None and "RuntimeError" in v["model_detail"]


def test_no_runner_has_ever_reported_so_nothing_is_listening(tmp_path):
    """The page must then say the request cannot be served, not that a swarm is coming — and name what
    to start, never a box: with no report there is no configured box to name, and a literal one would
    be a deployment written into the code (AGENTS.md, open-source ready)."""
    r = swarm(tmp_path).view()["runner"]
    assert r["listening"] is False and r["at"] is None
    assert "is not running" in r["error"] and "lb/bench_runner" in r["error"]
    assert "box-c" not in r["error"]
    assert r["box"] is None
    assert r["cap_min"] == CAP_MIN, "the button still has a number to show"


def test_the_runner_names_its_own_box_and_the_view_carries_it(tmp_path):
    """Where the runner is comes from the runner's configuration (`--box`), through its report."""
    q = swarm(tmp_path)
    reports(tmp_path, box="box-under-test")
    assert q.view()["runner"]["box"] == "box-under-test"
    reports(tmp_path, box="box-under-test", at=time.time() - RUNNER_STALE_S - 60)
    r = q.view()["runner"]
    assert r["listening"] is False and "on box-under-test" in r["error"]
    assert "box-c" not in r["error"]


def test_a_box_that_is_not_a_string_is_not_drawn(tmp_path):
    q = swarm(tmp_path)
    reports(tmp_path, box={"name": "x"})
    assert q.view()["runner"]["box"] is None


def test_a_runner_that_stopped_reporting_is_not_listening(tmp_path):
    q = swarm(tmp_path)
    reports(tmp_path, at=time.time() - RUNNER_STALE_S - 60)
    r = q.view()["runner"]
    assert r["listening"] is False
    assert "is not listening" in r["error"] and r["age_s"] > RUNNER_STALE_S


def test_a_live_runner_carries_the_session_and_the_matrix_apart(tmp_path):
    q = swarm(tmp_path)
    reports(tmp_path,
            session={"alive": True, "pid": 900104, "phase": "running", "cap_min": 15.0,
                     "started_at": time.time() - 240, "label": "S-20260101-000001"},
            matrix={"running": True, "pid": 900102, "script": "fullclaudevllm_queue.sh"})
    r = q.view()["runner"]
    assert r["listening"] is True and r["error"] == ""
    assert r["session"]["alive"] is True and r["session"]["phase"] == "running"
    assert r["matrix"]["running"] is True and r["matrix"]["pid"] == 900102


def test_a_corrupt_runner_report_is_reported_not_believed(tmp_path):
    q = swarm(tmp_path)
    state = tmp_path / "benchq"
    state.mkdir(parents=True, exist_ok=True)
    (state / RUNNER).write_text("{not json")
    r = q.view()["runner"]
    assert r["listening"] is False and "JSONDecodeError" in r["error"]


# ── the request: one session, and only a request ──────────────────────────────


def test_a_click_leaves_a_session_request_and_starts_nothing(tmp_path):
    q = swarm(tmp_path)
    v = q.request_session()
    left = json.loads((tmp_path / "benchq" / REQUEST).read_text())
    assert left["action"] == ACTION == "session"
    assert left["by"] == GATEWAY_MARK and left["id"]
    assert v["request"]["id"] == left["id"]
    assert v["request_pending"] is True, "nothing has answered it yet"


def test_the_gateway_has_no_way_to_ask_for_the_matrix(tmp_path):
    """Not as a second action, not as an argument. The thirty-run matrix is started at the keyboard
    with `bench_queue_runner.py --start-matrix`, and the runner refuses any other action by name."""
    q = swarm(tmp_path)
    q.request_session()
    left = json.loads((tmp_path / "benchq" / REQUEST).read_text())
    assert set(left) == {"id", "at", "by", "action"}
    assert left["action"] == "session"
    assert not hasattr(q, "set_hold"), "the matrix's park is not a lever on this page any more"


def test_a_request_stays_pending_until_the_runner_names_that_very_request(tmp_path):
    """The page says "requested" off this flag. A runner that answered some EARLIER request must not
    make a newer one read as served."""
    q = swarm(tmp_path)
    rid = q.request_session()["request"]["id"]
    reports(tmp_path, last={"id": "an-older-one", "result": "started", "detail": "..."})
    assert q.view()["request_pending"] is True
    reports(tmp_path, last={"id": rid, "result": "started", "detail": "pid 99"})
    v = q.view()
    assert v["request_pending"] is False
    assert v["runner"]["last"]["result"] == "started"


def test_two_clicks_leave_two_requests_and_neither_of_them_starts_anything(tmp_path):
    """Safe to click twice: the runner refuses to start a second swarm, so the worst a double click
    can do is ask twice."""
    q = swarm(tmp_path)
    first = q.request_session()["request"]["id"]
    second = q.request_session()["request"]["id"]
    assert first != second
    assert q.view()["request_pending"] is True


def test_a_second_swarm_is_refused_and_says_it_is_a_swarm(tmp_path):
    q = swarm(tmp_path)
    reports(tmp_path, session={"alive": True, "pid": 900104, "phase": "running",
                               "label": "S-20260101-000001"})
    with pytest.raises(SwarmRefused) as exc:
        q.request_session()
    assert "already running" in str(exc.value) and "S-20260101-000001" in str(exc.value)
    assert not (tmp_path / "benchq" / REQUEST).exists(), "a refusal writes no request"


def test_a_swarm_is_refused_while_the_matrix_is_running_and_says_it_is_the_matrix(tmp_path):
    q = swarm(tmp_path)
    reports(tmp_path, matrix={"running": True, "pid": 900102,
                              "script": "fullclaudevllm_queue.sh"})
    with pytest.raises(SwarmRefused) as exc:
        q.request_session()
    assert "run matrix is running" in str(exc.value) and "fullclaudevllm_queue.sh" in str(exc.value)
    assert "box-c" not in str(exc.value), "the box is the runner's to name, never a literal"
    reports(tmp_path, box="box-under-test", matrix={"running": True, "pid": 900102,
                                                    "script": "fullclaudevllm_queue.sh"})
    with pytest.raises(SwarmRefused) as exc:
        q.request_session()
    assert "run matrix is running on box-under-test" in str(exc.value)


def test_a_swarm_is_refused_when_no_model_is_loaded_because_it_never_loads_one(tmp_path):
    """The operator's rule: the swarm puts the model ALREADY LOADED under load. Loading one is their
    own deliberate act on the cluster panel, never a side effect of this button."""
    q = swarm(tmp_path, loaded=NOTHING)
    with pytest.raises(SwarmRefused) as exc:
        q.request_session()
    assert "no model is loaded" in str(exc.value)
    assert "never loads one" in str(exc.value) and "cluster panel" in str(exc.value)
    assert not (tmp_path / "benchq" / REQUEST).exists()


def test_a_swarm_may_be_asked_for_while_a_dead_session_report_sits_there(tmp_path):
    """The operator asked to be able to click it any time. A finished session is not a running one,
    and neither is one the runner lost."""
    q = swarm(tmp_path)
    for phase in ("finished", "lost", "failed"):
        reports(tmp_path, session={"alive": False, "phase": phase, "rc": 124})
        assert q.request_session()["request_pending"] is True


def test_a_gateway_with_no_state_directory_refuses_loudly(tmp_path):
    with pytest.raises(SwarmUnavailable) as exc:
        BenchSwarm("").request_session()
    assert "nowhere to be left" in str(exc.value)


# ── over HTTP ─────────────────────────────────────────────────────────────────


@pytest.fixture
def backend():
    b = FakeBackend()
    yield b
    b.close()


def serving(backend, state_dir, *, live=True):
    cfg = Config(vllm_boxes=(VllmBox("box-a-gpu", "box-a", backend.url),), 
                 listen_host="127.0.0.1", listen_port=0, swarm_state=state_dir)
    hub = Hub(cfg)
    if live:
        hub.update_backend(backend.url, live=True, served_models=("qwen3.8:latest",),
                           last_probe_at=time.time())
    p = Proxy(cfg, hub)
    p.start()
    return p


def call(proxy, method, path, body=None):
    c = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=10)
    headers = {"Content-Type": "application/json"} if body is not None else {}
    c.request(method, path, json.dumps(body) if body is not None else None, headers)
    r = c.getresponse()
    return r.status, json.loads(r.read() or b"null")


def test_the_status_poll_the_page_already_makes_carries_the_swarm(tmp_path, backend):
    """No second poll and no new channel: the panel rides /lb/status, the way the cluster does."""
    p = serving(backend, str(tmp_path / "benchq"))
    try:
        status, body = call(p, "GET", "/lb/status")
        assert status == 200
        assert body["swarm"]["configured"] is True
        assert body["swarm"]["model_loaded"] is True
        assert "qwen3.8:latest" in body["swarm"]["model_detail"]
        assert "bench_queue" not in body, "the matrix's progress is not what this control means"
    finally:
        p.stop()


def test_a_gateway_without_a_state_directory_carries_the_unconfigured_shape(tmp_path, backend):
    p = serving(backend, "")
    try:
        _, body = call(p, "GET", "/lb/status")
        assert body["swarm"] == {"configured": False}
    finally:
        p.stop()


def test_the_start_endpoint_records_a_request_and_calls_it_pending(tmp_path, backend):
    """202, not 200: the intent is written, and something else acts on it — the same shape the
    cluster switch uses for a restart the loop will perform."""
    state = tmp_path / "benchq"
    p = serving(backend, str(state))
    try:
        status, body = call(p, "POST", "/lb/swarm/start", {})
        assert status == 202
        assert body["swarm"]["request_pending"] is True
        assert body["swarm"]["runner"]["listening"] is False
        assert json.loads((state / REQUEST).read_text())["action"] == "session"
    finally:
        p.stop()


def test_the_start_endpoint_refuses_with_409_when_nothing_is_serving(tmp_path, backend):
    """Server-side, not only a disabled button: a curl must not be able to start a swarm against a
    pair that is not up, because the swarm would not load it."""
    state = tmp_path / "benchq"
    p = serving(backend, str(state), live=False)
    try:
        status, body = call(p, "POST", "/lb/swarm/start", {})
        assert status == 409
        assert body["error"]["type"] == "swarm_rejected"
        assert "no model is loaded" in body["error"]["message"]
        assert not (state / REQUEST).exists()
    finally:
        p.stop()


def test_the_start_endpoint_refuses_with_409_while_a_swarm_is_running(tmp_path, backend):
    state = tmp_path / "benchq"
    reports(tmp_path, session={"alive": True, "pid": 900104, "phase": "running"})
    p = serving(backend, str(state))
    try:
        status, body = call(p, "POST", "/lb/swarm/start", {})
        assert status == 409
        assert "already running" in body["error"]["message"]
    finally:
        p.stop()


def test_the_start_is_refused_where_no_swarm_state_is_configured(tmp_path, backend):
    p = serving(backend, "")
    try:
        status, body = call(p, "POST", "/lb/swarm/start", {})
        assert status == 409
        assert body["error"]["type"] == "swarm_rejected"
    finally:
        p.stop()


def test_the_matrix_has_no_endpoint_on_this_gateway(tmp_path, backend):
    """The park and the queue start are GONE. Nothing a browser, a prefetch or a crawler can reach
    touches the thirty-run matrix or the harness's own directory."""
    state = tmp_path / "benchq"
    p = serving(backend, str(state))
    try:
        for path in ("/lb/bench-queue/hold", "/lb/bench-queue/start", "/lb/swarm/hold",
                     "/lb/swarm/matrix"):
            status, _ = call(p, "POST", path, {"enabled": True})
            assert status in (404, 405), path
        assert not (state / REQUEST).exists()
    finally:
        p.stop()


def test_the_control_takes_no_other_method(tmp_path, backend):
    """A GET of it would be a link, and a link that changes state is one a prefetch or a crawler
    follows. Its state is read from /lb/status only."""
    state = tmp_path / "benchq"
    p = serving(backend, str(state))
    try:
        status, _ = call(p, "GET", "/lb/swarm/start")
        assert status in (404, 405)
        assert not (state / REQUEST).exists()
    finally:
        p.stop()


# ── the page's own honesty ────────────────────────────────────────────────────


def page_text() -> str:
    """Everything the gateway page can put in front of a person: its markup and its script."""
    import pathlib
    static = pathlib.Path(__file__).resolve().parent.parent / "vllm_lb" / "static"
    return ((static / "index.html").read_text() + (static / "gateway.js").read_text()).lower()


def test_the_page_says_what_the_control_does_and_does_not_do():
    """It generates load against the model already loaded, and it reconfigures nothing. Someone who
    has read none of this has to be able to tell that from the page alone."""
    text = page_text()
    assert "already loaded" in text
    assert "nothing is restarted" in text or "restarts nothing" in text
    assert "bench_queue_runner" in text, "the page names the host runner it depends on"


def _swarm_section() -> str:
    """gateway.js's swarm control, from its in-flight flag to its hold handler — the part of the page
    this control owns."""
    import pathlib
    src = (pathlib.Path(__file__).resolve().parent.parent / "vllm_lb" / "static" / "gateway.js").read_text()
    return src[src.index("let swarmBusy"):src.index("function swarmHeld")]


def test_the_swarm_control_names_no_box_of_its_own_and_has_words_for_an_unsealed_runner():
    """Which box the runner is on is the runner's configuration (`--box`, carried in its report), never
    a word in the page: the same page serves every deployment (AGENTS.md, open-source ready). And the
    sealed runner's refusal — its egress check failed, so claude was not started — has its own words."""
    section = _swarm_section()
    import pathlib
    html = (pathlib.Path(__file__).resolve().parent.parent / "vllm_lb" / "static" / "index.html").read_text()
    block = html[html.index("THE TEST SWARM"):html.index('id="swarm-error"')]
    for word in ("box-c", "box-a", "site-b"):
        assert word not in section.lower(), f"gateway.js's swarm section names {word}"
        assert word not in block.lower(), f"index.html's swarm block names {word}"
    assert "runner.box" in section or "(q.runner || {}).box" in section
    assert "unsealed" in section


def test_the_page_never_offers_the_thirty_run_matrix():
    """Not as a button, not behind a confirm, not in a menu (the operator, 2026-10-01: "the multi
    settings benchmark should not be part of the button run on web nor mobile")."""
    text = page_text()
    for gone in ("fullclaudevllm_queue", "park the queue", "release the queue", "bench-queue/hold",
                 "bench-queue/start", "queue.txt", "done.txt", "finished + "):
        assert gone not in text, f"the page still mentions {gone!r}"


def test_the_page_says_the_cap_is_a_success_and_an_early_end_is_not():
    """`timeout`'s 124 is the normal end of a swarm. A page that drew that as a failure — or an
    exit 0 as a success — would have the one number in this control backwards."""
    text = page_text()
    # The outcome line's one word per exit code (gateway.js `swOutcome`): "capped" for 124, "ended early"
    # for anything else on a finished session — the words the operator reads, not a sentence.
    assert "ended early" in text
    assert "capped" in text
    assert "the cap" in text


# ── HOW LONG, since 2026-10-05 ────────────────────────────────────────────────
# The operator asked for the duration to be theirs: 15 min stays the default, 30 and more are allowed.
# The cap is the ONE argument this control takes, so these hold the three things that can go wrong with
# a knob — the default drifting, a value the LB would refuse being offered, and a value out of range
# being rounded into one that is in it instead of being refused.


def test_the_lbs_half_and_the_hosts_half_agree_on_the_default_and_the_range():
    """Two copies, and not by choice: `.dockerignore` keeps `bench_session.py` out of the LB image (it
    runs in the sealed runner, not in the LB) so `benchswarm` CANNOT import it, and `bench_session.py`
    imports only `vllm_lb.engine_metrics`, not this module. This assertion is the only thing between them."""
    assert CAP_MIN == bench_session.CAP_MIN == 15.0, "fifteen minutes is still the default"
    assert CAP_MIN_RANGE == bench_session.CAP_MIN_RANGE
    lo, hi = CAP_MIN_RANGE
    assert lo <= CAP_MIN <= hi, "the default has to be inside the range it is the default of"
    assert hi >= 30.0, "the operator asked for 30 minutes or more (2026-10-05)"


def test_the_view_carries_the_range_so_the_page_offers_only_what_this_lb_takes(tmp_path):
    """The page builds its list from this, not from its own markup: an older gateway must not be made to
    offer a duration it would answer with a 400 — the same reason `cluster.limits` is published."""
    q = swarm(tmp_path)
    reports(tmp_path)
    v = q.view()
    assert v["cap_min_range"] == list(CAP_MIN_RANGE)
    assert v["cap_min"] == CAP_MIN, "and beside it the cap a click with no duration would get"


def test_a_click_that_names_no_duration_leaves_exactly_the_request_it_always_did(tmp_path):
    """The whole point of the default: an unchanged caller is unchanged. No `cap_min` field at all, so a
    runner started before this change — and the runner's own `--cap-min` — decide it exactly as before."""
    q = swarm(tmp_path)
    q.request_session()
    req = json.loads((tmp_path / "benchq" / REQUEST).read_text())
    assert set(req) == {"id", "at", "by", "action"}, f"an extra field reached the host runner: {req}"
    assert req["action"] == ACTION and req["by"] == GATEWAY_MARK


def test_a_click_may_ask_for_thirty_minutes_and_the_request_carries_them(tmp_path):
    q = swarm(tmp_path)
    q.request_session(30)
    req = json.loads((tmp_path / "benchq" / REQUEST).read_text())
    assert req["cap_min"] == 30.0 and req["action"] == ACTION


@pytest.mark.parametrize("asked", [240, 5, 17.5])
def test_every_duration_inside_the_range_is_written_as_it_was_asked_for(tmp_path, asked):
    """Including the ends and a fraction: `timeout_arg` writes a whole number of minutes as minutes and
    anything else exactly, so there is nothing here to round."""
    swarm(tmp_path).request_session(asked)
    assert json.loads((tmp_path / "benchq" / REQUEST).read_text())["cap_min"] == float(asked)


@pytest.mark.parametrize("asked", [4, 0, -15, 241, 10_000, True, "30", float("nan"), float("inf")])
def test_a_duration_the_gateway_would_not_take_is_refused_before_anything_is_written(tmp_path, asked):
    """Refused, not clamped. A swarm that ran fifteen minutes because two hours would not fit is a run
    whose minutes nobody can read back, and `cap_min: true` is one minute if `bool` is not excluded
    first (`isinstance(True, int)` is true in Python)."""
    q = swarm(tmp_path)
    with pytest.raises(ValueError) as exc:
        q.request_session(asked)
    assert "cap_min" in str(exc.value) and "5" in str(exc.value) and "240" in str(exc.value)
    assert not (tmp_path / "benchq" / REQUEST).exists(), "a refused duration still left a request"


# ── and over HTTP, the way the button asks ────────────────────────────────────


def test_the_start_endpoint_takes_a_duration_and_records_it(tmp_path, backend):
    state = tmp_path / "benchq"
    p = serving(backend, str(state))
    try:
        status, body = call(p, "POST", "/lb/swarm/start", {"cap_min": 30})
        assert status == 202 and body["swarm"]["request_pending"] is True
        assert json.loads((state / REQUEST).read_text())["cap_min"] == 30.0
    finally:
        p.stop()


def test_a_json_start_with_no_body_is_still_the_fifteen_minute_default(tmp_path, backend):
    """This endpoint took no arguments until 2026-10-05. A JSON start that names no duration — an empty
    body under `Content-Type: application/json` — is the host runner's own default, as it always was."""
    state = tmp_path / "benchq"
    p = serving(backend, str(state))
    try:
        c = http.client.HTTPConnection("127.0.0.1", p.port, timeout=10)
        c.request("POST", "/lb/swarm/start", None, {"Content-Type": "application/json"})
        r = c.getresponse()
        r.read()
        assert r.status == 202
        assert "cap_min" not in json.loads((state / REQUEST).read_text())
    finally:
        p.stop()


@pytest.mark.parametrize("headers, body", [
    ({}, None),                                                   # a bare POST: no type at all
    ({"Content-Type": "text/plain"}, b"{}"),                      # what a cross-site fetch may send unpreflighted
    ({"Content-Type": "text/plain; x=json"}, b"{}"),              # still text/plain to a browser: no preflight
    ({"Content-Type": "application/x-www-form-urlencoded"}, b"cap_min=30"),   # an auto-submitting form
    ({"Content-Type": "multipart/form-data; boundary=x"}, b"--x--"),
])
def test_a_start_that_a_cross_site_page_could_send_is_refused_and_starts_nothing(tmp_path, backend,
                                                                                   headers, body):
    """CSRF. A browser sends a "simple" POST — no type, text/plain, a form — to ANY origin without asking
    first, with the gateway's cached basic-auth credentials attached. Every such shape is one a page on
    another site can fire at this endpoint, so a start has to be JSON: that type makes the browser ask
    first (a preflight), and this gateway grants no cross-origin access. A 415, and no request file."""
    state = tmp_path / "benchq"
    p = serving(backend, str(state))
    try:
        c = http.client.HTTPConnection("127.0.0.1", p.port, timeout=10)
        c.request("POST", "/lb/swarm/start", body, headers)
        r = c.getresponse()
        answer = json.loads(r.read() or b"null")
        assert r.status == 415
        assert "application/json" in answer["error"]["message"]
        assert not (state / REQUEST).exists(), "a refused start left a request for the runner"
    finally:
        p.stop()


@pytest.mark.parametrize("body", [{"cap_min": 4}, {"cap_min": 241}, {"cap_min": 0},
                                  {"cap_min": "30"}, {"cap_min": True}, {"cap_min": -30}])
def test_a_duration_out_of_range_is_a_400_naming_the_field_and_its_range(tmp_path, backend, body):
    """The shape /lb/cluster refuses a knob with (`proxy.cluster_switch_request`): a 400, the field in
    backticks, and the range it would have had to be in — never a 202 for a run nobody asked for."""
    state = tmp_path / "benchq"
    p = serving(backend, str(state))
    try:
        status, answer = call(p, "POST", "/lb/swarm/start", body)
        assert status == 400, answer
        assert answer["error"]["type"] == "invalid_request"
        message = answer["error"]["message"]
        assert "`cap_min`" in message and "5" in message and "240" in message
        assert not (state / REQUEST).exists(), "a refused duration still started a swarm"
    finally:
        p.stop()


def test_the_page_offers_the_duration_and_takes_its_range_from_the_lb():
    """The chooser exists, it posts the field the gateway reads, and the numbers it offers come from
    /lb/status — a page that carried its own range would offer a duration an older LB would 400."""
    text = page_text()
    assert "swarm-cap" in text, "the page has no duration chooser"
    assert "cap_min" in text
    assert "cap_min_range" in text, "the page's range must come from the LB, not from its own markup"
