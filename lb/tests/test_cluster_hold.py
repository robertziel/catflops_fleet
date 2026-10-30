"""A PLANNED SWITCH IS WAITED OUT, not answered 503.

Evidence (transcript audit, 2026-10-01/02): 158 Claude Code errors "no vLLM or ollama backend is live",
every one inside a cluster restart or model switch — a window the operator CHOSE and that ends on its
own (restarting → checking → reserving → … → starting → running, 2-10 min; the boot log's 43 serving
starts: p50 501 s, p90 743 s). The proxy answered each turn 503 at once with no `Retry-After`, and a
client's retry budget is spent long before a pair comes up.

Pinned here: while the operator WANTS the cluster and it is on its way, a turn waits server-side —
no byte sent — on the hub's change condition and is served the moment the pair is; bounded by the hold
limit and by a cap on how many wait; a failed or blocked start, the limit, or the cap is a 503
`no_backend` WITH `Retry-After`; and with the cluster desired OFF nothing changes but the header.

The client's own rules, measured against Claude Code 2.1.288: on a 503 it reads ONLY `Retry-After`
(seconds), and one over 60 makes it give up after a single request (`api_request_retry_after_too_long`),
so the header never says more than 60. Where nothing is on its way — a failed or blocked start, or the
cluster desired off with nothing live — `x-should-retry: false` shows the error at once instead of after
ten pointless retries.
"""
from __future__ import annotations

import http.client
import json
import threading
import time

import pytest

from tests.fakes import FakeBackend
from vllm_lb.cluster import DesiredStore, parse_hosts
from vllm_lb.config import Config
from vllm_lb.hub import Hub
from vllm_lb.proxy import Proxy

HOSTS = "box-a-gpu=box-a=user@10.0.0.1=10.9.200.1,box-b-gpu=box-b=user@10.0.0.2=10.9.200.2"
SERVED = "Qwen/Qwen3.8-Flash-Next"


def cluster_coming(*args, **kwargs) -> bool:
    from vllm_lb.proxy import cluster_coming as coming     # imported here: a RED run fails per test
    return coming(*args, **kwargs)


def view(state: str, *, held: bool = False, mode: str | None = "cluster") -> dict:
    return {"configured": True, "state": state, "detail": f"the pair is {state}", "held": held,
            "desired": None if mode is None else {"mode": mode}, "max_depth": 8}


@pytest.fixture
def cluster(tmp_path):
    started: list[tuple[Proxy, FakeBackend]] = []

    def make(state: str | None = "starting", *, switch: bool = False, **over):
        head = FakeBackend(served=(SERVED,))
        cfg = Config(vllm_boxes=(), listen_host="127.0.0.1", listen_port=0,
                     connect_timeout_s=2.0, first_byte_timeout_s=30.0,
                     cluster_name="pair", cluster_hosts=parse_hosts(HOSTS), cluster_url=head.url,
                     cluster_state_path=str(tmp_path / "cluster.json"), **over)
        hub = Hub(cfg)
        if state is not None:
            hub.set_cluster(view(state))
        store = DesiredStore(cfg.cluster_state_path) if switch else None
        proxy = Proxy(cfg, hub, cluster_switch=store)
        proxy.start()
        started.append((proxy, head))
        return proxy, hub, head, store

    yield make
    for proxy, head in started:
        proxy.stop()
        head.close()


def pair_up(hub: Hub, head: FakeBackend) -> None:
    """What the control loop does when the pair answers: the probe says live, the cluster loop says held."""
    hub.update_backend(head.url, live=True, served_models=(SERVED,), max_model_len=262144)
    hub.set_cluster(view("running", held=True))


LAST: dict = {}


def post(proxy: Proxy, timeout: float = 30):
    conn = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=timeout)
    conn.request("POST", "/v1/messages", body=json.dumps({"model": "x", "max_tokens": 1}).encode(),
                 headers={"Content-Type": "application/json"})
    r = conn.getresponse()
    LAST["x-should-retry"] = r.getheader("x-should-retry")
    return r.status, json.loads(r.read() or b"{}"), r.getheader("Retry-After")


def test_a_turn_that_arrives_mid_switch_waits_for_the_pair_and_is_served_when_it_comes_up(cluster):
    proxy, hub, head, _ = cluster("starting")
    threading.Timer(0.8, pair_up, args=(hub, head)).start()
    t0 = time.monotonic()
    status, body, _ = post(proxy)
    assert status == 200, body
    assert 0.7 <= time.monotonic() - t0 < 5.0
    assert len([r for r in head.requests if r["method"] == "POST"]) == 1


@pytest.mark.parametrize("state", ["restarting", "checking", "reserving", "waiting", "draining", "freeing",
                                   "starting", "unknown"])
def test_every_state_a_switch_passes_through_is_waited_out(state):
    assert cluster_coming(view(state)) is True


@pytest.mark.parametrize("state", ["failed", "blocked", "stalled"])
def test_a_start_that_needs_the_operator_is_not_waited_for(state):
    """`stalled` is a hung start halted after its relaunches (cluster.plan): nothing is on its way until the
    operator's Stop, then Start — a held turn is released at once, exactly as for `failed`."""
    assert cluster_coming(view(state)) is False


def test_with_the_cluster_desired_off_nothing_is_waited_for():
    for state in ("unloading", "releasing", "off"):
        assert cluster_coming(view(state, mode="classic")) is False
    assert cluster_coming({"configured": False}) is False


def test_a_view_not_published_yet_goes_by_the_stored_switch():
    """The first seconds after an LB redeploy, before the cluster loop's first tick: no `desired` in the view
    yet. The switch on disk says what the operator wants — and a client retrying onto the new container
    lands exactly there."""
    assert cluster_coming(view("unknown", mode=None), stored_mode="cluster") is True
    assert cluster_coming(view("unknown", mode=None), stored_mode="classic") is False
    assert cluster_coming(view("unknown", mode=None)) is False


def test_a_turn_arriving_before_the_first_tick_after_a_redeploy_waits_for_the_pair(cluster):
    proxy, hub, head, store = cluster(None, switch=True)      # the Hub's own pre-tick view: desired None
    store.set(enabled=True)
    threading.Timer(0.6, pair_up, args=(hub, head)).start()
    status, body, _ = post(proxy)
    assert status == 200, body


def test_a_turn_held_past_the_hold_limit_is_refused_503_no_backend_with_retry_after(cluster):
    proxy, _, head, _ = cluster("starting", cluster_hold_s=0.6, retry_after_s=7)
    t0 = time.monotonic()
    status, body, retry_after = post(proxy)
    assert status == 503 and body["error"]["type"] == "no_backend"
    assert retry_after == "7"
    assert 0.5 <= time.monotonic() - t0 < 4.0
    assert "starting" in body["error"]["message"], "the refusal says what it waited for"
    assert LAST["x-should-retry"] is None, "the pair may still come: the client retries"
    assert not head.requests or all(r["method"] == "GET" for r in head.requests)


def test_a_start_that_fails_while_a_turn_is_held_releases_it_at_once_with_retry_after(cluster):
    proxy, hub, _, _ = cluster("starting")                    # the default hold: far longer than this test
    threading.Timer(0.5, hub.set_cluster, args=(view("failed"),)).start()
    t0 = time.monotonic()
    status, body, retry_after = post(proxy)
    elapsed = time.monotonic() - t0
    assert status == 503 and body["error"]["type"] == "no_backend" and retry_after
    assert LAST["x-should-retry"] == "false", "a failed start needs the operator: no retry storm"
    assert 0.4 <= elapsed < 3.0, "held until the start failed, and not a moment after"


def test_a_failed_cluster_is_refused_at_once_with_retry_after(cluster):
    proxy, _, _, _ = cluster("failed")
    t0 = time.monotonic()
    status, body, retry_after = post(proxy)
    assert status == 503 and body["error"]["type"] == "no_backend" and retry_after
    assert LAST["x-should-retry"] == "false"
    assert time.monotonic() - t0 < 1.0


def test_with_the_cluster_desired_off_nothing_live_is_refused_at_once_with_retry_after(cluster):
    proxy, hub, _, _ = cluster(None)
    hub.set_cluster(view("off", mode="classic"))
    t0 = time.monotonic()
    status, body, retry_after = post(proxy)
    assert status == 503 and body["error"]["type"] == "no_backend"
    assert retry_after is not None and 0 < int(retry_after) <= 60
    assert LAST["x-should-retry"] == "false", "nothing is live and nothing is on its way"
    assert time.monotonic() - t0 < 1.0


def test_held_turns_are_capped_and_the_one_past_the_cap_is_refused_at_once(cluster):
    proxy, hub, head, _ = cluster("starting", hold_max=1, retry_after_s=9)
    first: list = []
    threading.Thread(target=lambda: first.append(post(proxy)), daemon=True).start()
    time.sleep(0.4)                                           # the first is held
    t0 = time.monotonic()
    status, body, retry_after = post(proxy)
    assert status == 503 and body["error"]["type"] == "no_backend" and retry_after == "9"
    assert LAST["x-should-retry"] is None
    assert time.monotonic() - t0 < 1.0
    pair_up(hub, head)
    deadline = time.monotonic() + 5
    while not first and time.monotonic() < deadline:
        time.sleep(0.02)
    assert first and first[0][0] == 200, "the held one is still served when the pair comes"


def test_a_published_change_wakes_a_waiter_without_a_poll():
    hub = Hub(Config(vllm_boxes=()))
    seen = hub.change_version()
    threading.Timer(0.3, hub.set_cluster, args=({"configured": False}, "pair")).start()
    t0 = time.monotonic()
    after = hub.wait_for_change(seen, 10.0)
    assert after != seen and time.monotonic() - t0 < 2.0


def test_the_hold_comes_from_the_environment():
    cfg = Config.from_env({"LB_ALLOWED_HOSTS": "loopback", "LB_CLUSTER_HOLD_S": "300", "LB_HOLD_MAX": "4", "LB_RETRY_AFTER_S": "12"})
    assert (cfg.cluster_hold_s, cfg.hold_max, cfg.retry_after_s) == (300.0, 4, 12.0)
    d = Config()
    assert (d.cluster_hold_s, d.hold_max, d.retry_after_s) == (900.0, 64, 30.0)


def test_retry_after_never_says_more_than_the_client_will_wait(cluster):
    """Over 60 s, Claude Code gives up after ONE request rather than wait — a long Retry-After is worse
    than none."""
    proxy, _, _, _ = cluster("starting", cluster_hold_s=0.3, retry_after_s=300)
    status, _, retry_after = post(proxy)
    assert status == 503 and retry_after == "60"
