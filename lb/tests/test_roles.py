"""Roles at the LB (vllm_lb/roles.py): a MEMBER's models, turns and view of the fleet, from the gate's
X-Vllm-Role / X-Vllm-Models; an admin and a caller inside the trust boundary (no header) unchanged."""
from __future__ import annotations

import http.client
import json
from types import SimpleNamespace

import pytest

from tests.fakes import FakeBackend
from vllm_lb import roles
from vllm_lb.config import Config, VllmBox
from vllm_lb.hub import Hub
from vllm_lb.proxy import Proxy
from vllm_lb.state import LEASE_RUNNING, Lease

SERVED, OTHER = "qwen3.8:latest", "other:latest"
MEMBER = {"X-Vllm-Role": "member", "X-Vllm-Models": SERVED}


# ── the headers ─────────────────────────────────────────────────────────────────────────────────────────────────────
def test_no_role_header_is_a_caller_inside_the_trust_boundary():
    assert roles.caller({}) is None


def test_a_members_models_header_may_carry_spaces_after_its_commas():
    assert roles.caller({"X-Vllm-Role": "member", "X-Vllm-Models": "a/m, b/m ,"}).models == {"a/m", "b/m"}


def test_a_member_turn_routed_to_a_server_of_another_model_is_refused_as_that_not_as_unloaded():
    status, body = roles.routed_elsewhere(SERVED)
    assert status == 503 and body["error"]["type"] == "model_not_routed"
    assert SERVED in body["error"]["message"] and "not loaded" not in body["error"]["message"]


def test_an_admin_is_unrestricted():
    assert roles.caller({"X-Vllm-Role": "admin", "X-Vllm-Models": ""}) is None


def test_a_member_carries_their_models():
    assert roles.caller({"X-Vllm-Role": "member", "X-Vllm-Models": "a/m,b/m"}).models == {"a/m", "b/m"}
    assert roles.caller({"X-Vllm-Role": "member"}).models == frozenset()


@pytest.mark.parametrize("role", ["unset", "", "Admin", "tenant"])
def test_an_unknown_role_is_refused_never_read_as_either(role):
    with pytest.raises(roles.UnknownRole):
        roles.caller({"X-Vllm-Role": role})


# ── a member's turn ─────────────────────────────────────────────────────────────────────────────────────────────────
M = roles.Member(frozenset({"a/m"}))


def test_a_granted_served_model_goes():
    assert roles.turn_refusal(M, "a/m", served=True) is None


@pytest.mark.parametrize("model,status,kind", [(None, 400, "model_required"), ("", 400, "model_required"),
                                               ("b/m", 403, "model_not_granted")])
def test_a_turn_naming_no_model_or_another_is_refused_by_name(model, status, kind):
    got = roles.turn_refusal(M, model, served=True)
    assert got[0] == status and got[1]["error"]["type"] == kind


def test_a_granted_model_not_loaded_is_503_never_another_model():
    status, body = roles.turn_refusal(M, "a/m", served=False)
    assert status == 503 and body["error"]["type"] == "model_not_loaded" and "a/m" in body["error"]["message"]


# ── a member's view of /lb/status ───────────────────────────────────────────────────────────────────────────────────
def fleet():
    spec = lambda n: SimpleNamespace(served_name=n)                       # noqa: E731
    box = lambda n: SimpleNamespace(name=n)                               # noqa: E731
    return SimpleNamespace(clusters={
        "mine": SimpleNamespace(choices=(SimpleNamespace(spec=spec("a/m")),), boxes=(box("box-a"),)),
        "theirs": SimpleNamespace(choices=(SimpleNamespace(spec=spec("b/m")),), boxes=(box("box-b"),)),
    })


STATUS = {
    "now": 1, "backends": [{"url": "u1", "box": "box-a", "cluster": "mine", "served_models": ["a/m"]},
                           {"url": "u2", "box": "box-b", "cluster": "theirs", "served_models": ["b/m"]}],
    "clusters": {"mine": {"serves": None}, "theirs": {"serves": {"served_name": "b/m"}}},
    "cluster": {"configured": True, "state": "running"},
    "hardware": {"box-a": {"cpu_percent": 1}, "box-b": {"cpu_percent": 2}},
    "models": {"cluster": [{"served_name": "a/m"}, {"served_name": "b/m"}], "classic": [], "error": None},
    "leases": [{"lease_id": "x"}], "desired_leases": 2, "declared": [{"n": 1}], "swarm": {"configured": True},
    "queue": {"inflight": [{"prompt": "someone's secret"}], "inflight_total": 1, "recent": [{"prompt": "p"}],
              "counts": {"streaming": 1}},
    "start_rates": {"x": 1}, "setup_targets": {"x": 1},
}


def test_a_member_sees_the_servers_and_boxes_that_can_load_their_models_only():
    view = roles.member_status(STATUS, M, fleet(), cluster_name="theirs")
    assert [b["box"] for b in view["backends"]] == ["box-a"]
    assert list(view["clusters"]) == ["mine"]
    assert list(view["hardware"]) == ["box-a"]
    assert view["models"]["cluster"] == [{"served_name": "a/m"}]
    assert view["cluster"] == {"configured": False}            # the one cluster is "theirs"
    assert (view["role"], view["granted_models"]) == ("member", ["a/m"])


def test_a_member_sees_nobody_elses_requests_leases_or_runs():
    view = roles.member_status(STATUS, M, fleet())
    assert view["queue"]["inflight"] == [] and view["queue"]["recent"] == [] and view["queue"]["inflight_total"] == 0
    assert view["queue"]["counts"] == {"streaming": 1}              # how busy the fleet is: an aggregate
    assert (view["leases"], view["declared"], view["swarm"]) == ([], [], {"configured": False})
    assert "someone's secret" not in json.dumps(view)


def test_a_server_serving_a_granted_model_is_shown_even_when_no_fleet_cluster_declares_it():
    status = {**STATUS, "clusters": {"loose": {"serves": {"served_name": "a/m"}}}}
    assert list(roles.member_status(status, M, SimpleNamespace(clusters={}))["clusters"]) == ["loose"]


def test_a_member_with_no_models_sees_no_server():
    view = roles.member_status(STATUS, roles.Member(frozenset()), fleet())
    assert (view["backends"], view["clusters"], view["hardware"]) == ([], {}, {})


# ── through the proxy ───────────────────────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def lb():
    made = []

    def make():
        engine = FakeBackend(served=(SERVED,))
        cfg = Config(vllm_boxes=(VllmBox("box-a-gpu", "box-a", engine.url),), listen_host="127.0.0.1", listen_port=0,
                     connect_timeout_s=2.0, first_byte_timeout_s=30.0, admission_wait_s=0)
        hub = Hub(cfg)
        b = hub.backend_for(engine.url)
        b.live, b.served_models, b.depth_known, b.waiting = True, (SERVED,), True, 1
        hub.set_leases([Lease(lease_id="r", job_id="j", created_at=0, status=LEASE_RUNNING,
                              host_label="box-a-gpu", box="box-a")])
        proxy = Proxy(cfg, hub)
        proxy.start()
        made.append((proxy, engine))
        return proxy, engine

    yield make
    for proxy, engine in made:
        proxy.stop()
        engine.close()


def call(port, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    data = json.dumps(body).encode() if body is not None else None
    conn.request(method, path, body=data, headers={"Content-Type": "application/json", **(headers or {})})
    r = conn.getresponse()
    raw = r.read()
    return r.status, (json.loads(raw) if raw and r.getheader("Content-Type", "").startswith("application/json")
                      else raw)


def turn(model):
    return {"model": model, "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]}


def test_a_members_models_list_is_their_granted_models(lb):
    proxy, _ = lb()
    assert [d["id"] for d in call(proxy.port, "GET", "/v1/models")[1]["data"]] == [SERVED]
    status, body = call(proxy.port, "GET", "/v1/models", headers={"X-Vllm-Role": "member", "X-Vllm-Models": OTHER})
    assert status == 200 and body["data"] == []


def test_a_member_sends_a_turn_to_a_granted_model_and_the_engine_never_sees_the_role(lb):
    proxy, engine = lb()
    status, _ = call(proxy.port, "POST", "/v1/messages", turn(SERVED), MEMBER)
    assert status == 200
    (sent,) = [r for r in engine.requests if r["method"] == "POST"]
    assert not {"x-vllm-role", "x-vllm-models"} & {k.lower() for k in sent["headers"]}


def test_a_members_turn_to_another_model_is_refused_and_never_sent(lb):
    proxy, engine = lb()
    status, body = call(proxy.port, "POST", "/v1/messages", turn(OTHER),
                        {"X-Vllm-Role": "member", "X-Vllm-Models": f"{SERVED},{OTHER}"})
    assert status == 503 and body["error"]["type"] == "model_not_loaded"
    status, body = call(proxy.port, "POST", "/v1/messages", turn("nope:latest"), MEMBER)
    assert status == 403 and body["error"]["type"] == "model_not_granted"
    assert not [r for r in engine.requests if r["method"] == "POST"]


def test_an_admin_and_an_internal_caller_send_as_before(lb):
    proxy, _ = lb()
    assert call(proxy.port, "POST", "/v1/messages", turn(SERVED), {"X-Vllm-Role": "admin"})[0] == 200
    assert call(proxy.port, "POST", "/v1/messages", turn(SERVED))[0] == 200


@pytest.mark.parametrize("method,path", [("POST", "/lb/cluster"), ("POST", "/lb/clusters/pair"), ("POST", "/v1/embeddings"),
                                         ("POST", "/lb/requests/x/kill"), ("GET", "/lb/bench"),
                                         ("POST", "/lb/swarm/start"), ("GET", "/lb/perf")])
def test_a_member_is_refused_the_lbs_own_controls(lb, method, path):
    proxy, _ = lb()
    status, body = call(proxy.port, method, path, {} if method == "POST" else None, MEMBER)
    assert status == 403 and body["error"]["type"] == "admin_only"


def test_a_members_status_is_narrowed_and_says_so(lb):
    proxy, _ = lb()
    status, body = call(proxy.port, "GET", "/lb/status", headers=MEMBER)
    assert status == 200 and body["role"] == "member" and body["granted_models"] == [SERVED]
    assert [b["served_models"] for b in body["backends"]] == [[SERVED]]
    assert body["leases"] == []
    status, body = call(proxy.port, "GET", "/lb/status", headers={"X-Vllm-Role": "member"})
    assert body["backends"] == []


def test_an_unknown_role_is_403_on_every_route(lb):
    proxy, engine = lb()
    for method, path in (("GET", "/lb/status"), ("GET", "/v1/models"), ("POST", "/v1/messages"), ("GET", "/")):
        status, body = call(proxy.port, method, path, turn(SERVED) if method == "POST" else None,
                            {"X-Vllm-Role": "unset"})
        assert status == 403 and body["error"]["type"] == "role_unknown", (method, path)
    assert not [r for r in engine.requests if r["method"] == "POST"]


# ── the LB's member check is an ALLOWLIST on the raw path; the view is built from allowed keys ────────
STATIC_FILES = {"/gateway.js", "/gateway.css", "/access.js", "/styles/glass.css"}
is_static = STATIC_FILES.__contains__


@pytest.mark.parametrize("method,path", [
    ("GET", "/"), ("GET", "/index.html"), ("HEAD", "/"), ("GET", "/gateway.js"), ("GET", "/styles/glass.css"),
    ("GET", "/lb/status"), ("GET", "/v1/models"), ("POST", "/v1/messages"), ("POST", "/v1/chat/completions"),
    ("POST", "/v1/messages/count_tokens"), ("POST", "/v1/completions"), ("POST", "/v1/responses"),
    ("POST", "/v1/audio/transcriptions"), ("POST", "/v1/audio/translations"),
])
def test_the_lb_lets_a_member_reach_only_the_listed_routes_and_static_files(method, path):
    assert roles.member_may(method, path, is_static)


@pytest.mark.parametrize("method,path", [
    ("GET", "/benchmarks"), ("GET", "/sessions"), ("GET", "/benchmarks.html"), ("GET", "/sessions.html"),
    ("GET", "/nope.js"), ("GET", "/lb/bench"), ("GET", "/lb/perf"), ("GET", "/lb/hardware/a/recorder"),
    ("POST", "/v1/embeddings"), ("POST", "/v1/models"), ("POST", "/lb/cluster"), ("DELETE", "/lb/status"),
    ("GET", "/lb/%73tatus"), ("GET", "/x/../lb/bench"), ("GET", "//lb/status"), ("GET", "/./gateway.js"),
    ("GET", "/%2e%2e/lb/bench"), ("POST", "/v1//messages"), ("GET", "/lb/status/"),
])
def test_everything_else_is_refused_to_a_member_raw_spellings_included(method, path):
    assert not roles.member_may(method, path, is_static)


#: every key a member's /lb/status may carry -- an ALLOWLIST: a key /lb/status gains tomorrow is the admin's
MEMBER_KEYS = {"now", "uptime_s", "role", "granted_models", "backends", "clusters", "cluster", "hardware",
               "hardware_state", "models", "leases", "declared", "swarm", "queue", "db_ok",
               # the two periods the page draws time with: no deployment detail
               "config"}


def test_a_members_status_carries_only_the_allowed_keys():
    status = {**STATUS, "config": {"x": 1}, "connect": {"ssh": "x"}, "saturation": {"x": 1},
              "throughput_target": 9, "run_next": "x", "last_tick_error": "boom", "vllm_boxes_detail": "box-b",
              "vllm_capacity_detail": "x", "requests_total": 9, "gpu_slots": "x", "context": {}, "db_ok": True}
    view = roles.member_status(status, M, fleet())
    assert set(view) <= MEMBER_KEYS, set(view) - MEMBER_KEYS
    assert set(view["queue"]) <= {"inflight", "inflight_total", "recent", "counts", "limits"}

    # hardware_state names every box's address (hw.status `hosts`): a member reads only whether the listener runs
    view = roles.member_status({**status, "hardware_state": {"state": "listening", "port": 1, "received": 9,
                                                             "hosts": {"10.0.0.9": "box-b"}}}, M, fleet())
    assert view["hardware_state"] == {"state": "listening"}


#: what /lb/status holds that names the fleet's insides: engine addresses, ssh errors, the applier's log
INSIDES = {
    **STATUS,
    "backends": [{"url": "http://10.0.0.7:8000", "host_label": "box-a-gpu", "box": "box-a", "cluster": "mine",
                  "served_models": ["a/m"], "live": True, "history": [1, 2],
                  "last_error": "connect to 10.0.0.7:8000 refused"}],
    "clusters": {"mine": {"serves": None, "state": "running", "url": "http://10.0.0.7:8000",
                          "detail": "cannot observe box-a (ssh: connect to host 10.0.0.8 port 22)",
                          "catalog_error": "/srv/models/x: missing", "store_error": "/srv/state: read-only",
                          "applier": {"host": "box-a", "log_tail": ["/srv/run.sh: 10.0.0.8"], "listening": True},
                          "boxes": {"box-a": {"mem_total_gib": 128, "label": "box-a-gpu",
                                              "error": "status: ssh: connect to host 10.0.0.8 port 22"}}}},
    "models": {"cluster": [{"served_name": "a/m"}], "classic": [], "error": "/srv/models: permission denied"},
}
INSIDES["cluster"] = INSIDES["clusters"]["mine"]


def test_a_members_status_names_no_address_error_text_or_log_of_the_fleets_insides():
    view = roles.member_status(INSIDES, M, fleet(), cluster_name="mine")
    text = json.dumps(view)
    for inside in ("10.0.0.", "://", "/srv/", "box-a-gpu", "ssh"):
        assert inside not in text, inside
    b, = view["backends"]
    assert (b["box"], b["served_models"], b["live"], b["history"]) == ("box-a", ["a/m"], True, [1, 2])
    c = view["clusters"]["mine"]
    assert c["state"] == "running" and c["boxes"]["box-a"]["mem_total_gib"] == 128
    # a failure is still SAID, without its text (the reason is an admin's): never a box that looks healthy
    assert c["boxes"]["box-a"]["error"] == roles.WITHHELD and c["detail"] == roles.WITHHELD
    assert view["models"]["error"] == roles.WITHHELD
    assert view["cluster"]["boxes"]["box-a"]["error"] == roles.WITHHELD
    assert roles.member_status(STATUS, M, fleet())["models"]["error"] is None


#: a server view as /lb/status builds it, its admin's parts included: the catalog (granted or not), settings, checkpoint
#: search paths, ssh errors, the machine's hostname
REAL_VIEW = {
    "name": "mine", "configured": True, "state": "running", "since": 1, "up_since": 1, "task": "chat",
    "hardware": "GPU X", "topology": {"tp": 1, "boxes": ["box-a"]}, "held": False, "freeing": None, "max_depth": 6,
    "tiles": ["generated"], "serves": {"served_name": "a/m"}, "loaded_model": None, "loading": {},
    "desired": {"max_model_len": 4096}, "effective": None, "default": True,
    "url": "http://10.0.0.7:8000", "applier": {"host": "host-a", "log_tail": ["x"]}, "switch": "/lb/clusters/mine",
    "models": [{"id": "a", "served_name": "a/m"}, {"id": "b", "served_name": "b/m"}],
    "setups": {"b": {"x": 1}}, "kv_cache_dtypes": {"b": []}, "offers": ["b@x"], "setup_targets": {"b": {}},
    "recommended": {"x": 1}, "defaults": {"x": 1}, "limits": {"x": [1, 2]}, "clamp": "", "declared": None,
    "rescue": {"b/m": 1}, "engines": {"b": 1}, "detail": "ssh: connect to host 10.0.0.8", "catalog_error": "",
    "boxes": {"box-a": {"mem_total_gib": 128, "rank": 0, "label": "host-a-gpu", "network": {"ip": "10.0.0.8"},
                        "reservation": {"status": "running", "job_id": "job-1", "claimed_by": "host-a-gpu"},
                        "park": "host-a-gpu bench",
                        "hold": {"claimed_by": "host-a-gpu"}, "holds": ["b/m"], "heal": {"x": 1},
                        "availability": {"searched": ["/srv/models/b"], "dir": "/srv/models", "error": "mount: /srv"},
                        "error": "status: ssh: connect to host 10.0.0.8 port 22"}},
}


def test_a_members_server_view_is_built_from_an_allowlist_and_their_catalog_rows_only():
    status = {**STATUS, "clusters": {"mine": REAL_VIEW}, "cluster": REAL_VIEW,
              "hardware": {"box-a": {"host": "host-a", "cpu_percent": 1, "memory": "unified"}}}
    view = roles.member_status(status, M, fleet(), cluster_name="mine")
    text = json.dumps(view)
    for inside in ("10.0.0.", "://", "/srv/", "host-a", "ssh", "b/m", '"b"', "/lb/clusters"):
        assert inside not in text, inside
    c = view["clusters"]["mine"]
    assert set(c) <= roles.MEMBER_SERVER_KEYS, set(c) - roles.MEMBER_SERVER_KEYS
    assert c["models"] == [{"id": "a", "served_name": "a/m"}] and c["desired"] == {"max_model_len": 4096}
    box = c["boxes"]["box-a"]
    assert set(box) <= roles.MEMBER_BOX_KEYS and box["mem_total_gib"] == 128 and box["rank"] == 0
    assert box["error"] == roles.WITHHELD and box["availability"] == roles.WITHHELD
    # a reservation is its status only; a park says the box is held, not by whom
    assert box["reservation"] == {"status": "running"} and box["park"] == roles.PARKED
    bare = roles.member_status({**status, "clusters": {"mine": {**REAL_VIEW, "boxes": {"box-a": {"park": None,
                                "reservation": None}}}}}, M, fleet())
    assert bare["clusters"]["mine"]["boxes"]["box-a"] == {"park": None, "reservation": None}
    assert view["cluster"] == c
    assert view["hardware"] == {"box-a": {"cpu_percent": 1, "memory": "unified"}}


def test_a_member_gets_a_static_file_and_not_the_benchmarks_page_through_the_proxy(lb):
    proxy, _ = lb()
    assert call(proxy.port, "GET", "/access.js", headers=MEMBER)[0] == 200
    status, body = call(proxy.port, "GET", "/benchmarks", headers=MEMBER)
    assert status == 403 and body["error"]["type"] == "admin_only"
    status, body = call(proxy.port, "GET", "/lb/%73tatus", headers=MEMBER)
    assert status == 403 and body["error"]["type"] == "admin_only"
