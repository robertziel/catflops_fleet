"""PARALLEL CLUSTERS — what the independent audit found (AUDIT-FINAL, 2026-10-17), each pinned here, most ported from
its probes (worklog/parallel-clusters.md, "Audit fixes"):

  P1-1  two Starts on shared boxes with no tick between them: the second is refused naming the holder — a cluster
        HOLDS its boxes while its stored wish wants it up, not only once its view says so; check and write are one step;
  P1-2  holders come from EVERY declared cluster's persisted wish, in this run or not (an LB restart forgets a "+"
        activation, never the engine it started); two boot clusters sharing a box are refused at load;
  P1-3  ONE DEFAULT CLUSTER: with several declared, exactly one says `default: true`; a request with no model or a
        loose name goes to the default alone, under its served name — never moved to another cluster;
  P2    a ROCm node renders its runtime; the perf log never mixes two heads; no "cluster" default name; a declared
        cluster that cannot launch is refused at Activate, not at boot.
"""
from __future__ import annotations

import json
import os
from dataclasses import replace

import pytest

from tests.fleet_files import PARALLEL, write
from tests.test_parallel_clusters import AMD_NODE, SHARED, _posts, api, two_live  # noqa: F401 (fixtures)
from vllm_lb.config import Config
from vllm_lb.fleet import parse_fleet
from vllm_lb.state import CLUSTER, Backend, Snapshot


def _cfg(tmp_path, text=SHARED, **over) -> Config:
    return replace(Config.from_env({"LB_ALLOWED_HOSTS": "loopback", "LB_FLEET": write(tmp_path, text), "LB_CLUSTER_STATE": str(tmp_path / "c.json")}),
                   listen_host="127.0.0.1", listen_port=0, **over)


# ── P1-1: one atomic answer to "may this cluster start" ────────────────────────────────────────────────────────────

def test_two_starts_on_shared_boxes_with_no_tick_between_the_second_is_refused_naming_the_holder(tmp_path, api):
    hub, call = api(_cfg(tmp_path))
    hub.set_cluster({**hub.cluster_views["pair"], "state": "off", "desired": {"mode": "classic"}}, "pair")
    assert call("POST", "/lb/clusters/alt/activate")[0] == 200
    hub.set_cluster({**hub.cluster_views["alt"], "state": "off", "desired": {"mode": "classic"}}, "alt")
    s1, b1 = call("POST", "/lb/clusters/pair", {"enabled": True})
    assert s1 in (200, 202), b1
    s2, b2 = call("POST", "/lb/clusters/alt", {"enabled": True})
    assert s2 == 409 and b2["error"]["type"] == "cluster_blocked" and "pair" in b2["error"]["message"], b2


# ── P1-2: every declared cluster's persisted wish ──────────────────────────────────────────────────────────────────

def _run(cfg):
    from vllm_lb.cluster_run import ClusterRun
    from vllm_lb.hub import Hub
    hub = Hub(cfg)
    run = ClusterRun(cfg, hub, None)
    run.start(None)
    return hub, run


def test_an_inactive_clusters_stored_start_still_holds_its_boxes_after_a_restart(tmp_path):
    """alt was activated and Started in the run before; the LB restarted and forgot the activation, not the engine."""
    from vllm_lb.cluster_defaults import desired_store
    cfg = _cfg(tmp_path)
    alt = cfg.for_cluster("alt")
    os.makedirs(os.path.dirname(alt.cluster_state_path), exist_ok=True)
    desired_store(alt).set(enabled=True)
    hub, run = _run(cfg)
    hub.set_cluster({**hub.cluster_views["pair"], "state": "off", "desired": {"mode": "classic"}}, "pair")
    assert "alt" not in hub.cluster_cfgs
    assert [(b["box"], b["cluster"]) for b in run.blockers("pair")] == [("node-1", "alt"), ("node-2", "alt")]
    assert hub.snapshot().cluster_wanted() is True, "the classic planner stands aside for alt's engine too"
    alt_row = {d["name"]: d for d in run.declared()}["alt"]
    assert alt_row["holding"] is True and alt_row["active"] is False


LOCAL_ALT = PARALLEL.replace("    run_at_boot: true\n    model:", "    run_at_boot: false\n    model:").replace(
    "backends:", """  gui-alt:
    control: local
    run_at_boot: false
    model: qwen3.8-27b-nvidia-nvfp4
    url: http://198.51.100.21:8000
    launch:
      image: vllm/vllm-openai@sha256:""" + "1" * 64 + """
      max_model_len: 131072
      max_num_seqs: 8
      gpu_memory_utilization: 0.9
      reasoning_parser: qwen3
      tool_call_parser: qwen3_coder
      choices: [qwen3.8-27b-nvidia-nvfp4]
    ranks:
      - {box: solo}
backends:""")


def test_a_local_nodes_stored_up_intent_holds_its_box_in_this_run_or_not(tmp_path):
    from vllm_lb import node_control as nc
    cfg = _cfg(tmp_path, LOCAL_ALT)
    nc.IntentStore(cfg.for_cluster("gui").cluster_state_path).write(nc.UP, {}, now=1.0)
    _, run = _run(cfg)
    assert [(b["box"], b["cluster"]) for b in run.blockers("gui-alt")] == [("solo", "gui")]
    nc.IntentStore(cfg.for_cluster("gui").cluster_state_path).write(nc.DOWN, {}, now=2.0)
    assert run.blockers("gui-alt") == []


def test_two_boot_clusters_sharing_a_box_are_refused_at_load_naming_both():
    both = SHARED.replace("  alt:\n    control: ssh\n    iface: bond0\n    link: roce\n    run_at_boot: false",
                          "  alt:\n    control: ssh\n    iface: bond0\n    link: roce\n    run_at_boot: true")
    assert both != SHARED, "the plant took"
    with pytest.raises(ValueError) as refused:
        parse_fleet(both, "fleet.yml")
    assert "pair" in str(refused.value) and "alt" in str(refused.value) and "node-1" in str(refused.value)


# ── P1-3: one default cluster ──────────────────────────────────────────────────────────────────────────────────────

def test_several_clusters_name_exactly_one_default():
    none = PARALLEL.replace("    default: true\n", "")
    with pytest.raises(ValueError, match="default") as refused:
        parse_fleet(none, "fleet.yml")
    assert "pair" in str(refused.value) and "gui" in str(refused.value)
    two = PARALLEL.replace("    control: local\n", "    control: local\n    default: true\n")
    with pytest.raises(ValueError, match="pair, gui"):
        parse_fleet(two, "fleet.yml")
    f = parse_fleet(PARALLEL, "fleet.yml")
    assert [c.name for c in f.clusters.values() if c.default] == ["pair"]


def test_one_cluster_needs_no_flag_and_is_the_default(tmp_path):
    from tests.fleet_files import PAIR
    assert parse_fleet(PAIR, "fleet.yml").clusters["pair"].default is True
    assert Config.from_env({"LB_ALLOWED_HOSTS": "loopback", "LB_FLEET": write(tmp_path, PAIR)}).default_cluster == "pair"
    assert Config.from_env({"LB_ALLOWED_HOSTS": "loopback", "LB_FLEET": write(tmp_path, PARALLEL, "p.yml")}).default_cluster == "pair"


def test_no_model_or_a_loose_name_goes_to_the_default_cluster_alone(two_live):
    """Ported from the audit's probe: the default (pair, serving A) is off, gui (serving B) is up — a request that
    does not name B never reaches gui: it is refused 503 naming what is live, never rewritten onto B."""
    import http.client
    proxy, hub, a, b, post = two_live
    hub.update_backend(a.url, live=False)
    hub.set_cluster({**hub.cluster_views["pair"], "held": False, "state": "off", "desired": {"mode": "classic"}}, "pair")
    status, body = post("qwen3.8:latest")
    assert status == 503 and "B" in body["error"]["message"] and "gui" in body["error"]["message"], body
    conn = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=20)
    conn.request("POST", "/v1/messages", body=json.dumps({"max_tokens": 1}).encode(),
                 headers={"Content-Type": "application/json"})
    r = conn.getresponse()
    assert r.status == 503, r.read()
    assert _posts(b) == [], "a request not naming B reached the B cluster"
    assert post("B")[0] == 200 and [p["model"] for p in _posts(b)] == ["B"], "an exact name still goes to its own"


def test_with_the_default_up_a_loose_name_is_served_by_it_under_its_served_name(two_live):
    _, _, a, b, post = two_live
    assert post("qwen3.8:latest")[0] == 200
    assert [p["model"] for p in _posts(a)] == ["A"] and _posts(b) == []


def test_status_and_declared_say_which_cluster_is_the_default(tmp_path, api):
    _, call = api(_cfg(tmp_path, PARALLEL))
    _, body = call("GET", "/lb/status")
    assert body["clusters"]["pair"]["default"] is True and body["clusters"]["gui"]["default"] is False
    assert {d["name"]: d["default"] for d in body["declared"]} == {"pair": True, "gui": False}


# ── P2 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────

def test_a_rocm_node_renders_its_runtime_from_its_cards_type():
    """The data → env path for rocm (the audit's mutation hard-coding nvidia survived): the measured AMD node
    (tests/test_gui_node.py's fleet, the shipped type's measured figures) renders EDGE_NODE_GPU_RUNTIME=rocm."""
    from tests.test_gui_node import GUI
    from vllm_lb.fleet import node_env
    lines = node_env(parse_fleet(GUI, "fleet.yml"), "gui")
    assert "EDGE_NODE_GPU_RUNTIME=rocm" in lines, lines


def test_a_declared_cluster_that_cannot_launch_blocks_nothing_but_itself(tmp_path, api):
    """The AMD node, declared but not run at boot: the LB boots its other clusters; Activate is refused naming the
    missing bound (the box states no gpu_ram_max_gib), and the + tab says why. In the boot set it is still refused at load."""
    text = AMD_NODE.replace("    run_at_boot: true\n    model:", "    run_at_boot: false\n    model:")
    f = parse_fleet(text, "fleet.yml")
    assert "gpu_ram_max_gib" in f.clusters["gui"].unlaunchable and f.clusters["pair"].unlaunchable == ""
    with pytest.raises(ValueError, match="gpu_ram_max_gib"):
        parse_fleet(AMD_NODE, "fleet.yml")
    _, call = api(_cfg(tmp_path, text))
    status, body = call("POST", "/lb/clusters/gui/activate")
    assert status == 409 and body["error"]["type"] == "cluster_unlaunchable" and "gpu_ram_max_gib" in body["error"]["message"]
    gui = {d["name"]: d for d in call("GET", "/lb/status")[1]["declared"]}["gui"]
    assert gui["available"] is False and "gpu_ram_max_gib" in gui["unlaunchable"]


def test_the_perf_log_is_shown_every_head_with_its_own_view(tmp_path):
    """Each server's head is fed with ITS view (the recorder keeps them apart by name — test_per_server_figures);
    with one cluster, its head with its own view, as before."""
    from vllm_lb.control import Controller
    from vllm_lb.hub import Hub
    from vllm_lb.state import CLUSTER

    class Rec:
        def __init__(self):
            self.seen = []

        def observe(self, **kw):
            self.seen.append(kw["cluster"].get("name"))

    for text, want in ((PARALLEL, ["pair", "gui"]), (PARALLEL.replace("    run_at_boot: true\n    model:",
                                                                     "    run_at_boot: false\n    model:"), ["pair"])):
        hub = Hub(_cfg(tmp_path, text))
        rec = Rec()
        ctl = Controller.__new__(Controller)
        ctl.hub, ctl.recorder = hub, rec
        for b in hub.backends():
            if b.kind == CLUSTER:
                ctl._record(b.url, {"live": True, "scrape": {}}, 1.0)
        assert sorted(rec.seen) == sorted(want), (text[:0], rec.seen)


def test_a_cluster_config_names_its_cluster_or_is_refused():
    from vllm_lb.cluster import parse_hosts
    hosts = parse_hosts("a-gpu=a=operator@192.0.2.10=198.51.100.1,b-gpu=b=operator@192.0.2.11=198.51.100.2")
    with pytest.raises(ValueError, match="cluster_name"):
        Config(cluster_hosts=hosts, cluster_url="http://192.0.2.10:8100")
    assert Config(cluster_name="pair", cluster_hosts=hosts, cluster_url="http://192.0.2.10:8100").clusters[0].name == "pair"


# ── AUDIT-FINAL-2 (re-audit, APPROVED with three P2s) ──────────────────────────────────────────────────────────────

def test_an_unreadable_switch_of_a_cluster_outside_the_run_holds_its_boxes(tmp_path):
    """N1: `DesiredStore` reads a corrupt switch as its default (`classic`) and says so in `last_error`; the run must not
    call the boxes free on a file nobody could read."""
    cfg = _cfg(tmp_path)
    alt = cfg.for_cluster("alt")
    os.makedirs(os.path.dirname(alt.cluster_state_path), exist_ok=True)
    with open(alt.cluster_state_path, "w") as fh:
        fh.write("{not json")
    hub, run = _run(cfg)
    hub.set_cluster({**hub.cluster_views["pair"], "state": "off", "desired": {"mode": "classic"}}, "pair")
    assert run.wants_up("alt") is True
    assert [(b["box"], b["cluster"]) for b in run.blockers("pair")] == [("node-1", "alt"), ("node-2", "alt")]


def test_two_starts_racing_on_shared_boxes_give_exactly_one_start(tmp_path, api):
    """N2: the check and the write are ONE step under the run's lock — two threads released together by a barrier, 15
    times over: always one 2xx and one 409 `cluster_blocked` (red with the lock taken out: worklog)."""
    import threading
    for i in range(15):
        (tmp_path / str(i)).mkdir()
        hub, call = api(_cfg(tmp_path / str(i)))
        hub.set_cluster({**hub.cluster_views["pair"], "state": "off", "desired": {"mode": "classic"}}, "pair")
        assert call("POST", "/lb/clusters/alt/activate")[0] == 200
        hub.set_cluster({**hub.cluster_views["alt"], "state": "off", "desired": {"mode": "classic"}}, "alt")
        gate, out = threading.Barrier(2), {}

        def start(name):
            gate.wait()
            out[name] = call("POST", f"/lb/clusters/{name}", {"enabled": True})
        threads = [threading.Thread(target=start, args=(n,)) for n in ("pair", "alt")]
        for t in threads:
            t.start()
        for t in threads:
            t.join(20)
        codes = sorted(status for status, _ in out.values())
        assert len(codes) == 2 and codes[0] in (200, 202) and codes[1] == 409, (i, out)
        assert [b for s, b in out.values() if s == 409][0]["error"]["type"] == "cluster_blocked"


def test_the_503_for_an_unnamed_turn_names_the_default_clusters_state_and_what_is_live(two_live):
    """N3: never "no backend is live" while one is: the refusal says the default cluster's state and what IS live."""
    proxy, hub, a, b, post = two_live
    hub.update_backend(a.url, live=False)
    hub.set_cluster({**hub.cluster_views["pair"], "held": False, "state": "off", "detail": "stopped",
                     "desired": {"mode": "classic"}}, "pair")
    status, body = post("qwen3.8:latest")
    message = body["error"]["message"]
    assert status == 503 and "no vLLM or ollama backend is live" not in message, message
    assert "default cluster pair is off" in message and "B (cluster gui)" in message, message


# ── the default cluster owns what it can launch (2026-10-17) ──────────────────────────────────────────────────────
# A client names `Qwen/Qwen3.8-Flash-Next-SGLang` — a profile of the pair's own catalog — while the pair serves the 27B.
# Before parallel clusters the one cluster took it (rewritten to what it serves); R2's 503 broke that client. A model the
# DEFAULT cluster can launch is the default's, as a loose name is; only a model nothing but ANOTHER cluster runs is
# refused unserved — the 2B GUI agent's turns never land on the pair and the pair's never on it.

def test_one_cluster_takes_a_catalog_profile_it_is_not_serving_right_now(tmp_path):
    """The live regression: a one-cluster fleet behaves as before — a client's Flash-Next name reaches the pair."""
    from tests.fleet_files import PAIR
    from vllm_lb.router import eligible
    cfg = Config.from_env({"LB_ALLOWED_HOSTS": "loopback", "LB_FLEET": write(tmp_path, PAIR), "LB_CLUSTER_STATE": str(tmp_path / "c.json")})
    pair = Backend(kind=CLUSTER, url=cfg.cluster_url, live=True, served_models=("Qwen/Qwen3.8-27B-SGLang",),
                   depth_known=True, cluster=cfg.default_cluster)
    snap = Snapshot(now=100.0, backends=[pair], leases=[], last_request_at=None, last_pressure_at=None, db_ok=True,
                    extra={"clusters": {cfg.default_cluster: {"configured": True, "held": True, "max_depth": 8}}})
    verdict = eligible(snap, cfg, model="Qwen/Qwen3.8-Flash-Next-SGLang")
    assert verdict.why == "" and [b.url for b in verdict.backends] == [cfg.cluster_url], verdict


def test_a_profile_the_default_can_launch_goes_to_the_default_never_the_other_cluster(two_live):
    _, _, a, b, post = two_live
    status, _ = post("qwen3.8-27b-nvidia-nvfp4")
    assert status == 200 and [p["model"] for p in _posts(a)] == ["A"] and _posts(b) == []


def test_a_model_only_another_cluster_runs_is_refused_while_it_is_not_served(tmp_path):
    """The guard R2 was for: the node's own model, unserved, is a 503 — never rewritten onto the default pair."""
    from vllm_lb.router import eligible

    class PairLaunchesNothing:          # a catalog whose cluster profiles exclude the node's row
        def for_mode(self, mode):
            return ()
    cfg = Config.from_env({"LB_ALLOWED_HOSTS": "loopback", "LB_FLEET": write(tmp_path, PARALLEL), "LB_CLUSTER_STATE": str(tmp_path / "c.json")})
    pair = Backend(kind=CLUSTER, url="http://pair:8100", live=True, served_models=("A",), depth_known=True,
                   cluster="pair")
    held = {"configured": True, "held": True, "max_depth": 8}
    snap = Snapshot(now=100.0, backends=[pair], leases=[], last_request_at=None, last_pressure_at=None, db_ok=True,
                    extra={"clusters": {"pair": held, "node": {**held, "held": False}}})
    verdict = eligible(snap, cfg, model="qwen3.8-27b-nvidia-nvfp4", catalog=PairLaunchesNothing())
    assert verdict.backends == [] and "qwen3.8-27b-nvidia-nvfp4" in verdict.why, verdict
