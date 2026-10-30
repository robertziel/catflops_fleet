"""PARALLEL CLUSTERS — N independent engines under one LB, each keyed by its fleet NAME (worklog/parallel-clusters.md).

The pins: a fleet naming the ssh pair AND a local node parses and `Config` exposes both; every per-cluster thing
(state dir, intent, loop, hub view, backend, route) is keyed by the cluster's name; the singular `cluster_*` fields are
the ONE-cluster derivation and empty for several; routing narrows to the backends serving the requested model.

Tests marked FINISHER are RED on purpose: they pin what the next phase builds (the worklog lists them by name).
"""
from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from tests.fleet_files import LOCAL_ON_PAIR_BOX, ONE_HOST, PAIR, PARALLEL, write
from vllm_lb.cluster import ClusterHost, parse_hosts
from vllm_lb.config import ClusterConfig, Config, topology_of
from vllm_lb.fleet import load_fleet, parse_fleet
from vllm_lb.hub import Hub
from vllm_lb.router import choose
from vllm_lb.state import CLUSTER, Backend, Snapshot
from tests.publishable_words import offenders

LB = Path(__file__).resolve().parents[1]
HOSTS = "a-gpu=a=operator@192.0.2.10=198.51.100.1,b-gpu=b=operator@192.0.2.11=198.51.100.2"


# ── the fleet and the config ─────────────────────────────────────────────────────────────────────

def test_a_fleet_with_the_pair_and_a_local_node_parses_and_names_both():
    f = parse_fleet(PARALLEL, "fleet.yml")
    assert list(f.clusters) == ["pair", "gui"]
    assert (f.clusters["pair"].control, f.clusters["pair"].tp) == ("ssh", 2)
    assert (f.clusters["gui"].control, f.clusters["gui"].tp, f.clusters["gui"].ranks[0].box) == ("local", 1, "solo")


def test_config_exposes_every_cluster_by_name_and_no_singular_one(tmp_path):
    cfg = Config.from_env({"LB_ALLOWED_HOSTS": "loopback", "LB_FLEET": write(tmp_path, PARALLEL)})
    assert [c.name for c in cfg.clusters] == ["pair", "gui"]
    pair, gui = cfg.clusters
    assert (pair.control, pair.url, [h.box for h in pair.hosts]) == ("ssh", "http://192.0.2.10:8100", ["node-1", "node-2"])
    assert (gui.control, gui.url, gui.hosts) == ("local", "http://198.51.100.20:8000",
                                                 (ClusterHost(label="solo-gpu", box="solo"),))
    # Two clusters: nothing is "the cluster". The singular fields are EMPTY, never the first one.
    assert (cfg.cluster_hosts, cfg.cluster_url, cfg.cluster_name, cfg.cluster_configured) == ((), "", "", False)


def test_for_cluster_is_a_one_cluster_config_with_its_own_state_dir(tmp_path):
    cfg = Config.from_env({"LB_ALLOWED_HOSTS": "loopback", "LB_FLEET": write(tmp_path, PARALLEL), "LB_CLUSTER_STATE": str(tmp_path / "s" / "cluster.json")})
    gui = cfg.for_cluster("gui")
    assert gui.clusters == (cfg.clusters[1],)
    assert (gui.cluster_name, gui.cluster_control, gui.cluster_url, gui.cluster_configured) == \
        ("gui", "local", "http://198.51.100.20:8000", True)
    assert gui.cluster_state_path == str(tmp_path / "s" / "clusters" / "gui" / "cluster.json")
    assert cfg.for_cluster("pair").cluster_state_path == str(tmp_path / "s" / "clusters" / "pair" / "cluster.json")
    assert gui.fleet is cfg.fleet, "the same fleet: the per-cluster view narrows nothing but the cluster"
    with pytest.raises(ValueError, match="gui, pair|pair, gui"):
        cfg.for_cluster("nope")


def test_one_cluster_keeps_its_state_where_it_always_was(tmp_path):
    """Live continuity: a one-cluster deployment's switch stays at LB_CLUSTER_STATE — the file a live deployment has today."""
    cfg = Config.from_env({"LB_ALLOWED_HOSTS": "loopback", "LB_FLEET": write(tmp_path, PAIR), "LB_CLUSTER_STATE": str(tmp_path / "cluster.json")})
    assert [c.name for c in cfg.clusters] == ["pair"] and cfg.cluster_name == "pair"
    assert cfg.cluster_hosts == parse_hosts("node-1-gpu=node-1=operator@192.0.2.10=198.51.100.1,"
                                            "node-2-gpu=node-2=operator@192.0.2.11=198.51.100.2")
    assert cfg.for_cluster("pair") is cfg
    assert cfg.for_cluster("pair").cluster_state_path == str(tmp_path / "cluster.json")


def test_a_config_built_from_the_singular_fields_names_one_cluster():
    """Every test that builds `Config(cluster_hosts=…)` is a one-cluster LB: `clusters` is DERIVED, one definition."""
    cfg = Config(cluster_name="pair", cluster_hosts=parse_hosts(HOSTS), cluster_url="http://127.0.0.1:8100")
    assert cfg.clusters == (ClusterConfig(name=cfg.cluster_name, hosts=parse_hosts(HOSTS), url="http://127.0.0.1:8100",
                                          control="ssh", default=True),), "the one cluster is the default"
    assert cfg.cluster_name and cfg.for_cluster(cfg.cluster_name) is cfg
    assert Config().clusters == () and Config().cluster_name == ""


def test_a_legacy_state_file_beside_a_missing_per_cluster_one_refuses_naming_the_move(tmp_path):
    """The one dangerous transition: 1 → 2 clusters. The pair's switch at the legacy path, unread, would default to
    `classic` and STOP a running pair. Refused at boot, naming the mv — never a silent re-default."""
    root = tmp_path / "state" / "cluster.json"
    root.parent.mkdir()
    root.write_text("{}")
    cfg = Config.from_env({"LB_ALLOWED_HOSTS": "loopback", "LB_FLEET": write(tmp_path, PARALLEL), "LB_CLUSTER_STATE": str(root)})
    with pytest.raises(ValueError) as e:
        cfg.for_cluster("pair")
    assert str(root) in str(e.value) and str(root.parent / "clusters" / "pair" / "cluster.json") in str(e.value)
    (root.parent / "clusters" / "pair").mkdir(parents=True)
    (root.parent / "clusters" / "pair" / "cluster.json").write_text("{}")
    assert cfg.for_cluster("pair").cluster_name == "pair", "both present: the per-cluster file is read"


def test_topology_of_hands_back_the_clusters_not_a_singular():
    fields = topology_of(parse_fleet(PARALLEL, "fleet.yml"))
    assert [c.name for c in fields["clusters"]] == ["pair", "gui"]
    assert "cluster_hosts" not in fields and "cluster_url" not in fields


def test_declared_clusters_may_share_a_box_they_are_alternatives_not_all_run_at_once():
    """The fleet file is a CATALOG (operator, 2026-10-17): two declared clusters on the same boxes are legal — which
    runs is the run's (cluster_run.py). A box holding two ranks of ONE cluster is still refused."""
    two = PAIR.replace("    control: ssh\n", "    control: ssh\n    run_at_boot: true\n    default: true\n").replace("backends:", """  again:
    control: ssh
    iface: bond0
    link: roce
    run_at_boot: false
    url: http://192.0.2.12:8100
    ram_usable_fraction: 0.98
    max_num_batched_tokens: 8192
    ranks:
      - {box: node-1, interconnect: 198.51.100.1}
      - {box: node-2, interconnect: 198.51.100.2}
backends:""")
    f = parse_fleet(two, "fleet.yml")
    assert [(c.name, c.run_at_boot) for c in f.clusters.values()] == [("pair", True), ("again", False)]
    with pytest.raises(ValueError, match="two ranks of one cluster"):
        parse_fleet(PAIR.replace("{box: node-2, interconnect: 198.51.100.2}", "{box: node-1, interconnect: 198.51.100.2}"),
                    "fleet.yml")


def test_several_declared_clusters_each_say_whether_they_run_at_boot():
    unsaid = PARALLEL.replace("    run_at_boot: true\n", "", 1)
    with pytest.raises(ValueError, match="run_at_boot"):
        parse_fleet(unsaid, "fleet.yml")
    assert parse_fleet(PAIR, "fleet.yml").clusters["pair"].run_at_boot is True, "the one cluster of a fleet runs at boot"


def test_the_committed_parallel_example_is_a_valid_fleet_of_two_clusters():
    f = load_fleet(LB / "fleet.parallel.example.yml")
    assert len(f.clusters) == 2 and {c.control for c in f.clusters.values()} == {"ssh", "local"}
    assert [c.run_at_boot for c in f.clusters.values()] == [True, False], "the pair boots; the node is the + tab's"
    text = (LB / "fleet.parallel.example.yml").read_text()
    assert offenders(text) == [], "the committed example names a deployment (tests/publishable_words.py)"


# ── the state-path rule, shared with the host applier ───────────────────────────────────────────

def test_the_state_path_rule_is_one_function_the_lb_and_the_applier_share():
    from vllm_lb.node_control import node_dir, state_path_for
    assert state_path_for("/data/cluster.json", "pair", ["pair"]) == "/data/cluster.json"
    assert state_path_for("/data/cluster.json", "gui", ["pair", "gui"]) == "/data/clusters/gui/cluster.json"
    assert node_dir(state_path_for("/data/cluster.json", "gui", ["pair", "gui"])) == "/data/clusters/gui/node"
    with pytest.raises(ValueError, match="gui"):
        state_path_for("/data/cluster.json", "gui", ["pair"])


def test_the_applier_names_the_cluster_it_applies_when_the_fleet_offers_several(tmp_path):
    from vllm_lb.catalog import document
    from vllm_lb.node_applier import cluster_of
    (tmp_path / "lb").mkdir()
    (tmp_path / "lb" / "fleet.yml").write_text(PARALLEL)
    catalog = document()
    fleet, name = cluster_of(tmp_path, {"EDGE_MODELS_DIR": str(tmp_path)}, "gui", catalog=catalog)
    assert name == "gui" and list(fleet.clusters) == ["pair", "gui"]
    with pytest.raises(ValueError, match="pair"):           # the pair is not a node this host applies
        cluster_of(tmp_path, {"EDGE_MODELS_DIR": str(tmp_path)}, "pair", catalog=catalog)
    # One local node in the fleet: it is the one — a unique answer, not a guess.
    (tmp_path / "lb" / "fleet.yml").write_text(ONE_HOST.replace("control: none", "control: local").replace(
        "    ranks:", f"""    launch:
      image: vllm/vllm-openai@sha256:{'1' * 64}
      max_model_len: 131072
      max_num_seqs: 8
      gpu_memory_utilization: 0.9
      reasoning_parser: qwen3
      tool_call_parser: qwen3_coder
      choices: [qwen3.8-27b-nvidia-nvfp4@uncensored]
    ranks:"""))
    _, name = cluster_of(tmp_path, {"EDGE_MODELS_DIR": str(tmp_path)}, None, catalog=catalog)
    assert name == "node"


# ── the hub: one view, one backend per cluster ───────────────────────────────────────────────────

def _two(tmp_path) -> Config:
    return Config.from_env({"LB_ALLOWED_HOSTS": "loopback", "LB_FLEET": write(tmp_path, PARALLEL), "LB_CLUSTER_STATE": str(tmp_path / "cluster.json")})


def test_the_hub_dials_one_cluster_backend_per_cluster_named_for_it(tmp_path):
    hub = Hub(_two(tmp_path))
    clusters = {b.cluster: b.url for b in hub.backends() if b.kind == CLUSTER}
    assert clusters == {"pair": "http://192.0.2.10:8100", "gui": "http://198.51.100.20:8000"}


def test_the_hub_publishes_every_clusters_view_by_name_and_no_singular_one(tmp_path):
    hub = Hub(_two(tmp_path))
    extra = hub.snapshot().extra
    assert set(extra["clusters"]) == {"pair", "gui"}
    assert extra["clusters"]["pair"]["topology"]["control"] == "ssh"
    assert extra["clusters"]["gui"]["topology"]["control"] == "local"
    assert extra["clusters"]["gui"]["name"] == "gui"
    assert extra["cluster"]["configured"] is False and extra["cluster"]["clusters"] == ["pair", "gui"]
    hub.set_cluster({"configured": True, "state": "running", "held": True}, "gui")
    after = hub.snapshot().extra["clusters"]
    assert after["gui"]["state"] == "running" and after["pair"]["state"] == "unknown"
    with pytest.raises(ValueError, match="pair, gui|gui, pair"):
        hub.set_cluster({"configured": True}, "nope")
    with pytest.raises(ValueError):
        hub.set_cluster({"configured": True})                  # two clusters: name one


def test_one_cluster_is_published_under_its_name_and_as_the_cluster(tmp_path):
    cfg = Config.from_env({"LB_ALLOWED_HOSTS": "loopback", "LB_FLEET": write(tmp_path, PAIR), "LB_CLUSTER_STATE": str(tmp_path / "cluster.json")})
    hub = Hub(cfg)
    hub.set_cluster({"configured": True, "state": "running", "held": True})
    extra = hub.snapshot().extra
    assert extra["cluster"]["state"] == "running"
    assert extra["clusters"] == {"pair": {**extra["cluster"], "name": "pair", "switch": "/lb/clusters/pair",
                                          "default": True}}
    assert [b.cluster for b in hub.backends() if b.kind == CLUSTER] == ["pair"]


# ── the run: which declared clusters run NOW (cluster_run.py) ───────────────────────────────────

SHARED = PARALLEL.replace("    run_at_boot: true\n    model:", "    run_at_boot: false\n    model:").replace(
    "backends:", """  alt:
    control: ssh
    iface: bond0
    link: roce
    run_at_boot: false
    url: http://192.0.2.12:8100
    ram_usable_fraction: 0.98
    max_num_batched_tokens: 8192
    ranks:
      - {box: node-1, interconnect: 198.51.100.1}
      - {box: node-2, interconnect: 198.51.100.2}
backends:""")


def _run(tmp_path, text=SHARED):
    from vllm_lb.cluster_run import ClusterRun
    cfg = Config.from_env({"LB_ALLOWED_HOSTS": "loopback", "LB_FLEET": write(tmp_path, text), "LB_CLUSTER_STATE": str(tmp_path / "cluster.json")})
    hub = Hub(cfg)
    run = ClusterRun(cfg, hub, None)
    run.start(None)
    return cfg, hub, run


def test_at_boot_only_the_clusters_marked_run_at_boot_are_in_the_run(tmp_path):
    cfg, hub, run = _run(tmp_path)
    assert list(hub.cluster_cfgs) == ["pair"] and list(run.loops.loops) == ["pair"]
    assert [b.cluster for b in hub.backends() if b.kind == CLUSTER] == ["pair"]
    assert set(hub.snapshot().extra["clusters"]) == {"pair"}
    declared = {d["name"]: d for d in run.declared()}
    assert declared["pair"]["active"] is True and declared["pair"]["boot"] is True
    assert declared["pair"]["tp"] == 2, "the fleet's derived TP — the + tab prints it, never a box count"
    assert declared["gui"] == {"name": "gui", "control": "local", "boxes": ["solo"], "tp": 1, "boot": False, "active": False,
                               "default": False, "model": "qwen3.8-27b-nvidia-nvfp4", "blocked_by": [], "available": True,
                               "holding": False, "unlaunchable": "", "alerts": []}
    # The pair is not observed yet (`unknown`): its boxes count as HELD — a second engine never lands on an unread box.
    assert declared["alt"]["available"] is False
    assert [(b["box"], b["cluster"]) for b in declared["alt"]["blocked_by"]] == [("node-1", "pair"), ("node-2", "pair")]


def test_a_boot_of_todays_one_cluster_fleet_is_unchanged(tmp_path):
    cfg, hub, run = _run(tmp_path, PAIR)
    assert list(hub.cluster_cfgs) == ["pair"] and cfg.for_cluster("pair") is cfg
    assert hub.snapshot().extra["cluster"]["topology"]["control"] == "ssh", "the one cluster is still `status.cluster`"
    assert run.declared()[0]["active"] is True


def test_activation_adds_a_declared_cluster_to_this_run_with_its_view_backend_and_loop(tmp_path):
    cfg, hub, run = _run(tmp_path)
    assert run.activate("gui") == {"active": ["pair", "gui"]}
    assert "gui" in hub.cluster_cfgs and "gui" in run.loops.loops
    assert {b.cluster for b in hub.backends() if b.kind == CLUSTER} == {"pair", "gui"}
    views = hub.snapshot().extra["clusters"]
    assert views["gui"]["switch"] == "/lb/clusters/gui" and views["gui"]["topology"]["control"] == "local"
    assert next(d for d in run.declared() if d["name"] == "gui")["active"] is True


def test_a_cluster_whose_box_another_holds_is_refused_naming_box_and_holder(tmp_path):
    from vllm_lb.cluster_run import RunRefused
    cfg, hub, run = _run(tmp_path)
    with pytest.raises(RunRefused) as e:
        run.activate("alt")
    assert e.value.status == 409 and e.value.kind == "cluster_blocked"
    assert "node-1" in str(e.value) and "pair" in str(e.value)
    # The pair OFF frees its boxes: the alternative may run.
    hub.set_cluster({**hub.cluster_views["pair"], "state": "off", "desired": {"mode": "classic"}}, "pair")
    assert run.blockers("alt") == []
    assert run.activate("alt")["active"] == ["pair", "alt"]
    # … and the pair's Start is now what is blocked, by the same rule (the proxy refuses it naming alt once alt runs).
    hub.set_cluster({**hub.cluster_views["alt"], "state": "starting"}, "alt")
    assert [(b["box"], b["cluster"]) for b in run.blockers("pair")] == [("node-1", "alt"), ("node-2", "alt")]


def test_a_pair_starts_beside_a_local_node_its_gate_measures_but_not_the_other_way(tmp_path):
    """tp-rank's start gate reads the box's MemAvailable, the node's use in it — so a running node beside the pair's
    rank blocks no Start of the pair. A node starting beside a running pair has no such measurement: it is refused."""
    cfg, hub, run = _run(tmp_path, LOCAL_ON_PAIR_BOX)
    assert sorted(hub.cluster_cfgs) == ["pair", "speech"], "both boot: the pair measures what the node holds"
    hub.set_cluster({**hub.cluster_views["speech"], "state": "running"}, "speech")
    hub.set_cluster({**hub.cluster_views["pair"], "state": "off", "desired": {"mode": "classic"}}, "pair")
    assert run.blockers("pair") == []
    hub.set_cluster({**hub.cluster_views["pair"], "state": "running"}, "pair")
    hub.set_cluster({**hub.cluster_views["speech"], "state": "off"}, "speech")
    assert [(b["box"], b["cluster"], b["alert"]) for b in run.blockers("speech")] == [("node-2", "pair", True)]
    # … but a node ALREADY UP beside the running pair may relaunch (new settings): the pair started measuring it.
    hub.set_cluster({**hub.cluster_views["speech"], "state": "running"}, "speech")
    assert run.blockers("speech") == []


def test_a_node_starts_beside_a_running_pair_when_the_boxs_measured_memory_leaves_the_floor(tmp_path):
    """The pair's rank reports the box's MemAvailable (tp-rank status): a node starting beside it is judged on that
    measurement — its need taken from it must leave the danger floor — never refused as 'unknown'."""
    cfg, hub, run = _run(tmp_path, LOCAL_ON_PAIR_BOX)
    hub.set_cluster({**hub.cluster_views["speech"], "state": "off"}, "speech")

    def pair(state, avail):
        hub.set_cluster({**hub.cluster_views["pair"], "state": state,
                         "boxes": {"node-2": {"mem_available_gib": avail}}}, "pair")
    pair("running", 30.0)
    assert run.blockers("speech") == []
    pair("running", 15.5)
    # Operator 2026-10-24: "just estimate ram usage and show alert, but allow proceeding" — an ALERT, not a block.
    (b,) = run.blockers("speech")
    assert b["box"] == "node-2" and b["cluster"] == "pair" and b["alert"] is True
    assert "15.5 GiB available" in b["reason"] and "8.41" in b["reason"] and "12" in b["reason"], b["reason"]
    declared = next(d for d in run.declared() if d["name"] == "speech")
    assert declared["blocked_by"] == [] and declared["alerts"] == [b["reason"]]
    pair("starting", 30.0)                       # a load in progress: its use is not settled yet — still only alerted
    assert [(x["cluster"], x["alert"]) for x in run.blockers("speech")] == [("pair", True)]
    pair("running", None)                        # not measured: said so
    assert "not measured" in run.blockers("speech")[0]["reason"]


def test_activation_refuses_an_unknown_or_already_active_cluster(tmp_path):
    from vllm_lb.cluster_run import RunRefused
    cfg, hub, run = _run(tmp_path)
    with pytest.raises(RunRefused) as e:
        run.activate("nope")
    assert e.value.status == 404 and "pair" in str(e.value) and "gui" in str(e.value)
    with pytest.raises(RunRefused) as e:
        run.activate("pair")
    assert e.value.status == 409 and e.value.kind == "cluster_active"


def test_deactivation_removes_an_off_cluster_and_refuses_one_that_is_not(tmp_path):
    from vllm_lb.cluster_run import RunRefused
    cfg, hub, run = _run(tmp_path)
    run.activate("gui")
    with pytest.raises(RunRefused) as e:
        run.deactivate("gui")                       # unknown: not observed yet — never dropped while it may run
    assert e.value.status == 409 and e.value.kind == "cluster_not_off" and "/lb/clusters/gui" in str(e.value)
    hub.set_cluster({**hub.cluster_views["gui"], "state": "off"}, "gui")
    assert run.deactivate("gui") == {"active": ["pair"]}
    assert "gui" not in hub.cluster_cfgs and "gui" not in run.loops.loops
    assert [b.cluster for b in hub.backends() if b.kind == CLUSTER] == ["pair"]
    with pytest.raises(RunRefused) as e:
        run.deactivate("gui")
    assert e.value.kind == "cluster_inactive"


def test_an_activation_lives_in_memory_alone_a_restart_forgets_it(tmp_path):
    cfg, hub, run = _run(tmp_path)
    run.activate("gui")
    before = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*") if p.is_file())
    assert before == ["fleet.yml"], f"nothing written for the activation: {before}"
    # The next process: the same fleet file, the same state dir — the boot set, and nothing else.
    cfg2, hub2, run2 = _run(tmp_path)
    assert list(hub2.cluster_cfgs) == ["pair"]
    from tests.fleet_files import REFERENCE_CHECKPOINTS
    assert (tmp_path / "fleet.yml").read_text() == SHARED.rstrip("\n") + "\n" + REFERENCE_CHECKPOINTS, \
        "the fleet file is never written"


# ── the loops: one per cluster, ticked together ─────────────────────────────────────────────────

def test_one_loop_per_cluster_each_with_its_own_config(tmp_path):
    from vllm_lb.__main__ import build_cluster_loops
    from vllm_lb.cluster_defaults import desired_store
    cfg = _two(tmp_path)
    hub = Hub(cfg)
    loops = build_cluster_loops(cfg, hub, None, {"pair": desired_store(cfg.for_cluster("pair"))}, catalog=None)
    assert set(loops.loops) == {"pair", "gui"}
    assert loops.loops["gui"].cfg.cluster_name == "gui" and loops.loops["pair"].cfg.cluster_name == "pair"
    ticked = []
    for name, loop in loops.loops.items():
        loop.tick = lambda now, n=name: ticked.append(n)
    loops.tick(1.0)
    assert sorted(ticked) == ["gui", "pair"]


def test_the_loop_publishes_under_its_own_name(tmp_path):
    from vllm_lb.cluster_loop import ClusterLoop
    cfg = _two(tmp_path).for_cluster("gui")
    hub = Hub(_two(tmp_path))
    loop = ClusterLoop(cfg, hub, None, executor=None, store=None)
    loop.tick(1.0)
    views = hub.snapshot().extra["clusters"]
    assert views["gui"]["state"] != "unknown" or views["gui"]["detail"] != "not observed yet"
    assert views["pair"]["detail"] == "not observed yet"


# ── routing by model ────────────────────────────────────────────────────────────────────────────

HELD = {"configured": True, "held": True, "max_depth": 8, "desired": {"mode": "cluster"}}


def _snap(*backends, extra) -> Snapshot:
    return Snapshot(now=100.0, backends=list(backends), leases=[], last_request_at=None, last_pressure_at=None,
                    db_ok=True, extra=extra)


def _cluster(url, cluster, model) -> Backend:
    return Backend(kind=CLUSTER, url=url, live=True, served_models=(model,), depth_known=True, cluster=cluster)


def test_a_request_naming_a_served_model_goes_to_the_cluster_serving_it(tmp_path):
    pair, gui = _cluster("http://pair:8100", "pair", "Qwen/Qwen3.8-27B"), _cluster("http://gui:8000", "gui", "MAI-UI-2B")
    snap = _snap(pair, gui, extra={"cluster": HELD, "clusters": {"pair": HELD, "gui": HELD}})
    cfg = replace(_two(tmp_path), broker_enabled=False)
    assert choose(snap, cfg, model="MAI-UI-2B").url == "http://gui:8000"
    assert choose(snap, cfg, model="Qwen/Qwen3.8-27B").url == "http://pair:8100"
    # A loose name nobody serves, or none at all: THE DEFAULT cluster alone (the fleet's `default: true`, the pair),
    # rewritten to what it serves — never the gui node (AUDIT-FINAL P1-3).
    assert choose(snap, cfg, model="qwen3.8:latest").url == "http://pair:8100"
    assert choose(snap, cfg).url == "http://pair:8100"


def test_the_same_model_on_two_clusters_picks_the_shallower_queue():
    a = replace(_cluster("http://a:8100", "pair", "M"), running=3)
    b = replace(_cluster("http://b:8000", "gui", "M"), running=1)
    snap = _snap(a, b, extra={"cluster": HELD, "clusters": {"pair": HELD, "gui": HELD}})
    assert choose(snap, Config(broker_enabled=False), model="M").url == "http://b:8000"


def test_the_router_reads_each_backends_own_cluster_view(tmp_path):
    """R1: held, seats and the wanted mode are the backend's OWN cluster's, from `clusters[b.cluster]` — not one view
    for every cluster backend. (Each named by its model: an unnamed request goes to the default alone, P1-3.)"""
    pair, gui = _cluster("http://pair:8100", "pair", "A"), _cluster("http://gui:8000", "gui", "B")
    off = {"configured": True, "held": False, "max_depth": 8, "desired": {"mode": "classic"}}
    snap = _snap(pair, gui, extra={"cluster": {"configured": False, "clusters": ["pair", "gui"]},
                                   "clusters": {"pair": off, "gui": HELD}})
    cfg = replace(_two(tmp_path), broker_enabled=False)
    chosen = choose(snap, cfg, model="B")
    assert chosen is not None and chosen.url == "http://gui:8000", "gui is held"
    assert choose(snap, cfg, model="A") is None, "the pair is not: its own view says so"


def test_a_catalog_model_nobody_serves_is_refused_naming_what_is_live():
    """FINISHER (R2): a catalog row's id or served name that no live backend serves is `model_not_served` (503) naming
    the live ones — never rewritten onto another cluster's model."""
    from vllm_lb.router import eligible
    gui = _cluster("http://gui:8000", "gui", "MAI-UI-2B")
    snap = _snap(gui, extra={"cluster": HELD, "clusters": {"gui": HELD}})
    verdict = eligible(snap, Config(broker_enabled=False), model="qwen3.8-27b-nvidia-nvfp4")
    assert verdict.backends == [] and "MAI-UI-2B" in verdict.why and "qwen3.8-27b-nvidia-nvfp4" in verdict.why


# ── the API ─────────────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def api(tmp_path):
    import http.client
    from vllm_lb.proxy import Proxy
    started = []

    def make(cfg: Config):
        from vllm_lb.cluster_run import ClusterRun
        hub = Hub(cfg)
        run = ClusterRun(cfg, hub, None)
        run.start(None)
        proxy = Proxy(cfg, hub, cluster_switch=None, run=run)
        proxy.start()
        started.append(proxy)

        def call(method, path, body=None, *, ctype="application/json", raw=None):
            conn = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=10)
            conn.request(method, path, body=raw if raw is not None else None if body is None else json.dumps(body),
                         headers={"Content-Type": ctype} if ctype else {})
            r = conn.getresponse()
            return r.status, json.loads(r.read() or b"{}")
        return hub, call
    yield make
    for p in started:
        p.stop()


def test_status_carries_every_cluster_by_name(tmp_path, api):
    _, call = api(replace(_two(tmp_path), listen_host="127.0.0.1", listen_port=0))
    status, body = call("GET", "/lb/status")
    assert status == 200 and set(body["clusters"]) == {"pair", "gui"}
    assert body["clusters"]["gui"]["name"] == "gui" and body["cluster"]["configured"] is False
    # Each cluster backend says whose head it is — the page's engine card, tiles and badges are keyed by it.
    assert {b["url"]: b["cluster"] for b in body["backends"] if b["kind"] == "cluster"} == {
        "http://192.0.2.10:8100": "pair", "http://198.51.100.20:8000": "gui"}


def test_each_cluster_says_the_catalog_model_it_serves_and_whether_a_chat_may_use_it(tmp_path, api):
    """`serves` (operator 2026-10-18, the chat's server picker: "don't include models chat can't use"): what it LOADED,
    else the choice it was started with — so a server still coming up is known — with the catalog row's `chat` fact.
    null: the catalog knows neither."""
    hub, call = api(replace(_two(tmp_path), listen_host="127.0.0.1", listen_port=0))
    hub.cluster_views["pair"] = {**hub.cluster_views["pair"], "loaded_model": None,
                                 "desired": {"mode": "cluster", "model_choice": "qwen3.8-27b-sglang-dflash2"}}
    hub.cluster_views["gui"] = {**hub.cluster_views["gui"], "desired": {"mode": "cluster", "model_choice": "nope"},
                                "loaded_model": {"served_name": "Tongyi-MAI/MAI-UI-2B", "spec": "mai-ui-2b"}}
    _, body = call("GET", "/lb/status")
    assert body["clusters"]["pair"]["serves"] == {"spec": "qwen3.8-27b-sglang-dflash2", "chat": True,
                                                  "served_name": "Qwen/Qwen3.8-27B-SGLang",
                                                  "label": "Qwen3.8 27B · dense NVFP4 · SGLang DFlash2"}
    assert body["clusters"]["gui"]["serves"]["spec"] == "mai-ui-2b" and body["clusters"]["gui"]["serves"]["chat"] is False
    hub.cluster_views["gui"] = {**hub.cluster_views["gui"], "loaded_model": None}
    assert call("GET", "/lb/status")[1]["clusters"]["gui"]["serves"] is None


def test_the_api_activates_and_deactivates_a_declared_cluster_for_this_run(tmp_path, api):
    cfg = replace(Config.from_env({"LB_ALLOWED_HOSTS": "loopback", "LB_FLEET": write(tmp_path, SHARED), "LB_CLUSTER_STATE": str(tmp_path / "cluster.json")}),
                  listen_host="127.0.0.1", listen_port=0)
    hub, call = api(cfg)
    status, body = call("GET", "/lb/status")
    declared = {d["name"]: d for d in body["declared"]}
    assert declared["gui"]["available"] is True and (declared["alt"]["blocked_by"][0]["box"], declared["alt"]["blocked_by"][0]["cluster"]) == ("node-1", "pair")
    status, body = call("POST", "/lb/clusters/alt/activate")
    assert status == 409 and body["error"]["type"] == "cluster_blocked" and "pair" in body["error"]["message"]
    status, body = call("POST", "/lb/clusters/gui/activate")
    assert status == 200 and body == {"active": ["pair", "gui"]}
    status, body = call("GET", "/lb/status")
    assert set(body["clusters"]) == {"pair", "gui"} and body["clusters"]["gui"]["switch"] == "/lb/clusters/gui"
    status, body = call("POST", "/lb/clusters/gui/deactivate")
    assert status == 409 and body["error"]["type"] == "cluster_not_off"
    hub.set_cluster({**hub.cluster_views["gui"], "state": "off"}, "gui")
    status, body = call("POST", "/lb/clusters/gui/deactivate")
    assert status == 200 and body == {"active": ["pair"]}
    status, body = call("POST", "/lb/clusters/nope/activate")
    assert status == 404 and "gui" in body["error"]["message"]


@pytest.mark.parametrize("ctype", [None, "text/plain", "text/plain; x=json", "application/x-www-form-urlencoded"])
def test_a_run_or_switch_change_a_cross_site_page_could_send_is_refused_and_changes_nothing(tmp_path, api, ctype):
    """CSRF. A browser sends a "simple" POST (no type, text/plain with any parameters, a form) to any origin without a
    preflight. Activating, deactivating and switching a cluster are writes: each needs the media type
    application/json exactly, so a page on another site cannot fire one at the loopback gateway."""
    cfg = replace(Config.from_env({"LB_ALLOWED_HOSTS": "loopback", "LB_FLEET": write(tmp_path, SHARED), "LB_CLUSTER_STATE": str(tmp_path / "cluster.json")}),
                  listen_host="127.0.0.1", listen_port=0)
    hub, call = api(cfg)
    for path, raw in (("/lb/clusters/gui/activate", b"{}"), ("/lb/clusters/pair", b'{"enabled": false}')):
        status, _ = call("POST", path, ctype=ctype, raw=raw)
        assert status in (400, 415), (path, status)
    status, body = call("GET", "/lb/status")
    assert set(body["clusters"]) == {"pair"}, "a refused activate added a cluster to the run"


def test_a_start_is_refused_while_another_active_cluster_holds_the_box(tmp_path, api):
    """The same answer gates the Start (one definition of "taken"): alt is activated while the pair is off, then the
    pair's Start is refused naming alt once alt runs."""
    cfg = replace(Config.from_env({"LB_ALLOWED_HOSTS": "loopback", "LB_FLEET": write(tmp_path, SHARED), "LB_CLUSTER_STATE": str(tmp_path / "cluster.json")}),
                  listen_host="127.0.0.1", listen_port=0)
    hub, call = api(cfg)
    hub.set_cluster({**hub.cluster_views["pair"], "state": "off", "desired": {"mode": "classic"}}, "pair")
    assert call("POST", "/lb/clusters/alt/activate")[0] == 200
    hub.set_cluster({**hub.cluster_views["alt"], "state": "running", "held": True}, "alt")
    status, body = call("POST", "/lb/clusters/pair", {"enabled": True})
    assert status == 409 and body["error"]["type"] == "cluster_blocked" and "alt" in body["error"]["message"]
    status, body = call("POST", "/lb/clusters/pair", {"enabled": False})
    assert status != 409, f"a Stop is never blocked: {body}"


def test_the_singular_switch_is_refused_while_the_fleet_names_several(tmp_path, api):
    _, call = api(replace(_two(tmp_path), listen_host="127.0.0.1", listen_port=0))
    status, body = call("POST", "/lb/cluster", {"enabled": True})
    assert status == 409 and body["error"]["type"] == "cluster_unnamed"
    assert "/lb/clusters/pair" in body["error"]["message"] and "/lb/clusters/gui" in body["error"]["message"]


def test_post_lb_clusters_name_switches_that_cluster_alone(tmp_path, api):
    """The per-cluster switch writes THAT cluster's intent store — the local node's wish under clusters/gui — and
    leaves the pair's alone; an unknown name is a 404 listing the clusters."""
    from vllm_lb import node_control as nc
    cfg = replace(_two(tmp_path), listen_host="127.0.0.1", listen_port=0)
    hub, call = api(cfg)
    gui_dir = os.path.dirname(cfg.for_cluster("gui").cluster_state_path)
    os.makedirs(os.path.join(gui_dir, "node"), exist_ok=True)
    nc.write_report(os.path.join(gui_dir, "node"), nc.report_doc(heartbeat=hub.clock(), host="solo-host", applied=None,
                                                                    container=None, log=None))
    status, body = call("POST", "/lb/clusters/gui", {"enabled": True})
    assert status == 200, body
    assert nc.IntentStore(cfg.for_cluster("gui").cluster_state_path).read()["state"] == "up"
    assert not os.path.exists(cfg.for_cluster("pair").cluster_state_path)
    status, body = call("POST", "/lb/clusters/nope", {"enabled": True})
    assert status == 404 and "pair" in body["error"]["message"] and "gui" in body["error"]["message"]


def test_each_cluster_answers_its_own_estimate_by_name(tmp_path, api):
    """GET /lb/clusters/<name>/estimate — the page asks `${switch}/estimate` — prices a selection on THAT cluster: the
    local node's fit (its card, its launch), the pair's per-box estimate; the singular route names neither once two
    run, and an unknown name is a 404 listing the clusters."""
    _, call = api(replace(_two(tmp_path), listen_host="127.0.0.1", listen_port=0))
    status, body = call("GET", "/lb/clusters/gui/estimate?gpu_memory_utilization=0.85")
    assert status == 200 and set(body) == {"fit", "refused"}, body
    status, body = call("GET", "/lb/clusters/pair/estimate?model_choice=qwen3.8-27b-nvidia-nvfp4")
    # THE PAIR's own estimate: nothing observed yet, so its fleet reserve (ram_usable_fraction of MemTotal) has nothing
    # to be a share of — refused saying so, never priced on a reserve made up.
    assert status == 400 and "MemTotal" in body["error"]["message"], body
    status, body = call("GET", "/lb/cluster/estimate?model_choice=qwen3.8-27b-nvidia-nvfp4")
    assert status == 409 and body["error"]["type"] == "cluster_unnamed", body
    assert "/lb/clusters/gui/estimate" in body["error"]["message"]
    status, body = call("GET", "/lb/clusters/nope/estimate")
    assert status == 404 and "gui" in body["error"]["message"]


def test_a_driven_clusters_view_publishes_where_its_switch_is_posted(tmp_path):
    """The panel's Start/Stop is driven by the view — `switch` is the route for a cluster this LB drives, "" for one it
    only observes."""
    hub = Hub(_two(tmp_path))
    views = hub.snapshot().extra["clusters"]
    assert views["pair"]["switch"] == "/lb/clusters/pair" and views["gui"]["switch"] == "/lb/clusters/gui"
    observed = Config.from_env({"LB_ALLOWED_HOSTS": "loopback", "LB_FLEET": write(tmp_path, ONE_HOST, "o.yml")})
    assert Hub(observed).snapshot().extra["clusters"]["node"]["switch"] == ""


def test_the_classic_planner_stands_aside_while_any_cluster_is_wanted(tmp_path):
    """FINISHER (R7): control.py and the scaler read "a cluster is wanted" off EVERY cluster's view, not the alias."""
    from vllm_lb.control import cluster_wanted
    extra = {"cluster": {"configured": False, "clusters": ["pair", "gui"]},
             "clusters": {"pair": {"desired": {"mode": "classic"}}, "gui": {"desired": {"mode": "cluster"}}}}
    assert cluster_wanted(_snap(extra=extra)) is True


# ── the new box family: data, never a branch ────────────────────────────────────────────────────

#: The gui node on an AMD iGPU: a BF16 row (mai-ui-2b, no MFU patch — a modelopt_mixed one is refused on a type whose
#: mfu_patch_loads is false, tests/test_node_guards.py), so what refuses it here is the box's missing pool bound.
AMD_NODE = PARALLEL.replace("""      - {index: 0, type: rtx5090, serves: true}""", """      - {index: 0, type: radeon8060s, serves: true}""").replace(
    "    model: qwen3.8-27b-nvidia-nvfp4\n", "    model: mai-ui-2b\n").replace(
    "      choices: [qwen3.8-27b-nvidia-nvfp4]\n", "      choices: [mai-ui-2b]\n")


def test_a_local_node_on_the_amd_igpu_is_bounded_by_its_boxs_ram_not_by_vllms_free_figure():
    """R5, the motivating target (an AMD iGPU node). History: the RED version asked it to parse; the finisher pinned
    the honest refusal while no start memory was measured; MEASURED 2026-10-17 (radeon8060s.yml: 124.0 / 119.26 GiB),
    the start check now passes — and the box bounds the pool instead (unified memory: the pool is the box's own RAM,
    which vLLM's free figure overstates). The reference AMD box states no `gpu_ram_max_gib`: refused naming it.
    The launch that parses is tests/test_gui_node.py's."""
    from vllm_lb.hardware import types
    assert "local" in types()["radeon8060s"].controls
    with pytest.raises(ValueError) as refused:
        parse_fleet(AMD_NODE, "fleet.yml")
    why = str(refused.value)
    assert "gpu_ram_max_gib" in why and "solo" in why and "visible_gib" not in why, why


def test_every_hardware_type_states_how_a_container_attaches_it():
    """FINISHER (R6): `gpu_runtime` — nvidia (a device reservation) or rocm (/dev/kfd + /dev/dri) — is a FACT of the
    type, rendered into the node's env, never an `if amd:` (tests/test_hardware_guard.py)."""
    from vllm_lb.fleet import node_env
    from vllm_lb.hardware import types
    runtimes = {key: t.gpu_runtime.value for key, t in types().items()}
    assert set(runtimes.values()) <= {"nvidia", "rocm"} and None not in runtimes.values(), runtimes
    assert "EDGE_NODE_GPU_RUNTIME=nvidia" in node_env(parse_fleet(PARALLEL, "fleet.yml"), "gui")


# ── the proxy routes by the model a request names (R2's other half) ─────────────────────────────

@pytest.fixture
def two_live(tmp_path):
    """Two clusters whose heads are fake servers — the pair serving `A`, gui serving `B` — both held, behind a real
    Proxy. Yields (proxy, hub, pair_head, gui_head, post)."""
    import http.client
    from tests.fakes import FakeBackend
    from vllm_lb.proxy import Proxy
    a, b = FakeBackend(served=("A",)), FakeBackend(served=("B",))
    text = PARALLEL.replace("http://192.0.2.10:8100", a.url).replace("http://198.51.100.20:8000", b.url)
    cfg = replace(Config.from_env({"LB_ALLOWED_HOSTS": "loopback", "LB_FLEET": write(tmp_path, text), "LB_CLUSTER_STATE": str(tmp_path / "c.json")}),
                  listen_host="127.0.0.1", listen_port=0, connect_timeout_s=2.0, first_byte_timeout_s=10.0,
                  broker_enabled=False)
    hub = Hub(cfg)
    for name, head, model in (("pair", a, "A"), ("gui", b, "B")):
        hub.update_backend(head.url, live=True, served_models=(model,), max_model_len=4096, depth_known=True)
        hub.set_cluster({**hub.cluster_views[name], **HELD, "state": "running"}, name)
    proxy = Proxy(cfg, hub)
    proxy.start()

    def post(model, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=20)
        conn.request("POST", "/v1/messages", body=json.dumps({"model": model, "max_tokens": 1}).encode(),
                     headers={"Content-Type": "application/json"})
        r = conn.getresponse()
        if headers is not None:
            headers.update({k.lower(): v for k, v in r.getheaders()})
        return r.status, json.loads(r.read() or b"{}")
    yield proxy, hub, a, b, post
    proxy.stop()
    a.close()
    b.close()


def _posts(head) -> list[dict]:
    return [json.loads(r["body"]) for r in head.requests if r["method"] == "POST"]


def test_the_proxy_sends_each_request_to_the_cluster_serving_the_model_it_names(two_live):
    _, _, a, b, post = two_live
    assert post("B")[0] == 200 and post("A")[0] == 200 and post("B")[0] == 200
    assert [p["model"] for p in _posts(a)] == ["A"] and [p["model"] for p in _posts(b)] == ["B", "B"]


def test_an_answer_says_which_boxes_of_its_own_cluster_served_it(two_live):
    _, _, _, _, post = two_live
    seen: dict = {}
    assert post("B", seen)[0] == 200 and seen["x-lb-box"] == "solo", seen
    assert post("A", seen)[0] == 200 and seen["x-lb-box"] == "node-1 + node-2", seen


# (R2's proxy-level 503 for a catalog profile the DEFAULT can launch was retired 2026-10-17 — it broke a client; such a
# name is the default's: test_parallel_clusters_audit.py `test_a_profile_the_default_can_launch_goes_to_the_default…`.
# The refusal stands for a model only ANOTHER cluster runs: `test_a_model_only_another_cluster_runs_is_refused…`.)


def test_every_routing_decision_of_a_request_carries_the_model_it_names(two_live, monkeypatch):
    """The 3 call sites (the first choice, the claim wait's counterfactual and re-choice, the hold): each asks the
    router with the request's model — a held turn for gui is never released onto the pair."""
    from vllm_lb import router
    proxy, hub, a, b, post = two_live
    seen, real = [], router.choose
    monkeypatch.setattr(router, "choose", lambda *args, **kw: seen.append(kw.get("model")) or real(*args, **kw))
    hub.update_backend(b.url, live=False)
    hub.set_cluster({**hub.cluster_views["gui"], "held": False, "state": "starting"}, "gui")
    threading_timer = __import__("threading").Timer(0.5, lambda: (
        hub.update_backend(b.url, live=True, served_models=("B",)),
        hub.set_cluster({**hub.cluster_views["gui"], "held": True, "state": "running"}, "gui")))
    threading_timer.start()
    status, body = post("B")
    assert status == 200, body
    assert len(seen) >= 2 and set(seen) == {"B"}, seen
    assert _posts(a) == [] and [p["model"] for p in _posts(b)] == ["B"]
