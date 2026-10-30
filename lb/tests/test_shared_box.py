"""TWO CLUSTERS ON ONE BOX AT ONCE, WHERE THEIR MEMORY FITS (operator, 2026-10-23: the gui node and the speech node both
on the APU box).

  * a box's GPU memory is ONE pool where its card's family says so (unified memory: the box's `gpu_ram_max_gib`); the
    clusters holding it share it while the sum of their needs (`fleet.pool_need`: a vLLM launch's pool, a
    faster-whisper engine's measured memory) is within the bound;
  * over the bound, or with any need unknown, or on a box with no shared pool (discrete VRAM): the later cluster is
    BLOCKED naming the box, its holders and the GiB — never evicted, never guessed (fail closed);
  * the boot set follows the same rule;
  * broker_parrot claims one cluster row per lane, so nodes sharing a box share ONE reservation: the first reserves,
    the other rides it, and the last to stop returns it.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from tests.fleet_files import write
from vllm_lb import node_control as nc
from vllm_lb.config import Config
from vllm_lb.fleet import parse_fleet, pool_need, share_problem

FW = "faster-whisper-large-v3-turbo"


def fleet(bound: int = 20, *, gui_boot: bool = False, card: str = "radeon8060s") -> str:
    return f"""
boxes:
  apu:
    hostname: apu-host
    label: apu-gpu
    platform: linux/amd64
    gpu_ram_max_gib: {bound}
    node_limits: {{memory_gib: 24, cpus: 16, pids: 4096, nice: 10}}
    cards:
      - {{index: 0, type: {card}, serves: true}}
clusters:
  gui:
    control: local
    run_at_boot: {'true' if gui_boot else 'false'}
    default: true
    model: mai-ui-2b
    url: http://gui-node:8000
    launch:
      image: rocm/vllm@sha256:{'1' * 64}
      max_model_len: 32768
      max_num_seqs: 4
      gpu_memory_utilization: 0.10
      tool_call_parser: hermes
      choices: [mai-ui-2b]
    ranks:
      - {{box: apu}}
  stt:
    control: local
    run_at_boot: true
    model: whisper-large-v3-turbo
    url: http://stt-node:8001
    launch:
      image: {{vllm: "vllm-audio@sha256:{'3' * 64}", faster-whisper: "sha256:{'4' * 64}"}}
      max_model_len: 448
      max_num_seqs: 4
      gpu_memory_utilization: 0.05
      choices: [whisper-large-v3-turbo, {FW}]
    ranks:
      - {{box: apu}}
backends: {{}}
"""


# ── the fleet: one need per launch, one rule per box ───────────────────────────────────────────────────────────────

def test_a_launchs_need_is_its_pool_on_the_box():
    f = parse_fleet(fleet(), "fleet.yml")
    assert pool_need(f.clusters["gui"]) == pytest.approx(12.4), "0.10 × the type's 124.0 GiB visible pool"
    assert pool_need(f.clusters["stt"]) == pytest.approx(6.2), "vLLM's: 0.05 × 124.0 GiB"
    assert pool_need(f.clusters["stt"], {"model": FW}) == pytest.approx(2.59), "faster-whisper: its measured memory"
    with pytest.raises(ValueError):
        pool_need(f.clusters["stt"], {"gpu_memory_utilization": 0.5})     # 62 GiB > the box's 20: refused, not summed


def test_clusters_share_a_box_while_their_needs_fit_its_bound_and_are_refused_in_numbers_beyond():
    box = parse_fleet(fleet(20), "fleet.yml").boxes["apu"]
    assert share_problem(box, {"gui": 12.4, "stt": 6.2}) == ""
    over = share_problem(replace(box, gpu_ram_max_gib=16), {"gui": 12.4, "stt": 6.2})
    assert over == "gui 12.40 GiB + stt 6.20 GiB = 18.60 GiB > box apu's gpu_ram_max_gib 16 GiB", over


def test_an_unknown_need_or_a_box_with_no_shared_pool_holds_the_box_whole():
    box = parse_fleet(fleet(20), "fleet.yml").boxes["apu"]
    unknown = share_problem(box, {"gui": 12.4, "pair": "no launch"})
    assert "pair's memory need is unknown (no launch)" in unknown and "apu" in unknown
    discrete = parse_fleet(fleet(20).replace("radeon8060s", "rtx5090").replace("    run_at_boot: true\n", "    run_at_boot: false\n"),
                           "fleet.yml").boxes["apu"]
    assert "VRAM of their own" in share_problem(discrete, {"gui": 1.0, "stt": 1.0})


def test_the_boot_set_shares_a_box_by_the_same_rule():
    f = parse_fleet(fleet(20, gui_boot=True), "fleet.yml")
    assert [c.run_at_boot for c in f.clusters.values()] == [True, True]
    with pytest.raises(ValueError) as refused:
        parse_fleet(fleet(16, gui_boot=True), "fleet.yml")
    assert "box apu" in str(refused.value) and "18.60 GiB > box apu's gpu_ram_max_gib 16 GiB" in str(refused.value)


# ── the run: activate and Start ────────────────────────────────────────────────────────────────────────────────────

def _cfg(tmp_path, text) -> Config:
    return replace(Config.from_env({"LB_ALLOWED_HOSTS": "loopback", "LB_FLEET": write(tmp_path, text), "LB_CLUSTER_STATE": str(tmp_path / "c.json")}),
                   listen_host="127.0.0.1", listen_port=0, perf_dir=str(tmp_path / "perf"))


def _run(cfg, leases=None):
    from vllm_lb.cluster_run import ClusterRun
    from vllm_lb.hub import Hub
    hub = Hub(cfg)
    run = ClusterRun(cfg, hub, None)
    run.start(leases)
    return hub, run


def _wish(cfg, name, state, launch=None, now=1.0):
    path = cfg.for_cluster(name).cluster_state_path
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return nc.IntentStore(path).write(state, launch or {}, now=now)


def test_a_second_cluster_runs_beside_the_first_while_the_box_holds_both(tmp_path):
    cfg = _cfg(tmp_path, fleet(20))
    _wish(cfg, "stt", nc.UP)
    hub, run = _run(cfg)
    assert run.blockers("gui") == []
    assert run.activate("gui")["active"] == ["stt", "gui"]
    assert run.peers("gui") == {"stt": True}


def test_over_the_bound_the_later_cluster_is_blocked_naming_box_holders_and_gib(tmp_path):
    from vllm_lb.cluster_run import RunRefused
    cfg = _cfg(tmp_path, fleet(16))
    _wish(cfg, "stt", nc.UP)
    hub, run = _run(cfg)
    [blocked] = run.blockers("gui")
    assert blocked["box"] == "apu" and blocked["cluster"] == "stt"
    with pytest.raises(RunRefused) as e:
        run.activate("gui")
    assert e.value.kind == "cluster_blocked"
    assert "box apu is held by cluster stt: gui 12.40 GiB + stt 6.20 GiB = 18.60 GiB > box apu's gpu_ram_max_gib 16" \
        in str(e.value), str(e.value)
    # The holder's need is its STORED wish's: on faster-whisper (2.59 GiB) the same box holds both.
    _wish(cfg, "stt", nc.UP, {"model": FW}, now=2.0)
    assert run.blockers("gui") == []


def test_a_start_is_judged_at_the_launch_it_asks_for(tmp_path):
    import http.client
    import json
    from vllm_lb.proxy import Proxy
    cfg = _cfg(tmp_path, fleet(20))
    hub, run = _run(cfg)
    hub.set_cluster({**hub.cluster_views["stt"], "state": "off"}, "stt")
    run.activate("gui")
    _wish(cfg, "gui", nc.UP)                                     # gui holds 12.4 GiB of the 20
    proxy = Proxy(cfg, hub, cluster_switch=None, run=run)
    proxy.start()
    try:
        def post(body):
            conn = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=10)
            conn.request("POST", "/lb/clusters/stt", body=json.dumps(body), headers={"Content-Type": "application/json"})
            r = conn.getresponse()
            return r.status, json.loads(r.read() or b"{}")
        status, body = post({"enabled": True, "gpu_memory_utilization": 0.1})     # 12.4 + 12.4 > 20
        assert status == 409 and body["error"]["type"] == "cluster_blocked", body
        assert "gui 12.40 GiB" in body["error"]["message"] and "stt 12.40 GiB" in body["error"]["message"]
        status, body = post({"enabled": True})                                    # 12.4 + 6.2 fits: past the box check
        assert status != 409, body
    finally:
        proxy.stop()


# ── the broker: one reservation per box, shared ────────────────────────────────────────────────────────────────────

def _rows():
    from queue_workflows import db
    with db.connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT j.id, j.status, j.inputs, r.status AS run_status FROM workflow_node_jobs j "
                    "JOIN workflow_runs r ON r.id = j.run_id ORDER BY j.created_at")
        return [dict(r) for r in cur.fetchall()]


class Box:
    """Two local nodes on one box behind one run with the broker on; each node's host applies what it is released."""

    def __init__(self, tmp_path):
        from tests.test_cluster_one_host import Clock
        from vllm_lb.leases import BrokerLeases
        self.cfg = replace(_cfg(tmp_path, fleet(20)), broker_enabled=True, lease_workflow="vllm_lb_lease",
                           lease_node_module="system.vllm_lease")
        self.clock = Clock()
        self.hub, self.run = _run(self.cfg, BrokerLeases(self.cfg))
        self.run.activate("gui")
        for loop in self.run.loops.loops.values():
            loop.clock = self.clock
        self.beat()

    def node_dir(self, name):
        return nc.node_dir(self.cfg.for_cluster(name).cluster_state_path)

    def released(self, name):
        return nc.IntentStore(f"{self.node_dir(name)}/{nc.INTENT}").read()

    def beat(self):
        """Each host has applied what it was released (a down leaves the node off)."""
        import os
        for name in ("gui", "stt"):
            os.makedirs(self.node_dir(name), exist_ok=True)
            done = self.released(name)
            applied = {**done, "done_at": self.clock.t, "exit": 0, "stderr": "", "refused": ""} if done else None
            nc.write_report(self.node_dir(name), nc.report_doc(heartbeat=self.clock.t, host="apu-host",
                                                               applied=applied, applying=None, container=None, log=None))

    def tick(self):
        self.run.loops.tick(self.clock.t)
        self.clock.t += 2.0
        self.beat()
        return self.hub.cluster_views


def test_nodes_sharing_a_box_share_one_reservation_and_the_last_out_returns_it(tmp_path, broker_db):
    from tests.fakes import claim_gpu
    box = Box(tmp_path)
    _wish(box.cfg, "stt", nc.UP, now=box.clock.t)
    box.tick()
    assert [r["inputs"]["cluster"] for r in _rows()] == ["stt"]
    assert claim_gpu("apu-gpu", box="apu", served_types=("cluster",)) is not None
    box.tick()
    assert box.released("stt")["state"] == nc.UP
    # gui Starts on the held box: it rides stt's row — broker_parrot would never claim a second cluster row there.
    _wish(box.cfg, "gui", nc.UP, now=box.clock.t)
    views = box.tick()
    assert len(_rows()) == 1, _rows()
    assert box.released("gui")["state"] == nc.UP, views["gui"]
    assert views["gui"]["reservation"]["state"] == "held"
    # stt stops first: it is off, and its row is KEPT for gui, which still holds the box.
    _wish(box.cfg, "stt", nc.DOWN, now=box.clock.t)
    for _ in range(3):
        views = box.tick()
    assert views["stt"]["state"] == "off" and "kept for cluster gui" in views["stt"]["detail"], views["stt"]
    assert [(r["status"], r["run_status"]) for r in _rows()] == [("running", "running")]
    # gui stops: the last out returns the box to broker_parrot.
    _wish(box.cfg, "gui", nc.DOWN, now=box.clock.t)
    for _ in range(3):
        views = box.tick()
    assert views["gui"]["state"] == "off", views["gui"]
    assert _rows()[0]["run_status"] != "running", _rows()


# ── an observed node beside the pair ───────────────────────────────────────────────────────────────────────────────

def observed(local_beside: bool = False) -> str:
    """The pair on two GB10s, and on the second a speech node its own deployment starts (control none): what the pair's
    start measures (the box's free memory) already holds it."""
    card = "[{index: 0, type: gb10, serves: true}]"
    gui = f"""
  gui:
    control: local
    run_at_boot: false
    model: mai-ui-2b
    url: http://gui-node:8000
    launch:
      image: rocm/vllm@sha256:{'1' * 64}
      max_model_len: 32768
      max_num_seqs: 4
      gpu_memory_utilization: 0.10
      tool_call_parser: hermes
      choices: [mai-ui-2b]
    ranks: [{{box: apu}}]""" if local_beside else ""
    apu = f"""
  apu: {{hostname: apu-host, label: apu-gpu, platform: linux/amd64, gpu_ram_max_gib: 20,
        node_limits: {{memory_gib: 24, cpus: 16, pids: 4096, nice: 10}}, cards: [{{index: 0, type: radeon8060s, serves: true}}]}}""" \
        if local_beside else ""
    return f"""
boxes:
  box-a: {{hostname: box-a-host, label: box-a-gpu, platform: linux/arm64, ssh: op@192.0.2.10, cards: {card}}}
  box-b: {{hostname: box-b-host, label: box-b-gpu, platform: linux/arm64, ssh: op@192.0.2.11, cards: {card}}}{apu}
clusters:
  pair:
    control: ssh
    iface: bond0
    link: roce
    run_at_boot: true
    default: true
    url: http://192.0.2.10:8100
    ram_usable_fraction: 0.98
    max_num_batched_tokens: 8192
    ranks: [{{box: box-a, interconnect: 198.51.100.1}}, {{box: box-b, interconnect: 198.51.100.2}}]
  speech:
    control: none
    run_at_boot: true
    model: parakeet-tdt-0.6b-v3
    url: http://192.0.2.11:18000
    ranks: [{{box: {'apu' if local_beside else 'box-b'}}}]{gui}
backends: {{}}
"""


def test_an_observed_node_boots_beside_the_pair_and_neither_blocks_the_other(tmp_path):
    f = parse_fleet(observed(), "fleet.yml")
    assert [c.run_at_boot for c in f.clusters.values()] == [True, True]
    cfg = _cfg(tmp_path, observed())
    hub, run = _run(cfg)
    assert run.holding("speech")                                  # observed, not yet seen off: it may run there
    assert run.blockers("pair") == [], "the pair's start measures the box's free memory, the node's use in it"
    assert run.blockers("speech") == [], "an observed node starts nothing"


def test_beside_a_node_sized_from_stated_needs_an_observed_node_still_holds_its_box(tmp_path):
    cfg = _cfg(tmp_path, observed(local_beside=True))
    hub, run = _run(cfg)
    [blocked] = run.blockers("gui")
    assert blocked["cluster"] == "speech" and "speech's memory need is unknown" in blocked["reason"], blocked
