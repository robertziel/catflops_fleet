"""THE ONE-HOST TOPOLOGY: a standalone server's node as the cluster — observed over its own HTTP API, never driven
(worklog/standalone-cluster-panel.md).

One host with no SSH channel, control `none`: the same ClusterLoop and the same cluster view the TP=2 pair publishes,
the same backend kind, and a plan with no actions — `running` while the node answers, `unknown` (with the probe's own
words) while it does not. So the page's load figures — the KV tile, the targets, the queue against the seats, the
ceilings row — render for the node as they do for the pair, and nothing pretends the LB can start, stop or configure
what its server file runs.

WHERE THE TOPOLOGY COMES FROM is the fleet file (AGENTS.md, "Open-source ready"; its loader is the hardware domain's):
`Config.cluster_hosts`, `cluster_url` and `cluster_control` are the fields it fills, and no environment variable names
a control channel. The environment's one cluster form, `LB_CLUSTER_HOSTS` (retiring into that file), can only express
the SSH pair — so these tests build the node's Config directly, as the loader will.

box-c's two SSH hosts are untouched: control `ssh` is the field's default, and the pair's view gains one key
(`topology`) — pinned below beside the node's.
"""
from __future__ import annotations

import http.client
import json

import pytest

from tests.fakes import FakeClusterExecutor, FakeProber
from vllm_lb import router, topology
from vllm_lb.__main__ import build_cluster_loop
from vllm_lb.catalog import ModelCatalog
from vllm_lb.cluster import ClusterHost, parse_hosts
from vllm_lb.cluster_defaults import desired_store
from vllm_lb.cluster_loop import ClusterLoop
from vllm_lb.config import Config
from vllm_lb.control import Controller
from vllm_lb.hub import Hub
from vllm_lb.probe import SCRAPE_KEYS
from vllm_lb.proxy import Proxy
from vllm_lb.state import CLUSTER

NODE = "http://node-a:8000"
#: The node as the fleet file declares it: one box, no SSH target — a label and a box.
HOST = ClusterHost("node-a-gpu", "node-a")
PAIR = "pair-a-gpu=pair-a=ops@192.0.2.10=198.51.100.1,pair-b-gpu=pair-b=ops@192.0.2.11=198.51.100.2"
HEAD = "http://192.0.2.10:8100"
#: The engine's own process start, as vLLM's /metrics says it (`process_start_time_seconds`).
STARTED = 1_791_741_600.0
SERVED = "model-a"


def scrape(**read) -> dict:
    """One /metrics read in the probe's canonical keys: everything the engine did not say is None."""
    out = dict.fromkeys(key for key, _ in SCRAPE_KEYS)
    out.update(waiting=0, running=0, gen_tokens_total=7895, prompt_tokens_total=48504,
               process_start_time=STARTED, kv_cache_usage=0.0, kv_pool_tokens=168937, kv_pool_blocks=116,
               kv_block_tokens=1568)
    out.update(read)
    return out


LIVE = {"live": True, "served_models": (SERVED,), "max_model_len": 131072, "waiting": 0,
        "running": 0, "depth_known": True, "last_error": "", "scrape": scrape()}
DOWN = {"live": False, "last_error": "ConnectionRefusedError: [Errno 111] Connection refused"}


class Clock:
    def __init__(self, t: float = 1_791_742_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def node_cfg(**over) -> Config:
    """The node's Config as the fleet loader fills it."""
    return Config(**{"cluster_name": "node", "cluster_hosts": (HOST,), "cluster_url": NODE, "cluster_control": "none",
                     "probe_interval_s": 0, **over})


class Node:
    """A standalone server's LB as `__main__` builds it, its node answering from a table."""

    def __init__(self, answer: dict, **over) -> None:
        self.cfg = node_cfg(**over)
        self.clock = Clock()
        self.hub = Hub(self.cfg, clock=self.clock)
        self.loop = build_cluster_loop(self.cfg, self.hub, None, None, catalog=ModelCatalog.load())
        self.loop.clock = self.clock
        self.prober = FakeProber({NODE: answer})
        self.ctl = Controller(self.cfg, self.hub, None, prober=self.prober, clock=self.clock,
                              cluster_loop=self.loop)

    def tick(self, seconds: float = 2.0) -> dict:
        self.ctl.tick()
        self.clock.t += seconds
        return self.hub.snapshot().extra["cluster"]


# ── the topology ───────────────────────────────────────────────────────────────────────────────


def test_a_host_with_no_ssh_target_is_a_label_and_a_box():
    assert (HOST.ssh_target, HOST.interconnect) == ("", "")
    assert parse_hosts(PAIR)[0] == ClusterHost("pair-a-gpu", "pair-a", "ops@192.0.2.10", "198.51.100.1"), \
        "the SSH form parses exactly as before"


def test_a_one_host_node_is_a_configured_cluster_of_tp_1():
    cfg = node_cfg()
    assert cfg.cluster_configured is True
    assert (cfg.cluster_control, cfg.cluster_tp) == ("none", 1)


def test_the_fleet_file_configures_the_pair_and_the_environment_form_is_retired(tmp_path):
    from tests.fleet_files import PAIR as PAIR_FLEET, write
    cfg = Config.from_env({"LB_ALLOWED_HOSTS": "loopback", "LB_FLEET": write(tmp_path, PAIR_FLEET)})
    assert (cfg.cluster_control, cfg.cluster_tp, cfg.cluster_configured) == ("ssh", 2, True)
    # LB_CLUSTER_HOSTS moved into the fleet file (config.py RETIRED_KEYS): refused, naming where it went.
    with pytest.raises(ValueError, match="LB_CLUSTER_HOSTS"):
        Config.from_env({"LB_ALLOWED_HOSTS": "loopback", "LB_CLUSTER_HOSTS": PAIR, "LB_CLUSTER_URL": HEAD})
    with pytest.raises(ValueError):
        parse_hosts("node-a-gpu=node-a")


def test_control_none_with_nothing_named_is_simply_off():
    assert Config(cluster_control="none").cluster_configured is False


@pytest.mark.parametrize("over, says", [
    ({"cluster_control": "docker"}, "control"),
    ({"cluster_name": "node", "cluster_hosts": (HOST, ClusterHost("node-b-gpu", "node-b"))}, "ONE node"),
    ({"cluster_name": "node", "cluster_hosts": (ClusterHost("node-a-gpu", "node-a", "ops@192.0.2.9", "192.0.2.9"),)}, "SSH target"),
    ({"cluster_url": ""}, "URL"),
    # A host with no channel under tp-rank: there is no box to reach.
    ({"cluster_control": "ssh", "cluster_name": "node", "cluster_hosts": (HOST, ClusterHost("node-b-gpu", "node-b"))}, "no SSH target"),
])
def test_a_topology_this_lb_cannot_serve_refuses_naming_why(over, says):
    with pytest.raises(ValueError, match=says):
        node_cfg(**over)


# ── the loop: observed, never driven ───────────────────────────────────────────────────────────


def test_a_node_that_answers_is_the_cluster_running_and_held():
    node = Node(LIVE)
    view = node.tick()
    assert view["configured"] is True
    assert (view["state"], view["held"]) == ("running", True)
    assert view["topology"] == {"tp": 1, "boxes": ["node-a"], "control": "none", "why": topology.OBSERVED_WHY,
                                "steps": ["load", "warm up", "serve"]}
    assert view["url"] == NODE
    assert view["up_since"] == STARTED, "the engine's own process start, off its /metrics"
    box = view["boxes"]["node-a"]
    assert box["label"] == "node-a-gpu" and box["rank"]["present"] is True and box["rank"]["running"] is True
    assert view["loaded_model"]["served_name"] == SERVED
    # No fleet declares its seats here: none is claimed, and the classic boxes' limit is not borrowed
    # (tests/test_cluster_seats_routing.py).
    assert view["max_depth"] is None


def test_what_the_node_runs_is_its_own_report_and_nothing_else():
    """`effective` for a node this LB only observes is what it SAYS it runs: its served model and the checkpoint
    folder it loaded, its window, the GPU memory fraction and KV cache dtype it was started with, its prefix caching,
    its engine, its pool's own byte pin (none: not pinned), no KV tier in the host's RAM (it states none), and no
    drafter — it publishes no spec-decode counter, which vLLM registers only with a speculative config. Every setting
    it does not publish (its seats, its prefill budget) is null: never a guess, never a default. Null while it does not
    answer."""
    answer = dict(LIVE, model_roots=("/models/Model-A-NVFP4",),
                  scrape=scrape(engine="vllm", gpu_memory_utilization=0.9, kv_cache_dtype="fp8_e4m3",
                                prefix_caching=True))
    eff = Node(answer).tick()["effective"]
    assert eff == {"model_choice": None, "generation": None, "max_num_seqs": None, "max_num_batched_tokens": None,
                   "num_speculative_tokens": 0, "ple_replicate": None, "kv_cache_memory": None,
                   "kv_offload_gib": None, "kv_offload_ram_gib": None, "max_model_len": 131072,
                   "long_prefill_token_threshold": None,
                   "uncensored": None, "served_name": SERVED, "checkpoint": "Model-A-NVFP4", "engine": "vllm",
                   "gpu_memory_utilization": 0.9, "kv_cache_dtype": "fp8_e4m3", "prefix_caching": True}
    assert Node(DOWN).tick()["effective"] is None


def test_a_node_that_does_not_answer_is_unknown_in_the_probes_own_words():
    node = Node(DOWN)
    view = node.tick()
    assert (view["state"], view["held"]) == ("unknown", False)
    assert "Connection refused" in view["detail"]
    assert view["boxes"]["node-a"]["rank"] is None, "a node that does not answer has not said it is gone"
    assert view["up_since"] is None


def test_nothing_is_ever_asked_of_an_executor_and_the_state_clock_moves_on_a_change():
    node = Node(LIVE)
    assert node.loop.executor is None
    first = node.tick()
    assert first["since"] == pytest.approx(node.clock.t - 2.0)
    node.prober.table[NODE] = DOWN
    node.tick()
    second = node.tick()
    assert second["state"] == "unknown" and second["since"] > first["since"]


def test_the_router_sends_turns_to_the_observed_node_while_it_answers():
    node = Node(LIVE)
    node.tick()
    chosen = router.choose(node.hub.snapshot(), node.cfg)
    assert chosen is not None and (chosen.url, chosen.kind) == (NODE, CLUSTER)
    node.prober.table[NODE] = DOWN
    node.tick()
    assert router.choose(node.hub.snapshot(), node.cfg) is None


def test_the_observed_view_has_the_pairs_shape_and_the_pair_gains_only_its_topology(tmp_path):
    node = Node(LIVE)
    observed = node.tick()
    cfg = Config(broker_enabled=False, cluster_name="pair", cluster_hosts=parse_hosts(PAIR), cluster_url=HEAD,
                 cluster_state_path=str(tmp_path / "cluster.json"), probe_interval_s=0, cluster_observe_s=0.0)
    hub = Hub(cfg)
    loop = ClusterLoop(cfg, hub, None, executor=FakeClusterExecutor(("pair-a", "pair-b")),
                       store=desired_store(cfg))
    Controller(cfg, hub, None, prober=FakeProber({HEAD: DOWN}), cluster_loop=loop).tick()
    pair = hub.snapshot().extra["cluster"]
    assert set(observed) == set(pair)
    assert set(observed["boxes"]["node-a"]) == set(pair["boxes"]["pair-a"])
    assert pair["topology"] == {"tp": 2, "boxes": ["pair-a", "pair-b"], "control": "ssh", "why": "",
                                "steps": ["free", "load", "warm up", "serve"]}, "broker off: nothing to reserve"


def test_the_pre_tick_view_is_the_observed_shape_too():
    cfg = node_cfg()
    early = Hub(cfg).cluster_state()
    assert set(early) == set(Node(LIVE).tick())
    assert (early["state"], early["held"], early["max_depth"]) == ("unknown", False, None)
    assert early["topology"]["control"] == "none"


def test_the_start_steps_are_only_the_ones_that_apply():
    assert topology.start_steps("ssh", broker=True) == ("free", "reserve", "drain", "load", "warm up", "serve")
    assert topology.start_steps("ssh", broker=False) == ("free", "load", "warm up", "serve")
    assert topology.start_steps("none", broker=False) == ("load", "warm up", "serve")
    # a node the fleet says reserves nothing (`reserve: false`) passes no reserve/drain, the broker on or not
    assert topology.start_steps("local", broker=True, reserves=False) == ("load", "warm up", "serve")


# ── the switch and the estimate say why there is nothing to switch ──────────────────────────────


@pytest.fixture
def served():
    started: list[Proxy] = []

    def make():
        cfg = node_cfg(listen_host="127.0.0.1", listen_port=0)
        proxy = Proxy(cfg, Hub(cfg), cluster_switch=None)
        proxy.start()
        started.append(proxy)
        return proxy

    yield make
    for proxy in started:
        proxy.stop()


def call(proxy: Proxy, method: str, path: str, body=None) -> tuple[int, dict]:
    conn = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=10)
    conn.request(method, path, body=None if body is None else json.dumps(body),
                 headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    return resp.status, json.loads(resp.read() or b"{}")


@pytest.mark.parametrize("method, path, body", [
    ("POST", "/lb/cluster", {"enabled": False}),
    ("POST", "/lb/cluster", {"enabled": True, "max_num_seqs": 4}),
    ("GET", "/lb/cluster/estimate?model_choice=qwen3.8-27b-nvidia-nvfp4", None),
])
def test_a_node_this_lb_only_observes_refuses_a_switch_or_an_estimate_saying_why(served, method, path, body):
    status, out = call(served(), method, path, body)
    assert status == 409, out
    assert out["error"]["type"] == "cluster_observed_only"
    assert out["error"]["message"] == topology.OBSERVED_WHY


# ── saturation: one rank, its own card, no link ─────────────────────────────────────────────────


def test_the_saturation_row_is_one_rank_with_no_link_and_no_busiest_card(monkeypatch):
    """The node's SM fill is ITS card's (`topology.serving_card`, the one adapter the hardware domain's
    per-box configuration plugs into) — and with no reading, a gap with the reason, never the box's busiest
    card, which on a shared box can be another tenant's. The TP link does not exist at TP=1, and says so."""
    node = Node(LIVE)
    node.tick()
    proxy = Proxy(node.cfg, node.hub, cluster_switch=None)
    sat = proxy.status()["saturation"]
    assert [(b["box"], b["rank"], b["role"]) for b in sat["boxes"]] == [("node-a", 0, "head")]
    assert sat["boxes"][0]["sm_pct"] is None and sat["verdict"] == "not observed"
    assert sat["network"]["rate"] is None and sat["network"]["why"] == topology.NO_LINK
    assert sat["sm_fill_gap_pts"] is None
    monkeypatch.setattr(topology, "serving_card", lambda cfg, hardware, box: {"id": 1, "use_pct": 96})
    sat = proxy.status()["saturation"]
    assert sat["boxes"][0]["sm_pct"] == 96.0
