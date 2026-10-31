"""A NODE THE LB REQUESTS AND ITS HOST APPLIES (fleet control `local`) — the LB's half, end to end: the same loop and the
same cluster view as the pair and the observed node, one switch, and two intent-shaped files (vllm_lb/node_control.py):
the operator's WISH (the node's switch, `Config.cluster_state_path`) and what was RELEASED to the host's applier
(<state>/node/intent.json). The host is a report file written by hand here; the node answers from a table.

  * POST /lb/cluster validates with the fleet's own validator (fleet.validate_launch) and stores the wish — or refuses
    in its words, with nothing written; and refuses while no applier is listening, saying so;
  * the loop releases the wish once the node has finished what it serves (`finishing` until then; `force` at once);
  * the view says what the applier did, the start's progress read off the node's log by the pair's own reader, and
    what the node is expected to run — the boot default, moved by the last intent the applier applied — checked
    against what the node says (cluster.declared), and its seats are the router's limit;
  * the pair, and a node the LB only observes, are untouched: the pair's view gains `applier: null`.
"""
from __future__ import annotations

import http.client
import json

import pytest

from tests.fakes import FakeProber
from tests.test_cluster_one_host import DOWN, LIVE, NODE, SERVED, Clock, scrape
from vllm_lb import kv_dtype
from vllm_lb import node_control as nc
from vllm_lb import topology
from vllm_lb.__main__ import build_cluster_loop
from vllm_lb.catalog import ModelCatalog
from vllm_lb.cluster import ClusterHost
from vllm_lb.config import Config
from vllm_lb.control import Controller
from vllm_lb.fleet import parse_fleet
from vllm_lb.hub import Hub
from vllm_lb.perflog import PerfRecorder
from vllm_lb.proxy import Proxy

IMAGE = "vllm/vllm-openai@sha256:" + "2" * 64
#: The two checkpoints of the catalog's 27B row the panel may select, the twin the boot default.
TWIN, ORIGINAL = "qwen3.8-27b-nvidia-nvfp4@uncensored", "qwen3.8-27b-nvidia-nvfp4"
FLEET = f"""
panel:
  style: vibe
boxes:
  node-a:
    hostname: node-a-host
    label: node-a-gpu
    platform: linux/amd64
    node_limits: {{memory_gib: 24, cpus: 16, pids: 4096, nice: 10}}
    cards:
      - {{index: 0, type: rtx3090, serves: false}}
      - {{index: 1, type: rtx5090, serves: true}}
clusters:
  node:
    control: local
    model: {TWIN}
    url: {NODE}
    launch:
      image: {IMAGE}
      max_model_len: 131072
      max_num_seqs: 4
      gpu_memory_utilization: 0.9
      reasoning_parser: qwen3
      tool_call_parser: qwen3_coder
      choices: [{TWIN}, {ORIGINAL}]
    ranks:
      - {{box: node-a}}
backends: {{}}
"""
#: vLLM's own start, as its log prints it (the pair's reader, cluster_startup.read_startup, parses these words).
START_LOG = ("INFO 10-11 20:00:01 [core.py:73] Initializing a V1 LLM engine with config: model='/models/M'\n"
             "Loading safetensors checkpoint shards:  50% Completed | 2/4 [00:03<00:03,  1.50s/it]\n")


class Local:
    """A standalone server's LB with a local node, its host's report written by `beat`. Its performance and boot logs
    are under `tmp_path`/perf; `record` also wires the performance recorder the probe phase feeds (`__main__` does), and
    a second Local over the same `tmp_path` and `clock` is the same LB restarted."""

    def __init__(self, tmp_path, answer: dict, *, listening: bool = True, fleet: str = FLEET, record: bool = False,
                 clock: Clock | None = None) -> None:
        self.perf_dir = str(tmp_path / "perf")
        self.cfg = Config(cluster_hosts=(ClusterHost("node-a-gpu", "node-a"),), cluster_url=NODE,
                          cluster_control="local", fleet=parse_fleet(fleet, "fleet.yml"), probe_interval_s=0,
                          cluster_state_path=str(tmp_path / "cluster.json"), listen_host="127.0.0.1", listen_port=0,
                          perf_dir=self.perf_dir)
        self.dir = tmp_path / nc.NODE_DIR
        self.dir.mkdir(exist_ok=True)
        self.clock = clock or Clock()
        self.hub = Hub(self.cfg, clock=self.clock)
        self.loop = build_cluster_loop(self.cfg, self.hub, None, None, catalog=ModelCatalog.load())
        self.loop.clock = self.clock
        self.prober = FakeProber({NODE: answer})
        self.recorder = PerfRecorder(self.perf_dir) if record else None
        self.ctl = Controller(self.cfg, self.hub, None, prober=self.prober, clock=self.clock, cluster_loop=self.loop,
                              recorder=self.recorder)
        if listening:
            self.beat()

    def beat(self, applied=None, applying=None, container=None, log=None, age=0.0) -> None:
        nc.write_report(str(self.dir), nc.report_doc(heartbeat=self.clock.t - age, host="node-a-host",
                                                     applied=applied, applying=applying, container=container,
                                                     log=log))

    def tick(self, seconds: float = 2.0) -> dict:
        self.ctl.tick()
        self.clock.t += seconds
        return self.hub.snapshot().extra["cluster"]

    def released(self) -> dict | None:
        return nc.IntentStore(str(self.dir / nc.INTENT)).read()

    def applied(self, **over) -> dict:
        i = self.released()
        return {**i, "done_at": self.clock.t, "exit": 0, "stderr": "", "refused": "", **over}


@pytest.fixture
def served():
    started: list[Proxy] = []

    def make(node: Local) -> Proxy:
        proxy = Proxy(node.cfg, node.hub, cluster_switch=None)
        proxy.start()
        started.append(proxy)
        return proxy

    yield make
    for proxy in started:
        proxy.stop()


def post(proxy: Proxy, body: dict) -> tuple[int, dict]:
    conn = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=10)
    conn.request("POST", "/lb/cluster", body=json.dumps(body), headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    return resp.status, json.loads(resp.read() or b"{}")


# ── the topology ───────────────────────────────────────────────────────────────────────────────


def test_a_local_node_is_one_host_the_lb_may_start_and_stop(tmp_path):
    node = Local(tmp_path, LIVE)
    assert topology.problem(node.cfg) == ""
    view = node.tick()
    assert view["topology"] == {"tp": 1, "boxes": ["node-a"], "control": "local", "why": "",
                                "steps": ["load", "warm up", "serve"]}


def test_with_nothing_asked_it_runs_its_boot_default_and_the_view_says_who_listens(tmp_path):
    node = Local(tmp_path, LIVE)
    view = node.tick()
    assert (view["state"], view["held"], view["max_depth"]) == ("running", True, 4)
    assert view["desired"] == {"mode": "cluster", "id": None, "force": False, "model": TWIN, "model_choice":
                               ORIGINAL, "uncensored": True, "max_model_len": 131072,
                               "max_num_seqs": 4, "gpu_memory_utilization": 0.9, "max_num_batched_tokens": None,
                               "kv_cache_memory": None, "kv_offload_ram_gib": None, "kv_cache_dtype": None}
    assert view["declared"]["launch"]["max_num_seqs"] == 4 and view["declared"]["source"] == "fleet.yml"
    assert "max_num_batched_tokens" not in view["declared"]["launch"], "vLLM's own default: nothing is declared"
    assert view["declared"]["launch"]["checkpoint"] == "Qwen3.8-27B-NVFP4_uncensored", "the catalog's folder"
    assert view["applier"]["listening"] is True and view["applier"]["host"] == "node-a-host"
    assert view["limits"] == {"max_model_len": 262144, "max_num_seqs": [1, 48]}, "the model's own catalog limits"
    assert [(c["ref"], c["uncensored"], c["checkpoint"], c["overhead_chunk"]) for c in view["choices"]] == [
        (TWIN, True, "Qwen3.8-27B-NVFP4_uncensored", 2048), (ORIGINAL, False, "Qwen3.8-27B-NVFP4", 2048)]
    # The largest pool the fit takes pinned: the whole card less weights and engine — the catalog's calibration, which
    # reproduces vLLM's own "fully utilize" figure at the launch it was taken at (8,633,930,752 B, 4 seats).
    assert [c["kv_pin_max"] for c in view["choices"]] == [int((32 - 20.4 - 3.559) * 2**30)] * 2
    assert view["choices"][0]["kv_pin_max"] >= 8_501_810_176, "the live 16-seat figure fits under it"


def test_with_no_stored_wish_the_desired_state_is_the_fleets_run_at_boot(tmp_path):
    """Live 2026-10-17 (the box-c gui): a node declared `run_at_boot: false` and never started read `mode: cluster`,
    so the panel offered "Stop server" — and holding it posted a Stop. With no stored wish the desired state FOLLOWS
    the fleet's run_at_boot; a stored wish says it itself."""
    (tmp_path / "off").mkdir()
    (tmp_path / "on").mkdir()
    off = Local(tmp_path / "off", LIVE, fleet=FLEET.replace("    control: local\n", "    control: local\n    run_at_boot: false\n"))
    assert off.tick()["desired"]["mode"] == "off", "declared, not run at boot, never asked: off"
    on = Local(tmp_path / "on", LIVE, fleet=FLEET.replace("    control: local\n", "    control: local\n    run_at_boot: true\n"))
    assert on.tick()["desired"]["mode"] == "cluster", "run at boot, never asked: on"


def test_a_local_node_serving_the_27b_publishes_its_thinking_levels(tmp_path):
    """The chat's effort menu and the panel's picker read `loaded_model.efforts` — the standalone path must carry
    them as the pair's does (catalog.json `efforts`, the 27B template's off / low / medium / xhigh)."""
    view = Local(tmp_path, {**LIVE, "served_models": ("Qwen/Qwen3.8-27B",)}).tick()
    assert view["loaded_model"]["spec"] == ORIGINAL
    assert view["loaded_model"]["efforts"] == ["xhigh", "medium", "low", "off"]


# ── the switch ─────────────────────────────────────────────────────────────────────────────────


def test_a_change_is_validated_by_the_fleet_stored_as_the_wish_and_released_to_an_idle_node(tmp_path, served):
    node = Local(tmp_path, LIVE)
    node.tick()
    status, out = post(served(node), {"enabled": True, "max_num_seqs": 2, "max_model_len": 65536})
    assert status == 200, out
    wish = nc.IntentStore(node.cfg.cluster_state_path).read()
    assert (wish["state"], wish["launch"], wish["force"]) == ("up", {"max_num_seqs": 2, "max_model_len": 65536},
                                                              False)
    view = node.tick()
    released = node.released()
    assert (released["id"], released["state"], released["launch"]) == (1, "up", wish["launch"])
    assert (view["state"], view["held"]) == ("starting", False), "released: not routed to until applied"
    assert view["desired"]["max_num_seqs"] == 2


def get(proxy: Proxy, path: str) -> tuple[int, dict]:
    conn = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=10)
    conn.request("GET", path)
    resp = conn.getresponse()
    return resp.status, json.loads(resp.read() or b"{}")


def fit_of(ref: str, util: float, length: int) -> dict:
    """The fleet's own numbers for a launch (fleet.launch_fit), as the view and the estimate carry them."""
    from vllm_lb.fleet import launch_fit
    cluster = parse_fleet(FLEET, "fleet.yml").clusters["node"]
    model = next(m for m in cluster.choices if m.ref == ref)
    f = launch_fit(model, cluster.serving_cards[0].type, util, length)
    return {"weights_gib": f.weights_gib, "overhead_gib": f.overhead_gib, "kv_gib": f.kv_gib, "why": f.why,
            "budget_gib": f.budget_gib, "overhead_chunk": f.overhead_chunk, "note": f.note, "pool_gib": None,
            "ram_gib": None}


def test_the_view_carries_the_fit_of_what_is_asked_in_the_fleets_numbers(tmp_path):
    view = Local(tmp_path, LIVE).tick()
    assert view["fit"] == {"fit": fit_of(TWIN, 0.9, 131072), "refused": ""}
    assert view["fit"]["fit"]["weights_gib"] == 20.4 and view["fit"]["fit"]["kv_gib"] > 0


def test_the_estimate_prices_a_selection_before_it_is_sent_and_says_a_refusal_verbatim(tmp_path, served):
    node = Local(tmp_path, LIVE)
    node.tick()
    proxy = served(node)
    status, out = get(proxy, f"/lb/cluster/estimate?model={ORIGINAL}&max_model_len=65536&gpu_memory_utilization=0.85")
    assert status == 200 and out == {"fit": fit_of(ORIGINAL, 0.85, 65536), "refused": ""}, out
    status, out = get(proxy, "/lb/cluster/estimate?gpu_memory_utilization=0.5")
    assert status == 200 and "GiB of weights" in out["refused"] and out["fit"]["budget_gib"] == 16.0, out
    status, out = get(proxy, "/lb/cluster/estimate?max_num_seqs=two")
    assert status == 400, out


def test_a_stored_wish_the_fleet_now_refuses_is_shown_as_stored_with_the_refusal_so_it_can_be_changed(tmp_path):
    """site-b 2026-10-11: intent #3 asked 262,144, taken then; the fit check that came after refuses it. The panel must
    still show what is asked — the operator recovers by changing it there — and say why it is refused."""
    node = Local(tmp_path, LIVE)
    nc.IntentStore(node.cfg.cluster_state_path).write("up", {"max_model_len": 262144}, now=node.clock.t)
    view = node.tick()
    assert view["desired"]["max_model_len"] == 262144 and view["desired"]["model"] == TWIN
    assert "GiB" in view["fit"]["refused"] and view["store_error"] == ""


def test_what_the_host_applied_is_declared_as_applied_even_where_the_fleet_now_refuses_it(tmp_path):
    """The node runs what its host APPLIED: a launch the fleet refuses since is still that — never the boot default
    standing in for it — and the node is checked against it."""
    node = Local(tmp_path, LIVE)
    nc.IntentStore(node.cfg.cluster_state_path).write("up", {"max_model_len": 262144}, now=node.clock.t)
    node.tick()
    node.beat(applied=node.applied(restarts_at_apply=0))
    view = node.tick()
    assert view["declared"]["source"] == "fleet.yml + intent #1"
    assert view["declared"]["launch"]["max_model_len"] == 262144
    assert view["declared"]["checks"]["max_model_len"] == {"fleet": 262144, "node": 131072, "verdict": "mismatch"}
    assert view["store_error"] == ""


def test_the_operator_selects_the_model_among_the_fleets_choices(tmp_path, served):
    node = Local(tmp_path, LIVE)
    node.tick()
    status, out = post(served(node), {"enabled": True, "model": ORIGINAL})
    assert status == 200, out
    view = node.tick()
    assert node.released()["launch"] == {"model": ORIGINAL}
    assert (view["desired"]["model"], view["desired"]["uncensored"]) == (ORIGINAL, False)


@pytest.mark.parametrize("body, says", [
    ({"enabled": True, "max_num_seqs": 999}, "max_num_seqs must be a whole number in"),
    ({"enabled": True, "gpu_memory_utilization": 0}, "gpu_memory_utilization must be a fraction"),
    # THE FIT, in the fleet's own numbers: the 27B's 20.4 GiB of weights do not fit half the card.
    ({"enabled": True, "gpu_memory_utilization": 0.5}, "GiB of weights > 0.50 × 32 GiB"),
    ({"enabled": True, "model": "qwen3.8-flash-next-hibrid48"}, "is not among the fleet's choices"),
    ({"enabled": True, "image": "evil@sha256:" + "0" * 64}, "image"),
    ({"enabled": True, "tool_call_parser": "hermes"}, "tool_call_parser"),
    ({"enabled": True, "served_name": "x"}, "served_name follows the model"),
    ({"enabled": True, "force": "yes"}, "force"),
])
def test_a_change_the_fleet_refuses_is_refused_in_its_words_and_nothing_is_written(tmp_path, served, body, says):
    node = Local(tmp_path, LIVE)
    node.tick()
    status, out = post(served(node), body)
    assert status == 400 and says in out["error"]["message"], out
    assert nc.IntentStore(node.cfg.cluster_state_path).read() is None and node.released() is None


def test_nothing_is_asked_while_no_applier_listens(tmp_path, served):
    node = Local(tmp_path, LIVE, listening=False)
    node.beat(age=nc.APPLIER_STALE_S + 30)
    node.tick()
    status, out = post(served(node), {"enabled": False})
    assert status == 503 and out["error"]["type"] == "node_applier_not_listening"
    assert "no applier listening" in out["error"]["message"]
    assert nc.IntentStore(node.cfg.cluster_state_path).read() is None


def test_stop_is_a_down_wish_keeping_the_settings(tmp_path, served):
    node = Local(tmp_path, LIVE)
    node.tick()
    proxy = served(node)
    assert post(proxy, {"enabled": True, "max_num_seqs": 2})[0] == 200
    assert post(proxy, {"enabled": False})[0] == 200
    wish = nc.IntentStore(node.cfg.cluster_state_path).read()
    assert (wish["state"], wish["launch"]) == ("down", {"max_num_seqs": 2})
    view = node.tick()
    assert node.released()["state"] == "down" and view["state"] == "unloading"


# ── releasing: a relaunch waits for the node to finish ───────────────────────────────────────────


def test_a_relaunch_of_a_serving_node_waits_until_it_has_finished_unless_forced(tmp_path, served):
    busy = dict(LIVE, running=2, scrape=scrape(running=2))
    node = Local(tmp_path, busy)
    node.tick()
    proxy = served(node)
    assert post(proxy, {"enabled": True, "max_num_seqs": 2})[0] == 200
    view = node.tick()
    assert (view["state"], view["held"], node.released()) == ("finishing", False, None)
    node.prober.table[NODE] = LIVE
    node.tick()
    node.tick()
    assert node.released()["launch"] == {"max_num_seqs": 2}, "drained: released"
    assert post(proxy, {"enabled": True, "max_num_seqs": 3, "force": True})[0] == 200
    node.prober.table[NODE] = busy
    node.tick()
    node.tick()
    assert node.released()["launch"] == {"max_num_seqs": 3} and node.released()["force"] is True


# ── what the applier did, and what the node runs ───────────────────────────────────────────────


def test_applied_and_answering_runs_the_moved_launch_checked_against_the_node(tmp_path, served):
    node = Local(tmp_path, LIVE)
    node.tick()
    post(served(node), {"enabled": True, "max_num_seqs": 2})
    node.tick()
    node.beat(applied=node.applied())
    view = node.tick()
    assert (view["state"], view["held"], view["max_depth"]) == ("running", True, 2), "its seats: the applied launch"
    assert view["declared"]["launch"]["max_num_seqs"] == 2
    assert view["declared"]["source"] == "fleet.yml + intent #1"
    assert view["declared"]["checks"]["max_model_len"]["verdict"] == "agrees"
    assert view["effective"]["max_num_seqs"] == 2


def test_the_applied_prefill_chunk_is_what_the_node_is_declared_to_run(tmp_path, served):
    node = Local(tmp_path, LIVE)
    node.tick()
    post(served(node), {"enabled": True, "max_num_batched_tokens": 1024})
    node.tick()
    node.beat(applied=node.applied())
    view = node.tick()
    assert view["declared"]["launch"]["max_num_batched_tokens"] == 1024
    assert view["declared"]["checks"]["max_num_batched_tokens"]["verdict"] == "not exposed by the node"


# ── the prefill chunk and the card's share, priced as they are picked ─────────────────────────────


def test_the_prefill_chunk_is_set_from_the_panel_priced_before_it_is_sent_and_given_back_to_vllm(tmp_path, served):
    node = Local(tmp_path, LIVE)
    view = node.tick()
    assert view["desired"]["max_num_batched_tokens"] is None, "nothing asked: vLLM's own default"
    proxy = served(node)
    status, out = get(proxy, "/lb/cluster/estimate?max_num_batched_tokens=1024")
    assert status == 200 and out["refused"] == "", out
    assert (out["fit"]["overhead_gib"], out["fit"]["overhead_chunk"]) == (3.559, 2048)
    assert "not credited" in out["fit"]["note"], out["fit"]
    status, out = get(proxy, "/lb/cluster/estimate?max_num_batched_tokens=4096")
    assert status == 200 and "4,096" in out["refused"] and "2,048" in out["refused"], out
    status, out = get(proxy, "/lb/cluster/estimate?max_num_batched_tokens=lots")
    assert status == 400 and "max_num_batched_tokens" in out["error"]["message"], out
    assert post(proxy, {"enabled": True, "max_num_batched_tokens": 1024})[0] == 200
    assert nc.IntentStore(node.cfg.cluster_state_path).read()["launch"] == {"max_num_batched_tokens": 1024}
    assert node.tick()["desired"]["max_num_batched_tokens"] == 1024
    # BLANK is vLLM's own default (vLLM reads an empty value so too): priced as the measured launch, and a switch
    # with null gives the chunk back to vLLM — stored as null, so the boot default's own chunk cannot return.
    status, out = get(proxy, "/lb/cluster/estimate?max_num_batched_tokens=")
    assert status == 200 and (out["refused"], out["fit"]["note"]) == ("", ""), out
    assert post(proxy, {"enabled": True, "max_num_batched_tokens": None})[0] == 200
    assert nc.IntentStore(node.cfg.cluster_state_path).read()["launch"] == {"max_num_batched_tokens": None}
    assert node.tick()["desired"]["max_num_batched_tokens"] is None
    status, out = post(proxy, {"enabled": True, "max_num_batched_tokens": 2})
    assert status == 400 and "under max_num_seqs 4" in out["error"]["message"], out


@pytest.mark.parametrize("util, refused", [(0.97, ""), (0.98, ""),
                                           (0.99, "GPU memory 0.99 × 31.36 GiB = 31.05 GiB > 30.85 GiB free"),
                                           (1.0, "GPU memory 1.00 × 31.36 GiB = 31.36 GiB > 30.85 GiB free")])
def test_the_card_share_offered_097_to_1_is_priced_on_the_card_and_refused_where_vllm_cannot_start(tmp_path, served,
                                                                                                  util, refused):
    """Operator, 2026-10-11: 0.97, 0.98, 0.99 and 1.0 offered, each priced on the card's 32 GiB (its type's memory_gb).
    site-b 2026-10-11 23:21Z: 0.99 was taken here and vLLM refused it at start ("Free memory on device cuda:0
    (30.85/31.36 GiB) on startup is less than desired GPU memory utilization (0.99, 31.04 GiB)"), restart-looping the
    node — so the estimate now refuses it in those numbers before anything is sent (fleet.py `_start_check`)."""
    node = Local(tmp_path, LIVE)
    node.tick()
    proxy = served(node)
    status, out = get(proxy, f"/lb/cluster/estimate?gpu_memory_utilization={util}")
    assert status == 200 and (refused in out["refused"] if refused else out["refused"] == ""), out
    assert out["fit"]["budget_gib"] == pytest.approx(util * 32), "still priced: the panel shows both"
    if refused:
        assert "vLLM refuses before loading (v1/worker/utils.py:761" in out["refused"], out
        status, out = post(proxy, {"enabled": True, "gpu_memory_utilization": util})
        assert status == 400 and refused in out["error"]["message"], out
        assert nc.IntentStore(node.cfg.cluster_state_path).read() is None, "nothing released to the host"



# ── the KV pool pinned and the RAM tier, priced as they are picked ────────────────────────────────

#: The same node on a box whose fleet file bounds a KV tier in its RAM.
RAMMED = FLEET.replace("    platform: linux/amd64\n", "    platform: linux/amd64\n    kv_offload_ram_max_gib: 32\n")
#: The relaunch the coordinator posts (2026-10-12): the card's whole KV ("0.99") and a 32 GiB RAM tier.
RELAUNCH = {"max_model_len": 131072, "max_num_seqs": 16, "gpu_memory_utilization": 0.95,
            "kv_cache_memory": 8_501_810_176, "kv_offload_ram_gib": 32}


def test_the_view_says_how_much_ram_a_kv_tier_may_take_or_why_none(tmp_path):
    assert Local(tmp_path, LIVE, fleet=RAMMED).tick()["ram_tier"] == {"max_gib": 32, "refused": ""}
    other = tmp_path / "other"
    other.mkdir()
    tier = Local(other, LIVE).tick()["ram_tier"]
    assert tier["max_gib"] is None and "states no kv_offload_ram_max_gib" in tier["refused"], tier


def test_the_pinned_pool_and_the_ram_tier_are_priced_stored_and_given_back(tmp_path, served):
    node = Local(tmp_path, LIVE, fleet=RAMMED)
    view = node.tick()
    assert (view["desired"]["kv_cache_memory"], view["desired"]["kv_offload_ram_gib"]) == (None, None)
    assert (view["fit"]["fit"]["pool_gib"], view["fit"]["fit"]["ram_gib"]) == (None, None)
    proxy = served(node)
    q = "&".join(f"{k}={v}" for k, v in RELAUNCH.items())
    status, out = get(proxy, f"/lb/cluster/estimate?{q}")
    assert status == 200 and out["refused"] == "", out
    assert out["fit"]["ram_gib"] == 32 and out["fit"]["budget_gib"] == 32, "the pinned pool: the whole card"
    assert out["fit"]["pool_gib"] == pytest.approx(8_501_810_176 / 2**30)
    status, out = get(proxy, "/lb/cluster/estimate?kv_offload_ram_gib=48")
    assert status == 200 and "kv_offload_ram_max_gib 32" in out["refused"], out
    for key in ("kv_offload_ram_gib", "kv_cache_memory"):
        status, out = get(proxy, f"/lb/cluster/estimate?{key}=lots")
        assert status == 400 and key in out["error"]["message"], out
    assert post(proxy, {"enabled": True, **RELAUNCH})[0] == 200
    assert nc.IntentStore(node.cfg.cluster_state_path).read()["launch"] == RELAUNCH
    view = node.tick()
    assert (view["desired"]["kv_cache_memory"], view["desired"]["kv_offload_ram_gib"]) == (8_501_810_176, 32)
    # BLANK is vLLM's own: no pin (the pool from the fraction), no tier — and null gives both back.
    status, out = get(proxy, "/lb/cluster/estimate?kv_offload_ram_gib=&kv_cache_memory=")
    assert status == 200 and (out["fit"]["pool_gib"], out["fit"]["ram_gib"]) == (None, None), out
    assert post(proxy, {"enabled": True, "kv_cache_memory": None, "kv_offload_ram_gib": None})[0] == 200
    stored = nc.IntentStore(node.cfg.cluster_state_path).read()["launch"]
    assert (stored["kv_cache_memory"], stored["kv_offload_ram_gib"]) == (None, None)
    status, out = post(proxy, {"enabled": True, "kv_offload_ram_gib": 64})
    assert status == 400 and "kv_offload_ram_max_gib 32" in out["error"]["message"], out


def test_the_applied_ram_tier_and_pin_are_what_the_node_is_declared_to_run(tmp_path, served):
    node = Local(tmp_path, LIVE, fleet=RAMMED)
    node.tick()
    post(served(node), {"enabled": True, **RELAUNCH})
    node.tick()
    node.beat(applied=node.applied())
    view = node.tick()
    assert (view["declared"]["launch"]["kv_cache_memory"], view["declared"]["launch"]["kv_offload_ram_gib"]) == \
        (8_501_810_176, 32)

def test_a_start_shows_its_progress_read_off_the_nodes_log(tmp_path, served):
    node = Local(tmp_path, DOWN)
    node.tick()
    post(served(node), {"enabled": True})
    node.tick()
    node.beat(applied=node.applied(), container={"status": "running", "exit_code": 0, "restarts": 0},
              log=START_LOG)
    view = node.tick()
    assert view["state"] == "starting"
    assert view["loading"]["passes"][0]["completed"] == 2 and view["loading"]["passes"][0]["total"] == 4


def test_a_failure_is_shown_in_the_applier_s_own_words(tmp_path, served):
    node = Local(tmp_path, DOWN)
    node.tick()
    post(served(node), {"enabled": True})
    node.tick()
    node.beat(applied=node.applied(exit=1, stderr="Error response from daemon: could not select device driver"),
              container=None, log=None)
    view = node.tick()
    assert view["state"] == "failed" and "could not select device driver" in view["detail"]
    assert view["applier"]["applied"]["exit"] == 1


# ── the others are untouched ────────────────────────────────────────────────────────────────────


def test_the_pair_and_an_observed_node_carry_no_applier(tmp_path):
    from tests.test_cluster_one_host import Node
    assert Node(LIVE).tick()["applier"] is None
    from vllm_lb.cluster import parse_hosts
    pair = Config(cluster_name="pair", cluster_hosts=parse_hosts("p-gpu=pair-a=op@192.0.2.10=198.51.100.1,"
                                            "q-gpu=pair-b=op@192.0.2.11=198.51.100.2"),
                  cluster_url="http://192.0.2.10:8100")
    assert Hub(pair).cluster_state()["applier"] is None


# ── the KV TYPE (kv_cache_dtype) on the panel: what the node's image runs on its card ─────────────────────────────────
# Operator 2026-10-22, at the KV pool row: "can't we add nvfp4 and fp8 setting for KV here?". The pair publishes its
# choices per model (cluster_loop, `ModelSpec.kv_cache_dtype_choices`); a node the LB requests publishes the same shape,
# per fleet choice (the ref an intent sends), from the ONE library the fleet's validator asks (fleet._kv_dtype).

#: The 5090 node's image, as kv_cache_dtypes.json records it (vLLM 2eb491e: nvfp4 on sm_8x/9x/10x/12x).
FIVE = next(i for i in kv_dtype.library().images if i.startswith("vllm/vllm-openai@sha256:514cd427"))


def test_the_view_publishes_the_kv_types_the_nodes_image_runs_on_its_card_for_each_choice(tmp_path):
    view = Local(tmp_path, LIVE, fleet=FLEET.replace(IMAGE, FIVE)).tick()
    dtypes = view["kv_cache_dtypes"]
    assert set(dtypes) == {TWIN, ORIGINAL}, "keyed by the ref the panel selects"
    assert dtypes[TWIN] == kv_dtype.library().choices(FIVE, ["sm_120"]), "the library's words, the serving card's arch"
    by = {c["dtype"]: c for c in dtypes[TWIN]}
    assert by["nvfp4"]["refused"] == "" and "flashinfer.py" in by["nvfp4"]["source"], by["nvfp4"]
    assert "does not accept kv_cache_dtype fp4_mx_block16" in by["fp4_mx_block16"]["refused"]


def test_the_rocm_node_publishes_nvfp4_refused_in_the_images_words():
    from tests.test_gui_node import GUI
    cluster = parse_fleet(GUI, "fleet.yml").clusters["gui"]
    by = {c["dtype"]: c for c in topology.local_kv_cache_dtypes(cluster)["mai-ui-2b"]}
    assert "this ROCm vLLM has no FP4 KV cache" in by["nvfp4"]["refused"], by["nvfp4"]
    assert by["bfloat16"]["refused"] == ""


def test_an_image_the_library_has_no_record_of_offers_every_kv_type_refused(tmp_path):
    view = Local(tmp_path, LIVE).tick()
    assert all("no record of which KV cache dtypes" in c["refused"] for c in view["kv_cache_dtypes"][TWIN])


def test_the_estimate_checks_the_kv_type_asked(tmp_path, served):
    node = Local(tmp_path, LIVE)
    node.tick()
    proxy = served(node)
    status, out = get(proxy, "/lb/cluster/estimate?kv_cache_dtype=nvfp4")
    assert status == 200 and "no record of which KV cache dtypes" in out["refused"], out
    status, out = get(proxy, "/lb/cluster/estimate?kv_cache_dtype=")
    assert status == 200 and out["refused"] == "", "blank is vLLM's own"
