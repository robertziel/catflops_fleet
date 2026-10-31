"""WHAT THE FLEET DECLARES A NODE RUNS, beside what the node says it runs (worklog/standalone-cluster-panel.md).

A node this LB only observes (control `none`) cannot say which catalog model it is, nor its seats, its parsers or its
image: vLLM publishes none of them. Its fleet file declares them — `clusters.<name>.model` (`<catalog id>[@<variant>]`,
validated against the catalog by the loader) and a `launch:` block the server file renders into the node's command
line (vllm_lb/fleet.py `ClusterModel`, `NodeLaunch`). The LB publishes that declaration as `cluster.declared` and
CROSS-CHECKS the keys the node also reports — its window (/v1/models), its GPU memory fraction (vllm:cache_config_info)
and its served name — key by key:

  agrees                    both say the same;
  mismatch                  they disagree: both values are published, neither is chosen (`effective` keeps the node's
                            own word, which is what /v1/models and /metrics say; the page shows the disagreement);
  not exposed by the node   the node does not say (always so for seats, parsers and image).

`effective` gains what only the declaration can say — which catalog row the node is and its seats — so the card names
the model and the pool line counts the seats; the pair has no declaration (`declared` null: its Start names its model).
"""
from __future__ import annotations

from tests.test_cluster_one_host import DOWN, LIVE, NODE, SERVED, Node, node_cfg, scrape
from vllm_lb import topology
from vllm_lb.fleet import parse_fleet
from vllm_lb.hub import Hub

MODEL = "qwen3.8-27b-nvidia-nvfp4"
IMAGE = "vllm/vllm-openai@sha256:" + "0123456789abcdef" * 4
#: The node's `launch:` block as its fleet file declares it — every value what the node in LIVE runs with.
LAUNCH = {"image": IMAGE, "served_name": SERVED, "max_model_len": 131072, "max_num_seqs": 4,
          "gpu_memory_utilization": 0.9, "reasoning_parser": "qwen3", "tool_call_parser": "qwen3_coder"}
#: LIVE, reporting what vLLM's cache_config_info says.
REPORTS = dict(LIVE, model_roots=("/models/Model-A-NVFP4",),
               scrape=scrape(engine="vllm", gpu_memory_utilization=0.9, kv_cache_dtype="fp8_e4m3",
                             prefix_caching=True))


def declared_fleet(model: str = f"{MODEL}@uncensored", launch: bool = True, **over):
    """The node's fleet file (its box `node-a` — the harness's HOST — a 3090 beside the 5090 it serves on), declaring
    `model` and, with `launch`, LAUNCH with `over` replacing some of it."""
    block = "".join(f"\n      {k}: {v!r}" if isinstance(v, str) else f"\n      {k}: {v}"
                    for k, v in {**LAUNCH, **over}.items())
    return parse_fleet(f"""
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
    control: none
    model: {model}
    url: {NODE}
    ranks:
      - {{box: node-a}}{f'''
    launch:{block}''' if launch else ''}
backends: {{}}
""", "fleet.yml")


def checks(**over) -> dict:
    """Every declared key's check as the node in REPORTS answers it, `over` replacing some."""
    out = {"image": {"fleet": IMAGE, "node": None, "verdict": "not exposed by the node"},
           "served_name": {"fleet": SERVED, "node": SERVED, "verdict": "agrees"},
           "max_model_len": {"fleet": 131072, "node": 131072, "verdict": "agrees"},
           "max_num_seqs": {"fleet": 4, "node": None, "verdict": "not exposed by the node"},
           "gpu_memory_utilization": {"fleet": 0.9, "node": 0.9, "verdict": "agrees"},
           "reasoning_parser": {"fleet": "qwen3", "node": None, "verdict": "not exposed by the node"},
           "tool_call_parser": {"fleet": "qwen3_coder", "node": None, "verdict": "not exposed by the node"},
           # the server it runs, its row's node_engine: the node's /metrics names its engine family, not this
           "engine": {"fleet": "vllm", "node": None, "verdict": "not exposed by the node"}}
    out.update(over)
    return out


def test_the_fleet_declares_which_catalog_model_the_node_is_and_what_it_runs_with():
    view = Node(REPORTS, fleet=declared_fleet()).tick()
    assert view["declared"] == {"source": "fleet.yml", "cluster": "node", "model": f"{MODEL}@uncensored",
                                "model_choice": MODEL, "uncensored": True, "launch": {**LAUNCH, "engine": "vllm"},
                                "checks": checks()}
    eff = view["effective"]
    # What only the declaration can say: which catalog row the node is, and its seats.
    assert (eff["model_choice"], eff["uncensored"], eff["max_num_seqs"]) == (MODEL, True, 4)
    # What the node says is still the node's.
    assert (eff["max_model_len"], eff["gpu_memory_utilization"], eff["served_name"]) == (131072, 0.9, SERVED)
    assert (eff["checkpoint"], eff["engine"], eff["kv_cache_dtype"]) == ("Model-A-NVFP4", "vllm", "fp8_e4m3")


def test_the_original_checkpoint_and_a_fleet_with_no_launch_block():
    view = Node(REPORTS, fleet=declared_fleet(MODEL, launch=False)).tick()
    assert (view["declared"]["model"], view["declared"]["uncensored"]) == (MODEL, False)
    assert view["declared"]["launch"] == {} and view["declared"]["checks"] == {}
    assert view["effective"]["max_num_seqs"] is None, "no seats declared: none claimed"


def test_a_disagreement_is_published_as_one_and_never_resolved_either_way():
    fleet = declared_fleet(max_model_len=262144, gpu_memory_utilization=0.85, served_name="model-b")
    view = Node(REPORTS, fleet=fleet).tick()
    assert view["declared"]["checks"] == checks(
        served_name={"fleet": "model-b", "node": SERVED, "verdict": "mismatch"},
        max_model_len={"fleet": 262144, "node": 131072, "verdict": "mismatch"},
        gpu_memory_utilization={"fleet": 0.85, "node": 0.9, "verdict": "mismatch"})
    eff = view["effective"]
    assert (eff["max_model_len"], eff["gpu_memory_utilization"], eff["served_name"]) == (131072, 0.9, SERVED), \
        "`effective` is what the node runs, in its own words; the declaration is published beside it"


def test_a_served_name_among_several_the_node_serves_agrees():
    answer = dict(REPORTS, served_models=("model-alias", SERVED))
    got = Node(answer, fleet=declared_fleet()).tick()["declared"]["checks"]["served_name"]
    assert got == {"fleet": SERVED, "node": SERVED, "verdict": "agrees"}


def test_a_checked_key_the_node_did_not_report_is_not_exposed_never_assumed_to_agree():
    answer = dict(REPORTS, scrape=scrape(engine="vllm"))       # an exposition without cache_config_info
    got = Node(answer, fleet=declared_fleet()).tick()["declared"]["checks"]["gpu_memory_utilization"]
    assert got == {"fleet": 0.9, "node": None, "verdict": "not exposed by the node"}


def test_while_the_node_does_not_answer_the_declaration_stands_and_nothing_is_checked():
    view = Node(DOWN, fleet=declared_fleet()).tick()
    assert view["effective"] is None
    assert view["declared"]["model_choice"] == MODEL and view["declared"]["checks"] is None


def test_the_pre_tick_view_carries_the_declaration_unchecked():
    early = Hub(node_cfg(fleet=declared_fleet())).cluster_state()
    assert early["declared"]["launch"] == {**LAUNCH, "engine": "vllm"} and early["declared"]["checks"] is None


def test_an_undeclared_node_and_the_pair_publish_no_declaration():
    """No fleet: `declared` null and `effective` exactly the node's own report (as before the field). The pair's model
    is its Start's: a fleet's ssh cluster declares nothing."""
    view = Node(REPORTS).tick()
    assert view["declared"] is None
    assert (view["effective"]["model_choice"], view["effective"]["max_num_seqs"]) == (None, None)
    from tests.fleet_files import gb10_pair
    from vllm_lb.cluster import parse_hosts
    from vllm_lb.config import Config
    pair = Config(cluster_hosts=parse_hosts("box-a-gpu=box-a=op@192.0.2.10=198.51.100.1,"
                                            "box-b-gpu=box-b=op@192.0.2.11=198.51.100.2"),
                  cluster_url="http://192.0.2.10:8100", fleet=gb10_pair())
    assert topology.declaration(pair) is None
    assert Hub(pair).cluster_state()["declared"] is None
