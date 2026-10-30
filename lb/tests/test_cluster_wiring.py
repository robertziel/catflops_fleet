"""The cluster loop as the LB process builds it: the tp-rank that ships with the LB, run over SSH,
told the two things only the LB's own configuration knows — the port it probes the head on and
the head's pair-link address the ranks meet at."""
from __future__ import annotations

import re
from pathlib import Path

import vllm_lb
from vllm_lb.__main__ import build_cluster_loop
from vllm_lb.cluster import DesiredStore, parse_hosts
from vllm_lb.config import Config
from vllm_lb.hub import Hub
from vllm_lb.leases import BrokerLeases

HOSTS = ("box-a-gpu=box-a=user@192.0.2.2=192.0.2.201,"
         "box-b-gpu=box-b=user@192.0.2.12=192.0.2.202")


def build(tmp_path, switch=True):
    # The fleet file the cluster's TOPOLOGY is read from (executors.py: the TP size, nodes, GPUs, link, memory family,
    # platform tp-rank is told) — the reference pair's shape.
    from tests.fleet_files import PAIR
    from vllm_lb.fleet import parse_fleet
    cfg = Config(vllm_boxes=(), cluster_name="pair", cluster_hosts=parse_hosts(HOSTS), fleet=parse_fleet(PAIR, "fleet.yml"),
                 cluster_url="http://192.0.2.2:8100", cluster_iface="bond0", cluster_max_num_batched_tokens=8192,
                 cluster_state_path=str(tmp_path / "cluster.json"),
                 cluster_ssh_key="/keys/id_ed25519", cluster_known_hosts="/keys/known_hosts")
    store = DesiredStore(cfg.cluster_state_path) if switch else None
    return build_cluster_loop(cfg, Hub(cfg), BrokerLeases(cfg), store)


def test_no_switch_builds_no_loop(tmp_path):
    assert build(tmp_path, switch=False) is None


def test_the_executor_runs_the_tp_rank_that_ships_with_the_lb(tmp_path):
    executor = build(tmp_path).executor
    shipped = Path(vllm_lb.__file__).resolve().parent.parent / "cluster" / "tp-rank"
    assert Path(executor.script_path).resolve() == shipped
    assert shipped.read_bytes().startswith(b"#!"), "nothing is installed on the GB10 boxes: the script is fed over SSH"


def test_the_ranks_are_told_the_topology_the_fleet_file_states(tmp_path):
    """tp-rank assumes no DGX Spark pair any more (phase 5): the LB tells it the pair's shape, from the fleet file."""
    from tests.test_tp_rank import GB10_PAIR_FACTS
    env = build(tmp_path).executor.env
    assert {k: env[k] for k in GB10_PAIR_FACTS} == GB10_PAIR_FACTS


def test_the_ranks_serve_where_the_lb_probes_and_meet_on_the_head_interconnect(tmp_path):
    env = build(tmp_path).executor.env
    assert env["CLUSTER_PORT"] == "8100", "the head must serve on the port LB_CLUSTER_URL probes"
    assert env["CLUSTER_MASTER_ADDR"] == "192.0.2.201", "rendezvous on the head's pair-link address"
    assert env["CLUSTER_IFACE"] == "bond0", "NCCL binds the fleet's interface, never a default of the script's"


#: The only tp-rank setting that is the BOX's and not a model's: where this box keeps vLLM's caches.
LAUNCH_SETTINGS = ("CLUSTER_CACHE_DIR",)
#: Everything a MODEL decides. These come from the ModelChoice's LaunchProfile (vllm_lb/catalog.py), which
#: `cluster_exec.launch_env` turns into settings per launch — so the LB's own environment must not be a
#: SECOND source for any of them: one name, one meaning, one path.
PER_MODEL_SETTINGS = ("CLUSTER_IMAGE", "CLUSTER_MODELS_DIRS", "CLUSTER_MODEL_NAME", "CLUSTER_SERVED_MODEL_NAME",
                      "CLUSTER_GPU_MEMORY_UTILIZATION", "CLUSTER_KV_CACHE_MEMORY", "CLUSTER_TOOL_CALL_PARSER",
                      "CLUSTER_REASONING_PARSER", "CLUSTER_SPECULATIVE_CONFIG", "CLUSTER_MTP_PREFIX",
                      "CLUSTER_PLE_REPLICATE")


def test_the_operators_launch_settings_reach_both_ranks_as_set_and_only_when_set(tmp_path, monkeypatch):
    """Set once on the LB (lb/.env), never per GB10 box, so the two ranks cannot launch differently. Unset
    stays unset — tp-rank's defaults are the only defaults — and an empty value travels as empty,
    because empty is how the cache is switched off."""
    script = (Path(vllm_lb.__file__).resolve().parent.parent / "cluster" / "tp-rank").read_text()
    for key in LAUNCH_SETTINGS:
        assert re.search("^" + re.escape(f': "${{{key}') + ":?=", script, re.M), \
            f"tp-rank has no setting {key} to forward"
        monkeypatch.delenv(key, raising=False)
    assert not set(LAUNCH_SETTINGS) & set(build(tmp_path).executor.env)
    monkeypatch.setenv("CLUSTER_CACHE_DIR", "")
    env = build(tmp_path).executor.env
    assert {key: env.get(key) for key in LAUNCH_SETTINGS} == {"CLUSTER_CACHE_DIR": ""}
    assert env["CLUSTER_PORT"] == "8100" and env["CLUSTER_MASTER_ADDR"] == "192.0.2.201"


def test_a_models_own_launch_settings_are_never_taken_from_the_lbs_environment(tmp_path, monkeypatch):
    """DRY: a stray CLUSTER_MODEL_NAME in the LB's environment must not be able to launch a model the
    operator never chose, nor silently disagree with the ModelChoice's own LaunchProfile."""
    for key in PER_MODEL_SETTINGS:
        monkeypatch.setenv(key, "from-the-lb-environment")
    env = build(tmp_path).executor.env
    leaked = sorted(set(PER_MODEL_SETTINGS) & set(env))
    assert leaked == [], f"{leaked} would be a second source of a model's launch settings"


def test_the_switch_and_the_loop_read_ONE_model_catalog(tmp_path):
    """The classic picker and the cluster picker share it, and nothing loads a second copy."""
    loop = build(tmp_path)
    assert loop.store.catalog.error == "", loop.store.catalog.error
    assert getattr(loop, "catalog", loop.store.catalog) is loop.store.catalog


def test_only_the_operators_stored_choice_may_change_a_models_speculation(tmp_path, monkeypatch):
    """The rule at TP_RANK_SETTINGS (vllm_lb/__main__.py): a model's speculation is its catalog LaunchProfile's,
    and the LB's environment is never a second source — whatever a stray variable is called. The operator's
    choice, stored through POST /lb/cluster, is the one override: that is the operator choosing, on the record."""
    import dataclasses

    from vllm_lb.cluster import Desired

    stray = ("CLUSTER_SPECULATIVE_CONFIG", "CLUSTER_NUM_SPECULATIVE_TOKENS", "LB_CLUSTER_NUM_SPECULATIVE_TOKENS",
             "LB_CLUSTER_SPECULATIVE_CONFIG", "NUM_SPECULATIVE_TOKENS", "VLLM_SPECULATIVE_CONFIG")
    for key in stray:
        monkeypatch.setenv(key, '{"method":"ngram","num_speculative_tokens":7}' if "CONFIG" in key else "7")
    loop = build(tmp_path)
    assert not set(stray) & set(loop.executor.env), "a stray variable reached every launch"
    assert not [f.name for f in dataclasses.fields(Config) if "specul" in f.name], "Config grew a speculation knob"
    flash = Desired(mode="cluster", generation=1, model_choice="qwen3.8-flash-next-hibrid48")
    assert loop._resolve(flash)["num_speculative_tokens"] == 4, "the catalog's own, whatever the environment says"
    stored = loop.store.set(enabled=True, num_speculative_tokens=2)
    assert loop._resolve(stored)["num_speculative_tokens"] == 2, "the operator's stored choice is the override"
