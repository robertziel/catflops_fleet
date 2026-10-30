"""THE LB READS ITS TOPOLOGY FROM THE FLEET FILE (LB_FLEET) — and refuses the settings it replaces.

AGENTS.md, operator rule 2026-10-11: what a fleet IS lives in YAML, and `LB_CLUSTER_HOSTS`, `LB_CLUSTER_URL`,
`LB_VLLM_BOXES` and `LB_HW_HOSTS` move into it. A retired setting is REFUSED at boot naming its replacement — never
silently ignored, so no value ever has two definitions (the chat's auth.py RETIRED_KEYS pattern).

The bridge is pinned against the env it replaces: a fleet file of the box-c shape must give the LB EXACTLY the
cluster hosts, cluster URL, classic boxes and hardware map that shape's four env lines gave it — so a redeploy with
the file in place and the lines removed changes no routing, no lease and no tile.
"""
from __future__ import annotations

import pytest

from tests.fleet_files import ENV_OF_PAIR, ONE_HOST, PAIR, write
from vllm_lb.cluster import parse_hosts
from vllm_lb.config import RETIRED_KEYS, Config, parse_boxes, parse_hw_hosts


def test_the_pair_fleet_gives_the_lb_exactly_what_its_four_env_lines_gave(tmp_path):
    cfg = Config.from_env({"LB_ALLOWED_HOSTS": "loopback", "LB_FLEET": write(tmp_path, PAIR)})
    assert cfg.cluster_hosts == parse_hosts(ENV_OF_PAIR["LB_CLUSTER_HOSTS"])
    assert cfg.cluster_url == ENV_OF_PAIR["LB_CLUSTER_URL"]
    assert cfg.vllm_boxes == parse_boxes(ENV_OF_PAIR["LB_VLLM_BOXES"])
    assert cfg.hw_hosts == parse_hw_hosts(ENV_OF_PAIR["LB_HW_HOSTS"])
    assert cfg.cluster_configured is True
    assert cfg.fleet is not None and set(cfg.fleet.boxes) == {"node-1", "node-2", "gateway"}


def test_a_one_host_cluster_the_lb_does_not_start_is_the_node_it_observes(tmp_path):
    """control: none — its own deployment starts it; the LB observes it through the same cluster loop with no executor
    and no switch (vllm_lb/topology.py). Its one host has no SSH target and no interconnect: nothing would use them."""
    from vllm_lb.cluster import ClusterHost
    cfg = Config.from_env({"LB_ALLOWED_HOSTS": "loopback", "LB_FLEET": write(tmp_path, ONE_HOST)})
    assert cfg.cluster_control == "none" and cfg.cluster_configured is True
    assert cfg.cluster_hosts == (ClusterHost(label="solo-gpu", box="solo", ssh_target="", interconnect=""),)
    assert cfg.cluster_url == "http://vllm-node:8000" and cfg.vllm_boxes == ()
    assert cfg.hw_hosts == {"solo-host": "solo"}


def test_a_box_with_a_classic_server_and_a_leased_node_is_refused_an_observed_one_is_not(tmp_path):
    """A node the broker leases (control local) is named by its box's label — the lane a claim stamps — as the box's
    classic server is: refused. An observed node (control none) takes no lease and names no lane: beside it."""
    from tests.test_cluster_local import FLEET
    both = FLEET.replace("backends: {}", "backends:\n  node-a: {url: http://vllm-node:8001}")
    with pytest.raises(ValueError) as e:
        Config.from_env({"LB_ALLOWED_HOSTS": "loopback", "LB_FLEET": write(tmp_path, both)})
    assert "node-a" in str(e.value) and "both would be named" in str(e.value)
    observed = ONE_HOST.replace("backends: {}", "backends:\n  solo: {url: http://vllm-node:8001}")
    cfg = Config.from_env({"LB_ALLOWED_HOSTS": "loopback", "LB_FLEET": write(tmp_path, observed)})
    assert [b.box for b in cfg.vllm_boxes] == ["solo"] and cfg.clusters[0].control == "none"


def test_two_declared_clusters_may_share_a_box_each_says_whether_it_boots(tmp_path):
    """The fleet file is a CATALOG of alternatives (worklog/parallel-clusters.md): both are exposed; the run decides."""
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
    cfg = Config.from_env({"LB_ALLOWED_HOSTS": "loopback", "LB_FLEET": write(tmp_path, two)})
    assert [(c.name, c.boot) for c in cfg.clusters] == [("pair", True), ("again", False)]


def test_no_fleet_file_names_no_machine():
    """Unset is OFF, as the shipped defaults have always been: the LB dials no box and maps no sender. Compose
    always sets LB_FLEET and mounts the file, so a deployment never runs this way by accident."""
    cfg = Config.from_env({"LB_ALLOWED_HOSTS": "loopback"})
    assert (cfg.fleet, cfg.cluster_hosts, cfg.cluster_url, cfg.vllm_boxes, cfg.hw_hosts) == (None, (), "", (), {})


def test_a_fleet_file_that_is_not_there_refuses_the_start_naming_it(tmp_path):
    with pytest.raises(ValueError, match="nope.yml"):
        Config.from_env({"LB_ALLOWED_HOSTS": "loopback", "LB_FLEET": str(tmp_path / "nope.yml")})


def test_a_broken_fleet_file_refuses_the_start_naming_the_value(tmp_path):
    with pytest.raises(ValueError, match="radeon9999"):
        Config.from_env({"LB_ALLOWED_HOSTS": "loopback", "LB_FLEET": write(tmp_path, PAIR.replace("type: radeon8060s", "type: radeon9999"))})


@pytest.mark.parametrize("key", ["LB_CLUSTER_HOSTS", "LB_CLUSTER_URL", "LB_VLLM_BOXES", "LB_HW_HOSTS",
                                 "CLUSTER_MAX_NUM_BATCHED_TOKENS"])
def test_a_retired_setting_is_refused_naming_its_replacement(tmp_path, key):
    with pytest.raises(ValueError) as e:
        Config.from_env({"LB_ALLOWED_HOSTS": "loopback", "LB_FLEET": write(tmp_path, PAIR), key: ENV_OF_PAIR[key]})
    assert key in str(e.value) and "fleet" in str(e.value)


@pytest.mark.parametrize("key", ["LB_CLUSTER_HOSTS", "LB_CLUSTER_URL", "LB_VLLM_BOXES", "LB_HW_HOSTS",
                                 "CLUSTER_MAX_NUM_BATCHED_TOKENS"])
def test_a_retired_setting_left_blank_is_still_refused(key):
    """Never silently ignored: a blank line in lb/.env is still the old setting's name, and the fleet file is the
    one place its value lives now."""
    with pytest.raises(ValueError, match=key):
        Config.from_env({"LB_ALLOWED_HOSTS": "loopback", key: ""})


def test_every_retired_setting_names_where_its_value_went():
    for key in ("LB_CLUSTER_HOSTS", "LB_CLUSTER_URL", "LB_VLLM_BOXES", "LB_HW_HOSTS", "CLUSTER_MAX_NUM_BATCHED_TOKENS"):
        assert "fleet" in RETIRED_KEYS[key], key
    assert "clusters.<name>.max_num_batched_tokens" in RETIRED_KEYS["CLUSTER_MAX_NUM_BATCHED_TOKENS"]
    # A value the MODEL owns went to its catalog row (operator, 2026-10-29: "why do we have LB_CONTEXT_LENGTH? Isn't
    # that set in configs?") — the refusal names the catalog, not the fleet file.
    for key in ("LB_CONTEXT_LENGTH",):
        assert "catalog" in RETIRED_KEYS[key] and "fleet" not in RETIRED_KEYS[key], key
        with pytest.raises(ValueError, match=key) as e:
            Config.from_env({"LB_ALLOWED_HOSTS": "loopback", key: "1"})
        assert "catalog" in str(e.value) and "LB_FLEET" not in str(e.value), str(e.value)


def test_the_prefill_budget_is_the_fleets_per_cluster(tmp_path):
    """The box's prefill budget (vLLM --max-num-batched-tokens) is the fleet file's `clusters.<name>.max_num_batched_tokens`,
    carried to tp-rank on every call (`cluster_run.tp_rank_env`) — never the LB's environment (CLUSTER_MAX_NUM_BATCHED_TOKENS
    is retired) and never tp-rank's own number. A cluster Config with none refuses to build the executor's env."""
    from vllm_lb.cluster_run import TP_RANK_SETTINGS, tp_rank_env
    cfg = Config.from_env({"LB_ALLOWED_HOSTS": "loopback", "LB_FLEET": write(tmp_path, PAIR)})
    assert cfg.clusters[0].max_num_batched_tokens == cfg.cluster_max_num_batched_tokens == 8192
    assert "CLUSTER_MAX_NUM_BATCHED_TOKENS" not in TP_RANK_SETTINGS
    assert tp_rank_env(cfg)["CLUSTER_MAX_NUM_BATCHED_TOKENS"] == "8192"
    assert tp_rank_env(cfg)["CLUSTER_IFACE"] == "bond0"
    from dataclasses import replace
    with pytest.raises(ValueError, match=r"clusters\.pair\.max_num_batched_tokens"):
        tp_rank_env(replace(cfg, clusters=(replace(cfg.clusters[0], max_num_batched_tokens=None),),
                            cluster_max_num_batched_tokens=None))
