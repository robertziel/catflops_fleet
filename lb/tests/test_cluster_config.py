"""The cluster's configuration: two hosts, head first, and a URL — or it is not configured at all.

A host is `label=box=user@control-address=interconnect-address`: the broker lane label its park row
is keyed by, the physical box, where the LB reaches it (box-a-fast / box-b-fast, the direct links
from box-c) and the address the ranks talk on (pair-link, the 2×200 Gb bond between the GB10 boxes).
"""
from __future__ import annotations

import pytest

from vllm_lb.cluster import ClusterHost, parse_hosts
from vllm_lb.config import Config

HOSTS = ("box-a-gpu=box-a=user@192.0.2.2=192.0.2.201,"
         "box-b-gpu=box-b=user@192.0.2.12=192.0.2.202")


def test_hosts_parse_in_rank_order_with_their_interconnect():
    assert parse_hosts(HOSTS) == (
        ClusterHost("box-a-gpu", "box-a", "user@192.0.2.2", "192.0.2.201"),
        ClusterHost("box-b-gpu", "box-b", "user@192.0.2.12", "192.0.2.202"))
    assert parse_hosts("") == ()


@pytest.mark.parametrize("spec", ["box-a-gpu=box-a=user@x", "box-a-gpu==user@x=10.0.0.1",
                                  "=box-a=user@x=10.0.0.1", "box-a-gpu=box-a=user@x="])
def test_a_malformed_host_entry_is_refused(spec):
    with pytest.raises(ValueError):
        parse_hosts(spec)


def test_the_fleet_file_names_it_and_the_environment_tunes_it(tmp_path):
    from tests.fleet_files import PAIR, write
    cfg = Config.from_env({"LB_ALLOWED_HOSTS": "loopback", "LB_FLEET": write(tmp_path, PAIR.replace("http://192.0.2.10:8100", "http://192.0.2.10:8100/")),
                           "LB_CLUSTER_STATE": "/data/c.json",
                           "LB_CLUSTER_MEM_SAMPLES": "5", "LB_CLUSTER_FREEING_DEADLINE_S": "300"})
    assert cfg.cluster_configured is True
    assert cfg.cluster_url == "http://192.0.2.10:8100", "a trailing slash is not part of the head's address"
    assert cfg.cluster_state_path == "/data/c.json"
    # L3: how many consecutive observations must clear the need, and how long `freeing` may wait.
    assert (cfg.cluster_mem_samples, cfg.cluster_freeing_deadline_s) == (5, 300.0)
    assert (Config().cluster_mem_samples, Config().cluster_freeing_deadline_s) == (3, 600.0)
    assert (cfg.cluster_hosts[0].box, cfg.cluster_hosts[0].interconnect) == ("node-1", "198.51.100.1")


def test_unset_is_not_configured_and_half_a_config_is_not_either():
    assert Config.from_env({"LB_ALLOWED_HOSTS": "loopback"}).cluster_configured is False
    assert Config(cluster_name="pair", cluster_hosts=parse_hosts(HOSTS)).cluster_configured is False, "no URL"
    assert Config(cluster_name="pair", cluster_hosts=parse_hosts("box-a-gpu=box-a=user@x=10.0.0.1"),
                  cluster_url="http://x:8100").cluster_configured is False, "one host"


def test_LB_CLUSTER_MIN_MEM_GIB_is_retired_there_is_no_flat_requirement():
    """A selection is gated on what it COSTS (memory_estimate, per rank role); one the LB cannot size is refused
    naming why — never admitted on a guessed 100 GiB (operator rule: no fallbacks, 2026-10-29)."""
    for value in ("100", ""):
        with pytest.raises(ValueError, match="LB_CLUSTER_MIN_MEM_GIB") as e:
            Config.from_env({"LB_ALLOWED_HOSTS": "loopback", "LB_CLUSTER_MIN_MEM_GIB": value})
        assert "retired" in str(e.value) and "refused" in str(e.value), str(e.value)
    assert not hasattr(Config(), "cluster_min_mem_gib")
