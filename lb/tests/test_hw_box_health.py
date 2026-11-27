"""The gateway's side of BOX HEALTH: what /lb/status `hardware.<box>` makes of a sender's `rdma`,
`kernel`, `disks` and `models` blocks (vllm_lb/box_health.py, fed by vllm_lb/hw.py).

The rules pinned:
  * an RDMA counter's RATE is the gateway's: (this − last) / the two samples' own `sampled_at` gap. No last sample → no
    rate, with why; a counter that went BACKWARDS is a reset → no rate that sample, with why — never a negative, never 0;
  * what sysfs never states (rx_discards_phy, PFC, trust) is said once per box, never shown as a number;
  * the kernel's and the disks' figures pass through as the sender read them; anything malformed is withheld whole;
  * a models root's folders are named by the fleet file's checkpoints whose path starts with them;
  * a stale box withdraws them all (they are no longer now), like every other live figure.
"""
from __future__ import annotations

import json

import pytest

from vllm_lb import box_health
from vllm_lb.catalog import Checkpoint
from vllm_lb.config import Config
from vllm_lb.fleet import parse_fleet
from vllm_lb.hw import HwCollector

FLEET = parse_fleet("panel: {style: vibe}\nboxes:\n  box: {hostname: 'box-h', platform: linux/arm64, cards: "
                    "[{index: 0, type: gb10, serves: false}]}\nclusters: {}\nbackends: {}\n", "fleet.yml")


def port(**counters):
    return {"dev": "mlx5_0", "port": 1, "state": "ACTIVE", "phys_state": "LinkUp", "rate_gbps": 200.0,
            "link_layer": "Ethernet", "netdev": "enp1s0f0np0", "mtu": 9000, "counters": counters, "why": {}}


def sample(at: str, ports=None, **extra):
    return json.dumps({"host": "box-h", "sampled_at": at, "cpu_percent": 1.0, "gpus": [],
                       "rdma": {"ports": ports if ports is not None else [], "why": None if ports else "no ports"},
                       **extra})


def collector(checkpoints=None):
    return HwCollector(Config(hw_hosts=dict(FLEET.hw_hosts), fleet=FLEET, hw_max_age_s=30),
                       checkpoints=checkpoints or ({}, None))


def test_rates_are_per_second_over_the_samples_own_clock():
    c = collector()
    c.ingest(sample("2026-11-27T10:00:00+00:00", [port(out_of_buffer=100, port_rcv_errors=5)]), now=1000.0)
    box = c.ingest_and_snapshot(sample("2026-11-27T10:00:02+00:00", [port(out_of_buffer=140, port_rcv_errors=5)]),
                                now=1000.5)
    (p,) = box["rdma"]["ports"]
    assert p["counters"] == {"out_of_buffer": 140, "port_rcv_errors": 5}
    assert p["rates"] == {"out_of_buffer": 20.0, "port_rcv_errors": 0.0}, "the box's 2 s, not the LB's 0.5 s"
    assert p["rates_why"] == {}


def test_the_first_sample_has_no_rate_and_says_why():
    box = collector().ingest_and_snapshot(sample("2026-11-27T10:00:00+00:00", [port(out_of_buffer=1)]), now=1.0)
    (p,) = box["rdma"]["ports"]
    assert p["rates"] == {} and "first sample" in p["rates_why"]["out_of_buffer"]


def test_a_counter_that_went_backwards_is_a_reset_with_no_rate():
    c = collector()
    c.ingest(sample("2026-11-27T10:00:00+00:00", [port(out_of_buffer=100, link_downed=1)]), now=1.0)
    box = c.ingest_and_snapshot(sample("2026-11-27T10:00:01+00:00", [port(out_of_buffer=3, link_downed=1)]), now=2.0)
    (p,) = box["rdma"]["ports"]
    assert "out_of_buffer" not in p["rates"] and "reset" in p["rates_why"]["out_of_buffer"]
    assert p["rates"]["link_downed"] == 0.0


def test_a_ports_qos_passes_through_and_its_phy_counters_get_rates_like_the_rest():
    qos = {"pfc_prios": [3], "trust": None, "dscp_map": 4, "qos_at": 99.5}
    c = collector()
    c.ingest(sample("2026-11-27T10:00:00+00:00", [{**port(rx_discards_phy=10), **qos}]), now=1.0)
    box = c.ingest_and_snapshot(sample("2026-11-27T10:00:02+00:00", [{**port(rx_discards_phy=30), **qos,
                                                                       "why": {"trust": "no apptrust"}}]), now=2.0)
    (p,) = box["rdma"]["ports"]
    assert (p["pfc_prios"], p["trust"], p["dscp_map"], p["qos_at"]) == ([3], None, 4, 99.5)
    assert p["why"]["trust"] == "no apptrust" and p["rates"]["rx_discards_phy"] == 10.0
    assert "unread" not in box["rdma"], "rx_discards_phy, PFC and trust are read now, not listed as unread"
    none = collector().ingest_and_snapshot(sample("2026-11-27T10:00:00+00:00", []), now=1.0)
    assert none["rdma"] == {"ports": [], "why": "no ports"}


def test_a_malformed_qos_figure_is_withheld():
    bad = {**port(), "pfc_prios": [3, "x", True, 9], "trust": "dscp", "dscp_map": -1, "qos_at": float("inf")}
    (p,) = collector().ingest_and_snapshot(sample("t", [bad]), now=1.0)["rdma"]["ports"]
    assert (p["pfc_prios"], p["trust"], p["dscp_map"], p["qos_at"]) == (None, None, None, None)


def test_an_absent_more_count_is_absent_with_why_never_zero():
    models = [{"root": "/m", "dirs": {"x": 1}, "measured_at": 1.0, "took_s": 0.1, "unreadable": [], "why": None}]
    (r,) = collector().ingest_and_snapshot(sample("t", models=models, models_why=None), now=1.0)["models"]
    assert r["more_count"] is None and r["more_bytes"] is None
    assert "more_count" in r["more_why"]


def test_a_malformed_port_is_withheld_whole_and_a_bad_counter_is_not_a_number():
    bad = {**port(out_of_buffer=True, local_ack_timeout_err=-1, packet_seq_err=float("nan"), symbol_error=4)}
    box = collector().ingest_and_snapshot(sample("2026-11-27T10:00:00+00:00", [bad, {"dev": 7}, "x"]), now=1.0)
    (p,) = box["rdma"]["ports"]
    assert p["counters"] == {"symbol_error": 4}


def test_an_old_sender_without_the_blocks_is_absent_with_why():
    box = collector().ingest_and_snapshot(json.dumps({"host": "box-h", "sampled_at": "x", "gpus": []}), now=1.0)
    assert box["rdma"] == {"ports": [], "why": "the sender sends no rdma block (it predates box health)"}
    assert box["kernel"]["xid_total"] is None and "predates" in box["kernel"]["why"]
    assert box["disks"] == [] and "predates" in box["disks_why"]
    assert box["models"] == [] and "predates" in box["models_why"]


def test_kernel_and_disks_pass_through():
    kernel = {"xid_total": 2, "oom_kills_total": 3, "last_xid": {"code": 79, "at": 1.5}, "xid_since": 0.5,
              "xid_complete": True, "why": None}
    disks = [{"path": "/", "total_bytes": 100, "used_bytes": 40, "free_bytes": 55},
             {"path": "/gone", "total_bytes": None, "used_bytes": None, "free_bytes": None, "why": "statvfs: gone"}]
    box = collector().ingest_and_snapshot(sample("t", kernel=kernel, disks=disks, disks_why=None), now=1.0)
    assert box["kernel"] == kernel
    assert box["disks"] == disks and box["disks_why"] is None


def test_models_folders_are_named_by_the_fleets_checkpoints():
    folders = box_health.checkpoint_folders({
        "a": Checkpoint("qwen-cluster/Qwen-A", "qwen-cluster/Qwen-A_uncensored"),
        "b": Checkpoint("hf/hub/models--x/snapshots/1"), "c": Checkpoint("Solo", draft="Solo-draft")})
    assert folders == {"qwen-cluster": ["a", "a (uncensored)"], "hf": ["b"], "Solo": ["c"], "Solo-draft": ["c (draft)"]}
    models = [{"root": "/mnt/models", "dirs": {"qwen-cluster": 300, "other": 7, "hf": 50}, "measured_at": 10.0,
               "took_s": 2.0, "more_count": 1, "more_bytes": 4, "unreadable": ["locked"], "why": None}]
    box = collector((folders, None)).ingest_and_snapshot(sample("t", models=models, models_why=None), now=1.0)
    (r,) = box["models"]
    assert [f["name"] for f in r["folders"]] == ["qwen-cluster", "hf", "other"], "largest first"
    assert r["folders"][0] == {"name": "qwen-cluster", "bytes": 300, "checkpoints": ["a", "a (uncensored)"]}
    assert r["folders"][2]["checkpoints"] == []
    assert (r["measured_at"], r["took_s"], r["more_count"], r["more_bytes"], r["unreadable"]) == (10.0, 2.0, 1, 4,
                                                                                                  ["locked"])


def test_a_fleet_whose_checkpoints_cannot_be_read_says_so_on_each_root():
    models = [{"root": "/m", "dirs": {"x": 1}, "measured_at": 1.0, "took_s": 0.1, "more_count": 0, "more_bytes": 0,
               "unreadable": [], "why": None}]
    box = collector(({}, "the fleet file: broken")).ingest_and_snapshot(sample("t", models=models, models_why=None),
                                                                         now=1.0)
    assert box["models"][0]["names_why"] == "the fleet file: broken"


def test_a_stale_box_withdraws_its_health_figures():
    c = collector()
    c.ingest(sample("2026-11-27T10:00:00+00:00", [port(out_of_buffer=1)], disks=[], disks_why="x"), now=1.0)
    box = c.snapshot(now=100.0)["box"]
    assert box["stale"] is True
    assert box["rdma"] is None and box["kernel"] is None and box["disks"] is None and box["models"] is None


@pytest.mark.parametrize("env,expect", [({}, "names no fleet file")])
def test_deployment_folders_without_a_fleet_say_why(env, expect, monkeypatch):
    monkeypatch.delenv("LB_FLEET", raising=False)
    folders, why = box_health.deployment_folders(env)
    assert folders == {} and expect in why
