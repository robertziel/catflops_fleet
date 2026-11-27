"""HARDWARE HEALTH FINDINGS (vllm_lb/health.py): each box's latest sample judged every tick — GPU
temperature headroom below its own driver-stated throttle limit, RAM available, a busy GPU at idle draw, Xid / OOM-kill / RoCE
error counters rising, a server that stopped answering. A finding is PENDING until its condition has held for the
fleet's hold time, then FIRING; it clears when the condition stops. A figure the box did not send is NO DATA with why —
never a finding, never a default threshold. Limits: the hardware's own driver (read live by the sender), operator policy
from the fleet file's `health:` block (every key required, unknown refused)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.fleet_files import PAIR
from vllm_lb import events
from vllm_lb.events import EventLog
from vllm_lb.fleet import parse_fleet
from vllm_lb.health import ERROR_COUNTERS, HealthEvaluator, HealthPolicy, parse_policy

BLOCK = {"hold_s": 90, "temp_margin_c": 5, "low_ram_fraction": 0.05, "busy_gpu_pct": 90, "low_power_margin_w": 2,
         "counter_window_s": 3600, "backend_down_s": 120, "disk_full_pct": 90}
HEALTH_YAML = """health:
  hold_s: 90
  temp_margin_c: 5
  low_ram_fraction: 0.05
  busy_gpu_pct: 90
  low_power_margin_w: 2
  counter_window_s: 3600
  backend_down_s: 120
  disk_full_pct: 90
"""
#: The boxes the evaluator is judged on: `solo` a GB10 (its type states its idle draw, 8.9 W), `big` an RTX 5090 (its
#: type states none) — every figure through the fleet file and the hardware library, as the LB reads them.
FLEET = parse_fleet("""
panel:
  style: vibe
boxes:
  solo: {hostname: solo-host, platform: linux/arm64, cards: [{index: 0, type: gb10, serves: false}]}
  big: {hostname: big-host, platform: linux/amd64, cards: [{index: 0, type: rtx5090, serves: false}]}
clusters: {}
backends: {}
""", "fleet.yml")


# ── the policy: the fleet file's `health:` block ───────────────────────────────────────────────────────────────────

def test_a_complete_block_reads_as_the_policy():
    p = parse_policy("fleet.yml: health", BLOCK)
    assert p == HealthPolicy(**{k: float(v) for k, v in BLOCK.items()})


@pytest.mark.parametrize("change,says", [
    ({"hold_s": None}, "hold_s"),                                   # missing (None = drop it below)
    ({"colour": 1}, "colour"),                                      # unknown
    ({"busy_gpu_pct": 120}, "busy_gpu_pct"),                        # out of range
    ({"low_ram_fraction": 0}, "low_ram_fraction"),                  # a share of the box's RAM: 0 < f < 1
    ({"low_ram_fraction": 1}, "low_ram_fraction"),
    ({"low_ram_fraction": 4}, "low_ram_fraction"),                  # GiB is no longer the unit
    ({"low_power_margin_w": -1}, "low_power_margin_w"),
    ({"low_ram_gib": 4}, "low_ram_gib"),                            # the retired key is unknown
    ({"hold_s": "soon"}, "hold_s"),                                 # not a number
    ({"temp_margin_c": True}, "temp_margin_c"),                     # a boolean is no number
    ({"disk_full_pct": None}, "disk_full_pct"),                     # the panel's full disk is policy too: required
    ({"disk_full_pct": 0}, "disk_full_pct"),                        # a percent: 0 < x ≤ 100
    ({"disk_full_pct": -5}, "disk_full_pct"),
    ({"disk_full_pct": 101}, "disk_full_pct"),
])
def test_a_block_missing_a_key_naming_an_unknown_one_or_out_of_range_is_refused(change, says):
    raw = {**BLOCK, **change}
    raw = {k: v for k, v in raw.items() if v is not None}
    with pytest.raises(ValueError, match=says):
        parse_policy("fleet.yml: health", raw)


def test_the_fleet_reads_the_block_and_without_one_health_is_off():
    assert parse_fleet(PAIR, "fleet.yml").health is None
    f = parse_fleet(PAIR.rstrip("\n") + "\n" + HEALTH_YAML, "fleet.yml")
    assert f.health == parse_policy("x", BLOCK)
    with pytest.raises(ValueError, match="health"):
        parse_fleet(PAIR.rstrip("\n") + "\n" + HEALTH_YAML.replace("  hold_s: 90\n", ""), "fleet.yml")


def test_an_existing_health_block_without_disk_full_pct_refuses_the_boot_naming_the_key_and_the_file():
    """THE UPGRADE PATH: `disk_full_pct` is required, so a deployment's fleet file with a
    `health:` block written before it is refused at boot (fleet.py `parse_fleet` → `parse_policy`) — naming the key
    to add and the file to add it to (CHANGELOG.md Unreleased says so) — never a silent default threshold."""
    old = HEALTH_YAML.replace("  disk_full_pct: 90\n", "")
    assert "disk_full_pct" not in old
    with pytest.raises(ValueError) as got:
        parse_fleet(PAIR.rstrip("\n") + "\n" + old, "/etc/vllm-lb/fleet.yml")
    said = str(got.value)
    assert "missing disk_full_pct" in said and said.startswith("/etc/vllm-lb/fleet.yml: health"), said


# ── the evaluator ───────────────────────────────────────────────────────────────────────────────────────────────────

def disk(path: str, pct: float, total: float = 1e12) -> dict:
    """One configured disk path as box_health.py `disks_view` carries it, `pct` used."""
    return {"path": path, "total_bytes": total, "used_bytes": total * pct / 100,
            "free_bytes": total * (100 - pct) / 100}


def hw(**over) -> dict:
    """One box's /lb/status hardware row (hw.py snapshot) with a healthy reading of everything health reads."""
    row = {"age_s": 1.0, "stale": False, "gpu_pct": 10, "ram_total_mb": 131072, "ram_used_mb": 32768,
           "temp_headroom_c": 30, "temp_headroom_why": None,
           "recorder": {"figures": {"gpu_temp_c": 50, "gpu_power_w": 60.0}, "why": {}},
           "kernel": {"xid_total": 0, "oom_kills_total": 0, "last_xid": None, "why": None},
           "rdma": {"ports": [{"dev": "mlx5_0", "port": 1, "state": "ACTIVE",
                               "counters": {c: 0 for c in ERROR_COUNTERS}, "why": {}}], "why": None},
           "disks": [disk("/", 30)], "disks_why": None}
    for k, v in over.items():
        if k in ("gpu_temp_c", "gpu_power_w"):
            row["recorder"]["figures"][k] = v
        else:
            row[k] = v
    return row


@pytest.fixture
def feed(tmp_path):
    lg = EventLog(str(tmp_path))
    events.install(lg)
    yield lg
    events.install(None)


def solo(**kw) -> HealthEvaluator:
    return HealthEvaluator(parse_policy("x", {**BLOCK, **kw}), FLEET)


def run(ev: HealthEvaluator, t: float, box_rows: dict, backends=(), broker=False) -> dict:
    ev.evaluate(t, box_rows, list(backends), broker_enabled=broker)
    return ev.view()


def finding(view: dict, kind: str, box: str = "solo") -> dict | None:
    return next((f for f in view["findings"] if f["kind"] == kind and f["box"] == box), None)


def test_a_gpu_near_its_own_limit_is_pending_then_firing_after_the_hold_then_clears_with_one_event_each(feed):
    """GPU TEMP is held against the DRIVER's limit: the tightest card's headroom (hw.py `temp_headroom_c`: nvidia-smi
    temperature.gpu.tlimit, amdgpu crit − input) ≤ the fleet's margin (5 °C here)."""
    ev = solo()
    v = run(ev, 1000, {"solo": hw(temp_headroom_c=4)})
    f = finding(v, "gpu_temp")
    assert (f["state"], f["value"], f["threshold"], f["since"]) == ("pending", 4, 5, 1000)
    assert "throttle limit" in f["why"]
    assert finding(run(ev, 1089, {"solo": hw(temp_headroom_c=3)}), "gpu_temp")["state"] == "pending"
    f = finding(run(ev, 1090, {"solo": hw(temp_headroom_c=5)}), "gpu_temp")
    assert f["state"] == "firing" and f["since"] == 1000, "at the margin is inside it (≤)"
    run(ev, 1100, {"solo": hw(temp_headroom_c=2)})                   # still firing: no second event
    assert finding(run(ev, 1200, {"solo": hw(temp_headroom_c=20)}), "gpu_temp") is None
    feed.drain()
    got = [(e["kind"], e["subject"], e["finding"]) for e in reversed(feed.read()["events"])]
    assert got == [("finding.firing", "solo", "gpu_temp"), ("finding.cleared", "solo", "gpu_temp")]


def test_a_condition_that_stops_before_its_hold_never_fires(feed):
    ev = solo()
    run(ev, 1000, {"solo": hw(temp_headroom_c=4)})
    assert finding(run(ev, 1050, {"solo": hw(temp_headroom_c=25)}), "gpu_temp") is None
    feed.drain()
    assert feed.read()["events"] == []


def test_a_hold_of_zero_fires_on_the_first_tick():
    v = run(solo(hold_s=0), 1000, {"solo": hw(temp_headroom_c=1)})
    assert finding(v, "gpu_temp")["state"] == "firing"


def test_a_gpu_whose_driver_states_no_limit_is_no_data_with_the_drivers_reason_never_a_default():
    """The Radeon 8060S's amdgpu hwmon has temp1_input and no temp1_crit: no limit, so no GPU TEMP finding — and a
    GPU at 99 °C is not judged against a number the LB made up."""
    why = "card 0: no temp1_crit in the amdgpu hwmon: the driver states no limit for this card"
    v = run(solo(hold_s=0), 1000, {"apu": hw(temp_headroom_c=None, temp_headroom_why=why, gpu_temp_c=99)})
    assert finding(v, "gpu_temp", "apu") is None
    assert next(n for n in v["no_data"] if n["box"] == "apu" and n["kind"] == "gpu_temp")["why"] == why


def test_low_ram_is_a_share_of_the_boxs_own_ram_so_one_value_scales_across_boxes():
    """5 % of a 128 GiB box is 6.4 GiB; of a 16 GiB host, 0.8 GiB: the same policy, each box's own floor."""
    v = run(solo(hold_s=0), 1000, {"solo": hw(ram_total_mb=131072, ram_used_mb=131072 - 2048)})
    f = finding(v, "low_ram")
    assert (f["state"], f["value"], f["threshold"]) == ("firing", 2.0, 6.4) and "5 % of 128 GiB" in f["why"]
    small = run(solo(hold_s=0), 1000, {"solo": hw(ram_total_mb=16384, ram_used_mb=16384 - 2048)})
    assert finding(small, "low_ram") is None, "2 GiB free of 16 GiB is 12.5 %: over a 5 % floor"


def test_a_busy_gpu_at_its_types_idle_draw_plus_the_margin_fires_and_a_busy_gpu_drawing_power_does_not():
    """GB10 idle 8.9 W (its type's measured fact) + the fleet's 2 W margin = 10.9 W."""
    v = run(solo(hold_s=0), 1000, {"solo": hw(gpu_pct=97, gpu_power_w=10.9)})
    f = finding(v, "low_power")
    assert (f["state"], f["value"], f["threshold"]) == ("firing", 10.9, 10.9), "at the ceiling is inside it (≤)"
    assert "idle 8.9 W" in f["why"] and "margin 2 W" in f["why"]
    assert finding(run(solo(hold_s=0), 1000, {"solo": hw(gpu_pct=97, gpu_power_w=11.0)}), "low_power") is None
    assert finding(run(solo(hold_s=0), 1000, {"solo": hw(gpu_pct=20, gpu_power_w=8.0)}), "low_power") is None


def test_a_card_type_without_an_idle_draw_is_no_data_naming_its_source_never_a_default():
    v = run(solo(hold_s=0), 1000, {"big": hw(gpu_pct=99, gpu_power_w=5.0)})
    assert finding(v, "low_power", "big") is None
    why = next(n["why"] for n in v["no_data"] if n["box"] == "big" and n["kind"] == "low_power")
    assert "RTX 5090" in why and "not stated or measured" in why
    v = run(solo(hold_s=0), 1000, {"stranger": hw(gpu_pct=99, gpu_power_w=5.0)})
    assert any(n["box"] == "stranger" and n["kind"] == "low_power" for n in v["no_data"])


def test_a_missing_figure_is_no_data_with_the_senders_why():
    row = hw()
    row["temp_headroom_c"] = None
    row["temp_headroom_why"] = "card 0: nvidia-smi temperature.gpu.tlimit answered [N/A]"
    del row["kernel"]
    row["rdma"] = {"ports": [], "why": "no /sys/class/infiniband on this box"}
    v = run(solo(), 1000, {"solo": row})
    why = {n["kind"]: n["why"] for n in v["no_data"] if n["box"] == "solo"}
    assert "tlimit" in why["gpu_temp"]
    assert "kernel" in why["xid"] and "kernel" in why["oom"]
    assert "infiniband" in why["roce"]
    assert v["findings"] == []


def test_a_stale_box_drops_a_pending_finding_and_keeps_a_firing_one_saying_no_data(feed):
    ev = solo(hold_s=0)
    run(ev, 1000, {"solo": hw(temp_headroom_c=3)})
    # A stale row that still CARRIES its last figures (a hot GPU, nearly no RAM): only the age says they are not now.
    # So the stale branch, and nothing else, keeps them from being judged (review: the test must fail without it).
    stale = {**hw(temp_headroom_c=3, ram_used_mb=131072 - 1024), "age_s": 400.0, "stale": True}
    v = run(ev, 1400, {"solo": stale})
    f = finding(v, "gpu_temp")
    assert f["state"] == "firing" and "no data" in f["why"] and "no sample for 400 s" in f["why"]
    assert finding(v, "low_ram") is None, "a stale RAM figure never opens a finding"
    nd = {n["kind"]: n["why"] for n in v["no_data"] if n["box"] == "solo"}
    assert set(nd) >= {"gpu_temp", "low_ram", "low_power", "xid", "oom", "roce"}
    assert all("no sample for 400 s" in w for w in nd.values()), nd
    feed.drain()
    assert [e["kind"] for e in feed.read()["events"]] == ["finding.firing"], "no clear claimed on no data"
    ev2 = solo()
    run(ev2, 1000, {"solo": hw(temp_headroom_c=3)})
    assert finding(run(ev2, 1010, {"solo": {**stale, "age_s": 410.0}}), "gpu_temp") is None, "pending: dropped"


def test_an_xid_rise_fires_and_clears_once_it_is_older_than_the_window(feed):
    ev = solo(hold_s=0, counter_window_s=600)
    k = lambda xid: {"xid_total": xid, "oom_kills_total": 3, "last_xid": None, "why": None}   # noqa: E731
    assert finding(run(ev, 1000, {"solo": hw(kernel=k(5))}), "xid") is None, "the first reading is a baseline"
    f = finding(run(ev, 1010, {"solo": hw(kernel=k(7))}), "xid")
    assert (f["state"], f["value"], f["threshold"]) == ("firing", 2, 0)
    assert finding(run(ev, 1500, {"solo": hw(kernel=k(7))}), "xid")["state"] == "firing"
    assert finding(run(ev, 1611, {"solo": hw(kernel=k(7))}), "xid") is None
    assert finding(run(ev, 1620, {"solo": hw(kernel=k(1))}), "xid") is None, "a counter that went down is a reboot"


def test_an_oom_kill_rise_fires():
    ev = solo(hold_s=0)
    run(ev, 1000, {"solo": hw()})
    v = run(ev, 1005, {"solo": hw(kernel={"xid_total": 0, "oom_kills_total": 1, "last_xid": None, "why": None})})
    assert finding(v, "oom")["state"] == "firing"


def test_roce_error_counters_rising_on_any_port_fire():
    ev = solo(hold_s=0)
    run(ev, 1000, {"solo": hw()})
    row = hw()
    row["rdma"]["ports"][0]["counters"]["local_ack_timeout_err"] = 4
    f = finding(run(ev, 1005, {"solo": row}), "roce")
    assert (f["state"], f["value"]) == ("firing", 4)


def test_a_port_that_goes_unreadable_and_comes_back_unchanged_is_no_rise():
    """Review r4717113268: a SUM over the counters readable this tick jumped when a port's counters came back — a
    false finding. Each (box, dev, port, counter) is its own: unchanged is unchanged, however it was read between."""
    ev = solo(hold_s=0)
    two = hw()
    two["rdma"]["ports"].append({"dev": "mlx5_1", "port": 1, "state": "ACTIVE",
                                 "counters": {c: 7 for c in ERROR_COUNTERS}, "why": {}})
    assert finding(run(ev, 1000, {"solo": two}), "roce") is None
    gone = hw()
    gone["rdma"]["ports"].append({"dev": "mlx5_1", "port": 1, "state": "DOWN", "counters": {},
                                  "why": {c: "unreadable" for c in ERROR_COUNTERS}})
    assert finding(run(ev, 1005, {"solo": gone}), "roce") is None
    assert finding(run(ev, 1010, {"solo": two}), "roce") is None, "back, unchanged: nothing rose"


def test_any_single_counter_rising_fires_and_names_itself():
    ev = solo(hold_s=0)
    run(ev, 1000, {"solo": hw()})
    flap = hw()
    flap["rdma"]["ports"][0]["counters"]["link_downed"] = 1
    f = finding(run(ev, 1005, {"solo": flap}), "roce")
    assert f["state"] == "firing" and f["value"] == 1 and "mlx5_0/1 link_downed +1" in f["why"]
    # The same, as data: the machine card flags these counters from the finding (hwextra.js), never re-derived.
    assert f["rose"] == [{"dev": "mlx5_0", "port": "1", "counter": "link_downed", "rise": 1}]


def test_only_a_roce_finding_carries_rose():
    ev = solo(hold_s=0)
    run(ev, 1000, {"solo": hw()})
    v = run(ev, 1005, {"solo": hw(kernel={"xid_total": 0, "oom_kills_total": 1, "last_xid": None, "why": None})})
    assert "rose" not in finding(v, "oom")


def test_the_judged_counters_are_ones_the_sender_reads_each_with_the_why_key_the_sender_uses():
    """Review r4717113278: every judged counter is one the sender reads — the port's sysfs COUNTERS, or ethtool's
    PHY_COUNTERS (fleet_telemetry/readers/rdma.py) — so the two cannot drift apart. Each names the `why` key that
    explains it when absent: its own name, or its GROUP's (one reason for a whole hw_counters/ directory or a refused
    ethtool), exactly as the sender keys it. Pause frames are flow control, not faults: never judged."""
    from fleet_telemetry.readers.rdma import COUNTERS, PHY, PHY_COUNTERS
    sent = {name: sub for sub, name in COUNTERS} | {name: PHY for name in PHY_COUNTERS}
    assert set(ERROR_COUNTERS) <= set(sent), set(ERROR_COUNTERS) - set(sent)
    assert ERROR_COUNTERS == {c: sent[c] for c in ERROR_COUNTERS}, "each counter's why-group is the sender's"
    assert {"link_downed", "symbol_error", "port_xmit_discards", "rnr_nak_retry_err", "out_of_buffer",
            "local_ack_timeout_err", "port_rcv_errors", "rx_discards_phy", "tx_discards_phy", "rx_crc_errors_phy",
            "rx_symbol_err_phy"} == set(ERROR_COUNTERS)
    assert not {"rx_pause_ctrl_phy", "tx_pause_ctrl_phy"} & set(ERROR_COUNTERS)


def test_an_ethtool_phy_counter_rising_fires():
    ev = solo(hold_s=0)
    run(ev, 1000, {"solo": hw()})
    row = hw()
    row["rdma"]["ports"][0]["counters"]["rx_crc_errors_phy"] = 12
    f = finding(run(ev, 1005, {"solo": row}), "roce")
    assert f["state"] == "firing" and "rx_crc_errors_phy +12" in f["why"]


@pytest.mark.parametrize("why,says", [
    ({"hw_counters": "no hw_counters/ on this port: the driver states none of it",
      "ethtool": "ethtool -S eth0 exited 94: Operation not supported", "counters": "no counters/ on this port"},
     "no hw_counters/ on this port"),
    ({"ethtool": "no netdev in this namespace (see mtu)", "qos": "no netdev in this namespace (see mtu)",
      "hw_counters": "x", "counters": "y", "mtu": "netdev eth0 is not in this network namespace"}, "x"),
])
def test_a_port_with_no_readable_counter_says_why_from_the_group_keys_the_sender_uses(why, says):
    """The sender keys one reason per GROUP (`why.hw_counters`, `why.ethtool`) where a whole directory or the tool is
    missing: the same reason, read by the key it is filed under — and never another figure's (mtu, pfc, qos)."""
    row = hw()
    row["rdma"] = {"ports": [{"dev": "mlx5_0", "port": 1, "state": "ACTIVE", "counters": {}, "why": why}], "why": None}
    v = run(solo(), 1000, {"solo": row})
    got = next(n["why"] for n in v["no_data"] if n["box"] == "solo" and n["kind"] == "roce")
    assert got.startswith("mlx5_0/1: ") and says in got, got



def backend(label, live, parked=False, cluster=None):
    return SimpleNamespace(host_label=label, url=f"http://{label}:8000", live=live, parked=parked, cluster=cluster)


def test_a_server_is_held_by_backend_down_s_not_by_hold_s():
    """Review: the server hold is its OWN (`backend_down_s`). hold_s 0 must not fire it at once; backend_down_s 0
    must fire it at once even under a long hold_s."""
    ev = solo(hold_s=0, backend_down_s=120)
    run(ev, 1000, {}, [backend("solo-gpu", True)])
    assert finding(run(ev, 1010, {}, [backend("solo-gpu", False)]), "backend_down", "solo-gpu")["state"] == "pending"
    ev = solo(hold_s=3600, backend_down_s=0)
    run(ev, 1000, {}, [backend("solo-gpu", True)])
    assert finding(run(ev, 1010, {}, [backend("solo-gpu", False)]), "backend_down", "solo-gpu")["state"] == "firing"


def test_a_server_that_was_live_and_stops_answering_fires_after_its_own_hold():
    ev = solo()
    run(ev, 1000, {}, [backend("solo-gpu", True)])
    assert finding(run(ev, 1010, {}, [backend("solo-gpu", False)]), "backend_down", "solo-gpu")["state"] == "pending"
    assert finding(run(ev, 1130, {}, [backend("solo-gpu", False)]), "backend_down", "solo-gpu")["state"] == "firing"
    assert finding(run(ev, 1140, {}, [backend("solo-gpu", True)]), "backend_down", "solo-gpu") is None


@pytest.mark.parametrize("why,rows,broker", [
    ("never live since the LB started: a resting server", [[backend("a", False)], [backend("a", False)]], False),
    ("parked by the operator", [[backend("a", True)], [backend("a", False, parked=True)]], False),
    ("a cluster head: its cluster's state says it", [[backend("a", True, cluster="pair")],
                                                     [backend("a", False, cluster="pair")]], False),
    ("with the broker on, a lease decides when a server runs", [[backend("a", True)], [backend("a", False)]], True),
])
def test_a_server_down_by_design_is_no_finding(why, rows, broker):
    ev = solo(hold_s=0, backend_down_s=0)
    run(ev, 1000, {}, rows[0], broker)
    assert finding(run(ev, 1010, {}, rows[1], broker), "backend_down", "a") is None, why


# ── disk: a configured disk path at or over the fleet's disk_full_pct ───────────────────────────────────────────────

def test_a_full_disk_is_pending_then_firing_naming_every_full_path_and_clears(feed):
    """DISK: a configured disk path at or over `disk_full_pct`, judged on the WHOLE percent the panel
    prints (89.6 % reads "90%", so it is full at 90) — held `hold_s` like every other rule, alerted on firing. The
    finding names each full path (`full`), so the strip's red and the band's chip are this ONE judgement."""
    ev = solo(hold_s=60, disk_full_pct=90)
    rows = lambda *ds: {"solo": hw(disks=list(ds))}                   # noqa: E731
    v = run(ev, 1000, rows(disk("/", 40), disk("/data", 89.6), disk("/scratch", 95)))
    f = finding(v, "disk")
    assert f["state"] == "pending" and f["value"] == 95 and f["threshold"] == 90 and f["unit"] == "%"
    assert f["full"] == [{"path": "/data", "pct": 90}, {"path": "/scratch", "pct": 95}], f["full"]
    assert "/scratch 95 %" in f["why"] and "≥ 90 %" in f["why"], f["why"]
    f = finding(run(ev, 1060, rows(disk("/", 40), disk("/data", 89.6), disk("/scratch", 95))), "disk")
    assert f["state"] == "firing"
    feed.drain()
    assert [e["kind"] for e in feed.read()["events"] if e.get("finding") == "disk"] == ["finding.firing"]
    assert finding(run(ev, 1070, rows(disk("/", 40), disk("/data", 89.4))), "disk") is None, "89 % < 90: cleared"
    feed.drain()
    kinds = [e["kind"] for e in feed.read()["events"] if e.get("finding") == "disk"]
    assert kinds == ["finding.cleared", "finding.firing"]


@pytest.mark.parametrize("row,says", [
    ({"disks": [], "disks_why": "the sender sends no disks block (it predates box health)"}, "predates box health"),
    ({"disks": [], "disks_why": None}, "no disk path"),
    ({"disks": [{"path": "/gone", "total_bytes": None, "used_bytes": None, "free_bytes": None,
                 "why": "statvfs /gone: No such file"}], "disks_why": None}, "/gone: statvfs /gone: No such file"),
])
def test_a_box_with_no_readable_disk_is_no_data_with_why(row, says):
    v = run(solo(hold_s=0), 1000, {"solo": hw(**row)})
    assert finding(v, "disk") is None
    nd = [n for n in v["no_data"] if n["kind"] == "disk"]
    assert len(nd) == 1 and says in nd[0]["why"], nd


def test_an_unread_disk_beside_a_full_one_still_fires():
    v = run(solo(hold_s=0), 1000, {"solo": hw(disks=[{"path": "/gone", "total_bytes": None, "used_bytes": None,
                                                      "free_bytes": None, "why": "x"}, disk("/data", 97)])})
    assert finding(v, "disk")["full"] == [{"path": "/data", "pct": 97}]


def test_the_view_is_off_with_why_without_a_policy():
    ev = HealthEvaluator(None)
    ev.evaluate(1000, {"solo": hw(temp_headroom_c=0)}, [], broker_enabled=False)
    v = ev.view()
    assert v["enabled"] is False and "health:" in v["why"] and v["findings"] == []
    assert "disk_full_pct" not in v, "health off: no threshold, so the panel flags no disk full (hwextra.js says why)"


def test_a_full_disk_at_100_pct_is_allowed_and_the_view_carries_the_threshold():
    """The panel's "disk full" flag (hwextra.js) is the fleet's `disk_full_pct`, read from /lb/status `health` — one
    definition, the operator's, never a number of the panel's own. 100 is allowed (flag only a disk that is full)."""
    assert parse_policy("x", {**BLOCK, "disk_full_pct": 100}).disk_full_pct == 100.0
    v = run(solo(disk_full_pct=85), 1000, {})
    assert v["enabled"] is True and v["disk_full_pct"] == 85.0


# ── the monitor and /lb/status ──────────────────────────────────────────────────────────────────────────────────────

def test_the_monitor_judges_the_listeners_samples_and_status_carries_findings_and_the_alert_sender():
    from vllm_lb import health
    from vllm_lb.alerts import AlertSender
    rows = {"solo": hw(temp_headroom_c=3)}
    mon = health.HealthMonitor(solo(hold_s=0), hardware=lambda: rows, backends=list, broker_enabled=False,
                               period_s=1.0, clock=lambda: 1000.0)
    mon.tick()
    health.install(mon, AlertSender(()))
    try:
        v = health.status_view()
    finally:
        health.install(None)
    assert v["enabled"] and v["findings"][0]["kind"] == "gpu_temp" and v["findings"][0]["state"] == "firing"
    assert v["disk_full_pct"] == 90.0, "/lb/status carries the fleet's disk threshold for the panel"
    assert v["alerts"]["channels"] == [] and "LB_ALERT_" in v["alerts"]["why"]
    assert health.status_view()["enabled"] is False


def test_a_tick_that_raises_is_counted_and_shown_never_raised():
    from vllm_lb import health
    mon = health.HealthMonitor(solo(), hardware=lambda: 1 / 0, backends=list, broker_enabled=False, period_s=1.0)
    mon.tick()
    assert mon.view()["errors"] == 1 and "ZeroDivisionError" in mon.view()["last_error"]
