"""The per-box FLIGHT RECORDER (vllm_lb/recorder.py): one hour of every recorder figure a box broadcasts.

The figures are broker_parrot's `hw_watch.deep_sample()` — the persisted hardware flight recorder that caught a box
asserting a hardware-slowdown throttle mask seconds before a firmware power cut. The LB keeps the same figures off the
LAN broadcast, in memory, for one hour, two-tier (fine recent samples, coarse minute buckets), so a box's last hour
can be read without a database. The rules pinned here:

  * a figure the box does not send stays ABSENT with the reason — never 0, never a guess. The 1.5.0 sender broadcasts
    no `deep` block at all, and must keep working: CPU util is its `cpu_percent`, everything else says why it is n/a;
  * network throughput is the delta of MONOTONIC counters; a counter that went backwards (reboot, NIC reset) or a
    first sample is a gap, not a zero (hw_watch `net_rates`);
  * memory is bounded: the fine tier keeps minutes, the coarse tier one bucket per minute, whatever the sender's rate;
  * the peak and the average of the GPU power are the hour's, kept through staleness (history is what it DID);
  * containers are the latest list: count, total RAM, the top N by RAM and how many more.
"""
from __future__ import annotations

import http.client
import json

import pytest

from vllm_lb.config import Config
from vllm_lb.fleet import parse_fleet
from vllm_lb.hw import HwCollector
from vllm_lb.recorder import (BUCKET_S, FINE_S, FIGURES, TOP_CONTAINERS, WINDOW_S, FlightRecorder,
                              figures_of)

FLEET = parse_fleet("""
panel:
  style: vibe
boxes:
  box-a: {hostname: box-a-host, platform: linux/arm64, cards: [{index: 0, type: gb10, serves: false}]}
  site-b: {hostname: site-b-host, platform: linux/amd64, cards: [{index: 0, type: rtx3090, serves: false},
                                                                {index: 1, type: rtx5090, serves: false}]}
clusters: {}
backends: {}
""", "fleet.yml")

T0 = 100_000.0

#: The 1.5.0 sender's sample, as it is on the wire today: no `deep`, no `containers`.
OLD = {"host": "box-a-host", "cpu_percent": 5.9, "ram_percent": 40.0, "ram_used_mb": 50000,
       "ram_total_mb": 124610, "swap_used_mb": 0,
       "gpus": [{"id": 0, "use_pct": 3, "vram_used_mb": 0, "vram_total_mb": 0}]}


def deep(**over):
    """A `deep` block in hw_watch.deep_sample()'s shape (one GPU, two thermal zones, no fan tach)."""
    d = {"gpu": [{"temp_c": 44, "power_w": 9.0, "power_inst_w": 12.5, "sm_mhz": 2190, "util_pct": 3,
                  "pstate": "P0", "throttle_hex": "0x0000000000000000", "throttle_index": 0,
                  "throttle_reasons": [], "fan_pct": None}],
         "tz": [{"type": "acpitz", "milli_c": 47000}, {"type": "acpitz", "milli_c": 41500}],
         "fan": [], "fan_rpm_max": None, "gpu_fan_pct": None, "throttle_index": 0,
         "cpu": {"percent": 6.0, "freq_mhz": 3354, "temp_c": 47.0, "throttle_count": None, "ncpu": 20},
         "net_rx_bytes": 1_000_000, "net_tx_bytes": 2_000_000, "net_speed_mbps": 10000}
    d.update(over)
    return d


def sample(**over):
    s = dict(OLD, deep=deep())
    s.update(over)
    return s


CONTAINERS = [{"name": f"c{i}", "image": f"img/{i}:latest", "ram_mb": 100 * (i + 1)} for i in range(8)]


# ── what a figure is ──────────────────────────────────────────────────────────

def test_figures_read_hw_watch_deep_sample_fields():
    figs, why = figures_of(sample(), None, None)
    assert figs["gpu_temp_c"] == 44
    assert figs["board_temp_c"] == 47.0, "the hottest thermal zone, milli-°C to °C"
    assert figs["cpu_temp_c"] == 47.0
    assert figs["gpu_power_w"] == 9.0 and figs["gpu_power_inst_w"] == 12.5
    assert figs["throttle"] == 0
    assert figs["gpu_mhz"] == 2190 and figs["cpu_mhz"] == 3354
    assert figs["cpu_pct"] == 5.9, "CPU util is the broadcast's own cpu_percent"
    assert figs["fan_rpm"] is None and "fan" in why["fan_rpm"], "no tach: n/a with why, never 0"


def test_an_old_sender_keeps_working_cpu_util_live_everything_else_says_why():
    figs, why = figures_of(OLD, None, None)
    assert figs["cpu_pct"] == 5.9
    for key, *_ in FIGURES:
        if key != "cpu_pct":
            assert figs[key] is None, key
            assert "deep" in why[key], f"{key}: the reason names the missing block"


def test_two_cards_power_sums_temp_is_the_hottest_clock_the_slowest_throttle_the_worst():
    g = deep()["gpu"][0]
    two = deep(gpu=[dict(g, temp_c=60, power_w=200.0, power_inst_w=210.0, sm_mhz=1800, throttle_index=0),
                    dict(g, temp_c=70, power_w=300.0, power_inst_w=320.0, sm_mhz=1500, throttle_index=85)],
               throttle_index=85)
    figs, _ = figures_of(sample(deep=two), None, None)
    assert figs["gpu_power_w"] == 500.0 and figs["gpu_power_inst_w"] == 530.0
    assert figs["gpu_temp_c"] == 70 and figs["gpu_mhz"] == 1500 and figs["throttle"] == 85


def test_a_card_without_a_power_reading_withholds_the_box_sum():
    g = deep()["gpu"][0]
    two = deep(gpu=[dict(g, power_w=200.0), dict(g, power_w=None)])
    figs, why = figures_of(sample(deep=two), None, None)
    assert figs["gpu_power_w"] is None, "a partial sum would be a number the box never drew"
    assert why["gpu_power_w"]


def test_nan_and_booleans_are_not_readings():
    figs, _ = figures_of(sample(cpu_percent=float("nan"), deep=deep(cpu={"temp_c": True, "freq_mhz": "fast"})),
                         None, None)
    assert figs["cpu_pct"] is None and figs["cpu_temp_c"] is None and figs["cpu_mhz"] is None


# ── network: a delta of monotonic counters ────────────────────────────────────

def test_net_rate_is_the_counter_delta_in_megabits_and_a_first_sample_is_a_gap():
    r = FlightRecorder()
    r.ingest("box-a", sample(), T0)
    s = r.summary("box-a", T0, stale=False, age_s=0)
    assert s["figures"]["net_in_mbps"] is None and "first" in s["why"]["net_in_mbps"]
    r.ingest("box-a", sample(deep=deep(net_rx_bytes=1_000_000 + 625_000, net_tx_bytes=2_000_000)), T0 + 5)
    s = r.summary("box-a", T0 + 5, stale=False, age_s=0)
    assert s["figures"]["net_in_mbps"] == pytest.approx(1.0), "625 kB in 5 s = 1 Mb/s"
    assert s["figures"]["net_out_mbps"] == 0


def test_a_counter_that_went_backwards_is_a_gap_not_a_spike():
    r = FlightRecorder()
    r.ingest("box-a", sample(), T0)
    r.ingest("box-a", sample(deep=deep(net_rx_bytes=10, net_tx_bytes=10)), T0 + 5)
    s = r.summary("box-a", T0 + 5, stale=False, age_s=0)
    assert s["figures"]["net_in_mbps"] is None and "backwards" in s["why"]["net_in_mbps"]


# ── the two tiers, bounded ────────────────────────────────────────────────────

def test_memory_is_bounded_whatever_the_senders_rate():
    r = FlightRecorder()
    for i in range(3 * WINDOW_S):                       # three hours at one sample a second
        r.ingest("box-a", sample(), T0 + i)
    b = r._boxes["box-a"]
    assert len(b.coarse) <= WINDOW_S // BUCKET_S + 1
    assert len(b.fine) <= FINE_S + 1
    assert all(t >= T0 + 3 * WINDOW_S - 1 - FINE_S for t, _ in b.fine)


def test_series_grids_the_hour_in_minute_buckets_and_the_recent_minutes_finely():
    r = FlightRecorder()
    for i in range(0, 1800, 5):                         # half an hour at the sender's 5 s
        r.ingest("box-a", sample(deep=deep(gpu=[dict(deep()["gpu"][0], power_w=float(i % 60))])), T0 + i)
    now = T0 + 1795
    out = r.series("box-a", now, stale=False, age_s=0)
    assert out["window_s"] == WINDOW_S and out["bucket_s"] == BUCKET_S and out["fine_s"] == FINE_S
    coarse = out["coarse"]
    assert len(coarse["t"]) == WINDOW_S // BUCKET_S
    power = coarse["series"]["gpu_power_w"]
    assert power["mean"][:29] == [None] * 29, "the half hour before the first sample is a gap, not a zero"
    assert power["max"][-2] == 55.0 and power["mean"][-2] == pytest.approx(27.5)
    fine = out["fine"]
    assert len(fine["t"]) == len(fine["series"]["gpu_power_w"])
    assert fine["t"][0] >= now - FINE_S and fine["t"][-1] == now


# ── peak and average over the hour ────────────────────────────────────────────

def test_peak_inst_peak_avg_and_the_hour_mean():
    r = FlightRecorder()
    for i, (w, inst) in enumerate([(9.0, 12.0), (34.0, 29.0), (20.0, 22.0)]):
        r.ingest("box-a", sample(deep=deep(gpu=[dict(deep()["gpu"][0], power_w=w, power_inst_w=inst)])),
                 T0 + 5 * i)
    p = r.summary("box-a", T0 + 10, stale=False, age_s=0)["power"]
    assert p["peak_inst_w"] == 29.0 and p["peak_avg_w"] == 34.0
    assert p["mean_w"] == pytest.approx(21.0)
    assert p["input_rail_w"] is None and p["input_rail_why"]


def test_samples_older_than_the_hour_leave_the_peak():
    r = FlightRecorder()
    r.ingest("box-a", sample(deep=deep(gpu=[dict(deep()["gpu"][0], power_w=300.0)])), T0)
    r.ingest("box-a", sample(), T0 + WINDOW_S + 2 * BUCKET_S)
    p = r.summary("box-a", T0 + WINDOW_S + 2 * BUCKET_S, stale=False, age_s=0)["power"]
    assert p["peak_avg_w"] == 9.0


def test_an_old_sender_has_no_power_peak_and_says_why():
    r = FlightRecorder()
    r.ingest("box-a", OLD, T0)
    p = r.summary("box-a", T0, stale=False, age_s=0)["power"]
    assert p["peak_inst_w"] is None and p["peak_avg_w"] is None and "deep" in p["why"]


def test_stale_withdraws_the_now_figures_and_keeps_the_hour():
    r = FlightRecorder()
    r.ingest("box-a", sample(containers=CONTAINERS), T0)
    s = r.summary("box-a", T0 + 120, stale=True, age_s=120)
    assert all(v is None for v in s["figures"].values())
    assert "120" in s["why"]["gpu_temp_c"]
    assert s["power"]["peak_avg_w"] == 9.0, "the hour is what it did; it stays"
    assert s["containers"] is None and "120" in s["containers_why"]


# ── containers ────────────────────────────────────────────────────────────────

def test_containers_count_total_ram_top_n_and_how_many_more():
    r = FlightRecorder()
    r.ingest("box-a", sample(containers=CONTAINERS), T0)
    c = r.summary("box-a", T0, stale=False, age_s=0)["containers"]
    assert c["count"] == 8 and c["total_ram_mb"] == sum(100 * (i + 1) for i in range(8))
    assert [x["name"] for x in c["top"]] == [f"c{i}" for i in range(7, 7 - TOP_CONTAINERS, -1)], "by RAM, largest first"
    assert c["more"] == 8 - TOP_CONTAINERS
    full = r.series("box-a", T0, stale=False, age_s=0)["containers"]
    assert len(full["all"]) == 8, "the endpoint carries every container"


def test_a_container_without_ram_withholds_the_total_and_malformed_entries_are_dropped():
    r = FlightRecorder()
    r.ingest("box-a", sample(containers=[{"name": "a", "image": "x", "ram_mb": 100},
                                         {"name": "b", "image": "y", "ram_mb": None},
                                         {"image": "no-name"}, "junk", {"name": "c", "ram_mb": float("inf")}]), T0)
    c = r.summary("box-a", T0, stale=False, age_s=0)["containers"]
    assert c["count"] == 2
    assert c["total_ram_mb"] is None and c["total_why"]


def test_a_sender_without_containers_says_so():
    r = FlightRecorder()
    r.ingest("box-a", OLD, T0)
    s = r.summary("box-a", T0, stale=False, age_s=0)
    assert s["containers"] is None and "containers" in s["containers_why"]


def test_an_unknown_box_has_no_recorder():
    assert FlightRecorder().summary("nope", T0, stale=False, age_s=0) is None
    assert FlightRecorder().series("nope", T0, stale=False, age_s=0) is None


# ── through the collector and the API ─────────────────────────────────────────

def _collector():
    return HwCollector(Config(hw_hosts=dict(FLEET.hw_hosts), fleet=FLEET))


def test_the_collector_records_every_sample_and_status_carries_the_summary():
    c = _collector()
    c.ingest(json.dumps(sample(containers=CONTAINERS)), now=T0)
    snap = c.snapshot(now=T0 + 1)["box-a"]
    rec = snap["recorder"]
    assert rec["figures"]["gpu_temp_c"] == 44
    assert rec["containers"]["count"] == 8
    assert "coarse" not in rec and "fine" not in rec, "the hour's series is the endpoint's, not every poll's"


def test_the_recorder_endpoint_serves_the_hour_and_names_what_is_missing():
    from vllm_lb.hub import Hub
    from vllm_lb.proxy import Proxy
    cfg = Config(hw_hosts=dict(FLEET.hw_hosts), fleet=FLEET, listen_host="127.0.0.1", listen_port=0)
    hw = HwCollector(cfg)
    hw.ingest(json.dumps(sample(containers=CONTAINERS)))
    proxy = Proxy(cfg, Hub(cfg), hw=hw)
    proxy.start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=10)
        conn.request("GET", "/lb/hardware/box-a/recorder")
        resp = conn.getresponse()
        body = json.loads(resp.read())
        assert resp.status == 200
        assert body["box"] == "box-a" and "coarse" in body and "fine" in body
        assert len(body["containers"]["all"]) == 8
        conn.request("GET", "/lb/hardware/nowhere/recorder")
        resp = conn.getresponse()
        missing = json.loads(resp.read())
        assert resp.status == 404 and "nowhere" in missing["error"]
    finally:
        proxy.stop()


def test_the_recorder_endpoint_is_a_503_without_a_listener():
    from vllm_lb.hub import Hub
    from vllm_lb.proxy import Proxy
    cfg = Config(hw_hosts=dict(FLEET.hw_hosts), fleet=FLEET, listen_host="127.0.0.1", listen_port=0)
    proxy = Proxy(cfg, Hub(cfg), hw=None)
    proxy.start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=10)
        conn.request("GET", "/lb/hardware/box-a/recorder")
        resp = conn.getresponse()
        body = json.loads(resp.read())
        assert resp.status == 503 and body["error"]
    finally:
        proxy.stop()


# ── what a capped sender adds (lb/fleet_telemetry: containers_total, deep.why) ─

def test_a_capped_list_counts_every_container_from_the_senders_total():
    r = FlightRecorder()
    r.ingest("box-a", sample(containers=CONTAINERS[:3], containers_total={"count": 140, "ram_mb": 90_000}), T0)
    c = r.summary("box-a", T0, stale=False, age_s=0)["containers"]
    assert (c["count"], c["total_ram_mb"], c["unlisted"]) == (140, 90_000, 137)


@pytest.mark.parametrize("total", [{"count": -1, "ram_mb": 1}, {"count": True, "ram_mb": 1}, {"count": 9, "ram_mb": "x"},
                                   "junk", None])
def test_a_malformed_total_is_ignored_and_the_list_is_what_is_known(total):
    r = FlightRecorder()
    r.ingest("box-a", sample(containers=CONTAINERS, containers_total=total), T0)
    c = r.summary("box-a", T0, stale=False, age_s=0)["containers"]
    assert (c["count"], c["unlisted"]) == (8, 0)


def test_the_senders_reason_replaces_the_generic_one_bounded():
    figs, why = figures_of(sample(deep=deep(fan_rpm_max=None, why={"fan_rpm": "x" * 1000, "gpu_temp_c": 7})), None, None)
    assert why["fan_rpm"] == "x" * 240, "a sender's reason is text, bounded"
    assert "gpu_temp_c" not in why, "a reason for a figure that IS there is not shown"


def test_a_sender_that_could_not_read_its_containers_says_why():
    r = FlightRecorder()
    r.ingest("box-a", sample(containers_why="the containers could not be read: EACCES"), T0)
    s = r.summary("box-a", T0, stale=False, age_s=0)
    assert s["containers"] is None and "EACCES" in s["containers_why"]


# ── what the panel needs to say a figure plainly (recorder UX, 2026-10-30) ────

def test_every_figure_carries_its_format_from_the_one_table():
    """The panel formats a figure by the table's `fmt` (recorder.js `recFmt`), never by guessing from a unit."""
    r = FlightRecorder()
    r.ingest("box-a", sample(), T0)
    labels = r.summary("box-a", T0, stale=False, age_s=0)["labels"]
    assert [lab["key"] for lab in labels] == [k for k, *_ in FIGURES]
    assert {lab["fmt"] for lab in labels} <= {"int", "rate", "pct", "flags"}
    fmt = {lab["key"]: lab["fmt"] for lab in labels}
    assert fmt["net_in_mbps"] == "rate" and fmt["gpu_mhz"] == "int" and fmt["throttle"] == "flags"


def test_every_label_fits_a_phone_tile():
    """"BOARD ZONE MAX" read "BOARD ZON…" at desktop and phone: a label is at most 10 characters."""
    assert all(len(label) <= 10 for _k, label, *_ in FIGURES), [lab for _k, lab, *_ in FIGURES if len(lab) > 10]


def test_the_throttle_mask_is_decoded_into_its_reasons_union_over_cards():
    g = deep()["gpu"][0]
    two = deep(gpu=[dict(g, throttle_hex="0x0000000000000004"), dict(g, throttle_hex="0x48")], throttle_index=100)
    r = FlightRecorder()
    r.ingest("box-a", sample(deep=two), T0)
    assert r.summary("box-a", T0, stale=False, age_s=0)["throttle_reasons"] == [
        "sw_power_cap", "hw_slowdown", "hw_thermal"]
    r.ingest("box-a", sample(), T0 + 2)
    assert r.summary("box-a", T0 + 2, stale=False, age_s=0)["throttle_reasons"] == [], "mask 0: none"
    assert r.summary("box-a", T0 + 200, stale=True, age_s=198)["throttle_reasons"] is None, "stale: not now"
    amd = deep(gpu=[dict(g, throttle_hex=None, throttle_index=None)], throttle_index=None)
    r.ingest("box-a", sample(deep=amd), T0 + 4)
    assert r.summary("box-a", T0 + 4, stale=False, age_s=0)["throttle_reasons"] is None, "no mask: absent"


def test_one_power_reading_is_said_so_and_two_are_kept_apart():
    """An amdgpu APU's averaged and 'instantaneous' draw are the same sensor: one figure, not two peaks of it."""
    g = deep()["gpu"][0]
    r = FlightRecorder()
    for i, w in enumerate((60.0, 119.0, 80.0)):
        r.ingest("apu", sample(deep=deep(gpu=[dict(g, power_w=w, power_inst_w=w, power_label="PPT")])), T0 + 2 * i)
    p = r.summary("apu", T0 + 4, stale=False, age_s=0)["power"]
    assert p["one_reading"] is True and p["peak_avg_w"] == 119.0 and p["mean_w"] == pytest.approx(86.333, rel=1e-3)
    assert p["scope"] == "package", "amdgpu's PPT is the whole APU package, CPU and GPU"
    r.ingest("dgpu", sample(), T0)
    q = r.summary("dgpu", T0, stale=False, age_s=0)["power"]
    assert q["one_reading"] is False and q["scope"] is None
