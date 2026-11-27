"""PERSISTED HARDWARE HISTORY TIERS (vllm_lb/hw_tiers.py): 48 h @ 1 min and 30 d @ 15 min of coarse
min/avg/max per box per figure, fed by the hardware listener, saved with atomic.write_json on a slow cadence and
loaded at start.

Pinned here:
  * a bucket keeps min, max and the mean of what fell in it; an absent figure adds nothing (never a 0);
  * each tier keeps only its window (bounded memory whatever the age of the gateway);
  * a save then a load round-trips every bucket;
  * a CORRUPT file is not a crash and not a silent reset: the tiers start empty, the status says `corrupt` with the
    parser's reason, and the bad file is set aside (renamed) so the next save does not destroy the evidence;
  * the recorder (recorder.py) is untouched: the collector feeds both;
  * GET /lb/history?box=&metric=&tier= serves the points; a bad parameter is a 400 naming the choices.
"""
from __future__ import annotations

import http.client
import json
import os

import pytest

from tests.fleet_files import PAIR
from vllm_lb.config import Config
from vllm_lb.fleet import parse_fleet
from vllm_lb.hw import HwCollector
from vllm_lb.hw_tiers import METRICS, TIERS, HwTiers

FLEET = parse_fleet(PAIR, "fleet.yml")
T0 = 1_795_176_000.0          # a whole minute and a whole 15 minutes


def sample(temp=50, power=100.0, use=40, rx=1_000_000, tx=2_000_000, host="node-1-host", stamp=None):
    s = {"host": host, "cpu_percent": 5.0, "ram_percent": 40.0, "ram_used_mb": 50_000, "ram_total_mb": 120_000,
         "gpus": [{"id": 0, "use_pct": use, "vram_used_mb": 0, "vram_total_mb": 0}],
         "deep": {"gpu": [{"temp_c": temp, "power_w": power, "power_inst_w": power, "sm_mhz": 2000,
                           "throttle_index": 0}],
                  "net_rx_bytes": rx, "net_tx_bytes": tx}}
    if stamp is not None:
        s["sampled_at"] = stamp
    return s


def test_a_bucket_keeps_min_avg_max_and_an_absent_figure_adds_nothing(tmp_path):
    t = HwTiers(str(tmp_path / "t.json"))
    t.ingest("box", {"gpu_temp_c": 40, "gpu_pct": None}, T0 + 1)
    t.ingest("box", {"gpu_temp_c": 60, "gpu_pct": None}, T0 + 30)
    out = t.series("box", "gpu_temp_c", "48h", now=T0 + 59)
    assert out["points"] == [[T0, 40, 50.0, 60]]
    assert t.series("box", "gpu_pct", "48h", now=T0 + 59)["points"] == []
    assert t.series("box", "gpu_temp_c", "30d", now=T0 + 59)["points"] == [[T0, 40, 50.0, 60]]


def test_each_tier_keeps_only_its_window(tmp_path):
    t = HwTiers(str(tmp_path / "t.json"))
    t.ingest("box", {"gpu_temp_c": 1}, T0)
    later = T0 + TIERS["48h"][1] + 120
    t.ingest("box", {"gpu_temp_c": 2}, later)
    assert [p[1] for p in t.series("box", "gpu_temp_c", "48h", now=later)["points"]] == [2]
    assert [p[1] for p in t.series("box", "gpu_temp_c", "30d", now=later)["points"]] == [1, 2]


def test_the_48h_tier_is_exactly_48_hours_on_read_and_on_ingest(tmp_path):
    """A bucket 47 h old is IN the 48 h tier and one 49 h old is OUT — read at a later `now` with no new sample (the
    read's own window) and after a new sample (the ingest's prune). A window of 24 h or of 30 d fails one side."""
    bucket_s, window_s = TIERS["48h"]
    assert (bucket_s, window_s) == (60, 48 * 3600)
    t = HwTiers(str(tmp_path / "t.json"))
    now = T0 + 49 * 3600
    t.ingest("box", {"gpu_temp_c": 1}, T0)                    # 49 h before `now`
    t.ingest("box", {"gpu_temp_c": 2}, T0 + 2 * 3600)         # 47 h before `now`
    t.ingest("box", {"gpu_temp_c": 4}, T0 + 19 * 3600)        # 30 h before `now`
    assert [p[1] for p in t.series("box", "gpu_temp_c", "48h", now=now)["points"]] == [2, 4], "the read's window"
    assert [p[1] for p in t.series("box", "gpu_temp_c", "30d", now=now)["points"]] == [1, 2, 4]
    t.ingest("box", {"gpu_temp_c": 3}, now)
    assert [p[1] for p in t.series("box", "gpu_temp_c", "48h", now=now)["points"]] == [2, 4, 3]
    with t._lock:
        assert sorted(t._t["48h"]["box"]["gpu_temp_c"]) == [T0 + 2 * 3600, T0 + 19 * 3600, now], \
            "the ingest prunes the 49 h bucket"


def test_bounded_whatever_the_rate(tmp_path):
    t = HwTiers(str(tmp_path / "t.json"))
    for i in range(0, 3 * 86400, 30):
        t.ingest("box", {"gpu_temp_c": 50}, T0 + i)
    n48 = len(t.series("box", "gpu_temp_c", "48h", now=T0 + 3 * 86400)["points"])
    assert n48 <= TIERS["48h"][1] // TIERS["48h"][0] + 1


def test_save_then_load_round_trips(tmp_path):
    path = str(tmp_path / "t.json")
    t = HwTiers(path)
    t.ingest("box", {"gpu_temp_c": 40, "net_in_mbps": 1.5}, T0 + 1)
    t.save(now=T0 + 2)
    u = HwTiers(path)
    u.load()
    assert u.status()["state"] == "loaded"
    assert u.series("box", "gpu_temp_c", "48h", now=T0 + 3) == t.series("box", "gpu_temp_c", "48h", now=T0 + 3)
    assert u.series("box", "net_in_mbps", "30d", now=T0 + 3)["points"] == [[T0, 1.5, 1.5, 1.5]]


def test_no_file_yet_is_empty_and_said(tmp_path):
    t = HwTiers(str(tmp_path / "none.json"))
    t.load()
    st = t.status()
    assert st["state"] == "empty" and "no file" in st["why"]


@pytest.mark.parametrize("content", ["{not json", json.dumps({"v": 99}), json.dumps({"v": 1, "tiers": []})])
def test_a_corrupt_file_starts_empty_says_why_and_is_set_aside(tmp_path, content, caplog):
    path = tmp_path / "t.json"
    path.write_text(content)
    t = HwTiers(str(path))
    t.load()
    st = t.status()
    assert st["state"] == "corrupt" and st["why"]
    assert "set aside" in st["why"]
    assert not path.exists(), "the bad file is renamed, so the next save cannot overwrite the evidence"
    assert any(p.name.startswith("t.json.corrupt-") for p in tmp_path.iterdir())
    assert any("corrupt" in r.getMessage() for r in caplog.records)
    t.ingest("box", {"gpu_temp_c": 1}, T0)
    t.save(now=T0)
    assert t.status()["state"] == "corrupt", "the reason stays until the gateway restarts"
    assert t.status()["saved_at"] == T0


def test_a_failed_save_is_said_not_raised(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    t = HwTiers(str(blocker / "sub" / "t.json"))
    t.ingest("box", {"gpu_temp_c": 1}, T0)
    t.save(now=T0)
    assert t.status()["save_error"]


def test_off_when_no_path():
    t = HwTiers("")
    t.load()
    assert t.status()["state"] == "off"


def test_unknown_box_metric_or_tier():
    t = HwTiers("")
    assert t.series("box", "gpu_temp_c", "48h", now=T0) is None
    with pytest.raises(KeyError):
        t.series("box", "nope", "48h", now=T0)
    with pytest.raises(KeyError):
        t.series("box", "gpu_temp_c", "1y", now=T0)


# ── fed by the collector ───────────────────────────────────────────────────────────────────────

def test_the_collector_feeds_temp_power_util_mem_and_net_rates(tmp_path):
    tiers = HwTiers(str(tmp_path / "t.json"))
    hw = HwCollector(Config(hw_hosts=dict(FLEET.hw_hosts), fleet=FLEET), tiers=tiers)
    hw.ingest(json.dumps(sample(temp=50, rx=1_000_000, stamp="a")), now=T0 + 1)
    hw.ingest(json.dumps(sample(temp=70, rx=2_000_000, stamp="b")), now=T0 + 2)
    hw.ingest(json.dumps(sample(temp=99, rx=2_000_000, stamp="b")), now=T0 + 3)   # a duplicate: one sample
    pts = lambda m: tiers.series("node-1", m, "48h", now=T0 + 5)["points"]   # noqa: E731
    assert pts("gpu_temp_c") == [[T0, 50, 60.0, 70]]
    assert pts("gpu_power_w") == [[T0, 100.0, 100.0, 100.0]]
    assert pts("gpu_pct") == [[T0, 40, 40.0, 40]]
    assert pts("net_in_mbps") == [[T0, 8.0, 8.0, 8.0]], "the counter delta, the first sample a gap"
    assert set(METRICS) >= {"gpu_temp_c", "gpu_power_w", "gpu_pct", "mem_used_mb", "net_in_mbps", "net_out_mbps"}
    assert hw.snapshot(now=T0 + 3)["node-1"]["recorder"]["figures"]["gpu_temp_c"] == 70, "the recorder still fed"


# ── the endpoint ───────────────────────────────────────────────────────────────────────────────

def _get(port, path):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request("GET", path)
    r = c.getresponse()
    return r.status, json.loads(r.read())


def test_the_endpoint(tmp_path):
    import time
    from vllm_lb.hub import Hub
    from vllm_lb.proxy import Proxy
    cfg = Config(hw_hosts=dict(FLEET.hw_hosts), fleet=FLEET, listen_host="127.0.0.1", listen_port=0)
    tiers = HwTiers(str(tmp_path / "t.json"))
    tiers.load()
    hw = HwCollector(cfg, tiers=tiers)
    hw.ingest(json.dumps(sample()), now=time.time())
    p = Proxy(cfg, Hub(cfg), hw=hw)
    p.start()
    try:
        status, body = _get(p.port, "/lb/history?box=node-1&metric=gpu_temp_c&tier=48h")
        assert status == 200 and body["bucket_s"] == 60 and len(body["points"]) == 1
        assert body["label"] and body["unit"] == "°C"
        status, body = _get(p.port, "/lb/history?box=node-1&tier=30d")
        assert status == 200 and set(body["series"]) == set(METRICS)
        status, body = _get(p.port, "/lb/history?box=node-1&metric=gpu_temp_c&tier=1y")
        assert status == 400 and "48h" in body["error"]["message"]
        status, body = _get(p.port, "/lb/history?box=nowhere&tier=48h")
        assert status == 404
        status, body = _get(p.port, "/lb/status")
        assert body["hardware_tiers"]["state"] == "empty"
    finally:
        p.stop()


def test_the_endpoint_without_tiers_is_a_503_naming_why():
    from vllm_lb.hub import Hub
    from vllm_lb.proxy import Proxy
    cfg = Config(hw_hosts=dict(FLEET.hw_hosts), fleet=FLEET, listen_host="127.0.0.1", listen_port=0)
    p = Proxy(cfg, Hub(cfg), hw=HwCollector(cfg))
    p.start()
    try:
        status, body = _get(p.port, "/lb/history?box=node-1&tier=48h")
        assert status == 503 and "LB_HW_TIERS_FILE" in body["error"]["message"]
    finally:
        p.stop()


def test_the_file_setting():
    env = {"LB_ALLOWED_HOSTS": "loopback"}
    assert Config.from_env(env).hw_tiers_file == "/data/hw-tiers.json"
    assert Config.from_env({**env, "LB_HW_TIERS_FILE": ""}).hw_tiers_file == ""
    assert os.path.basename(Config.from_env({**env, "LB_HW_TIERS_FILE": "/x/y.json"}).hw_tiers_file) == "y.json"


# ── pruning (review round 1): a silent box and a box the fleet no longer names leave the file ────

def test_a_box_silent_longer_than_the_longest_tier_is_dropped_on_save(tmp_path):
    path = str(tmp_path / "t.json")
    t = HwTiers(path)
    t.ingest("gone", {"gpu_temp_c": 1}, T0)
    t.ingest("here", {"gpu_temp_c": 2}, T0)
    later = T0 + TIERS["30d"][1] + 900
    t.ingest("here", {"gpu_temp_c": 3}, later)
    t.save(now=later)
    assert t.boxes() == ["here"]
    assert set(json.loads(open(path).read())["tiers"]["30d"]) == {"here"}
    assert t.series("gone", "gpu_temp_c", "30d", now=later) is None


def test_a_box_heard_inside_the_longest_tier_is_kept(tmp_path):
    t = HwTiers(str(tmp_path / "t.json"))
    t.ingest("quiet", {"gpu_temp_c": 1}, T0)
    t.save(now=T0 + TIERS["30d"][1] - 3600)
    assert t.boxes() == ["quiet"]


def test_a_silent_box_in_the_file_is_dropped_at_load(tmp_path):
    path = str(tmp_path / "t.json")
    t = HwTiers(path)
    t.ingest("old", {"gpu_temp_c": 1}, T0)
    t.save(now=T0)
    later = T0 + TIERS["30d"][1] + 900
    u = HwTiers(path, clock=lambda: later)
    u.load()
    assert u.boxes() == [] and u.status()["state"] == "loaded"


def test_boxes_the_fleet_no_longer_names_are_dropped_at_load_and_named_in_the_log(tmp_path, caplog):
    path = str(tmp_path / "t.json")
    t = HwTiers(path)
    for box in ("node-1", "retired"):
        t.ingest(box, {"gpu_temp_c": 1}, T0)
    t.save(now=T0)
    u = HwTiers(path, clock=lambda: T0 + 60, boxes=("node-1", "node-2"))
    caplog.set_level("INFO")
    u.load()
    assert u.boxes() == ["node-1"]
    assert any("retired" in r.getMessage() and "fleet" in r.getMessage() for r in caplog.records)
