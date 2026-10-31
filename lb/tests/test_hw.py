"""Hardware reaches the LB off the LAN, holding no credential.

`worklog/vllm-lb-hardware.md` stopped at its own KILL line: the only way in was an ssh key, and
putting one inside a process reachable through the public tunnel was a worse trade than the panel
token the operator had already declined. That is no longer the ground — broker_parrot's
`hw-broadcast` now runs on all three boxes and sends a JSON datagram on UDP 48291 every 5 s.

Every expectation below is taken from a REAL packet captured on the wire at 2026-09-12 23:00 UTC
(three senders answered), not from the sender's source. The rules that matter:

  * a hostname is not a box — `box-a-host` is box-a, and an unmapped host is IGNORED rather
    than guessed, because a wrong box label is worse than a missing tile;
  * staleness is data (the `depth_known` lesson): past the max age the box reports its age and
    stops claiming numbers, instead of showing figures that quietly stopped moving;
  * a GB10 sends `vram_used_mb: 0, vram_total_mb: 0` because its memory is UNIFIED. Rendering
    that as "0GB VRAM" would be a lie the wire never told.
"""
from __future__ import annotations

import http.client
import json

import pytest

from vllm_lb.config import Config
from vllm_lb.fleet import parse_fleet
from vllm_lb.hw import HwCollector

# ── verbatim off the wire; do not tidy ────────────────────────────────────────
BOX_A = ('{"sampled_at": "2026-09-12T23:00:14.136027+00:00", "host": "box-a-host", '
         '"cpu_percent": 12.7, "cpu_ours_percent": null, "ram_percent": 74.6, '
         '"ram_used_mb": 92921, "ram_total_mb": 124610, "ram_ours_used_mb": null, '
         '"ram_ours_percent": null, "swap_used_mb": 0, '
         '"gpus": [{"id": 0, "use_pct": 96, "vram_used_mb": 0, "vram_total_mb": 0}]}')
BOX_C = ('{"sampled_at": "2026-09-12T23:00:15.000000+00:00", "host": "box-c", '
           '"cpu_percent": 8.0, "ram_percent": 41.0, "ram_used_mb": 26000, '
           '"ram_total_mb": 64000, "swap_used_mb": 12, '
           '"gpus": [{"id": 0, "use_pct": 3, "vram_used_mb": 900, "vram_total_mb": 16384}]}')

#: Captured 2026-09-30 19:32 UTC, after `hw_broadcast_main.py` gained the additive
#: `cpu_per_core` (one % per logical CPU; a GB10 has 20). Core 7 pinned at 100 is real.
BOX_B_CORES = ('{"sampled_at": "2026-09-30T19:32:06.843459+00:00", "host": "box-b-host", '
                '"cpu_percent": 9.5, "cpu_ours_percent": null, "ram_percent": 61.8, '
                '"ram_used_mb": 76963, "ram_total_mb": 124546, "ram_ours_used_mb": null, '
                '"ram_ours_percent": null, "swap_used_mb": 3821, '
                '"gpus": [{"id": 0, "use_pct": 2, "vram_used_mb": 0, "vram_total_mb": 0}], '
                '"cpu_per_core": [2.4, 2.2, 2.6, 1.4, 2.0, 4.6, 3.2, 100.0, 1.4, 2.8, 2.8, 2.6, '
                '17.0, 9.5, 5.5, 2.8, 3.8, 4.0, 3.6, 14.8]}')

NOW = 5000.0


#: Two discrete cards in one box, the site-b shape: card 0 a 3090 (another tenant's), card 1 the 5090 that serves.
TWO_CARDS = json.dumps({"host": "site-b-host", "cpu_percent": 4.0, "ram_percent": 30.0, "ram_used_mb": 28000,
                        "ram_total_mb": 94000, "swap_used_mb": 7168,
                        "gpus": [{"id": 0, "use_pct": 31, "vram_used_mb": 7270, "vram_total_mb": 24576},
                                 {"id": 1, "use_pct": 5, "vram_used_mb": 28058, "vram_total_mb": 32607}]})

#: WHAT EACH BOX IS — its cards, by type — is the fleet file's (vllm_lb/fleet.py): the receiver never infers a box's
#: memory model from what its sender reports. box-c is the Ryzen AI Max+ 395's Radeon 8060S: unified.
FLEET = parse_fleet("""
panel:
  style: vibe
boxes:
  box-a: {hostname: box-a-host, platform: linux/arm64, cards: [{index: 0, type: gb10, serves: false}]}
  box-b: {hostname: box-b-host, platform: linux/arm64, cards: [{index: 0, type: gb10, serves: false}]}
  box-c: {hostname: box-c, platform: linux/amd64, cards: [{index: 0, type: radeon8060s, serves: false}]}
  site-b: {hostname: site-b-host, platform: linux/amd64, cards: [{index: 0, type: rtx3090, serves: false},
                                                                {index: 1, type: rtx5090, serves: false}]}
clusters: {}
backends: {}
""", "fleet.yml")

def _c(**over):
    return HwCollector(Config(**{"hw_hosts": dict(FLEET.hw_hosts), "fleet": FLEET, **over}))


# ── the wire ──────────────────────────────────────────────────────────────────

def test_the_captured_packet_lands_on_the_box_it_came_from():
    c = _c()
    assert c.ingest(BOX_A, now=NOW) == "box-a", "box-a-host is box-a's hostname"
    box = c.snapshot(now=NOW)["box-a"]
    assert box["host"] == "box-a-host"
    assert box["cpu_percent"] == 12.7
    assert box["ram_percent"] == 74.6
    assert (box["ram_used_mb"], box["ram_total_mb"]) == (92921, 124610)
    assert box["gpu_pct"] == 96, "one GPU: the box's utilisation is that GPU's"
    assert box["stale"] is False and box["age_s"] == 0.0


def test_the_same_sample_twice_is_one_sample():
    """The sender sends every sample to the subnet AND the limited broadcast (fleet_telemetry/broadcast.py), so a
    gateway on its LAN hears each one twice, milliseconds apart. Counted twice it doubled every ring and the
    recorder's network rate read every other sample as a 0 Mb/s delta (a panel's "NET IN 0 Mb/s" under a
    chart peaking at 136): a repeat of the box's last `sampled_at` is a duplicate, counted and dropped."""
    c = _c()
    rx = lambda n, at: json.dumps({**json.loads(BOX_A), "sampled_at": at,   # noqa: E731
                                   "deep": {"net_rx_bytes": n, "net_tx_bytes": n}})
    c.ingest(rx(0, "t0"), now=NOW)
    c.ingest(rx(625_000, "t1"), now=NOW + 5)
    assert c.ingest(rx(625_000, "t1"), now=NOW + 5.001) == "box-a"
    assert c.received == 2 and c.duplicates == 1 and c.status()["duplicates"] == 1
    rec = c.recorder.summary("box-a", NOW + 5.001, stale=False, age_s=0)
    assert rec["figures"]["net_in_mbps"] == pytest.approx(1.0), "the duplicate is not a 0 Mb/s sample"
    assert len(c.recorder_series("box-a", now=NOW + 5.001)["fine"]["t"]) == 2
    no_stamp = json.dumps({k: v for k, v in json.loads(BOX_C).items() if k != "sampled_at"})
    c.ingest(no_stamp, now=NOW)
    c.ingest(no_stamp, now=NOW + 1)
    assert c.duplicates == 1, "a sender with no sampled_at cannot be told repeating: each is a sample"


def test_an_unmapped_host_is_ignored_never_guessed():
    c = _c()
    assert c.ingest(json.dumps({"host": "someone-elses-laptop", "cpu_percent": 99}), now=NOW) is None
    assert c.snapshot(now=NOW) == {}
    assert c.ignored == 1


@pytest.mark.parametrize("junk", [b"\x00\x01\x02", "not json at all", "[]", '{"no_host": 1}', ""])
def test_a_bad_packet_is_counted_and_never_raises(junk):
    c = _c()
    assert c.ingest(junk, now=NOW) is None
    assert c.dropped == 1, "a malformed datagram must not take the listener down"


def test_the_host_map_comes_from_the_fleet_file(tmp_path):
    from tests.fleet_files import write
    cfg = Config.from_env({"LB_ALLOWED_HOSTS": "loopback", "LB_FLEET": write(tmp_path, """
panel:
  style: vibe
boxes:
  box-c: {hostname: nuc-7, platform: linux/amd64, cards: [{index: 0, type: radeon8060s, serves: false}]}
  box-b: {hostname: box-b-host, platform: linux/arm64, cards: [{index: 0, type: gb10, serves: false}]}
clusters: {}
backends: {}
""")})
    assert cfg.hw_hosts == {"nuc-7": "box-c", "box-b-host": "box-b"}
    c = HwCollector(cfg)
    assert c.ingest(json.dumps({"host": "nuc-7", "cpu_percent": 1.0}), now=NOW) == "box-c"


# ── unified memory: the GB10 shape ────────────────────────────────────────────

def test_a_gb10_reports_unified_memory_rather_than_a_zero_vram_total():
    """THE KILL LINE of this frame: never show a number the wire did not send."""
    box = _c().ingest_and_snapshot(BOX_A, now=NOW)
    assert box["mem_unified"] is True, "vram_total 0 on a GB10 means unified, not empty"
    assert (box["mem_used_mb"], box["mem_total_mb"]) == (92921, 124610), "the box's own RAM"


def test_discrete_cards_report_their_own_vram_summed_as_the_box_aggregate():
    """The aggregates other readers take are kept: the busiest card's utilisation, the cards' VRAM summed. The
    machine tile draws each card on its own row instead (`rows`, below)."""
    box = _c().ingest_and_snapshot(TWO_CARDS, now=NOW)
    assert (box["memory"], box["mem_unified"]) == ("discrete", False)
    assert (box["mem_used_mb"], box["mem_total_mb"]) == (7270 + 28058, 24576 + 32607)
    assert (box["gpu_pct"], box["gpu_count"]) == (31, 2)


def test_box_cs_igpu_is_unified_its_carve_out_is_no_vram_figure():
    """The Radeon 8060S takes the box's RAM ("Dynamic + carveout", ROCm's own table): the 16 GB rocm-smi reports
    in this capture is a carve-out, not the memory the GPU computes in — so the GPU's memory is the box's RAM."""
    box = _c().ingest_and_snapshot(BOX_C, now=NOW)
    assert (box["memory"], box["mem_unified"]) == ("unified", True)
    assert (box["mem_used_mb"], box["mem_total_mb"]) == (26000, 64000)


# ── the cards: the fleet names them, the sample reads them, the memory model lays them out ─────────

def test_the_per_card_list_is_published_as_the_sender_broadcast_it():
    box = _c().ingest_and_snapshot(TWO_CARDS, now=NOW)
    assert box["gpus"] == [{"id": 0, "use_pct": 31, "vram_used_mb": 7270, "vram_total_mb": 24576},
                           {"id": 1, "use_pct": 5, "vram_used_mb": 28058, "vram_total_mb": 32607}]
    assert [c["type"] for c in box["cards"]] == ["rtx3090", "rtx5090"], "the fleet's cards, by index"


def test_a_discrete_box_draws_one_row_per_card_each_with_its_own_vram():
    rows = _c().ingest_and_snapshot(TWO_CARDS, now=NOW)["rows"]
    assert [(r["label"], r["card"], r["use_pct"], r["vram_label"], r["vram_used_mb"], r["vram_total_mb"])
            for r in rows] == [("GPU 0", 0, 31, "VRAM 0", 7270, 24576), ("GPU 1", 1, 5, "VRAM 1", 28058, 32607)]
    assert [r["box_gpu"] for r in rows] == [False, False]


def test_a_unified_box_draws_ONE_row_as_it_always_has():
    """The operator: the GB10's display is right as it is — one GPU ring (the box's GPU), its memory the MEM row."""
    box = _c().ingest_and_snapshot(BOX_A, now=NOW)
    assert box["rows"] == [{"label": "GPU", "card": None, "box_gpu": True, "use_pct": 96, "history": [96],
                            "vram_label": None, "vram_used_mb": None, "vram_total_mb": None}]
    # …and every figure today's page reads is exactly what it was.
    assert (box["memory"], box["mem_unified"], box["gpu_pct"], box["gpu_count"]) == ("unified", True, 96, 1)
    assert (box["mem_used_mb"], box["mem_total_mb"]) == (92921, 124610)


def test_a_card_name_the_sample_carries_is_shown_and_none_is_invented():
    named = json.loads(TWO_CARDS)
    named["gpus"][1]["name"] = "NVIDIA GeForce RTX 5090"
    rows = _c().ingest_and_snapshot(json.dumps(named), now=NOW)["rows"]
    assert [r["vram_label"] for r in rows] == ["VRAM 0", "VRAM 1 · NVIDIA GeForce RTX 5090"]


def test_each_card_keeps_a_past_of_its_own():
    c = _c(hw_history=3)
    for i in range(4):
        sample = json.loads(TWO_CARDS)
        sample["gpus"][0]["use_pct"], sample["gpus"][1]["use_pct"] = 10 * i, i
        c.ingest(json.dumps(sample), now=NOW + i)
    box = c.snapshot(now=NOW + 3)["site-b"]
    assert box["history"]["gpus"] == {"0": [10, 20, 30], "1": [1, 2, 3]}
    assert [r["history"] for r in box["rows"]] == [[10, 20, 30], [1, 2, 3]]


def test_a_card_the_fleet_names_but_the_sample_lacks_is_a_row_with_no_reading():
    """A card the GPU CLI stopped reporting is still in the box: its row stays, blank — never dropped, never 0."""
    one = json.loads(TWO_CARDS)
    one["gpus"] = one["gpus"][:1]
    rows = _c().ingest_and_snapshot(json.dumps(one), now=NOW)["rows"]
    assert [(r["label"], r["use_pct"], r["vram_used_mb"]) for r in rows] == [("GPU 0", 31, 7270), ("GPU 1", None, None)]


def test_a_card_with_no_vram_total_of_its_own_reports_none_not_zero():
    zero = json.loads(TWO_CARDS)
    zero["gpus"][1]["vram_total_mb"] = zero["gpus"][1]["vram_used_mb"] = 0
    row = _c().ingest_and_snapshot(json.dumps(zero), now=NOW)["rows"][1]
    assert (row["vram_used_mb"], row["vram_total_mb"]) == (None, None)


def test_a_stale_box_keeps_its_rows_and_withdraws_their_readings():
    c = _c(hw_max_age_s=30.0)
    c.ingest(TWO_CARDS, now=NOW)
    gone = c.snapshot(now=NOW + 31)["site-b"]
    assert [r["label"] for r in gone["rows"]] == ["GPU 0", "GPU 1"]
    assert all(r["use_pct"] is None and r["vram_used_mb"] is None for r in gone["rows"])
    assert gone["gpus"] is None and gone["rows"][0]["history"] == [31], "what it did stays drawn"


def test_a_box_the_fleet_does_not_name_claims_no_memory_model():
    """No auto-detection: a sender mapped by hand with no fleet entry says nothing about its GPU's memory."""
    c = HwCollector(Config(hw_hosts={"box-a-host": "box-a"}))
    box = c.ingest_and_snapshot(BOX_A, now=NOW)
    assert (box["memory"], box["mem_unified"], box["mem_used_mb"], box["mem_total_mb"]) == (None, None, None, None)
    assert box["rows"] == [] and box["cards"] == []
    assert box["gpus"] == [{"id": 0, "use_pct": 96, "vram_used_mb": 0, "vram_total_mb": 0}], "the wire, as sent"


@pytest.mark.parametrize("junk", [[{"id": True, "use_pct": 1}], [{"id": 0, "use_pct": float("nan")}],
                                  [{"id": -1}], ["card"], [{"use_pct": 3}]])
def test_a_malformed_card_is_withheld_never_guessed(junk):
    box = _c().ingest_and_snapshot(json.dumps({"host": "site-b-host", "gpus": junk}), now=NOW)
    assert box["gpus"] == [], junk


def test_the_per_card_rings_are_bounded():
    from vllm_lb.hw import MAX_CARDS
    many = {"host": "site-b-host", "gpus": [{"id": i, "use_pct": 1} for i in range(MAX_CARDS + 9)]}
    box = _c().ingest_and_snapshot(json.dumps(many), now=NOW)
    assert len(box["gpus"]) == MAX_CARDS and len(box["history"]["gpus"]) == MAX_CARDS


# ── per-core load (additive field) ────────────────────────────────────────────

def test_per_core_load_passes_through_in_the_senders_order():
    box = _c().ingest_and_snapshot(BOX_B_CORES, now=NOW)
    cores = box["cpu_per_core"]
    assert len(cores) == 20, "a GB10 has 20 logical CPUs"
    assert cores[7] == 100.0 and cores[0] == 2.4, "order is the sender's: cell N is core N"
    assert box["cpu_percent"] == 9.5, "the aggregate is untouched by the new field"


def test_a_sender_without_per_core_reports_none_not_an_empty_matrix():
    """The 2026-09-12 packet predates the field: old broadcasters keep working, no cells."""
    box = _c().ingest_and_snapshot(BOX_A, now=NOW)
    assert box["cpu_per_core"] is None
    assert box["cpu_percent"] == 12.7


@pytest.mark.parametrize("bad", ["12,13", {"0": 1.0}, [1.0, "2.0"], [1.0, None], [True, 2.0],
                                 [1.0, float("nan")], [float("inf")]])
def test_a_malformed_per_core_list_is_withheld_whole(bad):
    """Dropping one bad entry would slide every later core into the wrong cell; NaN/inf would
    make /lb/status invalid JSON for the apps."""
    packet = json.dumps({"host": "box-c", "cpu_percent": 5.0, "cpu_per_core": bad})
    box = _c().ingest_and_snapshot(packet, now=NOW)
    assert box["cpu_per_core"] is None
    assert box["cpu_percent"] == 5.0, "a bad new field costs only itself"


def test_per_core_is_capped_so_a_runaway_list_cannot_grow_the_status_page():
    from vllm_lb.hw import MAX_CORES
    packet = json.dumps({"host": "box-c", "cpu_per_core": [1.0] * (MAX_CORES + 50)})
    box = _c().ingest_and_snapshot(packet, now=NOW)
    assert len(box["cpu_per_core"]) == MAX_CORES


def test_a_stale_box_withdraws_its_per_core_matrix_too():
    c = _c(hw_max_age_s=30.0)
    c.ingest(BOX_B_CORES, now=NOW)
    assert c.snapshot(now=NOW + 31)["box-b"]["cpu_per_core"] is None, \
        "per-core cells that stopped moving are as false as a frozen aggregate"


# ── history and staleness ─────────────────────────────────────────────────────

def test_the_ring_is_bounded_and_keeps_the_newest():
    c = _c(hw_history=4)
    for i in range(10):
        c.ingest(json.dumps({"host": "box-c", "cpu_percent": float(i),
                             "gpus": [{"id": 0, "use_pct": i}]}), now=NOW + i)
    hist = c.snapshot(now=NOW + 9)["box-c"]["history"]
    assert hist["gpu_pct"] == [6, 7, 8, 9], "oldest first, newest last, bounded"
    assert len(hist["cpu_pct"]) == 4


def test_a_stale_box_reports_its_age_and_stops_claiming_numbers():
    c = _c(hw_max_age_s=30.0)
    c.ingest(BOX_A, now=NOW)
    fresh = c.snapshot(now=NOW + 29)["box-a"]
    assert fresh["stale"] is False and fresh["gpu_pct"] == 96

    gone = c.snapshot(now=NOW + 31)["box-a"]
    assert gone["stale"] is True and gone["age_s"] == 31.0
    assert gone["gpu_pct"] is None and gone["cpu_percent"] is None, \
        "a number that stopped moving is worse than a blank"
    assert gone["history"]["gpu_pct"] == [96], "what it DID is still true, and still drawn"


def test_the_ram_keeps_a_past_of_its_own_beside_the_gpus_memory():
    """The MEM ring shows `ram_percent`, so the graph under it needs THAT figure's past — not `mem_pct`,
    the GPU's memory. On a box with VRAM of its own they are two numbers: on 2026-10-01 box-c's iGPU
    sat at 485/512 MB (95 %) under 20 % of RAM, and the MEM graph was red beneath a green MEM ring."""
    box = _c().ingest_and_snapshot(TWO_CARDS, now=NOW)
    assert box["history"]["ram_pct"] == [box["ram_percent"]] == [30.0], "the ring's own figure, sampled"
    assert box["history"]["mem_pct"] == [61.8], "35,328 of 57,183 MB of VRAM: still kept, the cards' own"


def test_on_a_gb10_the_ram_and_the_gpus_memory_are_one_number():
    """Why the GB10 boxes never showed it: unified memory makes the two rings agree."""
    box = _c().ingest_and_snapshot(BOX_A, now=NOW)
    assert box["history"]["ram_pct"] == box["history"]["mem_pct"] == [74.6]


def test_history_survives_a_page_reload_because_it_lives_here():
    """The page used to accumulate its own sparkline, so F5 erased every stat's past."""
    c = _c()
    for i in range(3):
        c.ingest(json.dumps({"host": "box-c", "cpu_percent": 1.0,
                             "gpus": [{"id": 0, "use_pct": 10 * i}]}), now=NOW + i)
    assert c.snapshot(now=NOW + 2)["box-c"]["history"]["gpu_pct"] == [0, 10, 20]


# ── it reaches the page ───────────────────────────────────────────────────────

def test_lb_status_carries_a_hardware_block_per_box():
    from vllm_lb.hub import Hub
    from vllm_lb.proxy import Proxy
    cfg = Config(hw_hosts=dict(FLEET.hw_hosts), fleet=FLEET, listen_host="127.0.0.1", listen_port=0)
    hw = HwCollector(cfg)
    hw.ingest(BOX_A)        # the REAL clock: /lb/status ages the sample against its own `now`,
                            # and a sample stamped in 1970 is correctly reported as stale
    hw.ingest(BOX_B_CORES)
    proxy = Proxy(cfg, Hub(cfg), hw=hw)
    proxy.start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=10)
        conn.request("GET", "/lb/status")
        body = json.loads(conn.getresponse().read())
    finally:
        proxy.stop()
    assert "box-a" in body["hardware"]
    assert body["hardware"]["box-a"]["gpu_pct"] == 96
    assert body["hardware"]["box-a"]["mem_unified"] is True
    assert body["hardware"]["box-a"]["cpu_per_core"] is None, "an old sender: no matrix"
    assert len(body["hardware"]["box-b"]["cpu_per_core"]) == 20, "the matrix reaches the apps"
