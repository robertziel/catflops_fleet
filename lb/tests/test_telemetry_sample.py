"""fleet_telemetry's SAMPLE on the wire, and what the gateway makes of it.

  * the fields the LB already reads keep their names and meanings (host, cpu_percent, ram_*, swap_used_mb, gpus rows,
    cpu_per_core), so an old LB hears a new sender and a new LB an old one during a rolling redeploy;
  * `deep` and `containers` are what the per-box FLIGHT RECORDER reads (vllm_lb/recorder.py), and every figure the
    box cannot read is absent with the sender's reason (`deep.why`, keyed by the shared figure keys);
  * ONE definition of the figures: the LB's recorder uses fleet_telemetry.figures, not a copy;
  * the datagram stays under 8 KB on a box with 200 containers and 32 cores.
"""
from __future__ import annotations

import json

import pytest

from fleet_telemetry import figures
from fleet_telemetry.model import ContainerSet, ContainerUsage
from fleet_telemetry.readers import AmdgpuSysfs, Cgroups, CpuSensors, Fans, Meminfo, NetDev, NvidiaSmi, ProcStat, \
    ThermalZones
from fleet_telemetry.sampler import Sampler, envelope
from tests.telemetry_fixtures import AMD_APU, NVIDIA_UNIFIED, PROC_STAT_0, PROC_STAT_1, plant
from vllm_lb import recorder
from vllm_lb.config import Config
from vllm_lb.fleet import parse_fleet
from vllm_lb.hw import HwCollector
from vllm_lb.recorder import FlightRecorder

#: Exactly the top-level fields of today's broadcast that a gateway reads, plus the recorder's.
WIRE = {"sampled_at", "host", "cpu_percent", "ram_percent", "ram_used_mb", "ram_total_mb", "swap_used_mb", "gpus",
        "cpu_per_core", "deep", "containers", "containers_total"}


class Fixed:
    """A reader that answers what it is given (or raises it)."""

    def __init__(self, source: str, value):
        self.source, self.value = source, value

    def read(self):
        if isinstance(self.value, Exception):
            raise self.value
        return self.value


def many(n: int) -> ContainerSet:
    return ContainerSet([ContainerUsage(f"svc-{i:03d}-" + "x" * 40, f"registry.example/team/image-{i}:2026.10.30",
                                        i + 1) for i in range(n)])


def apu_sampler(tmp_path, *, network="host", containers=None, gpu=None) -> Sampler:
    root = plant(tmp_path / "root", AMD_APU)
    (root / "proc" / "stat").write_text(PROC_STAT_0)
    cpu = ProcStat(root=root)
    (root / "proc" / "stat").write_text(PROC_STAT_1)
    return Sampler(gpu=gpu or AmdgpuSysfs(root=root), cpu=cpu, memory=Meminfo(root=root),
                   sensors=CpuSensors(root=root), zones=ThermalZones(root=root), fans=Fans(root=root),
                   net=NetDev(root=root) if network == "host" else None,
                   containers=containers or Fixed("cgroup v2", many(3)))


def collector(host: str, cards: tuple[str, ...], platform: str) -> HwCollector:
    listed = ", ".join(f"{{index: {i}, type: {k}, serves: false}}" for i, k in enumerate(cards))
    fleet = parse_fleet(f"panel: {{style: vibe}}\nboxes:\n  box: {{hostname: '{host}', platform: {platform}, cards: [{listed}]}}\n"
                        "clusters: {}\nbackends: {}\n", "fleet.yml")
    return HwCollector(Config(hw_hosts=dict(fleet.hw_hosts), fleet=fleet))


# ── the wire ──────────────────────────────────────────────────────────────────

def test_the_wire_keeps_the_fields_the_gateway_reads_and_adds_the_recorders(tmp_path):
    env = json.loads(envelope(apu_sampler(tmp_path).sample(), host="box-c-host", now=1_761_840_000.0))
    assert set(env) == WIRE
    assert env["host"] == "box-c-host" and env["sampled_at"] == "2025-10-30T16:00:00+00:00"
    assert (env["cpu_percent"], env["cpu_per_core"]) == (15.0, [9.0, 10.0])
    assert (env["ram_total_mb"], env["ram_used_mb"], env["ram_percent"], env["swap_used_mb"]) == (127940, 24591,
                                                                                                    19.2, 26191)
    assert env["gpus"] == [{"id": 0, "use_pct": 0, "vram_used_mb": 194, "vram_total_mb": 512}], \
        "a card row exactly as the tiles read it"


def test_the_gateway_tiles_read_the_new_sender_as_they_read_the_old(tmp_path):
    payload = envelope(apu_sampler(tmp_path).sample(), host="box-c-host", now=1_761_840_000.0)
    c = collector("box-c-host", ("radeon8060s",), "linux/amd64")
    box = c.ingest_and_snapshot(payload)
    assert c.received == 1 and c.ignored == c.dropped == 0
    assert (box["gpu_count"], box["gpu_pct"], box["cpu_percent"], box["ram_percent"]) == (1, 0, 15.0, 19.2)
    assert box["cpu_per_core"] == [9.0, 10.0]


def test_a_unified_memory_cards_na_memory_is_none_on_the_wire_and_the_tile_shows_the_ram(tmp_path):
    s = apu_sampler(tmp_path, gpu=NvidiaSmi(run=lambda argv: NVIDIA_UNIFIED)).sample()
    env = json.loads(envelope(s, host="box-a-host", now=0.0))
    assert env["gpus"] == [{"id": 0, "use_pct": 96, "vram_used_mb": None, "vram_total_mb": None}], \
        "[N/A] is no reading — not the 0 the old sender invented"
    box = collector("box-a-host", ("gb10",), "linux/arm64").ingest_and_snapshot(json.dumps(env))
    assert box["mem_unified"] is True and box["mem_total_mb"] == 127940 and box["gpu_pct"] == 96


# ── the recorder's figures ────────────────────────────────────────────────────

def test_the_recorder_reads_every_figure_this_box_exposes(tmp_path):
    env = json.loads(envelope(apu_sampler(tmp_path).sample(), host="h", now=0.0))
    r = FlightRecorder()
    r.ingest("box", env, 100.0)
    nxt = json.loads(json.dumps(env))
    nxt["deep"]["net_rx_bytes"] += 10_000_000
    r.ingest("box", nxt, 105.0)
    s = r.summary("box", 105.0, stale=False, age_s=0)
    f = s["figures"]
    assert (f["gpu_temp_c"], f["gpu_power_w"], f["gpu_mhz"], f["cpu_temp_c"], f["cpu_mhz"], f["board_temp_c"]) == (
        48.0, 63.09, 600, 73.4, 5056, 20.0)
    assert f["net_in_mbps"] == pytest.approx(16.0) and f["net_out_mbps"] == 0
    assert f["fan_rpm"] is None and "tachometer" in s["why"]["fan_rpm"]
    assert f["throttle"] is None and "amdgpu" in s["why"]["throttle"]
    assert s["power"]["peak_inst_w"] == 63.09


def test_every_reason_the_sender_gives_is_a_figure_the_recorder_shows(tmp_path):
    s = apu_sampler(tmp_path, network="bridged", gpu=Fixed("nvidia-smi", RuntimeError("driver gone"))).sample()
    why = s.wire()["deep"]["why"]
    assert why and set(why) <= set(figures.KEYS)
    assert "driver gone" in why["gpu_temp_c"]
    assert "bridged" in why["net_in_mbps"]


def test_a_failing_reader_is_its_figures_reason_and_the_rest_still_flows(tmp_path):
    s = apu_sampler(tmp_path, containers=Fixed("cgroup v2", PermissionError(13, "Permission denied"))).sample()
    w = s.wire()
    assert "containers" not in w and "Permission denied" in w["containers_why"]
    assert w["cpu_percent"] == 15.0 and w["deep"]["gpu"]
    r = FlightRecorder()
    r.ingest("box", w, 1.0)
    assert "Permission denied" in r.summary("box", 1.0, stale=False, age_s=0)["containers_why"]


def test_containers_top_n_by_ram_with_the_whole_count_and_the_datagram_stays_small(tmp_path):
    s = apu_sampler(tmp_path, containers=Fixed("cgroup v2", many(200))).sample()
    payload = envelope(s, host="h", now=0.0)
    w = json.loads(payload)
    assert len(w["containers"]) == figures.SENT_CONTAINERS and w["containers"][0]["ram_mb"] == 200
    assert w["containers_total"] == {"count": 200, "ram_mb": sum(range(1, 201))}
    w["cpu_per_core"] = [99.9] * 32
    assert len(json.dumps(w).encode()) < 8192
    r = FlightRecorder()
    r.ingest("box", w, 1.0)
    c = r.summary("box", 1.0, stale=False, age_s=0)["containers"]
    assert (c["count"], c["unlisted"]) == (200, 200 - figures.SENT_CONTAINERS)


# ── DRY ───────────────────────────────────────────────────────────────────────

def test_the_recorder_and_the_sender_share_one_figure_table():
    assert recorder.FIGURES is figures.FIGURES_TABLE
    assert recorder.KEYS == figures.KEYS
