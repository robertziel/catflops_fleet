"""fleet_telemetry's READERS, one per source, each against what that source answered on a fleet box
(tests/telemetry_fixtures.py: captured, not invented).

The rules pinned per reader:
  * it reads the figure its source states, in the unit the domain model carries (MB, °C, W, MHz, bytes);
  * a value the source does not give — `[N/A]`, a missing file, a sensor the box lacks — is None, never 0, and
    never another source standing in for it (an ACPI board zone is not the CPU's temperature; /proc/cpuinfo is not
    cpufreq; a container network namespace is not the box's NICs);
  * the CPU and memory figures equal what psutil gave the senders before (the wire's meaning does not move under a
    rolling redeploy): utilisation over the window between two /proc/stat reads, used = total − MemAvailable.
"""
from __future__ import annotations

import json

import pytest

from fleet_telemetry.model import ContainerUsage, GpuReading, NetCounters, ThermalZone, throttle_index
from fleet_telemetry.readers import (GPU_READERS, AmdgpuSysfs, Cgroups, CpuSensors, Fans, Meminfo, NetDev,
                                     NvidiaSmi, ProcStat, ThermalZones)
from tests.telemetry_fixtures import (AMD_APU, CGROUP_MEMORY_CURRENT, CGROUP_MEMORY_STAT, NVIDIA_TWO_CARDS,
                                      NVIDIA_UNIFIED, PROC_STAT_0, PROC_STAT_1, plant)


@pytest.fixture
def apu(tmp_path):
    return plant(tmp_path / "root", AMD_APU)


def smi(answer: str, calls: list | None = None):
    def run(argv):
        if calls is not None:
            calls.append(argv)
        assert argv == NvidiaSmi.ARGV, f"unexpected argv {argv}"
        return answer
    return run


# ── GPUs ──────────────────────────────────────────────────────────────────────

def test_the_gpu_readers_are_one_per_named_kind():
    assert GPU_READERS == {"nvidia": NvidiaSmi, "amd": AmdgpuSysfs}


def test_nvidia_unified_memory_card_one_query_and_na_is_none():
    calls = []
    (g,) = NvidiaSmi(run=smi(NVIDIA_UNIFIED, calls)).read()
    assert len(calls) == 1, "ONE nvidia-smi run per sample: the tiles' and the recorder's columns together"
    assert g == GpuReading(id=0, use_pct=96, vram_used_mb=None, vram_total_mb=None, temp_c=84, power_w=73.1,
                           power_inst_w=112.78, sm_mhz=2405, throttle_hex="0x0000000000000048", fan_pct=None)
    assert g.throttle_index == 100, "hw_thermal (0x40) is at the wall: the max over active bits, never a sum"


def test_nvidia_two_cards_each_a_reading():
    a, b = NvidiaSmi(run=smi(NVIDIA_TWO_CARDS)).read()
    assert (a.id, a.use_pct, a.vram_used_mb, a.vram_total_mb, a.power_w, a.throttle_index) == (0, 36, 7256, 24576,
                                                                                              210.55, 30)
    assert (b.id, b.vram_total_mb, b.fan_pct, b.throttle_index) == (1, 32607, 0, 0), "a fan at 0 % is a reading"


def test_nvidia_smi_failing_raises_for_the_sampler_to_name():
    def boom(argv):
        raise RuntimeError("NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver")
    with pytest.raises(RuntimeError, match="NVIDIA-SMI has failed"):
        NvidiaSmi(run=boom).read()


def test_amdgpu_card_from_its_driver_sysfs(apu):
    (g,) = AmdgpuSysfs(root=apu).read()
    assert g == GpuReading(id=0, use_pct=0, vram_used_mb=194, vram_total_mb=512, temp_c=48.0, power_w=63.09,
                           power_inst_w=63.09, power_label="PPT", sm_mhz=600, throttle_hex=None, fan_pct=None)
    assert g.throttle_index is None, "amdgpu has no NVML throttle mask: absent, not 0"


def test_amdgpu_without_a_card_reads_none(tmp_path):
    assert AmdgpuSysfs(root=plant(tmp_path, {"sys/class/drm/card0/device/vendor": "0x10de"})).read() == []


@pytest.mark.parametrize("mask,index", [(None, None), ("[N/A]", None), ("0x0", 0), ("0x0000000000000001", 0),
                                        ("0x4", 30), ("0x24", 50), ("0x88", 90), ("0x48", 100)])
def test_throttle_index_is_the_worst_active_bit(mask, index):
    assert throttle_index(mask) == index


# ── CPU, memory ───────────────────────────────────────────────────────────────

def test_cpu_utilisation_is_the_window_between_two_reads_as_psutil_computed_it(apu):
    stat = apu / "proc" / "stat"
    stat.write_text(PROC_STAT_0)
    reader = ProcStat(root=apu)                  # the baseline: the window opens when the reader does
    stat.write_text(PROC_STAT_1)
    total, cores = reader.read()
    # busy = Δ(all − idle − iowait) / Δall: 479 / 3185 · 100
    assert total == 15.0 and cores == [9.0, 10.0]


def test_cpu_utilisation_over_an_empty_window_is_none_not_zero(apu):
    (apu / "proc" / "stat").write_text(PROC_STAT_0)
    reader = ProcStat(root=apu)
    assert reader.read() == (None, None)


def test_memory_as_psutil_reported_it(apu):
    m = Meminfo(root=apu).read()
    assert (m.total_mb, m.used_mb, m.percent, m.swap_used_mb) == (127940, 24591, 19.2, 26191)


def test_cpu_temperature_and_clock_from_the_cpus_own_sensors(apu):
    assert CpuSensors(root=apu).read() == (73.4, 5056)


def test_no_cpu_sensor_is_none_never_the_board_zone(apu):
    for f in ("name", "temp1_input", "temp1_label"):
        (apu / "sys/class/hwmon/hwmon3" / f).unlink()
    for f in (apu / "sys/devices/system/cpu").rglob("scaling_cur_freq"):
        f.unlink()
    (apu / "proc").joinpath("cpuinfo").write_text("cpu MHz\t\t: 3000.000\n")
    assert CpuSensors(root=apu).read() == (None, None), "acpitz is the board's; /proc/cpuinfo is not cpufreq"


def test_a_cpu_typed_thermal_zone_is_the_cpus(tmp_path):
    root = plant(tmp_path, {"sys/class/thermal/thermal_zone3/type": "cpu-thermal",
                            "sys/class/thermal/thermal_zone3/temp": "61500"})
    assert CpuSensors(root=root).read() == (61.5, None)


# ── board, fans ───────────────────────────────────────────────────────────────

def test_thermal_zones_as_the_kernel_lists_them(apu):
    assert ThermalZones(root=apu).read() == [ThermalZone(type="acpitz", milli_c=20000)]


def test_fans_the_loudest_tach_or_none(apu, tmp_path):
    assert Fans(root=apu).read() is None, "no fan*_input: no tach, never 0 rpm"
    root = plant(tmp_path / "f", {"sys/class/hwmon/hwmon2/name": "nct6798", "sys/class/hwmon/hwmon2/fan1_input": "0",
                                  "sys/class/hwmon/hwmon2/fan2_input": "1830"})
    assert Fans(root=root).read() == 1830


# ── network ───────────────────────────────────────────────────────────────────

def test_net_counters_are_the_boxs_physical_nics(apu):
    """Two wired NICs and the Wi-Fi, all up; loopback, docker0, a bridge, a veth and the tailnet are not the box's
    wire (plumbing, or a tunnel that rides on it)."""
    assert NetDev(root=apu).read() == NetCounters(rx_bytes=10578593761 + 409078410769 + 980893354366,
                                                  tx_bytes=55987635592 + 11943902952 + 82869054684,
                                                  speed_mbps=10000)


def test_a_bond_slave_and_a_down_link_are_not_counted(apu):
    (apu / "sys/class/net/enp1s0f0/master").mkdir()
    (apu / "sys/class/net/wlp193s0/operstate").write_text("down\n")
    assert NetDev(root=apu).read().rx_bytes == 409078410769


# ── containers ────────────────────────────────────────────────────────────────

def _containers(tmp_path, rows, driver="systemd"):
    cg, cd = tmp_path / "cgroup", tmp_path / "containers"
    files = {"cgroup.controllers": "cpu memory io pids", "user.slice/memory.current": "1"}
    configs = {}
    for i, (name, image, current, stat) in enumerate(rows):
        cid = f"{i + 1:064x}"
        scope = f"system.slice/docker-{cid}.scope" if driver == "systemd" else f"docker/{cid}"
        files[f"{scope}/memory.current"] = current
        files[f"{scope}/memory.stat"] = stat
        files[f"{scope}/init.scope/memory.current"] = "7"
        configs[f"{cid}/config.v2.json"] = json.dumps({"Name": f"/{name}", "Config": {"Image": image,
                                                                                    "Env": ["TOKEN=secret"]}})
    plant(cg, files)
    plant(cd, configs)
    return Cgroups(cgroup_root=cg, containers_dir=cd)


@pytest.mark.parametrize("driver", ["systemd", "cgroupfs"])
def test_containers_ram_is_memory_current_minus_inactive_file(tmp_path, driver):
    reader = _containers(tmp_path, [("svc", "svc:1", CGROUP_MEMORY_CURRENT, CGROUP_MEMORY_STAT),
                                    ("big", "big:2", str(3 * 2**30), "inactive_file 1073741824\n")], driver)
    got = reader.read()
    assert got.count == 2 and got.ram_mb == 2048 + 5
    assert got.top(16) == [ContainerUsage("big", "big:2", 2048), ContainerUsage("svc", "svc:1", 5)]


def test_a_container_whose_ram_cannot_be_read_withholds_the_total(tmp_path):
    reader = _containers(tmp_path, [("a", "a:1", "1048576", "garbage\n")])
    got = reader.read()
    assert got.ram_mb is None and got.top(16) == [ContainerUsage("a", "a:1", None)]


def test_a_cgroup_root_that_is_not_v2_or_a_missing_containers_dir_is_refused(tmp_path):
    with pytest.raises(ValueError, match="cgroup v2"):
        Cgroups(cgroup_root=tmp_path / "nope", containers_dir=tmp_path)
    plant(tmp_path / "cg", {"cgroup.controllers": "memory"})
    with pytest.raises(ValueError, match="HW_SENDER_DOCKER_ROOT"):
        Cgroups(cgroup_root=tmp_path / "cg", containers_dir=tmp_path / "missing")
