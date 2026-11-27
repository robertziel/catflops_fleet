"""What the boxes' kernels and GPU tools answer — the inputs fleet_telemetry's readers are tested against.

CAPTURED, NOT INVENTED. Every file's content below was read off a fleet box; do not tidy the numbers:

  * AMD_APU — an AMD APU box (Radeon 8060S, gfx1151, amdgpu driver), 2026-10-30: the amdgpu card's sysfs
    (`/sys/class/drm/card0/device`: utilisation, VRAM carve-out, its hwmon — edge temperature, the PPT package power
    average and input, the sclk), its hwmon chips (ACPI, NVMe, k10temp), its one thermal zone, two cores' cpufreq,
    two `/proc/stat` reads one second apart, `/proc/meminfo`, a `/proc/net/dev` cut to one interface of each kind
    and their `/sys/class/net` state, and one container's cgroup memory files.
  * NVIDIA_UNIFIED / NVIDIA_TWO_CARDS — `nvidia-smi --query-gpu=<NVIDIA_FIELDS> --format=csv,noheader,nounits` in
    nvidia-smi's own format. The index / utilisation / memory columns are the 2026-10-11 captures (a GB10: `[N/A]`
    memory, unified with the RAM; an x86 box with an RTX 3090 at work and an RTX 5090 holding a model); the GB10's
    temperature / average and instantaneous draw / SM clock / throttle mask are a captured hw_watch row (84 °C,
    73.10 W, 112.78 W, 2405 MHz, 0x48 = hw_slowdown + hw_thermal, no fan). The two-card row's sensor columns are in
    the same format with the 3090 power-capped (0x04).
"""
from __future__ import annotations

from pathlib import Path

#: The amdgpu card's directory and its hwmon (the driver names its hwmon `amdgpu`).
_CARD = "sys/class/drm/card0/device"
_MON = f"{_CARD}/hwmon/hwmon5"

AMD_APU: dict[str, str] = {
    f"{_CARD}/vendor": "0x1002",
    f"{_CARD}/gpu_busy_percent": "0",
    f"{_CARD}/mem_info_vram_total": "536870912",
    f"{_CARD}/mem_info_vram_used": "203853824",
    f"{_MON}/name": "amdgpu",
    f"{_MON}/temp1_label": "edge",
    f"{_MON}/temp1_input": "48000",
    f"{_MON}/power1_label": "PPT",
    f"{_MON}/power1_average": "63091000",
    f"{_MON}/power1_input": "63094000",
    f"{_MON}/freq1_label": "sclk",
    f"{_MON}/freq1_input": "600000000",
    # a connector of the same card: a `card0-*` entry is never a card
    "sys/class/drm/card0-DP-1/status": "disconnected",
    "sys/class/hwmon/hwmon0/name": "acpitz_0",
    "sys/class/hwmon/hwmon0/temp1_input": "20000",
    "sys/class/hwmon/hwmon1/name": "nvme",
    "sys/class/hwmon/hwmon1/temp1_input": "42850",
    "sys/class/hwmon/hwmon1/temp1_label": "Composite",
    "sys/class/hwmon/hwmon3/name": "k10temp",
    "sys/class/hwmon/hwmon3/temp1_input": "73375",
    "sys/class/hwmon/hwmon3/temp1_label": "Tctl",
    "sys/class/thermal/thermal_zone0/type": "acpitz",
    "sys/class/thermal/thermal_zone0/temp": "20000",
    "sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq": "5049414",
    "sys/devices/system/cpu/cpu1/cpufreq/scaling_cur_freq": "5062804",
    "proc/meminfo": (
        "MemTotal:       131010608 kB\nMemFree:        18202080 kB\nMemAvailable:   105828984 kB\n"
        "Buffers:        10401852 kB\nCached:         72298516 kB\nSwapCached:      4683928 kB\n"
        "Active:         19698712 kB\nInactive:       81814860 kB\nSwapTotal:      157286396 kB\n"
        "SwapFree:       130465964 kB\nDirty:            111700 kB\n"),
    "proc/net/dev": (
        "Inter-|   Receive                                                |  Transmit\n"
        " face |bytes    packets errs drop fifo frame compressed multicast|bytes    packets errs drop fifo colls carrier compressed\n"
        "    lo: 104346253682 118745908    0    0    0     0          0         0 104346253682 118745908    0    0    0     0       0          0\n"
        "enp1s0f0: 10578593761 51433996    1   22    0     1          0       340 55987635592 66127395    0   47    0     0       0          0\n"
        "enp197s0f1: 409078410769 304206864    0    0    0     0          0         0 11943902952 38799906    0   69    0     0       0          0\n"
        "wlp193s0: 980893354366 774101623    0 5310    0     0          0         0 82869054684 421845888    0  154    0     0       0          0\n"
        "tailscale0: 8298554948 29601200    0    0    0     0          0         0 6281895900 16003107    0    0    0     0       0          0\n"
        "docker0: 13682121045 21668828    0    0    0     0          0         3 10994790191 30040860    0    8    0     0       0          0\n"
        "br-33d0d39a43cd: 88767151995 376261390    0    0    0     0          0         0 828372744083 656362370    0    2    0     0       0          0\n"
        "veth734b8f2: 16506040088 55382232    0    0    0     0          0         0 14216497272 66323279    0    0    0     0       0          0\n"),
    "sys/class/net/enp1s0f0/operstate": "up",
    "sys/class/net/enp1s0f0/speed": "10000",
    "sys/class/net/enp197s0f1/operstate": "up",
    "sys/class/net/enp197s0f1/speed": "10000",
    "sys/class/net/wlp193s0/operstate": "up",
    # a Wi-Fi link answers EINVAL for speed: no rate, not a zero (absent here)
}

#: Two `/proc/stat` reads one second apart on that box: the aggregate line and two cores.
PROC_STAT_0 = ("cpu  159778460 987850 35254418 1139873703 11255937 0 2254344 0 0 0\n"
               "cpu0 2337089 18253 717812 38044982 271267 0 333207 0 0 0\n"
               "cpu1 2487844 20533 745042 38352305 287685 0 319939 0 0 0\n"
               "intr 1 2 3\n")
PROC_STAT_1 = ("cpu  159778783 987856 35254562 1139875888 11256458 0 2254350 0 0 0\n"
               "cpu0 2337094 18253 717815 38045073 271267 0 333208 0 0 0\n"
               "cpu1 2487850 20533 745045 38352378 287702 0 319940 0 0 0\n"
               "intr 1 2 3\n")

#: One running container's cgroup memory files (systemd driver), and its anon / file split.
CGROUP_MEMORY_CURRENT = "5906432"
CGROUP_MEMORY_STAT = "anon 4882432\nfile 57344\nactive_file 53248\ninactive_file 8192\n"

#: The columns fleet_telemetry asks nvidia-smi for (readers/gpu.py NvidiaSmi.FIELDS), in its answer's order.
NVIDIA_UNIFIED = "0, 96, [N/A], [N/A], 84, 73.10, 112.78, 2405, 0x0000000000000048, [N/A]\n"
NVIDIA_TWO_CARDS = ("0, 36, 7256, 24576, 64, 210.55, 233.10, 1740, 0x0000000000000004, 41\n"
                    "1, 0, 28049, 32607, 38, 31.02, 30.70, 2280, 0x0000000000000000, 0\n")
#: Its OWN call (NvidiaSmi.HEADROOM_ARGV): `index, temperature.gpu.tlimit` — °C of headroom below the GPU's own
#: throttle limit (a GB10 read 49 at 47 °C, 2026-11-27); `[N/A]` on a card whose driver states none.
NVIDIA_UNIFIED_HEADROOM = "0, 3\n"
NVIDIA_TWO_CARDS_HEADROOM = "0, 26\n1, [N/A]\n"
#: What an nvidia-smi that does not know the field answers (exit 2): the whole headroom call is refused.
NVIDIA_NO_TLIMIT = 'Field "temperature.gpu.tlimit" is not a valid field to query.'


def nvidia_run(main: str, headroom, calls: list | None = None):
    """A fake nvidia-smi: the sample's query answers `main`, the headroom query `headroom` — or raises it when it is an
    exception (what readers/gpu.py `_run` raises on a non-zero exit)."""
    from fleet_telemetry.readers.gpu import NvidiaSmi

    def run(argv):
        if calls is not None:
            calls.append(argv)
        if argv == NvidiaSmi.ARGV:
            return main
        assert argv == NvidiaSmi.HEADROOM_ARGV, f"unexpected argv {argv}"
        if isinstance(headroom, Exception):
            raise headroom
        return headroom
    return run

def plant(root: Path, files: dict[str, str]) -> Path:
    """Write `files` (relative path → content) under `root`; return `root`."""
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text if text.endswith("\n") else text + "\n")
    return root


# ── box health ───────────────────────────────────────────────────────────────────────────────────────
# NOT CAPTURED: no box this suite can reach exposes /sys/class/infiniband or logs an Xid. These are written in the
# kernel's documented formats — Documentation/ABI/stable/sysfs-class-infiniband (a port's `state` "4: ACTIVE",
# `phys_state` "5: LinkUp", `rate` "200 Gb/sec (4X HDR)", `link_layer`, `counters/`, `hw_counters/`,
# `gid_attrs/ndevs/<n>`), Documentation/ABI/testing/dev-kmsg (one record per read: "<prio>,<seq>,<µs since boot>,
# <flags>;<message>", continuation lines indented by a space) and the NVIDIA driver's own Xid line ("NVRM: Xid
# (PCI:<bdf>): <code>, …"). Replace them with a capture when a fleet box gives one.

_IB0 = "sys/class/infiniband/mlx5_0/ports/1"
_IB1 = "sys/class/infiniband/mlx5_1/ports/1"

#: Two RoCE ports: mlx5_0 up at 200 Gb/s with its netdev visible here (MTU 9000) and every counter; mlx5_1 down, a
#: driver without `hw_counters/`, and its netdev in another network namespace (no /sys/class/net entry here).
RDMA_TWO_PORTS: dict[str, str] = {
    f"{_IB0}/state": "4: ACTIVE",
    f"{_IB0}/phys_state": "5: LinkUp",
    f"{_IB0}/rate": "200 Gb/sec (4X HDR)",
    f"{_IB0}/link_layer": "Ethernet",
    f"{_IB0}/gid_attrs/ndevs/0": "enp1s0f0np0",
    f"{_IB0}/hw_counters/out_of_buffer": "1234",
    f"{_IB0}/hw_counters/local_ack_timeout_err": "7",
    f"{_IB0}/hw_counters/rnr_nak_retry_err": "0",
    f"{_IB0}/hw_counters/packet_seq_err": "2",
    f"{_IB0}/hw_counters/implied_nak_seq_err": "0",
    f"{_IB0}/counters/port_rcv_errors": "0",
    f"{_IB0}/counters/port_xmit_discards": "5",
    f"{_IB0}/counters/port_rcv_remote_physical_errors": "0",
    f"{_IB0}/counters/symbol_error": "0",
    f"{_IB0}/counters/link_downed": "1",
    "sys/class/net/enp1s0f0np0/mtu": "9000",
    f"{_IB1}/state": "1: DOWN",
    f"{_IB1}/phys_state": "3: Disabled",
    f"{_IB1}/rate": "40 Gb/sec (4X QDR)",
    f"{_IB1}/link_layer": "Ethernet",
    f"{_IB1}/gid_attrs/ndevs/0": "enp2s0np0",
    f"{_IB1}/counters/port_rcv_errors": "3",
    f"{_IB1}/counters/port_xmit_discards": "0",
    f"{_IB1}/counters/port_rcv_remote_physical_errors": "0",
    f"{_IB1}/counters/symbol_error": "0",
    f"{_IB1}/counters/link_downed": "0",
}

#: /proc/vmstat, cut to the line the kernel reader reads and two around it.
VMSTAT = "pgfault 123456789\noom_kill 3\nnr_free_pages 1000\n"

#: /dev/kmsg as four reads: the boot's first record (seq 0), a module line with continuation lines, two Xids.
KMSG_RECORDS = [
    "6,0,0,-;Linux version 6.8.0\n",
    "6,1,12000000,-;NVRM: loading NVIDIA UNIX Open Kernel Module\n SUBSYSTEM=module\n DEVICE=+module:nvidia\n",
    "4,1802,900000000,-;NVRM: Xid (PCI:0000:01:00): 79, pid='<unknown>', name=<unknown>, GPU has fallen off the bus.\n",
    "4,1803,905500000,-;NVRM: Xid (PCI:0000:01:00): 13, pid=1234, name=python3, Graphics Exception\n",
]

# ── RoCE QoS and PHY counters ───────────────────────────────────────────────────────────────────
# NOT CAPTURED either (no RoCE netdev on a reachable box). Written in the tools' documented output formats: `ethtool -S`
# (ethtool(8): a "NIC statistics:" line, then one indented "<name>: <value>" per counter; the names are mlx5e's,
# Documentation/networking/device_drivers/ethernet/mellanox/mlx5/counters.rst), and iproute2's dcb in its plain form
# (dcb-pfc(8): "prio-pfc 0:off … 7:off"; dcb-apptrust(8): "order: <selector> …"; dcb-app(8) with -N:
# "dscp-prio <dscp>:<prio> …"). The refusals the tests use are what iproute2 6.15 / ethtool 6.14 printed in the sender
# image for a netdev that is absent ("No such device") or a driver without the op ("Operation not supported").

#: `ethtool -S enp1s0f0np0` of an mlx5e netdev, cut to the PHY counters the sender reads and two around them.
ETHTOOL_S_MLX5 = ("NIC statistics:\n"
                  "     rx_packets: 912345678\n"
                  "     rx_discards_phy: 4242\n"
                  "     tx_discards_phy: 0\n"
                  "     rx_crc_errors_phy: 3\n"
                  "     rx_symbol_err_phy: 0\n"
                  "     rx_pause_ctrl_phy: 17\n"
                  "     tx_pause_ctrl_phy: 9\n"
                  "     tx_packets: 812345678\n")
#: `dcb pfc show dev enp1s0f0np0 prio-pfc`: PFC on priority 3 (the RoCE convention).
DCB_PFC = "prio-pfc 0:off 1:off 2:off 3:on 4:off 5:off 6:off 7:off\n"
#: `dcb apptrust show dev enp1s0f0np0 order` on a driver that implements the op.
DCB_APPTRUST = "order: dscp pcp \n"
#: `dcb -N app show dev enp1s0f0np0 dscp-prio`: four DSCP → priority entries.
DCB_APP_DSCP = "dscp-prio 0:0 24:3 26:3 48:6\n"
