"""DEMO TELEMETRY SENDER: three synthetic boxes broadcasting the box sender's EXACT datagram.

WHY THE REAL MODEL. The datagram is built by lb/fleet_telemetry itself — `model.Sample` and its readings, serialised by
`sampler.envelope` — so the gateway reads precisely what a box's hw-sender sends; only the readings are invented. A
change to the wire breaks this sender the same day it breaks the real one.

WHAT IT SENDS, every DEMO_SEND_EVERY_S, to DEMO_LB_HOST:DEMO_LB_HW_PORT (unicast on loopback: up.sh runs it in the gateway's network namespace — the real
sender broadcasts; the gateway does not care how a datagram arrived):

  box-a  unified-memory box (the `node` cluster): two RoCE ports, mlx5_0/1 with `out_of_buffer` +3, `local_ack_timeout_err`
         +1 and ethtool's `rx_discards_phy` +2 per send (RISING: the RoCE ERR finding fires) and `rx_pause_ctrl_phy` +40
         (PFC at work: shown, not a fault), mlx5_1/1 steady; both PFC on prio 3, DSCP map 8 entries, trust absent with
         mlx5's own refusal; the kernel's Xid count +1 every 15 sends
         (code 79: the XID finding fires); two disks; one models root sized by folder.
  box-b  unified-memory box (telemetry only): RAM nearly full (RAM AVAIL under the demo's 10 % of its RAM: fires); its GPU
         97 % busy at 9.5 W (GPU POWER: ≤ the GB10 type's 8.9 W idle + the demo's 2 W margin: fires); mlx5_0/1 steady,
         mlx5_1/1 DOWN with `link_downed` +1 every 10 sends (RoCE ERR fires) and its ethtool refused (PHY counters
         absent with that why, PFC/DSCP unread); mlx5_0/1 PFC off; Xid 0 for the whole boot, OOM kills 3.
  box-c  discrete-GPU box: 87 °C, 3 °C below its own throttle limit (nvidia-smi `temperature.gpu.tlimit`; ≤ the demo's
         5 °C margin: GPU TEMP fires); no RDMA (the reason the real reader gives); Xid unread (the reason the real reader
         gives without kmsg.yml); one disk.
  box-d  an amdgpu iGPU box, telemetry only: its hwmon states no `temp1_crit`, so the headroom is absent with the
         amdgpu reader's why (GPU TEMP is no data, never a guess); no RDMA.

THE INPUTS FILE. After each send the last sample per box is written whole to DEMO_INPUTS (JSON: box → the datagram
dict) so the Playwright audit (audit.mjs) compares what the panel shows against what was sent — not against the
gateway's own status, which would check the gateway against itself.
"""
from __future__ import annotations

import json
import os
import socket
import sys
import time

sys.path.insert(0, "/app")

from fleet_telemetry.model import (DiskUsage, GpuReading, KernelReading, MemoryReading, ModelsRoot, NetCounters,  # noqa: E402
                                   RdmaPort, Sample)
from fleet_telemetry.readers.rdma import COUNTERS, PHY, PHY_COUNTERS  # noqa: E402
from fleet_telemetry.sampler import envelope  # noqa: E402

GB = 10 ** 9
MB = 1024  # MiB per GiB, for the memory figures (the sender reports MiB)


#: What mlx5 answers to `dcb apptrust` (it implements none): the trust figure is absent with the tool's words.
NO_APPTRUST = "`dcb apptrust show dev {nd} order`: Operation not supported"


def port(dev: str, n: int, *, state: str, counters: dict[str, int], rate: float | None = 200.0,
         mtu: int | None = 9000, pfc: list[int] | None = None, dscp: int | None = None, now: float | None = None,
         ethtool_why: str | None = None) -> RdmaPort:
    """One port as readers/rdma.py states it: the sysfs counters and (unless ethtool refused) mlx5e's PHY counters,
    cumulative; its QoS from dcb (PFC prios, DSCP map; trust absent with mlx5's own refusal)."""
    nd = f"enp1s0f{n - 1}np{n - 1}"
    every = {name: counters.get(name, 0) for _sub, name in COUNTERS}
    why = {} if state == "ACTIVE" else {"rate_gbps": "rate unreadable ('Invalid argument')"}
    if ethtool_why:
        why[PHY] = ethtool_why
    else:
        every.update({name: counters.get(name, 0) for name in PHY_COUNTERS})
    why["trust"] = NO_APPTRUST.format(nd=nd)
    return RdmaPort(dev=dev, port=n, state=state, phys_state="LinkUp" if state == "ACTIVE" else "Disabled",
                    rate_gbps=rate if state == "ACTIVE" else None, link_layer="Ethernet", netdev=nd,
                    mtu=mtu, counters=every, why=why, pfc_prios=pfc, trust=None, dscp_map=dscp,
                    qos_at=None if now is None else now - 42)


def unified_gpu(use: float, temp: float, power: float, headroom: float) -> GpuReading:
    return GpuReading(id=0, use_pct=use, vram_used_mb=None, vram_total_mb=None, temp_c=temp, power_w=power,
                      power_inst_w=power + 1.5, sm_mhz=2418.0, throttle_hex="0x0000000000000000", fan_pct=None,
                      temp_headroom_c=headroom)


def models_root(now: float) -> ModelsRoot:
    return ModelsRoot(root="/mnt/models", measured_at=now - 240, took_s=3.4,
                      dirs={"Qwen3.8-27B-NVFP4": 20_412 * 10 ** 6, "Qwen3.8-27B-NVFP4_uncensored": 20_398 * 10 ** 6,
                            "dflash2-27b-tp2": 23_871 * 10 ** 6, "MAI-UI-2B": 4_470 * 10 ** 6,
                            "scratch-exports": 61_200 * 10 ** 6},
                      more_count=4, more_bytes=9_330 * 10 ** 6)


def box_a(k: int, now: float, t0: float) -> Sample:
    xids = 2 + k // 15
    last_rise = t0 + (k // 15) * 15 * EVERY if k >= 15 else t0 - 3 * 3600
    return Sample(
        cpu_percent=22.5, cpu_per_core=[22.5] * 20,
        memory=MemoryReading(total_mb=121_684, used_mb=71_210, percent=58.5, swap_used_mb=0),
        gpus=[unified_gpu(41.0, 61.0, 38.2, 35.0)], cpu_temp_c=52.0, cpu_mhz=3510, fan_rpm_max=2100,
        net=NetCounters(rx_bytes=5_000_000_000 + k * 12_500_000, tx_bytes=3_000_000_000 + k * 8_000_000,
                        speed_mbps=10000),
        rdma=[port("mlx5_0", 1, state="ACTIVE", counters={"out_of_buffer": 1_200 + 3 * k, "local_ack_timeout_err": 40 + k,
                                                          "rx_discards_phy": 310 + 2 * k, "rx_pause_ctrl_phy": 9_000 + 40 * k},
                   pfc=[3], dscp=8, now=now),
              port("mlx5_1", 1, state="ACTIVE", counters={"out_of_buffer": 17}, pfc=[3], dscp=8, now=now)],
        kernel=KernelReading(xid_total=xids, oom_kills_total=0, last_xid={"code": 79, "at": last_rise},
                             xid_since=t0 - 86400, xid_complete=True, why=None),
        disks=[DiskUsage(path="/", total_bytes=1_967 * GB, used_bytes=612 * GB, free_bytes=1_255 * GB),
               DiskUsage(path="/mnt/models", total_bytes=3_841 * GB, used_bytes=3_402 * GB, free_bytes=439 * GB)],
        models=[models_root(now)])


def box_b(k: int, now: float, t0: float) -> Sample:
    return Sample(
        cpu_percent=18.0, cpu_per_core=[18.0] * 20,
        memory=MemoryReading(total_mb=121_684, used_mb=113_492, percent=93.3, swap_used_mb=512),
        gpus=[unified_gpu(97.0, 59.0, 9.5, 37.0)], cpu_temp_c=50.0, cpu_mhz=3480, fan_rpm_max=2050,
        net=NetCounters(rx_bytes=4_000_000_000 + k * 12_000_000, tx_bytes=2_500_000_000 + k * 7_500_000,
                        speed_mbps=10000),
        rdma=[port("mlx5_0", 1, state="ACTIVE", counters={"out_of_buffer": 980}, pfc=[], dscp=0, now=now),
              port("mlx5_1", 1, state="DOWN", counters={"link_downed": 2 + k // 10}, mtu=1500, now=now,
                   ethtool_why="`ethtool -S enp1s0f0np0`: no such device")],
        kernel=KernelReading(xid_total=0, oom_kills_total=3, last_xid=None, xid_since=t0 - 86400,
                             xid_complete=True, why=None),
        disks=[DiskUsage(path="/", total_bytes=1_967 * GB, used_bytes=588 * GB, free_bytes=1_379 * GB)],
        models=[models_root(now)])


def box_c(k: int, now: float, t0: float) -> Sample:
    gpu = GpuReading(id=0, use_pct=88.0, vram_used_mb=29_840, vram_total_mb=32_607, temp_c=87.0, power_w=431.0,
                     power_inst_w=440.0, sm_mhz=2610.0, throttle_hex="0x0000000000000004", fan_pct=71.0,
                     temp_headroom_c=3.0)
    return Sample(
        cpu_percent=31.0, cpu_per_core=[31.0] * 16,
        memory=MemoryReading(total_mb=64_128, used_mb=22_016, percent=34.3, swap_used_mb=0),
        gpus=[gpu], cpu_temp_c=64.0, cpu_mhz=4700, fan_rpm_max=1450,
        net=NetCounters(rx_bytes=900_000_000 + k * 1_250_000, tx_bytes=700_000_000 + k * 900_000, speed_mbps=2500),
        rdma=None, rdma_why="no /sys/class/infiniband: this box has no RDMA device (or its driver is not loaded)",
        kernel=KernelReading(xid_total=None, oom_kills_total=1, last_xid=None, xid_since=None, xid_complete=None,
                             why="no /dev/kmsg in this container: add lb/hw_sender/kmsg.yml to count Xid records"),
        disks=[DiskUsage(path="/", total_bytes=3_841 * GB, used_bytes=1_120 * GB, free_bytes=2_721 * GB)],
        models=None, models_why="HW_SENDER_MODELS_DIRS is none: no models root is sized on this box")


def box_d(k: int, now: float, t0: float) -> Sample:
    gpu = GpuReading(id=0, use_pct=12.0, vram_used_mb=None, vram_total_mb=None, temp_c=48.0, power_w=21.0,
                     power_inst_w=None, sm_mhz=600.0, throttle_hex=None, fan_pct=None, power_label="PPT",
                     temp_headroom_c=None, temp_headroom_why=AMD_NO_CRIT)
    return Sample(
        cpu_percent=9.0, cpu_per_core=[9.0] * 32,
        memory=MemoryReading(total_mb=125_000, used_mb=31_000, percent=24.8, swap_used_mb=0),
        gpus=[gpu], cpu_temp_c=51.0, cpu_mhz=3000, fan_rpm_max=None,
        net=NetCounters(rx_bytes=200_000_000 + k * 400_000, tx_bytes=150_000_000 + k * 300_000, speed_mbps=2500),
        rdma=None, rdma_why="no /sys/class/infiniband: this box has no RDMA device (or its driver is not loaded)",
        kernel=KernelReading(xid_total=None, oom_kills_total=0, last_xid=None, xid_since=None, xid_complete=None,
                             why="no /dev/kmsg in this container: add lb/hw_sender/kmsg.yml to count Xid records"),
        disks=[DiskUsage(path="/", total_bytes=1_967 * GB, used_bytes=402 * GB, free_bytes=1_565 * GB)],
        models=None, models_why="HW_SENDER_MODELS_DIRS is none: no models root is sized on this box")


#: The amdgpu reader's words for an edge sensor with no crit file (readers/gpu.py): the headroom is absent, said.
AMD_NO_CRIT = "no temp1_crit in the amdgpu hwmon: the driver states no limit for this card"
BOXES = (("box-a-host", box_a), ("box-b-host", box_b), ("box-c-host", box_c), ("box-d-host", box_d))
EVERY = float(os.environ.get("DEMO_SEND_EVERY_S") or 2)


def main() -> None:
    host, hw_port, out = os.environ["DEMO_LB_HOST"], int(os.environ["DEMO_LB_HW_PORT"]), os.environ["DEMO_INPUTS"]
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    t0 = time.time()
    k = 0
    print(f"[sender] {len(BOXES)} boxes -> {host}:{hw_port} every {EVERY:g} s", flush=True)
    while True:
        now = time.time()
        inputs = {}
        for hostname, make in BOXES:
            payload = envelope(make(k, now, t0), host=hostname, now=now)
            try:
                sock.sendto(payload.encode(), (host, hw_port))
            except OSError as exc:            # the gateway not up yet: the next send tries again
                print(f"[sender] {hostname}: {exc}", flush=True)
            inputs[hostname.removesuffix("-host")] = json.loads(payload)
        tmp = f"{out}.tmp"
        with open(tmp, "w") as fh:
            json.dump({"k": k, "t0": t0, "at": now, "every_s": EVERY, "boxes": inputs}, fh)
        os.replace(tmp, out)
        k += 1
        time.sleep(max(0.0, t0 + k * EVERY - time.time()))


if __name__ == "__main__":
    main()
