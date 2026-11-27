"""GPU readers, one per kind a server can NAME (HW_SENDER_GPU, set by lb/hw_sender/gpu-<kind>.yml).

Never auto-detected: a box with both CLIs on PATH, or a driver that is not up yet, must not be read as another kind
or as "no GPU". The kind is named; its reader either answers or the sender refuses to start (__main__ preflight).

NvidiaSmi — ONE `nvidia-smi --query-gpu` per sample for both the tiles' row (index, utilisation, memory) and the
recorder's figures (temperature, averaged and instantaneous draw, SM clock, throttle-reason mask, fan). `[N/A]` — a
unified-memory card's VRAM, a fanless card's fan — is None. Then ONE MORE call, `--query-gpu=index,temperature.gpu.tlimit`:
the temperature HEADROOM (°C below the GPU's own throttle limit — a GB10 read 49 at 47 °C; the health GPU TEMP finding is
held against it, vllm_lb/health.py). Its own call because a driver that does not know the field (before R530) refuses
the WHOLE query it is in: in the main one it would cost the box every GPU figure (review r4717289755). That call failing
leaves every card's headroom None with nvidia-smi's own words as the why; a `[N/A]` likewise — never a substitute.

AmdgpuSysfs — the amdgpu driver's own sysfs under /sys/class/drm/card<N>/device: `gpu_busy_percent`,
`mem_info_vram_used/total` (exactly what rocm-smi reads for `--showuse --showmeminfo vram`), and the card's hwmon:
the `edge` temperature (and its `_crit`, when the driver states one: headroom = crit − input; an APU's hwmon often
has no crit, and the headroom is then absent naming the missing file), `power1_average` / `power1_input` (µW) with `power1_label`, the `sclk` (Hz). No rocm-smi
process per sample (a Python program costing a second of CPU) and no ROCm tree, device node or privilege: sysfs is
readable from any container. amdgpu has no throttle-reason mask.
"""
from __future__ import annotations

import glob
import os
import re
import subprocess
from dataclasses import replace
from typing import Callable

from fleet_telemetry.model import GpuReading
from fleet_telemetry.readers.base import Files

MIB = 1024 * 1024


def _num(v: str) -> float | int | None:
    """smi text → a number; `[N/A]`, `N/A`, empty → None."""
    try:
        f = float(v.strip())
    except ValueError:
        return None
    return int(f) if f.is_integer() else f


def _run(argv: list[str]) -> str:
    done = subprocess.run(argv, capture_output=True, text=True, timeout=5)
    if done.returncode != 0:
        raise RuntimeError(f"{argv[0]} exited {done.returncode}: {(done.stderr or done.stdout).strip()[:200]}")
    return done.stdout


class NvidiaSmi:
    source = "nvidia-smi"
    FIELDS = ("index", "utilization.gpu", "memory.used", "memory.total", "temperature.gpu", "power.draw",
              "power.draw.instant", "clocks.sm", "clocks_throttle_reasons.active", "fan.speed")
    ARGV = ["nvidia-smi", f"--query-gpu={','.join(FIELDS)}", "--format=csv,noheader,nounits"]
    HEADROOM_ARGV = ["nvidia-smi", "--query-gpu=index,temperature.gpu.tlimit", "--format=csv,noheader,nounits"]

    def __init__(self, run: Callable[[list[str]], str] = _run) -> None:
        self.run = run

    def read(self) -> list[GpuReading]:
        out = []
        for line in self.run(self.ARGV).splitlines():
            p = [x.strip() for x in line.split(",")]
            if len(p) != len(self.FIELDS) or _num(p[0]) is None:
                continue
            mask = p[8] if p[8].lower().startswith("0x") else None
            out.append(GpuReading(id=int(p[0]), use_pct=_num(p[1]), vram_used_mb=_num(p[2]), vram_total_mb=_num(p[3]),
                                  temp_c=_num(p[4]), power_w=_num(p[5]), power_inst_w=_num(p[6]), sm_mhz=_num(p[7]),
                                  throttle_hex=mask, fan_pct=_num(p[9])))
        if not out:
            return out                             # no card: no headroom to ask for
        heads, failed = self._headrooms()
        return [replace(g, **self._headroom(g.id, heads, failed)) for g in out]

    def _headrooms(self) -> tuple[dict[int, str], str | None]:
        """(card index → its raw tlimit answer, why the call failed) — the headroom call, run once per sample."""
        try:
            text = self.run(self.HEADROOM_ARGV)
        except Exception as exc:                   # noqa: BLE001 — its failure is the headroom's, never the sample's
            return {}, f"nvidia-smi --query-gpu=index,temperature.gpu.tlimit failed: {str(exc)[:200]}"
        heads = {}
        for line in text.splitlines():
            p = [x.strip() for x in line.split(",")]
            if len(p) == 2 and isinstance(_num(p[0]), int):
                heads[_num(p[0])] = p[1]
        return heads, None

    @staticmethod
    def _headroom(card: int, heads: dict[int, str], failed: str | None) -> dict:
        if failed:
            return {"temp_headroom_c": None, "temp_headroom_why": failed}
        if card not in heads:
            return {"temp_headroom_c": None,
                    "temp_headroom_why": f"nvidia-smi temperature.gpu.tlimit listed no row for card {card}"}
        v = _num(heads[card])
        return {"temp_headroom_c": v, "temp_headroom_why": None if v is not None else
                f"nvidia-smi temperature.gpu.tlimit answered {heads[card] or 'nothing'} (N/A: the driver states no "
                f"T.Limit for this card)"}


class AmdgpuSysfs:
    source = "amdgpu sysfs"
    VENDOR = "0x1002"

    def __init__(self, root: str = "/") -> None:
        self.fs = Files(root)

    def _channel(self, mon: str, kind: str, label: str) -> str | None:
        """The hwmon channel (`temp1`, `freq1` …) whose label is `label`."""
        for f in sorted(glob.glob(os.path.join(mon, f"{kind}*_label"))):
            if Files.text(f) == label:
                return f[: -len("_label")]
        return None

    def read(self) -> list[GpuReading]:
        out = []
        cards = [c for c in self.fs.glob("/sys/class/drm/card*") if re.fullmatch(r"card\d+", os.path.basename(c))]
        for card in sorted(cards, key=lambda c: int(os.path.basename(c)[4:])):
            dev = os.path.join(card, "device")
            if Files.text(os.path.join(dev, "vendor")) != self.VENDOR:
                continue
            mons = [m for m in sorted(glob.glob(os.path.join(dev, "hwmon", "hwmon*")))
                    if Files.text(os.path.join(m, "name")) == "amdgpu"]
            mon = mons[0] if mons else None
            temp = self._channel(mon, "temp", "edge") if mon else None
            sclk = self._channel(mon, "freq", "sclk") if mon else None
            milli = Files.int(temp + "_input") if temp else None
            crit = Files.int(temp + "_crit") if temp else None
            if milli is not None and crit is not None:
                headroom, h_why = round((crit - milli) / 1000, 1), None
            else:
                headroom = None
                h_why = (f"no {os.path.basename(temp)}_crit in the amdgpu hwmon: the driver states no limit for this "
                         f"card" if temp and milli is not None else "no `edge` temperature in the amdgpu hwmon")
            avg = Files.int(os.path.join(mon, "power1_average")) if mon else None
            inst = Files.int(os.path.join(mon, "power1_input")) if mon else None
            hz = Files.int(sclk + "_input") if sclk else None
            used, total = Files.int(os.path.join(dev, "mem_info_vram_used")), Files.int(
                os.path.join(dev, "mem_info_vram_total"))
            out.append(GpuReading(
                id=int(os.path.basename(card)[4:]), use_pct=Files.int(os.path.join(dev, "gpu_busy_percent")),
                vram_used_mb=used // MIB if used is not None else None,
                vram_total_mb=total // MIB if total is not None else None,
                temp_c=round(milli / 1000, 1) if milli is not None else None,
                power_w=round(avg / 1e6, 2) if avg is not None else None,
                power_inst_w=round(inst / 1e6, 2) if inst is not None else None,
                sm_mhz=round(hz / 1e6) if hz is not None else None, throttle_hex=None, fan_pct=None,
                power_label=Files.text(os.path.join(mon, "power1_label")) if mon else None,
                temp_headroom_c=headroom, temp_headroom_why=h_why))
        return out


#: HW_SENDER_GPU → its reader.
GPU_READERS = {"nvidia": NvidiaSmi, "amd": AmdgpuSysfs}
