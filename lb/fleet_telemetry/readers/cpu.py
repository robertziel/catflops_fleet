"""CPU readers: utilisation from /proc/stat, and the CPU's own temperature and clock sensors.

ProcStat — utilisation over the window between two reads, as psutil computed it for the senders before (so the wire's
`cpu_percent` / `cpu_per_core` keep their meaning): per `cpu` line, all = the sum of the fields minus guest and
guest_nice (the kernel already counts them in user / nice), busy = all − idle − iowait, percent = Δbusy / Δall,
rounded to 0.1. The first window opens when the reader is made. A window with no ticks is None, not 0.

CpuSensors — the hottest of the CPU's OWN sensors: a CPU hwmon chip (coretemp, k10temp, zenpower, x86_pkg_temp,
cpu_thermal) or a CPU-typed thermal zone; and the mean cpufreq `scaling_cur_freq`. An ACPI board zone (`acpitz`) is
the BOARD's — the recorder already shows it as BOARD MAX — and /proc/cpuinfo's "cpu MHz" is not cpufreq: neither
stands in. A box without them has no CPU temperature / clock figure, and the Sampler says why.
"""
from __future__ import annotations

import glob
import os

from fleet_telemetry.readers.base import Files

CPU_HWMON = ("coretemp", "k10temp", "zenpower", "x86_pkg_temp", "cpu_thermal")
CPU_ZONE = ("x86_pkg", "cpu", "coretemp")


class ProcStat:
    source = "/proc/stat"

    def __init__(self, root: str = "/") -> None:
        self.fs = Files(root)
        self._last = self._times()

    def _times(self) -> list[tuple[int, int]]:
        """(all, busy) per `cpu` line: the aggregate first, then each core in the kernel's order."""
        text = Files.text(self.fs.path("/proc/stat"))
        if text is None:
            raise OSError("/proc/stat is unreadable")
        out = []
        for line in text.splitlines():
            if not line.startswith("cpu"):
                continue
            v = [int(x) for x in line.split()[1:]]
            v += [0] * (10 - len(v))
            total = sum(v) - v[8] - v[9]
            out.append((total, total - v[3] - v[4]))
        return out

    def read(self) -> tuple[float | None, list[float] | None]:
        now, last = self._times(), self._last
        self._last = now
        pct = []
        for (t1, b1), (t0, b0) in zip(now, last):
            dt = t1 - t0
            pct.append(None if dt <= 0 else round(min(100.0, max(0.0, (b1 - b0) / dt * 100)), 1))
        if len(now) != len(last) or not pct or pct[0] is None:
            return None, None
        cores = pct[1:]
        return pct[0], (cores if all(c is not None for c in cores) else None)


class CpuSensors:
    source = "cpu sensors"

    def __init__(self, root: str = "/") -> None:
        self.fs = Files(root)

    def read(self) -> tuple[float | None, int | None]:
        milli = [v for mon in self.fs.glob("/sys/class/hwmon/hwmon*")
                 if Files.text(os.path.join(mon, "name")) in CPU_HWMON
                 for v in (Files.int(f) for f in sorted(glob.glob(os.path.join(mon, "temp*_input"))))
                 if v is not None]
        for z in self.fs.glob("/sys/class/thermal/thermal_zone*"):
            t = (Files.text(os.path.join(z, "type")) or "").lower()
            v = Files.int(os.path.join(z, "temp"))
            if v is not None and any(k in t for k in CPU_ZONE):
                milli.append(v)
        khz = [v for v in (Files.int(f) for f in
                           self.fs.glob("/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_cur_freq")) if v is not None]
        return (round(max(milli) / 1000, 1) if milli else None,
                round(sum(khz) / len(khz) / 1000) if khz else None)
