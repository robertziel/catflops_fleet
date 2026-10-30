"""Meminfo — the box's RAM and swap from /proc/meminfo, as psutil reported them to the senders before:
total = MemTotal, used = MemTotal − MemAvailable, percent = used / total rounded to 0.1, swap used = SwapTotal −
SwapFree. In MB (2^20 bytes, floored). A kernel without MemAvailable has no `used` figure here — `MemFree` is not it.
"""
from __future__ import annotations

from fleet_telemetry.model import MemoryReading
from fleet_telemetry.readers.base import Files


class Meminfo:
    source = "/proc/meminfo"

    def __init__(self, root: str = "/") -> None:
        self.fs = Files(root)

    def read(self) -> MemoryReading:
        text = Files.text(self.fs.path("/proc/meminfo"))
        if text is None:
            raise OSError("/proc/meminfo is unreadable")
        kb = {ln.split(":")[0]: int(ln.split()[1]) for ln in text.splitlines() if ":" in ln and len(ln.split()) >= 2}
        missing = [k for k in ("MemTotal", "MemAvailable") if k not in kb]
        if missing:
            raise ValueError(f"/proc/meminfo has no {', '.join(missing)}")
        used = kb["MemTotal"] - kb["MemAvailable"]
        swap = kb["SwapTotal"] - kb["SwapFree"] if "SwapTotal" in kb and "SwapFree" in kb else None
        return MemoryReading(total_mb=kb["MemTotal"] // 1024, used_mb=used // 1024,
                             percent=round(used / kb["MemTotal"] * 100, 1),
                             swap_used_mb=swap // 1024 if swap is not None else None)
