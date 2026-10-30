"""ThermalZones — every /sys/class/thermal zone as the kernel lists it (type, milli-°C); the gateway's BOARD MAX
is their hottest. Fans — the loudest hwmon tachometer (`fan*_input`, rpm) on any chip; None when the box has none
(many SoC boxes): a missing tach is not a stopped fan, while a tach reading 0 is.
"""
from __future__ import annotations

import glob
import os

from fleet_telemetry.model import ThermalZone
from fleet_telemetry.readers.base import Files


class ThermalZones:
    source = "/sys/class/thermal"

    def __init__(self, root: str = "/") -> None:
        self.fs = Files(root)

    def read(self) -> list[ThermalZone]:
        out = []
        for z in self.fs.glob("/sys/class/thermal/thermal_zone*"):
            t, milli = Files.text(os.path.join(z, "type")), Files.int(os.path.join(z, "temp"))
            if t is not None and milli is not None:
                out.append(ThermalZone(type=t, milli_c=milli))
        return out


class Fans:
    source = "hwmon fans"

    def __init__(self, root: str = "/") -> None:
        self.fs = Files(root)

    def read(self) -> int | None:
        rpm = [v for mon in self.fs.glob("/sys/class/hwmon/hwmon*")
               for v in (Files.int(f) for f in sorted(glob.glob(os.path.join(mon, "fan*_input")))) if v is not None]
        return max(rpm) if rpm else None
