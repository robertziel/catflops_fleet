"""One reader per SOURCE, behind one interface (base.Reader): `source` names it (the reason a failed read gives) and
`read()` returns domain values (model.py) or raises. A reader holds no policy: what an absent figure MEANS, and the
reason the wire carries, is the Sampler's (sampler.py).

A GPU kind is a reader class in GPU_READERS — a new kind is a new class here, never a branch elsewhere.
"""
from __future__ import annotations

from fleet_telemetry.readers.base import Reader
from fleet_telemetry.readers.containers import Cgroups
from fleet_telemetry.readers.cpu import CpuSensors, ProcStat
from fleet_telemetry.readers.gpu import GPU_READERS, AmdgpuSysfs, NvidiaSmi
from fleet_telemetry.readers.memory import Meminfo
from fleet_telemetry.readers.net import NetDev
from fleet_telemetry.readers.thermal import Fans, ThermalZones

__all__ = ["GPU_READERS", "AmdgpuSysfs", "Cgroups", "CpuSensors", "Fans", "Meminfo", "NetDev", "NvidiaSmi",
           "ProcStat", "Reader", "ThermalZones"]
