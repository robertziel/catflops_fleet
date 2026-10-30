"""The domain: what a box's hardware reading IS, independent of where it was read and of the wire.

Readers (readers/) produce these values; the Sampler (sampler.py) assembles one Sample; `Sample.wire()` is the only
place the wire's field names (figures.py) meet the model. Every figure is Optional: None is "the source does not say",
which the wire carries as null and the gateway shows as n/a with the reason — never as 0.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fleet_telemetry import figures as F

#: NVML clock-throttle-reason bits → (name, 0-100 severity). The public NVML constants; ONE hw-thermal bit already
#: means "at the wall", so a card's index is the MAX over its active bits, never a sum.
THROTTLE_BITS: tuple[tuple[int, str, int], ...] = (
    (0x001, "gpu_idle", 0),
    (0x002, "applications_clocks_setting", 5),
    (0x004, "sw_power_cap", 30),
    (0x008, "hw_slowdown", 85),
    (0x010, "sync_boost", 5),
    (0x020, "sw_thermal", 50),
    (0x040, "hw_thermal", 100),
    (0x080, "hw_power_brake", 90),
    (0x100, "display_clock_setting", 5),
)


def _mask(mask_hex: str | None) -> int | None:
    try:
        return int(mask_hex.strip(), 16) if isinstance(mask_hex, str) and mask_hex.strip() else None
    except ValueError:
        return None


def throttle_index(mask_hex: str | None) -> int | None:
    """0-100 severity of a throttle-reason mask; None when the card reports none (a gap, not a zero)."""
    mask = _mask(mask_hex)
    if mask is None:
        return None
    return max((w for bit, _n, w in THROTTLE_BITS if mask & bit), default=0)


def throttle_reasons(mask_hex: str | None) -> list[str] | None:
    """The mask's active reasons by NVML name, in bit order ([] for a zero mask); None when the card reports none.
    The panel shows these, not the index: "hw_slowdown" says what the card did, "85" only how bad it is."""
    mask = _mask(mask_hex)
    if mask is None:
        return None
    return [n for bit, n, _w in THROTTLE_BITS if mask & bit]


@dataclass(frozen=True)
class GpuReading:
    """One card. `power_w` is the driver's averaged draw, `power_inst_w` its instantaneous one; `power_label` names
    what the sensor measures when the driver says (an APU's `PPT` is the whole package, CPU and GPU)."""
    id: int
    use_pct: float | None
    vram_used_mb: int | None
    vram_total_mb: int | None
    temp_c: float | None
    power_w: float | None
    power_inst_w: float | None
    sm_mhz: float | None
    throttle_hex: str | None
    fan_pct: float | None
    power_label: str | None = None

    @property
    def throttle_index(self) -> int | None:
        return throttle_index(self.throttle_hex)

    def row(self) -> dict[str, Any]:
        """The tiles' card row — the shape every gateway reads."""
        return {"id": self.id, "use_pct": self.use_pct, "vram_used_mb": self.vram_used_mb,
                "vram_total_mb": self.vram_total_mb}

    def deep(self) -> dict[str, Any]:
        out = {F.CARD_TEMP_C: self.temp_c, F.CARD_POWER_W: self.power_w, F.CARD_POWER_INST_W: self.power_inst_w,
               F.CARD_SM_MHZ: self.sm_mhz, F.CARD_THROTTLE_HEX: self.throttle_hex,
               F.CARD_THROTTLE_INDEX: self.throttle_index, F.CARD_FAN_PCT: self.fan_pct}
        if self.power_label:
            out[F.CARD_POWER_LABEL] = self.power_label
        return out


@dataclass(frozen=True)
class ThermalZone:
    type: str
    milli_c: int


@dataclass(frozen=True)
class MemoryReading:
    """The box's RAM as `free` reads it: used = total − available (page cache that can be dropped is not used)."""
    total_mb: int
    used_mb: int
    percent: float
    swap_used_mb: int | None


@dataclass(frozen=True)
class NetCounters:
    """The box's physical NICs: summed MONOTONIC byte counters (a rate is the gateway's delta) and the fastest link."""
    rx_bytes: int
    tx_bytes: int
    speed_mbps: int | None


@dataclass(frozen=True)
class ContainerUsage:
    name: str
    image: str | None
    ram_mb: int | None


@dataclass
class ContainerSet:
    """Every running container. `ram_mb` is None when one container's RAM could not be read: a total of the others
    would understate the box."""
    containers: list[ContainerUsage]

    @property
    def count(self) -> int:
        return len(self.containers)

    @property
    def ram_mb(self) -> int | None:
        rams = [c.ram_mb for c in self.containers]
        return sum(rams) if all(r is not None for r in rams) else None

    def top(self, n: int) -> list[ContainerUsage]:
        """The `n` largest by RAM; a container with no RAM figure sorts last."""
        return sorted(self.containers, key=lambda c: -(c.ram_mb if c.ram_mb is not None else -1))[:n]


@dataclass
class Sample:
    """One reading of the whole box. `why` maps a figure key (figures.KEYS) to the reason it is absent."""
    cpu_percent: float | None = None
    cpu_per_core: list[float] | None = None
    memory: MemoryReading | None = None
    gpus: list[GpuReading] = field(default_factory=list)
    zones: list[ThermalZone] = field(default_factory=list)
    cpu_temp_c: float | None = None
    cpu_mhz: int | None = None
    fan_rpm_max: int | None = None
    net: NetCounters | None = None
    containers: ContainerSet | None = None
    containers_why: str | None = None
    why: dict[str, str] = field(default_factory=dict)

    def wire(self) -> dict[str, Any]:
        m, net = self.memory, self.net
        worst = [g.throttle_index for g in self.gpus if g.throttle_index is not None]
        out: dict[str, Any] = {
            F.CPU_PERCENT: self.cpu_percent,
            F.RAM_PERCENT: m.percent if m else None,
            F.RAM_USED_MB: m.used_mb if m else None,
            F.RAM_TOTAL_MB: m.total_mb if m else None,
            F.SWAP_USED_MB: m.swap_used_mb if m else None,
            F.GPUS: [g.row() for g in self.gpus],
            F.CPU_PER_CORE: self.cpu_per_core,
            F.DEEP: {
                F.DEEP_GPU: [g.deep() for g in self.gpus],
                F.DEEP_THROTTLE_INDEX: max(worst) if worst else None,
                F.DEEP_ZONES: [{F.ZONE_TYPE: z.type, F.ZONE_MILLI_C: z.milli_c} for z in self.zones],
                F.DEEP_CPU: {F.CPU_TEMP_C: self.cpu_temp_c, F.CPU_FREQ_MHZ: self.cpu_mhz},
                F.DEEP_FAN_RPM_MAX: self.fan_rpm_max,
                F.NET_RX_BYTES: net.rx_bytes if net else None,
                F.NET_TX_BYTES: net.tx_bytes if net else None,
                F.NET_SPEED_MBPS: net.speed_mbps if net else None,
                F.DEEP_WHY: dict(self.why),
            },
        }
        if self.containers is not None:
            out[F.CONTAINERS] = [{F.C_NAME: c.name, F.C_IMAGE: c.image, F.C_RAM_MB: c.ram_mb}
                                 for c in self.containers.top(F.SENT_CONTAINERS)]
            out[F.CONTAINERS_TOTAL] = {F.TOTAL_COUNT: self.containers.count, F.TOTAL_RAM_MB: self.containers.ram_mb}
        else:
            out[F.CONTAINERS_WHY] = self.containers_why
        return out
