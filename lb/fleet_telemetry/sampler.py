"""The Sampler: the box's readers composed into one Sample, and the envelope that goes on the wire.

POLICY LIVES HERE, not in the readers: a reader that raises costs only ITS figures, each absent with the reason
(`Sample.why`, keyed by figures.KEYS — what the gateway shows in place of its generic n/a); a figure a source does not
state is absent with why that source is silent. The sender never dies for a sensor.

`net=None` is the BRIDGED sender (lb/hw_sender/bridged.yml): its /proc/net/dev is a container's veth, not the box —
so there are no counters, with that reason, never the container's figures under the box's name.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from fleet_telemetry import figures as F
from fleet_telemetry.model import Sample
from fleet_telemetry.readers.base import Reader

GPU_KEYS = ("gpu_temp_c", "gpu_power_w", F.GPU_POWER_INST_W, "gpu_mhz", "throttle")
BRIDGED = "the sender runs bridged (bridged.yml): its network namespace is a container's, not the box's"


class Sampler:
    def __init__(self, *, gpu: Reader, cpu: Reader, memory: Reader, sensors: Reader, zones: Reader, fans: Reader,
                 net: Reader | None, containers: Reader) -> None:
        self.gpu, self.cpu, self.memory, self.sensors = gpu, cpu, memory, sensors
        self.zones, self.fans, self.net, self.containers = zones, fans, net, containers

    @staticmethod
    def _read(reader: Reader, why: dict[str, str], keys: tuple[str, ...]):
        """(value, None) or (None, reason) — the reason also lands in `why` under every key the reader feeds."""
        try:
            return reader.read(), None
        except Exception as exc:  # noqa: BLE001 — any failure is this sample's reason, not the sender's death
            reason = f"{reader.source} failed: {exc}"
            for k in keys:
                why[k] = reason
            return None, reason

    def sample(self) -> Sample:
        why: dict[str, str] = {}
        s = Sample(why=why)
        cpu, _ = self._read(self.cpu, why, ("cpu_pct",))
        if cpu is not None:
            s.cpu_percent, s.cpu_per_core = cpu
        s.memory, _ = self._read(self.memory, why, ())
        gpus, err = self._read(self.gpu, why, GPU_KEYS)
        s.gpus = gpus or []
        if err is None:
            self._gpu_why(s, why)
        zones, _ = self._read(self.zones, why, ("board_temp_c",))
        s.zones = zones or []
        if zones == []:
            why["board_temp_c"] = "the box exposes no /sys/class/thermal zone"
        sensors, err = self._read(self.sensors, why, ("cpu_temp_c", "cpu_mhz"))
        if sensors is not None:
            s.cpu_temp_c, s.cpu_mhz = sensors
            if s.cpu_temp_c is None:
                why["cpu_temp_c"] = ("no CPU sensor: no coretemp/k10temp/zenpower/x86_pkg_temp/cpu_thermal hwmon and "
                                     "no cpu thermal zone (a board zone is BOARD MAX, not the CPU's)")
            if s.cpu_mhz is None:
                why["cpu_mhz"] = "no cpufreq scaling_cur_freq on this box"
        s.fan_rpm_max, err = self._read(self.fans, why, ("fan_rpm",))
        if err is None and s.fan_rpm_max is None:
            why["fan_rpm"] = "no fan tachometer (hwmon fan*_input) on this box"
        if self.net is None:
            why["net_in_mbps"] = why["net_out_mbps"] = BRIDGED
        else:
            s.net, _ = self._read(self.net, why, ("net_in_mbps", "net_out_mbps"))
        s.containers, s.containers_why = self._read(self.containers, {}, ())
        if s.containers_why:
            s.containers_why = f"the containers could not be read: {s.containers_why}"
        return s

    @staticmethod
    def _gpu_why(s: Sample, why: dict[str, str]) -> None:
        if not s.gpus:
            for k in GPU_KEYS:
                why[k] = "the GPU reader lists no card"
            return
        for attr, key, what in (("temp_c", "gpu_temp_c", "temperature"), ("power_w", "gpu_power_w", "averaged draw"),
                                ("power_inst_w", F.GPU_POWER_INST_W, "instantaneous draw"),
                                ("sm_mhz", "gpu_mhz", "clock")):
            missing = [str(g.id) for g in s.gpus if getattr(g, attr) is None]
            if missing:
                why[key] = f"card {', '.join(missing)} reports no {what}"
        if all(g.throttle_hex is None for g in s.gpus):
            why["throttle"] = ("no card reports a throttle-reason mask (an NVIDIA/NVML reading; amdgpu exposes none)")


def envelope(sample: Sample, *, host: str, now: float) -> str:
    """The datagram: when, from where, and the sample. `host` is the name the gateway maps to a box."""
    return json.dumps({F.SAMPLED_AT: datetime.fromtimestamp(now, timezone.utc).isoformat(), F.HOST: host,
                       **sample.wire()}, separators=(",", ":"))
