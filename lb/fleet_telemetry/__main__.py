"""`python -m fleet_telemetry [--once]` — the sender: settings, preflight, then one broadcast every INTERVAL_S.

SETTINGS are the environment the compose files give it, every one REQUIRED (no default, no detection):
  HW_SENDER_GPU      the GPU kind its gpu-*.yml names (readers.GPU_READERS)
  HW_SENDER_NETWORK  host (docker-compose.yml) | bridged (bridged.yml): whether /proc/net/dev is the box's
  HW_SENDER_PORT     the gateway's LB_HW_PORT (lb/hw_sender/.env)
and two read-only mounts at the image's paths (CGROUP_ROOT, CONTAINERS_DIR; docker-compose.yml). A retired engine
name (QUEUE_WORKFLOWS_*) still set is refused, naming its replacement.

PREFLIGHT: the named GPU kind's reader answers with at least one card. Otherwise nothing is sent: an empty GPU row
would look exactly like a box without a GPU. Refusals exit with sysexits codes — 78 (EX_CONFIG) for configuration,
69 (EX_UNAVAILABLE) for hardware — so `docker ps` (Restarting (78)) tells them apart, and `restart: unless-stopped`
retries a driver that is not up yet at boot.

`--once` prints ONE envelope (after a one-second window, so the CPU figure covers a real interval) and sends nothing:
a check on a box that does not disturb the running sender.
"""
from __future__ import annotations

import logging
import os
import socket
import sys
import time
from dataclasses import dataclass
from typing import Mapping

from fleet_telemetry.broadcast import LanBroadcaster
from fleet_telemetry.readers import (GPU_READERS, Cgroups, CpuSensors, Fans, Meminfo, NetDev, ProcStat,
                                     ThermalZones)
from fleet_telemetry.sampler import Sampler, envelope

INTERVAL_S = 5.0
NETWORKS = ("host", "bridged")
#: The host's /sys/fs/cgroup and <DockerRootDir>/containers, mounted read-only by docker-compose.yml.
CGROUP_ROOT = "/host/sys/fs/cgroup"
CONTAINERS_DIR = "/host/docker/containers"
#: Names the sender read when it was the engine's; refused so a stale .env or compose file is found, not ignored.
RETIRED = {"QUEUE_WORKFLOWS_HW_BROADCAST_PORT": "HW_SENDER_PORT",
           "QUEUE_WORKFLOWS_BROADCAST_HW_STATS": "nothing (this sender always broadcasts)"}
EX_UNAVAILABLE, EX_CONFIG = 69, 78

log = logging.getLogger("fleet_telemetry")


class Refused(Exception):
    def __init__(self, message: str, code: int) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class Settings:
    gpu_kind: str
    network: str
    port: int

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "Settings":
        stale = sorted(k for k in env if k.startswith("QUEUE_WORKFLOWS_"))
        if stale:
            raise Refused("retired names are set: " + "; ".join(
                f"{k} → {RETIRED.get(k, 'nothing (not read)')}" for k in stale), EX_CONFIG)
        kind = env.get("HW_SENDER_GPU", "")
        if kind not in GPU_READERS:
            raise Refused(f"HW_SENDER_GPU={kind!r}: COMPOSE_FILE in lb/hw_sender/.env must add exactly one of "
                          f"{', '.join(f'gpu-{k}.yml' for k in GPU_READERS)} after docker-compose.yml", EX_CONFIG)
        network = env.get("HW_SENDER_NETWORK", "")
        if network not in NETWORKS:
            raise Refused(f"HW_SENDER_NETWORK={network!r}: one of {', '.join(NETWORKS)} "
                          f"(docker-compose.yml sets host, bridged.yml bridged)", EX_CONFIG)
        raw = env.get("HW_SENDER_PORT", "")
        if not raw.isdigit() or not 0 < int(raw) < 65536:
            raise Refused(f"HW_SENDER_PORT={raw!r}: the gateway's LB_HW_PORT, a UDP port (lb/hw_sender/.env)",
                          EX_CONFIG)
        return cls(gpu_kind=kind, network=network, port=int(raw))


def sampler_for(settings: Settings, root: str = "/") -> Sampler:
    try:
        containers = Cgroups(CGROUP_ROOT, CONTAINERS_DIR)
    except ValueError as exc:
        raise Refused(str(exc), EX_CONFIG) from exc
    return Sampler(gpu=GPU_READERS[settings.gpu_kind](), cpu=ProcStat(root), memory=Meminfo(root),
                   sensors=CpuSensors(root), zones=ThermalZones(root), fans=Fans(root),
                   net=NetDev(root) if settings.network == "host" else None, containers=containers)


def preflight(sampler: Sampler, kind: str) -> str:
    grant = f"lb/hw_sender/gpu-{kind}.yml"
    try:
        gpus = sampler.gpu.read()
    except Exception as exc:  # noqa: BLE001 — named in the refusal
        raise Refused(f"HW_SENDER_GPU={kind}: {sampler.gpu.source} failed: {exc} ({grant}; does the box have the "
                      f"driver?)", EX_UNAVAILABLE) from exc
    if not gpus:
        raise Refused(f"HW_SENDER_GPU={kind}: {sampler.gpu.source} reports no GPU ({grant}; does the box have the "
                      f"driver and that GPU?)", EX_UNAVAILABLE)
    return f"HW_SENDER_GPU={kind}: {sampler.gpu.source} reports {len(gpus)} GPU(s)"


def run(sampler: Sampler, broadcaster: LanBroadcaster, *, interval_s: float = INTERVAL_S, samples: int | None = None,
        host: str | None = None) -> None:
    """Sample and broadcast every `interval_s`; forever unless `samples` is given."""
    host = host or socket.gethostname()
    sent = 0
    while samples is None or sent < samples:
        started = time.monotonic()
        try:
            broadcaster.publish(envelope(sampler.sample(), host=host, now=time.time()).encode())
        except Exception:  # noqa: BLE001 — one bad sample must not end a box's telemetry
            log.exception("sample failed")
        sent += 1
        if sent % 60 == 0:
            log.info("sent %d samples", sent)
        time.sleep(max(0.0, interval_s - (time.monotonic() - started)))


def main(argv: list[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    try:
        settings = Settings.from_env(os.environ)
        sampler = sampler_for(settings)
        ok = preflight(sampler, settings.gpu_kind)
    except Refused as exc:
        print(f"fleet_telemetry: REFUSED — {exc}", file=sys.stderr, flush=True)
        return exc.code
    log.info("preflight ok: %s", ok)
    if "--once" in argv:
        sampler.sample()
        time.sleep(1.0)
        print(envelope(sampler.sample(), host=socket.gethostname(), now=time.time()), flush=True)
        return 0
    caster = LanBroadcaster(port=settings.port)
    log.info("broadcasting to %s on port %s every %.0f s", caster.targets, settings.port, INTERVAL_S)
    run(sampler, caster)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
