"""NetDev — the box's network counters from /proc/net/dev: the PHYSICAL NICs and bonds that are up and not a bond's
slave (its bytes are its master's), summed. Loopback, container plumbing (docker*, veth*, br-*, …) and tunnels
(tailscale*, wg*, … — they ride on the wire and would count it twice) are not the box's wire. The counters are
monotonic; the gateway's recorder derives the rate from two samples. Meaningful only in the HOST network namespace:
the Sampler does not read it for a bridged sender.
"""
from __future__ import annotations

import os

from fleet_telemetry.model import NetCounters
from fleet_telemetry.readers.base import Files

VIRTUAL = ("lo", "docker", "veth", "br-", "virbr", "tap", "tun", "cni", "flannel", "kube", "cali", "vnet", "vmnet",
           "vxlan", "wg", "tailscale", "zt", "ppp", "gre", "sit")


class NetDev:
    source = "/proc/net/dev"

    def __init__(self, root: str = "/") -> None:
        self.fs = Files(root)

    def read(self) -> NetCounters:
        text = Files.text(self.fs.path("/proc/net/dev"))
        if text is None:
            raise OSError("/proc/net/dev is unreadable")
        rx = tx = 0
        speeds = []
        for line in text.splitlines()[2:]:
            name, _, rest = line.partition(":")
            name, cols = name.strip(), rest.split()
            if not name or name.startswith(VIRTUAL) or len(cols) < 16:
                continue
            base = self.fs.path(f"/sys/class/net/{name}")
            if os.path.exists(os.path.join(base, "master")) or Files.text(os.path.join(base, "operstate")) == "down":
                continue
            rx, tx = rx + int(cols[0]), tx + int(cols[8])
            speed = Files.int(os.path.join(base, "speed"))
            if speed is not None and speed > 0:
                speeds.append(speed)
        return NetCounters(rx_bytes=rx, tx_bytes=tx, speed_mbps=max(speeds) if speeds else None)
