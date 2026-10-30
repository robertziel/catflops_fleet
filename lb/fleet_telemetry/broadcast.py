"""The LAN broadcaster: one UDP datagram per sample, to the default-route interface's broadcast address and to
255.255.255.255 (some stacks deliver one and drop the other; the gateway keys on the host, so a duplicate is the same
sample twice). A listener needs no address for the box, and one datagram serves every gateway on the segment.

Never fatal: a send that fails is logged and the next sample tries again — a box's telemetry must not stop for a
downed interface. The socket is non-blocking: a full buffer drops this sample, the next supersedes it.
"""
from __future__ import annotations

import fcntl
import logging
import socket
import struct

log = logging.getLogger("fleet_telemetry")

#: linux/sockios.h — an interface's broadcast address.
SIOCGIFBRDADDR = 0x8919
LIMITED = "255.255.255.255"


def _default_interface(route: str = "/proc/net/route") -> str | None:
    try:
        with open(route) as fh:
            for line in fh.readlines()[1:]:
                parts = line.split()
                if len(parts) > 2 and parts[1] == "00000000":
                    return parts[0]
    except OSError:
        pass
    return None


def _broadcast_address(name: str) -> str | None:
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        packed = fcntl.ioctl(probe.fileno(), SIOCGIFBRDADDR, struct.pack("256s", name[:15].encode()))
        addr = socket.inet_ntoa(packed[20:24])
    except OSError:
        return None
    finally:
        probe.close()
    return addr if addr and addr != "0.0.0.0" else None


def discover_targets() -> tuple[str, ...]:
    """The default route's broadcast address (when it has one), then the limited broadcast."""
    iface = _default_interface()
    addr = _broadcast_address(iface) if iface else None
    return tuple(a for a in (addr, LIMITED) if a)


class LanBroadcaster:
    def __init__(self, *, port: int, targets: tuple[str, ...] | None = None) -> None:
        self.port = port
        self.targets = targets if targets is not None else discover_targets()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self._sock.setblocking(False)

    def publish(self, payload: bytes) -> int:
        """Send to every target; how many took it."""
        sent = 0
        for addr in self.targets:
            try:
                self._sock.sendto(payload, (addr, self.port))
                sent += 1
            except OSError as exc:
                log.warning("send to %s:%s failed: %s", addr, self.port, exc)
        return sent
