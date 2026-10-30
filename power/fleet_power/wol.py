"""Wake-on-LAN, unprivileged: a UDP subnet-directed broadcast on the box's own direct cable.

Each box's wired NIC with Wake-on armed is the /30 cable to this host, so the magic packet goes to
that cable's broadcast address (the inventory's `wol.broadcast`). No raw socket and no
capability are needed. The packet is sent only if the kernel's route to that broadcast address
really is the cable: with the link down the route disappears, the packet would leave by the
Wi-Fi default route and wake nothing, and the operator would be told "sent". Refusing is honest.
"""
from __future__ import annotations

import ipaddress
import re
import socket

_MAC = re.compile(r"^([0-9a-fA-F]{2})(:[0-9a-fA-F]{2}){5}$")


class WolError(Exception):
    pass


def magic_packet(mac: str) -> bytes:
    if not isinstance(mac, str) or not _MAC.match(mac):
        raise WolError(f"malformed MAC {mac!r}")
    return b"\xff" * 6 + bytes.fromhex(mac.replace(":", "")) * 16


def _le_ip(hexstr: str) -> int:
    """/proc/net/route prints each address as the host-order u32 of its network-order bytes;
    on the little-endian hosts here that equals the little-endian read of the address bytes."""
    return int(hexstr, 16)


def route_iface(dest: str, table: str) -> str | None:
    """The interface the kernel's longest-prefix match in /proc/net/route picks for `dest`."""
    d = int.from_bytes(ipaddress.IPv4Address(dest).packed, "little")
    best: tuple[int, str] | None = None
    for line in table.splitlines()[1:]:
        f = line.split()
        if len(f) < 8:
            continue
        iface, net, flags, mask = f[0], _le_ip(f[1]), int(f[3], 16), _le_ip(f[7])
        if not flags & 0x1 or (d & mask) != net:
            continue
        plen = bin(mask).count("1")
        if best is None or plen > best[0]:
            best = (plen, iface)
    return best[1] if best else None


def _udp_broadcast_socket():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    return s


def send_wol(box, *, route_table: str | None = None, sock_factory=_udp_broadcast_socket,
             send: bool = True) -> str:
    """send=False (DRY RUN): every check, no packet — a dry run must not wake a box that is off."""
    w = box.wol
    if w is None:
        raise WolError(f"{box.name} has no Wake-on-LAN path")
    pkt = magic_packet(w.mac)
    if route_table is None:
        with open("/proc/net/route") as f:
            route_table = f.read()
    via = route_iface(w.broadcast, route_table)
    if via != w.iface:
        raise WolError(f"no route to {w.broadcast} via {w.iface} (the cable to {box.name} looks "
                       f"down; the kernel would use {via or 'nothing'}). Not sent.")
    if not send:
        return f"DRY RUN: route ok ({w.broadcast} via {w.iface}); magic packet for {w.mac} NOT sent"
    s = sock_factory()
    try:
        for _ in range(3):
            s.sendto(pkt, (w.broadcast, 9))
    finally:
        s.close()
    return f"magic packet for {w.mac} sent 3x to {w.broadcast}:9 on {w.iface}"
