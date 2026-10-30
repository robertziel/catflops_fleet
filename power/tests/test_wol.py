"""Wake-on-LAN: the packet, and the refusal to send it anywhere but the box's own cable."""
from __future__ import annotations

import pytest

from tests.fakes import suite_boxes
from fleet_power.wol import WolError, magic_packet, route_iface, send_wol

# /proc/net/route as the host has it (hex, little-endian): the two direct cables + the default route.
# Documentation addresses (RFC 5737): 192.0.2.0/30 is box-a's cable, 198.51.100.0/30 is box-b's,
# 203.0.113.0/24 is the default (Wi-Fi) network with gateway 203.0.113.1.
ROUTE = """Iface\tDestination\tGateway \tFlags\tRefCnt\tUse\tMetric\tMask\t\tMTU\tWindow\tIRTT
wlan-up\t00000000\t017100CB\t0003\t0\t0\t600\t00000000\t0\t0\t0
eth-box-a\t000200C0\t00000000\t0001\t0\t0\t100\tFCFFFFFF\t0\t0\t0
eth-box-b\t006433C6\t00000000\t0001\t0\t0\t101\tFCFFFFFF\t0\t0\t0
wlan-up\t007100CB\t00000000\t0001\t0\t0\t600\t00FFFFFF\t0\t0\t0
"""
BOX_B_MAC = "02:00:00:00:00:03"


def test_magic_packet_is_six_ff_then_the_mac_sixteen_times():
    pkt = magic_packet(BOX_B_MAC)
    assert len(pkt) == 102
    assert pkt[:6] == b"\xff" * 6
    assert pkt[6:] == bytes.fromhex("020000000003") * 16


@pytest.mark.parametrize("bad", ["", "b0:82:e2:50:c9", "zz:82:e2:50:c9:6c", "b0-82-e2-50-c9-6c-00"])
def test_a_malformed_mac_is_refused(bad):
    with pytest.raises(WolError):
        magic_packet(bad)


def test_route_iface_finds_the_direct_link():
    assert route_iface("198.51.100.3", ROUTE) == "eth-box-b"
    assert route_iface("192.0.2.3", ROUTE) == "eth-box-a"
    assert route_iface("8.8.8.8", ROUTE) == "wlan-up"     # the default route


class FakeSock:
    def __init__(self):
        self.sent = []
        self.opts = []

    def setsockopt(self, *a):
        self.opts.append(a)

    def sendto(self, data, addr):
        self.sent.append((data, addr))

    def close(self):
        pass


def test_send_wol_broadcasts_on_the_boxs_own_link():
    box_b = suite_boxes()["box-b"]
    sock = FakeSock()
    msg = send_wol(box_b, route_table=ROUTE, sock_factory=lambda: sock)
    assert {addr for _, addr in sock.sent} == {("198.51.100.3", 9)}
    assert len(sock.sent) == 3
    assert all(data == magic_packet(BOX_B_MAC) for data, _ in sock.sent)
    assert "eth-box-b" in msg


def test_send_wol_refuses_when_the_cable_route_is_gone():
    """With the link down the broadcast would leave by the Wi-Fi default route: refuse instead."""
    box_b = suite_boxes()["box-b"]
    table = "\n".join(l for l in ROUTE.splitlines() if not l.startswith("eth-box-b"))
    sock = FakeSock()
    with pytest.raises(WolError, match="eth-box-b"):
        send_wol(box_b, route_table=table, sock_factory=lambda: sock)
    assert sock.sent == []


def test_send_false_checks_the_route_and_sends_nothing():
    box_b = suite_boxes()["box-b"]
    sock = FakeSock()
    msg = send_wol(box_b, route_table=ROUTE, sock_factory=lambda: sock, send=False)
    assert sock.sent == [] and "DRY RUN" in msg and "eth-box-b" in msg
