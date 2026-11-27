"""RdmaPorts — the box's RDMA (RoCE / InfiniBand) PORTS, one row each, from /sys/class/infiniband.

WHY PER PORT. tp-rank's `observe_net` (lb/cluster/tp-rank) sums the bytes over every RDMA device, because NCCL spreads
an all-reduce over all of them; that answers "how fast is the TP link", not "is it healthy". A port that dropped to
DOWN, renegotiated to a lower rate, or is running out of receive buffers (`out_of_buffer`) and timing out on ACKs
(`local_ack_timeout_err`) slows every decode step of a two-box cluster while the byte rate still looks plausible. So
each port is read on its own: its state, physical state, rate, link layer, the netdev it maps to and that netdev's MTU,
and the error counters below.

CUMULATIVE, NOT RATES. Every counter is the device's raw monotonic value; the gateway (vllm_lb/hw.py) derives the
per-second rates from two samples and treats a counter that went backwards as a reset (no rate that sample). The
sender keeps no state between samples.

ABSENT, NEVER ZERO. (`why["ethtool"]` is the reason for EVERY PHY counter when the tool itself refused; `why["qos"]`
for the three QoS figures when the port has no netdev to ask.) A driver that has no `hw_counters/` (not every RDMA driver is mlx5) states none of those: each is
absent from `counters` with its reason in the port's `why`. A netdev in another network namespace (a bridged sender)
has no /sys/class/net entry here: the MTU is absent with that reason. No /sys/class/infiniband at all is a failure
(OSError): the Sampler turns it into `rdma.why` — a box without RDMA has no ports, and says so.

BEYOND SYSFS — two tools, run against the port's netdev (the sender has host networking, so the netdev is the box's):
  * `ethtool -S <netdev>` for mlx5e's PHY counters (PHY_COUNTERS: `rx_discards_phy` — the NIC dropping frames for want
    of a receive buffer, the classic lossless-fabric misconfiguration — its tx twin, CRC and symbol errors, pause
    frames). ONE call per netdev per sample, shared by every port on it (a few ms). The counters join `counters`, so
    the gateway derives their rates like the rest.
  * iproute2's `dcb` for the QoS setting: PFC's enabled priorities (`dcb pfc show dev <nd> prio-pfc`), the trust order
    (`dcb apptrust show dev <nd> order`) and how many DSCP → priority entries the app table holds (`dcb -N app show
    dev <nd> dscp-prio`). A setting, not a counter: read every QOS_EVERY_S, and every sample carries the last reading
    with its time (`qos_at`). Read-only netlink GETs: no capability is needed (only SETs need CAP_NET_ADMIN).
  `mlnx_qos` (MLNX_OFED) is not used: it is not in a stock image, and iproute2's dcb reads the same kernel DCB state.
  A tool missing from the image, a netdev not in this namespace (a bridged sender), a driver without the op
  (mlx5 implements no apptrust: "Operation not supported"), or any refusal leaves THAT figure absent with the tool's
  own words as its why — never 0, never a guess (a non-empty DSCP map is not reported as "trust dscp").
The `rate` and `state` files' wording is the kernel's (Documentation/ABI/stable/sysfs-class-infiniband).
"""
from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import dataclass
from typing import Callable

from fleet_telemetry.model import RdmaPort
from fleet_telemetry.readers.base import Files

#: (directory under the port, counter): the error counters read. `hw_counters/` is the driver's own set (mlx5: the
#: RoCE congestion and retransmit signals); `counters/` is the IB PMA set every RDMA driver exposes.
COUNTERS: tuple[tuple[str, str], ...] = (
    ("hw_counters", "out_of_buffer"),
    ("hw_counters", "local_ack_timeout_err"),
    ("hw_counters", "rnr_nak_retry_err"),
    ("hw_counters", "packet_seq_err"),
    ("hw_counters", "implied_nak_seq_err"),
    ("counters", "port_rcv_errors"),
    ("counters", "port_xmit_discards"),
    ("counters", "port_rcv_remote_physical_errors"),
    ("counters", "symbol_error"),
    ("counters", "link_downed"),
)

#: mlx5e's PHY-layer counters, from `ethtool -S <netdev>` (mlx5 counters.rst): drops for want of a buffer, frame
#: errors on the wire, and pause frames (PFC / global pause at work).
PHY_COUNTERS: tuple[str, ...] = ("rx_discards_phy", "tx_discards_phy", "rx_crc_errors_phy", "rx_symbol_err_phy",
                                 "rx_pause_ctrl_phy", "tx_pause_ctrl_phy")

#: The `why` key for every PHY counter at once, when ethtool itself refused (one reason, not six copies of it).
PHY = "ethtool"

_STAT = re.compile(r"^\s*([A-Za-z0-9_\[\].:-]+):\s*(\d+)\s*$")


def _run(argv: list[str]) -> str:
    """The tool's stdout; RuntimeError with its own words when it refuses; FileNotFoundError when it is not installed."""
    done = subprocess.run(argv, capture_output=True, text=True, timeout=5)
    if done.returncode != 0:
        raise RuntimeError(f"exit {done.returncode}: {(done.stderr or done.stdout).strip()[:160]}")
    return done.stdout


def _refusal(argv: list[str], exc: Exception) -> str:
    """`<tool> <object>: <the tool's words>` — the command is the netdev's, named by the port row it sits in."""
    if isinstance(exc, FileNotFoundError):
        return f"{argv[0]} is not in the sender image (lb/hw_sender/Dockerfile installs it)"
    return f"{argv[0]} {argv[1] if argv[1] != '-N' else argv[2]}: {exc}"[:200]


@dataclass(frozen=True)
class _Qos:
    at: float
    pfc: list[int] | None
    trust: list[str] | None
    dscp_map: int | None
    why: dict[str, str]


_RATE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*Gb/sec")


def _after_colon(text: str | None) -> str | None:
    """`4: ACTIVE` → `ACTIVE`; the kernel prefixes the enum's number."""
    if text is None:
        return None
    return text.split(":", 1)[1].strip() if ":" in text else text.strip()


class RdmaPorts:
    source = "/sys/class/infiniband"
    #: A QoS setting changes when an operator changes it; one dcb round a minute per netdev.
    QOS_EVERY_S = 60.0

    def __init__(self, root: str = "/", *, run: Callable[[list[str]], str] = _run,
                 clock: Callable[[], float] = time.time) -> None:
        self.fs = Files(root)
        self.run, self.clock = run, clock
        self._qos: dict[str, _Qos] = {}
        self._stats: dict[str, tuple[dict[str, int] | None, str | None]] = {}

    def _call(self, argv: list[str]) -> tuple[str | None, str | None]:
        try:
            return self.run(argv), None
        except Exception as exc:  # noqa: BLE001 — any refusal is that figure's why
            return None, _refusal(argv, exc)

    def _ethtool(self, netdev: str) -> tuple[dict[str, int] | None, str | None]:
        """`ethtool -S <netdev>` once per sample (cached for the read in progress)."""
        if netdev not in self._stats:
            out, why = self._call(["ethtool", "-S", netdev])
            stats = None
            if out is not None:
                stats = {m.group(1): int(m.group(2)) for ln in out.splitlines() if (m := _STAT.match(ln))}
            self._stats[netdev] = (stats, why)
        return self._stats[netdev]

    def _qos_of(self, netdev: str) -> _Qos:
        now = self.clock()
        last = self._qos.get(netdev)
        if last is not None and now - last.at < self.QOS_EVERY_S:
            return last
        why: dict[str, str] = {}
        pfc = trust = dscp = None
        out, err = self._call(["dcb", "pfc", "show", "dev", netdev, "prio-pfc"])
        pairs = re.findall(r"(\d+):(on|off)", out or "")
        if err:
            why["pfc"] = err
        elif not pairs:
            why["pfc"] = f"`dcb pfc show dev {netdev} prio-pfc` printed no prio:on/off pairs: {out.strip()[:80]!r}"
        else:
            pfc = [int(p) for p, state in pairs if state == "on"]
        out, err = self._call(["dcb", "apptrust", "show", "dev", netdev, "order"])
        if err:
            why["trust"] = err
        else:
            words = out.replace("order:", " ").split()
            if words:
                trust = words
            else:
                why["trust"] = f"`dcb apptrust show dev {netdev} order` printed no selector"
        out, err = self._call(["dcb", "-N", "app", "show", "dev", netdev, "dscp-prio"])
        if err:
            why["dscp_map"] = err
        else:
            dscp = len(re.findall(r"\b\d+:\d+\b", out))
        q = _Qos(now, pfc, trust, dscp, why)
        self._qos[netdev] = q
        return q

    def read(self) -> list[RdmaPort]:
        self._stats = {}
        base = self.fs.path("/sys/class/infiniband")
        if not os.path.isdir(base):
            raise OSError("no /sys/class/infiniband: this box has no RDMA device (or its driver is not loaded)")
        ports = []
        for dev in sorted(os.listdir(base)):
            pdir = os.path.join(base, dev, "ports")
            names = sorted(os.listdir(pdir), key=lambda n: int(n) if n.isdigit() else 0) if os.path.isdir(pdir) else []
            for n in names:
                if n.isdigit():
                    ports.append(self._port(dev, int(n), os.path.join(pdir, n)))
        return ports

    def _port(self, dev: str, port: int, d: str) -> RdmaPort:
        why: dict[str, str] = {}
        counters: dict[str, int] = {}
        for sub, name in COUNTERS:
            v = Files.int(os.path.join(d, sub, name))
            if v is not None:
                counters[name] = v
            elif not os.path.isdir(os.path.join(d, sub)):
                why[sub] = f"no {sub}/ on this port: the driver states none of it"   # once for the whole directory
            else:
                why[name] = f"no readable {sub}/{name} on this port"
        rate_text = Files.text(os.path.join(d, "rate"))
        m = _RATE.match(rate_text or "")
        if m is None:
            why["rate_gbps"] = f"rate unreadable ({rate_text!r})"
        netdev = Files.text(os.path.join(d, "gid_attrs", "ndevs", "0")) or None
        mtu = None
        here = netdev is not None and os.path.isdir(self.fs.path(f"/sys/class/net/{netdev}"))
        if netdev is None:
            why["mtu"] = "maps to no netdev (gid_attrs/ndevs/0): not RoCE, or GID 0 unset"
        elif not here:
            why["mtu"] = f"netdev {netdev} is not in this network namespace (no /sys/class/net/{netdev})"
        else:
            mtu = Files.int(self.fs.path(f"/sys/class/net/{netdev}/mtu"))
            if mtu is None:
                why["mtu"] = f"no readable /sys/class/net/{netdev}/mtu"
        qos = None
        if not here:
            # nothing to ask ethtool or dcb about: one reason for all of their figures, the MTU's
            why[PHY] = why["qos"] = "no netdev in this namespace (see mtu)"
        else:
            stats, err = self._ethtool(netdev)
            if err:
                why[PHY] = err          # once for every PHY counter: the tool refused them all
            for name in PHY_COUNTERS if not err else ():
                if name in stats:
                    counters[name] = stats[name]
                else:
                    why[name] = f"ethtool -S {netdev} states no {name} (not an mlx5e netdev?)"
            qos = self._qos_of(netdev)
            why.update(qos.why)
        return RdmaPort(dev=dev, port=port, state=_after_colon(Files.text(os.path.join(d, "state"))),
                        phys_state=_after_colon(Files.text(os.path.join(d, "phys_state"))),
                        rate_gbps=float(m.group(1)) if m else None,
                        link_layer=Files.text(os.path.join(d, "link_layer")), netdev=netdev, mtu=mtu,
                        counters=counters, why=why, pfc_prios=qos.pfc if qos else None,
                        trust=qos.trust if qos else None, dscp_map=qos.dscp_map if qos else None,
                        qos_at=qos.at if qos else None)
