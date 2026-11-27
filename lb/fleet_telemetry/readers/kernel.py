"""KernelLog — the kernel's own counts of what went wrong on the box: GPU Xid errors and OOM kills (
inputs; the findings that judge them are the gateway's, not this reader's).

SOURCES, AND WHAT EACH NEEDS FROM THE CONTAINER:

  * OOM kills: /proc/vmstat `oom_kill` (kernel >= 4.13) — the kernel's count of OOM-killer victims since boot. vmstat
    is not namespaced, so the sender's container reads the HOST's count with no privilege at all. A kernel without the
    line → the count is absent with why.
  * Xid: the NVIDIA driver reports a GPU fault ONLY as a kernel log line ("NVRM: Xid (PCI:<bdf>): <code>, …"); no
    sysfs file or nvidia-smi field counts them. The kernel log is /dev/kmsg. Docker gives a container no /dev/kmsg,
    and with `kernel.dmesg_restrict=1` (the Ubuntu default) reading it needs CAP_SYSLOG. So the grant is OPT-IN:
    lb/hw_sender/kmsg.yml adds the device READ-ONLY (cgroup permission `r`: the sender can never write a line into the
    host's log) and CAP_SYSLOG. CAP_SYSLOG also lets a process clear the ring buffer via syslog(2); this reader only
    reads /dev/kmsg, never calls syslog(2). Without kmsg.yml the Xid figures are absent with that reason, and the OOM
    count is still read. Chosen over `privileged`, a host /dev bind or the journal (journald's files need the host's
    /var/log/journal and a reader library the stdlib-only image does not carry).

HOW XIDS ARE COUNTED. /dev/kmsg is opened once, non-blocking, and drained every sample: each read hands back one record
("<prio>,<seq>,<µs since boot>,<flags>;<message>", continuation lines indented). The count starts at the oldest record
the ring buffer still holds. When that is seq 0 and nothing was lost (EPIPE: records overwritten before they were read)
the count is the whole boot: `xid_complete`. Otherwise it is the count since `xid_since`, and says so — never presented
as the boot's total. A record's time is boot time + its µs: boot = now − /proc/uptime. The kernel stamps records with
its local clock, which does not run through a suspend; on a server that never suspends the two agree.
"""
from __future__ import annotations

import errno
import os
import re
import time
from typing import Callable, Protocol

from fleet_telemetry.model import KernelReading
from fleet_telemetry.readers.base import Files

KMSG = "/dev/kmsg"
XID = re.compile(r"NVRM: Xid \([^)]*\): (\d+)")
GRANT = "lb/hw_sender/kmsg.yml"


class KmsgLike(Protocol):
    def drain(self) -> tuple[list[str], int]:
        """Every chunk readable now, and how many reads reported lost records (EPIPE)."""


class Kmsg:
    """/dev/kmsg, opened once non-blocking. Each os.read returns one record; EAGAIN ends a drain."""

    def __init__(self, path: str = KMSG) -> None:
        self.fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)

    def drain(self) -> tuple[list[str], int]:
        out, lost = [], 0
        while True:
            try:
                chunk = os.read(self.fd, 8192)
            except BlockingIOError:
                return out, lost
            except OSError as exc:
                if exc.errno == errno.EPIPE:
                    lost += 1
                    continue
                raise
            if not chunk:
                return out, lost
            out.append(chunk.decode("utf-8", "replace"))


def parse_kmsg(chunks: list[str]) -> list[tuple[int, int, str]]:
    """(seq, µs since boot, message) per record. A continuation line (it starts with a space) is not a record."""
    out = []
    for chunk in chunks:
        for line in chunk.splitlines():
            if not line or line.startswith(" ") or ";" not in line:
                continue
            head, _, msg = line.partition(";")
            fields = head.split(",")
            try:
                out.append((int(fields[1]), int(fields[2]), msg))
            except (IndexError, ValueError):
                continue
    return out


class KernelLog:
    source = "/proc/vmstat, /dev/kmsg"

    def __init__(self, root: str = "/", *, kmsg: Callable[[], KmsgLike] = Kmsg,
                 clock: Callable[[], float] = time.time) -> None:
        self.fs = Files(root)
        self._open, self.clock = kmsg, clock
        self._kmsg: KmsgLike | None = None
        self._kmsg_why: str | None = None
        self.xids = 0
        self.last_xid: dict[str, float] | None = None
        self.first_seq: int | None = None
        self.first_us: int | None = None
        self.lost = 0

    def _boot(self) -> float | None:
        up = Files.text(self.fs.path("/proc/uptime"))
        try:
            return self.clock() - float(up.split()[0]) if up else None
        except (ValueError, IndexError):
            return None

    def _drain(self) -> str | None:
        """Read what the log holds now into the counts; the reason the Xid figures are absent, or None."""
        if self._kmsg is None and self._kmsg_why is None:
            try:
                self._kmsg = self._open()
            except FileNotFoundError:
                self._kmsg_why = f"no {KMSG} in the sender's container: COMPOSE_FILE does not add {GRANT}"
            except PermissionError as exc:
                self._kmsg_why = f"{KMSG} refused ({exc.strerror}): reading it needs CAP_SYSLOG ({GRANT})"
            except OSError as exc:
                self._kmsg_why = f"{KMSG} unreadable: {exc}"
        if self._kmsg is None:
            return self._kmsg_why
        try:
            chunks, lost = self._kmsg.drain()
        except OSError as exc:
            return f"{KMSG} read failed: {exc}"
        self.lost += lost
        boot = self._boot()
        for seq, us, msg in parse_kmsg(chunks):
            if self.first_seq is None:
                self.first_seq, self.first_us = seq, us
            m = XID.search(msg)
            if m:
                self.xids += 1
                self.last_xid = {"code": int(m.group(1)), "at": boot + us / 1e6 if boot is not None else None}
        return None

    def read(self) -> KernelReading:
        whys = []
        oom = None
        vmstat = Files.text(self.fs.path("/proc/vmstat"))
        if vmstat is None:
            whys.append("/proc/vmstat is unreadable")
        else:
            line = next((ln for ln in vmstat.splitlines() if ln.startswith("oom_kill ")), None)
            if line is None:
                whys.append("this kernel's /proc/vmstat has no oom_kill line (it needs >= 4.13)")
            else:
                oom = int(line.split()[1])
        xid_why = self._drain()
        if xid_why:
            whys.append(xid_why)
            return KernelReading(xid_total=None, oom_kills_total=oom, last_xid=None, xid_since=None,
                                 xid_complete=None, why="; ".join(whys))
        boot = self._boot()
        since = boot + self.first_us / 1e6 if boot is not None and self.first_us is not None else None
        return KernelReading(xid_total=self.xids, oom_kills_total=oom, last_xid=self.last_xid, xid_since=since,
                             xid_complete=self.first_seq == 0 and self.lost == 0, why="; ".join(whys) or None)
