"""Disks — how full each filesystem the sender's config NAMES is.

EXPLICIT PATHS, NO DISCOVERY. Which disks matter is the deployment's knowledge (the models disk, the root, a scratch
volume), not something a mount table says: /proc/mounts on a box lists overlays, tmpfs, snaps and bind mounts by the
dozen. So the sender reads EXACTLY HW_SENDER_DISKS (lb/hw_sender/.env), each a host path, and nothing else.

HOW IT SEES THE HOST — ONE READ-ONLY BIND PER PATH, NOTHING MORE. lb/hw_sender/disks.yml binds each configured host
path at `<HOST_FS><path>` (base.Binds; the wizard renders the file, disks.example.yml is the hand-written form): the
sender sees those directories and no other host file. A path with no bind is absent with the bind it needs — never the
container's own disk under the host path's name. statvfs on the bind reports the filesystem holding the host's
`<path>`. Counted as `df` counts it:
total = blocks × fragment size, used = (blocks − free) × fragment size, free = blocks available to an unprivileged
user (df's Avail: the root-reserved blocks are not free to anyone who writes models). A path statvfs refuses is that
path's `why`; the others still read. statvfs on a hung network mount blocks: a path named here is the operator's
choice, so a dead NFS mount there stalls this sender's samples, visibly (the box goes stale), never silently.
"""
from __future__ import annotations

import os

from fleet_telemetry.model import DiskUsage
from fleet_telemetry.readers.base import Binds


class Disks:
    source = "statvfs"

    def __init__(self, paths: list[str], binds: Binds) -> None:
        self.paths, self.binds = list(paths), binds

    def read(self) -> list[DiskUsage]:
        out = []
        for p in self.paths:
            why = self.binds.missing(p)
            if why:
                out.append(DiskUsage(p, None, None, None, why=why))
                continue
            try:
                st = os.statvfs(self.binds.target(p))
            except OSError as exc:
                out.append(DiskUsage(p, None, None, None, why=f"statvfs {p}: {exc.strerror or exc}"))
                continue
            out.append(DiskUsage(p, total_bytes=st.f_blocks * st.f_frsize,
                                 used_bytes=(st.f_blocks - st.f_bfree) * st.f_frsize,
                                 free_bytes=st.f_bavail * st.f_frsize))
        return out
