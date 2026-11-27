"""ModelDirs — how much disk each top-level folder of the box's MODELS ROOTS takes (checkpoint
folder sizes).

WHERE IT IS MEASURED, AND WHY HERE. A checkpoint's folder sits on the box that serves it (the fleet file's
`checkpoints.<id>.path` is relative to the models roots ON THE BOX: lb/.env MODELS_SEARCH_DIRS). The gateway cannot see
those disks, so the box measures its own, and the gateway names each folder by the checkpoints the fleet file puts in
it (vllm_lb/hw.py). The roots are HW_SENDER_MODELS_DIRS (lb/hw_sender/.env), explicit host paths, each read through
its own read-only bind (lb/hw_sender/disks.yml, base.Binds) — no discovery; a root with no bind is absent with the bind
it needs.

WHY TOP-LEVEL FOLDERS. The sender does not read the fleet file (it runs on every box; the fleet file is the gateway's),
so it cannot know which nested folder is a checkpoint. The first path segment of every checkpoint is a top-level
folder of a root, so each top-level folder's size is exact for a one-segment checkpoint and is the folder that HOLDS a
deeper one (`tensorfold-hf/hub/models--…`): the gateway labels it by what it holds, never as one checkpoint's size.

HOW. `du -s` semantics: allocated blocks (st_blocks × 512 — a sparse file is what it occupies), a hard link counted
once, a symlink NOT followed (a HF snapshot's links point at its own blobs/, counted where they live). A folder with an
entry the walk could not read is `unreadable` and gets no size — a partial sum would be a number that is not the
folder's.

SLOW CADENCE. A walk over a few TB of checkpoints is thousands of stats and can take seconds on an HDD; the sample is
every second. So the walk runs on its own daemon thread every WALK_EVERY_S, and every sample carries the LAST walk's
result with `measured_at` and `took_s` — the panel shows its age. Before the first walk ends, each root says so.
The SENT largest folders are named per root; the rest are counted (`more_count`, `more_bytes`) so the datagram stays
small however many folders a root has.
"""
from __future__ import annotations

import os
import stat
import threading
import time
from typing import Callable

from fleet_telemetry.model import ModelsRoot
from fleet_telemetry.readers.base import Binds

#: Ten minutes between walks: a checkpoint is staged in minutes to hours, and a walk must not keep an HDD busy.
WALK_EVERY_S = 600.0


def _du(top: str) -> int | None:
    """On-disk bytes under `top` (du -s), or None when any part of it could not be read."""
    seen: set[tuple[int, int]] = set()
    total = 0
    failed = []
    for dirpath, _dirs, files in os.walk(top, onerror=failed.append, followlinks=False):
        for name in files:
            try:
                st = os.lstat(os.path.join(dirpath, name))
            except OSError as exc:
                failed.append(exc)
                continue
            if not stat.S_ISREG(st.st_mode):        # a symlink (not followed), a socket, a device: no blocks of its own
                continue
            if st.st_nlink > 1:
                key = (st.st_dev, st.st_ino)
                if key in seen:
                    continue
                seen.add(key)
            total += st.st_blocks * 512
    return None if failed else total


class ModelDirs:
    source = "models roots walk"
    #: Folders named per root, largest first (the rest are counted). 16: the reference fleet stages its checkpoints in
    #: 14 top-level folders, and the health blocks stay inside their 4 KB of the datagram beside 4 RoCE ports, each
    #: now carrying its ethtool PHY counters and dcb QoS (test_telemetry_box_health).
    SENT = 16

    def __init__(self, roots: list[str], binds: Binds, *, clock: Callable[[], float] = time.time,
                 every_s: float = WALK_EVERY_S, start: bool = True) -> None:
        self.roots, self.binds, self.clock, self.every_s = list(roots), binds, clock, every_s
        self._lock = threading.Lock()
        self._last: list[ModelsRoot] | None = None
        self._started_at: float | None = None
        self._thread: threading.Thread | None = None
        self._start = start

    def walk(self) -> list[ModelsRoot]:
        """One walk of every root, now (the thread's body; the tests call it directly)."""
        out = []
        for r in self.roots:
            began = self.clock()
            unbound = self.binds.missing(r)
            if unbound:
                out.append(ModelsRoot(r, {}, None, None, why=unbound))
                continue
            top = self.binds.target(r)
            try:
                entries = [e for e in os.scandir(top) if e.is_dir(follow_symlinks=False)]
            except OSError as exc:
                out.append(ModelsRoot(r, {}, None, None, why=f"{r}: {exc.strerror or exc}"))
                continue
            sizes, unreadable = {}, []
            for e in entries:
                size = _du(e.path)
                if size is None:
                    unreadable.append(e.name)
                else:
                    sizes[e.name] = size
            ranked = sorted(sizes.items(), key=lambda kv: -kv[1])
            rest = ranked[self.SENT:]
            ended = self.clock()
            out.append(ModelsRoot(r, dict(ranked[:self.SENT]), measured_at=ended, took_s=round(ended - began, 1),
                                  more_count=len(rest), more_bytes=sum(v for _k, v in rest),
                                  unreadable=tuple(sorted(unreadable))))
        return out

    def _loop(self) -> None:
        while True:
            result = self.walk()
            with self._lock:
                self._last = result
            time.sleep(self.every_s)

    def read(self) -> list[ModelsRoot]:
        with self._lock:
            if self._started_at is None:
                self._started_at = self.clock()
                if self._start:
                    self._thread = threading.Thread(target=self._loop, name="models-walk", daemon=True)
                    self._thread.start()
            if self._last is not None:
                return list(self._last)
            ago = round(self.clock() - self._started_at)
        return [ModelsRoot(r, {}, None, None, why=f"the first walk of {r} is running ({ago} s so far)")
                for r in self.roots]
