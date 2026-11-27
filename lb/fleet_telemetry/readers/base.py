"""The reader interface, and the file access every procfs / sysfs reader shares.

`root` prefixes every absolute path a reader opens (`/sys/...`, `/proc/...`): "/" in the sender, a planted tree of
captured files in the tests. One seam, so no reader reaches the real host in a test.
"""
from __future__ import annotations

import glob
import os
import re
from pathlib import Path
from typing import Protocol, TypeVar

T = TypeVar("T", covariant=True)


class Reader(Protocol[T]):
    source: str

    def read(self) -> T: ...


class Files:
    """Reads under `root`. A file that is missing, unreadable or not the type asked for is None — the caller decides
    what that absence means."""

    def __init__(self, root: str | Path = "/") -> None:
        self.root = str(root)

    def path(self, p: str) -> str:
        return os.path.join(self.root, p.lstrip("/"))

    def glob(self, pattern: str) -> list[str]:
        return sorted(glob.glob(self.path(pattern)))

    @staticmethod
    def text(path: str) -> str | None:
        try:
            with open(path) as f:
                return f.read().strip()
        except (OSError, UnicodeDecodeError):
            return None

    @classmethod
    def int(cls, path: str) -> int | None:
        v = cls.text(path)
        try:
            return int(v) if v is not None else None
        except ValueError:
            return None


class Binds:
    """The host paths bind-mounted READ-ONLY into the sender, one bind per configured path (disks and models
    roots): the host's `<path>` is at `<prefix><path>` (HOST_FS, fleet_telemetry/__main__.py), mounted by
    lb/hw_sender/disks.yml (rendered by deploy/setup/render.py, or written by hand from disks.example.yml).

    A configured path is read ONLY when `<prefix><path>` is a mount point in /proc/self/mountinfo. Without its bind the
    directory may still exist in the container (an image path, a parent of another bind): statvfs there would report
    the CONTAINER's disk under the host path's name, so an unbound path is absent with the bind it needs."""

    def __init__(self, prefix: str, mountinfo: str = "/proc/self/mountinfo") -> None:
        self.prefix, self.mountinfo = prefix, mountinfo

    def target(self, path: str) -> str:
        return os.path.normpath(self.prefix + "/" + path.lstrip("/"))

    def _mounts(self) -> set[str] | None:
        text = Files.text(self.mountinfo)
        if text is None:
            return None
        out = set()
        for line in text.splitlines():
            parts = line.split()
            if len(parts) > 4:
                out.add(re.sub(r"\\([0-7]{3})", lambda m: chr(int(m.group(1), 8)), parts[4]))
        return out

    def missing(self, path: str) -> str | None:
        """None when `path` is bound; otherwise why it is not read."""
        mounts = self._mounts()
        if mounts is None:
            return f"{self.mountinfo} is unreadable: no bind can be confirmed"
        if self.target(path) in mounts:
            return None
        return (f"{path} is not bind-mounted into the sender (no mount at {self.target(path)}): add "
                f"`- {path}:{self.target(path)}:ro` to lb/hw_sender/disks.yml")
