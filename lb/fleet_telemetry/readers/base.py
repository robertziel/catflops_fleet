"""The reader interface, and the file access every procfs / sysfs reader shares.

`root` prefixes every absolute path a reader opens (`/sys/...`, `/proc/...`): "/" in the sender, a planted tree of
captured files in the tests. One seam, so no reader reaches the real host in a test.
"""
from __future__ import annotations

import glob
import os
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
