"""Cgroups — every running container's name, image and RAM, from two READ-ONLY mounts and never the docker socket
(even mounted `:ro`, the socket is the docker API: control of the host).

  * the host's cgroup v2 tree: one directory per running container, named by its 64-hex id (`docker-<id>.scope`
    under the systemd driver, `<id>` under cgroupfs). RAM = `memory.current` − `inactive_file` (memory.stat): the
    figure `docker stats` shows — page cache the kernel can drop is not the container's;
  * docker's per-container state (<DockerRootDir>/containers/<id>/config.v2.json): `Name` and `Config.Image`, re-read
    only when the file changes. That file also holds the container's environment; nothing of it but the name and the
    image leaves this reader.

A container whose config cannot be read (removed between the two reads) is not listed; one whose RAM cannot be read is
listed with None, which withholds the box total (model.ContainerSet).
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from fleet_telemetry.model import ContainerSet, ContainerUsage
from fleet_telemetry.readers.base import Files

SCOPE = re.compile(r"^(?:docker-)?([0-9a-f]{64})(?:\.scope)?$")
#: How deep under the root a container's directory may sit (system.slice/docker-<id>.scope is 2). The walk never
#: enters a container's own subtree.
DEPTH = 4
MAX_NAME, MAX_IMAGE = 64, 96
MIB = 1024 * 1024


class Cgroups:
    source = "cgroup v2"

    def __init__(self, cgroup_root: str | Path, containers_dir: str | Path) -> None:
        self.cgroup_root, self.containers_dir = str(cgroup_root), str(containers_dir)
        if not os.path.isfile(os.path.join(self.cgroup_root, "cgroup.controllers")):
            raise ValueError(f"{self.cgroup_root} is not a cgroup v2 root: lb/hw_sender/docker-compose.yml mounts the "
                             f"host's /sys/fs/cgroup there read-only (the containers' RAM)")
        if not os.path.isdir(self.containers_dir):
            raise ValueError(f"{self.containers_dir} is missing: lb/hw_sender/docker-compose.yml mounts "
                             f"${{HW_SENDER_DOCKER_ROOT}}/containers there read-only (the containers' names)")
        self._names: dict[str, tuple[float, str | None, str | None]] = {}

    def _scopes(self) -> dict[str, str]:
        found: dict[str, str] = {}
        base = self.cgroup_root.rstrip("/").count("/")
        for dirpath, dirnames, _files in os.walk(self.cgroup_root, onerror=self._raise):
            keep = []
            for d in dirnames:
                m = SCOPE.match(d)
                if m:
                    found[m.group(1)] = os.path.join(dirpath, d)
                elif dirpath.count("/") - base < DEPTH:
                    keep.append(d)
            dirnames[:] = keep
        return found

    @staticmethod
    def _raise(err: OSError) -> None:
        raise err

    @staticmethod
    def _ram_mb(scope: str) -> int | None:
        current = Files.int(os.path.join(scope, "memory.current"))
        stat = Files.text(os.path.join(scope, "memory.stat")) or ""
        inactive = next((int(ln.split()[1]) for ln in stat.splitlines() if ln.startswith("inactive_file ")), None)
        if current is None or inactive is None:
            return None
        return max(0, current - inactive) // MIB

    def _name(self, cid: str) -> tuple[str | None, str | None]:
        path = os.path.join(self.containers_dir, cid, "config.v2.json")
        try:
            mtime = os.stat(path).st_mtime
        except OSError:
            return None, None
        hit = self._names.get(cid)
        if hit and hit[0] == mtime:
            return hit[1], hit[2]
        try:
            with open(path) as f:
                cfg = json.load(f)
        except (OSError, ValueError):
            return None, None
        name = (cfg.get("Name") or "").lstrip("/")[:MAX_NAME] or None
        image = ((cfg.get("Config") or {}).get("Image") or "")[:MAX_IMAGE] or None
        self._names[cid] = (mtime, name, image)
        return name, image

    def read(self) -> ContainerSet:
        scopes = self._scopes()
        rows = []
        for cid, scope in scopes.items():
            name, image = self._name(cid)
            if name is not None:
                rows.append(ContainerUsage(name=name, image=image, ram_mb=self._ram_mb(scope)))
        for gone in set(self._names) - set(scopes):
            del self._names[gone]
        return ContainerSet(rows)
