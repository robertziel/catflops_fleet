"""THE IMAGE MATRIX — which container image runs an engine's BUILD on an ACCELERATOR for a CPU PLATFORM
(library/images/<engine>.yml, one file per engine).

Until 2026-10-30 every catalog row pinned its own `launch.<mode>.image`: the image a model ran was glued to the one
machine it was first started on, and a second kind of card had no way to say "the same model, its image for MY
kernels". The engine redesign (docs/design/engines.md) makes the image a function of three things the deployment
already states, and nothing else:

  engine × build      the recipe's (a model's engine settings name its engine and which BUILD of it — a build is one
                      image recipe: its base and the patches it carries, e.g. vLLM `fleet`, SGLang `fleet.4`)
  accelerator         the cluster's cards' (hardware/accelerators/<key>.yml, through each card type)
  platform            the cluster's boxes' (the fleet file's `platform`)

`image_for` answers with the ONE entry for that combination, or raises `ImageMissing` naming the combination and the
ones the matrix does have for that build. Never a nearest match: an sm_121a image handed to an sm_120 card fails at
its first kernel, far from here (accelerator.py says why the two are separate keys).

The file shape is ONE ENTRY PER LINE, keys in a fixed order (build, accelerator, platform, image, source): the deploy
build scripts read a tag from these lines with grep on a host that has no YAML parser (deploy/images/build.sh),
and tests/test_images.py holds every entry to that shape so the two readers can never disagree.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from vllm_lb import yamlfile
from vllm_lb.hardware import PLATFORMS, accelerators, read_keys

#: The image matrix this repo ships: one YAML file per engine, named by the catalog's engine word.
IMAGES_DIR = Path(__file__).resolve().parent / "library" / "images"
#: An entry's keys, in the one order a line states them (the deploy scripts' grep relies on it).
ENTRY_KEYS = ("build", "accelerator", "platform", "image", "source")


class ImageMissing(ValueError):
    """No image for an engine build on an accelerator and platform: the refusal names the missing combination."""


@dataclass(frozen=True)
class ImageEntry:
    """One cell of the matrix: the image a build of `engine` runs on `accelerator` for `platform`, and its source."""

    engine: str
    build: str
    accelerator: str
    platform: str
    image: str
    source: str

    def view(self) -> dict:
        return {"engine": self.engine, "build": self.build, "accelerator": self.accelerator,
                "platform": self.platform, "image": self.image, "source": self.source}


@dataclass(frozen=True)
class EngineImages:
    """One engine's matrix: its builds (name → what the build is) and every entry."""

    engine: str
    builds: Mapping[str, str]
    entries: tuple[ImageEntry, ...]


def _text(where: str, v) -> str:
    if not isinstance(v, str) or not v.strip() or v != v.strip():
        raise ValueError(f"{where} must be a non-empty string without surrounding space, got {v!r}")
    return v


def _engine(engine: str, raw, source: str, accs) -> EngineImages:
    read_keys(f"{source}: the matrix", raw, ("builds", "images"))
    builds_raw = raw["builds"]
    if not isinstance(builds_raw, dict) or not builds_raw:
        raise ValueError(f"{source}: builds must name at least one build of {engine}")
    builds = {}
    for name, b in builds_raw.items():
        read_keys(f"{source}: builds.{name}", b, ("source",))
        builds[_text(f"{source}: a build's name", name)] = _text(f"{source}: builds.{name}.source", b["source"])
    if not isinstance(raw["images"], list) or not raw["images"]:
        raise ValueError(f"{source}: images must list at least one entry")
    entries, seen = [], set()
    for i, e in enumerate(raw["images"]):
        w = f"{source}: images[{i}]"
        read_keys(w, e, ENTRY_KEYS)
        entry = ImageEntry(engine, *(_text(f"{w}.{k}", e[k]) for k in ENTRY_KEYS))
        if entry.build not in builds:
            raise ValueError(f"{w}: build {entry.build!r} is not one of this file's builds: {', '.join(builds)}")
        if entry.accelerator not in accs:
            raise ValueError(f"{w}: no accelerator {entry.accelerator!r}: the accelerators are "
                             f"{', '.join(sorted(accs))} (vllm_lb/hardware/accelerators/<key>.yml)")
        if entry.platform not in PLATFORMS:
            raise ValueError(f"{w}: platform must be one of {', '.join(PLATFORMS)}, got {entry.platform!r}")
        platforms = accs[entry.accelerator].platforms
        if platforms.known and entry.platform not in platforms.value:
            raise ValueError(f"{w}: {entry.accelerator} never comes in a {entry.platform} host: it lists "
                             f"{', '.join(platforms.value)} ({platforms.source})")
        cell = (entry.build, entry.accelerator, entry.platform)
        if cell in seen:
            raise ValueError(f"{w}: {' × '.join(cell)} is stated twice — one image per combination")
        seen.add(cell)
        entries.append(entry)
    return EngineImages(engine, MappingProxyType(builds), tuple(entries))


def load_images(directory: str | Path, accelerator_library=None) -> Mapping[str, EngineImages]:
    """Every engine's matrix in `directory` (`<engine>.yml`), validated against the accelerator library — or
    ValueError naming the file and the value."""
    files = sorted(Path(directory).glob("*.yml"))
    if not files:
        raise ValueError(f"{directory}: no image matrix (<engine>.yml) there")
    accs = accelerators() if accelerator_library is None else accelerator_library
    return MappingProxyType({f.stem: _engine(f.stem, yamlfile.read(f), f.name, accs) for f in files})


@functools.cache
def images() -> Mapping[str, EngineImages]:
    """The matrix this repo ships (IMAGES_DIR), read once."""
    return load_images(IMAGES_DIR)


def image_for(engine: str, build: str, accelerator: str, platform: str,
              library: Mapping[str, EngineImages] | None = None) -> ImageEntry:
    """The image for `engine`'s `build` on `accelerator` for `platform` — or ImageMissing naming the missing
    combination and what the matrix holds for that build."""
    lib = images() if library is None else library
    m = lib.get(engine)
    if m is None:
        raise ImageMissing(f"no image matrix for engine {engine!r} (vllm_lb/library/images/<engine>.yml); the "
                           f"matrices are {', '.join(sorted(lib))}")
    if build not in m.builds:
        raise ImageMissing(f"{engine} has no build {build!r}: its builds are {', '.join(m.builds)} "
                           f"(vllm_lb/library/images/{engine}.yml)")
    for e in m.entries:
        if (e.build, e.accelerator, e.platform) == (build, accelerator, platform):
            return e
    have = [f"{e.accelerator} × {e.platform}" for e in m.entries if e.build == build]
    raise ImageMissing(f"no image of {engine} build {build} for {accelerator} × {platform}: it is built for "
                       f"{', '.join(have) or 'nothing'} (vllm_lb/library/images/{engine}.yml) — build it for this "
                       f"accelerator and add the entry")
