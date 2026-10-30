"""WHAT AN ENGINE'S KERNELS TARGET — an ACCELERATOR: data (accelerators/<key>.yml, one file per target), shared by
every card type that names it.

The engine redesign (docs/design/engines.md, operator-approved 2026-10-30) splits what was glued together in a card
type: a TYPE is one PRODUCT (its memory, cores, bandwidth, peaks — types/<key>.yml); its ACCELERATOR is what the
kernels an image carries are compiled for and can run. Two products of one accelerator share one file, and the
image matrix (vllm_lb/library/images/<engine>.yml) is keyed by accelerator, never by product: a new card of a known
architecture is one type file and no image, measurement or recipe changes.

  arch           what kernels are compiled for (sm_121, sm_120, sm_86, gfx1151)
  gpu_runtime    how a container attaches it: `nvidia` or `rocm` (facts.GPU_RUNTIMES) — which node compose override a
                 one-host node runs with (deploy/standalone/node-gpu.<value>.yml)
  platforms      the CPU platforms its hosts come in (facts.PLATFORMS) — an image for it is built for one of these;
                 a box's `platform` (the fleet file) must be one, and a type's `host_platform` too
  precisions     the element types its tensor units compute (bf16, fp16, fp8, nvfp4 …): a type's measured or spec
                 peak at a precision the accelerator cannot run is refused
  quant_formats  the checkpoint quantisation formats served on it (a model's `quant_method`, `none` for unquantised
                 weights): what a model's format is checked against before any image is asked for

WHY TWO BLACKWELL KEYS (sm_121 for the GB10, sm_120 for the RTX 5090) and not one: this repo's GB10 images are
compiled for the arch-specific target 12.1a (TORCH_CUDA_ARCH_LIST=12.1a, CUTE_DSL_ARCH=sm_121a in their Dockerfiles),
and arch-specific cubins run on that exact compute capability only (CUDA C++ Programming Guide, Feature
Availability). One key would let the resolver hand a 5090 an image that cannot load on it.

The KV cache dtypes an accelerator runs are NOT restated here: each image's record (vllm_lb/kv_cache_dtypes.json)
already says which dtypes it takes and on which arch (`arch` regex) — the resolver asks it with the accelerator's arch.

`compatibility` is the check a model's format and a box's platform get against an accelerator. Nothing outside the
hardware package branches on an accelerator: callers ask it.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from vllm_lb import yamlfile
from vllm_lb.hardware.facts import Fact, fact, keys

#: The accelerator library this repo ships: one YAML file per accelerator, named by its key.
ACCELERATORS_DIR = Path(__file__).resolve().parent / "accelerators"

_TOP = ("name", "facts")
_FACTS = {"arch": "str", "gpu_runtime": "runtime", "platforms": "platforms", "precisions": "words",
          "quant_formats": "words"}


@dataclass(frozen=True)
class Accelerator:
    """One kernel target: its facts, each with its source."""

    key: str
    name: str
    arch: Fact
    gpu_runtime: Fact
    platforms: Fact
    precisions: Fact
    quant_formats: Fact

    def view(self) -> dict:
        """As /lb/status carries it: every figure with its source, plain JSON."""
        return {"key": self.key, "name": self.name, **{k: getattr(self, k).view() for k in _FACTS}}


def _accelerator(key: str, raw, source: str) -> Accelerator:
    keys(f"{source}: the accelerator", raw, _TOP)
    name = raw["name"]
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{source}: name must be the accelerator's name")
    facts = keys(f"{source}: facts", raw["facts"], tuple(_FACTS))
    return Accelerator(key=key, name=name.strip(),
                       **{k: fact(f"{source}: facts.{k}", facts[k], kind) for k, kind in _FACTS.items()})


def load_accelerators(directory: str | Path) -> Mapping[str, Accelerator]:
    """Every accelerator in `directory` (`<key>.yml`), validated — or ValueError naming the file and the value."""
    files = sorted(Path(directory).glob("*.yml"))
    if not files:
        raise ValueError(f"{directory}: no accelerator (<key>.yml) there")
    return MappingProxyType({f.stem: _accelerator(f.stem, yamlfile.read(f), f.name) for f in files})


@functools.cache
def accelerators() -> Mapping[str, Accelerator]:
    """The library this repo ships (ACCELERATORS_DIR), read once."""
    return load_accelerators(ACCELERATORS_DIR)


def accelerator(key: str, library: Mapping[str, Accelerator] | None = None) -> Accelerator:
    """The accelerator `key` names — or ValueError naming every key there is. Never a nearest match."""
    lib = accelerators() if library is None else library
    found = lib.get(key) if isinstance(key, str) else None
    if found is None:
        raise ValueError(f"no accelerator {key!r}: the accelerators are {', '.join(sorted(lib))} "
                         f"(vllm_lb/hardware/accelerators/<key>.yml)")
    return found


def _listed(acc: Accelerator, attr: str, value: str, what: str) -> str:
    """"" when `value` is among the accelerator's `attr`; otherwise why not, in its own source's words."""
    f: Fact = getattr(acc, attr)
    if not f.known:
        return f"{acc.key} states no {what} ({f.source}), so {value!r} cannot be checked against it"
    if value not in f.value:
        return (f"{acc.key} ({acc.name}) has no {what} {value!r}: it lists {', '.join(f.value) or 'none'} "
                f"({f.source})")
    return ""


def compatibility(acc: Accelerator, *, quant_format: str | None = None, platform: str | None = None,
                  precision: str | None = None) -> str:
    """"" when the accelerator runs each thing named; otherwise the FIRST refusal, naming the accelerator and the
    missing piece. What is not named is not checked — the caller names what its launch needs."""
    for attr, value, what in (("platforms", platform, "platform"), ("quant_formats", quant_format, "quant format"),
                              ("precisions", precision, "precision")):
        if value is not None:
            why = _listed(acc, attr, value, what)
            if why:
                return why
    return ""
