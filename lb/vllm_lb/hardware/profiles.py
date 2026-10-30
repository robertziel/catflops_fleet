"""WHAT A KIND OF GPU IS — its facts are DATA (types/<key>.yml, one file per type); its behaviour is a class.

AGENTS.md, operator rule 2026-10-11: a deployment is configuration, never code. So no hardware figure is written in
Python: a TYPE is a YAML file in `types/`, named by its key (`gb10.yml` is the type `gb10`), and a fleet file names
each box's cards by those keys (vllm_lb/fleet.py). What the gateway may CLAIM about a GPU it did not measure:

  cuda_cores          the CUDA CORES tile's denominator
  peak_gbps           one GPU's DRAM bandwidth, GB/s — the MEMORY BANDWIDTH tile's ceiling
  peak_tflops         one GPU's DENSE TFLOP/s per precision ("bf16", "fp8", "nvfp4") — the COMPUTE tile's ceiling
  bandwidth_counter   whether a DRAM-traffic counter can be read on this GPU at all. Where there is none, a MEMORY
                      BANDWIDTH figure is a MODEL (vLLM's analytical bytes, the LB's own for SGLang). Whether an
                      ENGINE publishes such an estimate is the engine's (`--enable-mfu-metrics`; rates.py NOT_EXPOSED,
                      "not exposed by this server") — never a hardware fact
  accelerator         WHAT ITS KERNELS TARGET (accelerator.py, accelerators/<key>.yml): the type names one, and its
                      `arch` (sm_121 on a GB10, gfx1151 on the Radeon 8060S) and `gpu_runtime` are read THROUGH it —
                      one place per value since 2026-10-30; a type file stating either is refused as unknown. A peak
                      at a precision the accelerator cannot run, or a host CPU it never comes in, is refused
  memory_gb           the memory the spec gives it (absent where it is the box's own RAM, sized by the box)
  host_platform       the CPU platform its images must be built for, where the chip carries the box's CPU (a GB10
                      superchip: linux/arm64) — absent for a discrete card, whose box says (the fleet's `platform`)
  visible_gib         the memory vLLM SEES as the card's total at start (its MemorySnapshot.total_memory, CUDA's own
                      figure — under the spec's memory_gb), GiB as vLLM prints it: what gpu_memory_utilization takes
                      its share of in vLLM's start check (v1/worker/utils.py:761 @2eb491e)
  free_at_start_gib   the memory FREE on an otherwise idle card when that check runs — after vLLM's own CUDA context
                      exists (gpu_worker.py:516-523), so the context is already taken out of it. The check refuses a
                      fraction with util × visible_gib above it, before anything loads (fleet.py `_start_check`).
                      Both are MEASURED (a vLLM boot log on the card, its date): absent where none was, and said so
  gpu_runtime         HOW A CONTAINER ATTACHES IT — its accelerator's: `nvidia` (Docker's NVIDIA device reservation) or
                      `rocm` (/dev/kfd and /dev/dri). A one-host node's compose is SELECTED by this value
                      (deploy/standalone/node-gpu.<value>.yml, `fleet.node_env` EDGE_NODE_GPU_RUNTIME) — data, never
                      an `if <vendor>:`
  controls            the cluster control methods proven on it (the fleet's `control: ssh` is lb/cluster/tp-rank)
  memory              its BEHAVIOUR FAMILY — `UnifiedProfile` (the GPU's memory is the box's RAM) or
                      `DiscreteProfile` (VRAM of its own); the class, never a caller, decides what that changes

EVERY FIGURE IS A `Fact`: a value and where it comes from — a vendor's spec page, or a measurement on this project's
fleet with its date and where it is recorded. A Fact without a source cannot be built, and a type file with one is
refused. Where no source states a figure, its value is null and its source says why: absent, and said so. Peaks are
DENSE (NVIDIA prints `dense/sparse`; the gateway's FLOPs are dense) and at the GPU's boost clock, as the source says.

A NEW TYPE is a new YAML file (and its test in tests/test_hardware_profiles.py); a new BEHAVIOUR FAMILY is a new
subclass here. Never an `if <type>:` at a caller. A type is chosen by its key (`profile`), never guessed.
"""
from __future__ import annotations

import functools
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Mapping

from vllm_lb import yamlfile
from vllm_lb.hardware.accelerator import Accelerator, accelerator, accelerators, compatibility
from vllm_lb.hardware.facts import GPU_RUNTIMES, PLATFORMS, Fact, fact as _fact, keys as _keys
from vllm_lb.hardware.memory import DISCRETE, UNIFIED, MemoryModel

#: The hardware-type library this repo ships: one YAML file per type.
TYPES_DIR = Path(__file__).resolve().parent / "types"
__all__ = ["GPU_RUNTIMES", "PLATFORMS", "Fact"]


@dataclass(frozen=True, eq=False)
class HardwareProfile:
    """One hardware TYPE: its facts (from its YAML file) and its behaviour family (the subclass)."""

    key: str
    name: str
    controls: tuple[str, ...]
    #: What its kernels target (accelerator.py): its `arch` and `gpu_runtime` are the accelerator's, read through.
    accelerator: Accelerator
    memory_gb: Fact
    cuda_cores: Fact
    peak_gbps: Fact
    bandwidth_counter: Fact
    peak_tflops: Mapping[str, Fact]
    host_platform: Fact
    #: What vLLM's start check reads off the card (fleet.py `_start_check`): its visible total and its free memory at
    #: start, GiB — measured, or absent with why.
    visible_gib: Fact
    free_at_start_gib: Fact
    #: Whether the node's MFU byte-size patch (deploy/standalone/vllm-mfu) LOADS in a container on this card's runtime
    #: and image — measured, or null with why. A launch whose checkpoint needs it is refused where it is not true
    #: (fleet.py `_mfu_patch`): under ROCm the patch's sitecustomize recursed until OOM (2026-10-17).
    mfu_patch_loads: Fact

    #: The family's memory model (memory.py) — a property of the CLASS: every type of a family shares it.
    memory: ClassVar[MemoryModel]

    def __post_init__(self) -> None:
        object.__setattr__(self, "peak_tflops", MappingProxyType(dict(self.peak_tflops)))

    @property
    def arch(self) -> Fact:
        """What its kernels are compiled for — its accelerator's (one place per value)."""
        return self.accelerator.arch

    @property
    def gpu_runtime(self) -> Fact:
        """How a container attaches the card (`GPU_RUNTIMES`) — its accelerator's: which node compose override its
        node runs with."""
        return self.accelerator.gpu_runtime

    def peak_tflops_for(self, precision: str) -> Fact:
        """The dense peak at `precision`, or an absent Fact naming the precisions this type does list."""
        fact = self.peak_tflops.get(precision)
        if fact is not None:
            return fact
        listed = ", ".join(sorted(self.peak_tflops)) or "none"
        return Fact(None, f"{self.name} lists no {precision!r} peak here; it lists {listed}")

    def gpu_memory(self) -> Fact:
        """The memory a launch's gpu_memory_utilization is a fraction of, in GiB, as its FAMILY reads the type
        (`memory.fraction_of`): a discrete card's own VRAM (`memory_gb`; GB as a spec sheet prints DRAM capacity is
        binary — a 32 GB card is 16 x 2 GB GDDR7 devices — taken as GiB); on unified memory the GPU pool vLLM measures
        (`visible_gib`) — absent, with the type's own reason, where nothing states it."""
        fact = getattr(self, self.memory.fraction_of)
        return Fact(fact.value, f"{self.memory.fraction_holds}: {fact.source}")

    def view(self) -> dict:
        """The type as /lb/status carries it: every figure with its source, plain JSON."""
        return {"key": self.key, "name": self.name, "memory": self.memory.name,
                "controls": list(self.controls), "accelerator": self.accelerator.view(), "arch": self.arch.view(), "memory_gb": self.memory_gb.view(),
                "cuda_cores": self.cuda_cores.view(), "peak_gbps": self.peak_gbps.view(),
                "bandwidth_counter": self.bandwidth_counter.view(), "host_platform": self.host_platform.view(),
                "visible_gib": self.visible_gib.view(), "free_at_start_gib": self.free_at_start_gib.view(),
                "gpu_runtime": self.gpu_runtime.view(), "mfu_patch_loads": self.mfu_patch_loads.view(),
                "peak_tflops": {p: f.view() for p, f in self.peak_tflops.items()}}


class UnifiedProfile(HardwareProfile):
    """The GPU's memory IS the box's RAM — one pool. What changes with it lives in UnifiedMemory: the box's RAM is
    the GPU's memory figure, a vendor CLI's own "VRAM" is no figure at all (a GB10's [N/A], an APU's carve-out), and
    the box draws ONE GPU row. A cluster on such boxes is gated on their MemAvailable (memory_estimate.py)."""

    memory = UNIFIED


class DiscreteProfile(HardwareProfile):
    """VRAM of its own, beside the host's RAM: each card is its own memory figure and its own row (DiscreteMemory)."""

    memory = DISCRETE


#: The behaviour families a type file may name (`memory:`).
FAMILIES: Mapping[str, type[HardwareProfile]] = MappingProxyType({"unified": UnifiedProfile,
                                                                   "discrete": DiscreteProfile})

_TOP = ("name", "memory", "accelerator", "controls", "facts")
_FACTS = {"memory_gb": "number", "cuda_cores": "count", "peak_gbps": "number",
          "bandwidth_counter": "bool", "host_platform": "platform", "visible_gib": "number",
          "free_at_start_gib": "number", "mfu_patch_loads": "bool",
          "peak_tflops": "per precision"}
_PRECISION = re.compile(r"[a-z0-9_]+")


def _type(key: str, raw: Any, source: str, library: Mapping[str, Accelerator]) -> HardwareProfile:
    _keys(f"{source}: the type", raw, _TOP)
    name = raw["name"]
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{source}: name must be the type's name")
    family = FAMILIES.get(raw["memory"]) if isinstance(raw["memory"], str) else None
    if family is None:
        raise ValueError(f"{source}: memory {raw['memory']!r} is no behaviour family: {', '.join(FAMILIES)}")
    controls = raw["controls"]
    if not isinstance(controls, list) or not all(isinstance(c, str) and c for c in controls):
        raise ValueError(f"{source}: controls must be a list of control methods (e.g. [ssh] or [])")
    facts = _keys(f"{source}: facts", raw["facts"], tuple(_FACTS))
    peaks = facts["peak_tflops"]
    if not isinstance(peaks, dict) or not all(isinstance(p, str) and _PRECISION.fullmatch(p) for p in peaks):
        raise ValueError(f"{source}: facts.peak_tflops must map precisions (bf16, fp8, nvfp4 …) to facts")
    try:
        acc = accelerator(raw["accelerator"], library)
    except ValueError as exc:
        raise ValueError(f"{source}: accelerator: {exc}") from None
    out = family(key=key, name=name.strip(), controls=tuple(controls), accelerator=acc,
                 **{k: _fact(f"{source}: facts.{k}", facts[k], kind) for k, kind in _FACTS.items()
                    if k != "peak_tflops"},
                 peak_tflops={p: _fact(f"{source}: facts.peak_tflops.{p}", f, "number") for p, f in peaks.items()})
    # THE PRODUCT AGREES WITH WHAT ITS KERNELS TARGET: a peak at a precision the accelerator has no tensor unit for,
    # or a host CPU its accelerator never comes in, is a contradiction between two files — refused naming both.
    for p, f in out.peak_tflops.items():
        why = compatibility(acc, precision=p) if f.known else ""
        if why:
            raise ValueError(f"{source}: facts.peak_tflops.{p} states {f.value}, but {why}")
    if out.host_platform.known:
        why = compatibility(acc, platform=out.host_platform.value)
        if why:
            raise ValueError(f"{source}: facts.host_platform is {out.host_platform.value}, but {why}")
    return out


def load_types(directory: str | Path,
               library: Mapping[str, Accelerator] | None = None) -> Mapping[str, HardwareProfile]:
    """Every type in `directory` (`<key>.yml`), validated against the accelerator `library` (default: the shipped
    one) — or ValueError naming the file and the value."""
    files = sorted(Path(directory).glob("*.yml"))
    if not files:
        raise ValueError(f"{directory}: no hardware type (<key>.yml) there")
    accs = accelerators() if library is None else library
    return MappingProxyType({f.stem: _type(f.stem, yamlfile.read(f), f.name, accs) for f in files})


@functools.cache
def types() -> Mapping[str, HardwareProfile]:
    """The library this repo ships (TYPES_DIR), read once."""
    return load_types(TYPES_DIR)


def profile(key: str, library: Mapping[str, HardwareProfile] | None = None) -> HardwareProfile:
    """The type `key` names — or ValueError naming every key there is. Never a guess, never a nearest match."""
    lib = types() if library is None else library
    found = lib.get(key) if isinstance(key, str) else None
    if found is None:
        raise ValueError(f"no hardware type {key!r}: the types are {', '.join(sorted(lib))} "
                         f"(vllm_lb/hardware/types/<key>.yml)")
    return found
