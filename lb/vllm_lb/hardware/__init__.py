"""THE HARDWARE DOMAIN — what this gateway knows about a KIND of GPU, and the one place it lives.

The fleet spans hardware that differs in what a figure MEANS, not only in its size: a GB10 (DGX Spark) whose GPU
memory is the box's own RAM, discrete NVIDIA cards with VRAM of their own, an AMD iGPU whose memory is the box's RAM
beside a carve-out. AGENTS.md (operator rule, 2026-10-11): a deployment is configuration, never code — so the FACTS
are data and only the BEHAVIOUR is code:

  types/*.yml  one file per hardware TYPE (one product): memory family, its accelerator, memory, CUDA cores, peak
               GB/s, dense TFLOP/s per precision, DRAM counter, control methods — every figure {value, source},
               absent = null + why.
  accelerators/*.yml  one file per ACCELERATOR (what kernels target, shared by the products of one architecture):
               arch, gpu_runtime, platforms, precisions, quant formats (accelerator.py; docs/design/engines.md).
  facts.py     Fact — one figure and its source — and the strict readers both libraries share.
  profiles.py  HardwareProfile — UnifiedProfile / DiscreteProfile, one class per behaviour family; `load_types`
               reads and validates a library, `types()` is the shipped one, `profile(key)` names one.
  memory.py    MemoryModel — UnifiedMemory / DiscreteMemory: how a box's GPU memory is read off its sample, and
               how its GPUs are drawn (one row for a unified box, one per card for a discrete one).

Which box has which cards, and which of them serve, is the FLEET's (vllm_lb/fleet.py, the deployment's fleet file),
never this package's.

THE RULES (AGENTS.md):
  * a hardware figure is written in a type file, once, with its source; a caller asks the profile for it;
  * a new type is a new YAML file and its test; a new behaviour is a new class here — never an `if <type>:` in
    hw.py, the planner, the proxy or the panel;
  * a type is NAMED (in the fleet file), never guessed from a reading — a box's memory model too: what its sender
    reports never decides it (memory.py).

Import from `vllm_lb.hardware`, never from a submodule.
"""
from __future__ import annotations

from vllm_lb.hardware.accelerator import (ACCELERATORS_DIR, Accelerator, accelerator, accelerators, compatibility,
                                          load_accelerators)
from vllm_lb.hardware.facts import GPU_RUNTIMES, fact as read_fact, keys as read_keys
from vllm_lb.hardware.memory import DISCRETE, UNIFIED, DiscreteMemory, GpuRow, MemoryModel, UnifiedMemory
from vllm_lb.hardware.profiles import (FAMILIES, PLATFORMS, TYPES_DIR, DiscreteProfile, Fact, HardwareProfile,
                                       UnifiedProfile, load_types, profile, types)

__all__ = ["read_fact", "read_keys", "ACCELERATORS_DIR", "Accelerator", "GPU_RUNTIMES", "accelerator", "accelerators", "compatibility",
           "load_accelerators", "DISCRETE", "FAMILIES", "PLATFORMS", "TYPES_DIR", "UNIFIED", "DiscreteMemory", "DiscreteProfile", "Fact", "GpuRow",
           "HardwareProfile", "MemoryModel", "UnifiedMemory", "UnifiedProfile", "load_types", "profile", "types"]
