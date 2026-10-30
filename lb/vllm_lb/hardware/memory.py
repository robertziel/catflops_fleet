"""HOW A BOX'S GPU MEMORY IS READ — one class per memory model, and the one rule that picks between them.

The boxes that send hardware samples (hw.py) have GPU memory of two different KINDS:

  UnifiedMemory   the GPU's memory IS the box's RAM. A GB10 (DGX Spark) has 128 GB of LPDDR5X that its Arm cores
                  and its GPU share; nvidia-smi has no VRAM of its own to report there (`[N/A]`), and the sender
                  turns that into `vram_used_mb: 0, vram_total_mb: 0`. An APU's iGPU (the Radeon 8060S: "Dynamic +
                  carveout") takes the box's RAM too, beside a small carve-out rocm-smi calls VRAM. Neither is a
                  figure, so the memory read is the box's own `ram_used_mb` / `ram_total_mb`.
  DiscreteMemory  each card has VRAM of its own: an RTX 5090's 32 GB beside the host's RAM. The box's figure is the
                  cards' VRAM SUMMED — the aggregate every reader of /lb/status takes — and each card is its own row.

WHICH ONE a box is, is NAMED — by its cards' hardware types in the deployment's fleet file (vllm_lb/fleet.py) — never
inferred from what its sender reports: a GB10's 0/0, an APU's carve-out and a dead GPU CLI's empty list all look alike
on the wire, and a box that names no card claims no memory model at all.

The model also decides how a box's GPUs are DRAWN (`rows`): a unified box has ONE GPU row — the box's GPU, its memory
the MEM row the card already shows — and a discrete box one row per card, each with its own utilisation and its own
VRAM line (operator, 2026-10-11: two cards' VRAM summed is no card's). The panel draws the rows it is given and never
asks which kind of box it is.

A third kind of memory is a third subclass with its own tests. A caller never asks `if unified:` — it asks the
model (`read`), and the model answers for its kind.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _num(value: Any) -> float | int | None:
    """A number, or None. Booleans are not numbers here (`True` would read as 1 MB)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


@dataclass(frozen=True)
class GpuRow:
    """One GPU row of a box's machine tile: its ring's `label`, the card it reads (`card`: nvidia-smi's index, None =
    the box's one GPU), and whether it has a VRAM line of its own (`vram`)."""

    label: str
    card: int | None
    vram: bool


class MemoryModel:
    """One kind of GPU memory: how a box's memory figure is read off its hardware sample, and how its GPUs are drawn."""

    #: True where the GPU's memory is the box's own RAM — what `/lb/status` reports as `mem_unified`.
    unified = False
    #: The family's name — `/lb/status.hardware[box].memory`, a type file's `memory:` — so no caller has to ask
    #: `unified` to say which one a box is.
    name = ""
    #: What a type's `memory_gb` IS to its GPU in this family — the memory an engine's gpu_memory_utilization is a
    #: fraction of (HardwareProfile.gpu_memory).
    holds = ""
    #: Why a KV TIER IN THE HOST'S RAM (vLLM --kv-offloading-size: the native OffloadingConnector, a pinned region in
    #: the host's /dev/shm) adds nothing in this family — "" where it is a second, larger pool beside the GPU's own
    #: memory. The fleet refuses the tier in these words (fleet.py `_ram_tier`), never by asking which family it is.
    ram_tier_refusal = ""
    #: The type fact a launch's gpu_memory_utilization is a fraction of here (HardwareProfile.gpu_memory), and what it
    #: is, in words.
    fraction_of = ""
    fraction_holds = ""

    def pool_problem(self, box, pool_gib: float) -> str:
        """Why a GPU pool of `pool_gib` cannot be had on `box` beyond what its card's own figures say — "" when it can."""
        raise NotImplementedError

    def shared_pool(self, box) -> tuple[int | None, str]:
        """`(GiB, "")`: the ONE pool engines of several clusters on `box` share, summed against it (fleet.share_problem)
        — or `(None, why)`: the box has none, and one cluster holds it whole."""
        raise NotImplementedError

    def read(self, sample: dict, gpus: list) -> tuple[float | int | None, float | int | None]:
        """`(used_mb, total_mb)` of the GPU's memory on the box `sample` came from; None where the wire did
        not send it. `gpus` is the sample's card list, as the receiver already took it out."""
        raise NotImplementedError

    def rows(self, cards: tuple[int, ...]) -> tuple[GpuRow, ...]:
        """The machine tile's GPU rows for a box whose cards are `cards` (nvidia-smi indices, the fleet's)."""
        raise NotImplementedError


class UnifiedMemory(MemoryModel):
    """The GPU's memory IS the box's RAM (a GB10). One pool, so one figure: the RAM the box reports."""

    unified = True
    name = "unified"
    holds = "the box's RAM, which its GPU shares"
    #: What a launch's gpu_memory_utilization is a FRACTION OF on this family: the GPU pool vLLM measures at start
    #: (the type's `visible_gib`) — an APU's spec states no size, and the box's RAM is shared with everything on it.
    fraction_of, fraction_holds = "visible_gib", "the GPU pool vLLM measures, carved from the box's RAM"
    ram_tier_refusal = ("unified memory: the host RAM is the GPU's own pool, so a KV tier in it takes the very memory "
                        "the KV pool lives in instead of adding a second one")

    def read(self, sample: dict, gpus: list) -> tuple[float | int | None, float | int | None]:
        return _num(sample.get("ram_used_mb")), _num(sample.get("ram_total_mb"))

    def rows(self, cards: tuple[int, ...]) -> tuple[GpuRow, ...]:
        """ONE row: the box's GPU. Its memory is the box's RAM — the MEM row — so it has no VRAM line."""
        return (GpuRow("GPU", None, False),)

    def pool_problem(self, box, pool_gib: float) -> str:
        """THE BOX BOUNDS THE POOL: a GPU pool of `pool_gib` (util × the measured pool) is carved from `box`'s own RAM,
        which the LB, the chat, the broker's work share — and vLLM's own start check does not guard it (on an APU it
        reads the pool's free figure, ~29 GiB above the box's free RAM, measured 2026-10-17). So the box states the
        most GiB of its RAM a node's GPU pool may take (`gpu_ram_max_gib`, the fleet's: no default), and a pool above
        it is refused in its numbers; a box that states none is refused naming the key."""
        bound = getattr(box, "gpu_ram_max_gib", None)
        if bound is None:
            return (f"box {box.name} has unified memory — a GPU pool is carved from its own RAM — and states no "
                    f"gpu_ram_max_gib (the most GiB of it a node's pool may take), so no launch on it can be bounded")
        if pool_gib > bound:
            return (f"a {pool_gib:.2f} GiB GPU pool > box {box.name}'s gpu_ram_max_gib {bound} GiB: the pool is the "
                    f"box's own RAM — lower gpu_memory_utilization")
        return ""

    def shared_pool(self, box) -> tuple[int | None, str]:
        """Every node's pool on the box is carved from its one RAM, so the box's bound is what they share."""
        bound = getattr(box, "gpu_ram_max_gib", None)
        return (bound, "" if bound is not None else f"box {box.name} states no gpu_ram_max_gib")


class DiscreteMemory(MemoryModel):
    """Each card has VRAM of its own. The box's figure is the VRAM of the cards that report a total, summed."""

    unified = False
    name = "discrete"
    holds = "its own VRAM"
    fraction_of, fraction_holds = "memory_gb", "its own VRAM"

    def read(self, sample: dict, gpus: list) -> tuple[int, int]:
        used = total = 0
        for g in gpus:
            if isinstance(g, dict) and (_num(g.get("vram_total_mb")) or 0) > 0:
                total += int(g["vram_total_mb"])
                used += int(_num(g.get("vram_used_mb")) or 0)
        return used, total

    def rows(self, cards: tuple[int, ...]) -> tuple[GpuRow, ...]:
        """One row per card, in index order, each with a VRAM line of its own."""
        return tuple(GpuRow(f"GPU {i}", i, True) for i in sorted(cards))

    def pool_problem(self, box, pool_gib: float) -> str:
        """A discrete card's pool is its own VRAM: the box's RAM bounds nothing of it."""
        return ""

    def shared_pool(self, box) -> tuple[int | None, str]:
        """Each card's VRAM is its own and nothing here bounds two engines on one card: none is shared."""
        return (None, f"box {box.name}'s cards have VRAM of their own, which no fleet bound divides between engines")


UNIFIED = UnifiedMemory()
DISCRETE = DiscreteMemory()

