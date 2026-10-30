"""RESOLUTION — what a model's launch IS on a given set of cards: engine × accelerator × platform × TP → the image,
or the refusal that names the missing piece (docs/design/engines.md, "Resolution at Start").

The five axes of the engine redesign meet here, and only here:

  model        its facts, read from its config.json (format, architectures, quant format — `catalog.ModelSpec`)
  engine       the recipe's engine and build; what the engine serves (engine_library.py)
  accelerator  what the target cards' kernels are (hardware/accelerators/<key>.yml, through each card type)
  card type    the product (hardware/types) — named by the fleet, never here
  topology     how many cards and how joined: the TARGET's `tp` (and, in phase 5, its executor)

A TARGET is what a ServingMode launches onto in THIS deployment: the cards of the fleet's cluster (or classic
servers) that mode starts, as ONE accelerator, ONE platform and a TP size. `fleet.catalog_targets` derives it from the
fleet file; mixed accelerators or platforms under one target are refused there (one image cannot serve both).

`resolve` runs the checks IN ORDER and stops at the first refusal, so the operator reads the nearest missing piece:

  1. the target exists        — the fleet has a cluster this mode launches onto;
  2. the engine serves it     — format, architectures, quant format (engine_library.model_problem);
  3. the accelerator runs it  — quant format, platform (hardware.compatibility);
  4. an image exists          — engine × build × accelerator × platform (images.image_for);
  5. the KV dtype runs there  — the profile's own dtype on that image and that arch (kv_cache_dtypes.json `problem`,
                                the image's record of which arch each dtype runs on).

MEMORY is the sixth check and stays where it always ran: at Start, against the boxes' memory (cluster.py's switch,
cluster_defaults.memory_admission, memory_estimate). Whether the launch is MEASURED on this target is the catalog's
(measurements keyed model × engine × accelerator × tp, phase 4).

A refusal is a string the caller shows verbatim — never an exception the LB's boot could die of, and never a
substitute: no "nearest" accelerator, no "default" image (AGENTS.md: no fallbacks).
"""
from __future__ import annotations

from dataclasses import dataclass

from vllm_lb import engine_library, images
from vllm_lb.hardware import Accelerator, compatibility
from vllm_lb.kv_dtype import library as kv_dtypes


@dataclass(frozen=True)
class Target:
    """The cards a ServingMode launches onto in this deployment: their one accelerator, their boxes' one CPU platform,
    and how many of them one launch spans (TP)."""

    accelerator: Accelerator
    platform: str
    tp: int
    #: Where it comes from (the fleet cluster or servers), for the refusal and /lb/status.
    source: str = ""

    @property
    def key(self) -> str:
        return f"{self.accelerator.key} × {self.platform} × TP={self.tp}"

    def view(self) -> dict:
        return {"accelerator": self.accelerator.key, "arch": self.accelerator.arch.value, "platform": self.platform,
                "tp": self.tp, "source": self.source}


def target_of(cards, platforms, *, where: str) -> Target:
    """The ONE target `cards` (fleet Cards, head first) on boxes of `platforms` make — or ValueError naming the mix:
    two accelerators under one launch would need two images, and one launch runs one."""
    accs = {c.type.accelerator.key: c.type.accelerator for c in cards}
    plats = set(platforms)
    if not cards:
        raise ValueError(f"{where}: no serving card — nothing to launch onto")
    if len(accs) > 1:
        raise ValueError(f"{where}: its serving cards are of {len(accs)} accelerators ({', '.join(sorted(accs))}): one "
                         f"launch runs one image, built for one accelerator")
    if len(plats) > 1:
        raise ValueError(f"{where}: its boxes are of {len(plats)} platforms ({', '.join(sorted(plats))}): one launch "
                         f"runs one image, built for one platform")
    return Target(next(iter(accs.values())), next(iter(plats)), len(cards), source=where)


@dataclass(frozen=True)
class Resolution:
    """What a launch resolved to on a target — or why it cannot run there (`refused`, "" when it can)."""

    engine: str
    build: str
    target: Target | None
    image: str | None = None
    image_source: str = ""
    refused: str = ""
    #: How its memory is known on this target: "measured" (a measurement covers engine × accelerator × TP),
    #: "unmeasured" (none does: refused by default), "estimated" (the operator chose the estimate for this start);
    #: "" where the mode sizes nothing itself (classic) or it did not resolve.
    sizing: str = ""
    #: The measurement key it is sized by ("vllm × nvidia-sm121 × TP=2"); "" when none.
    measurement: str = ""
    #: How the ESTIMATE was made for an unmeasured target (catalog.estimated_footprint), or why it cannot be.
    estimate: str = ""

    def view(self) -> dict:
        return {"engine": self.engine, "build": self.build,
                "target": self.target.view() if self.target is not None else None,
                "image": self.image, "image_source": self.image_source, "refused": self.refused,
                "sizing": self.sizing, "measurement": self.measurement, "estimate": self.estimate}


def resolve(*, engine: str, build: str, target: Target | None, fmt, architectures, quant_format,
            kv_cache_dtype: str | None = None, kv_refuses=None, kv_backend=None, drafting: bool = False,
            no_target: str = "") -> Resolution:
    """The image `engine`'s `build` runs for this model on `target`, or the first refusal (module docstring)."""
    def refused(why: str) -> Resolution:
        return Resolution(engine, build, target, refused=why)

    if target is None:
        return refused(no_target or "this deployment names no cards for it to launch onto")
    why = engine_library.model_problem(engine_library.engine(engine), fmt=fmt, architectures=architectures,
                                       quant_format=quant_format)
    if why:
        return refused(why)
    why = compatibility(target.accelerator, quant_format=quant_format, platform=target.platform)
    if why:
        return refused(why)
    try:
        entry = images.image_for(engine, build, target.accelerator.key, target.platform)
    except images.ImageMissing as exc:
        return refused(str(exc))
    if kv_cache_dtype is not None:
        why = kv_dtypes().problem(entry.image, kv_cache_dtype, [target.accelerator.arch.value], kv_refuses,
                                  drafting=drafting, backend=kv_backend)
        if why:
            return refused(f"its KV cache dtype: {why}")
    return Resolution(engine, build, target, entry.image, entry.source)
