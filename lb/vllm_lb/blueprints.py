"""REFERENCE SETUPS — the BLUEPRINTS: the project's real, proven setups by hardware type, which a coding agent running
the setup wizard (SETUP.md; deploy/setup/questions.yml + render.py) uses to WRITE a deployment's own fleet settings.
Never a runtime object: no LB module imports this one, the panel offers none, the API takes none, and a fleet file naming
one is refused at boot (fleet.RETIRED_FLEET_KEYS / RETIRED_CLUSTER_KEYS). Operator, 2026-10-30: "Blueprints meant to be
only for agent to build their own fleet settings, not something user chooses" (docs/design/engines.md, "Blueprints").

WHY A THIRD THING BESIDE THE LIBRARY. The model library (library/models.json, recipes.json, measurements.json, the image
matrix, the accelerators) is MEASURED FACT: what a checkpoint is, how an engine launches it, what it was measured to
take. It must not carry anybody's choice — the day the pair's live 12 seats × 131,072 became a library default, every
other deployment would start at a figure chosen for one pair of boxes and nobody could tell it from a measurement. A
blueprint is the choice, said as one: a model and EVERY knob its target's Start takes, the target it is for (an
accelerator × TP and the control that starts it), a label and a `source` — why these numbers. The wizard asks "start
from a reference blueprint?" with the ones whose target is the answered hardware, and choosing one PREFILLS the
cluster's answers (`render.py --prefill`), which the user confirms or edits; what is rendered is plain fleet settings.

ONE HOME: library/blueprints/<id>.yml, one file per blueprint, its id the file's stem, named by hardware TYPE and model,
never by a host or a deployment. A deployment's own choices are its answers and its fleet file, not a blueprint.

EXPLICIT-COMPLETE (the design's choice, stated here once): every knob of the target's Start is PRESENT — a blueprint
that leaves one out is refused, naming it, never filled in. What a value may be:
  * a cluster Start (control `ssh`): every knob the model's engine takes is a VALUE — the figure the row resolves to
    written out (`cluster_exec.resolved_launch`), so a blueprint never changes when a library row does; a knob its
    engine does not take (`catalog.ENGINE_REFUSES`), or a PLE placement for a model with no PLE table, is `null` and
    must be: a number there would be prefilled and dropped.
  * a node launch (control `local`): the knob set is the node ENGINE's settings (`fleet.NodeEngine.settings`), and
    `null` on an optional one (fleet.OPTIONAL) is a real choice — vLLM's own, no flag — exactly what the fleet's launch
    block means by leaving it out. The parsers are the launch block's, so a node blueprint names them under `fixed`.

THE LIBRARY CHECK (`check_library`, every test run — lb/tests/test_blueprints.py): keys complete, the model and its twin
known, the knob set the engine's, the image matrix has the engine × build × accelerator, the model's limits bound seats
and context, and the combination is MEASURED (`covers`) or the blueprint says `estimated: true` — and says it only where
it is not measured. Whether a prefilled answer fits ONE deployment (its offers, KV dtypes, staged checkpoints, its box's
RAM tier, the card) is that deployment's loader's and its first Start's check, run by render.py --check.
"""
from __future__ import annotations

import functools
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from vllm_lb import yamlfile

#: Where the shipped blueprints sit: one `<id>.yml` a blueprint, beside the rest of the library.
BLUEPRINTS_DIR = Path(__file__).resolve().parent / "library" / "blueprints"

#: THE KNOBS OF A CLUSTER START, in the order the switch takes them (`cluster.DesiredStore.set`): seats and context
#: (always numbers in the switch), then every engine knob the POST /lb/clusters/<name> body names. The wizard's
#: `cluster.first_start_settings` takes exactly these (deploy/setup/render.py reads this tuple).
CLUSTER_KNOBS = ("max_num_seqs", "max_model_len", "max_num_batched_tokens", "num_speculative_tokens", "kv_cache_dtype",
                 "kv_cache_memory", "ple_replicate", "kv_offload_gib", "long_prefill_token_threshold")
#: The keys of a blueprint, all required — explicit-complete; a node's also names the parsers it was run with.
KEYS = ("label", "source", "target", "model", "uncensored", "estimated", "knobs")
NODE_KEYS = (*KEYS, "fixed")
#: A target: the accelerator × TP it is for and the control that starts it — a hardware TYPE's, never one cluster's.
TARGET_KEYS = ("accelerator", "tp", "control")
#: A blueprint id: the file's stem, what the wizard's `cluster.blueprint` answer names.
ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,95}")
#: The pool sized at each Start from what the boxes have left (cluster.KV_AUTO's word) — a value, not a size.
KV_AUTO = "auto"


@dataclass(frozen=True)
class Target:
    """What a blueprint is FOR: the accelerator its kernels target, the TP, the control that starts it (fleet.CONTROLS:
    it decides the knob set) — what the wizard matches against a cluster's answered cards and control."""

    accelerator: str
    tp: int
    control: str


@dataclass(frozen=True)
class Blueprint:
    """One named, complete reference setup (module docstring)."""

    id: str
    label: str
    source: str
    target: Target
    model: str
    uncensored: bool
    estimated: bool
    knobs: Mapping[str, Any]
    #: A node's parsers (`fleet.OPTIONAL_FIXED` its task takes) — the wizard prefills them as its launch block's; {} for
    #: a pair.
    fixed: Mapping[str, Any] = field(default_factory=dict)

    @property
    def ref(self) -> str:
        """The model as the fleet and a node's intent write it: `<id>[@uncensored]`."""
        return f"{self.model}@uncensored" if self.uncensored else self.model

    @property
    def node(self) -> bool:
        """Is it a node launch (its control selects among a launch block's choices) rather than a cluster Start?"""
        from vllm_lb.fleet import CONTROLS
        return CONTROLS[self.target.control].selects


# ── reading ──────────────────────────────────────────────────────────────────────────────────────


def _keys(where: str, raw: Any, required, optional=()) -> dict:
    if not isinstance(raw, dict):
        raise ValueError(f"{where} must be a mapping, got {type(raw).__name__}")
    missing = [k for k in required if k not in raw]
    unknown = [str(k) for k in raw if k not in required and k not in optional]
    if missing or unknown:
        raise ValueError(f"{where} takes {', '.join([*required, *(f'{o} (optional)' for o in optional)])}"
                         + (f"; missing {', '.join(missing)} — a blueprint is complete: every key is stated, none is "
                            f"filled in" if missing else "")
                         + (f"; unknown {', '.join(unknown)}" if unknown else ""))
    return raw


def _text(where: str, v: Any) -> str:
    if not isinstance(v, str) or not v.strip():
        raise ValueError(f"{where} must be a non-empty string, got {v!r}")
    return v.strip()


def _bool(where: str, v: Any) -> bool:
    if not isinstance(v, bool):
        raise ValueError(f"{where} must be true or false, got {v!r}")
    return v


def _whole(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def parse(bid: str, raw: Any, where: str) -> Blueprint:
    """One blueprint's text as read — its SHAPE only (ids, keys, types); `check_library` judges it against the library.
    ValueError naming `where` and the value."""
    from vllm_lb.fleet import CONTROLS
    if not isinstance(bid, str) or not ID.fullmatch(bid):
        raise ValueError(f"{where}: a blueprint id is lowercase letters, digits, '.', '_' or '-', got {bid!r}")
    if not isinstance(raw, dict):
        raise ValueError(f"{where} must be a mapping, got {type(raw).__name__}")
    t = _keys(f"{where}.target", raw.get("target"), TARGET_KEYS)
    control = t["control"]
    if control not in CONTROLS or not CONTROLS[control].drives:
        raise ValueError(f"{where}.target.control must be a control the LB starts "
                         f"({', '.join(c for c, k in CONTROLS.items() if k.drives)}), got {control!r}")
    _keys(where, raw, NODE_KEYS if CONTROLS[control].selects else KEYS)
    if not _whole(t["tp"]) or t["tp"] < 1:
        raise ValueError(f"{where}.target.tp must be a whole number from 1, got {t['tp']!r}")
    target = Target(_text(f"{where}.target.accelerator", t["accelerator"]), t["tp"], control)
    knobs = raw["knobs"]
    if not isinstance(knobs, dict):
        raise ValueError(f"{where}.knobs must be a mapping of knob → value, got {type(knobs).__name__}")
    fixed = raw.get("fixed", {})
    if not isinstance(fixed, dict):
        raise ValueError(f"{where}.fixed must be a mapping of the node's parsers, got {type(fixed).__name__}")
    return Blueprint(id=bid, label=_text(f"{where}.label", raw["label"]), source=_text(f"{where}.source", raw["source"]),
                     target=target, model=_text(f"{where}.model", raw["model"]),
                     uncensored=_bool(f"{where}.uncensored", raw["uncensored"]),
                     estimated=_bool(f"{where}.estimated", raw["estimated"]),
                     knobs=MappingProxyType(dict(knobs)), fixed=MappingProxyType(dict(fixed)))


# ── the library check ────────────────────────────────────────────────────────────────────────────


@functools.cache
def _library_rows() -> tuple[dict, dict, dict]:
    """(models, recipes, measurements) of the shipped library, each by model id — read once, as `catalog.document` does."""
    from vllm_lb.catalog import _by_model, _library
    models = {m["id"]: m for m in _library("models.json", "models") if isinstance(m, dict) and "id" in m}
    return (models, _by_model(_library("recipes.json", "recipes"), "recipes.json"),
            _by_model(_library("measurements.json", "measurements"), "measurements.json"))


def _value(fact: Any) -> Any:
    """A models.json `{value, source}` fact's value (a bare value as it is)."""
    return fact.get("value") if isinstance(fact, dict) and "value" in fact else fact


def engine_of(bp: Blueprint) -> str:
    """The engine the blueprint's target runs its model on: the node's `node_engine`, the cluster recipe's `engine`."""
    _, recipes, _ = _library_rows()
    recipe = recipes.get(bp.model) or {}
    if bp.node:
        return recipe.get("node_engine") or ""
    from vllm_lb.catalog import VLLM
    return ((recipe.get("launch") or {}).get("cluster") or {}).get("engine", VLLM)


def knob_set(bp: Blueprint) -> tuple[str, ...]:
    """THE KNOBS this blueprint's target takes — a cluster Start's (`CLUSTER_KNOBS`), or the node engine's settings."""
    if not bp.node:
        return CLUSTER_KNOBS
    from vllm_lb.fleet import ENGINES
    models, _, _ = _library_rows()
    task = _value((models.get(bp.model) or {}).get("task"))
    return tuple(s.key for s in ENGINES[engine_of(bp)].settings(task) if s.key != "model")


def fixed_set(bp: Blueprint) -> tuple[str, ...]:
    """The fleet-fixed parsers a node blueprint names (`fleet.OPTIONAL_FIXED` its task takes); () for a pair."""
    if not bp.node:
        return ()
    from vllm_lb.fleet import TASK_KEYS
    models, _, _ = _library_rows()
    return ("reasoning_parser", *TASK_KEYS[_value(models[bp.model].get("task"))])


def check_library(bp: Blueprint) -> None:
    """THE LIBRARY CHECK (module docstring, 1): ValueError naming the blueprint and what is wrong — the model and its
    twin, the knob set complete and nothing else, the null rule, the image matrix, measured-or-explicit-estimate."""
    from vllm_lb.catalog import CLUSTER_MODE, ENGINE_REFUSES
    from vllm_lb.fleet import ENGINES
    from vllm_lb.hardware import accelerators
    from vllm_lb.images import ImageMissing, image_for, images
    w = f"blueprint {bp.id}"
    accs = accelerators()
    if bp.target.accelerator not in accs:
        raise ValueError(f"{w}: target.accelerator {bp.target.accelerator!r} is no accelerator: the library has "
                         f"{', '.join(sorted(accs))}")
    models, recipes, measured = _library_rows()
    row = models.get(bp.model)
    if row is None:
        raise ValueError(f"{w}: no model {bp.model!r} in library/models.json")
    if bp.uncensored and "uncensored" not in row:
        raise ValueError(f"{w}: {bp.model} has no uncensored twin in the library: uncensored must be false")
    engine = engine_of(bp)
    recipe = (recipes.get(bp.model) or {}).get("launch", {}).get(CLUSTER_MODE)
    if bp.node and engine not in ENGINES:
        raise ValueError(f"{w}: {bp.model} names no node engine (library/recipes.json node_engine): a "
                         f"{bp.target.control} node cannot run it")
    if not bp.node and recipe is None:
        raise ValueError(f"{w}: {bp.model} has no cluster recipe (library/recipes.json launch.cluster)")
    # THE KNOB SET, complete and nothing else.
    want = knob_set(bp)
    missing = [k for k in want if k not in bp.knobs]
    unknown = [k for k in bp.knobs if k not in want]
    if missing or unknown:
        raise ValueError(f"{w}: knobs are exactly {', '.join(want)} ({engine}, control {bp.target.control})"
                         + (f"; missing {', '.join(missing)} — every knob is stated, none is filled in"
                            if missing else "")
                         + (f"; {', '.join(unknown)} is no knob of this target" if unknown else ""))
    fixed = fixed_set(bp)
    if bp.node and sorted(bp.fixed) != sorted(fixed):
        raise ValueError(f"{w}: fixed names exactly {', '.join(fixed)} (the node's parsers, the fleet launch "
                         f"block's), got {', '.join(map(str, bp.fixed)) or 'none'}")
    if not bp.node:
        # THE NULL RULE of a cluster Start: a value for every knob the engine takes; null — and only null — for one it
        # does not, and for a PLE placement where the model has no PLE table.
        refuses = ENGINE_REFUSES.get(engine, {})
        no_table = recipe.get("ple_replicate") is None
        for k in CLUSTER_KNOBS:
            v = bp.knobs[k]
            must_null = k in refuses or (k == "ple_replicate" and no_table)
            if must_null and v is not None:
                why = refuses.get(k) or f"{bp.model} has no PLE table to place"
                raise ValueError(f"{w}: knobs.{k} must be null: {why}")
            if not must_null and v is None:
                raise ValueError(f"{w}: knobs.{k} is null — write the value it resolves to (an explicit-complete "
                                 f"blueprint never changes when a library row does)")
            if v is None:
                continue
            if k == "ple_replicate" and not isinstance(v, bool):
                raise ValueError(f"{w}: knobs.ple_replicate must be true or false, got {v!r}")
            if k == "kv_cache_dtype" and not isinstance(v, str):
                raise ValueError(f"{w}: knobs.kv_cache_dtype must be a dtype name, got {v!r}")
            if k == "kv_cache_memory" and v != KV_AUTO and not _whole(v):
                raise ValueError(f"{w}: knobs.kv_cache_memory must be bytes or \"{KV_AUTO}\", got {v!r}")
            if k not in ("ple_replicate", "kv_cache_dtype", "kv_cache_memory") and not _whole(v):
                raise ValueError(f"{w}: knobs.{k} must be a whole number, got {v!r}")
    # THE MODEL'S OWN LIMITS (models.json `limits`) bound seats and context, wherever the target.
    limits = row.get("limits") or {}
    lo, hi = limits.get("max_num_seqs", [1, None])
    seats, ctx = bp.knobs.get("max_num_seqs"), bp.knobs.get("max_model_len")
    if "max_num_seqs" in want and (not _whole(seats) or seats < lo or (hi is not None and seats > hi)):
        raise ValueError(f"{w}: knobs.max_num_seqs must be from {lo} to {hi} for {bp.model}, got {seats!r}")
    if "max_model_len" in want and (not _whole(ctx) or ctx < 1 or ctx > limits.get("max_model_len", ctx)):
        raise ValueError(f"{w}: knobs.max_model_len must be up to {limits.get('max_model_len')} for {bp.model}, "
                         f"got {ctx!r}")
    # THE IMAGE MATRIX has the engine for this accelerator (a pair's: the recipe's build; a node's: any build — its
    # image is the fleet launch block's, pinned by digest, matched to the matrix by nobody yet: engines.md Open items).
    platforms = accs[bp.target.accelerator].platforms
    plats = list(platforms.value) if platforms.known else []
    if bp.node:
        matrix = images().get(engine)
        if matrix is None or not any(e.accelerator == bp.target.accelerator for e in matrix.entries):
            raise ValueError(f"{w}: the image matrix has no {engine} image for {bp.target.accelerator} "
                             f"(vllm_lb/library/images/{engine}.yml)")
    else:
        problems = []
        for p in plats:
            try:
                image_for(engine, recipe.get("build", ""), bp.target.accelerator, p)
                break
            except ImageMissing as exc:
                problems.append(str(exc))
        else:
            raise ValueError(f"{w}: {'; '.join(problems) or f'{bp.target.accelerator} lists no platform'}")
    # MEASURED, OR THE ESTIMATE SAID OUT LOUD — and the estimate only where it is not measured.
    covers = [(c["engine"], c["accelerator"], c["tp"]) for c in (measured.get(bp.model) or {}).get("covers", [])]
    here = (engine, bp.target.accelerator, bp.target.tp)
    if here not in covers and not bp.estimated:
        raise ValueError(f"{w}: {bp.model} is not measured on {' × '.join(map(str, here))} (it covers "
                         f"{', '.join(' × '.join(map(str, c)) for c in covers) or 'nothing'}): measure it there, or "
                         f"say estimated: true")
    if here in covers and bp.estimated:
        raise ValueError(f"{w}: estimated: true, but {bp.model} is measured on {' × '.join(map(str, here))}: "
                         f"estimated must be false")


def read_library(directory: str | Path = BLUEPRINTS_DIR) -> Mapping[str, Blueprint]:
    """Every shipped blueprint in `directory` (`<id>.yml`), each checked against the library — ValueError naming the
    file on the first that is not."""
    out = {}
    for f in sorted(Path(directory).glob("*.yml")):
        bp = parse(f.stem, yamlfile.read(f), f"library/blueprints/{f.name}")
        check_library(bp)
        out[bp.id] = bp
    return MappingProxyType(out)


@functools.cache
def library() -> Mapping[str, Blueprint]:
    """The shipped blueprints, read and checked once."""
    return read_library()
