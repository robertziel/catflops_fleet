"""THE KV CACHE'S ELEMENT TYPE — what each engine image accepts, on which cards, and what one element costs.

Operator 2026-10-22: "Implement kv on fp4 then, allow set in model params on run, don't switch yet, dispatch and
implement to all models". The KV cache dtype is a RUN knob (`cluster.Desired.kv_cache_dtype` for the pair, the fleet's
MOVABLE `kv_cache_dtype` for a node the LB requests), null = the profile's own. Whether a value can run is not the
LB's opinion: it is what the ENGINE IMAGE accepts, read in that image's own source and recorded here as data with
its source (vllm_lb/kv_cache_dtypes.json), in the style of the hardware types' {value, source}:

  layouts   dtype → bytes one element costs in the pool (`bytes_per_element`) + source. NVFP4 is 9/16 B: packed FP4
            data and one FP8 scale per 16 elements (vLLM's own page width). What the memory model scales a figure
            measured at one dtype by (`ratio`) — `catalog.MemoryFootprint.at`.
  images    image ref → engine, build, source, and
              dtypes    the dtypes it ACCEPTS: {source, arch?} — `arch`, a regex over the hardware type's `arch` fact,
                        where the engine runs that dtype only on some cards (vLLM's NVFP4 pages: trtllm-gen on sm_10x,
                        and from 2eb491e FA2 on 8x/9x/12x);
              refused   dtypes the engine names but this launch cannot run, each with why in the engine's own terms
                        (SGLang's FP4 pool asserts an attention backend tp-rank does not launch).

One reader (`library`), which REFUSES the file on anything missing or unknown — no default, no guessed layout. A dtype
an image does not accept, an image nobody recorded, or a card outside a dtype's arch is refused where it is given
(`problem`, the switch and the fleet's validator) and disabled with the same words on the page (`choices`). A dtype
the engine accepts whose kernels never ran here is ALLOWED — the operator's choice — and its source says
"not measured"."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

LIBRARY_PATH = Path(__file__).resolve().with_name("kv_cache_dtypes.json")
ENGINES = ("vllm", "sglang", "tensorfold")
_NAME = re.compile(r"[a-z0-9_]{1,32}")


@dataclass(frozen=True)
class Layout:
    name: str
    bytes_per_element: float
    source: str


@dataclass(frozen=True)
class Accepted:
    dtype: str
    source: str
    #: The hardware types' `arch` this dtype runs on (fullmatch); None = any card the image runs on.
    arch: re.Pattern | None = None
    #: The attention backends the engine READS this dtype's pool with, {prefill, decode}, where they are not the
    #: launch's own (SGLang's NVFP4 pool: flashinfer prefill + trtllm_mha decode; fp4_mx_block16: triton —
    #: fp4_kv_cache_quant_method.py:766-771, 840-845); None = the launch's. Sent with the dtype, never apart from it.
    attention_backends: Mapping[str, str] | None = None
    #: Full-attention LAYERS' worth of K/V at 1 B an element the pool adds per token beside its own (SGLang's FP4
    #: dequant workspace, pool_configurator.py:416-417); 0 = none. `catalog.ModelSpec.at_kv_cache_dtype` needs the
    #: model's layer shapes to price it, and refuses the dtype without them.
    workspace_layers: int = 0
    #: THE SPECULATIVE DRAFTER'S OWN POOL dtype, where it cannot share this one (SGLang --speculative-draft-kv-cache-dtype,
    #: arg_groups/fields/spec.py:144-156): an FP4 drafter pool killed generations 214, 215 and 217. None = the target's.
    draft_kv_cache_dtype: str | None = None
    #: Why this dtype is REFUSED while a speculative drafter runs (depth > 0) — the image's own record, "" = it is not.
    #: SGLang's FP4 pools: generations 214/215/217 died on the drafter, 219 served wrong output (2026-10-22).
    with_drafter: str = ""


@dataclass(frozen=True)
class Image:
    image: str
    engine: str
    build: str
    source: str
    accepted: Mapping[str, Accepted]
    refused: Mapping[str, str]


@dataclass(frozen=True)
class Library:
    layouts: Mapping[str, Layout]
    images: Mapping[str, Image]

    @property
    def vocabulary(self) -> tuple[str, ...]:
        """Every dtype the page may offer: the ones with a layout, then the ones an image names only to refuse."""
        refused = [d for i in self.images.values() for d in i.refused if d not in self.layouts]
        return (*self.layouts, *dict.fromkeys(refused))

    def ratio(self, dtype: str, measured: str) -> float:
        """What a figure measured at `measured` is multiplied by at `dtype`: their bytes per element."""
        return self.layouts[dtype].bytes_per_element / self.layouts[measured].bytes_per_element

    def problem(self, image: str, dtype: Any, archs=(), refuses: Mapping[str, str] | None = None,
                drafting: bool = False, backend: Mapping | None = None) -> str:
        """Why `dtype` cannot run on `image` on cards of `archs` (the serving cards' `arch` facts, None where a type
        states none) — "" when it can. `refuses`: a profile's own refusals (`LaunchProfile.kv_cache_dtype_refuses`,
        a model × kernel fact the image cannot know). `drafting`: a speculative drafter runs beside it (depth > 0).
        `backend`: the MODEL's own attention backend (`LaunchProfile.kv_cache_dtype_backend` — name, the dtypes it reads,
        source): it alone reads the pool, so the image's backend rules (arch) are not this model's, and a dtype it does
        not read is refused even where the image takes it."""
        if not isinstance(dtype, str) or dtype not in self.vocabulary:
            return f"kv_cache_dtype must be one of {', '.join(self.vocabulary)}, or null for the profile's own"
        entry = self.images.get(image)
        if entry is None:
            return (f"no record of which KV cache dtypes {image} accepts (vllm_lb/kv_cache_dtypes.json): leave "
                    f"kv_cache_dtype null for the profile's own, or record the image from its source")
        if dtype in entry.refused:
            return f"{dtype} on {image}: {entry.refused[dtype]}"
        if refuses and dtype in refuses:
            return f"{dtype}: {refuses[dtype]}"
        took = entry.accepted.get(dtype)
        if took is None:
            return (f"{image} ({entry.build}) does not accept kv_cache_dtype {dtype}: it takes "
                    f"{', '.join(entry.accepted)} ({entry.source})")
        if backend is not None:
            if dtype not in backend["reads"]:
                return (f"{dtype}: this model's attention is {backend['name']}, which reads "
                        f"{', '.join(backend['reads'])} only ({backend['source']})")
            return f"{dtype} on {image}: {took.with_drafter}" if drafting and took.with_drafter else ""
        if took.arch is not None:
            archs = tuple(archs)
            if not archs or any(a is None for a in archs):
                return (f"{dtype} on {image} runs only on a card whose arch is {took.arch.pattern} ({took.source}), "
                        f"and this server's card states no arch")
            off = [a for a in archs if not took.arch.fullmatch(a)]
            if off:
                return (f"{dtype} on {image} runs only on a card whose arch is {took.arch.pattern} ({took.source}); "
                        f"this server's is {', '.join(dict.fromkeys(off))}")
        if drafting and took.with_drafter:
            return f"{dtype} on {image}: {took.with_drafter}"
        return ""

    def accepted(self, image: str, dtype: str) -> Accepted | None:
        """The image's record of `dtype` (its source, arch, attention backend, workspace) — None where it has none."""
        entry = self.images.get(image)
        return entry.accepted.get(dtype) if entry is not None else None

    def choices(self, image: str, archs=(), refuses: Mapping[str, str] | None = None,
                backend: Mapping | None = None) -> list[dict]:
        """The page's options for one model on one server: every dtype of the vocabulary, `refused` "" where it runs
        (with the image's `source` for it) and otherwise the switch's own words."""
        entry = self.images.get(image)
        out = []
        for dtype in self.vocabulary:
            took = entry.accepted.get(dtype) if entry is not None else None
            out.append({"dtype": dtype, "refused": self.problem(image, dtype, archs, refuses, backend=backend),
                        "source": took.source if took is not None else "",
                        "attention_backends": (dict(took.attention_backends)
                                               if took is not None and took.attention_backends else None)})
        return out


def _keys(where: str, d: Any, required, optional=()) -> dict:
    if not isinstance(d, dict):
        raise ValueError(f"{where} must be a mapping")
    missing = [k for k in required if k not in d]
    unknown = [k for k in d if k not in (*required, *optional)]
    if missing or unknown:
        raise ValueError(f"{where}: " + "; ".join(
            ([f"missing {', '.join(missing)}"] if missing else []) + ([f"unknown {', '.join(unknown)}"] if unknown else [])))
    return d


def _mapping(where: str, d: Any) -> dict:
    """A mapping keyed by names the caller checks — its items, or ValueError."""
    if not isinstance(d, dict):
        raise ValueError(f"{where} must be a mapping")
    return d


def _text(where: str, v: Any) -> str:
    if not isinstance(v, str) or not v.strip():
        raise ValueError(f"{where} must be non-empty text")
    return v


def parse(raw: Any, where: str = LIBRARY_PATH.name) -> Library:
    """The library, validated — ValueError naming the key on anything missing, unknown or malformed."""
    _keys(where, raw, ("layouts", "images"))
    layouts = {}
    for name, lay in _mapping(f"{where}.layouts", raw["layouts"]).items():
        w = f"{where}.layouts.{name}"
        if not _NAME.fullmatch(name):
            raise ValueError(f"{w}: a dtype name is [a-z0-9_]")
        _keys(w, lay, ("bytes_per_element", "source"))
        b = lay["bytes_per_element"]
        if not isinstance(b, (int, float)) or isinstance(b, bool) or b <= 0:
            raise ValueError(f"{w}.bytes_per_element must be a positive number")
        layouts[name] = Layout(name, float(b), _text(f"{w}.source", lay["source"]))
    if not layouts:
        raise ValueError(f"{where}.layouts names no dtype")
    images = {}
    for ref, img in _mapping(f"{where}.images", raw["images"]).items():
        w = f"{where}.images.{ref}"
        _keys(w, img, ("engine", "build", "source", "dtypes", "refused"))
        if img["engine"] not in ENGINES:
            raise ValueError(f"{w}.engine must be one of {', '.join(ENGINES)}")
        accepted = {}
        for dtype, entry in _mapping(f"{w}.dtypes", img["dtypes"]).items():
            if dtype not in layouts:
                raise ValueError(f"{w}.dtypes.{dtype}: no layout — {where}.layouts names {', '.join(layouts)}")
            _keys(f"{w}.dtypes.{dtype}", entry, ("source",),
                  ("arch", "attention_backends", "workspace_layers", "draft_kv_cache_dtype", "with_drafter"))
            arch = entry.get("arch")
            if arch is not None:
                try:
                    arch = re.compile(_text(f"{w}.dtypes.{dtype}.arch", arch))
                except re.error as exc:
                    raise ValueError(f"{w}.dtypes.{dtype}.arch is no regex: {exc}") from None
            backend = entry.get("attention_backends")
            if backend is not None:
                _keys(f"{w}.dtypes.{dtype}.attention_backends", backend, ("prefill", "decode"))
                if not all(isinstance(v, str) and _NAME.fullmatch(v) for v in backend.values()):
                    raise ValueError(f"{w}.dtypes.{dtype}.attention_backends: each a backend name [a-z0-9_]")
                backend = MappingProxyType(dict(backend))
            ws = entry.get("workspace_layers", 0)
            if not isinstance(ws, int) or isinstance(ws, bool) or ws < 0:
                raise ValueError(f"{w}.dtypes.{dtype}.workspace_layers must be a whole number from 0")
            draft = entry.get("draft_kv_cache_dtype")
            if draft is not None and (draft not in layouts or layouts[draft].bytes_per_element < 1):
                raise ValueError(f"{w}.dtypes.{dtype}.draft_kv_cache_dtype must be a whole-byte layout "
                                 f"({', '.join(k for k, v in layouts.items() if v.bytes_per_element >= 1)})")
            with_drafter = (_text(f"{w}.dtypes.{dtype}.with_drafter", entry["with_drafter"])
                            if "with_drafter" in entry else "")
            accepted[dtype] = Accepted(dtype, _text(f"{w}.dtypes.{dtype}.source", entry["source"]), arch, backend, ws,
                                       draft, with_drafter)
        if not accepted:
            raise ValueError(f"{w}.dtypes names no dtype")
        refused = {}
        for dtype, why in _mapping(f"{w}.refused", img["refused"]).items():
            if not _NAME.fullmatch(dtype) or dtype in accepted:
                raise ValueError(f"{w}.refused.{dtype}: a dtype name, not also accepted")
            refused[dtype] = _text(f"{w}.refused.{dtype}", why)
        images[ref] = Image(ref, img["engine"], _text(f"{w}.build", img["build"]), _text(f"{w}.source", img["source"]),
                            MappingProxyType(accepted), MappingProxyType(refused))
    return Library(MappingProxyType(layouts), MappingProxyType(images))


@lru_cache(maxsize=1)
def library() -> Library:
    """The committed library (LIBRARY_PATH), read once — ValueError naming the file on anything it cannot mean."""
    return parse(json.loads(LIBRARY_PATH.read_text()))
