"""What configuration is *recorded* for a model, as opposed to what happens to be running.

Two sources, and the difference between them is usually the question:

* `vllm_lb/library` (mounted read-only) - the fleet's own record, split three ways since 2026-10-30:
  models.json (what each checkpoint is: limits, quant, architectures), recipes.json (the engine, its
  build, the launch arguments) and measurements.json (memory, tuned input, and the **calibration**
  line naming the run each figure was measured on, keyed by the engine x accelerator x TP it covers).
  Merged here by model id, as the LB's own `catalog.document` does minus the deployment's folders.
  The IMAGE a profile runs is resolved per deployment (the LB's image matrix): /lb/status has it.
  That is the recorded configuration and it survives the cluster being off.
* `GET /lb/status` -> `cluster.{effective,desired,defaults,recommended,limits}` - what the LB has
  right now. Absent when the LB is down, which is reported rather than raised.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .readonly import Sources, SourceUnavailable, read_json

BYTES_PER_GIB = 1024 ** 3


#: The library's files, each with the list it holds.
LIBRARY_FILES = (("models.json", "models"), ("recipes.json", "recipes"), ("measurements.json", "measurements"))


def load(src: Sources) -> dict[str, Any]:
    """The library merged into one row a model: its facts, its recipe's launch (with the measured tuned input and
    overhead), its node engine, its memory and engine memory, and what that measurement covers."""
    parts = {}
    for name, key in LIBRARY_FILES:
        doc = read_json(Path(src.catalog) / name, src)
        if not isinstance(doc, dict) or not isinstance(doc.get(key), list):
            raise SourceUnavailable(f"{src.catalog} is not a model library: {name} holds no `{key}` list")
        parts[key] = doc[key]
    recipes = {r.get("model"): r for r in parts["recipes"] if isinstance(r, dict)}
    measured = {r.get("model"): r for r in parts["measurements"] if isinstance(r, dict)}
    rows = []
    for m in parts["models"]:
        if not isinstance(m, dict):
            continue
        row = dict(m)
        recipe, meas = recipes.get(m.get("id")) or {}, measured.get(m.get("id")) or {}
        launch = {mode: {**(p or {}), **((meas.get("launch") or {}).get(mode) or {})}
                  for mode, p in (recipe.get("launch") or {}).items()}
        row["launch"] = launch
        for k in ("node_engine",):
            if k in recipe:
                row[k] = recipe[k]
        for k in ("memory", "engine_memory", "covers"):
            if k in meas:
                row[k] = meas[k]
        rows.append(row)
    return {"models": rows}


def engines(doc: dict[str, Any]) -> dict[str, str]:
    """model id -> engine. A catalog entry without `launch.cluster.engine` runs on vLLM, which is
    the same default the LB's own models view fills in."""
    out: dict[str, str] = {}
    for m in doc.get("models") or ():
        if isinstance(m, dict) and m.get("id"):
            out[str(m["id"])] = str((((m.get("launch") or {}).get("cluster")) or {}).get("engine") or "vllm")
    return out


def index(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(m["id"]): m for m in doc.get("models") or () if isinstance(m, dict) and m.get("id")}


def shape_one(entry: dict[str, Any]) -> dict[str, Any]:
    launch = (entry.get("launch") or {}).get("cluster") or {}
    mem = entry.get("memory") or {}
    limits = entry.get("limits") or {}
    spec = launch.get("speculative_config") or {}
    return {
        "id": str(entry.get("id") or ""),
        "label": _s(entry.get("label")),
        "served_name": _s(entry.get("served_name")),
        "engine": str(launch.get("engine") or "vllm"),
        "size_gib": _n(entry.get("size_gib")),
        "is_cluster_default": bool(launch.get("default")),
        "limits": {
            "max_model_len": _n(limits.get("max_model_len")),
            "max_num_seqs": limits.get("max_num_seqs") if isinstance(limits.get("max_num_seqs"), list) else None,
        },
        "launch": {
            "build": _s(launch.get("build")) or None,
            "tool_call_parser": _s(launch.get("tool_call_parser")) or None,
            "reasoning_parser": _s(launch.get("reasoning_parser")) or None,
            "speculative_method": _s(spec.get("method")) or _s(launch.get("speculative_method")) or None,
            "num_speculative_tokens": _n(spec.get("num_speculative_tokens"))
                                      or _n(launch.get("num_speculative_tokens")),
            "kv_cache_memory_gib": _div(_n(launch.get("kv_cache_memory")), BYTES_PER_GIB),
            "ple_replicate": launch.get("ple_replicate"),
            "engine_args": launch.get("engine_args") if isinstance(launch.get("engine_args"), (list, dict)) else None,
        },
        "memory": {
            "weights_gib": _n(mem.get("weights_gib")),
            "ple_gib": _n(mem.get("ple_gib")),
            "mtp_gib": _n(mem.get("mtp_gib")),
            "kv_bytes_per_token": _n(mem.get("kv_bytes_per_token")),
            "seq_state_gib": _n(mem.get("seq_state_gib")),
            "calibrated_max_num_seqs": _n(mem.get("calibrated_max_num_seqs")),
            "calibrated_batched_tokens": _n(mem.get("calibrated_batched_tokens")),
            "calibration": _s(mem.get("calibration")) or None,
        },
    }


def shape_live(cluster: dict[str, Any] | None) -> dict[str, Any] | None:
    """What the LB holds right now for the cluster, if it answered."""
    if not cluster:
        return None
    loaded = cluster.get("loaded_model") or {}
    return {
        "state": _s(cluster.get("state")) or None,
        "loaded_model_choice": _s(loaded.get("spec")) or None,
        "effective": _plain(cluster.get("effective")),
        "desired": _plain(cluster.get("desired")),
        "defaults": _plain(cluster.get("defaults")),
        "recommended": _plain(cluster.get("recommended")),
        "limits": _plain(cluster.get("limits")),
        "clamp": _s(cluster.get("clamp")) or None,
    }


_LIVE_KEYS = ("mode", "model_choice", "generation", "max_num_seqs", "max_num_batched_tokens",
              "num_speculative_tokens", "max_model_len", "kv_cache_memory", "kv_offload_gib",
              "ple_replicate", "long_prefill_token_threshold", "kv_cache_auto", "kv_tokens",
              "source")


def _plain(block: Any) -> dict[str, Any] | None:
    """Only the configuration keys, by name - never the whole sub-document."""
    if not isinstance(block, dict):
        return None
    out: dict[str, Any] = {}
    for k in _LIVE_KEYS:
        if k in block:
            v = block[k]
            out[k] = v if isinstance(v, (int, float, str, bool, type(None), list)) else None
    return out or None


def _s(v: Any) -> str:
    return v if isinstance(v, str) else ""


def _n(v: Any) -> Any:
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _div(v: Any, d: float) -> Any:
    return None if v is None else round(v / d, 2)
