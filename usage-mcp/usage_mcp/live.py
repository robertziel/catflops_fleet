"""`GET /lb/status` and `GET /lb/bench`, reduced to the fields an answer needs.

`/lb/status` is a 58 KB live document; handing it to a model whole is worse than curl. Every field
below is picked **by name** - an allowlist, not a denylist - so a value the LB starts publishing
later cannot leak through a tool result just by appearing in the document, and every URL goes
through `scrub_url` on the way out. `tests/test_secrets.py` proves that by salting a captured real
status with credentials at several depths and asserting no tool result carries them.
"""
from __future__ import annotations

from typing import Any

from . import caveats
from .readonly import Sources, lb_get, scrub_url

BYTES_PER_GIB = 1024 ** 3


def status(src: Sources) -> dict[str, Any]:
    return lb_get("/lb/status", src)


def bench(src: Sources) -> dict[str, Any]:
    return lb_get("/lb/bench", src)


def engines_from_status(doc: dict[str, Any]) -> dict[str, str]:
    """model choice -> engine, as the LB's own models view reports it (missing key means vLLM)."""
    out: dict[str, str] = {}
    for entry in (doc.get("models") or {}).get("cluster") or ():
        if not isinstance(entry, dict) or not entry.get("id"):
            continue
        launch = ((entry.get("launch") or {}).get("cluster") or {})
        out[str(entry["id"])] = str(launch.get("engine") or "vllm")
    return out


def shape_status(doc: dict[str, Any]) -> dict[str, Any]:
    """The live snapshot: what the pair is serving, how fast, and on what hardware."""
    cluster = doc.get("cluster") or {}
    state = _s(cluster.get("state")) or "unknown"
    loaded = cluster.get("loaded_model") or None
    effective = cluster.get("effective") or {}
    serving = state == "running" and bool(loaded)
    engines = engines_from_status(doc)
    choice = _s(effective.get("model_choice")) or (_s(loaded.get("spec")) if loaded else None)

    out: dict[str, Any] = {
        "serving": serving,
        "cluster_state": state,
        "detail": _s(cluster.get("detail")),
        "since": cluster.get("since") if isinstance(cluster.get("since"), (int, float)) else None,
        "held": bool(cluster.get("held")),
        "url": scrub_url(cluster.get("url")),
        "model": None,
        "configuration": None,
        "rate": None,
        "boxes": {},
        "backends": [],
        "saturation": None,
        "swarm": _shape_swarm(doc.get("swarm") or {}),
        "lb": {
            "uptime_s": _n(doc.get("uptime_s")),
            "requests_total": _n(doc.get("requests_total")),
            "db_ok": bool(doc.get("db_ok")) if doc.get("db_ok") is not None else None,
            "db_error": _s(doc.get("db_error")) or None,
            "last_request_at": _n(doc.get("last_request_at")),
            "last_tick_error": _s(doc.get("last_tick_error")) or None,
        },
        "caveats": [],
    }

    if loaded:
        out["model"] = {
            "served_name": _s(loaded.get("served_name")),
            "model_choice": choice,
            "label": _s(loaded.get("label")),
            "engine": engines.get(choice or "", "unknown"),
            "max_model_len": _n(loaded.get("max_model_len")),
            "generation": _n(effective.get("generation")),
        }
        out["configuration"] = _shape_config(effective)

    target = doc.get("throughput_target") or {}
    if target:
        out["rate"] = {
            "gen_tok_s": _n(target.get("gen_tok_s")),
            "prompt_tok_s": _n(target.get("prompt_tok_s")),
            "prompt_computed_tok_s": _n(target.get("computed_tok_s")),
            "windows": _n(target.get("windows")),
            "basis": _s(target.get("basis")),
            "means": "the best recorded minutes for THIS configuration, not the current minute",
        }

    for box, hw in sorted((doc.get("hardware") or {}).items()):
        if not isinstance(hw, dict):
            continue
        out["boxes"][str(box)] = {
            "gpu_pct": _n(hw.get("gpu_pct")),
            "ram_used_gib": _r(_n(hw.get("ram_used_mb")), 1024.0),
            "ram_total_gib": _r(_n(hw.get("ram_total_mb")), 1024.0),
            "cpu_percent": _n(hw.get("cpu_percent")),
            "age_s": _n(hw.get("age_s")),
            "stale": bool(hw.get("stale")),
        }
    for box, cb in sorted((cluster.get("boxes") or {}).items()):
        if isinstance(cb, dict) and str(box) in out["boxes"]:
            out["boxes"][str(box)].update({
                "rank_state": _s(cb.get("state")) or None,
                "power_w": _n(cb.get("power_w")) or _n(cb.get("pwr")),
            })

    for be in doc.get("backends") or ():
        if not isinstance(be, dict):
            continue
        out["backends"].append({
            "box": _s(be.get("box")),
            "url": scrub_url(be.get("url")),
            "live": bool(be.get("live")),
            "running": _n(be.get("running")),
            "waiting": _n(be.get("waiting")),
            "gen_tokens_total": _n(be.get("gen_tokens_total")),
            "prompt_tokens_total": _n(be.get("prompt_tokens_total")),
            "tokens_per_s": _n(be.get("tokens_per_s")),
            "loaded_state": _s((be.get("loaded") or {}).get("state")) or None,
        })

    sat = doc.get("saturation") or {}
    if sat:
        out["saturation"] = {
            "verdict": _s(sat.get("verdict")),
            "because": _s(sat.get("because")),
            "sm_fill_pct": _n(sat.get("sm_fill_pct")),
            "sm_fill_means": _s(sat.get("sm_fill_means")),
            "compute_tflops_per_gpu": _n((sat.get("compute") or {}).get("rate")),
            "bandwidth_gbps_per_gpu": _n((sat.get("bandwidth") or {}).get("rate")),
            "rates_means": _s(sat.get("rates_means")),
        }
        out["caveats"].append(caveats.MFU_PHANTOM)

    if not serving:
        out["caveats"].insert(0, "Nothing is loaded on the pair: the cluster state is "
                                f"{state!r}. That is an answer, not a failure.")
    return out


def _shape_config(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "max_num_seqs": _n(cfg.get("max_num_seqs")),
        "max_num_batched_tokens": _n(cfg.get("max_num_batched_tokens")),
        "num_speculative_tokens": _n(cfg.get("num_speculative_tokens")),
        "max_model_len": _n(cfg.get("max_model_len")),
        "kv_cache_memory_gib": _r(_n(cfg.get("kv_cache_memory")), BYTES_PER_GIB),
        "kv_offload_gib": _n(cfg.get("kv_offload_gib")),
        "ple_replicate": cfg.get("ple_replicate"),
        "long_prefill_token_threshold": _n(cfg.get("long_prefill_token_threshold")),
    }


def _shape_swarm(sw: dict[str, Any]) -> dict[str, Any]:
    runner = sw.get("runner") or {}
    session = runner.get("session") or {}
    return {
        "configured": bool(sw.get("configured")),
        "runner_listening": bool(runner.get("listening")),
        "cap_min": _n(sw.get("cap_min")),
        "request_pending": bool(sw.get("request_pending")),
        "last_session": {
            "label": _s(session.get("label")) or None,
            "phase": _s(session.get("phase")) or None,
            "alive": bool(session.get("alive")),
            "rc": _n(session.get("rc")),
            "started_at": _n(session.get("started_at")),
            "ended_at": _n(session.get("ended_at")),
            "served_model": _s(session.get("served_model")) or None,
            "model_choice": _s(session.get("model_choice")) or None,
            "generation": _n(session.get("generation")),
            "detail": _s(session.get("detail")) or None,
            "guards": {k: bool(v) for k, v in (session.get("guards") or {}).items()},
        } if session else None,
    }


def shape_bench(doc: dict[str, Any], limit: int = 20) -> dict[str, Any]:
    """Saved benchmark runs, one compact row each. An empty directory is a legitimate answer."""
    runs = [r for r in (doc.get("runs") or ()) if isinstance(r, dict)]
    rows = []
    for run in runs[: max(1, int(limit))]:
        cfg = run.get("config") or {}
        rows.append({
            "name": _s(run.get("name")) or _s(run.get("id")),
            "model": _s(run.get("model")) or _s(cfg.get("model_choice")),
            "kind": _s(run.get("kind")),
            "workload": _s(run.get("workload")),
            "concurrency": _n(run.get("concurrency")),
            "max_num_seqs": _n(cfg.get("max_num_seqs")) or _n(run.get("max_num_seqs")),
            "max_num_batched_tokens": _n(cfg.get("max_num_batched_tokens")) or _n(run.get("batched")),
            "num_speculative_tokens": _n(cfg.get("num_speculative_tokens")) or _n(run.get("spec")),
            "gen_tok_s": _n(run.get("gen_tok_s")) or _n(run.get("tok_s")),
            "at": _n(run.get("at")) or _n(run.get("t_start")),
        })
    facets = {k: v for k, v in (doc.get("facets") or {}).items() if isinstance(v, list) and v}
    return {
        "dir": _s(doc.get("dir")),
        "saved_runs": len(runs),
        "shown": len(rows),
        "runs": rows,
        "facets": facets,
        "skipped": len(doc.get("skipped") or ()),
        "error": _s(doc.get("error")) or None,
    }


def _s(v: Any) -> str:
    return v if isinstance(v, str) else ""


def _n(v: Any) -> Any:
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _r(v: Any, divisor: float) -> Any:
    return None if v is None else round(v / divisor, 2)
