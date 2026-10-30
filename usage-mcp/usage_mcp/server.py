"""The MCP surface: eight read-only tools over the fleet's own usage data.

Each tool answers a question an agent actually has - "what is the pair serving and how fast", "how
many tokens has this model served today", "what did the last benchmark measure", "which
configuration is recorded for this model" - and returns a shaped answer with a `summary` line
first, not a slab of the source document. Where a number means something engine-specific, the
caveat travels with it.

Every tool is annotated `read_only_hint=True` / `destructive_hint=False` / `open_world_hint=False`,
but the annotation is a label: the guarantee is in `readonly.py` and in the container's `:ro`
mounts, and `tests/test_readonly.py` proves it from this package's AST.
"""
from __future__ import annotations

from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from . import catalog as catalog_mod
from . import caveats, inventory, live, perf, swarm
from .readonly import ReadOnlyViolation, Sources, SourceUnavailable

INSTRUCTIONS = """\
Read-only usage data for the local two-box vLLM/SGLang gateway behind http://127.0.0.1:8010.

These are the LOCAL gateway's numbers: tokens the pair served, per-minute windows the LB recorded,
cluster starts, saved benchmarks and benchmark-swarm sessions. This fleet records NO Anthropic
subscription usage - call usage_data_inventory before answering any question about Claude account
spend, and report that it is not available rather than substituting local tokens for it.

Two numbers mean different things depending on what produced them, and the tools say so in their
own output: ITL is per TOKEN on SGLang but per speculative STEP on vLLM (mixing them is wrong by
~3x), and one benchmark swarm session's throughput is not its configuration's throughput.

Nothing here can change anything: there is no start, stop, park or write tool, and none can be
added without the suite's structural check failing.
"""

READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False,
                            idempotent_hint=True, open_world_hint=False)


def build(src: Sources | None = None) -> MCPServer:
    sources = src or Sources.from_env()
    server = MCPServer(
        name="fleet-usage",
        title="Fleet usage (read-only)",
        version="1.0.0-beta",
        instructions=INSTRUCTIONS,
        warn_on_duplicate_tools=True,
    )

    # ── live ───────────────────────────────────────────────────────────────────────────────────
    @server.tool(
        title="What the pair is serving",
        description="The live snapshot of the two-box gateway: whether anything is loaded, which "
                    "model and generation, the seats/batched-tokens/speculative configuration, the "
                    "recorded throughput for that configuration, per-box GPU and memory, the "
                    "backends, the saturation verdict and the benchmark swarm's state. "
                    "'Nothing is loaded' is a normal answer, not an error.",
        annotations=READ_ONLY,
    )
    def fleet_status() -> dict[str, Any]:
        try:
            shaped = live.shape_status(live.status(sources))
        except (SourceUnavailable, ReadOnlyViolation) as exc:
            return _unavailable(exc, "the LB at %s did not answer; the cluster's state is unknown "
                                     "(historical tools still work from disk)" % sources.lb_url)
        model = shaped.get("model") or {}
        cfg = shaped.get("configuration") or {}
        rate = shaped.get("rate") or {}
        if shaped["serving"]:
            summary = (f"serving {model.get('served_name')} ({model.get('engine')}, generation "
                       f"{model.get('generation')}) - {cfg.get('max_num_seqs')} seats, "
                       f"{cfg.get('max_num_batched_tokens')} batched, spec "
                       f"{cfg.get('num_speculative_tokens')}; recorded {rate.get('gen_tok_s')} "
                       f"gen tok/s over {rate.get('windows')} windows")
        else:
            summary = f"nothing loaded - cluster state {shaped['cluster_state']!r}: {shaped['detail']}"
        return {"available": True, "summary": summary, **shaped}

    # ── tokens ─────────────────────────────────────────────────────────────────────────────────
    @server.tool(
        title="Tokens served",
        description="How many tokens the gateway served, from the LB's per-minute windows on disk. "
                    "`day` takes 'today', 'yesterday' or YYYY-MM-DD; without it, the last `days` "
                    "recorded day-files. `group_by` is one of model, generation, day, hour, config. "
                    "Returns generated and prompt tokens, what the engine actually prefilled, the "
                    "prefix-cache hit rate, speculative acceptance and the rate per group.",
        annotations=READ_ONLY,
    )
    def token_usage(day: str | None = None, days: int = 1, model: str | None = None,
                    group_by: str = "model") -> dict[str, Any]:
        if group_by not in perf.GROUPS:
            return {"available": False, "why": f"group_by must be one of {list(perf.GROUPS)}"}
        try:
            windows, meta = perf.load_windows(sources, day=day, days=days, model=model)
        except (SourceUnavailable, ReadOnlyViolation) as exc:
            return _unavailable(exc)
        if not windows:
            return {"available": True, "summary": "no recorded minutes in that selection - the "
                                                  "gateway served nothing then", "selection": meta,
                    "totals": None, "groups": []}
        whole = perf.fold(windows)
        groups = [
            {"key": key, **f.rates(), **f.totals(), "mean_concurrency": f.mean_concurrency(),
             "kv_pct": f.kv_pct(), "windows": f.windows}
            for key, f in sorted(perf.fold_by(windows, group_by).items(),
                                 key=lambda kv: -kv[1].gen_tokens)
        ]
        unusable = whole.prefix_hit_note()
        return {
            "available": True,
            "summary": (f"{int(whole.gen_tokens):,} generated + {int(whole.prompt_tokens):,} prompt "
                        f"tokens over {whole.rates()['minutes_recorded']} recorded minutes "
                        f"({', '.join(meta['days'])}) at {whole.rates()['gen_tok_s']} gen tok/s"
                        + (" -- no prefix hit rate for this selection, see "
                           "totals.prefix_hit_rate_unusable" if unusable else "")),
            "selection": meta,
            "range": {"from": perf.iso(whole.t_first), "to": perf.iso(whole.t_last)},
            "totals": {**whole.rates(), **whole.totals(),
                       "mean_concurrency": whole.mean_concurrency(), "kv_pct": whole.kv_pct()},
            "group_by": group_by,
            "groups": groups,
            "caveats": [caveats.PREFIX_CACHE, caveats.NOT_ANTHROPIC],
        }

    # ── latency ────────────────────────────────────────────────────────────────────────────────
    @server.tool(
        title="Latency profile",
        description="TTFT and inter-token latency over the chosen windows, folded from the LB's "
                    "histogram buckets. Reports the raw figure, what one ITL sample counts "
                    "(MEASURED from gen_tokens / itl.n: 1.0 means per token, ~3 means per "
                    "speculative step), and the per-token figure after dividing - so a vLLM and an "
                    "SGLang number are never compared raw.",
        annotations=READ_ONLY,
    )
    def latency_profile(day: str | None = None, days: int = 1,
                        model: str | None = None) -> dict[str, Any]:
        try:
            windows, meta = perf.load_windows(sources, day=day, days=days, model=model)
        except (SourceUnavailable, ReadOnlyViolation) as exc:
            return _unavailable(exc)
        if not windows:
            return {"available": True, "summary": "no recorded minutes in that selection",
                    "selection": meta, "overall": None, "by_model": []}
        whole = perf.fold(windows)
        by_model = []
        for key, f in sorted(perf.fold_by(windows, "model").items()):
            lat = f.latency()
            by_model.append({"model_choice": key, "windows": f.windows,
                             "gen_tokens": int(f.gen_tokens), **lat})
        overall = whole.latency()
        sem = overall["itl_semantics"]
        return {
            "available": True,
            "summary": (f"ITL raw mean {overall['itl_raw_s']['mean']} s over "
                        f"{overall['itl_raw_s']['samples']:,} samples; each sample is "
                        f"per {sem['means']} (measured {sem['tokens_per_sample']} tokens per "
                        f"sample) -> {overall['itl_per_token_s']['mean']} s per token"
                        + (f" -- WARNING: {sem['disagreement']}" if sem.get("disagreement") else "")),
            "selection": meta,
            "overall": overall,
            "by_model": by_model,
            "caveats": (["A mixed selection folds engines together: read by_model, not overall, "
                         "when more than one model_choice appears."] if len(by_model) > 1 else [])
                       + (["The LB publishes itl_basis ('token' / 'step') on its scrape since "
                           "9c32807 but does not persist it into a perf window, so the basis above "
                           "is MEASURED from gen_tokens / itl.n, not read from the engine."]
                          if not whole.itl_bases else []),
        }

    # ── starts ─────────────────────────────────────────────────────────────────────────────────
    @server.tool(
        title="Cluster starts",
        description="Every recorded cluster start and stop: the configuration it was asked for, "
                    "how long each phase took, the outcome (serving / failed / stopped) and the "
                    "failure text. Answers 'why is it not up' and 'what changed'. `outcome` "
                    "filters to one kind.",
        annotations=READ_ONLY,
    )
    def cluster_starts(days: int = 7, outcome: str | None = None,
                       limit: int = 25) -> dict[str, Any]:
        try:
            boots, meta = perf.load_boots(sources, days=days, outcome=outcome)
        except (SourceUnavailable, ReadOnlyViolation) as exc:
            return _unavailable(exc)
        tally: dict[str, int] = {}
        for b in boots:
            key = str(b.get("outcome") or "unknown")
            tally[key] = tally.get(key, 0) + 1
        rows = []
        for b in boots[-max(1, int(limit)):][::-1]:
            cfg = b.get("config") or {}
            rows.append({
                "generation": cfg.get("generation") if isinstance(cfg.get("generation"), int)
                              else b.get("generation"),
                "at": perf.iso(b.get("t_start")),
                "model_choice": cfg.get("model_choice"),
                "max_num_seqs": cfg.get("max_num_seqs"),
                "max_num_batched_tokens": cfg.get("max_num_batched_tokens"),
                "num_speculative_tokens": cfg.get("num_speculative_tokens"),
                "max_model_len": cfg.get("max_model_len"),
                "outcome": b.get("outcome"),
                "total_s": b.get("total_s"),
                "states": b.get("states"),
                "failed_in": b.get("failed_in"),
                "failure": b.get("failure"),
            })
        return {
            "available": True,
            "summary": (f"{len(boots)} starts in {meta['files']}: "
                        + ", ".join(f"{n} {k}" for k, n in sorted(tally.items(), key=lambda kv: -kv[1]))),
            "selection": meta,
            "tally": tally,
            "starts": rows,
        }

    # ── recorded configuration ─────────────────────────────────────────────────────────────────
    @server.tool(
        title="Recorded model configuration",
        description="The configuration the fleet records for a model: engine, image, limits, "
                    "launch arguments, and the calibrated memory figures with the run each was "
                    "measured on. Without `model`, lists every catalog entry. Also reports what "
                    "the LB currently holds (effective / desired / defaults / recommended) when "
                    "the LB answers - which it need not for the recorded part.",
        annotations=READ_ONLY,
    )
    def model_configuration(model: str | None = None) -> dict[str, Any]:
        try:
            doc = catalog_mod.load(sources)
        except (SourceUnavailable, ReadOnlyViolation) as exc:
            return _unavailable(exc, f"the model catalog at {sources.catalog} was not readable")
        entries = catalog_mod.index(doc)
        live_block = None
        try:
            live_block = catalog_mod.shape_live((live.status(sources) or {}).get("cluster"))
        except (SourceUnavailable, ReadOnlyViolation):
            live_block = None
        if model is None:
            rows = [{k: v for k, v in catalog_mod.shape_one(e).items()
                     if k in ("id", "label", "engine", "served_name", "size_gib",
                              "is_cluster_default", "limits")}
                    for e in entries.values()]
            loaded = (live_block or {}).get("loaded_model_choice")
            return {"available": True,
                    "summary": f"{len(rows)} catalog entries; loaded now: {loaded or 'none'}",
                    "models": rows, "loaded_model_choice": loaded, "live": live_block,
                    "catalog": str(sources.catalog)}
        if model not in entries:
            return {"available": False,
                    "why": f"{model!r} is not in the catalog",
                    "known": sorted(entries)}
        shaped = catalog_mod.shape_one(entries[model])
        return {
            "available": True,
            "summary": (f"{shaped['id']} on {shaped['engine']}: seats "
                        f"{shaped['limits']['max_num_seqs']}, max_model_len "
                        f"{shaped['limits']['max_model_len']}, "
                        f"{shaped['memory']['kv_bytes_per_token']} KV bytes/token"),
            "recorded": shaped,
            "live": live_block,
            "is_loaded": bool(live_block and live_block.get("loaded_model_choice") == model),
            "caveats": ["The memory figures are calibrated from one named run (see "
                        "recorded.memory.calibration); a different seat count or token budget is "
                        "an extrapolation."],
        }

    # ── saved benchmarks ───────────────────────────────────────────────────────────────────────
    @server.tool(
        title="Saved benchmark runs",
        description="The benchmark runs saved in the LB's bench directory, one compact row each, "
                    "plus the facets the /benchmarks page groups by. An empty directory is "
                    "reported as such, together with the archived run sets that are on disk but "
                    "not loaded.",
        annotations=READ_ONLY,
    )
    def benchmark_runs(limit: int = 20) -> dict[str, Any]:
        shaped, why = None, None
        try:
            shaped = live.shape_bench(live.bench(sources), limit=limit)
        except (SourceUnavailable, ReadOnlyViolation) as exc:
            why = f"{type(exc).__name__}: {exc}"
        archives: list[dict[str, Any]] = []
        try:
            archives = inventory.take(sources)["sources"]["archives"]["bench"]
        except (SourceUnavailable, ReadOnlyViolation):
            archives = []
        if shaped is None:
            return {"available": False, "why": why, "archives": archives}
        n = shaped["saved_runs"]
        extra = sum(a.get("files") or 0 for a in archives)
        return {
            "available": True,
            "summary": (f"{n} saved benchmark run(s) in {shaped['dir']}"
                        + (f"; {extra} more in {len(archives)} archive directory/ies, not loaded"
                           if extra else "")),
            **shaped,
            "archives": archives,
        }

    # ── swarm sessions ─────────────────────────────────────────────────────────────────────────
    @server.tool(
        title="Benchmark swarm sessions",
        description="Each benchmark swarm session (a capped deep-research run against the local "
                    "gateway), with what the LB's own per-minute windows recorded inside its "
                    "bracket: tokens, throughput, mean and max concurrency, acceptance. Carries "
                    "the between-session variance caveat, which is larger than any configuration "
                    "effect measured here.",
        annotations=READ_ONLY,
    )
    def swarm_sessions(limit: int = 10, days: int = 7) -> dict[str, Any]:
        try:
            found = swarm.sessions(sources)
        except (SourceUnavailable, ReadOnlyViolation) as exc:
            return _unavailable(exc)
        rows = []
        for session in found[-max(1, int(limit)):][::-1]:
            measured = swarm.measure(sources, session, days=days)
            rows.append({**session, "started": perf.iso(session.get("started_at")),
                         "ended": perf.iso(session.get("ended_at")), "measured": measured,
                         "client_accounting": swarm.client_accounting(sources, session["label"])})
        worked = [r for r in rows if r["measured"].get("measured")]
        conc = [r["measured"]["mean_concurrency"] for r in worked
                if r["measured"].get("mean_concurrency") is not None]
        return {
            "available": True,
            "summary": (f"{len(found)} session captures, {len(rows)} shown, {len(worked)} with "
                        f"recorded windows"
                        + (f"; mean concurrency {min(conc)}-{max(conc)} across them" if conc else "")),
            "sessions": rows,
            "caveats": [swarm.SPREAD, swarm.CLIENT_COST_IS_FICTION, caveats.NOT_ANTHROPIC],
        }

    # ── what exists ────────────────────────────────────────────────────────────────────────────
    @server.tool(
        title="What usage data exists",
        description="An inventory of the usage data this fleet holds, counted off disk: which "
                    "files, which days, how much, and what each one contains. Call this before "
                    "answering any question about Anthropic or Claude-subscription spend: that "
                    "data is NOT recorded here and this tool says so explicitly.",
        annotations=READ_ONLY,
    )
    def usage_data_inventory() -> dict[str, Any]:
        try:
            take = inventory.take(sources)
        except (SourceUnavailable, ReadOnlyViolation) as exc:
            return _unavailable(exc)
        perf_group = take["sources"]["perf_windows"]
        return {
            "available": True,
            "summary": (f"local gateway usage only: {perf_group['files']} perf day-file(s) "
                        f"({', '.join(perf_group['days']) or 'none'}), "
                        f"{take['sources']['cluster_boots']['files']} boot log(s), "
                        f"{take['sources']['saved_benchmarks']['files']} saved benchmark file(s), "
                        f"{take['sources']['swarm_sessions']['count']} swarm session capture(s). "
                        "No Anthropic-account usage is recorded anywhere."),
            **take,
        }

    return server


def _unavailable(exc: Exception, extra: str | None = None) -> dict[str, Any]:
    """A source that did not answer is a sentence, not a tool error."""
    return {
        "available": False,
        "why": f"{type(exc).__name__}: {exc}",
        "note": extra or "nothing was changed; this server only reads",
    }
