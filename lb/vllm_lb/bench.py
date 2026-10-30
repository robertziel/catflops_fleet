"""Saved benchmark runs, read off disk and normalised for the page at /benchmarks.

WHERE THE RUNS LIVE. `bench_dir` (LB_BENCH_DIR, default /data/bench) — inside the state volume
docker-compose.yml already bind-mounts from lb/state. That directory is gitignored, survives an
image rebuild and a redeploy, and is writable by the harness on the host, so a result is never
"on a branch": worklog/bench/*.json on bench/seqs-sweep could not be served at all, and the next
rebase would have moved them. Nothing in here is ever written by the LB — this module only reads.

WHAT A RESULT LOOKS LIKE. Two harnesses write results and they do NOT agree on key names:

    worklog/bench/load100.py      synthetic, fixed workload   AGG_TOK_S_overall  curve_tok_s
                                  series sample key `tok_s`    requests_ok
    worklog/bench/load_corpus.py  real 24 h corpus, 2 phases  AGG_GEN_TOK_S      gen_tok_s_curve
                                  series sample key `gen_tok_s`  ok

    worklog/bench/prefill_probe.py  pure prefill, max_tokens 1  PREFILL_TOK_S  TFLOPS_PER_GPU
                                  a FLAT object, no `summary`   PCT_OF_NVFP4_PEAK  wall_s  ok

`view()` flattens all three into ONE shape so the page has a single grammar. Adding a harness
means one more branch here, not a second page.

WHAT A RESULT SAYS ABOUT ITS OWN SETTINGS. Since 2026-09-30 worklog/bench/annotate.py stamps every
result with the configuration it ran under — `summary.config` in the two load shapes, top-level
`config` in the prefill probe's flat one: `{model, max_num_seqs, batched, spec, generation,
workload, desc, source}`. `source` is the provenance and the page shows it: "engine argv, verified at
launch" (read back off the engine's own argv after the launch) or "label (...)" (the older synthetic
series, whose settings were known only from their names). The stamp is the ONLY thing that gives a
run a `workload`, bar one shape that cannot be anything else (a prefill probe): the workloads
measure different things and the page never draws two of them on one chart, so a run is never put
in one by guessing at its name.

WHAT A RESULT DOES NOT SAY. Until worklog/bench/runmeta.py shipped (2026-09-29) a result recorded
`args.model` but NOT max_num_seqs — the one setting the whole sweep varied. It existed only as the
text "seqs24" inside the label. This module therefore carries the number AND where it got it:
`max_num_seqs_source` is "recorded" (the harness asked the LB what the cluster was launched with),
"label" (parsed out of the label, which is a reading of a filename, not a measurement) or
"unknown". The page prints the derived ones differently. A guess that cannot be told from a
measurement is the thing this avoids.

A KILLED RUN IS STILL A RUN. Several files in worklog/bench are from benchmarks that were killed
mid-write, and worklog/bench also holds corpus INPUTS (corpus_v3.json, 6 MB) and a moe_sweep
result (T1.json, a completely different shape). None of them may take the page down, and none may
silently vanish either: a file that cannot be read appears in the index with `ok: false` and the
parser's own message, so the operator sees the truncated run rather than wondering where it went.
"""
from __future__ import annotations

import json
import logging
import pathlib
import re
import time

log = logging.getLogger(__name__)

#: Anything bigger than this is not a benchmark result and is not parsed. The corpus inputs that
#: live beside the results are 4–6 MB and parse to something with no `summary`; a cap keeps a
#: stray checkpoint or a log from turning one page load into a gigabyte of parsing.
MAX_BYTES = 32 * 1024 * 1024

#: Samples sent to the page per run. A 3-minute synthetic run samples every 2 s (≈90 points) and a
#: corpus run every 3 s, so nothing real is near this; it bounds a pathological file.
MAX_SERIES = 4000

#: Per-request records sent to the page. 100–120 per run today.
MAX_REQUESTS = 2000

#: `seqs08`, `seqs16-rep1`, `post-seqs24-rep2`, `real-seqs24` — the concurrency cap the sweep
#: varied, as the label spells it. Leading zeros are the sweep's own padding, not a value.
_SEQS_RE = re.compile(r"(?:^|[^0-9a-z])seqs[-_]?0*([0-9]{1,3})(?![0-9])", re.I)
#: `-rep2` / `_rep2` at the end: which repetition of the same setting this was.
_REP_RE = re.compile(r"[-_]rep0*([0-9]{1,3})$", re.I)


def _num(v):
    """A finite number, or None. JSON gives us None for a statistic over an empty list, and a
    killed run can leave a bare string where a number belongs."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    if v != v or v in (float("inf"), float("-inf")):   # NaN / inf: json.loads accepts both
        return None
    return v


def _first(d: dict, *names):
    """The first of these keys that carries a number — how the two harnesses' different names for
    the same quantity are read without asking which harness wrote the file."""
    for name in names:
        v = _num(d.get(name))
        if v is not None:
            return v
    return None


def _stats(block) -> dict:
    """A stat block from either harness, under ONE set of names. load100 writes
    `{mean, median, p10_min10pct, p90_top10pct, min, max}` for the curve but bare `{mean, p10, p90}`
    for per-request and TTFT; load_corpus writes the long names everywhere."""
    if not isinstance(block, dict):
        return {}
    return {"n": _num(block.get("n")), "mean": _num(block.get("mean")),
            "median": _num(block.get("median")),
            "p10": _first(block, "p10_min10pct", "p10"),
            "p90": _first(block, "p90_top10pct", "p90"),
            "min": _num(block.get("min")), "max": _num(block.get("max"))}


def seqs_from_label(label: str):
    """The max_num_seqs a label claims, or None. This is a READING OF A NAME: whatever it returns
    is reported to the page as `source: "label"`, never as if the harness had recorded it."""
    m = _SEQS_RE.search(label or "")
    if not m:
        return None
    n = int(m.group(1))
    return n if 1 <= n <= 999 else None


def rep_from_label(label: str):
    m = _REP_RE.search(label or "")
    return int(m.group(1)) if m else None


def group_of(label: str) -> str:
    """The sweep cell this run belongs to: the label without its repetition suffix, so three reps
    of `post-seqs24` average together and `seqs24-boot` stays its own thing (it is not a rep — it
    was the first run after a boot and it measures a cold cluster)."""
    return _REP_RE.sub("", label or "")


#: The keys only load_corpus.py writes. `prefix_hit_rate` is NOT one of them any more: load100.py has
#: written it too since 2026-09-30, and reading it as "corpus" put every real-prompt decode run in
#: the wrong harness.
_CORPUS_KEYS = ("PHASE2_SERVE_WALL_S", "PHASE1_FILL_WALL_S", "AGG_GEN_TOK_S", "gen_tok_s_curve")


def _text(v, limit: int = 200):
    """A non-empty string, trimmed, or None — for the free-text fields of a stamp."""
    if not isinstance(v, str) or not v.strip():
        return None
    return v.strip()[:limit]


def config_source(text) -> str:
    """Where a stamp says its settings came from, as one word the page can mark by:
    "verified" (read back off the engine's own argv at launch), "label" (read out of the run's
    name — the older synthetic series), "stated" (a stamp that does not say which), "unknown"."""
    t = (_text(text) or "").lower()
    if t.startswith("engine argv"):
        return "verified"
    if t.startswith("label"):
        return "label"
    return "stated" if t else "unknown"


def _stamp(data: dict, summary) -> dict:
    """The configuration a result was stamped with (annotate.py): `summary.config` in the load
    shapes, a top-level `config` in the prefill probe's flat one. {} when there is none — an older
    file, or one written after the last annotate.py pass."""
    for holder in (summary, data):
        c = holder.get("config") if isinstance(holder, dict) else None
        if isinstance(c, dict):
            return c
    return {}


def _positive(v):
    n = _num(v)
    return n if n is not None and n > 0 else None


def _settings(stamp: dict, cluster: dict, label: str) -> dict:
    """max_num_seqs, spec, batched and generation, each with where it came from.

    Precedence, strongest first: a stamp VERIFIED off the engine's argv; what runmeta.py recorded
    by asking the LB; a stamp that itself says it came from the label; the label, parsed here. The
    last two are the same kind of evidence — a reading of a name — and are marked the same way.
    """
    src = config_source(stamp.get("source")) if stamp else "unknown"
    s_seqs = _positive(stamp.get("max_num_seqs"))
    recorded = _positive(cluster.get("max_num_seqs"))
    if s_seqs is not None and src == "verified":
        seqs, seqs_src = s_seqs, "verified"
    elif recorded is not None:
        seqs, seqs_src = recorded, "recorded"
    elif s_seqs is not None:
        seqs, seqs_src = s_seqs, src
    else:
        seqs = seqs_from_label(label)
        seqs_src = "label" if seqs is not None else "unknown"
    spec, batched = _num(stamp.get("spec")), _positive(stamp.get("batched"))
    generation = _num(cluster.get("generation"))
    if generation is None:
        generation = _num(stamp.get("generation"))
    return {"max_num_seqs": seqs, "max_num_seqs_source": seqs_src,
            "spec": spec if spec is not None and spec >= 0 else None, "batched": batched,
            # The provenance of spec / batched / the stamp's workload. None of those has any other
            # source: a file with no stamp does not know them.
            "config_source": src if (spec is not None or batched is not None) else "unknown",
            "config_text": _text(stamp.get("source")), "generation": generation}


class BenchStore:
    """Every saved run in one directory. Parsed results are cached on (mtime, size), so the index
    re-reads only what changed — a poll every few seconds must not re-parse 14 files, and the 6 MB
    corpus inputs sitting in the same directory must not be parsed twice at all.
    """

    def __init__(self, directory: str) -> None:
        self.dir = pathlib.Path(directory)
        #: name -> (mtime, size, view)
        self._cache: dict[str, tuple[float, int, dict]] = {}
        #: Why there is nothing to show, when the reason is the directory itself rather than its
        #: contents. An empty page with no explanation is the failure mode this avoids.
        self.error = ""

    # ── reading ────────────────────────────────────────────────────────────

    def _files(self) -> list[pathlib.Path]:
        try:
            found = sorted(p for p in self.dir.iterdir() if p.is_file() and p.suffix == ".json")
        except FileNotFoundError:
            self.error = f"no results directory at {self.dir} yet — nothing has been saved there"
            return []
        except OSError as exc:
            self.error = f"cannot read {self.dir}: {exc}"
            return []
        self.error = ""
        return found

    def path_of(self, run_id: str) -> pathlib.Path | None:
        """The file a run id names, or None.

        Containment is checked AFTER resolution, exactly as proxy.static_file does it: this id
        arrives in a query string on an internet-facing endpoint, so `..`, an absolute path, a
        symlink out of the directory and a 300-byte segment all have to end at the same refusal.
        """
        if not run_id or "\x00" in run_id or "/" in run_id or "\\" in run_id:
            return None
        try:
            target = (self.dir / f"{run_id}.json").resolve()
            target.relative_to(self.dir.resolve())
            if not target.is_file():
                return None
        except (ValueError, OSError, RuntimeError):
            return None
        return target

    def _load(self, path: pathlib.Path) -> dict:
        """The raw payload, or a `_broken` marker naming what went wrong. Never raises: a file
        written by a killed benchmark is the normal case here, not an exception."""
        try:
            size = path.stat().st_size
        except OSError as exc:
            return {"_broken": f"cannot stat: {exc}"}
        if size > MAX_BYTES:
            return {"_broken": f"{size / 1e6:.0f} MB — too large to be a benchmark result"}
        try:
            with path.open("rb") as fh:
                data = json.loads(fh.read().decode("utf-8", "replace"))
        except json.JSONDecodeError as exc:
            # The commonest failure by far: the harness was killed between `json.dump` opening the
            # file and finishing it. Quote the parser, with the position — that is what tells the
            # operator the run died rather than that the page is broken.
            return {"_broken": f"truncated or malformed JSON: {exc}"}
        except OSError as exc:
            return {"_broken": f"cannot read: {exc}"}
        if not isinstance(data, dict):
            # Valid JSON that is not a result at all (a bare list). Not a killed run — a different
            # kind of file — so it belongs with the corpus inputs, not among the runs.
            return {"_broken": "top level is not an object — not a benchmark result",
                    "_not_a_result": True}
        return data

    def view(self, path: pathlib.Path, *, full: bool = False) -> dict:
        """One run, normalised. `full` adds the time series and the per-request records; the index
        never carries them (14 runs × 90 samples × 7 fields is a page nobody needs to download to
        pick one)."""
        try:
            st = path.stat()
            mtime, size = st.st_mtime, st.st_size
        except OSError:
            mtime, size = 0.0, 0
        key = path.name
        if not full:
            hit = self._cache.get(key)
            if hit and hit[0] == mtime and hit[1] == size:
                return hit[2]
        data = self._load(path)
        out = _view(path.stem, data, mtime=mtime, size=size, full=full)
        if not full:
            self._cache[key] = (mtime, size, out)
            if len(self._cache) > 512:                  # bounded; the directory is not
                self._cache.clear()
        return out

    def index(self) -> dict:
        """What the page opens with: every run it may browse, newest first, plus the values its
        filters can offer. `skipped` is what was in the directory and is NOT a run — the corpus
        inputs and the moe_sweep table — named, so the operator is not left wondering."""
        runs, skipped = [], []
        for path in self._files():
            v = self.view(path)
            (skipped if v.get("not_a_result") else runs).append(v)
        runs.sort(key=lambda r: (r.get("mtime") or 0), reverse=True)
        return {"dir": str(self.dir), "error": self.error, "now": time.time(),
                "runs": runs, "skipped": skipped, "facets": facets(runs)}

    def run(self, run_id: str) -> dict | None:
        """One run with everything the graphs need, or None when no such run exists."""
        path = self.path_of(run_id)
        if path is None:
            return None
        return self.view(path, full=True)


# ── normalising one file ───────────────────────────────────────────────────

def _view(stem: str, data: dict, *, mtime: float, size: int, full: bool) -> dict:
    """The one shape the page knows, out of whichever harness wrote this file."""
    base = {"id": stem, "file": f"{stem}.json", "mtime": mtime, "size": size}
    broken = data.get("_broken")
    if broken and data.get("_not_a_result"):
        return {**base, "ok": False, "not_a_result": True, "error": broken, "label": stem}
    if broken:
        # Still a row on the page: a killed run that disappears looks like a page bug.
        return {**base, "ok": False, "error": broken, "label": stem,
                "kind": "synthetic" if seqs_from_label(stem) else "unknown",
                "max_num_seqs": seqs_from_label(stem),
                "max_num_seqs_source": "label" if seqs_from_label(stem) else "unknown",
                "workload": None, "workload_source": "unknown", "spec": None, "batched": None,
                "config_source": "unknown",
                "group": group_of(stem), "rep": rep_from_label(stem)}
    summary = data.get("summary")
    if not isinstance(summary, dict):
        if "PREFILL_TOK_S" in data:
            # prefill_probe.py's FLAT result: a run like any other, just not a load run.
            return _prefill_view(base, stem, data, full=full)
        # A corpus input (corpus_v3.json) or another tool's output (T1.json: `test`/`args`/`rows`).
        # Not an error and not a run — the directory is shared with the harness's own inputs.
        return {**base, "ok": False, "not_a_result": True,
                "error": "no `summary` block — not a benchmark result", "label": stem}

    args = summary.get("args") if isinstance(summary.get("args"), dict) else {}
    # `run` is what runmeta.py records: what the CLUSTER was actually launched with, asked of the
    # LB at the moment the benchmark started. Its absence is why the label has to be parsed.
    meta = data.get("run") if isinstance(data.get("run"), dict) else {}
    cluster = meta.get("cluster") if isinstance(meta.get("cluster"), dict) else {}
    label = summary.get("label") if isinstance(summary.get("label"), str) else stem
    stamp = _stamp(data, summary)
    settings = _settings(stamp, cluster, label)

    corpus = any(k in summary for k in _CORPUS_KEYS)
    # load100.py --prompts-file sends the file's REAL prompts and never uses --prompt-tokens (whose
    # default, 2000, is still saved in `args`). Such a run is neither synthetic nor 2000 tokens.
    prompts_file = _text(args.get("prompts_file"))
    prompts_file = prompts_file.rsplit("/", 1)[-1] if prompts_file else None
    if prompts_file:
        kind = "real"
    elif meta.get("kind") in ("synthetic", "corpus"):
        kind = meta.get("kind")
    else:
        kind = "corpus" if corpus else "synthetic"
    workload = _text(stamp.get("workload"), 64)

    model = cluster.get("served_model") or _text(stamp.get("model")) or args.get("model")
    model_src = ("recorded" if cluster.get("served_model") else "config" if _text(stamp.get("model"))
                 else "args" if args.get("model") else "unknown")

    curve = _stats(summary.get("curve_tok_s") or summary.get("gen_tok_s_curve"))
    out = {
        **base, "ok": True, "error": "", "label": label,
        "kind": kind, "group": group_of(label), "rep": rep_from_label(label),
        "harness": meta.get("harness") or ("load_corpus.py" if corpus else "load100.py"),
        # ── the settings the page filters by ──
        "model": model if isinstance(model, str) else None, "model_source": model_src,
        "model_choice": cluster.get("model_choice") if isinstance(cluster.get("model_choice"), str) else None,
        **settings,
        # Only a stamp names the workload. A file without one is not put in any comparison.
        "workload": workload, "workload_source": "config" if workload else "unknown",
        "workload_desc": _text(stamp.get("desc")),
        "prompts_file": prompts_file,
        "max_model_len": _num(cluster.get("max_model_len")),
        # What the cluster was DOING when the run started. A run measured against a cluster that was
        # still `starting` is not comparable with one measured against `serving`, and the only honest
        # thing to do is show which it was.
        "cluster_state": cluster.get("state") if isinstance(cluster.get("state"), str) else "",
        "recorded_at": _num(meta.get("at")),
        "concurrency": _num(args.get("concurrency")),
        "prompt_tokens": None if prompts_file else _num(args.get("prompt_tokens")),
        "max_tokens": _num(args.get("max_tokens")),
        "corpus_file": args.get("corpus") if corpus and isinstance(args.get("corpus"), str) else None,
        # ── the numbers ──
        "requests": _first(summary, "requests_total", "requests"),
        "requests_ok": _first(summary, "requests_ok", "ok"),
        "errors": _num(summary.get("errors")),
        "error_sample": summary.get("error_sample") if isinstance(summary.get("error_sample"), str) else None,
        "wall_s": _num(summary.get("TOTAL_WALL_S")),
        "agg_tok_s": _first(summary, "AGG_TOK_S_overall", "AGG_GEN_TOK_S"),
        "engine_tok_s": _first(summary, "engine_agg_tok_s", "engine_gen_tok_s"),
        # Prompt tokens the ENGINE processed per second over the whole run (cache hits included) —
        # what a long-prompt workload is really made of.
        "prompt_tok_s": _num(summary.get("engine_prompt_tok_s")),
        "ms_per_iter": _num(summary.get("engine_ms_per_iter")),
        "output_tokens_total": _num(summary.get("output_tokens_total")),
        "curve": curve,
        "per_request": _stats(summary.get("per_request_tok_s") or summary.get("per_request_decode_tok_s")),
        "ttft": _stats(summary.get("ttft_s") or summary.get("ttft_s_ALL")),
        "preemptions": _num(summary.get("preemptions")),
        "spec_accept": _num(summary.get("spec_accept")),
        "prefix_hit": _first(summary, "prefix_hit_rate", "prefix_hit"),
        "running_mean": _num(summary.get("running_mean")),
        "running_max": _num(summary.get("running_max")),
        "kv_max": _num(summary.get("kv_max")),
    }
    if corpus:
        # The corpus run's own story: filling 262k-token contexts costs minutes before a single
        # warm request is served, and mixing that into one aggregate hides both halves.
        out["phases"] = {
            "fill_s": _num(summary.get("PHASE1_FILL_WALL_S")),
            "serve_s": _num(summary.get("PHASE2_SERVE_WALL_S")),
            "serve_tok_s": _num(summary.get("PHASE2_SERVE_TOK_S")),
            "cold_ttft": _stats(summary.get("FILL_cold_ttft_s")),
            "warm_ttft": _stats(summary.get("WARM_ttft_s")),
            "cold_prefill_tok_s": _stats(summary.get("FILL_cold_prefill_tok_s")),
            "warm_prefill_tok_s": _stats(summary.get("WARM_prefill_tok_s")),
        }
    series = data.get("series") if isinstance(data.get("series"), list) else []
    out["samples"] = len(series)
    if full:
        out["series"] = _series(series)
        out["requests_detail"] = _requests(data.get("requests"))
    return out


def _prefill_view(base: dict, stem: str, data: dict, *, full: bool) -> dict:
    """prefill_probe.py's result: real ~32k-token prompts with max_tokens 1, so everything the
    engine did in the window was prefill. It reads the engine's counters at the two EDGES of the
    run and nothing in between — no time series, no per-request records, no generated throughput.
    Every load-run field it cannot have is None, never 0: a prefill probe did not measure a
    generation rate of zero, it measured none.

    `iterations` / `tokens_per_iter` / `ms_per_iter` in the first probe (B0) are NOT read: the probe
    itself withdrew them — vLLM's iteration histogram only records iterations that emit output,
    which mid-prefill chunks do not (prefill_probe.py, next to where it stopped writing them)."""
    stamp = _stamp(data, None)
    label = _text(data.get("label")) or stem
    settings = _settings(stamp, {}, label)
    requests, ok = _num(data.get("requests")), _num(data.get("ok"))
    raw = data.get("errors")
    # The probe saves the first two error MESSAGES, not a count; the count is requests - ok.
    messages = [e for e in raw if isinstance(e, str)] if isinstance(raw, list) else []
    if requests is not None and ok is not None:
        errors = max(0, requests - ok)
    elif isinstance(raw, list):
        errors = len(raw)
    else:
        errors = _num(raw)
    workload = _text(stamp.get("workload"), 64)
    model = _text(stamp.get("model"))
    out = {
        **base, "ok": True, "error": "", "label": label,
        "kind": "prefill", "group": group_of(label), "rep": rep_from_label(label),
        "harness": "prefill_probe.py",
        "model": model, "model_source": "config" if model else "unknown", "model_choice": None,
        **settings,
        # The one shape that names its own workload: nothing but the probe writes PREFILL_TOK_S.
        "workload": workload or "prefill-probe", "workload_source": "config" if workload else "shape",
        "workload_desc": _text(stamp.get("desc")),
        "prompts_file": None, "max_model_len": None, "cluster_state": "", "recorded_at": None,
        # The probe's own arguments are not saved in its result, so none is claimed here.
        "concurrency": None, "prompt_tokens": None, "max_tokens": None, "corpus_file": None,
        "requests": requests, "requests_ok": ok, "errors": errors,
        "error_sample": messages[0][:300] if messages else None,
        "wall_s": _num(data.get("wall_s")),
        "agg_tok_s": None, "engine_tok_s": None, "prompt_tok_s": None, "ms_per_iter": None,
        "output_tokens_total": None, "curve": {}, "per_request": {}, "ttft": {},
        "preemptions": _num(data.get("preemptions")), "spec_accept": None, "prefix_hit": None,
        "running_mean": None, "running_max": None, "kv_max": None,
        # ── the probe's own numbers ──
        "prefill_tok_s": _num(data.get("PREFILL_TOK_S")),
        "tflops_per_gpu": _num(data.get("TFLOPS_PER_GPU")),
        "gbps_per_gpu": _num(data.get("GBPS_PER_GPU")),
        "pct_nvfp4_peak": _num(data.get("PCT_OF_NVFP4_PEAK")),
        "prompt_tokens_total": _num(data.get("prompt_tokens")),
        "prefix_hit_tokens": _num(data.get("prefix_hit_tokens")),
        "computed_tokens": _num(data.get("computed_tokens")),
        "samples": 0,
    }
    if full:
        out["series"] = []
        out["requests_detail"] = []
    return out


def _series(series: list) -> list[dict]:
    """The throughput-over-time samples, under ONE set of names. load100 calls the generated rate
    `tok_s`; load_corpus calls it `gen_tok_s` and adds `prompt_tok_s`. A sample that carries no
    rate at all stays in the list with `gen_tok_s: null` — it is a moment nobody measured, and the
    graph draws a gap there rather than a zero (the gateway's own rule, gateway.js `sumSeries`)."""
    out = []
    for s in series[:MAX_SERIES]:
        if not isinstance(s, dict):
            continue
        out.append({"t": _num(s.get("t")),
                    "gen_tok_s": _first(s, "gen_tok_s", "tok_s"),
                    "prompt_tok_s": _num(s.get("prompt_tok_s")),
                    "ms_per_iter": _num(s.get("ms_per_iter")),
                    "running": _num(s.get("running")), "waiting": _num(s.get("waiting")),
                    "kv": _num(s.get("kv"))})
    return out


def _requests(requests) -> list[dict]:
    """Per-request records, slimmed to what the page draws: when it ran, what it waited, what it
    got. An errored record keeps its message and no numbers."""
    if not isinstance(requests, list):
        return []
    out = []
    for r in requests[:MAX_REQUESTS]:
        if not isinstance(r, dict):
            continue
        rec = {"idx": _num(r.get("idx")), "start_rel": _num(r.get("start_rel")),
               "end_rel": _num(r.get("end_rel")), "ttft_s": _num(r.get("ttft_s")),
               "e2e_s": _num(r.get("e2e_s")), "out_tokens": _num(r.get("out_tokens")),
               "prompt_tokens": _num(r.get("prompt_tokens")),
               "decode_tok_s": _num(r.get("decode_tok_s"))}
        if isinstance(r.get("error"), str):
            rec["error"] = r["error"][:200]
        if r.get("cold_fill") is not None:
            rec["cold_fill"] = bool(r.get("cold_fill"))
        out.append(rec)
    return out


def facets(runs: list[dict]) -> dict:
    """The values the page's filters may offer — taken from the runs that are actually there, so a
    filter can never select an empty set, and a new model appears in the picker the moment its
    first run is saved. No list of models is hard-coded anywhere on the page."""
    def seen(key, numeric=False):
        vals = {r.get(key) for r in runs if r.get(key) is not None}
        return sorted(vals) if numeric else sorted(str(v) for v in vals)

    return {"models": seen("model"), "kinds": seen("kind"),
            # The stamped workloads, which the page never mixes on one chart, and the two settings
            # beside the concurrency cap that the comparison draws as its series.
            "workloads": seen("workload"),
            "spec": seen("spec", numeric=True), "batched": seen("batched", numeric=True),
            "max_num_seqs": seen("max_num_seqs", numeric=True),
            "concurrency": seen("concurrency", numeric=True),
            "prompt_tokens": seen("prompt_tokens", numeric=True),
            "max_tokens": seen("max_tokens", numeric=True),
            "groups": seen("group")}
