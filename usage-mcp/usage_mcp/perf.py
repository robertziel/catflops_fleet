"""The per-minute windows and the cluster starts, read off disk and folded into answers.

The LB's recorder closes one window a minute into `<state>/perf/perf-<day>.jsonl` and appends one
record per cluster start to `boots-<day>.jsonl`. Those files are the durable record - `GET /lb/perf`
returns the same rows but 3.5 MB of them at a time, which is the thing this server exists not to
hand a model. Everything here reads the files directly (mounted read-only) and returns totals.
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from . import caveats
from .readonly import Sources, SourceUnavailable, jsonl, state_glob

PERF_GLOB = "perf-*.jsonl"
BOOTS_GLOB = "boots-*.jsonl"
_DAY_RE = re.compile(r"-(\d{4}-\d{2}-\d{2})\.jsonl$")

GROUPS = ("model", "generation", "day", "hour", "config")


def day_of(path: Path) -> str | None:
    m = _DAY_RE.search(path.name)
    return m.group(1) if m else None


def _files(src: Sources, pattern: str, day: str | None, days: int) -> list[Path]:
    found = state_glob(pattern, src, subdir="perf")
    if not found:
        raise SourceUnavailable(
            f"no {pattern} under {src.state_dir}/perf - the LB's recorder has written nothing "
            "there, or the state directory is not mounted"
        )
    if day:
        want = _resolve_day(day)
        picked = [p for p in found if day_of(p) == want]
        if not picked:
            raise SourceUnavailable(
                f"no {pattern} for {want}; recorded days: {sorted({day_of(p) for p in found if day_of(p)})}"
            )
        return picked
    return found[-max(1, int(days)):]


def _resolve_day(day: str) -> str:
    if day in ("today", "now"):
        return dt.date.today().isoformat()
    if day == "yesterday":
        return (dt.date.today() - dt.timedelta(days=1)).isoformat()
    return day


# ── one folded bag of numbers ──────────────────────────────────────────────────────────────────
@dataclass
class Fold:
    """Additive totals over a set of windows. Every field is a sum or a span-weighted accumulator."""

    windows: int = 0
    span_s: float = 0.0
    gen_tokens: float = 0.0
    prompt_tokens: float = 0.0
    prompt_computed_tokens: float = 0.0
    prefix_hit_tokens: float = 0.0
    prefix_query_tokens: float = 0.0
    requests_finished: float = 0.0
    draft_tokens: float = 0.0
    accepted: float = 0.0
    engine_restarts: float = 0.0
    itl_n: float = 0.0
    itl_sum: float = 0.0
    itl_buckets: dict[float, float] = field(default_factory=dict)
    ttft_n: float = 0.0
    ttft_sum: float = 0.0
    ttft_buckets: dict[float, float] = field(default_factory=dict)
    running_span: float = 0.0
    running_max: float = 0.0
    kv_span: float = 0.0
    waiting_max: float = 0.0
    t_first: float | None = None
    t_last: float | None = None
    models: set[str] = field(default_factory=set)
    generations: set[int] = field(default_factory=set)
    #: Any `itl_basis` the window carried. The LB publishes it on the scrape since 9c32807 but does
    #: NOT yet persist it into a window, so for recorded data this stays empty and the basis is
    #: measured instead (see `caveats.itl_semantics`).
    itl_bases: set[str] = field(default_factory=set)

    def add(self, w: dict[str, Any]) -> None:
        self.windows += 1
        span = _f(w.get("span_s"))
        self.span_s += span
        for key in ("gen_tokens", "prompt_tokens", "prompt_computed_tokens", "prefix_hit_tokens",
                    "prefix_query_tokens", "requests_finished", "engine_restarts"):
            setattr(self, key, getattr(self, key) + _f(w.get(key)))
        spec = w.get("spec") or {}
        self.draft_tokens += _f(spec.get("draft_tokens"))
        self.accepted += _f(spec.get("accepted"))
        itl, ttft = w.get("itl") or {}, w.get("ttft") or {}
        for place in (itl.get("basis"), itl.get("itl_basis"), w.get("itl_basis")):
            if place in caveats.ITL_BASIS_VALUES:
                self.itl_bases.add(place)
        self.itl_n += _f(itl.get("n"))
        self.itl_sum += _f(itl.get("mean")) * _f(itl.get("n"))
        self.ttft_n += _f(ttft.get("n"))
        self.ttft_sum += _f(ttft.get("mean")) * _f(ttft.get("n"))
        _buckets(self.itl_buckets, itl.get("buckets"))
        _buckets(self.ttft_buckets, ttft.get("buckets"))
        running, kv, waiting = w.get("running") or {}, w.get("kv") or {}, w.get("waiting") or {}
        self.running_span += _f(running.get("mean")) * span
        self.running_max = max(self.running_max, _f(running.get("max")))
        self.waiting_max = max(self.waiting_max, _f(waiting.get("max")))
        self.kv_span += _f(kv.get("mean")) * span
        t0, t1 = _f(w.get("t_start")), _f(w.get("t_end"))
        if t0:
            self.t_first = t0 if self.t_first is None else min(self.t_first, t0)
        if t1:
            self.t_last = t1 if self.t_last is None else max(self.t_last, t1)
        cfg = w.get("config") or {}
        if cfg.get("model_choice"):
            self.models.add(str(cfg["model_choice"]))
        if isinstance(cfg.get("generation"), int):
            self.generations.add(cfg["generation"])

    # ── derived ────────────────────────────────────────────────────────────────────────────────
    def rates(self) -> dict[str, Any]:
        s = self.span_s or None
        return {
            "minutes_recorded": round(self.span_s / 60.0, 2),
            "gen_tok_s": _r(self.gen_tokens / s) if s else None,
            "prompt_tok_s": _r(self.prompt_tokens / s) if s else None,
            "prompt_computed_tok_s": _r(self.prompt_computed_tokens / s) if s else None,
        }

    def prefix_hit_rate(self) -> float | None:
        """The prefix-cache share, or None when the recorded counts cannot express one.

        Until lb 9c32807 the probe recorded SGLang's `mode="input"` series as the queried total,
        and that series EXCLUDES the hits - so 39 of the windows recorded on 2026-10-03 have
        `prefix_hit_tokens` above `prefix_query_tokens`, up to 6.27x. A share above 1 is impossible,
        so none is returned and `prefix_hit_note()` says why rather than a caller reading 627%.
        """
        if not self.prefix_query_tokens:
            return None
        ratio = self.prefix_hit_tokens / self.prefix_query_tokens
        return _r(ratio, 4) if ratio <= 1.0 else None

    def prefix_hit_note(self) -> str | None:
        if not self.prefix_query_tokens:
            return None
        if self.prefix_hit_tokens / self.prefix_query_tokens > 1.0:
            return caveats.PREFIX_RATE_DEFECT
        return None

    def acceptance(self) -> float | None:
        return _r(self.accepted / self.draft_tokens, 4) if self.draft_tokens else None

    def mean_concurrency(self) -> float | None:
        return _r(self.running_span / self.span_s, 3) if self.span_s else None

    def kv_pct(self) -> float | None:
        return _r(100.0 * self.kv_span / self.span_s, 2) if self.span_s else None

    def totals(self) -> dict[str, Any]:
        out = {
            "gen_tokens": int(self.gen_tokens),
            "prompt_tokens": int(self.prompt_tokens),
            "prompt_computed_tokens": int(self.prompt_computed_tokens),
            "requests_finished": int(self.requests_finished),
            "prefix_hit_tokens": int(self.prefix_hit_tokens),
            "prefix_query_tokens": int(self.prefix_query_tokens),
            "prefix_hit_rate": self.prefix_hit_rate(),
            "spec_acceptance": self.acceptance(),
            "engine_restarts": int(self.engine_restarts),
        }
        note = self.prefix_hit_note()
        if note:
            out["prefix_hit_rate_unusable"] = note
        return out

    def latency(self) -> dict[str, Any]:
        itl_mean = _r(self.itl_sum / self.itl_n, 5) if self.itl_n else None
        stated = next(iter(self.itl_bases)) if len(self.itl_bases) == 1 else None
        sem = caveats.itl_semantics(self.gen_tokens, self.itl_n, basis=stated)
        if len(self.itl_bases) > 1:
            sem["disagreement"] = (f"this selection mixes engines: itl_basis values "
                                   f"{sorted(self.itl_bases)} appear in it, so no single basis "
                                   "applies - fold by model instead")
        div = sem.get("per_token_divisor") or 1.0
        return {
            "ttft_s": {"samples": int(self.ttft_n),
                       "mean": _r(self.ttft_sum / self.ttft_n, 4) if self.ttft_n else None,
                       **percentiles(self.ttft_buckets, self.ttft_n)},
            "itl_raw_s": {"samples": int(self.itl_n), "mean": itl_mean,
                          **percentiles(self.itl_buckets, self.itl_n)},
            "itl_per_token_s": {"mean": _r(itl_mean / div, 5) if itl_mean else None},
            "itl_semantics": sem,
        }


def _buckets(into: dict[float, float], raw: Any) -> None:
    for pair in raw or ():
        try:
            le, n = float(pair[0]), float(pair[1])
        except (TypeError, ValueError, IndexError):
            continue
        into[le] = into.get(le, 0.0) + n


def percentiles(buckets: dict[float, float], n: float,
                wanted: Iterable[float] = (0.5, 0.9, 0.99)) -> dict[str, Any]:
    """Quantiles from de-cumulated `le` histogram buckets, reported as the bucket's upper bound.

    The LB writes one `[le, count]` pair per bucket whose counts sum to the sample count, so this
    walks them in order; a quantile falling past the last finite bucket returns None rather than a
    number the data cannot support.
    """
    out: dict[str, Any] = {}
    if not buckets or not n:
        return {f"p{int(q * 100)}": None for q in wanted}
    ordered = sorted(buckets.items())
    for q in wanted:
        target, seen, value = q * n, 0.0, None
        for le, count in ordered:
            seen += count
            if seen >= target:
                value = le
                break
        out[f"p{int(q * 100)}"] = value
    return out


# ── window selection ───────────────────────────────────────────────────────────────────────────
def load_windows(src: Sources, day: str | None = None, days: int = 1,
                 model: str | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Every window in the chosen day-files, newest file last, plus what was skipped and why."""
    files = _files(src, PERF_GLOB, day, days)
    windows: list[dict[str, Any]] = []
    malformed: list[dict[str, Any]] = []
    for path in files:
        for line_no, rec in jsonl(path, src):
            if rec is None:
                malformed.append({"file": path.name, "line": line_no})
                continue
            if not isinstance(rec, dict) or rec.get("kind") == "boot":
                continue
            if model and (rec.get("config") or {}).get("model_choice") != model:
                continue
            # The day a window belongs to is the day of the FILE the LB wrote it into. Deriving
            # it from t_start instead would bucket by whatever timezone this process happens to
            # run in, which in a bare container is UTC and not the LB's own day boundary.
            rec["_day"] = day_of(path)
            windows.append(rec)
    meta = {
        "files": [p.name for p in files],
        "days": sorted({d for d in (day_of(p) for p in files) if d}),
        "windows": len(windows),
        "malformed_lines": len(malformed),
        "malformed": malformed[:5],
        "filtered_to_model": model,
    }
    return windows, meta


def group_key(w: dict[str, Any], by: str) -> str:
    cfg = w.get("config") or {}
    if by == "model":
        return str(cfg.get("model_choice") or "unknown")
    if by == "generation":
        return str(cfg.get("generation") if cfg.get("generation") is not None else "unknown")
    if by == "day":
        return str(w.get("_day") or _stamp(w.get("t_start"), "%Y-%m-%d"))
    if by == "hour":
        return _stamp(w.get("t_start"), "%Y-%m-%d %H:00")
    if by == "config":
        return (f"{cfg.get('model_choice')} seats={cfg.get('max_num_seqs')} "
                f"batched={cfg.get('max_num_batched_tokens')} spec={cfg.get('num_speculative_tokens')}")
    raise ValueError(f"group_by must be one of {GROUPS}, got {by!r}")


def fold(windows: Iterable[dict[str, Any]]) -> Fold:
    f = Fold()
    for w in windows:
        f.add(w)
    return f


def fold_by(windows: Iterable[dict[str, Any]], by: str) -> dict[str, Fold]:
    out: dict[str, Fold] = {}
    for w in windows:
        out.setdefault(group_key(w, by), Fold()).add(w)
    return out


# ── cluster starts ─────────────────────────────────────────────────────────────────────────────
def load_boots(src: Sources, days: int = 7, outcome: str | None = None) -> tuple[list[dict], dict]:
    files = _files(src, BOOTS_GLOB, None, days)
    boots, malformed = [], 0
    for path in files:
        for _, rec in jsonl(path, src):
            if rec is None:
                malformed += 1
                continue
            if not isinstance(rec, dict):
                continue
            if outcome and rec.get("outcome") != outcome:
                continue
            boots.append(rec)
    boots.sort(key=lambda b: _f(b.get("t_start")))
    return boots, {"files": [p.name for p in files], "malformed_lines": malformed,
                   "filtered_to_outcome": outcome}


# ── small helpers ──────────────────────────────────────────────────────────────────────────────
def _f(v: Any) -> float:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0.0


def _r(v: float | None, places: int = 2) -> float | None:
    return None if v is None else round(v, places)


def _stamp(epoch: Any, fmt: str) -> str:
    """Local wall-clock, with the offset appended so the string cannot be read as UTC by mistake."""
    e = _f(epoch)
    if not e:
        return "unknown"
    local = dt.datetime.fromtimestamp(e).astimezone()
    return local.strftime(fmt) + local.strftime("%z")


def iso(epoch: Any) -> str | None:
    """An offset-aware ISO stamp: this process's timezone need not match the LB host's."""
    e = _f(epoch)
    return dt.datetime.fromtimestamp(e).astimezone().isoformat(timespec="seconds") if e else None
