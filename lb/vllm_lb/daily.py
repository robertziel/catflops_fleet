"""THE DAILY ROLLUP: per model, the last 30 UTC days of generated tok/s and tokens, and the last 24 h's token total —
a pure fold over the real-traffic performance log (perflog.py) the LB already writes.

WHY A FOLD AND NOT A COLLECTOR. Every minute the cluster served real traffic is already a line in `perf-*.jsonl`, with
its generated and prompt token counts and the seconds it covered. A second collector would keep a second definition of
"tokens served" that drifts from the first; folding the same lines means /benchmarks, /sessions and this rollup can
never disagree about what a day served. Nothing is written: the endpoint reads, folds, answers.

WHY NOT `PerfLog.index`. /lb/perf hands over at most `perflog.MAX_WINDOWS` (14 days of minutes) — the newest — because
it ships every window to a browser. Thirty days of minutes is more than that, and a rollup that silently lost its
oldest two weeks would draw them as idle. `scan` reads every line in range through PerfLog's own reader (its cache, its
malformed-line accounting) with no cut; the fold keeps only per-day sums, so its answer is small whatever the log holds.

WHAT A DAY SAYS (UTC days, by each minute's END, the same `day_of` the log names its files by):
  gen_tokens / prompt_tokens  the day's sums over minutes whose engine exposed the counter; null when NO minute did —
                              a day of minutes nobody could count is not a day of 0 tokens (`unmeasured` counts them);
  gen_tok_s                   the day's generated tokens over the seconds its measured minutes covered — pooled, never
                              a mean of per-minute rates (a 5-second minute would weigh as much as a full one);
                              null for a day without a measured minute: a rate needs a minute;
  peak_gen_tok_s              the best single minute;
  minutes                     how many minutes the log holds (idle minutes are never written, so a day with none is a
                              day the server served nothing — 0 tokens is then the truth, and the rate is null).
`last_24h` is the minutes that ENDED inside the last 86 400 s, per model; `total_24h` sums the models.
"""
from __future__ import annotations

import time

from vllm_lb.perflog import PerfLog, _num, day_of, window_model

#: The rollup's range, in UTC days, today included.
DAYS = 30
_DAY_S = 86400.0


def scan(log: PerfLog, *, now: float | None = None, days: int = DAYS, keep=None) -> dict:
    """Every window in the log that ended inside the last `days` UTC days (today's whole day included), oldest
    first, with NO MAX_WINDOWS cut; plus what could not be read. `keep`: the same per-server filter /lb/perf takes."""
    now = time.time() if now is None else now
    first_day = day_of(now - (days - 1) * _DAY_S)
    files, error = log._files()
    windows: list[dict] = []
    malformed = 0
    unreadable: list[dict] = []
    for path in files:
        if path.name < f"{log.prefix}-{first_day}.jsonl":
            continue                                   # a whole day before the range
        ws, bad, err = log._read(path)
        malformed += len(bad)
        if err:
            unreadable.append({"file": path.name, "error": err})
        windows.extend(w for w in ws if day_of(w["t_end"]) >= first_day)
    if keep is not None:
        windows = keep(windows)
    windows.sort(key=lambda w: (w["t_start"], w["t_end"]))
    return {"windows": windows, "error": error, "malformed_total": malformed, "unreadable": unreadable}


def _empty(day: str) -> dict:
    return {"day": day, "gen_tokens": None, "prompt_tokens": None, "minutes": 0, "unmeasured": 0,
            "_span": 0.0, "_peak": None}


def _add(slot: dict, w: dict) -> None:
    """One minute into a day's (or the 24 h's) sums."""
    slot["minutes"] += 1
    gen, prompt, span = _num(w.get("gen_tokens")), _num(w.get("prompt_tokens")), _num(w.get("span_s"))
    if gen is None:
        slot["unmeasured"] += 1
    else:
        slot["gen_tokens"] = (slot["gen_tokens"] or 0) + gen
        if span is not None and span > 0:
            slot["_span"] = slot.get("_span", 0.0) + span
            rate = gen / span
            slot["_peak"] = rate if slot.get("_peak") is None else max(slot["_peak"], rate)
    if prompt is not None:
        slot["prompt_tokens"] = (slot["prompt_tokens"] or 0) + prompt


def _day_view(slot: dict) -> dict:
    span = slot["_span"]
    gen = slot["gen_tokens"]
    if slot["minutes"] == 0:                           # nothing served that day: 0 tokens is the truth, no rate
        gen, prompt = 0, 0
    else:
        prompt = slot["prompt_tokens"]
    return {"day": slot["day"], "gen_tokens": None if gen is None else int(round(gen)),
            "prompt_tokens": None if prompt is None else int(round(prompt)),
            "gen_tok_s": round(slot["gen_tokens"] / span, 2) if slot["gen_tokens"] is not None and span > 0 else None,
            "peak_gen_tok_s": None if slot["_peak"] is None else round(slot["_peak"], 2),
            "minutes": slot["minutes"], "unmeasured": slot["unmeasured"]}


def _last_view(slot: dict) -> dict:
    gen, prompt = slot["gen_tokens"], slot["prompt_tokens"]
    if slot["minutes"] == 0:
        gen, prompt = 0, 0
    return {"gen_tokens": None if gen is None else int(round(gen)),
            "prompt_tokens": None if prompt is None else int(round(prompt)),
            "minutes": slot["minutes"], "unmeasured": slot["unmeasured"]}


def fold(windows, *, now: float, days: int = DAYS) -> dict:
    """The rollup of `windows` (perf records) at `now`: the day axis (oldest first), one entry per model — its day
    series on that axis and its last 24 h — ordered by its range's generated tokens, most first; and the 24 h total."""
    axis = [day_of(now - i * _DAY_S) for i in range(days - 1, -1, -1)]
    on_axis = set(axis)
    cutoff_24h = now - _DAY_S
    per: dict[str, dict] = {}
    total = {"day": "", "gen_tokens": None, "prompt_tokens": None, "minutes": 0, "unmeasured": 0, "_span": 0.0,
             "_peak": None}
    for w in windows:
        t_end = _num(w.get("t_end"))
        if t_end is None or t_end > now:
            continue
        day = day_of(t_end)
        if day not in on_axis:
            continue
        model = window_model(w)
        m = per.get(model)
        if m is None:
            m = per[model] = {"days": {d: _empty(d) for d in axis},
                              "last": {"day": "", "gen_tokens": None, "prompt_tokens": None, "minutes": 0,
                                       "unmeasured": 0, "_span": 0.0, "_peak": None}}
        _add(m["days"][day], w)
        if t_end >= cutoff_24h:
            _add(m["last"], w)
            _add(total, w)
    models = []
    for model, m in per.items():
        series = [_day_view(m["days"][d]) for d in axis]
        models.append({"id": model, "series": series, "last_24h": _last_view(m["last"]),
                       "gen_tokens_range": sum(d["gen_tokens"] or 0 for d in series)})
    models.sort(key=lambda m: (-m["gen_tokens_range"], m["id"]))
    return {"now": now, "days": axis, "models": models, "total_24h": _last_view(total)}
