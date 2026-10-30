"""The benchmark swarm's sessions - what each one actually put through the pair.

A session is one capped deep-research run driven against the local gateway. Its durable record is
the capture directory `<state>/benchq/sessions/S-<local start>/`, whose `metrics-before.txt` and
`metrics-after.txt` are written at the first and last moment of the run - so their mtimes bracket
the session exactly (checked on one recorded session: they match `session.json`'s started_at /
ended_at to the millisecond). The *measurements* then come from the LB's own per-minute windows
inside that bracket, not from anything the session reports about itself.

Every session answer carries `caveats.SWARM_SPREAD`: between-session variance here is bigger than
any configuration effect ever measured on this fleet, so a single session's throughput is not its
configuration's throughput.

A session's own `stream.jsonl` also carries what the CLIENT thought it spent, and that number is a
trap worth reporting explicitly rather than leaving for a caller to find: `total_cost_usd` is
computed client-side at Anthropic list prices even though the two GB10 boxes served the request.
Checked on one recorded session - 27,849 in + 798 out, `total_cost_usd` 0.127356, which is exactly
27849 x $4/MTok + 798 x $20/MTok - and the record admits it, with
`modelUsage[...].costBasis == "unknown"`. So `client_accounting` is reported with the dollars
flagged as fiction and the engine-side measurement kept separate.
"""
from __future__ import annotations

import datetime as dt
import re
from pathlib import Path
from typing import Any

from . import caveats, perf
from .readonly import Sources, SourceUnavailable, jsonl, mtime, read_json, state_dirs

SESSIONS_SUBDIR = "benchq/sessions"
SESSION_JSON = "benchq/session.json"
_LABEL_RE = re.compile(r"^S-(\d{8})-(\d{6})$")


def sessions(src: Sources) -> list[dict[str, Any]]:
    """Every capture directory, oldest first, with the bracket each one ran in."""
    dirs = state_dirs(SESSIONS_SUBDIR, src)
    if not dirs:
        raise SourceUnavailable(
            f"no session captures under {src.state_dir}/{SESSIONS_SUBDIR} - the swarm has not run, "
            "or the state directory is not mounted"
        )
    latest = _latest(src)
    out = []
    for d in dirs:
        started = mtime(d / "metrics-before.txt", src)
        ended = mtime(d / "metrics-after.txt", src)
        label_start = _from_label(d.name)
        rec: dict[str, Any] = {
            "label": d.name,
            "started_at": started or label_start,
            "ended_at": ended,
            "bracket_source": ("capture file mtimes" if started and ended else
                              "the label's timestamp, read in THIS process's timezone - only a "
                              "guess, since the label was written in the LB host's local time and "
                              "there is no metrics-after.txt (the session did not finish)"),
            "capture_files": sorted(p.name for p in _files(d)),
        }
        if latest and latest.get("label") == d.name:
            rec.update({
                "started_at": latest.get("started_at") or rec["started_at"],
                "ended_at": latest.get("ended_at") or rec["ended_at"],
                "bracket_source": "benchq/session.json (this is the most recent session)",
                "phase": latest.get("phase"),
                "rc": latest.get("rc"),
                "cap_min": latest.get("cap_min"),
                "served_model": latest.get("served_model"),
                "model_choice": latest.get("model_choice"),
                "generation": latest.get("generation"),
                "guards": latest.get("guards"),
                "detail": latest.get("detail"),
            })
        out.append(rec)
    # Ordered by LABEL, not by the derived epoch: the label is `S-<YYYYMMDD>-<HHMMSS>` and so
    # sorts chronologically as a string, without depending on this process's timezone matching the
    # LB host's. A container with no /etc/localtime would otherwise reorder unfinished sessions.
    out.sort(key=lambda r: r["label"])
    return out


def measure(src: Sources, session: dict[str, Any], days: int = 7) -> dict[str, Any]:
    """Fold the LB's windows that fall inside this session's bracket."""
    t0, t1 = session.get("started_at"), session.get("ended_at")
    if not t0:
        return {"measured": False, "why": "no start time for this session"}
    windows, meta = perf.load_windows(src, day=None, days=days)
    inside = [w for w in windows
              if isinstance(w.get("t_start"), (int, float))
              and w["t_start"] >= t0 - 1.0
              and (t1 is None or w["t_start"] <= t1 + 1.0)]
    if not inside:
        return {"measured": False,
                "why": f"no recorded windows inside this bracket (searched {meta['days']}) - either "
                       "the session served nothing or its day's perf log has rotated out"}
    f = perf.fold(inside)
    cfgs = sorted(f.models)
    return {
        "measured": True,
        "windows": f.windows,
        "minutes": round(f.span_s / 60.0, 2),
        "gen_tokens": int(f.gen_tokens),
        "prompt_tokens": int(f.prompt_tokens),
        "gen_tok_s": f.rates()["gen_tok_s"],
        "mean_concurrency": f.mean_concurrency(),
        "max_concurrency": int(f.running_max),
        "spec_acceptance": f.acceptance(),
        "prefix_hit_rate": f.prefix_hit_rate(),
        "prefix_hit_rate_unusable": f.prefix_hit_note(),
        "kv_pct": f.kv_pct(),
        "model_choices": cfgs,
        "generations": sorted(f.generations),
    }


CLIENT_COST_IS_FICTION = (
    "client_accounting is what the CLIENT (Claude Code) recorded, not what the engine measured, "
    "and its dollars are fiction: total_cost_usd is computed client-side at ANTHROPIC LIST PRICES "
    "for a request the two GB10 boxes served, so no money was spent. The record says so itself - "
    "costBasis is 'unknown'. Use the `measured` block for what actually happened."
)


def client_accounting(src: Sources, label: str) -> dict[str, Any]:
    """The session's own closing `result` event: tokens the client counted, and its fake cost."""
    path = Path(src.state_dir) / SESSIONS_SUBDIR / label / "stream.jsonl"
    last = None
    try:
        for _, rec in jsonl(path, src):
            if isinstance(rec, dict) and rec.get("type") == "result":
                last = rec
    except SourceUnavailable as exc:
        return {"recorded": False, "why": str(exc)}
    if last is None:
        return {"recorded": False,
                "why": "no result event in this session's stream (it was cut off before finishing)"}
    usage = last.get("usage") or {}
    models = last.get("modelUsage") or {}
    basis = sorted({str((v or {}).get("costBasis")) for v in models.values() if isinstance(v, dict)})
    return {
        "recorded": True,
        "input_tokens": _i(usage.get("input_tokens")),
        "output_tokens": _i(usage.get("output_tokens")),
        "cache_read_input_tokens": _i(usage.get("cache_read_input_tokens")),
        "cache_creation_input_tokens": _i(usage.get("cache_creation_input_tokens")),
        "num_turns": _i(last.get("num_turns")),
        "subtype": last.get("subtype") if isinstance(last.get("subtype"), str) else None,
        "client_models": sorted(str(k) for k in models),
        "cost_basis": basis,
        "total_cost_usd_IS_FICTION": last.get("total_cost_usd")
            if isinstance(last.get("total_cost_usd"), (int, float)) else None,
        "note": CLIENT_COST_IS_FICTION,
    }


def _i(v: Any) -> int | None:
    return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _files(d: Path) -> list[Path]:
    try:
        return [p for p in d.iterdir() if p.is_file()]
    except OSError:                                            # pragma: no cover - unreadable dir
        return []


def _latest(src: Sources) -> dict[str, Any] | None:
    try:
        doc = read_json(Path(src.state_dir) / SESSION_JSON, src)
    except SourceUnavailable:
        return None
    return doc if isinstance(doc, dict) else None


def _from_label(label: str) -> float | None:
    m = _LABEL_RE.match(label)
    if not m:
        return None
    try:
        return dt.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").timestamp()
    except ValueError:                                         # pragma: no cover - malformed label
        return None


SPREAD = caveats.SWARM_SPREAD
