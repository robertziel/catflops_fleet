"""What usage data this fleet actually holds - counted off disk, not described from memory.

The audit this server was built from asked whether "API usage" here means the local gateway's own
usage or the operator's Anthropic subscription. It is the first, and the second does not exist
locally:

* The swarm harness (`lb/bench_session.py`) launches Claude Code with `env -i` and
  `ANTHROPIC_BASE_URL` pointed at the LB plus a dummy `ANTHROPIC_API_KEY`, refuses to start if any
  inherited `ANTHROPIC*` / `CLAUDE_CODE*` variable could beat it, and asserts the session's own
  `init` event reports `apiKeySource: "ANTHROPIC_API_KEY"` - `none` would mean a subscription and
  aborts the run. So the tokens a session spends are served by the two GB10 boxes.
* Consistent with that, `benchq/session.json` records `"lb": "http://127.0.0.1:8010"` and
  `"model": "qwen3.8:latest"`, and a capture's `stderr.txt` carries
  `[claude-code:unrecognized_model] {"model":"qwen3.8:latest"}` - a local served name, not an
  Anthropic model id.

Audited across all 38 recorded session streams: 37 show that local signature. **One does not** -
`worklog/bench/claude/wrong-backend/P1-s8-b8k-sp4-stream.jsonl`, the 2026-09-30 accident
`bench_session.py` was then hardened against: `model: "claude-opus-5"`, `apiKeySource: "none"`, one
turn, and the only `rate_limit_event` on this host (five_hour utilisation 0.14, seven_day 0.11 at
~2026-09-30T07:06Z). That is one instant of one window, not a usage series - and it is under
`worklog/`, outside this server's read roots, so no tool reports it either.

The dollar figures the local sessions DO carry are not spend. `total_cost_usd` is computed
client-side at Anthropic list prices for traffic the GB10 boxes served: on one recorded session it is
0.127356, exactly 27,849 in x $4/MTok + 798 out x $20/MTok, and the record itself says
`costBasis: "unknown"`. The `swarm_sessions` tool reports that figure only with the fiction
labelled.

What is deliberately NOT wired in: Claude Code's own per-session accounting under `~/.claude`
(transcript `usage` blocks, `~/.claude.json`'s `lastModelUsage`, the telemetry queue's
`tengu_exit` totals). It exists, but it is the operator's client state rather than fleet data, it
is per-session with no account-level total, `~/.claude` also holds `.credentials.json`, and
mounting it into an MCP server any agent can call would put credentials one tool away from a model.
A real account total would come from Anthropic's own usage API with an admin credential, which this
server has no path to and was explicitly not built to reach.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .readonly import LB_PATHS, Sources, state_dirs, state_glob

NO_ANTHROPIC_USAGE = (
    "This fleet records NO usable Anthropic-account usage: no token series, no cost, no rate-limit "
    "history. The benchmark sessions run Claude Code against the LOCAL gateway - lb/bench_session.py "
    "pins ANTHROPIC_BASE_URL at the LB with a dummy key under `env -i` and aborts if the session's "
    "init event reports apiKeySource 'none' - so every token this server reports was served by the "
    "two GB10 boxes. Audited: 37 of 38 recorded session streams show that local signature. The 38th, "
    "worklog/bench/claude/wrong-backend/P1-s8-b8k-sp4-stream.jsonl, is the 2026-09-30 accident that "
    "guard was added for: ONE claude-opus-5 turn with apiKeySource 'none' plus the only "
    "rate_limit_event on this host (five_hour 0.14 / seven_day 0.11 at ~2026-09-30T07:06Z). One "
    "instant of one window is not a usage series, and it sits under worklog/, outside this "
    "server's read roots. Do NOT substitute local tokens for subscription spend, and do NOT read "
    "total_cost_usd from a local session as money: it is client-side arithmetic at Anthropic list "
    "prices with costBasis 'unknown'."
)

NOT_WIRED_IN = (
    "Claude Code's own per-session accounting under ~/.claude (transcript usage blocks, "
    "~/.claude.json lastModelUsage, the telemetry queue's tengu_exit totals) does exist on this "
    "host, but it is the operator's client state, not fleet data; it has no account-level total; "
    "and ~/.claude also holds .credentials.json, so mounting it into an MCP server any agent can "
    "call would put credentials one tool away from a model. Not mounted, not read, not reported."
)


def take(src: Sources) -> dict[str, Any]:
    """Count what is on disk and name what each file holds."""
    perf_files = state_glob("perf-*.jsonl", src, subdir="perf")
    boots_files = state_glob("boots-*.jsonl", src, subdir="perf")
    bench_files = state_glob("*.json", src, subdir="bench")
    archives = [d for d in state_dirs("", src) if d.name.startswith("bench-archive")]
    perf_archives = [d for d in state_dirs("", src) if d.name.startswith("perf-archive")]
    sessions = state_dirs("benchq/sessions", src)

    return {
        "meaning": "the LOCAL vLLM/SGLang gateway's own usage - what the two-box pair served",
        "sources": {
            "perf_windows": _group("per-minute windows: generated and prompt tokens, prefix hit "
                                   "rate, spec acceptance, TTFT/ITL histograms, KV usage, per-box "
                                   "GPU and power, plus the model_choice and generation each "
                                   "minute ran under", perf_files),
            "cluster_boots": _group("one record per cluster start: the configuration asked for, "
                                    "the phases, the outcome and the failure if any", boots_files),
            "saved_benchmarks": _group("saved benchmark runs the /benchmarks page browses",
                                       bench_files),
            "swarm_sessions": {
                "what": "one capture directory per benchmark swarm session: metrics before and "
                        "after, the session's stream, stderr",
                "count": len(sessions),
                "newest": sessions[-1].name if sessions else None,
            },
            "archives": {
                "what": "earlier runs moved aside by hand; this server reads them for counts only",
                "bench": [{"dir": d.name, "files": len(_json_in(d))} for d in archives],
                "perf": [d.name for d in perf_archives],
            },
        },
        "live_endpoints": sorted(LB_PATHS),
        "state_dir": str(src.state_dir),
        "anthropic_usage": NO_ANTHROPIC_USAGE,
        "not_wired_in": NOT_WIRED_IN,
    }


def _group(what: str, files: list[Path]) -> dict[str, Any]:
    return {
        "what": what,
        "files": len(files),
        "days": sorted({f.name.split("-", 1)[1].removesuffix(".jsonl")
                        for f in files if "-" in f.name and f.name.endswith(".jsonl")}),
        "bytes": sum(_size(f) for f in files),
        "newest": files[-1].name if files else None,
    }


def _json_in(d: Path) -> list[Path]:
    try:
        return [p for p in d.iterdir() if p.is_file() and p.suffix == ".json"]
    except OSError:                                            # pragma: no cover
        return []


def _size(f: Path) -> int:
    try:
        return f.stat().st_size
    except OSError:                                            # pragma: no cover
        return 0
