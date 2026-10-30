"""THE REQUEST QUEUE: the turns the ENGINE is working on right now, one row each with the LB's own measurements off
the relayed stream, and the last ones it finished — /lb/status `queue`, the page's REQUESTS card.

The engine publishes no per-request state (SGLang's /metrics: `num_running_reqs`, `num_queue_reqs`, aggregates —
they stand beside the rows as `engine`), so every state here is the LB's own view of the turn:

  * `held`      — admitted, NOTHING sent to any backend (the cluster hold, a claim being waited for, a turn that
                  arrived during a drain). A COUNT, never a row: the engine is not working on it.
  * no bytes yet — sent, nothing back. SGLang serves FCFS and chunk-prefills one request at a time, and the one
                  being prefilled is in NEITHER gauge (worklog/sglang-bottleneck-2026-10-03.md: 30 running + 6
                  waiting + 1 = 37 LB in flight). So of a backend's N no-byte turns, with its gauge reading W
                  waiting, the OLDEST N − W are `prefill` and the rest `queued` — INFERRED, and the row says so
                  (`inferred`). With no gauge (`depth_known` false, not live) the split is unknown: `sent`.
  * why the queued wait — per server, `admission`: a seat, KV blocks, the engine's deferral, or not proven, from the
                  engine's own figures (its running / waiting and their reasons, its KV gauge and pool, the seats it
                  was launched with).
  * `streaming` — bytes reaching the client. `prefill_s` = send → `message_start` (the page's TTFT: the engine's
                  queue AND the prefill, which it reports no split of per request) (SGLang's adapter defers it to
                  the first chunk carrying usage: the end of the prefill — and of any queueing before it), and
                  `prefill_tok_s` = the prompt it reported ÷ that. `deltas` = content events since, split thinking /
                  text / tool — events, NOT tokens (the captured cluster stream carried 32 tokens in 9) — and
                  `delta_rate` their LIFETIME rate; `output_tokens` stays null until the engine's own count
                  arrives. A turn whose stream has said its stop reason is finishing: not a row either.

THE DECODE RATE A ROW SHOWS, `tok_s`, is DERIVED (`DERIVED_DECODE`), never measured — operator 2026-10-16:
"requests tok/s doesn't add up to official total" (rows Δ… · 8.6/s summing to ~44/s beside ~180 tok/s GENERATED).
No relayed stream carries a per-chunk token count (every captured body: usage at `message_start` and in the final
`message_delta` / usage chunk only), so per turn mid-stream there is NO token figure to read. What there is: the
engine's GENERATED tok/s over its two-probe window (throughput.py, the card's own headline), and each turn's content
events stamped on the same monotonic clock (UsageTap `events_between`). A row's `tok_s` = GENERATED × its events in
THAT window ÷ the events of every turn relayed to that server in it — a prefilling or queued turn has none (0, and
`window_events` 0 says why), and a turn that finished or is finishing inside the window keeps its share as
`decode.backends[url].ended_tok_s`, rows past the cap theirs as `unlisted_tok_s`, so the rows + ended + unlisted ADD
UP to GENERATED (each figure rounded to 0.1). It assumes every token the engine counted was a turn relayed here —
`DERIVED_DECODE` says so. With no GENERATED figure (first sample, scrape failed, no counter), or no turn sending an
event inside the window while the engine generated, or a window older than the stamps kept, NO row gets a number:
`tok_s_why` says which.

and once a turn is over, its OUTCOME, in `recent`:

  * `done`        — answered < 400, whole: an Anthropic stream's own `message_stop` went out.
  * `killed`      — the operator's kill ended it (request_kill.py, POST /lb/requests/<id>/kill), `why` = by whom and
                    in which state; first of all, since a killed stream also lacks its `message_stop`.
  * `client_gone` — a write to the CLIENT failed: it hung up (the relay's `writing`).
  * `failed`      — anything else, with `why`: the LB's own error type (`no_backend`, `draining`,
                    `backend_timeout`, `backend_dropped_request`, `all_backends_failed`), `HTTP <status>` from the
                    backend, `backend went silent` / `backend failed mid-stream`, `no message_stop`, `error event`.

THE CONTENT is the operator's own traffic on a page behind nginx basic auth, and it is BOUNDED: per row the LAST
USER TURN's text (`PROMPT_CAP`, the page shows one line and the rest on hover), per finished turn a
`SNIPPET_CHARS` snippet. Never an earlier message, the system prompt, a tool's name, schema or description, or an
assistant's text. Read off the body the proxy has ALREADY parsed (`_as_json` in `handle`, the object the router
rewrites) — never a second parse.

The body shape is CAPTURED (Claude Code 2.1.288 `claude -p` against a local listener, 2026-10-03):
`metadata.user_id` is a JSON string carrying `session_id`, which also rides `X-Claude-Code-Session-Id`; the LAST
message is `role: system` (mid-conversation-system beta), so the last user turn is the last `role: user`
message; and that turn opens with a `<system-reminder>` block — counted, never quoted.
"""
from __future__ import annotations

import json
import re

from vllm_lb import prompt_count
from vllm_lb.router import priority_lock
from vllm_lb.usage_tap import EVENTS_KEPT_S

HELD, SENT, PREFILL, QUEUED, STREAMING = "held", "sent", "prefill", "queued", "streaming"
#: A turn whose stream has said its stop reason: not a row (above), but a state a kill can still land in.
FINISHING = "finishing"
DONE, CLIENT_GONE, KILLED, FAILED = "done", "client_gone", "killed", "failed"
#: The backend of a turn the LB answers itself (`Proxy.answer_web_search`, SearXNG): no engine, so never a row.
WEBSEARCH = "websearch"
#: The rows' states, in the order the rows are drawn: what is streaming, what is being prefilled, what waits.
ROW_STATES = (STREAMING, PREFILL, QUEUED, SENT)
OUTCOMES = (DONE, CLIENT_GONE, KILLED, FAILED)

#: The bounds. /lb/status is polled every 3 s by every open page, so the view is capped as a whole:
#: 64 rows × (4,000 chars + figures) + 30 × a 160-char snippet stays well under half a megabyte.
INFLIGHT_ROWS = 64
RECENT_ROWS = 30
PROMPT_CAP = 4000
SNIPPET_CHARS = 160
TOOL_RESULT_CHARS = 120
NAME_CHARS = 64
CLIENT_CHARS = 24
#: A rate over less than this is noise, not a figure.
MIN_RATE_S = 1.0
#: How the DECODE column's `tok_s` is made, said beside it wherever it is shown — never presented as measured.
DERIVED_DECODE = ("derived: the engine's GENERATED tok/s over its last probe window, apportioned by each turn's "
                  "content events in that same window (no stream carries a per-chunk token count); assumes every "
                  "token the engine counted belongs to a turn relayed here")
#: Why a row has no `tok_s`: the closed set beside the engine's own `why` (rates.py) for its GENERATED.
NO_EVENT_IN_WINDOW = "the engine generated but no content event was relayed in its window: nothing to apportion by"
OLDER_THAN_KEPT = f"the engine's window is older than the {EVENTS_KEPT_S:.0f} s of events kept"
NOT_PROBED = "no probe of this server's token counter yet"
SERVER_NOT_LIVE = "its server is not live"

#: Claude Code's User-Agent (captured): `claude-cli/2.1.288 (external, sdk-cli)` — `sdk-cli` is `claude -p`
#: (the swarm, lb/bench_session.py), `cli` the interactive session.
_CLAUDE_UA = re.compile(r"^claude-cli/\S+\s*\(([^)]*)\)")
_REMINDER = re.compile(r"<system-reminder>.*?(?:</system-reminder>|\Z)", re.S)
#: The legacy `metadata.user_id`: `user_<hash>_account_<uuid>_session_<uuid>`.
_LEGACY_SESSION = re.compile(r"_session_([0-9a-fA-F-]{8,})$")


def client_of(user_agent: str | None) -> tuple[str, str | None]:
    """WHO SENT IT, by its User-Agent and nothing else: `("claude", <entrypoint>)` for Claude Code, `browser` for
    anything that says `Mozilla/` (the gateway page's own chat), `no ua` for none — a relay built on
    `http.client` sends none (`Content-Type` only) — else the UA's first product name."""
    ua = (user_agent or "").strip()
    if not ua:
        return "no ua", None
    m = _CLAUDE_UA.match(ua)
    if m:
        entry = m.group(1).split(",")[-1].strip()
        return "claude", entry[:CLIENT_CHARS] or None
    if ua.startswith("claude-cli/"):
        return "claude", None
    if ua.startswith("Mozilla/"):
        return "browser", None
    return ua.split("/", 1)[0].split()[0][:CLIENT_CHARS], None


def session_of(headers, body: dict | None) -> str | None:
    """The client's session, its last 8 characters: `X-Claude-Code-Session-Id`, else `metadata.user_id` — a JSON
    string with `session_id` (Claude Code 2.1.288) or the legacy `…_session_<uuid>`."""
    sid = headers.get("X-Claude-Code-Session-Id") if headers is not None else None
    if not sid and isinstance(body, dict):
        uid = (body.get("metadata") or {}).get("user_id") if isinstance(body.get("metadata"), dict) else None
        if isinstance(uid, str):
            try:
                parsed = json.loads(uid)
                sid = parsed.get("session_id") if isinstance(parsed, dict) else None
            except ValueError:
                m = _LEGACY_SESSION.search(uid)
                sid = m.group(1) if m else None
    return sid[-8:] if isinstance(sid, str) and sid else None


def _text_of(content, limit: int) -> str:
    """The text of a block's `content` — a string, or text blocks — up to `limit` characters."""
    if isinstance(content, str):
        return content[:limit]
    out, n = [], 0
    for block in content if isinstance(content, list) else ():
        if isinstance(block, dict) and isinstance(block.get("text"), str) and n < limit:
            out.append(block["text"][:limit - n])
            n += len(out[-1])
    return " ".join(out)[:limit]


def _last_user_turn(messages) -> tuple[str, int] | None:
    """The last `role: user` message as one bounded string, and its whole length: text blocks joined, tool results
    summarised as `[tool_result ×N]` + the first TOOL_RESULT_CHARS of their text, images as `[image ×N]`, and
    `<system-reminder>`s — Claude Code's injections — counted as `[system-reminder ×N]`."""
    last = next((m for m in reversed(messages) if isinstance(m, dict) and m.get("role") == "user"), None)
    if last is None:
        return None
    content = last.get("content")
    blocks = [{"type": "text", "text": content}] if isinstance(content, str) else content
    if not isinstance(blocks, list):
        return None
    texts: list[str] = []
    results: list[str] = []
    images = reminders = whole_texts = kept = 0
    for block in blocks:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text" and isinstance(block.get("text"), str):
            text = block["text"]
            if "<system-reminder>" in text:
                reminders += len(_REMINDER.findall(text))
                text = _REMINDER.sub("", text)
            text = text.strip()
            if text:
                whole_texts += len(text)
                if kept <= PROMPT_CAP:                     # past the cap a text is measured, not kept
                    texts.append(text[:PROMPT_CAP + 1])
                    kept += len(texts[-1])
        elif kind == "tool_result":
            results.append(_text_of(block.get("content"), TOOL_RESULT_CHARS))
        elif kind in ("image", "document"):
            images += 1
    parts = []
    if results:
        parts.append(f"[tool_result ×{len(results)}] " + " ".join(r for r in results if r)[:TOOL_RESULT_CHARS])
    parts.extend(texts)
    if images:
        parts.append(f"[image ×{images}]")
    if reminders:
        parts.append(f"[system-reminder ×{reminders}]")
    joined = "\n".join(parts)
    return joined, len(joined) - kept + whole_texts


def request_facts(body: dict | None) -> dict:
    """What the queue keeps of a request body: the model asked for, the number of messages and tools, and the last
    user turn bounded as above. Never raises — a body of any other shape is a row of nulls."""
    facts = {"model": None, "messages": None, "tools": None, "prompt": None, "prompt_chars": None}
    if not isinstance(body, dict):
        return facts
    try:
        if isinstance(body.get("model"), str):
            facts["model"] = body["model"][:NAME_CHARS]
        messages = body.get("messages")
        if isinstance(messages, list):
            facts["messages"] = len(messages)
            turn = _last_user_turn(messages)
            if turn is not None:
                text, total = turn
                if len(text) > PROMPT_CAP:
                    text = f"{text[:PROMPT_CAP]}… ({total:,} chars)"
                facts["prompt"], facts["prompt_chars"] = text, total
        if isinstance(body.get("tools"), list):
            facts["tools"] = len(body["tools"])
    except Exception:                         # a view is never allowed to break a turn
        pass
    return facts


# ── the rows ─────────────────────────────────────────────────────────────────────────────────────────────

_NO_PROGRESS = {"deltas": 0, "thinking": 0, "text": 0, "tool": 0, "tool_uses": 0, "prompt_tokens": None,
                "cached_tokens": None, "prompt_from": None, "output_tokens": None, "start_at": None,
                "stop_reason": None}


def _progress(turn) -> dict:
    return turn.tap.progress() if turn.tap is not None else dict(_NO_PROGRESS)


def _ago(now: float, at: float | None) -> float | None:
    return None if at is None else round(max(0.0, now - at), 1)


def _rate(n, seconds) -> float | None:
    return round(n / seconds, 1) if n and seconds is not None and seconds >= MIN_RATE_S else None


def _figures(turn, progress: dict, now: float) -> dict:
    """The prompt, prefill and decode figures every row carries, live or finished. `start` is when `message_start`
    arrived (the tap's clock is the relay's monotonic one), else the first byte — a stream with no `message_start`."""
    prompt, cached, source = progress["prompt_tokens"], progress["cached_tokens"], progress["prompt_from"]
    counted = getattr(turn, "counted", None) or {}
    if prompt is None and counted.get("tokens") is not None:   # the engine's count until its usage says (prompt_count.py)
        prompt, source = counted["tokens"], prompt_count.SOURCE
    start = progress["start_at"] if progress["start_at"] is not None else turn.first_at
    prefill = None if start is None or turn.sent_at is None else max(0.0, start - turn.sent_at)
    decode = None if start is None else max(0.0, now - start)
    return {"prompt_tokens": prompt, "cached_tokens": cached, "prompt_from": source,
            "prompt_error": counted.get("error") if prompt is None else None,
            "cached_share": round(cached / prompt, 3) if prompt and cached is not None else None,
            "prefill_s": None if prefill is None else round(prefill, 3),
            "prefill_tok_s": round(prompt / prefill, 1) if prompt and prefill else None,
            "decode_s": None if decode is None else round(decode, 1),
            "delta_rate": _rate(progress["deltas"], decode),
            "deltas": progress["deltas"], "thinking": progress["thinking"], "text": progress["text"],
            "tool": progress["tool"], "tool_uses": progress["tool_uses"],
            "output_tokens": progress["output_tokens"], "stop_reason": progress["stop_reason"]}


def _identity(turn) -> dict:
    facts = turn.facts
    return {"id": turn.id, "started_at": turn.started_at, "client": turn.client_kind,
            "client_detail": turn.client_detail, "session": turn.session, "model": facts["model"],
            "messages": facts["messages"], "tools": facts["tools"], "box": turn.box}


#: A row built outside `queue_view` (a kill's facts) is given no share of any window: said, not a number.
_UNSHARED = {"tok_s": None, "window_events": None, "tok_s_why": "not apportioned outside the queue view"}


def live_row(turn, now: float, state: str, inferred: dict | None = None, decode: dict | None = None) -> dict:
    """One turn the engine is working on. `now` is the monotonic clock every turn timestamp is on; `inferred` is the
    FCFS split's evidence for a `prefill`/`queued` row: how many no-byte turns, and the engine's waiting gauge;
    `decode` its derived share of the engine's window (`_shares`)."""
    figures = _figures(turn, _progress(turn), now)
    return {**_identity(turn), "age_s": _ago(now, turn.started), "state": state, **(decode or _UNSHARED),
            "phase_s": _ago(now, turn.first_at if state == STREAMING else turn.sent_at),
            "inferred": inferred, "backend": turn.backend, "path": turn.path,
            "prompt": turn.facts["prompt"], "prompt_chars": turn.facts["prompt_chars"], **figures,
            "bytes": turn.sent, "last_byte_s": _ago(now, turn.last_at), "keepalives": turn.keepalives,
            # THE CALLER'S PRIORITY (router.PRIORITY_HEADER, or the page's toggle) and why it cannot move, if not.
            "priority": getattr(turn, "priority", None),
            "priority_locked": priority_lock(turn.path, getattr(turn, "committed", False))}


def outcome_of(turn) -> tuple[str, str | None]:
    """How a finished turn ended: (`done` | `killed` | `client_gone` | `failed`, why). `killed` only where the kill
    TOOK EFFECT (`kill_answered`): one that landed after the turn had ended on its own leaves its own outcome."""
    from vllm_lb.usage_tap import ERROR_EVENT    # the tap's own word for an `error` event in the stream
    if getattr(turn, "kill_answered", False):
        return KILLED, (getattr(turn, "killed", None) or {}).get("why")
    if turn.client_gone:
        return CLIENT_GONE, None
    if turn.error:
        return FAILED, turn.error
    if turn.status is None:
        return FAILED, "no answer"
    if turn.status >= 400:
        return FAILED, f"HTTP {turn.status}"
    if turn.anthropic_sse and not turn.ended:
        return FAILED, "no message_stop"
    if turn.tap is not None and turn.tap.verdict == ERROR_EVENT:
        return FAILED, "error event"
    return DONE, None


def finished_row(turn, now: float, wall: float) -> dict:
    """One finished turn, kept in `recent`: its outcome, the same figures, the engine's own counts where it gave them
    (a counted Usage), its decode rate as the tap timed it, and a SNIPPET_CHARS snippet — nothing more."""
    from vllm_lb.usage_tap import Usage
    outcome, why = outcome_of(turn)
    progress = _progress(turn)
    usage = turn.tap.verdict if turn.tap is not None else None
    if isinstance(usage, Usage):
        progress["output_tokens"] = usage.output_tokens
        if usage.prompt_tokens is not None:
            progress["prompt_tokens"] = usage.prompt_tokens
    figures = _figures(turn, progress, now)
    prompt = turn.facts["prompt"]
    if prompt is not None and len(prompt) > SNIPPET_CHARS:
        prompt = prompt[:SNIPPET_CHARS] + "…"
    # `backend`, as the live row carries it: the panel scopes its Requests card to one server's rows, finished included.
    return {**_identity(turn), "backend": turn.backend, "ended_at": round(wall, 3), "duration_s": _ago(now, turn.started),
            "outcome": outcome, "why": why, "status": turn.status, "prompt": prompt, **figures,
            "decode_tok_s": round(usage.decode_tok_s, 1)
            if isinstance(usage, Usage) and usage.decode_tok_s is not None else None,
            "bytes": turn.sent, "keepalives": turn.keepalives}


def _split(no_bytes: list, waiting: int | None) -> list[tuple[object, str, dict | None]]:
    """One backend's no-byte turns, oldest first, as (turn, state, evidence): FCFS — the oldest N − W prefilling."""
    ordered = sorted(no_bytes, key=lambda t: (t.sent_at or t.started, t.started))
    if waiting is None:
        return [(t, SENT, None) for t in ordered]
    evidence = {"no_bytes": len(ordered), "waiting": waiting}
    prefilling = max(0, len(ordered) - waiting)
    return [(t, PREFILL if i < prefilling else QUEUED, evidence) for i, t in enumerate(ordered)]


# ── why the queued wait: the engine's own admission rule, over its own figures ─────────────────────────────

#: The verdicts. vLLM's v1 scheduler admits a waiting request only while a seat is free (`max_num_running_reqs`, the
#: launch's max_num_seqs) and only when its blocks fit (`allocate_slots` → None stops the loop at that request: FCFS,
#: so the head of the queue holds every request behind it); a request whose KV is still being transferred (the RAM
#: tier's reload) is `deferred`, skipped meanwhile.
WAIT_SEATS, WAIT_KV, WAIT_DEFERRED = "seats", "kv", "deferred"
#: The engine figures the verdict reads, as the probe names them (probe.SCRAPE_KEYS) — all off ONE scrape.
ADMISSION_KEYS = ("running", "waiting", "waiting_capacity", "waiting_deferred", "kv_cache_usage", "kv_pool_blocks",
                  "kv_block_tokens", "kv_max_concurrency")


def _whole(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v >= 0


def _real(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def admission(facts: dict | None) -> dict:
    """WHY THIS SERVER'S QUEUED TURNS WAIT, from the engine's own figures and nothing else — `facts`: its scrape's
    ADMISSION_KEYS, `max_model_len` (its /v1/models) and `seats` (the max_num_seqs it was launched with, where this LB
    knows it; a classic vLLM publishes none). `wait` is one of:

      * WAIT_SEATS    running ≥ seats: a turn starts when one ends.
      * WAIT_KV       the free blocks are fewer than the least ANY request takes: its fixed blocks (a hybrid model's
                      linear-attention state) + one block of context. Free = the usable blocks (num_gpu_blocks less
                      vLLM's null block — 0.9878048780487805 = 162/164 on a 165-block pool) × (1 − kv_cache_usage);
                      fixed = num_gpu_blocks / kv_cache_max_concurrency (the blocks a full-window request takes, the
                      engine's own figure) − ceil(max_model_len / block_size). With fixed unknown only 0 free proves it.
      * WAIT_DEFERRED every waiting request is the engine's `deferred` (a KV transfer, a LoRA budget, a blocked status).
      * None          not proven by these figures — `why` says which they are: free blocks and a free seat mean the
                      head of the queue needs more blocks than are free (its context, a prefix reloaded from a RAM
                      tier) or the step's token budget is spent, and the engine publishes neither per request.

    Every figure it read is returned beside the verdict, None where not read."""
    f = dict(facts or {})
    out = {"wait": None, "why": "", "running": f.get("running"), "waiting": f.get("waiting"),
           "waiting_capacity": f.get("waiting_capacity"), "waiting_deferred": f.get("waiting_deferred"),
           "seats": f.get("seats") if _whole(f.get("seats")) and f.get("seats") > 0 else None,
           "kv_pct": None, "kv_usable_blocks": None, "kv_free_blocks": None, "kv_free_tokens": None,
           "kv_fixed_blocks": None, "kv_least_blocks": None, "mean_context": None}
    running, waiting, seats = out["running"], out["waiting"], out["seats"]
    if not (_whole(running) and _whole(waiting)):
        out["why"] = "the engine's /metrics did not answer: no running or waiting count"
        return out
    usage, blocks, block_tokens = f.get("kv_cache_usage"), f.get("kv_pool_blocks"), f.get("kv_block_tokens")
    conc, ctx = f.get("kv_max_concurrency"), f.get("max_model_len")
    kv_known = _real(usage) and 0 <= usage <= 1 and _whole(blocks) and blocks > 1
    if kv_known:
        usable = blocks - 1
        free = round(usable * (1.0 - usage))
        out.update(kv_pct=round(usage * 100, 2), kv_usable_blocks=usable, kv_free_blocks=free)
        if _whole(block_tokens) and block_tokens > 0:
            out["kv_free_tokens"] = free * block_tokens
        fixed = None
        if _real(conc) and conc > 0 and _whole(ctx) and ctx > 0 and _whole(block_tokens) and block_tokens > 0:
            fixed = round(blocks / conc) - -(-ctx // block_tokens)
            fixed = fixed if fixed >= 0 else None
        out["kv_fixed_blocks"] = fixed
        out["kv_least_blocks"] = (fixed or 0) + 1
        if running > 0 and fixed is not None and _whole(block_tokens) and block_tokens > 0:
            out["mean_context"] = max(0, (usable - free) // running - fixed) * block_tokens
    seat_text = f"{running} of {seats} seats" if seats else f"{running} running (max_num_seqs not published)"
    kv_text = (f"{out['kv_free_blocks']} of {out['kv_usable_blocks']} KV blocks free, a request takes ≥ "
               f"{out['kv_least_blocks']}" + ("" if out["kv_fixed_blocks"] is not None else
                                               " (its fixed blocks not published)")
               if kv_known else "KV usage or pool not read")
    reasons = (f"; engine: {out['waiting_capacity']} capacity · {out['waiting_deferred']} deferred"
               if _whole(out["waiting_capacity"]) and _whole(out["waiting_deferred"]) else "")
    if waiting == 0:
        out["why"] = "the engine reports none waiting"
        return out
    if seats and running >= seats:
        out["wait"], out["why"] = WAIT_SEATS, f"{seat_text}: a turn starts when one ends · {kv_text}{reasons}"
    elif kv_known and (out["kv_free_blocks"] < out["kv_least_blocks"] if out["kv_fixed_blocks"] is not None
                       else out["kv_free_blocks"] == 0):
        out["wait"], out["why"] = WAIT_KV, f"{kv_text}: no request fits · {seat_text}{reasons}"
    elif out["waiting_capacity"] == 0 and _whole(out["waiting_deferred"]) and out["waiting_deferred"] == waiting:
        out["wait"] = WAIT_DEFERRED
        out["why"] = (f"all {waiting} deferred by the engine (a KV transfer, a LoRA budget, a blocked status) · "
                      f"{seat_text} · {kv_text}")
    else:
        out["why"] = (f"{seat_text} · {kv_text}: neither full — the queue's head needs more blocks than are free, or "
                      f"the step's token budget is spent; the engine says neither per request{reasons}")
    return out


def _events_in(turn, start: float, end: float) -> int:
    tap = getattr(turn, "tap", None)
    try:
        return tap.events_between(start, end) if tap is not None else 0
    except Exception:                         # a view is never allowed to break a turn
        return 0


def _shares(window: dict | None, rows: list, others: list, now: float) -> tuple[dict, dict]:
    """ONE server's GENERATED tok/s apportioned (module docstring, `DERIVED_DECODE`): `rows` the turns placed as rows
    (listed or past the cap), `others` the ones that are not — finishing, or ended within EVENTS_KEPT_S. Returns
    ({id(turn): its decode figures}, the server's `decode.backends` entry, its row sums left to the caller)."""
    g = window or {}
    tok_s, span, end = g.get("tok_s"), g.get("window_s"), g.get("end_at")
    why = None
    if window is None:
        why = NOT_PROBED
    elif tok_s is None or span is None or end is None:
        why = g.get("why") or NOT_PROBED
    elif end - span < now - EVENTS_KEPT_S:
        why = OLDER_THAN_KEPT
    counts = {} if why else {id(t): _events_in(t, end - span, end) for t in rows + others}
    events = sum(counts.values())
    if why is None and events == 0 and tok_s > 0:
        why = NO_EVENT_IN_WINDOW
    per_turn, ended_sum = {}, 0.0
    for t in rows + others:
        n = counts.get(id(t))
        share = None if why else (tok_s * n / events if events else 0.0)
        per_turn[id(t)] = {"tok_s": None if share is None else round(share, 1), "window_events": n, "tok_s_why": why}
    for t in others:
        ended_sum += per_turn[id(t)]["tok_s"] or 0.0
    entry = {"engine_tok_s": None if why else round(tok_s, 1), "window_s": None if why else round(span, 2),
             "events": None if why else events, "rows_tok_s": None, "unlisted_tok_s": None,
             "ended_tok_s": None if why else round(ended_sum, 1), "why": why}
    return per_turn, entry


def queue_view(turns, recent, backends, now: float, engines: dict | None = None, rates: dict | None = None,
               ended=()) -> dict:
    """/lb/status `queue`: the rows — streaming, then prefill, then queued (then `sent`, split unknown), oldest first
    in each, at most INFLIGHT_ROWS (`inflight_total` counts all) — the counts per state, held and recent included,
    the finished newest first, and the engine's own aggregates summed over the live backends whose /metrics answered
    (`depth_known`), null when none did. `backends` (by url, every live one): its rows per state and why its queued
    wait (`admission` over `engines[url]`, the facts the caller read for it). `rates` (by url): each server's
    GENERATED window (Throughput `generated_window`); `ended`: the turns that ended within EVENTS_KEPT_S — with them
    every row's derived `tok_s` and `decode`, the sums beside the engine's figure (module docstring)."""
    gauges = {b.url: b.waiting for b in backends if b.live and b.depth_known}
    streaming, no_bytes, held, finishing = [], {}, 0, []
    for t in turns:
        if t.backend == WEBSEARCH:
            continue
        if t.holding or t.backend is None:
            held += 1
        elif t.first_at is None:
            no_bytes.setdefault(t.backend, []).append(t)
        elif t.tap is None or t.tap.progress()["stop_reason"] is None:
            streaming.append(t)
        else:
            finishing.append(t)
    placed = [(t, STREAMING, None) for t in sorted(streaming, key=lambda t: t.started)]
    for url, group in no_bytes.items():
        placed += _split(group, gauges.get(url))
    placed.sort(key=lambda p: ROW_STATES.index(p[1]))     # stable: each group keeps its oldest-first order
    counts = {state: 0 for state in ROW_STATES}
    for _, state, _ in placed:
        counts[state] += 1
    counts.update(held=held, recent=len(recent))
    known = [b for b in backends if b.live and b.depth_known]
    engines = engines or {}
    per = {}
    for b in backends:
        if not b.live:
            continue
        rows = {state: 0 for state in ROW_STATES}
        for t, state, _ in placed:
            if t.backend == b.url:
                rows[state] += 1
        per[b.url] = {**admission(engines.get(b.url) if b.depth_known else None), "rows": rows}
    engine = None
    if known:
        engine = {"running": sum(b.running for b in known), "waiting": sum(b.waiting for b in known),
                  "backends": len(known)}
        for key in ("waiting_capacity", "waiting_deferred"):
            vals = [per[b.url][key] for b in known]
            engine[key] = sum(vals) if all(_whole(v) for v in vals) else None
    # THE DECODE SHARES, per live server: every turn relayed to it, row or not, against its one GENERATED window.
    # Each row's figure is rounded as shown, and the sums are of the shown figures: what the page adds is what it says.
    rates = rates or {}
    listed = {id(t) for t, _, _ in placed[:INFLIGHT_ROWS]}
    shares, decode_per = {}, {}
    for b in backends:
        if not b.live:
            continue
        rows_b = [t for t, _, _ in placed if t.backend == b.url]
        others = [t for t in finishing + list(ended or ()) if t.backend == b.url]
        got, entry = _shares(rates.get(b.url), rows_b, others, now)
        shares.update(got)
        if entry["why"] is None:
            entry["rows_tok_s"] = round(sum(got[id(t)]["tok_s"] for t in rows_b if id(t) in listed), 1)
            entry["unlisted_tok_s"] = round(sum(got[id(t)]["tok_s"] for t in rows_b if id(t) not in listed), 1)
        decode_per[b.url] = entry
    measured = [e for e in decode_per.values() if e["why"] is None]

    def total(key):
        return round(sum(e[key] for e in measured), 1) if measured else None
    decode = {"backends": decode_per, "engine_tok_s": total("engine_tok_s"), "rows_tok_s": total("rows_tok_s"),
              "ended_tok_s": total("ended_tok_s"), "unlisted_tok_s": total("unlisted_tok_s"),
              "window_s": max((e["window_s"] for e in measured), default=None),
              "unmeasured": len(decode_per) - len(measured),
              "why": None if measured else next((e["why"] for e in decode_per.values()), SERVER_NOT_LIVE),
              "derived": DERIVED_DECODE}
    unshared = {"tok_s": None, "window_events": None, "tok_s_why": SERVER_NOT_LIVE}
    return {"inflight": [live_row(t, now, state, why, shares.get(id(t), unshared))
                         for t, state, why in placed[:INFLIGHT_ROWS]],
            "inflight_total": len(placed), "counts": counts, "recent": list(reversed(recent)), "engine": engine,
            "backends": per, "decode": decode,
            "limits": {"rows": INFLIGHT_ROWS, "recent": RECENT_ROWS, "prompt_chars": PROMPT_CAP,
                       "snippet_chars": SNIPPET_CHARS}}
