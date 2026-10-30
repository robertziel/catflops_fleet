"""KILLING ONE REQUEST: what the operator's kill (POST /lb/requests/<id>/kill — the REQUESTS card's red ×, held 1 s)
does to the ENGINE working on a turn and what it tells the CLIENT whose turn it ends.

THE MECHANISM is one for every engine path the LB proxies — a classic box, the fallback, the cluster head (a pair, a
local node and an observed node are all that one backend to the relay): the turn's own connection to its engine is
closed (`proxy._Turn.kill`, `socket.shutdown(SHUT_RDWR)`). A turn the LB still HOLDS has no connection — nothing was
sent — and is answered where it waits; it never reaches an engine.

THE ENGINE aborts a request whose HTTP client disconnects — queued, prefilling or streaming — and frees its seat and
KV (read 2026-10-12 from the sources the fleet runs):
  * vLLM 2eb491e (`vllm/vllm-openai@sha256:514cd427…`, `_version.py` 0.31.1rc1.dev273+g2eb491e18; paths under
    vllm/): `@with_cancellation` on `create_chat_completion` (entrypoints/openai/chat_completion/api_router.py:52-54)
    and `create_messages` (entrypoints/anthropic/api_router.py:61-63) races the handler against
    `listen_for_disconnect` and cancels it on `http.disconnect` (entrypoints/serve/utils/api_utils.py:36-49, 80-87);
    a stream is cancelled by starlette's own disconnect listener (starlette responses.py:269-282, uvicorn spec 2.3);
    the cancelled generator calls `self.abort(request_id)` (v1/engine/async_llm.py:772-777, 904-912) →
    `scheduler.finish_requests(…, FINISHED_ABORTED)` (v1/engine/core.py:549-554), which takes a WAITING request out
    of the queue (`self.waiting.remove_requests`, v1/core/sched/scheduler.py:2694-2707) and frees its KV
    (`_free_request` → `kv_cache_manager.free`, :2720, :2781). uvicorn reads our FIN as the disconnect
    (`eof_received` returns None → `connection_lost` → `http.disconnect`, httptools_impl.py:116-136, 574-578).
  * SGLang 708f51e (the cluster image's base `lmsysorg/sglang@sha256:00205b89…` = nightly-cu134-20260909-708f51e,
    worklog/sglang-27b-tp2.md:21; paths under python/sglang/srt/): every 4 s a request still waiting for output
    checks `request.is_disconnected()` and calls `abort_request` (managers/tokenizer_manager.py:1749-1765,
    SGLANG_REQUEST_STATE_WAIT_TIMEOUT = 4, environ.py:1358) — queued and prefilling, stream or not; a cancelled
    stream generator aborts too (:836-845, :3517-3519), and every stream carries `create_abort_task` (:2187-2197,
    entrypoints/openai/serving_chat.py:1656-1673, entrypoints/anthropic/serving.py:825-836); the scheduler pops an
    aborted request from its waiting queue (managers/scheduler.py:5181-5193) and finishes a running one with
    FINISH_ABORT, freeing its KV (:5309-5324). So SGLang lets go of a killed QUEUED turn within ≤ 4 s.

THE CLIENT is told in its API's own shape, and in a way no client retries:
  * Before any byte of the answer: HTTP 400, `x-should-retry: false`. The SDKs retry only 408, 409, 429 and ≥ 500,
    and obey `x-should-retry` before any status (anthropic 1.13.0 `_base_client.py:863-896`, openai 3.29.0
    `_base_client.py:885-925`, @anthropic-ai/sdk 0.133.0 `src/client.ts:1588-1626`, openai-node 7.32.0
    `src/client.ts:2074-2094`). Claude Code 2.1.296 runs with SDK retries off and decides itself (binary offsets):
    it retries 401 (re-auth) and, under some auth setups, 403 (@218337904); honours `x-should-retry: false`
    (@218338257); retries 408/409/429/≥ 500 (@218338574); re-sends a 404 without streaming (@218340549); and re-runs
    any status outside {401,407,429,404,403,413} on a fallback model unless the error is a 400 whose `error.type` is
    `policy_blocked` (`rRs`, @209501745) — which also skips its strip-a-beta-header retry of a 400 (@218423002).
    Its embedded gateway guide states that very contract for a request a gateway refuses: "HTTP 400.
    x-should-retry: false. {"type":"error","request_id":…,"error":{"type":"policy_blocked","message":…}}" (@236539300).
    So: 400 + `policy_blocked` for Anthropic, 400 + the OpenAI error object for OpenAI.
  * Mid-stream (the status line is the client's already): an Anthropic `event: error`; an OpenAI error chunk then
    `data: [DONE]` — the engines' own mid-stream shapes (vLLM chat_completion/serving.py:926-933, SGLang
    serving_base.py:213-227 + serving_chat.py:2004). Both SDKs raise it while iterating and never re-request a 200
    (anthropic `_streaming.py:139-152`, openai `_streaming.py:94-106`). Claude Code, though, RE-SENDS a status-less
    stream error once without streaming while no content block has finished (@218341941, "Error streaming, falling
    back to non-streaming mode" @218481321) — `policy_blocked` is not checked there — unless the error carries
    `details.error_code: "dlp_request_denied"` (`I3e`, @209501680), which it throws with no retry and no fallback in
    both the stream and its retry loop (@218479075, @218324834), shown as "API Error: <message>" (@216500046). That
    code is Claude Code's one terminal marker for a refused request, so the error carries it — in both cases, one
    shape. The `overloaded_error` the relay sends for a BACKEND failure is exactly what a kill must never send.
"""
from __future__ import annotations

import json
import urllib.parse

#: The status of a kill answered BEFORE any byte of the answer — a held turn, or one queued or prefilling at the
#: engine (the module docstring: the one status no SDK and no Claude Code path retries, re-sends or re-routes).
KILL_STATUS = 400
#: Sent with every kill answered as an HTTP error: both SDKs and Claude Code obey it before any status rule.
NO_RETRY = ("x-should-retry", "false")
#: The error's `type` — a gateway's refusal, Claude Code's `policy_blocked` contract — in both APIs' shapes, and the
#: OpenAI object's `code`.
ANTHROPIC_TYPE = OPENAI_TYPE = "policy_blocked"
OPENAI_CODE = "request_killed"
#: Claude Code's terminal marker for a refused request (`error.details.error_code`): without it a status-less
#: mid-stream error is re-sent once without streaming (module docstring).
NO_RESEND_CODE = "dlp_request_denied"
#: The most of the gate's user name a log line or a row keeps.
WHO_CHARS = 120
#: What a kill reports of a turn, off its REQUESTS row (request_queue.live_row / finished_row).
FACT_KEYS = ("id", "state", "age_s", "client", "client_detail", "session", "model", "box", "backend", "deltas",
             "output_tokens", "prompt_tokens", "bytes")


def is_anthropic(path: str) -> bool:
    """An Anthropic Messages turn (its errors are `{"type": "error", "error": {...}}`); everything else the LB relays
    under /v1/ speaks OpenAI's `{"error": {...}}`."""
    p = urllib.parse.urlsplit(path).path
    return p == "/v1/messages" or p.startswith("/v1/messages/")


def message(rid: str) -> str:
    return (f"request killed by the operator (gateway request {rid}); it is not retried — send it again to run it "
            f"again")


def _error(path: str, rid: str) -> dict:
    """ONE error per API, before the answer and mid-stream alike: Anthropic's with the id the kill's log line is keyed
    by as its `request_id` (Claude Code's guide: the id a user quotes back), OpenAI's four fields."""
    if is_anthropic(path):
        return {"type": "error", "request_id": rid,
                "error": {"type": ANTHROPIC_TYPE, "message": message(rid), "details": {"error_code": NO_RESEND_CODE}}}
    return {"error": {"message": message(rid), "type": OPENAI_TYPE, "param": None, "code": OPENAI_CODE}}


def error_body(path: str, rid: str) -> bytes:
    """The HTTP error body for a kill answered before the answer began, in the turn's API's shape."""
    return json.dumps(_error(path, rid)).encode()


def stream_error(path: str, rid: str) -> bytes:
    """What a stream killed mid-answer ends with, at an event boundary: Anthropic's `event: error`; OpenAI's error
    chunk then `data: [DONE]` — the engines' own mid-stream error shape (module docstring)."""
    data = json.dumps(_error(path, rid)).encode()
    if is_anthropic(path):
        return b"event: error\ndata: " + data + b"\n\n"
    return b"data: " + data + b"\n\ndata: [DONE]\n\n"


def facts_of(row: dict) -> dict:
    """A kill's facts off a REQUESTS row, live or finished (a finished row's `duration_s` is its age)."""
    out = {k: row.get(k) for k in FACT_KEYS}
    if out["age_s"] is None:
        out["age_s"] = row.get("duration_s")
    return out


def why(who: str | None, state: str, age_s) -> str:
    """The finished row's `why` for a killed turn: by whom, and the state and age it was killed in."""
    age = f" {age_s:.0f} s" if isinstance(age_s, (int, float)) else ""
    return f"by {who or 'an unnamed caller'} · {state}{age}"


def describe(f: dict) -> str:
    """The kill's log line, before its `by`: the turn, its state and age, who sent it, what it had produced."""
    age = f"{f['age_s']:.1f} s" if isinstance(f.get("age_s"), (int, float)) else "age unknown"
    client = " ".join(x for x in (f.get("client"), f.get("client_detail")) if x) or "unknown client"
    session = f" · session {f['session']}" if f.get("session") else ""
    out = f"{f['output_tokens']:,} output tokens" if isinstance(f.get("output_tokens"), int) \
        else "output tokens not counted yet"
    prompt = f"prompt {f['prompt_tokens']:,} tokens" if isinstance(f.get("prompt_tokens"), int) \
        else "prompt not reported yet"
    on = f" · on {f['backend']}" + (f" ({f['box']})" if f.get("box") else "") if f.get("backend") else ""
    return (f"request {f['id']} · {f.get('state')} {age} · {client}{session} · {f.get('deltas') or 0} events · "
            f"{out} · {prompt} · {f.get('bytes') or 0:,} B{on}")


def by_line(who: str | None, asked_by: str) -> str:
    return f"by {who} · {asked_by}" if who else f"by an unnamed caller (no X-Vllm-User) · {asked_by}"
