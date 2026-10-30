"""Anthropic's SERVER-side `web_search` tool, performed by the LB.

Claude Code's `WebSearch` is not run on the client. Its implementation issues a second
`POST /v1/messages` — one user turn, `"Perform a web search for the query: <q>"`, and
`tools:[{type:"web_search_20250305",name:"web_search"}]` — then reads only two block types
out of the answer: `server_tool_use` (counted) and `web_search_tool_result` (each entry
mapped to `{title, url}`). On api.anthropic.com the search happens inside the API; a
third-party endpoint cannot do it, and box-c's ollama answers the block honestly with
`web_search_tool_result_error: unavailable`.

So the LB answers it. No model is involved: the query is in the prompt, the search runs
against a local SearXNG, and the two blocks are synthesised — which is all the client reads.
The snippet is folded into the title because `title` and `url` are the only fields that
survive the client's own mapping, and a title alone is a poor result for a local model.

One search produces one `SearchOutcome`: the results, or a NAMED failure. That distinction is
the whole of `worklog/vllm-websearch-resilience.md` — for four days every search came back
`results: []` while every engine was suspended or timing out, and an empty success tells the
model the web has nothing rather than that the search failed.
"""
from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

#: The preamble Claude Code wraps the query in before sending it to the model.
_PREAMBLE = "Perform a web search for the query:"
#: The paths that may carry the tool. `/v1/messages/count_tokens` must NOT be intercepted —
#: the client sizes the very same body there and expects a token count back.
_MESSAGES_PATH = "/v1/messages"

# ── the failure codes ────────────────────────────────────────────────────────
#
# Anthropic's own, for `web_search_tool_result_error`: `too_many_requests`,
# `invalid_tool_input`, `max_uses_exceeded`, `query_too_long`, `request_too_large`,
# `unavailable` (platform.claude.com → web search tool → Errors). The client prints whatever
# code it is handed (`Web search error: ${E.content.error_code}`), so emitting one Anthropic
# does not define would reach the operator as noise and match nothing in the docs.

#: Every engine refused us: rate limited, access denied, CAPTCHA'd — or suspended for one of
#: those (SearXNG's `search.suspended_times`, minutes to days).
TOO_MANY_REQUESTS = "too_many_requests"
#: Nothing answered: timeouts, connection/parse errors, SearXNG itself unreachable.
UNAVAILABLE = "unavailable"
#: The request carried no query to search for.
INVALID_TOOL_INPUT = "invalid_tool_input"
#: What this LB may emit. A test pins it against Anthropic's documented set.
ERROR_CODES = frozenset({TOO_MANY_REQUESTS, UNAVAILABLE, INVALID_TOOL_INPUT})

#: What SearXNG says when an engine REFUSED us rather than broke — `searx/webutils.py`'s
#: `exception_classname_to_text`, which is a closed set. `get_translated_errors` prefixes
#: `Suspended: ` while the engine is serving out its suspension, and a suspension is itself
#: "you are asking too much": SearXNG only imposes one after a refusal or repeated failures.
_REFUSED = ("too many requests", "access denied", "captcha")
_SUSPENDED = "suspended"


def _refused(message: str) -> bool:
    lowered = message.lower()
    return lowered.startswith(_SUSPENDED) or any(r in lowered for r in _REFUSED)


@dataclass(frozen=True)
class SearchOutcome:
    """What ONE search produced: results, or a named failure — and who did not answer.

    An empty `results` with nothing unresponsive is an honest "the web has nothing", which is
    also Anthropic's rule for this block: *"A search that succeeds but matches no results
    returns an empty `content` list, not an error."* An empty `results` because every engine
    was suspended or timed out is a FAILURE, and the client is told so.
    """

    results: tuple[dict, ...] = ()
    error_code: str | None = None
    #: `(engine, what SearXNG said)`, kept whether or not the search failed: it is the only
    #: thing that says WHY an answer was thin, and it is what the LB's log line prints.
    unresponsive: tuple[tuple[str, str], ...] = field(default=())

    @classmethod
    def failure(cls, error_code: str, unresponsive: tuple[tuple[str, str], ...] = ()) -> "SearchOutcome":
        return cls(error_code=error_code, unresponsive=tuple(unresponsive))

    @property
    def failed(self) -> bool:
        return self.error_code is not None

    def detail(self) -> str:
        """The LB's log line. `'<query>' → 0 results` said nothing for four days; while
        nothing comes back, the engines and SearXNG's own words for them are the whole story."""
        if self.results:
            return f"{len(self.results)} results"
        head = self.error_code or "0 results"
        if self.unresponsive:
            return head + " — unresponsive: " + ", ".join(f"{n}: {m}" for n, m in self.unresponsive)
        return head + " — every engine answered"


def is_web_search_request(path: str, body: dict | None) -> bool:
    """True for a `/v1/messages` request whose tools carry Anthropic's server-side block."""
    if urllib.parse.urlsplit(path).path.rstrip("/") != _MESSAGES_PATH:
        return False
    if not isinstance(body, dict):
        return False
    tools = body.get("tools")
    if not isinstance(tools, list):
        return False
    return any(isinstance(t, dict) and str(t.get("type") or "").startswith("web_search_")
               for t in tools)


def extract_query(body: dict) -> str | None:
    """The search query: the last user turn's text, minus the client's preamble."""
    texts: list[str] = []
    for msg in reversed(body.get("messages") or []):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            texts.extend(b["text"] for b in content
                         if isinstance(b, dict) and b.get("type") == "text"
                         and isinstance(b.get("text"), str))
        if texts:
            break
    text = "\n".join(t.strip() for t in texts).strip()
    if not text:
        return None
    lowered = text.lower()
    idx = lowered.rfind(_PREAMBLE.lower())
    if idx != -1:
        text = text[idx + len(_PREAMBLE):].strip()
    return text or None


# ── the search ───────────────────────────────────────────────────────────────

def results_from_searxng(payload: dict, *, max_results: int, snippet_chars: int) -> list[dict]:
    """SearXNG's JSON → the `{title, url}` pairs the client keeps, snippet folded in.

    This answers "what results are in this payload" and nothing else. Whether an empty list
    means the search failed is `outcome_from_searxng`'s call, which is the only place that
    judges it.
    """
    raw = payload.get("results")
    if raw is None:
        raw = []
    if not isinstance(raw, list):
        # Malformed must NOT read as "the web has nothing": the caller turns a raise into
        # `unavailable`, which is the truth, while an empty list would be a lie.
        raise ValueError(f"searxng returned {type(raw).__name__} for 'results'")
    out: list[dict] = []
    for r in raw:
        if not isinstance(r, dict):
            continue
        url = (r.get("url") or "").strip()
        if not url:
            continue
        title = (r.get("title") or url).strip()
        snippet = " ".join((r.get("content") or "").split())
        if snippet_chars > 0 and snippet:
            if len(snippet) > snippet_chars:
                snippet = snippet[:snippet_chars].rstrip() + "…"
            title = f"{title} — {snippet}"
        out.append({"title": title, "url": url})
        if len(out) >= max_results:
            break
    return out


def _unresponsive_from_searxng(payload: dict) -> tuple[tuple[str, str], ...]:
    """SearXNG's `unresponsive_engines`: `[["brave", "Suspended: too many requests"], …]`."""
    raw = payload.get("unresponsive_engines")
    if not isinstance(raw, list):
        return ()
    out = []
    for entry in raw:
        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
            out.append((str(entry[0]), str(entry[1])))
    return tuple(out)


def outcome_from_searxng(payload: dict, *, max_results: int, snippet_chars: int) -> SearchOutcome:
    """SearXNG's answer → ONE outcome. The only place that decides whether a search failed."""
    results = tuple(results_from_searxng(payload, max_results=max_results,
                                         snippet_chars=snippet_chars))
    unresponsive = _unresponsive_from_searxng(payload)
    if results:
        # Breadth doing its job: some engine carried the query even if another was blocked.
        return SearchOutcome(results=results, unresponsive=unresponsive)
    if unresponsive:
        # Nothing came back AND engines did not answer. Being throttled is the actionable
        # one, so one refusal names the whole query even among plain breakages.
        code = TOO_MANY_REQUESTS if any(_refused(m) for _, m in unresponsive) else UNAVAILABLE
        return SearchOutcome(error_code=code, unresponsive=unresponsive)
    return SearchOutcome()


def searxng_search(query: str, cfg) -> SearchOutcome:
    """Ask the local SearXNG for `query`. Raises on any transport/format failure."""
    url = cfg.search_url + ("&" if "?" in cfg.search_url else "?") + urllib.parse.urlencode(
        {"q": query, "format": "json", "language": cfg.search_language,
         "safesearch": "0"})
    req = urllib.request.Request(url, headers={"Accept": "application/json",
                                               "User-Agent": "vllm-lb/1.0"})
    with urllib.request.urlopen(req, timeout=cfg.search_timeout_s) as resp:
        payload = json.loads(resp.read().decode("utf-8", "replace"))
    return outcome_from_searxng(payload, max_results=cfg.search_max_results,
                                snippet_chars=cfg.search_snippet_chars)


# ── the answer ───────────────────────────────────────────────────────────────

def build_message(query: str, outcome: SearchOutcome, *, model: str) -> dict:
    """The Anthropic message the client harvests: one `server_tool_use`, one result block.

    A failure makes that block's `content` a single OBJECT rather than a list — the exact
    shape Anthropic's API sends, and the one the client branches on
    (`if(!Array.isArray(E.content)) push("Web search error: "+E.content.error_code)`).
    """
    use_id = "srvtoolu_" + uuid.uuid4().hex[:24]
    if outcome.failed:
        content: object = {"type": "web_search_tool_result_error",
                           "error_code": outcome.error_code}
    else:
        content = [{"type": "web_search_result", "title": r["title"], "url": r["url"],
                    "encrypted_content": "", "page_age": None} for r in outcome.results]
    return {
        "id": "msg_" + uuid.uuid4().hex[:24],
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [
            {"type": "server_tool_use", "id": use_id, "name": "web_search",
             "input": {"query": query}},
            {"type": "web_search_tool_result", "tool_use_id": use_id, "content": content},
        ],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": 0,
                  "server_tool_use": {"web_search_requests": 1}},
    }


def sse_events(message: dict) -> list[tuple[str, dict]]:
    """The same message as the event sequence the SDK's accumulator rebuilds it from."""
    head = {k: v for k, v in message.items() if k not in ("content", "stop_reason", "usage")}
    events: list[tuple[str, dict]] = [
        ("message_start", {"type": "message_start",
                           "message": {**head, "content": [], "stop_reason": None,
                                       "usage": {"input_tokens": 0, "output_tokens": 0}}}),
    ]
    for i, block in enumerate(message["content"]):
        if block["type"] == "server_tool_use":
            events.append(("content_block_start", {
                "type": "content_block_start", "index": i,
                "content_block": {**block, "input": {}}}))
            events.append(("content_block_delta", {
                "type": "content_block_delta", "index": i,
                "delta": {"type": "input_json_delta",
                          "partial_json": json.dumps(block["input"])}}))
        else:
            events.append(("content_block_start", {
                "type": "content_block_start", "index": i, "content_block": block}))
        events.append(("content_block_stop", {"type": "content_block_stop", "index": i}))
    events.append(("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": message["stop_reason"], "stop_sequence": None},
        "usage": message["usage"]}))
    events.append(("message_stop", {"type": "message_stop"}))
    return events


def sse_body(message: dict) -> bytes:
    return b"".join(f"event: {name}\ndata: {json.dumps(data)}\n\n".encode()
                    for name, data in sse_events(message))
