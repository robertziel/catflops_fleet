"""PROMPT TOKENS BEFORE THE FIRST BYTE: the engine's own count of an OpenAI chat turn's prompt, asked at send time.

An Anthropic stream reports its prompt in `message_start`; an OpenAI `/v1/chat/completions` stream only in its final
usage chunk (usage_tap.py), so the REQUESTS card read "—" for a turn from the `openai` SDK (User-Agent
`OpenAI/Python`) until it finished. `continuous_usage_stats` is NOT the way: a client whose drop detection reads any
usage object as "the provider finished" would then miss a cut stream.

The engine counts it: vLLM's and SGLang's `POST /tokenize` take the same `messages` + `tools` +
`chat_template_kwargs` and apply the served chat template — measured 2026-10-22 on the pair's SGLang: one "hello"
53 tokens, 311 with one tool, 13 with `enable_thinking: false`, so each of the three changes the figure. This is
the ENGINE'S figure, never an estimate: a count that fails is said (`error`), never replaced by a guess, and the
final usage chunk supersedes it (request_queue._figures).
"""
from __future__ import annotations

import http.client
import json
import urllib.parse

PATH = "/tokenize"
#: What `prompt_from` says of a figure counted here.
SOURCE = "tokenize"
#: What the served chat template reads off a chat body; nothing else is sent.
FIELDS = ("model", "messages", "tools", "chat_template_kwargs")
TIMEOUT_S = 30.0
ERROR_CHARS = 160


def request_of(body) -> dict | None:
    """The tokenize request for a chat body (as the engine will get it, after the LB's rewrites), else None."""
    if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
        return None
    return {k: body[k] for k in FIELDS if body.get(k) is not None}


def count(url: str, request: dict) -> dict:
    """`{"tokens": n}` from the engine at `url`, else `{"error": why}`. Never raises."""
    u = urllib.parse.urlsplit(url)
    conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=TIMEOUT_S)
    try:
        conn.request("POST", PATH, body=json.dumps(request).encode(), headers={"Content-Type": "application/json"})
        r = conn.getresponse()
        data = r.read()
        if r.status != 200:
            return {"error": f"{PATH} HTTP {r.status}: {data[:ERROR_CHARS].decode('utf-8', 'replace')}"}
        n = json.loads(data).get("count")
        return {"tokens": n} if isinstance(n, int) else {"error": f"{PATH} answered no integer count"}
    except Exception as exc:                  # a view is never allowed to break a turn
        return {"error": f"{PATH} {type(exc).__name__}: {exc}"[:ERROR_CHARS]}
    finally:
        conn.close()
