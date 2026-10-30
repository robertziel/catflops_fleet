"""PROMPT TOKENS BEFORE THE FIRST BYTE (vllm_lb/prompt_count.py): an OpenAI /v1/chat/completions stream reports its
prompt only in its final usage chunk, so the REQUESTS card read "—" for an `openai` SDK turn until it finished
(2026-10-22). The engine's own `/tokenize` counts the same messages + tools + template
kwargs at send time; the final usage chunk supersedes it; a failed count is said, never replaced by a guess."""
from __future__ import annotations

import http.client
import json
import threading

from tests.fakes import FakeBackend
from tests.test_request_queue import finished, row, stack, wait  # noqa: F401 — `stack` is the fixture
from vllm_lb import prompt_count
from vllm_lb import request_queue as rq

OPENAI_SDK_UA = "OpenAI/Python 2.8.1"


def openai_body() -> dict:
    return {"model": "qwen3.8:latest", "stream": True, "stream_options": {"include_usage": True},
            "messages": [{"role": "system", "content": "be brief"}, {"role": "user", "content": "say hi"}],
            "tools": [{"type": "function", "function": {"name": "web_search", "parameters": {"type": "object"}}}],
            "chat_template_kwargs": {"enable_thinking": True}, "temperature": 0.6, "max_tokens": 64}


def openai_stream(n: int = 6, prompt: int = 1500) -> list[bytes]:
    def chunk(obj: dict) -> bytes:
        return f"data: {json.dumps(obj)}\n\n".encode()
    deltas = [chunk({"id": "c", "choices": [{"index": 0, "delta": {"reasoning_content": f" t{k}"}}]}) for k in range(n)]
    return (deltas + [chunk({"id": "c", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
                      chunk({"id": "c", "choices": [], "usage": {"prompt_tokens": prompt, "completion_tokens": n}}),
                      b"data: [DONE]\n\n"])


def send(port: int, body: dict, out: dict) -> threading.Thread:
    def go():
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
        conn.request("POST", "/v1/chat/completions", body=json.dumps(body).encode(),
                     headers={"Content-Type": "application/json", "User-Agent": OPENAI_SDK_UA})
        r = conn.getresponse()
        out["r"] = (r.status, r.read())
    t = threading.Thread(target=go, daemon=True)
    t.start()
    return t


def test_the_tokenize_request_carries_what_the_chat_template_reads_and_nothing_else():
    got = prompt_count.request_of(openai_body())
    assert got == {"model": "qwen3.8:latest", "messages": openai_body()["messages"], "tools": openai_body()["tools"],
                   "chat_template_kwargs": {"enable_thinking": True}}
    assert prompt_count.request_of({"model": "m", "prompt": "x"}) is None, "no messages: not a chat turn"
    assert prompt_count.request_of(None) is None


def test_an_openai_stream_shows_the_engines_count_before_its_final_usage_then_the_usage(stack):
    v = FakeBackend()
    v.tokenize_count = 777
    v.stream_chunks, v.stream_delay_s = openai_stream(), 0.15
    proxy, _ = stack(v)
    out: dict = {}
    send(proxy.port, openai_body(), out)
    r = wait(lambda: (x := row(proxy, rq.STREAMING)) and x["prompt_tokens"] is not None and x)
    assert (r["prompt_tokens"], r["prompt_from"]) == (777, prompt_count.SOURCE)
    sent = json.loads(v.tokenized[0])
    assert sent["tools"] and sent["chat_template_kwargs"] == {"enable_thinking": True} and "max_tokens" not in sent
    assert all(q["path"] != prompt_count.PATH for q in v.requests), "the count is not one of the relayed requests"
    wait(lambda: "r" in out)
    done = finished(proxy)[0]
    assert (done["prompt_tokens"], done["prompt_from"]) == (1500, "usage"), "the engine's usage supersedes the count"


def test_a_failed_count_is_said_not_guessed(stack):
    v = FakeBackend()                      # tokenize_count None: the fake answers /tokenize 404
    v.stream_chunks, v.stream_delay_s = openai_stream(n=8), 0.15
    proxy, _ = stack(v)
    send(proxy.port, openai_body(), {})
    r = wait(lambda: (x := row(proxy, rq.STREAMING)) and x["prompt_error"] and x)
    assert r["prompt_tokens"] is None and "404" in r["prompt_error"]
