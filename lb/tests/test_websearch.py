"""S9 — web search: the LB answers Anthropic's SERVER-side `web_search` tool itself.

Claude Code's `WebSearch` tool is not executed on the client. It issues a SECOND
`POST /v1/messages` carrying `tools:[{type:"web_search_20250305",name:"web_search"}]` and
harvests `server_tool_use` + `web_search_tool_result` blocks out of the answer — work the
real Anthropic API does server-side and no third-party backend performs (box-c's ollama
recognises the block and replies `web_search_tool_result_error: unavailable`).

Pinned here: the LB recognises that request, never forwards it, runs the search itself
(SearXNG) and synthesises the two blocks — streaming or not; a search that fails comes back
as the error block the client already knows how to render; everything else still forwards.

Since `worklog/vllm-websearch-resilience.md` one `SearchOutcome` carries what a search
produced — results, or a NAMED failure — and a query no engine answered is a failure rather
than an empty web.
"""
from __future__ import annotations

import http.client
import json
import logging

import pytest

from tests.fakes import FakeBackend
from vllm_lb import websearch
from vllm_lb.config import Config, VllmBox
from vllm_lb.hub import Hub
from vllm_lb.proxy import Proxy

def box_a_lease():
    """The router serves ONLY from leased backends since 2026-09-12
    (worklog/vllm-lb-leased-only.md). These tests are about the proxy/search/e2e path, not
    about the lease gate, so the backend is held throughout."""
    from vllm_lb.state import LEASE_RUNNING, Lease
    return Lease(lease_id="r", job_id="j", created_at=0, status=LEASE_RUNNING, host_label="box-a-gpu", box="box-a")


SEARCH_TOOL = {"type": "web_search_20250305", "name": "web_search", "max_uses": 8}

#: Every error code Anthropic documents for `web_search_tool_result_error`
#: (platform.claude.com → web search tool → Errors). The LB may only ever emit one of these:
#: the client prints the code it is given, so an invented one reaches the operator as noise.
ANTHROPIC_ERROR_CODES = {"too_many_requests", "invalid_tool_input", "max_uses_exceeded",
                         "query_too_long", "request_too_large", "unavailable"}

#: What SearXNG can say about an engine that produced nothing — the closed set in the pinned
#: image's `searx/webutils.py` (`exception_classname_to_text`), which `get_translated_errors`
#: prefixes with `Suspended: ` while the engine is suspended.
SEARXNG_REFUSALS = ["too many requests", "access denied", "CAPTCHA"]
SEARXNG_FAILURES = ["timeout", "HTTP error", "HTTP connection error", "HTTP protocol error",
                    "SSL error: certificate validation has failed", "proxy error",
                    "network error", "server API error", "parsing error", "unexpected crash"]


def _req(text="Perform a web search for the query: capital of France", *, stream=False,
         tools=(SEARCH_TOOL,)):
    return {"model": "qwen3.8:latest", "max_tokens": 1024, "stream": stream,
            "system": [{"type": "text", "text": "You are an assistant for performing a web search tool use"}],
            "messages": [{"role": "user", "content": [{"type": "text", "text": text}]}],
            "tools": list(tools), "tool_choice": {"type": "auto"}}


def _found(*results):
    return websearch.SearchOutcome(results=tuple(results))


# ── recognising the request ──────────────────────────────────────────────────

def test_the_server_side_web_search_request_is_recognised():
    assert websearch.is_web_search_request("/v1/messages", _req()) is True
    assert websearch.is_web_search_request("/v1/messages?beta=true", _req()) is True


def test_an_ordinary_messages_request_is_not_a_web_search_request():
    body = _req(tools=[{"name": "Bash", "input_schema": {}}])
    assert websearch.is_web_search_request("/v1/messages", body) is False
    assert websearch.is_web_search_request("/v1/messages", {"model": "x"}) is False


def test_count_tokens_is_never_intercepted_even_carrying_the_tool():
    assert websearch.is_web_search_request("/v1/messages/count_tokens", _req()) is False
    assert websearch.is_web_search_request("/v1/chat/completions", _req()) is False


# ── the query ────────────────────────────────────────────────────────────────

def test_the_query_is_taken_from_the_clients_prompt_without_its_preamble():
    assert websearch.extract_query(_req()) == "capital of France"


def test_a_prompt_without_the_preamble_is_the_query_itself():
    assert websearch.extract_query(_req("who owns vLLM")) == "who owns vLLM"


def test_no_user_text_means_no_query():
    assert websearch.extract_query({"messages": []}) is None


# ── the outcome: what ONE search produced ────────────────────────────────────

def test_results_are_an_outcome_that_succeeded():
    out = websearch.outcome_from_searxng(
        {"results": [{"title": "Paris", "url": "https://en.wikipedia.org/wiki/Paris"}]},
        max_results=10, snippet_chars=0)
    assert out.failed is False and out.error_code is None
    assert [r["url"] for r in out.results] == ["https://en.wikipedia.org/wiki/Paris"]


def test_zero_results_while_engines_are_rate_limited_is_a_failure_not_an_empty_web():
    """THE DEFECT. brave and google cse suspended, duckduckgo timing out, `results: []` —
    handed to the model as a success it reads as "the web has nothing"."""
    out = websearch.outcome_from_searxng(
        {"results": [], "unresponsive_engines": [["brave", "Suspended: too many requests"],
                                                 ["duckduckgo", "timeout"],
                                                 ["google cse", "Suspended: too many requests"]]},
        max_results=10, snippet_chars=0)
    assert out.failed is True
    assert out.error_code == "too_many_requests"
    assert out.results == ()


def test_zero_results_with_only_engines_that_broke_is_unavailable():
    out = websearch.outcome_from_searxng(
        {"results": [], "unresponsive_engines": [["duckduckgo", "timeout"],
                                                 ["yahoo", "parsing error"]]},
        max_results=10, snippet_chars=0)
    assert out.failed is True and out.error_code == "unavailable"


def test_zero_results_with_every_engine_answering_is_an_honest_empty_result():
    """Anthropic's own rule: "A search that succeeds but matches no results returns an empty
    `content` list, not an error." Nothing is unresponsive, so nothing failed."""
    out = websearch.outcome_from_searxng({"results": [], "unresponsive_engines": []},
                                         max_results=10, snippet_chars=0)
    assert out.failed is False and out.error_code is None and out.results == ()


def test_results_despite_one_dead_engine_are_still_results():
    """Breadth is the whole point: one engine may be blocked while the others carry the query."""
    out = websearch.outcome_from_searxng(
        {"results": [{"title": "Paris", "url": "https://en.wikipedia.org/wiki/Paris"}],
         "unresponsive_engines": [["brave", "Suspended: too many requests"]]},
        max_results=10, snippet_chars=0)
    assert out.failed is False and len(out.results) == 1
    assert out.unresponsive == (("brave", "Suspended: too many requests"),)


@pytest.mark.parametrize("message", SEARXNG_REFUSALS)
def test_an_engine_that_refused_us_maps_to_too_many_requests(message):
    """Rate limiting, an access denial and a CAPTCHA are all "you are asking too much" — and
    SearXNG suspends the engine for `search.suspended_times` either way."""
    for said in (message, f"Suspended: {message}"):
        out = websearch.outcome_from_searxng(
            {"results": [], "unresponsive_engines": [["some-engine", said]]},
            max_results=10, snippet_chars=0)
        assert out.error_code == "too_many_requests", said


@pytest.mark.parametrize("message", SEARXNG_FAILURES)
def test_an_engine_that_broke_maps_to_unavailable(message):
    out = websearch.outcome_from_searxng(
        {"results": [], "unresponsive_engines": [["some-engine", message]]},
        max_results=10, snippet_chars=0)
    assert out.error_code == "unavailable", message


def test_one_rate_limited_engine_names_the_failure_for_the_whole_query():
    """Mixed causes: a client that is being throttled must hear that, not "unavailable"."""
    out = websearch.outcome_from_searxng(
        {"results": [], "unresponsive_engines": [["duckduckgo", "timeout"],
                                                 ["brave", "Suspended: too many requests"]]},
        max_results=10, snippet_chars=0)
    assert out.error_code == "too_many_requests"


def test_the_outcome_says_why_nothing_came_back_in_searxngs_own_words():
    """The LB's log line. `'<query>' → 0 results` told the operator nothing for four days."""
    out = websearch.outcome_from_searxng(
        {"results": [], "unresponsive_engines": [["brave", "Suspended: too many requests"],
                                                 ["duckduckgo", "timeout"]]},
        max_results=10, snippet_chars=0)
    assert "brave: Suspended: too many requests" in out.detail()
    assert "duckduckgo: timeout" in out.detail()
    assert out.detail().startswith("too_many_requests")


def test_an_honest_empty_result_says_so_too():
    out = websearch.outcome_from_searxng({"results": []}, max_results=10, snippet_chars=0)
    assert "every engine answered" in out.detail()


def test_a_search_that_found_things_just_counts_them():
    assert _found({"title": "T", "url": "https://e.example/"}).detail() == "1 results"


def test_every_error_code_the_lb_can_emit_is_one_anthropic_documents():
    """The client prints the code verbatim (`Web search error: <code>`), so an invented one
    reaches the operator as noise and matches nothing in Anthropic's own docs."""
    assert websearch.ERROR_CODES <= ANTHROPIC_ERROR_CODES


# ── the answer's shape ───────────────────────────────────────────────────────

def test_the_answer_carries_the_two_blocks_the_client_harvests():
    msg = websearch.build_message("capital of France", _found(
        {"title": "Paris", "url": "https://en.wikipedia.org/wiki/Paris"},
    ), model="qwen3.8:latest")
    kinds = [b["type"] for b in msg["content"]]
    assert kinds == ["server_tool_use", "web_search_tool_result"]
    use, res = msg["content"]
    assert use["name"] == "web_search" and use["input"] == {"query": "capital of France"}
    assert res["tool_use_id"] == use["id"]
    assert res["content"] == [{"type": "web_search_result", "title": "Paris",
                               "url": "https://en.wikipedia.org/wiki/Paris",
                               "encrypted_content": "", "page_age": None}]
    assert msg["stop_reason"] == "end_turn" and msg["model"] == "qwen3.8:latest"


def test_a_failed_search_becomes_the_error_block_the_client_renders():
    """The exact JSON Anthropic's API sends, and the exact block ollama used to send for this
    request (worklog/vllm-lb-websearch.md §1c): `content` is a single OBJECT, not a list."""
    msg = websearch.build_message("x", websearch.SearchOutcome.failure("unavailable"), model="m")
    res = msg["content"][1]
    assert res["content"] == {"type": "web_search_tool_result_error", "error_code": "unavailable"}


def test_a_rate_limited_search_carries_that_code_to_the_client():
    msg = websearch.build_message("x", websearch.SearchOutcome.failure("too_many_requests"),
                                  model="m")
    assert msg["content"][1]["content"] == {"type": "web_search_tool_result_error",
                                            "error_code": "too_many_requests"}


def test_an_empty_but_honest_search_is_an_empty_list_not_an_error_block():
    msg = websearch.build_message("x", websearch.SearchOutcome(), model="m")
    assert msg["content"][1]["content"] == []


def test_the_stream_form_says_the_same_thing_in_sse():
    events = websearch.sse_events(websearch.build_message("q", _found(
        {"title": "T", "url": "https://e.example/"}), model="m"))
    names = [n for n, _ in events]
    assert names[0] == "message_start" and names[-1] == "message_stop"
    starts = [d["content_block"]["type"] for n, d in events if n == "content_block_start"]
    assert starts == ["server_tool_use", "web_search_tool_result"]
    # every block that opens also closes, in order
    assert len([n for n in names if n == "content_block_stop"]) == 2
    assert any(n == "message_delta" and d["delta"]["stop_reason"] == "end_turn" for n, d in events)


def test_the_stream_form_carries_the_failure_too():
    """Claude Code's web_search request is `"stream": true`, so the error block has to survive
    the SSE form — as the object Anthropic sends, not as an empty result list."""
    msg = websearch.build_message("q", websearch.SearchOutcome.failure("too_many_requests"),
                                  model="m")
    blocks = [d["content_block"] for n, d in websearch.sse_events(msg)
              if n == "content_block_start" and d["content_block"]["type"] == "web_search_tool_result"]
    assert blocks[0]["content"] == {"type": "web_search_tool_result_error",
                                    "error_code": "too_many_requests"}
    assert b'"web_search_tool_result_error"' in websearch.sse_body(msg)


# ── the SearXNG call ─────────────────────────────────────────────────────────

def test_searxng_results_become_title_and_url_with_the_snippet_kept():
    payload = {"results": [
        {"title": "Paris - Wikipedia", "url": "https://en.wikipedia.org/wiki/Paris",
         "content": "Paris is the capital and largest city of France."},
        {"title": "No snippet", "url": "https://b.example/", "content": ""},
        {"title": "junk", "url": ""},
    ]}
    out = websearch.results_from_searxng(payload, max_results=10, snippet_chars=180)
    assert [r["url"] for r in out] == ["https://en.wikipedia.org/wiki/Paris", "https://b.example/"]
    assert out[0]["title"] == "Paris - Wikipedia — Paris is the capital and largest city of France."
    assert out[1]["title"] == "No snippet"


def test_the_result_parser_reports_only_what_is_in_results():
    """Layering: this parser answers "what results are there", never "did the search work".
    A captcha'd engine drops out of `results`; whether THAT is a failure is the outcome's
    call (see the outcome tests above), and it is."""
    assert websearch.results_from_searxng(
        {"results": [], "unresponsive_engines": [["duckduckgo", "CAPTCHA"]]},
        max_results=10, snippet_chars=180) == []


def test_a_searxng_answer_that_is_not_the_expected_shape_raises_rather_than_corrupts():
    """Malformed must not read to the model as "the web has nothing"."""
    with pytest.raises(ValueError):
        websearch.results_from_searxng({"results": "nonsense"}, max_results=3, snippet_chars=0)


def test_a_malformed_answer_is_not_an_outcome_at_all():
    with pytest.raises(ValueError):
        websearch.outcome_from_searxng({"results": "nonsense"}, max_results=3, snippet_chars=0)


def test_an_unreadable_unresponsive_engines_list_does_not_crash_the_search():
    """SearXNG's own shape, defended: whatever that field turns out to be, the results still rule."""
    out = websearch.outcome_from_searxng(
        {"results": [{"title": "T", "url": "https://e.example/"}], "unresponsive_engines": "nonsense"},
        max_results=10, snippet_chars=0)
    assert out.failed is False and len(out.results) == 1


def test_a_searxng_that_never_answers_gives_up_instead_of_hanging():
    """A stuck search must not hold the client's request open. Bounded by search_timeout_s."""
    import socket
    import time as _time

    lsock = socket.socket()
    lsock.bind(("127.0.0.1", 0))
    lsock.listen(1)
    cfg = Config(search_url=f"http://127.0.0.1:{lsock.getsockname()[1]}/search",
                 search_timeout_s=0.5)
    t0 = _time.monotonic()
    try:
        with pytest.raises(OSError):
            websearch.searxng_search("anything", cfg)
    finally:
        lsock.close()
    assert _time.monotonic() - t0 < 5.0


def test_only_max_results_are_returned():
    payload = {"results": [{"title": f"t{i}", "url": f"https://e{i}.example/"} for i in range(20)]}
    assert len(websearch.results_from_searxng(payload, max_results=5, snippet_chars=0)) == 5


# ── through the live proxy ───────────────────────────────────────────────────

@pytest.fixture
def stack():
    created = []

    def make(*, backend: FakeBackend, searcher=None, **over):
        opts = {"search_url": "http://searxng:8080/search", **over}
        cfg = Config(vllm_boxes=(VllmBox("box-a-gpu", "box-a", backend.url),), listen_host="127.0.0.1", listen_port=0,
                     connect_timeout_s=2.0, first_byte_timeout_s=10.0, **opts)
        hub = Hub(cfg)
        b = hub.backend_for(backend.url)
        b.live, b.served_models = True, tuple(backend.served)
        hub.set_leases([box_a_lease()])
        proxy = Proxy(cfg, hub, searcher=searcher)
        proxy.start()
        created.append(proxy)
        return proxy, hub

    yield make
    for p in created:
        p.stop()


def _post(proxy, path, body: dict, timeout=10):
    conn = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=timeout)
    conn.request("POST", path, body=json.dumps(body).encode(),
                 headers={"Content-Type": "application/json"})
    return conn.getresponse()


def test_the_web_search_request_is_answered_by_the_lb_and_never_forwarded(stack):
    o = FakeBackend()
    seen = []

    def searcher(q):
        seen.append(q)
        return _found({"title": "Paris", "url": "https://en.wikipedia.org/wiki/Paris"})

    proxy, _ = stack(backend=o, searcher=searcher)
    r = _post(proxy, "/v1/messages?beta=true", _req())
    body = json.loads(r.read())
    assert r.status == 200
    assert seen == ["capital of France"]
    assert o.requests == []                      # the backend never saw it
    assert body["content"][1]["content"][0]["url"] == "https://en.wikipedia.org/wiki/Paris"


def test_a_search_that_raises_comes_back_as_unavailable_not_as_a_500(stack):
    o = FakeBackend()

    def searcher(q):
        raise OSError("searxng is down")

    proxy, _ = stack(backend=o, searcher=searcher)
    r = _post(proxy, "/v1/messages", _req())
    body = json.loads(r.read())
    assert r.status == 200
    assert body["content"][1]["content"]["error_code"] == "unavailable"
    assert o.requests == []


def test_a_search_no_engine_answered_reaches_the_client_as_a_failure_not_as_no_results(stack):
    """End to end over the wire: the shape the live LB has been sending as an empty success."""
    o = FakeBackend()
    outcome = websearch.SearchOutcome.failure(
        "too_many_requests", unresponsive=(("brave", "Suspended: too many requests"),))
    proxy, _ = stack(backend=o, searcher=lambda q: outcome)
    r = _post(proxy, "/v1/messages", _req())
    body = json.loads(r.read())
    assert r.status == 200
    assert body["content"][1]["content"] == {"type": "web_search_tool_result_error",
                                             "error_code": "too_many_requests"}
    assert o.requests == []


def test_the_streamed_client_is_told_about_the_failure_too(stack):
    o = FakeBackend()
    proxy, _ = stack(backend=o,
                     searcher=lambda q: websearch.SearchOutcome.failure("too_many_requests"))
    r = _post(proxy, "/v1/messages?beta=true", _req(stream=True))
    raw = r.read().decode().replace(" ", "")
    assert r.status == 200 and "text/event-stream" in r.getheader("Content-Type")
    assert '"error_code":"too_many_requests"' in raw
    assert o.requests == []


def test_the_log_line_names_the_engines_that_did_not_answer(stack, caplog):
    """W4's operator handle: while results are empty the LB's own line has to say WHY."""
    o = FakeBackend()
    outcome = websearch.SearchOutcome.failure(
        "too_many_requests", unresponsive=(("brave", "Suspended: too many requests"),
                                           ("duckduckgo", "timeout")))
    proxy, _ = stack(backend=o, searcher=lambda q: outcome)
    with caplog.at_level(logging.INFO, logger="vllm_lb.proxy"):
        _post(proxy, "/v1/messages", _req()).read()
    line = "\n".join(r.getMessage() for r in caplog.records if "[websearch]" in r.getMessage())
    assert "too_many_requests" in line
    assert "brave: Suspended: too many requests" in line and "duckduckgo: timeout" in line


def test_the_streaming_client_gets_sse_with_both_blocks(stack):
    o = FakeBackend()
    proxy, _ = stack(backend=o, searcher=lambda q: _found({"title": "T", "url": "https://e.example/"}))
    r = _post(proxy, "/v1/messages?beta=true", _req(stream=True))
    raw = r.read().decode()
    assert r.status == 200 and "text/event-stream" in r.getheader("Content-Type")
    assert "event: message_start" in raw and "event: message_stop" in raw
    assert '"type":"server_tool_use"' in raw.replace(" ", "")
    assert '"url":"https://e.example/"' in raw.replace(" ", "")
    assert o.requests == []


def test_the_lbs_own_search_answer_cannot_be_framed_or_sniffed_streamed_or_not(stack):
    """This answer is WRITTEN by the LB, so it carries the LB's own headers. Claude Code's WebSearch
    streams (the captured request has `"stream": true`), and that hand-built SSE reply is the one a
    helper-by-helper fix missed (worklog/vllm-gateway-deep-audit.md, G5)."""
    from tests.test_static_ui import OWN_HEADERS

    o = FakeBackend()
    proxy, _ = stack(backend=o, searcher=lambda q: _found({"title": "T", "url": "https://e.example/"}))
    for stream in (False, True):
        r = _post(proxy, "/v1/messages?beta=true", _req(stream=stream))
        r.read()
        assert r.status == 200
        assert {name: r.getheader(name) for name in OWN_HEADERS} == OWN_HEADERS, f"stream={stream}"


def test_the_search_answer_keeps_its_status_line_and_headers_over_an_http_0_9_request_line(stack):
    """The third audit's other named writer: `http.client` cannot send a request line ending in `HTTP/0.9`,
    and until the version was normalised where it is STORED, the hand-built SSE reply — the form Claude
    Code's WebSearch actually uses — went out with no status line at all."""
    from tests.test_raw_requests import head_of, raw
    from tests.test_static_ui import OWN_HEADERS

    o = FakeBackend()
    proxy, _ = stack(backend=o, searcher=lambda q: _found({"title": "T", "url": "https://e.example/"}))
    for stream in (False, True):
        payload = json.dumps(_req(stream=stream)).encode()
        reply = raw(proxy, f"POST /v1/messages?beta=true HTTP/0.9\r\nHost: 127.0.0.1:{proxy.port}\r\n".encode()
                           + b"Content-Type: application/json\r\n"
                           b"Content-Length: " + str(len(payload)).encode() + b"\r\n\r\n" + payload)
        status, headers = head_of(reply)
        assert status.startswith("HTTP/1.1 200"), f"stream={stream} → {status[:60]!r}"
        assert {name: headers.get(name) for name in OWN_HEADERS} == OWN_HEADERS, f"stream={stream}"
        assert b"https://e.example/" in reply, f"stream={stream}: the search answer itself is missing"
    assert o.requests == []


def test_with_no_search_url_configured_the_request_is_forwarded_as_before(stack):
    o = FakeBackend()
    o.reply_json = {"id": "msg_1", "content": [{"type": "text", "text": "from the backend"}]}
    proxy, _ = stack(backend=o, search_url="")
    r = _post(proxy, "/v1/messages", _req())
    assert json.loads(r.read())["content"][0]["text"] == "from the backend"
    assert len(o.requests) == 1


def test_an_ordinary_request_still_reaches_the_backend(stack):
    o = FakeBackend()
    o.reply_json = {"id": "msg_1", "content": [{"type": "text", "text": "hi"}]}
    proxy, _ = stack(backend=o, searcher=lambda q: websearch.SearchOutcome())
    r = _post(proxy, "/v1/messages", {"model": "qwen3.8:latest", "max_tokens": 4,
                                      "messages": [{"role": "user", "content": "hi"}]})
    assert json.loads(r.read())["content"][0]["text"] == "hi"
    assert len(o.requests) == 1
