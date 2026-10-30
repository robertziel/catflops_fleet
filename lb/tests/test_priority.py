"""PRIORITY BY HEADER (operator 2026-10-28: "Implement priority header then" / "enable priority each time").

A caller sends `X-Priority: <integer>`; the LB hands it to the engine as the request's own `priority` field — the
field both engines read on their OpenAI endpoints (vLLM `priority`, lower = sooner, ties by arrival; SGLang the same
with --schedule-low-priority-values-first). The engine does the ordering: its waiting queue, and which running
request is preempted first when the KV cache runs short. Every launch turns priority scheduling on (tp-rank, the
node's compose); with no header a request is the engine's default 0, so unlabelled traffic is FCFS as before.

Both engines IGNORE `priority` silently where it is not read — under vLLM's fcfs policy, SGLang without
--enable-priority-scheduling, and on either engine's Anthropic /v1/messages adapter, whose request carries no such
field. So the header is REFUSED (400) on any path that would drop it, never forwarded to be ignored.
"""
from __future__ import annotations

import json

import pytest

from tests.fakes import FakeBackend
from tests.test_proxy import _post, stack  # noqa: F401 — the live-proxy fixture
from vllm_lb.router import PRIORITY_HEADER, priority_of, rewrite_priority

CHAT = "/v1/chat/completions"


@pytest.mark.parametrize("raw, want", [(None, None), ("5", 5), (" -3 ", -3), ("0", 0), ("999999999", 999999999)])
def test_the_header_is_an_integer_or_absent(raw, want):
    assert priority_of(raw, CHAT, {"model": "m"}) == want


@pytest.mark.parametrize("raw", ["high", "1.5", "", "+", "1e3", "9999999999", "--1"])
def test_anything_else_is_refused_naming_the_header(raw):
    with pytest.raises(ValueError, match=PRIORITY_HEADER):
        priority_of(raw, CHAT, {})


@pytest.mark.parametrize("path", ["/v1/audio/transcriptions", "/v1/embeddings", "/v1/messages/count_tokens"])
def test_a_path_whose_engine_would_drop_it_is_refused(path):
    with pytest.raises(ValueError, match=path):
        priority_of("1", path, {})


@pytest.mark.parametrize("path", ["/v1/completions", "/v1/messages"])
def test_the_completions_and_messages_paths_carry_it_too(path):
    """/v1/messages: the fleet's patched images carry it into the chat request (deploy/images/vllm/priority/priority.patch)."""
    assert priority_of("2", path, {}) == 2


def test_a_body_naming_another_priority_is_refused_one_that_agrees_passes():
    with pytest.raises(ValueError, match="priority"):
        priority_of("1", CHAT, {"priority": 4})
    assert priority_of("4", CHAT, {"priority": 4}) == 4


def test_the_rewrite_sets_the_field_and_never_mutates_the_callers_body():
    body = {"model": "m"}
    assert rewrite_priority(body, None) is body
    out = rewrite_priority(body, -5)
    assert out == {"model": "m", "priority": -5} and body == {"model": "m"}


def test_the_engine_receives_the_headers_priority(stack):  # noqa: F811
    v = FakeBackend()
    proxy, _ = stack(vllm=v)
    _post(proxy, CHAT, {"model": "m", "messages": []}, headers={PRIORITY_HEADER: "-5"}).read()
    assert json.loads(v.requests[-1]["body"])["priority"] == -5
    _post(proxy, CHAT, {"model": "m", "messages": []}).read()
    assert "priority" not in json.loads(v.requests[-1]["body"]), "no header: the engine's own default"


def test_a_header_the_engine_would_ignore_is_answered_400_and_never_sent(stack):  # noqa: F811
    v = FakeBackend()
    proxy, _ = stack(vllm=v)
    r = _post(proxy, "/v1/embeddings", {"model": "m", "input": "x"}, headers={PRIORITY_HEADER: "1"})
    assert r.status == 400
    assert json.loads(r.read())["error"]["type"] == "invalid_priority"
    assert v.requests == []


# ── THE PAGE'S TOGGLE (operator 2026-10-28: "small square priority column, similar to kill task, green when active,
# user in GUI can activate too or deactivate"): POST /lb/requests/<id>/priority {"active": bool}. Neither engine can
# change a request's priority once it holds it, so a turn the client has had NO byte of — queued or prefilling at the
# engine — is cut from its engine (which drops a request whose client goes, request_kill.py) and sent again at the new
# priority; one already streaming, or on a path whose engine drops the field, is locked, and the row says why.

from tests.test_request_kill import (OPENAI_DELTA, OPENAI_HEAD, Engine, anthropic_body, ask,  # noqa: E402
                                     lb, live_row, openai_body, wait)  # noqa: F401 — the stack fixture
from vllm_lb.router import PRIORITY_GUI  # noqa: E402


def chats(engine) -> list[dict]:
    """The turns the engine was sent (its /tokenize — the LB's prompt count — is not one)."""
    return [p for p in engine.posts() if p["path"] == CHAT]


def toggle(port: int, rid: str, body=b'{"active": true}', ctype="application/json"):
    import http.client
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    conn.request("POST", f"/lb/requests/{rid}/priority", body=body, headers={"Content-Type": ctype} if ctype else {})
    r = conn.getresponse()
    return r.status, json.loads(r.read() or b"{}")


def test_the_toggles_value_is_one_step_sooner_than_none():
    assert PRIORITY_GUI == -1


def test_a_queued_turn_toggled_on_is_resent_at_its_new_priority_the_client_seeing_nothing_of_the_first(lb):  # noqa: F811
    engine = Engine("hold")
    proxy, hub = lb(engine)
    out: dict = {}
    ask(proxy.port, CHAT, openai_body(), out)
    r = live_row(proxy)
    assert r["priority"] is None and r["priority_locked"] is None, r
    status, got = toggle(proxy.port, r["id"])
    assert status == 200 and got["priority"] == {"id": r["id"], "active": True, "value": PRIORITY_GUI,
                                                 "applied": "requeued"}, got
    assert engine.gone.wait(3), "the first copy's connection was closed: the engine drops it"
    wait(lambda: len(chats(engine)) == 2)
    assert "priority" not in json.loads(chats(engine)[0]["body"])
    assert json.loads(chats(engine)[1]["body"])["priority"] == PRIORITY_GUI
    assert out == {}, "nothing reached the client"
    assert wait(lambda: live_row(proxy)["priority"] == PRIORITY_GUI)
    b = hub.backend_for(engine.url)
    assert b.penalty_until == 0.0 and b.last_error == "", "the operator's re-send is no backend failure"
    status, got = toggle(proxy.port, r["id"], b'{"active": false}')
    assert status == 200 and got["priority"]["value"] is None and got["priority"]["applied"] == "requeued"
    wait(lambda: len(chats(engine)) == 3)
    assert "priority" not in json.loads(chats(engine)[2]["body"]), "off: the engine's own default again"


def test_a_streaming_turn_is_locked_and_the_row_says_why(lb):  # noqa: F811
    engine = Engine("stream", head=OPENAI_HEAD, delta=OPENAI_DELTA)
    proxy, _ = lb(engine)
    ask(proxy.port, CHAT, openai_body(), {})
    r = live_row(proxy, "streaming")
    assert "streaming" in r["priority_locked"], r["priority_locked"]
    status, got = toggle(proxy.port, r["id"])
    assert status == 409 and got["error"]["type"] == "priority_locked", got
    assert len(chats(engine)) == 1, "never re-sent"


def test_a_queued_messages_turn_is_resent_with_its_priority_in_the_anthropic_body(lb):  # noqa: F811
    engine = Engine("hold")
    proxy, _ = lb(engine)
    ask(proxy.port, "/v1/messages", anthropic_body(), {})
    r = live_row(proxy)
    assert r["priority_locked"] is None, r["priority_locked"]
    assert toggle(proxy.port, r["id"])[0] == 200
    posts = wait(lambda: [p for p in engine.posts() if p["path"] == "/v1/messages"][1:])
    assert json.loads(posts[0]["body"])["priority"] == PRIORITY_GUI


def test_the_header_shows_on_the_row(lb):  # noqa: F811
    engine = Engine("hold")
    proxy, _ = lb(engine)
    ask(proxy.port, CHAT, openai_body(), {}, headers={PRIORITY_HEADER: "-5"})
    assert live_row(proxy)["priority"] == -5


@pytest.mark.parametrize("body, ctype, want", [(b'{"active": true}', "text/plain", 415),
                                               (b'{"active": true}', "text/plain; x=json", 415),
                                               (b'{"active": 1}', None, 400),
                                               (b"[]", None, 400), (b"{}", None, 400)])
def test_a_toggle_that_is_not_a_json_active_flag_is_refused(lb, body, ctype, want):  # noqa: F811
    engine = Engine("hold")
    proxy, _ = lb(engine)
    ask(proxy.port, CHAT, openai_body(), {})
    r = live_row(proxy)
    assert toggle(proxy.port, r["id"], body, ctype or "application/json")[0] == want
    assert len(chats(engine)) == 1


def test_an_unknown_id_is_404(lb):  # noqa: F811
    proxy, _ = lb(Engine("hold"))
    status, got = toggle(proxy.port, "ffff")
    assert status == 404 and got["error"]["type"] == "unknown_request"
