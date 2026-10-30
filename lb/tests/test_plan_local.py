"""THE PLAN FOR A NODE THE LB REQUESTS (fleet control `local`; cluster.plan_local) — a pure function of what the operator
wants (`desired`, the node's switch), what was released to the host (`intent`), what the host's applier reports, and
what the node's own HTTP API says. One action exists: `release` — write the intent the applier applies.

  * A RELAUNCH (the node serves and the wish changes it) is released only once the node has nothing in flight, as the
    pair finishes its turns before a restart (`finishing`: no new turn is routed to it, so it drains); `force` releases
    at once. A node that does not answer has nothing in flight.
  * Released, the node is not routed to until it answers again with what was applied (`starting`), or it is down
    (`unloading` → `off`).
  * What the applier did is said in its own words: a refusal (nothing ran: the node still serves what it served), a
    failed command (`failed`, its exit code), a container restarting.
  * Nobody listening never strands a serving node: an intent the applier has not picked up leaves a live node routed to,
    and says why.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from vllm_lb.cluster import plan_local

AT = 1_791_742_000.0
BOX = "node-a"


def wish(state="up", launch=None, force=False, id=1) -> dict:
    return {"id": id, "at": AT, "state": state, "force": force, "launch": launch or {}}


def report(applied=None, applying=None, listening=True, container=None, error="") -> dict:
    return {"listening": listening, "heartbeat_age_s": 5.0, "host": "h", "applied": applied, "applying": applying,
            "container": container if container is not None else {"status": "running", "exit_code": 0,
                                                                   "restarts": 0},
            "log": None, "error": error}


def done(intent: dict, **over) -> dict:
    return {**{k: intent[k] for k in ("id", "at", "state", "force", "launch")}, "done_at": AT + 3, "exit": 0,
            "stderr": "", "refused": "", **over}


def plan(desired=None, intent=None, rep=None, live=True, busy=0, error=""):
    return plan_local(desired=desired, intent=intent, report=rep or report(), live=live, error=error, busy=busy,
                      box=BOX)


# ── nothing asked: the fleet's boot default, observed ───────────────────────────────────────────


def test_with_nothing_ever_asked_the_node_is_observed_as_its_boot_default():
    p = plan()
    assert (p.actions, p.state, p.held) == ([], "running", True)
    p = plan(live=False, error="ConnectionRefusedError")
    assert (p.actions, p.state, p.held) == ([], "unknown", False)
    assert "ConnectionRefusedError" in p.detail


# ── releasing a wish ─────────────────────────────────────────────────────────────────────────────


def test_a_wish_for_an_idle_node_is_released_at_once():
    p = plan(desired=wish(launch={"max_num_seqs": 2}))
    assert p.actions == ["release"] and p.held is False


def test_a_relaunch_waits_for_the_node_to_finish_what_it_serves_unless_forced():
    p = plan(desired=wish(launch={"max_num_seqs": 2}), busy=3)
    assert (p.actions, p.state, p.held) == ([], "finishing", False), "no new turn: it drains"
    assert "3 request(s)" in p.detail and "force" in p.detail
    assert plan(desired=wish(launch={"max_num_seqs": 2}, force=True), busy=3).actions == ["release"]


def test_a_node_that_does_not_answer_has_nothing_to_finish():
    assert plan(desired=wish(), live=False, busy=0).actions == ["release"]


def test_a_wish_already_released_is_not_released_again():
    w = wish(launch={"max_num_seqs": 2})
    released = {**w, "id": 7}
    assert plan(desired=w, intent=released, rep=report(applying={**released, "since": AT})).actions == []


# ── what the applier did ─────────────────────────────────────────────────────────────────────────


def test_released_and_not_yet_picked_up_is_starting_and_not_routed_to():
    w = wish(launch={"max_num_seqs": 2})
    p = plan(desired=w, intent={**w, "id": 7}, live=True)
    assert (p.state, p.held) == ("starting", False)
    assert "#7" in p.detail


def test_applying_says_so():
    w = wish(launch={"max_num_seqs": 2})
    i = {**w, "id": 7}
    p = plan(desired=w, intent=i, rep=report(applying={**{k: i[k] for k in ("id", "at", "state", "force")},
                                                       "since": AT}), live=False)
    assert (p.state, p.held) == ("starting", False) and "applying" in p.detail


def test_applied_and_answering_is_running_and_routed_to():
    w = wish(launch={"max_num_seqs": 2})
    i = {**w, "id": 7}
    p = plan(desired=w, intent=i, rep=report(applied=done(i)), live=True)
    assert (p.actions, p.state, p.held) == ([], "running", True)


def test_applied_and_loading_is_starting():
    w = wish()
    i = {**w, "id": 7}
    p = plan(desired=w, intent=i, rep=report(applied=done(i, restarts_at_apply=0)), live=False,
             error="Connection refused")
    assert (p.state, p.held) == ("starting", False)


def test_a_container_that_keeps_restarting_is_failed_in_its_own_numbers():
    w = wish()
    i = {**w, "id": 7}
    rep = report(applied=done(i), container={"status": "restarting", "exit_code": 1, "restarts": 6})
    p = plan(desired=w, intent=i, rep=rep, live=False)
    assert p.state == "failed" and "restarting" in p.detail and "exit 1" in p.detail and "6" in p.detail


def test_a_failed_command_is_failed_with_its_exit_code_and_last_words():
    w = wish()
    i = {**w, "id": 7}
    rep = report(applied=done(i, exit=1, stderr="pulling…\nError response from daemon: no such image"))
    p = plan(desired=w, intent=i, rep=rep, live=False)
    assert p.state == "failed" and "exit 1" in p.detail and "no such image" in p.detail


def test_a_refusal_is_said_and_a_node_that_still_serves_stays_routed_to():
    w = wish(launch={"max_num_seqs": 999})
    i = {**w, "id": 7}
    rep = report(applied=done(i, exit=None, refused="max_num_seqs must be a whole number in 1..48, got 999"))
    p = plan(desired=w, intent=i, rep=rep, live=True)
    assert (p.state, p.held) == ("running", True)
    assert "refused" in p.detail and "1..48" in p.detail
    assert plan(desired=w, intent=i, rep=rep, live=False).state == "failed"


#: vLLM's own refusal, as site-b's node printed it on 2026-10-11 (intent #3, 262,144 at 0.90) before its container
#: restart-looped — `status: running` between attempts, so its state alone never said so.
REFUSAL = ("ValueError: To serve at least one request with the model's max seq len (262144), (8.33 GiB KV cache is "
           "needed, which is larger than the available KV cache memory (5.55 GiB). Based on the available memory, the "
           "estimated maximum model length is 170912.")
LOOP_LOG = ("(EngineCore pid=295) INFO 10-11 20:56:04 [gpu_worker.py:745] Available KV cache memory: 5.6 GiB\n"
            f"(EngineCore pid=295) ERROR 10-11 20:56:04 [core.py:1505] {REFUSAL}\n"
            "Traceback (most recent call last):\n"
            f"{REFUSAL}\n"
            "(APIServer pid=1) INFO 10-11 20:56:40 [api_utils.py:286] non-default args: {'max_model_len': 262144}\n")


def looping(restarts: int, at_apply: int = 0, log: str = LOOP_LOG) -> tuple[dict, dict]:
    w = wish(launch={"max_model_len": 262144})
    i = {**w, "id": 3}
    rep = report(applied=done(i, restarts_at_apply=at_apply),
                 container={"status": "running", "exit_code": 0, "restarts": restarts})
    rep["log"] = log
    return w, i, rep


def test_a_container_restarted_since_its_intent_is_failed_in_the_nodes_own_last_words():
    w, i, rep = looping(restarts=2)
    p = plan(desired=w, intent=i, rep=rep, live=False, error="ConnectionRefusedError: [Errno 111] Connection refused")
    assert (p.state, p.held) == ("failed", False)
    assert REFUSAL in p.detail, p.detail
    assert "restarted 2" in p.detail and "#3" in p.detail
    assert "ConnectionRefusedError" not in p.detail, "the probe's socket error is no reason"


def test_with_no_error_line_in_the_log_it_is_failed_in_its_numbers():
    w, i, rep = looping(restarts=1, log="INFO loading\n")
    p = plan(desired=w, intent=i, rep=rep, live=False, error="ConnectionRefusedError")
    assert p.state == "failed" and "restarted 1" in p.detail and "ConnectionRefusedError" not in p.detail


def test_it_stays_failed_until_a_new_intent_and_a_node_that_answers_after_all_runs():
    w, i, rep = looping(restarts=5)
    assert plan(desired=w, intent=i, rep=rep, live=False).state == "failed"
    assert plan(desired=w, intent=i, rep=rep, live=True).state == "running", "it answers: it serves"
    fresh = {**w, "launch": {"max_model_len": 131072}}
    assert plan(desired=fresh, intent=i, rep=rep, live=False).actions == ["release"], "a new wish is released"
    i4 = {**fresh, "id": 4}
    assert plan(desired=fresh, intent=i4, rep=rep, live=False).state == "starting", "…and starts afresh"


def test_a_start_still_loading_says_so_without_the_probes_socket_error():
    w, i, rep = looping(restarts=0)
    p = plan(desired=w, intent=i, rep=rep, live=False, error="ConnectionRefusedError: [Errno 111] Connection refused")
    assert p.state == "starting" and "ConnectionRefusedError" not in p.detail
    assert "loading" in p.detail


# ── which line of the node's log it failed on: vLLM's ROOT CAUSE, never its wrapper ─────────────────────────────────
#: The 5090 node, 2026-10-11 23:21Z, started at GPU memory 0.99: the panel's Failed card said "RuntimeError: Engine core
#: initialization failed. See root cause above. Failed core proc(s): {'EngineCore': 1}" — the log's LAST exception line,
#: the API server's wrapper — while vLLM's own cause sat above it. The container was relaunched before its log could be
#: read again, so this fixture is RECONSTRUCTED around the two lines read then (the ValueError and the wrapper, verbatim)
#: in vLLM's own layout at 2eb491e: whatever the engine core process logs carries "(EngineCore pid=N) LEVEL MM-DD
#: HH:MM:SS [file:line] " (vllm/logger.py:28-32) on every line of a traceback too (logging_utils/formatter.py
#: NewLineFormatter), logged at core.py:1505 ("EngineCore failed to start."); multiprocessing's own traceback and the
#: API server's uncaught one (v1/engine/utils.py:1431) are bare stderr; the restarted container's next attempt follows.
#: Frames are vLLM's call path at 2eb491e (gpu_worker.py:521 → utils.py:762), some trimmed.
BOOT = (Path(__file__).resolve().parent / "fixtures" / "vllm-2eb491e-start-refused-free-memory.log").read_text()
ROOT = ("ValueError: Free memory on device cuda:0 (30.85/31.36 GiB) on startup is less than desired GPU memory "
        "utilization (0.99, 31.04 GiB). Decrease GPU memory utilization or reduce GPU memory used by other processes.")
WRAPPER = ("RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): "
           "{'EngineCore': 1}")
OOM = "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB. GPU 0 has a total capacity of 31.36 GiB"


def _failed_with(log: str, container: dict | None = None) -> str:
    w, i, rep = looping(restarts=1, log=log)
    if container is not None:
        rep["container"] = container
    p = plan(desired=w, intent=i, rep=rep, live=False, error="ConnectionRefusedError: [Errno 111] Connection refused")
    assert p.state == "failed", (p.state, p.detail)
    return p.detail


@pytest.mark.parametrize("container", [None, {"status": "exited", "exit_code": 1, "restarts": 0}],
                         ids=["restart-looping", "exited"])
def test_a_start_vllm_refused_is_failed_in_its_root_cause_not_its_wrapper(container):
    assert WRAPPER in BOOT and BOOT.count(ROOT) == 2, "the fixture: the cause logged and re-raised, then the wrapper"
    detail = _failed_with(BOOT, container)
    assert ROOT in detail, detail
    assert "Engine core initialization failed" not in detail and "See root cause above" not in detail, detail


def test_the_root_cause_is_the_latest_failed_attempts():
    """A crash loop leaves every attempt in the log tail: the cause said is the one of the attempt that ended last."""
    oom = BOOT.replace(ROOT, OOM)
    latest_free = _failed_with(oom + BOOT)
    assert ROOT in latest_free and OOM not in latest_free, latest_free
    latest_oom = _failed_with(BOOT + oom)
    assert OOM in latest_oom and ROOT not in latest_oom, ("with its module: torch.", latest_oom)


def test_a_coloured_or_crlf_log_reads_the_same():
    """vLLM colours its prefix on a terminal (logger.py `_use_color`, ColoredFormatter) and joins a traceback's lines
    with "\\r\\n" (NewLineFormatter): neither moves the cause."""
    crlf = BOOT.replace("\n(EngineCore", "\r\n(EngineCore")
    coloured = re.sub(r"^\((EngineCore pid=\d+)\) ERROR (\S+ \S+) (\[[^\]]+\])",
                      "\x1b[0;36m(\\1)\x1b[0;0m \x1b[31mERROR\x1b[0m \x1b[90m\\2\x1b[0m \x1b[90m\\3\x1b[0m", BOOT,
                      flags=re.M)
    assert coloured != BOOT and crlf != BOOT
    for log in (crlf, coloured):
        detail = _failed_with(log)
        assert ROOT in detail and "\x1b" not in detail and "\r" not in detail, repr(detail)


def test_a_wrapper_whose_cause_left_the_tail_is_said_as_it_stands():
    """The cause is above the wrapper or nowhere in what was read: then the wrapper is the node's last words, verbatim —
    never another attempt's cause, never a guess."""
    tail = BOOT[BOOT.index('Traceback (most recent call last):\n  File "/usr/local/bin/vllm"'):]
    assert WRAPPER in _failed_with(tail)
    earlier = BOOT.replace(ROOT, OOM)                     # an earlier attempt's cause, before ITS wrapper…
    assert OOM not in _failed_with(earlier + tail), "…is not this attempt's"


def test_an_error_another_process_logged_is_no_engine_cores_root_cause():
    other = BOOT.replace("(EngineCore pid=295) ERROR 10-11 23:21:02 [core.py:1505] EngineCore failed to start.\n",
                         "(APIServer pid=1) ERROR 10-11 23:21:01 [api_utils.py:12] KeyError: 'nothing'\n"
                         "(EngineCore pid=295) ERROR 10-11 23:21:02 [core.py:1505] EngineCore failed to start.\n")
    assert other != BOOT
    detail = _failed_with(other)
    assert ROOT in detail and "KeyError" not in detail, detail


def test_stopping_and_stopped():
    w = wish(state="down")
    i = {**w, "id": 8}
    assert (plan(desired=w, intent=i, rep=report(applied=done(i)), live=True).state) == "unloading"
    p = plan(desired=w, intent=i, rep=report(applied=done(i), container={"status": "exited", "exit_code": 0,
                                                                          "restarts": 0}), live=False)
    assert (p.state, p.held) == ("off", False)


def test_nobody_listening_never_strands_a_serving_node():
    w = wish(launch={"max_num_seqs": 2})
    i = {**w, "id": 7}
    rep = report(listening=False, error="no applier listening: its last heartbeat was 400 s ago")
    p = plan(desired=w, intent=i, rep=rep, live=True)
    assert (p.state, p.held) == ("running", True)
    assert "not applied" in p.detail and "400 s" in p.detail
