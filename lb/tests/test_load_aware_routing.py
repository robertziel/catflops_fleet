"""Waiting must never become failing: send the request where it will be ANSWERED.

2026-09-12 — a turn waited the whole 60-minute first-byte budget and 504'd, which Claude Code
shows as "Waiting for API response · will retry in 2m 27s". At that moment:

    vllm   box-a    waiting=10 running=2    <- two fleet llm_chat jobs held both seq slots
    vllm   box-b   waiting=2  running=2
    ollama           waiting=0  running=0   <- idle, and never tried

`choose` ranked the vLLM tier above ollama unconditionally, so it queued the turn behind ten
others while an idle backend sat there. The prober already collects waiting/running from
/metrics; the router just never looked at them.

Sharing the vLLM with the fleet is intended. Failing because of it is not. (The idle backend then was the ollama
fallback, retired 2026-10-17; the rule — a deep vLLM is demoted below one with room — is pinned here with a second,
idle vLLM: `o()`.)
"""
from __future__ import annotations

from vllm_lb.config import Config
from vllm_lb.router import choose
from vllm_lb.state import LEASE_RUNNING, VLLM, Backend, Lease, Snapshot

NOW = 1000.0
CFG = Config(vllm_max_depth=4)


def v(url, *, waiting=0, running=0, label="box-a-gpu", box="box-a"):
    b = Backend(kind=VLLM, url=url, host_label=label, box=box)
    b.live, b.served_models, b.waiting, b.running = True, ("qwen3.8:latest",), waiting, running
    return b


IDLE = "http://box-b:8000"


def o(url=IDLE, *, inflight=0):
    """The OTHER server, idle (box-b)."""
    b = Backend(kind=VLLM, url=url, host_label="box-b-gpu", box="box-b")
    b.live, b.served_models, b.inflight = True, ("qwen3.8:latest",), inflight
    return b


def snap(backends, leased=("box-a-gpu", "box-b-gpu")):
    return Snapshot(now=NOW, backends=list(backends),
                    leases=[Lease(lease_id=f"r{i}", job_id=f"j{i}", created_at=NOW,
                                  status=LEASE_RUNNING, host_label=l, box=l.split("-")[0])
                            for i, l in enumerate(leased)],
                    # Both leased: since 2026-09-12 the router serves ONLY from backends the LB holds
                    # (worklog/vllm-lb-leased-only.md), and these tests are about TIERING, not the lease gate.
                    last_request_at=NOW, last_pressure_at=None, db_ok=True)


def test_a_deeply_queued_vllm_loses_to_an_idle_one():
    """THE regression: 10 waiting on box-a while another backend sat idle."""
    s = snap([v("http://box-a:8000", waiting=10, running=2), o()])
    assert choose(s, CFG).url == IDLE, "an hour in a queue is worse than a slow answer"


def test_the_shallower_of_two_vllms_wins():
    s = snap([v("http://box-a:8000", waiting=9, running=2),
              v("http://box-b:8000", waiting=1, running=2, label="box-b-gpu", box="box-b")])
    assert choose(s, CFG).url == "http://box-b:8000"


def test_a_deep_vllm_is_still_used_when_it_is_all_there_is():
    """Demoted, never banned — the rule passes 10 and 11 had to learn."""
    s = snap([v("http://box-a:8000", waiting=99, running=2)])
    assert choose(s, CFG).url == "http://box-a:8000"


def test_a_dead_other_server_does_not_strand_the_request():
    dead = o()
    dead.live = False
    s = snap([v("http://box-a:8000", waiting=50, running=2), dead])
    assert choose(s, CFG).url == "http://box-a:8000"


def test_the_threshold_is_configurable():
    # box-a leased and 5 deep; box-b idle but NOT leased (allowed by use_unleased_vllm: a lower tier). Only the
    # threshold decides whether box-a keeps its tier or drops below box-b.
    s = snap([v("http://box-a:8000", waiting=3, running=2), o()], leased=("box-a-gpu",))
    assert choose(s, Config(broker_enabled=True, vllm_max_depth=2, use_unleased_vllm=True)).url == IDLE
    assert choose(s, Config(broker_enabled=True, vllm_max_depth=10, use_unleased_vllm=True)).url == "http://box-a:8000"


def test_the_shipped_default_matches_the_benchmarked_knee():
    """RE-MEASURED 2026-09-12 against the deployed server (--max-num-seqs 3, effort=low):
    throughput scaled linearly to concurrency 3 (13.0 / 27.0 / 38.9 tok/s) and collapsed past
    it. So every slot is worth filling and the threshold is `max-num-seqs - 1`. The earlier
    value of 1 abandoned a free slot for a fallback measured at 258116 ms vs vLLM's 476 ms."""
    assert Config().vllm_max_depth == 2


def test_env_tunes_it_for_a_bigger_max_num_seqs():
    assert Config.from_env({"LB_ALLOWED_HOSTS": "loopback", "LB_VLLM_MAX_DEPTH": "16"}).vllm_max_depth == 16


def test_a_FULL_vllm_is_demoted_even_though_it_reports_no_waiting():
    """The flaw a live probe exposed: box-b showed `wait=0 run=2` with --max-num-seqs 2 —
    every slot taken — and a waiting-only test called it idle. The next request queued for
    over five minutes."""
    s = snap([v("http://box-a:8000", waiting=0, running=2), o()])
    assert choose(s, Config(vllm_max_depth=1)).url == IDLE


def test_a_vllm_with_a_free_slot_is_still_preferred():
    s = snap([v("http://box-a:8000", waiting=0, running=1), o(inflight=1)])
    assert choose(s, Config(vllm_max_depth=1)).url in ("http://box-a:8000", IDLE), "both have a free slot"
    full = snap([v("http://box-a:8000", waiting=0, running=1), o(inflight=2)])
    assert choose(full, Config(vllm_max_depth=1)).url == "http://box-a:8000", "the one with a free slot"


def test_the_old_env_name_still_sets_the_knob():
    assert Config.from_env({"LB_ALLOWED_HOSTS": "loopback", "LB_VLLM_MAX_WAITING": "5"}).vllm_max_depth == 5
