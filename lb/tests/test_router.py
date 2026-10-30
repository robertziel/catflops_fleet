"""S2 — the router is a PURE decision over a state snapshot.

Order: a leased + live vLLM (fewest in-flight) → any live vLLM (if allowed) → a demoted one → nothing.
There is NO fallback tier (retired 2026-10-17: "If can't run then can't run"). The request's `model` is
rewritten to what the chosen backend serves.
"""
from vllm_lb.config import Config
from vllm_lb.router import choose, rewrite_model
from vllm_lb.state import Backend, Lease, Snapshot

NOW = 1_000.0
def _auto_leases(backends):
    """Leases for everything present. Since 2026-09-12 the router serves ONLY from backends the
    LB holds a broker lease on (worklog/vllm-lb-leased-only.md), so a helper that omits leases
    would make every tiering test vacuous. Tests that are ABOUT the lease gate pass `leases=`
    explicitly — including an empty list — and this never runs for them."""
    from vllm_lb.state import LEASE_RUNNING, VLLM, Lease
    out = []
    for b in backends:
        if b.kind == VLLM and b.host_label:
            out.append(Lease(lease_id="auto-" + b.host_label, job_id="j", created_at=0,
                             status=LEASE_RUNNING, host_label=b.host_label, box=b.box,
                             kind=VLLM, claimed_at=0))
    return out



def cfg(**over) -> Config:
    return Config(**{"broker_enabled": True, **over})


def vllm(label, url, *, live=True, inflight=0, served=("qwen3.8:latest",), penalty_until=0.0):
    return Backend(kind="vllm", url=url, host_label=label, box=label.removesuffix("-gpu"),
                   live=live, served_models=served, inflight=inflight,
                   penalty_until=penalty_until, max_model_len=131072)


def snap(backends, leases=None, now=NOW):
    leases = _auto_leases(backends) if leases is None else list(leases)
    return Snapshot(now=now, backends=list(backends), leases=list(leases),
                    last_request_at=None, last_pressure_at=None, db_ok=True)


def running_lease(label):
    return Lease(lease_id="r-" + label, job_id="j-" + label, created_at=NOW - 60,
                 status="running", host_label=label, box=label.removesuffix("-gpu"),
                 claimed_at=NOW - 30)


def test_nothing_live_means_none_not_a_crash():
    s = snap([vllm("box-a-gpu", "http://a:8000", live=False), vllm("box-b-gpu", "http://b:8000", live=False)])
    assert choose(s, cfg()) is None


def test_no_fallback_serves_when_no_server_is_live():
    """The fallback tier is retired: with no server live nothing is chosen — the proxy then waits or refuses."""
    s = snap([vllm("box-a-gpu", "http://a:8000", live=False)])
    assert choose(s, cfg()) is None


def test_leased_vllm_beats_unleased_vllm_at_COMPARABLE_load():
    """Tier still decides between backends of similar depth."""
    s = snap([vllm("box-a-gpu", "http://a:8000", inflight=1),
              vllm("box-b-gpu", "http://b:8000", inflight=1)],
             [running_lease("box-a-gpu")])
    assert choose(s, cfg()).url == "http://a:8000"


def test_a_SATURATED_leased_vllm_loses_to_an_idle_unleased_one():
    """Changed 2026-09-12 (worklog/vllm-lb-audit-deltas-11-13.md). This used to assert that a
    lease outranks load unconditionally. It does not any more: depth now counts our own
    in-flight requests, and a vLLM past `vllm_max_depth` is demoted below every ordinary tier.
    Holding the lease does not make a full box answer faster, and the operator's requirement is
    that waiting never becomes failing."""
    s = snap([vllm("box-a-gpu", "http://a:8000", inflight=3),
              vllm("box-b-gpu", "http://b:8000", inflight=0)],
             [running_lease("box-a-gpu"), running_lease("box-b-gpu")])
    assert choose(s, cfg()).url == "http://b:8000"


def test_among_leased_the_least_loaded_wins():
    s = snap([vllm("box-a-gpu", "http://a:8000", inflight=3),
              vllm("box-b-gpu", "http://b:8000", inflight=1)],
             [running_lease("box-a-gpu"), running_lease("box-b-gpu")])
    assert choose(s, cfg()).url == "http://b:8000"


def test_unleased_live_vllm_is_used_ONLY_when_explicitly_allowed():
    """box-a is deliberately NOT leased; box-b is. Since 2026-09-12 the default is
    `use_unleased_vllm=False` — no lease, no traffic (worklog/vllm-lb-leased-only.md) — and the
    flag remains for a box broker_parrot does not manage."""
    # box-b, the leased one, is past its depth (demoted): only an unleased box-a could take the turn ahead of it.
    s = snap([vllm("box-a-gpu", "http://a:8000"), vllm("box-b-gpu", "http://b:8000", inflight=3)],
             [running_lease("box-b-gpu")])
    assert choose(s, cfg(use_unleased_vllm=True)).url == "http://a:8000"
    assert choose(s, cfg(use_unleased_vllm=False)).url == "http://b:8000"


def test_a_lease_that_is_still_queued_does_not_make_its_box_leased():
    queued = Lease(lease_id="r", job_id="j", created_at=NOW, status="queued")
    s = snap([vllm("box-a-gpu", "http://a:8000")], [queued])
    assert choose(s, cfg()) is None


def test_a_penalised_backend_is_skipped_until_its_penalty_expires():
    s = snap([vllm("box-a-gpu", "http://a:8000", penalty_until=NOW + 5), vllm("box-b-gpu", "http://b:8000")],
             [running_lease("box-a-gpu"), running_lease("box-b-gpu")])
    assert choose(s, cfg()).url == "http://b:8000"
    assert choose(snap(s.backends, s.leases, now=NOW + 6), cfg()).url == "http://a:8000"


def test_exclude_walks_down_the_tiers_for_failover():
    s = snap([vllm("box-a-gpu", "http://a:8000"), vllm("box-b-gpu", "http://b:8000", inflight=1)],
             [running_lease("box-a-gpu"), running_lease("box-b-gpu")])
    assert choose(s, cfg(), exclude=frozenset({"http://a:8000"})).url == "http://b:8000"
    assert choose(s, cfg(), exclude=frozenset({"http://a:8000", "http://b:8000"})) is None


def test_model_is_rewritten_to_what_the_backend_serves():
    body = {"model": "qwen3.6", "messages": []}
    out = rewrite_model(body, vllm("box-a-gpu", "http://a:8000"), cfg())
    assert out["model"] == "qwen3.8:latest" and out["messages"] == []
    assert body["model"] == "qwen3.6", "the caller's dict is not mutated"


def test_model_is_kept_when_the_backend_already_serves_it():
    body = {"model": "qwen3.8:latest"}
    assert rewrite_model(body, vllm("box-a-gpu", "http://a:8000"), cfg()) is body


def test_a_vllm_with_unknown_served_models_leaves_the_model_alone():
    b = vllm("box-a-gpu", "http://a:8000", served=())
    body = {"model": "x"}
    assert rewrite_model(body, b, cfg()) is body
