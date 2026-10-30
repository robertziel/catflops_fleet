"""No lease, no traffic.

Operator: "remove any not leased inference server!!!" — every backend the LB dials is already
a broker_parrot service, but OWNING a server is not HOLDING it. Two paths took capacity without
telling the broker: `use_unleased_vllm` (tier 2 = any live vLLM), and the ollama tier, which had
no lease condition at all (retired with the fallback, 2026-10-17).
"""
from __future__ import annotations

from vllm_lb.config import Config
from vllm_lb.router import choose
from vllm_lb.state import LEASE_RUNNING, VLLM, Backend, Lease, Snapshot

NOW = 1000.0


def v(url="http://box-a:8000", *, label="box-a-gpu", box="box-a", waiting=0, running=0, live=True):
    b = Backend(kind=VLLM, url=url, host_label=label, box=box)
    b.live, b.served_models, b.waiting, b.running = live, ("qwen3.8:latest",), waiting, running
    return b


def lease(host_label=None, box=None, kind=VLLM):
    return Lease(lease_id="r", job_id="j", created_at=NOW, status=LEASE_RUNNING,
                 host_label=host_label, box=box, kind=kind)


def snap(backends, leases=()):
    return Snapshot(now=NOW, backends=list(backends), leases=list(leases),
                    last_request_at=NOW, last_pressure_at=None, db_ok=True)


def test_the_shipped_defaults_refuse_unleased_servers():
    c = Config(broker_enabled=True)
    assert c.use_unleased_vllm is False


def test_a_live_but_UNLEASED_vllm_gets_no_traffic():
    assert choose(snap([v()]), Config(broker_enabled=True)) is None


def test_the_same_vllm_serves_the_moment_its_lease_is_held():
    s = snap([v()], [lease(host_label="box-a-gpu", box="box-a")])
    assert choose(s, Config(broker_enabled=True)).url == "http://box-a:8000"


def test_an_unleased_vllm_is_not_reachable_through_the_DEEP_tier():
    """THE KILL LINE. `deep` holds backends demoted for queue depth and is consulted after
    every ordinary tier — an exception would hide there."""
    s = snap([v(waiting=99, running=9)])
    assert choose(s, Config(broker_enabled=True, vllm_max_depth=1)) is None


def test_an_unleased_backend_is_not_reachable_through_the_PENALISED_tier():
    """The other bypass: when nothing unpenalised is left, `choose` falls back to penalised
    backends. That must still respect the lease."""
    b = v()
    b.penalty_until = NOW + 60
    assert choose(snap([b]), Config(broker_enabled=True)) is None


def test_a_leased_backend_is_still_reachable_when_penalised():
    b = v()
    b.penalty_until = NOW + 60
    s = snap([b], [lease(host_label="box-a-gpu", box="box-a")])
    assert choose(s, Config(broker_enabled=True)).url == "http://box-a:8000", "a penalty is a preference, not a ban"


def test_from_env_honours_the_shipped_default_when_the_var_is_unset():
    """It hardcoded the string "1", so every deployment that had not set the variable served
    from unleased servers no matter what the class default said — including this one, caught
    on the live box after the policy was already deployed."""
    assert Config.from_env({"LB_ALLOWED_HOSTS": "loopback"}).use_unleased_vllm is False
    assert Config.from_env({"LB_ALLOWED_HOSTS": "loopback", "LB_USE_UNLEASED_VLLM": "1"}).use_unleased_vllm is True
    assert Config.from_env({"LB_ALLOWED_HOSTS": "loopback", "LB_USE_UNLEASED_VLLM": "0"}).use_unleased_vllm is False
