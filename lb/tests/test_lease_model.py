"""Pass 18 — the lease SAYS which model it is for, without telling the CLAIM.

See `worklog/vllm-lb-lease-model.md`. The operator asked why the lease shows "no model".
The answer is not to fill in `required_model`: in `claim_next_gpu_job` that column is a hard
capability FILTER (`node_queue.py:1064-1075`) whose input — the box's advertised
`known_models` — is narrowed by live memory (`claim_worker.py:2771-2820`) and holds
`qwen3.8:latest` on a GB10 box only while the vLLM the lease exists to START is already serving
it. So it can strand the lease precisely when the lease matters.

These tests hold both halves down at once:

  * the model is DECLARED, from config, in `inputs` — a column no claim predicate reads;
  * `required_model` stays NULL, and the engine itself is asked to demonstrate why.

The engine-truth tests are the frame's KILL line, run on every suite pass: if a
`required_model` a box does not advertise stopped making the row unclaimable, this whole
decision would be wrong and the value should simply be set.
"""
from __future__ import annotations

import pytest

from vllm_lb.config import Config
from vllm_lb.leases import BrokerLeases

CFG = Config(lease_max_hold_s=1234, lease_workflow="vllm_lb_lease",
             lease_node_module="system.vllm_lease")


@pytest.fixture
def leases(broker_db):
    return BrokerLeases(CFG, clock=lambda: 100.0)


def _job(job_id):
    from queue_workflows import node_queue
    return node_queue.get_node_job(job_id)


def _name_a_model_on(job_id: str, model: str) -> None:
    """Set `required_model` on an EXISTING lease row — the control that keeps every other
    column byte-identical, so the claim's behaviour can only be about this one value."""
    from queue_workflows import db
    with db.connection() as conn, conn.cursor() as cur:
        cur.execute("UPDATE workflow_node_jobs SET required_model = %(m)s WHERE id = %(id)s",
                    {"m": model, "id": job_id})
        conn.commit()


# ── the model is declared, from config, where nothing can filter on it ────────────────

def test_the_lease_declares_the_model_it_is_for_in_its_inputs(leases):
    lease = leases.create(avoid_boxes=())
    assert _job(lease.job_id)["inputs"]["model"] == "qwen3.8:latest"


def test_the_declared_model_comes_from_config_not_a_hardcoded_string(broker_db):
    """Any row's served name (the lease's window is that row's: a name no row serves gets no lease)."""
    from vllm_lb.catalog import ModelCatalog
    other = next(s.served_name for s in ModelCatalog.load().specs if s.served_name != "qwen3.8:latest")
    cfg = Config(lease_model=other)
    lease = BrokerLeases(cfg, clock=lambda: 100.0).create(avoid_boxes=())
    assert _job(lease.job_id)["inputs"]["model"] == other


def test_LB_LEASE_MODEL_names_the_model_and_the_catalogs_classic_default_is_the_default():
    """Blank: the model a classic vLLM launches — the catalog's classic default's served name (the retired
    LB_FALLBACK_MODEL no longer stands in, and is refused at boot)."""
    import pytest
    from vllm_lb.catalog import CLASSIC, ModelCatalog
    assert Config.from_env({"LB_ALLOWED_HOSTS": "loopback"}).lease_model_id == ModelCatalog.load().default(CLASSIC).served_name == "qwen3.8:latest"
    assert Config.from_env({"LB_ALLOWED_HOSTS": "loopback", "LB_LEASE_MODEL": "y:latest"}).lease_model_id == "y:latest"
    with pytest.raises(ValueError, match="LB_FALLBACK_MODEL is retired"):
        Config.from_env({"LB_ALLOWED_HOSTS": "loopback", "LB_FALLBACK_MODEL": "x:latest"})


def test_declaring_the_model_leaves_the_claim_facing_columns_alone(leases):
    """`inputs` is written at insert and read by no claim predicate, no ORDER BY and no
    sweep. The columns the claim DOES read must be exactly what pass 11/16 left them."""
    job = _job(leases.create(avoid_boxes=()).job_id)
    assert job["required_model"] is None, "worklog/vllm-lb-lease-model.md — a filter, not a hint"
    assert job["server_types"] == ["vllm"] and job["avoid_box"] is None
    assert job["context_length"] == 262144 and job["priority"] == 0


# ── the engine's own proof that naming it would be a filter (the frame's KILL line) ───

def test_a_required_model_the_box_does_not_advertise_makes_the_lease_unclaimable(leases):
    """The stranding case, on the real engine: a GB10 box whose vLLM is DOWN no longer has
    qwen3.8 resident, so the memory gate can drop it from `known_models` — and the row that
    would have started that very vLLM is then claimable by nobody.

    The second claim is the control: the same row, same box, same tick, claimed the moment
    the box advertises the tag. Nothing but the capability gate held it."""
    from tests.fakes import claim_gpu

    lease = leases.create(avoid_boxes=())
    _name_a_model_on(lease.job_id, "qwen3.8:latest")

    assert claim_gpu("box-a-gpu", known_models=["moss-tts-v1.5"]) is None
    claimed = claim_gpu("box-a-gpu", known_models=["moss-tts-v1.5", "qwen3.8:latest"])
    assert claimed is not None and claimed["id"] == lease.job_id


def test_the_lease_we_actually_enqueue_is_claimed_by_that_same_box(leases):
    """The contrast that makes the decision: the row we DO write — identical but for the
    NULL `required_model` — is claimed by the box that just refused the model-named twin."""
    from tests.fakes import claim_gpu

    lease = leases.create(avoid_boxes=())
    claimed = claim_gpu("box-a-gpu", known_models=["moss-tts-v1.5"])
    assert claimed is not None and claimed["id"] == lease.job_id
