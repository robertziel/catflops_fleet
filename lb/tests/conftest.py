"""A throwaway sqlite broker DB per test — the same engine calls the GB10 boxes' lanes make,
no Postgres, nothing shared. Mirrors broker_parrot's own sqlite test mode.

broker_parrot is OPT-IN (LB_BROKER_ENABLED): the LB runs without it, and so does this suite. A test that
needs the engine asks for it through `broker_db` (or `needs_broker`), and is SKIPPED, with the reason, when
/opt/broker_parrot is not mounted — the README's test command mounts it; tests/test_broker_off.py runs
either way.
"""
from __future__ import annotations

import os
import tempfile

import pytest

#: Why a broker test did not run. Said once, in the skip line, so a short count is never a silent one.
NO_ENGINE = ("broker_parrot's engine (queue_workflows) is not importable — mount the checkout at "
             "/opt/broker_parrot to run the LB_BROKER_ENABLED=1 tests (README.md, Tests)")


def needs_broker() -> None:
    """Skip the calling test unless broker_parrot's engine can be imported."""
    pytest.importorskip("queue_workflows", reason=NO_ENGINE)

#: Where a cluster checkpoint is searched for on each box, best first — lb/.env's MODELS_SEARCH_DIRS. Test data: the
#: layout this suite was written against; a deployment states its own (vllm_lb.catalog.MODELS_SEARCH_DIRS).
os.environ["MODELS_SEARCH_DIRS"] = "~/models:/mnt/box-a-home/models:/srv/models"
#: The deployment the model catalog RESOLVES its images against (vllm_lb.catalog.deployment_targets): the reference
#: fleet the suite is written against — the committed example, a TP=2 pair of GB10s and a classic server on each.
os.environ["LB_FLEET"] = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fleet.example.yml")


@pytest.fixture(autouse=True)
def reference_checkpoints(monkeypatch):
    """THE SUITE'S FLEETS STAGE THE REFERENCE FOLDERS. Where a checkpoint sits is a fleet file's `checkpoints` block
    (since 2026-10-30; vllm_lb.fleet.catalog_deployment), and the suite's many inline fleets were written before it:
    a test fleet that states NO such block is read as staging what the reference fleet (lb/fleet.example.yml) stages.
    Test scaffolding only — the LB itself refuses a model whose folder its fleet does not name (tests/test_library.py
    proves it with an explicit `checkpoints: {}`)."""
    from vllm_lb import fleet, yamlfile
    staged = yamlfile.read(os.environ["LB_FLEET"])["checkpoints"]
    real = fleet.catalog_deployment

    def with_reference(raw, source, library=None):
        if isinstance(raw, dict) and "checkpoints" not in raw:
            raw = {**raw, "checkpoints": staged}
        return real(raw, source, library)

    monkeypatch.setattr(fleet, "catalog_deployment", with_reference)


@pytest.fixture
def broker_db(monkeypatch):
    needs_broker()
    path = os.path.join(tempfile.mkdtemp(prefix="lb_broker_"), "broker.db")
    monkeypatch.setenv("LB_TEST_DB", path)
    import queue_workflows as qw
    from queue_workflows import config as _cfg, db

    qw.configure(db_backend="sqlite", db_url_env="LB_TEST_DB")
    db.bootstrap()
    yield path
    db.close_pool()
    _cfg.reset_for_tests()
