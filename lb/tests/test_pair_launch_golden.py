"""THE GB10 PAIR LAUNCHES BYTE FOR BYTE WHAT IT LAUNCHED BEFORE THE ENGINE REDESIGN (operator, 2026-10-30).

Phase 5 of docs/design/engines.md generalised tp-rank's DGX Spark assumptions into FACTS the LB sends it (the TP size,
the node count, the GPUs, the link, the memory family, the platform — vllm_lb/executors.py). This test pins what that
must not change: for every cluster profile of the reference fleet (lb/fleet.example.yml, a TP=2 GB10 pair), each rank's
`docker run` argv as tp-rank builds it from the LB's own launch settings (`cluster_exec.launch_env`) — or, where the
fake box cannot launch it (a checkpoint layout it does not hold), tp-rank's own refusal — recorded at commit 65b24de1
(before phase 5) in tests/fixtures/pair-launch-golden.json and compared after.

Every profile launches against the fake box's ONE laid-out checkpoint (Flash-Next hibrid48, as test_tp_rank's box): the
model folder is the only thing held constant across profiles; image, engine, flags, parsers, drafter, KV dtype, pin and
every docker option are each profile's own. Regenerate ONLY to record an intended change:
    TP_RANK_GOLDEN_WRITE=1 python -m pytest -n 0 tests/test_pair_launch_golden.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from tests.test_tp_rank import MODEL, box_a  # noqa: F401  (the fake GB10 box fixture)
from vllm_lb.catalog import CLUSTER_MODE, ModelCatalog
from vllm_lb.cluster_exec import launch_env

#: The pair's RoCE ports as lb/.env names them (tests/test_tensorfold_engine.py FABRIC): sent on every call.
FABRIC = {"CLUSTER_IB_HCA": "mlx_a,mlx_b", "CLUSTER_IB_GID_INDEX": "3"}
GOLDEN = Path(__file__).resolve().parent / "fixtures" / "pair-launch-golden.json"
#: Where the fake box keeps paths that differ per run (its tmp tree): replaced by a stable word.
_STABLE = "<box>"


def _stable(value, root: str):
    if isinstance(value, list):
        return [_stable(v, root) for v in value]
    return value.replace(root, _STABLE) if isinstance(value, str) else value


def pair_launches(box) -> dict:
    """{profile id: {rank: argv | "refused: …"}} for every cluster profile of the shipped catalog on the pair."""
    out = {}
    catalog = ModelCatalog.load()
    # Every vLLM image's CacheDType, read as the box reads the default one (its list is the v5 image's): the KV dtype
    # check then passes or refuses on the profile's dtype, as on the pair.
    site = box._state()["vllm_site"]["myllmbox/qwen38-flash-next-cluster-vllm:v5"]
    box.set(vllm_site={p.image: site for s in catalog.for_mode(CLUSTER_MODE)
                       for p in (s.profile(CLUSTER_MODE),) if p.engine == "vllm"})
    for spec in catalog.for_mode(CLUSTER_MODE):
        env = launch_env(spec)
        env["CLUSTER_MODEL_NAME"] = MODEL                  # the box's one checkpoint (module docstring)
        env["CLUSTER_MODELS_DIRS"] = str(box.models)
        if env.get("CLUSTER_DRAFT_NAME"):              # a drafter checkpoint beside it, as the pair stages one
            drafter = box.models / env["CLUSTER_DRAFT_NAME"]
            drafter.mkdir(exist_ok=True)
            (drafter / "config.json").write_text('{"architectures": ["DFlash2DraftModel"]}')
        ranks = {}
        for rank in (0, 1):
            box.set(containers={})                     # each launch on an empty box
            proc = box.run("up", str(rank), "7", "8", "131072", spec.id,
                           config={"CLUSTER_HOST_IP": "192.0.2.201", **FABRIC, **env})
            runs = [c for c in box.calls() if c and c[0] == "run"]
            ranks[str(rank)] = (_stable(runs[-1], str(box.root)) if proc.returncode == 0 and runs else
                                f"refused: {_stable(proc.stdout.strip(), str(box.root))}")
            (box.root / "docker-calls.jsonl").write_text("")
        out[spec.id] = ranks
    return out


def test_every_pair_profile_launches_exactly_as_before_the_redesign(box_a):  # noqa: F811
    got = pair_launches(box_a)
    if os.environ.get("TP_RANK_GOLDEN_WRITE") == "1":
        GOLDEN.write_text(json.dumps(got, indent=1, sort_keys=True) + "\n")
    want = json.loads(GOLDEN.read_text())
    assert sorted(got) == sorted(want)
    for pid in want:
        assert got[pid] == want[pid], pid
    launched = sum(isinstance(a, list) for r in want.values() for a in r.values())
    assert launched >= 8, f"the golden must pin real launches, not refusals alone ({launched})"
