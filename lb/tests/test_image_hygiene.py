"""What the LB image may contain. `COPY . /app` takes the whole build context, and lb/ is where the
operator keeps the secrets: `.env` (the broker DB password), `cluster-ssh/` (the key that logs into
both GB10 boxes) and `state/` (the cluster switch, and backups of `.env`). All three are MOUNTED at run
time (docker-compose.yml); none may be baked into an image layer. Until 2026-09-15 all three were."""
from __future__ import annotations

import fnmatch
from pathlib import Path

LB = Path(__file__).resolve().parent.parent
#: `fleet.yml` is this deployment's fleet file (vllm_lb/fleet.py): its boxes' hostnames, SSH targets and addresses.
#: Mounted read-only at LB_FLEET like .env is read; never a layer. The committed `*.example.yml` are generic.
SECRETS = (".env", "cluster-ssh/id_ed25519", "cluster-ssh/known_hosts", "state/cluster.json",
           "state/env.before-cluster-rebuild", "fleet.yml")
RUNTIME = ("vllm_lb/__main__.py", "vllm_lb/static/index.html",
           # the recorder's figure table is fleet_telemetry's (one definition, vllm_lb/recorder.py imports it)
           "fleet_telemetry/figures.py",
           "vllm_lb/static/benchmarks.html", "vllm_lb/static/benchmarks.js",
           "vllm_lb/static/benchmarks.css", "cluster/tp-rank")


def patterns() -> list[str]:
    f = LB / ".dockerignore"
    assert f.is_file(), "no lb/.dockerignore: `COPY . /app` bakes every file in lb/ into the image"
    return [line.strip() for line in f.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


def excluded(rel: str, pats: list[str]) -> bool:
    """Docker's rule for the plain patterns this file uses: the last pattern matching the path, or
    any directory above it, decides; `!` re-includes."""
    parts = rel.split("/")
    prefixes = ["/".join(parts[:i]) for i in range(1, len(parts) + 1)]
    out = False
    for pat in pats:
        keep = pat.startswith("!")
        glob = (pat[1:] if keep else pat).strip("/")
        if any(fnmatch.fnmatchcase(prefix, glob) for prefix in prefixes):
            out = not keep
    return out


def test_no_secret_is_part_of_the_build_context():
    pats = patterns()
    assert [s for s in SECRETS if not excluded(s, pats)] == []


#: What the running LB never imports: the test suite (the README runs it from a mount) and bytecode left
#: by host-side runs. The second audit of worklog/vllm-gateway-deep-audit.md found the image carrying both —
#: `.pyc` files for tests that no longer exist (`test_box_lease_toggle`, `test_cluster_mode`).
NOT_SHIPPED = ("tests/test_proxy.py", "tests/fixtures/qwen38-start-2026-09-15-rank0.log",
               "vllm_lb/__pycache__/proxy.cpython-312.pyc", "__pycache__/conftest.cpython-312.pyc",
               # The host half of the swarm button: both run on box-c, under the operator's
               # account, and the LB never imports either of them.
               "bench_queue_runner.py", "bench_session.py",
               # The hardware SENDER's image files (lb/hw_sender, tests/test_hw_sender.py): a server's
               # lb/hw_sender/.env is that server's configuration, not a layer of this image.
               "hw_sender/.env", "hw_sender/Dockerfile",
               # The SEALED swarm runner is its own image as well (lb/bench_runner, tests/test_bench_runner_image.py).
               "bench_runner/egress.py", "bench_runner/.env", "bench_runner/Dockerfile")


def test_the_image_carries_no_tests_and_no_stale_bytecode():
    pats = patterns()
    assert [p for p in NOT_SHIPPED if not excluded(p, pats)] == []


def test_what_the_lb_runs_is_still_built_in():
    pats = patterns()
    assert [p for p in RUNTIME if excluded(p, pats)] == []
