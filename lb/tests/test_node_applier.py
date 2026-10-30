"""THE HOST'S APPLIER for a node the LB requests (fleet control `local`; vllm_lb/node_applier.py), run for real against
a copy of a server's layout: a real fleet file and the shipped catalog, a fake `up.sh` that records what it was asked
and the node.env it was handed, and a fake `docker` that says what the node container is doing.

  * a new intent is rendered by the FLEET's validator and renderer (fleet.node_env — the same code the LB ran) into
    node.env, then `up.sh … node` (or node-stop) runs once; the report carries the exit code and stderr tail;
  * an intent the fleet refuses — a key it may not move, a figure out of the catalog's bounds, a malformed file — runs
    NOTHING and is reported in the validator's own words;
  * a RELAUNCH while the node serves requests is refused unless the intent says force; a node that does not answer
    serves nothing;
  * node.env can carry nothing compose would interpolate or split, whatever the renderer produced;
  * every pass is a heartbeat, with the node container's state and log tail.
"""
from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from tests.test_fleet_local import LOCAL
from vllm_lb import node_applier, node_control as nc
from vllm_lb.fleet import node_env, parse_fleet

UP_SH = """#!/usr/bin/env bash
here="$(cd "$(dirname "$0")" && pwd)"
cp "$here/../../lb/state/node/report.json" "$here/report.during" 2>/dev/null
{ printf 'ARGS %s\\n' "$*"; cat "$here/../../lb/state/node/node.env"; } >> "$here/up.calls"
[ -n "${FAKE_UP_STDERR:-}" ] && echo "$FAKE_UP_STDERR" >&2
exit "${FAKE_UP_EXIT:-0}"
"""
DOCKER = """#!/usr/bin/env python3
import os, sys
if sys.argv[1] == "inspect":
    state = os.environ.get("FAKE_DOCKER_STATE", "")
    if not state:
        sys.exit(1)
    print(state)
elif sys.argv[1] == "logs":
    print(os.environ.get("FAKE_DOCKER_LOG", ""))
"""


@pytest.fixture
def server(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "lb" / "state" / nc.NODE_DIR).mkdir(parents=True)
    (repo / "lb" / "fleet.yml").write_text(LOCAL)
    standalone = repo / "deploy" / "standalone"
    standalone.mkdir(parents=True)
    (standalone / "up.sh").write_text(UP_SH)
    (standalone / "up.sh").chmod(0o755)
    env = standalone / ".env"
    env.write_text("EDGE_MODELS_DIR=/srv/models\nEDGE_NODE_NAME=node-test\nEDGE_NODE_PUBLISH=127.0.0.1:8020\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "docker").write_text(DOCKER)
    (bin_dir / "docker").chmod(0o755 | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    for k in ("FAKE_UP_EXIT", "FAKE_UP_STDERR", "FAKE_DOCKER_STATE", "FAKE_DOCKER_LOG"):
        monkeypatch.delenv(k, raising=False)
    return repo


def write_intent(repo: Path, **over) -> dict:
    doc = {"id": 1, "at": 1_791_742_000.0, "state": "up", "force": False, "launch": {"max_num_seqs": 2}, **over}
    (repo / "lb" / "state" / nc.NODE_DIR / nc.INTENT).write_text(json.dumps(doc))
    return doc


def calls(repo: Path) -> list[str]:
    path = repo / "deploy" / "standalone" / "up.calls"
    return path.read_text().splitlines() if path.exists() else []


def run(repo: Path, **kw) -> dict:
    return node_applier.run_pass(repo, repo / "deploy" / "standalone" / ".env", **kw)


def metrics(running: int, waiting: int):
    return lambda url: f"vllm:num_requests_running{{engine=\"0\"}} {running}\nvllm:num_requests_waiting{{engine=\"0\"}} {waiting}\n"


def test_a_new_intent_is_rendered_by_the_fleet_and_brought_up_once(server):
    write_intent(server)
    report = run(server)
    applied = report["applied"]
    assert (applied["id"], applied["state"], applied["exit"], applied["refused"]) == (1, "up", 0, "")
    assert applied["launch"] == {"max_num_seqs": 2}, "what was applied, for the LB to check the node against"
    assert applied["restarts_at_apply"] is None, "no container after the command: nothing counted"
    env = server / "deploy" / "standalone" / ".env"
    want = node_env(parse_fleet(LOCAL, "fleet.yml"), "node", {"max_num_seqs": 2})
    fragment = server / "lb" / "state" / nc.NODE_DIR / nc.FRAGMENT
    # up.sh is TOLD which node.env it applies (`--node-env`): with several clusters each has its own node dir.
    assert calls(server) == [f"ARGS --env {env} --node-env {fragment} node", *want], \
        "node.env is the fleet's own rendering of the intent"
    assert "EDGE_NODE_MAX_NUM_SEQS=2" in want
    assert (server / "lb" / "state" / nc.NODE_DIR / nc.FRAGMENT).read_text().splitlines() == want
    during = json.loads((server / "deploy" / "standalone" / "report.during").read_text())
    assert during["applying"] == {"id": 1, "at": 1_791_742_000.0, "state": "up", "force": False,
                                  "since": during["heartbeat"]}, "said BEFORE up.sh runs, which may pull for minutes"
    assert report["applying"] is None
    run(server)
    assert len([c for c in calls(server) if c.startswith("ARGS")]) == 1, "an intent is applied once"
    assert nc.read_report(str(server / "lb" / "state" / nc.NODE_DIR), now=report["heartbeat"])["listening"]


def test_the_containers_restart_count_is_taken_when_the_intent_is_applied(server, monkeypatch):
    """A node that crash-loops after its intent shows `running` between attempts: only its RestartCount, against the
    count right after the command, says so (cluster.plan_local)."""
    monkeypatch.setenv("FAKE_DOCKER_STATE", "running|0|3|2026-10-11T20:51:42Z")
    write_intent(server, state="down", launch={})
    assert run(server, fetch=metrics(0, 0))["applied"]["restarts_at_apply"] == 3


@pytest.mark.parametrize("launch, says", [
    ({"image": "evil/image@sha256:" + "0" * 64}, "the fleet file's alone"),
    ({"tool_call_parser": "hermes"}, "the fleet file's alone"),
    ({"max_num_seqs": 999}, "max_num_seqs must be a whole number in"),
    ({"gpu_memory_utilization": 1.5}, "gpu_memory_utilization must be a fraction"),
    ({"max_model_len": 10_000_000}, "max_model_len must be a whole number of tokens up to"),
    ({"seats": 4}, "is no launch key"),
])
def test_an_intent_the_fleet_refuses_runs_nothing_and_says_why_in_its_words(server, launch, says):
    write_intent(server, launch=launch)
    applied = run(server)["applied"]
    assert says in applied["refused"] and applied["exit"] is None
    assert calls(server) == [] and not (server / "lb" / "state" / nc.NODE_DIR / nc.FRAGMENT).exists()


def test_a_malformed_intent_file_runs_nothing_and_is_said(server):
    (server / "lb" / "state" / nc.NODE_DIR / nc.INTENT).write_text('{"id": 1, "state": "up", "rm": "-rf"}')
    applied = run(server)["applied"]
    assert "intent.json" in applied["refused"] and calls(server) == []


def test_a_relaunch_while_the_node_serves_is_refused_unless_forced(server, monkeypatch):
    monkeypatch.setenv("FAKE_DOCKER_STATE", "running|0|0|2026-10-11T20:00:00Z")
    write_intent(server)
    applied = run(server, fetch=metrics(2, 1))["applied"]
    assert applied["refused"] == ("3 request(s) in flight on the node (running 2, waiting 1): not relaunched — "
                                  "send it again with force to cut them")
    assert calls(server) == []
    write_intent(server, id=2, force=True)
    applied = run(server, fetch=metrics(2, 1))["applied"]
    assert (applied["id"], applied["exit"], applied["refused"]) == (2, 0, "")


def test_an_idle_node_or_one_that_does_not_answer_is_relaunched(server, monkeypatch):
    monkeypatch.setenv("FAKE_DOCKER_STATE", "running|0|0|2026-10-11T20:00:00Z")
    write_intent(server)
    assert run(server, fetch=metrics(0, 0))["applied"]["exit"] == 0

    def refused(url):
        raise ConnectionRefusedError(111, "Connection refused")
    write_intent(server, id=2)
    assert run(server, fetch=refused)["applied"]["exit"] == 0


def test_down_stops_the_node(server):
    write_intent(server, state="down", launch={})
    assert run(server)["applied"]["exit"] == 0
    assert calls(server)[0].endswith(" node-stop")


def test_a_failed_bring_up_reports_its_exit_code_and_stderr(server, monkeypatch):
    monkeypatch.setenv("FAKE_UP_EXIT", "1")
    monkeypatch.setenv("FAKE_UP_STDERR", "Error response from daemon: could not select device driver")
    write_intent(server)
    applied = run(server)["applied"]
    assert applied["exit"] == 1 and "could not select device driver" in applied["stderr"]


def test_node_env_can_carry_nothing_compose_would_interpolate_or_split(server, monkeypatch):
    from vllm_lb import fleet
    monkeypatch.setattr(fleet, "node_env", lambda *a, **k: ["EDGE_NODE_SERVED_NAME=x${HOME}", "EDGE_NODE_GPU=1"])
    write_intent(server)
    applied = run(server)["applied"]
    assert "cannot carry safely" in applied["refused"] and calls(server) == []
    for bad in ("EDGE_NODE_X=a b", "EDGE_NODE_X=a\nEDGE_NODE_GPU=0", "EDGE_NODE_X=$(id)", "EDGE_NODE_X='q'",
                "OTHER=1", "EDGE_NODE_X="):
        assert not node_applier.SAFE_LINE.fullmatch(bad), bad
        assert not node_applier.safe_line(bad), bad


def test_an_optional_launch_key_left_to_vllm_is_carried_empty_and_nothing_else_is(server):
    """The prefill chunk left to vLLM's own default is rendered EMPTY (fleet.OPTIONAL_NODE_ENV): node.env is read after
    the server file, so an empty line is what keeps a server file's own value from standing in for vLLM's default. A
    required key may never be empty — compose's `:?` would refuse it — so only the optional ones may."""
    write_intent(server, launch={"max_num_seqs": 2, "max_num_batched_tokens": None})
    applied = run(server)["applied"]
    assert (applied["exit"], applied["refused"]) == (0, "")
    lines = (server / "lb" / "state" / nc.NODE_DIR / nc.FRAGMENT).read_text().splitlines()
    assert "EDGE_NODE_MAX_NUM_BATCHED_TOKENS=" in lines
    write_intent(server, id=2, launch={"max_num_batched_tokens": 1024})
    run(server)
    lines = (server / "lb" / "state" / nc.NODE_DIR / nc.FRAGMENT).read_text().splitlines()
    assert "EDGE_NODE_MAX_NUM_BATCHED_TOKENS=1024" in lines
    assert node_applier.safe_line("EDGE_NODE_MAX_NUM_BATCHED_TOKENS=")
    assert not node_applier.safe_line("EDGE_NODE_MAX_MODEL_LEN=") and not node_applier.safe_line("EDGE_NODE_GPU=")



def test_the_pinned_pool_and_the_ram_tier_reach_node_env_and_their_absence_is_empty(server):
    """kv_cache_memory and kv_offload_ram_gib are OPTIONAL node lines too (fleet.OPTIONAL_NODE_ENV): empty when left to
    vLLM, the number when set — the relaunch the coordinator posts, on a box whose fleet bounds the tier."""
    from tests.test_fleet_local import RAMMED, RELAUNCH
    (server / "lb" / "fleet.yml").write_text(RAMMED)
    write_intent(server, launch={"max_num_seqs": 2})
    run(server)
    lines = (server / "lb" / "state" / nc.NODE_DIR / nc.FRAGMENT).read_text().splitlines()
    assert "EDGE_NODE_KV_CACHE_MEMORY=" in lines and "EDGE_NODE_KV_OFFLOAD_RAM_GIB=" in lines
    write_intent(server, id=2, launch=RELAUNCH)
    applied = run(server)["applied"]
    assert (applied["exit"], applied["refused"]) == (0, ""), applied
    lines = (server / "lb" / "state" / nc.NODE_DIR / nc.FRAGMENT).read_text().splitlines()
    assert "EDGE_NODE_KV_CACHE_MEMORY=8501810176" in lines and "EDGE_NODE_KV_OFFLOAD_RAM_GIB=32" in lines
    assert all(node_applier.safe_line(f"{k}=") for k in ("EDGE_NODE_KV_CACHE_MEMORY", "EDGE_NODE_KV_OFFLOAD_RAM_GIB"))
    write_intent(server, id=3, launch={"kv_offload_ram_gib": 33})
    assert "kv_offload_ram_max_gib 32" in run(server)["applied"]["refused"], "the host refuses past the bound too"

def test_the_kv_cache_dtype_an_intent_asks_is_checked_against_the_nodes_image_then_rendered(server):
    """kv_cache_dtype (vLLM --kv-cache-dtype) is read where max_num_seqs is — the intent's launch — and judged by the
    fleet against what the node's IMAGE accepts (vllm_lb/kv_cache_dtypes.json): nvfp4 on the CUDA build is
    rendered; on the ROCm build, whose CacheDType has no fp4, it is refused in the fleet's words and nothing runs."""
    from tests.test_fleet_local import DIGEST
    from tests.test_node_kv_dtype import CUDA, ROCM
    fragment = server / "lb" / "state" / nc.NODE_DIR / nc.FRAGMENT
    (server / "lb" / "fleet.yml").write_text(LOCAL.replace(DIGEST, CUDA))
    write_intent(server, launch={"max_num_seqs": 2})
    run(server)
    assert "EDGE_NODE_KV_CACHE_DTYPE=" in fragment.read_text().splitlines(), "absent: vLLM's own, carried empty"
    write_intent(server, id=2, launch={"kv_cache_dtype": "nvfp4"})
    applied = run(server)["applied"]
    assert (applied["exit"], applied["refused"]) == (0, ""), applied
    assert "EDGE_NODE_KV_CACHE_DTYPE=nvfp4" in fragment.read_text().splitlines()
    (server / "lb" / "fleet.yml").write_text(LOCAL.replace(DIGEST, ROCM))
    before = calls(server)
    write_intent(server, id=3, launch={"kv_cache_dtype": "nvfp4"})
    refused = run(server)["applied"]["refused"]
    assert "kv_cache_dtype" in refused and "no FP4 KV cache" in refused and calls(server) == before


def test_every_pass_is_a_heartbeat_with_the_containers_state_and_log(server, monkeypatch):
    monkeypatch.setenv("FAKE_DOCKER_STATE", "restarting|1|7|2026-10-11T20:00:00Z")
    monkeypatch.setenv("FAKE_DOCKER_LOG", "ERROR torch.OutOfMemoryError: CUDA out of memory")
    report = run(server)
    assert report["applied"] is None, "no intent: nothing applied"
    assert report["container"] == {"status": "restarting", "exit_code": 1, "restarts": 7,
                                   "started_at": "2026-10-11T20:00:00Z"}
    assert "CUDA out of memory" in report["log"]
    monkeypatch.setenv("FAKE_DOCKER_STATE", "")
    assert run(server)["container"] is None and run(server)["log"] is None, "no container: nothing to read"


def test_the_pass_fails_loudly_only_when_it_cannot_read_its_own_inputs(server, capsys):
    assert node_applier.main(["--repo", str(server), "--env", str(server / "missing.env")]) == 1
    assert "missing.env" in capsys.readouterr().err
    assert node_applier.main(["--repo", str(server)]) == 0


# ── several clusters: the applier is run PER CLUSTER (`--cluster`), in that cluster's node dir ─────────────────────

GUI_UP_SH = """#!/usr/bin/env bash
here="$(cd "$(dirname "$0")" && pwd)"
printf 'ARGS %s\\n' "$*" >> "$here/up.calls"
"""


def test_a_named_cluster_is_applied_from_its_own_node_dir_and_up_is_handed_its_node_env(server):
    """A fleet of several: the gui node's intent is lb/state/clusters/gui/node/intent.json (node_control.state_path_for,
    the LB's own rule); the pass renders into THAT dir's node.env, hands it to up.sh by name, and reports there — the
    one-cluster dir is never read or written."""
    from tests.fleet_files import PARALLEL
    (server / "lb" / "fleet.yml").write_text(PARALLEL)
    (server / "deploy" / "standalone" / "up.sh").write_text(GUI_UP_SH)
    mine = server / "lb" / "state" / "clusters" / "gui" / nc.NODE_DIR
    mine.mkdir(parents=True)
    doc = {"id": 7, "at": 1_791_742_000.0, "state": "up", "force": False, "launch": {}}
    (mine / nc.INTENT).write_text(json.dumps(doc))
    report = run(server, cluster="gui")
    assert report["applied"]["id"] == 7 and report["applied"]["exit"] == 0, report["applied"]
    assert calls(server) == [f"ARGS --env {server / 'deploy/standalone/.env'} --node-env {mine / nc.FRAGMENT} node"]
    assert (mine / nc.FRAGMENT).read_text().startswith("EDGE_NODE_IMAGE=")
    assert json.loads((mine / nc.REPORT).read_text())["applied"]["id"] == 7
    assert not (server / "lb" / "state" / nc.NODE_DIR / nc.REPORT).exists(), "the one-cluster dir is not this one"
    assert run(server)["applied"]["id"] == 7 and len(calls(server)) == 1, "the fleet's one local node: the same dir"


def test_node_dir_prints_the_clusters_node_dir_for_the_installer(server, capsys):
    """The installer asks the applier where a cluster's intent lives (`--node-dir`) — the LB's rule, never re-derived
    in shell — to render the .path unit's watch and check the dir is writable."""
    from tests.fleet_files import PARALLEL
    (server / "lb" / "fleet.yml").write_text(PARALLEL)
    assert node_applier.main(["--repo", str(server), "--cluster", "gui", "--node-dir"]) == 0
    assert capsys.readouterr().out.strip() == str(server / "lb" / "state" / "clusters" / "gui" / nc.NODE_DIR)
    assert not (server / "deploy" / "standalone" / "up.calls").exists(), "--node-dir runs no pass"
    assert node_applier.main(["--repo", str(server), "--cluster", "pair", "--node-dir"]) == 1
    assert "pair" in capsys.readouterr().err


def test_a_node_on_another_box_is_driven_through_that_boxs_docker(server, tmp_path):
    """EDGE_NODE_DOCKER_HOST in the server file (a node whose box is not the applier's, e.g. ssh://user@host): every
    docker call of the pass — up.sh's and the applier's own inspect and logs — goes to that daemon."""
    seen = tmp_path / "docker.hosts"
    (tmp_path / "bin" / "docker").write_text(DOCKER.replace("import os, sys\n", "import os, sys\nopen(%r, 'a').write("
                                                            "os.environ.get('DOCKER_HOST', '-') + '\\n')\n" % str(seen)))
    (server / "deploy" / "standalone" / "up.sh").write_text(UP_SH.replace(
        'exit "${FAKE_UP_EXIT:-0}"', 'echo "UPSH_DOCKER_HOST=${DOCKER_HOST:--}" >> "$here/up.calls"\nexit "${FAKE_UP_EXIT:-0}"'))
    env = server / "deploy" / "standalone" / ".env"
    env.write_text(env.read_text() + "EDGE_NODE_DOCKER_HOST=ssh://operator@node-2\n")
    write_intent(server)
    run(server)
    assert "UPSH_DOCKER_HOST=ssh://operator@node-2" in calls(server)
    hosts = seen.read_text().split()
    assert hosts and set(hosts) == {"ssh://operator@node-2"}
    assert "DOCKER_HOST" not in os.environ, "the pass names the daemon per call, never for the process"
