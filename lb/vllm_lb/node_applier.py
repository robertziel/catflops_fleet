"""THE HOST'S HALF of a one-host node the LB requests (fleet control `local`): one PASS of the node applier.

Run on the node's host — never in the LB — by `systemd --user`, one unit set PER CLUSTER
(vllm-node-applier-<cluster>.{service,path,timer}, rendered from deploy/standalone/vllm-node-applier.*.in by
deploy/standalone/install-node-applier.sh --cluster <cluster>) with the host's python and this checkout's lb/ on
PYTHONPATH:

    python3 -m vllm_lb.node_applier --repo <checkout> [--env <server file>] [--cluster <name>] [--node-dir]

`--cluster`: the fleet cluster this pass applies (required when the fleet names several local nodes); its node dir is
the LB's own rule (`state_dir_of`). `--node-dir` prints that dir and exits — the installer renders its units from it.

The .path unit runs a pass when the LB writes an intent; the .timer runs one every node_control.APPLIER_PERIOD_S, which
is the heartbeat (and picks up an intent written while a pass ran). A pass:

  1. reads the intent (vllm_lb/node_control.py `parse_intent`, the LB's own reading) — nothing to do unless it is one
     this applier has not applied (`same_intent`: its (id, at));
  2. renders the node's env with the FLEET's validator and renderer — `fleet.node_env(fleet, <its control-local
     cluster>, intent.launch)`, i.e. `validate_launch`: the launch block's own rules, the catalog's limits and whatever
     else the fleet checks (the card's fit) — the same code the LB ran before it wrote. A refusal is reported in its
     words and nothing runs;
  3. refuses a RELAUNCH — the node container is running — while the node has requests in flight, unless the intent
     says `force` (operator, 2026-10-11: a restart cut five requests). In flight is the node's OWN count, off its
     /metrics at EDGE_NODE_PUBLISH (vllm:num_requests_running + waiting, read by the LB's own parser); a node that does
     not answer serves nothing;
  4. writes node.env (atomic) — every line `EDGE_NODE_<KEY>=<value>` with the value in [A-Za-z0-9._:/@+-] alone, so
     nothing in it can be interpolated by compose or break a line, whatever the fleet's renderer produced; EMPTY only
     for an optional key left to vLLM's own default (fleet.OPTIONAL_NODE_ENV: the prefill chunk) — and runs
     deploy/standalone/up.sh --env <server file> --node-env <its node.env> node (state up; compose reads node.env as a SECOND --env-file, later
     wins, so a hand-run `up.sh node` converges on it) or node-stop (state down), and keeps its exit code and stderr.

Every pass then reads the node container's docker state and log tail, and writes the report (node_control.report_doc)
whole. The pass itself fails (exit 1, journal) only when it cannot read its own inputs — no server file, no fleet.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from vllm_lb import node_control as nc

#: A line node.env may carry: a node key, and a value compose can neither interpolate (`$`) nor split.
SAFE_LINE = re.compile(r"EDGE_NODE_[A-Z0-9_]+=[A-Za-z0-9._:/@+-]+")
DOCKER_TIMEOUT_S = 30


def safe_line(line: str) -> bool:
    """A SAFE_LINE, or an OPTIONAL key's empty line (fleet.OPTIONAL_NODE_ENV): left to vLLM's own default, and empty
    so a server file's own value cannot stand in for it. A required key is never empty — compose's `:?` refuses it."""
    from vllm_lb.fleet import OPTIONAL_NODE_ENV
    return bool(SAFE_LINE.fullmatch(line)) or (line.endswith("=") and line[:-1] in OPTIONAL_NODE_ENV)


def read_server_file(path: Path) -> dict[str, str]:
    """KEY=VALUE lines, read the way up.sh reads them — never sourced, nothing in it executed."""
    out = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"not KEY=VALUE in {path}: {line}")
        key, _, value = line.partition("=")
        out[key] = value
    return out


def cluster_of(repo: Path, server: dict[str, str], name: str | None, catalog: dict | None = None):
    """(the fleet, the name of the cluster THIS applier applies): a node the LB requests from a launch block (control
    `local`) — `name` (`--cluster`) when the fleet offers several, the one when it offers one (a unique answer, not a
    guess). ValueError naming the choices for no name among several, or a name that is not such a cluster."""
    from vllm_lb import fleet as fleet_mod
    from vllm_lb.catalog import ModelCatalog, document
    # The catalog the fleet validates against — the library ASSEMBLED FOR THIS FLEET (its checkpoints, its targets),
    # its folders the node's own mount (EDGE_MODELS_DIR) — as the LB on this server reads it
    # (deploy/standalone/lb.override.yml: MODELS_SEARCH_DIRS=${EDGE_MODELS_DIR}).
    deployment = fleet_mod.catalog_deployment_of(repo / "lb" / "fleet.yml")
    doc = document(deployment) if catalog is None else catalog
    fleet = fleet_mod.load_fleet(repo / "lb" / "fleet.yml",
                                 catalog=ModelCatalog.from_dict(doc, roots=[server["EDGE_MODELS_DIR"]],
                                                                targets=deployment.targets))
    ours = [c.name for c in fleet.clusters.values() if c.launch is not None and fleet_mod.CONTROLS[c.control].drives]
    if name is None:
        if len(ours) != 1:
            raise ValueError(f"{fleet.source}: {len(ours)} clusters this host could apply ({', '.join(ours) or 'none'})"
                             f" — name one with --cluster")
        name = ours[0]
    elif name not in ours:
        raise ValueError(f"{fleet.source}: cluster {name!r} is not a node this host applies; those are "
                         f"{', '.join(ours) or 'none'}")
    return fleet, name


def state_dir_of(repo: Path, fleet, name: str) -> Path:
    """This cluster's node dir under the checkout's lb/state — the LB's own rule (node_control.state_path_for), so the
    intent the LB writes is the one read here."""
    # shortcut: the LB's switch file name is LB_CLUSTER_STATE's basename, /data/cluster.json by default; the applier
    # has always assumed lb/state is /data — upgrade when a deployment names the file otherwise.
    return Path(nc.node_dir(nc.state_path_for(str(repo / "lb" / "state" / "cluster.json"), name, list(fleet.clusters))))


def render(repo: Path, server: dict[str, str], launch: dict, cluster: str | None = None) -> list[str]:
    """The node's env lines for `launch`, by the fleet's own validator and renderer — ValueError, in its words, for an
    intent the fleet refuses or a line node.env could not carry safely."""
    from vllm_lb import fleet as fleet_mod
    fleet, name = cluster_of(repo, server, cluster)
    lines = fleet_mod.node_env(fleet, name, launch)
    unsafe = [ln for ln in lines if not safe_line(ln)]
    if unsafe:
        raise ValueError(f"node-env rendered a line node.env cannot carry safely: {unsafe[0]!r}")
    return lines


def in_flight(publish: str, fetch=None) -> tuple[int, str]:
    """(requests the node is serving or queueing, how that was read) — off its own /metrics. A node that does not
    answer serves nothing: 0, and the reading says why."""
    from vllm_lb.probe import parse_vllm_metrics
    url = f"http://{publish}/metrics"
    try:
        text = fetch(url) if fetch else urllib.request.urlopen(url, timeout=5).read().decode("utf-8", "replace")
    except Exception as exc:                       # refused, reset, timeout: nothing is being served there
        return 0, f"the node does not answer ({type(exc).__name__})"
    m = parse_vllm_metrics(text)
    running, waiting = m.get("running"), m.get("waiting")
    if running is None or waiting is None:
        return 0, "the node's /metrics names no request counts"
    return int(running) + int(waiting), f"running {int(running)}, waiting {int(waiting)}"


def docker_env(server: dict[str, str]) -> dict[str, str]:
    """The environment of this node's docker calls: the server file's EDGE_NODE_DOCKER_HOST (a node on another box's
    daemon, e.g. ssh://user@host) as DOCKER_HOST — named per call, never set on the process."""
    host = server.get("EDGE_NODE_DOCKER_HOST")
    return {**os.environ, "DOCKER_HOST": host} if host else dict(os.environ)


def docker(*args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=DOCKER_TIMEOUT_S, env=env)


def container_state(name: str, env: dict | None = None) -> dict | None:
    """The node container's docker state ({status, exit_code, restarts, started_at}), or None when it does not
    exist."""
    r = docker("inspect", "-f", "{{.State.Status}}|{{.State.ExitCode}}|{{.RestartCount}}|{{.State.StartedAt}}", name,
               env=env)
    if r.returncode != 0:
        return None
    status, code, restarts, started = (r.stdout.strip().split("|") + ["", "", "", ""])[:4]
    return {"status": status, "exit_code": int(code) if code.lstrip("-").isdigit() else None,
            "restarts": int(restarts) if restarts.isdigit() else None, "started_at": started or None}


def log_tail(name: str, env: dict | None = None) -> str | None:
    r = subprocess.run(["docker", "logs", "--tail", str(nc.LOG_TAIL_LINES), name], stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, text=True, timeout=DOCKER_TIMEOUT_S, env=env)
    return r.stdout[-nc.LOG_TAIL_CHARS:] if r.returncode == 0 else None


def apply(intent: dict, repo: Path, server_file: Path, server: dict[str, str], now: float, fetch=None,
          previous: dict | None = None, cluster: str | None = None, state_dir: Path | None = None) -> dict:
    """Apply ONE intent; how it went, as report.applied. `previous`: the last report's applied, kept in the report
    that says this intent is being applied. `cluster`: which cluster's (`cluster_of`), its node dir `state_dir`."""
    out = {k: intent[k] for k in ("id", "at", "state", "force", "launch")}
    out.update(done_at=None, exit=None, stderr="", refused="", restarts_at_apply=None)
    state_dir = (repo / "lb" / "state" / nc.NODE_DIR) if state_dir is None else state_dir
    try:
        lines = render(repo, server, intent["launch"], cluster)
    except ValueError as exc:
        out.update(refused=str(exc), done_at=time.time())
        return out
    env = docker_env(server)
    running = (container_state(server["EDGE_NODE_NAME"], env) or {}).get("status") in ("running", "restarting")
    if running and not intent["force"]:
        count, how = in_flight(server["EDGE_NODE_PUBLISH"], fetch)
        if count > 0:
            out.update(refused=f"{count} request(s) in flight on the node ({how}): not relaunched — send it again "
                               f"with force to cut them", done_at=time.time())
            return out
    tmp = state_dir / f"{nc.FRAGMENT}.tmp"
    tmp.write_text("".join(f"{ln}\n" for ln in lines))
    os.replace(tmp, state_dir / nc.FRAGMENT)
    target = "node" if intent["state"] == nc.UP else "node-stop"
    # SAID BEFORE IT RUNS: the command may pull an image for minutes, and the panel must read "applying", not a
    # heartbeat going stale.
    since = time.time()
    nc.write_report(str(state_dir), nc.report_doc(
        heartbeat=since, host=socket.gethostname(), applied=previous, container=None, log=None,
        applying={**{k: intent[k] for k in ("id", "at", "state", "force")}, "since": since}))
    try:
        # up.sh reads THIS cluster's node.env (its node dir: node_control.state_path_for), named — never a path it
        # would have to guess among several clusters.
        r = subprocess.run(["bash", str(repo / "deploy" / "standalone" / "up.sh"), "--env", str(server_file),
                            "--node-env", str(state_dir / nc.FRAGMENT), target],
                           capture_output=True, text=True, timeout=nc.UP_TIMEOUT_S, env=env)
        out.update(exit=r.returncode, stderr=(r.stderr or "")[-nc.STDERR_TAIL_CHARS:])
    except subprocess.TimeoutExpired as exc:
        tail = exc.stderr.decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        out.update(exit=None, stderr=f"up.sh {target} did not finish in {nc.UP_TIMEOUT_S} s: "
                                     f"{tail[-nc.STDERR_TAIL_CHARS:]}")
    # The container's RestartCount right after the command: a node that crash-loops after its intent reads `running`
    # between attempts, and only this count, grown, says so (cluster.plan_local).
    out["restarts_at_apply"] = (container_state(server["EDGE_NODE_NAME"], env) or {}).get("restarts")
    out["done_at"] = time.time()
    return out


def run_pass(repo: Path, server_file: Path, *, now: float | None = None, fetch=None,
             cluster: str | None = None) -> dict:
    """One pass for `cluster` (`--cluster`; the fleet's one local node when left out); the report it wrote."""
    now = time.time() if now is None else now
    server = read_server_file(server_file)
    fleet, cluster = cluster_of(repo, server, cluster)
    state_dir = state_dir_of(repo, fleet, cluster)
    previous = nc.read_report(str(state_dir), now)
    applied = previous["applied"]
    try:
        with open(state_dir / nc.INTENT, encoding="utf-8") as fh:
            intent = nc.parse_intent(json.load(fh))
    except FileNotFoundError:
        intent = None
    except ValueError as exc:
        intent = None
        applied = {"id": None, "at": None, "state": None, "force": None, "done_at": now, "exit": None,
                   "stderr": "", "refused": f"{nc.INTENT}: {exc}"}
    if intent is not None and not nc.same_intent(intent, applied):
        applied = apply(intent, repo, server_file, server, now, fetch, previous=applied, cluster=cluster,
                        state_dir=state_dir)
    name, env = server["EDGE_NODE_NAME"], docker_env(server)
    container = container_state(name, env)
    doc = nc.report_doc(heartbeat=time.time(), host=socket.gethostname(), applied=applied, container=container,
                        log=log_tail(name, env) if container is not None else None)
    nc.write_report(str(state_dir), doc)
    return doc


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python3 -m vllm_lb.node_applier", description=__doc__.split("\n\n")[0])
    p.add_argument("--repo", required=True, type=Path, help="the checkout deploy/standalone runs from")
    p.add_argument("--env", type=Path, default=None, help="the server file (default: deploy/standalone/.env)")
    p.add_argument("--cluster", default=None, help="the fleet cluster this host applies (required when the fleet "
                                                   "names several local nodes; worklog/parallel-clusters.md)")
    p.add_argument("--node-dir", action="store_true", help="print the cluster's node dir (where the LB writes its "
                                                           "intent) and exit: the installer's watch, never re-derived")
    a = p.parse_args(argv)
    server_file = a.env or a.repo / "deploy" / "standalone" / ".env"
    try:
        if a.node_dir:
            fleet, name = cluster_of(a.repo, read_server_file(server_file), a.cluster)
            print(state_dir_of(a.repo, fleet, name))
            return 0
        doc = run_pass(a.repo, server_file, cluster=a.cluster)
    except (OSError, ValueError, KeyError) as exc:
        print(f"node-applier: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    applied = doc["applied"] or {}
    how = applied.get("refused") or f"exit {applied.get('exit')}"
    print(f"node-applier: heartbeat; last applied #{applied.get('id')} ({applied.get('state')}): {how}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
