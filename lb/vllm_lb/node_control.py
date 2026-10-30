"""A ONE-HOST NODE THE LB REQUESTS AND ITS HOST APPLIES (fleet control `local`) — the files between them.

The LB has no docker socket, no SSH key and no shell on the node's host, and must never pretend it started anything
(operator, 2026-10-11: no new credential). So the panel's Start, Stop and settings travel as ONE FILE, the pattern
the swarm button already uses (vllm_lb/benchswarm.py: start.request → runner.json):

    the page     ->  POST /lb/cluster
    the LB       ->  validates (fleet.validate_launch: the fleet's launch block's own validator) and writes ONE file,
                     <state>/node/intent.json — whole or not at all
    the applier  ->  vllm_lb/node_applier.py, a `systemd --user` oneshot on the node's host (a .path unit on the intent,
                     a .timer for its heartbeat; deploy/standalone/install-node-applier.sh): validates the SAME intent
                     with the SAME code, renders the node's env into <state>/node/node.env and runs
                     deploy/standalone/up.sh node (or node-stop), then reports into <state>/node/report.json
    the LB       ->  reads the report back into /lb/status; whether the node SERVES is its HTTP probe, as on box-c

THE INTENT is the node's desired state, persisted — it is what the LB reads back after a restart, as the pair's switch
is cluster.json — and nothing but:

    {"id": n, "at": epoch, "state": "up" | "down", "force": bool, "launch": {<fleet.MOVABLE key>: value, …}}

never an image, a GPU, a container name, a network or a port: `launch` holds only what fleet.validate_launch lets an
intent move, and the applier renders everything else from the fleet file. `force` is the operator's word that a
relaunch may cut what the node is serving; without it the applier refuses a relaunch while any request is in flight
(operator, 2026-10-11, after five were cut by a restart). An intent is applied ONCE, by its identity (id, at), so an
id restarted at 1 after an unreadable file is still new.

THE REPORT is the applier's: its heartbeat (every APPLIER_PERIOD_S; past APPLIER_STALE_S nobody is listening, which
the panel says rather than show a request as if it were being served), the last intent it applied and how (the
command's exit code and stderr tail, or the refusal in the validator's own words), the node container's state, and
its log tail — the LB reads the start's progress out of that tail with the pair's own reader (cluster_startup.py).

Standard library only: the applier imports this module with the host's python.
"""
from __future__ import annotations

import json
import math
import os
import threading
from typing import Any

from vllm_lb.atomic import retire_blueprint_field, write_json

#: The directory beside the LB's cluster switch (`Config.cluster_state_path`'s), lb/state/node on the host.
NODE_DIR = "node"
INTENT, REPORT, FRAGMENT = "intent.json", "report.json", "node.env"
#: The node's current START identity for its broker_parrot reservation (cluster_loop `_start`), written by the LB.
HOLD = "hold.json"
UP, DOWN = "up", "down"
STATES = (UP, DOWN)
INTENT_KEYS = ("id", "at", "state", "force", "launch")

#: The applier's heartbeat: its timer runs a pass every APPLIER_PERIOD_S (deploy/standalone/vllm-node-applier.timer
#: says the same number, pinned by deploy/tests). Past APPLIER_STALE_S — three missed beats — it is not listening.
APPLIER_PERIOD_S = 30
APPLIER_STALE_S = 3 * APPLIER_PERIOD_S
#: How long the applier lets `up.sh node` run — a first pull of a vLLM image is minutes. A pass says it is APPLYING
#: (report `applying`, with a heartbeat) before it starts the command, and is listening that long.
UP_TIMEOUT_S = 1800

#: How much of a failed command's stderr, and of the node's log, a report carries.
STDERR_TAIL_CHARS = 4000
LOG_TAIL_LINES = 400
LOG_TAIL_CHARS = 65536


def state_path_for(root: str, name: str, names) -> str:
    """WHERE cluster `name`'s switch lives (worklog/parallel-clusters.md) — the ONE rule the LB (`Config.for_cluster`)
    and the host applier (node_applier.py) share, so both read the same intent. `root` is LB_CLUSTER_STATE: the switch
    itself when the fleet names ONE cluster (live continuity: a live deployment's file stays where it is), else
    `<dir>/clusters/<name>/<file>`, its node dir beside it (`node_dir`). ValueError for a name not among `names`."""
    names = list(names)
    if name not in names:
        raise ValueError(f"no cluster {name!r}: the clusters are {', '.join(names) or 'none'}")
    if len(names) == 1:
        return root
    return os.path.join(os.path.dirname(root), "clusters", name, os.path.basename(root))


def node_dir(cluster_state_path: str) -> str:
    """The node's directory: beside the LB's cluster switch, in the LB's own state directory."""
    return os.path.join(os.path.dirname(cluster_state_path), NODE_DIR)


def _number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def parse_intent(raw: Any, where: str = INTENT) -> dict:
    """THE ONE READING of an intent, for the LB and the applier: exactly INTENT_KEYS, each in its domain — or
    ValueError naming what is wrong. The launch keys and their values are fleet.validate_launch's to judge."""
    if not isinstance(raw, dict):
        raise ValueError(f"{where} must be one JSON object, got {type(raw).__name__}")
    missing = [k for k in INTENT_KEYS if k not in raw]
    unknown = [str(k) for k in raw if k not in INTENT_KEYS]
    if missing or unknown:
        raise ValueError(f"{where} takes {', '.join(INTENT_KEYS)}"
                         + (f"; missing {', '.join(missing)}" if missing else "")
                         + (f"; unknown {', '.join(unknown)}" if unknown else ""))
    if not isinstance(raw["id"], int) or isinstance(raw["id"], bool) or raw["id"] < 1:
        raise ValueError(f"{where}: id must be a whole number from 1, got {raw['id']!r}")
    if not _number(raw["at"]):
        raise ValueError(f"{where}: at must be a time in epoch seconds, got {raw['at']!r}")
    if raw["state"] not in STATES:
        raise ValueError(f"{where}: state must be {' or '.join(STATES)}, got {raw['state']!r}")
    if not isinstance(raw["force"], bool):
        raise ValueError(f"{where}: force must be true or false, got {raw['force']!r}")
    launch = raw["launch"]
    if not isinstance(launch, dict):
        raise ValueError(f"{where}: launch must be an object of launch keys, got {type(launch).__name__}")
    bad = [k for k in launch if not isinstance(k, str) or not k.replace("_", "").isalnum()]
    if bad:
        raise ValueError(f"{where}: launch key {bad[0]!r} is no launch key")
    return {"id": raw["id"], "at": float(raw["at"]), "state": raw["state"], "force": raw["force"],
            "launch": dict(launch)}


def same_intent(intent: dict | None, applied: dict | None) -> bool:
    """Whether the applier's last applied intent IS `intent` — one identity, (id, at)."""
    return (intent is not None and applied is not None
            and (intent.get("id"), intent.get("at")) == (applied.get("id"), applied.get("at")))


class IntentStore:
    """One intent-shaped file, written whole and read back: the LB keeps TWO — the operator's WISH (the node's switch,
    at `Config.cluster_state_path`, as the pair's) and what it RELEASED to the host (<state>/node/intent.json, once
    the node has finished what it serves: cluster.plan_local). Missing: none was ever written — for the wish, the
    fleet's launch block (the boot default) stands."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.name = os.path.basename(path)
        self._lock = threading.Lock()

    def read(self) -> dict | None:
        """The stored intent, or None when none was ever written. ValueError for a file that cannot be read as one —
        said by the caller, never read as "no intent"."""
        try:
            with open(self.path, encoding="utf-8") as fh:
                text = fh.read()
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise ValueError(f"{self.name}: unreadable: {exc}") from None
        try:
            raw = json.loads(text)
        except ValueError as exc:
            raise ValueError(f"{self.name}: not JSON: {exc}") from None
        # A WISH OF THE BLUEPRINT ERA named the blueprint it came from: removed from the file once, said, and the wish
        # read as it is — same id, so nothing is released or applied again (atomic.retire_blueprint_field).
        return parse_intent(retire_blueprint_field(self.path, raw), self.name)

    def write(self, state: str, launch: dict, *, now: float, force: bool = False) -> dict:
        """A new intent — the next id — written whole; returned as stored. An unreadable file is replaced."""
        with self._lock:
            try:
                last = self.read()
            except ValueError:
                last = None
            doc = parse_intent({"id": (last["id"] + 1) if last else 1, "at": float(now), "state": state,
                                "force": bool(force), "launch": dict(launch)})
            write_json(self.path, doc)
            return doc


# ── the applier's report ───────────────────────────────────────────────────────────────────────


def report_doc(*, heartbeat: float, host: str, applied: dict | None, container: dict | None,
               log: str | None, applying: dict | None = None) -> dict:
    """The report, as the applier writes it — one shape, defined here for both sides.

    `applied`: the last intent applied — its id, at, state, force and launch, `done_at`, and how it went: `refused`
    (the validator's or the in-flight gate's words; nothing was run), else `exit` and `stderr` (the command's tail).
    `applying`: the intent whose command is running right now ({id, at, state, force, since}), or None.
    `container`: the node container's docker state ({status, exit_code, restarts}) or None when it does not exist.
    `log`: the node's log tail, or None."""
    return {"heartbeat": float(heartbeat), "host": host, "applied": applied, "applying": applying,
            "container": container, "log": log[-LOG_TAIL_CHARS:] if isinstance(log, str) else None}


def write_report(directory: str, doc: dict) -> None:
    write_json(os.path.join(directory, REPORT), doc)


def read_report(directory: str, now: float) -> dict:
    """The report as /lb/status carries it: whether an applier is LISTENING (a heartbeat within APPLIER_STALE_S — or,
    while it runs a command (`applying`), within that command's own limit), how old its heartbeat is, what it is
    applying and what it last applied and how, the node container's state and log tail — and `error`, the one line
    the panel shows when nobody is listening or the report cannot be read ("" when all is well)."""
    out = {"listening": False, "heartbeat_age_s": None, "host": None, "applied": None, "applying": None,
           "container": None, "log": None, "error": ""}
    path = os.path.join(directory, REPORT)
    try:
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
    except FileNotFoundError:
        out["error"] = ("no applier has reported: none is installed on the node's host "
                        "(deploy/standalone/install-node-applier.sh)")
        return out
    except (OSError, ValueError) as exc:
        out["error"] = f"{REPORT} unreadable: {exc}"
        return out
    if not isinstance(raw, dict) or not _number(raw.get("heartbeat")):
        out["error"] = f"{REPORT} unreadable: no heartbeat"
        return out
    age = round(float(now) - float(raw["heartbeat"]), 1)
    applied = raw.get("applied") if isinstance(raw.get("applied"), dict) else None
    applying = raw.get("applying") if isinstance(raw.get("applying"), dict) else None
    container = raw.get("container") if isinstance(raw.get("container"), dict) else None
    log = raw.get("log") if isinstance(raw.get("log"), str) else None
    limit = APPLIER_STALE_S + (UP_TIMEOUT_S if applying is not None else 0)
    out.update(listening=age <= limit, heartbeat_age_s=age,
               host=raw.get("host") if isinstance(raw.get("host"), str) else None,
               applied=applied, applying=applying, container=container, log=log)
    if not out["listening"]:
        out["error"] = f"no applier listening: its last heartbeat was {age:.0f} s ago"
    return out
