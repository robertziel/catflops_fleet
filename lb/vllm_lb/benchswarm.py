"""THE SWARM CONTROL: ask the host for ONE deep-research session, and read back what came of it.

WHAT THE BUTTON MEANS NOW. Until 2026-10-01 it started `worklog/bench/fullclaudevllm_queue.sh`: a
thirty-run MATRIX that walks configurations, restarts the TP=2 pair between each one and takes
thirteen hours. The operator does not want that behind a button:

    "I should be able to click run benchmark any time, but that only starts deep research swarm,
     nothing more, we assume benchmark data are collected all the way even on normal requests"
    "benchmark run should not unload model, should use already loaded model, the multi settings
     benchmark should not be part of the button run on web nor mobile"

So one click asks for ONE `fullclaudevllm` deep-research session against the model ALREADY LOADED.
HOW LONG is the one thing a click may say (`cap_min`, since 2026-10-05): 15 min when it says nothing, and
anything in `CAP_MIN_RANGE` when it does. Nothing else about the run is this control's to choose — the
configuration under it is the cluster panel's, and that is the whole point of the operator's words below.
Nothing is restarted, no setting is changed, no model is loaded or unloaded, and the matrix is not
reachable from this page at all — it is started at the keyboard with
`lb/bench_queue_runner.py --start-matrix`, which is also why this module no longer reads
`claude/queue.txt`, writes no park file and is not mounted into the harness's directory. The LB
cannot touch the matrix's bookkeeping any more, by construction.

AND IT MEASURES NOTHING. `vllm_lb/perflog.py` already records every busy minute of real traffic,
which is what `/benchmarks` draws — including the minutes a swarm makes. A control that generates
load needs no recording path of its own.

THE SPLIT, AND WHY IT IS A REQUEST RATHER THAN A START. This container has no docker socket, no host
mount of `/` and no shell on its host. It cannot start a process there and must never say it did:

    the page    ->  POST /lb/swarm/start
    the LB      ->  writes ONE small file, <state>/start.request  (it starts nothing)
    the runner  ->  lb/bench_queue_runner.py sees it, decides, launches lb/bench_session.py — sealed in
                    its own container (lb/bench_runner) since 2026-10-11, or as a host unit
    the LB      ->  reads the runner's report back into /lb/status

so the page can say three different true things — requested, no runner is listening, or exactly
what the runner answered — instead of one hopeful one. Which BOX the runner is on is the runner's own
configuration (`--box`), carried in its report; this module never names one.

WHAT IT REFUSES HERE, BEFORE THE REQUEST IS EVEN WRITTEN. The host runner refuses the same things
again from `/proc`, which is the authority; these are so that a click (or a curl) is answered in
words rather than by a request that was never going to be served:

  * a swarm already running, or the matrix running — two at once would make both measurements
    meaningless, and the refusal says WHICH one is up;
  * nothing serving at all. The swarm never loads a model, so with none loaded there is nothing to
    put under load; loading one is the operator's own deliberate act on the cluster panel.

NOTHING IS CACHED. Every call reads two small files out of the LB's own state directory.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid

from vllm_lb.atomic import write_json

#: In the LB's OWN state directory: what the page asked the host for, and what the host says back.
REQUEST, RUNNER = "start.request", "runner.json"
#: The one action this gateway ever asks for. The runner refuses anything else — including the
#: `start` of the old matrix request, so a request file left over from before cannot start a matrix.
ACTION = "session"
#: Whose request it is, in the request itself.
GATEWAY_MARK = "vllm-lb/gateway"
#: The runner reports every couple of seconds. Past this it is not listening, and a request would
#: sit unread — which the page says, rather than showing a request as if it were a session.
RUNNER_STALE_S = 20.0
#: The cap the host runner uses when it does not say otherwise, so the button can name its minutes.
CAP_MIN = 15.0
#: AND THE DURATION A CLICK MAY NAME INSTEAD, since 2026-10-05 (the operator: keep 15 the default, allow
#: 30 or more). The same pair as `bench_session.CAP_MIN_RANGE`, which is where both numbers are argued
#: from measurements — the floor is the guards' own ~200 s of the session's clock, the ceiling is heat,
#: not disk (2026-10-02: the dense 27B at 16 seats hard-killed box-a in 7 min of this load). The two
#: copies are not a choice: `.dockerignore` keeps `bench_session.py` OUT of this image (it is the
#: runner's half, and runs in the runner's own container or host unit), so this module cannot import it at run
#: time — `from bench_session import CAP_MIN_RANGE` in here is a ModuleNotFoundError in the container.
#: tests/test_bench_swarm.py pins the two equal instead, so neither can drift unnoticed.
CAP_MIN_RANGE = (5.0, 240.0)
_MARK_BYTES = 4096
_REPORT_BYTES = 64 * 1024


class SwarmUnavailable(OSError):
    """There is nowhere to leave the request where the host runner would find it. Refused loudly
    rather than written hopefully into a directory nothing reads."""


class SwarmRefused(Exception):
    """A swarm may not start right now, and this says which of the two reasons it is."""


def check_cap(cap_min) -> float:
    """The minutes a click asked for, as a float — or ValueError with the words the 400 carries.

    Here as well as in `proxy.swarm_start` on purpose: the proxy's check is what gives the caller a 400
    naming the field and its range (the same shape /lb/cluster's knobs are refused with), and this one
    is so that NO path writes a request file this LB would not accept back. `bool` is excluded before
    the numeric test because `isinstance(True, int)` is true in Python, and a request carrying
    `cap_min: true` would otherwise be written as one minute."""
    lo, hi = CAP_MIN_RANGE
    if isinstance(cap_min, bool) or not isinstance(cap_min, (int, float)):
        raise ValueError(f"`cap_min` must be a number of minutes from {lo:g} to {hi:g}")
    if not lo <= float(cap_min) <= hi:
        raise ValueError(f"`cap_min` must be a number of minutes from {lo:g} to {hi:g}; "
                         f"{float(cap_min):g} is outside it")
    return float(cap_min)


def _read_json(path: str, limit: int):
    with open(path, "rb") as fh:
        return json.loads(fh.read(limit) or b"null")


def _on(box) -> str:
    """` on <box>` for a box the runner named, and nothing at all for one it did not."""
    return f" on {box}" if isinstance(box, str) and box else ""


class BenchSwarm:
    """The swarm request and the host runner's report, read on demand.

    `loaded` is how this object learns whether anything is serving: a callable returning
    `(True|False|None, words)` — None where this gateway genuinely cannot tell, which is never
    treated as "no". An empty `state_dir` is the feature switched off, and `view()` is then the same
    `{"configured": False}` the cluster reports when there is none, so nothing renders."""

    def __init__(self, state_dir: str, loaded=None) -> None:
        self.state_dir = state_dir or ""
        self._loaded = loaded
        self._lock = threading.Lock()

    @property
    def configured(self) -> bool:
        return bool(self.state_dir)

    # ── what the host runner says ──────────────────────────────────────────

    def _request(self) -> dict | None:
        try:
            req = _read_json(os.path.join(self.state_dir, REQUEST), _MARK_BYTES)
        except (OSError, ValueError):
            return None
        return req if isinstance(req, dict) and req.get("id") else None

    def _runner(self, now: float) -> dict:
        """What `lb/bench_queue_runner.py` last reported. `listening` is the whole point: a request
        left for a runner that is not there must read as a request that cannot be served, never as a
        session about to happen.

        `box` is where the runner runs, as ITS configuration names it (`--box`, in its report) — the
        page's only source for that word. With no report there is no box to name, and none is: a box
        written here would be one deployment's name in every deployment's page."""
        out: dict = {"listening": False, "at": None, "age_s": None, "cap_min": CAP_MIN, "box": None,
                     "session": None, "matrix": None, "last": None, "error": ""}
        path = os.path.join(self.state_dir, RUNNER)
        try:
            report = _read_json(path, _REPORT_BYTES)
        except FileNotFoundError:
            out["error"] = (f"no bench runner has ever reported at {path} — the swarm runner for this "
                            f"gateway is not running (lb/bench_runner, sealed; or the host unit from "
                            f"deploy/install-bench-queue-runner.sh)")
            return out
        except (OSError, ValueError) as exc:
            out["error"] = f"{path}: {type(exc).__name__}: {exc}"[:200]
            return out
        if not isinstance(report, dict):
            out["error"] = f"{path} is not a runner report"
            return out
        at = report.get("at")
        at = float(at) if isinstance(at, (int, float)) else None
        out["at"] = at
        out["age_s"] = round(now - at, 1) if at is not None else None
        out["listening"] = at is not None and (now - at) <= RUNNER_STALE_S
        for key in ("session", "matrix", "last"):
            value = report.get(key)
            out[key] = value if isinstance(value, dict) else None
        cap = report.get("cap_min")
        if isinstance(cap, (int, float)) and cap > 0:
            out["cap_min"] = float(cap)
        box = report.get("box")
        out["box"] = box if isinstance(box, str) and box.strip() else None
        if not out["listening"] and at is not None:
            out["error"] = (f"the bench runner{_on(out['box'])} last reported {out['age_s']:.0f}s ago — "
                            f"it is not listening for a request")
        return out

    def _model(self) -> tuple[bool | None, str]:
        if self._loaded is None:
            return None, "this gateway cannot tell what is loaded"
        try:
            loaded, detail = self._loaded()
        except Exception as exc:                    # a probe must never break /lb/status
            return None, f"what is loaded could not be read: {type(exc).__name__}: {exc}"[:200]
        return loaded, detail

    # ── the whole view /lb/status carries ──────────────────────────────────

    def view(self, now: float | None = None) -> dict:
        if not self.configured:
            return {"configured": False}
        now = time.time() if now is None else now
        request = self._request()
        runner = self._runner(now)
        last = runner.get("last") or {}
        loaded, model_detail = self._model()
        return {"configured": True, "state_dir": self.state_dir,
                # The cap A CLICK WOULD GET with no duration named — the host runner's own default —
                # and the range it will accept instead. The page offers only what this LB takes, the
                # way the cluster's knobs are offered from `cluster.limits` and not from the markup.
                "cap_min": runner["cap_min"], "cap_min_range": list(CAP_MIN_RANGE),
                "model_loaded": loaded, "model_detail": model_detail,
                "request": request, "runner": runner,
                # The page's one question about a click it has already made: has the runner answered
                # THIS request yet, or is it still only a request?
                "request_pending": bool(request) and last.get("id") != request.get("id")}

    # ── the one thing a click writes ───────────────────────────────────────

    def _running(self, runner: dict) -> tuple[str, str] | None:
        """`(what, words)` for the one thing that may be up already, or None. The runner decides
        this again from `/proc`; here it is so that a click is answered rather than queued."""
        session = runner.get("session") or {}
        if session.get("alive") is True:
            label = session.get("label") or "a session"
            return "swarm", (f"a swarm is already running ({label}"
                             f"{', pid %s' % session['pid'] if session.get('pid') else ''}) — two "
                             f"at once would make both measurements meaningless")
        matrix = runner.get("matrix") or {}
        if matrix.get("running") is True:
            return "matrix", (f"the run matrix is running{_on(runner.get('box'))} "
                              f"({matrix.get('script') or 'the queue'}"
                              f"{', pid %s' % matrix['pid'] if matrix.get('pid') else ''}) — it "
                              f"holds the pair for its own configurations")
        return None

    def request_session(self, cap_min: float | None = None) -> dict:
        """Leave a request for ONE deep-research session. This starts nothing: it writes one small
        file in the LB's own state directory and answers with the state that caused, in which
        `request_pending` is true and stays true until the runner names this request's id.

        A second click writes a second request rather than raising; the runner acts on whichever it
        reads and starts nothing while a session is live, so clicking twice cannot make two.

        `cap_min` is the minutes the click asked for, or None for the host runner's own default. None
        writes NO `cap_min` field at all, so a page that names no duration leaves byte for byte the
        request this control has always left, and the runner's 15 min still decide it."""
        if not self.configured:
            raise SwarmUnavailable(
                "no swarm state directory is configured on this gateway, so a request has nowhere "
                "to be left where the host runner would find it")
        cap = None if cap_min is None else check_cap(cap_min)
        now = time.time()
        runner = self._runner(now)
        busy = self._running(runner)
        if busy is not None:
            raise SwarmRefused(busy[1])
        loaded, detail = self._model()
        if loaded is False:
            raise SwarmRefused(f"no model is loaded: {detail}. A swarm puts the model that is "
                               f"already there under load and never loads one — start the pair on "
                               f"the cluster panel first")
        with self._lock:
            write_json(os.path.join(self.state_dir, REQUEST),
                       {"id": uuid.uuid4().hex, "at": now, "by": GATEWAY_MARK, "action": ACTION,
                        **({} if cap is None else {"cap_min": cap})})
        return self.view()
