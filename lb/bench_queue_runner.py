#!/usr/bin/env python3
"""The runner half of the gateway's swarm button. Stdlib only.

It runs one of two ways, the same code either way:
  * SEALED, in its own container (lb/bench_runner, since 2026-10-11): no route out but an egress proxy
    that refuses Anthropic and every private range, a pinned Claude Code, a HOME with no credentials,
    and `--seal-check` proving all of that again before every launch. This is the way to run it.
  * as a host user unit (deploy/install-bench-queue-runner.sh), with the operator's own `claude` and
    HOME — what it was until 2026-10-11, and the reason the sealed way exists.

WHY IT EXISTS. The gateway is served by the `vllm-lb` container, which has no docker socket, no host
mount of `/` and no shell on its host. It cannot start a process there, and a button that only wrote
a file nobody reads would be a dead switch, not a control. So the split is:

    the page  ->  POST /lb/swarm/start
    the LB    ->  writes ONE small file, lb/state/benchq/start.request  (it starts nothing)
    THIS      ->  sees the request, decides whether a swarm may start, starts it, and says what it
                  did in lb/state/benchq/runner.json
    the LB    ->  reads that report back into /lb/status, so the page can say "requested", "no host
                  runner is listening", or exactly what the runner answered — never a guess.

WHAT THE BUTTON ASKS FOR, SINCE 2026-10-01. ONE deep-research session — `lb/bench_session.py` — for
one cap's worth of minutes, against the model ALREADY LOADED. Since 2026-10-05 the request may NAME
those minutes (`cap_min`, 5 to 240 — `bench_session.CAP_MIN_RANGE`, where both ends are argued from
measurements); a request that names none runs for this runner's `--cap-min`, which is still 15. The
operator's words:

    "I should be able to click run benchmark any time, but that only starts deep research swarm,
     nothing more, we assume benchmark data are collected all the way even on normal requests"
    "the multi settings benchmark should not be part of the button run on web nor mobile"

THE MATRIX IS NO LONGER REQUESTABLE. `worklog/bench/fullclaudevllm_queue.sh` — thirty runs, a restart
of the pair between each one, thirteen hours — is a terminal-only operation now:

    python3 lb/bench_queue_runner.py --start-matrix

which runs the same preflight the button used to (nothing else benchmarking, no park, the frozen
script's seal intact, a label left to take) and then launches it. A REQUEST asking for anything but
a session is refused with those words, so a `start.request` left behind by an older LB cannot start
a thirteen-hour matrix after a deploy.

WHAT IT REFUSES, AND WHY EACH ONE MATTERS.

  * A swarm already running — or, for `--start-matrix`, a swarm or matrix already running. Two at
    once would drive the same TP=2 pair through different work and spoil both measurements. This is
    the refusal the whole design is for, so it is checked from /proc rather than from a log file.
  * For the matrix only: `claude/hold` or `claude/pause` in place. Those are the gate the frozen
    bench script itself waits on, and `pause` in particular is the operator's own lever, placed
    before an LB deploy. A start that cleared someone else's park would be the runner overruling a
    person. A SWARM does not consult that gate: it is a quarter of an hour against what is already
    loaded, and the operator asked to be able to click it any time.
  * For the matrix only: `fullclaudevllm_bench.sh` not matching `fullclaudevllm_bench.sha256`. The
    queue verifies that seal before every run and exits 7 if it differs; failing here says why, in
    one line, instead of launching a queue that dies a second later. THIS RUNNER NEVER WRITES EITHER
    FILE, and nothing here writes anywhere under `worklog/bench/claude` at all.
  * For the matrix only: every label in `queue.txt` already in `done.txt`. The queue would print
    QUEUE EMPTY and exit. Clearing `done.txt` to "fix" that would destroy a night's bookkeeping, so
    it is refused with words that say what to do instead.
  * For the matrix only, since 2026-10-02: the LB's STORED `model_choice` is not the one `queue.txt`
    declares on its `# model: <id>` line. The frozen bench script restarts the pair with a body that
    carries no model_choice, so it runs on whatever is stored, and its label, runs.txt, done.txt and
    summary carry no model either: a matrix resumed on the wrong model would overwrite Flash-Next's
    `claude/<label>-*` with another model's minutes under the same names. The declaration is a `#`
    line because the queue's own parser (`grep -v '^#'`) skips those, so the frozen file reads the
    queue exactly as before. No line, or an LB that cannot say what is stored, is refused too.

HOW A LIVE RUN IS DETECTED, AND WHY NOT `pgrep`. `pgrep -f` and `ps | grep` match a pattern ANYWHERE
in a command line. Both therefore match the watchdog shells whose `bash -c '<script text>'` merely
MENTIONS these filenames, and `grep` matches the caller's own command — which is how `pkill`/`pgrep
-f` came to kill an operator's shell twice in one day. This reads `/proc/<pid>/cmdline`, splits it
into real arguments, and compares the BASENAME OF EACH ARGUMENT to the script names. A name buried
inside one long `-c` string is not an argument, so it cannot match; the real `/bin/bash
./fullclaudevllm_queue.sh` matches on its second argument, and `python3 …/bench_session.py` on its
second. Nothing is ever signalled: this runner starts processes and never stops them.

RUNNING IT

    python3 lb/bench_queue_runner.py      # the head is LB_CLUSTER_URL: the environment's, else lb/.env's

  and, to survive a reboot, the user unit rendered from deploy/:

    LB_CLUSTER_URL=http://<head>:8100 deploy/install-bench-queue-runner.sh
    sudo loginctl enable-linger "$(id -un)"  # so it runs with nobody logged in

  `--once` does a single tick and exits (what the tests use). `--dry-run` does everything except the
  launch, and reports `dry-run` instead of `started`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time
import urllib.error
import urllib.request

import bench_session

#: The harness's own scripts, by the exact filename each runs under, and OURS beside them.
QUEUE_SCRIPT = "fullclaudevllm_queue.sh"
BENCH_SCRIPT = "fullclaudevllm_bench.sh"
SESSION_SCRIPT = "bench_session.py"
SEAL = "fullclaudevllm_bench.sha256"
#: The matrix is the two frozen scripts; a swarm is one of ours. Both are "something benchmarking".
MATRIX_SCRIPTS = (QUEUE_SCRIPT, BENCH_SCRIPT)
SCRIPTS = MATRIX_SCRIPTS + (SESSION_SCRIPT,)
#: The single-session launcher this runner performs a request with, beside it in lb/.
SESSION_PY = pathlib.Path(__file__).resolve().parent / SESSION_SCRIPT
#: The matrix's own stdout, appended so a night's history is never truncated by a restart. The
#: operator's manual invocation writes the same file and their watchdog reads it.
QUEUE_LOG = "claude-queue.log"
#: A swarm's stdout, in the LB's state directory — never in the harness's.
SESSION_LOG = "session.log"
#: The matrix's gate, and the two files it is made of (fullclaudevllm_bench.sh, first lines).
HOLD, PAUSE = "hold", "pause"
QUEUE_TXT, DONE_TXT = "queue.txt", "done.txt"
#: The line in queue.txt that names the model its rows are for — `# model: qwen3.8-flash-next-hibrid48`.
#: A comment to the frozen queue script, which skips every `#` line; a declaration to this runner.
MODEL_LINE = "# model:"
#: What the LB leaves for us, and what we leave for the LB.
REQUEST, REPORT = "start.request", "runner.json"
#: 2: the report carries `matrix` and `session` where it used to carry one `queue`.
#: 3: and `box`, the box this runner was configured to name (`--box`), or None.
REPORT_VERSION = 3
#: How long the sealed runner's egress check may take before its silence counts as a failure. It makes a
#: dozen connections that each fail or answer within seconds (lb/bench_runner/egress_check.py).
SEAL_CHECK_TIMEOUT_S = 120
#: The one action the gateway asks for. Anything else is refused by name.
SESSION_ACTION = "session"
#: A command line can be long; this is more than enough to find an argument in.
_CMDLINE_BYTES = 128 * 1024
_LOG_MAX_BYTES = 2 * 1024 * 1024


# ── what is alive ──────────────────────────────────────────────────────────────


def argv_of(pid: str, proc_root: str) -> list[str]:
    with open(os.path.join(proc_root, pid, "cmdline"), "rb") as fh:
        raw = fh.read(_CMDLINE_BYTES)
    return [part.decode("utf-8", "replace") for part in raw.split(b"\0") if part]


def running_scripts(proc_root: str = "/proc", exclude: set[int] | None = None) -> list[dict]:
    """Every live process that IS one of the benchmark scripts, by ARGUMENT, never by substring.

    Each entry is `{pid, script, since, cmd}`; `since` is the /proc entry's mtime, which Linux sets
    when the process is created, or None where it could not be read."""
    skip = set(exclude or ())
    skip.add(os.getpid())
    skip.add(os.getppid())
    found: list[dict] = []
    try:
        entries = sorted(os.listdir(proc_root))
    except OSError:
        return found
    for pid in entries:
        if not pid.isdigit() or int(pid) in skip:
            continue
        try:
            argv = argv_of(pid, proc_root)
        except OSError:
            continue                          # the process ended between listing and reading
        name = next((s for a in argv for s in SCRIPTS if os.path.basename(a) == s), None)
        if name is None:
            continue
        try:
            since = os.stat(os.path.join(proc_root, pid)).st_mtime
        except OSError:
            since = None
        found.append({"pid": int(pid), "script": name, "since": since,
                      "cmd": " ".join(argv)[:200]})
    return found


def matrix_block(live: list[dict]) -> dict:
    """The `matrix` block of the report: whether the thirty-run queue is up on box-c. The QUEUE
    process is preferred over the bench run it spawned, because it is the thing that keeps going. A
    swarm is NOT the matrix and never fills this in."""
    runs = [p for p in live if p["script"] in MATRIX_SCRIPTS]
    best = next((p for p in runs if p["script"] == QUEUE_SCRIPT), None) or (runs[0] if runs else None)
    if best is None:
        return {"running": False, "pid": None, "since": None, "script": None}
    return {"running": True, "pid": best["pid"], "since": best["since"], "script": best["script"]}


def session_block(live: list[dict], state_dir: pathlib.Path) -> dict:
    """The `session` block: the /proc fact FIRST, then the session's own report folded into it.

    The process is the authority. A session killed outright leaves a report still saying `running`,
    and a swarm that has only just been launched has not written one yet — both would otherwise read
    as something they are not, so each gets its own phase instead."""
    proc = next((p for p in live if p["script"] == SESSION_SCRIPT), None)
    out: dict = {"alive": proc is not None,
                 "pid": proc["pid"] if proc else None,
                 "since": proc["since"] if proc else None,
                 "label": None, "phase": None, "cap_min": None, "claude_pid": None,
                 "started_at": None, "ended_at": None, "rc": None, "detail": "",
                 "guards": None, "at": None}
    out.update(dict.fromkeys(bench_session.IDENTITY_KEYS))
    report = read_json(state_dir / bench_session.REPORT)
    if isinstance(report, dict):
        for key in ("label", "phase", "cap_min", "claude_pid", "started_at", "ended_at", "rc",
                    "detail", "guards", "at", *bench_session.IDENTITY_KEYS):
            if key in report:
                out[key] = report[key]
        if proc is not None and report.get("pid") != proc["pid"]:
            # The live swarm is not the one this report describes: it is still coming up. Nothing of the
            # previous run is this one's — not its clock (the page drew "743 of 15 min" from it) and not
            # its cap: until the session reports, its clock is its own process's start.
            out.update({"label": None, "phase": "starting", "rc": None, "ended_at": None,
                        "guards": None, "claude_pid": None, "started_at": proc["since"], "cap_min": None,
                        "detail": "a swarm is starting; it has not reported yet",
                        **dict.fromkeys(bench_session.IDENTITY_KEYS)})
    elif proc is not None:
        out.update({"phase": "starting", "started_at": proc["since"],
                    "detail": "a swarm is starting; it has not reported yet"})
    if proc is None and out["phase"] in ("starting", "guards", "running"):
        out["phase"] = "lost"
        out["detail"] = ("the session stopped without reporting how it ended — its process is no "
                         "longer there")
    return out


# ── may it start ───────────────────────────────────────────────────────────────


def session_preflight(live: list[dict], box: str | None = None) -> tuple[str | None, str]:
    """`(refusal, detail)` for ONE swarm. `refusal` None means it may start.

    Two refusals, and they name which thing is up, because "already running" without a subject is
    the one answer an operator cannot act on. The park is deliberately NOT consulted: it gates the
    matrix between its runs, and a swarm is a quarter of an hour against what is already loaded,
    which the operator asked to be able to start at any time. `box` is where this runner runs, as its
    configuration names it (`--box`); with none, no box is named — never a literal one."""
    swarm = [p for p in live if p["script"] == SESSION_SCRIPT]
    if swarm:
        pids = ", ".join(f"pid {p['pid']}" for p in swarm[:3])
        return "already running", (f"a swarm is already up ({SESSION_SCRIPT}, {pids}) — nothing was "
                                  f"started; two at once would make both measurements meaningless")
    matrix = [p for p in live if p["script"] in MATRIX_SCRIPTS]
    if matrix:
        who = ", ".join(f"{p['script']} (pid {p['pid']})" for p in matrix[:3])
        return "matrix running", (f"the run matrix is up{f' on {box}' if box else ''}: {who} — it "
                                 f"drives the pair through its own configurations, so nothing was started")
    return None, "one deep-research swarm against the model already loaded"


def seal_check(command: str | None) -> tuple[bool, str]:
    """`(sealed, words)`: the sealed runner's own proof, run before EVERY launch, that the network it is
    on still cannot reach Anthropic (lb/bench_runner/egress_check.py). None = no check configured (the
    host unit), which is `(True, "")` — the host runner never had one.

    A check that cannot be run, times out, or exits non-zero is a refusal, never a pass: the whole point
    is that claude does not start unless the seal has just been PROVEN. Its last lines are the words."""
    if not command:
        return True, ""
    try:
        done = subprocess.run([command], stdin=subprocess.DEVNULL, capture_output=True, text=True,
                              timeout=SEAL_CHECK_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"the seal check {command} could not be run: {type(exc).__name__}: {exc}"
    said = " | ".join(line.strip() for line in (done.stdout + done.stderr).splitlines()
                      if line.strip())[-600:]
    if done.returncode != 0:
        return False, f"the seal check failed (exit {done.returncode}), so claude was not started: {said}"
    return True, said


def seal_matches(bench_dir: pathlib.Path) -> tuple[bool, str]:
    """Is `fullclaudevllm_bench.sh` byte-for-byte the committed one? Read-only, always: the queue
    checks this same seal before every run and halts when it differs, and changing either file
    needs the operator's explicit OK."""
    script, seal = bench_dir / BENCH_SCRIPT, bench_dir / SEAL
    try:
        want = seal.read_text().split()[0].strip()
        got = hashlib.sha256(script.read_bytes()).hexdigest()
    except (OSError, IndexError) as exc:
        return False, f"{type(exc).__name__} reading {BENCH_SCRIPT} and its seal: {exc}"
    if got != want:
        return False, (f"{BENCH_SCRIPT} does not match {SEAL} ({got[:12]}… vs {want[:12]}…) — the "
                       f"frozen script was changed; the queue would refuse it too")
    return True, ""


def next_label(queue_dir: pathlib.Path) -> tuple[str | None, str]:
    """The label the queue would take: the first line of queue.txt not in done.txt — its own rule."""
    try:
        matrix = [line.split()[0] for line in (queue_dir / QUEUE_TXT).read_text().splitlines()
                  if line.strip() and not line.startswith("#")]
    except OSError as exc:
        return None, f"{queue_dir / QUEUE_TXT} cannot be read: {exc}"
    try:
        done = {line.strip() for line in (queue_dir / DONE_TXT).read_text().splitlines() if line.strip()}
    except OSError:
        done = set()
    left = [label for label in matrix if label not in done]
    if not left:
        return None, (f"every one of the {len(matrix)} labels in {QUEUE_TXT} is already in "
                      f"{DONE_TXT} — add lines to {QUEUE_TXT}, or clear {DONE_TXT} at the keyboard, "
                      f"to run the matrix again")
    return left[0], f"{len(left)} of {len(matrix)} left"


def queue_model(queue_dir: pathlib.Path) -> str | None:
    """The model `queue.txt` declares its rows are for: the first `# model: <id>` line, or None.

    It is a `#` line so the frozen queue script's parser (`grep -v '^#'`) never sees it, and it is
    read here rather than inferred from the rows, which carry no model at all."""
    try:
        lines = (queue_dir / QUEUE_TXT).read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        if line.startswith(MODEL_LINE):
            value = line[len(MODEL_LINE):].strip()
            if value:
                return value.split()[0]
    return None


def stored_model(lb_url: str, timeout: float = 8.0) -> dict:
    """What the frozen script's restart would load, read from the LB with one GET and nothing else.

    `cluster.desired.model_choice` is the STORED choice: the bench script's restart body carries no
    model_choice, and the LB keeps the stored one for a body that names none. `loaded` is what the
    head serves right now (`cluster.loaded_model.spec`), for the message only. An LB that does not
    answer gives `model_choice: None` and an `error`, and the preflight refuses on that."""
    out = {"model_choice": None, "generation": None, "loaded": None, "state": None, "error": ""}
    try:
        with urllib.request.urlopen(f"{lb_url}/lb/status", timeout=timeout) as r:
            status = json.loads(r.read(4 * 1024 * 1024).decode("utf-8", "replace"))
    except (urllib.error.URLError, OSError, ValueError) as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out
    cluster = status.get("cluster") if isinstance(status, dict) else None
    if not isinstance(cluster, dict):
        out["error"] = "/lb/status carries no cluster block"
        return out
    desired = cluster.get("desired") if isinstance(cluster.get("desired"), dict) else {}
    loaded = cluster.get("loaded_model") if isinstance(cluster.get("loaded_model"), dict) else {}
    choice = desired.get("model_choice")
    out.update({"model_choice": str(choice) if choice else None, "generation": desired.get("generation"),
                "loaded": loaded.get("spec"), "state": cluster.get("state")})
    return out


def model_preflight(queue_dir: pathlib.Path, stored: dict) -> tuple[str | None, str]:
    """`(refusal, detail)` for the model: the matrix runs only on the model its queue declares.

    Three refusals, each naming what to do. The rows of `queue.txt` say nothing about a model, the
    frozen script restarts the pair on whatever is STORED, and a run's label, runs.txt line and
    summary say nothing either — so a matrix resumed on another model would file that model's
    minutes under Flash-Next's names. The declaration and the store have to agree before a single
    run is taken."""
    declared = queue_model(queue_dir)
    if declared is None:
        return "model", (f"{QUEUE_TXT} does not declare which model its rows are for — add a line "
                         f"`{MODEL_LINE} <id>` (a comment to the frozen queue, a declaration to this "
                         f"runner) naming the catalog model the matrix was written for")
    if stored.get("model_choice") is None:
        why = stored.get("error") or "the LB named no stored model_choice"
        return "model", (f"the LB could not say which model is stored ({why}) — the bench script "
                         f"restarts the pair on the stored model, so without that answer the matrix "
                         f"({declared}) cannot be known to run on its own model")
    if stored["model_choice"] != declared:
        return "model", (f"the stored model_choice is {stored['model_choice']} (generation "
                         f"{stored.get('generation')}) but {QUEUE_TXT} is {declared}'s matrix — the "
                         f"bench script would restart the pair on the stored model and file its runs "
                         f"under {declared}'s labels. Choose {declared} on the cluster panel first, or "
                         f"write a queue for {stored['model_choice']}")
    return None, f"on {declared}, which is the stored model_choice"


def matrix_preflight(bench_dir: pathlib.Path, queue_dir: pathlib.Path,
                     live: list[dict], stored: dict) -> tuple[str | None, str]:
    """The thirty-run matrix's own preflight, for `--start-matrix` at the keyboard. Order matters: a
    second run of anything is the worst outcome, a park is a person's decision, a broken seal is a
    queue that would die anyway, and the wrong model is a night filed under the wrong name.
    `stored` is `stored_model()`'s answer, fetched by the caller so this stays a pure decision."""
    if live:
        who = ", ".join(f"{p['script']} (pid {p['pid']})" for p in live[:3])
        return "already running", f"{who} — nothing was started; two at once would spoil both"
    held = [name for name in (HOLD, PAUSE) if (queue_dir / name).exists()]
    if held:
        return "parked", (f"the matrix is parked: {' and '.join('claude/' + h for h in held)} "
                          f"{'are' if len(held) > 1 else 'is'} in place. Release it first — a start "
                          f"that cleared someone else's park would be this runner overruling them")
    ok, why = seal_matches(bench_dir)
    if not ok:
        return "seal", why
    refusal, model_detail = model_preflight(queue_dir, stored)
    if refusal is not None:
        return refusal, model_detail
    label, detail = next_label(queue_dir)
    if label is None:
        return "finished", detail
    return None, f"the first run would be {label} ({detail}), {model_detail}"


# ── doing it ───────────────────────────────────────────────────────────────────


def _appending(path: pathlib.Path):
    """A log kept across restarts, rotated rather than allowed to grow without end."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if path.exists() and path.stat().st_size > _LOG_MAX_BYTES:
            os.replace(path, path.with_suffix(path.suffix + ".prev"))
    except OSError:
        pass
    return open(path, "a", encoding="utf-8")


def launch_session(opts, label: str, cap_min: float | None = None) -> int:
    """Start ONE deep-research session, in a session of its own so it outlives this runner.

    Everything it needs is on its command line, so what is running says what it is doing, and so
    this runner holds no copy of a default that could drift from `bench_session.py`'s own. `cap_min`
    is the duration THIS request asked for (`act`); None is `--cap-min` as this runner was started,
    which is how every request written before 2026-10-05 still runs for its fifteen minutes."""
    log = _appending(pathlib.Path(opts.state_dir) / SESSION_LOG)
    cap = opts.cap_min if cap_min is None else cap_min
    argv = [sys.executable or "python3", str(SESSION_PY),
            "--label", label, "--cap-min", f"{cap:g}",
            "--state-dir", str(opts.state_dir), "--lb-url", opts.lb_url,
            "--metrics-url", opts.metrics_url, "--model", opts.model]
    # The sealed runner's own paths and proxy (lb/bench_runner). Absent on the host unit, whose session
    # argv therefore stays exactly what it always was.
    for flag, value in (("--results-dir", opts.results_dir), ("--https-proxy", opts.https_proxy),
                        ("--no-proxy", opts.no_proxy)):
        if value:
            argv += [flag, str(value)]
    try:
        proc = subprocess.Popen(argv, cwd=str(SESSION_PY.parent), stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                close_fds=True)
    finally:
        log.close()
    return proc.pid


def launch_matrix(bench_dir: pathlib.Path) -> int:
    """Start the frozen queue script exactly as the operator does by hand, in a session of its own
    so it outlives this runner, with its stdout appended to the log their watchdog reads."""
    log = _appending(bench_dir / QUEUE_LOG)
    try:
        proc = subprocess.Popen(["bash", f"./{QUEUE_SCRIPT}"], cwd=str(bench_dir),
                                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True, close_fds=True)
    finally:
        log.close()
    return proc.pid


def act(req: dict, opts, live: list[dict]) -> dict:
    """Perform one request and return the `last` block: what was asked, and what actually happened."""
    answer = {"id": req.get("id"), "asked_at": req.get("at"), "at": time.time(),
              "action": req.get("action")}
    if req.get("action") != SESSION_ACTION:
        return {**answer, "result": "not mine",
                "detail": (f"this runner only performs {SESSION_ACTION!r} requests: one "
                           f"deep-research swarm against the model already loaded. The thirty-run "
                           f"matrix is started at the keyboard with `bench_queue_runner.py "
                           f"--start-matrix`, never from the page")}
    # HOW LONG THE REQUEST ASKED FOR, decided here and not taken on trust: a request is a file, and the
    # LB's range check (proxy.py `swarm_start`) did not run on the one left behind by an older build or
    # dropped in by hand. Refused by name rather than rounded to this runner's default, so that nobody
    # reads back a quarter of an hour as the two hours they asked for (`bench_session.cap_from_request`).
    cap, cap_detail = bench_session.cap_from_request(req.get("cap_min"), opts.cap_min)
    if cap is None:
        return {**answer, "result": "bad cap", "detail": cap_detail}
    answer["cap_min"] = cap
    refusal, detail = session_preflight(live, opts.box)
    if refusal is not None:
        return {**answer, "result": refusal, "detail": detail}
    sealed, seal_words = seal_check(opts.seal_check)
    if not sealed:
        return {**answer, "result": "unsealed", "detail": seal_words}
    if opts.dry_run:
        return {**answer, "result": "dry-run",
                "detail": f"would have started {detail}, capped at {cap:g} min ({cap_detail})"}
    label = time.strftime("S-%Y%m%d-%H%M%S")
    try:
        pid = launch_session(opts, label, cap)
    except OSError as exc:
        return {**answer, "result": "failed", "detail": f"{type(exc).__name__}: {exc}"}
    return {**answer, "result": "started", "label": label,
            "detail": (f"{SESSION_SCRIPT} is up as pid {pid} — {label}, one deep-research swarm "
                       f"capped at {cap:g} min against the model already loaded")}


def write_json(path: pathlib.Path, obj) -> None:
    """Whole, or not at all: the LB reads this file while we write it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(obj, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def read_json(path: pathlib.Path):
    try:
        return json.loads(path.read_text() or "null")
    except (OSError, ValueError):
        return None


def report(opts, live: list[dict], last: dict | None) -> dict:
    """The heartbeat. Written EVERY tick whether or not anything changed: its `at` is how the LB
    knows a runner is listening at all, and a request left for a runner that is not there must read
    as a request that cannot be served."""
    state_dir = pathlib.Path(opts.state_dir)
    out = {"version": REPORT_VERSION, "at": time.time(), "pid": os.getpid(), "box": opts.box,
           "started_at": opts.started_at, "cap_min": opts.cap_min,
           "bench_dir": str(opts.bench_dir), "queue_dir": str(opts.queue_dir),
           "session": session_block(live, state_dir), "matrix": matrix_block(live), "last": last}
    write_json(state_dir / REPORT, out)
    return out


def note(state_dir: pathlib.Path, line: str) -> None:
    path = state_dir / "runner.log"
    try:
        state_dir.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size > _LOG_MAX_BYTES:
            os.replace(path, path.with_suffix(".log.prev"))
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {line}\n")
    except OSError:
        pass
    print(line, flush=True)


def tick(opts, last: dict | None) -> dict | None:
    """One pass: look, act on a request we have not acted on, then report. Returns the new `last`."""
    state_dir = pathlib.Path(opts.state_dir)
    live = running_scripts(opts.proc_root)
    req = read_json(state_dir / REQUEST)
    if isinstance(req, dict) and req.get("id") and (last or {}).get("id") != req["id"]:
        last = act(req, opts, live)
        note(state_dir, f"request {req['id'][:8]}: {last['result']} — {last['detail']}")
        live = running_scripts(opts.proc_root)      # a launch changes what is alive
    report(opts, live, last)
    return last


def start_matrix(opts) -> int:
    """`--start-matrix`: the thirty-run matrix, from the keyboard, with the preflight that used to
    sit behind the button. The page cannot ask for this and neither can a request file."""
    live = running_scripts(opts.proc_root)
    refusal, detail = matrix_preflight(opts.bench_dir, opts.queue_dir, live, stored_model(opts.lb_url))
    if refusal is not None:
        print(f"NOT STARTED ({refusal}): {detail}", flush=True)
        return 2
    if opts.dry_run:
        print(f"dry run: would have started {QUEUE_SCRIPT} — {detail}", flush=True)
        return 0
    pid = launch_matrix(opts.bench_dir)
    note(pathlib.Path(opts.state_dir), f"--start-matrix: {QUEUE_SCRIPT} is up as pid {pid} — {detail}")
    return 0


def parse(argv=None):
    here = pathlib.Path(__file__).resolve().parent            # <repo>/lb
    repo = here.parent
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--bench-dir", default=str(repo / "worklog" / "bench"))
    p.add_argument("--queue-dir", default=str(repo / "worklog" / "bench" / "claude"))
    p.add_argument("--state-dir", default=str(here / "state" / "benchq"))
    p.add_argument("--proc-root", default="/proc")
    p.add_argument("--cap-min", type=float, default=bench_session.CAP_MIN,
                   help="minutes of load a swarm gets when its request names none; the page offers "
                        "this number as the default and may ask for another within CAP_MIN_RANGE")
    p.add_argument("--lb-url", default=bench_session.LB_URL)
    p.add_argument("--metrics-url", default=bench_session.default_metrics_url(),
                   help="the head engine's /metrics (default: LB_CLUSTER_URL + /metrics, from the environment or lb/.env)")
    p.add_argument("--model", default=bench_session.MODEL)
    p.add_argument("--tick", type=float, default=2.0, help="seconds between passes")
    p.add_argument("--once", action="store_true", help="one pass, then exit")
    p.add_argument("--dry-run", action="store_true", help="never launch; report `dry-run` instead")
    p.add_argument("--start-matrix", action="store_true",
                   help="start the thirty-run matrix here and now, then exit (not the button's job)")
    # The sealed runner's (lb/bench_runner). Every one is off unless given, so the host unit is unchanged.
    p.add_argument("--box", default=None,
                   help="the box this runner runs on, as the deployment's configuration names it; it is "
                        "reported so the page can say where — none = no box is named")
    p.add_argument("--results-dir", default=None,
                   help="where each session keeps its answers (passed to bench_session.py)")
    p.add_argument("--https-proxy", default=None,
                   help="the egress proxy each session's claude fetches through (passed to bench_session.py)")
    p.add_argument("--no-proxy", default=None,
                   help="the in-network hosts each session's claude dials directly (passed to bench_session.py)")
    p.add_argument("--seal-check", default=None,
                   help="an executable that must exit 0 before EVERY launch: the proof that this "
                        "runner's network still cannot reach Anthropic; anything else refuses the request")
    opts = p.parse_args(argv)
    if not opts.metrics_url:
        p.error("no head engine to read: LB_CLUSTER_URL is set neither in the environment nor in lb/.env "
                "(the head rank's vLLM URL, e.g. http://<head>:8100), and no --metrics-url was passed")
    opts.bench_dir = pathlib.Path(opts.bench_dir)
    opts.queue_dir = pathlib.Path(opts.queue_dir)
    opts.started_at = time.time()
    return opts


def main(argv=None) -> int:
    opts = parse(argv)
    state_dir = pathlib.Path(opts.state_dir)
    if opts.start_matrix:
        return start_matrix(opts)
    # A restarted runner must not re-run the request it already answered, so `last` comes back from
    # our own report rather than starting empty.
    previous = read_json(state_dir / REPORT)
    last = previous.get("last") if isinstance(previous, dict) else None
    note(state_dir, f"runner up: pid {os.getpid()}, watching {state_dir / REQUEST}"
                    f"{' (dry run)' if opts.dry_run else ''}")
    try:
        while True:
            last = tick(opts, last)
            if opts.once:
                return 0
            time.sleep(opts.tick)
    except KeyboardInterrupt:
        note(state_dir, "runner stopped")
        return 0


if __name__ == "__main__":
    sys.exit(main())
