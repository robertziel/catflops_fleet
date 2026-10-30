#!/usr/bin/env python3
"""ONE deep-research session on the cluster AS IT IS. The host half of the Run benchmark button.

WHAT THIS IS FOR. The operator's words, 2026-10-01:

    "I should be able to click run benchmark any time, but that only starts deep research swarm,
     nothing more, we assume benchmark data are collected all the way even on normal requests"

So this starts a single `fullclaudevllm` deep-research session against WHATEVER CONFIGURATION IS
ALREADY LOADED, for one cap's worth of minutes, and then stops. It does not restart the pair, does
not change a setting, does not read `claude/queue.txt`, marks nothing done and banks no line. The
thirty-run matrix that does all of that is still `worklog/bench/fullclaudevllm_queue.sh`, started at
the keyboard (`lb/bench_queue_runner.py --start-matrix`), and it is no longer what the button means.

AND IT NEVER LOADS OR UNLOADS A MODEL (the operator, 2026-10-01: "benchmark run should not unload
model, should use already loaded model"). Nothing here POSTs to `/lb/cluster`, touches `enabled`,
`restart` or an engine knob, or waits for a generation. When no engine answers its own `/metrics`
there is no model to put under load, and the session refuses with exactly that (exit 2) rather than
starting one: loading is a deliberate act of the operator's on the cluster panel.

AND IT MEASURES NOTHING. `vllm_lb/perflog.py` already records every busy minute of real traffic into
`lb/state/perf/perf-*.jsonl`, which is what `/benchmarks` draws. This script's whole job is to
GENERATE LOAD; a second recording path would be a second truth to keep honest.

THE FOUR GUARDS, AND WHY THEY ARE NOT NEGOTIABLE. On 2026-09-30 an edit to the benchmark script left
the backend variables on a different command from `claude`, and a session ran for fifteen minutes on
the OPERATOR'S OWN Claude subscription (claude-opus-5, `apiKeySource: none`) instead of the local
cluster. Nothing failed. The money was simply spent. `worklog/bench/fullclaudevllm_bench.sh` is
frozen and sha256-pinned because of it, and its four guards are reproduced here exactly:

  0. BEFORE ANYTHING LAUNCHES (`check_argv`), the command about to run is checked structurally:
     `env -i` first, the assignments on that same command, `ANTHROPIC_BASE_URL` and
     `ANTHROPIC_API_KEY` among them, and NOTHING about the backend in the environment this process
     hands down — an inherited variable must never be there to make a wrong argv work by accident.
  A. the launched `claude`'s OWN `/proc/<pid>/environ` names the LB and the dummy key (exit 4);
  B. the session's first `init` event names the local model and `apiKeySource: ANTHROPIC_API_KEY`
     — this is how the accident actually showed itself in the log (exit 5);
  C. the engine's own `/metrics` shows prompt tokens MOVE, i.e. the traffic really landed on the
     cluster and not somewhere that merely answered (exit 6).

Any of them failing kills the session within seconds and exits non-zero, measuring nothing.

WHAT IT LEAVES BEHIND. `<state>/session.json`, rewritten as the session moves from `starting` to
`guards` to `running` to `finished` or `failed`, which is what `lb/bench_queue_runner.py` reports to
the gateway and what the page draws: elapsed against the cap while it runs, and the exit code when
it ends. `timeout`'s 124 is the NORMAL end — the session is supposed to be stopped by the clock. An
exit 0 means `claude` stopped on its own before the cap, which is a defect to show, not a tick.
The stream, the stderr and the metrics either side go to `<state>/sessions/<label>/`; the session's
own answers go to the results directory, wiped at the start of every run, because the model answers
in chat and does not reliably call Write (proven on 2026-09-30: two workflow notifications, zero
Write calls, an empty results directory).

RUNNING IT. The gateway's button is the normal way: the LB leaves a request, and the host runner
performs it. By hand, with nothing else benchmarking:

    LB_CLUSTER_URL=http://<head>:8100 python3 lb/bench_session.py            # 15 minutes, the live LB
    LB_CLUSTER_URL=http://<head>:8100 python3 lb/bench_session.py --cap-min 5
    LB_CLUSTER_URL=http://<head>:8100 python3 lb/bench_session.py --cap-min 30

Stdlib only. From `vllm_lb` it imports ONE module, `vllm_lb.engine_metrics` (itself stdlib only, no
config or state): the engines' metric vocabularies, so "is a model loaded" and "did prompt tokens move"
read every engine exactly as the LB's probe does. It runs in the sealed runner image (bench_runner/),
which COPYs that module beside it, never in the LB's image.
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import pathlib
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

from vllm_lb import engine_metrics

#: The ONE client address of the cluster (never through the LB's own health, the guard reads the engine).
LB_URL = "http://127.0.0.1:8010"


#: The LB's own settings (gitignored), where the operator's LB_CLUSTER_URL lives. The host runner is a
#: systemd unit with no LB_CLUSTER_URL of its own, so it reads the SAME value the LB reads: the head it
#: measures is the head the LB serves from. One definition, two readers — never a second address.
LB_ENV_FILE = pathlib.Path(__file__).resolve().parent / ".env"


def _env_file_value(path: pathlib.Path, name: str) -> str:
    """`name`'s value in a compose-style .env file, by compose's own rules: blank and `#` lines skipped,
    an `export ` prefix and surrounding quotes dropped, the LAST definition wins. "" when the file or the
    name is absent. Only `name` is read out of the file; nothing else in it is kept."""
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return ""
    value = ""
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, raw = line.removeprefix("export ").partition("=")
        if not sep or key.strip() != name:
            continue
        raw = raw.strip()
        if raw[:1] in ("'", '"') and raw.endswith(raw[:1]) and len(raw) >= 2:
            value = raw[1:-1]
        else:
            value = raw.split(" #", 1)[0].strip()
    return value


def default_metrics_url(environ=None) -> str | None:
    """The head engine's own /metrics, derived from LB_CLUSTER_URL (the head rank's vLLM API on its own
    port) — from the environment, else from the LB's own lb/.env (compose's precedence). Never through
    the LB: guard C has to see the ENGINE's counter, not a proxy's idea of it. None when neither names
    it: the session refuses to guess which host is the head."""
    env = os.environ if environ is None else environ
    base = (env.get("LB_CLUSTER_URL", "").strip()
            or _env_file_value(LB_ENV_FILE, "LB_CLUSTER_URL").strip()).rstrip("/")
    return f"{base}/metrics" if base else None


def operator_name() -> str:
    """The account the session runs as: whoever started it (`claude` gets this as its USER)."""
    return getpass.getuser()
#: The name `claude` is told to ask for: the LB's own alias, which it routes to whatever the pair
#: serves. It is NOT the model — what is actually loaded is read from the LB at the start of every
#: session (`served_identity`) and written into the report, never assumed from this name.
MODEL = "qwen3.8:latest"
#: The context window the session is given when the LB's `/v1/models` names none: Flash-Next's
#: 262144, the frozen script's figure. With an answer, the served model's own window is used.
DEFAULT_WINDOW = 262144
#: Fifteen minutes is the frozen script's cap and the unit `/benchmarks` draws its minutes in. It is a
#: DEFAULT and not a limit: the frozen script takes the cap as its fifth argument as well
#: (`CAP_MIN="${5:-15}"`, worklog/bench/fullclaudevllm_bench.sh:27), and nothing downstream of here is
#: sized in quarter-hours — `timeout_arg` below writes any duration, and `vllm_lb/perflog.py` records one
#: window per busy MINUTE whatever the cap (MAX_WINDOWS = 20160 of them, fourteen days).
CAP_MIN = 15.0
#: WHAT A CLICK MAY ASK FOR INSTEAD, in minutes (the operator, 2026-10-05: keep 15 the default, allow 30
#: or more). Both ends are measured rather than chosen:
#:
#:   5   — the guards own up to ~200 s of the session's OWN clock before it is `running`, and `timeout`
#:         is already counting through all of it: 20 s to find claude's pid (`max(10, poll * 20)`), then
#:         `--init-timeout` 60 s for guard B and `--traffic-timeout` 120 s for guard C. Under ~3.5 min
#:         the cap can SIGINT the session while guard C is still waiting for the engine's counter to
#:         move, and the report would then say "ended early" about a cap that was simply too short. 5 is
#:         also the smallest cap this file's own usage note offers.
#:   240 — a full-cap session writes 2.4–2.9 MB of stream.jsonl per 902 s, i.e. 9–11 MB/h, measured over
#:         the four captures in lb/state/benchq/sessions from 2026-10-03 and 2026-10-05; a capture is
#:         never pruned, so four hours is ~44 MB per click against the 2.9 MB the largest one there has
#:         ever held. Disk is therefore not the binding end — HEAT is: on 2026-10-02 the dense 27B at 16
#:         seats hard-killed box-a in 7 min of exactly this load, and a cap is nothing but time under it.
#:         So this stays a number a person types, and past four hours the thirty-run matrix (13 h,
#:         `bench_queue_runner.py --start-matrix`) is the right tool — which is why it is not on a button.
CAP_MIN_RANGE = (5.0, 240.0)
#: What the page reads, through the host runner's report; and where a run's captures go.
REPORT = "session.json"
#: 2: the report carries the served model, its catalog choice, the generation and the window.
REPORT_VERSION = 2
CAPTURES = "sessions"
#: The four facts about what was under load, read from the LB at the start (`served_identity`). The
#: host runner copies exactly these into its own report, so the page says which model a run was on.
IDENTITY_KEYS = ("served_model", "model_choice", "generation", "max_model_len")
#: The exits. 4/5/6 are the frozen script's own guard codes, kept the same so one number means one
#: thing across both paths; 124 (or 130) from `timeout` is the cap and the success.
GUARD_ENVIRON, GUARD_INIT, GUARD_TRAFFIC = 4, 5, 6
NO_MODEL, NO_CLAUDE, ENDED_EARLY, REFUSED = 2, 3, 1, 8
CAP_RCS = (124, 130, -signal.SIGINT)
#: The two assignments guard A checks for, and the shape of the key: a dummy, because the LB does
#: not authenticate and a real key must never be within reach of this process.
DUMMY = "dummy"
REQUIRED = ("ANTHROPIC_BASE_URL", "ANTHROPIC_API_KEY")
#: Nothing about the backend may be in the environment handed to the child. `env -i` discards it
#: anyway; this is the belt to that braces, and `check_argv` refuses rather than hoping.
FORBIDDEN_PREFIXES = ("ANTHROPIC", "CLAUDE_CODE", "API_TIMEOUT", "API_FORCE")
#: The bench's deep-research workload prompt (frozen):
#: two researches at once, thirty agents each, relaunched the moment one finishes, never idle.
PROMPT = """/deep-research Run TWO deep researches CONCURRENTLY, both at the same time, NOT one after
  the other: (1) grid-scale energy storage, (2) the history of the printing press.
  Dispatch 30 agents running in parallel for EACH of them, so 60 agents are working at once.
  Save each finished report under {results}, numbering each round. The MOMENT a research finishes,
  immediately start another round on that same topic with 30 fresh agents and save it too — keep
  both tracks running continuously, never idle, until you are stopped. Do not pause to summarise,
  do not ask permission, do not stop early, and write nothing outside {results}."""


class ArgvRefused(Exception):
    """The command about to be launched is not the guarded shape, so nothing is launched. This is
    the 2026-09-30 defect caught before it can cost anything."""


# ── the command, and the two places a variable must and must not be ────────────


def session_vars(lb_url: str, model: str, home: str, user: str, path: str,
                 window: int = DEFAULT_WINDOW, https_proxy: str | None = None,
                 no_proxy: str | None = None) -> list[str]:
    """The environment `claude` is given, as ASSIGNMENTS — the frozen script's list, in its order.

    Every one of these is an argument of the command being launched. None of them is exported,
    because an export has a parent, and a parent is a place for the wrong value to live. `window`
    is the served model's context length; the auto-compact threshold is 80 % of it, as the frozen
    script's 209715 is of 262144.

    THE PROXY, AND ONLY THE PROXY, WAS ADDED HERE (operator, 2026-10-11). The sealed runner
    (lb/bench_runner) sits on a docker network with no route out; the one way to a public page is its
    egress proxy, which refuses Anthropic's domains and published ranges and every private range.
    `claude`'s WebFetch fetches from THIS process, so without HTTPS_PROXY on this command every fetch
    of a deep research would fail. NO_PROXY names the in-network hosts (the LB, the engine) that are
    dialled directly. Both are absent unless asked for: the host runner's command is unchanged."""
    out = [f"HOME={home}", f"USER={user}", f"PATH={path}", "TERM=dumb", "LANG=C.UTF-8",
           f"ANTHROPIC_BASE_URL={lb_url}", f"ANTHROPIC_API_KEY={DUMMY}",
           f"ANTHROPIC_AUTH_TOKEN={DUMMY}", f"ANTHROPIC_MODEL={model}",
           f"CLAUDE_CODE_MAX_CONTEXT_TOKENS={int(window)}",
           f"CLAUDE_CODE_AUTO_COMPACT_WINDOW={int(window * 0.8)}",
           "CLAUDE_CODE_MAX_OUTPUT_TOKENS=16384",
           "API_FORCE_IDLE_TIMEOUT=0", "API_TIMEOUT_MS=3600000",
           "CLAUDE_STREAM_FIRST_BYTE_TIMEOUT_MS=3600000",
           "CLAUDE_CODE_ATTRIBUTION_HEADER=0", "CLAUDE_CODE_ENABLE_TASKS=1",
           "CLAUDE_CODE_WORKFLOW_MAX_CONCURRENT_AGENTS=50"]
    if https_proxy:
        out.append(f"HTTPS_PROXY={https_proxy}")
    if no_proxy:
        out.append(f"NO_PROXY={no_proxy}")
    return out


def claude_argv() -> list[str]:
    """`claude` and its flags. `-p` with stream-json in and out: the input format is what keeps the
    session alive past its first reply, so the background workflow is not stopped with it."""
    return ["claude", "-p", "--verbose", "--input-format", "stream-json",
            "--output-format", "stream-json", "--permission-mode", "bypassPermissions"]


def timeout_arg(cap_min: float) -> str:
    """The cap as `timeout` takes it: whole minutes read as minutes in `ps`, anything else exact."""
    seconds = cap_min * 60.0
    if abs(cap_min - round(cap_min)) < 1e-9:
        return f"{int(round(cap_min))}m"
    return f"{seconds:.0f}s" if abs(seconds - round(seconds)) < 1e-9 else f"{seconds:g}s"


def cap_from_request(asked, default: float) -> tuple[float | None, str]:
    """The cap a gateway request named, checked AGAIN here — `(minutes, words)`, or `(None, why)`.

    `proxy.swarm_start` range-checks `cap_min` before the request is written, but a request is a FILE:
    one left behind by another build, or dropped in by hand, reaches the host runner having been checked
    by nobody — which is exactly why `bench_queue_runner.py` already re-decides from /proc what the LB
    thinks it knows. Nothing asked for is the runner's own default, so a request written by an LB that
    never heard of this field behaves precisely as it did before (15 min, byte for byte the same file).

    An out-of-range number is REFUSED rather than rounded down to the default: a swarm that ran a
    quarter of an hour when two hours were asked for is a measurement whose minutes nobody can read
    back. The range is the REQUEST's; `--cap-min` itself is left unbounded, because a hand-run session
    and the suite's own stubs legitimately ask for seconds (tests/test_bench_session.py, cap_min=0.06)."""
    if asked is None:
        return float(default), f"the runner's own default of {default:g} min"
    lo, hi = CAP_MIN_RANGE
    if isinstance(asked, bool) or not isinstance(asked, (int, float)) or not lo <= float(asked) <= hi:
        return None, (f"the request asked for a cap of {asked!r}, which is not a number of minutes "
                      f"between {lo:g} and {hi:g}")
    return float(asked), f"the {float(asked):g} min the request asked for"


def session_argv(cap_min: float, lb_url: str, model: str, home: str, user: str,
                 path: str, window: int = DEFAULT_WINDOW, https_proxy: str | None = None,
                 no_proxy: str | None = None) -> list[str]:
    """`env -i <assignments> timeout -s INT -k 30 <cap> claude …` — one command, nothing inherited.

    `timeout` sits between the assignments and `claude` on purpose: it is what ENDS the session, and
    it passes the environment `env` built straight through to the process it starts. The session is
    meant to be ended by the clock, so there is no early-exit heuristic anywhere in this file."""
    return (["env", "-i", *session_vars(lb_url, model, home, user, path, window,
                                        https_proxy=https_proxy, no_proxy=no_proxy),
             "timeout", "-s", "INT", "-k", "30", timeout_arg(cap_min)] + claude_argv())


def launch_env(path: str) -> dict[str, str]:
    """The environment this process hands DOWN: `PATH`, so `env` and `claude` can be found, and
    nothing else. `env -i` would discard the rest; not having it is better than discarding it."""
    return {"PATH": path}


def check_argv(argv: list[str], env: dict[str, str], lb_url: str) -> None:
    """Refuse to launch anything that is not the guarded shape. Guard 0, and the cheapest of them.

    Checked: `env -i` leads (without `-i`, an inherited `ANTHROPIC_BASE_URL` would beat anything on
    the command line); the assignments are the run of arguments right after it, i.e. the ones `env`
    actually reads; `ANTHROPIC_BASE_URL` points at the LB we were told to use and the key is the
    dummy; no assignment sits behind the command, where `env` would hand it to nobody; and the
    environment we pass down names nothing about the backend."""
    if argv[:2] != ["env", "-i"]:
        raise ArgvRefused(f"the session must be launched through `env -i`, so that nothing of this "
                          f"process's environment is inherited; got {argv[:2]}")
    command_at = next((i for i, a in enumerate(argv[2:], 2) if "=" not in a), len(argv))
    assignments = dict(a.split("=", 1) for a in argv[2:command_at])
    want = {"ANTHROPIC_BASE_URL": lb_url, "ANTHROPIC_API_KEY": DUMMY}
    for name, value in want.items():
        if name not in assignments:
            raise ArgvRefused(f"{name} is not on the command `env -i` reads, so the session would "
                              f"not be on the local cluster — this is the 2026-09-30 defect")
        if assignments[name] != value:
            raise ArgvRefused(f"{name} on the command is not {value!r}")
    stray = [a for a in argv[command_at:] if "=" in a and a.split("=", 1)[0].isupper()]
    if stray:
        raise ArgvRefused(f"an assignment sits behind the command, where `env` never reads it: "
                          f"{[s.split('=', 1)[0] for s in stray]}")
    leaked = sorted(k for k in env if k.startswith(FORBIDDEN_PREFIXES))
    if leaked:
        raise ArgvRefused(f"the launched process would have inherited {leaked} from this one; the "
                          f"backend variables belong on the command line and nowhere else")
    if "claude" not in argv[command_at:]:
        raise ArgvRefused("the command being launched is not claude")


# ── reading /proc, because what we asked for is not evidence ───────────────────


def _argv_of(pid: int, proc_root: str) -> list[str]:
    with open(os.path.join(proc_root, str(pid), "cmdline"), "rb") as fh:
        raw = fh.read(128 * 1024)
    return [part.decode("utf-8", "replace") for part in raw.split(b"\0") if part]


def _ppid_of(pid: int, proc_root: str) -> int | None:
    try:
        with open(os.path.join(proc_root, str(pid), "status"), "r", errors="replace") as fh:
            for line in fh:
                if line.startswith("PPid:"):
                    return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return None


def is_claude(argv: list[str]) -> bool:
    """Is this command line `claude`? By the BASENAME OF AN ARGUMENT, never by a substring.

    `pgrep -f` and `ps | grep` match a pattern anywhere in a command line, which is how they came to
    match a watchdog shell and an operator's own shell on the same day. An argument cannot be
    mistaken that way, and taking ANY argument rather than only argv[0] covers the launchers the CLI
    is packaged behind: `claude -p …` as the frozen script's own `pgrep` pattern has it, and
    `node /usr/local/bin/claude …` or a shebang's `python3 …/claude` just as well."""
    return any(os.path.basename(a) == "claude" for a in argv)


def claude_pid(parent: int, proc_root: str = "/proc") -> int | None:
    """The `claude` that `timeout` started: the child of the process we hold, by name as well, so a
    `sleep` or a helper in the same session is never mistaken for the session itself."""
    try:
        entries = sorted(os.listdir(proc_root))
    except OSError:
        return None
    for name in entries:
        if not name.isdigit():
            continue
        pid = int(name)
        if _ppid_of(pid, proc_root) != parent:
            continue
        try:
            argv = _argv_of(pid, proc_root)
        except OSError:
            continue
        if is_claude(argv):
            return pid
    return None


def environ_of(pid: int, proc_root: str = "/proc") -> dict[str, str]:
    with open(os.path.join(proc_root, str(pid), "environ"), "rb") as fh:
        raw = fh.read(256 * 1024)
    out = {}
    for part in raw.split(b"\0"):
        if b"=" in part:
            key, value = part.decode("utf-8", "replace").split("=", 1)
            out[key] = value
    return out


def guard_environ(pid: int, lb_url: str, proc_root: str = "/proc") -> tuple[bool, str]:
    """GUARD A. What the KERNEL says the process was given — the one check that does not take this
    script's word for it. No value is ever echoed: whatever the wrong key was, it stays unprinted."""
    try:
        argv = _argv_of(pid, proc_root)
        env = environ_of(pid, proc_root)
    except OSError as exc:
        return False, f"pid {pid}'s own environment could not be read: {type(exc).__name__}: {exc}"
    if not is_claude(argv):
        return False, f"pid {pid} is not claude ({' '.join(argv[:2])!r})"
    wrong = []
    for name, value in (("ANTHROPIC_BASE_URL", lb_url), ("ANTHROPIC_API_KEY", DUMMY)):
        if name not in env:
            wrong.append(f"{name} is not set")
        elif env[name] != value:
            wrong.append(f"{name} is set to something else than {value}")
    if wrong:
        return False, (f"claude pid {pid} was not launched against the local cluster: "
                       + "; ".join(wrong))
    return True, f"claude pid {pid} is on {lb_url} with the dummy key"


# ── the session's own stream, and the engine's own counter ─────────────────────


def init_event(stream_path: str) -> dict | None:
    """The session's first `init` event, which is where the 2026-09-30 accident was visible all
    along: it named `claude-opus-5` and `apiKeySource: none`."""
    try:
        with open(stream_path, "r", errors="replace") as fh:
            for line in fh:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if isinstance(event, dict) and event.get("subtype") == "init":
                    return event
    except OSError:
        return None
    return None


def guard_init(stream_path: str, model: str, deadline: float, poll: float) -> tuple[bool, str]:
    """GUARD B. The session says for itself which model it is talking to and where the credential
    came from. `ANTHROPIC_API_KEY` is the only acceptable source: `none` means a subscription."""
    while True:
        event = init_event(stream_path)
        if event is not None:
            got_model, source = event.get("model"), event.get("apiKeySource")
            if got_model == model and source == "ANTHROPIC_API_KEY":
                return True, f"the session reports model {got_model} via {source}"
            return False, (f"the session reports model {got_model!r} with "
                           f"apiKeySource {source!r} — it is NOT on {model} via ANTHROPIC_API_KEY")
        if time.time() >= deadline:
            return False, "the session never wrote an init event, so what it is talking to is unknown"
        time.sleep(poll)


def engine_loaded(text: str | None) -> bool:
    """A CHAT model is resident: the engine's own /metrics speaks one of the chat engines' vocabularies
    (vllm_lb/engine_metrics.py — the one definition the LB's probe reads too). A text in no vocabulary,
    or the transcription service's, is nothing a deep-research session can talk to. Two private lists of
    prefixes here refused healthy pairs twice: SGLang (2026-10-03) and TensorFold (2026-10-29)."""
    return text is not None and engine_metrics.engine_of(text) in engine_metrics.CHAT_ENGINES


def prompt_tokens(text: str) -> float | None:
    """The prompt tokens the engine has processed — the canonical `prompt_tokens_total` its own parser
    reads (vLLM's `vllm:prompt_tokens_total`, SGLang's billed prefill over every mode, TensorFold's
    unlabelled counter). None when the scrape carries no such counter: guard C then has no baseline."""
    return engine_metrics.parse_metrics(text).get("prompt_tokens_total")


def fetch(url: str, timeout: float = 8.0) -> str | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read(4 * 1024 * 1024).decode("utf-8", "replace")
    except (urllib.error.URLError, OSError, ValueError):
        return None


def fetch_json(url: str, timeout: float = 8.0) -> dict | None:
    text = fetch(url, timeout)
    if text is None:
        return None
    try:
        body = json.loads(text)
    except ValueError:
        return None
    return body if isinstance(body, dict) else None


def served_identity(lb_url: str) -> dict:
    """WHAT is under load, from the LB, at the start of the session: two GETs and no assumption.

    `/v1/models` is the served id the LB routes `MODEL` to and the window it serves it with;
    `/lb/status` adds the catalog choice behind that id (`cluster.loaded_model.spec`) and the
    generation. Since 2026-10-02 the pair can be either Flash-Next or the dense 27B, and a report
    that said only `qwen3.8:latest` would not say which. Anything the LB did not answer stays None —
    a None in the report is a fact (the LB did not say), a default would be a guess."""
    out: dict = dict.fromkeys(IDENTITY_KEYS)
    models = fetch_json(f"{lb_url}/v1/models")
    data = (models or {}).get("data")
    first = data[0] if isinstance(data, list) and data and isinstance(data[0], dict) else {}
    if first.get("id"):
        out["served_model"] = str(first["id"])
    if isinstance(first.get("max_model_len"), (int, float)) and first["max_model_len"] > 0:
        out["max_model_len"] = int(first["max_model_len"])
    status = fetch_json(f"{lb_url}/lb/status")
    cluster = (status or {}).get("cluster")
    cluster = cluster if isinstance(cluster, dict) else {}
    loaded = cluster.get("loaded_model") if isinstance(cluster.get("loaded_model"), dict) else {}
    desired = cluster.get("desired") if isinstance(cluster.get("desired"), dict) else {}
    if loaded.get("spec"):
        out["model_choice"] = str(loaded["spec"])
    if out["served_model"] is None and loaded.get("served_name"):
        out["served_model"] = str(loaded["served_name"])
    if out["max_model_len"] is None and isinstance(loaded.get("max_model_len"), (int, float)):
        out["max_model_len"] = int(loaded["max_model_len"])
    if isinstance(desired.get("generation"), int):
        out["generation"] = desired["generation"]
    return out


def guard_traffic(metrics_url: str, before: float | None, deadline: float,
                  poll: float) -> tuple[bool, str]:
    """GUARD C. The engine's own prompt-token counter moving is the only proof that the load landed
    on this cluster. A session can report the right model and still be talking to something else."""
    baseline = before
    while True:
        text = fetch(metrics_url, timeout=4.0)
        if text is not None:
            total = prompt_tokens(text)
            if baseline is None:
                baseline = total
            elif total > baseline:
                return True, (f"the cluster counted {total - baseline:,.0f} prompt tokens of this "
                              f"session")
        if time.time() >= deadline:
            if baseline is None:
                return False, f"the engine's own metrics could not be read at {metrics_url}"
            return False, (f"the cluster counted no prompt tokens: the session's traffic did not "
                           f"land on this engine")
        time.sleep(poll)


def answers(stream_path: str, results_dir: pathlib.Path) -> str:
    """The reports, captured BY THIS SCRIPT. The model answers in chat and does not reliably call
    Write — proven 2026-09-30, two workflow notifications and an empty results directory — so what
    the session actually said is what we keep, beside anything it did write."""
    texts: list[str] = []
    notifications = writes = 0
    try:
        with open(stream_path, "r", errors="replace") as fh:
            for line in fh:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(event, dict):
                    continue
                if event.get("subtype") == "task_notification":
                    notifications += 1
                if event.get("type") == "assistant":
                    for block in event.get("message", {}).get("content", []) or []:
                        if block.get("type") == "text" and (block.get("text") or "").strip():
                            texts.append(block["text"])
                        if block.get("type") == "tool_use" and block.get("name") in ("Write", "Edit"):
                            writes += 1
    except OSError as exc:
        return f"the stream could not be read back: {exc}"
    results_dir.mkdir(parents=True, exist_ok=True)
    head = (f"# {results_dir.name} — {len(texts)} answer blocks, {notifications} workflow "
            f"notification{'' if notifications == 1 else 's'}, {writes} model Write/Edit calls\n\n")
    (results_dir / "session-answers.md").write_text(head + "\n\n---\n\n".join(texts))
    return (f"{len(texts)} answer blocks, {notifications} workflows, {writes} model writes "
            f"→ {results_dir}")


# ── the report the page is drawn from ──────────────────────────────────────────


def write_json(path: pathlib.Path, obj) -> None:
    """Whole, or not at all: the host runner reads this file while we write it."""
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


class Session:
    """One session: launch it guarded, keep the report honest while it runs, end at the cap."""

    def __init__(self, opts) -> None:
        self.opts = opts
        self.state = pathlib.Path(opts.state_dir)
        self.capture = self.state / CAPTURES / opts.label
        self.results = pathlib.Path(opts.results_dir) / opts.label
        self.report_path = self.state / REPORT
        self.started_at = time.time()
        self.proc: subprocess.Popen | None = None
        self.claude: int | None = None
        self.report = {"version": REPORT_VERSION, "label": opts.label, "pid": os.getpid(),
                       "claude_pid": None, "cap_min": opts.cap_min, "lb": opts.lb_url,
                       "model": opts.model, "started_at": self.started_at, "at": self.started_at,
                       "phase": "starting", "rc": None, "ended_at": None, "detail": "",
                       "guards": {"environ": None, "init": None, "traffic": None},
                       "results": str(self.results), "capture": str(self.capture),
                       **dict.fromkeys(IDENTITY_KEYS)}

    # the report, and nothing said in it that has not happened

    def say(self, **fields) -> None:
        self.report.update(fields)
        self.report["at"] = time.time()
        write_json(self.report_path, self.report)

    def note(self, line: str) -> None:
        print(line, flush=True)

    def guard(self, name: str, ok: bool) -> None:
        self.report["guards"][name] = ok

    # the launch

    def stop(self) -> None:
        """End the session now: SIGINT the group we started, then SIGKILL what is left. By GROUP,
        never by pattern — a `pkill -f` has killed an operator's own shell here before."""
        proc = self.proc
        if proc is None or proc.poll() is not None:
            return
        try:
            group = os.getpgid(proc.pid)
        except OSError:
            group = None
        for sig, wait in ((signal.SIGINT, 5.0), (signal.SIGKILL, 5.0)):
            try:
                if group is not None:
                    os.killpg(group, sig)
                else:
                    proc.send_signal(sig)
            except OSError:
                pass
            try:
                proc.wait(timeout=wait)
                return
            except subprocess.TimeoutExpired:
                continue

    def abort(self, code: int, detail: str) -> int:
        self.note(f"  !! GUARD: {detail} — the session is being stopped, nothing is measured")
        self.stop()
        rc = self.proc.poll() if self.proc else None
        self.say(phase="failed", detail=detail, rc=rc, ended_at=time.time())
        return code

    def run(self) -> int:
        opts = self.opts
        # The results directory is WIPED per run, so a run never reads or re-uses the previous
        # one's reports (the operator's own rule, 2026-09-30).
        if self.results.exists():
            for child in sorted(self.results.rglob("*"), reverse=True):
                child.rmdir() if child.is_dir() else child.unlink()
            self.results.rmdir()
        self.results.mkdir(parents=True, exist_ok=True)
        self.capture.mkdir(parents=True, exist_ok=True)
        stream_path = self.capture / "stream.jsonl"
        self.say(phase="starting", detail=f"starting one deep-research session, cap "
                                         f"{opts.cap_min:g} min, on {opts.lb_url}")

        # WHAT is loaded, from the LB itself, before the command is even built: the served id and
        # its window set the session's context, and all four facts go into the report so the page —
        # and anyone reading the capture later — knows which model these minutes were on.
        identity = served_identity(opts.lb_url)
        self.say(**identity)
        window = identity["max_model_len"] or DEFAULT_WINDOW
        self.note(f"  serving {identity['served_model'] or '? (the LB named no model)'} "
                  f"({identity['model_choice'] or '?'}, gen {identity['generation']}), "
                  f"window {window}{'' if identity['max_model_len'] else ' (default)'}")

        path = os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")
        argv = session_argv(opts.cap_min, opts.lb_url, opts.model,
                            os.environ.get("HOME", str(pathlib.Path.home())),
                            operator_name(), path, window,
                            https_proxy=opts.https_proxy, no_proxy=opts.no_proxy)
        env = launch_env(path)
        try:
            check_argv(argv, env, opts.lb_url)
        except ArgvRefused as exc:
            self.say(phase="failed", detail=f"refused before launching: {exc}", ended_at=time.time())
            self.note(f"  !! REFUSED: {exc}")
            return REFUSED

        # THE MODEL ALREADY LOADED, or nothing at all. The operator's rule, 2026-10-01: "benchmark
        # run should not unload model, should use already loaded model". So this never loads one —
        # and when nothing is serving it says so instead of starting a session that would have
        # nothing to talk to. An engine that answers its own /metrics is a model that is resident.
        before_text = fetch(opts.metrics_url, timeout=25.0)
        if not engine_loaded(before_text):
            detail = (f"no model is loaded: the engine at {opts.metrics_url} did not answer, and "
                      f"this control never loads one — start the pair on the cluster panel first")
            self.say(phase="failed", detail=detail, ended_at=time.time())
            self.note(f"  !! {detail}")
            return NO_MODEL
        (self.capture / "metrics-before.txt").write_text(before_text)
        before = prompt_tokens(before_text)

        message = json.dumps({"type": "user", "message": {
            "role": "user", "content": PROMPT.format(results=self.results)}})
        self.note(f"  claude start {time.strftime('%Y-%m-%dT%H:%M:%S%z')} — "
                  f"{' '.join(argv[:2])} … {timeout_arg(opts.cap_min)} claude")
        with open(stream_path, "wb") as out, open(self.capture / "stderr.txt", "wb") as err:
            # A session of its own, so stopping it stops the whole tree and nothing else, and so it
            # outlives the runner that started it.
            self.proc = subprocess.Popen(argv, env=env, stdin=subprocess.PIPE, stdout=out,
                                         stderr=err, start_new_session=True, close_fds=True,
                                         cwd=str(self.capture))
        # stdin stays OPEN for the whole session: in plain `-p` the process exits after its first
        # reply and the background deep-research workflow is stopped with it (2026-09-30).
        try:
            self.proc.stdin.write((message + "\n").encode())
            self.proc.stdin.flush()
        except OSError as exc:
            return self.abort(NO_CLAUDE, f"the session would not take its prompt: {exc}")
        self.say(phase="guards", claude_pid=None, detail="launched; checking the guards")

        deadline = time.time() + max(10.0, opts.poll * 20)
        while self.claude is None and time.time() < deadline:
            self.claude = claude_pid(self.proc.pid, opts.proc_root)
            if self.claude is None:
                if self.proc.poll() is not None:
                    break
                time.sleep(opts.poll)
        if self.claude is None:
            rc = self.proc.poll()
            if rc is None:
                return self.abort(NO_CLAUDE, "claude did not start, so nothing was measured")
            return self.abort(NO_CLAUDE, f"claude exited (exit {rc}) before the guards could look "
                                         f"at it, so nothing was measured")
        self.say(claude_pid=self.claude)

        ok, detail = guard_environ(self.claude, opts.lb_url, opts.proc_root)
        self.guard("environ", ok)
        if not ok:
            return self.abort(GUARD_ENVIRON, detail)
        self.note(f"  guard A ok: {detail}")

        ok, detail = guard_init(str(stream_path), opts.model,
                               time.time() + opts.init_timeout, opts.poll)
        self.guard("init", ok)
        if not ok:
            return self.abort(GUARD_INIT, detail)
        self.note(f"  guard B ok: {detail}")

        ok, detail = guard_traffic(opts.metrics_url, before,
                                   time.time() + opts.traffic_timeout, opts.poll)
        self.guard("traffic", ok)
        if not ok:
            return self.abort(GUARD_TRAFFIC, detail)
        self.note(f"  guard C ok: {detail}")

        # ── to the cap, always. The workload is continuous, so there is nothing to detect and
        # nothing to decide here: `timeout` ends the session, and until it does we keep the report
        # fresh so the page can say how far into the cap it is.
        self.say(phase="running", detail=f"a deep-research swarm is under way on {opts.lb_url}")
        while self.proc.poll() is None:
            try:
                self.proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self.say()
        rc = self.proc.returncode
        ended = time.time()
        minutes = (ended - self.started_at) / 60.0
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        after_text = fetch(opts.metrics_url, timeout=25.0)
        (self.capture / "metrics-after.txt").write_text(after_text or "")
        kept = answers(str(stream_path), self.results)
        if rc in CAP_RCS:
            detail = (f"the session ran {minutes:.1f} min and was ended by its "
                      f"{opts.cap_min:g} min cap (exit {rc}) · {kept}")
            code = 0
        else:
            detail = (f"the session ended early, after {minutes:.1f} min of its {opts.cap_min:g} "
                      f"min cap (exit {rc}) — a continuous workload should have been stopped by "
                      f"the clock · {kept}")
            code = ENDED_EARLY
        self.say(phase="finished", rc=rc, ended_at=ended, detail=detail)
        self.note(f"  claude exit {rc} after {minutes:.1f} min  [124/130 = the cap]")
        self.note(f"  {kept}")
        return code


def parse(argv=None):
    here = pathlib.Path(__file__).resolve().parent               # <repo>/lb
    repo = here.parent
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--label", default=time.strftime("S-%Y%m%d-%H%M%S"),
                   help="this session's name, in its capture and results directories")
    p.add_argument("--cap-min", type=float, default=CAP_MIN,
                   help="minutes of load; `timeout` ends the session at this")
    p.add_argument("--lb-url", default=LB_URL)
    p.add_argument("--metrics-url", default=default_metrics_url(),
                   help="the HEAD engine's own /metrics (default: LB_CLUSTER_URL + /metrics, from the environment or lb/.env)")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--state-dir", default=str(here / "state" / "benchq"))
    p.add_argument("--results-dir", default=str(repo / "worklog" / "bench" / "research"),
                   help="gitignored; this run's own directory under it is wiped before it starts")
    p.add_argument("--proc-root", default="/proc")
    p.add_argument("--init-timeout", type=float, default=60.0)
    p.add_argument("--traffic-timeout", type=float, default=120.0)
    p.add_argument("--poll", type=float, default=1.0)
    p.add_argument("--https-proxy", default=None,
                   help="the egress proxy claude's own fetches go through (the sealed runner's); none = none")
    p.add_argument("--no-proxy", default=None,
                   help="the hosts claude dials directly despite --https-proxy (the LB, the engine)")
    opts = p.parse_args(argv)
    if not opts.metrics_url:
        p.error("no head engine to read: LB_CLUSTER_URL is set neither in the environment nor in lb/.env "
                "(the head rank's vLLM URL, e.g. http://<head>:8100), and no --metrics-url was passed")
    return opts


def main(argv=None) -> int:
    opts = parse(argv)
    session = Session(opts)
    try:
        return session.run()
    except KeyboardInterrupt:
        # STOPPED IS NOT FAILED. This path is someone ending the run on purpose — a SIGINT from the
        # keyboard or from an operator clearing the way for a restart — and it exits cleanly. Recording
        # it as "failed" painted a red FAILED on the panel for every swarm anyone stopped, which is the
        # same defect as an exit code with no reason: a signal that cries wolf stops being read.
        session.stop()
        session.say(phase="stopped", detail="stopped at the keyboard", ended_at=time.time(),
                    rc=session.proc.poll() if session.proc else None)
        return 130


if __name__ == "__main__":
    sys.exit(main())
