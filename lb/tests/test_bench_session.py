"""ONE deep-research session, against the configuration already loaded (lb/bench_session.py).

WHAT THIS GUARDS, AND WHY IT IS WORTH A FILE OF ITS OWN. On 2026-09-30 an edit to the frozen
benchmark script left the backend variables on a DIFFERENT command from `claude`, and a benchmark
session ran for a quarter of an hour on the OPERATOR'S OWN Claude subscription instead of the local
cluster. Nothing failed; the money was simply spent. The frozen script now guards against that four
ways, and the single-session launcher behind the gateway's Run benchmark button has to carry the
same four or it is a cheaper way to repeat the same mistake:

  * `env -i` with an explicit list ON `claude`'s own command line, never exported into a parent
    shell — tested structurally, AND end to end with a POISONED parent environment holding a
    key-shaped fake, which must not reach the process that is launched;
  * the launched process's own `/proc/<pid>/environ` actually naming the LB and the dummy key;
  * the session's first `init` event naming the local model with `apiKeySource: ANTHROPIC_API_KEY`;
  * the engine's own `/metrics` showing prompt tokens move, i.e. the traffic really landed there.

NO MONEY IS SPENT HERE AND NO CLUSTER IS TOUCHED. `claude` is a fake on PATH that records the
environment and the argv it was given, answers two stream-json events and then idles; the engine's
`/metrics` is a stdlib server in this process which counts prompt tokens only when that fake
actually calls the base URL it was handed. So "the traffic landed" is a real causal chain and not a
mock: point the fake at the operator's own API instead and the guard fires.

The cap is seconds rather than fifteen minutes, and it is `timeout` that ends the session — exit
124, which is the NORMAL outcome and the one the page must not draw as a failure.
"""
from __future__ import annotations

import json
import os
import pathlib
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import pytest

import bench_session as session

#: What the fixture LB says is loaded: a window that is NOT the 262144 default, so a session that
#: took its context from the served model is told apart from one that assumed Flash-Next's.
SERVED, CHOICE, WINDOW, GENERATION = "Qwen/Qwen3.8-27B", "qwen3.8-27b-nvidia-nvfp4", 131072, 104

#: Key-shaped FAKE values in the parent environment, pointing at the real API: the `sk-ant-` prefix
#: is what the leak check looks for, and no credential is in them. `env -i` is the only reason they
#: cannot reach the session; that is exactly what the end-to-end test proves.
POISON = {"ANTHROPIC_BASE_URL": "https://api.anthropic.com",
          "ANTHROPIC_API_KEY": "sk-ant-FAKE-SUBSCRIPTION-KEY",
          "ANTHROPIC_AUTH_TOKEN": "sk-ant-FAKE-OAUTH-TOKEN",
          "CLAUDE_CODE_USE_BEDROCK": "1"}

#: The fake `claude`. It records what it was handed, emits an init event built FROM ITS OWN
#: environment (so a session pointed elsewhere reports elsewhere), calls its base URL once — which
#: is what moves the fixture engine's prompt tokens — answers, and then idles until it is stopped.
FAKE_CLAUDE = '''#!/usr/bin/env python3
import json, os, pathlib, sys, time, urllib.request
OUT = pathlib.Path(__OUT__)
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "environ.txt").write_text("\\n".join(f"{k}={v}" for k, v in sorted(os.environ.items())))
(OUT / "argv.json").write_text(json.dumps(sys.argv))
(OUT / "stdin.txt").write_text(sys.stdin.readline())
mode = __MODE__
base = os.environ.get("ANTHROPIC_BASE_URL", "")
model = os.environ.get("ANTHROPIC_MODEL", "")
source = "ANTHROPIC_API_KEY" if os.environ.get("ANTHROPIC_API_KEY") else "none"
if mode == "rogue":
    model, source = "claude-opus-5", "none"
print(json.dumps({"type": "system", "subtype": "init", "model": model,
                  "apiKeySource": source}), flush=True)
if mode not in ("rogue", "quiet"):
    try:
        urllib.request.urlopen(base + "/prompt-from-claude", timeout=5).read()
    except Exception as exc:
        print(f"the fake could not reach {base}: {exc}", file=sys.stderr, flush=True)
print(json.dumps({"type": "assistant", "message": {"role": "assistant", "content": [
    {"type": "text", "text": "ROUND 1 - energy storage"}]}}), flush=True)
print(json.dumps({"type": "system", "subtype": "task_notification"}), flush=True)
print(json.dumps({"type": "assistant", "message": {"role": "assistant", "content": [
    {"type": "text", "text": "ROUND 1 - printing press"}]}}), flush=True)
if mode == "instant":
    sys.exit(0)
if mode == "early":
    time.sleep(3)            # long enough to be guarded, far short of its cap
    sys.exit(0)
while True:
    time.sleep(0.1)
'''


# ── the fixture engine: prompt tokens that move only when the fake really calls in ──


class Engine:
    """The head engine's `/metrics`, and the one call the fake `claude` makes to its base URL. Every
    path asked for is recorded, so a test can prove the launcher asked for NOTHING ELSE — no
    `/lb/cluster`, no restart, no configuration change."""

    def __init__(self) -> None:
        self.prompt_tokens = 1_000_000.0
        self.paths: list[str] = []
        self.posts: list[str] = []
        self.lock = threading.Lock()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    @property
    def metrics_url(self) -> str:
        return f"{self.url}/metrics"

    def seen(self) -> list[str]:
        with self.lock:
            return list(self.paths)

    def close(self) -> None:
        self.server.shutdown()

    def _handler(self):
        engine = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a):
                pass

            def do_POST(self):
                """Nothing may POST here: a restart, an `enabled`, a generation — the swarm runs
                against what is loaded and changes nothing."""
                with engine.lock:
                    engine.posts.append(urlsplit(self.path).path)
                self.send_response(202)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def do_GET(self):
                path = urlsplit(self.path).path
                with engine.lock:
                    engine.paths.append(path)
                    if path == "/prompt-from-claude":
                        engine.prompt_tokens += 18_000.0
                    body = (f'vllm:prompt_tokens_total{{model_name="qwen3.8"}} '
                            f'{engine.prompt_tokens}\n'
                            f'vllm:num_requests_running{{model_name="qwen3.8"}} 3.0\n').encode()
                # The LB's two read-only answers about WHAT is loaded, as the real one shapes them.
                if path == "/v1/models":
                    body = json.dumps({"object": "list", "data": [
                        {"id": SERVED, "object": "model", "owned_by": "vllm-lb",
                         "max_model_len": WINDOW}]}).encode()
                elif path == "/lb/status":
                    body = json.dumps({"cluster": {
                        "state": "running",
                        "desired": {"generation": GENERATION, "model_choice": CHOICE},
                        "loaded_model": {"served_name": SERVED, "spec": CHOICE,
                                         "label": "Qwen3.8 27B · dense NVFP4",
                                         "max_model_len": WINDOW}}}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        return Handler


@pytest.fixture
def engine():
    e = Engine()
    yield e
    e.close()


def fake_claude(tmp_path, mode: str = "good") -> pathlib.Path:
    """A `claude` on PATH, in a directory of its own, recording into `<tmp>/fake-<mode>`."""
    binned = tmp_path / f"bin-{mode}"
    binned.mkdir(parents=True, exist_ok=True)
    out = tmp_path / f"fake-{mode}"
    script = binned / "claude"
    script.write_text(FAKE_CLAUDE.replace("__OUT__", repr(str(out))).replace("__MODE__", repr(mode)))
    script.chmod(0o755)
    return out


def run(tmp_path, engine, mode="good", *, cap_min=0.06, label="S-test", extra=()):
    """One session, in this process, against the fake `claude` and the fixture engine. Returns
    `(exit code, the session report, the fake's recordings directory)`."""
    out = fake_claude(tmp_path, mode)
    os.environ["PATH"] = f"{tmp_path / f'bin-{mode}'}:{os.environ['PATH']}"
    code = session.main(["--label", label, "--cap-min", str(cap_min),
                         "--lb-url", engine.url, "--metrics-url", engine.metrics_url,
                         "--model", "qwen3.8:latest",
                         "--state-dir", str(tmp_path / "benchq"),
                         "--results-dir", str(tmp_path / "bench" / "research"),
                         "--init-timeout", "20", "--traffic-timeout", "20",
                         "--poll", "0.2", *extra])
    report = json.loads((tmp_path / "benchq" / session.REPORT).read_text())
    return code, report, out


def environ_seen(out: pathlib.Path) -> dict[str, str]:
    lines = (out / "environ.txt").read_text().splitlines()
    return dict(line.split("=", 1) for line in lines if "=" in line)


# ── THE REGRESSION THAT COST REAL MONEY: the variables sit on claude's own argv ──


def test_the_backend_variables_sit_on_claudes_own_argv_between_env_i_and_claude():
    """The whole lesson of 2026-09-30 in one assertion. `env -i` wipes the inherited environment and
    the assignments are arguments of the very command being launched, so there is no shell, no
    export and no parent whose environment could be the one that counts."""
    argv = session.session_argv(cap_min=15, lb_url="http://127.0.0.1:8010",
                                model="qwen3.8:latest", home="/home/user", user="user",
                                path="/usr/bin:/bin")
    assert argv[0] == "env" and argv[1] == "-i", argv[:4]
    # `env` reads assignments until the first argument that is not one; that argument is the
    # command it execs, and the environment it built is what that command — and `claude` under it —
    # runs with. So the assignments must be exactly that leading run, and nothing may sit behind.
    command_at = next(i for i, a in enumerate(argv[2:], 2) if "=" not in a)
    assignments = argv[2:command_at]
    for want in ("ANTHROPIC_BASE_URL=http://127.0.0.1:8010", "ANTHROPIC_API_KEY=dummy",
                 "ANTHROPIC_MODEL=qwen3.8:latest", "HOME=/home/user", "PATH=/usr/bin:/bin"):
        assert want in assignments, f"{want} is not on claude's own command line: {argv}"
    assert "claude" in argv[command_at:], argv
    assert [a for a in argv[command_at:] if "=" in a] == [], \
        f"an assignment sits behind the command, where env never reads it: {argv[command_at:]}"
    claude_at = argv.index("claude")
    assert argv[claude_at + 1] == "-p", "claude's own flags follow it"


def test_nothing_about_the_backend_is_exported_into_the_launched_process_environment():
    """The environment handed to the child carries no ANTHROPIC_* at all: were the argv ever wrong,
    an inherited variable must not be there to make it work by accident."""
    env = session.launch_env("/usr/bin:/bin")
    assert [k for k in env if k.startswith(("ANTHROPIC", "CLAUDE"))] == [], env
    assert env.get("PATH") == "/usr/bin:/bin", "PATH is the one thing the child is given"


def test_an_argv_with_the_variables_in_the_wrong_place_is_refused_before_anything_launches():
    """The structural check the launcher runs on ITSELF before spending a minute of the pair: the
    exact shape of the 2026-09-30 defect — the assignments present, but not on claude's command."""
    good = session.session_argv(cap_min=15, lb_url="http://127.0.0.1:8010", model="m",
                                home="/h", user="u", path="/usr/bin")
    session.check_argv(good, session.launch_env("/usr/bin"), lb_url="http://127.0.0.1:8010")

    # the defect: the variables moved off claude, onto a wrapper that runs before it
    moved = ["env", "-i", "HOME=/h", "USER=u", "PATH=/usr/bin", "timeout", "-s", "INT", "-k", "30",
             "15m", "claude", "-p"]
    with pytest.raises(session.ArgvRefused) as exc:
        session.check_argv(moved, session.launch_env("/usr/bin"), lb_url="http://127.0.0.1:8010")
    assert "ANTHROPIC_BASE_URL" in str(exc.value)

    # and the other half: an inherited environment that would make a wrong argv work by accident
    with pytest.raises(session.ArgvRefused) as exc:
        session.check_argv(good, {"PATH": "/usr/bin", "ANTHROPIC_API_KEY": "sk-ant-FAKE"},
                           lb_url="http://127.0.0.1:8010")
    assert "inherited" in str(exc.value)


def test_without_env_i_nothing_is_launched_at_all():
    """`env` without `-i` inherits, and an inherited ANTHROPIC_BASE_URL beats nothing on argv."""
    argv = ["env", "ANTHROPIC_BASE_URL=http://127.0.0.1:8010", "ANTHROPIC_API_KEY=dummy", "claude"]
    with pytest.raises(session.ArgvRefused) as exc:
        session.check_argv(argv, {"PATH": "/usr/bin"}, lb_url="http://127.0.0.1:8010")
    assert "env -i" in str(exc.value)


# ── THE EGRESS PROXY: the one addition the operator approved for this file (2026-10-11) ───────────
#
# The sealed runner (lb/bench_runner) has no route out but its own egress proxy, which refuses Anthropic
# and every private range. `claude`'s WebFetch reaches a public page only through that proxy, so its
# address has to be on claude's command — and, like every other variable here, ONLY there.


def _assignments(argv: list[str]) -> list[str]:
    """The run of `KEY=VALUE` arguments `env` reads: right after `env -i`, up to the command."""
    command_at = next(i for i, a in enumerate(argv[2:], 2) if "=" not in a)
    return argv[2:command_at]


def test_without_a_proxy_the_command_names_no_proxy_at_all():
    """The host runner passes no proxy, and its command must stay exactly what it was: not one
    `*_PROXY` assignment, in any case."""
    argv = session.session_argv(cap_min=15, lb_url="http://127.0.0.1:8010", model="m", home="/h",
                                user="u", path="/usr/bin")
    assert [a for a in argv if a.split("=", 1)[0].upper().endswith("_PROXY")] == [], argv
    assert session.session_vars("http://127.0.0.1:8010", "m", "/h", "u", "/usr/bin") == \
        session.session_vars("http://127.0.0.1:8010", "m", "/h", "u", "/usr/bin",
                             https_proxy=None, no_proxy=None)


def test_a_proxy_rides_on_claudes_own_command_and_the_guarded_shape_still_holds():
    """HTTPS_PROXY and NO_PROXY are assignments `env` reads, before `timeout`, exactly like the backend
    variables — never exported, never inherited — and guard 0 still accepts the command."""
    argv = session.session_argv(cap_min=15, lb_url="http://lb.test:8010", model="m", home="/h",
                                user="u", path="/usr/bin", https_proxy="http://egress.test:3128",
                                no_proxy="lb.test,engine.test")
    assigned = _assignments(argv)
    assert "HTTPS_PROXY=http://egress.test:3128" in assigned, argv
    assert "NO_PROXY=lb.test,engine.test" in assigned, argv
    assert argv.index("timeout") == len(assigned) + 2, "the proxy sits in the run env reads, nothing behind"
    assert "ANTHROPIC_BASE_URL=http://lb.test:8010" in assigned and "ANTHROPIC_API_KEY=dummy" in assigned
    session.check_argv(argv, session.launch_env("/usr/bin"), lb_url="http://lb.test:8010")
    assert session.launch_env("/usr/bin") == {"PATH": "/usr/bin"}, "the child is still handed PATH alone"


def test_the_proxy_flags_default_to_none(monkeypatch):
    monkeypatch.delenv("LB_CLUSTER_URL", raising=False)
    opts = session.parse(["--metrics-url", "http://head.test:8100/metrics"])
    assert (opts.https_proxy, opts.no_proxy) == (None, None)
    opts = session.parse(["--metrics-url", "http://head.test:8100/metrics",
                          "--https-proxy", "http://egress.test:3128", "--no-proxy", "lb.test"])
    assert (opts.https_proxy, opts.no_proxy) == ("http://egress.test:3128", "lb.test")


def test_the_cap_is_a_timeout_argument_so_the_session_is_ended_by_the_clock():
    """The session runs to its cap and `timeout` ends it; no early-exit heuristic is wanted, and
    124 is the outcome the page must read as success."""
    argv = session.session_argv(cap_min=15, lb_url="http://x", model="m", home="/h", user="u",
                                path="/usr/bin")
    at = argv.index("timeout")
    assert argv[at:at + 5] == ["timeout", "-s", "INT", "-k", "30"]
    assert argv[at + 5] == "15m" and argv[at + 6] == "claude"
    assert session.timeout_arg(15) == "15m" and session.timeout_arg(0.05) == "3s"


def test_a_thirty_minute_cap_is_thirty_minutes_of_timeout_and_nothing_else_changes():
    """The duration is a knob of the operator's since 2026-10-05, and the ONLY thing it moves is
    `timeout`'s argument: the same `env -i`, the same assignments, the same claude, so none of the four
    guards can be weakened by asking for a longer run."""
    argv = session.session_argv(cap_min=30, lb_url="http://x", model="m", home="/h", user="u",
                                path="/usr/bin")
    fifteen = session.session_argv(cap_min=15, lb_url="http://x", model="m", home="/h", user="u",
                                   path="/usr/bin")
    at = argv.index("timeout")
    assert argv[at + 5] == "30m" and argv[at + 6] == "claude"
    assert [a if a != "30m" else "15m" for a in argv] == fifteen
    assert session.timeout_arg(240) == "240m", "four hours as `ps` can read it, not 14400s"


def test_the_cap_a_request_names_is_taken_and_a_request_that_names_none_keeps_the_default():
    """`cap_from_request` is what the host runner decides with. The default is the runner's, so a
    request from an LB that never heard of the field runs for the fifteen minutes it always did."""
    assert session.cap_from_request(None, session.CAP_MIN) == (15.0, "the runner's own default of 15 min")
    taken, words = session.cap_from_request(30, session.CAP_MIN)
    assert taken == 30.0 and "30 min the request asked for" in words
    lo, hi = session.CAP_MIN_RANGE
    for ok in (lo, hi, 17.5):
        assert session.cap_from_request(ok, session.CAP_MIN)[0] == float(ok)


@pytest.mark.parametrize("asked", [4, 0, -1, 241, 10_000, True, "30", [30], float("nan")])
def test_a_cap_outside_the_range_is_refused_rather_than_rounded_to_the_default(asked):
    """(None, why) and never (15.0, …): a run at the default when another was asked for would be filed
    under minutes it never had. `True` is here because `isinstance(True, int)` is true in Python."""
    taken, why = session.cap_from_request(asked, session.CAP_MIN)
    assert taken is None, f"{asked!r} was accepted as {taken}"
    assert "5" in why and "240" in why


def test_the_range_leaves_room_for_the_guards_and_for_what_the_operator_asked_for():
    """The floor has to be longer than the ~200 s the three guards can spend before the session is even
    `running` (20 s to find the pid, `--init-timeout` 60 s, `--traffic-timeout` 120 s), or the cap could
    end a run while guard C was still proving the traffic landed. The ceiling has to be at least the 30
    minutes the operator asked for on 2026-10-05."""
    opts = session.parse(["--metrics-url", "http://head.test:8100/metrics"])
    lo, hi = session.CAP_MIN_RANGE
    guards_s = 20.0 + opts.init_timeout + opts.traffic_timeout
    assert lo * 60.0 > guards_s, f"a {lo:g} min cap can expire inside the guards' own {guards_s:g} s"
    assert hi >= 30.0 and session.CAP_MIN == 15.0
    assert opts.cap_min == session.CAP_MIN, "`--cap-min` left out is still a quarter of an hour"


# ── the session, end to end, against a fake claude and a fixture engine ────────


def test_a_session_runs_to_its_cap_and_the_cap_is_a_success(tmp_path, engine, monkeypatch):
    monkeypatch.setattr(os, "environ", dict(os.environ))
    code, report, out = run(tmp_path, engine)
    assert report["phase"] == "finished", report
    assert report["rc"] in (124, 130), f"the cap must end it, not an early exit: {report}"
    assert report["ended_at"] and report["ended_at"] >= report["started_at"]
    assert code == 0
    assert "cap" in report["detail"]
    assert report["guards"] == {"environ": True, "init": True, "traffic": True}
    assert report["cap_min"] == pytest.approx(0.06)


def test_a_poisoned_parent_environment_never_reaches_the_launched_claude(tmp_path, engine,
                                                                        monkeypatch):
    """THE one that cost money, end to end. The parent holds a key-shaped fake for the real API;
    the process that runs sees the local LB and the dummy key, and nothing else."""
    monkeypatch.setattr(os, "environ", dict(os.environ))
    for key, value in POISON.items():
        os.environ[key] = value
    code, report, out = run(tmp_path, engine)
    seen = environ_seen(out)
    assert seen["ANTHROPIC_BASE_URL"] == engine.url
    assert seen["ANTHROPIC_API_KEY"] == "dummy"
    assert "api.anthropic.com" not in (out / "environ.txt").read_text()
    assert "sk-ant" not in (out / "environ.txt").read_text()
    assert "CLAUDE_CODE_USE_BEDROCK" not in seen, "env -i leaves nothing of the parent behind"
    assert code == 0 and report["phase"] == "finished"


def test_the_launched_claude_gets_the_proxy_it_was_given_and_never_an_inherited_one(tmp_path, engine,
                                                                                   monkeypatch):
    """End to end, with a parent environment that carries a proxy of its own: `--https-proxy` and
    `--no-proxy` are what the launched process sees, and without them it sees no proxy at all.

    The parent's NO_PROXY=* keeps THIS process's own reads of the fixture engine direct, and urllib's
    global opener — which captures the proxies it saw at its first use — is put back afterwards, so
    the poisoned environment cannot leak into the next test's fetches."""
    import urllib.request
    monkeypatch.setattr(os, "environ", dict(os.environ))
    monkeypatch.setattr(urllib.request, "_opener", None)
    for key, value in {"HTTPS_PROXY": "http://inherited.test:1", "https_proxy": "http://inherited.test:1",
                       "HTTP_PROXY": "http://inherited.test:1", "NO_PROXY": "*"}.items():
        os.environ[key] = value
    code, report, out = run(tmp_path, engine, label="S-proxy",
                            extra=["--https-proxy", "http://egress.test:3128", "--no-proxy", "lb.test"])
    seen = environ_seen(out)
    assert seen.get("HTTPS_PROXY") == "http://egress.test:3128"
    assert seen.get("NO_PROXY") == "lb.test"
    assert "inherited.test" not in (out / "environ.txt").read_text()
    assert "HTTP_PROXY" not in seen and "https_proxy" not in seen
    assert code == 0 and report["phase"] == "finished"

    code, report, out = run(tmp_path, engine, mode="plain", label="S-noproxy")
    assert [k for k in environ_seen(out) if k.upper().endswith("_PROXY")] == []
    assert code == 0


def test_the_session_asks_the_cluster_for_nothing_but_its_metrics(tmp_path, engine, monkeypatch):
    """No restart, no configuration change, no queue, and above all NOTHING THAT LOADS OR UNLOADS A
    MODEL (the operator, 2026-10-01: "benchmark run should not unload model, should use already
    loaded model"). The fixture engine records every path it was asked for, and this is the list."""
    monkeypatch.setattr(os, "environ", dict(os.environ))
    code, report, out = run(tmp_path, engine)
    asked = set(engine.seen())
    # `/v1/models` and `/lb/status` are the two READS about what is loaded (since 2026-10-02);
    # they change nothing and are the whole of what is asked beyond the counter.
    assert asked <= {"/metrics", "/prompt-from-claude", "/v1/models", "/lb/status"}, asked
    assert "/prompt-from-claude" in asked, "the fake claude's own traffic is what moves the tokens"
    assert engine.posts == [], "a swarm POSTs nothing: no /lb/cluster, no restart, no setting"
    assert code == 0


def test_the_session_reads_which_model_is_loaded_and_gives_claude_that_models_window(tmp_path, engine,
                                                                                    monkeypatch):
    """The pair can be Flash-Next or the dense 27B (2026-10-02). The session asks the LB which —
    `/v1/models` for the served id and its window, `/lb/status` for the catalog choice and the
    generation — writes all four into its report, and sizes `claude`'s context from the served
    window rather than from Flash-Next's 262144. The name `claude` is told to ask for stays the
    LB's alias; what that alias IS this time is the report's job to say."""
    monkeypatch.setattr(os, "environ", dict(os.environ))
    code, report, out = run(tmp_path, engine)
    assert code == 0
    assert report["version"] == session.REPORT_VERSION == 2
    assert (report["served_model"], report["model_choice"]) == (SERVED, CHOICE)
    assert (report["generation"], report["max_model_len"]) == (GENERATION, WINDOW)
    seen = environ_seen(out)
    assert seen["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == str(WINDOW)
    assert seen["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] == str(int(WINDOW * 0.8))
    assert seen["ANTHROPIC_MODEL"] == "qwen3.8:latest", "the alias the LB routes, as the frozen script"
    assert "/v1/models" in engine.seen() and "/lb/status" in engine.seen()


def test_an_lb_that_names_no_model_leaves_the_identity_none_and_the_window_at_the_default():
    """Nothing answered is four Nones — the LB did not say — never a guessed Flash-Next; and the
    window falls back to the frozen script's 262144 / 209715 pair, exactly as before."""
    identity = session.served_identity("http://127.0.0.1:1")
    assert identity == dict.fromkeys(session.IDENTITY_KEYS)
    argv = session.session_argv(cap_min=15, lb_url="http://x", model="m", home="/h", user="u",
                                path="/usr/bin")
    assert "CLAUDE_CODE_MAX_CONTEXT_TOKENS=262144" in argv
    assert "CLAUDE_CODE_AUTO_COMPACT_WINDOW=209715" in argv
    windowed = session.session_argv(cap_min=15, lb_url="http://x", model="m", home="/h", user="u",
                                    path="/usr/bin", window=131072)
    assert "CLAUDE_CODE_MAX_CONTEXT_TOKENS=131072" in windowed
    assert "CLAUDE_CODE_AUTO_COMPACT_WINDOW=104857" in windowed


def test_a_swarm_refuses_to_start_when_nothing_is_serving_because_it_never_loads_a_model(
        tmp_path, monkeypatch):
    """The operator's rule, 2026-10-01: the swarm runs against the model ALREADY LOADED. If nothing
    is serving, the honest answer is to refuse — loading a model is a deliberate act of theirs on
    the cluster panel, never a side effect of a button whose job is to generate load."""
    monkeypatch.setattr(os, "environ", dict(os.environ))
    out = fake_claude(tmp_path, "good")
    os.environ["PATH"] = f"{tmp_path / 'bin-good'}:{os.environ['PATH']}"
    code = session.main(["--label", "S-test", "--cap-min", "0.05",
                         "--lb-url", "http://127.0.0.1:1",
                         "--metrics-url", "http://127.0.0.1:1/metrics",   # nothing listens there
                         "--state-dir", str(tmp_path / "benchq"),
                         "--results-dir", str(tmp_path / "research"), "--poll", "0.2"])
    report = json.loads((tmp_path / "benchq" / session.REPORT).read_text())
    assert code == session.NO_MODEL and report["phase"] == "failed"
    assert "no model is loaded" in report["detail"]
    assert not (out / "environ.txt").exists(), "claude was never launched, so nothing was spent"


def test_the_prompt_is_the_two_concurrent_researches_relaunched_continuously(tmp_path, engine,
                                                                            monkeypatch):
    monkeypatch.setattr(os, "environ", dict(os.environ))
    code, report, out = run(tmp_path, engine)
    sent = json.loads((out / "stdin.txt").read_text())
    text = sent["message"]["content"]
    assert sent["type"] == "user"
    assert "/deep-research" in text
    assert "energy storage" in text and "printing press" in text
    assert "30 agents" in text and "60 agents" in text
    assert report["results"] in text, "the prompt names the directory this run writes into"
    argv = json.loads((out / "argv.json").read_text())
    assert argv[1:] == ["-p", "--verbose", "--input-format", "stream-json", "--output-format",
                        "stream-json", "--permission-mode", "bypassPermissions"]


def test_the_answers_are_captured_by_the_launcher_because_the_model_does_not_write(tmp_path, engine,
                                                                                  monkeypatch):
    """Proven on 2026-09-30: the model answers in chat and never calls Write. The session's own
    answers are what we keep, in a results directory wiped at the start of every run."""
    monkeypatch.setattr(os, "environ", dict(os.environ))
    stale = tmp_path / "bench" / "research" / "S-test" / "from-the-last-run.md"
    stale.parent.mkdir(parents=True)
    stale.write_text("a report from a run that is over")
    code, report, out = run(tmp_path, engine)
    answers = pathlib.Path(report["results"]) / "session-answers.md"
    body = answers.read_text()
    assert "ROUND 1 - energy storage" in body and "ROUND 1 - printing press" in body
    assert "1 workflow notification" in body
    assert not stale.exists(), "a run must never read or re-use the previous run's reports"
    assert code == 0


def test_the_capture_keeps_the_stream_the_stderr_and_the_metrics_either_side(tmp_path, engine,
                                                                            monkeypatch):
    monkeypatch.setattr(os, "environ", dict(os.environ))
    code, report, out = run(tmp_path, engine)
    capture = pathlib.Path(report["capture"])
    stream = (capture / "stream.jsonl").read_text()
    assert '"subtype": "init"' in stream or '"subtype":"init"' in stream
    before = float(session.prompt_tokens((capture / "metrics-before.txt").read_text()))
    after = float(session.prompt_tokens((capture / "metrics-after.txt").read_text()))
    assert after > before, "the engine's own counter is snapshotted either side of the run"
    assert (capture / "stderr.txt").exists()


def test_a_session_that_ends_before_its_cap_is_reported_as_that_and_not_as_success(tmp_path, engine,
                                                                                   monkeypatch):
    """`timeout` returning 0 means `claude` exited on its own: the workload is continuous, so an
    early end is a defect to show, never a tick."""
    monkeypatch.setattr(os, "environ", dict(os.environ))
    code, report, out = run(tmp_path, engine, mode="early", cap_min=1.0)
    assert report["phase"] == "finished" and report["rc"] == 0
    assert "early" in report["detail"]
    assert code != 0, "the runner's log must not call an early end a clean run"
    assert report["guards"] == {"environ": True, "init": True, "traffic": True}


def test_a_session_that_dies_before_the_guards_can_look_is_not_a_measured_run(tmp_path, engine,
                                                                             monkeypatch):
    """A `claude` that is gone in the first fraction of a second cannot be guarded at all: its
    `/proc` is already empty. That is a failure with its own words, not "did not start" and
    certainly not a run.

    The `/proc` it is guarded through is made EMPTY rather than raced. The fake's "instant" mode
    still imports urllib, writes its recordings and calls the engine before it exits — a few hundred
    milliseconds — and under a loaded xdist run the guards would find it alive, pass all three, and
    the test would fail on a race it was never about (it failed that way on an unmodified tree,
    2026-10-01). An empty `--proc-root` is the scenario stated literally: the session can never see
    the process, waits for it to exit, and must say so in its own words."""
    monkeypatch.setattr(os, "environ", dict(os.environ))
    empty_proc = tmp_path / "proc-already-empty"
    empty_proc.mkdir()
    code, report, out = run(tmp_path, engine, mode="instant", cap_min=1.0,
                            extra=["--proc-root", str(empty_proc)])
    assert code == session.NO_CLAUDE and report["phase"] == "failed"
    assert "before the guards" in report["detail"]
    assert report["guards"] == {"environ": None, "init": None, "traffic": None}


# ── the three guards, each one firing for real ─────────────────────────────────


def test_a_session_on_the_wrong_model_or_credential_is_killed_within_seconds(tmp_path, engine,
                                                                            monkeypatch):
    """GUARD B. A session whose init event says `claude-opus-5` and `apiKeySource: none` is the
    2026-09-30 accident as it actually looked in the log: it is stopped, not measured."""
    monkeypatch.setattr(os, "environ", dict(os.environ))
    started = time.time()
    code, report, out = run(tmp_path, engine, mode="rogue", cap_min=5.0)
    assert code == 5 and report["phase"] == "failed"
    assert report["guards"]["init"] is False
    assert "apiKeySource" in report["detail"] or "model" in report["detail"]
    assert time.time() - started < 60, "a wrong session is killed in seconds, not minutes"
    assert report["rc"] is not None, "the session's own exit is still recorded"
    assert not (pathlib.Path(report["results"]) / "session-answers.md").exists(), \
        "a session that failed its guards measured nothing, so it keeps nothing"


def test_a_session_the_cluster_never_sees_is_killed_and_not_measured(tmp_path, engine, monkeypatch):
    """GUARD C. The init event can be right while the traffic goes somewhere else; the engine's own
    prompt-token counter is the only thing that proves where it landed."""
    monkeypatch.setattr(os, "environ", dict(os.environ))
    code, report, out = run(tmp_path, engine, mode="quiet", cap_min=5.0,
                            extra=["--traffic-timeout", "3"])
    assert code == 6 and report["phase"] == "failed"
    assert report["guards"] == {"environ": True, "init": True, "traffic": False}
    assert "prompt tokens" in report["detail"]


def test_the_environ_guard_reads_the_launched_process_own_environment(tmp_path):
    """GUARD A, against a /proc of our own making: it is the ONE guard that looks at what the kernel
    says the process was given, rather than at what we believe we asked for."""
    proc = tmp_path / "proc"
    for pid, env in ((4001, {"ANTHROPIC_BASE_URL": "http://127.0.0.1:8010",
                             "ANTHROPIC_API_KEY": "dummy"}),
                     (4002, {"ANTHROPIC_BASE_URL": "https://api.anthropic.com",
                             "ANTHROPIC_API_KEY": "sk-ant-FAKE"}),
                     (4003, {"ANTHROPIC_API_KEY": "dummy"})):
        d = proc / str(pid)
        d.mkdir(parents=True)
        (d / "environ").write_bytes(b"".join(f"{k}={v}\0".encode() for k, v in env.items()))
        (d / "cmdline").write_bytes(b"claude\0-p\0")
    ok, detail = session.guard_environ(4001, "http://127.0.0.1:8010", str(proc))
    assert ok is True and "4001" in detail
    ok, detail = session.guard_environ(4002, "http://127.0.0.1:8010", str(proc))
    assert ok is False and "api.anthropic.com" not in detail  # a key is never echoed
    ok, detail = session.guard_environ(4003, "http://127.0.0.1:8010", str(proc))
    assert ok is False


def test_the_launched_claude_is_found_as_the_timeouts_own_child(tmp_path):
    """`timeout` is what Popen holds; `claude` is its child, and guard A needs that pid and no
    other. Read from /proc the way the kernel presents it: PPid in `status`, name in `cmdline`."""
    proc = tmp_path / "proc"
    for pid, ppid, argv in ((5000, 1, ("timeout", "-s", "INT", "15m", "claude")),
                            (5001, 5000, ("claude", "-p", "--verbose")),
                            (5002, 5000, ("sleep", "960")),
                            (5003, 1, ("claude", "-p"))):
        d = proc / str(pid)
        d.mkdir(parents=True)
        (d / "status").write_text(f"Name:\tx\nPid:\t{pid}\nPPid:\t{ppid}\n")
        (d / "cmdline").write_bytes(b"\0".join(a.encode() for a in argv) + b"\0")
    assert session.claude_pid(5000, str(proc)) == 5001, "not the sleep, and not the unrelated claude"
    assert session.claude_pid(5002, str(proc)) is None


# ── what it must never touch ───────────────────────────────────────────────────


def test_the_session_writes_nothing_in_the_harnesss_own_directory(tmp_path, engine, monkeypatch):
    """`worklog/bench/claude` is the matrix's bookkeeping: queue.txt, done.txt, runs.txt and every
    capture of a matrix run. A session is not a matrix run — it marks nothing done and banks no
    line — so it must leave every byte of that directory alone."""
    monkeypatch.setattr(os, "environ", dict(os.environ))
    claude_dir = tmp_path / "bench" / "claude"
    claude_dir.mkdir(parents=True)
    (claude_dir / "queue.txt").write_text("P1-s8-b8k-sp4 8 8192 4 15\n")
    (claude_dir / "done.txt").write_text("P1-s8-b8k-sp4\n")
    (claude_dir / "runs.txt").write_text("P1-s8-b8k-sp4 8 8192 4 1 2 0\n")
    before = {p.name: p.read_bytes() for p in claude_dir.iterdir()}
    code, report, out = run(tmp_path, engine)
    after = {p.name: p.read_bytes() for p in claude_dir.iterdir()}
    assert after == before and code == 0


def test_the_report_names_the_session_while_it_is_still_running(tmp_path, engine, monkeypatch):
    """The page's whole middle state. The report is written when the session starts, not when it
    ends, so "running, 3 of 15 minutes" is a fact read off disk and never an extrapolation."""
    monkeypatch.setattr(os, "environ", dict(os.environ))
    seen: list[dict] = []
    report_path = tmp_path / "benchq" / session.REPORT
    stop = threading.Event()

    def watch():
        while not stop.is_set():
            try:
                seen.append(json.loads(report_path.read_text()))
            except (OSError, ValueError):
                pass
            time.sleep(0.05)

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    code, report, out = run(tmp_path, engine, cap_min=0.1)
    stop.set()
    watcher.join(timeout=2)
    running = [r for r in seen if r.get("phase") == "running"]
    assert running, f"no report ever said the session was running: {[r.get('phase') for r in seen]}"
    assert running[0]["cap_min"] == pytest.approx(0.1)
    assert running[0]["claude_pid"] and running[0]["pid"] == os.getpid()
    assert running[0]["label"] == "S-test" and running[0]["rc"] is None
    assert code == 0


def test_the_session_recognises_either_engine_at_its_own_metrics():
    """The 15-minute swarm is the benchmark of record, and it refused to run against SGLang. The gate was
    `"vllm:" not in text`: a pair serving SGLang answers entirely in `sglang:` series, so a healthy
    generation-121 pair was read as nothing loaded and two sessions failed before a token was sent
    (2026-10-03). The prompt counter had the same assumption."""
    import importlib.util
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("bench_session_mod", root / "bench_session.py")
    bs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bs)

    sglang = (root / "tests" / "fixtures" / "sglang-metrics-2026-10-02-busy-a.txt").read_text()
    assert bs.engine_loaded(sglang), "a live SGLang pair must read as loaded"
    assert bs.prompt_tokens(sglang) == 5489.0, "SGLang's own prefill counter, summed over its labels"

    # vLLM's reading is untouched: the frozen script's awk summed exactly this.
    vllm = 'vllm:prompt_tokens_total{model_name="x"} 1200.0\nvllm:generation_tokens_total{a="b"} 7.0\n'
    assert bs.engine_loaded(vllm) and bs.prompt_tokens(vllm) == 1200.0
    # And nothing at all is still nothing: an engine that did not answer must not read as loaded.
    assert not bs.engine_loaded("# HELP something_else 1\n")
    assert not bs.engine_loaded(None)

    # TensorFold (the pair's third engine, 2026-10-29) answers in `tensorfold:` series and its prompt counter
    # carries NO labels — the gate read generation 280 as nothing loaded (2026-10-29).
    tf = (root / "tests" / "fixtures" / "tensorfold-metrics-2026-10-29-gen280.txt").read_text()
    assert bs.engine_loaded(tf), "a live TensorFold pair must read as loaded"
    assert bs.prompt_tokens(tf) == 56.0, "TensorFold's unlabelled prompt counter"
    assert bs.prompt_tokens("tensorfold:prompt_tokens_total_other 9\n") is None, "the name, not a prefix of it"


def test_the_session_reads_the_engines_through_the_probes_one_vocabulary_table():
    """DRY: the swarm and the LB's probe read an engine's /metrics through ONE definition
    (vllm_lb/engine_metrics.py). Two private prefix lists in the session refused healthy pairs twice —
    SGLang, then TensorFold — while the probe already read both. A transcription scrape is an engine,
    but not a chat model a deep-research session can use, so it does not count as loaded."""
    from vllm_lb import engine_metrics
    assert session.engine_metrics is engine_metrics
    assert not hasattr(session, "_ENGINE_SERIES") and not hasattr(session, "_PROMPT_TOKENS")
    for prefix, engine, _ in engine_metrics.VOCABULARIES:
        assert session.engine_loaded(f"{prefix}x 1\n") is (engine in engine_metrics.CHAT_ENGINES), engine
    speech = (pathlib.Path(__file__).resolve().parent / "fixtures" / "transcription-metrics-2026-10-23-busy.txt").read_text()
    assert engine_metrics.engine_of(speech) == engine_metrics.ENGINE_TRANSCRIPTION
    assert not session.engine_loaded(speech), "a live speech engine is no chat model"


# ── DEP-16: nothing about the operator's cluster or account is a constant in this file ──────────

def test_the_metrics_url_is_the_head_ranks_own_metrics_from_lb_cluster_url(monkeypatch):
    """LB_CLUSTER_URL is the head rank's vLLM API on its own port; its /metrics is the engine's
    counter the guard reads. One variable, not a second address in the source."""
    monkeypatch.setenv("LB_CLUSTER_URL", "http://head.example:8100/")
    assert session.parse([]).metrics_url == "http://head.example:8100/metrics"


def test_without_a_cluster_url_the_session_refuses_to_guess_the_head(monkeypatch, capsys, tmp_path):
    monkeypatch.delenv("LB_CLUSTER_URL", raising=False)
    monkeypatch.setattr(session, "LB_ENV_FILE", tmp_path / "absent.env")
    with pytest.raises(SystemExit) as refused:
        session.parse([])
    assert refused.value.code == 2
    assert "LB_CLUSTER_URL" in capsys.readouterr().err


def test_an_explicit_metrics_url_needs_no_cluster_url(monkeypatch):
    monkeypatch.delenv("LB_CLUSTER_URL", raising=False)
    assert session.parse(["--metrics-url", "http://x:8100/metrics"]).metrics_url == "http://x:8100/metrics"


def test_the_operator_is_the_account_running_the_session(monkeypatch):
    """The claude process runs as whoever started the session, not as a name written in the source."""
    monkeypatch.setattr(session.getpass, "getuser", lambda: "someone-else")
    assert session.operator_name() == "someone-else"


def test_the_head_is_the_one_the_lbs_own_env_file_names_when_the_environment_has_none(monkeypatch, tmp_path):
    """The host runner is a systemd unit with no LB_CLUSTER_URL of its own: it reads the SAME value the LB
    reads, from lb/.env, so the head it measures is the head the LB serves from — one definition, two
    readers. Compose's own .env rules: comments and blanks skipped, quotes stripped, the last line wins."""
    monkeypatch.delenv("LB_CLUSTER_URL", raising=False)
    env = tmp_path / ".env"
    env.write_text("# the LB's settings\nPG_PW=not-read\nLB_CLUSTER_URL=http://old.example:8100\n\n"
                   "LB_CLUSTER_URL='http://head.example:8100/'\nLB_CLUSTER_HOSTS=a=b=c=d\n")
    monkeypatch.setattr(session, "LB_ENV_FILE", env)
    assert session.parse([]).metrics_url == "http://head.example:8100/metrics"


def test_the_environment_names_the_head_before_the_lbs_env_file(monkeypatch, tmp_path):
    """Compose's precedence: a variable set in the environment wins over the same one in .env."""
    env = tmp_path / ".env"
    env.write_text("LB_CLUSTER_URL=http://file.example:8100\n")
    monkeypatch.setattr(session, "LB_ENV_FILE", env)
    monkeypatch.setenv("LB_CLUSTER_URL", "http://shell.example:8100")
    assert session.parse([]).metrics_url == "http://shell.example:8100/metrics"
