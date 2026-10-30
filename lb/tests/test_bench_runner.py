"""The host runner behind the gateway's swarm button (lb/bench_queue_runner.py).

The LB cannot start a process on box-c, so the button's start is a request and this runner is what
performs it. Since 2026-10-01 the request is for ONE deep-research session against the model already
loaded — `lb/bench_session.py` — and the thirty-run MATRIX is a terminal-only operation
(`--start-matrix`). Everything that could go badly wrong is here rather than in the page:

  * a SECOND swarm while one is up, or a swarm while the matrix is up: two at once would drive the
    same TP=2 pair through different work and spoil both measurements, and the refusal has to name
    WHICH one is up, because "already running" without a subject is the one answer an operator
    cannot act on;
  * a request asking for the matrix. A `start.request` left behind by an older LB must not start a
    thirteen-hour matrix after a deploy, so anything but a session is refused by name;
  * the matrix starting over a `hold` or `pause` somebody else placed, or against a frozen script
    whose seal no longer matches;
  * and — the one that has actually bitten twice — deciding "is something running" with a pattern
    match that also matches the asker's own command line.

Nothing here runs the real harness and nothing here spends money: the launch tests point the runner
at stub scripts in a throwaway directory, so `launch_session` and `launch_matrix` are exercised for
real without the operator's GPUs.
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import bench_queue_runner as runner
import bench_session

#: The watchdog shells that run beside a real queue: their `bash -c '<script text>'` MENTIONS the
#: harness's filenames, so `pgrep -f fullclaudevllm_queue.sh` and `ps | grep` both match them. This
#: is a trimmed copy of one that was live on box-c on 2026-09-30.
WATCHDOG = ("/bin/bash", "-c",
            "cd /srv/repo/worklog/bench && python3 - <<'PY'\n"
            "if not procs('fullclaudevllm_queue.sh'): print('queue GONE')\n"
            "if 'fullclaudevllm_bench.sh' in c: pass\nPY")
#: The real things, as `ps -eo pid,cmd` shows them: the name is an ARGUMENT of its own.
REAL_QUEUE = ("/bin/bash", "./fullclaudevllm_queue.sh")
REAL_BENCH = ("/bin/bash", "./fullclaudevllm_bench.sh", "P1-s16-b8k-sp4", "16", "8192", "4", "15")
REAL_SWARM = ("/usr/bin/python3", "/srv/repo/lb/bench_session.py",
              "--label", "S-20260101-000001", "--cap-min", "15")

BENCH_BODY = b"#!/bin/bash\n# FROZEN - DO NOT EDIT\nexit 0\n"
FLASH_NEXT, DENSE_27B = "qwen3.8-flash-next-hibrid48", "qwen3.8-27b-nvidia-nvfp4"
MATRIX = (f"# LABEL SEQS BATCHED SPEC CAP_MIN\n# model: {FLASH_NEXT}\n"
          "P1-s16-b8k-sp4 16 8192 4 15\nP1-s8-b8k-sp4 8 8192 4 15\n")
#: `stored_model()`'s answer when the LB has Flash-Next stored: what the frozen script's restart loads.
STORED = {"model_choice": FLASH_NEXT, "generation": 90, "loaded": None, "state": "off", "error": ""}
#: The frozen queue script's own parser, verbatim from fullclaudevllm_queue.sh line 12 — the proof
#: that a `# model:` line is invisible to it is running THAT, not a reading of it.
FROZEN_PARSER = ("grep -v '^#' claude/queue.txt | grep -v '^\\s*$' | while read -r l q b s c; do "
                 "grep -qx \"$l\" claude/done.txt || { echo \"$l $q $b $s $c\"; break; }; done")


class StatusStub:
    """`/lb/status` with a chosen stored model_choice, so `--start-matrix` never asks the live LB."""

    def __init__(self, model_choice: str | None, generation: int = 90) -> None:
        body = {"cluster": {"state": "off", "loaded_model": None,
                            "desired": {"model_choice": model_choice, "generation": generation}}}
        self.asked: list[str] = []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a):
                pass

            def do_GET(self):
                stub.asked.append(self.path)
                raw = json.dumps(body).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self) -> None:
        self.server.shutdown()


@pytest.fixture
def lb_stub():
    stub = StatusStub(FLASH_NEXT)
    yield stub
    stub.close()


def fake_proc(tmp_path, procs: dict[int, tuple[str, ...]]) -> str:
    """A /proc to scan: one directory per pid holding a NUL-separated `cmdline`, as Linux has it."""
    root = tmp_path / "proc"
    root.mkdir(exist_ok=True)
    for pid, argv in procs.items():
        d = root / str(pid)
        d.mkdir(exist_ok=True)
        (d / "cmdline").write_bytes(b"\0".join(a.encode() for a in argv) + b"\0")
    (root / "meminfo").write_text("MemTotal: 1 kB\n")      # a non-pid entry, as /proc has many
    return str(root)


def harness(tmp_path, *, done=(), seal=True):
    """A throwaway copy of the harness's layout: bench dir, its frozen script and seal, claude/."""
    bench = tmp_path / "bench"
    queue = bench / "claude"
    queue.mkdir(parents=True, exist_ok=True)
    (bench / runner.BENCH_SCRIPT).write_bytes(BENCH_BODY)
    import hashlib
    digest = hashlib.sha256(BENCH_BODY).hexdigest() if seal else "0" * 64
    (bench / runner.SEAL).write_text(f"{digest}\n")
    (queue / runner.QUEUE_TXT).write_text(MATRIX)
    (queue / runner.DONE_TXT).write_text("".join(f"{d}\n" for d in done))
    return bench, queue


@pytest.fixture(autouse=True)
def _a_head_engine(monkeypatch):
    """The runner reads the head engine's /metrics from LB_CLUSTER_URL (no address in the source)."""
    monkeypatch.setenv("LB_CLUSTER_URL", "http://head.test:8100")


def options(tmp_path, bench, queue, *, proc=None, dry_run=False, extra=()):
    return runner.parse(["--bench-dir", str(bench), "--queue-dir", str(queue),
                         "--state-dir", str(tmp_path / "benchq"),
                         "--proc-root", proc or fake_proc(tmp_path, {}),
                         "--once"] + (["--dry-run"] if dry_run else []) + list(extra))


def request(state_dir: pathlib.Path, rid: str, action: str = runner.SESSION_ACTION,
            cap_min=None) -> None:
    """What the LB leaves behind. `cap_min` left out is a request that names no duration — which is
    every request written before 2026-10-05, and what the gateway still writes when nobody picks one."""
    state_dir.mkdir(parents=True, exist_ok=True)
    body = {"id": rid, "at": time.time(), "by": "vllm-lb/gateway", "action": action}
    if cap_min is not None:
        body["cap_min"] = cap_min
    (state_dir / runner.REQUEST).write_text(json.dumps(body))


def session_report(state_dir: pathlib.Path, **fields) -> None:
    """What `bench_session.py` leaves behind as it goes."""
    state_dir.mkdir(parents=True, exist_ok=True)
    body = {"version": 2, "label": "S-20260101-000001", "pid": 900104, "claude_pid": 900109,
            "cap_min": 15.0, "started_at": time.time() - 300, "at": time.time(),
            "phase": "running", "rc": None, "ended_at": None, "detail": "a swarm is under way",
            "guards": {"environ": True, "init": True, "traffic": True},
            "served_model": "Qwen/Qwen3.8-27B", "model_choice": DENSE_27B, "generation": 104,
            "max_model_len": 262144}
    body.update(fields)
    (state_dir / bench_session.REPORT).write_text(json.dumps(body))


# ── is something running: the question that must not be answered by a substring ──


def test_a_shell_that_only_mentions_the_script_is_not_a_running_queue(tmp_path):
    """`pgrep -f` and `ps | grep` match a pattern anywhere in a command line, so both call the
    operator's watchdog a running queue — and `grep` matches the asker's own command, which is how
    `pkill`/`pgrep -f` killed a shell twice in one day. An ARGUMENT cannot be mistaken that way."""
    proc = fake_proc(tmp_path, {900101: WATCHDOG})
    assert runner.running_scripts(proc) == []
    assert runner.matrix_block(runner.running_scripts(proc))["running"] is False


def test_the_matrix_a_bench_run_and_a_swarm_are_all_found(tmp_path):
    proc = fake_proc(tmp_path, {900101: WATCHDOG, 900102: REAL_QUEUE, 900103: REAL_BENCH,
                                900104: REAL_SWARM})
    found = {p["pid"]: p["script"] for p in runner.running_scripts(proc)}
    assert found == {900102: runner.QUEUE_SCRIPT, 900103: runner.BENCH_SCRIPT,
                     900104: runner.SESSION_SCRIPT}


def test_the_queue_process_is_the_one_the_matrix_block_names(tmp_path):
    """A bench run is a child of the queue; the queue is the thing that keeps going, so it is what
    the page is told about."""
    proc = fake_proc(tmp_path, {900103: REAL_BENCH, 900102: REAL_QUEUE})
    block = runner.matrix_block(runner.running_scripts(proc))
    assert block["running"] is True and block["pid"] == 900102
    assert block["script"] == runner.QUEUE_SCRIPT and block["since"] is not None


def test_a_swarm_is_never_reported_as_the_matrix(tmp_path):
    """The page draws them in different words: one is the control it has, the other is a thing
    happening at the keyboard that it may only refuse over."""
    proc = fake_proc(tmp_path, {900104: REAL_SWARM})
    live = runner.running_scripts(proc)
    assert runner.matrix_block(live) == {"running": False, "pid": None, "since": None,
                                         "script": None}
    assert runner.session_block(live, tmp_path / "benchq")["alive"] is True


def test_the_runner_never_counts_itself_or_its_parent(tmp_path):
    """Even were this process's own command line to name the script, it is not a running queue."""
    proc = fake_proc(tmp_path, {os.getpid(): REAL_QUEUE, os.getppid(): REAL_QUEUE})
    assert runner.running_scripts(proc) == []


def test_a_process_that_ends_mid_scan_costs_that_process_not_the_scan(tmp_path):
    proc = fake_proc(tmp_path, {900102: REAL_QUEUE})
    (pathlib.Path(proc) / "900199").mkdir()          # a pid dir with no readable cmdline
    assert [p["pid"] for p in runner.running_scripts(proc)] == [900102]


# ── the session block: the process is the authority, the report is the detail ───


def test_a_live_swarm_carries_its_own_report_into_the_block(tmp_path):
    proc = fake_proc(tmp_path, {900104: REAL_SWARM})
    state = tmp_path / "benchq"
    session_report(state, pid=900104)
    block = runner.session_block(runner.running_scripts(proc), state)
    assert block["alive"] is True and block["pid"] == 900104
    assert block["phase"] == "running" and block["cap_min"] == 15.0
    assert block["label"] == "S-20260101-000001" and block["claude_pid"] == 900109
    assert block["guards"] == {"environ": True, "init": True, "traffic": True}


def test_the_block_says_which_model_the_swarm_is_on_from_the_sessions_own_report(tmp_path):
    """Since 2026-10-02 the pair can be Flash-Next or the dense 27B; the page must say which one
    these minutes are on, from what the session read off the LB when it started."""
    proc = fake_proc(tmp_path, {900104: REAL_SWARM})
    state = tmp_path / "benchq"
    session_report(state, pid=900104)
    block = runner.session_block(runner.running_scripts(proc), state)
    assert (block["served_model"], block["model_choice"]) == ("Qwen/Qwen3.8-27B", DENSE_27B)
    assert (block["generation"], block["max_model_len"]) == (104, 262144)
    # and no report at all is four honest Nones, not a missing key
    empty = runner.session_block([], tmp_path / "nowhere")
    assert {k: empty[k] for k in bench_session.IDENTITY_KEYS} == dict.fromkeys(bench_session.IDENTITY_KEYS)


def test_a_report_from_an_earlier_swarm_is_not_shown_as_the_live_one(tmp_path):
    """Between the launch and the session's first report there is a moment where the newest report
    describes the PREVIOUS run. Showing that as this one's state would date a figure by hours."""
    proc = fake_proc(tmp_path, {900104: REAL_SWARM})
    state = tmp_path / "benchq"
    session_report(state, pid=900001, phase="finished", rc=124, ended_at=time.time() - 3600)
    block = runner.session_block(runner.running_scripts(proc), state)
    assert block["alive"] is True and block["phase"] == "starting"
    assert block["rc"] is None and block["ended_at"] is None and block["label"] is None
    assert block["model_choice"] is None, "the previous run's model is not this run's"


def test_a_swarm_still_starting_is_clocked_from_its_own_process_not_the_previous_report(tmp_path):
    """The previous run's `started_at` and `cap_min` are not this run's either. Left in, the page drew a
    swarm launched seconds earlier as "Test swarm running · 743 of 15 min" (box-c, 2026-10-11 20:33Z:
    the newest report was 12.4 h old). Until the session reports, its clock is its process's own start
    and its cap is not known."""
    proc = fake_proc(tmp_path, {900104: REAL_SWARM})
    state = tmp_path / "benchq"
    session_report(state, pid=900001, phase="finished", rc=124, cap_min=30.0,
                   started_at=time.time() - 12 * 3600, ended_at=time.time() - 11 * 3600)
    live = runner.running_scripts(proc)
    block = runner.session_block(live, state)
    assert block["phase"] == "starting"
    assert block["started_at"] == live[0]["since"], "the clock is the live process's own"
    assert block["cap_min"] is None, "the previous run's cap is not this run's"
    no_report = tmp_path / "fresh"
    no_report.mkdir()
    block = runner.session_block(live, no_report)
    assert block["phase"] == "starting" and block["started_at"] == live[0]["since"]


def test_a_swarm_whose_process_is_gone_is_lost_rather_than_running(tmp_path):
    """A session killed outright leaves a report still saying `running`. The process is the
    authority: a swarm that is not there is never drawn as one that is."""
    state = tmp_path / "benchq"
    session_report(state, phase="running")
    block = runner.session_block([], state)
    assert block["alive"] is False and block["phase"] == "lost"
    assert "without reporting how it ended" in block["detail"]


def test_a_finished_swarm_keeps_its_exit_code_and_its_words(tmp_path):
    state = tmp_path / "benchq"
    session_report(state, phase="finished", rc=124, ended_at=time.time(),
                   detail="the session ran 15.0 min and was ended by its 15 min cap (exit 124)")
    block = runner.session_block([], state)
    assert block["alive"] is False and block["phase"] == "finished" and block["rc"] == 124
    assert "ended by its 15 min cap" in block["detail"]


def test_no_swarm_and_no_report_is_a_block_of_honest_nulls(tmp_path):
    block = runner.session_block([], tmp_path / "benchq")
    assert block["alive"] is False and block["phase"] is None and block["rc"] is None


# ── may a swarm start ──────────────────────────────────────────────────────────


def test_a_clear_field_lets_one_swarm_start(tmp_path):
    refusal, detail = runner.session_preflight([])
    assert refusal is None
    assert "already loaded" in detail


def test_a_second_swarm_is_refused_and_says_it_is_a_swarm(tmp_path):
    refusal, detail = runner.session_preflight([{"pid": 900104, "script": runner.SESSION_SCRIPT}])
    assert refusal == "already running"
    assert runner.SESSION_SCRIPT in detail and "900104" in detail
    assert "both measurements meaningless" in detail


@pytest.mark.parametrize("script", [runner.QUEUE_SCRIPT, runner.BENCH_SCRIPT])
def test_a_swarm_is_refused_while_the_matrix_is_up_and_says_it_is_the_matrix(tmp_path, script):
    refusal, detail = runner.session_preflight([{"pid": 900102, "script": script}])
    assert refusal == "matrix running"
    assert script in detail and "900102" in detail


def test_a_swarm_does_not_consult_the_matrixs_park(tmp_path):
    """The operator, 2026-10-01: "I should be able to click run benchmark any time". `hold` and
    `pause` gate the matrix BETWEEN its runs; a swarm is a quarter of an hour against what is
    already loaded, and nothing about that is what those files are for."""
    bench, queue = harness(tmp_path)
    for name in (runner.HOLD, runner.PAUSE):
        (queue / name).write_text("")
    assert runner.session_preflight([]) == (None, runner.session_preflight([])[1])
    assert runner.session_preflight([])[0] is None
    # and the matrix's own preflight still refuses over them, untouched
    assert runner.matrix_preflight(bench, queue, [], STORED)[0] == "parked"


# ── may the matrix start, from the keyboard only ───────────────────────────────


def test_the_matrix_preflight_names_the_run_that_would_be_taken(tmp_path):
    bench, queue = harness(tmp_path)
    refusal, detail = runner.matrix_preflight(bench, queue, [], STORED)
    assert refusal is None
    assert "P1-s16-b8k-sp4" in detail and "2 of 2 left" in detail
    assert FLASH_NEXT in detail, "the start says which model the matrix is on"


@pytest.mark.parametrize("script", [runner.QUEUE_SCRIPT, runner.BENCH_SCRIPT,
                                    runner.SESSION_SCRIPT])
def test_the_matrix_is_refused_while_anything_else_is_benchmarking(tmp_path, script):
    bench, queue = harness(tmp_path)
    refusal, detail = runner.matrix_preflight(bench, queue, [{"pid": 7, "script": script}], STORED)
    assert refusal == "already running"
    assert "nothing was started" in detail and "spoil both" in detail


# ── the matrix runs on the model its queue declares, and on no other ──────────


def test_the_matrix_is_refused_when_the_stored_model_is_not_the_queues(tmp_path):
    """The frozen bench script restarts the pair with a body that names no model_choice, so it runs
    on whatever is STORED — and its label, runs.txt and summary name no model either. Resumed with
    the 27B stored, the remaining rows would be filed under Flash-Next's labels, over Flash-Next's
    `claude/<label>-*`. The runner refuses, naming both models and what to do."""
    bench, queue = harness(tmp_path)
    stored = dict(STORED, model_choice=DENSE_27B, generation=104)
    refusal, detail = runner.matrix_preflight(bench, queue, [], stored)
    assert refusal == "model"
    assert DENSE_27B in detail and FLASH_NEXT in detail and "generation 104" in detail
    assert "cluster panel" in detail
    assert runner.model_preflight(queue, stored)[0] == "model"
    assert runner.model_preflight(queue, STORED) == (None, f"on {FLASH_NEXT}, which is the stored model_choice")


def test_the_matrix_is_refused_when_the_queue_declares_no_model(tmp_path):
    """A queue without a `# model:` line is one this runner cannot vouch for: the rows say nothing
    about a model, so there is nothing to compare the store against. Refused, with the line to add."""
    bench, queue = harness(tmp_path)
    (queue / runner.QUEUE_TXT).write_text("# LABEL SEQS BATCHED SPEC CAP_MIN\nP1-s16-b8k-sp4 16 8192 4 15\n")
    assert runner.queue_model(queue) is None
    refusal, detail = runner.matrix_preflight(bench, queue, [], STORED)
    assert refusal == "model" and f"`{runner.MODEL_LINE} <id>`" in detail


def test_the_matrix_is_refused_when_the_lb_cannot_say_what_is_stored(tmp_path):
    """No answer from the LB is not "probably fine": the bench script would restart the pair on a
    model nobody checked. `stored_model()` against a port nobody listens on says so, and the
    preflight refuses on that answer."""
    bench, queue = harness(tmp_path)
    stored = runner.stored_model("http://127.0.0.1:1", timeout=1.0)
    assert stored["model_choice"] is None and stored["error"]
    refusal, detail = runner.matrix_preflight(bench, queue, [], stored)
    assert refusal == "model" and "could not say" in detail and FLASH_NEXT in detail


def test_stored_model_reads_the_desired_model_choice_off_lb_status(tmp_path):
    """`cluster.desired.model_choice` is the STORED choice — the one a restart body without a
    model_choice keeps — not `loaded_model`, which is empty while the pair is off."""
    stub = StatusStub(DENSE_27B, generation=104)
    try:
        stored = runner.stored_model(stub.url)
    finally:
        stub.close()
    assert stored == {"model_choice": DENSE_27B, "generation": 104, "loaded": None, "state": "off",
                      "error": ""}
    assert stub.asked == ["/lb/status"], "one GET, nothing else"


def test_the_model_line_is_read_by_the_runner_and_invisible_to_the_frozen_queue(tmp_path):
    """The declaration is a `#` line. The runner reads it; the frozen queue script's own parser —
    run here verbatim, as bash — skips it and still takes the first undone row, exactly as before.
    (The real queue.txt carries the same line; see worklog/bench/claude/queue.txt.)"""
    bench, queue = harness(tmp_path, done=["P1-s16-b8k-sp4"])
    assert runner.queue_model(queue) == FLASH_NEXT
    assert runner.next_label(queue)[0] == "P1-s8-b8k-sp4", "the runner's own row reading skips it too"
    took = subprocess.run(["bash", "-c", FROZEN_PARSER], cwd=str(bench), capture_output=True,
                          text=True, timeout=30)
    assert took.returncode == 0, took.stderr
    assert took.stdout.strip() == "P1-s8-b8k-sp4 8 8192 4 15", took.stdout
    # the live queue, as committed: declared for Flash-Next, and its parser still never sees the line
    live = pathlib.Path(__file__).resolve().parents[2] / "worklog" / "bench" / "claude"
    if (live / runner.QUEUE_TXT).exists():
        assert runner.queue_model(live) == FLASH_NEXT
        for line in (live / runner.QUEUE_TXT).read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                assert len(line.split()) == 5, f"a row the queue would misread: {line!r}"


@pytest.mark.parametrize("held", [(runner.HOLD,), (runner.PAUSE,), (runner.HOLD, runner.PAUSE)])
def test_the_matrix_is_refused_while_the_gate_is_held_by_anyone(tmp_path, held):
    """`pause` is the operator's own lever, placed before an LB deploy. Starting over it — or
    clearing it — would be the runner overruling a person."""
    bench, queue = harness(tmp_path)
    for name in held:
        (queue / name).write_text("")
    refusal, detail = runner.matrix_preflight(bench, queue, [], STORED)
    assert refusal == "parked"
    for name in held:
        assert f"claude/{name}" in detail
        assert (queue / name).exists(), "a refusal must not clear the park it refused over"


def test_the_matrix_is_refused_when_the_frozen_scripts_seal_does_not_match(tmp_path):
    """The queue verifies this same seal before every run and exits 7 when it differs. Saying so
    here beats launching a queue that dies a second later."""
    bench, queue = harness(tmp_path, seal=False)
    refusal, detail = runner.matrix_preflight(bench, queue, [], STORED)
    assert refusal == "seal"
    assert runner.BENCH_SCRIPT in detail and "frozen script was changed" in detail


def test_the_matrix_is_refused_when_every_label_is_already_done(tmp_path):
    """Clearing done.txt to "fix" this would destroy a night's bookkeeping, so it says what to do
    instead of doing it."""
    bench, queue = harness(tmp_path, done=["P1-s16-b8k-sp4", "P1-s8-b8k-sp4"])
    refusal, detail = runner.matrix_preflight(bench, queue, [], STORED)
    assert refusal == "finished"
    assert "clear done.txt at the keyboard" in detail


# ── one tick ───────────────────────────────────────────────────────────────────


def test_the_report_is_written_every_tick_so_the_lb_can_tell_a_runner_is_listening(tmp_path):
    """Its `at` is the whole heartbeat: a request left for a runner that is not there has to read as
    a request that cannot be served, never as a swarm about to happen."""
    bench, queue = harness(tmp_path)
    opts = options(tmp_path, bench, queue)
    runner.tick(opts, None)
    out = json.loads((tmp_path / "benchq" / runner.REPORT).read_text())
    assert out["matrix"] == {"running": False, "pid": None, "since": None, "script": None}
    assert out["session"]["alive"] is False and out["session"]["phase"] is None
    assert out["last"] is None and abs(out["at"] - time.time()) < 30
    assert out["pid"] == os.getpid() and out["version"] == runner.REPORT_VERSION
    assert out["cap_min"] == bench_session.CAP_MIN, "the page names the cap from the runner's own"


def test_the_report_tells_a_swarm_and_the_matrix_apart(tmp_path):
    bench, queue = harness(tmp_path)
    opts = options(tmp_path, bench, queue,
                   proc=fake_proc(tmp_path, {900102: REAL_QUEUE, 900104: REAL_SWARM}))
    session_report(tmp_path / "benchq", pid=900104)
    runner.tick(opts, None)
    out = json.loads((tmp_path / "benchq" / runner.REPORT).read_text())
    assert out["matrix"]["pid"] == 900102 and out["matrix"]["script"] == runner.QUEUE_SCRIPT
    assert out["session"]["pid"] == 900104 and out["session"]["phase"] == "running"


def test_a_request_is_answered_once_and_a_second_tick_does_not_repeat_it(tmp_path):
    bench, queue = harness(tmp_path)
    opts = options(tmp_path, bench, queue, dry_run=True)
    request(tmp_path / "benchq", "abc123")
    first = runner.tick(opts, None)
    assert first["result"] == "dry-run" and first["id"] == "abc123"
    again = runner.tick(opts, first)
    assert again is first, "the same request must not be acted on twice"


def test_a_request_for_the_matrix_is_refused_by_name(tmp_path):
    """A `start.request` written by the OLD gateway asks for `start`. After this deploy that must not
    start a thirteen-hour matrix: it is refused, and the refusal says where the matrix now lives."""
    bench, queue = harness(tmp_path)
    opts = options(tmp_path, bench, queue)
    request(tmp_path / "benchq", "abc123", action="start")
    last = runner.tick(opts, None)
    assert last["result"] == "not mine"
    assert "--start-matrix" in last["detail"] and "never from the page" in last["detail"]
    assert not (bench / runner.QUEUE_LOG).exists(), "nothing was launched"


def test_a_restarted_runner_does_not_re_run_the_request_it_already_answered(tmp_path):
    """`main` reads `last` back out of its own report; starting empty would re-launch on reboot."""
    bench, queue = harness(tmp_path)
    state = tmp_path / "benchq"
    request(state, "abc123")
    argv = ["--bench-dir", str(bench), "--queue-dir", str(queue), "--state-dir", str(state),
            "--proc-root", fake_proc(tmp_path, {}), "--once", "--dry-run"]
    runner.main(argv)
    answered = json.loads((state / runner.REPORT).read_text())["last"]
    assert answered["result"] == "dry-run"
    runner.main(argv)
    assert json.loads((state / runner.REPORT).read_text())["last"]["at"] == answered["at"]


def test_a_request_that_arrives_while_a_swarm_is_up_is_answered_with_the_refusal(tmp_path):
    bench, queue = harness(tmp_path)
    opts = options(tmp_path, bench, queue, proc=fake_proc(tmp_path, {900104: REAL_SWARM}))
    request(tmp_path / "benchq", "abc123")
    last = runner.tick(opts, None)
    assert last["result"] == "already running" and "900104" in last["detail"]
    assert json.loads((tmp_path / "benchq" / runner.REPORT).read_text())["session"]["pid"] == 900104


def test_a_request_that_arrives_while_the_matrix_is_up_is_told_it_is_the_matrix(tmp_path):
    bench, queue = harness(tmp_path)
    opts = options(tmp_path, bench, queue, proc=fake_proc(tmp_path, {900102: REAL_QUEUE}))
    request(tmp_path / "benchq", "abc123")
    last = runner.tick(opts, None)
    assert last["result"] == "matrix running" and runner.QUEUE_SCRIPT in last["detail"]


def test_the_runner_writes_nothing_in_the_harnesss_directories_when_it_serves_a_swarm(tmp_path,
                                                                                     monkeypatch):
    """The frozen script and its seal above all — and `claude/queue.txt`, `done.txt`, `runs.txt`,
    which are the matrix's bookkeeping. A swarm marks nothing done and banks no line."""
    bench, queue = harness(tmp_path)
    (queue / "runs.txt").write_text("P1-s16-b8k-sp4 16 8192 4 1 2 0\n")
    before = {p.name: p.read_bytes() for p in list(bench.iterdir()) + list(queue.iterdir())
              if p.is_file()}
    stub = tmp_path / "stub_session.py"
    stub.write_text("import pathlib,sys\npathlib.Path(sys.argv[-1]).write_text('ran')\n")
    monkeypatch.setattr(runner, "SESSION_PY", stub)
    opts = options(tmp_path, bench, queue, extra=["--metrics-url", str(tmp_path / "marker")])
    request(tmp_path / "benchq", "abc123")
    last = runner.tick(opts, None)
    assert last["result"] == "started"
    after = {p.name: p.read_bytes() for p in list(bench.iterdir()) + list(queue.iterdir())
             if p.is_file()}
    assert after == before


def test_a_dry_run_launches_nothing(tmp_path):
    bench, queue = harness(tmp_path)
    opts = options(tmp_path, bench, queue, dry_run=True)
    request(tmp_path / "benchq", "abc123")
    last = runner.tick(opts, None)
    assert last["result"] == "dry-run"
    assert not (tmp_path / "benchq" / runner.SESSION_LOG).exists()


# ── the launches themselves, against stub scripts ──────────────────────────────


def test_a_real_launch_starts_the_single_session_launcher_with_everything_on_its_argv(tmp_path,
                                                                                     monkeypatch):
    """`launch_session()` for real, against a stub that records the command line it was given. What
    is running has to say what it is doing: the cap, the LB, the engine and the model are arguments,
    not defaults this runner keeps a second copy of."""
    bench, queue = harness(tmp_path)
    stub = tmp_path / "stub_session.py"
    stub.write_text("import json,pathlib,sys\n"
                    "pathlib.Path(sys.argv[0]).with_name('argv.json').write_text(json.dumps(sys.argv))\n"
                    "print('STUB SESSION RAN')\n")
    monkeypatch.setattr(runner, "SESSION_PY", stub)
    opts = options(tmp_path, bench, queue, extra=["--cap-min", "7", "--lb-url", "http://lb:8010",
                                                  "--metrics-url", "http://head:8100/metrics",
                                                  "--model", "qwen3.8:latest"])
    request(tmp_path / "benchq", "abc123")
    last = runner.tick(opts, None)
    assert last["result"] == "started" and last["label"].startswith("S-")
    assert "7 min" in last["detail"] and "already loaded" in last["detail"]
    argv_path = tmp_path / "argv.json"
    for _ in range(100):
        if argv_path.exists():
            break
        time.sleep(0.05)
    argv = json.loads(argv_path.read_text())
    assert argv[1:] == ["--label", last["label"], "--cap-min", "7",
                        "--state-dir", str(tmp_path / "benchq"), "--lb-url", "http://lb:8010",
                        "--metrics-url", "http://head:8100/metrics", "--model", "qwen3.8:latest"]
    log = tmp_path / "benchq" / runner.SESSION_LOG
    for _ in range(100):
        if log.exists() and "STUB SESSION RAN" in log.read_text():
            break
        time.sleep(0.05)
    assert "STUB SESSION RAN" in log.read_text(), "a swarm's own output is kept"


def test_start_matrix_is_the_only_way_the_thirty_run_matrix_begins(tmp_path, lb_stub):
    """`--start-matrix` at the keyboard, with the preflight that used to sit behind the button. The
    stub queue script writes a marker and exits; nothing of the operator's harness is touched."""
    bench, queue = harness(tmp_path)
    (bench / runner.QUEUE_SCRIPT).write_text("#!/bin/bash\necho STUB QUEUE RAN\ntouch ./it-ran\n")
    (bench / runner.QUEUE_LOG).write_text("an earlier night\n")
    code = runner.main(["--bench-dir", str(bench), "--queue-dir", str(queue),
                        "--state-dir", str(tmp_path / "benchq"), "--lb-url", lb_stub.url,
                        "--proc-root", fake_proc(tmp_path, {}), "--start-matrix"])
    assert code == 0
    assert lb_stub.asked == ["/lb/status"], "the stored model was checked, with one GET"
    for _ in range(100):
        if (bench / "it-ran").exists():
            break
        time.sleep(0.05)
    assert (bench / "it-ran").exists(), "the stub queue never ran"
    for _ in range(100):
        if "STUB QUEUE RAN" in (bench / runner.QUEUE_LOG).read_text():
            break
        time.sleep(0.05)
    log = (bench / runner.QUEUE_LOG).read_text()
    assert log.startswith("an earlier night\n"), "the night's log must be appended to, never cut"
    assert "STUB QUEUE RAN" in log


def test_start_matrix_refuses_and_exits_non_zero_rather_than_overruling_a_park(tmp_path, capsys,
                                                                             lb_stub):
    bench, queue = harness(tmp_path)
    (queue / runner.PAUSE).write_text("")
    code = runner.main(["--bench-dir", str(bench), "--queue-dir", str(queue),
                        "--state-dir", str(tmp_path / "benchq"), "--lb-url", lb_stub.url,
                        "--proc-root", fake_proc(tmp_path, {}), "--start-matrix"])
    assert code == 2
    out = capsys.readouterr().out
    assert "NOT STARTED (parked)" in out and "claude/pause" in out
    assert not (bench / runner.QUEUE_LOG).exists()


def test_start_matrix_refuses_and_launches_nothing_when_the_wrong_model_is_stored(tmp_path, capsys):
    """End to end at the keyboard: the 27B stored, Flash-Next's queue — exit 2, the refusal naming
    both, and the frozen queue script never started (no log, no marker)."""
    bench, queue = harness(tmp_path)
    (bench / runner.QUEUE_SCRIPT).write_text("#!/bin/bash\ntouch ./it-ran\n")
    stub = StatusStub(DENSE_27B, generation=104)
    try:
        code = runner.main(["--bench-dir", str(bench), "--queue-dir", str(queue),
                            "--state-dir", str(tmp_path / "benchq"), "--lb-url", stub.url,
                            "--proc-root", fake_proc(tmp_path, {}), "--start-matrix"])
    finally:
        stub.close()
    assert code == 2
    out = capsys.readouterr().out
    assert "NOT STARTED (model)" in out and DENSE_27B in out and FLASH_NEXT in out
    time.sleep(0.3)
    assert not (bench / "it-ran").exists() and not (bench / runner.QUEUE_LOG).exists()


# ── the duration the request names, since 2026-10-05 ──────────────────────────
# `--cap-min` is still the runner's own default and still 15. A request may name another, within
# `bench_session.CAP_MIN_RANGE`, and this runner checks it AGAIN: the LB's check did not run on a
# request file left behind by an older build or dropped in by hand.


def test_a_request_that_names_its_own_duration_is_launched_with_exactly_it(tmp_path, monkeypatch):
    bench, queue = harness(tmp_path)
    stub = tmp_path / "stub_session.py"
    stub.write_text("import json,pathlib,sys\n"
                    "pathlib.Path(sys.argv[0]).with_name('argv.json').write_text(json.dumps(sys.argv))\n")
    monkeypatch.setattr(runner, "SESSION_PY", stub)
    opts = options(tmp_path, bench, queue)                       # --cap-min left at its default, 15
    request(tmp_path / "benchq", "abc123", cap_min=30)
    last = runner.tick(opts, None)
    assert last["result"] == "started" and last["cap_min"] == 30.0
    assert "30 min" in last["detail"]
    argv_path = tmp_path / "argv.json"
    for _ in range(100):
        if argv_path.exists():
            break
        time.sleep(0.05)
    argv = json.loads(argv_path.read_text())
    assert "--cap-min" in argv and argv[argv.index("--cap-min") + 1] == "30", argv


def test_a_request_that_names_no_duration_still_runs_for_the_runners_own(tmp_path):
    """The default, from the only place it lives on this side: `--cap-min`. A request from an LB that
    never heard of the field has to behave exactly as it did before the field existed."""
    bench, queue = harness(tmp_path)
    opts = options(tmp_path, bench, queue, dry_run=True)
    request(tmp_path / "benchq", "abc123")
    last = runner.tick(opts, None)
    assert last["result"] == "dry-run" and last["cap_min"] == bench_session.CAP_MIN == 15.0
    assert "15 min" in last["detail"] and "runner's own default" in last["detail"]


def _request_with_cap(state: pathlib.Path, asked) -> None:
    """A request whose `cap_min` is written WHATEVER it is — including JSON's null and a string, which
    `request()` above cannot express because it leaves the field out for None."""
    state.mkdir(parents=True, exist_ok=True)
    (state / runner.REQUEST).write_text(json.dumps({"id": "abc123", "at": time.time(),
                                                    "by": "vllm-lb/gateway",
                                                    "action": runner.SESSION_ACTION, "cap_min": asked}))


@pytest.mark.parametrize("asked", [4, 0, 241, -30, "30", True])
def test_a_request_whose_duration_is_out_of_range_is_refused_and_launches_nothing(tmp_path, asked):
    """REFUSED by name, not quietly run at fifteen minutes: a swarm whose minutes are not the minutes
    that were asked for is a measurement nobody can read back. `True` is in the list because
    `isinstance(True, int)` is true in Python, so a cap of `true` is one minute to a careless check."""
    bench, queue = harness(tmp_path)
    opts = options(tmp_path, bench, queue)        # NOT a dry run: a refusal must launch nothing for real
    state = tmp_path / "benchq"
    _request_with_cap(state, asked)
    last = runner.tick(opts, None)
    assert last["result"] == "bad cap", last
    assert "5" in last["detail"] and "240" in last["detail"]
    assert not (state / runner.SESSION_LOG).exists(), "a refused duration still launched something"


def test_a_duration_written_as_null_is_the_same_as_none_at_all(tmp_path):
    """The LB sends no field when nobody picks a duration, but a hand-written request may say `null`.
    Both mean the runner's own default; neither is a refusal."""
    bench, queue = harness(tmp_path)
    opts = options(tmp_path, bench, queue, dry_run=True)
    _request_with_cap(tmp_path / "benchq", None)
    last = runner.tick(opts, None)
    assert last["result"] == "dry-run" and last["cap_min"] == bench_session.CAP_MIN


# ── the sealed runner (lb/bench_runner), since 2026-10-11 ──────────────────────
# The same runner, run inside a container that has no route out but its own egress proxy. Four things
# are new and every one of them is OFF unless asked for, so the host unit's runner is byte-for-byte
# the runner it was: where a session keeps its answers (`--results-dir`), the proxy its claude fetches
# through (`--https-proxy`, `--no-proxy`), which box this runner is on (`--box`, for the page's words —
# never a literal), and the check that the network is still sealed before every launch (`--seal-check`).


def _stub_recording_argv(tmp_path, monkeypatch) -> pathlib.Path:
    stub = tmp_path / "stub_session.py"
    stub.write_text("import json,pathlib,sys\n"
                    "pathlib.Path(sys.argv[0]).with_name('argv.json').write_text(json.dumps(sys.argv))\n")
    monkeypatch.setattr(runner, "SESSION_PY", stub)
    return tmp_path / "argv.json"


def _wait_for(path: pathlib.Path) -> None:
    for _ in range(200):
        if path.exists():
            return
        time.sleep(0.05)
    raise AssertionError(f"{path} never appeared")


def test_the_sealed_runners_session_flags_reach_the_launched_session(tmp_path, monkeypatch):
    """`--results-dir`, `--https-proxy` and `--no-proxy` go onto the session's argv as they were given:
    the container's paths and its proxy are the runner's configuration, never bench_session's defaults."""
    bench, queue = harness(tmp_path)
    argv_path = _stub_recording_argv(tmp_path, monkeypatch)
    opts = options(tmp_path, bench, queue, extra=[
        "--lb-url", "http://lb.test:8010", "--metrics-url", "http://engine.test:8000/metrics",
        "--results-dir", str(tmp_path / "research"), "--https-proxy", "http://egress.test:3128",
        "--no-proxy", "lb.test,engine.test"])
    request(tmp_path / "benchq", "abc123")
    assert runner.tick(opts, None)["result"] == "started"
    _wait_for(argv_path)
    argv = json.loads(argv_path.read_text())
    for flag, value in (("--results-dir", str(tmp_path / "research")),
                        ("--https-proxy", "http://egress.test:3128"),
                        ("--no-proxy", "lb.test,engine.test")):
        assert flag in argv and argv[argv.index(flag) + 1] == value, argv


def test_without_the_sealed_flags_the_session_argv_is_what_the_host_unit_always_launched(tmp_path,
                                                                                        monkeypatch):
    bench, queue = harness(tmp_path)
    argv_path = _stub_recording_argv(tmp_path, monkeypatch)
    opts = options(tmp_path, bench, queue)
    request(tmp_path / "benchq", "abc123")
    assert runner.tick(opts, None)["result"] == "started"
    _wait_for(argv_path)
    argv = json.loads(argv_path.read_text())
    assert not {"--results-dir", "--https-proxy", "--no-proxy"} & set(argv), argv


def test_the_report_names_the_box_the_runner_was_configured_with_and_none_otherwise(tmp_path):
    bench, queue = harness(tmp_path)
    runner.tick(options(tmp_path, bench, queue, extra=["--box", "box-under-test"]), None)
    assert json.loads((tmp_path / "benchq" / runner.REPORT).read_text())["box"] == "box-under-test"
    runner.tick(options(tmp_path, bench, queue), None)
    assert json.loads((tmp_path / "benchq" / runner.REPORT).read_text())["box"] is None


def test_the_matrix_refusal_names_the_configured_box_and_never_a_literal_one(tmp_path):
    live = [{"pid": 900102, "script": runner.QUEUE_SCRIPT}]
    refusal, detail = runner.session_preflight(live, box="box-under-test")
    assert refusal == "matrix running" and "on box-under-test" in detail
    refusal, detail = runner.session_preflight(live)
    assert refusal == "matrix running" and " on " not in detail.split(":")[0], detail
    assert "box-c" not in detail


def _seal_check(tmp_path, body: str) -> pathlib.Path:
    check = tmp_path / "seal_check.sh"
    check.write_text("#!/bin/sh\n" + body)
    check.chmod(0o755)
    return check


def test_a_failed_seal_check_refuses_the_request_and_launches_nothing(tmp_path, monkeypatch):
    """The container's own proof that it cannot reach Anthropic, run again before EVERY session: a
    non-zero answer is a refusal in the check's own words, and claude is never started."""
    bench, queue = harness(tmp_path)
    argv_path = _stub_recording_argv(tmp_path, monkeypatch)
    check = _seal_check(tmp_path, "echo 'api.anthropic.com:443 direct  CONNECTED  <- must fail'\nexit 1\n")
    opts = options(tmp_path, bench, queue, extra=["--seal-check", str(check)])
    request(tmp_path / "benchq", "abc123")
    last = runner.tick(opts, None)
    assert last["result"] == "unsealed", last
    assert "api.anthropic.com" in last["detail"]
    time.sleep(0.3)
    assert not argv_path.exists(), "an unsealed network launched a session"
    assert not (tmp_path / "benchq" / runner.SESSION_LOG).exists()


def test_a_seal_check_that_cannot_run_is_a_refusal_too(tmp_path, monkeypatch):
    bench, queue = harness(tmp_path)
    argv_path = _stub_recording_argv(tmp_path, monkeypatch)
    opts = options(tmp_path, bench, queue, extra=["--seal-check", str(tmp_path / "no-such-check")])
    request(tmp_path / "benchq", "abc123")
    last = runner.tick(opts, None)
    assert last["result"] == "unsealed" and "no-such-check" in last["detail"]
    time.sleep(0.3)
    assert not argv_path.exists()


def test_a_passed_seal_check_lets_the_session_start(tmp_path, monkeypatch):
    bench, queue = harness(tmp_path)
    argv_path = _stub_recording_argv(tmp_path, monkeypatch)
    check = _seal_check(tmp_path, "echo sealed\nexit 0\n")
    opts = options(tmp_path, bench, queue, extra=["--seal-check", str(check)])
    request(tmp_path / "benchq", "abc123")
    assert runner.tick(opts, None)["result"] == "started"
    _wait_for(argv_path)


def test_the_seal_is_checked_only_after_the_field_is_clear(tmp_path):
    """A swarm already up is answered as that — the seal check is not run for a request that was never
    going to launch anything."""
    bench, queue = harness(tmp_path)
    marker = tmp_path / "seal-was-checked"
    check = _seal_check(tmp_path, f"touch {marker}\nexit 0\n")
    opts = options(tmp_path, bench, queue, proc=fake_proc(tmp_path, {900104: REAL_SWARM}),
                   extra=["--seal-check", str(check)])
    request(tmp_path / "benchq", "abc123")
    assert runner.tick(opts, None)["result"] == "already running"
    assert not marker.exists()
