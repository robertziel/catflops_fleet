"""THE START'S STALL CLOCK RUNS ON THE HEAD'S OWN LOG, so an LB restarted in the middle of a hang does not start the
silence over.

THE FAILURE, OBSERVED 2026-10-03. Generation 134's last progress line was printed at 07:21:13 (both ranks "Capture
draft verify CUDA graph begin"); a redeployed LB came up at 07:23:14, and at 07:25:29 the panel read "no progress for
131 s" — the LB had measured the silence from when IT first read that line (~07:23:18), not from when the head printed
it, so the rescue at the 180 s capture limit (065668a) would have fired ~2 minutes late. The silence is now measured
from the time the head printed the line that last moved its start: SGLang's `[2026-10-03 07:21:13 TP0]`, vLLM's
`INFO 10-02 20:46:36 [file.py:N]`, docker's own `--timestamps` prefix — on the BOX's clock, and against the box's own
`date` (tp-rank's status `net.t`) so a box whose clock runs behind is not declared hung early. A line without a time
(a tqdm bar) keeps the old rule: the silence counts from the read that first saw it.
"""
from __future__ import annotations

import calendar
import time

import pytest

from tests.test_cluster_loop import World, clock_of, writes  # noqa: F401 — `writes` is a fixture
from tests.test_start_rescue import CAPTURING, arm, gen131_at, reboot, replay, rescued, sglang_launched, whole
from vllm_lb.cluster_startup import StartupPhase, read_startup

#: 2026-10-03 00:00:00 UTC — generation 131's day, the day every replay below runs on.
DAY = calendar.timegm((2026, 10, 3, 0, 0, 0))
#: When docker started generation 131's head: its first line is stamped 06:26:19.
HEAD_STARTED = "2026-10-03T06:26:12.493118327Z"
#: The draft-verify capture that never ended: the last line that moved generation 131's start.
LAST_MOVE = "06:27:57"


def at(w: World, when: str) -> None:
    """Set the LB's clock (which is the wall clock: UTC, 2026-10-03) to `when`."""
    w.clock.t = DAY + clock_of(when)


@pytest.fixture
def world(broker_db, writes, tmp_path):  # noqa: F811 — `writes` is the fixture imported above
    w = World(tmp_path, writes)
    at(w, "06:25:30")
    w.boot()                                       # an LB whose clock reads 2026-10-03, as the real one does
    arm(w)
    w.executor.started_at = {"box-a": HEAD_STARTED, "box-b": "2026-10-03T06:26:12.871Z"}
    return w


def launched(w: World) -> int:
    gen = sglang_launched(w)
    at(w, "06:26:15")
    return gen


def quiet_said(v: dict) -> int | None:
    """The silence the panel states ("no progress for 123 s"), or None while it states none."""
    words = v["detail"].split("no progress for ", 1)
    return int(words[1].split(" s", 1)[0]) if len(words) == 2 else None


# ── the head's own times, read off its log ────────────────────────────────────

def test_the_line_that_last_moved_the_start_says_when_the_head_printed_it():
    p = read_startup(gen131_at("06:31:30"), now=DAY + clock_of("06:31:30"))
    assert p.phase == CAPTURING
    assert p.printed_at == DAY + clock_of(LAST_MOVE), "SGLang's `[2026-10-03 06:27:57 TP0] Capture draft verify …`"
    # The torch warnings after it (`[rank0]:W1003 06:28:00.854000 …`) move nothing, and so are not its time.


def test_vllm_stamps_carry_no_year_and_take_the_readers():
    log = whole("qwen38-dflash2-start-2026-10-02-rank0.log")
    cut = log[:log.index("Application startup complete")]           # uvicorn's own line carries no stamp
    now = calendar.timegm((2026, 10, 2, 21, 0, 0))
    p = read_startup(cut, now=now)
    assert p.phase == StartupPhase.STARTING_API_SERVER
    assert p.printed_at == calendar.timegm((2026, 10, 2, 20, 46, 38)), "`INFO 10-02 20:46:38 [core.py:361] init engine`"
    # Read on New Year's morning, a December line is last year's — never a line from the future.
    december = "(EngineCore pid=1) INFO 12-31 23:59:58 [core.py:361] init engine (profile, create kv cache, " \
               "warmup model) took 136.63 s\n"
    assert read_startup(december, now=calendar.timegm((2027, 1, 1, 0, 0, 5))).printed_at == \
        calendar.timegm((2026, 12, 31, 23, 59, 58))


def test_a_line_without_a_time_gives_the_start_no_time():
    """A LoadPass counting is a tqdm bar: no stamp of its own, so the time of the line BEFORE it would be a lie —
    the silence is then counted from the read that first saw it, as before."""
    log = whole("qwen38-dflash2-start-2026-10-02-rank0.log")
    counting = log[:log.index("Loading safetensors checkpoint shards: 100%")]
    p = read_startup(counting, now=calendar.timegm((2026, 10, 2, 21, 0, 0)))
    assert p.counting and p.printed_at is None
    assert read_startup(whole("qwen38-dflash2-start-2026-10-02-rank0.log"),
                        now=time.time()).printed_at is None, "uvicorn's startup line has no stamp"


def test_dockers_own_timestamps_are_read_where_the_log_has_them():
    log = whole("qwen38-start-2026-09-18-hung-rank0.log")
    p = read_startup(log, now=calendar.timegm((2026, 9, 19, 0, 0, 0)))
    assert p is not None and p.printed_at is not None
    assert calendar.timegm((2026, 9, 18, 0, 0, 0)) < p.printed_at < calendar.timegm((2026, 9, 19, 0, 0, 0))


# ── the loop: the silence survives the LB ─────────────────────────────────────

def test_an_lb_restarted_mid_hang_measures_the_silence_from_the_heads_last_line(world):
    """Generation 134's case on generation 131's log: the old LB read the capture begin, was redeployed, and the new
    one came up a minute later. The panel states the head's silence, and the rescue fires at the capture limit
    (START_STALL_FLOOR_S, 400 s) after the head's line — not 400 s after the new LB's first read."""
    launched(world)
    timeline = replay(world, "06:26:15", "06:29:00", until=rescued)
    assert not rescued(timeline[-1][1]), "nothing is hung 63 s into a capture"
    reboot(world)
    at(world, "06:30:00")                                        # the LB was down for a minute
    timeline = replay(world, "06:30:00", "06:40:00", until=rescued)
    starting = [(t, v) for t, v in timeline if v["state"] == "starting" and quiet_said(v) is not None]
    assert starting, [v["detail"] for _, v in timeline]
    t, v = starting[0]
    assert quiet_said(v) >= t - clock_of(LAST_MOVE) - 15, \
        f"at {t}: {v['detail']} — the silence is the head's, not the new LB's"
    at_, v = timeline[-1]
    assert rescued(v), f"never rescued: {v['state']} — {v['detail']}"
    # +30: the next 15 s read, and the plan that weighs it one tick later.
    assert at_ <= clock_of(LAST_MOVE) + 400 + 30, \
        f"rescued at {at_ - clock_of(LAST_MOVE)} s after the head's last line; the limit is 400 s"


def test_a_box_whose_clock_runs_behind_is_read_on_its_own_clock(world):
    """box-a's clock 90 s behind the LB's: its lines are stamped 90 s "early". Measured against the box's own `date`
    (tp-rank's `net.t`), the silence is still the head's — never 90 s longer, which would rescue a healthy capture."""
    behind = 90
    world.executor.box_clock = {b: (lambda: world.clock.t - behind) for b in ("box-a", "box-b")}

    def boxed(when: str) -> str:
        return gen131_at(time.strftime("%H:%M:%S", time.gmtime(clock_of(when) - behind)))

    launched(world)
    replay(world, "06:26:15", "06:29:00", log=boxed, until=rescued)
    reboot(world)
    at(world, "06:30:00")
    timeline = replay(world, "06:30:00", "06:40:00", log=boxed, until=rescued)
    at_, v = timeline[-1]
    assert rescued(v), f"never rescued: {v['state']} — {v['detail']}"
    printed = clock_of(LAST_MOVE) + behind                        # the head's line, on the LB's clock
    assert printed + 400 <= at_ <= printed + 400 + 30, at_ - printed


def test_stamps_the_head_cannot_have_printed_are_no_evidence(world):
    """A log whose stamps fall before its container started — another time zone, another container's — is not the
    start's clock: the silence is counted from the LB's own first read, as it was before."""
    world.executor.started_at = {"box-a": "2026-10-03T08:26:12.0Z", "box-b": "2026-10-03T08:26:12.0Z"}
    launched(world)
    replay(world, "06:26:15", "06:29:00", until=rescued)
    reboot(world)
    at(world, "06:30:00")
    timeline = replay(world, "06:30:00", "06:40:00", until=rescued)
    at_, v = timeline[-1]
    assert rescued(v), f"never rescued: {v['state']} — {v['detail']}"
    assert at_ >= clock_of("06:30:00") + 400, "measured from the restarted LB's first read"


def test_with_no_start_time_reported_the_old_rule_stands(world):
    world.executor.started_at = {"box-a": None, "box-b": None}
    launched(world)
    replay(world, "06:26:15", "06:29:00", until=rescued)
    reboot(world)
    at(world, "06:30:00")
    timeline = replay(world, "06:30:00", "06:40:00", until=rescued)
    assert rescued(timeline[-1][1]) and timeline[-1][0] >= clock_of("06:30:00") + 400
