"""THE CUDA-GRAPH CAPTURE STEP, and the rule it broke: a StartupPhase narrower than one log poll is
never reported, so a step is named from the line that OPENS it, never from its terminator alone.

The LB does not follow the head's log. `cluster_loop._track` re-reads a tail of it once per
`Config.cluster_logs_every_s` (15.0 s, config.py:307) and keeps no cursor; `read_startup` reports the
LAST recognised line in that tail as the phase. So a phase whose line is superseded by the next step's
line sooner than one poll is read and then discarded — unreportable, however right its pattern is.

Graph capture was the worst case in the whole start, because `_PHASE_LINES` carried only its TERMINATOR
("Graph capturing finished") and no entry for the line vLLM prints when it begins. On the 2026-10-02
generation-109 start (fixtures/qwen38-dflash2-start-2026-10-02-rank0.log, read off box-a before it went
down at 21:03Z) that made the step reportable for 2 seconds out of the 39 it ran:

    [autotune] INFO 10-02 20:45:57 … Autotuning process ends
    [capture ] (Worker_TP0 pid=289) \rCapturing CUDA graphs (PIECEWISE): 0%| | 0/19 …   <- 39 s, UNNAMED
    [capture ] (Worker_TP0 pid=289) INFO 10-02 20:46:36 … Capturing model for DFlash2 speculator...
    [capture ] (Worker_TP0 pid=289) \rCapturing dflash2 CUDA graphs (FULL): … 8/8
    [done    ] INFO 10-02 20:46:36 [model_runner.py:960] Graph capturing finished in 39 secs, took 1.56 GiB
    [next    ] INFO 10-02 20:46:38 [jit_monitor.py:…] init engine (profile, create kv cache, warmup …) took

20:46:36 − 20:45:57 = 39 s, which is exactly the engine's own "finished in 39 secs": the bars filled the
whole gap, so the step's OPENING line is at 20:45:57 and its window is the step. The terminator's window
is 20:46:38 − 20:46:36 = 2 s. Against a 15 s poll the first cannot be missed and the second almost always
is — which is why `boots-*.jsonl` recorded "CUDA graphs captured" in 9 of the 43 boots that recorded any
phase at all, on every model, while "loading model weights" (minutes wide) is in 43 of 43.

The drafter is NOT the cause (the brief's hypothesis): the engine captures the target's graphs AND the
drafter's inside ONE step and totals them in one line ("in 39 secs, took 1.56 GiB"), so the drafter's
bars are named as that same step, not as a second one.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

import pytest

from tests.start_logs import DFLASH2_HEAD, body_lines, grown_to
from vllm_lb.cluster_startup import LoadPass, StartupPhase, read_startup
from vllm_lb.config import Config

HEAD = body_lines(DFLASH2_HEAD)

#: vLLM's own stamp on an INFO line: "INFO 10-02 20:46:36". The tqdm capture bars carry none.
_STAMP = re.compile(r"\b(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d)\b")


def when(line: str) -> float | None:
    """The engine's own time for `line`, or None for a line it did not stamp (every tqdm bar)."""
    m = _STAMP.search(line)
    if m is None:
        return None
    month, day, hh, mm, ss = (int(g) for g in m.groups())
    return datetime(2026, month, day, hh, mm, ss, tzinfo=timezone.utc).timestamp()


def line_with(needle: str, start: int = 0) -> int:
    """The first line at or after `start` carrying `needle` — an assertion, so a fixture that stopped
    containing the line fails loudly instead of silently proving nothing."""
    for i in range(start, len(HEAD)):
        if needle in HEAD[i]:
            return i
    raise AssertionError(f"{needle!r} is not in {DFLASH2_HEAD}")


def phase_at(last: int) -> str | None:
    """The StartupPhase the LB would report from a poll that read the log through line `last`."""
    got = read_startup(grown_to(HEAD, last))
    return None if got is None else got.phase


AUTOTUNE_ENDS = line_with("Autotuning process ends")
CAPTURE_BEGINS = line_with("Capturing CUDA graphs")
SPECULATOR = line_with("Capturing model for DFlash2 speculator")
DRAFTER_BARS = line_with("Capturing dflash2 CUDA graphs")
CAPTURE_ENDS = line_with("Graph capturing finished")
AFTER_CAPTURE = line_with("init engine (profile, create kv cache, warmup model) took")

#: Every log a poll could have read while the graphs were being captured: the whole capture block, from
#: its first bar to the line before the engine declared it finished.
DURING_CAPTURE = list(range(CAPTURE_BEGINS, CAPTURE_ENDS))


def test_the_fixture_is_the_real_generation_109_start_in_the_order_the_head_printed_it():
    assert AUTOTUNE_ENDS < CAPTURE_BEGINS <= SPECULATOR <= DRAFTER_BARS < CAPTURE_ENDS < AFTER_CAPTURE
    assert "Graph capturing finished in 39 secs, took 1.56 GiB" in HEAD[CAPTURE_ENDS]
    assert when(HEAD[CAPTURE_BEGINS]) is None, "a tqdm bar carries no stamp of its own"


def test_the_capture_filled_the_whole_gap_the_engine_timed_itself():
    """Why the step's opening line is at `Autotuning process ends`'s own second, not an estimate: the gap
    between the two stamped lines IS the duration the engine printed for the step between them."""
    ran = when(HEAD[CAPTURE_ENDS]) - when(HEAD[AUTOTUNE_ENDS])
    engine_said = int(re.search(r"finished in (\d+) secs", HEAD[CAPTURE_ENDS]).group(1))
    assert (ran, engine_said) == (39.0, 39)


def test_the_step_is_reportable_for_longer_than_one_log_poll():
    """THE MECHANISM. The LB samples the log every `cluster_logs_every_s` with no cursor, so a step is
    reportable only for as long as its own line is the last recognised one. Matched by its terminator
    alone that was 2 s of the 39 the step ran — narrower than one poll, so 39 of 48 boots never recorded
    it. Matched by its opening line too, the window is the step."""
    poll = Config.cluster_logs_every_s
    terminator_only = when(HEAD[AFTER_CAPTURE]) - when(HEAD[CAPTURE_ENDS])
    assert terminator_only == 2.0 and terminator_only < poll          # the window that lost the step

    named = [i for i in DURING_CAPTURE if phase_at(i) not in (None, StartupPhase.AUTOTUNE_DONE)]
    assert named, "no poll during the capture names the step: it is reportable for 2 s against a 15 s poll"
    window = when(HEAD[CAPTURE_ENDS]) - when(HEAD[AUTOTUNE_ENDS])
    assert window >= poll


@pytest.mark.parametrize("last", DURING_CAPTURE, ids=[f"line{i}" for i in DURING_CAPTURE])
def test_every_log_the_lb_could_read_during_the_capture_names_the_capture(last):
    """Not the step before it. For 39 s the newest thing the head had said was a capture bar, and the LB
    reported "FlashInfer autotune done" — a step that had ended when the bars began."""
    assert phase_at(last) == StartupPhase.CAPTURING_CUDA_GRAPHS


def test_the_drafters_own_capture_is_the_same_step_the_engine_totals():
    """The brief's hypothesis, killed by the engine's own output: the drafter's capture is not a second
    step with a second figure — `Capturing model for DFlash2 speculator...` and `Capturing dflash2 CUDA
    graphs` are inside the one step whose single closing line totals both ("in 39 secs, took 1.56 GiB")."""
    assert phase_at(SPECULATOR) == StartupPhase.CAPTURING_CUDA_GRAPHS
    assert phase_at(DRAFTER_BARS) == StartupPhase.CAPTURING_CUDA_GRAPHS
    assert phase_at(CAPTURE_ENDS) == StartupPhase.CUDA_GRAPHS_CAPTURED


def test_the_terminator_still_wins_a_line_that_carries_both():
    """Docker can flush a bar and the line that follows it as one `\\n`-line. The step is over then, so
    the finish must win — which is why the capture bars are read only where no _PHASE_LINES marker hit."""
    merged = HEAD[DRAFTER_BARS] + HEAD[CAPTURE_ENDS]
    assert read_startup(grown_to(HEAD, CAPTURE_BEGINS - 1) + merged + "\n").phase \
        == StartupPhase.CUDA_GRAPHS_CAPTURED


def test_the_step_is_named_by_the_very_first_bar_the_head_printed():
    """THE RULE, as a guard: the step is entered on its OPENING line, so its window starts where the step
    does — not two seconds before the next one. The line before it is still the step before it."""
    assert phase_at(CAPTURE_BEGINS - 1) == StartupPhase.AUTOTUNE_DONE
    assert phase_at(CAPTURE_BEGINS) == StartupPhase.CAPTURING_CUDA_GRAPHS


# ── the rule behind the lost item: one read must report the TRAIL, not only its last line ────────────

#: The phases generation 109's boot record actually holds (lb/state/perf/boots-2026-10-02.jsonl, the one
#: row with "generation": 109) — nine, because the recorder kept `loading["phase"]` once per 15 s poll.
RECORDED_BY_THE_POLL = ("starting vLLM", "connecting the ranks (NCCL)", "loading model weights",
                        "model weights loaded", "profiling the encoder cache",
                        "autotuning FlashInfer kernels", "FlashInfer autotune done",
                        "starting the API server", "API server up")


def test_one_read_reports_every_phase_the_log_shows_not_only_the_last_line():
    """THE RULE. `read_startup` is handed the whole tail on every poll, so every phase the start passed
    through is in front of it — and reporting only the last recognised line is what threw ten of them
    away. The trail ends at the phase the page shows, so nothing about `phase` changes."""
    got = read_startup(grown_to(HEAD, len(HEAD) - 1))
    steps = got.steps
    assert steps[-1] == got.phase == StartupPhase.API_SERVER_UP
    assert len(steps) > len(RECORDED_BY_THE_POLL)

    lost = [StartupPhase.INITIALIZING_ENGINE, StartupPhase.MODEL_LOADED, StartupPhase.PROFILING_RUN,
            StartupPhase.PROFILING_RUN_DONE, StartupPhase.ALLOCATING_KV_CACHE,
            StartupPhase.KV_CACHE_ALLOCATED, StartupPhase.WARMING_UP_KERNELS,
            StartupPhase.CAPTURING_CUDA_GRAPHS, StartupPhase.CUDA_GRAPHS_CAPTURED]
    assert [p for p in lost if p in RECORDED_BY_THE_POLL] == [], "these are the ones the poll dropped"
    assert [p for p in lost if p not in steps] == []


def test_the_trail_is_in_the_order_the_head_printed_it_and_carries_no_repeat():
    steps = read_startup(grown_to(HEAD, len(HEAD) - 1)).steps
    assert list(steps) == [p for i, p in enumerate(steps) if i == 0 or steps[i - 1] != p]
    order = [steps.index(p) for p in (StartupPhase.CONNECTING_RANKS, StartupPhase.AUTOTUNE_DONE,
                                     StartupPhase.CAPTURING_CUDA_GRAPHS, StartupPhase.API_SERVER_UP)]
    assert order == sorted(order)


def test_the_trail_reaches_cluster_loading_so_a_recorder_can_keep_all_of_it():
    """`cluster.loading` is the contract perflog.BootRecorder and the page read."""
    view = read_startup(grown_to(HEAD, len(HEAD) - 1)).view()
    assert view["steps"] == list(read_startup(grown_to(HEAD, len(HEAD) - 1)).steps)
    assert view["phase"] == view["steps"][-1]


# ── the drafter's own LoadPass is named for the drafter this engine runs ─────────────────────────────

def test_a_dflash_drafters_second_pass_is_draft_weights_not_weights_pass_2():
    """Generation 109 declared `method='dflash'` and `Resolved architecture: DFlash2DraftModel`, neither
    of which the MTP-only detector matched, so the drafter's own read of the checkpoint was published as
    the anonymous "weights pass 2"."""
    passes = read_startup(grown_to(HEAD, len(HEAD) - 1)).passes
    assert [p.name for p in passes] == [LoadPass.MODEL, LoadPass.DRAFT]
