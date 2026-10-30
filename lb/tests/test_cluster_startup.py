"""The cluster's startup in the head rank's own words — StartupPhase, LoadPass, StartupProgress
(vllm_lb/cluster_startup.py) — against the real log of the 2026-09-15 start, at every moment of it.

Frame: worklog/vllm-gateway-program.md, pass 3 — S1, the parser against real logs, and K1, pass 2 told
from pass 1 by evidence in the log, never guessed. Every expectation is pinned to a line the head really
printed (Start pressed 19:55:44Z; `tests/start_logs.py`), read the way the LB reads it. Two things the
page said that day were wrong, and both are here by time: the MTP drafter's LoadPass over the same 28
shards drawn as the ONE bar falling from 27/28 to 3/28 — read as a restart — and a "28 / 28" that stayed
up from 20:05:03, when the weights were in, until the pair served at 20:08:43, through every StartupPhase
between (worklog/vllm-cluster-loading-status.md).
"""
from __future__ import annotations

import pytest

from tests.start_logs import FIXTURES, HEAD, RANK1, log_at, stamped, tail
from vllm_lb.cluster_startup import LoadPass, StartupPhase, StartupProgress, read_startup

MODEL, MTP = LoadPass.MODEL, LoadPass.MTP_DRAFT


def at(when: str, log: str = HEAD) -> StartupProgress | None:
    return read_startup(log_at(log, when))


def figures(got: StartupProgress) -> dict:
    """`cluster.loading` without its `steps` trail: the figures a tile draws, for the assertions that are
    about those. The trail has assertions of its own."""
    return {k: v for k, v in got.view().items() if k != "steps"}


def now(got: StartupProgress) -> StartupProgress:
    """`got` with its trail dropped, to compare the step a log ENDS on against a bare StartupProgress."""
    return StartupProgress(got.phase, got.passes, got.counting)


def counted(name: str, completed: int) -> LoadPass:
    """A LoadPass of the 28-shard checkpoint at tqdm's own percent (100·c/28 is never exactly a half)."""
    return LoadPass(name, completed, 28, round(100 * completed / 28))


FULL = (counted(MODEL, 28), counted(MTP, 28))

#: (moment, the StartupPhase, every LoadPass so far) — and the line that makes it so.
MOMENTS = [
    ("19:56:15", StartupPhase.STARTING_VLLM, ()),                      # 19:56:12.693 non-default args: {…}
    ("19:56:40", StartupPhase.INITIALIZING_ENGINE, ()),                # 19:56:38.808 Initializing a V1 LLM engine
    ("19:56:47", StartupPhase.CONNECTING_RANKS, ()),                   # 19:56:45.368 world_size=2 rank=0 … backend=nccl
    ("19:56:52", StartupPhase.INITIALIZING_MODEL, ()),                 # 19:56:50.933 Loading model from scratch...
    ("19:57:10", StartupPhase.loading(MODEL), (counted(MODEL, 1),)),   # 19:57:04.420 shards:   4% Completed | 1/28
    ("20:03:50", StartupPhase.loading(MODEL), (counted(MODEL, 27),)),  # 20:03:41.817 shards:  96% Completed | 27/28
    ("20:04:05", StartupPhase.loaded(MODEL), (counted(MODEL, 28),)),   # 20:04:04.283 Loading weights took 429.83 seconds
    ("20:04:25.3", StartupPhase.loading(MTP), (counted(MODEL, 28), counted(MTP, 7))),  # 20:04:25.216 7/28
    ("20:05:02.47", StartupPhase.loading(MTP), FULL),                  # 20:05:02.459 28/28, its "took" not out yet
    ("20:05:02.5", StartupPhase.loaded(MTP), FULL),                    # 20:05:02.487 Loading weights took 56.16 seconds
    ("20:05:10", StartupPhase.MODEL_LOADED, FULL),                     # 20:05:03.430 Model loading took 64.46 GiB memory
    ("20:05:30", StartupPhase.PROFILING_ENCODER_CACHE, FULL),          # 20:05:22.888 Encoder cache will be initialized
    ("20:06:00", StartupPhase.COMPILING_MODEL, FULL),                  # 20:05:44.243 …/backbone for vLLM's torch.compile
    ("20:06:30", StartupPhase.PROFILING_RUN, FULL),                    # 20:06:12.569 torch.compile took 34.23 s in total
    ("20:06:53", StartupPhase.COMPILING_MTP_DRAFTER, FULL),            # 20:06:52.370 …/eagle_head
    ("20:06:55.7", StartupPhase.ALLOCATING_KV_CACHE, FULL),            # 20:06:55.596 … memory for KV Cache …
    ("20:06:56", StartupPhase.KV_CACHE_ALLOCATED, FULL),               # 20:06:55.789 GPU KV cache size: 1,708,119 tokens
    ("20:07:20", StartupPhase.AUTOTUNING, FULL),                       # 20:06:57.756 Autotuning process starts
    ("20:07:35", StartupPhase.AUTOTUNE_DONE, FULL),                    # 20:07:33.091 Autotuning process ends
    ("20:07:37.7", StartupPhase.CAPTURING_CUDA_GRAPHS, FULL),          # 20:07:37.634 Capturing CUDA graphs (PIECEWISE)
    ("20:07:37.9", StartupPhase.CAPTURING_CUDA_GRAPHS, FULL),          # 20:07:37.843 Capturing model for speculator...
    ("20:07:40", StartupPhase.CUDA_GRAPHS_CAPTURED, FULL),             # 20:07:37.962 Graph capturing finished in 5 secs
    ("20:08:30", StartupPhase.STARTING_API_SERVER, FULL),              # 20:07:42.985 init engine (…) took 146.41 s
    ("20:08:40", StartupPhase.API_SERVER_UP, FULL),                    # 20:08:38.066 Application startup complete.
]


@pytest.mark.parametrize("when, phase, passes", MOMENTS, ids=[m[0] for m in MOMENTS])
def test_the_startup_phase_is_the_last_thing_the_head_said_and_every_load_pass_keeps_its_count(when, phase, passes):
    got = at(when)
    assert got is not None and (got.phase, got.passes) == (phase, passes)


def test_s1_the_model_and_the_mtp_drafter_are_two_load_passes_each_counting_0_to_28():
    """Frame pass 3, S1: pass 1 "model weights" 0 → 28/28 (`Loading weights took 429.83 seconds`) and pass 2
    "MTP draft weights" 0 → 28/28 (`… 56.16 seconds`), kept as two LoadPasses."""
    assert at("19:56:54.5").passes == (counted(MODEL, 0),)                   # 19:56:54.444  0/28
    assert "Loading weights took 429.83 seconds" in log_at(HEAD, "20:04:04.3")
    assert at("20:04:04.3").passes == (counted(MODEL, 28),)                  # 20:04:03.799 28/28
    assert at("20:04:06.4").passes == (counted(MODEL, 28), counted(MTP, 0))  # 20:04:06.331  0/28 again
    assert "Loading weights took 56.16 seconds" in log_at(HEAD, "20:05:02.5")
    assert at("20:05:02.5").passes == FULL                                   # 20:05:02.459 28/28


def test_the_view_is_the_contract_the_page_reads():
    """`cluster.loading`: `phase` once the head has said anything recognisable; `passes` once a LoadPass has
    counted; `completed`/`total`/`percent` only while the last LoadPass is still counting — the fields a page
    that reads no `passes` still draws, so it never draws a finished pass as the current one."""
    assert at("19:56:05") is None and at("19:56:12.69") is None          # no log yet, then a CLI warning only
    assert at("19:56:52").view() == {"phase": "initializing model", "steps": [
        "starting vLLM", "initializing the engine", "connecting the ranks (NCCL)", "initializing model"]}
    assert at("20:04:05").view() == {"phase": "model weights loaded", "steps": [
        "starting vLLM", "initializing the engine", "connecting the ranks (NCCL)", "initializing model",
        "loading model weights", "model weights loaded"], "passes": [
        {"name": "model weights", "completed": 28, "total": 28, "percent": 100}]}
    assert figures(at("20:04:25.3")) == {"phase": "loading MTP draft weights", "completed": 7, "total": 28,
                                         "percent": 25, "passes": [
        {"name": "model weights", "completed": 28, "total": 28, "percent": 100},
        {"name": "MTP draft weights", "completed": 7, "total": 28, "percent": 25}]}
    assert figures(at("20:06:00")) == {"phase": "compiling the model (torch.compile)", "passes": [
        {"name": "model weights", "completed": 28, "total": 28, "percent": 100},
        {"name": "MTP draft weights", "completed": 28, "total": 28, "percent": 100}]}
    # and the trail reaches `cluster.loading` whole, so a consumer need not catch a phase as it passes
    assert at("20:06:00").view()["steps"] == [
        "starting vLLM", "initializing the engine", "connecting the ranks (NCCL)", "initializing model",
        "loading model weights", "model weights loaded", "loading MTP draft weights",
        "MTP draft weights loaded", "model loaded", "profiling the encoder cache",
        "compiling the model (torch.compile)"]


def test_replayed_line_by_line_no_load_pass_loses_its_name_or_counts_down_and_no_count_outlives_it():
    """The LB reads every 15 s, so any line can be the last one it sees. At every one of them: a LoadPass once
    shown keeps its name and never counts down; a top-level count is the LoadPass still counting; and once the
    model is loaded no StartupPhase says it is loading, or initializing, again."""
    lines = stamped(HEAD)
    before: tuple[LoadPass, ...] = ()
    loaded, last = False, None
    for i, (stamp, _) in enumerate(lines, start=1):
        now = read_startup("".join(f"{body}\n" for _, body in lines[:i]))
        if now is None:
            assert last is None, f"{stamp}: the startup went blank"
            continue
        assert [p.name for p in now.passes[:len(before)]] == [p.name for p in before], f"{stamp}: {now.passes}"
        assert all(c.completed >= w.completed for w, c in zip(before, now.passes)), f"{stamp}: {now.passes}"
        view = now.view()
        if "completed" in view:
            assert now.counting and view["phase"] == StartupPhase.loading(now.passes[-1].name), f"{stamp}: {view}"
        loaded = loaded or now.phase == StartupPhase.MODEL_LOADED
        if loaded:
            assert "completed" not in view and not now.phase.startswith("loading ") \
                and now.phase != StartupPhase.INITIALIZING_MODEL, f"{stamp}: {view}"
        before, last = now.passes, now
    assert last is not None and (last.phase, last.passes) == (StartupPhase.API_SERVER_UP, FULL)


def test_a_rank_that_prints_no_count_is_not_initializing_once_its_weights_are_in():
    """Rank 1 read the same checkpoint for 430 s and again for 66 s and printed no count at all. Any such log was
    "initializing model" to the old fallback, from its first load line to the end of the start."""
    assert now(at("20:00:00", RANK1)) == StartupProgress(StartupPhase.INITIALIZING_MODEL)
    assert now(at("20:04:05", RANK1)) == StartupProgress(StartupPhase.loaded(MODEL))  # 20:04:04.382 took 430.28 s
    assert now(at("20:05:13", RANK1)) == StartupProgress(StartupPhase.loaded(MTP))    # 20:05:12.793 took 65.78 s
    assert now(at("20:07:42.1", RANK1)) == StartupProgress(StartupPhase.CUDA_GRAPHS_CAPTURED)  # 20:07:37.967
    # rank 1 prints no count, so its trail is the phases alone — and it carries the capture step too
    assert StartupPhase.CAPTURING_CUDA_GRAPHS in at("20:07:42.1", RANK1).steps


def test_k1_the_second_load_pass_is_the_mtp_drafter_only_on_the_logs_own_evidence():
    """Frame pass 3, K1 — never guess. Pass 2 is the drafter because the head said it runs one: `speculative_config`
    with method 'mtp', and `Resolved architecture: Qwen4ExpMTP`. Without those lines it is only a second pass."""
    log = "".join(f"{line}\n" for line in log_at(HEAD, "20:04:25.3").split("\n")
                  if line and "speculative_config" not in line and "Resolved architecture" not in line)
    got = read_startup(log)
    assert got.phase == "loading weights pass 2"
    assert [p.name for p in got.passes] == [MODEL, "weights pass 2"]


def test_out_of_view_of_the_loads_first_line_a_load_pass_gets_no_name_it_cannot_prove():
    """By the head's first /health 200 its start had printed 289 lines. The last 200 of them hold neither
    `Loading model from scratch` nor a line naming the drafter, so which LoadPass is which cannot be read from
    them — why the loop reads the whole log (test_cluster_loop.py)."""
    whole = log_at(HEAD, "20:08:43.3")
    assert whole.count("\n") == 289
    got = read_startup(tail(whole, 200))
    assert got.phase == StartupPhase.API_SERVER_UP
    assert got.passes == (counted(LoadPass.CHECKPOINT, 28),) * 2 and LoadPass.CHECKPOINT == "checkpoint shards"


def test_nothing_recognisable_is_nothing_and_an_impossible_count_is_ignored():
    assert read_startup("INFO: Started server process") is None
    assert read_startup("Loading safetensors checkpoint shards: 150% Completed | 30/28") is None


SGLANG_START = FIXTURES / "sglang-lb-start-2026-10-03-rank0.log"


def test_an_sglang_start_reads_in_the_same_words_a_vllm_one_does():
    """The panel draws ONE vocabulary whichever engine is behind :8010. Before these markers the detector
    recognised exactly one line of a real SGLang start — "Application startup complete", which both
    engines happen to print — so a two-and-a-half-minute boot showed a single step and the operator
    watched an empty progress trail. Read off the first SGLang start this LB ever drove (generation 121,
    2026-10-03), not off documentation."""
    from vllm_lb.cluster_startup import StartupPhase, read_startup
    got = read_startup(SGLANG_START.read_text())
    assert got is not None and got.phase == StartupPhase.API_SERVER_UP
    # Every step the boot really passed through, in order, in the page's own words.
    assert got.steps[0] == StartupPhase.STARTING_SGLANG
    for expected in (StartupPhase.CONNECTING_RANKS, StartupPhase.KV_CACHE_ALLOCATED,
                     StartupPhase.AUTOTUNING, StartupPhase.AUTOTUNE_DONE,
                     StartupPhase.CAPTURING_CUDA_GRAPHS, StartupPhase.CUDA_GRAPHS_CAPTURED,
                     StartupPhase.STARTING_API_SERVER, StartupPhase.API_SERVER_UP):
        assert expected in got.steps, f"{expected!r} missing from {list(got.steps)}"
    assert len(got.steps) >= 15, f"one SGLang line was recognised before this; now {len(got.steps)}"
    # SGLang opens a pass per checkpoint, so its drafter must not be read as a second pass over the model.
    assert [(x.name, x.completed, x.total) for x in got.passes] == [
        ("model weights", 3, 3), ("draft weights", 1, 1)]


def test_the_vllm_trail_is_unchanged_by_the_sglang_markers():
    """The other engine's start must read exactly as it did: the table is chosen by what the log IS."""
    from vllm_lb.cluster_startup import StartupPhase, read_startup
    got = read_startup((FIXTURES / "qwen38-dflash2-start-2026-10-02-rank0.log").read_text())
    assert got is not None and got.phase == StartupPhase.API_SERVER_UP
    assert got.steps[0] == StartupPhase.STARTING_VLLM, "vLLM's own opener, not SGLang's"
    assert StartupPhase.STARTING_SGLANG not in got.steps
    assert [(x.name, x.completed, x.total) for x in got.passes][:2] == [
        ("model weights", 3, 3), ("draft weights", 1, 1)]


# ── generation 188 (2026-10-10): a seven-minute SGLang weight load the LB read as silence ──────────────

GEN188_HEAD = "sglang-gen188-2026-10-10-flash-next-rank0.log"
GEN188_DAY = "2026-10-10T"
DRAFT = LoadPass.DRAFT


def gen188_at(when: str) -> str:
    """The head's log of generation 188 as `docker logs` held it at `when` (UTC HH:MM:SS, 2026-10-10)."""
    return log_at(GEN188_HEAD, when, day=GEN188_DAY)


def test_dockers_json_file_driver_holds_a_carriage_return_bar_until_its_newline():
    """THE ROOT CAUSE, pinned to the capture (`docker logs -t` on box-a, 2026-10-10, json-file driver, docker 29.2.1):
    SGLang's loader draws its shard bar with `\\r` only, and docker logs one entry per `\\n` — so the whole
    seven-minute bar is ONE entry, stamped when its last frame ended (21:11:33), six and a half minutes after
    "Load weight begin" (21:04:47). Nothing the head printed in between reached `docker logs`, so nothing reached
    the LB, and the boot log (lb/state/perf/boots-2026-10-10.jsonl) holds no "loading model weights" phase at all
    for this start: the bar and "Load weight end" landed 10 s apart, inside one 15 s poll."""
    rows = stamped(GEN188_HEAD)
    begin = next(stamp for stamp, body in rows if "Load weight begin" in body)
    bars = [(stamp, body) for stamp, body in rows if "Multi-thread loading shards" in body]
    assert begin.startswith("2026-10-10T21:04:47"), begin
    assert bars[0][0].startswith("2026-10-10T21:11:33"), bars[0][0]
    assert bars[0][1].count("\r") >= 80, "every frame of the bar, in the one entry"
    between = [stamp for stamp, _ in rows if "2026-10-10T21:04:59" < stamp < "2026-10-10T21:11:33"]
    assert between == [], f"docker logged nothing for the whole load: {between}"


FULL_145 = LoadPass(MODEL, 145, 145, 100)


@pytest.mark.parametrize("when, phase, counting, passes", [
    ("21:04:47.0", StartupPhase.INITIALIZING_MODEL, False, ()),               # 21:04:46 Init torch distributed ends
    ("21:04:48", StartupPhase.loading(MODEL), False, ()),                     # 21:04:47 Load weight begin — no bar yet
    ("21:10:00", StartupPhase.loading(MODEL), False, ()),                     # six minutes in: the step it IS in
    ("21:11:35", StartupPhase.loading(MODEL), True, (FULL_145,)),             # 21:11:33 the bar lands, whole
    ("21:11:45", StartupPhase.loaded(MODEL), False, (FULL_145,)),             # 21:11:43 Load weight end
    ("21:12:00", StartupPhase.loading(DRAFT), False, (FULL_145,)),            # 21:11:58 Load weight begin (drafter)
    ("21:12:40", StartupPhase.loaded(DRAFT), False, (FULL_145, LoadPass(DRAFT, 145, 145, 100))),
])
def test_sglangs_load_weight_begin_opens_the_loading_step_before_its_bar_is_in_sight(when, phase, counting, passes):
    """"Load weight begin" IS the start of the load: the step is "loading model weights" from that line, bar or
    no bar, and the count joins it when docker finally shows one. Before, the opener mapped to "initializing
    model" — the same step "Init torch distributed ends" had already named — so the trail did not move at all."""
    got = read_startup(gen188_at(when))
    assert got is not None
    assert (got.phase, got.counting, got.passes) == (phase, counting, passes)


def test_the_step_the_panel_shows_six_minutes_in_is_the_load_not_initializing_model():
    """21:10, as the operator saw it: "initializing model", three steps, painted red. The head had been reading
    shards for five minutes."""
    got = read_startup(gen188_at("21:10:00"))
    assert got.steps == (StartupPhase.STARTING_SGLANG, StartupPhase.CONNECTING_RANKS,
                         StartupPhase.INITIALIZING_MODEL, StartupPhase.loading(MODEL))
    assert got.view()["phase"] == "loading model weights"
    assert "completed" not in got.view(), "no bar in sight: the LB does not invent a count"


def test_vllms_bar_prefix_opens_the_loading_step_even_without_a_readable_count():
    """"Loading safetensors checkpoint shards" names the step whatever follows it: a frame cut short (a 16 KiB chunk
    boundary, a truncated tail) still says the weights are loading."""
    log = log_at(HEAD, "19:56:52") + "(Worker_TP0 pid=398) Loading safetensors checkpoint shards:\n"
    got = read_startup(log)
    assert got.phase == StartupPhase.loading(MODEL) and not got.counting and got.passes == ()
    assert got.steps[-2:] == (StartupPhase.INITIALIZING_MODEL, StartupPhase.loading(MODEL))


def test_a_loading_step_is_known_by_its_name():
    """What the loop reads memory as progress under: a LoadPass's own step, and nothing else."""
    assert StartupPhase.is_loading(StartupPhase.loading(MODEL))
    assert StartupPhase.is_loading(StartupPhase.loading(LoadPass.DRAFT))
    assert StartupPhase.is_loading(StartupPhase.loading(LoadPass.CHECKPOINT))
    for other in (StartupPhase.INITIALIZING_MODEL, StartupPhase.loaded(MODEL), StartupPhase.AUTOTUNING,
                  StartupPhase.CAPTURING_CUDA_GRAPHS, ""):
        assert not StartupPhase.is_loading(other), other
