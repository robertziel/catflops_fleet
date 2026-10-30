"""The cluster's startup, in the head rank's own words.

One vocabulary, the same in this module, in `/lb/status` (`cluster.loading`), on the page
(`static/gateway.js` `startupProgress`) and in the tests (`tests/test_cluster_startup.py`):

  StartupPhase     the step of a start the head is in, named from the last line it printed that says so:
                   "connecting the ranks (NCCL)", "loading MTP draft weights", "API server up".
  LoadPass         one read of the checkpoint's shards, counted from 0 by the head's own loader line:
                   "model weights", then "MTP draft weights". A start has as many as its log shows, of any
                   size, and a finished one keeps its count.
  StartupProgress  what `cluster.loading` carries (`view`): the StartupPhase, every LoadPass so far, and —
                   only while the last LoadPass is still counting — its count at the top level.

Nothing is predicted: a StartupPhase is named once the head has printed its line, a LoadPass once it has
counted. The lines are quoted from the 2026-09-15 start (tests/fixtures/qwen38-start-2026-09-15-rank0.log).
Audit: worklog/vllm-cluster-loading-status.md. Frame: worklog/vllm-gateway-program.md, pass 3.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import ClassVar


@dataclass(frozen=True)
class LoadPass:
    """One read of the checkpoint's shards, at its last real count."""
    name: str
    completed: int
    total: int
    percent: int

    #: The first LoadPass after the load begins: the served model's own weights.
    MODEL: ClassVar[str] = "model weights"
    #: The second, when the head says it runs an MTP drafter: the drafter reading the checkpoint again.
    MTP_DRAFT: ClassVar[str] = "MTP draft weights"
    #: The same read by a drafter that is not an MTP head — a checkpoint of its own (`dflash`).
    DRAFT: ClassVar[str] = "draft weights"
    #: A LoadPass whose place in the load cannot be read: the line that begins the load is out of view.
    CHECKPOINT: ClassVar[str] = "checkpoint shards"

    @classmethod
    def named(cls, number: int, *, load_in_view: bool, drafter: str | None) -> str:
        """The name of a load's `number`th LoadPass, as far as the log proves it — never a guess.
        `drafter` is the speculative METHOD the head declared, so a drafter that is not an MTP head
        still has its own read of the checkpoint named for what it is."""
        if not load_in_view:
            return cls.CHECKPOINT
        if number == 1:
            return cls.MODEL
        if number == 2 and drafter:
            return cls.MTP_DRAFT if drafter == "mtp" else cls.DRAFT
        return f"weights pass {number}"

    def view(self) -> dict:
        return {"name": self.name, "completed": self.completed, "total": self.total, "percent": self.percent}


class StartupPhase:
    """The step of a start the head rank is in, in the words the page shows. A LoadPass's own phases carry
    its name: `StartupPhase.loading(LoadPass.MODEL)` is "loading model weights"."""

    STARTING_VLLM = "starting vLLM"
    STARTING_SGLANG = "starting SGLang"
    STARTING_TENSORFOLD = "starting TensorFold"
    #: TensorFold 0.6.6 compiles its CUDA extensions at a first start ("building CUDA extension …") — a step of its own,
    #: minutes long, that is neither a load nor a capture.
    BUILDING_KERNELS = "building CUDA kernels"
    INITIALIZING_ENGINE = "initializing the engine"
    CONNECTING_RANKS = "connecting the ranks (NCCL)"
    INITIALIZING_MODEL = "initializing model"
    MODEL_LOADED = "model loaded"
    PROFILING_ENCODER_CACHE = "profiling the encoder cache"
    COMPILING_MODEL = "compiling the model (torch.compile)"
    COMPILING_MTP_DRAFTER = "compiling the MTP drafter (torch.compile)"
    PROFILING_RUN = "initial profiling/warmup run"
    PROFILING_RUN_DONE = "profiling/warmup run done"
    ALLOCATING_KV_CACHE = "allocating the KV cache"
    KV_CACHE_ALLOCATED = "KV cache allocated"
    WARMING_UP_KERNELS = "warming up kernels"
    AUTOTUNING = "autotuning FlashInfer kernels"
    AUTOTUNE_DONE = "FlashInfer autotune done"
    CAPTURING_CUDA_GRAPHS = "capturing CUDA graphs"
    CUDA_GRAPHS_CAPTURED = "CUDA graphs captured"
    STARTING_API_SERVER = "starting the API server"
    API_SERVER_UP = "API server up"

    @staticmethod
    def loading(load_pass: str) -> str:
        return f"loading {load_pass}"

    @staticmethod
    def is_loading(phase: str) -> bool:
        """Is `phase` a LoadPass's own step — the weights being read? The one kind of step whose length is the
        STORAGE's, and the only one under which the loop reads the box's memory as progress (`cluster_loop._track`)."""
        return phase.startswith("loading ")

    @staticmethod
    def loaded(load_pass: str) -> str:
        return f"{load_pass} loaded"

    @staticmethod
    def compiling(what: str) -> str:
        return f"compiling {what} (torch.compile)"


@dataclass(frozen=True)
class StartupProgress:
    """The start so far: its StartupPhase, every LoadPass in order, whether the last one still counts, and
    `steps` — every StartupPhase this log shows, in order, consecutive repeats collapsed. `phase` is the
    last of `steps`; the trail is there because the LB SAMPLES this (one read per 15 s, no cursor), so a
    consumer that keeps only `phase` loses every step narrower than a poll."""
    phase: str
    passes: tuple[LoadPass, ...] = ()
    counting: bool = False
    steps: tuple[str, ...] = ()
    #: WHEN the head printed the line that last moved its start (UTC epoch seconds, on the BOX's clock), or None
    #: when that line carries no time of its own (a tqdm bar). Not part of `view()`: it is the stall clock's
    #: (`cluster_loop._track`), so the silence a restarted LB states is the head's, not the LB's.
    printed_at: float | None = None

    def view(self) -> dict:
        """`cluster.loading`, the contract the page reads."""
        out: dict = {"phase": self.phase, "steps": list(self.steps)}
        if self.counting and self.passes:
            now = self.passes[-1]
            out.update(completed=now.completed, total=now.total, percent=now.percent)
        if self.passes:
            out["passes"] = [p.view() for p in self.passes]
        return out


# ── reading the head's log ────────────────────────────────────────────────────

#: A LoadPass counting: tqdm, on the head only — rank 1 prints no count.
_COUNT = re.compile(r"(?:Loading safetensors checkpoint shards|Multi-thread loading shards):"
                    r"\s*(\d+)%\s+Completed\s+\|\s+(\d+)/(\d+)")
#: The same bar by its PREFIX alone: a frame docker cut short (a 16 KiB chunk, a truncated tail) still names the
#: step, even when its count cannot be read.
_BAR = re.compile(r"Loading safetensors checkpoint shards|Multi-thread loading shards")
#: The head says it runs a DRAFTER, and WHICH: the speculative method in its CLI args or engine config,
#: else the drafter's own architecture line. An MTP-only pattern read generation 109's `method='dflash'`
#: and `Resolved architecture: DFlash2DraftModel` as no drafter at all, so the drafter's own read of the
#: checkpoint was published as the anonymous "weights pass 2" (2026-10-02).
_DRAFTER_METHOD = re.compile(r"speculative_config.{0,30}?method'?\s*[=:]\s*'(\w+)'"
                             r"|'speculative_algorithm':\s*'(\w+)'")
_DRAFTER_ARCH = re.compile(r"Resolved architecture: (\w*(?:MTP|Draft\w*))\b")
_COMPILE = re.compile(r"Using cache directory: \S*/(\w+) for vLLM's torch\.compile")
_LOAD_BEGINS = "Loading model from scratch"
_LOAD_PASS_ENDS = "Loading weights took"
#: A line the head prints → the StartupPhase it begins, in the order a start prints them. A closing line
#: ("… took") names what it closed, or the step vLLM's own monitor says follows it.
#:
#: A STEP IS ENTERED ON ITS OPENING LINE, never left to its terminator alone. The LB does not follow this
#: log: `cluster_loop._track` re-reads a tail of it once per `Config.cluster_logs_every_s` (15 s) and keeps
#: no cursor, so `read_startup` reports the LAST recognised line in that tail — and a phase whose line is
#: superseded sooner than one poll is read and then discarded, however right its pattern is. Graph capture
#: was the one step here matched by its terminator only: on the 2026-10-02 generation-109 start it ran 39 s
#: ("Graph capturing finished in 39 secs") and was reportable for the 2 s between that line and the API
#: server's own, so `boots-*.jsonl` holds "CUDA graphs captured" for 9 of the 43 boots that recorded any
#: phase, on every model, against 43 of 43 for "loading model weights". Recorded gaps are quantised to the
#: poll (modal 16.6 s, none under 15.0 s of 339), which is the sampler, not a lost line.
_PHASE_LINES = (
    ("non-default args:", StartupPhase.STARTING_VLLM),
    ("Initializing a V1 LLM engine", StartupPhase.INITIALIZING_ENGINE),
    ("distributed_init_method=", StartupPhase.CONNECTING_RANKS),
    ("Model loading took", StartupPhase.MODEL_LOADED),
    ("Encoder cache will be initialized", StartupPhase.PROFILING_ENCODER_CACHE),
    ("torch.compile took", StartupPhase.PROFILING_RUN),
    ("Initial profiling/warmup run took", StartupPhase.PROFILING_RUN_DONE),
    ("memory for KV Cache", StartupPhase.ALLOCATING_KV_CACHE),
    ("GPU KV cache size:", StartupPhase.KV_CACHE_ALLOCATED),
    ("JIT kernel warmup starting", StartupPhase.WARMING_UP_KERNELS),
    ("Autotuning process starts", StartupPhase.AUTOTUNING),
    ("Autotuning process ends", StartupPhase.AUTOTUNE_DONE),
    ("Graph capturing finished", StartupPhase.CUDA_GRAPHS_CAPTURED),
    ("init engine (profile, create kv cache, warmup model) took", StartupPhase.STARTING_API_SERVER),
    ("Application startup complete", StartupPhase.API_SERVER_UP),
)
#: THE SAME STEPS, IN SGLANG'S OWN WORDS. The page draws one vocabulary whichever engine is behind
#: :8010, so these map onto the StartupPhase values above rather than introducing a second set. Read off
#: the first SGLang start this LB ever drove (generation 121, 2026-10-03,
#: lb/tests/fixtures/sglang-lb-start-2026-10-03-rank0.log) — not from documentation, because the whole
#: class of bug this fixes came from figures that were derived instead of read. Before it, exactly ONE
#: line of a real SGLang start was recognised ("Application startup complete", which both engines print),
#: so the panel showed a single step for a two-and-a-half-minute boot.
_SGLANG_PHASE_LINES = (
    ("server_args={", StartupPhase.STARTING_SGLANG),
    ("Init torch distributed begin", StartupPhase.CONNECTING_RANKS),
    ("Init torch distributed ends", StartupPhase.INITIALIZING_MODEL),
    ("Mamba Cache is allocated", StartupPhase.ALLOCATING_KV_CACHE),
    ("KV Cache is allocated", StartupPhase.KV_CACHE_ALLOCATED),
    ("Running FlashInfer autotune", StartupPhase.AUTOTUNING),
    ("FlashInfer autotune completed", StartupPhase.AUTOTUNE_DONE),
    ("CUDA graph begin", StartupPhase.CAPTURING_CUDA_GRAPHS),
    ("CUDA graph end", StartupPhase.CUDA_GRAPHS_CAPTURED),
    ("Engine startup timings", StartupPhase.STARTING_API_SERVER),
    ("Application startup complete", StartupPhase.API_SERVER_UP),
    ("The server is fired up and ready to roll", StartupPhase.API_SERVER_UP),
)
#: THE SAME STEPS, IN TENSORFOLD'S OWN WORDS — read off the spike's real starts (2026-10-29,
#: tests/fixtures/tensorfold-{native,py066}-2026-10-29-rank0.log), both servers' head rank:
#:
#:   native (tensorfold-native)  `[tensorfold] vision:` → `info: rank 0: waiting for rank 1` (rank 1: `joining rank 0`)
#:                               → `settings agreed, NCCL joined` → (91 s, no line) → `info: sequence memory:` →
#:                               `[tensorfold] serving … loaded in 91.4s`. The silence after NCCL IS the weight load: its
#:                               next line is the caches its weights left room for, sized from MemAvailable after the
#:                               load ("43.35 GiB free"), and the box gives up ~67 GiB in between — so the step it opens
#:                               is "loading model weights", under which the loop reads the box's memory as progress.
#:   0.6.6 (tensorfold)          `[tensorfold] loading <name>: … rank 0 of 2` (its own opener) → `building CUDA extension`
#:                               → `self-check on cuda` → `Flash Next on CUDA: … free for their caches` → `serving`.
_TENSORFOLD_PHASE_LINES = (
    ("[tensorfold] vision:", StartupPhase.STARTING_TENSORFOLD),
    (": waiting for rank 1 on", StartupPhase.CONNECTING_RANKS),
    (": joining rank 0 at", StartupPhase.CONNECTING_RANKS),
    ("settings agreed, NCCL joined", StartupPhase.loading(LoadPass.MODEL)),
    ("[tensorfold] loading ", StartupPhase.loading(LoadPass.MODEL)),
    ("[tensorfold] building CUDA extension", StartupPhase.BUILDING_KERNELS),
    ("self-check on cuda", StartupPhase.WARMING_UP_KERNELS),
    ("info: sequence memory:", StartupPhase.KV_CACHE_ALLOCATED),
    ("free for their caches", StartupPhase.KV_CACHE_ALLOCATED),
    ("[tensorfold] serving ", StartupPhase.API_SERVER_UP),
)
#: Which engine printed this log, by TensorFold's own line prefixes (checked before SGLang's: a TensorFold log never
#: says sglang, but nothing should depend on that).
_IS_TENSORFOLD = re.compile(r"\[tensorfold\] |\binfo: rank \d: (?:waiting for|joining) rank")
#: SGLang opens and closes each weight pass with these; the target's pass and the drafter's print the
#: same pair, and `Load weight end.` names which it closed by its `type=` (Qwen3_5ForConditionalGeneration,
#: then DFlash2DraftModel).
#:
#: THE OPENER IS THE LOAD'S OWN STEP ON SGLANG, bar or no bar. Its shard bar is tqdm's `\r` frames with no `\n`
#: until the pass is done, and docker's json-file driver logs one entry per `\n` — so the whole bar is ONE entry,
#: stamped at its end, and `docker logs` shows NOTHING of a load while it runs. Generation 188 (2026-10-10,
#: tests/fixtures/sglang-gen188-2026-10-10-flash-next-rank0.log): "Load weight begin" 21:04:47, the bar's one
#: entry 21:11:33, "Load weight end" 21:11:43 — read every 15 s, the LB saw "initializing model" (the step
#: `Init torch distributed ends` had already named, so nothing MOVED) for 417 s, said "no progress", painted it
#: red, and had a relaunch due at 600 s (generation 179 was relaunched inside a healthy load). From the opener on
#: the step is "loading model weights", and `cluster_loop._track` reads the box's memory as the load's progress.
_SGLANG_LOAD_BEGINS = "Load weight begin"
_SGLANG_LOAD_PASS_ENDS = "Load weight end"
#: Which engine printed this log. The LOG says, not the profile: a pair that came up as the other engine
#: is then read as what it actually is, the same reasoning the metrics probe uses.
_IS_SGLANG = re.compile(r"sglang|server_args=\{", re.I)


#: WHEN A LINE WAS PRINTED, in the three spellings a rank's log has, every one UTC on the box's own clock (the
#: GB10 boxes and both engines' containers run Etc/UTC; checked 2026-10-03): docker's own `--timestamps` prefix
#: (`2026-09-18T23:30:41.785999778Z …`), SGLang's `[2026-10-03 07:21:13 TP0]` (or `[2026-10-03 06:26:20]` before
#: the ranks exist), vLLM's `INFO 10-02 20:46:38 [core.py:361]`, which has no year. A line with none — a tqdm bar,
#: uvicorn's `INFO:     Application startup complete.`, torch's `W1003 06:28:00.854000` — has no time here.
#: Generation 134 (2026-10-03) is why: its last line was printed 07:21:13, a redeployed LB first read it ~07:23:18
#: and stated "no progress for 131 s" at 07:25:29 — the silence measured from the LB's read, not from the head.
_DOCKER_STAMP = re.compile(r"^(\d{4})-(\d\d)-(\d\d)T(\d\d):(\d\d):(\d\d)(?:\.\d+)?Z ")
_SGLANG_STAMP = re.compile(r"\[(\d{4})-(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d)(?: [^\]]*)?\]")
_VLLM_STAMP = re.compile(r"\b(?:DEBUG|INFO|WARNING|ERROR|CRITICAL) (\d\d)-(\d\d) (\d\d):(\d\d):(\d\d) \[")
#: A year-less vLLM stamp more than this far past the reader's own clock is LAST year's (a December line read in
#: January), never a line from the future.
_YEAR_SLACK_S = 86400.0


def _epoch(*parts) -> float | None:
    try:
        return datetime(*(int(p) for p in parts), tzinfo=timezone.utc).timestamp()
    except (ValueError, OverflowError):
        return None


def printed_at(line: str, now: float) -> float | None:
    """When the rank printed `line`, as UTC epoch seconds on its box's clock — None when it says no time. `now` dates
    vLLM's stamps, which carry no year: the reader's year, or the one before if that would put the line a day or
    more in the future. Where a line carries several (tqdm's `\\r` frames run into the next logged line), the LAST
    is the one that printed the words a marker matched."""
    m = _DOCKER_STAMP.match(line)
    if m is not None:
        return _epoch(*m.groups())
    stamps = _SGLANG_STAMP.findall(line)
    if stamps:
        return _epoch(*stamps[-1])
    stamps = _VLLM_STAMP.findall(line)
    if not stamps:
        return None
    year = time.gmtime(now).tm_year
    at = _epoch(year, *stamps[-1])
    if at is not None and at > now + _YEAR_SLACK_S:
        at = _epoch(year - 1, *stamps[-1])
    return at


#: GRAPH CAPTURE BEGINS: the only thing vLLM prints when it starts — the tqdm bars themselves, one per
#: capture pass, and the drafter's own capture line. The qualifier the bars carry is the PASS, not a step
#: of its own ("Capturing CUDA graphs (PIECEWISE)" / "(FULL)", "Capturing prefill/decode CUDA graphs" on
#: 2026-09-15, "Capturing dflash2 CUDA graphs" and "Capturing model for DFlash2 speculator..." on
#: 2026-10-02): the engine totals every pass, the target's and the drafter's, in the one closing line it
#: times the step with, so they are all that one step. Read only where no _PHASE_LINES marker hit, so a
#: `\r` bar that docker flushed together with the line after it still reads as the step that FOLLOWS.
_CAPTURING = re.compile(r"Capturing (?:\S+ )?CUDA graphs|Capturing model for (?:\S+ )?speculator")


def _drafter(log: str) -> str | None:
    """The speculative METHOD the head declared, else the one its resolved drafter architecture names —
    None when the log shows no drafter at all. An architecture carrying MTP is an MTP head however it is
    spelled (rank 1 of the 2026-09-15 start printed no `speculative_config` at all, only `Resolved
    architecture: Qwen4ExpMTP`), so it answers "mtp" like the method line would."""
    m = _DRAFTER_METHOD.search(log)
    if m is not None:
        # Two spellings, two groups: vLLM's `speculative_config … method='dflash'` and SGLang's own
        # `'speculative_algorithm': 'DFLASH'`. Whichever matched is the one that is not None.
        return next(g for g in m.groups() if g).lower()
    arch = _DRAFTER_ARCH.search(log)
    if arch is None:
        return None
    return "mtp" if "MTP" in arch.group(1) else arch.group(1).lower()


def read_startup(log: str, *, now: float | None = None) -> StartupProgress | None:
    """The head rank's start so far, from its log since its container started — None while it has printed
    nothing recognisable. The last recognised line names the StartupPhase. A count opens or advances a
    LoadPass, and "Loading weights took" or any later phase's line ends it; SGLang's "Load weight begin" opens
    the pass's step before any count (its bar is out of docker's sight until the pass is done), and the bar's
    bare prefix (`_BAR`) names it too. A tqdm `\\r` is not a line end: the last count on a line wins. A line no
    marker claims may still name a step the engine opens with its capture bars alone (`_CAPTURING`), so the
    39 s of graph capture is not read as the step before it.

    `printed_at` is the time of the line that last MOVED the start — changed its phase, a LoadPass or whether one
    counts — as that line states it (`printed_at()`; `now`, default the reader's clock, dates vLLM's year-less
    stamps). A line that re-states the step it is in moves nothing, exactly as the LB's own comparison of two reads
    sees it."""
    now = time.time() if now is None else now
    tensorfold = bool(_IS_TENSORFOLD.search(log))
    sglang = not tensorfold and bool(_IS_SGLANG.search(log))
    phase_lines = (_TENSORFOLD_PHASE_LINES if tensorfold else _SGLANG_PHASE_LINES if sglang else _PHASE_LINES)
    load_begins = _SGLANG_LOAD_BEGINS if sglang else _LOAD_BEGINS
    load_pass_ends = _SGLANG_LOAD_PASS_ENDS if sglang else _LOAD_PASS_ENDS
    drafter = _drafter(log)
    mtp = drafter == "mtp"
    load_in_view, counting, finished = False, False, 0
    passes: list[LoadPass] = []
    phase = ""
    steps: list[str] = []
    moved_at: float | None = None
    for line in log.split("\n"):
        before = (phase, counting, len(passes), passes[-1] if passes else None)
        if load_begins in line:
            # vLLM prints its opener ONCE, before the whole load, so it resets what came before. SGLang
            # prints "Load weight begin." per PASS — the target's, then the drafter's — so resetting there
            # renumbered the drafter's pass as the first and published it as a second read of the model
            # weights. The opener still names the step; only vLLM's clears the passes behind it.
            load_in_view, counting = True, False
            if not sglang:
                finished, passes = 0, []
                # vLLM's "Loading model from scratch" comes a few seconds BEFORE its bar, whose every frame
                # reaches docker on its own line (the worker's prefixer ends each with `\n`): the bar opens
                # the load's step itself, as it always has.
                phase = StartupPhase.INITIALIZING_MODEL
            else:
                # SGLang's "Load weight begin" IS the load: its bar is one `\r` entry docker shows only at the
                # end (see _SGLANG_LOAD_BEGINS), so the pass is named here, from the opener, and counts later.
                phase = StartupPhase.loading(LoadPass.named(max(finished, len(passes)) + 1,
                                                            load_in_view=True, drafter=drafter))
        elif (counts := _COUNT.findall(line)):
            percent, completed, total = (int(v) for v in counts[-1])
            if not (total > 0 and 0 <= completed <= total and 0 <= percent <= 100):
                continue
            last = passes[-1] if passes else None
            if counting and last is not None and total == last.total and completed >= last.completed:
                passes[-1] = LoadPass(last.name, completed, total, percent)
            else:
                name = LoadPass.named(max(finished, len(passes)) + 1, load_in_view=load_in_view,
                                      drafter=drafter)
                passes.append(LoadPass(name, completed, total, percent))
            counting, phase = True, StartupPhase.loading(passes[-1].name)
        elif load_pass_ends in line:
            name = passes[-1].name if counting else LoadPass.named(finished + 1, load_in_view=load_in_view,
                                                                   drafter=drafter)
            finished += 1
            counting, phase = False, StartupPhase.loaded(name)
        elif (compiled := _COMPILE.search(line)):
            what = compiled.group(1)
            counting = False
            phase = (StartupPhase.COMPILING_MODEL if what == "backbone"
                     else StartupPhase.COMPILING_MTP_DRAFTER if what == "eagle_head" and mtp
                     else StartupPhase.compiling(what))
        else:
            for marker, named in phase_lines:
                if marker in line:
                    counting, phase = False, named
                    break
            else:
                if _CAPTURING.search(line):
                    counting, phase = False, StartupPhase.CAPTURING_CUDA_GRAPHS
                elif _BAR.search(line):
                    # The bar's prefix with no count a reader can trust: the weights ARE loading — the pass
                    # that is counting, else the next one — and the count stays what it was.
                    name = (passes[-1].name if counting and passes
                            else LoadPass.named(max(finished, len(passes)) + 1, load_in_view=load_in_view,
                                                drafter=drafter))
                    phase = StartupPhase.loading(name)
        # THE TRAIL. One read is handed the whole tail, so every step the start passed through is in
        # front of it; reporting only `phase` is what threw ten of generation 109's nineteen away.
        if phase and (not steps or steps[-1] != phase):
            steps.append(phase)
        if (phase, counting, len(passes), passes[-1] if passes else None) != before:
            moved_at = printed_at(line, now)
    return StartupProgress(phase, tuple(passes), counting, tuple(steps), moved_at) if phase else None


#: THE PREFILL BUDGET THE ENGINE RUNS, in its own words — what the LB SENT is not evidence of what the engine took
#: (generation 135, 2026-10-03: 2048 sent, `chunked_prefill_size=8192` run). vLLM's scheduler says
#: `Chunked prefill is enabled with max_num_batched_tokens=4096.`; SGLang states its resolved figures as
#: `max_total_num_tokens=…, chunked_prefill_size=8192, max_prefill_tokens=16384, …` once its pools are allocated,
#: and the value it was given in `server_args={… 'chunked_prefill_size': 8192 …}` at its first line. Best first.
_STATED_BUDGET = (re.compile(r"Chunked prefill is enabled with max_num_batched_tokens=(\d+)"),
                  re.compile(r"\bchunked_prefill_size=(\d+), max_prefill_tokens="),
                  re.compile(r"'chunked_prefill_size': (\d+)"),
                  # The native TensorFold server's own chunk: "[tensorfold] prompt chunks of up to 2,048 tokens".
                  re.compile(r"\[tensorfold\] prompt chunks of up to ([\d,]+) tokens"))


def stated_prefill_budget(log: str) -> int | None:
    """The prefill budget the head says it runs (the last time it says it), or None while it has not said."""
    for pattern in _STATED_BUDGET:
        found = pattern.findall(log)
        if found:
            return int(found[-1].replace(",", ""))
    return None
