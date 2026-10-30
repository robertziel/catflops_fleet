"""The two-box TP=2 cluster: what the operator WANTS, what the fleet IS, and the plan between.

Three parts, kept apart on purpose (worklog/vllm-cluster-control-plane.md, worklog/vllm-gateway-program.md
pass 2):

  * `Desired` — the switch, on disk (`DesiredStore`). A restart re-reads it; nothing is
    "remembered" in a process that can die.
  * `Observed` — read fresh: each GB10 box's RESERVATION as broker_parrot's rows say it is, the
    operator's park (READ, for the words only), the fleet's other running jobs per box, each
    GB10 box's rank container (`tp-rank status`), and whether the head answers `/health`. An
    observation that could not be made is `None`, never a guess.
  * `plan()` — a PURE function of the two.

THE RESERVATION. A GB10 box belongs to the pair only through its reservation: ONE broker_parrot node_job
per box per generation — pinned to the box, weighing the whole box, flagged run-next — whose states
are the lease's (queued → running → ended). broker_parrot decides when to grant it: the pinned lane
drains, claims it, and while it RUNS stops its own vLLM and unloads ollama there. The LB never parks
a lane and never answers an operator's park or unpark (operator, 2026-09-15: "We must not interfere
with parked anywhere in vllm service, we just add node_job with priority and wait for broker_parrot
to take it").

The rules, the previous attempts' defects inverted (worklog/vllm-cluster-audit.md F1–F5, and the live
2026-09-15 20:18 fight with the operator's unpark):

  - never `Up` unless BOTH current reservations RUN, claimed by the pinned box's own lane, with no
    other fleet job there, no broker vLLM and enough memory for what THIS selection costs on THAT box
    in THAT rank's role (`BoxObs.needs_gib`, measured) — all observed;
  - never release a reservation while a rank is observed on its box, or while it cannot be observed;
  - a rank comes `Down` only when its reservation ended, was claimed by another lane, or has sat back
    in the queue longer than the engine's own lease duration — never on a guess, never for a park;
  - a reservation is asked for ONCE per generation: one that ended is a sticky failure, like an
    exited rank — the operator's Stop/Start is the retry;
  - ASK BEFORE YOU LET GO: an older generation's reservation is returned only once THIS generation's
    has been asked for on that box, so a restart never hands the fleet a window to occupy the card
    (worklog/vllm-cluster-handover.md H1);
  - Start frees RAM first: our turns on classic vLLM end before the GB10 boxes are asked for, and the
    state says `freeing` while a broker vLLM is still up;
  - A FAILED START LETS GO OF THE BOXES' MEMORY, AND SAYS WHY (worklog/vllm-cluster-start-robustness.md):
    a rank whose partner exited, a start whose log has said nothing new for its step's limit
    (`start_stall_limit`), and a pair under which a box has run out of memory are HALTED — stopped, never
    removed, because a dead rank's log is the only record of why it died. The failure stays sticky; Stop is
    still the operator's.
  - …BUT A HUNG START IS RESCUED FIRST: a start that has said nothing new for its step's limit is torn down
    and relaunched — same generation, same settings, its reservations kept — up to `START_RETRIES` times,
    and only then halted as above. Only a HANG: a rank that exited, a box out of memory or a head serving
    another model is failed at once, since a relaunch would only repeat it. The loop saves both ranks' logs
    and stacks before the Downs land (`cluster_loop`), so tearing down loses no post-mortem.
  - FAILED MEANS AN ERROR, NEVER A CLOCK (operator, 2026-10-10: "there must be real error, not timeout to
    show failed"). `failed` is a rank that exited, a box the memory guard stopped, a head serving another
    model, a reservation broker_parrot ended, a launch the LB refused — something that happened. A start
    that ran out its step's clock and was halted after its relaunches is `stalled`: the same Halt, the same
    logs kept, a line that says what hung and for how long, and a warning tone on the page. A RAM wait past
    its budget stays `freeing` and says so. A box that cannot be observed is `unknown`, with the executor's
    reason. A box that cannot SEE a folder (its NFS share not mounted yet, a stale handle, a hang) has not
    said the checkpoint is missing: the start waits for it (`checking`), and "choose another model" is said
    only of a folder the box could read and found empty. A box that REBOOTED under its rank (its container
    older than the box's boot) did not fail: both ranks come down and the same generation is relaunched.
"""
from __future__ import annotations

import inspect
import json
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field, replace

from vllm_lb.atomic import retire_blueprint_field, write_json
from vllm_lb.catalog import (CLASSIC, CLUSTER_MODE, MAX_NUM_BATCHED_TOKENS_CAP, availability_key,  # the ServingModes are
                             NUM_SPECULATIVE_TOKENS_CAP, ModelCatalog)             # defined once, in the catalog
from vllm_lb.cluster_startup import LoadPass, StartupPhase

TAG_PREFIX = "vllm-lb/cluster"
#: How long a reservation may sit back in the queue under a standing rank (a lane restart re-queues it
#: and the same box re-claims it) before the pair comes down: broker_parrot's own `DEFAULT_LEASE_S`.
REQUEUE_GRACE_S = 600.0
#: How long `freeing` may wait for the RAM before it says it is PAST ITS BUDGET, beside what is still holding
#: it. The budget is the cold start it protects (~13 min); `waiting` has had such a bound from the start and
#: `freeing` had none, so a handover that never freed waited silently forever
#: (worklog/vllm-cluster-hold-robustness.md L3). Past it the wait goes on, said — never `failed`: nothing
#: errored, and the start launches the moment the RAM is back.
FREEING_DEADLINE_S = 600.0
#: How long a RESTART waits for the gateway's turns still streaming on the pair it replaces, before taking it
#: down anyway — the redeploy drain's own window (LB_DRAIN_S): p99 of 1,179 real turns was 852 s. Meanwhile
#: the old generation is not `held`, so the router sends it nothing new and the proxy holds new turns.
FINISHING_LIMIT_S = 900.0
#: How long a start may go without its StartupPhase or a shard count changing before it is HUNG — the CEILING
#: of every step's own limit (`start_stall_limit`), and the limit itself of a step with no measured basis. A
#: healthy start's longest quiet window is 56 s (tests/fixtures/qwen38-start-2026-09-15-rank0.log) and ~2 min per
#: shard off the HDD; on 2026-09-18 three starts each sat 53 minutes in one phase while the page said
#: `starting`. The same 600 s every other wait here is bounded by.
START_STALL_S = 600.0
#: From when a quiet start says so on the page, long before it gives up.
START_QUIET_NOTE_S = 120.0
#: THE SLOWEST HEALTHY START OF EACH STEP, read off the boot log (lb/state/perf/boots-2026-09-30..10-03.jsonl,
#: 56 starts through generation 132, frozen as tests/fixtures/boots-2026-09-30-to-10-03.jsonl): over every start
#: that reached serving, a step's time is from the read that first named it to the read that first named the
#: next one, or to serving — the most `start_quiet_s` could have reached in it, plus one 15 s read. Step →
#: (slowest seconds, healthy starts it was seen in). A step is here only when it was seen on at least
#: START_STALL_MIN_SEEN healthy starts AND its length is not the STORAGE's: a LoadPass counting ("loading …")
#: and the step that opens the load ("initializing model") are I/O, ~220 MB/s off box-c's HDD over NFS, which no
#: start recorded here read cold — and 383-442 s for today's checkpoints (2026-10-10), against this 600 s ceiling.
#: On SGLang the shard bar has no line end until the pass is done, so docker shows none of it while it runs; the
#: loop reads the box's memory as the load's progress instead (`cluster_loop._load_taken`), and the ceiling is for
#: a load that has truly stopped. tests/test_start_rescue.py re-derives this table from that file.
#:
#: Graph capture reads as 16.8 s here because the 15 s sampler rarely catches it; the engines time it
#: themselves, and the slowest any captured log shows is 39 s (vLLM generation 109) — SGLang's 32-seat
#: capture is two rounds of ~9-10 s (generation 131's target round: 9.40 s). The floor covers both.
START_STEP_SLOWEST_S: dict[str, tuple[float, int]] = {
    StartupPhase.AUTOTUNING: (83.8, 41),                      # generation 103, vLLM 27B nvfp4
    StartupPhase.STARTING_API_SERVER: (82.8, 35),             # 97
    StartupPhase.PROFILING_RUN: (52.4, 22),                   # 99
    StartupPhase.AUTOTUNE_DONE: (50.0, 31),                   # 116
    StartupPhase.COMPILING_MODEL: (49.6, 18),                 # 85
    StartupPhase.PROFILING_ENCODER_CACHE: (33.6, 32),         # 88
    StartupPhase.STARTING_VLLM: (33.4, 28),                   # 103
    StartupPhase.KV_CACHE_ALLOCATED: (16.8, 9),
    StartupPhase.CAPTURING_CUDA_GRAPHS: (16.8, 6),            # the step generations 122 and 131 hung in
    StartupPhase.INITIALIZING_ENGINE: (16.7, 20),
    StartupPhase.CUDA_GRAPHS_CAPTURED: (16.7, 11),
    StartupPhase.MODEL_LOADED: (16.7, 12),
    StartupPhase.COMPILING_MTP_DRAFTER: (16.7, 3),
    StartupPhase.CONNECTING_RANKS: (16.7, 14),
    StartupPhase.compiling("dflash_head"): (16.7, 3),
    StartupPhase.loaded(LoadPass.MODEL): (16.6, 4),
    StartupPhase.API_SERVER_UP: (12.4, 9),
}
#: A step is HUNG once it has been quiet this many times its slowest healthy start…
START_STALL_FACTOR = 4.0
#: …but never sooner than this: 4.1 x the slowest graph capture any engine has timed here — 97.75 s, generation 188's
#: target-verify capture at 16 seats (SGLang Flash-Next, 2026-10-10, tests/fixtures/sglang-gen188-…-rank0.log), the
#: step quiet 100 s in the boot log against the 180 s this floor was (4.6 x vLLM generation 109's 39 s) — and
#: twenty-six of the LB's 15 s log reads, so neither a slow-but-healthy capture nor the sampler can be taken for a
#: hang. tests/test_start_rescue.py re-derives the slowest capture from every fixture's own timing line.
START_STALL_FLOOR_S = 400.0
#: How many healthy starts must have shown a step before its own limit replaces `START_STALL_S`.
START_STALL_MIN_SEEN = 3
#: How many times a HUNG start is torn down and relaunched before it is failed. Generations 122 and 131 hung on 2
#: of 7 SGLang starts, and the identical configuration relaunched came up both times: a third hang in a row is a
#: start that will not come up by itself, and the operator's.
START_RETRIES = 2


def start_stall_limit(phase: str, ceiling: float = START_STALL_S) -> float:
    """How long a start may say nothing new in `phase` before it is HUNG: START_STALL_FACTOR x the step's
    slowest healthy start (`START_STEP_SLOWEST_S`), never under START_STALL_FLOOR_S, never over `ceiling` —
    the deployment's `cluster_start_stall_s`, which also stands for every step with no measured basis."""
    seen = START_STEP_SLOWEST_S.get(phase)
    if seen is None or seen[1] < START_STALL_MIN_SEEN:
        return ceiling
    return min(ceiling, max(START_STALL_FLOOR_S, START_STALL_FACTOR * seen[0]))
#: A box with less than this free on EVERY recent sample, under a STARTING rank of ours, is running dry —
#: and the rank is ours to stop: broker_parrot's sentinel sheds only its own servers at its 512 MB kill
#: floor, which is this number (box-b, 2026-09-18 23:21-23:34Z: 0.1 GB for thirteen minutes). Never a
#: SERVING pair: one that answers at 1.7 GiB is working (live 2026-09-19), and a rank that really runs out
#: dies on its own, which the half-pair rule already answers.
KILL_FLOOR_GIB = 0.5

#: What the page may ask for. 48 seats is the recipe's measured ceiling on this pair (its
#: GDN recurrent state pins ~1.9 % of the KV pool per running request); 262144 is the model's
#: trained window.
MAX_NUM_SEQS_RANGE = (1, 48)
MAX_MODEL_LEN_RANGE = (1024, 262144)
#: The PREFILL BUDGET: prompt tokens the engine may take in one step (vLLM `--max-num-batched-tokens`), as
#: tp-rank bounds CLUSTER_MAX_NUM_BATCHED_TOKENS (`check_launch_settings`) — the same bound the catalog holds a
#: model's tuned budget to, read from there so the two cannot drift.
MAX_NUM_BATCHED_TOKENS_RANGE = MAX_NUM_BATCHED_TOKENS_CAP
#: The LONG-PREFILL THRESHOLD: the most prompt tokens ONE request may take in one step (vLLM
#: `--long-prefill-token-threshold`, tp-rank's CLUSTER_LONG_PREFILL_TOKEN_THRESHOLD). 0 is vLLM's own default,
#: OFF — a request's chunk is then bounded by the prefill budget alone, and tp-rank passes no flag at all.
#: vLLM refuses one past `max_model_len`; the ceiling is the budget's.
LONG_PREFILL_TOKEN_THRESHOLD_RANGE = (0, 65536)
#: Tokens the MTP drafter proposes per step (`num_speculative_tokens` in the speculative config). 0 turns
#: speculation OFF: tp-rank is sent an empty CLUSTER_SPECULATIVE_CONFIG and passes no --speculative-config.
NUM_SPECULATIVE_TOKENS_RANGE = NUM_SPECULATIVE_TOKENS_CAP
#: The KV POOL one rank pins (vLLM `--kv-cache-memory`, tp-rank's CLUSTER_KV_CACHE_MEMORY), in BYTES. The
#: floor is a pool too small to hold one full-context request; the ceiling is well past what any box here
#: can rest at — the START GATE refuses an unaffordable pin on the measured estimate, which is the check
#: that matters. A range cannot know a box's free memory; the gate can.
KV_CACHE_MEMORY_RANGE = (8_000_000_000, 96_000_000_000)
#: The pool's unit on the panel — whole decimal GB, as `--kv-cache-memory` takes it and the page writes it — and
#: so the step an `auto` pool is rounded DOWN to.
KV_CACHE_MEMORY_STEP = 1_000_000_000
#: The KV pool's third value besides a size and null: SIZED AT EACH START to what the boxes have left
#: (`memory_estimate.resolve_kv_auto`). The switch stores the size it resolved to, beside the choice
#: (`Desired.kv_cache_auto`), so everything downstream — the gate, tp-rank, `cluster.effective`, the
#: performance log — only ever sees a number.
KV_AUTO = "auto"
#: The DISK tier for cold KV (LMCache's `max_local_disk_size`), in GB. 0 is off, and off is the default:
#: see `cluster_exec.offload_yaml` for what is and is not proven about this path.
KV_OFFLOAD_GIB_RANGE = (0, 512)


def limits_view() -> dict:
    """`cluster.limits` in /lb/status — what the page may ask for, as the ONE place both publishers of the
    cluster view (the Hub before the loop's first tick, and the loop) read it from."""
    return {"max_num_seqs": list(MAX_NUM_SEQS_RANGE), "max_model_len": list(MAX_MODEL_LEN_RANGE),
            "max_num_batched_tokens": list(MAX_NUM_BATCHED_TOKENS_RANGE),
            "num_speculative_tokens": list(NUM_SPECULATIVE_TOKENS_RANGE),
            "kv_cache_memory": list(KV_CACHE_MEMORY_RANGE),
            "kv_offload_gib": list(KV_OFFLOAD_GIB_RANGE),
            "long_prefill_token_threshold": list(LONG_PREFILL_TOKEN_THRESHOLD_RANGE),
            # The PLE placement is true/false, not a span; published as the 0/1 it IS so the page can tell
            # "this LB takes it" from "it does not" the same way it does for every other knob.
            "ple_replicate": [0, 1],
            # Which checkpoint of the model: its abliterated twin (1) or the original (0). Published the same
            # way, for the same reason: a page that offers it to an LB without it would reload a pair for nothing.
            "uncensored": [0, 1]}


class _Unchanged:
    def __repr__(self) -> str:
        return "UNCHANGED"


#: `DesiredStore.set`'s "not given": keep what is stored. For the two engine knobs None is a VALUE — the
#: default — so "not given" needs a name of its own (`POST /lb/cluster`: a key left out vs a key set to null).
UNCHANGED = _Unchanged()

QUEUED = "queued"
RUNNING = "running"


def tag_for(generation: int) -> str:
    """A reservation's purpose tag, carried in its node_job's `inputs.purpose`."""
    return f"{TAG_PREFIX}#{int(generation)}"


def is_cluster_purpose(purpose) -> bool:
    return isinstance(purpose, str) and purpose.startswith(TAG_PREFIX + "#")


def generation_of(purpose) -> int | None:
    if not is_cluster_purpose(purpose):
        return None
    try:
        return int(purpose[len(TAG_PREFIX) + 1:])
    except ValueError:
        return None


# ── where the ranks run ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class ClusterHost:
    #: The broker lane label (`box-a-gpu`) — what a claim stamps into the reservation's `claimed_by`.
    label: str
    #: The physical box (`box-a`) — what the reservation's `force_box` pins.
    box: str
    #: Where the LB reaches it over SSH — its direct link from box-c (the `user@control` of its LB_CLUSTER_HOSTS entry,
    #: e.g. `user@192.0.2.10`). "" = no SSH channel: a node this LB only observes over HTTP (vllm_lb/topology.py).
    ssh_target: str = ""
    #: The address its rank talks to the other rank on — pair-link, the 2x200 Gb bond (`192.0.2.201`). "" = no peer.
    interconnect: str = ""


def parse_hosts(spec: str) -> tuple[ClusterHost, ...]:
    """`label=box=user@control=interconnect,…` → hosts, HEAD FIRST: the first entry is rank 0."""
    out = []
    for item in filter(None, (s.strip() for s in spec.split(","))):
        parts = [p.strip() for p in item.split("=", 3)]
        if len(parts) != 4 or not all(parts):
            raise ValueError(f"LB_CLUSTER_HOSTS entry must be label=box=user@control=interconnect, got {item!r}")
        out.append(ClusterHost(*parts))
    return tuple(out)


# ── desired ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Desired:
    mode: str = CLASSIC
    #: Bumped on every Start. A rank carries the generation it was launched for, so a rank of
    #: an older generation is replaced and a Stop/Start pair is a real restart.
    generation: int = 0
    max_num_seqs: int = 16
    max_model_len: int = 262144
    #: The cluster's ModelChoice: the ModelSpec id its ranks launch (vllm_lb/catalog.py). None only when the
    #: catalog could not load and has no default — and then no Start can be taken.
    model_choice: str | None = None
    #: THE OPERATOR'S ENGINE KNOBS. None is the default, resolved where it always has been and only at launch
    #: (`cluster_exec.resolved_launch`), so a pair nobody changed launches byte for byte as before:
    #:   * the prefill budget (vLLM --max-num-batched-tokens): the model's own (its tuned budget, SGLang's chunk),
    #:     else the FLEET's — clusters.<name>.max_num_batched_tokens, sent as CLUSTER_MAX_NUM_BATCHED_TOKENS on
    #:     every call (cluster_run.tp_rank_env); tp-rank defaults none;
    #:   * the drafter's depth: the ModelChoice's catalog LaunchProfile (qwen3.8-flash-next: 4). This stored
    #:     choice is the ONLY thing that may override a model's speculation — never an environment variable
    #:     (vllm_lb/__main__.py, TP_RANK_SETTINGS). 0 turns speculation off.
    #: A file written before they existed reads as None: the defaults, i.e. exactly what ran before.
    max_num_batched_tokens: int | None = None
    num_speculative_tokens: int | None = None
    #: WHERE THE PLE TABLE LIVES. None: the ModelChoice's LaunchProfile decides (hibrid48: replicated).
    #: False splits its rows across the ranks — 13.4 GiB less on every box, for one more exchange per
    #: decode step. tp-rank's own note says to adopt that only on a measured A/B, and none has been run
    #: here, so the operator chooses it deliberately or not at all.
    ple_replicate: bool | None = None
    #: THE KV POOL one rank pins, in bytes. None: the LaunchProfile's (hibrid48: 28 GB = 1,708,150 tokens).
    kv_cache_memory: int | None = None
    #: THE DISK TIER for cold KV, in GB. None or 0: off, and nothing about the launch changes.
    kv_offload_gib: int | None = None
    #: THE POOL IS `auto`: every Start sizes it afresh to what the boxes have left (`DesiredStore.set`'s
    #: `resolve_kv_auto`) and `kv_cache_memory` holds what THIS generation resolved to — the number the
    #: launch, `cluster.effective` and the performance log carry, so two auto pools are never one setup.
    kv_cache_auto: bool = False
    #: THE LONG-PREFILL THRESHOLD (vLLM --long-prefill-token-threshold): the most prompt tokens one request may
    #: take in one step. None or 0: off — vLLM's own default, and tp-rank is sent nothing, so a pair nobody
    #: changed launches byte for byte as before.
    long_prefill_token_threshold: int | None = None
    #: WHICH CHECKPOINT of the ModelChoice the ranks load: its abliterated twin (True, `catalog.UncensoredVariant`)
    #: or the original (False). Unlike the nullable knobs above this is RESOLVED BY THE START and stored as a
    #: bool — the model's own default (the twin, where it has one) unless the operator named one — because it
    #: changes WHAT is launched, not how, and the planner must read the box's availability answer for that very
    #: checkpoint without a catalog to resolve a default from (`availability_key`). None only in a file from
    #: before the knob, and such a file launched the original; a Stop keeps the stored choice, a model change
    #: hands it back to the new model's default.
    uncensored: bool | None = None
    #: THE KV CACHE'S ELEMENT TYPE for this run (vllm_lb/kv_dtype.py): None is the profile's own — a vLLM profile's
    #: flag, or none (vLLM's auto); an SGLang profile's engine_args, else fp8_e4m3 — and launches byte for byte as
    #: before the knob. Sized for ONE model, like the pool: a model change hands it back to null.
    kv_cache_dtype: str | None = None
    #: THE OPERATOR CHOSE THE ESTIMATE (docs/design/engines.md, "Unmeasured combination"): a model whose memory was
    #: never measured on this cluster's accelerator at this TP is REFUSED by default ("not measured on …"); True is
    #: the operator explicitly starting it sized by memory_estimate's computation from its config.json instead —
    #: labelled "estimated" wherever it shows. Never set by the LB on anyone's behalf: false unless a Start says so,
    #: kept by a Stop and a restart of the same model, handed back to false by a model change (like the pool).
    estimated: bool = False


def _flag(value) -> bool | None:
    """A stored true/false knob: null — or absent, in a file from before it existed — is the default."""
    if value is None:
        return None
    if not isinstance(value, bool):
        raise TypeError(f"a true/false setting must be true, false or null, not {value!r}")
    return value


def _override(value) -> int | None:
    """A stored engine knob: null — or absent, in a file from before the knobs — is the default; anything
    else is read as a whole number exactly as leniently as every other field of the switch (`int(...)`), so
    a knob adds no new way for the file to become unreadable — which reads as classic, and stops a pair."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"an engine setting must be a whole number or null, not {value!r}")
    return int(value)


def _whole(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _check_pool(pin, offload, *, pin_given: bool, offload_given: bool) -> None:
    """The KV pool knobs within what tp-rank takes. None, the default, always is, and so is `auto`, which a
    Start turns into a size within the range. The pin is bounded here only against the absurd: whether a box
    can AFFORD it is the start gate's call on the measured estimate (`memory_estimate`), because only the gate
    knows what the box has free."""
    if pin_given and pin is not None and pin != KV_AUTO:
        lo, hi = KV_CACHE_MEMORY_RANGE
        if not _whole(pin) or not lo <= pin <= hi:
            raise ValueError(f"kv_cache_memory must be a whole number of bytes between {lo} and {hi}, "
                             f"\"{KV_AUTO}\", or null for the model's default")
    if offload_given and offload is not None:
        lo, hi = KV_OFFLOAD_GIB_RANGE
        if not _whole(offload) or not lo <= offload <= hi:
            raise ValueError(f"kv_offload_gib must be a whole number of GB between {lo} and {hi} "
                             f"(0 turns the disk tier off), or null for off")


def _check_knobs(spec, batched, drafts, *, batched_given: bool, drafts_given: bool, threshold=None,
                 threshold_given: bool = False, max_model_len: int | None = None) -> None:
    """The engine knobs within what tp-rank and the ModelChoice really take — None, the default, always is.
    Checked only where the rest of the selection is: on a Start, or where a value is given, so a Stop is never
    blocked by what it did not touch."""
    if batched_given and batched is not None:
        lo, hi = MAX_NUM_BATCHED_TOKENS_RANGE
        if not _whole(batched) or not lo <= batched <= hi:
            raise ValueError(f"max_num_batched_tokens must be a whole number between {lo} and {hi}, "
                             f"or null for the box's default")
    if threshold_given and threshold is not None:
        lo, hi = LONG_PREFILL_TOKEN_THRESHOLD_RANGE
        if not _whole(threshold) or not lo <= threshold <= hi:
            raise ValueError(f"long_prefill_token_threshold must be a whole number between {lo} and {hi} "
                             f"(0 turns it off), or null for off")
        # vLLM's own refusal (SchedulerConfig: "cannot be greater than the max_model_len"), said before two
        # boxes load a model only to exit on it.
        if max_model_len is not None and threshold > max_model_len:
            raise ValueError(f"long_prefill_token_threshold ({threshold}) cannot be greater than max_model_len "
                             f"({max_model_len})")
    if drafts_given and drafts is not None:
        lo, hi = NUM_SPECULATIVE_TOKENS_RANGE
        if not _whole(drafts) or not lo <= drafts <= hi:
            raise ValueError(f"num_speculative_tokens must be a whole number between {lo} and {hi} "
                             f"(0 turns speculation off), or null for the model's default")
        profile = spec.profile(CLUSTER_MODE) if spec is not None else None
        if drafts > 0 and profile is not None and "num_speculative_tokens" in profile.refuses:
            raise ValueError(f"{spec.label}: {profile.refuses['num_speculative_tokens']} — num_speculative_tokens "
                             f"must be 0 or null")
        if drafts > 0 and profile is not None and not profile.speculative_config:
            raise ValueError(f"{spec.label} has no speculative decoding to set: num_speculative_tokens must "
                             f"be 0 or null")


def _engine_refusal(spec, profile, values: dict) -> str:
    """The first of `values` (knob → the operator's value) that `profile`'s ENGINE does not take, in words — "" when
    there is none. A knob at its default never is: null, and 0 for the disk tier and the long-prefill threshold
    (off). `catalog.ENGINE_REFUSES` says which knobs and why."""
    for knob, why in (profile.refuses if profile is not None else {}).items():
        value = values.get(knob)
        # 0 is a drafter depth's OFF too, and the only depth a profile with no speculative config has (the page sends
        # it for one): an engine that drafts by itself (TensorFold) refuses a depth, never the absence of one.
        off_at_zero = knob in ("kv_offload_gib", "long_prefill_token_threshold", "num_speculative_tokens")
        if value is None or (off_at_zero and value == 0):
            continue
        return f"{spec.label}: {why} — {knob} must be {'0 or null' if off_at_zero else 'null'}"
    return ""


def _call_fits(fits, ctx: int, pool_bytes, bytes_per_token, spec=None) -> int:
    """`fits` with as many of (context, pool, bytes per token, spec) as it takes: the LB's own takes all four
    (`cluster_defaults.fits_for`); a one- to three-argument callable, as a test may pass, still answers."""
    try:
        params = inspect.signature(fits).parameters.values()
        takes = 4 if any(p.kind is p.VAR_POSITIONAL for p in params) else sum(
            p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) for p in params)
    except (TypeError, ValueError):
        takes = 4
    return fits(*(ctx, pool_bytes, bytes_per_token, spec)[:max(1, min(4, takes))])


def _dtype(value) -> str | None:
    """A stored KV dtype: null (or absent, in a file from before the knob) is the profile's own; else a name."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"kv_cache_dtype must be a dtype name or null, not {value!r}")
    return value


def _check_pin_floor(spec, pin, ctx: int) -> None:
    """THE ENGINE'S FLOOR: a pool under ONE request's blocks at this context does not start — vLLM needs
    ceil(ctx / block_tokens) + F blocks for a request and dies at init without them (2026-10-02: an 8 GB pin
    on the 27B at 262,144, 340 blocks = 9.66 GB). Refused in the switch, where the operator can still change
    it. Only for a model whose block rule is recorded, and only for a pin that is a number."""
    memory = spec.memory if spec is not None else None
    if memory is None or memory.kv_blocks is None or not _whole(pin):
        return
    least = memory.kv_blocks.min_pool_bytes(ctx)
    if pin < least:
        raise ValueError(f"kv_cache_memory {pin / 1e9:.0f} GB < {least / 1e9:.1f} GB: one {ctx:,}-token request "
                         f"needs {memory.kv_blocks.per_request(ctx)} blocks × {memory.kv_blocks.block_bytes / 1e6:.2f} "
                         f"MB, and the engine does not start without them")


def _check_pin_ceiling(spec, pin) -> None:
    """THE PROFILE'S CEILING: a pin its LAUNCH would refuse to build is refused HERE, in the launch's own words with the
    maximum (`LaunchProfile.kv_pin_problem` — for SGLang the calibrated `mem_fraction_static` bound). Generation 187,
    2026-10-10 20:55:20Z: a 24 GB pin on the SGLang Flash-Next row was accepted here, a generation made and the serving
    pair torn down, and the launch refused it on every tick for six minutes ("SGLang KV pin 24,000,000,000 B needs
    mem_fraction_static 0.885, above this profile's safe 0.85 ceiling (maximum 19,918,363,626 B)"). Only for a pin
    that is a number: null is the model's own, `auto` is sized under the ceiling (`cluster_defaults.kv_auto`) and
    checked again once it is a number. Never clamped: the operator chose a number, and gets its refusal."""
    profile = spec.profile(CLUSTER_MODE) if spec is not None else None
    if profile is None or not _whole(pin):
        return
    problem = profile.kv_pin_problem(pin)
    if problem:
        raise ValueError(f"kv_cache_memory: {spec.label}: {problem}")


class DesiredStore:
    """The switch, on disk. Atomic writes; a missing or unreadable file is the classic default.

    Next to the switch it keeps the START ID: a random name written by every Start and kept by Stop.
    A reservation's identity is generation + start id + box (`leases.cluster_reservation_id`), so a
    generation number reused after the file is lost can never name an old, ended reservation.

    And the cluster's ModelChoice, validated by the ONE validator `ModelCatalog.choice` and bounded by that
    ModelSpec's ModelLimits. A file written before model choices existed reads as the catalog's cluster
    default — the model tp-rank launched until then."""

    def __init__(self, path: str, *, catalog: ModelCatalog | None = None, fits=None, archs=(), offers=None,
                 kv_dtypes=None) -> None:
        self.path = path
        #: The catalog rows this deployment offers (fleet `choices`), as row keys; None = every row. A Start of any
        #: other is refused, naming them.
        self.offers = tuple(offers) if offers is not None else None
        #: The KV dtypes this deployment runs (fleet `kv_cache_dtypes`); None = every one. A Start that would run another
        #: — asked for, or the row's own — is refused, naming them.
        self.kv_dtypes = tuple(kv_dtypes) if kv_dtypes is not None else None
        #: The `arch` of every card this cluster SERVES on (the fleet's, `cluster_defaults.desired_store`): what a KV
        #: dtype the engine runs only on some cards is checked against. () = none named, and such a dtype is refused.
        self.archs = tuple(archs)
        #: The one ModelCatalog this LB loaded; the cluster loop reads the same instance.
        self.catalog = catalog if catalog is not None else ModelCatalog.load()
        self._lock = threading.Lock()
        default = self.catalog.default(CLUSTER_MODE)
        #: The default row's own tuned seats and context (every cluster row states them: `catalog._spec`) — what the
        #: cold start offers; only a model choice the catalog does not know falls back to them, and such a choice no
        #: Start can take. No catalog default (the catalog did not load): `Desired`'s own field values, and no
        #: model_choice — nothing can be started from that either.
        base = default.profile(CLUSTER_MODE).tuned if default is not None else None
        self._base = (Desired(max_num_seqs=base.max_num_seqs, max_model_len=base.max_model_len, model_choice=default.id)
                      if base is not None else Desired(model_choice=None))
        self._default = self.defaults_for(self._base.model_choice)
        #: `max_model_len -> the greatest max_num_seqs its KV pool holds` (cluster_defaults.fits_for),
        #: or None to store whatever is asked for. Consulted on the WRITE path ONLY: a read never
        #: rewrites what is stored, so a pair already running is published exactly as it was
        #: launched (worklog/vllm-cluster-defaults.md C3).
        self._fits = fits
        self.last_error = ""
        #: Why the last write did not store the concurrency it was asked for; "" when it did.
        self.last_clamp = ""

    def load(self) -> Desired:
        with self._lock:
            return self._read()[0]

    def current(self) -> tuple[Desired, str | None]:
        """The switch and its start id, read together."""
        with self._lock:
            return self._read()

    def _read(self) -> tuple[Desired, str | None]:
        try:
            with open(self.path, "rb") as fh:
                raw = json.load(fh)
            raw = retire_blueprint_field(self.path, raw)
            choice = raw.get("model_choice")
            out = Desired(mode=CLUSTER_MODE if raw.get("mode") == CLUSTER_MODE else CLASSIC,
                          generation=int(raw.get("generation", 0)),
                          max_num_seqs=int(raw.get("max_num_seqs", self._default.max_num_seqs)),
                          max_model_len=int(raw.get("max_model_len", self._default.max_model_len)),
                          model_choice=str(choice) if choice else self._default.model_choice,
                          max_num_batched_tokens=_override(raw.get("max_num_batched_tokens")),
                          num_speculative_tokens=_override(raw.get("num_speculative_tokens")),
                          ple_replicate=_flag(raw.get("ple_replicate")),
                          kv_cache_memory=_override(raw.get("kv_cache_memory")),
                          kv_offload_gib=_override(raw.get("kv_offload_gib")),
                          kv_cache_auto=bool(_flag(raw.get("kv_cache_auto"))),
                          long_prefill_token_threshold=_override(raw.get("long_prefill_token_threshold")),
                          uncensored=_flag(raw.get("uncensored")),
                          kv_cache_dtype=_dtype(raw.get("kv_cache_dtype")),
                          estimated=bool(_flag(raw.get("estimated"))))
            start_id = raw.get("start_id")
            self.last_error = ""
            return out, (str(start_id) if start_id else None)
        except FileNotFoundError:
            self.last_error = ""
            return self._default, None
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            self.last_error = f"{self.path}: {type(exc).__name__}: {exc}"[:200]
            return self._default, None

    def set(self, *, enabled: bool, max_num_seqs: int | None = None, max_model_len: int | None = None,
            model_choice: str | None = None, restart: bool = False,
            max_num_batched_tokens=UNCHANGED, num_speculative_tokens=UNCHANGED,
            ple_replicate=UNCHANGED, kv_cache_memory=UNCHANGED, kv_offload_gib=UNCHANGED,
            resolve_kv_auto=None, long_prefill_token_threshold=UNCHANGED, uncensored=UNCHANGED,
            kv_cache_dtype=UNCHANGED, admit=None, estimated=UNCHANGED, _check_only: bool = False) -> Desired:
        """The switch. Start bumps the generation; the ModelChoice and the settings are frozen while it is on.

        A Start needs its whole selection valid: the ModelChoice (`ModelCatalog.choice` — UnknownModelChoice for
        an id the catalog cannot launch in the cluster) and the settings within that ModelSpec's own limits. A Stop
        checks only what it is given, so a choice the catalog no longer has, or a catalog that could not load,
        never blocks one.

        `restart` is the operator saying "reload the pair at these settings" — the one way the selection may
        change under a running cluster. It bumps the generation and keeps the mode `cluster`, so the handover
        never passes through classic, which is precisely what hands the GB10 boxes back to the fleet: live on
        2026-09-16 a Stop released both reservations and the lane claimed a pool `llm_chat` job in the same
        second, re-loading ollama on the box generation 11 was about to need
        (worklog/vllm-cluster-handover.md D1). Without it a changed selection is still refused, so no stray
        Start can replace a serving pair by accident (the 00:30:32Z incident).

        THE ENGINE KNOBS — `max_num_batched_tokens`, the prefill budget, `num_speculative_tokens`, the
        drafter's depth (0 = off), and `long_prefill_token_threshold`, one request's share of a step (0 = off) —
        are part of that selection, frozen and restarted the same way. For them None is a value, the default
        (`Desired`); leaving one out (UNCHANGED) keeps what is stored.

        THE KV POOL MAY BE `auto` (KV_AUTO). It is a choice like any other — frozen under a running pair, kept by
        a Stop, applied by a restart — but its SIZE belongs to the Start: every Start that makes a generation
        while it is the stored choice calls `resolve_kv_auto(the selection)` for the pin, and stores THAT, so the
        gate, the launch and `cluster.effective` see one number per generation. A resolver that refuses
        (ValueError) refuses the Start; a Start at auto with no resolver to ask is refused, never guessed.

        `admit(the selection)`, given, is asked last on every Start: a ValueError refuses it with nothing stored —
        the memory a box at rest cannot give (`cluster_defaults.memory_admission`), said before the Start.

        WHICH CHECKPOINT (`uncensored`) is part of the selection too — frozen under a running pair, kept by a
        Stop, reset by a model change — and a Start RESOLVES it: null becomes the model's own default (its twin,
        where it has one: `ModelSpec.uncensored_default`) and the file holds true or false from then on. True for
        a model with no twin is refused where it is given and at a Start that would launch it; the one thing it
        never does is launch the original in the twin's place.

        `_check_only` is `check`: the whole validation of a FRESH Start, nothing sized from the boxes, nothing
        admitted, nothing written."""
        with self._lock:
            if _check_only:
                current, start_id = replace(self._default, mode=CLASSIC), None
            else:
                current, start_id = self._read()
            choice = current.model_choice if model_choice is None else model_choice
            # A CHANGED MODEL TAKES ITS OWN: every knob that is sized for ONE model goes back to that model's
            # default unless this call names it — its KV pool (and `auto`), where its PLE table sits, its
            # drafter's depth and its PREFILL BUDGET to null, and its concurrency and context, which are never
            # null, to its measured-best (`defaults_for`). A 30 GB pool, a split table, a depth, a budget and
            # a seat count chosen for one model would otherwise ride into the first start of the next — which
            # is exactly what the operator had to undo by hand, three controls at a time, until 2026-10-05.
            # Only the two that are nobody's model in particular (the long-prefill threshold, off, and the
            # disk tier, off) are the operator's for the pair and stay.
            stored = current
            if choice != current.model_choice:
                offered = self.defaults_for(choice)
                current = replace(current, num_speculative_tokens=None, ple_replicate=None,
                                  kv_cache_memory=None, kv_cache_auto=False, max_num_batched_tokens=None,
                                  max_num_seqs=offered.max_num_seqs, max_model_len=offered.max_model_len,
                                  uncensored=None, kv_cache_dtype=None, estimated=False)
            seqs = current.max_num_seqs if max_num_seqs is None else int(max_num_seqs)
            ctx = current.max_model_len if max_model_len is None else int(max_model_len)
            batched = (current.max_num_batched_tokens if max_num_batched_tokens is UNCHANGED
                       else max_num_batched_tokens)
            drafts = (current.num_speculative_tokens if num_speculative_tokens is UNCHANGED
                      else num_speculative_tokens)
            threshold = (current.long_prefill_token_threshold if long_prefill_token_threshold is UNCHANGED
                         else long_prefill_token_threshold)
            ple = current.ple_replicate if ple_replicate is UNCHANGED else ple_replicate
            # The pool: a size, null (the model's), or AUTO — whose stored size is only the last Start's.
            if kv_cache_memory is UNCHANGED:
                auto, pin = current.kv_cache_auto, current.kv_cache_memory
            elif kv_cache_memory == KV_AUTO:
                auto, pin = True, current.kv_cache_memory
            else:
                auto, pin = False, kv_cache_memory
            offload = current.kv_offload_gib if kv_offload_gib is UNCHANGED else kv_offload_gib
            checkpoint = current.uncensored if uncensored is UNCHANGED else uncensored
            dtype = current.kv_cache_dtype if kv_cache_dtype is UNCHANGED else kv_cache_dtype
            sized_by_estimate = current.estimated if estimated is UNCHANGED else estimated
            if not isinstance(sized_by_estimate, bool):
                raise ValueError("estimated must be true or false — true starts a model never measured on these "
                                 "cards sized by the estimate, the operator's explicit choice")
            # THE ONE VALIDATOR of a choice, resolution included (catalog.choice): an unmeasured model is refused
            # unless the operator explicitly chose the estimate for this Start.
            spec = (self.catalog.choice(CLUSTER_MODE, choice, estimated=sized_by_estimate)
                    if enabled or model_choice is not None else None)
            lo_seqs, hi_seqs = spec.limits.max_num_seqs if spec is not None else MAX_NUM_SEQS_RANGE
            hi_ctx = spec.limits.max_model_len if spec is not None else MAX_MODEL_LEN_RANGE[1]
            whose = f" for {spec.label}" if spec is not None else ""
            if (enabled or max_num_seqs is not None) and not lo_seqs <= seqs <= hi_seqs:
                raise ValueError(f"max_num_seqs must be between {lo_seqs} and {hi_seqs}{whose}")
            if (enabled or max_model_len is not None) and not MAX_MODEL_LEN_RANGE[0] <= ctx <= hi_ctx:
                raise ValueError(f"max_model_len must be between {MAX_MODEL_LEN_RANGE[0]} and {hi_ctx}{whose}")
            _check_knobs(spec, batched, drafts,
                         batched_given=enabled or max_num_batched_tokens is not UNCHANGED,
                         drafts_given=enabled or num_speculative_tokens is not UNCHANGED,
                         threshold=threshold,
                         threshold_given=enabled or long_prefill_token_threshold is not UNCHANGED,
                         max_model_len=ctx)
            _check_pool(KV_AUTO if auto else pin, offload,
                        pin_given=enabled or kv_cache_memory is not UNCHANGED,
                        offload_given=enabled or kv_offload_gib is not UNCHANGED)
            if (enabled or ple_replicate is not UNCHANGED) and ple is not None and not isinstance(ple, bool):
                raise ValueError("ple_replicate must be true, false or null for the model's default")
            if (enabled or uncensored is not UNCHANGED) and checkpoint is not None and not isinstance(checkpoint, bool):
                raise ValueError("uncensored must be true, false or null for the model's default checkpoint")
            # THE KV DTYPE MUST BE ONE THE MODEL'S ENGINE IMAGE RUNS ON THESE CARDS (`ModelSpec.kv_cache_dtype_problem`):
            # refused where it is given and where a stored one would launch, in the image's own words.
            dtype_subject = spec if spec is not None else self.catalog.get(choice)
            # At the DEPTH this selection drafts at (null: the profile's own) — an FP4 pool is refused beside a drafter.
            problem = (dtype_subject.kv_cache_dtype_problem(dtype, self.archs, drafts)
                       if dtype is not None and dtype_subject is not None else "")
            if problem and (enabled or kv_cache_dtype is not UNCHANGED or num_speculative_tokens is not UNCHANGED):
                raise ValueError(f"kv_cache_dtype: {dtype_subject.label}: {problem}")
            # Every KV figure below at the run's dtype — none at all for a stored one this model cannot run (a Stop
            # that does not touch it): a figure at another dtype would be a different pool's.
            priced = None if problem or dtype_subject is None else dtype_subject.at_kv_cache_dtype(_runs_kv(dtype_subject, dtype))
            # THE POOL MUST HOLD ONE REQUEST of the chosen context — the pin given, else the model's own —
            # checked where the selection is: a Start, or a pin or a context given.
            if enabled or kv_cache_memory is not UNCHANGED or max_model_len is not None:
                subject_pin = priced
                subject_profile = subject_pin.profile(CLUSTER_MODE) if subject_pin is not None else None
                _check_pin_floor(subject_pin, pin if (not auto and pin is not None) else
                                 (subject_profile.kv_cache_memory if subject_profile is not None else None), ctx)
                # …AND MUST NOT BE PAST WHAT THE PROFILE CAN TAKE: the operator's number, where it is given and where a
                # stored one would launch — the launch's own refusal, said here, before any generation is made.
                if not auto and pin is not None:
                    _check_pin_ceiling(subject_pin, pin)
            # WHAT THE MODEL CANNOT TAKE: a PLE placement for a model with no PLE table (its LaunchProfile says
            # null — there is nothing to place), and an `auto` pool for a model whose footprint nobody
            # measured (auto would size the pool from arithmetic). Refused where it is GIVEN — on a save too,
            # against the model the switch holds — and where a STORED one would launch: a Start or a restart
            # that makes a generation, checked below, past the idempotent Start, so a pair already running
            # with a value from an older file is still returned as it is, and a Stop is never blocked.
            subject = spec if spec is not None else self.catalog.get(choice)
            profile = subject.profile(CLUSTER_MODE) if subject is not None else None
            no_table = (f"{subject.label} has no PLE table to place: ple_replicate must be null"
                        if ple is not None and profile is not None and profile.ple_replicate is None else "")
            unsized = (f"kv_cache_memory \"{KV_AUTO}\": {subject.label} is not calibrated: choose a size"
                       if auto and profile is not None and not subject.calibrated else "")
            if no_table and ple_replicate is not UNCHANGED:
                raise ValueError(no_table)
            # A TWIN FOR A MODEL THAT HAS NONE: refused where it is given, and below where a stored one would
            # launch — never resolved to the original (`ModelSpec.launched_from` refuses the same way).
            no_twin = (f"{subject.id} has no uncensored counterpart: uncensored must be false or null"
                       if checkpoint is True and subject is not None and subject.uncensored is None else "")
            if no_twin and uncensored is not UNCHANGED:
                raise ValueError(no_twin)
            # WHAT THE ENGINE DOES NOT TAKE (`catalog.ENGINE_REFUSES`: SGLang's pool, disk tier, long-prefill
            # threshold, PLE placement) is refused like a placement for a model with no table: where it is given,
            # and below where a stored one would launch — never accepted and dropped.
            # TensorFold's prefill budget and drafter depth too (it chooses both itself).
            asked = {"kv_cache_memory": KV_AUTO if auto else pin, "kv_offload_gib": offload,
                     "long_prefill_token_threshold": threshold, "ple_replicate": ple,
                     "max_num_batched_tokens": batched, "num_speculative_tokens": drafts}
            named = {"kv_cache_memory": kv_cache_memory, "kv_offload_gib": kv_offload_gib,
                     "long_prefill_token_threshold": long_prefill_token_threshold, "ple_replicate": ple_replicate,
                     "max_num_batched_tokens": max_num_batched_tokens,
                     "num_speculative_tokens": num_speculative_tokens}
            dropped = _engine_refusal(subject, profile, asked)
            given = _engine_refusal(subject, profile,
                                    {k: v for k, v in asked.items() if named[k] is not UNCHANGED})
            if given:
                raise ValueError(given)
            if unsized and kv_cache_memory is not UNCHANGED:
                raise ValueError(unsized)
            changed = ((choice, seqs, ctx, batched, drafts, ple, KV_AUTO if auto else pin, offload, threshold,
                        checkpoint, dtype, sized_by_estimate)
                       != (stored.model_choice, stored.max_num_seqs, stored.max_model_len,
                           stored.max_num_batched_tokens, stored.num_speculative_tokens,
                           stored.ple_replicate,
                           KV_AUTO if stored.kv_cache_auto else stored.kv_cache_memory, stored.kv_offload_gib,
                           stored.long_prefill_token_threshold, stored.uncensored, stored.kv_cache_dtype,
                           stored.estimated))
            if enabled and current.mode == CLUSTER_MODE and not restart:
                if changed:
                    raise ValueError("stop the cluster before changing its model choice, checkpoint, concurrency, "
                                     "context, prefill budget, speculative tokens, PLE placement, KV pool, KV cache "
                                     "dtype, disk offload or long-prefill threshold, or restart it at the new ones (both "
                                     "ranks would have to reload the model)")
                return current
            if enabled and (no_table or unsized or dropped or no_twin):
                raise ValueError(no_table or unsized or dropped or no_twin)
            # THE CHECKPOINT THIS START LAUNCHES, resolved now and stored as a bool (`Desired.uncensored`).
            if enabled and checkpoint is None:
                checkpoint = subject.uncensored_default
            if enabled and self.offers is not None and availability_key(choice, checkpoint) not in self.offers:
                raise ValueError(f"{availability_key(choice, checkpoint)} is not offered here: "
                                 f"{', '.join(self.offers)} (the fleet's choices)")
            runs_kv = dtype or (subject.profile(CLUSTER_MODE).kv_cache_dtype if subject is not None else None) or "auto"
            if enabled and self.kv_dtypes is not None and runs_kv not in self.kv_dtypes:
                raise ValueError(f"KV dtype {runs_kv} is not run here: {', '.join(self.kv_dtypes)} (the fleet's "
                                 f"kv_cache_dtypes)")
            # ADVICE, NOT A LIMIT. What the measurement recommends is not what the operator may have:
            # asking for 8 and being handed 6 is the gateway overruling the person running it. Past the KV
            # pool vLLM ADMITS fewer sequences — it queues the rest, it does not OOM and it does not fail
            # to launch (worklog/vllm-cluster-ram-headroom.md §5.2) — so a tight setting earns a warning on
            # the page, and the value asked for is what gets stored. The limits tp-rank and the model spec
            # really enforce are checked above, and they still bind.
            if enabled and auto and not _check_only:
                # AUTO IS SIZED NOW, for the selection this Start stores — never by the loop afterwards, so the
                # number the operator was shown is the number the gate weighs and the ranks are sent.
                if resolve_kv_auto is None:
                    raise ValueError(f"kv_cache_memory \"{KV_AUTO}\" is sized from what the boxes report, and "
                                     f"nothing was given to ask them: choose a size")
                pin = resolve_kv_auto(Desired(
                    mode=CLUSTER_MODE, generation=current.generation + 1, max_num_seqs=seqs, max_model_len=ctx,
                    model_choice=choice, max_num_batched_tokens=batched, num_speculative_tokens=drafts,
                    ple_replicate=ple, kv_cache_memory=None, kv_offload_gib=offload, kv_cache_auto=True,
                    long_prefill_token_threshold=threshold, uncensored=checkpoint, kv_cache_dtype=dtype))
                _check_pool(pin, offload, pin_given=True, offload_given=False)   # a size, in range — or no Start
                _check_pin_ceiling(subject, pin)                                # …and one the launch builds
            if not _check_only:
                self._note_if_tight(seqs, ctx, pin, priced)
            if enabled:
                new = Desired(mode=CLUSTER_MODE, generation=current.generation + 1,
                              max_num_seqs=seqs, max_model_len=ctx, model_choice=choice,
                              max_num_batched_tokens=batched, num_speculative_tokens=drafts,
                              ple_replicate=ple, kv_cache_memory=pin, kv_offload_gib=offload,
                              kv_cache_auto=auto, long_prefill_token_threshold=threshold,
                              uncensored=checkpoint, kv_cache_dtype=dtype, estimated=sized_by_estimate)
                if _check_only:
                    return new
                if admit is not None:
                    admit(new)
                start_id = uuid.uuid4().hex
            else:
                new = replace(current, mode=CLASSIC, max_num_seqs=seqs, max_model_len=ctx, model_choice=choice,
                              max_num_batched_tokens=batched, num_speculative_tokens=drafts,
                              ple_replicate=ple, kv_cache_memory=pin, kv_offload_gib=offload,
                              kv_cache_auto=auto, long_prefill_token_threshold=threshold,
                              uncensored=checkpoint, kv_cache_dtype=dtype, estimated=sized_by_estimate)
            self._save(new, start_id)
            return new

    def check(self, **settings) -> Desired:
        """THE START'S OWN VALIDATION, WITHOUT THE START: what a fresh Start of `settings` would store — the
        ModelChoice and its resolution, the limits, the engine's refusals, the KV dtype on these cards, the pin's floor
        and ceiling, the fleet's offers and KV dtypes — or the ValueError the Start would raise. Nothing is read from
        the switch's file, sized from the boxes (`auto` stays a choice), admitted or written. One validator: the setup
        wizard's first Start (fleet.start_checker; deploy/setup/render.py) passes exactly when the switch would take it."""
        return self.set(enabled=True, _check_only=True, **settings)

    @property
    def defaults(self) -> Desired:
        """What an unwritten switch offers: the catalog's cluster default model, at ITS OWN measured-best
        concurrency and context — `cluster.defaults` on the page, beside what the engine knobs resolve to."""
        return self._default

    def defaults_for(self, model_choice) -> Desired:
        """WHAT THIS MODEL IS OFFERED when the operator has chosen nothing: its own start input — its cluster
        LaunchProfile's `TunedSetup`, which every cluster row states the seats and context in (`catalog._spec`;
        the gateway-wide LB_CLUSTER_MAX_NUM_SEQS / LB_CLUSTER_MAX_MODEL_LEN are retired, 2026-10-29). Only a model
        choice the catalog does not know — which no Start can take — reads the default row's.

        The two nullable knobs stay NULL here: the prefill budget and the drafter's depth are resolved at
        launch, from the same `TunedSetup`, by `cluster_exec.resolved_launch` — so one tuned figure is never
        copied into the switch's file, where a later catalog change could not reach it. The concurrency and
        the context are always numbers in `Desired`, so for those two this IS the resolution.

        ONE method, read by the cold-start default, by a model switch (`set`) and by what the loop publishes
        as `cluster.defaults`, so the number the page shows for a model and the number a Start of it stores
        can never be two different things.
        """
        spec = self.catalog.get(model_choice)
        profile = spec.profile(CLUSTER_MODE) if spec is not None else None
        tuned = profile.tuned if profile is not None else None
        stated = tuned.stated if tuned is not None else {}
        return replace(self._base,
                       model_choice=model_choice if model_choice is not None else self._base.model_choice,
                       max_num_seqs=stated.get("max_num_seqs", self._base.max_num_seqs),
                       max_model_len=stated.get("max_model_len", self._base.max_model_len),
                       # Which checkpoint a Start of this model launches when the operator names none.
                       uncensored=spec.uncensored_default if spec is not None else None)

    def _note_if_tight(self, seqs: int, ctx: int, pool_bytes: int | None = None, spec=None) -> None:
        """Warn when `seqs` is past what the KV pool holds at `ctx`, and store what was ASKED FOR anyway.

        The operator must never be silently given something other than what they asked for — and that cuts
        both ways: reducing 8 to 6 because a measurement recommends 6 is the same failure as ignoring them.
        Past the pool vLLM admits fewer sequences at a time and queues the rest; it does not OOM and it does
        not refuse to launch, because the pool is paged and shared (worklog/vllm-cluster-ram-headroom.md
        §5.2). So this is advice on the page, not a limit. The limits that really bind — the model spec's
        and tp-rank's own 1..48 — are enforced by the validation above.
        """
        self.last_clamp = self._clamp(seqs, ctx, pool_bytes, spec)

    def published_clamp(self, desired: Desired, *, engine=None) -> str:
        """THE SENTENCE THE PAGE SHOWS, worked out for the selection that is STORED — with the engine's
        own figure for the pool it is serving from (`engine`, a `cluster_defaults.EnginePool`) whenever
        one describes that selection, and the arithmetic only when none does.

        Not `last_clamp`, which is what the last WRITE through this object worked out. Three things were
        wrong with publishing that: it is the arithmetic even while the engine is running and reporting
        its own pool — the page drew "fits 2 @ 262,144" from the engine's 629,682 tokens beside "KV pool
        holds 3", off the same card, on generation 109 (worklog/gateway-figures-audit.md §2); it is empty
        after an LB restart under a serving pair, which loses the warning entirely; and it describes the
        last write even when that write was refused. This is a pure read — nothing stored is rewritten,
        the C3 invariant the `_fits` note above states.
        """
        spec = self.catalog.get(desired.model_choice)
        spec = spec.at_kv_cache_dtype(_runs_kv(spec, desired.kv_cache_dtype)) if spec is not None else None
        return self._clamp(desired.max_num_seqs, desired.max_model_len, desired.kv_cache_memory, spec,
                           engine=engine)

    def _clamp(self, seqs: int, ctx: int, pool_bytes: int | None, spec, engine=None) -> str:
        """The warning itself, or "" when `seqs` fits: ONE sentence, from ONE seat count."""
        if self._fits is None and engine is None:
            return ""
        # A pool the engine reserves one window a seat holds every seat at the context it starts with.
        profile = spec.profile(CLUSTER_MODE) if spec is not None else None
        if profile is not None and profile.pool_per_seat:
            return ""
        # The pool THE PAIR WILL LAUNCH WITH — the operator's pin, else the chosen model's own — at THAT
        # model's cost per token: a 35 GB pin read against the 28 GB constant said "6" on the page while the
        # engine itself reported 8.14, and a dense model's 32,768 B tokens are not the hybrid's 16,392.
        pool = pool_bytes or (profile.kv_cache_memory if profile is not None else None)
        memory = spec.memory if spec is not None else None
        per_token = memory.kv_bytes_per_token if memory is not None else None
        if engine is not None:
            # MEASURED BEATS DERIVED. Everything `fits` can do is extrapolate this pin through constants
            # taken at another one; the engine printed the answer for the pool it actually allocated.
            room = engine.seats(ctx, memory.kv_blocks if memory is not None else None,
                                seqs_range=MAX_NUM_SEQS_RANGE)
        else:
            room = _call_fits(self._fits, ctx, pool, per_token, spec)
        if seqs <= room:
            return ""
        return (f"KV pool holds {room} x {ctx:,}-token requests: concurrency {seqs} runs "
                f"{room} at a time and queues the rest")

    def _save(self, desired: Desired, start_id: str | None) -> None:
        # Whole or not at all, through the ONE atomic writer every operator switch uses.
        write_json(self.path, {**asdict(desired), "start_id": start_id, "updated_at": time.time()})
        self.last_error = ""


# ── observed ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RankObs:
    present: bool
    running: bool
    gen: int | None
    exit_code: int | None
    #: The ModelChoice this rank was LAUNCHED for, read off its own label — never what we want now.
    #: None: a rank from before model choices existed, or one labelled with something that is not ours.
    model_choice: str | None = None
    #: WHY it is not running, in the box's own words — the memory guard's verdict, else the last fault
    #: line in the engine's log (tp-rank `observe_exit_reason`). Empty from a box that does not report
    #: one, and then the planner says exactly what it said before.
    exit_reason: str = ""
    #: The BOX REBOOTED under this rank: its container started before the box's current boot (tp-rank's
    #: `started_at` against the box's `boot_at`) and it is not running. Docker records 255 for a container
    #: killed by a host going away, but the code is not the evidence — the two clocks are. A lost rank did not
    #: EXIT, and nothing in its log says why the box went; the planner relaunches the pair instead of keeping
    #: a post-mortem of a power cut (2026-09-19 08:07Z, 2026-10-10 20:12Z: 3.5 h of `failed` with both GB10 boxes
    #: reserved and idle, until the operator did the Stop+Start by hand).
    lost: bool = False


@dataclass(frozen=True)
class Reservation:
    """One GB10 box's reservation, as broker_parrot's rows say it is."""
    #: The `workflow_runs` row we own — what a release cancels.
    reservation_id: str
    job_id: str
    #: The node_job's own status: queued | running | completed | failed | cancelled | skipped.
    status: str
    #: The lane that claimed it (a host_label); None while queued.
    claimed_by: str | None = None
    #: Our run is still `running`. A released reservation — or one whose run someone closed — is not.
    run_open: bool = True


def _alive(r: Reservation) -> bool:
    return r.run_open and r.status in (QUEUED, RUNNING)


#: The closed set of reasons a box is not free for the pair yet — the codes the page reads.
OLLAMA_RESIDENT = "ollama_resident"
FLEET_JOBS = "fleet_jobs"
BROKER_VLLM = "broker_vllm"
MEM_SHORT = "mem_short"
NO_LANE = "no_lane"


@dataclass(frozen=True)
class RamHolder:
    """Something holding RAM the pair needs: a model this box's own ollama still has resident, as its
    `/api/ps` said through `tp-rank status`.

    This is the LB's answer to "why does box-b wait", which the page could not give: the unload works,
    but a chat client on the box reloads it, and 17.6 GiB of qwen3.8 is the difference between a 103.2
    GiB rank fitting and not (worklog/vllm-cluster-hold-robustness.md §2, §6)."""
    name: str
    gib: float
    expiry: str = ""

    def sentence(self) -> str:
        return f"ollama holds {self.name}, {self.gib:.1f} GiB"

    def view(self) -> dict:
        return {"name": self.name, "gib": round(self.gib, 1), "expiry": self.expiry}


@dataclass(frozen=True)
class BoxObs:
    #: The current generation's reservation; None = broker_parrot holds no row for it (never asked).
    reservation: Reservation | None = None
    #: Whether broker_parrot's rows could be read at all. Nothing below is judged without them.
    reservation_known: bool = True
    #: Still-open reservations of another generation on this box — returned once no rank stands here.
    stale: tuple[Reservation, ...] = ()
    #: `requested_by` of an OFF `worker_controls` row — the OPERATOR's park. READ for the words, never acted on.
    parked_by: str | None = None
    #: The fleet's OTHER running gpu jobs claimed by this box's lane (our reservations left out). None = unreadable.
    running_jobs: int | None = None
    #: The rank container, from `tp-rank status`. None = the executor could not answer.
    rank: RankObs | None = None
    broker_vllm_running: bool | None = None
    #: The NEWEST MemAvailable observation — what the page shows.
    mem_available_gib: float | None = None
    #: What this box can be COUNTED ON to have: the LOWEST MemAvailable over the last
    #: `cluster_mem_samples` observations. None = fewer samples than that, or a read that failed — and
    #: then nothing launches here, because one sample of a number that swings 17.6 GiB is not evidence
    #: (worklog/vllm-cluster-hold-robustness.md §3, L3). The planner gates on THIS, never on the newest.
    mem_stable_gib: float | None = None
    #: MemTotal, as tp-rank's status reports it: what the fleet's `ram_usable_fraction` is a share of.
    mem_total_gib: float | None = None
    #: How long the current reservation has been back in the queue while this box's rank stands.
    requeued_s: float | None = None
    #: What this box HAS: one `Availability` per ModelSpec id, as the box itself answered
    #: (`cluster_exec.availability`). None = never asked, or it could not be asked — and a box that has
    #: not said is never taken from the fleet on a guess.
    availability: dict | None = None
    #: THE REQUIREMENT: what THIS selection costs on this box, measured
    #: (`memory_estimate.MemoryEstimate.needed_now_gib` — model, context, concurrency). None = the
    #: selection cannot be estimated at all, and then the Start is REFUSED naming `needs_why` — no flat
    #: requirement stands in (LB_CLUSTER_MIN_MEM_GIB is retired, 2026-10-29; worklog/vllm-cluster-handover.md H3).
    needs_gib: float | None = None
    #: WHY it cannot be estimated, in the estimate's own words (`memory_estimate.estimate_memory`'s refusal) — "" when
    #: `needs_gib` is a number.
    needs_why: str = ""
    #: WHICH KNOB is adding to that requirement, in the LB's own words — `MemoryEstimate.needs_note`, "" when
    #: no setting of the operator's is. A refusal the operator cannot act on is only half a refusal: "box-a:
    #: 116 GiB free, need 120" does not say that lowering the prefill budget from 16,384 to 8,192 is the
    #: difference, and on 2026-09-30 that was the whole difference between a pair that serves and two boxes
    #: that had to be power-cycled.
    needs_note: str = ""
    #: THE HANDOVER. What this box had free BEFORE the rank that has since come down launched on it — its
    #: `mem_at_launch`, kept past the teardown. None = no rank of ours has come down here since the LB
    #: started. A start that fits only because the previous engine has finished letting go is a start that
    #: can hang the box, and on 2026-09-30 the fatal NVRM `Out of memory [NV_ERR_NO_MEMORY]` landed in the
    #: minute one 16,384 generation ended and the next began. The memory coming back is what says the
    #: handover is done, so a box still short of it is short for a reason worth naming.
    released_from_gib: float | None = None
    #: The HIGHEST MemAvailable over the same observations — what the kill floor is judged on, so that one
    #: dip never stops a pair. None = too few samples, or a read that failed.
    mem_peak_gib: float | None = None
    #: What is still holding this box's RAM, as the box itself said (its own ollama's `/api/ps`, through
    #: tp-rank's status). None = it could not be asked — never an empty tuple, which means "nothing is
    #: resident" (worklog/vllm-cluster-hold-robustness.md §6).
    holds: tuple[RamHolder, ...] | None = None
    #: What this box's GPU is doing, as `tp-rank status` read it off the driver (SM fill, power, clock,
    #: temperature — `saturation.gpu_view`). None = no nvidia-smi there, the driver did not answer, or the
    #: telemetry is switched off. Display only: nothing plans on it, so a box that cannot be asked
    #: costs a line on the page and no decision.
    gpu: dict | None = None
    #: The TP link under this rank, as RATES the loop derived from two readings of tp-rank's RDMA byte
    #: counters: {tx_gbps, rx_gbps, link_gbps}. None = no reading yet, the first one, or a counter reset.
    network: dict | None = None
    #: Is this box's broker_parrot gpu lane heartbeating? False is what left cluster#18's box-b
    #: reservation queued for 27 minutes behind an empty blocker list: the worker container had failed its
    #: boot start and nothing on the box could claim anything (2026-09-23). None = unreadable, or no gpu
    #: lane heartbeats anywhere — "we cannot tell", never "dead".
    lane_live: bool | None = None
    #: WHY this box could not be asked, in the executor's words (`SshExecutor.last_error`: "status: timed out
    #: after 15 s", "status: exit 255: ssh: connect to host … No route to host"), "" when it answered or nothing
    #: is known. `reserving → unknown: cannot observe box-b` (2026-10-10, a GB10 box rebooting) gave the operator
    #: nothing to act on; the reason was in the LB's hands the whole time.
    error: str = ""


#: How far below its pre-launch MemAvailable a box may read and still count as having released. The page
#: cache and the fleet's own churn move MemAvailable by well under a GiB between observations, while a rank
#: of this pair holds ~100; this only decides whether a SHORTAGE is explained by the teardown, never whether
#: a start may proceed, so it costs nothing if it is a little tight.
RELEASE_SLACK_GIB = 1.0


def _not_released(o: BoxObs, free: float | None) -> float | None:
    """The pre-launch MemAvailable this box has NOT come back to yet, or None once it has (or was never
    asked to). Self-limiting on purpose: it explains a shortage while the memory is genuinely still out
    there, and never becomes a permanent label on the box."""
    target = o.released_from_gib
    if target is None or free is None or free >= target - RELEASE_SLACK_GIB:
        return None
    return target


def _no_lane(o: BoxObs) -> bool:
    """A reservation that is still QUEUED on a box whose lane is not heartbeating waits for a claim that
    will never come. Once claimed, the claim itself is the proof a lane is there."""
    return o.lane_live is False and o.reservation is not None and o.reservation.status == QUEUED


def ram_short(o: BoxObs, need: float | None) -> float | None:
    """THE ONE RAM GATE: how far this box's COUNTED-ON memory (`mem_stable_gib`, the lowest of its recent readings) falls
    under `need`, or None when it does not — the comparison the planner waits on and the box row prints, so the two can
    never disagree. None too while either figure is unknown: that is "memory unknown", not "short"."""
    if need is None or o.mem_stable_gib is None:
        return None
    short = need - o.mem_stable_gib
    return short if short > 0 else None


def _short_gib(short: float) -> str:
    """A shortfall as the page and the state line print it: to the tenth, and to the hundredth under a tenth — the
    gate waited six minutes on 0.04 GiB that both lines rounded to `0.0` and `116 free, need 116` (2026-10-10)."""
    return f"{short:.1f}" if round(short, 1) >= 0.1 else f"{short:.2f}"


def _runs_kv(spec, dtype):
    """The KV dtype a run of `spec` LAUNCHES — the run's, else its profile's own — for the seat note: a pool measured at
    one dtype (Flash-Next's at bfloat16) holds twice the tokens at the fp8_e4m3 its profile launches."""
    if dtype is not None:
        return dtype
    profile = spec.profile(CLUSTER_MODE)
    return profile.kv_cache_dtype if profile is not None else None


def at_rest(o: BoxObs, stable: float, *, broker: bool) -> bool:
    """NOTHING THE LB CAN SEE OR WAIT FOR holds this box's RAM, and its reading is not moving: what it has free now is
    the most it frees at rest. Every clause is a wait that is NOT over: its own ollama asked and empty (None: it could
    not be asked, which is not "nothing"), its last rank's memory back (`_not_released`), no fleet job and no classic
    vLLM (True is `freeing`/`draining` before this; None is unknown), and its recent readings within RELEASE_SLACK_GIB of
    each other — a box still letting ~100 GiB go climbs several GiB a sample."""
    if o.holds is None or o.holds or _not_released(o, stable) is not None:
        return False
    if o.broker_vllm_running is not False or (broker and o.running_jobs != 0):
        return False
    top = o.mem_peak_gib if o.mem_peak_gib is not None else o.mem_available_gib
    return top is not None and top - stable < RELEASE_SLACK_GIB


def blockers(o: BoxObs, need: float | None, *, broker: bool = True) -> list[dict]:
    """WHY this box is not free for the pair yet, in the closed vocabulary above and in the LB's own
    words — the `hold.blockers` the page shows (worklog/vllm-cluster-hold-robustness.md §6). An empty
    list means nothing is in the way; a box that could not be asked contributes no ollama blocker
    rather than a reassuring absence. A box whose OWN rank stands is waiting for nothing: the memory it
    lacks is the memory that rank holds (live, 2026-09-19: a serving pair's rows read "short 97.8 GiB").

    `broker` False (LB_BROKER_ENABLED unset): no lane, no fleet job and no broker vLLM exist to name — a
    vLLM container still up on the box is named as what it is, which nothing will stop for the pair."""
    if o.rank is not None and o.rank.present and o.rank.running:
        return []
    out = [{"code": NO_LANE, "message": "no broker_parrot gpu lane is heartbeating here, so nothing "
                                        "will claim the reservation: is its worker container up?"}
           ] if _no_lane(o) else []
    out += [{"code": OLLAMA_RESIDENT, "message": h.sentence()} for h in (o.holds or ())]
    if o.running_jobs:
        out.append({"code": FLEET_JOBS,
                    "message": f"{o.running_jobs} fleet job(s) still running on this box"})
    if o.broker_vllm_running:
        out.append({"code": BROKER_VLLM, "message": "broker_parrot's own vLLM is still up here" if broker else
                    "a classic vLLM container is still up here; nothing stops it without the broker"})
    free, short = o.mem_stable_gib, ram_short(o, need)
    if short is not None:
        held_over = _not_released(o, free)
        out.append({"code": MEM_SHORT, "message": f"{free:.1f} GiB free, needs {need:.1f} GiB: "
                                                  f"short {_short_gib(short)} GiB{o.needs_note}"
                                                  + ("" if held_over is None else
                                                     f" — the rank that came down here has not given back "
                                                     f"the {held_over:.0f} GiB this box had before it "
                                                     f"launched")})
    return out


@dataclass(frozen=True)
class Observed:
    boxes: dict[str, BoxObs] = field(default_factory=dict)
    #: The head's vLLM API answers (`/health` 200 and `/v1/models`) — the LB's own HTTP probe.
    head_live: bool = False
    #: The gateway's own turns still in flight on classic vLLM. A claimed reservation makes the broker
    #: stop that vLLM, so they end first (the router sends no new ones meanwhile).
    classic_inflight: int = 0
    #: …and on the TP=2 head: what a restart lets finish before it takes the old generation down.
    head_inflight: int = 0
    #: "" when broker_parrot's engine can claim a reservation; otherwise why it cannot — and Start is refused.
    reservations_unsupported: str = ""
    #: Does the head serve the ModelChoice? As its OWN /v1/models said it (`ModelCatalog.loaded_model`),
    #: never what we asked it to load. None = not live, or nothing to compare against.
    head_serves: bool | None = None
    #: What it serves, for the words.
    head_model: str | None = None
    #: How long the head's START has said nothing new — its StartupPhase and shard counts unchanged between
    #: two readable reads of its log and, while a weight pass is the step, its box not having given up another
    #: GiB (`cluster_loop._load_taken`: the bar is out of docker's sight until the pass is done). None = its log
    #: could not be read: a start nobody can see has not been SEEN to stall. And the phase it is in, for the words.
    start_quiet_s: float | None = None
    start_phase: str = ""
    #: How many times THIS start (generation and start id) has already been torn down and relaunched after a
    #: hang — the loop's count, kept beside the launch record so an LB restart does not reset it.
    start_rescues: int = 0
    #: "" unless the LB itself REFUSES to build THIS start's launch (`cluster_exec.launch_problem` on the stored
    #: selection — e.g. a KV pin past the profile's safe ceiling, a twin the model does not have — or an executor that
    #: raised before anything reached a box); then its reason. The same on every retry, so the start fails with it on
    #: the FIRST plan, before a box is asked or reserved and before the RAM gate, rather than relaunching every tick
    #: (generation 175, 2026-10-09: 566 refusals in 21 minutes; generation 187, 2026-10-10: 131 in six, alternating
    #: `starting` and `freeing` because the refusal was judged after the gate).
    launch_refused: str = ""


# ── actions and the plan ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class Reserve:
    """Ask broker_parrot for this box's reservation of generation `gen` (idempotent)."""
    box: str
    gen: int


@dataclass(frozen=True)
class ReleaseReservation:
    """Give a reservation back: cancel its run, exactly like a classic lease's release."""
    box: str
    reservation_id: str
    job_id: str


@dataclass(frozen=True)
class Up:
    box: str
    rank: int
    gen: int
    max_num_seqs: int
    max_model_len: int
    #: The ModelChoice this rank is launched FOR — the executor turns it into the spec's LaunchProfile
    #: and labels the rank with it.
    model_choice: str | None = None
    #: The operator's engine knobs as stored (`Desired`) — None is the default, which the executor leaves to
    #: where it has always come from, so a launch nobody changed is sent byte for byte as before.
    max_num_batched_tokens: int | None = None
    num_speculative_tokens: int | None = None
    ple_replicate: bool | None = None
    kv_cache_memory: int | None = None
    kv_offload_gib: int | None = None
    long_prefill_token_threshold: int | None = None
    #: Which checkpoint, as the switch resolved it (`Desired.uncensored`): None launched the original.
    uncensored: bool | None = None
    #: The run's KV element type (`Desired.kv_cache_dtype`): None launched the profile's own.
    kv_cache_dtype: str | None = None
    #: The operator chose the estimate for a model unmeasured on these cards (`Desired.estimated`).
    estimated: bool = False


@dataclass(frozen=True)
class Down:
    box: str


@dataclass(frozen=True)
class Halt:
    """Stop this box's rank and KEEP it: its memory is freed, its exit code and its log stay until a Down."""
    box: str


@dataclass(frozen=True)
class Heal:
    """Bring back the broker worker whose lane would claim this box's reservation: remount its NFS, restart
    its boot backstop. Planned on EVERY tick the lane is dead; the loop paces how often it really runs.

    A dead lane is healed, never waited out or failed: the reservation stands and is asked for as before,
    and nothing about the pair changes (operator, 2026-09-28: "we want healing service not killing service")."""
    box: str


Action = Reserve | ReleaseReservation | Up | Down | Halt | Heal


@dataclass(frozen=True)
class StartHang:
    """A start that said nothing new in `phase` for `quiet_s`, past that step's `limit_s`: HUNG. `attempt` is
    which attempt of this start it was (1 = the operator's own launch), `retries` how many relaunches a start
    may have, and `relaunch` whether this one is torn down and relaunched (else it is failed and halted)."""
    phase: str
    quiet_s: float
    limit_s: float
    attempt: int
    retries: int
    relaunch: bool

    def view(self) -> dict:
        return {"phase": self.phase, "quiet_s": round(self.quiet_s, 1), "limit_s": round(self.limit_s, 1),
                "attempt": self.attempt, "retries": self.retries, "relaunch": self.relaunch}


@dataclass(frozen=True)
class Plan:
    actions: list
    state: str
    detail: str = ""
    #: The router's gate: the cluster is ours to send traffic to. True only while both reservations
    #: run for their boxes, both ranks of the current generation run, and the head answers.
    held: bool = False
    #: Set on the tick a start is declared HUNG — rescued or failed — so the loop saves both ranks' evidence
    #: before the plan's Downs (or Halts) land, and the boot log records the attempt.
    hang: StartHang | None = None


def _present(o: BoxObs) -> bool:
    return o.rank is not None and o.rank.present


def _current_running(o: BoxObs, gen: int) -> bool:
    return o.rank is not None and o.rank.present and o.rank.running and o.rank.gen == gen


def _stale_rank(o: BoxObs, desired: Desired) -> str:
    """Why the rank standing here is not the one wanted now — its generation, or its ModelChoice — or ""
    when it is. A generation alone cannot tell two models apart, so the rank's own label is asked too.

    A rank reporting NO label says nothing about which model it runs — it was launched before the label
    existed. Treating that silence as "a different model" took the operator's SERVING pair down for an
    eight-minute reload nobody asked for, on the first tick after a deploy
    (worklog/vllm-gateway-program.md, INCIDENT 2026-09-16 00:30:32Z). What may condemn a standing rank is
    the head's own `/v1/models`, checked where the head is observed — never the absence of a label here.
    """
    if o.rank.gen != desired.generation:
        return "an older rank"
    if o.rank.model_choice and o.rank.model_choice != desired.model_choice:
        return f"a rank of {o.rank.model_choice}"
    return ""


def _requeue_expired(o: BoxObs, grace: float) -> bool:
    r = o.reservation
    return r is not None and _alive(r) and r.status == QUEUED and (o.requeued_s or 0.0) >= grace


def _grants(o: BoxObs, label: str, grace: float) -> bool:
    """Does broker_parrot grant this box to the pair right now? Running for its own lane — or back in
    the queue (a lane restart) for less than the engine's own lease duration."""
    r = o.reservation
    if r is None or not r.run_open:
        return False
    if r.status == RUNNING:
        return r.claimed_by == label
    return r.status == QUEUED and not _requeue_expired(o, grace)


def _old_grant_holds(o: BoxObs, desired: Desired, label: str) -> bool:
    """A RESTART's old rank still stands on its OWN generation's reservation: running, open and claimed by
    this box's lane. That is a box broker_parrot still grants the pair — what lets the old pair finish its
    turns ("finishing") instead of coming down the tick the new generation is asked for. Only a running rank
    of an older generation qualifies; the replacement itself is `_plan_cluster`'s to schedule."""
    if o.rank is None or not o.rank.running or not _stale_rank(o, desired):
        return False
    return any(r.status == RUNNING and r.run_open and r.claimed_by == label for r in o.stale)


def _and(names) -> str:
    names = list(names)
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"


def _job8(r: Reservation) -> str:
    return (r.job_id or "")[:8] or "—"


def _times(n: int) -> str:
    return "once" if n == 1 else "twice" if n == 2 else f"{n} times"


def _park_notes(obs: dict, boxes) -> list[str]:
    return [f"{b} parked by {obs[b].parked_by} — reservation kept" for b in boxes
            if obs[b].parked_by and obs[b].reservation is not None and obs[b].reservation.status == RUNNING]


def _unobserved(obs: dict, boxes) -> str:
    """`cannot observe box-b (status: timed out after 15 s)`: every box whose rank could not be read and, where
    the executor said, why — a reboot in progress, no route, a key refused. Nothing is invented for a box that
    gave no reason."""
    said = [f"{b} ({obs[b].error})" if obs[b].error else b for b in boxes if obs[b].rank is None]
    return "cannot observe " + ", ".join(said)


def plan(desired: Desired, observed: Observed, *, boxes: tuple[str, ...], labels: tuple[str, ...],
         requeue_grace_s: float = REQUEUE_GRACE_S,
         freeing_s: float = 0.0, freeing_deadline_s: float = FREEING_DEADLINE_S,
         start_stall_s: float = START_STALL_S, kill_floor_gib: float = KILL_FLOOR_GIB,
         start_retries: int = START_RETRIES, finishing_s: float = 0.0,
         finishing_limit_s: float = FINISHING_LIMIT_S, broker: bool = True) -> Plan:
    """`broker` False — LB_BROKER_ENABLED unset, the operator's own choice — plans with no broker_parrot at all:
    no reservation is asked for, judged, waited on or given back, no lane is healed, no park or fleet job is
    read. The rest is the same plan: what each box HAS, its memory against what this selection needs, the
    restart that lets turns finish, the hung start rescued, the failed pair halted."""
    obs = {b: observed.boxes.get(b, BoxObs(reservation_known=False)) for b in boxes}
    label_of = dict(zip(boxes, labels))
    gen = desired.generation
    actions: list[Action] = []

    def down(b: str) -> None:
        if Down(b) not in actions:
            actions.append(Down(b))

    # SAFETY FIRST, in every mode: a rank stands only on a box broker_parrot grants the pair. Judged
    # only on rows that could be read — an unreadable broker is not an ended reservation. Without the
    # broker nothing grants a box: it is the operator's.
    for b in boxes if broker else ():
        o = obs[b]
        if _present(o) and o.reservation_known and not _grants(o, label_of[b], requeue_grace_s) \
                and not (desired.mode == CLUSTER_MODE and _old_grant_holds(o, desired, label_of[b])):
            down(b)

    if desired.mode != CLUSTER_MODE:
        return _plan_stop(obs, boxes, actions, down, broker=broker)
    return _plan_cluster(desired, observed, obs, boxes, label_of, actions, down, requeue_grace_s,
                         freeing_s, freeing_deadline_s, start_stall_s, kill_floor_gib, start_retries,
                         finishing_s, finishing_limit_s, broker=broker)


def plan_observed(*, live: bool, error: str, box: str) -> Plan:
    """THE PLAN FOR A NODE THIS LB ONLY OBSERVES (control `none`, vllm_lb/topology.py): no action, because nothing here
    can start, stop or reconfigure it — its own compose service does, from its server file — and a state read off its
    HTTP API alone, the probe every backend gets. `running` (held: the router's gate) while it answers; `unknown` while
    it does not, in the probe's own words: a node that refuses connections may be stopped, loading or gone, and from
    outside those look the same, so none of them is claimed."""
    if live:
        return Plan([], "running", f"TP=1 up on {box} (observed)", held=True)
    return Plan([], "unknown", f"the node does not answer: {error or 'not probed yet'}")


#: The one action of a node the LB requests: write the intent its host's applier applies (vllm_lb/node_control.py).
RELEASE = "release"


@dataclass(frozen=True)
class NodeDown:
    """Release a DOWN intent to a one-host node's host at once (force): its box is no longer held for it."""
    box: str


def plan_node_hold(*, up: bool, start: str | None, current: Reservation | None, readable: bool, node_up: bool,
                   node_off: bool, strays: tuple = (), requeued_s: float | None = None, support: str, box: str,
                   label: str, jobs: int | None, local: Plan, grace_s: float = REQUEUE_GRACE_S,
                   kept_for: tuple = ()) -> tuple[list, Plan]:
    """THE BOX HOLD OF A ONE-HOST NODE THAT RESERVES (`fleet.Control.reserves`, the broker on) — pure: the actions on its
    broker_parrot reservation and the plan it leaves `local` (`plan_local`'s) as, or replaces it with.

    The pair's contract, for one box (worklog/parallel-clusters.md, "The local node's reservation"). `start`: this
    start's identity (minted when an `up` begins from off, kept through relaunches, dropped once the node is off and the
    reservation returned) — the reservation's run id is derived from it and the box. `current`: that run's reservation,
    read WHATEVER its run's status (None: never asked, or its run is open with no job yet — asked again). `node_up`: the
    host was told `up` or the node answers; `node_off`: it answers nothing and its host applied a down (or was never
    told anything). `strays`: this cluster's OTHER open reservations on the box (`BrokerLeases.node_reservations`) — a
    lost start file's, an older derivation's — returned once the node is off, whatever is wished: the pair's stale scan.
      * WANTED UP: nothing is released to the host until broker_parrot HOLDS the box — the box's OWN lane (`label`)
        claimed the reservation (`running`), its work drained, its ollama unloaded. Not asked: `Reserve`, `reserving`;
        queued: `draining` while the lane still runs `jobs`, else `waiting`. A NODE UP that no claim by this box's lane
        backs — its reservation ENDED (its job, or its run closed from outside), claimed by another lane, re-queued
        under it past the pair's `grace_s` (a lane restart re-queues it; the pair grants that long, `_grants`), or no
        reservation for this start — is taken DOWN (`NodeDown`) and the plan is `failed`. An ENDED reservation is
        sticky (its run id is this start's; a closed run is never asked again) until Stop, then Start. "No reservation
        for this start" is NOT sticky: once the node is off the next tick asks for one, and it starts again only on a
        box broker_parrot then holds for it. Nor is re-queued past the grace (the reservation never ended): the node
        stays down while it is queued, and starts again once the box's own lane claims it. Unreadable rows: `unknown`, nothing done (an unreadable broker is not an
        ended reservation). An engine that cannot claim one: `failed`.
      * NOT UP: the reservation is returned (`ReleaseReservation`) only once `local` says the node is `off` — never
        under an engine still running — and the plan reads `releasing` meanwhile.
      * ONE RESERVATION PER BOX, SHARED: broker_parrot claims one cluster row per lane (its per-kind cap of 1), so a node
        on a box a peer cluster already reserved RIDES that row — `current` is the peer's — and asks for none. Whoever
        stops while `kept_for` (the peers still holding the box) is not empty keeps the row, off; the last out returns it.
    """
    down = [NodeDown(box)] if node_up else []
    sweep = [ReleaseReservation(box, r.reservation_id, r.job_id) for r in strays] if node_off else []
    alive = current is not None and current.run_open and current.status in (QUEUED, RUNNING)
    if not up:
        if local.state == "off" and alive and kept_for:
            return sweep, replace(local, detail=f"{local.detail}; {box}'s reservation is kept for cluster "
                                                f"{', '.join(kept_for)}, which shares the box")
        if local.state == "off" and (alive or sweep):
            return (sweep + ([ReleaseReservation(box, current.reservation_id, current.job_id)] if alive else []),
                    Plan([], "releasing", f"the node is off; returning {box}'s reservation to broker_parrot"))
        return [], local
    if support:
        return down, Plan([], "failed", f"cannot reserve {box}: {support}")
    if not readable:
        return [], Plan([], "unknown", f"broker_parrot's reservations cannot be read: nothing is released to the node "
                                       f"until {box} is known to be held")
    if current is not None and not alive:
        why = current.status if current.status not in (QUEUED, RUNNING) else "its run was closed"
        return down + sweep, Plan([], "failed", f"broker_parrot ended the reservation on {box} (node_job "
                                                f"{(current.job_id or current.reservation_id)[:8]} {why}); the node is "
                                                f"taken down — Stop, then Start again")
    if current is not None and current.status == RUNNING:
        if current.claimed_by != label:
            return down + sweep, Plan([], "failed", f"broker_parrot ran {box}'s reservation on "
                                                    f"{current.claimed_by or 'no lane'}, not its own lane {label}: the "
                                                    f"node is taken down — Stop, then Start")
        return sweep, local
    if current is not None:                      # queued
        if node_up:
            if (requeued_s or 0.0) >= grace_s:
                return down, Plan([], "failed", f"reservation {current.job_id[:8]} on {box} has been back in the queue "
                                                f"under the running node for {requeued_s:.0f} s (the pair grants "
                                                f"{grace_s:.0f} s): the node is taken down; it starts again once "
                                                f"{label} claims {box} for it")
            return [], Plan([], "waiting", f"reservation {current.job_id[:8]} on {box} is queued again under the "
                                           f"running node (a lane restart re-queues it): kept for {grace_s:.0f} s")
        if jobs:
            return sweep, Plan([], "draining", f"reservation {current.job_id[:8]} queued on {box}: its lane still "
                                               f"runs {jobs} broker job(s), drained before it claims the box")
        return sweep, Plan([], "waiting", f"reservation {current.job_id[:8]} queued on {box}: its lane has not "
                                          f"claimed it")
    if node_up:
        return down, Plan([], "failed", f"no reservation backs the node on {box} for this start"
                                        f"{'' if start else ' (its start is unknown)'}: the node is taken down; once "
                                        f"it is off a reservation is asked for again")
    return sweep + [Reserve(box, gen=0)], Plan([], "reserving", f"asking broker_parrot to reserve {box}")


def plan_local(*, desired: dict | None, intent: dict | None, report: dict, live: bool, error: str, busy: int,
               box: str) -> Plan:
    """THE PLAN FOR A NODE THE LB REQUESTS AND ITS HOST APPLIES (control `local`, vllm_lb/node_control.py) — pure.

    `desired`: the operator's wish (the node's switch; None = never set: the fleet's boot default, observed). `intent`:
    what was last RELEASED to the host. `report`: node_control.read_report. `live`/`error`: the node's HTTP probe.
    `busy`: requests on the node now (the router's depth: our in-flight, its running and waiting).

    A wish differing from the released intent is released (`RELEASE`) — but a RELAUNCH of a node that serves waits
    until it has nothing in flight (`finishing`: not routed to, so it drains, as the pair finishes its turns before a
    restart), unless the wish says force (operator, 2026-10-11). Released, the node is routed to again only once the
    applier applied it and the node answers (`running`); before that it is `starting` (or `unloading` → `off` for a
    stop). The applier's own words decide the rest: a refusal ran nothing — a node still serving is still routed to —
    a failed command or a container that keeps restarting is `failed`, in its numbers. And an intent nobody picks up
    never strands a node that serves: it stays routed to, and the detail says what is not applied and why."""
    from vllm_lb.node_control import UP, same_intent
    if desired is None:
        return plan_observed(live=live, error=error, box=box)
    up = desired["state"] == UP
    released = intent is not None and (intent["state"], intent["launch"]) == (desired["state"], desired["launch"])
    if not released:
        if live and busy > 0 and not desired["force"]:
            what = "restarted" if up else "stopped"
            return Plan([], "finishing", f"finishing {busy} request(s) before the node is {what} — force cuts them")
        return Plan([RELEASE], "starting" if up else "unloading", f"releasing the {desired['state']} intent")
    tag = f"intent #{intent['id']} ({intent['state']})"
    applying, applied = report.get("applying"), report.get("applied")
    if same_intent(intent, applying):
        return Plan([], "starting" if up else "unloading", f"the applier is applying {tag}")
    if not same_intent(intent, applied):
        if not report.get("listening"):
            why = f"{tag} not applied: {report.get('error') or 'no applier listening'}"
            if live:
                return Plan([], "running", f"TP=1 up on {box}; {why}", held=up)
            return Plan([], "unknown", why)
        return Plan([], "starting" if up else "unloading", f"{tag} written; the applier picks it up")
    if applied.get("refused"):
        why = f"{tag} refused: {applied['refused']}"
        if live:
            return Plan([], "running", f"TP=1 up on {box}; {why}", held=True)
        return Plan([], "failed", why)
    if applied.get("exit") != 0:
        last = (applied.get("stderr") or "").strip().splitlines()
        return Plan([], "failed", f"{tag}: up.sh exit {applied.get('exit')}" + (f": {last[-1]}" if last else ""),
                    held=False)
    if not up:
        return Plan([], "unloading", f"{tag} applied; the node is going down") if live else \
            Plan([], "off", f"{tag} applied: the node is stopped")
    if live:
        return Plan([], "running", f"TP=1 up on {box} ({tag})", held=True)
    # A CRASH LOOP: the container restarts on its own (restart: unless-stopped) and reads `running` between attempts,
    # so its RestartCount against the count right after the command (`restarts_at_apply`) is what says so; a container
    # restarting, exited or dead says it outright. Either is FAILED — until a new intent — in the node's own words
    # (`root_cause`: vLLM's cause where its wrapper points up the log, else the last exception line), never the probe's
    # socket error.
    container = report.get("container") or {}
    at, now_restarts = applied.get("restarts_at_apply"), container.get("restarts")
    restarted = (now_restarts - at if isinstance(at, int) and isinstance(now_restarts, int) else 0)
    words = root_cause(report.get("log"))
    if restarted > 0:
        return Plan([], "failed", f"{tag}: the node container restarted {restarted} time(s)"
                                  + (f": {words}" if words else f" (exit {container.get('exit_code')})"))
    if container.get("status") in ("restarting", "exited", "dead"):
        return Plan([], "failed", f"the node container is {container.get('status')} (exit {container.get('exit_code')}, "
                                  f"{container.get('restarts')} restarts) after {tag}" + (f": {words}" if words else ""))
    return Plan([], "starting", f"{tag} applied; the node is loading")


#: An exception line in an engine's log: its class (…Error / …Exception, under the module it was printed with —
#: torch.OutOfMemoryError) and message, from the class name on.
_EXCEPTION_LINE = re.compile(r"\b((?:[a-z_]\w*\.)*[A-Z]\w*(?:Error|Exception): .+?)\s*$")
#: vLLM's front end, when an engine core process died starting: "Engine core initialization failed. See root cause
#: above. Failed core proc(s): {…}" (v1/engine/utils.py:1421-1441 @2eb491e) — its own words point UP the log.
_SEE_ABOVE = "See root cause above"
#: A line the ENGINE CORE process logged at ERROR, and what it says after vLLM's prefix — "(EngineCore pid=295) ERROR
#: 10-11 23:21:02 [core.py:1505] …" (vllm/logger.py:28-32), on every line of a logged traceback (logging_utils/
#: formatter.py NewLineFormatter): the one core.py:1505 logs when the engine fails to start.
_ENGINE_CORE_ERROR = re.compile(r"\(EngineCore\w* pid=\d+\) ERROR \S+ \S+ \[[^\]]+\] (.*)$")
#: All of what follows that prefix, when it is an exception line: its class (with its module) and, mostly, a message.
_EXCEPTION = re.compile(r"(?:[a-z_]\w*\.)*[A-Z]\w*(?:Error|Exception)(?:: .*)?")
#: A terminal colour code — vLLM colours its prefix on a tty (vllm/logger.py `_use_color`).
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def root_cause(log) -> str:
    """What a node's log says it failed on, verbatim from the exception's name — "" where it printed no exception.

    Its LAST exception line — unless that line is vLLM's wrapper pointing up the log ("See root cause above"): then
    the FIRST exception the ENGINE CORE logged in the attempt that wrapper ended (after the previous wrapper; a crash
    loop leaves several in the tail), vLLM's own cause. The 5090 node, 2026-10-11 23:21Z, at GPU memory 0.99: "ValueError:
    Free memory on device cuda:0 (30.85/31.36 GiB) on startup is less than desired GPU memory utilization (0.99, 31.04
    GiB) …", where the card said "RuntimeError: Engine core initialization failed. See root cause above. …". A
    wrapper whose cause is no longer in the tail read is said as it stands — never another attempt's cause."""
    lines = [_ANSI.sub("", ln).rstrip() for ln in str(log or "").splitlines()]
    for end in range(len(lines) - 1, -1, -1):
        m = _EXCEPTION_LINE.search(lines[end])
        if m is None:
            continue
        last = m.group(1)
        if _SEE_ABOVE not in last:
            return last
        start = next((i + 1 for i in range(end - 1, -1, -1) if _SEE_ABOVE in lines[i]), 0)
        for line in lines[start:end]:
            core = _ENGINE_CORE_ERROR.search(line)
            if core and _EXCEPTION.fullmatch(core.group(1).strip()):
                return core.group(1).strip()
        return last
    return ""


def _plan_stop(obs: dict, boxes, actions: list, down, *, broker: bool = True) -> Plan:
    for b in boxes:
        if _present(obs[b]):
            down(b)
    unobserved = [b for b in boxes if obs[b].rank is None]
    present = [b for b in boxes if _present(obs[b])]
    if not broker:                               # no reservation was ever asked for: none to give back
        if unobserved:
            return Plan(actions, "unknown", _unobserved(obs, boxes))
        if present:
            return Plan(actions, "unloading", f"removing the rank on {_and(present)}")
        return Plan(actions, "off", "no pair is running")
    known = all(obs[b].reservation_known for b in boxes)
    releases = []
    # Return the reservations only once NO box carries a rank: a TP pair is one unit, and a box handed
    # back while its partner still runs half a model is a box the fleet fills while we tear down the rest.
    if known and not unobserved and not present:
        for b in boxes:
            mine = ([obs[b].reservation] if obs[b].reservation is not None else []) + list(obs[b].stale)
            releases += [ReleaseReservation(b, r.reservation_id, r.job_id) for r in mine if r.run_open]
        actions.extend(releases)
    if unobserved:
        return Plan(actions, "unknown", _unobserved(obs, boxes))
    if present:
        return Plan(actions, "unloading", f"removing the rank on {_and(present)}")
    if not known:
        return Plan(actions, "unknown", "cannot read broker_parrot's reservations")
    if releases:
        return Plan(actions, "releasing",
                    f"returning {_and(dict.fromkeys(a.box for a in releases))} to broker_parrot "
                    f"(node_job {', '.join((a.job_id or '')[:8] or '—' for a in releases)})")
    returning = [obs[b].reservation for b in boxes if obs[b].reservation is not None
                 and not obs[b].reservation.run_open and obs[b].reservation.status in (QUEUED, RUNNING)]
    if returning:
        return Plan(actions, "releasing", f"broker_parrot is ending node_job {', '.join(map(_job8, returning))}")
    return Plan(actions, "off", "no reservation: the pair's boxes are the classic fleet's")


def _plan_cluster(desired: Desired, observed: Observed, obs: dict, boxes, label_of: dict, actions: list, down,
                  grace: float, freeing_s: float = 0.0,
                  freeing_deadline_s: float = FREEING_DEADLINE_S, start_stall_s: float = START_STALL_S,
                  kill_floor_gib: float = KILL_FLOOR_GIB, start_retries: int = START_RETRIES,
                  finishing_s: float = 0.0, finishing_limit_s: float = FINISHING_LIMIT_S,
                  broker: bool = True) -> Plan:
    gen = desired.generation

    def halted_pair(state: str, why: str) -> Plan:
        """A FAILED (OR STALLED) PAIR IS HALTED, NEVER REMOVED: every rank of this generation still running is
        stopped — half a pair is nothing, and it holds a whole box — and keeps its exit code and its log. A
        rank already coming Down (its reservation is gone) is not also halted. `state` is the word: `failed`
        for something that happened, `stalled` for a start that ran out its clock."""
        for b in boxes:
            if _current_running(obs[b], gen) and Down(b) not in actions:
                actions.append(Halt(b))
        return Plan(actions, state, f"{why}; Stop, then Start again")

    def failed_pair(why: str) -> Plan:
        return halted_pair("failed", why)
    if broker and not all(obs[b].reservation_known for b in boxes):
        return Plan(actions, "unknown", "cannot read broker_parrot's reservations")
    unobserved = [b for b in boxes if obs[b].rank is None]
    if unobserved:
        return Plan(actions, "unknown", _unobserved(obs, boxes))
    if broker and observed.reservations_unsupported:
        # A row nothing can claim is a Start that silently never starts: refuse it, and say why.
        return Plan(actions, "failed", f"{observed.reservations_unsupported}; Stop")

    # A RESTART LETS THE TURNS ON THE OLD PAIR FINISH. Taking a serving pair down cut every stream on it — a
    # Claude Code turn that dies mid-response is re-sent whole and re-prefilled from nothing. While the whole
    # old pair still runs and the head answers, the restart waits for the gateway's own turns on it, up to
    # `finishing_limit_s`; the old generation is not `held` meanwhile, so nothing new is routed there and a
    # new turn is held for the new pair (proxy `_hold_for_cluster`).
    replaced = [b for b in boxes if obs[b].rank.present and obs[b].rank.running and _stale_rank(obs[b], desired)]
    if (len(replaced) == len(boxes) and observed.head_live and observed.head_inflight > 0
            and finishing_s < finishing_limit_s):
        return Plan(actions, "finishing",
                    f"{observed.head_inflight} turn(s) still streaming on {_stale_rank(obs[replaced[0]], desired)}: "
                    f"the restart waits for them ({finishing_s:.0f} of {finishing_limit_s:.0f} s), new turns are held")
    for b in boxes:                              # an older generation, or another model, is replaced …
        if obs[b].rank.present and _stale_rank(obs[b], desired):
            down(b)
    # … and its reservations go back once nothing stands there AND this generation's has been asked
    # for on that box. ASK BEFORE YOU LET GO (worklog/vllm-cluster-handover.md H1): broker_parrot
    # cannot hold two reservations for one box at once — `gpu_slots='all'` weighs the box's whole
    # budget (node_queue.py:1326-1337) and one running `cluster` row caps that kind per lane
    # (node_queue.py:1442-1447) — but a QUEUED run-next reservation is the only row that lane may then
    # claim (node_queue.py:1363-1376) and already forbids its own LLM starts (claim_worker.py:4632-4662).
    # So the successor is queued first and the predecessor released after, and the box is never the
    # fleet's again in between. Live 2026-09-16 01:01:15Z, released with no successor: the lane claimed
    # a pool llm_chat job in the same second and re-loaded ollama on the box generation 11 needed.
    for b in boxes if broker else ():
        if not obs[b].rank.present and obs[b].reservation is not None:
            actions += [ReleaseReservation(b, r.reservation_id, r.job_id) for r in obs[b].stale if r.run_open]

    exited = [b for b in boxes if obs[b].rank.present and not _stale_rank(obs[b], desired)
              and not obs[b].rank.running]
    lost = [b for b in exited if obs[b].rank.lost]
    if lost:
        # THE BOX WENT AWAY UNDER THE RANK (`RankObs.lost`: its container is older than the box's boot). That is
        # not an exit, and no log of its explains a power cut or an operator's reboot, so there is nothing to
        # keep: every rank of this generation comes down — the partner too, whether it still runs (NCCL has
        # not noticed yet) or exited with the error its lost peer caused — and the very next plan launches the
        # same generation again through every gate a start passes. Before this rule the pair sat `failed` with
        # both GB10 boxes reserved and idle until the operator's Stop+Start (3.5 h on 2026-09-19).
        for b in boxes:
            if obs[b].rank.present and not _stale_rank(obs[b], desired):
                down(b)
        return Plan(actions, "restarting", f"{_and(lost)} rebooted under {'its' if len(lost) == 1 else 'their'} "
                                           f"rank (generation {gen}): relaunching the pair")
    if exited:
        b = exited[0]
        # AN EXIT CODE IS NOT A REASON. Five distinct faults reached the page as "exited (code 0)" on
        # 2026-10-05 — a CLI flag, a quant config, a MoE runner, a missing shard, the memory guard — and
        # an operator reading FAILED had nothing to act on. The box's own words go in the line.
        why = f"rank on {b} exited (code {obs[b].rank.exit_code})"
        if obs[b].rank.exit_reason:
            why = f"{why}: {obs[b].rank.exit_reason}"
        return failed_pair(why)

    for b in boxes if broker else ():            # asked for once per generation: an ended one is sticky
        r = obs[b].reservation
        if r is not None and not _alive(r):
            why = r.status if r.status not in (QUEUED, RUNNING) else "its run was closed"
            return Plan(actions, "failed", f"broker_parrot ended the reservation on {b} "
                                           f"(node_job {_job8(r)} {why}); Stop, then Start again")
        if r is not None and r.status == RUNNING and r.claimed_by != label_of[b]:
            return Plan(actions, "failed", f"broker_parrot ran {b}'s reservation on {r.claimed_by or 'no lane'}; "
                                           f"Stop, then Start again")

    older = [b for b in boxes if obs[b].rank.present and _stale_rank(obs[b], desired)]
    if older:
        # Its Down is planned above; launching on a box whose old rank still stands would collide on
        # the container name and the port. The next tick sees it gone.
        return Plan(actions, "restarting",
                    "replacing " + _and([f"{_stale_rank(obs[b], desired)} on {b}" for b in older]))

    standing = [b for b in boxes if _current_running(obs[b], gen) and Down(b) not in actions]
    if standing and len(standing) < len(boxes):
        # Half a pair is nothing: NCCL rendezvous needs both ranks in the same window.
        for b in standing:
            down(b)
        gone = [b for b in boxes if b not in standing]
        why = [f"broker_parrot did not re-claim the reservation on {b} within {grace:.0f} s"
               if _requeue_expired(obs[b], grace) else
               (f"nothing reserves {b}" if obs[b].reservation is None and _present(obs[b]) else f"rank missing on {b}")
               for b in gone]
        return Plan(actions, "restarting", "; ".join(why) + "; restarting the pair")

    # WHAT A BOX HAS, before it is taken from the fleet. Reserving both GB10 boxes for a model that is not
    # on them is the expensive mistake: ~13 minutes of cold start to find out. A pair already running is
    # past this — what a box could be asked says nothing about a model already loaded on it.
    if not all(_current_running(obs[b], gen) for b in boxes):
        # A LAUNCH THE LB ITSELF CANNOT BUILD is refused HERE — before a box is asked what it has, before either GB10 box
        # is taken from the fleet, and before the RAM gate. The refusal is of the SETTINGS and is the same whatever the
        # boxes have: judged after the gate (as it was until 2026-10-10) a box whose lowest reading wandered 0.1 GiB
        # either side of the need showed `freeing` and `starting` by turns while the launch was refused on every fresh
        # tick (generation 187: 131 refusals in six minutes). A pair already running is past this: what it RUNS built.
        if observed.launch_refused:
            return failed_pair(f"the LB refused to launch generation {gen}: {observed.launch_refused} — change "
                               f"that setting")
        unasked, unseen, blocked = [], [], []
        # THE CHECKPOINT THIS SELECTION LAUNCHES — the twin where the switch says so — and no other: a twin
        # that is not there blocks the start naming its dir, and the original beside it is not an answer.
        wanted = availability_key(desired.model_choice, desired.uncensored)
        for b in boxes:
            seen = (obs[b].availability or {}).get(wanted)
            if seen is None:
                unasked.append(b)                       # never asked, or it could not be asked at all
            elif seen.present is None:
                # IT LOOKED AND COULD NOT SEE — its NFS share not mounted yet, a stale handle, a folder that
                # did not answer, a docker that could not speak about the image. That is not "missing": on
                # 2026-10-10 20:15:22Z box-b, two minutes after a reboot and before box-a's export was up,
                # was told it lacked a checkpoint that was exactly where it always is, and the page said
                # `blocked … choose another model`. The box's own words say what it waits for.
                unseen.append(f"waiting for {b} to see {seen.name}: {seen.reason()}")
            elif seen.present is False:
                blocked.append(f"{b}: {seen.reason()}")
        if blocked:
            # Only a folder the box could READ and found without the file: that one is certain.
            return Plan(actions, "blocked", "; ".join(blocked) + "; choose another model, or copy it there")
        if unseen or unasked:
            return Plan(actions, "checking", "; ".join(
                unseen + ([f"asking {_and(unasked)} whether {wanted} is there"] if unasked else [])))

    missing = [b for b in boxes if obs[b].reservation is None] if broker else []
    if missing:
        if observed.classic_inflight > 0:
            return Plan(actions, "freeing",
                        f"freeing RAM: {observed.classic_inflight} gateway turn(s) still on classic vLLM")
        actions += [Reserve(b, gen=gen) for b in missing]
        return Plan(actions, "reserving", f"asking broker_parrot to reserve {_and(missing)}")

    notes = _park_notes(obs, boxes) if broker else []
    if standing:                                 # both ranks of this generation run on granted boxes
        # A STARTING RANK NEVER TAKES ITS BOX DOWN WITH IT: every recent sample under the floor. A serving
        # pair is left alone — it is working, whatever the number says.
        dry = [] if observed.head_live else [
            b for b in boxes if obs[b].mem_peak_gib is not None and obs[b].mem_peak_gib < kill_floor_gib]
        if dry:
            return failed_pair("; ".join(f"{b} is out of memory under its rank ({obs[b].mem_peak_gib:.1f} GiB "
                                         f"free, floor {kill_floor_gib:.1f})" for b in dry)
                               + " — the pair was stopped before the box died")
        requeued = [b for b in boxes if obs[b].reservation.status == QUEUED] if broker else []
        # A requeue no lane can take back is a worker that died under a serving pair: bring it back
        # inside the grace, rather than let the grace take the pair down.
        actions += [Heal(b) for b in requeued if _no_lane(obs[b])]
        if requeued:
            waits = [f"broker_parrot re-queued the reservation on {b}; waiting for the re-claim "
                     f"({obs[b].requeued_s or 0.0:.0f} s of {grace:.0f} s)"
                     + (f" — parked by {obs[b].parked_by}" if obs[b].parked_by else "")
                     + (f" — {label_of[b]} is not heartbeating: healing it" if _no_lane(obs[b]) else "")
                     for b in requeued]
            return Plan(actions, "waiting", "; ".join(waits + notes))
        if observed.head_live:
            # What it SERVES decides, not what we asked it to load.
            if observed.head_serves is False:
                return Plan(actions, "failed",
                            f"the head serves {observed.head_model or 'another model'}, not the "
                            f"{desired.model_choice} you chose; Stop, then Start again")
            return Plan(actions, "running", "; ".join([f"TP=2 pair up (generation {gen})"] + notes), held=True)
        quiet, phase = observed.start_quiet_s, observed.start_phase or "its start"
        # THE STEP'S OWN CLOCK: a start quiet for 3 minutes in a step that never takes 20 s is hung, and was
        # left to sit 10 (generations 122 and 131, "capturing CUDA graphs").
        limit = start_stall_limit(observed.start_phase, start_stall_s)
        rescues = max(0, observed.start_rescues)
        if quiet is not None and quiet >= limit:
            if rescues < start_retries:
                # A HANG IS RESCUED, NOT HALTED: both ranks come DOWN — the loop has saved their logs and stacks
                # by the time these land — and with this generation's reservations still running and the switch
                # unchanged, the very next plan launches them again. A Stop or a new Start in between is simply
                # what the next plan acts on instead.
                for b in boxes:
                    down(b)
                n = rescues + 1
                return Plan(actions, "restarting",
                            "; ".join([f"hung in “{phase}” for {quiet:.0f} s (the limit for this step is "
                                       f"{limit:.0f} s) — relaunching (retry {n}/{start_retries}); both ranks' logs "
                                       f"and stacks are saved first"] + notes),
                            hang=StartHang(phase=phase, quiet_s=quiet, limit_s=limit, attempt=n,
                                           retries=start_retries, relaunch=True))
            # OUT OF RELAUNCHES: HALTED, NOT FAILED. The same protective stop as ever — both ranks `docker stop`ped
            # and KEPT, their logs and stacks on the LB's disk — but a start that ran out its clock did not
            # error, and the page must not say it did (operator, 2026-10-10). `stalled` is the word; the
            # line says what hung, for how long against which limit, and what the LB did about it.
            after = f" after {rescues} relaunch{'' if rescues == 1 else 'es'}" if rescues else ""
            return replace(halted_pair("stalled", f"hung in “{phase}” for {quiet:.0f} s (the limit for this step "
                                                  f"is {limit:.0f} s){after} — both ranks halted, their logs kept"),
                           hang=StartHang(phase=phase, quiet_s=quiet, limit_s=limit, attempt=rescues + 1,
                                          retries=start_retries, relaunch=False))
        waiting = "both ranks running; waiting for the head to answer /health"
        if quiet is not None and quiet >= START_QUIET_NOTE_S:
            # WHAT COMES NEXT IS A THRESHOLD, NOT A THING DONE: "(relaunched at 600 s, retry 1/2)" was read by the
            # operator and the watchdog alike as a relaunch that had happened (generation 188, 2026-10-10 — nothing
            # had). The figures ahead, the retries left; "relaunched" only once one has, and how many.
            left = start_retries - rescues
            done = f"relaunched {_times(rescues)} already · " if rescues else ""
            then = (f"relaunch at {limit:.0f} s if still quiet · {left} retr{'y' if left == 1 else 'ies'} left"
                    if left > 0 else f"halt at {limit:.0f} s if still quiet · no retries left")
            waiting += f" — no progress for {quiet:.0f} s in “{phase}” ({done}{then})"
        return Plan(actions, "starting", "; ".join([waiting] + notes))

    if not broker and observed.classic_inflight > 0:
        # No reservation to ask for, but the same order: our own turns on classic vLLM end before a pair
        # that may need their memory is launched (the router sends them no new one meanwhile).
        return Plan(actions, "freeing",
                    f"freeing RAM: {observed.classic_inflight} gateway turn(s) still on classic vLLM")
    queued = [b for b in boxes if obs[b].reservation.status == QUEUED] if broker else []
    parked = [b for b in queued if obs[b].parked_by]
    asking = [b for b in queued if b not in parked]
    ask = ([f"asking broker_parrot to reserve {_and(asking)} "
            f"(node_job {', '.join(_job8(obs[b].reservation) for b in asking)})"] if asking else [])
    dead = [b for b in asking if _no_lane(obs[b])]
    actions += [Heal(b) for b in dead]
    ask += [f"{label_of[b]} is not heartbeating: no broker_parrot worker is up on {b} to claim it — "
            f"healing it (remounting its NFS, restarting its worker)" for b in dead]
    if parked:
        return Plan(actions, "waiting", "; ".join(
            [f"waiting for broker_parrot to claim {b} — parked by {obs[b].parked_by}" for b in parked] + ask))
    def past_budget() -> str:
        """THE RAM WAIT HAS A BUDGET, AND SAYS WHEN IT IS PAST IT. `freeing` once sat on one sentence forever
        (worklog/vllm-cluster-hold-robustness.md L3); past the budget the line says how long it has been and
        that the wait is over its budget, beside what still holds the RAM. It is not a FAILURE: nothing
        errored, the RAM may still come back (a chat client's ollama model expiring is the usual way), and
        then the start goes on — launching short is what is never done. Only the RAM wait: `draining` is
        the fleet's own jobs ending, and our turns on classic vLLM are bounded by those turns."""
        if freeing_s < freeing_deadline_s:
            return ""
        return (f" — {freeing_s:.0f} s so far, past the {freeing_deadline_s:.0f} s budget of a start; nothing has "
                f"errored, the launch goes on once the RAM is back")

    vllm_up = [b for b in boxes if obs[b].broker_vllm_running is True]
    if vllm_up:
        # Without the broker nothing stops it: the box is not free until the operator does.
        waiting_for = ([f"freeing RAM: stopping classic vLLM on {', '.join(vllm_up)}{past_budget()}"] if broker else
                       [f"a classic vLLM container is still running on {', '.join(vllm_up)}; nothing stops it "
                        f"without the broker — stop it, or Stop the pair{past_budget()}"]) + ask + notes
        return Plan(actions, "freeing", "; ".join(waiting_for))
    if ask:
        return Plan(actions, "reserving", "; ".join(ask + notes))

    busy: list[str] = []
    short: list[str] = []
    never: list[str] = []
    unsized: list[str] = []
    for b in boxes:
        o = obs[b]
        if not broker:
            pass                                 # no fleet: no job of its to wait for
        elif o.running_jobs is None:
            busy.append(f"{b}: running jobs unreadable")
        elif o.running_jobs > 0:
            busy.append(f"{b}: {o.running_jobs} fleet job(s) running")
        if o.broker_vllm_running is None:
            short.append(f"{b}: {'broker' if broker else 'classic'} vLLM state unknown")
        # THE REQUIREMENT is what THIS selection costs on THIS box, in this rank's ROLE — 103.2 GiB on
        # the head and 97.7 on the worker for qwen3.8-flash-next at 2 x 262,144 — measured off the
        # committed 2026-09-15 start (memory_estimate.py). One flat 100 was both too low for rank 0,
        # which is how generation 10 launched onto 77 GiB and exited code 1 88 s later, and too high
        # for rank 1 (worklog/vllm-cluster-hold-robustness.md §3-§4, L2). NO FLAT REQUIREMENT STANDS IN
        # (LB_CLUSTER_MIN_MEM_GIB retired, 2026-10-29): a selection the LB cannot size is refused below,
        # in the estimate's own words — once the box has reported, since a box that has not said yet is
        # a wait, not a verdict.
        # …AND IT IS JUDGED ON WHAT THE BOX CAN BE COUNTED ON TO HAVE — the lowest of its last
        # observations — never on the newest sample. While a cluster hold and a chat client fight over
        # ollama, MemAvailable swings by the size of the model every few seconds, and at 00:59:06 one
        # lucky sample launched 99 GB of weights onto a box that was really at 77 GiB (L3). A box with
        # fewer samples than that has not said yet, which is not the same as saying no. ONE comparison
        # (`ram_short`) for this gate and for the box row the page draws, printed to the tenth: `116 GiB
        # free, need 116` and `short 0.0 GiB` were two roundings of a 0.04 GiB shortfall (2026-10-10).
        stable = o.mem_stable_gib
        if o.mem_available_gib is None or stable is None:
            short.append(f"{b}: memory unknown")
            continue
        if o.needs_gib is None:
            unsized.append(f"{b}: {o.needs_why or 'the selection cannot be estimated'}")
            continue
        need = o.needs_gib
        lacking = ram_short(o, need)
        if lacking is not None:
            # A NEED THE BOX AT REST CAN NEVER MEET IS NOT A WAIT. Nothing resident, no fleet job, no classic vLLM,
            # its last rank's memory back, its readings steady — and still short: no budget of waiting brings that
            # memory, so the selection is refused with both numbers (generation 187 sat on exactly this).
            if at_rest(o, stable, broker=broker):
                never.append(f"{b}: {stable:.1f} GiB free is the most it frees at rest (nothing on it holds RAM); "
                             f"need {need:.1f}: short {_short_gib(lacking)} GiB{o.needs_note}")
                continue
            swing = ("" if o.mem_available_gib - stable < RELEASE_SLACK_GIB
                     else f" (swinging: {o.mem_available_gib:.1f} GiB in the newest sample)")
            # …AND WHAT IS HOLDING IT, when the box said. "77 GiB free, need 103" is not an answer to
            # "why does box-b wait"; "ollama holds qwen3.8:latest, 17.6 GiB" is (L8).
            held_by = "" if not o.holds else " — " + "; ".join(h.sentence() for h in o.holds)
            # …AND WHICH KNOB IS RESPONSIBLE, when one of the operator's own settings is what put the
            # requirement past what the box has. They can act on "lower the prefill budget"; they cannot
            # act on "need 120" (2026-09-30: batched 16,384 is 8.8 GiB of the head's requirement).
            # …AND WHETHER THE HANDOVER IS SIMPLY NOT DONE. Right after a teardown the memory the box is
            # missing is the memory its own last rank has not let go of yet, which is a wait, not a fault.
            held_over = _not_released(o, stable)
            waiting_on = ("" if held_over is None else
                          f" — the rank that came down here has not given back the {held_over:.0f} GiB "
                          f"this box had before it launched")
            short.append(f"{b}: {stable:.1f} GiB free, need {need:.1f}: short {_short_gib(lacking)} GiB"
                         f"{o.needs_note}{waiting_on}{swing}{held_by}")
    if unsized:
        return failed_pair(f"the LB cannot size generation {gen}: " + "; ".join(unsized)
                           + " — a Start is gated on what its selection costs on each box, never on a flat "
                           "requirement; choose a model the catalog gives a cluster LaunchProfile and memory footprint")
    if never:
        return failed_pair(f"the LB refused to launch generation {gen}: " + "; ".join(never)
                           + " — lower the KV pool, the prefill budget or the concurrency")
    if busy:
        return Plan(actions, "draining", "; ".join(busy + short + notes))
    if short:
        return Plan(actions, "freeing", "; ".join([f"freeing RAM{past_budget()}"] + short + notes))

    for i, b in enumerate(boxes):
        actions.append(Up(b, rank=i, gen=gen, max_num_seqs=desired.max_num_seqs,
                          max_model_len=desired.max_model_len, model_choice=desired.model_choice,
                          max_num_batched_tokens=desired.max_num_batched_tokens,
                          num_speculative_tokens=desired.num_speculative_tokens,
                          ple_replicate=desired.ple_replicate,
                          kv_cache_memory=desired.kv_cache_memory,
                          kv_offload_gib=desired.kv_offload_gib,
                          long_prefill_token_threshold=desired.long_prefill_token_threshold,
                          uncensored=desired.uncensored, kv_cache_dtype=desired.kv_cache_dtype,
                          estimated=desired.estimated))
    return Plan(actions, "starting", "; ".join([f"launching both ranks (generation {gen})"] + notes))
