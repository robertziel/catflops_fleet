"""What the TP=2 pair can actually hold — and therefore what the page offers when nobody has chosen.

One domain concept, kept out of `cluster.py` (which owns the switch and the plan) and out of
`config.py` (which owns wiring):

  KvPool            the KV cache ONE RANK reserves. PINNED by `--kv-cache-memory`, so it does not
                    grow with `max_num_seqs`: raising concurrency spends CUDA graphs, not KV.
  RecommendedSetup  the greatest `max_num_seqs` that fits a context in that pool, clamped to what
                    tp-rank accepts, carrying the evidence for a number nobody typed.

THE SOURCE. Every constant here is a measurement from `worklog/vllm-cluster-ram-headroom.md` — the
read-only audit of box-a and box-b taken while generation 5 was serving, merged `1812ed0`. Nothing
below is guessed, and nothing is re-derived:

  * §4, vLLM's own log, quoted verbatim and IDENTICAL at `max_num_seqs=1` (generation 1) and
    `max_num_seqs=2` (generation 5):

        GPU KV cache size: 1,708,119 tokens,
        Maximum concurrency for 262,144 tokens per request: 6.52x

    identical because `--kv-cache-memory` makes vLLM skip memory profiling entirely and reserve
    exactly the pool it is told to. Concurrency is therefore ALREADY PAID FOR up to that line.
  * §5.1, the per-token cost, cross-checked against that pool to a measured factor of 1.334:

        12,288 B  architectural, per rank: 12 full-attention layers (of 48 — the other 36 are
                  linear-attention and carry a fixed per-sequence state, not a per-token one)
                  x 2 kv heads x 256 head_dim x 2 (K and V) x 2 B (bf16), halved by TP=2
        x 1.334   the hybrid's mamba page, which vLLM pads to the attention page size
        = 16,392 B per token per rank

  * §5.2, the arithmetic this module IS:

        kv_tokens    = pool_bytes / 16,392          = 1,708,150 for the shipped 28 GB pool
        max_num_seqs = floor(kv_tokens / max_model_len)

    which reproduces vLLM's own "Maximum concurrency" line: 6 at 262,144 (6.52x), 13 at 131,072
    (13.03x), 26 at 65,536 (26.06x). 1,708,150 against vLLM's printed 1,708,119 is 0.002 %.

Sizing the POOL was once deliberately not here, while `CLUSTER_KV_CACHE_MEMORY` was a tp-rank default the
LB did not forward (§5.3). It is an operator knob now (`Desired.kv_cache_memory`), and its `auto` value —
the pool sized at each Start to what the boxes have left — is resolved here (`kv_auto`), from what the
cluster loop publishes, with M8's arithmetic (`memory_estimate.resolve_kv_auto`).

PER MODEL. Those two numbers are Flash-Next's. Every other ModelSpec says its own — the pool its cluster
LaunchProfile pins and its MemoryFootprint's `kv_bytes_per_token` (the dense 27B: 48 GB at 36,860 B, vLLM's own
1,302,238 tokens and 4.97x on generation 103) — so the arithmetic takes them as ARGUMENTS and owns no
configuration of its own: `pool_of(cfg, spec)` picks them, `recommended_for` and `fits_for` use them, and the
`Config` pair is only what stands in when no spec says. `auto` is never sized for a spec nobody calibrated.
There is one estimator, and it is not here: GiB, weights, overhead and the danger floor are
`memory_estimate.py`'s M8.
"""
from __future__ import annotations

from dataclasses import dataclass

from vllm_lb.catalog import CLUSTER_MODE, RANK_ROLES
from vllm_lb.cluster import (KV_AUTO, KV_CACHE_MEMORY_RANGE, KV_CACHE_MEMORY_STEP, MAX_NUM_SEQS_RANGE, Desired,
                             DesiredStore)
from vllm_lb.memory_estimate import KvAuto, KvAutoBox, estimate_memory, resolve_kv_auto
from vllm_lb.topology import serving_cards

#: The KV cache one rank reserves, in bytes: tp-rank's `CLUSTER_KV_CACHE_MEMORY` default
#: (lb/cluster/tp-rank:66), which vLLM echoed back as `kv_cache_memory_bytes: 28000000000`.
CLUSTER_KV_CACHE_MEMORY = 28_000_000_000
#: Bytes of KV pool one token costs on one rank. Measured — see the module docstring, §5.1.
KV_BYTES_PER_TOKEN_PER_RANK = 16_392
#: Where a reader checks every number above.
SOURCE = "worklog/vllm-cluster-ram-headroom.md §5 (measured 2026-09-15, generation 5 serving)"


def kv_tokens_per_rank(pool_bytes: int,
                       *, bytes_per_token: int = KV_BYTES_PER_TOKEN_PER_RANK) -> int:
    """How many tokens of KV one rank's pool holds. Floored: a partial token is not a token."""
    if bytes_per_token <= 0:
        raise ValueError(f"bytes per token must be positive, not {bytes_per_token!r}")
    return max(0, int(pool_bytes)) // bytes_per_token


def max_num_seqs_that_fits(kv_tokens: int, max_model_len: int,
                           *, seqs_range: tuple[int, int] = MAX_NUM_SEQS_RANGE) -> int:
    """THE ceiling: the greatest `max_num_seqs` whose worst case fits in `kv_tokens`.

        floor(kv_tokens / max_model_len), clamped into the range tp-rank accepts

    It is vLLM's own "Maximum concurrency for N tokens per request" line, floored: 6.52x is SIX
    seats, because the seventh does not fit and vLLM would queue it.

    Clamped, and never zero. tp-rank refuses `max_num_seqs` outside 1..48 (`cmd_up`), and a pool too
    small for even one full-window request still serves that request: past the pool vLLM ADMITS
    fewer sequences, it does not fail to launch and it does not OOM (§5.2 — "past that vLLM queues,
    it does not OOM", because the pool is paged and shared). So one seat is the honest floor and 48
    the cap, whatever the division says.
    """
    if max_model_len <= 0:
        raise ValueError(f"max_model_len must be positive, not {max_model_len!r}")
    lo, hi = seqs_range
    return max(lo, min(hi, max(0, int(kv_tokens)) // int(max_model_len)))


@dataclass(frozen=True)
class RecommendedSetup:
    """The setup the measurement recommends, carrying its own evidence: the page shows a number the
    operator did not choose, so it must be able to say where the number came from."""
    max_num_seqs: int
    max_model_len: int
    kv_tokens: int
    kv_cache_memory: int
    source: str = SOURCE

    def view(self) -> dict:
        """`cluster.recommended` in /lb/status — the contract the page reads."""
        return {"max_num_seqs": self.max_num_seqs, "max_model_len": self.max_model_len,
                "kv_tokens": self.kv_tokens, "kv_cache_memory": self.kv_cache_memory,
                "source": self.source}


def recommend(pool_bytes: int, max_model_len: int,
              *, bytes_per_token: int = KV_BYTES_PER_TOKEN_PER_RANK,
              seqs_range: tuple[int, int] = MAX_NUM_SEQS_RANGE) -> RecommendedSetup:
    """The measured maximum for one pool at one context."""
    tokens = kv_tokens_per_rank(pool_bytes, bytes_per_token=bytes_per_token)
    return RecommendedSetup(
        max_num_seqs=max_num_seqs_that_fits(tokens, max_model_len, seqs_range=seqs_range),
        max_model_len=int(max_model_len), kv_tokens=tokens, kv_cache_memory=int(pool_bytes))


@dataclass(frozen=True)
class EnginePool:
    """WHAT THE ENGINE ITSELF SAYS ITS POOL HOLDS — the only seat figure on the page that was measured.

    vLLM prints one line per start, and publishes the same two numbers as `vllm:cache_config_info`:

        GPU KV cache size: 629,682 tokens, Maximum concurrency for 262,144 tokens per request: 2.40x

    They are ONE number. The engine works its token count out FROM its block count as
    `kv_cache_max_concurrency x max_model_len`, so `tokens / max_model_len` IS the concurrency it
    printed — exactly, on every generation recorded in this repo:

        1,708,119 / 262,144 = 6.5160   Flash-Next, 28 GB   (2026-09-30, and the 2026-09-15 start log)
        1,302,238 / 262,144 = 4.9676   27B, 48 GB          (generation 105, fixtures/cluster-27b-…json)
          629,682 / 262,144 = 2.4020   27B DFlash2, 28 GB  (generation 109, fixtures/qwen38-dflash2-…log)

    `tokens` is the field the LB already scrapes and publishes (`throughput.kv_pool.tokens`); the
    concurrency itself is not published and does not need to be.

    WHY THIS WINS over the arithmetic in this module. `KvBlocks` turns a pin into blocks by dividing it
    by a `block_bytes` back-computed from ONE measured pin, and that extrapolation is only as good as the
    constants: on generation 109 it said 985 blocks and 3 seats where the engine seated 2.40, because the
    dflash profile pages at 832 tokens a block (its own start log) against the 864 the catalog records
    and reuses the MTP profile's `block_bytes`. The engine does not extrapolate.

    THE FIGURE IS THAT CONTEXT'S, because the engine divided by the window it was launched for. A pool
    measured at 262,144 says nothing directly about 65,536 — `blocks` turns the measurement back into the
    count the engine really holds so a what-if at another context stays anchored on it, and it reproduces
    `num_gpu_blocks` exactly wherever the model's block rule matches the engine (6.51595744680851 x 188 =
    1,225; 4.9676470588235295 x 340 = 1,689).
    """
    #: `vllm:cache_config_info` `kv_cache_size_tokens` — the pool the engine REALLY allocated.
    tokens: int
    #: The window it divided by: `cluster.effective.max_model_len`, the context the pair was launched for.
    max_model_len: int

    @property
    def concurrency(self) -> float:
        """The engine's own `kv_cache_max_concurrency` for the context it runs."""
        return self.tokens / self.max_model_len

    def blocks(self, rule) -> int:
        """THE ENGINE'S OWN BLOCK COUNT, recovered through the model's `KvBlocks` rule: what it must have
        held for the concurrency it printed. Measured, where `rule.blocks_in(pin)` is extrapolated."""
        return max(1, round(self.concurrency * rule.per_request(self.max_model_len)))

    def seats(self, max_model_len: int, rule=None, *,
              seqs_range: tuple[int, int] = MAX_NUM_SEQS_RANGE) -> int:
        """How many requests of `max_model_len` this pool holds at once, floored and clamped into
        tp-rank's range — the engine's own concurrency at the context it was measured at, and its own
        block count through `rule` at any other."""
        ctx = int(max_model_len)
        if ctx <= 0:
            raise ValueError(f"max_model_len must be positive, not {max_model_len!r}")
        # At the engine's own context, divide the engine's own tokens: `blocks()` rounds, and a
        # concurrency a hair under a whole seat must not round up into one.
        seats = (self.tokens // ctx if rule is None or ctx == self.max_model_len
                 else self.blocks(rule) // rule.per_request(ctx))
        lo, hi = seqs_range
        return max(lo, min(hi, int(seats)))

    def view(self) -> dict:
        return {"tokens": self.tokens, "max_model_len": self.max_model_len,
                "max_concurrency": round(self.concurrency, 4)}


def engine_pool(view, spec=None, pool=None, *, kv_cache_memory=None) -> EnginePool | None:
    """THE SERVING PAIR'S OWN POOL — but only when it is the pool being asked about, else None.

    `view` is the cluster view (`held` and `effective`), `pool` the head's `throughput.kv_pool`. The
    guards are `proxy._engine_blocks`'s: a pair that is held, launched for THIS model choice and THIS
    pin, with a window of its own and a token count it really reported. Anything else — nothing
    running, a start in flight, another model, another pin — is a question measurement cannot answer,
    and the arithmetic answers it instead.
    """
    view = view if isinstance(view, dict) else {}
    eff = view.get("effective") if isinstance(view.get("effective"), dict) else None
    if not bool(view.get("held")) or eff is None:
        return None
    if spec is not None and eff.get("model_choice") != spec.id:
        return None
    profile = spec.profile(CLUSTER_MODE) if spec is not None else None
    want = kv_cache_memory if kv_cache_memory is not None else (
        profile.kv_cache_memory if profile is not None else None)
    if want is None or eff.get("kv_cache_memory") != want:
        return None
    ctx = eff.get("max_model_len")
    count = _number(pool.get("tokens")) if isinstance(pool, dict) else None
    if not _whole(ctx) or ctx <= 0 or count is None or count <= 0:
        return None
    return EnginePool(tokens=int(count), max_model_len=int(ctx))


def seats_for(spec, pool_bytes: int, max_model_len: int, *, engine: EnginePool | None = None,
              pool_blocks: int | None = None,
              seqs_range: tuple[int, int] = MAX_NUM_SEQS_RANGE, bytes_per_token: int | None = None) -> int:
    """THE SEATS a pool holds at a context — THE ENGINE'S OWN ANSWER (`engine`, what it printed about the
    pool it is serving from) whenever there is one, else the ENGINE's rule as the ModelSpec records it
    (`KvBlocks`: blocks over blocks-per-request, 17 at 65,536 on the 27B where tokens over context said
    19) with its own block count (`pool_blocks`, vllm:cache_config_info) preferred while it serves, else
    tokens over context at the spec's cost per token (`bytes_per_token`, or the spec's). Clamped into
    tp-rank's range.

    `engine` first because it is the only term here that was MEASURED: everything below extrapolates a
    pin through constants taken at another pin, and on generation 109 that said 3 seats where the engine
    said 2.40 (`EnginePool`)."""
    memory = spec.memory if spec is not None else None
    if max_model_len <= 0:
        raise ValueError(f"max_model_len must be positive, not {max_model_len!r}")
    lo, hi = seqs_range
    if engine is not None:
        return engine.seats(max_model_len, memory.kv_blocks if memory is not None else None,
                            seqs_range=seqs_range)
    if memory is not None and memory.kv_blocks is not None:
        return max(lo, min(hi, memory.kv_blocks.seats(pool_bytes, max_model_len, pool_blocks=pool_blocks)))
    per_token = bytes_per_token or (memory.kv_bytes_per_token if memory is not None else KV_BYTES_PER_TOKEN_PER_RANK)
    return max_num_seqs_that_fits(kv_tokens_per_rank(pool_bytes, bytes_per_token=per_token), max_model_len,
                                  seqs_range=seqs_range)


def pool_of(cfg, spec=None) -> tuple[int, int]:
    """(pool bytes, bytes per token per rank) a recommendation is made for: the ModelSpec's own cluster pin
    and KV cost, and the gateway's configured pair (`cfg`, the shipped model's measurement) for whatever a
    spec does not say — or when there is no spec at all."""
    profile = spec.profile(CLUSTER_MODE) if spec is not None else None
    pin = profile.kv_cache_memory if profile is not None else None
    memory = spec.memory if spec is not None else None
    return (int(pin) if pin else cfg.cluster_kv_cache_memory,
            memory.kv_bytes_per_token if memory is not None else cfg.cluster_kv_bytes_per_token)


def recommended_for(cfg, spec, *, max_model_len: int | None = None) -> RecommendedSetup:
    """The recommendation for `spec` (the stored ModelChoice, each time the view is published) at ITS context —
    its row's `tuned.max_model_len`, or `max_model_len` when the caller names one — for its own pool and
    per-token cost, the gateway's configured pool standing in for what the spec does not say. The LB has no
    configured context of its own (LB_CLUSTER_MAX_MODEL_LEN is retired): no spec, nothing to recommend for."""
    if spec is None:
        raise ValueError("no model to recommend for: the context is the model's own (its row's tuned.max_model_len)")
    if max_model_len is None:
        profile = spec.profile(CLUSTER_MODE)
        tuned = profile.tuned if profile is not None else None
        if tuned is None or tuned.max_model_len is None:
            raise ValueError(f"{spec.id} states no cluster tuned.max_model_len: nothing to recommend at")
        max_model_len = tuned.max_model_len
    pool, per_token = pool_of(cfg, spec)
    r = recommend(pool, max_model_len, bytes_per_token=per_token)
    if spec.memory is not None and spec.memory.kv_blocks is not None:
        r = RecommendedSetup(max_num_seqs=seats_for(spec, pool, max_model_len),
                             max_model_len=r.max_model_len, kv_tokens=r.kv_tokens, kv_cache_memory=r.kv_cache_memory,
                             source=f"{SOURCE}; seats by the engine's block rule: "
                                    f"{spec.memory.kv_blocks.source or 'blocks per request'}")
    return r


def fits_for(cfg):
    """`(max_model_len, pool_bytes, bytes_per_token) -> the concurrency that fits it`, for a store to warn a
    write with. The store passes the pool the pair would launch with — the operator's pin, else the chosen
    ModelSpec's — and that spec's own cost per token; either left out is the gateway's configured one."""
    def fits(max_model_len: int, pool_bytes: int | None = None, bytes_per_token: int | None = None,
             spec=None) -> int:
        return seats_for(spec, pool_bytes or cfg.cluster_kv_cache_memory, max_model_len,
                         bytes_per_token=bytes_per_token or cfg.cluster_kv_bytes_per_token)

    return fits


def desired_store(cfg, *, catalog=None) -> DesiredStore:
    """The cluster switch as the LB builds it: on disk at `cfg.cluster_state_path`, offering the default row's
    own tuned seats and context when nothing is stored, clamping a write that would not fit, and validating the
    operator's ModelChoice against the ONE `ModelCatalog` (pass 4).

    ONE place builds it — `__main__` and the tests both come here — so the default the operator is
    offered and the default the tests prove are the same object, wired the same way. `catalog` is the
    instance this process loaded; left out, the store loads its own, which is what the tests want.
    """
    return DesiredStore(cfg.cluster_state_path, catalog=catalog,
                        offers=cfg.clusters[0].offers if len(cfg.clusters) == 1 else None,
                        kv_dtypes=cfg.clusters[0].kv_dtypes if len(cfg.clusters) == 1 else None,
                        fits=fits_for(cfg),
                        # The serving cards' arch facts: what a KV dtype bound to some cards is checked against.
                        archs=tuple(c.type.arch.value for c in serving_cards(cfg)))


# ── auto: the pool sized at Start to what the boxes have left ────────────────


def _number(v) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _whole(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def kv_auto(cfg, catalog, view, spec, *, max_num_seqs: int, max_model_len: int, max_num_batched_tokens: int,
            num_speculative_tokens: int | None = None, ple_replicate: bool | None = None) -> KvAuto:
    """What a Start at `auto` would pin NOW for this selection, from what the cluster loop last published (`view`,
    the `cluster` of /lb/status): each box's lowest recent MemAvailable and the most it had while the pair
    rested, whether the pair serves, and what it was launched with (`cluster.effective`) — sized by M8.

    A pair that SERVES is measured, and what its rank on each box costs comes back when a restart replaces it.
    With no rank of ours standing the boxes are sized as they are: whatever the fleet holds counts against
    them, and a pool that would come out under the model's own is refused rather than shrunk to fit a busy
    box. A pair that is starting, or half up, is neither — its memory is still moving — and auto says so
    instead of sizing it. The baseline auto may not fall under is the pool the serving pair runs, else the
    chosen model's own."""
    view = view if isinstance(view, dict) else {}
    published = view.get("boxes") if isinstance(view.get("boxes"), dict) else {}
    eff = view.get("effective") if isinstance(view.get("effective"), dict) else None
    serving = bool(view.get("held")) and eff is not None
    profile = spec.profile(CLUSTER_MODE)
    if profile is not None and "kv_cache_memory" in profile.refuses:
        # NO PIN TO SIZE: the engine sizes its own pools (SGLang: engine_args.mem_fraction_static).
        return KvAuto((), None, profile.kv_cache_memory, None,
                      f"{spec.label}: {profile.refuses['kv_cache_memory']} — auto has nothing to size")
    if not spec.calibrated:
        # NEVER SIZED FROM NUMBERS NOBODY MEASURED: an uncalibrated footprint is arithmetic, and auto would pin
        # everything the boxes have left on it (2026-10-02: ~79 GB for the dense 27B).
        return KvAuto((), None, profile.kv_cache_memory if profile is not None else None, None,
                      f"{spec.label} is not calibrated: choose a size")
    common = dict(tp=len(cfg.cluster_hosts) or 2, available_gib=None)
    same = serving and eff.get("model_choice") == spec.id and _whole(eff.get("kv_cache_memory"))
    baseline = int(eff["kv_cache_memory"]) if same else (profile.kv_cache_memory if profile is not None else None)
    boxes, standing = [], []
    try:
        for i, host in enumerate(cfg.cluster_hosts):
            role = RANK_ROLES[i] if i < len(RANK_ROLES) else RANK_ROLES[-1]
            seen = published.get(host.box) if isinstance(published.get(host.box), dict) else {}
            rank = seen.get("rank") if isinstance(seen.get("rank"), dict) else {}
            if not serving and rank.get("running"):
                standing.append(host.box)
            policy = cfg.memory_policy(_number(seen.get("mem_total_gib")))
            new = estimate_memory(spec, box=host.box, role=role, max_num_seqs=max_num_seqs,
                                  max_model_len=max_model_len, max_num_batched_tokens=max_num_batched_tokens,
                                  speculative=(None if num_speculative_tokens is None
                                               else num_speculative_tokens > 0),
                                  ple_replicate=ple_replicate, **common, **policy)
            returned = 0.0
            if serving:
                drafts = eff.get("num_speculative_tokens")
                returned = estimate_memory(
                    catalog.get(eff.get("model_choice")), box=host.box, role=role,
                    max_num_seqs=int(eff["max_num_seqs"]), max_model_len=int(eff["max_model_len"]),
                    max_num_batched_tokens=eff.get("max_num_batched_tokens"),
                    speculative=drafts > 0 if _whole(drafts) else None, ple_replicate=eff.get("ple_replicate"),
                    kv_cache_memory=eff.get("kv_cache_memory"), **common, **policy).total_gib
            boxes.append(KvAutoBox(box=host.box, available_gib=_number(seen.get("mem_stable_gib")),
                                   rest_gib=_number(seen.get("mem_rest_gib")) if serving else None,
                                   returned_gib=returned, beside_pool_gib=new.beside_pool_gib,
                                   floor_gib=policy["floor_gib"],
                                   load_gib=max(cfg.cluster_kv_auto_load_gib, new.load_gib)))
    except (ValueError, TypeError, KeyError) as exc:          # a selection, or a running pair, M8 cannot size
        return KvAuto(tuple(boxes), None, baseline, None, f"it cannot be estimated: {exc}")
    if standing:
        return KvAuto(tuple(boxes), None, baseline, None,
                      f"the pair on {' and '.join(standing)} is {view.get('state') or 'not serving'}: auto sizes "
                      f"the pool from a pair that serves, or from boxes with no rank of ours on them — wait for "
                      f"it, or choose a size")
    # THE LARGEST POOL THAT FITS EVERY BOUND: the range, the boxes — and the profile's own ceiling where it has one
    # (`LaunchProfile.kv_cache_memory_max`, SGLang's calibrated mem_fraction bound), rounded down to the panel's step.
    # Boxes with room for 27 GB on the SGLang Flash-Next row are sized at 19 GB, not at a pin the launch refuses.
    lo, hi = KV_CACHE_MEMORY_RANGE
    ceiling = profile.kv_cache_memory_max() if profile is not None else None
    if ceiling is not None:
        hi = min(hi, ceiling // KV_CACHE_MEMORY_STEP * KV_CACHE_MEMORY_STEP)
    return resolve_kv_auto(boxes, baseline=baseline, step=KV_CACHE_MEMORY_STEP, pin_range=(lo, hi))


def kv_choices_run(choices: list, allowed) -> list:
    """A model's KV dtype choices as the page is offered them: only those this deployment runs (fleet `kv_cache_dtypes`;
    None = all), so a dtype it never runs is no option at all."""
    return list(choices) if allowed is None else [c for c in choices if c.get("dtype") in allowed]


def box_total(view, box: str) -> float | None:
    """`box`'s MemTotal as the loop last published it (tp-rank's status), or None."""
    boxes = view.get("boxes") if isinstance(view, dict) and isinstance(view.get("boxes"), dict) else {}
    seen = boxes.get(box) if isinstance(boxes.get(box), dict) else {}
    return _number(seen.get("mem_total_gib"))


def memory_admission(cfg, catalog, view_of):
    """`DesiredStore.set`'s `admit`: a Start whose selection does not fit a box that is AT REST — nothing the LB can
    wait for holds its RAM (`cluster.at_rest`, as the loop publishes it) — is refused NOW, with the figures, instead of
    being stored and refused at the launch minutes later (operator, 2026-10-26: "if not enough ram tell before I hit
    start"). The SAME requirement as the launch gate: `needed_now_gib` against `mem_stable_gib`. A box not at rest
    (its rank standing, a fleet job, ollama resident, still releasing) is the loop's to wait for, as before."""
    def admit(selection: Desired) -> None:
        spec = catalog.get(selection.model_choice)
        view = view_of() or {}
        published = view.get("boxes") if isinstance(view.get("boxes"), dict) else {}
        short = []
        for i, host in enumerate(cfg.cluster_hosts):
            seen = published.get(host.box) if isinstance(published.get(host.box), dict) else {}
            free = _number(seen.get("mem_stable_gib"))
            if seen.get("at_rest") is not True or free is None or spec is None:
                continue
            drafts, budget = selection.num_speculative_tokens, selection.max_num_batched_tokens
            if budget is None:                   # the launch's own resolution, as the loop published it; else the fleet's
                published_budget = (view.get("defaults") or {}).get("max_num_batched_tokens")
                budget = published_budget if _whole(published_budget) else cfg.cluster_max_num_batched_tokens
            try:
                est = estimate_memory(
                    spec, box=host.box, role=RANK_ROLES[i] if i < len(RANK_ROLES) else RANK_ROLES[-1],
                    max_num_seqs=selection.max_num_seqs, max_model_len=selection.max_model_len,
                    max_num_batched_tokens=budget, speculative=None if drafts is None else drafts > 0,
                    ple_replicate=selection.ple_replicate, kv_cache_memory=selection.kv_cache_memory,
                    kv_cache_dtype=selection.kv_cache_dtype, tp=len(cfg.cluster_hosts) or 2, available_gib=free,
                    **cfg.memory_policy(_number(seen.get("mem_total_gib"))))
            except ValueError:                   # not estimable here: the launch gate is the loop's, as before
                continue
            if free < est.needed_now_gib:
                short.append(f"{host.box}: {free:.1f} GiB free is the most it frees at rest; need "
                             f"{est.needed_now_gib:.1f} ({est.total_gib:.1f} + {est.start_headroom_gib:.1f} reserve): "
                             f"short {est.needed_now_gib - free:.1f} GiB")
        if short:
            raise ValueError("not enough RAM: " + "; ".join(short) + " — lower the KV pool, the prefill budget, the "
                             "concurrency or the context, or free the box")

    return admit


def running_rank_gib(cfg, catalog, view, box: str, role: str) -> float | None:
    """What OUR serving rank on `box` costs by its own estimate — the launch record's settings
    (`cluster.effective`) sized by M8 — or None when no pair serves, or it cannot be estimated."""
    view = view if isinstance(view, dict) else {}
    eff = view.get("effective") if isinstance(view.get("effective"), dict) else None
    if not (bool(view.get("held")) and eff is not None):
        return None
    drafts = eff.get("num_speculative_tokens")
    try:
        return estimate_memory(
            catalog.get(eff.get("model_choice")), box=box, role=role, max_num_seqs=int(eff["max_num_seqs"]),
            max_model_len=int(eff["max_model_len"]), max_num_batched_tokens=eff.get("max_num_batched_tokens"),
            speculative=drafts > 0 if _whole(drafts) else None, ple_replicate=eff.get("ple_replicate"),
            kv_cache_memory=eff.get("kv_cache_memory"), tp=len(cfg.cluster_hosts) or 2, available_gib=None,
            **cfg.memory_policy(box_total(view, box))).total_gib
    except (ValueError, TypeError, KeyError):
        return None


def restart_basis(cfg, catalog, view, box: str, role: str) -> tuple[float | None, str, float]:
    """What a box has for a selection that would REPLACE the serving pair: `(available_gib, basis, credited_gib)`.

    While our rank serves there, what the box reports free already has the rank in it, and an estimate against
    that figure refuses the very selection that is running (UX audit 2026-10-02: every Apply preview read
    `45.7 GiB free now: this selection needs 76.0 GiB`). So: what the box HAD when the pair launched
    (`mem_at_launch`, basis `launch`) when the loop publishes it; else free now plus what our rank costs by
    its own estimate (`free+running`); else, with no pair of ours standing, free now (`free`)."""
    view = view if isinstance(view, dict) else {}
    published = view.get("boxes") if isinstance(view.get("boxes"), dict) else {}
    seen = published.get(box) if isinstance(published.get(box), dict) else {}
    free = _number(seen.get("mem_available_gib"))
    rank = seen.get("rank") if isinstance(seen.get("rank"), dict) else {}
    ours = bool(rank.get("running"))
    returned = running_rank_gib(cfg, catalog, view, box, role) if ours else None
    if returned is None:
        return free, "free", 0.0
    launched = _number(seen.get("mem_at_launch"))
    if launched is not None:
        return launched, "launch", round(launched - free, 1) if free is not None else 0.0
    if free is None:
        return None, "free", 0.0
    return round(free + returned, 1), "free+running", round(returned, 1)


def kv_auto_resolver(cfg, catalog, view_of):
    """`DesiredStore.set`'s `resolve_kv_auto` over the view the loop last published (`view_of()`): the selection a
    Start is about to store → the pin `auto` resolves to, or ValueError carrying the refusal — which POST
    /lb/cluster answers as the switch's 409, in the gate's words."""
    def resolve(selection: Desired) -> int:
        spec = catalog.get(selection.model_choice)
        if spec is None:
            raise ValueError(f"kv_cache_memory \"{KV_AUTO}\": the model choice {selection.model_choice!r} is not "
                             f"in the model catalog")
        view = view_of() or {}
        budget = selection.max_num_batched_tokens
        if budget is None:                       # the launch's own resolution, as the loop published it; else the fleet's
            published = (view.get("defaults") or {}).get("max_num_batched_tokens")
            budget = published if _whole(published) else cfg.cluster_max_num_batched_tokens
        auto = kv_auto(cfg, catalog, view, spec, max_num_seqs=selection.max_num_seqs,
                       max_model_len=selection.max_model_len, max_num_batched_tokens=budget,
                       num_speculative_tokens=selection.num_speculative_tokens,
                       ple_replicate=selection.ple_replicate)
        if auto.kv_cache_memory is None:
            raise ValueError(f"kv_cache_memory \"{KV_AUTO}\" cannot be sized now: {auto.refusal}")
        return auto.kv_cache_memory

    return resolve
