"""A MemoryEstimate per box for the cluster's selection — model, concurrency, context (M8: worklog/vllm-gateway-
program.md, pass 4's frame amendment; worklog/vllm-model-selection.md).

  MemoryEstimate   what one box needs for a selection, and the MemAvailable that leaves — marked against
                   broker_parrot's danger floor and the planner's start gate — and, once the pair runs, next to
                   what the box actually used.

The formula is the catalog's DATA, with no branch per model (tests/test_memory_estimate.py reads every constant
off the committed 2026-09-15 start):

  weights  = MemoryFootprint.weights_gib + ple_gib, or ple_gib / TP when the PLE table is SPLIT across the
             ranks — the LaunchProfile's placement, unless the operator chose one (`Desired.ple_replicate`)
  mtp      = MemoryFootprint.mtp_gib when the launch runs a speculative drafter — the LaunchProfile's, unless the
             operator turned it off (num_speculative_tokens 0) — else 0
  kv       = the KV pin when there is one — the operator's (`Desired.kv_cache_memory`) else the
             LaunchProfile's — else
             concurrency × (context × kv_bytes_per_token + seq_state_gib)
  budget   = MemoryFootprint.activation_gib_per_1k_tokens x (max_num_batched_tokens −
             calibrated_batched_tokens) / 1024, never below zero — what the PREFILL BUDGET costs beyond
             the budget `overhead_gib` was measured at (8,192 in the committed start log's own argv)
  seats    = MemoryFootprint.concurrency_gib_per_seq x (max_num_seqs − calibrated_max_num_seqs), never
             below zero — what the CONCURRENCY costs beyond the highest one MEASURED to rest safely on
             these boxes (16, 2026-10-01)
  total    = weights + mtp + kv + LaunchProfile.overhead_gib[role] + budget + seats
  leaves   = the MemAvailable the box reports now − total — or, once its rank is LAUNCHED, what the box had
             free at that launch − total: what it reports then is free WITH the rank loaded, and
             subtracting the rank from that counted it twice (`leaves -22.2 GiB` beside 45.8 free, 2026-10-02)
  needs    = total + the gateway's `cluster_start_headroom_gib` — what a box must have FREE to START

`overhead_gib` is one measured constant at ONE point (2026-09-15: max_num_seqs 1, max_num_batched_tokens
8192), so everything that moves away from that point has to be charged on top of it, or the estimate is
only right where it was taken. The prefill budget is charged because the operator was given a control
that raises it eightfold (2048..65536) while the estimate stayed blind: `batched 8192` and `batched
16384` asked the box for the identical 103.2 GiB, and on 2026-09-30 the pair ran at 16,384 down to
0.8 GiB MemAvailable and both GB10 boxes hung — NVRM `Out of memory [NV_ERR_NO_MEMORY]` in the kernel
log, the CUDA allocator on a box whose CPU and GPU share one LPDDR5X pool, recurring all evening and
last at the generation-80 handover (lb/tests/fixtures/cluster-b16384-2026-09-30-memory.json).

`max_num_seqs` moves away from that point too, and until 2026-10-01 it was deliberately not charged:
the only figures were CUDA graphs at 0.29 GiB for 1 seat and 0.59 for 2 (worklog/vllm-cluster-ram-
headroom.md §4), two points do not justify a slope out to 48 seats, and a coefficient nobody measured
that far would start refusing configurations that work. That judgement was right with that data. It is
charged now because the live pair supplied the data — same model, same context, same `batched 8192`:
at 16 seats the head RESTS at 11.3 GiB MemAvailable, at 24 seats it rests at 1.7 GiB with no traffic at
all, inside the 0.8-4.2 GiB band that hung both GB10 boxes the night before, with nothing left to absorb the
generation handover that is the minute they died (lb/tests/fixtures/cluster-seqs24-2026-10-01-memory.json).
The KV pool is pinned and allocated up front and the decode graphs are captured during startup, so that
is the configuration's RESTING cost, not a load effect.

The seat term is a BOUND, not a coefficient: the two idle samples differ in `num_speculative_tokens`
too (4 and 3), and that confound has a known sign — fewer proposals is a smaller uniform decode batch,
seqs x (spec + 1), so less captured graph and less per-step activation, never more — so the step from 4
to 3 released memory while the eight extra seats took it, and 1.2 GiB per seat understates a seat if
anything. It is charged from 16 rather than from the calibration's single seat because 16 is the highest
concurrency measured to rest safely, and charging below it would refuse what demonstrably works: the
two-point CUDA-graph slope would want ~4.4 GiB over 15 seats, where the pair's own footprint at 16 seats
is 1.0-1.5 GiB above the 1-seat estimate. The start headroom below is what covers being a little wrong.

A pin smaller than concurrency × context caps real concurrency, so the need is always carried beside the pin.
Classic vLLM's memory is broker_parrot's to size: it is never estimated here.

AUTO — the KV pool sized to what the boxes have left (`resolve_kv_auto`, `Desired.kv_cache_auto`). vLLM cannot
do this itself on these boxes: the pool is allocated once at engine init and never resized, and with no pin its
`--gpu-memory-utilization` is a fraction of the box's TOTAL memory (`requested_memory = total x util`,
vllm/v1/worker/utils.py `request_memory`), from which it subtracts only its OWN usage — what the host and every
other process hold never enters it; on an integrated GPU the one "free" it reads is psutil's MemAvailable
(vllm/utils/mem_utils.py `MemorySnapshot.measure`), and only to refuse a start. So the LB resolves a pin at Start
from what each box reports, keeping broker_parrot's floor UNDER LOAD:

  loaded = min(free now, free at rest − load)        load = max(the gateway's minimum, `load_gib`)
  pool   = loaded − floor + what our serving rank there returns − what the new selection costs besides its pool
  pin    = the smaller box's pool, rounded down to the panel's GB, refused under the pool that runs now

Measured on generation 99 (seqs 16, budget 4,096, PLE split, 35 GB): 19.4 / 21.8 GiB free at rest, 13.5 / 16.1
under 16 running sequences — 5.9 / 5.7 GiB of load (lb/tests/fixtures/cluster-load-2026-10-01-memory.json).
"""
from __future__ import annotations

from dataclasses import dataclass

from vllm_lb.catalog import CLUSTER_MODE, RANK_ROLES, ModelSpec

GIB = 1024 ** 3
CLASSIC_MEMORY = "set by broker_parrot"

#: The closed vocabulary of an estimate's warnings — the same codes on the page and in the tests.
GATE = "gate"                 # free memory now is under what THIS selection needs (`needed_now_gib`)
FLOOR = "floor"               # what the box would leave is under broker_parrot's danger floor
NEAR_FLOOR = "near_floor"     # … or within the margin above it
NO_HEADROOM = "no_headroom"   # it fits, but leaves the allocator less than a start may leave
BUDGET = "budget"             # …and the prefill budget above the calibrated one is why
CONCURRENCY = "concurrency"   # …or the concurrency above the anchor measured to rest safely is why
PIN_CAPS = "pin_caps"         # the KV pin holds fewer tokens than concurrency × context
PIN_FLOOR = "pin_floor"       # the KV pin holds less than ONE request of this context: the engine dies at init
DEVIATION = "deviation"       # once running, the estimate is off the actual by more than DEVIATION_PCT
#: Past this, the catalog's constants are wrong for this box and should be corrected.
DEVIATION_PCT = 10.0
#: The codes that are the SELECTION's, identical on every box — the page says each once, not per box.
PAIR_SCOPED = frozenset({PIN_CAPS, PIN_FLOOR})


def _tokens(n: int | None) -> str:
    """A token count as the page prints one: 1,302,224 → `1.30M`, 65,536 → `65,536`."""
    if n is None:
        return "—"
    return f"{n / 1e6:.2f}M" if n >= 1_000_000 else f"{n:,}"


@dataclass(frozen=True)
class MemoryEstimate:
    box: str
    role: str
    model_choice: str
    calibration: str
    weights_gib: float
    mtp_gib: float
    kv_needed_gib: float
    kv_pinned_gib: float | None
    overhead_gib: float
    kv_bytes_per_token: int
    needed_tokens: int
    available_gib: float | None
    floor_gib: float
    margin_gib: float
    actual_used_gib: float | None = None
    #: The prefill budget this launch would use (vLLM's `--max-num-batched-tokens`). None = the caller
    #: did not say, and then the budget is not charged.
    max_num_batched_tokens: int | None = None
    #: The footprint's own calibration anchor and slope, carried so the page and a post-mortem can see
    #: where the extra GiB came from. Both 0 = this model's budget was never measured.
    calibrated_batched_tokens: int = 0
    activation_gib_per_1k_tokens: float = 0.0
    #: The CONCURRENCY this launch would run (vLLM's `--max-num-seqs`), the anchor it is charged from,
    #: and what one seat above that anchor costs. The anchor is the highest concurrency MEASURED to rest
    #: safely on these boxes, not the seat count `overhead_gib` was taken at. Anchor or slope 0 = this
    #: model's concurrency was never measured, and then it is not charged.
    max_num_seqs: int = 0
    calibrated_max_num_seqs: int = 0
    concurrency_gib_per_seq: float = 0.0
    #: What a START must LEAVE free beyond what it costs — the gateway's policy, never a model's fact.
    start_headroom_gib: float = 0.0
    #: What SPLITTING the PLE table across the ranks frees on this box: ple_gib − ple_gib / tp. 0 = the
    #: model has no table to split.
    ple_split_frees_gib: float = 0.0
    #: What the box had free when its rank LAUNCHED (`cluster_loop._mem_at_launch`); None until then. Once
    #: set, `leaves_gib` is measured from it — `available_gib` is then what is free with the rank loaded.
    launched_gib: float | None = None
    #: The context this selection runs, for the words: "fits N @ ctx".
    max_model_len: int = 0
    #: The model's block rule (`catalog.KvBlocks`), None when not recorded; and the block count the ENGINE
    #: reports while it serves this pin, None otherwise. Together they make `pin_seats` the engine's figure.
    kv_blocks: object = None
    engine_blocks: int | None = None

    @property
    def kv_gib(self) -> float:
        return self.kv_pinned_gib if self.kv_pinned_gib is not None else self.kv_needed_gib

    @property
    def activation_gib(self) -> float:
        """What the PREFILL BUDGET costs beyond the budget the rest of this estimate was calibrated at.

        Never negative: we measured what a bigger budget costs, never what a smaller one releases, and a
        credit nobody measured could let a start through that should have waited."""
        if self.max_num_batched_tokens is None or not self.calibrated_batched_tokens:
            return 0.0
        over = max(0, int(self.max_num_batched_tokens) - self.calibrated_batched_tokens)
        return self.activation_gib_per_1k_tokens * over / 1024.0

    @property
    def concurrency_gib(self) -> float:
        """What the CONCURRENCY costs beyond the highest one measured to rest safely on these boxes.

        Never negative, for the same reason the budget is not credited: 1.2 GiB per seat is what the
        live pair LOST going from 16 seats to 24, and what a seat gives back below 16 was never
        measured — a credit nobody measured could let a start through that should have waited."""
        if not self.calibrated_max_num_seqs or not self.concurrency_gib_per_seq:
            return 0.0
        over = max(0, int(self.max_num_seqs) - self.calibrated_max_num_seqs)
        return self.concurrency_gib_per_seq * over

    @property
    def total_gib(self) -> float:
        return (self.weights_gib + self.mtp_gib + self.kv_gib + self.overhead_gib + self.activation_gib
                + self.concurrency_gib)

    @property
    def beside_pool_gib(self) -> float:
        """Everything this selection costs on the box EXCEPT its KV pool — what `auto` sizes the pool around."""
        return self.total_gib - self.kv_gib

    @property
    def load_gib(self) -> float:
        """What LOAD adds to this selection once traffic arrives: the activations of a full prefill chunk, which
        a pair at rest has not allocated yet and PyTorch's caching allocator keeps once it has. Charged up to the
        budget this estimate was calibrated at — past it `activation_gib` already charges the rest — at the
        model's own activation slope; 0 for a model whose budget was never measured.

        It is why the load is not one number: generations 98/99 (budget 4,096) lost 3.8-5.9 GiB to it, and
        generations 96/97 (8,192) 9.5-10.2 over hours of 16-sequence load (lb/tests/fixtures/
        cluster-load-2026-10-01-memory.json) — 4.4 and 8.8 at 1.1 GiB per 1k tokens, so at 4,096 the gateway's
        measured minimum (`cluster_kv_auto_load_gib`) is the larger and is what auto charges."""
        if self.max_num_batched_tokens is None or not self.activation_gib_per_1k_tokens:
            return 0.0
        tokens = int(self.max_num_batched_tokens)
        if self.calibrated_batched_tokens:
            tokens = min(tokens, self.calibrated_batched_tokens)
        return self.activation_gib_per_1k_tokens * tokens / 1024.0

    @property
    def calibrated_total_gib(self) -> float:
        """What this same selection would cost at the calibrated prefill budget — the number that says
        whether lowering that one knob is the answer."""
        return self.total_gib - self.activation_gib

    @property
    def calibrated_seqs_total_gib(self) -> float:
        """…and at the anchor concurrency — the number that says whether lowering that one dropdown is."""
        return self.total_gib - self.concurrency_gib

    @property
    def calibrated_knobs_total_gib(self) -> float:
        """What the selection costs with EVERY knob of the operator's back at the point it was measured
        at. A knob is only blamed when this would start, because advice the operator can act on has to
        be advice that works: on a box that could not hold the selection with both knobs back, neither
        knob is the problem."""
        return self.total_gib - self.activation_gib - self.concurrency_gib

    def _fits_at(self, total: float) -> bool:
        """Would this box START the selection at that cost? Only then may a knob's message say "fits"."""
        return self.available_gib is not None and self.available_gib >= total + self.start_headroom_gib

    @property
    def leaves_gib(self) -> float | None:
        """What this box would have free with the rank loaded: free now − total before the launch; free AT the
        launch − total once the rank is up (what it reports then already has the rank in it)."""
        base = self.launched_gib if self.launched_gib is not None else self.available_gib
        return None if base is None else base - self.total_gib

    @property
    def needed_now_gib(self) -> float:
        """THE REQUIREMENT: what this box must have FREE before this rank launches on it.

        What the selection costs — model, context, concurrency, prefill budget — PLUS the headroom a
        start must leave (`start_headroom_gib`, 0 unless the gateway sets it). The planner waits on this
        exact number and the page warns about this exact number, so "need N" can never again be
        decoration (worklog/vllm-cluster-handover.md H3; live 2026-09-16 00:58:06Z the planner announced
        `need 100` for every model at every setting, a constant it had never measured). No configured
        requirement stands in for a selection that cannot be estimated: the planner refuses it, naming why.

        The headroom is here and not in `total_gib` because it is not a cost: `total_gib` is what the
        rank takes, and is what the actual is compared against. It exists because a start that fits by
        its estimate alone is a start betting the box on the estimate being right, and these captures
        say it runs 3.2-6.0 % LOW (gen 52: predicted 103.2, the head really took 106.6). Measured, the
        boundary between a configuration that serves and one that hangs these boxes lies between 4.2
        and 9.1 GiB of MemAvailable left."""
        return self.total_gib + self.start_headroom_gib

    @property
    def deviation_pct(self) -> float | None:
        if self.actual_used_gib is None or self.actual_used_gib <= 0:
            return None
        return 100.0 * (self.total_gib - self.actual_used_gib) / self.actual_used_gib

    @property
    def pin_tokens(self) -> int | None:
        return None if self.kv_pinned_gib is None else int(self.kv_pinned_gib * GIB / self.kv_bytes_per_token)

    @property
    def pin_seats(self) -> int | None:
        """How many requests of this context the pin holds at once — the figure beside the cap: the ENGINE's
        rule (blocks over blocks-per-request, its own block count while it serves) when the model records
        it, else tokens over context."""
        if self.kv_pinned_gib is None or not self.max_model_len:
            return None
        if self.kv_blocks is not None:
            return self.kv_blocks.seats(int(self.kv_pinned_gib * GIB), int(self.max_model_len),
                                        pool_blocks=self.engine_blocks)
        return self.pin_tokens // int(self.max_model_len)

    @property
    def kv_floor_gib(self) -> float | None:
        """The smallest pool that holds ONE request of this context — under it the engine does not start.
        None for a model whose block rule is not recorded."""
        if self.kv_blocks is None or not self.max_model_len:
            return None
        return self.kv_blocks.min_pool_bytes(int(self.max_model_len)) / GIB

    @property
    def codes(self) -> tuple[str, ...]:
        out = []
        # PRE-FLIGHT ONLY. The gate, the floor and its margin answer one question — should this be
        # started on this box? — and a launched rank has already answered it. `available_gib` is then
        # what is free WITH the model loaded, so what it "leaves" goes negative, and warning that a
        # serving pair is under the danger floor would be noise. The deviation below is the check that
        # only exists after a launch.
        if self.available_gib is not None and self.actual_used_gib is None:
            if self.available_gib < self.total_gib:
                out.append(GATE)                      # it does not fit at all
            elif self.available_gib < self.needed_now_gib:
                out.append(NO_HEADROOM)               # it fits, and a start may not take the last of it
            # …and a KNOB of the operator's is NAMED as the reason only when turning their knobs back to
            # what was measured would actually make this start possible. On a box that could not hold
            # the selection either way, "lower the budget" is advice that does not work. Both are named
            # when both are adding, because a refusal that blames one of two knobs sends the operator to
            # restart the pair at a setting that is still refused.
            short = self.available_gib < self.needed_now_gib
            fixable = self._fits_at(self.calibrated_knobs_total_gib)
            if self.activation_gib and short and fixable:
                out.append(BUDGET)
            if self.concurrency_gib and short and fixable:
                out.append(CONCURRENCY)
            if self.leaves_gib < self.floor_gib:
                out.append(FLOOR)
            elif self.leaves_gib < self.floor_gib + self.margin_gib:
                out.append(NEAR_FLOOR)
        floor = self.kv_floor_gib
        if self.kv_pinned_gib is not None and floor is not None and self.kv_pinned_gib < floor:
            out.append(PIN_FLOOR)                     # one request does not fit: nothing else about the pin matters
        elif self.kv_pinned_gib is not None and self.kv_needed_gib > self.kv_pinned_gib:
            out.append(PIN_CAPS)
        if self.deviation_pct is not None and abs(self.deviation_pct) > DEVIATION_PCT:
            out.append(DEVIATION)
        return tuple(out)

    def message(self, code: str) -> str:
        """ONE LINE OF FIGURES per code — GiB throughout, no sentence (operator rule, 2026-10-01): the page
        prints it beside the box as it is."""
        fits = " · fits"
        if code == GATE:
            return f"free {self.available_gib:.1f} < need {self.needed_now_gib:.1f} GiB"
        if code == NO_HEADROOM:
            return (f"free {self.available_gib:.1f} < {self.total_gib:.1f} + {self.start_headroom_gib:.0f} "
                    f"headroom GiB")
        if code == BUDGET:
            return (f"batched {self.max_num_batched_tokens:,}: +{self.activation_gib:.1f} GiB · "
                    f"@ {self.calibrated_batched_tokens:,} needs {self.calibrated_total_gib:.1f}"
                    f"{fits if self._fits_at(self.calibrated_total_gib) else ''}")
        if code == CONCURRENCY:
            return (f"seqs {self.max_num_seqs}: +{self.concurrency_gib:.1f} GiB · "
                    f"@ {self.calibrated_max_num_seqs} needs {self.calibrated_seqs_total_gib:.1f}"
                    f"{fits if self._fits_at(self.calibrated_seqs_total_gib) else ''}")
        if code == FLOOR:
            return f"leaves {self.leaves_gib:.1f} < floor {self.floor_gib:.0f} GiB"
        if code == NEAR_FLOOR:
            return f"leaves {self.leaves_gib:.1f} < floor {self.floor_gib:.0f} + {self.margin_gib:.0f} GiB"
        if code == PIN_CAPS:
            seats = "" if self.pin_seats is None else f" · fits {self.pin_seats} @ {self.max_model_len:,}"
            return f"pin {_tokens(self.pin_tokens)} < {_tokens(self.needed_tokens)} tokens{seats}"
        if code == PIN_FLOOR:
            return (f"pin {self.kv_pinned_gib:.1f} < {self.kv_floor_gib:.1f} GiB for one {self.max_model_len:,} "
                    f"request ({self.kv_blocks.per_request(int(self.max_model_len))} blocks)")
        if code == DEVIATION:
            return (f"est {self.total_gib:.1f} vs actual {self.actual_used_gib:.1f} GiB "
                    f"({'+' if self.deviation_pct > 0 else '-'}{abs(self.deviation_pct):.0f} %)")
        return ""

    @staticmethod
    def scope(code: str) -> str:
        """`pair`: a fact of the selection, the same on every box, said ONCE on the page. `box`: this box's."""
        return "pair" if code in PAIR_SCOPED else "box"

    @property
    def needs_note(self) -> str:
        """The clause a REFUSAL carries beyond the numbers: which knobs the operator can turn, and what the
        selection would need without each. Empty when no knob of theirs is adding anything, so the planner's
        state line only ever names a setting that is actually responsible — and it names BOTH when both
        are, because sending them to lower one of two knobs is sending them to be refused again."""
        notes = []
        if self.activation_gib:
            notes.append(f"batched {self.max_num_batched_tokens:,} +{self.activation_gib:.1f} GiB "
                         f"(@ {self.calibrated_batched_tokens:,}: {self.calibrated_total_gib:.0f})")
        if self.concurrency_gib:
            notes.append(f"seqs {self.max_num_seqs} +{self.concurrency_gib:.1f} GiB "
                         f"(@ {self.calibrated_max_num_seqs}: {self.calibrated_seqs_total_gib:.0f})")
        return f" — {', '.join(notes)}" if notes else ""

    def view(self) -> dict:
        def gib(v):
            return None if v is None else round(v, 1)
        return {"box": self.box, "role": self.role, "model_choice": self.model_choice,
                "calibration": self.calibration, "weights_gib": gib(self.weights_gib), "mtp_gib": gib(self.mtp_gib),
                "kv_gib": gib(self.kv_gib), "kv_needed_gib": gib(self.kv_needed_gib),
                "kv_pinned_gib": gib(self.kv_pinned_gib), "overhead_gib": gib(self.overhead_gib),
                "total_gib": gib(self.total_gib), "available_gib": gib(self.available_gib),
                "activation_gib": gib(self.activation_gib),
                "max_num_batched_tokens": self.max_num_batched_tokens,
                "calibrated_batched_tokens": self.calibrated_batched_tokens,
                # THE SLOPES the page's notes are written from, per MODEL (0 = never measured, not counted).
                "activation_gib_per_1k_tokens": self.activation_gib_per_1k_tokens,
                "concurrency_gib": gib(self.concurrency_gib), "max_num_seqs": self.max_num_seqs,
                "calibrated_max_num_seqs": self.calibrated_max_num_seqs,
                "concurrency_gib_per_seq": self.concurrency_gib_per_seq,
                "ple_split_frees_gib": gib(self.ple_split_frees_gib),
                "needed_now_gib": gib(self.needed_now_gib), "start_headroom_gib": self.start_headroom_gib,
                "leaves_gib": gib(self.leaves_gib), "floor_gib": self.floor_gib,
                "launched_gib": gib(self.launched_gib), "max_model_len": self.max_model_len,
                # The engine's rule: the pool under which ONE request of this context does not start, and the
                # seats the pin holds by blocks (null for a model whose block rule is not recorded).
                "kv_floor_gib": gib(self.kv_floor_gib), "pin_seats": self.pin_seats,
                "actual_used_gib": gib(self.actual_used_gib),
                "deviation_pct": None if self.deviation_pct is None else round(self.deviation_pct, 1),
                "warnings": [{"code": c, "message": self.message(c), "scope": self.scope(c)} for c in self.codes]}


def estimate_memory(spec: ModelSpec | None, *, box: str, role: str, max_num_seqs: int, max_model_len: int, tp: int,
                    available_gib: float | None, floor_gib: float, margin_gib: float,
                    actual_used_gib: float | None = None, speculative: bool | None = None,
                    max_num_batched_tokens: int | None = None,
                    ple_replicate: bool | None = None, kv_cache_memory: int | None = None,
                    start_headroom_gib: float = 0.0, launched_gib: float | None = None,
                    engine_blocks: int | None = None, kv_cache_dtype: str | None = None) -> MemoryEstimate:
    """One box's MemoryEstimate for a cluster selection. Refuses what it cannot estimate — a spec with no cluster
    LaunchProfile or MemoryFootprint (classic vLLM: set by broker_parrot), or a role no rank has.

    `speculative`: whether this launch loads the drafter. None: as the LaunchProfile says; False: the operator
    turned it off, and its weights are not there. True never adds a drafter a profile does not have.

    `max_num_seqs`: the concurrency this launch would run. It is charged above the footprint's own
    measured anchor (`calibrated_max_num_seqs`), never below it, and not at all for a model whose
    concurrency was never measured.

    `max_num_batched_tokens`: the prefill budget this launch would RESOLVE to — the operator's knob, else the
    model's own, else the fleet's (`CLUSTER_MAX_NUM_BATCHED_TOKENS` in every call's env). None leaves the budget
    uncharged, so every caller resolves it rather than passing what the operator happened to type.
    `start_headroom_gib`: the gateway's policy for what a start must LEAVE free; 0 charges nothing extra.
    `launched_gib`: what the box had free when its rank LAUNCHED, once it has; `leaves_gib` is then measured
    from it, because `available_gib` already has the rank in it."""
    profile = spec.profile(CLUSTER_MODE) if spec is not None else None
    if spec is None or profile is None or spec.memory is None:
        raise ValueError(f"{CLASSIC_MEMORY}: {spec.id if spec is not None else 'no model'} has no cluster "
                         f"LaunchProfile to estimate")
    # THE RUN'S KV DTYPE re-prices every KV figure (`ModelSpec.at_kv_cache_dtype`); None is the profile's own.
    spec = spec.at_kv_cache_dtype(kv_cache_dtype)
    if role not in RANK_ROLES or role not in profile.overhead_gib:
        raise ValueError(f"no rank role {role!r}: the pair's ranks are {', '.join(RANK_ROLES)}")
    m = spec.memory
    # A KNOB THE ENGINE DOES NOT TAKE (`LaunchProfile.refuses`: SGLang sizes its pools by its mem fraction and reads
    # no PLE placement) is charged as the engine runs it — the profile's — never as a value the launch drops.
    if "ple_replicate" in profile.refuses:
        ple_replicate = None
    if "kv_cache_memory" in profile.refuses:
        kv_cache_memory = None
    # THE OPERATOR'S PLACEMENT OVERRIDES THE PROFILE'S, and the estimate has to follow it or the start gate
    # charges for a table that is not there (refusing a split that fits) — or, worse, fails to charge for one
    # that is (admitting a replicated table the box cannot hold). Split: ple_gib / tp on every rank.
    replicate = profile.ple_replicate if ple_replicate is None else ple_replicate
    ple = m.ple_gib / tp if replicate is False else m.ple_gib
    # Same for the KV pin: it is an operator knob now, and it is the single largest number the gate adds up.
    pin = profile.kv_cache_memory if kv_cache_memory is None else kv_cache_memory
    return MemoryEstimate(
        box=box, role=role, model_choice=spec.id, calibration=m.calibration,
        weights_gib=m.weights_gib + ple,
        # None is the profile's DEFAULT depth (its tuned one first): a drafter that default runs off (DeepSeek-V4's
        # dspark, tuned 0) loads no weights — the depth launch_env sends, never the config's own count.
        mtp_gib=(m.mtp_gib if profile.speculative_config is not None
                 and (speculative if speculative is not None else profile.num_speculative_tokens > 0) else 0.0),
        kv_needed_gib=max_num_seqs * (max_model_len * m.kv_bytes_per_token / GIB + m.seq_state_gib),
        kv_pinned_gib=pin / GIB if pin else None,
        overhead_gib=float(profile.overhead_gib[role]), kv_bytes_per_token=m.kv_bytes_per_token,
        needed_tokens=max_num_seqs * max_model_len, available_gib=available_gib, floor_gib=floor_gib,
        margin_gib=margin_gib, actual_used_gib=actual_used_gib,
        max_num_batched_tokens=None if max_num_batched_tokens is None else int(max_num_batched_tokens),
        calibrated_batched_tokens=m.calibrated_batched_tokens,
        activation_gib_per_1k_tokens=m.activation_gib_per_1k_tokens,
        max_num_seqs=int(max_num_seqs), calibrated_max_num_seqs=m.calibrated_max_num_seqs,
        concurrency_gib_per_seq=m.concurrency_gib_per_seq,
        start_headroom_gib=float(start_headroom_gib),
        ple_split_frees_gib=(m.ple_gib - m.ple_gib / tp) if profile.ple_replicate is not None and tp else 0.0,
        launched_gib=None if launched_gib is None else float(launched_gib), max_model_len=int(max_model_len),
        kv_blocks=m.kv_blocks, engine_blocks=engine_blocks if isinstance(engine_blocks, int) and engine_blocks > 0
        else None)


# ── AUTO: the KV pool sized to what the boxes have left ───────────────────────


@dataclass(frozen=True)
class KvAutoBox:
    """One box's share of an `auto` KV pool: the largest pool it can hold and still keep broker_parrot's danger
    floor once LOAD has taken its share — the floor UNDER LOAD, because that is where the memory goes.

        loaded = min(free now, free at rest − load)      what this box will have free under load
        pool   = loaded − floor + returned − beside       the pool that leaves it exactly the floor

    `free now` is the lowest of the box's recent observations (`mem_stable_gib`). `free at rest` is the most it
    has had while the serving pair rested (`mem_rest_gib`, kept with the launch record): load grows a pair and
    the caching allocator keeps what it grew, so a box read after its first busy minutes has ALREADY paid part
    of the load, and charging all of it again on that figure would pay it twice (generation 99: 19.4 GiB at
    rest, 13.5 under load — the same pair). With no resting figure, free now is taken as one. `returned` is what
    our serving rank here costs by its own estimate — it comes back when a restart replaces it, 0 when no pair
    serves — and `beside` what the new selection costs besides its pool (`MemoryEstimate.beside_pool_gib`). With
    the settings unchanged the two differ by exactly the running pin, which is the operator's formula:
    pin = current pin + (free − floor − load)."""
    box: str
    available_gib: float | None
    rest_gib: float | None
    returned_gib: float
    beside_pool_gib: float
    floor_gib: float
    load_gib: float

    @property
    def loaded_gib(self) -> float | None:
        if self.available_gib is None:
            return None
        rest = self.available_gib if self.rest_gib is None else max(self.rest_gib, self.available_gib)
        return min(self.available_gib, rest - self.load_gib)

    @property
    def pool_gib(self) -> float | None:
        loaded = self.loaded_gib
        return None if loaded is None else loaded - self.floor_gib + self.returned_gib - self.beside_pool_gib

    def left_gib(self, pin: int) -> float | None:
        """What this box would have free under load with a pool of `pin` bytes."""
        loaded = self.loaded_gib
        return None if loaded is None else loaded + self.returned_gib - self.beside_pool_gib - pin / GIB

    def view(self) -> dict:
        def gib(v):
            return None if v is None else round(v, 1)
        return {"box": self.box, "available_gib": gib(self.available_gib), "rest_gib": gib(self.rest_gib),
                "load_gib": gib(self.load_gib), "loaded_gib": gib(self.loaded_gib),
                "returned_gib": gib(self.returned_gib), "beside_pool_gib": gib(self.beside_pool_gib),
                "floor_gib": self.floor_gib, "pool_gib": gib(self.pool_gib)}


@dataclass(frozen=True)
class KvAuto:
    """What `auto` resolves to: the pin both ranks take, the box that set it — or why there is none."""
    boxes: tuple[KvAutoBox, ...]
    #: Bytes, a whole number of the panel's GB; None = refused, and `refusal` says why.
    kv_cache_memory: int | None
    #: The pool it may not fall under: the one the serving pair runs, else the model's own. None = only the
    #: range binds.
    baseline: int | None = None
    #: The box with the least room. TP=2 ranks take ONE pin, so the tighter box sizes both.
    binding: str | None = None
    refusal: str = ""

    def view(self) -> dict:
        """`auto` in GET /lb/cluster/estimate — what a Start at `auto` would pin NOW, and each box's share."""
        return {"kv_cache_memory": self.kv_cache_memory, "baseline": self.baseline, "binding": self.binding,
                "refusal": self.refusal, "boxes": [b.view() for b in self.boxes]}


def resolve_kv_auto(boxes, *, baseline: int | None, step: int, pin_range: tuple[int, int]) -> KvAuto:
    """THE ONE POOL BOTH RANKS PIN, sized to the box with the least room:

        pin = min over the boxes of their pool, rounded DOWN to `step`, never past the range's ceiling

    REFUSED, in the start gate's own words ("short N GiB"), when it would not reach `baseline` — auto fills
    what is left; it never quietly shrinks the pool the pair already runs, or a model's own when none runs —
    or the range's floor, and when any box's free memory has not been observed: a pool sized to a guess is
    the 2026-09-16 launch onto a box that was really at 77 GiB."""
    boxes = tuple(boxes)
    if not boxes:
        return KvAuto(boxes, None, baseline, None, "there is no box to size the pool against")
    unknown = [b.box for b in boxes if b.pool_gib is None]
    if unknown:
        return KvAuto(boxes, None, baseline, None,
                      f"{' and '.join(unknown)}: free memory not observed yet — auto sizes the pool from what the "
                      f"boxes report, never from a guess")
    lo, hi = pin_range
    binding = min(boxes, key=lambda b: b.pool_gib)
    raw = int(binding.pool_gib * GIB)
    pin = min(hi, raw // step * step)
    if baseline is not None and raw >= baseline:
        pin = max(pin, min(int(baseline), hi))       # the step's rounding never takes it under a pool that fits
    least = lo if baseline is None else max(lo, int(baseline))
    if raw < least:
        b = binding
        left = b.left_gib(least)
        what = (f"even the current {least / 1e9:.0f} GB pool" if baseline is not None and least == baseline
                else f"the smallest pool ({least / 1e9:.0f} GB)")
        return KvAuto(boxes, None, baseline, b.box,
                      f"{b.box}: {b.available_gib:.1f} GiB free now, {b.loaded_gib:.1f} under load — {what} would "
                      f"leave it {left:.1f} GiB, under broker_parrot's {b.floor_gib:.0f} GB danger floor: short "
                      f"{b.floor_gib - left:.1f} GiB")
    return KvAuto(boxes, pin, baseline, binding.box)
