"""Is there headroom left, or is the pair flat out — and WHICH ceiling it is against.

The operator asked for "real GPU usage, GPU cores fill rate, teraflops/s rate in real time, so we
can indicate if any process can still be optimised or is fully saturated already", after a 96 % GPU
tile had already misled them once. That tile was not wrong; it was answering a different question.
This module exists to keep the two apart:

* **SM fill is not compute utilisation.** `nvidia-smi`'s `sm` is the share of the sample during which
  at least one kernel was resident — occupancy in TIME. A memory-bound decode keeps the SMs resident
  ~96 % of the time while using a small fraction of the card's FLOP/s. So `sm_pct` is reported as
  *SM fill*, and never as "the GPU is 96 % used up".
* **`nvidia-smi`'s `mem` is not bandwidth either.** It is a memory duty-cycle, and on a GB10 — whose
  LPDDR5X is UNIFIED with the host's RAM — it reads 0 even at `sm 96 %` (measured on box-a and box-b,
  2026-09-29). It is carried through under `mem_duty_pct` and never called bandwidth.
* **The only honest FLOP/s and GB/s come from the ENGINE.** vLLM 0.29 estimates both analytically per
  engine step and exports them as counters — `vllm:estimated_flops_per_gpu_total`,
  `vllm:estimated_read_bytes_per_gpu_total`, `vllm:estimated_write_bytes_per_gpu_total` — from which a
  rate is TFLOP/s and GB/s per GPU (the formula is vLLM's own, `v1/metrics/perf.py`). They are only
  fed when the engine was started with `--enable-mfu-metrics`; the counters exist regardless and sit
  at 0.0. They move once per COMPLETED step, so a counter flat for one window while requests run is a
  step in flight; "the engine is not measuring it" is said only on evidence (tokens out, counter still
  0 — `throughput.Throughput.engine_perf`), and a flat counter is never shown as 0.
* **Those two rates are not per box.** Only the head rank exposes /metrics (rank 1 is --headless), and
  vLLM's estimate is the model's cost per step divided by the TP size — the same for both ranks by
  construction. The panel shows them once, labelled as that (`RATES_MEAN`), and never as a per-rank
  figure. What each box DOES measure on its own — SM fill, power, clock, temperature — is shown per
  rank, in rank order, each SM fill with its own history (`History`).

* **For an SGLang generation the GB/s is the LB's MODEL, not the engine's.** SGLang's own byte estimate
  charges the whole weight read per generated token and read 73,600 GB/s a GPU live; the LB prices each
  forward pass from the catalog instead (`throughput.Throughput.bandwidth_model`) and passes it as
  `bytes_model`, which the block carries as `bandwidth_model` and the page labels "modeled".
* **…and so is its TFLOP/s.** SGLang's FLOP estimate charges the 27B's 64 layers as dense attention with an
  unsharded MLP — 236.8 TFLOP/s beside a CUDA CORES tile at 742 of 6,144 cores. The LB prices the window's
  passes itself (cores.py `modeled_flops`, the one figure the cores tile draws too) and passes it as
  `flops_model`, which the block carries as `compute_model` with SGLang's own figure beside it.

The verdict is therefore about what IS measured. `BANDWIDTH_BOUND` is only ever said when a real GB/s
figure exists; without one the page says so and stops, because "compute saturated" inferred from
`sm 96 %` alone is the exact mistake this module was written to prevent.

Pure: no I/O, no clock, and nothing from the LB but the equally pure `rates` vocabulary. The caller
passes what it has read.
"""
from __future__ import annotations

from typing import Any, NamedTuple

from vllm_lb.rates import MFU_NOT_MEASURED

# ── the closed set of verdicts ─────────────────────────────────────────────────
IDLE = "idle"
BANDWIDTH_BOUND = "memory-bandwidth bound"
COMPUTE_BOUND = "compute saturated"
BUSY_UNMEASURED = "busy — no ceiling measured"
UNKNOWN = "not observed"
VERDICTS = frozenset({IDLE, BANDWIDTH_BOUND, COMPUTE_BOUND, BUSY_UNMEASURED, UNKNOWN})

#: Why a verdict cannot go further than BUSY_UNMEASURED when the switch is PROVEN off — the one
#: sentence the page shows, and the one place the deploy step is named, so the operator reads it where
#: the gap is, not in a worklog. Said only when the engine's rates carry `MFU_NOT_MEASURED`, i.e. the
#: engine produced tokens while its FLOP counter stayed at 0; any other missing rate says its own why.
MFU_OFF = ("the engine exports no FLOP/s or GB/s: its FLOP counter is still 0 after it produced tokens, "
           "i.e. it was started without --enable-mfu-metrics, so only SM fill is measured here")
NO_GPU = "no box reported its GPU"
#: One rank's entry when its box sent no GPU reading: a gap, never a zeroed reading.
NO_GPU_BOX = ("no GPU reading from this box: it could not be asked, nvidia-smi did not answer, or its "
              "GPU telemetry is off")
#: What the TFLOP/s and GB/s ARE, stated beside them, because a per-box reading would be natural to
#: assume on a two-box panel and none exists (module docstring).
def rates_mean(engine: str = "vllm", modeled: bool = False, flops_modeled: bool = False) -> str:
    """What the two rates ARE, naming the engine that published them. It said "vLLM's" unconditionally
    until 2026-10-03, which was true of every generation until the pair first served SGLang — and then
    told the operator the wrong engine's name on a figure whose whole purpose is to say where it came
    from (worklog/sglang-ui-audit.md C6). `modeled`: the GB/s is the LB's own model, not the engine's;
    `flops_modeled`: so is the TFLOP/s."""
    name = ENGINE_NAMES.get(engine, engine)
    if modeled and flops_modeled:
        return (f"both modeled by this LB per GPU from {name}'s own pass, token, context and busy-time counters: "
                f"the FLOP/s prices the model's real layers, split by TP, and full attention only where the model "
                f"runs it, at the depth each prompt token attends to — the figure the CUDA CORES tile draws; the GB/s "
                f"is the weights a pass reads (all of a dense model, the experts its rows touch of a routed one), the "
                f"KV over every running context, the GDN state. Not {name}'s own estimates, and the same for both "
                f"ranks by construction, not measured on either box")
    if modeled:
        return (f"{name}'s analytical FLOP estimate per GPU, read off the head rank's /metrics; the GB/s is "
                f"modeled by this LB from {name}'s own pass, context and sequence counters — the weights a pass reads, "
                f"the KV over every running context, the GDN state — not {name}'s byte estimate: the same "
                f"for both ranks by construction, not measured on either box")
    return (f"{name}'s analytical estimate per GPU — the model's FLOPs and "
            f"bytes for each engine step, divided by the TP size — read off the head rank's /metrics: "
            f"the same for both ranks by construction, not measured on either box")


#: How each engine is written where an operator reads it.
ENGINE_NAMES = {"vllm": "vLLM", "sglang": "SGLang", "tensorfold": "TensorFold"}
#: The vLLM wording, kept as the name the rest of the code imported before `rates_mean` existed.
RATES_MEAN = rates_mean("vllm")

#: An SM fill at or above this is "the SMs always have work" — busy, whatever the ceiling turns out
#: to be. Below it the pair has idle time, and no ceiling can be the binding one.
BUSY_SM_PCT = 50.0
#: Bandwidth at or above this share of the box's peak is the binding constraint. The peak it is judged
#: against (the serving card's hardware type: `peak_gbps`) is the VENDOR'S NAMEPLATE, and a DRAM bus does not reach its
#: nameplate — so the threshold sits well under 100 %. Raise it only against a measured achievable
#: figure for these boxes; nothing in this repo establishes one yet.
BOUND_BANDWIDTH_PCT = 70.0

#: Below this share of peak, a bandwidth figure taken while the SMs are BUSY is not believable and
#: is treated as unmeasured rather than reported.
#:
#: vLLM's MFU model drops any component whose quantisation it does not recognise
#: (`v1/metrics/perf.py`, `FfnQuantizationConfigParser` raises `InvalidComponent`, and
#: `_QUANT_WEIGHT_BYTE_SIZE` has no `modelopt_mixed`). This checkpoint IS `modelopt_mixed`, so its
#: 512-expert MoE FFN — 67.9 GB of the 105 GB checkpoint — is excluded from
#: `vllm:estimated_read_bytes_per_gpu_total`. Measured 2026-09-29: 87-88 % SM fill reported
#: alongside 2.0-2.6 GB/s, i.e. ~1 % of peak. Reporting that share would tell the operator the box
#: has 99 % bandwidth headroom while it is in fact saturated — the exact false claim this panel
#: exists to prevent. So: name the gap instead of printing the number.
IMPLAUSIBLE_BANDWIDTH_PCT = 10.0

BYTES_INCOMPLETE = ("the engine's byte estimate excludes this model's quantised weights (vLLM's MFU model "
                    "has no entry for its quantisation), so no share of peak is claimed")

FLOPS_INCOMPLETE = ("the engine's FLOP estimate excludes this model's quantised weights (vLLM's MFU model "
                    "has no entry for its quantisation), so this rate is a floor, not the real one")

#: A busy pair computing essentially nothing is the same dropped-component symptom as the byte rate.
IMPLAUSIBLE_TFLOPS = 0.5
#: And compute is only called saturated against a peak the deployment actually configured.
BOUND_COMPUTE_PCT = 70.0


class Ceiling(NamedTuple):
    """One measured rate against the peak it is judged on. `pct` is None when there is no peak to
    judge it against — the rate is still a measurement, and is still shown."""
    rate: float | None
    peak: float | None
    pct: float | None
    why: str | None

    def view(self) -> dict:
        return {"rate": _round(self.rate), "peak": self.peak, "pct": _round(self.pct),
                "why": self.why}


def _round(value: float | None, places: int = 1) -> float | None:
    return None if value is None else round(value, places)


def _num(value: Any) -> float | None:
    """A number, or None. A bool is not a number here."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _share(rate: float | None, peak: float | None) -> float | None:
    if rate is None or peak is None or peak <= 0:
        return None
    return 100.0 * rate / peak


#: HOW A PER-BOX RATE IS DERIVED, said beside it wherever more than one card serves — never presented as measured.
#: The engine publishes ONE rate for the whole pair (vLLM's `estimated_*_per_gpu` off the head rank — rank 1 runs
#: headless, with no /metrics — or the LB's own model for SGLang) and no per-rank counter; tensor parallelism splits
#: every layer's work evenly over the ranks, so a box's share is that per-GPU rate times the serving cards it holds.
DERIVED_PER_BOX = ("derived: the engine's one rate per GPU (it has no per-rank counter) x this box's serving cards — "
                   "tensor parallelism splits each layer's work evenly over the ranks")


def per_box(rate_per_gpu: float | None, cards, *, withheld: str | None = None) -> list[dict]:
    """The rate SPLIT PER SERVING BOX, each against its own box's peak (operator, 2026-10-11): never one figure for
    the LB or the pair. `cards` is the serving cards HEAD FIRST as (box, peak, why) — the peak its hardware type
    states (at the model's precision, for compute), None with the reason when it states none.

    A box's rate is the per-GPU rate times its serving cards and its peak their peaks summed, so the boxes' rates add
    back up to the engine-wide rate (per GPU x cards). `derived` carries DERIVED_PER_BOX wherever more than one card
    serves — a share of one engine-wide figure — and None where one card is the whole engine. `withheld`: the reason
    the pair's own share was withheld (`judge`: an incomplete estimate) — then no box claims one either."""
    order: list[str] = []
    by_box: dict[str, list] = {}
    for box, peak, why in cards:
        if box not in by_box:
            order.append(box)
            by_box[box] = []
        by_box[box].append((_num(peak), why))
    derived = DERIVED_PER_BOX if len(cards) > 1 else None
    out = []
    for box in order:
        peaks = by_box[box]
        rate = None if _num(rate_per_gpu) is None else rate_per_gpu * len(peaks)
        known = all(p is not None and p > 0 for p, _ in peaks)
        peak = sum(p for p, _ in peaks) if known else None
        why = withheld or (None if known else next(w for p, w in peaks if p is None or p <= 0))
        out.append({"box": box, "cards": len(peaks), "rate": _round(rate), "peak": peak,
                    "pct": None if withheld else _round(_share(rate, peak)), "why": why, "derived": derived})
    return out


def _fill(gpus: list[dict]) -> float | None:
    """The BUSIEST box's SM fill. Two ranks of one model do the same work in lockstep; the pair is as
    saturated as its busier half, and an average would hide the half that is pinned."""
    seen = [v for v in (_num(g.get("sm_pct")) for g in gpus) if v is not None]
    return max(seen) if seen else None


def gpu_view(raw: Any) -> dict | None:
    """One box's `gpu` object from `tp-rank status`, keeping only the keys it may state and only as
    numbers — or None when the box did not report a GPU at all. `headroom_pct` is the clock's
    distance from its own ceiling, computed here so the page never divides.

    A key the box did not send stays absent rather than becoming 0: `pwr_w` is `[N/A]` in
    `--query-gpu` form on a GB10, and a 0 W GPU would read as a dead one."""
    if not isinstance(raw, dict):
        return None
    out = {key: _num(raw.get(key)) for key in
           ("count", "sm_pct", "mem_duty_pct", "pwr_w", "gtemp_c", "pclk_mhz", "pclk_max_mhz")}
    if out["sm_pct"] is None:
        return None                      # a GPU object with no fill in it observed nothing
    out["count"] = None if out["count"] is None else int(out["count"])
    out["clock_pct"] = _round(_share(out["pclk_mhz"], out["pclk_max_mhz"]))
    return out


class History:
    """A few minutes of the saturation readings, so the panel shows a trend and not just the instant a
    poll happened to land on: the pair's three (busier SM fill, GB/s, TFLOP/s) and EACH box's own SM
    fill, all on one shared clock (`at`) so the two ranks can be read side by side.

    Sampled on READ with a minimum interval, deliberately: the block is derived, not probed, so
    there is nothing to hook a timer to — and gating on elapsed time keeps the series independent
    of how many browsers are polling and how fast.
    """

    def __init__(self, keep: int = 60, min_interval_s: float = 4.0) -> None:
        self._keep, self._min = keep, min_interval_s
        self._at: list[float] = []
        self._sm: list[float | None] = []
        self._gbps: list[float | None] = []
        self._tflops: list[float | None] = []
        self._net: list[float | None] = []
        self._boxes: dict[str, list[float | None]] = {}

    def observe(self, now: float, sm: float | None, gbps: float | None,
                tflops: float | None, boxes: dict[str, float | None] | None = None,
                net: float | None = None) -> dict:
        """`boxes` is each box's SM fill at this read, None for a box that was not read — a gap in its
        row, never a zero. A box seen for the first time has no past: its earlier samples are gaps."""
        if not self._at or now - self._at[-1] >= self._min:
            seen = boxes or {}
            for box in seen:
                self._boxes.setdefault(box, [None] * len(self._at))
            self._at.append(now); self._sm.append(sm)
            self._gbps.append(gbps); self._tflops.append(tflops); self._net.append(_num(net))
            for box, series in self._boxes.items():
                series.append(_num(seen.get(box)))
            for series in (self._at, self._sm, self._gbps, self._tflops, self._net,
                           *self._boxes.values()):
                del series[:max(0, len(series) - self._keep)]
            # A box no longer passed at all, whose kept samples have all aged into gaps, is gone — it
            # must not keep a row of gaps forever. A configured box that was not read IS passed.
            for box in [b for b, series in self._boxes.items()
                        if b not in seen and all(v is None for v in series)]:
                del self._boxes[box]
        return self.view()

    def view(self) -> dict:
        return {"at": list(self._at), "sm_fill_pct": list(self._sm),
                "gbps": list(self._gbps), "tflops": list(self._tflops), "net_gbps": list(self._net),
                "boxes": {box: list(series) for box, series in self._boxes.items()}}


def _ranked(gpus: list[dict], ranks: list[dict] | None) -> list[dict]:
    """One entry per rank, rank 0 first: the box's own GPU reading plus its rank and role — or, for a
    rank whose box sent none, a null reading and why. A reporting box outside the rank list is still
    shown, after the ranks. With no rank list, the boxes as they came."""
    if not ranks:
        return list(gpus)
    by_box = {g.get("box"): g for g in gpus}
    out = []
    for r in ranks:
        g = by_box.pop(r.get("box"), None)
        reading = dict(g) if g is not None else {"sm_pct": None, "why": NO_GPU_BOX}
        out.append({**reading, "box": r.get("box"), "rank": r.get("rank"), "role": r.get("role")})
    out.extend({**g, "rank": None, "role": None} for g in by_box.values())
    return out


def _rank_gap(boxes: list[dict]) -> float | None:
    """Rank 0's SM fill minus rank 1's, in points — None unless both were measured. Two ranks of one
    model step in lockstep, so a gap is time one of them spent with NO kernel resident: host-side work
    or waiting outside a collective. (A rank spinning in an NCCL wait still counts as filled.)"""
    fill = {b.get("rank"): _num(b.get("sm_pct")) for b in boxes}
    if fill.get(0) is None or fill.get(1) is None:
        return None
    return _round(fill[0] - fill[1])


def judge(*, gpus: list[dict], tflops_per_gpu: float | None, gbps_per_gpu: float | None,
          peak_tflops: float | None, peak_gbps: float | None,
          flops_why: str | None = None, bytes_why: str | None = None,
          window_s: float | None = None, gpu_count: int = 1,
          peak_tflops_precision: str = "", ranks: list[dict] | None = None,
          rates_from: str | None = None, window_widened: str | None = None,
          engine: str = "vllm", bytes_model: dict | None = None, flops_model: dict | None = None) -> dict:
    """The saturation block of /lb/status.

    `gpus` are `gpu_view` objects, one per box that reported one, each with its `box`. `ranks` is the
    pair in rank order ({box, rank, role}, head first); the block's `boxes` follows it. `tflops_per_gpu`
    / `gbps_per_gpu` are the engine's own rates PER GPU, read off `rates_from` (the head), with `*_why`
    giving the reason when there is no rate and `window_widened` when they were taken over a wider
    window than two probes. The cluster totals are per-GPU × `gpu_count`, which is what the operator
    sees as "the cluster's".

    `bytes_model`: the LB's modeled bandwidth (`throughput.Throughput.bandwidth_model`) when `gbps_per_gpu` is
    that model's figure — an SGLang generation — else None. A modeled GB/s may judge BANDWIDTH_BOUND at the
    same 70 %: vLLM's figure is no less a model (analytical bytes per step; a GB10 has no DRAM counter), and
    this one's constants are checked against the engine's own allocations. Its `floor`, when set, makes the
    bandwidth a lower bound: it can prove the ceiling reached, never headroom.

    `flops_model`: the LB's modeled FLOPs (cores.py `modeled_flops`) when `tflops_per_gpu` is that model's figure
    — an SGLang generation — else None. It judges COMPUTE_BOUND at the same 70 %.
    """
    fill = _fill(gpus)
    compute = Ceiling(tflops_per_gpu, peak_tflops, _share(tflops_per_gpu, peak_tflops), flops_why)
    busy = fill is not None and fill >= BUSY_SM_PCT
    # Rates over a WIDENED window are FLOORS: the step still in flight has done work the counter does
    # not hold yet. A low floor is expected, so it proves no component missing from the estimate; and
    # a floor under a ceiling does not prove the ceiling far away. A floor above one does prove it.
    floor = window_widened is not None
    bw_pct = _share(gbps_per_gpu, peak_gbps)
    # The dropped-component signature is BOTH rates near zero while the SMs are busy: before
    # CLUSTER_MFU_QUANT_FIX, 96 % SM fill read 2-4 GB/s AND 0.0-0.1 TFLOP/s. A low byte rate ALONE is
    # ordinary prefill, which is compute-heavy and light on bytes - measured 2026-09-30 at ~25-40 GB/s
    # (9-15 % of 273) beside ~50 TFLOP/s per GPU - so on its own it must not be called incomplete.
    # vLLM's dropped-component signature; a modeled figure prices every component, so it cannot have it.
    incomplete = (busy and not floor and bytes_model is None and bw_pct is not None
                  and bw_pct < IMPLAUSIBLE_BANDWIDTH_PCT
                  and tflops_per_gpu is not None and tflops_per_gpu < IMPLAUSIBLE_TFLOPS)
    if incomplete:
        bandwidth = Ceiling(gbps_per_gpu, peak_gbps, None, BYTES_INCOMPLETE)
        compute = Ceiling(tflops_per_gpu, peak_tflops, None, FLOPS_INCOMPLETE)
    else:
        bandwidth = Ceiling(gbps_per_gpu, peak_gbps, bw_pct, bytes_why)
    verdict, because, headroom = _verdict(fill, compute, bandwidth, floor=floor,
                                          bandwidth_floor=(bytes_model or {}).get("floor"),
                                          modeled=bytes_model is not None,
                                          flops_modeled=flops_model is not None)
    ranked = _ranked(gpus, ranks)
    return {
        "verdict": verdict,
        # WHY, in the LB's own words. The page shows this line; it is the whole point of the panel.
        "because": because,
        # Is there room for more work? True / False / None = we cannot tell, which is NOT "no".
        "headroom": headroom,
        "sm_fill_pct": _round(fill),
        # Named so nothing can read it as compute utilisation (module docstring, rule 1).
        "sm_fill_means": "share of time the SMs had work resident — not a share of peak FLOP/s",
        # A GPU has a different ceiling per precision, and this model mixes several, so the
        # denominator is named: 14 TFLOP/s is 15 % of bf16's ceiling but 4 % of nvfp4's.
        "compute": {**compute.view(), "peak_precision": peak_tflops_precision or None},
        "bandwidth": bandwidth.view(),
        # THE LB'S MODEL behind `bandwidth`, its parts as figures — or null when the bandwidth is the engine's
        # own estimate. The page labels the figure "modeled" and puts the parts in its hover.
        "bandwidth_model": _model_view(bytes_model),
        # …and behind `compute`: the modeled FLOPs, their prefill and decode parts, and the engine's own figure
        # (`engine_tflops`) — or null when the TFLOP/s is the engine's own estimate.
        "compute_model": _model_view(flops_model),
        "per_gpu": True,
        "gpu_count": gpu_count,
        # The same two rates for the whole pair, since the engine counts PER GPU and the operator
        # asked about the cluster. Null stays null: a total of an unmeasured rate is not a number.
        "cluster_tflops": _round(None if compute.rate is None else compute.rate * gpu_count),
        "cluster_gbps": _round(None if bandwidth.rate is None else bandwidth.rate * gpu_count),
        "window_s": _round(window_s),
        # Why the window is wider than two probes, when it is: no engine step completed in the last
        # one, so the rates are over the last completed step instead (throughput.engine_perf).
        "window_widened": window_widened,
        # What the two rates ARE, and whose /metrics they were read off: one analytical figure per GPU,
        # the same for both ranks — so the page never draws them as a per-box reading.
        "rates_from": rates_from,
        "rates_means": rates_mean(engine, modeled=bytes_model is not None, flops_modeled=flops_model is not None),
        # Every rank in rank order, each with what ITS box measured — SM fill, power, clock,
        # temperature — and nothing it did not: no FLOP/s or GB/s exists per box.
        "boxes": ranked,
        # Rank 0's SM fill minus rank 1's: an imbalance between two ranks that step in lockstep.
        "sm_fill_gap_pts": _rank_gap(ranked),
    }


def _model_view(model: dict | None) -> dict | None:
    """The modeled bandwidth as the page reads it: GB/s and pass rates to two places, the rest as given."""
    if model is None:
        return None
    return {key: (round(value, 2) if isinstance(value, float) else value) for key, value in model.items()}


def _verdict(fill: float | None, compute: Ceiling, bandwidth: Ceiling,
             floor: bool = False, bandwidth_floor: str | None = None,
             modeled: bool = False, flops_modeled: bool = False) -> tuple[str, str, bool | None]:
    """(verdict, the sentence behind it, is there headroom). Ordered so that nothing is claimed from
    a figure that was not measured. `floor`: the rates are lower bounds (a widened window), so they
    can prove a ceiling reached but never one NOT reached. `bandwidth_floor`: why the bandwidth ALONE
    is one (a modeled window that held a prefill pass); `modeled`: the GB/s is the LB's model;
    `flops_modeled`: the TFLOP/s is."""
    least = "at least " if floor else ""
    least_bw = "at least " if floor or bandwidth_floor else ""
    bw = "modeled " if modeled else ""
    fl = "modeled " if flops_modeled else ""
    if fill is None:
        return UNKNOWN, NO_GPU, None
    if fill < BUSY_SM_PCT:
        return (IDLE, f"the SMs had work {fill:.0f} % of the time: the pair is not the constraint "
                      f"right now", True)
    if bandwidth.pct is not None and bandwidth.pct >= BOUND_BANDWIDTH_PCT:
        # The operator's actual question is "can anything still be optimised". Saying only "bandwidth
        # bound" leaves them to guess what compute is doing, so the FLOP/s goes in the same sentence —
        # as a share of peak where one is configured, and as the bare measured rate where none is.
        if compute.pct is not None and floor:
            room = f", with {fl}compute at {compute.pct:.0f} % of peak or more"
        elif compute.pct is not None:
            room = f", while {fl}compute runs at {compute.pct:.0f} % of peak — compute headroom exists"
        elif compute.rate is not None:
            room = (f", while the engine computes {compute.rate:.1f} TFLOP/s per GPU — no compute peak "
                    f"is configured, so how much of it is left is not claimed here")
        else:
            room = ""
        return (BANDWIDTH_BOUND,
                f"{least_bw}{bw}{bandwidth.rate:.0f} GB/s per GPU is {bandwidth.pct:.0f} % of the "
                f"{bandwidth.peak:.0f} GB/s this box can move{room}", False)
    if compute.pct is not None and compute.pct >= BOUND_COMPUTE_PCT:
        return (COMPUTE_BOUND,
                f"{least}{fl}{compute.rate:.0f} TFLOP/s per GPU is {compute.pct:.0f} % of peak", False)
    if compute.rate is not None or bandwidth.rate is not None:
        parts = []
        if bandwidth.pct is not None:
            parts.append(f"{bw}bandwidth {'at least' if bandwidth_floor else 'at'} {bandwidth.pct:.0f} % of peak")
        elif bandwidth.rate is not None:
            # `why` beats the generic line: for bandwidth a peak IS configured, so "no peak
            # configured" would be a plain untruth about why the share is missing.
            parts.append(f"{bandwidth.rate:.0f} GB/s per GPU — {bandwidth.why}" if bandwidth.why
                         else f"{bandwidth.rate:.0f} GB/s per GPU, with no peak configured to judge it")
        if compute.pct is not None:
            parts.append(f"{fl}compute at {compute.pct:.0f} % of peak")
        elif compute.rate is not None:
            parts.append(f"{compute.rate:.1f} TFLOP/s per GPU — {compute.why}" if compute.why
                         else f"{compute.rate:.1f} TFLOP/s per GPU, with no peak configured to judge it")
        if floor:
            return (BUSY_UNMEASURED,
                    f"the SMs had work {fill:.0f} % of the time, {' and '.join(parts)} — floors: a step "
                    f"is still running and its work is not counted yet, so no headroom is claimed "
                    f"from them", None)
        if bandwidth_floor and bandwidth.rate is not None:
            return (BUSY_UNMEASURED,
                    f"the SMs had work {fill:.0f} % of the time, {' and '.join(parts)} — {bandwidth_floor}, "
                    f"so no headroom is claimed from it", None)
        return (BUSY_UNMEASURED,
                f"the SMs had work {fill:.0f} % of the time, {' and '.join(parts)}: neither "
                f"ceiling is reached, so more concurrency should still buy throughput", True)
    # Busy, and neither rate is a number. The switch is named only when the engine's own counters
    # PROVE it off; any other gap — a step still running, an LB that just started — says its own why.
    whys = [c.why for c in (compute, bandwidth) if c.why]
    if MFU_NOT_MEASURED in whys:
        return BUSY_UNMEASURED, f"the SMs had work {fill:.0f} % of the time, but {MFU_OFF}", None
    return (BUSY_UNMEASURED,
            f"the SMs had work {fill:.0f} % of the time, but neither FLOP/s nor GB/s is measured right "
            f"now — {whys[0] if whys else 'the engine reported neither'}", None)
