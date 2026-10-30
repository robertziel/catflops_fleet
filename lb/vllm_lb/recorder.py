"""The per-box FLIGHT RECORDER: one hour of every hardware figure a box broadcasts, two-tier, in memory.

WHY. broker_parrot's `hw_watch` is the persisted flight recorder that earned the name: a unified-memory box's firmware
cut its power on every heavy render with no kernel trace, and only a sample-every-2-s recorder caught the GPU asserting
a hardware-slowdown throttle mask seconds before. That recorder writes a database on the box's own engine. The gateway
already hears every box on the LAN (hw.py, UDP); this module keeps the same figures for the last hour so the panel can
show what a box was doing — its temperatures, power, clocks, throttle, network and containers — without a credential,
a database or a login on the box.

WHAT IT READS (and the contract the sender must meet). The figures are fleet_telemetry's (lb/fleet_telemetry, the
same figures hw_watch recorded), under the datagram's `deep` key, and the box's containers under `containers` — `[{name, image, ram_mb}]`:

    deep.gpu[]          temp_c · power_w (the driver's averaged draw) · power_inst_w · sm_mhz · throttle_index
    deep.tz[]           every /sys/class/thermal zone, {type, milli_c} — the BOARD MAX is the hottest
    deep.cpu            temp_c · freq_mhz
    deep.fan_rpm_max    the loudest fan tach (None: the box has none)
    deep.net_rx_bytes / net_tx_bytes   summed MONOTONIC counters; throughput is their delta (hw_watch `net_rates`)
    deep.why            {figure key: reason} for what THIS box cannot read — shown in place of the generic reason
                        (the sender knows "amdgpu has no throttle mask"; the LB only knows the figure is missing)
    containers_total    {count, ram_mb} over EVERY container, when the list is the sender's top N by RAM (it caps the
                        list to keep the datagram small); without it the list is the whole set
    containers_why      the sender could not read its containers this sample, and why

CPU UTIL is the broadcast's own `cpu_percent` (every sender sends it). A sender that predates fleet_telemetry (lb/fleet_telemetry)
broadcasts neither `deep` nor `containers`; fleet_telemetry sends both, and its figures.py is the one definition of
the wire's names and the figure table this module reads. Against a sender without them every figure but CPU UTIL is ABSENT and says so, and nothing breaks. No figure is ever made up here — a None is a None with the reason beside it.

A BOX HEADLINE FROM SEVERAL CARDS is the constraint, the hw.py rule (`_gpu_percent`: the busiest card): the HOTTEST
card's temperature, the SLOWEST card's clock, the WORST throttle, and the box's draw as the SUM of its cards — withheld
whole when one card does not report, because a partial sum is a number the box never drew.

TWO TIERS, BOUNDED (hw_watch's detail + history, scaled to one hour):
  * fine   — every sample of the last FINE_S, as sent (the "death-minutes" ring);
  * coarse — one bucket per BUCKET_S over WINDOW_S: per figure the sum, count and max, so the hour's mean and its
             peaks survive at a fixed cost whatever the sender's rate.
Every sample folds into the coarse tier too, so the hour's peak and mean read from the coarse tier alone.

IN MEMORY, NOT PERSISTED. hw_watch persists because its job is the autopsy of a box that died — the box's own trail
must survive the box. Here the recorder lives in the gateway, which outlives a box by design; what a gateway
restart costs is the hour, and the boxes refill it in an hour. Writing a state file every few seconds for that is
not cheap enough to earn it (`lb/state/` is for the cluster's intent). The page says how much of the hour it holds.

Thread-safety: one lock; `ingest` is called from the listener thread, `summary`/`series` from request threads.
"""
from __future__ import annotations

import math
import threading
from collections import deque
from typing import Any

from fleet_telemetry import figures as F
from fleet_telemetry.model import THROTTLE_BITS, throttle_reasons

#: The recorder's window: the panel's "FLIGHT RECORDER · 1H".
WINDOW_S = 3600
#: The coarse tier's bucket: one point a minute over the hour.
BUCKET_S = 60
#: The fine tier: every sample of the last ten minutes.
FINE_S = 600
#: A sender faster than one sample a second does not grow the fine tier past this.
FINE_MAX = FINE_S + 1
#: A network delta across a silence longer than this is a gap: a mean over a box gone quiet is not a rate it ran at.
NET_MAX_GAP_S = 60.0
#: How many containers /lb/status carries per box (largest RAM first); the rest are "+k more" and the endpoint's.
TOP_CONTAINERS = 6
#: A list that runs on is not a box's containers.
MAX_CONTAINERS = 256
MAX_TEXT = 96
MAX_WHY = 240

#: (key, label, unit) in the panel's order, and every key the recorder keeps — the SENDER's table (fleet_telemetry
#: writes each absent figure's reason under these keys): one definition, imported, never a second copy here.
FIGURES = F.FIGURES_TABLE
KEYS = F.KEYS

NO_DEEP = ("this box's sender broadcasts no recorder block (`deep`) — it predates it; "
           "a fleet_telemetry sender (lb/hw_sender) fills this")
NO_CONTAINERS = "this box's sender broadcasts no `containers` list — it predates it"
NO_INPUT_RAIL = ("no sender reports an input-rail (PSU side) reading: a GPU's driver reports the card's draw, "
                 "not the box's input")

_WHY = {
    "gpu_temp_c": "no GPU temperature in the sample (no vendor SMI on the box, or it answered N/A)",
    "board_temp_c": "the box exposes no /sys/class/thermal zone",
    "cpu_temp_c": "the box exposes no CPU temperature sensor",
    "fan_rpm": "no fan tachometer on this box",
    "gpu_power_w": "a GPU reported no averaged power draw",
    "gpu_power_inst_w": "a GPU reported no instantaneous power draw",
    "throttle": "the GPU reports no throttle-reason mask",
    "gpu_mhz": "no GPU clock in the sample",
    "cpu_mhz": "no CPU clock (cpufreq / cpuinfo) in the sample",
    "cpu_pct": "the sample carries no cpu_percent",
}


def _finite(v: Any) -> float | int | None:
    """A finite number, or None. A boolean is not a reading (True would draw as 1)."""
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        return None
    return v


def _text(v: Any) -> str | None:
    return v.strip()[:MAX_TEXT] if isinstance(v, str) and v.strip() else None


def _text_long(v: str) -> str:
    """A sender's reason: text, bounded (it lands in a tooltip, not a layout)."""
    return v.strip()[:MAX_WHY]


def _cards(deep: dict, key: str) -> list:
    gpus = deep.get(F.DEEP_GPU) if isinstance(deep.get(F.DEEP_GPU), list) else []
    return [_finite(g.get(key)) if isinstance(g, dict) else None for g in gpus]


def _over_cards(deep: dict, key: str, how) -> float | int | None:
    """The box headline over its cards (`how`: max / min / sum); None when there is no card or one lacks the
    reading — a headline over the cards that DID report would name a box that is not this one."""
    vals = _cards(deep, key)
    if not vals or any(v is None for v in vals):
        return None
    return how(vals)


def figures_of(data: dict, prev_net: tuple[float, float, float] | None, now: float | None
               ) -> tuple[dict[str, Any], dict[str, str]]:
    """One sample's figures and, for each absent one, why. `prev_net` is the previous sample's `(t, rx, tx)` (None:
    no previous counter) and `now` this sample's time — the network rates are their delta."""
    figs: dict[str, Any] = {k: None for k in KEYS}
    why: dict[str, str] = {}
    figs["cpu_pct"] = _finite(data.get(F.CPU_PERCENT))
    deep = data.get(F.DEEP)
    if not isinstance(deep, dict):
        for k in KEYS:
            if k != "cpu_pct":
                why[k] = NO_DEEP
        if figs["cpu_pct"] is None:
            why["cpu_pct"] = _WHY["cpu_pct"]
        return figs, why
    told = deep.get(F.DEEP_WHY) if isinstance(deep.get(F.DEEP_WHY), dict) else {}
    # The SENDER's reason for a figure it could not read (it knows which sensor is missing; the LB knows only that the
    # figure is). A counter that is there but has no predecessor keeps the LB's own reason (first sample, a gap).
    said = lambda k: _text_long(told[k]) if _text(told.get(k)) else None   # noqa: E731
    figs["gpu_temp_c"] = _over_cards(deep, F.CARD_TEMP_C, max)
    figs["gpu_power_w"] = _over_cards(deep, F.CARD_POWER_W, sum)
    figs[F.GPU_POWER_INST_W] = _over_cards(deep, F.CARD_POWER_INST_W, sum)
    figs["gpu_mhz"] = _over_cards(deep, F.CARD_SM_MHZ, min)
    figs["throttle"] = _finite(deep.get(F.DEEP_THROTTLE_INDEX))
    zones = [_finite(z.get(F.ZONE_MILLI_C)) for z in (deep.get(F.DEEP_ZONES) or []) if isinstance(z, dict)]
    zones = [z for z in zones if z is not None]
    figs["board_temp_c"] = round(max(zones) / 1000.0, 1) if zones else None
    cpu = deep.get(F.DEEP_CPU) if isinstance(deep.get(F.DEEP_CPU), dict) else {}
    figs["cpu_temp_c"] = _finite(cpu.get(F.CPU_TEMP_C))
    figs["cpu_mhz"] = _finite(cpu.get(F.CPU_FREQ_MHZ))
    figs["fan_rpm"] = _finite(deep.get(F.DEEP_FAN_RPM_MAX))
    rx, tx = _finite(deep.get(F.NET_RX_BYTES)), _finite(deep.get(F.NET_TX_BYTES))
    for key, cur, idx in (("net_in_mbps", rx, 1), ("net_out_mbps", tx, 2)):
        if cur is None:
            why[key] = said(key) or "the sample carries no network counter"
        elif prev_net is None or prev_net[idx] is None or now is None:
            why[key] = "first sample: a rate needs two counters"
        elif now - prev_net[0] <= 0 or now - prev_net[0] > NET_MAX_GAP_S:
            why[key] = f"the previous counter is {now - prev_net[0]:.0f}s old: a gap, not a rate"
        elif cur < prev_net[idx]:
            why[key] = "the counter went backwards (reboot / NIC reset): a gap, not a spike"
        else:
            figs[key] = (cur - prev_net[idx]) * 8 / 1e6 / (now - prev_net[0])
    for k in KEYS:
        if figs[k] is None and k not in why:
            why[k] = said(k) or _WHY[k]
    return figs, why


def containers_of(raw: Any) -> list[dict] | None:
    """The sample's containers, largest RAM first; an entry with no name or a RAM that is not a finite, non-negative
    number is withheld (one bad entry does not cost the list); None when the sample carries no list."""
    if not isinstance(raw, list):
        return None
    out: list[dict] = []
    for c in raw[:MAX_CONTAINERS]:
        if not isinstance(c, dict) or _text(c.get(F.C_NAME)) is None:
            continue
        ram = c.get(F.C_RAM_MB)
        if ram is not None and (_finite(ram) is None or ram < 0):
            continue
        out.append({"name": _text(c.get(F.C_NAME)), "image": _text(c.get(F.C_IMAGE)), "ram_mb": ram})
    out.sort(key=lambda c: -(c["ram_mb"] if c["ram_mb"] is not None else -1))
    return out


def totals_of(raw: Any) -> dict | None:
    """The sender's count and RAM over every container (`containers_total`), or None when it sent none (the list
    is then the whole set). A malformed total is None too — the list's own count is then what is known."""
    if not isinstance(raw, dict):
        return None
    count, ram = raw.get(F.TOTAL_COUNT), raw.get(F.TOTAL_RAM_MB)
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        return None
    if ram is not None and (_finite(ram) is None or ram < 0):
        return None
    return {"count": count, "ram_mb": ram}


class _Bucket:
    __slots__ = ("sum", "n", "max")

    def __init__(self) -> None:
        self.sum: dict[str, float] = {}
        self.n: dict[str, int] = {}
        self.max: dict[str, float] = {}

    def add(self, figs: dict) -> None:
        for k, v in figs.items():
            if v is None:
                continue
            self.sum[k] = self.sum.get(k, 0.0) + v
            self.n[k] = self.n.get(k, 0) + 1
            self.max[k] = v if k not in self.max else max(self.max[k], v)

    def mean(self, k: str) -> float | None:
        return self.sum[k] / self.n[k] if self.n.get(k) else None


class _Box:
    def __init__(self) -> None:
        self.fine: deque = deque(maxlen=FINE_MAX)          # (t, figs)
        self.coarse: dict[int, _Bucket] = {}               # minute index → bucket
        self.latest: dict[str, Any] = {}
        self.why: dict[str, str] = {}
        self.net: tuple[float, float, float] | None = None
        self.containers: list[dict] | None = None
        self.containers_total: dict | None = None
        self.containers_why: str | None = None
        self.first_t: float | None = None
        self.throttle_reasons: list[str] | None = None     # the latest sample's, decoded (model.throttle_reasons)
        self.power_scope: str | None = None                # F.POWER_SCOPE of the cards' power label ("package")


def _throttle_reasons(deep: dict) -> list[str] | None:
    """The active throttle reasons over the box's cards (their union, in NVML bit order), [] for a zero mask, None when
    no card reports a mask (amdgpu has none) — the panel says "none" for [] and "—" with the reason for None."""
    gpus = deep.get(F.DEEP_GPU) if isinstance(deep.get(F.DEEP_GPU), list) else []
    per = [throttle_reasons(g.get(F.CARD_THROTTLE_HEX)) for g in gpus if isinstance(g, dict)]
    per = [r for r in per if r is not None]
    if not per:
        return None
    active = {n for r in per for n in r}
    return [n for _bit, n, _w in THROTTLE_BITS if n in active]


def _power_scope(deep: dict) -> str | None:
    """What the GPU power reading covers when every card's sensor names it alike ("package": an APU's PPT); None when
    the driver says nothing more than "the card"."""
    gpus = [g for g in (deep.get(F.DEEP_GPU) if isinstance(deep.get(F.DEEP_GPU), list) else []) if isinstance(g, dict)]
    scopes = {F.POWER_SCOPE.get(_text(g.get(F.CARD_POWER_LABEL)) or "") for g in gpus}
    return scopes.pop() if len(scopes) == 1 else None


class FlightRecorder:
    """Per box, the last hour of its figures. Thread-safe."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._boxes: dict[str, _Box] = {}

    def ingest(self, box: str, data: dict, now: float) -> None:
        with self._lock:
            b = self._boxes.setdefault(box, _Box())
            figs, why = figures_of(data, b.net, now)
            deep = data.get(F.DEEP) if isinstance(data.get(F.DEEP), dict) else {}
            rx, tx = _finite(deep.get(F.NET_RX_BYTES)), _finite(deep.get(F.NET_TX_BYTES))
            b.net = (now, rx, tx) if rx is not None or tx is not None else None
            b.latest, b.why = figs, why
            b.throttle_reasons, b.power_scope = _throttle_reasons(deep), _power_scope(deep)
            b.containers = containers_of(data.get(F.CONTAINERS))
            b.containers_total = totals_of(data.get(F.CONTAINERS_TOTAL)) if b.containers is not None else None
            b.containers_why = _text_long(data[F.CONTAINERS_WHY]) if _text(data.get(F.CONTAINERS_WHY)) else None
            b.fine.append((now, figs))
            while b.fine and b.fine[0][0] < now - FINE_S:
                b.fine.popleft()
            idx = int(now // BUCKET_S)
            b.coarse.setdefault(idx, _Bucket()).add(figs)
            for old in [i for i in b.coarse if i <= idx - WINDOW_S // BUCKET_S]:
                del b.coarse[old]
            if b.first_t is None:
                b.first_t = now

    # ── read ────────────────────────────────────────────────────────────────

    def _held(self, b: _Box, now: float) -> list[_Bucket]:
        lo = int(now // BUCKET_S) - WINDOW_S // BUCKET_S
        return [bk for i, bk in b.coarse.items() if i > lo]

    def _power(self, b: _Box, now: float) -> dict:
        held = self._held(b, now)
        peak = lambda k: max((bk.max[k] for bk in held if k in bk.max), default=None)   # noqa: E731
        s = sum(bk.sum.get("gpu_power_w", 0.0) for bk in held)
        n = sum(bk.n.get("gpu_power_w", 0) for bk in held)
        # ONE READING. amdgpu answers power1_average and power1_input from the same sensor on an APU: two "peaks" of
        # one figure read as two measurements. When every held minute's instantaneous figures equal its averaged ones
        # (or there are none), the panel shows one peak and the hour's mean instead.
        inst = F.GPU_POWER_INST_W
        same = lambda x, y: abs(x - y) < 1e-9   # noqa: E731
        one = all(not bk.n.get(inst) or (
            bk.n.get(inst) == bk.n.get("gpu_power_w")
            and same(bk.sum[inst], bk.sum.get("gpu_power_w", 0.0))
            and same(bk.max[inst], bk.max.get("gpu_power_w", 0.0))) for bk in held)
        out = {"peak_inst_w": peak(inst), "peak_avg_w": peak("gpu_power_w"),
               "mean_w": s / n if n else None, "input_rail_w": None, "input_rail_why": NO_INPUT_RAIL,
               "one_reading": bool(one and n), "scope": b.power_scope}
        if out["peak_avg_w"] is None or (out["peak_inst_w"] is None and not out["one_reading"]):
            out["why"] = b.why.get("gpu_power_w") or b.why.get("gpu_power_inst_w") or "no GPU power in the hour"
        return out

    @staticmethod
    def _containers(b: _Box, stale: bool, age_s, everything: bool) -> tuple[dict | None, str | None]:
        if stale:
            return None, f"no sample for {age_s:.0f}s: the box's containers are not known now"
        if b.containers is None:
            return None, b.containers_why or NO_CONTAINERS
        rams = [c["ram_mb"] for c in b.containers]
        listed = len(b.containers)
        total = b.containers_total
        count = max(total["count"], listed) if total else listed
        out = {"count": count,
               "total_ram_mb": total["ram_mb"] if total else (sum(rams) if all(r is not None for r in rams) else None),
               "top": b.containers[:TOP_CONTAINERS], "more": max(0, listed - TOP_CONTAINERS),
               # How many the datagram did not carry (the sender's list is its top N by RAM): the head counts them,
               # "+k more" can only reveal what was sent.
               "unlisted": count - listed}
        if out["total_ram_mb"] is None:
            out["total_why"] = "a container reported no RAM: a total of the others would understate the box"
        if everything:
            out["all"] = list(b.containers)
        return out, None

    def summary(self, box: str, now: float, *, stale: bool, age_s) -> dict | None:
        """What /lb/status carries per box: the latest figures (withdrawn when stale, with why), the hour's power
        peak and mean, how much of the hour is held, and the containers' head. None for a box never heard."""
        with self._lock:
            b = self._boxes.get(box)
            if b is None:
                return None
            figs = {k: b.latest.get(k) for k, *_ in FIGURES}
            why = {k: b.why[k] for k in figs if k in b.why}
            if stale:
                figs = {k: None for k in figs}
                why = {k: f"no sample for {age_s:.0f}s: not a reading of now" for k in figs}
            containers, cwhy = self._containers(b, stale, age_s, False)
            out = {"window_s": WINDOW_S, "held_s": round(min(WINDOW_S, now - b.first_t), 1),
                   # The figures' labels and units, in the panel's order: one list (FIGURES), never a second in JS.
                   "labels": [{"key": k, "label": lab, "unit": u, "fmt": fmt} for k, lab, u, fmt in FIGURES],
                   "figures": figs, "why": why, "power": self._power(b, now), "containers": containers,
                   # The throttle mask's reasons by name (the THROTTLE figure's text); withdrawn with the figures.
                   "throttle_reasons": None if stale else b.throttle_reasons}
            if cwhy:
                out["containers_why"] = cwhy
            return out

    def series(self, box: str, now: float, *, stale: bool, age_s) -> dict | None:
        """The whole hour for the charts: the coarse tier as a grid of WINDOW_S / BUCKET_S minutes ending at `now`'s
        minute (mean and max per figure, None for a minute with no reading — a gap), the fine tier as the samples of
        the last FINE_S, and every container. None for a box never heard."""
        head = self.summary(box, now, stale=stale, age_s=age_s)
        if head is None:
            return None
        with self._lock:
            b = self._boxes[box]
            end = int(now // BUCKET_S)
            idxs = list(range(end - WINDOW_S // BUCKET_S + 1, end + 1))
            coarse = {"t": [i * BUCKET_S for i in idxs], "series": {}}
            for k in KEYS:
                bks = [b.coarse.get(i) for i in idxs]
                coarse["series"][k] = {"mean": [bk.mean(k) if bk else None for bk in bks],
                                       "max": [bk.max.get(k) if bk else None for bk in bks]}
            fine_pts = [(t, f) for t, f in b.fine if t >= now - FINE_S]
            fine = {"t": [t for t, _ in fine_pts], "series": {k: [f.get(k) for _, f in fine_pts] for k in KEYS}}
            containers, cwhy = self._containers(b, stale, age_s, True)
        return {**head, "now": now, "bucket_s": BUCKET_S, "fine_s": FINE_S,
                "coarse": coarse, "fine": fine, "containers": containers,
                **({"containers_why": cwhy} if cwhy else {})}
