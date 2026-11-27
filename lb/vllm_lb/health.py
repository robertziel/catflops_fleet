"""HARDWARE HEALTH FINDINGS — small, explainable rules over figures the LB already receives.

WHY. The panel draws every box's temperature, memory, power and error counters, and nobody watches a panel at 03:00.
A finding is the LB saying, in one line, that a figure has been where it should not be for long enough to matter —
and the alert sender (alerts.py) says it to the operator's phone. Not done here: a
silent 85/90 °C when a setting is unusable, fixed watts and MB. Here every threshold is either a
figure the hardware's own driver states (the GPU's temperature headroom), or OPERATOR POLICY the fleet file states —
and a finding without one says so.

THE RULES (`KINDS`), each per box, judged on its latest sample (hw.py `snapshot`, the /lb/status `hardware.<box>` row):
  gpu_temp     the tightest card's TEMPERATURE HEADROOM — °C below the GPU's own throttle limit, as its DRIVER states
               it (hw.py `temp_headroom_c`: nvidia-smi `temperature.gpu.tlimit`, amdgpu `temp*_crit − temp*_input`) —
               ≤ the fleet's `temp_margin_c`. One definition of the limit, the driver's, read live; a card whose
               driver states none (an APU hwmon with no `_crit`, a T.Limit of N/A) is no data with that reason,
               never a number of the LB's own.
  low_ram      the box's RAM available (total − used: `free`'s available) < `low_ram_fraction` of the box's OWN RAM
               total — one policy value that scales from a 128 GB unified box to a small discrete host. On unified
               memory this is also the GPU's memory; the rule does not need to know which family a box is.
  low_power    the busiest card ≥ `busy_gpu_pct` while the box's GPUs draw ≤ their IDLE draw + `low_power_margin_w`:
               a GPU that says it is busy at idle draw is wedged (no fixed wattage: the threshold is per card type). The
               idle draw is each card TYPE's measured `idle_gpu_power_w` (hardware/types/*.yml, read through the
               hardware profile), summed over the box's cards as the box's power is; a type with none is no data
               naming its source — so a wedged 400 W card is never judged against a GB10's 11 W.
  xid / oom    the kernel's NVIDIA Xid count / OOM-kill count (the sender's `kernel` block) ROSE within
               `counter_window_s`. The first reading is a baseline; a counter that went down is a reboot (rebaselined).
  roce         an RDMA port's error counter (`ERROR_COUNTERS` in the sender's `rdma` block) rose within the window.
  backend_down a classic server that has answered since this LB started stops answering (not parked) — held for
               `backend_down_s`, not `hold_s`. Not judged: a server never live since the LB started (the fleet's
               resting state: a box with no engine up), a cluster head (its cluster's state says it — `failed` is
               its own alert), and any server while the broker is on (a lease decides when it runs).

THE STATE OF A FINDING. Its condition holds → PENDING (`since` = when it began); held for its hold time → FIRING, one
`finding.firing` event; stops holding → gone, one `finding.cleared` event if it had fired. A figure the box did not
send is NO DATA (listed with the sender's own why): a pending finding is dropped, a FIRING one stays firing and says
"no data" — the LB cannot claim a hot GPU cooled because it stopped hearing it.

NO POLICY, NO FINDINGS. The fleet's `health:` block is opt-in: without it the evaluator says health is off and why.
With it, every key is required and an unknown one refused (`parse_policy`) — no default threshold anywhere.

NEVER IN THE WAY. `HealthMonitor` runs the evaluator on its own thread every `tick_s` off the samples the listener
already holds; an exception in a tick is counted and shown, never raised into the control loop.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, fields
from typing import Any, Callable

from vllm_lb import events

log = logging.getLogger(__name__)

KINDS = ("gpu_temp", "low_ram", "low_power", "xid", "oom", "roce", "backend_down")
#: The RDMA port counters that count an error → the `why` GROUP key the sender files their absence under when a whole
#: source is missing (`hw_counters` / `counters`: the port's sysfs directory; `ethtool`: `ethtool -S <netdev>` refused or
#: no netdev in this namespace). Each one the sender reads (fleet_telemetry/readers/rdma.py COUNTERS, PHY_COUNTERS;
#: tests/test_health.py pins the names AND the groups against those): a rise of ANY single one, on any port, is a
#: finding. Not judged: packet_seq_err / implied_nak_seq_err / port_rcv_remote_physical_errors (shown on the panel, not
#: alerted), and the pause counters rx/tx_pause_ctrl_phy — flow control at work, not a fault.
ERROR_COUNTERS = {"out_of_buffer": "hw_counters", "local_ack_timeout_err": "hw_counters",
                  "rnr_nak_retry_err": "hw_counters", "port_rcv_errors": "counters", "port_xmit_discards": "counters",
                  "symbol_error": "counters", "link_downed": "counters", "rx_discards_phy": "ethtool",
                  "tx_discards_phy": "ethtool", "rx_crc_errors_phy": "ethtool", "rx_symbol_err_phy": "ethtool"}
#: Each kind's label and unit, as the panel prints a finding (`view`).
LABELS = {"gpu_temp": ("GPU TEMP", "°C"), "low_ram": ("RAM AVAIL", "GiB"), "low_power": ("GPU POWER", "W"),
          "xid": ("XID", "new"), "oom": ("OOM KILLS", "new"), "roce": ("RoCE ERR", "new"),
          "backend_down": ("SERVER", "")}
OFF = "health findings are off: the fleet file has no `health:` block (fleet.example.yml documents it)"


@dataclass(frozen=True)
class HealthPolicy:
    """The fleet file's `health:` block — the operator's policy; the hardware's limits are its driver's."""

    hold_s: float              # a condition must hold this long before it fires
    temp_margin_c: float       # fire when a GPU is this many °C (or fewer) below its own throttle limit
    low_ram_fraction: float    # fire when a box's RAM available drops under this share (0-1) of its own RAM total
    busy_gpu_pct: float        # a GPU at or over this utilisation is busy…
    low_power_margin_w: float  # …and busy at no more than its card type's idle draw + this many watts is wedged
    counter_window_s: float    # a rise of an error counter keeps its finding true this long
    backend_down_s: float      # a server that was live and stopped answering, for this long, fires


#: (minimum, maximum, minimum allowed?, maximum allowed?) per key.
_RANGES = {"hold_s": (0, None, True, True), "temp_margin_c": (0, None, True, True),
           "low_ram_fraction": (0, 1, False, False), "busy_gpu_pct": (0, 100, False, True),
           "low_power_margin_w": (0, None, True, True), "counter_window_s": (0, None, False, True),
           "backend_down_s": (0, None, True, True)}


def parse_policy(where: str, raw: Any) -> HealthPolicy:
    """The `health:` block, every key stated and in range — or ValueError naming `where` and the key."""
    names = [f.name for f in fields(HealthPolicy)]
    if not isinstance(raw, dict):
        raise ValueError(f"{where} must be a mapping of {', '.join(names)}, got {type(raw).__name__}")
    missing = [k for k in names if k not in raw]
    unknown = [str(k) for k in raw if k not in names]
    if missing or unknown:
        raise ValueError(f"{where} takes {', '.join(names)}"
                         + (f"; missing {', '.join(missing)}" if missing else "")
                         + (f"; unknown {', '.join(unknown)}" if unknown else ""))
    out = {}
    for k in names:
        v = raw[k]
        lo, hi, lo_ok, hi_ok = _RANGES[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) \
                or v < lo or (v == lo and not lo_ok) or (hi is not None and (v > hi or (v == hi and not hi_ok))):
            shape = (f"{'≥' if lo_ok else '>'} {lo}" + (f" and {'≤' if hi_ok else '<'} {hi}" if hi is not None else ""))
            raise ValueError(f"{where}.{k} must be a number {shape}, got {v!r}")
        out[k] = float(v)
    return HealthPolicy(**out)


def _num(v: Any) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None


class _Reading:
    """One rule's verdict on one box this tick: holds (True/False) with figure and threshold, or no data with why."""

    __slots__ = ("holds", "value", "threshold", "why", "rose")

    def __init__(self, holds: bool | None, value: Any = None, threshold: Any = None, why: str = "",
                 rose: list | None = None) -> None:
        self.holds, self.value, self.threshold, self.why = holds, value, threshold, why
        #: roce only: the counters that rose in the window, as data — [{dev, port, counter, rise}] — so the machine
        #: card flags exactly what the finding names (hwextra.js), never a second rise computation in the browser.
        self.rose = rose


def _no(why: str) -> _Reading:
    return _Reading(None, why=why)


class HealthEvaluator:
    """The rules and the findings' state. Not thread-safe by itself: `HealthMonitor` owns the lock."""

    def __init__(self, policy: HealthPolicy | None, fleet=None) -> None:
        self.policy = policy
        #: The fleet file (vllm_lb/fleet.py): each box's cards and, through their hardware profiles, each card type's
        #: idle draw (low_power). None: no box is named, and low_power is no data for every box, saying so.
        self.fleet = fleet
        #: id → the finding as the page reads it (`view`), pending or firing.
        self._findings: dict[str, dict] = {}
        self._no_data: list[dict] = []
        #: (box, kind) → the counter's last value; (box, kind) → [(t, increment)] inside the window.
        self._last: dict[tuple, float] = {}
        self._rises: dict[tuple, list] = {}
        #: backend url → it has answered since this LB started.
        self._ever_live: set[str] = set()
        self.evaluated_at: float | None = None

    # ── the rules ──────────────────────────────────────────────────────────

    def _gpu_temp(self, box: str, row: dict) -> _Reading:
        headroom = _num(row.get("temp_headroom_c"))
        if headroom is None:
            return _no(row.get("temp_headroom_why") or "no GPU temperature headroom in the sample")
        margin = self.policy.temp_margin_c
        return _Reading(headroom <= margin, headroom, margin,
                        f"GPU {headroom:g} °C below its own throttle limit ≤ margin {margin:g} °C")

    def _low_ram(self, box: str, row: dict) -> _Reading:
        total, used = _num(row.get("ram_total_mb")), _num(row.get("ram_used_mb"))
        if total is None or used is None or total <= 0:
            return _no("no RAM reading in the sample")
        avail = round((total - used) / 1024.0, 1)
        f = self.policy.low_ram_fraction
        floor = round(f * total / 1024.0, 1)
        return _Reading(avail < floor, avail, floor,
                        f"RAM available {avail:g} GiB < {floor:g} GiB ({f * 100:g} % of {total / 1024.0:.0f} GiB)")

    def _idle_draw(self, box: str) -> tuple[float | None, str]:
        """(the box's GPUs' idle draw in all, W — each card's TYPE fact `idle_gpu_power_w`, summed as the box's power
        figure is), or (None, why): a box no fleet names, or a card type that states none, naming its source."""
        named = self.fleet.boxes.get(box) if self.fleet is not None else None
        if named is None or not named.cards:
            return None, f"box {box!r} is not named in the fleet file with its cards, so no idle GPU draw is known"
        total = 0.0
        for c in named.cards:
            fact = c.type.idle_gpu_power_w
            if not fact.known:
                return None, f"no idle GPU draw for {c.type.name}: {fact.source}"
            total += float(fact.value)
        return round(total, 2), ""

    def _low_power(self, box: str, row: dict) -> _Reading:
        util = _num(row.get("gpu_pct"))
        rec = row.get("recorder") if isinstance(row.get("recorder"), dict) else {}
        watts = _num((rec.get("figures") or {}).get("gpu_power_w"))
        if util is None:
            return _no("no GPU utilisation in the sample")
        if watts is None:
            return _no((rec.get("why") or {}).get("gpu_power_w") or "no GPU power draw in the sample")
        idle, why = self._idle_draw(box)
        if idle is None:
            return _no(why)
        p = self.policy
        ceiling = round(idle + p.low_power_margin_w, 2)
        return _Reading(util >= p.busy_gpu_pct and watts <= ceiling, watts, ceiling,
                        f"GPU {util:g} % busy at {watts:g} W ≤ {ceiling:g} W (idle {idle:g} W + margin "
                        f"{p.low_power_margin_w:g} W)")

    def _rise(self, key: tuple, now: float, value: float) -> None:
        """Record one cumulative counter's reading: a rise over its last reading is kept with its time; the first
        reading is a baseline, a fall a reset (rebaselined). A counter not read this tick keeps its last value, so one
        that goes unreadable and comes back unchanged has not risen."""
        last = self._last.get(key)
        if last is not None and value > last:
            self._rises.setdefault(key, []).append((now, value - last))
        self._last[key] = value

    def _in_window(self, key: tuple, now: float) -> float:
        rises = self._rises.get(key)
        if not rises:
            return 0
        rises[:] = [(t, d) for t, d in rises if now - t < self.policy.counter_window_s]
        return sum(d for _, d in rises)

    @staticmethod
    def _whole(n: float) -> float | int:
        return int(n) if float(n).is_integer() else n

    def _rising(self, key: tuple, now: float, value: float | None, why: str) -> _Reading:
        """One cumulative counter judged as a rise inside the window."""
        if value is None:
            return _no(why)
        self._rise(key, now, value)
        n = self._whole(self._in_window(key, now))
        return _Reading(n > 0, n, 0, f"{n} new in the last {self.policy.counter_window_s:g} s")

    def _kernel(self, box: str, row: dict, now: float, kind: str, field: str) -> _Reading:
        k = row.get("kernel")
        if not isinstance(k, dict):
            return self._rising((box, kind), now, None,
                                "this box's sender broadcasts no `kernel` block (Xid / OOM counts): it predates it")
        return self._rising((box, kind), now, _num(k.get(field)),
                            k.get("why") or f"the sender read no {field.replace('_', ' ')}")

    def _roce(self, box: str, row: dict, now: float) -> _Reading:
        r = row.get("rdma")
        if not isinstance(r, dict):
            return self._rising((box, "roce"), now, None,
                                "this box's sender broadcasts no `rdma` block (RoCE port counters): it predates it")
        ports = [p for p in (r.get("ports") or []) if isinstance(p, dict)]
        if not ports:
            return self._rising((box, "roce"), now, None, r.get("why") or "no RDMA port on this box")
        read = 0
        for p in ports:
            counters = p.get("counters") if isinstance(p.get("counters"), dict) else {}
            for c in ERROR_COUNTERS:
                v = _num(counters.get(c))
                if v is not None:
                    read += 1
                    # EACH COUNTER ON ITS OWN (review r4717113268): a sum over whatever is readable this tick jumps
                    # when a port's counters come back readable — a rise nothing had.
                    self._rise((box, "roce", str(p.get("dev")), str(p.get("port")), c), now, v)
        if not read:
            return _no(self._roce_why(ports))
        rose = {k: self._in_window(k, now) for k in list(self._rises) if k[:2] == (box, "roce")}
        rose = {k: n for k, n in rose.items() if n > 0}
        n = self._whole(sum(rose.values()))
        said = ", ".join(f"{k[2]}/{k[3]} {k[4]} +{self._whole(v):g}" for k, v in sorted(rose.items()))
        return _Reading(bool(rose), n, 0, f"{said or 'none'} in the last {self.policy.counter_window_s:g} s",
                        rose=[{"dev": k[2], "port": k[3], "counter": k[4], "rise": self._whole(v)}
                              for k, v in sorted(rose.items())])

    @staticmethod
    def _roce_why(ports: list) -> str:
        """Why no judged counter is readable: the first port's reason for its first judged counter, read by the key
        the sender files it under — the counter's own name, else its group's (`why.hw_counters`, `why.ethtool`: one
        reason for a whole directory or a refused tool). Never another figure's reason (mtu, pfc, qos)."""
        for p in ports:
            why = p.get("why") if isinstance(p.get("why"), dict) else {}
            for c, group in ERROR_COUNTERS.items():
                said = why.get(c) or why.get(group)
                if isinstance(said, str) and said:
                    return f"{p.get('dev')}/{p.get('port')}: {said}"
        return "no RDMA error counter readable on any port, and the sender gives no reason"

    # ── one tick ───────────────────────────────────────────────────────────

    def evaluate(self, now: float, hardware: dict[str, dict], backends: list, *, broker_enabled: bool) -> None:
        self.evaluated_at = now
        if self.policy is None:
            return
        readings: dict[tuple[str, str], _Reading] = {}
        for box, row in sorted((hardware or {}).items()):
            if not isinstance(row, dict):
                continue
            if row.get("stale"):
                age = _num(row.get("age_s"))
                why = f"no sample for {age:.0f} s: not a reading of now" if age is not None else "no recent sample"
                for kind in KINDS[:-1]:
                    readings[(box, kind)] = _no(why)
                continue
            readings[(box, "gpu_temp")] = self._gpu_temp(box, row)
            readings[(box, "low_ram")] = self._low_ram(box, row)
            readings[(box, "low_power")] = self._low_power(box, row)
            readings[(box, "xid")] = self._kernel(box, row, now, "xid", "xid_total")
            readings[(box, "oom")] = self._kernel(box, row, now, "oom", "oom_kills_total")
            readings[(box, "roce")] = self._roce(box, row, now)
        for b in backends:
            if b.live:
                self._ever_live.add(b.url)
            if broker_enabled or b.cluster is not None or b.url not in self._ever_live:
                continue
            down = not b.live and not b.parked
            readings[(b.host_label, "backend_down")] = _Reading(
                down, "not live" if down else "live", "live",
                f"{b.host_label} answered since this LB started and has stopped answering")
        self._no_data = [{"box": box, "kind": kind, "why": r.why}
                         for (box, kind), r in readings.items() if r.holds is None]
        for (box, kind), r in readings.items():
            self._step(now, box, kind, r)
        # A finding whose box or server is no longer read at all (gone from the listener, the backend removed).
        for fid, f in list(self._findings.items()):
            if (f["box"], f["kind"]) not in readings:
                self._step(now, f["box"], f["kind"], _no("no longer reported"))

    def _hold(self, kind: str) -> float:
        return self.policy.backend_down_s if kind == "backend_down" else self.policy.hold_s

    def _step(self, now: float, box: str, kind: str, r: _Reading) -> None:
        fid = f"{box}:{kind}"
        f = self._findings.get(fid)
        if r.holds is None:
            if f is None:
                return
            if f["state"] == "pending":
                del self._findings[fid]
            else:
                f["why"] = f"no data: {r.why} (firing since {time.strftime('%H:%M:%S', time.gmtime(f['firing_since']))} UTC)"
            return
        if not r.holds:
            if f is not None:
                del self._findings[fid]
                if f["state"] == "firing":
                    label, unit = LABELS[kind]
                    figure = (f" · {r.value:g} {unit} vs {r.threshold:g} {unit}"
                              if isinstance(r.value, (int, float)) and isinstance(r.threshold, (int, float)) else "")
                    events.record("finding.cleared", box, f"{label} cleared on {box}{figure}", id=fid, finding=kind,
                                  value=r.value, threshold=r.threshold, unit=unit,
                                  fired_s=round(now - f["firing_since"], 1))
            return
        label, unit = LABELS[kind]
        if f is None:
            f = self._findings[fid] = {"id": fid, "box": box, "kind": kind, "label": label, "unit": unit,
                                       "state": "pending", "since": now, "firing_since": None}
        f.update(value=r.value, threshold=r.threshold, why=r.why)
        if r.rose is not None:
            f["rose"] = r.rose
        if f["state"] == "pending" and now - f["since"] >= self._hold(kind):
            f["state"], f["firing_since"] = "firing", now
            events.record("finding.firing", box, f"{label} on {box}: {r.why}", id=fid, finding=kind,
                          value=r.value, threshold=r.threshold, unit=unit, held_s=round(now - f["since"], 1))

    def view(self) -> dict:
        """/lb/status `health`: the findings (firing first, then pending; oldest first) and what has no data."""
        if self.policy is None:
            return {"enabled": False, "why": OFF, "findings": [], "no_data": []}
        order = {"firing": 0, "pending": 1}
        found = sorted(({k: v for k, v in f.items()} for f in self._findings.values()),
                       key=lambda f: (order[f["state"]], f["since"], f["id"]))
        return {"enabled": True, "findings": found, "no_data": sorted(self._no_data, key=lambda n: (n["box"], n["kind"])),
                "hold_s": self.policy.hold_s}


class HealthMonitor:
    """The evaluator on its own thread: every `period_s`, the listener's samples and the hub's backends, judged."""

    def __init__(self, evaluator: HealthEvaluator, *, hardware: Callable[[], dict], backends: Callable[[], list],
                 broker_enabled: bool, period_s: float, clock: Callable[[], float] = time.time) -> None:
        self.evaluator = evaluator
        self._hardware = hardware
        self._backends = backends
        self.broker_enabled = broker_enabled
        self.period_s = max(0.1, float(period_s))
        self.clock = clock
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.errors = 0
        self.last_error = ""

    def tick(self) -> None:
        try:
            hw, backends = self._hardware(), self._backends()
            with self._lock:
                self.evaluator.evaluate(self.clock(), hw, backends, broker_enabled=self.broker_enabled)
        except Exception as exc:                   # noqa: BLE001 — a rule's surprise must not stop the monitor
            with self._lock:
                self.errors += 1
                self.last_error = f"{type(exc).__name__}: {exc}"[:300]
            log.exception("[health] a tick failed")

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="health", daemon=True)
            self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.period_s):
            self.tick()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(5)
            self._thread = None

    def view(self) -> dict:
        with self._lock:
            out = self.evaluator.view()
            if self.errors:
                out["errors"], out["last_error"] = self.errors, self.last_error
            return out


# ── the process's monitor and alert sender, as /lb/status reads them ──────

_monitor: HealthMonitor | None = None
_sender = None


def install(monitor: HealthMonitor | None, sender=None) -> None:
    """main, at boot: the monitor whose findings /lb/status carries, and the alert sender whose state it carries."""
    global _monitor, _sender
    _monitor, _sender = monitor, sender


def status_view() -> dict:
    """/lb/status `health`: the findings and what has no data (`HealthEvaluator.view`) and the alert sender's state —
    small and stable: it changes when a finding, a figure of one, or a send does, never on its own clock."""
    out = _monitor.view() if _monitor is not None else {"enabled": False, "why": "the health monitor is not running",
                                                         "findings": [], "no_data": []}
    if _sender is not None:
        out["alerts"] = _sender.status()
    return out
