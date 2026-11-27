"""PERSISTED HARDWARE HISTORY TIERS: 48 h at one minute and 30 days at fifteen minutes, per box, per figure — coarse
min / avg / max only, kept across gateway restarts.

WHY A SECOND STORE BESIDE THE RECORDER. The flight recorder (recorder.py) holds one hour, every figure, in memory, and
says why it is not persisted: a gateway restart costs the hour, and the boxes refill it in an hour. That answers "what
was this box doing a minute before it died". It cannot answer "has this box been running hotter this week" or "what
did the night look like" — a question about days. That needs few points over a long range, which is cheap to keep and
cheap to write; so this module keeps exactly that and nothing else, and the recorder stays as it is.

WHAT A BUCKET IS. Per tier, per box, per figure, one bucket per `bucket_s` aligned on the epoch: the min, the max, the
sum and the count of the readings that fell in it (avg = sum / count, computed on read). An absent reading (None — a
sender that does not send the figure, a first network sample) adds NOTHING: a bucket with no reading does not exist,
and a gap in the series is a gap on the chart, never a 0.

    tier   bucket   window   buckets per figure per box
    48h    60 s     48 h     2 880
    30d    900 s    30 d     2 880

THE FIGURES (`METRICS`): GPU temperature, GPU power (the recorder's box headlines — the hottest card, the summed
draw), GPU utilisation (hw.py's busiest card), GPU memory used (the box's memory model's read), network in and out
(the recorder's counter-delta rate). The figure definitions are the recorder's and hw.py's, imported, never re-derived
here: `recorder.figures_of` is called with this module's own previous network counter.

PERSISTENCE. One JSON file (`LB_HW_TIERS_FILE`, on the state volume), written whole by `atomic.write_json` every
`SAVE_EVERY_S` by one daemon thread, and once more on a clean stop. A crash costs at most that interval.

A CORRUPT FILE is neither a crash nor a silent reset. The tiers start empty, the reason (the parser's words) is logged
at ERROR and kept in the status (`state: corrupt`, `why`) until the next restart, and the file is RENAMED aside
(`<file>.corrupt-<epoch>`) so the first save after start cannot overwrite the evidence. No file yet is `empty`, said.
An empty setting is `off`, said, and the endpoint answers 503 naming the setting.

PRUNED, NEVER GROWING WITH THE FLEET'S PAST. A box whose newest bucket is older than the longest tier's window (30 d)
has nothing left to draw: it is dropped on every save and at load. A box the fleet file no longer names is dropped at
load. Each drop is logged by name — a box's history never vanishes without a line saying so.

Thread-safety: one lock; `ingest` runs on the hardware listener's thread, `series` / `status` on request threads,
`save` on the saver thread.
"""
from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from typing import Any, Callable

from fleet_telemetry import figures as F
from vllm_lb import atomic
from vllm_lb.recorder import figures_of

log = logging.getLogger(__name__)

#: tier → (bucket seconds, window seconds).
TIERS: dict[str, tuple[int, int]] = {"48h": (60, 48 * 3600), "30d": (900, 30 * 86400)}
#: (key, label, unit) — the figures kept, in the panel's order.
METRICS_TABLE: tuple[tuple[str, str, str], ...] = (
    ("gpu_temp_c", "GPU TEMP", "°C"),
    ("gpu_power_w", "GPU POWER", "W"),
    ("gpu_pct", "GPU UTIL", "%"),
    ("mem_used_mb", "GPU MEM", "MiB"),
    ("net_in_mbps", "NET IN", "Mb/s"),
    ("net_out_mbps", "NET OUT", "Mb/s"),
)
METRICS: tuple[str, ...] = tuple(k for k, _, _ in METRICS_TABLE)
#: How often the saver thread writes the file.
SAVE_EVERY_S = 300.0
FILE_VERSION = 1
#: A box name that runs on is not a box; a file with more boxes than this is not ours.
MAX_BOXES = 256


def _finite(v: Any) -> float | int | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        return None
    return v


class _Corrupt(ValueError):
    pass


class HwTiers:
    """The two tiers, their file and their saver thread."""

    def __init__(self, path: str, *, clock: Callable[[], float] = time.time,
                 save_every_s: float = SAVE_EVERY_S, boxes=None) -> None:
        self.path = path
        #: The boxes the fleet file names (None = no fleet to check against): a box in the file that is not one of
        #: them is dropped at load, named in the log — a retired box's history is not this fleet's.
        self.fleet_boxes = None if boxes is None else frozenset(boxes)
        self.clock = clock
        self.save_every_s = save_every_s
        self._lock = threading.Lock()
        #: tier → box → metric → {bucket_start: [min, max, sum, count]}
        self._t: dict[str, dict[str, dict[str, dict[int, list]]]] = {k: {} for k in TIERS}
        #: box → (t, rx, tx): the previous network counters, for the rate.
        self._net: dict[str, tuple[float, Any, Any]] = {}
        self.state = "off" if not path else "not loaded"
        self.why = "LB_HW_TIERS_FILE is empty: hardware history is not kept" if not path else ""
        self.saved_at: float | None = None
        self.save_error = ""
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ── intake ─────────────────────────────────────────────────────────────────────────────────

    def ingest(self, box: str, figs: dict, now: float) -> None:
        """Fold one sample's figures (`figs`: METRICS keys → number or None) into both tiers."""
        with self._lock:
            for tier, (bucket_s, window_s) in TIERS.items():
                per_box = self._t[tier].setdefault(box, {})
                start = int(now // bucket_s * bucket_s)
                for key in METRICS:
                    v = _finite(figs.get(key))
                    series = per_box.setdefault(key, {})
                    if v is None:
                        continue
                    b = series.get(start)
                    if b is None:
                        series[start] = [v, v, v, 1]
                        floor = now - window_s
                        for old in [s for s in series if s + bucket_s <= floor]:
                            del series[old]
                    else:
                        b[0] = min(b[0], v)
                        b[1] = max(b[1], v)
                        b[2] += v
                        b[3] += 1

    def ingest_sample(self, box: str, data: dict, now: float, *, gpu_pct, mem_used_mb) -> None:
        """One hardware datagram (hw.py's, already mapped to `box` and de-duplicated): the recorder's figures with this
        store's own previous network counter, plus hw.py's busiest-card utilisation and memory read."""
        deep = data.get(F.DEEP) if isinstance(data.get(F.DEEP), dict) else {}
        rx, tx = _finite(deep.get(F.NET_RX_BYTES)), _finite(deep.get(F.NET_TX_BYTES))
        figs, _why = figures_of(data, self._net.get(box), now)
        self._net[box] = (now, rx, tx)
        self.ingest(box, {**figs, "gpu_pct": gpu_pct, "mem_used_mb": mem_used_mb}, now)

    # ── read ───────────────────────────────────────────────────────────────────────────────────

    def boxes(self) -> list[str]:
        with self._lock:
            return sorted(set().union(*(t.keys() for t in self._t.values())))

    def series(self, box: str, metric: str, tier: str, *, now: float | None = None) -> dict | None:
        """One figure's points in a tier: [[bucket_start, min, avg, max], …] oldest first, inside the window. None for a
        box never heard; KeyError for a metric or tier that does not exist."""
        if tier not in TIERS:
            raise KeyError(tier)
        if metric not in METRICS:
            raise KeyError(metric)
        now = self.clock() if now is None else now
        bucket_s, window_s = TIERS[tier]
        with self._lock:
            per_box = self._t[tier].get(box)
            if per_box is None:
                return None
            floor = now - window_s
            pts = [[float(s), b[0], round(b[2] / b[3], 3), b[1]]
                   for s, b in sorted(per_box.get(metric, {}).items()) if s + bucket_s > floor]
        label = next((lab, unit) for k, lab, unit in METRICS_TABLE if k == metric)
        return {"box": box, "metric": metric, "tier": tier, "bucket_s": bucket_s, "window_s": window_s,
                "label": label[0], "unit": label[1], "points": pts}

    def _prune(self, now: float) -> list[str]:
        """Drop every box whose newest bucket in any tier is older than the LONGEST tier's window: nothing of it is
        left to draw, and a box that went silent must not stay in the file forever. Returns the boxes dropped.
        Called with the lock held."""
        longest = max(w for _, w in TIERS.values())
        newest: dict[str, float] = {}
        for tier, (bucket_s, _w) in TIERS.items():
            for box, per in self._t[tier].items():
                top = max((s + bucket_s for series in per.values() for s in series), default=None)
                if top is not None:
                    newest[box] = max(newest.get(box, top), top)
        boxes = set().union(*(t.keys() for t in self._t.values()))
        gone = sorted(b for b in boxes if newest.get(b) is None or newest[b] <= now - longest)
        for tier in self._t.values():
            for b in gone:
                tier.pop(b, None)
        for b in gone:
            self._net.pop(b, None)
        return gone

    def status(self) -> dict:
        """/lb/status `hardware_tiers` — small and stable: no points, only what the store is and how it fares."""
        return {"state": self.state, "why": self.why, "path": self.path, "saved_at": self.saved_at,
                "save_error": self.save_error, "tiers": {k: {"bucket_s": b, "window_s": w} for k, (b, w) in TIERS.items()},
                "metrics": [{"key": k, "label": lab, "unit": u} for k, lab, u in METRICS_TABLE]}

    # ── the file ───────────────────────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Read the file once, at start. A missing file is `empty`; an unreadable one is `corrupt`, set aside."""
        if not self.path:
            return
        try:
            with open(self.path, encoding="utf-8") as fh:
                raw = fh.read()
        except FileNotFoundError:
            self.state, self.why = "empty", f"no file at {self.path} yet: history starts now"
            return
        except OSError as exc:
            self.state, self.why = "corrupt", f"cannot read {self.path}: {exc}"
            log.error("[hw_tiers] corrupt: %s — starting empty", self.why)
            return
        try:
            tiers = self._parse(raw)
        except (_Corrupt, ValueError) as exc:
            aside = f"{self.path}.corrupt-{int(self.clock())}"
            try:
                os.replace(self.path, aside)
                kept = f"set aside as {aside}"
            except OSError as move:
                kept = f"NOT set aside ({move}): the next save will replace it"
            self.state, self.why = "corrupt", f"{self.path}: {exc} — starting empty; the file is {kept}"
            log.error("[hw_tiers] corrupt history file: %s", self.why)
            return
        retired = sorted(set().union(*(t.keys() for t in tiers.values())) - self.fleet_boxes) \
            if self.fleet_boxes is not None else []
        for tier in tiers.values():
            for b in retired:
                tier.pop(b, None)
        if retired:
            log.info("[hw_tiers] dropped %s from %s: no longer in the fleet file", ", ".join(retired), self.path)
        with self._lock:
            self._t = tiers
            silent = self._prune(self.clock())
        if silent:
            log.info("[hw_tiers] dropped %s from %s: silent past the %s tier", ", ".join(silent), self.path,
                     max(TIERS, key=lambda k: TIERS[k][1]))
        self.state, self.why = "loaded", ""
        log.info("[hw_tiers] loaded %s (%d boxes)", self.path, len(self.boxes()))

    @staticmethod
    def _parse(raw: str) -> dict:
        doc = json.loads(raw)
        if not isinstance(doc, dict) or doc.get("v") != FILE_VERSION:
            raise _Corrupt(f"not a version-{FILE_VERSION} history file")
        tiers = doc.get("tiers")
        if not isinstance(tiers, dict):
            raise _Corrupt("no `tiers` mapping")
        out: dict = {k: {} for k in TIERS}
        for tier, boxes in tiers.items():
            if tier not in TIERS:
                raise _Corrupt(f"unknown tier {tier!r}")
            if not isinstance(boxes, dict) or len(boxes) > MAX_BOXES:
                raise _Corrupt(f"tier {tier}: not a mapping of boxes")
            for box, metrics in boxes.items():
                if not isinstance(metrics, dict):
                    raise _Corrupt(f"tier {tier} box {box!r}: not a mapping of figures")
                per = out[tier][box] = {}
                for key, rows in metrics.items():
                    if key not in METRICS or not isinstance(rows, list):
                        raise _Corrupt(f"tier {tier} box {box!r}: unknown figure {key!r}")
                    series = per[key] = {}
                    for row in rows:
                        if (not isinstance(row, list) or len(row) != 5
                                or any(_finite(x) is None for x in row) or row[4] < 1):
                            raise _Corrupt(f"tier {tier} box {box!r} {key}: a bucket is not [t, min, max, sum, n]")
                        series[int(row[0])] = [row[1], row[2], row[3], int(row[4])]
        return out

    def save(self, now: float | None = None) -> None:
        """Write both tiers whole (atomic.write_json). A failure is logged and kept in the status, never raised."""
        if not self.path:
            return
        now = self.clock() if now is None else now
        with self._lock:
            silent = self._prune(now)
            doc = {"v": FILE_VERSION, "saved_at": now, "tiers": {
                tier: {box: {k: [[s, *b] for s, b in sorted(series.items())] for k, series in per.items() if series}
                       for box, per in boxes.items()}
                for tier, boxes in self._t.items()}}
        try:
            atomic.write_json(self.path, doc)
        except (OSError, ValueError, TypeError) as exc:
            self.save_error = f"{type(exc).__name__}: {exc}"[:240]
            log.error("[hw_tiers] save failed: %s", self.save_error)
            return
        self.saved_at, self.save_error = now, ""
        if silent:
            log.info("[hw_tiers] dropped %s: silent past the longest tier", ", ".join(silent))

    # ── lifecycle ──────────────────────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Load, then save every `save_every_s` on one daemon thread."""
        self.load()
        if not self.path:
            return
        self._thread = threading.Thread(target=self._loop, name="hw-tiers", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(self.save_every_s):
            self.save()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self.save()
