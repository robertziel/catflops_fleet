"""What the boxes ARE doing, taken off the LAN — and no credential anywhere.

`worklog/vllm-lb-hardware.md` framed this as an ssh-fed host-side collector writing a file,
because at the time nothing on the fleet served hardware without a login and the KILL line was
"if the LB ever needs a credential to obtain this, stop". broker_parrot moved first:
`hw-broadcast` now runs on box-c, box-a and box-b and sends one JSON datagram every 5 s on
UDP 48291. So the LB listens, holds nothing secret, dials nowhere, and a box that stops
broadcasting simply goes quiet.

Three rules the shape of this file exists to keep:

* **A hostname is not a box.** The wire says `<hostname>` (e.g. a GB10 box's own name); `avoid_box` says `box-a`. The map
  is configuration (the fleet file's `boxes.<box>.hostname`), and an unmapped sender is ignored rather than guessed —
  labelling box-b's memory as box-a's would be worse than showing nothing.
* **Staleness is data.** Every box carries `age_s`; past `hw_max_age_s` its live numbers go
  `None` while its history stays. The `depth_known` lesson: a figure that quietly stopped moving
  is worse than a blank.
* **Never report a number the wire did not send.** A GB10 broadcasts `vram_used_mb: 0,
  vram_total_mb: 0` because its memory is UNIFIED with system RAM — "0GB VRAM" would be an
  invention. What a box's GPU memory IS, and how it is read, is the hardware domain's
  (`vllm_lb.hardware`: UnifiedMemory, DiscreteMemory); `mem_unified` says which one a box is.

Nothing here can take the gateway down: a malformed datagram is counted, a port that will not
bind costs the hardware tiles and nothing else, and the listener owns no lock the proxy needs.
"""
from __future__ import annotations

import json
import logging
import math
import socket
import threading
import time
from collections import deque
from typing import Any, Callable

from vllm_lb.config import Config
from vllm_lb.hardware import MemoryModel
from vllm_lb.recorder import FlightRecorder

log = logging.getLogger(__name__)

#: Bigger than any sample observed (337 bytes for a 1-GPU box); a jumbo datagram is not ours.
MAX_DATAGRAM = 65535

#: `cpu_per_core` longer than this is cut here: the fleet's widest box has 32 logical CPUs, and
#: a list that runs on is not a CPU — the status page should not grow with it.
MAX_CORES = 256

#: Cards per box this listener keeps (the `gpus` list and one utilisation ring each): the fleet's widest box has
#: two, and a list that runs on is not a box's GPUs — a sender cycling ids must not grow the rings without bound.
MAX_CARDS = 16
#: A card name the sample carries (no sender this repo ships sends one), clipped to what a label can show.
MAX_NAME = 64


class _Box:
    """One box's latest sample plus its rings. Plain object: the collector owns the lock."""

    def __init__(self, host: str, ring_n: int) -> None:
        self.host = host
        self.at = 0.0
        self.sample: dict[str, Any] = {}
        #: One ring per figure the page draws a past of. `mem_pct` is the GPU's memory (its memory model's
        #: `read`: a card's own VRAM, else the RAM a GB10 shares with it); `ram_pct` is the box's own RAM, the
        #: figure its MEM ring shows. On a GB10 they are one number. On a box with VRAM of its own they
        #: are not: box-c's iGPU sat at 485/512 MB (95 %) under 20 % of RAM, and a MEM graph drawn
        #: from `mem_pct` was red beneath a green MEM ring.
        self.ring: dict[str, deque] = {k: deque(maxlen=max(1, ring_n))
                                       for k in ("gpu_pct", "cpu_pct", "mem_pct", "ram_pct")}
        #: One utilisation ring per CARD (nvidia-smi's id): a discrete box draws each card's own past. A card a
        #: sample lacks gets None for that sample, so every ring keeps one clock.
        self.cards: dict[int, deque] = {}
        self.ring_n = max(1, ring_n)


def _num(value: Any) -> float | int | None:
    """A number, or None. Booleans are not numbers here (`True` would render as 1%)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


def _cores(value: Any) -> list[float | int] | None:
    """`cpu_per_core` as sent (one % per logical CPU, psutil order), or None. All or nothing:
    one non-number and the list is withheld, because dropping it would shift every later core
    into the wrong cell. NaN/inf too — they would make `/lb/status` invalid JSON. A sender
    that predates the field sends none, and gets None."""
    if not isinstance(value, list):
        return None
    out: list[float | int] = []
    for v in value[:MAX_CORES]:
        n = _num(v)
        if n is None or not math.isfinite(n):
            return None
        out.append(n)
    return out


def _finite(value: Any) -> float | int | None:
    n = _num(value)
    return n if n is not None and math.isfinite(n) else None


def _card(raw: Any) -> dict | None:
    """One card as the sender broadcast it — `{id, use_pct, vram_used_mb, vram_total_mb}`, plus `name` only when the
    sample carries one — or None for an entry that is not a card: no id, a negative or boolean id, a figure that is
    NaN/inf (they would make /lb/status invalid JSON). Withheld whole, never half-read."""
    if not isinstance(raw, dict):
        return None
    cid = raw.get("id")
    if isinstance(cid, bool) or not isinstance(cid, int) or cid < 0:
        return None
    out: dict[str, Any] = {"id": cid}
    for key in ("use_pct", "vram_used_mb", "vram_total_mb"):
        v = raw.get(key)
        if v is not None and _finite(v) is None:
            return None
        out[key] = _finite(v)
    name = raw.get("name")
    if isinstance(name, str) and name.strip():
        out["name"] = name.strip()[:MAX_NAME]
    return out


def _cards_of(gpus: list) -> list[dict]:
    """The sample's cards, malformed entries withheld, a repeated id kept once, at most MAX_CARDS."""
    out: list[dict] = []
    for raw in gpus:
        card = _card(raw)
        if card is not None and all(c["id"] != card["id"] for c in out):
            out.append(card)
        if len(out) == MAX_CARDS:
            break
    return out


def _gpu_percent(gpus: list) -> int | None:
    """The BUSIEST card. A box with two GPUs where one is pinned is not half idle — the tile
    has one number, and the honest one is the constraint."""
    seen = [_finite(g.get("use_pct")) for g in gpus if isinstance(g, dict)]
    seen = [int(v) for v in seen if v is not None]          # NaN/inf are no reading (and int() would raise)
    return max(seen) if seen else None


class HwCollector:
    """Listens for hardware datagrams and answers `snapshot()`. Thread-safe."""

    def __init__(self, cfg: Config, *, clock: Callable[[], float] = time.time) -> None:
        self._cfg = cfg
        self.clock = clock
        self._lock = threading.Lock()
        self._boxes: dict[str, _Box] = {}
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        #: Counters, shown in /lb/status so a silent fleet can be told from a silent bug.
        self.received = 0
        self.dropped = 0            # undecodable / not ours
        self.ignored = 0            # decoded fine, but the host maps to no box
        self.duplicates = 0         # a repeat of the box's last sample (the sender sends each to two broadcasts)
        self.state = "not started"
        #: The per-box FLIGHT RECORDER (recorder.py): the last hour of every recorder figure a box sends. Fed by
        #: `ingest` with every sample that maps to a box; read by `snapshot` (the summary) and the
        #: /lb/hardware/<box>/recorder endpoint (`recorder_series`, the whole hour).
        self.recorder = FlightRecorder()

    # ── intake ─────────────────────────────────────────────────────────────

    def ingest(self, payload: bytes | str, *, now: float | None = None) -> str | None:
        """Decode one datagram. Returns the box it belongs to, or None. Never raises."""
        now = self.clock() if now is None else now
        try:
            if isinstance(payload, (bytes, bytearray)):
                payload = bytes(payload).decode("utf-8")
            data = json.loads(payload)
            host = data.get("host") if isinstance(data, dict) else None
            if not isinstance(host, str) or not host:
                raise ValueError("no host in sample")
        except Exception:
            self.dropped += 1
            return None

        box = self._cfg.hw_hosts.get(host)
        if not box:
            self.ignored += 1
            return None

        gpus = data.get("gpus") if isinstance(data.get("gpus"), list) else []
        gpu_pct = _gpu_percent(gpus)
        memory = self._memory_of(box)
        used, total = memory.read(data, gpus) if memory is not None else (None, None)
        mem_pct = round(100.0 * used / total, 1) if used is not None and total else None
        cards = {c["id"]: c for c in _cards_of(gpus)}

        with self._lock:
            b = self._boxes.get(box)
            if b is None:
                b = self._boxes[box] = _Box(host, self._cfg.hw_history)
            # THE SAME SAMPLE TWICE IS ONE SAMPLE. The sender sends each to the subnet and the limited broadcast
            # (fleet_telemetry/broadcast.py), so a gateway on its LAN hears it twice; the repeat doubled every ring
            # and made the recorder's network rate a 0 Mb/s delta every other sample. A sender without `sampled_at`
            # cannot be told repeating, and each of its datagrams stays a sample.
            stamp = data.get("sampled_at")
            if isinstance(stamp, str) and stamp and b.sample.get("sampled_at") == stamp:
                self.duplicates += 1
                return box
            b.host, b.at, b.sample = host, now, data
            b.ring["gpu_pct"].append(gpu_pct)
            b.ring["cpu_pct"].append(_num(data.get("cpu_percent")))
            b.ring["mem_pct"].append(mem_pct)
            b.ring["ram_pct"].append(_num(data.get("ram_percent")))     # the live `ram_percent`'s past
            for cid in cards:
                if cid not in b.cards and len(b.cards) < MAX_CARDS:
                    b.cards[cid] = deque(maxlen=b.ring_n)
            for cid, ring in b.cards.items():
                ring.append(cards[cid]["use_pct"] if cid in cards else None)
            self.received += 1
        self.recorder.ingest(box, data, now)
        return box

    def _named(self, box: str):
        """The fleet's entry for `box` (vllm_lb/fleet.py `Box`), or None where no fleet names it."""
        fleet = self._cfg.fleet
        return fleet.boxes.get(box) if fleet is not None else None

    def _memory_of(self, box: str) -> MemoryModel | None:
        """The box's memory model, NAMED by its cards' types in the fleet file — never inferred from the sample. None
        for a box no fleet names: then no memory figure and no GPU row is claimed for it."""
        named = self._named(box)
        return named.cards[0].type.memory if named is not None and named.cards else None

    def ingest_and_snapshot(self, payload: bytes | str, *, now: float | None = None) -> dict:
        """Test/diagnostic convenience: ingest one sample and read that box back."""
        box = self.ingest(payload, now=now)
        return self.snapshot(now=now).get(box, {}) if box else {}

    # ── read ───────────────────────────────────────────────────────────────

    def snapshot(self, now: float | None = None) -> dict[str, dict]:
        now = self.clock() if now is None else now
        out: dict[str, dict] = {}
        with self._lock:
            for box, b in self._boxes.items():
                age = round(now - b.at, 1)
                stale = age > self._cfg.hw_max_age_s
                s = b.sample
                gpus = s.get("gpus") if isinstance(s.get("gpus"), list) else []
                memory = self._memory_of(box)
                used, total = memory.read(s, gpus) if memory is not None else (None, None)
                cards = _cards_of(gpus)
                live: dict[str, Any] = {
                    "cpu_percent": _num(s.get("cpu_percent")),
                    # Current value only — no ring: 32 cores × 60 samples is a page, not a tile.
                    "cpu_per_core": _cores(s.get("cpu_per_core")),
                    "ram_percent": _num(s.get("ram_percent")),
                    "ram_used_mb": _num(s.get("ram_used_mb")),
                    "ram_total_mb": _num(s.get("ram_total_mb")),
                    "swap_used_mb": _num(s.get("swap_used_mb")),
                    "gpu_pct": _gpu_percent(gpus),
                    "gpu_count": len(gpus),
                    "mem_used_mb": used,
                    "mem_total_mb": total,
                    # Every card as the sender broadcast it — what a reader that needs ONE card (the card a
                    # server runs on, vllm_lb/topology.py) reads, never the busiest.
                    "gpus": cards,
                }
                if stale:
                    # The history is what it DID; that stays true and stays drawn. Only the
                    # "now" numbers are withdrawn, because they are no longer now.
                    live = {k: None for k in live}
                named = self._named(box)
                history = {k: list(v) for k, v in b.ring.items()}
                history["gpus"] = {str(cid): list(ring) for cid, ring in sorted(b.cards.items())}
                out[box] = {
                    "host": b.host, "age_s": age, "stale": stale,
                    "mem_unified": memory.unified if memory is not None else None,
                    # The box's memory model and cards, as its FLEET names them (never read off the wire).
                    "memory": memory.name if memory is not None else None,
                    "cards": [c.view() for c in named.cards] if named is not None else [],
                    "rows": self._rows(memory, named, {c["id"]: c for c in cards}, history,
                                       live["gpu_pct"], stale),
                    **live,
                    "history": history,
                    "recorder": self.recorder.summary(box, now, stale=stale, age_s=age),
                }
        return out

    @staticmethod
    def _rows(memory: MemoryModel | None, named, cards: dict[int, dict], history: dict, gpu_pct,
              stale: bool) -> list[dict]:
        """The machine tile's GPU rows, as the box's memory model lays them out (vllm_lb.hardware `rows`): ONE row
        for a unified box — the box's GPU, its utilisation the busiest card's and its past `gpu_pct`'s — and one
        per card the fleet names for a discrete box, each with its own utilisation, past and VRAM line. A card the
        sample lacks is a row with no reading; a stale box keeps its rows and withdraws their readings."""
        if memory is None or named is None:
            return []
        rows = []
        for spec in memory.rows(tuple(c.index for c in named.cards)):
            if spec.card is None:
                rows.append({"label": spec.label, "card": None, "box_gpu": True, "use_pct": gpu_pct,
                             "history": history["gpu_pct"], "vram_label": None, "vram_used_mb": None,
                             "vram_total_mb": None})
                continue
            card = {} if stale else cards.get(spec.card, {})
            has_vram = spec.vram and (card.get("vram_total_mb") or 0) > 0
            label = f"VRAM {spec.card}" + (f" · {card['name']}" if card.get("name") else "")
            rows.append({"label": spec.label, "card": spec.card, "box_gpu": False, "use_pct": card.get("use_pct"),
                         "history": history["gpus"].get(str(spec.card), []),
                         "vram_label": label if spec.vram else None,
                         "vram_used_mb": card.get("vram_used_mb") if has_vram else None,
                         "vram_total_mb": card.get("vram_total_mb") if has_vram else None})
        return rows

    def recorder_series(self, box: str, now: float | None = None) -> dict | None:
        """The box's whole recorder hour (recorder.py `series`), aged like `snapshot`; None for a box never heard."""
        now = self.clock() if now is None else now
        with self._lock:
            b = self._boxes.get(box)
            if b is None:
                return None
            age = round(now - b.at, 1)
        return {"box": box, **(self.recorder.series(box, now, stale=age > self._cfg.hw_max_age_s, age_s=age) or {})}

    def status(self) -> dict:
        return {"state": self.state, "port": self._cfg.hw_port, "received": self.received,
                "dropped": self.dropped, "ignored": self.ignored, "duplicates": self.duplicates,
                "hosts": dict(self._cfg.hw_hosts)}

    # ── lifecycle ──────────────────────────────────────────────────────────

    def start(self) -> None:
        """Bind and listen. Best-effort by design: a port we cannot have costs hardware tiles,
        never the gateway."""
        if not self._cfg.hw_port:
            self.state = "off: LB_HW_PORT=0"
            return
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("", self._cfg.hw_port))
            sock.settimeout(1.0)
        except OSError as exc:
            self.state = f"off: cannot bind udp/{self._cfg.hw_port}: {exc}"[:160]
            log.warning("[hw] %s — hardware tiles will be blank", self.state)
            return
        self._sock = sock
        self._thread = threading.Thread(target=self._loop, name="hw-listen", daemon=True)
        self._thread.start()
        self.state = f"listening on udp/{self._cfg.hw_port}"
        log.info("[hw] %s; hosts=%s", self.state, dict(self._cfg.hw_hosts))

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                payload, _addr = self._sock.recvfrom(MAX_DATAGRAM)
            except socket.timeout:
                continue
            except OSError:
                if self._stop.is_set():
                    return
                time.sleep(0.5)             # a transient socket error is not a reason to exit
                continue
            try:
                self.ingest(payload)
            except Exception:               # ingest swallows its own; this is the last net
                log.exception("[hw] ingest failed")

    def stop(self) -> None:
        self._stop.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.state = "stopped"
