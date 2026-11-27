"""PROMETHEUS TEXT EXPOSITION OF THE GATEWAY'S OWN VIEW — `GET /metrics`, opt-in (LB_METRICS).

WHY FROM THE STATUS DOCUMENT. The panel reads one document, /lb/status (`Proxy.status()`): every backend's liveness
and depth, every cluster's state, every box's hardware sample. A second set of collectors for Prometheus would be a
second definition of each figure — the day they disagree, the alert and the panel argue. So this module is a pure
RENDERER of that same dict: what the panel shows is what is scraped, and a figure added to the status is one line here.

ABSENT IS OMITTED, NEVER 0 (the project's no-fallback rule, and Prometheus's own: a missing series is "no data", a 0 is
a measurement). A None, a NaN/inf, a boolean where a number belongs, a stale box's withdrawn live figures, a key a
sender does not send (the RoCE ports, kernel counters and disks of the box-health contract when a box's sender predates
them) — each makes NO line. The one series every heard box always has is
`collector_last_success_timestamp_seconds{box}`: when its last sample arrived (status `now` − `age_s`), which is how a
dashboard tells a silent box from an idle one.

UNITS are Prometheus's base units where the status has a convertible one: MiB → bytes, Mb/s → bytes/s; percentages
and °C and W stay as they are (the conventional `_percent`, `_celsius`, `_watts` suffixes say so).

WHO MAY READ IT. The route sits behind the same trust boundary as /lb/status: the host allowlist first, and a member
(roles.py) is refused — `/metrics` is in no member allowlist and is no static file. Off (the default) it is a 404 that
names the switch; it is never a silently empty page.
"""
from __future__ import annotations

import math
from typing import Any, Iterable

CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"
_MIB = 1 << 20


def _n(v: Any) -> float | int | None:
    """A finite number, or None — a boolean is not a reading."""
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        return None
    return v


def _esc(v: Any) -> str:
    return str(v).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _fmt(v: float | int) -> str:
    if isinstance(v, int):
        return str(v)
    return str(int(v)) if v.is_integer() and abs(v) < 1e15 else repr(v)


class _Out:
    """Families in first-seen order; each family's HELP/TYPE once, then its samples."""

    def __init__(self) -> None:
        self.families: dict[str, tuple[str, str, list[str]]] = {}

    def add(self, name: str, kind: str, help_: str, labels: Iterable[tuple[str, Any]], value: Any,
            scale: float = 1.0) -> None:
        v = _n(value)
        if v is None:
            return
        if scale != 1.0:
            v = v * scale
        fam = self.families.setdefault(name, (kind, help_, []))
        # An empty or absent value is NO label: Prometheus treats `cluster=""` as the label not being there, so
        # writing it out only adds noise to every series of a backend that is in no cluster.
        lab = ",".join(f'{k}="{_esc(x)}"' for k, x in labels if x is not None and str(x) != "")
        fam[2].append(f"{name}{{{lab}}} {_fmt(v)}" if lab else f"{name} {_fmt(v)}")

    def text(self) -> str:
        out = []
        for name, (kind, help_, samples) in self.families.items():
            if not samples:
                continue
            out.append(f"# HELP {name} {help_}")
            out.append(f"# TYPE {name} {kind}")
            out.extend(samples)
        return "\n".join(out) + "\n"


def _backends(o: _Out, backends) -> None:
    for b in backends or []:
        if not isinstance(b, dict):
            continue
        lab = (("url", b.get("url") or ""), ("box", b.get("box") or ""), ("kind", b.get("kind") or ""),
               ("cluster", b.get("cluster") or ""))
        o.add("fleet_backend_up", "gauge", "1 when the backend answered its last probe as able to serve", lab,
              1 if b.get("live") else 0)
        if b.get("depth_known"):
            o.add("fleet_backend_requests_waiting", "gauge", "requests the engine reports waiting", lab, b.get("waiting"))
            o.add("fleet_backend_requests_running", "gauge", "requests the engine reports running", lab, b.get("running"))
        o.add("fleet_backend_requests_inflight", "gauge", "requests this gateway has relayed and not finished", lab,
              b.get("inflight"))
        o.add("fleet_backend_tokens_per_second", "gauge", "generated tokens per second, measured across probes", lab,
              b.get("tokens_per_s"))
        o.add("fleet_backend_generated_tokens_total", "counter", "the engine's generated-token counter", lab,
              b.get("gen_tokens_total"))
        o.add("fleet_backend_prompt_tokens_total", "counter", "the engine's prompt-token counter", lab,
              b.get("prompt_tokens_total"))


def _clusters(o: _Out, clusters) -> None:
    for name, c in (clusters or {}).items():
        state = c.get("state") if isinstance(c, dict) else None
        if isinstance(state, str) and state:
            o.add("fleet_cluster_state", "gauge", "1 for the state the cluster's loop last published",
                  (("cluster", name), ("state", state)), 1)


def _box(o: _Out, box: str, h: dict, now) -> None:
    b = (("box", box),)
    age = _n(h.get("age_s"))
    if age is not None and _n(now) is not None:
        o.add("collector_last_success_timestamp_seconds", "gauge",
              "unix time the box's last hardware sample arrived", b, now - age)
    o.add("fleet_box_sample_age_seconds", "gauge", "seconds since the box's last hardware sample", b, age)
    o.add("fleet_box_cpu_percent", "gauge", "CPU utilisation", b, h.get("cpu_percent"))
    o.add("fleet_box_ram_percent", "gauge", "RAM used", b, h.get("ram_percent"))
    o.add("fleet_box_gpu_memory_used_bytes", "gauge", "GPU memory used, as the box's memory model reads it", b,
          h.get("mem_used_mb"), _MIB)
    o.add("fleet_box_gpu_memory_total_bytes", "gauge", "GPU memory total, as the box's memory model reads it", b,
          h.get("mem_total_mb"), _MIB)
    for g in h.get("gpus") or []:
        if isinstance(g, dict):
            o.add("fleet_box_gpu_utilization_percent", "gauge", "one card's utilisation",
                  b + (("card", g.get("id")),), g.get("use_pct"))
    figs = ((h.get("recorder") or {}).get("figures") or {}) if isinstance(h.get("recorder"), dict) else {}
    o.add("fleet_box_gpu_temperature_celsius", "gauge", "the hottest card's temperature", b, figs.get("gpu_temp_c"))
    o.add("fleet_box_gpu_power_watts", "gauge", "the box's cards' summed averaged draw", b, figs.get("gpu_power_w"))
    o.add("fleet_box_board_temperature_celsius", "gauge", "the hottest thermal zone", b, figs.get("board_temp_c"))
    o.add("fleet_box_cpu_temperature_celsius", "gauge", "the CPU temperature sensor", b, figs.get("cpu_temp_c"))
    o.add("fleet_box_network_receive_bytes_per_second", "gauge", "network in, from the box's counters", b,
          figs.get("net_in_mbps"), 1e6 / 8)
    o.add("fleet_box_network_transmit_bytes_per_second", "gauge", "network out, from the box's counters", b,
          figs.get("net_out_mbps"), 1e6 / 8)
    rdma = h.get("rdma")
    for p in (rdma.get("ports") or []) if isinstance(rdma, dict) else []:
        if not isinstance(p, dict):
            continue
        pl = b + (("dev", p.get("dev") or ""), ("port", p.get("port") if p.get("port") is not None else ""))
        if isinstance(p.get("state"), str):
            o.add("fleet_box_rdma_port_active", "gauge", "1 when the RDMA port's state is ACTIVE", pl,
                  1 if p["state"].upper() == "ACTIVE" else 0)
        o.add("fleet_box_rdma_port_rate_gbps", "gauge", "the RDMA port's link rate", pl, p.get("rate_gbps"))
        o.add("fleet_box_rdma_port_mtu", "gauge", "the RDMA port's active MTU", pl, p.get("mtu"))
        for k, v in sorted((p.get("counters") or {}).items()):
            o.add("fleet_box_rdma_counter_total", "counter", "an RDMA port's cumulative error/drop counter",
                  pl + (("counter", k),), v)
        for k, v in sorted((p.get("rates") or {}).items()):
            o.add("fleet_box_rdma_counter_rate", "gauge", "an RDMA port counter's per-second rate",
                  pl + (("counter", k),), v)
    kernel = h.get("kernel")
    if isinstance(kernel, dict):
        o.add("fleet_box_gpu_xid_errors_total", "counter", "GPU Xid events the kernel logged", b,
              kernel.get("xid_total"))
        o.add("fleet_box_oom_kills_total", "counter", "OOM kills the kernel logged", b, kernel.get("oom_kills_total"))
    disks = h.get("disks")
    for d in disks if isinstance(disks, list) else []:
        if not isinstance(d, dict):
            continue
        dl = b + (("path", d.get("path") or ""),)
        o.add("fleet_box_disk_total_bytes", "gauge", "the filesystem's size", dl, d.get("total_bytes"))
        o.add("fleet_box_disk_used_bytes", "gauge", "the filesystem's used bytes", dl, d.get("used_bytes"))
        o.add("fleet_box_disk_free_bytes", "gauge", "the filesystem's free bytes", dl, d.get("free_bytes"))


def render(status: dict) -> str:
    """The exposition of `status` (Proxy.status()'s document)."""
    o = _Out()
    _backends(o, status.get("backends"))
    _clusters(o, status.get("clusters"))
    for box, h in sorted((status.get("hardware") or {}).items()):
        if isinstance(h, dict):
            _box(o, box, h, status.get("now"))
    return o.text()
