"""BOX HEALTH on the gateway's side: what /lb/status `hardware.<box>` says about a box's RoCE ports,
its kernel's counts, its disks and its models roots, made from the blocks its sender broadcasts
(lb/fleet_telemetry/readers/{rdma,kernel,disk,models}.py; wire names in fleet_telemetry/figures.py).

WHY THE RATES ARE HERE. The sender keeps no state between samples and sends each RDMA error counter CUMULATIVE; a
per-second rate needs two samples, and the gateway holds the last one (vllm_lb/hw.py `_Box`). The gap is the two
samples' OWN `sampled_at` (the box's clock): the LB's receive times jitter with the LAN and with the duplicate datagram
the sender sends to two broadcast addresses. A counter that went BACKWARDS was reset (a driver reload, a reboot): that
sample has no rate for it, with the reason — a negative rate, or a 0 standing in for one, would be a number nobody
measured. The first sample of a port has no rate either, and says so.

WHAT IS NEVER A NUMBER. The PHY counters (`rx_discards_phy`, …: `ethtool -S`) arrive among `counters` and get rates
like the rest; the QoS setting (`pfc_prios`, `trust`, `dscp_map`, read by `dcb` at `qos_at`) passes through, absent
with the tool's words in the port's `why` (readers/rdma.py). A sender that predates these blocks sends
none: each block is then empty with that reason, never zeros. Anything malformed on the wire (a boolean, a NaN, a
negative counter, a port with no device name) is withheld — the port whole, or that one figure.

CHECKPOINT FOLDERS. The sender sizes each top-level folder of its models roots; the fleet file's `checkpoints` say which
model sits where, relative to those roots (vllm_lb/fleet.py `catalog_deployment`). A folder is named by every
checkpoint whose path STARTS with it (`checkpoint_folders`): exact for a one-segment path, and for a deeper one the
folder that holds it — the panel shows the folder's size labelled by what it holds, never as one checkpoint's size.

THE SHAPES (the contract streams B and C read):
  rdma    {"ports": [{dev, port, state, phys_state, rate_gbps, link_layer, netdev, mtu, counters, why, rates,
                      rates_why, pfc_prios, trust, dscp_map, qos_at}], "why": str|None}
  kernel  {xid_total, oom_kills_total, last_xid, xid_since, xid_complete, why}
  disks   [{path, total_bytes, used_bytes, free_bytes[, why]}] + disks_why
  models  [{root, folders: [{name, bytes, checkpoints}], measured_at, took_s, more_count, more_bytes, unreadable,
            why[, names_why, more_why]}] + models_why      (more_count / more_bytes None, with more_why, when not sent)
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Mapping

from fleet_telemetry import figures as F

#: Ports per box kept: the widest box has four RDMA functions; a list that runs on is not a box's ports.
MAX_PORTS = 32
MAX_DISKS = 32
MAX_ROOTS = 8
MAX_FOLDERS = 64
MAX_TEXT = 200
PREDATES = "the sender sends no {} block (it predates box health)"


def _text(v: Any, n: int = MAX_TEXT) -> str | None:
    return v[:n] if isinstance(v, str) and v else None


def _count(v: Any) -> int | None:
    """A cumulative counter: a non-negative int, never a boolean."""
    return v if isinstance(v, int) and not isinstance(v, bool) and v >= 0 else None


def _num(v: Any) -> float | int | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        return None
    return v


def _whys(v: Any) -> dict[str, str]:
    return {str(k)[:64]: t for k, val in (v.items() if isinstance(v, dict) else []) if (t := _text(val))}


def _prios(v: Any) -> list[int] | None:
    """PFC's enabled priorities: a list of 0-7, all or nothing."""
    if not isinstance(v, list) or not all(_count(p) is not None and p <= 7 for p in v):
        return None
    return list(v)


def _words(v: Any) -> list[str] | None:
    if not isinstance(v, list) or not v or not all(_text(w, 16) for w in v):
        return None
    return [w[:16] for w in v[:8]]


def stamp(sample: Mapping[str, Any]) -> float | None:
    """The sample's own time (`sampled_at`, the box's clock) as epoch seconds, or None."""
    try:
        return datetime.fromisoformat(sample.get(F.SAMPLED_AT)).timestamp()
    except (TypeError, ValueError):
        return None


def _port(raw: Any) -> dict | None:
    if not isinstance(raw, dict) or not _text(raw.get(F.P_DEV), 64) or _count(raw.get(F.P_PORT)) is None:
        return None
    counters = raw.get(F.P_COUNTERS) if isinstance(raw.get(F.P_COUNTERS), dict) else {}
    return {F.P_DEV: _text(raw[F.P_DEV], 64), F.P_PORT: raw[F.P_PORT], F.P_STATE: _text(raw.get(F.P_STATE), 32),
            F.P_PHYS: _text(raw.get(F.P_PHYS), 32), F.P_RATE_GBPS: _num(raw.get(F.P_RATE_GBPS)),
            F.P_LINK_LAYER: _text(raw.get(F.P_LINK_LAYER), 32), F.P_NETDEV: _text(raw.get(F.P_NETDEV), 64),
            F.P_MTU: _count(raw.get(F.P_MTU)),
            F.P_PFC: _prios(raw.get(F.P_PFC)), F.P_TRUST: _words(raw.get(F.P_TRUST)),
            F.P_DSCP_MAP: _count(raw.get(F.P_DSCP_MAP)), F.P_QOS_AT: _num(raw.get(F.P_QOS_AT)),
            F.P_COUNTERS: {str(k)[:64]: c for k, v in counters.items() if (c := _count(v)) is not None},
            F.P_WHY: _whys(raw.get(F.P_WHY))}


def rdma_view(sample: Mapping[str, Any], prev: Mapping[tuple[str, int], tuple[float, dict]] | None
              ) -> tuple[dict, dict[tuple[str, int], tuple[float, dict]]]:
    """The box's `rdma` block with per-second `rates`, and the (time, counters) per port to keep for the next one."""
    raw = sample.get(F.RDMA)
    if not isinstance(raw, dict):
        return {F.RDMA_PORTS: [], F.RDMA_WHY: PREDATES.format("rdma")}, {}
    at = stamp(sample)
    ports, keep = [], {}
    for p in (raw.get(F.RDMA_PORTS) if isinstance(raw.get(F.RDMA_PORTS), list) else [])[:MAX_PORTS]:
        p = _port(p)
        if p is None:
            continue
        key = (p[F.P_DEV], p[F.P_PORT])
        last = (prev or {}).get(key)
        rates, why = {}, {}
        for name, value in p[F.P_COUNTERS].items():
            if at is None:
                why[name] = "the sample carries no readable sampled_at: no interval to divide by"
            elif last is None or name not in last[1]:
                why[name] = "first sample of this counter: a rate needs two"
            elif at <= last[0]:
                why[name] = "the sample is not newer than the last one: no interval"
            elif value < last[1][name]:
                why[name] = f"counter went backwards ({last[1][name]} → {value}): reset, no rate this sample"
            else:
                rates[name] = round((value - last[1][name]) / (at - last[0]), 3)
        p[F.P_RATES], p["rates_why"] = rates, why
        ports.append(p)
        if at is not None:
            keep[key] = (at, dict(p[F.P_COUNTERS]))
    out: dict[str, Any] = {F.RDMA_PORTS: ports, F.RDMA_WHY: _text(raw.get(F.RDMA_WHY))}
    return out, keep


def kernel_view(sample: Mapping[str, Any]) -> dict:
    raw = sample.get(F.KERNEL)
    if not isinstance(raw, dict):
        return {F.K_XID_TOTAL: None, F.K_OOM_TOTAL: None, F.K_LAST_XID: None, F.K_XID_SINCE: None,
                F.K_XID_COMPLETE: None, F.K_WHY: PREDATES.format("kernel")}
    last = raw.get(F.K_LAST_XID)
    last = ({"code": last["code"], "at": _num(last.get("at"))}
            if isinstance(last, dict) and _count(last.get("code")) is not None else None)
    complete = raw.get(F.K_XID_COMPLETE)
    return {F.K_XID_TOTAL: _count(raw.get(F.K_XID_TOTAL)), F.K_OOM_TOTAL: _count(raw.get(F.K_OOM_TOTAL)),
            F.K_LAST_XID: last, F.K_XID_SINCE: _num(raw.get(F.K_XID_SINCE)),
            F.K_XID_COMPLETE: complete if isinstance(complete, bool) else None, F.K_WHY: _text(raw.get(F.K_WHY))}


def disks_view(sample: Mapping[str, Any]) -> tuple[list[dict], str | None]:
    raw = sample.get(F.DISKS)
    if not isinstance(raw, list):
        return [], PREDATES.format("disks")
    out = []
    for d in raw[:MAX_DISKS]:
        if not isinstance(d, dict) or not _text(d.get(F.D_PATH)):
            continue
        row = {F.D_PATH: _text(d[F.D_PATH]), **{k: _count(d.get(k)) for k in (F.D_TOTAL, F.D_USED, F.D_FREE)}}
        if _text(d.get(F.D_WHY)):
            row[F.D_WHY] = _text(d[F.D_WHY])
        out.append(row)
    return out, _text(sample.get(F.DISKS_WHY), 600)


def models_view(sample: Mapping[str, Any], folders: Mapping[str, list[str]], names_why: str | None
                ) -> tuple[list[dict], str | None]:
    raw = sample.get(F.MODELS)
    if not isinstance(raw, list):
        return [], PREDATES.format("models")
    out = []
    for r in raw[:MAX_ROOTS]:
        if not isinstance(r, dict) or not _text(r.get(F.M_ROOT)):
            continue
        dirs = r.get(F.M_DIRS) if isinstance(r.get(F.M_DIRS), dict) else {}
        sized = sorted(((str(k)[:128], b) for k, v in dirs.items() if (b := _count(v)) is not None),
                       key=lambda kv: -kv[1])[:MAX_FOLDERS]
        unreadable = r.get(F.M_UNREADABLE) if isinstance(r.get(F.M_UNREADABLE), list) else []
        row = {F.M_ROOT: _text(r[F.M_ROOT]),
               "folders": [{"name": n, "bytes": b, "checkpoints": list(folders.get(n, []))} for n, b in sized],
               F.M_MEASURED_AT: _num(r.get(F.M_MEASURED_AT)), F.M_TOOK_S: _num(r.get(F.M_TOOK_S)),
               F.M_MORE_COUNT: _count(r.get(F.M_MORE_COUNT)), F.M_MORE_BYTES: _count(r.get(F.M_MORE_BYTES)),
               F.M_UNREADABLE: [str(u)[:128] for u in unreadable[:MAX_FOLDERS]], F.M_WHY: _text(r.get(F.M_WHY))}
        if names_why:
            row["names_why"] = names_why
        if row[F.M_MORE_COUNT] is None or row[F.M_MORE_BYTES] is None:
            row["more_why"] = "the sender sent no readable more_count / more_bytes: the folders past the named ones " \
                              "are not counted"
        out.append(row)
    return out, _text(sample.get(F.MODELS_WHY))


def checkpoint_folders(checkpoints: Mapping[str, Any]) -> dict[str, list[str]]:
    """Top-level folder → the checkpoints the fleet file puts under it (`id`, `id (uncensored)`, `id (draft)`)."""
    out: dict[str, list[str]] = {}
    for mid, ck in checkpoints.items():
        for path, label in ((ck.path, mid), (ck.uncensored, f"{mid} (uncensored)"), (ck.draft, f"{mid} (draft)")):
            if path:
                out.setdefault(path.strip("/").split("/", 1)[0], []).append(label)
    return out


def deployment_folders(env: Mapping[str, str] | None = None) -> tuple[dict[str, list[str]], str | None]:
    """THIS LB's checkpoint folders from its fleet file (catalog.current_deployment), or ({}, why none are named)."""
    from vllm_lb.catalog import current_deployment
    try:
        dep = current_deployment(env)
    except (OSError, ValueError) as exc:
        return {}, f"the fleet file's checkpoints could not be read: {exc}"[:MAX_TEXT]
    if not dep.checkpoints:
        return {}, f"{dep.source}: no checkpoints named"
    return checkpoint_folders(dep.checkpoints), None
