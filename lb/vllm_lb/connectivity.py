"""THE PER-BOX CONNECTIVITY CHECK — GET /lb/check: for every box the fleet file names, does each channel the gateway
needs to it work right now, and if not, what to do.

WHY ONE PLACE. The answers already exist, scattered: the engine probe (probe.py) every tick, SSH to a pair's ranks
inside the cluster loop's tp-rank calls (cluster_exec.py), the hardware sample's age (hw.py). When something is wrong
the operator has to know which of the three to look at and where it is shown. This module asks all of them, for one
box at a time, and lays the answers side by side with a one-line hint each. It REPORTS; it repairs nothing.

ON DEMAND ONLY. Nothing here runs in the control tick: an SSH round-trip per rank box and an extra probe per engine
are cheap once, not every second. The panel's "check" button calls it; the rows are computed when asked and never
cached into /lb/status.

THE ROWS (`cap`):
  telemetry  the box's last hardware sample (hw.py): none heard — the sender is not running, or its hostname is not
             the fleet's `boxes.<box>.hostname`; stale — the sender went quiet; fresh — its age.
  engine     one per backend on the box: the probe the tick runs (probe.probe_backend), its own verdict and words.
  ssh        one per distinct SSH address of the box — its fleet-file `ssh` (what the cluster control dials;
             fleet-power dials its own POWER_BOXES, not this) and each ssh-controlled cluster rank's target: `SshExecutor.reach` — `ssh … true`, read-only, short
             timeout. A box with no ssh address has no row, and `ssh_why` says so in one line.
A row is {cap, target, ok, detail, hint_short, hint, ms}; both hints are None on a pass. A box is ok when every row
passed.

TWO LENGTHS OF HINT. `hint_short` is what is wrong, as a figure the panel prints on the row ("http://e:8001
unreachable from the gateway", at most 60 characters); `hint` is the whole recovery — which command, which env name —
that the row's title carries. One sentence on every failed row read as prose (the UX audit); the full words are
still one hover / long-press away, never dropped.

Every probe runs in parallel (one thread per row), so the check takes as long as its slowest channel, not their sum.
A probe that raises is a failed row with its exception's words — never a 500 for the whole check.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

#: The SSH reachability timeout: a box that does not answer `true` in this is not reachable for the loop either.
SSH_TIMEOUT_S = 8.0


def plan(cfg, backends) -> list[dict]:
    """What to check, per fleet box in the fleet's order: the backends on it (`backends`: the hub snapshot's, which
    carry `box`), and every SSH address the gateway's side of the fleet reaches it by — the box's own `ssh` in the
    fleet file (what the cluster control dials — fleet-power dials its own POWER_BOXES, not this) and each ssh-controlled cluster rank's target —
    one row per DISTINCT address, `via` naming who uses it ("fleet · pair"). A box with none has no SSH row and
    `ssh_why` says so."""
    from vllm_lb.cluster import ClusterHost
    fleet = getattr(cfg, "fleet", None)
    if fleet is None:
        return []
    targets: dict[str, dict[str, dict]] = {}          # box → address → {via: [..], host}
    for name, box in fleet.boxes.items():
        if box.ssh:
            targets.setdefault(name, {})[box.ssh] = {"via": ["fleet"],
                                                     "host": ClusterHost(box.label or name, name, box.ssh)}
    for c in getattr(cfg, "clusters", ()) or ():
        if c.control != "ssh":
            continue
        for host in c.hosts:
            if host.ssh_target:
                slot = targets.setdefault(host.box, {}).setdefault(host.ssh_target, {"via": [], "host": host})
                slot["via"].append(c.name)
    out = []
    for name in fleet.boxes:
        ssh = [{"via": " · ".join(t["via"]), "host": t["host"]} for t in targets.get(name, {}).values()]
        out.append({"box": name, "engines": [b for b in backends if getattr(b, "box", None) == name], "ssh": ssh,
                    "ssh_why": None if ssh else f"no ssh address in the fleet file (boxes.{name}.ssh): no SSH row"})
    return out


def _row(cap: str, target: str, ok: bool, detail: str, hint: tuple[str, str] | None, ms: float) -> dict:
    """`hint` is (short, full) on a failure — see TWO LENGTHS OF HINT — and None on a pass."""
    short, full = (None, None) if ok or hint is None else hint
    return {"cap": cap, "target": target, "ok": ok, "detail": detail, "hint_short": short, "hint": full,
            "ms": round(ms, 1)}


def _short(target: str, what: str) -> str:
    """`<target> <what>` in at most 60 characters: a long target loses its middle ("http://a-very-lo…:8001"), never
    the words saying what is wrong — the full target is in the row's `target` and its full hint."""
    room = 60 - len(what) - 1
    if len(target) > room:
        keep = room - 1
        target = target[:keep - keep // 3] + "…" + target[len(target) - keep // 3:]
    return f"{target} {what}"


def _telemetry(box: str, h: dict | None, hw_state: str, max_age_s: float) -> dict:
    if not hw_state.startswith("listening"):
        return _row("telemetry", box, False, f"the hardware listener is {hw_state}",
                    ("LB_HW_PORT not listening",
                     "set LB_HW_PORT (lb/.env) and publish its UDP port, then restart the gateway"), 0.0)
    if not h:
        return _row("telemetry", box, False, "no sample heard from this box",
                    (_short(box, "no hw_sender heard"),
                     f"start lb/hw_sender on {box}, and check its hostname is the fleet's boxes.{box}.hostname"), 0.0)
    age = h.get("age_s")
    if h.get("stale"):
        return _row("telemetry", box, False, f"last sample {age:.0f} s ago (stale past {max_age_s:g} s)",
                    (_short(box, "hw_sender quiet"),
                     f"the sender on {box} went quiet: `docker compose ps` in its lb/hw_sender; UDP to the gateway"), 0.0)
    return _row("telemetry", box, True, f"sample {age:.1f} s old", None, 0.0)


def _engine(b, probe: Callable[[Any], dict]) -> dict:
    t0 = time.monotonic()
    try:
        f = probe(b) or {}
    except Exception as exc:                    # a probe that raises is a failed row, not a failed check
        f = {"live": False, "last_error": f"{type(exc).__name__}: {exc}"[:200]}
    ms = (time.monotonic() - t0) * 1000
    if f.get("live"):
        models = list(f.get("served_models") or ())
        note = f" · {f['last_error']}" if f.get("last_error") else ""
        return _row("engine", b.url, True, f"live · {len(models)} model{'s' if len(models) != 1 else ''}{note}",
                    None, ms)
    return _row("engine", b.url, False, f.get("last_error") or "not live",
                (_short(b.url, "unreachable from the gateway"),
                 f"engine on {b.box}: `docker logs` its container; {b.url} must be reachable from the gateway"), ms)


def _ssh(item: dict, reach: Callable[[Any], tuple[bool, str]]) -> dict:
    host = item["host"]
    t0 = time.monotonic()
    try:
        ok, detail = reach(host)
    except Exception as exc:
        ok, detail = False, f"{type(exc).__name__}: {exc}"[:200]
    return _row("ssh", f"{host.ssh_target} ({item['via']})", ok, detail or ("ssh ok" if ok else "ssh failed"),
                (_short(host.ssh_target, "unreachable over ssh"),
                 f"the gateway's key (LB_CLUSTER_SSH_KEY) in {host.ssh_target}'s authorized_keys, its host key in "
                 f"LB_CLUSTER_KNOWN_HOSTS; try `ssh {host.ssh_target} true` from the gateway"),
                (time.monotonic() - t0) * 1000)


def check(plans: list[dict], *, hardware: dict, hw_state: str, max_age_s: float, now: float,
          probe: Callable[[Any], dict], reach: Callable[[Any], tuple[bool, str]]) -> dict:
    """Run every row of `plans` in parallel; the answer per box, in plan order."""
    jobs: list[tuple[int, Callable[[], dict]]] = []
    for i, p in enumerate(plans):
        for b in p["engines"]:
            jobs.append((i, lambda b=b: _engine(b, probe)))
        for item in p["ssh"]:
            jobs.append((i, lambda item=item: _ssh(item, reach)))
    rows: dict[int, list[dict]] = {i: [_telemetry(p["box"], hardware.get(p["box"]), hw_state, max_age_s)]
                                   for i, p in enumerate(plans)}
    if jobs:
        with ThreadPoolExecutor(max_workers=min(16, len(jobs)), thread_name_prefix="lb-check") as pool:
            futures = [(i, pool.submit(fn)) for i, fn in jobs]
            for i, fut in futures:
                rows[i].append(fut.result())
    boxes = [{"box": p["box"], "ok": all(r["ok"] for r in rows[i]), "checks": rows[i], "ssh_why": p.get("ssh_why")}
             for i, p in enumerate(plans)]
    every = [r for b in boxes for r in b["checks"]]
    return {"at": now, "boxes": boxes, "passed": sum(1 for r in every if r["ok"]),
            "failed": sum(1 for r in every if not r["ok"])}
