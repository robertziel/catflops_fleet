"""The power API. `PowerApp.handle` is a pure request → response function; `server.py` puts HTTP
around it, so every rule below is tested without a socket.

The door (every request, before anything else):
  * `Host` must be a name this panel is reached by — the listener's loopback names at its port, or one
    POWER_ALLOWED_HOSTS names (host_guard.py: the DNS-rebinding guard, shared with the gateway); else 421,
    audited on a write route. A write's `Origin` is judged by POWER_ORIGINS below, exactly.
  * `X-Power-Gate` must equal the secret that ONLY the edge's nginx `location /power/` sets. The
    service listens on 127.0.0.1 on box-c, where host-network containers can also reach it;
    without the secret none of them can use it. An empty configured secret refuses everything.
  * `X-Power-User` must be non-empty: nginx sets it from `$remote_user`, so it is the
    basic-auth user who clicked. It is what the audit log records.
State changes: POST only, `Content-Type: application/json` (else 415), `Origin` in the allowed
set and `Sec-Fetch-Site` (if sent) same-origin (else 403), bodies ≤ 4 KiB.
Two steps: `confirm` hands out a short-lived single-use token bound to (user, box, action);
`execute` needs that token AND, for a power action, a 5 s press-and-hold: the client reports
`held_ms` ≥ 5000, and the SERVER refuses a token younger than 5 s — the wait is enforced here, so
a script cannot skip it. Like the typed box name it replaced, the hold is an accident guard, not
an authentication. Powering off a GB10 box while the cluster runs, or while its state is unknown,
also needs `ack_cluster: true`.
"""
from __future__ import annotations

import datetime
import hmac
import json
import math
import pathlib
import secrets
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

from fleet_power.boxes import Box
from fleet_power.host_guard import Authority, HostGuard
from fleet_power.wol import WolError

STATIC_DIR = pathlib.Path(__file__).resolve().parent / "static"
CLOCK_MIN_MHZ = 300
# The caps the panel can apply, one ssh key each: the MHz live in that key's forced command on the
# box (host-setup.sh), so the action IS the key and no number a client sends can reach root.
# Adding a cap here means adding its key + forced command + sudoers line on the box as well.
CLOCK_CAPS_MHZ = (2200, 2300)
GPU_MAX_MHZ = 3003                              # the GB10's own maximum, for the consequence text
TOKEN_TTL_S = 120
# A power action is a 5 s press-and-hold. The client reports how long it held (`held_ms`), and the
# server itself refuses a token younger than the hold: the token is issued BEFORE the hold starts
# (the dialog/sheet asks for it to show the consequences), so an honest hold always clears this.
HOLD_S = 5.0
HOLD_MS = int(HOLD_S * 1000)
MAX_BODY = 4096
PER_BOX_WINDOW_S = 60          # one executed power action per box per minute
CLOCK_WINDOW_S = 10            # a clock is a knob, not a cliff: enough to stop a double click
CONFIRM_MAX, CONFIRM_WINDOW_S = 10, 60
WAKE_TIMEOUT_S = 600
POWEROFF_PENDING_S = 600
SSH_TIMEOUT_S = 25
CLOCK_RECONCILE_S = 10         # how often the configured clock is checked for (and re-applied after a reboot)

BASE_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Cache-Control": "no-store",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'",
}
# The button script is loaded by the gateway page (same origin), not rendered as a document.
JS_HEADERS = {**BASE_HEADERS, "Content-Type": "application/javascript; charset=utf-8"}

# ssh's own words for "the box is going down under us" vs "we never got in".
_SSH_NEVER_GOT_IN = ("Permission denied", "Host key verification failed", "Connection refused",
                     "Connection timed out", "No route to host", "Could not resolve",
                     "Network is unreachable", "REMOTE HOST IDENTIFICATION HAS CHANGED",
                     "Too many authentication failures", "Connection reset")

CLOCK_NOTE = ("The driver reports no lock state, so this is what this panel last applied since the "
              "service started; an nvidia-smi typed on the box itself does not show up here.")
# Said once for every clock action, instead of three times over.
CLOCK_APPLIES_NOTE = ("A clock change applies in place: the cluster keeps serving, with no restart "
                      "and no dropped request. A benchmark running right now would measure the new "
                      "clock. A lock does not survive a reboot: whenever the box comes back (or "
                      "this service restarts) the service applies its configured clock again, so a "
                      "change made here holds only until then.")


@dataclass(frozen=True)
class ActionSpec:
    """One button. `key_attr` names the Settings field holding the ssh key whose FORCED command is
    this action and nothing else — the MHz (2200, 2300) live in the key's forced command on the
    box, so no number a client sends can ever reach root. `limit_mhz` is what this action caps the
    GPU at, for the panel to show; None for a release or a power action."""
    kind: str            # "power" | "clock": which rate-limit bucket, and which runner
    needs_hold: bool     # a 5 s press-and-hold to confirm — for the actions you cannot take back
    key_attr: str        # the Settings field naming this action's own key
    sent: str            # the word handed to ssh; the box ignores it and runs its forced command
    title: str
    consequence: str
    limit_mhz: int | None = None


def clock_action(mhz: int) -> str:
    return f"clock_limit_{mhz}"


def clock_key_attr(mhz: int) -> str:
    """The first cap keeps the original field (and key file) name: that key is already on the
    boxes, and a rename here would silently stop capping at all."""
    return "clock_limit_key_path" if mhz == CLOCK_CAPS_MHZ[0] else f"clock_limit_{mhz}_key_path"


def _cap_consequence(mhz: int) -> str:
    lowest = mhz == min(CLOCK_CAPS_MHZ)
    rest = ("Less heat and less power, and the model gets slower in proportion."
            if lowest else
            f"{mhz - min(CLOCK_CAPS_MHZ)} MHz above the {min(CLOCK_CAPS_MHZ)} MHz cap: faster than "
            "it, hotter than it, and still short of the GPU's own maximum.")
    return (f"Caps this box's GPU at {CLOCK_MIN_MHZ}-{mhz} MHz (its own maximum is "
            f"{GPU_MAX_MHZ}). {rest}")


ACTION_SPECS: dict[str, ActionSpec] = {
    "poweroff": ActionSpec("power", True, "key_path", "poweroff", "Power off", ""),
    "poweron": ActionSpec("power", True, "key_path", "poweroff", "Power on", ""),
    **{clock_action(mhz): ActionSpec(
        "clock", False, clock_key_attr(mhz), f"clock-limit-{mhz}",
        f"Limit the GPU to {mhz} MHz", _cap_consequence(mhz), limit_mhz=mhz)
       for mhz in CLOCK_CAPS_MHZ},
    "clock_release": ActionSpec(
        "clock", False, "clock_release_key_path", "clock-release",
        "Release the clock limit",
        "Hands the GPU clock back to the driver (nvidia-smi -rgc): full speed, more heat and more "
        "power."),
}
ACTIONS = tuple(ACTION_SPECS)


@dataclass
class Settings:
    origins: tuple[str, ...]
    gate_secret: str
    key_path: str
    dry_key_path: str
    known_hosts: str
    dry_run: bool
    audit_path: str
    # One key per clock action: its forced command on the box IS the action (see host-setup.sh).
    clock_limit_key_path: str = "/keys/fleet-clock-limit"              # the first cap (2200 MHz)
    clock_limit_2300_key_path: str = "/keys/fleet-clock-limit-2300"
    clock_release_key_path: str = "/keys/fleet-clock-release"
    clock_dry_key_path: str = "/keys/fleet-clock-dryrun"
    #: THE HOST ALLOWLIST (host_guard.py): the names beyond the listener's loopback names this panel answers to — the
    #: Host the edge's nginx presents through the tunnel. POWER_ALLOWED_HOSTS, REQUIRED (`settings_from_env`); the
    #: class default is the strictest set, loopback only.
    allowed_hosts: tuple[Authority, ...] = ()
    #: The listener, host:port (POWER_BIND): its port is the one the loopback names are allowed at.
    bind: str = "127.0.0.1:8016"


def _held_long_enough(held_ms) -> bool:
    """The client's own report of its hold: a finite JSON number ≥ HOLD_MS (not a bool, a string or
    NaN). It is an attestation only — the token-age check in `execute` is what the server knows."""
    if isinstance(held_ms, bool) or not isinstance(held_ms, (int, float)):
        return False
    if isinstance(held_ms, float) and not math.isfinite(held_ms):    # json.loads takes NaN/Infinity
        return False
    return held_ms >= HOLD_MS         # a huge int compares fine; math.isfinite() would overflow on it


class RateLimit:
    def __init__(self, limit: int, window: float):
        self.limit, self.window, self.hits = limit, window, deque()

    def allow(self, now: float) -> bool:
        while self.hits and now - self.hits[0] >= self.window:
            self.hits.popleft()
        if len(self.hits) >= self.limit:
            return False
        self.hits.append(now)
        return True


class PowerApp:
    def __init__(self, settings: Settings, boxes: dict[str, Box], *, runner: Callable,
                 wol: Callable, probe: Callable, cluster_state: Callable,
                 clock: Callable[[], float] = time.time):
        self.settings, self.boxes = settings, boxes
        self.runner, self.wol, self.probe = runner, wol, probe
        self.cluster_state, self.clock = cluster_state, clock
        self.lock = threading.Lock()
        self.tokens: dict[str, dict] = {}
        self.last_exec: dict[str, float] = {}
        self.confirm_rl = RateLimit(CONFIRM_MAX, CONFIRM_WINDOW_S)
        self.pending: dict[str, dict] = {}
        self.clock_applied: dict[str, dict] = {}     # box -> what the last clock click set
        # box -> its clock was set (by config or by hand) since this service last saw it come up
        self.clock_settled: dict[str, bool] = {}
        host, port = settings.bind.rsplit(":", 1)
        self.listening_on(host, int(port))

    def listening_on(self, host: str, port: int) -> None:
        """THE HOST ALLOWLIST for the address the server is really bound to (`__main__` calls it once it listens):
        the loopback names at that port plus settings.allowed_hosts. Each distinct Host is printed once."""
        self.host_guard = HostGuard("POWER_ALLOWED_HOSTS", self.settings.allowed_hosts, host.strip("[]"), port,
                                    lambda line: print(line, flush=True))

    # ── plumbing ───────────────────────────────────────────────────────────────────────────

    def _json(self, status: int, obj: dict, extra: dict | None = None):
        h = {**BASE_HEADERS, "Content-Type": "application/json"}
        return status, h, json.dumps(obj).encode()

    def _err(self, status: int, typ: str, message: str, **more):
        return self._json(status, {"error": {"type": typ, "message": message}, **more})

    def audit(self, event: str, ctx: dict, result: str, **fields):
        rec = {
            "ts": datetime.datetime.fromtimestamp(self.clock(), datetime.timezone.utc)
                  .strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "event": event, "user": ctx.get("user"), "ip": ctx.get("ip"),
            "box": fields.pop("box", None), "action": fields.pop("action", None),
            "result": result, "dry_run": self.settings.dry_run, **fields,
        }
        line = json.dumps(rec, ensure_ascii=True)       # one physical line, whatever the input
        with self.lock:
            with open(self.settings.audit_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
        print("audit", line, flush=True)

    @staticmethod
    def _h(headers: dict, name: str) -> str | None:
        for k, v in headers.items():
            if k.lower() == name.lower():
                return v
        return None

    # ── entry point ────────────────────────────────────────────────────────────────────────

    def handle(self, method: str, path: str, headers: dict, body: bytes):
        path = path.split("?", 1)[0]
        ctx = {"user": self._h(headers, "X-Power-User") or None,
               "ip": self._h(headers, "X-Real-IP")}
        # THE HOST ALLOWLIST FIRST (host_guard.py): a DNS-rebound page names itself in Host — answered here, before
        # the gate is asked, and audited like every other refusal of a write route. The Origin of a write stays
        # `_check_post`'s exact POWER_ORIGINS rule: one rule per value.
        refused = self.host_guard.host_refusal(self._h(headers, "Host"))
        if refused is not None:
            if method == "POST" and path in ("/power/api/confirm", "/power/api/execute"):
                self.audit(path.rsplit("/", 1)[1], ctx, "refused:misdirected_host")
            return self._json(*refused)
        gate = self._h(headers, "X-Power-Gate") or ""
        secret = self.settings.gate_secret or ""
        if not secret or not hmac.compare_digest(gate.encode(), secret.encode()):
            return self._err(403, "forbidden", "not through the gateway")
        if not ctx["user"]:
            return self._err(403, "forbidden", "no authenticated user")

        if path == "/power/button.js":
            if method not in ("GET", "HEAD"):
                return self._err(405, "method_not_allowed", "GET only")
            return 200, dict(JS_HEADERS), (STATIC_DIR / "button.js").read_bytes()
        if path == "/power/api/state":
            if method not in ("GET", "HEAD"):
                return self._err(405, "method_not_allowed", "GET only")
            return self._json(200, self.state())
        if path in ("/power/api/confirm", "/power/api/execute"):
            event = path.rsplit("/", 1)[1]
            if method != "POST":
                return self._err(405, "method_not_allowed", "POST only")
            refusal = self._check_post(headers, body)
            if refusal:
                status, typ, msg = refusal
                self.audit(event, ctx, f"refused:{typ}")
                return self._err(status, typ, msg)
            try:
                data = json.loads(body)
            except ValueError:
                data = None
            if not isinstance(data, dict):
                self.audit(event, ctx, "refused:bad_body")
                return self._err(400, "bad_body", "a JSON object is required")
            return (self.confirm if event == "confirm" else self.execute)(data, ctx)
        return self._err(404, "not_found", "no such endpoint")

    def _check_post(self, headers: dict, body: bytes):
        ctype = (self._h(headers, "Content-Type") or "").split(";", 1)[0].strip().lower()
        if ctype != "application/json":
            return 415, "content_type", "Content-Type: application/json is required"
        origin = self._h(headers, "Origin")
        if not origin or origin not in self.settings.origins:
            return 403, "origin", "cross-origin request refused"
        site = self._h(headers, "Sec-Fetch-Site")
        if site is not None and site != "same-origin":
            return 403, "origin", "cross-site request refused"
        if len(body) > MAX_BODY:
            return 413, "too_large", "body too large"
        return None

    # ── confirm ────────────────────────────────────────────────────────────────────────────

    def _cluster(self) -> dict:
        try:
            c = self.cluster_state() or {}
        except Exception as e:   # noqa: BLE001 — an unreadable state is "unknown", never "off"
            c = {"ok": False, "state": None, "boxes": [], "error": str(e)}
        return {"ok": bool(c.get("ok")), "state": c.get("state"),
                "boxes": list(c.get("boxes") or [])}

    @staticmethod
    def _cluster_blocks(box: Box, action: str, cluster: dict) -> bool:
        """Does powering this box off drop a running (or maybe-running) cluster?"""
        if action != "poweroff" or not box.in_cluster:
            return False
        if not cluster["ok"] or cluster["state"] is None:
            return True
        return cluster["state"] not in ("off", "failed", "stopped")

    def confirm(self, data: dict, ctx: dict):
        now = self.clock()
        name, action = data.get("box"), data.get("action")
        box = self.boxes.get(name) if isinstance(name, str) else None
        if box is None:
            self.audit("confirm", ctx, "refused:unknown_box", box=str(name)[:40])
            return self._err(404, "unknown_box", "no such box")
        if action not in ACTIONS:
            self.audit("confirm", ctx, "refused:unknown_action", box=box.name,
                       action=str(action)[:40])
            return self._err(404, "unknown_action", "no such action")
        if action == "poweron" and box.wol is None:
            self.audit("confirm", ctx, "refused:no_power_on", box=box.name, action=action)
            return self._err(404, "no_power_on", box.power_on_note)
        spec = ACTION_SPECS[action]
        if spec.kind == "clock" and not box.gpu_clock:
            self.audit("confirm", ctx, "refused:no_gpu_clock", box=box.name, action=action)
            return self._err(404, "no_gpu_clock", f"{box.name} has no GPU clock this service may set")
        with self.lock:
            allowed = self.confirm_rl.allow(now)
        if not allowed:
            self.audit("confirm", ctx, "refused:rate_limited", box=box.name, action=action)
            return self._err(429, "rate_limited", "too many requests; wait a minute")
        cluster = self._cluster()
        token = secrets.token_urlsafe(24)
        with self.lock:
            self.tokens = {t: v for t, v in self.tokens.items() if v["exp"] > now}
            self.tokens[token] = {"user": ctx["user"], "box": box.name, "action": action,
                                  "issued": now, "exp": now + TOKEN_TTL_S}
        if spec.kind == "clock":
            consequence = spec.consequence
        elif action == "poweroff":
            consequence = box.consequence
        else:
            consequence = box.power_on_note
        self.audit("confirm", ctx, "issued", box=box.name, action=action)
        return self._json(200, {
            "token": token, "box": box.name, "action": action,
            "needs_hold": spec.needs_hold, "hold_ms": HOLD_MS if spec.needs_hold else None,
            "title": spec.title, "kind": spec.kind,
            "expires_in": TOKEN_TTL_S, "consequence": consequence,
            "power_on_note": box.power_on_note, "clock": self._clock_state(box),
            "cluster": cluster, "needs_cluster_ack": self._cluster_blocks(box, action, cluster),
            "dry_run": self.settings.dry_run,
        })

    # ── execute ────────────────────────────────────────────────────────────────────────────

    def execute(self, data: dict, ctx: dict):
        now = self.clock()
        token = data.get("token")
        if not isinstance(token, str) or not token:
            self.audit("execute", ctx, "refused:no_token")
            return self._err(400, "no_token", "a confirm token is required")
        with self.lock:
            t = self.tokens.get(token)
            if t is not None and (t["exp"] <= now or t["user"] != ctx["user"]):
                t = None
        if t is None:
            self.audit("execute", ctx, "refused:token")
            return self._err(409, "confirm_expired",
                             "this confirmation is used, expired or not yours; start again")
        box, action = self.boxes[t["box"]], t["action"]
        spec = ACTION_SPECS[action]
        if spec.needs_hold:
            # Neither refusal burns the token: an early release, or a client that raced ahead,
            # can simply hold again with the same confirmation.
            if not _held_long_enough(data.get("held_ms")):
                self.audit("execute", ctx, "refused:hold_too_short", box=box.name, action=action)
                return self._err(400, "hold_too_short",
                                 f"press and hold for {HOLD_S:g} s to confirm", hold_ms=HOLD_MS)
            age = now - t["issued"]
            if age < HOLD_S:           # the part a script cannot fake: the wait happens HERE
                self.audit("execute", ctx, "refused:too_soon", box=box.name, action=action)
                shown = math.floor(age * 10) / 10          # 4.97 s is "4.9 s old", never "5.0"
                return self._err(400, "too_soon",
                                 f"this confirmation is {shown:.1f} s old; a {HOLD_S:g} s hold "
                                 "cannot have finished yet", hold_ms=HOLD_MS)
        cluster = self._cluster()
        if self._cluster_blocks(box, action, cluster) and data.get("ack_cluster") is not True:
            self.audit("execute", ctx, "refused:cluster_running", box=box.name, action=action,
                       cluster=cluster["state"])
            return self._err(409, "cluster_running",
                             "the model cluster is running (or its state is unknown); stop it "
                             "in the panel first, or confirm again acknowledging it will drop",
                             cluster=cluster)
        bucket = f"{box.name}:{spec.kind}"
        window = CLOCK_WINDOW_S if spec.kind == "clock" else PER_BOX_WINDOW_S
        with self.lock:
            last = self.last_exec.get(bucket)
            if last is not None and now - last < window:
                limited = True
            else:
                limited = False
                self.last_exec[bucket] = now
                self.tokens.pop(token, None)             # single use: gone before we act
        if limited:
            self.audit("execute", ctx, "refused:rate_limited", box=box.name, action=action)
            return self._err(429, "rate_limited",
                             f"{box.name} was just acted on; wait {int(window)}s")

        if spec.kind == "clock":
            ok, output, rc = self._clock(box, action)
        elif action == "poweroff":
            ok, output, rc = self._poweroff(box)
        else:
            ok, output, rc = self._poweron(box)
        result = "sent" if ok else "failed"
        if ok and not self.settings.dry_run:
            with self.lock:
                if spec.kind == "clock":
                    self._record_clock(box, spec, now, ctx["user"])
                else:
                    self.pending[box.name] = {"action": action, "since": now}
        self.audit("execute", ctx, result, box=box.name, action=action, rc=rc,
                   output=output[-500:], cluster=cluster["state"])
        return self._json(200 if ok else 502, {
            "result": result, "box": box.name, "action": action, "output": output[-2000:],
            "clock": self._clock_state(box), "dry_run": self.settings.dry_run,
        })

    def _record_clock(self, box: Box, spec: ActionSpec, now: float, by: str) -> None:
        """Caller holds self.lock."""
        self.clock_applied[box.name] = {"state": "limited" if spec.limit_mhz else "released",
                                        "mhz": spec.limit_mhz, "at": now, "by": by}

    # ── the configured clock ───────────────────────────────────────────────────────────────

    def reconcile_clocks(self) -> None:
        """One pass of the loop that keeps every GPU box at its configured clock:
        `default_gpu_clock_mhz`, or released when that is absent. The driver reports no lock state,
        so the trigger is the box's uptime as this service sees it (the sshd probe): a box seen down
        — a lock dies with a reboot or a power cut — or not yet handled since this service started
        gets its configured clock as soon as its sshd answers, and a failed apply (the driver may not
        be up yet) is retried every pass. A manual clock click is left alone while the box stays up: it
        holds until the box is next seen down or this service restarts, then the config wins again."""
        gpu = [b for b in self.boxes.values() if b.gpu_clock]
        up = self._probe_all(gpu)
        for box in gpu:
            with self.lock:
                if not up.get(box.name, False):
                    self.clock_settled[box.name] = False
                    continue
                if self.clock_settled.get(box.name):
                    continue
            mhz = box.default_gpu_clock_mhz
            action = clock_action(mhz) if mhz else "clock_release"
            ok, output, rc = self._clock(box, action)
            with self.lock:
                self.clock_settled[box.name] = ok
                if ok and not self.settings.dry_run:
                    self._record_clock(box, ACTION_SPECS[action], self.clock(), "config")
            self.audit("reconcile", {"user": "config", "ip": None}, "sent" if ok else "failed",
                       box=box.name, action=action, rc=rc, output=output[-500:])

    def _key_for(self, action: str) -> str:
        """Each action has its own key because each key's forced command on the box is that one
        action. DRY RUN swaps in a key whose forced command only ASKS sudo whether it would be
        allowed, so a dry run changes nothing anywhere."""
        spec = ACTION_SPECS[action]
        key = self.settings.dry_key_path if self.settings.dry_run else self.settings.key_path
        if spec.kind == "clock":
            key = (self.settings.clock_dry_key_path if self.settings.dry_run
                   else getattr(self.settings, spec.key_attr))
        return key

    def ssh_argv(self, box: Box, action: str = "poweroff") -> list[str]:
        key = self._key_for(action)
        return ["ssh", "-i", key, "-F", "none",
                "-o", "IdentitiesOnly=yes", "-o", "IdentityAgent=none", "-o", "BatchMode=yes",
                "-o", "StrictHostKeyChecking=yes",
                "-o", f"UserKnownHostsFile={self.settings.known_hosts}",
                "-o", "ForwardAgent=no", "-o", "ClearAllForwardings=yes",
                "-o", "ConnectTimeout=5", "-o", "ServerAliveInterval=3",
                "-o", "ServerAliveCountMax=2", "-T",
                "-p", str(box.ssh_port), f"{box.ssh_user}@{box.ssh_host}",
                ACTION_SPECS[action].sent]

    def _poweroff(self, box: Box):
        try:
            rc, out, err = self.runner(self.ssh_argv(box), SSH_TIMEOUT_S)
        except Exception as e:  # noqa: BLE001 — a timeout or a missing binary is a failure
            return False, f"ssh did not complete: {e}", None
        output = (out or "") + (err or "")
        if rc == 0:
            return True, output, rc
        if rc == 255 and not any(s in output for s in _SSH_NEVER_GOT_IN):
            return True, output or "connection dropped (the box is going down)", rc
        return False, output or f"ssh exited {rc}", rc

    def _clock(self, box: Box, action: str):
        """A clock change is not a poweroff: the box stays up, so rc 0 is the only success. A
        dropped connection means the command did NOT land, and the panel says so."""
        try:
            rc, out, err = self.runner(self.ssh_argv(box, action), SSH_TIMEOUT_S)
        except Exception as e:  # noqa: BLE001 — a timeout or a missing binary is a failure
            return False, f"ssh did not complete: {e}", None
        output = (out or "") + (err or "")
        landed = rc == 0        # and nothing else: a drop here means the command did not run
        if landed:
            return True, output or "done", rc
        return False, output or f"ssh exited {rc}", rc

    def _poweron(self, box: Box):
        try:
            return True, self.wol(box, send=not self.settings.dry_run), 0
        except (WolError, OSError) as e:
            return False, str(e), None

    # ── state ──────────────────────────────────────────────────────────────────────────────

    def state(self) -> dict:
        now = self.clock()
        up = self._probe_all()
        out = {}
        for name, box in self.boxes.items():
            reachable = up.get(name, False)
            with self.lock:
                p = self.pending.get(name)
            phase = "on" if reachable else "off"
            if p:
                age = now - p["since"]
                if p["action"] == "poweroff":
                    if reachable and age < POWEROFF_PENDING_S:
                        phase = "powering_off"
                    else:
                        with self.lock:
                            self.pending.pop(name, None)
                else:
                    if reachable:
                        with self.lock:
                            self.pending.pop(name, None)
                    else:
                        phase = "waking" if age < WAKE_TIMEOUT_S else "wake_timeout"
            out[name] = {"reachable": reachable, "phase": phase,
                         "can_power_on": box.wol is not None,
                         "power_on_note": box.power_on_note, "consequence": box.consequence,
                         "in_cluster": box.in_cluster, "clock": self._clock_state(box)}
        return {"boxes": out, "dry_run": self.settings.dry_run, "now": now}

    def _clock_state(self, box: Box) -> dict | None:
        """None for a box with no GPU clock to set. `applied` is what the last click here set —
        never a read-back: the driver exposes no lock-state field (CLOCK_NOTE)."""
        if not box.gpu_clock:
            return None
        with self.lock:
            a = self.clock_applied.get(box.name)
        return {"limits_mhz": list(CLOCK_CAPS_MHZ), "min_mhz": CLOCK_MIN_MHZ,
                "configured_mhz": box.default_gpu_clock_mhz,
                "applied": a["state"] if a else None,
                "applied_mhz": a["mhz"] if a else None,
                "applied_at": a["at"] if a else None,
                "applied_by": a["by"] if a else None,
                "actions": {k: {"title": v.title, "consequence": v.consequence,
                                "limit_mhz": v.limit_mhz}
                            for k, v in ACTION_SPECS.items() if v.kind == "clock"},
                "note": CLOCK_NOTE, "applies_note": CLOCK_APPLIES_NOTE}

    def _probe_all(self, boxes=None) -> dict[str, bool]:
        results: dict[str, bool] = {}

        def one(b):
            try:
                results[b.name] = bool(self.probe(b))
            except Exception:  # noqa: BLE001
                results[b.name] = False

        threads = [threading.Thread(target=one, args=(b,))
                   for b in (self.boxes.values() if boxes is None else boxes)]
        for th in threads:
            th.start()
        for th in threads:
            th.join(3)
        return results
