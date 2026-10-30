"""vllm-power: HTTP around `PowerApp`, and the real ssh / Wake-on-LAN / probe / cluster readers.

Configuration (environment):
  POWER_BIND            127.0.0.1:8016 (never a public address; the edge's nginx reaches it via frp)
  POWER_ALLOWED_HOSTS   REQUIRED: the Host names this panel answers to beyond its loopback names (the name the edge's
                        nginx presents through the tunnel), comma-separated host[:port]; `loopback` = none (host_guard.py)
  POWER_ORIGINS         comma-separated allowed Origins: the public origin of the page (empty = refuse all)
  POWER_BOXES           JSON array, one object per machine this service may act on (empty = none; boxes.py)
  POWER_GATE_SECRET_FILE  file holding the secret the edge's nginx sends as X-Power-Gate
  POWER_KEY / POWER_DRY_KEY / POWER_KNOWN_HOSTS   the ssh keys and pinned host keys
  POWER_CLOCK_LIMIT_KEY / POWER_CLOCK_LIMIT_2300_KEY / POWER_CLOCK_RELEASE_KEY /
  POWER_CLOCK_DRY_KEY   one key per clock action — one per cap, so the MHz stay on the box
  POWER_DRY_RUN         1 = use the dry-run key (its forced command powers nothing off)
  POWER_AUDIT           the JSON-lines audit log
  POWER_LB_STATUS       the gateway's read-only status URL (cluster state only; never a control call)
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from fleet_power.app import CLOCK_RECONCILE_S, MAX_BODY, PowerApp, Settings
from fleet_power.boxes import load_boxes
from fleet_power.host_guard import parse_allowed
from fleet_power.wol import send_wol


def run_ssh(argv, timeout):
    p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                       stdin=subprocess.DEVNULL, env={"PATH": "/usr/bin:/bin", "HOME": "/tmp"})
    return p.returncode, p.stdout, p.stderr


def probe(box) -> bool:
    try:
        with socket.create_connection((box.ssh_host, box.ssh_port), timeout=1.0):
            return True
    except OSError:
        return False


def cluster_reader(url: str):
    def read():
        with urllib.request.urlopen(url, timeout=3) as r:
            s = json.load(r)
        c = s.get("cluster") or {}
        return {"ok": True, "state": c.get("state") if c.get("configured") else "off",
                "boxes": sorted((c.get("boxes") or {}).keys())}
    return read


def settings_from_env(env=None) -> Settings:
    """The settings from the environment (or `env`, for the tests). Every default REFUSES: no origin
    (so no request passes), no gate secret, dry run. POWER_ALLOWED_HOSTS has no default at all: absent or blank, this
    raises ValueError naming it and the service does not start."""
    e = os.environ if env is None else env
    secret_file = e.get("POWER_GATE_SECRET_FILE", "/run/secrets/gate")
    try:
        with open(secret_file) as f:
            secret = f.read().strip()
    except OSError:
        secret = ""              # fail closed: every request is refused
    return Settings(
        origins=tuple(o.strip() for o in e.get("POWER_ORIGINS", "").split(",") if o.strip()),
        gate_secret=secret,
        key_path=e.get("POWER_KEY", "/keys/fleet-power"),
        dry_key_path=e.get("POWER_DRY_KEY", "/keys/fleet-power-dryrun"),
        known_hosts=e.get("POWER_KNOWN_HOSTS", "/keys/known_hosts"),
        clock_limit_key_path=e.get("POWER_CLOCK_LIMIT_KEY", "/keys/fleet-clock-limit"),
        clock_limit_2300_key_path=e.get("POWER_CLOCK_LIMIT_2300_KEY", "/keys/fleet-clock-limit-2300"),
        clock_release_key_path=e.get("POWER_CLOCK_RELEASE_KEY", "/keys/fleet-clock-release"),
        clock_dry_key_path=e.get("POWER_CLOCK_DRY_KEY", "/keys/fleet-clock-dryrun"),
        dry_run=e.get("POWER_DRY_RUN", "1") != "0",
        audit_path=e.get("POWER_AUDIT", "/audit/audit.jsonl"),
        allowed_hosts=parse_allowed("POWER_ALLOWED_HOSTS", e.get("POWER_ALLOWED_HOSTS")),
        bind=e.get("POWER_BIND", Settings.bind),
    )


def reconcile_forever(app: PowerApp):
    """Keep every GPU box at its configured clock (PowerApp.reconcile_clocks), for the life of the
    service. A pass that raises is printed and the next pass runs anyway."""
    while True:
        try:
            app.reconcile_clocks()
        except Exception as e:  # noqa: BLE001
            print(f"clock reconcile pass failed: {e!r}", flush=True)
        time.sleep(CLOCK_RECONCILE_S)


def make_handler(app: PowerApp):
    class Handler(BaseHTTPRequestHandler):
        server_version = "vllm-power"
        sys_version = ""

        def _serve(self):
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                n = -1
            if n < 0:
                status, h, body = app._err(400, "bad_length", "bad Content-Length")
                self.close_connection = True
            elif n > MAX_BODY:
                status, h, body = app._err(413, "too_large", "body too large")
            else:
                raw = self.rfile.read(n) if n > 0 else b""
                status, h, body = app.handle(self.command, self.path, dict(self.headers), raw)
            self.send_response(status)
            for k, v in h.items():
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        do_GET = do_POST = do_HEAD = do_PUT = do_DELETE = do_PATCH = _serve

        def log_message(self, fmt, *args):
            sys.stderr.write("%s %s\n" % (self.log_date_time_string(), fmt % args))

    return Handler


def main():
    settings = settings_from_env()
    boxes = load_boxes(os.environ)                # ValueError at boot if POWER_BOXES is malformed
    app = PowerApp(settings, boxes, runner=run_ssh, wol=send_wol, probe=probe,
                   cluster_state=cluster_reader(os.environ.get(
                       "POWER_LB_STATUS", "http://127.0.0.1:8010/lb/status")))
    host, port = settings.bind.rsplit(":", 1)
    srv = ThreadingHTTPServer((host, int(port)), make_handler(app))
    app.listening_on(host, srv.server_address[1])
    threading.Thread(target=reconcile_forever, args=(app,), daemon=True).start()
    print(f"vllm-power on {host}:{port} dry_run={settings.dry_run} "
          f"gate={'set' if settings.gate_secret else 'MISSING (refusing all)'} "
          f"origins={len(settings.origins)} boxes={len(boxes)} "
          f"hosts={','.join(str(a) for a in app.host_guard.allowed)}"
          + ("" if boxes else " (POWER_BOXES unset: no box may be acted on)")
          + ("" if settings.origins else " (POWER_ORIGINS unset: every request is refused)"),
          flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
