"""Fakes for the power service: nothing in the tests opens a socket, runs ssh or sends a packet."""
from __future__ import annotations

import json

from fleet_power.app import PowerApp, Settings
from fleet_power.host_guard import parse_allowed
from fleet_power.boxes import boxes_from_json

ORIGIN = "https://power.example.test"
#: The name the edge's nginx presents (`proxy_set_header Host $host`: the name alone) — the suite's POWER_ALLOWED_HOSTS.
HOST = "power.example.test"
GATE = "g" * 40
USER = "operator"

#: The inventory the suite runs against. Its SHAPE is the operator's (two GB10 boxes on their own cables, and a
#: host with no Wake-on-LAN path); its VALUES are documentation addresses (RFC 5737) and locally
#: administered MACs, because the real ones live only in the operator's power/.env.
TEST_INVENTORY = """[
 {"name": "box-a", "ssh_host": "192.0.2.2", "in_cluster": true, "gpu_clock": true,
  "default_gpu_clock_mhz": 2200,
  "wol": {"mac": "02:00:00:00:00:02", "iface": "eth-box-a", "broadcast": "192.0.2.3"},
  "power_on_note": "Wake-on-LAN from the host over the direct cable. Untested until the first real power-off.",
  "consequence": "The model cluster goes down. After power-on it needs Start in the panel."},
 {"name": "box-b", "ssh_host": "198.51.100.2", "in_cluster": true, "gpu_clock": true,
  "wol": {"mac": "02:00:00:00:00:03", "iface": "eth-box-b", "broadcast": "198.51.100.3"},
  "power_on_note": "Wake-on-LAN from the host over the direct cable. Untested until the first real power-off.",
  "consequence": "The model cluster goes down. After power-on it needs Start in the panel."},
 {"name": "box-c", "ssh_host": "127.0.0.1",
  "power_on_note": "Power it on with its smart plug: this page runs on it.",
  "consequence": "This page and every service on the host go down."}
]"""


def suite_boxes():
    """The inventory the suite runs against, parsed exactly as the service parses POWER_BOXES."""
    return boxes_from_json(TEST_INVENTORY)


class FakeRunner:
    """Records every argv it is handed; answers with a scripted (rc, out, err)."""

    def __init__(self, result=(0, "", "")):
        self.calls: list[list[str]] = []
        self.result = result

    def __call__(self, argv, timeout):
        self.calls.append(list(argv))
        return self.result


class FakeWol:
    def __init__(self, error: Exception | None = None):
        self.sent: list[str] = []
        self.checked_only: list[str] = []
        self.error = error

    def __call__(self, box, *, send=True):
        if self.error:
            raise self.error
        (self.sent if send else self.checked_only).append(box.name)
        return f"magic packet for {box.wol.mac} on {box.wol.iface}"


class FakeProbe:
    def __init__(self, up=None):
        self.up = dict(up or {"box-a": True, "box-b": True, "box-c": True})

    def __call__(self, box):
        return self.up.get(box.name, False)


class FakeClock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def make_app(tmp_path, *, dry_run=False, cluster=None, runner=None, wol=None, probe=None,
             clock=None):
    settings = Settings(
        origins=(ORIGIN,),
        gate_secret=GATE,
        key_path="/keys/fleet-power",
        dry_key_path="/keys/fleet-power-dryrun",
        known_hosts="/keys/known_hosts",
        dry_run=dry_run,
        audit_path=str(tmp_path / "audit.jsonl"),
        allowed_hosts=parse_allowed("POWER_ALLOWED_HOSTS", HOST),
    )
    cluster_state = (lambda: cluster) if cluster is not None else (
        lambda: {"ok": True, "state": "off", "boxes": ["box-a", "box-b"]})
    app = PowerApp(settings, suite_boxes(), runner=runner or FakeRunner(),
                   wol=wol or FakeWol(), probe=probe or FakeProbe(),
                   cluster_state=cluster_state, clock=clock or FakeClock())
    return app


def headers(**extra):
    h = {"Host": HOST, "X-Power-Gate": GATE, "X-Power-User": USER, "X-Real-IP": "203.0.113.7",
         "Origin": ORIGIN, "Content-Type": "application/json"}
    for k, v in extra.items():
        k = k.replace("_", "-")
        if v is None:
            h.pop(k, None)
        else:
            h[k] = v
    return h


def post(app, path, body, **hdrs):
    raw = body if isinstance(body, bytes) else json.dumps(body).encode()
    status, _h, out = app.handle("POST", path, headers(**hdrs), raw)
    return status, (json.loads(out) if out[:1] == b"{" else out)


def get(app, path, **hdrs):
    hh = headers(**hdrs)
    hh.pop("Content-Type", None)
    status, h, out = app.handle("GET", path, hh, b"")
    return status, h, out


def audit_lines(tmp_path):
    p = tmp_path / "audit.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
