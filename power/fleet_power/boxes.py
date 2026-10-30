"""The machines this service may act on — an allowlist that comes from the OPERATOR's environment.

A box that is not in the inventory is a 404. The inventory is `POWER_BOXES`: a JSON array, one object per
machine. It is operator hardware (addresses, Wake-on-LAN MACs, NIC names, the operator's own wording for
each machine), so it lives in the operator's gitignored env file (power/.env) and never in code.

Unset or blank means NO box (fail closed). A malformed value is a ValueError that names POWER_BOXES, raised
at boot, so a typo is seen at once instead of as an empty page.

Addresses in an inventory are the direct cables from the host this service runs on, never the FRP tunnels:
each box's authorized_keys entry pins `from=` to the host's end of its own cable (install/host-setup.sh).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from collections.abc import Mapping


@dataclass(frozen=True)
class Wol:
    mac: str          # the target NIC's MAC (the box's end of the direct cable)
    iface: str        # the host's interface on that cable; the packet may leave by no other
    broadcast: str    # the cable's /30 broadcast address


@dataclass(frozen=True)
class Box:
    name: str
    ssh_host: str
    ssh_port: int = 22
    ssh_user: str = "fleet-power"
    in_cluster: bool = False            # powering it off drops a cluster (a TP group) it belongs to
    wol: Wol | None = None              # None = this service cannot power it on
    power_on_note: str = ""
    consequence: str = ""
    gpu_clock: bool = False             # has an NVIDIA GPU whose clock this service may cap
    # The cap this service keeps on the GPU (one of app.CLOCK_CAPS_MHZ); None = keep it released.
    # Re-applied every time the box comes back (a clock lock does not survive a reboot).
    default_gpu_clock_mhz: int | None = None


_BOX_FIELDS = {"name", "ssh_host", "ssh_port", "ssh_user", "in_cluster", "wol", "power_on_note",
               "consequence", "gpu_clock", "default_gpu_clock_mhz"}
_WOL_FIELDS = {"mac", "iface", "broadcast"}


def _wol(raw, where: str) -> Wol | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"POWER_BOXES {where}.wol must be an object or null")
    unknown = set(raw) - _WOL_FIELDS
    missing = _WOL_FIELDS - set(raw)
    if unknown or missing:
        raise ValueError(f"POWER_BOXES {where}.wol needs exactly mac, iface, broadcast"
                         + (f"; missing {sorted(missing)}" if missing else "")
                         + (f"; unknown {sorted(unknown)}" if unknown else ""))
    return Wol(mac=str(raw["mac"]), iface=str(raw["iface"]), broadcast=str(raw["broadcast"]))


def _default_clock(raw: dict, where: str) -> int | None:
    from fleet_power.app import CLOCK_CAPS_MHZ     # app imports this module; the caps live there
    mhz = raw.get("default_gpu_clock_mhz")
    if mhz is None:
        return None
    if type(mhz) is not int or mhz not in CLOCK_CAPS_MHZ:     # type(): a JSON true is not 1 MHz
        raise ValueError(f"POWER_BOXES {where}.default_gpu_clock_mhz must be one of "
                         f"{list(CLOCK_CAPS_MHZ)} (a cap this service has a key for) or absent")
    if raw.get("gpu_clock") is not True:
        raise ValueError(f"POWER_BOXES {where}.default_gpu_clock_mhz needs gpu_clock: true")
    return mhz


def boxes_from_json(text: str) -> dict[str, Box]:
    """POWER_BOXES text → the allowlist, keyed by name. Blank → {} (no box may be acted on)."""
    if not text.strip():
        return {}
    try:
        items = json.loads(text)
    except ValueError as e:
        raise ValueError(f"POWER_BOXES is not valid JSON ({e.msg} at character {e.pos})") from None
    if not isinstance(items, list):
        raise ValueError("POWER_BOXES must be a JSON array with one object per box")
    out: dict[str, Box] = {}
    for i, raw in enumerate(items):
        where = f"[{i}]"
        if not isinstance(raw, dict):
            raise ValueError(f"POWER_BOXES {where} must be an object")
        unknown = set(raw) - _BOX_FIELDS
        if unknown:
            raise ValueError(f"POWER_BOXES {where} has unknown field(s) {sorted(unknown)}")
        for required in ("name", "ssh_host"):
            if not isinstance(raw.get(required), str) or not raw[required].strip():
                raise ValueError(f"POWER_BOXES {where} needs a non-empty string {required!r}")
        name = raw["name"].strip()
        if name in out:
            raise ValueError(f"POWER_BOXES names box {name!r} twice")
        out[name] = Box(
            name=name,
            ssh_host=raw["ssh_host"].strip(),
            ssh_port=int(raw.get("ssh_port", 22)),
            ssh_user=str(raw.get("ssh_user", "fleet-power")),
            in_cluster=bool(raw.get("in_cluster", False)),
            wol=_wol(raw.get("wol"), where),
            power_on_note=str(raw.get("power_on_note", "")),
            consequence=str(raw.get("consequence", "")),
            gpu_clock=bool(raw.get("gpu_clock", False)),
            default_gpu_clock_mhz=_default_clock(raw, where),
        )
    return out


def load_boxes(env: Mapping[str, str]) -> dict[str, Box]:
    """The allowlist from the environment (POWER_BOXES). Unset = empty = every box is refused."""
    return boxes_from_json(env.get("POWER_BOXES", ""))
