"""Publishable: no operator machine, address or origin ships in the power service's code or compose,
and the settings refuse by default (no origin, no box) until the operator's environment says otherwise.
"""
from __future__ import annotations

import hashlib
import pathlib

from fleet_power.__main__ import settings_from_env

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: Operator literals, as SHA-256 of the lowercase word (the username, the two NIC names) or of the MAC's first three
#: octets — digests, so this file does not publish them. The LAN /30 cables are a prefix, matched as text.
OPERATOR_DIGESTS = {
    "72a9728227c5731a661480bb836c0f330d30d07a60749406942a28e3fdb3330f",
    "7816742cc79c786081a21cf3dbc719e2de796dfbd3f380b25daf02d607706bb8",
    "98e2bab598efe4a3802bc5d0e80725c30a32ef76656171a5db258dda3d9e2f12",
    "a52b65ef7f8805544c4b423904d0c13e9040a14006f46ac8324cbf64a0611081",
    "e6e828899f467633d293cc9346860b34adbcfcc97d856cef3d3599ba204247a2",
}
OPERATOR_PREFIXES = ("10.10.",)
SHIPPED = ("docker-compose.yml", "fleet_power/__main__.py", "fleet_power/boxes.py",
           "fleet_power/app.py", "fleet_power/wol.py", "fleet_power/static/button.js")


def _operator_hits(text: str) -> set[str]:
    import re
    low = text.lower()
    words = set(re.findall(r"\w+", low)) | set(re.findall(r"\b[0-9a-f]{2}:[0-9a-f]{2}:[0-9a-f]{2}", low))
    hits = {w for w in words if hashlib.sha256(w.encode()).hexdigest() in OPERATOR_DIGESTS}
    return hits | {p for p in OPERATOR_PREFIXES if p in text}


def test_no_operator_literal_ships_in_the_code_or_the_compose_file():
    for rel in SHIPPED:
        hits = _operator_hits((ROOT / rel).read_text())
        assert not hits, f"an operator literal is in {rel}: {sorted(hits)}"


def test_the_guard_reads_digests_not_words():
    assert _operator_hits("box-a on 192.0.2.2, iface eth-box-a, mac 02:00:00:00:00:02") == set()
    assert _operator_hits("a cable on 10.10." + "10.2") == {"10.10."}


def test_with_no_environment_no_origin_is_allowed():
    """An empty origin list refuses every request (app.py checks `origin in settings.origins`)."""
    assert settings_from_env({"POWER_ALLOWED_HOSTS": "loopback"}).origins == ()


def test_the_origin_is_read_from_the_environment():
    env = {"POWER_ALLOWED_HOSTS": "loopback", "POWER_ORIGINS": "https://a.example.test, https://b.example.test"}
    assert settings_from_env(env).origins \
        == ("https://a.example.test", "https://b.example.test")


def test_the_compose_takes_origin_and_inventory_from_the_environment_with_empty_defaults():
    text = (ROOT / "docker-compose.yml").read_text()
    assert 'POWER_ORIGINS: "${POWER_ORIGINS:-}"' in text
    assert 'POWER_BOXES: "${POWER_BOXES:-}"' in text


def test_the_root_env_file_is_optional_for_the_frp_client():
    """A missing repo-root .env must not fail the whole project: the frp token is only needed by frpc."""
    text = (ROOT / "docker-compose.yml").read_text()
    assert "path: ../.env" in text and "required: false" in text
    assert "env_file: ../.env" not in text
