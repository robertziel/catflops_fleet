"""What must hold before this panel is published: the public origin it answers is the
operator's to name (POWER_ORIGINS, no default: unset, every request is refused), and the
compose file takes it from the environment rather than carrying one domain."""
from __future__ import annotations

from pathlib import Path

POWER = Path(__file__).resolve().parents[1]


def test_the_compose_takes_the_public_origin_from_the_environment():
    text = (POWER / "docker-compose.yml").read_text()
    assert 'POWER_ORIGINS: "${POWER_ORIGINS:-}"' in text, "POWER_ORIGINS must come from power/.env"
    services = text.split("services:", 1)[1]
    from tests.test_publishable_config import _operator_hits
    assert not _operator_hits(services), "an operator literal is compiled into the power compose"
    assert "example.com" not in services and ".co/" not in services, "a domain is compiled into the power compose"


def test_without_an_origin_the_panel_allows_none():
    """No compiled-in origin: an unset POWER_ORIGINS is no allowed origin, so every request is refused."""
    from fleet_power.__main__ import settings_from_env
    assert settings_from_env({"POWER_ALLOWED_HOSTS": "loopback"}).origins == ()


def test_the_panel_names_its_origin_from_the_environment():
    from fleet_power.__main__ import settings_from_env
    env = {"POWER_ALLOWED_HOSTS": "loopback", "POWER_ORIGINS": "https://a.example.com, https://b.example.com"}
    assert settings_from_env(env).origins == ("https://a.example.com", "https://b.example.com")


def test_every_environment_name_the_panel_reads_is_documented():
    """A name the code reads that neither the compose file nor power/.env.example names is a knob
    an operator cannot find."""
    import re
    names: set[str] = set()
    for p in (POWER / "fleet_power").rglob("*.py"):
        names |= set(re.findall(r'"(POWER_[A-Z0-9_]+)"', p.read_text()))
    docs = (POWER / "docker-compose.yml").read_text() + (POWER / ".env.example").read_text()
    missing = sorted(n for n in names if not re.search(r"\b%s\b" % n, docs))
    assert not missing, f"read by the code, documented nowhere: {missing}"
