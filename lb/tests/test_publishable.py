"""What must hold before this gateway is published: no fleet address and no credential compiled
in as a default, every environment name the code reads written down where an operator
looks (lb/.env.example or the compose file) — and NOTHING under lb/ naming the deployment this
was built on (AGENTS.md, the open-source rule): no host, box, user, path, sibling project or
LAN prefix, in code, static pages, compose, examples, tests or fixtures.

The fleet's addresses are the operator's. A default that names them makes this repository a
description of one installation; an empty default makes the deployment say what it is missing.
The SearXNG secret is a credential: a default for it is a fallback secret.

The words themselves are not written here: tests/publishable_words.py holds their digests and
the one check (`offenders`), and `test_the_guard_bites_a_planted_word` proves it bites.
"""
from __future__ import annotations

import re
from pathlib import Path

from tests.publishable_words import CANARY, offenders

LB = Path(__file__).resolve().parents[1]                 # lb/
#: What the whole-tree scan skips: vendored code (its own project's words), git's own files, bytecode.
SKIPPED_PARTS = {".git", "__pycache__", "vendor", ".pytest_cache", "node_modules"}
#: The one file that spells the canary out, by design.
SKIPPED_FILES = {"tests/publishable_words.json"}


def _compose() -> str:
    return (LB / "docker-compose.yml").read_text()


def test_the_compose_names_no_fleet_address_as_a_default():
    text = _compose()
    assert offenders(text) == [], "a fleet address or host is compiled into lb/docker-compose.yml"
    from vllm_lb.config import RETIRED_KEYS
    for key in RETIRED_KEYS:
        assert not re.search(rf"^\s+- {key}=", text, re.M), f"{key} is retired into the fleet file"
    assert re.search(r"^\s+- LB_FLEET=/etc/vllm-lb/fleet\.yml$", text, re.M), "the fleet file's container path"


def test_the_code_defaults_name_no_fleet_and_no_fallback_box():
    from vllm_lb.config import Config
    cfg = Config()
    assert cfg.vllm_boxes == ()
    assert not [f for f in Config.__dataclass_fields__ if f.startswith("fallback_")], "the fallback tier is retired"
    assert cfg.hw_hosts == {}


def test_the_code_names_no_operator_host_in_its_defaults():
    for p in sorted((LB / "vllm_lb").rglob("*.py")):
        assert offenders(p.read_text()) == [], f"{p.relative_to(LB)} names a host of the deployment"


# ── the whole tree ───────────────────────────────────────────────────────────────────────────────

def _tree_files():
    for p in sorted(LB.rglob("*")):
        rel = p.relative_to(LB)
        if p.is_file() and not (SKIPPED_PARTS & set(rel.parts)) and str(rel) not in SKIPPED_FILES:
            yield p


def _text(p: Path) -> str | None:
    try:
        return p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None                 # a binary: nothing to read a word from


def test_nothing_under_lb_names_the_deployment_it_was_built_on():
    """Every text file under lb/ — code, static pages, compose files, examples, tests, captured fixtures — holds no
    word of the deployment (tests/publishable_words.json). A fixture captured from a live box keeps its numbers and
    loses its names; vendored code (a `vendor/` directory) is its own project's and is skipped."""
    found = {str(p.relative_to(LB)): hits for p in _tree_files()
             if (text := _text(p)) is not None and (hits := offenders(text))}
    assert found == {}, f"deployment words in the tree: {found}"


def test_the_guard_bites_a_planted_word(tmp_path):
    """The proof the scan is live: a file holding the canary — as a word, inside a hyphenated name, in any case — is
    caught; a file that only has the product spelling (ALLOWED) is not."""
    assert offenders(f"a box called {CANARY}") == [CANARY]
    assert offenders(f"the {CANARY.upper()}-gpu lane") == [CANARY], "any case, inside a hyphenated name"
    assert offenders("an NVIDIA DGX Spark, per eugr/spark-vllm-docker") == []
    assert offenders("TP=2 on box-a, box-b at 192.0.2.10 and 198.51.100.1") == []


def test_the_searxng_secret_has_no_default_and_no_settings_literal():
    """The compose default and the settings.yml copy were the same committed secret. Now the
    operator's lb/.env must carry it, or compose refuses to start."""
    text = _compose()
    assert "SEARXNG_SECRET=${SEARXNG_SECRET:?" in text
    assert "vllm-lb-searxng-local" not in text
    settings = (LB / "searxng" / "settings.yml").read_text()
    assert "vllm-lb-searxng-local" not in settings
    assert "vllm-lb-searxng-local" not in (LB / ".env.example").read_text()


def test_every_environment_name_the_lb_reads_is_documented():
    """A knob the code reads that no compose file (the base one or the broker override) or
    example names is a knob an operator cannot find. The cluster's CLUSTER_* names are written TO the rank's environment, not read from
    the operator's, so they are not in this list."""
    pat = re.compile(r'"(LB_[A-Z0-9_]+)"')
    names: set[str] = set()
    for p in (LB / "vllm_lb").rglob("*.py"):
        names |= set(pat.findall(p.read_text()))
    # A GONE name (a retired tier) is no knob to find: read only to refuse the boot naming itself (test_defaults).
    from vllm_lb.config import RETIRED_KEYS, Gone
    names -= {k for k, v in RETIRED_KEYS.items() if isinstance(v, Gone)}
    docs = _compose() + (LB / "docker-compose.broker.yml").read_text() + (LB / ".env.example").read_text()
    missing = sorted(n for n in names if not re.search(r"\b%s\b" % n, docs))
    assert not missing, f"read by the code, documented nowhere: {missing}"
