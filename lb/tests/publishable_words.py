"""The words this repository must not ship — one deployment's names — as digests, and the one check every guard runs.

A published tree describes no installation: no host, box, user, path, sibling project or LAN prefix of the fleet this
was built on (AGENTS.md, the open-source rule). The guards that enforce it cannot name those words in plain text — the
guard would be the leak — so tests/publishable_words.json holds their sha256 digests, and a text is checked by
tokenizing it the same way and hashing each token. Product names stay allowed by spelling (`ALLOWED`): a hardware
type's vendor citation ("DGX Spark") and an upstream recipe's repository are not this deployment's.

One definition: the python guards import `offenders`, the node guards read the same JSON (tests/connect.node.js).
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
_LIST = json.loads((HERE / "publishable_words.json").read_text())
DIGESTS: frozenset[str] = frozenset(_LIST["digests"])
#: The one listed word spelled out: a guard plants it to prove it bites. No shipped file has a reason to hold it.
CANARY: str = _LIST["canary"]

#: Spellings that contain a forbidden token but name something else: NVIDIA's product and the upstream recipes built
#: for it. Removed before tokenizing.
ALLOWED = re.compile(r"(?i)dgx[ -]sparks?|eugr/spark-vllm-docker")
_WORD = re.compile(r"\w+")
_HYPHENATED = re.compile(r"[\w-]+")
_IPV4 = re.compile(r"\b(\d{1,3}\.\d{1,3})\.\d{1,3}\.\d{1,3}\b")


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def tokens(text: str) -> set[str]:
    """Every word, every hyphenated word and every IPv4 address's first two octets of `text`, lowercase."""
    text = ALLOWED.sub(" ", unicodedata.normalize("NFC", text).lower())
    return set(_WORD.findall(text)) | set(_HYPHENATED.findall(text)) | set(_IPV4.findall(text))


def offenders(text: str) -> list[str]:
    """The tokens of `text` that are a deployment's word, sorted — [] is the only publishable answer."""
    return sorted(t for t in tokens(text) if _digest(t) in DIGESTS)
