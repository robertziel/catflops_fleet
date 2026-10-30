"""ONE FIGURE AND ITS SOURCE — the `Fact` every hardware library file is made of, and the strict readers of one.

Both hardware libraries speak it: a card TYPE (types/<key>.yml, profiles.py) and an ACCELERATOR (accelerators/<key>.yml,
accelerator.py). It lives here, beneath both, so neither imports the other's loader to read a figure (DRY: one Fact,
one reader per kind of value — a second copy of `_fact` in the accelerator loader would drift from the type loader's).

A Fact is `{value, source}`: a figure and where it comes from — a vendor's spec page, or a measurement on this
project's fleet with its date and where it is recorded. A Fact without a source cannot be built, and a file holding
one is refused. Where no source states a figure its value is null and its source says why: absent, and said so.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

#: How a container may attach a GPU — each the name of a node compose override (deploy/standalone/node-gpu.<rt>.yml).
GPU_RUNTIMES = ("nvidia", "rocm")
#: The CPU platforms an image is built for — the fleet's `platform`, a type's `host_platform` and an accelerator's
#: `platforms` speak these.
PLATFORMS = ("linux/amd64", "linux/arm64")
#: A precision or a quant format's word: lower case, digits, `_` and `-` (compressed-tensors).
WORD = re.compile(r"[a-z0-9][a-z0-9_-]*")


@dataclass(frozen=True)
class Fact:
    """One figure about a kind of GPU and its source. `value` None = no source states it, and `source` says why."""

    value: Any
    source: str

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError(f"a hardware fact needs its source (a spec page or a measurement), got {self.value!r} "
                             f"with none")

    @property
    def known(self) -> bool:
        return self.value is not None

    def view(self) -> dict:
        value = list(self.value) if isinstance(self.value, tuple) else self.value
        return {"value": value, "source": self.source}


def keys(where: str, d: Any, required, optional=()) -> dict:
    """`d` as a mapping holding every `required` key and nothing else but `optional` ones — or ValueError."""
    if not isinstance(d, dict):
        raise ValueError(f"{where} must be a mapping, got {type(d).__name__}")
    missing = [k for k in required if k not in d]
    unknown = [k for k in d if k not in required and k not in optional]
    if missing or unknown:
        raise ValueError(f"{where} needs {', '.join(required)}"
                         + (f"; missing {', '.join(map(str, missing))}" if missing else "")
                         + (f"; unknown {', '.join(map(str, unknown))}" if unknown else ""))
    return d


def _ok(kind: str, v: Any) -> bool:
    if kind == "str":
        return isinstance(v, str) and bool(v.strip())
    if kind == "bool":
        return isinstance(v, bool)
    if kind == "platform":
        return v in PLATFORMS
    if kind == "runtime":
        return v in GPU_RUNTIMES
    if kind == "count":
        return isinstance(v, int) and not isinstance(v, bool) and v > 0
    if kind == "number":
        return isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0
    if kind == "platforms":
        return isinstance(v, list) and bool(v) and all(p in PLATFORMS for p in v) and len(set(v)) == len(v)
    if kind == "names":
        return (isinstance(v, list) and all(isinstance(w, str) and w.strip() == w and w for w in v)
                and len(set(v)) == len(v))
    if kind == "words":
        return (isinstance(v, list) and all(isinstance(w, str) and WORD.fullmatch(w) for w in v)
                and len(set(v)) == len(v))
    raise AssertionError(f"no fact kind {kind!r}")


def fact(where: str, raw: Any, kind: str) -> Fact:
    """`raw` ({value, source}) as a Fact of `kind` — or ValueError naming `where` and the value. A list becomes a tuple:
    a Fact is a frozen value."""
    keys(where, raw, ("value", "source"))
    v = raw["value"]
    if not (v is None or _ok(kind, v)):
        raise ValueError(f"{where}.value must be a {kind} or null, got {v!r}")
    if not isinstance(raw["source"], str) or not raw["source"].strip():
        raise ValueError(f"{where}.source must name where the figure comes from (or why it is absent)")
    return Fact(tuple(v) if isinstance(v, list) else v, raw["source"])
