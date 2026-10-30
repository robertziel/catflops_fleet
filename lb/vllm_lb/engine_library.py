"""WHAT EACH ENGINE SERVES — the capability half of an engine, as data (library/engines/<engine>.yml).

The engine redesign (docs/design/engines.md, operator-approved 2026-10-30) gives an ENGINE two halves:

  behaviour     CODE, one class or table per engine, as before: the command tp-rank builds (`vllm_cmd`,
                `sglang_cmd`, `tensorfold_cmd`), the knobs it refuses (catalog.ENGINE_REFUSES), its flags
                (catalog.VLLM_FLAGS, TENSORFOLD_*), a one-host node's fit (fleet.NodeEngine), its metrics
                (engine_metrics.py);
  capabilities  DATA, one YAML file per engine (this module reads them): the checkpoint FORMATS it loads (hf,
                ctranslate2, nemo), the model ARCHITECTURES it serves (a Hugging Face config.json's
                `architectures`), the QUANT FORMATS it has a method for (a model's quant_method; `none` =
                unquantised). The KV dtypes it serves are its images' (kv_cache_dtypes.json, per image, with the arch
                each runs on) — one place, not restated.

`model_problem` is the check a model gets against an engine before any image is asked for: "" when the engine serves
it, else the first refusal, naming the engine and the missing piece. A model whose architectures were never read
(null) is REFUSED, naming the tool that reads them (`python -m vllm_lb.model_facts <checkpoint>`) — never waved
through: an unknown is not a yes (AGENTS.md: no fallbacks). A format whose engine file states an EMPTY architecture
list (a CTranslate2 conversion, a .nemo archive name none) is checked by its format alone — the engine's own file
says so, with its source.

A new engine is this file's YAML plus its behaviour class (docs/guides/bring-your-own-hardware.md).
"""
from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from vllm_lb import yamlfile
from vllm_lb.hardware import Fact, read_fact, read_keys

#: The engine capability library this repo ships: one YAML file per engine, named by the catalog's engine word.
ENGINES_DIR = Path(__file__).resolve().parent / "library" / "engines"
_FACTS = {"formats": "words", "architectures": "names", "quant_formats": "words"}
#: How a model's architectures are read when they are missing (vllm_lb/model_facts.py).
FACTS_TOOL = "python -m vllm_lb.model_facts <checkpoint folder>"


@dataclass(frozen=True)
class Engine:
    """One engine's capabilities: what it loads and serves, each with its source."""

    key: str
    name: str
    formats: Fact
    architectures: Fact
    quant_formats: Fact

    def view(self) -> dict:
        return {"key": self.key, "name": self.name, **{k: getattr(self, k).view() for k in _FACTS}}


def _engine(key: str, raw, source: str) -> Engine:
    read_keys(f"{source}: the engine", raw, ("name", "facts"))
    if not isinstance(raw["name"], str) or not raw["name"].strip():
        raise ValueError(f"{source}: name must be the engine's name")
    facts = read_keys(f"{source}: facts", raw["facts"], tuple(_FACTS))
    return Engine(key=key, name=raw["name"].strip(),
                  **{k: read_fact(f"{source}: facts.{k}", facts[k], kind) for k, kind in _FACTS.items()})


def load_engines(directory: str | Path) -> Mapping[str, Engine]:
    """Every engine in `directory` (`<engine>.yml`), validated — or ValueError naming the file and the value."""
    files = sorted(Path(directory).glob("*.yml"))
    if not files:
        raise ValueError(f"{directory}: no engine (<engine>.yml) there")
    return MappingProxyType({f.stem: _engine(f.stem, yamlfile.read(f), f.name) for f in files})


@functools.cache
def engines() -> Mapping[str, Engine]:
    """The library this repo ships (ENGINES_DIR), read once."""
    return load_engines(ENGINES_DIR)


def engine(key: str, library: Mapping[str, Engine] | None = None) -> Engine:
    """The engine `key` names — or ValueError naming every one there is."""
    lib = engines() if library is None else library
    found = lib.get(key) if isinstance(key, str) else None
    if found is None:
        raise ValueError(f"no engine {key!r}: the engines are {', '.join(sorted(lib))} "
                         f"(vllm_lb/library/engines/<engine>.yml)")
    return found


def _lists(e: Engine, attr: str, value: str, what: str) -> str:
    f: Fact = getattr(e, attr)
    if not f.known:
        return f"{e.key} states no {what} ({f.source}), so {value!r} cannot be checked against it"
    if value not in f.value:
        return f"{e.key} ({e.name}) serves no {what} {value!r}: it lists {', '.join(f.value) or 'none'} ({f.source})"
    return ""


def model_problem(e: Engine, *, fmt: str | None, architectures, quant_format: str | None) -> str:
    """"" when engine `e` serves a checkpoint of format `fmt`, these `architectures` (None = never read) and this
    `quant_format`; otherwise the first refusal, naming the engine and the missing piece."""
    if fmt is None:
        return f"the checkpoint's format was never read: run `{FACTS_TOOL}` on a box that holds it"
    why = _lists(e, "formats", fmt, "checkpoint format")
    if why:
        return why
    if not e.architectures.known:
        return f"{e.key} states no architectures ({e.architectures.source})"
    if e.architectures.value:
        if architectures is None:
            return (f"the model's architectures were never read from its config.json: run `{FACTS_TOOL}` on a box "
                    f"that holds it")
        if not set(architectures) & set(e.architectures.value):
            return (f"{e.key} ({e.name}) serves none of the architectures {', '.join(architectures) or 'none'}: it "
                    f"lists {', '.join(e.architectures.value)} ({e.architectures.source})")
    return _lists(e, "quant_formats", quant_format, "quant format") if quant_format is not None else (
        "the model's quant format was never read: run `" + FACTS_TOOL + "` on a box that holds it")
