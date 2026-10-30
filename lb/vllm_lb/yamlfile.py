"""A YAML file read STRICTLY — the one way this gateway reads its declarative configuration: the hardware types
(vllm_lb/hardware/types/*.yml) and the fleet file (vllm_lb/fleet.py). AGENTS.md: a deployment is configuration,
never code, and a configuration is refused loudly on anything it cannot mean.

PyYAML's safe loader is lenient in three ways a configuration must not be:
  * a key written twice silently keeps the LAST value — two `boxes:` sections, and the first box list is gone;
  * YAML 1.1 reads `yes`, `no`, `on`, `off` as booleans — a box named `no` would be False, `serves: no` a quiet
    False where `serves: false` was meant to be the only spelling;
  * an error does not say which file it came from.
Here a duplicate key is refused, only true/false (any case YAML 1.2 allows) are booleans, and every error names the
file. Anything else is PyYAML's safe loader unchanged: no tags that build objects.

PyYAML is a runtime dependency for this module alone (lb/Dockerfile says why, and pins it).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

_BOOL = "tag:yaml.org,2002:bool"


class _StrictLoader(yaml.SafeLoader):
    """SafeLoader with the YAML 1.2 booleans only, and a mapping that refuses a key it has already seen."""

    def construct_mapping(self, node, deep=False):
        if isinstance(node, yaml.MappingNode):
            seen: set[Any] = set()
            for key_node, _ in node.value:
                key = self.construct_object(key_node, deep=deep)
                if key in seen:
                    raise yaml.constructor.ConstructorError(None, None, f"{key!r} is written twice",
                                                            key_node.start_mark)
                seen.add(key)
        return super().construct_mapping(node, deep=deep)


_StrictLoader.yaml_implicit_resolvers = {
    first: [(tag, rx) for tag, rx in resolvers if tag != _BOOL]
    for first, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()}
_StrictLoader.add_implicit_resolver(_BOOL, re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"), list("tTfF"))


def parse(text: str, source: str) -> Any:
    """The document in `text`, read strictly; ValueError naming `source` (a file name) on anything wrong."""
    try:
        return yaml.load(text, Loader=_StrictLoader)  # noqa: S506 — _StrictLoader IS a SafeLoader
    except yaml.YAMLError as exc:
        raise ValueError(f"{source}: not valid YAML: {exc}") from None


def read_text(path: str | Path) -> str:
    """The text of the file at `path`; ValueError naming the path when it is missing, not a file or unreadable."""
    p = Path(path)
    if not p.exists():
        raise ValueError(f"{p}: no such file")
    if not p.is_file():
        raise ValueError(f"{p}: not a file")
    try:
        return p.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"{p}: cannot be read: {exc}") from None


def read(path: str | Path) -> Any:
    """The document in the file at `path`, read strictly; ValueError naming the file when it is missing, not a
    file, unreadable or not valid YAML."""
    return parse(read_text(path), Path(path).name)
