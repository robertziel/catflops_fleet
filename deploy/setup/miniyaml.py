"""A STRICT YAML SUBSET, read and written with the standard library alone — what the setup wizard (SETUP.md) reads its
question library and answer sheets with, and writes lb/fleet.yml with.

Why not PyYAML: the wizard's renderer runs in python:3.12-slim, before any image of this repo is built (CLAUDE.md: a
third-party dependency belongs to a service's Dockerfile, and deploy/setup is no service). The LB reads its fleet file
with PyYAML's safe loader made strict (lb/vllm_lb/yamlfile.py); this module reads the same documents the same way —
deploy/tests/test_setup.py compares the two on every wizard file and every shipped fleet example inside the vllm-lb
image — and refuses, naming the source, anything outside the subset rather than guessing what it means:

  * block mappings and sequences (two-space indents by convention, any consistent indent accepted), flow sequences
    `[a, b]` and flow mappings `{k: v}`, `# comments`, plain / 'single' / "double" quoted scalars, and folded `>` or
    literal `|` block scalars;
  * the YAML 1.2 core scalars: null / ~, true / false (yamlfile's only booleans — `yes`, `no`, `on`, `off` stay strings,
    so `serves: no` is never a quiet False), integers, decimals; everything else a string;
  * REFUSED: a key written twice, a tab in the indent, anchors, aliases, tags, document markers, an unclosed bracket or
    quote, and anything an indent cannot place.

`dump` writes block mappings, flow sequences of scalars, and flow mappings for a list of small records (a box's cards, a
cluster's ranks — the shape lb/fleet.example.yml uses), quoting every string that would read back as something else.
"""
from __future__ import annotations

import json
import re
from typing import Any

_INT = re.compile(r"[-+]?[0-9]+")
_FLOAT = re.compile(r"[-+]?(?:[0-9]+\.[0-9]*|\.[0-9]+)(?:[eE][-+]?[0-9]+)?")
_NULL = ("null", "Null", "NULL", "~", "")
_TRUE = ("true", "True", "TRUE")
_FALSE = ("false", "False", "FALSE")


class _Err(ValueError):
    pass


def _scalar(text: str) -> Any:
    """A plain scalar's value (the YAML 1.2 core schema, yamlfile's booleans)."""
    t = text.strip()
    if t in _NULL:
        return None
    if t in _TRUE:
        return True
    if t in _FALSE:
        return False
    if _INT.fullmatch(t):
        return int(t)
    if _FLOAT.fullmatch(t):
        return float(t)
    return t


def _strip_comment(line: str) -> str:
    """`line` without a trailing `# comment` — a `#` at the start or after whitespace, outside quotes."""
    quote = None
    for i, ch in enumerate(line):
        if quote:
            if ch == quote:
                if quote == "'" and line[i + 1:i + 2] == "'":
                    continue
                if quote == '"' and line[i - 1:i] == "\\" and line[i - 2:i - 1] != "\\":
                    continue
                quote = None
        elif ch in "'\"" and (i == 0 or line[i - 1] in " \t[{,:-"):
            quote = ch
        elif ch == "#" and (i == 0 or line[i - 1] in " \t"):
            return line[:i].rstrip()
    return line.rstrip()


class _Flow:
    """A flow collection or quoted scalar, read character by character from `text` at `pos`."""

    def __init__(self, text: str):
        self.text, self.pos = text, 0

    def _ws(self):
        while self.pos < len(self.text) and self.text[self.pos] in " \t":
            self.pos += 1

    def value(self, stops: str) -> Any:
        self._ws()
        if self.pos >= len(self.text):
            raise _Err("a value is missing")
        ch = self.text[self.pos]
        if ch == "[":
            return self._seq()
        if ch == "{":
            return self._map()
        if ch in "'\"":
            return self.quoted()
        if ch in "&*!|>%@`":
            raise _Err(f"{ch!r} (an anchor, alias, tag or reserved indicator) is outside the subset this file is read in")
        start = self.pos
        while self.pos < len(self.text) and self.text[self.pos] not in stops:
            if self.text[self.pos] == ":" and self.text[self.pos + 1:self.pos + 2] in (" ", "") and ":" in stops:
                break
            self.pos += 1
        return _scalar(self.text[start:self.pos])

    def quoted(self) -> str:
        q = self.text[self.pos]
        i = self.pos + 1
        if q == "'":
            out = []
            while True:
                j = self.text.find("'", i)
                if j < 0:
                    raise _Err("a 'single-quoted' string is not closed")
                out.append(self.text[i:j])
                if self.text[j + 1:j + 2] == "'":
                    out.append("'")
                    i = j + 2
                    continue
                self.pos = j + 1
                return "".join(out)
        j = i
        while j < len(self.text):
            if self.text[j] == "\\":
                j += 2
                continue
            if self.text[j] == '"':
                try:
                    value = json.loads(self.text[self.pos:j + 1])
                except ValueError:
                    raise _Err(f"an escape in {self.text[self.pos:j + 1]} is outside the subset (JSON's escapes)") from None
                self.pos = j + 1
                return value
            j += 1
        raise _Err('a "double-quoted" string is not closed')

    def _seq(self) -> list:
        self.pos += 1
        out = []
        while True:
            self._ws()
            if self.pos >= len(self.text):
                raise _Err("a [flow sequence] is not closed")
            if self.text[self.pos] == "]":
                self.pos += 1
                return out
            out.append(self.value(",]"))
            self._ws()
            if self.pos < len(self.text) and self.text[self.pos] == ",":
                self.pos += 1

    def _map(self) -> dict:
        self.pos += 1
        out: dict = {}
        while True:
            self._ws()
            if self.pos >= len(self.text):
                raise _Err("a {flow mapping} is not closed")
            if self.text[self.pos] == "}":
                self.pos += 1
                return out
            key = self.value(":,}")
            self._ws()
            if self.text[self.pos:self.pos + 1] != ":":
                raise _Err(f"a flow mapping key {key!r} has no ': value'")
            self.pos += 1
            if key in out:
                raise _Err(f"{key!r} is written twice")
            out[key] = self.value(",}")
            self._ws()
            if self.pos < len(self.text) and self.text[self.pos] == ",":
                self.pos += 1

    def end(self):
        self._ws()
        if self.pos != len(self.text):
            raise _Err(f"unexpected text after a value: {self.text[self.pos:]!r}")


def _inline(text: str) -> Any:
    f = _Flow(text)
    v = f.value("")
    f.end()
    return v


class _Lines:
    def __init__(self, text: str):
        self.raw = text.split("\n")
        self.items: list[tuple[int, int, str]] = []      # (line number, indent, content)
        for n, line in enumerate(self.raw, 1):
            if line.strip().startswith("#") or not line.strip():
                continue
            lead = line[:len(line) - len(line.lstrip(" \t"))]
            if "\t" in lead:
                raise _Err(f"line {n}: a tab in the indent")
            if line.strip() in ("---", "...") or line.startswith("%"):
                raise _Err(f"line {n}: a document marker or directive is outside the subset")
            self.items.append((n, len(lead), _strip_comment(line.strip())))
        self.i = 0

    def peek(self):
        return self.items[self.i] if self.i < len(self.items) else None


def _key(text: str, n: int) -> tuple[Any, str] | None:
    """(key, rest) when `text` is a `key: rest` line, else None."""
    if text[:1] in "'\"":
        f = _Flow(text)
        key = f.quoted()
        rest = text[f.pos:].lstrip()
        if not rest.startswith(":"):
            return None
        return key, rest[1:].strip()
    m = re.match(r"([^\s:#\[\]{},][^:#]*?)\s*:(?:\s+|$)(.*)$", text)
    if m is None:
        return None
    return _scalar(m.group(1)), m.group(2).strip()


def _block_scalar(lines: _Lines, style: str, parent: int, n: int) -> str:
    """A `>` or `|` scalar: the raw lines indented past `parent` that follow, folded with spaces or kept."""
    start = n                                    # 1-based number of the indicator's line
    body, j = [], start
    while j < len(lines.raw):
        raw = lines.raw[j]
        if raw.strip() and len(raw) - len(raw.lstrip(" ")) <= parent:
            break
        body.append(raw)
        j += 1
    while lines.peek() is not None and lines.peek()[0] <= j:
        lines.i += 1
    texts = [b.strip() for b in body]
    while texts and not texts[-1]:
        texts.pop()
    return (" ".join(t for t in texts if t) if style == ">" else "\n".join(texts) + "\n")


def _value(lines: _Lines, rest: str, indent: int, n: int, *, in_seq_item: bool = False) -> Any:
    """The value after `key:` (or `- `) whose line is indented `indent`."""
    if rest in (">", "|", ">-", "|-"):
        return _block_scalar(lines, rest[0], indent, n)
    if rest:
        try:
            return _inline(rest)
        except _Err as exc:
            raise _Err(f"line {n}: {exc}") from None
    nxt = lines.peek()
    if nxt is None:
        return None
    if nxt[1] > indent or (nxt[1] == indent and nxt[2].startswith("-") and not in_seq_item
                           and (nxt[2] == "-" or nxt[2].startswith("- "))):
        return _block(lines, nxt[1])
    return None


def _block(lines: _Lines, indent: int) -> Any:
    first = lines.peek()
    if first[2] == "-" or first[2].startswith("- "):
        out = []
        while (cur := lines.peek()) is not None and cur[1] == indent and (cur[2] == "-" or cur[2].startswith("- ")):
            n, _, text = cur
            lines.i += 1
            body = text[1:].strip()
            kv = _key(body, n) if body and body[0] not in "[{" else None
            if kv is not None:
                # `- key: v` opens a mapping whose further keys sit at the item's content indent.
                inner = indent + (len(text) - len(text[1:].lstrip()))
                item: dict = {}
                _map_entry(lines, item, kv, inner, n)
                while (c := lines.peek()) is not None and c[1] == inner and not c[2].startswith("- "):
                    lines.i += 1
                    kv2 = _key(c[2], c[0])
                    if kv2 is None:
                        raise _Err(f"line {c[0]}: expected `key: value` in a sequence item's mapping")
                    _map_entry(lines, item, kv2, inner, c[0])
                out.append(item)
            else:
                out.append(_value(lines, body, indent, n, in_seq_item=True))
        return out
    out: dict = {}
    while (cur := lines.peek()) is not None and cur[1] == indent:
        n, _, text = cur
        if text.startswith("- "):
            raise _Err(f"line {n}: a sequence item where a mapping key belongs")
        lines.i += 1
        kv = _key(text, n)
        if kv is None:
            raise _Err(f"line {n}: expected `key: value`, got {text!r}")
        _map_entry(lines, out, kv, indent, n)
    nxt = lines.peek()
    if nxt is not None and nxt[1] > indent:
        raise _Err(f"line {nxt[0]}: indented under nothing")
    return out


def _map_entry(lines: _Lines, out: dict, kv: tuple, indent: int, n: int) -> None:
    key, rest = kv
    if key in out:
        raise _Err(f"line {n}: {key!r} is written twice")
    if isinstance(key, (dict, list)):
        raise _Err(f"line {n}: a key must be a scalar")
    out[key] = _value(lines, rest, indent, n)


def parse(text: str, source: str) -> Any:
    """The document in `text`; ValueError naming `source` on anything outside the subset."""
    try:
        lines = _Lines(text)
        if lines.peek() is None:
            return None
        first = lines.peek()
        if first[1] != 0:
            raise _Err(f"line {first[0]}: the document must start at column 0")
        if first[2][:1] in "[{\"'" and _key(first[2], first[0]) is None:
            lines.i += 1
            doc = _inline(first[2])
        else:
            doc = _block(lines, 0)
        if lines.peek() is not None:
            raise _Err(f"line {lines.peek()[0]}: unexpected text")
        return doc
    except _Err as exc:
        raise ValueError(f"{source}: {exc}") from None


# ── writing ───────────────────────────────────────────────────────────────────────────────────────────────────────

_PLAIN = re.compile(r"[A-Za-z0-9_./@+-][A-Za-z0-9_./@+=,()-]*")


def _out_scalar(v: Any) -> str:
    if v is None:
        return "null"
    if v is True:
        return "true"
    if v is False:
        return "false"
    if isinstance(v, (int, float)):
        return repr(v)
    s = str(v)
    if _PLAIN.fullmatch(s) and _scalar(s) == s and not s.startswith("-"):
        return s
    return json.dumps(s, ensure_ascii=False)


def _flow(v: Any) -> str:
    if isinstance(v, dict):
        return "{" + ", ".join(f"{_out_scalar(k)}: {_flow(x)}" for k, x in v.items()) + "}"
    if isinstance(v, list):
        return "[" + ", ".join(_flow(x) for x in v) + "]"
    return _out_scalar(v)


def _small(v: Any) -> bool:
    return not isinstance(v, (dict, list)) or (isinstance(v, dict) and all(not isinstance(x, (dict, list)) for x in v.values()))


def dump(doc: Any, indent: int = 0) -> str:
    """`doc` (mappings, lists, scalars) as YAML this module and PyYAML both read back as `doc`."""
    pad = " " * indent
    if isinstance(doc, dict):
        if not doc:
            return pad + "{}\n"
        out = []
        for k, v in doc.items():
            if isinstance(v, dict) and v:
                out.append(f"{pad}{_out_scalar(k)}:\n{dump(v, indent + 2)}")
            elif isinstance(v, list) and v and not all(_small(x) for x in v):
                out.append(f"{pad}{_out_scalar(k)}:\n" + "".join(f"{pad}  - {_flow(x)}\n" for x in v))
            elif isinstance(v, list) and v and any(isinstance(x, dict) for x in v):
                out.append(f"{pad}{_out_scalar(k)}:\n" + "".join(f"{pad}  - {_flow(x)}\n" for x in v))
            else:
                out.append(f"{pad}{_out_scalar(k)}: {_flow(v)}\n")
        return "".join(out)
    return pad + _flow(doc) + "\n"
