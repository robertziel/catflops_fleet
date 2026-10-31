"""A stylesheet read the way the style tests need it (the python twin of tests/palette_env.js): comments stripped, the
top-level rules with their bodies (CSS nesting and @media kept whole), a rule's own declarations, and the token
blocks — the `:root { … }` rules — told apart from everything else."""
from __future__ import annotations

import re


def strip_comments(css: str) -> str:
    return re.sub(r"/\*.*?\*/", "", css, flags=re.S)


def top_rules(css: str) -> list[dict]:
    """[{prelude, body}] for every top-level block; a body may hold nested blocks."""
    text, out, depth, start, opened = strip_comments(css), [], 0, 0, -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                opened = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                out.append({"prelude": text[start:opened].strip(), "body": text[opened + 1:i]})
                start = i + 1
        elif ch == ";" and depth == 0:
            start = i + 1
    return out


def own_declarations(body: str) -> list[tuple[str, str]]:
    """A body's declarations at its own depth, nested blocks skipped."""
    out, depth, cur = [], 0, ""
    for ch in body:
        if ch == "{":
            depth += 1
            cur = ""
        elif ch == "}":
            depth -= 1
            cur = ""
        elif depth == 0:
            if ch == ";":
                out.append(cur)
                cur = ""
            else:
                cur += ch
    out.append(cur)
    pairs = []
    for d in out:
        if ":" in d:
            k, v = d.split(":", 1)
            if k.strip():
                pairs.append((k.strip(), v.strip()))
    return pairs


def rules_flat(css: str):
    """Every (selector, declarations) at any depth — @media bodies opened, :root blocks marked."""
    for r in top_rules(css):
        if r["prelude"].startswith("@media") or r["prelude"].startswith("@supports"):
            yield from rules_flat(r["body"])
        else:
            yield r["prelude"], own_declarations(r["body"])


def root_tokens(css: str) -> dict[str, str]:
    """The custom properties every top-level `:root` block declares (the default style's tokens)."""
    out = {}
    for r in top_rules(css):
        if r["prelude"] == ":root":
            out.update({k: v for k, v in own_declarations(r["body"]) if k.startswith("--")})
    return out
