"""THE PANEL'S DEFAULT STYLE — the one style fact the server holds (operator, 2026-10-31: "allow setup default in yml
config").

A STYLE is a set of values for the panel's CSS tokens (static/styles.js; docs/guides/panel-styles.md). Which style a
BROWSER shows is that browser's own choice, kept in its localStorage — the server stores none. What the server does hold
is the DEPLOYMENT'S DEFAULT: the style a browser that chose nothing opens in. That is a choice about what this
deployment IS, so it lives in the fleet file (`panel: {style: <key>}`, AGENTS.md "a deployment is configuration"),
REQUIRED like every fleet value: a fleet that names none, or a key the registry does not list, refuses the boot naming
the keys it lists — never a silently assumed style.

  THE REGISTRY  static/styles/index.json — the ONE list of styles (the menu's, styles.js's, this module's, the setup
                wizard's `panel_styles` source). Read from the package's own static dir: in the image that copy is the
                BUILT one, whose `file`s carry the build's content hash (lb/Dockerfile), so the href stamped below is the
                same cache-busted URL the menu loads.
  THE STAMP     every page the LB serves carries `<html … data-style-default="" data-style-default-href="">`, both empty
                in the source; `stamp` fills them with the fleet's key and its sheet's href, so styles.js applies the
                default IN <head>, BEFORE FIRST PAINT, with no request of its own (no flash of another style). An LB with
                no fleet file stamps nothing, and styles.js says so on the page.
"""
from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any

INDEX = Path(__file__).resolve().parent / "static" / "styles" / "index.json"
_KEY_ATTR = b'data-style-default=""'
_HREF_ATTR = b'data-style-default-href=""'


@functools.lru_cache(maxsize=1)
def registry() -> tuple[dict, ...]:
    """The styles the panel offers, in menu order: `{key, label, file}` each (`file` None for the base, gateway.css)."""
    data = json.loads(INDEX.read_text(encoding="utf-8"))
    return tuple({"key": s["key"], "label": s["label"], "file": s["file"]} for s in data["styles"])


def keys() -> tuple[str, ...]:
    return tuple(s["key"] for s in registry())


def known() -> str:
    """The keys as a refusal names them."""
    return ", ".join(keys())


def check(where: str, value: Any) -> str:
    """`value` as a style key the registry lists — or ValueError naming `where`, the value and every known key."""
    if not isinstance(value, str) or value not in keys():
        shown = repr(value) if isinstance(value, str) else str(value)
        raise ValueError(f"{where} {shown} is no style of the panel — one of {known()} (lb/vllm_lb/static/styles/index.json)")
    return value


def stamp(html: bytes, key: str) -> bytes:
    """`html` with the deployment's default style `key` and its sheet's href filled into the empty attributes on
    <html>. A page without them is returned unchanged (a page that loads no styles.js)."""
    entry = next((s for s in registry() if s["key"] == key), None)
    if entry is None:
        raise ValueError(f"the panel's default style {key!r} is no style of the panel — one of {known()}")
    href = (entry["file"] or "").encode()
    return (html.replace(_KEY_ATTR, b'data-style-default="' + key.encode() + b'"', 1)
                .replace(_HREF_ATTR, b'data-style-default-href="' + href + b'"', 1))
