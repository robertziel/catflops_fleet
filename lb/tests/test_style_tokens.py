"""EVERY LOOK OF THE PANEL IS A TOKEN (operator, 2026-10-30: "Add switch between styles"): a style re-values tokens, so a
literal written into a rule is a part of the panel no style can reach — the default ("Vibe coded") would be the only
look it has.

  COLOURS         no hex, rgb()/rgba()/hsl()/hsla() of numbers, or bare "r, g, b" triplet outside a `:root` token
                  block, in any of the panel's stylesheets. (A triplet is how a hue is tinted: rgba(var(--ok-rgb), .15).)
  TYPE AND SHAPE  in the gateway page's sheets (gateway.css, layout.css, recorder.css) a font family, a font size and
                  a border radius are tokens too: var(), or `inherit`, an em ratio, 0 or 50 % — geometry, not style.
  WHAT JS READS   the default style defines every token palette.js reads; tests/palette.node.js proves its scales are
                  the colours gateway.js computed before styles existed.
"""
from __future__ import annotations

import pathlib
import re

import pytest

from tests.palette_tokens import root_tokens, rules_flat, strip_comments

STATIC = pathlib.Path(__file__).resolve().parent.parent / "vllm_lb" / "static"
COLOUR_SHEETS = ("gateway.css", "layout.css", "recorder.css", "benchmarks.css", "sessions.css")
TYPE_SHEETS = ("gateway.css", "layout.css", "recorder.css")

HEX = re.compile(r"#[0-9a-fA-F]{3,8}\b")
COLOUR_FN = re.compile(r"\b(?:rgba?|hsla?|hwb|oklch|lab|lch)\(\s*[\d.]")
TRIPLET = re.compile(r"(?<![\w.#-])\d{1,3}\s*,\s*\d{1,3}\s*,\s*\d{1,3}(?=\s*(?:[,;)]|$))")


def outside_tokens(name: str):
    for selector, decls in rules_flat((STATIC / name).read_text()):
        if selector == ":root":
            continue
        for prop, value in decls:
            yield selector, prop, value


@pytest.mark.parametrize("name", COLOUR_SHEETS)
def test_no_colour_is_written_outside_the_token_block(name):
    stray = [(s, p, v) for s, p, v in outside_tokens(name) if HEX.search(v) or COLOUR_FN.search(v) or TRIPLET.search(v)]
    assert not stray, f"{name}: a colour literal outside :root — make it a token: {stray[:5]}"


FS_OK = re.compile(r"^(?:var\(--[\w-]+\)|inherit|[\d.]+em)$")
RADIUS_PART_OK = re.compile(r"^(?:var\(--[\w-]+\)|0|50%)$")


@pytest.mark.parametrize("name", TYPE_SHEETS)
def test_type_and_radius_are_tokens(name):
    bad = []
    for selector, prop, value in outside_tokens(name):
        v = value.replace("!important", "").strip()
        if prop == "font-size" and not FS_OK.match(v):
            bad.append((selector, prop, v))
        if prop == "font-family" and not re.fullmatch(r"var\(--[\w-]+\)|inherit", v):
            bad.append((selector, prop, v))
        if prop == "font" and v != "inherit":
            # "<style> <weight> <size>/<line-height> <family>": the size and the family are tokens
            m = re.search(r"(\S+)/\S+\s+(.+)$", v)
            if not m or not FS_OK.match(m.group(1)) or not re.fullmatch(r"var\(--[\w-]+\)", m.group(2).strip()):
                bad.append((selector, prop, v))
        if re.fullmatch(r"border(?:-[a-z]+)*-radius", prop) and not all(RADIUS_PART_OK.match(x) for x in v.split()):
            bad.append((selector, prop, v))
    assert not bad, f"{name}: a type size, family or radius written as a literal: {bad[:6]}"


def test_the_default_style_defines_what_palette_js_reads():
    t = root_tokens((STATIC / "gateway.css").read_text())
    for name in ("--scale-space", "--scale-load", "--scale-rate", "--scale-queue", "--core-idle", "--panel-2"):
        assert name in t, f"gateway.css :root does not define {name} (palette.js reads it)"
    assert t["--scale-space"] == "hsb"
    js = strip_comments((STATIC / "palette.js").read_text())
    assert not HEX.search(js) and not re.search(r"\bhsla?\(\s*\d", js), "palette.js holds no colour of its own"
    gw = strip_comments((STATIC / "gateway.js").read_text())
    assert not re.search(r"rgba?\(\s*\d+\s*,\s*\d+\s*,\s*\d+", gw), "gateway.js writes an RGB colour of its own"
    assert not re.search(r"\bhsla?\(\$\{\(?\s*\(?(?:hue|h)\b", gw), "gateway.js computes an HSL colour of its own"
