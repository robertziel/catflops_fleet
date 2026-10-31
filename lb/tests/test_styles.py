"""THE PANEL'S VISUAL STYLES (operator, 2026-10-30: "Add switch between styles, current is 'Vibe coded'").

What is pinned here, and why:

  THE REGISTRY   `static/styles/index.json` lists the styles in menu order; the first is "Vibe coded" (`vibe`), which
                 IS gateway.css's :root block and has no file; every other entry names `/styles/<key>.css`, and every
                 sheet under styles/ is listed — a sheet nobody can pick, or an entry with no sheet, is a mistake.
  A STYLE SHEET  is scoped: each top-level rule is `:root[data-style="<key>"]` (nesting inside it), an @font-face or an
                 @keyframes — a sheet that styled the panel unscoped would restyle every style, the default included.
                 Its fonts are vendored beside it with their licences: the panel fetches nothing from a CDN.
  SERVED         the registry, every sheet, every font (as font/woff2) and both scripts are answered by the LB's static
                 handler; nothing about a style is posted — the choice is the browser's (localStorage `lb.style`).
  STAMPED        lb/Dockerfile's stamp steps, RUN here on a copy of static/: every sheet in index.json, every font in
                 its sheet, the index's URL in styles.js and both scripts in every page carry their content hash.
  EVERY PAGE     index, benchmarks and sessions load styles.js in <head> (before first paint), and carry the control's
                 slot and the notice line; palette.js comes before the script that computes colours.
  THE NAMES      no style's source — a site, a company, a product, a typeface — is named anywhere in the shipped static
                 tree (operator): a style is named for what it looks like.
  APPEARANCE     "Glass" follows the system's light or dark appearance: one nested dark block re-values its tokens
                 (and styles.js tells palette.js when the appearance changes — tests/styles.node.js).
  PHOSPHOR       "Terminal" is one dark appearance on a black ground, one monospace family, glowing; every motion it
                 adds (a blinking cursor) stops under prefers-reduced-motion, and its CRT overlay never takes a click.
The browser-side logic is tests/styles.node.js and tests/palette.node.js, run here when node is on the path.
"""
from __future__ import annotations

import hashlib
import http.client
import json
import pathlib
import re
import shutil
import subprocess

import pytest

from tests.fakes import FakeBackend
from tests.palette_tokens import own_declarations, top_rules
from vllm_lb.config import Config, VllmBox
from vllm_lb.hub import Hub
from vllm_lb.proxy import Proxy

HERE = pathlib.Path(__file__).resolve().parent
LB = HERE.parent
STATIC = LB / "vllm_lb" / "static"
STYLES = STATIC / "styles"
PAGES = ("index.html", "benchmarks.html", "sessions.html")


def index() -> list[dict]:
    return json.loads((STYLES / "index.json").read_text())["styles"]


# ── the registry ────────────────────────────────────────────────────────────────────────────────────

def test_the_registry_starts_with_vibe_coded_and_lists_every_sheet():
    styles = index()
    assert styles[0] == {"key": "vibe", "label": "Vibe coded", "file": None}
    keys = [s["key"] for s in styles]
    assert len(keys) == len(set(keys))
    for s in styles[1:]:
        assert set(s) == {"key", "label", "file"} and s["label"].strip()
        assert re.fullmatch(r"[a-z][a-z0-9-]{0,31}", s["key"]), s
        assert s["file"] == f"/styles/{s['key']}.css" and (STYLES / f"{s['key']}.css").is_file(), s
    sheets = sorted(p.stem for p in STYLES.glob("*.css"))
    assert sheets == sorted(keys[1:]), f"a sheet the menu cannot pick, or an entry with no sheet: {sheets} vs {keys}"
    assert [(s["key"], s["label"]) for s in styles] == [
        ("vibe", "Vibe coded"), ("clean", "Clean"), ("mono", "Mono"), ("glass", "Glass"), ("terminal", "Terminal")]


@pytest.mark.parametrize("sheet", sorted(STYLES.glob("*.css")), ids=lambda p: p.name)
def test_a_style_sheet_is_scoped_to_its_key_and_its_fonts_are_vendored(sheet):
    key = sheet.stem
    scope = f':root[data-style="{key}"]'

    def scoped(rules):
        for rule in rules:
            p = rule["prelude"]
            if p.startswith("@media"):          # a media query holds only scoped rules
                for inner in top_rules(rule["body"]):
                    assert all(s.strip().startswith(scope) for s in inner["prelude"].split(",")), \
                        f"{sheet.name}: an unscoped rule inside {p!r}: {inner['prelude']!r}"
                continue
            assert p == scope or p == "@font-face" or p.startswith(f"@keyframes {key}-"), \
                f"{sheet.name}: an unscoped top-level rule (or a keyframes name outside {key}-…) restyles every style: {p!r}"

    scoped(top_rules(sheet.read_text()))
    for url in re.findall(r'url\("([^"]+)"\)', sheet.read_text()):
        assert url.startswith("fonts/") and (STYLES / url).is_file(), f"{sheet.name}: {url} is not vendored"
    fonts = list((STYLES / "fonts").glob("*.woff2"))
    for f in fonts:
        family = f.name.split("-")[0]
        assert (STYLES / "fonts" / f"{family}-OFL.txt").is_file(), f"{f.name} ships without its licence"


# ── served ──────────────────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def proxy():
    b = FakeBackend()
    cfg = Config(vllm_boxes=(VllmBox("box-a-gpu", "box-a", b.url),), listen_host="127.0.0.1", listen_port=0)
    p = Proxy(cfg, Hub(cfg))
    p.start()
    yield p
    p.stop()
    b.close()


def test_the_registry_every_sheet_every_font_and_both_scripts_are_served(proxy):
    c = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=10)

    def get(path):
        c.request("GET", path)
        r = c.getresponse()
        return r.status, r.getheader("Content-Type") or "", r.read()

    status, ctype, body = get("/styles/index.json?v=0123456789ab")      # stamped, as the build sends it
    assert status == 200 and "json" in ctype and json.loads(body)["styles"][0]["key"] == "vibe"
    for s in index()[1:]:
        status, ctype, body = get(s["file"] + "?v=0123456789ab")
        assert status == 200 and ctype.startswith("text/css") and f':root[data-style="{s["key"]}"]'.encode() in body
    for f in sorted((STYLES / "fonts").glob("*.woff2")):
        status, ctype, body = get(f"/styles/fonts/{f.name}")
        assert status == 200 and ctype == "font/woff2" and body[:4] == b"wOF2", f.name
    for js in ("styles.js", "palette.js"):
        status, ctype, _ = get(f"/{js}")
        assert status == 200 and "javascript" in ctype
    c.request("POST", "/styles/index.json", body="{}", headers={"Content-Type": "application/json"})
    r = c.getresponse()
    assert r.status == 405
    r.read()


def test_the_server_keeps_no_browsers_style_only_the_deployments_default():
    """A browser's choice (`lb.style`, `data-style`) is never the server's; the one style fact it holds is the
    deployment's default (the fleet's `panel.style`, stamped by panel_styles.py — tests/test_panel_style.py). It is a
    fleet value, so never an env name too (one place per value)."""
    for p in (LB / "vllm_lb").rglob("*.py"):
        text = p.read_text()
        assert "lb.style" not in text and 'data-style"' not in text and "data-style=" not in text, f"{p.name} knows a style"
    assert "STYLE" not in (LB / ".env.example").read_text()


# ── stamped: the Dockerfile's own steps, run on a copy ──────────────────────────────────────────────

def stamp_steps() -> list[str]:
    """The Dockerfile's static-stamping RUN commands, as shell, in build order."""
    text = (LB / "Dockerfile").read_text().replace("\\\n", " ")
    runs = [m.group(1) for m in re.finditer(r"^RUN (cd /app/vllm_lb/static.*)$", text, re.M)]
    assert len(runs) == 2, "the styles step, then the pages' step"
    return runs


def sha12(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12]


def test_the_build_stamps_every_sheet_font_and_script(tmp_path):
    app = tmp_path / "app" / "vllm_lb"
    shutil.copytree(STATIC, app / "static")
    built = app / "static"
    originals = {p.name: p.read_bytes() for p in (STYLES / "fonts").glob("*.woff2")}
    for cmd in stamp_steps():
        r = subprocess.run(["sh", "-c", cmd.replace("/app/vllm_lb", str(app))], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
    sheets = built / "styles"
    for name in originals:                                   # each font, in the sheet that uses it
        v = hashlib.sha256(originals[name]).hexdigest()[:12]
        users = [p for p in sheets.glob("*.css") if f"fonts/{name}" in p.read_text()]
        assert all(f'url("fonts/{name}?v={v}")' in p.read_text() for p in users), name
    idx = json.loads((sheets / "index.json").read_text())["styles"]
    for s in idx[1:]:                                        # each sheet, in the index (hashed AFTER its fonts)
        path, _, v = s["file"].partition("?v=")
        assert v == sha12(built / path.lstrip("/")), s
    js = (built / "styles.js").read_text()                   # the index's URL in styles.js
    assert f"'/styles/index.json?v={sha12(sheets / 'index.json')}'" in js
    for page in PAGES:                                       # styles.js (stamped AFTER the index) in every page
        html = (built / page).read_text()
        assert f'"/styles.js?v={sha12(built / "styles.js")}"' in html, page
    for page in ("index.html", "benchmarks.html"):
        assert f'"/palette.js?v={sha12(built / "palette.js")}"' in (built / page).read_text(), page


def test_a_missing_stamp_fails_the_build(tmp_path):
    """A sheet whose name the index does not carry would ship unversioned: the step refuses."""
    app = tmp_path / "app" / "vllm_lb"
    shutil.copytree(STATIC, app / "static")
    (app / "static" / "styles" / "stray.css").write_text(':root[data-style="stray"] { --bg: #000; }\n')
    r = subprocess.run(["sh", "-c", stamp_steps()[0].replace("/app/vllm_lb", str(app))], capture_output=True, text=True)
    assert r.returncode != 0


# ── every page ──────────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("page", PAGES)
def test_every_page_applies_the_style_before_it_paints(page):
    html = (STATIC / page).read_text()
    head = html[:html.index("</head>")]
    assert '<script src="/styles.js"></script>' in head, "in <head>: the stored style is applied before first paint"
    assert head.index('href="/gateway.css"') < head.index('"/styles.js"'), "after the sheets it overrides"
    assert 'id="style-slot"' in html and 'id="style-notice"' in html
    body = html[html.index("</head>"):]
    for script in ("gateway.js", "benchmarks.js"):
        if f'"/{script}"' in body:
            assert '"/palette.js"' in body and body.index('"/palette.js"') < body.index(f'"/{script}"'), page


# ── the names ───────────────────────────────────────────────────────────────────────────────────────

# A style is named for what it looks like, never for where its look came from (operator): no site, company, product or
# typeface name. Each word is assembled, so this file does not carry it either. A font stack's `-apple-system` keyword
# is CSS, not a name: the vendor's word is banned only standing alone.
BANNED_NAMES = [
    re.compile("open" + r"[\s_-]?" + "wear" + "ables", re.I),
    re.compile(r"(?<![-\w])" + "app" + r"le(?![-\w])", re.I),
    re.compile(r"\b" + "mac" + r"\s?os\b", re.I),
    re.compile(r"\bi" + "OS" + r"\b"),
    re.compile(r"\bSF[\s-]?" + "Pro" + r"\b", re.I),
    # the green-phosphor terminal style's film: its title used as a look ("… style", "… rain"), its makers, its people
    # and places. The bare noun stays: the panel has a core matrix and a benchmark matrix of its own.
    re.compile("mat" + r"rix[\s_-]?(style|green|rain|theme|look|terminal|code|effect)\b", re.I),
    re.compile(r"\b(" + "|".join(["wach" + "owski", "mor" + "pheus", "zi" + "on", "nebuchad" + "nezzar",
                                   "white " + "rabbit", "(red|blue) " + "pill"]) + r")\b", re.I),
]


def test_no_source_name_anywhere_in_the_shipped_tree():
    for p in STATIC.rglob("*"):
        for banned in BANNED_NAMES:
            assert not banned.search(p.name), p
            if p.is_file() and p.suffix in {".css", ".js", ".json", ".html", ".txt", ".md"}:
                hit = banned.search(p.read_text(errors="replace"))
                assert not hit, f"{p.relative_to(STATIC)}: {hit.group(0)!r}"


def test_the_banned_names_catch_what_they_must_and_spare_the_font_keyword():
    stack = '--sans: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui;'
    assert not any(b.search(stack) for b in BANNED_NAMES)
    for word in ("App" + "le", "mac" + "OS", "Mac " + "OS", "i" + "OS", "SF " + "Pro", "SF" + "-Pro",
                 "Mat" + "rix-style", "mat" + "rix rain", "Mor" + "pheus", "red " + "pill", "Wach" + "owski"):
        assert any(b.search(f"a style like {word} apps") for b in BANNED_NAMES), word
    assert not any(b.search("ratios bios scenarios") for b in BANNED_NAMES)
    for fine in ("the core matrix", "renderPerfMatrix", "the matrix holds the pair", "a hold-to-start pill", "region"):
        assert not any(b.search(fine) for b in BANNED_NAMES), fine


# ── a style that follows the system appearance ──────────────────────────────────────────────────────

GLASS = STYLES / "glass.css"


def test_glass_follows_the_system_appearance():
    """"Glass" is light or dark as the system is: its scope block declares the light tokens and, nested in it, ONE
    `@media (prefers-color-scheme: dark)` block re-values every surface, ink and meaning token palette.js and the
    sheets read — a token the dark block forgot would draw a light-appearance colour on a dark card."""
    scope = [r for r in top_rules(GLASS.read_text()) if r["prelude"] == ':root[data-style="glass"]']
    assert len(scope) == 1
    light = dict(own_declarations(scope[0]["body"]))
    dark_blocks = [r for r in top_rules(scope[0]["body"]) if "prefers-color-scheme" in r["prelude"]]
    assert [r["prelude"] for r in dark_blocks] == ["@media (prefers-color-scheme: dark)"], "one nested dark block"
    dark = dict(own_declarations(dark_blocks[0]["body"]))
    assert light["color-scheme"] == "light" and dark["color-scheme"] == "dark"
    for name in ("--bg", "--panel", "--panel-2", "--surface-3", "--line", "--line-strong", "--ink", "--ink-dim",
                 "--ink-faint", "--ok", "--ok-rgb", "--warn", "--warn-rgb", "--bad", "--bad-rgb", "--accent",
                 "--accent-rgb", "--park", "--park-rgb", "--load-idle", "--load-ok", "--load-warn", "--load-bad",
                 "--rate-rest", "--rate-mid", "--rate-high", "--rate-target", "--core-idle", "--tag-vllm",
                 "--tag-sglang", "--tag-tensorfold", "--tag-drafter", "--neutral-rgb", "--ink-faint-rgb", "--track"):
        assert name in light, f"light appearance: {name}"
        assert name in dark and dark[name] != light[name], f"dark appearance: {name}"
    assert light["--scale-space"] == "oklch" and light["--glow"] == "0"


# ── a phosphor terminal: one dark appearance, one family, motion that stops when asked ─────────────────────────────

TERMINAL = STYLES / "terminal.css"
TERMINAL_SCOPE = ':root[data-style="terminal"]'


def nested(body: str):
    """Every rule nested anywhere in a body, as (prelude, its own declarations)."""
    for r in top_rules(body):
        yield r["prelude"], dict(own_declarations(r["body"]))
        yield from nested(r["body"])


def test_terminal_is_one_dark_phosphor_appearance_in_one_family():
    scope = [r for r in top_rules(TERMINAL.read_text()) if r["prelude"] == TERMINAL_SCOPE]
    assert len(scope) == 1
    tokens = dict(own_declarations(scope[0]["body"]))
    assert tokens["color-scheme"] == "dark" and tokens["--bg"] == "#000000", "a black page, one appearance"
    assert not [p for p, _ in nested(scope[0]["body"]) if "prefers-color-scheme" in p], "no light appearance"
    assert tokens["--sans"] == "var(--mono)" and tokens["--sans-ui"] == "var(--mono)", "one family everywhere"
    assert tokens["--mono"].startswith('"JetBrains Mono"'), "the vendored face first"
    assert tokens["--scale-space"] == "oklch" and tokens["--glow"] == "1", "glowing, mixed in oklch"


def test_terminal_motion_stops_under_reduced_motion_and_its_overlay_takes_no_click():
    css = TERMINAL.read_text()
    rules = list(nested(css))
    scope_body = next(r["body"] for r in top_rules(css) if r["prelude"] == TERMINAL_SCOPE)
    moving = set()
    for p, d in nested(scope_body):
        for k, v in d.items():
            if k in ("animation", "animation-name") and v != "none":
                assert re.match(r"terminal-[a-z-]+\b", v), f"{p}: an animation outside the sheet's own keyframes: {v}"
                moving.update(TERMINAL_SCOPE + " " + s.strip().removeprefix("&").strip() for s in p.split(","))
    assert moving, "the cursor blinks"
    reduced = [r for r in top_rules(css) if r["prelude"] == "@media (prefers-reduced-motion: reduce)"]
    assert len(reduced) == 1, "one reduced-motion block"
    still = {s.strip(): d for p, d in nested(reduced[0]["body"]) for s in p.split(",")}
    for sel in moving:
        assert still.get(sel, {}).get("animation") == "none", f"{sel} keeps moving under reduced motion"
    overlays = [(p, d) for p, d in rules if "repeating-linear-gradient" in d.get("background", "")
                and d.get("position") == "fixed"]
    assert overlays, "the scanlines are a fixed overlay"
    for p, d in overlays:
        assert d.get("pointer-events") == "none" and "::" in p, f"{p}: an overlay must be a pseudo-element no click hits"


# ── the browser-side halves ─────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node is not in the LB image: run tests/styles.node.js and tests/palette.node.js under node:22-slim")
@pytest.mark.parametrize("script", ["styles.node.js", "palette.node.js"])
def test_the_switch_and_the_palette_under_node(script):
    r = subprocess.run(["node", str(HERE / script)], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ok" in r.stdout
