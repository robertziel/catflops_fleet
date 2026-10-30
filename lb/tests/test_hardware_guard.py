"""GUARDS: a hardware fact lives in the hardware domain and nowhere else (AGENTS.md, "Hardware").

WHAT THEY HOLD the LB's executed code to — `lb/vllm_lb/**/*.py` outside `lb/vllm_lb/hardware/`, and the panel's
`lb/vllm_lb/static/*.js`:

  1. no hardware IDENTITY as a literal: a card or chip name (GB10, RTX 5090, Radeon …), an arch (sm_121, gfx1151,
     aarch64 …), a memory technology (LPDDR5X, GDDR7) or a hardware type's key (`gb10`, `rtx5090` …) — a box's cards
     are the fleet file's, and what a card IS is its type file's (vllm_lb/hardware/types);
  2. no hardware FIGURE as a literal: any number a type file states (its CUDA cores, peak GB/s, dense TFLOP/s) —
     the GB10's 6,144 cores sat in three model shapes until 2026-10-11;
  3. no BRANCH on a memory family: `if ….unified:`, `isinstance(x, UnifiedProfile)` — a family's behaviour is its
     class's (vllm_lb.hardware), never a caller's conditional;
  4. the panel never asks which kind of box it draws: no `mem_unified`, no 'unified'/'discrete' compared — the
     server lays a box's GPU rows out (hw.py `rows`) and the page draws what it is given.

What they do NOT read: comments and docstrings (prose may name hardware — it explains), the hardware package itself
and its YAML, tests, vendored code (chat/vendor — never edited here), the box-side sender (lb/fleet_telemetry, which
reads the GPU tools by the kind a server names), and the other services
(power/), which do not import the LB's hardware domain (their hardware literals are proposals in the report).

Each scanner is proven against PLANTED violations below (and was proven against a real plant in the tree once).
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from vllm_lb.hardware import FAMILIES, MemoryModel, accelerators, types

LB = Path(__file__).resolve().parent.parent
PACKAGE = LB / "vllm_lb"
HARDWARE = PACKAGE / "hardware"
STATIC = PACKAGE / "static"

#: Hardware identities that are never a code literal outside the domain — matched case-insensitively as whole words.
_IDENTITY = (r"gb10|dgx|geforce|rtx ?\d{4}|radeon|rdna\d?|sm_?1[0-2]\d|gfx\d{3,4}|rocm|aarch64|arm64|x86_64|amd64"
             r"|lpddr\w*|gddr\w*")
#: …and every hardware type's key and every ACCELERATOR's key (accelerators/<key>.yml, 2026-10-30): an image, a
#: measurement or a recipe is keyed by accelerator in DATA (vllm_lb/library), never in code.
IDENTITY = re.compile(rf"(?i)(?<![a-z0-9])(?:{_IDENTITY}|{'|'.join(map(re.escape, [*types(), *accelerators()]))})"
                      rf"(?![a-z0-9])")

#: Every figure a hardware type states, but its memory size (128, 32, 24 GB are everyday numbers — a 32 is not a
#: 5090) — and, for the same reason, a measured memory figure that is a WHOLE number of GiB (the Radeon 8060S's pool,
#: 124.0, is also `timeout`'s exit code 124; its 119.26 and the 5090's 31.36 / 30.85 stay guarded): derived from the
#: library, so a new type's figures are guarded the day its file lands.
FIGURES = frozenset(
    f.value for p in types().values()
    for f in (p.cuda_cores, p.peak_gbps, *p.peak_tflops.values(),
              *(m for m in (p.visible_gib, p.free_at_start_gib)
                if not (isinstance(m.value, (int, float)) and float(m.value).is_integer())))
    if isinstance(f.value, (int, float)) and not isinstance(f.value, bool))

#: The behaviour-family classes a caller may not branch on.
FAMILY_NAMES = frozenset({c.__name__ for c in FAMILIES.values()}
                         | {c.__name__ for c in MemoryModel.__subclasses__()})


# ── the scanners ────────────────────────────────────────────────────────────────────────────────────

def _docstrings(tree: ast.AST) -> set[int]:
    out = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and body \
                and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                and isinstance(body[0].value.value, str):
            out.add(id(body[0].value))
    return out


def python_literals(source: str) -> list[tuple[int, str]]:
    """(line, what) for every hardware identity or figure written as a literal in EXECUTED code — docstrings and
    comments excluded (the parser never sees a comment)."""
    tree = ast.parse(source)
    skip, out = _docstrings(tree), []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or id(node) in skip:
            continue
        v = node.value
        if isinstance(v, str) and IDENTITY.search(v):
            out.append((node.lineno, f"identity {IDENTITY.search(v).group(0)!r} in {v[:60]!r}"))
        elif isinstance(v, (int, float)) and not isinstance(v, bool) and v in FIGURES:
            out.append((node.lineno, f"figure {v!r}"))
    return out


def python_branches(source: str) -> list[tuple[int, str]]:
    """(line, what) for every conditional that asks a memory family: a test reading `.unified`, or an
    isinstance()/`type(…) is` against a family class."""
    out = []
    for node in ast.walk(ast.parse(source)):
        tests = []
        if isinstance(node, (ast.If, ast.IfExp, ast.While)):
            tests.append(node.test)
        elif isinstance(node, ast.comprehension):
            tests.extend(node.ifs)
        elif isinstance(node, ast.match_case) and node.guard is not None:
            tests.append(node.guard)
        for test in tests:
            for sub in ast.walk(test):
                if isinstance(sub, ast.Attribute) and sub.attr == "unified":
                    out.append((sub.lineno, "branches on `.unified`"))
                elif isinstance(sub, ast.Name) and sub.id in FAMILY_NAMES:
                    out.append((sub.lineno, f"branches on the family {sub.id}"))
                elif isinstance(sub, ast.Attribute) and sub.attr in FAMILY_NAMES:
                    out.append((sub.lineno, f"branches on the family {sub.attr}"))
    return out


_REGEX_AFTER = set("(,=:[!&|?{};+-*%<>~^")
_REGEX_KEYWORDS = ("return", "typeof", "case", "in", "of", "delete", "void", "throw", "new", "do", "else")


def js_code(source: str) -> tuple[str, list[tuple[int, str]]]:
    """The panel's source with every comment, string and regex literal blanked (newlines kept, so line numbers hold),
    and those literals themselves as (line, text). A small lexer: quotes, template literals with their `${…}`
    expressions, regex literals (a `/` where a value cannot end), line and block comments."""
    code: list[str] = []
    literals: list[tuple[int, str]] = []
    i, n, line = 0, len(source), 1
    depth: list[int] = []        # open `${…}` inside templates: the brace depth at which each one closes
    last, word = "", ""          # the last significant code character ('x' after a literal) and identifier

    def blank(text: str) -> str:
        return "".join(ch if ch == "\n" else " " for ch in text)

    while i < n:
        ch, nxt = source[i], source[i + 1] if i + 1 < n else ""
        if ch == "/" and nxt in "/*":
            j = (source.find("\n", i) if nxt == "/" else source.find("*/", i + 2) + 2)
            j = n if j < (i + 2 if nxt == "*" else 0) or j == -1 else j
            code.append(blank(source[i:j])); line += source[i:j].count("\n"); i = j
            continue
        resume = ch == "}" and depth and depth[-1] == 0
        if ch in "'\"`" or resume:
            quote = "`" if resume else ch
            if resume:
                depth.pop()
            j = i + 1
            while j < n:
                c = source[j]
                if c == "\\":
                    j += 2
                    continue
                if c == quote:
                    j += 1
                    break
                if quote == "`" and c == "$" and source[j + 1:j + 2] == "{":
                    depth.append(0)
                    j += 2
                    break
                j += 1
            literals.append((line, source[i:j]))
            code.append(blank(source[i:j])); line += source[i:j].count("\n"); i = j
            last, word = "x", ""
            continue
        if ch == "/" and (not last or last in _REGEX_AFTER or (word in _REGEX_KEYWORDS and last.isalpha())):
            j, klass = i + 1, False
            while j < n and source[j] != "\n":
                c = source[j]
                if c == "\\":
                    j += 2
                    continue
                if c == "[":
                    klass = True
                elif c == "]":
                    klass = False
                elif c == "/" and not klass:
                    j += 1
                    break
                j += 1
            while j < n and source[j].isalpha():
                j += 1                               # the flags
            literals.append((line, source[i:j]))
            code.append(blank(source[i:j])); i = j
            last, word = "x", ""
            continue
        if depth and ch == "{":
            depth[-1] += 1
        elif depth and ch == "}":
            depth[-1] -= 1
        code.append(ch)
        if ch == "\n":
            line += 1
        if ch.isalnum() or ch in "_$":
            word = word + ch if last and (last.isalnum() or last in "_$") else ch
            last = ch
        elif not ch.isspace():
            last, word = ch, ""
        i += 1
    return "".join(code), literals


def js_findings(source: str) -> list[tuple[int, str]]:
    """(line, what): a memory-family question, a hardware identity in a string, or a hardware figure in code."""
    code, strings = js_code(source)
    out = []
    for lineno, text in enumerate(code.splitlines(), 1):
        if re.search(r"\bmem_unified\b", text):
            out.append((lineno, "asks `mem_unified`"))
        for m in re.finditer(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w.])", text):
            if float(m.group(1)) in FIGURES:
                out.append((lineno, f"figure {m.group(1)}"))
    for lineno, text in strings:
        inner = text[1:-1] if len(text) >= 2 else text
        if IDENTITY.search(inner):
            out.append((lineno, f"identity {IDENTITY.search(inner).group(0)!r}"))
        if inner in ("unified", "discrete"):
            out.append((lineno, f"compares the memory family {inner!r}"))
    return out


def _py_files():
    return sorted(p for p in PACKAGE.rglob("*.py") if HARDWARE not in p.parents and "__pycache__" not in p.parts)


# ── the guards ──────────────────────────────────────────────────────────────────────────────────────

def test_no_hardware_identity_or_figure_is_a_python_literal_outside_the_hardware_domain():
    found = [f"{p.relative_to(LB)}:{line}: {what}" for p in _py_files() for line, what in python_literals(p.read_text())]
    assert found == [], "a hardware fact belongs in vllm_lb/hardware (a type file), not here:\n" + "\n".join(found)


def test_no_python_outside_the_hardware_domain_branches_on_a_memory_family():
    found = [f"{p.relative_to(LB)}:{line}: {what}" for p in _py_files() for line, what in python_branches(p.read_text())]
    assert found == [], "ask the memory model (vllm_lb.hardware), never branch on its kind:\n" + "\n".join(found)


def test_the_panel_never_asks_which_kind_of_box_and_names_no_hardware():
    found = [f"{p.relative_to(LB)}:{line}: {what}" for p in sorted(STATIC.glob("*.js"))
             for line, what in js_findings(p.read_text())]
    assert found == [], "the server lays a box out (hw.py `rows`); the page draws it:\n" + "\n".join(found)


def test_the_guards_scan_what_they_claim_to():
    assert len(_py_files()) > 30 and HARDWARE / "profiles.py" not in _py_files()
    assert {"gateway.js", "benchmarks.js", "sessions.js"} <= {p.name for p in STATIC.glob("*.js")}
    assert {6144, 273, 323.2, 21760, 1792, 1676, 31.36, 30.85} <= FIGURES and 128 not in FIGURES
    assert {"UnifiedProfile", "DiscreteProfile", "UnifiedMemory", "DiscreteMemory"} <= FAMILY_NAMES


# ── the scanners, proven against planted violations ─────────────────────────────────────────────────

@pytest.mark.parametrize("planted", [
    'CORES = 6144\n',
    'PEAK = {"gbps": 273}\n',
    'def f(card):\n    return card == "gb10"\n',
    'NAME = "NVIDIA GeForce RTX 5090"\n',
    'ARCH = f"built for sm_121 on {box}"\n',
    'KEYS = ("rtx3090", "rtx5090")\n',
    'TARGET = "nvidia-sm86"\n',                       # an accelerator key (accelerators/<key>.yml)
    'if acc.key == "amd-gfx1151":\n    pass\n',
    'FREE_AT_START_GIB = 30.85\n',
])
def test_the_python_literal_scanner_catches_a_plant(planted):
    assert python_literals(planted), planted


@pytest.mark.parametrize("clean", [
    '"""A GB10 has 6,144 CUDA cores and 273 GB/s."""\n',
    '# 6144 cores on a gb10 (sm_121)\nx = 1\n',
    'def f():\n    """sm_121a and GDDR7, in prose."""\n    return 128\n',
])
def test_the_python_literal_scanner_ignores_prose(clean):
    assert python_literals(clean) == [], clean


@pytest.mark.parametrize("planted", [
    'def f(m):\n    if m.unified:\n        return 1\n',
    'def f(p):\n    return 1 if isinstance(p, UnifiedProfile) else 2\n',
    'def f(rows):\n    return [r for r in rows if r.memory.unified]\n',
    'def f(p):\n    while type(p) is hardware.DiscreteProfile:\n        break\n',
])
def test_the_branch_scanner_catches_a_plant(planted):
    assert python_branches(planted), planted


def test_the_branch_scanner_lets_a_model_answer_for_its_kind():
    assert python_branches('def f(m, s, g):\n    used, total = m.read(s, g)\n    return {"u": m.unified}\n') == []


@pytest.mark.parametrize("planted", [
    "if (!h.mem_unified && h.mem_total_mb) figs.push(['VRAM', x]);\n",
    "const cores = 6144;\n",
    "const label = `GPU on a GB10 ${box}`;\n",
    "if (h.memory === 'discrete') rows();\n",
    "const kind = \"rtx5090\";\n",
])
def test_the_js_scanner_catches_a_plant(planted):
    assert js_findings(planted), planted


@pytest.mark.parametrize("clean", [
    "// a GB10 has 6144 cores; mem_unified says so\nconst x = 1;\n",
    "/* rtx5090: 1792 GB/s\n   mem_unified */ const y = 2;\n",
    "const re = /mem_unified|6144/g; const z = a / b / 6;\n",
    "const s = 'http://x/y'; // gb10\n",
    "const t = `${a ? `GPU ${b}` : 'GPU'}`;\n",
])
def test_the_js_scanner_ignores_comments_regexes_and_ordinary_code(clean):
    assert js_findings(clean) == [], (clean, js_findings(clean))
