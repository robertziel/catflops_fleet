"""The one place this package touches anything outside itself — and the only shape it can take.

Two chokepoints, both deliberately unparameterised in the direction that would make them dangerous:

* `lb_get(path)` issues the *literal* string "GET". No function in this package accepts an HTTP
  method, builds one from a variable, or carries a request body, so there is no code path that
  reaches `POST /lb/cluster` (start/stop the swarm), `POST /lb/power` or any other control route.
  `path` is checked against `LB_PATHS`, a frozenset of three documented read endpoints.
* `read_text(path)` opens the *literal* mode "r" after resolving the path inside a frozen root. The
  package imports no `subprocess`, `shutil`, `os.remove`, `tempfile` or `urllib.request`, and calls
  no `.write()`.

`tests/test_readonly.py` re-derives all of that from this package's own AST, so a future edit that
adds a write or a second verb fails the suite rather than shipping quietly. The container adds a
third layer (`read_only: true`, `cap_drop: [ALL]`, every mount `:ro`) — see docker-compose.yml.
"""
from __future__ import annotations

import http.client
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlsplit, urlunsplit

#: Every LB endpoint this server may read. All three are documented GETs that return a JSON
#: document and change nothing. A path outside this set is refused before a socket is opened.
LB_PATHS = frozenset({"/lb/status", "/lb/perf", "/lb/bench"})

#: The file reader accepts these modes and no others.
READ_MODES = frozenset({"r", "rt"})

DEFAULT_LB_URL = "http://127.0.0.1:8010"
DEFAULT_STATE_DIR = "/data"
#: The LB's MODEL LIBRARY directory (lb/vllm_lb/library, mounted read-only): models.json, recipes.json and
#: measurements.json — the split of the retired catalog.json (2026-10-30).
DEFAULT_CATALOG = "/library"
#: Environment names that are retired, each with what replaced it: refused, never silently ignored.
RETIRED_ENV = {"USAGE_MCP_CATALOG": "USAGE_MCP_LIBRARY (the model library directory; catalog.json is split into "
                                    "library/models.json, recipes.json, measurements.json)"}
DEFAULT_TIMEOUT_S = 8.0


class ReadOnlyViolation(RuntimeError):
    """Raised instead of reaching outside the allowlist. Never caught inside this package."""


class SourceUnavailable(RuntimeError):
    """The source exists but could not be read right now (LB down, file absent).

    Callers turn this into a plain sentence in a tool result: the cluster being off, or the LB
    being down, is a legitimate answer and not a tool error.
    """


@dataclass(frozen=True)
class Sources:
    """Where the data is. Frozen: a tool cannot repoint the server at another host mid-call."""

    lb_url: str = DEFAULT_LB_URL
    state_dir: Path = Path(DEFAULT_STATE_DIR)
    catalog: Path = Path(DEFAULT_CATALOG)
    timeout_s: float = DEFAULT_TIMEOUT_S

    @staticmethod
    def from_env(env: dict[str, str] | None = None) -> "Sources":
        e = os.environ if env is None else env
        stale = [k for k in RETIRED_ENV if k in e]
        if stale:
            raise ValueError("; ".join(f"{k} is retired: use {RETIRED_ENV[k]}" for k in stale))
        return Sources(
            lb_url=e.get("USAGE_MCP_LB_URL", DEFAULT_LB_URL).rstrip("/"),
            state_dir=Path(e.get("USAGE_MCP_STATE_DIR", DEFAULT_STATE_DIR)),
            catalog=Path(e.get("USAGE_MCP_LIBRARY", DEFAULT_CATALOG)),
            timeout_s=float(e.get("USAGE_MCP_TIMEOUT_S", DEFAULT_TIMEOUT_S)),
        )

    @property
    def roots(self) -> tuple[Path, ...]:
        return (self.state_dir, self.catalog)


# ── outbound: GET, three paths, nothing else ──────────────────────────────────────────────────
def lb_get(path: str, src: Sources) -> Any:
    """GET one allowlisted LB endpoint and parse its JSON. The verb is a literal, not an argument."""
    if path not in LB_PATHS:
        raise ReadOnlyViolation(f"{path!r} is not one of the read endpoints {sorted(LB_PATHS)}")
    parts = urlsplit(src.lb_url if "//" in src.lb_url else "//" + src.lb_url)
    if parts.scheme not in ("", "http"):
        raise ReadOnlyViolation(f"only plain http is read; got {parts.scheme!r}")
    host, port = parts.hostname or "127.0.0.1", parts.port or 8010
    conn = http.client.HTTPConnection(host, port, timeout=src.timeout_s)
    try:
        conn.request("GET", path)
        resp = conn.getresponse()
        body = resp.read()
        if resp.status != 200:
            raise SourceUnavailable(f"GET {path} -> HTTP {resp.status}")
        return json.loads(body.decode("utf-8", "replace"))
    except (OSError, ValueError) as exc:
        raise SourceUnavailable(f"GET {src.lb_url}{path}: {type(exc).__name__}: {exc}") from exc
    finally:
        conn.close()


# ── filesystem: mode "r", inside a frozen root ────────────────────────────────────────────────
def _confine(path: Path, src: Sources) -> Path:
    resolved = Path(path).resolve()
    if resolved == Path(src.catalog).resolve():
        return resolved
    for root in src.roots:
        try:
            root_r = Path(root).resolve()
        except OSError:                                        # pragma: no cover - unreadable root
            continue
        if resolved == root_r or root_r in resolved.parents:
            return resolved
    raise ReadOnlyViolation(f"{path} is outside the read roots {[str(r) for r in src.roots]}")


def read_text(path: Path, src: Sources, mode: str = "r") -> str:
    """Read one confined file as text. Undecodable bytes are replaced, never raised."""
    if mode not in READ_MODES:
        raise ReadOnlyViolation(f"mode {mode!r} is not a read mode")
    target = _confine(path, src)
    try:
        with open(target, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError as exc:
        raise SourceUnavailable(f"read {target}: {type(exc).__name__}: {exc}") from exc


def read_json(path: Path, src: Sources) -> Any:
    try:
        return json.loads(read_text(path, src))
    except ValueError as exc:
        raise SourceUnavailable(f"{path} is not JSON: {exc}") from exc


def jsonl(path: Path, src: Sources) -> Iterator[tuple[int, Any | None]]:
    """Yield `(line_number, record_or_None)`. A malformed line yields None instead of raising.

    The fleet has one in the real log: perf-2026-10-02.jsonl line 470 is a run of NUL bytes from a
    crash mid-append. Every reader counts those rather than losing the day's windows to them.
    """
    for n, line in enumerate(read_text(path, src).splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            yield n, json.loads(line)
        except ValueError:
            yield n, None


def state_glob(pattern: str, src: Sources, subdir: str = "") -> list[Path]:
    """Sorted matches for a glob inside the state root. Missing directory -> empty list."""
    if ".." in pattern or pattern.startswith("/") or ".." in subdir or subdir.startswith("/"):
        raise ReadOnlyViolation(f"glob {subdir}/{pattern} escapes the state root")
    base = Path(src.state_dir) / subdir if subdir else Path(src.state_dir)
    try:
        found = sorted(base.glob(pattern))
    except OSError:                                            # pragma: no cover - unreadable dir
        return []
    return [p for p in found if _safe(p, src)]


def state_dirs(subdir: str, src: Sources) -> list[Path]:
    """Sorted sub-directories of one state directory. Missing directory -> empty list."""
    if ".." in subdir or subdir.startswith("/"):
        raise ReadOnlyViolation(f"{subdir} escapes the state root")
    base = Path(src.state_dir) / subdir
    try:
        found = sorted(p for p in base.iterdir() if p.is_dir())
    except OSError:
        return []
    return [p for p in found if _safe(p, src)]


def _safe(path: Path, src: Sources) -> bool:
    try:
        _confine(path, src)
    except ReadOnlyViolation:
        return False
    return True


def mtime(path: Path, src: Sources) -> float | None:
    try:
        return _confine(path, src).stat().st_mtime
    except (OSError, ReadOnlyViolation):
        return None


def scrub_url(url: Any) -> Any:
    """Drop any `user:password@` from a URL before it leaves in a tool result.

    The LB's own status document carries no credentials today (checked against lb/.env's PG_PW),
    but a tool result is a place a secret must never be able to appear, so every URL this server
    emits goes through here rather than relying on that staying true.
    """
    if not isinstance(url, str) or "//" not in url:
        return url
    parts = urlsplit(url)
    if not parts.hostname:
        return url
    netloc = parts.hostname + (f":{parts.port}" if parts.port else "")
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))
