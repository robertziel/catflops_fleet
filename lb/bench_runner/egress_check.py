#!/usr/bin/env python3
"""THE SEAL, PROVEN — run before the sealed runner starts and again before EVERY session it launches.

`claude` is never started unless this has just passed (bench_queue_runner.py `--seal-check`; the
entrypoint runs it first). It tries the things that must fail, and passes only if every one of them did:

  credentials  no OAuth or API-key file in HOME (~/.claude/.credentials.json, an `oauthAccount` or
               `primaryApiKey` in ~/.claude.json, AWS or gcloud credentials), no `apiKeyHelper` in any
               settings file, and no backend or cloud credential variable in this environment. A
               credential is NAMED, never printed.
  direct       Anthropic — by name and by its published addresses — and a public address are dialled
               with no proxy: none may connect. A connection is a network that is not sealed.
  proxy        the egress proxy is asked for Anthropic's names and ranges, every private/LAN range,
               Tailscale's CGNAT, link-local metadata, loopback and an in-network name: each must be
               answered 403. A 200 is the hole itself; a 502 means the proxy TRIED to dial what policy
               must refuse unasked; no answer is a proxy that cannot prove anything.
  lb           the LB answers /healthz through its relay — the one thing the runner is for.
  public       (`--public-url`, by hand, never before a launch) one public page fetched THROUGH the
               proxy: the seal still lets the research read the web.

The addresses come from the same variables the entrypoint builds the runner's command from
(BENCH_EGRESS_PROXY, BENCH_LB_URL) — one definition each — or from `--proxy` / `--lb`.

    python3 -m bench_runner.egress_check                          (what the runner runs)
    python3 -m bench_runner.egress_check --public-url https://example.com/
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import socket
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

#: Dialled with NO proxy: Anthropic by name, by its published v4 and v6 addresses (policy.py cites the
#: page), and a public resolver's address, which proves there is no route out at all.
DIRECT = (("api.anthropic.com", 443), ("claude.ai", 443), ("claude.com", 443),
          ("160.79.104.10", 443), ("2607:6bc0::10", 443), ("1.1.1.1", 443))
#: Asked of the proxy, each of which must be answered 403.
VIA_PROXY = (("api.anthropic.com", 443), ("claude.ai", 443), ("claude.com", 443),
             ("statsig.anthropic.com", 443), ("platform.claude.com", 443),
             ("160.79.104.10", 443), ("[2607:6bc0::10]", 443), ("api.anthropic.com", 80),
             ("10.0.0.1", 443), ("172.16.0.1", 443), ("192.168.0.1", 443), ("100.64.0.1", 443),
             ("169.254.169.254", 443), ("127.0.0.1", 443), ("[::1]", 443), ("localhost", 443),
             ("lb", 443), ("engine", 443), ("lb", 8010))
#: Credential files, relative to HOME.
CREDENTIAL_FILES = (".claude/.credentials.json", ".aws/credentials", ".aws/config",
                    ".config/gcloud/application_default_credentials.json", ".config/gcloud/credentials.db")
#: Keys in ~/.claude.json that mean someone logged in, or approved a real key.
CLAUDE_JSON_KEYS = ("oauthAccount", "primaryApiKey")
#: Variables that are a credential, or that point Claude Code at a paid backend.
CREDENTIAL_VARS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN",
                   "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX", "AWS_ACCESS_KEY_ID",
                   "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_PROFILE", "AWS_BEARER_TOKEN_BEDROCK",
                   "GOOGLE_APPLICATION_CREDENTIALS")
#: Settings files Claude Code reads: managed first. HOME's own are added at run time.
MANAGED_SETTINGS = "/etc/claude-code/managed-settings.json"


@dataclass(frozen=True)
class Result:
    kind: str
    target: str
    ok: bool
    said: str


def default_dial(address, timeout=None):
    return socket.create_connection(address, timeout=timeout)


# ── the checks ─────────────────────────────────────────────────────────────


def credentials(home: str, environ: dict, settings_files) -> list[Result]:
    out: list[Result] = []
    root = pathlib.Path(home)
    for rel in CREDENTIAL_FILES:
        present = (root / rel).exists()
        out.append(Result("credentials", f"~/{rel}", not present,
                          "PRESENT — a credential is within the runner's reach" if present else "absent"))
    found: list[str] = []
    try:
        data = json.loads((root / ".claude.json").read_text())
        if isinstance(data, dict):
            found = [k for k in CLAUDE_JSON_KEYS if data.get(k)]
    except (OSError, ValueError):
        pass
    out.append(Result("credentials", "~/.claude.json", not found,
                      f"carries {', '.join(found)} — someone logged in or approved a key" if found
                      else "no login, no approved key"))
    named = sorted(k for k in CREDENTIAL_VARS if environ.get(k))
    out.append(Result("credentials", "environment", not named,
                      f"{', '.join(named)} set" if named else "no backend or cloud credential variable"))
    for path in settings_files:
        try:
            data = json.loads(pathlib.Path(path).read_text())
        except OSError:
            continue
        except ValueError as exc:
            out.append(Result("credentials", path, False, f"unreadable settings: {exc}"))
            continue
        helper = isinstance(data, dict) and bool(data.get("apiKeyHelper"))
        out.append(Result("credentials", path, not helper,
                          "defines apiKeyHelper — a key could be fetched" if helper else "no apiKeyHelper"))
    return out


def direct(host: str, port: int, dial, timeout: float) -> Result:
    target = f"{host}:{port}"
    try:
        conn = dial((host, port), timeout=timeout)
    except OSError as exc:
        return Result("direct", target, True, f"no connection ({type(exc).__name__}: {exc})")
    try:
        conn.close()
    except OSError:
        pass
    return Result("direct", target, False, "CONNECTED — this network has a route out; it is not sealed")


def via_proxy(proxy: tuple[str, int], host: str, port: int, timeout: float) -> Result:
    target = f"{host}:{port}"
    try:
        with socket.create_connection(proxy, timeout=timeout) as s:
            s.sendall(f"CONNECT {target} HTTP/1.1\r\nHost: {target}\r\n\r\n".encode())
            head = b""
            while b"\r\n\r\n" not in head and len(head) < 16384:
                data = s.recv(4096)
                if not data:
                    break
                head += data
    except OSError as exc:
        return Result("proxy", target, False, f"the proxy did not answer ({type(exc).__name__}: {exc})")
    lines = head.decode("latin-1").split("\r\n")
    parts = lines[0].split(" ", 2)
    code = parts[1] if len(parts) > 1 else "?"
    reason = next((ln.split(":", 1)[1].strip() for ln in lines if ln.lower().startswith("x-egress-reason:")), "")
    if code == "403":
        return Result("proxy", target, True, f"403 {reason}".strip())
    return Result("proxy", target, False, f"answered {lines[0] or 'nothing'} — must be refused (403)")


def lb_answers(lb_url: str, timeout: float) -> Result:
    url = lb_url.rstrip("/") + "/healthz"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url, timeout=timeout) as r:
            return Result("lb", url, 200 <= r.status < 300, f"{r.status}")
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return Result("lb", url, False, f"no answer ({type(exc).__name__}: {exc})")


def public_fetch(proxy_url: str, url: str, timeout: float) -> Result:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({"https": proxy_url}))
    try:
        with opener.open(url, timeout=timeout) as r:
            body = r.read(65536)
            return Result("public", url, 200 <= r.status < 400, f"{r.status}, {len(body)} bytes through the proxy")
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return Result("public", url, False, f"not fetched ({type(exc).__name__}: {exc})")


def _proxy_address(proxy_url: str) -> tuple[str, int]:
    parts = urllib.parse.urlsplit(proxy_url)
    if parts.scheme != "http" or not parts.hostname or not parts.port:
        raise ValueError(f"the proxy must be http://host:port, not {proxy_url!r}")
    return parts.hostname, parts.port


def run(proxy_url: str, lb_url: str, *, home: str, dial=None, environ=None, settings_files=(),
        timeout: float = 4.0, public_url: str | None = None) -> list[Result]:
    dial = dial or default_dial
    environ = os.environ if environ is None else environ
    proxy = _proxy_address(proxy_url)
    results = credentials(home, environ, settings_files)
    results += [direct(h, p, dial, timeout) for h, p in DIRECT]
    results += [via_proxy(proxy, h, p, timeout) for h, p in VIA_PROXY]
    results.append(lb_answers(lb_url, timeout))
    if public_url:
        results.append(public_fetch(proxy_url, public_url, max(timeout, 20.0)))
    return results


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--proxy", default=os.environ.get("BENCH_EGRESS_PROXY"))
    p.add_argument("--lb", default=os.environ.get("BENCH_LB_URL"))
    p.add_argument("--public-url", default=None, help="also fetch this public page through the proxy")
    p.add_argument("--timeout", type=float, default=4.0)
    p.add_argument("--no-settings", action="store_true", help="(tests) read no settings file")
    opts = p.parse_args(argv)
    missing = [name for name, value in (("BENCH_EGRESS_PROXY", opts.proxy), ("BENCH_LB_URL", opts.lb))
               if not value]
    if missing:
        print(f"egress_check: {', '.join(missing)} not set — nothing can be proven, so nothing may start",
              file=sys.stderr, flush=True)
        return 2
    home = os.environ.get("HOME") or str(pathlib.Path.home())
    settings = () if opts.no_settings else (MANAGED_SETTINGS, os.path.join(home, ".claude", "settings.json"))
    results = run(opts.proxy, opts.lb, home=home, dial=default_dial, environ=os.environ,
                  settings_files=settings, timeout=opts.timeout, public_url=opts.public_url)
    for r in results:
        print(f"{'PASS' if r.ok else 'FAIL'}  {r.kind:<11} {r.target:<44} {r.said}", flush=True)
    failed = [r for r in results if not r.ok]
    if failed:
        print(f"NOT SEALED: {len(failed)} of {len(results)} checks failed", flush=True)
        return 1
    print(f"SEALED: {len(results)} of {len(results)} checks passed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
