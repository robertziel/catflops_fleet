#!/usr/bin/env python3
"""Smoke test: prove the running vllm-crawl container actually scrapes a page to Markdown.

This service is the upstream Crawl4AI image unchanged, so there is no project image to run a
suite inside — this is a single stdlib script that hits the LIVE container over its loopback port.

Run it (from the repo root, with the service up via `cd crawl && docker compose up -d`):

    docker run --rm --network host -v "$PWD/crawl/tests":/t -w /t \
      -e CRAWL_BASE="http://127.0.0.1:${CRAWL_PORT}" -e CRAWL_TOKEN="${CRAWL_API_TOKEN}" \
      python:3.12-slim python test_smoke.py

--network host lets the throwaway container reach the 127.0.0.1-published port. It uses only the
stdlib, so the bare python image needs no pip install.

CRAWL_TOKEN must be the same CRAWL_API_TOKEN the service runs with — Crawl4AI 0.9.x requires
`Authorization: Bearer <token>` on every endpoint except /health.

SKIPS (exit 0, naming why) when CRAWL_BASE is unset or the service is not reachable — matching how
the lb broker tests skip when the broker is absent, so a short run is never a silent failure.
FAILS (exit 1) only when the service is up but does not return non-empty Markdown.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

# A stable, tiny public page. example.com is maintained by IANA for exactly this kind of use.
TARGET = "https://example.com"


def _skip(reason: str) -> None:
    print(f"SKIP: {reason}")
    sys.exit(0)


def _get(base: str, path: str, timeout: float = 15.0):
    with urllib.request.urlopen(base + path, timeout=timeout) as r:
        return r.status, r.read()


def _post_crawl(base: str, url: str, token: str, timeout: float = 120.0):
    body = json.dumps(
        {
            "urls": [url],
            "browser_config": {"type": "BrowserConfig", "params": {"headless": True}},
            "crawler_config": {"type": "CrawlerRunConfig", "params": {"cache_mode": "bypass"}},
        }
    ).encode()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(base + "/crawl", data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read())


def _markdown_of(result: dict) -> str:
    """Crawl4AI's markdown field is a string in some versions and an object
    ({raw_markdown, fit_markdown, ...}) in others — accept either."""
    md = result.get("markdown")
    if isinstance(md, str):
        return md
    if isinstance(md, dict):
        return md.get("raw_markdown") or md.get("fit_markdown") or ""
    return ""


def main() -> None:
    base = os.environ.get("CRAWL_BASE")
    if not base:
        _skip("CRAWL_BASE unset — point it at the running service, e.g. http://127.0.0.1:11235")
    base = base.rstrip("/")

    try:
        status, _ = _get(base, "/health")
    except (urllib.error.URLError, ConnectionError, OSError) as e:
        _skip(f"{base} not reachable ({e}) — is `docker compose up -d` running in crawl/?")

    assert status == 200, f"/health returned {status}, expected 200"
    print(f"health OK: {base}/health -> 200")

    token = os.environ.get("CRAWL_TOKEN", "")
    status, payload = _post_crawl(base, TARGET, token)
    assert status == 200, f"/crawl returned HTTP {status}"
    assert payload.get("success") is True, f"/crawl success != true: {str(payload)[:300]}"

    results = payload.get("results") or []
    assert results, f"/crawl returned no results: {str(payload)[:300]}"
    md = _markdown_of(results[0])
    assert md.strip(), f"/crawl returned empty Markdown for {TARGET}"
    # example.com's body always mentions its domain purpose; a loose check that real rendered text
    # came back, not a brittle exact title (the page's wording changes over time).
    assert "domain" in md.lower(), f"Markdown has no recognizable text from {TARGET}: {md[:200]!r}"

    print(f"crawl OK: {TARGET} -> {len(md)} chars of Markdown")
    print("--- Markdown snippet ---")
    print(md.strip()[:400])
    print("--- PASS ---")


if __name__ == "__main__":
    main()
