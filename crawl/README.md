# vllm-crawl — Crawl4AI web scraper (standalone service)

A thin deployment of the upstream **Crawl4AI** server image (headless Chromium + a REST API) as its
own compose project, matching the house pattern (own directory, own `.env`, published on `127.0.0.1`
only, nothing about one box hardcoded). **This repo adds no code to Crawl4AI** — only the deployment.

Callers: any agent, as MCP server `crawl4ai` (`/mcp/sse`, `transport: sse`, `Authorization: Bearer
<the token>`) — page → Markdown via its `md` tool. `lb/` is untouched.

- **Image (pinned):** `unclecode/crawl4ai:0.9.4` (current stable on Docker Hub, published 2026-09-23;
  `linux/amd64` + `linux/arm64`). Never `:latest`.
- **Container port:** 11235 (fixed by the image). The host-side port is `CRAWL_PORT`.

## Why a token is mandatory

Crawl4AI 0.9.x refuses to serve on a reachable interface without a credential. With **no**
`CRAWL4AI_API_TOKEN` it binds the *container's own* loopback (not `0.0.0.0`), so any published port is
reset on connect (`Connection reset by peer`). With a token set it binds all interfaces **inside the
container** — still loopback-only on the host via the `127.0.0.1:${CRAWL_PORT}` port map — and
requires `Authorization: Bearer <token>` on every endpoint except `GET /health`. So the service is
**loopback-only on the host AND token-gated**. The token is a secret in `crawl/.env` (gitignored).

## Run (any box — the LB's box or a standalone server)

```bash
cd crawl
cp .env.example .env
# set CRAWL_PORT (a free loopback port on THIS box) and CRAWL_API_TOKEN=$(openssl rand -hex 32)
docker compose up -d
```

The same compose runs on any box; only `crawl/.env` differs (different port and token per box).
Nothing about a deployment is in the compose file — both vars are `${VAR:?}`, so a missing one fails
the boot loudly with no default.

### site-b (standalone topology)

The crawl service is fully independent of the LB/edge/node stack, so it does **not** go through
`deploy/standalone/up.sh` or any override. On site-b, bring it up exactly as above from the repo's
`crawl/` directory with site-b's own `crawl/.env` (its own `CRAWL_PORT` + `CRAWL_API_TOKEN`). Each
fleet (the LB's box, the standalone server) runs its **own** local instance — service ports are not reachable between
the boxes by design, and no caller ever points across hosts (see *Call contract*).

## Smoke test

Proves the running container actually renders a page to Markdown. Stdlib-only; runs in a throwaway
`python:3.12-slim` with `--network host` so it reaches the loopback port. Skips (exit 0, naming why)
if `CRAWL_BASE` is unset or the service is down; fails only if an up service returns no Markdown.

```bash
cd crawl
set -a && . ./.env && set +a
docker run --rm --network host -v "$PWD/tests":/t -w /t \
  -e CRAWL_BASE="http://127.0.0.1:${CRAWL_PORT}" -e CRAWL_TOKEN="${CRAWL_API_TOKEN}" \
  python:3.12-slim python test_smoke.py
```

## Call contract (for a future lb/ caller)

A caller reads **one env var**, its own box's local URL — `CRAWL_URL=http://127.0.0.1:${CRAWL_PORT}`
— plus the token (`CRAWL_API_TOKEN`, the same secret the service runs with). One value, one place;
never a cross-host URL. Each fleet's caller hits its own local instance.

**Health** (no auth):
```
GET  {CRAWL_URL}/health            ->  200  {"status":"ok","version":"0.9.4",...}
```

**Crawl** (bearer required):
```
POST {CRAWL_URL}/crawl
Headers: Authorization: Bearer <CRAWL_API_TOKEN>
         Content-Type: application/json
Body:    {
           "urls": ["https://example.com"],
           "browser_config":  {"type":"BrowserConfig",     "params":{"headless":true}},
           "crawler_config":  {"type":"CrawlerRunConfig",  "params":{"cache_mode":"bypass"}}
         }
```
Non-primitive config values use Crawl4AI's `{"type":"ClassName","params":{...}}` wrapper. Omitting
`browser_config`/`crawler_config` uses the server defaults.

**Response** (`200`):
```jsonc
{
  "success": true,
  "results": [
    {
      "url": "https://example.com",
      "markdown": {              // object, not a string, in 0.9.x
        "raw_markdown": "...",       // <- the Markdown a caller wants
        "fit_markdown": "...",       // content-filtered variant
        "markdown_with_citations": "...",
        "references_markdown": "...",
        "fit_html": "..."
      },
      "html": "...", "cleaned_html": "...", "links": {...}, "media": {...},
      "metadata": {...}, "error_message": ""
    }
  ],
  "server_processing_time_s": 1.2, "server_peak_memory_mb": ...
}
```
Markdown path: `results[0].markdown.raw_markdown`. A missing token → `401`.

## Notes

- `shm_size: 1gb` — headless Chromium needs more than Docker's 64 MB default or tabs crash with
  "Target closed" (matches upstream's `docker run --shm-size=1g`).
- `restart: unless-stopped`, loopback-only publish, json-file log rotation — same as the other
  services here.
- No `frpc` / public vhost: this service is internal-only and must not be exposed. Do not add one
  without an explicit decision.
