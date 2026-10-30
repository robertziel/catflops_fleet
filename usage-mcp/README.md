# vllm-usage-mcp — the fleet's usage data as read-only MCP tools

Eight tools over what the two-box vLLM/SGLang gateway has served, so an agent can ask instead of
curling `:8010` and parsing JSON. **NOT DEPLOYED and NOT REGISTERED** — building the image and
registering it are the operator's to run (below).

## Run it, and register it

```sh
REPO=/path/to/catflops_fleet               # this checkout
docker compose build                       # in usage-mcp/ — builds vllm-usage-mcp:latest, starts nothing

claude mcp add fleet-usage -- docker run --rm -i --network host \
  -v $REPO/lb/state:/data:ro \
  -v $REPO/lb/vllm_lb/library:/library:ro \
  -v /etc/localtime:/etc/localtime:ro vllm-usage-mcp:latest
```

That is the whole registration. `claude mcp remove fleet-usage` undoes it.

## The tools

| tool | answers |
|---|---|
| `fleet_status` | what the pair is serving and how fast — model, generation, seats/batched/spec, recorded throughput, per-box GPU and memory, backends, saturation, swarm state. "Nothing is loaded" is an answer. |
| `token_usage` | tokens served over a day or N days, grouped by model / generation / day / hour / config, with the prefix-cache hit rate and speculative acceptance. |
| `latency_profile` | TTFT and ITL percentiles, plus what one ITL sample **counts** — measured, not assumed. |
| `cluster_starts` | every start and stop: configuration asked for, phases, outcome, failure text. |
| `model_configuration` | the configuration *recorded* for a model (engine, image, limits, calibrated memory and the run each figure came from) and what the LB holds now. |
| `benchmark_runs` | saved benchmark runs and the archived sets that are on disk but not loaded. |
| `swarm_sessions` | each benchmark swarm session, measured from the LB's own windows inside its bracket. |
| `usage_data_inventory` | what usage data exists, counted off disk — and what does not. |

## Two numbers that mean different things, carried in the output

* **ITL is per TOKEN on SGLang, per speculative STEP on vLLM.** Quoting a vLLM figure raw
  overstated latency 3.2× (measured on the reference fleet). The LB's scrape states it as
  `itl_basis` (`"token"` / `"step"`) since `9c32807`, and this package uses the same two strings;
  `itl_basis` is **not yet persisted into a perf window or published in `/lb/status`**, so for
  recorded data `latency_profile` measures the basis from `gen_tokens / itl.n` — 1.004 on
  `qwen3.8-27b-sglang-dflash2`, 2.91–3.36 on the vLLM choices over the real logs — returns the
  per-token figure alongside the raw one, and says which source it used. When a window does carry
  the field it wins, and a disagreement with the measurement is reported rather than resolved.
* **One swarm session is not a configuration.** Mean concurrency has spanned 2.36–11.64 and
  throughput 47.8–130.4 tok/s between sessions, wider than any configuration effect measured here
  (measured on the reference fleet). `swarm_sessions` says so in every answer.

A session's own `total_cost_usd` is also returned, labelled fiction: it is client-side arithmetic
at Anthropic list prices for traffic the GB10 boxes served (`costBasis: "unknown"` in the record).

* **A prefix-cache share above 1 is refused, not returned.** Until `9c32807` the probe recorded
  SGLang's `prefill_effective_tokens_total{mode="input"}` as the queried total, and that series
  *excludes* the cache hits — so 39 windows recorded on 2026-10-03 imply shares up to **627%**.
  `token_usage` returns `prefix_hit_rate: null` for such a selection, plus
  `prefix_hit_rate_unusable` naming the defect, plus the raw hit and queried counts. It does not
  reconstruct a share: whether a window predates the fix is not recorded. Checked both ways —
  reverting the guard makes the suite report 610.34% for `qwen3.8-27b-sglang-dflash2` and two
  tests fail.

## Read-only, structurally

1. **One outbound function.** `readonly.lb_get` issues the literal verb `"GET"` against a frozen
   three-path allowlist (`/lb/status`, `/lb/perf`, `/lb/bench`). No function in the package takes a
   method, body or payload argument, so there is no code path to `POST /lb/cluster`.
2. **One file function.** `readonly.read_text` opens the literal mode `"r"` after resolving the
   path inside a frozen root. Nothing imports `subprocess`, `shutil`, `urllib.request` or a DB
   driver; nothing calls `.write()`.
3. **A test that proves it from the source.** `tests/test_readonly.py` parses every module's AST and
   fails on a forbidden import, a write/send call, a second `open`, a non-GET verb, or a literal
   mentioning a control route. Adding a control path means deleting a test, which shows in a diff.
4. **The container cannot write either.** `read_only: true`, `cap_drop: [ALL]`,
   `no-new-privileges`, non-root, and every mount `:ro` — including the LB's own state volume.

No secrets: `lb/.env` is never read and is outside the read roots. Every tool result is built from
a **named field allowlist**, so a value the LB starts publishing later cannot leak by appearing in
the document; `tests/test_secrets.py` plants credentials at several depths in the real captured
status and asserts no tool returns them.

## Tests

```sh
docker build -t vllm-usage-mcp:latest .
docker run --rm -v "$PWD":/app -w /app \
  -v $REPO/lb/state:/data:ro \
  -v $REPO/lb/vllm_lb/library:/library:ro \
  -v /etc/localtime:/etc/localtime:ro vllm-usage-mcp:latest python -m pytest
```

They run against the fleet's **real** recorded data — `lb/state/perf/*.jsonl`, the boot logs, the
swarm captures — not invented fixtures. `tests/data/` holds only what cannot be mounted: verbatim
`GET /lb/status` and `/lb/bench` captures, and the `files` section of `GET /lb/perf`, which
`test_perf.py` uses to assert this reader agrees with the LB's own counts file by file (596 windows
and 1 malformed line in `perf-2026-10-02.jsonl`). `test_server.py` drives the server as a
subprocess over real stdio with the SDK's own client. Add `--network host -e
USAGE_MCP_LIVE_LB=http://127.0.0.1:8010` to also exercise the live LB (five extra tests).

## What is not here, and why

Anthropic-side usage. The swarm harness points Claude Code at the **local** gateway under `env -i`
and aborts if the session reports `apiKeySource: none`, so 37 of the 38 recorded session streams
are local traffic. The 38th —
a stream recorded outside this repository — is the 2026-09-30 accident that
guard was added for: one `claude-opus-5` turn and the only `rate_limit_event` on this host. One
instant of one window is not a usage series, and it is outside this server's read roots.

Claude Code's own per-session accounting under `~/.claude` does exist, and is deliberately **not**
wired in: it is client state rather than fleet data, has no account-level total, and `~/.claude`
also holds `.credentials.json` — mounting it would put credentials one tool away from a model.
An account total would need Anthropic's usage API with an admin credential, which this server has
no path to.
