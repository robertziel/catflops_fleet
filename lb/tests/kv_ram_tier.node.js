/* THE KV TIER IN THE HOST'S RAM on the engine card — checked where gateway.js draws it, under node.
 *
 * Operator, 2026-10-12: "implement offload KV and non-VRAM KV graph, real-time poll as well" — then, over the first
 * version (a reload rate headed "kv · ram" over "occupancy not reported", and a second tile for both rates): "I asked
 * to show how much data is filled on other fallback KV, not GB/s, though GB/s is also useful. Maybe add two charts as
 * two rows in one per memory type, total usage and available on top and Gb/s below" and "Use same colors as memory
 * bandwidth, since high is not a good thing so should be red". What this pins:
 *
 *   * ONE tile beside KV, "kv · ram", headed like KV: the fill and what it holds of the tier — "100 % · 32.0 / 32 GiB"
 *     (`kv_offload.used_gib` / `tier_gib`, DERIVED: the hover says how, `occupancy_derived`), coloured on the LOAD
 *     scale memory bandwidth uses (full = red); a fill not known is "—" with its reason, the rates still drawn;
 *   * two graphs, stacked in the room one graph takes (the row keeps its height): on top the fill's past
 *     (`history.kv_ram_pct`, 0-100 %), below the reload rate with the store's as the quieter line on ONE axis, coloured
 *     on the same load scale against the copy speed the engine measured — a reload as fast as the copy can go is red;
 *   * under them both rates and both copy speeds — "1.42 · 0.08 GB/s · copy 13.2 · 13.0"; a direction not measured
 *     is "—", never 0;
 *   * a server with NO tier (`kv_offload: null`) draws no tile: the row is the five it always was;
 *   * the row's grid holds six, and at a phone's card width it wraps (gateway.css `.tiles6`).
 *
 *   node tests/kv_ram_tier.node.js   (tests/test_kv_ram_tier_page.py runs this when node exists)
 */
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');

// palette.js first: the colours this script computes are read from the style's tokens (tests/palette_env.js).
const src = require('./palette_env').withPalette(fs.readFileSync(path.join(__dirname, '..', 'vllm_lb', 'static', 'gateway.js'), 'utf8'));
const css = fs.readFileSync(path.join(__dirname, '..', 'vllm_lb', 'static', 'gateway.css'), 'utf8');

const el = () => ({ addEventListener() {}, removeAttribute() {}, setAttribute() {}, append() {},
  classList: { toggle() {}, add() {}, remove() {} }, style: {}, hidden: false, innerHTML: '',
  textContent: '', value: '', dataset: {}, querySelectorAll: () => [], querySelector: () => null,
  focus() {}, observe() {}, closest: () => null, getBoundingClientRect: () => ({}) });
const document = { getElementById: el, createElementNS: el, createElement: el, addEventListener() {},
  querySelectorAll: () => [], querySelector: () => null, body: { style: {}, classList: { toggle() {} } },
  documentElement: { style: {}, classList: { toggle() {} } } };
const window = { addEventListener() {}, innerWidth: 1280, innerHeight: 800, matchMedia: () => ({
  matches: false, addEventListener() {} }), localStorage: { getItem: () => null, setItem() {} } };
const api = new Function('document', 'window', 'fetch', 'location', 'setInterval', 'setTimeout',
  'getComputedStyle', 'ResizeObserver', 'localStorage', 'navigator',
  `${src}\n;return { serverCard, kvRamTile, tileCount, SPARKLINE_H, scaleColor };`)(
  document, window, () => new Promise(() => {}), { search: '', href: '' }, () => 0, () => 0,
  () => ({}), undefined, window.localStorage, { userAgent: 'node' });

let checks = 0;
const ok = (c, m, d) => { assert.ok(c, `${m}${d === undefined ? '' : ` — ${JSON.stringify(d)}`}`); checks++; };

/* site-b's node as /lb/status serves it: the two real reads 5 s apart (tests/fixtures/throughput/vllm-2eb491e-ram-tier-
   {1,2}.metrics) and three probes before them, the store moving faster than the reload at one sample. */
const AT = [100, 105, 110, 115, 120];
const RELOAD = [null, 0, 482e6, 0, 1417976217.6];
const STORE = [null, 2.1e9, 51e6, 0, 82129715.2];
const FILL = [60, 80, 95, 100, 100];
const RULE = 'derived: min(tier, bytes stored since the engine started) — an LRU tier …';
const OFFLOAD = {
  tier_gib: 32.0,
  reload: { bytes_s: 1417976217.6, window_s: 5, why: null, copy_gbps: 13.213, copy_why: null },
  store: { bytes_s: 82129715.2, window_s: 5, why: null, copy_gbps: 12.991, copy_why: null },
  occupancy_pct: 100.0, used_gib: 32.0, occupancy_why: null, occupancy_derived: RULE,
};
const node = (offload, hist) => ({
  kind: 'cluster', url: 'http://node-a:8000', live: true, running: 3, waiting: 0, inflight: 3, depth_known: true,
  served_models: ['Qwen/Qwen3.8-27B'], max_model_len: 131072,
  throughput: { source: 'metrics', generated: { tok_s: 62 }, prompt: { tok_s: 900, computed_tok_s: 300 },
                kv_pool: { tokens: 240298 }, kv_offload: offload },
  history: { at: AT, running: [3, 3, 3, 3, 3], depth: [3, 3, 3, 3, 3], gen_tok_s: [60, 61, 62, 61, 62],
             kv_pct: [91, 92, 92, 93, 93], ...(hist || {}) },
});
const S = { now: 121, cluster: { configured: true, state: 'running', topology: { tp: 1 } } };
const RINGS = { kv_reload_bytes_s: RELOAD, kv_store_bytes_s: STORE, kv_ram_pct: FILL };

const tileOf = (html, label) => {
  const i = html.indexOf(`<span class="t-label">${label}</span>`);
  if (i < 0) return null;
  const start = html.lastIndexOf('<div class="tile', i);
  const next = html.indexOf('<div class="tile', i);
  return html.slice(start, next < 0 ? undefined : next);
};
const valOf = (t) => (t.match(/<b class="t-val"[^>]*>([^<]*)/) || [])[1];
const valAttrs = (t) => (t.match(/<b class="t-val"([^>]*)>/) || [])[1] || '';
const unitOf = (t) => (t.match(/<span class="t-unit">([^<]*)/) || [])[1];
const subOf = (t) => (t.match(/<span class="t-sub">(.*?)<\/span><\/div>/s) || [])[1];
/** The tile's graphs, in order: each one's samples mask, main line, quiet line and mark. */
const graphsOf = (t) => [...t.matchAll(/<div class="t-graph"([^>]*)>(.*?)<\/div>/gs)].map(([, attrs, inner]) => ({
  attrs, samples: (inner.match(/data-samples="([01]*)"/) || [])[1],
  main: (inner.match(/<path d="([^"]+)" fill="none" stroke="url\(#[^)]+\)" stroke-width="1\.5"/) || [])[1] || '',
  quiet: (inner.match(/<path class="sparkline-per-req" d="([^"]+)"/) || [])[1] || '',
  stops: [...inner.matchAll(/stop-color="([^"]+)"/g)].map((m) => m[1]),
  mark: (inner.match(/class="t-mark"[^>]*>([^<]*)/) || [])[1] }));
const ptsOf = (d) => [...d.matchAll(/([\d.]+) ([\d.]+)/g)].map(([, x, y]) => [Number(x), Number(y)]);

// ── ONE tile for the tier, beside the KV tile ─────────────────────────────────────────────────
{
  const b = node(OFFLOAD, RINGS);
  const html = api.serverCard(S, b);
  ok(/class="tiles tiles6"/.test(html), 'the row holds six tiles', html.match(/class="tiles tiles\d"/));
  const labels = [...html.matchAll(/<span class="t-label">([^<]*)<\/span>/g)].map((m) => m[1]);
  ok(labels.indexOf('kv · ram') === labels.indexOf('kv') + 1, 'KV · RAM sits right beside KV', labels);
  ok(!labels.includes('reload · store'), 'no second tile for the rates', labels);

  const ram = tileOf(html, 'kv · ram');
  ok(valOf(ram) === '100' && unitOf(ram) === '% · 32.0 / 32 GiB', 'headed like KV: the fill, used / tier',
     [valOf(ram), unitOf(ram)]);
  ok(valAttrs(ram).includes(`color:${api.scaleColor(1)}`), 'coloured on the load scale: full is its top (red)',
     valAttrs(ram));
  ok(valAttrs(ram).includes(`title="${RULE}"`), 'the hover says the fill is derived, and how', valAttrs(ram));
  ok(subOf(ram) === '1.42 · 0.08 GB/s · copy 13.2 · 13.0', 'under the graphs: reload · store, then the copy speeds',
     subOf(ram));

  const g = graphsOf(ram);
  ok(g.length === 2, 'two graphs, one tile', g.length);
  ok(/class="t-pair"/.test(ram), 'stacked in one graph cell (`t-pair`), so the row keeps its height');
  ok(g[0].samples.endsWith('11111') && !g[0].quiet, 'on top: the fill ring alone', g[0]);
  const fill = ptsOf(g[0].main), H = api.SPARKLINE_H;
  ok(Math.abs(fill[4][1] - 1) < 1e-6 && Math.abs(fill[0][1] - (1 + (H - 2) * 0.4)) < 0.01,
     'its axis is 0-100 %: 100 at the top, 60 at 40 % down', [fill[0], fill[4]]);
  ok(g[0].mark === 'max 100%', 'its mark', g[0].mark);
  ok(g[1].samples.endsWith('01111') && g[1].quiet, 'below: reload, with the store quieter', g[1]);
  // ONE AXIS for both directions, at least the copy speed tall: 2.1 GB/s of store < 13.2 GB/s of copy.
  const main = ptsOf(g[1].main), quiet = ptsOf(g[1].quiet), top = 13.213e9;
  ok(Math.abs(main[3][1] - (1 + (H - 2) * (1 - 1417976217.6 / top))) < 0.01, 'reload against the copy speed', main[3]);
  ok(Math.abs(quiet[0][1] - (1 + (H - 2) * (1 - 2.1e9 / top))) < 0.01, 'store on the same axis', quiet[0]);
  ok(g[1].stops.every((c) => c !== api.scaleColor(1)), 'a reload far under the copy speed is not red', g[1].stops);
  ok(g[1].mark === 'max 1.42 GB/s', 'its mark says GB/s', g[1].mark);
}

// ── the fill not known: "—", the rates still drawn ──────────────────────────────────────────────
{
  const unknown = { ...OFFLOAD, occupancy_pct: null, used_gib: null, occupancy_why: 'not exposed by this server' };
  const b = node(unknown, { ...RINGS, kv_ram_pct: [null, null, null, null, null] });
  const ram = api.kvRamTile(S, b, b.history);
  ok(valOf(ram) === '—' && /class="tile dim/.test(ram), 'no fill: "—", dim', valOf(ram));
  ok(/title="not exposed by this server"/.test(ram), 'with the reason in its title');
  ok(subOf(ram) === '1.42 · 0.08 GB/s · copy 13.2 · 13.0', 'the rates still stand', subOf(ram));
  ok(graphsOf(ram)[1].main, 'and their graph is drawn');
}

// ── no tier: no tile, not zeros ─────────────────────────────────────────────────────────────────
{
  const b = node(null, { kv_reload_bytes_s: [null, null, null, null, null], kv_store_bytes_s: [null, null, null, null, null] });
  const html = api.serverCard(S, b);
  ok(/class="tiles tiles5"/.test(html), 'a node without the tier keeps its five', html.match(/class="tiles tiles\d"/));
  // (Its NAME is in the "not collected" tags under the graphs — unused_graphs.node.js; no TILE carries it.)
  ok(!/class="t-label">kv · ram</.test(html), 'and draws no tier tile');
  ok(api.tileCount(b) === 5, 'tileCount agrees');
  const old = node(undefined);
  delete old.throughput.kv_offload;
  ok(api.tileCount(old) === 5 && !api.serverCard(S, old).includes('kv · ram'), 'an older LB: nothing drawn');
}

// ── nothing measured yet; a measured zero ───────────────────────────────────────────────────────
{
  const first = { ...OFFLOAD,
    reload: { bytes_s: null, window_s: null, why: 'first sample', copy_gbps: 13.256, copy_why: null },
    store: { bytes_s: null, window_s: null, why: 'not exposed by this server', copy_gbps: null,
             copy_why: 'not exposed by this server' } };
  const b = node(first, { kv_reload_bytes_s: [null, null, null, null, null], kv_store_bytes_s: [null, null, null, null, null],
                          kv_ram_pct: [null, null, null, null, 100] });
  const ram = api.kvRamTile(S, b, b.history);
  ok(subOf(ram) === '— · — GB/s · copy 13.3 · —', 'a rate or copy speed not measured is "—", never 0', subOf(ram));
  ok(!/\b0\.00\b/.test(ram), 'no invented zero anywhere');
  const idle = { ...OFFLOAD, reload: { ...OFFLOAD.reload, bytes_s: 0 }, store: { ...OFFLOAD.store, bytes_s: 0 } };
  const z = node(idle, { kv_reload_bytes_s: [0, 0, 0, 0, 0], kv_store_bytes_s: [0, 0, 0, 0, 0], kv_ram_pct: FILL });
  ok(subOf(api.kvRamTile(S, z, z.history)).startsWith('0 · 0 GB/s'), 'an idle tier moves 0 GB/s, measured');
}

// ── the grid holds six, and wraps on a phone; the pair fills one graph cell ─────────────────────
{
  ok(/\.tiles6\s*\{[^}]*grid-template-columns:\s*repeat\(6,/.test(css), '.tiles6 is a grid of six');
  // each tile is a box on its row's rows (option D, 2026-10-30), auto-placed one to a column
  ok(/:is\(\.tiles3, \.tiles4, \.tiles5, \.tiles6\) > \.tile \{[^}]*grid-template-rows: subgrid/.test(css),
     'every tile of the row is a subgrid box');
  ok(/@container \(max-width: \d+px\) \{\s*\.tiles6 \{ grid-template-columns: repeat\(6,[^}]*\}\s*\.tiles6 > \.tile \{ grid-column: span 2; \}/.test(css),
     'a narrow card wraps the six into rows of three');
  ok(!/\.tiles7/.test(css), 'no grid of seven left behind');
  ok(/\.t-pair\s*\{[^}]*flex-direction:\s*column/.test(css), 'the two graphs stack');
}

console.log(`kv_ram_tier: ${checks} checks ok`);
