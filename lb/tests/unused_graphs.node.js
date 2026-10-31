/* THE GRAPHS THIS SETUP DOES NOT COLLECT, NAMED UNDER THE CARD — checked where gateway.js builds the engine card.
 *
 * Operator, 2026-10-12, over site-b's card: "add below small tags of graphs available but not collecting data under
 * current setup cluster, just for information". A cluster card leaves out a graph its setup has nothing to feed
 * (303cd2b: no NETWORK tile on one rank; 18e23aa: no KV · RAM / RELOAD · STORE without a RAM tier). The tile stays
 * out — this adds one quiet row of tags under the graphs naming each one, with why. What this pins:
 *
 *   1. TP=1 with a RAM tier (site-b): one tag, `network · TP=1`, under the ceiling row; no network TILE;
 *   2. TP=2 with no RAM tier (the box-c pair): `kv · ram · no tier`; network is a tile;
 *   3. TP=1 and no tier: both, in the row order of the tiles they stand for;
 *   4. a setup that draws every graph: no row at all — never an empty label;
 *   5. unknown is not not-applicable: an LB that sends no `kv_offload` key (older than 18e23aa) names no RAM tag;
 *      a server that is not live draws no row;
 *   6. the CSS: the row wraps and its tags are the quiet outlined chip.
 *
 *   docker run --rm -v "$PWD/lb":/lb:ro -w /lb node:22-slim node tests/unused_graphs.node.js
 *   (tests/test_unused_graphs_page.py runs this when node exists)
 */
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');

// palette.js first: the colours this script computes are read from the style's tokens (tests/palette_env.js).
const src = require('./palette_env').withPalette(fs.readFileSync(path.join(__dirname, '..', 'vllm_lb', 'static', 'gateway.js'), 'utf8'));
const css = fs.readFileSync(path.join(__dirname, '..', 'vllm_lb', 'static', 'gateway.css'), 'utf8');

const el = () => ({ addEventListener() {}, removeAttribute() {}, setAttribute() {}, append() {},
  classList: { toggle() {}, add() {}, remove() {}, contains: () => false }, style: {}, hidden: false, innerHTML: '',
  textContent: '', value: '', dataset: {}, querySelectorAll: () => [], querySelector: () => null,
  focus() {}, observe() {}, closest: () => null, getBoundingClientRect: () => ({}) });
const document = { getElementById: el, createElementNS: el, createElement: el, addEventListener() {},
  querySelectorAll: () => [], querySelector: () => null, activeElement: null,
  body: { style: {}, classList: { toggle() {} } }, documentElement: { style: {}, classList: { toggle() {} } } };
const window = { addEventListener() {}, innerWidth: 1280, innerHeight: 800, matchMedia: () => ({
  matches: false, addEventListener() {} }), localStorage: { getItem: () => null, setItem() {} } };
const api = new Function('document', 'window', 'fetch', 'location', 'setInterval', 'setTimeout',
  'getComputedStyle', 'ResizeObserver', 'localStorage', 'navigator',
  `${src}\n;return { serverCard };`)(
  document, window, () => new Promise(() => {}), { search: '', href: '' }, () => 0, () => 0,
  () => ({}), undefined, window.localStorage, { userAgent: 'node' });

let checks = 0;
const ok = (c, m, d) => { assert.ok(c, `${m}${d === undefined ? '' : ` — ${JSON.stringify(d)}`}`); checks++; };

const strip = (h) => h.replace(/<[^>]*>/g, '').replace(/&middot;/g, '·').replace(/\s+/g, ' ').trim();
/** The unused-graphs row of a card: its markup and each tag's text and title, in order — or null. */
function unused(html) {
  const at = html.indexOf('<div class="unused"');
  if (at < 0) return null;
  const end = html.indexOf('</div>', at);
  const part = html.slice(at, end + 6);
  const tags = [...part.matchAll(/<span class="tag unused"( title="([^"]*)")?>(.*?)<\/span>(?=<span class="tag|<\/div>)/g)]
    .map((m) => ({ text: strip(m[3]), title: m[2] || '' }));
  return { html: part, at, tags };
}
const tileLabels = (html) => [...html.matchAll(/class="t-label">([^<]*)</g)].map((m) => strip(m[1]));

// ── the documents (the shapes network_tile.node.js uses) ─────────────────────────────────────────────────────────
const NOW = 1791800000;
const CORES = (n) => ({ cores: n, multiplying: 49, memory: 6209, other: n - 6258, tflops: 3.8, gbps: 511.3,
  bytes: 'weights', steps_s: 58.1, tokens_step: 2.6, mixed_share: 0, idle_share: 0,
  cases: { decode: { multiplying: 19, memory: 6209, other: n - 6228, step_ms: 17, time_share: 1 }, prefill: null } });
const perBoxOf = (boxes) => boxes.map((box) => ({ box, cards: 1, rate: 1539, peak: 1792, pct: 85.9, why: null }));
const sat = (boxes, network) => ({ verdict: 'memory-bandwidth bound', because: 'x', sm_fill_pct: 90,
  compute: { rate: 10.9, peak: 1676, pct: 0.7, why: null, peak_precision: 'nvfp4', boxes: perBoxOf(boxes) },
  bandwidth: { rate: 1539, peak: 1792, pct: 85.9, why: null, boxes: perBoxOf(boxes) },
  network, cores: CORES(21760), boxes: [], history: { gbps: [1500, 1539], tflops: [10.2, 10.9], net_gbps: [0.5, 0.6] } });
// What the LB sends for a RAM tier (throughput.py `kv_offload`) — and `null` where the engine runs none.
const TIER = { tier_gib: 32, occupancy_pct: 100, used_gib: 32, occupancy_why: null, occupancy_derived: 'derived',
  reload: { bytes_s: 0, why: null, copy_gbps: 11.9 }, store: { bytes_s: 0, why: null, copy_gbps: 10.7 } };
const head = (url, kvOffload, live = true) => {
  const throughput = { source: 'metrics', generated: { tok_s: 62 }, prompt: {}, engine_totals: {} };
  if (kvOffload !== undefined) throughput.kv_offload = kvOffload;
  return { kind: 'cluster', url, live, running: 1, waiting: 0, inflight: 1, depth_known: true, served_models: ['m'],
    max_model_len: 131072, history: { gen_tok_s: [60, 62], depth: [1, 1], kv_pct: [40, 53] }, throughput };
};
const status = (topology, boxes, saturation, b) => ({ now: NOW, backends: [b], leases: [], hardware: {},
  broker: { enabled: false }, config: {}, models: { cluster: [], classic: [] }, saturation,
  cluster: { configured: true, state: 'running', held: true, topology,
    boxes: Object.fromEntries(boxes.map((x) => [x, { label: x, rank: { present: true, running: true }, hold: { blockers: [] } }])) } });

const NO_LINK = { rate: null, peak: null, pct: null, why: 'TP=1: one GPU, no link between ranks', boxes: {} };
const NET = { rate: 0.66, peak: 31.5, pct: 2.1, why: null, boxes: {} };
const ONE = { tp: 1, boxes: ['node-a'], control: 'local', why: '', steps: ['load', 'warm up', 'serve'] };
const PAIR = { tp: 2, boxes: ['pair-a', 'pair-b'], control: 'ssh', why: '', steps: ['free', 'reserve', 'drain', 'load', 'warm up', 'serve'] };
const one = (kvOffload, live) => { const b = head('http://node-a:8000', kvOffload, live);
  return api.serverCard(status(ONE, ['node-a'], sat(['node-a'], NO_LINK), b), b); };
const pair = (kvOffload) => { const b = head('http://pair-a:8100', kvOffload);
  return api.serverCard(status(PAIR, ['pair-a', 'pair-b'], sat(['pair-a', 'pair-b'], NET), b), b); };

// 1. 'site-b': TP=1 with a RAM tier — one tag, network, under the graphs; no network tile.
{
  const html = one(TIER);
  const u = unused(html);
  ok(u, 'a one-rank card names the graphs it does not collect');
  ok(JSON.stringify(u.tags.map((t) => t.text)) === JSON.stringify(['network · TP=1']), 'one tag: network · TP=1',
     u && u.tags);
  ok(/between ranks/.test(u.tags[0].title), 'its title says what the graph measures', u.tags[0].title);
  ok(!tileLabels(html).includes('network'), 'and still no network tile', tileLabels(html));
  ok(u.at > html.indexOf('<div class="ceils'), 'the row sits below the ceiling tiles');
  ok(tileLabels(html).includes('kv · ram'), 'the tier\'s tile stays');
  ok(/<span class="unused-k">not collected<\/span>/.test(u.html), 'headed "not collected"', strip(u.html));
}

// 2. the pair: TP=2, no RAM tier — the tier's two; network is a tile, not a tag.
{
  const html = pair(null);
  const u = unused(html);
  ok(u && JSON.stringify(u.tags.map((t) => t.text)) === JSON.stringify(['kv · ram · no tier']),
     'the pair names the RAM tier\'s tile', u && u.tags);
  ok(u.tags.every((t) => /RAM tier/.test(t.title)), 'each title names the RAM tier', u.tags);
  ok(tileLabels(html).includes('network'), 'network is drawn as a tile');
  ok(!tileLabels(html).includes('kv · ram'), 'no kv · ram tile');
}

// 3. one rank, no tier: all three, in the order of the tiles they stand for.
{
  const u = unused(one(null));
  ok(u && JSON.stringify(u.tags.map((t) => t.text))
     === JSON.stringify(['kv · ram · no tier', 'network · TP=1']),
     'both, the tile row\'s order', u && u.tags);
}

// 4. every graph drawn: no row, no empty label.
{
  const html = pair(TIER);
  ok(unused(html) === null && !html.includes('not collected'), 'a setup that draws every graph: no row');
}

// 5. unknown is not not-applicable; a server that is not live draws no row.
{
  const u = unused(one(undefined));
  ok(u && JSON.stringify(u.tags.map((t) => t.text)) === JSON.stringify(['network · TP=1']),
     'an LB that sends no kv_offload key: the tier is unknown, not named', u && u.tags);
  ok(unused(pair(undefined)) === null, 'the pair from an older LB: no row');
  ok(unused(one(null, false)) === null, 'a server that is not live: no row');
}

// 6. the CSS.
{
  ok(/\.unused\s*\{[^}]*flex-wrap:\s*wrap/.test(css), '.unused wraps');
  ok(/\.tag\.unused\s*\{[^}]*color:\s*var\(--ink-faint\)/.test(css), 'its tags are the quiet faint chip');
  ok(/\.unused-k\s*\{[^}]*color:\s*var\(--ink-faint\)/.test(css), 'its label is faint');
}

console.log(`unused_graphs: ${checks} checks ok`);
