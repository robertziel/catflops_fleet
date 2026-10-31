/* THE NETWORK TILE IS DRAWN ONLY WHERE THERE IS A LINK — checked where gateway.js builds the engine card, under node.
 *
 * Operator, 2026-10-12, over site-b's card (a NETWORK tile reading "—" over a dotted meter and the amber "TP=1: one
 * GPU, no link between ranks"): "we don't need to show if not used". The tile measures the RDMA link BETWEEN RANKS
 * (cluster_loop._net_rate); one rank has none, so on a one-rank cluster the tile is not applicable — not unknown. What
 * this pins:
 *
 *   1. TP=1 (the topology /lb/status sends, `cluster.topology.tp`): no NETWORK tile, no placeholder, no note (its name
 *      is only in the "not collected" tags under the graphs — unused_graphs.node.js) — the row
 *      is memory bandwidth, compute and CUDA cores, and its grid holds three (`ceils3`), so nothing leaves a gap;
 *   2. the rule is the TOPOLOGY's: a one-rank node named like a pair box still draws none, and an older LB that sends
 *      no reason (or no network block) changes nothing;
 *   3. TP=2 (and a document from before `topology`, whose two boxes are its TP): the tile is where it always was —
 *      after the per-box tiles, before the cores — built from the same figures, and the row stays four across;
 *      a link not measured YET ("awaiting 2nd RDMA reading") keeps its tile: unknown is not not-applicable;
 *   4. the CSS: a row of three is three across, and on a narrow card two with the third spanning the row; a row of two
 *      is two across. No rule for five or six: the pair's row is untouched.
 *
 * Every other tile on the one-rank card (generated, prefill, prompt, queue, KV, the RAM tier's two, memory bandwidth,
 * compute, CUDA cores) measures that one GPU and stays.
 *
 *   docker run --rm -v "$PWD/lb":/lb:ro -w /lb node:22-slim node tests/network_tile.node.js
 *   (tests/test_network_tile_page.py runs this when node exists)
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
  `${src}\n;return { serverCard, ceilingTile };`)(
  document, window, () => new Promise(() => {}), { search: '', href: '' }, () => 0, () => 0,
  () => ({}), undefined, window.localStorage, { userAgent: 'node' });

let checks = 0;
const ok = (c, m, d) => { assert.ok(c, `${m}${d === undefined ? '' : ` — ${JSON.stringify(d)}`}`); checks++; };

const strip = (h) => h.replace(/<[^>]*>/g, '').replace(/&middot;/g, '·').replace(/\s+/g, ' ').trim();
/** The ceiling row of a card: its class, and each tile's markup and label, in order. */
function row(html) {
  const at = html.indexOf('<div class="ceils');
  if (at < 0) return { cls: null, tiles: [] };
  const cls = html.slice(at).match(/^<div class="([^"]*)"/)[1];
  const starts = [...html.matchAll(/<div class="ceil(?: [^"]*)?"[^>]*>/g)].map((m) => m.index).filter((i) => i > at);
  const tiles = starts.map((s, i) => {
    const part = html.slice(s, i + 1 < starts.length ? starts[i + 1] : undefined);
    return { html: part, label: strip((part.match(/class="t-label">([^<]*)/) || [, ''])[1]) };
  });
  return { cls, tiles };
}

// ── the documents ───────────────────────────────────────────────────────────────────────────────
const NOW = 1791800000;
const CORES = (n) => ({ cores: n, multiplying: 49, memory: 6209, other: n - 6258, tflops: 3.8, gbps: 511.3,
  bytes: 'weights', steps_s: 58.1, tokens_step: 2.6, mixed_share: 0, idle_share: 0,
  cases: { decode: { multiplying: 19, memory: 6209, other: n - 6228, step_ms: 17, time_share: 1 }, prefill: null } });
const perBoxOf = (boxes, rate, peak, derived) => boxes.map((box) => ({ box, cards: 1, rate, peak,
  pct: Math.round((rate / peak) * 1000) / 10, why: null, derived }));
/** The saturation block as proxy.py `_saturation` serves it for these boxes. `network` as given. */
const sat = (boxes, network, cores) => ({ verdict: 'memory-bandwidth bound', because: 'x', sm_fill_pct: 90,
  compute: { rate: 10.9, peak: 1676, pct: 0.7, why: null, peak_precision: 'nvfp4',
    boxes: perBoxOf(boxes, 10.9, 1676, boxes.length > 1 ? 'derived' : null) },
  bandwidth: { rate: 1539, peak: 1792, pct: 85.9, why: null,
    boxes: perBoxOf(boxes, 1539, 1792, boxes.length > 1 ? 'derived' : null) },
  network, cores, boxes: [], history: { gbps: [1500, 1539], tflops: [10.2, 10.9], net_gbps: [0.5, 0.61, null, 0.66] } });
const head = (url) => ({ kind: 'cluster', url, live: true, running: 1, waiting: 0, inflight: 1, depth_known: true,
  served_models: ['m'], max_model_len: 131072, history: {},
  throughput: { source: 'metrics', generated: { tok_s: 62 }, prompt: {}, engine_totals: {} } });
const status = (topology, boxes, saturation, b) => ({ now: NOW, backends: [b], leases: [], hardware: {},
  broker: { enabled: false }, config: {}, models: { cluster: [], classic: [] }, saturation,
  cluster: Object.assign({ configured: true, state: 'running', held: true,
    boxes: Object.fromEntries(boxes.map((x) => [x, { label: x, rank: { present: true, running: true }, hold: { blockers: [] } }])) },
  topology ? { topology } : {}) });

// What the LB sends at TP=1 (vllm_lb/topology.py NO_LINK) — read off site-b's node through a preview LB, 2026-10-12.
const NO_LINK = { rate: null, peak: null, pct: null, why: 'TP=1: one GPU, no link between ranks', boxes: {} };
const ONE = (box) => ({ tp: 1, boxes: [box], control: 'none', why: 'observed only', steps: ['load', 'warm up', 'serve'] });
const NET = { rate: 0.66, peak: 31.5, pct: 2.1, why: null,
  boxes: { 'pair-a': { tx_gbps: 0.66, rx_gbps: 0.64, link_gbps: 31.5 }, 'pair-b': { tx_gbps: 0.64, rx_gbps: 0.66, link_gbps: 31.5 } } };
const PAIR = { tp: 2, boxes: ['pair-a', 'pair-b'], control: 'ssh', why: '',
  steps: ['free', 'reserve', 'drain', 'load', 'warm up', 'serve'] };

// 1. TP=1: no network tile, no placeholder, no note; three tiles on a grid of three.
{
  const b = head('http://node-a:8000');
  const html = api.serverCard(status(ONE('node-a'), ['node-a'], sat(['node-a'], NO_LINK, CORES(21760)), b), b);
  const r = row(html);
  ok(r.tiles.length === 3, 'a one-rank card draws three ceiling tiles', r.tiles.map((t) => t.label));
  ok(JSON.stringify(r.tiles.map((t) => t.label))
     === JSON.stringify(['memory bandwidth · node-a', 'compute · node-a', 'CUDA cores / GPU']),
     'memory bandwidth, compute, CUDA cores — no network', r.tiles.map((t) => t.label));
  ok(!/>network</.test(html), 'no tile labelled network anywhere on the card');
  // The one place it IS named: the quiet "not collected" tags under the graphs (unused_graphs.node.js, operator
  // 2026-10-12). Nowhere else — no placeholder tile, no note.
  const rest = html.replace(/<div class="unused">.*?<\/div>/s, '');
  ok(!rest.includes('between ranks') && !rest.includes('TP=1: one GPU'), 'no "no link" note anywhere else on the card');
  ok(r.cls === 'ceils ceils3', 'the row is a grid of three, so no column stands empty', r.cls);
}

// 2. The rule is the topology's, never a name, never the server's reason.
{
  const b = head('http://box-a:8000');
  const named = api.serverCard(status(ONE('pair-a'), ['pair-a'], sat(['pair-a'], NO_LINK, CORES(6144)), b), b);
  ok(!/>network</.test(named) && row(named).cls === 'ceils ceils3', 'a one-rank node named like a pair box: none');
  const quiet = api.serverCard(status(ONE('node-a'), ['node-a'],
    sat(['node-a'], { rate: null, peak: null, pct: null, why: null, boxes: {} }, CORES(21760)), b), b);
  ok(!/>network</.test(quiet), 'an older LB that sends no reason: still none');
  const absent = api.serverCard(status(ONE('node-a'), ['node-a'], sat(['node-a'], undefined, CORES(21760)), b), b);
  ok(!/>network</.test(absent) && row(absent).tiles.length === 3, 'and no network block at all: none');
}

// 3. TP=1 with no cores estimate (no fleet card named): two tiles, a grid of two.
{
  const b = head('http://node-a:8000');
  const r = row(api.serverCard(status(ONE('node-a'), ['node-a'], sat(['node-a'], NO_LINK, null), b), b));
  ok(r.tiles.length === 2 && r.cls === 'ceils ceils2', 'two tiles, two across', [r.cls, r.tiles.map((t) => t.label)]);
}

// 4. TP=2: the tile where it always was, built from the same figures; the row of six stays four across.
{
  const b = head('http://pair-a:8100');
  const s = status(PAIR, ['pair-a', 'pair-b'], sat(['pair-a', 'pair-b'], NET, CORES(6144)), b);
  const html = api.serverCard(s, b);
  const r = row(html);
  ok(JSON.stringify(r.tiles.map((t) => t.label)) === JSON.stringify(['memory bandwidth · pair-a',
    'memory bandwidth · pair-b', 'compute · pair-a', 'compute · pair-b', 'network', 'CUDA cores / GPU']),
     'the pair: per-box tiles, then network, then the cores', r.tiles.map((t) => t.label));
  const net = r.tiles[4].html;
  const same = api.ceilingTile('network', NET, 'GB/s', 2, s.saturation.history.net_gbps);
  ok(net.trimEnd() === same.trimEnd(), 'the network tile is the one ceilingTile builds from the same figures');
  ok(/<b class="t-val"[^>]*>0\.66<span class="t-unit"> GB\/s</.test(net), 'its figure: 0.66 GB/s', strip(net));
  ok(/<span class="c-note"[^>]*>2 % of 31.5 GB\/s<\/span>/.test(net), 'its share of the link', strip(net));
  ok(/<span class="sat-bar"><i style="width:2%/.test(net), 'its bar at 2 %');
  ok(r.cls === 'ceils ceils6', 'a row of six', r.cls);
  // A document from before `topology`: its two boxes are its TP — the same card.
  const before = status(null, ['pair-a', 'pair-b'], s.saturation, b);
  ok(api.serverCard(before, b) === html, 'a pre-topology document draws the same card, network and all');
}

// 5. A link not measured YET keeps its tile: unknown is not not-applicable.
{
  const b = head('http://pair-a:8100');
  const waiting = { rate: null, peak: null, pct: null, why: 'awaiting 2nd RDMA reading', boxes: {} };
  const r = row(api.serverCard(status(PAIR, ['pair-a', 'pair-b'], sat(['pair-a', 'pair-b'], waiting, CORES(6144)), b), b));
  const net = r.tiles.find((t) => t.label === 'network');
  ok(net && net.html.includes('awaiting 2nd RDMA reading') && /class="ceil dim"/.test(net.html),
     'TP=2 awaiting a reading: the tile, dim, with its reason', net && strip(net.html));
}

// 6. The CSS: three across, two across; on a narrow card the third spans the row; nothing for five or six.
{
  ok(/\.ceils3\s*\{\s*grid-template-columns:\s*repeat\(3, minmax\(0, 1fr\)\);?\s*\}/.test(css), '.ceils3 is three across');
  ok(/\.ceils2\s*\{\s*grid-template-columns:\s*repeat\(2, minmax\(0, 1fr\)\);?\s*\}/.test(css), '.ceils2 is two across');
  // The card's narrow rule for the row: `@container (max-width: 700px) { .ceils { … } … }` (with its hairlines since
  // option D, 2026-10-30, over several lines).
  const narrow = css.match(/@container \(max-width: 700px\) \{\s*(\.ceils \{[\s\S]*?)\n\}/);
  ok(narrow && /^\.ceils\s*\{\s*grid-template-columns:\s*repeat\(2,/.test(narrow[1])
     && /\.ceils3 > \.ceil:last-child\s*\{\s*grid-column:\s*1 \/ -1;?\s*\}/.test(narrow[1]),
     'a narrow card: two across, the third tile spanning the row', narrow && narrow[1]);
  // The cores tile there spans ONE grid row: over four, its height sized the flexible row the two graphs above share
  // (`grid-auto-rows: … minmax(0, 1fr)`) and a ~90 px gap opened under them at 390 px (preview, 2026-10-12).
  ok(narrow && /\.ceils3 > \.gpu-cores:last-child\s*\{\s*grid-row:\s*span 1;?\s*\}/.test(narrow[1]),
     'and the cores tile alone on its row spans one grid row', narrow && narrow[1]);
  // Order matters: `.ceils3` must come BEFORE the narrow rule, or its three columns would win on a phone too.
  const wide = css.indexOf('.ceils3 {');
  ok(wide >= 0 && wide < css.search(/@container \(max-width: 700px\) \{\s*\.ceils \{/), 'the wide rule precedes the narrow');
  ok(!/\.ceils[4-9]\b/.test(css), 'no rule for a row of four or more: the pair keeps the base four columns');
}

console.log(`network_tile: ${checks} checks ok`);
