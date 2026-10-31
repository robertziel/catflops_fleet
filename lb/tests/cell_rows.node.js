/* EVERY GRAPH CELL IS ONE FIXED, COMPACT HEIGHT, whatever its data — operator 2026-10-30: "fix UX UI, to ensure each
 * graph cell is compact and doesn't change height over data change".
 *
 * THE BUG: a cell's height followed its data. QUEUE grew a line when "2 running · 4 queued" appeared under it and lost
 * it when the queue drained; a MEMORY BANDWIDTH cell's basis sentence wrapped to two lines while COMPUTE's beside it
 * took one; a cell reading "—" (no figure, no sparkline yet) was shorter than one with a figure; the summary row under
 * the tiles wrapped to one, two or three lines as its pairs came and went. Every one of those moved the card under the
 * operator's pointer on a probe.
 *
 * THE RULE, pinned here twice — in the markup (gateway.js) and in the sizing (gateway.css):
 *   1. a cell's ROWS are fixed by its kind, never by its data: the same structural slots, in the same order, with or
 *      without a subline, a value, a series, a long or short basis (an empty slot is still drawn);
 *   2. each slot has a FIXED block size (height, or min-height = height), and a text slot is ONE line, cut with an
 *      ellipsis — the whole text stays in the DOM (what a screen reader reads) and in the slot's `title`.
 * Sizing and overflow only: no colour is set here (another change owns the cells' backgrounds).
 *
 *   docker run --rm -v "$PWD":/lb:ro -w /lb node:22-slim node tests/cell_rows.node.js
 */
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const static_ = path.join(__dirname, '..', 'vllm_lb', 'static');
// palette.js first: the colours this script computes are read from the style's tokens (tests/palette_env.js).
const src = require('./palette_env').withPalette(fs.readFileSync(path.join(static_, 'gateway.js'), 'utf8'));
const css = fs.readFileSync(path.join(static_, 'gateway.css'), 'utf8');

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
  `${src}\n;return { tile, ceilingTile, satMeter, ceilingMeter, engineStats, machineBody };`)(
  document, window, () => new Promise(() => {}), { search: '', href: '' }, () => 0, () => 0,
  () => ({}), undefined, window.localStorage, { userAgent: 'node' });

/** The cell's structural slots, in document order: the classes that make a row of it (never a colour or a state). */
const SLOTS = ['tile', 'ceil', 'sat-meter', 't-head', 't-label', 't-val', 't-graph', 't-note', 'sat-bar', 'c-note',
  'c-trend', 'sat-label', 'sat-sub', 'sat-graph', 'sat-val', 'sat-note', 'sat-trend'];
const slots = (html) => [...html.matchAll(/class="([^"]*)"/g)].map((m) => m[1].split(/\s+/)[0])
  .filter((c) => SLOTS.includes(c));
const titleOf = (html, cls) => {
  const m = html.match(new RegExp(`class="${cls}(?: [^"]*)?"[^>]*title="([^"]*)"`));
  return m ? m[1] : null;
};
const LONG = 'no per-pass byte model is recorded for this model, and the engine publishes no byte estimate of its own '
  + 'to stand in for it, so nothing is drawn';
const series = [1, 2, 3, 4, 5, 6];

// ── 1. a throughput tile: head / graph / ONE note slot, with or without a subline or a value ──
const tiles = [
  api.tile('queue', 6, '', null, series, {}),
  api.tile('queue', 6, '', '2 running &middot; 4 queued', series, {}),
  api.tile('queue', null, '', `<span class="sat-why">${LONG}</span>`, [], {}),
  api.tile('generated', 1234.5, 'tok/s', null, [], {}),
];
const want = ['tile', 't-head', 't-label', 't-val', 't-graph', 't-note'];
tiles.forEach((h) => assert.deepStrictEqual(slots(h), want, h));
// the subline's whole text is the note's title (its tags dropped); an empty note has none
assert.strictEqual(titleOf(tiles[1], 't-note'), '2 running · 4 queued');
assert.strictEqual(titleOf(tiles[2], 't-note'), LONG);
assert.strictEqual(titleOf(tiles[0], 't-note'), null);
// …and still the whole text in the DOM, for a screen reader
assert.ok(tiles[2].includes(LONG));

// a machine card's graphs carry no subline BY KIND (`noNote`): head and graph, every one of them, never a note slot
// reserved for nothing — compact, and still fixed by the kind rather than the data
const MACHINE = { hardware: { 'box-a': { host: 'box-a', age_s: 1, stale: false, cpu_percent: 12, ram_percent: 40,
  ram_used_mb: 4096, ram_total_mb: 8192, swap_used_mb: 0, rows: [{ label: 'GPU', card: null, box_gpu: true,
    use_pct: 30, history: series, vram_label: null }], history: { cpu_pct: series, ram_pct: series } } } };
const machine = api.machineBody(MACHINE, 'box-a');
const cells = machine.split(/(?=<div class="tile[" ])/).filter((h) => /^<div class="tile[" ]/.test(h));
assert.ok(cells.length >= 3, machine);
cells.forEach((h) => assert.deepStrictEqual(slots(h), want.slice(0, 5), h));

// ── 2. a ceiling tile (memory bandwidth, compute): head / bar / ONE basis line / graph slot ──
const DERIVED = "derived: the engine's one rate per GPU (it has no per-rank counter) x this box's serving cards";
const ceils = [
  api.ceilingTile('memory bandwidth · box-a', { rate: 194.9, peak: 273, pct: 71.4, derived: DERIVED }, 'GB/s', 0, series),
  api.ceilingTile('memory bandwidth · box-a', { rate: null, peak: 273, pct: null, why: LONG, derived: DERIVED }, 'GB/s', 0, []),
  api.ceilingTile('compute · box-a', { rate: 2.2, peak: 323, pct: 1, peak_precision: 'nvfp4' }, 'TFLOP/s', 1, null, 'rate'),
];
const wantCeil = ['ceil', 't-head', 't-label', 't-val', 'sat-bar', 'c-note', 'c-trend'];
ceils.forEach((h) => assert.deepStrictEqual(slots(h), wantCeil, h));
assert.ok(titleOf(ceils[1], 'c-note').startsWith('no per-pass byte model'), ceils[1]);
assert.ok(titleOf(ceils[1], 'c-note').endsWith('· derived'), ceils[1]);
assert.strictEqual(titleOf(ceils[0], 't-head'), 'memory bandwidth · box-a');

// ── 3. a saturation meter: label / sub slot / value / bar / ONE note line / graph slot ──
const meters = [
  api.satMeter('generated', '12 tok/s', null, 'over 5 s', null, null, null, 'x', '<svg class="sparkline"></svg>'),
  api.satMeter('generated', null, null, `<span class="sat-why">${LONG}</span>`, null, null, null, 'x', ''),
  api.satMeter('memory bandwidth · box-a', '195 GB/s', 71, '71 % of 273 GB/s', null, 'modeled · derived', null, 'x', ''),
];
const wantMeter = ['sat-meter', 'sat-label', 'sat-sub', 'sat-val', 'sat-bar', 'sat-note', 'sat-trend'];
meters.forEach((h) => assert.deepStrictEqual(slots(h), wantMeter, h));
assert.strictEqual(titleOf(meters[1], 'sat-note'), LONG);
// a rank's meter carries its box's ring as well — by kind, so with or without a ring yet
const ranks = [
  api.satMeter('SM fill · box-a', '96 %', 96, 'clock 2424', series, 'rank 0 · head', null, null, '', true),
  api.satMeter('SM fill · box-a', null, null, 'not observed', null, 'rank 0 · head', null, null, '', true),
];
ranks.forEach((h) => assert.deepStrictEqual(slots(h),
  ['sat-meter', 'sat-label', 'sat-sub', 'sat-graph', 'sat-val', 'sat-bar', 'sat-note', 'sat-trend'], h));
// ceilingMeter routes its history into the graph slot, never into the one-line note it would be cut out of
const cm = api.ceilingMeter('memory bandwidth · box-a', { rate: 194.9, peak: 273, pct: 71.4 }, 'GB/s', 0, series);
assert.ok(/class="sat-trend"[^>]*>\s*<svg/.test(cm), cm);
assert.ok(!/class="sat-note"[^>]*>[^<]*<svg/.test(cm), cm);

// ── 4. the summary row is ONE line however many pairs it carries, the whole row in its title ──
const b = { running: 2, waiting: 4, depth_known: true, throughput: {
  prefill_speed: { tok_s: 1234, why: null }, prompt: { cached_share: 0.9 },
  engine_totals: { generated: 123456, prompt: 654321, computed: 55555 },
  ours: { uncounted: {}, counted: 12 } } };
const stats = api.engineStats(b, true, {}, 'mtp', 0, 3600);
assert.ok(/^<div class="stats" title="[^"]*prefill \/ request 1,234 tok\/s · cached 90 %[^"]*uncounted 0/.test(stats), stats);

// ── 5. the sizing: fixed block sizes and one-line cuts in gateway.css ──
/** Every declaration the stylesheet gives `selector` (the rules whose selector list names it exactly). */
function decls(selector) {
  const out = {};
  for (const m of css.replace(/\/\*[\s\S]*?\*\//g, '').matchAll(/([^{}@]+)\{([^{}]*)\}/g)) {
    const sels = m[1].split(',').map((s) => s.trim());
    if (!sels.includes(selector) && m[1].trim() !== selector) continue;   // a whole `:is(a, b)` list too
    for (const d of m[2].split(';')) {
      const i = d.indexOf(':');
      if (i > 0) out[d.slice(0, i).trim()] = d.slice(i + 1).trim();
    }
  }
  return out;
}
const oneLine = (sel) => {
  const d = decls(sel);
  assert.strictEqual(d['white-space'], 'nowrap', `${sel} ${JSON.stringify(d)}`);
  assert.strictEqual(d.overflow, 'hidden', sel);
  assert.strictEqual(d['text-overflow'], 'ellipsis', sel);
};
const fixed = (sel) => {
  const d = decls(sel);
  assert.ok(d.height && !/auto|%/.test(d.height), `${sel} has a fixed height: ${JSON.stringify(d)}`);
};
['.t-note', '.c-note', '.sat-note', '.sat-sub', '.t-label', '.sat-label', '.stats'].forEach(oneLine);
['.t-head', '.t-note', '.t-graph', '.c-note', '.c-trend', '.sat-sub', '.sat-graph', '.sat-note', '.sat-trend',
  '.stats'].forEach(fixed);
assert.strictEqual(decls('.t-head')['flex-wrap'], 'nowrap');
assert.strictEqual(decls('.stat').display, 'inline');

// ── 6. option D (operator 2026-10-30): one inset band per metric group, cells divided by hairlines, no fill a cell ──
const band = decls(':is(.tiles3, .tiles4, .tiles5, .tiles6)');
assert.strictEqual(band.background, 'var(--panel-2)');
assert.strictEqual(band['border-radius'], 'var(--radius-band)');   // 10px in "Vibe coded" (gateway.css :root)
const cell = decls(':is(.tiles3, .tiles4, .tiles5, .tiles6) > .tile');
assert.strictEqual(cell['grid-template-rows'], 'subgrid', 'a tile is a box on its row\'s rows, not `display: contents`');
assert.strictEqual(cell['border-left'], 'var(--rule)');   // 1px solid var(--line) in "Vibe coded"
assert.ok(!('background' in cell), 'no fill per cell');
assert.strictEqual(decls(':is(.tiles3, .tiles4, .tiles5, .tiles6) > .tile:first-child')['border-left'], '0');
assert.ok(/\.ceils \{[^}]*background: var\(--panel-2\)/.test(css));
assert.strictEqual(decls('.ceils > .ceil')['border-left'], 'var(--rule)');

console.log('ok');
