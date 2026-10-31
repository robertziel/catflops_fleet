/* COMPUTE AND MEMORY BANDWIDTH PER BOX USED, drawn where they are built: gateway.js, under node.
 *
 * Operator, 2026-10-11: never one figure for the LB or the cluster. The TP=2 pair shows box-a and box-b
 * SEPARATELY, each its own rate against ITS OWN box's peak ("box-a 4 % of 323 · box-b 4 % of 323"), and a
 * one-host node its serving box against that card's peak. The server sends the split (/lb/status
 * `compute.boxes`, `bandwidth.boxes`: saturation.per_box); where more than one card serves, a box's rate is
 * DERIVED from the engine's one rate — and the page says "derived", the sentence in the hover, never presenting
 * it as measured.
 *
 *   docker run --rm -v "$PWD":/lb:ro -w /lb node:22-slim node tests/per_box_ceilings.node.js
 */
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');

// palette.js first: the colours this script computes are read from the style's tokens (tests/palette_env.js).
const src = require('./palette_env').withPalette(fs.readFileSync(path.join(__dirname, '..', 'vllm_lb', 'static', 'gateway.js'), 'utf8'));

function load() {
  const byId = {};
  const el = () => ({ addEventListener() {}, removeAttribute() {}, setAttribute() {}, append() {},
    classList: { toggle() {}, add() {}, remove() {} }, style: {}, hidden: false, innerHTML: '',
    textContent: '', value: '', dataset: {}, querySelectorAll: () => [], querySelector: () => null,
    focus() {}, observe() {}, closest: () => null, getBoundingClientRect: () => ({}) });
  const document = {
    getElementById: (id) => (byId[id] || (byId[id] = el())),
    createElementNS: el, createElement: el, addEventListener() {},
    querySelectorAll: () => [], querySelector: () => null,
    body: { style: {}, classList: { toggle() {} } },
    documentElement: { style: {}, classList: { toggle() {} } } };
  const window = { addEventListener() {}, innerWidth: 1280, innerHeight: 800, matchMedia: () => ({
    matches: false, addEventListener() {} }), localStorage: { getItem: () => null, setItem() {} } };
  const api = new Function('document', 'window', 'fetch', 'location', 'setInterval', 'setTimeout',
    'getComputedStyle', 'ResizeObserver', 'localStorage', 'navigator',
    `${src}\n;return { renderSaturation, ceilingTiles };`)(
    document, window, () => new Promise(() => {}), { search: '', href: '' }, () => 0, () => 0,
    () => ({}), undefined, window.localStorage, { userAgent: 'node' });
  return { api, byId };
}

const strip = (h) => h.replace(/<[^>]*>/g, '').replace(/&middot;/g, '·').replace(/&ge;/g, '≥')
  .replace(/&nbsp;/g, ' ').replace(/&amp;/g, '&').replace(/&#39;/g, "'").replace(/\s+/g, ' ').trim();

const DERIVED = "derived: the engine's one rate per GPU (it has no per-rank counter) x this box's serving cards — " +
  "tensor parallelism splits each layer's work evenly over the ranks";

/** The pair as the LB serves it since per-box peaks (vllm_lb/proxy.py `_saturation`): 12.0 TFLOP/s and 220 GB/s a
 *  GPU, each GB10 at its own 323.2 / 273. */
function pair() {
  const box = (name, rate, peak, pct) => ({ box: name, cards: 1, rate, peak, pct, why: null, derived: DERIVED });
  return { verdict: 'memory-bandwidth bound', because: 'x', headroom: null, sm_fill_pct: 96, sm_fill_means: 'x',
    compute: { rate: 12.0, peak: 323.2, pct: 3.7, why: null, peak_precision: 'nvfp4',
      boxes: [box('box-a', 12.0, 323.2, 3.7), box('box-b', 12.0, 323.2, 3.7)] },
    bandwidth: { rate: 220.0, peak: 273, pct: 80.6, why: null,
      boxes: [box('box-a', 220.0, 273, 80.6), box('box-b', 220.0, 273, 80.6)] },
    bandwidth_model: null, compute_model: null, per_gpu: true, gpu_count: 2, cluster_tflops: 24.0,
    cluster_gbps: 440.0, window_s: 5, window_widened: null, rates_from: 'box-a', rates_means: 'means', boxes: [],
    sm_fill_gap_pts: null, history: { tflops: [11, 12], gbps: [210, 220] },
    network: { rate: null, peak: null, pct: null, why: 'x' }, cores: null };
}

/** The markup of the <span> that opens at `at`, nested spans and all. */
function spanAt(html, at) {
  let depth = 0;
  const re = /<span\b|<\/span>/g;
  re.lastIndex = at;
  for (let m; (m = re.exec(html));) {
    depth += m[0] === '</span>' ? -1 : 1;
    if (depth === 0) return html.slice(at, m.index + m[0].length);
  }
  return html.slice(at);
}

/** The tiles under the throughput row, as read: label, value, note, hover. */
function tiles(sat) {
  const { api } = load();
  const html = api.ceilingTiles(sat);
  return [...html.matchAll(/<div class="ceil(?: [^"]*)?"([^>]*)>([\s\S]*?)(?=<span class="c-note")/g)]
    .map((m) => ({ title: strip((m[1].match(/title="([^"]*)"/) || [, ''])[1]),
      label: strip((m[2].match(/class="t-label">([\s\S]*?)<\/span>/) || [, ''])[1]),
      value: strip((m[2].match(/<b class="t-val"[^>]*>([\s\S]*?)<\/b>/) || [, ''])[1]),
      note: strip(spanAt(html, m.index + m[0].length)) }));
}

/** The saturation panel's meters, as read. */
function meters(sat) {
  const { api, byId } = load();
  api.renderSaturation({ backends: [], hardware: {}, saturation: sat });
  return [...byId['cluster-saturation'].innerHTML.matchAll(/<div class="sat-meter(?: [^"]*)?"([^>]*)>([\s\S]*?)<\/div>/g)]
    .map(([, attrs, body]) => ({ title: strip((attrs.match(/title="([^"]*)"/) || [, ''])[1]),
      label: strip((body.match(/class="sat-label">([\s\S]*?)<\/span>/) || [, ''])[1]),
      sub: strip((body.match(/class="sat-sub">([\s\S]*?)<\/span>/) || [, ''])[1]),
      value: strip((body.match(/<b class="sat-val">([\s\S]*?)<\/b>/) || [, ''])[1]),
      note: strip((body.match(/class="sat-note"[^>]*>([\s\S]*)$/) || [, ''])[1]) }))
    .filter((m) => /^(memory bandwidth|compute)/.test(m.label));
}

let checks = 0;

// 1. The tile row: one bandwidth and one compute tile PER BOX, each its own rate and its own share.
{
  const t = tiles(pair()).filter((x) => /^(memory bandwidth|compute)/.test(x.label));
  assert.deepStrictEqual(t.map((x) => x.label),
    ['memory bandwidth · box-a', 'memory bandwidth · box-b', 'compute · box-a', 'compute · box-b']);
  assert.deepStrictEqual(t.map((x) => x.value), ['220 GB/s', '220 GB/s', '12.0 TFLOP/s', '12.0 TFLOP/s']);
  assert.deepStrictEqual(t.map((x) => x.note), ['81 % of 273 GB/s · derived', '81 % of 273 GB/s · derived',
    '4 % of 323 TFLOP/s (nvfp4) · derived', '4 % of 323 TFLOP/s (nvfp4) · derived']);
  assert.ok(t.every((x) => x.title.includes('derived: the engine')), 'the derivation is in the hover');
  checks += 4;
}

// 2. The saturation panel says the same, per box, "derived" under each label.
{
  const m = meters(pair());
  assert.deepStrictEqual(m.map((x) => x.label),
    ['memory bandwidth · box-a', 'memory bandwidth · box-b', 'compute · box-a', 'compute · box-b']);
  assert.ok(m.every((x) => x.sub === 'derived'), `each says derived: ${m.map((x) => x.sub)}`);
  assert.deepStrictEqual(m.map((x) => x.value), ['220 GB/s', '220 GB/s', '12.0 TFLOP/s', '12.0 TFLOP/s']);
  checks += 3;
}

// 3. Two different cards: each box against ITS OWN peak — the same rate is two different shares.
{
  const s = pair();
  s.bandwidth.boxes[1] = { ...s.bandwidth.boxes[1], box: 'other', peak: 1792, pct: 12.3 };
  const t = tiles(s).filter((x) => x.label.startsWith('memory bandwidth'));
  assert.deepStrictEqual(t.map((x) => x.note), ['81 % of 273 GB/s · derived', '12 % of 1,792 GB/s · derived']);
  checks += 1;
}

// 4. One card is the whole engine: its box's figure is the engine's own, NOT derived.
{
  const s = pair();
  s.compute.boxes = [{ box: 'solo', cards: 1, rate: 12.0, peak: 1676, pct: 0.7, why: null, derived: null }];
  s.bandwidth.boxes = [{ box: 'solo', cards: 1, rate: 220.0, peak: 1792, pct: 12.3, why: null, derived: null }];
  const t = tiles(s).filter((x) => /^(memory bandwidth|compute)/.test(x.label));
  assert.deepStrictEqual(t.map((x) => [x.label, x.note]),
    [['memory bandwidth · solo', '12 % of 1,792 GB/s'], ['compute · solo', '1 % of 1,676 TFLOP/s (nvfp4)']]);
  assert.ok(meters(s).every((x) => x.sub === ''), 'nothing derived, nothing labelled so');
  checks += 2;
}

// 5. A box whose card states no peak: no share, and the card type's own reason.
{
  const s = pair();
  s.compute.boxes = s.compute.boxes.map((b) => ({ ...b, peak: null, pct: null, why: "N/A for this card" }));
  const t = tiles(s).filter((x) => x.label.startsWith('compute'));
  assert.ok(t.every((x) => x.note === 'N/A for this card · derived'), t.map((x) => x.note));
  checks += 1;
}

console.log(`ok - ${checks} checks: compute and memory bandwidth are drawn per box used, derived figures say so`);
