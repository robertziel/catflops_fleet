/* THE MODELED COMPUTE, drawn where it is built: gateway.js, under node.
 *
 * For an SGLang generation the COMPUTE tile and meter draw the LB's OWN FLOP model — the one the CUDA CORES
 * tile prices its cores with (`saturation.compute_model`, lb/vllm_lb/cores.py `modeled_flops`) — not
 * SGLang's own counter, which read 236.8 TFLOP/s beside a cores tile at 742 of 6,144 (2026-10-03). The page
 * says which of the two a figure is: "modeled" under the label, the parts and SGLang's own figure in the
 * hover. An engine's own estimate (vLLM) is drawn as it always was.
 *
 *   docker run --rm -v "$PWD":/lb:ro -w /lb node:22-slim node tests/compute_model.node.js
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

/** The live chunks (tests/test_sglang_compute_model.py CHUNKS): five 8,192-token passes at 7.95 s each. */
const MODEL = { tflops_per_gpu: 52.76, window_s: 39.76, why: null, engine_tflops: 125.6,
  prefill_passes: 5, prefill_tokens: 40960, prefill_chunk: 8192, prefill_depth: 135168.5,
  depth_from: 'engine', prefill_s: 39.76, prefill_tflops: 52.76, decode_passes: 0, decode_s: 0,
  decode_tflops: null, busy_s: 39.76, split: 'one kind' };

/** The pair's per-box split as the LB sends it (saturation.py `per_box`): each GB10 its own share, derived. */
const DERIVED = "derived: the engine's one rate per GPU (it has no per-rank counter) x this box's serving cards — tensor parallelism splits each layer's work evenly over the ranks";
const split = (c) => ({ ...c, boxes: ['box-a', 'box-b'].map((box) => ({ box, cards: 1, rate: c.rate, peak: c.peak,
  pct: c.pct, why: null, derived: DERIVED })) });

function sat(model, rate) {
  return { verdict: 'busy — no ceiling measured', because: 'x', headroom: null, sm_fill_pct: 96,
    sm_fill_means: 'x', compute: split({ rate, peak: 323.2, pct: Math.round(rate / 3.232 * 10) / 10, why: null,
      peak_precision: 'nvfp4' }),
    bandwidth: split({ rate: 99.8, peak: 273, pct: 36.6, why: null }), bandwidth_model: null, compute_model: model,
    per_gpu: true, gpu_count: 2, cluster_tflops: rate * 2, cluster_gbps: 199.6, window_s: 40,
    window_widened: null, rates_from: 'box-a', rates_means: 'means', boxes: [], sm_fill_gap_pts: null,
    history: {}, network: { rate: null, peak: null, pct: null, why: 'x' }, cores: null };
}

/** The saturation panel's compute meter, read the way a person reads it. */
function meter(status) {
  const { api, byId } = load();
  api.renderSaturation(status);
  const html = byId['cluster-saturation'].innerHTML;
  const m = [...html.matchAll(/<div class="sat-meter(?: [^"]*)?"([^>]*)>([\s\S]*?)<\/div>/g)]
    .find((x) => /class="sat-label">compute · box-a</.test(x[2]));
  assert.ok(m, 'the compute meter is drawn');
  return { title: strip((m[1].match(/title="([^"]*)"/) || [, ''])[1]),
    sub: strip((m[2].match(/class="sat-sub">([\s\S]*?)<\/span>/) || [, ''])[1]),
    value: strip((m[2].match(/<b class="sat-val">([\s\S]*?)<\/b>/) || [, ''])[1]) };
}

// ── 1. a modeled figure is labelled a model; its parts and SGLang's own estimate are in the hover ──
const modeled = meter({ backends: [], hardware: {}, saturation: sat(MODEL, 52.8) });
assert.strictEqual(modeled.sub, 'modeled · derived');
assert.strictEqual(modeled.value, '52.8 TFLOP/s');
for (const part of ['prefill 52.8 TFLOP/s', '5 passes', '8,192 tok', '7.95 s', '135k deep',
  "SGLang's own estimate 125.6"])
  assert.ok(modeled.title.includes(part), `${part} in "${modeled.title}"`);
assert.ok(!modeled.title.includes('≥'));

// ── 2. an engine's own estimate is drawn as it always was ──
const engine = meter({ backends: [], hardware: {}, saturation: sat(null, 12.3) });
assert.strictEqual(engine.sub, 'derived');
assert.strictEqual(engine.value, '12.3 TFLOP/s');
assert.strictEqual(engine.title, DERIVED);

// ── 3. the tile row under the throughput tiles says the same ──
const { api } = load();
const tile = (s) => {
  const html = api.ceilingTiles(s);
  const tiles = [...html.matchAll(/<div class="ceil(?: [^"]*)?"([^>]*)>([\s\S]*?)<span class="c-note"[^>]*>/g)];
  const t = tiles.find((x) => /class="t-label">compute/.test(x[2]));
  assert.ok(t, 'the compute tile is drawn');
  return { title: strip((t[1].match(/title="([^"]*)"/) || [, ''])[1]),
    label: strip(t[2].match(/class="t-label">([\s\S]*?)<\/span>/)[1]),
    value: strip(t[2].match(/<b class="t-val"[^>]*>([\s\S]*?)<\/b>/)[1]) };
};
const t1 = tile(sat(MODEL, 52.8));
assert.strictEqual(t1.label, 'compute · box-a · modeled');
assert.strictEqual(t1.value, '52.8 TFLOP/s');
assert.ok(t1.title.includes("SGLang's own estimate 125.6"), t1.title);
const t2 = tile(sat(null, 12.3));
assert.strictEqual(t2.label, 'compute · box-a');
assert.strictEqual(t2.title, DERIVED);

console.log('ok');
