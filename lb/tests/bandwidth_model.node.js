/* THE MODELED MEMORY BANDWIDTH, drawn where it is built: gateway.js, under node.
 *
 * For an SGLang generation the LB does not draw the engine's own byte estimate (73,600 GB/s a GPU on a
 * 273 GB/s part, live 2026-10-03) but its OWN model of what each forward pass moves, from SGLang's
 * counters (`saturation.bandwidth_model`, lb/vllm_lb/throughput.py `bandwidth_model`). The page has to
 * say which of the two a figure is — a model is labelled one, its parts are figures in the hover, and a
 * floor (a window that held a prefill pass) reads "≥". An engine's own estimate is drawn as it always was.
 * The CUDA CORES tile's hover names its bytes the same way.
 *
 *   docker run --rm -v "$PWD":/lb:ro -w /lb node:22-slim node tests/bandwidth_model.node.js
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
    `${src}\n;return { renderSaturation, ceilingTiles, coresTile };`)(
    document, window, () => new Promise(() => {}), { search: '', href: '' }, () => 0, () => 0,
    () => ({}), undefined, window.localStorage, { userAgent: 'node' });
  return { api, byId };
}

const strip = (h) => h.replace(/<[^>]*>/g, '').replace(/&middot;/g, '·').replace(/&ge;/g, '≥')
  .replace(/&nbsp;/g, ' ').replace(/&amp;/g, '&').replace(/&#39;/g, "'").replace(/\s+/g, ' ').trim();

/** The live pair's window (tests/test_sglang_bandwidth_model.py): 99.8 GB/s a GPU, a floor. */
const MODEL = { gbps_per_gpu: 99.8, window_s: 5.0, why: null,
  floor: 'the window held prefill passes, charged their weights only — not the prefix KV their attention reads — so the figure is a floor',
  weights_gbps: 27.9, kv_gbps: 54.5, state_gbps: 17.3, decode_passes_s: 2.19, prefill_passes_s: 0.4,
  context_tokens: 1502161, seqs: 30, verify_width: 4 };

/** The pair's per-box split as the LB sends it (saturation.py `per_box`): each GB10 its own share, derived. */
const DERIVED = "derived: the engine's one rate per GPU (it has no per-rank counter) x this box's serving cards — tensor parallelism splits each layer's work evenly over the ranks";
const split = (c) => ({ ...c, boxes: ['box-a', 'box-b'].map((box) => ({ box, cards: 1, rate: c.rate, peak: c.peak,
  pct: c.pct, why: null, derived: DERIVED })) });

function sat(model, rate) {
  return { verdict: 'busy — no ceiling measured', because: 'x', headroom: null, sm_fill_pct: 96,
    sm_fill_means: 'x', compute: split({ rate: 126.4, peak: null, pct: null, why: null, peak_precision: null }),
    bandwidth: split({ rate, peak: 273, pct: Math.round(rate / 2.73 * 10) / 10, why: null }),
    bandwidth_model: model, per_gpu: true, gpu_count: 2, cluster_tflops: 252.8, cluster_gbps: rate * 2,
    window_s: 5, window_widened: null, rates_from: 'box-a', rates_means: 'means', boxes: [],
    sm_fill_gap_pts: null, history: {}, network: { rate: null, peak: null, pct: null, why: 'x' }, cores: null };
}

/** The saturation panel's bandwidth meter, read the way a person reads it. */
function meter(status) {
  const { api, byId } = load();
  api.renderSaturation(status);
  const html = byId['cluster-saturation'].innerHTML;
  // `sat-meter( dim)?`, never the `sat-meters` group that holds them.
  const m = [...html.matchAll(/<div class="sat-meter(?: [^"]*)?"([^>]*)>([\s\S]*?)<\/div>/g)]
    .find((x) => /class="sat-label">memory bandwidth · box-a</.test(x[2]));
  assert.ok(m, 'the bandwidth meter is drawn');
  return { title: strip((m[1].match(/title="([^"]*)"/) || [, ''])[1]),
    sub: strip((m[2].match(/class="sat-sub">([\s\S]*?)<\/span>/) || [, ''])[1]),
    value: strip((m[2].match(/<b class="sat-val">([\s\S]*?)<\/b>/) || [, ''])[1]) };
}

// ── 1. a modeled figure is labelled a model, its parts in the hover, ≥ on a floor ──
const modeled = meter({ backends: [], hardware: {}, saturation: sat(MODEL, 99.8) });
assert.strictEqual(modeled.sub, 'modeled · derived');
assert.strictEqual(modeled.value, '≥ 100 GB/s');
for (const part of ['weights 28', 'KV 55', 'GDN state 17', '2.2 decode', '0.4 prefill'])
  assert.ok(modeled.title.includes(part), `${part} in "${modeled.title}"`);
assert.ok(modeled.title.includes('floor'), modeled.title);

// …and not a floor: no ≥.
const exact = meter({ backends: [], hardware: {}, saturation: sat({ ...MODEL, floor: null }, 125.4) });
assert.strictEqual(exact.value, '125 GB/s');
assert.ok(!exact.title.includes('floor'));

// ── 2. an engine's own estimate is not a model: only "derived" (each box's share of the pair's one rate) ──
const engine = meter({ backends: [], hardware: {}, saturation: sat(null, 220) });
assert.strictEqual(engine.sub, 'derived');
assert.strictEqual(engine.value, '220 GB/s');
assert.strictEqual(engine.title, DERIVED);

// ── 3. the tile row under the throughput tiles says the same ──
const { api } = load();
const tile = (s) => {
  const html = api.ceilingTiles(s);
  // `ceil( dim)?`, never the `ceils` row that holds the tiles.
  const t = html.match(/<div class="ceil(?: [^"]*)?"([^>]*)>([\s\S]*?)<span class="c-note"[^>]*>/);
  return { title: strip((t[1].match(/title="([^"]*)"/) || [, ''])[1]),
    label: strip(t[2].match(/class="t-label">([\s\S]*?)<\/span>/)[1]),
    value: strip(t[2].match(/<b class="t-val"[^>]*>([\s\S]*?)<\/b>/)[1]) };
};
const t1 = tile(sat(MODEL, 99.8));
assert.strictEqual(t1.label, 'memory bandwidth · box-a · modeled');
assert.strictEqual(t1.value, '≥100 GB/s');
assert.ok(t1.title.includes('KV 55'), t1.title);
const t2 = tile(sat(null, 220));
assert.strictEqual(t2.label, 'memory bandwidth · box-a');
assert.strictEqual(t2.value, '220 GB/s');
assert.strictEqual(t2.title, DERIVED);

// ── 3b. a TensorFold generation (operator 2026-10-30, "Fix bandwidth in tensor fold"): GB/s, % of the GB10's 273,
// "modeled", ≥ because its drafter's passes are uncounted — and the hover names the engine's counters, not SGLang's ──
const TF_OMITS = "the MTP drafter's passes are not charged: TensorFold counts no draft pass";
const tf = tile(sat({ ...MODEL, floor: TF_OMITS, weights_gbps: 190.1, kv_gbps: 0.9, state_gbps: 3.9 }, 194.9));
assert.strictEqual(tf.label, 'memory bandwidth · box-a · modeled');
assert.strictEqual(tf.value, '≥195 GB/s');
assert.ok(tf.title.includes("modeled from the engine's own pass counters"), tf.title);
assert.ok(!tf.title.includes('SGLang'), tf.title);
assert.ok(tf.title.includes('drafter'), tf.title);
const tfNote = strip(api.ceilingTiles(sat({ ...MODEL, floor: TF_OMITS }, 194.9)).match(/<span class="c-note"[^>]*>([\s\S]*?)<\/span>/)[1]);
assert.ok(tfNote.includes('71 % of 273 GB/s'), tfNote);

// ── 4. the CUDA CORES hover names what its bytes are ──
const cores = (bytes) => api.coresTile({ cores: 6144, multiplying: 224, memory: 2366, other: 3554, tflops: 12.3,
  gbps: 105.1, steps_s: 2.5, tokens_step: 91, mixed_share: 0, idle_share: 0, bytes,
  cases: { decode: null, prefill: null } });
assert.ok(cores('modeled').includes('105.1 GB/s modeled (weights, KV, GDN state)'), cores('modeled'));
assert.ok(cores('weights').includes('105.1 GB/s of weights'));

console.log('ok');
