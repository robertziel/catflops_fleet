/* A configuration's ONE encoding on /benchmarks, checked where it is computed: benchmarks.js `cfgStyle`.
 *
 * Every chart, legend line, matrix square and card draws a real-traffic configuration the same way: its
 * cap's step on the ordinal blue ramp, and — when a second configuration shares that cap — a hatched fill
 * for the second (135° for a third, a dotted outline past that), never a second hue. The ranks come from
 * `recomputeDomains`, over every minute AND every boot loaded, so a configuration that only ever booted has
 * a style too. No DOM: the file is loaded with a stub `document` and a `fetch` that never answers.
 *
 *   node tests/level_style.node.js          (tests/test_level_style.py runs this when node exists)
 */
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');

// the page loads server-scope.js first (its `serverScope`): so does this harness
const src = fs.readFileSync(path.join(__dirname, '..', 'vllm_lb', 'static', 'server-scope.js'), 'utf8') + '\n'
  + fs.readFileSync(path.join(__dirname, '..', 'vllm_lb', 'static', 'benchmarks.js'), 'utf8');
const el = () => ({ addEventListener() {}, setAttribute() {}, append() {}, classList: { toggle() {}, add() {} },
  style: {}, hidden: false, innerHTML: '', value: '', dataset: {}, querySelectorAll: () => [], observe() {} });
const document = { getElementById: el, createElementNS: el, createElement: el, addEventListener() {},
  querySelectorAll: () => [], body: { style: {} } };
const window = { addEventListener() {}, innerWidth: 1280, innerHeight: 800 };
const api = new Function('document', 'window', 'fetch', 'location', 'setInterval', 'getComputedStyle', 'ResizeObserver',
  `${src}\n;return { perf, state, recomputeDomains, cfgStyle, variantKey, VARIANTS, tooFewAt, MIN_PAIRS, MIN_WINDOWS, CURVE };`)(
  document, window, () => new Promise(() => {}), { search: '' }, () => 0, () => ({}), undefined);

const cfg = (seqs, batched, spec) => ({ model_choice: 'm', max_num_seqs: seqs, max_num_batched_tokens: batched, num_speculative_tokens: spec, max_model_len: 262144 });
const win = (c, t) => ({ t_start: t, t_end: t + 60, span_s: 60, config: c, config_source: 'effective' });
/** /lb/perf `models` for these minutes, all of model 'm' (perflog.recorded_models). */
const recorded = (ws) => [{ id: 'm', label: 'm', known: false, minutes: ws.length, first: ws[0].t_start, last: ws[ws.length - 1].t_end }];
/** A variant key's spec|batched — the parts these fixtures vary (benchmarks.js `variantKey` carries every
 *  part but the model and the cap; the rest are equal here). */
const sb = (ks) => (ks || []).map((k) => k.split('|').slice(0, 2).join('|'));
let checks = 0;

// 1. Tonight's four configurations, and one that only booted: caps rank on the ramp, variants at one cap
//    rank by spec then batched — the first solid, the second hatched — and the boot-only one has a style.
{
  const windows = [win(cfg(16, 16384, 4), 0), win(cfg(8, 8192, 4), 60), win(cfg(24, 8192, 4), 120), win(cfg(16, 8192, 4), 180)];
  api.perf.data = { windows, models: recorded(windows),
    boots: [{ t_start: 0, t_end: 1, config: cfg(24, 16384, 4), config_source: 'launch', outcome: 'serving' }] };
  api.recomputeDomains();
  assert.deepStrictEqual(api.state.caps, [8, 16, 24]);
  assert.deepStrictEqual(sb(api.perf.variants.get(16)), ['4|8192', '4|16384']);
  assert.deepStrictEqual(sb(api.perf.variants.get(24)), ['4|8192', '4|16384']);   // the boot counts
  const s8 = api.cfgStyle(cfg(8, 8192, 4)), s16 = api.cfgStyle(cfg(16, 8192, 4)), s16b = api.cfgStyle(cfg(16, 16384, 4));
  const s24 = api.cfgStyle(cfg(24, 8192, 4)), s24b = api.cfgStyle(cfg(24, 16384, 4));
  assert.deepStrictEqual([s8.color, s16.color, s24.color], ['var(--seq-1)', 'var(--seq-3)', 'var(--seq-5)']);   // dim → bright
  assert.strictEqual(s16b.color, s16.color);                     // same cap, same blue: fill tells them apart
  assert.deepStrictEqual([s8.variant, s16.variant, s16b.variant, s24.variant, s24b.variant], ['solid', 'solid', 'hatch', 'solid', 'hatch']);
  assert.strictEqual(s16b.dash, '6 3'); assert.strictEqual(s16.dash, null);   // the time charts dash the variant
  checks += 9;
}
// 2. A third and a fourth configuration at one cap: 135°, then a dotted outline — never a fifth kind.
{
  const windows = [win(cfg(8, 8192, 0), 0), win(cfg(8, 8192, 4), 60), win(cfg(8, 16384, 4), 120), win(cfg(8, 32768, 4), 180), win(cfg(8, 65536, 4), 240)];
  api.perf.data = { windows, models: recorded(windows), boots: [] };
  api.recomputeDomains();
  assert.deepStrictEqual(sb(api.perf.variants.get(8)), ['0|8192', '4|8192', '4|16384', '4|32768', '4|65536']);
  assert.deepStrictEqual(['0|8192', '4|8192', '4|16384', '4|32768', '4|65536'].map((k) => {
    const [sp, bt] = k.split('|').map(Number); return api.cfgStyle(cfg(8, bt, sp)).variant;
  }), ['solid', 'hatch', 'hatch2', 'dotted', 'dotted']);
  assert.strictEqual(api.cfgStyle(cfg(8, 8192, 0)).color, 'var(--seq-3)');   // one cap: the middle step
  checks += 3;
}
// 3. A cap the recorder never named gets the neutral ink and the solid fill — a step it did not earn, never.
{
  const s = api.cfgStyle({});
  assert.strictEqual(s.color, 'var(--ink-faint)'); assert.strictEqual(s.variant, 'solid');
  checks += 2;
}
// 4. Shown, not compared: under MIN_PAIRS pairs at the level, or a configuration under MIN_WINDOWS minutes.
{
  assert.strictEqual(api.tooFewAt({ minutes: api.MIN_WINDOWS }, { pairs: api.MIN_PAIRS }), false);
  assert.strictEqual(api.tooFewAt({ minutes: api.MIN_WINDOWS }, { pairs: api.MIN_PAIRS - 1 }), true);
  assert.strictEqual(api.tooFewAt({ minutes: api.MIN_WINDOWS - 1 }, { pairs: 1000 }), true);
  checks += 3;
}
// 5. Four measures lead, the rest are one fold away — and the four are the page's question.
{
  assert.deepStrictEqual(api.CURVE.filter((c) => c.lead).map((c) => c.key), ['agg_tok_s', 'per_request', 'ttft_mean', 'prompt_tok_s']);
  assert.strictEqual(api.CURVE.filter((c) => !c.lead).length, 7);
  checks += 2;
}
console.log(`ok — ${checks} checks`);
