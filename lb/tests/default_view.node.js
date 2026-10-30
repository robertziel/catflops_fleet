/* EVERYTHING RECORDED IS WHAT LOADS on /benchmarks, checked where the defaults are set and where the
 * matrix's groups are built: benchmarks.js `perf.days`, `groupsOf`, `ofModel`.
 *
 * Two things used to narrow the page before the operator had touched it:
 *   1. the range picker opened on `7` — a window, not the log;
 *   2. the matrix drew the model ON SHOW alone, so a visit that landed on a model with four configurations
 *      hid every other configuration in the same payload, with nothing on the page saying they existed.
 * What must NOT change in fixing either: a configuration is still one model's, the charts still compare one
 * model at a time, and the hatch rank is still taken among a model's own configurations. The matrix is the
 * INVENTORY — it widens; the comparison does not.
 *
 *   node tests/default_view.node.js        (tests/test_default_view.py runs this when node exists)
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
const store = new Map();
const window = { addEventListener() {}, innerWidth: 1280, innerHeight: 800,
  localStorage: { getItem: (k) => (store.has(k) ? store.get(k) : null), setItem: (k, v) => { store.set(k, String(v)); } } };
// `groupsOf` / `groupKey` are read through `typeof`, which does not throw on an identifier a page does not
// declare: run against a page that still narrows the matrix to the model on show, this file must fail on its
// ASSERTIONS, with the values it found, not on a ReferenceError out of the harness.
const api = new Function('document', 'window', 'fetch', 'location', 'setInterval', 'getComputedStyle', 'ResizeObserver',
  `${src}\n;return { perf, state, recomputeDomains, resolveModel, ofModel, modelOf,
    groupsOf: typeof groupsOf === 'function' ? groupsOf : null, groupKey: typeof groupKey === 'function' ? groupKey : null,
    settingsKey, cfgStyle, timed, perfScales, perfTier, MIN_WINDOWS };`)(
  document, window, () => new Promise(() => {}), { search: '' }, () => 0, () => ({}), undefined);

const A = 'qwen3.8-flash-next-hibrid48', B = 'qwen3.8-27b-nvidia-nvfp4', C = 'qwen3.8-27b-dflash2';
const cfg = (model, seqs, batched, spec) => ({ model_choice: model, max_num_seqs: seqs, max_num_batched_tokens: batched,
  num_speculative_tokens: spec, max_model_len: 262144 });
const win = (c, t) => ({ t_start: t, t_end: t + 60, span_s: 60, config: c, config_source: 'effective',
  gen_tok_s: 100, prompt_tok_s: 10, running: { mean: 2 }, kv: { mean: 0.2 }, kv_pool_tokens: 1000 });
const boot = (c, t) => ({ t_start: t, t_end: t + 60, config: c, outcome: 'serving' });
/** /lb/perf `models`: every model the windows name, as the server derives it (perflog.recorded_models). */
const recorded = (windows) => [...windows.reduce((by, w) => {
  const id = w.config.model_choice, m = by.get(id) || { id, label: id, known: false, minutes: 0, first: w.t_start, last: w.t_end };
  m.minutes += 1; m.last = Math.max(m.last, w.t_end); return by.set(id, m);
}, new Map()).values()];
/** A variant key's spec|batched — the parts these fixtures vary (benchmarks.js `variantKey` carries every
 *  part but the model and the cap; the rest are equal here). */
const sb = (ks) => (ks || []).map((k) => k.split('|').slice(0, 2).join('|'));
let checks = 0;

// 1. THE RANGE OPENS ON EVERYTHING RECORDED. `all` is the picker's own fourth option (benchmarks.html
//    #p-days, labelled "everything recorded"); any day-window default drops the older minutes unasked.
{
  assert.strictEqual(api.perf.days, 'all', `the page loads the range ${JSON.stringify(api.perf.days)}, not every minute recorded`);
  const html = fs.readFileSync(path.join(__dirname, '..', 'vllm_lb', 'static', 'benchmarks.html'), 'utf8');
  const checked = [...html.matchAll(/<input[^>]*name="p-days"[^>]*>/g)].filter((m) => m[0].includes(' checked'));
  assert.strictEqual(checked.length, 1, 'exactly one range option is checked');
  assert.ok(/value="all"/.test(checked[0][0]), `the checked range option is ${checked[0][0]}`);
  checks += 3;
}

// 2. THE MATRIX IS THE INVENTORY: `groupsOf` over every model's records yields a group per configuration
//    RECORDED, not per configuration of the model on show — and a configuration that only ever booted is
//    a group of its own, so a model that never served a minute is still in the inventory.
{
  assert.ok(api.groupsOf && api.groupKey,
    'the page has no groupsOf(): the matrix is still built from the model on show alone, so every other '
    + "model's configurations are nowhere on the page");
  const windows = [win(cfg(A, 16, 8192, 4), 0), win(cfg(A, 16, 8192, 4), 60), win(cfg(A, 8, 8192, 4), 120),
    win(cfg(B, 16, 16384, 4), 180), win(cfg(C, 8, 4096, 7), 240)];
  const boots = [boot(cfg(C, 8, 4096, 4), 300), boot(cfg('never-served', 8, 8192, 4), 360)];
  api.perf.model = null; api.perf.data = { windows, boots, models: recorded(windows) }; store.clear();
  api.resolveModel();
  assert.strictEqual(api.perf.current.id, C);                  // a minute beats a later boot (resolveModel)

  const all = api.groupsOf(api.timed(windows), api.timed(boots));
  assert.strictEqual(all.length, 6);                           // A:16/8192/4, A:8/8192/4, B, C:7, C:4(boot), never-served
  const want = new Set(windows.concat(boots).map(api.settingsKey));
  assert.strictEqual(want.size, 6);
  assert.deepStrictEqual(new Set(all.map(api.groupKey)), want);
  // ...and it is strictly more than the model on show would have drawn: that is the narrowing being removed.
  // The model on show here has two configurations of its own; the payload holds six.
  const mine = api.groupsOf(api.ofModel(api.timed(windows)), api.ofModel(api.timed(boots)));
  assert.strictEqual(mine.length, 2);
  assert.ok(all.length > mine.length);
  checks += 6;
}

// 3. NOTHING IS POOLED BY WIDENING IT. The group key's first field is the model, so no group holds two
//    models' records however wide the input — the same settings under two models stay two groups.
{
  const windows = [win(cfg(A, 16, 8192, 4), 0), win(cfg(B, 16, 8192, 4), 60), win(cfg(C, 16, 8192, 4), 120)];
  api.perf.data = { windows, boots: [], models: recorded(windows) }; api.perf.model = null; store.clear();
  api.resolveModel();
  const all = api.groupsOf(api.timed(windows), []);
  assert.strictEqual(all.length, 3, 'the same settings under three models are three groups');
  for (const g of all) {
    const models = new Set(g.ws.concat(g.boots).map(api.modelOf));
    assert.strictEqual(models.size, 1, `a group holds one model only, got ${[...models]}`);
  }
  // Model first in the sort, so each model's configurations come out together.
  assert.deepStrictEqual(all.map((g) => api.modelOf(g.ws[0])), [C, B, A].sort());
  checks += 5;
}

// 4. THE SHADE IS STILL PER MODEL. A tier is computed against the same model's other configurations, never
//    against another machine's figures — so a wide matrix cannot rank a 27B dense square against a MoE one.
{
  // aggregate() derives agg_tok_s from gen_tokens over the span, so a minute at `tok` tok/s is tok*60 tokens.
  const many = (model, batched, tok, n) => Array.from({ length: n }, (_, i) =>
    ({ ...win(cfg(model, 16, batched, 4), i * 60), gen_tokens: tok * 60 }));
  const windows = [...many(A, 8192, 100, api.MIN_WINDOWS), ...many(A, 16384, 300, api.MIN_WINDOWS),
    ...many(B, 8192, 80, api.MIN_WINDOWS), ...many(B, 16384, 90, api.MIN_WINDOWS)];
  api.perf.data = { windows, boots: [], models: recorded(windows) }; api.perf.model = null; store.clear();
  api.resolveModel();
  const all = api.groupsOf(api.timed(windows), []);
  const scales = api.perfScales(all);
  assert.deepStrictEqual([...scales.keys()].filter((k) => k.endsWith('|agg_tok_s')).sort(),
    [`${B}|agg_tok_s`, `${A}|agg_tok_s`].sort());               // one scale per model, never one for both
  assert.deepStrictEqual(scales.get(`${A}|agg_tok_s`), { lo: 100, hi: 300, n: 2 });
  assert.deepStrictEqual(scales.get(`${B}|agg_tok_s`), { lo: 80, hi: 90, n: 2 });
  // B's 90 tok/s is its own best — top tier — though it is far under A's worst. Ranked across models it
  // would have been the bottom of four.
  const b90 = all.find((g) => api.modelOf(g.ws[0]) === B && g.ws[0].config.max_num_batched_tokens === 16384);
  const a100 = all.find((g) => api.modelOf(g.ws[0]) === A && g.ws[0].config.max_num_batched_tokens === 8192);
  assert.strictEqual(api.perfTier(b90, 'agg_tok_s', scales), 4);
  assert.strictEqual(api.perfTier(a100, 'agg_tok_s', scales), 1);
  checks += 5;
}

// 5. THE HATCH RANK IS STILL THE MODEL ON SHOW'S OWN. recomputeDomains keeps `perf.variants` to the model
//    on show, so widening the matrix did not make a model's first configuration at a cap hatched for
//    sharing that cap with another model's. The ramp's domain stays every cap in the log (it is the INDEX's,
//    never the filtered set's), so a square's colour does not move when the model does.
{
  const windows = [win(cfg(A, 16, 8192, 4), 0), win(cfg(A, 16, 16384, 4), 60), win(cfg(B, 16, 16384, 4), 120),
    win(cfg(B, 32, 8192, 4), 180)];
  api.state.index = null;
  api.perf.data = { windows, boots: [], models: recorded(windows) };
  api.perf.model = B; api.recomputeDomains();
  assert.deepStrictEqual(sb(api.perf.variants.get(16)), ['4|16384']);        // B's only one at cap 16: solid
  assert.strictEqual(api.cfgStyle(cfg(B, 16, 16384, 4)).variant, 'solid');
  api.perf.model = A; api.recomputeDomains();
  assert.deepStrictEqual(sb(api.perf.variants.get(16)), ['4|8192', '4|16384']);
  assert.strictEqual(api.cfgStyle(cfg(A, 16, 16384, 4)).variant, 'hatch');
  assert.deepStrictEqual(api.state.caps, [16, 32]);                      // every cap in the log, not A's alone
  checks += 5;
}

console.log(`ok — ${checks} checks`);
