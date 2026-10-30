/* ONE MODEL AT A TIME on /benchmarks, checked where the split is made: benchmarks.js `resolveModel`,
 * `ofModel`, `recomputeDomains`, `contextRows`.
 *
 * Two models are two machines. The page shows the minutes of ONE — the selector's — and hands nothing of
 * another model to any chart, table, legend or scale: the default is the model with the most recent minute,
 * the models offered are EXACTLY the ones the gateway recorded a minute of (/lb/perf `models`) — never the
 * catalog's, so no segment ever reads "0 min" (operator 2026-10-12, on a one-model deployment showing four
 * catalog models at 0 min: "should not be hardcoded, added on the way when new model loaded") — a
 * configuration's hatch rank is taken among its own model's configurations, and the context table reads a
 * minute only through the KV pool that minute recorded. No DOM: a stub `document` and a `fetch` that never answers.
 *
 *   node tests/model_split.node.js          (tests/test_model_split.py runs this when node exists)
 */
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');

// the page loads server-scope.js first (its `serverScope`): so does this harness
const page = fs.readFileSync(path.join(__dirname, '..', 'vllm_lb', 'static', 'benchmarks.js'), 'utf8');
const src = fs.readFileSync(path.join(__dirname, '..', 'vllm_lb', 'static', 'server-scope.js'), 'utf8') + '\n' + page;
const el = () => ({ addEventListener() {}, setAttribute() {}, append() {}, classList: { toggle() {}, add() {} },
  style: {}, hidden: false, innerHTML: '', value: '', dataset: {}, querySelectorAll: () => [], observe() {} });
const document = { getElementById: el, createElementNS: el, createElement: el, addEventListener() {},
  querySelectorAll: () => [], body: { style: {} } };
const store = new Map();
const window = { addEventListener() {}, innerWidth: 1280, innerHeight: 800,
  localStorage: { getItem: (k) => (store.has(k) ? store.get(k) : null), setItem: (k, v) => { store.set(k, String(v)); } } };
const api = new Function('document', 'window', 'fetch', 'location', 'setInterval', 'getComputedStyle', 'ResizeObserver',
  `${src}\n;return { perf, state, recomputeDomains, resolveModel, ofModel, modelOf, modelLabel, storeModel, cfgStyle,
    contextRows, settingsKey, cutMinutes, panelId, MODEL_STORE };`)(
  document, window, () => new Promise(() => {}), { search: '' }, () => 0, () => ({}), undefined);

const FLASH = 'qwen3.8-flash-next-hibrid48', DENSE = 'qwen3.8-27b-nvidia-nvfp4';
const cfg = (model, seqs, batched, spec) => ({ model_choice: model, max_num_seqs: seqs, max_num_batched_tokens: batched,
  num_speculative_tokens: spec, max_model_len: 262144 });
const win = (c, t, more) => ({ t_start: t, t_end: t + 60, span_s: 60, config: c, config_source: 'effective', ...(more || {}) });
const reset = (data) => { store.clear(); api.perf.model = null; api.perf.data = data; };
/** /lb/perf's `models` entry, as the server writes it (perflog.recorded_models + the catalog's name). */
const rec = (id, minutes, last, label) => ({ id, minutes, first: 0, last, label: label || id, known: !!label });
/** A variant key's spec|batched — the parts these fixtures vary (benchmarks.js `variantKey` carries every
 *  part but the model and the cap; the rest are equal here). */
const sb = (ks) => (ks || []).map((k) => k.split('|').slice(0, 2).join('|'));
let checks = 0;

// 1. THE SEGMENTS ARE THE RECORDED MODELS, nothing else: the server's list in its order, named as it names
//    them; the default is the model with the MOST RECENT minute — not the one with the most minutes.
{
  reset({ windows: [win(cfg(FLASH, 16, 8192, 4), 0), win(cfg(FLASH, 16, 8192, 4), 60), win(cfg(FLASH, 8, 8192, 4), 120),
    win(cfg(DENSE, 16, 8192, 4), 600)], boots: [],
  models: [rec(FLASH, 3, 180, 'Flash-Next'), rec(DENSE, 1, 660, '27B dense')] });
  const cur = api.resolveModel();
  assert.strictEqual(cur.id, DENSE); assert.strictEqual(cur.label, '27B dense'); assert.strictEqual(cur.minutes, 1);
  assert.deepStrictEqual(api.perf.models.map((m) => [m.id, m.label, m.minutes]), [[FLASH, 'Flash-Next', 3], [DENSE, '27B dense', 1]]);
  assert.deepStrictEqual(api.ofModel(api.perf.data.windows).map(api.modelOf), [DENSE]);
  checks += 5;
}
// 2. The viewer's pick is remembered and wins over the default; a pick the gateway has no minute of (a
//    model this deployment never served, one from another gateway's page, a stale stored id) is no segment
//    and so no page — the default stands.
{
  api.storeModel(FLASH); api.perf.model = null;
  assert.strictEqual(store.get(api.MODEL_STORE), FLASH);
  assert.strictEqual(api.resolveModel().id, FLASH);
  assert.strictEqual(api.ofModel(api.perf.data.windows).length, 3);
  api.perf.model = 'never-ran';
  assert.strictEqual(api.resolveModel().id, DENSE);
  assert.ok(!api.perf.models.some((m) => m.id === 'never-ran'));
  checks += 5;
}
// 3. NOTHING BUT THE LOG: the page asks no catalog for its segments (no /lb/status read on this page), a
//    model the catalog does not know is offered under its raw id, a minute that named no model is offered
//    as "model not recorded", a model that only BOOTED is no segment, and no minute at all is no model —
//    never a stand-in. A storage that throws costs nothing. Every segment has a minute.
{
  assert.ok(!/fetch\(\s*['"`]\/lb\/status/.test(page), 'benchmarks.js still reads /lb/status: the catalog is not the list');
  assert.ok(!/perf\.catalog/.test(src), 'benchmarks.js still keeps a catalog to enumerate models from');
  reset({ windows: [win(cfg('someones-own-finetune', 16, 8192, 4), 0), win(cfg(FLASH, 16, 8192, 4), 60)],
    boots: [{ t_start: 900, t_end: 960, config: cfg(DENSE, 16, 8192, 4), outcome: 'serving' }],
    models: [rec('someones-own-finetune', 1, 60), rec(FLASH, 1, 120, 'Flash-Next')] });
  assert.strictEqual(api.resolveModel().id, FLASH);              // a minute beats a later boot
  assert.deepStrictEqual(api.perf.models.map((m) => [m.id, m.label, m.minutes]),
    [['someones-own-finetune', 'someones-own-finetune', 1], [FLASH, 'Flash-Next', 1]]);
  assert.strictEqual(api.modelLabel('someones-own-finetune'), 'someones-own-finetune');
  assert.strictEqual(api.modelLabel(FLASH), 'Flash-Next');
  reset({ windows: [win({ max_num_seqs: 8 }, 0)], boots: [], models: [rec('', 1, 60)] });
  api.resolveModel();
  assert.deepStrictEqual(api.perf.models.map((m) => [m.id, m.label]), [['', 'model not recorded']]);
  reset({ windows: [], boots: [{ t_start: 0, t_end: 60, config: cfg(FLASH, 16, 8192, 4) }], models: [] });
  assert.strictEqual(api.resolveModel(), null);
  assert.deepStrictEqual(api.perf.models, []);
  reset({ windows: [win(cfg(DENSE, 16, 8192, 4), 0)], boots: [], models: [rec(DENSE, 1, 60)] });
  const get = window.localStorage.getItem;
  window.localStorage.getItem = () => { throw new Error('denied'); };
  api.perf.model = null;
  assert.strictEqual(api.resolveModel().id, DENSE);
  window.localStorage.getItem = get;
  reset({ windows: [], boots: [], models: [] });
  assert.strictEqual(api.resolveModel(), null);
  assert.strictEqual(api.modelLabel(''), 'model not recorded');
  checks += 14;
}
// 4. THE SAME settings under two models are two configurations, and the hatch rank is each model's own:
//    Flash-Next ran 16 seqs at 8,192 and 16,384 (solid, hatched); the dense model only at 16,384 — on ITS
//    page that one is solid, not hatched for sharing a cap with a configuration of another model.
{
  const data = { windows: [win(cfg(FLASH, 16, 8192, 4), 0), win(cfg(FLASH, 16, 16384, 4), 60), win(cfg(DENSE, 16, 16384, 4), 120)], boots: [],
    models: [rec(FLASH, 2, 120), rec(DENSE, 1, 180)] };
  reset(data);
  assert.notStrictEqual(api.settingsKey(data.windows[1]), api.settingsKey(data.windows[2]));
  api.perf.model = FLASH; api.recomputeDomains();
  assert.deepStrictEqual(sb(api.perf.variants.get(16)), ['4|8192', '4|16384']);
  assert.strictEqual(api.cfgStyle(cfg(FLASH, 16, 16384, 4)).variant, 'hatch');
  api.perf.model = DENSE; api.recomputeDomains();
  assert.deepStrictEqual(sb(api.perf.variants.get(16)), ['4|16384']);
  assert.strictEqual(api.cfgStyle(cfg(DENSE, 16, 16384, 4)).variant, 'solid');
  checks += 5;
}
// 5. A minute's context is read through THE POOL IT RECORDED (`kv_pool_tokens`) and nothing else: a minute
//    without one is not banded and is counted out loud (`unpooled`), whatever its model — no model's pool is
//    assumed for another's minutes, or for its own older ones.
{
  const load = { kv: { mean: 0.5 }, running: { mean: 4 }, gen_tokens: 6000, prompt_computed_tokens: 600 };
  const one = (model, more) => api.contextRows([win(cfg(model, 16, 8192, 4), 0, { ...load, ...(more || {}) })]);
  for (const model of [FLASH, DENSE]) {
    const bare = one(model);
    assert.deepStrictEqual([bare.rows, bare.unpooled], [[], 1], model);
    const own = one(model, { kv_pool_tokens: 400000 });
    assert.deepStrictEqual([own.rows.length, own.rows[0].ctx, own.unpooled], [1, 50000, 0], model);
  }
  checks += 4;
}
// 6. "The N oldest not shown" is the model on show's own count (perflog `truncated_by_model`); a model
//    whose every minute was cut is still offered — with the minutes the log RECORDED of it (the server's
//    count, not the handed-over ones), never 0; a gateway that does not say whose the cut minutes were
//    leaves the whole log's count rather than none.
{
  reset({ windows: [win(cfg(FLASH, 16, 8192, 4), 0)], boots: [], truncated: 7, truncated_by_model: { [FLASH]: 5, [DENSE]: 2 },
    models: [rec(DENSE, 2, -100), rec(FLASH, 6, 60)] });
  assert.strictEqual(api.resolveModel().id, FLASH); assert.strictEqual(api.cutMinutes(), 5);
  api.perf.model = DENSE;
  assert.deepStrictEqual([api.resolveModel().id, api.perf.current.minutes, api.cutMinutes()], [DENSE, 2, 2]);
  reset({ windows: [win(cfg(FLASH, 16, 8192, 4), 0)], boots: [], truncated: 0, truncated_by_model: {}, models: [rec(FLASH, 1, 60)] });
  api.resolveModel(); assert.strictEqual(api.cutMinutes(), 0);
  reset({ windows: [win(cfg(FLASH, 16, 8192, 4), 0)], boots: [], truncated: 7, models: [rec(FLASH, 8, 60)] });
  api.resolveModel(); assert.strictEqual(api.cutMinutes(), 7);
  checks += 5;
}
// 7. The archive: a comparison panel is one workload of ONE model, and so is its id — the "full comparison"
//    link of a run must land on its own model's panel.
{
  const a = api.panelId({ workload: 'realdecode', model: 'Qwen/Qwen3.8-Flash-Next' });
  const b = api.panelId({ workload: 'realdecode', model: 'Qwen/Qwen3.8-27B' });
  assert.strictEqual(a, 'cmp-realdecode--qwen-qwen3-8-flash-next'); assert.notStrictEqual(a, b);
  assert.strictEqual(api.panelId({ workload: 'realdecode' }), 'cmp-realdecode');
  checks += 3;
}
console.log(`ok — ${checks} checks`);
