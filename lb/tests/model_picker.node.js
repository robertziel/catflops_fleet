/* WHAT A MODEL PICKER ROW SAYS — checked where the page builds it, against the SHIPPED catalog.
 *
 * The operator (2026-10-09): "make select more detailed, include tok/s targets if set". A row used to be the
 * label, the size and "uncensored". It now also says what a Start of that row launches when nothing else is
 * chosen — `cluster.setups[row]`, the same resolve as `cluster.defaults` (cluster_loop._setups) — in the cluster
 * card's own words ("32 seqs · 256k ctx · DFlash2 4"), and the rate to aim for there: `setup_targets[row]`, the
 * best tenth of the minutes recorded at exactly that setup of exactly that checkpoint (perflog.throughput_target),
 * marked "other setup" when only another setup of it was recorded. No minute, no rate.
 *
 *   node tests/model_picker.node.js      (tests/test_model_picker.py runs this when node exists)
 */
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const STATIC = path.join(__dirname, '..', 'vllm_lb', 'static');
const src = fs.readFileSync(path.join(STATIC, 'gateway.js'), 'utf8');
const raw = { models: require('./library_rows.js')() };     // the shipped model library, as rows

const el = () => ({ addEventListener() {}, removeAttribute() {}, setAttribute() {}, append() {},
  classList: { toggle() {}, add() {}, remove() {} }, style: {}, hidden: false, innerHTML: '',
  textContent: '', value: '', dataset: {}, querySelectorAll: () => [], querySelector: () => null,
  focus() {}, observe() {}, closest: () => null, getBoundingClientRect: () => ({}) });
// The picker element persists, so what renderModelChoice writes into it can be read back.
const picker = el();
const document = { getElementById: (id) => (id === 'cluster-model' ? picker : el()), createElementNS: el,
  createElement: el, addEventListener() {}, querySelectorAll: () => [], querySelector: () => null,
  activeElement: null, body: { style: {}, classList: { toggle() {} } },
  documentElement: { style: {}, classList: { toggle() {} } } };
const window = { addEventListener() {}, innerWidth: 1280, innerHeight: 800, matchMedia: () => ({
  matches: false, addEventListener() {} }), localStorage: { getItem: () => null, setItem() {} } };
const api = new Function('document', 'window', 'fetch', 'location', 'setInterval', 'setTimeout',
  'getComputedStyle', 'ResizeObserver', 'localStorage', 'navigator',
  `${src}\n;return { pickerRowText: typeof pickerRowText === 'function' ? pickerRowText : null, renderModelChoice };`)(
  document, window, () => new Promise(() => {}), { search: '', href: '' }, () => 0, () => 0,
  () => ({}), undefined, window.localStorage, { userAgent: 'node' });
assert.ok(api.pickerRowText, 'gateway.js has no pickerRowText');

/* The fields of `ModelSpec.view()` a row reads, for one row of the model library. */
const viewOf = (m) => {
  const lc = m.launch.cluster || {};
  return { id: m.id, label: m.label, size_gib: typeof m.size_gib === 'number' ? m.size_gib : 20.4,
    uncensored: m.uncensored || null,
    launch: { cluster: { engine: lc.engine || 'vllm', speculative_method: (lc.speculative_config || {}).method || null,
      // What `LaunchProfile.view()` adds for an engine that chooses its prompt chunk and drafter depth itself.
      ...(lc.engine === 'tensorfold' ? { refuses: { max_num_batched_tokens: 'its own chunk', num_speculative_tokens: 'its own chain' },
        engine_drafts: !!(lc.engine_args || {}).drafts } : {}) } } };
};
const CLUSTER = raw.models.filter((m) => m.launch.cluster).map(viewOf);
assert.ok(CLUSTER.length >= 3, `the shipped catalog ships ${CLUSTER.length} cluster models`);
const WORD = { mtp: 'MTP', dflash: 'DFlash2', nextn: 'NEXTN', dspark: 'DSpark' };
const setup = { max_num_seqs: 32, max_model_len: 262144, max_num_batched_tokens: 2048, num_speculative_tokens: 4,
  kv_cache_memory: 73435823273 };
let checks = 0;

// 1. EVERY MODEL'S ROW names the checkpoint and what a Start of it launches, in the card's words.
for (const m of CLUSTER) {
  const t = api.pickerRowText(m, false, setup, null, false);
  const method = m.launch.cluster.speculative_method;
  // A profile whose engine drafts by its own rule (TensorFold: no speculative_config the LB shapes) has no method
  // word to print beside a depth; every other part is the same.
  const own = m.launch.cluster.refuses || {};
  const drafter = method ? [`${WORD[method] || method} 4`] : own.num_speculative_tokens ? ['MTP · engine depth'] : [];
  const prefill = own.max_num_batched_tokens ? [] : ['prefill 2,048'];
  for (const part of [m.label, '32 seqs', '256k ctx', ...prefill, ...drafter, 'KV 73.4 GB']) {
    assert.ok(t.includes(part), `${m.id}: "${part}" missing from "${t}"`);
  }
  assert.ok(!t.includes('tok/s'), `${m.id}: no target was given, yet the row prints a rate: ${t}`);
  // A budget the engine never takes is never drawn as the setup's (TensorFold chunks its prompts itself).
  if (own.max_num_batched_tokens) assert.ok(!t.includes('prefill'), `${m.id}: prints a budget it never runs: ${t}`);
  // The engine is named once: added where the label does not say it, never repeated where it does.
  const engine = { sglang: 'SGLang', tensorfold: 'TensorFold' }[m.launch.cluster.engine] || 'vLLM';
  assert.strictEqual(t.split(engine).length - 1, 1, `${m.id}: "${engine}" should appear once in "${t}"`);
  checks += 1;
}

// 2. THE TWIN ROW says so; a setup with no drafter says that.
const m0 = CLUSTER[0];
assert.ok(api.pickerRowText(m0, true, setup, null, false).includes(' · uncensored'));
assert.ok(api.pickerRowText(m0, false, { ...setup, num_speculative_tokens: 0 }, null, false).includes('no drafter'));
checks += 2;

// 3. THE RATE: at this setup a target, at another setup of the checkpoint marked as such, none without a minute.
assert.ok(api.pickerRowText(m0, false, setup, { gen_tok_s: 159.5, basis: 'setup', windows: 1334 }, false)
  .endsWith('· target 160 tok/s'));
assert.ok(api.pickerRowText(m0, false, setup, { gen_tok_s: 95, basis: 'model', windows: 12 }, false)
  .endsWith('· 95.0 tok/s other setup'));
assert.ok(!api.pickerRowText(m0, false, setup, { gen_tok_s: null, basis: 'setup' }, false).includes('tok/s'));
checks += 3;

// 4. A row no GB10 box fully holds still says why, last — naming the boxes when the caller does (the page does).
assert.ok(api.pickerRowText(m0, false, setup, null, true).endsWith('· not on every box'));
assert.ok(api.pickerRowText(m0, false, setup, null, ['box-b']).endsWith('· missing on box-b'));
assert.ok(api.pickerRowText(m0, false, setup, null, ['box-a', 'box-b']).endsWith('· missing on box-a, box-b'));
assert.ok(!api.pickerRowText(m0, false, setup, null, []).includes('missing'), 'present everywhere: nothing to say');
checks += 4;

// 5. WIRED: renderModelChoice writes one option per checkpoint, each with its own setup and its own target.
const twin = CLUSTER.find((m) => m.uncensored);
assert.ok(twin, 'the shipped catalog names a twin');
const setups = {};
for (const m of CLUSTER) {
  setups[m.id] = { ...setup, model_choice: m.id, uncensored: false };
  if (m.uncensored) setups[`${m.id}@uncensored`] = { ...setup, model_choice: m.id, uncensored: true };
}
const c = { models: CLUSTER, limits: { uncensored: [false, true] }, boxes: {}, setups,
  desired: { model_choice: twin.id, uncensored: true } };
const s = { setup_targets: { [`${twin.id}@uncensored`]: { gen_tok_s: 159.5, basis: 'setup', windows: 1334 } } };
api.renderModelChoice(c, s, true);
const options = [...picker.innerHTML.matchAll(/<option value="([^"]*)"([^>]*)>([^<]*)<\/option>/g)];
assert.strictEqual(options.length, Object.keys(setups).length, picker.innerHTML);
const byValue = Object.fromEntries(options.map((o) => [o[1], { attrs: o[2], text: o[3] }]));
const hit = byValue[`${twin.id}@uncensored`];
assert.ok(hit.text.includes('target 160 tok/s') && hit.text.includes('uncensored'), hit.text);
assert.ok(hit.attrs.includes('title="1,334 min · this setup"'), hit.attrs);
assert.ok(!byValue[twin.id].text.includes('tok/s'), 'the original row borrowed the twin\'s rate');
assert.strictEqual(picker.value, `${twin.id}@uncensored`, 'the stored pair is the picked row');
checks += 5;

// 6. WHAT THIS DEPLOYMENT OFFERS (fleet `choices` on the pair, published as `cluster.offers`): the operator's rule
// (2026-10-26, "remove original censored from our vllm setup") — rows not offered are not drawn at all; no list =
// every catalog row, as before.
picker.dataset.choices = '';
api.renderModelChoice({ ...c, offers: [`${twin.id}@uncensored`] }, s, true);
const offered = [...picker.innerHTML.matchAll(/<option value="([^"]*)"/g)].map((o) => o[1]);
assert.deepStrictEqual(offered, [`${twin.id}@uncensored`], picker.innerHTML);
picker.dataset.choices = '';
api.renderModelChoice({ ...c, offers: null }, s, true);
assert.strictEqual([...picker.innerHTML.matchAll(/<option /g)].length, Object.keys(setups).length);
checks += 2;

console.log(`ok — ${checks} checks`);
