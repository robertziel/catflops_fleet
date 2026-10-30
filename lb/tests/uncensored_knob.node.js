/* WHICH CHECKPOINT, ON THE PAGE — one row per checkpoint in the model picker, checked where the page builds it.
 *
 * Operator, 2026-10-07: the uncensored twin is its OWN row in the dropdown, right under the original it twins,
 * ` · uncensored` as its distinguishing text — nine rows for a six-entry catalog with four twins. Picking the row
 * IS picking the checkpoint: the row sets the (`model_choice`, `uncensored`) pair the switch stores, so there is no
 * second control. A row whose checkpoint is not on every GB10 box stays visible, disabled, with the reason — exactly as
 * a missing original does — and the ✓ (the select's value) is the row matching the stored pair.
 *
 *   node tests/uncensored_knob.node.js   (tests/test_uncensored_page.py runs this when node exists)
 */
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const src = fs.readFileSync(path.join(__dirname, '..', 'vllm_lb', 'static', 'gateway.js'), 'utf8');

function load() {
  const byId = {};
  const el = () => ({ addEventListener() {}, removeAttribute() {}, setAttribute() {}, append() {},
    classList: { toggle() {}, add() {}, remove() {} }, style: {}, hidden: false, innerHTML: '',
    textContent: '', value: '', dataset: {}, querySelectorAll: () => [], querySelector: () => null,
    focus() {}, observe() {}, closest: () => null, getBoundingClientRect: () => ({}), options: [] });
  const document = {
    getElementById: (id) => (byId[id] || (byId[id] = el())),
    createElementNS: el, createElement: el, addEventListener() {},
    querySelectorAll: () => [], querySelector: () => null, activeElement: null,
    body: { style: {}, classList: { toggle() {} } },
    documentElement: { style: {}, classList: { toggle() {} } } };
  const window = { addEventListener() {}, innerWidth: 1280, innerHeight: 800, matchMedia: () => ({
    matches: false, addEventListener() {} }), localStorage: { getItem: () => null, setItem() {} } };
  const api = new Function('document', 'window', 'fetch', 'location', 'setInterval', 'setTimeout',
    'getComputedStyle', 'ResizeObserver', 'localStorage', 'navigator',
    `${src}\n;return { renderKnobs, clusterChanges, fieldNote, storedOf, defaultOf, fmtSetting, renderModelChoice,
      shownOf, pickedRow, CLUSTER_FIELDS, KNOBS };`)(
    document, window, () => new Promise(() => {}), { search: '', href: '' }, () => 0, () => 0,
    () => ({}), undefined, window.localStorage, { userAgent: 'node' });
  return { api, byId };
}

const FLASH = 'qwen3.8-flash-next-hibrid48';
const FLASH_SGLANG = 'qwen3.8-flash-next-sglang';
const DENSE = 'qwen3.8-27b-nvidia-nvfp4';
const DFLASH = 'qwen3.8-27b-dflash2';
const SGLANG = 'qwen3.8-27b-sglang-dflash2';
const NOTE = 'ARCH-EXACT, NOT CALIBRATION-VERIFIED (2026-10-07): the memory block is the original\'s.';
const FLASH_TWIN = { path: 'qwen38-flash-next-cluster/Qwen3.8-Flash-Next-hibrid48_uncensored', default: true, calibration: NOTE };
const DENSE_TWIN = { path: 'Qwen3.8-27B-NVFP4_uncensored', default: true, calibration: NOTE };
const PROFILE = { engine: 'vllm', speculative: true, num_speculative_tokens: 4, speculative_method: 'mtp',
  kv_cache_memory: 28000000000, ple_replicate: true, tuned: null, refuses: {} };
const LIMITS = { max_model_len: 262144, max_num_seqs: [1, 48] };
//: The five cluster entries as /lb/status publishes them (the classic-only seed is not in the cluster picker).
const MODELS = [
  { id: FLASH, label: 'Qwen3.8 Flash-Next hibrid48 · MoE NVFP4', size_gib: 98.1, limits: LIMITS, calibrated: true,
    uncensored: FLASH_TWIN, launch: { cluster: PROFILE } },
  { id: FLASH_SGLANG, label: 'Qwen3.8 Flash-Next hibrid48 · MoE NVFP4 · SGLang NEXTN', size_gib: 98.1, limits: LIMITS,
    calibrated: true, uncensored: FLASH_TWIN, launch: { cluster: { ...PROFILE, engine: 'sglang' } } },
  { id: DENSE, label: 'Qwen3.8 27B · dense NVFP4', size_gib: 20.4, limits: LIMITS, calibrated: true,
    uncensored: DENSE_TWIN, launch: { cluster: PROFILE } },
  { id: DFLASH, label: 'Qwen3.8 27B · dense NVFP4, BF16 head · DFlash2', size_gib: 22.1, limits: LIMITS,
    calibrated: true, uncensored: null, launch: { cluster: { ...PROFILE, ple_replicate: null, speculative_method: 'dflash' } } },
  { id: SGLANG, label: 'Qwen3.8 27B · dense NVFP4 · SGLang DFlash2', size_gib: 20.4, limits: LIMITS, calibrated: true,
    uncensored: DENSE_TWIN, launch: { cluster: { ...PROFILE, engine: 'sglang', speculative_method: 'dflash' } } },
];
const CLUSTER_LIMITS = { max_num_seqs: [1, 48], max_model_len: [1024, 262144], max_num_batched_tokens: [2048, 65536],
  num_speculative_tokens: [0, 8], kv_cache_memory: [8000000000, 96000000000], kv_offload_gib: [0, 512],
  long_prefill_token_threshold: [0, 65536], ple_replicate: [0, 1], uncensored: [0, 1] };
//: Generation 169 as it is stored right now: the SGLang DFlash2 entry, from its uncensored twin.
const C = {
  configured: true, state: 'starting', held: true, models: MODELS, limits: CLUSTER_LIMITS,
  desired: { mode: 'cluster', generation: 169, model_choice: SGLANG, max_num_seqs: 16, max_model_len: 262144,
    max_num_batched_tokens: null, num_speculative_tokens: null, ple_replicate: null, kv_cache_memory: null,
    kv_cache_auto: false, kv_offload_gib: null, long_prefill_token_threshold: null, uncensored: true },
  defaults: { max_num_seqs: 8, max_model_len: 262144, max_num_batched_tokens: 8192, num_speculative_tokens: 4,
    kv_cache_memory: 28000000000, kv_offload_gib: 0, ple_replicate: null, uncensored: true },
  effective: null,
  // The Flash-Next twin is still downloading: box-b says its dir is not there.
  boxes: { 'box-b': { availability: { [`${FLASH}@uncensored`]: { present: false }, [`${FLASH_SGLANG}@uncensored`]: { present: false } } } },
};
const rowsOf = (el) => el.innerHTML.split('</option>').filter(Boolean)
  .map((r) => ({ value: r.match(/value="([^"]+)"/)[1], disabled: r.includes(' disabled'), text: r.replace(/^.*?>/, '') }));

{ // NINE ROWS, each checkpoint its own, the original first and its twin right under it
  const { api, byId } = load();
  api.renderModelChoice(C, {}, true);
  const rows = rowsOf(byId['cluster-model']);
  // Since 2026-10-09 a row names its engine where the label does not (model_picker.node.js); this view is an LB's
  // from before `cluster.setups`, so it carries no launch facts and no rate.
  // Since 2026-10-10 a row missing on a box NAMES the box (model_picker_ui.node.js): "missing on box-b".
  assert.deepStrictEqual(rows.map((r) => r.text), [
    'Qwen3.8 Flash-Next hibrid48 · MoE NVFP4 · vLLM · 98.1 GiB',
    'Qwen3.8 Flash-Next hibrid48 · MoE NVFP4 · vLLM · 98.1 GiB · uncensored · missing on box-b',
    'Qwen3.8 Flash-Next hibrid48 · MoE NVFP4 · SGLang NEXTN · 98.1 GiB',
    'Qwen3.8 Flash-Next hibrid48 · MoE NVFP4 · SGLang NEXTN · 98.1 GiB · uncensored · missing on box-b',
    'Qwen3.8 27B · dense NVFP4 · vLLM · 20.4 GiB',
    'Qwen3.8 27B · dense NVFP4 · vLLM · 20.4 GiB · uncensored',
    'Qwen3.8 27B · dense NVFP4, BF16 head · DFlash2 · vLLM · 22.1 GiB',
    'Qwen3.8 27B · dense NVFP4 · SGLang DFlash2 · 20.4 GiB',
    'Qwen3.8 27B · dense NVFP4 · SGLang DFlash2 · 20.4 GiB · uncensored',
  ]);
  assert.deepStrictEqual(rows.map((r) => r.value), [FLASH, `${FLASH}@uncensored`, FLASH_SGLANG, `${FLASH_SGLANG}@uncensored`,
    DENSE, `${DENSE}@uncensored`, DFLASH, SGLANG, `${SGLANG}@uncensored`]);
  assert.deepStrictEqual(rows.map((r) => r.disabled), [false, true, false, true, false, false, false, false, false],
    'a twin missing from a box is greyed with the reason, never dropped; its original stays startable');
  // The ✓: the row matching the stored (model_choice, uncensored) pair — generation 169's twin.
  assert.strictEqual(byId['cluster-model'].value, `${SGLANG}@uncensored`);
  assert.deepStrictEqual(api.pickedRow(), { model: SGLANG, uncensored: true });
  assert.strictEqual(api.shownOf('model_choice'), SGLANG, 'everything that reads the picker as a MODEL still gets the id');
  // ONE control: no Checkpoint select, no `uncensored` knob in the knob list.
  assert.ok(!api.CLUSTER_FIELDS.some((f) => f.id === 'cluster-uncensored'), JSON.stringify(api.CLUSTER_FIELDS));
  assert.ok(!api.KNOBS.includes('uncensored'), JSON.stringify(api.KNOBS));
  assert.ok(!src.includes("$('cluster-uncensored')"), 'the separate control is gone from the page script');
  assert.ok(!fs.readFileSync(path.join(__dirname, '..', 'vllm_lb', 'static', 'index.html'), 'utf8').includes('cluster-uncensored'));
  // Nothing changed while the ✓ row is the one on screen.
  assert.ok(!api.clusterChanges(C).some((ch) => ch.key === 'uncensored' || ch.key === 'model_choice'), JSON.stringify(api.clusterChanges(C)));
}

{ // picking the ORIGINAL row of the stored model: one listed change, the checkpoint, sent as false
  const { api, byId } = load();
  api.renderModelChoice(C, {}, true);
  byId['cluster-model'].value = SGLANG;
  const changes = api.clusterChanges(C);
  assert.ok(!changes.some((ch) => ch.key === 'model_choice'), 'the model is the same');
  const ch = changes.find((x) => x.key === 'uncensored');
  assert.ok(ch && ch.send === false && ch.from === 1 && ch.to === 0 && ch.name === 'checkpoint', JSON.stringify(ch));
  assert.strictEqual(api.fmtSetting(C, 'uncensored', ch.to), 'original');
}

{ // picking ANOTHER model's twin row: the model changes and the checkpoint travels with it, as the row says
  const { api, byId } = load();
  api.renderModelChoice(C, {}, true);
  byId['cluster-model'].value = `${DENSE}@uncensored`;
  const changes = api.clusterChanges(C);
  assert.ok(changes.some((ch) => ch.key === 'model_choice' && ch.to === DENSE), JSON.stringify(changes));
  assert.deepStrictEqual(api.pickedRow(), { model: DENSE, uncensored: true });
  assert.strictEqual(api.shownOf('model_choice'), DENSE);
  // …and another model's ORIGINAL row, where that model's default is its twin: listed, sent as false.
  byId['cluster-model'].value = DFLASH;
  assert.deepStrictEqual(api.pickedRow(), { model: DFLASH, uncensored: false });
  const toDflash = api.clusterChanges(C);
  assert.ok(toDflash.some((ch) => ch.key === 'model_choice' && ch.to === DFLASH), JSON.stringify(toDflash));
  // A model with no twin: its one row IS what its next launch uses, so nothing is listed — the Start body still
  // carries the row's checkpoint (`pickedRow`), it is simply not a change.
  assert.ok(!toDflash.some((x) => x.key === 'uncensored'), JSON.stringify(toDflash));
  byId['cluster-model'].value = DENSE;                      // another model's ORIGINAL row, whose default is its twin
  const ck = api.clusterChanges(C).find((x) => x.key === 'uncensored');
  assert.ok(ck && ck.send === false && ck.to === 0 && ck.from === 1, JSON.stringify(ck));
}

{ // a switch from before the knob (uncensored null) marks the model's DEFAULT row
  const { api, byId } = load();
  const c = { ...C, desired: { ...C.desired, model_choice: FLASH, uncensored: null } };
  api.renderModelChoice(c, {}, true);
  assert.strictEqual(byId['cluster-model'].value, `${FLASH}@uncensored`, 'default: true preselects the twin row');
  const c2 = { ...C, desired: { ...C.desired, model_choice: DFLASH, uncensored: null } };
  const { api: api2, byId: byId2 } = load();
  api2.renderModelChoice(c2, {}, true);
  assert.strictEqual(byId2['cluster-model'].value, DFLASH);
}

{ // an LB that does not publish the knob: one row per model, nothing about checkpoints is sent
  const { api, byId } = load();
  const { uncensored, ...limits } = CLUSTER_LIMITS;
  const c = { ...C, limits, desired: { ...C.desired, uncensored: undefined } };
  api.renderModelChoice(c, {}, true);
  const rows = rowsOf(byId['cluster-model']);
  assert.deepStrictEqual(rows.map((r) => r.value), [FLASH, FLASH_SGLANG, DENSE, DFLASH, SGLANG]);
  assert.strictEqual(byId['cluster-model'].value, SGLANG);
  byId['cluster-model'].value = DENSE;
  assert.ok(!api.clusterChanges(c).some((ch) => ch.key === 'uncensored'));
}

console.log('ok uncensored_knob');
