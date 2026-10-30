/* THE KNOBS AN ENGINE DOES NOT TAKE, ON THE PAGE — checked where the page builds them, under node.
 *
 * The catalog view says, per model, which of the panel's knobs its engine refuses and why (`launch.cluster.refuses`,
 * catalog.py ENGINE_REFUSES). For such a model the knob is LOCKED at what the engine runs, says why, and a value
 * stored for it is listed as the reset it needs — the Apply sends null, which the LB takes, where sending the stored
 * value would be refused. Generation 135 is the case: SGLang, a 40 GB pin stored that SGLang never read.
 *
 *   node tests/sglang_knobs.node.js   (tests/test_sglang_knobs_page.py runs this when node exists)
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
    focus() {}, observe() {}, closest: () => null, getBoundingClientRect: () => ({}) });
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
    `${src}\n;return { renderKnobs, clusterChanges, fieldNote };`)(
    document, window, () => new Promise(() => {}), { search: '', href: '' }, () => 0, () => 0,
    () => ({}), undefined, window.localStorage, { userAgent: 'node' });
  return { api, byId };
}

const SGLANG = 'qwen3.8-27b-sglang-dflash2';
const FLASH = 'qwen3.8-flash-next-hibrid48';
const REFUSES = {
  kv_cache_memory: 'SGLang sizes its KV pools as a fraction of the box (engine_args.mem_fraction_static), not by a pin',
  kv_offload_gib: "the disk tier is vLLM's LMCache connector, which SGLang is launched without",
  long_prefill_token_threshold: 'SGLang has no per-request prefill cap: its prefill budget bounds every chunk',
  ple_replicate: 'the PLE placement is a patch in the vLLM image, which SGLang does not read',
};
const MODELS = [
  { id: SGLANG, label: 'Qwen3.8 27B · dense NVFP4 · SGLang DFlash2', limits: { max_model_len: 262144, max_num_seqs: [1, 48] },
    kv_bytes_per_token: 21502, calibrated: true,
    launch: { cluster: { engine: 'sglang', speculative: true, num_speculative_tokens: 4, speculative_method: 'dflash',
      kv_cache_memory: 73435823273, ple_replicate: null, refuses: REFUSES } } },
  { id: FLASH, label: 'Qwen3.8 Flash-Next', limits: { max_model_len: 262144, max_num_seqs: [1, 48] },
    kv_bytes_per_token: 16392, calibrated: true,
    launch: { cluster: { engine: 'vllm', speculative: true, num_speculative_tokens: 4, speculative_method: 'mtp',
      kv_cache_memory: 28000000000, ple_replicate: true, refuses: {} } } },
];
// /lb/status `cluster` as generation 135 serves: SGLang, budget 2048, a 40 GB pin and a disk tier stored.
const C = {
  configured: true, state: 'running', held: true, models: MODELS,
  desired: { mode: 'cluster', generation: 135, model_choice: SGLANG, max_num_seqs: 32, max_model_len: 262144,
    max_num_batched_tokens: 2048, num_speculative_tokens: null, ple_replicate: null,
    kv_cache_memory: 40000000000, kv_cache_auto: false, kv_offload_gib: 64, long_prefill_token_threshold: null },
  limits: { max_num_seqs: [1, 48], max_model_len: [1024, 262144], max_num_batched_tokens: [2048, 65536],
    num_speculative_tokens: [0, 8], kv_cache_memory: [8000000000, 96000000000], kv_offload_gib: [0, 512],
    long_prefill_token_threshold: [0, 65536], ple_replicate: [0, 1] },
  defaults: { max_num_seqs: 8, max_model_len: 262144, max_num_batched_tokens: 8192, num_speculative_tokens: 4,
    kv_cache_memory: 73435823273, kv_offload_gib: 0, ple_replicate: null },
  effective: { model_choice: SGLANG, generation: 135, max_num_seqs: 32, max_num_batched_tokens: 8192,
    num_speculative_tokens: 4, ple_replicate: null, kv_cache_memory: 73435823273, kv_offload_gib: 0,
    max_model_len: 262144, long_prefill_token_threshold: 0 },
  boxes: {},
};

{
  const { api, byId } = load();
  byId['cluster-model'] = Object.assign(byId['cluster-model'] || {}, { value: SGLANG });
  api.renderKnobs(C, true, SGLANG);
  const kv = byId['cluster-kv'];
  assert.strictEqual(kv.disabled, true, 'SGLang takes no pin: the pool is locked');
  assert.strictEqual(kv.value, '73435823273', `locked at the pool SGLang allocates, not the stored 40 GB: ${kv.value}`);
  assert.ok(!kv.innerHTML.includes('value="auto"'), 'and auto is not offered: there is nothing to size');
  const off = byId['cluster-offload'];
  assert.strictEqual(off.disabled, true, 'the disk tier is vLLM\'s: locked');
  assert.strictEqual(off.value, '0', `locked at off: ${off.value}`);
  assert.strictEqual(byId['cluster-batched'].disabled, false, 'the prefill budget reaches SGLang: it stays open');
  assert.strictEqual(byId['cluster-batched'].value, '2048');

  // What a stored value of a refused knob needs is a reset, and the Apply sends null — which the LB takes.
  const changes = api.clusterChanges(C);
  const kvc = changes.find((ch) => ch.key === 'kv_cache_memory');
  const offc = changes.find((ch) => ch.key === 'kv_offload_gib');
  assert.ok(kvc && kvc.reset && kvc.send === null, JSON.stringify(kvc));
  assert.ok(offc && offc.reset && offc.send === null, JSON.stringify(offc));
  assert.ok(!changes.some((ch) => ch.key === 'max_num_batched_tokens'), 'the budget is what is stored');

  // Each says why, in the engine's own name.
  for (const key of ['kv_cache_memory', 'kv_offload_gib']) {
    const note = api.fieldNote(C, key, SGLANG, true);
    assert.ok(note.includes('not taken by SGLang'), `${key}: ${note}`);
  }
  assert.ok(!api.fieldNote(C, 'max_num_batched_tokens', SGLANG, true).includes('not taken'));
}
{
  // The same switch with Flash-Next in the picker: vLLM takes every knob, nothing is locked by the engine.
  const { api, byId } = load();
  byId['cluster-model'] = Object.assign(byId['cluster-model'] || {}, { value: FLASH });
  const c = { ...C, desired: { ...C.desired, model_choice: FLASH, kv_cache_memory: 30000000000 } };
  api.renderKnobs(c, true, FLASH);
  assert.strictEqual(byId['cluster-kv'].disabled, false);
  assert.strictEqual(byId['cluster-offload'].disabled, false);
  assert.strictEqual(byId['cluster-offload'].value, '64');
  assert.ok(!api.fieldNote(c, 'kv_cache_memory', FLASH, true).includes('not taken'));
}
console.log('ok');
