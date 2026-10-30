/* THE KV POOL CHOOSER NEVER OFFERS A PIN THE PROFILE CANNOT TAKE — checked where the page builds it, under node.
 *
 * GENERATION 187, 2026-10-10 20:55Z: the chooser offered 24 GB on the SGLang Flash-Next row, the switch accepted it, and
 * the launch refused it on every tick for six minutes ("SGLang KV pin 24,000,000,000 B needs mem_fraction_static 0.885,
 * above this profile's safe 0.85 ceiling (maximum 19,918,363,626 B)"). The LB now publishes the profile's bound
 * (`launch.cluster.kv_cache_memory_max`) and refuses a pin past it at the switch; the page reads the same figure:
 *
 *   - every pin past the bound is listed DISABLED, naming the maximum as a figure ("… · max 19.9 GB");
 *   - the field's note says "max 19.9 GB";
 *   - a STORED pin past it (an older LB's file, or one chosen for another model) is still shown — and
 *     `selectionProblem`/`startProblem` refuse it by name before the Start is held, as the LB would (409);
 *   - a row with no bound of its own (vLLM) is untouched: nothing greyed, no note, no refusal.
 *
 *   node tests/kv_ceiling.node.js   (tests/test_kv_ceiling_page.py runs this when node exists)
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
    `${src}\n;return { renderKnobs, clusterChanges, fieldNote, selectionProblem, startProblem, kvMaxOf };`)(
    document, window, () => new Promise(() => {}), { search: '', href: '' }, () => 0, () => 0,
    () => ({}), undefined, window.localStorage, { userAgent: 'node' });
  return { api, byId };
}

const FLASH_SGLANG = 'qwen3.8-flash-next-sglang';
const FLASH = 'qwen3.8-flash-next-hibrid48';
const MAX = 19918363626;                                  // the profile's bound, as the LB publishes it
const SGLANG_REFUSES = {
  kv_offload_gib: "the disk tier is vLLM's LMCache connector, which SGLang is launched without",
  long_prefill_token_threshold: 'SGLang has no per-request prefill cap: its prefill budget bounds every chunk',
  ple_replicate: 'the PLE placement is a patch in the vLLM image, which SGLang does not read',
};
const LIMITS = { max_model_len: 262144, max_num_seqs: [1, 48] };
const MODELS = [
  { id: FLASH_SGLANG, label: 'Qwen3.8 Flash-Next hibrid48 · MoE NVFP4 · SGLang NEXTN', limits: LIMITS,
    kv_bytes_per_token: 13058, calibrated: true,
    launch: { cluster: { engine: 'sglang', speculative: true, num_speculative_tokens: 4, speculative_method: 'nextn',
      kv_cache_memory: 18000000000, kv_cache_memory_max: MAX, ple_replicate: false, refuses: SGLANG_REFUSES,
      tuned: { max_num_seqs: 16, max_model_len: 262144, max_num_batched_tokens: 4096, num_speculative_tokens: 4, source: 'MEASURED' } } } },
  { id: FLASH, label: 'Qwen3.8 Flash-Next hibrid48 · MoE NVFP4', limits: LIMITS,
    kv_bytes_per_token: 16392, calibrated: true,
    launch: { cluster: { engine: 'vllm', speculative: true, num_speculative_tokens: 4, speculative_method: 'mtp',
      kv_cache_memory: 28000000000, kv_cache_memory_max: null, ple_replicate: true, refuses: {} } } },
];
// /lb/status `cluster` after generation 187's Stop: the SGLang row stored with the 24 GB pin the switch then accepted.
const C = {
  configured: true, state: 'off', held: false, models: MODELS,
  desired: { mode: 'classic', generation: 187, model_choice: FLASH_SGLANG, max_num_seqs: 16, max_model_len: 262144,
    max_num_batched_tokens: null, num_speculative_tokens: null, ple_replicate: null,
    kv_cache_memory: 24000000000, kv_cache_auto: false, kv_offload_gib: null, long_prefill_token_threshold: null,
    uncensored: true },
  limits: { max_num_seqs: [1, 48], max_model_len: [1024, 262144], max_num_batched_tokens: [2048, 65536],
    num_speculative_tokens: [0, 8], kv_cache_memory: [8000000000, 96000000000], kv_offload_gib: [0, 512],
    long_prefill_token_threshold: [0, 65536], ple_replicate: [0, 1], uncensored: [0, 1] },
  defaults: { max_num_seqs: 16, max_model_len: 262144, max_num_batched_tokens: 4096, num_speculative_tokens: 4,
    kv_cache_memory: 18000000000, kv_offload_gib: 0, ple_replicate: false },
  effective: null,
  boxes: {},
};

/** The <option>s of a select as the page wrote them: [{value, disabled, text}]. */
function options(sel) {
  return [...sel.innerHTML.matchAll(/<option value="([^"]*)"( disabled)?>([^<]*)<\/option>/g)]
    .map((m) => ({ value: m[1], disabled: !!m[2], text: m[3] }));
}

function controls(byId, model) {
  byId['cluster-model'] = Object.assign(byId['cluster-model'] || {}, { value: model });
  byId['cluster-seqs'] = Object.assign(byId['cluster-seqs'] || {}, { value: '16' });
  byId['cluster-context'] = Object.assign(byId['cluster-context'] || {}, { value: '262144' });
  byId['cluster-batched'] = Object.assign(byId['cluster-batched'] || {}, { value: '4096' });
  byId['cluster-spec'] = Object.assign(byId['cluster-spec'] || {}, { value: '4' });
  byId['cluster-ple'] = Object.assign(byId['cluster-ple'] || {}, { value: '2' });
  byId['cluster-offload'] = Object.assign(byId['cluster-offload'] || {}, { value: '0' });
}

// 1. The SGLang row: the pins past 19.9 GB are listed disabled and name the maximum; the stored 24 GB is shown.
{
  const { api, byId } = load();
  controls(byId, FLASH_SGLANG);
  assert.strictEqual(api.kvMaxOf(C, FLASH_SGLANG), MAX);
  api.renderKnobs(C, true, FLASH_SGLANG);
  const kv = byId['cluster-kv'];
  assert.strictEqual(kv.disabled, false, 'SGLang with a calibrated pool takes a pin: the chooser is live');
  const opts = options(kv);
  const by = Object.fromEntries(opts.map((o) => [o.value, o]));
  assert.ok(by['16000000000'] && !by['16000000000'].disabled, `16 GB is offered: ${JSON.stringify(opts)}`);
  assert.ok(by['18000000000'] && !by['18000000000'].disabled && /default/.test(by['18000000000'].text), '18 GB · default is offered');
  for (const v of ['20000000000', '24000000000', '28000000000', '32000000000', '40000000000', '48000000000', '64000000000']) {
    assert.ok(by[v], `${v} is listed`);
    assert.strictEqual(by[v].disabled, true, `${v} is past the bound: disabled`);
    assert.ok(by[v].text.includes('max 19.9 GB'), `${v} names the maximum as a figure: ${by[v].text}`);
  }
  assert.ok(!by['16000000000'].text.includes('max'), 'a pin under the bound carries no maximum');
  assert.strictEqual(kv.value, '24000000000', `the STORED pin is shown, so the refusal can name it: ${kv.value}`);
  assert.ok(api.fieldNote(C, 'kv_cache_memory', FLASH_SGLANG, false).includes('max 19.9 GB'),
    `the field note says the bound: ${api.fieldNote(C, 'kv_cache_memory', FLASH_SGLANG, false)}`);
  // …and the selection on screen is refused BY NAME before the Start, as the LB would refuse it (409).
  const bad = api.startProblem(C);
  assert.ok(bad && bad.key === 'kv_cache_memory', `the Start is refused on the pool: ${JSON.stringify(bad)}`);
  assert.ok(/24 GB/.test(bad.why) && /max 19\.9 GB/.test(bad.why) && /19,918,363,626/.test(bad.why), bad.why);
  assert.ok(bad.why.includes('SGLang NEXTN'), `the row is named: ${bad.why}`);
  // Picking a pin the profile takes clears it.
  kv.value = '18000000000';
  kv.dataset.dirty = '1';
  assert.strictEqual(api.startProblem(C), null, 'the model\'s own pin starts');
  kv.value = '19000000000';
  assert.strictEqual(api.selectionProblem(C, api.clusterChanges(C)), null, 'a pin under the bound is taken');
  kv.value = '20000000000';
  assert.ok(api.selectionProblem(C, api.clusterChanges(C)), 'a pin one step past the bound is refused');
}

// 2. The vLLM row has no bound of its own: nothing greyed, no "max" note, no refusal — the start gate decides.
{
  const { api, byId } = load();
  const vllm = { ...C, desired: { ...C.desired, model_choice: FLASH, uncensored: true, kv_cache_memory: 48000000000 } };
  controls(byId, FLASH);
  assert.strictEqual(api.kvMaxOf(vllm, FLASH), null);
  api.renderKnobs(vllm, true, FLASH);
  const kv = byId['cluster-kv'];
  const opts = options(kv);
  assert.ok(opts.length >= 9 && opts.every((o) => !o.disabled), `every pin offered: ${JSON.stringify(opts)}`);
  assert.ok(opts.every((o) => !o.text.includes('max')), 'no maximum named');
  assert.ok(!api.fieldNote(vllm, 'kv_cache_memory', FLASH, false).includes('max '), 'no bound in the note');
  assert.strictEqual(api.startProblem(vllm), null, 'a 48 GB pin is the gate\'s to judge, not the chooser\'s');
}

// 3. A pin stored for the vLLM row does not ride into the SGLang row's refusal: in the picker the other model's
//    own default is what the next launch uses, and it is under the bound.
{
  const { api, byId } = load();
  const stored = { ...C, desired: { ...C.desired, model_choice: FLASH, kv_cache_memory: 48000000000 } };
  controls(byId, FLASH_SGLANG);
  api.renderKnobs(stored, true, FLASH_SGLANG);
  assert.strictEqual(byId['cluster-kv'].value, '18000000000', 'the SGLang row shows its own pool, not the vLLM row\'s 48 GB');
  assert.strictEqual(api.startProblem(stored), null);
}

console.log('ok kv_ceiling: pins past the profile\'s bound are greyed with the maximum, the note says it, a stored one is refused by name');
