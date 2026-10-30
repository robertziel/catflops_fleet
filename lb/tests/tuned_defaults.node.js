/* THE MEASURED-BEST START INPUT, ON THE PAGE — checked where the page decides what each control shows.
 *
 * Operator, 2026-10-05: "can we set specific defaults per model, best we got so far, so switching model
 * changes automatically values to best defaults hardcoded in the system". The catalog states them per model
 * (`launch.cluster.tuned`, catalog.py TunedSetup); this is the half that puts them in front of the operator:
 * pick another model and the concurrency, context, prefill budget and drafter depth on screen become THAT
 * model's own, with the run they were measured on behind the note — and a model with nothing measured still
 * shows the gateway's configured defaults, as every model did before.
 *
 *   node tests/tuned_defaults.node.js   (tests/test_tuned_defaults_page.py runs this when node exists)
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
    `${src}\n;return { renderKnobs, clusterChanges, fieldNote, storedOf, defaultOf, defaultsToFill };`)(
    document, window, () => new Promise(() => {}), { search: '', href: '' }, () => 0, () => 0,
    () => ({}), undefined, window.localStorage, { userAgent: 'node' });
  return { api, byId };
}

const SGLANG = 'qwen3.8-27b-sglang-dflash2';
const FLASH = 'qwen3.8-flash-next-hibrid48';
const DENSE = 'qwen3.8-27b-nvidia-nvfp4';
//: The three models' catalog entries as /lb/status publishes them — two with a tuned setup, one without.
const FLASH_TUNED = { max_num_seqs: 6, max_model_len: 262144, max_num_batched_tokens: 4096,
  num_speculative_tokens: 4, source: "OPERATOR'S CHOICE 2026-10-05, generation 99's budget and P1's depth" };
const SGLANG_TUNED = { max_num_seqs: 32, max_model_len: 262144, max_num_batched_tokens: 2048,
  num_speculative_tokens: 4, source: 'GENERATION 137, 2026-10-03 to 2026-10-04: 13,210 requests' };
const MODELS = [
  { id: SGLANG, label: 'Qwen3.8 27B · SGLang DFlash2', limits: { max_model_len: 262144, max_num_seqs: [1, 48] },
    kv_bytes_per_token: 21502, calibrated: true,
    launch: { cluster: { engine: 'sglang', speculative: true, num_speculative_tokens: 4, speculative_method: 'dflash',
      kv_cache_memory: 73435823273, ple_replicate: null, tuned: SGLANG_TUNED, refuses: {} } } },
  { id: FLASH, label: 'Qwen3.8 Flash-Next', limits: { max_model_len: 262144, max_num_seqs: [1, 48] },
    kv_bytes_per_token: 16392, calibrated: true,
    launch: { cluster: { engine: 'vllm', speculative: true, num_speculative_tokens: 4, speculative_method: 'mtp',
      kv_cache_memory: 28000000000, ple_replicate: true, tuned: FLASH_TUNED, refuses: {} } } },
  { id: DENSE, label: 'Qwen3.8 27B · dense NVFP4', limits: { max_model_len: 262144, max_num_seqs: [1, 48] },
    kv_bytes_per_token: 36860, calibrated: true,
    launch: { cluster: { engine: 'vllm', speculative: true, num_speculative_tokens: 4, speculative_method: 'mtp',
      kv_cache_memory: 48000000000, ple_replicate: null, tuned: null, refuses: {} } } },
];
//: /lb/status `cluster` with the SGLang pair stored at its own measured-best, stopped (so everything is editable).
const C = {
  configured: true, state: 'off', held: false, models: MODELS,
  desired: { mode: 'classic', generation: 137, model_choice: SGLANG, max_num_seqs: 32, max_model_len: 262144,
    max_num_batched_tokens: 2048, num_speculative_tokens: null, ple_replicate: null,
    kv_cache_memory: null, kv_cache_auto: false, kv_offload_gib: 0, long_prefill_token_threshold: null },
  limits: { max_num_seqs: [1, 48], max_model_len: [1024, 262144], max_num_batched_tokens: [2048, 65536],
    num_speculative_tokens: [0, 8], kv_cache_memory: [8000000000, 96000000000], kv_offload_gib: [0, 512],
    long_prefill_token_threshold: [0, 65536], ple_replicate: [0, 1] },
  // Resolved by the LB FOR THE STORED MODEL — SGLang's own tuned figures (cluster_loop._defaults).
  defaults: { max_num_seqs: 32, max_model_len: 262144, max_num_batched_tokens: 2048, num_speculative_tokens: 4,
    kv_cache_memory: 73435823273, kv_offload_gib: 0, ple_replicate: null },
  effective: null, boxes: {},
};

{
  // SWITCHING THE MODEL IN THE PICKER moves the whole start input to the model picked, not the one stored.
  const { api } = load();
  for (const [key, want] of Object.entries({ max_num_seqs: 6, max_model_len: 262144,
                                             max_num_batched_tokens: 4096, num_speculative_tokens: 4 })) {
    assert.strictEqual(api.storedOf(C, key, FLASH), want, `${key} for the picked model`);
    assert.strictEqual(api.defaultOf(C, key, FLASH), want, `${key} default for the picked model`);
  }
  // …and the model the switch holds still shows what is stored for it.
  assert.strictEqual(api.storedOf(C, 'max_num_seqs', SGLANG), 32);
  assert.strictEqual(api.storedOf(C, 'max_num_batched_tokens', SGLANG), 2048);
}
{
  // The two controls `renderKnobs` owns really take those values when another model is picked.
  const { api, byId } = load();
  byId['cluster-model'] = Object.assign(byId['cluster-model'] || {}, { value: FLASH });
  api.renderKnobs(C, true, FLASH);
  assert.strictEqual(byId['cluster-batched'].value, '4096', 'the prefill budget Flash-Next was measured at');
  assert.strictEqual(byId['cluster-spec'].value, '4');
  assert.ok(byId['cluster-batched'].innerHTML.includes('4096</option>')
            || byId['cluster-batched'].innerHTML.includes('4,096 · default'), byId['cluster-batched'].innerHTML);
}
{
  // A MEASURED DEFAULT SAYS SO, and carries the run it was measured on — a number the page fills in for the
  // operator must be able to say where it came from.
  const { api } = load();
  const note = api.fieldNote(C, 'max_num_seqs', FLASH, false);
  assert.ok(note.includes('default 6') && note.includes('measured'), note);
  assert.ok(note.includes('title="') && note.includes('2026-10-05'), `the run it was measured on: ${note}`);
  // The pool is not a tuned knob: its note is the plain default, with no measured claim.
  assert.ok(!api.fieldNote(C, 'kv_cache_memory', FLASH, false).includes('measured'));
}
{
  // A MODEL WITH NOTHING MEASURED is offered the LB's own published defaults, exactly as before.
  const { api } = load();
  assert.strictEqual(api.storedOf(C, 'max_num_seqs', DENSE), C.defaults.max_num_seqs);
  assert.strictEqual(api.storedOf(C, 'max_num_batched_tokens', DENSE), C.defaults.max_num_batched_tokens);
  assert.ok(!api.fieldNote(C, 'max_num_seqs', DENSE, false).includes('measured'));
}
{
  // WHAT APPLY WOULD SEND after the switch: every knob that moved, named, with the stored value it replaces.
  const { api, byId } = load();
  byId['cluster-model'] = Object.assign(byId['cluster-model'] || {}, { value: FLASH });
  for (const [id, value] of Object.entries({ 'cluster-seqs': '6', 'cluster-context': '262144',
                                             'cluster-batched': '4096', 'cluster-spec': '4',
                                             'cluster-kv': '28000000000', 'cluster-ple': '2',
                                             'cluster-offload': '0' })) {
    byId[id] = Object.assign(byId[id] || {}, { value });
  }
  const changes = api.clusterChanges(C);
  const moved = Object.fromEntries(changes.map((ch) => [ch.key, ch.to]));
  assert.strictEqual(moved.model_choice, FLASH, JSON.stringify(moved));
  assert.strictEqual(moved.max_num_seqs, 6, JSON.stringify(moved));
  assert.strictEqual(moved.max_num_batched_tokens, 4096, JSON.stringify(moved));
  assert.ok(!('max_model_len' in moved), 'both models are measured at the full window: nothing to change');
  // The two the switch always holds as numbers travel as numbers: null would mean "keep what is stored",
  // and the old model's seat count would ride into the new model's first start.
  const seats = changes.find((ch) => ch.key === 'max_num_seqs');
  assert.strictEqual(seats.send, 6, JSON.stringify(seats));
  assert.strictEqual(seats.from, 32, 'listed from what the OLD model held');
  // The nullable ones go as null, so they follow the new model's own default from then on.
  const budget = changes.find((ch) => ch.key === 'max_num_batched_tokens');
  assert.ok(budget.reset && budget.send === null, JSON.stringify(budget));
}
{
  // NO CONTROL SAYS "default" TWICE. The words a value is written in are read INSIDE sentences that
  // already carry the word — the note under a field is "default <words>", the option list marks the
  // default one "<words> · default" — so a value whose own words were "model default" printed
  // "default model default" under the PLE table, and "model default · default" in its list
  // (operator's screenshot, 2026-10-05).
  const { api, byId } = load();
  byId['cluster-model'] = Object.assign(byId['cluster-model'] || {}, { value: FLASH });
  api.renderKnobs(C, true, FLASH);
  for (const key of ['max_num_seqs', 'max_model_len', 'max_num_batched_tokens', 'num_speculative_tokens',
                     'kv_cache_memory', 'ple_replicate', 'kv_offload_gib']) {
    const clause = api.fieldNote(C, key, FLASH, false).replace(/<[^>]*>/g, '').split('\u00b7')[0];
    assert.ok(/^\s*default\s/.test(clause), `${key}: the note opens with its default — ${clause}`);
    assert.ok(!/default[\s\S]*default/i.test(clause), `${key} says default twice: ${clause}`);
  }
  const opts = byId['cluster-ple'].innerHTML.replace(/<[^>]*>/g, '|');
  assert.ok(opts.includes('as the model says'), opts);
  for (const opt of opts.split('|').filter(Boolean)) {
    assert.ok(!/default[\s\S]*default/i.test(opt), `the PLE list says default twice: ${opt}`);
  }
}
console.log('ok');
