/* A SPEECH NODE'S SETTINGS ARE ITS SELECTED ENGINE'S (operator 2026-10-23: knobs an engine does not read must not exist
 * for it). The rows drawn are the declaration's (`cluster.engines`, tests/fixtures/engine-declarations.json — what
 * fleet.py NodeEngine.view publishes): faster-whisper shows seats, queue and compute type — no context, no GPU memory
 * fraction, no prefill budget, no KV rows — and a Start sends those alone; the vLLM choice of the same cluster shows
 * vLLM's speech knobs. The queue and the compute type have no house editor: they are drawn from the declaration.
 *
 *   node tests/speech_settings.node.js   (tests/test_speech_settings.py runs it)
 */
'use strict';
const ENGINE_DECLS = JSON.parse(require('fs').readFileSync(require('path').join(__dirname, 'fixtures', 'engine-declarations.json'), 'utf8'));
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const src = fs.readFileSync(path.join(__dirname, '..', 'vllm_lb', 'static', 'gateway.js'), 'utf8');
const els = new Map();
const el = () => ({ addEventListener() {}, removeAttribute() {}, setAttribute() {}, append() {},
  classList: { _s: new Set(), toggle(c, on) { if (on === undefined ? !this._s.has(c) : on) this._s.add(c); else this._s.delete(c); },
    add(c) { this._s.add(c); }, remove(c) { this._s.delete(c); }, contains(c) { return this._s.has(c); } },
  style: {}, hidden: false, innerHTML: '', textContent: '', value: '', dataset: {}, disabled: false, open: false,
  title: '', querySelectorAll: () => [], querySelector: () => null, focus() {}, observe() {}, closest: () => null,
  contains: () => false, getBoundingClientRect: () => ({}) });
const byId = (id) => { if (!els.has(id)) els.set(id, el()); return els.get(id); };
const document = { getElementById: byId, createElementNS: el, createElement: el, addEventListener() {},
  querySelectorAll: () => [], querySelector: () => null, activeElement: null,
  body: { style: {}, classList: { toggle() {} } }, documentElement: { style: {}, classList: { toggle() {} } } };
const window = { addEventListener() {}, innerWidth: 1280, innerHeight: 800, matchMedia: () => ({
  matches: false, addEventListener() {} }), localStorage: { getItem: () => null, setItem() {} } };

/** What the page's fetch answers: nothing, unless a check says otherwise (an estimate, asked as a field is picked). */
let fetchImpl = () => new Promise(() => {});
const api = new Function('document', 'window', 'fetch', 'location', 'setInterval', 'setTimeout',
  'getComputedStyle', 'ResizeObserver', 'localStorage', 'navigator',
  `${src}\n;return { renderCluster, localBody, localChanges, askLocalFit, toggleLocal };`)(
  document, window, (...a) => fetchImpl(...a), { search: '', href: '' }, () => 0, () => 0,
  () => ({}), undefined, window.localStorage, { userAgent: 'node' });

const FW = 'faster-whisper-large-v3-turbo', VW = 'whisper-large-v3-turbo';
const choice = (ref, engine) => ({ ref, model_choice: ref, uncensored: false, label: ref, checkpoint: ref, size_gib: 1.5,
  limits: { max_model_len: 448, max_num_seqs: [1, 4] }, overhead_chunk: null, kv_pin_max: null, engine });
const desired = (ref, extra) => Object.assign({ mode: 'cluster', id: 2, force: false, model: ref, model_choice: ref,
  uncensored: false, max_num_seqs: 4 }, extra);
const clusterOf = (d) => ({ configured: true, task: 'transcription', tiles: ['speech'],
  topology: { tp: 1, boxes: ['apu'], control: 'local', why: '', steps: ['load', 'warm up', 'serve'] },
  state: 'running', held: true, detail: '', desired: d, effective: {}, boxes: {}, loaded_model: null, models: [],
  limits: { max_model_len: 448, max_num_seqs: [1, 4] }, since: 1, up_since: 1, declared: null,
  applier: { listening: true, heartbeat_age_s: 3, host: 'apu-host', applying: null, applied: null, container: null,
    error: '', log_tail: null },
  choices: [choice(VW, 'vllm'), choice(FW, 'faster-whisper')], fit: { fit: { text: '2.59 GiB measured whole engine @ 4 requests ≤ 32 GiB' }, refused: '' },
  loading: {}, store_error: '', ram_tier: { max_gib: null, refused: 'unified' }, kv_cache_dtypes: {},
  engines: { vllm: ENGINE_DECLS['vllm/transcription'], 'faster-whisper': ENGINE_DECLS['faster-whisper/transcription'] } });
const statusOf = (cluster) => ({ now: 2, cluster, backends: [], leases: [], hardware: {}, broker: { enabled: false },
  config: {}, models: { cluster: [], classic: [] } });
const html = (id) => byId(id).innerHTML;
let checks = 0;

// 1. faster-whisper selected: seats shown, vLLM's rows hidden, the queue and compute type drawn from the declaration.
{
  const c = clusterOf(desired(FW, { max_queue: 8, compute_type: null }));
  api.renderCluster(statusOf(c));
  assert.strictEqual(byId('cluster-seqs').hidden, false, 'seats: both engines read them');
  for (const id of ['cluster-context', 'cluster-util-row', 'cluster-batched', 'cluster-kv', 'cluster-offload',
    'cluster-dtype-row', 'cluster-spec', 'cluster-ple']) {
    assert.strictEqual(byId(id).hidden, true, `${id}: not a faster-whisper setting`);
  }
  const fields = html('cluster-engine-fields');
  assert.strictEqual(byId('cluster-engine-fields').hidden, false);
  assert.ok(fields.includes('id="cluster-set-max_queue"') && fields.includes('id="cluster-set-compute_type"'), fields);
  assert.ok(html('cluster-set-max_queue').includes('value="8" selected'), html('cluster-set-max_queue'));
  assert.ok(html('cluster-set-compute_type').includes('value="int8_float16"'), html('cluster-set-compute_type'));
  assert.ok(!html('cluster-set-compute_type').includes('float32'), 'the declaration offers no float32');
  assert.ok(html('cluster-set-compute_type').includes('the image&#39;s own'), 'its unset word');
  assert.ok(html('cluster-set-max_queue-note').includes('EDGE_NODE_MAX_QUEUE'), 'the node variable it reaches');
  assert.strictEqual(byId('cluster-fit').textContent, '2.59 GiB measured whole engine @ 4 requests ≤ 32 GiB');
  const body = api.localBody(c, 'start');
  assert.deepStrictEqual(Object.keys(body).sort(), ['compute_type', 'enabled', 'max_num_seqs', 'max_queue', 'model']);
  // An edit of a declared field is a change, in its own words.
  byId('cluster-set-compute_type').value = 'int8'; byId('cluster-set-compute_type').dataset.dirty = '1';
  const ch = api.localChanges(c);
  assert.deepStrictEqual(ch.map((x) => [x.key, x.to]), [['compute_type', 'int8']]);
  delete byId('cluster-set-compute_type').dataset.dirty;
  checks += 18;
}

// 2. The vLLM choice of the same cluster: its speech knobs, none of faster-whisper's.
{
  const c = clusterOf(desired(VW, { max_model_len: 448, gpu_memory_utilization: 0.05, max_num_batched_tokens: null,
    kv_cache_memory: null }));
  api.renderCluster(statusOf(c));
  for (const id of ['cluster-seqs', 'cluster-context', 'cluster-util-row', 'cluster-batched', 'cluster-kv']) {
    assert.strictEqual(byId(id).hidden, false, `${id}: vLLM reads it`);
  }
  for (const id of ['cluster-offload', 'cluster-dtype-row']) assert.strictEqual(byId(id).hidden, true, id);
  assert.strictEqual(byId('cluster-engine-fields').hidden, true, 'no setting only a declaration names');
  const body = api.localBody(c, 'start');
  assert.ok(!('max_queue' in body) && !('compute_type' in body), JSON.stringify(body));
  checks += 9;
}
console.log(`ok — ${checks} checks`);
