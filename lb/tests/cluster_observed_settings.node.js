/* THE SETTINGS OF A NODE THIS LB ONLY OBSERVES — the same group the pair's card has, showing what the node SAYS it
 * runs with (`cluster.effective` from its own HTTP API, topology.py `observed_effective`), every field locked, every
 * note its SOURCE: `node` for what it reports, `not exposed by the node` for what it does not. No default is offered,
 * no change can be made — there is no switch to send one to (worklog/standalone-cluster-panel.md, slice 3).
 *
 *   1. the fields: context and drafter from the node; seats and prefill budget "—", not exposed; no Apply, no
 *      defaults, no estimate, no "locked until off";
 *   2. the KV pool line from the ENGINE's pool (tokens, no GB — the node pinned none) and its window;
 *   3. the facts line: engine, GPU memory fraction, KV cache dtype, prefix caching, checkpoint;
 *   4. the engine on the model badge and the server card is the node's own word, `TP=1 vLLM`.
 *
 *   node tests/cluster_observed_settings.node.js   (tests/test_cluster_observed_settings_page.py runs it)
 */
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');

// palette.js first: the colours this script computes are read from the style's tokens (tests/palette_env.js).
const src = require('./palette_env').withPalette(fs.readFileSync(path.join(__dirname, '..', 'vllm_lb', 'static', 'gateway.js'), 'utf8'));
const els = new Map();
const el = () => ({ addEventListener() {}, removeAttribute() {}, setAttribute() {}, append() {},
  classList: { toggle() {}, add() {}, remove() {}, contains: () => false }, style: {}, hidden: false, innerHTML: '',
  textContent: '', value: '', dataset: {}, disabled: false, open: false, querySelectorAll: () => [],
  querySelector: () => null, focus() {}, observe() {}, closest: () => null, contains: () => false,
  getBoundingClientRect: () => ({}) });
const byId = (id) => { if (!els.has(id)) els.set(id, el()); return els.get(id); };
const document = { getElementById: byId, createElementNS: el, createElement: el, addEventListener() {},
  querySelectorAll: () => [], querySelector: () => null, activeElement: null,
  body: { style: {}, classList: { toggle() {} } }, documentElement: { style: {}, classList: { toggle() {} } } };
const window = { addEventListener() {}, innerWidth: 1280, innerHeight: 800, matchMedia: () => ({
  matches: false, addEventListener() {} }), localStorage: { getItem: () => null, setItem() {} } };

const api = new Function('document', 'window', 'fetch', 'location', 'setInterval', 'setTimeout',
  'getComputedStyle', 'ResizeObserver', 'localStorage', 'navigator',
  `${src}\n;return { renderCluster, serverCard, clusterModelInfo };`)(
  document, window, () => new Promise(() => {}), { search: '', href: '' }, () => 0, () => 0,
  () => ({}), undefined, window.localStorage, { userAgent: 'node' });

const TOPO = { tp: 1, boxes: ['node-a'], control: 'none', why: 'observed only — its server file starts, stops and sets it',
  steps: ['load', 'warm up', 'serve'] };
const EFFECTIVE = { model_choice: null, generation: null, max_num_seqs: null, max_num_batched_tokens: null,
  num_speculative_tokens: 0, ple_replicate: null, kv_cache_memory: null, kv_offload_gib: null, max_model_len: 131072,
  long_prefill_token_threshold: null, uncensored: null, served_name: 'model-a', checkpoint: 'Model-A-NVFP4',
  engine: 'vllm', gpu_memory_utilization: 0.9, kv_cache_dtype: 'fp8_e4m3', prefix_caching: true };
const cluster = { configured: true, topology: TOPO, state: 'running', held: true, detail: 'TP=1 up on node-a (observed)',
  desired: null, effective: EFFECTIVE, defaults: null, boxes: { 'node-a': { label: 'node-a-gpu',
    rank: { present: true, running: true }, hold: { blockers: [] } } }, loaded_model: { served_name: 'model-a',
    spec: null, label: null, max_model_len: 131072 }, models: [], limits: {}, since: 1791741000, up_since: 1791741000 };
const node = { kind: 'cluster', url: 'http://node-a:8000', live: true, running: 0, waiting: 0, inflight: 0,
  depth_known: true, served_models: ['model-a'], max_model_len: 131072, history: {},
  throughput: { source: 'metrics', generated: { tok_s: 0 }, prompt: {}, engine_totals: {},
    kv_pool: { tokens: 168937, bytes: null, blocks: 116, block_tokens: 1568 } } };
const s = { now: 1791742000, cluster, backends: [node], leases: [], hardware: {}, broker: { enabled: false },
  config: {}, models: { cluster: [], classic: [] } };
let checks = 0;

api.renderCluster(s);

// 1. The fields: the node's own figures, locked, each note its source; nothing offered that cannot be sent.
{
  assert.strictEqual(byId('cluster-settings').hidden, false, 'the settings are shown');
  const field = (id) => ({ text: byId(id).innerHTML, disabled: byId(id).disabled, note: byId(`${id}-note`).innerHTML });
  const ctx = field('cluster-context');
  assert.ok(ctx.text.includes('131,072') && ctx.disabled && ctx.note === 'node', `context: ${JSON.stringify(ctx)}`);
  const seqs = field('cluster-seqs');
  assert.ok(seqs.text.includes('—') && seqs.disabled && seqs.note === 'not exposed by the node',
    `seats: ${JSON.stringify(seqs)}`);
  const batched = field('cluster-batched');
  assert.ok(batched.text.includes('—') && batched.note === 'not exposed by the node', `budget: ${JSON.stringify(batched)}`);
  const spec = field('cluster-spec');
  assert.ok(spec.text.includes('off') && spec.disabled && spec.note === 'node', `drafter: ${JSON.stringify(spec)}`);
  for (const id of ['cluster-apply', 'cluster-defaults', 'cluster-estimate', 'cluster-hint']) {
    assert.strictEqual(byId(id).hidden, true, `${id} is not offered for a node this LB only observes`);
  }
  checks += 8;
}

// 2. The KV pool line: the engine's own pool and window — no GB, the node pinned none.
{
  const line = byId('cluster-recommended').textContent;
  assert.ok(line.startsWith('KV pool 168,937 tokens') && line.includes('fits 1 @ 131,072'), `pool: ${line}`);
  assert.ok(!/ GB/.test(line), 'no pin, no GB');
  checks += 2;
}

// 3. The facts line: what the node says it runs with.
{
  const facts = byId('cluster-node-facts').innerHTML;
  for (const want of ['vLLM', 'GPU memory 0.90', 'KV fp8_e4m3', 'prefix caching on', 'Model-A-NVFP4']) {
    assert.ok(facts.includes(want), `facts carry ${want}: ${facts}`);
  }
  assert.strictEqual(byId('cluster-node-facts').hidden, false);
  checks += 6;
}

// 4. The engine is the node's own word, on the badge and on the server card.
{
  assert.strictEqual(api.clusterModelInfo(s).engine, 'vllm');
  assert.ok(/&middot; TP=1 vLLM</.test(api.serverCard(s, node)), 'TP=1 vLLM on the server card');
  checks += 2;
}

console.log(`ok — ${checks} checks`);
