/* WHAT THE FLEET DECLARES A NODE RUNS, on the page beside what the node says it runs (`cluster.declared`,
 * topology.py `declaration` / `cross_check`; worklog/standalone-cluster-panel.md).
 *
 *   1. the model is the catalog row the fleet declares — its name, build and UNCENSORED tag, locked, its note the
 *      source — with the node's OWN engine and drafter (never the row's pair profile) and no catalog size; the card's
 *      model line and the server card name it too;
 *   2. seats show the declared figure, `declared by the fleet (not exposed by the node)`, and the pool line counts them;
 *   3. a key both say and agree on is marked `= fleet` / `node = fleet`; what only the fleet says (image, parsers) is
 *      one line under its words; no mismatch line;
 *   4. a DISAGREEMENT is loud: the card's mismatch line names every key with the fleet's value and the node's, the
 *      context field keeps the node's figure and warns with the fleet's — neither is chosen;
 *   5. a checked key the node did not report shows the fleet's value as declared, never as agreeing;
 *   6. a node that does not answer: no settings, no mismatch line; the pair: no declaration, nothing of this drawn.
 *
 *   node tests/cluster_declared.node.js   (tests/test_cluster_declared_page.py runs it)
 */
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const src = fs.readFileSync(path.join(__dirname, '..', 'vllm_lb', 'static', 'gateway.js'), 'utf8');
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

// The catalog row the fleet declares. Its PAIR profile says SGLang with an MTP drafter, and its size is the catalog's
// checkpoint's: none of the three is the node's, and none may be drawn for it.
const ROW = { id: 'model-a-nvfp4', label: 'Model A 27B · dense NVFP4', served_name: 'Org/Model-A', size_gib: 17.6,
  limits: {}, launch: { cluster: { engine: 'sglang', speculative_method: 'mtp', num_speculative_tokens: 3,
    default: true } }, kv_bytes_per_token: 1000, kv_blocks: null, calibrated: true,
  uncensored: { path: 'Model-A-NVFP4_uncensored', default: true, calibration: 'c' }, unsupported: '' };
const IMAGE = 'vllm/vllm-openai@sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef';
const LAUNCH = { image: IMAGE, served_name: 'model-a', max_model_len: 131072, max_num_seqs: 4,
  gpu_memory_utilization: 0.9, reasoning_parser: 'qwen3', tool_call_parser: 'qwen3_coder' };
const NX = 'not exposed by the node';
const agree = (v) => ({ fleet: v, node: v, verdict: 'agrees' });
const CHECKS = { image: { fleet: IMAGE, node: null, verdict: NX }, served_name: agree('model-a'),
  max_model_len: agree(131072), max_num_seqs: { fleet: 4, node: null, verdict: NX },
  gpu_memory_utilization: agree(0.9), reasoning_parser: { fleet: 'qwen3', node: null, verdict: NX },
  tool_call_parser: { fleet: 'qwen3_coder', node: null, verdict: NX } };
const TOPO = { tp: 1, boxes: ['node-a'], control: 'none', why: 'observed only — its server file starts, stops and sets it',
  steps: ['load', 'warm up', 'serve'] };
const EFFECTIVE = { model_choice: 'model-a-nvfp4', generation: null, max_num_seqs: 4, max_num_batched_tokens: null,
  num_speculative_tokens: 0, ple_replicate: null, kv_cache_memory: null, kv_offload_gib: null, max_model_len: 131072,
  long_prefill_token_threshold: null, uncensored: true, served_name: 'model-a', checkpoint: 'Model-A-abliterated-NVFP4',
  engine: 'vllm', gpu_memory_utilization: 0.9, kv_cache_dtype: 'fp8_e4m3', prefix_caching: true };
const declared = (checks) => ({ source: 'fleet.yml', cluster: 'node', model: 'model-a-nvfp4@uncensored',
  model_choice: 'model-a-nvfp4', uncensored: true, launch: LAUNCH, checks });
const clusterOf = (over) => Object.assign({ configured: true, topology: TOPO, state: 'running', held: true,
  detail: 'TP=1 up on node-a (observed)', desired: null, effective: EFFECTIVE, defaults: null,
  boxes: { 'node-a': { label: 'node-a-gpu', rank: { present: true, running: true }, hold: { blockers: [] } } },
  loaded_model: { served_name: 'model-a', spec: null, label: null, max_model_len: 131072 }, models: [], limits: {},
  since: 1791741000, up_since: 1791741000, declared: declared(CHECKS) }, over || {});
const node = { kind: 'cluster', url: 'http://node-a:8000', live: true, running: 0, waiting: 0, inflight: 0,
  depth_known: true, served_models: ['model-a'], max_model_len: 131072, history: {},
  throughput: { source: 'metrics', generated: { tok_s: 0 }, prompt: {}, engine_totals: {},
    kv_pool: { tokens: 168937, bytes: null, blocks: 116, block_tokens: 1568 } } };
const statusOf = (cluster) => ({ now: 1791742000, cluster, backends: [node], leases: [], hardware: {},
  broker: { enabled: false }, config: {}, models: { cluster: [ROW], classic: [] } });
const field = (id) => ({ text: byId(id).innerHTML, disabled: byId(id).disabled, note: byId(`${id}-note`).innerHTML,
  noteHidden: byId(`${id}-note`).hidden, warn: byId(`${id}-warn`).innerHTML, warnHidden: byId(`${id}-warn`).hidden });
let checks = 0;

// 1. The model: the declared row, locked, its note the source; the node's own engine; no drafter, no size.
{
  const s = statusOf(clusterOf());
  api.renderCluster(s);
  const trigger = byId('cluster-model-button').innerHTML;
  assert.ok(trigger.includes('Model A 27B') && trigger.includes('dense NVFP4'), `the declared row: ${trigger}`);
  assert.ok(/class="tag twin"/.test(trigger), `UNCENSORED, as declared: ${trigger}`);
  assert.ok(trigger.includes('vLLM') && !trigger.includes('SGLang'), `the node's engine, not the row's: ${trigger}`);
  assert.ok(!/MTP|17\.6 GiB/.test(trigger), `no drafter, no catalog size: ${trigger}`);
  assert.strictEqual(byId('cluster-model-button').disabled, true, 'locked');
  assert.strictEqual(byId('cluster-model-note').innerHTML, 'declared by the fleet (not exposed by the node)');
  assert.strictEqual(byId('cluster-model-note').hidden, false);
  const m = api.clusterModelInfo(s);
  assert.deepStrictEqual([m.label, m.engine, m.uncensored, m.speculativeTokens, m.draftMethod],
    ['Model A 27B · dense NVFP4', 'vllm', true, null, null], `the card's model: ${JSON.stringify(m)}`);
  const line = byId('cluster-model-line').innerHTML;
  assert.ok(line.includes('Model A 27B') && line.includes('4 seats') && /class="tag twin"/.test(line), `model line: ${line}`);
  const card = api.serverCard(s, node);
  assert.ok(card.includes('Model A 27B') && !card.includes('SGLang') && !/>MTP /.test(card), 'the server card names the row too');
  checks += 11;
}

// 2. Seats: the declared figure, its note the parent's words; the pool line counts them.
{
  const seqs = field('cluster-seqs');
  assert.ok(seqs.text.includes('>4<') && seqs.disabled, `seats: ${JSON.stringify(seqs)}`);
  assert.strictEqual(seqs.note, 'declared by the fleet (not exposed by the node)');
  const pool = byId('cluster-recommended').textContent;
  assert.ok(pool.includes('fits 1 @ 131,072') && pool.endsWith('4 seats > 1'), `pool line: ${pool}`);
  checks += 3;
}

// 3. Agreement marked where it is checked; the fleet-only keys on one line; no mismatch line.
{
  const ctx = field('cluster-context');
  assert.ok(ctx.text.includes('131,072') && ctx.note === 'node = fleet' && ctx.warnHidden, `context: ${JSON.stringify(ctx)}`);
  const facts = byId('cluster-node-facts').innerHTML;
  assert.ok(facts.includes('GPU memory 0.90 = fleet') && facts.includes('served model-a = fleet'), `facts: ${facts}`);
  const decl = byId('cluster-node-declared');
  assert.strictEqual(decl.hidden, false);
  assert.ok(decl.innerHTML.startsWith('declared by the fleet (not exposed by the node): '), decl.innerHTML);
  for (const want of [`image ${IMAGE}`, 'reasoning parser qwen3', 'tool-call parser qwen3_coder']) {
    assert.ok(decl.innerHTML.includes(want), `declared line carries ${want}: ${decl.innerHTML}`);
  }
  assert.ok(!decl.innerHTML.includes('seats'), 'seats have their own field');
  assert.strictEqual(byId('cluster-mismatch').hidden, true, 'nothing disagrees');
  checks += 8;
}

// 4. A disagreement, loud: both values on the card's mismatch line, the node's figure in the field with a warning.
{
  const s = statusOf(clusterOf({ declared: declared(Object.assign({}, CHECKS, {
    max_model_len: { fleet: 262144, node: 131072, verdict: 'mismatch' },
    gpu_memory_utilization: { fleet: 0.85, node: 0.9, verdict: 'mismatch' },
    served_name: { fleet: 'model-b', node: 'model-a', verdict: 'mismatch' } })) }));
  api.renderCluster(s);
  const bad = byId('cluster-mismatch');
  assert.strictEqual(bad.hidden, false, 'the mismatch line shows');
  for (const want of ['fleet ≠ node', 'context 262,144 ≠ 131,072', 'GPU memory 0.85 ≠ 0.90', 'served name model-b ≠ model-a']) {
    assert.ok(bad.innerHTML.includes(want), `mismatch line carries ${want}: ${bad.innerHTML}`);
  }
  const ctx = field('cluster-context');
  assert.ok(ctx.text.includes('131,072') && ctx.note === 'node', `the node's figure stays: ${JSON.stringify(ctx)}`);
  assert.ok(!ctx.warnHidden && ctx.warn.includes('≠ fleet 262,144'), `…and warns with the fleet's: ${JSON.stringify(ctx)}`);
  const facts = byId('cluster-node-facts').innerHTML;
  assert.ok(facts.includes('GPU memory 0.90 ≠ fleet 0.85') && facts.includes('served model-a ≠ fleet model-b'),
    `facts: ${facts}`);
  checks += 8;
}

// 5. A checked key the node did not report: the fleet's value, declared — never "= fleet".
{
  const eff = Object.assign({}, EFFECTIVE, { gpu_memory_utilization: null });
  const s = statusOf(clusterOf({ effective: eff, declared: declared(Object.assign({}, CHECKS, {
    gpu_memory_utilization: { fleet: 0.9, node: null, verdict: NX } })) }));
  api.renderCluster(s);
  assert.ok(!byId('cluster-node-facts').innerHTML.includes('GPU memory'), 'the node did not say it');
  assert.ok(byId('cluster-node-declared').innerHTML.includes('GPU memory 0.90'), 'the fleet did');
  assert.strictEqual(byId('cluster-mismatch').hidden, true);
  checks += 3;
}

// 6. A node that does not answer: no settings, no mismatch line. The pair: nothing of this drawn.
{
  const down = statusOf(clusterOf({ state: 'unknown', held: false, effective: null, loaded_model: null,
    declared: declared(null) }));
  api.renderCluster(down);
  assert.strictEqual(byId('cluster-settings').hidden, true);
  assert.strictEqual(byId('cluster-mismatch').hidden, true);
  checks += 2;
  // The pair, drawn right after a node's disagreement: its card carries none of the node's lines.
  api.renderCluster(statusOf(clusterOf({ declared: declared(Object.assign({}, CHECKS, {
    max_model_len: { fleet: 262144, node: 131072, verdict: 'mismatch' } })) })));
  assert.strictEqual(byId('cluster-mismatch').hidden, false);
  const PAIR = { tp: 2, boxes: ['pair-a', 'pair-b'], control: 'ssh', why: '', steps: ['free', 'load', 'warm up', 'serve'] };
  const rank = { present: true, running: true, gen: 189 };
  const pairBox = (label) => ({ label, reservation: null, running_jobs: 0, rank, mem_available_gib: 20,
    hold: { blockers: [] } });
  const pair = { configured: true, topology: PAIR, state: 'running', held: true, detail: 'serving',
    desired: { mode: 'cluster', model_choice: 'model-a-nvfp4', max_num_seqs: 8 },
    effective: { generation: 189, model_choice: 'model-a-nvfp4', max_num_seqs: 8, max_model_len: 262144 },
    defaults: {}, setups: {}, boxes: { 'pair-a': pairBox('pair-a-gpu'), 'pair-b': pairBox('pair-b-gpu') },
    loaded_model: null, models: [ROW], limits: {}, since: 1791741000, up_since: 1791741000, declared: null };
  api.renderCluster(statusOf(pair));
  assert.strictEqual(byId('cluster-mismatch').hidden, true, 'no mismatch line on the pair');
  assert.strictEqual(byId('cluster-node-declared').hidden, true, 'no declared line on the pair');
  assert.strictEqual(byId('cluster-model-note').hidden, true, 'the pair\'s model note is its own');
  checks += 4;
}

console.log(`ok — ${checks} checks`);
