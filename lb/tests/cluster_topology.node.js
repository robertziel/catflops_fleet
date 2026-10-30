/* THE CLUSTER CARD NAMES WHAT THE CLUSTER IS — checked where the page builds it, under node.
 *
 * `/lb/status.cluster.topology` (vllm_lb/topology.py): the TP, the boxes, the control channel and the start steps
 * that apply. Until it existed the page named every cluster after one deployment's pair, `TP=2`, and drew six start steps,
 * whatever it was — so a standalone server's one-GPU node could not be shown as the cluster at all
 * (worklog/standalone-cluster-panel.md). Now:
 *
 *   1. the PAIR renders exactly as it always did: the same name, the same `TP=2 <engine>`, the same six steps — and a
 *      captured document from before the field (no `topology`) renders identically, its boxes being the topology's;
 *   2. ONE observed node is named by its box, `TP=1`, its strip only the steps that apply (load, warm up, serve) and
 *      none while it does not answer; its Start/Stop is not offered — the reason is (`topology.why`).
 *
 * No DOM: gateway.js is loaded with stub elements and a `fetch` that never answers.
 *
 *   node tests/cluster_topology.node.js   (tests/test_cluster_topology_page.py runs this when node exists)
 */
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const src = fs.readFileSync(path.join(__dirname, '..', 'vllm_lb', 'static', 'gateway.js'), 'utf8');
const els = new Map();
const el = () => ({ addEventListener() {}, removeAttribute() {}, setAttribute() {}, append() {},
  classList: { toggle() {}, add() {}, remove() {}, contains: () => false }, style: {}, hidden: false, innerHTML: '',
  textContent: '', value: '', dataset: {}, querySelectorAll: () => [], querySelector: () => null,
  focus() {}, observe() {}, closest: () => null, getBoundingClientRect: () => ({}) });
const byId = (id) => { if (!els.has(id)) els.set(id, el()); return els.get(id); };
const document = { getElementById: byId, createElementNS: el, createElement: el, addEventListener() {},
  querySelectorAll: () => [], querySelector: () => null, activeElement: null,
  body: { style: {}, classList: { toggle() {} } }, documentElement: { style: {}, classList: { toggle() {} } } };
const window = { addEventListener() {}, innerWidth: 1280, innerHeight: 800, matchMedia: () => ({
  matches: false, addEventListener() {} }), localStorage: { getItem: () => null, setItem() {} } };

const api = new Function('document', 'window', 'fetch', 'location', 'setInterval', 'setTimeout',
  'getComputedStyle', 'ResizeObserver', 'localStorage', 'navigator',
  `${src}\n;return { serverCard, clusterTrack, renderPrimary, topologyOf, clusterName, ranksOf };`)(
  document, window, () => new Promise(() => {}), { search: '', href: '' }, () => 0, () => 0,
  () => ({}), undefined, window.localStorage, { userAgent: 'node' });

const NOW = 1791742000;
const box = (rank) => ({ label: 'x', reservation: { status: 'running', job_id: 'abcdef12' }, running_jobs: 0,
  rank, mem_available_gib: 20, hold: { blockers: [] } });
const RUNNING = { present: true, running: true, gen: 189 };
const PAIR_TOPO = { tp: 2, boxes: ['pair-a', 'pair-b'], control: 'ssh', why: '',
  steps: ['free', 'reserve', 'drain', 'load', 'warm up', 'serve'] };
const pairCluster = (topo) => Object.assign({ configured: true, state: 'running', held: true,
  desired: { mode: 'cluster', model_choice: 'm' }, effective: { generation: 189, model_choice: 'm' },
  boxes: { 'pair-a': box(RUNNING), 'pair-b': box(RUNNING) }, models: [] }, topo ? { topology: topo } : {});
const head = { kind: 'cluster', url: 'http://192.0.2.10:8100', live: true, running: 0, waiting: 0, inflight: 0,
  depth_known: true, served_models: ['m'], max_model_len: 262144, history: {},
  throughput: { source: 'metrics', generated: { tok_s: 0 }, prompt: {}, engine_totals: {} } };
const status = (cluster, extra) => Object.assign({ now: NOW, cluster, backends: [head], leases: [], hardware: {},
  broker: { enabled: true }, config: {}, models: { cluster: [], classic: [] } }, extra || {});
let checks = 0;

// 1. THE PAIR, with its topology and without it (a document from before the field): one rendering, today's.
{
  const withTopo = status(pairCluster(PAIR_TOPO));
  const before = status(pairCluster(null));
  assert.strictEqual(api.clusterName(withTopo), 'pair-a + pair-b');
  assert.strictEqual(api.clusterName(before), 'pair-a + pair-b', 'a pre-topology document names its own boxes');
  const card = api.serverCard(withTopo, head);
  assert.ok(card.includes('pair-a + pair-b') && /&middot; TP=2</.test(card), `the pair's card: ${card.slice(0, 400)}`);
  assert.strictEqual(card, api.serverCard(before, head), 'the pair renders identically with and without `topology`');
  const labels = api.clusterTrack(withTopo.cluster).steps.map((s) => s.label);
  assert.deepStrictEqual(labels, ['free', 'reserve', 'drain', 'load', 'warm up', 'serve']);
  assert.deepStrictEqual(api.clusterTrack(before.cluster).steps.map((s) => s.label), labels);
  assert.strictEqual(api.ranksOf(withTopo), 2);
  checks += 7;
}

// 2. ONE OBSERVED NODE: its box's name, TP=1, three steps that are facts while it serves, none while it does not.
const NODE_TOPO = { tp: 1, boxes: ['node-a'], control: 'none', why: 'observed only (control: none): reasons',
  steps: ['load', 'warm up', 'serve'] };
const nodeCluster = (state) => ({ configured: true, topology: NODE_TOPO, state, held: state === 'running',
  desired: null, effective: null, boxes: { 'node-a': { label: 'node-a-gpu', rank: state === 'running' ? RUNNING : null,
    hold: { blockers: [] } } }, loaded_model: null, models: [] });
const node = Object.assign({}, head, { url: 'http://node-a:8000', served_models: ['model-a'], max_model_len: 131072 });
{
  const s = Object.assign(status(nodeCluster('running')), { backends: [node], broker: { enabled: false } });
  assert.strictEqual(api.clusterName(s), 'node-a');
  const card = api.serverCard(s, node);
  assert.ok(/class="box-name">node-a <span class="box-kind">&middot; TP=1<\/span>/.test(card),
    `named by its box, TP=1: ${card.slice(0, 400)}`);
  assert.ok(!card.includes('TP=2'), 'no pair TP on a node');
  const track = api.clusterTrack(s.cluster);
  assert.deepStrictEqual(track.steps.map((x) => [x.label, x.status]),
    [['load', 'done'], ['warm up', 'done'], ['serve', 'done']]);
  assert.strictEqual(api.clusterTrack(nodeCluster('unknown')), null, 'no strip while it does not answer');
  const down = Object.assign(status(nodeCluster('unknown')), { backends: [Object.assign({}, node, { live: false })],
    broker: { enabled: false } });
  const downCard = api.serverCard(down, down.backends[0]);
  assert.ok(downCard.includes('NOT ANSWERING') && !downCard.includes('NOT HELD'),
    `a node that does not answer says so, amber: ${downCard.slice(0, 500)}`);
  checks += 1;
  assert.strictEqual(api.ranksOf(s), 1);
  checks += 7;

  // Its Start/Stop is not offered — the one sentence why is.
  api.renderPrimary(s.cluster);
  assert.strictEqual(byId('cluster-toggle').hidden, true, 'no Start/Stop for a node this LB cannot drive');
  assert.strictEqual(byId('cluster-caption').textContent, NODE_TOPO.why);
  assert.strictEqual(byId('cluster-caption').hidden, false);
  checks += 3;
}

// 4. TWO CLUSTERS (worklog/parallel-clusters.md): a cluster backend is named and held by ITS OWN view, by the name it
//    carries — never by the one-cluster alias, which names none of several.
{
  const second = { kind: 'cluster', url: 'http://198.51.100.20:8000', live: true, cluster: 'gui', running: 0, waiting: 0,
    inflight: 0, depth_known: true, served_models: ['g'], max_model_len: 32768, history: {},
    throughput: { source: 'metrics', generated: { tok_s: 0 }, prompt: {}, engine_totals: {} } };
  const gui = { configured: true, name: 'gui', state: 'running', held: true, desired: { mode: 'cluster' },
    topology: { tp: 1, boxes: ['node-c'], control: 'local', why: '', steps: ['load', 'warm up', 'serve'] }, boxes: {} };
  const s = status({ configured: false, clusters: ['pair', 'gui'] }, {
    backends: [Object.assign({}, head, { cluster: 'pair' }), second],
    clusters: { pair: Object.assign(pairCluster(PAIR_TOPO), { name: 'pair' }), gui } });
  assert.strictEqual(api.clusterName(s, s.backends[0]), 'pair-a + pair-b', 'the pair is named by its own view');
  assert.strictEqual(api.clusterName(s, second), 'node-c', 'the node by its own');
  const html = api.serverCard(s, second, 'cluster');
  assert.ok(/node-c/.test(html) && !/TP=0/.test(html), `the node's card is its own: ${html.slice(0, 300)}`);
  checks += 3;
}

console.log(`ok — ${checks} checks`);
