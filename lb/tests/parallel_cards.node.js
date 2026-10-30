/* EACH CLUSTER'S CARD IS ITS OWN (worklog/parallel-clusters.md; AUDIT-02 P0/P1): with two clusters in the run, every
 * figure on a cluster backend's card is read off THAT backend's cluster view (`clusterViewOf`, by `b.cluster`) — never
 * off the SELECTED one (`theCluster`), which is the detail card's alone:
 *
 *   1. the model line: TP, label and engine — the gui node is not the pair's "MoE NVFP4 · TP=2";
 *   2. the queue: its seats and the KV pool behind them — 4 for the node, not the pair's 8;
 *   3. the router's demotion (router.py `busy`): by its own seats;
 *   4. which cards show: a starting second cluster is visible while it is not yet live;
 *   6. the detail card's head: its title is the selected cluster's NAME, and it resets when another is picked;
 *      no wording names a box kind (no "GB10 box") on the hold;
 *   7. the catalog the badges read is every cluster's rows.
 *
 *   node tests/parallel_cards.node.js   (tests/test_parallel_cards_page.py runs it when node exists)
 */
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const src = fs.readFileSync(path.join(__dirname, '..', 'vllm_lb', 'static', 'gateway.js'), 'utf8');
const els = new Map();
const el = () => ({ addEventListener() {}, removeAttribute() {}, setAttribute() {}, append() {},
  classList: { toggle() {}, add() {}, remove() {}, contains: () => false }, style: {}, hidden: false, innerHTML: '',
  textContent: '', value: '', title: '', dataset: {}, querySelectorAll: () => [], querySelector: () => null,
  focus() {}, observe() {}, closest: () => null, getBoundingClientRect: () => ({}) });
const byId = (id) => { if (!els.has(id)) els.set(id, el()); return els.get(id); };
const document = { getElementById: byId, createElementNS: el, createElement: el, addEventListener() {},
  querySelectorAll: () => [], querySelector: () => null, activeElement: null,
  body: { style: {}, classList: { toggle() {} } }, documentElement: { style: {}, classList: { toggle() {} } } };
const window = { addEventListener() {}, innerWidth: 1280, innerHeight: 800, matchMedia: () => ({
  matches: false, addEventListener() {} }), localStorage: { getItem: () => null, setItem() {} } };

const api = new Function('document', 'window', 'fetch', 'location', 'setInterval', 'setTimeout',
  'getComputedStyle', 'ResizeObserver', 'localStorage', 'navigator',
  `var selectedCluster = null; ${src}
  ;return { clusterTrack, targetOf, figureOf, clusterBackend, modelInfoOf, queueOpts, demoted, visibleBackends, renderClusterHead, catalogOf, serverCard, scopeQueue, renderServers,
    select: (n) => { selectedCluster = n; } };`)(
  document, window, () => new Promise(() => {}), { search: '', href: '' }, () => 0, () => 0,
  () => ({}), undefined, window.localStorage, { userAgent: 'node' });

let checks = 0;
const ok = (cond, msg) => { assert.ok(cond, msg); checks += 1; };

const NOW = 1791742000;
const PAIR_TOPO = { tp: 2, boxes: ['pair-a', 'pair-b'], control: 'ssh', why: '',
  steps: ['free', 'reserve', 'drain', 'load', 'warm up', 'serve'] };
const GUI_TOPO = { tp: 1, boxes: ['node-c'], control: 'local', why: '', steps: ['load', 'warm up', 'serve'] };
const pair = { configured: true, name: 'pair', state: 'running', held: true, max_depth: 8, topology: PAIR_TOPO,
  desired: { mode: 'cluster', model_choice: 'm' }, effective: { generation: 9, model_choice: 'm', max_num_seqs: 8,
    max_model_len: 262144 }, switch: '/lb/clusters/pair',
  loaded_model: { served_name: 'M', label: 'Pair Model · MoE NVFP4', max_model_len: 262144 }, boxes: {}, models: [] };
const gui = { configured: true, name: 'gui', state: 'running', held: true, max_depth: 4, topology: GUI_TOPO,
  desired: { mode: 'cluster', model: 'gui-row' }, effective: { max_num_seqs: 4, max_model_len: 32768 },
  switch: '/lb/clusters/gui', applier: { listening: true }, declared: { model_choice: 'gui-row', launch: {} },
  loaded_model: { served_name: 'org/gui-2b', label: null, max_model_len: 32768 }, boxes: {},
  models: [{ id: 'gui-row', served_name: 'org/gui-2b', label: 'GUI agent 2B · VL' }] };
const backend = (url, cluster, served, over) => Object.assign({ kind: 'cluster', url, live: true, cluster, running: 0,
  waiting: 0, inflight: 0, depth_known: true, served_models: [served], max_model_len: 32768, history: {},
  throughput: { source: 'metrics', generated: { tok_s: 0 }, prompt: {}, engine_totals: {} } }, over || {});
const status = (over) => Object.assign({ now: NOW, cluster: { configured: false, clusters: ['pair', 'gui'] },
  clusters: { pair, gui }, backends: [backend('http://192.0.2.10:8100', 'pair', 'M'),
    backend('http://198.51.100.20:8000', 'gui', 'org/gui-2b')], leases: [], hardware: {}, broker: { enabled: false },
  config: {}, models: { cluster: [{ id: 'm', served_name: 'M', label: 'Pair Model · MoE NVFP4' }], classic: [] },
  queue: { backends: {} } }, over || {});

// 1. the model line — selected: the pair (the first), and the gui card still reads its own view
{
  const s = status();
  const g = s.backends[1];
  const info = api.modelInfoOf(s, g);
  ok(info.tensorParallel === 1, `gui is TP=1, not the selected pair's TP=2: ${info.tensorParallel}`);
  ok(info.label === 'GUI agent 2B · VL', `gui's label is its own row's, never the pair's: ${info.label}`);
  ok(api.modelInfoOf(s, s.backends[0]).tensorParallel === 2, 'the pair is TP=2');
  ok(api.catalogOf(s).some((m) => m.id === 'gui-row'), "the badges' catalog carries every cluster's rows");
  api.select('gui');
  ok(api.modelInfoOf(s, s.backends[0]).tensorParallel === 2, 'selecting gui does not change the pair card');
  api.select(null);
}

// 2. the queue's seats
{
  const s = status();
  ok(api.queueOpts(s, s.backends[1]).concurrency === 4, "gui's queue is held to gui's 4 seats");
  ok(api.queueOpts(s, s.backends[0]).concurrency === 8, "the pair's to its 8");
}

// 3. the router's demotion, by each cluster's own seats
{
  const s = status();
  s.backends[1].running = 6;
  ok(api.demoted(s, s.backends[1]) === true, '6 deep on gui (4 seats): demoted');
  s.backends[0].running = 6;
  ok(api.demoted(s, s.backends[0]) === false, '6 deep on the pair (8 seats): not');
}

// 4. a starting second cluster is a card before it answers
{
  const s = status({ clusters: { pair, gui: Object.assign({}, gui, { state: 'starting', held: false }) } });
  s.backends[1].live = false;
  ok(api.visibleBackends(s).some((b) => b.cluster === 'gui'), 'the starting gui head is shown');
  const off = status({ clusters: { pair, gui: Object.assign({}, gui, { state: 'off', held: false }) } });
  off.backends[1].live = false;
  ok(!api.visibleBackends(off).some((b) => b.cluster === 'gui'), 'an off, silent one is not');
}

// 6. the detail card's head: the selected cluster's name, reset on every render
{
  api.renderClusterHead(gui);
  ok(byId('cluster-title').innerHTML === 'gui', `title gui: ${byId('cluster-title').innerHTML}`);
  api.renderClusterHead(pair);
  ok(byId('cluster-title').innerHTML === 'pair', `title back to pair: ${byId('cluster-title').innerHTML}`);
  ok(!/GB10 box/.test(src.split('\n').filter((l) => /toggle\.title|textContent = `/.test(l)).join('\n')),
    'no hold or confirm text names a box kind');
}

// 8. one word for max_num_seqs on the page: "seats" (the queue's), never "seqs" beside it (AUDIT-02 P2)
ok(!/\$\{seqs\} seqs/.test(src), 'the model line says seats, as the queue does');

// 9. EACH SERVER'S OWN FIGURES (/lb/status.clusters[name], 2026-10-20): its throughput target and saturation are read
//    off ITS view (`figureOf`), never the top-level alias and never the selected server's
{
  const s = status({ clusters: { pair: Object.assign({}, pair, { throughput_target: { gen_tok_s: 150 },
    saturation: { verdict: 'pair-sat' } }), gui: Object.assign({}, gui, { throughput_target: { gen_tok_s: 40 },
    saturation: { verdict: 'gui-sat' } }) }, throughput_target: null, saturation: null });
  api.select('pair');
  ok(api.targetOf(s, backend('http://g', 'gui', 'org/gui-2b'), 'gen_tok_s').target === 40, "gui's tile aims at gui's rate");
  ok(api.targetOf(s, backend('http://p', 'pair', 'M'), 'gen_tok_s').target === 150, "the pair's at the pair's");
  ok(api.figureOf(s, gui, 'saturation') === null, 'a view without the figure has none — never another server\'s');
  ok(api.figureOf(s, s.clusters.gui, 'saturation').verdict === 'gui-sat', "gui's saturation is gui's");
  ok(api.figureOf({ saturation: { verdict: 'one' } }, {}, 'saturation').verdict === 'one',
    'a document from before names: the one engine\'s top-level figure');
  ok(!/renderOneClusterOnly|one_cluster_only/.test(src), 'nothing is published empty for being one engine\'s');
  // the Saturation section's tokens (GENERATED, PREFILL / REQUEST) are the SELECTED server's head's (live 2026-10-20:
  // the gui tab showed the pair's 56 tok/s and 2,610 tok/s)
  api.select('gui');
  ok((api.clusterBackend(s) || {}).cluster === 'gui', `gui's tab reads gui's head: ${(api.clusterBackend(s) || {}).cluster}`);
  api.select('pair');
  ok((api.clusterBackend(s) || {}).cluster === 'pair', "the pair's tab, the pair's head");
}

// 11. A LOCAL NODE THAT RESERVES draws reserve and drain as the pair does, off its view (`topology.steps`, its box's
//     `reservation`), and its stop releases the reservation; one that reserves nothing draws neither
{
  const steps = ['reserve', 'drain', 'load', 'warm up', 'serve'];
  const node = (state, reservation, more) => Object.assign({}, gui, { state, topology: Object.assign({}, GUI_TOPO, { steps }),
    boxes: { 'node-c': { reservation: { status: reservation }, running_jobs: 0 } } }, more || {});
  const t = api.clusterTrack(node('reserving', 'queued'));
  ok(t.steps.map((x) => x.label).join() === steps.join() && t.steps[0].status === 'current', `reserving: ${JSON.stringify(t.steps)}`);
  const d = api.clusterTrack(node('draining', 'queued'));
  ok(d.steps[1].status === 'current' && d.steps[0].status === 'done', 'draining: reserve done, drain current');
  const stop = api.clusterTrack(node('releasing', 'running'));
  ok(stop.kind === 'stop' && stop.steps.some((x) => x.label === 'release'), `its stop releases: ${JSON.stringify(stop.steps)}`);
  const plain = api.clusterTrack(Object.assign({}, gui, { state: 'unloading' }));
  ok(!plain.steps.some((x) => x.label === 'release'), 'a node that reserves nothing has nothing to release');
}

// 12. THE REQUESTS CARD IS THE SELECTED SERVER'S (operator, 2026-10-17): its head's rows — live and finished, by their
//     `backend` — its own per-state counts and engine gauges (`queue.backends[url]`), its decode share; HELD turns (sent
//     to no server yet) stay counted; another server's rows never appear. No server in the run: the whole queue.
{
  const P = 'http://192.0.2.10:8100', G = 'http://198.51.100.20:8000';
  const q = { inflight: [{ id: 'a', backend: P, state: 'streaming' }, { id: 'b', backend: G, state: 'queued' }, { id: 'c', backend: P, state: 'prefill' }],
    inflight_total: 3, counts: { streaming: 1, prefill: 1, queued: 1, sent: 0, held: 2, recent: 2 },
    recent: [{ id: 'r1', backend: P, outcome: 'done' }, { id: 'r2', backend: G, outcome: 'failed' }],
    engine: { running: 5, waiting: 1, backends: 2 },
    backends: { [P]: { rows: { streaming: 1, prefill: 1, queued: 0, sent: 0 }, running: 4, waiting: 0, waiting_capacity: 0, waiting_deferred: 0 },
      [G]: { rows: { streaming: 0, prefill: 0, queued: 1, sent: 0 }, running: 1, waiting: 1 } },
    decode: { derived: 'how', engine_tok_s: 50, rows_tok_s: 40, backends: { [P]: { engine_tok_s: 30, rows_tok_s: 25, ended_tok_s: 0, unlisted_tok_s: 0, window_s: 10, why: null },
      [G]: { engine_tok_s: null, why: 'first sample' } } } };
  const sp = api.scopeQueue(q, P);
  ok(sp.inflight.map((r) => r.id).join() === 'a,c' && sp.recent.map((r) => r.id).join() === 'r1', `the pair's rows alone: ${JSON.stringify(sp.inflight)}`);
  ok(sp.counts.streaming === 1 && sp.counts.prefill === 1 && !sp.counts.queued && sp.counts.held === 2, `its counts, held kept: ${JSON.stringify(sp.counts)}`);
  ok(sp.inflight_total === 2 && sp.engine.running === 4 && sp.engine.backends === 1, 'its own total and engine gauges');
  ok(sp.decode.engine_tok_s === 30 && sp.decode.derived === 'how' && sp.decode.unmeasured === 0, 'its decode share, the method kept');
  const sg = api.scopeQueue(q, G);
  ok(sg.inflight.length === 1 && sg.inflight[0].id === 'b' && sg.decode.why === 'first sample' && sg.decode.unmeasured === 1, "gui's: one row, its decode unmeasured with the reason");
  ok(api.scopeQueue(q, null) === q, 'no server: the whole queue');
  // the engine card beside the selected server's card is ITS head's; the other server's head is nowhere else
  const s = status();
  api.select('gui');
  api.renderServers(s);
  const upper = byId('lb-servers').innerHTML;
  ok(/org\/gui-2b/.test(upper) && !/Pair Model/.test(upper), `the upper section is gui's engine card: ${upper.slice(0, 200)}`);
  api.select(null);
}
// 13. NO FALLBACK (operator, 2026-10-17: "If can't run then can't run, no fallback box"): the page draws no fallback
//     server, tile, lease or chip
ok(!/kind === 'ollama'|fallback_model|'proxy'/.test(src), 'no ollama/fallback branch is left in the page');

console.log(`ok — ${checks} checks`);
