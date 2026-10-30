/* THE PAGE'S DEMOTION READS A CLUSTER'S OWN SEATS, as router.py `busy` does (tests/test_cluster_seats_routing.py):
 * `cluster.max_depth` — the pair's concurrency, a node's declared seats — and never `config.vllm_max_depth`, the
 * classic boxes' limit. A cluster whose seats nobody declared has no limit to be within: DEMOTED, as the router
 * treats it. (The page draws no classic box's card: Other servers is retired, 2026-10-17.)
 *
 *   node tests/cluster_seats.node.js   (tests/test_cluster_seats_page.py runs it)
 */
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const src = fs.readFileSync(path.join(__dirname, '..', 'vllm_lb', 'static', 'gateway.js'), 'utf8');
const el = () => ({ addEventListener() {}, removeAttribute() {}, setAttribute() {}, append() {},
  classList: { toggle() {}, add() {}, remove() {}, contains: () => false }, style: {}, hidden: false, innerHTML: '',
  textContent: '', value: '', dataset: {}, querySelectorAll: () => [], querySelector: () => null,
  focus() {}, observe() {}, closest: () => null, contains: () => false, getBoundingClientRect: () => ({}) });
const els = new Map();
const byId = (id) => { if (!els.has(id)) els.set(id, el()); return els.get(id); };
const document = { getElementById: byId, createElementNS: el, createElement: el, addEventListener() {},
  querySelectorAll: () => [], querySelector: () => null, activeElement: null,
  body: { style: {}, classList: { toggle() {} } }, documentElement: { style: {}, classList: { toggle() {} } } };
const window = { addEventListener() {}, innerWidth: 1280, innerHeight: 800, matchMedia: () => ({
  matches: false, addEventListener() {} }), localStorage: { getItem: () => null, setItem() {} } };
const api = new Function('document', 'window', 'fetch', 'location', 'setInterval', 'setTimeout',
  'getComputedStyle', 'ResizeObserver', 'localStorage', 'navigator',
  `${src}\n;return { statusOf };`)(
  document, window, () => new Promise(() => {}), { search: '', href: '' }, () => 0, () => 0,
  () => ({}), undefined, window.localStorage, { userAgent: 'node' });

const server = (kind, running) => ({ kind, url: `http://${kind}:8000`, live: true, depth_known: true, running,
  waiting: 0, inflight: 0 });
const status = (maxDepth) => ({ config: { vllm_max_depth: 2 }, broker: { enabled: false }, hardware: {},
  cluster: { configured: true, held: true, state: 'running', max_depth: maxDepth } });
const label = (s, b) => api.statusOf(s, b).label;
let checks = 0;

assert.strictEqual(label(status(4), server('cluster', 4)), 'SERVING', '4 of its 4 seats: the classic limit (2) plays no part');
assert.strictEqual(label(status(4), server('cluster', 5)), 'DEMOTED', 'past its seats');
assert.strictEqual(label(status(16), server('cluster', 16)), 'SERVING', 'the pair at its concurrency');
assert.strictEqual(label(status(null), server('cluster', 0)), 'DEMOTED', 'no seats declared: no limit to be within');
checks += 4;

console.log(`ok — ${checks} checks`);
