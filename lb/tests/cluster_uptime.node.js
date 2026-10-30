/* HOW LONG THE PAIR HAS BEEN UP, in the stats row beside the engine's totals — checked where the page builds it.
 *
 * `cluster.up_since` (cluster_loop.py `_up_since`): docker's StartedAt of the head's rank on the LB's clock, null
 * unless the pair serves. The row says "up 1h 23m"; a backend without it says nothing rather than a guess.
 *
 *   node tests/cluster_uptime.node.js   (tests/test_cluster_uptime_page.py runs this when node exists)
 */
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const src = fs.readFileSync(path.join(__dirname, '..', 'vllm_lb', 'static', 'gateway.js'), 'utf8');
const el = () => ({ addEventListener() {}, removeAttribute() {}, setAttribute() {}, append() {},
  classList: { toggle() {}, add() {}, remove() {} }, style: {}, hidden: false, innerHTML: '',
  textContent: '', value: '', dataset: {}, querySelectorAll: () => [], querySelector: () => null,
  focus() {}, observe() {}, closest: () => null, getBoundingClientRect: () => ({}) });
const startup = el();
const document = { getElementById: (id) => (id === 'cluster-startup' ? startup : el()), createElementNS: el,
  createElement: el, addEventListener() {}, querySelectorAll: () => [], querySelector: () => null,
  body: { style: {}, classList: { toggle() {} } }, documentElement: { style: {}, classList: { toggle() {} } } };
const window = { addEventListener() {}, innerWidth: 1280, innerHeight: 800, matchMedia: () => ({
  matches: false, addEventListener() {} }), localStorage: { getItem: () => null, setItem() {} } };

const api = new Function('document', 'window', 'fetch', 'location', 'setInterval', 'setTimeout',
  'getComputedStyle', 'ResizeObserver', 'localStorage', 'navigator',
  `${src}\n;return { engineStats };`)(
  document, window, () => new Promise(() => {}), { search: '', href: '' }, () => 0, () => 0,
  () => ({}), undefined, window.localStorage, { userAgent: 'node' });

// The live row's shape (2026-10-03): a cluster backend with the engine's counters and its totals since it loaded.
const B = { kind: 'cluster', live: true, running: 30, waiting: 3, depth_known: true,
  throughput: { source: 'metrics', engine_totals: { generated: 137900, prompt: 10.9e6, computed: 2.3e6 } } };
const NOW = 1791016000;

{
  const html = api.engineStats(B, true, {}, 'dflash', NOW - 5025, NOW);
  assert.ok(/up<\/[^>]*>\s*<[^>]*>1h 23m/.test(html) || html.includes('1h 23m'), `the uptime is drawn: ${html}`);
  assert.ok(html.indexOf('1h 23m') > html.indexOf('prefill'), 'after the totals since the engine loaded');
}
assert.ok(!api.engineStats(B, true, {}, 'dflash', null, NOW).includes(' up'), 'no start time, no uptime');
assert.ok(!api.engineStats(B, true, {}, 'dflash').includes('1h'), 'a caller that passes none draws none');
console.log('ok');
