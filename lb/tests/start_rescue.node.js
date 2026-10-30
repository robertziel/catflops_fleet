/* A RESCUED START, ON THE PAGE — checked where the page builds it, under node.
 *
 * `cluster.rescue` (cluster_loop.py `_rescue_view`) is the last hang of the start being shown: which attempt
 * hung, in which step, for how long against that step's limit, the retry it is on and where its evidence went.
 * The planner's `detail` already says it in words (it is quoted verbatim); the trail outside the live region
 * carries the same fact as one line while the relaunch is on its way, and says nothing once the pair serves or
 * has failed for good.
 *
 *   node tests/start_rescue.node.js   (tests/test_start_rescue_page.py runs this when node exists)
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
  `${src}\n;return { rescueNote, renderStartup, clusterStateRow };`)(
  document, window, () => new Promise(() => {}), { search: '', href: '' }, () => 0, () => 0,
  () => ({}), undefined, window.localStorage, { userAgent: 'node' });

// /lb/status `cluster` one tick after generation 131's first attempt was torn down and relaunched.
const RESCUE = { attempt: 1, retries: 2, phase: 'capturing CUDA graphs', quiet_s: 182.4, limit_s: 180,
  relaunch: true, evidence: 'start-hangs/gen131-attempt1', at: 1791009060.0, generation: 131 };
const VIEW = { configured: true, state: 'starting', since: 1791009075.0, boxes: {}, loading: {},
  detail: 'launching both ranks (generation 131) — retry 1/2 after the start hung 182 s in “capturing CUDA graphs”',
  rescue: RESCUE };

{
  const html = api.rescueNote(VIEW);
  assert.ok(html.includes('hung in &ldquo;capturing CUDA graphs&rdquo; for 3 min 2 s'), html);
  assert.ok(html.includes('relaunching (retry 1/2)'), html);
  assert.ok(html.includes('the limit for this step is 3 min'), html);
  assert.ok(html.includes('start-hangs/gen131-attempt1'), `where its evidence went: ${html}`);
}
for (const state of ['restarting', 'freeing', 'reserving', 'starting']) {
  assert.ok(api.rescueNote({ ...VIEW, state }).includes('retry 1/2'), `said while ${state}`);
}
for (const state of ['running', 'failed', 'off', 'unloading']) {
  assert.strictEqual(api.rescueNote({ ...VIEW, state }), '', `nothing once it is ${state}`);
}
assert.strictEqual(api.rescueNote({ ...VIEW, rescue: null }), '', 'nothing without a rescue');
assert.strictEqual(api.rescueNote({ ...VIEW, state: 'restarting', rescue: { ...RESCUE, served: true } }), '',
  'a relaunch that has served is history: a later restart of the same generation is not that rescue');
{
  // The trail element shows it even before the relaunched head has printed a step of its own.
  api.renderStartup(VIEW);
  assert.strictEqual(startup.hidden, false, 'the trail element stays hidden');
  assert.ok(startup.innerHTML.includes('relaunching (retry 1/2)'), startup.innerHTML);
  api.renderStartup({ ...VIEW, state: 'running' });
  assert.strictEqual(startup.hidden, true, 'and goes once the pair serves');
}
{
  const row = api.clusterStateRow(VIEW, { now: 1791009080.0 });
  assert.ok(row.includes('retry 1/2 after the start hung 182 s'), 'the planner\'s own words are quoted');
}
console.log('ok');
