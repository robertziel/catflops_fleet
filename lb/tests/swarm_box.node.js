/* WHERE THE SWARM RUNNER IS, IN THE PAGE'S WORDS — from the runner's own report, never a literal box.
 *
 * The same gateway page serves every deployment (AGENTS.md, "open-source ready"). Until 2026-10-11 the
 * swarm control said "no runner listening on box-c" on a gateway whose runner, had it existed, would
 * have been on another box entirely. The runner now reports the box its configuration names (`--box`,
 * runner.json `box`), and the page names THAT — or no box at all when nothing has reported.
 *
 * And the sealed runner (lb/bench_runner) has one refusal of its own: its egress check failed before a
 * launch, so claude was never started. That is `unsealed`, drawn as a failure, in its own words.
 *
 * No DOM: gateway.js is loaded with a stub `document` and a `fetch` that never answers.
 *
 *   node tests/swarm_box.node.js   (tests/test_swarm_box_page.py runs this when node exists)
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
const document = { getElementById: el, createElementNS: el, createElement: el, addEventListener() {},
  querySelectorAll: () => [], querySelector: () => null, body: { style: {}, classList: { toggle() {} } },
  documentElement: { style: {}, classList: { toggle() {} } } };
const window = { addEventListener() {}, innerWidth: 1280, innerHeight: 800, matchMedia: () => ({
  matches: false, addEventListener() {} }), localStorage: { getItem: () => null, setItem() {} } };

const api = new Function('document', 'window', 'fetch', 'location', 'setInterval', 'setTimeout',
  'getComputedStyle', 'ResizeObserver', 'localStorage', 'navigator',
  `${src}\n;return { swStateLines, swWhyNot, SWARM_RESULT };`)(
  document, window, () => new Promise(() => {}), { search: '', href: '' }, () => 0, () => 0,
  () => ({}), undefined, window.localStorage, { userAgent: 'node' });

const NOW = 1791743469.0;
const BOX = 'box-under-test';

// ── nothing has ever reported: no box to name, so none is named ─────────────────────────────────
{
  const never = { runner: { listening: false, at: null, box: null,
    error: 'no bench runner has ever reported at /data/benchq/runner.json' } };
  assert.strictEqual(api.swWhyNot(never), 'no runner listening');
  const html = api.swStateLines(never, NOW);
  assert.ok(!/box-c/i.test(html), html);
}

// ── a runner that reported its box, then went quiet: that box, by name ──────────────────────────
{
  const quiet = { runner: { listening: false, at: NOW - 300, box: BOX,
    error: `the bench runner on ${BOX} last reported 300s ago — it is not listening for a request` } };
  assert.strictEqual(api.swWhyNot(quiet), `no runner listening on ${BOX}`);
}

// ── the matrix runs where the runner is, and the line says so ───────────────────────────────────
{
  const matrix = { runner: { listening: true, at: NOW, box: BOX, session: { alive: false },
    matrix: { running: true, pid: 900102, since: NOW - 60, script: 'fullclaudevllm_queue.sh' } } };
  assert.strictEqual(api.swWhyNot(matrix), `matrix running on ${BOX}`);
  const html = api.swStateLines(matrix, NOW);
  assert.ok(html.includes(`matrix up on ${BOX}`), html);
  assert.ok(!/box-c/i.test(html), html);
  const nameless = { runner: { ...matrix.runner, box: null } };
  assert.strictEqual(api.swWhyNot(nameless), 'matrix running');
  assert.ok(api.swStateLines(nameless, NOW).includes('matrix up'), 'the matrix is still said to be up');
}

// ── the sealed runner refused a launch: a failure, in the check's own words ─────────────────────
{
  assert.ok(api.SWARM_RESULT.unsealed, 'the page has a word for an unsealed runner');
  assert.strictEqual(api.SWARM_RESULT.unsealed[0], 'bad');
  const refused = { runner: { listening: true, at: NOW, box: BOX, session: { alive: false },
    matrix: { running: false }, last: { id: 'abc', at: NOW - 5, result: 'unsealed',
      detail: 'the seal check failed (exit 1), so claude was not started: api.anthropic.com:443 direct CONNECTED' } } };
  const html = api.swStateLines(refused, NOW);
  assert.ok(html.includes('sw-bad'), html);
  assert.ok(html.includes('egress not sealed'), html);
  assert.ok(html.includes('so claude was not started'), 'the check\'s own sentence is the title');
}

// ── and nowhere in the swarm control is a box written down ──────────────────────────────────────
{
  const section = src.slice(src.indexOf('let swarmBusy'), src.indexOf('function swarmHeld'));
  assert.ok(section.length > 1000, 'the swarm section was found');
  assert.ok(!/box-c/i.test(section), 'a box name is configuration, never a word in the page');
}

console.log('swarm_box: ok');
