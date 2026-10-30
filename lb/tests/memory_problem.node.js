/* NOT ENOUGH RAM, SAID BEFORE THE START — operator 2026-10-26, after generation 249 was stored, reserved both GB10 boxes and
 * only then refused ("box-b: 111.0 GiB free is the most it frees at rest; need 115.1"): "if not enough ram tell before
 * I hit start". `memoryProblem(estimate.boxes)` is the launch gate's question asked of the selection on screen: a box
 * AT REST (the LB publishes `at_rest`) whose free memory is under `needed_now_gib`. selectionProblem returns it, so the
 * Start is held back with the figures.
 *
 *   node tests/memory_problem.node.js   (tests/test_memory_problem_page.py runs it when node exists)
 */
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const src = fs.readFileSync(path.join(__dirname, '..', 'vllm_lb', 'static', 'gateway.js'), 'utf8');
const el = () => ({ addEventListener() {}, removeAttribute() {}, setAttribute() {}, append() {},
  classList: { toggle() {}, add() {}, remove() {}, contains: () => false }, style: {}, dataset: {},
  querySelectorAll: () => [], querySelector: () => null, closest: () => null });
const document = { getElementById: el, createElementNS: el, createElement: el, addEventListener() {},
  querySelectorAll: () => [], querySelector: () => null,
  body: { style: {}, classList: { toggle() {} } }, documentElement: { style: {}, classList: { toggle() {} } } };
const window = { addEventListener() {}, innerWidth: 1280, innerHeight: 800, matchMedia: () => ({
  matches: false, addEventListener() {} }), localStorage: { getItem: () => null, setItem() {} } };
const { memoryProblem } = new Function('document', 'window', 'fetch', 'location', 'setInterval', 'setTimeout',
  'getComputedStyle', 'ResizeObserver', 'localStorage', 'navigator', `${src}\n;return { memoryProblem };`)(
  document, window, () => new Promise(() => {}), { search: '', href: '' }, () => 0, () => 0,
  () => ({}), undefined, window.localStorage, { userAgent: 'node' });
let checks = 0;

// Generation 249's boxes: box-a fits, box-b at rest is 4.0 short.
const box = (name, free, atRest) => ({ box: name, available_gib: free, needed_now_gib: 115.1, total_gib: 107.1,
  start_headroom_gib: 8.0, at_rest: atRest });
assert.strictEqual(memoryProblem([box('box-a', 115.5, true), box('box-b', 111.1, true)]),
  'not enough RAM: box-b 111.1 GiB free at rest, needs 115.1 (107.1 + 8.0 reserve): short 4.0 GiB'
  + ' — lower the KV pool, prefill budget, concurrency or context'); checks++;
// Not at rest (a fleet job, ollama, its rank standing): the loop waits for it — no refusal here.
assert.strictEqual(memoryProblem([box('box-b', 50.0, false)]), ''); checks++;
// An LB older than `at_rest`, or no estimate: nothing said.
assert.strictEqual(memoryProblem([{ ...box('box-b', 50.0, true), at_rest: undefined }]), ''); checks++;
assert.strictEqual(memoryProblem(null), ''); checks++;
// Fits everywhere.
assert.strictEqual(memoryProblem([box('box-a', 120, true), box('box-b', 116, true)]), ''); checks++;
// Both short: both named.
assert.match(memoryProblem([box('box-a', 100, true), box('box-b', 90, true)]), /box-a 100\.0 .*; box-b 90\.0 /); checks++;

console.log(`memory_problem: all ok (${checks} checks)`);
