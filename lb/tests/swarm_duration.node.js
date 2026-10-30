/* THE DURATION THE OPERATOR PICKS, checked where the page actually builds it — under node.
 *
 * The swarm's cap was a fixed quarter of an hour until 2026-10-05, in the LB, in the host runner and in
 * this page's label ("Test swarm running · 0.3 of 15 min"). The server halves are pinned by
 * tests/test_bench_swarm.py and tests/test_bench_runner.py; what cannot be seen from Python is the part
 * the browser computes, and there are exactly two claims about it worth holding:
 *
 *  1. THE LABEL NAMES THE CHOSEN DURATION, not the default and not the last session's. Idle it is the
 *     chooser's value (`swChosenCap`), running it is the cap THAT session was given (`swCap`, which
 *     reads the host runner's own report) — two different questions, because a 120-minute run does not
 *     become a quarter-hour one when the list is put back to 15 under it.
 *  2. THE TRACK ALONG THE BUTTON SCALES TO IT. A bar that filled in fifteen minutes and then sat there
 *     would say nothing at all about the three-quarters of a two-hour run still to go.
 *
 * And the refusal: a number outside what /lb/status publishes greys the button BEFORE the three-second
 * hold, because three seconds of holding must never end in "that number is out of range".
 *
 * No DOM: gateway.js is loaded with a stub `document` whose getElementById returns ONE node per id, so
 * the chooser keeps the value it was given and the button's label can be read back.
 *
 *   node tests/swarm_duration.node.js   (tests/test_swarm_duration.py runs this when node exists)
 */
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const src = fs.readFileSync(path.join(__dirname, '..', 'vllm_lb', 'static', 'gateway.js'), 'utf8');

const track = { style: {} };
const nodes = new Map();
function node(id) {
  const n = { id, addEventListener() {}, removeAttribute() {}, setAttribute() {}, append() {},
    classList: { toggle() {}, add() {}, remove() {} }, style: {}, hidden: false, innerHTML: '',
    textContent: '', value: '', dataset: {}, disabled: false, querySelectorAll: () => [],
    querySelector: (sel) => (sel === '.hold-track i' ? track : node(`${id}${sel}`)),
    focus() {}, observe() {}, closest: () => null, getBoundingClientRect: () => ({}) };
  if (id === 'swarm-cap') n.tagName = 'SELECT';
  return n;
}
const el = (id) => {
  if (!nodes.has(id)) nodes.set(id, node(id));
  return nodes.get(id);
};
const document = { getElementById: el, createElementNS: el, createElement: el, addEventListener() {},
  querySelectorAll: () => [], querySelector: () => null, body: { style: {}, classList: { toggle() {} } },
  documentElement: { style: {}, classList: { toggle() {} } } };
const window = { addEventListener() {}, innerWidth: 1280, innerHeight: 800, matchMedia: () => ({
  matches: false, addEventListener() {} }), localStorage: { getItem: () => null, setItem() {} } };

const api = new Function('document', 'window', 'fetch', 'location', 'setInterval', 'setTimeout',
  'getComputedStyle', 'ResizeObserver', 'localStorage', 'navigator',
  `${src}\n;return { renderSwarmCap, renderSwarmControls, swCap, swChosenCap, swCapRange, swDefaultCap,
                     swCapProblem, swProgress, swCanStart, fmtCap, CAP_STEPS };`)(
  document, window, () => new Promise(() => {}), { search: '', href: '' }, () => 0, () => 0,
  () => ({}), undefined, window.localStorage, { userAgent: 'node' });

const NOW = 1_791_200_000;
/** What /lb/status carries: the runner's own default, the range it accepts, and a session or none. */
const view = (session = null) => ({
  configured: true, cap_min: 15, cap_min_range: [5, 240], model_loaded: true,
  runner: { listening: true, cap_min: 15, session, matrix: null, last: null },
});
const label = () => el('swarm-start').dataset.label;
const fill = () => Number(/scaleX\(([\d.]+)\)/.exec(track.style.transform || 'scaleX(0)')[1]);
const draw = (q, now = NOW) => {
  api.renderSwarmCap(q, (q.runner.session || {}).alive === true);
  api.renderSwarmControls(q, now);
};

/* ── the range comes from the LB, never from this file ──────────────────────── */
assert.deepStrictEqual(api.swCapRange(view()), [5, 240]);
assert.deepStrictEqual(api.swCapRange({}), [5, 240], 'the shipped pair, before the first poll lands');
assert.deepStrictEqual(api.swCapRange({ cap_min_range: [10, 60] }), [10, 60],
  'an LB with another range must be obeyed, not second-guessed');
assert.ok(api.CAP_STEPS.includes(15) && api.CAP_STEPS.includes(30),
  'the operator asked for 30 beside the 15, not instead of it');

/* ── 1. the default is still fifteen minutes ────────────────────────────────── */
draw(view());
assert.strictEqual(api.swDefaultCap(view()), 15);
assert.strictEqual(api.swChosenCap(view()), 15, 'an untouched chooser is the runner\'s own default');
assert.strictEqual(label(), 'Start test swarm · 15 min');
assert.ok(/>15 · default</.test(el('swarm-cap').innerHTML), 'the list marks which one is the default');
assert.ok(/value="30"/.test(el('swarm-cap').innerHTML) && /value="240"/.test(el('swarm-cap').innerHTML),
  'the longer stretches are on the list');
assert.strictEqual(fill(), 0, 'nothing is running, so the track is the hold\'s and empty');

/* ── the chosen duration, in the label ──────────────────────────────────────── */
el('swarm-cap').value = '30';
draw(view());
assert.strictEqual(api.swChosenCap(view()), 30);
assert.strictEqual(label(), 'Start test swarm · 30 min', 'the button names what a hold would ask for');
el('swarm-cap').value = '240';
draw(view());
assert.strictEqual(label(), 'Start test swarm · 240 min');

/* ── 2. the running label and the bar are the SESSION's cap, and they scale ── */
const running = (cap, elapsedMin) => view({ alive: true, cap_min: cap, label: 'S-1',
  started_at: NOW - elapsedMin * 60 });
el('swarm-cap').value = '15';                      // the chooser is irrelevant while one runs
draw(running(30, 7));
assert.strictEqual(api.swCap(running(30, 7)), 30, 'the cap of the run on screen, not the default');
assert.strictEqual(api.swProgress(running(30, 7), NOW), '7.0 of 30 min');
assert.strictEqual(label(), 'Test swarm running · 7.0 of 30 min');
assert.ok(Math.abs(fill() - 7 / 30) < 1e-6, `the bar is 7 of 30 minutes, got ${fill()}`);

draw(running(120, 30));
assert.strictEqual(label(), 'Test swarm running · 30 of 120 min');
assert.ok(Math.abs(fill() - 0.25) < 1e-6, `a quarter of two hours, got ${fill()}`);
draw(running(15, 7));
assert.ok(Math.abs(fill() - 7 / 15) < 1e-6, 'and the same elapsed fills further in a shorter cap');
draw(running(30, 90));
assert.strictEqual(fill(), 1, 'a run past its own cap pins the bar, it never overflows it');
draw(view());
assert.strictEqual(fill(), 0, 'and the bar is given back to the hold the moment none is running');

/* ── the refusal, before the hold ───────────────────────────────────────────── */
el('swarm-cap').value = '241';
draw(view());
assert.ok(/from 5 to 240/.test(api.swCapProblem(view())), api.swCapProblem(view()));
assert.strictEqual(api.swCanStart(view()), false, '241 minutes must not be holdable');
assert.strictEqual(el('swarm-start').disabled, true);
assert.ok(/from 5 to 240/.test(el('swarm-why').textContent), 'and the reason is under the button');
el('swarm-cap').value = 'custom';                  // picked custom… and typed nothing yet
draw(view());
assert.strictEqual(api.swCanStart(view()), false, 'an empty custom field is not a duration');
el('swarm-cap-custom').value = '45';
draw(view());
assert.strictEqual(api.swCapProblem(view()), '');
assert.strictEqual(api.swChosenCap(view()), 45, 'a typed duration is the one that is sent');
assert.strictEqual(label(), 'Start test swarm · 45 min');

/* ── and the formatting of the one number ───────────────────────────────────── */
assert.strictEqual(api.fmtCap(15), '15');
assert.strictEqual(api.fmtCap(17.5), '17.5');

console.log('ok — the label and the track are the chosen duration, and a value out of range is refused');
