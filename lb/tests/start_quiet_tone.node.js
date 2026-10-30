/* A QUIET START IS A TIMER, NOT AN ERROR — checked where the page draws it, under node.
 *
 * Generation 188 (2026-10-10): six minutes into a healthy weight load the trail painted "initializing model" and
 * "no new step for 6 min 12 s" RED, while nothing had failed. Red is for an error (the operator's rule, a8b495e:
 * `failed` alone is red; a halted hang is `stalled`, amber). Past the LB's own thresholds the step and the note turn
 * amber, then bold amber with the spinner stopped — never red, in the markup or in gateway.css.
 *
 * And the GiB the head's box has given up during a load (`loading.taken_gib`, cluster_loop.py `_track`) is part of
 * what the page watches: a load whose bar docker cannot show yet (one `\r` entry until its newline) still moves the
 * page's clock when memory moves, and the figure is on the row.
 *
 *   node tests/start_quiet_tone.node.js   (tests/test_start_quiet_tone_page.py runs this when node exists)
 */
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const STATIC = path.join(__dirname, '..', 'vllm_lb', 'static');
const src = fs.readFileSync(path.join(STATIC, 'gateway.js'), 'utf8');
const el = () => ({ addEventListener() {}, removeAttribute() {}, setAttribute() {}, append() {},
  classList: { toggle() {}, add() {}, remove() {} }, style: {}, hidden: false, innerHTML: '',
  textContent: '', value: '', dataset: {}, querySelectorAll: () => [], querySelector: () => null,
  focus() {}, observe() {}, closest: () => null, getBoundingClientRect: () => ({}) });
const document = { getElementById: el, createElementNS: el, createElement: el, addEventListener() {},
  querySelectorAll: () => [], querySelector: () => null,
  body: { style: {}, classList: { toggle() {} } }, documentElement: { style: {}, classList: { toggle() {} } } };
const window = { addEventListener() {}, innerWidth: 1280, innerHeight: 800, matchMedia: () => ({
  matches: false, addEventListener() {} }), localStorage: { getItem: () => null, setItem() {} } };

const api = new Function('document', 'window', 'fetch', 'location', 'setInterval', 'setTimeout',
  'getComputedStyle', 'ResizeObserver', 'localStorage', 'navigator',
  `${src}\n;return { watch: (c, s) => { startupNow = watchStartup(c, s); return startupNow; }, startupTrail,
     startupProgress, STARTUP_SLOW_S };`)(
  document, window, () => new Promise(() => {}), { search: '', href: '' }, () => 0, () => 0,
  () => ({}), undefined, window.localStorage, { userAgent: 'node' });

const T0 = 1791666300.0;                                   // 21:05:00Z — the LB's clock when the load opened
const STEPS = ['starting SGLang', 'connecting the ranks (NCCL)', 'initializing model', 'loading model weights'];
const view = (taken, steps = STEPS) => ({ configured: true, state: 'starting', since: T0 - 40, boxes: {},
  loading: { phase: steps[steps.length - 1], steps, ...(taken == null ? {} : { taken_gib: taken }) } });

// The page watched "loading model weights" arrive, then nothing new in the log for 374 s (gen 188 at 21:10).
api.watch(view(null, STEPS.slice(0, 3)), { now: T0 - 5 });
api.watch(view(12), { now: T0 });
let w = api.watch(view(12), { now: T0 + 374 });
assert.strictEqual(w.quiet, 374);
let trail = api.startupTrail();
assert.ok(trail.includes('no new step for 6 min 14 s'), trail);
assert.ok(!/stuck|bad/.test(trail), `a timer is never red: ${trail}`);
assert.ok(/class="trail-step now long"/.test(trail), `past five minutes: bold amber, not red — ${trail}`);
assert.ok(/class="trail-note long"/.test(trail), trail);
assert.ok(trail.includes('12 GiB taken'), `the memory the load has taken is on the row: ${trail}`);
let head = api.startupProgress(view(12));
assert.ok(head.includes('class="startup-phase long"'), head);
assert.ok(!/stuck|bad/.test(head), head);

// The box gave up two more GiB: progress. The clock starts over, the note goes, the figure moves.
w = api.watch(view(14), { now: T0 + 389 });
assert.strictEqual(w.quiet, 0, 'memory moving during a load resets the page\'s clock');
trail = api.startupTrail();
assert.ok(!trail.includes('no new step'), trail);
assert.ok(/class="trail-step now"/.test(trail) && trail.includes('14 GiB taken'), trail);

// Between the two thresholds: amber, as before. The figure stays as long as the step is a load's.
api.watch(view(14), { now: T0 + 389 + 150 });
trail = api.startupTrail();
assert.ok(/class="trail-step now slow"/.test(trail) && /class="trail-note slow"/.test(trail), trail);
assert.ok(trail.includes('14 GiB taken'), trail);

// A step that is not a load carries no figure.
api.watch(view(null, [...STEPS, 'model weights loaded']), { now: T0 + 600 });
trail = api.startupTrail();
assert.ok(!trail.includes('GiB taken'), trail);

// Nothing in the stylesheet paints a start's clock red either.
const css = fs.readFileSync(path.join(STATIC, 'gateway.css'), 'utf8');
let checked = 0;
for (const rule of css.split('}')) {
  const open = rule.indexOf('{');
  if (open < 0) continue;
  const selector = rule.slice(0, open), body = rule.slice(open + 1);
  if (!/\.(startup-phase|trail-step|trail-note|trail-taken)/.test(selector)) continue;
  checked += 1;
  assert.ok(!/--bad/.test(body), `red on a clock: ${selector.trim()} { ${body.trim()} }`);
}
assert.ok(checked >= 8, `the trail's rules were read: ${checked}`);
assert.ok(!/\.stuck\b/.test(css), 'the word is gone from the stylesheet too: a page cannot know a start is stuck');
console.log('ok');
