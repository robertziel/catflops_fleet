/* RED IS AN ERROR, NEVER A CLOCK — checked where the page paints the cluster, under node.
 *
 * Operator, 2026-10-10: "there must be real error, not timeout to show failed". On the gateway's cluster panel
 * (vllm_lb/static/gateway.js) that means: the `bad` tone, the red step and the FAILED badge belong to `failed`
 * alone — a rank that exited, the memory guard, a refused launch, a reservation the broker ended (cluster.py
 * `plan`). A hung start the LB halted after its relaunches is `stalled`: amber, its own step status, its own
 * badge, and the same "Stop, then Start again" cue, because it is the operator's to lift. A box that could not be
 * observed is `unknown`: amber, with the planner's reason quoted. Nothing here is inferred from a clock.
 *
 * No DOM: gateway.js is loaded with a stub `document` and a `fetch` that never answers.
 *
 *   node tests/cluster_tone.node.js   (tests/test_cluster_tone_page.py runs this when node exists)
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
  `${src}\n;return { clusterTone, clusterTrack, clusterStateRow, clusterStatusOf, startStepOf, boxFacts };`)(
  document, window, () => new Promise(() => {}), { search: '', href: '' }, () => 0, () => 0,
  () => ({}), undefined, window.localStorage, { userAgent: 'node' });

let checks = 0;
const ok = (cond, msg) => { assert.ok(cond, msg); checks++; };
const eq = (a, b, msg) => { assert.strictEqual(a, b, msg); checks++; };

// ── 1. the tone: red is `failed` alone ────────────────────────────────────────────────────────────
eq(api.clusterTone('failed'), 'bad', 'failed is red');
eq(api.clusterTone('running'), 'ok');
eq(api.clusterTone('off'), 'off');
for (const s of ['stalled', 'unknown', 'checking', 'blocked', 'freeing', 'reserving', 'waiting', 'draining',
  'starting', 'restarting', 'finishing', 'unloading', 'releasing', 'some-state-of-tomorrow']) {
  eq(api.clusterTone(s), 'warn', `${s} is amber, never red`);
}

// ── 2. a hung start the LB halted: `stalled`, as the planner publishes it (cluster_loop `_pinned`) ───
// Generation 131's third hang, had it happened: both ranks halted (137), both reservations still running.
const halted = (box) => ({ label: `${box}-gpu`, reservation: { job_id: '0123456789ab', status: 'running', claimed_by: `${box}-gpu` },
  rank: { present: true, running: false, gen: 131, exit_code: 137, model_choice: 'qwen3.8-27b-sglang-dflash2',
    exit_reason: '', lost: false }, running_jobs: 0, mem_available_gib: 104.2, hold: { blockers: [] } });
const STALLED = { configured: true, state: 'stalled', since: 1791009075.0, held: false,
  detail: 'hung in “capturing CUDA graphs” for 190 s (the limit for this step is 180 s) after 2 relaunches — '
    + 'both ranks halted, their logs kept; Stop, then Start again',
  loading: { phase: 'capturing CUDA graphs', passes: [{ name: 'model', completed: 28, total: 28, percent: 100 }],
    steps: ['starting SGLang', 'loading model', 'capturing CUDA graphs'] },
  boxes: { 'box-a': halted('box-a'), 'box-b': halted('box-b') }, desired: { mode: 'cluster', generation: 131 } };
{
  const track = api.clusterTrack(STALLED);
  ok(track && track.kind === 'start', 'a stalled start is still on the start sequence');
  const statuses = track.steps.map((s) => s.status);
  ok(statuses.includes('stalled'), `the step it hung in is marked stalled: ${statuses}`);
  ok(!statuses.includes('failed'), `…and no step is red: ${statuses}`);
  eq(api.startStepOf(STALLED, api.boxFacts(STALLED)), 4, 'the shards were read: it hung warming up');

  const html = api.clusterStateRow(STALLED, { now: 1791009300.0 });
  ok(html.includes('class="diode warn"'), `an amber diode: ${html.slice(0, 200)}`);
  ok(!html.includes('diode bad'), 'no red diode on a stall');
  ok(html.includes('<b class="cp-title">stalled</b>'), 'the state word is the headline');
  ok(html.includes('class="what stalled"'), `the detail is styled as a stall, not a failure: ${html.slice(0, 300)}`);
  ok(!html.includes('what failed'), 'never as a failure');
  ok(html.includes('hung in'), 'the planner\'s own words are quoted');
  ok(html.includes('Stop, then Start again'), 'and the way out is said, as for a failure');
  ok(html.includes('<li class="step stalled">'), 'the strip carries the stalled step');
  ok(!html.includes('step failed'), 'and no failed step');
}

// ── 3. a real failure stays red, exactly as before ────────────────────────────────────────────────
{
  const FAILED = { ...STALLED, state: 'failed',
    detail: 'rank on box-a exited (code 2): sglang serve: error: ambiguous option: --cuda-graph-max-bs; Stop, then Start again' };
  eq(api.clusterTrack(FAILED).steps.filter((s) => s.status === 'failed').length, 1, 'one red step');
  const html = api.clusterStateRow(FAILED, { now: 1791009300.0 });
  ok(html.includes('class="diode bad"') && html.includes('class="what failed"'), 'a failure is red');
  eq(api.clusterStatusOf({ cluster: FAILED }, { live: false }).label, 'FAILED');
  eq(api.clusterStatusOf({ cluster: FAILED }, { live: false }).tone, 'bad');
}

// ── 4. the cluster backend's badge ────────────────────────────────────────────────────────────────
{
  const st = api.clusterStatusOf({ cluster: STALLED }, { live: false });
  eq(st.label, 'STALLED', 'its own word');
  eq(st.tone, 'warn', 'amber');
  for (const state of ['unknown', 'checking', 'blocked', 'freeing']) {
    const s = api.clusterStatusOf({ cluster: { ...STALLED, state } }, { live: false });
    eq(s.tone, 'warn', `${state} is NOT HELD, amber: ${s.label}`);
  }
}

// ── 5. a box that could not be observed: unknown, the reason quoted, nothing red ──────────────────
{
  const UNKNOWN = { configured: true, state: 'unknown', since: 1791009075.0, held: false,
    detail: 'cannot observe box-b (status: exit 255: ssh: connect to host 192.0.2.3 port 22: No route to host)',
    loading: {}, boxes: { 'box-a': halted('box-a'), 'box-b': { label: 'box-b-gpu', rank: null, error: 'status: exit 255: ssh: connect to host 192.0.2.3 port 22: No route to host' } },
    desired: { mode: 'cluster', generation: 186 } };
  const html = api.clusterStateRow(UNKNOWN, { now: 1791009300.0 });
  ok(html.includes('class="diode warn"') && !html.includes('bad'), `unknown is amber: ${html.slice(0, 200)}`);
  ok(html.includes('No route to host'), 'the executor\'s reason reaches the page');
  eq(api.clusterTrack(UNKNOWN), null, 'no sequence claims a step it cannot see');
}

// ── 6. a start waiting on a share that is not mounted yet: checking, amber, no "choose another model" ──
{
  const CHECKING = { configured: true, state: 'checking', since: 1791009075.0, held: false,
    detail: 'waiting for box-b to see Qwen3.8-Flash-Next-hibrid48_uncensored: /mnt/box-a-home/models/'
      + 'qwen38-flash-next-cluster is not mounted: NFS /mnt/box-a-home (192.0.2.201:/) is not up on this box yet',
    loading: {}, boxes: {}, desired: { mode: 'cluster', generation: 186 } };
  const html = api.clusterStateRow(CHECKING, { now: 1791009300.0 });
  ok(html.includes('class="diode warn"') && !html.includes('bad'), 'amber');
  ok(html.includes('not mounted') && !html.includes('choose another model'), 'the wait is said; no other model is asked for');
  eq(api.clusterTrack(CHECKING).current, 0, 'the first step of a start');
}

console.log(`ok — ${checks} checks`);
