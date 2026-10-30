/* The server card's QUEUE tile, checked where it is built: gateway.js, under node.
 *
 * THE COMPLAINT THIS PINS (operator, 2026-09-30): a tile headed QUEUE DEPTH read 25 directly above
 * "0 waiting · 6 running · 25 ours". A headline cannot contradict its own breakdown, so the rule
 * asserted here is the one a reader applies without being told: EVERY NUMBER PRINTED FOR THE QUEUE IS
 * A PART OF ITS HEADLINE, AND THE PARTS ADD TO IT.
 *
 * And the parts are the ENGINE's OWN TWO: running and waiting. Naming the leftover `inflight` as a
 * third part was rejected in turn — "there's no ours that is not running nor waiting" — so this
 * file also pins that no third quantity, under any wording, returns to the card.
 *
 * WHERE THE PARTS ARE (2026-10-01, "only data matters"): the tile heads the sum over the concurrency
 * and prints NOTHING under its graph — the breakdown is the RUN + WAIT figure of the engine's stats
 * under the tiles (`engineStats`), "16 + 0". A sub-line that restated the headline in words was noise.
 * So the sum is asserted against THAT figure, the tile is asserted empty under its graph, and a figure
 * nobody measured stays "—" in both places.
 *
 * NOTHING HERE IS ABOUT ROUTING. The router's depth — `depthOf`, mirroring router.py `depth` — is
 * `waiting + max(running, inflight)` deliberately: the probe is up to `probe_interval_s` stale, and
 * without our own inflight every turn arriving inside one interval reads the same pre-admission
 * depth and piles onto one box. That quantity is asserted UNCHANGED, so a display fix can never
 * quietly move the router.
 *
 * No DOM: gateway.js is loaded with a stub `document` and a `fetch` that never answers, so `poll()`
 * waits forever and nothing is drawn.
 *
 *   node tests/gateway_depth.node.js      (tests/test_gateway_depth.py runs this when node exists)
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
  `${src}\n;return { depthOf, queueTile, engineStats };`)(
  document, window, () => new Promise(() => {}), { search: '', href: '' }, () => 0, () => 0,
  () => ({}), undefined, window.localStorage, { userAgent: 'node' });

// ── reading the rendered tile and the stats the way a person reads them ──────
const strip = (h) => h.replace(/<[^>]*>/g, '').replace(/&middot;/g, '·')
  .replace(/&nbsp;/g, ' ').replace(/&amp;/g, '&').trim();
const headline = (html) => strip((html.match(/<b class="t-val"[^>]*>([\s\S]*?)<\/b>/) || [, ''])[1]);
const subline = (html) => strip((html.match(/class="t-sub">([\s\S]*)<\/span>/) || [, ''])[1]);
/** The RUN + WAIT figure of the engine's stats: "16 + 0", or "—". */
const runWait = (html) => strip((html.match(/<span>run \+ wait<\/span><b>([\s\S]*?)<\/b>/) || [, ''])[1]);
// THE PAGE'S GRAMMAR (gateway.js `tile`): " + " joins the parts of a headline. The RUN + WAIT figure is
// exactly those parts, and they must add up to the queue's headline.
const partsOf = (figure) => (figure.match(/\d+/g) || []).map(Number);
const sum = (a) => a.reduce((x, y) => x + y, 0);

const backend = (waiting, running, inflight, over) => Object.assign(
  { kind: 'vllm', waiting, running, inflight, depth_known: true }, over || {});

let checks = 0;

// 1. THE COMPLAINT ITSELF, and every shape the three counters take.
//    A vLLM's `running`/`waiting` are the ENGINE's own counters and the only ones printed; `inflight`
//    — what this LB forwarded — stays off the page. Whatever the card shows, its printed parts (the
//    RUN + WAIT figure) must add to the queue tile's headline.
{
  const cases = [
    ['the operator’s card', backend(0, 6, 25)],
    ['the gap measured live', backend(2, 14, 35)],
    ['engine busier than our share', backend(2, 14, 3)],
    ['idle', backend(0, 0, 0)],
    ['one of ours, nothing admitted yet', backend(0, 0, 1)],
  ];
  for (const [what, b] of cases) {
    const html = api.queueTile(b, { depth: [] });
    const n = Number(headline(html));
    assert.ok(Number.isFinite(n), `${what}: headline is not a number: ${headline(html)}`);
    const figure = runWait(api.engineStats(b, true));
    assert.strictEqual(sum(partsOf(figure)), n,
      `${what}: headline ${n} but RUN + WAIT ${JSON.stringify(partsOf(figure))} adds to ${sum(partsOf(figure))} — "${figure}"`);
    checks += 2;
  }
}

// 2. NO THIRD NUMBER, AND NOTHING UNDER THE GRAPH. `inflight` — what this LB forwarded — is not
//    printed anywhere on the card, under any name. A request is running or it is waiting; the
//    operator rejected every variant of a third count ("there's no ours that is not running nor
//    waiting", 2026-09-30), so the card carries the ENGINE's two counters — as the one RUN + WAIT
//    figure — and the headline they add to. The tile prints no sub-line at all: "16 running + 0
//    waiting" under 16/16 restated the headline (2026-10-01).
{
  for (const b of [backend(0, 6, 25), backend(2, 14, 35), backend(2, 14, 3), backend(0, 0, 1)]) {
    const tile = api.queueTile(b, { depth: [] });
    const stats = api.engineStats(b, true);
    assert.ok(!/ours|admitted|in flight|inflight/i.test(strip(tile) + strip(stats)),
      `a third quantity crept back: "${strip(tile)}" / "${strip(stats)}"`);
    assert.strictEqual(subline(tile), '', `the queue tile prints a sub-line again: "${subline(tile)}"`);
    const figure = runWait(stats);
    assert.deepStrictEqual(partsOf(figure), [b.running, b.waiting],
      `RUN + WAIT is running then waiting, nothing else: "${figure}"`);
    checks += 3;
  }
  // The headline is the engine's two counters summed — never the router's pessimistic depth.
  assert.strictEqual(headline(api.queueTile(backend(0, 6, 25), { depth: [] })), '6');
  assert.strictEqual(headline(api.queueTile(backend(2, 14, 35), { depth: [] })), '16');
  checks += 2;
}

// 3. AN ENGINE THAT DID NOT ANSWER IS NOT AN IDLE ENGINE. probe.py sets `waiting: 0, running: 0,
//    depth_known: False` when /metrics fails — placeholder zeros, not readings. Printing "0 running"
//    from them is the page inventing a measurement, which is the `depth_known` lesson already learnt
//    once on this fleet. Both places must decline to give a number instead.
{
  const b = backend(0, 0, 4, { depth_known: false });
  const tile = api.queueTile(b, { depth: [] });
  assert.ok(!Number.isFinite(Number(headline(tile))),
    `an unmeasured depth must not be printed as a number: "${headline(tile)}"`);
  assert.ok(!/\d/.test(subline(tile)), `nothing measured here, yet a number is printed: "${subline(tile)}"`);
  const figure = runWait(api.engineStats(b, true));
  assert.strictEqual(figure, '—', `/metrics did not answer, yet RUN + WAIT reports a figure: "${figure}"`);
  assert.ok(/engine not answering/.test(api.engineStats(b, true)), 'the unmeasured figure carries the LB\'s reason');
  checks += 4;
}

// 4. THE ROUTER IS UNTOUCHED. These are the exact values router.py `depth` computes for the same
//    backends; if a display change ever moves them, routing and demotion moved with it.
{
  assert.strictEqual(api.depthOf(backend(0, 6, 25)), 25);
  assert.strictEqual(api.depthOf(backend(2, 14, 35)), 37);
  assert.strictEqual(api.depthOf(backend(2, 14, 3)), 16);
  assert.strictEqual(api.depthOf(backend(0, 0, 0)), 0);
  checks += 4;
}

console.log(`ok — ${checks} checks`);
