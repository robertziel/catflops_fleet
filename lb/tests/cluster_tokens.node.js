/* The Two-GB10 box cluster card's SATURATION row, checked where it is built: gateway.js, under node.
 *
 * THE ASK (operator, 2026-10-01): "add tok/s generate and fill on two rows on left of memory
 * bandwidth". The row reads left to right — what the GPUs are doing (SM fill, per box), what that is
 * PRODUCING in tokens, and the bandwidth and compute it costs — so the token pair has to sit BETWEEN
 * the two groups that were already there, and be two rows, not two columns. Position is the ask, so
 * position is asserted.
 *
 * AND THE QUANTITY IS THE POINT. There are two prompt rates in `throughput`, and they differ by an
 * order of magnitude on this fleet because of the prefix cache: `prompt.tok_s` is what the engine
 * BILLED (6,230 tok/s in the captured document, 90 % of it cache hits it never computed) and
 * `prefill_speed.tok_s` is what it COMPUTED, per second it actually spent prefilling (1,918 tok/s).
 * Only the second one belongs beside a bandwidth and a FLOP/s cost; the first would tell the operator
 * the pair reads prompts ten times faster than it does. So this file pins that the billed figure does
 * not appear in the group at all, and that the caption says which of the two is shown.
 *
 * NOTHING MEASURED IS "—", NEVER 0 (worklog/vllm-gateway-throughput.md): a null rate prints the LB's
 * own reason in its place. A MEASURED zero is a different thing and stays a 0 — an engine that was
 * read and had produced nothing — and says "idle" so it cannot be read as a missing reading.
 *
 * No DOM: gateway.js is loaded with a stub `document` whose getElementById is stable per id, so
 * `renderSaturation` can be called and its markup read back. `fetch` never answers, so `poll()`
 * waits forever and nothing else is drawn.
 *
 *   node tests/cluster_tokens.node.js      (tests/test_cluster_tokens.py runs this when node exists)
 *   docker run --rm -v "$PWD":/repo:ro -w /repo/lb node:22-slim node tests/cluster_tokens.node.js
 */
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');

// palette.js first: the colours this script computes are read from the style's tokens (tests/palette_env.js).
const src = require('./palette_env').withPalette(fs.readFileSync(path.join(__dirname, '..', 'vllm_lb', 'static', 'gateway.js'), 'utf8'));
const FIXTURE = path.join(__dirname, 'fixtures', 'lb-status-live-pre-knobs.json');

/** gateway.js in a fresh closure, with a document whose elements persist per id. */
function load() {
  const byId = {};
  const el = () => ({ addEventListener() {}, removeAttribute() {}, setAttribute() {}, append() {},
    classList: { toggle() {}, add() {}, remove() {} }, style: {}, hidden: false, innerHTML: '',
    textContent: '', value: '', dataset: {}, querySelectorAll: () => [], querySelector: () => null,
    focus() {}, observe() {}, closest: () => null, getBoundingClientRect: () => ({}) });
  const document = {
    getElementById: (id) => (byId[id] || (byId[id] = el())),
    createElementNS: el, createElement: el, addEventListener() {},
    querySelectorAll: () => [], querySelector: () => null,
    body: { style: {}, classList: { toggle() {} } },
    documentElement: { style: {}, classList: { toggle() {} } } };
  const window = { addEventListener() {}, innerWidth: 1280, innerHeight: 800, matchMedia: () => ({
    matches: false, addEventListener() {} }), localStorage: { getItem: () => null, setItem() {} } };
  const api = new Function('document', 'window', 'fetch', 'location', 'setInterval', 'setTimeout',
    'getComputedStyle', 'ResizeObserver', 'localStorage', 'navigator',
    `${src}\n;return { renderSaturation };`)(
    document, window, () => new Promise(() => {}), { search: '', href: '' }, () => 0, () => 0,
    () => ({}), undefined, window.localStorage, { userAgent: 'node' });
  return { api, byId };
}

/** The saturation panel's markup for a status document. */
function saturationHtml(status) {
  const { api, byId } = load();
  api.renderSaturation(status);
  return byId['cluster-saturation'].innerHTML;
}

// ── reading the rendered meters the way a person reads them ──────────────────
const strip = (h) => h.replace(/<[^>]*>/g, '').replace(/&middot;/g, '·')
  .replace(/&nbsp;/g, ' ').replace(/&amp;/g, '&').replace(/&deg;/g, '°')
  .replace(/\s+/g, ' ').trim();

/** Each <section class="sat-group"> of the panel, in document order. */
function groups(html) {
  return [...html.matchAll(/<section class="sat-group[^"]*"[\s\S]*?<\/section>/g)].map((m) => m[0]);
}

/** One group's meters, in document order: label, sub, headline, note — as read, not as coded — and
 *  the `title` that says what the quantity is (since 2026-10-01 a hover, never a row). */
function meters(group) {
  // `satMeter` nests no <div>, so the first </div> closes each meter.
  return [...group.matchAll(/<div class="sat-meter[^"]*"([^>]*)>([\s\S]*?)<\/div>/g)]
    .map((m) => [m[1], m[2]])
    .map(([attrs, body]) => ({
      label: strip((body.match(/class="sat-label">([\s\S]*?)<\/span>/) || [, ''])[1]),
      sub: strip((body.match(/class="sat-sub">([\s\S]*?)<\/span>/) || [, ''])[1]),
      title: strip((attrs.match(/title="([^"]*)"/) || [, ''])[1]),
      value: strip((body.match(/<b class="sat-val">([\s\S]*?)<\/b>/) || [, ''])[1]),
      note: strip((body.match(/class="sat-note"[^>]*>([\s\S]*)$/) || [, ''])[1]),
    }));
}

/** What a group says its numbers are: its section's `title` (it was a caption row until 2026-10-01). */
const caption = (group) => strip((group.match(/<section class="sat-group[^"]*"[^>]*?title="([^"]*)"/) || [, ''])[1]);

/** The group the token rows live in: the one whose meters are the two token rates. */
function tokenGroup(html) {
  const found = groups(html).filter((g) => meters(g).some((m) => m.label === 'generated'));
  assert.strictEqual(found.length, 1, 'exactly one group holds the token rows');
  return found[0];
}

const LIVE = JSON.parse(fs.readFileSync(FIXTURE, 'utf8'));
// The captured document predates the per-box ceilings (saturation.py `per_box`): the LB now sends each box's share
// beside the pair's figures. Added here exactly as it computes them — 1 card a box, the same peaks, derived.
for (const key of ['compute', 'bandwidth']) {
  const c = LIVE.saturation[key];
  c.boxes = ['box-a', 'box-b'].map((box) => ({ box, cards: 1, rate: c.rate, peak: c.peak, pct: c.pct, why: null,
    derived: "derived: the engine's one rate per GPU (it has no per-rank counter) x this box's serving cards — " +
      "tensor parallelism splits each layer's work evenly over the ranks" }));
}
const PER_BOX = ['memory bandwidth · box-a', 'memory bandwidth · box-b', 'compute · box-a', 'compute · box-b'];
const clusterOf = (s) => s.backends.find((b) => b.kind === 'cluster');

/** The captured document with the cluster's `throughput` block replaced. */
function withThroughput(patch) {
  const s = JSON.parse(JSON.stringify(LIVE));
  Object.assign(clusterOf(s).throughput, patch);
  return s;
}

// ── 1. the ask: two rows, between the SM fill pair and the bandwidth pair ────
{
  const html = saturationHtml(LIVE);
  const gs = groups(html);
  assert.strictEqual(gs.length, 3, `three groups, got ${gs.length}`);
  const labels = gs.map((g) => meters(g).map((m) => m.label));
  // Left: one SM fill meter per box. Middle: the tokens. Right: the two engine rates.
  assert.ok(labels[0].every((l) => l.startsWith('SM fill')), `left group is SM fill, got ${labels[0]}`);
  assert.deepStrictEqual(labels[1], ['generated', 'prefill / request'],
    `the middle group is the two token rows, got ${labels[1]}`);
  assert.deepStrictEqual(labels[2], PER_BOX, `the right group is the engine's rates, per box, got ${labels[2]}`);
  // TWO ROWS, not two columns: its meters are stacked by a class of its own, which the CSS keys on.
  assert.ok(/<section class="sat-group[^"]*\bsat-tokens\b/.test(gs[1]),
    'the token group carries sat-tokens, the hook the CSS stacks it with');
}

// ── 2. the quantity: COMPUTED prefill, never the cache-inflated billed rate ──
{
  const html = saturationHtml(LIVE);
  const g = tokenGroup(html);
  const [generated, prefill] = meters(g);
  const t = clusterOf(LIVE).throughput;

  // 189.9 tok/s generated, over the same ~6 s window the engine's own rates are taken over.
  assert.strictEqual(generated.value, '190 tok/s', `generated headline: ${generated.value}`);
  assert.ok(/\b6 s\b/.test(generated.note), `generated names its window: ${generated.note}`);

  // 1918.2 tok/s computed prefill — NOT the 6230 tok/s the engine billed, 90 % of which was cache.
  assert.strictEqual(Math.round(t.prefill_speed.tok_s), 1918);
  assert.strictEqual(Math.round(t.prompt.tok_s), 6230);
  assert.strictEqual(prefill.value, '1,918 tok/s', `prefill headline: ${prefill.value}`);
  assert.ok(!/6,?230/.test(strip(g)),
    `the billed prompt rate must not appear in the token group: ${strip(g)}`);
  // And the reader is told which of the two it is, where the group says what its numbers are (its title).
  const cap = caption(g).toLowerCase();
  assert.ok(cap.includes('computed') && cap.includes('cache'),
    `the group's title says the rate is the computed one and the cache is excluded: ${caption(g)}`);
  // The denominator is not the wall clock, and the row says so — in its title, under the label PREFILL.
  assert.ok(/prefilling/.test(prefill.title) && /prompt/.test(prefill.title),
    `the prefill row names its denominator: ${prefill.title}`);
  assert.strictEqual(prefill.sub, '', `no sentence under the label, the title carries it: ${prefill.sub}`);
  // 54 requests finished in the 5-minute window, and the row says that too.
  assert.ok(/\b54 requests?\b/.test(prefill.note) && /5m/.test(prefill.note),
    `prefill note carries its sample: ${prefill.note}`);
}

// ── 3. the page's grammar: " · " separates context, " + " adds to a headline ─
{
  const g = tokenGroup(saturationHtml(LIVE));
  meters(g).concat([{ note: caption(g), sub: '', value: '', label: '' }]).forEach((m) => {
    assert.ok(!m.note.includes(' + ') && !m.sub.includes(' + '),
      `nothing under these headlines adds to them, so " + " has no business here: ${m.note}${m.sub}`);
  });
}

// ── 4. an IDLE engine: a measured 0 is a 0 and says so; an unmeasured rate is "—" ──
{
  // The pair is up and was read; nothing was asked of it. `generated` is a measured 0 (a counter
  // read twice, flat), and no request finished inside the 5-minute prefill window.
  const html = saturationHtml(withThroughput({
    generated: { tok_s: 0, window_s: 6.3, why: null },
    prefill_speed: { tok_s: null, requests: 0, window_s: 301.6, why: 'window not reached' },
  }));
  const [generated, prefill] = meters(tokenGroup(html));
  assert.strictEqual(generated.value, '0 tok/s', `a measured zero stays a zero: ${generated.value}`);
  assert.ok(/idle/.test(generated.note), `and says it is idle: ${generated.note}`);
  assert.strictEqual(prefill.value, '—', `an unmeasured rate is "—": ${prefill.value}`);
  assert.ok(/window not reached/.test(prefill.note),
    `with the LB's own reason in its place: ${prefill.note}`);
  assert.ok(!/\b0 tok\/s\b/.test(prefill.value + prefill.note),
    `and never a 0 nobody measured: ${prefill.value} / ${prefill.note}`);
  assert.ok(/\b0 requests\b/.test(prefill.note),
    `the empty window is quantified, not just named: ${prefill.note}`);
}

// ── 5. nothing read at all: both rows are "—" with the reason, neither is 0 ──
{
  const html = saturationHtml(withThroughput({
    generated: { tok_s: null, window_s: null, why: 'first sample' },
    prefill_speed: { tok_s: null, requests: null, window_s: null, why: 'first sample' },
  }));
  meters(tokenGroup(html)).forEach((m) => {
    assert.strictEqual(m.value, '—', `${m.label} is "—" when nothing was read: ${m.value}`);
    assert.ok(/measuring/.test(m.note), `${m.label} prints the reason: ${m.note}`);
    assert.ok(!/\b0\b/.test(m.value), `${m.label} never invents a zero: ${m.value}`);
  });
}

// ── 6. a tok/s TARGET colours the generated line and changes nothing it says ──
// `throughput_target` arrived after these rows did (`targetOf`, 2026-10-01) and feeds the generated
// graph's scale. The captured document predates it, so the rows above exercise the no-target
// fallback; this pins that a target present changes the graph and not one word of either row.
{
  const without = tokenGroup(saturationHtml(LIVE));
  const s = JSON.parse(JSON.stringify(LIVE));
  s.throughput_target = { gen_tok_s: 420.9, prompt_tok_s: 9500.0, windows: 37, basis: 'setup',
                          config: { model_choice: 'qwen3.8' } };
  const withTarget = tokenGroup(saturationHtml(s));
  assert.notStrictEqual(withTarget, without, 'the target reaches the generated graph');
  assert.deepStrictEqual(meters(withTarget).map((m) => [m.label, m.sub, m.value, m.note]),
    meters(without).map((m) => [m.label, m.sub, m.value, m.note]),
    'and says nothing new: a scale is not a measurement');
}

// ── 7. a status with no cluster backend draws the two original groups, and does not throw ──
{
  const s = JSON.parse(JSON.stringify(LIVE));
  s.backends = s.backends.filter((b) => b.kind !== 'cluster');
  const gs = groups(saturationHtml(s));
  assert.strictEqual(gs.length, 2, `no cluster server, no token rows: ${gs.length} groups`);
  assert.deepStrictEqual(meters(gs[1]).map((m) => m.label), PER_BOX);
}

// ── 8. PREFILL draws its OWN past: the `prefill_tok_s` ring, never the billed `prompt_tok_s` line ──
// It headed "1,170 tok/s · 5 requests over 5m 1s" over an empty slot while GENERATED beside it drew its
// ring: `prefill_speed` had no history behind it. The hub now samples it on every probe.
{
  /** One token row's raw markup, by label. */
  const rowOf = (status, label) => {
    const g = tokenGroup(saturationHtml(status));
    const body = [...g.matchAll(/<div class="sat-meter[^"]*"[^>]*>([\s\S]*?)<\/div>/g)].map((m) => m[1])
      .find((b) => strip((b.match(/class="sat-label">([\s\S]*?)<\/span>/) || [, ''])[1]) === label);
    assert.ok(body, `a ${label} row`);
    return body;
  };
  const drawn = (body) => (((body.match(/data-samples="([01]*)"/) || [, ''])[1]).match(/1/g) || []).length;

  // The captured document predates the ring. Without it there is nothing to draw — and PREFILL does not
  // borrow the billed prompt line, which that document does carry.
  const h0 = clusterOf(LIVE).history;
  assert.ok(Array.isArray(h0.prompt_tok_s) && !('prefill_tok_s' in h0), 'the fixture has no prefill ring');
  assert.ok(!/<svg/.test(rowOf(LIVE, 'prefill / request')), 'no prefill ring, no prefill graph — never the billed one');

  // With the ring: real samples drawn, and a gap where no request had finished its prefill.
  const s = JSON.parse(JSON.stringify(LIVE));
  clusterOf(s).history.prefill_tok_s = h0.at.map((_, i) => (i < 4 ? null : (i < 40 ? 1870.4 : 1918.2)));
  const withRing = rowOf(s, 'prefill / request');
  assert.ok(/<svg class="sparkline sat-sparkline/.test(withRing), 'PREFILL draws a graph once its ring exists');
  assert.ok(drawn(withRing) >= 2, `at least two drawn samples, got ${drawn(withRing)}`);

  // Drawn from THAT ring: moving prefill_tok_s moves the line; moving prompt_tok_s does not.
  const billed = JSON.parse(JSON.stringify(s));
  clusterOf(billed).history.prompt_tok_s = clusterOf(billed).history.prompt_tok_s.map((v) => (v == null ? v : v * 3));
  assert.strictEqual(rowOf(billed, 'prefill / request'), withRing, 'the billed prompt rate does not reach the PREFILL line');
  const moved = JSON.parse(JSON.stringify(s));
  clusterOf(moved).history.prefill_tok_s = clusterOf(moved).history.prefill_tok_s
    .map((v, i) => (v == null ? v : v + 40 * (i % 3)));
  assert.notStrictEqual(rowOf(moved, 'prefill / request'), withRing, 'the PREFILL line is the prefill_tok_s ring');

  // And a graph is not a measurement: every word of both rows is what it was without one.
  assert.deepStrictEqual(meters(tokenGroup(saturationHtml(s))).map((m) => [m.label, m.sub, m.value, m.note]),
    meters(tokenGroup(saturationHtml(LIVE))).map((m) => [m.label, m.sub, m.value, m.note]),
    'the ring changes the graph and nothing the rows say');
}

console.log('ok');
