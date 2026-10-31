/* THE PER-REQUEST LINE on a machine card's rate graphs — checked where gateway.js draws it.
 *
 * Operator, 2026-10-05: "Add additional graph value to existing one per one request average, make it
 * smaller, greyish transparent so it is just subtle addition to main data on that graph."
 *
 * GENERATED, PREFILL and PROMPT are the PAIR's totals, and the same total is a very different
 * experience at 2 streams and at 20. The second line is that: the same measure divided by the
 * requests the engine was RUNNING at each sample (`history.running`). What this file pins down:
 *
 *   * it is the arithmetic it claims — value ÷ running, sample by sample, on the SAME y axis as the
 *     line above it (one axis: it is the same quantity in the same unit);
 *   * it is SUBORDINATE — thinner than the headline's line, no fill, no gradient, no colour from the
 *     card's target scale, and it never glows with the line above it;
 *   * a sample where nothing was running is a GAP, not a zero;
 *   * an LB too old to publish `running` simply has no second line, and nothing else changes.
 *
 *   node tests/per_request_line.node.js   (tests/test_per_request_line.py runs this when node exists)
 */
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');

// palette.js first: the colours this script computes are read from the style's tokens (tests/palette_env.js).
const src = require('./palette_env').withPalette(fs.readFileSync(path.join(__dirname, '..', 'vllm_lb', 'static', 'gateway.js'), 'utf8'));
const css = fs.readFileSync(path.join(__dirname, '..', 'vllm_lb', 'static', 'gateway.css'), 'utf8');

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
  `${src}\n;return { generatedTile, promptTile, prefillTile, perRequestOf, SPARKLINE_H, SPARKLINE_W };`)(
  document, window, () => new Promise(() => {}), { search: '', href: '' }, () => 0, () => 0,
  () => ({}), undefined, window.localStorage, { userAgent: 'node' });

/* The pair as /lb/status serves it: six samples, the engine running 8 requests at the last of them,
   one sample where it was running none (a gap for a per-request figure, never a zero). */
const AT = [100, 105, 110, 115, 120, 125];
const GEN = [80, 96, 112, 96, 120, 119];
const RUN = [4, 6, 8, 0, 10, 8];
const PROMPT = [1000, 2000, 4000, 8000, 16000, 19500];
const COMPUTED = [100, 200, 400, 600, 700, 695];
const pair = (over) => ({
  kind: 'cluster', url: 'http://192.0.2.2:8100', live: true, running: 8, waiting: 0, inflight: 8,
  depth_known: true,
  throughput: { source: 'metrics', generated: { tok_s: 119 },
                prompt: { tok_s: 19500, computed_tok_s: 695 } },
  history: { at: AT, running: RUN, gen_tok_s: GEN, prompt_tok_s: PROMPT, computed_tok_s: COMPUTED,
             ...(over || {}) },
});
const S = { now: 126, throughput_target: { gen_tok_s: 153, prompt_tok_s: 19200, computed_tok_s: 2300 } };

const quietOf = (html) => (html.match(/<path class="sparkline-per-req" d="([^"]+)"/) || [])[1] || '';
const mainOf = (html) =>
  (html.match(/<path d="([^"]+)" fill="none" stroke="url\(#[^)]+\)" stroke-width="1\.5"/) || [])[1] || '';
const ptsOf = (d) => [...d.matchAll(/([\d.]+) ([\d.]+)/g)].map(([, x, y]) => [Number(x), Number(y)]);

let checks = 0;
const ok = (c, m, d) => { assert.ok(c, `${m}${d === undefined ? '' : ` — ${JSON.stringify(d)}`}`); checks++; };

// ── the arithmetic, and the shared axis ────────────────────────────────────────────────────────
{
  const html = api.generatedTile(S, pair());
  const quiet = ptsOf(quietOf(html));
  const main = ptsOf(mainOf(html));
  ok(quiet.length, 'GENERATED draws a per-request line');
  // The gap: sample 3 ran nothing, so the quiet line is TWO runs and has one point fewer than the
  // headline's six.
  ok(quiet.length === 5, 'a sample with nothing running is a gap, not a zero', quiet.length);
  ok((quietOf(html).match(/M/g) || []).length === 2, 'and the gap really breaks the line in two',
     quietOf(html));
  // ONE AXIS: the last sample is 119 tok/s over 8 running = 14.875, and the axis both lines are
  // drawn on tops out at the target 153 — so y = 1 + (H-2)(1 - 14.875/153).
  const H = api.SPARKLINE_H;
  const y = (v) => 1 + (H - 2) * (1 - v / 153);
  const last = quiet[quiet.length - 1];
  ok(Math.abs(last[1] - y(119 / 8)) < 0.02, 'the newest point is the rate ÷ the running count',
     { drawn: last[1], wanted: y(119 / 8) });
  ok(Math.abs(main[main.length - 1][1] - y(119)) < 0.02, 'on the SAME axis as the headline\'s line',
     { drawn: main[main.length - 1][1], wanted: y(119) });
  ok(last[0] === main[main.length - 1][0], 'and at the same x: one sample, two readings');
  // SUBORDINATE: thinner, no fill, no gradient, no colour of its own.
  ok(/class="sparkline-per-req"[^>]*stroke-width="1"/.test(html.replace(/\n\s*/g, ' ')),
     'the quiet line is 1px against the headline line\'s 1.5');
  ok(!/class="sparkline-per-req"[^>]*fill="url/.test(html) && /class="sparkline-per-req"[^>]*fill="none"/
      .test(html.replace(/\n\s*/g, ' ')), 'it carries no fill');
  ok(!/class="sparkline-per-req"[^>]*stroke="/.test(html.replace(/\n\s*/g, ' ')),
     'and no stroke of its own: the stylesheet gives it grey, never the target scale');
  // It is drawn BEFORE the headline's line, so it can never cover it.
  ok(html.indexOf('sparkline-per-req') < html.indexOf('stroke-width="1.5"'),
     'it is drawn under the line it belongs to');
}

// ── the same on the other two rate tiles, each with its own measure ────────────────────────────
for (const [name, tile, series, now] of [['PROMPT', api.promptTile, PROMPT, 19500],
                                         ['PREFILL', api.prefillTile, COMPUTED, 695]]) {
  const html = tile(S, pair());
  const quiet = ptsOf(quietOf(html));
  ok(quiet.length === 5, `${name} draws its own per-request line`, quiet.length);
  const per = now / 8;
  const all = series.map((v, i) => (RUN[i] > 0 ? v / RUN[i] : null)).filter((v) => v != null);
  const top = Math.max(...series, S.throughput_target[name === 'PROMPT' ? 'prompt_tok_s' : 'computed_tok_s']);
  const y = (v) => 1 + (api.SPARKLINE_H - 2) * (1 - v / top);
  ok(Math.abs(quiet[quiet.length - 1][1] - y(per)) < 0.02,
     `${name}: the newest point is its own rate ÷ running`, { drawn: quiet[quiet.length - 1][1], wanted: y(per) });
  ok(all.every((v) => v <= Math.max(...series)), `${name}: a share is never above its own total`);
}

// ── the headline says the number, where numbers belong on this card ────────────────────────────
{
  const html = api.generatedTile(S, pair());
  ok(/title="[^"]*per running request/.test(html),
     'the headline\'s tooltip carries the per-request figure', (html.match(/title="[^"]*"/) || [])[0]);
  ok(/title="[^"]*\b15 tok\/s per running request/.test(html), '…as the number it is (119 ÷ 8)',
     (html.match(/title="[^"]*"/) || [])[0]);
}

// ── an LB that does not publish `running`: no second line, nothing else changes ────────────────
{
  const old = pair();
  delete old.history.running;
  const html = api.generatedTile(S, old);
  ok(quietOf(html) === '', 'an LB with no `running` ring draws no per-request line');
  ok(mainOf(html) !== '', 'and the headline\'s line is untouched');
  ok(!/per running request/.test(html), 'and claims no per-request figure');
  // A ring of the wrong length is the same case: nothing is invented.
  const short = pair({ running: [8, 8] });
  ok(quietOf(api.generatedTile(S, short)) === '', 'a `running` ring that does not line up is ignored');
  // Nothing running at all, ever: no line rather than a flat zero.
  const idle = pair({ running: [0, 0, 0, 0, 0, 0] });
  ok(quietOf(api.generatedTile(S, idle)) === '', 'nothing running at any sample draws nothing');
}

// ── the stylesheet keeps it quiet ──────────────────────────────────────────────────────────────
{
  const rule = (css.match(/\.sparkline-per-req\s*\{([^}]*)\}/) || [])[1] || '';
  ok(/stroke:\s*var\(--ink-faint\)/.test(rule), 'it is drawn in the faintest ink token', rule);
  const op = Number((rule.match(/opacity:\s*([\d.]+)/) || [])[1]);
  ok(op > 0 && op < 0.5, 'and half-transparent, so it reads as an addition', op);
  ok(/\.sparkline\.glow path\.sparkline-per-req\s*\{\s*filter:\s*none/.test(css),
     'a glowing headline never drags the quiet line into its halo');
}

console.log(`ok — ${checks} checks: the per-request line is value ÷ running, on one axis, and quiet`);
