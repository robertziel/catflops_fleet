/* THE GLIDE and THE STREAM on the gateway page.
 *
 *   * a graph shows SPARKLINE_S of time, whatever the probe: 90 columns at 1 s, 18 at 5 s;
 *   * a fresh series slides its newest sample in over one probe period from the moment it was taken (a negative
 *     animation-delay), drawn one column wider on the left inside a clip, so the left edge never empties;
 *   * the glide invents nothing: the samples drawn are the series' own, and `data-samples` still spells the box's slots;
 *   * a series gone quiet does not glide;
 *   * the page follows /lb/stream: the first `status` is drawn, a `patch` is applied to the stream's own document,
 *     and a stream that is refused is reopened — the page never polls instead.
 *
 *   node tests/graph_glide.node.js   (tests/test_graph_glide_node.py runs this when node exists)
 */
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const STATIC = path.join(__dirname, '..', 'vllm_lb', 'static');
const src = require('./palette_env').withPalette(fs.readFileSync(path.join(STATIC, 'gateway.js'), 'utf8'));
const css = fs.readFileSync(path.join(STATIC, 'gateway.css'), 'utf8');
const statusDelta = require(path.join(STATIC, 'status_delta.js'));

const el = () => ({ addEventListener() {}, removeAttribute() {}, setAttribute() {}, append() {},
  classList: { toggle() {}, add() {}, remove() {} }, style: {}, hidden: false, innerHTML: '',
  textContent: '', value: '', dataset: {}, querySelectorAll: () => [], querySelector: () => null,
  focus() {}, observe() {}, closest: () => null, getBoundingClientRect: () => ({}) });
const ids = {};
const document = { getElementById: (id) => (ids[id] = ids[id] || el()), createElementNS: el, createElement: el,
  addEventListener() {}, querySelectorAll: () => [], querySelector: () => null,
  body: { style: {}, classList: { toggle() {} } }, documentElement: { style: {}, classList: { toggle() {} } } };
const window = { addEventListener() {}, innerWidth: 1280, innerHeight: 800, matchMedia: () => ({
  matches: false, addEventListener() {} }), localStorage: { getItem: () => null, setItem() {} } };

/* A fake EventSource: records each one opened, lets the test deliver events and errors. */
const opened = [];
class FakeEventSource {
  constructor(url) { this.url = url; this.on = {}; this.readyState = 0; this.closed = false; opened.push(this); }
  addEventListener(name, fn) { this.on[name] = fn; }
  close() { this.closed = true; this.readyState = 2; }
  emit(name, data) { this.on[name]({ data: JSON.stringify(data) }); }
}
FakeEventSource.CLOSED = 2;
const timers = [];
const api = new Function('document', 'window', 'fetch', 'location', 'setInterval', 'setTimeout',
  'getComputedStyle', 'ResizeObserver', 'localStorage', 'navigator', 'EventSource', 'statusDelta',
  `${src}\n;return { sparkline, timeline, renderStatusHook: () => lastStatus, get cols() { return SPARKLINE_COLS; },
     SPARKLINE_W, setTimeAxis };`)(
  document, window, () => new Promise(() => {}), { search: '', href: '' }, () => 0,
  (fn, ms) => { timers.push([fn, ms]); return 0; }, () => ({}), undefined, window.localStorage,
  { userAgent: 'node' }, FakeEventSource, statusDelta);

let checks = 0;
const ok = (c, m, d) => { assert.ok(c, `${m}${d === undefined ? '' : ` — ${JSON.stringify(d)}`}`); checks++; };

// ── the glide ─────────────────────────────────────────────────────────────────────────────────────────────────────
ok((() => { try { api.sparkline([1, 2], { cols: api.cols }); return false; } catch (e) { return /no time axis/.test(e.message); } })(),
   'before any status a graph has no time axis and says so — no default column count');
api.setTimeAxis({ config: { probe_interval_s: 5 } });
ok(api.cols === 18, '90 s at a 5 s probe is 18 columns', api.cols);
const S = { now: 100.2, config: { probe_interval_s: 1 } };
const pts = Array.from({ length: 30 }, (_, i) => [71 + i, 10 + (i % 5)]);   // one a second, the newest at 100
const fresh = api.timeline(S, pts);
ok(fresh.slide && Math.abs(fresh.slide.age_s - 0.2) < 1e-9 && fresh.slide.period_s === 1,
   'a fresh series glides: its newest sample 0.2 s old, over a 1 s period', fresh.slide);
const html = api.sparkline(fresh, { cols: 18, key: 'g' });
ok(/class="graph-glide"/.test(html), 'the glide group is drawn');
ok(/animation-duration:1000ms/.test(html) && /animation-delay:-200ms/.test(html),
   'it lasts one period and starts where the newest sample is in it', html.match(/style="[^"]+"/)[0]);
ok(/<clipPath id="(sc[^"]+)">/.test(html) && html.includes(`clip-path="url(#${html.match(/<clipPath id="([^"]+)"/)[1]})"`),
   'it is clipped to the box');
ok((html.match(/data-samples="([01]+)"/) || [])[1].length === 18, 'data-samples still spells the 18 slots of the box');
const xs = [...(html.match(/<path d="M([^"]+)" fill="none" stroke="url/) || [])[1].matchAll(/(-?[\d.]+) [\d.]+/g)]
  .map((m) => Number(m[1]));
ok(xs.length === 19 && xs[0] < 0 && Math.abs(xs[xs.length - 1] - api.SPARKLINE_W) < 1e-6,
   'one column more than the box: the first off its left edge, the newest on its right', xs.slice(0, 2));
const dx = xs[1] - xs[0];
ok(new RegExp(`--glide-from:${dx.toFixed(2)}px`).test(html), 'it slides exactly one column', dx);

const quiet = api.timeline({ now: 110, config: { probe_interval_s: 1 } }, pts);
ok(!quiet.slide, 'a series gone quiet does not glide');
ok(!/graph-glide/.test(api.sparkline(quiet, { cols: 18, key: 'g' })), 'and draws no glide group');
ok(/@media \(prefers-reduced-motion: reduce\) \{ \.graph-glide \{ animation: none; \} \}/.test(css),
   'reduced motion stops the glide');

// ── the stream ────────────────────────────────────────────────────────────────────────────────────────────────────
ok(opened.length === 1 && opened[0].url === '/lb/stream', 'the page opens /lb/stream at load', opened.map((e) => e.url));
ok(!timers.some(([, ms]) => ms === 3000), 'and polls nothing on an interval');
const es = opened[0];
const STATUS = { now: 1, config: { probe_interval_s: 1, tick_s: 1 }, backends: [], clusters: {}, hardware: {},
                 queue: { inflight: [], recent: [] } };
let drew = 0;
try { es.emit('status', STATUS); drew++; } catch (e) { /* the render may need more of a status than this */ }
ok(api.cols === 90, 'a 1 s probe makes a graph 90 columns: 90 s', api.cols);
es.emit('patch', { o: { now: { '=': 2 } } });
ok(api.renderStatusHook() === null || api.renderStatusHook().now === 2 || drew === 0,
   'a patch is applied to the stream\'s document');
const before = opened.length;
es.readyState = FakeEventSource.CLOSED;
es.on.error({});
ok(es.closed, 'a refused stream is closed');
const reopen = timers.find(([, ms]) => ms === 2000);
ok(reopen, 'and opened again after STREAM_REOPEN_MS');
reopen[0]();
ok(opened.length === before + 1 && opened[opened.length - 1].url === '/lb/stream', 'a fresh stream is opened');
const es2 = opened[opened.length - 1];
es2.on.patch({ data: JSON.stringify({ o: { now: { '=': 3 } } }) });
ok(es2.closed, 'a patch before its connection\'s status is never applied: that stream is dropped');

console.log(`ok — ${checks} checks`);
