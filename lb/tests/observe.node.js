/* THE OBSERVABILITY VIEWS as drawn (static/observe.js), under node:
 *
 *   1. CHECK (GET /lb/check?box=): one row per capability — ✓/✗, the capability, its target, the probe's words and
 *      ms, and the hint on a failed row only — its short form printed ("http://…:8001 unreachable from the gateway"),
 *      the full recovery in its title; the head counts "3/4 pass"; a failed fetch says its HTTP status;
 *   2. HISTORY (GET /lb/history?box=&tier=): a 48 h / 30 d toggle (aria-pressed), one chart per figure — a min..max
 *      band and the avg line — headlined with numbers ("avg 54 · max 71 °C"); a figure with no reading in the range
 *      draws no chart and says "—"; a corrupt history file's reason is shown; under each chart its start and end
 *      ticks, read from the samples' own timestamps ("Nov 25 14:00" … "now"), placed where those samples are drawn;
 *   3. DAILY (GET /lb/perf/daily): per model its 24 h tokens, its 30 d tokens and its peak-minute tok/s in the head,
 *      then ONE bar row of 30 bars (tokens/day — a tok/s row drew nearly the same shape, the UX audit), each bar
 *      titled with its day and figures; the dialog's summary line carries the best peak minute of all models; a day
 *      with no tokens is an empty slot, never a 0-height bar posing as a measurement;
 *   3b. the "daily" button is the card buttons' one neutral style (`obs-btn`), never the accent `cn-open` of connect;
 *   4. every server string is escaped.
 *
 *   node tests/observe.node.js      (tests/test_observe_node.py runs this when node exists)
 */
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const STATIC = path.join(__dirname, '..', 'vllm_lb', 'static');
const src = require('./palette_env').withPalette(fs.readFileSync(path.join(STATIC, 'gateway.js'), 'utf8') + '\n'
  + fs.readFileSync(path.join(STATIC, 'observe.js'), 'utf8'));
const el = () => ({ addEventListener() {}, removeAttribute() {}, setAttribute() {}, append() {}, appendChild() {},
  insertBefore() {}, classList: { toggle() {}, add() {}, remove() {}, contains: () => false }, style: {}, hidden: false,
  innerHTML: '', textContent: '', value: '', dataset: {}, querySelectorAll: () => [], querySelector: () => null,
  focus() {}, observe() {}, closest: () => null, getBoundingClientRect: () => ({}) });
const document = { getElementById: el, createElementNS: el, createElement: el, createTextNode: el, addEventListener() {},
  querySelectorAll: () => [], querySelector: () => null, body: { style: {}, classList: { toggle() {} }, appendChild() {} },
  documentElement: { style: {}, classList: { toggle() {} } }, readyState: 'complete' };
const window = { addEventListener() {}, innerWidth: 1280, innerHeight: 800, matchMedia: () => ({
  matches: false, addEventListener() {} }), localStorage: { getItem: () => null, setItem() {} } };

const api = new Function('document', 'window', 'fetch', 'location', 'setInterval', 'setTimeout',
  'getComputedStyle', 'ResizeObserver', 'localStorage', 'navigator',
  `${src}\n;return { obsCheckHtml, obsHistoryHtml, obsDailyHtml, obsBandSvg, obsBarsSvg, obsNum, obsFig, obsWhen };`)(
  document, window, () => new Promise(() => {}), { search: '', href: '' }, () => 0, () => 0,
  () => ({}), undefined, window.localStorage, { userAgent: 'node' });

// ── 1. check ──────────────────────────────────────────────────────────────────────────────────
const check = { at: 1, passed: 2, failed: 1, boxes: [{ box: 'node-1', ok: false, checks: [
  { cap: 'telemetry', target: 'node-1', ok: true, detail: 'sample 1.2 s old', hint: null, ms: 0 },
  { cap: 'engine', target: 'http://e:8000', ok: true, detail: 'live · 1 model', hint: null, ms: 12.3 },
  { cap: 'ssh', target: 'u@a (pair)', ok: false, detail: 'ssh exit 255: <denied>', hint_short: 'u@a <unreachable>',
    hint: 'add the key <k>', ms: 80 }] }] };
let html = api.obsCheckHtml('node-1', { data: check });
assert.ok(html.includes('2/3 pass'), 'the head counts the passes');
assert.strictEqual((html.match(/obs-row ok/g) || []).length, 2);
assert.strictEqual((html.match(/obs-row bad/g) || []).length, 1);
assert.strictEqual((html.match(/obs-hint/g) || []).length, 1, 'a hint on a failed row only');
// The row prints the short form; the full recovery is its title (one hover / long-press away), both escaped.
assert.ok(html.includes('<span class="obs-hint" title="add the key &lt;k&gt;">u@a &lt;unreachable&gt;</span>'), html);
assert.ok(html.includes('&lt;denied&gt;') && !html.includes('<denied>'), 'escaped');
assert.ok(html.includes('12 ms'), 'the probe time');
assert.ok(!html.includes('no ssh address'));
html = api.obsCheckHtml('gw', { data: { boxes: [{ box: 'gw', ok: true, ssh_why: 'no ssh address <x>', checks: [
  { cap: 'telemetry', target: 'gw', ok: true, detail: 'sample 1 s old', hint: null, ms: 0 }] }] } });
assert.ok(html.includes('no ssh address &lt;x&gt;'), 'a box without an ssh address says why, escaped');
// …as a figure, not a sentence: "ssh —" opening (tap / keyboard / hover) to the reason (the UX audit).
assert.ok(html.includes('<details class="obs-why"><summary title="no ssh address &lt;x&gt;">ssh —</summary>'), html);
assert.ok(!/<p class="obs-na">no ssh/.test(html), 'never a bare sentence under the rows');
html = api.obsCheckHtml('node-1', { error: 'HTTP 404: no box' });
assert.ok(html.includes('HTTP 404'));
assert.ok(api.obsCheckHtml('node-1', null).includes('checking'));

// ── 2. history ────────────────────────────────────────────────────────────────────────────────
const T = 1_795_176_000;
const hist = { box: 'node-1', tier: '48h', bucket_s: 60, window_s: 172800, state: 'loaded', why: '', series: {
  gpu_temp_c: { label: 'GPU TEMP', unit: '°C', points: [[T - 120, 40, 50, 60], [T - 60, 45, 55, 71]] },
  gpu_power_w: { label: 'GPU POWER', unit: 'W', points: [] } } };
html = api.obsHistoryHtml('node-1', '48h', { data: hist }, T);
assert.ok(/aria-pressed="true"[^>]*>48 h/.test(html) && /aria-pressed="false"[^>]*>30 d/.test(html), 'the toggle');
assert.ok(html.includes('avg 52.5') && html.includes('max 71') && html.includes('min 40'), 'numbers in the headline');
assert.strictEqual((html.match(/<svg/g) || []).length, 1, 'no chart for a figure with no reading');
assert.ok(html.includes('GPU POWER') && html.includes('—'));
html = api.obsHistoryHtml('node-1', '30d', { data: Object.assign({}, hist, { tier: '30d', state: 'corrupt',
  why: 'bad <file>' }) }, T);
assert.ok(html.includes('bad &lt;file&gt;'), 'the corrupt reason, escaped');
const svg = api.obsBandSvg([[T - 120, 1, 2, 3], [T - 60, 2, 3, 4]], T - 172800, T, 60);
assert.ok(svg.includes('obs-band') && svg.includes('obs-avg'));
const gap = api.obsBandSvg([[T - 7200, 1, 2, 3], [T - 60, 2, 3, 4]], T - 172800, T, 60);
assert.strictEqual((gap.match(/class="obs-band"/g) || []).length, 2, 'a gap breaks the band');
// The ticks: the first sample's own time on the left, "now" on the right when the last sample is the current bucket;
// each placed where its sample is drawn (a sample that started 40 h into a 48 h window ticks at 83 %, not at 0).
const when = (t) => api.obsWhen(t);
assert.match(when(T - 120), /^[A-Z][a-z]{2} \d{1,2} \d{2}:\d{2}$/, 'a tick reads "Nov 25 14:00"');
html = api.obsHistoryHtml('node-1', '48h', { data: hist }, T);
assert.ok(html.includes(`<span class="obs-t01" style="right:0.0%">${when(T - 120)} – now</span>`),
  'the last sample is the current bucket: "now"');
const late = { ...hist, series: { gpu_temp_c: { label: 'T', unit: '°C', points: [[T - 8 * 3600, 1, 2, 3], [T - 4 * 3600, 1, 2, 3]] } } };
html = api.obsHistoryHtml('node-1', '48h', { data: late }, T);
// Samples drawn close together (8 h … 4 h ago in a 48 h window) get one range label, right-aligned on the last sample
// (two labels would overlap), ending at the series' own last sample — not "now".
assert.ok(html.includes(`<span class="obs-t01" style="right:8.3%">${when(T - 8 * 3600)} – ${when(T - 4 * 3600)}</span>`), html);
assert.ok(!html.includes('>now<') && !html.includes('obs-t0"'));
const half = { ...hist, series: { gpu_temp_c: { label: 'T', unit: '°C', points: [[T - 40 * 3600, 1, 2, 3], [T - 8 * 3600, 1, 2, 3]] } } };
html = api.obsHistoryHtml('node-1', '48h', { data: half }, T);
assert.ok(html.includes(`<span class="obs-t0" style="left:16.7%">${when(T - 40 * 3600)}</span>`), html);
assert.ok(html.includes(`<span class="obs-t1" style="right:16.6%">${when(T - 8 * 3600)}</span>`), 'each tick where its sample is drawn');
assert.strictEqual((api.obsHistoryHtml('node-1', '48h', { data: hist }, T).match(/obs-tt/g) || []).length, 1,
  'a figure with no reading has no ticks');

// ── 3. daily ──────────────────────────────────────────────────────────────────────────────────
const days = Array.from({ length: 30 }, (_, i) => `2026-10-${String(i + 1).padStart(2, '0')}`);
const series = days.map((d, i) => ({ day: d, gen_tokens: i === 29 ? 1708150 : 0, prompt_tokens: 0,
  gen_tok_s: i === 29 ? 41.25 : null, peak_gen_tok_s: i === 29 ? 55.5 : null, minutes: i === 29 ? 60 : 0, unmeasured: 0 }));
const daily = { days, models: [{ id: 'm', label: 'Model <A>', series, gen_tokens_range: 1708150,
  last_24h: { gen_tokens: 1708150, prompt_tokens: 9, minutes: 60, unmeasured: 0 } }],
  total_24h: { gen_tokens: 1708150 }, error: '', malformed_total: 0 };
html = api.obsDailyHtml({ data: daily });
assert.ok(html.includes('Model &lt;A&gt;'), 'escaped');
assert.ok(html.includes('<section class="obs-model" data-model="m">'), 'a model\'s section names its id: the label is for people, the id for a reader');
assert.ok(html.includes('24 h 1,708,150 tok') && html.includes('30 d 1,708,150 tok'));
assert.ok(html.includes('peak min 55.5 tok/s'), 'the peak is the best MINUTE (peak_gen_tok_s), not the best day\'s pooled rate');
assert.ok(!html.includes('peak 41') && !html.includes('41.25'), 'a day\'s pooled rate is not drawn');
assert.strictEqual((html.match(/<svg class="obs-bars"/g) || []).length, 1, 'one bar row per model: tokens/day only');
assert.ok(!html.includes('tok/s day avg'), 'no tok/s bar row');
assert.ok(/<p class="obs-sum">[^]*peak min 55\.5 tok\/s[^]*<\/p>/.test(html), 'the summary line carries the peak');
const two = { ...daily, models: [daily.models[0], { ...daily.models[0], id: 'n', label: 'N',
  series: series.map((d, i) => ({ ...d, peak_gen_tok_s: i === 3 ? 80 : null })) }] };
assert.ok(/<p class="obs-sum">[^<]*peak min 80 tok\/s/.test(api.obsDailyHtml({ data: two })), 'the best minute of all models');
assert.ok(!/<p class="obs-sum">[^<]*peak/.test(api.obsDailyHtml({ data: { ...daily, models: [{ ...daily.models[0],
  series: series.map((d) => ({ ...d, peak_gen_tok_s: null })) }] } })), 'no measured minute: no peak figure (never 0)');
const bars = api.obsBarsSvg(series.map((d) => d.gen_tokens), days, (v) => `${api.obsNum(v)} tok`);
assert.strictEqual((bars.match(/<rect/g) || []).length, 1, 'a day without tokens draws no bar');
assert.ok(bars.includes('<title>2026-10-30 · 1,708,150 tok</title>'));
assert.ok(api.obsDailyHtml({ data: Object.assign({}, daily, { models: [] }) }).includes('0 models'));
assert.strictEqual(api.obsNum(1708150), '1,708,150');
// A history headline in MiB runs to five digits: grouped, like every other count on the panel.
assert.strictEqual(api.obsFig(68999), '68,999');
assert.strictEqual(api.obsFig(431.4), '431');
assert.strictEqual(api.obsFig(55.83), '55.8');
assert.strictEqual(api.obsFig(-12345.6), '-12,346');

// ── 3b. the daily button ─────────────────────────────────────────────────────────────────────────
const made = [];
const links = { querySelector: () => null, appendChild: (n) => made.push(n) };
const doc2 = { ...document, getElementById: (id) => (id === 'cluster-links' ? links : el()),
  createElement: () => Object.assign(el(), { setAttribute(k, v) { this[k] = v; } }) };
new Function('document', 'window', 'fetch', 'location', 'setInterval', 'setTimeout', 'getComputedStyle',
  'ResizeObserver', 'localStorage', 'navigator', `${src}\n;obsDailyButton();`)(
  doc2, window, () => new Promise(() => {}), { search: '', href: '' }, () => 0, () => 0, () => ({}), undefined,
  window.localStorage, { userAgent: 'node' });
const btn = made.find((n) => n.textContent === 'daily');
assert.ok(btn, 'the daily button is appended');
assert.strictEqual(btn.className, 'obs-btn obs-daily', 'the card buttons\' neutral style, not connect\'s accent');

console.log('ok');
