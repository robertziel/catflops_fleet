/* EACH CHOOSER ROW SAYS WHETHER ITS CHECKPOINT STARTS — checked where the page builds the row.
 *
 * `/lb/status.start_rates` (vllm_lb/start_rates.py): per row (`<id>` / `<id>@uncensored`), of the LB's launches of
 * that checkpoint over the last 7 UTC days, how many reached serving (`ok` of `n`) and each day's [ok, n] — null on
 * a day with no launch. The row draws the days as tiny bars, one slot a day: a day with launches is one bar, the
 * share that served green from the base and the share that failed red above it (0 % is an all-red bar, never an
 * empty slot); a day without a launch is an empty slot — no bar, never 0 % — and the window's figures sit under them:
 * "1/6 · 17 %". A row with no launch says "no starts in 7 d"; starts stopped before they served (no verdict) are
 * named, never counted as either; a log the LB cannot read says so ("starts unknown"), never "no starts". An LB that
 * publishes no `start_rates` gets no block at all.
 *
 *   node tests/start_rates.node.js      (tests/test_start_rates_page.py runs this when node exists)
 */
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const STATIC = path.join(__dirname, '..', 'vllm_lb', 'static');
const src = fs.readFileSync(path.join(STATIC, 'gateway.js'), 'utf8');
const css = fs.readFileSync(path.join(STATIC, 'gateway.css'), 'utf8');

function load() {
  const byId = {};
  const el = (id) => ({ id, attrs: {}, style: {}, hidden: false, innerHTML: '', textContent: '', value: '', dataset: {},
    disabled: false, addEventListener() {}, setAttribute(k, v) { this.attrs[k] = String(v); },
    removeAttribute(k) { delete this.attrs[k]; }, classList: { add() {}, remove() {}, toggle() {}, contains: () => false },
    append() {}, querySelectorAll: () => [], querySelector: () => null, observe() {}, closest: () => null,
    getBoundingClientRect: () => ({}), contains: () => false, focus() {}, blur() {} });
  const document = { getElementById: (id) => (byId[id] || (byId[id] = el(id))), createElementNS: () => el(''),
    createElement: () => el(''), addEventListener() {}, querySelectorAll: () => [], querySelector: () => null,
    activeElement: null, body: { style: {}, classList: { toggle() {} } },
    documentElement: { style: {}, classList: { toggle() {} } } };
  const window = { addEventListener() {}, innerWidth: 1280, innerHeight: 800, matchMedia: () => ({
    matches: false, addEventListener() {} }), localStorage: { getItem: () => null, setItem() {} },
    requestAnimationFrame: () => 0 };
  const api = new Function('document', 'window', 'fetch', 'location', 'setInterval', 'setTimeout', 'clearTimeout',
    'getComputedStyle', 'ResizeObserver', 'localStorage', 'navigator', 'requestAnimationFrame',
    `${src}\n;return { startsOf: typeof startsOf === 'function' ? startsOf : null, renderModelChoice, pickerRowText };`)(
    document, window, () => new Promise(() => {}), { search: '', href: '' }, () => 0, () => 0, () => 0,
    () => ({}), undefined, window.localStorage, { userAgent: 'node' }, () => 0);
  return { api, byId };
}

const FLASH = 'qwen3.8-flash-next-hibrid48';
const FLASH_SGLANG = 'qwen3.8-flash-next-sglang';
const DENSE = 'qwen3.8-27b-nvidia-nvfp4';
const TWIN = { path: 'x_uncensored', default: true };
const PROFILE = { engine: 'vllm', speculative: true, num_speculative_tokens: 4, speculative_method: 'mtp' };
const MODELS = [
  { id: FLASH, label: 'Qwen3.8 Flash-Next hibrid48 · MoE NVFP4', size_gib: 98.1, uncensored: TWIN,
    launch: { cluster: PROFILE } },
  { id: FLASH_SGLANG, label: 'Qwen3.8 Flash-Next hibrid48 · MoE NVFP4 · SGLang NEXTN', size_gib: 98.1, uncensored: TWIN,
    launch: { cluster: { ...PROFILE, engine: 'sglang', speculative_method: 'nextn' } } },
  { id: DENSE, label: 'Qwen3.8 27B · dense NVFP4', size_gib: 20.4, uncensored: TWIN, launch: { cluster: PROFILE } },
];
const ROWS = [FLASH, `${FLASH}@uncensored`, FLASH_SGLANG, `${FLASH_SGLANG}@uncensored`, DENSE, `${DENSE}@uncensored`];
const C = { configured: true, state: 'running', held: true, models: MODELS, limits: { uncensored: [0, 1] }, setups: {},
  desired: { mode: 'cluster', generation: 193, model_choice: FLASH, uncensored: true },
  effective: { model_choice: FLASH, uncensored: true }, boxes: {} };
const n = null;
//: box-c's own, 2026-10-05..11 UTC (tests/test_start_rates.py pins the same off the boot log).
const RATES = { tz: 'UTC', days: ['2026-10-05', '2026-10-06', '2026-10-07', '2026-10-08', '2026-10-09', '2026-10-10', '2026-10-11'],
  unread: 0, malformed: 0, error: '', rows: {
    [FLASH]: { ok: 1, n: 2, stopped: 1, refused: 0, days: [[1, 1], n, n, n, n, n, [0, 1]] },
    [`${FLASH}@uncensored`]: { ok: 1, n: 6, stopped: 0, refused: 0, days: [n, n, n, n, n, [0, 2], [1, 4]] },
    [FLASH_SGLANG]: { ok: 12, n: 26, stopped: 4, refused: 0, days: [[7, 20], [2, 2], n, n, [1, 1], [2, 3], n] },
    [`${FLASH_SGLANG}@uncensored`]: { ok: 4, n: 6, stopped: 1, refused: 0, days: [n, n, n, n, n, [2, 4], [2, 2]] },
    [`${DENSE}@uncensored`]: { ok: 0, n: 0, stopped: 1, refused: 1, days: [n, n, n, n, n, n, n] },
  } };

/* The listbox's rows and each one's starts block, parsed from what the page wrote. */
const OPTION = /<div class="mp-option([^"]*)" id="cluster-model-opt-(\d+)" role="option"[^>]* data-value="([^"]*)" data-index="\d+" title="([^"]*)">/g;
function rowsOf(list) {
  const ms = [...list.innerHTML.matchAll(OPTION)];
  return ms.map((m, i) => {
    const html = list.innerHTML.slice(m.index, i + 1 < ms.length ? ms[i + 1].index : list.innerHTML.length);
    const block = html.match(/<span class="mp-starts ([a-z]+)" title="([^"]*)">([\s\S]*?)<span class="mp-starts-n">([^<]*)<\/span><\/span>/);
    const days = block && block[3].match(/<svg class="start-bars"[^>]* data-days="([^"]*)"/);
    return { classes: m[1].trim(), value: m[3], title: m[4], html, tone: block && block[1], hover: block && block[2],
      label: block && block[4], days: days && days[1], graph: block ? block[3] : '' };
  });
}

let checks = 0;
const { api, byId } = load();
assert.ok(api.startsOf, 'gateway.js has no startsOf');

{ // THE FIGURES: served/launched and the share, in the tone of the verdict; days without a launch are gaps.
  const twin = api.startsOf(RATES, `${FLASH}@uncensored`);
  assert.strictEqual(twin.label, '1/6 · 17 %');
  assert.strictEqual(twin.tone, 'warn');
  assert.deepStrictEqual(twin.cells, [n, n, n, n, n, [0, 2], [1, 4]], 'a day is its own; no day is filled in');
  assert.strictEqual(api.startsOf(RATES, FLASH).label, '1/2 · 50 %');
  assert.deepStrictEqual(api.startsOf(RATES, FLASH).cells, [[1, 1], n, n, n, n, n, [0, 1]]);
  const all = { ...RATES, rows: { a: { ok: 3, n: 3, stopped: 0, refused: 0, days: [n, n, n, n, n, [1, 1], [2, 2]] },
    b: { ok: 0, n: 2, stopped: 0, refused: 0, days: [n, n, n, n, n, n, [0, 2]] } } };
  assert.deepStrictEqual([api.startsOf(all, 'a').tone, api.startsOf(all, 'a').label], ['ok', '3/3 · 100 %']);
  assert.deepStrictEqual([api.startsOf(all, 'b').tone, api.startsOf(all, 'b').label], ['bad', '0/2 · 0 %']);
  // The hover names each day that had a launch, and what was counted apart.
  assert.strictEqual(api.startsOf(RATES, FLASH_SGLANG).title,
    'starts · 7 d UTC · 10-05 7/20 · 10-06 2/2 · 10-09 1/1 · 10-10 2/3 · 4 stopped');
  checks += 9;
}

{ // NO LAUNCH, NO RATE: "no starts in 7 d" and an empty axis; a start stopped before it served is named, not counted.
  const none = api.startsOf(RATES, DENSE);
  assert.deepStrictEqual([none.label, none.tone], ['no starts in 7 d', 'none']);
  assert.deepStrictEqual(none.cells, [n, n, n, n, n, n, n]);
  const apart = api.startsOf(RATES, `${DENSE}@uncensored`);
  assert.deepStrictEqual([apart.label, apart.tone], ['1 stopped · 1 refused', 'none']);
  // A log the LB could not read: unknown, with its reason — never "no starts", and no graph to misread.
  const off = api.startsOf({ ...RATES, rows: {}, error: 'the boot log is off (LB_PERF=0): no start is recorded' }, FLASH);
  assert.deepStrictEqual([off.label, off.tone, off.cells], ['starts unknown', 'unknown', null]);
  assert.ok(off.title.includes('LB_PERF=0'), off.title);
  // An LB that publishes no start_rates: nothing to draw.
  assert.strictEqual(api.startsOf(undefined, FLASH), null);
  checks += 7;
}

{ // ON THE ROWS: one block per row, beside the name and the chips: the day bars over the figures.
  api.renderModelChoice(C, { start_rates: RATES }, true);
  const rows = rowsOf(byId['cluster-model-list']);
  assert.deepStrictEqual(rows.map((r) => r.value), ROWS);
  assert.deepStrictEqual(rows.map((r) => r.label),
    ['1/2 · 50 %', '1/6 · 17 %', '12/26 · 46 %', '4/6 · 67 %', 'no starts in 7 d', '1 stopped · 1 refused']);
  assert.deepStrictEqual(rows.map((r) => r.tone), ['warn', 'warn', 'warn', 'warn', 'none', 'none']);
  assert.deepStrictEqual(rows.map((r) => r.days), ['1/1,-,-,-,-,-,0/1', '-,-,-,-,-,0/2,1/4', '7/20,2/2,-,-,1/1,2/3,-',
    '-,-,-,-,-,2/4,2/2', '-,-,-,-,-,-,-', '-,-,-,-,-,-,-'], 'one slot a day, oldest first');
  // A day's bar: the share that served from the base (green), the share that failed above it (red). 0 % is an
  // all-red bar — a mark, not an empty slot; a day with no launch has no bar at all, only its empty slot.
  const bars = (graph) => [...graph.matchAll(/<rect class="(sb-ok|sb-bad|sb-none)" x="[\d.]+" y="([\d.]+)" width="[\d.]+" height="([\d.]+)"/g)]
    .map((b) => `${b[1]}:${b[2]}+${b[3]}`);
  assert.deepStrictEqual(bars(rows[0].graph), ['sb-ok:0.0+16.0', 'sb-none:15.0+1.0', 'sb-none:15.0+1.0', 'sb-none:15.0+1.0',
    'sb-none:15.0+1.0', 'sb-none:15.0+1.0', 'sb-bad:0.0+16.0'], rows[0].graph);
  assert.deepStrictEqual(bars(rows[1].graph).slice(5), ['sb-bad:0.0+16.0', 'sb-bad:0.0+11.0', 'sb-ok:12.0+4.0'], rows[1].graph);
  assert.ok(/<title>10-11 · 1\/4 served<\/title>/.test(rows[1].graph), 'each day\'s counts are its own hover');
  assert.ok(rows.every((r) => r.classes.includes('has-starts')), rows.map((r) => r.classes).join(' / '));
  // The row's one line (its title, the hidden select's option) carries the same figures.
  assert.strictEqual(rows[1].title.slice(-' · starts 1/6 · 17 %'.length), ' · starts 1/6 · 17 %');
  assert.ok(byId['cluster-model'].innerHTML.includes('· starts no starts in 7 d') === false, 'no doubled word');
  assert.ok(rows[4].title.endsWith(' · no starts in 7 d'), rows[4].title);
  // A screen reader hears what the figures are of.
  assert.ok(rows[0].html.includes('<span class="vh">starts, last 7 days: </span>'), rows[0].html);
  checks += 12;
}

{ // AN OLDER LB (no start_rates): the rows are exactly what they were.
  const { api: older, byId: ids } = load();
  older.renderModelChoice(C, {}, true);
  const rows = rowsOf(ids['cluster-model-list']);
  assert.ok(rows.length === ROWS.length && rows.every((r) => r.label === null && !r.classes.includes('has-starts')));
  assert.ok(!ids['cluster-model-list'].innerHTML.includes('mp-starts'));
  checks += 2;
}

{ // THE LOOK: the page's own tokens — served green, failed red, the figures in the verdict's tone (ok / warn / bad).
  const rule = (sel) => { const at = css.indexOf(sel); return at < 0 ? '' : css.slice(at, css.indexOf('}', at)); };
  assert.ok(rule('.mp-starts.ok').includes('var(--ok)') && rule('.mp-starts.warn').includes('var(--warn)')
    && rule('.mp-starts.bad').includes('var(--bad)'), 'the tones are the page\'s');
  assert.ok(rule('.start-bars .sb-ok').includes('var(--ok)') && rule('.start-bars .sb-bad').includes('var(--bad)'));
  const block = css.slice(css.indexOf('.mp-starts'), css.lastIndexOf('.mp-starts') + 200).replace(/\/\*[\s\S]*?\*\//g, '');
  assert.ok(!/#[0-9a-f]{3,8}\b/i.test(block), 'no literal colour in the starts rules');
  checks += 3;
}

console.log(`ok — ${checks} checks`);
