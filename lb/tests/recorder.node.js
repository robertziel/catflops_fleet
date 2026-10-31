/* THE FLIGHT RECORDER as drawn (static/recorder.js), under node: the strip under each machine card, the open recorder
 * and the expand view — built from /lb/status `hardware[box].recorder` (vllm_lb/recorder.py `summary`) and the hour
 * of GET /lb/hardware/<box>/recorder.
 *
 * What it pins:
 *   1. the strip is ONE line of figures: the power peak, GPU and CPU temperature, net in/out, containers;
 *   2. a figure the box does not send is "n/a" (strip) or "—" (grid) with its reason in the title — never 0: the
 *      1.5.0 sender broadcasts no recorder block, and its strip is n/a everywhere but says why;
 *   3. CONTAINERS · count · total RAM, the top rows by RAM and "+k more"; every row once "more" is asked for; the
 *      containers a capped sender did not send are counted in the head and said ("+k smaller not sent"); the rows
 *      are a table of name · RAM · image, the image cut with its whole ref in the title;
 *   4. the charts are the panel's own tiles (gateway.js `tile` → sparkline), one per figure the hour HAS — a figure
 *      the box never sent has no chart, only its "—" with why;
 *   5. the strip is a button with aria-expanded/aria-controls; the open state is the browser's (RecStore), and a
 *      storage that throws costs only the memory of it;
 *   6. a container's name is text: `<script>` is escaped;
 *   7. every figure in its unit, formatted by the table's `fmt` (a 0.14 Mb/s link is not "0"); THROTTLE is its
 *      decoded reasons, "none" for a clear mask;
 *   8. each figure ONCE: charts closed → the grid; charts open → the charts (their headlines carry the values);
 *      the dialog → the hour's and the ten minutes' charts, no grid;
 *   9. one power reading is one peak and the hour's mean (and "package" when the sensor is the APU package); two
 *      readings are kept apart, named;
 *  10. a chart block that does not yet hold its window says how much: "filling · 2 of 60 min".
 *
 *   node tests/recorder.node.js      (tests/test_recorder_page.py runs this when node exists)
 */
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const STATIC = path.join(__dirname, '..', 'vllm_lb', 'static');
// palette.js first: the colours this script computes are read from the style's tokens (tests/palette_env.js).
const src = require('./palette_env').withPalette(fs.readFileSync(path.join(STATIC, 'gateway.js'), 'utf8') + '\n'
  + fs.readFileSync(path.join(STATIC, 'recorder.js'), 'utf8'));
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
  `${src}\n;return { recStripHtml, recBodyHtml, recDialogHtml, RecStore, recFmt };`)(
  document, window, () => new Promise(() => {}), { search: '', href: '' }, () => 0, () => 0,
  () => ({}), undefined, window.localStorage, { userAgent: 'node' });

/** The figure table as recorder.py `summary` serves it (fleet_telemetry/figures.py FIGURES_TABLE). */
const LABELS = [['gpu_temp_c', 'GPU TEMP', '°C', 'int'], ['board_temp_c', 'BOARD MAX', '°C', 'int'],
  ['cpu_temp_c', 'CPU TEMP', '°C', 'int'], ['fan_rpm', 'FAN SPEED', 'rpm', 'int'], ['gpu_power_w', 'GPU POWER', 'W', 'rate'],
  ['throttle', 'THROTTLE', '', 'flags'], ['gpu_mhz', 'GPU CLOCK', 'MHz', 'int'], ['cpu_mhz', 'CPU CLOCK', 'MHz', 'int'],
  ['cpu_pct', 'CPU UTIL', '%', 'pct'], ['net_in_mbps', 'NET IN', 'Mb/s', 'rate'], ['net_out_mbps', 'NET OUT', 'Mb/s', 'rate'],
].map(([key, label, unit, fmt]) => ({ key, label, unit, fmt }));

const containers = (n) => Array.from({ length: n }, (_, i) => ({ name: `svc-${i}`, image: `img/${i}:1`, ram_mb: 3584 - 400 * i }));

/** A box with the recorder block, as recorder.py `summary` serves it: an NVIDIA card (inst and avg distinct). */
const FULL = {
  window_s: 3600, held_s: 3600, labels: LABELS,
  figures: { gpu_temp_c: 44, board_temp_c: 47.0, cpu_temp_c: 47.0, fan_rpm: null, gpu_power_w: 9.0, throttle: 0,
             gpu_mhz: 2190, cpu_mhz: 3354, cpu_pct: 5.9, net_in_mbps: 0.14, net_out_mbps: 0 },
  why: { fan_rpm: 'no fan tachometer on this box' },
  throttle_reasons: [],
  power: { peak_inst_w: 29, peak_avg_w: 34, mean_w: 12.3, input_rail_w: null, one_reading: false, scope: null,
           input_rail_why: 'no sender reports an input-rail (PSU side) reading' },
  containers: { count: 7, total_ram_mb: 5120, top: containers(6), more: 1 },
};

/** An AMD APU: one power sensor (the package's PPT), no fan tach, no throttle mask. */
const APU = Object.assign({}, FULL, {
  figures: Object.assign({}, FULL.figures, { throttle: null, gpu_power_w: 78.1, gpu_mhz: 600, cpu_mhz: 4730, cpu_pct: 19.2 }),
  why: { fan_rpm: 'no fan tachometer on this box', throttle: 'amdgpu exposes no throttle mask' },
  throttle_reasons: null,
  power: { peak_inst_w: 119.08, peak_avg_w: 119.08, mean_w: 88.6, input_rail_w: null, one_reading: true,
           scope: 'package', input_rail_why: 'no input rail' },
});

/** The 1.5.0 sender: CPU util only, every other figure absent with the reason. */
const NO_DEEP = 'this box\'s sender broadcasts no recorder block (`deep`) — it predates it';
const OLD = {
  window_s: 3600, held_s: 120, labels: LABELS,
  figures: Object.fromEntries(LABELS.map((l) => [l.key, l.key === 'cpu_pct' ? 5.9 : null])),
  why: Object.fromEntries(LABELS.filter((l) => l.key !== 'cpu_pct').map((l) => [l.key, NO_DEEP])),
  throttle_reasons: null,
  power: { peak_inst_w: null, peak_avg_w: null, mean_w: null, input_rail_w: null, input_rail_why: 'no input rail',
           one_reading: false, scope: null, why: NO_DEEP },
  containers: null, containers_why: 'this box\'s sender broadcasts no `containers` list — it predates it',
};

/** The hour as GET …/recorder serves it: every figure in it but `missing`. */
const hourOf = (missing, heldS) => {
  const t = Array.from({ length: 60 }, (_, i) => i * 60);
  const series = Object.fromEntries(LABELS.map((l) => [l.key, {
    mean: t.map((_, i) => (missing.includes(l.key) ? null : i % 7)), max: t.map(() => null) }]));
  const fine = { t: [1, 2], series: Object.fromEntries(LABELS.map((l) => [l.key, missing.includes(l.key) ? [null, null] : [1, 2]])) };
  return { data: { coarse: { t, series }, fine, bucket_s: 60, fine_s: 600, held_s: heldS == null ? 3600 : heldS,
                   containers: { all: containers(7) } } };
};

const text = (html) => html.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim();
const count = (s, re) => (s.match(re) || []).length;
let checks = 0;
const ok = (cond, msg) => { assert.ok(cond, msg); checks++; };

// 1. the strip: one line of figures
{
  const s = api.recStripHtml('box-a', FULL, false);
  const t = text(s);
  ok(t.startsWith('⏺ 1h'), `the strip opens with the recorder's window: ${t}`);
  ok(t.includes('peak 29 W inst · 34 W avg'), `two readings, named: ${t}`);
  ok(t.includes('GPU 44°C') && t.includes('CPU 47°C'), `the temperatures: ${t}`);
  ok(t.includes('net 0.14↓/0↑ Mb/s'), `net in/out, a sub-1 rate not rounded to 0: ${t}`);
  ok(t.includes('7 containers 5.0G'), `containers and their RAM: ${t}`);
  ok(/<button type="button" class="rec-strip" aria-expanded="false" aria-controls="rec-box-a"/.test(s), 'a button, collapsed');
  ok(/aria-expanded="true"/.test(api.recStripHtml('box-a', FULL, true)), 'open says so');
}

// 2. absent figures: n/a with why, never 0
{
  const s = api.recStripHtml('box-a', OLD, false);
  const t = text(s);
  ok(t.includes('power n/a') && t.includes('GPU n/a') && t.includes('CPU n/a') && t.includes('net n/a')
     && t.includes('containers n/a'), `every absent figure is n/a: ${t}`);
  ok(!/\b0 ?W\b|GPU 0|CPU 0/.test(t), `nothing absent is a zero: ${t}`);
  ok(s.includes('predates it'), 'the strip\'s title says why');
  const b = api.recBodyHtml('box-a', OLD, { open: true, charts: false, more: false }, null);
  ok(text(b).includes('CPU UTIL 5.9%'), 'CPU util is live from the 1.5.0 sender, in %');
  const gpuTemp = b.match(/<div class="rec-fig" title="([^"]*)"><span>GPU TEMP<\/span><b class="rec-na">—<\/b>/);
  ok(gpuTemp && gpuTemp[1].includes('predates'), 'a grid figure with no reading is — with its reason');
  ok(text(b).includes('peak n/a') && text(b).includes('input rail n/a'), 'the power line says n/a');
  ok(text(b).includes('holds 2m 0s'), 'how much of the hour the recorder holds');
  const fan = api.recBodyHtml('box-a', FULL, { open: true }, null).match(/title="([^"]*)"><span>FAN SPEED<\/span><b class="rec-na">—/);
  ok(fan && fan[1] === 'no fan tachometer on this box', 'a box with no fan tach: — and why');
}

// 3. containers: count, RAM, top N, "+k more"; a table of name · RAM · image
{
  const b = api.recBodyHtml('box-a', FULL, { open: true, charts: false, more: false }, null);
  const t = text(b);
  ok(t.includes('CONTAINERS · 7 · 5.0G RAM'), `the containers head: ${t}`);
  ok(count(b, /<li>/g) === 6, 'the top six');
  ok(t.includes('svc-0 3.5G img/0:1'), `a row: name, RAM, image: ${t}`);
  ok(/<li><span class="rec-name"[^>]*>svc-0<\/span><b class="rec-cram">3.5G<\/b><span class="rec-img" title="img\/0:1">img\/0:1<\/span><\/li>/.test(b),
     'a row is three cells in column order: name · RAM · image (whole in the title)');
  ok(/<ul class="rec-list rec-ctab">/.test(b), 'the rows are one table');
  ok(/class="rec-more"[^>]*aria-expanded="false">\+1 more</.test(b), '+1 more');
  const all = { data: { containers: { all: containers(7) } } };
  const more = api.recBodyHtml('box-a', FULL, { open: true, charts: false, more: true }, all);
  ok(count(more, /<li>/g) === 7, 'every container once more is asked for');
  ok(text(more).includes('FLIGHT RECORDER · 1H') && /class="rec-expand"/.test(more), 'the head and its expand');
}

// 3b. a sender that caps its list (its top N by RAM): the head counts every container, the rest is said, not drawn
{
  const capped = Object.assign({}, FULL, { containers: { count: 147, total_ram_mb: 90000, top: containers(6), more: 1,
                                                         unlisted: 140 } });
  const b = api.recBodyHtml('box-a', capped, { open: true, charts: false, more: false }, null);
  const t = text(b);
  ok(t.includes('CONTAINERS · 147'), `the head counts every container: ${t}`);
  ok(/class="rec-unlisted rec-na"[^>]*>\+140 smaller not sent</.test(b), `the containers the datagram did not carry: ${b}`);
  const plain = api.recBodyHtml('box-a', FULL, { open: true, charts: false, more: false }, null);
  ok(!/rec-unlisted/.test(plain), 'a whole list says nothing more');
}

// 4. charts are the panel's tiles — one per figure the hour has
{
  const hour = hourOf(['fan_rpm']);
  const closed = api.recBodyHtml('box-a', FULL, { open: true, charts: false }, hour);
  ok(/<details class="rec-charts"[^>]*><summary>charts<\/summary><\/details>/.test(closed), 'closed charts draw nothing');
  const b = api.recBodyHtml('box-a', FULL, { open: true, charts: true }, hour);
  ok(count(b, /<div class="(rec-plain-tile )?tile/g) === LABELS.length - 1, 'one tile per figure the hour has');
  ok(count(b, /<svg class="sparkline/g) === LABELS.length - 1, 'each with its sparkline');
  ok(!/t-label">FAN SPEED</.test(b), 'a figure the box never sent has no chart (an empty graph is noise)');
  const fan = b.match(/<div class="rec-fig" title="([^"]*)"><span>FAN SPEED<\/span><b class="rec-na">—/);
  ok(fan && fan[1] === 'no fan tachometer on this box', 'it stays a "—" beside the charts, with why');
  const tileOf = (label) => {
    const i = b.indexOf(`t-label">${label}<`);
    return b.slice(Math.max(b.lastIndexOf('<div class="tile', i), b.lastIndexOf('<div class="rec-plain-tile', i)), i);
  };
  ok(/rec-plain-tile/.test(tileOf('GPU TEMP')), 'no ceiling known: the temperature is drawn plain, never red against its own peak');
  ok(!/rec-plain/.test(tileOf('CPU UTIL')), 'a percent keeps the load scale');
  ok(text(api.recBodyHtml('box-a', FULL, { open: true, charts: true }, { error: 'HTTP 503: no listener' }))
     .includes('HTTP 503: no listener'), 'a failed fetch says why where the charts would be');
  ok(text(api.recBodyHtml('box-a', FULL, { open: true, charts: true }, null)).includes('loading'), 'not yet: loading');
  const d = api.recDialogHtml('box-a', FULL, hour);
  ok(text(d).includes('LAST HOUR · 60 s BUCKETS') && text(d).includes('LAST 10 MIN · EVERY SAMPLE'), 'the expand view: both tiers');
  ok(count(d, /<li>/g) === 7 && !/rec-more/.test(d), 'the expand view lists every container');
}

// 5. the browser keeps the open state; a storage that throws costs only the memory
{
  const mem = {};
  const store = new api.RecStore({ getItem: (k) => mem[k] || null, setItem: (k, v) => { mem[k] = v; } });
  ok(store.get('box-a').open === false, 'collapsed by default');
  ok(store.set('box-a', { open: true }) === true && JSON.parse(mem['lb.recorder'])['box-a'].open === true, 'kept');
  ok(store.get('box-a').open === true && store.get('box-b').open === false, 'per box');
  const broken = new api.RecStore({ getItem() { throw new Error('denied'); }, setItem() { throw new Error('denied'); } });
  ok(broken.set('box-a', { open: true }) === false, 'the browser kept nothing, and says so');
  ok(broken.get('box-a').open === true, 'it still opens for the visit');
}

// 6. a container's name is text
{
  const evil = Object.assign({}, FULL, { containers: { count: 1, total_ram_mb: 1, more: 0,
    top: [{ name: '<script>x</script>', image: '"><img>', ram_mb: 1 }] } });
  const b = api.recBodyHtml('box-a', evil, { open: true }, null);
  ok(!b.includes('<script>') && !b.includes('"><img>'), 'escaped');
}

// 7. every figure in its unit, by the table's fmt
{
  const f = (v, fmt) => api.recFmt(v, fmt);
  ok(f(136.4, 'rate') === '136' && f(10, 'rate') === '10', `≥10: whole: ${f(136.4, 'rate')}`);
  ok(f(4.25, 'rate') === '4.3' && f(1, 'rate') === '1.0', `1–10: one decimal: ${f(4.25, 'rate')}`);
  ok(f(0.14, 'rate') === '0.14' && f(0.004, 'rate') === '<0.01', `<1: two decimals: ${f(0.14, 'rate')} ${f(0.004, 'rate')}`);
  ok(f(0, 'rate') === '0', 'a true zero is "0"');
  ok(f(null, 'rate') === null && f(NaN, 'int') === null, 'no reading: null, never 0');
  ok(f(4729.6, 'int') === '4730' && f(19.24, 'pct') === '19.2', 'whole and one-decimal');
  const b = text(api.recBodyHtml('box-a', APU, { open: true, charts: false }, null));
  ok(b.includes('GPU CLOCK 600 MHz') && b.includes('CPU CLOCK 4730 MHz') && b.includes('CPU UTIL 19.2%'), `units: ${b}`);
  ok(b.includes('NET IN 0.14 Mb/s') && b.includes('NET OUT 0 Mb/s'), `rates: ${b}`);
  ok(text(api.recBodyHtml('box-a', FULL, { open: true }, null)).includes('THROTTLE none'), 'a clear mask: none');
  const hot = Object.assign({}, FULL, { figures: Object.assign({}, FULL.figures, { throttle: 85 }),
                                        throttle_reasons: ['sw_power_cap', 'hw_slowdown'] });
  ok(text(api.recBodyHtml('box-a', hot, { open: true }, null)).includes('THROTTLE sw_power_cap, hw_slowdown'), 'the reasons, by name');
  const amd = api.recBodyHtml('box-a', APU, { open: true }, null).match(/title="([^"]*)"><span>THROTTLE<\/span><b class="rec-na">—/);
  ok(amd && amd[1].includes('amdgpu'), 'no mask: — with why');
  const tiles = api.recBodyHtml('box-a', hot, { open: true, charts: true }, hourOf([]));
  ok(/t-label">THROTTLE<\/span>\s*<b class="t-val"[^>]*>sw_power_cap, hw_slowdown/.test(tiles), 'the throttle chart\'s headline is its reasons');
}

// 8. each figure once
{
  const hour = hourOf(['fan_rpm', 'throttle']);
  const closed = api.recBodyHtml('box-a', APU, { open: true, charts: false }, hour);
  ok(count(closed, /<span>GPU TEMP<\/span>/g) === 1 && !/class="(rec-plain-tile )?tile/.test(closed), 'charts closed: the grid');
  const open = api.recBodyHtml('box-a', APU, { open: true, charts: true }, hour);
  ok(count(open, />GPU TEMP</g) === 1, `charts open: GPU TEMP once (its chart): ${count(open, />GPU TEMP</g)}`);
  ok(count(open, />FAN SPEED</g) === 1 && count(open, />THROTTLE</g) === 1, 'the absent ones once, as —');
  const d = api.recDialogHtml('box-a', APU, hour);
  ok(count(d, />GPU TEMP</g) === 2, `the dialog: the hour's chart and the ten minutes' chart, no grid: ${count(d, />GPU TEMP</g)}`);
  ok(count(d, />FAN SPEED</g) === 1, 'an absent figure once in the dialog');
}

// 9. power: one reading is one peak and the mean; two are named
{
  const s = text(api.recStripHtml('box-a', APU, false));
  ok(s.includes('peak 119 W') && !s.includes('avg') && !s.includes('inst'), `one reading, one peak: ${s}`);
  const b = text(api.recBodyHtml('box-a', APU, { open: true }, null));
  ok(b.includes('GPU POWER (package) peak 119 W · mean 89 W'), `the package's peak and the hour's mean: ${b}`);
  const n = text(api.recBodyHtml('box-a', FULL, { open: true }, null));
  ok(n.includes('GPU POWER peak 29 W inst · 34 W avg · mean 12 W'), `two readings, named: ${n}`);
}

// 10. a chart block that does not yet hold its window says so
{
  const young = Object.assign({}, APU, { held_s: 103 });
  const b = text(api.recBodyHtml('box-a', young, { open: true, charts: true }, hourOf([], 103)));
  ok(b.includes('filling · 2 of 60 min'), `the hour filling: ${b}`);
  const d = text(api.recDialogHtml('box-a', young, hourOf([], 103)));
  ok(d.includes('filling · 2 of 60 min') && d.includes('filling · 2 of 10 min'), `both blocks: ${d}`);
  const full = text(api.recDialogHtml('box-a', APU, hourOf([])));
  ok(!full.includes('filling'), 'a full hour says nothing');
  const mid = text(api.recDialogHtml('box-a', Object.assign({}, APU, { held_s: 900 }), hourOf([], 900)));
  ok(mid.includes('filling · 15 of 60 min') && !mid.includes('of 10 min'), `ten minutes held: that block is full: ${mid}`);
}

console.log(`ok — recorder ${checks} checks`);
