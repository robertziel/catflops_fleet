/* A box's GPU rows are drawn AS GRAPHICS CARDS, in the CPU chip's visual language: gateway.js `gpuColumn`, under node.
 *
 * Operator, 2026-10-30, over a box's machine tile: "try to implement CPU like GPU icon, but make it GPU shape, fan is the
 * circle, on the side vRAM if applies". So each GPU row is a card — a board with a bracket and PCIe fingers along its
 * bottom edge — whose FAN is the ring the row used to be (the utilisation's arc around a hub with blades), with the
 * label and the utilisation as a tag on the one line under it, as the CPU has. "If applies" is the row's own data:
 * a row that carries its own VRAM (`vram_label`, hw.py `rows`: a discrete card) gets VRAM chips beside the fan, lit in
 * proportion to used / total; a row without (a unified box's GPU, whose memory is the MEM ring) gets none. The page
 * never asks which kind of box it draws (tests/test_hardware_guard.py rule 4).
 *
 * A reading not known is SHOWN as not known: the card dims, the tag is "—" with its reason, the name says "not
 * reported", no arc is drawn — never a 0 %. VRAM figures not known light no chip and say "—".
 *
 * The FAN TURNS at the GPU's utilisation (operator 2026-10-30: "make GPU rotate in svg based on usage"): only the
 * blades, about the fan's centre, their rotation speed linear in the utilisation; no reading = no spin, never a made-up
 * speed. The panel re-renders the card every poll, so the phase comes from a page clock (§7), not from the render.
 *
 *   node tests/gpu_glyph.node.js      (tests/test_gpu_glyph.py runs this when node exists)
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
  `${src}\n;return { gpuColumn, machineBody, loadColour, fanSpin, FAN_SPIN };`)(
  document, window, () => new Promise(() => {}), { search: '', href: '' }, () => 0, () => 0,
  () => ({}), undefined, window.localStorage, { userAgent: 'node' });

// ── rows as /lb/status `hardware[box].rows` carries them ───────────────────────────────────────────
const card = (i, pct, used, total) => ({ label: `GPU ${i}`, card: i, box_gpu: false, use_pct: pct,
  history: [pct, pct], vram_label: `VRAM ${i}`, vram_used_mb: used, vram_total_mb: total });
const boxRow = (pct) => ({ label: 'GPU', card: null, box_gpu: true, use_pct: pct, history: [pct],
  vram_label: null, vram_used_mb: null, vram_total_mb: null });

// ── reading the glyph the way it is drawn ──────────────────────────────────────────────────────────
const attrs = (tag) => Object.fromEntries([...tag.matchAll(/([\w-]+)="([^"]*)"/g)].map(([, k, v]) => [k, v]));
const num = (a, k) => Number(a[k]);
const rectsIn = (svg, cls) => {
  const g = (svg.match(new RegExp(`<g class="${cls}(?: [^"]*)?"[^>]*>([\\s\\S]*?)</g>`)) || [])[1];
  return g === undefined ? null : [...g.matchAll(/<rect\b[^>]*>/g)].map(([t]) => attrs(t));
};
/** One GPU column: its card (the svg), and the label + tag line under it. */
function glyphOf(html) {
  const col = html.match(/<div class="gpu">(<svg class="gpu-card[^"]*"[\s\S]*?<\/svg>)<div class="gpu-line"([^>]*)>([\s\S]*?)<\/div><\/div>/);
  assert.ok(col, `no GPU column (a card, then ONE line) in ${html.slice(0, 200)}`);
  const [, svg, lineAttrs, line] = col;
  // The VRAM's name and share sit INSIDE the card beside the fan, over its one row of chips (operator 2026-10-30:
  // "move vram label on side of the card, and make one row memory graph").
  const vl = svg.match(/<text class="gpu-vram-label"[^>]*>([^<]*)<\/text>/);
  const vp = svg.match(/<text class="gpu-vram-pct( none)?"([^>]*)>([^<]*)<\/text>/);
  const vramLine = vl || vp ? { label: vl ? vl[1] : null, none: !!(vp && vp[1]),
    colour: vp ? (attrs(vp[2]).fill || null) : null, text: vp ? vp[3] : null } : null;
  const head = attrs(svg.match(/^<svg[^>]*>/)[0]);
  const tag = line.match(/<b class="gpu-tag( none)?"(?: style="--tag:([^"]+)")?(?: title="([^"]+)")?>([^<]*)<\/b>/);
  const fg = svg.match(/<circle class="gpu-fan-fg"[^>]*>/);
  return { svg, head, lineAttrs, cls: head.class,
    title: (svg.match(/<title>([^<]*)<\/title>/) || [])[1] || null,
    label: (line.match(/<span class="gpu-label">([^<]*)<\/span>/) || [])[1] || null,
    tagNone: !!(tag && tag[1]), tagColour: tag ? tag[2] || null : null, tagTitle: tag ? tag[3] || null : null,
    tagText: tag ? tag[4] : null, arc: fg ? attrs(fg[0]) : null,
    fanBg: /<circle class="gpu-fan-bg"/.test(svg), blades: (svg.match(/<g class="gpu-blades[^"]*"[^>]*>([\s\S]*?)<\/g>/) || [])[1],
    vram: rectsIn(svg, 'gpu-vram'), lit: rectsIn(svg, 'gpu-vram-lit'), vramLine };
}
/** The VRAM share the chips show: the lit area over every chip's area. */
const litShare = (g) => g.lit.reduce((a, r) => a + num(r, 'width') * num(r, 'height'), 0)
  / g.vram.reduce((a, r) => a + num(r, 'width') * num(r, 'height'), 0);

let checks = 0;

// 1. The card's SHAPE, for every row: the CPU's 5:4 frame (gateway.css sizes both alike), a board, a bracket on one
//    end, PCIe fingers along the board's bottom edge, a fan — the ring's track, a hub and blades — inside the board.
for (const r of [card(0, 31, 7270, 24576), boxRow(94), boxRow(null)]) {
  const g = glyphOf(api.gpuColumn(r, r.use_pct, 'why'));
  const what = `${r.label} at ${r.use_pct}`;
  const [, , W, H] = g.head.viewBox.split(' ').map(Number);
  assert.strictEqual(W / H, 5 / 4, `${what}: the frame is ${W}×${H}, not the CPU's 5:4`);
  assert.strictEqual(g.head.role, 'img', `${what}: not an image to a screen reader`);
  const board = attrs((g.svg.match(/<rect class="gpu-board"[^>]*>/) || assert.fail(`${what}: no board`))[0]);
  const bx = num(board, 'x'), by = num(board, 'y'), bw = num(board, 'width'), bh = num(board, 'height');
  assert.ok(bw > bh, `${what}: the board is ${bw}×${bh}, not a card's long shape`);
  const fingers = rectsIn(g.svg, 'gpu-fingers');
  assert.ok(fingers && fingers.length >= 8, `${what}: ${fingers && fingers.length} PCIe fingers`);
  for (const f of fingers) {
    assert.ok(Math.abs(num(f, 'y') - (by + bh)) < 0.01, `${what}: a finger is not on the board's bottom edge`);
    assert.ok(num(f, 'x') >= bx && num(f, 'x') + num(f, 'width') <= bx + bw, `${what}: a finger overhangs the board`);
    assert.ok(num(f, 'y') + num(f, 'height') <= H, `${what}: a finger leaves the frame`);
  }
  const bracket = attrs((g.svg.match(/<rect class="gpu-bracket"[^>]*>/) || assert.fail(`${what}: no bracket`))[0]);
  assert.ok(num(bracket, 'x') + num(bracket, 'width') <= bx, `${what}: the bracket is not at the board's end`);
  assert.ok(g.fanBg, `${what}: no fan track`);
  assert.ok(g.blades && (g.blades.match(/<ellipse/g) || []).length >= 5, `${what}: the fan has no blades`);
  const hub = attrs((g.svg.match(/<circle class="gpu-hub"[^>]*>/) || assert.fail(`${what}: no hub`))[0]);
  const fan = attrs(g.svg.match(/<circle class="gpu-fan-bg"[^>]*>/)[0]);
  const cx = num(fan, 'cx'), cy = num(fan, 'cy'), rr = num(fan, 'r');
  assert.ok(num(hub, 'cx') === cx && num(hub, 'cy') === cy, `${what}: the hub is off the fan's centre`);
  assert.ok(cx - rr >= bx && cx + rr <= bx + bw && cy - rr >= by && cy + rr <= by + bh, `${what}: the fan leaves the board`);
  checks += 9 + fingers.length;
}

// 2. A DISCRETE card: VRAM chips beside the fan, lit in proportion to ITS used / total (the CPU's lit cores), in the
//    load scale's colour at that share; the fan sits clear of them. Empty, full, and between.
for (const [used, total] of [[7270, 24576], [28058, 32607], [0, 24576], [24576, 24576], [1, 32607]]) {
  const g = glyphOf(api.gpuColumn(card(1, 5, used, total), 5, 'why'));
  const what = `VRAM ${used}/${total}`, share = used / total;
  assert.ok(g.vram && g.vram.length >= 4, `${what}: ${g.vram && g.vram.length} VRAM chips`);
  assert.ok(g.lit, `${what}: no lit group`);
  assert.ok(Math.abs(litShare(g) - share) < 0.005, `${what}: the chips show ${litShare(g).toFixed(3)}, not ${share.toFixed(3)}`);
  for (const l of g.lit) assert.strictEqual(l.fill, api.loadColour(share * 100), `${what}: a lit chip is ${l.fill}`);
  const fan = attrs(g.svg.match(/<circle class="gpu-fan-bg"[^>]*>/)[0]);
  const right = num(fan, 'cx') + num(fan, 'r');
  for (const c of g.vram) assert.ok(num(c, 'x') > right, `${what}: a VRAM chip under the fan`);
  // The VRAM line, right of the utilisation's, as the CPU's tag: "VRAM", the share used, in the chips' colour.
  assert.ok(g.vramLine, `${what}: no VRAM label beside the fan`);
  assert.strictEqual(g.vramLine.label, 'VRAM', `${what}: the VRAM label`);
  assert.strictEqual(g.vramLine.text, `${Math.round(share * 100)}%`, `${what}: the VRAM share says ${g.vramLine.text}`);
  assert.strictEqual(g.vramLine.colour, api.loadColour(share * 100), `${what}: the VRAM share is ${g.vramLine.colour}`);
  const ys = new Set(g.vram.map((c) => c.y));
  assert.strictEqual(ys.size, 1, `${what}: the VRAM graph is ${ys.size} rows, not one`);
  checks += 8 + g.lit.length + g.vram.length;
}

// 3. A UNIFIED box's GPU row carries no VRAM of its own (its memory is the MEM ring): no chip is drawn, and the fan
//    takes the board's middle.
{
  const g = glyphOf(api.gpuColumn(boxRow(94), 94, 'why'));
  assert.strictEqual(g.vram, null, 'a unified row draws VRAM chips');
  assert.strictEqual(g.lit, null, 'a unified row lights VRAM');
  assert.strictEqual(g.vramLine, null, 'a unified row has a VRAM line');
  assert.ok(!/VRAM/.test(g.head['aria-label']), `a unified row's name says VRAM: ${g.head['aria-label']}`);
  const board = attrs(g.svg.match(/<rect class="gpu-board"[^>]*>/)[0]);
  const fan = attrs(g.svg.match(/<circle class="gpu-fan-bg"[^>]*>/)[0]);
  assert.ok(Math.abs(num(fan, 'cx') - (num(board, 'x') + num(board, 'width') / 2)) < 0.01, 'the fan is off the board\'s middle');
  checks += 4;
}

// 4. The utilisation: the fan's arc is the share (of its own track), in `loadColour`; the tag says it in that colour;
//    the name says it — and the label, and the VRAM figure where the row has one. One value, one colour.
for (const [r, name] of [[card(0, 31, 7270, 24576), 'GPU 0 · 31 % · VRAM 7.1/24 GiB'],
  [card(1, 5, 28058, 32607), 'GPU 1 · 5 % · VRAM 27/32 GiB'], [boxRow(94), 'GPU · 94 %']]) {
  const g = glyphOf(api.gpuColumn(r, r.use_pct, 'why'));
  const what = r.label;
  assert.strictEqual(g.head['aria-label'], name, `${what}: its name`);
  assert.strictEqual(g.title, name, `${what}: its hover title`);
  assert.ok(/aria-hidden="true"/.test(g.lineAttrs), `${what}: the line is read twice (the card says it)`);
  assert.strictEqual(g.label, r.label, `${what}: the line's label`);
  assert.strictEqual(g.tagText, `${r.use_pct}%`, `${what}: the tag says ${g.tagText}`);
  assert.strictEqual(g.tagColour, api.loadColour(r.use_pct), `${what}: the tag is ${g.tagColour}`);
  assert.ok(g.arc, `${what}: no arc`);
  assert.strictEqual(g.arc.stroke, g.tagColour, `${what}: the arc is ${g.arc.stroke}, the tag ${g.tagColour}`);
  const [on, track] = g.arc['stroke-dasharray'].split(' ').map(Number);
  assert.ok(Math.abs(on / track - r.use_pct / 100) < 0.001, `${what}: the arc is ${on}/${track}`);
  assert.ok(Math.abs(track - 2 * Math.PI * num(g.arc, 'r')) < 0.01, `${what}: the track is not the fan's circle`);
  assert.ok(!/\bnone\b/.test(g.cls), `${what}: a known reading dimmed`);
  checks += 11;
}

// 5. NOT REPORTED — no utilisation for the row (a stale box, a broadcast without the GPU): the card dims, no arc, the
//    tag is "—" with its reason, the name says "not reported". Never a 0, never a colour.
for (const r of [boxRow(null), card(0, null, 7270, 24576)]) {
  const html = api.gpuColumn(r, null, 'no sample for 42s');
  const g = glyphOf(html), what = `${r.label} unknown`;
  assert.ok(/\bnone\b/.test(g.cls), `${what}: the card is not dimmed (${g.cls})`);
  assert.strictEqual(g.arc, null, `${what}: an arc drawn for no reading`);
  assert.strictEqual(g.tagText, '—', `${what}: the tag says ${g.tagText}`);
  assert.ok(g.tagNone && g.tagColour === null, `${what}: an unknown reading wears a colour`);
  assert.strictEqual(g.tagTitle, 'no sample for 42s', `${what}: the reason`);
  assert.ok(g.head['aria-label'].startsWith(`${r.label} · not reported`), `${what}: "${g.head['aria-label']}"`);
  assert.ok(!/\b0 ?%/.test(html), `${what}: a fake 0 %`);
  checks += 7;
}
// VRAM not reported on a card that has it: the chips are drawn (the card HAS VRAM), none lit, the figure says "—".
{
  const g = glyphOf(api.gpuColumn(card(0, 12, null, 24576), 12, 'why'));
  assert.ok(g.vram && g.vram.length, 'a card with VRAM lost its chips when the used figure is missing');
  assert.ok(!g.lit || g.lit.length === 0, 'VRAM lit with no used figure');
  assert.ok(/<g class="gpu-vram none">/.test(g.svg), 'VRAM not reported is not dimmed');
  assert.strictEqual(g.head['aria-label'], 'GPU 0 · 12 % · VRAM —/24 GiB', 'the name with no used figure');
  checks += 4;
}

// 6. SEVERAL ROWS, through machineBody: one card per row in the rows' order, each at its own reading; MEM stays a ring;
//    the grid is told how many glyph columns follow the CPU (`--rings`).
{
  const h = { host: 'h', age_s: 1, stale: false, cpu_percent: 4, ram_percent: 30, ram_used_mb: 28672,
    ram_total_mb: 96256, swap_used_mb: 0, rows: [card(0, 31, 7270, 24576), card(1, 5, 28058, 32607)],
    history: { cpu_pct: [4], ram_pct: [30] } };
  const html = api.machineBody({ hardware: { b: h } }, 'b');
  const labels = [...html.matchAll(/<svg class="gpu-card[^"]*"[^>]*aria-label="([^"]+)"/g)].map(([, a]) => a);
  assert.deepStrictEqual(labels, ['GPU 0 · 31 % · VRAM 7.1/24 GiB', 'GPU 1 · 5 % · VRAM 27/32 GiB'], 'the cards');
  assert.ok(/<div class="ring" role="img" aria-label="MEM 30%">/.test(html), 'MEM is not a ring');
  assert.ok(!/<div class="ring" role="img" aria-label="GPU/.test(html), 'a GPU still drawn as a ring');
  assert.ok(/<div class="rings" style="--rings: 3; --gpus: 2">/.test(html), 'the grid is not told three glyph columns, two of them cards');
  // A stale box: every card dimmed and "not reported", with the age as the reason.
  const stale = api.machineBody({ hardware: { b: { ...h, stale: true, age_s: 42.2,
    rows: h.rows.map((r) => ({ ...r, use_pct: null })) } } }, 'b');
  const titles = [...stale.matchAll(/<b class="gpu-tag none" title="([^"]+)">—<\/b>/g)].map(([, t]) => t);
  assert.deepStrictEqual(titles, ['no sample for 42s', 'no sample for 42s'], 'a stale box\'s cards');
  // A one-card unified box: one card, no VRAM chip, two glyph columns (the default).
  const one = api.machineBody({ hardware: { b: { ...h, rows: [boxRow(null)], gpu_pct: 94 } } }, 'b');
  assert.strictEqual((one.match(/<svg class="gpu-card/g) || []).length, 1, 'a one-GPU box');
  assert.ok(!/gpu-vram/.test(one), 'a unified box with VRAM chips');
  assert.ok(/<div class="rings">/.test(one), 'a one-GPU box sets --rings');
  checks += 8;
}

// 6. VRAM IN USE NOT REPORTED (a card that says its total and no used figure): the VRAM tag is "—" with its reason,
//    no colour — never 0 % (operator 2026-10-30: the VRAM's share is a label beside the fan, inside the card).
{
  const g = glyphOf(api.gpuColumn(card(0, 40, null, 24576), 40, 'why'));
  assert.ok(g.vramLine && g.vramLine.none, 'a VRAM not reported has a coloured tag');
  assert.strictEqual(g.vramLine.text, '—', `a VRAM not reported says ${g.vramLine && g.vramLine.text}`);
  assert.strictEqual(g.vramLine.colour, null, 'a VRAM not reported wears a colour');
  checks += 3;
}

// 7. THE FAN TURNS AT THE UTILISATION (operator 2026-10-30: "make GPU rotate in svg based on usage").
{
  const { still, slow, fast } = api.FAN_SPIN;
  assert.ok(still > 0 && slow > fast && fast > 0, `FAN_SPIN ${JSON.stringify(api.FAN_SPIN)}`);
  // No reading, or (nearly) idle: no spin at all — never a made-up speed.
  for (const p of [null, undefined, NaN, 0, still / 2]) {
    assert.strictEqual(api.fanSpin(p, 1234), null, `a spin at ${p}`);
  }
  // The period falls monotonically with the utilisation: slowest just over idle, fastest at 100 % (and above).
  let last = Infinity;
  for (let p = Math.ceil(still); p <= 100; p += 1) {
    const s = api.fanSpin(p, 0);
    assert.ok(s && s.period > 0, `no spin at ${p} %`);
    assert.ok(s.period < last, `the period at ${p} % (${s.period}s) is not shorter than at ${p - 1} % (${last}s)`);
    last = s.period;
    checks += 1;
  }
  assert.ok(Math.abs(api.fanSpin(100, 0).period - fast) < 0.011, `100 % turns once per ${api.fanSpin(100, 0).period}s`);
  assert.strictEqual(api.fanSpin(150, 0).period, api.fanSpin(100, 0).period, 'over 100 % turns faster than 100 %');
  assert.ok(Math.abs(api.fanSpin(still, 0).period - slow) < 0.011, `${still} % turns once per ${api.fanSpin(still, 0).period}s`);
  // The phase is the PAGE CLOCK's (the injected `now`, ms) — a negative delay of now mod period — so a re-render on
  // the next poll continues the turn the last one drew instead of starting it at 0°.
  const p = api.fanSpin(50, 0).period;
  for (const now of [0, 250, 1000, 123456.7, p * 1000 * 7 + 100]) {
    const s = api.fanSpin(50, now);
    const want = ((now / 1000) % p);
    assert.ok(s.delay <= 0 && Math.abs(-s.delay - want) < 0.002, `at ${now} ms the delay is ${s.delay}, not -${want}`);
    checks += 1;
  }
  checks += 8;
}
// The card carries the spin on the BLADES group only (period and phase as custom properties); the arc, track, hub
// and the card do not turn. Its centre is the fan's, in either layout (a unified row's middle, a discrete card's left).
for (const [r, pct] of [[boxRow(94), 94], [card(0, 31, 7270, 24576), 31], [card(1, 63, 1, 2), 63]]) {
  const html = api.gpuColumn(r, pct, 'why', 2500);
  const want = api.fanSpin(pct, 2500);
  const g = html.match(/<g class="gpu-blades spin" style="([^"]*)">/);
  assert.ok(g, `${r.label} at ${pct}: the blades do not spin: ${html.slice(0, 600)}`);
  const style = Object.fromEntries(g[1].split(';').map((d) => d.split(':').map((x) => x.trim())).filter(([k]) => k));
  assert.strictEqual(style['--spin'], `${want.period}s`, `${r.label}: --spin ${style['--spin']}`);
  assert.strictEqual(style['--spin-delay'], `${want.delay}s`, `${r.label}: --spin-delay ${style['--spin-delay']}`);
  const fan = attrs(html.match(/<circle class="gpu-fan-bg"[^>]*>/)[0]);
  assert.strictEqual(style['transform-origin'], `${fan.cx}px ${fan.cy}px`, `${r.label}: turns about ${style['transform-origin']}`);
  assert.strictEqual((html.match(/--spin/g) || []).length, 2, `${r.label}: --spin outside the blades group`);
  assert.ok(!/<(?:svg|circle|rect)[^>]*style="[^"]*--spin/.test(html), `${r.label}: something but the blades spins`);
  assert.ok(/<g class="gpu-blades spin"[^>]*>(?:<ellipse[^>]*\/>)+<\/g>/.test(html), `${r.label}: the spinning group holds more than blades`);
  checks += 6;
}
// No reading, or idle: the blades are drawn still — no spin class, no custom property.
for (const [r, pct] of [[boxRow(null), null], [card(0, 0, 0, 16384), 0], [card(0, 1, 0, 16384), 1]]) {
  const html = api.gpuColumn(r, pct, 'why', 2500);
  assert.ok(/<g class="gpu-blades">/.test(html) && !/--spin|\bspin\b/.test(html), `${r.label} at ${pct}: spins`);
  checks += 1;
}
// gateway.css: the rotation, run by the custom properties; still under reduced motion and on a stale box.
{
  const css = fs.readFileSync(path.join(__dirname, '..', 'vllm_lb', 'static', 'gateway.css'), 'utf8');
  const kf = css.match(/@keyframes gpu-fan-spin\s*\{([\s\S]*?)\}\s*\}/);
  assert.ok(kf && /rotate\(360deg\)/.test(kf[1]), 'no @keyframes gpu-fan-spin to 360deg');
  const rule = css.match(/\.gpu-card \.gpu-blades\.spin\s*\{([^}]*)\}/);
  assert.ok(rule, 'no .gpu-card .gpu-blades.spin rule');
  assert.ok(/animation:\s*gpu-fan-spin var\(--spin\) linear infinite/.test(rule[1]), `the spin rule: ${rule[1]}`);
  assert.ok(/animation-delay:\s*var\(--spin-delay\)/.test(rule[1]), 'the phase is not the delay');
  assert.ok(/transform-box:\s*view-box/.test(rule[1]), 'transform-origin not in the card\'s units');
  assert.ok(/@media \(prefers-reduced-motion: reduce\)\s*\{[^{}]*\.gpu-blades\.spin[^{}]*\{\s*animation: none;\s*\}\s*\}/.test(css),
    'reduced motion does not stop the fan');
  assert.ok(/\.card\.machine\.stale \.gpu-blades\.spin\s*\{\s*animation: none;\s*\}/.test(css), 'a stale box\'s fan turns');
  checks += 7;
}

console.log(`ok - ${checks} checks: each GPU row is a card — board, bracket, PCIe fingers, a fan that is its utilisation's `
  + 'ring — with VRAM chips lit by used/total only where the row has VRAM; not reported is dimmed, never 0; the fan '
  + 'turns at the utilisation, phased by the page clock, still with no reading');
