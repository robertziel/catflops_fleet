/* A MACHINE card reads ONE value per figure, checked where the card is built: gateway.js `machineBody`,
 * under node.
 *
 * THE BUG THIS PINS (2026-10-01): box-c's MEM ring and the headline over its graph said 20 % in green;
 * the graph under them was red, "max 95%" beside it. It was drawn from `history.mem_pct` — the GPU's
 * memory (hw.py `_memory`: a card's own VRAM, else the RAM a GB10 shares) — while the ring and the
 * headline read `ram_percent`, the box's RAM. On a GB10 those are one number, which is why the GB10 boxes
 * never showed it; box-c's iGPU sat at 485/512 MB. The colour rule was applied correctly, to another
 * figure.
 *
 * So, for every figure on every card: the graph's NEWEST POINT is at the ring's value, its colour there
 * IS the ring's colour, and the headline wears that colour too (to the rounding of the figure it prints).
 *
 * And the cores are drawn inside a CPU (§6): a package, pins on its four sides, one cell per core in `coreColour`.
 * The CPU is drawn ONCE (§7, operator 2026-10-12: "unify this into one, we don't need circle"): no CPU ring; under
 * the chip, side by side, a "CPU" label and the box's CPU load as a tag in the ring's colour rule (`loadColour`).
 *
 * No DOM: gateway.js is loaded with a stub `document` and a `fetch` that never answers.
 *
 *   node tests/machine_card.node.js      (tests/test_machine_card.py runs this when node exists)
 */
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');

// palette.js first: the colours this script computes are read from the style's tokens (tests/palette_env.js).
const src = require('./palette_env').withPalette(fs.readFileSync(path.join(__dirname, '..', 'vllm_lb', 'static', 'gateway.js'), 'utf8'));
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
  'getComputedStyle', 'ResizeObserver', 'localStorage', 'navigator', 'performance',
  `${src}\n;return { machineBody, scaleColor, coreColour, loadColour, SPARKLINE_H };`)(
  document, window, () => new Promise(() => {}), { search: '', href: '' }, () => 0, () => 0,
  () => ({}), undefined, window.localStorage, { userAgent: 'node' },
  // A FIXED page clock: the GPU fan's phase is the clock's (gateway.js `fanSpin`), so the pinned card (§3) is
  // byte-for-byte the same on every run.
  { now: () => 4321 });

// ── /lb/status `hardware` as the live LB served it on 2026-10-01 (rings of 60 cut to 6) ─────────────
// Each box carries `rows`: its GPU rows as its MEMORY MODEL laid them out (hw.py, vllm_lb.hardware) — ONE for a
// unified box, one per card for a discrete one. The page draws the rows it is given.
const ring = (v) => new Array(6).fill(v);
const boxRow = (pct, history) => ({ label: 'GPU', card: null, box_gpu: true, use_pct: pct, history,
  vram_label: null, vram_used_mb: null, vram_total_mb: null });
const STATUS = { hardware: {
  // The Radeon 8060S: unified ("Dynamic + carveout"). Its `mem_pct` past is the 485/512 MB carve-out of the
  // 2026-10-01 bug — kept in the fixture so the MEM graph is proven to ignore it.
  'box-c': { host: 'box-c', age_s: 0.7, stale: false, mem_unified: true, memory: 'unified', cpu_percent: 17.8,
    ram_percent: 20.3, ram_used_mb: 26009, ram_total_mb: 127940, swap_used_mb: 0, gpu_pct: 0, gpu_count: 1,
    mem_used_mb: 26009, mem_total_mb: 127940, rows: [boxRow(0, ring(0))],
    history: { gpu_pct: ring(0), cpu_pct: [13.8, 9.3, 9.3, 13.8, 17.8, 17.8], mem_pct: ring(94.7),
               ram_pct: ring(20.3) } },
  // A GB10: unified, so its two memory rings are one number.
  'box-a': JSON.parse(fs.readFileSync(path.join(__dirname, 'fixtures', 'machine-card', 'unified-status.json'),
    'utf8')).hardware['box-a'],
  // Two discrete cards in one box (site-b): card 0 a 3090 at 31 %, card 1 the 5090 at 5 % — each its own VRAM.
  'site-b': { host: 'site-b-host', age_s: 1.2, stale: false, mem_unified: false, memory: 'discrete', cpu_percent: 4.0,
    ram_percent: 30.0, ram_used_mb: 28672, ram_total_mb: 96256, swap_used_mb: 7168, gpu_pct: 31, gpu_count: 2,
    mem_used_mb: 7270 + 28058, mem_total_mb: 24576 + 32607,
    rows: [{ label: 'GPU 0', card: 0, box_gpu: false, use_pct: 31, history: [20, 25, 28, 30, 31, 31],
             vram_label: 'VRAM 0', vram_used_mb: 7270, vram_total_mb: 24576 },
           { label: 'GPU 1', card: 1, box_gpu: false, use_pct: 5, history: [60, 40, 20, 9, 5, 5],
             vram_label: 'VRAM 1 · NVIDIA GeForce RTX 5090', vram_used_mb: 28058, vram_total_mb: 32607 }],
    history: { gpu_pct: [60, 40, 28, 30, 31, 31], cpu_pct: ring(4.0), mem_pct: ring(61.8), ram_pct: ring(30.0),
               gpus: { 0: [20, 25, 28, 30, 31, 31], 1: [60, 40, 20, 9, 5, 5] } } },
} };

// ── reading the card the way it is drawn ──────────────────────────────────────────────────────────
/** Every machine figure drawn as a gauge: each GPU row's card (gateway.js `gpuColumn`, 2026-10-30 — its fan is the ring,
 *  its number the tag under it), then the MEM ring; the value, the arc's colour and the number's colour of each. */
const ringsOf = (html) => Object.fromEntries([...[...html.matchAll(
  /<div class="gpu"><svg class="gpu-card"[^>]*>([\s\S]*?)<\/svg>(?:<div class="gpu-lines">)?<div class="gpu-line"[^>]*><span class="gpu-label">([^<]+)<\/span><b class="gpu-tag" style="--tag:([^"]+)">([\d.]+)%<\/b>/g)]
  .map(([, inner, label, colour, value]) => [label, { value: Number(value),
    arc: (inner.match(/class="gpu-fan-fg"[^>]*\bstroke="([^"]+)"/) || [])[1] || null, number: colour }]),
  ...[...html.matchAll(
  /<div class="ring" role="img" aria-label="([\w ]+?) ([\d.]+)%">([\s\S]*?)<span class="ring-label">/g)]
  .map(([, label, value, inner]) => [label, { value: Number(value),
    arc: (inner.match(/class="ring-fg"[^>]*\bstroke="([^"]+)"/) || [])[1] || null,
    number: (inner.match(/<b style="color:([^"]+)">/) || [])[1] || null }])]);
/** The CPU column (§7): the chip's name, the label and the tag on the one line under it. */
const cpuOf = (html) => {
  const col = (html.match(/<div class="cpu">([\s\S]*?<div class="cpu-line"[^>]*>[\s\S]*?<\/div>)<\/div>/) || [])[1];
  if (!col) return null;
  const line = col.match(/<div class="cpu-line"([^>]*)>([\s\S]*?)<\/div>$/);
  const tag = line[2].match(/<b class="cpu-tag( none)?"(?: style="--tag:([^"]+)")?(?: title="([^"]+)")?>([^<]*)<\/b>/);
  return { chip: (col.match(/^<svg class="cores"[^>]*aria-label="([^"]+)"/) || [])[1] || null,
    lineAttrs: line[1], label: (line[2].match(/<span class="cpu-label">([^<]*)<\/span>/) || [])[1] || null,
    tagFirst: line[2].indexOf('cpu-label') < line[2].indexOf('cpu-tag'),
    none: !!(tag && tag[1]), colour: tag ? tag[2] || null : null, title: tag ? tag[3] || null : null,
    text: tag ? tag[4] : null, value: tag && /^\d+%$/.test(tag[4]) ? Number(tag[4].slice(0, -1)) : null };
};
const tilesOf = (html) => Object.fromEntries(html.split(/<div class="tile(?=[" ])/).slice(1).map((t) => {
  const stops = [...t.matchAll(/<stop offset="([\d.]+)" stop-color="([^"]+)"\/>/g)]
    .map(([, offset, colour]) => ({ offset: Number(offset), colour }));
  const line = (t.match(/<path d="([^"]+)" fill="none" stroke="url\(#[^)]+\)" stroke-width="1\.5"/) || [])[1] || '';
  const pts = [...line.matchAll(/([\d.]+) ([\d.]+)/g)].map(([, x, y]) => [Number(x), Number(y)]);
  return [(t.match(/<span class="t-label">([^<]+)<\/span>/) || [])[1], {
    headline: (t.match(/<b class="t-val"[^>]*style="color:([^";]+)/) || [])[1] || null,
    stops, newest: pts.length ? pts[pts.length - 1] : null,
    mark: (t.match(/<span class="t-mark"[^>]*>([^<]*)<\/span>/) || [])[1] || null }];
}));
const hue = (c) => Number((/hsla?\(([\d.]+)/.exec(c || '') || [])[1]);
/** The value a point is drawn at: gateway.js `sparkline`'s y, inverted, on a percent's 0-100 axis. */
const drawnAt = (y) => 100 * (1 - (y - 1) / (api.SPARKLINE_H - 2));

let checks = 0;

// 1. Every figure on every card: the graph ends at the ring's value — the CPU's at its tag's (§7) — in that one
//    colour, under a headline in that colour.
for (const box of Object.keys(STATUS.hardware)) {
  const html = api.machineBody(STATUS, box);
  const rings = ringsOf(html), tiles = tilesOf(html), cpu = cpuOf(html);
  const gauges = [...STATUS.hardware[box].rows.map((r) => r.label), 'MEM'];
  assert.deepStrictEqual(Object.keys(rings), gauges, `${box}: the rings`);
  assert.deepStrictEqual(Object.keys(tiles), ['CPU', ...gauges], `${box}: the graphs`);
  assert.ok(cpu && cpu.colour, `${box}: no CPU tag in a colour`);
  rings.CPU = { value: cpu.value, arc: null, number: cpu.colour };
  for (const fig of ['CPU', ...gauges]) {
    const r = rings[fig], t = tiles[fig], what = `${box} ${fig}`;
    if (r.arc) assert.strictEqual(r.arc, r.number, `${what}: the ring's arc is ${r.arc}, its number ${r.number}`);
    assert.ok(t.newest, `${what}: the graph draws no line`);
    const at = drawnAt(t.newest[1]);
    assert.ok(Math.abs(at - r.value) <= 0.55,
      `${what}: the ring says ${r.value}% but the graph's newest point is at ${at.toFixed(1)}%`);
    const end = t.stops[t.stops.length - 1];
    assert.strictEqual(end.offset, 1, `${what}: the gradient has no stop at the newest slot`);
    assert.strictEqual(end.colour, r.number, `${what}: the graph ends ${end.colour}, the ring is ${r.number}`);
    assert.ok(Math.abs(hue(t.headline) - hue(r.number)) <= 1,
      `${what}: the headline is ${t.headline}, the ring ${r.number}`);
    checks += 5;
  }
}

// 2. The complaint itself: box-c's MEM graph is its RAM's — green, labelled by the RAM's own peak —
//    and no slot of it wears the iGPU's 95 %.
{
  const mem = tilesOf(api.machineBody(STATUS, 'box-c')).MEM;
  assert.strictEqual(mem.mark, 'max 20%', `box-c MEM's graph is labelled "${mem.mark}"`);
  const red = api.scaleColor(0.947);
  assert.ok(mem.stops.every((s) => s.colour !== red), "box-c MEM's graph still wears the iGPU memory's 95 %");
  assert.ok(mem.stops.every((s) => s.colour === api.scaleColor(0.203)), 'a flat 20.3 % is one colour end to end');
  checks += 3;
}

// 3. A UNIFIED box keeps EXACTLY today's display (operator, 2026-10-11): the GB10's card, drawn from the same
//    status, is byte for byte what the page drew before per-card rows existed. A deliberate change to it
//    regenerates the pin: render tests/fixtures/machine-card/unified-status.json with the new machineBody.
//    Regenerated 2026-10-30: the GPU row is a graphics card (`gpuColumn`, operator: "make it GPU shape, fan is the
//    circle") — with no VRAM chips, as the GB10's row carries no VRAM of its own; nothing else on the card moved.
//    Regenerated 2026-10-30: the fan's blades turn at the utilisation (`fanSpin`: `spin` class, `--spin`, its phase
//    `--spin-delay` from the fixed clock above, the fan's centre as transform-origin); nothing else moved.
//    Regenerated 2026-10-30: each graph's head carries its label as a `title` (its label is cut to one line with an
//    ellipsis: every graph cell one fixed height, tests/cell_rows.node.js); nothing else moved.
//    Regenerated 2026-10-30: the load scale's colours come from the style's tokens (palette.js), printed with one
//    rule — "hsl(19.8,89.0%,54.6%)" where gateway.js wrote "89%" — and the gradient ids hash those strings; the same
//    colours, nothing else moved.
{
  const today = fs.readFileSync(path.join(__dirname, 'fixtures', 'machine-card', 'unified-today.html'), 'utf8');
  assert.strictEqual(api.machineBody(STATUS, 'box-a'), today, "the GB10's machine card is not today's");
  checks += 1;
}

const figsOf = (html) => [...html.matchAll(/<div class="fig"><span>([^<]*)<\/span><b>([^<]*)<\/b><\/div>/g)]
  .map(([, k, v]) => [k, v]);

// 4. A DISCRETE box: one VRAM line per card, each its own, labelled by index (and the name the sample carried) —
//    never the cards' VRAM summed, which is no card's; and one GPU ring per card at THAT card's own utilisation,
//    never the busiest card's.
{
  const html = api.machineBody(STATUS, 'site-b');
  assert.deepStrictEqual(figsOf(html), [['memory', '28/94 GiB'], ['swap', '7.0 GiB'], ['VRAM 0', '7.1/24 GiB'],
    ['VRAM 1 · NVIDIA GeForce RTX 5090', '27/32 GiB']], 'site-b: its figures');
  const rings = ringsOf(html);
  assert.strictEqual(rings['GPU 0'].value, 31, 'card 0 at its own 31 %');
  assert.strictEqual(rings['GPU 1'].value, 5, 'card 1 at its own 5 %, not the busiest card\'s 31 %');
  assert.ok(!('GPU' in rings), 'no ring for "the box" on a discrete box');
  checks += 4;
}

// 5. A unified box draws no VRAM line at all: its GPU's memory IS the MEM ring.
for (const box of ['box-a', 'box-c']) {
  const keys = figsOf(api.machineBody(STATUS, box)).map(([k]) => k);
  assert.ok(!keys.some((k) => k.startsWith('VRAM')), `${box}: a VRAM line on a unified box (${keys})`);
  checks += 1;
}

// 6. The cores are drawn INSIDE A CPU (operator, 2026-10-12: "Can we make inside nice CPU svg shape?"): a package
//    outline with pins on all four sides, and inside it exactly one cell per logical core — core 0 first, left to
//    right, top to bottom — each in its load's colour by the one rule (`coreColour`), for ANY core count. The
//    frame keeps the tile's 5:4 (gateway.css `.cores`), so nothing beside it moves.
{
  const attrs = (tag) => Object.fromEntries([...tag.matchAll(/([\w-]+)="([^"]*)"/g)].map(([, k, v]) => [k, v]));
  const rectOf = (a) => ({ x: Number(a.x), y: Number(a.y), w: Number(a.width), h: Number(a.height), fill: a.fill });
  const rectsIn = (svg, cls) => {
    const g = (svg.match(new RegExp(`<g class="${cls}"[^>]*>([\\s\\S]*?)</g>`)) || [])[1];
    return g === undefined ? null : [...g.matchAll(/<rect\b[^>]*>/g)].map(([t]) => rectOf(attrs(t)));
  };
  const LOADS = [0, 3.5, 9.9, 10, 42, 77.5, 100, 61];
  for (const n of [1, 16, 20, 32, 64]) {
    const load = Array.from({ length: n }, (_, i) => LOADS[(i * 5) % LOADS.length]);
    const status = { hardware: { box: { ...STATUS.hardware['site-b'], cpu_per_core: load } } };
    const svg = (api.machineBody(status, 'box').match(/<svg class="cores"[\s\S]*?<\/svg>/) || [])[0];
    const what = `${n} core${n === 1 ? '' : 's'}`;
    assert.ok(svg, `${what}: no core chip drawn`);
    const head = attrs(svg.match(/^<svg[^>]*>/)[0]);
    assert.strictEqual(head.role, 'img', `${what}: the chip is not an image to a screen reader`);
    assert.strictEqual(head['aria-label'], `CPU · 4 % · ${n} core${n === 1 ? '' : 's'} · busiest ${Math.round(Math.max(...load))} %`,
      `${what}: its name`);
    const [, , W, H] = head.viewBox.split(' ').map(Number);
    assert.strictEqual(W / H, 5 / 4, `${what}: the frame is ${W}×${H}, not the tile's 5:4`);

    const pkgTag = svg.match(/<rect class="chip-pkg"[^>]*>/);
    assert.ok(pkgTag, `${what}: no package outline`);
    const pkg = rectOf(attrs(pkgTag[0]));
    const inside = (r, o, pad = 0) => r.x >= o.x + pad && r.y >= o.y + pad
      && r.x + r.w <= o.x + o.w - pad && r.y + r.h <= o.y + o.h - pad;
    assert.ok(inside(pkg, { x: 0, y: 0, w: W, h: H }, 0.01), `${what}: the package leaves the frame`);

    const pins = rectsIn(svg, 'chip-pins');
    assert.ok(pins && pins.length, `${what}: no pins`);
    const side = (p) => {
      const cx = p.x + p.w / 2, cy = p.y + p.h / 2;
      const sides = [cy < pkg.y && 'top', cy > pkg.y + pkg.h && 'bottom', cx < pkg.x && 'left', cx > pkg.x + pkg.w && 'right']
        .filter(Boolean);
      return sides.length === 1 ? sides[0] : null;
    };
    const bySide = { top: 0, bottom: 0, left: 0, right: 0 };
    for (const p of pins) {
      const s = side(p);
      assert.ok(s, `${what}: a pin at ${p.x},${p.y} is on no one side of the package`);
      assert.ok(inside(p, { x: 0, y: 0, w: W, h: H }), `${what}: a pin leaves the frame`);
      bySide[s] += 1;
    }
    for (const s of Object.keys(bySide)) assert.ok(bySide[s] >= 2, `${what}: ${bySide[s]} pins on the ${s}`);
    assert.strictEqual(bySide.top, bySide.bottom, `${what}: top and bottom pins differ`);
    assert.strictEqual(bySide.left, bySide.right, `${what}: left and right pins differ`);

    const cells = rectsIn(svg, 'chip-cores');
    assert.ok(cells, `${what}: no core cells inside the chip`);
    assert.strictEqual(cells.length, n, `${what}: ${cells.length} cells`);
    cells.forEach((c, i) => {
      assert.ok(inside(c, pkg, 0.5), `${what}: core ${i} is not inside the package`);
      assert.ok(c.w === c.h && c.w === cells[0].w && c.w > 0, `${what}: core ${i} is ${c.w}×${c.h}`);
      assert.strictEqual(c.fill, api.coreColour(load[i]), `${what}: core ${i} at ${load[i]} %`);
      // The rule itself, so a change to `coreColour` is seen here too: grey under 10 %, the load scale from it.
      if (load[i] < 10) assert.ok(c.fill.startsWith('rgba(139,148,158,'), `${what}: idle core ${i} is ${c.fill}`);
      else assert.strictEqual(c.fill, api.scaleColor(load[i] / 100), `${what}: core ${i} off the load scale`);
      if (i) {
        const p = cells[i - 1];
        assert.ok((c.y === p.y && c.x > p.x) || (c.y > p.y && c.x === cells[0].x),
          `${what}: core ${i} is not after core ${i - 1} in reading order`);
      }
    });
    checks += 10 + 3 * n;
  }
}

// 7. The CPU is drawn ONCE (operator, 2026-10-12, over site-b's card: "unify this into one, we don't need circle, just
//    add CPU label and percentage tag with color maybe below CPU svg next to each other? make it global on any fleet").
//    Every box — unified, discrete, one with no per-core reading, one not broadcasting a CPU at all — gets the one
//    column: the chip (where there are cores), then ONE line under it, "CPU" and then the box's CPU load as a tag,
//    coloured by the rule every ring wears (`loadColour`, one definition: the load scale at value / 100). No CPU ring.
//    A load not known is "—" with its reason in the title — never 0, never a colour. The chip's name says the tag's value.
{
  const known = { ...STATUS.hardware['site-b'], cpu_per_core: [3, 61, 12, 100] };
  const BOXES = {
    unified: STATUS.hardware['box-a'], igpu: STATUS.hardware['box-c'], discrete: known,
    'no cores': { ...STATUS.hardware['site-b'], cpu_per_core: null, cpu_percent: 66.6 },
    'no cpu': { ...known, cpu_percent: null },
    stale: { ...STATUS.hardware['site-b'], stale: true, age_s: 42349.9, cpu_percent: null, cpu_per_core: null,
      ram_percent: null, rows: [] },
  };
  for (const [kind, h] of Object.entries(BOXES)) {
    const html = api.machineBody({ hardware: { any: h } }, 'any'), what = `${kind} box`;
    assert.ok(!/aria-label="CPU [\d.]+%"|aria-label="CPU not reported"|<span class="ring-label">CPU</.test(html),
      `${what}: a CPU ring is still drawn`);
    const cpu = cpuOf(html);
    assert.ok(cpu, `${what}: no CPU column with a label line`);
    assert.strictEqual(cpu.label, 'CPU', `${what}: the line's label`);
    assert.ok(cpu.tagFirst, `${what}: the tag is not after the label`);
    const cores = (h.cpu_per_core || []).length;
    assert.strictEqual(!!cpu.chip, cores > 0, `${what}: chip drawn ${!!cpu.chip} for ${cores} cores`);
    if (cpu.chip) assert.ok(/aria-hidden="true"/.test(cpu.lineAttrs), `${what}: the line is read twice (the chip says it)`);
    const v = h.cpu_percent;
    if (typeof v === 'number') {
      assert.strictEqual(cpu.text, `${Math.round(v)}%`, `${what}: the tag says ${cpu.text}`);
      assert.strictEqual(cpu.colour, api.loadColour(v), `${what}: the tag is ${cpu.colour}`);
      assert.strictEqual(api.loadColour(v), api.scaleColor(v / 100), 'loadColour is the load scale');
      assert.ok(!cpu.none && !cpu.title, `${what}: a known load marked unknown`);
      if (cpu.chip) assert.ok(cpu.chip.startsWith(`CPU · ${Math.round(v)} % · ${cores} core`), `${what}: chip "${cpu.chip}"`);
    } else {
      assert.strictEqual(cpu.text, '—', `${what}: an unknown load reads ${cpu.text}`);
      assert.ok(cpu.none && cpu.colour === null, `${what}: an unknown load wears a colour`);
      assert.ok(cpu.title && cpu.title.length > 8, `${what}: no reason in the title`);
      if (h.stale) assert.ok(cpu.title.includes(`${Math.round(h.age_s)}s`), `${what}: "${cpu.title}"`);
      if (cpu.chip) assert.ok(cpu.chip.startsWith(`CPU · not reported · ${cores} core`), `${what}: chip "${cpu.chip}"`);
    }
    checks += 8;
  }
  // The ring a GPU or MEM still draws wears the same rule: one definition.
  assert.strictEqual(ringsOf(api.machineBody(STATUS, 'site-b'))['GPU 0'].number, api.loadColour(31), 'GPU 0 ring off loadColour');
  checks += 1;
}

console.log(`ok - ${checks} checks: every machine figure's ring (the CPU's tag), headline and graph read one value in one colour; `
  + 'the cores sit in a CPU, one cell per core in its load\'s colour; the CPU is drawn once, label + tag');
