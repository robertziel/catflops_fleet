/* /benchmarks — the cluster's PERFORMANCE: first under real traffic, as the gateway records it one
 * busy minute at a time (/lb/perf, perflog.py) — pooled per configuration, compared against
 * max_num_seqs, and drawn over time — and then, only while any exist, the saved benchmark runs of the
 * controlled experiments that came before (/lb/bench, bench.py). Served BY the LB, so every route is
 * this process's own, read-only and same-origin; the authenticated nginx vhost on the edge is what stands
 * between this page and the internet, exactly as for the gateway page.
 *
 * REAL TRAFFIC IS NOT A CONTROLLED EXPERIMENT, and the page says so where the comparison starts: each
 * configuration served whatever arrived while it ran. So every figure carries its sample count, a
 * configuration with fewer than MIN_WINDOWS minutes is shown but never compared, and the traffic each
 * one was given (prompt and output tokens per request) is printed beside what it achieved.
 *
 * CHART RULES this file keeps, and why (they are not decoration):
 *   * ONE measure per chart, one y-axis. Aggregate tok/s and wall time are two charts, not two
 *     scales on one — with two scales any pair of lines can be made to cross wherever the drawer
 *     likes. `running`, `waiting` and KV are small multiples for the same reason.
 *   * max_num_seqs is ORDERED, so its colour is a one-hue ordinal ramp (--seq-1..5, dim cap to
 *     bright), never a set of unrelated hues. A corpus run is marked by a DASHED stroke, so
 *     workload is never carried by colour alone.
 *   * --ok/--warn/--bad stay STATUS. No run is ever drawn in them.
 *   * A NUMBER in the run table is washed on a one-hue violet ramp by how good it is INSIDE ITS
 *     OWN comparable group, never against the whole table: a corpus run's 6 tok/s and a synthetic
 *     run's 305 are both right and measure different work. Direction is DECLARED per measure
 *     (`MEASURES`), never inferred, because half of them are better when smaller.
 *   * A sample nobody measured is a GAP, never a zero — the same rule as the gateway's dot grid
 *     (gateway.js `dots`). Drawing it as zero would claim we measured idleness.
 *   * The spread is drawn, not just the mean: the operator's question is that throughput "changes
 *     up and down", and an average is exactly the statistic that hides it.
 *   * TWO WORKLOADS ARE NEVER ON ONE CHART. A workload is what the result's own config stamp says it
 *     ran (worklog/bench/annotate.py): decode-bound real prompts, a cold and a warm pass of 10-20k
 *     token prompts, a pure prefill probe, the older synthetic series. They measure different work
 *     — the cold and warm passes of the SAME prompts differ up to 2x — so the batch-size comparison
 *     is one panel per workload, and a run with no stamp is in no panel at all (it is named, not
 *     guessed into one from its label).
 *   * In the ARCHIVE's batch-size comparison max_num_seqs is the X AXIS, so colour is free to carry the
 *     other two settings: HUE = speculative tokens, SHAPE = max_num_batched_tokens. Hue is a validated
 *     categorical pair (--cmp-1/2, benchmarks.css) that borrows neither the cap ramp's blue nor the
 *     value wash's violet nor a status colour. Every repetition is its own mark over the mean's bar.
 *   * REAL TRAFFIC has ONE encoding for a configuration, the same in every chart, legend, square and
 *     card (cfgStyle): its cap's ramp step, and — for a second configuration at the same cap — a hatched
 *     fill (a third 135°, then a dotted outline), never a second hue. The charts against the parallel
 *     level are GROUPED BARS (one band per level, one bar per configuration, yesterday's bar-per-cap with
 *     today's second dimension); a bar too thin to compare is a ghost of itself; the y scale is the
 *     bars', never a spread's. THE SPREAD IS BACKGROUND: p10–p90 is a pale wash behind the bar with the
 *     median as a tick across it, brought forward on hover, where the figures are — not a whisker in
 *     front of the data with its p90 printed out of the top of the plot, which is what the outliers of a
 *     145 s tail did to the chart they were meant to annotate. A band past the top is cut off there and
 *     the hover says how far it went. WHAT A BAND IS A BAND OF is per measure and is carried in WORDS,
 *     never by the drawing: across the REQUESTS where the engine keeps a histogram (time to first token
 *     alone), across the RECORDED MINUTES everywhere else — one observation per minute, which is a real
 *     sample and a different claim. Neither is ever invented for a measure that has no samples.
 *   * Every chart is backed by the run table, which is the table view and holds the same numbers.
 *
 * SVGs are built at the container's real pixel width (and rebuilt on resize) rather than scaled
 * from a fixed viewBox: a scaled viewBox shrinks the tick labels with the card, and a 5 px axis
 * label on a phone is a smudge.
 */
'use strict';

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s == null ? '' : s).replace(/[&<>"']/g, (c) =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const NS = 'http://www.w3.org/2000/svg';

/** The ordinal ramp for the concurrency cap, dim → bright. Fixed order, assigned by the cap's rank
 *  among the caps present, so a filter that drops a cap never repaints the survivors' neighbours
 *  arbitrarily — the ramp is recomputed from the caps the INDEX holds, not from the filtered set. */
const RAMP = ['var(--seq-1)', 'var(--seq-2)', 'var(--seq-3)', 'var(--seq-4)', 'var(--seq-5)'];

const num = (v) => (typeof v === 'number' && Number.isFinite(v) ? v : null);

function fmt(v, digits) {
  const n = num(v);
  if (n == null) return '—';
  if (digits != null) return n.toFixed(digits);
  if (Math.abs(n) >= 1000) return Math.round(n).toLocaleString('en-US');
  if (Math.abs(n) >= 100) return n.toFixed(0);
  if (Math.abs(n) >= 10) return n.toFixed(1);
  return n.toFixed(2).replace(/0$/, '');
}
const fmtInt = (v) => (num(v) == null ? '—' : Math.round(v).toLocaleString('en-US'));
function fmtAgo(ts) {
  const n = num(ts);
  if (n == null) return '—';
  const s = Math.max(0, Date.now() / 1000 - n);
  if (s < 90) return `${Math.round(s)} s ago`;
  if (s < 5400) return `${Math.round(s / 60)} min ago`;
  if (s < 172800) return `${Math.round(s / 3600)} h ago`;
  return new Date(n * 1000).toISOString().slice(0, 16).replace('T', ' ');
}

// ── the gateway's two value scales, for a FIGURE printed as text ────────────────────────────────────
// The context table wears the fleet app's colours (FleetStats `scaleColor` / `rateColor`, and the
// gateway page's own copies in gateway.js): a LOAD — a share of a ceiling, here KV occupancy — reads
// blue near zero, then green, yellow, red; a RATE — tok/s — stays in the blue family, steel at rest and
// a bright cyan at the best of the rows it is read against, so a throughput is never read as "hot".
// PRESENTATION ONLY: the colour is a function of the number already printed in the cell, and nothing is
// computed, pooled or compared with it. The same HSB→HSL arithmetic as gateway.js, returned as channels
// because a figure in a table is SMALL TEXT, which the gateway's graph lines and 19 px headlines are
// not: `figureInk` lifts the lightness until the figure holds 4.6:1 on the lighter of this page's two
// card surfaces (--panel-2 #1b2129) — the floor gateway.css sets for its faintest text. The hue, which
// is what the scale says, is never touched.
function loadHSL(t) {
  const x = Math.min(Math.max(Number(t) || 0, 0), 1);
  if (x < 0.12) {
    const u = x / 0.12, sat = 0.40 + 0.45 * u, v = 1.0 - 0.05 * u, l = v * (1 - sat / 2);
    return { h: 360 * (0.52 - 0.16 * u), s: l === 0 || l === 1 ? 0 : (v - l) / Math.min(l, 1 - l), l };
  }
  const hue = x < 0.60 ? 0.36 - 0.21 * ((x - 0.12) / 0.48) : 0.15 - 0.15 * ((x - 0.60) / 0.40);
  return { h: hue * 360, s: 0.89, l: 0.546 };
}
function rateHSL(t) {
  const x = Math.min(Math.max(Number(t) || 0, 0), 1);
  let h, sat, v;
  if (x < 0.4) { const u = x / 0.4; h = 0.60; sat = 0.05 + 0.25 * u; v = 0.50 + 0.12 * u; }
  else if (x < 0.8) { const u = (x - 0.4) / 0.4; h = 0.60 - 0.02 * u; sat = 0.30 + 0.45 * u; v = 0.62 + 0.23 * u; }
  else { const u = (x - 0.8) / 0.2; h = 0.58 - 0.04 * u; sat = 0.75 + 0.20 * u; v = 0.85 + 0.15 * u; }
  const l = v * (1 - sat / 2);
  return { h: 360 * h, s: l === 0 || l === 1 ? 0 : (v - l) / Math.min(l, 1 - l), l };
}
/** WCAG relative luminance of an HSL colour. */
function hslLuminance({ h, s, l }) {
  const a = s * Math.min(l, 1 - l);
  const ch = (n) => { const k = (n + h / 30) % 12; const c = l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1));
    return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4); };
  return 0.2126 * ch(0) + 0.7152 * ch(8) + 0.0722 * ch(4);
}
const FIGURE_SURFACE_LUM = 0.0148;               // --panel-2 #1b2129
const FIGURE_MIN_CONTRAST = 4.6;
function figureInk(hsl) {
  const c = { ...hsl };
  while (c.l < 0.95 && (hslLuminance(c) + 0.05) / (FIGURE_SURFACE_LUM + 0.05) < FIGURE_MIN_CONTRAST) c.l += 0.01;
  return `hsl(${c.h.toFixed(1)},${(100 * c.s).toFixed(1)}%,${(100 * c.l).toFixed(1)}%)`;
}

// ── state ───────────────────────────────────────────────────────────────────

const state = {
  index: null,          // the whole /lb/bench payload
  caps: [],             // every max_num_seqs in the index, ascending — the ramp's domain
  specs: [],            // every num_speculative_tokens in the index, ascending — the series hues
  batched: [],          // every max_num_batched_tokens in the index, ascending — the series shapes
  filters: { model: '', workload: '', seqs: new Set(), conc: '', prompt: '', out: '' },
  selected: null,       // run id
  detail: null,         // the full run for `selected`
  heat: null,           // workload -> per-measure lo/hi, the value wash's scales
  benchError: '',       // why /lb/bench could not be read, when it could not
};

/** The ramp step for a cap: its rank among ALL caps in the index. A cap the page could not
 *  determine gets the neutral ink, never a step it did not earn. */
function capColor(seqs) {
  const i = state.caps.indexOf(num(seqs));
  if (i < 0) return 'var(--ink-faint)';
  return RAMP[state.caps.length <= 1 ? 2 : Math.round(i * (RAMP.length - 1) / (state.caps.length - 1))];
}

const filtered = () => (state.index ? state.index.runs.filter(passes) : []);

function passes(r) {
  const f = state.filters;
  if (f.model && r.model !== f.model) return false;
  if (f.workload && r.workload !== f.workload) return false;
  if (f.seqs.size && !f.seqs.has(r.max_num_seqs)) return false;
  if (f.conc !== '' && r.concurrency !== Number(f.conc)) return false;
  if (f.prompt !== '' && r.prompt_tokens !== Number(f.prompt)) return false;
  if (f.out !== '' && r.max_tokens !== Number(f.out)) return false;
  return true;
}

// ── the chart kit ───────────────────────────────────────────────────────────

function svgEl(name, attrs) {
  const n = document.createElementNS(NS, name);
  for (const k in attrs || {}) if (attrs[k] != null) n.setAttribute(k, attrs[k]);
  return n;
}
function txt(s, attrs) {
  const n = svgEl('text', attrs);
  n.textContent = s;
  return n;
}

/** An axis tick prints as the round number it is: `0`, not `0.0`. */
const tickText = (v) => (Number.isInteger(v) ? String(v) : fmt(v));
/** ...and an axis that reaches ten thousand counts in thousands (`5k`, `12.5k`), every tick alike, as the
 *  fleet app's axes do: a tick is a place on a scale, not a figure, and `30000` cost a narrow chart a fifth
 *  of its width in gutter. Every figure the page PRINTS (labels, tooltips, tables) keeps its digits. */
const kTickText = (v) => (v === 0 ? '0' : `${Math.round(v / 100) / 10}k`);

/** Axis ticks on round numbers, at most `want` of them. A tick on 0.37 is noise. */
function ticks(lo, hi, want) {
  if (!(hi > lo)) return [lo];
  const raw = (hi - lo) / Math.max(1, want);
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) || 10 * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(Math.round(v * 1e6) / 1e6);
  return out;
}

/** A chart frame: the plot box, the scales, the recessive grid and the axis titles.
 *  `xCat` gives a band scale (one slot per category); otherwise x is linear over [x0, x1]. */
/** The width a chart's SVG is drawn at: its host's CONTENT box. `clientWidth` includes the padding, and
 *  an SVG built that wide is then scaled down into the content box — its text shrinks with it and a
 *  pointer's pixel no longer lands on the same x the chart computed (the crosshair picked a sample ~10%
 *  to the left of the pointer on a 0.9rem-padded card). */
function contentWidth(host) {
  const cs = getComputedStyle(host);
  return (host.clientWidth || 320) - (parseFloat(cs.paddingLeft) || 0) - (parseFloat(cs.paddingRight) || 0);
}

function frame(host, opts) {
  const o = opts || {};
  const width = Math.max(260, Math.floor(contentWidth(host)));
  const height = o.height || Math.max(180, Math.min(300, Math.round(width * (o.aspect || 0.55))));
  // THE FLEET APP'S CHART FRAME: a few light grid lines, the tick labels in a narrow gutter, and the unit
  // the axis counts in set ONCE, upright, over the top of that gutter — where the app's charts carry it in
  // their title. It used to be a sentence turned on its side beside the plot, which cost every chart
  // 46 px of gutter on a 350 px phone card for a word the eye had to tilt to read.
  const pad = { t: 18, r: o.padRight != null ? o.padRight : 6,
    b: o.padBottom != null ? o.padBottom : (o.xLabels === false ? 20 : 34), l: 34 };
  const svg = svgEl('svg', { width, height, viewBox: `0 0 ${width} ${height}`, role: 'img' });
  if (o.title) svg.append(txt(o.title, { class: 'g-atitle', x: 0, y: 0, opacity: 0 }));  // a11y name
  const W = width - pad.l - pad.r, H = height - pad.t - pad.b;
  const yLo = o.yLo != null ? o.yLo : 0;
  const yHi = o.yHi > yLo ? o.yHi : yLo + 1;
  const y = (v) => pad.t + H - ((v - yLo) / (yHi - yLo)) * H;
  const cats = o.xCat || null;
  const band = cats ? W / Math.max(1, cats.length) : 0;
  const x = cats
    ? (i) => pad.l + band * (i + 0.5)
    : (v) => pad.l + ((v - (o.x0 || 0)) / Math.max(1e-9, (o.x1 || 1) - (o.x0 || 0))) * W;

  // grid + y ticks, drawn first so every mark sits on top of them. Three steps, not four: the grid is
  // there to be read past, and the app's charts carry a floor, a middle and a top.
  const tickFmt = o.yFmt || (yHi >= 10000 ? kTickText : tickText);
  for (const t of ticks(yLo, yHi, o.yTicks || 3)) {
    svg.append(svgEl('line', { class: 'g-line', x1: pad.l, x2: pad.l + W, y1: y(t), y2: y(t) }));
    svg.append(txt(tickFmt(t), { class: 'g-tick', x: pad.l - 5, y: y(t) + 3, 'text-anchor': 'end' }));
  }
  svg.append(svgEl('line', { class: 'g-axis', x1: pad.l, x2: pad.l + W, y1: y(yLo), y2: y(yLo) }));
  if (o.yTitle) svg.append(txt(o.yTitle, { class: 'g-atitle g-unit', x: 0, y: 9 }));
  return { svg, x, y, band, pad, W, H, width, height, yLo, yHi, cats };
}

/** A bar with its data-end rounded (4 px) and its base square on the axis — the end that carries
 *  the value is the one that gets the radius. */
function barPath(x0, x1, yTop, yBase) {
  const r = Math.min(4, (x1 - x0) / 2, Math.abs(yBase - yTop));
  return `M${x0},${yBase} L${x0},${yTop + r} Q${x0},${yTop} ${x0 + r},${yTop} `
       + `L${x1 - r},${yTop} Q${x1},${yTop} ${x1},${yTop + r} L${x1},${yBase} Z`;
}

// ── the tooltip ─────────────────────────────────────────────────────────────

const tip = $('tip');
/** Every tooltip a chart can show, by a stable id (chart, configuration, level…): its text, or a function
 *  that recomputes it, or — under `<chart>|*` — a resolver for a whole chart's points. A rebuild (the
 *  minute tick, a resize) redraws every SVG; the tooltip that was showing then keeps showing the SAME
 *  point with its rebuilt figures (retip) instead of vanishing under the reader's pointer. */
const TIPS = new Map();
/** The tip was opened by a TAP. It then STAYS — the compat mouse events a tap fires afterwards (a
 *  mousemove, then a mouseleave ~50 ms later, which used to hide it before it could be read) must not
 *  touch it — until a tap on nothing, or a second tap on the same mark. */
let tipTouched = false;
function showTip(ev, html, id) {
  tip.innerHTML = html;
  tip.dataset.id = id || '';
  tip.hidden = false;
  // A narrow window: a sheet across the top, dismissed by a tap on nothing — beside a finger there is no
  // room for a 17 rem box that does not cover the chart it describes.
  const sheet = window.innerWidth < 640;
  tip.classList.toggle('sheet', sheet);
  if (sheet) { tip.style.left = ''; tip.style.top = ''; return; }
  const b = tip.getBoundingClientRect();
  const x = Math.min(window.innerWidth - b.width - 8, Math.max(8, ev.clientX + 12));
  // Above the pointer when there is room, else BELOW it: a tip clamped to the top edge blanketed the very
  // point it described on a short window.
  const above = ev.clientY - b.height - 12;
  const y = above >= 8 ? above : Math.max(8, Math.min(window.innerHeight - b.height - 8, ev.clientY + 18));
  tip.style.left = `${x}px`;
  tip.style.top = `${y}px`;
}
const hideTip = () => { tip.hidden = true; tipTouched = false; };
/** After a rebuild: the tooltip that was showing stays, with its point's rebuilt text — or goes, when
 *  that point no longer exists. */
function retip() {
  if (tip.hidden) return;
  const id = tip.dataset.id || '';
  let t = TIPS.get(id);
  if (t == null && id.includes('|')) {
    const cut = id.lastIndexOf('|');
    const chart = TIPS.get(`${id.slice(0, cut)}|*`);
    t = chart ? chart(id.slice(cut + 1)) : null;
  }
  if (t == null) { hideTip(); return; }
  tip.innerHTML = typeof t === 'function' ? t() : t;
}
/** Hover on a mark: a hit area bigger than the mark itself, and the tooltip follows the pointer. A TAP
 *  toggles it instead, and it stays put (tipTouched) — mouse events never hide a tapped tip. `id` names
 *  the point across rebuilds; its text is registered under it (TIPS). */
function hoverable(node, html, id) {
  if (id) TIPS.set(id, html);
  node.classList.add('tip-src');
  node.addEventListener('mousemove', (ev) => { if (!tipTouched) showTip(ev, html, id); });
  node.addEventListener('mouseleave', () => { if (!tipTouched) hideTip(); });
  node.addEventListener('touchstart', (ev) => {
    if (id && tipTouched && !tip.hidden && tip.dataset.id === id) { hideTip(); return; }
    showTip(ev.touches[0], html, id);
    tipTouched = true;
  }, { passive: true });
}
// A tap on anything that is not a mark closes a tapped tip.
document.addEventListener('touchstart', (ev) => {
  if (tipTouched && !(ev.target.closest && ev.target.closest('.tip-src, .tip'))) hideTip();
}, { passive: true });

/** A rebuild replaces markup, and a `<details>` the reader opened must not fold on the minute tick: the
 *  open ones under `host` are remembered by their `data-keep` name and reopened after `fn` has rebuilt. */
function keepingOpen(host, fn) {
  const open = new Set([...host.querySelectorAll('details[data-keep][open]')].map((d) => d.dataset.keep));
  fn();
  for (const d of host.querySelectorAll('details[data-keep]')) if (open.has(d.dataset.keep)) d.open = true;
}

// ── the batch-size comparison: one workload, one measure, against max_num_seqs ─

/** The workloads a result's config stamp can name (worklog/bench/annotate.py), in the order the
 *  page shows them, and the measures each one is compared on — the operator's own list, per
 *  workload. `what` is the line that says what the workload answers. A workload this list has
 *  never heard of still gets its panel: the stamp's own name, and whichever of GENERIC_MEASURES its
 *  runs actually carry. */
const WORKLOADS = {
  realdecode: { name: 'Real decode',
    what: 'Decode-bound: real user prompts with ~2k tokens of real context each',
    measures: ['agg_tok_s', 'per_request', 'ttft_mean', 'wall_s', 'spec_accept'] },
  'realmixed-cold': { name: 'Real mixed · cold cache',
    what: 'The deep-research shape, first pass: every 10–20k-token prompt prefilled from scratch (0% prefix hits)',
    measures: ['wall_s', 'agg_tok_s', 'prompt_tok_s', 'prefix_hit'] },
  'realmixed-warm': { name: 'Real mixed · warm cache',
    what: 'The same prompts re-sent straight after the cold pass, so the prefix cache is warm — a different question from the cold pass, never averaged with it',
    measures: ['wall_s', 'agg_tok_s', 'prompt_tok_s', 'prefix_hit'] },
  'prefill-probe': { name: 'Prefill probe',
    what: 'Pure prefill (max_tokens 1): what the engine computes when it does nothing else. It reads the engine counters at the start and the end only',
    measures: ['prefill_tok_s', 'tflops', 'pct_peak', 'wall_s'] },
  synthetic: { name: 'Synthetic, 2000-token prompts',
    what: 'The older synthetic series. Its settings were stamped from the run labels, not read off the engine',
    measures: ['agg_tok_s', 'wall_s', 'per_request', 'ttft_mean'] },
};
const GENERIC_MEASURES = ['agg_tok_s', 'wall_s', 'per_request', 'ttft_mean', 'prompt_tok_s',
  'prefill_tok_s', 'tflops'];
const workloadName = (w) => (WORKLOADS[w] ? WORKLOADS[w].name : String(w));
const workloadRank = (w) => { const i = Object.keys(WORKLOADS).indexOf(w); return i < 0 ? 99 : i; };
/** The measures a workload is drawn on: its declared list minus any that no run of it recorded —
 *  an empty chart says nothing a missing one does not. */
const measuresOf = (w, runs) => ((WORKLOADS[w] && WORKLOADS[w].measures) || GENERIC_MEASURES)
  .filter((m) => MEASURES[m] && runs.some((r) => num(MEASURES[m].get(r)) != null));

/** Why a run is in no comparison, or '' when it is placed. A run is only ever placed by what its
 *  own file records — a stamped workload and a known cap — never by a guess at its name. */
function unplacedWhy(r) {
  if (!r.ok) return 'unreadable';
  if (!r.workload) return 'no config stamp';
  if (num(r.max_num_seqs) == null) return 'no max_num_seqs';
  if (!comparable(r)) return 'not serving';
  return '';
}
const UNPLACED_TEXT = {
  'no config stamp': 'no config stamp, so the file does not say which workload or settings it ran under '
    + '(worklog/bench/annotate.py writes the stamp; a run saved after its last pass has none yet)',
  'no max_num_seqs': 'the file records no max_num_seqs, so there is no place on the axis to put it',
  'not serving': 'the cluster was not serving when it started, so it is not comparable with a run that was',
  unreadable: 'the file cannot be read',
};

/** Settings a reader must not take for a measurement: a cap, or a spec / batched pair, read out of
 *  a LABEL (the older synthetic series) rather than off the engine's argv or asked of the LB. */
const derivedSettings = (r) => !['verified', 'recorded'].includes(r.max_num_seqs_source)
  || ['label', 'stated'].includes(r.config_source);

/** A series is one (spec, batched) pair: HUE from spec, SHAPE from batched, each by the value's rank
 *  among ALL the saved runs — never among the filtered ones — so narrowing a filter never repaints
 *  a survivor (the cap ramp's own rule). A third spec value takes --cmp-3, validated with the first
 *  two; past that, or unrecorded, a series is the neutral ink and its legend line names it. */
const SPEC_HUES = ['var(--cmp-1)', 'var(--cmp-2)', 'var(--cmp-3)'];
const SHAPES = ['circle', 'diamond', 'square', 'triangle'];
/** A recorded MINUTE (a window as a run, `r.w`) is in the series of its whole configuration but the cap —
 *  variantKey: the real-traffic panel must never pool two KV pins into one bar. A saved run keeps its (spec,
 *  batched) pair, the only settings it records. */
const seriesKey = (r) => (r.w ? variantKey(cfgOf(r.w)) : `${num(r.spec) ?? '?'}|${num(r.batched) ?? '?'}`);
function specColor(spec) {
  const i = state.specs.indexOf(num(spec));
  return i >= 0 && i < SPEC_HUES.length ? SPEC_HUES[i] : 'var(--ink-faint)';
}
function batchedShape(b) {
  const i = state.batched.indexOf(num(b));
  return i >= 0 && i < SHAPES.length ? SHAPES[i] : 'circle';
}
const seriesText = (s) => (s.cfg ? partsText(s.cfg) : `spec ${s.spec ?? '?'} · batched ${s.batched == null ? '?' : fmtInt(s.batched)}`);
/** The series in these runs, in one fixed order: spec, then batched, unrecorded last. */
function seriesIn(runs) {
  const seen = new Map();
  for (const r of runs) {
    const k = seriesKey(r);
    if (!seen.has(k)) seen.set(k, { key: k, spec: num(r.spec), batched: num(r.batched), cfg: r.w ? cfgOf(r.w) : null });
  }
  const last = (v) => (v == null ? Infinity : v);
  return [...seen.values()].sort((a, b) => (a.cfg && b.cfg ? cmpTokens(a.key, b.key)
    : last(a.spec) - last(b.spec) || last(a.batched) - last(b.batched)));
}

/** One mark in its series' hue and shape. HOLLOW when its settings were read from a label: the
 *  table's "*", drawn. A hollow mark's stroke is set as a STYLE — the .g-dot rule would beat a
 *  presentation attribute and paint it the surface colour. */
function markEl(shape, cx, cy, r, color, hollow) {
  let n;
  if (shape === 'diamond') {
    const d = r * 1.3;
    n = svgEl('path', { d: `M${cx},${cy - d}L${cx + d},${cy}L${cx},${cy + d}L${cx - d},${cy}Z` });
  } else if (shape === 'square') {
    const d = r * 0.92;
    n = svgEl('rect', { x: cx - d, y: cy - d, width: 2 * d, height: 2 * d, rx: 1 });
  } else if (shape === 'triangle') {
    const d = r * 1.3;
    n = svgEl('path', { d: `M${cx},${cy - d}L${cx + d},${cy + d * 0.75}L${cx - d},${cy + d * 0.75}Z` });
  } else {
    n = svgEl('circle', { cx, cy, r });
  }
  n.setAttribute('class', hollow ? 'g-dot hollow' : 'g-dot');
  if (hollow) { n.setAttribute('fill', 'var(--panel)'); n.style.stroke = color; }
  else n.setAttribute('fill', color);
  return n;
}
/** The same mark as a 14 px icon, for a legend line. */
function markIcon(s, hollow) {
  const svg = svgEl('svg', { width: 14, height: 14, viewBox: '0 0 14 14', 'aria-hidden': 'true', class: 'lg-mark' });
  svg.append(markEl(batchedShape(s.batched), 7, 7, 4, specColor(s.spec), hollow));
  return svg.outerHTML;
}

/** A value as the reader should see it: a fraction as a percent, the unit said, and one decimal in
 *  a tooltip or the table view — 179.6 s against 387.6 s must survive being printed. */
function valText(m, v, precise) {
  const n = num(v);
  if (n == null) return '—';
  if (m.frac) return `${(n * 100).toFixed(1)}%`;
  const t = precise ? (Math.abs(n) >= 1000 ? fmtInt(n) : n.toFixed(Math.abs(n) >= 10 ? 1 : 2)) : fmt(n);
  return m.unit === '%' ? `${t}%` : (m.unit ? `${t} ${m.unit}` : t);
}
const axisTitle = (m) => (m.frac || m.unit === '%' ? 'percent' : m.unit === 's' ? 'seconds' : m.unit || m.name);
const meanOf = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;
/** A quantile of a LIST OF SAMPLES, sorted ascending, by NEAREST RANK — the figure the comparison table
 *  has printed for a cell's minutes since it was written, named here because three places now read it.
 *
 *  It is not pooledQuantile and must not be: that one reads a HISTOGRAM, where the lowest bucket's range
 *  really does begin at 0, so it interpolates up from 0 to the lowest bound. Over a list of recorded
 *  minutes that interpolation would report a p10 BELOW the slowest minute anybody recorded — a number no
 *  minute measured. Nearest rank only ever returns a figure that was actually observed. */
const sampleQuantile = (sorted, q) => (sorted && sorted.length
  ? sorted[Math.min(sorted.length - 1, Math.max(0, Math.round(q * (sorted.length - 1))))] : null);
const cap1 = (s) => s.charAt(0).toUpperCase() + s.slice(1);
/** Which way is better, from MEASURES' one declaration. Beside a chart title the WORDS are printed —
 *  which way is better is load-bearing, and a lone 0.6 rem arrow was read past; a table heading keeps
 *  the arrow alone (its column is narrow) with the words for a screen reader. */
function dirHTML(m, wordy = true) {
  const spec = MEASURES[m];
  if (!spec) return '';
  const word = spec.dir > 0 ? 'higher is better' : 'lower is better';
  const arrow = spec.dir > 0 ? '▲' : '▼';
  if (wordy) return `<span class="dir"><i aria-hidden="true">${arrow}</i> ${word}</span>`;
  return `<i class="dir" title="${word}" aria-hidden="true">${arrow}</i><span class="vh"> — ${word}</span>`;
}

/** Repetitions of one cluster generation side by side, and a wider step between generations: reps
 *  of one generation agree within 1-3% while two generations have differed by 5-10%, and a cluster
 *  of marks per generation is how that reads without a legend of its own. A run whose generation
 *  was not recorded is grouped by its label's sweep group (`post-seqs24`) instead. */
const genKey = (r) => (num(r.generation) != null ? `g${r.generation}` : `l${r.group}`);
function byGeneration(a, b) {
  const ga = num(a.r.generation) ?? Infinity, gb = num(b.r.generation) ?? Infinity;
  return ga - gb || String(a.r.group).localeCompare(String(b.r.group))
    || (num(a.r.rep) || 0) - (num(b.r.rep) || 0) || String(a.r.label).localeCompare(String(b.r.label));
}
function spreadXs(items, cx, room) {
  const offs = [];
  let o = 0;
  items.forEach((p, k) => {
    if (k) o += genKey(p.r) === genKey(items[k - 1].r) ? 1 : 1.6;
    offs.push(o);
  });
  const step = o ? Math.min(8, room / o) : 0;
  return offs.map((v) => cx + (v - o / 2) * step);
}

/** ONE measure of ONE workload against max_num_seqs. Per setting and series: a bar at the mean of
 *  its runs, and every run its own mark on it — the spread between repetitions is part of what is
 *  being read, not noise to average away. `highlight` rings one run (the one that is open) and
 *  shades its column, so "where does this run sit" is answered in the chart itself.
 *
 *  The same chart draws the RECORDED MINUTES of real traffic: `spec` is then the measure with a
 *  window getter, `pool` gives the bar its pooled figure (the table's number, not a mean of means),
 *  past `dense` items a cell's marks become small translucent dots with the hover on the bar, and a
 *  cell with fewer than `minN` items is an OUTLINE labelled with its count — shown, never compared. */
function compChart(host, o) {
  const m = o.spec || MEASURES[o.measure];
  const noun = o.noun || 'runs';
  host.querySelectorAll('svg, .empty').forEach((n) => n.remove());
  const pts = [];
  for (const r of o.runs) {
    const v = num(m.get(r));
    if (v != null) pts.push({ r, v });
  }
  if (!pts.length) {
    host.append(Object.assign(document.createElement('p'), { className: 'empty', textContent: 'no run of this workload recorded it' }));
    return;
  }
  const top = Math.max(...pts.map((p) => p.v));
  const f = frame(host, { xCat: o.caps, yHi: (m.frac ? Math.min(1, top * 1.25) : top * 1.25) || 1,
    yTitle: axisTitle(m), yFmt: m.frac ? (v) => `${Math.round(v * 100)}%` : null,
    aspect: o.aspect || 0.62, title: `${m.name} by max_num_seqs` });
  const S = o.series;
  const inner = f.band * 0.86;
  // One bar width for the whole chart (the same visual weight at every setting), sized as if every
  // series were present; the series that ARE present at a setting are then centred on its tick in
  // the fixed series order. Hue and shape carry the identity, so a position that shifts with the
  // company a bar keeps costs nothing — a bar sitting beside the wrong tick would.
  const barW = Math.min(24, (inner / Math.max(1, S.length)) * 0.6);
  const hl = pts.find((p) => p.r.id === o.highlight);
  const hlCap = hl ? o.caps.indexOf(num(hl.r.max_num_seqs)) : -1;
  if (hlCap >= 0) {
    f.svg.append(svgEl('rect', { class: 'g-hl', x: f.x(hlCap) - f.band / 2 + 1, y: f.pad.t,
      width: Math.max(0, f.band - 2), height: f.H }));
  }
  let ring = null;
  o.caps.forEach((cap, ci) => {
    const here = S.map((s) => ({ s, items: pts.filter((p) => seriesKey(p.r) === s.key && num(p.r.max_num_seqs) === cap) }))
      .filter((c) => c.items.length);
    const sub = inner / Math.max(1, here.length);
    here.forEach(({ s, items }, k) => {
      items.sort(byGeneration);
      // Real traffic: the configuration's one page-wide style (its cap's step, hatched for a variant).
      // A saved run keeps the archive's own encoding: hue = spec, shape = batched.
      const st = o.style ? o.style(s, cap) : null;
      const col = st ? st.color : specColor(s.spec);
      const vals = items.map((p) => p.v);
      const pooled = o.pool ? num(o.pool(items.map((p) => p.r))) : null;
      const mu = pooled != null ? pooled : meanOf(vals), lo = Math.min(...vals), hi = Math.max(...vals);
      const few = o.minN != null && items.length < o.minN;
      const dense = o.dense != null && items.length > o.dense;
      const x = f.x(ci) - inner / 2 + sub * (k + 0.5);
      // Real-traffic cells carry the settings key of their minutes (data-k), so the matrix's selection can
      // recede every other configuration here as it does in the time charts. A saved run has none.
      const kOf = (p) => (p.r.w ? settingsKey(p.r.w) : null);
      const cellK = kOf(items[0]);
      const bar = svgEl('path', { class: `g-bar cmp-bar${few ? ' few' : ''}`, fill: st ? fillFor(f.svg, st) : col, 'data-k': cellK,
        d: barPath(x - barW / 2, x + barW / 2, f.y(mu), f.y(f.yLo)) });
      if (few) bar.style.stroke = col;             // an outline in its hue: shown, not compared
      // The archive's bars carry no shape, so a second batched value gets a dashed ink outline there (its
      // legend says which); a real-traffic variant is hatched by its style instead.
      if (!st && dense && !few && state.batched.length > 1 && num(s.batched) !== state.batched[0]) bar.classList.add('alt-batched');
      f.svg.append(bar);
      const sorted = [...vals].sort((a, b) => a - b);
      const rank = (q) => sampleQuantile(sorted, q);
      hoverable(bar, `<b>${esc(valText(m, mu, true))}</b> <span class="tk">${o.pool
        ? `pooled over ${items.length} ${noun}` : `mean of ${items.length} run${items.length > 1 ? 's' : ''}`}</span>`
        + `<br><span class="tk">max_num_seqs</span> ${esc(String(cap))} &middot; ${esc(seriesText(s))}`
        + (dense
          ? `<br><span class="tk">one ${esc(noun.replace(/s$/, ''))}: p10</span> ${esc(valText(m, rank(0.1), true))}`
            + ` <span class="tk">median</span> ${esc(valText(m, rank(0.5), true))}`
            + ` <span class="tk">p90</span> ${esc(valText(m, rank(0.9), true))}`
          : (items.length > 1 ? `<br><span class="tk">range</span> ${esc(valText(m, lo, true))} – ${esc(valText(m, hi, true))}`
            + (mu ? ` <span class="tk">(${esc((((hi - lo) / Math.abs(mu)) * 100).toFixed(1))}% of the mean)</span>` : '') : '')
            + `<br><span class="tk">${esc(items.map((p) => p.r.label).join(', '))}</span>`)
        + (few ? `<br><span class="tk">fewer than ${o.minN} ${esc(noun)}: shown, not compared</span>` : ''),
      o.tipId ? `${o.tipId}|${m.name}|${cap}|${s.key}` : null);
      if (dense) {
        // Every item still gets its dot — small, translucent, spread by a fixed low-discrepancy
        // sequence so the same data always draws the same picture. Its hover is the bar's.
        const room = Math.max(barW, Math.min(sub * 0.86, barW + 18));
        items.forEach((p, i) => {
          f.svg.append(svgEl('circle', { class: 'g-dot dense', fill: col, r: 1.6, 'data-k': kOf(p),
            cx: x + ((((i + 1) * 0.6180339887) % 1) - 0.5) * room, cy: f.y(p.v) }));
        });
        // The pooled figure over the cloud, in ink: the number the table prints, readable through it.
        f.svg.append(svgEl('line', { class: 'g-meanline', x1: x - room / 2 - 3, x2: x + room / 2 + 3,
          y1: f.y(mu), y2: f.y(mu), 'data-k': cellK }));
        const label = few ? `n=${items.length}` : (m.frac ? `${(mu * 100).toFixed(0)}%` : fmt(mu));
        if (label.length * 6.4 + 4 <= sub) {
          f.svg.append(txt(label, { class: 'g-label', x, y: f.y(Math.max(mu, rank(0.9))) - 8, 'text-anchor': 'middle', 'data-k': cellK }));
        }
        return;
      }
      const xs = spreadXs(items, x, Math.max(barW, Math.min(sub * 0.86, barW + 8 * items.length)));
      // Each mark's hover target is a 22 px tall strip reaching HALFWAY to its neighbours, never
      // over them: with 22 px circles the next mark's target covered this one's centre, and hovering
      // a run showed its neighbour's tooltip. The outer two reach 11 px out, or less where the next
      // series' marks are closer than that.
      const reach = Math.max(4, Math.min(11, (sub - (xs[xs.length - 1] - xs[0])) / 2 - 1));
      items.forEach((p, i) => {
        const derived = derivedSettings(p.r);
        const mark = markEl(st ? 'circle' : batchedShape(s.batched), xs[i], f.y(p.v), 4, col, derived);
        if (kOf(p)) mark.setAttribute('data-k', kOf(p));
        f.svg.append(mark);
        const left = i ? (xs[i - 1] + xs[i]) / 2 : xs[i] - reach;
        const right = i < xs.length - 1 ? (xs[i] + xs[i + 1]) / 2 : xs[i] + reach;
        const hit = svgEl('rect', { class: 'g-hit', x: left, y: f.y(p.v) - 11, width: Math.max(1, right - left), height: 22 });
        f.svg.append(hit);
        const mine = p.r.id === o.highlight;
        if (mine) ring = { x: xs[i], y: f.y(p.v) };
        hoverable(hit, `<b>${esc(valText(m, p.v, true))}</b> ${esc(p.r.label)}${mine ? ' <span class="tk">&mdash; this run</span>' : ''}`
          + `<br><span class="tk">max_num_seqs</span> ${esc(String(cap))} &middot; ${esc(seriesText(s))}`
          + `<br><span class="tk">generation</span> ${esc(p.r.generation == null ? 'not recorded' : String(p.r.generation))}`
          + `<br><span class="tk">${derived ? 'settings read from the run label, not verified' : 'settings read back off the engine argv'}</span>`,
        o.tipId ? `${o.tipId}|${m.name}|${p.r.id}` : null);
      });
      // The mean, printed once per bar when it fits its slot. One that would run into the next
      // series' label is left to the tooltip and the table view rather than drawn over it.
      const label = few ? `n=${items.length}` : (m.frac ? `${(mu * 100).toFixed(0)}%` : fmt(mu));
      if (label.length * 6.4 + 4 <= sub) {
        f.svg.append(txt(label, { class: 'g-label', x, y: f.y(Math.max(mu, hi)) - 8, 'text-anchor': 'middle', 'data-k': cellK }));
      }
    });
    if (!here.length) {
      f.svg.append(txt('not run', { class: 'g-none', x: f.x(ci), y: f.pad.t + f.H - 6, 'text-anchor': 'middle' }));
    }
    f.svg.append(txt(String(cap), { class: 'g-tick', x: f.x(ci), y: f.pad.t + f.H + 14, 'text-anchor': 'middle' }));
  });
  if (ring) f.svg.append(svgEl('circle', { class: 'g-ring', cx: ring.x, cy: ring.y, r: 8.5 }));
  f.svg.append(txt('max_num_seqs', { class: 'g-atitle', x: f.pad.l + f.W / 2, y: f.height - 4, 'text-anchor': 'middle' }));
  host.append(f.svg);
}

/** The legend a panel's charts share: each series' own mark and name, hollow when every run of it
 *  had its settings read from a label; and the ring, when a run is open. */
function legendHTML(series, runs, ring) {
  const items = series.map((s) => {
    const mine = runs.filter((r) => seriesKey(r) === s.key);
    const hollow = mine.length > 0 && mine.every(derivedSettings);
    return `<span class="lg">${markIcon(s, hollow)}${esc(seriesText(s))}${hollow
      ? ' <span class="derived" title="spec, batched and max_num_seqs were stamped from the run labels, not read back off the engine">from labels</span>' : ''}</span>`;
  });
  if (ring) {
    items.push('<span class="lg"><svg class="lg-mark" width="18" height="18" viewBox="0 0 18 18" aria-hidden="true">'
      + '<circle class="g-ring" cx="9" cy="9" r="7"></circle></svg>the run that is open</span>');
  }
  return items.join('');
}

/** The table view the panel's charts owe: per setting and series, how many runs, and each measure's
 *  mean with the range the runs covered — one decimal, so the numbers the bars round are exact here. */
function twinHTML(runs, caps, series, measures) {
  const rows = [];
  for (const cap of caps) {
    for (const s of series) {
      const cell = runs.filter((r) => seriesKey(r) === s.key && num(r.max_num_seqs) === cap);
      if (!cell.length) continue;
      const tds = measures.map((m) => {
        const vals = cell.map((r) => num(MEASURES[m].get(r))).filter((v) => v != null);
        if (!vals.length) return '<td class="num dim">—</td>';
        const lo = Math.min(...vals), hi = Math.max(...vals);
        return `<td class="num"><b>${esc(valText(MEASURES[m], meanOf(vals), true))}</b>${vals.length > 1
          ? `<span class="rng">${esc(valText(MEASURES[m], lo, true))} – ${esc(valText(MEASURES[m], hi, true))}</span>` : ''}</td>`;
      }).join('');
      rows.push(`<tr><td class="num">${esc(String(cap))}</td><td>${markIcon(s, cell.every(derivedSettings))} ${esc(seriesText(s))}</td>`
        + `<td class="num" title="${esc(cell.map((r) => r.label).join(', '))}">${cell.length}</td>${tds}</tr>`);
    }
  }
  return `<details class="cmp-table"><summary>Table view — mean and range per setting</summary>
    <div class="tablewrap"><table class="runs-t twin"><thead><tr><th scope="col" class="num">seqs</th>
    <th scope="col">series</th><th scope="col" class="num">runs</th>${measures.map((m) =>
      `<th scope="col" class="num">${esc(MEASURES[m].name)}${dirHTML(m, false)}</th>`).join('')}</tr></thead>
    <tbody>${rows.join('')}</tbody></table></div></details>`;
}

/** The placed runs, one group per workload (and model: two models are two machines), in the page's
 *  workload order. */
function byWorkload(runs) {
  const map = new Map();
  for (const r of runs) {
    if (unplacedWhy(r)) continue;
    const k = `${r.workload}\u0000${r.model ?? ''}`;
    if (!map.has(k)) map.set(k, { workload: r.workload, model: r.model, runs: [] });
    map.get(k).runs.push(r);
  }
  return [...map.values()].sort((a, b) => workloadRank(a.workload) - workloadRank(b.workload)
    || String(a.workload).localeCompare(String(b.workload)) || String(a.model).localeCompare(String(b.model)));
}
const capsOf = (runs) => [...new Set(runs.map((r) => num(r.max_num_seqs)).filter((v) => v != null))].sort((a, b) => a - b);
/** A panel is one workload OF ONE MODEL, and so is its id: two models' panels of one workload shared an
 *  id, and "the full comparison ↑" of a run of the second jumped to the first model's panel. */
const panelId = (g) => `cmp-${[g.workload, g.model].filter((v) => v != null && v !== '').map((v) => String(v).toLowerCase().replace(/[^a-z0-9]+/g, '-')).join('--')}`;

/** One workload's panel: what it is, the legend its charts share, a small multiple per measure and
 *  the table view. The charts are drawn by `drawPanel` once the panel is in the DOM — they are built
 *  at its real pixel width. */
function panelHTML(g, caps, o) {
  const runs = g.runs;
  const series = seriesIn(runs);
  const measures = measuresOf(g.workload, runs);
  const desc = runs.map((r) => r.workload_desc).find(Boolean);
  const gens = [...new Set(runs.map((r) => num(r.generation)).filter((v) => v != null))].sort((a, b) => a - b);
  const models = new Set((state.index ? state.index.runs : []).map((r) => r.model).filter(Boolean));
  const derived = runs.filter(derivedSettings).length;
  return `<article class="cmp-panel"${o.id ? ` id="${esc(o.id)}"` : ''}>
    <header class="cmp-phead">
      <h3>${esc(workloadName(g.workload))}${models.size > 1 && g.model ? ` <span class="sub">${esc(g.model)}</span>` : ''}</h3>
      ${desc ? `<span class="sub">${esc(desc)}</span>` : ''}
      <span class="sub">${runs.length} run${runs.length === 1 ? '' : 's'}${gens.length
        ? ` · cluster generation${gens.length === 1 ? '' : 's'} ${esc(gens.join(', '))}` : ''}</span>
    </header>
    ${WORKLOADS[g.workload] ? `<p class="why">${esc(WORKLOADS[g.workload].what)}.</p>` : ''}
    <div class="cmp-legend">${legendHTML(series, runs, o.ring)}</div>
    <div class="cmp-grid">${measures.map((m) => `<figure class="cmp-chart" data-m="${esc(m)}">
      <h4>${esc(cap1(MEASURES[m].chart || MEASURES[m].name))}${dirHTML(m)}</h4></figure>`).join('')}</div>
    ${derived ? `<p class="why cmp-note">Hollow marks: ${derived === runs.length ? 'every run here' : `${derived} of these runs`}
      had its settings stamped from its label (<i>${esc(runs.find(derivedSettings).config_text || 'label')}</i>),
      not read back off the engine&rsquo;s argv.</p>` : ''}
    ${o.twin === false ? '' : twinHTML(runs, caps, series, measures)}
  </article>`;
}
function drawPanel(panel, g, caps, highlight) {
  const series = seriesIn(g.runs);
  for (const fig of panel.querySelectorAll('.cmp-chart')) {
    compChart(fig, { runs: g.runs, measure: fig.dataset.m, caps, series, highlight });
  }
}

/** The whole comparison, over the FILTERED runs: one panel per workload, every panel on the same
 *  max_num_seqs axis so a column reads across panels, and — so nothing looks silently dropped — the
 *  runs that are in no panel, with the reason each is not. */
function renderCompare(runs) {
  const box = $('cmp-panels');
  const groups = byWorkload(runs);
  const caps = capsOf(groups.flatMap((g) => g.runs));
  box.innerHTML = groups.map((g) => panelHTML(g, caps, { id: panelId(g) })).join('')
    || '<p class="empty">no run matches these filters and carries a config stamp</p>';
  const panels = box.querySelectorAll('.cmp-panel');
  groups.forEach((g, i) => drawPanel(panels[i], g, caps, state.selected));
  const out = new Map();
  for (const r of runs) {
    const why = r.ok ? unplacedWhy(r) : '';
    if (!why) continue;
    if (!out.has(why)) out.set(why, []);
    out.get(why).push(r.label);
  }
  const note = $('cmp-unplaced');
  const n = [...out.values()].reduce((a, l) => a + l.length, 0);
  note.hidden = !n;
  if (n) {
    note.innerHTML = `<b>${n} run${n === 1 ? ' is' : 's are'} in no comparison</b>`
      + [...out.entries()].map(([why, labels]) => `<span>${esc(UNPLACED_TEXT[why] || why)}: `
        + `<span class="labels">${labels.map(esc).join(', ')}</span></span>`).join('');
  }
}

/** "Where does this run sit": its workload's panel over EVERY saved run of that workload — not the
 *  filtered view, exactly as the table's value wash is scaled — with this run ringed, and a
 *  sentence that puts it against the other runs at its own setting. */
function whereHTML(r) {
  const why = unplacedWhy(r);
  if (why) {
    return `<div class="where"><h3>Batch size comparison</h3><p class="why">This run is in no
      comparison: ${esc(UNPLACED_TEXT[why] || why)}.</p></div>`;
  }
  return `<div class="where">
    <div class="where-head"><h3>Where this run sits &mdash; ${esc(workloadName(r.workload))}</h3>
      <a class="jump" href="#${esc(panelId(r))}">the full comparison &uarr;</a></div>
    <p class="why" id="d-where-read"></p>
    <div id="d-where-panel"></div></div>`;
}
function drawWhere(r) {
  const host = $('d-where-panel');
  if (!host || !state.index) return;
  const all = state.index.runs.filter((x) => !unplacedWhy(x) && x.workload === r.workload && x.model === r.model);
  if (!all.some((x) => x.id === r.id)) return;
  const g = { workload: r.workload, model: r.model, runs: all };
  const caps = capsOf(state.index.runs.filter((x) => !unplacedWhy(x)));
  host.innerHTML = panelHTML(g, caps, { ring: true, twin: false });
  drawPanel(host.firstElementChild, g, caps, r.id);
  const m = measuresOf(r.workload, all).find((k) => num(MEASURES[k].get(r)) != null);
  if (!m) return;
  const M = MEASURES[m];
  const cell = all.filter((x) => seriesKey(x) === seriesKey(r) && num(x.max_num_seqs) === num(r.max_num_seqs));
  const vals = cell.map((x) => num(M.get(x))).filter((v) => v != null);
  const v = num(M.get(r)), mu = meanOf(vals);
  const d = mu ? ((v - mu) / Math.abs(mu)) * 100 : 0;
  const s = { spec: num(r.spec), batched: num(r.batched) };
  $('d-where-read').innerHTML = `At <b>max_num_seqs ${esc(String(r.max_num_seqs))}</b> &middot; ${esc(seriesText(s))}`
    + (vals.length > 1
      ? ` there are <b>${vals.length} runs</b>; their ${esc(M.name)} averages <b>${esc(valText(M, mu, true))}</b>`
        + ` (${esc(valText(M, Math.min(...vals), true))} – ${esc(valText(M, Math.max(...vals), true))}) and this run's`
        + ` <b>${esc(valText(M, v, true))}</b> is ${esc(Math.abs(d).toFixed(1))}% ${d >= 0 ? 'above' : 'below'} that mean.`
      : ` this is the only run; its ${esc(M.name)} is <b>${esc(valText(M, v, true))}</b>.`)
    + ' It is ringed in every chart below, its column shaded.';
}

// ── chart: the spread inside each run ───────────────────────────────────────

/** One bar per RUN spanning its throughput curve's bottom 10% to its top 10%, with the mean and
 *  the median ticked. This is the chart that answers "it changes up and down": at seqs 8 / 16 / 24
 *  the bottom decile was measured at 53 / 19 / 12 tok/s while the means stayed within a third of
 *  each other, and no chart of means can show that. */
function spreadChart(host, runs) {
  host.querySelectorAll('svg, .empty, .chart-key').forEach((n) => n.remove());
  const items = runs.filter((r) => r.curve && num(r.curve.p10) != null && num(r.curve.p90) != null)
    .sort((a, b) => (num(a.max_num_seqs) || 0) - (num(b.max_num_seqs) || 0) || String(a.label).localeCompare(String(b.label)));
  if (!items.length) {
    host.append(Object.assign(document.createElement('p'), { className: 'empty', textContent: 'no runs match these filters' }));
    return;
  }
  const hi = Math.max(...items.map((r) => r.curve.p90));
  // Two label lines under the axis: the worst decile, then which run it was. Drawn on the SURFACE
  // rather than inside the bar — a dim label over a light bar is a label nobody reads.
  // Twenty-four runs on a phone leave a bar 13 px and its two labels 25: side by side they ran into each
  // other as one unreadable line. Where a bar's slot is narrower than its labels (`tight`), the same two
  // labels are stood on end under their own bar — the decile first, then the run — and none is dropped.
  const tight = (contentWidth(host) - 40) / items.length < 30;
  const f = frame(host, { xCat: items, yHi: hi * 1.12, yTitle: 'generated tok/s', aspect: 0.42,
    padBottom: tight ? 86 : 48, height: tight ? 230 : null, title: 'throughput spread inside each run' });
  const under = (text, cls, x, y) => f.svg.append(txt(text, tight
    ? { class: cls, x, y, 'text-anchor': 'end', transform: `rotate(-90 ${x} ${y})`, dy: '.35em' }
    : { class: cls, x, y, 'text-anchor': 'middle' }));
  items.forEach((r, i) => {
    const col = capColor(r.max_num_seqs);
    const w = Math.min(28, f.band * 0.5);
    const c = r.curve;
    const bar = svgEl('path', { class: 'g-bar', fill: col, opacity: r.kind === 'corpus' ? .62 : .85,
      d: barPath(f.x(i) - w / 2, f.x(i) + w / 2, f.y(c.p90), f.y(c.p10)) });
    f.svg.append(bar);
    hoverable(bar, `<b>${esc(r.label)}</b> &middot; ${esc(String(r.max_num_seqs ?? '?'))} seqs`
      + `<br><span class="tk">bottom 10%</span> ${esc(fmt(c.p10))} tok/s`
      + `<br><span class="tk">median</span> ${esc(fmt(c.median))}`
      + `<br><span class="tk">mean</span> ${esc(fmt(c.mean))}`
      + `<br><span class="tk">top 10%</span> ${esc(fmt(c.p90))}`
      + `<br><span class="tk">samples</span> ${esc(fmtInt(c.n ?? r.samples))}`);
    // Mean and median are annotations on the run's own bar, in ink — not series colours.
    if (num(c.mean) != null) {
      f.svg.append(svgEl('line', { stroke: 'var(--ink)', 'stroke-width': 2,
        x1: f.x(i) - w / 2 - 2, x2: f.x(i) + w / 2 + 2, y1: f.y(c.mean), y2: f.y(c.mean) }));
    }
    if (num(c.median) != null) {
      f.svg.append(svgEl('line', { class: 'g-ann', x1: f.x(i) - w / 2 - 2, x2: f.x(i) + w / 2 + 2,
        y1: f.y(c.median), y2: f.y(c.median) }));
    }
    // The bottom decile gets the one direct label: it is the number that collapses.
    under(fmt(c.p10), 'g-label', f.x(i), f.pad.t + f.H + (tight ? 6 : 14));
    // Which run, in the shortest form that stays unique: the cap and its repetition. With one cap
    // selected, a column of identical "24"s would say nothing.
    under(`${r.max_num_seqs ?? '?'}${r.rep ? `\u00b7r${r.rep}` : ''}`, 'g-tick', f.x(i), f.pad.t + f.H + (tight ? 38 : 27));
  });
  f.svg.append(txt('runs, grouped by max_num_seqs', { class: 'g-atitle', x: f.pad.l + f.W / 2, y: f.height - 4, 'text-anchor': 'middle' }));
  host.append(f.svg);
  const key = document.createElement('p');
  key.className = 'why chart-key';
  key.innerHTML = 'bar = bottom 10% &rarr; top 10% of the samples &middot; '
    + '<span style="color:var(--ink)">solid tick</span> = mean &middot; dashed tick = median &middot; '
    + 'the number under each bar is its worst decile';
  host.append(key);
}

/** One spread chart per comparable group — the groups the table's wash is scaled in — so a real
 *  decode run's curve is never drawn on the axis of a synthetic run's, nor a warm pass beside a cold
 *  one. An unstamped real-prompt run is in no group and is not drawn here; its numbers are in the
 *  table, and the batch-size comparison says why it is in no comparison. */
function renderSpread(runs) {
  const grid = $('spread-grid');
  grid.innerHTML = '';
  const groups = new Map();
  for (const r of runs) {
    if (!r.ok || !r.curve || num(r.curve.p10) == null || num(r.curve.p90) == null) continue;
    const key = workloadOf(r);
    if (key == null) continue;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(r);
  }
  const named = (rs) => rs.find((r) => r.workload) || rs[0];
  const list = [...groups.values()].sort((a, b) => workloadRank(named(a).workload) - workloadRank(named(b).workload));
  if (!list.length) {
    grid.append(Object.assign(document.createElement('p'), { className: 'empty', textContent: 'no runs match these filters' }));
    return;
  }
  // Every figure goes into the grid BEFORE any chart is built: a chart is built at its card's real
  // width, and a card measured while it was still alone in an auto-fit row is the whole row wide.
  // A group is one workload of ONE model (workloadOf): with two models saved, each chart names its own.
  const models = new Set((state.index ? state.index.runs : []).map((r) => r.model).filter(Boolean));
  const figs = list.map((rs) => {
    const first = named(rs);
    const fig = document.createElement('figure');
    fig.className = 'chart';
    fig.innerHTML = `<h3>${esc(first.workload ? workloadName(first.workload) : workloadText(first))}${first.workload && models.size > 1 && first.model
      ? ` <span class="sub">${esc(first.model)}</span>` : ''}</h3>`;
    grid.append(fig);
    return fig;
  });
  list.forEach((rs, i) => spreadChart(figs[i], rs));
}

// ── chart: one measure over time, for one run ───────────────────────────────

/** A single series over the run's clock, with a crosshair and a tooltip. A sample with no value
 *  BREAKS the line — it is a moment nobody measured, and a line drawn through it would be an
 *  invented measurement. `band` shades the p10–p90 range of the samples across the whole width, so
 *  the spread is visible in the same picture as the curve. */
function timeChart(host, series, key, o) {
  host.querySelectorAll('svg, .empty').forEach((n) => n.remove());
  const pts = series.map((s) => ({ t: num(s.t), v: num(s[key]), s })).filter((p) => p.t != null);
  const vals = pts.map((p) => p.v).filter((v) => v != null);
  if (!vals.length) {
    host.append(Object.assign(document.createElement('p'), { className: 'empty', textContent: 'this run recorded no samples' }));
    return;
  }
  const t1 = Math.max(...pts.map((p) => p.t));
  const hi = o.yHi != null ? o.yHi : Math.max(...vals) * 1.1;
  const f = frame(host, { x0: 0, x1: t1 || 1, yHi: hi, yTitle: o.yTitle, aspect: o.aspect || 0.4,
    yFmt: o.yFmt, title: o.title });
  if (o.band && num(o.band.p10) != null && num(o.band.p90) != null) {
    f.svg.append(svgEl('rect', { class: 'g-band', fill: o.color, x: f.pad.l, width: f.W,
      y: f.y(Math.min(o.band.p90, f.yHi)), height: Math.max(1, f.y(o.band.p10) - f.y(Math.min(o.band.p90, f.yHi))) }));
    if (num(o.band.mean) != null) {
      f.svg.append(svgEl('line', { class: 'g-ann', x1: f.pad.l, x2: f.pad.l + f.W, y1: f.y(o.band.mean), y2: f.y(o.band.mean) }));
      f.svg.append(txt(`mean ${fmt(o.band.mean)}`, { class: 'g-annlab', x: f.pad.l + f.W, y: f.y(o.band.mean) - 4, 'text-anchor': 'end' }));
    }
  }
  // The line, broken at every gap.
  let d = '', pen = false;
  for (const p of pts) {
    if (p.v == null) { pen = false; continue; }
    d += `${pen ? 'L' : 'M'}${f.x(p.t).toFixed(1)},${f.y(p.v).toFixed(1)} `;
    pen = true;
  }
  f.svg.append(svgEl('path', { class: `g-mark${o.dashed ? ' corpus' : ''}`, stroke: o.color, d }));
  // x ticks in seconds
  for (const t of ticks(0, t1, 5)) {
    f.svg.append(txt(`${Math.round(t)}s`, { class: 'g-tick', x: f.x(t), y: f.pad.t + f.H + 14, 'text-anchor': 'middle' }));
  }
  // crosshair: one hit rect over the plot, nearest sample wins
  const cross = svgEl('line', { class: 'g-cross', y1: f.pad.t, y2: f.pad.t + f.H, x1: 0, x2: 0, opacity: 0 });
  const mark = svgEl('circle', { class: 'g-dot', r: 4, fill: o.color, opacity: 0 });
  const hit = svgEl('rect', { class: 'g-hit', x: f.pad.l, y: f.pad.t, width: f.W, height: f.H });
  hit.addEventListener('mousemove', (ev) => {
    const box = f.svg.getBoundingClientRect();
    const px = (ev.clientX - box.left) * (f.width / box.width);   // the SVG's own units
    let best = null;
    for (const p of pts) {
      const dx = Math.abs(f.x(p.t) - px);
      if (p.v != null && (best == null || dx < best.dx)) best = { p, dx };
    }
    if (!best) return;
    const p = best.p;
    cross.setAttribute('x1', f.x(p.t)); cross.setAttribute('x2', f.x(p.t)); cross.setAttribute('opacity', 1);
    mark.setAttribute('cx', f.x(p.t)); mark.setAttribute('cy', f.y(p.v)); mark.setAttribute('opacity', 1);
    showTip(ev, `<b>t = ${esc(fmt(p.t, 1))} s</b><br><span class="tk">${esc(o.yTitle)}</span> ${esc(fmt(p.v))}`
      + `<br><span class="tk">running</span> ${esc(fmtInt(p.s.running))} &middot; <span class="tk">waiting</span> ${esc(fmtInt(p.s.waiting))}`
      + `<br><span class="tk">kv</span> ${esc(p.s.kv == null ? '—' : `${(p.s.kv * 100).toFixed(1)}%`)}`);
  });
  hit.addEventListener('mouseleave', () => { cross.setAttribute('opacity', 0); mark.setAttribute('opacity', 0); hideTip(); });
  f.svg.append(cross, mark, hit);
  f.svg.append(txt('time into the run', { class: 'g-atitle', x: f.pad.l + f.W / 2, y: f.height - 4, 'text-anchor': 'middle' }));
  host.append(f.svg);
}

/** Per-request time-to-first-token, one dot per request at the moment it was issued. A run whose
 *  TTFT climbs through the batch is queueing, which no aggregate shows. */
function ttftChart(host, requests) {
  host.querySelectorAll('svg, .empty').forEach((n) => n.remove());
  const pts = requests.filter((r) => num(r.start_rel) != null && num(r.ttft_s) != null);
  if (!pts.length) {
    host.append(Object.assign(document.createElement('p'), { className: 'empty', textContent: 'no per-request records in this run' }));
    return;
  }
  const t1 = Math.max(...pts.map((r) => r.start_rel));
  const f = frame(host, { x0: 0, x1: t1 || 1, yHi: Math.max(...pts.map((r) => r.ttft_s)) * 1.1,
    yTitle: 'ttft s', aspect: 0.34, title: 'time to first token per request' });
  for (const r of pts) {
    const dot = svgEl('circle', { class: 'g-dot', cx: f.x(r.start_rel), cy: f.y(r.ttft_s), r: 3.5,
      fill: r.cold_fill ? 'var(--seq-5)' : 'var(--seq-3)' });
    f.svg.append(dot);
    hoverable(dot, `<b>request ${esc(fmtInt(r.idx))}</b><br><span class="tk">issued at</span> ${esc(fmt(r.start_rel, 1))} s`
      + `<br><span class="tk">ttft</span> ${esc(fmt(r.ttft_s, 2))} s<br><span class="tk">end-to-end</span> ${esc(fmt(r.e2e_s, 1))} s`
      + `<br><span class="tk">prompt / out</span> ${esc(fmtInt(r.prompt_tokens))} / ${esc(fmtInt(r.out_tokens))} tok`
      + (r.cold_fill ? '<br><span class="tk">cold prefix fill</span>' : ''));
  }
  for (const t of ticks(0, t1, 5)) {
    f.svg.append(txt(`${Math.round(t)}s`, { class: 'g-tick', x: f.x(t), y: f.pad.t + f.H + 14, 'text-anchor': 'middle' }));
  }
  f.svg.append(txt('when the request was issued', { class: 'g-atitle', x: f.pad.l + f.W / 2, y: f.height - 4, 'text-anchor': 'middle' }));
  host.append(f.svg);
}

// ── the filter row ──────────────────────────────────────────────────────────

function fillSelect(sel, values, label) {
  const keep = sel.value;
  sel.innerHTML = `<option value="">all</option>`
    + values.map((v) => `<option value="${esc(v)}">${esc(label ? label(v) : v)}</option>`).join('');
  if (values.map(String).includes(keep)) sel.value = keep;
}

function renderFilters() {
  const f = state.index.facets;
  $('filters').hidden = false;
  fillSelect($('f-model'), f.models);
  // The stamped workloads, in the page's own order. A run with no stamp is under "all" only.
  fillSelect($('f-workload'), [...(f.workloads || [])].sort((a, b) => workloadRank(a) - workloadRank(b)), workloadName);
  fillSelect($('f-conc'), f.concurrency);
  fillSelect($('f-prompt'), f.prompt_tokens, (v) => `${fmtInt(v)} tok`);
  fillSelect($('f-out'), f.max_tokens, (v) => `${fmtInt(v)} tok`);
  const box = $('f-seqs');
  box.innerHTML = '';
  if (f.max_num_seqs.length) {
    box.append(Object.assign(document.createElement('span'), {
      textContent: 'seqs', style: 'font-size:.72rem;color:var(--ink-faint)' }));
  }
  for (const cap of f.max_num_seqs) {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'chip';
    b.setAttribute('aria-pressed', state.filters.seqs.has(cap) ? 'true' : 'false');
    b.style.color = state.filters.seqs.has(cap) ? capColor(cap) : '';
    b.innerHTML = `<i class="swatch" style="background:${capColor(cap)}"></i>${cap}`;
    b.addEventListener('click', () => {
      if (state.filters.seqs.has(cap)) state.filters.seqs.delete(cap);
      else state.filters.seqs.add(cap);
      render();
    });
    box.append(b);
  }
}

// ── how good is this number? the value wash ────────────────────────────

/** DIRECTION IS DECLARED, NEVER INFERRED. `dir: 1` = higher is better, `dir: -1` = lower is
 *  better, and the wash normalises by it, so the strongest tier means "best in its group" in a
 *  lower-is-better column too and a reader never has to invert a column in their head. Inferring
 *  the direction from the numbers — or letting one ramp mean "bigger" everywhere — would paint the
 *  slowest run's 220 s wall time as its best result, which is worse than no colour at all.
 *
 *  The keys are what a heading carries as `data-m`. Measures with no column (the opened run's
 *  tiles, the batch-size comparison's charts) declare their direction here too: a chart's ▲/▼ and
 *  a heading's are then one declaration, and nothing is ever left to decide from the numbers.
 *  `unit` is what the value is counted in; `frac` marks a 0..1 ratio the page prints as a percent. */
const MEASURES = {
  agg_tok_s:    { dir:  1, name: 'aggregate tok/s',          unit: 'tok/s', get: (r) => r.agg_tok_s },
  wall_s:       { dir: -1, name: 'total wall seconds',        unit: 's',     get: (r) => r.wall_s },
  curve_p10:    { dir:  1, name: 'bottom 10% of the curve',   unit: 'tok/s', get: (r) => (r.curve || {}).p10 },
  curve_mean:   { dir:  1, name: 'curve mean',                unit: 'tok/s', get: (r) => (r.curve || {}).mean },
  curve_median: { dir:  1, name: 'curve median',              unit: 'tok/s', get: (r) => (r.curve || {}).median },
  curve_p90:    { dir:  1, name: 'top 10% of the curve',      unit: 'tok/s', get: (r) => (r.curve || {}).p90 },
  ttft_mean:    { dir: -1, name: 'time to first token',       unit: 's',     get: (r) => (r.ttft || {}).mean,
                  chart: 'mean time to first token' },
  per_request:  { dir:  1, name: 'per-request tok/s',         unit: 'tok/s', get: (r) => (r.per_request || {}).mean,
                  chart: 'mean per-request tok/s' },
  prefix_hit:   { dir:  1, name: 'prefix cache hit rate',     frac: true,    get: (r) => r.prefix_hit },
  spec_accept:  { dir:  1, name: 'speculation acceptance',    frac: true,    get: (r) => r.spec_accept },
  ms_per_iter:  { dir: -1, name: 'ms per engine iteration',   unit: 'ms',    get: (r) => r.ms_per_iter },
  preemptions:  { dir: -1, name: 'preemptions',               get: (r) => r.preemptions },
  kv_max:       { dir: -1, name: 'peak KV cache used',        frac: true,    get: (r) => r.kv_max },
  // What the long-prompt and prefill workloads are made of (the store's `prompt_tok_s`, and the
  // prefill probe's own PREFILL_TOK_S / TFLOPS_PER_GPU / PCT_OF_NVFP4_PEAK).
  prompt_tok_s:  { dir: 1, name: 'engine prompt tok/s',      unit: 'tok/s',   get: (r) => r.prompt_tok_s },
  prefill_tok_s: { dir: 1, name: 'prefill tok/s',            unit: 'tok/s',   get: (r) => r.prefill_tok_s },
  tflops:        { dir: 1, name: 'TFLOP/s per GPU',          unit: 'TFLOP/s', get: (r) => r.tflops_per_gpu },
  pct_peak:      { dir: 1, name: 'share of the nvfp4 peak',  unit: '%',       get: (r) => r.pct_nvfp4_peak },
  // The recorded minutes of real traffic (perflog.py). `get` reads a WINDOW here: none of these has a
  // column in the saved-runs table, and the real-traffic table and charts read them per minute and
  // pooled (`aggregate`). Inter-token latency is shown in ms; the log keeps seconds.
  computed_tok_s: { dir: 1, name: 'computed prompt tok/s',   unit: 'tok/s',   get: (w) => w.prompt_computed_tok_s },
  ttft_p90:      { dir: -1, name: 'time to first token, p90', unit: 's',       get: (w) => (w.ttft || {}).p90 },
  ttft_p99:      { dir: -1, name: 'time to first token, p99', unit: 's',       get: (w) => pooledQuantile([w], 'ttft', 0.99) },
  // ONE LABEL OVER TWO QUANTITIES, until the recorded window says which engine produced it. A gap is
  // whatever the engine samples: SGLang samples once per EMITTED TOKEN (itl.n / generated pooled to
  // 0.994 over the recorded windows), vLLM once per SPECULATIVE STEP (0.342 / 0.317 / 0.298 for the
  // three vLLM profiles, ~1/accept_len). Drawn as one "inter-token latency", an SGLang pair reads
  // about three times quicker per gap than a vLLM pair at the SAME tok/s. The window carries no
  // engine (perflog.py), so the page cannot normalise one to the other and does not pretend to: the
  // label says whose gap it is, and the comparison is the reader's to make knowingly.
  itl_mean:      { dir: -1, name: 'inter-token latency, per engine gap', unit: 'ms', get: (w) => ms((w.itl || {}).mean) },
  itl_p90:       { dir: -1, name: 'inter-token latency, per engine gap, p90', unit: 'ms', get: (w) => ms((w.itl || {}).p90) },
  itl_p99:       { dir: -1, name: 'inter-token latency, per engine gap, p99', unit: 'ms', get: (w) => ms(pooledQuantile([w], 'itl', 0.99)) },
  waiting_mean:  { dir: -1, name: 'requests waiting',         unit: 'requests', get: (w) => (w.waiting || {}).mean },
  // errors declares its direction and is deliberately NOT washed. A non-zero count is already
  // drawn in --bad, and --bad cannot hold 4.5:1 over the top two tiers (2.21:1 and 1.68:1): the
  // louder, older signal keeps the cell. The heading still shows which way it counts.
  errors:       { dir: -1, name: 'errors', wash: false,       get: (r) => r.errors },
};

/** Seconds as milliseconds, null staying null. */
function ms(v) { const n = num(v); return n == null ? null : n * 1000; }

const TIERS = 4;
const TIER_TOKENS = ['var(--heat-1)', 'var(--heat-2)', 'var(--heat-3)', 'var(--heat-4)'];

/** The set a number may honestly be compared against: the SAME MODEL doing the SAME WORK. A
 *  real-corpus run at 6 tok/s and a 2000-token synthetic run at 305 are both correct and measure
 *  different things — one ramp over both would paint the corpus run as a catastrophic regression.
 *  Runs that differ in ANY of these are never scaled against each other; the tier is not drawn at
 *  all rather than drawn against the wrong set.
 *
 *  The WORK is the result's stamped workload (`workload`). Before the stamp, a real-prompt decode
 *  run and a synthetic one shared every other field here — 100 requests, 32 offered, 512 out, and
 *  load100's unused 2000-token default — and were scaled as one group. A run with no stamp falls
 *  back to its harness kind, which is what lets the unstamped synthetic runs (their args say
 *  synthetic) join the stamped synthetic series. An unstamped REAL-prompt run has no group at all:
 *  the cold and the warm pass of one prompt file carry identical args and differ up to 2x. */
function workloadOf(r) {
  if (!r.workload && r.kind === 'real') return null;
  return [r.workload || r.kind, r.model, r.prompt_tokens, r.max_tokens, r.requests, r.concurrency]
    .map((v) => (v == null ? '?' : v)).join(' | ');
}

/** ...and the same thing in English, for the key above the table and every cell's own tooltip.
 *  A scale the reader cannot see the basis of is a scale they have to trust. */
function workloadText(r) {
  if (r.workload) return [workloadName(r.workload), r.workload_desc, r.model].filter(Boolean).join(' \u00b7 ');
  const bits = [r.kind === 'corpus' ? 'real-corpus' : r.kind === 'real' ? `real prompts (${r.prompts_file})` : 'synthetic'];
  if (num(r.requests) != null) bits.push(`${fmtInt(r.requests)} requests`);
  if (num(r.prompt_tokens) != null || num(r.max_tokens) != null) {
    bits.push(`${fmtInt(r.prompt_tokens)}\u2009\u2192\u2009${fmtInt(r.max_tokens)} tokens`);
  }
  if (num(r.concurrency) != null) bits.push(`offered ${fmtInt(r.concurrency)}`);
  if (r.model) bits.push(r.model);
  return bits.join(' \u00b7 ');
}

/** A run the table already marks "!" — the cluster was not serving when it started — is the run
 *  this page says outright is "not comparable with a run measured while serving". So it sets
 *  nobody's scale and gets no tier of its own. An UNRECORDED state ('' on every run saved before
 *  runmeta.py) is not a claim that it was starting: those stay in, exactly as the "!" mark does. */
const comparable = (r) => !!r.ok && !(r.cluster_state && r.cluster_state !== 'serving');

/** lo/hi per (workload, measure) over EVERY saved run — not over the filtered view. Narrowing a
 *  filter must not repaint the rows that survive it: a cell's colour is a property of the run and
 *  of every run it is comparable with, never of what else happens to be on screen. */
function buildHeat(runs) {
  const scales = new Map();
  for (const r of runs) {
    if (!comparable(r)) continue;
    const key = workloadOf(r);
    if (key == null) continue;
    let g = scales.get(key);
    if (!g) scales.set(key, (g = { runs: 0, text: workloadText(r), m: {} }));
    g.runs += 1;
    for (const [name, spec] of Object.entries(MEASURES)) {
      const v = num(spec.get(r));
      if (v == null) continue;
      const m = g.m[name] || (g.m[name] = { lo: v, hi: v, n: 0 });
      m.lo = Math.min(m.lo, v);
      m.hi = Math.max(m.hi, v);
      m.n += 1;
    }
  }
  return scales;
}

/** One numeric cell: the tier its value earns, and a tooltip that spells out what that tier is
 *  relative to. The value is printed whatever happens, so the wash only ever repeats what the
 *  cell already says — it is an enhancement, never the signal. */
function numTd(r, measure, text) {
  const spec = MEASURES[measure];
  const v = spec ? num(spec.get(r)) : null;
  const dir = spec && spec.dir > 0 ? 'higher is better' : 'lower is better';
  const title = (t) => ` title="${esc(t)}"`;
  if (!spec || spec.wash === false || v == null) return `<td class="num">${text}</td>`;
  if (!comparable(r)) {
    return `<td class="num"${title(`${spec.name} \u2014 not tiered: the cluster was `
      + `'${r.cluster_state}' when this run started, so it is not comparable with a run measured `
      + 'while serving')}>${text}</td>`;
  }
  if (workloadOf(r) == null) {
    return `<td class="num"${title(`${spec.name} ${fmt(v)} \u2014 not tiered: this real-prompt run `
      + 'carries no config stamp, so which workload it ran (a cold or a warm pass?) is not recorded')}>${text}</td>`;
  }
  const g = state.heat && state.heat.get(workloadOf(r));
  const m = g && g.m[measure];
  if (!m || m.n < 2) {
    return `<td class="num"${title(`${spec.name} ${fmt(v)} \u2014 not tiered: no other saved run `
      + `measured this on the same workload (${g ? g.text : workloadText(r)})`)}>${text}</td>`;
  }
  if (m.hi === m.lo) {
    return `<td class="num"${title(`${spec.name} ${fmt(v)} \u2014 not tiered: all ${m.n} runs of `
      + `this workload measured the same value (${g.text})`)}>${text}</td>`;
  }
  const t = (v - m.lo) / (m.hi - m.lo);
  const good = spec.dir > 0 ? t : 1 - t;
  const tier = Math.min(TIERS, 1 + Math.floor(good * TIERS));
  const best = spec.dir > 0 ? m.hi : m.lo;
  const worst = spec.dir > 0 ? m.lo : m.hi;
  return `<td class="num heat" data-heat="${tier}"${title(
    `${spec.name} ${fmt(v)} \u2014 ${dir}, tier ${tier} of ${TIERS}\n`
    + `${Math.round(good * 100)}% of the way from the worst (${fmt(worst)}) to the best `
    + `(${fmt(best)}) of the ${m.n} comparable runs\n`
    + `scale: ${g.text}`)}>${text}</td>`;
}

/** The ▲/▼ beside each measured heading, stamped from MEASURES so the direction a heading
 *  claims and the direction the wash normalises by are one declaration, not two. */
function stampHeadings() {
  for (const th of document.querySelectorAll('table.runs-t th[data-m]')) {
    if (th.querySelector('.dir')) continue;
    const spec = MEASURES[th.dataset.m];
    if (!spec) continue;
    const word = spec.dir > 0 ? 'higher is better' : 'lower is better';
    const mark = document.createElement('i');
    mark.className = 'dir';
    mark.title = word;
    mark.setAttribute('aria-hidden', 'true');
    mark.textContent = spec.dir > 0 ? '\u25b2' : '\u25bc';
    const said = document.createElement('span');
    said.className = 'vh';
    said.textContent = ` — ${word}`;
    th.append(mark, said);
  }
}

/** The key. It says the one thing a value scale can silently lie about: what it is relative to. */
function renderHeatKey(groups) {
  const key = $('heat-key');
  const scaled = [...groups.values()].filter((g) => Object.values(g.m).some((m) => m.n > 1));
  key.hidden = !scaled.length;
  if (!scaled.length) return;
  const ramp = TIER_TOKENS.map((c) => `<i style="background:${c}"></i>`).join('');
  const runs = scaled.reduce((n, g) => n + g.runs, 0);
  key.innerHTML = `<span>cell colour = where the number sits inside its own comparable group,</span>`
    + `<span class="ramp" aria-hidden="true">${ramp}</span>`
    + `<span><b>worst \u2192 best</b>; the \u25b2/\u25bc in each heading says which of those is the `
    + `bigger number.</span>`
    + (scaled.length === 1
      ? `<span>Every tier here is out of <b>${runs} run${runs === 1 ? '' : 's'}</b> of one workload `
        + `&mdash; ${esc(scaled[0].text)}.</span>`
      : `<span>The saved runs form <b>${scaled.length} comparable groups</b> (same model, same `
        + `stamped workload, prompt and output size, request count and offered concurrency); nothing is ever `
        + `scaled across two of them.</span>`)
    + `<span>A run with no comparable sibling is left bare \u2014 hover any number for its scale.</span>`;
}

// ── the run table (and the table view every chart owes) ─────────────────────

function seqCell(r) {
  if (r.max_num_seqs == null) return '<span class="dim" title="neither the result nor its label says">—</span>';
  const dot = `<i class="seqdot" style="background:${capColor(r.max_num_seqs)}"></i>`;
  if (r.max_num_seqs_source === 'verified') {
    return `${dot}<span title="read back off the engine's own argv at launch (the result's config stamp)">${r.max_num_seqs}</span>`;
  }
  if (r.max_num_seqs_source === 'recorded') return `${dot}${r.max_num_seqs}`;
  return `${dot}<span class="derived" title="read out of the run's LABEL, not off the engine — ${esc(r.config_text || 'runs saved before runmeta.py shipped do not state their own concurrency cap')}">${r.max_num_seqs}*</span>`;
}

/** spec / batched as the stamp gives them — marked, like the cap, when the stamp read them off a
 *  label. A run with no stamp does not know them, and says so. */
function settingCell(r, v, what) {
  if (num(v) == null) return '<span class="dim" title="no config stamp records it">\u2014</span>';
  const t = what === 'batched' ? fmtInt(v) : String(v);
  if (r.config_source === 'verified') return `<span title="read back off the engine's own argv at launch">${esc(t)}</span>`;
  return `<span class="derived" title="${esc(r.config_text || 'not verified off the engine')}">${esc(t)}*</span>`;
}

/** The workload column: the stamped workload's name, or — with no stamp — the harness kind, marked. */
function workloadCell(r) {
  if (r.workload) return `<span class="kindtag" title="${esc(r.workload_desc || r.workload)}">${esc(workloadName(r.workload))}</span>`;
  const k = r.kind === 'corpus' ? 'corpus' : r.kind === 'real' ? 'real prompts' : r.kind || 'unknown';
  return `<span class="kindtag unstamped" title="no config stamp: the workload is not recorded in the file">${esc(k)}?</span>`;
}

function renderTable(runs) {
  $('runs').hidden = false;
  stampHeadings();
  const body = $('runs-body');
  body.innerHTML = runs.map((r) => {
    if (!r.ok) {
      return `<tr class="broken" data-id="${esc(r.id)}" tabindex="0">
        <td class="name">${esc(r.label)}</td>
        <td colspan="14" class="why">unreadable — ${esc(r.error)}</td>
        <td>${esc(fmtAgo(r.mtime))}</td></tr>`;
    }
    const c = r.curve || {};
    return `<tr data-id="${esc(r.id)}" tabindex="0" aria-selected="${state.selected === r.id}">
      <td class="name">${esc(r.label)}${r.cluster_state && r.cluster_state !== 'serving'
        ? ` <span class="derived" title="the cluster was '${esc(r.cluster_state)}' when this run started — not comparable with a run measured while serving">!</span>` : ''}</td>
      <td>${workloadCell(r)}</td>
      <td>${seqCell(r)}</td>
      <td class="num">${settingCell(r, r.spec, 'spec')}</td>
      <td class="num">${settingCell(r, r.batched, 'batched')}</td>
      <td class="num">${esc(fmtInt(r.concurrency))}</td>
      <td class="num">${r.prompts_file ? `<span class="dim" title="real prompts from ${esc(r.prompts_file)} — load100.py never used its --prompt-tokens default here">real</span>` : esc(fmtInt(r.prompt_tokens))}</td>
      <td class="num">${esc(fmtInt(r.max_tokens))}</td>
      ${numTd(r, 'agg_tok_s', esc(fmt(r.agg_tok_s)))}
      ${numTd(r, 'wall_s', esc(fmt(r.wall_s)))}
      ${numTd(r, 'curve_p10', esc(fmt(c.p10)))}
      ${numTd(r, 'curve_mean', esc(fmt(c.mean)))}
      ${numTd(r, 'curve_p90', esc(fmt(c.p90)))}
      ${numTd(r, 'ttft_mean', esc(fmt(r.ttft && r.ttft.mean)))}
      <td class="num">${num(r.errors) ? `<b style="color:var(--bad)">${esc(fmtInt(r.errors))}</b>` : '0'}</td>
      <td class="dim">${esc(fmtAgo(r.mtime))}</td></tr>`;
  }).join('') || `<tr><td colspan="16" class="dim" style="padding:1rem">no runs match these filters</td></tr>`;
  for (const tr of body.querySelectorAll('tr[data-id]')) {
    const open = () => select(tr.dataset.id);
    tr.addEventListener('click', open);
    tr.addEventListener('keydown', (ev) => { if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); open(); } });
  }
}

// ── one run, opened ─────────────────────────────────────────────────────────

const kpi = (label, value, unit) =>
  `<div class="kpi"><b>${esc(value)}${unit ? `<span class="t-unit"> ${esc(unit)}</span>` : ''}</b><span>${esc(label)}</span></div>`;
const pctKpi = (v) => (v == null ? '—' : `${(v * 100).toFixed(1)}`);

/** The settings this run ran under, and — every time — where each number came from. */
function settingsNote(r) {
  if (r.max_num_seqs == null) return { text: 'concurrency cap unknown', derived: true };
  const bits = [`max_num_seqs ${r.max_num_seqs}`];
  if (num(r.spec) != null) bits.push(`spec ${r.spec}`);
  if (num(r.batched) != null) bits.push(`batched ${fmtInt(r.batched)}`);
  if (r.generation != null) bits.push(`generation ${r.generation}`);
  const from = r.max_num_seqs_source === 'verified' ? 'read back off the engine argv at launch'
    : r.max_num_seqs_source === 'recorded' ? 'recorded by the harness'
      : `read out of the run's label, not off the engine${r.config_text ? ` \u2014 its stamp says \u201c${r.config_text}\u201d` : ''}`;
  return { text: `${bits.join(' · ')} — ${from}`, derived: derivedSettings(r) };
}

function renderDetail() {
  const box = $('detail');
  const r = state.detail;
  if (!state.selected) { box.hidden = true; box.innerHTML = ''; return; }
  box.hidden = false;
  if (!r) { box.innerHTML = `<p class="why">loading ${esc(state.selected)}&hellip;</p>`; return; }
  if (!r.ok) {
    box.innerHTML = `<div class="head"><h2>${esc(r.label)}</h2></div>
      <div class="alert"><b>this run cannot be read</b> <span class="why">${esc(r.error)}</span></div>`;
    return;
  }
  const c = r.curve || {}, tt = r.ttft || {}, pr = r.per_request || {};
  const set = settingsNote(r);
  const head = `
    <div class="head">
      <h2>${esc(r.label)}</h2>
      <span class="sub">${esc(r.workload ? workloadName(r.workload) : r.kind === 'corpus' ? 'real-corpus' : r.kind === 'real' ? 'real prompts, no config stamp' : r.kind === 'prefill' ? 'prefill probe' : 'synthetic')}
        &middot; ${esc(r.harness || '')}
        &middot; ${esc(r.model || 'model not recorded')}${r.model_choice ? ` &middot; ${esc(r.model_choice)}` : ''}</span>
      <span class="sub ${set.derived ? 'derived' : ''}">${esc(set.text)}</span>
      ${r.cluster_state ? `<span class="sub">cluster was <b>${esc(r.cluster_state)}</b></span>` : ''}
      <button type="button" class="close" id="d-close">close</button>
    </div>`;
  if (r.kind === 'prefill') {
    // A prefill probe reads the engine's counters at the two edges of the run and nothing between:
    // its own numbers, where it sits among the other probes, and no empty time-series cards.
    box.innerHTML = `${head}
      <div class="kpis">
        ${kpi('prefill', fmt(r.prefill_tok_s), 'tok/s')}
        ${kpi('compute per GPU', fmt(r.tflops_per_gpu), 'TFLOP/s')}
        ${kpi('of the nvfp4 peak', fmt(r.pct_nvfp4_peak), '%')}
        ${kpi('memory traffic per GPU', fmt(r.gbps_per_gpu), 'GB/s')}
        ${kpi('total wall', fmt(r.wall_s), 's')}
        ${kpi('requests ok', `${fmtInt(r.requests_ok)}/${fmtInt(r.requests)}`, '')}
        ${kpi('errors', fmtInt(r.errors), '')}
        ${kpi('preemptions', fmtInt(r.preemptions), '')}
        ${kpi('prompt tokens', fmtInt(r.prompt_tokens_total), '')}
        ${kpi('prefix-cache hits', fmtInt(r.prefix_hit_tokens), 'tok')}
        ${kpi('computed', fmtInt(r.computed_tokens), 'tok')}
      </div>
      ${r.error_sample ? `<div class="alert"><b>first error</b> <span class="why">${esc(r.error_sample)}</span></div>` : ''}
      ${whereHTML(r)}
      <p class="why detail-note">A prefill probe sends real ~32k-token prompts with max_tokens 1 and reads
        the engine's counters at its start and its end only, so it has no throughput curve and no
        per-request records to draw.</p>`;
    $('d-close').addEventListener('click', () => select(null));
    drawWhere(r);
    return;
  }
  box.innerHTML = `${head}
    <div class="kpis">
      ${kpi('aggregate', fmt(r.agg_tok_s), 'tok/s')}
      ${kpi('total wall', fmt(r.wall_s), 's')}
      ${kpi('curve mean', fmt(c.mean), 'tok/s')}
      ${kpi('bottom 10%', fmt(c.p10), 'tok/s')}
      ${kpi('top 10%', fmt(c.p90), 'tok/s')}
      ${kpi('per request', fmt(pr.mean), 'tok/s')}
      ${kpi('ttft mean', fmt(tt.mean), 's')}
      ${kpi('requests ok', `${fmtInt(r.requests_ok)}/${fmtInt(r.requests)}`, '')}
      ${kpi('errors', fmtInt(r.errors), '')}
      ${kpi('preemptions', fmtInt(r.preemptions), '')}
      ${kpi('spec accept', pctKpi(r.spec_accept), '%')}
      ${kpi('prefix hits', pctKpi(r.prefix_hit), '%')}
      ${r.prompt_tok_s != null ? kpi('engine prompt', fmt(r.prompt_tok_s), 'tok/s') : ''}
      ${kpi('kv peak', pctKpi(r.kv_max), '%')}
      ${kpi('ms / engine iter', fmt(r.ms_per_iter), '')}
    </div>
    ${r.phases ? `<div class="kpis">
      ${kpi('phase 1 fill', fmt(r.phases.fill_s), 's')}
      ${kpi('phase 2 serve', fmt(r.phases.serve_s), 's')}
      ${kpi('serve throughput', fmt(r.phases.serve_tok_s), 'tok/s')}
      ${kpi('cold ttft mean', fmt(r.phases.cold_ttft && r.phases.cold_ttft.mean), 's')}
      ${kpi('warm ttft mean', fmt(r.phases.warm_ttft && r.phases.warm_ttft.mean), 's')}
    </div>` : ''}
    ${r.error_sample ? `<div class="alert"><b>first error</b> <span class="why">${esc(r.error_sample)}</span></div>` : ''}
    ${whereHTML(r)}
    <div class="charts"><figure class="chart" id="d-gen">
      <h3>Generated throughput over time</h3>
      <p class="why">One sample every ${esc(fmt(r.kind === 'corpus' ? 3 : 2, 0))}&nbsp;s from the engine's own
        counters. The shaded band is the bottom 10% &rarr; top 10% of these samples and the dashed line their
        mean &mdash; the gap between them IS the "up and down". A break in the line is a sample that could
        not be taken, never a zero.</p>
    </figure></div>
    <div class="charts two">
      <figure class="chart" id="d-run"><h3>Requests running</h3>
        <p class="why">What the engine actually had in flight &mdash; the cap it really honoured.</p></figure>
      <figure class="chart" id="d-wait"><h3>Requests waiting</h3>
        <p class="why">The queue behind the cap. Offered concurrency above the cap is deliberate: it keeps
          the server saturated so the cap is what differs.</p></figure>
      <figure class="chart" id="d-kv"><h3>KV cache used</h3>
        <p class="why">Percent of the pool. A run that never fills it was not limited by it.</p></figure>
      <figure class="chart" id="d-iter"><h3>Milliseconds per engine iteration</h3>
        <p class="why">How long one decode step took. Lower is faster per token, whatever the aggregate says.</p></figure>
    </div>
    <div class="charts"><figure class="chart" id="d-ttft">
      <h3>Time to first token, per request</h3>
      <p class="why">One dot per request, placed at the moment it was issued.${r.kind === 'corpus'
        ? ' The lighter dots are the cold prefix fills of phase 1.' : ''}</p>
    </figure></div>`;
  $('d-close').addEventListener('click', () => select(null));
  drawWhere(r);
  const series = r.series || [];
  const col = capColor(r.max_num_seqs);
  timeChart($('d-gen'), series, 'gen_tok_s', { yTitle: 'generated tok/s', color: col,
    band: c, dashed: r.kind === 'corpus', aspect: 0.32, title: 'generated throughput over time' });
  timeChart($('d-run'), series, 'running', { yTitle: 'requests', color: col, aspect: 0.5, title: 'requests running' });
  timeChart($('d-wait'), series, 'waiting', { yTitle: 'requests', color: col, aspect: 0.5, title: 'requests waiting' });
  timeChart($('d-kv'), series, 'kv', { yTitle: 'kv used', color: col, aspect: 0.5, yHi: 1,
    yFmt: (v) => `${Math.round(v * 100)}%`, title: 'kv cache used' });
  timeChart($('d-iter'), series, 'ms_per_iter', { yTitle: 'ms / iter', color: col, aspect: 0.5,
    title: 'milliseconds per engine iteration' });
  ttftChart($('d-ttft'), r.requests_detail || []);
}

async function select(id) {
  state.selected = id;
  state.detail = null;
  renderTable(filtered());
  renderCompare(filtered());                        // the open run is ringed in its workload's panel
  renderDetail();
  if (!id) { history.replaceState(null, '', location.pathname); return; }
  history.replaceState(null, '', `${location.pathname}?run=${encodeURIComponent(id)}`);
  try {
    const res = await fetch(`/lb/bench/run?id=${encodeURIComponent(id)}`, { cache: 'no-store' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const body = await res.json();
    if (state.selected !== id) return;              // the operator moved on while this was in flight
    state.detail = body.run;
  } catch (err) {
    if (state.selected !== id) return;
    state.detail = { ok: false, label: id, error: String(err && err.message || err) };
  }
  renderDetail();
  $('detail').scrollIntoView({ block: 'nearest', behavior: 'smooth' });
}

// ══ REAL TRAFFIC: the performance log (/lb/perf, perflog.py) ═══════════════════════════════════════

/** A configuration with fewer recorded minutes than this is SHOWN (its row, its outline bar) but never
 *  compared: it sets no tier of the value wash and its bar is an outline labelled with its count. Half
 *  an hour of busy minutes is the least that averages over more than one burst of traffic. */
const MIN_WINDOWS = 30;
/** Two recorded minutes further apart than this are not joined by a line: the cluster was idle (an idle
 *  minute is not recorded) or the gateway was not reading it. */
const JOIN_GAP_S = 90;
// EVERYTHING RECORDED IS WHAT LOADS. The range picker opens on `all` — its own fourth option, labelled
// "everything recorded" — not on a window that silently drops the older minutes. A reader who wants a
// narrower window asks for one; nobody has to ask to be shown what the gateway saved.
const perf = { days: 'all', data: null, error: '', timer: null, loadCols: false,
  /** cap -> the variant keys (spec|batched) seen at that cap BY THE MODEL ON SHOW, in fixed order: the rank cfgStyle hatches by. */
  variants: new Map(),
  /** ONE MODEL AT A TIME (resolveModel): the viewer's pick (an id, null = none made), every model the
   *  selector offers — the ones this gateway RECORDED (/lb/perf `models`) — and the one everything below draws. */
  model: null, models: [], current: null,
  /** model id -> the CFG_PARTS keys that differ among its recorded configurations (cfgText names those). */
  varying: new Map() };

/** The window getter where a window names a measure differently from a saved run. Everything else —
 *  name, unit, and above all the DIRECTION — is MEASURES' one declaration. */
const WIN_GET = {
  agg_tok_s: (w) => w.gen_tok_s,
  prompt_tok_s: (w) => w.prompt_tok_s,
  prefix_hit: (w) => w.prefix_hit_rate,
  spec_accept: (w) => (w.spec || {}).acceptance,
  preemptions: (w) => w.preemptions,
};
const winSpec = (key, over) => ({ ...MEASURES[key], ...(WIN_GET[key] ? { get: WIN_GET[key] } : {}), ...(over || {}) });

/** What the comparison compares, per minute and pooled: throughput first (it follows demand, and the
 *  caveat above says so), then the per-request measures that say more about a configuration. */
const PERF_COMPARE = [
  ['agg_tok_s', { name: 'generated tok/s' }], ['ttft_mean', { name: 'mean time to first token' }],
  ['ttft_p90', {}], ['itl_mean', { name: 'mean inter-token latency, per engine gap' }], ['waiting_mean', { name: 'mean requests waiting' }],
  ['prefix_hit', {}], ['spec_accept', {}],
];
/** The quality table's columns, each a MEASURES key (so each carries its declared ▲/▼ and is washed). */
const PERF_QUALITY = [
  ['agg_tok_s', 'generated tok/s'], ['prompt_tok_s', 'billed prompt tok/s'],
  ['computed_tok_s', 'computed prompt tok/s'], ['prefix_hit', 'prefix hits'], ['ttft_mean', 'ttft mean'],
  ['ttft_p90', 'ttft p90'], ['ttft_p99', 'ttft p99'], ['itl_mean', 'itl mean'], ['itl_p90', 'itl p90'],
  ['itl_p99', 'itl p99'], ['waiting_mean', 'waiting'],
  ['spec_accept', 'spec accept'], ['preemptions', 'preempted'],
];

const cfgOf = (w) => (w && typeof w.config === 'object' && w.config) || {};

// ── WHAT A CONFIGURATION IS ─────────────────────────────────────────────────────────────────────────
// Every launch setting a minute is tagged with that changes what the engine does: the model, the cap
// (max_num_seqs — the colour), then CFG_PARTS in this order. Operator 2026-10-12, a one-host node's page:
// 190 minutes under a 7.92 GiB KV pin and 77 without one were ONE "16 seqs · spec 0 · batched ?" row, because
// the identity stopped at the context. The GPU memory fraction, the KV pin and the KV tier in the host's RAM
// are settings too (perflog CONFIG_KEYS / NODE_CONFIG_KEYS), and two minutes that differ in one are two
// configurations. The generation is not a setting: a restart at the same settings pools with the last.
//
// A part's value is read THREE ways, never folded into one another: ABSENT — a minute recorded before the
// key existed — is "not recorded"; null is the launch leaving it to the ENGINE (said by name, from the
// model's recorded engine: "batched vLLM default"); a value is itself. Never a bare "?".
const NOT_RECORDED = 'not recorded';
const GIB = 2 ** 30;
const hasKey = (c, k) => Object.prototype.hasOwnProperty.call(c, k);
/** `always`: on every label (the two the page always named); every other part only where it DIFFERS among
 *  the model's configurations (varyingParts) — enough to tell them apart, no column of identical words. */
const CFG_PARTS = [
  { key: 'num_speculative_tokens', always: true, name: 'spec', val: (v) => `spec ${v}`, unset: (e) => `spec ${e} default` },
  { key: 'max_num_batched_tokens', always: true, name: 'batched', val: (v) => `batched ${fmtInt(v)}`, unset: (e) => `batched ${e} default` },
  { key: 'max_model_len', name: 'context', val: (v) => `${fmtInt(v)} ctx`, unset: (e) => `context ${e} default` },
  { key: 'gpu_memory_utilization', name: 'GPU mem', val: (v) => `GPU mem ${v}`, unset: (e) => `GPU mem ${e} default` },
  { key: 'kv_cache_memory', name: 'KV', val: (v) => `KV ${(v / GIB).toFixed(2)} GiB`, unset: () => 'KV not pinned' },
  { key: 'kv_offload_ram_gib', name: 'RAM tier', val: (v) => `RAM ${v} GiB`, unset: () => 'no RAM tier' },
];
const PART = Object.fromEntries(CFG_PARTS.map((p) => [p.key, p]));
/** The token a setting puts in a configuration's key: absent, null and a value kept three things. */
const partToken = (c, k) => (!hasKey(c, k) ? NOT_RECORDED : c[k] == null ? 'default' : String(c[k]));
/** The engine's name for a recorded model (/lb/perf `models` engine_name, the catalog's), or "engine". */
function engineWord(id) {
  const m = perf.models.find((x) => x.id === id);
  return (m && m.engine) || 'engine';
}
/** A setting as a label says it. */
function partText(p, c) {
  if (!hasKey(c, p.key)) return `${p.name} ${NOT_RECORDED}`;
  const v = c[p.key];
  if (v == null) return p.unset(engineWord(c.model_choice));
  return p.val(num(v) ?? v);
}
/** ...without its name, where a heading already says what it is (the matrix's batched column). */
function partBare(p, c) {
  const t = partText(p, c);
  return t.startsWith(`${p.name} `) ? t.slice(p.name.length + 1) : t;
}
/** The SETTINGS a minute ran under, as one key: the model first (no group can hold two models), the cap,
 *  then every part. */
function settingsKey(w) {
  const c = cfgOf(w);
  return [c.model_choice == null ? '?' : c.model_choice, partToken(c, 'max_num_seqs'),
    ...CFG_PARTS.map((p) => partToken(c, p.key))].join(' | ');
}
/** The parts that DIFFER among these configurations — what their labels must name to be told apart. */
function varyingParts(cfgs) {
  return new Set(CFG_PARTS.filter((p) => new Set(cfgs.map((c) => partToken(c, p.key))).size > 1).map((p) => p.key));
}
/** Two '|'-joined token lists in the page's one fixed order, part by part: "not recorded", then the
 *  engine's default, then values ascending — so a configuration is solid or hatched the same way whatever
 *  the range shows. */
function cmpTokens(a, b) {
  const A = a.split('|'), B = b.split('|');
  const rank = (t) => (t === NOT_RECORDED ? 0 : t === 'default' ? 1 : 2);
  for (let i = 0; i < Math.max(A.length, B.length); i++) {
    const d = rank(A[i]) - rank(B[i]) || (rank(A[i]) === 2 ? (Number(A[i]) - Number(B[i])) || String(A[i]).localeCompare(String(B[i])) : 0);
    if (d) return d;
  }
  return 0;
}
const fallbackCfg = (w) => w.config_source !== 'effective';

// ── ONE MODEL AT A TIME ─────────────────────────────────────────────────────────────────────────────
// Two models are two machines: a minute of one is never pooled, banded, ranked, hatched or listed with a
// minute of the other. The page therefore shows ONE model — the selector's (#p-models) — and the split is
// made ONCE, at the door: drawPerf and recomputeDomains hand everything below them `ofModel(...)` records
// only, so no chart, table, legend or scale downstream can mix two models whatever key it pools by.
const MODEL_STORE = 'vllm.benchmarks.model';
const modelOf = (w) => { const m = cfgOf(w).model_choice; return typeof m === 'string' ? m : ''; };
/** A recorded model's name — the catalog's, as /lb/perf `models` hands it over — or its raw id. */
function modelLabel(id) {
  const m = perf.models.find((x) => x.id === id);
  return m ? m.label : (id || 'model not recorded');
}
function storedModel() { try { return window.localStorage.getItem(MODEL_STORE); } catch (e) { return null; } }
function storeModel(id) { try { window.localStorage.setItem(MODEL_STORE, id); } catch (e) { /* a private window: the pick lasts the visit */ } }
const timed = (xs) => (xs || []).filter((w) => w && num(w.t_start) != null && num(w.t_end) != null);
/** The models the selector offers: EXACTLY the ones this gateway recorded a minute of in the range loaded
 *  (/lb/perf `models`, perflog.recorded_models), in the order each was first recorded — never the catalog's
 *  (operator 2026-10-12, a one-model deployment offering four catalog models at "0 MIN": "should not be
 *  hardcoded, added on the way when new model loaded"). So a model appears the first time a minute of it is written, none reads
 *  0 min, one the catalog does not know is offered under its raw id, and a log with no minute offers none.
 *  `minutes` is the server's count, the ones older than the newest MAX_WINDOWS included — a model whose every
 *  minute was cut is still one this log recorded. The one on show: the viewer's pick when it is offered, else
 *  the model with the MOST RECENT minute. */
function resolveModel() {
  const d = perf.data;
  perf.models = (d && Array.isArray(d.models) ? d.models : [])
    .filter((m) => m && typeof m.id === 'string' && num(m.minutes) > 0)
    .map((m) => ({ id: m.id, label: (typeof m.label === 'string' && m.label) || m.id || 'model not recorded',
      engine: (typeof m.engine_name === 'string' && m.engine_name) || null, minutes: num(m.minutes), last: num(m.last) }));
  if (perf.model == null) perf.model = storedModel();
  const newest = [...perf.models].sort((a, b) => (b.last ?? -Infinity) - (a.last ?? -Infinity))[0];
  perf.current = perf.models.find((m) => m.id === perf.model) || newest || null;
  return perf.current;
}
/** The minutes the server cut from the range (perflog.MAX_WINDOWS), per model — or null from a gateway
 *  that does not say whose they were yet. */
function cutByModel() {
  const by = perf.data ? perf.data.truncated_by_model : null;
  return by && typeof by === 'object' ? by : null;
}
/** ...and the model on show's count of them. Only when the gateway cannot say whose they were is it the
 *  whole log's count: a number too large is better than a cut that goes unsaid. */
function cutMinutes() {
  const by = cutByModel();
  if (by) return perf.current ? num(by[perf.current.id]) || 0 : 0;
  return perf.data ? num(perf.data.truncated) || 0 : 0;
}
/** Only the records (minutes, boots) of the model on show. */
const ofModel = (xs) => (perf.current ? xs.filter((w) => modelOf(w) === perf.current.id) : []);
/** THE MATRIX'S RECORDS, the inventory: every minute in the range, and every boot OF A MODEL THIS GATEWAY
 *  RECORDED A MINUTE OF — a model that only booted has no grid, as it has no segment (resolveModel). A boot of
 *  a recorded model at settings that never served a minute keeps its square: that configuration is the model's. */
function inventory(d) {
  const recorded = new Set(perf.models.map((m) => m.id));
  const byStart = (a, b) => a.t_start - b.t_start;
  return { ws: d ? timed(d.windows).sort(byStart) : [],
    boots: d ? timed(d.boots).filter((b) => recorded.has(modelOf(b))).sort(byStart) : [] };
}

/** The selector: one segment per model, its label over its recorded minutes; and the model's label in the
 *  page header. Native radios, like the range picker. Rewritten only when what it says changed, and the
 *  focus is put back, so the minute tick does not take the keyboard away. */
let modelsHTML = null;
function renderModels() {
  const host = $('p-models'), cur = perf.current;
  host.hidden = !perf.data || !perf.models.length;
  $('p-model-name').textContent = cur && perf.data ? cur.label : '';
  const html = perf.models.map((m) => `<label class="mdl-opt"><input type="radio" name="p-model" value="${esc(m.id)}" autocomplete="off"${cur && m.id === cur.id ? ' checked' : ''}>`
    + `<span class="mdl-face"><span class="mdl-name">${esc(m.label)}</span><span class="mdl-n"><b>${esc(fmtInt(m.minutes))}</b> min</span></span></label>`).join('');
  if (html === modelsHTML) return;
  const focused = host.contains(document.activeElement);
  host.innerHTML = modelsHTML = html;
  if (focused) { const c = host.querySelector('input:checked'); if (c) c.focus(); }
}
const RANGE_TEXT = { 1: '24 h', 7: '7 d', 30: '30 d', all: 'all' };
/** A matrix group's configuration: its first minute's, or — a configuration that only ever BOOTED — its first boot's. */
const gcfg = (g) => cfgOf(g.ws[0] || (g.boots || [])[0]);

/** Seconds since the epoch in the reader's own clock. */
function when(t, withDate) {
  const d = new Date(t * 1000);
  const hm = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: false });
  return withDate ? `${d.toLocaleDateString([], { month: 'short', day: 'numeric' })} ${hm}` : hm;
}

/** The windows' own bucket counts ([le, n], +Inf as null) summed per bound: the engine's histogram over
 *  a set of minutes, exact. */
function pooledCounts(ws, key) {
  const counts = new Map();
  for (const w of ws) {
    for (const b of ((w[key] || {}).buckets || [])) {
      if (!Array.isArray(b) || num(b[1]) == null) continue;
      const le = b[0] == null ? Infinity : num(b[0]);
      if (le == null) continue;
      counts.set(le, (counts.get(le) || 0) + b[1]);
    }
  }
  return counts;
}
/** The pooled histogram as the rows a reader reads (histRows): the distribution behind a mean and a p90. */
const pooledRows = (ws, key) => histRows(pooledCounts(ws, key));

/** A MEAN ABOVE A P90 looks like a bug and is not one: in half of tonight's minutes 90% of the inter-token
 *  gaps land in one bucket at ~150 ms and a thin tail of multi-second stalls drags the arithmetic mean past
 *  the 90th percentile (mean 203 ms, p90 150 ms, p99 2.4 s). Both numbers are right; printed side by side
 *  with no word they cost the page its credibility, so wherever both are printed this sentence goes with
 *  them. '' when there is nothing to explain. */
function tailNote(mean, p90, noun) {
  const m = num(mean), p = num(p90);
  if (m == null || p == null || m <= p) return '';
  return `the mean sits above the p90: most ${noun} are alike and a few far longer ones pull the average past the 90th percentile — the p99 says how far that tail reaches`;
}

/** Pooled quantile over windows' own bucket counts ([le, n], +Inf as null): the engine's histogram
 *  summed across the minutes, then Prometheus' interpolation — the exact p90 of every request in them. */
function pooledQuantile(ws, key, q) {
  const counts = pooledCounts(ws, key);
  const les = [...counts.keys()].sort((a, b) => a - b);
  let cum = 0;
  const pairs = les.map((le) => [le, (cum += counts.get(le))]);
  if (!pairs.length || cum <= 0) return null;
  const rank = q * cum;
  let pl = 0, pc = 0;
  for (const [le, c] of pairs) {
    if (c >= rank) {
      if (!Number.isFinite(le)) return pl;
      return c <= pc ? le : pl + (le - pl) * (rank - pc) / (c - pc);
    }
    pl = le; pc = c;
  }
  return pl;
}

/** Every figure of a set of minutes, POOLED — never an average of averages: rates are total increment
 *  over total span, latencies request-weighted, p90 from the summed histograms, gauges span-weighted. */
function aggregate(ws) {
  const span = (w) => Math.max(0, num(w.span_s) || 0);
  const total = (get) => { let s = 0, seen = false; for (const w of ws) { const v = num(get(w)); if (v != null) { s += v; seen = true; } } return seen ? s : null; };
  const rate = (get) => {
    let inc = 0, t = 0, seen = false;
    for (const w of ws) { const v = num(get(w)); if (v == null) continue; inc += v; t += span(w); seen = true; }
    return seen && t > 0 ? inc / t : null;
  };
  const ratio = (a, b) => {
    let x = 0, y = 0, seen = false;
    for (const w of ws) { const u = num(a(w)), v = num(b(w)); if (u == null || v == null) continue; x += u; y += v; seen = true; }
    return seen && y > 0 ? x / y : null;
  };
  const weighted = (get, weight = span) => {
    let s = 0, k = 0;
    for (const w of ws) { const v = num(get(w)), c = weight(w); if (v == null || !(c > 0)) continue; s += v * c; k += c; }
    return k > 0 ? s / k : null;
  };
  const most = (get) => { const vs = ws.map(get).map(num).filter((v) => v != null); return vs.length ? Math.max(...vs) : null; };
  const boxes = [...new Set(ws.flatMap((w) => Object.keys(w.boxes || {})))].sort();
  return {
    n: ws.length, minutes: (total(span) || 0) / 60,
    first: Math.min(...ws.map((w) => w.t_start)), last: Math.max(...ws.map((w) => w.t_end)),
    generations: [...new Set(ws.map((w) => cfgOf(w).generation).filter((g) => g != null))].sort((a, b) => a - b),
    fallback: ws.some(fallbackCfg),
    requests: total((w) => w.requests_finished),
    agg_tok_s: rate((w) => w.gen_tokens), prompt_tok_s: rate((w) => w.prompt_tokens),
    computed_tok_s: rate((w) => w.prompt_computed_tokens),
    prefix_hit: ratio((w) => w.prefix_hit_tokens, (w) => w.prefix_query_tokens),
    ttft_mean: weighted((w) => (w.ttft || {}).mean, (w) => (w.ttft || {}).n || 0),
    ttft_p90: pooledQuantile(ws, 'ttft', 0.9),
    itl_mean: ms(weighted((w) => (w.itl || {}).mean, (w) => (w.itl || {}).n || 0)),
    itl_p90: ms(pooledQuantile(ws, 'itl', 0.9)),
    // The tail: where the slowest 1% begin. A mean and a p90 between them cannot show it (tailNote).
    ttft_p99: pooledQuantile(ws, 'ttft', 0.99),
    itl_p99: ms(pooledQuantile(ws, 'itl', 0.99)),
    waiting_mean: weighted((w) => (w.waiting || {}).mean),
    spec_accept: ratio((w) => (w.spec || {}).accepted, (w) => (w.spec || {}).draft_tokens),
    preemptions: total((w) => w.preemptions),
    running_mean: weighted((w) => (w.running || {}).mean), running_max: most((w) => (w.running || {}).max),
    kv_mean: weighted((w) => (w.kv || {}).mean), kv_max: most((w) => (w.kv || {}).max),
    prompt_per_req: ratio((w) => w.prompt_tokens, (w) => w.requests_finished),
    out_per_req: ratio((w) => w.gen_tokens, (w) => w.requests_finished),
    tflops: weighted((w) => w.tflops_per_gpu), gbps: weighted((w) => w.gbps_per_gpu),
    boxes: Object.fromEntries(boxes.map((b) => [b, {
      sm: weighted((w) => ((w.boxes || {})[b] || {}).sm_mean),
      pwr: weighted((w) => ((w.boxes || {})[b] || {}).pwr_mean)}])),
  };
}

/** A window as the comparison kit reads a run: its settings where the kit looks for them. */
function winAsRun(w) {
  const c = cfgOf(w);
  return { w, id: `w${w.t_start}`, label: when(w.t_start, true), max_num_seqs: num(c.max_num_seqs),
    spec: num(c.num_speculative_tokens), batched: num(c.max_num_batched_tokens),
    generation: num(c.generation), group: `g${c.generation}`, rep: null,
    max_num_seqs_source: fallbackCfg(w) ? 'stated' : 'verified', config_source: fallbackCfg(w) ? 'stated' : 'verified' };
}

/** The configuration, as a table cell says it: cap with its ramp swatch, then the two launch values —
 *  each marked when the recorder had to ASSUME it (no `cluster.effective` published yet). */
function configCell(g) {
  const c = gcfg(g);
  const assumed = (v) => (g.agg.fallback
    ? `<span class="derived" title="assumed: this gateway did not publish cluster.effective yet, so the recorder took cluster.desired and the launch default for this value — not a value read off the running engine">${v}*</span>` : v);
  const st = cfgStyle(c);
  // The memory settings that tell this configuration from its model's others (cfgText's rule); the context
  // has its own place in the sub-line.
  const vary = perf.varying.get(c.model_choice) || new Set();
  const more = CFG_PARTS.filter((p) => !p.always && p.key !== 'max_model_len' && vary.has(p.key));
  return `<i class="seqdot v${st.rank}" style="--c:${st.color}"></i><b>${esc(c.max_num_seqs ?? NOT_RECORDED)}</b> seqs`
    + ` &middot; ${assumed(esc(partText(PART.num_speculative_tokens, c)))}`
    + ` &middot; ${assumed(esc(partText(PART.max_num_batched_tokens, c)))}`
    + more.map((p) => ` &middot; ${esc(partText(p, c))}`).join('')
    + `<span class="cfg-sub">${esc(modelLabel(c.model_choice))} &middot; ${esc(partText(PART.max_model_len, c))}`
    + `${g.agg.generations.length ? ` &middot; gen ${esc(g.agg.generations.join(', '))}` : ''}</span>`;
}

/** The value wash's scales: per (model, measure), over the configurations with MIN_WINDOWS+ minutes. */
function perfScales(groups) {
  const scales = new Map();
  for (const g of groups) {
    if (g.agg.n < MIN_WINDOWS) continue;
    for (const [key] of PERF_QUALITY) {
      const v = num(g.agg[key]);
      if (v == null) continue;
      const id = `${gcfg(g).model_choice}|${key}`;
      const s = scales.get(id) || { lo: v, hi: v, n: 0 };
      s.lo = Math.min(s.lo, v); s.hi = Math.max(s.hi, v); s.n += 1;
      scales.set(id, s);
    }
  }
  return scales;
}

/** The tier (1..TIERS, TIERS best) a pooled figure sits in, or null when it is not tiered. */
function perfTier(g, key, scales) {
  const v = num(g.agg[key]);
  if (v == null || g.agg.n < MIN_WINDOWS) return null;
  const m = scales.get(`${gcfg(g).model_choice}|${key}`);
  if (!m || m.n < 2 || m.hi === m.lo) return null;
  const good = MEASURES[key].dir > 0 ? (v - m.lo) / (m.hi - m.lo) : (m.hi - v) / (m.hi - m.lo);
  return Math.min(TIERS, 1 + Math.floor(good * TIERS));
}

/** WHY a configuration that HAS its MIN_WINDOWS minutes still carries no shade — perfTier said only
 *  "null". The matrix key used to carry this sentence once, for the whole matrix; the key is gone, so it
 *  is said per square, in the square's own words, and more exactly: per MODEL, which is the group a tier
 *  is actually computed over. Only ever asked of a square with MIN_WINDOWS+ minutes and no tier. */
function perfTierWhy(g, key, scales) {
  const m = scales.get(`${gcfg(g).model_choice}|${key}`);
  if (num(g.agg[key]) == null) return 'this measure was not recorded for it';
  if (!m || m.n < 2) return `it is the only configuration of this model with ${MIN_WINDOWS} recorded minutes, so there is nothing to compare it with yet`;
  return `the ${fmtInt(m.n)} configurations of this model with ${MIN_WINDOWS} recorded minutes read alike here, so the matrix has no tiers`;
}

/** The square's headline: generated tok/s — the measure the matrix is coloured by. */
const PM_KEY = 'agg_tok_s';
/** The bar's state: the groups it draws, the wash scales, the SELECTED configuration (an index into
 *  groups, or null = "all"), and whether its details are unfolded. */
const pm = { groups: [], scales: new Map(), pinned: null, open: false, returnTo: null };

/** One configuration read out in full: the configuration, then every pooled figure (washed like the
 *  table cells were), then the traffic, load and hardware it ran under. */
function pmDetailHTML(g) {
  const a = g.agg;
  const val = (key, v) => valText(MEASURES[key], v, true).replace(/ (tok\/s|s|ms|requests)$/, '');
  const plain = (v, digits) => (num(v) == null ? '—' : fmt(v, digits));
  const boxes = Object.keys(a.boxes).sort();
  const row = (label, text, heat) => `<tr><th scope="row">${label}</th><td class="num${heat ? ' heat' : ''}"${heat ? ` data-heat="${heat}"` : ''}>${text}</td></tr>`;
  // The two latencies' mean / p90 / p99 rows, and — right under the p99, where a reader who has just seen a
  // mean above a p90 is looking — the sentence that says why (tailNote), never left silent.
  const tails = { ttft_p99: tailNote(a.ttft_mean, a.ttft_p90, 'waits'), itl_p99: tailNote(a.itl_mean, a.itl_p90, 'gaps') };
  return `<div class="pm-dhead"><div class="pm-cfg">${configCell(g)}</div>
      <button type="button" class="pm-close" title="hide these figures (Escape does too)">hide details</button></div>
    <p class="pm-n">${a.n ? `${esc(fmtInt(a.n))} recorded minute${a.n === 1 ? '' : 's'} &middot; ${esc(fmtInt(a.requests))} requests
      &middot; ${esc(when(a.first, true))} &rarr; ${esc(when(a.last, true))}` : 'no traffic minute recorded'}${a.n < MIN_WINDOWS
        ? ` <span class="few-tag" title="fewer than ${MIN_WINDOWS} recorded minutes: shown, not compared">too few to compare</span>` : ''}</p>
    <div class="pm-tables">${a.n ? `<table class="runs-t pm-t"><caption>results</caption><tbody>
      ${PERF_QUALITY.map(([key, label]) => {
        const unit = MEASURES[key].unit;
        const u = unit && !MEASURES[key].frac && unit !== 'requests' && !label.includes(unit) ? ` <span class="th-unit">${esc(unit)}</span>` : '';
        return row(`${esc(label)}${dirHTML(key, false)}${u}`, esc(val(key, a[key])), perfTier(g, key, pm.scales))
          + (tails[key] ? `<tr class="tail-row"><td colspan="2">${esc(tails[key])}; the distribution itself is under &ldquo;distributions and cluster boot&rdquo; below.</td></tr>` : '');
      }).join('')}
    </tbody></table>
    <table class="runs-t pm-t"><caption>traffic, load and hardware</caption><tbody>
      ${row('prompt tok / req', esc(fmtInt(a.prompt_per_req)))}
      ${row('output tok / req', esc(fmtInt(a.out_per_req)))}
      ${row('running mean / max', `${esc(plain(a.running_mean, 1))} / ${esc(plain(a.running_max, 0))}`)}
      ${row('kv mean / max', a.kv_mean == null ? '—' : `${(a.kv_mean * 100).toFixed(0)}% / ${(a.kv_max * 100).toFixed(0)}%`)}
      ${boxes.map((b) => { const x = a.boxes[b] || {}; return row(`${esc(b)} sm fill`, x.sm == null ? '—' : `${x.sm.toFixed(0)}%`)
        + row(`${esc(b)} power`, x.pwr == null ? '—' : `${x.pwr.toFixed(0)} W`); }).join('')}
      ${row('tflop/s per GPU', esc(plain(a.tflops, 1)))}
      ${row('gb/s per GPU', esc(plain(a.gbps, 0)))}
    </tbody></table>` : ''}
    </div>
    ${a.n ? `<details class="pm-more" data-keep="pm-more"><summary>distributions and cluster boot</summary>
      <div class="pm-tables">${distBlock('time to first token', pooledRows(g.ws, 'ttft'), 'waits')}${distBlock('inter-token latency, per engine gap', pooledRows(g.ws, 'itl'), 'gaps')}${bootTableHTML(g.boots || [])}</div></details>`
      : `<div class="pm-tables">${bootTableHTML(g.boots || [])}</div>`}`;
}

/** Seconds as the eye reads a boot: "7 m 12 s", "48 s". */
function dur(s) {
  const v = num(s);
  if (v == null) return '—';
  if (v < 60) return `${Math.round(v)} s`;
  const m = Math.floor(v / 60), r = Math.round(v - m * 60);
  return r ? `${m} m ${r} s` : `${m} m`;
}
const median = (xs) => {
  const v = xs.map(num).filter((x) => x != null).sort((a, b) => a - b);
  return v.length ? (v.length % 2 ? v[(v.length - 1) / 2] : (v[v.length / 2 - 1] + v[v.length / 2]) / 2) : null;
};
/** The loading phases the LB reads off the head's log (cluster_startup.StartupPhase), grouped into the
 *  steps a boot is made of, in the order a start goes through them. */
const BOOT_STEPS = [
  // BOTH ENGINES' opening word: vLLM says "starting vLLM", SGLang "starting SGLang". With only the
  // first, every SGLang boot's opening phase fell through every rule to `other`. The alternation is
  // anchored and word-bounded so "starting the API server" still belongs to the api server step.
  ['init', /^starting (?:vllm|sglang)\b|initializing|connecting the ranks/i],
  ['weights', /^loading |loaded$/i],
  ['compile', /compil/i],
  ['warmup', /profil|warm|autotun|graphs|kv cache/i],
  ['api server', /api server/i],
];
const bootStep = (phase) => (BOOT_STEPS.find(([, re]) => re.test(phase)) || ['other'])[0];

/** THE BOOT BLOCK: every recorded START of this configuration (perflog.BootRecorder) — how many, how many
 *  failed, the median / fastest / slowest time from the Start to serving, then the median of each step:
 *  from the Start to the ranks launched (reservation, RAM freed), and each loading step the head's log
 *  named. A phase's duration runs to the next phase the LB read (every ~15 s), so the steps are coarse. */
function bootTableHTML(boots) {
  if (!boots.length) return '';
  const row = (label, text) => `<tr><th scope="row">${label}</th><td class="num">${text}</td></tr>`;
  const served = boots.filter((b) => b.outcome === 'serving' && num(b.total_s) != null);
  const failed = boots.filter((b) => b.outcome === 'failed');
  const stopped = boots.length - served.length - failed.length;
  const totals = served.map((b) => b.total_s);
  const launch = served.map((b) => num((b.states || {}).starting)).filter((v) => v != null);
  const steps = new Map();                          // step -> per-boot seconds
  for (const b of served) {
    const ph = (b.phases || []).filter((p) => p && typeof p.phase === 'string' && num(p.at_s) != null);
    const mine = new Map();
    ph.forEach((p, i) => {
      const end = i + 1 < ph.length ? ph[i + 1].at_s : b.total_s;
      const step = bootStep(p.phase);
      mine.set(step, (mine.get(step) || 0) + Math.max(0, end - p.at_s));
    });
    for (const [step, secs] of mine) steps.set(step, [...(steps.get(step) || []), secs]);
  }
  const last = [...failed].sort((a, b) => b.t_end - a.t_end)[0];
  const stepRows = BOOT_STEPS.map(([step]) => step).concat(['other']).filter((s) => steps.has(s))
    .map((s) => row(`${esc(s)} <span class="th-unit">median</span>`, esc(dur(median(steps.get(s))))));
  return `<table class="runs-t pm-t pm-boot-t"><caption>cluster boot</caption><tbody>
    ${row('starts', `${esc(fmtInt(boots.length))}${failed.length ? ` &middot; <b>${esc(fmtInt(failed.length))} failed</b>` : ''}${stopped ? ` &middot; ${esc(fmtInt(stopped))} stopped` : ''}`)}
    ${served.length ? row('to serving <span class="th-unit">median</span>', esc(dur(median(totals))))
      + row('fastest / slowest', `${esc(dur(Math.min(...totals)))} / ${esc(dur(Math.max(...totals)))}`)
      + (launch.length ? row('to ranks launched <span class="th-unit">median</span>', esc(dur(median(launch)))) : '')
      + stepRows.join('') : row('to serving', 'never reached')}
    ${last ? row('last failure', `<span class="pm-fail" title="${esc(last.failure || '')}">${esc(when(last.t_end, true))}${last.failed_phase ? ` in &ldquo;${esc(last.failed_phase)}&rdquo;` : last.failed_in ? ` while ${esc(last.failed_in)}` : ''} &middot; ${esc((last.failure || '').slice(0, 90))}${(last.failure || '').length > 90 ? '…' : ''}</span>`) : ''}
  </tbody></table>`;
}

/** One pooled latency histogram read out in the details: the rows of the distribution (distHTML) under a
 *  caption that says how many samples they are of. Nothing when the engine exposed no such histogram. */
function distBlock(name, rows, noun) {
  if (!rows.length) return '';
  const n = rows.reduce((s, r) => s + r.n, 0);
  return `<div class="pm-dist"><h5>${esc(name)}: how the ${esc(fmtInt(n))} ${esc(noun)} were spread</h5>${distHTML(rows)}</div>`;
}

/** The selected configuration's one line in the panel: its settings and its headline figure. Nothing is
 *  selected — nothing is printed: the line that counted the configurations and said "click a square" was
 *  the other half of the key the operator asked gone, and the charts' own heading says it anyway. An empty
 *  line also hides `all` and `details` (pmSelect), so the panel is then exactly the matrix. */
function pmLabelHTML(g) {
  if (!g) return '';
  const a = g.agg;
  const head = a.n ? `${esc(fmt(a[PM_KEY], 0))} tok/s &middot; ${esc(fmtInt(a.n))} min` : `${esc(fmtInt((g.boots || []).length))} boot${(g.boots || []).length === 1 ? '' : 's'}, no traffic`;
  return `${configCell(g)}<span class="pb-fig">${head}</span>`;
}

/** SELECT a configuration (null = all): the square is pressed, the bar names it, every chart brings it
 *  forward (pmScopeCharts), and the details card — only if it is already open — reads it instead. Selecting
 *  never opens the card: that is the `details` button's job. */
function pmSelect(i) {
  if (i != null && !pm.groups[i]) i = null;
  pm.pinned = i;
  const g = i == null ? null : pm.groups[i];
  for (const sq of document.querySelectorAll('#pm-grids button.pm-sq')) {
    sq.setAttribute('aria-pressed', String(Number(sq.dataset.i) === i));
  }
  $('pm-sel').innerHTML = pmLabelHTML(g);
  $('pm-all').hidden = g == null;
  $('pm-toggle').hidden = g == null;
  const card = $('pm-detail');
  keepingOpen(card, () => { card.innerHTML = g ? pmDetailHTML(g) : ''; });
  // The card's edge is the cap's ramp step — the same blue as the square's dot and the charts' marks.
  card.style.borderLeftColor = g ? capColor(gcfg(g).max_num_seqs) : '';
  const close = card.querySelector('.pm-close');
  if (close) close.addEventListener('click', () => { pmExpand(false); $('pm-toggle').focus(); });
  if (!g) pmExpand(false);
  pmScopeCharts();
  renderContext(perf.ctxWs || []);               // the context bands follow the selection, like the charts
}

/** Unfold or fold the selected configuration's figures: the card at the head of the charts, in the page's
 *  own flow — never an overlay, so it cannot cover the marks the selection has just brought forward. The
 *  bar's `details` button is the one way in (and Escape, or the card's own button, the way out). */
function pmExpand(open) {
  const was = pm.open;
  pm.open = !!open && pm.pinned != null;
  $('pm-detail').hidden = !pm.open;
  const t = $('pm-toggle');
  t.setAttribute('aria-expanded', String(pm.open));
  t.textContent = pm.open ? 'hide details' : 'details';
  // Closed: back to where the reader was when they opened it, not stranded at the head of the page.
  if (was && !pm.open && pm.returnTo != null) {
    window.scrollTo({ top: pm.returnTo, behavior: scrollHow() });
    pm.returnTo = null;
  }
}

/** "Switch between configurations": every mark of the "Every recorded minute" charts, of the batch-size
 *  curve and of the by-max_num_seqs panel carries the settings key of its minutes (data-k, set where it
 *  is drawn), so scoping is a class flip — no redraw. The other configurations RECEDE, never vanish: the
 *  charts exist to compare, and the selected one is read against the company it keeps. Called again after
 *  every rebuild (a refresh, a range change), so freshly drawn SVG takes the selection too. */
function pmScopeCharts() {
  const g = pm.pinned == null ? null : pm.groups[pm.pinned];
  const key = g && g.ws[0] ? settingsKey(g.ws[0]) : null;
  for (const host of [$('perf-time-grid'), $('perf-cmp')]) {
    host.classList.toggle('scoped', key != null);
    if (key == null) continue;
    for (const m of host.querySelectorAll('[data-k]')) m.classList.toggle('off', m.dataset.k !== key);
  }
  // Each panel's header says a scope is on — faded marks alone read as a dim chart, not as a choice.
  for (const t of document.querySelectorAll('.scope-tag')) {
    t.hidden = key == null;
    if (key != null) t.textContent = `showing ${cfgText(gcfg(g))} — the others faded`;
  }
}

/** The settings a group IS — read off whichever record it has, a minute or (a configuration that only ever
 *  booted) a boot. The one handle on a group that survives a redraw, where its index does not. */
const groupKey = (g) => settingsKey(g.ws[0] || (g.boots || [])[0]);

/** One group per configuration: its minutes AND its boots. A configuration that only ever booted (its
 *  traffic minutes not recorded, or none yet) is a group with no minutes, so its boot figures are readable.
 *
 *  THE GROUP KEY IS `settingsKey`, WHOSE FIRST FIELD IS THE MODEL — so handing this every model's records
 *  cannot pool two models into one group, however wide the input. Model first in the sort, so each model's
 *  configurations come out together and in the matrix's own row/column order. */
function groupsOf(ws, boots) {
  const byKey = new Map();
  const at = (r) => { const k = settingsKey(r); const g = byKey.get(k) || { ws: [], boots: [] }; byKey.set(k, g); return g; };
  for (const w of ws || []) at(w).ws.push(w);
  for (const b of boots || []) at(b).boots.push(b);
  return [...byKey.values()]
    .map((g) => ({ ws: g.ws, boots: g.boots, agg: aggregate(g.ws) }))
    .sort((a, b) => String(gcfg(a).model_choice).localeCompare(String(gcfg(b).model_choice))
      || (num(gcfg(a).max_num_seqs) || 0) - (num(gcfg(b).max_num_seqs) || 0)
      || (num(gcfg(a).num_speculative_tokens) || 0) - (num(gcfg(b).num_speculative_tokens) || 0)
      || (num(gcfg(a).max_num_batched_tokens) || 0) - (num(gcfg(b).max_num_batched_tokens) || 0));
}

/** THE MATRIX: per model, rows = max_num_seqs, columns = prefill budget · speculative tokens (and the
 *  context, only when a model ran at more than one). A cell no configuration ran is an empty outline. */
function renderPerfMatrix(groups) {
  pm.groups = groups;
  pm.scales = perfScales(groups);
  const models = [...new Set(groups.map((g) => gcfg(g).model_choice || 'model not recorded'))];
  const html = models.map((model) => {
    const mine = groups.map((g, i) => [g, i]).filter(([g]) => (gcfg(g).model_choice || 'model not recorded') === model);
    // A column is every setting but the cap: the prefill budget and the speculative tokens, then each other
    // part that DIFFERS among this model's configurations (the context, the GPU memory fraction, the KV pin,
    // the RAM tier) — so no two configurations ever share a cell, and a part they all share is no column.
    const vary = varyingParts(mine.map(([g]) => gcfg(g)));
    const extra = CFG_PARTS.filter((p) => !p.always && vary.has(p.key));
    const colParts = [PART.max_num_batched_tokens, PART.num_speculative_tokens, ...extra];
    const colKey = (c) => colParts.map((p) => partToken(c, p.key)).join('|');
    const cols = [...new Map(mine.map(([g]) => [colKey(gcfg(g)), gcfg(g)])).entries()].sort((x, y) => cmpTokens(x[0], y[0]));
    const rows = [...new Set(mine.map(([g]) => gcfg(g).max_num_seqs))].sort((x, y) => (num(x) || 0) - (num(y) || 0));
    const cell = new Map(mine.map(([g, i]) => [`${gcfg(g).max_num_seqs}#${colKey(gcfg(g))}`, [g, i]]));
    // The budget on one line, "spec N" on the next, then one line per differing part — "8,192 · 4" said
    // neither, and the axis row above names what the first two ARE.
    const colHead = ([, c]) => `<th scope="col"><span>${esc(partBare(PART.max_num_batched_tokens, c))}</span>`
      + `<span class="pm-sub">${esc(partText(PART.num_speculative_tokens, c))}</span>`
      + extra.map((p) => `<span class="pm-sub">${esc(partText(p, c))}</span>`).join('') + '</th>';
    // Whose grid this is, in the catalog's words, and which one the charts below are actually comparing —
    // so a reader of a page that now lists every model is never in doubt about what the charts are of.
    const onShow = !!perf.current && perf.current.id === model;
    const whose = onShow ? 'compared below' : 'recorded — click a square to compare it';
    return `<figure class="pm-grid${models.length > 1 && onShow ? ' on-show' : ''}">${models.length > 1
      ? `<figcaption title="${esc(`${modelLabel(model)} — ${whose}`)}"><span class="pm-who">${esc(modelLabel(model))}</span>`
        + `<span class="pm-whose">${esc(whose)}</span></figcaption>` : ''}
      <table class="pm-m"><thead>
        <tr><td class="pm-corner"></td><th scope="colgroup" colspan="${cols.length}" class="pm-axis">batched tokens &middot; speculative tokens</th></tr>
        <tr><th scope="col" class="pm-corner">seqs</th>${cols.map(colHead).join('')}</tr></thead>
      <tbody>${rows.map((r) => `<tr><th scope="row"><i class="seqdot" style="background:${capColor(r)}"></i>${esc(r ?? '?')}</th>
        ${cols.map(([k, c]) => {
          // The panel has NO key any more, so each cell's title and aria-label carry every convention the
          // key used to draw: which settings it is, its figure, what its shade means (its ramp step, or why
          // it has none), the under-MIN_WINDOWS rule that outlines it, the boot-only ↻, and "never ran".
          // A hover and a screen reader both get the whole sentence; the panel spends no height on it.
          const where = [`${r ?? NOT_RECORDED} seqs`, ...colParts.map((p) => partText(p, c))].join(', ');
          const hit = cell.get(`${r}#${k}`);
          if (!hit) {
            const none = 'never ran — no configuration the gateway recorded used these settings';
            return `<td><span class="pm-sq empty" role="img" title="${esc(none)}" aria-label="${esc(`${where}: ${none}`)}"></span></td>`;
          }
          const [g, i] = hit;
          const tier = perfTier(g, PM_KEY, pm.scales);
          const bootOnly = !g.agg.n && (g.boots || []).length;
          const v = bootOnly ? `booted ${g.boots.length} time${g.boots.length === 1 ? '' : 's'}, no traffic recorded — no figure of its own to compare`
            : `${valText(MEASURES[PM_KEY], g.agg[PM_KEY], true)} generated over ${fmtInt(g.agg.n)} recorded minute${g.agg.n === 1 ? '' : 's'}`
              + (g.agg.n < MIN_WINDOWS
                ? ` — fewer than ${MIN_WINDOWS} recorded minutes, so it is shown but never compared: its square is outlined, not shaded`
                : tier ? ` — shaded ${tier} of ${TIERS} on the tok/s ramp, worst to best`
                  : ` — not shaded: ${perfTierWhy(g, PM_KEY, pm.scales)}`);
          // A square with traffic prints its tok/s; one that only ever booted is a different KIND of cell
          // (`boot`: no fill, a dotted edge, the ↻ count) — not a number in a smaller font.
          return `<td><button type="button" class="pm-sq${tier ? '' : ' untiered'}${g.agg.n < MIN_WINDOWS ? ' few' : ''}${bootOnly ? ' boot' : ''}" data-i="${i}"`
            + ` title="${esc(v)}"${tier ? ` data-heat="${tier}"` : ''} aria-pressed="false" aria-label="${esc(`${where}: ${v}`)}">`
            + `<span class="pm-v">${bootOnly ? `<span class="pm-boot">${esc(g.boots.length)}&#x21bb;</span>` : esc(fmt(g.agg[PM_KEY], 0))}</span></button></td>`;
        }).join('')}</tr>`).join('')}</tbody></table></figure>`;
  }).join('');
  const host = $('pm-grids');
  host.classList.toggle('many', models.length > 1);   // stacked down the gutter, not across it — see the CSS
  host.innerHTML = html;
  // NO KEY ROW is written under the matrix. The operator asked for the panel's legend gone, and an
  // earlier instruction of mine — never remove information to make a thing prettier — does not survive
  // that: what it protected is the INFORMATION, and the information did not move off the page, it moved
  // ONTO THE SQUARE. Each cell's title and aria-label above now say what the four swatches said, about
  // that cell: its ramp step out of TIERS (or perfTierWhy, when it has none), the under-MIN_WINDOWS
  // outline, the boot-only ↻, "never ran". perfTier and MIN_WINDOWS are untouched — only the key is.
  for (const sq of host.querySelectorAll('button.pm-sq')) {
    const i = Number(sq.dataset.i);
    // A click SELECTS — every chart brings that configuration forward — and a click on the selected square
    // deselects it (a pressed button pressed again lets go; `all` does the same in words). A click never
    // opens the details: almost every click is made to point the charts at one configuration, and a card
    // that opened on each one covered the very marks the click had just pointed at. `details` opens it.
    sq.addEventListener('click', () => {
      const g = pm.groups[i];
      const m = g ? gcfg(g).model_choice || '' : '';
      // A square in ANOTHER model's grid: show that model first, then land on the square. Scoping the
      // charts to a configuration whose model they hold no mark of would fade every mark on the page.
      if (g && perf.current && m !== perf.current.id) { showModel(m, groupKey(g)); return; }
      pmSelect(pm.pinned === i ? null : i);
    });
  }
  // A re-render (the range changed, the timer fired) keeps the selection when it still exists.
  pmSelect(pm.pinned != null && groups[pm.pinned] ? pm.pinned : null);
  pmExpand(pm.open);
  pmPadBody();
}

/** WHERE THE PANEL LIVES, so that it can never cover a chart in any state. A fixed overlay covers whatever
 *  scrolls under it — a bottom padding only cleared the END of the page, and the panel still sat over the y
 *  axis of whichever chart was at the foot of the window. So on a WIDE window (≥ 900 px) the page reserves a
 *  LEFT GUTTER the panel's width (`main.page` is indented by --pm-gutter while `body.has-panel`) and the
 *  panel is fixed bottom-left INSIDE that gutter: the operator's corner, and no chart is ever under it, open
 *  or folded. A phone has no room for a gutter, so there the panel is DOCKED IN THE FLOW at the head of
 *  "Configurations compared", beside the legend it belongs with (#perf-matrix-dock); it folds there too.
 *  Called after every render, on resize, and when the panel is folded. */
const WIDE = '(min-width: 900px)';
const isWide = () => (typeof matchMedia === 'function' ? matchMedia(WIDE).matches : true);
function pmPlace() {
  const bar = $('perf-matrix');
  const wide = isWide();
  const dock = $('perf-matrix-dock');
  const home = wide ? document.body : dock;
  if (bar.parentNode !== home) home.append(bar);
  bar.classList.toggle('docked', !wide);
  document.body.classList.toggle('has-panel', wide && !bar.hidden);
  document.body.style.paddingBottom = '';
}
const pmPadBody = pmPlace;                       // the old name, for the callers that pad nothing now
if (typeof matchMedia === 'function') matchMedia(WIDE).addEventListener('change', pmPlace);
$('pm-all').addEventListener('click', () => pmSelect(null));
/** The panel FOLDS to a one-word strip and back: fixed to the corner, it sat over the y axis of whichever
 *  chart was at the foot of the window (a third of a chart on a phone). Folded, nothing it holds is lost —
 *  the squares, the selection and both buttons come back with one click — and the state survives the tick. */
function pmFold(folded) {
  pm.folded = !!folded;
  const bar = $('perf-matrix');
  bar.classList.toggle('folded', pm.folded);
  const b = $('pm-fold');
  b.setAttribute('aria-expanded', String(!pm.folded));
  b.textContent = pm.folded ? 'configurations ▸' : '▾';
  b.title = pm.folded ? 'unfold the configurations panel' : 'fold this panel to a strip, so it covers no chart';
  pmPadBody();
}
$('pm-fold').addEventListener('click', () => pmFold(!pm.folded));
const scrollHow = () => (matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth');
$('pm-toggle').addEventListener('click', () => {
  if (pm.open) { pmExpand(false); return; }
  // Opened from the bar at the foot of the window, the card is thousands of pixels up: remember where the
  // reader was (closing returns there), bring the card into view and put the focus in it.
  pm.returnTo = window.scrollY;
  pmExpand(true);
  const card = $('pm-detail');
  card.scrollIntoView({ block: 'start', behavior: scrollHow() });
  card.focus({ preventScroll: true });
});
document.addEventListener('keydown', (ev) => {
  if (ev.key !== 'Escape' || !pm.open) return;
  const inside = $('pm-detail').contains(document.activeElement);
  pmExpand(false);
  if (inside) $('pm-toggle').focus();       // focus must not be left inside a hidden card
});

/** The comparison as charts: the table's pooled figures as bars against max_num_seqs, and every
 *  recorded minute as its own dot on them. */
function renderPerfPanel(ws) {
  const host = $('perf-cmp-panel');
  const runs = ws.map(winAsRun).filter((r) => r.max_num_seqs != null);
  if (!runs.length) { host.innerHTML = ''; return; }
  const caps = capsOf(runs);
  const series = seriesIn(runs);
  const configs = new Set(ws.map(settingsKey)).size;
  const measures = PERF_COMPARE.filter(([key, over]) => runs.some((r) => num(winSpec(key, over).get(r.w)) != null));
  // The bars wear the section's one encoding (cfgStyle), so this panel needs no legend of its own: the
  // section's legend above names every bar here as it names every bar of the level charts.
  const styleOf = (s, cap) => cfgStyle({ ...s.cfg, max_num_seqs: cap });
  host.innerHTML = `<article class="cmp-panel">
    <header class="cmp-phead"><h3>Pooled over every level, by max_num_seqs</h3>
      <span class="sub">${configs} configuration${configs === 1 ? '' : 's'} &middot; ${runs.length} recorded minutes</span>
      <span class="scope-tag" hidden></span></header>
    <details class="how" data-keep="pooled-how"><summary>how to read this</summary>
    <p class="why">The bar is the configuration&rsquo;s pooled figure over all its minutes &mdash; the number in the
      table &mdash; and every recorded minute is its own dot on it. An outline bar labelled <i>n=&hellip;</i> has fewer
      than ${MIN_WINDOWS} minutes behind it: shown, not compared.</p></details>
    <div class="cmp-grid">${measures.map(([key, over]) => `<figure class="cmp-chart" data-m="${esc(key)}">
      <h4>${esc(cap1(winSpec(key, over).name))}${dirHTML(key)}</h4></figure>`).join('')}</div>
    ${perfTwinHTML(runs, caps, series, measures, styleOf)}</article>`;
  scrollHint(host.querySelector('details.cmp-table'));
  for (const fig of host.querySelectorAll('.cmp-chart')) {
    const [key, over] = PERF_COMPARE.find(([k]) => k === fig.dataset.m);
    const spec = winSpec(key, over);
    // ONE encoding whatever the count: the pooled bar, the cloud of minutes, the mean line. Nineteen
    // minutes drawn as nineteen time-ordered marks read as a trend line beside a cell drawn as a cloud.
    compChart(fig, { runs, caps, series, measure: key, noun: 'minutes', minN: MIN_WINDOWS, dense: 0, tipId: 'perf',
      spec: { ...spec, get: (r) => spec.get(r.w) }, pool: (rs) => aggregate(rs.map((r) => r.w))[key], style: styleOf });
  }
}

/** The table view the real-traffic panel's bars owe — and the keyboard's way to their numbers, since no
 *  hit area is a tab stop (249 of them would be a tab order nobody could use): per setting and series, the
 *  minutes, then each measure's POOLED figure with the p10–p90 of its minutes under it. */
function perfTwinHTML(runs, caps, series, measures, styleOf) {
  const rows = [];
  for (const cap of caps) {
    for (const s of series) {
      const cell = runs.filter((r) => seriesKey(r) === s.key && num(r.max_num_seqs) === cap);
      if (!cell.length) continue;
      const icon = swatchHTML(styleOf(s, cap), { few: cell.length < MIN_WINDOWS });
      const agg = aggregate(cell.map((r) => r.w));
      const tds = measures.map(([key, over]) => {
        const spec = winSpec(key, over);
        const vals = cell.map((r) => num(spec.get(r.w))).filter((v) => v != null).sort((a, b) => a - b);
        const pooled = num(agg[key]);
        if (pooled == null && !vals.length) return '<td class="num dim">—</td>';
        const rank = (q) => sampleQuantile(vals, q);
        return `<td class="num"><b>${esc(valText(spec, pooled != null ? pooled : meanOf(vals), true))}</b>${vals.length > 1
          ? `<span class="rng">${esc(valText(spec, rank(0.1), true))} – ${esc(valText(spec, rank(0.9), true))}</span>` : ''}</td>`;
      }).join('');
      rows.push(`<tr><td class="num">${esc(String(cap))}</td><td>${icon} ${esc(seriesText(s))}</td>`
        + `<td class="num">${cell.length}${cell.length < MIN_WINDOWS
          ? ` <span class="few-mark" title="fewer than ${MIN_WINDOWS} recorded minutes: shown, not compared" role="img" aria-label="shown, not compared">&#9676;</span>` : ''}</td>${tds}</tr>`);
    }
  }
  return `<details class="cmp-table" data-keep="cmp-table"><summary>Table view — the pooled figure and the p10–p90 of its minutes, per setting</summary>
    <p class="table-tools"><span class="scroll-hint">wider than the window &mdash; scrolls sideways; the first column stays put</span></p>
    <div class="tablewrap"><table class="runs-t twin"><thead><tr><th scope="col" class="num">seqs</th>
    <th scope="col">series</th><th scope="col" class="num">minutes</th>${measures.map(([key, over]) =>
      `<th scope="col" class="num">${esc(winSpec(key, over).name)}${dirHTML(key, false)}</th>`).join('')}</tr></thead>
    <tbody>${rows.join('')}</tbody></table></div></details>`;
}

// ── the real-traffic batch-size curve: each configuration against its parallel level ────────────────

/** The recorder's parallel buckets (perflog.BUCKETS), in order: the engine's running count. These are
 *  the data keys; the axis ALWAYS carries all of them, 1 through 30, so a level no configuration ever
 *  reached is an empty slot — the recorder's open-ended last bucket ("25+") is shown as 25–30. */
const PAR_BUCKETS = ['1', '2', '3-4', '5-8', '9-12', '13-16', '17-24', '25+'];
const parLabel = (lv) => (lv === '25+' ? '25\u201330' : lv);
/** A (configuration, parallel level) point with fewer probe pairs than this — or whose configuration
 *  has fewer than MIN_WINDOWS minutes in all, the table's own rule — is drawn hollow and joined by no
 *  line: shown, not compared. 100 pairs is ~10 minutes of reads at that level. */
const MIN_PAIRS = 100;
const tooFewAt = (l, f) => f.pairs < MIN_PAIRS || l.minutes < MIN_WINDOWS;
const tooFewWhy = (l, f) => (l.minutes < MIN_WINDOWS
  ? `its configuration has fewer than ${MIN_WINDOWS} recorded minutes` : `fewer than ${MIN_PAIRS} probe pairs at this level`);

/** An empty per-level accumulator, and the fold of ONE window's view of that level into it. Hoisted out
 *  of bucketSums because the PER-MINUTE pass folds exactly one window into one of these and reads it with
 *  levelFigures itself: a single minute's figure is then the cell's own arithmetic applied to one minute,
 *  never a second formula that could drift from the pooled one. `obs` keeps the views themselves — the
 *  recorded minutes, which are the sample the per-minute band is a percentile of. */
const newLevelAcc = () => ({ has: {}, ttft: new Map(), boxes: {}, obs: [] });
function addLevelBucket(acc, b) {
  for (const k of ['pairs', 'span_s', 'req_s', 'gen_tokens', 'prompt_tokens', 'prompt_computed_tokens',
    'prefix_hit_tokens', 'prefix_query_tokens', 'requests_finished', 'ttft_n', 'ttft_sum', 'itl_n', 'itl_sum',
    'draft_tokens', 'accepted', 'flops', 'bytes', 'kv_s', 'kv_t', 'waiting_s']) {
    const v = num(b[k]);
    if (v != null) { acc[k] = (acc[k] || 0) + v; acc.has[k] = true; }
  }
  for (const [le, n] of (b.ttft_buckets || [])) {
    const key = le == null ? Infinity : num(le);
    if (key != null && num(n) != null) acc.ttft.set(key, (acc.ttft.get(key) || 0) + n);
  }
  for (const [box, x] of Object.entries(b.boxes || {})) {
    const t = acc.boxes[box] || (acc.boxes[box] = { sm_s: 0, sm_t: 0, pwr_s: 0, pwr_t: 0 });
    for (const k of ['sm_s', 'sm_t', 'pwr_s', 'pwr_t']) t[k] += num(x[k]) || 0;
  }
  acc.obs.push(b);
  return acc;
}

/** Every window's per-level sums, summed per (configuration, level). A sum of sums: exact. */
function bucketSums(ws) {
  const out = new Map();                          // settingsKey -> { ws, levels: Map(label -> sums) }
  for (const w of ws) {
    const key = settingsKey(w);
    const g = out.get(key) || { ws: [], levels: new Map() };
    out.set(key, g);
    g.ws.push(w);
    for (const [label, b] of Object.entries(w.by_running || {})) {
      if (!b || typeof b !== 'object') continue;
      const acc = g.levels.get(label) || newLevelAcc();
      g.levels.set(label, acc);
      addLevelBucket(acc, b);
    }
  }
  return out;
}

/** The engine's latency histogram at one level, as a reader reads a DISTRIBUTION: each non-empty bucket
 *  (the recorder writes no other) with the bound below it, its count, its share of the level's requests
 *  and the share up to it. A sum of the recorded counts — exact, no interpolation; the open-ended last
 *  bucket keeps `le` Infinity. Empty when nothing was measured. */
function histRows(counts) {
  const les = [...counts.keys()].sort((a, b) => a - b);
  const total = les.reduce((s, le) => s + (num(counts.get(le)) || 0), 0);
  if (!(total > 0)) return [];
  const rows = [];
  let cum = 0, lo = 0;
  for (const le of les) {
    const n = num(counts.get(le)) || 0;
    cum += n;
    if (n > 0) rows.push({ lo, le, n, share: n / total, cum: cum / total });
    if (Number.isFinite(le)) lo = le;
  }
  return rows;
}

/** What one (configuration, level) sum says, per measure. `oneMinute` is the per-minute pass reading a
 *  single window's view — it takes no sample of itself (that is the recursion's floor). */
function levelFigures(a, oneMinute) {
  const div = (x, y, k = 1) => (a.has[x] && num(a[y]) > 0 ? a[x] / a[y] / k : null);
  // The level's own TTFT histogram (per-bucket counts summed over windows), in the shape pooledQuantile reads.
  const ttft = [{ h: { buckets: [...a.ttft.entries()].map(([le, n]) => [Number.isFinite(le) ? le : null, n]) } }];
  const rows = histRows(a.ttft);
  const busiest = Object.values(a.boxes).map((b) => (b.sm_t > 0 ? b.sm_s / b.sm_t : null)).filter((v) => v != null);
  return {
    pairs: a.pairs || 0, minutes: (a.span_s || 0) / 60,
    agg_tok_s: div('gen_tokens', 'span_s'), per_request: div('gen_tokens', 'req_s'),
    prompt_tok_s: div('prompt_tokens', 'span_s'), prefix_hit: div('prefix_hit_tokens', 'prefix_query_tokens'),
    ttft_mean: div('ttft_sum', 'ttft_n'),
    // The spread of the requests' TTFT at this level, off the pooled histogram: the requests it counted,
    // p10 / median / p90 (Prometheus' interpolation, exact within a bucket) and the distribution itself.
    ttft_n: rows.reduce((s, r) => s + r.n, 0), ttft_rows: rows,
    ttft_p10: pooledQuantile(ttft, 'h', 0.1), ttft_p50: pooledQuantile(ttft, 'h', 0.5), ttft_p90: pooledQuantile(ttft, 'h', 0.9),
    ttft_p99: pooledQuantile(ttft, 'h', 0.99),
    waiting_mean: div('waiting_s', 'span_s'), kv_mean: div('kv_s', 'kv_t'),
    spec_accept: div('accepted', 'draft_tokens'), sm_fill: busiest.length ? Math.max(...busiest) : null,
    tflops: a.has.flops && a.flops > 0 ? div('flops', 'span_s', 1e12) : null,
    gbps: a.has.flops && a.flops > 0 && a.has.bytes ? div('bytes', 'span_s', 1e9) : null,
    requests: a.requests_finished || 0,
    // THE SAMPLE NOBODY HAD USED: each recorded minute's own figure at this level, per measure. The count
    // of minutes is the cell's (every observation), the per-measure arrays only the minutes that measured
    // that one — so a band's n is its own array's length, never the cell's count.
    minute_n: a.obs ? a.obs.length : 0,
    minute: oneMinute ? {} : minuteSamples(a),
  };
}

/** The measures whose spread at a level is across the RECORDED MINUTES. Every measure but time to first
 *  token, which has the engine's own per-REQUEST histogram behind it and keeps that instead: one measure,
 *  one spread, and the page never mixes the two quantities. */
const MINUTE_SPREAD = ['agg_tok_s', 'per_request', 'prompt_tok_s', 'prefix_hit', 'waiting_mean',
  'spec_accept', 'kv_mean', 'sm_fill', 'tflops', 'gbps'];

/** Each recorded minute's own figure at this level, per measure, sorted ascending. A minute is folded into
 *  an accumulator of its own and read with levelFigures — so "this minute's tok/s" is exactly the formula
 *  the cell's bar uses, over one minute. A minute that measured nothing for a measure is absent from that
 *  measure's array rather than counted as a zero. */
function minuteSamples(a) {
  const out = {};
  for (const k of MINUTE_SPREAD) out[k] = [];
  for (const b of (a.obs || [])) {
    const one = levelFigures(addLevelBucket(newLevelAcc(), b), true);
    for (const k of MINUTE_SPREAD) { const v = num(one[k]); if (v != null) out[k].push(v); }
  }
  for (const k of MINUTE_SPREAD) out[k].sort((x, y) => x - y);
  return out;
}

/** The curve's measures. Those with a direction take it from MEASURES; load and hardware have none,
 *  and say so by carrying no ▲/▼.
 *
 *  WHAT A POINT CAN HONESTLY SAY BESIDES ITS MEAN, AND WHAT ITS SPREAD IS A SPREAD OF. These are two
 *  different quantities and the page must never let them be read as one:
 *    * `spread: 'ttft'` — ACROSS THE REQUESTS. Time to first token alone has the engine's own latency
 *      HISTOGRAM per (configuration, level), so a percentile of the REQUESTS is computable, and the hover
 *      lists the distribution it came from.
 *    * `spread: 'minutes'` — ACROSS THE RECORDED MINUTES. Every other measure is a sum or a time-weighted
 *      mean over the level's seconds, with no per-request sample kept: a per-request spread here would be
 *      invented from nothing, and never is. But each of the cell's MINUTES is one recorded observation of
 *      it (perflog keeps `by_running` per window, and this page already draws the windows one at a time),
 *      so a percentile across those minutes is real — and it is a different sentence: "of the 12 minutes
 *      recorded at 13–16 parallel, the slowest tenth ran below X tok/s".
 *  Both are DRAWN the same way on purpose — one spread encoding on the page, so a reader learns to read
 *  it once — which means the words have to carry the distinction: every measure's `note` says which of
 *  the two its band is, and the hover, the legend and the table view all repeat it. */
const RATE_NOTE = 'the bar is tokens ÷ seconds over all of this level’s minutes — one rate, never a per-request figure: the band is the spread of those recorded minutes, not across the requests';
const GAUGE_NOTE = 'the bar is the time-weighted mean of this level’s reads — the recorder keeps the sum, not the samples: the band is the spread of the recorded minutes, not across the requests';
const RATIO_NOTE = 'the bar is a ratio of two of this level’s sums: the band is the spread of the recorded minutes, not across the requests';
const CURVE = [
  // `lead`: the four that answer the page's question, drawn open; the rest open under one fold.
  { key: 'agg_tok_s', name: 'generated tok/s, all requests', unit: 'tok/s', note: RATE_NOTE, spread: 'minutes', lead: true },
  { key: 'per_request', name: 'generated tok/s per request', unit: 'tok/s', spread: 'minutes', lead: true,
    note: 'the bar is tokens ÷ request-seconds over all of this level’s minutes — one rate: the band is the spread of those recorded minutes, not across the requests' },
  { key: 'ttft_mean', name: 'mean time to first token', unit: 's', spread: 'ttft', lead: true },
  { key: 'prompt_tok_s', name: 'billed prompt tok/s', unit: 'tok/s', note: RATE_NOTE, spread: 'minutes', lead: true },
  { key: 'prefix_hit', name: 'prefix cache hit rate', frac: true, note: RATIO_NOTE, spread: 'minutes' },
  { key: 'waiting_mean', name: 'requests waiting', unit: 'requests', note: GAUGE_NOTE, spread: 'minutes' },
  { key: 'spec_accept', name: 'speculation acceptance', frac: true, note: RATIO_NOTE, spread: 'minutes' },
  { key: 'kv_mean', name: 'KV cache used', frac: true, load: true, note: GAUGE_NOTE, spread: 'minutes' },
  { key: 'sm_fill', name: 'SM fill, busier box', unit: '%', load: true, spread: 'minutes',
    note: 'the bar is the busier box’s time-weighted mean of its nvidia-smi reads at this level — the recorder keeps the sum, not the samples: the band is the spread of the recorded minutes, not across the requests' },
  { key: 'tflops', name: 'engine TFLOP/s per GPU', unit: 'TFLOP/s', load: true, spread: 'minutes',
    note: 'the bar is FLOPs ÷ seconds at this level — one rate: the band is the spread of the recorded minutes, not across the requests' },
  { key: 'gbps', name: 'engine GB/s per GPU', unit: 'GB/s', load: true, spread: 'minutes',
    note: 'the bar is bytes ÷ seconds at this level — one rate: the band is the spread of the recorded minutes, not across the requests' },
];
/** A p10–p90 over fewer REQUESTS than this is the two extremes of a handful, not a spread: the point keeps
 *  its mean, the hover says how many requests it saw, and no band is drawn. */
const MIN_SPREAD_N = 20;
/** The same rule for the per-minute band, in the unit a MINUTE comes in — and its own number, because the
 *  samples are not comparable: a level's TTFT histogram counts requests in the hundreds per minute, while
 *  a cell holds one observation per minute recorded. Ten is where "the slowest 10% of these minutes" first
 *  names a whole recorded minute; below it the phrase has no referent, so no band is drawn and the hover
 *  says how many minutes there were instead. (Measured against the log the gateway actually serves on
 *  2026-10-01: the biggest (configuration, level) cell in seven days holds 12 minutes, so MIN_SPREAD_N's
 *  20 — a low bar for a histogram of requests — would have drawn this band on nothing at all.) */
const MIN_SPREAD_MINUTES = 10;

/** The per-minute band of one (configuration, level) cell for one measure: p10, the median and p90 across
 *  the minutes recorded there, with the count behind them. null when the measure keeps no per-minute
 *  sample, or when too few minutes were recorded for a tenth of them to name a minute.
 *
 *  The MEDIAN is in here because the bar is the cell POOLED — tokens ÷ seconds over every minute, so a long
 *  minute weighs more than a short one — which is a mean, and not the middle of the minutes. Two different
 *  numbers, both right; the page prints both and names each (meanVsMedianNote). */
function minuteBand(f, key) {
  const vs = f && f.minute ? f.minute[key] : null;
  if (!vs || vs.length < MIN_SPREAD_MINUTES) return null;
  return { kind: 'minutes', n: vs.length,
    lo: sampleQuantile(vs, 0.1), mid: sampleQuantile(vs, 0.5), hi: sampleQuantile(vs, 0.9) };
}

/** THE MEAN IS NOT THE MEDIAN, and on this page the gap is the finding rather than a discrepancy to hide:
 *  the bar is seconds-weighted over the whole cell, the median counts each recorded minute once. Printed
 *  side by side with no word they read as a contradiction, so wherever both appear this sentence goes with
 *  them. '' when they agree inside a twentieth — there is then nothing to explain. */
function meanVsMedianNote(mean, med) {
  const a = num(mean), b = num(med);
  if (a == null || b == null || !(Math.abs(a) > 0) || Math.abs(a - b) / Math.abs(a) < 0.05) return '';
  return `the mean sits ${a < b ? 'below' : 'above'} the median: the mean is seconds-weighted over the whole`
    + ' cell, the median counts each recorded minute once — their gap is this cell’s skew';
}
/** True when the p90 rank fell in the open-ended last bucket: the quantile then returns the last finite
 *  bound (perflog.quantile's rule), which is a FLOOR of the real p90, and the tooltip says so. */
const p90IsFloor = (rows) => {
  const last = rows[rows.length - 1];
  return !!last && !Number.isFinite(last.le) && last.cum - last.share < 0.9;
};
/** A latency as a tooltip row prints it: the bucket bound, short. */
const secText = (v) => (v >= 10 ? `${Math.round(v)} s` : v >= 1 ? `${fmt(v)} s` : v >= 0.01 ? `${Math.round(v * 1000)} ms` : `${(v * 1000).toFixed(1)} ms`);
/** The distribution rows of a point's tooltip: one line per non-empty bucket, its range, a bar of its
 *  share (ink, never a hue: it is context) and the share as a number. */
function distHTML(rows) {
  return `<div class="dist">${rows.map((r) => `<span class="tk">${esc(Number.isFinite(r.le)
    ? (r.lo ? `${secText(r.lo)}–${secText(r.le)}` : `≤ ${secText(r.le)}`) : `> ${secText(r.lo)}`)}</span>`
    + `<i style="width:${(r.share * 100).toFixed(1)}%"></i><span>${esc(`${(r.share * 100).toFixed(0)}%`)}</span>`).join('')}</div>`;
}
const curveSpec = (c) => ({ ...(MEASURES[c.key] || {}), name: c.name, unit: c.unit, frac: !!c.frac,
  dir: c.load ? 0 : (MEASURES[c.key] || {}).dir });

/** THE ONE ENCODING of a configuration, everywhere on this page — the level charts, the by-max_num_seqs
 *  panel, the time charts, the legend, the matrix's dot and the details card all draw it the same way:
 *    * its COLOUR is its cap's ramp step (max_num_seqs is ordered, so the page's one-hue ordinal blue —
 *      yesterday's rule, and the one that made those charts readable);
 *    * when a SECOND configuration shares that cap (16 seqs at batched 8,192 and at 16,384), the two are
 *      told apart by FILL, never by a second hue: the cap's first configuration (spec, then batched order)
 *      is solid, the second 45° hatched, a third 135°, and any past that a dotted outline. A hatched bar
 *      reads at 10 px where a diamond-against-a-circle at 4.5 px did not.
 *  The time charts dash the same variants; `swatchHTML` draws the same swatch for a legend line. */
const VARIANTS = ['solid', 'hatch', 'hatch2', 'dotted'];
const VARIANT_DASH = { solid: null, hatch: '6 3', hatch2: '2 3', dotted: '1 3' };
/** Everything that tells two configurations AT ONE CAP apart: every part of the key but the model and the cap. */
const variantKey = (cfg) => CFG_PARTS.map((p) => partToken(cfg, p.key)).join('|');
function cfgStyle(cfg) {
  const cap = num(cfg.max_num_seqs);
  const rank = Math.max(0, (perf.variants.get(cap) || []).indexOf(variantKey(cfg)));
  const variant = VARIANTS[Math.min(rank, VARIANTS.length - 1)];
  return { color: capColor(cap), variant, rank, dash: VARIANT_DASH[variant] };
}
/** A configuration in words: its cap, spec and batched, then every other setting that differs among its
 *  model's configurations ("16 seqs · spec 0 · batched vLLM default · KV 7.92 GiB"). */
function cfgText(cfg) {
  return `${cfg.max_num_seqs ?? NOT_RECORDED} seqs · ${partsText(cfg)}`;
}
/** cfgText as HTML that breaks only BETWEEN settings, never inside one: a name that now carries the pin, the
 *  fraction and the RAM tier is wider than a phone, and "KV 7.92" over "GiB" is not a setting. */
const cfgTextHTML = (cfg) => cfgText(cfg).split(' · ').map((t) => `<span class="cfg-seg">${esc(t)}</span>`).join(' · ');
/** ...and everything after the cap, where the cap has a column of its own (the pooled panel's table). */
function partsText(cfg) {
  const vary = perf.varying.get(cfg.model_choice) || new Set();
  return CFG_PARTS.filter((p) => p.always || vary.has(p.key)).map((p) => partText(p, cfg)).join(' · ');
}

/** The fill of a bar in a configuration's style: its colour, or a hatch of it — one <pattern> per chart
 *  and style, in the chart's own <defs>, so two charts never share an id. */
let patSeq = 0;
function fillFor(svg, st) {
  if (st.variant === 'solid') return st.color;
  if (!svg.dataset.seq) svg.dataset.seq = String(++patSeq);
  let defs = svg.querySelector('defs');
  if (!defs) { defs = svgEl('defs'); svg.prepend(defs); }
  const id = `p${svg.dataset.seq}-${st.rank}-${st.color.replace(/[^a-z0-9]/gi, '')}`;
  if (!defs.querySelector(`#${id}`)) {
    const pat = svgEl('pattern', { id, patternUnits: 'userSpaceOnUse', width: 6, height: 6,
      patternTransform: `rotate(${st.variant === 'hatch2' ? 135 : 45})` });
    // A wash of the colour under the strokes, so the bar still reads as its cap's blue from across the room.
    pat.append(svgEl('rect', { width: 6, height: 6, fill: st.color, opacity: st.variant === 'dotted' ? 0.18 : 0.26 }));
    pat.append(svgEl('line', { x1: 0, y1: 0, x2: 0, y2: 6, stroke: st.color, 'stroke-width': st.variant === 'dotted' ? 1 : 2.2,
      'stroke-dasharray': st.variant === 'dotted' ? '1 2' : null }));
    defs.append(pat);
  }
  return `url(#${id})`;
}
/** A legend swatch in a configuration's style: the bar as the charts draw it (an outline when `few`), or a
 *  short line with the variant's dashes for the time charts. */
function swatchHTML(st, o) {
  const opt = o || {};
  const svg = svgEl('svg', { width: 18, height: 12, viewBox: '0 0 18 12', 'aria-hidden': 'true', class: 'lg-mark' });
  if (opt.line) {
    svg.append(svgEl('line', { x1: 1, x2: 17, y1: 6, y2: 6, stroke: st.color, 'stroke-width': 2, 'stroke-linecap': 'round', 'stroke-dasharray': st.dash }));
  } else {
    // `few`: the ghost the charts draw — the fill faded, a thin dashed edge in its colour.
    svg.append(svgEl('rect', { x: 1, y: 1, width: 16, height: 10, rx: 2, fill: fillFor(svg, st), 'fill-opacity': opt.few ? 0.3 : 1,
      stroke: st.color, 'stroke-width': opt.few ? 1 : 0, 'stroke-dasharray': opt.few ? '3 2' : null }));
  }
  return svg.outerHTML;
}
/** The section's ONE legend: every configuration charted, one line each — its swatch, its settings and its
 *  recorded minutes — then, quieter, what the charts' conventions mean. `lines` are renderCurve's. */
function cfgLegendHTML(lines, o) {
  const opt = o || {};
  // A configuration that is compared reads in full ink and weight; one shown-but-not-compared (`few`)
  // is set quieter — the app's bold row against its hollow one. Its words still say why, on its own line.
  const items = lines.map((l) => `<span class="lg${l.minutes < MIN_WINDOWS ? ' few' : ''}" data-k="${esc(l.key)}">${swatchHTML(l.style, { line: opt.line })}<span class="lg-name">${cfgTextHTML(l.cfg)}</span>`
    + `${l.fallback ? '<span class="derived" title="spec and batched assumed: no cluster.effective published yet">*</span>' : ''}`
    + ` <span class="lg-n">${esc(fmtInt(l.minutes))} min${l.minutes < MIN_WINDOWS ? ' — too few to compare' : ''}</span></span>`);
  return `<div class="cmp-legend cfg-legend">${items.join('')}</div>`;
}

/** ONE measure at every parallel level, every configuration a BAR in each level's band — yesterday's bar
 *  per cap with today's second dimension. The bars are grouped in the configurations' fixed order (the
 *  legend's), 2 px of surface between them, in each configuration's style (cfgStyle). A point too thin to
 *  compare is an OUTLINE bar. The y scale is the BARS': a whisker (time to first token's p10–p90 of the
 *  requests, in ink over the bar) that runs past the top is clipped with an open end and its p90 printed
 *  there — a 540 s tail on a 5 s mean used to set the axis and squash every bar to the floor. */
function levelChart(host, lines, levels, c) {
  const m = curveSpec(c);
  host.querySelectorAll('svg, .empty').forEach((n) => n.remove());
  const pts = lines.flatMap((l) => levels.map((lv, i) => {
    const f = l.levels.get(lv);
    const v = f ? num(f[c.key]) : null;
    return v == null ? null : { l, i, lv, v, f, few: tooFewAt(l, f) };
  }).filter(Boolean));
  if (!pts.length) {
    host.append(Object.assign(document.createElement('p'), { className: 'empty', textContent: 'not measured at any level' }));
    return;
  }
  // The band a point can draw, and WHAT IT IS A BAND OF — the two are different quantities (CURVE):
  // time to first token's is the p10–p90 of its REQUESTS, off the engine's own histogram; every other
  // measure's is the p10–p90 of the MINUTES recorded at that level. Both carry their median, because the
  // bar is a mean and a mean is not the middle. A point shown-not-compared keeps its band, fainter.
  const bandOf = (p) => (c.spread === 'ttft'
    ? (num(p.f.ttft_p10) != null && num(p.f.ttft_p90) != null && p.f.ttft_n >= MIN_SPREAD_N
      ? { kind: 'requests', n: p.f.ttft_n, lo: p.f.ttft_p10, mid: p.f.ttft_p50, hi: p.f.ttft_p90 } : null)
    : (c.spread === 'minutes' ? minuteBand(p.f, c.key) : null));
  // THE SCALE IS THE BARS' — every pooled figure, compared or not — and never a band's. The spread is
  // BACKGROUND now, so it has even less business setting the axis than the old whisker had: a 145 s p90
  // over a 10 s mean would squash every bar to the floor to annotate one cell. A band that runs past the
  // top is cut off at the ceiling, with no arrow and no printed number — the hover says how far it goes.
  const top = Math.max(...pts.map((p) => p.v));
  // Eight slots on a narrow chart: the labels would touch, so every other one drops to a second row.
  const stagger = (contentWidth(host) - 40) / levels.length < 36;
  const f = frame(host, { xCat: levels, yHi: (m.frac ? Math.min(1, top * 1.25) : top * 1.25) || 1,
    yTitle: axisTitle(m), yFmt: m.frac ? (v) => `${Math.round(v * 100)}%` : null, aspect: 0.58,
    padBottom: stagger ? 44 : 34, title: `${m.name} by requests running in parallel` });
  // One slot per configuration inside every band, in the fixed order, whether or not it has a bar there:
  // a configuration is at the same offset in every band, so the eye can follow it across the levels.
  const inner = f.band * 0.84, sub = inner / Math.max(1, lines.length);
  const barW = Math.max(3, Math.min(24, sub - 2));
  // The band is the BAR'S OWN COLUMN continued, never a wider one beside it: at this density a bar is 3 px
  // and a band 3 px wider than it read as the louder of the two — which is the opposite of what a spread
  // drawn as background is for. The MEDIAN tick is the one mark allowed to overhang, so the centre is what
  // the eye picks out of the column.
  const bandW = barW;
  const medW = Math.min(sub, barW + 4);
  const px = (p) => f.x(p.i) - inner / 2 + sub * (lines.indexOf(p.l) + 0.5);
  const yBase = f.y(f.yLo);
  const vlab = (v) => (m.frac ? `${(v * 100).toFixed(0)}%` : fmt(v));
  for (const p of pts) {
    const x = px(p), st = p.l.style, b = bandOf(p);
    // ONE GROUP PER CELL, so the hover can bring its own band forward (benchmarks.css) and the matrix's
    // selection can fade the whole cell by its one data-k rather than four marks separately.
    const g = svgEl('g', { class: `lv-cell${p.few ? ' few' : ''}`, 'data-k': p.l.key });
    // A bar shown-not-compared is a GHOST of the bar it would be: its own fill faded, a thin dashed edge in
    // its colour — quieter than the compared bars beside it, still its cap's blue and its variant's hatch.
    const bar = svgEl('path', { class: `g-bar lv-bar${p.few ? ' few' : ''}`,
      d: barPath(x - barW / 2, x + barW / 2, f.y(p.v), yBase), fill: fillFor(f.svg, st) });
    if (p.few) bar.style.stroke = st.color;
    g.append(bar);
    // THE EXTREMES RECEDE: the p10–p90 as a soft wash over the bar and the surface, no line and no cap.
    // Clipped to the plot, floored at 1.5 px so a tight spread is still a mark rather than nothing.
    if (b && num(b.lo) != null && num(b.hi) != null) {
      const yTop = Math.max(f.pad.t, f.y(Math.min(b.hi, f.yHi)));
      const yBot = Math.min(yBase, f.y(Math.max(f.yLo, Math.min(b.lo, f.yHi))));
      g.append(svgEl('rect', { class: 'lv-band', x: x - bandW / 2, width: bandW, rx: 1.5,
        y: yTop, height: Math.max(1.5, yBot - yTop) }));
    }
    // THE CENTRE STAYS LEGIBLE: the bar's top edge is the MEAN, and this tick is the MEDIAN of the very
    // sample the band is over. They are different numbers on purpose (meanVsMedianNote) — the dark halo
    // keeps the tick readable on the pale end of the cap ramp as well as on the surface.
    if (b && num(b.mid) != null && b.mid <= f.yHi && b.mid >= f.yLo) {
      const ym = f.y(b.mid);
      for (const cls of ['lv-med halo', 'lv-med']) {
        g.append(svgEl('line', { class: cls, x1: x - medW / 2, x2: x + medW / 2, y1: ym, y2: ym }));
      }
    }
    // The hover target is the cell's whole slot, from the top of its band or bar to the baseline.
    const y0 = Math.min(f.y(p.v), b ? Math.max(f.pad.t, f.y(Math.min(b.hi, f.yHi))) : f.y(p.v)) - 6;
    const hit = svgEl('rect', { class: 'g-hit', x: x - sub / 2, y: y0, width: sub, height: Math.max(8, yBase - y0) });
    // One direct label per bar when it fits its slot (it does with one or two configurations); otherwise
    // the tooltip and the table view carry the number — a label over every one of 32 bars is noise. It sits
    // just over the bar's own top, which is where the MEAN is: the band may well reach higher, and the
    // reader is meant to see the centre first.
    if (!p.few && vlab(p.v).length * 6.4 + 4 <= sub) {
      g.append(txt(vlab(p.v), { class: 'g-label lv-lab', x, y: f.y(p.v) - 6, 'text-anchor': 'middle' }));
    }
    // THE FIGURES ARE IN THE HOVER, not burned into the plot: what the band is a band OF, in words, every
    // time — "10% of the requests" and "10% of the minutes" are different claims and must never be read
    // as one — then the exact p10, median and p90, and how far a cut-off band really went.
    const cut = b && b.hi > f.yHi
      ? `<br><span class="tk">its p90 is ${esc(c.spread === 'ttft' ? secText(b.hi) : vlab(b.hi))}, past the top of the`
        + ' chart: the band is cut off there rather than drawn out of it, and the axis stays the bars’</span>' : '';
    let more = '';
    if (b && b.kind === 'requests') {
      more = `<br><span class="tk">of its ${esc(fmtInt(b.n))} requests — p10</span> ${esc(valText(m, b.lo, true))}`
        + ` <span class="tk">median</span> <b>${esc(valText(m, b.mid, true))}</b>`
        + ` <span class="tk">p90</span> ${esc(valText(m, b.hi, true))} <span class="tk">p99</span> ${esc(valText(m, p.f.ttft_p99, true))}`
        + `<br><span class="tk">p10 and p90 across those <b>requests</b> themselves, off the engine’s own histogram`
        + ` at this level${p90IsFloor(p.f.ttft_rows) ? ' — its p90 is past the histogram’s last bound, so that figure is a floor' : ''}</span>`
        + cut
        + (tailNote(p.v, b.hi, 'waits') ? `<br><span class="tk">${esc(tailNote(p.v, b.hi, 'waits'))}</span>` : '')
        + distHTML(p.f.ttft_rows);
    } else if (b && b.kind === 'minutes') {
      const skew = meanVsMedianNote(p.v, b.mid);
      more = `<br><span class="tk">of its ${esc(fmtInt(b.n))} recorded minutes — p10</span> ${esc(valText(m, b.lo, true))}`
        + ` <span class="tk">median</span> <b>${esc(valText(m, b.mid, true))}</b>`
        + ` <span class="tk">p90</span> ${esc(valText(m, b.hi, true))}`
        + `<br><span class="tk">the slowest tenth of those <b>minutes</b> ran below the p10 and the fastest tenth`
        + ' above the p90 — one figure per minute, never one per request</span>'
        + cut
        + (skew ? `<br><span class="tk">${esc(skew)}</span>` : '')
        + `<br><span class="tk">${esc(c.note)}</span>`;
    } else if (c.spread === 'minutes') {
      const k = ((p.f.minute || {})[c.key] || []).length;
      more = `<br><span class="tk">${k ? `only ${esc(fmtInt(k))} recorded minute${k === 1 ? '' : 's'} at this level:`
        + ` no band drawn — a tenth of fewer than ${MIN_SPREAD_MINUTES} minutes is not a whole minute`
        : 'nothing recorded for this at this level'}</span>`
        + `<br><span class="tk">${esc(c.note)}</span>`;
    } else if (c.spread === 'ttft') {
      const n = p.f.ttft_n;
      more = `<br><span class="tk">${n ? `only ${esc(fmtInt(n))} request${n === 1 ? '' : 's'} counted here: no band drawn (fewer than ${MIN_SPREAD_N})`
        : 'no first-token histogram recorded at this level'}</span>`;
    } else if (c.note) {
      more = `<br><span class="tk">${esc(c.note)}</span>`;
    }
    hoverable(hit, `<b>${esc(valText(m, p.v, true))}</b> <span class="tk">${esc(m.name)} · ${esc(c.spread === 'ttft'
      ? 'the mean, weighted by the requests' : 'the mean, pooled over every minute of this cell')}</span>`
      + `<br>${esc(cfgText(p.l.cfg))}<br><span class="tk">running in parallel</span> ${esc(parLabel(p.lv))}`
      + `<br><span class="tk">${esc(fmtInt(p.f.pairs))} probe pairs, ${esc(fmt(p.f.minutes, 1))} min of traffic here, `
      + `${esc(fmtInt(p.f.requests))} requests finished</span>`
      + (p.few ? `<br><span class="tk">${esc(tooFewWhy(p.l, p.f))}: shown, not compared</span>` : '')
      + more, `curve|${c.key}|${p.l.key}|${p.lv}`);
    g.append(hit);
    f.svg.append(g);
  }
  levels.forEach((lv, i) => f.svg.append(txt(parLabel(lv), { class: 'g-tick', x: f.x(i),
    y: f.pad.t + f.H + 14 + (stagger && i % 2 ? 10 : 0), 'text-anchor': 'middle' })));
  f.svg.append(txt('requests running in parallel', { class: 'g-atitle', x: f.pad.l + f.W / 2, y: f.height - 4, 'text-anchor': 'middle' }));
  host.append(f.svg);
}

function renderCurve(ws) {
  const host = $('perf-curve');
  const sums = bucketSums(ws);
  const lines = [...sums.entries()].map(([key, g]) => {
    const cfg = gcfg(g);
    // `key` is the settings key of the line's minutes (data-k on everything drawn for it), so the matrix's
    // selection can scope these charts the way it scopes the time charts: a class flip, no redraw.
    return { key, cfg, style: cfgStyle(cfg), fallback: g.ws.some(fallbackCfg), minutes: g.ws.length,
      levels: new Map([...g.levels.entries()].map(([lv, a]) => [lv, levelFigures(a)])) };
  }).filter((l) => l.levels.size)
    .sort((a, b) => (num(a.cfg.max_num_seqs) || 0) - (num(b.cfg.max_num_seqs) || 0)
      || (num(a.cfg.num_speculative_tokens) || 0) - (num(b.cfg.num_speculative_tokens) || 0)
      || (num(a.cfg.max_num_batched_tokens) || 0) - (num(b.cfg.max_num_batched_tokens) || 0));
  if (!lines.length) { host.innerHTML = ''; $('perf-legend').innerHTML = ''; return; }
  const levels = PAR_BUCKETS;                     // the whole axis, 1 through 30: reached or not
  const seen = levels.filter((lv) => lines.some((l) => l.levels.has(lv))).length;
  // THE SECTION'S ONE LEGEND, above both panels: the configurations, then the conventions.
  $('perf-legend').innerHTML = cfgLegendHTML(lines)
    + `<div class="cmp-legend cfg-key"><span class="lg">${swatchHTML({ color: 'var(--ink-dim)', variant: 'solid', rank: 0 }, { few: true })}outline: too few to compare
        (under ${MIN_PAIRS} probe pairs at that level, or a configuration under ${MIN_WINDOWS} minutes)</span>
      <span class="lg"><svg class="lg-mark" width="16" height="16" viewBox="0 0 16 16" aria-hidden="true">
        <rect class="lv-band" x="4" y="2" width="8" height="12" rx="1.5"></rect>
        <line class="lv-med halo" x1="1.5" x2="14.5" y1="9" y2="9"></line><line class="lv-med" x1="1.5" x2="14.5" y1="9" y2="9"></line></svg>the
        pale band is p10 to p90, the tick their median &mdash; across the <b>${MIN_SPREAD_MINUTES}+ minutes recorded at that level</b>, one figure each</span>
      <span class="lg">&hellip; except on <b>time to first token</b>, where the band is across the <b>requests</b> themselves &mdash; the engine&rsquo;s own histogram</span>
      <span class="lg">the bar&rsquo;s top edge is the mean. Hover a bar for the figures, and for how far a band cut off at the top really went</span></div>`;
  const rows = lines.flatMap((l) => levels.filter((lv) => l.levels.has(lv)).map((lv) => {
    const x = l.levels.get(lv);
    // The mean, and under it the per-minute p10 · median · p90 wherever the cell has the minutes for one —
    // the twin-line convention the by-max_num_seqs table already uses, and the keyboard's way to the figures
    // the bands draw (no hit area on this page is a tab stop).
    const cell = (key, load) => {
      const c = CURVE.find((k) => k.key === key), sp = curveSpec(c), v = num(x[key]), b = minuteBand(x, key);
      return `<td class="num${load ? ' load' : ''}">${v == null ? '—' : esc(valText(sp, v, true))}`
        + (b ? `<span class="rng" title="p10 · median · p90 across the ${esc(fmtInt(b.n))} minutes recorded at this level — minutes, not requests">`
          + `${esc(valText(sp, b.lo, true))} · <b>${esc(valText(sp, b.mid, true))}</b> · ${esc(valText(sp, b.hi, true))}</span>` : '')
        + '</td>'; };
    return `<tr><td>${swatchHTML(l.style, { few: tooFewAt(l, x) })} ${esc(cfgText(l.cfg))}${l.fallback ? '<span class="derived" title="spec and batched assumed: no cluster.effective published yet">*</span>' : ''}</td>
      <td class="num">${esc(parLabel(lv))}</td><td class="num">${esc(fmtInt(x.pairs))}${tooFewAt(l, x)
        ? ` <span class="few-mark" title="${esc(tooFewWhy(l, x))}: shown, not compared" role="img" aria-label="shown, not compared">&#9676;</span>` : ''}</td>
      <td class="num">${esc(fmt(x.minutes, 1))}</td>
      <td class="num">${esc(fmtInt(x.minute_n))}${x.minute_n < MIN_SPREAD_MINUTES
        ? ` <span class="few-mark" title="fewer than ${MIN_SPREAD_MINUTES} recorded minutes at this level: the mean is shown, no per-minute band" role="img" aria-label="too few minutes for a band">&#9676;</span>` : ''}</td>${cell('agg_tok_s')}${cell('per_request')}${cell('ttft_mean')}
      <td class="num">${x.ttft_p10 == null ? '—' : esc(valText({ unit: 's' }, x.ttft_p10, true))}</td>
      <td class="num">${x.ttft_p90 == null ? '—' : esc(valText({ unit: 's' }, x.ttft_p90, true))}</td>
      <td class="num">${x.ttft_p99 == null ? '—' : esc(valText({ unit: 's' }, x.ttft_p99, true))}</td>
      <td class="num">${esc(fmtInt(x.ttft_n))}</td>
      ${cell('prompt_tok_s')}${cell('prefix_hit')}${cell('waiting_mean')}${cell('spec_accept')}${cell('kv_mean', true)}
      ${cell('sm_fill', true)}${cell('tflops', true)}${cell('gbps', true)}</tr>`;
  }));
  const figure = (c) => `<figure class="cmp-chart" data-c="${esc(c.key)}"><h4>${esc(cap1(c.name))}${c.load ? '' : dirHTML(c.key)}</h4></figure>`;
  const lead = CURVE.filter((c) => c.lead), more = CURVE.filter((c) => !c.lead);
  host.innerHTML = `<article class="cmp-panel">
    <header class="cmp-phead"><h3>At each parallel level</h3>
      <span class="sub">${lines.length} configuration${lines.length === 1 ? '' : 's'} &middot; seen at ${esc(seen)} of ${esc(levels.length)} levels</span>
      <span class="scope-tag" hidden></span></header>
    <details class="how" data-keep="curve-how"><summary>how to read this</summary>
    <p class="why">Real traffic runs at every concurrency, and performance depends on it more than on anything
      else &mdash; one band per number of requests the engine was running, one bar per configuration in it, pooled exactly.</p>
    <p class="why">Each bar is every probe pair the gateway read at that level under that configuration, pooled
      exactly (the recorder keeps each level&rsquo;s own sums). Colour is the configuration&rsquo;s
      <code>max_num_seqs</code> on the page&rsquo;s blue ramp; a hatched bar is a second configuration at the same cap
      (the legend draws each one&rsquo;s swatch). An outline bar has fewer than ${MIN_PAIRS} pairs (~10 minutes) behind it,
      or belongs to a configuration with fewer than ${MIN_WINDOWS} minutes in all: shown, not compared. The axis always
      runs 1 through 30 &mdash; an empty band is a level no configuration reached. The y scale is the bars&rsquo;.
      <b>The pale band</b> over each bar is its p10 to p90 and the tick across it their median; the bar&rsquo;s own top
      edge is the mean. The spread is background, so it never sets the axis &mdash; a band that runs past the top is cut
      off there rather than drawn out of it, and the hover says how far it went. <b>What the band is a band OF differs
      by measure, and that difference is the whole of it:</b> on <b>time to first token</b> it is across the
      <b>requests</b>, read off the engine&rsquo;s own latency histogram at that level (hover for the distribution).
      Every other measure is a sum or a time-weighted mean over the level&rsquo;s seconds with no per-request sample
      behind it, so there the band is across the <b>recorded minutes</b> instead &mdash; one figure per minute, and the
      claim is &ldquo;the slowest tenth of those minutes ran below this&rdquo;, never &ldquo;the slowest tenth of the
      requests&rdquo;. A cell with fewer than ${MIN_SPREAD_MINUTES} minutes draws no band, because a tenth of fewer than
      ten minutes is not a whole minute: it keeps its mean, its hover says how many minutes it had, and the table view
      prints that count in <i>n min</i>. The mean and the median part company wherever the slow minutes are the short
      ones &mdash; the mean is seconds-weighted over the whole cell, the median counts each minute once &mdash; and the
      hover says which way it leans.</p></details>
    <div class="cmp-grid lead">${lead.map(figure).join('')}</div>
    <details class="more-charts" data-keep="curve-more"><summary>${more.length} more measures at each level &mdash;
      ${esc(more.map((c) => c.name.replace(/,.*$/, '')).join(', '))}</summary>
      <div class="cmp-grid">${more.map(figure).join('')}</div></details>
    <details class="cmp-table" data-keep="curve-table"><summary>Table view — every configuration at every parallel
      level, each measure&rsquo;s mean over the p10 · median · p90 of its recorded minutes</summary>
      <p class="table-tools"><label class="cols-toggle"><input type="checkbox" id="curve-load-cols"${perf.loadCols ? ' checked' : ''}> load and hardware columns</label>
        <span class="scroll-hint">wider than the window &mdash; scrolls sideways; the first column stays put</span></p>
      <div class="tablewrap"><table class="runs-t twin${perf.loadCols ? '' : ' no-load'}"><thead><tr><th scope="col">configuration</th>
        <th scope="col" class="num">parallel</th><th scope="col" class="num">pairs</th>
        <th scope="col" class="num" title="minutes of traffic recorded at this level">min</th>
        <th scope="col" class="num" title="recorded minutes at this level: the sample each per-minute band is a percentile of">n min</th>
        <th scope="col" class="num">gen tok/s</th><th scope="col" class="num">per request</th><th scope="col" class="num">ttft mean</th>
        <th scope="col" class="num">ttft p10</th><th scope="col" class="num">ttft p90</th><th scope="col" class="num">ttft p99</th><th scope="col" class="num">ttft n</th>
        <th scope="col" class="num">prompt tok/s</th><th scope="col" class="num">prefix hits</th>
        <th scope="col" class="num">waiting</th><th scope="col" class="num">spec accept</th><th scope="col" class="num load">kv</th>
        <th scope="col" class="num load">sm fill</th><th scope="col" class="num load">tflop/s</th><th scope="col" class="num load">gb/s</th></tr></thead>
        <tbody>${rows.join('')}</tbody></table></div></details></article>`;
  const drawIn = (root) => {
    for (const fig of root.querySelectorAll('.cmp-chart')) levelChart(fig, lines, levels, CURVE.find((c) => c.key === fig.dataset.c));
    pmScopeCharts();                             // freshly drawn SVG takes the selection too
  };
  drawIn(host.querySelector('.cmp-grid.lead'));
  // The seven other measures are drawn when their fold OPENS (closed, the figures have no width to draw
  // at). keepingOpen reopens the fold after a rebuild, and setting `open` fires `toggle` too.
  const fold = host.querySelector('details.more-charts');
  fold.addEventListener('toggle', () => { if (fold.open) drawIn(fold); });
  // Four load columns are off by default: eighteen columns clipped the table on every window.
  host.querySelector('#curve-load-cols').addEventListener('change', (ev) => {
    perf.loadCols = ev.target.checked;
    host.querySelector('table.twin').classList.toggle('no-load', !perf.loadCols);
    scrollHint(host.querySelector('details.cmp-table'));
  });
  scrollHint(host.querySelector('details.cmp-table'));
}

/** A table wider than its box scrolls sideways, and says so (a line under the summary) — nothing else tells
 *  a reader that fifteen of eighteen columns are off to the right. Checked when the fold opens and on resize. */
function scrollHint(details) {
  const wrap = details.querySelector('.tablewrap');
  const check = () => details.classList.toggle('overflows', wrap.scrollWidth > wrap.clientWidth + 1);
  details.addEventListener('toggle', check);
  check();
}

/** One measure of every recorded minute, in order. The x axis is BUSY TIME: every recorded minute is
 *  one step and the idle time between two bursts of traffic is cut out, with a break drawn where it was
 *  and each burst labelled with the clock time it started — on a real clock a week of short bursts is a
 *  row of slivers in which no minute can be read. A line joins only consecutive minutes of one
 *  configuration; a lone minute is a dot; a rule marks each change of settings. Series flagged `cap`
 *  take the minute's max_num_seqs ramp step, the rest are ink — context, not identity. */
const GAP_UNITS = 6;
function perfTimeChart(host, ws, o) {
  host.querySelectorAll('svg, .empty, .chart-key').forEach((n) => n.remove());
  const vals = [];
  for (const s of o.series) for (const w of ws) { const v = num(s.get(w)); if (v != null) vals.push(v); }
  if (!vals.length) {
    host.append(Object.assign(document.createElement('p'), { className: 'empty', textContent: 'nothing recorded for this in the range' }));
    return;
  }
  // Positions on the busy-time axis, and where each burst starts.
  const pos = [], starts = [];
  let at = 0;
  ws.forEach((w, i) => {
    if (i === 0 || w.t_start - ws[i - 1].t_end > JOIN_GAP_S) {
      if (i) at += GAP_UNITS;
      starts.push(i);
    }
    pos.push(at + 0.5);
    at += 1;
  });
  const hi = o.yHi != null ? o.yHi : (Math.max(...vals) * 1.12 || 1);
  const f = frame(host, { x0: 0, x1: Math.max(at, 1), yHi: hi, yTitle: o.yTitle, aspect: 0.42, title: o.title,
    yFmt: o.frac ? (v) => `${Math.round(v * 100)}%` : null, padBottom: 36 });
  // A break in the axis wherever idle time was cut out.
  for (const i of starts.slice(1)) {
    const x = f.x(pos[i] - 0.5 - GAP_UNITS / 2);
    f.svg.append(svgEl('line', { class: 'g-gap', x1: x, x2: x, y1: f.pad.t, y2: f.pad.t + f.H }));
  }
  // A change of SETTINGS (not a restart at the same settings) is a rule, labelled when it has room. The
  // label is the app's: a small plate on the rule, turned to the rule's left where it would run off the plot.
  const yBase = f.y(f.yLo), right = f.pad.l + f.W;
  const labels = [];
  let lastLabel = -Infinity;
  for (let i = 1; i < ws.length; i++) {
    if (settingsKey(ws[i]) === settingsKey(ws[i - 1])) continue;
    const x = f.x(pos[i] - 0.5);
    f.svg.append(svgEl('line', { class: 'g-rule', x1: x, x2: x, y1: f.pad.t, y2: f.pad.t + f.H }));
    if (x - lastLabel > 70) {
      const c = cfgOf(ws[i]);
      labels.push({ x, text: `${c.max_num_seqs ?? '?'} seqs` });
      lastLabel = x;
    }
  }
  for (const s of o.series) {
    let d = '', pts = 0, prev = null, col = null, first = null, last = null, key = '', dash = null;
    const flush = () => {
      if (pts === 1) f.svg.append(svgEl('circle', { class: 'g-dot', r: 2.5, cx: first[0], cy: first[1], fill: col, 'data-k': key }));
      else if (pts > 1) {
        // THE APP'S SOFT FILL, under the chart's own series only (a second series is context, and stays a
        // bare line): the same run of minutes closed down to the axis, in the line's colour, faint. It
        // carries the run's data-k like the line does, so a selection fades the two together.
        if (s.cap) {
          f.svg.append(svgEl('path', { class: 'g-area', fill: col, 'data-k': key,
            d: `${d} L${last[0].toFixed(1)},${yBase.toFixed(1)} L${first[0].toFixed(1)},${yBase.toFixed(1)} Z` }));
        }
        f.svg.append(svgEl('path', { class: `g-mark${s.dash ? ' dashed' : ''}`, stroke: col, d, 'data-k': key, 'stroke-dasharray': dash }));
      }
      d = ''; pts = 0;
    };
    ws.forEach((w, i) => {
      const v = num(s.get(w));
      if (v == null) { flush(); prev = null; return; }
      // A configuration series wears the page's one style: its cap's step, and the variant's dashes.
      const st = s.cap ? cfgStyle(cfgOf(w)) : null;
      const c = st ? st.color : s.color;
      const joined = prev && w.t_start - prev.t_end <= JOIN_GAP_S && settingsKey(w) === settingsKey(prev) && c === col;
      const x = f.x(pos[i]), y = f.y(Math.min(v, f.yHi));
      if (!joined) { flush(); col = c; dash = st ? st.dash : (s.dash ? '4 3' : null); key = settingsKey(w); d = `M${x.toFixed(1)},${y.toFixed(1)}`; first = [x, y]; pts = 1; }
      else { d += ` L${x.toFixed(1)},${y.toFixed(1)}`; pts += 1; }
      last = [x, y];
      prev = w;
    });
    flush();
  }
  // The rules' labels, over the lines: a plate keeps one legible where a series runs through it.
  for (const l of labels) {
    const w = l.text.length * 5.6 + 8;
    const flip = l.x + 3 + w > right;
    const x0 = flip ? l.x - 3 - w : l.x + 3;
    f.svg.append(svgEl('rect', { class: 'g-plate', x: x0, y: f.pad.t + 1, width: w, height: 13, rx: 3 }));
    f.svg.append(txt(l.text, { class: 'g-annlab', x: x0 + w / 2, y: f.pad.t + 11, 'text-anchor': 'middle' }));
  }
  // Each burst's start, in the reader's clock, where two labels do not collide. A burst that starts
  // too near the right edge for its label to fit is labelled leftwards from the edge instead — when
  // that clears the label before it — rather than drawn out of the card, over its neighbour.
  const tickW = 68;
  let lastTick = -Infinity;
  for (const i of starts) {
    const x = f.x(pos[i] - 0.5);
    if (x - lastTick < 92) continue;
    const over = x + tickW > f.width;
    if (over && f.width - tickW < lastTick + tickW + 8) continue;
    f.svg.append(txt(when(ws[i].t_start, true), { class: 'g-tick', x: over ? f.width : x, y: f.pad.t + f.H + 14,
      'text-anchor': over ? 'end' : 'start' }));
    lastTick = over ? f.width - tickW : x;
  }
  // The crosshair finds the nearest minute; the tooltip lists every series at it.
  const cross = svgEl('line', { class: 'g-cross', y1: f.pad.t, y2: f.pad.t + f.H, x1: 0, x2: 0, opacity: 0 });
  const hit = svgEl('rect', { class: 'g-hit', x: f.pad.l, y: f.pad.t, width: f.W, height: f.H });
  const xs = pos.map((p) => f.x(p));
  const tipFor = (i) => {
    const w = ws[i], c = cfgOf(w);
    return `<b>${esc(when(w.t_start, true))}\u2013${esc(when(w.t_end))}</b>`
      + `<br><span class="tk">${esc(c.max_num_seqs ?? '?')} seqs &middot; spec ${esc(c.num_speculative_tokens ?? '?')} &middot; batched ${esc(c.max_num_batched_tokens == null ? '?' : fmtInt(c.max_num_batched_tokens))} &middot; gen ${esc(c.generation ?? '?')}</span>`
      + o.series.map((s) => `<br><span class="tk">${esc(s.label)}</span> ${esc(o.fmt(s.get(w)))}`).join('');
  };
  // This chart's points, for a tooltip kept across a rebuild: the minute is named by its start time.
  TIPS.set(`${o.title}|*`, (t) => { const i = ws.findIndex((w) => String(w.t_start) === t); return i < 0 ? null : tipFor(i); });
  const pointAt = (ev) => {
    const box = f.svg.getBoundingClientRect();
    const px = (ev.clientX - box.left) * (f.width / box.width);
    let lo = 0, hiI = xs.length - 1;
    while (hiI - lo > 1) { const m = (lo + hiI) >> 1; if (xs[m] < px) lo = m; else hiI = m; }
    const i = Math.abs(xs[lo] - px) <= Math.abs(xs[hiI] - px) ? lo : hiI;
    cross.setAttribute('x1', xs[i]); cross.setAttribute('x2', xs[i]); cross.setAttribute('opacity', 1);
    showTip(ev, tipFor(i), `${o.title}|${ws[i].t_start}`);
  };
  hit.classList.add('tip-src');
  hit.addEventListener('mousemove', (ev) => { if (!tipTouched) pointAt(ev); });
  hit.addEventListener('mouseleave', () => { if (tipTouched) return; cross.setAttribute('opacity', 0); hideTip(); });
  // A finger reads the same way: a tap puts the crosshair on the nearest minute and the tip stays.
  hit.addEventListener('touchstart', (ev) => { pointAt(ev.touches[0]); tipTouched = true; }, { passive: true });
  f.svg.append(cross, hit);
  f.svg.append(txt('busy minutes in order \u2014 idle time cut out at each break', { class: 'g-atitle',
    x: f.pad.l + f.W / 2, y: f.height - 4, 'text-anchor': 'middle' }));
  host.append(f.svg);
  if (o.series.length > 1) {
    const key = document.createElement('p');
    key.className = 'why chart-key';
    key.innerHTML = o.series.map((s) => `<span class="lg"><svg class="lg-mark" width="18" height="8" aria-hidden="true">`
      + `<line x1="1" x2="17" y1="4" y2="4" stroke="${s.cap ? 'var(--seq-3)' : s.color}" stroke-width="2"${s.dash ? ' stroke-dasharray="4 3"' : ''}></line>`
      + `</svg>${esc(s.label)}</span>`).join(' ');
    host.append(key);
  }
}

function perfTimeCharts(ws) {
  const boxes = [...new Set(ws.flatMap((w) => Object.keys(w.boxes || {})))].sort();
  const boxInk = ['var(--ink-dim)', 'var(--ink-faint)'];
  const box = (b, field) => (w) => ((w.boxes || {})[b] || {})[field];
  const num1 = (unit, digits) => (v) => (num(v) == null ? '—' : `${fmt(v, digits)}${unit}`);
  const pct = (v) => (num(v) == null ? '—' : `${(v * 100).toFixed(1)}%`);
  return [
    { title: 'Generated tokens per second', yTitle: 'tok/s', fmt: num1(' tok/s'),
      series: [{ label: 'generated', get: (w) => w.gen_tok_s, cap: true }] },
    { title: 'Prompt tokens per second', yTitle: 'tok/s', fmt: num1(' tok/s'),
      series: [{ label: 'computed (billed minus prefix-cache hits)', get: (w) => w.prompt_computed_tok_s, cap: true },
        { label: 'billed', get: (w) => w.prompt_tok_s, color: 'var(--ink-faint)', dash: true }] },
    { title: 'Time to first token', yTitle: 'seconds', fmt: num1(' s', 2),
      series: [{ label: 'mean', get: (w) => (w.ttft || {}).mean, cap: true },
        { label: 'p90', get: (w) => (w.ttft || {}).p90, color: 'var(--ink-faint)', dash: true }] },
    { title: 'Inter-token latency', yTitle: 'ms', fmt: num1(' ms', 1),
      series: [{ label: 'mean', get: (w) => ms((w.itl || {}).mean), cap: true },
        { label: 'p90', get: (w) => ms((w.itl || {}).p90), color: 'var(--ink-faint)', dash: true }] },
    { title: 'Requests in the engine', yTitle: 'requests', fmt: num1('', 1),
      series: [{ label: 'running (mean)', get: (w) => (w.running || {}).mean, cap: true },
        { label: 'waiting (mean)', get: (w) => (w.waiting || {}).mean, color: 'var(--ink-faint)', dash: true }] },
    { title: 'KV cache used', yTitle: 'percent', frac: true, yHi: 1, fmt: pct,
      series: [{ label: 'mean', get: (w) => (w.kv || {}).mean, cap: true },
        { label: 'max', get: (w) => (w.kv || {}).max, color: 'var(--ink-faint)', dash: true }] },
    { title: 'Prefix cache hit rate', yTitle: 'percent', frac: true, yHi: 1, fmt: pct,
      series: [{ label: 'hit rate', get: (w) => w.prefix_hit_rate, cap: true }] },
    { title: 'SM fill, each box', yTitle: 'percent', yHi: 100, fmt: num1('%', 0),
      series: boxes.map((b, i) => ({ label: b, get: box(b, 'sm_mean'), color: boxInk[i % 2], dash: i % 2 === 1 })) },
    { title: 'Power, each box', yTitle: 'watts', fmt: num1(' W', 0),
      series: boxes.map((b, i) => ({ label: b, get: box(b, 'pwr_mean'), color: boxInk[i % 2], dash: i % 2 === 1 })) },
    { title: 'Engine compute per GPU', yTitle: 'TFLOP/s', fmt: num1(' TFLOP/s', 1),
      series: [{ label: 'TFLOP/s per GPU', get: (w) => w.tflops_per_gpu, cap: true }] },
    { title: 'Engine memory traffic per GPU', yTitle: 'GB/s', fmt: num1(' GB/s', 0),
      series: [{ label: 'GB/s per GPU', get: (w) => w.gbps_per_gpu, cap: true }] },
  ].filter((c) => c.series.length);
}

/** Everything the performance log draws, rebuilt — with every `<details>` the reader opened kept open and
 *  the tooltip that was showing kept on its point (keepingOpen, retip): the minute tick must never undo
 *  what the reader just did. */
// ══ CONTEXT: what the working set costs ═══════════════════════════════════════════════════════════
// The log already holds this. Every busy minute records KV occupancy (`kv.mean`) and the requests in
// flight (`running.mean`), and the KV pin holds a known number of tokens — so the context the pair was
// carrying needs no extra scrape and no extra load on the engine:
//
//     tokens resident   = kv.mean × kv_pool_tokens   (the minute's OWN pool, recorded with it)
//     context / request = tokens resident ÷ running.mean
//
// It is a LOWER bound on what the requests actually carry: a prefix shared by two requests is held
// once, and a hybrid model's linear-attention state pages share the same pool. It is the shape of
// the curve that is being read here, never a per-request figure.
//: The pool a minute ran under comes FROM THE MINUTE (`kv_pool_tokens`, scraped off the engine's own
//: cache_config_info) and from nowhere else. A minute that recorded none — from before the field existed,
//: or from an engine that exposed no cache_config_info — is NOT BANDED: it is counted (`unpooled`) and the
//: page says so. No pool is assumed for it: the pin is an operator knob, so any constant would re-read
//: minutes run under another pool, and a model id written here to scope one is a deployment fact in code.
const CTX_BANDS = [[0, 8e3], [8e3, 20e3], [20e3, 40e3], [40e3, 80e3], [80e3, Infinity]];

/** One row per context band — of the SELECTED configuration's minutes, or (nothing selected) the model on
 *  show's (renderContext scopes it) — pooled over the minutes in it and weighted by the seconds each covered —
 *  the same pooling the configuration panel uses, so the two can be read against each other. `unpooled`:
 *  the busy minutes that recorded no KV pool of their own, and so are in no band. */
function contextRows(ws) {
  const pts = [];
  let unpooled = 0;
  for (const w of ws) {
    const kv = num((w.kv || {}).mean);
    const run = num((w.running || {}).mean);
    const span = num(w.span_s);
    if (!kv || !run || !span || span <= 0) continue;    // a minute without both is not banded, not zeroed
    const tokens = num(w.kv_pool_tokens);
    if (!tokens) { unpooled += 1; continue; }           // its own pool or none: never another minute's
    pts.push({ ctx: (kv * tokens) / run, span, kv,
               seats: run / (kv || 1),                  // pool ÷ context = how many of THIS size it holds
               gen: (num(w.gen_tokens) || 0) / span,
               pref: (num(w.prompt_computed_tokens) || 0) / span,
               hit: num(w.prefix_hit_rate),
               itl: num((w.itl || {}).mean),
               ttft: num((w.ttft || {}).mean) });
  }
  const rows = [];
  for (const [lo, hi] of CTX_BANDS) {
    const sel = pts.filter((p) => p.ctx >= lo && p.ctx < hi);
    if (!sel.length) continue;
    const total = sel.reduce((a, p) => a + p.span, 0);
    const mean = (k) => {
      const have = sel.filter((p) => num(p[k]) != null);
      const w2 = have.reduce((a, p) => a + p.span, 0);
      return w2 ? have.reduce((a, p) => a + p[k] * p.span, 0) / w2 : null;
    };
    rows.push({ lo, hi, minutes: sel.length, ctx: mean('ctx'), kv: mean('kv'), gen: mean('gen'),
                pref: mean('pref'), hit: mean('hit'), itl: mean('itl'), ttft: mean('ttft'), seats: mean('seats') });
  }
  return { rows, unpooled };
}

function renderContext(modelWs) {
  const host = $('perf-context-panel');
  // SCOPE: a selected square (pmSelect) narrows the bands to THAT configuration's minutes, so its per-context
  // throughput is read apart from the model's other configurations — the operator's "wedged because it is not
  // per context". Nothing selected is the whole model, as it was. A caption names the scope; the band math and
  // the pooling are the same either way (contextRows), so the two scopes read against each other.
  const sel = pm.pinned != null ? pm.groups[pm.pinned] : null;
  const scoped = !!(sel && sel.ws && sel.ws.length);
  const ws = scoped ? sel.ws : (modelWs || []);
  const cap = scoped ? `<p class="why ctx-scope">${esc(cfgText(gcfg(sel)))} &middot; its minutes, banded by context</p>` : '';
  const { rows, unpooled } = contextRows(ws);
  // The minutes no band holds, as figures, and why: unknown is said, never filled in.
  const note = unpooled ? `<p class="why" title="the engine reported no KV pool for these minutes (recorded before the gateway logged it, or an engine without cache_config_info): their context is unknown, so they are in no band">${esc(fmtInt(unpooled))} min &middot; KV pool not recorded &middot; not banded</p>` : '';
  // For the WHOLE MODEL, fewer than two bands is not a curve and a one-row table would invite reading a trend
  // that is not there; for a SELECTED configuration one band is its answer, so it is shown. Minutes whose
  // context is unknown are still said, with the section, rather than hidden with it.
  const floor = scoped ? 1 : 2;
  $('perf-context').hidden = rows.length < floor && !unpooled;
  if (rows.length < floor) { host.innerHTML = cap + note; return; }
  const band = (r) => r.hi === Infinity ? `${fmtInt(r.lo / 1e3)}k+`
    : `${fmtInt(r.lo / 1e3)}k–${fmtInt(r.hi / 1e3)}k`;
  const n1 = (v) => v == null ? '—' : esc(v.toFixed(1));
  const n0 = (v) => v == null ? '—' : esc(fmtInt(Math.round(v)));
  // THE APP'S CONTEXT CARD. On a desk this is the table it was; on a phone each band becomes its own
  // block of the card — the band and its minutes on one line, then a small label over each figure
  // (benchmarks.css, from each cell's data-label: the column's own heading) — where the table had to be
  // scrolled sideways to reach six of its nine columns. The cells, their order and their numbers are the
  // same; the explicit roles keep it a table to a screen reader when CSS lays the rows out as a grid.
  // KV occupancy wears the load scale and the two rates the rate scale, each against the best of the
  // rows shown (figureInk above) — the colour repeats the number printed in the cell and adds nothing.
  const peak = (k) => Math.max(0, ...rows.map((r) => num(r[k]) || 0));
  const genPeak = peak('gen'), prefPeak = peak('pref');
  const ink = (hsl) => ` style="color:${figureInk(hsl)}"`;
  const rate = (v, top) => (v == null || !(top > 0) ? '' : ink(rateHSL(v / top)));
  const td = (label, text, style) => `<td class="num" role="cell" data-label="${label}"${style || ''}>${text}</td>`;
  host.innerHTML = cap + `<div class="tablewrap ctx-wrap"><table class="runs-t twin ctx-t" role="table">
    <thead role="rowgroup"><tr role="row"><th scope="col" role="columnheader">context / request</th><th scope="col" role="columnheader" class="num">minutes</th>
      <th scope="col" role="columnheader" class="num">KV&nbsp;%</th><th scope="col" role="columnheader" class="num">generated tok/s</th>
      <th scope="col" role="columnheader" class="num">prefill tok/s</th><th scope="col" role="columnheader" class="num">prefix hit&nbsp;%</th>
      <th scope="col" role="columnheader" class="num">ITL&nbsp;ms</th><th scope="col" role="columnheader" class="num">TTFT&nbsp;s</th>
      <th scope="col" role="columnheader" class="num">seats</th></tr></thead>
    <tbody role="rowgroup">${rows.map((r) => `<tr role="row"><th scope="row" role="rowheader">${esc(band(r))}
      <span class="pm-sub ctx-mean">${n0(r.ctx)} tokens, mean</span></th>
      <td class="num ctx-min" role="cell" data-label="minutes">${esc(fmtInt(r.minutes))}</td>
      ${td('KV %', n1(r.kv == null ? null : r.kv * 100), r.kv == null ? '' : ink(loadHSL(r.kv)))}
      ${td('generated tok/s', n1(r.gen), rate(r.gen, genPeak))}
      ${td('prefill tok/s', n0(r.pref), rate(r.pref, prefPeak))}
      ${td('prefix hit %', n1(r.hit == null ? null : r.hit * 100))}
      ${td('ITL ms', n0(r.itl == null ? null : r.itl * 1000))}
      ${td('TTFT s', n1(r.ttft))}
      ${td('seats', n1(r.seats))}</tr>`).join('')}</tbody></table></div>` + note;
}

function renderPerf() {
  keepingOpen(document, drawPerf);
  retip();
}

function drawPerf() {
  const d = perf.data;
  $('loading').hidden = true;
  const alert = $('perf-alert');
  const blank = $('perf-blank');
  const rec = (d && d.recorder) || {};
  // ONE MODEL: everything below this line reads `ws`, which holds the model on show only.
  const cur = resolveModel();
  renderModels();
  const ws = d ? ofModel(timed(d.windows)).sort((a, b) => a.t_start - b.t_start) : [];
  perf.ctxWs = ws;                               // the model's minutes, the context table's fallback scope (renderContext)
  // EVERY model's minutes and every cluster start in the range (perflog.BootRecorder), for the matrix alone —
  // the inventory of what was recorded. Only renderPerfMatrix reads these two; no chart, legend, scale or
  // table below is handed anything but `ws`.
  const { ws: allWs, boots: allBoots } = inventory(d);
  $('perf-filters').hidden = !d;
  // The recorder's own trouble, in its own words, above every number.
  const troubles = [];
  if (perf.error) troubles.push(`<b>the performance log could not be read</b> <span class="why">${esc(perf.error)}</span>`);
  if (d && rec.enabled && num(rec.errors)) {
    troubles.push(`<b>${esc(fmtInt(rec.errors))} recorded minute${rec.errors === 1 ? '' : 's'} could not be written</b>`
      + ` <span class="why">${esc(rec.last_error || '')}</span>`);
  }
  if (d && rec.enabled && num(rec.dropped)) {
    troubles.push(`<b>${esc(fmtInt(rec.dropped))} minute${rec.dropped === 1 ? '' : 's'} dropped</b> <span class="why">the writer fell behind; the LB kept serving</span>`);
  }
  alert.hidden = !troubles.length;
  alert.innerHTML = troubles.join('<br>');
  const has = ws.length > 0;
  blank.hidden = has || !d;
  blank.classList.toggle('model-empty', false);
  if (d && !has && cur) {
    // A RECORDED MODEL WHOSE EVERY MINUTE WAS CUT (perflog.MAX_WINDOWS): its label, the minutes the log holds
    // of it, and that none of them is among the newest handed over. Every model offered has minutes.
    blank.classList.add('model-empty');
    blank.innerHTML = `<span class="me-name">${esc(cur.label)}</span><span class="me-fig"><b>${esc(fmtInt(cur.minutes))}</b> min</span>`
      + `<span class="me-sub">${esc(RANGE_TEXT[perf.days] || perf.days)} &middot; every one older than the minutes shown</span>`;
  } else if (d && !has) {
    const why = !d.cluster_configured ? 'This gateway has no cluster configured, so there is nothing to record.'
      : rec.enabled === false ? `The recorder is switched off on this gateway (<code>LB_PERF=0</code>).`
        : 'The gateway writes one line for every minute the cluster serves real requests; an idle minute writes nothing.';
    // NOTHING RECORDED: no model to offer, and the page says so — never a catalog model at 0 min in its place.
    blank.innerHTML = `<p><b>Nothing recorded ${d.days ? `in the last ${esc(perf.days === '1' ? '24 hours' : `${perf.days} days`)}` : 'yet'}.</b></p>
      <p>${why}</p><p class="why">Recorded under <code>${esc(d.dir)}</code> (<code>lb/state/perf</code>).
      ${d.error ? esc(d.error) : ''}</p>`;
  }
  $('perf-caveat').hidden = !has;
  $('perf-cmp').hidden = !has;
  if (!has) {
    $('perf-context').hidden = true;
    // Nothing the last model drew stays in the page, hidden or not: its marks, rows and legend lines go.
    for (const id of ['perf-legend', 'perf-curve', 'perf-cmp-panel', 'perf-context-panel', 'perf-time-key', 'perf-time-grid']) $(id).innerHTML = '';
  }
  // The inventory stands while ANYTHING was recorded — a model with no minute of its own still reads the
  // others' squares, so the page never goes blank on a configuration that exists.
  $('perf-matrix').hidden = !allWs.length && !allBoots.length;
  pmPadBody();
  $('perf-time').hidden = !has;
  const minutes = ws.reduce((a, w) => a + (num(w.span_s) || 0), 0) / 60;
  // THE SUMMARY CARD, in the app's form: a small label over each figure, and the recorder's state at the
  // card's end beside a lit dot. The WORDS are the line this used to be, in the same order — "633 busy
  // minutes · the 12 oldest not shown · recording (… in progress)" is still what the element's text says
  // and what a screen reader reads; benchmarks.css only stands each label over its figure (.stat) and
  // does not draw the separators (.sep) or the "the" (.vh).
  $('p-count').innerHTML = d ? `<span class="stat"><b>${fmtInt(ws.length)}</b> <span>busy minute${ws.length === 1 ? '' : 's'}</span></span>`
    + (cutMinutes() ? `<span class="sep"> &middot; </span><span class="stat"><span class="vh">the </span><b>${esc(fmtInt(cutMinutes()))}</b> <span>oldest not shown</span></span>` : '')
    + '<span class="sep"> &middot; </span>'
    + (rec.enabled ? `<span class="rec on"><i class="dot" aria-hidden="true"></i>recording${rec.open ? ` <span class="rec-open">(${esc(when(rec.open.t_start, true))} in progress)</span>` : ''}</span>`
      : '<span class="rec"><i class="dot" aria-hidden="true"></i>recorder off</span>') : '';
  // THE MATRIX IS THE INVENTORY, so it carries EVERY configuration the gateway recorded — every model's,
  // not only the model the charts below compare. Nothing is pooled by widening it: `groupsOf` keys a group
  // by `settingsKey`, whose first field IS the model, so a minute of one model can never land in a group
  // with a minute of another; renderPerfMatrix draws one captioned grid per model with its own rows and
  // columns; and the shade is `perfTier`, whose scale is keyed per model already. What widens is the
  // inventory, never the comparison.
  if (allWs.length || allBoots.length) renderPerfMatrix(groupsOf(allWs, allBoots));
  else renderPerfMatrix([]);                     // no square, no selection, no card left open from another model
  if (has) {
    renderCurve(ws);
    renderPerfPanel(ws);
    renderContext(ws);
    const grid = $('perf-time-grid');
    grid.innerHTML = '';
    const charts = perfTimeCharts(ws);
    const figs = charts.map((c) => {
      const fig = document.createElement('figure');
      fig.className = 'chart';
      fig.innerHTML = `<h3>${esc(c.title)}</h3>`;
      grid.append(fig);
      return fig;
    });
    charts.forEach((c, i) => perfTimeChart(figs[i], ws, c));
    // The key the time charts' colour needs: every configuration the minutes ran under, as a line in its
    // own style (the cap's step, the variant's dashes) — the same list as the section legend above.
    const byCfg = new Map();
    for (const w of ws) { const k = settingsKey(w); const g = byCfg.get(k) || { key: k, cfg: cfgOf(w), minutes: 0, fallback: false }; g.minutes += 1; g.fallback ||= fallbackCfg(w); byCfg.set(k, g); }
    const cfgLines = [...byCfg.values()].map((g) => ({ ...g, style: cfgStyle(g.cfg) }))
      .sort((a, b) => (num(a.cfg.max_num_seqs) || 0) - (num(b.cfg.max_num_seqs) || 0)
        || (num(a.cfg.num_speculative_tokens) || 0) - (num(b.cfg.num_speculative_tokens) || 0)
        || (num(a.cfg.max_num_batched_tokens) || 0) - (num(b.cfg.max_num_batched_tokens) || 0));
    $('perf-time-key').innerHTML = cfgLegendHTML(cfgLines, { line: true })
      + `<div class="cmp-legend cfg-key"><span class="lg"><svg class="lg-mark" width="18" height="8" aria-hidden="true"><line x1="1" x2="17" y1="4" y2="4" stroke="var(--ink-faint)"`
      + ` stroke-width="2" stroke-dasharray="4 3"></line></svg>ink: a chart&rsquo;s own second series, named under it</span></div>`;
    pmScopeCharts();
  }
  const bad = (d && d.malformed) || [];
  $('perf-malformed').hidden = !bad.length;
  if (bad.length) {
    $('perf-malformed').innerHTML = `<b>${esc(fmtInt(d.malformed_total))} line${d.malformed_total === 1 ? '' : 's'} of the log`
      + ` ${d.malformed_total === 1 ? 'is' : 'are'} not a recorded minute</b> (a write cut short, a hand edit) &mdash; listed so`
      + ` nothing looks silently dropped:<ul>${bad.map((b) => `<li>${esc(b.file)}:${esc(b.line)} &mdash; ${esc(b.error)}`
        + `${b.excerpt ? ` &mdash; <code>${esc(b.excerpt)}</code>` : ''}</li>`).join('')}</ul>`;
  }
}

// ── render ──────────────────────────────────────────────────────────────────

function render() {
  const idx = state.index;
  const alert = $('bench-alert');
  if (idx.error) {
    alert.hidden = false;
    alert.innerHTML = `<b>no saved runs</b> <span class="why">${esc(idx.error)}</span>`;
  } else alert.hidden = true;

  const runs = filtered();
  const ok = runs.filter((r) => r.ok);
  const has = idx.runs.length > 0;
  $('blank').hidden = has;
  if (!has) {
    $('blank').innerHTML = `<p><b>No benchmark runs saved yet.</b></p>
      <p>Runs are read from <code>${esc(idx.dir)}</code> inside the gateway container
         &mdash; <code>lb/state/bench</code>, which the compose file already mounts.</p>
      <p class="why">The harnesses write there themselves: <code>worklog/bench/load100.py --label &lt;name&gt;</code>
         (synthetic) and <code>worklog/bench/load_corpus.py --label &lt;name&gt;</code> (real corpus).
         Each records what the cluster was launched with by asking this gateway, so a saved run states
         its own model and concurrency cap.</p>`;
  }
  $('filters').hidden = !has;
  $('batch').hidden = $('compare2').hidden = !has;
  $('runs').hidden = !has;
  if (has) renderFilters();

  $('f-count').innerHTML = `<b>${ok.length}</b> of ${idx.runs.length} run${idx.runs.length === 1 ? '' : 's'}`
    + (runs.length - ok.length ? ` &middot; ${runs.length - ok.length} unreadable` : '');

  if (has) {
    renderCompare(runs);
    renderSpread(ok);
  }
  renderTable(runs);

  const skipped = idx.skipped || [];
  $('skipped').hidden = !skipped.length;
  if (skipped.length) {
    $('skipped').innerHTML = `<b>${skipped.length} file${skipped.length === 1 ? '' : 's'} in ${esc(idx.dir)} `
      + `${skipped.length === 1 ? 'is' : 'are'} not a benchmark result</b> (a corpus input, another tool's output) `
      + `&mdash; listed so nothing looks silently dropped:<ul>`
      + skipped.map((s) => `<li>${esc(s.file)} &mdash; ${esc(s.error)}</li>`).join('') + '</ul>';
  }
}

// ── loading: the performance log and the saved runs, both before anything is drawn ──────────────

/** Every palette domain — caps, spec values, batched values — over BOTH sources and over everything
 *  loaded, never a filtered view: the two sections must never paint one cap two ways. */
function recomputeDomains() {
  const runs = state.index ? state.index.runs : [];
  const wins = perf.data ? (perf.data.windows || []) : [];
  const values = (fromRun, fromWin) => [...new Set([...runs.map(fromRun), ...wins.map(fromWin)]
    .map(num).filter((v) => v != null))].sort((a, b) => a - b);
  state.caps = values((r) => r.max_num_seqs, (w) => cfgOf(w).max_num_seqs);
  state.specs = values((r) => r.spec, (w) => cfgOf(w).num_speculative_tokens);
  state.batched = values((r) => r.batched, (w) => cfgOf(w).max_num_batched_tokens);
  // The variants at each cap, over every minute AND every boot loaded (a configuration that only booted
  // has a square in the matrix and needs its style too), in spec-then-batched order — so the same
  // configuration is solid or hatched the same way whatever the range shows.
  // ...of the MODEL ON SHOW alone: another model's configurations are not on the page, so they take no
  // rank here — a model's first configuration at a cap is solid whatever a second model ran.
  const boots = perf.data ? (perf.data.boots || []) : [];
  resolveModel();
  const seen = new Map();
  for (const w of ofModel([...wins, ...boots])) {
    const c = cfgOf(w), cap = num(c.max_num_seqs);
    if (cap == null) continue;
    (seen.get(cap) || seen.set(cap, new Set()).get(cap)).add(variantKey(c));
  }
  perf.variants = new Map([...seen.entries()].map(([cap, set]) => [cap, [...set].sort(cmpTokens)]));
  // What each model's labels must name: the parts that differ among ITS recorded configurations (the
  // matrix's records — every model's, a model that only booted none).
  const inv = inventory(perf.data), byModel = new Map();
  for (const r of [...inv.ws, ...inv.boots]) {
    const c = cfgOf(r);
    (byModel.get(c.model_choice) || byModel.set(c.model_choice, []).get(c.model_choice)).push(c);
  }
  perf.varying = new Map([...byModel].map(([m, cs]) => [m, varyingParts(cs)]));
}

function setStatus() {
  const conn = $('connection-status');
  if (!perf.data && !state.index) {
    conn.className = 'conn bad';
    $('status-text').textContent = `gateway unreachable — ${perf.error || state.benchError || 'no answer'}`;
    return;
  }
  conn.className = perf.error ? 'conn bad' : 'conn ok';
  const n = perf.current ? perf.current.minutes : 0;   // the model on show's, like every figure below it
  const runs = state.index ? state.index.runs.length : 0;
  $('status-text').textContent = [perf.data ? `${n} busy minute${n === 1 ? '' : 's'}` : 'performance log unreadable',
    runs ? `${runs} saved run${runs === 1 ? '' : 's'}` : ''].filter(Boolean).join(' \u00b7 ');
}

async function fetchBench() {
  try {
    const res = await fetch('/lb/bench', { cache: 'no-store' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    state.index = await res.json();
    state.benchError = '';
  } catch (err) {
    state.index = null;
    state.benchError = String(err && err.message || err);
  }
}

async function fetchPerf() {
  try {
    const res = await fetch(`/lb/perf?days=${encodeURIComponent(perf.days)}${serverQuery()}`, { cache: 'no-store' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    perf.data = await res.json();
    perf.error = '';
  } catch (err) {
    perf.error = String(err && err.message || err);   // the last good read stays on screen
  }
}

/** The archive: shown only while saved runs exist. The operator cleared them deliberately, so an
 *  empty directory is not a state to explain — the section simply is not there. */
function renderArchive() {
  const runs = state.index ? state.index.runs : [];
  $('archive').hidden = !runs.length;
  if (!runs.length) return;
  state.heat = buildHeat(runs);
  renderHeatKey(state.heat);
  render();
}

/* `?server=<name>`: server-scope.js (one helper, shared with the sessions page) — the log's minutes are that server's. */
async function boot() {
  serverScope('Performance');
  await Promise.all([fetchPerf(), fetchBench()]);
  recomputeDomains();
  renderPerf();
  renderArchive();
  setStatus();
  const want = new URLSearchParams(location.search).get('run');
  if (want && state.index && state.index.runs.some((r) => r.id === want)) select(want);
  // A new window lands once a minute; read it then. The frame is kept: charts redraw in place.
  perf.timer = setInterval(refreshPerf, 60000);
}

async function refreshPerf() {
  const before = JSON.stringify([state.caps, state.specs, state.batched]);
  await fetchPerf();
  recomputeDomains();
  renderPerf();                                  // keeps every open table view and the showing tooltip
  if (JSON.stringify([state.caps, state.specs, state.batched]) !== before) { renderArchive(); renderDetail(); }
  setStatus();
}

$('p-days').addEventListener('change', (ev) => { perf.days = ev.target.value; refreshPerf(); });
/** Another model: its own configurations, so no square of the last one stays selected or read out. With
 *  `selectKey`, the configuration to select once the page is redrawn — how a click on a square in another
 *  model's matrix grid lands on that square with the charts already behind it. Nothing is fetched: the
 *  range already holds every model's minutes, the split is made in the page. */
function showModel(id, selectKey) {
  perf.model = id;
  storeModel(perf.model);
  pm.pinned = null; pm.open = false; pm.returnTo = null;
  hideTip();
  recomputeDomains();
  renderPerf();
  setStatus();
  if (selectKey == null) return;
  const i = pm.groups.findIndex((g) => groupKey(g) === selectKey);
  if (i >= 0) pmSelect(i);
}
$('p-models').addEventListener('change', (ev) => showModel(ev.target.value));

for (const [id, key] of [['f-model', 'model'], ['f-workload', 'workload'], ['f-conc', 'conc'],
  ['f-prompt', 'prompt'], ['f-out', 'out']]) {
  $(id).addEventListener('change', (ev) => { state.filters[key] = ev.target.value; render(); });
}
$('f-reset').addEventListener('click', () => {
  state.filters = { model: '', workload: '', seqs: new Set(), conc: '', prompt: '', out: '' };
  for (const id of ['f-model', 'f-workload', 'f-conc', 'f-prompt', 'f-out']) $(id).value = '';
  render();
});

// The SVGs are built at real pixel WIDTH, so a resize that changes it has to rebuild them. One that changes
// only the height — a phone's URL bar showing or hiding, which happens on every scroll — changes nothing a
// chart is built from, and a rebuild would fold every open table view: it is ignored.
let resizeTimer = null, drawnWidth = window.innerWidth;
window.addEventListener('resize', () => {
  if (window.innerWidth === drawnWidth) return;
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    drawnWidth = window.innerWidth;
    renderPerf();
    if (state.index && state.index.runs.length) { render(); renderDetail(); }
    retip();
  }, 150);
});

boot();
