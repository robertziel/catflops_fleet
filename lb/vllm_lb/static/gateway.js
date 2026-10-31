/* The gateway view: what the fleet is doing.
 *
 * Served BY the load balancer, so every call is same-origin: /lb/status is this process's own route.
 * The authenticated nginx vhost on the edge is what stands between this page and the internet — the LB has
 * no auth of its own.
 *
 * THE PAGE READS LIKE THE FLEET APP (FleetStats/App/FleetStatsApp.swift), in the app's order:
 *
 *   1. the cluster, ONCE — one card: the state and how long it has been that, the detail, the start
 *      (or stop) sequence with the current step marked, that step's own facts, the model, ONE
 *      press-and-hold action for the state, and the test swarm as a row (`renderCluster`);
 *   2. each LIVE server's throughput — generated, prompt and queue side by side, memory bandwidth and
 *      compute under them, then the engine's figures (`serverCard`);
 *   3. each MACHINE — its cores, the CPU · GPU · MEM rings with the memory figures, then the three
 *      histories (`renderMachines`). The power control (/power/button.js, injected by the vhost) lands
 *      on these cards: they are `#lb-backends article.card`, named by `.box-name`.
 *
 * What the page can do and the app cannot — the engine settings and their estimate, the saturation's
 * evidence, the planner's view of each GB10 box — is all still here, under the app's three parts.
 *
 * WHAT IS WRITTEN: figures, graphs, badges, state words and the server's own refusals. Never a sentence
 * that explains what a figure means or restates what a badge says (operator, 2026-10-01: "Be compact —
 * only data matters"). A fact worth keeping that is not a figure goes in a `title`, where it costs no row.
 */
'use strict';

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s == null ? '' : s).replace(/[&<>"']/g, (c) =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
/** A fragment of this page's markup as the words it shows — tags dropped, the entities it writes read back — for the
 *  `title` of a one-line slot that cuts it (gateway.css: every graph cell is one fixed height, its text one line). */
const plain = (html) => String(html == null ? '' : html).replace(/<[^>]*>/g, '').replace(/&middot;/g, '·')
  .replace(/&nbsp;/g, ' ').replace(/&ge;/g, '≥').replace(/&deg;/g, '°').replace(/&lt;/g, '<').replace(/&gt;/g, '>')
  .replace(/&quot;/g, '"').replace(/&#39;/g, "'").replace(/&amp;/g, '&').replace(/\s+/g, ' ').trim();
/** ` title="…"` for a one-line slot holding `html`, or nothing when it holds nothing. */
const cutTitle = (html) => { const t = plain(html); return t ? ` title="${esc(t)}"` : ''; };

const POLL_MS = 3000;
/** A status poll that has not answered in this long is abandoned; `poll` never starts a second. */
const POLL_TIMEOUT_MS = 5000;
/** No success for this long reads as stale — whether or not any request has settled. */
const STALE_AFTER_MS = 2 * POLL_MS;

/** A refusal in the LB's own words: "HTTP 502 — no live backend could be reached …". Every error
 *  the LB answers with is `{"error": {"type", "message"}}`, and the message is the part that names
 *  the cause; a bare "HTTP 502" sent the operator to the logs for it. A body that is not JSON (the
 *  vhost in front, answering for a dead LB) leaves the status on its own. */
async function refusal(r) {
  let why = '';
  try {
    const e = (await r.json()).error;
    why = typeof e === 'string' ? e : ((e || {}).message || (e || {}).type || '');
  } catch (_) { /* not JSON */ }
  return `HTTP ${r.status}${why ? ` — ${why}` : ''}`;
}

// ── the tile: history, then the number, then what qualifies it ───────────────
// The operator's grammar, drawn the way the Fleet app draws it (FleetStats/Shared/Views.swift
// `Sparkline`, `GraphTile`). One slot is one sample on a fixed time axis, newest on the right; the
// line's height IS the value, and its colour is the value's own on the one scale below — the stroke
// is a gradient with a stop per sample, so the colour follows the value along the time axis. A faint
// fill of the same colours sits under it, and the headline wears the current value's colour.
//
// History comes from /lb/status, never from this page: it used to be accumulated in a local Map
// and every reload erased the minute before the tab was opened.
const SPARKLINE_COLS = 18;

const isNum = (v) => typeof v === 'number' && Number.isFinite(v);

/** THE ONE COLOUR SCALE every figure uses — the app's `scaleColor`: blue near zero, green, yellow,
 *  red at the top. `t` is 0…1: a percentage over 100, a rate over its own series' peak. It says HOW LOADED: red is a
 *  pinned GPU, a full queue. A figure where HIGHER IS BETTER — tok/s — is not read on it: see `rateColor`.
 *  The stops are the ACTIVE STYLE's `--scale-load` token (gateway.css; palette.js does the arithmetic): near zero a
 *  pale, half-transparent cyan — idle reads as quiet, not as a dark blue mass — filling in to the solid green by
 *  12 %, yellow at 60 %, red at 100 % in "Vibe coded". */
const scaleColor = (t) => palette.css(palette.hsl('--scale-load', t));

/** THE SCALE FOR A RATE (generated / prompt / prefill tok/s, GB/s, TFLOPS): the blue family — a dim
 *  steel-blue at rest, a bright cyan at its TARGET — so a throughput is never read as "hot". `t` is
 *  the rate over its target (the best minute recorded at this cluster setup,
 *  `/lb/status.throughput_target`; a ceiling's peak for a share) or, with none, over the graph's own
 *  peak. Grey under 40 % of the target, blue around 40 %, a glowing bright blue from 80 % up — the eye finds the
 *  minutes that reached the target and ignores the rest. The style's `--scale-rate` token. */
const rateColor = (t) => palette.css(palette.hsl('--scale-rate', t));

/** A rate at this share of its target or better GLOWS: its figure and line get a halo of their colour — in a style
 *  that glows (its --glow token: 1 in "Vibe coded"; a style without halos sets 0 and nothing glows). */
const RATE_GLOW_FROM = 0.8;
const styleGlows = () => Number(palette.token('--glow')) > 0;

/** THE SCALE FOR A QUEUE, against the engine's CONCURRENCY (max_num_seqs): red when nearly empty (the
 *  pair is starved), green when about as deep as the concurrency, red again at three times it (work
 *  piles up in the waiting line). Without a concurrency to read against, a queue is a load. The style's
 *  `--scale-queue` token, its t in multiples of the concurrency. */
const queueColor = (depth, concurrency) => palette.css(palette.hsl('--scale-queue', depth / concurrency));

/** The window a graph draws — the last `cols` samples, right-aligned, so a short history sits at the
 *  newest end instead of being stretched across the axis — and what its colours are measured against
 *  (the app's `max == 100 ? loadColor : rateColor`): the metric's own ceiling when it has one (a
 *  percent's 100), else the window's peak. */
function windowOf(series, o) {
  const raw = o.cols ? (series || []).slice(-o.cols) : (series || []);
  const cells = o.cols ? new Array(o.cols - raw.length).fill(null).concat(raw) : raw.slice();
  const nums = cells.filter(isNum);
  const peak = nums.length ? Math.max(...nums) : 0;
  // `kind: 'rate'` reads the value against its target (else its own peak) on the rate scale;
  // `kind: 'queue'` with a `concurrency` reads it on the queue scale; everything else is a load.
  if (o.kind === 'queue' && o.concurrency > 0) {
    // A queue never glows — only a rate at its target does. `glows` is still a function here: the
    // headline asks every window, and a window without one threw, which took the whole card list down.
    return { cells, nums, peak, colourOf: (v) => queueColor(v, o.concurrency), glows: () => false };
  }
  const ceil = (o.kind === 'rate' && o.target) || o.max || peak;
  const scale = o.kind === 'rate' ? rateColor : scaleColor;
  return { cells, nums, peak, colourOf: (v) => scale(ceil > 0 ? v / ceil : 0),
           glows: (v) => o.kind === 'rate' && ceil > 0 && v / ceil >= RATE_GLOW_FROM && styleGlows() };
}

/** The gradient's id is a function of what it paints (and whose graph it is), so an unchanged poll
 *  writes unchanged markup — `paint` then leaves a live region alone — and two identical graphs share
 *  one identical definition. */
function hashOf(s) {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619);
  return (h >>> 0).toString(36);
}

const SPARKLINE_W = 100, SPARKLINE_H = 24;

/** The graph. `max` fixes the height's ceiling (100 for a percent; a value over it still fits);
 *  without one the window's peak is the top. A gap is a sample we could NOT take (tok/s across a
 *  restart, a box gone quiet): the line BREAKS there — drawing it as zero would claim we measured
 *  idleness — and a sample with a gap on both sides is a dot, so no measurement goes undrawn.
 *  `data-samples` spells the axis: 1 a sample, 0 a gap, one per slot. */
function sparkline(series, opts) {
  const o = opts || {};
  const w = windowOf(series, o);
  const n = w.cells.length;
  // The rule a graph is read against — a rate's target, a queue's concurrency — is on the axis, so the
  // axis is at least that tall.
  const rule = isNum(o.target) && o.target > 0 ? o.target : (o.kind === 'queue' && o.concurrency > 0 ? o.concurrency : null);
  const top = Math.max(o.max || 0, rule || 0, w.peak, 1);
  const x = (i) => (n > 1 ? (i * SPARKLINE_W) / (n - 1) : SPARKLINE_W);
  const y = (v) => 1 + (SPARKLINE_H - 2) * (1 - Math.min(Math.max(v, 0), top) / top);   // 0 sits on the base
  const pt = (i) => `${x(i).toFixed(2)} ${y(w.cells[i]).toFixed(2)}`;
  let line = '', fill = '', lone = '';
  for (let i = 0; i < n; i++) {
    if (!isNum(w.cells[i]) || isNum(w.cells[i - 1])) continue;          // i opens a run of samples
    let j = i;
    while (isNum(w.cells[j + 1])) j++;
    if (j === i) { lone += `M${pt(i)}h0`; continue; }
    const run = [];
    for (let k = i; k <= j; k++) run.push(pt(k));
    line += `M${run.join('L')}`;
    fill += `M${x(i).toFixed(2)} ${SPARKLINE_H}L${run.join('L')}L${x(j).toFixed(2)} ${SPARKLINE_H}Z`;
  }
  // THE SECOND, QUIETER LINE: the same measure PER RUNNING REQUEST (`o.under`, perRequestOf). ONE
  // AXIS — it is the same quantity in the same unit, and it is never above the headline's line while
  // more than one request runs, so it reads as what it is: the share of that total one request was
  // getting. (The KV tier's STORE under its reload, `kvRamTile`, is the same unit too; it can top the reload, so
  // that tile passes `max` — the axis then holds both.) No colour and no fill: colour on this card means "how close
  // to the target" and a second coloured line would claim a second reading of that. It is drawn FIRST, so the
  // headline's line is never covered by it.
  const under = o.under ? windowOf(o.under, { cols: o.cols }) : null;
  let quiet = '';
  if (under) {
    const uy = (v) => 1 + (SPARKLINE_H - 2) * (1 - Math.min(Math.max(v, 0), top) / top);
    for (let i = 0; i < under.cells.length; i++) {
      if (!isNum(under.cells[i]) || isNum(under.cells[i - 1])) continue;
      let j = i;
      while (isNum(under.cells[j + 1])) j++;
      const run = [];
      for (let k = i; k <= j; k++) run.push(`${x(k).toFixed(2)} ${uy(under.cells[k]).toFixed(2)}`);
      quiet += j === i ? `M${run[0]}h0` : `M${run.join('L')}`;
    }
  }
  // One stop per slot, each sample's own colour; a gap carries the colour before it (nothing is
  // drawn there, so it only keeps the gradient from blending across the hole). A stop between two
  // of its own colour changes nothing and is left out.
  let last = w.nums.length ? w.colourOf(w.nums[0]) : scaleColor(0);
  const colours = w.cells.map((v) => (isNum(v) ? (last = w.colourOf(v)) : last));
  const stops = colours.map((c, i) => (c === colours[i - 1] && c === colours[i + 1] ? ''
    : `<stop offset="${(n > 1 ? i / (n - 1) : 0).toFixed(4)}" stop-color="${c}"/>`)).join('');
  const id = `sg${hashOf(`${o.key || ''}|${stops}`)}`;
  const ink = `url(#${id})`;
  const samples = w.cells.map((v) => (isNum(v) ? '1' : '0')).join('');
  const newest = w.nums[w.nums.length - 1];
  const glow = isNum(newest) && w.glows(newest);
  return `<svg class="sparkline${o.cls ? ` ${o.cls}` : ''}${glow ? ' glow' : ''}" viewBox="0 0 ${SPARKLINE_W} ${SPARKLINE_H}" preserveAspectRatio="none"
    aria-hidden="true" data-samples="${samples}">`
    + (w.nums.length ? `<defs><linearGradient id="${id}" gradientUnits="userSpaceOnUse" x1="0" y1="0" x2="${SPARKLINE_W}" y2="0">${stops}</linearGradient></defs>` : '')
    + `<line class="sparkline-base" x1="0" y1="${SPARKLINE_H - 0.25}" x2="${SPARKLINE_W}" y2="${SPARKLINE_H - 0.25}" vector-effect="non-scaling-stroke"/>`
    // THE TARGET, a dashed rule at its own height: what this setup has reached, to be read against.
    + (rule != null ? `<line class="sparkline-target" x1="0" y1="${y(rule).toFixed(2)}" x2="${SPARKLINE_W}" y2="${y(rule).toFixed(2)}" vector-effect="non-scaling-stroke"/>` : '')
    + (fill ? `<path d="${fill}" fill="${ink}" fill-opacity="${palette.token('--area-opacity')}"/>` : '')
    + (quiet ? `<path class="sparkline-per-req" d="${quiet}" fill="none" stroke-width="1"
         stroke-linejoin="round" stroke-linecap="round" vector-effect="non-scaling-stroke"/>` : '')
    + (line ? `<path d="${line}" fill="none" stroke="${ink}" stroke-width="1.5" stroke-linejoin="round" stroke-linecap="round" vector-effect="non-scaling-stroke"/>` : '')
    + (lone ? `<path d="${lone}" fill="none" stroke="${ink}" stroke-width="3" stroke-linecap="round" vector-effect="non-scaling-stroke"/>` : '')
    + '</svg>';
}

/** The colour a headline wears: its CURRENT value's on its graph's scale — the app's `GraphTile`.
 *  `now` is the number the headline shows (else the newest sample). None when there is no reading, or
 *  no scale to read it on (no ceiling and no sample to take a peak from). */
function nowColour(now, series, opts) {
  const o = opts || {};
  const w = windowOf(series, o);
  const v = isNum(now) ? now : w.nums[w.nums.length - 1];
  return isNum(v) && (o.max || o.target || o.concurrency || w.nums.length) ? w.colourOf(v) : null;
}

/** Whether the headline glows: a rate at RATE_GLOW_FROM of its target or better. */
function nowGlows(now, series, opts) {
  const o = opts || {};
  const w = windowOf(series, o);
  const v = isNum(now) ? now : w.nums[w.nums.length - 1];
  return isNum(v) && w.glows(v);
}

/** tok/s as a REFERENCE prints it — the app's `tokText`: 71.6 → "72", 3813 → "3.8k". Only the quiet
 *  label on a graph uses it; a headline goes through `fmtNum`, which keeps the exact value in `title`. */
const tokText = (v) => (v >= 1000 ? `${(v / 1000).toFixed(1)}k` : String(Math.round(v)));

/** THE REFERENCE LABEL of a graph — "target 197" for a rate with a target, else "max 43" — and what its
 *  placement needs: the level it names and every sample's height, as fractions down from the top of the
 *  axis. `placeMarks` puts it where the line is not; here it is only described. None for an empty graph. */
function markOf(series, o) {
  const w = windowOf(series, o);
  const goal = o.kind === 'rate' && isNum(o.target) && o.target > 0 ? o.target : null;
  if (!(w.peak > 0) && goal == null) return null;
  const rule = goal != null ? goal : (o.kind === 'queue' && o.concurrency > 0 ? o.concurrency : null);
  const top = Math.max(o.max || 0, rule || 0, w.peak, 1);
  const down = (v) => (1 - Math.min(Math.max(v, 0), top) / top).toFixed(3);
  // `markText`: a graph in another unit than tokens (the KV tier's bytes a second) names its peak in that unit.
  const peak = o.markText ? o.markText(w.peak) : o.max === 100 ? `${Math.round(w.peak)}%`
    : (o.kind === 'rate' ? tokText(w.peak) : String(Math.round(w.peak)));
  return { text: goal != null ? `target ${tokText(goal)}` : `max ${peak}`, level: down(goal != null ? goal : w.peak),
           ys: w.cells.map((v) => (isNum(v) ? down(v) : '')).join(','), ruled: goal != null };
}

/** A graph with its reference label. The label is positioned after layout (`placeMarks`), because where
 *  the line leaves room depends on the graph's real width. */
function graph(series, o) {
  const m = markOf(series, o);
  return `<div class="t-graph"${m ? ` data-level="${m.level}" data-ys="${m.ys}"${m.ruled ? ' data-ruled="1"' : ''}` : ''}>`
    + sparkline(series, o) + (m ? `<span class="t-mark" aria-hidden="true">${esc(m.text)}</span>` : '') + '</div>';
}

/** THE LABEL KEEPS OFF THE LINE — the app's `GraphTile.labelSpot`, in the graph's own pixels. It sits just
 *  under the level it names, at the trailing end; if the line runs through that spot it moves to the
 *  leading end, then above the level, then to the graph's far edges; and where no spot is clear of the
 *  line (a short graph with a busy line) it is not drawn at all. With a dashed target rule on the graph,
 *  an edge spot that would sit on the rule is skipped too. The line is never covered. */
function placeMarks(root) {
  (root || document).querySelectorAll('.t-graph[data-ys]').forEach((g) => {
    const mark = g.querySelector('.t-mark');
    if (!mark) return;
    const W = g.clientWidth, H = g.clientHeight;
    const ys = g.dataset.ys.split(',').map((v) => (v === '' ? null : Number(v)));
    const n = ys.length;
    // The graph's own mapping (`sparkline`): the axis runs from 1 to SPARKLINE_H - 1 of a SPARKLINE_H-tall box.
    const px = (f) => (H * (1 + (SPARKLINE_H - 2) * f)) / SPARKLINE_H;
    const half = mark.offsetHeight / 2, clearance = 1.5, width = Math.min(mark.offsetWidth, W);
    let spot = null;
    if (W > 0 && n > 1 && H >= 2 * half + 2) {
      const step = W / (n - 1);
      const extent = (trailing) => {
        const from = trailing ? W - width - step : -step, to = trailing ? W + step : width + step;
        let lo = Infinity, hi = -Infinity;
        ys.forEach((f, i) => {
          if (f == null || i * step < from || i * step > to) return;
          lo = Math.min(lo, px(f)); hi = Math.max(hi, px(f));
        });
        return lo === Infinity ? null : [lo - clearance, hi + clearance];
      };
      const ends = [true, false].map((trailing) => ({ trailing, line: extent(trailing) }));
      const markY = px(Number(g.dataset.level));
      const ruled = g.dataset.ruled === '1';
      const levels = [markY + half + 0.5, markY - half - 0.5, half, H - half];
      for (let k = 0; k < levels.length && !spot; k++) {
        const y = levels[k];
        if (y < half || y > H - half) continue;
        if (k >= 2 && ruled && Math.abs(y - markY) < half + 0.5) continue;
        const end = ends.find((e) => !(e.line && e.line[0] <= y + half && e.line[1] >= y - half));
        if (end) spot = { y, trailing: end.trailing };
      }
    }
    mark.classList.toggle('placed', !!spot);
    if (!spot) return;
    mark.style.top = `${(spot.y - half).toFixed(1)}px`;
    mark.style.left = spot.trailing ? 'auto' : '0';
    mark.style.right = spot.trailing ? '0' : 'auto';
  });
}

/* THE SUB-LINE'S GRAMMAR, and it is load-bearing: " + " joins the PARTS OF THE HEADLINE, which
   must add to it; " &middot; " separates CONTEXT, which is a different noun and never adds. A tile
   that mixed the two read "25" over "0 waiting &middot; 6 running &middot; 25 ours" and the operator
   added it up, as anyone would.

   THE TILE IS THE APP'S `GraphTile`: the label and the figure on one line, the graph under them with its
   quiet max/target label. Under the graph, in the tile's own column, only the reason there is no figure
   or a state with numbers (a prefill under way). Every other small figure — the draft rate, the cached
   share, the prefix hit, the prefill speed, running + waiting — is a label·value pair in ONE wrapping
   row under the whole grid (`engineStats`; operator, 2026-10-01: "wrap to labels listed one next to each other"). It
   used to be a row per tile under the grid, headed by the tile's label again, with totals, targets in
   words and whose tokens they were — noise over the figures.

   ONE FIXED HEIGHT, WHATEVER THE DATA (operator 2026-10-30: "ensure each graph cell is compact and doesn't change
   height over data change"): the tile is always the same three slots — head, graph, note — and the NOTE SLOT IS
   DRAWN EMPTY when `sub` is null, so "2 running · 4 queued" appearing under QUEUE no longer grows the row and its
   leaving no longer shrinks it. The note is one line (gateway.css), its whole text in its `title` and in the DOM.
   `o.noNote`: a cell that never has a subline BY ITS KIND (a machine card's CPU / GPU / MEM graphs) draws no slot —
   decided by the kind, never by the data. */
function tile(label, value, unit, sub, series, opts) {
  const o = Object.assign({ cols: SPARKLINE_COLS, key: label }, opts);
  const blank = value == null || value === '';
  // `title`: the exact value behind a rounded one; `dim`: a reading that has gone old — faint, so
  // it wears no colour. `now`: the number behind a formatted headline, for its colour.
  const title = o.title ? ` title="${esc(o.title)}"` : '';
  const colour = blank || o.dim ? null : nowColour(isNum(value) ? value : o.now, series, o);
  const glow = !!colour && nowGlows(isNum(value) ? value : o.now, series, o);
  const note = o.noNote ? '' : `<div class="t-note"${cutTitle(sub)}>${sub ? `<span class="t-sub">${sub}</span>` : ''}</div>`;
  return `<div class="tile${blank || o.dim ? ' dim' : ''}${glow ? ' glow' : ''}">
    <div class="t-head" title="${esc(label)}"><span class="t-label">${esc(label)}</span>
    <b class="t-val"${title}${colour ? ` style="color:${colour}${glow ? `;text-shadow:0 0 8px ${colour}` : ''}"` : ''}>${blank ? '—' : esc(value)}${blank ? '' : `<span class="t-unit">${esc(unit || '')}</span>`}</b></div>
    ${graph(series, o)}
    ${note}
  </div>`;
}

const gb = (mb) => (mb == null ? null : (mb / 1024).toFixed(mb >= 10240 ? 0 : 1));

// ── throughput: measured, or "—" with the reason (worklog/vllm-gateway-throughput.md) ──────────
// null means NOT MEASURED and is drawn "—" with its `why`; a number, 0 included, is a measurement.
// The page never turns one into the other, and every age is `s.now` minus a time on the LB's clock —
// never Date.now(), which is the browser's.

/** THE number formatter (§7): what a reading shows, and the exact value for `title` when the shown
 *  one is rounded or refused. null, NaN and ±Infinity are not measured: `text` is null. */
function fmtNum(v) {
  if (typeof v !== 'number' || !Number.isFinite(v)) return { text: null, title: '' };
  if (v === 0) return { text: '0', title: '' };                   // -0 as well
  if (v < 0) return { text: '?', title: String(v) };               // no count or rate here is negative
  if (v < 0.1) return { text: '<0.1', title: String(v) };
  if (v < 100) return { text: v.toFixed(1), title: '' };
  if (v < 10000) return { text: Math.round(v).toLocaleString('en-US'), title: '' };
  const exact = Math.round(v).toLocaleString('en-US');
  for (const [suffix, scale] of [['k', 1e3], ['M', 1e6], ['B', 1e9]]) {
    const shown = (v / scale).toPrecision(3);                     // three significant digits
    if (Number(shown) < 1000 || suffix === 'B') return { text: `${shown}${suffix}`, title: exact };
  }
  return { text: null, title: '' };
}

/** A share from 0 to 1 as a percent; outside [0, 1] it is `?`, with the raw value in `title`. */
function fmtShare(v) {
  if (typeof v !== 'number' || !Number.isFinite(v)) return { text: null, title: '' };
  if (v < 0 || v > 1) return { text: '?', title: String(v) };
  return { text: `${Math.round(v * 100)} %`, title: '' };
}

/** Seconds as "40 s", "2m 14s" or "1h 2m". */
function fmtDuration(sec) {
  if (typeof sec !== 'number' || !Number.isFinite(sec)) return null;
  const n = Math.max(0, Math.round(sec));
  if (n < 100) return `${n} s`;
  if (n < 3600) return `${Math.floor(n / 60)}m ${n % 60}s`;
  return `${Math.floor(n / 3600)}h ${Math.floor((n % 3600) / 60)}m`;
}

/** Samples on a TIME axis, newest last, one column per probe: a hole longer than the probes' own
 *  period becomes that many gap columns, so a stalled tick shows as a hole rather than squeezing
 *  time together. The period is the samples' median spacing. `points` are [at, value]. */
function timeline(s, points) {
  if (!points.length) return [];
  const spacings = points.slice(1).map(([at], i) => at - points[i][0]).filter((d) => d > 0)
    .sort((a, b) => a - b);
  const period = spacings.length ? spacings[Math.floor(spacings.length / 2)]
    : ((s.config || {}).probe_interval_s || 5);
  const cols = [];
  points.forEach(([at, v], i) => {
    const missing = i ? Math.round((at - points[i - 1][0]) / period) - 1 : 0;
    for (let k = 0; k < Math.min(missing, SPARKLINE_COLS); k++) cols.push(null);
    cols.push(v);
  });
  const quiet = Math.round((s.now - points[points.length - 1][0]) / period) - 1;
  for (let k = 0; k < Math.min(quiet, SPARKLINE_COLS); k++) cols.push(null);
  return cols.slice(-SPARKLINE_COLS);
}

/** One of a backend's history rings on the time axis; a history without `at` keeps sample order. */
function seriesOf(s, history, key) {
  const h = history || {};
  const values = h[key] || [];
  if (!Array.isArray(h.at) || h.at.length !== values.length) return values;
  return timeline(s, values.map((v, i) => [h.at[i], v]));
}

/** WHAT ONE REQUEST WAS GETTING, sample by sample: the tile's own rate divided by the requests the
 *  engine was RUNNING at that sample (`history.running`, beside the depth that adds the queue to it).
 *  Drawn under the headline's line as a second, quieter line — the headline is the pair's total, and
 *  the same total is a very different experience at 2 streams and at 20 (operator, 2026-10-05).
 *
 *  A sample with no `running` reading, or none running at all, is a GAP: a rate per request is not a
 *  number when nobody was being served, and 0 would claim we measured one. `running` is the engine's
 *  own count, so a rate it earned over a window where the count moved is attributed to the count at
 *  the window's end — the same approximation every other figure on this card makes.
 *
 *  An LB too old to publish the ring returns nothing and the tile simply has no second line. */
function perRequestOf(s, history, key) {
  const h = history || {};
  const values = h[key] || [];
  const running = h.running || [];
  if (!Array.isArray(values) || values.length === 0 || running.length !== values.length) return [];
  const per = values.map((v, i) => {
    const n = running[i];
    return isNum(v) && isNum(n) && n > 0 ? v / n : null;
  });
  if (!per.some(isNum)) return [];
  if (!Array.isArray(h.at) || h.at.length !== per.length) return per;
  return timeline(s, per.map((v, i) => [h.at[i], v]));
}

/** THE QUIET LINE'S NUMBER, said where a number belongs on this card: in the headline's tooltip,
 *  beside the exact value it already carries. The line itself wears no label — it is deliberately
 *  subordinate, and a second mark in a 28 px graph would compete with the target's. */
function titleWithPer(title, series, unit) {
  const v = perRequestNow(series);
  if (!isNum(v)) return title;
  const said = `${fmtInt(Math.round(v))} ${unit} per running request`;
  return title ? `${title} \u00b7 ${said}` : said;
}

/** The newest per-request figure of a series, for the headline's tooltip. */
function perRequestNow(series) {
  for (let i = (series || []).length - 1; i >= 0; i--) if (isNum(series[i])) return series[i];
  return null;
}

const generatedOf = (b) => (b.throughput || {}).generated || {};
/** A figure's reason, in the server's own words — the closed `why` vocabulary. */
const WHY_TEXT = { 'first sample': 'first sample &middot; measuring' };
const whyText = (w) => (w ? (WHY_TEXT[w] || esc(w)) : 'not reported');
/** THE PAIR ON ITS WAY UP: its server refuses connections by design, so "server down" and "engine not
 *  answering" (rates.py's words for a probe that got nothing) are not news on its card — the state is
 *  `starting`, and that is the word every figure's reason gives while `cluster.state` says so. */
const STARTING_STATES = new Set(['starting', 'restarting']);
const startingPair = (s, b) => !!b && b.kind === 'cluster' && !b.live
  && STARTING_STATES.has(String(clusterViewOf(s, b).state));
/** `whyText`, for a server: the state word while the pair starts, else the server's own reason. */
const reasonText = (s, b, w) => (startingPair(s, b) ? '<span class="pending">starting</span>' : whyText(w));
/** After this long a last request is old news: its tile dims. */
const LAST_REQUEST_DIMS_AFTER_S = 300;

/** The target for a backend's rate tiles: the dashed line on its graph, as `sparkline` options. Only the
 *  CLUSTER has one — the perf log records the pair's minutes, keyed by its settings
 *  (`throughput_target`). `which` is gen_tok_s / prompt_tok_s. Not written in words: the line IS it. */
function targetOf(s, b, which) {
  const t = b.kind === 'cluster' ? figureOf(s, clusterViewOf(s, b), 'throughput_target') : null;
  const v = t && isNum(t[which]) && t[which] > 0 ? t[which] : null;
  return v == null ? { kind: 'rate' } : { kind: 'rate', target: v };
}

/** GENERATED: tokens the server emitted over the last probe interval, every client's (its own
 *  counters). Under it only why there is no figure, or a prefill under way (a state with numbers). The
 *  draft rate is a pair in the row under the tiles; the totals since the LB started, whose tokens they
 *  were, "now · idle" and the target in words are gone — the target is the dashed line, a measured 0
 *  is the 0, and the rest were sentences under a figure (operator, 2026-10-01). */
function generatedTile(s, b) {
  const t = b.throughput || {};
  const g = generatedOf(b);
  const f = fmtNum(g.tok_s);
  let sub = '';
  if (f.text == null) {
    sub = reasonText(s, b, g.why);
  } else if (t.prefilling) {
    const kv = ((((b.loaded || {}).models || [])[0]) || {}).kv_usage;
    const from = (fmtShare(t.prefilling.kv_from).text || '?').replace(' %', '');
    sub = `prefilling ${fmtDuration(s.now - t.prefilling.since)} (KV ${from} → ${fmtShare(kv).text || '?'})`;
  }
  const per = perRequestOf(s, b.history, 'gen_tok_s');
  return tile('generated', f.text, ' tok/s', sub, seriesOf(s, b.history, 'gen_tok_s'),
    Object.assign({ now: g.tok_s, title: titleWithPer(f.title, per, 'tok/s'), under: per },
                  targetOf(s, b, 'gen_tok_s')));
}

/** PROMPT: tokens LOADED, over a window — vLLM credits a prompt in one lump at its first token. Under
 *  it only why there is no figure; its cached share and the prefill speed are pairs in the row under
 *  the tiles (`engineStats`). */
function promptTile(s, b) {
  const p = (b.throughput || {}).prompt || {};
  const f = fmtNum(p.tok_s);
  const per = perRequestOf(s, b.history, 'prompt_tok_s');
  return tile('prompt', f.text, ' tok/s', f.text == null ? reasonText(s, b, p.why) : '', seriesOf(s, b.history, 'prompt_tok_s'),
    Object.assign({ now: p.tok_s, title: titleWithPer(f.title, per, 'tok/s'), under: per },
                  targetOf(s, b, 'prompt_tok_s')));
}

/** PREFILL: the part of PROMPT the engine COMPUTED — `prompt.computed_tok_s`, PROMPT's own window and
 *  crediting with the prefix cache's hits left out, so PROMPT × (1 − cached) IS this figure. Its line is
 *  its own ring (`computed_tok_s`), its target the setup's best recorded minute of the same quantity.
 *  NOT `prefill_speed` — one request's pace per second it spent prefilling, the `prefill / request`
 *  pair under the tiles. With no PROMPT figure either, PROMPT's column already says why. */
function prefillTile(s, b) {
  const p = (b.throughput || {}).prompt || {};
  const f = fmtNum(p.computed_tok_s);
  const per = perRequestOf(s, b.history, 'computed_tok_s');
  return tile('prefill', f.text, ' tok/s', f.text == null && isNum(p.tok_s) ? whyText(p.computed_why) : '',
    seriesOf(s, b.history, 'computed_tok_s'),
    Object.assign({ now: p.computed_tok_s, title: titleWithPer(f.title, per, 'tok/s'), under: per },
                  targetOf(s, b, 'computed_tok_s')));
}

const throughputTiles = (s, b) => generatedTile(s, b) + prefillTile(s, b) + promptTile(s, b);
/** How many tiles a server's row holds — its grid's class (`tiles3` … `tiles6`): the three throughput tiles, the
 *  queue, the cluster's KV, and beside it the one of a KV tier in the host's RAM where the engine runs one
 *  (`kvRamTile`). (The ollama fallback's two last-request tiles went with the fallback tier, 2026-10-17.) */
const tileCount = (b) => 3 + 1 + (b.kind === 'cluster' ? 1 : 0)
  + (b.kind === 'cluster' && kvOffloadOf(b) ? 1 : 0);


// ── the model badge: which model, and how it was launched ────────────────────
// The app's wording (FleetStats/Shared/ModelInfo.swift `ModelBadge`): the catalog's name for the served
// id, its build, then the launch facts most telling first — "Qwen3.8 Flash-Next hibrid48 · MoE NVFP4 ·
// 256k ctx · TP=2 · MTP 4". One line; it gives up detail before it wraps (the build, then the long name,
// then TP/drafter — gateway.css, by the card's own width) and its `title` always holds all of it. This is
// the CATALOG's description; what the server itself reports holding is `modelLine`, under it.
//
// WHICH MODEL is the one thing it may never give up. The name it prints is the label's FIRST segment, and
// the shipped catalog gives three models the same one ("Qwen3.8 27B": dense NVFP4, the BF16-head DFlash2
// build, the SGLang DFlash2 build) — while the segment that tells them apart, `mb-build`, is dropped at a
// 600 px container, and the cluster panel measures 440 px at a 1440 viewport and 371 px at 390. So all
// three read "Qwen3.8 27B · 256k ctx · 8 seqs · TP=2" on every screen there is. `sharedModelName` sees
// that collision in the catalog and the badge then prints the SERVED ID, which is unique by construction
// (one engine, one id, one address) and narrower than the build it replaces.

const catalogOf = (s) => [].concat(((s || {}).models || {}).cluster || [], ((s || {}).models || {}).classic || [],
  ...activeViews(s).map((c) => (c || {}).models || []));

/** The NAME the badge prints for `label` — its first segment, the part before the build. */
const firstSegment = (label) => String(label || '').split(' · ')[0].trim();

/** Whether `label`'s name is one the CATALOG gives to more than one MODEL, so printing it does not say
 *  which model this is. Counted over distinct ids: /lb/status publishes every cluster spec twice
 *  (`models.cluster` and `cluster.models`), which is not two models. */
function sharedModelName(s, label) {
  const name = firstSegment(label);
  if (!name) return false;
  const ids = new Set(catalogOf(s).filter((m) => m && firstSegment(m.label) === name)
    .map((m) => m.id || m.served_name));
  return ids.size > 1;
}

/** "Qwen/Qwen3.8-Flash-Next" → "Qwen3.8-Flash-Next", "qwen3.8:latest" → "qwen3.8". */
function shortModel(id) {
  let x = String(id).split('/').pop() || String(id);
  if (x.endsWith(':latest')) x = x.slice(0, -':latest'.length);
  return x || String(id);
}

/** A LABEL'S PARTS, the catalog's own (" · "): the first segment names the family ("Qwen3.8 27B"), the rest the
 *  build ("dense NVFP4, BF16 head · DFlash2"); a segment that names the engine ("SGLang NEXTN") is the engine's —
 *  drawn as the engine tag, never again as words. */
function labelParts(label) {
  const segments = String(label || '').split(' · ').map((p) => p.trim()).filter(Boolean);
  const namesEngine = (p) => /\b(sglang|vllm|tensorfold)\b/i.test(p);
  return { family: segments[0] || '', build: segments.slice(1).filter((p) => !namesEngine(p)).join(' · '),
    engineInLabel: segments.some(namesEngine) };
}

// ── THE TAGS: one colour-coded family wherever a model is shown ───────────────
// Operator, 2026-10-10, on the live panel: "UX UI is terrible, need to read text to know if uncensored, etc.
// Would be better to add multicolor tags that will make it easier to read." So a model is never described in
// words alone: the picker's trigger and rows, the cluster card's model line and every server card's badge all
// wear the SAME tags, from this one list, so the eye learns one code (gateway.css `.tag`):
//   engine   vLLM and SGLang in two hues, filled            twin     UNCENSORED, loud — filled in the warning amber
//   drafter  MTP · NEXTN · DFlash2 with its depth, one hue  running  the row the pair is serving, lit green
//   arch     the build ("MoE NVFP4"), neutral               size     "98.1 GiB", neutral
//   fig      seqs · ctx · prefill · KV · TP, neutral chips  rate     "target 155 tok/s" in the positive colour,
//   note     "loading 42%", cold, other model                        "other setup" muted;  gone  "missing on box-b"
// The ORDER is the same everywhere: the decisive kinds first (build, engine, checkpoint, running, drafter), then
// the figures — so a narrow line shows what the model IS before how it is set.
const tag = (kind, text, extra) => `<span class="tag ${esc(kind)}${extra ? ` ${esc(extra)}` : ''}">${esc(text)}</span>`;

/** The tags for one model's FACTS — `pickerRowFacts` (a catalog row and its setup) or `modelBadge`'s (the
 *  generation a card names) — as `{kind, html}`, in the one order; `tagsHtml` joins the kinds a place shows. */
function modelTagList(f, { running = false } = {}) {
  const out = [];
  const add = (kind, text, extra) => out.push({ kind, html: tag(kind, text, extra) });
  if (f.build) add('arch', f.build, 'mb-build');
  if (f.engine) add('engine', f.engine, f.engineKey);
  if (f.uncensored === true) add('twin', 'uncensored');
  // The row the pair's launch record names, in the STATE's own word: "running", or "starting" while it comes
  // up, "failed" when it did not — never "running" over a pair that is not (the class carries the word too).
  if (running) {
    const word = typeof running === 'string' && running ? running : 'running';
    add('running', word, word === 'running' ? '' : word);
  }
  if (f.drafter) add('drafter', f.drafter.text, `${f.drafter.off ? 'off ' : ''}mb-launch`);
  if (f.size) add('size', f.size);
  (f.figures || []).forEach((x) => add('fig', x.text, x.extra || ''));
  if (f.note) add('note', f.note);
  if (f.rate) add('rate', f.rate.text, f.rate.other ? 'other' : '');
  if (f.gone) add('gone', f.gone);
  return out;
}
const tagsHtml = (list, kinds) => list.filter((t) => !kinds || kinds.includes(t.kind)).map((t) => t.html).join('');

/** WHICH CHECKPOINT a generation launched — read, never guessed: the launch record (`cluster.effective.uncensored`)
 *  once there is one, the stored switch (`desired.uncensored`) before it; null from an LB that says neither. */
function checkpointOf(c) {
  const from = (c || {}).effective || (c || {}).desired || {};
  return typeof from.uncensored === 'boolean' ? from.uncensored : null;
}

/** Token counts as model cards print them: 262144 → "256k", 1048576 → "1M", 200000 → "200k". */
function tokensText(n) {
  if (n >= 1048576 && n % 1048576 === 0) return `${n / 1048576}M`;
  if (n >= 1024 && n % 1024 === 0) return `${n / 1024}k`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1000) return `${Math.round(n / 1000)}k`;
  return String(n);
}

/** THE DRAFTER'S NAME as this page writes it, for the method /lb/status publishes
 *  (`launch.<mode>.speculative_method`, ef00125). The catalog's words are `mtp` and `dflash`; DFlash2 is
 *  what the dflash drafter is called everywhere it is written about — the engine's own
 *  `DFlash2DraftModel`, the checkpoint beside the target, worklog/sglang-27b-tp2.md. A method nobody has
 *  named here prints the WIRE's own word rather than a guess, and no method at all prints "draft", which
 *  is true of every drafter: an LB older than ef00125 publishes none. `nextn` is SGLang's name for the
 *  checkpoint's own MTP head, written NEXTN where its catalog label names it ("SGLang NEXTN"). */
const DRAFTER_NAMES = { mtp: 'MTP', dflash: 'DFlash2', nextn: 'NEXTN', dspark: 'DSpark' };
const drafterName = (method) => DRAFTER_NAMES[String(method || '').toLowerCase()] || method || 'draft';

/** The METHOD the model's drafter uses, as its profile publishes it. null: no drafter — or an LB that
 *  does not say, and then the page prints the depth without naming a method. */
function methodOf(spec, mode) {
  return (((spec || {}).launch || {})[mode] || {}).speculative_method || null;
}

/** THE ENGINE the profile launches (`launch.<mode>.engine`, ef00125) — the same read as `methodOf`,
 *  because it is the same kind of fact: something only the CATALOG knows about the model, never
 *  something the page may assume. One engine is on the pair at a time and it is not always vLLM: the
 *  cluster card called every pair "TP=2 vLLM" from a string literal, and said it over a serving
 *  SGLang pair on 2026-10-03. An LB too old to publish `engine` gets no engine word at all rather
 *  than the wrong one. */
const ENGINE_NAMES = { vllm: 'vLLM', sglang: 'SGLang', tensorfold: 'TensorFold' };
const engineOf = (spec, mode) => (((spec || {}).launch || {})[mode] || {}).engine || null;
const engineName = (engine) => ENGINE_NAMES[String(engine || '').toLowerCase()] || engine || null;

/** The DRAFT DEPTH for `spec` under `settings` — tokens its drafter proposes per step, whichever method
 *  draws them (the wire says `num_speculative_tokens`; `methodOf` says which). 0/absent: not speculative. */
function mtpOf(spec, mode, settings) {
  const profile = ((spec || {}).launch || {})[mode] || null;
  if (profile && profile.speculative === false) return null;
  const set = (settings || {}).num_speculative_tokens;
  const k = isNum(set) ? set : (profile || {}).num_speculative_tokens;
  return isNum(k) && k > 0 ? k : null;
}

/** WHICH ENGINE the cluster runs: the profile it was launched from for the pair, and for a node this LB only
 *  observes the node's OWN word (`effective.engine`, the vocabulary its /metrics is written in) — nothing
 *  launched it from a profile here, so a profile could only be a guess about it. */
function clusterEngineOf(c, spec) {
  return oneHost(c) ? ((c || {}).effective || {}).engine || null : engineOf(spec, 'cluster');
}

/** A cluster's ranks — its TP (`topology.tp`), one rank per box it spans; null without a cluster. `view`: whose — a
 *  cluster backend's own (`clusterViewOf`); left out, the selected cluster's (the detail card's). */
function ranksOf(s, view) {
  const c = view || theCluster(s);
  const n = c && c.configured !== false ? topologyOf(c).tp : 0;
  return n >= 1 ? n : null;
}

/** A server's model as /lb/status describes it, with the catalog's name where the catalog knows the
 *  served id. None when the server names no model. */
function modelInfoOf(s, b) {
  // ITS OWN cluster's view (by `b.cluster`), never the selected one's: a second cluster's card read the pair's label,
  // TP and engine (AUDIT-02 P0). A box server has none.
  const c = b.kind === 'cluster' ? clusterViewOf(s, b) : {};
  const served = (b.served_models || []).filter(Boolean);
  const isCluster = b.kind === 'cluster';
  const known = (name) => catalogOf(s).find((m) => m && m.served_name === name) || null;
  const name = served.find(known) || served[0] || null;
  if (!name) return null;
  // A node this LB only observes is the row its fleet declares, launched by its server file — never by the row's
  // profile (`clusterModelInfo`).
  const observed = isCluster && oneHost(c);
  const spec = observed ? declaredRowOf(s, c) : known(name);
  const launch = observed ? null : spec;
  const settings = isCluster ? (c.effective || c.desired) : null;
  const ctx = [b.max_model_len,
    isCluster ? (c.loaded_model || {}).max_model_len : null, isCluster ? (settings || {}).max_model_len : null].find(isNum);
  const label = (spec || {}).label || (isCluster && !observed ? (c.loaded_model || {}).label : null) || null;
  const mode = isCluster ? 'cluster' : 'classic';
  return { servedName: name, state: (b.loaded || {}).state || null, label,
    nameShared: sharedModelName(s, label),
    contextLen: isNum(ctx) ? ctx : null, tensorParallel: isCluster ? ranksOf(s, c) : null,
    speculativeTokens: mtpOf(launch, mode, settings),
    draftMethod: methodOf(launch, mode),
    engine: isCluster ? clusterEngineOf(c, spec) : engineOf(spec, mode),
    // The checkpoint only the pair's launch record states; a box server's is unknown, and unknown wears no tag.
    uncensored: isCluster ? checkpointOf(c) : null };
}

/** The cluster's model: the one its head serves, else the one it is set to bring up (`pending`). */
function clusterModelInfo(s, view) {
  const c = view || theCluster(s);
  if (!c.configured) return null;
  const byId = (id) => catalogOf(s).find((m) => m && m.id === id) || null;
  const settings = c.effective || c.desired || {};
  // The checkpoint of the generation shown — its launch record, or the switch before one — never a guess.
  const uncensored = checkpointOf(c);
  const l = c.loaded_model;
  if (l && l.served_name) {
    // A NODE THIS LB ONLY OBSERVES is the row its fleet declares (`declaredRowOf`), with its OWN engine and drafter:
    // its server file launched it, so the row's launch profile (the pair's) says nothing about it.
    const observed = oneHost(c);
    const spec = observed ? declaredRowOf(s, c)
      : (byId(l.spec) || catalogOf(s).find((m) => m && m.served_name === l.served_name) || null);
    const launch = observed ? null : spec;
    const ctx = [l.max_model_len, settings.max_model_len].find(isNum);
    const label = observed ? (spec || {}).label || null : (l.label || (spec || {}).label || null);
    return { servedName: l.served_name, state: 'serving', label, nameShared: sharedModelName(s, label),
      contextLen: isNum(ctx) ? ctx : null, tensorParallel: ranksOf(s, c),
      speculativeTokens: mtpOf(launch, 'cluster', settings), draftMethod: methodOf(launch, 'cluster'),
      engine: clusterEngineOf(c, spec), uncensored };
  }
  if (!(c.effective || (c.desired || {}).mode === 'cluster')) return null;
  const choice = settings.model_choice || (c.defaults || {}).model_choice;
  if (!choice) return null;
  const spec = byId(choice);
  // A one-host node's launch is its host's, never the row's PAIR profile: no drafter or engine from the row.
  const launch = oneHost(c) ? null : spec;
  const ctx = [settings.max_model_len, (c.defaults || {}).max_model_len].find(isNum);
  const pct = (c.loading || {}).percent;
  return { servedName: (spec || {}).served_name || choice, state: 'pending', label: (spec || {}).label || null,
    nameShared: sharedModelName(s, (spec || {}).label || null),
    contextLen: isNum(ctx) ? ctx : null, tensorParallel: ranksOf(s, c),
    speculativeTokens: mtpOf(launch, 'cluster', isNum(settings.num_speculative_tokens) ? settings : c.defaults),
    draftMethod: methodOf(launch, 'cluster'), engine: clusterEngineOf(c, spec), uncensored,
    loadingPercent: isNum(pct) ? Math.round(pct) : null };
}

/** The badge: the name, then the model's TAGS (`modelTagList`) — the build, the engine, UNCENSORED when the
 *  generation launched the twin, the drafter named and counted ("MTP 4", "DFlash2 4" — the profile's own
 *  `speculative_method`; "draft 4" from an LB that publishes none, never a guess), then the figures. `extra` are
 *  figures of the place it is drawn in (the cluster card adds "16 seqs"). The line wraps: a tag the operator
 *  cannot see says nothing (2026-10-10), and the decisive tags come first so a narrow card still shows them. */
function modelBadge(m, extra) {
  if (!m) return '';
  const { family, build } = labelParts(m.label);
  const short = shortModel(m.servedName);
  const long = family || short;
  // A name the catalog shares with another model names a family, not a checkpoint. The served id does
  // name one, so it is the name at EVERY width (gateway.css `.ambiguous`) — never a fact a container
  // query may drop. `title` keeps the label, the build and the served id whichever is printed.
  const ambiguous = !!m.nameShared && !!short && short !== long;
  const note = m.state === 'pending' ? (m.loadingPercent != null ? `loading ${m.loadingPercent}%` : '')
    : ({ cold: 'cold', other_model: 'other model' }[m.state] || '');
  const figures = [];
  if (m.contextLen != null) figures.push({ text: `${tokensText(m.contextLen)} ctx` });
  (extra || []).forEach((x) => figures.push({ text: x }));
  // `mb-launch` is the pair's own droppable fact (the card's subtitle says TP=2 as well).
  if (m.tensorParallel >= 1) figures.push({ text: `TP=${m.tensorParallel}`, extra: 'mb-launch' });
  const facts = {
    family: long, build, engine: engineName(m.engine), engineKey: String(m.engine || '').toLowerCase(),
    uncensored: typeof m.uncensored === 'boolean' ? m.uncensored : null,
    drafter: m.speculativeTokens > 0 ? { text: `${drafterName(m.draftMethod)} ${m.speculativeTokens}`, off: false } : null,
    size: null, figures, note: note || null, rate: null, gone: null,
  };
  const whole = [String(m.label || '') || long, facts.engine && !labelParts(m.label).engineInLabel ? facts.engine : null,
    facts.uncensored ? 'uncensored' : null].concat(figures.map((x) => x.text), [note || null, facts.drafter ? facts.drafter.text : null])
    .concat(m.servedName && m.servedName !== long ? [m.servedName] : []).filter(Boolean).join(' · ');
  const name = long === short ? `<b>${esc(long)}</b>`
    : `<b class="mb-long">${esc(long)}</b><b class="mb-short">${esc(short)}</b>`;
  return `<div class="model-badge${ambiguous ? ' ambiguous' : ''}${m.state === 'pending' ? ' pending' : ''}" title="${esc(whole)}"><i class="mb-icon" aria-hidden="true"></i>`
    + `<span>${name}${tagsHtml(modelTagList(facts))}</span></div>`;
}

const gbOfBytes = (bytes) => `${(bytes / 1e9).toFixed(1)} GB`;

/** THE MODEL LINE: what the server holds, in its own words — /v1/models and KV usage. With a model badge above it (`badged`) the name and the window are not written
 *  again; with the KV share on a tile (`kvOnTile`) nor is that. What is left — the KV share, what is on
 *  the GPU, an expiry, another client's use — or nothing. A narrow card cuts it with an ellipsis;
 *  `title` keeps all of it, the name and the window included. */
function modelLine(s, b, badged, kvOnTile) {
  const l = b.loaded;
  if (!l || !b.live) return '';
  const ours = (b.served_models || [])[0] || 'the model';
  const m = (l.models || [])[0] || {};
  let parts;
  let raw = '';
  if (l.state === 'cold') {
    parts = ['nothing loaded', `${ours} loads on the next turn`];
  } else if (l.state === 'other_model') {
    parts = [`${m.name} loaded (not ours)`, `${ours} loads on the next turn`];
  } else if (l.state === 'resident' || l.state === 'serving') {
    const kv = fmtShare(m.kv_usage);
    if (kv.title) raw = ` (KV ${kv.title})`;
    const gpu = typeof m.gpu_bytes !== 'number' ? null
      : (m.offloaded && typeof m.size_bytes === 'number'
        ? `${(m.gpu_bytes / 1e9).toFixed(1)} of ${gbOfBytes(m.size_bytes)} on GPU` : `${gbOfBytes(m.gpu_bytes)} on GPU`);
    const expiry = m.expiry === 'counting down' && typeof m.expires_in_s === 'number'
      ? `unloads in ${fmtDuration(m.expires_in_s)}` : (m.expiry === 'in use' || m.expiry === 'unloading' ? m.expiry : null);
    parts = [m.name, typeof m.context === 'number' ? `${fmtInt(m.context)} ctx` : null,
             kv.text != null ? `KV ${kv.text}` : null, gpu, expiry].filter(Boolean);
  } else {
    parts = [`model — ${l.why || 'not reported'}`];
  }
  if (typeof l.other_client_at === 'number') {
    parts.push(`used by another client ${fmtDuration(s.now - l.other_client_at)} ago`);
  }
  const whole = parts.join(' · ') + raw;
  if (badged && (l.state === 'resident' || l.state === 'serving')) {
    parts = parts.filter((p) => p !== m.name && !/ ctx$/.test(p) && !(kvOnTile && /^KV /.test(p)));
  }
  if (!parts.length) return '';
  return `<div class="model-line" title="${esc(whole)}">${parts.map(esc).join(' &middot; ')}</div>`;
}

// ── shared with the router ───────────────────────────────────────────────────
// depth = waiting + max(running, inflight): vLLM's `running` already counts the request we
// sent, so summing our own inflight on top would count it twice.
//
// ROUTING ONLY, and it must keep agreeing with router.py `depth` — it decides `demoted` and
// SERVING vs IDLE in `statusOf`. It is deliberately the PESSIMISTIC reading, because the probe is stale and under-counting hands one box every turn
// that arrives inside an interval. That is the right input to an admission decision and the
// wrong thing to print: the page prints the ENGINE's own two counters and nothing else.
const depthOf = (b) => b.waiting + Math.max(b.running, b.inflight);

/** WHAT IS ON THIS SERVER: the engine's two counters, and they are the whole of it. `inflight` —
 *  what this LB has forwarded — is NOT a third quantity on the page. A request is running or it is
 *  waiting; anything else is a number the operator cannot act on, so it is not printed (operator,
 *  2026-09-30: "there's no ours that is not running nor waiting"). It still drives ROUTING, above. */
const onServer = (b) => b.running + b.waiting;

/** THE ENGINE'S COUNT, for the REQUESTS card's head: the SAME backend fields the QUEUE tile reads (`running`,
 *  `waiting`), summed over the live servers whose /metrics answered, and how old the oldest read is (`read_s`, off
 *  probe.py's `depth_read_at` and the status's own `now`; null when either is missing — no age is guessed). Null
 *  when no server's depth was read: the head then prints a dash, never a 0 (operator 2026-10-30: "why shows only
 *  one request if queue tells 2?" — the two figures must be one definition each, and say whose they are). */
function engineCount(backends, now) {
  const read = (backends || []).filter((b) => b && b.live !== false && b.depth_known !== false
    && isNum(b.running) && isNum(b.waiting));
  if (!read.length) return null;
  const ages = read.map((b) => (isNum(b.depth_read_at) && isNum(now) ? Math.max(0, now - b.depth_read_at) : null));
  return { running: read.reduce((a, b) => a + b.running, 0), waiting: read.reduce((a, b) => a + b.waiting, 0),
    backends: read.length, read_s: ages.every(isNum) ? Math.max(...ages) : null };
}

/** THE SERVER CARD'S QUEUE TILE: the engine's running + waiting over the concurrency, and NOTHING under
 *  it. The breakdown is the RUN + WAIT figure under the tiles (`engineStats`), and the two must agree.
 *
 *  It used to head `waiting + max(running, inflight)` over "0 waiting &middot; 6 running &middot; 25
 *  ours", and the operator read it exactly as written: "queue is not valid, 25 but only 6 running?".
 *  Three numbers that do not add to the one above them are not a breakdown — they are three
 *  unrelated facts sharing a box, and the biggest one wins the reader's attention by accident. The
 *  sub-line that then restated the headline ("16 running + 0 waiting" under 16/16) was noise. */
function queueTile(b, hist, opts) {
  const series = (hist || {}).depth;
  const o = opts || {};
  // /metrics DID NOT ANSWER, so `running` and `waiting` are probe.py's placeholder zeros rather than
  // readings. Printing "0 running" off them is the page inventing an idle engine out of a missing
  // one — the `depth_known` lesson, already learnt once on this fleet.
  if (b.depth_known === false) {
    return tile('queue', null, '', o.starting ? '<span class="pending">starting</span>'
      : '<span class="stale">engine not answering</span>', series);
  }
  // "5/6": the depth over the concurrency it is read against, the "/6" in the unit's quiet ink — and,
  // where the KV pool seats fewer than that at the configured context, how many it does ("· fits 4"):
  // the effective depth, which is what the queue really drains at (`queueOpts`).
  const fits = isNum(o.fits) && o.concurrency > 0 && o.fits < o.concurrency ? ` · fits ${o.fits}` : '';
  const of = o.concurrency ? `/${o.concurrency} seats${fits}` : '';
  return tile('queue', onServer(b), of, queueParts(b, o.engine), series, o);
}

/** THE QUEUE'S PARTS, under its graph (operator, 2026-10-12: "tell how many are streaming, filling, queued"), and they
 *  ADD TO THE HEADLINE: the engine's running split into STREAMING (the LB's rows on this server with bytes reaching
 *  the client) and FILLING (running, no byte back yet: prefilling — the rest of `running`), then QUEUED = its waiting,
 *  and why they wait where the figures prove it (request_queue.admission). With more LB streams than the engine runs
 *  (a probe older than the streams) no split is guessed: "R running". No rows for this server: nothing. */
function queueParts(b, eng) {
  if (!eng || !eng.rows || !isNum(b.running) || !isNum(b.waiting)) return null;
  const streams = eng.rows.streaming || 0;
  // A figure and its word never part at a wrap (&nbsp;); the parts may.
  const parts = streams <= b.running
    // "stream" / "fill": the short words, so the three parts and "engine:" fit one line of the fixed-height cell
    // (1500 px cut "0 queued", 2026-10-30); the title spells them out.
    ? [`${fmtInt(streams)}&nbsp;stream`, `${fmtInt(b.running - streams)}&nbsp;fill`]
    : [`${fmtInt(b.running)}&nbsp;running`];
  parts.push(`${fmtInt(b.waiting)}&nbsp;queued`);
  const why = { kv: 'kv&nbsp;full', seats: 'seats&nbsp;full', deferred: 'deferred' }[eng.wait];
  if (why && b.waiting > 0) parts.push(why);
  const title = `the engine's own /metrics read, not this gateway's list · streaming: the LB's rows with bytes back · `
    + `filling: the engine's running less those (prefilling) · queued: the engine's waiting${eng.why ? ` — ${eng.why}` : ''}`;
  // "engine:" — WHOSE count (operator 2026-10-30): the REQUESTS card's own figure is the gateway's, and the two were
  // read as the same thing.
  return `<span title="${esc(title)}">engine: ${parts.join(' &middot; ')}</span>`;
}

/** THE NEWEST SAMPLE of a probe ring, or null: a ring reads newest last, and a gap at its end IS the
 *  reading — the engine was not answering. The last NUMBER in the ring was the previous generation's
 *  final sample, and it stood on a starting pair's card as "KV 25 %" for ten minutes. */
const newestOf = (ring) => {
  const v = Array.isArray(ring) && ring.length ? ring[ring.length - 1] : null;
  return isNum(v) ? v : null;
};

/** KV: the pool filling, drawn from the probe history (`kv_pct`), with the tokens that occupancy is.
 *  The prefix hit rate over the last interval (`hit_pct`) — the cache the pool evicts as it fills — is
 *  the "prefix hit" pair in the row under the tiles (`prefixHitOf`). Tonight's failure, read together:
 *  occupancy climbing to 99 % while the hit rate fell to 0. */
function kvTile(s, b, hist) {
  const h = hist || {};
  const series = Array.isArray(h.kv_pct) ? h.kv_pct : [];
  const last = newestOf(series);
  const c = clusterViewOf(s, b);            // the pool of THIS card's cluster, never the selected one's
  // THE DENOMINATOR IS THE ENGINE'S OWN POOL while it reports one (`throughput.kv_pool`, read off its
  // cache_config_info). Before that it is the estimate for the model the pair runs — its pin at ITS
  // bytes per token — and with neither, no token figure at all: never another model's constant.
  const eff = c.effective || c.desired || {};
  const engine = (((b || {}).throughput || {}).kv_pool || {}).tokens;
  const pin = eff.kv_cache_memory || profileOf(c, eff.model_choice).kv_cache_memory;
  const tok = isNum(engine) && engine > 0 ? engine
    : poolOf(c, eff.model_choice, pin, eff.max_num_seqs, eff.max_model_len).tokens;
  if (!isNum(last)) {
    return tile('kv', null, '', startingPair(s, b) ? '<span class="pending">starting</span>'
      : '<span class="stale">no KV reading</span>', series);
  }
  // BOTH the share and the size: a percentage alone hides the scale, and the scale is a setting —
  // 24 % of a 2.13M-token pool is not 24 % of a 1.71M one. Resident / pool, in millions of tokens.
  const M = (n) => `${(n / 1e6).toFixed(2)}M`;
  const unit = tok ? `% · ${M(tok * last / 100)} / ${M(tok)}` : '%';
  return tile('kv', Math.round(last), unit, null, series, { kind: 'load', max: 100, now: last });
}

/** THE KV TIER IN THE HOST'S RAM (`throughput.kv_offload`, throughput.py `kv_offload`): null where the engine runs
 *  none — and then neither of its tiles is drawn. Operator, 2026-10-12: "implement offload KV and non-VRAM KV graph". */
const kvOffloadOf = (b) => (((b || {}).throughput || {}).kv_offload) || null;

/** GB/s as a tile prints it: two decimals below 10 (a store of 0.08 GB/s is not "<0.1"), one below 100. */
const gbpsText = (gb) => (gb === 0 ? '0' : gb < 0.01 ? '<0.01' : gb < 10 ? gb.toFixed(2)
  : gb < 100 ? gb.toFixed(1) : String(Math.round(gb)));
/** A bytes-a-second ring's quiet "max" label, in the headline's own unit. */
const gbpsMark = (bytesS) => gbpsText(bytesS / 1e9);
const gibText = (g) => (Number.isInteger(g) ? String(g) : g.toFixed(1));

/** KV · RAM, BESIDE KV — the tier in the host's RAM as ONE tile, headed like KV (operator, 2026-10-12: "show how much
 *  data is filled on other fallback KV … two charts as two rows in one per memory type, total usage and available on
 *  top and Gb/s below"; "same colors as memory bandwidth, since high is not a good thing so should be red"):
 *
 *    * the head: the fill and what it holds of the tier the engine states — "100 % · 32.0 / 32 GiB". DERIVED by the LB
 *      (`kv_offload.occupancy_derived`, which the hover carries): vLLM publishes no occupancy. On the LOAD scale memory
 *      bandwidth uses, so a full tier is red. Not known: "—", its reason in the hover — the rates still drawn;
 *    * two graphs stacked in the room one takes (`t-pair`), so the row keeps its height: the fill's past on top
 *      (`history.kv_ram_pct`, 0-100 %), the reload rate below with the store's as the quieter line on ONE axis at least
 *      as tall as the copy speed the engine measured, on the same load scale — a reload as fast as the copy goes is red;
 *    * under them both rates and both copy speeds; a direction not measured is "—", never a 0. */
function kvRamTile(s, b, hist) {
  const off = kvOffloadOf(b);
  const rl = off.reload || {}, st = off.store || {};
  const known = isNum(off.occupancy_pct);
  const used = known && isNum(off.used_gib) ? `${off.used_gib.toFixed(1)} / ` : '';
  const fill = seriesOf(s, hist, 'kv_ram_pct');
  const reload = seriesOf(s, hist, 'kv_reload_bytes_s');
  const store = seriesOf(s, hist, 'kv_store_bytes_s');
  const rate = (d) => (isNum(d.bytes_s) ? gbpsText(d.bytes_s / 1e9) : '—');
  const copy = (d) => (isNum(d.copy_gbps) ? gbpsText(d.copy_gbps) : '—');
  // The axis for both directions: the copy's own speed (the faster direction's), or the window's peak if over it.
  const copyTop = Math.max(0, ...[rl, st].map((d) => d.copy_gbps).filter(isNum)) * 1e9;
  const peak = Math.max(0, ...[...reload.slice(-SPARKLINE_COLS), ...store.slice(-SPARKLINE_COLS)].filter(isNum));
  const flowTop = Math.max(copyTop, peak);
  const pct = known ? off.occupancy_pct : null;
  const colour = known ? scaleColor(pct / 100) : null;
  const title = known ? off.occupancy_derived || '' : off.occupancy_why || 'not measured';
  const pair = `<div class="t-pair">${graph(fill, { cols: SPARKLINE_COLS, key: 'kv · ram|fill', max: 100 })}`
    + `${graph(reload, { cols: SPARKLINE_COLS, key: 'kv · ram|flow', under: store, max: flowTop > 0 ? flowTop : undefined,
                         markText: (v) => `${gbpsMark(v)} GB/s` })}</div>`;
  const note = `${rate(rl)} · ${rate(st)} GB/s · copy ${copy(rl)} · ${copy(st)}`;
  return `<div class="tile${known ? '' : ' dim'}">
    <div class="t-head" title="kv · ram"><span class="t-label">kv · ram</span>
    <b class="t-val" title="${esc(title)}"${colour ? ` style="color:${colour}"` : ''}>${known ? Math.round(pct) : '—'}${known ? `<span class="t-unit">% · ${used}${gibText(off.tier_gib)} GiB</span>` : ''}</b></div>
    ${pair}
    <div class="t-note"${cutTitle(note)}><span class="t-sub">${note}</span></div>
  </div>`;
}

/** The prefix hit rate over the last interval, from the probe history's `hit_pct` ring; null unread.
 *  The ring is written only on a probe that read a rate, so its newest NUMBER is the current reading. */
function prefixHitOf(hist) {
  const ring = (hist || {}).hit_pct;
  const last = Array.isArray(ring) ? [...ring].reverse().find(isNum) : null;
  return isNum(last) ? last : null;
}

/** The queue scale's option for a backend: the cluster's concurrency, from the settings the pair runs
 *  (`cluster.effective`, else `desired`), and how many requests its KV pool SEATS at the configured
 *  context (`fits`) — the engine's own pool, else the model's estimate at its bytes per token; the
 *  same arithmetic as the pool line under the settings. */
function queueOpts(s, b) {
  // The server's rows per state and why its queued wait (/lb/status `queue.backends`, request_queue.queue_view).
  const engine = (((s || {}).queue || {}).backends || {})[b.url];
  const c = clusterViewOf(s, b);            // ITS seats and pool: a second cluster showed the pair's "/8 seats"
  const live = c.effective || c.desired || {};
  const n = live.max_num_seqs;
  if (!(isNum(n) && n > 0)) return { starting: startingPair(s, b), engine };
  const ctx = live.max_model_len || (c.defaults || {}).max_model_len;
  const pin = live.kv_cache_memory || profileOf(c, live.model_choice).kv_cache_memory;
  const tok = (c.effective && enginePoolTokens(s, b)) || poolOf(c, live.model_choice, pin, n, ctx).tokens;
  const fits = isNum(tok) && isNum(ctx) && ctx > 0 ? Math.max(1, Math.floor(tok / ctx)) : null;
  return { kind: 'queue', concurrency: n, fits, starting: startingPair(s, b), engine,
           title: `max_num_seqs ${n}${isNum(fits) && isNum(ctx) ? ` · the KV pool holds ${fits} request${fits === 1 ? '' : 's'} of ${fmtInt(ctx)} tokens` : ''}` };
}

/** How long a server may move NO token — generated or prefilled — with work queued before it
 *  is called WEDGED. The server measures it (`stalled_s`); the page only draws the line. */
const WEDGED_AFTER_S = 120;

/** THE ROUTER'S DEMOTION (router.py `busy`) for a cluster: queued past its own seats (`cluster.max_depth`: the pair's
 *  max_num_seqs, a node's declared seats), or with its depth unknown — demoted, not banned. A cluster whose seats
 *  nobody declared has none to be within, and is demoted as the router demotes it. */
function demoted(s, b) {
  if (b.depth_known === false) return true;
  const seats = clusterViewOf(s, b).max_depth;       // its OWN seats, as router.py reads them
  return !Number.isInteger(seats) || depthOf(b) > seats;
}

/** ONE state for a HELD, live cluster (`clusterStatusOf` decides the rest), read as a lit diode. The order is a
 * precedence, not a list. */
function statusOf(s, b) {
  const d = depthOf(b);
  // WEDGED is a server that has STOPPED, not one that is busy. "0 tok/s with work queued" used
  // to be the rule, and it painted a normal long prefill red: an 84 KB prompt generates no token
  // while it is being read. `stalled_s` counts the seconds in which NEITHER the generated nor
  // the prompt counter moved with work queued; null (idle, or unmeasured) is never a stall.
  if (typeof b.stalled_s === 'number' && b.stalled_s >= WEDGED_AFTER_S) return { tone: 'bad', label: 'WEDGED' };
  if (b.penalised) return { tone: 'warn', label: 'PENALISED' };
  // The TP=2 cluster is a vLLM too: with its /metrics silent, SERVING or IDLE would be a guess.
  if (b.depth_known === false) return { tone: 'warn', label: 'DEPTH ?' };
  if (demoted(s, b)) return { tone: 'warn', label: 'DEMOTED' };
  if (d > 0) return { tone: 'ok', label: 'SERVING', live: true };
  return { tone: 'ok', label: brokerOff(s) ? 'IDLE' : 'LEASED · IDLE', dim: true };
}

const diode = (st) => `<span class="diode ${st.tone}${st.live ? ' pulse' : ''}${st.dim ? ' soft' : ''}"
  role="img" aria-label="${esc(st.label)}"></span>`;

/** broker_parrot OFF (LB_BROKER_ENABLED unset): /lb/status says `broker.enabled: false` and carries no lease
 *  or broker DB state. Absent — an LB from before the switch — is the broker ON, as it always was. */
const brokerOff = (s) => !!(s && s.broker) && s.broker.enabled === false;

/** WHAT THE CLUSTER IS (`cluster.topology`, vllm_lb/topology.py): its TP, its boxes head first, how this LB reaches
 *  it (`control`: `ssh` drives the pair, `none` only observes a node — `why` says so) and the start steps that apply.
 *  The page named every cluster `box-a + box-b` and `TP=2` from literals until a one-GPU node had to be shown as one
 *  (worklog/standalone-cluster-panel.md). A document from before the field — a capture, an older LB — is the pair it
 *  always was: its boxes are `cluster.boxes`' own, reached over SSH, every step applying. */
function topologyOf(c) {
  const t = (c || {}).topology;
  if (t && Array.isArray(t.boxes) && t.boxes.length) return t;
  const boxes = Object.keys((c || {}).boxes || {});
  return { tp: boxes.length, boxes, control: 'ssh', why: '', steps: START_STEPS };
}
/** THE VIEW A CLUSTER BACKEND BELONGS TO (worklog/parallel-clusters.md): its own cluster's, by the name the backend
 *  carries (`status.clusters[b.cluster]`), else the one cluster's (`theCluster(status)`) — a document from before names. */
const clusterViewOf = (s, b) => ((b || {}).cluster && ((s || {}).clusters || {})[b.cluster]) || (s || {}).cluster || {};
/** SERVER `view`'s OWN figure `key` — `saturation`, `throughput_target`, `setup_targets`, each built off that server's
 *  config, head and perf minutes (/lb/status.clusters[name], proxy `_server_figures`) — or, a document from before
 *  names (no `clusters`), the one engine's top-level one. null when it has none. */
function figureOf(s, view, key) {
  const views = (s || {}).clusters;
  const named = views && typeof views === 'object' && Object.keys(views).length;
  const v = named ? (view || {})[key] : (s || {})[key];
  return v == null ? null : v;
}
/** EVERY active cluster's view (`status.clusters`), or — a document from before names — the one `status.cluster`. */
const activeViews = (s) => {
  const views = (s || {}).clusters;
  return views && typeof views === 'object' && Object.keys(views).length ? Object.values(views) : [(s || {}).cluster || {}];
};
/** The cluster backend whose head `view` is (`b.cluster === view.name`) — or, a document from before names, the one. */
const headOf = (s, view) => {
  const heads = ((s || {}).backends || []).filter((b) => b && b.kind === 'cluster');
  return (view && view.name ? heads.find((b) => b.cluster === view.name) : heads[0]) || null;
};
/** THE SERVER the page's detail card, its switch, its engine card, its requests and every singular reader mean
 *  (worklog/parallel-clusters.md): the SELECTED one of `status.clusters` (clusters.js `selectedCluster`, the server
 *  tabs; the first of the run until one is picked, or when the picked one left the run) — or `status.cluster` for a
 *  document from before names. The one-server page reads exactly what it always did. */
function theCluster(s) {
  const views = (s || {}).clusters;
  const names = views && typeof views === 'object' ? Object.keys(views) : [];
  if (!names.length) return (s || {}).cluster || {};
  const picked = typeof selectedCluster === 'string' ? selectedCluster : null;
  return views[picked] || views[names[0]];
}
/** Where the selected cluster's switch and estimate are POSTed: the view's own route, or the one-cluster route. */
const clusterRoute = () => theCluster(lastStatus).switch || '/lb/cluster';
/** The cluster's name: its boxes, head first — "box-a + box-b" for a pair, the box for a node. `b`: the cluster
 *  backend asking, so with several clusters each server card is named by its own. */
const clusterName = (s, b) => topologyOf(clusterViewOf(s, b)).boxes.join(' + ');
/** A node this LB only observes (`control: none`): nothing here may start, stop or reconfigure it. */
const observedOnly = (c) => topologyOf(c).control === 'none';
/** A one-host node (control `local` or `none`): its identity is its fleet's declaration, its engine its own word. */
const oneHost = (c) => ['local', 'none'].includes(topologyOf(c).control);
/** A link BETWEEN RANKS exists: the topology has more than one (`tp`). What the NETWORK tile measures; never a box's
 *  name or a card's, never the server's reason for an empty reading. */
const rankLink = (c) => topologyOf(c).tp > 1;

/** An idle cluster is not a server anyone is waiting on. Its card — and its place in the
 *  counts — appear once the operator starts it, or if it answers anyway. */
function visibleBackends(s) {
  return s.backends.filter((b) => {
    if (b.kind !== 'cluster' || b.live) return true;
    const c = clusterViewOf(s, b);         // each head by its OWN cluster: a starting second one is shown too
    return !!c.configured && c.state !== 'off';
  });
}

// ── status ───────────────────────────────────────────────────────────────────

const fmtInt = (n) => (typeof n === 'number' ? n.toLocaleString('en-US') : '—');

// ── the two-box cluster ────────────────────────────────────────────────────
// One switch for a TP=2 Qwen3.8 across box-a + box-b. The LB's planner owns every step — park
// both lanes through broker_parrot, let their work drain, start both ranks, serve once the head
// answers — so this panel only posts the operator's intent and says which step the planner is
// on, in the planner's own words. Nothing here is inferred from a clock.

/** A GB10 box belongs to the pair only through its RESERVATION — one broker_parrot node_job per box — and
 *  only while broker_parrot runs it. A park is always the operator's, never the cluster's. */
const reservedForCluster = (reservation) => !!reservation && reservation.status === 'running';
const CONTEXT_CHOICES = [32768, 65536, 131072, 262144];
/** THE ONE TONE PER STATE. Red is `failed` ALONE — a rank that exited, the memory guard, a refused launch, a
 *  reservation the broker ended: something that happened (cluster.py `plan`). Everything else short of running
 *  or off is amber, "not there yet": a start waiting on a box, a share, the RAM, a reservation — and `stalled`,
 *  a hung start the LB halted after its relaunches, and `unknown`, a box that could not be observed. A clock
 *  running out is never painted as an error (operator, 2026-10-10: "there must be real error, not timeout to
 *  show failed"). */
const clusterTone = (state) => ({ off: 'off', running: 'ok', failed: 'bad', stalled: 'warn', unknown: 'warn' }[state] || 'warn');
/** The states that do not end without the operator — the planner's `failed`, `stalled`, `blocked`
 *  (proxy.py CLUSTER_NEEDS_OPERATOR): the Stop button's title says what it does for them. */
const NEEDS_OPERATOR = ['failed', 'stalled', 'blocked'];

let clusterBusy = false;    // a POST /lb/cluster is in flight

/** THE STARTUP, in the head rank's own words (vllm_lb/cluster_startup.py): one bar per LoadPass — a
 *  read of the checkpoint's shards with its own count from 0 — and the StartupPhase by name. vLLM
 *  reads the checkpoint once for the model and again for the drafter; one bar for both fell from
 *  27/28 to 3/28 and read as a restart, so every LoadPass keeps its own bar and a finished one stays
 *  full. A StartupPhase with no count (compile, KV cache, warmup) is named under the bars. No ETA.
 *  (The app folds finished passes into one line while the head warms up; the page keeps the bars.) */
const countsOf = (p) => !!p && Number.isInteger(p.total) && p.total > 0
  && Number.isInteger(p.completed) && p.completed >= 0 && p.completed <= p.total;
const loadPassesOf = (c) => (Array.isArray((c.loading || {}).passes) ? c.loading.passes : []).filter(countsOf);

function startupProgress(c) {
  if (c.state !== 'starting') return '';
  const l = c.loading || {};
  const loadPasses = loadPassesOf(c).map((p) => {
    const name = esc(p.name || 'checkpoint shards');
    const pct = typeof p.percent === 'number' ? Math.round(p.percent) : Math.round((100 * p.completed) / p.total);
    return `<div class="load-pass${p.completed >= p.total ? ' full' : ''}">
      <progress value="${p.completed}" max="${p.total}"
        aria-label="load pass ${name}: ${p.completed} of ${p.total} checkpoint shards"></progress>
      <span>${name} &middot; ${p.completed} / ${p.total} (${pct}%)</span>
    </div>`;
  });
  // While a LoadPass counts, its bar says what runs; the StartupPhase is named only when none counts.
  // The tone of the clock, never its number: #cluster-state is a live region, so the headline may
  // say WHICH step and how it is going, and the trail outside says for how long.
  const quiet = startupNow ? startupNow.quiet : null;
  const tone = quiet == null ? ''
    : (quiet >= STARTUP_LONG_S ? ' long' : (quiet >= STARTUP_SLOW_S ? ' slow' : ''));
  const phase = l.phase && !(loadPasses.length && countsOf(l))
    ? `<span class="startup-phase${tone}">${esc(l.phase)}</span>` : '';
  return loadPasses.length || phase ? `<div class="startup-progress">${loadPasses.join('')}${phase}</div>` : '';
}

// ── THE TRAIL: every step the head printed, and how long it has been on the last one ─────────────
// `cluster.loading.steps` (cluster_startup.py `StartupProgress.steps`) is the WHOLE trail of
// StartupPhases a start has passed through, consecutive repeats collapsed. It has been on the wire
// since f614bf2 and until now the page dropped it on the floor: `startupProgress` reads `phase` and
// `passes` only, so a 19-step SGLang boot drew as one line of text and a start that stopped moving
// looked exactly like one that was moving.
//
// IT IS A LIST THAT ONLY GROWS, NEVER ONE ADVANCING BAR. SGLang's shape is genuinely doubled: the
// real start this is built from (tests/fixtures/sglang-lb-start-2026-10-03-rank0.log, generation
// 121) prints "connecting the ranks (NCCL)" and "initializing model" twice — the target's round at
// 23:51:05, then the drafter's at 23:51:52 — and "capturing CUDA graphs" twice (Capture target
// verify, then Capture draft verify), so four of its nineteen steps are repeats of a name already
// in the trail. A list that only ever appends cannot read as a restart, which is what ONE bar
// falling from 27/28 to 3/28 did once and why `LoadPass` exists; each repeat carries its ordinal so
// it reads as "it did this twice", which is what the log says.
//
// WHOSE round a repeat is, the page does not say. SGLang names it ("Capture target verify CUDA
// graph begin" / "Capture draft verify …", `Load weight end … type=DFlash2DraftModel`) but
// `read_startup` keeps none of the qualifier, so the ordinal is the whole of what reached the page.

/** How long the page has seen NO NEW STEP, on the LB's own clock.
 *
 *  The LB measures exactly this server-side — `cluster_loop._track` keeps `_progress` = (the whole
 *  `loading` view, when it was first read saying that, when it was last read) and `cluster.plan`
 *  turns it into a stall — but publishes only the view, so the page keeps the same clock itself:
 *  `s.now` at the poll where `loading` last CHANGED. Changed means ANY part of it, a step appended,
 *  a shard counted or another GiB taken (`taken_gib`: what the head's box has given up during a weight
 *  pass — docker shows SGLang's `\r` bar only once the pass is done, so the box's memory is the load's
 *  progress; cluster_loop.py `_load_taken`), which is the comparison `_track` makes
 *  (`self._progress[0] == self._loading`) — so a weight load that is landing is never quiet, and only a
 *  FROZEN one is.
 *
 *  `sawBegin` is whether this page watched the step arrive. Opened mid-step it did not, and then
 *  the clock is a floor and says so ("≥"); the comparison against the steps before it, which would
 *  then be an upper bound dressed as a measurement, is not drawn at all. */
let startupWatch = null;          // {since, sig, at, sawBegin} for the start being watched
//: THIS RENDER'S reading of it, taken once in `renderCluster` before anything draws. Two things show
//: it — the StartupPhase headline inside #cluster-state and the trail outside it — and they must not
//: disagree: a spinner turning calmly over "capturing CUDA graphs" while the trail underneath reads
//: 13 min is the panel saying "working" about the exact minute it is failing in.
let startupNow = null;

function watchStartup(c, s) {
  const l = c.loading || {};
  const steps = (Array.isArray(l.steps) ? l.steps : []).filter((x) => typeof x === 'string' && x);
  if (!steps.length) { startupWatch = null; return null; }
  if (!isNum(s.now) || !isNum(c.since)) { startupWatch = null; return { steps, quiet: null }; }
  const counts = loadPassesOf(c).map((p) => `${p.name}:${p.completed}/${p.total}`).join(',');
  const taken = isNum(l.taken_gib) ? l.taken_gib : null;
  const sig = `${l.phase || ''}|${counts}|${taken == null ? '' : taken}|${steps.join('\u0000')}`;
  const prev = startupWatch;
  const sameStart = !!prev && prev.since === c.since;
  if (!sameStart || prev.sig !== sig) startupWatch = { since: c.since, sig, at: s.now, sawBegin: sameStart };
  const quiet = Math.max(0, s.now - startupWatch.at);
  // What the steps BEFORE this one took, from the two exact figures the wire carries: how long the
  // pair has been `starting` less the quiet. It over-counts by the seconds between the container
  // starting and its first recognised line, which can only make the earlier steps look SLOWER —
  // the safe direction for a comparison whose whole job is to not manufacture alarm.
  return { steps, quiet, taken, sawBegin: startupWatch.sawBegin,
           before: Math.max(0, (s.now - c.since) - quiet) };
}

//: THE LB'S OWN THRESHOLDS, not the page's taste. `cluster.py` adds "no progress for N s" to its
//: detail at START_QUIET_NOTE_S (120 s) and relaunches or halts the start at the step's limit (at most
//: START_STALL_S, 600 s); neither is on the wire, so they are restated here and the page turns amber
//: exactly when the planner starts saying so. Both are far clear of ONE log sample
//: (Config.cluster_logs_every_s, 15 s): steps arrive in bursts, and quiet shorter than a sample is
//: the sampler, never the engine.
const STARTUP_SLOW_S = 120;
const STARTUP_LONG_S = 300;
//: Past five minutes the clock is said louder — bold, the spinner stopped — and STILL amber. A quiet
//: clock is a timer, not an error: red is for `failed` alone (operator, 2026-10-10; a halted hang is
//: `stalled`, amber). Generation 188 sat six healthy minutes in a weight load painted red by this page.

const ORDINALS = ['', '', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th', '9th'];
const trailAgain = (n) => (n < 2 ? '' : `${ORDINALS[n] || `${n}th`} time`);

/** A quiet stretch, to the second under the hour: "47 s", "1 min 36 s", "13 min". `fmtAge` floors to
 *  whole minutes, which would print the steps BEFORE this one shorter than they really were (1 min
 *  36 s as "1 min") and so make the step it is stuck on look worse by the difference — the one
 *  direction a stall warning must never round. Past an hour it is `fmtAge` again. */
const fmtQuiet = (sec) => {
  const n = Math.max(0, Math.round(sec));
  if (n < 60) return `${n} s`;
  const m = Math.floor(n / 60);
  return m < 60 ? (n % 60 ? `${m} min ${n % 60} s` : `${m} min`) : fmtAge(n);
};

/** The trail: one row per step the head printed, oldest first, a repeat carrying its ordinal, and on
 *  the last row how long it has been there. No ETA, no percentage between samples, no row the log
 *  did not print. */
function startupTrail() {
  const w = startupNow;
  if (!w) return '';
  const { steps, quiet, taken, sawBegin, before } = w;
  const seen = new Map();
  const rows = steps.map((name, i) => {
    const n = (seen.get(name) || 0) + 1;
    seen.set(name, n);
    const again = trailAgain(n);
    const last = i === steps.length - 1;
    const tone = !last ? 'done'
      : `now${quiet >= STARTUP_LONG_S ? ' long' : (quiet >= STARTUP_SLOW_S ? ' slow' : '')}`;
    return `<li class="trail-step ${tone}"><i aria-hidden="true"></i>`
      + `<span class="trail-what">${esc(name)}</span>`
      + (again ? `<span class="trail-again" title="the ${esc(again)} this start reached this step">`
                 + `${esc(again)}</span>` : '')
      // What the load has taken so far, while docker shows no bar: the GiB the head's box has given
      // up since this start was first read (`loading.taken_gib`). Only a weight pass carries it.
      + (last && taken != null
        ? `<span class="trail-taken" title="GiB the head's box has given up since this start was first `
          + `read — the weights landing while the shard bar is out of docker's sight">${esc(taken)} GiB taken</span>`
        : '')
      // The clock, on the step it is on — but only once it says something. A page opened mid-step
      // did not watch it arrive, so all it can honestly report is "at least as long as I have been
      // looking", which starts at nothing; below the threshold at which the planner itself would
      // speak, a floor is not worth a number.
      + (last && quiet != null && (sawBegin || quiet >= STARTUP_SLOW_S)
        ? `<span class="trail-clock"><span class="vh">for </span>${sawBegin ? '' : '≥ '}`
          + `${esc(fmtQuiet(quiet))}</span>` : '')
      + '</li>';
  });
  // THE ONE SENTENCE THAT MAKES A QUIET START LOOK QUIET, and only once the planner would say it
  // too: what this step has cost against what every step before it cost. Generation 122 sat in
  // "capturing CUDA graphs" for 13 min where the whole boot before it took under two. A timer's
  // sentence: amber, louder past five minutes, never red.
  const note = quiet != null && quiet >= STARTUP_SLOW_S
    ? `<p class="trail-note ${quiet >= STARTUP_LONG_S ? 'long' : 'slow'}">`
      + `no new step for ${sawBegin ? '' : 'at least '}${esc(fmtQuiet(quiet))}`
      + (sawBegin && steps.length > 1
        ? ` &mdash; the ${steps.length - 1} before it came in ${esc(fmtQuiet(before))}` : '')
      + '</p>'
    : '';
  return `<div class="trail-head">the head's own steps <b>${steps.length}</b></div>`
    + `<ol class="trail" aria-label="the head rank's start, step by step, oldest first">`
    + `${rows.join('')}</ol>${note}`;
}

/** A HUNG START BEING RELAUNCHED, as one line over the trail: which attempt hung, in which step, for how long
 *  against that step's limit, which retry this is and where the LB put its evidence — `cluster.rescue`
 *  (cluster_loop.py `_rescue_view`). Only while the relaunch is on its way: once the pair serves, or has
 *  failed for good, the planner's own detail says what happened. */
const RELAUNCHING = ['finishing', 'restarting', 'checking', 'reserving', 'waiting', 'draining', 'freeing', 'starting'];
function rescueNote(c) {
  const r = (c && c.rescue) || null;
  if (!r || r.served || !RELAUNCHING.includes(String(c.state))) return '';
  return `<p class="trail-note slow">attempt ${esc(r.attempt)} hung in &ldquo;${esc(r.phase)}&rdquo; for `
    + `${esc(fmtQuiet(r.quiet_s))} (the limit for this step is ${esc(fmtQuiet(r.limit_s))}) &mdash; `
    + `relaunching (retry ${esc(r.attempt)}/${esc(r.retries)})`
    + (r.evidence ? ` &middot; its logs and stacks: ${esc(r.evidence)}` : '') + '</p>';
}

/** The trail has its own element OUTSIDE #cluster-state, for the reason the age does: that is a live
 *  region, and a clock that changes every poll would be announced every three seconds. */
function renderStartup(c) {
  const el = $('cluster-startup');
  if (!el) return;
  const html = rescueNote(c) + startupTrail();
  el.hidden = !html;
  paint(el, html);
}

// ── the sequence: where a start (or a stop) is, as a strip of steps ───────────
// The app's `ClusterTrack` (FleetStats/Shared/ClusterPanelModel.swift), in the planner's order
// (lb/vllm_lb/cluster.py `plan`): the GB10 boxes are freed of the classic fleet's vLLM, reserved through
// broker_parrot, their jobs drained; then both ranks launch and the head loads the checkpoint, warms up
// (compile, profiling, KV cache, kernels, CUDA graphs, API server) and serves once it answers /health.
// A stop is shorter: the ranks come down, then both reservations are released. Every step's status is
// read off fields /lb/status already carries — the state, the load passes, each GB10 box's reservation.
// Nothing is inferred from a clock.
const START_STEPS = ['free', 'reserve', 'drain', 'load', 'warm up', 'serve'];
const STOP_STEPS = ['unload', 'release', 'off'];

/** Each GB10 box as the step strip needs it: its reservation's status, whether its rank runs, its fleet
 *  jobs, what it has free and what is in its way. */
function boxFacts(c) {
  return Object.entries(c.boxes || {}).map(([box, o]) => {
    const b = o || {};
    const rank = b.rank && b.rank.present ? b.rank.running === true : null;
    const reservation = (b.reservation || {}).status || null;
    return { box, reservation, rank, jobs: isNum(b.running_jobs) ? b.running_jobs : null,
      free: isNum(b.mem_available_gib) ? b.mem_available_gib : null,
      blockers: (((b.hold || {}).blockers) || []).map((x) => x && x.message).filter(Boolean),
      done: rank === true || reservation === 'running' };
  });
}

/** The phases of `starting` that come after the checkpoint is read: nothing counts any more
 *  (cluster_startup.py `StartupPhase`). */
function isWarmUpPhase(phase) {
  const p = String(phase || '').toLowerCase();
  // Both engines' own opening phase, because the pair at :8010 is whichever one was launched: vLLM
  // says "starting vLLM" and SGLang "starting SGLang" (cluster_startup.py `StartupPhase`). Without
  // the second, an SGLang start's first phase answered "yes, warming up" — the engine's name, which
  // the page has no business knowing, deciding where the strip put the start.
  return !!p && !['loading ', 'starting vllm', 'starting sglang', 'starting tensorfold', 'initializ', 'connecting']
    .some((head) => p.startsWith(head));
}

/** The index in START_STEPS of the step a start — or a frozen, failed or stalled one — is in; null off a start. */
function startStepOf(c, boxes) {
  switch (String(c.state)) {
    case 'finishing': case 'checking': case 'blocked': case 'freeing': return 0;
    case 'reserving': case 'waiting': return 1;
    case 'draining': return 2;
    case 'starting': case 'restarting': case 'failed': case 'stalled': {
      const passes = loadPassesOf(c);
      if (passes.length && passes.every((p) => p.completed >= p.total) && isWarmUpPhase((c.loading || {}).phase)) return 4;
      // Failed with nothing read and a box not reserved: it never got as far as the ranks.
      // (A cluster that reserves nothing — no `reserve` step — fails where it fails, loading.)
      if (!passes.length && c.state === 'failed' && (topologyOf(c).steps || []).includes('reserve')
        && !boxes.every((b) => b.done)) return 1;
      return 3;
    }
    default: return null;
  }
}

/** Which sequence the cluster is on and each step's status (done / current / failed / stalled / pending);
 *  null for a state no sequence describes (`unknown`). `freeing` can come before OR after the reservations —
 *  the RAM wait is the later one — so while the current step is `free`, the reservations and the drain
 *  are read off the boxes rather than off the order. A `stalled` step is the one the hung start was halted
 *  in: amber, like the warning it is, never the red of a failure. */
function clusterTrack(c) {
  const state = String(c.state || 'unknown');
  const stop = { unloading: 0, releasing: 1 }[state];
  if (stop != null) {
    // A cluster that holds no reservation (its topology passes no `reserve` step: a node that reserves nothing, or the
    // broker off) has none to release: it unloads, and is off. One that reserves — the pair, a reserving local node —
    // returns it.
    const labels = (topologyOf(c).steps || []).includes('reserve') ? STOP_STEPS : STOP_STEPS.filter((l) => l !== 'release');
    return { kind: 'stop', current: stop,
      steps: labels.map((label, i) => ({ label, status: i < stop ? 'done' : (i === stop ? 'current' : 'pending') })) };
  }
  if (state === 'unknown') return null;
  const boxes = boxFacts(c);
  const current = state === 'off' || state === 'running' ? null : startStepOf(c, boxes);
  if (current == null && state !== 'off' && state !== 'running') return null;   // a state this page has not heard of
  const reserved = boxes.length > 0 && boxes.every((b) => b.done);
  const drained = reserved && boxes.every((b) => !(b.jobs > 0));
  // ONLY THE STEPS THIS TOPOLOGY PASSES THROUGH (`topology.steps`): no reserve or drain without the broker, no free for
  // a node this LB only observes — a step that cannot happen is never drawn, done or otherwise.
  const applies = topologyOf(c).steps;
  return { kind: 'start', current, boxes, steps: START_STEPS.map((label, i) => {
    let status = 'pending';
    if (state === 'running') status = 'done';
    else if (current == null) status = 'pending';                               // off: nothing has happened
    else if (i === current) status = state === 'failed' ? 'failed' : (state === 'stalled' ? 'stalled' : 'current');
    else if (i < current) status = 'done';
    else if (current === 0 && ((i === 1 && reserved) || (i === 2 && drained))) status = 'done';
    return { label, status };
  }).filter((st) => applies.includes(st.label)) };
}

/** The strip: a row of nodes joined by a line — done is green, the current one amber with a halo, a
 *  failed one red, the rest outlines — with the labels under the nodes. One image to a screen reader,
 *  named by every step and its status. */
function stepStrip(track) {
  if (!track) return '';
  const said = track.steps.map((st) => `${st.label}: ${st.status}`).join(', ');
  return `<ol class="steps" role="img" aria-label="${track.kind === 'stop' ? 'stop' : 'start'} sequence — ${esc(said)}">`
    + track.steps.map((st) => `<li class="step ${st.status}"><i></i><span>${esc(st.label)}</span></li>`).join('') + '</ol>';
}

/** One GB10 box while it is freed, reserved or drained: its mark (reserved; while draining, reserved and
 *  no job left), its name, the planner's summary and what it has free; then what is in its way. */
function boxFact(b, draining) {
  const ok = b.done && (!draining || !(b.jobs > 0));
  const summary = b.rank === true ? 'rank running'
    : (b.reservation === 'running' ? (b.jobs > 0 ? `reserved · ${b.jobs} job(s) draining` : 'reserved')
      : (b.reservation === 'queued' ? 'queued' : (b.reservation || 'not reserved')));
  return `<div class="fact">
    <i class="tick ${ok ? 'done' : 'wait'}"></i><b>${esc(b.box)}</b> <span class="fact-what">${esc(summary)}</span>
    ${b.free != null ? `<span class="fact-free">${b.free.toFixed(0)} GiB free</span>` : ''}
    ${b.blockers.map((m) => `<span class="fact-block">${esc(m)}</span>`).join('')}
  </div>`;
}

/** The ranks as one quiet line: which are up, and what each box has free. */
function ranksFact(boxes) {
  if (!boxes.length) return '';
  const up = boxes.filter((b) => b.rank === true);
  const all = up.length === boxes.length;
  const names = (up.length ? up : boxes).map((b) => b.box).join(' + ');
  const text = all ? `ranks up on ${names}` : (up.length ? `rank up on ${names}` : `ranks starting on ${names}`);
  const free = boxes.filter((b) => b.free != null).map((b) => b.free.toFixed(0));
  return `<div class="fact quiet"><i class="tick ${all ? 'done' : 'wait'}"></i><span class="fact-what">${esc(text)}</span>
    ${free.length ? `<span class="fact-free">${free.join(' &middot; ')} GiB free</span>` : ''}</div>`;
}

/** UNDER THE STRIP, ONLY WHAT THE CURRENT STEP IS ABOUT: the GB10 boxes while they are freed, reserved or
 *  drained; the load passes while the checkpoint is read, and the named phase while the head warms up,
 *  with the ranks. After a failure the same facts stay, where it failed. Everything the planner sees
 *  on each GB10 box is one disclosure further down (`#cluster-boxes`). */
function stepFacts(c, track) {
  if (!track || track.kind !== 'start' || track.current == null) return '';
  if (track.current <= 2) return `<div class="facts">${track.boxes.map((b) => boxFact(b, track.current === 2)).join('')}</div>`;
  return `${startupProgress(c)}<div class="facts">${ranksFact(track.boxes)}</div>`;
}

/** "42 s", "10 min", "3 h 12 min", "2 d 4 h" — the app's `elapsedText`. */
function fmtAge(sec) {
  const n = Math.max(0, Math.round(sec));
  if (n < 60) return `${n} s`;
  const m = Math.floor(n / 60);
  if (m < 60) return `${m} min`;
  const h = Math.floor(m / 60);
  if (h < 24) return m % 60 ? `${h} h ${m % 60} min` : `${h} h`;
  const d = Math.floor(h / 24);
  return h % 24 ? `${d} d ${h % 24} h` : `${d} d`;
}

/** HOW LONG IT HAS BEEN THAT: `s.now - cluster.since`, both on the LB's clock — a measurement, not an
 *  estimate, and nothing about how long is left. It has its own element OUTSIDE #cluster-state: that is a
 *  live region, and an age that ticks inside it would be announced on every poll. */
function renderSince(c, s) {
  const el = $('cluster-since');
  if (!el) return;
  const at = isNum(c.since) && isNum(s.now) ? c.since : null;
  el.hidden = at == null;
  if (at == null) return;
  el.title = `in this state since ${new Date(at * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
  paint(el, `<i class="clock" aria-hidden="true"></i><span class="vh">in this state for </span>${esc(fmtAge(s.now - at))}`);
}

/** THE STATE, ONCE: the state word as the headline with ONE diode, then the detail, the sequence and
 *  the current step's facts. The planner's `detail` is quoted rather than paraphrased — it is the only
 *  thing that knows WHY a start is waiting. Every state short of running or failed (parking, draining,
 *  starting, restarting, unloading, unparking, blocked, stalled, unknown, and any this page has not heard
 *  of yet) is "not there yet": amber. `failed` and `stalled` both end in the operator's hands, so both
 *  carry the "Stop, then Start again" cue; only `failed` is red. */
function clusterStateRow(c, s) {
  const state = String(c.state || 'unknown');
  let line = c.detail ? `<span class="detail">${esc(c.detail)}</span>` : '';
  if (state === 'off') {
    // What serves right now, from /lb/status's own leases and backends — never assumed.
    const running = ((s && s.leases) || []).filter((l) => l.status === 'running');
    const vllm = running.filter((l) => l.kind === 'vllm');
    const live = (l) => ((s && s.backends) || []).some((b) => b.kind === 'vllm' && b.live && b.host_label === l.host_label);
    line = vllm.length
      ? vllm.map((l) => `classic vLLM ${live(l) ? 'on' : 'starting on'} ${esc(l.box || l.host_label || '?')} `
        + `(node_job ${esc((l.job_id || '').slice(0, 8))})`).join(' + ')
      : 'no vLLM running';
  }
  if (state === 'running') {
    // The state word and the generation. WHAT it serves is the model badge under this row, from the
    // head's own /v1/models (`clusterModelInfo`, its served id in the badge's title); naming it here
    // too was the same model twice on one card.
    const gen = (c.effective || {}).generation;
    line = `<b>serving</b>${isNum(gen) ? ` &middot; generation ${esc(gen)}` : (line ? ` &middot; ${line}` : '')}`;
  }
  const ended = state === 'failed' || state === 'stalled';
  if (ended) line = `${line}${line ? ' &middot; ' : ''}<span class="next">Stop, then Start again</span>`;
  const track = clusterTrack(c);
  return `<div class="cp-headline">${diode({ tone: clusterTone(state), label: state })}<b class="cp-title">${esc(state)}</b></div>
    <div class="what${ended ? ` ${state}` : ''}">${line ? `<span class="cp-detail">${line}</span>` : ''}${stepStrip(track)}${stepFacts(c, track)}</div>`;
}
/** One GB10 box as the planner last observed it. Any field can be null — the box could not be
 *  asked — and null prints "—", never a zero or a "down" nobody measured. */
function clusterBoxRow(s, box, o) {
  const dash = '—';
  const val = (v, cls) => `<b${cls ? ` class="${cls}"` : ''}>${v == null ? dash : v}</b>`;
  // The box's RESERVATION as broker_parrot's rows say it is; the operator's park only as the reason a
  // queued one cannot be claimed, or as a note beside a running one broker_parrot keeps.
  const res = o.reservation;
  const job = res ? `node_job <span class="job">${esc((res.job_id || '').slice(0, 8) || dash)}</span>` : '';
  const parkedBy = o.park ? esc(o.park) : '';
  let park = '<span>not reserved</span>';
  if (reservedForCluster(res)) {
    park = `<span class="park">reserved by ${job} &middot; ${esc(res.status)}</span>`
      + (parkedBy ? ` <span class="park">parked by ${parkedBy}</span>` : '');
  } else if (res && res.status === 'queued') {
    park = parkedBy ? `<span class="park">parked by ${parkedBy} &middot; cannot claim &middot; ${job}</span>`
      : `<span class="park">queued &middot; ${job}</span>`;
  } else if (res) {
    park = `<span class="warn">${job} &middot; ${esc(res.status)}</span>`;
  } else if (parkedBy) {
    park = `<span class="park">parked by ${parkedBy}</span>`;
  }
  const vllm = o.broker_vllm_running === true ? 'up' : (o.broker_vllm_running === false ? 'down' : null);
  const mem = typeof o.mem_available_gib === 'number' ? `${o.mem_available_gib.toFixed(1)} GiB` : null;
  const r = o.rank;
  const rank = r == null ? '<span class="warn">rank unobserved</span>'
    : (!r.present ? '<span>no rank</span>'
      : (r.running ? `<span class="ok">rank gen ${esc(r.gen == null ? dash : r.gen)} running</span>`
        : `<span class="bad">rank exited (code ${esc(r.exit_code == null ? dash : r.exit_code)})</span>`));
  // WHY this box is not free for the pair yet, in the LB's own words — "ollama holds qwen3.8:latest,
  // 17.6 GiB", "77.0 GiB free, needs 103.2 GiB: short 26.2 GiB". The operator asked why a GB10 box was
  // waiting and the page could only say "77 GiB free, need 100" (worklog/vllm-cluster-hold-robustness.md
  // §6). Nothing is drawn for a box with nothing in its way.
  const blocked = (((o.hold || {}).blockers) || [])
    .map((b) => `<span class="cb-block">${esc(b.message)}</span>`).join('');
  return `<div class="cluster-box" data-box="${esc(box)}">
    <span class="cb-name">${esc(box)} <span class="cb-label">${esc(o.label || '')}</span></span>
    ${park}
    <span>fleet jobs ${val(typeof o.running_jobs === 'number' ? o.running_jobs : null)}</span>
    <span>broker vLLM ${val(vllm)}</span>
    <span>free ${val(mem)}</span>
    ${rank}
    ${gpuLine(o.gpu)}
    ${blocked}
    ${o.error ? `<span class="cb-err">${esc(o.error)}</span>` : ''}
  </div>`;
}

/** Write a server value into a control only while the operator is not editing it. The page
 *  re-renders every 3 s, and a poll that resets a half-typed number makes the control unusable. */
const pristine = (el) => !inUse(el) && el.dataset.dirty !== '1';

/** DOES THIS ROW'S CHECKPOINT START — its record over the last 7 UTC days (`/lb/status.start_rates`, vllm_lb/start_rates.py:
 *  of the LB's launches of it, `ok` served of `n` with a verdict, and per day [ok, n] or null). The figures — "1/6 · 17 %"
 *  — in the VERDICT's tone: `ok` every launch served, `bad` none did, `warn` between; `cells`, the days for `startBars`
 *  (null = a day with no launch). No launch: "no starts in 7 d" — or, when every launch was stopped before it served (no
 *  verdict) or refused before it launched, those counts. A log the LB could not read is `unknown`, with its reason as the
 *  hover and no days to draw — never "no starts". null: an LB that publishes no `start_rates` (nothing to draw). `text`
 *  is the same in the row's one line (`pickerRowText`). */
function startsOf(rates, key) {
  if (!rates || typeof rates !== 'object' || !Array.isArray(rates.days)) return null;
  const days = rates.days.map(String);
  if (rates.error) {
    return { tone: 'unknown', label: 'starts unknown', text: 'starts unknown', title: String(rates.error), cells: null,
      days };
  }
  const r = (rates.rows || {})[key] || {};
  const cells = days.map((_, i) => {
    const c = Array.isArray(r.days) ? r.days[i] : null;
    return Array.isArray(c) && isNum(c[0]) && isNum(c[1]) && c[1] > 0 ? c : null;
  });
  const ok = isNum(r.ok) ? r.ok : 0, n = isNum(r.n) ? r.n : 0;
  const apart = [r.stopped > 0 ? `${r.stopped} stopped` : '', r.refused > 0 ? `${r.refused} refused` : ''].filter(Boolean);
  const title = [`starts · ${days.length} d ${rates.tz || ''}`.trim(),
    ...cells.map((c, i) => (c ? `${days[i].slice(5)} ${c[0]}/${c[1]}` : '')).filter(Boolean), ...apart].join(' · ');
  if (!(n > 0)) {
    const label = apart.length ? apart.join(' · ') : `no starts in ${days.length} d`;
    return { tone: 'none', label, text: apart.length ? `starts ${label}` : label, title, cells, days };
  }
  const label = `${ok}/${n} · ${Math.round((100 * ok) / n)} %`;
  return { tone: ok >= n ? 'ok' : (ok > 0 ? 'warn' : 'bad'), label, text: `starts ${label}`, title, cells, days };
}

/** THE DAYS AS TINY BARS, one slot a day, oldest left. A day the LB launched the checkpoint is one full-height bar split
 *  by its verdicts — the share that SERVED green from the base, the share that FAILED red above it, a hairline of the
 *  ground between — so 0 % is an all-red bar, a mark, and a failing week reads as red at a glance. A day with no launch
 *  has no bar, only its empty slot on the base (never 0 %, never a guess). 100 %-stacked because the question is the
 *  share; the day's own counts are its hover, the window's are the figures under the bars. A share too thin to see keeps
 *  one unit of height. `data-days` spells the slots ("1/4", "-"). */
const START_BAR_W = 70, START_BAR_H = 16;
function startBars(cells, days) {
  const H = START_BAR_H, slot = START_BAR_W / cells.length, w = (slot - 2).toFixed(1);
  const f = (v) => v.toFixed(1);
  const bars = cells.map((c, i) => {
    const x = f(i * slot + 1);
    if (!c) return `<rect class="sb-none" x="${x}" y="${f(H - 1)}" width="${w}" height="1.0"/>`;
    const [ok, n] = c, gap = ok > 0 && ok < n ? 1 : 0;
    let good = ok > 0 ? (H * ok) / n : 0;
    if (gap) good = Math.min(Math.max(good, 1), H - gap - 1);
    const bad = ok < n ? H - good - gap : 0;
    return `<g><title>${esc(`${days[i].slice(5)} · ${ok}/${n} served`)}</title>`
      + (bad > 0 ? `<rect class="sb-bad" x="${x}" y="0.0" width="${w}" height="${f(bad)}"/>` : '')
      + (good > 0 ? `<rect class="sb-ok" x="${x}" y="${f(H - good)}" width="${w}" height="${f(good)}"/>` : '') + '</g>';
  }).join('');
  return `<svg class="start-bars" viewBox="0 0 ${START_BAR_W} ${H}" preserveAspectRatio="none" aria-hidden="true"`
    + ` data-days="${cells.map((c) => (c ? `${c[0]}/${c[1]}` : '-')).join(',')}">${bars}</svg>`;
}

/** A row's starts as drawn beside it: the day bars over the window's figures. '' for an LB without `start_rates`. */
function startsHtml(st) {
  if (!st) return '';
  return `<span class="mp-starts ${st.tone}" title="${esc(st.title)}"><span class="vh">starts, last ${st.days.length} days: </span>`
    + `${st.cells ? startBars(st.cells, st.days) : ''}<span class="mp-starts-n">${esc(st.label)}</span></span>`;
}

/** A PICKER ROW'S FACTS (operator, 2026-10-09: "make select more detailed, include tok/s targets if set"): the
 *  checkpoint, then what a Start of that row launches when nothing else is chosen (`cluster.setups[row]`, the SAME
 *  resolve as `cluster.defaults`) in the cluster card's own words — "32 seqs · 256k ctx · DFlash2 4" — and the rate
 *  to aim for there (`setup_targets[row]`): the best tenth of the minutes recorded at exactly that setup of exactly
 *  that checkpoint, or, marked "other setup", at another setup of it. No minute, no rate: a row never prints a
 *  number its checkpoint did not run.
 *
 *  ONE READ for two faces: `pickerRowText` joins these into the one line the hidden <select>'s option and every
 *  row's title carry, and `renderPicker` lays the same facts out as a row — the family as the group, the build as
 *  the name, the engine and the twin as badges, the figures as chips — so the two can never disagree on a number.
 *  The label's segments are the catalog's (" · "): the first names the family ("Qwen3.8 27B"), the rest the build
 *  ("dense NVFP4"); a segment that names the engine ("SGLang NEXTN") is the engine's, not the build's.
 *  `gone`: the boxes the checkpoint is missing on, by name ("missing on box-b"); a bare true from an older
 *  caller still reads "not on every GB10 box". `starts`: the row's start record (`startsOf`), or null from an LB that
 *  publishes none. */
function pickerRowFacts(m, uncensored, setup, target, gone, starts) {
  const lc = (m.launch || {}).cluster || {};
  const label = String(m.label || '');
  const parts = labelParts(label);
  const figures = [];
  let drafter = null;
  if (setup) {
    if (isNum(setup.max_num_seqs)) figures.push({ text: `${setup.max_num_seqs} seqs` });
    if (isNum(setup.max_model_len)) figures.push({ text: `${tokensText(setup.max_model_len)} ctx` });
    // A knob the ENGINE chooses itself (`launch.cluster.refuses`, TensorFold's prefill chunk and drafter depth) is no
    // figure of the setup: the budget is not drawn, and the drafter is the engine's own (`engine_drafts`).
    const own = lc.refuses || {};
    if (isNum(setup.max_num_batched_tokens) && !own.max_num_batched_tokens) {
      figures.push({ text: `prefill ${fmtInt(setup.max_num_batched_tokens)}` });
    }
    drafter = typeof lc.engine_drafts === 'boolean'
      ? (lc.engine_drafts ? { text: 'MTP · engine depth', off: false } : { text: 'no drafter', off: true })
      : setup.num_speculative_tokens > 0
        ? { text: `${drafterName(lc.speculative_method)} ${setup.num_speculative_tokens}`, off: false }
        : { text: 'no drafter', off: true };
    if (isNum(setup.kv_cache_memory) && setup.kv_cache_memory > 0) figures.push({ text: `KV ${gbOfBytes(setup.kv_cache_memory)}` });
  }
  const rate = target ? fmtNum(target.gen_tok_s).text : null;
  return {
    label,
    family: parts.family || String(m.id || ''),
    build: parts.build,
    engine: lc.engine ? engineName(lc.engine) : null,
    engineKey: lc.engine ? String(lc.engine).toLowerCase() : '',
    engineInLabel: parts.engineInLabel,
    uncensored: !!uncensored,
    // SIZED HOW on these cards (`resolution.sizing`): a model never measured here starts only by the operator's
    // explicit estimate — the row says so before anyone picks it.
    unmeasured: ((lc.resolution || {}).sizing === 'unmeasured'),
    drafter,
    size: typeof m.size_gib === 'number' ? `${m.size_gib} GiB` : null,
    figures,
    note: null,
    rate: rate ? { text: target.basis === 'setup' ? `target ${rate} tok/s` : `${rate} tok/s other setup`, other: target.basis !== 'setup' } : null,
    gone: Array.isArray(gone) ? (gone.length ? `missing on ${gone.join(', ')}` : null) : (gone ? 'not on every box' : null),
    starts: starts || null,
  };
}

/** The facts as ONE LINE: the <select>'s option text and every row's title. The engine is written only where the
 *  label does not already name it — every SGLang label does ("· SGLang NEXTN"), no vLLM one does — and the
 *  drafter sits where the card's words put it, between the prefill budget and the pool. */
function pickerRowText(m, uncensored, setup, target, gone, starts) {
  const f = pickerRowFacts(m, uncensored, setup, target, gone, starts);
  const parts = [f.label];
  if (f.engine && !f.engineInLabel) parts.push(f.engine);
  if (f.size) parts.push(f.size);
  if (f.uncensored) parts.push('uncensored');
  if (f.unmeasured) parts.push('not measured here · estimate only');
  const figs = f.figures.map((x) => x.text);
  const kvAt = figs.findIndex((t) => t.startsWith('KV '));
  const drafter = f.drafter ? [f.drafter.text] : [];
  parts.push(...(kvAt < 0 ? figs.concat(drafter) : [...figs.slice(0, kvAt), ...drafter, ...figs.slice(kvAt)]));
  if (f.rate) parts.push(f.rate.text);
  if (f.gone) parts.push(f.gone);
  if (f.starts) parts.push(f.starts.text);
  return parts.join(' · ');
}

/** THE MODEL PICKER: ONE ROW PER CHECKPOINT the cluster can launch — every ModelSpec, and right under each
 *  that names an uncensored twin (`models[].uncensored`) a second row for the twin, ` · uncensored` its
 *  distinguishing text (operator, 2026-10-07: the twin is its own row, not a tag plus a second control).
 *  Picking a row IS picking the checkpoint: a row's value is `catalog.availability_key` — `<id>` or
 *  `<id>@uncensored` — and `pickedRow` splits it into the (`model_choice`, `uncensored`) pair the switch stores,
 *  so the wire and the file keep the shape a running generation was stored in. Each row is marked with what the
 *  boxes HAVE for THAT checkpoint: one no GB10 box holds is offered DISABLED rather than hidden — the operator has
 *  to see that it exists and why it cannot be started — and the planner blocks a Start for it anyway. Twin rows
 *  appear only under an LB that takes the knob (`limits.uncensored`); an older one lists the models alone. */
function renderModelChoice(c, s, editable) {
  const el = $('cluster-model');
  if (!el) return;
  const models = Array.isArray(c.models) ? c.models : (((s || {}).models || {}).cluster || []);
  const boxes = Object.entries(c.boxes || {});
  // WHICH boxes lack the checkpoint (`availability[key].present === false`), by name: the row says which.
  const missingOn = (key) => boxes.filter(([, o]) => o && o.availability && o.availability[key]
    && o.availability[key].present === false).map(([box]) => box);
  const twins = takesRows(c);
  const setups = c.setups || {};
  const targets = figureOf(s, c, 'setup_targets') || {};
  // THE RUNNING ROW'S FIGURES ARE WHAT IT RUNS (operator 2026-10-30: "I selected other setup than default, why still
  // shows default in select?"): the row of the generation that is up (`cluster.effective`) shows its own seats, context,
  // budget, drafts and pool — not the catalog default a fresh Start of it would take. Every other row keeps its default.
  const runModel = runningOf(c, 'model_choice');
  const runKey = runModel ? rowKey(runModel, twins && runningOf(c, 'uncensored') === 1) : null;
  const runSetup = runKey ? Object.fromEntries(['max_num_seqs', 'max_model_len', 'max_num_batched_tokens',
    'num_speculative_tokens', 'kv_cache_memory'].map((k) => [k, runningOf(c, k)])) : null;
  const setupOf = (value) => (value === runKey ? Object.assign({}, setups[value] || {},
    Object.fromEntries(Object.entries(runSetup).filter(([, v]) => v !== null))) : setups[value] || null);
  const rows = [];
  for (const m of models) {
    for (const uncensored of (twins && m.uncensored ? [false, true] : [false])) {
      const value = rowKey(m.id, uncensored);
      // Only what THIS deployment offers (fleet `choices`, `cluster.offers`); null: every catalog row.
      if (Array.isArray(c.offers) && !c.offers.includes(value)) continue;
      const gone = missingOn(value);
      const target = targets[value] || null;
      // Each row's own start record, under the key the server counts it by (start_rates.py: the row key).
      const starts = startsOf((s || {}).start_rates, value);
      rows.push({ value, m, uncensored, gone, target,
        facts: pickerRowFacts(m, uncensored, setupOf(value), target, gone, starts),
        text: pickerRowText(m, uncensored, setupOf(value), target, gone, starts) });
    }
  }
  el.disabled = !editable;
  const key = rows.map((r) => `${r.value}:${r.text}`).join('|');
  if (el.dataset.choices !== key && !inUse(el)) {
    el.innerHTML = rows.map((r) => {
      const t = r.target;
      const title = t && isNum(t.windows) ? ` title="${esc(`${fmtInt(t.windows)} min · ${t.basis === 'setup' ? 'this setup' : 'other setup'}`)}"` : '';
      return `<option value="${esc(r.value)}"${r.gone.length ? ' disabled' : ''}${title}>${esc(r.text)}</option>`;
    }).join('');
    el.dataset.choices = key;
  }
  // The ✓: the row matching the stored pair. A switch from before the knob (null) marks the row a Start of
  // that model would resolve to — its default (`models[].uncensored.default`).
  const d = c.desired || {};
  const stored = typeof d.uncensored === 'boolean' ? d.uncensored : defaultOf(c, 'uncensored', d.model_choice) === 1;
  const want = d.model_choice ? (twins ? rowKey(d.model_choice, stored) : d.model_choice) : null;
  // …and a pick the deployment no longer offers (gone from `cluster.offers`) gives way to it too: left on the select it
  // read "no model" over a running pair (operator 2026-10-26).
  const offered = rows.some((r) => r.value === el.value);
  if ((pristine(el) || !offered) && want && rows.some((r) => r.value === want)) {
    el.value = want;
    if (!offered) delete el.dataset.dirty;
  }
  // THE RUNNING ROW: what the pair was launched with (`cluster.effective`), while a pair is up or on its way —
  // kept for the old generation through a restart, as every "running" figure on the card is (`runningOf`) —
  // tagged with the state's own word ("starting", "running", "failed").
  renderPicker(rows, el.value, runKey ? { key: runKey, word: String(c.state || 'running') } : null,
    editable);
  renderEstimated(rows, el.value);
}

/** THE ESTIMATE CONTROL (docs/design/engines.md, "Unmeasured combination"): a model never measured on this cluster's
 *  cards is refused by default; its row offers ONE checkbox to start it sized by the estimate instead, labelled with
 *  the server's own words (`resolution.refused`, `resolution.estimate`). It is the operator's choice each time: a
 *  different pick unchecks it, a measured model hides it, and nothing is sent unless it is checked. */
function unmeasuredOf(row) {
  const r = (((row || {}).m || {}).launch || {}).cluster || {};
  const res = r.resolution || {};
  return res.sizing === 'unmeasured' ? res : null;
}
function renderEstimated(rows, value) {
  const row = $('cluster-estimated-row');
  if (!row) return;
  const res = unmeasuredOf(rows.find((r) => r.value === value));
  const box = $('cluster-estimated');
  if (row.dataset.for !== String(value)) box.checked = false;     // a new pick: never carried over
  row.dataset.for = String(value);
  row.hidden = !res;
  if (res) $('cluster-estimated-why').textContent = `${res.refused || 'not measured here'} · ${res.estimate || ''}`;
}
/** Whether the Start carries `estimated: true`: the control is shown AND checked. */
const estimatedChosen = () => !!($('cluster-estimated-row') && !$('cluster-estimated-row').hidden
  && $('cluster-estimated') && $('cluster-estimated').checked);

/** A picker row's value for a (model, checkpoint) pair — `catalog.availability_key`, the key the boxes'
 *  availability answers are published under, so a row and its answer are looked up by one string. */
const rowKey = (model, uncensored) => (uncensored ? `${model}@uncensored` : model);
/** The pair a row's value names. */
function rowOf(value) {
  const v = String(value || '');
  return v.endsWith('@uncensored') ? { model: v.slice(0, -'@uncensored'.length), uncensored: true }
    : { model: v, uncensored: false };
}
/** What is picked: `{model, uncensored}`, or null while the picker is empty. */
function pickedRow() {
  const el = $('cluster-model');
  return el && el.value ? rowOf(el.value) : null;
}
/** Whether this LB takes the checkpoint with a Start (`cluster.limits.uncensored`): only then are twin rows
 *  offered and the pair sent — an older LB would accept the key and ignore it. */
const takesRows = (c) => Array.isArray((c.limits || {}).uncensored);

// ── the model picker: a select-only combobox over the hidden <select> ─────────
// Operator, 2026-10-10: "Let's not use standard select, use UX UI". A native <select> holds one line of text
// per row, so nine checkpoints read as nine 120-character lines with the figures lost past the arrow. The
// picker is the WAI-ARIA APG select-only combobox: a trigger (`#cluster-model-button`, role=combobox) showing
// the picked row, a listbox (`#cluster-model-list`) of rows grouped by model family, the keyboard on the
// trigger with `aria-activedescendant` naming the row it is on — focus never leaves the trigger.
//
// WHY THE <select> STAYS: it is the page's state holder. Its value (`<id>` or `<id>@uncensored`), its 'change'
// event and its `dataset.dirty` are what `pickedRow`, `shownOf`, `clusterChanges`, the field listeners, the
// estimate and the Start body read — none of them changed. Choosing a row here WRITES the <select> and fires
// the same 'change', so a keyboard pick, a tap and the old dropdown are one write path; it is `.vh`,
// aria-hidden and out of the tab order, so the combobox is the ONE control an assistive reader meets.
// The <select> also keeps every node test of the state semantics (uncensored_knob.node.js) true as written.

/** The picker's state: its rows as drawn (value, disabled, ✓, running, facts), whether the listbox is open,
 *  the row the keyboard is on (an index into `rows`) and the type-ahead buffer. The VALUE is the <select>'s. */
const picker = { rows: [], open: false, active: -1, typed: '', typedAt: 0 };
const pickerState = () => picker;
const PICKER_TYPEAHEAD_MS = 500;
const PICKER_PHONE = '(max-width: 600px)';     // gateway.css: under this the listbox is a sheet at the bottom
const optId = (i) => `cluster-model-opt-${i}`;

/** A row's two lines, from the one tag list (`modelTagList`): line 1 after the name carries the decisive kinds —
 *  the engine, the twin, the pair running it, the drafter — line 2 the size, the figures, the rate, a missing box.
 *  The trigger adds the build to line 1 (`arch`); a row's name IS its build, under the family's group label. */
const PICKER_LINE_1 = ['engine', 'twin', 'running', 'drafter'];
const PICKER_LINE_2 = ['size', 'fig', 'note', 'rate', 'gone'];
const pickerBadges = (f, running, kinds) => tagsHtml(modelTagList(f, { running }), kinds || PICKER_LINE_1);
const pickerChips = (f) => tagsHtml(modelTagList(f), PICKER_LINE_2);

/** Paint the trigger and the listbox from `rows` (`renderModelChoice`'s, in the <select>'s order). Both are
 *  rebuilt only when what they show changes (`paint`), so a poll never redraws a list the operator is moving
 *  through; the keyboard's mark is reapplied after any rebuild. */
function renderPicker(rows, value, running, editable) {
  const btn = $('cluster-model-button');
  const list = $('cluster-model-list');
  if (!btn || !list) return;
  picker.rows = rows.map((r, i) => ({ value: r.value, index: i, disabled: r.gone.length > 0, selected: r.value === value,
    running: running && r.value === running.key ? running.word : false, facts: r.facts, text: r.text, label: r.facts.label }));
  btn.disabled = !editable;
  if (!editable && picker.open) pickerClose({ focus: false });
  // The trigger: the picked row whole — family and build, badges, the figures — or that there is none.
  const picked = picker.rows.find((r) => r.selected) || null;
  const name = picked ? `<span class="mp-name"><b>${esc(picked.facts.family)}</b></span>` : '<span class="mp-name mp-none">no model</span>';
  paint(btn, `<span class="mp-value"><span class="mp-line">${name}${picked ? pickerBadges(picked.facts, picked.running, ['arch'].concat(PICKER_LINE_1)) : ''}</span>`
    + (picked ? `<span class="mp-chips">${pickerChips(picked.facts)}</span>` : '') + '</span><i class="mp-arrow" aria-hidden="true"></i>');
  btn.title = picked ? picked.text : '';
  // The listbox: the rows under their family, in order; a row's name is its build (the family is the group's).
  const groups = [];
  picker.rows.forEach((r) => {
    const last = groups[groups.length - 1];
    if (last && last.family === r.facts.family) last.rows.push(r);
    else groups.push({ family: r.facts.family, rows: [r] });
  });
  paint(list, groups.map((g, gi) => `<div class="mp-group" role="group" aria-labelledby="cluster-model-g${gi}">`
    + `<div class="mp-group-name" id="cluster-model-g${gi}">${esc(g.family)}</div>`
    + g.rows.map((r) => `<div class="mp-option${r.selected ? ' selected' : ''}${r.running ? ' running' : ''}${r.disabled ? ' gone' : ''}`
      + `${r.facts.starts ? ' has-starts' : ''}"`
      + ` id="${optId(r.index)}" role="option" aria-selected="${r.selected}"${r.disabled ? ' aria-disabled="true"' : ''}`
      + ` data-value="${esc(r.value)}" data-index="${r.index}" title="${esc(r.text)}">`
      + `<span class="mp-line"><span class="mp-check" aria-hidden="true"></span><span class="mp-name">${esc(r.facts.build || r.facts.family)}</span>`
      + `${pickerBadges(r.facts, r.running)}</span><span class="mp-chips">${pickerChips(r.facts)}</span>`
      // Beside both lines, in a column of its own: does this checkpoint start (`startsHtml`).
      + `${startsHtml(r.facts.starts)}</div>`).join('')
    + '</div>').join(''));
  // Re-marked WITHOUT scrolling: this runs on every status poll, and scrolling here pulled a list the operator had
  // scrolled back to the active row every few seconds (operator 2026-10-26, on a phone).
  if (picker.open) pickerMark({ scroll: false });
}

/** The enabled row next to `from` in direction `dir` (±1), or `from` itself at the end: no wrap, as a select. */
function pickerStep(from, dir) {
  for (let i = from + dir; i >= 0 && i < picker.rows.length; i += dir) if (!picker.rows[i].disabled) return i;
  return from;
}
/** The first (dir 1) or last (dir -1) enabled row, or -1. */
function pickerEdge(dir) {
  const rows = picker.rows;
  for (let i = dir > 0 ? 0 : rows.length - 1; i >= 0 && i < rows.length; i += dir) if (!rows[i].disabled) return i;
  return -1;
}
/** Where the keyboard is: the active row's mark, `aria-activedescendant` on the trigger, the row scrolled into view —
 *  unless `scroll` is false (a repaint: the keyboard did not move). */
function pickerMark({ scroll = true } = {}) {
  const btn = $('cluster-model-button');
  picker.rows.forEach((r, i) => {
    const o = $(optId(i));
    if (o) o.classList.toggle('active', picker.open && i === picker.active);
  });
  const on = picker.open && picker.active >= 0 ? $(optId(picker.active)) : null;
  if (btn) {
    if (on) btn.setAttribute('aria-activedescendant', optId(picker.active));
    else btn.removeAttribute('aria-activedescendant');
  }
  if (scroll && on && on.scrollIntoView) on.scrollIntoView({ block: 'nearest' });
}
/** Where the listbox opens on a desk: under the trigger, or above it when the room below is short and the room
 *  above is more — never past the viewport. On a phone (gateway.css `PICKER_PHONE`) it is a sheet at the bottom
 *  and sizes itself. */
function pickerPlace() {
  const btn = $('cluster-model-button');
  const list = $('cluster-model-list');
  if (!btn || !list || !btn.getBoundingClientRect) return;
  const phone = !!(window.matchMedia && window.matchMedia(PICKER_PHONE).matches);
  const r = btn.getBoundingClientRect();
  const vh = window.innerHeight || 0;
  if (phone || !isNum(r.bottom) || !isNum(r.top) || !vh) {
    list.classList.toggle('up', false);
    list.style.maxHeight = '';
    return;
  }
  const below = vh - r.bottom - 12;
  const above = r.top - 12;
  const up = below < 280 && above > below;
  list.classList.toggle('up', up);
  list.style.maxHeight = `${Math.max(160, Math.min(Math.floor(up ? above : below), 520))}px`;
}
function pickerOpen() {
  const btn = $('cluster-model-button');
  const list = $('cluster-model-list');
  if (!btn || !list || btn.disabled || picker.open) return;
  picker.open = true;
  picker.typed = '';
  const sel = picker.rows.findIndex((r) => r.selected);
  picker.active = sel >= 0 && !picker.rows[sel].disabled ? sel : pickerEdge(1);
  list.hidden = false;
  btn.setAttribute('aria-expanded', 'true');
  pickerPlace();
  pickerMark();
}
/** Close; `focus`: back on the trigger (a key, a tap in the list), or left where a click outside put it. */
function pickerClose({ focus = true } = {}) {
  if (!picker.open) return;
  const btn = $('cluster-model-button');
  const list = $('cluster-model-list');
  picker.open = false;
  picker.active = -1;
  picker.typed = '';
  if (list) list.hidden = true;
  if (btn) btn.setAttribute('aria-expanded', 'false');
  pickerMark();
  if (focus && btn && btn.focus) btn.focus({ preventScroll: true });
}
/** Choose row `i`: the <select> takes its value and fires the 'change' every listener on the page already has
 *  (dirty, re-render, the estimate) — the ONE write path, so a key and a tap are the same pick. Refused for a
 *  disabled row (false; the list stays open, as a select's does); the ✓ row again is only a close. */
function pickerChoose(i) {
  const r = picker.rows[i];
  if (!r || r.disabled) return false;
  const el = $('cluster-model');
  if (el && el.value !== r.value) {
    el.value = r.value;
    el.dispatchEvent(new Event('change', { bubbles: true }));
  }
  pickerClose({ focus: true });
  return true;
}
/** Type-ahead: the next enabled row whose label starts with what was typed, from the row after the active one;
 *  more letters within half a second REFINE (the active row keeps if it still matches — a select's way, not
 *  APG's example, which walks a row per letter); one letter repeated walks the rows starting with it. */
function pickerType(ch) {
  const now = Date.now();
  if (now - picker.typedAt > PICKER_TYPEAHEAD_MS) picker.typed = '';
  picker.typedAt = now;
  picker.typed += ch.toLowerCase();
  const rows = picker.rows;
  if (!rows.length) return;
  const same = picker.typed.length > 1 && [...picker.typed].every((c) => c === picker.typed[0]);
  const needle = same ? picker.typed[0] : picker.typed;
  const start = needle.length > 1 ? picker.active : picker.active + 1;
  for (let n = 0; n < rows.length; n += 1) {
    const i = (((start + n) % rows.length) + rows.length) % rows.length;
    if (!rows[i].disabled && rows[i].label.toLowerCase().startsWith(needle)) {
      picker.active = i;
      pickerMark();
      return;
    }
  }
}
/** The keyboard, on the trigger (WAI-ARIA APG select-only combobox). Closed: arrows, Enter, Space, Home, End
 *  and a letter open it (Home/End/a letter also move); open: arrows move over the enabled rows, Home/End to the
 *  ends, Enter/Space/Alt+Up choose, Tab chooses and moves on, Escape closes without choosing, a letter seeks. */
function pickerKeydown(e) {
  const k = e.key;
  if (e.ctrlKey || e.metaKey) return;
  const printable = typeof k === 'string' && k.length === 1 && !e.altKey;
  if (!picker.open) {
    if (['ArrowDown', 'ArrowUp', 'Enter', ' ', 'Home', 'End'].includes(k)) {
      e.preventDefault();
      pickerOpen();
      if (k === 'Home') picker.active = pickerEdge(1);
      if (k === 'End') picker.active = pickerEdge(-1);
      pickerMark();
    } else if (printable) {
      e.preventDefault();
      pickerOpen();
      pickerType(k);
    }
    return;
  }
  switch (k) {
    case 'ArrowDown': e.preventDefault(); picker.active = pickerStep(picker.active, 1); pickerMark(); return;
    case 'ArrowUp':
      e.preventDefault();
      if (e.altKey) pickerChoose(picker.active);
      else { picker.active = pickerStep(picker.active, -1); pickerMark(); }
      return;
    case 'Home': e.preventDefault(); picker.active = pickerEdge(1); pickerMark(); return;
    case 'End': e.preventDefault(); picker.active = pickerEdge(-1); pickerMark(); return;
    case 'Enter': e.preventDefault(); pickerChoose(picker.active); return;
    case ' ':
      e.preventDefault();
      if (picker.typed && Date.now() - picker.typedAt <= PICKER_TYPEAHEAD_MS) pickerType(' ');
      else pickerChoose(picker.active);
      return;
    case 'Escape': e.preventDefault(); pickerClose({ focus: true }); return;
    case 'Tab': pickerChoose(picker.active); return;          // not prevented: focus moves on
    default:
      if (printable) { e.preventDefault(); pickerType(k); }
  }
}
/** Wire the picker once: the trigger's click and keys, the list's taps, a click or Escape anywhere else. */
function initModelPicker() {
  const wrap = $('cluster-model-picker');
  const btn = $('cluster-model-button');
  const list = $('cluster-model-list');
  if (!wrap || !btn || !list) return;
  btn.addEventListener('click', () => (picker.open ? pickerClose({ focus: true }) : pickerOpen()));
  btn.addEventListener('keydown', pickerKeydown);
  list.addEventListener('mousedown', (e) => e.preventDefault());      // the trigger keeps focus through a click in the list
  list.addEventListener('click', (e) => {
    const t = e.target;
    const opt = t && t.closest ? t.closest('[role="option"]') : null;
    if (opt) pickerChoose(Number(opt.dataset.index));
    else if (!(t && t.closest && t.closest('.mp-group'))) pickerClose({ focus: true });   // the sheet's scrim, the list's own padding
  });
  document.addEventListener('pointerdown', (e) => {
    if (picker.open && !(wrap.contains && wrap.contains(e.target))) pickerClose({ focus: false });
  });
  document.addEventListener('keydown', (e) => {
    if (picker.open && e.key === 'Escape') { e.preventDefault(); pickerClose({ focus: true }); }
  });
  btn.addEventListener('focusout', (e) => {
    if (picker.open && !(e.relatedTarget && wrap.contains && wrap.contains(e.relatedTarget))) pickerClose({ focus: false });
  });
  window.addEventListener('resize', () => { if (picker.open) pickerPlace(); });
}

/** M8 — what the SELECTION costs on each box. While the operator is choosing, this is the LB's answer
 *  for the selection in the controls (`/lb/cluster/estimate`); otherwise it is the estimate the loop
 *  published for the stored one, which once the pair has launched also carries what the box really gave
 *  up. An estimate and an actual side by side is how a wrong constant shows itself. */
let estimateAsked = 0;
let estimateShown = null;       // {model_choice, boxes} from /lb/cluster/estimate, or null

function estimateLine(m) {
  const gib = (v) => (typeof v === 'number' ? `${v.toFixed(1)} GiB` : '—');
  const parts = [`needs ${gib(m.total_gib)}`];
  // What the PREFILL BUDGET is of that, when it is anything: the operator can read "8.8 of this is the
  // budget I set" off the row, instead of only off the warning under it.
  if (typeof m.activation_gib === 'number' && m.activation_gib > 0) {
    parts.push(`incl. ${gib(m.activation_gib)} prefill @ ${fmtInt(m.max_num_batched_tokens)}`);
  }
  if (typeof m.leaves_gib === 'number') parts.push(`leaves ${gib(m.leaves_gib)}`);
  if (typeof m.actual_used_gib === 'number') {
    const off = typeof m.deviation_pct === 'number'
      ? ` (${m.deviation_pct > 0 ? '+' : ''}${Math.round(m.deviation_pct)}%)` : '';
    parts.push(`actual ${gib(m.actual_used_gib)}${off}`);
  }
  const warn = (m.warnings || []).map((w) => `<span class="est-warn">${esc(w.message)}</span>`).join('');
  // A footprint nobody measured on a real start says so, beside the box: the figures are arithmetic.
  const uncal = m.calibration === '' ? ' <span class="est-uncal">uncalibrated</span>' : '';
  return `<div class="est-box"><b>${esc(m.box)}</b> <span class="est-role">${esc(m.role)}</span>${uncal}`
    + `<span>${esc(parts.join(' · '))}</span>${warn}</div>`;
}

function renderClusterEstimate(c) {
  const el = $('cluster-estimate');
  if (!el) return;
  const picked = pickedRow();
  const chosen = picked ? picked.model : (c.desired || {}).model_choice;
  // The freshly sized selection wins, but only while it is still the selection on screen.
  const fresh = estimateShown && estimateShown.model_choice === chosen ? estimateShown.boxes : null;
  const boxes = fresh || Object.values(c.boxes || {}).map((o) => (o || {}).memory).filter(Boolean);
  el.hidden = !boxes.length;
  // A warning EVERY box carries word for word (the pin's cap is one fact about the pair, published per
  // box) is drawn once, under the boxes — not under each; a box's own stays on its row.
  const texts = boxes.map((m) => (m.warnings || []).map((w) => w.message));
  const shared = boxes.length > 1 ? texts[0].filter((t) => texts.every((ts) => ts.includes(t))) : [];
  const own = boxes.map((m) => ({ ...m, warnings: (m.warnings || []).filter((w) => !shared.includes(w.message)) }));
  paint(el, own.map(estimateLine).join('')
    + shared.map((t) => `<div class="est-box est-shared"><span class="est-warn">${esc(t)}</span></div>`).join(''));
}

/** Ask the LB what the selection in the controls would cost. Only the newest answer is kept: a slow
 *  reply for a model the operator has already moved off must never overwrite a newer one. */
async function askEstimate() {
  const lc = theCluster(lastStatus) || {};
  if (localNode(lc)) return askLocalFit(lc);
  const picked = pickedRow();
  if (!picked) return;
  const seq = ++estimateAsked;
  const q = new URLSearchParams({ model_choice: picked.model });
  // The concurrency ON SCREEN, a value typed through "custom…" included (`shownOf`, as the Start reads it).
  const seqs = shownOf('max_num_seqs');
  const len = Number($('cluster-context').value);
  if (Number.isInteger(seqs) && seqs > 0) q.set('max_num_seqs', String(seqs));
  if (Number.isInteger(len) && len > 0) q.set('max_model_len', String(len));
  // The PREFILL BUDGET too: it is worth GiB per extra 1,024 tokens per box (the model's own slope), so an estimate without
  // it is the estimate that let 16,384 through onto boxes that could not hold it (2026-09-30).
  const budget = shownOf('max_num_batched_tokens');
  if (Number.isInteger(budget) && budget > 0) q.set('max_num_batched_tokens', String(budget));
  // …and the pool and the PLE placement on screen: both move GiB on every box, and `auto` is sized around them.
  const pool = shownOf('kv_cache_memory');
  if (pool === KV_AUTO || (Number.isInteger(pool) && pool > 0)) q.set('kv_cache_memory', String(pool));
  const ple = shownOf('ple_replicate');
  const c = (lastStatus && theCluster(lastStatus)) || {};
  // …and the KV type off the profile's own: the switch's own check runs on it (refused: no estimate is drawn).
  const dtype = takesKnob(c, 'kv_cache_dtype') ? shownOf('kv_cache_dtype') : null;
  if (dtype && dtype !== defaultOf(c, 'kv_cache_dtype', picked.model)) q.set('kv_cache_dtype', dtype);
  if ((ple === 0 || ple === 1) && hasPle(c, el.value)) q.set('ple_replicate', ple === 1 ? 'true' : 'false');
  let found = null;
  try {
    const r = await fetch(`${clusterRoute()}/estimate?${q}`, { cache: 'no-store' });
    if (r.ok) found = await r.json();
  } catch (_) {
    found = null;                       // a sizing that could not be had is simply not shown
  }
  if (seq !== estimateAsked) return;    // a newer selection is already out
  estimateShown = found ? { model_choice: found.model_choice, boxes: found.boxes || [], auto: found.auto || null }
    : null;
  if (lastStatus) renderCluster(lastStatus);
}

// ── the engine settings: what the pair is launched WITH ──────────────────────
// One field per vLLM variable. A switched-on cluster takes a change by RESTARTING the pair in place
// (`restart: true`: a new generation, and the GB10 boxes stay reserved — never handed back to the fleet
// between generations); a stopped one SAVES it for its next Start. Only the fields that differ from what
// is stored travel: absent is "keep" to the LB, so a field nobody touched can never be rewritten by a
// page that read the switch a poll ago.

/** What the LB launches with when it publishes no `cluster.defaults` — for ~2 s after it starts, or an
 *  LB older than the knobs — and the limits when it publishes none. Shown so a field is never blank;
 *  never sent unless the operator changed it. */
//: The last-resort defaults, for an LB too old to publish `cluster.defaults` for a field. Without an
//: entry here a control renders EMPTY and `shownOf` reads NaN, which the page then counts as a change
//: the operator never made — seen on the first screenshot of the pool knobs against the old gateway.
//: No pool here: a pool is ONE MODEL's (its profile's `kv_cache_memory`, `cluster.defaults`, the LB's
//: `recommended`) — a constant would be some model's pin offered as every model's default.
const CLUSTER_DEFAULTS = { max_num_seqs: 8, max_model_len: 262144, max_num_batched_tokens: 8192,
  num_speculative_tokens: 4, kv_offload_gib: 0, ple_replicate: 2 };
const CLUSTER_LIMITS = { max_num_seqs: [1, 48], max_model_len: [1024, 262144],
  max_num_batched_tokens: [2048, 65536], num_speculative_tokens: [0, 8],
  kv_cache_memory: [8000000000, 96000000000], kv_offload_gib: [0, 512], ple_replicate: [0, 1],
  uncensored: [0, 1] };
/** The KV pool, in BYTES as the engine takes it. Offered as round GiB because that is how the per-box
 *  estimate beside it is written; `fmtSetting` also says what each holds in tokens, which is the figure
 *  that decides concurrency (kv_tokens = bytes / the MODEL's bytes per token — `kvTokensOf`). */
const KV_STEPS = [16e9, 20e9, 24e9, 28e9, 32e9, 40e9, 48e9, 64e9];   // eight: one row of boxes (`syncSegments`)
/** The pool's third value: SIZED BY EACH START to what the boxes have left, keeping broker_parrot's floor
 *  under load (`kv_cache_memory: "auto"`, vllm_lb/cluster_defaults.py `kv_auto`). Offered only by an LB that
 *  publishes `desired.kv_cache_auto`; what it would pin now is the estimate's `auto`, asked as the operator picks. */
const KV_AUTO = 'auto';
/** Its option: "auto ≈ 36 GB" once the estimate has sized it, "auto" until then, "auto · not sizable now"
 *  when the estimate refused. What auto DOES is the select's title (index.html), not the option's words. */
function kvAutoLabel() {
  const a = (estimateShown || {}).auto;
  if (a && Number.isInteger(a.kv_cache_memory)) return `auto \u2248 ${fmtInt(Math.round(a.kv_cache_memory / 1e9))} GB`;
  return a && a.refusal ? 'auto \u00b7 not sizable now' : 'auto';
}
/** Disk for cold KV, in GB. 0 is off and off is the default: the path is not proven on this fleet. */
const OFFLOAD_STEPS = [0, 32, 64, 128, 256, 512];
/** The PLE placement as a TRI-STATE INTEGER so every control path stays generic: 2 is "as the model
 *  says" (sent as null), 1 the whole table on each rank, 0 its rows split across them. 2 and not -1
 *  because `wholeOf` accepts digits only — and loosening THAT to admit a minus sign would let a negative
 *  into every numeric knob on the page to save one line here. */
const PLE_DEFAULT = 2;
const PLE_STEPS = [PLE_DEFAULT, 1, 0];
/** WHICH CHECKPOINT of the model: 1 its uncensored twin, 0 the original — true/false on the wire. No third
 *  value: the LB resolves null to the model's own default at the Start and stores the bool, so what the
 *  control shows for a model is always one of the two (`models[].uncensored.default` names which). */
/** The twin a model's catalog entry names (`models[].uncensored`: path, default, calibration), or null. */
const twinOf = (c, id) => (modelOf(c, id) || {}).uncensored || null;
/** The knobs sized for ONE model: its pool, where its PLE table sits, its drafter's depth — and, since
 *  2026-10-05, the three that make up the rest of its START INPUT: the concurrency, the context and the
 *  prefill budget. The LB resets every one of them to the new model's defaults when the model changes and
 *  they are not sent with it (`DesiredStore.set`, `defaults_for`), so the page shows them at the PICKER
 *  model's own measured-best (`models[].launch.cluster.tuned`) and lists what resets. */
const MODEL_SCOPED = ['kv_cache_memory', 'ple_replicate', 'num_speculative_tokens', 'max_num_seqs',
  'max_model_len', 'max_num_batched_tokens', 'kv_cache_dtype'];
/** The two of them the switch ALWAYS holds as numbers: there is no null to reset them to, so a model change
 *  sends the new model's figure itself (`DesiredStore.set` reads null for these as "keep what is stored",
 *  which on a model change would keep the OLD model's seat count). The other four are nullable, and a reset
 *  of one travels as null so it follows the model's own default from then on. */
const MODEL_SCOPED_NUMBERS = ['max_num_seqs', 'max_model_len'];
/** A model's MEASURED-BEST start input, as the LB publishes it (`catalog.TunedSetup`): the value for `key`,
 *  or null where that model has none measured. This is what "default" means for a model that has one. */
function tunedOf(c, model, key) {
  const t = profileOf(c, model).tuned;
  const v = t && typeof t === 'object' ? t[key] : null;
  return Number.isInteger(v) ? v : null;
}
/** Which run measured it, for the note under the control; '' when the model has no tuned setup. */
function tunedSource(c, model) {
  const t = profileOf(c, model).tuned;
  return t && typeof t.source === 'string' ? t.source : '';
}
/** The prefill budgets offered as steps; any whole number within the limits may be typed instead. The pair takes
 *  those within its own limits (2,048 up); a one-host node the LB requests, those up to the chunk its catalog row's
 *  overhead was measured at (`renderLocalSettings`) — where the smaller ones are its KV lever. */
const BATCHED_STEPS = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536];
/** Concurrency is picked, not typed: nobody remembers which numbers make sense. The usual steps, within
 *  the LB's limits; the value in force is always added when it is off the list. */
const SEQS_STEPS = [1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 32, 48];
/** (Re)fill a <select> with numeric choices, keeping the operator's pick across polls: options are only
 *  rebuilt when the list changes and the control is not being used. `mark` names the default. */
function fillNumberSelect(el, steps, current, dflt) {
  if (el.value === CUSTOM) return;      // typing a value of their own: the list must not move under them
  const choices = steps.slice();
  if (Number.isInteger(current) && current > 0 && !choices.includes(current)) choices.push(current);
  choices.sort((a, b) => a - b);
  const key = `${choices.join(',')}|${dflt}`;
  if (el.dataset.choices !== key && !inUse(el)) {
    el.innerHTML = choices.map((v) => `<option value="${v}">${fmtInt(v)}${v === dflt ? ' \u00b7 default' : ''}</option>`).join('')
      + `<option value="${CUSTOM}">custom\u2026</option>`;
    el.dataset.choices = key;
    if (Number.isInteger(current)) el.value = String(current);
  }
}
/** Show the typed-value field only while "custom…" is picked, and lock it with its select. */
function syncCustom(el) {
  const c = customOf(el);
  if (!c) return;
  c.hidden = el.value !== CUSTOM;
  c.disabled = el.disabled;
}
/** The two knobs an older LB does not take. It would ACCEPT a POST carrying one and ignore the field, so a
 *  restart for it would reload the pair for nothing: each is offered only while `cluster.limits` names it,
 *  which is how an LB that takes it says so. For these two `null` is a value — the default — so a knob set
 *  back to its default is sent as null, and follows the default (a model's own drafter depth) from then on. */
const KNOBS = ['max_num_batched_tokens', 'num_speculative_tokens', 'kv_cache_memory', 'ple_replicate',
  'kv_offload_gib', 'kv_cache_dtype'];
const CLUSTER_FIELDS = [
  { key: 'model_choice', id: 'cluster-model', name: 'model' },
  { key: 'max_num_seqs', id: 'cluster-seqs', name: 'concurrency' },
  { key: 'max_model_len', id: 'cluster-context', name: 'context' },
  { key: 'max_num_batched_tokens', id: 'cluster-batched', name: 'prefill budget' },
  { key: 'num_speculative_tokens', id: 'cluster-spec', name: 'speculative tokens' },
  { key: 'kv_cache_memory', id: 'cluster-kv', name: 'KV pool' },
  { key: 'kv_cache_dtype', id: 'cluster-dtype', name: 'KV type' },
  { key: 'ple_replicate', id: 'cluster-ple', name: 'PLE table' },
  { key: 'kv_offload_gib', id: 'cluster-offload', name: 'KV offload' },
];
const fieldOf = (key) => CLUSTER_FIELDS.find((f) => f.key === key);
// The KV type is taken where the LB publishes its choices (`cluster.kv_cache_dtypes`, per model or node choice).
const takesKnob = (c, key) => (key === 'kv_cache_dtype'
  ? Object.keys(((c || {}).kv_cache_dtypes) || {}).length > 0
  : !KNOBS.includes(key) || Array.isArray((c.limits || {})[key]));
/** The KV dtypes the LB publishes for `id` (the pair's model, a local node's choice ref) on this server's cards:
 *  [{dtype, refused, source, attention_backends}] — or null. `depth`, the drafter depth the selection shows: above 0,
 *  each dtype's `refused_with_drafter` (the pair's, where the LB publishes it) is its refusal — SGLang's FP4 pools. */
const kvDtypesOf = (c, id, depth) => {
  const l = (((c || {}).kv_cache_dtypes) || {})[id];
  if (!Array.isArray(l) || !l.length) return null;
  return depth > 0 ? l.map((ch) => (typeof ch.refused_with_drafter === 'string'
    ? Object.assign({}, ch, { refused: ch.refused_with_drafter }) : ch)) : l;
};
/** The drafter depth the pair's selection launches `model` at: the field's while edited, else the stored or model's
 *  own; 0 for a model with no drafter. */
function draftDepthOf(c, model) {
  if (!hasDrafter(c, model)) return 0;
  const el = $('cluster-spec');
  const v = el && !pristine(el) ? wholeOf(el) : NaN;
  return Number.isInteger(v) ? v : storedOf(c, 'num_speculative_tokens', model);
}
/** THE KV TYPE ROW, for both topologies: the default first (`dflt`, worded `dfltLabel`), then every published dtype —
 *  a refused one DISABLED, the image's refusal its title (never hidden, never remapped); an accepted one's title its
 *  source and the attention backends that read its pool. No choices: no row. `note`, where given, is written under it. */
function fillKvDtype(choices, dflt, dfltLabel, current, enabled, note) {
  const el = $('cluster-dtype');
  if (!el) return;
  const row = $('cluster-dtype-row');
  if (row) row.hidden = !choices;
  el.disabled = !enabled || !choices;
  if (!choices) { delete el.dataset.dirty; return; }
  const title = (ch) => ch.refused || [ch.source, ch.attention_backends ? `attention ${Object.entries(ch.attention_backends)
    .map(([k, v]) => `${k} ${v}`).join(', ')}` : ''].filter(Boolean).join(' · ');
  paint(el, `<option value="${esc(dflt)}">${esc(dfltLabel)}</option>` + choices.filter((ch) => ch.dtype !== dflt)
    .map((ch) => `<option value="${esc(ch.dtype)}"${ch.refused ? ' disabled' : ''} title="${esc(title(ch))}">`
      + `${esc(ch.dtype)}${ch.refused ? ' · refused' : ''}</option>`).join(''));
  el.value = current;
  const n = $('cluster-dtype-note');
  if (n && note !== undefined) { paint(n, esc(note)); n.hidden = !note; }
}
/** A field's FACES: the control that holds its value and, for the model, the combobox the operator sees over
 *  the hidden <select> (`initModelPicker`) — both wear `changed` and `aria-invalid`, or neither. */
const facesOf = (id) => [$(id), id === 'cluster-model' ? $('cluster-model-button') : null].filter(Boolean);

// ── the preset rows as boxes ──────────────────────────────────────────────────
// Operator, 2026-10-24: "boxes with values next to each other to select from (allow select one) instead of select
// dropdown". THE <select> STAYS THE STATE HOLDER, as under the model picker: every reader (`wholeOf`, `pristine`, the
// Start body), listener and node test reads it. `syncSegments` mirrors its options into one radio group beside it —
// native radios, so arrows, Tab and the radiogroup role come free — and a pick writes the select and fires its
// 'change'. A MODE (the pair's pool "auto", a node's "% of card": a value that is no size) leads the row, apart.
const SEG_IDS = ['cluster-seqs', 'cluster-context', 'cluster-batched', 'cluster-spec', 'cluster-util', 'cluster-kv',
  'cluster-dtype', 'cluster-ple', 'cluster-offload'];
const segOf = (el) => (el && el.id ? $(`${el.id}-seg`) : null);
/** Is the operator on this control — its select, or a box of its group? A poll must not repaint under them. */
const inUse = (el) => {
  const seg = segOf(el);
  return document.activeElement === el || !!(seg && typeof seg.contains === 'function' && seg.contains(document.activeElement));
};
const SEG_MODES = new Set(['auto', '']);
function syncSegments() {
  SEG_IDS.forEach((id) => {
    const el = $(id);
    if (!el || typeof el.insertAdjacentElement !== 'function') return;
    let seg = segOf(el);
    if (!seg) {
      seg = document.createElement('div');
      seg.id = `${id}-seg`;
      seg.className = 'seg';
      seg.setAttribute('role', 'radiogroup');
      const label = document.querySelector(`label[for="${id}"]`);
      if (label) { if (!label.id) label.id = `${id}-label`; seg.setAttribute('aria-labelledby', label.id); }
      if (el.getAttribute('aria-describedby')) seg.setAttribute('aria-describedby', el.getAttribute('aria-describedby'));
      el.insertAdjacentElement('afterend', seg);
      el.classList.add('seg-held');
      seg.addEventListener('pointerdown', () => { seg.dataset.pointer = '1'; });
      seg.addEventListener('keydown', () => { delete seg.dataset.pointer; });
      seg.addEventListener('change', (e) => {
        if (e.target.name !== id) return;
        el.value = e.target.value;
        el.dispatchEvent(new Event('change', { bubbles: true }));
      });
    }
    const opts = [...el.options];
    const key = JSON.stringify([el.disabled, el.value, el.className, el.getAttribute('aria-invalid'),
      opts.map((o) => [o.value, o.text, o.disabled, o.title])]);
    if (seg.dataset.key === key) return;
    seg.dataset.key = key;
    const back = seg.contains(document.activeElement) ? document.activeElement.value : null;
    seg.classList.toggle('changed', el.classList.contains('changed'));
    seg.classList.toggle('invalid', el.getAttribute('aria-invalid') === 'true');
    seg.classList.toggle('off', el.disabled);
    const box = (o) => {
      // "20 GB · 363,563 tokens" / "8,192 · default": the value, then each qualifier on its own small line.
      const [head, ...rest] = o.text.split(' \u00b7 ');
      const cls = ['seg-box', SEG_MODES.has(o.value) ? 'seg-mode' : '', o.value === CUSTOM ? 'seg-custom' : '',
        rest.includes('default') ? 'seg-default' : ''].filter(Boolean).join(' ');
      return `<label class="${cls}"${o.title ? ` title="${esc(o.title)}"` : ''}><input type="radio" name="${esc(id)}"`
        + ` value="${esc(o.value)}"${o.value === el.value ? ' checked' : ''}${o.disabled || el.disabled ? ' disabled' : ''}>`
        + `<span class="seg-v">${esc(head)}</span>${rest.map((r) => r === 'default'   // the dot says it; the word stays for a reader
          ? '<span class="vh"> default</span>' : `<span class="seg-s">${esc(r)}</span>`).join('')}</label>`;
    };
    seg.innerHTML = opts.map(box).join('');
    if (back != null) {
      const r = [...seg.querySelectorAll('input')].find((i) => i.value === back);
      if (r) r.focus();
    }
  });
}

/** The LB's own range for `key` (`cluster.limits`), or the page's last resort. */
function rangeOf(c, key) {
  const l = (c.limits || {})[key];
  return Array.isArray(l) && Number.isInteger(l[0]) && Number.isInteger(l[1]) ? l : CLUSTER_LIMITS[key];
}

/** What the selection may be: the LB's range, NARROWED BY THE MODEL IT IS FOR — the picker's, unless one
 *  is named. A ModelSpec may be started only within its own ModelLimits (`models[].limits`), which the LB's
 *  switch enforces; the page mirrors it, so a number the LB would refuse for THAT model is refused here
 *  first. Every shipped model takes the LB's whole range today, so for them this narrows nothing. */
function limitsOf(c, key, model) {
  const range = rangeOf(c, key);
  if (key !== 'max_num_seqs' && key !== 'max_model_len') return range;
  const own = (modelOf(c, model || shownOf('model_choice') || storedOf(c, 'model_choice')) || {}).limits || {};
  let [lo, hi] = range;
  if (key === 'max_num_seqs' && Array.isArray(own.max_num_seqs) && own.max_num_seqs.length === 2
      && own.max_num_seqs.every(Number.isInteger)) {
    lo = Math.max(lo, own.max_num_seqs[0]);
    hi = Math.min(hi, own.max_num_seqs[1]);
  }
  if (key === 'max_model_len' && Number.isInteger(own.max_model_len)) hi = Math.min(hi, own.max_model_len);
  return [lo, hi];
}

/** A control's whole number as typed, or NaN for '', a fraction or anything else that is not one. */
const CUSTOM = 'custom';
/** The companion number field of a preset <select>: what the operator typed when they chose "custom…". */
const customOf = (el) => (el && el.tagName === 'SELECT' ? $(`${el.id}-custom`) : null);
const wholeOf = (el) => {
  const raw = el && el.value === CUSTOM && customOf(el) ? customOf(el).value : (el ? el.value : '');
  return /^\s*\d+\s*$/.test(raw) ? Number(raw) : NaN;
};

function modelOf(c, id) {
  const models = Array.isArray(c.models) ? c.models : ((((lastStatus || {}).models) || {}).cluster || []);
  return models.find((m) => m && m.id === id) || null;
}
const profileOf = (c, id) => ((modelOf(c, id) || {}).launch || {}).cluster || {};
/** Bytes one token of KV costs one rank — the MODEL's own (`models[].kv_bytes_per_token`); null when the
 *  LB does not say, and then no token figure is drawn rather than another model's. */
function kvBytesOf(c, id) {
  const v = (modelOf(c, id) || {}).kv_bytes_per_token;
  return isNum(v) && v > 0 ? v : null;
}
/** THE LARGEST PIN model `id`'s profile can take (`launch.cluster.kv_cache_memory_max`, the LB's
 *  LaunchProfile.kv_cache_memory_max — SGLang's calibrated mem_fraction bound), or null where it sets none. The LB
 *  refuses a pin past it at the switch, so the page greys those pins and refuses a stored one before the Start:
 *  generation 187 (2026-10-10) was started at 24 GB on a row whose maximum is 19.9, and looped for six minutes. */
function kvMaxOf(c, id) {
  const v = profileOf(c, id).kv_cache_memory_max;
  return isNum(v) && v > 0 ? v : null;
}
/** A pool bound as a figure: "19.9 GB" — to the tenth, because "20 GB" would read as a pin that fits. */
const gbTenth = (bytes) => (bytes / 1e9).toFixed(1);
/** How many tokens a pool of `bytes` holds for model `id` — the estimate before a start; null unknown. */
function kvTokensOf(c, id, bytes) {
  const per = kvBytesOf(c, id);
  return per && isNum(bytes) && bytes > 0 ? Math.floor(bytes / per) : null;
}
/** THE POOL model `id` runs with, as {bytes, tokens}: an engine that reserves it one window a seat (`pool_per_seat`,
 *  TensorFold) holds `seqs` x `ctx` tokens at the model's bytes a token, whatever pin is around; a shared pool is the
 *  pin's, at those bytes. Either null where the page does not know it — never another model's figure. */
function poolOf(c, id, pin, seqs, ctx) {
  if (profileOf(c, id).pool_per_seat) {
    const tokens = isNum(seqs) && seqs > 0 && isNum(ctx) && ctx > 0 ? seqs * ctx : null;
    const per = kvBytesOf(c, id);
    return { bytes: tokens != null && per ? tokens * per : null, tokens };
  }
  return { bytes: isNum(pin) && pin > 0 ? pin : null, tokens: kvTokensOf(c, id, pin) };
}
/** The pool the serving pair's ENGINE reports (`throughput.kv_pool.tokens`), or null. */
function enginePoolTokens(s, head) {
  // `head`: the cluster backend asking; left out, the SELECTED cluster's head (the detail card) — never "the first live".
  const b = head || headOf(s, theCluster(s));
  const t = (((b || {}).throughput || {}).kv_pool || {}).tokens;
  return isNum(t) && t > 0 ? t : null;
}
/** A profile that says `ple_replicate: null` has no PLE table: there is nothing to place. (An LB that
 *  does not publish the field leaves the control as it was.) */
const hasPle = (c, id) => profileOf(c, id).ple_replicate !== null;
/** `calibrated: false` — the model's footprint was never measured, so nothing is SIZED from it: no auto. */
const isCalibrated = (c, id) => (modelOf(c, id) || {}).calibrated !== false;
/** One box's estimate for model `id` — the freshly asked one, else the loop's — or null. It carries the
 *  model's own slopes and anchors, which the notes under the knobs are written from. */
function estimateOf(c, id) {
  const fresh = estimateShown && estimateShown.model_choice === id ? estimateShown.boxes : null;
  const boxes = fresh || Object.values(c.boxes || {}).map((o) => (o || {}).memory).filter(Boolean);
  return boxes.find((m) => m && m.model_choice === id) || null;
}
/** A model whose profile says `speculative: false` has no drafter to set: off is its only value. */
const hasDrafter = (c, id) => profileOf(c, id).speculative !== false;
/** Why model `id`'s ENGINE does not take `key` (`launch.cluster.refuses`, the LB's catalog.ENGINE_REFUSES), or ''.
 *  Such a knob is locked at what the engine runs, and a value stored for it is listed as the reset it needs: the LB
 *  refuses it rather than launch with it dropped (SGLang's pool, disk tier and PLE placement). */
const refusedBy = (c, id, key) => String((profileOf(c, id).refuses || {})[key] || '');

/** What the LB launches for `key` when nothing is stored. The drafter's depth is the MODEL's own:
 *  `cluster.defaults` resolves it for the stored model, and any other model's is its catalog profile's. */
function defaultOf(c, key, model) {
  const d = c.defaults || {};
  if (key === 'ple_replicate') return PLE_DEFAULT;      // "as the model says" IS the default
  // The KV type the profile launches with when the run names none (`launch.cluster.kv_cache_dtype`; null: vLLM's auto).
  if (key === 'kv_cache_dtype') return profileOf(c, model).kv_cache_dtype || 'auto';
  if (key === 'uncensored') {                           // the twin where the model has one and says so
    const t = twinOf(c, model);
    return t && t.default ? 1 : 0;
  }
  if (key === 'num_speculative_tokens') {
    const p = profileOf(c, model);
    if (p.speculative === false) return 0;              // no drafter: off is its only depth
    const tuned = tunedOf(c, model, key);
    if (tuned !== null) return tuned;                   // what THIS model's A/B measured
    if (model === (c.desired || {}).model_choice && Number.isInteger(d[key])) return d[key];
    if (Number.isInteger(p.num_speculative_tokens)) return p.num_speculative_tokens;
    return CLUSTER_DEFAULTS[key];
  }
  // The concurrency, the context and the prefill budget are the PICKER MODEL's too, when its catalog entry
  // records a measured-best one: `cluster.defaults` is resolved for the STORED model, so reading it for
  // another model in the picker is reading the previous model's tuning (which is exactly what the operator
  // had to undo by hand before 2026-10-05).
  if (key === 'max_num_seqs' || key === 'max_model_len' || key === 'max_num_batched_tokens') {
    const tuned = tunedOf(c, model, key);
    if (tuned !== null) return key === 'max_num_seqs' ? clampSeats(c, tuned, model) : tuned;
  }
  // The pool is the MODEL's too: the picker model's own profile, before `cluster.defaults` (which is
  // resolved for the STORED model, and is all an LB older than the profile field publishes), before
  // the LB's measured recommendation. Never a constant.
  if (key === 'kv_cache_memory') {
    const own = profileOf(c, model).kv_cache_memory;
    if (Number.isInteger(own)) return own;
    if (Number.isInteger(d[key])) return d[key];
    const rec = (c.recommended || {}).kv_cache_memory;
    return Number.isInteger(rec) ? rec : null;
  }
  const v = Number.isInteger(d[key]) ? d[key] : CLUSTER_DEFAULTS[key];
  // The concurrency offered as "default" is one the MODEL may be started with: inside its own limits.
  if (key === 'max_num_seqs' && model) return clampSeats(c, v, model);
  return v;
}

/** A concurrency inside `model`'s own limits — the LB refuses one outside them, so the page never offers one. */
function clampSeats(c, seats, model) {
  if (!model) return seats;
  const [lo, hi] = limitsOf(c, 'max_num_seqs', model);
  return Math.min(Math.max(seats, lo), hi);
}

/** Whether the switch holds a value of the operator's own for a model-scoped knob (null is the default). */
function storedOwn(c, key) {
  const d = c.desired || {};
  if (key === 'kv_cache_memory') return d.kv_cache_auto === true || Number.isInteger(d[key]);
  if (key === 'ple_replicate' || key === 'uncensored') return typeof d[key] === 'boolean';
  if (key === 'kv_cache_dtype') return typeof d[key] === 'string' && d[key] !== '';
  return Number.isInteger(d[key]);
}

/** What the next launch of `model` uses for `key`: the stored value, or — null is the default — that. */
function storedOf(c, key, model) {
  const v = (c.desired || {})[key];
  if (key === 'model_choice') return typeof v === 'string' && v ? v : null;
  // ANOTHER MODEL in the picker: what is stored for a model-scoped knob was chosen for the stored model
  // and does not ride across (the LB resets it) — the next launch of `model` uses its own default.
  const of = (c.desired || {}).model_choice;
  if (MODEL_SCOPED.includes(key) && model && of && model !== of) return defaultOf(c, key, model);
  if (key === 'kv_cache_dtype') return typeof v === 'string' && v ? v : defaultOf(c, key, model);
  // The PLE placement is true/false on the wire and null for "as the model says"; the controls carry it
  // as -1/1/0 so that every path from here to the confirm line stays the one the numeric knobs use.
  if (key === 'ple_replicate') return typeof v === 'boolean' ? (v ? 1 : 0) : PLE_DEFAULT;
  // The checkpoint is a bool once a Start resolved it; null (a switch from before it) is the model's default —
  // and so is ANOTHER model's in the picker: what is stored was chosen for the stored model (the LB resets it).
  if (key === 'uncensored') {
    if (model && of && model !== of) return defaultOf(c, key, model);
    return typeof v === 'boolean' ? (v ? 1 : 0) : defaultOf(c, key, model);
  }
  if (key === 'kv_cache_memory' && (c.desired || {}).kv_cache_auto === true) return KV_AUTO;
  return Number.isInteger(v) ? v : defaultOf(c, key, model);
}

/** What the RUNNING pair was launched with (`cluster.effective`) — kept for the old generation while a
 *  restart replaces it — or null when no pair runs, or the LB does not say. */
function runningOf(c, key) {
  if (key === 'ple_replicate' || key === 'uncensored') {
    const r = (c.effective || {})[key];
    return typeof r === 'boolean' ? (r ? 1 : 0) : null;
  }
  const e = c.effective;
  const v = e && typeof e === 'object' ? e[key] : null;
  if (key === 'model_choice' || key === 'kv_cache_dtype') return typeof v === 'string' && v ? v : null;
  return Number.isInteger(v) ? v : null;
}

function shownOf(key) {
  const el = $(fieldOf(key).id);
  if (!el) return null;
  if (key === 'kv_cache_memory' && el.value === KV_AUTO) return KV_AUTO;
  if (key === 'kv_cache_dtype') return el.value || null;
  // The picker's value names a ROW (model + checkpoint); as a model it is the id part (`rowOf`).
  return key === 'model_choice' ? (el.value ? rowOf(el.value).model : null) : wholeOf(el);
}

function fmtSetting(c, key, v, model) {
  if (v == null || Number.isNaN(v)) return '—';
  if (key === 'model_choice') return (modelOf(c, v) || {}).label || String(v);
  if (key === 'uncensored') return v ? 'uncensored' : 'original';
  if (key === 'kv_cache_dtype') return String(v);
  if (key === 'ple_replicate') {
    // "as the model says", not "model default": these words are read INSIDE sentences that already carry
    // the word default — the field note writes "default <words>" and the option list marks the default
    // one "<words> · default", so the old wording printed "default model default" (operator, 2026-10-05).
    return v === PLE_DEFAULT ? 'as the model says' : v ? 'whole table on each rank' : 'split across ranks';
  }
  // The pool in the unit the ENGINE FLAG takes (--kv-cache-memory is decimal bytes, and the shipped pin
  // is exactly 28,000,000,000), then in the one that decides concurrency. Labelling the decimal steps as
  // binary GiB read as an off-by-one — 28 GB is 26.1 GiB, which is what the box estimate beside it says.
  if (key === 'kv_cache_memory' && v === KV_AUTO) return kvAutoLabel();
  if (key === 'kv_cache_memory') {
    // …in the tokens of the model it is FOR (default: the picker's); no figure when its cost is unknown.
    const tok = kvTokensOf(c, model || shownOf('model_choice') || storedOf(c, 'model_choice'), v);
    return `${fmtInt(Math.round(v / 1e9))} GB${tok == null ? '' : ` · ${fmtInt(tok)} tokens`}`;
  }
  if (key === 'kv_offload_gib') return v ? `${fmtInt(v)} GB on disk` : 'off';
  if (key === 'num_speculative_tokens') return v === 0 ? 'off' : String(v);
  return fmtInt(v);
}

/** The two knobs that travel as true/false: 1/0 in the controls, a bool on the wire. */
const BOOL_KNOBS = ['ple_replicate', 'uncensored'];

/** Every field whose control differs from what the next launch would use anyway — all a POST carries
 *  besides the switch. A stored null is resolved for the model IN THE PICKER, so a drafter depth left at
 *  its default follows the model rather than being pinned to the last one's. `send` is what travels: a knob
 *  set back to its default goes as null. */
function clusterChanges(c) {
  const model = shownOf('model_choice') || storedOf(c, 'model_choice');
  const was = storedOf(c, 'model_choice');
  const out = [];
  for (const f of CLUSTER_FIELDS) {
    if (!takesKnob(c, f.key)) continue;
    const from = storedOf(c, f.key, model);
    const to = shownOf(f.key);
    // A MODEL CHANGE RESETS what was stored for the old model's own knobs. Each one that held a value is
    // LISTED — from what it was, for that model — and sent: null where the field shows the new model's
    // default (the reset), the value where the operator set one for the new model.
    if (MODEL_SCOPED.includes(f.key) && was && model !== was && storedOwn(c, f.key)) {
      const stored = storedOf(c, f.key, was);
      if (MODEL_SCOPED_NUMBERS.includes(f.key)) {
        // Always a number on the wire, and listed only where the two models really differ: both are measured
        // at the full 262,144 window, so a model change must not claim the context as a change.
        if (to === stored) continue;
        out.push({ ...f, from: stored, fromModel: was, to, reset: false, moved: true, send: to });
        continue;
      }
      const reset = to === defaultOf(c, f.key, model);
      out.push({ ...f, from: stored, fromModel: was, to, reset, moved: true,
        send: reset ? null : (BOOL_KNOBS.includes(f.key) ? to === 1 : to) });
      continue;
    }
    if (f.key === 'model_choice' ? !to || to === from : to === from) continue;
    const reset = KNOBS.includes(f.key) && to === defaultOf(c, f.key, model);
    // The wire takes the PLE placement and the checkpoint as true/false; a reset goes as null, which the LB
    // reads as the model's own default (and, for the checkpoint, resolves and stores at the Start).
    const send = reset ? null : (BOOL_KNOBS.includes(f.key) ? to === 1 : to);
    out.push({ ...f, from, to, send, reset });
  }
  // WHICH CHECKPOINT travels with the picked ROW (`pickedRow`), not with a control: listed when the row's checkpoint
  // is not what the next launch of that model would use anyway (the stored one, or the model's default after a model
  // change), and sent as the bool it is — the row IS the choice, so there is no reset to null.
  if (takesRows(c)) {
    const row = pickedRow();
    if (row) {
      const to = row.uncensored ? 1 : 0;
      const from = storedOf(c, 'uncensored', row.model);
      if (to !== from) {
        out.push({ key: 'uncensored', id: 'cluster-model', name: 'checkpoint', from, to, send: to === 1,
          reset: false, moved: model !== was });
      }
    }
  }
  return out;
}

/** Why the selection cannot be sent, in words, before the LB is asked — `{key, why}`, or null. It
 *  mirrors `cluster.limits`; the LB still checks everything, and its own refusal is shown as it words it. */
function selectionProblem(c, changes) {
  const model = shownOf('model_choice') || storedOf(c, 'model_choice');
  for (const ch of changes) {
    if (ch.key === 'model_choice' || ch.reset) continue;   // a reset sends null, which no range can refuse
    if (ch.key === 'kv_cache_dtype') continue;          // checked below, changed or not, at the depth shown
    const [lo, hi] = limitsOf(c, ch.key);
    if (Number.isInteger(ch.to) && ch.to >= lo && ch.to <= hi) continue;
    if (ch.key === 'kv_cache_memory' && ch.to === KV_AUTO) continue;   // the LB sizes it, in range
    let why = `${ch.name} must be a whole number from ${fmtInt(lo)} to ${fmtInt(hi)} tokens`;
    if (ch.key === 'max_num_seqs') why = `concurrency must be a whole number from ${lo} to ${hi}`;
    if (ch.key === 'num_speculative_tokens') why = `speculative tokens must be a whole number from ${lo} (off) to ${hi}`;
    return { key: ch.key, why };
  }
  // THE KV TYPE, in the image's own words as the switch would refuse it — at the DEPTH shown, so a drafter turned back
  // on under a stored FP4 pool is refused as surely as an FP4 pick beside one.
  const kvCh = changes.find((ch) => ch.key === 'kv_cache_dtype');
  const dtype = kvCh ? (kvCh.reset ? null : kvCh.to) : (takesKnob(c, 'kv_cache_dtype') ? shownOf('kv_cache_dtype') : null);
  if (dtype) {
    const why = ((kvDtypesOf(c, model, draftDepthOf(c, model)) || []).find((x) => x.dtype === dtype) || {}).refused;
    if (why) return { key: 'kv_cache_dtype', why: `KV type ${dtype}: ${why}` };
  }
  // Checked whether or not they changed: what is stored for one model may be past ANOTHER model's own
  // limits, and a model change alone must not carry it into a POST the LB would refuse.
  const seats = shownOf('max_num_seqs');
  const [sLo, sHi] = limitsOf(c, 'max_num_seqs', model);
  if (!(Number.isInteger(seats) && seats >= sLo && seats <= sHi)) {
    return { key: 'max_num_seqs', why: `concurrency must be a whole number from ${sLo} to ${sHi}` };
  }
  const len = shownOf('max_model_len');
  const [cLo, cHi] = limitsOf(c, 'max_model_len', model);
  if (!(Number.isInteger(len) && len >= cLo && len <= cHi)) {
    return { key: 'max_model_len', why: `context must be a whole number from ${fmtInt(cLo)} to ${fmtInt(cHi)} tokens` };
  }
  // …and a depth stored for one model is refused for a model with no drafter.
  const drafts = shownOf('num_speculative_tokens');
  if (takesKnob(c, 'num_speculative_tokens') && drafts > 0 && !hasDrafter(c, model)) {
    return { key: 'num_speculative_tokens',
      why: `${fmtSetting(c, 'model_choice', model)} has no speculative decoding: set speculative tokens to off` };
  }
  // …and a pool past the most this model's profile can pin — stored by an older LB, or chosen for another model —
  // is the switch's refusal (409, the launch's own words): said here, with the maximum, before the Start is held.
  const pool = shownOf('kv_cache_memory');
  const max = kvMaxOf(c, model);
  if (takesKnob(c, 'kv_cache_memory') && Number.isInteger(pool) && max !== null && pool > max) {
    return { key: 'kv_cache_memory',
      why: `${fmtSetting(c, 'model_choice', model)}: a ${fmtInt(Math.round(pool / 1e9))} GB KV pool is past the most this `
        + `profile can pin: max ${gbTenth(max)} GB (${fmtInt(max)} B)` };
  }
  // …and NOT ENOUGH RAM on a box at rest — the launch gate's own refusal (needed_now_gib against what nothing the LB
  // can wait for holds), said HERE from the estimate of the selection on screen, before the Start is held: generation
  // 249 was stored, reserved both GB10 boxes and only then refused (operator, 2026-10-26: "tell before I hit start").
  const fresh = estimateShown && estimateShown.model_choice === model ? estimateShown.boxes : null;
  const short = memoryProblem(fresh);
  if (short) return { key: takesKnob(c, 'kv_cache_memory') ? 'kv_cache_memory' : 'max_num_seqs', why: short };
  return null;
}

/** The boxes of an estimate (GET …/estimate `boxes`) a Start would be refused on — AT REST (nothing the LB can wait
 *  for holds their RAM) and short of `needed_now_gib` — in one line of figures, or ''. Its rank standing, a box is not
 *  at rest: a restart is the loop's to wait for. */
function memoryProblem(boxes) {
  const short = (boxes || []).filter((b) => b && b.at_rest === true && typeof b.available_gib === 'number'
    && typeof b.needed_now_gib === 'number' && b.available_gib < b.needed_now_gib);
  if (!short.length) return '';
  return 'not enough RAM: ' + short.map((b) => `${b.box} ${b.available_gib.toFixed(1)} GiB free at rest, needs `
    + `${b.needed_now_gib.toFixed(1)} (${b.total_gib.toFixed(1)} + ${b.start_headroom_gib.toFixed(1)} reserve): short `
    + `${(b.needed_now_gib - b.available_gib).toFixed(1)} GiB`).join('; ')
    + ' — lower the KV pool, prefill budget, concurrency or context';
}

/** One field's note: its default, and — while a pair runs — what it RUNS with, where that is not what the
 *  field shows: "running 8 · pending 16 after restart" once a restart is stored, "running 8" beside an edit
 *  not sent yet. `model` is the picker's, which the default follows; pending is the STORED switch's. */
function fieldNote(c, key, model, on) {
  const bits = [];
  if (key !== 'model_choice') {
    const shown = esc(fmtSetting(c, key, defaultOf(c, key, model), model));
    // A default that was MEASURED says so, and carries the run it was measured on as its tooltip: a number
    // the page fills in for the operator must be able to say where it came from (`catalog.TunedSetup.source`).
    const src = tunedOf(c, model, key) !== null ? tunedSource(c, model) : '';
    bits.push(src ? `<span title="${esc(src)}">default ${shown} &middot; measured</span>` : `default ${shown}`);
  }
  if (!takesKnob(c, key)) return bits.concat('not taken by this LB').join(' &middot; ');
  const run = runningOf(c, key);
  const runs = runningOf(c, 'model_choice') || storedOf(c, 'model_choice');   // whose tokens "running" is in
  const kept = storedOf(c, 'model_choice');
  const stored = storedOf(c, key, kept);
  // A stored `auto` that resolved to the size running is the running pool, not a change waiting for a restart.
  const resolved = stored === KV_AUTO && run === (c.desired || {}).kv_cache_memory;
  if (run != null && stored != null && run !== stored && !resolved) {
    bits.push(`running ${esc(fmtSetting(c, key, run, runs))}`, '<span class="cs-pending">pending '
      + `${esc(fmtSetting(c, key, stored, kept))} ${on ? 'after restart' : 'at next start'}</span>`);
  } else if (run != null && shownOf(key) !== run) {
    bits.push(`running ${esc(fmtSetting(c, key, run, runs))}`);
  }
  if (key === 'num_speculative_tokens' && !hasDrafter(c, model)) bits.push('no drafter');
  const refused = refusedBy(c, model, key);
  if (refused) {
    bits.push(`<span title="${esc(refused)}">not taken by ${esc(engineName(profileOf(c, model).engine) || 'this engine')}</span>`);
  }
  if (key === 'kv_cache_memory' && !isCalibrated(c, model)) bits.push('uncalibrated &middot; no auto');
  // The most this model's profile can pin ("max 19.9 GB"), where it has a bound: the pins past it are greyed above.
  if (key === 'kv_cache_memory' && kvMaxOf(c, model) !== null) bits.push(`max ${gbTenth(kvMaxOf(c, model))} GB`);
  // The model's own ceiling, where it is under the LB's ("max 12"): nothing for a model that takes the whole range.
  if (key === 'max_num_seqs' && limitsOf(c, key, model)[1] < rangeOf(c, key)[1]) {
    bits.push(`max ${limitsOf(c, key, model)[1]}`);
  }
  return bits.join(' &middot; ');
}

const changeKey = (changes) => changes.map((ch) => `${ch.key}=${ch.send}`).join('&');
function changeList(c, changes) {
  return changes.map((ch) => `${ch.name} ${fmtSetting(c, ch.key, ch.from, ch.fromModel)} → ${fmtSetting(c, ch.key, ch.to)}`
    + (ch.reset ? (ch.moved ? ' (reset)' : ' (default)') : '')).join('; ');
}

/** Every draw of the settings ends with their boxes mirroring the selects they stand for. */
function renderClusterSettings(c, s) {
  drawClusterSettings(c, s);
  syncSegments();
}
function drawClusterSettings(c, s) {
  if (localNode(c)) return renderLocalSettings(c, s);
  // The pair's rows are its own, whatever a node's engine hid on another tab.
  HOUSE_ROWS.forEach(([, id]) => { const row = settingRowOf(id); if (row && id !== 'cluster-util-row' && id !== 'cluster-dtype-row') row.hidden = false; });
  if ($('cluster-engine-fields')) $('cluster-engine-fields').hidden = true;
  kvRowWords(false);
  const d = c.desired || {};
  const seqs = $('cluster-seqs');
  const ctx = $('cluster-context');
  // Editable whenever something can take a change: a switched-on cluster restarts the pair at it, a stopped
  // one saves it for its next Start. A cluster on its way down — or a switch not read yet — has neither.
  const editable = !!c.desired && (d.mode === 'cluster' || c.state === 'off');
  seqs.disabled = !editable;
  ctx.disabled = !editable;
  $('cluster-hint').hidden = editable;
  if (!editable) {
    CLUSTER_FIELDS.forEach((f) => { const el = $(f.id); if (el) delete el.dataset.dirty; });
    closeConfirm();
  }
  renderModelChoice(c, s, editable);
  const model = (pickedRow() || {}).model || d.model_choice;

  // The steps offered are the PICKER MODEL's own range, and NOTHING PAST IT is offered — not even what the
  // switch holds: a stored 16 shown to a model that takes 8 was a number the LB would refuse, waiting to
  // be sent. What is stored past the range is CLAMPED to it, and the clamp is listed as the change it is
  // (`clusterChanges`: "concurrency 16 → 8"), so what a Start sends is always what is on screen.
  const [lo, hi] = limitsOf(c, 'max_num_seqs', model);
  const clamp = (v) => (Number.isInteger(v) ? Math.min(Math.max(v, lo), hi) : v);
  const seqsDefault = defaultOf(c, 'max_num_seqs', model);
  // WHAT THE NEXT START OF THE PICKER MODEL USES (`storedOf`): the stored concurrency for the model the
  // switch holds, and the OTHER model's own measured-best the moment another is picked — the LB resets it
  // the same way (`DesiredStore.set`), so the control shows what would really be sent, not the seat count
  // tuned for the model before it.
  const seqsWant = storedOf(c, 'max_num_seqs', model);
  const seqsNow = clamp(pristine(seqs) ? seqsWant : wholeOf(seqs));
  fillNumberSelect(seqs, SEQS_STEPS.filter((v) => v >= lo && v <= hi), seqsNow, seqsDefault);
  if (pristine(seqs) && Number.isInteger(seqsWant)) seqs.value = String(clamp(seqsWant));
  else if (seqs.value !== CUSTOM && Number.isInteger(seqsNow) && seqs.value !== String(seqsNow)) seqs.value = String(seqsNow);
  const typed = customOf(seqs);
  if (typed) {
    typed.min = String(lo);
    typed.max = String(hi);
  }

  // The context the same way: the steps within the model's own window, the window's top itself as a
  // step, and a stored context past it clamped to it.
  const [cLo, cHi] = limitsOf(c, 'max_model_len', model);
  const stored = Number(pristine(ctx) ? storedOf(c, 'max_model_len', model) : ctx.value);
  const current = Number.isInteger(stored) && stored > 0 ? Math.min(Math.max(stored, cLo), cHi) : stored;
  const choices = CONTEXT_CHOICES.filter((v) => v >= cLo && v <= cHi);
  if (Number.isInteger(cHi) && !choices.includes(cHi)) choices.push(cHi);
  // The CURRENT value is always offered, even off the step list: a select that cannot show what the
  // cluster is set to would quietly post a different number on the next Start.
  if (Number.isInteger(current) && current > 0 && !choices.includes(current)) choices.push(current);
  choices.sort((a, b) => a - b);
  const key = choices.join(',');
  if (ctx.dataset.choices !== key && document.activeElement !== ctx) {
    ctx.innerHTML = choices.map((v) => `<option value="${v}">${fmtInt(v)}</option>`).join('');
    ctx.dataset.choices = key;
    if (Number.isInteger(current)) ctx.value = String(current);
  }
  if (pristine(ctx) && Number.isInteger(current)) ctx.value = String(current);

  renderKnobs(c, editable, model);
  [seqs, $('cluster-batched')].forEach((el) => el && syncCustom(el));
  renderClusterActions(c);
  renderSettingsDisclosure(c);

  // The number nobody typed. With nothing stored the concurrency above IS the measured maximum for
  // the ranks' KV pool (vllm_lb/cluster_defaults.py, from worklog/vllm-cluster-ram-headroom.md), so
  // the page says what fits and at which context — and shows the server's own sentence instead when
  // a write was clamped down to it. The ceiling is NOT recomputed here: one place owns that rule.
  const note = $('cluster-recommended');
  if (note) {
    // THE POOL AS NUMBERS, from the settings the pair runs (`effective`, else `desired`): GB, tokens,
    // and how many requests fit at the configured context and at the sizes real sessions reach.
    // Computed from the OPERATOR'S pin, so a 35 GB pool is not described as a 28 GB one.
    // The pool falls back to the LB's default when nothing is running and no pin was chosen — a stopped
    // or cold cluster still gets its numbers, as it got "measured maximum" before this line replaced it.
    // WHOSE pool: the running pair's (`effective`), with the tokens its ENGINE reports; a pair on its way
    // up, the stored selection's; and before a Start the selection ON SCREEN — the picker's model, at its
    // own bytes per token. A model whose cost the LB does not publish gets its GB and no token figure.
    const dflt = c.defaults || {};
    const live = c.effective || (d.mode === 'cluster' ? d : null);
    const whole = (key, fallback) => (Number.isInteger(shownOf(key)) && shownOf(key) > 0 ? shownOf(key) : fallback);
    const of = live ? (live.model_choice || d.model_choice) : model;
    const pool = live
      ? (live.kv_cache_memory || profileOf(c, of).kv_cache_memory || dflt.kv_cache_memory
        || (c.recommended || {}).kv_cache_memory)
      : whole('kv_cache_memory', d.kv_cache_memory || defaultOf(c, 'kv_cache_memory', of));
    const ctx = live ? (live.max_model_len || dflt.max_model_len) : whole('max_model_len', d.max_model_len || dflt.max_model_len);
    const seqs = live ? (live.max_num_seqs || dflt.max_num_seqs) : whole('max_num_seqs', d.max_num_seqs || dflt.max_num_seqs);
    // The pool's numbers always; then the VERDICT — the server's own clamp sentence when it has one
    // (one place owns that rule), and this page's arithmetic only to fill the gap, since the server's
    // clamp lives in memory and is gone after an LB restart.
    let msg = '';
    const sized = poolOf(c, of, pool, seqs, ctx);
    if (isNum(ctx) && ctx > 0 && (sized.bytes != null || sized.tokens != null)) {
      const tok = (c.effective && enginePoolTokens(s)) || sized.tokens;
      msg = poolLine(sized.bytes, tok, ctx, c.clamp ? null : seqs);
    }
    if (c.clamp) msg = msg ? `${msg} — ${c.clamp}` : c.clamp;
    note.textContent = msg;
    note.hidden = !msg;
  }
}

/** THE POOL AS NUMBERS: "KV pool 18 GB · 1,378,465 tokens · fits 5 @ 262,144 · 13 @ 100k · 27 @ 50k" — its pin in
 *  GB where it has one, its tokens, how many requests of the window and of the sizes real sessions reach it seats, and
 *  `seqs` past that. A pool the engine sized itself (no pin: a node's --gpu-memory-utilization) is its tokens alone. */
function poolLine(bytes, tok, ctx, seqs) {
  const gb = isNum(bytes) && bytes > 0 ? `${fmtInt(Math.round(bytes / 1e9))} GB` : '';
  if (tok == null) return gb ? `KV pool ${gb}` : '';
  const at = (n) => Math.max(1, Math.floor(tok / n));
  return `KV pool ${gb ? `${gb} · ` : ''}${fmtInt(tok)} tokens · fits ${at(ctx)} @ ${fmtInt(ctx)}`
    + ` · ${at(100000)} @ 100k · ${at(50000)} @ 50k`
    + (isNum(seqs) && seqs > at(ctx) ? ` · ${seqs} seats > ${at(ctx)}` : '');
}

/** THE SETTINGS OF A NODE THIS LB ONLY OBSERVES (`control: none`): the pair's own group, showing what the node SAYS it
 *  runs with — `cluster.effective`, built from its HTTP API alone (topology.py `observed_effective`) — and what its
 *  FLEET FILE declares it runs with (`cluster.declared`: the catalog row, the seats, the image and the parsers), checked
 *  against the node wherever the node says it too (`declared.checks`, topology.py `cross_check`). Every field is locked,
 *  because there is no switch to send a change to, and every note is the figure's SOURCE (`observedSource`): `node`,
 *  `node = fleet` where both say it alike, `declared by the fleet (not exposed by the node)`, `not exposed by the node`
 *  — never a default dressed as a setting. A disagreement keeps the node's figure and warns with the fleet's, and the
 *  card says it loudly (`renderMismatch`). No Apply, no defaults, no estimate; the facts line carries what the node
 *  says that has no field, the declared line what only the fleet says. Hidden while the node does not answer: nothing
 *  is known then. */
const NOT_EXPOSED_BY_NODE = 'not exposed by the node';
const DECLARED_BY_FLEET = 'declared by the fleet (not exposed by the node)';
/** The fleet's declaration checked against the node — key → {fleet, node, verdict} — or {} where there is none. */
const declaredChecks = (c) => (((c || {}).declared || {}).checks) || {};
/** A declared key as the page names it, and a declared value as the page writes it. */
const DECLARED_NAMES = { served_name: 'served name', max_model_len: 'context', gpu_memory_utilization: 'GPU memory',
  max_num_seqs: 'seats', reasoning_parser: 'reasoning parser', tool_call_parser: 'tool-call parser', image: 'image',
  max_num_batched_tokens: 'prefill budget' };
const declaredName = (key) => DECLARED_NAMES[key] || key;
const declaredText = (key, v) => (key === 'gpu_memory_utilization' && isNum(v) ? v.toFixed(2) : (isNum(v) ? fmtInt(v) : String(v)));
/** The declared keys that have a FIELD of their own; every other one the node does not expose is on the declared line. */
const DECLARED_FIELDS = ['max_num_seqs', 'max_model_len', 'model', 'max_num_batched_tokens'];

/** THE CATALOG ROW a node this LB only observes is DECLARED to be — its fleet file's `model` (`cluster.declared`) — or
 *  null: the pair (its row is its launch record's), an undeclared node, a row the catalog does not list. The node's
 *  identity comes from here alone: its served name is its own, and its launch is its server file's. */
function declaredRowOf(s, view) {
  const d = ((view || theCluster(s)) || {}).declared;
  return d && d.model_choice ? catalogOf(s).find((m) => m && m.id === d.model_choice) || null : null;
}

/** `key`'s value and note on a node's field, `nodeValue` being what the node says: the node's (`node`), both alike
 *  (`node = fleet`), the fleet's alone (`declared by the fleet …`), or — they disagree — the node's figure, with the
 *  fleet's as the warning. */
function observedSource(c, key, nodeValue) {
  const ck = declaredChecks(c)[key];
  if (!ck) return { value: nodeValue, note: nodeValue == null ? NOT_EXPOSED_BY_NODE : 'node', warn: '' };
  if (ck.verdict === 'agrees') return { value: nodeValue, note: 'node = fleet', warn: '' };
  if (ck.verdict === 'mismatch') return { value: nodeValue, note: 'node', warn: `≠ fleet ${declaredText(key, ck.fleet)}` };
  return { value: ck.fleet, note: DECLARED_BY_FLEET, warn: '' };
}

/** A one-host node's field nobody moves here: its one value shown, locked, its note the source, its warning
 *  (`warnText`) shown only when there is one. */
function lockField(id, shown, note, warnText) {
  const el = $(id);
  if (!el) return;
  el.disabled = true;
  delete el.dataset.dirty;
  delete el.dataset.choices;
  paint(el, `<option value="">${esc(shown == null ? '—' : shown)}</option>`);
  const custom = customOf(el);
  if (custom) custom.hidden = true;
  const n = $(`${id}-note`);
  if (n) { paint(n, esc(note)); n.hidden = !note; }
  const warn = $(`${id}-warn`);
  if (warn && warnText) paint(warn, esc(warnText));
  if (warn) warn.hidden = !warnText;
}

function renderObservedSettings(c, s) {
  const e = c.effective;
  $('cluster-settings').hidden = !e;
  if (!e) return;
  paint($('cluster-settings-summary'), 'Settings');
  ['cluster-apply', 'cluster-defaults', 'cluster-estimate', 'cluster-hint'].forEach((id) => { $(id).hidden = true; });
  paint($('cluster-changes'), '');
  closeConfirm();
  const field = lockField;
  const whole = (v) => (isNum(v) ? fmtInt(v) : null);
  const put = (id, key, nodeValue) => {
    const x = observedSource(c, key, nodeValue);
    field(id, whole(x.value), x.note, x.warn);
    return x.value;
  };
  const seats = put('cluster-seqs', 'max_num_seqs', e.max_num_seqs);
  put('cluster-batched', 'max_num_batched_tokens', e.max_num_batched_tokens);
  put('cluster-context', 'max_model_len', e.max_model_len);
  field('cluster-spec', isNum(e.num_speculative_tokens) ? (e.num_speculative_tokens ? String(e.num_speculative_tokens) : 'off')
    : null, e.num_speculative_tokens == null ? NOT_EXPOSED_BY_NODE : 'node');
  const tok = enginePoolTokens(s);
  const pinned = isNum(e.kv_cache_memory) && e.kv_cache_memory > 0;
  field('cluster-kv', pinned ? fmtSetting(c, 'kv_cache_memory', e.kv_cache_memory) : (tok == null ? null : `${fmtInt(tok)} tokens`),
    pinned || tok != null ? (pinned ? 'node' : 'node · not pinned') : NOT_EXPOSED_BY_NODE);
  field('cluster-offload', null, NOT_EXPOSED_BY_NODE);
  const ple = $('cluster-ple');
  const pleRow = ple && ple.closest ? ple.closest('.chooser') : null;
  if (pleRow) pleRow.hidden = true;
  fillKvDtype(null);                    // an observed node's launch is its own: no KV type to choose
  // The model: the catalog row its fleet declares, else its own words (its served name) — one row, the running one,
  // locked. Its engine is the node's own word and it has no drafter or size from the row: the row's launch profile
  // and checkpoint are the pair's, and the node's server file launched it.
  const d = c.declared || null;
  const m = declaredRowOf(s);
  const own = { engine: engineName(e.engine), engineKey: String(e.engine || '').toLowerCase(), drafter: null, size: null };
  const name = d ? d.model_choice : e.served_name;
  const facts = Object.assign(m ? pickerRowFacts(m, e.uncensored === true, null, null, [])
    : { label: name || '', family: name || '—', build: '', engineInLabel: false, uncensored: e.uncensored === true,
      figures: [], note: null, rate: null, gone: null }, own);
  const select = $('cluster-model');
  if (select) {
    select.disabled = true;
    paint(select, `<option value="observed">${esc(facts.label || facts.family)}</option>`);
    select.value = 'observed';
  }
  renderPicker([{ value: 'observed', gone: [], facts, text: facts.label || facts.family }], 'observed',
    { key: 'observed', word: String(c.state || 'running') }, false);
  const modelNote = $('cluster-model-note');
  if (modelNote) { paint(modelNote, esc(d ? DECLARED_BY_FLEET : 'node')); modelNote.hidden = false; }
  // The pool line: the ENGINE's own pool at the window it serves — no GB where it pinned none — and the seats.
  const note = $('cluster-recommended');
  if (note) {
    const msg = isNum(e.max_model_len) && e.max_model_len > 0 ? poolLine(e.kv_cache_memory, tok, e.max_model_len, seats) : '';
    note.textContent = msg;
    note.hidden = !msg;
  }
  renderNodeFacts(c);
}

/** A one-host node's two lines under its settings: what it says it runs with that has no field — `= fleet` where its
 *  fleet (or the intent its host applied) says it alike, `≠ fleet …` where not — and what only the fleet says. */
function renderNodeFacts(c) {
  const e = c.effective || {};
  // What the node says that has no field — `= fleet` where its fleet says it alike, `≠ fleet …` where not.
  const mark = (key, text) => {
    const ck = declaredChecks(c)[key];
    if (ck && ck.verdict === 'agrees') return `${esc(text)} = fleet`;
    if (ck && ck.verdict === 'mismatch') return `<span class="cs-bad">${esc(text)} ≠ fleet ${esc(declaredText(key, ck.fleet))}</span>`;
    return esc(text);
  };
  const factsEl = $('cluster-node-facts');
  if (factsEl) {
    const line = [e.engine ? esc(engineName(e.engine)) : null,
      isNum(e.gpu_memory_utilization) ? mark('gpu_memory_utilization', `GPU memory ${e.gpu_memory_utilization.toFixed(2)}`) : null,
      e.kv_cache_dtype ? esc(`KV ${e.kv_cache_dtype}`) : null,
      typeof e.prefix_caching === 'boolean' ? esc(`prefix caching ${e.prefix_caching ? 'on' : 'off'}`) : null,
      e.checkpoint ? mark('checkpoint', e.checkpoint) : null,
      e.served_name ? mark('served_name', `served ${e.served_name}`) : null].filter(Boolean);
    paint(factsEl, line.join(' &middot; '));
    factsEl.hidden = !line.length;
  }
  // What only the fleet says: every declared key the node does not expose that has no field of its own.
  const declEl = $('cluster-node-declared');
  if (declEl) {
    const only = Object.entries(declaredChecks(c))
      .filter(([key, ck]) => ck.verdict === NOT_EXPOSED_BY_NODE && !DECLARED_FIELDS.includes(key))
      .map(([key, ck]) => esc(`${declaredName(key)} ${declaredText(key, ck.fleet)}`));
    paint(declEl, only.length ? `${esc(DECLARED_BY_FLEET)}: ${only.join(' &middot; ')}` : '');
    declEl.hidden = !only.length;
  }
}

/** THE FLEET AND THE NODE DISAGREE (`declared.checks`, verdict `mismatch`): every such key, the fleet's value then the
 *  node's, on the card itself — loud, outside the folded settings — and never resolved: the field keeps the node's
 *  figure (what runs), the fleet file its own (what the server file was told to run). Hidden on the pair and while
 *  nothing disagrees. */
function renderMismatch(c) {
  const el = $('cluster-mismatch');
  if (!el) return;
  const bad = Object.entries(declaredChecks(c)).filter(([, ck]) => ck.verdict === 'mismatch')
    .map(([key, ck]) => esc(`${declaredName(key)} ${declaredText(key, ck.fleet)} ≠ ${declaredText(key, ck.node)}`));
  paint(el, bad.length ? `<b>fleet ≠ node</b> &middot; ${bad.join(' &middot; ')}` : '');
  el.hidden = !bad.length;
}

/** The lines only a node this LB observes draws — its facts, its declaration, its model's source — hidden on the pair. */
const OBSERVED_LINES = ['cluster-node-facts', 'cluster-node-declared', 'cluster-model-note'];

// ── A NODE THE LB REQUESTS AND ITS HOST APPLIES (fleet control `local`, vllm_lb/node_control.py) ─────────────
// The operator, 2026-10-11: "load, unload, select model, parameters". Start and Stop are the pair's hold button; the
// model is the pair's picker over the FLEET's choices (`cluster.choices`); seats, context, the prefill budget (vLLM's
// own default, or a chunk up to the one the engine was measured at) and the share of the card
// (`gpu_memory_utilization`, a field only this topology shows) are editable; every change is priced on the card in the
// fleet's own numbers before it is sent (`cluster.fit`, GET /lb/cluster/estimate: fleet.launch_fit) and a refusal is
// shown in the fleet's words, verbatim. The LB WRITES an intent; the host's applier applies it — so the applier line
// says who listens and what it last did, and nothing is offered while nobody listens.

/** The lines only a node the LB requests draws — hidden on every other card. */
const LOCAL_LINES = ['cluster-util-row', 'cluster-fit', 'cluster-applier', 'cluster-force', 'cluster-alert'];
const localNode = (c) => topologyOf(c).control === 'local';
/** What a field offers — within the selected model's catalog limits — beside the value it holds. */
const SEAT_PRESETS = [1, 2, 4, 6, 8, 12, 16, 24, 32, 48];
const CONTEXT_PRESETS = [8192, 16384, 32768, 65536, 98304, 131072, 196608, 262144];
/** 0.97-1.0: the operator, 2026-10-11. The fit prices each on the card's own memory; whether vLLM finds that much FREE
 *  at start is vLLM's to say (its boot refusal, in the applier line). */
const UTIL_PRESETS = [0.7, 0.75, 0.8, 0.85, 0.88, 0.9, 0.92, 0.95, 0.97, 0.98, 0.99, 1];
/** The local node's fields: the key the switch takes (fleet.MOVABLE), the element, the word in a change list. */
const LOCAL_FIELDS = [
  { key: 'model', id: 'cluster-model', name: 'model' },
  { key: 'max_num_seqs', id: 'cluster-seqs', name: 'seats' },
  { key: 'max_model_len', id: 'cluster-context', name: 'context' },
  { key: 'gpu_memory_utilization', id: 'cluster-util', name: 'GPU memory' },
  { key: 'max_num_batched_tokens', id: 'cluster-batched', name: 'prefill budget' },
  { key: 'kv_cache_memory', id: 'cluster-kv', name: 'KV pool' },
  { key: 'kv_offload_ram_gib', id: 'cluster-offload', name: 'KV offload' },
  { key: 'kv_cache_dtype', id: 'cluster-dtype', name: 'KV type' },
];
/** The local fields where NULL is a value: not set, the engine's own (never a number of ours) — its default chunk, the
 *  pool sized from the GPU memory fraction, no KV tier in the host's RAM. */
const LOCAL_UNSET = { max_num_batched_tokens: 'vLLM default', kv_cache_memory: '% of card', kv_offload_ram_gib: 'off',
  kv_cache_dtype: 'engine auto' };
/** KV offload's sizes, GiB of the host's RAM (operator, 2026-10-12: Off / 8 / 16 / 32 / 48) — up to the fleet's bound. */
const RAM_TIER_STEPS = [8, 16, 32, 48];
/** THE ENGINE OF A SELECTED CHOICE, as the LB declares it (`cluster.engines`, fleet.py NodeEngine.view): the settings
 *  it reads — each its key, word, node variable, options and what null means — and the tiles it publishes. */
const engineDeclOf = (c, ref) => (((c || {}).engines || {})[(choiceOf(c, ref) || {}).engine]) || null;
/** The settings ONLY a declaration names (no house editor here): drawn from it, keyed by launch key. Kept from the last
 *  cluster drawn, for the change list's words (`localText`). */
let declaredOnly = {};
function declaredOnlyOf(c) {
  const out = {};
  Object.values((c || {}).engines || {}).forEach((e) => (e.settings || []).forEach((st) => {
    if (!LOCAL_FIELDS.some((f) => f.key === st.key)) out[st.key] = st;
  }));
  declaredOnly = out;
  return out;
}
/** The node's fields FOR THE MODEL ON SCREEN: the house fields and the declared ones, those its engine reads alone. */
function localFields(c) {
  const decl = engineDeclOf(c, localShown(c, 'model'));
  const reads = new Set(['model', ...((decl || {}).settings || []).map((st) => st.key)]);
  const extra = Object.values(declaredOnlyOf(c)).map((st) => ({ key: st.key, id: `cluster-set-${st.key}`, name: st.name }));
  return LOCAL_FIELDS.concat(extra).filter((f) => reads.has(f.key));
}
/** `key` as the operator has it on screen: the field's value once edited, else the wish's. */
function localShown(c, key) {
  // An LB that publishes no KV dtype choices takes no kv_cache_dtype: nothing to show, nothing sent.
  if (key === 'kv_cache_dtype' && !takesKnob(c, key)) return undefined;
  const f = LOCAL_FIELDS.find((x) => x.key === key);
  const st = f ? null : declaredOnly[key];
  const el = f ? $(f.id) : (st ? $(`cluster-set-${key}`) : null);
  if (el && el.dataset.dirty) {
    if (key === 'model') return el.value;
    if (key === 'kv_cache_dtype') return el.value || null;
    if (st) return el.value === '' ? null : (typeof st.options[0] === 'number' ? Number(el.value) : el.value);
    return key in LOCAL_UNSET && el.value === '' ? null : Number(el.value);
  }
  return ((c || {}).desired || {})[key];
}
const localValid = (key, v) => (key === 'model' ? typeof v === 'string' && v !== ''
  : key in declaredOnly ? v === null || isNum(v) || (typeof v === 'string' && v !== '')
    : key === 'kv_cache_dtype' ? v === null || (typeof v === 'string' && v !== '')
      : (key in LOCAL_UNSET && v === null) || isNum(v));
/** Every field whose value on screen is not the wish's: {key, id, name, from, to}. */
function localChanges(c) {
  const d = (c || {}).desired || {};
  return localFields(c).map((f) => Object.assign({}, f, { from: d[f.key], to: localShown(c, f.key) }))
    .filter((ch) => localValid(ch.key, ch.to) && ch.to !== ch.from);
}
const localText = (key, v) => (key in declaredOnly ? (v == null ? declaredOnly[key].unset : String(v))
  : v === null && key in LOCAL_UNSET ? LOCAL_UNSET[key]
  : key === 'gpu_memory_utilization' ? Number(v).toFixed(2)
    : key === 'kv_cache_memory' ? `${(Number(v) / 2 ** 30).toFixed(2)} GiB`
      : key === 'kv_offload_ram_gib' ? `${fmtInt(v)} GiB RAM`
        : (key === 'model' || key === 'kv_cache_dtype' ? String(v) : fmtInt(v)));
/** What the switch is sent: `start` the whole selection on screen, `apply` / `save` what changed (switched on /
 *  off), `stop` nothing but the stop — the LB keeps the wish's settings (proxy.py `_local_switch`). */
function localBody(c, kind) {
  if (kind === 'stop') return { enabled: false };
  const body = { enabled: kind !== 'save' };
  if (kind === 'start') {
    localFields(c).forEach((f) => { const v = localShown(c, f.key); if (localValid(f.key, v)) body[f.key] = v; });
  } else {
    localChanges(c).forEach((ch) => { body[ch.key] = ch.to; });
  }
  return body;
}
const choiceOf = (c, ref) => ((c || {}).choices || []).find((x) => x.ref === ref) || null;

/** The fit for the selection ON SCREEN: the LB's estimate of it once the operator edited, else the wish's own. */
let localEstimate = null;
function localSelectionKey(c) {
  return localFields(c).map((f) => `${f.key}=${localShown(c, f.key)}`).join('&');
}
function localFitOf(c) {
  if (!localChanges(c).length) return c.fit || null;
  return localEstimate && localEstimate.key === localSelectionKey(c) ? localEstimate : null;
}
async function askLocalFit(c) {
  const key = localSelectionKey(c);
  const q = new URLSearchParams();
  // A field left to the engine is asked BLANK: the LB reads it as vLLM does an empty value, its own default.
  localFields(c).forEach((f) => { const v = localShown(c, f.key); if (localValid(f.key, v)) q.set(f.key, v === null ? '' : String(v)); });
  try {
    const r = await fetch(`${clusterRoute()}/estimate?${q}`);
    const body = await r.json();
    localEstimate = r.ok ? Object.assign({ key }, body)
      : { key, fit: null, refused: ((body || {}).error || {}).message || `HTTP ${r.status}` };
  } catch (e) {
    localEstimate = { key, fit: null, refused: `the estimate could not be read: ${e.message}` };
  }
  if (lastStatus && theCluster(lastStatus) && localNode(theCluster(lastStatus))) renderLocalSettings(theCluster(lastStatus), lastStatus);
}
const gib = (v) => `${Number(v).toFixed(1)} GiB`;
/** "20.4 GiB weights + 3.4 GiB engine + 4.3 GiB KV ≤ 28.8 GiB" — the fleet's numbers (fleet.launch_fit), the sign
 *  its verdict; a term the catalog does not state for one card is said so, never counted as zero. A prefill chunk
 *  under the one the engine term was measured at is charged that measurement, and the term says at which chunk. A
 *  PINNED pool replaces the one-request KV term and is priced on the whole card; a KV tier in the host's RAM follows
 *  the sum, never in it ("· + 32 GiB RAM tier"). */
function fitText(f) {
  if (f && typeof f.text === 'string') return f.text;      // an engine that states its own memory model in words
  if (!f || !isNum(f.weights_gib) || !isNum(f.budget_gib)) return '';
  const term = (v, what) => (isNum(v) ? ` + ${gib(v)} ${what}` : ` + ${what} not stated`);
  const engine = f.note && isNum(f.overhead_chunk) ? `engine (measured @ ${fmtInt(f.overhead_chunk)} chunk)` : 'engine';
  const pinned = isNum(f.pool_gib);
  const kv = pinned ? f.pool_gib : f.kv_gib;
  const need = f.weights_gib + (isNum(f.overhead_gib) ? f.overhead_gib : 0) + (isNum(kv) ? kv : 0);
  return `${gib(f.weights_gib)} weights${term(f.overhead_gib, engine)}${term(kv, pinned ? 'KV pool (pinned)' : 'KV')} `
    + `${need <= f.budget_gib ? '≤' : '>'} ${gib(f.budget_gib)}${pinned ? ' whole card' : ''}`
    + (isNum(f.ram_gib) ? ` · + ${fmtInt(f.ram_gib)} GiB RAM tier` : '');
}
/** The KV rows' words: the pair's knobs (a pool per rank, a disk tier) or a local node's (the pool on its one card, a
 *  tier in the host's RAM) — the same two rows, DRY. */
function kvRowWords(local) {
  const put = (id, text) => { const n = $(id); if (n) n.textContent = text; };
  put('cluster-kv-unit', local ? 'on the card' : 'per rank');
  put('cluster-offload-var', local ? 'kv_offload_ram_gib' : 'kv_offload_gib');
  put('cluster-offload-unit', local ? 'host RAM' : 'NVMe');
}
/** Why a Start or an Apply cannot be sent with the selection on screen — the fleet's own words — or ''. */
function localRefusal(c) {
  const f = localFitOf(c);
  return f && f.refused ? f.refused : '';
}

/** One of the node's editable fields: the presets within its limits and the value it holds, `selected` — and, where
 *  `unset` names it, the choice of no value at all (the engine's own default), first. */
function editField(id, values, shown, fmt, note, enabled, unset) {
  const el = $(id);
  if (!el) return;
  el.disabled = !enabled;
  // A list the pair's own filler wrote (`fillNumberSelect`, marked by `choices`) is not what `paint` last wrote.
  if (el.dataset.choices) { painted.delete(el); delete el.dataset.choices; }
  const vals = [...new Set([...values, shown].filter((v) => isNum(v)))].sort((a, b) => a - b);
  paint(el, (unset ? `<option value=""${shown == null ? ' selected' : ''}>${esc(unset)}</option>` : '')
    + vals.map((v) => `<option value="${esc(String(v))}"${v === shown ? ' selected' : ''}>${esc(fmt(v))}</option>`)
      .join(''));
  el.value = isNum(shown) ? String(shown) : '';
  const custom = customOf(el);
  if (custom) custom.hidden = true;
  const n = $(`${id}-note`);
  if (n) { paint(n, esc(note)); n.hidden = !note; }
  const warn = $(`${id}-warn`);
  if (warn) warn.hidden = true;
}

/** The house rows, by the launch key each edits — a row whose key the selected engine does not read is hidden. The
 *  speculative tokens and the PLE table are the pair's: no node engine declares them. */
const HOUSE_ROWS = [['max_num_seqs', 'cluster-seqs'], ['max_model_len', 'cluster-context'],
  ['gpu_memory_utilization', 'cluster-util-row'], ['max_num_batched_tokens', 'cluster-batched'],
  ['kv_cache_memory', 'cluster-kv'], ['kv_offload_ram_gib', 'cluster-offload'], ['kv_cache_dtype', 'cluster-dtype-row'],
  ['num_speculative_tokens', 'cluster-spec'], ['ple_replicate', 'cluster-ple']];
const OWN_VISIBILITY = new Set(['cluster-dtype-row']);
const settingRowOf = (id) => { const el = $(id); return el && el.closest ? (el.closest('.chooser') || el) : el; };

/** THE ROWS A LOCAL NODE SHOWS ARE ITS SELECTED ENGINE'S SETTINGS (`cluster.engines`): the house rows it reads, and
 *  each setting only its declaration names drawn from that declaration — its word, its options, its unset word, the
 *  node variable it reaches. A new engine's knob is drawn here without a line of this page changing. */
function renderDeclaredRows(c, ref, listening, source) {
  const decl = engineDeclOf(c, ref);
  const reads = new Set(((decl || {}).settings || []).map((st) => st.key));
  // A row with its own rule for showing (the KV type: only where the LB publishes choices) keeps it where read.
  HOUSE_ROWS.forEach(([key, id]) => {
    const row = settingRowOf(id);
    if (row && (!reads.has(key) || !OWN_VISIBILITY.has(id))) row.hidden = !reads.has(key);
  });
  const box = $('cluster-engine-fields');
  if (!box) return;
  const own = ((decl || {}).settings || []).filter((st) => st.key in declaredOnlyOf(c));
  paint(box, own.map((st) => `<div class="chooser cs-field">
      <div class="cs-head"><label for="cluster-set-${esc(st.key)}">${esc(st.name.charAt(0).toUpperCase() + st.name.slice(1))}</label>
        <span class="cs-var">${esc(st.key)}</span></div>
      <div class="cs-control"><select id="cluster-set-${esc(st.key)}" aria-describedby="cluster-set-${esc(st.key)}-note"></select></div>
      <span id="cluster-set-${esc(st.key)}-note" class="cs-note"></span></div>`).join(''));
  box.hidden = !own.length;
  own.forEach((st) => {
    const el = $(`cluster-set-${st.key}`);
    if (!el) return;
    const shown = localShown(c, st.key);
    const vals = [...new Set([...st.options, ...(shown == null ? [] : [shown])])];
    el.disabled = !listening;
    paint(el, (st.unset != null ? `<option value=""${shown == null ? ' selected' : ''}>${esc(st.unset)}</option>` : '')
      + vals.map((v) => `<option value="${esc(String(v))}"${v === shown ? ' selected' : ''}>${esc(String(v))}</option>`)
        .join(''));
    el.value = shown == null ? '' : String(shown);
    const note = $(`cluster-set-${st.key}-note`);
    if (note) paint(note, esc(`${source} · ${st.env}`));
    if (!el.dataset.wired) {
      el.dataset.wired = '1';
      const mark = () => {
        el.dataset.dirty = '1';
        closeConfirm();
        if (lastStatus && theCluster(lastStatus)) renderClusterSettings(theCluster(lastStatus), lastStatus);
        askEstimate();
      };
      el.addEventListener('change', mark);
    }
  });
}

/** THE SETTINGS OF A NODE THE LB REQUESTS: the model among the fleet's choices, seats, context, the prefill budget and
 *  GPU memory — editable while an applier listens — what the node says it runs with for the rest, locked; the fit
 *  line; Apply & restart (switched on, confirmed) or Save for next start (off). */
function renderLocalSettings(c, s) {
  const d = c.desired || {};
  const on = d.mode === 'cluster';
  const listening = !!(c.applier || {}).listening;
  $('cluster-settings').hidden = !c.desired;
  if (!c.desired) return;
  paint($('cluster-settings-summary'), on ? 'Settings' : 'Next start');
  ['cluster-defaults', 'cluster-estimate', 'cluster-hint'].forEach((id) => { $(id).hidden = true; });
  const e = c.effective || {};
  const source = d.id ? `intent #${d.id}` : 'fleet';
  // What the node says, which no intent moves.
  lockField('cluster-spec', isNum(e.num_speculative_tokens) ? (e.num_speculative_tokens ? String(e.num_speculative_tokens) : 'off')
    : null, e.num_speculative_tokens == null ? NOT_EXPOSED_BY_NODE : 'node');
  const tok = enginePoolTokens(s);
  kvRowWords(true);
  const ple = $('cluster-ple');
  const pleRow = ple && ple.closest ? ple.closest('.chooser') : null;
  if (pleRow) pleRow.hidden = true;
  // The model: one row per fleet choice, the node's own engine; the row the node runs lit.
  const ref = localShown(c, 'model');
  const rows = (c.choices || []).map((ch) => {
    const m = catalogOf(s).find((x) => x && x.id === ch.model_choice) || null;
    // DOES THIS CHOICE START: its own record off the boot log, under the pair's row key (`<id>` / `<id>@uncensored`) —
    // the node's starts are recorded as the pair's are (cluster_loop `_record_local_boot`).
    const starts = startsOf((s || {}).start_rates, rowKey(ch.model_choice, ch.uncensored));
    const facts = Object.assign(m ? pickerRowFacts(m, ch.uncensored, null, null, [], starts)
      : { label: ch.label, family: ch.label, build: '', engineInLabel: false, uncensored: ch.uncensored, figures: [],
        note: null, rate: null, gone: null, starts }, { engine: engineName(e.engine), engineKey: String(e.engine || '').toLowerCase(),
      drafter: null, size: isNum(ch.size_gib) ? `${ch.size_gib} GiB` : null });
    return { value: ch.ref, gone: [], facts, text: `${facts.label || ch.ref}${ch.uncensored ? ' · uncensored' : ''}` };
  });
  const select = $('cluster-model');
  if (select) {
    select.disabled = !listening;
    paint(select, rows.map((r) => `<option value="${esc(r.value)}">${esc(r.text)}</option>`).join(''));
    select.value = ref || '';
  }
  const runs = ((c.declared || {}).model) || null;
  renderPicker(rows, ref, runs && c.state !== 'off' ? { key: runs, word: String(c.state || 'running') } : null, listening);
  const modelNote = $('cluster-model-note');
  if (modelNote) { paint(modelNote, esc(`${source} · ${rows.length} choice${rows.length === 1 ? '' : 's'} in the fleet`)); modelNote.hidden = false; }
  // The figures, within the SELECTED model's own catalog limits.
  const limits = (choiceOf(c, ref) || {}).limits || c.limits || {};
  const [lo, hi] = Array.isArray(limits.max_num_seqs) ? limits.max_num_seqs : [1, 48];
  const maxLen = isNum(limits.max_model_len) ? limits.max_model_len : 262144;
  const nodeSays = (key, v) => (isNum(v) && v !== localShown(c, key) ? ` · node ${localText(key, v)}` : '');
  editField('cluster-seqs', SEAT_PRESETS.filter((v) => v >= lo && v <= hi), localShown(c, 'max_num_seqs'), String,
    source, listening);
  editField('cluster-context', CONTEXT_PRESETS.filter((v) => v <= maxLen), localShown(c, 'max_model_len'), fmtInt,
    source + nodeSays('max_model_len', e.max_model_len), listening);
  editField('cluster-util', UTIL_PRESETS, localShown(c, 'gpu_memory_utilization'), (v) => Number(v).toFixed(2),
    source + nodeSays('gpu_memory_utilization', e.gpu_memory_utilization), listening);
  // The prefill budget (vLLM's --max-num-batched-tokens): vLLM's own default, or a step no smaller than the seats
  // (vLLM's own floor) and no larger than the chunk the selected row's engine overhead was measured at — the largest
  // the fleet can check; a row that names none offers the pair's list, and the fleet refuses a pick in its words.
  const measured = (choiceOf(c, ref) || {}).overhead_chunk;
  const [blo, bhi] = rangeOf(c, 'max_num_batched_tokens');
  const seatsShown = localShown(c, 'max_num_seqs');
  editField('cluster-batched', BATCHED_STEPS.filter((v) => (isNum(measured) ? v <= measured : v >= blo && v <= bhi)
    && !(isNum(seatsShown) && v < seatsShown)), localShown(c, 'max_num_batched_tokens'), fmtInt,
    source + nodeSays('max_num_batched_tokens', e.max_num_batched_tokens), listening, LOCAL_UNSET.max_num_batched_tokens);
  // The KV pool (vLLM --kv-cache-memory-bytes): from GPU memory — vLLM sizes it from the fraction — or PINNED, a whole
  // GB up to what the fit leaves the selected row on the whole card (`kv_pin_max`), the stored size kept; the fit
  // prices a pinned pool on the whole card.
  const pinMax = (choiceOf(c, ref) || {}).kv_pin_max;
  const pins = isNum(pinMax) ? Array.from({ length: Math.floor(pinMax / 1e9) }, (_, i) => (i + 1) * 1e9) : [];
  editField('cluster-kv', pins, localShown(c, 'kv_cache_memory'), (v) => localText('kv_cache_memory', v),
    source + (tok == null ? '' : ` · node ${fmtInt(tok)} tokens`), listening, LOCAL_UNSET.kv_cache_memory);
  // KV offload: a tier in the host's RAM (vLLM --kv-offloading-size), up to the fleet's bound on the box — or locked
  // at off, the fleet's reason beside it (no bound stated there, or memory the GPU shares with the host).
  const tier = c.ram_tier || {};
  if (isNum(tier.max_gib)) {
    editField('cluster-offload', RAM_TIER_STEPS.filter((v) => v <= tier.max_gib), localShown(c, 'kv_offload_ram_gib'),
      (v) => localText('kv_offload_ram_gib', v), `${source} · fleet bound ${tier.max_gib} GiB`, listening,
      LOCAL_UNSET.kv_offload_ram_gib);
  } else {
    lockField('cluster-offload', LOCAL_UNSET.kv_offload_ram_gib, tier.refused || 'the fleet states no bound for a RAM tier');
  }
  // The KV type (vLLM --kv-cache-dtype): the engine's auto, or a dtype the node's image runs on its card (the LB's
  // `kv_cache_dtypes` for the selected choice) — and, where it differs, what the node says it runs.
  const dtShown = localShown(c, 'kv_cache_dtype');
  const dtRuns = typeof e.kv_cache_dtype === 'string' && e.kv_cache_dtype !== (dtShown || 'auto')
    ? ` · node runs ${e.kv_cache_dtype}` : '';
  fillKvDtype(takesKnob(c, 'kv_cache_dtype') ? kvDtypesOf(c, ref) : null, '', LOCAL_UNSET.kv_cache_dtype,
    dtShown || '', listening, source + dtRuns);
  // ONE POOL, TWO MODES (operator 2026-10-24, "allow choose one"): a pinned size, or "% of card" — vLLM sizes it from
  // gpu_memory_utilization only while no size is pinned (a pin wins, and the share would be a number that does nothing).
  $('cluster-util-row').hidden = $('cluster-kv').value !== '';
  // The fit: the fleet's numbers for the selection on screen, or its refusal, verbatim.
  const fit = localFitOf(c);
  const fitEl = $('cluster-fit');
  const refused = localRefusal(c);
  const text = refused || fitText((fit || {}).fit) || (fit ? '' : 'pricing the selection…');
  fitEl.textContent = text;
  fitEl.hidden = !text;
  fitEl.classList.toggle('bad', !!refused);
  fitEl.title = ((fit || {}).fit || {}).note || ((fit || {}).fit || {}).why || '';
  // RAM beside a running pair: the run's estimate as an ALERT (cluster_run `_beside_problem`) — Start still proceeds.
  const alerts = (((s && s.declared) || []).find((d) => d.name === c.name) || {}).alerts || [];
  const alertEl = $('cluster-alert');
  alertEl.textContent = alerts.length ? `RAM: ${alerts.join('; ')} — Start proceeds` : '';
  alertEl.hidden = !alerts.length;
  // The pool line: the engine's own pool at the window it serves, and the seats.
  const note = $('cluster-recommended');
  if (note) {
    const msg = isNum(e.max_model_len) && e.max_model_len > 0 ? poolLine(null, tok, e.max_model_len, e.max_num_seqs) : '';
    note.textContent = msg;
    note.hidden = !msg;
  }
  renderNodeFacts(c);
  renderDeclaredRows(c, ref, listening, source);
  // Apply & restart (on) / Save for next start (off): what changed, listed; held back while refused or unheard.
  const changes = localChanges(c);
  const apply = $('cluster-apply');
  apply.hidden = false;
  apply.textContent = on ? 'Apply & restart' : 'Save for next start';
  apply.classList.toggle('save', !on);
  apply.disabled = clusterBusy || !listening || !changes.length || !!refused;
  paint($('cluster-changes'), changes.length ? esc(`${changes.length === 1 ? '1 change' : `${changes.length} changes`}: `
    + changes.map((ch) => `${ch.name} ${localText(ch.key, ch.from)} → ${localText(ch.key, ch.to)}`).join(', ')) : '');
  const box = $('cluster-confirm');
  if (!box.hidden && (!on || box.dataset.key !== localSelectionKey(c))) closeConfirm();
  $('cluster-confirm-go').disabled = clusterBusy;
  // Finishing what it serves before a restart or a stop: the same wish, now, cutting them.
  const force = $('cluster-force');
  force.hidden = c.state !== 'finishing';
  force.disabled = clusterBusy;
}

/** WHO ON THE HOST APPLIES THE PANEL'S INTENT and what it last did — in its own words when it failed (the command's
 *  exit code and stderr tail, the node container restarting, the node's last log lines); red when nobody listens. */
function renderApplier(c) {
  const el = $('cluster-applier');
  if (!el) return;
  const a = c.applier || null;
  el.hidden = !a;
  if (!a) return;
  const parts = [];
  let bad = !a.listening;
  parts.push(a.listening ? `applier · ${esc(a.host || '?')} · ${Math.round(a.heartbeat_age_s || 0)} s ago`
    : `<b>${esc(a.error || 'no applier listening')}</b>`);
  if (a.applying) parts.push(`applying #${esc(String(a.applying.id))} (${esc(a.applying.state)})`);
  const last = a.applied;
  if (last && last.refused) {
    bad = true;
    parts.push(`<b>#${esc(String(last.id))} refused</b>: ${esc(last.refused)}`);
  } else if (last && last.exit !== 0) {
    bad = true;
    const words = String(last.stderr || '').trim().split('\n').slice(-2).join(' · ');
    parts.push(`<b>#${esc(String(last.id))} (${esc(last.state)}): up.sh exit ${esc(String(last.exit))}</b>${words ? `: ${esc(words)}` : ''}`);
  } else if (last) {
    parts.push(`last #${esc(String(last.id))} (${esc(last.state)}) · exit 0`);
  }
  const k = a.container;
  // A crash loop reads `running` between attempts: the restarts since the command (`restarts_at_apply`) say it.
  const since = last && Number.isInteger(last.restarts_at_apply) && k && Number.isInteger(k.restarts)
    ? k.restarts - last.restarts_at_apply : 0;
  if (since > 0 && c.state === 'running') {
    // Restarted, and answering since: history (the plan says running), not a crash loop.
    parts.push(`recovered after ${since} restart${since === 1 ? '' : 's'} since #${esc(String(last.id))}`);
  } else if (since > 0) {
    bad = true;
    parts.push(`<b>container restarted ${since} time${since === 1 ? '' : 's'} since #${esc(String(last.id))}</b>`);
  } else if (k && k.status === 'exited' && k.exit_code === 0 && last && last.state === 'down' && !last.refused
    && last.exit === 0) {
    parts.push('container stopped');                  // the down intent, applied: what was asked
  } else if (k && ['restarting', 'exited', 'dead'].includes(k.status)) {
    bad = true;
    parts.push(`<b>container ${esc(k.status)}</b> · exit ${esc(String(k.exit_code))} · ${esc(String(k.restarts))} restarts`);
  }
  const tail = Array.isArray(a.log_tail) && a.log_tail.length ? `<pre class="cp-log">${esc(a.log_tail.join('\n'))}</pre>` : '';
  paint(el, `<span>${parts.join(' &middot; ')}</span>${tail}`);
  el.classList.toggle('bad', bad);
}

/** Start / Stop for a node the LB requests: the selection on screen with Start, nothing but the stop with Stop. */
async function toggleLocal(c) {
  const start = (c.desired || {}).mode !== 'cluster';
  const refused = start ? localRefusal(c) : '';
  if (refused) { showClusterError(refused); return; }
  closeConfirm();
  clusterBusy = true;
  renderBusy();
  try {
    await postCluster(localBody(c, start ? 'start' : 'stop'));
  } finally {
    clusterBusy = false;
    renderBusy();
  }
}

/** Apply & restart asks first; Save for next start goes. */
function applyLocal(c) {
  if (clusterBusy || !localChanges(c).length || localRefusal(c)) return;
  showClusterError('');
  if ((c.desired || {}).mode !== 'cluster') { sendLocal('save'); return; }
  const box = $('cluster-confirm');
  $('cluster-confirm-title').textContent = 'Restart the node at these settings?';
  $('cluster-confirm-go').textContent = 'Restart the node';
  $('cluster-confirm-text').textContent = `${localChanges(c).map((ch) => `${ch.name} ${localText(ch.key, ch.from)} → `
    + `${localText(ch.key, ch.to)}`).join(', ')}. The node first finishes what it serves, then its host relaunches it: `
    + 'a cold start, nothing served meanwhile.';
  box.dataset.key = localSelectionKey(c);
  box.hidden = false;
  $('cluster-confirm-cancel').focus();
}
async function sendLocal(kind, extra) {
  if (clusterBusy) return;
  closeConfirm();
  clusterBusy = true;
  renderBusy();
  try {
    await postCluster(Object.assign(localBody(theCluster(lastStatus) || {}, kind), extra || {}));
  } finally {
    clusterBusy = false;
    renderBusy();
  }
}

/** WHERE THE SETTINGS ARE. As in the app, the picks belong to a Start: while Start is offered the model
 *  and the concurrency sit open in the card, as the two choosers, with what the selection needs on each
 *  box under them. While the cluster is switched ON the card shows the model line instead and the same
 *  fields fold away behind "Change settings" — still there, because the page can do what the app cannot:
 *  apply a change by restarting the pair. Context, prefill budget and speculative tokens are one
 *  disclosure further ("Advanced") in both.
 *
 *  A disclosure opens BY ITSELF only when something inside needs to be seen — an edit not sent, a restart
 *  pending, a value this page refused, a setting off its default — and only on the poll where that
 *  begins; after that it is the operator's to open and close, and no poll moves it. */
function renderSettingsDisclosure(c) {
  const outer = $('cluster-settings');
  if (!outer) return;
  const on = (c.desired || {}).mode === 'cluster';
  const editable = !!c.desired && (on || c.state === 'off');
  const model = shownOf('model_choice') || storedOf(c, 'model_choice');
  const changes = editable ? clusterChanges(c) : [];
  const off = (key) => takesKnob(c, key) && (changes.some((ch) => ch.key === key)
    || (runningOf(c, key) != null && runningOf(c, key) !== storedOf(c, key, storedOf(c, 'model_choice')))
    || ($(fieldOf(key).id) || { getAttribute: () => null }).getAttribute('aria-invalid') === 'true'
    || (c.desired && shownOf(key) !== defaultOf(c, key, model)));
  const attention = changes.length > 0 || CLUSTER_FIELDS.map((f) => f.key).some((key) =>
    runningOf(c, key) != null && runningOf(c, key) !== storedOf(c, key, storedOf(c, 'model_choice')));
  const mode = on ? 'on' : 'off';
  if (outer.dataset.mode !== mode) {
    outer.dataset.mode = mode;
    outer.open = !on;                      // open with the choosers while Start is offered; folded while on
    delete outer.dataset.attention;
  }
  if (attention && outer.dataset.attention !== '1') outer.open = true;
  outer.dataset.attention = attention ? '1' : '0';
  paint($('cluster-settings-summary'), on ? 'Change settings' : 'Next start');
}

/** The same, for a knob whose VALUE is not what the operator should read: a pool in bytes, a placement
 *  carried as -1/1/0. `label` writes each option, and there is no "custom…": these three are picked from
 *  what has been measured or reasoned about, never typed. */
function fillChoiceSelect(el, steps, current, dflt, label, extra = [], locked = () => false) {
  const choices = steps.slice();
  if (Number.isInteger(current) && !choices.includes(current)) choices.push(current);
  // THE DEFAULT IS ALWAYS A CHOICE: a model's own pool (the SGLang Flash-Next row's 18 GB) is no step of the list,
  // and on a row whose bound greys everything from 20 GB up it would otherwise leave 16 GB as the only pick.
  if (Number.isInteger(dflt) && !choices.includes(dflt)) choices.push(dflt);
  choices.sort((a, b) => a - b);
  // `extra`: choices that are not numbers (the pool's "auto"), after the numbers, keyed by their words too.
  // Keyed by the WORDS too: a pool's label carries the picker model's tokens, which change with the model.
  // `locked(v)`: a choice the LB would refuse for this model (a pool past its profile's maximum) stays in the list,
  // DISABLED, so what is stored can still be shown and named \u2014 it is never offered.
  const key = `${choices.map((v) => `${v}=${label(v)}${locked(v) ? '!' : ''}`).join(',')}|${dflt}|${extra.map((o) => `${o.value}=${o.label}`).join(',')}`;
  if (el.dataset.choices !== key && !inUse(el)) {
    el.innerHTML = choices.map((v) =>
      `<option value="${v}"${locked(v) ? ' disabled' : ''}>${esc(label(v))}${v === dflt ? ' \u00b7 default' : ''}</option>`).join('')
      + extra.map((o) => `<option value="${esc(o.value)}">${esc(o.label)}</option>`).join('');
    el.dataset.choices = key;
    if (Number.isInteger(current) || extra.some((o) => o.value === current)) el.value = String(current);
  }
}

/** The two engine knobs' controls: offered only while the LB takes them (`takesKnob`), and otherwise
 *  shown at the default an older LB launches with, locked. */
function renderKnobs(c, editable, model) {
  const batched = $('cluster-batched');
  if (batched) {
    const takes = takesKnob(c, 'max_num_batched_tokens');
    const [lo, hi] = limitsOf(c, 'max_num_batched_tokens');
    const dflt = defaultOf(c, 'max_num_batched_tokens', model);
    // An engine that chunks its prompts itself (TensorFold) takes no budget: locked, the LB refuses one with why.
    batched.disabled = !editable || !takes || !!refusedBy(c, model, 'max_num_batched_tokens');
    if (!takes) delete batched.dataset.dirty;
    const want = storedOf(c, 'max_num_batched_tokens', model);
    fillNumberSelect(batched, BATCHED_STEPS.filter((v) => v >= lo && v <= hi),
                     pristine(batched) ? want : wholeOf(batched), dflt);
    if (pristine(batched)) batched.value = String(storedOf(c, 'max_num_batched_tokens', model));
  }
  const kv = $('cluster-kv');
  if (kv) {
    // An engine that takes no pin (SGLang sizes its own pools) is locked at the model's pool, with no auto.
    const refused = !!refusedBy(c, model, 'kv_cache_memory');
    const takes = takesKnob(c, 'kv_cache_memory') && !refused;
    const [lo, hi] = limitsOf(c, 'kv_cache_memory');
    const dflt = defaultOf(c, 'kv_cache_memory', model);
    kv.disabled = !editable || !takes;
    if (!takes) delete kv.dataset.dirty;
    // AUTO, where this LB takes it (it publishes `desired.kv_cache_auto`): sized by the Start to what is left
    // — and never for a model whose footprint nobody measured: the LB refuses it, so it is not offered, and
    // an `auto` stored or picked for another model falls back to this one's own pool.
    const sizable = isCalibrated(c, model) && !refused;
    if (!sizable && kv.value === KV_AUTO && document.activeElement !== kv) delete kv.dataset.dirty;
    const kept = storedOf(c, 'kv_cache_memory', model);
    const want = refused || (!sizable && kept === KV_AUTO) ? dflt : kept;
    const auto = sizable && typeof (c.desired || {}).kv_cache_auto === 'boolean'
      ? [{ value: KV_AUTO, label: kvAutoLabel() }] : [];
    // THE MOST THIS PROFILE CAN PIN (`kvMaxOf`): the pins past it are listed greyed, each naming the maximum as a
    // figure, so the list says why 24 GB cannot be chosen on a row whose bound is 19.9 — and a stored pin past it is
    // still shown, for `selectionProblem` to refuse by name before the Start.
    const max = kvMaxOf(c, model);
    const over = (v) => max !== null && Number.isInteger(v) && v > max;
    fillChoiceSelect(kv, KV_STEPS.filter((v) => v >= lo && v <= hi),
                     pristine(kv) ? want : shownOf('kv_cache_memory'), dflt,
                     (v) => fmtSetting(c, 'kv_cache_memory', v, model) + (over(v) ? ` · max ${gbTenth(max)} GB` : ''),
                     auto, over);
    if (pristine(kv)) kv.value = String(want);
  }
  if ($('cluster-dtype')) {
    const dflt = defaultOf(c, 'kv_cache_dtype', model);
    fillKvDtype(takesKnob(c, 'kv_cache_dtype') ? kvDtypesOf(c, model, draftDepthOf(c, model)) : null, dflt, `profile (${dflt})`,
      pristine($('cluster-dtype')) ? storedOf(c, 'kv_cache_dtype', model) : (shownOf('kv_cache_dtype') || dflt), editable);
  }
  const ple = $('cluster-ple');
  if (ple) {
    const takes = takesKnob(c, 'ple_replicate');
    // A model with NO PLE TABLE has nothing to place: the row is gone, and the field rests at "as the
    // model says" — so a placement stored for it shows up as the reset it needs (the LB refuses it).
    const placed = hasPle(c, model) && !refusedBy(c, model, 'ple_replicate');
    const row = ple.closest('.chooser');
    if (row) row.hidden = !placed;
    ple.disabled = !editable || !takes || !placed;
    if (!takes || !placed) delete ple.dataset.dirty;
    const want = placed ? storedOf(c, 'ple_replicate', model) : PLE_DEFAULT;
    fillChoiceSelect(ple, PLE_STEPS, pristine(ple) ? want : wholeOf(ple), PLE_DEFAULT,
                     (v) => fmtSetting(c, 'ple_replicate', v));
    if (pristine(ple)) ple.value = String(want);
  }
  const off = $('cluster-offload');
  if (off) {
    const refused = !!refusedBy(c, model, 'kv_offload_gib');     // SGLang has no disk tier: locked at off
    const takes = takesKnob(c, 'kv_offload_gib') && !refused;
    const [lo, hi] = limitsOf(c, 'kv_offload_gib');
    const dflt = defaultOf(c, 'kv_offload_gib', model);
    off.disabled = !editable || !takes;
    if (!takes) delete off.dataset.dirty;
    const want = refused ? 0 : storedOf(c, 'kv_offload_gib', model);
    fillChoiceSelect(off, OFFLOAD_STEPS.filter((v) => v >= lo && v <= hi),
                     pristine(off) ? want : wholeOf(off), dflt, (v) => fmtSetting(c, 'kv_offload_gib', v));
    if (pristine(off)) off.value = String(want);
  }
  const spec = $('cluster-spec');
  if (spec) {
    const takes = takesKnob(c, 'num_speculative_tokens');
    // A model with NO DRAFTER has one depth, off: the list is that one word and the control is locked —
    // a depth stored for another model is clamped to it and listed as the reset it is.
    const drafts = hasDrafter(c, model);
    const [lo, hi] = drafts ? limitsOf(c, 'num_speculative_tokens') : [0, 0];
    spec.disabled = !editable || !takes || !drafts;
    if (!takes || !drafts) delete spec.dataset.dirty;
    const want = storedOf(c, 'num_speculative_tokens', model);
    const current = drafts ? (pristine(spec) ? want : wholeOf(spec)) : 0;
    const choices = [];
    for (let v = lo; v <= Math.min(hi, lo + 32); v++) choices.push(v);
    // The value in force is always offered, even off the list, like the context's.
    if (Number.isInteger(current) && !choices.includes(current)) choices.push(current);
    choices.sort((a, b) => a - b);
    const key = choices.join(',');
    if (spec.dataset.choices !== key && document.activeElement !== spec) {
      spec.innerHTML = choices.map((v) => `<option value="${v}">${v === 0 ? 'off' : v}</option>`).join('');
      spec.dataset.choices = key;
      if (Number.isInteger(current)) spec.value = String(current);
    }
    if (!drafts) spec.value = '0';
    else if (pristine(spec) && Number.isInteger(want)) spec.value = String(want);
  }
}

/** The notes, the edit marks and the buttons that send them — drawn on every poll AND every keystroke, so
 *  what the button says it would send is always what is on screen. */
function renderClusterActions(c) {
  const d = c.desired || {};
  const on = d.mode === 'cluster';
  const editable = !!c.desired && (on || c.state === 'off');
  const model = shownOf('model_choice') || storedOf(c, 'model_choice');
  const changes = editable ? clusterChanges(c) : [];
  for (const f of CLUSTER_FIELDS) {
    facesOf(f.id).forEach((el) => el.classList.toggle('changed', changes.some((ch) => ch.key === f.key)));
    const note = $(`${f.id}-note`);
    if (note) {
      const html = fieldNote(c, f.key, model, on);
      paint(note, html);
      note.hidden = !html;
    }
  }
  // WHAT THE START GATE CHARGES for the two knobs it counts, from the estimate of the model IN THE PICKER
  // (its own anchor and slope — memory_estimate.py): the figure above the anchor, "not measured" for a
  // model nobody has calibrated, and nothing from an LB that does not publish the slopes.
  const est = estimateOf(c, model) || {};
  const charged = (el, slope, anchor, shown, unit, what) => {
    if (!el) return;
    const known = isNum(slope) && isNum(anchor);
    const measured = known && slope > 0 && anchor > 0;
    el.hidden = !known || (measured && !(shown > anchor));
    paint(el, measured
      ? `above ${fmtInt(anchor)} &middot; ~${slope}&nbsp;GiB per ${unit} per box &middot; checked at start`
      : `${what} &middot; not measured`);
    // The sentence behind the figures is the tooltip's, and it is this model's too.
    el.title = measured
      ? `above ${fmtInt(anchor)} the start's memory check counts about ${slope} GiB per extra ${unit} on every box, `
        + 'and refuses a start that would not leave the CUDA allocator its headroom; the estimate beside each '
        + 'box says what this setting needs and what the box has'
      : 'never measured for this model: the start\'s memory check does not count this setting, and the estimate '
        + 'beside each box leaves it out';
  };
  charged(takesKnob(c, 'max_num_batched_tokens') ? $('cluster-batched-warn') : null,
          est.activation_gib_per_1k_tokens, est.calibrated_batched_tokens, shownOf('max_num_batched_tokens'),
          '1,024 tokens', 'GiB per 1,024 tokens');
  if (!takesKnob(c, 'max_num_batched_tokens') && $('cluster-batched-warn')) $('cluster-batched-warn').hidden = true;
  charged($('cluster-seqs-warn'), est.concurrency_gib_per_seq, est.calibrated_max_num_seqs,
          shownOf('max_num_seqs'), 'seat', 'GiB per seat');
  // Each of the three new knobs says its own thing, and only while it is away from the shipped value:
  // a bigger pool the box must find, a placement nobody has A/B'd, a disk tier nobody has started.
  const kvWarn = $('cluster-kv-warn');
  if (kvWarn) kvWarn.hidden = !(takesKnob(c, 'kv_cache_memory') && isNum(defaultOf(c, 'kv_cache_memory', model))
    && shownOf('kv_cache_memory') > defaultOf(c, 'kv_cache_memory', model));
  const pleWarn = $('cluster-ple-warn');
  if (pleWarn) {
    pleWarn.hidden = !(takesKnob(c, 'ple_replicate') && hasPle(c, model) && shownOf('ple_replicate') === 0);
    // What a split frees is the MODEL's: its table over the ranks (`ple_split_frees_gib`), never a literal.
    const frees = est.ple_split_frees_gib;
    paint(pleWarn, `split${isNum(frees) && frees > 0 ? `: frees ${frees.toFixed(1)}&nbsp;GiB per box` : ''}`
      + ' &middot; +1 exchange per step &middot; <b>never A/B&rsquo;d</b>');
    pleWarn.title = `splitting the table frees ${isNum(frees) && frees > 0 ? `${frees.toFixed(1)} GiB` : 'part of it'} `
      + 'on every box for one more exchange per decode step; never A/B\'d here — tp-rank\'s own note says to '
      + 'adopt it only on a measured comparison';
  }
  const offWarn = $('cluster-offload-warn');
  if (offWarn) offWarn.hidden = !(takesKnob(c, 'kv_offload_gib') && shownOf('kv_offload_gib') > 0);
  const apply = $('cluster-apply');
  apply.hidden = !editable;
  apply.textContent = on ? 'Apply & restart' : 'Save for next start';
  apply.classList.toggle('save', !on);
  apply.disabled = clusterBusy || !changes.length;
  const reset = $('cluster-defaults');
  reset.hidden = !editable;
  reset.disabled = clusterBusy || !defaultsToFill(c, model).length;
  paint($('cluster-changes'), changes.length
    ? esc(`${changes.length === 1 ? '1 change' : `${changes.length} changes`}: ${changeList(c, changes)}`) : '');
  // A confirm that no longer describes what would be sent is withdrawn, never left to be clicked.
  const box = $('cluster-confirm');
  if (!box.hidden && (!on || box.dataset.key !== changeKey(changes))) closeConfirm();
  $('cluster-confirm-go').disabled = clusterBusy;
  renderPrimary(c);                     // Start is held back while the selection on screen is refused
}

/** The fields "Use defaults" would change: each numeric one the LB takes, not at its default now. */
function defaultsToFill(c, model) {
  return CLUSTER_FIELDS.filter((f) => f.key !== 'model_choice' && takesKnob(c, f.key)
    && defaultOf(c, f.key, model) != null && shownOf(f.key) !== defaultOf(c, f.key, model));
}

/** Fill every field with its default — for the model in the picker — as an EDIT: nothing is sent until
 *  Apply & restart or Save, and the list beside the button says what would change. */
function useDefaults() {
  const c = (lastStatus && theCluster(lastStatus)) || {};
  if (clusterBusy) return;
  const model = shownOf('model_choice') || storedOf(c, 'model_choice');
  for (const f of defaultsToFill(c, model)) {
    const el = $(f.id);
    const v = String(defaultOf(c, f.key, model));
    if (el.tagName === 'SELECT' && ![...el.options].some((o) => o.value === v)) {
      el.insertAdjacentHTML('beforeend', `<option value="${esc(v)}">${esc(fmtSetting(c, f.key, Number(v)))}</option>`);
    }
    el.value = v;
    el.dataset.dirty = '1';
    el.removeAttribute('aria-invalid');
  }
  closeConfirm();
  showClusterError('');
  askEstimate();
  renderClusterSettings(c, lastStatus);
}

/** Apply & restart asks first, in words that name the cost; Save for next start costs nothing, and goes. */
function applyClicked() {
  const c = (lastStatus && theCluster(lastStatus)) || {};
  if (localNode(c)) return applyLocal(c);
  const changes = clusterChanges(c);
  if (clusterBusy || !changes.length) return;
  const bad = selectionProblem(c, changes);
  if (bad) {
    refuseSelection(bad);
    return;
  }
  showClusterError('');
  if ((c.desired || {}).mode === 'cluster') openConfirm(c, changes);
  else sendSettings(false);
}

function refuseSelection(bad) {
  showClusterError(bad.why);
  facesOf(fieldOf(bad.key).id).forEach((el) => el.setAttribute('aria-invalid', 'true'));
}

function openConfirm(c, changes) {
  const box = $('cluster-confirm');
  $('cluster-confirm-text').textContent = `${changeList(c, changes)}. Every rank reloads the model: a cold `
    + 'start of about 10 minutes, and every request in flight on this server is dropped. Its boxes stay '
    + 'reserved for it meanwhile — they are not handed back to the fleet.';
  box.dataset.key = changeKey(changes);
  box.hidden = false;
  $('cluster-confirm-cancel').focus();          // the safe answer is the one Enter gives
}

function closeConfirm() {
  const box = $('cluster-confirm');
  if (!box || box.hidden) return;
  const hadFocus = box.contains(document.activeElement);
  box.hidden = true;
  delete box.dataset.key;
  const apply = $('cluster-apply');
  if (hadFocus && apply && !apply.hidden && !apply.disabled) apply.focus();
}

/** Save for next start (stopped) or Apply & restart (switched on, confirmed): POST exactly the fields that
 *  changed. The switch is re-read FIRST — Start, Stop and these post to one endpoint, so a Save must never
 *  land on a cluster someone has just started (`enabled: false` would STOP it), nor a restart on one just
 *  stopped (`enabled: true` would START it). */
async function sendSettings(restart) {
  if (clusterBusy) return;
  if (localNode(theCluster(lastStatus) || {})) return sendLocal(restart ? 'apply' : 'save');
  closeConfirm();
  clusterBusy = true;
  renderBusy();
  try {
    await poll({ fresh: true });
    const c = (lastStatus && theCluster(lastStatus)) || {};
    const on = (c.desired || {}).mode === 'cluster';
    if (on !== restart || (!on && c.state !== 'off')) {
      showClusterError(restart ? 'the server is not switched on any more: nothing was restarted'
        : 'the server is not off any more: nothing was saved');
      return;
    }
    const changes = clusterChanges(c);
    if (!changes.length) return;
    const bad = selectionProblem(c, changes);
    if (bad) {
      refuseSelection(bad);
      return;
    }
    const body = restart ? { enabled: true, restart: true } : { enabled: false };
    changes.forEach((ch) => { body[ch.key] = ch.send; });
    await postCluster(body);
  } finally {
    clusterBusy = false;
    renderBusy();
  }
}

// ── saturation: is there headroom, and against WHICH ceiling ─────────────────
// The operator's question — "can any process still be optimised, or is it fully saturated?" — has
// three readings behind it, and they are NOT interchangeable:
//
//   SM fill     how much of the time the SMs had work resident. A memory-bound decode pins this at
//               96 % while using a fraction of the card's FLOP/s, which is exactly how a "GPU 96 %"
//               tile misled once. Drawn, and labelled with what it is not. MEASURED ON EACH BOX, so it
//               is drawn once per rank — "we see only one as consolidated" was the operator's word on
//               the single busiest-box meter this replaced — with the gap between the two ranks.
//   bandwidth   GB/s per GPU, from the ENGINE's own estimate of the bytes each step reads and
//               writes — or, for SGLang, the LB's own MODEL of each forward pass (`bandwidthModel`),
//               labelled "modeled". The binding ceiling on this fleet.
//   compute     TFLOP/s per GPU, same source.
//
//   tokens      what the busy-ness PRODUCED: generated tok/s and the computed prefill rate, from the
//               pair's own `throughput` block. Their own group, between the other two, because the row
//               argues from left to right — the GPUs are busy, this is what came out, this is what it
//               cost (`clusterTokenGroup`, and the note above it on which prefill rate and why).
//
// Bandwidth and compute are NOT per box: only the head rank serves /metrics, and vLLM's estimate is the
// model's cost per step divided by the TP size — identical for both ranks. They are drawn once, in their
// own group, saying so (`rates_means`). When either is missing the LB sends null and the reason — a switch
// it has PROVEN off, a step still in flight, a first sample — and this panel prints the reason. It
// never prints a 0 in its place. When no step completed in the last probe window the LB widens the
// window to the last one that did, and says so (`window_widened`); the foot prints that sentence.
const SAT_TONE = {
  'idle': 'ok',
  'memory-bandwidth bound': 'warn',
  'compute saturated': 'bad',
  'busy — no ceiling measured': 'warn',
  'not observed': 'off',
};

/** One reading as a labelled bar: the number, the share of its ceiling if there is one, and — when
 *  there is no number — the LB's own reason in its place. `series` draws the same graph the tiles
 *  use, and only where /lb/status actually carries a history for it. `sub` says whose reading it is
 *  (a rank and its role), under the label; `title` is what the quantity is, for a hover, never a row.
 *  The bar wears the share's colour on the one scale, as the app's `BarRow` does; the figure beside it
 *  stays in ink, as the app's does.
 *
 *  ONE FIXED HEIGHT, WHATEVER THE DATA (operator 2026-10-30): the same slots every time — label, sub, value, bar, a
 *  ONE-LINE note (its whole text in its `title` and the DOM), and `trend`, the meter's history (`satSparkline`), in a
 *  slot of its own height that is drawn empty before there are two samples. `graph`: a meter that carries a ring
 *  graph BY ITS KIND (a rank's SM fill: its box's hardware ring, `series`) keeps that slot too, with or without one. */
function satMeter(label, shown, pct, note, series, sub, kind, title, trend, graph) {
  const colour = typeof pct === 'number' ? (kind === 'rate' ? rateColor(pct / 100) : scaleColor(pct / 100)) : null;
  const bar = colour
    ? `<span class="sat-bar"><i style="width:${Math.max(0, Math.min(100, pct)).toFixed(0)}%;background:${colour}"></i></span>`
    : '<span class="sat-bar none"></span>';
  const ring = graph ? `<div class="sat-graph">${series ? sparkline(series, { max: 100, cols: 12, key: label }) : ''}</div>` : '';
  return `<div class="sat-meter${shown == null ? ' dim' : ''}"${title ? ` title="${esc(title)}"` : ''}>
    <span class="sat-label">${esc(label)}</span>
    <span class="sat-sub">${sub ? esc(sub) : ''}</span>
    ${ring}
    <b class="sat-val">${shown == null ? '—' : esc(shown)}</b>
    ${bar}
    <span class="sat-note"${cutTitle(note)}>${note || ''}</span>
    <div class="sat-trend">${trend || ''}</div>
  </div>`;
}

/** The last few minutes of one saturation series, as a sparkline. Gaps stay gaps: a sample the
 *  LB could not measure is a break in the line, never a zero, or a quiet pair would read as a
 *  crash. Scaled — height and colour — to `ceil` when one is configured, else to the series' own
 *  max, so the shape is readable even when the numbers are small. `key` names whose line it is. */
function satSparkline(values, ceil, key, extra) {
  if ((values || []).filter(isNum).length < 2) return '';
  return sparkline(values, Object.assign({ max: ceil > 0 ? ceil : 0, key: `${key}|sat`, cls: 'sat-sparkline' }, extra));
}

/** THE BANDWIDTH THE LB MODELS for an SGLang or TensorFold pair (`saturation.bandwidth_model`, throughput.py
 *  `bandwidth_model`) — SGLang's own byte estimate read 73,600 GB/s a GPU and TensorFold publishes none, so the LB
 *  prices each forward pass itself (catalog `traffic`). Its parts as figures for the hover, and whether the figure is a floor (a window that held
 *  a prefill pass): null for a bandwidth the engine estimated, which is drawn as it always was. */
function bandwidthModel(sat) {
  const m = sat && sat.bandwidth_model;
  if (!m) return null;
  const gb = (v) => (isNum(v) ? v.toFixed(0) : '—');
  const per = (v) => (isNum(v) ? v.toFixed(1) : '—');
  return {
    floor: !!m.floor,
    title: `modeled from the engine's own pass counters, per GPU: weights ${gb(m.weights_gbps)} · `
      + `KV ${gb(m.kv_gbps)} · `
      + `GDN state ${gb(m.state_gbps)} GB/s · ${per(m.decode_passes_s)} decode + ${per(m.prefill_passes_s)} `
      + `prefill passes/s` + (m.floor ? ` · floor: ${m.floor}` : ''),
  };
}

/** THE FLOP/s THE LB MODELS for an SGLang pair (`saturation.compute_model`, cores.py `modeled_flops`) — the one
 *  figure the CUDA CORES tile draws too. SGLang's own FLOP counter charges the 27B's 64 layers as dense
 *  attention with an unsharded MLP (236.8 TFLOP/s beside 742 cores multiplying, 2026-10-03), so it is not the
 *  figure: it stays in the hover beside the model's parts. Null for a rate the engine estimated (vLLM). */
function computeModel(sat) {
  const m = sat && sat.compute_model;
  if (!m) return null;
  const tf = (v) => (isNum(v) ? v.toFixed(1) : '—');
  const kind = (name, rate, passes, seconds, extra) => (isNum(passes) && passes > 0
    ? `${name} ${tf(rate)} TFLOP/s · ${fmtInt(Math.round(passes))} passes${extra}, ${isNum(seconds) ? (seconds / passes).toFixed(2) : '—'} s each`
    : `no ${name} pass`);
  // HOW DEEP its prompt tokens sat: read back out of SGLang's own counter ('engine'), or approximated from the
  // cached prefix when that counter is off — then marked "~".
  const deep = isNum(m.prefill_depth)
    ? `, ${m.depth_from === 'engine' ? '' : '~'}${Math.round(m.prefill_depth / 1000)}k deep` : '';
  const chunk = isNum(m.prefill_chunk) ? ` of ${fmtInt(Math.round(m.prefill_chunk))} tok` : '';
  return {
    floor: false,
    title: `modeled from the engine's own pass counters, per GPU, over ${isNum(m.window_s) ? m.window_s.toFixed(0) : '—'} s: `
      + `${kind('prefill', m.prefill_tflops, m.prefill_passes, m.prefill_s, `${chunk}${deep}`)} · `
      + `${kind('decode', m.decode_tflops, m.decode_passes, m.decode_s, '')}`
      + (m.split ? ` · busy time split: ${m.split}` : '')
      + (isNum(m.engine_tflops) ? ` · SGLang's own estimate ${m.engine_tflops.toFixed(1)} TFLOP/s` : ''),
  };
}

/** A Ceiling from /lb/status ({rate, peak, pct, why}) as a meter. `unit` and `digits` are how the
 *  operator reads that rate; `why` is shown verbatim, because it is the LB's sentence, not ours.
 *  `model`: `bandwidthModel`'s reading when the rate is the LB's model — "modeled" under the label, its
 *  parts in the hover, "≥" on a floor. */
function ceilingMeter(label, c, unit, digits, series, kind, model) {
  const o = c || {};
  // A box's share of the engine's one rate (`perBox`) is DERIVED: it says so under its label, the sentence in the hover.
  const sub = [model ? 'modeled' : null, o.derived ? 'derived' : null].filter(Boolean).join(' · ') || null;
  const title = [model ? model.title : null, o.derived || null].filter(Boolean).join(' — ') || undefined;
  const shown = typeof o.rate === 'number'
    ? `${model && model.floor ? '≥ ' : ''}${o.rate.toFixed(digits)} ${unit}` : null;
  const note = shown == null
    ? `<span class="sat-why">${whyText(o.why)}</span>`
    : (typeof o.pct === 'number'
      ? `${o.pct.toFixed(0)} % of ${fmtNum(o.peak).text} ${unit}` +
        (o.peak_precision ? ` (${esc(o.peak_precision)})` : '')
      // A share can be missing because no peak is configured OR because the LB withheld it; only
      // the LB knows which, so its sentence wins whenever it gave one.
      : (o.why ? `<span class="sat-why">${whyText(o.why)}</span>` : 'no peak configured'));
  // Memory bandwidth is a LOAD — 120 % of the bus is red. Compute is a RATE: grey when little of the
  // predefined peak is used, blue near 40 %, glowing from 80 % — the same scale as tok/s, with the
  // nameplate as its 100 % instead of a benchmark.
  const opts = kind === 'rate' ? { kind: 'rate', target: o.peak } : {};
  return satMeter(label, shown, o.pct, note, null, sub, kind, title, satSparkline(series, o.peak, label, opts));
}

/** A ceiling PER BOX USED (/lb/status `compute.boxes`, `bandwidth.boxes`: saturation.py `per_box`): each serving
 *  box's rate against ITS OWN cards' peak — never one figure for the pair (operator, 2026-10-11). Where more than
 *  one card serves, a box's rate is DERIVED from the engine's one rate (`derived`, the server's sentence) and every
 *  figure drawn from it says so. A box's `why` is its own (its card states no peak); with no rate at all the pair's
 *  reason (an MFU switch that is off) is the box's too. */
function perBox(c) {
  const o = c || {};
  return (Array.isArray(o.boxes) ? o.boxes : []).map((b) => ({ box: b.box, cards: b.cards, rate: b.rate,
    peak: b.peak, pct: b.pct, derived: b.derived || null, peak_precision: o.peak_precision,
    why: b.why || (isNum(b.rate) ? null : o.why) }));
}

/** The pair's per-GPU past as a box's: times the box's serving cards (one, on every box today). */
const boxSeries = (series, b) => (Array.isArray(series) && b.cards > 1
  ? series.map((v) => (isNum(v) ? v * b.cards : v)) : series);

/** What one GB10 box's driver says: clock, power, temperature — each only when the box sent it, never
 *  zeroed. The clock is shown against its OWN ceiling, because 2424 of 3003 MHz on a memory-bound
 *  decode is normal and reads as throttling without it. `withFill` leads with the SM fill, for a line
 *  that has no meter of its own. */
function gpuBits(g, withFill) {
  const bits = withFill ? [`SM fill <b>${g.sm_pct.toFixed(0)}%</b>`] : [];
  if (typeof g.pclk_mhz === 'number') {
    const of = typeof g.pclk_max_mhz === 'number' ? `/${g.pclk_max_mhz.toFixed(0)}` : '';
    const pct = typeof g.clock_pct === 'number' ? ` (${g.clock_pct.toFixed(0)}%)` : '';
    bits.push(`clock <b>${g.pclk_mhz.toFixed(0)}${of}</b> MHz${pct}`);
  }
  if (typeof g.pwr_w === 'number') bits.push(`<b>${g.pwr_w.toFixed(0)}</b> W`);
  if (typeof g.gtemp_c === 'number') bits.push(`<b>${g.gtemp_c.toFixed(0)}</b> &deg;C`);
  // Each bit is one piece to the line: "60 °C" never breaks between its figure and its unit.
  return bits.map((b) => `<span class="gb">${b}</span>`);
}

/** The same, as a line in its box row. Absent when the box has no nvidia-smi or the telemetry is off. */
function gpuLine(g) {
  if (!g || typeof g.sm_pct !== 'number') return '';
  return `<span class="cb-gpu">${gpuBits(g, true).join(' &middot; ')}</span>`;
}

/** What each rank does in the pair, as the meter under its label says it. */
const RANK_ROLE = { head: 'head', worker: '' };

/** ONE RANK's SM fill as its own meter: what THAT box's nvidia-smi said, with its own history — the
 *  graph from the hardware broadcast's ring for the box, the line from the LB's saturation ring for the
 *  box. A rank whose box sent nothing keeps its place, as a gap with the LB's reason, never a zero. */
function rankMeter(b, hist, hw) {
  const fill = typeof b.sm_pct === 'number' ? b.sm_pct : null;
  const role = b.role in RANK_ROLE ? RANK_ROLE[b.role] : (b.role || '');
  const sub = [typeof b.rank === 'number' ? `rank ${b.rank}` : '', role].filter(Boolean).join(' · ');
  const bits = fill == null ? [] : gpuBits(b, false);
  const note = fill == null
    ? `<span class="sat-why">${esc(b.why || 'not observed')}</span>`
    : (bits.length ? bits.join(' &middot; ') : '');
  const ring = ((hw[b.box] || {}).history || {}).gpu_pct;
  return satMeter(`SM fill · ${b.box}`, fill == null ? null : `${fill.toFixed(0)} %`, fill, note, ring, sub,
    undefined, undefined, satSparkline(((hist || {}).boxes || {})[b.box], 100, b.box), true);
}

/** Rank 0's SM fill minus rank 1's. Two ranks of one model step in lockstep, so a gap is time one of
 *  them had NO kernel resident — host-side work, or a wait outside a collective. Both samples are one
 *  ~1 s nvidia-smi reading per box, taken together each cluster tick; and a rank spinning in an NCCL
 *  wait still reads as filled, so an even pair does not prove that neither rank waited. */
function rankGap(sat) {
  const g = sat.sm_fill_gap_pts;
  const bxs = sat.boxes || [];
  if (typeof g !== 'number' || bxs.length < 2) return '';
  const sign = g > 0 ? '+' : (g < 0 ? '−' : '±');
  return `<div class="sat-gap" title="SM fill, rank 0 minus rank 1: one ~1 s nvidia-smi sample per box, both taken each tick; a rank spinning in an NCCL wait still reads as filled">${esc(bxs[0].box)} − ${esc(bxs[1].box)}: <b>${sign}${Math.abs(g).toFixed(0)} pts</b></div>`;
}

// ── what the pair is PRODUCING, between what the GPUs do and what it costs ────
// The operator asked for "tok/s generate and fill on two rows on left of memory bandwidth", and the
// order is the argument: the SM fill pair says the GPUs are busy, these two rows say what that busy-ness
// produced in tokens, and the bandwidth and compute beside them say what it cost. Both numbers come off
// the SAME `throughput` block the server cards read (throughput.py) — the pair is ONE server in
// /lb/status, and its counters are the head rank's, the only rank that serves /metrics. No second path
// to the same number, and no /metrics scraped from here.
//
// WHICH PREFILL RATE, and it is a deliberate choice, because `throughput` carries two and they differ by
// an order of magnitude on this fleet:
//
//   prompt.tok_s         what the engine BILLED — every prompt token a client sent, credited in one lump
//                        at that request's first output token. In the captured document behind
//                        tests/cluster_tokens.node.js it reads 6,230 tok/s with `cached_share` 0.90: nine
//                        tokens in ten were prefix-cache hits no GPU ever read.
//   prefill_speed.tok_s  what it COMPUTED: `vllm:prefill_computed_tokens_sum / vllm:prefill_time_sum`
//                        over the requests that FINISHED in the last five minutes — cache hits excluded
//                        by the counter itself, per second actually spent prefilling. 1,918 tok/s.
//
// Beside a bandwidth and a FLOP/s that this hardware really paid, only the second is an honest
// neighbour: the billed rate would tell the operator the pair reads prompts ten times faster than it
// does. So the computed one is shown, the caption says which it is, and the row names its denominator
// under its label — it is per second spent PREFILLING, not per second of wall clock like GENERATED
// beside it.
//
// No peak tok/s is measured for this pair (nothing in this repo establishes one), so no share of one is
// claimed and both bars stay empty — the same way `ceilingMeter` draws a rate with no peak behind it.

/** The TP=2 pair as a server in /lb/status: one backend of kind `cluster`, whose `throughput` is the
 *  pair's. None when this LB's status carries no such server. */
/** The SELECTED server's head (`headOf` its view): whose tokens the Saturation section shows beside its saturation. */
const clusterBackend = (s) => headOf(s, theCluster(s));

/** GENERATED on the saturation row: what the pair emitted over the last probe window — the same two
 *  probes the engine's own FLOP/s and GB/s are taken over, so the tokens and their cost are read off one
 *  measurement. A measured 0 stays a 0 and says `idle`, because the counter WAS read twice and did not
 *  move; a null is `—` with the LB's own reason, never a zero. */
function clusterGeneratedMeter(s, b) {
  const g = (b.throughput || {}).generated || {};
  const f = fmtNum(g.tok_s);
  const over = fmtDuration(g.window_s);
  const note = f.text == null
    ? `<span class="sat-why">${whyText(g.why)}</span>`
    : [over ? `over ${over}` : null, g.tok_s === 0 ? 'idle' : null].filter(Boolean).join(' &middot; ');
  // tok/s: higher is better, so its line is read on the good scale against the setup's target.
  return satMeter('generated', f.text == null ? null : `${f.text} tok/s`, null, note, null,
    null, null, 'output tokens per second, all clients — the pair\'s own counter, read off the head rank',
    satSparkline(seriesOf(s, b.history, 'gen_tok_s'), 0, 'generated', targetOf(s, b, 'gen_tok_s')));
}

/** PREFILL on the saturation row: the operator's "fill". The COMPUTED rate (see the note above), with
 *  the sample it was taken over — and, when it is not a number, the LB's reason plus that same sample,
 *  so an empty window reads as "no request finished in it" rather than as a silence.
 *
 *  Its graph is its OWN past: the hub's `prefill_tok_s` ring, sampled from this same `prefill_speed` on
 *  every probe — never the billed `prompt_tok_s` that PROMPT draws, which counts the prefix cache's hits
 *  as if they were read. The rate moves only when a request finishes its prefill or leaves the
 *  five-minute window, so the line is a staircase, and that is what the number did. No target: the
 *  setup's recorded best is a BILLED prompt rate, another quantity, so the line is read against its own
 *  peak. */
function clusterPrefillMeter(s, b) {
  const p = (b.throughput || {}).prefill_speed || {};
  const f = fmtNum(p.tok_s);
  const over = fmtDuration(p.window_s);
  const sample = [typeof p.requests === 'number'
    ? `${fmtInt(p.requests)} request${p.requests === 1 ? '' : 's'}` : null,
                  over ? `over ${over}` : null].filter(Boolean).join(' ');
  const note = [f.text == null ? `<span class="sat-why">${whyText(p.why)}</span>` : null, sample]
    .filter(Boolean).join(' &middot; ');
  return satMeter('prefill / request', f.text == null ? null : `${f.text} tok/s`, null, note, null,
    null, null, 'one request\'s pace: prompt tokens computed per second it spent prefilling — prefix-cache hits never inflate it',
    satSparkline(seriesOf(s, b.history, 'prefill_tok_s'), 0, 'prefill', { kind: 'rate' }));
}

/** The two token rows as a group of their own; what their numbers are is each meter's title, as it is
 *  for the two groups either side (`sm_fill_means`, `rates_means`). Nothing is drawn when this LB's
 *  status has no cluster server to read. */
function clusterTokenGroup(s) {
  const b = clusterBackend(s);
  if (!b) return '';
  return `<section class="sat-group sat-tokens" aria-label="what the pair is producing, in tokens"
      title="per engine: the pair's own counters, read off the head rank; prefill is the rate the GPUs computed at, so prefix-cache hits never inflate it">
      <div class="sat-meters">${clusterGeneratedMeter(s, b)}${clusterPrefillMeter(s, b)}</div>
    </section>`;
}

function renderSaturation(s) {
  const el = $('cluster-saturation');
  if (!el) return;
  const sat = tileGroupsOf(theCluster(s)).includes('saturation') ? figureOf(s, theCluster(s), 'saturation') : null;
  el.hidden = !sat;
  const section = $('sat-section');          // the headed section it lives in, below the three parts
  if (section) section.hidden = !sat;
  if (!sat) return;
  const tone = SAT_TONE[sat.verdict] || 'warn';
  const hist = sat.history || {};
  // Each rank's graph is ITS box's ring off the hardware broadcast — five minutes of gpu_pct, per box.
  const hw = s.hardware || {};
  const totals = [
    typeof sat.cluster_gbps === 'number' ? `${sat.cluster_gbps.toFixed(0)} GB/s` : null,
    typeof sat.cluster_tflops === 'number' ? `${sat.cluster_tflops.toFixed(1)} TFLOP/s` : null,
  ].filter(Boolean);
  // The verdict is the word; WHY (`because`, the LB's sentence) and what each group measures
  // (`sm_fill_means`, `rates_means`) are titles — a hover, never a row of prose over the meters.
  const over = typeof sat.window_s === 'number' ? ` &middot; over ${sat.window_s.toFixed(0)} s` : '';
  const from = sat.rates_from ? `read off ${sat.rates_from}, the head rank` : 'read off the head rank';
  // WHOSE estimate it is, from the profile on the pair — the engine behind :8010 has not always been
  // vLLM since 2026-10-03, and this caption credited vLLM over a serving SGLang pair. The LB's own
  // `rates_means` still wins where it sends one; this is the page's fallback when it does not.
  const ratesEngine = engineName((clusterModelInfo(s) || {}).engine) || 'the engine';
  paint(el, `<div class="sat-verdict ${tone}" title="${esc(sat.because || '')}">
      <i class="diode ${tone}${tone === 'bad' || tone === 'warn' ? ' pulse' : ''}"></i>
      <span><b>${esc(sat.verdict)}</b></span>
    </div>
    <div class="sat-groups">
      <section class="sat-group" aria-label="measured on each box"
        title="${esc(`per box: nvidia-smi on each box; SM fill is the ${sat.sm_fill_means || 'share of time the SMs had work resident'}`)}">
        <div class="sat-meters">${(sat.boxes || []).map((b) => rankMeter(b, hist, hw)).join('')}</div>
        ${rankGap(sat)}
      </section>
      ${clusterTokenGroup(s)}
      <section class="sat-group" aria-label="the engine's estimate, per box used"
        title="${esc(sat.rates_means || `per GPU: ${ratesEngine}'s analytical estimate, ${from}; the same for both ranks by construction, not measured per box`)}">
        <div class="sat-meters">
          ${perBox(sat.bandwidth).map((b) => ceilingMeter(`memory bandwidth · ${b.box}`, b, 'GB/s', 0,
            boxSeries(hist.gbps, b), undefined, bandwidthModel(sat))).join('')}
          ${perBox(sat.compute).map((b) => ceilingMeter(`compute · ${b.box}`, b, 'TFLOP/s', 1,
            boxSeries(hist.tflops, b), 'rate', computeModel(sat))).join('')}
        </div>
      </section>
    </div>
    <div class="sat-foot"${sat.window_widened ? ` title="${esc(sat.window_widened)}"` : ''}>${totals.length
      ? `${esc(String(sat.gpu_count))} GPUs &middot; ${esc(totals.join(' · '))}` : 'per GPU'}${over}</div>`);
}

/** Replace an element's markup only when it CHANGED. #cluster-state is a live region: writing
 *  the same words back every 3 s would have a screen reader announce them every 3 s. */
const painted = new WeakMap();
function paint(el, html) {
  if (painted.get(el) === html) return;
  painted.set(el, html);
  el.innerHTML = html;
}

// ── press and hold ─────────────────────────────────────────────────────────────
// Start, Stop and the test swarm must not happen by accident, so each fires only once its button has
// been PRESSED AND HELD — 5 s for the cluster, 3 s for a swarm, 1 s for a request's kill × (the app's
// `HoldToConfirm`; the same gesture the power control uses, power/fleet_power/static/button.js). A track along the
// bottom fills while it is held and the label counts the seconds down; letting go, sliding off, losing focus or
// Escape cancels, and nothing is sent. A plain click does nothing but say so. The fill is driven by the
// clock, frame by frame — under `prefers-reduced-motion` it moves in static steps (`holdProgress`).
// Pointer (mouse, pen, touch) and keyboard (Space or Enter held down; auto-repeat ignored) both work.
// The server's contract is unchanged: the LB asks for no confirmation of its own on /lb/cluster or
// /lb/swarm/start, so the hold is the whole guard, and it is the page's.

const reducedMotion = () => !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);

/** How full a hold's fill is `held` ms into a hold of `ms`: the clock's own fraction — a sweep — or, under
 *  prefers-reduced-motion, STATIC STEPS: one per whole second of a hold of seconds, and for a hold of a second or
 *  less (the kill ×) one half step from the press until it fires — a hold that showed nothing at all for its whole
 *  second would read as a press that did nothing. */
function holdProgress(held, ms, reduced) {
  if (!reduced) return held / ms;
  return ms <= 1000 ? 0.5 : Math.floor(held / 1000) * 1000 / ms;
}

function holdButton(btn, ms, onDone) {
  const part = (cls) => (btn.querySelector ? btn.querySelector(cls) : null);
  let t0 = 0, raf = 0, hint = 0;
  const fill = (p) => {
    [part('.hold-fill'), part('.hold-track i')].forEach((el) => { if (el) el.style.transform = `scaleX(${p})`; });
  };
  const say = (text) => { const l = part('.hold-label'); if (l) l.textContent = text; };
  const stop = () => { cancelAnimationFrame(raf); raf = 0; t0 = 0; btn.classList.remove('holding'); };
  function frame() {
    const held = performance.now() - t0;
    if (held >= ms) {
      stop(); fill(0); say(btn.dataset.label || '');
      onDone();
      return;
    }
    fill(holdProgress(held, ms, reducedMotion()));
    say(`Hold… ${Math.ceil((ms - held) / 1000)}`);
    raf = requestAnimationFrame(frame);
  }
  function start(e) {
    e.preventDefault();
    if (btn.disabled || t0) return;
    clearTimeout(hint);
    t0 = performance.now();
    btn.classList.add('holding');
    if (btn.focus) btn.focus({ preventScroll: true });       // so blur can cancel, and the ring shows
    raf = requestAnimationFrame(frame);
  }
  /** `quiet`: the page took the hold back (the state under the button changed); nothing to explain. */
  function cancel(quiet) {
    if (!t0) return;
    stop(); fill(0);
    if (quiet === true) { say(btn.dataset.label || ''); return; }
    say(`Let go too early — hold for ${ms / 1000} s`);
    hint = setTimeout(() => { if (!t0) say(btn.dataset.label || ''); }, 1800);
  }
  const isKey = (e) => e.key === ' ' || e.key === 'Enter';
  btn.addEventListener('pointerdown', (e) => { if (e.button === 0) start(e); });
  ['pointerup', 'pointercancel', 'pointerleave', 'blur'].forEach((ev) => btn.addEventListener(ev, () => cancel()));
  btn.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') cancel();
    else if (isKey(e)) { if (e.repeat) e.preventDefault(); else start(e); }
  });
  btn.addEventListener('keyup', (e) => { if (isKey(e)) cancel(); });
  btn.addEventListener('click', (e) => e.preventDefault());          // a click is not a hold
  btn.addEventListener('contextmenu', (e) => e.preventDefault());    // a long touch is not a menu
  btn.cancelHold = () => cancel(true);
  btn.holding = () => !!t0;
}

/** What a hold button says and whether it can be held. A hold under way is taken back when the button's
 *  meaning changes under the finger (a poll turned Start into Stop) or it stops being available. */
function setHold(btn, label, disabled, kind) {
  if (!btn) return;
  if (btn.dataset.label !== label || disabled) { if (btn.cancelHold) btn.cancelHold(); }
  if (btn.dataset.label !== label) {
    btn.dataset.label = label;
    const l = btn.querySelector ? btn.querySelector('.hold-label') : null;
    if (l && !(btn.holding && btn.holding())) l.textContent = label;
  }
  if (kind && btn.dataset.kind !== kind) {
    btn.dataset.kind = kind;
    btn.className = `hold ${kind}`;
  }
  btn.disabled = !!disabled;
}

/** Why a Start cannot be sent with the selection on screen, in words — or null. The same checks
 *  `toggleCluster` makes before it posts, made while the button is still unheld: five seconds of
 *  holding must never end in "that number is out of range". */
function startProblem(c) {
  const seqs = shownOf('max_num_seqs');
  const [lo, hi] = limitsOf(c, 'max_num_seqs');
  if (!Number.isInteger(seqs) || seqs < lo || seqs > hi) {
    return { key: 'max_num_seqs', why: `concurrency must be a whole number from ${lo} to ${hi}` };
  }
  return selectionProblem(c, clusterChanges(c).filter((ch) => KNOBS.includes(ch.key)));
}

/** THE ONE PRIMARY ACTION for the state — Stop exactly while the switch is on, Start otherwise — and the
 *  one line under it: why it cannot be held, or what holding it does. Never the detail again. */
function renderPrimary(c) {
  const toggle = $('cluster-toggle');
  const el = $('cluster-caption');
  // A NODE THIS LB ONLY OBSERVES has no Start and no Stop here — its server file and compose service do that — so
  // neither is offered; the one sentence why is (`topology.why`), where the button's refusals are written.
  const observed = observedOnly(c);
  toggle.hidden = observed;
  if (observed) {
    el.textContent = topologyOf(c).why;
    el.hidden = false;
    el.classList.toggle('bad', false);
    return;
  }
  // A NODE THE LB REQUESTS: Start / Stop node, held back — with the reason under it — while nobody on its host
  // listens, or (Start) while the selection on screen is one the fleet refuses.
  if (localNode(c)) {
    const a = c.applier || {};
    const on = (c.desired || {}).mode === 'cluster';
    const why = !a.listening ? (a.error || 'no applier listening') : (!on ? localRefusal(c) : '');
    setHold(toggle, on ? 'Stop server' : 'Start server', clusterBusy || !!why, on ? 'stop' : 'start');
    toggle.classList.toggle('busy', clusterBusy);
    el.textContent = why;
    el.hidden = !why;
    el.classList.toggle('bad', !!why);
    toggle.title = on ? 'Press and hold 5 s: the node finishes what it serves, then its host stops it'
      : 'Press and hold 5 s: its host starts the node with the model and settings chosen above';
    return;
  }
  const stop = (c.desired || {}).mode === 'cluster';
  const bad = !stop && c.desired ? startProblem(c) : null;
  setHold(toggle, stop ? 'Stop server' : 'Start server', clusterBusy || !!bad, stop ? 'stop' : 'start');
  toggle.classList.toggle('busy', clusterBusy);
  const state = String(c.state || '');
  // Only a REFUSAL is written under the button; what a Stop does in this state is the button's title.
  const caption = bad ? bad.why : '';
  el.textContent = caption;
  el.hidden = !caption;
  el.classList.toggle('bad', !!bad);
  toggle.title = !stop ? 'Press and hold 5 s: starts the model and settings chosen above'
    : (state === 'running' ? 'Press and hold 5 s: the ranks come down, then its boxes are released'
      : (NEEDS_OPERATOR.includes(state)
        ? 'Press and hold 5 s: brings its ranks down and releases its boxes; then Start again'
        : 'Press and hold 5 s: cancels the start — the ranks come down, then its boxes are released'));
}

/** THE MODEL LINE of the card: the model served or being brought up, with its launch facts and the
 *  concurrency. While Start is offered the choosers under it say what a Start loads, so it is left out. */
function renderClusterModel(c, s) {
  const el = $('cluster-model-line');
  if (!el) return;
  const on = (c.desired || {}).mode === 'cluster';
  const m = on || c.loaded_model ? clusterModelInfo(s) : null;
  el.hidden = !m;
  paint(el, m ? modelBadge(m, clusterFigures(c)) : '');
}

/** The card's own chips beside the model's: its seats, and the KV type it RUNS — the launch record's
 *  (`cluster.effective.kv_cache_dtype`), so a pair still starting with no record says none (operator 2026-10-26: "Why
 *  no KV precision data?"). */
function clusterFigures(c) {
  const on = (c.desired || {}).mode === 'cluster';
  const seqs = (c.effective || {}).max_num_seqs || (on ? (c.desired || {}).max_num_seqs : null);
  const dtype = runningOf(c, 'kv_cache_dtype');
  return [isNum(seqs) ? `${seqs} seats` : null, dtype ? `KV ${kvDtypeName(dtype)}` : null].filter(Boolean);
}

/** A KV dtype as a chip says it: bfloat16 → BF16, fp8_e4m3 → FP8 E4M3, nvfp4 → NVFP4; vLLM's `auto` as itself. */
const kvDtypeName = (d) => ({ bfloat16: 'BF16', float16: 'FP16', auto: 'auto' }[d] || d.toUpperCase().replace(/_/g, ' '));

function renderCluster(s) {
  const c = theCluster(s) || {};
  const panel = $('cluster-panel');
  if (!panel) return;
  // Not configured is not "off": this LB has no cluster to offer, so there is nothing to draw.
  // Hidden, the card's column collapses by the layout's own rule (layout.js `refreshEmpty`), wherever it stands.
  panel.hidden = !c.configured;
  if (!c.configured) return;
  panel.className = `cluster ${clusterTone(c.state)}`;
  renderClusterHead(c, s);
  // The start's clock FIRST: the headline inside #cluster-state and the trail outside it both read
  // this one value, so they can never show the same step in two different states.
  startupNow = watchStartup(c, s);
  paint($('cluster-state'), clusterStateRow(c, s));
  renderSince(c, s);
  renderStartup(c);
  renderClusterModel(c, s);
  renderMismatch(c);
  renderSaturation(s);
  // A NODE THIS LB ONLY OBSERVES has no switch to edit and no planner looking at its box: its settings are what it
  // SAYS it runs with and what its fleet declares, locked (`renderObservedSettings`); the estimate and the per-box
  // planner rows are the pair's. Its one action line says why there is no Start/Stop.
  const observed = observedOnly(c);
  const local = localNode(c);
  $('cluster-boxes-more').hidden = observed || local;
  if (!local) LOCAL_LINES.forEach((id) => { const el = $(id); if (el) el.hidden = true; });
  renderApplier(c);
  if (observed) {
    renderObservedSettings(c, s);
    renderPrimary(c);
    return;
  }
  // A NODE THE LB REQUESTS: its own settings (the fleet's choices, seats, context, GPU memory) and Start/Stop.
  if (local) {
    renderLocalSettings(c, s);
    renderPrimary(c);
    return;
  }
  OBSERVED_LINES.forEach((id) => { const el = $(id); if (el) el.hidden = true; });
  $('cluster-settings').hidden = false;
  renderClusterSettings(c, s);           // … and, through its actions, the one primary button
  renderClusterEstimate(c);
  paint($('cluster-boxes'), Object.entries(c.boxes || {})
    .map(([box, o]) => clusterBoxRow(s, box, o || {})).join(''));
}

/** The card's title and subtitle, from what the server IS (`topologyOf`): its name, and `TP=2 on box-a + box-b` for
 *  a pair, `TP=1` for one node — and its own pages: performance and sessions scoped by `?server=<name>`, connect for
 *  what its head serves (`serverOf`), disabled with the reason while it serves nothing. */
function renderClusterHead(c, s) {
  const t = topologyOf(c);
  const sub = $('cluster-sub');
  if (sub) paint(sub, esc(t.tp === 1 ? 'TP=1' : `TP=${t.tp} on ${t.boxes.join(' + ')}`));
  // Titled by its fleet NAME (a document from before names: its boxes) on EVERY render — picking gui, then the pair,
  // left the node's box name up (AUDIT-02 P1). No kind of box is named: the fleet says what it is.
  const title = $('cluster-title');
  if (title) paint(title, esc(c.name || t.boxes.join(' + ')));
  const links = $('cluster-links');
  if (!links) return;
  links.hidden = !c.name;
  const q = c.name ? `?server=${encodeURIComponent(c.name)}` : '';
  const perf = $('cluster-perf'), sess = $('cluster-sessions'), connect = $('cluster-connect');
  if (perf) perf.href = `/benchmarks${q}`;
  if (sess) sess.href = `/sessions${q}`;
  if (connect) {
    const srv = serverOf(c, s);
    connect.disabled = !srv;
    connect.title = srv ? `connect a client to ${srv.name}: ${srv.model.id} at ${srv.url}` : `${c.name || 'this server'} serves no model yet`;
  }
}
/** WHAT A CLIENT CONNECTS TO on server `c`: its head's served model and window (`backends[].served_models`,
 *  `max_model_len`, the LB's own reading of its /v1/models) and the head's address — null while it serves nothing. */
function serverOf(c, s) {
  const b = headOf(s, c);
  const id = b && Array.isArray(b.served_models) ? b.served_models[0] : null;
  if (!b || !b.live || typeof id !== 'string' || !Number.isInteger(b.max_model_len) || b.max_model_len <= 0) return null;
  return { name: c.name, url: b.url, model: { id, max_model_len: b.max_model_len } };
}
(() => {
  const el = $('cluster-connect');
  if (!el || !el.addEventListener) return;
  el.addEventListener('click', () => {
    const srv = lastStatus ? serverOf(theCluster(lastStatus), lastStatus) : null;
    if (srv) openConnect(srv);
  });
})();
function showClusterError(msg) {
  const el = $('cluster-error');
  el.textContent = msg;
  el.hidden = !msg;
}

/** The operator's intent, posted once — by a completed hold. The button stays disabled until the
 *  answer is in AND the page has re-read the state it caused, so a second hold cannot race the first.
 *  A refusal is shown IN the panel, in the server's words — the header line is overwritten by the next
 *  poll within 3 s. */
async function toggleCluster() {
  const c = (lastStatus && theCluster(lastStatus)) || {};
  if (clusterBusy || !c.configured) return;
  if (localNode(c)) return toggleLocal(c);
  const start = (c.desired || {}).mode !== 'cluster';
  const body = { enabled: start };
  if (start) {
    // WHICH MODEL: the operator's pick travels with the Start, so the pair loads what the page shows.
    const row = pickedRow();
    if (row) {
      body.model_choice = row.model;
      if (takesRows(c)) body.uncensored = row.uncensored;   // the checkpoint, to an LB that takes it
      if (estimatedChosen()) body.estimated = true;          // the operator's explicit choice, never the page's
    }
    // The concurrency ON SCREEN — a value typed through "custom…" included (`shownOf`) — within what the
    // model in the picker may be started with.
    const seqs = shownOf('max_num_seqs');
    const [lo, hi] = limitsOf(c, 'max_num_seqs');
    if (!Number.isInteger(seqs) || seqs < lo || seqs > hi) {
      showClusterError(`concurrency must be a whole number from ${lo} to ${hi}`);
      return;
    }
    body.max_num_seqs = seqs;
    const len = Number($('cluster-context').value);
    if (Number.isInteger(len) && len > 0) body.max_model_len = len;
    // The engine knobs travel only when CHANGED: absent keeps what is stored, and a default stays one.
    const knobs = clusterChanges(c).filter((ch) => KNOBS.includes(ch.key));
    const bad = selectionProblem(c, knobs);
    if (bad) {
      refuseSelection(bad);
      return;
    }
    knobs.forEach((ch) => { body[ch.key] = ch.send; });
  }
  closeConfirm();
  clusterBusy = true;
  renderBusy();
  try {
    await postCluster(body);
  } finally {
    clusterBusy = false;
    renderBusy();
  }
}

/** POST the switch once, then re-read the state it caused. The caller holds `clusterBusy`. */
async function postCluster(body) {
  showClusterError('');
  try {
    const r = await fetch(clusterRoute(), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!r.ok) throw new Error(await refusal(r));
    // Accepted: the values are in the desired state now, and the re-poll shows them from there.
    CLUSTER_FIELDS.concat(LOCAL_FIELDS).forEach((f) => { const el = $(f.id); if (el) delete el.dataset.dirty; });
    document.querySelectorAll('[id^="cluster-set-"]').forEach((el) => { delete el.dataset.dirty; });
    estimateShown = null;               // the stored selection's own estimate takes over again
  } catch (e) {
    showClusterError(e.message);        // the edits stay in their fields, to be fixed and sent again
  } finally {
    await poll({ fresh: true });        // the state THIS post caused, not a poll already out
  }
}

/** Every button that posts the switch is disabled while a POST is out — and, for Start, while the
 *  selection on screen is one the LB would refuse (`startProblem`). */
function renderBusy() {
  const c = lastStatus && theCluster(lastStatus);
  if (c && c.configured && localNode(c)) {
    renderLocalSettings(c, lastStatus);
    renderPrimary(c);
  } else if (c && c.configured) {
    renderClusterActions(c);
  } else {
    $('cluster-toggle').disabled = clusterBusy;
  }
}

// ── machines: one card per box ───────────────────────────────────────────────
// The app's `BoxSection` (FleetStats/Shared/Sections.swift `BoxRings`, `BoxGraphs`): the cores, then
// the CPU they belong to, the GPU and memory as rings with the memory figures beside them, then the
// three histories on the 0–100 load scale, the full row wide. Everything is the box's own hardware
// broadcast (`/lb/status.hardware`); a box that is not broadcasting says so — blank, never zero.
//
// These cards are where the power control lands. /power/button.js looks for `#lb-backends article.card`,
// reads the box from the first text of `.box-name` and appends its control to `.card-head`; it puts the
// control back when a card is ADDED to #lb-backends. So each card is built once and kept — only its
// insides are repainted — and the control, with any keyboard focus on it, survives every poll.

/** An idle core is grey (the style's --core-idle), not the scale's blue: under 10 % it reads as "not in use",
 *  and only a core doing real work takes a colour (the app's `CoreMatrix.color`). */
const CORE_IDLE = 10;
const coreColour = (pct) => (pct < CORE_IDLE
  ? `rgba(${palette.rgb('--core-idle')},${(0.35 + (0.25 * Math.max(pct, 0)) / CORE_IDLE).toFixed(2)})` : scaleColor(pct / 100));

/** The chip's drawing, in the units of the tile's 5:4 frame (gateway.css `.cores`): the gap between cores, the
 *  package's margin around them and its corner, and a pin's length, width and pitch along a side. */
const CHIP = { W: 75, H: 60, gap: 1.5, pad: 3.5, corner: 3.5, pin: 4.5, pinW: 2.4, pitch: 7.5 };

/** The cores DRAWN AS A CPU (operator, 2026-10-12: "Can we make inside nice CPU svg shape?"): a package with pins
 *  on its four sides, and inside it one rounded square per logical core, core 0 first, left to right, top to bottom,
 *  in the rows × columns that make the biggest square cells in the room the package leaves (on a tie, the fewest
 *  empty slots). The package wraps the grid that results — a 4 × 4 is a square chip, a 5 × 4 a wide one — and its
 *  pins sit at one pitch along each side, so how many there are follows the grid, never a box. No figures: the tag
 *  under it (`cpuColumn`) and the CPU graph carry the numbers; this shows how the load is spread. Its name says the
 *  box's load as the tag does (`load`: "52 %", or "not reported"). */
function coreMatrix(cores, load) {
  const list = (cores || []).map((v) => (isNum(v) ? v : 0));
  const n = list.length;
  if (!n) return '';
  const { W, H, gap, pad, corner, pin, pinW, pitch } = CHIP;
  const roomW = W - 2 * (pin + pad), roomH = H - 2 * (pin + pad);
  let best = { rows: 1, cols: n, cell: 0 }, bestEmpty = Infinity;
  for (let rows = 1; rows <= n; rows++) {
    const cols = Math.ceil(n / rows);
    const cell = Math.floor(Math.min((roomW - (cols - 1) * gap) / cols, (roomH - (rows - 1) * gap) / rows) * 2) / 2;
    const empty = rows * cols - n;
    if (cell > best.cell || (cell === best.cell && empty < bestEmpty)) { best = { rows, cols, cell }; bestEmpty = empty; }
  }
  const { rows, cols, cell } = best;
  const pw = cols * cell + (cols - 1) * gap + 2 * pad, ph = rows * cell + (rows - 1) * gap + 2 * pad;
  const px = (W - pw) / 2, py = (H - ph) / 2;
  const f = (v) => v.toFixed(1);
  // Pin centres along a side of `len`: as many as fit at the pitch clear of the corners (two at least), centred.
  const along = (len) => {
    const k = Math.max(2, Math.floor((len - 2 * (corner + pinW)) / pitch) + 1);
    const step = Math.min(pitch, (len - 2 * corner - pinW) / (k - 1));
    return Array.from({ length: k }, (_, i) => (len - (k - 1) * step) / 2 + i * step - pinW / 2);
  };
  const pins = along(pw).flatMap((o) => [[px + o, py - pin, pinW, pin], [px + o, py + ph, pinW, pin]])
    .concat(along(ph).flatMap((o) => [[px - pin, py + o, pin, pinW], [px + pw, py + o, pin, pinW]]))
    .map(([x, y, w, h]) => `<rect x="${f(x)}" y="${f(y)}" width="${w}" height="${h}" rx="0.5"/>`).join('');
  const r = Math.min(1.5, cell * 0.2);
  const rects = list.map((pct, i) => `<rect x="${f(px + pad + (i % cols) * (cell + gap))}" `
    + `y="${f(py + pad + Math.floor(i / cols) * (cell + gap))}" width="${cell}" height="${cell}" rx="${r}" fill="${coreColour(pct)}"/>`).join('');
  return `<svg class="cores" viewBox="0 0 ${W} ${H}" role="img"
    aria-label="CPU · ${load} · ${n} core${n === 1 ? '' : 's'} · busiest ${Math.round(Math.max(...list))} %"><g class="chip-pins">${pins}</g>`
    + `<rect class="chip-pkg" x="${f(px)}" y="${f(py)}" width="${f(pw)}" height="${f(ph)}" rx="${corner}"/>`
    + `<g class="chip-cores">${rects}</g></svg>`;
}

/** A machine figure's percentage in its colour: the load scale at value / 100, null where it is not known. ONE rule for
 *  the rings (GPU, MEM) and the CPU's tag. */
const loadColour = (pct) => (isNum(pct) ? scaleColor(pct / 100) : null);

/** A percentage as a ring, on the load scale, with the metric's name under it. */
function ringGauge(label, value) {
  const has = isNum(value);
  const share = has ? Math.min(Math.max(value / 100, 0), 1) : 0;
  const around = 2 * Math.PI * 16;
  const colour = loadColour(value);
  return `<div class="ring" role="img" aria-label="${esc(label)} ${has ? `${Math.round(value)}%` : 'not reported'}">
    <span class="ring-dial"><svg viewBox="0 0 40 40" aria-hidden="true"><circle class="ring-bg" cx="20" cy="20" r="16"/>`
    + (share > 0 ? `<circle class="ring-fg" cx="20" cy="20" r="16" stroke="${colour}" stroke-dasharray="${(share * around).toFixed(2)} ${around.toFixed(2)}" transform="rotate(-90 20 20)"/>` : '')
    + `</svg><b${colour ? ` style="color:${colour}"` : ''}>${has ? Math.round(value) : '—'}</b></span>
    <span class="ring-label">${esc(label)}</span></div>`;
}

/** THE LINE UNDER A MACHINE GLYPH — the CPU's chip, a GPU's card: its label, then its load as a tag in `loadColour`
 *  (operator, 2026-10-12: "just add CPU label and percentage tag with color … below CPU svg next to each other"). A load
 *  not known is "—" with its reason in the title, never 0 and never a colour. `kind` names the classes (`cpu-line`,
 *  `gpu-line` …) so each glyph styles its own; `hidden` when the glyph's name already says the same value. */
function glyphLine(kind, label, pct, why, hidden) {
  const tag = isNum(pct) ? `<b class="${kind}-tag" style="--tag:${loadColour(pct)}">${Math.round(pct)}%</b>`
    : `<b class="${kind}-tag none" title="${esc(why)}">—</b>`;
  return `<div class="${kind}-line"${hidden ? ' aria-hidden="true"' : ''}>`
    + `<span class="${kind}-label">${esc(label)}</span>${tag}</div>`;
}

/** Why a box's figure is not known: the box went quiet (its age says how long), or it broadcasts without that figure. */
const missingWhy = (h, what) => (h.stale ? `no sample for ${Math.round(h.age_s)}s` : `the hardware broadcast carries no ${what}`);

/** THE BOX'S CPU, DRAWN ONCE (operator, 2026-10-12: "unify this into one, we don't need circle, just add CPU label and
 *  percentage tag with color … below CPU svg next to each other"): the chip where the box sends its cores, and under
 *  it one line — "CPU", then the load as a tag (`glyphLine`). The chip's name carries the same value, so the line
 *  under it is hidden from a screen reader; with no chip it is read. */
function cpuColumn(h) {
  const has = isNum(h.cpu_percent);
  const chip = coreMatrix(h.cpu_per_core, has ? `${Math.round(h.cpu_percent)} %` : 'not reported');
  return `<div class="cpu">${chip}${glyphLine('cpu', 'CPU', h.cpu_percent, missingWhy(h, 'CPU load'), !!chip)}</div>`;
}

/** The card's drawing, in the CPU chip's 5:4 frame (`CHIP`), so the two glyphs stand the same size side by side: the
 *  bracket at the card's end and the board ([x, y, w, h]), the board's corner, the PCIe fingers along its bottom edge
 *  (first x, width, pitch, length, slots, and the slot left empty as the key notch), the fan's centre height, track
 *  radius and stroke, its blades and hub, and the VRAM chips' grid (origin, columns × rows, a chip's size, the gap). */
const CARD = { bracket: [1, 4, 3, 50], board: [6, 6, 67, 40], corner: 3.5, finger: [15, 1.6, 2.6, 4.5, 20, 5],
  fanY: 26, fanR: 15, fanW: 4, blades: 7, hub: 3.5,
  vram: { x: 46, y: 20, cols: 4, rows: 1, w: 5.5, h: 9, gap: 1, labelY: 15, pctY: 40 } };

/** THE FAN TURNS AT THE GPU'S UTILISATION (operator 2026-10-30: "make GPU rotate in svg based on usage").
 *  The mapping: the blades' SPEED (rev/s) is linear in the utilisation, from one turn per `slow` s at `still` % to
 *  one per `fast` s at 100 % (over 100 % is 100 %). Linear in speed, not in period, so twice the load looks twice as
 *  fast; the period is 1 / speed, to the ms. Under `still` % the fan stands still: a GPU idling at 1 % is idle, and
 *  a crawl the eye cannot tell from still would only say "something moves". */
const FAN_SPIN = { still: 2, slow: 6, fast: 0.6 };

/** The blades' spin at `pct` and the page clock `now` (ms, `performance.now()`; injected so it is testable), or null:
 *  no reading — not reported, a stale box — is NO spin, never a guessed speed (the card is dimmed "not reported").
 *  THE PHASE IS THE CLOCK'S, NOT THE RENDER'S: the panel rewrites the machine card's innerHTML every poll (~2 s), and
 *  a CSS animation starts at 0° when its element is inserted, so the fan would jerk back each poll. A NEGATIVE delay
 *  of (now mod period) starts the new element where a fan turning since the page's clock zero would be — where the
 *  old one was. A change of speed changes the period and so re-phases the blades once; that jump is accepted. */
function fanSpin(pct, now) {
  if (!isNum(pct) || pct < FAN_SPIN.still) return null;
  const { still, slow, fast } = FAN_SPIN;
  const share = (Math.min(pct, 100) - still) / (100 - still);
  const period = +(1 / (1 / slow + share * (1 / fast - 1 / slow))).toFixed(3);
  return { period, delay: +(-((now / 1000) % period)).toFixed(3) || 0 };
}

/** A GPU ROW DRAWN AS A GRAPHICS CARD (operator, 2026-10-30: "try to implement CPU like GPU icon, but make it GPU shape,
 *  fan is the circle, on the side vRAM if applies"), in the CPU chip's language — the same package ground, muted ink
 *  and frame — so the CPU and its GPUs read as one set of parts:
 *    - the board with a bracket at one end and PCIe fingers along its bottom edge: the shape that says "a card";
 *    - its FAN is the ring the row used to be: the utilisation's arc, in `loadColour`, around a hub and blades;
 *    - VRAM chips beside the fan ONLY where the row carries VRAM of its own (`vram_label`, hw.py `rows`: a discrete
 *      card's). Chip i, in reading order, is lit over min(1, max(0, share × n − i)) of its width, so the lit area is
 *      used / total exactly, in the load scale's colour at that share; a used figure not reported lights none. A row
 *      without VRAM (a unified box's GPU, whose memory is the MEM ring) draws none and its fan takes the board's middle.
 *      The page decides by what the row carries, never by asking which kind of box it is.
 *  The number is NOT in the fan: at 64 px the hub is a few px across, and a figure big enough to read would cover the
 *  blades that make it a fan. It is the tag on the line under the card (`glyphLine`), exactly where the CPU's is. A
 *  utilisation not known draws no arc and dims the card (`.gpu-card.none`); the tag is "—" with `why`; never a 0.
 *  Its BLADES turn at the utilisation (`fanSpin`, phased by the page clock `now`), about the fan's own centre — the
 *  board's middle or its left, wherever `cx` put it; the arc, track, hub and card stay still. The turn is decorative:
 *  the card's name already says the figure. */
function gpuColumn(r, pct, why, now = performance.now()) {
  const has = isNum(pct);
  const own = !!r.vram_label;
  const { bracket, board, corner, finger, fanY, fanR, fanW, blades, hub, vram } = CARD;
  const f = (v) => +v.toFixed(2);
  const [bx, by, bw, bh] = board;
  const cx = own ? bx + fanW + fanR : bx + bw / 2;
  const rect = (cls, [x, y, w, h], rx, fill) => `<rect${cls ? ` class="${cls}"` : ''} x="${f(x)}" y="${f(y)}" `
    + `width="${f(w)}" height="${f(h)}" rx="${rx}"${fill ? ` fill="${fill}"` : ''}/>`;
  const [fx, fw, fpitch, fh, fn, notch] = finger;
  const fingers = Array.from({ length: fn }, (_, i) => i).filter((i) => i !== notch)
    .map((i) => rect('', [fx + i * fpitch, by + bh, fw, fh], 0.4)).join('');
  // A blade: an ellipse from near the hub to near the track, swept 30° about its own middle, one per 360 / blades.
  const blade = (i) => { const ex = cx + fanR * 0.43;
    return `<ellipse cx="${f(ex)}" cy="${fanY}" rx="${f(fanR * 0.37)}" ry="${f(fanR * 0.16)}" `
      + `transform="rotate(${f((360 / blades) * i)} ${f(cx)} ${fanY}) rotate(30 ${f(ex)} ${fanY})"/>`; };
  const around = 2 * Math.PI * fanR;
  const share = has ? Math.min(Math.max(pct / 100, 0), 1) : 0;
  const arc = share > 0 ? `<circle class="gpu-fan-fg" cx="${f(cx)}" cy="${fanY}" r="${fanR}" stroke="${loadColour(pct)}" `
    + `stroke-dasharray="${(share * around).toFixed(2)} ${around.toFixed(2)}" transform="rotate(-90 ${f(cx)} ${fanY})"/>` : '';
  let chips = '';
  if (own) {
    const n = vram.cols * vram.rows;
    const used = isNum(r.vram_used_mb) && isNum(r.vram_total_mb) && r.vram_total_mb > 0
      ? Math.min(Math.max(r.vram_used_mb / r.vram_total_mb, 0), 1) : null;
    const at = (i) => [vram.x + (i % vram.cols) * (vram.w + vram.gap), vram.y + Math.floor(i / vram.cols) * (vram.h + vram.gap)];
    const all = Array.from({ length: n }, (_, i) => rect('', [...at(i), vram.w, vram.h], 1)).join('');
    const lit = used === null ? '' : Array.from({ length: n }, (_, i) => Math.min(1, Math.max(0, used * n - i)))
      .map((part, i) => (part > 0 ? rect('', [...at(i), vram.w * part, vram.h], f(Math.min(1, (vram.w * part) / 2)),
        loadColour(used * 100)) : '')).join('');
    // ONE ROW of memory chips — the VRAM's graph — with its name above and its share under it, inside the card beside
    // the fan (operator 2026-10-30: "vram and usage don't have space for two, maybe move vram label on side of the
    // card, and make one row memory graph"): the line under the card keeps the utilisation alone.
    const mid = vram.x + (vram.cols * vram.w + (vram.cols - 1) * vram.gap) / 2;
    const pct = used === null ? '—' : `${Math.round(used * 100)}%`;
    chips = `<g class="gpu-vram${used === null ? ' none' : ''}">${all}</g><g class="gpu-vram-lit">${lit}</g>`
      + `<text class="gpu-vram-label" x="${f(mid)}" y="${vram.labelY}" text-anchor="middle">VRAM</text>`
      + `<text class="gpu-vram-pct${used === null ? ' none' : ''}" x="${f(mid)}" y="${vram.pctY}" text-anchor="middle"`
      + `${used === null ? '' : ` fill="${loadColour(used * 100)}"`}>${pct}</text>`;
  }
  const spin = fanSpin(pct, now);
  const name = `${r.label} · ${has ? `${Math.round(pct)} %` : 'not reported'}${own ? ` · VRAM ${vramOf(r)}` : ''}`;
  return `<div class="gpu"><svg class="gpu-card${has ? '' : ' none'}" viewBox="0 0 ${CHIP.W} ${CHIP.H}" role="img" `
    + `aria-label="${esc(name)}"><title>${esc(name)}</title>`
    + rect('gpu-bracket', bracket, 0.6) + `<g class="gpu-fingers">${fingers}</g>` + rect('gpu-board', board, corner)
    + `<circle class="gpu-fan-bg" cx="${f(cx)}" cy="${fanY}" r="${fanR}"/>`
    + `<g class="gpu-blades${spin ? ` spin" style="--spin: ${spin.period}s; --spin-delay: ${spin.delay}s; `
      + `transform-origin: ${f(cx)}px ${fanY}px` : ''}">${Array.from({ length: blades }, (_, i) => blade(i)).join('')}</g>`
    + `<circle class="gpu-hub" cx="${f(cx)}" cy="${fanY}" r="${hub}"/>${arc}${chips}</svg>`
    + `${glyphLine('gpu', r.label, pct, why, true)}</div>`;
}

/** The machine behind the servers. Blank — never zero — when the box is not broadcasting.
 *
 *  ITS GPUs ARE ITS `rows` (hw.py), laid out by the box's MEMORY MODEL (vllm_lb.hardware, named by the fleet file):
 *  a unified box — a GB10, whose memory IS the box's RAM — sends ONE row, the box's GPU, with no VRAM line of its
 *  own; a discrete box sends one row per card, each with its own utilisation, past and VRAM line (two cards' VRAM
 *  summed is no card's). This page draws the rows it is given and never asks which kind of box it is.
 *
 *  ONE VALUE PER FIGURE: a figure's ring, headline and graph are drawn from one row of `figures` — the
 *  reading and that reading's own past — so the three wear one colour. MEM is the box's RAM:
 *  `ram_percent`, and its past `history.ram_pct`. `history.mem_pct` is the GPU's memory (hw.py
 *  `_memory`), the same number only on a GB10: under box-c's MEM ring it drew the iGPU's 485/512 MB,
 *  a red 95 % graph beneath a green 20 %. */
/** A BOX'S GPU: the hardware broadcast's `gpu_pct` and its ring — or, where the broadcast carries no
 *  GPU for the box (box-b's agent reports `gpu_count: 0` while its rank runs at 96 %), what the
 *  cluster's planner read off that box's nvidia-smi (`cluster.boxes[box].gpu.sm_pct`, the same number
 *  the SM FILL meter shows) with the saturation ring for the box as its past. Blank when neither says. */
function gpuOf(s, box) {
  const h = ((s || {}).hardware || {})[box] || {};
  const ring = (h.history || {}).gpu_pct;
  if (isNum(h.gpu_pct)) return { pct: h.gpu_pct, series: Array.isArray(ring) ? ring : [] };
  // The planner's read of THIS box: the active cluster whose view carries it (each box under one running cluster).
  const view = activeViews(s).find((c) => ((c || {}).boxes || {})[box]);
  const g = (((view || {}).boxes || {})[box] || {}).gpu;
  if (g && isNum(g.sm_pct)) {
    const sat = (((figureOf(s, view, 'saturation') || {}).history || {}).boxes || {})[box];
    return { pct: g.sm_pct, series: Array.isArray(sat) ? sat : [] };
  }
  return { pct: null, series: Array.isArray(ring) ? ring : [] };
}

/** One GPU row's reading: the row's own utilisation and past — or, for the row that IS the box's GPU (`box_gpu`,
 *  a unified box's one row), what `gpuOf` has always read for the box. */
function rowReading(s, box, r) {
  if (r.box_gpu) return gpuOf(s, box);
  return { pct: isNum(r.use_pct) ? r.use_pct : null, series: Array.isArray(r.history) ? r.history : [] };
}

/** A card's own VRAM, as the figure beside the rings says it: GiB, '—' where the card reports none. */
const vramOf = (r) => (isNum(r.vram_total_mb) ? `${isNum(r.vram_used_mb) ? gb(r.vram_used_mb) : '—'}/${gb(r.vram_total_mb)} GiB` : '—');

function machineBody(s, box) {
  const h = (s.hardware || {})[box];
  if (!h) return '<p class="machine-none">no hardware broadcast</p>';
  // The broadcast counts in MiB; ÷1024 is GiB, and the figure says so.
  const whole = (mb) => (isNum(mb) ? (mb / 1024).toFixed(0) : '—');
  const figs = [['memory', `${whole(h.ram_used_mb)}/${whole(h.ram_total_mb)} GiB`]];
  if (h.swap_used_mb > 0) figs.push(['swap', `${(h.swap_used_mb / 1024).toFixed(1)} GiB`]);
  const rows = Array.isArray(h.rows) ? h.rows : [];
  rows.filter((r) => r.vram_label).forEach((r) => figs.push([r.vram_label, vramOf(r)]));
  const hist = h.history || {};
  const gpus = rows.map((r) => { const g = rowReading(s, box, r); return [r.label, g.pct, g.series]; });
  const gauges = [...gpus, ['MEM', h.ram_percent, hist.ram_pct]];
  const figures = [['CPU', h.cpu_percent, hist.cpu_pct], ...gauges];
  const pct = (v) => (isNum(v) ? Math.round(v) : null);
  const graphOf = (label, v, series) => tile(label, pct(v), '%', null, series,
    { max: 100, cols: 60, key: `${box}|${label}`, dim: !!h.stale, noNote: true });
  // The CPU is its column (`cpuColumn`); each GPU row is a card (`gpuColumn`), then MEM's ring. Two glyph columns after
  // the CPU is a one-GPU box and gateway.css's default; each further row widens that grid (`--rings`) and the graph
  // row (tiles4, tiles5) by one.
  // `--gpus`: two cards or more widen each card's column over MEM's (gateway.css), so its two lines stay under it.
  const rings = gauges.length === 2 ? '' : ` style="--rings: ${gauges.length}; --gpus: ${gpus.length}"`;
  const why = missingWhy(h, 'utilisation for this GPU');
  return `<div class="rings"${rings}>
      ${cpuColumn(h)}${rows.map((r, i) => gpuColumn(r, gpus[i][1], why)).join('')}${ringGauge('MEM', h.ram_percent)}
      <div class="figs">${figs.map(([k, v]) => `<div class="fig"><span>${esc(k)}</span><b>${esc(v)}</b></div>`).join('')}</div>
    </div>
    <div class="tiles tiles${Math.min(5, Math.max(3, figures.length))} flat">
      ${figures.map(([label, v, series]) => graphOf(label, v, series)).join('')}
    </div>`;
}

/** The boxes, in the order their servers are listed, then any that only broadcast hardware. */
function machineBoxes(s) {
  const boxes = [];
  (s.backends || []).forEach((b) => { if (b.kind !== 'cluster' && b.box && !boxes.includes(b.box)) boxes.push(b.box); });
  Object.keys(s.hardware || {}).sort().forEach((box) => { if (!boxes.includes(box)) boxes.push(box); });
  return boxes;
}

/** The ACTIVE clusters a box holds a rank of (`status.clusters[*].topology.boxes`), in the run's order. */
const clustersOfBox = (s, box) => Object.values((s || {}).clusters || {})
  .filter((c) => (topologyOf(c).boxes || []).includes(box)).map((c) => c.name);

function renderMachines(s) {
  const host = $('lb-backends');
  if (!host) return;
  const have = new Map([...host.children].map((el) => [el.dataset.box, el]));
  machineBoxes(s).forEach((box, i) => {
    // THE SLOT: the card and, under it, its flight recorder's strip (recorder.js) — two rows of one subgrid
    // (recorder.css `.machine-slot`), so a closed strip adds nothing to the card's height and an open one grows only
    // the strip's row. The power control still finds `#lb-backends article.card`.
    let slot = have.get(box);
    have.delete(box);
    if (!slot) {
      slot = document.createElement('div');
      slot.className = 'machine-slot';
      slot.dataset.box = box;
      slot.innerHTML = `<article class="card machine" data-box="${esc(box)}"><div class="card-head"><div class="box-id"></div></div>`
        + '<div class="machine-body"></div></article><div class="rec"></div>';
    }
    if (host.children[i] !== slot) host.insertBefore(slot, host.children[i] || null);
    const card = slot.querySelector('article');
    const h = (s.hardware || {})[box] || {};
    card.classList.toggle('stale', !!h.stale);
    // The Cluster section is every box, whichever server is selected; the selected server's boxes are MARKED.
    const servers = clustersOfBox(s, box);
    card.classList.toggle('selected', !!(theCluster(s) || {}).name && servers.includes(theCluster(s).name));
    // The box's name is the FIRST TEXT of .box-name: that is how the power control knows whose card this is.
    paint(card.querySelector('.box-id'), `<div class="box-name">${esc(box)} <span class="box-host">${esc(h.host && h.host !== box ? h.host : '')}</span></div>`
      + servers.map((n) => `<span class="tag box-cluster" title="server ${esc(n)} runs a rank here">${esc(n)}</span>`).join('')
      + (h.stale ? `<span class="stale">no sample for ${Math.round(h.age_s)}s</span>` : ''));
    paint(card.querySelector('.machine-body'), machineBody(s, box));
    if (typeof recorderPaint === 'function') recorderPaint(slot.querySelector('.rec'), box, h.recorder || null);
  });
  have.forEach((el) => el.remove());
}

function renderQueued(s) {
  const q = s.leases.filter((l) => l.status !== 'running');
  const el = $('lb-queued');
  el.hidden = q.length === 0;
  el.innerHTML = q.map((l) => {
    const mins = Math.round((s.now - l.created_at) / 60);
    // A vLLM lease asks the fleet to START a server on a free box (the fallback's box-holding lease is retired).
    const what = '<b>vLLM lease</b>';
    return `<div class="strip">${what} &middot; node_job ${esc((l.job_id || '').slice(0, 8))}
      &middot; ${esc(l.status)} ${mins}m</div>`;
  }).join('');
}

// ── THE REQUESTS CARD ────────────────────────────────────────────────────────────────────────────────
// /lb/status `queue` (request_queue.py): one row per turn the ENGINE is working on — streaming, then prefill, then
// queued, oldest first in each — under the machines; a held turn (nothing sent) and a finished one are counts, the
// finished folding open from theirs. Every figure is the LB's own, off the relayed stream: prompt tokens and their
// cached share from `message_start`, prefill = send → `message_start` and its tok/s, decode = ≈ tok/s DERIVED (the
// engine's GENERATED over its probe window, apportioned by the turn's content events in that window — the rows add up
// to it, summed beside it in the counts line) and Δ content events since (events, NOT tokens), the thinking share, tool_use blocks, the last byte and keepalives. The
// prompt is the operator's traffic: ONE narrow line, the capped whole only in its title, always `esc`aped — a
// `<script>` in it is text. Rows are kept as the SAME elements across polls, so a hover is not torn down.

/** Each state's badge: the page's own badge, in words the LB can stand behind. PREFILL and QUEUED are inferred
 *  (FCFS: the oldest N − W of N no-byte turns), and say so on hover; NO BYTES is that split with no gauge. */
const REQUEST_STATES = {
  streaming: { label: 'STREAMING', tone: 'ok', title: 'bytes reaching the client' },
  prefill: { label: 'PREFILL', tone: 'warn', title: 'no byte back · prefilling' },
  queued: { label: 'QUEUED', tone: 'park', title: 'no byte back · waiting at the engine' },
  sent: { label: 'NO BYTES', tone: 'idle', title: 'no byte back · queued or prefilling: no engine gauge' },
  done: { label: 'DONE', tone: 'ok', title: 'answered whole' },
  killed: { label: 'KILLED', tone: 'bad', title: 'killed by the operator · its client told not to retry' },
  client_gone: { label: 'CLIENT GONE', tone: 'off', title: 'the client hung up' },
  failed: { label: 'FAILED', tone: 'bad', title: 'not answered whole' },
};
/** The prompt's one line, in characters; the capped whole (4,000) is the cell's title. */
const PROMPT_LINE = 96;
const fig = (v) => fmtNum(v).text || '—';
const pct = (share) => (isNum(share) ? `${Math.round(share * 100)}%` : null);
/** Claude Code's entrypoint (captured UA `claude-cli/2.1.288 (external, sdk-cli)`): `sdk-cli` is `claude -p`. */
const clientText = (r) => {
  if (r.client !== 'claude' || !r.client_detail || r.client_detail === 'cli') return r.client || '—';
  return r.client_detail === 'sdk-cli' ? 'claude -p' : `claude ${r.client_detail}`;
};
const lineOf = (p) => (p == null ? '—' : p.length > PROMPT_LINE ? `${p.slice(0, PROMPT_LINE)}…` : p);
const dash = (v, f) => (isNum(v) ? f(v) : '—');

function stateBadge(key, r) {
  const st = REQUEST_STATES[key] || { label: String(key || '?').toUpperCase(), tone: 'idle', title: '' };
  const why = r.inferred ? ` · inferred: oldest ${Math.max(0, r.inferred.no_bytes - r.inferred.waiting)} of `
    + `${r.inferred.no_bytes} no-byte turns, engine waiting ${r.inferred.waiting}` : '';
  return `<span class="badge ${st.tone}" title="${esc(st.title + why)}">${esc(st.label)}</span>`;
}

/** Why a prompt figure is "—": what the stream has (not) said yet — an OpenAI stream reports its prompt only in its
 *  final usage chunk (until then the engine's own /tokenize count, prompt_count.py), an Anthropic one in
 *  `message_start`; nothing back yet is nothing to read. */
function promptWhy(r) {
  if (r.prompt_error) return `the engine's prompt count failed: ${r.prompt_error}`;
  if (!r.bytes && !isNum(r.prefill_s)) return 'no byte back yet';
  return /\/messages$/.test(r.path || '') ? 'message_start carried no prompt usage'
    : 'an OpenAI stream reports its prompt tokens in its final usage chunk';
}

/** The cells a live and a finished row share: who, model, messages / tools, prompt tokens · cached share, TTFT
 *  (send → message_start or the first byte: the engine's queue AND the prefill — it reports no split per request, so
 *  no prompt ÷ TTFT is printed as a prefill rate), then (by the caller) decode, the thinking share and tool_use blocks,
 *  box, the prompt's line. */
function requestCells(r, decode, last, say) {
  const prompt = isNum(r.prompt_tokens)
    ? `${fig(r.prompt_tokens)}${pct(r.cached_share) ? `<i>·</i>${pct(r.cached_share)}` : ''}` : '—';
  const promptTitle = isNum(r.prompt_tokens)
    ? `prompt tokens${r.prompt_from ? ` (${r.prompt_from})` : ''} · read from the prefix cache` : promptWhy(r);
  const ttft = isNum(r.prefill_s) ? (r.prefill_s < 100 ? `${r.prefill_s.toFixed(1)} s` : fmtDuration(r.prefill_s)) : '—';
  const think = r.deltas ? `think ${Math.round((r.thinking || 0) / r.deltas * 100)}%` : '';
  const split = [think, r.tool_uses ? `${fmtInt(r.tool_uses)} tool` : ''].filter(Boolean).join(' · ') || '—';
  const who = [r.client, r.client_detail, r.session && `session …${r.session}`, r.model, r.box,
    `${fmtInt(r.messages)} messages / ${fmtInt(r.tools)} tools`, split !== '—' && split].filter(Boolean);
  return `<span class="rq-who" title="${esc(who.join(' · '))}">`
      + `${esc(clientText(r))}${r.session ? ` <span class="rq-sess">·${esc(r.session)}</span>` : ''}</span>`
    + `<span class="rq-model" title="${esc(r.model || '')}">${esc(r.model ? shortModel(r.model) : '—')}</span>`
    + `<span class="rq-num rq-msgs" title="messages / tools">${fmtInt(r.messages)}<i>/</i>${fmtInt(r.tools)}</span>`
    + `<span class="rq-num rq-prompt" title="${esc(promptTitle)}">${prompt}</span>`
    + `<span class="rq-num rq-ttft" title="time to first byte: send → message_start or the first chunk — the engine's queue and the prefill together; it reports no split per request">${ttft}</span>`
    + decode
    + `<span class="rq-num rq-split" title="thinking share of the content events · tool_use blocks started">${split}</span>`
    + last
    + `<span class="rq-box">${esc(r.box || '—')}</span>`
    + (say ? `<span class="rq-text rq-kill-why" title="${esc(say)}">${esc(say)}</span>`
      : `<span class="rq-text" title="${esc(r.prompt == null ? '' : r.prompt)}">${esc(lineOf(r.prompt))}</span>`);
}

/** A server's WAIT verdict in a cell's words (request_queue.admission): "kv full 99%", "seats 16/16", "deferred 4";
 *  null when its figures prove none. */
function waitText(eng) {
  if (!eng || !eng.wait) return null;
  if (eng.wait === 'kv') return `kv full${isNum(eng.kv_pct) ? ` ${Math.round(eng.kv_pct)}%` : ''}`;
  if (eng.wait === 'seats') return `seats ${fmtInt(eng.running)}/${fmtInt(eng.seats)}`;
  if (eng.wait === 'deferred') return `deferred ${fmtInt(eng.waiting_deferred)}`;
  return null;
}

/** The WAIT cell: on a queued row why its server's queue waits — the verdict, its rule and figures on hover; "—" with
 *  the figures where they prove none (the engine publishes no reason per request). A row the engine runs waits on
 *  nothing: an empty cell. */
function waitCell(r, eng) {
  const span = (text, title) => `<span class="rq-num rq-wait" title="${esc(title)}">${esc(text)}</span>`;
  if (r.state === 'streaming' || r.state === 'prefill') return span('', 'admitted: the engine is running it');
  if (r.state === 'sent') return span('—', 'no engine gauge: queued or prefilling is not known');
  if (!eng) return span('—', "this server's admission figures were not read");
  return span(waitText(eng) || '—', eng.why || '');
}

/** A live row's DECODE (request_queue.py DERIVED_DECODE, operator 2026-10-16 "requests tok/s doesn't add up to
 *  official total"): `tok_s` — the engine's GENERATED over its probe window, apportioned by this turn's content events
 *  in that same window — marked ≈ because it is DERIVED, never measured (no stream carries a per-chunk token count);
 *  its method and the window's events on hover. Beside it Δ = content events since message_start, called `ev`: events,
 *  not tokens. No figure for the total is no number here: the reason (`tok_s_why`) in its place. */
let decodeHow = null;   // the LB's own words for the method (`queue.decode.derived`), set by `renderRequests`
function decodeCell(r) {
  const ev = r.deltas ? `Δ${fmtInt(r.deltas)} ev` : '';
  if (!isNum(r.tok_s)) {
    const why = r.tok_s_why || 'not reported';
    return `<span class="rq-num rq-decode" title="${esc(`no tok/s: ${why} · content events since message_start`)}">`
      + `${esc(shortWhy(why))}${ev ? `<i>·</i>${ev}` : ''}</span>`;
  }
  const title = [decodeHow || 'derived', `${fmtInt(r.window_events)} events in the window`,
    `${fmtInt(r.deltas)} since message_start`].filter(Boolean).join(' · ');
  return `<span class="rq-num rq-decode" title="${esc(title)}">≈${fig(r.tok_s)} tok/s${ev ? `<i>·</i>${ev}` : ''}</span>`;
}

/** A reason in a cell: its first clause, the whole on hover. */
const shortWhy = (w) => String(w).split(/[:;(]/)[0].trim();

/** One live row: state and time in it, why it waits (`eng`: its server's `queue.backends` entry), then the shared
 *  cells; decode = content events (Δ, not tokens) and their rate since `message_start`; last = the last byte's age
 *  and the keepalives sent through the backend's silence. */
function requestRow(r, eng, kill) {
  const p = requestParts(r, eng, kill);
  return `<span class="rq-state" title="${esc(p.title)}">${p.state}</span>` + p.cells;
}

/** A live row in its two parts: what goes in its STATE cell after the kill button — the badge (or the kill's own,
 *  `kill`: this page's kill of it under way or answered) and the time in that state — and the rest of its cells. The
 *  live table keeps the state cell and its button as the same elements across polls and paints these into them. */
function requestParts(r, eng, kill) {
  const key = r.state || r.outcome;
  const decode = decodeCell(r);
  const last = `<span class="rq-num" title="since the last byte · keepalives">${dash(r.last_byte_s, fmtDuration)}`
    + `${r.keepalives ? `<i>·</i>${fmtInt(r.keepalives)} ka` : ''}</span>`;
  return { title: `age ${fmtDuration(r.age_s) || '—'}`,
    state: (kill ? killBadge(kill) : stateBadge(key, r)) + `<span class="rq-phase">${dash(r.phase_s, fmtDuration)}</span>`,
    cells: waitCell(r, eng) + requestCells(r, decode, last, kill && kill.phase === 'refused' ? kill.text : null) };
}

// ── KILL ONE REQUEST ─────────────────────────────────────────────────────────────────────────────────────────────
// Operator, 2026-10-12: "implement KILL red x button, hold it for 1s to work". A small red × at the start of every live
// row, inside its pinned state cell — no column of its own, so the table is no wider. It is the page's ONE hold
// (`holdButton`: the same fill, letting go early cancels, a click does nothing) at 1 s, and a completed hold posts
// POST /lb/requests/<id>/kill once (request_kill.py). The row says "killing…", then the LB's answer: "killed" — the LB
// answers only once the turn has ENDED as killed — or the refusal in its words, in the row's text cell. A killed row
// stays drawn KILL_LINGER_MS after the answer (the next poll no longer lists it), then the finished list has it.

const KILL_HOLD_MS = 1000;
const KILL_LINGER_MS = 6000;
/** This page's kills, by row id: `{ phase: 'killing' | 'killed' | 'refused', text, until }`. */
const kills = new Map();

/** The kill's badge in place of the state's: under way, done, or refused (its words on hover and in the text cell). */
function killBadge(k) {
  const label = k.phase === 'killing' ? 'killing…' : (k.phase === 'killed' ? 'killed' : 'not killed');
  return `<span class="badge ${k.phase === 'killing' ? 'warn' : 'bad'} rq-kill-state" title="${esc(k.text || '')}">`
    + `${esc(label)}</span>`;
}

/** A live row's persistent parts: the state cell holding the × (a hold under way must survive a poll's repaint) and
 *  the painted badge beside it, then the rest of the cells (`display: contents`, so they stay the table's cells). */
function newRequestRow(id) {
  const row = document.createElement('div');
  row.dataset.id = id;
  const cell = document.createElement('span');
  cell.className = 'rq-state';
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'hold kill rq-kill';
  btn.setAttribute('aria-label', `Kill request ${id} (hold 1 s)`);
  btn.title = 'Hold 1 s: kill this request — its client is told, the engine stops working on it';
  btn.innerHTML = '<span class="hold-fill" aria-hidden="true"></span><span class="rq-x" aria-hidden="true">×</span>';
  holdButton(btn, KILL_HOLD_MS, () => killRequest(id));
  const prio = document.createElement('button');
  prio.type = 'button';
  prio.className = 'rq-prio';
  prio.innerHTML = '<span aria-hidden="true">▲</span>';
  prio.addEventListener('click', () => { if (!prio.disabled) togglePriority(id, !priorityOn(row.rowData)); });
  const st = document.createElement('span');
  st.className = 'rq-st';
  const cells = document.createElement('span');
  cells.className = 'rq-cells';
  cell.appendChild(btn);
  cell.appendChild(prio);
  cell.appendChild(st);
  row.appendChild(cell);
  row.appendChild(cells);
  Object.assign(row, { killBtn: btn, prioBtn: prio, stateCell: cell, stateBody: st, cellsBody: cells });
  return row;
}

/** Paint a live row from its last data and this page's kill of it, if any. */
function paintRequest(row) {
  const kill = kills.get(row.dataset.id);
  // A just-ended row (`lingerOf`): its finished cells, nothing left to kill or prioritise.
  if (row.ended) {
    const p = lingerParts(row.rowData);
    row.stateCell.setAttribute('title', p.title);
    paint(row.stateBody, p.state);
    paint(row.cellsBody, p.cells);
    row.killBtn.disabled = true;
    row.prioBtn.disabled = true;
    row.prioBtn.className = 'rq-prio';
    return;
  }
  const p = requestParts(row.rowData, row.engData, kill);
  row.stateCell.setAttribute('title', p.title);
  paint(row.stateBody, p.state);
  paint(row.cellsBody, p.cells);
  row.killBtn.disabled = !!kill && kill.phase !== 'refused';
  const d = row.rowData || {};
  const on = priorityOn(d);
  const sq = row.prioBtn;
  sq.className = `rq-prio${on ? ' on' : ''}`;
  sq.setAttribute('aria-pressed', on ? 'true' : 'false');
  sq.setAttribute('aria-label', `Priority for request ${row.dataset.id}: ${on ? 'on' : 'off'}`);
  sq.disabled = !!d.priority_locked || prioPending.has(row.dataset.id) || (!!kill && kill.phase !== 'refused');
  sq.title = d.priority_locked ? `Priority ${on ? `on (${d.priority})` : 'off'} — locked: ${d.priority_locked}`
    : (on ? `Priority on (${d.priority}) — click: back to normal` : 'Click: run sooner (re-queued at priority −1)');
}

// ── PRIORITY ONE REQUEST ─────────────────────────────────────────────────────────────────────────────────────────
// Operator, 2026-10-28: "small square priority column, similar to kill task, green when active, user in GUI can
// activate too or deactivate". A small square after the ×, in the same pinned state cell; green while the row's priority
// is sooner than none (< 0: the toggle's −1, or a caller's X-Priority). A click posts POST /lb/requests/<id>/priority
// {active} once; the LB re-sends a turn the client has had nothing of at its new priority. Locked where the LB says why
// (`priority_locked`: streaming, or a path whose engine drops the field).

/** Is the row's priority on — sooner than a request naming none? */
const priorityOn = (r) => isNum((r || {}).priority) && r.priority < 0;
/** Rows with a toggle out: one POST at a time per row. */
const prioPending = new Set();

async function togglePriority(id, active) {
  if (prioPending.has(id)) return;
  prioPending.add(id);
  const row = [...(($('requests-live') || {}).children || [])].find((el) => el.dataset && el.dataset.id === id);
  if (row && row.rowData) paintRequest(row);
  let said;
  try {
    const r = await fetch(`/lb/requests/${encodeURIComponent(id)}/priority`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ active }) });
    said = r.status === 200 ? `priority ${active ? 'on' : 'off'} (${(((await r.json()) || {}).priority || {}).applied || 'set'})`
      : await refusal(r);
  } catch (e) {
    said = e.message;
  }
  prioPending.delete(id);
  if (row && row.rowData) paintRequest(row);
  const say = $('requests-kill-say');
  if (say) say.textContent = `request ${id}: ${said}`;
  poll({ fresh: true });
}

function setKill(id, k) {
  kills.set(id, k);
  const row = [...(($('requests-live') || {}).children || [])].find((el) => el.dataset && el.dataset.id === id);
  if (row && row.rowData) paintRequest(row);
  const say = $('requests-kill-say');
  if (say) say.textContent = `request ${id}: ${k.phase === 'killing' ? 'killing…' : k.text}`;
}

/** A completed hold: ONE post per row while one is out or has succeeded; the LB's answer in the row. */
async function killRequest(id) {
  const k = kills.get(id);
  if (k && k.phase !== 'refused') return;
  setKill(id, { phase: 'killing', text: 'killing…' });
  let next;
  try {
    const r = await fetch(`/lb/requests/${encodeURIComponent(id)}/kill`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
    if (r.status === 200) next = { phase: 'killed', text: (((await r.json()) || {}).killed || {}).message || 'killed' };
    else if (r.status === 202) next = { phase: 'killing', text: (((await r.json()) || {}).killed || {}).message || 'killing…' };
    else next = { phase: 'refused', text: await refusal(r) };
  } catch (e) {
    next = { phase: 'refused', text: e.message };
  }
  setKill(id, { ...next, until: Date.now() + KILL_LINGER_MS });
  poll({ fresh: true });
}

/** One finished row: outcome and duration, the shared cells, the engine's output count and the decode tok/s the
 *  tap timed, and the stop reason — or, for a failure, its why. */
function recentRow(r) {
  const f = finishedCells(r);
  return `<div class="rq-row rq-done ${esc(r.outcome || '')}"><span class="rq-state">${stateBadge(r.outcome, r)}`
    + `<span class="rq-phase">${dash(r.duration_s, fmtDuration)}</span></span>` + requestCells(r, f.decode, f.last) + '</div>';
}

/** A finished turn's own cells — its output count and decode tok/s, its stop reason or why — for the finished list
 *  and for a just-ended row still in the live list (`lingerParts`). */
function finishedCells(r) {
  return { decode: `<span class="rq-num" title="output tokens (the engine's count) · decode tok/s">`
    + `${dash(r.output_tokens, fmtInt)}${isNum(r.decode_tok_s) ? `<i>·</i>${r.decode_tok_s.toFixed(1)} tok/s` : ''}</span>`,
  last: r.why ? `<span class="rq-why" title="${esc(r.status != null ? `HTTP ${r.status}` : '')}">${esc(r.why)}</span>`
    : `<span class="rq-stop" title="stop reason · HTTP ${esc(r.status)}">${esc(r.stop_reason || '—')}</span>` };
}

/** The finished table's head: its decode column is the engine's output count, its last the stop reason. */
const RECENT_HEAD = '<div class="rq-row rq-head" aria-hidden="true"><span class="rq-state">state</span><span>client</span><span class="rq-model">model</span><span class="rq-msgs" title="messages / tools">msg/tool</span><span title="prompt tokens · cached share">prompt</span><span title="time to first byte: queue + prefill">ttft</span><span title="output tokens (the engine count) · decode tok/s">output</span><span class="rq-split" title="thinking share · tool_use blocks">split</span><span title="stop reason, or why it failed">stop</span><span class="rq-box">box</span><span>text</span></div>';

/** The GATEWAY's count: its relays in flight to the server(s) this card shows — every row state, listed or past the
 *  cap; held turns (sent to no server) are not among them. One definition, for the head and the note. */
const gatewayCount = (q) => (isNum(q.inflight_total) ? q.inflight_total : (q.inflight || []).length);

/** The head's figures: the GATEWAY's in-flight and the ENGINE's running / waiting SIDE BY SIDE (operator
 *  2026-10-30: "why shows only one request if queue tells 2?" — the QUEUE tile is the engine's read, the rows are
 *  this gateway's, and printing them apart hid that they are two counts), then the gateway's rows per state, the
 *  held as a count, the decode sum. No engine read: a dash, never a 0. */
function requestCounts(q) {
  const c = q.counts || {};
  const shown = (q.inflight || []).length;
  const total = gatewayCount(q);
  const parts = [`<span class="rq-fig rq-gateway" title="this gateway's relays in flight now (the rows below)">gateway <b>${fmtInt(total)}</b></span>`];
  const e = q.engine;
  if (e) {
    const why = isNum(e.waiting_capacity) && isNum(e.waiting_deferred)
      ? ` <span title="vllm:num_requests_waiting_by_reason — capacity: a seat, KV blocks or the step's token budget; deferred: a KV transfer, a LoRA budget, a blocked status">`
        + `<b>${fmtInt(e.waiting_capacity)}</b> capacity <b>${fmtInt(e.waiting_deferred)}</b> deferred</span>` : '';
    parts.push(`<span class="rq-fig rq-engine" title="the engine's own /metrics${e.backends === 1 ? '' : ', every live backend'} — the QUEUE tile's figure">engine `
      + `<b>${fmtInt(e.running)}</b> running <b>${fmtInt(e.waiting)}</b> waiting${why}</span>`);
  } else {
    parts.push('<span class="rq-fig rq-engine" title="no /metrics read: running / waiting unknown">engine <b>—</b></span>');
  }
  ['streaming', 'prefill', 'queued', 'sent'].filter((k) => c[k])
    .forEach((k) => parts.push(`<span class="rq-fig ${REQUEST_STATES[k].tone}"><b>${fmtInt(c[k])}</b> ${REQUEST_STATES[k].label.toLowerCase()}</span>`));
  if (total > shown) parts.push(`<span class="rq-fig"><b>+${fmtInt(total - shown)}</b> unlisted</span>`);
  if (c.held) parts.push(`<span class="rq-fig warn" title="admitted · nothing sent to any backend"><b>${fmtInt(c.held)}</b> held</span>`);
  const d = decodeSum(q.decode);
  if (d) parts.push(d);
  return parts.join('<span class="rq-sep">·</span>');
}

/** WHY THE TWO COUNTS DIFFER, one muted line under the head, or '' when they agree or the engine was not read (no
 *  difference is claimed against a figure nobody read). The engine's is a /metrics read seconds old, the gateway's is
 *  now: with ~1 s turns they part by one or two. More on the engine: a turn that just ended (the read predates the
 *  end) or one sent to the engine past this gateway. More here: relayed after the read, not yet admitted. */
function engineGap(q) {
  const e = q.engine;
  if (!e || !isNum(e.running) || !isNum(e.waiting)) return '';
  const diff = e.running + e.waiting - gatewayCount(q);
  if (!diff) return '';
  const why = diff > 0 ? 'engine only — just finished, not yet relayed, or sent past this gateway'
    : 'gateway only — not yet admitted by the engine';
  const age = isNum(e.read_s) ? ` &middot; engine read ${fmtInt(Math.round(e.read_s))} s ago` : '';
  return `<b>+${fmtInt(Math.abs(diff))}</b> ${why}${age}`;
}

/** THE ROWS' DECODE ADDED UP, beside the engine's own GENERATED over the same window (request_queue.py `decode`):
 *  Σ rows + ended (finished or finishing inside the window) + unlisted (past the row cap) = engine — so the operator
 *  sees the column add up. The method on hover; servers whose figure is missing counted, never folded in. No figure:
 *  the reason in its place, no number. */
function decodeSum(d) {
  if (!d) return '';
  const title = esc([d.derived, isNum(d.window_s) ? `window ${d.window_s} s` : null].filter(Boolean).join(' · '));
  if (!isNum(d.engine_tok_s)) {
    return `<span class="rq-fig rq-decode-sum" title="${title}">decode Σ — <i>${esc(shortWhy(d.why || 'not reported'))}</i></span>`;
  }
  const extra = [[d.ended_tok_s, 'ended'], [d.unlisted_tok_s, 'unlisted']].filter(([v]) => isNum(v) && v > 0)
    .map(([v, w]) => ` <b>+${fig(v)}</b> ${w}`).join('');
  const unm = d.unmeasured ? ` <b>${fmtInt(d.unmeasured)}</b> unmeasured` : '';
  return `<span class="rq-fig rq-decode-sum" title="${title}">decode <b>Σ ${fig(d.rows_tok_s)}</b> rows${extra}`
    + ` <b>= ${fig(d.engine_tok_s)}</b> tok/s engine${unm}</span>`;
}

function recentSummary(recent) {
  const by = {};
  recent.forEach((r) => { by[r.outcome] = (by[r.outcome] || 0) + 1; });
  return `recent <b>${fmtInt(recent.length)}</b>` + ['done', 'failed', 'killed', 'client_gone'].filter((k) => by[k])
    .map((k) => `<span class="rq-sep">·</span><span class="rq-fig ${REQUEST_STATES[k].tone}"><b>${fmtInt(by[k])}</b> ${REQUEST_STATES[k].label.toLowerCase()}</span>`)
    .join('');
}

/** THE QUEUE AS ONE SERVER'S (`url`, its head): the rows that went to it (`backend` on live and finished rows), its own
 *  per-state counts and engine gauges (`queue.backends[url]`), its decode share (`queue.decode.backends[url]`) — the
 *  same shape `requestCounts` and the rows read, so nothing is drawn twice. HELD turns were sent to no server yet:
 *  they stay in the line, their title says so. No url (no server in the run): the whole queue, as before. */
function scopeQueue(q, url) {
  if (!q || !url) return q;
  const per = (q.backends || {})[url] || null;
  const rows = per && per.rows ? per.rows : null;
  const inflight = (q.inflight || []).filter((r) => r.backend === url);
  const d = ((q.decode || {}).backends || {})[url] || null;
  return Object.assign({}, q, {
    inflight, recent: (q.recent || []).filter((r) => r.backend === url),
    inflight_total: rows ? Object.values(rows).reduce((a, b) => a + (isNum(b) ? b : 0), 0) : inflight.length,
    counts: Object.assign({}, rows || {}, { held: (q.counts || {}).held }),
    engine: per && isNum(per.running) ? { running: per.running, waiting: per.waiting, backends: 1,
      waiting_capacity: per.waiting_capacity, waiting_deferred: per.waiting_deferred } : null,
    decode: d ? Object.assign({}, d, { unmeasured: d.why ? 1 : 0, derived: (q.decode || {}).derived }) : null,
  });
}

/** THE JUST-ENDED, kept in the live list (operator 2026-10-30): a ~1 s turn that started and ended between two polls
 *  was never a row, so the gateway's list read one short of the engine's. A finished turn (`recent`, its `ended_at`
 *  on the server's clock) ended at most LINGER_S before the status's `now` stays a muted live row — the SAME element
 *  it was while live, so nothing jumps — then only the finished list has it. No server clock: none kept. */
const LINGER_S = 10;
function lingerOf(recent, now, live) {
  if (!isNum(now)) return [];
  return (recent || []).filter((r) => isNum(r.ended_at) && now - r.ended_at >= 0 && now - r.ended_at <= LINGER_S
    && !live.has(r.id));
}

/** A just-ended row's two parts, in the live table's columns: "done · 1.2 s" in a muted badge (the badge's own box,
 *  so the row keeps the live rows' height), an empty WAIT cell, then the finished row's own cells. */
function lingerParts(r) {
  const st = REQUEST_STATES[r.outcome];
  const took = isNum(r.duration_s) ? (r.duration_s < 100 ? `${r.duration_s.toFixed(1)} s` : fmtDuration(r.duration_s)) : '—';
  const f = finishedCells(r);
  return { title: `ended · took ${took} · ${st ? st.title : (r.outcome || '')}`,
    state: `<span class="badge idle rq-linger-badge">${esc(st ? st.label.toLowerCase() : (r.outcome || 'ended'))} · ${took}</span>`,
    cells: '<span class="rq-num rq-wait"></span>' + requestCells(r, f.decode, f.last) };
}

function renderRequests(s) {
  const card = $('lb-requests');
  if (!card) return;
  const head = headOf(s, theCluster(s));
  let q = scopeQueue((s || {}).queue, head ? head.url : null);
  card.hidden = !q;
  if (!q) return;
  // THE ENGINE'S FIGURES ARE THE QUEUE TILE'S (`engineCount`): the head's server, or every server when the run has
  // none; the engine's reasons to wait stay the queue's own.
  const eng = engineCount(head ? [head] : (s.backends || []), s.now);
  q = Object.assign({}, q, { engine: eng && Object.assign({}, eng, {
    waiting_capacity: (q.engine || {}).waiting_capacity, waiting_deferred: (q.engine || {}).waiting_deferred }) });
  decodeHow = (q.decode || {}).derived || null;
  paint($('requests-counts'), requestCounts(q));
  const gap = $('requests-gap');
  if (gap) paint(gap, engineGap(q));
  const host = $('requests-live');
  const kids = [...host.children];
  const fixed = kids.filter((el) => !el.dataset.id).length;          // the table's head row, first
  const have = new Map(kids.filter((el) => el.dataset.id).map((el) => [el.dataset.id, el]));
  const now = Date.now();
  (q.inflight || []).forEach((r, i) => {
    let row = have.get(r.id);
    have.delete(r.id);
    if (!row) row = newRequestRow(r.id);
    // A refusal is said for KILL_LINGER_MS; a turn still live after it shows its own state again.
    const k = kills.get(r.id);
    if (k && k.phase === 'refused' && k.until <= now) kills.delete(r.id);
    row.className = `rq-row ${r.state}`;
    if (host.children[fixed + i] !== row) host.insertBefore(row, host.children[fixed + i] || null);
    row.rowData = r;
    row.ended = false;
    row.engData = (q.backends || {})[r.backend];
    paintRequest(row);
  });
  // The just-ended, after the live: the same element a live row was, muted (`lingerOf`).
  const liveCount = (q.inflight || []).length;
  const ended = lingerOf(q.recent, s.now, new Set((q.inflight || []).map((r) => r.id)));
  ended.forEach((r, j) => {
    let row = have.get(r.id);
    have.delete(r.id);
    if (!row) row = newRequestRow(r.id);
    row.className = `rq-row gone rq-linger ${r.outcome || ''}`;
    const at = fixed + liveCount + j;
    if (host.children[at] !== row) host.insertBefore(row, host.children[at] || null);
    row.rowData = r;
    row.ended = true;
    paintRequest(row);
  });
  // A row this page killed (or tried to) STAYS WHERE IT WAS until its answer is in and KILL_LINGER_MS on: the poll
  // stops listing a killed turn at once, and the operator is owed the word "killed" where they held.
  const lingering = [];
  have.forEach((el, id) => {
    const k = kills.get(id);
    if (k && (k.phase === 'killing' || k.until > now)) {
      el.className = 'rq-row gone';
      lingering.push(el);
    } else {
      kills.delete(id);
      el.remove();
    }
  });
  const was = new Map(kids.map((el, i) => [el, i]));
  lingering.sort((a, b) => was.get(a) - was.get(b))
    .forEach((el) => host.insertBefore(el, host.children[was.get(el)] || null));
  const shownLive = new Set(ended.map((r) => r.id));
  const recent = (q.recent || []).filter((r) => !shownLive.has(r.id));
  const fold = $('requests-recent');
  if (fold) fold.hidden = !recent.length;
  paint($('requests-recent-sum'), recentSummary(recent));
  paint($('requests-recent-list'), recent.length ? RECENT_HEAD + recent.map(recentRow).join('') : '');
}

/** The cluster card's ONE state. Held (both parks ours, both ranks of this generation up, the
 *  head answering) it reads exactly like a leased vLLM — SERVING, or DEMOTED past its own
 *  depth; short of that it says why it carries no traffic. */
function clusterStatusOf(s, b) {
  const c = clusterViewOf(s, b);
  if (c.held && b.live) return statusOf(s, b);
  // A node this LB only observes is held whenever it answers; when it does not, that is the whole of what is known —
  // amber like the card's `unknown`, never "not held" (nothing here holds it) and never a red nobody measured.
  if (observedOnly(c) || (localNode(c) && c.state === 'unknown')) return { tone: 'warn', label: 'NOT ANSWERING' };
  // A node the LB requests drains before its host relaunches or stops it: no new turn, its own still answered.
  if (localNode(c) && c.state === 'finishing') return { tone: 'warn', label: 'FINISHING' };
  if (c.state === 'starting' || c.state === 'restarting') return { tone: 'warn', label: 'STARTING' };
  if (c.state === 'failed') return { tone: 'bad', label: 'FAILED' };
  // A hung start halted after its relaunches: the operator's, like FAILED, but no error happened — amber.
  if (c.state === 'stalled') return { tone: 'warn', label: 'STALLED' };
  return { tone: 'warn', label: 'NOT HELD' };
}

/** A box's PARK, in the operator's own words, or null. The wire field is `park` — `cluster_loop.py:661`
 *  publishes `"park": o.parked_by` — and it is a STRING: the `requested_by` of an OFF `worker_controls`
 *  row (`cluster.py:644`). Reading `parked_by` here, as this page did, was always `undefined`, and
 *  putting it through `reservedForCluster` asked a string for a reservation's `.status`. */
const parkOf = (o) => {
  const park = (o || {}).park;
  return typeof park === 'string' && park ? park : null;
};

/** Under the pair's card: who holds it, when it is held and named. A park is the one hold with words of
 *  its own, and it is always the operator's. STARTING, FAILED and NOT HELD are the badge's own words;
 *  nothing restates them under it. */
function clusterFoot(st, c) {
  if (st.label === 'STARTING' || st.label === 'FAILED' || st.label === 'NOT HELD') return '';
  const tag = Object.values(c.boxes || {}).map(parkOf).find(Boolean);
  return tag ? `held by <span class="job">${esc(tag)}</span>` : '';
}

// ── servers: throughput per live backend ─────────────────────────────────────
// The app's `BackendSection`: generated, prompt and queue side by side, each figure over its history;
// then ONE wrapping row of label·value pairs — the engine's figures and what qualified each tile
// (`engineStats`); memory bandwidth and compute under them, for the server that has ceilings (the
// pair). The page adds what only it knows — the server's ONE state as a diode and a badge, what the
// server itself says it holds (`modelLine`), who holds it (the footer).

/** One ceiling as a tile — the app's `CeilingTile`: the figure, a bar for its share of the peak, the
 *  share in words, and the rate's history on the same axis (0 … peak). Without a peak there is no share:
 *  no bar is drawn (an empty one would read as "measured, and none") and the LB's own reason stands in
 *  the note. Memory bandwidth is a LOAD — 120 % of the bus is red; compute is a RATE, read against its
 *  predefined peak: grey when little of it is used, blue near 40 %, glowing from 80 %. */
function ceilingTile(label, c, unit, digits, series, kind, model) {
  const o = c || {};
  const has = isNum(o.rate);
  const pct = isNum(o.pct) ? o.pct : null;
  const colour = pct == null ? null : (kind === 'rate' ? rateColor(pct / 100) : scaleColor(pct / 100));
  const glow = kind === 'rate' && pct != null && pct / 100 >= RATE_GLOW_FROM && styleGlows();
  const share = !has ? `<span class="sat-why">${whyText(o.why)}</span>`
    : (pct != null ? `${pct.toFixed(0)} % of ${fmtNum(o.peak).text} ${esc(unit)}${o.peak_precision ? ` (${esc(o.peak_precision)})` : ''}`
      : (o.why ? `<span class="sat-why">${whyText(o.why)}</span>` : 'no peak configured'));
  // A box's share of the engine's one rate (`perBox`) is DERIVED, and the note and the hover say so.
  const note = o.derived ? `${share} &middot; derived` : share;
  const title = [model ? model.title : null, o.derived || null].filter(Boolean).join(' — ');
  const bar = colour
    ? `<span class="sat-bar"><i style="width:${Math.max(0, Math.min(100, pct)).toFixed(0)}%;background:${colour}"></i></span>`
    : '<span class="sat-bar none"></span>';
  // A MODELED rate (`bandwidthModel`) says so beside its label, its parts in the hover, "≥" on a floor.
  // ONE FIXED HEIGHT (operator 2026-10-30): the label and the basis are ONE line each, cut with an ellipsis — the
  // basis sentence wrapped to two lines on one tile and one on its neighbour — the whole of each in its `title`.
  const shownLabel = model ? `${label} · modeled` : label;
  return `<div class="ceil${has ? '' : ' dim'}"${title ? ` title="${esc(title)}"` : ''}>
    <div class="t-head" title="${esc(shownLabel)}"><span class="t-label">${esc(shownLabel)}</span>
    <b class="t-val"${colour ? ` style="color:${colour}${glow ? `;text-shadow:0 0 8px ${colour}` : ''}"` : ''}>${has ? `${model && model.floor ? '≥' : ''}${o.rate.toFixed(digits)}` : '—'}${has ? `<span class="t-unit"> ${esc(unit)}</span>` : ''}</b></div>
    ${bar}
    <span class="c-note"${cutTitle(note)}>${note}</span>
    <div class="c-trend">${sparkline(series, Object.assign({ max: o.peak > 0 ? o.peak : 0, key: `${label}|ceil` }, kind === 'rate' ? { kind: 'rate', target: o.peak } : {}))}</div>
  </div>`;
}

/** Memory bandwidth and compute side by side, the row under the throughput tiles. Per GPU, from the
 *  engine's own estimate — SGLang's bandwidth and compute from the LB's models (`bandwidthModel`,
 *  `computeModel`); the evidence behind them — and the verdict — is the Saturation section.
 *
 *  NETWORK measures the link BETWEEN RANKS, so it is drawn only where there is one (`link`: `rankLink`, the
 *  topology's TP). On one rank it is not applicable — not unknown — and nothing stands in its place (operator,
 *  2026-10-12: "we don't need to show if not used"); a link not read YET keeps its tile and its reason. The row's class
 *  carries its tile count (`ceils2`, `ceils3` …): a shorter row lays itself out with no empty column (gateway.css). */
function ceilingTiles(sat, link) {
  const hist = sat.history || {};
  // PER BOX USED: one memory bandwidth and one compute tile per serving box, each against its own card's peak.
  const boxes = (label, c, unit, digits, series, kind, model) => perBox(c).map((b) =>
    ceilingTile(`${label} · ${b.box}`, b, unit, digits, boxSeries(series, b), kind, model));
  const tiles = [...boxes('memory bandwidth', sat.bandwidth, 'GB/s', 0, hist.gbps, undefined, bandwidthModel(sat)),
    ...boxes('compute', sat.compute, 'TFLOP/s', 1, hist.tflops, 'rate', computeModel(sat)),
    link ? ceilingTile('network', sat.network, 'GB/s', 2, hist.net_gbps) : '', coresTile(sat.cores)].filter(Boolean);
  return `<div class="ceils ceils${tiles.length}">${tiles.join('')}</div>`;
}

/** THE GRAPHS THIS SETUP DOES NOT COLLECT, as a quiet row of tags under the card's graphs (operator, 2026-10-12: "add
 *  below small tags of graphs available but not collecting data under current setup cluster, just for information").
 *  Each is a tile the card leaves out because its setup has nothing to feed it — the tile stays out (303cd2b,
 *  18e23aa); only its name and why are said, in the tile row's order. NOT-APPLICABLE only, never unknown: the RAM
 *  tier's tile is named where the LB SAYS there is no tier (`kv_offload: null`; an older LB sends no key), and the
 *  network where the topology has one rank (`rankLink`). Nothing left out: no row. */
function unusedGraphs(b, sat, c) {
  const t = b.throughput || {};
  const tags = [];
  if ('kv_offload' in t && !kvOffloadOf(b)) {
    tags.push(['kv · ram · no tier', 'no RAM tier: the engine runs no KV tier in the host\'s RAM (--kv-offloading-size not set)']);
  }
  if (sat && !rankLink(c)) {
    tags.push([`network · TP=${topologyOf(c).tp}`, 'measures the link between ranks; one rank has none']);
  }
  if (!tags.length) return '';
  return `<div class="unused"><span class="unused-k">not collected</span>`
    + `${tags.map(([text, why]) => `<span class="tag unused" title="${esc(why)}">${esc(text)}</span>`).join('')}</div>`;
}

/** CUDA CORES: one GPU's cores as a field of 128 × 48, filled in reading order by what they did over
 *  the last minute of steps — multiplying, waiting on memory, the rest (`cores.py`: an ESTIMATE from
 *  the engine's step counters; the GPU has no counter for it). The areas are the shares.
 *
 *  Three paths however many cores: the whole field in grey, then the first `multiplying + memory`
 *  cores in cyan, then the first `multiplying` in green, each a staircase of at most six corners
 *  (`coreStair`) — the full rows and the partial last row exact in core units. The cell texture
 *  over it is CSS (`.core-grid::after`, a repeating gradient in the panel's colour), not markup. */
const CORE_COLS = 32, CORE_ROWS = 8;
/** The first `n` CELLS of the field, filled row by row, as one staircase path. */
function coreStair(n) {
  if (!(n > 0)) return '';
  if (n >= CORE_COLS * CORE_ROWS) return `M0 0H${CORE_COLS}V${CORE_ROWS}H0Z`;
  const full = Math.floor(n / CORE_COLS), part = n - full * CORE_COLS;
  if (!part) return `M0 0H${CORE_COLS}V${full}H0Z`;
  return full ? `M0 0H${CORE_COLS}V${full}H${part}V${full + 1}H0Z` : `M0 0H${part}V1H0Z`;
}
/** The field is 32 × 8 squares, each `total / 256` cores (24 on a GB10) — a square a person can see:
 *  drawn one core a square, 45 multiplying cores were a third of a 0.7 px row and nothing showed.
 *  Multiplying rounds UP to a whole square (any at all is one square); memory fills to the nearest. */
function coreGrid(c, total, title) {
  const cells = CORE_COLS * CORE_ROWS, per = total / cells;
  const mul = c.multiplying > 0 ? Math.max(1, Math.ceil(c.multiplying / per)) : 0;
  const mem = Math.min(cells, Math.max(mul, Math.round((c.multiplying + c.memory) / per)));
  return `<span class="core-grid"><svg viewBox="0 0 ${CORE_COLS} ${CORE_ROWS}" preserveAspectRatio="none" role="img" aria-label="${esc(title)}">
      <path class="core-other" d="${coreStair(cells)}"/><path class="core-mem" d="${coreStair(mem)}"/><path class="core-mul" d="${coreStair(mul)}"/></svg></span>`;
}
/** One kind of step — decode, or a step carrying a prefill chunk — one of the tile's two sub-columns:
 *  the name and its share of the wall clock, the cores multiplying and the step time, the field, then
 *  the cores waiting on memory and the rest. A kind the engine has no figure for is a dash. */
function coreCase(name, k, total) {
  if (!k) return `<div class="core-case dim"><span class="core-name">${name}</span><b>—</b></div>`;
  const title = `${name} step, ${Math.round(100 * k.time_share)} % of the time, ${fmtInt(k.step_ms)} ms a step: ${k.multiplying} multiplying, ${k.memory} waiting on memory, ${k.other} other of ${total}`;
  return `<div class="core-case" title="${esc(title)}"><span class="core-name"><span>${name}</span><span>${Math.round(100 * k.time_share)} %</span></span>
    <span class="core-fig"><b>${fmtInt(k.multiplying)}</b><span>${fmtInt(k.step_ms)} ms</span></span>${coreGrid(k, total, title)}
    <span class="core-rest"><span><i class="core-key core-mem"></i>${fmtInt(k.memory)}</span><span><i class="core-key core-other"></i>${fmtInt(k.other)}</span></span></div>`;
}
/** The "i" beside the label: what the figures are an average OF. Shown on hover, focus or tap. */
const CORES_INFO = 'estimate · average of the last ~1 min · top: all steps · below: during decode steps | during prefill steps';
function coresTile(c) {
  if (!c || !isNum(c.cores)) return '';
  // WHAT ITS BYTES ARE (`cores.estimate` `bytes`): the weights alone, or — for SGLang — every byte the
  // MEMORY BANDWIDTH tile models a pass moving, so the two tiles never disagree.
  const bytes = c.bytes === 'modeled' ? 'modeled (weights, KV, GDN state)' : 'of weights';
  // WHAT ITS FLOPs ARE: for SGLang, the COMPUTE tile's own modeled figure over the same window (`flops`).
  const flops = c.flops === 'modeled' ? 'modeled, the compute tile\'s figure' : 'real';
  const title = `estimate, per GPU: ${c.tflops} TFLOP/s ${flops} · ${c.gbps} GB/s ${bytes} · ${c.steps_s} steps/s · ${c.tokens_step} tok/step · ${Math.round(100 * c.mixed_share)} % of steps carry prefill`;
  const k = c.cases || {};
  return `<div class="ceil gpu-cores" title="${esc(title)}">
    <div class="t-head"><span class="t-label">CUDA cores / GPU <span class="info" tabindex="0" role="note" aria-label="${esc(CORES_INFO)}" data-tip="${esc(CORES_INFO)}">i</span></span>
    <b class="t-val">${fmtInt(c.multiplying)}<span class="t-unit"> / ${fmtInt(c.cores)}</span></b></div>
    <div class="core-cases">${coreCase('decode', k.decode, c.cores)}${coreCase('prefill', k.prefill, c.cores)}</div>
    <span class="c-note core-legend"><span><i class="core-key core-mul"></i>multiplying</span><span title="waiting on memory"><i class="core-key core-mem"></i>memory</span><span><i class="core-key core-other"></i>other</span><span>&#9632; = ${Math.round(c.cores / (CORE_COLS * CORE_ROWS))}</span></span>
  </div>`;
}

/** THE ROW UNDER THE TILES: every small figure the card measures as a label·value pair, side by side,
 *  wrapping — "draft 3.3 tok/step · accepted 57 % · prefill 1,190 tok/s · cached 93 % · prefix hit 91 %
 *  · run + wait 16 + 0" (operator, 2026-10-01: "wrap to labels listed one next to each other"). It holds
 *  what the four label-headed rows under the grid used to, and the engine's own figures (the app's
 *  `EngineStats`): the drafts accepted per step and their share, the prefill speed (the COMPUTED
 *  rate), the prompt's cached share, the prefix hit rate off the KV ring, running + waiting, and — for
 *  a server with no ceiling tiles of its own — compute and memory bandwidth per GPU. A figure nobody
 *  measured is "—", with the LB's reason in its `title`; never a zero. A pair with nothing measured
 *  behind it (no drafter, no cached share, no hit ring) is not drawn. */
const fmtTokens = (n) => (n >= 1e9 ? `${(n / 1e9).toFixed(2)}B` : n >= 1e6 ? `${(n / 1e6).toFixed(n >= 1e8 ? 0 : 1)}M`
  : n >= 1e3 ? `${(n / 1e3).toFixed(1)}k` : String(Math.round(n)));

function engineStats(b, hasCeilings, hist, draftMethod, upSince, now) {
  const t = b.throughput || {};
  const stats = [];
  {
    const spec = t.spec_decode;
    if (spec && typeof spec.acceptance_rate === 'number' && typeof spec.mean_accepted_len === 'number') {
      // vLLM's spec_decode counters are the DRAFTER's, whichever method it is, so the pair is labelled
      // with the method the CATALOG publishes for the model this card holds — not "MTP" over a dflash
      // drafter, which is what it read until ef00125 put `speculative_method` on the wire.
      stats.push([drafterName(draftMethod), `${spec.mean_accepted_len.toFixed(1)} tok/step`, '']);
      stats.push(['accepted', `${Math.round(100 * spec.acceptance_rate)} %`, '']);
    }
    const speed = fmtNum((t.prefill_speed || {}).tok_s);
    stats.push(['prefill / request', speed.text == null ? null : `${speed.text} tok/s`, speed.text == null ? (t.prefill_speed || {}).why : speed.title]);
    const cached = (t.prompt || {}).cached_share;
    if (typeof cached === 'number') stats.push(['cached', fmtShare(cached).text, '']);
    const hits = prefixHitOf(hist);
    if (hits != null) stats.push(['prefix hit', `${Math.round(hits)} %`, '']);
    // TOKENS SINCE THE ENGINE LOADED: its own counters (`engine_totals`), every client's.
    const tot = t.engine_totals || {};
    [['Σ generated', tot.generated], ['Σ prompt', tot.prompt], ['Σ prefill', tot.computed]].forEach(([k, v]) => {
      if (isNum(v)) stats.push([k, fmtTokens(v), `${fmtInt(v)} tokens since the engine loaded`]);
    });
    // HOW LONG THE PAIR HAS BEEN UP: docker's start of the head's rank (`cluster.up_since`, the LB's clock).
    if (isNum(upSince) && isNum(now)) stats.push(['up', fmtDuration(now - upSince), 'since the head\'s rank started']);
    if (!hasCeilings) {
      const perf = t.engine_perf || {};
      stats.push(['compute / GPU', isNum(perf.tflops_per_gpu) ? `${perf.tflops_per_gpu.toFixed(1)} TFLOPS` : null, perf.flops_why]);
      stats.push(['mem bw / GPU', isNum(perf.gbps_per_gpu) ? `${perf.gbps_per_gpu.toFixed(0)} GB/s` : null, perf.bytes_why]);
    }
  }
  stats.push(['run + wait', b.depth_known === false ? null : `${b.running} + ${b.waiting}`,
              b.depth_known === false ? 'engine not answering' : '']);
  // TURNS THE LB COULD NOT ACCOUNT FOR (`throughput.ours`) — its OWN bookkeeping, not an engine figure,
  // drawn for every server. 0 is a measurement and says so.
  const ours = t.ours;
  if (ours && ours.uncounted && typeof ours.uncounted === 'object') {
    const reasons = Object.entries(ours.uncounted).filter(([, n]) => isNum(n) && n > 0);
    const n = reasons.reduce((sum, [, k]) => sum + k, 0);
    stats.push(['uncounted', fmtInt(n),
      [reasons.map(([why, k]) => `${fmtInt(k)} ${why.replace(/_/g, ' ')}`).join(', '),
       isNum(ours.counted) ? `${fmtInt(ours.counted)} turns counted` : null].filter(Boolean).join(' · ')]);
  }
  // (A starting pair is not live, so its stats row is never drawn: no reason to override here.)
  return statsRow(stats);
}

/** THE ROW OF label·value PAIRS under a card's tiles ([label, value or null, why]), ONE LINE whatever pairs it carries
 *  (operator 2026-10-30: "ensure each graph cell is compact and doesn't change height over data change"): it wrapped
 *  to one, two or three lines as pairs came and went, moving the card. The line is cut with an ellipsis (gateway.css
 *  `.stats`); every pair is in the row's `title` and each pair's reason in its own. */
function statsRow(stats) {
  const whole = stats.map(([k, v]) => `${k} ${v == null ? '—' : v}`).join(' · ');
  return `<div class="stats" title="${esc(whole)}">${stats.map(([k, v, why]) => `<span class="stat${v == null ? ' dim' : ''}"${why ? ` title="${esc(why)}"` : ''}>`
    + `<span>${esc(k)}</span><b>${v == null ? '—' : esc(v)}</b></span>`).join('')}</div>`;
}

/** WHICH TILE GROUPS A SERVER'S CARD DRAWS: its cluster's `tiles`, which the LB takes from the engine's declaration
 *  (fleet.py NodeEngine / TASK_TILES) — a new engine is a new class there, never a branch here. A server of no fleet
 *  cluster is a classic chat vLLM. */
const CLASSIC_TILES = ['generated', 'prefill', 'prompt', 'queue', 'kv', 'bandwidth', 'compute', 'saturation', 'swarm'];
const tileGroupsOf = (c) => (Array.isArray((c || {}).tiles) ? c.tiles : CLASSIC_TILES);

/** THE SPEECH TILES (a transcription server, `throughput.speech`, throughput.py `speech`): in flight against its seats,
 *  audio transcribed per second, how many times faster than real time, answers per minute — the engine's own counters
 *  over one window, or "—" with the reason. */
function speechTiles(s, b, c) {
  const sp = (b.throughput || {}).speech || {};
  const audio = fmtNum(sp.audio_s_per_s);
  const speed = fmtNum(sp.speed_x);
  const per = sp.per_min || {};
  const done = fmtNum(per.finished);
  const over = isNum(sp.window_s) ? `over ${Math.round(sp.window_s)} s` : '';
  const rest = ['silence', 'failed', 'rejected'].map((k) => `${fmtNum(per[k]).text ?? '—'}&nbsp;${k}`).join(' &middot; ');
  return queueTile(b, b.history, queueOpts(s, b))
    + tile('audio', audio.text, ' s/s', audio.text == null ? reasonText(s, b, sp.audio_why) : over, null,
      { now: sp.audio_s_per_s, title: 'audio seconds transcribed per second' })
    + tile('speed', speed.text, '× real time', speed.text == null ? reasonText(s, b, sp.speed_why) : over, null,
      { now: sp.speed_x, title: 'audio seconds ÷ the seconds the engine spent on them' })
    + tile('answers', done.text, '/min', done.text == null ? reasonText(s, b, sp.why) : rest, null,
      { now: per.finished, title: 'requests answered per minute (silence included); under it the others per minute' });
}

/** THE SPEECH STATS under its tiles: latency, the engine's totals since it started, its own seats and queue, the GPU
 *  memory its process holds — each "—" where the engine does not say (`throughput.speech`). */
function speechStats(b, c, upSince, now) {
  const sp = (b.throughput || {}).speech || {};
  const tot = sp.totals || {};
  const stats = [
    ['latency', isNum(sp.latency_s) ? `${sp.latency_s.toFixed(1)} s` : null, sp.speed_why || 'per request, over the window'],
    ['Σ audio', isNum(tot.audio_seconds) ? fmtDuration(tot.audio_seconds) : null, 'audio transcribed since the engine started'],
    ...['finished', 'silence', 'failed', 'rejected'].map((k) => [`Σ ${k}`, isNum(tot[k]) ? fmtInt(tot[k]) : null,
      `requests ${k} since the engine started`]),
    ['seats', isNum(sp.seats) ? fmtInt(sp.seats) : null, 'the engine\'s own concurrency (MAX_ACTIVE_REQUESTS)'],
    ['queue', isNum(sp.queue_capacity) ? fmtInt(sp.queue_capacity) : null, 'requests the engine may queue behind its seats'],
    ['GPU memory', isNum(sp.gpu_memory_bytes) ? `${(sp.gpu_memory_bytes / 2 ** 30).toFixed(2)} GiB` : null,
      'what the engine process holds on the GPU (its DRM fdinfo)'],
    ['run + wait', b.depth_known === false ? null : `${b.running} + ${b.waiting}`, b.depth_known === false ? 'engine not answering' : ''],
  ];
  if (isNum(upSince) && isNum(now)) stats.push(['up', fmtDuration(now - upSince), 'since the node started']);
  return statsRow(stats);
}

function serverCard(s, b) {
  const c = clusterViewOf(s, b);            // ITS cluster's view: with several, each card is its own
  const groups = tileGroupsOf(c);
  const speech = groups.includes('speech');
  const hist = b.history || {};
  const st = clusterStatusOf(s, b);
  const tone = st.tone;
  const name = clusterName(s, b);
  // THE CLUSTER'S OWN ENGINE, from the profile it serves — never a literal — beside its TP (`topology.tp`).
  // "TP=N" alone where the catalog does not say which, because naming the wrong engine is worse than naming none.
  // A fleet node's engine is its declaration's (`cluster.engine`, the selected choice's); the pair's, its profile's.
  const engine = c.engine ? c.engine.label : engineName((clusterModelInfo(s, c) || {}).engine);
  const tp = `TP=${topologyOf(c).tp}`;
  // Several clusters in the run: each engine card names ITS cluster (the strip's key), after its TP and engine.
  const several = Object.keys((s || {}).clusters || {}).length > 1;
  const kind = `${engine ? `${tp} ${engine}` : tp}${several && b.cluster ? ` · ${b.cluster}` : ''}`;
  // The cluster spans boxes, so its head rank's address is what names it.
  let url = null;
  try { url = new URL(b.url); } catch (_) { url = null; }
  // …and the card it serves on as the FLEET names it (`cluster.hardware`), never the engine's own device string.
  const meta = `${url ? url.host : b.url}${c.hardware ? ` · ${c.hardware}` : ''}`;
  // The footer carries only what the badge does not say: whose park holds the cluster.
  const foot = clusterFoot(st, c);
  // What the badge names (`modelInfoOf`), once; and whether the KV share is already a tile. A pair on
  // its way up names the model it is bringing up (the cluster card's own pending badge).
  const info = b.live ? modelInfoOf(s, b) : (startingPair(s, b) ? clusterModelInfo(s, c) : null);
  const kvOnTile = Array.isArray(hist.kv_pct) && hist.kv_pct.some(isNum);
  // A server that is not answering and was never measured — no sample in its rings — is ONE LINE: its
  // name and its badge. Three tiles reading "—" over empty graphs carried nothing.
  const figures = b.live || ['gen_tok_s', 'depth'].some((k) => Array.isArray(hist[k]) && hist[k].some(isNum));
  // A cluster short of running refuses connections by design — STARTING and NOT HELD already
  // say so — and printing the refusal in red would be an alarm about the system working as designed.
  const quiet = c.state !== 'running';
  const cardTone = tone === 'ok' ? 'ok' : tone === 'bad' ? 'bad'
    : tone === 'warn' ? 'warn' : tone === 'park' ? 'park' : 'idle';
  const sat = b.live && groups.includes('saturation') ? figureOf(s, clusterViewOf(s, b), 'saturation') : null;
  return `<article class="card server ${cardTone}${b.live ? '' : ' down'}">
    <div class="card-head">
      <div class="box-id">
        ${diode(st)}
        <div>
          <div class="box-name">${esc(name)} <span class="box-kind">&middot; ${esc(kind)}</span></div>
          <div class="box-meta">${esc(meta)}</div>
        </div>
      </div>
      <span class="badge ${tone}">${esc(st.label)}</span>
    </div>
    ${info ? modelBadge(info) : ''}
    ${modelLine(s, b, !!info, kvOnTile)}
    ${figures && speech ? `<div class="tiles tiles4">${speechTiles(s, b, c)}</div>` : ''}
    ${figures && !speech ? `<div class="tiles tiles${tileCount(b)}">
      ${throughputTiles(s, b)}
      ${queueTile(b, hist, queueOpts(s, b))}
      ${kvTile(s, b, hist)}
      ${kvOffloadOf(b) ? kvRamTile(s, b, hist) : ''}
    </div>` : ''}
    ${b.live && speech ? speechStats(b, c, c.up_since, s.now) : ''}
    ${b.live && !speech ? engineStats(b, !!sat, hist, (info || {}).draftMethod, c.up_since, s.now) : ''}
    ${sat ? ceilingTiles(sat, rankLink(c)) : ''}
    ${b.live && figures && groups.includes('bandwidth') ? unusedGraphs(b, sat, c) : ''}
    ${foot ? `<div class="card-foot">${foot}</div>` : ''}
    ${b.last_error && !quiet ? `<div class="card-err">${esc(b.last_error)}</div>` : ''}
  </article>`;
}

/** The engine card: the SELECTED server's head in its section (`#lb-servers`, beside its card). The other servers of
 *  the run are their own tabs, drawn nowhere else. Markup that did not change is not written again (`paint`), so a
 *  quiet poll moves nothing. */
function renderServers(s) {
  const head = headOf(s, theCluster(s));
  paint($('lb-servers'), visibleBackends(s).filter((b) => head && b === head).map((b) => serverCard(s, b)).join(''));
}

/* ── THE SWARM ────────────────────────────────────────────────────────────────────────────────
   One HOLD — three seconds, the app's `SwarmRow` — asks for ONE deep-research session against the
   model ALREADY LOADED, for its cap. While one runs the same button says so and cannot be held; when
   one cannot be asked for it is greyed, with the reason under it. It
   restarts nothing, changes no setting and loads no model — and /benchmarks records those minutes
   the way it records any other traffic, so the control's whole job is to generate load.

   It is NOT the thirty-run matrix. That tears the pair down and rebuilds it per configuration for
   thirteen hours, and on 2026-10-01 a click on this very button did exactly that to a working
   default. The matrix is a terminal-only operation now (`bench_queue_runner.py --start-matrix`) and
   this page has no way to ask for it.

   The container serving this page cannot start a host process and cannot see one. So the panel is
   built from three kinds of fact and never blurs them together:

     * THE HOST RUNNER'S REPORT — whether a session process is up, how far into its cap it is, and
       what it did with the last request. It carries its own clock, and when that clock goes quiet
       the panel says the runner is not listening instead of guessing on its behalf.
     * WHAT THIS PAGE ASKED FOR and has not been answered yet — shown as "requested", never as a
       swarm that started.
     * WHETHER A MODEL IS LOADED AT ALL, because a swarm never loads one: with nothing serving the
       control says so rather than spending a click finding out. */

let swarmBusy = false;      // a POST to /lb/swarm/start is out

/** What the host runner answered, as state words that never promise more than it did: `started` is
 *  the ONLY one that means a swarm is now running, and the runner said so. Its own sentence (`detail`)
 *  is the line's title. */
const SWARM_RESULT = {
  started: ['ok', 'started'],
  'already running': ['warn', 'refused &middot; swarm already up'],
  'matrix running': ['warn', 'refused &middot; matrix up'],
  // The sealed runner (lb/bench_runner) proves its network cannot reach Anthropic before EVERY launch;
  // when that check fails, claude is not started, and the check's own sentence is the line's title.
  unsealed: ['bad', 'refused &middot; egress not sealed'],
  'not mine': ['warn', 'refused'],
  failed: ['bad', 'start failed'],
  'dry-run': ['warn', 'dry run &middot; nothing started'],
  // The host runner re-checks the duration in the request file and refuses one outside the range rather
  // than quietly running its own default (`bench_session.cap_from_request`), so the page has a word for it.
  'bad cap': ['bad', 'refused &middot; duration out of range'],
};

/** An age against the LB's OWN clock, not the browser's: the two are different machines' ideas of
 *  now, and a few seconds of skew would show a file written "in 3 s". */
function swAgo(now, at) {
  if (at == null || now == null) return '';
  const secs = Math.max(0, Math.round(now - at));
  if (secs < 90) return `${secs}s ago`;
  if (secs < 5400) return `${Math.round(secs / 60)} min ago`;
  return `${(secs / 3600).toFixed(1)} h ago`;
}

function swMinutes(now, at) {
  if (at == null || now == null) return null;
  return Math.max(0, (now - at) / 60);
}

/** The session block, or an empty one: a page that reads `undefined.phase` shows nothing at all. */
function swSession(q) { return (q.runner || {}).session || {}; }

/** ` on <box>` for the box the runner's own configuration names (runner.json `box`), or nothing: the page
 *  serves every deployment, so it never writes a box down itself. */
function swOn(q) {
  const box = (q.runner || {}).box;
  return typeof box === 'string' && box ? ` on ${box}` : '';
}

/** Is a swarm RUNNING? `null` when no runner is listening: not knowing is its own answer and must
 *  not be drawn as "no". */
function swarmLive(q) {
  const runner = q.runner || {};
  if (!runner.listening || !runner.session) return null;
  return runner.session.alive === true;
}

function matrixLive(q) {
  const runner = q.runner || {};
  if (!runner.listening || !runner.matrix) return null;
  return runner.matrix.running === true;
}

/** Minutes as this panel writes them everywhere: whole where they are whole, exact where someone typed
 *  a fraction. One formatting of the one number, instead of the same ternary in four places. */
const fmtCap = (v) => (v % 1 ? String(v) : v.toFixed(0));

/** The durations the list offers. 15 FIRST and marked the default: it is what the host runner uses for a
 *  request that names no duration, and the operator's ask (2026-10-05) was for 30 and more BESIDE it, not
 *  instead of it. Any whole number inside the LB's own range may be typed instead (`custom…`). */
const CAP_STEPS = [15, 30, 45, 60, 90, 120, 180, 240];

/** The durations this gateway will accept, as IT says (`swarm.cap_min_range`) — never from the markup, so
 *  this page cannot offer a number an older LB would answer with a 400. The fallback pair is the shipped
 *  one (`bench_session.CAP_MIN_RANGE`), for the moment before the first poll lands. */
function swCapRange(q) {
  const r = q.cap_min_range;
  return Array.isArray(r) && r.length === 2 && isNum(r[0]) && isNum(r[1]) && r[0] < r[1]
    ? [r[0], r[1]] : [5, 240];
}

/** The cap a click gets when it names no duration at all: the host runner's own. */
function swDefaultCap(q) {
  const cap = q.cap_min || (q.runner || {}).cap_min;
  return cap && cap > 0 ? cap : 15;
}

/** The cap OF THE SESSION this panel is about — the one running, or the last one reported. The progress
 *  and the outcome are drawn against this and never against the chooser: a swarm asked for at 120 min
 *  does not become a quarter-hour one because the list was put back to its default under it. */
function swCap(q) {
  const cap = swSession(q).cap_min || q.cap_min || (q.runner || {}).cap_min;
  return cap && cap > 0 ? cap : 15;
}

/** What the chooser says now — the minutes a hold would ask for — or the default while it is unfilled. */
function swChosenCap(q) {
  const picked = wholeOf($('swarm-cap'));
  return Number.isInteger(picked) && picked > 0 ? picked : swDefaultCap(q);
}

/** Why the duration on screen cannot be sent, in words, or ''. Made while the button is still unheld:
 *  three seconds of holding must never end in "that number is out of range" — the same rule
 *  `startProblem` keeps for the cluster's five. */
function swCapProblem(q) {
  const el = $('swarm-cap');
  if (!el || !el.dataset.choices) return '';        // not filled yet: there is nothing to be wrong
  const [lo, hi] = swCapRange(q);
  const picked = wholeOf(el);
  return Number.isInteger(picked) && picked >= lo && picked <= hi
    ? '' : `the duration must be a whole number of minutes from ${fmtCap(lo)} to ${fmtCap(hi)}`;
}

/** `3 of 15 min`, from the LB's clock and the session's own start — against the cap THAT session was
 *  given, so a 30- or a 120-minute run reads its own progress and not the default's. */
function swProgress(q, now) {
  const elapsed = swMinutes(now, swSession(q).started_at);
  const cap = swCap(q);
  if (elapsed == null) return `${fmtCap(cap)} min`;
  return `${elapsed < 10 ? elapsed.toFixed(1) : Math.round(elapsed)} of ${fmtCap(cap)} min`;
}

/** How the last session ENDED, as data: `[tone, word, bits, title]` — "last swarm" (or "last swarm
 *  failed" / "lost"), then how long it ran, its exit code and the counts the runner wrote into its
 *  detail ("5 answer blocks, 2 workflows"); the runner's whole sentence is the title. `timeout`'s 124
 *  is the cap and the SUCCESS — the session is meant to be stopped by the clock. An exit 0 means
 *  `claude` stopped on its own before the cap, which is a defect to show, not a tick. */
function swOutcome(session) {
  const phase = session.phase;
  if (phase !== 'finished' && phase !== 'failed' && phase !== 'lost' && phase !== 'stopped') return null;
  const bits = [];
  const ran = swMinutes(session.ended_at, session.started_at);
  if (ran != null && session.ended_at != null) bits.push(`${ran < 10 ? ran.toFixed(1) : Math.round(ran)} min`);
  if (isNum(session.rc)) bits.push(`exit ${session.rc}`);
  const detail = String(session.detail || '');
  const answers = /(\d+) answer/.exec(detail), flows = /(\d+) workflow/.exec(detail);
  if (answers) bits.push(`${answers[1]} answer${answers[1] === '1' ? '' : 's'}`);
  if (flows) bits.push(`${flows[1]} workflow${flows[1] === '1' ? '' : 's'}`);
  const capped = session.rc === 124 || session.rc === 130;
  // The one word that reads the exit code for the operator: 124 is the cap, the normal end; anything
  // else on a finished session ended early.
  if (phase === 'finished' && isNum(session.rc)) bits.push(capped ? 'capped' : 'ended early');
  // `stopped` is a deliberate end, not an outcome to alarm about: faint, like a capped run.
  const tone = phase === 'failed' ? 'bad'
    : phase === 'stopped' ? 'faint'
    : (phase === 'lost' || !capped ? 'warn' : 'faint');
  return [tone, phase === 'finished' ? 'last swarm' : `last swarm ${phase}`, bits, detail];
}

/** The session, the request, the runner and the matrix — each as the thing it is, as data, ordered by
 *  what the operator would act on first. The button above says what a hold starts and whether one is
 *  running (`renderSwarmControls`), so no line here repeats that. */
function swStateLines(q, now) {
  const lines = [];
  const session = swSession(q);
  const live = swarmLive(q) === true;
  // The class value is CLOSED here: left open, a browser's attribute value ran on past the `>` to the
  // next quote, which lost the title and, on a line with no title, swallowed the line's own text.
  const sub = (tone, html, title) => `<span class="sw-sub${tone ? ` sw-${tone}` : ''}"`
    + `${title ? ` title="${esc(title)}"` : ''}>${html}</span>`;
  if (live) {
    const bits = [];
    if (session.label) bits.push(esc(session.label));
    if (session.pid) bits.push(`pid ${esc(session.pid)}`);
    if (session.claude_pid) bits.push(`claude pid ${esc(session.claude_pid)}`);
    if (session.phase && session.phase !== 'running') bits.push(esc(session.phase));
    lines.push(sub('', `<span class="diode ok pulse"></span>${bits.length ? bits.join(' &middot; ') : 'session up'}`,
                   session.detail));
  }
  const runner = q.runner || {};
  const outcome = live ? null : swOutcome(session);
  if (q.request_pending) {
    const at = (q.request || {}).at;
    // The MINUTES THIS REQUEST ASKED FOR, from the request itself: a duration the operator chose is the
    // one thing about a pending click that is not yet visible anywhere else on the panel.
    const asked = (q.request || {}).cap_min;
    lines.push(sub('warn', `requested${isNum(asked) ? ` &middot; ${esc(fmtCap(asked))} min` : ''}`
      + `${at ? ` ${esc(swAgo(now, at))}` : ''}`
      + `${runner.listening ? '' : ' &middot; no runner listening'}`));
  } else if (runner.last && runner.last.result && !(runner.last.result === 'started' && (live || outcome))) {
    // A `started` the session block already accounts for — up, or finished — is not said twice.
    const [tone, said] = SWARM_RESULT[runner.last.result] || ['warn', `runner said ${esc(runner.last.result)}`];
    lines.push(sub(tone === 'ok' ? 'faint' : tone,
      `${said}${runner.last.at ? ` ${esc(swAgo(now, runner.last.at))}` : ''}`, runner.last.detail));
  }
  if (outcome) {
    const [tone, word, bits, title] = outcome;
    lines.push(sub(tone, [`${word}${session.ended_at ? ` ${esc(swAgo(now, session.ended_at))}` : ''}`]
      .concat(bits).join(' &middot; '), title));
  }
  if (q.model_loaded === false) {
    lines.push(sub('warn', '<span class="diode bad"></span>no model loaded', q.model_detail || ''));
  }
  if (!runner.listening && runner.error) {
    lines.push(sub('warn', esc(runner.error)));
  }
  if (matrixLive(q) === true) {
    const matrix = runner.matrix || {};
    lines.push(sub('faint', [esc(`matrix up${swOn(q)}`), matrix.script ? esc(matrix.script) : '',
      matrix.pid ? `pid ${esc(matrix.pid)}` : '', matrix.since ? esc(swAgo(now, matrix.since)) : '']
      .filter(Boolean).join(' &middot; ')));
  }
  return lines.join('');
}

function renderSwarm(s) {
  const panel = $('swarm');
  if (!panel) return;
  const q = (s && s.swarm) || {};
  // A deep-research swarm loads a CHAT server: a server whose declaration publishes no `swarm` group is not offered one.
  if (!q.configured || !tileGroupsOf(theCluster(s)).includes('swarm')) {
    panel.hidden = true;
    return;
  }
  panel.hidden = false;
  paint($('swarm-state'), swStateLines(q, s.now));
  // The chooser first: the button's label names whatever it says, so it must be filled by the time the
  // button is drawn from it.
  renderSwarmCap(q, swarmLive(q) === true);
  renderSwarmControls(q, s.now);
}

/** THE DURATION — a list of the stretches worth clicking plus a typed value, which is the control the
 *  cluster's own knobs use (`fillNumberSelect`, `syncCustom`, `wholeOf`). The steps are clipped to what
 *  this LB accepts, the runner's own default is marked as such, and the whole field is locked while a
 *  swarm is up or a request is out: that session's cap is already decided, and a field moving under it
 *  would say something untrue about the run on screen. */
function renderSwarmCap(q, live) {
  const el = $('swarm-cap');
  if (!el) return;
  const [lo, hi] = swCapRange(q);
  const dflt = swDefaultCap(q);
  const steps = CAP_STEPS.filter((v) => v >= lo && v <= hi);
  if (!steps.includes(dflt) && dflt >= lo && dflt <= hi) steps.push(dflt);
  const picked = wholeOf(el);
  fillNumberSelect(el, steps, Number.isInteger(picked) ? picked : dflt, dflt);
  const custom = customOf(el);
  // The typed field's own bounds come from the LB too, so the browser's number spinner stops where the
  // gateway stops rather than where this file was written.
  if (custom) { custom.min = String(lo); custom.max = String(hi); }
  syncCustom(el);
  el.disabled = live || swarmBusy;
  if (custom) custom.disabled = el.disabled;
  const note = $('swarm-cap-note');
  if (note) {
    note.textContent = swCapProblem(q)
      || (live ? `the swarm up is capped at ${fmtCap(swCap(q))} min`
        : `${fmtCap(swChosenCap(q))} min of load \u00b7 ${fmtCap(lo)}\u2013${fmtCap(hi)} accepted`);
  }
}

/** May a swarm be asked for at all? Not while one is running — two at once would make both
 *  measurements meaningless — not while the matrix holds the pair, not while no host runner is
 *  listening (nothing would ever read the request), and not with nothing loaded, because this
 *  control never loads a model. Anything else: any time, which is what the operator asked for. */
function swCanStart(q) {
  return swarmLive(q) !== true && matrixLive(q) !== true && q.model_loaded !== false
    && (q.runner || {}).listening === true && !swCapProblem(q);
}

/** Why the control is not offered, in the order the host runner itself would refuse. */
function swWhyNot(q) {
  if (swarmLive(q) === true) return 'swarm running';
  if (matrixLive(q) === true) return `matrix running${swOn(q)}`;
  if (q.model_loaded === false) return 'no model loaded';
  if (!(q.runner || {}).listening) return `no runner listening${swOn(q)}`;
  // Last, because it is the only one of these the operator can fix on this page: the four above are
  // about the fleet, this one is about the number in the field.
  return swCapProblem(q);
}

/** The button: what it would start (its label carries the cap the runner reports), or the swarm that is
 *  running, and whether it can be held. A hold under way is taken back the moment a poll says a swarm
 *  can no longer be asked for — the state under it changes every 3 s on its own. */
function renderSwarmControls(q, now) {
  const start = $('swarm-start');
  const live = swarmLive(q) === true;
  // THE SESSION'S cap while one is up, THE CHOOSER'S when none is: the running label must say what the
  // run on screen was given, and the idle label what a hold would ask for. The last session's cap is
  // not the next one's, which is why these are two different questions (`swCap`, `swChosenCap`).
  const cap = live ? swCap(q) : swChosenCap(q);
  setHold(start, swarmBusy ? 'requesting…'
    : (live ? `Test swarm running · ${swProgress(q, now)}`
      : `Start test swarm · ${fmtCap(cap)} min`), swarmBusy || !swCanStart(q), 'swarm');
  start.classList.toggle('busy', swarmBusy);
  start.classList.toggle('live', live);
  // THE TRACK ALONG THE BOTTOM, while a swarm is up, is that session's own clock against ITS cap — so a
  // two-hour run crawls where a quarter-hour one fills, instead of every run filling in fifteen minutes
  // and then sitting there saying nothing. At every other moment the track is the HOLD's fill
  // (`holdButton`), which is why this is written after `setHold` and zeroed the moment one is not live.
  const track = start.querySelector ? start.querySelector('.hold-track i') : null;
  if (track) {
    const elapsed = live ? swMinutes(now, swSession(q).started_at) : null;
    track.style.transform = `scaleX(${elapsed != null && cap > 0 ? Math.max(0, Math.min(1, elapsed / cap)) : 0})`;
  }
  // What a swarm IS — the paragraph the row used to carry — is the button's title.
  start.title = swWhyNot(q) || `Press and hold 3 s: one deep-research swarm (two researches at once, 30 agents each, `
    + `relaunched until the ${fmtCap(cap)} min cap) against the model already loaded, carried out by `
    + `the swarm runner (bench_queue_runner.py)${swOn(q)}. Nothing is restarted, no setting is changed, no model is loaded or `
    + 'unloaded; /benchmarks records its minutes like any traffic. The thirty-run matrix is started at the keyboard, never here.';
  // The ONE reason it cannot be held, where the lines above do not already say it. A duration out of
  // range comes first: it is the only one of them the operator fixes here, and the field it is about is
  // right above the button.
  const why = $('swarm-why');
  const reason = swCapProblem(q)
    || (!live && !swarmBusy && !(q.runner || {}).listening && !(q.runner || {}).error ? swWhyNot(q) : '');
  why.textContent = reason;
  why.hidden = !reason;
}

function showSwarmError(msg) {
  const el = $('swarm-error');
  el.textContent = msg;
  el.hidden = !msg;
}

/** Post the intent, then re-read the state it caused. The control stays disabled until the answer
 *  is in AND that re-read has landed, so a second hold cannot race the first; and what is drawn
 *  afterwards is what /lb/status says, never what was asked. */
async function swarmAct(cap) {
  if (swarmBusy) return;
  swarmBusy = true;
  const asked = (lastStatus && lastStatus.swarm) || {};
  // The chooser locks with the button: while a request is out the duration is already decided, and a
  // field still moving would be inviting an edit to something that has been sent.
  renderSwarmCap(asked, swarmLive(asked) === true);
  renderSwarmControls(asked, lastStatus && lastStatus.now);
  showSwarmError('');
  try {
    const r = await fetch('/lb/swarm/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      // `cap_min` is the minutes asked for; an empty object is the host runner's own default, which is
      // what this control sent for its whole life before 2026-10-05 and what the LB still takes.
      body: JSON.stringify(isNum(cap) ? { cap_min: cap } : {}),
    });
    if (!r.ok) throw new Error(await refusal(r));
  } catch (e) {
    showSwarmError(e.message);
  } finally {
    swarmBusy = false;
    await poll({ fresh: true });      // the state THIS hold caused, not a poll already out
  }
}

/** Not one stray click away: it puts the operator's own hardware under a quarter of an hour of real
 *  load, and this page has no authentication of its own beyond the vhost's. So it takes a three-second
 *  hold — and what it will and will not do is written beside the button, above all that nothing is
 *  reconfigured. The state is checked again when the hold completes: three seconds is a poll ago. */
function swarmHeld() {
  const q = (lastStatus && lastStatus.swarm) || {};
  if (swarmBusy || !swCanStart(q)) return;
  swarmAct(swChosenCap(q));
}

let lastOk = null;      // WHEN we last heard from /lb/status
let lastError = '';     // why the latest poll failed; a success clears it
let lastStatus = null;  // WHAT it said — the controls redraw from it between polls
let polling = null;     // the ONE /lb/status request out, if any
const loadedAt = Date.now();

/** WHY NOTHING CAN BE LEASED, quoted. With broker_parrot's database unreachable the LB takes no
 *  lease, so a turn with no held server is refused — the "broker db FAIL" chip never said why, and
 *  a failing control tick was not on the page at all. */
function renderAlert(s) {
  const el = $('lb-alert');
  if (!el) return;
  const causes = [];
  if (s.db_ok === false) {
    causes.push('<b>broker db unreachable</b>'
      + (s.db_error ? ` &middot; <span class="why">${esc(s.db_error)}</span>` : ''));
  }
  if (s.last_tick_error) {
    causes.push(`<b>control loop tick failed</b> &middot; <span class="why">${esc(s.last_tick_error)}</span>`);
  }
  el.hidden = !causes.length;
  paint(el, causes.map((c) => `<div>${c}</div>`).join(''));
}

/** The header's freshness, from the AGE of the last success — not from a request settling. A poll
 *  that hangs settles nothing, and "updated just now" stayed up for as long as it hung. */
function renderFreshness() {
  const since = Date.now() - (lastOk == null ? loadedAt : lastOk);
  if (!lastError && since < STALE_AFTER_MS) {
    if (lastOk == null) return;                 // still connecting
    $('connection-status').className = 'conn ok';
    $('status-text').textContent = 'updated just now';
    return;
  }
  $('connection-status').className = 'conn bad';
  const last = lastOk == null ? 'never' : `${Math.round(since / 1000)}s ago`;
  $('status-text').textContent = `stale (last ${last})${lastError ? ` — ${lastError}` : ''}`;
}

/** Read /lb/status: ONE request at a time, abandoned after POLL_TIMEOUT_MS. The interval used to
 *  start a fetch every 3 s whatever the last was doing; against a hung LB none settled, so none
 *  could say so, and they held the browser's six connections to this origin until the page's own
 *  POSTs queued behind them. A caller arriving mid-poll shares the one out — unless it needs the
 *  state its own write caused (`fresh`), and then it waits for that one and starts its own. */
async function poll({ fresh = false } = {}) {
  if (fresh && polling) await polling;
  if (!polling) polling = pollOnce().finally(() => { polling = null; });
  return polling;
}

/** THE WHOLE PAGE from one status — every poll, and again when a server tab changes the selection (clusters.js
 *  `selectTab` → `renderStatus(lastStatus)`): the selected server's card, its head's engine card and its requests, the
 *  Cluster section's tiles (its marks), the fleet's figures. */
function renderStatus(s) {
  renderAlert(s); renderCluster(s); renderSwarm(s); renderQueued(s); renderServers(s);
  // The server tabs (clusters.js, loaded before this file); the guard is for the node harnesses that load
  // gateway.js alone (tests/*.node.js).
  if (typeof renderClusters === 'function') renderClusters(s);
  renderMachines(s); renderRequests(s);
  placeMarks();
}

async function pollOnce() {
  const abandon = new AbortController();
  const timer = setTimeout(() => abandon.abort(), POLL_TIMEOUT_MS);
  try {
    const r = await fetch('/lb/status', { signal: abandon.signal });
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const s = await r.json();
    renderStatus(s);
    lastStatus = s;
    lastOk = Date.now();
    lastError = '';
  } catch (e) {
    // Keep the last data on screen. Blanking a monitoring page on a transient error is worse
    // than showing old numbers, as long as they are labelled as old.
    lastError = abandon.signal.aborted ? `no answer in ${POLL_TIMEOUT_MS / 1000}s` : e.message;
  } finally {
    clearTimeout(timer);
    renderFreshness();
  }
}

// ── connect an agent ─────────────────────────────────────────────────────────

/** CONNECT AN AGENT — the top bar's `connect` (operator, 2026-10-12: "Can we include somewhere connect button, top
 *  navbar? Will show how to connect agent to this vllm"). A modal dialog that hands over, ready to paste, how an API
 *  client — Claude Code, the OpenAI SDK, curl — reaches THIS LB from the address the page was opened at; and where
 *  that address takes no API client, says so and offers what does.
 *
 *  NOTHING HERE NAMES A HOST. The base is `location.origin`, or the SSH forward the deployment declares; the model
 *  and its window are the LB's live /v1/models (what fullclaudevllm reads); the DOOR is what /auth/session answers
 *  at this origin; /lb/status `connect` (vllm_lb/connect.py) states what the LB cannot see — `tunnel`, its address
 *  on its host, and `basic`, an edge that takes HTTP Basic on the API — and the budgets a client must out-wait.
 *
 *  THE DOORS (2026-10-12, each checked against a live one):
 *    lb      the LB itself 404s /auth/session (proxy.py: an unknown path outside /v1/): no credential.
 *    open    vllm-gate VLLM_GATE_AUTH=false answers `{"auth":"false"}`. A GET passes; a POST must carry Origin ==
 *            the gate's origin (the gate's own check), and an API client sends none → 403.
 *    signin  vllm-gate jwt: a session (200 + email) or 401. Only the browser's session cookie opens it — nginx drops
 *            Authorization before it asks the gate, the issuer mints no token without a browser — so an API client
 *            gets 401, unless the edge also takes Basic (`basic`).
 *  A door that takes no API client gets no snippet. A credential is never a value: `<user>:<password>`.
 *
 *  The Claude Code lines are fullclaudevllm's (~/.bashrc), each from what it depends on: the window and 80 % compact
 *  from /v1/models; the first-byte and idle budgets from the LB's own, the idle watchdog at most its documented
 *  30 min; a FIXED 16384-token reply cap (a share of the window reserved most of it); CLAUDE_CODE_ATTRIBUTION_HEADER=0
 *  keeps the `x-anthropic-billing-header: cc_version=…` line out of the system prompt (captured, 2.1.296) — a prefix
 *  that changes with each release and entrypoint misses the prefix cache. An open route sends a token, not an API
 *  key (a key asks the CLI to approve it); Basic sends an API key, because Claude Code's bearer REPLACES a custom
 *  Authorization header (captured: Basic + ANTHROPIC_AUTH_TOKEN went out as `Bearer`).
 *
 *  PER-USER API KEYS (operator, 2026-10-12; the sign-in gate's key store). Where the sign-in gate's /auth/session carries
 *  `api_keys` — the ONE place that says this door takes them — this address is an API route (`apikey`): a key on
 *  GET /v1/models + POST /v1/*, `Authorization: Bearer` or `x-api-key`. Claude Code gets it as ANTHROPIC_AUTH_TOKEN
 *  (captured 2.1.296: AUTH_TOKEN → `Authorization: Bearer vgk_…`, API_KEY → `x-api-key: vgk_…`; the gate takes both,
 *  the token asks for no approval), curl / the OpenAI SDK as a bearer, Anthropic's curl as x-api-key — always
 *  `<your key>`, never a value. The key section beside that route creates one (POST /auth/keys: a name and one of the
 *  gate's lifetimes, its default preselected), shows it ONCE with copy (forgotten when the dialog closes: the gate
 *  keeps only its hash), lists the person's own (GET /auth/keys: name, prefix + first 4, created, expires, last used) and
 *  revokes one on a second press (POST /auth/keys/revoke). Every refusal is said in the gate's words. */
const CONNECT = {
  compact: 0.8,
  outputTokens: 16384,
  idleMaxMs: 30 * 60 * 1000,
  retries: 10,
  prompt: 'Reply with 7',
  maxTokens: 1024,
  focusable: 'button, input, select, a[href]',
  hostKey: 'vllm-connect-host',
  portKey: 'vllm-connect-port',
  keyPlaceholder: '<your key>',
  keyArmMs: 4000,
};
const CONNECT_TABS = [
  { key: 'claude', label: 'Claude Code' },
  { key: 'openai', label: 'OpenAI' },
  { key: 'anthropic', label: 'Anthropic' },
];
const cn = { parts: null, server: null, door: null, doorStatus: 0, model: null, modelError: '', connect: null, routes: [],
  route: '', tab: 'claude', host: '', port: '', downOnBackdrop: false,
  keys: null, keyList: null, keyError: '', keyNotice: '', newKey: null, keyArmed: null };

/** Which door this origin is, from what its /auth/session answered. */
function connectDoor(status, body) {
  if (status === 404) return 'lb';
  if (status === 200 && body && body.auth === 'false') return 'open';
  if (status === 401 || (status === 200 && body && typeof body.email === 'string')) return 'signin';
  return 'unknown';
}

/** The routes an API client has from here: this address when its door takes one, then what the deployment declares. */
function connectRoutes(door, origin, connect, local) {
  const c = connect || {};
  const out = [];
  if (door === 'lb') out.push({ key: 'here', label: 'this address', base: origin, auth: 'none' });
  if (door === 'signin' && local.keys) out.push({ key: 'apikey', label: 'this address · API key', base: origin, auth: 'key' });
  if (door === 'signin' && c.basic) out.push({ key: 'basic', label: 'this address · Basic', base: origin, auth: 'basic' });
  if (c.tunnel) {
    out.push({ key: 'ssh', label: 'SSH forward', base: `http://127.0.0.1:${local.port}`, auth: 'none',
      ssh: `ssh -N -L ${cnSh(`127.0.0.1:${local.port}:${c.tunnel}`)} ${local.host ? cnSh(local.host) : '<server>'}` });
  }
  return out;
}

/** A shell word: as is when it is plainly safe, else single-quoted. */
const cnSh = (v) => (/^[A-Za-z0-9_./:@%+=,-]+$/.test(String(v)) ? String(v) : cnQuote(v));
const cnQuote = (v) => `'${String(v).replace(/'/g, "'\\''")}'`;
const cnPortOf = (tunnel) => String(tunnel || '').slice(String(tunnel || '').lastIndexOf(':') + 1);
const cnPortOk = (p) => /^\d+$/.test(p) && +p >= 1 && +p <= 65535;

/** Every client's snippet for one route and model: `{ claude, openai, anthropic }`, each a list of `{ title, text }`. */
function connectSnippets(route, model, connect) {
  const basic = route.auth === 'basic';
  const key = route.auth === 'key';
  const yourKey = CONNECT.keyPlaceholder;
  const firstByteMs = Math.round(connect.first_byte_timeout_s * 1000);
  const idleMs = Math.min(Math.round(connect.stall_timeout_s * 1000), CONNECT.idleMaxMs);
  const win = model.max_model_len;
  const claude = [
    `ANTHROPIC_BASE_URL=${cnSh(route.base)}`,
    `ANTHROPIC_MODEL=${cnSh(model.id)}`,
    ...(basic
      ? ['ANTHROPIC_API_KEY=unused',
        `ANTHROPIC_CUSTOM_HEADERS="Authorization: Basic $(printf %s '<user>:<password>' | base64 | tr -d '\\n')"`]
      : [`ANTHROPIC_AUTH_TOKEN=${key ? cnSh(yourKey) : 'unused'}`]),
    `CLAUDE_CODE_MAX_CONTEXT_TOKENS=${win}`,
    `CLAUDE_CODE_AUTO_COMPACT_WINDOW=${Math.floor(win * CONNECT.compact)}`,
    `CLAUDE_CODE_MAX_OUTPUT_TOKENS=${CONNECT.outputTokens}`,
    `API_TIMEOUT_MS=${firstByteMs}`,
    `CLAUDE_STREAM_FIRST_BYTE_TIMEOUT_MS=${firstByteMs}`,
    `CLAUDE_STREAM_IDLE_TIMEOUT_MS=${idleMs}`,
    'API_FORCE_IDLE_TIMEOUT=0',
    `CLAUDE_CODE_MAX_RETRIES=${CONNECT.retries}`,
    'CLAUDE_CODE_ATTRIBUTION_HEADER=0',
    'claude',
  ].join(' \\\n');
  const body = JSON.stringify({ model: model.id, max_tokens: CONNECT.maxTokens,
    messages: [{ role: 'user', content: CONNECT.prompt }] });
  const user = basic ? ["  -u '<user>:<password>'"] : [];
  const curl = (path, headers) => [`curl ${cnSh(route.base + path)}`, "  -H 'Content-Type: application/json'",
    ...headers, ...user, `  -d ${cnQuote(body)}`].join(' \\\n');
  const bearer = key ? [`  -H ${cnQuote(`Authorization: Bearer ${yourKey}`)}`] : [];
  const py = (s) => JSON.stringify(s);
  const python = [
    ...(basic ? ['import base64', ''] : []),
    'from openai import OpenAI',
    '',
    `client = OpenAI(base_url=${py(route.base + '/v1')}, api_key=${py(key ? yourKey : 'unused')}, timeout=${connect.first_byte_timeout_s}${basic
      ? ',\n                default_headers={"Authorization": "Basic " + base64.b64encode(b"<user>:<password>").decode()}'
      : ''})`,
    'reply = client.chat.completions.create(',
    `    model=${py(model.id)}, max_tokens=${CONNECT.maxTokens},`,
    `    messages=[{"role": "user", "content": ${py(CONNECT.prompt)}}])`,
    'print(reply.choices[0].message.content)',
  ].join('\n');
  return {
    claude: [{ title: 'shell', text: claude }],
    openai: [{ title: 'curl · /v1/chat/completions', text: curl('/v1/chat/completions', bearer) },
      { title: 'python · openai', text: python }],
    anthropic: [{ title: 'curl · /v1/messages', text: curl('/v1/messages', [
      ...(key ? [`  -H ${cnQuote(`x-api-key: ${yourKey}`)}`] : []), "  -H 'anthropic-version: 2023-06-01'"]) }],
  };
}

function cnEl(tag, cls, text) {
  const el = document.createElement(tag);
  if (cls) el.className = cls;
  if (text != null) el.textContent = text;
  return el;
}
function cnLoad(key) { try { return localStorage.getItem(key); } catch (_) { return null; } }
function cnSave(key, value) { try { localStorage.setItem(key, value); } catch (_) { /* this viewer keeps nothing */ } }

/** A copy button: writes `text()` and says whether it landed — a refusal in the browser's words. */
function cnCopyButton(text, label) {
  const b = cnEl('button', 'cn-copy', 'copy');
  b.type = 'button';
  b.dataset.copy = '';
  b.setAttribute('aria-label', label);
  b.addEventListener('click', () => {
    const wrote = navigator.clipboard
      ? Promise.resolve().then(() => navigator.clipboard.writeText(text()))
      : Promise.reject(new Error('no clipboard: this page is not a secure context (https or localhost)'));
    wrote.then(() => { b.textContent = 'copied'; b.title = ''; },
      (e) => { b.textContent = 'copy failed'; b.title = `${(e && e.message) || e} — select the text to copy it`; })
      .then(() => setTimeout(() => { b.textContent = 'copy'; }, 2000));
  });
  return b;
}

function cnField(field, label, placeholder) {
  const wrap = cnEl('label', 'cn-field');
  const input = cnEl('input', 'cn-input');
  input.type = 'text';
  input.dataset.field = field;
  input.setAttribute('autocomplete', 'off');
  input.setAttribute('spellcheck', 'false');
  if (placeholder) input.setAttribute('placeholder', placeholder);
  if (field === 'port') input.setAttribute('inputmode', 'numeric');
  input.addEventListener('input', () => {
    cn[field] = input.value.trim();
    cnSave(field === 'host' ? CONNECT.hostKey : CONNECT.portKey, cn[field]);
    connectRender();
  });
  wrap.append(cnEl('span', null, label), input);
  return { wrap, input };
}

/** The dialog's fixed parts, once: what a render rewrites is text, pressed states and the snippet panel — never an
 *  input being typed into or a button being pressed. */
function connectBuild() {
  const dlg = $('connect');
  const box = cnEl('div', 'cn-box');
  const head = cnEl('div', 'cn-head');
  const title = cnEl('h2', 'cn-title', 'Connect an agent');
  title.id = 'connect-title';
  const x = cnEl('button', 'cn-x', '×');
  x.type = 'button';
  x.setAttribute('aria-label', 'Close');
  x.addEventListener('click', closeConnect);
  head.append(title, x);

  const facts = cnEl('dl', 'cn-facts');
  const model = cnEl('dd', 'cn-model');
  const door = cnEl('dd', 'cn-door');
  const route = cnEl('dd', 'cn-route');
  facts.append(cnEl('dt', null, 'model'), model, cnEl('dt', null, 'door'), door, cnEl('dt', null, 'route'), route);
  const routes = cnEl('div', 'cn-routes');
  routes.setAttribute('role', 'group');
  routes.setAttribute('aria-label', 'Route');
  const note = cnEl('p', 'cn-note');
  const ssh = cnEl('div', 'cn-ssh');
  const host = cnField('host', 'ssh host', '<server>');
  const port = cnField('port', 'local port', '');
  const fields = cnEl('div', 'cn-fields');
  fields.append(host.wrap, port.wrap);
  const sshLine = cnEl('div', 'cn-snip');
  const sshCmd = cnEl('pre', 'cn-pre');
  sshCmd.dataset.ssh = '';
  const sshHead = cnEl('div', 'cn-snip-head');
  sshHead.append(cnEl('span', null, 'ssh'), cnCopyButton(() => sshCmd.textContent, 'Copy the ssh command'));
  sshLine.append(sshHead, sshCmd);
  ssh.append(fields, sshLine);
  route.append(routes, note, ssh);
  const keysDt = cnEl('dt', null, 'keys');
  const keys = keysBuild();
  facts.append(keysDt, keys.dd);

  const tablist = cnEl('div', 'cn-tabs');
  tablist.setAttribute('role', 'tablist');
  tablist.setAttribute('aria-label', 'Client');
  const tabs = CONNECT_TABS.map((t, i) => {
    const b = cnEl('button', 'cn-tab', t.label);
    b.type = 'button';
    b.id = `connect-tab-${t.key}`;
    b.setAttribute('role', 'tab');
    b.setAttribute('aria-controls', 'connect-panel');
    b.dataset.tab = t.key;
    b.addEventListener('click', () => { cn.tab = t.key; connectRender(); });
    b.addEventListener('keydown', (e) => {
      const to = { ArrowRight: i + 1, ArrowLeft: i - 1, Home: 0, End: CONNECT_TABS.length - 1 }[e.key];
      if (to === undefined) return;
      e.preventDefault();
      const next = (to + CONNECT_TABS.length) % CONNECT_TABS.length;
      cn.tab = CONNECT_TABS[next].key;
      connectRender();
      tabs[next].focus();
    });
    tablist.appendChild(b);
    return b;
  });
  const panel = cnEl('div', 'cn-panel');
  panel.id = 'connect-panel';
  panel.setAttribute('role', 'tabpanel');
  box.append(head, facts, tablist, panel);
  dlg.appendChild(box);
  dlg.addEventListener('keydown', connectKeydown);
  dlg.addEventListener('cancel', (e) => { e.preventDefault(); closeConnect(); });
  // The backdrop closes — a press that STARTED on it: a drag that selects a snippet and ends past the box is a click
  // on the dialog too, and must not close it.
  dlg.addEventListener('pointerdown', (e) => { cn.downOnBackdrop = e.target === dlg; });
  dlg.addEventListener('click', (e) => { if (e.target === dlg && cn.downOnBackdrop) closeConnect(); });
  cn.parts = { dlg, x, model, door, routes, note, ssh, host: host.input, port: port.input, sshCmd, tablist, tabs, panel,
    keysDt, ...keys };
}

/** The key section's fixed parts: the form (name, lifetime, create), the shown-once box, the list, a status line. */
function keysBuild() {
  const dd = cnEl('dd', 'cn-keys');
  dd.dataset.keys = '';
  const form = cnEl('div', 'cn-key-form');
  const keyName = cnEl('input', 'cn-input cn-key-name-in');
  keyName.type = 'text';
  keyName.dataset.keyField = 'name';
  keyName.setAttribute('aria-label', 'Key name');
  keyName.setAttribute('placeholder', 'name, e.g. laptop');
  keyName.setAttribute('maxlength', '64');
  keyName.setAttribute('autocomplete', 'off');
  keyName.setAttribute('spellcheck', 'false');
  keyName.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); keysCreate(); } });
  keyName.addEventListener('input', () => { keyName.removeAttribute('aria-invalid'); });
  const keyDays = cnEl('select', 'cn-input cn-key-days');
  keyDays.dataset.keyField = 'days';
  keyDays.setAttribute('aria-label', 'Expires after');
  const keyCreate = cnEl('button', 'cn-route-btn cn-key-create', 'Create API key');
  keyCreate.type = 'button';
  keyCreate.dataset.keyCreate = '';
  keyCreate.addEventListener('click', () => { keysCreate(); });
  form.append(keyName, keyDays, keyCreate);
  const keyNew = cnEl('div', 'cn-snip cn-key-new');
  const keyNewHead = cnEl('div', 'cn-snip-head');
  const keyNewWhat = cnEl('span');
  const keyNewPre = cnEl('pre', 'cn-pre');
  keyNewPre.dataset.newKey = '';
  const keyNewCopy = cnCopyButton(() => (cn.newKey ? cn.newKey.key : ''), 'Copy the new API key');
  keyNewHead.append(keyNewWhat, keyNewCopy);
  // The one-time warning (UX audit 2026-10-15): an alert a screen reader announces, set apart in colour and
  // weight, saying what is left once the dialog closes — the prefix + first 4, nothing more.
  const keyOnce = cnEl('p', 'cn-key-once');
  keyOnce.dataset.keyOnce = '';
  keyOnce.setAttribute('role', 'alert');
  keyNew.append(keyNewHead, keyNewPre, keyOnce);
  const keyMsg = cnEl('p', 'cn-key-msg');
  keyMsg.dataset.keyMsg = '';
  keyMsg.setAttribute('role', 'status');
  const keyListEl = cnEl('ul', 'cn-key-list');
  keyListEl.setAttribute('aria-label', 'Your API keys');
  const keyFacts = cnEl('p', 'cn-key-facts');
  dd.append(form, keyNew, keyMsg, keyListEl, keyFacts);
  return { keysDd: dd, dd, keyName, keyDays, keyCreate, keyNew, keyNewWhat, keyNewPre, keyNewCopy, keyOnce, keyMsg,
    keyListEl, keyFacts };
}

/** Esc closes; Tab and Shift+Tab stay inside (the modal makes the page inert, not the browser's own chrome). */
function connectKeydown(e) {
  if (e.key === 'Escape') { e.preventDefault(); closeConnect(); return; }
  if (e.key !== 'Tab') return;
  const dlg = cn.parts.dlg;
  const list = [...dlg.querySelectorAll(CONNECT.focusable)]
    .filter((n) => !n.disabled && !n.closest('[hidden]') && n.getAttribute('tabindex') !== '-1');
  if (!list.length) return;
  const first = list[0], last = list[list.length - 1], at = document.activeElement;
  if (e.shiftKey && (at === first || !dlg.contains(at))) { e.preventDefault(); last.focus(); }
  else if (!e.shiftKey && (at === last || !dlg.contains(at))) { e.preventDefault(); first.focus(); }
}

async function connectAsk(url) {
  try {
    const r = await fetch(url, { cache: 'no-store', credentials: 'same-origin' });
    let body = null;
    try { body = await r.json(); } catch (_) { body = null; }
    return { status: r.status, body };
  } catch (e) {
    return { status: 0, body: null, error: e.message };
  }
}

/** The model a client names, and its window: the first /v1/models entry, as fullclaudevllm reads it — or why not. */
function connectModel(a) {
  if (a.status !== 200) return { error: `/v1/models: ${a.status ? `HTTP ${a.status}` : `no answer (${a.error})`}` };
  const m = ((a.body || {}).data || [])[0];
  if (!m || typeof m.id !== 'string') return { error: '/v1/models lists no model' };
  if (!Number.isInteger(m.max_model_len) || m.max_model_len <= 0) {
    return { error: `${m.id}: /v1/models gives no numeric max_model_len` };
  }
  return { model: { id: m.id, max_model_len: m.max_model_len } };
}

/** Open the dialog. `server` (gateway.js `serverOf`: a server's name, head url and served model) scopes it to that
 *  server: its model is the snippet's, so a client naming it routes to that server exactly (router.py: an exact name
 *  goes to the server that serves it, never rewritten); without it, the LB's default route — the model /v1/models
 *  lists first, what a request naming none is given. */
async function openConnect(server) {
  if (!cn.parts) connectBuild();
  const p = cn.parts;
  if (!p.dlg.open) p.dlg.showModal();
  p.x.focus();
  cn.server = server || null;
  cn.door = null; cn.model = null; cn.modelError = server ? '' : 'reading /v1/models…';
  p.routes.replaceChildren();
  connectRender();
  const [models, door] = await Promise.all([server ? null : connectAsk('/v1/models'), connectAsk('/auth/session')]);
  if (!lastStatus) await poll();
  cn.connect = (lastStatus && lastStatus.connect) || null;
  const m = server ? { model: server.model } : connectModel(models);
  cn.model = m.model || null;
  cn.modelError = m.error || '';
  cn.door = connectDoor(door.status, door.body);
  cn.doorStatus = door.status;
  const offered = cn.door === 'signin' && door.status === 200 && door.body ? door.body.api_keys : null;
  cn.keys = offered && Array.isArray(offered.days) && offered.days.length ? offered : null;
  cn.keyList = null; cn.keyError = ''; cn.keyNotice = ''; cn.keyArmed = null;
  if (cn.keys) keysOffer(cn.keys);
  cn.host = cnLoad(CONNECT.hostKey) || '';
  cn.port = cnLoad(CONNECT.portKey) || cnPortOf((cn.connect || {}).tunnel);
  p.host.value = cn.host;
  p.port.value = cn.port;
  cn.routes = connectRoutes(cn.door, location.origin, cn.connect, cn);
  if (!cn.routes.some((r) => r.key === cn.route)) cn.route = cn.routes.length ? cn.routes[0].key : '';
  p.routes.replaceChildren(...cn.routes.map((r) => {
    const b = cnEl('button', 'cn-route-btn', r.label);
    b.type = 'button';
    b.dataset.route = r.key;
    b.addEventListener('click', () => { cn.route = r.key; connectRender(); });
    return b;
  }));
  connectRender();
  if (cn.keys) keysLoad();
}

function closeConnect() {
  const p = cn.parts;
  // The key is shown once: closing the dialog forgets it — the gate keeps only its hash.
  cn.newKey = null; cn.keyArmed = null; cn.keyNotice = '';
  if (p) { p.keyNewPre.textContent = ''; p.keyOnce.textContent = ''; p.keyNew.hidden = true; }
  if (p && p.dlg.open) p.dlg.close();
  const back = $('cluster-connect');
  if (back) back.focus();
}

/** One door's line: the address, what it is, and what an API client gets there. */
function connectDoorLine(door, origin, connect) {
  const where = cnEl('b', null, origin);
  const said = (text, cls, title) => { const s = cnEl('span', cls, text); if (title) s.title = title; return s; };
  if (door === null) return [where, said(' · reading /auth/session…')];
  if (door === 'lb') return [where, said(' · the LB itself · no credential', 'ok')];
  if (door === 'open') {
    return [where, said(' · gate, no sign-in · a POST needs '), cnEl('code', null, `Origin: ${origin}`),
      said(' → API clients 403', 'bad', 'vllm-gate (VLLM_GATE_AUTH=false) passes a write only from its own origin; '
        + 'Claude Code, curl and the SDKs send no Origin')];
  }
  if (door === 'signin' && cn.keys) {
    return [where, said(' · sign-in gate · browser session · '),
      said('API key on GET /v1/models, POST /v1/*', 'ok', 'vllm-gate checks a per-user key (Authorization: Bearer or '
        + 'x-api-key) on these paths only; nginx drops it before the LB'),
    ...((connect || {}).basic ? [said(' · HTTP Basic on GET /v1/models, POST /v1/*', 'ok')] : [])];
  }
  if (door === 'signin') {
    return [where, said(' · sign-in gate · browser session only → API clients 401', 'bad',
      'vllm-gate reads only its session cookie; nginx drops Authorization before it asks; the issuer mints no '
      + 'token without a browser'),
    ...((connect || {}).basic ? [said(' · HTTP Basic on GET /v1/models, POST /v1/*', 'ok')] : [])];
  }
  return [where, said(` · /auth/session: ${cn.doorStatus ? `HTTP ${cn.doorStatus}` : 'no answer'} · door unknown`, 'bad')];
}

function connectRender() {
  const p = cn.parts;
  if (!p) return;
  const origin = location.origin;
  p.model.textContent = cn.model ? `${cn.model.id} · ${fmtInt(cn.model.max_model_len)} tokens`
    + (cn.server ? ` · server ${cn.server.name} · ${cn.server.url}` : '') : cn.modelError;
  p.model.classList.toggle('bad', !cn.model && cn.door !== null);
  p.door.replaceChildren(...connectDoorLine(cn.door, origin, cn.connect));

  const routes = cn.door === null ? [] : connectRoutes(cn.door, origin, cn.connect, cn);
  const route = routes.find((r) => r.key === cn.route) || null;
  [...p.routes.children].forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.route === cn.route)));
  const portOk = cnPortOk(cn.port);
  p.port.setAttribute('aria-invalid', String(!!route && route.key === 'ssh' && !portOk));
  let note = '';
  if (cn.door !== null && !cn.connect) note = `/lb/status unread${lastError ? `: ${lastError}` : ''}`;
  else if (cn.door !== null && !routes.length) note = 'no API route declared · LB_CONNECT_TUNNEL unset';
  else if (route && route.key === 'ssh' && !portOk) note = 'local port: 1–65535';
  p.note.textContent = note;
  p.note.hidden = !note;
  p.ssh.hidden = !route || route.key !== 'ssh';
  if (route && route.ssh) p.sshCmd.textContent = route.ssh;
  keysRender(!!route && route.key === 'apikey');

  const ready = !!(route && cn.model && cn.connect && (route.key !== 'ssh' || portOk));
  p.tablist.hidden = !ready;
  p.panel.hidden = !ready;
  p.tabs.forEach((t) => {
    const on = t.dataset.tab === cn.tab;
    t.setAttribute('aria-selected', String(on));
    t.setAttribute('tabindex', on ? '0' : '-1');
  });
  p.panel.setAttribute('aria-labelledby', `connect-tab-${cn.tab}`);
  if (!ready) { p.panel.replaceChildren(); return; }
  const blocks = connectSnippets(route, cn.model, cn.connect)[cn.tab];
  p.panel.replaceChildren(...blocks.map(({ title, text }) => {
    const block = cnEl('div', 'cn-snip');
    const bar = cnEl('div', 'cn-snip-head');
    const pre = cnEl('pre', 'cn-pre');
    pre.dataset.snippet = cn.tab;
    pre.appendChild(cnEl('code', null, text));
    const client = CONNECT_TABS.find((t) => t.key === cn.tab).label;
    bar.append(cnEl('span', null, title), cnCopyButton(() => text, `Copy the ${client} snippet (${title})`));
    block.append(bar, pre);
    return block;
  }));
}

// ── per-user API keys (the gate's /auth/keys) ──

/** The gate's lifetimes into the select, its default preselected — once per open, so a pick is never undone. */
function keysOffer(offered) {
  const p = cn.parts;
  p.keyDays.replaceChildren(...offered.days.map((d) => {
    const o = cnEl('option', null, `${d} days`);
    o.value = String(d);
    return o;
  }));
  p.keyDays.value = String(offered.default_days);
  p.keyFacts.textContent = `a key opens GET /v1/models · POST /v1/* · it stops working after ${offered.owner_days} d `
    + 'without a sign-in here';
}

async function keysAsk(url, init) {
  try {
    const r = await fetch(url, { cache: 'no-store', credentials: 'same-origin', ...(init || {}) });
    let body = null;
    try { body = await r.json(); } catch (_) { body = null; }
    return { status: r.status, body };
  } catch (e) {
    return { status: 0, body: null, error: e.message };
  }
}

const keysPost = (url, data) => keysAsk(url, { method: 'POST', headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(data) });

/** A refusal in the gate's words: its code, and its detail when it gave one. */
function keysRefusal(a) {
  if (!a.status) return `no answer (${a.error})`;
  const b = a.body || {};
  if (a.status === 401) return 'signed out: sign in again to manage keys';
  return `${b.error || `HTTP ${a.status}`}${b.detail ? `: ${b.detail}` : ''}`;
}

async function keysLoad() {
  const a = await keysAsk('/auth/keys');
  if (a.status === 200 && a.body && Array.isArray(a.body.keys)) {
    cn.keyList = a.body.keys;
    cn.keyError = '';
  } else {
    cn.keyList = null;
    cn.keyError = `/auth/keys: ${keysRefusal(a)}`;
  }
  keysRenderList();
  connectRender();
}

async function keysCreate() {
  const p = cn.parts;
  if (p.keyCreate.disabled) return;
  p.keyCreate.disabled = true;
  const a = await keysPost('/auth/keys', { name: p.keyName.value.trim(), days: Number(p.keyDays.value) });
  p.keyCreate.disabled = false;
  if (a.status === 201 && a.body && typeof a.body.key === 'string') {
    cn.newKey = { key: a.body.key, id: a.body.id, name: a.body.name, hint: a.body.hint };
    cn.keyError = ''; cn.keyNotice = '';
    p.keyNewCopy.textContent = 'copy';    // a "copied" from the previous key is not this one's
    p.keyName.value = '';
    await keysLoad();
    p.keyNewCopy.focus();                 // the one thing to do next: copy it
    return;
  }
  // The refused name is marked and given back to be fixed (the button under the pointer was disabled, so the focus
  // had nowhere to be); the list is re-read — what the gate refused for (its count of live keys) is then in view —
  // and the refusal is said after that read, which would otherwise clear it.
  if (a.body && a.body.error === 'bad_name') p.keyName.setAttribute('aria-invalid', 'true');
  await keysLoad();
  cn.keyError = `create: ${keysRefusal(a)}`;
  cn.keyNotice = '';
  connectRender();
  p.keyName.focus();
}

/** Revoke on the SECOND press: the first arms the row's button for a few seconds and says so — on the button, in
 *  its label and on the status line (which key, what to press); the second is confirmed there by name and hint. */
async function keysRevoke(k, button) {
  const id = k.id;
  const disarm = () => {
    button.textContent = 'revoke';
    button.classList.remove('armed');
    button.setAttribute('aria-label', keysRevokeLabel(k));
  };
  if (cn.keyArmed !== id) {
    cn.keyArmed = id;
    button.textContent = 'confirm revoke';
    button.classList.add('armed');
    button.setAttribute('aria-label', `Confirm: revoke API key ${k.name} (${k.hint}) — press again`);
    cn.keyError = ''; cn.keyNotice = `revoke ${k.name} (${k.hint})? press confirm revoke again within ${CONNECT.keyArmMs / 1000} s`;
    connectRender();
    setTimeout(() => {
      if (cn.keyArmed !== id) return;
      cn.keyArmed = null;
      disarm();
      cn.keyNotice = '';
      connectRender();
    }, CONNECT.keyArmMs);
    return;
  }
  cn.keyArmed = null;
  button.disabled = true;
  const a = await keysPost('/auth/keys/revoke', { id });
  if (a.status === 200) {
    if (cn.newKey && cn.newKey.id === id) cn.newKey = null;
    cn.keyError = ''; cn.keyNotice = `revoked ${k.name} (${k.hint})`;
    await keysLoad();
    cn.parts.keyName.focus();             // the pressed row is gone: focus stays in the dialog
    return;
  }
  button.disabled = false;
  disarm();
  cn.keyNotice = '';
  cn.keyError = `revoke: ${keysRefusal(a)}`;
  connectRender();
}
const keysRevokeLabel = (k) => `Revoke API key ${k.name} (${k.hint})`;

const cnDay = (s) => new Date(s * 1000).toISOString().slice(0, 10);
function cnAgo(s) {
  const d = Math.max(0, Math.floor(Date.now() / 1000) - s);
  if (d < 60) return 'just now';
  if (d < 3600) return `${Math.floor(d / 60)} min ago`;
  if (d < 86400) return `${Math.floor(d / 3600)} h ago`;
  return `${Math.floor(d / 86400)} d ago`;
}

/** The person's keys, newest first — rebuilt only when the list was read, never under a pressed button. */
function keysRenderList() {
  const p = cn.parts;
  if (!p) return;
  if (!cn.keyList) { p.keyListEl.replaceChildren(); return; }
  if (!cn.keyList.length) { p.keyListEl.replaceChildren(cnEl('li', 'cn-key-none', 'no keys yet')); return; }
  p.keyListEl.replaceChildren(...cn.keyList.map((k) => {
    const li = cnEl('li', `cn-key${k.state === 'expired' ? ' expired' : ''}`);
    li.dataset.keyId = k.id;
    const who = cnEl('span', 'cn-key-who');
    who.append(cnEl('span', 'cn-key-name', k.name), cnEl('code', 'cn-key-hint', k.hint));
    const when = cnEl('span', 'cn-key-when',
      `created ${cnDay(k.created_at)} · ${k.state === 'expired' ? 'expired' : 'expires'} ${cnDay(k.expires_at)}`
      + ` · used ${k.last_used_at ? cnAgo(k.last_used_at) : 'never'}`);
    const revoke = cnEl('button', 'cn-copy cn-key-revoke', 'revoke');
    revoke.type = 'button';
    revoke.dataset.revoke = k.id;
    revoke.setAttribute('aria-label', keysRevokeLabel(k));
    revoke.addEventListener('click', () => { keysRevoke(k, revoke); });
    li.append(who, when, revoke);
    return li;
  }));
}

/** The section's visibility, the shown-once key and the status line: text only, nothing rebuilt. */
function keysRender(onKeyRoute) {
  const p = cn.parts;
  const show = !!cn.keys && onKeyRoute;
  p.keysDt.hidden = !show;
  p.keysDd.hidden = !show;
  if (!show) return;
  p.keyNew.hidden = !cn.newKey;
  p.keyNewPre.textContent = cn.newKey ? cn.newKey.key : '';
  p.keyNewWhat.textContent = cn.newKey ? `new key · ${cn.newKey.name} · shown once: copy it now` : '';
  p.keyOnce.textContent = cn.newKey
    ? `Copy it now — it will never be shown again. Once this dialog closes, only ${cn.newKey.hint} stays listed.` : '';
  const msg = cn.keyError || cn.keyNotice || '';
  p.keyMsg.textContent = msg;
  p.keyMsg.hidden = !msg;
  p.keyMsg.classList.toggle('bad', !!cn.keyError);
}


if ($('cluster-panel')) {
  holdButton($('cluster-toggle'), 5000, toggleCluster);
  initModelPicker();                     // the combobox over #cluster-model, which the listeners below watch
  $('cluster-apply').addEventListener('click', applyClicked);
  $('cluster-defaults').addEventListener('click', useDefaults);
  $('cluster-confirm-go').addEventListener('click', () => sendSettings(true));
  $('cluster-confirm-cancel').addEventListener('click', closeConfirm);
  $('cluster-confirm').addEventListener('keydown', (e) => { if (e.key === 'Escape') closeConfirm(); });
  // A node the LB requests: its share of the card is an edit like the others (priced as it is picked), and a
  // node finishing what it serves can be told to cut it.
  if ($('cluster-util')) {
    const util = $('cluster-util');
    const markUtil = () => {
      util.dataset.dirty = '1';
      closeConfirm();
      if (lastStatus && theCluster(lastStatus)) renderClusterSettings(theCluster(lastStatus), lastStatus);
      askEstimate();
    };
    util.addEventListener('input', markUtil);
    util.addEventListener('change', markUtil);
  }
  if ($('cluster-force')) {
    $('cluster-force').addEventListener('click', () => {
      const c = (lastStatus && theCluster(lastStatus)) || {};
      if (localNode(c)) sendLocal((c.desired || {}).mode === 'cluster' ? 'apply' : 'stop', { force: true });
    });
  }
  // An edit is the operator's until it is sent or the cluster locks it; a poll must not undo it. Model,
  // concurrency, context and the PREFILL BUDGET also change what the selection would COST, so those ask
  // the LB to size it — the budget included, because it is the one that took box-a to 0.8 GiB free and
  // has to be answered while it can still be turned back; and any edit withdraws an open confirm, whose
  // words described the fields before it.
  CLUSTER_FIELDS.forEach(({ id }) => {
    const el = $(id);
    if (!el) return;
    const mark = () => {
      el.dataset.dirty = '1';
      facesOf(id).forEach((face) => face.removeAttribute('aria-invalid'));
      closeConfirm();
      // Drawn FIRST: a model change puts the new model's own pool and placement into the fields nobody
      // touched, and the estimate must be asked for THOSE — not for the last model's, still on screen.
      if (lastStatus && theCluster(lastStatus)) renderClusterSettings(theCluster(lastStatus), lastStatus);
      if (['cluster-seqs', 'cluster-context', 'cluster-model', 'cluster-batched', 'cluster-kv',
        'cluster-ple', 'cluster-dtype'].includes(id) || (id === 'cluster-offload' && localNode(theCluster(lastStatus) || {}))) {
        askEstimate();
      }
    };
    el.addEventListener('input', mark);
    el.addEventListener('change', mark);
    const custom = customOf(el);
    if (custom) {
      custom.addEventListener('input', mark);
      custom.addEventListener('change', mark);
      // choosing "custom…" moves the operator straight into the field
      // …by a pointer: arrowing through a row of boxes onto "custom…" opens the field but leaves focus in the row.
      el.addEventListener('change', () => {
        syncCustom(el);
        const seg = segOf(el);
        if (el.value === CUSTOM && (!seg || seg.dataset.pointer === '1')) custom.focus();
      });
    }
  });
}
if ($('swarm')) holdButton($('swarm-start'), 3000, swarmHeld);
// THE DURATION's list and its typed field. A change redraws the button straight away — its label names
// the minutes a hold would ask for, and three seconds is a poll away — and picking `custom…` moves into
// the field, which is the gesture the cluster's own knobs have (`syncCustom`).
['swarm-cap', 'swarm-cap-custom'].forEach((id) => {
  const el = $(id);
  if (!el) return;
  const again = () => {
    const q = (lastStatus && lastStatus.swarm) || {};
    renderSwarmCap(q, swarmLive(q) === true);
    renderSwarmControls(q, lastStatus && lastStatus.now);
  };
  el.addEventListener('input', again);
  el.addEventListener('change', again);
});
if ($('swarm-cap')) {
  $('swarm-cap').addEventListener('change', () => {
    const el = $('swarm-cap');
    syncCustom(el);
    const custom = customOf(el);
    if (el.value === CUSTOM && custom) custom.focus();
  });
}
// A STYLE SWITCH (styles.js) changes the tokens every computed colour reads (palette.js): the page is drawn again
// from the status it has, not three seconds later from the next poll.
document.addEventListener('fleet:style', () => { if (lastStatus) renderStatus(lastStatus); });
// A graph's reference label is placed in the graph's own pixels, so a new width places it again.
window.addEventListener('resize', () => placeMarks());
poll();
setInterval(poll, POLL_MS);
setInterval(renderFreshness, 1000);     // the age keeps counting while a poll hangs
