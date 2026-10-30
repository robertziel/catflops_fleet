/* /sessions — THE HISTORY OF LOADED MODELS. One row per START: a generation of the TP=2 pair, an intent a
 * one-host node's host applied (its id is the generation), or — for an engine the LB did not start — one engine
 * process, known by its own start (no generation). The model that was loaded, how long it stood, what it
 * generated, what it cost to start and what it drew — read from /lb/sessions, which folds the real-traffic log's
 * minutes (perflog.py `sessions`) into one record per session. Minutes no session can claim (recorded before a
 * one-host node's minutes named their start) are counted and said, never folded into a guess. Served BY the LB,
 * so every route is this process's own, read-only and same-origin.
 *
 * WHY THE FOLD IS SERVER-SIDE. /lb/perf hands the browser at most MAX_WINDOWS minutes; this page is the
 * part that must still be right in a year, so the arithmetic is Python's and the payload is one small
 * row per session.
 *
 * CHART RULES, the same ones /benchmarks keeps:
 *   * ONE measure per chart, ONE y-axis. Tokens and hours are two charts, never two scales on one.
 *   * IDENTITY IS NEVER HUE HERE. Four models need four categorical hues and this surface has three:
 *     scripts/validate_palette.js (dataviz) measures every fourth candidate against --cmp-3 at 1.9–6.3
 *     ΔE under protanopia, below its floor of 8. So the per-model chart is SMALL MULTIPLES — the facet's
 *     own title says which model — and every mark on the page is one hue.
 *   * A figure nobody measured is a GAP, never a zero: an idle minute is not recorded at all, a box that
 *     reported no power is not 0 W.
 *   * Every bar carries its own numbers in a <title>, because a bar's height is a comparison and the
 *     number behind it is the fact.
 */
'use strict';

const $ = (id) => document.getElementById(id);
const NS = 'http://www.w3.org/2000/svg';
//: The ranges, widest first — the history is the subject, so `all` is the default and reads first.
//: Each carries the SHORT word a phone shows and the full one a wide window and a screen reader get,
//: exactly as /benchmarks' own range picker does (benchmarks.html `.seg`).
const RANGES = [
  { key: 'all', short: 'All', label: 'everything recorded', days: 'all' },
  { key: '30', short: '30 d', label: 'last 30 days', days: '30' },
  { key: '7', short: '7 d', label: 'last 7 days', days: '7' },
  { key: '1', short: '24 h', label: 'last 24 hours', days: '1' },
];
let range = 'all';
let last = null;

/* ── numbers, in the words the operator reads them in ──────────────────────────────────── */

const isNum = (v) => typeof v === 'number' && Number.isFinite(v);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const int = (v) => (isNum(v) ? Math.round(v).toLocaleString('en-US') : '—');

/** A token count at the size it is read at: 20.5 M, 7.9 M, 174 k, 912. */
function tokens(v) {
  if (!isNum(v)) return '—';
  if (v >= 1e9) return `${(v / 1e9).toFixed(2)} G`;
  if (v >= 1e6) return `${(v / 1e6).toFixed(2)} M`;
  if (v >= 1e4) return `${Math.round(v / 1e3).toLocaleString('en-US')} k`;
  return int(v);
}
/** Seconds as the unit that fits: 63.1 h, 22 min, 45 s. */
function dur(v) {
  if (!isNum(v)) return '—';
  if (v >= 3600) return `${(v / 3600).toFixed(v >= 36000 ? 0 : 1)} h`;
  if (v >= 60) return `${Math.round(v / 60)} min`;
  return `${v < 10 ? v.toFixed(1) : Math.round(v)} s`;
}
/** Tokens per kWh at the size it is read at: 5.14 M · 912 k. A session with no power reading or no tokens
 *  is a dash — never 0, and never infinite efficiency. */
function perKwh(v) {
  return isNum(v) && v > 0 ? tokens(v) : '—';
}
function energy(v) {
  if (!isNum(v)) return '—';
  return v >= 1000 ? `${(v / 1000).toFixed(2)} kWh` : `${Math.round(v)} Wh`;
}
const rate = (v, places = 1) => (isNum(v) ? v.toFixed(places) : '—');
/** Dollars at the size they are read at: $1,716 · $207 · $12.34 · $0.44. A cost nobody could compute
 *  (no published price) is a dash — never $0, which would say it cost nothing. */
function usd(v) {
  if (!isNum(v)) return '—';
  if (v === 0) return '$0';          // priced, and nothing to price: not "$0.000", which reads as a measurement
  if (v >= 100) return `$${Math.round(v).toLocaleString('en-US')}`;
  if (v >= 1) return `$${v.toFixed(2)}`;
  return `$${v.toFixed(v >= 0.1 ? 2 : 3)}`;
}
const pct = (v) => (isNum(v) ? `${(v * 100).toFixed(1)}%` : '—');
/** A stamp the operator can line up with the fleet's own logs: UTC, to the minute. */
function when(t) {
  if (!isNum(t)) return '—';
  const d = new Date(t * 1000);
  return `${d.toISOString().slice(0, 10)} ${d.toISOString().slice(11, 16)}Z`;
}
const day = (t) => (isNum(t) ? new Date(t * 1000).toISOString().slice(5, 10) : '');
/** WHICH START a session (or `current`) is: its generation — or, for an engine the LB did not start, that engine
 *  process's own start, the figure its minutes carry (`engine_started_at`). */
const whose = (r) => (isNum(r.generation) ? `generation ${r.generation}` : `engine up since ${when(r.engine_started_at)}`);
/** The same session as `current` names it: one generation, or one engine process. */
const sameStart = (r, c) => r.model_choice === c.model_choice && (isNum(c.generation)
  ? r.generation === c.generation : isNum(c.engine_started_at) && r.engine_started_at === c.engine_started_at);
/** THE MINUTES NO SESSION CLAIMS, said with their count: those recorded before minutes named their start (a one-host
 *  node's, before 2026-10-12 — nothing recorded says which start they served), and any other. */
function unclaimed(d) {
  const before = isNum(d.windows_before_sessions) ? d.windows_before_sessions : 0;
  const other = (isNum(d.windows_without_a_session) ? d.windows_without_a_session : 0) - before;
  return [before > 0 ? `${int(before)} minutes from before sessions were recorded` : '',
    other > 0 ? `${int(other)} minutes in no session: no model, generation or engine start recorded` : '']
    .filter(Boolean);
}

/* ── the SVG a chart is ────────────────────────────────────────────────────────────────── */

function el(name, attrs, text) {
  const n = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs || {})) if (v != null) n.setAttribute(k, String(v));
  if (text != null) n.textContent = String(text);
  return n;
}
/** A mark's own numbers, as its native tooltip: a height is a comparison, the figure behind it is a fact. */
const titled = (node, text) => (node.append(el('title', {}, text)), node);

/** The y scale's top: the data's maximum, rounded up to a round number so the axis label is readable. */
function ceilNice(max) {
  if (!(max > 0)) return 1;
  const pow = 10 ** Math.floor(Math.log10(max));
  for (const step of [1, 1.5, 2, 2.5, 3, 4, 5, 7.5, 10]) {
    if (max <= step * pow) return step * pow;
  }
  return 10 * pow;
}

/* ── the page ──────────────────────────────────────────────────────────────────────────── */

function renderRange() {
  const box = $('s-range');
  if (box.dataset.built) return;
  box.dataset.built = '1';
  for (const r of RANGES) {
    const label = document.createElement('label');
    label.className = 'seg-opt';
    label.innerHTML = `<input type="radio" name="s-range" value="${esc(r.key)}" autocomplete="off"`
      + `${r.key === range ? ' checked' : ''}><span class="seg-face">`
      + `<span class="seg-short" aria-hidden="true">${esc(r.short)}</span>`
      + `<span class="seg-long">${esc(r.label)}</span></span>`;
    label.querySelector('input').addEventListener('change', () => { range = r.key; load(); });
    box.append(label);
  }
}

function tile(label, value, note, title) {
  const d = document.createElement('div');
  d.className = 's-tile';
  if (title) d.title = title;
  d.innerHTML = `<span class="s-tile-v">${esc(value)}</span><span class="s-tile-k">${esc(label)}</span>`
    + (note ? `<span class="s-tile-n">${esc(note)}</span>` : '');
  return d;
}

/** A price list as the tooltip that says where its numbers came from: a price with no date and no
 *  source is a guess wearing a number, so the page carries both wherever it shows a dollar. */
function listTitle(l) {
  if (!l) return '';
  const r = l.usd_per_mtok || {};
  return [`${l.label} · ${l.vendor}`,
    `$${r.input} in · $${r.output} out · $${r.cache_read} cache read, per million tokens`,
    `as of ${l.as_of} · ${l.source}`, l.basis, l.excludes ? `Not counted: ${l.excludes}` : '',
    l.tokenizer ? `Tokens: ${l.tokenizer}` : ''].filter(Boolean).join('\n\n');
}

/** What a total covers: "88 of 88" while every session was priced, the shortfall otherwise — a sum over
 *  part of the history must never read as the whole. */
const coverage = (c) => (c ? `${int(c.priced)} of ${int(c.of)} sessions` : '');

/** THE SESSION RUNNING NOW, first on the page (operator, 2026-10-06: "on top show current session
 *  running model costs and Opus, below total all models as is shown now"). `d.current` is the LAUNCH
 *  record of the pair that serves — the same model id + generation the log stamps its minutes with —
 *  so a tile can only ever price the generation actually running, never the model someone last wished
 *  for. The dollars are that ONE session's, every figure below stays the whole history. A pair that
 *  has no recorded minute inside the chosen range says so instead of a $0; no pair serving at all,
 *  and the server says `current: null`, so no tile is drawn — never a stand-in. */
function nowTiles(d) {
  const c = d.current;
  if (!c) return [];
  const pl = d.price_lists || {};
  const row = (d.sessions || []).find((r) => sameStart(r, c));
  const note = (isNum(c.generation) ? `gen ${int(c.generation)}` : whose(c))
    + (row && isNum(row.active_s) ? ` · ${dur(row.active_s)} busy` : '');
  const why = 'this pair has no recorded minute in this range yet';   // d.current means a pair IS serving
  const tokens = row ? `Tokens so far: ${int(row.gen_tokens)} generated · ${int(row.prompt_tokens)} prompt`
    + ` · ${int(row.requests)} requests` : '';
  const out = [];
  const ownId = row && (row.own_price || {}).list;
  const ownList = ownId ? (pl.lists || {})[ownId] : null;
  out.push(tile(`now · ${ownList ? ownList.label : 'own vendor'}`,
    usd(row && ownId && row.cost ? row.cost[ownId] : null), note,
    row ? (ownList ? listTitle(ownList) + '\n\n' + tokens : `no published price: ${(row.own_price || {}).why || ''}`)
      : why));
  for (const id of pl.references || []) {
    const l = (pl.lists || {})[id];
    out.push(tile(`now · vs ${l.label}`, usd(row && row.cost ? row.cost[id] : null), note,
      row ? listTitle(l) + '\n\n' + tokens : why));
  }
  return out;
}

/** HOW MUCH WAS SAVED, against each comparison, side by side (pricing.py): what the same model costs from
 *  its own vendor, and what a model you would otherwise have used costs through its API. Each says how
 *  many sessions it covers — Flash-Next is not on Alibaba's list, so the own-vendor total is smaller than
 *  the history and the tile says so instead of presenting a part as the whole. A server that publishes
 *  no costs (an LB older than this) draws no tile. */
function savedTiles(d) {
  const cost = (d.totals || {}).cost;
  const pl = d.price_lists || {};
  if (!cost || !pl.lists) return [];
  const out = [];
  const own = cost.own;
  const ownList = pl.lists[(pl.own || [])[0]];
  if (own && own.of) {
    out.push(tile(ownList ? `saved vs ${ownList.label} list` : 'saved vs own vendor', usd(own.usd),
      coverage(own), own.usd == null ? 'no session has a published price from its own vendor'
        : `${listTitle(ownList)}\n\nUnpriced: ${Object.entries(pl.unpriced || {}).map(
          ([m, why]) => `${m} — ${why}`).join('\n') || 'none'}`));
  }
  for (const id of pl.references || []) {
    const l = pl.lists[id];
    out.push(tile(`saved vs ${l.label}`, usd((cost[id] || {}).usd), coverage(cost[id]), listTitle(l)));
  }
  return out;
}

/** The prices the dollars above were made from, with their dates — printed, not tucked in a tooltip,
 *  because a price list ages and the reader has to see how old this one is. */
function renderPrices(d) {
  const pl = d.price_lists || {};
  const box = $('s-prices');
  const used = [...(pl.own || []), ...(pl.references || [])].map((id) => pl.lists && pl.lists[id]).filter(Boolean);
  box.hidden = !used.length;
  if (!used.length) return;
  box.innerHTML = `<span class="s-prices-k">prices per M tokens · in / out / cache read</span>` + used.map((l) => {
    const r = l.usd_per_mtok;
    return `<span class="s-price" title="${esc(listTitle(l))}"><b>${esc(l.label)}</b> `
      + `${esc(`$${r.input} / $${r.output} / $${r.cache_read}`)} <i>${esc(l.as_of)}</i></span>`;
  }).join('');
}

/** THE FLEET'S EFFICIENCY: generated tokens per kWh over the whole history — a ratio of sums, taken over
 *  the sessions where both the tokens and the power were measured, and the tile says how many that was. The
 *  power is what each box's nvidia-smi reported for its GPU while the session was busy: it leaves out idle
 *  time and whatever else the box draws, so this is the efficiency of the work, not of the wall socket. */
function efficiencyTile(t) {
  const e = (t || {}).efficiency;
  if (!e) return tile('tokens per kWh', '—', '', 'this server does not publish an efficiency figure');
  return tile('tokens per kWh', perKwh(e.tok_per_kwh), `${int(e.measured)} of ${int(e.of)} sessions`,
    'Generated tokens over the energy the two boxes\' GPUs drew while serving (nvidia-smi power × busy '
    + 'seconds). Idle time and the rest of the box are not counted.\n\nA ratio of sums over the sessions '
    + 'where both were measured.');
}

function renderTiles(d) {
  const t = d.totals || {};
  const box = $('s-tiles');
  box.replaceChildren(
    ...nowTiles(d),
    tile('tokens generated', tokens(t.gen_tokens), `${int(t.requests)} requests`),
    tile('time serving', dur(t.active_s), 'busy minutes only'),
    tile('mean rate', `${rate(t.gen_tok_s)} tok/s`, 'tokens ÷ busy seconds'),
    tile('sessions', int(t.sessions), `${int(t.models)} models`),
    tile('prompt tokens', tokens(t.prompt_tokens), 'billed, cache hits included'),
    tile('energy', energy(t.wh), 'both boxes, while busy'),
    efficiencyTile(t),
    ...savedTiles(d));
  box.hidden = false;
  $('s-span').textContent = isNum(t.first_at)
    ? `${when(t.first_at)} → ${when(t.last_at)}` : '';
}

/** One model's strip: its sessions in time order, bar height = tokens generated, ON ITS OWN SCALE.
 *
 *  It was ONE scale across the strips, so two models could be compared by bar height. On the real log
 *  that made every strip unreadable: a single 7.9 M session set the ceiling and the other 87 sessions
 *  drew as identical 1 px slivers, their colour with them. A small multiple whose marks all collapse
 *  to the axis compares nothing. So each strip now has its own ceiling, PRINTED on it (`tokens(top)`,
 *  the number the label carries), and the cross-model comparison is where it was always readable: the
 *  totals beside each model's name, and the table. */
/** What a model's sessions would have cost on each comparison — " · $818 vs opus.5.5" — where there is a
 *  price. An unpriced comparison is left out of the line, not drawn as $0. */
function savedOf(model) {
  const eff = (model.efficiency || {}).tok_per_kwh;
  const kwh = isNum(eff) && eff > 0 ? ` &middot; ${esc(tokens(eff))} tok/kWh` : '';
  const c = model.cost;
  if (!c) return kwh;
  const bits = Object.entries(c).filter(([k, v]) => k !== 'own' && v && isNum(v.usd))
    .map(([k, v]) => ` &middot; ${esc(usd(v.usd))} vs ${esc(k.replace(/^anthropic-claude-/, '').replace(/-/g, ' '))}`);
  if (c.own && isNum(c.own.usd)) bits.unshift(` &middot; ${esc(usd(c.own.usd))} vs own list`);
  return kwh + bits.join('');
}

/** THE RATES a model's dollars were made at, printed on the strip that shows them. The saved figures
 *  are two different currencies side by side — qwen3.8 at $0.50/$3/$0.10, opus at $4/$20/$0.20 — so a
 *  "$818 vs own list" with no rate beside it reads as a comparison when it is two price lists being
 *  compared. The chip is the model's OWN list, read from the rows: pricing.py keys `own_price` by
 *  model_choice, so every row of a model carries the same list id and the first is enough. A model
 *  with no list of its own prints NOTHING — no $0, and never the reference list's numbers standing in
 *  for a list it does not have; that is the substitution the whole price rule refuses. */
function ratesOf(rows, pl) {
  const id = ((rows[0] || {}).own_price || {}).list;
  const l = id && pl && pl.lists && pl.lists[id];
  if (!l) return '';
  const r = l.usd_per_mtok;
  return ` <span class="s-facet-rates" title="${esc(listTitle(l))}">$${r.input.toFixed(2)} fill`
    + ` &middot; $${r.output.toFixed(2)} decode &middot; $${r.cache_read.toFixed(2)} cache</span>`;
}

function facet(model, rows, chip) {
  const top = ceilNice(Math.max(0, ...rows.map((x) => x.gen_tokens || 0)));
  // The bar's height is HOW MUCH it generated; its colour is HOW FAST it was among this model's own
  // sessions — the same four-tier ramp the table uses, so one colour means one thing on this page.
  const rates = rows.map((x) => x.gen_tok_s).filter(isNum);
  const wrap = document.createElement('div');
  wrap.className = 's-facet';
  const tokensTotal = rows.reduce((a, r) => a + (r.gen_tokens || 0), 0);
  const busy = rows.reduce((a, r) => a + (r.active_s || 0), 0);
  wrap.innerHTML = `<div class="s-facet-head">
      <span class="s-facet-name">${esc(model.label || model.model_choice)}</span>
      <span class="s-facet-sum">${esc(tokens(tokensTotal))} tok &middot; ${esc(dur(busy))} &middot; `
    + `${esc(rate(model.gen_tok_s))} tok/s &middot; ${rows.length} session${rows.length === 1 ? '' : 's'}`
    + `${model.engine ? ` &middot; ${esc(model.engine)}` : ''}${savedOf(model)}</span>`
    // the rates this model's own vendor bills, beside the dollars they produced — the date and
    // source ride in the tooltip, exactly as they do on the printed price strip and the tiles
    + `${chip || ''}</div>`;
  const W = 760, H = 86, PAD = { t: 6, r: 8, b: 16, l: 46 };
  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, class: 's-svg', role: 'img',
    'aria-label': `${model.label || model.model_choice}: tokens generated per session` });
  const plotW = W - PAD.l - PAD.r, plotH = H - PAD.t - PAD.b;
  // the shared scale's own gridline and its number — the only y label a strip this short can carry
  svg.append(el('line', { x1: PAD.l, x2: W - PAD.r, y1: PAD.t, y2: PAD.t, class: 's-grid' }));
  svg.append(el('text', { x: PAD.l - 6, y: PAD.t + 4, class: 's-ylab', 'text-anchor': 'end' }, tokens(top)));
  svg.append(el('line', { x1: PAD.l, x2: W - PAD.r, y1: PAD.t + plotH, y2: PAD.t + plotH, class: 's-axis' }));
  // 2px of surface between neighbours, and a bar never thinner than 3px — a mark too thin to see is a gap
  const slot = plotW / Math.max(rows.length, 1);
  const bw = Math.max(3, Math.min(26, slot - 2));
  rows.forEach((r, i) => {
    const h = top > 0 ? Math.max(r.gen_tokens > 0 ? 1.5 : 0, (r.gen_tokens || 0) / top * plotH) : 0;
    const x = PAD.l + i * slot + (slot - bw) / 2;
    const t = tier(r.gen_tok_s, rates, 1);
    const bar = el('rect', { x, y: PAD.t + plotH - h, width: bw, height: h, rx: Math.min(3, bw / 2),
      class: 's-bar' + ((r.gen_tokens || 0) > 0 ? '' : ' s-bar-none') + (t ? ` s-t${t}` : '') });
    titled(bar, `${whose(r)} · ${when(r.started_at)}\n${int(r.gen_tokens)} tokens`
      + ` · ${rate(r.gen_tok_s)} tok/s · ${dur(r.active_s)} busy · ${int(r.requests)} requests`
      + `\n${r.config && r.config.max_num_seqs ? `${r.config.max_num_seqs} seats · ` : ''}`
      + `${r.config && r.config.max_num_batched_tokens ? `budget ${int(r.config.max_num_batched_tokens)}` : ''}`);
    svg.append(bar);
  });
  // the first and last session's day, so the strip has a time direction
  if (rows.length) {
    svg.append(el('text', { x: PAD.l, y: H - 4, class: 's-xlab' }, day(rows[0].started_at)));
    svg.append(el('text', { x: W - PAD.r, y: H - 4, class: 's-xlab', 'text-anchor': 'end' },
      day(rows[rows.length - 1].ended_at || rows[rows.length - 1].started_at)));
  }
  wrap.append(svg);
  return wrap;
}

function renderFacets(d) {
  const rows = (d.sessions || []).slice().reverse();          // oldest first: a history reads left to right
  const byModel = new Map();
  for (const r of rows) {
    if (!byModel.has(r.model_choice)) byModel.set(r.model_choice, []);
    byModel.get(r.model_choice).push(r);
  }
  const box = $('s-facets');
  box.replaceChildren();
  // The models in the order the ROLLUP ranks them (most tokens first) — a facet's place is stable while
  // the data is, and its identity is its title, never its position.
  for (const m of d.by_model || []) {
    const mine = byModel.get(m.model_choice) || [];
    if (mine.length) box.append(facet(m, mine, ratesOf(mine, d.price_lists)));
  }
  $('s-models-sub').textContent = 'bar = one session · height: tokens generated, each strip to its '
    + 'own ceiling · colour: its rate among this model\'s own sessions';
  $('s-models').hidden = (d.by_model || []).length === 0;
}

/** Every token this fleet has generated, as it accumulated. ONE series, so one scale and nothing to confuse. */
function renderCumulative(d) {
  const rows = (d.sessions || []).slice().reverse().filter((r) => isNum(r.ended_at));
  const card = $('s-cumulative');
  if (rows.length < 2) { card.hidden = true; return; }
  const W = 980, H = 220, PAD = { t: 14, r: 14, b: 26, l: 62 };
  const plotW = W - PAD.l - PAD.r, plotH = H - PAD.t - PAD.b;
  let acc = 0;
  const pts = rows.map((r) => ({ t: r.ended_at, y: (acc += r.gen_tokens || 0), r }));
  const t0 = rows[0].started_at || pts[0].t, t1 = pts[pts.length - 1].t;
  const top = ceilNice(acc);
  const x = (t) => PAD.l + (t1 > t0 ? (t - t0) / (t1 - t0) : 0) * plotW;
  const y = (v) => PAD.t + plotH - (top > 0 ? v / top * plotH : 0);
  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, class: 's-svg s-svg-big', role: 'img',
    'aria-label': 'tokens generated, cumulative over time' });
  for (let i = 0; i <= 4; i++) {
    const v = top * i / 4;
    svg.append(el('line', { x1: PAD.l, x2: W - PAD.r, y1: y(v), y2: y(v), class: i ? 's-grid' : 's-axis' }));
    svg.append(el('text', { x: PAD.l - 8, y: y(v) + 4, class: 's-ylab', 'text-anchor': 'end' }, tokens(v)));
  }
  const line = pts.map((p, i) => `${i ? 'L' : 'M'}${x(p.t).toFixed(1)},${y(p.y).toFixed(1)}`).join(' ');
  svg.append(el('path', { d: `${line} L${x(t1).toFixed(1)},${y(0)} L${x(t0).toFixed(1)},${y(0)} Z`,
    class: 's-area' }));
  svg.append(el('path', { d: `M${x(t0).toFixed(1)},${y(0)} ${line}`, class: 's-line' }));
  for (const p of pts) {
    svg.append(titled(el('circle', { cx: x(p.t), cy: y(p.y), r: 4.5, class: 's-dot' }),
      `${whose(p.r)} · ${p.r.label || p.r.model_choice}\n`
      + `${int(p.r.gen_tokens)} this session · ${int(p.y)} in total · ${when(p.t)}`));
  }
  svg.append(el('text', { x: PAD.l, y: H - 6, class: 's-xlab' }, when(t0)));
  svg.append(el('text', { x: W - PAD.r, y: H - 6, class: 's-xlab', 'text-anchor': 'end' }, when(t1)));
  $('s-cum-chart').replaceChildren(svg);
  $('s-cum-sub').textContent = `${rows.length} sessions · ${tokens(acc)} tokens in total`;
  card.hidden = false;
}

/* ── colour: ONE RAMP, and it means "how good", exactly as it does on /benchmarks ────────
 *
 * The violet `--heat` ramp (benchmarks.css, four tiers, measured on these same row surfaces) is this
 * fleet's encoding for a value's standing among its peers. It is the only colour this page carries,
 * on the bars and in the table, and it answers one question: how good is this figure INSIDE ITS OWN
 * COMPARABLE GROUP. Two groups, declared per column, never inferred:
 *
 *   model  a RATE or a QUALITY — tok/s, acceptance, TTFT. Two models are different machines doing
 *          different work; washing them against each other would say SGLang's 87 tok/s is worse than
 *          Flash-Next's 97 when they are not the same measurement.
 *   all    a VOLUME — tokens, requests, hours. A token is a token whoever generated it.
 *
 * DIRECTION IS DECLARED (`better`), never inferred: half of these are better when smaller. A column
 * that is a SETTING (seats, budget, drafter depth) is never washed at all — a configuration is not
 * good or bad, it is what was asked for.
 */
const TIERS = 4;

/** Which tier `v` falls in among `peers` — 1 (weakest) to TIERS (best), or 0 when there is nothing
 *  to compare it against. Rank-based, not value-based: one freak session must not flatten the rest
 *  into a single tier (`worklog/bench` lesson — a run at 305 tok/s beside a dozen at 6). */
function tier(v, peers, better) {
  if (!isNum(v)) return 0;
  const vals = peers.filter(isNum).sort((a, b) => a - b);
  if (vals.length < 2 || vals[0] === vals[vals.length - 1]) return 0;
  const below = vals.filter((x) => (better > 0 ? x < v : x > v)).length;
  return 1 + Math.min(TIERS - 1, Math.floor((below / (vals.length - 1)) * TIERS));
}

/** The one place a tier becomes paint. */
const heat = (t) => (t ? ` style="background:var(--heat-${t})"` : '');

/** The columns, in the order they are read: what ran, how long, what it produced, what it cost. */
const COLUMNS = [
  // An engine the LB did not start has no generation: a dash, and its own start as the cell's title.
  ['gen', (r) => (isNum(r.generation) ? String(r.generation) : '—'), 'num', null,
   (r) => (isNum(r.generation) ? '' : whose(r))],
  ['model', (r) => esc(r.label || r.model_choice), 'name'],
  ['started', (r) => when(r.started_at), 'num'],
  ['busy', (r) => dur(r.active_s), 'num', { of: (r) => r.active_s, better: 1, group: 'all' }],
  ['standing', (r) => dur(r.wall_s), 'num'],
  ['tokens', (r) => int(r.gen_tokens), 'num', { of: (r) => r.gen_tokens, better: 1, group: 'all' }],
  ['tok/s', (r) => rate(r.gen_tok_s), 'num', { of: (r) => r.gen_tok_s, better: 1, group: 'model' }],
  ['peak', (r) => rate(r.peak_gen_tok_s), 'num',
   { of: (r) => r.peak_gen_tok_s, better: 1, group: 'model' }],
  ['requests', (r) => int(r.requests), 'num', { of: (r) => r.requests, better: 1, group: 'all' }],
  ['tok/req', (r) => rate(r.tokens_per_request, 0), 'num',
   { of: (r) => r.tokens_per_request, better: 1, group: 'model' }],
  ['prompt', (r) => tokens(r.prompt_tokens), 'num',
   { of: (r) => r.prompt_tokens, better: 1, group: 'all' }],
  ['cache hit', (r) => pct(r.prefix_hit_rate), 'num',
   { of: (r) => r.prefix_hit_rate, better: 1, group: 'model' }],
  // TTFT and ITL are WAITING: less is better, and the ramp is told so rather than guessing.
  ['TTFT', (r) => (isNum(r.ttft_mean_s) ? `${r.ttft_mean_s.toFixed(1)} s` : '—'), 'num',
   { of: (r) => r.ttft_mean_s, better: -1, group: 'model' }],
  ['ITL', (r) => (isNum(r.itl_mean_s) ? `${(r.itl_mean_s * 1000).toFixed(0)} ms` : '—'), 'num',
   { of: (r) => r.itl_mean_s, better: -1, group: 'model' }],
  // seats / budget / spec are the SETTINGS this session ran: never washed. A configuration is not
  // good or bad, it is what was asked for — the page that judges those is /benchmarks.
  ['seats', (r) => (r.config || {}).max_num_seqs ?? '—', 'num'],
  ['budget', (r) => int((r.config || {}).max_num_batched_tokens), 'num'],
  ['spec', (r) => (r.config || {}).num_speculative_tokens ?? '—', 'num'],
  ['accept', (r) => pct(r.acceptance), 'num', { of: (r) => r.acceptance, better: 1, group: 'model' }],
  ['boot', (r) => dur(r.boot_s), 'num', { of: (r) => r.boot_s, better: -1, group: 'model' }],
  ['W', (r) => rate(r.watts_mean, 0), 'num', { of: (r) => r.watts_mean, better: -1, group: 'model' }],
  ['Wh', (r) => energy(r.wh), 'num'],
  // EFFICIENCY: generated tokens per kWh the boxes drew while this session was busy. A QUALITY, so it
  // is judged against the model's own sessions (a 27B and a MoE are different machines), more = better.
  ['tok/kWh', (r) => perKwh(r.tok_per_kwh), 'num', { of: (r) => r.tok_per_kwh, better: 1, group: 'model' }],
];

/** THE COST COLUMNS, from what the server says it priced against: one for each session's OWN vendor's
 *  list and one per reference. They are built from the payload, not declared, because which lists exist
 *  is the pricing file's to say — and an LB that publishes none gets none. A cost is a VOLUME (a dollar
 *  is a dollar whoever's tokens it priced), so it is washed across the whole table, bigger = more saved. */
function costColumns(d) {
  const pl = d.price_lists;
  if (!pl || !pl.lists) return [];
  const cols = [];
  const wash = (of) => ({ of, better: 1, group: 'all' });
  if ((pl.own || []).length || Object.keys(pl.unpriced || {}).length) {
    const of = (r) => { const id = (r.own_price || {}).list; return id && r.cost ? r.cost[id] : null; };
    cols.push(['$ own', (r) => usd(of(r)), 'num', wash(of),
      (r) => (of(r) == null ? ((r.own_price || {}).why || '') : '')]);
  }
  for (const id of pl.references || []) {
    const of = (r) => (r.cost ? r.cost[id] : null);
    cols.push([`$ ${(pl.lists[id].label || id).replace(/^Claude /, '').toLowerCase()}`,
      (r) => usd(of(r)), 'num', wash(of)]);
  }
  return cols;
}

function renderTable(d) {
  const rows = d.sessions || [];
  const COLUMNS_ALL = COLUMNS.concat(costColumns(d));
  // The peers each washed column is judged against, worked out ONCE per render.
  const peers = new Map();
  for (const [key, , , wash] of COLUMNS_ALL) {
    if (!wash) continue;
    const pick = (list) => list.map(wash.of).filter(isNum);
    peers.set(key, wash.group === 'all' ? { all: pick(rows) }
      : rows.reduce((m, r) => ({ ...m,
          [r.model_choice]: m[r.model_choice] || pick(rows.filter((x) => x.model_choice === r.model_choice)) }), {}));
  }
  const head = `<thead><tr>${COLUMNS_ALL.map(([k, , cls, wash]) =>
    `<th class="${cls}">${esc(k)}${wash ? `<span class="s-dir" aria-hidden="true">${
      wash.better > 0 ? '\u25b2' : '\u25bc'}</span>` : ''}</th>`).join('')}</tr></thead>`;
  const body = rows.map((r) => {
    const failed = r.outcome && r.outcome !== 'serving';
    const dead = !(r.gen_tokens > 0);
    return `<tr class="${failed ? 's-row-failed' : ''}${dead ? ' s-row-idle' : ''}"${
      r.failure ? ` title="${esc(r.failure)}"` : ''}>`
      + COLUMNS_ALL.map(([key, read, cls, wash, why]) => {
        const group = wash && (wash.group === 'all' ? peers.get(key).all
          : peers.get(key)[r.model_choice]);
        const reason = why ? why(r) : '';
        return `<td class="${cls}"${reason ? ` title="${esc(reason)}"` : ''}${
          wash ? heat(tier(wash.of(r), group || [], wash.better)) : ''}>`
          + `${read(r)}</td>`;
      }).join('') + '</tr>';
  }).join('');
  $('s-table').innerHTML = `${head}<tbody>${body}</tbody>`;
  const idle = rows.filter((r) => !(r.gen_tokens > 0)).length;
  $('s-table-sub').textContent = [`${rows.length} newest first`,
    idle ? `${idle} loaded and served nothing` : '', ...unclaimed(d)].filter(Boolean).join(' · ');
  $('s-table-card').hidden = rows.length === 0;
}

function renderAlert(d) {
  const bits = [];
  if (d.error) bits.push(d.error);
  if (d.windows_truncated) bits.push(`${int(d.windows_truncated)} of the oldest minutes were not read `
    + `(the log hands over at most 20,160 at a time): the totals leave them out`);
  if (d.malformed) bits.push(`${int(d.malformed)} unreadable lines in the performance log`);
  if ((d.price_lists || {}).error) bits.push(`no prices: ${d.price_lists.error}`);
  if (d.boots_malformed) bits.push(`${int(d.boots_malformed)} unreadable lines in the boot log`);
  const rec = d.recorder || {};
  if (rec.enabled === false) bits.push('the performance recorder is OFF: nothing new is being written');
  if (rec.error) bits.push(String(rec.error));
  if (rec.dropped) bits.push(`${int(rec.dropped)} windows were dropped by a full queue`);
  const box = $('s-alert');
  box.innerHTML = bits.map((b) => `<div>${esc(b)}</div>`).join('');
  box.hidden = bits.length === 0;
}

function renderEmpty(d) {
  const none = !(d.sessions || []).length;
  $('s-loading').textContent = none
    ? (d.cluster_configured === false
      ? 'This gateway has no cluster: no model has ever been loaded here.'
      : (unclaimed(d).length ? `No session in this range · ${unclaimed(d).join(' · ')}.`
        : 'No session in this range — the log records a minute only while the engine is busy.'))
    : '';
  $('s-loading').hidden = !none;
}

function conn(ok, text) {
  $('connection-status').className = `conn${ok ? '' : ' bad'}`;
  $('status-text').textContent = text;
}

async function load() {
  renderRange();
  const days = (RANGES.find((r) => r.key === range) || RANGES[0]).days;
  try {
    const res = await fetch(`/lb/sessions?days=${encodeURIComponent(days)}${serverQuery()}`, { headers: { accept: 'application/json' } });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const d = await res.json();
    last = d;
    $('s-filters').hidden = false;
    renderAlert(d);
    renderTiles(d);
    renderPrices(d);
    renderFacets(d);
    renderCumulative(d);
    renderTable(d);
    renderEmpty(d);
    conn(true, `${int((d.totals || {}).sessions)} sessions · read ${when(d.now)}`);
  } catch (e) {
    conn(false, String(e.message || e));
    $('s-loading').hidden = false;
    $('s-loading').textContent = `The gateway did not answer: ${e.message || e}`;
  }
}

/* `?server=<name>`: server-scope.js (one helper, shared with the performance page) — that server's sessions. */
serverScope('Sessions');
load();
// The history moves only when a session ends; a minute is plenty, and this page is never the reason the
// LB is busy.
setInterval(load, 60000);
