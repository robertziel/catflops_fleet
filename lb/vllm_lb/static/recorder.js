/* THE FLIGHT RECORDER, one per box: a narrow strip under each machine card in the Cluster section, and the box's last
 * hour behind it (operator, 2026-10-30: "make them wrapped as small narrow section below each box in fleet, allow
 * expand").
 *
 *   THE STRIP       one line of figures — "⏺ 1h · peak 29 W inst · 34 W avg · GPU 44°C · CPU 47°C ·
 *                   net 0.14↓/0↑ Mb/s · 7 containers 5.0G" — a button (aria-expanded) that opens the recorder.
 *                   Collapsed by default; it sits in its own grid row under the card (gateway.css `.machine-slot`, a
 *                   subgrid), so a closed strip never makes a card taller and an open one never stretches its
 *                   neighbours' cards.
 *   OPEN            "FLIGHT RECORDER · 1H" with ⤢ expand; the power line; `charts` (a <details>); CONTAINERS · count
 *                   · total RAM, the top few by RAM as a name · RAM · image table, and "+k more".
 *   EXPAND          a dialog (the connect dialog's `.cn` frame): the hour in minute buckets, the last ten minutes
 *                   sample by sample, and every container.
 *
 * EACH FIGURE ONCE (operator, 2026-10-30: the open strip drew every figure twice, the dialog three times). Charts
 * closed: the compact grid. Charts open: the charts alone — each chart's headline IS the figure's value. The dialog:
 * the two chart blocks, no grid. A figure the hour holds no reading of (no fan tach, amdgpu's absent throttle mask)
 * gets NO chart — an empty graph is noise — and stays in the grid as "—" with the reason in its title.
 *
 * ONE TABLE, ONE FORMATTER. The labels, units and number formats are the server's (fleet_telemetry/figures.py
 * FIGURES_TABLE, served as `labels`: key · label · unit · fmt); `recFmt` is the only place a figure becomes text.
 *
 * ONE POWER READING IS ONE FIGURE. amdgpu answers its averaged and instantaneous draw from one sensor on an APU, and
 * "119W peak / 119W avg" read as two measurements: when the recorder says `one_reading`, the line is the hour's peak
 * and mean; when the sensor is the APU package (`scope`, the sender's PPT label), it says "(package)".
 *
 * THE DATA. /lb/status `hardware[box].recorder` (vllm_lb/recorder.py `summary`) every poll: the latest figures, their
 * labels, units and formats (the server's one list), the throttle's decoded reasons, the hour's power, the
 * containers' head. The hour itself is GET /lb/hardware/<box>/recorder, fetched only while charts, "+k more" or the
 * dialog is open, at most every REC_REFRESH_MS — never on every poll. A chart block that does not yet hold its window
 * says so ("filling · 2 of 60 min"): an hour with two minutes in it is one bar at the right edge, not a broken chart.
 *
 * NO FIGURE IS MADE UP. A figure the box does not send is "n/a" in the strip and "—" in the grid, its reason in the
 * title (a sender that predates the recorder block, a box with no fan tach, a counter that went backwards); a failed
 * fetch says its HTTP status where the charts would be. Never a 0.
 *
 * STATELESS SERVER. Which strips are open, and whether their charts are, is this browser's: localStorage `lb.recorder`,
 * every touch wrapped (layout.js's `LyStore` rule) — a browser that keeps nothing still opens and closes for the visit.
 *
 * Loaded after gateway.js (it uses `tile`, `esc`, `isNum`, `fmtDuration`, `paint`, `placeMarks`); gateway.js
 * `renderMachines` calls `recorderPaint` per box when this file is present.
 */
'use strict';

const REC_KEY = 'lb.recorder';
const REC_REFRESH_MS = 15000;

/** Open/closed per box, in this browser. `get` never throws; `set` reports whether the browser kept it. */
class RecStore {
  constructor(storage) { this.storage = storage; this.mem = {}; }
  all() {
    try {
      const raw = this.storage && this.storage.getItem(REC_KEY);
      const v = raw ? JSON.parse(raw) : {};
      return v && typeof v === 'object' && !Array.isArray(v) ? v : {};
    } catch (e) { return Object.assign({}, this.mem); }
  }
  get(box) { return Object.assign({ open: false, charts: false, more: false }, this.all()[box] || this.mem[box] || {}); }
  set(box, patch) {
    const all = this.all();
    all[box] = Object.assign(this.get(box), patch);
    this.mem[box] = all[box];
    try { this.storage.setItem(REC_KEY, JSON.stringify(all)); return true; } catch (e) { return false; }
  }
}

/** MiB as the screenshot writes RAM: "705M", "3.5G". */
const recRam = (mb) => (isNum(mb) ? (mb >= 1024 ? `${(mb / 1024).toFixed(1)}G` : `${Math.round(mb)}M`) : null);

/** THE ONE FORMATTER: a figure's number as text by the table's `fmt`, or null when there is no reading.
 *  "int" whole; "pct" one decimal; "rate" with the precision its size needs — ≥10 whole, 1–10 one decimal, under 1
 *  two decimals ("<0.01" below that), a true zero "0": a 0.14 Mb/s link drawn "0" read as a dead NIC. */
function recFmt(v, fmt) {
  if (!isNum(v)) return null;
  if (fmt === 'int') return String(Math.round(v));
  if (fmt === 'pct') return v.toFixed(1);
  if (fmt === 'rate') {
    const a = Math.abs(v);
    if (v === 0) return '0';
    if (a >= 10) return String(Math.round(v));
    if (a >= 1) return v.toFixed(1);
    return a < 0.005 ? '<0.01' : v.toFixed(2);
  }
  return null;
}

/** The space between a number and its unit: none for °C and %, a space for the rest ("600 MHz", "0.14 Mb/s"). */
const recGap = (unit) => (unit && unit !== '°C' && unit !== '%' ? ' ' : '');

const recLabels = (rec) => (Array.isArray(rec.labels) ? rec.labels : []);
const recLabel = (rec, key) => recLabels(rec).find((l) => l.key === key) || { key, label: key, unit: '', fmt: '' };
const recWhy = (rec, k) => ((rec.why || {})[k] || '');

/** One figure's value text WITHOUT its unit, or null: a "flags" figure (THROTTLE) is its decoded reasons — "none" for
 *  a clear mask — never its 0-100 index; the rest are `recFmt`. */
function recValue(rec, l) {
  if (l.fmt === 'flags') {
    const r = rec.throttle_reasons;
    return Array.isArray(r) ? (r.length ? r.join(', ') : 'none') : null;
  }
  return recFmt((rec.figures || {})[l.key], l.fmt);
}

/** One figure as text in its unit ("600 MHz", "44°C", "none"), or null. */
function recText(rec, l) {
  const v = recValue(rec, l);
  return v == null ? null : `${v}${l.fmt === 'flags' ? '' : `${recGap(l.unit)}${l.unit || ''}`}`;
}

/** Watts in the GPU POWER figure's own format and unit. */
function recW(rec, v) {
  const l = recLabel(rec, 'gpu_power_w');
  const n = recFmt(v, l.fmt);
  return n == null ? null : `${n}${recGap(l.unit)}${l.unit}`;
}

/** The hour's power peak as words: ONE reading → "peak 119 W"; two → "peak 29 W inst · 34 W avg"; none → null. */
function recPeak(rec) {
  const p = rec.power || {};
  if (!isNum(p.peak_inst_w) && !isNum(p.peak_avg_w)) return null;
  if (p.one_reading) return `peak ${recW(rec, isNum(p.peak_avg_w) ? p.peak_avg_w : p.peak_inst_w)}`;
  return `peak ${recW(rec, p.peak_inst_w) || 'n/a'} inst · ${recW(rec, p.peak_avg_w) || 'n/a'} avg`;
}

/** "GPU POWER" from the table, and what the sensor covers when the sender says ("GPU POWER (package)"). */
function recPowerLabel(rec) {
  const scope = (rec.power || {}).scope;
  return `${recLabel(rec, 'gpu_power_w').label}${scope ? ` (${scope})` : ''}`;
}

/** The strip's figures, in order: [text, why-when-absent]. */
function recStripParts(rec) {
  const f = rec.figures || {};
  const p = rec.power || {};
  const part = (name, key) => {
    const t = recText(rec, recLabel(rec, key));
    return t ? [`${name} ${t}`, ''] : [`${name} n/a`, recWhy(rec, key)];
  };
  const peak = recPeak(rec);
  const power = peak ? [peak, p.why || ''] : ['power n/a', p.why || ''];
  const nin = recLabel(rec, 'net_in_mbps'), nout = recLabel(rec, 'net_out_mbps');
  const net = isNum(f.net_in_mbps) || isNum(f.net_out_mbps)
    ? [`net ${recValue(rec, nin) || 'n/a'}↓/${recValue(rec, nout) || 'n/a'}↑${recGap(nin.unit)}${nin.unit}`, '']
    : ['net n/a', recWhy(rec, 'net_in_mbps')];
  const c = rec.containers;
  const cont = c ? [`${c.count} containers${recRam(c.total_ram_mb) ? ` ${recRam(c.total_ram_mb)}` : ''}`, c.total_why || '']
    : ['containers n/a', rec.containers_why || ''];
  return [power, part('GPU', 'gpu_temp_c'), part('CPU', 'cpu_temp_c'), net, cont];
}

/** How much of the window the recorder holds, when it is less than all of it: "holds 12m". */
function recHeld(rec) {
  return isNum(rec.held_s) && isNum(rec.window_s) && rec.held_s < rec.window_s - 60
    ? `holds ${fmtDuration(rec.held_s)}` : '';
}

/** A chart block's window while it fills: "filling · 2 of 60 min" ('' once it holds the whole window). */
function recFilling(heldS, windowS) {
  if (!isNum(heldS) || !isNum(windowS) || windowS <= 0 || heldS >= windowS) return '';
  const of = Math.round(windowS / 60);
  return `filling · ${Math.min(of, Math.max(1, Math.ceil(heldS / 60)))} of ${of} min`;
}

/** THE STRIP: one button, one line, figures only; every absent figure's reason in its title. */
function recStripHtml(box, rec, open) {
  const parts = recStripParts(rec);
  const why = parts.filter(([, w]) => w).map(([t, w]) => `${t}: ${w}`);
  const held = recHeld(rec);
  const title = `flight recorder · ${box}${held ? ` · ${held} of the hour` : ''} · power: ${recPowerLabel(rec)}`
    + `${why.length ? `\n${why.join('\n')}` : ''}`;
  return `<button type="button" class="rec-strip" aria-expanded="${open ? 'true' : 'false'}" aria-controls="rec-${esc(box)}"
    title="${esc(title)}"><span class="rec-dot" aria-hidden="true">⏺</span> 1h${parts.map(([t, w]) =>
    ` · <span class="${w ? 'rec-na' : ''}">${esc(t)}</span>`).join('')}</button>`;
}

/** The figures grid: label over value, "—" with the reason when there is none. `only` (a Set of keys): just those —
 *  beside the charts, the figures that have none. */
function recGridHtml(rec, only) {
  const ls = recLabels(rec).filter((l) => !only || only.has(l.key));
  if (!ls.length) return '';
  const scope = (rec.power || {}).scope;
  return `<div class="rec-grid">${ls.map((l) => {
    const t = recText(rec, l);
    const title = !t ? recWhy(rec, l.key) : l.key === 'gpu_power_w' && scope ? recPowerLabel(rec) : '';
    return `<div class="rec-fig"${title ? ` title="${esc(title)}"` : ''}><span>${esc(l.label)}</span>`
      + `<b${t ? '' : ' class="rec-na"'}>${esc(t || '—')}</b></div>`;
  }).join('')}</div>`;
}

/** The power line: what the sensor is, the hour's peak (one reading, or inst and avg named), the hour's mean, the
 *  input rail; `held` adds how much of the hour is held (left out where the charts say "filling"). */
function recPowerHtml(rec, held) {
  const p = rec.power || {};
  const peak = recPeak(rec);
  const mean = recW(rec, p.mean_w);
  const draw = peak
    ? `<span title="${esc(`the hour's highest ${p.one_reading ? '' : 'instantaneous / averaged '}draw and its mean`)}">${esc(peak)}${mean ? ` · mean ${esc(mean)}` : ''}</span>`
    : `<span class="rec-na" title="${esc(p.why || '')}">peak n/a</span>`;
  const rail = isNum(p.input_rail_w) ? `input rail ${esc(recW(rec, p.input_rail_w))}`
    : `<span class="rec-na" title="${esc(p.input_rail_why || '')}">input rail n/a</span>`;
  const h = held ? recHeld(rec) : '';
  return `<div class="rec-power"><span class="rec-plab">${esc(recPowerLabel(rec))}</span> ${draw} · ${rail}`
    + `${h ? ` · <span class="rec-held">${esc(h)}</span>` : ''}</div>`;
}

/** CONTAINERS · count · total RAM, the rows, and "+k more" (all of them once `all` is fetched and asked for); the
 *  containers a sender did not send (it sends its top N by RAM) are counted in the head and said, never drawn.
 *  THE ROWS ARE A TABLE (recorder.css `.rec-ctab`): name · RAM right-aligned · image cut with its whole ref in the
 *  title — aligned columns where there is room, two-line rows where there is not. */
function recContainersHtml(box, rec, all, more) {
  const c = rec.containers;
  if (!c) return `<div class="rec-cont"><div class="rec-h">CONTAINERS</div><div class="rec-na" title="${esc(rec.containers_why || '')}">n/a</div></div>`;
  const rows = more && Array.isArray(all) ? all : (c.top || []);
  const ram = recRam(c.total_ram_mb);
  return `<div class="rec-cont"><div class="rec-h">CONTAINERS · ${esc(c.count)}${ram ? ` · ${esc(ram)} RAM` : ''}</div>`
    + `<ul class="rec-list rec-ctab">${rows.map((x) => `<li><span class="rec-name" title="${esc(x.name)}">${esc(x.name)}</span>`
      + `<b class="rec-cram">${esc(recRam(x.ram_mb) || '—')}</b>`
      + `<span class="rec-img" title="${esc(x.image || '')}">${esc(x.image || '')}</span></li>`).join('')}</ul>`
    + (c.more > 0 ? `<button type="button" class="rec-more" data-box="${esc(box)}" aria-expanded="${more ? 'true' : 'false'}">${more ? 'fewer' : `+${esc(c.more)} more`}</button>` : '')
    + (c.unlisted > 0 ? `<span class="rec-unlisted rec-na" title="the sender sends its largest containers by RAM; the count and the RAM above are every container's">+${esc(c.unlisted)} smaller not sent</span>` : '')
    + '</div>';
}

const recCoarse = (hour, k) => ((((hour || {}).coarse || {}).series || {})[k] || {}).mean;
const recFine = (hour, k) => (((hour || {}).fine || {}).series || {})[k];

/** The figures the hour holds at least one reading of — the ones that get a chart. */
function recCharted(rec, hour) {
  return new Set(recLabels(rec).filter((l) => (recCoarse(hour, l.key) || []).some(isNum)).map((l) => l.key));
}

/** Every charted figure's past as the panel's tile (gateway.js `tile`), `pick` choosing the series out of the
 *  recorder hour (the minute means, or the last minutes' samples). The headline is the figure's value now.
 *  COLOUR IS A JUDGEMENT AGAINST A CEILING. A percent and the 0-100 throttle index have one, and wear the panel's
 *  load scale; a temperature, a draw, a clock or a rate has none the recorder knows (a thermal or power limit is a
 *  hardware figure, the type file's, not this page's), and against its own peak it would read red at any steady
 *  value — so those tiles are drawn plain (`rec-plain`, recorder.css). */
function recTilesHtml(box, rec, hour, pick, tag, charted) {
  const f = rec.figures || {};
  const scope = (rec.power || {}).scope;
  return `<div class="rec-tiles">${recLabels(rec).filter((l) => charted.has(l.key)).map((l) => {
    const series = pick(hour, l.key) || [];
    const shown = recValue(rec, l);
    const ceiling = l.unit === '%' || l.fmt === 'flags' ? 100 : undefined;
    const sub = l.key === 'gpu_power_w' && scope ? esc(scope) : null;
    const html = tile(l.label, shown, l.fmt === 'flags' ? '' : `${recGap(l.unit)}${l.unit || ''}`, sub, series,
      { cols: Math.max(2, series.length), key: `${box}|rec|${tag}|${l.key}`, now: f[l.key],
        max: ceiling, cls: ceiling ? '' : 'rec-plain', title: shown == null ? recWhy(rec, l.key) : '' });
    return ceiling ? html : html.replace('<div class="tile', '<div class="rec-plain-tile tile');
  }).join('')}</div>`;
}

/** A chart block's head: what it is and, while it fills, how much of its window it holds. */
function recBlockHead(name, fill) {
  return `<div class="rec-h">${esc(name)}${fill ? ` · <span class="rec-fill">${esc(fill)}</span>` : ''}</div>`;
}

/** THE OPEN RECORDER under the strip. `st` is this box's stored state, `got` its fetched hour ({data} | {error}).
 *  Charts drawn → the charts and, after them, only the figures that have none; otherwise the whole grid. */
function recBodyHtml(box, rec, st, got) {
  const hour = got && got.data;
  const all = hour && hour.containers ? hour.containers.all : null;
  const drawn = st.charts && hour;
  const charted = drawn ? recCharted(rec, hour) : null;
  const inner = !st.charts ? '' : !got ? '<p class="rec-na">loading…</p>' : got.error ? `<p class="rec-na">${esc(got.error)}</p>`
    : `${recFilling(rec.held_s, rec.window_s) ? recBlockHead(`LAST HOUR · ${Math.round((hour.bucket_s || 60) / 60)} min BUCKETS`, recFilling(rec.held_s, rec.window_s)) : ''}`
      + recTilesHtml(box, rec, hour, recCoarse, 'h', charted);
  const absent = drawn ? new Set(recLabels(rec).map((l) => l.key).filter((k) => !charted.has(k))) : null;
  return `<div class="rec-body" id="rec-${esc(box)}">
    <div class="rec-top"><span class="rec-h">FLIGHT RECORDER · 1H</span>
      <button type="button" class="rec-expand" data-box="${esc(box)}" aria-label="expand the flight recorder of ${esc(box)}">⤢ expand</button></div>
    ${recPowerHtml(rec, !drawn)}
    <details class="rec-charts" data-box="${esc(box)}"${st.charts ? ' open' : ''}><summary>charts</summary>${inner}</details>
    ${recGridHtml(rec, absent)}
    ${recContainersHtml(box, rec, all, st.more)}
  </div>`;
}

/** THE EXPAND VIEW: the hour (minute means) and the last minutes sample by sample — the charts carry the values, so
 *  no grid repeats them; the figures with no chart; every container. */
function recDialogHtml(box, rec, got) {
  const hour = got && got.data;
  let body;
  if (!got) body = '<p class="rec-na">loading…</p>';
  else if (got.error) body = `<p class="rec-na">${esc(got.error)}</p>`;
  else {
    const charted = recCharted(rec, hour);
    const absent = new Set(recLabels(rec).map((l) => l.key).filter((k) => !charted.has(k)));
    body = recBlockHead(`LAST HOUR · ${Math.round(hour.bucket_s || 60)} s BUCKETS`, recFilling(rec.held_s, rec.window_s))
      + recTilesHtml(box, rec, hour, recCoarse, 'dh', charted)
      + recBlockHead(`LAST ${Math.round((hour.fine_s || 0) / 60)} MIN · EVERY SAMPLE`, recFilling(rec.held_s, hour.fine_s))
      + recTilesHtml(box, rec, hour, recFine, 'df', charted)
      + recGridHtml(rec, absent);
  }
  const all = hour && hour.containers ? hour.containers.all : null;
  return `<div class="cn-box rec-dlg-box"><div class="cn-head"><h2 class="cn-title">Flight recorder · ${esc(box)}</h2>
      <button type="button" class="cn-x rec-x" aria-label="close">×</button></div>
    ${recPowerHtml(rec, !hour)}${body}${recContainersHtml(box, rec, all, true).replace(/<button[^>]*class="rec-more"[\s\S]*?<\/button>/, '')}</div>`;
}

// ── the DOM ──────────────────────────────────────────────────────────────────────────────────────

const recStore = new RecStore(typeof window !== 'undefined' ? (() => { try { return window.localStorage; } catch (e) { return null; } })() : null);
const recGot = {};          // box → {at, data} | {at, error}
const recLast = {};         // box → [slot, rec]: what to repaint when a fetch lands
let recDialogBox = null;

async function recFetch(box) {
  const prev = recGot[box];
  if (prev && (prev.pending || Date.now() - prev.at < REC_REFRESH_MS)) return;
  recGot[box] = Object.assign({}, prev || {}, { pending: true });
  let got;
  try {
    const r = await fetch(`/lb/hardware/${encodeURIComponent(box)}/recorder`);
    const body = await r.json().catch(() => ({}));
    got = r.ok ? { data: body } : { error: `HTTP ${r.status}${body.error ? `: ${body.error}` : ''}` };
  } catch (e) {
    got = { error: `the recorder could not be read: ${e.message || e}` };
  }
  recGot[box] = Object.assign(got, { at: Date.now() });
  const last = recLast[box];
  if (last) recorderPaint(last[0], box, last[1]);
}

function recDialog() {
  let d = document.getElementById('rec-dialog');
  if (!d) {
    d = document.createElement('dialog');
    d.id = 'rec-dialog';
    d.className = 'cn rec-dlg';
    d.setAttribute('aria-label', 'flight recorder');
    d.addEventListener('click', (e) => {
      if (e.target === d || e.target.closest('.rec-x')) d.close();
    });
    // Escape, × or the backdrop: the keyboard goes back to the ⤢ expand it came from (the strip's repaints replace
    // that button, so it is looked up by box, not remembered as a node).
    d.addEventListener('close', () => {
      const box = recDialogBox;
      recDialogBox = null;
      const host = (recLast[box] || [])[0];
      const back = host && host.querySelector('.rec-expand');
      if (back) back.focus();
    });
    document.body.appendChild(d);
  }
  return d;
}

/** The controls a repaint replaces, by the selector that finds their successor. */
const REC_FOCUSABLE = ['.rec-strip', '.rec-expand', '.rec-charts > summary', '.rec-more', '.rec-x'];

/** Repaint `el`, keeping the keyboard where it was. EVERY POLL REPAINTS (the figures move), and an innerHTML repaint
 *  drops the focused button: a keyboard user's Space on the strip toggled it and then lost focus to <body> on the next
 *  poll. The focused control is found again by its selector and refocused. */
function recPaintKeepingFocus(el, html) {
  const a = typeof document !== 'undefined' ? document.activeElement : null;
  const sel = a && el.contains && el.contains(a) ? REC_FOCUSABLE.find((s) => a.matches && a.matches(s)) : null;
  paint(el, html);
  const back = sel && el.querySelector(sel);
  if (back && document.activeElement !== back) back.focus({ preventScroll: true });
}

/** Paint one box's recorder into its host under the card (`host`: gateway.js `.rec`). Wires its events once. */
function recorderPaint(host, box, rec) {
  if (!host) return;
  recLast[box] = [host, rec];
  if (!rec) { paint(host, ''); return; }
  const st = recStore.get(box);
  const wantsHour = st.open && (st.charts || st.more);
  if (wantsHour || recDialogBox === box) recFetch(box);
  recPaintKeepingFocus(host, recStripHtml(box, rec, st.open) + (st.open ? recBodyHtml(box, rec, st, recGot[box] && (recGot[box].data || recGot[box].error) ? recGot[box] : null) : ''));
  if (recDialogBox === box) {
    const d = recDialog();
    recPaintKeepingFocus(d, recDialogHtml(box, rec, recGot[box] && (recGot[box].data || recGot[box].error) ? recGot[box] : null));
    if (typeof placeMarks === 'function') placeMarks(d);
  }
  if (typeof placeMarks === 'function') placeMarks(host);
  if (host.dataset.recWired) return;
  host.dataset.recWired = '1';
  host.addEventListener('click', (e) => {
    const [h, r] = recLast[box] || [];
    // Each repaint keeps the keyboard on the control that was pressed (`recPaintKeepingFocus`).
    if (e.target.closest('.rec-strip')) {
      recStore.set(box, { open: !recStore.get(box).open });
      recorderPaint(h, box, r);
    } else if (e.target.closest('.rec-more')) {
      recStore.set(box, { more: !recStore.get(box).more });
      recorderPaint(h, box, r);
    } else if (e.target.closest('.rec-expand')) {
      recDialogBox = box;
      const d = recDialog();
      recorderPaint(h, box, r);
      if (!d.open) d.showModal();
    }
  });
  // <details> toggles itself; its state is remembered, and opening it fetches the hour.
  host.addEventListener('toggle', (e) => {
    if (!e.target.matches || !e.target.matches('.rec-charts')) return;
    if (recStore.get(box).charts === e.target.open) return;
    recStore.set(box, { charts: e.target.open });
    const [h, r] = recLast[box] || [];
    recorderPaint(h, box, r);
  }, true);
}
