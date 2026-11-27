/* OBSERVABILITY ON THE PANEL: three on-demand views in one dialog, none of them on the poll.
 *
 *   CHECK      each machine card's head gets "check": GET /lb/check?box=<box> (vllm_lb/connectivity.py) — one row per
 *              capability (telemetry, each engine, each SSH rank): ✓/✗ · capability · target · the probe's own words
 *              · ms, and on a failed row its short hint ("http://…:8001 unreachable from the gateway") with the full
 *              recovery in its title (connectivity.py `hint_short` / `hint`). "again" re-runs it. Nothing runs until asked.
 *   HISTORY    each machine card's head gets "history": GET /lb/history?box=<box>&tier=48h|30d (vllm_lb/hw_tiers.py) —
 *              a 48 h / 30 d toggle and one chart per figure: the min..max band and the avg line, headlined with
 *              numbers ("avg 54 · max 71 °C"). A figure with no reading in the range draws no chart: "—". Under each
 *              chart its start / end ticks, read from the samples' OWN timestamps ("Nov 25 14:00" … "now") and placed
 *              where those samples are drawn — a series that began 8 h ago in a 48 h range does not tick "48 h ago".
 *   DAILY      the selected server's links get "daily": GET /lb/perf/daily?server=<it> (vllm_lb/daily.py) — per
 *              model, its 24 h and 30 d tokens and its peak-minute tok/s, then 30 bars of tokens/day, each titled with
 *              its day and figures. The summary line carries the best peak minute of all models. No tok/s bar row: a
 *              day's pooled tok/s drew nearly the shape of its tokens (the UX audit) — the peak is the figure
 *              that says something the bars do not. A day with no tokens has no bar (an empty slot, not a 0).
 *
 * The dialog is the connect dialog's frame (gateway.css `.cn`), so every style re-values it with the panel's tokens,
 * light or dark. The server's words are always escaped (`esc`). Figures, not sentences: labels carry numbers.
 *
 * Loaded after gateway.js (it uses `esc`, `isNum`); gateway.js `renderMachines` calls `observePaint(card, box)` per
 * machine card when this file is present.
 */
'use strict';

const OBS_TIERS = [['48h', '48 h'], ['30d', '30 d']];

/** 1708150 → "1,708,150". */
const obsNum = (n) => (isNum(n) ? Math.round(n).toLocaleString('en-US') : '—');
/** A figure with the precision its size needs: ≥100 whole (thousands grouped: "68,999 MiB"), ≥10 one decimal, else two. */
function obsFig(v) {
  if (!isNum(v)) return '—';
  const a = Math.abs(v);
  return a >= 100 ? Math.round(v).toLocaleString('en-US') : a >= 10 ? String(Math.round(v * 10) / 10) : String(Math.round(v * 100) / 100);
}
/** 12.3 ms → "12 ms"; 1234 → "1.2 s". */
const obsMs = (ms) => (!isNum(ms) ? '' : ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`);
/** 1708150 → "1.7M" for an axis label. */
function obsShort(n) {
  if (!isNum(n)) return '—';
  const a = Math.abs(n);
  return a >= 1e9 ? `${(n / 1e9).toFixed(1)}G` : a >= 1e6 ? `${(n / 1e6).toFixed(1)}M`
    : a >= 1e3 ? `${(n / 1e3).toFixed(1)}k` : obsFig(n);
}

function obsFrame(title, body, extra) {
  return `<div class="cn-box obs-box"><div class="cn-head"><h2 class="cn-title">${title}</h2>${extra || ''}
    <button type="button" class="cn-x obs-x" aria-label="close">×</button></div>${body}</div>`;
}

// ── CHECK ────────────────────────────────────────────────────────────────────────────────────

/** The check dialog's body for `box` from `got` ({data} | {error} | null while it runs). */
function obsCheckHtml(box, got) {
  const again = '<button type="button" class="cn-open obs-again">again</button>';
  if (!got) return obsFrame(`Check · ${esc(box)}`, '<p class="obs-na">checking…</p>');
  if (got.error) return obsFrame(`Check · ${esc(box)}`, `<p class="obs-na bad">${esc(got.error)}</p>`, again);
  const b = ((got.data || {}).boxes || []).find((x) => x.box === box) || { checks: [] };
  const rows = b.checks || [];
  const pass = rows.filter((r) => r.ok).length;
  const list = rows.map((r) => `<li class="obs-row ${r.ok ? 'ok' : 'bad'}">
      <span class="obs-mark" aria-label="${r.ok ? 'pass' : 'fail'}">${r.ok ? '✓' : '✗'}</span>
      <b class="obs-cap">${esc(r.cap)}</b><span class="obs-target" title="${esc(r.target)}">${esc(r.target)}</span>
      <span class="obs-detail">${esc(r.detail)}${isNum(r.ms) && r.ms > 0 ? ` · ${obsMs(r.ms)}` : ''}</span>
      ${!r.ok && r.hint_short ? `<span class="obs-hint" title="${esc(r.hint)}">${esc(r.hint_short)}</span>` : ''}</li>`).join('');
  return obsFrame(`Check · ${esc(box)} · <span class="${pass === rows.length ? 'ok' : 'bad'}">${pass}/${rows.length} pass</span>`,
    `<ul class="obs-rows">${list}</ul>${b.ssh_why ? `<details class="obs-why"><summary title="${esc(b.ssh_why)}">ssh —</summary>`
      + `<p class="obs-why-r">${esc(b.ssh_why)}</p></details>` : ''}`, again);
}

// ── HISTORY ──────────────────────────────────────────────────────────────────────────────────

const OBS_W = 300;
const OBS_H = 60;

/** One figure's band (min..max) and avg line over [t0, t1]: `points` [[t, min, avg, max]] oldest first; a silence
 *  longer than 1.5 buckets breaks both (a gap is drawn as a gap). */
function obsBandSvg(points, t0, t1, bucketS) {
  const lo = Math.min(...points.map((p) => p[1]));
  const hi = Math.max(...points.map((p) => p[3]));
  const pad = hi > lo ? (hi - lo) * 0.08 : Math.max(1, Math.abs(hi) * 0.05);
  const y0 = lo - pad;
  const y1 = hi + pad;
  const x = (t) => (((t + bucketS / 2) - t0) / Math.max(1, t1 - t0)) * OBS_W;
  const y = (v) => OBS_H - ((v - y0) / (y1 - y0)) * OBS_H;
  const runs = [];
  points.forEach((p, i) => {
    if (i === 0 || p[0] - points[i - 1][0] > bucketS * 1.5) runs.push([]);
    runs[runs.length - 1].push(p);
  });
  const f = (n) => n.toFixed(1);
  const parts = runs.map((r) => {
    const top = r.map((p) => `${f(x(p[0]))},${f(y(p[3]))}`);
    const bottom = r.slice().reverse().map((p) => `${f(x(p[0]))},${f(y(p[1]))}`);
    const band = r.length === 1
      ? `<rect class="obs-band" x="${f(x(r[0][0]) - 1)}" y="${f(y(r[0][3]))}" width="2" height="${f(Math.max(1, y(r[0][1]) - y(r[0][3])))}"/>`
      : `<polygon class="obs-band" points="${top.join(' ')} ${bottom.join(' ')}"/>`;
    const avg = r.length === 1 ? '' : `<polyline class="obs-avg" points="${r.map((p) => `${f(x(p[0]))},${f(y(p[2]))}`).join(' ')}"/>`;
    return band + avg;
  });
  return `<svg class="obs-svg" viewBox="0 0 ${OBS_W} ${OBS_H}" preserveAspectRatio="none" aria-hidden="true">${parts.join('')}</svg>`;
}

const OBS_MON = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
/** Epoch s → "Nov 25 14:00" in the reader's own clock. */
function obsWhen(t) {
  const d = new Date(t * 1000);
  const p = (n) => String(n).padStart(2, '0');
  return `${OBS_MON[d.getMonth()]} ${d.getDate()} ${p(d.getHours())}:${p(d.getMinutes())}`;
}

/** The time ticks under one chart, from its samples' own first and last timestamps, each placed where obsBandSvg
 *  draws that sample (the same x). The last reads "now" only when it is the current bucket. Samples closer than 65 % of
 *  the width get one range label right-aligned on the last sample: two labels that close would overlap. */
function obsTicksHtml(pts, t0, t1, bucketS) {
  const x = (t) => Math.min(1, Math.max(0, ((t + bucketS / 2) - t0) / Math.max(1, t1 - t0)));
  const first = pts[0][0];
  const last = pts[pts.length - 1][0];
  const end = last + bucketS * 2 >= t1 ? 'now' : obsWhen(last);
  const left = (x(first) * 100).toFixed(1);
  const right = ((1 - x(last)) * 100).toFixed(1);
  const body = x(last) - x(first) >= 0.65
    ? `<span class="obs-t0" style="left:${left}%">${obsWhen(first)}</span><span class="obs-t1" style="right:${right}%">${end}</span>`
    : `<span class="obs-t01" style="right:${right}%">${obsWhen(first)} – ${end}</span>`;
  return `<div class="obs-tt" aria-label="${obsWhen(first)} – ${end}">${body}</div>`;
}

/** The history dialog's body for `box` in `tier` from `got`; `now` is the right edge (epoch s). */
function obsHistoryHtml(box, tier, got, now) {
  const toggle = `<span class="obs-seg" role="group" aria-label="range">${OBS_TIERS.map(([k, label]) =>
    `<button type="button" class="obs-tier" data-tier="${k}" aria-pressed="${k === tier}">${label}</button>`).join('')}</span>`;
  const title = `History · ${esc(box)}`;
  if (!got) return obsFrame(title, '<p class="obs-na">loading…</p>', toggle);
  if (got.error) return obsFrame(title, `<p class="obs-na bad">${esc(got.error)}</p>`, toggle);
  const d = got.data || {};
  const t1 = isNum(now) ? now : Date.now() / 1000;
  const t0 = t1 - (d.window_s || 0);
  const note = d.state === 'corrupt' ? `<p class="obs-na bad">history file: ${esc(d.why)}</p>` : '';
  const tiles = Object.entries(d.series || {}).map(([key, s]) => {
    const pts = (s.points || []).filter((p) => p[0] + (d.bucket_s || 0) > t0);
    const unit = s.unit ? ` ${esc(s.unit)}` : '';
    if (!pts.length) {
      return `<div class="obs-tile empty" data-metric="${esc(key)}"><div class="obs-th"><span>${esc(s.label)}</span>
        <b>—</b></div><span class="obs-na">no reading · ${esc(tier)}</span></div>`;
    }
    const avg = pts.reduce((a, p) => a + p[2], 0) / pts.length;
    const max = Math.max(...pts.map((p) => p[3]));
    const min = Math.min(...pts.map((p) => p[1]));
    return `<div class="obs-tile" data-metric="${esc(key)}"><div class="obs-th"><span>${esc(s.label)}</span>
        <b>avg ${obsFig(avg)} · max ${obsFig(max)}${unit}</b></div>${obsBandSvg(pts, t0, t1, d.bucket_s || 60)}
        ${obsTicksHtml(pts, t0, t1, d.bucket_s || 60)}
        <div class="obs-tf"><span>min ${obsFig(min)}${unit}</span><span>${pts.length} × ${Math.round((d.bucket_s || 60) / 60)} min</span></div></div>`;
  }).join('');
  return obsFrame(title, `${note}<div class="obs-tiles">${tiles}</div>`, toggle);
}

// ── DAILY ────────────────────────────────────────────────────────────────────────────────────

const OBS_BW = 300;
const OBS_BH = 44;

/** 30 bars of `values` (null = no bar) on `days`, each titled "<day> · <fmt(v)>"; scaled to the largest. */
function obsBarsSvg(values, days, fmt) {
  const max = Math.max(0, ...values.filter(isNum));
  const n = Math.max(1, values.length);
  const w = OBS_BW / n;
  const bars = values.map((v, i) => {
    if (!isNum(v) || max <= 0 || v <= 0) return '';
    const h = Math.max(1, (v / max) * OBS_BH);
    return `<rect class="obs-bar" x="${(i * w + w * 0.12).toFixed(1)}" y="${(OBS_BH - h).toFixed(1)}" width="${(w * 0.76).toFixed(1)}" height="${h.toFixed(1)}"><title>${esc(days[i])} · ${esc(fmt(v))}</title></rect>`;
  }).join('');
  return `<svg class="obs-bars" viewBox="0 0 ${OBS_BW} ${OBS_BH}" preserveAspectRatio="none" role="img">${bars}</svg>`;
}

/** The daily rollup's body from `got`. */
function obsDailyHtml(got, server) {
  const title = `Daily · 30 d${server ? ` · ${esc(server)}` : ''}`;
  if (!got) return obsFrame(title, '<p class="obs-na">loading…</p>');
  if (got.error) return obsFrame(title, `<p class="obs-na bad">${esc(got.error)}</p>`);
  const d = got.data || {};
  const days = d.days || [];
  const models = d.models || [];
  // The best single MINUTE of any model (daily.py `peak_gen_tok_s`); absent when no minute was measured — never a 0.
  const peakOf = (m) => Math.max(0, ...(m.series || []).map((x) => x.peak_gen_tok_s).filter(isNum));
  const best = Math.max(0, ...models.map(peakOf));
  const head = `<p class="obs-sum">${models.length} model${models.length === 1 ? '' : 's'} · 24 h ${obsNum((d.total_24h || {}).gen_tokens)} tok`
    + `${best > 0 ? ` · peak min ${obsFig(best)} tok/s` : ''}`
    + `${d.malformed_total ? ` · <span class="bad">${obsNum(d.malformed_total)} unreadable lines</span>` : ''}</p>`
    + (d.error ? `<p class="obs-na bad">${esc(d.error)}</p>` : '');
  const axis = days.length ? `<div class="obs-axis"><span>${esc(days[0])}</span><span>${esc(days[days.length - 1])}</span></div>` : '';
  const rows = models.map((m) => {
    const s = m.series || [];
    const toks = s.map((x) => x.gen_tokens);
    const peak = peakOf(m);
    const most = Math.max(0, ...toks.filter(isNum));
    return `<section class="obs-model" data-model="${esc(m.id)}"><div class="obs-mh"><b>${esc(m.label || m.id)}</b>
        <span>24 h ${obsNum((m.last_24h || {}).gen_tokens)} tok · 30 d ${obsNum(m.gen_tokens_range)} tok`
      + `${peak > 0 ? ` · peak min ${obsFig(peak)} tok/s` : ''}</span></div>
      <div class="obs-brow"><span class="obs-bl">tokens/day · max ${obsShort(most)}</span>
        ${obsBarsSvg(toks, days, (v) => `${obsNum(v)} tok`)}</div>${axis}</section>`;
  }).join('');
  return obsFrame(title, head + rows);
}

// ── the DOM ──────────────────────────────────────────────────────────────────────────────────

const obsState = { view: null, box: null, tier: '48h', server: null, got: null, seq: 0 };

function obsDialog() {
  let d = document.getElementById('obs-dialog');
  if (!d) {
    d = document.createElement('dialog');
    d.id = 'obs-dialog';
    d.className = 'cn obs-dlg';
    d.setAttribute('aria-label', 'observability');
    d.addEventListener('click', (e) => {
      if (e.target === d || e.target.closest('.obs-x')) { d.close(); return; }
      if (e.target.closest('.obs-again')) obsOpen('check', obsState.box);
      const t = e.target.closest('.obs-tier');
      if (t) { obsState.tier = t.dataset.tier; obsOpen('history', obsState.box); }
    });
    d.addEventListener('close', () => { obsState.view = null; });
    document.body.appendChild(d);
  }
  return d;
}

function obsRender() {
  const d = obsDialog();
  const s = obsState;
  const html = s.view === 'check' ? obsCheckHtml(s.box, s.got)
    : s.view === 'history' ? obsHistoryHtml(s.box, s.tier, s.got, Date.now() / 1000)
      : obsDailyHtml(s.got, s.server);
  const focused = document.activeElement && d.contains(document.activeElement) ? document.activeElement : null;
  const sel = focused && focused.classList.contains('obs-tier') ? `.obs-tier[data-tier="${focused.dataset.tier}"]`
    : focused && focused.classList.contains('obs-again') ? '.obs-again' : null;
  d.innerHTML = html;
  const back = sel && d.querySelector(sel);
  if (back) back.focus({ preventScroll: true });
}

async function obsFetch(url) {
  try {
    const r = await fetch(url);
    const body = await r.json().catch(() => ({}));
    if (r.ok) return { data: body };
    const msg = body && body.error ? (typeof body.error === 'string' ? body.error : body.error.message) : '';
    return { error: `HTTP ${r.status}${msg ? `: ${msg}` : ''}` };
  } catch (e) {
    return { error: `not reachable: ${e.message || e}` };
  }
}

/** Open (or re-run) one view; a reply that lands after another view opened is dropped. */
async function obsOpen(view, box) {
  obsState.view = view;
  obsState.box = box;
  obsState.got = null;
  const seq = ++obsState.seq;
  const d = obsDialog();
  obsRender();
  if (!d.open) d.showModal();
  const q = encodeURIComponent;
  const url = view === 'check' ? `/lb/check?box=${q(box)}`
    : view === 'history' ? `/lb/history?box=${q(box)}&tier=${q(obsState.tier)}`
      : `/lb/perf/daily${obsState.server ? `?server=${q(obsState.server)}` : ''}`;
  const got = await obsFetch(url);
  if (seq !== obsState.seq || obsState.view !== view) return;
  obsState.got = got;
  obsRender();
}

/** The machine card's "check" and "history" buttons, once per card (gateway.js `renderMachines`). */
function observePaint(card, box) {
  if (!card || card.querySelector('.obs-tools')) return;
  const head = card.querySelector('.card-head');
  if (!head) return;
  const tools = document.createElement('span');
  tools.className = 'obs-tools';
  tools.innerHTML = `<button type="button" class="obs-btn" data-obs="check" aria-haspopup="dialog">check</button>`
    + `<button type="button" class="obs-btn" data-obs="history" aria-haspopup="dialog">history</button>`;
  tools.addEventListener('click', (e) => {
    const b = e.target.closest('[data-obs]');
    if (b) obsOpen(b.dataset.obs, box);
  });
  const id = head.querySelector('.box-id');
  head.insertBefore(tools, id ? id.nextSibling : null);
}

/** The selected server's "daily" link-button, beside performance · sessions · connect; its scope is the server the
 *  performance link names (gateway.js keeps that link's `?server=`). */
function obsDailyButton() {
  const links = document.getElementById('cluster-links');
  if (!links || links.querySelector('.obs-daily')) return;
  const btn = document.createElement('button');
  btn.type = 'button';
  // The card buttons' one neutral style (check / history / daily): connect's accent marks the one action that hands
  // something out; these three only look (the UX audit).
  btn.className = 'obs-btn obs-daily';
  btn.setAttribute('aria-haspopup', 'dialog');
  btn.textContent = 'daily';
  btn.addEventListener('click', () => {
    const perf = document.getElementById('cluster-perf');
    let server = null;
    try { server = perf ? new URL(perf.href, 'http://x').searchParams.get('server') : null; } catch (e) { server = null; }
    obsState.server = server;
    obsOpen('daily', null);
  });
  links.appendChild(document.createTextNode(' · '));
  links.appendChild(btn);
}

if (typeof document !== 'undefined' && document.addEventListener) {
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', obsDailyButton);
  else obsDailyButton();
}
