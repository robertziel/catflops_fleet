/* THE FIDELITY SECTION of /benchmarks:
 *
 *   GET  /lb/fidelity                 the runs, newest first, the corpus, and the run in progress (`active`)
 *   POST /lb/fidelity/run  {}         one run against the LOADED model — the operator's click only; 409 while one runs
 *   GET  /lb/fidelity/compare?a=&b=   paired ΔNLL = NLL(B) − NLL(A) with its 95 % bootstrap interval
 *
 * WHAT IT DRAWS. A Run button carrying its state ("Run" · "starting…" · "running 7/15"), a line of corpus figures, and
 * one table row per run: model · quant · KV dtype · mean NLL · perplexity · tokens · when, with engine × accelerator ×
 * TP under the model — and, when the LB could not read some of those, "<n> unread" opening to its reasons. A failed / interrupted / unreadable run is a row with its status and the server's words, and no
 * figure. Ticking two finished runs (the older is A, the reference) asks the LB for the paired comparison and prints
 * ΔNLL with its interval, and "same quality" when that interval spans 0. A figure the LB could not read is "—".
 *
 * NEVER ON ITS OWN. Nothing here POSTs except the button's click handler. The page polls GET /lb/fidelity only while
 * a run is active (every POLL_MS), so a run started from another tab is picked up on the next load, never started.
 *
 * Self-contained (own helpers, `fid` prefix): it loads after benchmarks.js but borrows nothing from it, so
 * tests/fidelity.node.js runs it alone.
 */
'use strict';

const FID_POLL_MS = 3000;
const fidEsc = (s) => String(s == null ? '' : s).replace(/[&<>"']/g, (c) =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const fidN = (v) => (typeof v === 'number' && Number.isFinite(v) ? v : null);
const fidFix = (v, d) => (fidN(v) == null ? '—' : v.toFixed(d));
const fidInt = (v) => (fidN(v) == null ? '—' : Math.round(v).toLocaleString('en-US'));
const fidSigned = (v, d) => (fidN(v) == null ? '—' : `${v > 0 ? '+' : v < 0 ? '−' : '±'}${Math.abs(v).toFixed(d)}`);

function fidAgo(ts, now) {
  const n = fidN(ts);
  if (n == null) return '—';
  const s = Math.max(0, now - n);
  if (s < 90) return `${Math.round(s)} s ago`;
  if (s < 5400) return `${Math.round(s / 60)} min ago`;
  if (s < 172800) return `${Math.round(s / 3600)} h ago`;
  return new Date(n * 1000).toISOString().slice(0, 16).replace('T', ' ');
}

/** The button's words and whether it may be pressed: a run in progress or a POST in flight disables it. */
function fidButton(view, posting) {
  const a = view && view.active;
  if (a) return { label: `running ${fidInt(a.done_docs)}/${fidInt(a.total_docs)}`, disabled: true };
  if (posting) return { label: 'starting…', disabled: true };
  return { label: 'Run', disabled: !(view && view.corpus) };
}

/** engine × accelerator × TP, each "—" when the LB could not read it. */
function fidTarget(meta) {
  const m = meta || {};
  return `${m.engine || '—'} × ${m.accelerator || '—'} × TP ${m.tp == null ? '—' : m.tp}`;
}

/** What the LB could not read about the run's model (`meta.why`, fidelity.describe_cluster): "<n> unread" — a figure,
 *  not a sentence — that opens on tap or keyboard (<details>) and hovers (title) to the reasons, each distinct reason
 *  once with the figures it covers. '' when nothing is unread. */
function fidWhyHtml(meta, id) {
  const why = (meta && meta.why && typeof meta.why === 'object') ? meta.why : {};
  const keys = Object.keys(why).filter((k) => why[k]);
  if (!keys.length) return '';
  const by = new Map();
  keys.forEach((k) => { const r = String(why[k]); by.set(r, [...(by.get(r) || []), k]); });
  const lines = [...by].map(([r, ks]) => [ks.join(', '), r]);
  return `<details class="fid-why" data-fid-why="${fidEsc(id)}"><summary title="${fidEsc(lines.map(([k, r]) => `${k}: ${r}`).join('\n'))}">`
    + `${keys.length} unread</summary><ul>${lines.map(([k, r]) => `<li><b>${fidEsc(k)}</b> · ${fidEsc(r)}</li>`).join('')}</ul></details>`;
}

/** The tbody rows. `sel` = the ticked run ids (at most two). */
function fidRowsHtml(runs, sel, now) {
  if (!runs || !runs.length) return '<tr class="fid-none"><td colspan="8">0 runs</td></tr>';
  return runs.map((r) => {
    const m = r.meta || {};
    const done = r.status === 'done';
    const ticked = done && sel.includes(r.id);
    const box = done
      ? `<input type="checkbox" data-fid-run="${fidEsc(r.id)}"${ticked ? ' checked' : ''} aria-label="compare ${fidEsc(r.id)}">`
      : '';
    const model = `<span class="fid-model">${fidEsc(m.model_label || m.served_name || r.id)}</span>`
      + `<span class="fid-target">${fidEsc(fidTarget(m))}</span>${fidWhyHtml(m, r.id)}`;
    // Every cell names its column (`data-label`): at phone width fidelity.css stacks a row into a card and prints it.
    const when = `<td data-label="when" title="${fidEsc(r.id)}">${fidEsc(fidAgo(r.started_at, now))}</td>`;
    if (!done) {
      const what = r.status === 'running' ? `running ${fidInt(r.done_docs)}/${fidInt(r.total_docs)}` : r.status;
      return `<tr class="fid-row fid-${fidEsc(r.status)}"><td class="fid-pick">${box}</td><td class="fid-m">${model}</td>`
        + `<td class="fid-wide" colspan="5"><span class="fid-status">${fidEsc(what)}</span>`
        + `${r.error ? ` <span class="fid-err">${fidEsc(r.error)}</span>` : ''}</td>${when}</tr>`;
    }
    return `<tr class="fid-row"${ticked ? ' aria-selected="true"' : ''}><td class="fid-pick">${box}</td><td class="fid-m">${model}</td>`
      + `<td data-label="quant">${fidEsc(m.quant || '—')}</td><td data-label="KV dtype">${fidEsc(m.kv_cache_dtype || '—')}</td>`
      + `<td class="num" data-label="mean NLL">${fidFix(r.mean_nll, 4)}</td><td class="num" data-label="perplexity">${fidFix(r.ppl, 3)}</td>`
      + `<td class="num" data-label="tokens">${fidInt(r.tokens)}</td>${when}</tr>`;
  }).join('');
}

/** The comparison line for two ticked runs; `cmp` is the LB's answer, or {error}, or null while it is asked. */
function fidCompareHtml(cmp, sel) {
  if (!sel || sel.length < 2) return `<p class="fid-cmp-hint">${sel && sel.length ? '1/2' : '0/2'} runs ticked</p>`;
  if (!cmp) return '<p class="fid-cmp">comparing…</p>';
  if (cmp.error) return `<p class="fid-cmp bad">${fidEsc(cmp.error)}</p>`;
  if (!cmp.comparable) return `<p class="fid-cmp bad">not comparable: ${fidEsc(cmp.reason)}</p>`;
  const [lo, hi] = cmp.ci95 || [];
  const verdict = cmp.same_quality ? 'same quality' : (cmp.d_nll > 0 ? 'B worse' : 'B better');
  return `<p class="fid-cmp ${cmp.same_quality ? 'same' : 'diff'}">`
    + `<b>ΔNLL ${fidSigned(cmp.d_nll, 4)}</b> · 95 % CI [${fidSigned(lo, 4)}, ${fidSigned(hi, 4)}] · `
    + `<span class="fid-verdict">${verdict}</span></p>`
    + `<p class="fid-cmp-figs">A ${fidFix(cmp.nll_a, 4)} · ppl ${fidFix(cmp.ppl_a, 3)} &nbsp; B ${fidFix(cmp.nll_b, 4)}`
    + ` · ppl ${fidFix(cmp.ppl_b, 3)} · ${fidInt(cmp.matched_docs)} docs · ${fidInt(cmp.tokens)} tokens`
    + `${cmp.mismatched_docs ? ` · ${fidInt(cmp.mismatched_docs)} docs tokenised differently` : ''}</p>`;
}

/** The corpus figures line: "15 docs · 9,812 bytes · sha 830241b5b7b2". */
function fidCorpusLine(view) {
  if (!view) return 'reading…';
  if (!view.corpus) return `corpus unavailable: ${view.corpus_error || 'unknown'}`;
  const c = view.corpus;
  return `${fidInt(c.docs)} docs · ${fidInt(c.bytes)} bytes · sha ${String(c.sha256 || '').slice(0, 12)}`;
}

/** The whole section body. `st` = {view, sel, cmp, posting, error, now}. */
function fidSectionHtml(st) {
  const b = fidButton(st.view, st.posting);
  const runs = (st.view && st.view.runs) || [];
  return `<div class="fid-head"><h2 id="fid-h">Fidelity</h2>`
    + `<button type="button" class="fid-run" data-fid="run"${b.disabled ? ' disabled' : ''}>${fidEsc(b.label)}</button>`
    + `<span class="fid-corpus">${fidEsc(fidCorpusLine(st.view))}</span></div>`
    + (st.error ? `<div class="alert fid-alert" role="alert">${fidEsc(st.error)}</div>` : '')
    + `<div class="tablewrap"><table class="runs-t fid-t"><thead><tr><th scope="col"></th><th scope="col">model</th>`
    + `<th scope="col">quant</th><th scope="col">KV dtype</th><th scope="col" class="num">mean NLL</th>`
    + `<th scope="col" class="num">perplexity</th><th scope="col" class="num">tokens</th><th scope="col">when</th>`
    + `</tr></thead><tbody>${fidRowsHtml(runs, st.sel, st.now)}</tbody></table></div>`
    + `<div class="fid-cmp-box" aria-live="polite">${fidCompareHtml(st.cmp, st.sel)}</div>`;
}

/** Tick a run: at most two, the oldest tick dropped first. Returns the new selection. */
function fidToggle(sel, id, on) {
  const out = sel.filter((x) => x !== id);
  if (on) out.push(id);
  return out.slice(-2);
}

/** A and B for the comparison: the OLDER run is A (the reference), whatever order they were ticked in. */
function fidPair(sel, runs) {
  const at = (id) => { const r = (runs || []).find((x) => x.id === id); return r ? (r.started_at || 0) : 0; };
  return sel.slice().sort((x, y) => at(x) - at(y));
}

// ── wiring (browser only) ───────────────────────────────────────────────────────────────────────────────────────

const fidSt = { view: null, sel: [], cmp: null, posting: false, error: '', now: Date.now() / 1000 };
let fidTimer = null;

function fidPaint() {
  const root = document.getElementById('fid-root');
  if (!root) return;
  fidSt.now = Date.now() / 1000;
  // A repaint (a compare, a poll) keeps the reasons the reader opened open.
  const open = new Set([...root.querySelectorAll('details.fid-why[open]')].map((d) => d.dataset.fidWhy));
  root.innerHTML = fidSectionHtml(fidSt);
  open.forEach((id) => root.querySelectorAll('details.fid-why').forEach((d) => { if (d.dataset.fidWhy === id) d.open = true; }));
}

async function fidLoad() {
  try {
    const r = await fetch('/lb/fidelity', { cache: 'no-store' });
    const body = await r.json().catch(() => null);
    if (!r.ok || !body) throw new Error(`HTTP ${r.status}`);
    fidSt.view = body;
    const done = new Set((body.runs || []).filter((x) => x.status === 'done').map((x) => x.id));
    fidSt.sel = fidSt.sel.filter((id) => done.has(id));
    if (fidSt.error.startsWith('/lb/fidelity')) fidSt.error = '';
  } catch (e) {
    fidSt.error = `/lb/fidelity: ${e.message}`;
  }
  fidPaint();
  clearTimeout(fidTimer);
  if (fidSt.view && fidSt.view.active) fidTimer = setTimeout(fidLoad, FID_POLL_MS);
}

async function fidCompare() {
  fidSt.cmp = null;
  fidPaint();
  if (fidSt.sel.length < 2) return;
  const [a, b] = fidPair(fidSt.sel, fidSt.view && fidSt.view.runs);
  try {
    const r = await fetch(`/lb/fidelity/compare?a=${encodeURIComponent(a)}&b=${encodeURIComponent(b)}`, { cache: 'no-store' });
    const body = await r.json().catch(() => null);
    fidSt.cmp = r.ok && body ? body : { error: (body && body.error && body.error.message) || `HTTP ${r.status}` };
  } catch (e) {
    fidSt.cmp = { error: e.message };
  }
  fidPaint();
}

async function fidStart() {
  fidSt.posting = true;
  fidSt.error = '';
  fidPaint();
  try {
    const r = await fetch('/lb/fidelity/run', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
    const body = await r.json().catch(() => null);
    if (!r.ok) fidSt.error = (body && body.error && body.error.message) || `HTTP ${r.status}`;
  } catch (e) {
    fidSt.error = e.message;
  }
  fidSt.posting = false;
  await fidLoad();
}

function fidInit() {
  const root = document.getElementById('fid-root');
  if (!root) return;
  root.addEventListener('click', (ev) => {
    const btn = ev.target.closest && ev.target.closest('button[data-fid="run"]');
    if (btn && !btn.disabled) fidStart();
  });
  root.addEventListener('change', (ev) => {
    const id = ev.target && ev.target.dataset && ev.target.dataset.fidRun;
    if (!id) return;
    fidSt.sel = fidToggle(fidSt.sel, id, ev.target.checked);
    fidCompare();
  });
  fidLoad();
}

if (typeof document !== 'undefined' && document.addEventListener) {
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', fidInit);
  else fidInit();
}
