/* HEALTH AND ACTIVITY: the findings band at the top of the page, and the Activity tile — the fleet's
 * event feed, newest first, filtered by kind.
 *
 * THE BAND (`#lb-health`) is drawn from /lb/status `health` (vllm_lb/health.py `status_view`), which the page follows
 * on /lb/stream like every other key: one chip per finding PENDING or FIRING — its box, its figure against its
 * threshold, how long — and the alert sender's last failed send (alerts.py: never retried, so it is shown). Hidden
 * while nothing is pending, firing or failed. A member's status carries no `health` (roles.py allowlist): no band.
 *
 * THE ACTIVITY TILE (`#events-section`, `data-ly="events"`) reads GET /lb/events (vllm_lb/events.py): cluster state
 * changes, request kills, findings, alerts. Read when the tile is open — on open, every EVENTS_POLL_MS while open and
 * the page is visible, and at once when the findings change (a new event is likely). A 403 (a member) hides the tile.
 * Below the list, the health line: how many firing / pending / no data, the no-data reasons behind a disclosure.
 *
 * Figures, not sentences (operator rule): every label carries its number.
 */
'use strict';

const EVENTS_POLL_MS = 15000;
const EVENTS_PAGE = 60;
/** An alert failure older than this no longer holds the band open (it stays in the feed). */
const ALERT_FAIL_SHOWN_S = 3600;
/** The kind chips: one per family of the feed's kinds (events.py KINDS). */
const EVENT_GROUPS = [
  ['cluster', ['cluster.state']],
  ['kill', ['request.killed']],
  ['finding', ['finding.firing', 'finding.cleared']],
  ['alert', ['alert.sent', 'alert.failed']],
  ['power', ['power.action']],
];

function hEsc(x) {
  return String(x == null ? '' : x).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

/** "40s" · "3m" · "2h 5m" · "3d 4h" — an age in whole units, never negative. */
function healthAge(s) {
  s = Math.max(0, Math.floor(Number(s) || 0));
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m`;
  if (s < 86400) { const m = Math.floor((s % 3600) / 60); return `${Math.floor(s / 3600)}h${m ? ` ${m}m` : ''}`; }
  const h = Math.floor((s % 86400) / 3600);
  return `${Math.floor(s / 86400)}d${h ? ` ${h}h` : ''}`;
}

function hNum(v) {
  if (typeof v !== 'number' || !isFinite(v)) return String(v == null ? '—' : v);
  return Number.isInteger(v) ? v.toLocaleString('en-US') : v.toLocaleString('en-US', { maximumFractionDigits: 1 });
}

/** One finding as its chip says it: the figure against the threshold, in the finding's unit. */
function findingFigure(f) {
  if (f.kind === 'backend_down') return `${f.label} ${f.value}`;
  if (f.unit === 'new') return `${f.label} +${hNum(f.value)}`;
  // GPU TEMP is the driver's headroom below the GPU's own throttle limit, against the fleet's margin.
  if (f.kind === 'gpu_temp') return `${f.label} ${hNum(f.value)} °C to limit ≤ ${hNum(f.threshold)} °C`;
  const u = f.unit ? ` ${f.unit}` : '';
  // RAM AVAIL fires under its floor; GPU POWER at or under its type's idle draw + the margin (health.py).
  const op = f.kind === 'low_ram' ? '<' : f.kind === 'low_power' ? '≤' : 'vs';
  return `${f.label} ${hNum(f.value)}${u} ${op} ${hNum(f.threshold)}${u}`;
}

/** The band's chips, newest state first: [{cls, text, title}]. `now` is the status's own clock (s.now); `backends`
 *  (s.backends) names a SERVER finding's box — health.py keys it by the server's lane label, every other by box. */
function healthChips(health, now, backends) {
  if (!health || !Array.isArray(health.findings)) return [];
  const boxOf = new Map((backends || []).filter((b) => b && b.host_label && b.box).map((b) => [b.host_label, b.box]));
  const out = health.findings.map((f) => {
    const since = f.state === 'firing' && f.firing_since != null ? f.firing_since : f.since;
    const box = f.kind === 'backend_down' ? (boxOf.get(f.box) || f.box) : f.box;
    return { cls: f.state === 'firing' ? 'hl-firing' : 'hl-pending',
             text: `${f.state === 'firing' ? 'FIRING' : 'PENDING'} · ${box} · ${findingFigure(f)} · ${healthAge(now - since)}`,
             title: f.why || '' };
  });
  const fail = health.alerts && Array.isArray(health.alerts.recent_failures) ? health.alerts.recent_failures[0] : null;
  if (fail && now - fail.at < ALERT_FAIL_SHOWN_S) {
    out.push({ cls: 'hl-failed', text: `ALERT ${fail.channel} failed · ${fail.why} · ${healthAge(now - fail.at)}`,
               title: fail.title || '' });
  }
  return out;
}

/** The health line under the feed: "HEALTH 1 firing · 0 pending · 12 no data", or "HEALTH off". */
function healthSummary(health) {
  if (!health) return '';
  if (!health.enabled) return 'HEALTH off';
  const n = (st) => (health.findings || []).filter((f) => f.state === st).length;
  const alerts = health.alerts;
  const ch = alerts ? ` · ALERTS ${alerts.channels.length ? alerts.channels.map((c) => c.name).join('+') : 'none'} · ${hNum(alerts.sent)} sent · ${hNum(alerts.failed)} failed` : '';
  return `HEALTH ${n('firing')} firing · ${n('pending')} pending · ${(health.no_data || []).length} no data${ch}`;
}

/** An event's time: HH:MM:SS today, "Mon D HH:MM" before. */
function eventTime(t, today) {
  const d = new Date(t * 1000);
  const p = (x) => String(x).padStart(2, '0');
  const hms = `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
  const sameDay = today && d.getFullYear() === today.getFullYear() && d.getMonth() === today.getMonth()
    && d.getDate() === today.getDate();
  return sameDay ? hms : `${d.toLocaleString('en-US', { month: 'short' })} ${d.getDate()} ${hms.slice(0, 5)}`;
}

/** The tone of an event's kind tag. */
function eventTone(e) {
  if (e.kind === 'finding.firing' || e.kind === 'alert.failed') return 'bad';
  if (e.kind === 'cluster.state') return (e.state === 'failed' || e.state === 'stalled') ? 'bad' : e.state === 'running' ? 'ok' : '';
  if (e.kind === 'finding.cleared' || e.kind === 'alert.sent') return 'ok';
  if (e.kind === 'request.killed') return 'warn';
  return '';
}

function eventRowHtml(e, today) {
  const tone = eventTone(e);
  return `<li class="ev-row"><time class="ev-t" datetime="${hEsc(new Date(e.t * 1000).toISOString())}">${hEsc(eventTime(e.t, today))}</time>`
    + `<span class="ev-k${tone ? ` ev-${tone}` : ''}">${hEsc(e.kind)}</span>`
    + `<span class="ev-s">${hEsc(e.subject || '—')}</span>`
    + `<span class="ev-m" title="${hEsc(e.summary)}">${hEsc(e.summary)}</span></li>`;
}

/** The kinds a chip selection asks /lb/events for; null = every kind. */
function selectedKinds(groups) {
  if (!groups || !groups.size) return null;
  return EVENT_GROUPS.filter(([g]) => groups.has(g)).flatMap(([, ks]) => ks);
}

// ── the page ──────────────────────────────────────────────────────────────

const evState = { groups: new Set(), shown: EVENTS_PAGE, timer: null, sig: '', loading: false, hidden: false, last: null };

function renderHealthBand(s) {
  const el = typeof document !== 'undefined' && document.getElementById('lb-health');
  if (!el) return;
  const chips = healthChips(s && s.health, s && s.now, s && s.backends);
  el.hidden = !chips.length;
  el.innerHTML = chips.map((c) => `<span class="hl-chip ${c.cls}" title="${hEsc(c.title)}">${hEsc(c.text)}</span>`).join('');
}

function renderHealthLine(health) {
  const line = document.getElementById('events-health');
  if (!line) return;
  const nd = (health && health.no_data) || [];
  line.innerHTML = health ? `<span>${hEsc(healthSummary(health))}</span>`
    + (nd.length ? `<details class="ev-nd"><summary>no data · ${nd.length}</summary><ul>${nd.map((n) =>
      `<li><b>${hEsc(n.box)}</b> · ${hEsc(n.kind)} · ${hEsc(n.why)}</li>`).join('')}</ul></details>` : '')
    + (health.why && !health.enabled ? `<span class="ev-why">${hEsc(health.why)}</span>` : '') : '';
}

function renderEventChips() {
  const host = document.getElementById('events-kinds');
  if (!host) return;
  const all = !evState.groups.size;
  host.innerHTML = `<button type="button" class="ev-chip" data-g="" aria-pressed="${all}">all</button>`
    + EVENT_GROUPS.map(([g]) => `<button type="button" class="ev-chip" data-g="${g}" aria-pressed="${evState.groups.has(g)}">${g}</button>`).join('');
}

function renderEvents(body) {
  const list = document.getElementById('events-list');
  const count = document.getElementById('events-count');
  if (!list) return;
  const evs = (body && body.events) || [];
  const today = new Date();
  const shown = evs.slice(0, evState.shown);
  list.innerHTML = shown.map((e) => eventRowHtml(e, today)).join('')
    || `<li class="ev-empty">0 events${body && body.why ? ` · ${hEsc(body.why)}` : ''}</li>`;
  if (count) count.textContent = `${evs.length.toLocaleString('en-US')}${evs.length >= 500 ? '+' : ''}`;
  const more = document.getElementById('events-more');
  if (more) { more.hidden = evs.length <= evState.shown; more.textContent = `+${Math.min(EVENTS_PAGE, evs.length - evState.shown)}`; }
  const bad = document.getElementById('events-bad');
  if (bad) { const n = (body && body.bad || []).length; bad.hidden = !n; bad.textContent = n ? `${n} unreadable line${n === 1 ? '' : 's'}` : ''; }
}

async function loadEvents() {
  const sec = document.getElementById('events-section');
  if (!sec || evState.hidden || evState.loading || !sec.open) return;
  evState.loading = true;
  try {
    const kinds = selectedKinds(evState.groups);
    const r = await fetch(`/lb/events${kinds ? `?kind=${encodeURIComponent(kinds.join(','))}` : ''}`, { cache: 'no-store' });
    if (r.status === 403) { evState.hidden = true; sec.hidden = true; return; }
    const body = await r.json();
    if (!r.ok) throw new Error((body.error && body.error.message) || `HTTP ${r.status}`);
    evState.last = body;
    renderEvents(body);
  } catch (e) {
    const list = document.getElementById('events-list');
    if (list) list.innerHTML = `<li class="ev-empty ev-bad">events unreadable · ${hEsc(e.message)}</li>`;
  } finally {
    evState.loading = false;
  }
}

function eventsTimer() {
  const sec = document.getElementById('events-section');
  const on = sec && sec.open && !evState.hidden && document.visibilityState !== 'hidden';
  if (on && !evState.timer) evState.timer = setInterval(loadEvents, EVENTS_POLL_MS);
  if (!on && evState.timer) { clearInterval(evState.timer); evState.timer = null; }
}

function initEvents() {
  const sec = document.getElementById('events-section');
  if (!sec || sec.dataset.evInit) return;
  sec.dataset.evInit = '1';
  renderEventChips();
  sec.addEventListener('toggle', () => { eventsTimer(); if (sec.open) loadEvents(); });
  document.addEventListener('visibilitychange', () => { eventsTimer(); if (document.visibilityState === 'visible') loadEvents(); });
  document.getElementById('events-kinds').addEventListener('click', (ev) => {
    const b = ev.target.closest('button[data-g]');
    if (!b) return;
    const g = b.dataset.g;
    if (!g) evState.groups.clear();
    else if (evState.groups.has(g)) evState.groups.delete(g);
    else evState.groups.add(g);
    evState.shown = EVENTS_PAGE;
    renderEventChips();
    loadEvents();
  });
  const more = document.getElementById('events-more');
  if (more) more.addEventListener('click', () => { evState.shown += EVENTS_PAGE; renderEvents(evState.last); });
  eventsTimer();
  loadEvents();
}

/** gateway.js `renderStatus` calls this with every status. */
function renderHealth(s) {
  renderHealthBand(s);
  if (typeof document === 'undefined') return;
  const sec = document.getElementById('events-section');
  if (!sec) return;
  if (!s || !s.health) { sec.hidden = true; return; }       // a member: no health, no feed
  if (evState.hidden) return;
  sec.hidden = false;
  initEvents();
  renderHealthLine(s.health);
  // A finding that moved is a new event on the feed: read it now, not at the next poll.
  const sig = (s.health.findings || []).map((f) => `${f.id}:${f.state}`).join('|')
    + `|${s.health.alerts ? s.health.alerts.sent + s.health.alerts.failed : ''}`;
  if (sig !== evState.sig) { evState.sig = sig; loadEvents(); }
}

if (typeof module !== 'undefined') {
  module.exports = { healthAge, findingFigure, healthChips, healthSummary, eventTime, eventTone, eventRowHtml,
                     selectedKinds, EVENT_GROUPS };
}
