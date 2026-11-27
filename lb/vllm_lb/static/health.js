/* HEALTH AND ACTIVITY: the findings band at the top of the page, and the Activity tile — the fleet's
 * event feed, newest first, filtered by kind.
 *
 * THE BAND (`#lb-health`) is drawn from /lb/status `health` (vllm_lb/health.py `status_view`), which the page follows
 * on /lb/stream like every other key. ONE CHIP PER BOX AND STATE (a chip per finding, all one red, one per
 * row on a phone, took 36 % of a 390 px first screen; a box with both firing and pending findings has one chip of
 * each): every box with a finding FIRING, then every box with one PENDING (its own warn tone) — the chip says each
 * finding's figure; its threshold, age and why are the chip's title — and, LAST, the alert sender's last failed send
 * (alerts.py: never retried, so it is shown) in the warn tone: it is the alert path's trouble, not a box's fault. A
 * chip is a button: it flashes the box's machine card and scrolls to it, and for a kind the flight recorder's strip
 * details (REC_KINDS) opens that strip through recorder.js `recOpen(box)` — whose scroll is then the tap's only one (it
 * centres the strip line under the flashed card). A box with no strip drawn (recorder.js `recWhyNot`), or a page
 * without recorder.js: the card's scroll and flash, and the reason on the chip's title — never a throw in the click
 * handler. A phone (≤ 600 px) shows 2 rows of chips and a "+n" that shows the rest (aria-expanded). Hidden while
 * nothing is pending, firing or failed. A member's status carries no `health` (roles.py allowlist): no band. Each
 * machine card's head carries the count of its box's FIRING findings, in the fault tone only above 0.
 *
 * THE ACTIVITY TILE (`#events-section`, `data-ly="events"`, under the band at the fleet's level) reads GET /lb/events
 * (vllm_lb/events.py): cluster state changes, request kills, findings, alerts. An alert send is FOLDED into the event
 * it sent (`foldAlerts`: a "✗ webhook" / "✓ ntfy" mark on that row), never a row of its own — unless the event it was
 * about is not on the page. Collapsed to the 3 newest rows with an "<n> ›" that shows the rest; the choice is this
 * browser's (localStorage `lb.activity`, every touch wrapped: a browser that keeps nothing starts collapsed). Read when
 * the tile is open — on open, every EVENTS_POLL_MS while open and the page is visible, and at once when the findings
 * change (a new event is likely). A 403 (a member) hides the tile. Below the list, the health line: what the band does
 * NOT say — how many readings have no data (the reasons behind a disclosure), the alert channels and their counts.
 *
 * Figures, not sentences (operator rule): every label carries its number.
 */
'use strict';

const EVENTS_POLL_MS = 15000;
const EVENTS_PAGE = 60;
/** An alert failure older than this no longer holds the band open (it stays in the feed). */
const ALERT_FAIL_SHOWN_S = 3600;
/** Rows the Activity tile shows collapsed: the newest few, enough to see what just happened. */
const ACT_COLLAPSED = 3;
/** This browser's open / collapsed choice for the Activity tile. */
const ACT_KEY = 'lb.activity';
/** Finding kinds the machine card's flight-recorder strip details (recorder.js + hwextra.js: the RoCE ports' counters,
 *  the disk rows the `disk` finding flags). Every one is a health.py KINDS entry (tests/health_ui.node.js). */
const REC_KINDS = ['roce', 'disk'];
/** A phone's band: this many rows of chips, then "+n". */
const BAND_PHONE_ROWS = 2;
const BAND_PHONE_MQ = '(max-width: 600px)';
/** How long a jumped-to machine card is outlined. */
const FLASH_MS = 1600;
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

/** One finding as its TITLE says it: the figure against the threshold, in the finding's unit. */
function findingFigure(f) {
  if (f.kind === 'backend_down') return `${f.label} ${f.value}`;
  if (f.unit === 'new') return `${f.label} +${hNum(f.value)}`;
  // GPU TEMP is the driver's headroom below the GPU's own throttle limit, against the fleet's margin.
  if (f.kind === 'gpu_temp') return `${f.label} ${hNum(f.value)} °C to limit ≤ ${hNum(f.threshold)} °C`;
  const u = f.unit ? ` ${f.unit}` : '';
  // RAM AVAIL fires under its floor; GPU POWER at or under its type's idle draw + the margin; DISK at or over the
  // fleet's disk_full_pct (health.py).
  const op = f.kind === 'low_ram' ? '<' : f.kind === 'low_power' ? '≤' : f.kind === 'disk' ? '≥' : 'vs';
  return `${f.label} ${hNum(f.value)}${u} ${op} ${hNum(f.threshold)}${u}`;
}

/** One finding as its CHIP says it: the figure alone — the threshold is the title's (`findingFigure`). GPU TEMP keeps
 *  "to limit": its figure is the headroom below the GPU's throttle limit, and "GPU TEMP 3 °C" would read as a
 *  temperature. */
function findingValue(f) {
  if (f.kind === 'backend_down') return `${f.label} ${f.value}`;
  if (f.unit === 'new') return `${f.label} +${hNum(f.value)}`;
  if (f.kind === 'gpu_temp') return `${f.label} ${hNum(f.value)} °C to limit`;
  return `${f.label} ${hNum(f.value)}${f.unit ? ` ${f.unit}` : ''}`;
}

/** The box a finding is about: health.py keys a SERVER finding by the server's lane label (`backends[].host_label`),
 *  every other by box — the status's backends name that lane's box. */
function findingBox(f, backends) {
  if (f.kind !== 'backend_down') return f.box;
  const b = (backends || []).find((x) => x && x.host_label === f.box && x.box);
  return b ? b.box : f.box;
}

/** The band's chips: [{cls, box, kinds, text, label, title}] — one per box and state, every FIRING box before every
 *  PENDING one (health.py's order inside each: oldest first), then the last failed alert send (box null). `now` is the
 *  status's own clock (s.now); `backends` (s.backends) names a SERVER finding's box. */
function healthChips(health, now, backends) {
  if (!health || !Array.isArray(health.findings)) return [];
  const groups = new Map();
  health.findings.forEach((f) => {
    const box = findingBox(f, backends);
    const key = `${f.state === 'firing' ? 0 : 1}|${box}`;
    if (!groups.has(key)) groups.set(key, { state: f.state === 'firing' ? 'firing' : 'pending', box, items: [] });
    groups.get(key).items.push(f);
  });
  const out = [...groups.values()].sort((a, b) => (a.state === b.state ? 0 : a.state === 'firing' ? -1 : 1)).map((g) => {
    const text = `${g.box} · ${g.items.map(findingValue).join(' · ')}`;
    return { cls: g.state === 'firing' ? 'hl-firing' : 'hl-pending', box: g.box, kinds: g.items.map((f) => f.kind),
             text, label: `${g.state} · ${text}`,
             title: g.items.map((f) => {
               const since = f.state === 'firing' && f.firing_since != null ? f.firing_since : f.since;
               return `${f.state === 'firing' ? 'FIRING' : 'PENDING'} ${findingFigure(f)} · ${healthAge(now - since)}`
                 + (f.why ? ` · ${f.why}` : '');
             }).join('\n') };
  });
  const fail = health.alerts && Array.isArray(health.alerts.recent_failures) ? health.alerts.recent_failures[0] : null;
  if (fail && now - fail.at < ALERT_FAIL_SHOWN_S) {
    const text = `ALERT ${fail.channel} failed · ${fail.why} · ${healthAge(now - fail.at)}`;
    out.push({ cls: 'hl-failed', box: null, kinds: [], text, label: text, title: fail.title || '' });
  }
  return out;
}

/** box → how many of its findings are FIRING (the machine card's badge); a SERVER finding counts on its box. */
function firingByBox(health, backends) {
  const out = new Map();
  ((health && health.findings) || []).filter((f) => f.state === 'firing').forEach((f) => {
    const box = findingBox(f, backends);
    out.set(box, (out.get(box) || 0) + 1);
  });
  return out;
}

/** How many leading chips sit in the first `rows` rows of a wrapping band, from each chip's offsetTop (non-decreasing
 *  in a flex-wrap row; a couple of pixels apart is one row — sub-pixel rounding). */
function bandCut(tops, rows) {
  let row = 0;
  let top = null;
  for (let i = 0; i < tops.length; i += 1) {
    if (top === null) top = tops[i];
    else if (tops[i] - top > 2) { row += 1; top = tops[i]; }
    if (row >= rows) return i;
  }
  return tops.length;
}

/** A chip's tap: the box's machine card (its Cluster section opened) outlined for FLASH_MS, and for a kind the flight
 *  recorder details, its strip opened through `rec` = {open: recOpen, whyNot: recWhyNot} (recorder.js; null where
 *  that file is not loaded). ONE SCROLL: recOpen's when the strip opens (it centres the strip's line, the flashed card
 *  just above it) — two smooth scrolls cancel, and the card's own block:'center' left a tall card's flash above the
 *  view; otherwise the card's, centred. `later` is setTimeout (injected for the harness).
 *  → {card: false} when the panel draws no card for that box; else {card: true, strip}: 'opened', null (not a strip
 *  kind), or why no strip was opened — for the chip's title (`chipSays`). */
function jumpToBox(doc, box, kinds, rec, later) {
  const card = [...doc.querySelectorAll('article.card.machine[data-box]')].find((a) => a.dataset.box === box);
  if (!card) return { card: false, strip: null };
  const sec = card.closest('details');
  if (sec && !sec.open) sec.open = true;
  const wants = kinds.some((k) => REC_KINDS.includes(k));
  const why = !wants ? '' : rec ? rec.whyNot(box) : `no recorder strip for ${box}: recorder.js is not loaded`;
  const opens = wants && !why;
  if (!opens) {
    const still = typeof window !== 'undefined' && window.matchMedia
      && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    card.scrollIntoView({ behavior: still ? 'auto' : 'smooth', block: 'center' });
  }
  card.classList.remove('hl-flash');
  void card.offsetWidth;                                   // restart the outline's animation on a second tap
  card.classList.add('hl-flash');
  (later || setTimeout)(() => card.classList.remove('hl-flash'), FLASH_MS);
  if (opens) rec.open(box);
  return { card: true, strip: opens ? 'opened' : why || null };
}

/** A tap's reason the strip did not open, on the chip's title (a line of its own, once however often it is tapped):
 *  the state is visible where the user tapped, not swallowed. */
function chipSays(chip, why) {
  if (why && !(chip.title || '').split('\n').includes(why)) chip.title = chip.title ? `${chip.title}\n${why}` : why;
}

/** The health line under the feed — what the band does not say: "HEALTH 12 no data · ALERTS ntfy · 3 sent · 0 failed",
 *  or "HEALTH off". The firing / pending counts are the band's chips and the cards' badges, so not repeated here. */
function healthSummary(health) {
  if (!health) return '';
  if (!health.enabled) return 'HEALTH off';
  const alerts = health.alerts;
  const ch = alerts ? ` · ALERTS ${alerts.channels.length ? alerts.channels.map((c) => c.name).join('+') : 'none'} · ${hNum(alerts.sent)} sent · ${hNum(alerts.failed)} failed` : '';
  return `HEALTH ${(health.no_data || []).length} no data${ch}`;
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
  if (e.kind === 'finding.firing') return 'bad';
  // A failed send is the alert path's trouble, not a box's fault: the warn tone, as its band chip.
  if (e.kind === 'alert.failed') return 'warn';
  if (e.kind === 'cluster.state') return (e.state === 'failed' || e.state === 'stalled') ? 'bad' : e.state === 'running' ? 'ok' : '';
  if (e.kind === 'finding.cleared' || e.kind === 'alert.sent') return 'ok';
  if (e.kind === 'request.killed') return 'warn';
  return '';
}

const isAlert = (e) => e.kind === 'alert.sent' || e.kind === 'alert.failed';

/** The feed with each alert send FOLDED into the event it sent: a copy of every other event, carrying `alerts`
 *  [{channel, ok, why}] (newest send first). alerts.py records `about` (the event's id, or its subject) and `alerted`
 *  (its kind); a send is always newer than its event, so walking newest → oldest meets the send first and hands it to
 *  the next event of that kind and key — a re-fired finding's sends land on their own firing. A send whose event is
 *  not on this page (older than the page, or filtered out by the kind chips) stays a row of its own. */
function foldAlerts(evs) {
  const waiting = new Map();
  const out = [];
  (evs || []).forEach((e) => {
    if (isAlert(e) && e.about != null && e.alerted) {
      const row = { ...e };
      out.push(row);
      const key = `${e.alerted}|${e.about}`;
      if (!waiting.has(key)) waiting.set(key, []);
      waiting.get(key).push(row);
      return;
    }
    const row = { ...e, alerts: [] };
    const key = `${e.kind}|${e.id != null ? e.id : e.subject}`;
    (waiting.get(key) || []).forEach((a) => {
      a.folded = true;
      row.alerts.push({ channel: a.channel, ok: a.kind === 'alert.sent', why: a.why || '' });
    });
    waiting.delete(key);
    out.push(row);
  });
  return out.filter((r) => !r.folded);
}

/** What the Activity list shows of `n` rows: collapsed, the ACT_COLLAPSED newest and how many the expander hides;
 *  open, a page of `page` (the "+60" pages further). */
function activityView(n, collapsed, page) {
  if (collapsed) return { shown: Math.min(n, ACT_COLLAPSED), rest: Math.max(0, n - ACT_COLLAPSED) };
  return { shown: Math.min(n, page), rest: 0 };
}

/** This browser's Activity choice: open (true) or collapsed (false, also a fresh or a keep-nothing browser). Every
 *  storage touch is wrapped — a blocked origin or a full quota throws from localStorage, and the tile must still draw. */
function actOpenGet(storage) {
  try {
    const raw = storage ? storage.getItem(ACT_KEY) : null;
    return !!(raw && JSON.parse(raw).open === true);
  } catch (e) {
    return false;
  }
}

function actOpenSet(storage, open) {
  try { if (storage) storage.setItem(ACT_KEY, JSON.stringify({ open: !!open })); } catch (e) { /* the choice lasts the visit */ }
}

function eventRowHtml(e, today) {
  const tone = eventTone(e);
  const marks = (e.alerts || []).map((a) => `<span class="ev-al ev-${a.ok ? 'ok' : 'warn'}" title="${hEsc(`${a.channel} ${a.ok ? 'sent' : 'failed'}${a.why ? ` · ${a.why}` : ''}`)}">`
    + `${a.ok ? '✓' : '✗'} ${hEsc(a.channel)}</span>`).join('');
  return `<li class="ev-row"><time class="ev-t" datetime="${hEsc(new Date(e.t * 1000).toISOString())}">${hEsc(eventTime(e.t, today))}</time>`
    + `<span class="ev-k${tone ? ` ev-${tone}` : ''}">${hEsc(e.kind)}</span>`
    + `<span class="ev-s">${hEsc(e.subject || '—')}</span>`
    + `<span class="ev-m"><span title="${hEsc(e.summary)}">${hEsc(e.summary)}</span>${marks}</span></li>`;
}

/** The kinds a chip selection asks /lb/events for; null = every kind. */
function selectedKinds(groups) {
  if (!groups || !groups.size) return null;
  return EVENT_GROUPS.filter(([g]) => groups.has(g)).flatMap(([, ks]) => ks);
}

// ── the page ──────────────────────────────────────────────────────────────

const actStorage = (() => { try { return typeof window !== 'undefined' ? window.localStorage : null; } catch (e) { return null; } })();
const evState = { groups: new Set(), shown: EVENTS_PAGE, timer: null, sig: '', loading: false, hidden: false, last: null,
                  open: actOpenGet(actStorage) };
/** The band: what it last drew (chips are kept across a status that only ages them, so a focused chip keeps its
 *  focus), and whether a phone's "+n" is open (this visit only). */
/** `said`: box → why a tap opened no strip for it (jumpToBox), written into that box's chip titles by every
 *  renderHealthBand until the box has a strip again (recorder.js `recWhyNot` → '') — a status redraw seconds later
 *  must not wipe the reason the tap just showed. */
const bandState = { sig: '', open: false, wired: false, said: new Map() };

/** Keep `why` for `box`'s chips (see bandState.said). */
function bandSay(box, why) {
  if (box != null && why) bandState.said.set(box, why);
}

/** A chip's title as drawn: its findings' lines, then the box's kept tap reason, if any. */
const bandTitle = (c) => (c.box != null && bandState.said.has(c.box)
  ? `${c.title}\n${bandState.said.get(c.box)}` : c.title);

const onPhone = () => typeof window !== 'undefined' && !!window.matchMedia && window.matchMedia(BAND_PHONE_MQ).matches;

/** A phone's band to BAND_PHONE_ROWS rows: the chips past them hidden and the "+n" shown — measured, because a chip's
 *  width is its text's. Open, every chip and a "−n" that folds them again. */
function fitBand(el) {
  const more = el.querySelector('.hl-more');
  const chips = [...el.querySelectorAll('.hl-chip')];
  chips.forEach((c) => { c.hidden = false; });
  if (!more) return;
  more.hidden = true;
  if (!onPhone()) return;
  let keep = bandCut(chips.map((c) => c.offsetTop), BAND_PHONE_ROWS);
  if (keep >= chips.length) return;
  const hide = chips.length - keep;
  more.hidden = false;
  if (bandState.open) {
    more.textContent = `−${hide}`;
    more.setAttribute('aria-expanded', 'true');
    more.setAttribute('aria-label', `show ${keep} chips`);
    return;
  }
  more.setAttribute('aria-expanded', 'false');
  // The "+n" takes room on the last kept row: give back chips until it fits there.
  for (;;) {
    chips.forEach((c, i) => { c.hidden = i >= keep; });
    more.textContent = `+${chips.length - keep}`;
    more.setAttribute('aria-label', `show ${chips.length - keep} more chips`);
    const tops = chips.slice(0, keep).map((c) => c.offsetTop).concat(more.offsetTop);
    if (keep === 0 || bandCut(tops, BAND_PHONE_ROWS) === tops.length) break;
    keep -= 1;
  }
}

function wireBand(el) {
  if (bandState.wired) return;
  bandState.wired = true;
  el.addEventListener('click', (ev) => {
    if (ev.target.closest('.hl-more')) { bandState.open = !bandState.open; fitBand(el); return; }
    const chip = ev.target.closest('.hl-chip[data-box]');
    if (chip) {
      const got = jumpToBox(document, chip.dataset.box, (chip.dataset.kinds || '').split(' '),
        typeof recOpen === 'function' && typeof recWhyNot === 'function' ? { open: recOpen, whyNot: recWhyNot } : null);
      if (got.strip && got.strip !== 'opened') { bandSay(chip.dataset.box, got.strip); chipSays(chip, got.strip); }
    }
  });
  if (window.matchMedia) {
    const mq = window.matchMedia(BAND_PHONE_MQ);
    if (mq.addEventListener) mq.addEventListener('change', () => fitBand(el));
  }
  window.addEventListener('resize', () => fitBand(el));
}

function renderHealthBand(s) {
  const el = typeof document !== 'undefined' && document.getElementById('lb-health');
  if (!el) return;
  const chips = healthChips(s && s.health, s && s.now, s && s.backends);
  el.hidden = !chips.length;
  // A kept reason goes once its box has a strip (the recorder says so); without recorder.js it stays true.
  for (const box of [...bandState.said.keys()]) {
    if (typeof recWhyNot === 'function' && !recWhyNot(box)) bandState.said.delete(box);
  }
  const sig = chips.map((c) => `${c.cls}|${c.box}|${c.text}`).join('\n');
  if (sig === bandState.sig) {
    // Only the ages moved: the titles are rewritten in place, the buttons (and the focus on one) stay.
    const els = el.querySelectorAll('.hl-chip');
    chips.forEach((c, i) => {
      if (!els[i]) return;
      els[i].title = bandTitle(c);
      if (c.box == null) els[i].textContent = c.text;
    });
    return;
  }
  bandState.sig = sig;
  el.innerHTML = chips.map((c) => (c.box == null
    ? `<span class="hl-chip ${c.cls}" title="${hEsc(bandTitle(c))}">${hEsc(c.text)}</span>`
    : `<button type="button" class="hl-chip ${c.cls}" data-box="${hEsc(c.box)}" data-kinds="${hEsc(c.kinds.join(' '))}"`
      + ` title="${hEsc(bandTitle(c))}" aria-label="${hEsc(c.label)}">${hEsc(c.text)}</button>`)).join('')
    + (chips.length ? '<button type="button" class="hl-more" aria-expanded="false" hidden></button>' : '');
  wireBand(el);
  fitBand(el);
}

/** Each machine card's head: its box's FIRING count, the fault tone only above 0. Drawn after gateway.js
 *  `renderMachines` (renderStatus's order), beside `.box-id` — outside what that function repaints. */
function renderBoxCounts(s) {
  if (typeof document === 'undefined') return;
  const health = s && s.health;
  const counts = firingByBox(health, s && s.backends);
  document.querySelectorAll('#lb-backends article.card.machine[data-box]').forEach((card) => {
    const head = card.querySelector('.card-head');
    if (!head) return;
    let badge = head.querySelector('.hl-count');
    if (!health) { if (badge) badge.remove(); return; }
    if (!badge) {
      badge = document.createElement('span');
      badge.className = 'hl-count';
      const id = head.querySelector('.box-id');
      head.insertBefore(badge, id ? id.nextSibling : head.firstChild);
    }
    const n = counts.get(card.dataset.box) || 0;
    const text = String(n);
    if (badge.textContent !== text) badge.textContent = text;
    badge.classList.toggle('hl-count-bad', n > 0);
    badge.title = `${n} firing`;
    badge.setAttribute('aria-label', `${n} firing findings`);
  });
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

/** The Activity count: the rows drawn (`rows`, after `foldAlerts`), "+" when the server's page was CUT — judged on the
 *  RAW page (`body.events`) against the limit the server states (events.py `read` → `limit`), because folding a full
 *  page's alert sends into their events leaves fewer rows than the cut and would hide it. No stated limit: no cut. */
function activityCount(body, rows) {
  const raw = ((body && body.events) || []).length;
  const cut = body && typeof body.limit === 'number' && raw >= body.limit;
  return `${rows.toLocaleString('en-US')}${cut ? '+' : ''}`;
}

function renderEvents(body) {
  const list = document.getElementById('events-list');
  const count = document.getElementById('events-count');
  if (!list) return;
  const evs = foldAlerts((body && body.events) || []);
  const today = new Date();
  const view = activityView(evs.length, !evState.open, evState.shown);
  const shown = evs.slice(0, view.shown);
  list.innerHTML = shown.map((e) => eventRowHtml(e, today)).join('')
    || `<li class="ev-empty">0 events${body && body.why ? ` · ${hEsc(body.why)}` : ''}</li>`;
  if (count) count.textContent = activityCount(body, evs.length);
  const more = document.getElementById('events-more');
  if (more) { more.hidden = !evState.open || evs.length <= evState.shown; more.textContent = `+${Math.min(EVENTS_PAGE, evs.length - evState.shown)}`; }
  const expand = document.getElementById('events-expand');
  if (expand) {
    expand.hidden = evs.length <= ACT_COLLAPSED;
    expand.setAttribute('aria-expanded', String(evState.open));
    expand.textContent = evState.open ? `‹ ${ACT_COLLAPSED}` : `${view.rest.toLocaleString('en-US')} ›`;
    expand.setAttribute('aria-label', evState.open ? `show the ${ACT_COLLAPSED} newest` : `show ${view.rest} more events`);
  }
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
  const expand = document.getElementById('events-expand');
  if (expand) {
    expand.addEventListener('click', () => {
      evState.open = !evState.open;
      evState.shown = EVENTS_PAGE;
      actOpenSet(actStorage, evState.open);
      renderEvents(evState.last);
    });
  }
  eventsTimer();
  loadEvents();
}

/** gateway.js `renderStatus` calls this with every status. */
function renderHealth(s) {
  renderHealthBand(s);
  renderBoxCounts(s);
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
  module.exports = { healthAge, findingFigure, findingValue, healthChips, firingByBox, bandCut, jumpToBox, chipSays,
                     bandSay, renderHealthBand, healthSummary, eventTime, eventTone, eventRowHtml, foldAlerts,
                     activityView, activityCount, actOpenGet, actOpenSet, selectedKinds, EVENT_GROUPS, REC_KINDS };
}
