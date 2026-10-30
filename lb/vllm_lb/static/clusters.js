/* THE SERVERS OF THIS RUN, as the page's tabs (worklog/parallel-clusters.md, "Servers, not clusters").
 *
 *   WORDS         to the operator a running engine is a SERVER and the boxes it runs on are the CLUSTER (the hardware
 *                 section below the server). The code, the API fields and the ids keep `cluster` for the engine — the
 *                 domain's name (status.clusters, /lb/clusters/<name>); renaming it is its own change.
 *   the TABS      ONE TAB PER SERVER in this run (its name, a state dot, the default one marked) and "+". A server tab
 *                 only SELECTS that server: gateway.js draws it (its card, its engine, its requests) in the one server
 *                 panel — there is no separate view. "+" is the one tab with its own panel: the fleet's DECLARED
 *                 servers, to activate one for this run / deactivate one that is off. WAI-ARIA tablist: aria-selected,
 *                 roving tabindex, arrow keys, Home/End; the hash keeps the selection (`#<name>`, `#add`).
 *   the SELECTION `selectedCluster`: which server gateway.js's `theCluster` means (its detail card, its switch, its head's
 *                 engine card, its requests). Set by a tab, or the first server of the run.
 *   KEYED RENDER  the tab buttons are rebuilt only when the run's names change; a changed state repaints its dot; so a
 *                 poll drops no focus. The + tab's cards likewise (`data-sig`): a refusal stays on screen.
 *   DEFAULT       the server a request naming no model goes to (fleet `default: true`) is marked on its tab and + row.
 *
 * Self-contained (its own escape, no gateway.js helper at load): tests/server_tabs.node.js loads it alone. Loaded before
 * gateway.js, which hands it every status it polls (`renderClusters`) and redraws on a selection (`renderStatus`).
 */
'use strict';

const CL_TONE = { off: 'off', running: 'ok', failed: 'bad', stalled: 'warn', unknown: 'warn' };
const CL_ADD = 'add';
const clEsc = (v) => String(v == null ? '' : v).replace(/[&<>"']/g,
  (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const clById = (id) => document.getElementById(id);
/** The DEFAULT server's chip (`default: true` in the view / the declared row): where an unnamed request goes. */
const clDefault = (c) => (c && c.default
  ? '<span class="cl-default" title="a request naming no model, or a name no server serves, goes here">default</span>' : '');
const clTabId = (name) => `tab-btn-s-${name}`;

/** The server gateway.js's detail card, engine card and requests draw (`theCluster`); null = the first of the run. */
let selectedCluster = null;
/** The run's server names, in order, as last rendered — the tab order is these then "+". */
let clNames = [];
/** The tab ids in order: a server's per name, then the + tab's. */
let clTabIds = ['tab-btn-add'];
/** Which tab shows: a server (the page) or the + panel. */
let clTab = 'server';
/** The last refusal per declared server (the + tab's actions), kept across polls until the next action on it. */
const clErrors = new Map();
/** Actions in flight, by server: their buttons stay disabled until the answer. */
const clPending = new Set();
let clStatus = null;

function selectCluster(name) { selectedCluster = name; }

// ── the tabs ──────────────────────────────────────────────────────────────────────────────────────

/** The aria state of every tab from `clTab` + `selectedCluster`; the hash; the + panel shown or hidden. */
function paintTabs(focus) {
  const current = clTab === CL_ADD ? 'tab-btn-add' : clTabId(selectedCluster);
  clTabIds.forEach((id) => {
    const btn = clById(id);
    if (!btn) return;
    const on = id === current;
    btn.setAttribute('aria-selected', on ? 'true' : 'false');
    btn.tabIndex = on ? 0 : -1;
    if (on && focus && btn.focus) btn.focus();
  });
  if (document.body && document.body.dataset) document.body.dataset.tab = clTab;
  const panel = clById('tab-add');
  if (panel) panel.hidden = clTab !== CL_ADD;
  try {
    const want = clTab === CL_ADD ? `#${CL_ADD}` : (selectedCluster ? `#${selectedCluster}` : '');
    if (typeof location !== 'undefined' && location && (location.hash || '') !== want) {
      if (typeof history !== 'undefined' && history.replaceState) history.replaceState(null, '', want || location.pathname || '/');
      else location.hash = want;
    }
  } catch (e) { /* no location to keep */ }
}

/** Select a tab: a server's name (the page then shows that server), or `add` (the + panel). An unknown name: nothing. */
function selectTab(name, focus) {
  if (name === CL_ADD) clTab = CL_ADD;
  else if (clNames.includes(name)) {
    clTab = 'server';
    const changed = selectedCluster !== name;
    selectedCluster = name;
    // gateway.js draws the selected server from the status it last polled — the one entry, so nothing is drawn twice.
    if (changed && typeof renderStatus === 'function' && typeof lastStatus !== 'undefined' && lastStatus) renderStatus(lastStatus);
  } else if (name !== 'server') return;
  paintTabs(focus);
}

/** Arrow keys move between tabs and select (WAI-ARIA tabs, automatic activation); Home/End to the ends. */
function tabKey(e) {
  const i = clTabIds.indexOf(String(e.target && e.target.id || ''));
  if (i < 0) return;
  const n = clTabIds.length;
  const next = { ArrowRight: (i + 1) % n, ArrowLeft: (i + n - 1) % n, Home: 0, End: n - 1 }[e.key];
  if (next == null) return;
  if (e.preventDefault) e.preventDefault();
  const id = clTabIds[next];
  selectTab(id === 'tab-btn-add' ? CL_ADD : clNames[next], true);
}

/** One server's tab: its state dot, its name, the default mark. The + tab is static markup (index.html). */
function serverTab(c) {
  const state = String(c.state || 'unknown');
  return `<button id="${clTabId(c.name)}" role="tab" type="button" aria-selected="false" aria-controls="cluster-panel" tabindex="-1" `
    + `data-server="${clEsc(c.name)}" data-state="${clEsc(state)}">`
    + `<i class="diode ${CL_TONE[state] || 'warn'}" aria-hidden="true"></i>${clEsc(c.name)}${clDefault(c)}</button>`;
}

/** The server tabs, keyed by the run's names: rebuilt when the names change, each repainted when its markup did. */
function renderTabs(list) {
  const host = clById('tablist');
  if (!host) return;
  const names = list.map((c) => c.name);
  const sig = names.join('\u0001');
  if (host.dataset.names !== sig) {
    const add = clById('tab-btn-add');
    // The + tab's own markup is kept (its count, its label); every server tab goes before it.
    host.innerHTML = list.map(serverTab).join('') + (add ? add.outerHTML || '' : '');
    host.dataset.names = sig;
  } else {
    list.forEach((c) => {
      const btn = clById(clTabId(c.name));
      if (!btn) return;
      const html = serverTab(c);
      if (btn.dataset.sig === html) return;
      btn.dataset.sig = html;
      const state = String(c.state || 'unknown');
      btn.dataset.state = state;
      btn.innerHTML = `<i class="diode ${CL_TONE[state] || 'warn'}" aria-hidden="true"></i>${clEsc(c.name)}${clDefault(c)}`;
    });
  }
  list.forEach((c) => { const btn = clById(clTabId(c.name)); if (btn && !btn.dataset.sig) btn.dataset.sig = serverTab(c); });
  clNames = names;
  clTabIds = names.map(clTabId).concat('tab-btn-add');
  if (!names.includes(selectedCluster)) selectedCluster = names[0] || null;
  paintTabs(false);
}

/** Every status gateway.js polls: the tabs from /lb/status.clusters (the run's order), the + tab from .declared. */
function renderClusters(s) {
  clStatus = s || null;
  const views = (s || {}).clusters;
  renderTabs(views && typeof views === 'object' ? Object.values(views) : []);
  // The + tab's count: the declared servers NOT in this run — what the tab offers to activate (or says why not).
  const n = clById('tab-n-add');
  const add = clById('tab-btn-add');
  const out = (Array.isArray((s || {}).declared) ? s.declared : []).filter((d) => !d.active).length;
  if (n) n.textContent = out ? String(out) : '';
  if (add) add.title = `declared servers not in this run: ${out}`;
  renderAdd(s);
}

// ── the + tab: the fleet's declared servers, for this run ─────────────────────────────────────────

/** Keyed render into `panel`: the set of articles is rebuilt only when the names change; each article's inside only
 *  when its markup changed — so a refusal, a disabled button and the focus survive the poll. */
function renderKeyed(panel, items, idOf, inner, outer) {
  const names = items.map(idOf);
  const sig = names.join('\u0001');
  if (panel.dataset.names !== sig) {
    panel.innerHTML = items.map(outer).join('');
    panel.dataset.names = sig;
    items.forEach((it) => { const el = clById(idOf(it)); if (el) el.dataset.sig = inner(it); });
    return;
  }
  items.forEach((it) => {
    const el = clById(idOf(it));
    if (!el) return;
    const html = inner(it);
    if (el.dataset.sig === html) return;
    el.dataset.sig = html;
    el.innerHTML = html;
  });
}

function declaredInner(d, s) {
  const err = clErrors.get(`+${d.name}`);
  const busy = clPending.has(d.name);
  const view = ((s || {}).clusters || {})[d.name] || null;
  const off = !!view && view.state === 'off';
  let action;
  if (d.active) {
    // Only an OFF server may leave the run (cluster_run.py, 409 otherwise): Deactivate is offered then and only then;
    // for any other state "in this run · <state>" is the reason, readable on touch too.
    action = `<span class="cl-run">in this run${view && view.state ? ` · ${clEsc(view.state)}` : ''}</span>`
      + (off ? `<button class="cl-act leave" type="button" data-act="deactivate" data-name="${clEsc(d.name)}"`
        + `${busy ? ' disabled' : ''}>Deactivate</button>` : '');
  } else if (d.unlaunchable) {
    // The fleet's fit refused its launch (fleet.py `unlaunchable`): it never activates, and the reason is the fleet's.
    action = `<span class="cl-blocked">cannot launch · ${clEsc(d.unlaunchable)}</span>`;
  } else if (d.holding) {
    // Not in this run, its stored switch still on: an engine an earlier run started may be on its boxes.
    action = '<span class="cl-blocked">not in this run · its switch is on — its boxes stay held; activate it to stop it</span>'
      + (d.available ? `<button class="cl-act join" type="button" data-act="activate" data-name="${clEsc(d.name)}"`
        + `${busy ? ' disabled' : ''}>Activate</button>` : '');
  } else if (d.available) {
    action = `<button class="cl-act join" type="button" data-act="activate" data-name="${clEsc(d.name)}"${busy ? ' disabled' : ''}>Activate</button>`;
  } else {
    // The API's `blocked_by`, grouped by holder: "blocked · node-1 + node-2 → pair".
    const by = new Map();
    (d.blocked_by || []).forEach((b) => by.set(b.cluster, (by.get(b.cluster) || []).concat(b.box)));
    action = `<span class="cl-blocked">blocked · ${[...by].map(([holder, boxes]) =>
      `${clEsc(boxes.join(' + '))} → ${clEsc(holder)}`).join('; ')}</span>`;
  }
  // TP as the API states it (`tp`, the fleet's derived size) — none where it states none, never the box count.
  const topo = [Number.isInteger(d.tp) ? `TP=${d.tp}` : '', (d.boxes || []).length ? `on ${d.boxes.join(' + ')}` : '']
    .filter(Boolean).join(' ');
  return `<div class="cl-head"><h3>${clEsc(d.name)}</h3><span class="cl-control">${clEsc(d.control || '')}</span>`
    + `<span class="cl-topo">${clEsc(topo)}</span>${clDefault(d)}`
    + (d.boot ? '<span class="cl-boot" title="runs when the gateway starts">at boot</span>' : '') + '</div>'
    + (d.model ? `<div class="cl-model">${clEsc(d.model)}</div>` : '')
    + `<div class="cl-actions">${action}</div>`
    + `<p class="cl-error" role="status"${err ? '' : ' hidden'}>${clEsc(err || '')}</p>`;
}

function declaredCard(d, s) {
  return `<article class="cl-card ${d.active ? 'ok' : (d.available ? 'off' : 'warn')}" id="add-${clEsc(d.name)}" `
    + `data-cluster="${clEsc(d.name)}" data-available="${d.available ? 'true' : 'false'}" data-active="${d.active ? 'true' : 'false'}" `
    + `aria-label="declared server ${clEsc(d.name)}">${declaredInner(d, s)}</article>`;
}

function renderAdd(s) {
  const panel = clById('add-panel');
  if (!panel) return;
  const list = Array.isArray((s || {}).declared) ? s.declared : [];
  const empty = clById('add-empty');
  if (empty) empty.hidden = !!list.length;
  if (!list.length) { panel.innerHTML = ''; panel.dataset.names = ''; return; }
  renderKeyed(panel, list, (d) => `add-${d.name}`, (d) => declaredInner(d, s), (d) => declaredCard(d, s));
}

/** The run's actions, POSTed to the API; its refusal is kept on the card in its own words until the next action. */
function postRunAction(name, action) {
  clPending.add(name);
  clErrors.delete(`+${name}`);
  if (clStatus) renderAdd(clStatus);
  return fetch(`/lb/clusters/${encodeURIComponent(name)}/${action}`, { method: 'POST',
    headers: { 'Content-Type': 'application/json' }, body: '{}' })
    .then(async (r) => {
      if (!r.ok) {
        let msg = `HTTP ${r.status}`;
        try { const j = await r.json(); msg = (j.error || {}).message || msg; } catch (e) { /* the status is the word */ }
        clErrors.set(`+${name}`, msg);
      }
    })
    .catch((e) => { clErrors.set(`+${name}`, String(e && e.message || e)); })
    .then(() => { clPending.delete(name); if (clStatus) renderAdd(clStatus); });
}

(() => {
  const list = clById('tablist');
  if (list && list.addEventListener) {
    list.addEventListener('click', (e) => {
      const b = e.target && e.target.closest ? e.target.closest('[role="tab"]') : null;
      if (b) selectTab(b.id === 'tab-btn-add' ? CL_ADD : b.dataset.server, false);
    });
    list.addEventListener('keydown', tabKey);
  }
  const add = clById('add-panel');
  if (add && add.addEventListener) {
    add.addEventListener('click', (e) => {
      const b = e.target && e.target.closest ? e.target.closest('[data-act]') : null;
      if (b && !b.disabled) postRunAction(b.dataset.name, b.dataset.act);
    });
  }
  // The hash at load: `#add` opens the + panel; `#<name>` is the selection to keep once the run's names are known
  // (`renderTabs` keeps it if the run has it). shortcut: a server literally named "add" is shadowed by the + tab's hash.
  let hash = '';
  try { hash = String((typeof location !== 'undefined' && location && location.hash) || '').replace('#', ''); } catch (e) { hash = ''; }
  if (hash === CL_ADD) clTab = CL_ADD;
  else if (hash) selectedCluster = decodeURIComponent(hash);
  paintTabs(false);
})();
