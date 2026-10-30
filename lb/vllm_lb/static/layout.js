/* THE PANEL'S LAYOUT (operator, 2026-10-30: "implement custom UX UI, you can choose template in config, you can
 * arrange elements in the view, move around, make it all stateless, per browser that keeps state data").
 *
 *   THE ELEMENTS    every movable part of the panel is a host in index.html carrying `data-ly="<id>"` and a
 *                   `data-ly-label`: the registry is the markup, read from the DOM at boot — never a second list.
 *                   gateway.js keeps drawing into those hosts by id wherever they stand: this file MOVES the nodes
 *                   (appendChild keeps every listener, every `paint` signature, the power control's observer) and
 *                   never rebuilds them, so a poll lands in a card the operator just dragged.
 *   A LAYOUT        rows, each of up to three columns, each column an ordered list of items (an id and a size,
 *                   full or half of its column); and the ids hidden. A column's width is a token — `side` (the
 *                   server card's minmax column, as gateway.css `.fleet` had it) or an fr weight `1` `2` `3` —
 *                   never CSS from a file: the tokens are classes in layout.css. Below a desk's width every row
 *                   is one column, as the old grid was.
 *   THE TEMPLATES   /layouts/<name>.json, a committed library; /layouts/index.json names them in menu order and
 *                   the FIRST is what a fresh browser starts with ("the template in config" — a file, because
 *                   nothing about a layout is a deployment's; no env key, no fleet key). `default` is the panel as
 *                   it was before this file existed.
 *   STATELESS       the server keeps NOTHING. The chosen template and the arrangement live in this browser's
 *                   localStorage under `lb.layout`, versioned (`schema`) with a migration table; every storage
 *                   touch is wrapped (`Store`), and a browser that keeps nothing is told so in the notice line and
 *                   works for the visit. Nothing is stored until the operator chooses or moves something, so a
 *                   browser that never customised follows the shipped default as it evolves.
 *   LOUD, NEVER SILENT  a stored or imported layout naming an id this panel lacks draws a visible "unknown element"
 *                   tile where it stood and a notice offering Reset — never a quiet drop; an element new to the
 *                   panel that a stored layout does not name is appended and said; a template that will not load
 *                   or parse is a notice with the HTTP status or the reason, and the page keeps the layout it had.
 *   ARRANGING       the header's Layout menu picks a template, or opens edit mode: every tile grows a bar with a
 *                   grip (drag it between columns; arrow keys move it up / down / left / right), ←↑↓→ buttons for
 *                   touch, full/half, hide; a column's width cycles; `+ col` / `+ row` add room; Done prunes the
 *                   empty room and saves; Reset re-reads the chosen template; Export downloads the state as JSON
 *                   and Import pastes it — a layout carried to another browser.
 *
 * The pure half (`LY`: normalize, validate, the reducer, migrate, the Store, export/import) has no DOM and is what
 * tests/layout.node.js exercises; the engine below is the DOM. Vanilla, no library: the LB ships no third-party
 * script (AGENTS.md). Loaded after gateway.js; its boot runs only where `#layout` exists.
 */
'use strict';

const LY_SCHEMA = 1;
const LY_KEY = 'lb.layout';
const LY_WIDTHS = ['side', '1', '2', '3'];
const LY_SIZES = ['full', 'half'];
const LY_DIRS = ['up', 'down', 'left', 'right'];
const LY_MAX_COLS = 3;
/** schema → the function that lifts a state one schema up. Schema 1 is the first that shipped: the table is empty,
 *  and `lyMigrate` walks it one step at a time so a gap is a loud stop, never a guess. */
const LY_MIGRATIONS = {};

const lyClone = (v) => JSON.parse(JSON.stringify(v));
const lyIsObj = (v) => !!v && typeof v === 'object' && !Array.isArray(v);
const lyFail = (msg) => { throw new Error(msg); };
const lyKeys = (o, allowed, where) => {
  Object.keys(o).forEach((k) => { if (!allowed.includes(k)) lyFail(`${where}: unknown key "${k}" (expected ${allowed.join(', ')})`); });
};

// ── the pure half ─────────────────────────────────────────────────────────────────────────────────

/** A layout from a template file or a stored state: `{rows: [{cols: [{width, items: [{id, size}]}]}], hidden}`,
 *  every token checked, every unknown key refused with where it was found. Extra top-level keys (a template's
 *  name, label, what, schema) are the file's and are left to `lyTemplateMeta`. */
function lyNormalize(raw) {
  if (!lyIsObj(raw)) lyFail('a layout is an object with rows');
  if (!Array.isArray(raw.rows)) lyFail('a layout has rows (a list)');
  if (!raw.rows.length) lyFail('rows is empty: a layout has at least one row');
  const rows = raw.rows.map((row, r) => {
    const where = `row ${r + 1}`;
    if (!lyIsObj(row)) lyFail(`${where}: not an object`);
    lyKeys(row, ['cols'], where);
    if (!Array.isArray(row.cols)) lyFail(`${where}: cols is not a list`);
    if (!row.cols.length) lyFail(`${where}: cols is empty: a row has at least one column`);
    if (row.cols.length > LY_MAX_COLS) lyFail(`${where}: ${row.cols.length} columns, at most ${LY_MAX_COLS}`);
    const cols = row.cols.map((col, c) => {
      const at = `${where}, column ${c + 1}`;
      if (!lyIsObj(col)) lyFail(`${at}: not an object`);
      lyKeys(col, ['width', 'items'], at);
      if (!LY_WIDTHS.includes(col.width)) lyFail(`${at}: width ${JSON.stringify(col.width)} is not one of ${LY_WIDTHS.join(', ')}`);
      if (!Array.isArray(col.items)) lyFail(`${at}: items is not a list`);
      const items = col.items.map((it, i) => {
        const here = `${at}, item ${i + 1}`;
        if (typeof it === 'string') return { id: it, size: 'full' };
        if (!lyIsObj(it)) lyFail(`${here}: not an id (${JSON.stringify(it)})`);
        lyKeys(it, ['id', 'size'], here);
        if (typeof it.id !== 'string' || !it.id) lyFail(`${here}: not an id (${JSON.stringify(it.id)})`);
        const size = it.size === undefined ? 'full' : it.size;
        if (!LY_SIZES.includes(size)) lyFail(`${here}: size ${JSON.stringify(size)} is not one of ${LY_SIZES.join(', ')}`);
        return { id: it.id, size };
      });
      return { width: col.width, items };
    });
    return { cols };
  });
  const hidden = raw.hidden === undefined ? [] : raw.hidden;
  if (!Array.isArray(hidden) || hidden.some((h) => typeof h !== 'string')) lyFail('hidden is not a list of ids');
  return { rows, hidden: hidden.slice() };
}

/** A template file's menu facts, each required: a template with no `what` is a choice the menu cannot explain. */
function lyTemplateMeta(t) {
  if (!lyIsObj(t) || typeof t.name !== 'string' || !t.name) lyFail('a template has a name');
  if (typeof t.label !== 'string' || !t.label) lyFail(`template "${t.name}" has no label`);
  if (typeof t.what !== 'string' || !t.what) lyFail(`template "${t.name}" has no what`);
  return { name: t.name, label: t.label, what: t.what };
}

/** Every placed id, in reading order: rows, then columns, then items. */
function lyPlacedIds(layout) {
  const out = [];
  layout.rows.forEach((r) => r.cols.forEach((c) => c.items.forEach((it) => out.push(it.id))));
  return out;
}

/** What a layout says against the registry — CLASSIFIED, never dropped: `unknown` ids the panel lacks, `missing`
 *  registry ids it does not name, `duplicate` ids it names twice. The engine draws a notice for each kind. */
function lyValidate(layout, registry) {
  const named = lyPlacedIds(layout).concat(layout.hidden);
  const seen = new Set(), duplicate = [];
  named.forEach((id) => { if (seen.has(id) && !duplicate.includes(id)) duplicate.push(id); seen.add(id); });
  const unknown = named.filter((id, i) => !registry.includes(id) && named.indexOf(id) === i);
  const missing = registry.filter((id) => !seen.has(id));
  return { unknown, missing, duplicate };
}

/** Where an id stands: `{r, c, i}`, or null when it is hidden or absent. */
function lyLocate(layout, id) {
  for (let r = 0; r < layout.rows.length; r++) {
    const cols = layout.rows[r].cols;
    for (let c = 0; c < cols.length; c++) {
      const i = cols[c].items.findIndex((it) => it.id === id);
      if (i >= 0) return { r, c, i };
    }
  }
  return null;
}

const lyItemOf = (layout, id) => { const at = lyLocate(layout, id); return at ? layout.rows[at.r].cols[at.c].items[at.i] : null; };

/** The layout without `id` anywhere (placed or hidden). */
function lyRemove(layout, id) {
  const next = lyClone(layout);
  next.rows.forEach((r) => r.cols.forEach((c) => { c.items = c.items.filter((it) => it.id !== id); }));
  next.hidden = next.hidden.filter((h) => h !== id);
  return next;
}

/** `item` placed at row r, column c, index i (clamped to the column's end). */
function lyInsert(layout, item, r, c, i) {
  const next = lyClone(layout);
  const row = next.rows[r];
  if (!row) lyFail(`row ${r + 1} does not exist`);
  const col = row.cols[c];
  if (!col) lyFail(`row ${r + 1} has no column ${c + 1}`);
  col.items.splice(Math.max(0, Math.min(i, col.items.length)), 0, lyClone(item));
  return next;
}

/** A drop: `id` to row r, column c, index i. */
function lyPlace(layout, id, r, c, i) {
  const item = lyItemOf(layout, id) || { id, size: 'full' };
  // Removing first shifts a later index in the same column by one: the index is counted among the OTHER items.
  const at = lyLocate(layout, id);
  const index = at && at.r === r && at.c === c && at.i < i ? i - 1 : i;
  return lyInsert(lyRemove(layout, id), item, r, c, index);
}

/** One step: up / down swap inside the column and cross into the row above's end or the row below's start; left /
 *  right move to the neighbouring column at the same index. At an edge — or for an id not placed — the same
 *  arrangement comes back, so a held arrow key is harmless. */
function lyMove(layout, id, dir) {
  if (!LY_DIRS.includes(dir)) lyFail(`direction ${JSON.stringify(dir)} is not one of ${LY_DIRS.join(', ')}`);
  const at = lyLocate(layout, id);
  if (!at) return layout;
  const { r, c, i } = at;
  const row = layout.rows[r], col = row.cols[c];
  if (dir === 'up') {
    if (i > 0) return lyPlace(layout, id, r, c, i - 1);
    if (r === 0) return layout;
    const above = layout.rows[r - 1], tc = Math.min(c, above.cols.length - 1);
    return lyPlace(layout, id, r - 1, tc, above.cols[tc].items.length);
  }
  if (dir === 'down') {
    if (i < col.items.length - 1) return lyPlace(layout, id, r, c, i + 2);
    if (r === layout.rows.length - 1) return layout;
    const below = layout.rows[r + 1];
    return lyPlace(layout, id, r + 1, Math.min(c, below.cols.length - 1), 0);
  }
  const tc = dir === 'left' ? c - 1 : c + 1;
  if (tc < 0 || tc >= row.cols.length) return layout;
  return lyPlace(layout, id, r, tc, Math.min(i, row.cols[tc].items.length));
}

function lyHide(layout, id) {
  if (layout.hidden.includes(id)) return layout;
  const next = lyRemove(layout, id);
  next.hidden.push(id);
  return next;
}

/** Back on the page: at the end of the last row's last column, where the eye looks for what just appeared. */
function lyShow(layout, id) {
  const next = lyRemove(layout, id);
  const r = next.rows.length - 1, c = next.rows[r].cols.length - 1;
  return lyInsert(next, { id, size: 'full' }, r, c, next.rows[r].cols[c].items.length);
}

function lyResize(layout, id, size) {
  if (!LY_SIZES.includes(size)) lyFail(`size ${JSON.stringify(size)} is not one of ${LY_SIZES.join(', ')}`);
  const at = lyLocate(layout, id);
  if (!at) return layout;
  const next = lyClone(layout);
  next.rows[at.r].cols[at.c].items[at.i].size = size;
  return next;
}

function lySetWidth(layout, r, c, width) {
  if (!LY_WIDTHS.includes(width)) lyFail(`width ${JSON.stringify(width)} is not one of ${LY_WIDTHS.join(', ')}`);
  const next = lyClone(layout);
  if (!next.rows[r] || !next.rows[r].cols[c]) lyFail(`row ${r + 1}, column ${c + 1} does not exist`);
  next.rows[r].cols[c].width = width;
  return next;
}

function lyAddCol(layout, r) {
  const next = lyClone(layout);
  if (!next.rows[r]) lyFail(`row ${r + 1} does not exist`);
  if (next.rows[r].cols.length >= LY_MAX_COLS) lyFail(`row ${r + 1} has ${LY_MAX_COLS} columns already: at most ${LY_MAX_COLS}`);
  next.rows[r].cols.push({ width: '1', items: [] });
  return next;
}

function lyAddRow(layout) {
  const next = lyClone(layout);
  next.rows.push({ cols: [{ width: '1', items: [] }] });
  return next;
}

/** No blank room in a saved layout: empty columns go, then empty rows — but one row with one column always stays,
 *  so a layout that hides everything is still a layout `lyShow` can put an element back into. */
function lyPrune(layout) {
  const next = lyClone(layout);
  next.rows = next.rows.map((row) => ({ cols: row.cols.filter((c) => c.items.length) })).filter((row) => row.cols.length);
  if (!next.rows.length) next.rows = [{ cols: [{ width: '1', items: [] }] }];
  return next;
}

/** A stored or pasted state, lifted to this schema: `{schema, template, layout, edited}`. Refused with the reason
 *  when it is not a state, was saved by a newer panel, or has no migration from its schema. */
function lyMigrate(stored, migrations) {
  const table = migrations || LY_MIGRATIONS;
  let s = stored;
  if (!lyIsObj(s)) lyFail('the stored layout is not an object');
  if (!Number.isInteger(s.schema)) lyFail('the stored layout has no schema');
  if (s.schema > LY_SCHEMA) lyFail(`the stored layout is schema ${s.schema}; this panel reads 1`);
  while (s.schema < LY_SCHEMA) {
    const lift = table[s.schema];
    if (typeof lift !== 'function') lyFail(`no migration from schema ${s.schema}`);
    s = lift(s);
    if (!lyIsObj(s) || !Number.isInteger(s.schema)) lyFail(`the migration from schema ${stored.schema} returned no schema`);
  }
  if (typeof s.template !== 'string' || !s.template) lyFail('the stored layout names no template');
  if (!lyIsObj(s.layout)) lyFail('the stored layout has no layout');
  return { schema: LY_SCHEMA, template: s.template, layout: lyNormalize(s.layout), edited: !!s.edited };
}

const lyExportText = (state) => JSON.stringify({ schema: LY_SCHEMA, template: state.template, layout: state.layout, edited: !!state.edited }, null, 2) + '\n';

function lyImportText(text) {
  let parsed;
  try { parsed = JSON.parse(text); } catch (e) { lyFail(`not JSON: ${e.message}`); }
  if (!lyIsObj(parsed)) lyFail('the pasted layout is not an object');
  try { return lyMigrate(parsed); } catch (e) { lyFail(e.message.replace('the stored layout', 'the pasted layout')); }
  return null;
}

/** This browser's keeping of the state — or its honest absence. Every touch is wrapped: a private window, a
 *  blocked origin or a full quota throws from `localStorage`, and the panel must still arrange for the visit. The
 *  probe at construction (write, read, remove) is what decides `available`; `reason` is the browser's own words. */
class LyStore {
  constructor(storage) {
    this.storage = storage;
    this.available = false;
    this.reason = '';
    if (!storage || typeof storage.setItem !== 'function') { this.reason = 'no localStorage in this browser'; return; }
    const probe = `${LY_KEY}.probe`;
    try {
      storage.setItem(probe, '1');
      if (storage.getItem(probe) !== '1') throw new Error('a written value did not read back');
      storage.removeItem(probe);
      this.available = true;
    } catch (e) {
      this.reason = String((e && e.message) || e);
    }
  }
  /** The state, lifted to this schema; null when nothing is stored (or nothing can be); a throw names what is
   *  stored and wrong, for the notice. */
  load() {
    if (!this.available) return null;
    let text = null;
    try { text = this.storage.getItem(LY_KEY); } catch (e) { this.available = false; this.reason = String((e && e.message) || e); return null; }
    if (text == null) return null;
    let parsed;
    try { parsed = JSON.parse(text); } catch (e) { lyFail(`the stored layout is not JSON: ${e.message}`); }
    return lyMigrate(parsed);
  }
  save(state) {
    if (!this.available) return false;
    try { this.storage.setItem(LY_KEY, lyExportText(state)); return true; } catch (e) {
      this.available = false; this.reason = String((e && e.message) || e); return false;
    }
  }
  clear() {
    try { this.storage.removeItem(LY_KEY); } catch (e) { /* nothing was kept */ }
  }
}

const LY = {
  SCHEMA: LY_SCHEMA, KEY: LY_KEY, WIDTHS: LY_WIDTHS, SIZES: LY_SIZES, DIRS: LY_DIRS, MAX_COLS: LY_MAX_COLS, MIGRATIONS: LY_MIGRATIONS,
  normalize: lyNormalize, templateMeta: lyTemplateMeta, placedIds: lyPlacedIds, validate: lyValidate, locate: lyLocate,
  move: lyMove, place: lyPlace, hide: lyHide, show: lyShow, resize: lyResize, setWidth: lySetWidth, addCol: lyAddCol,
  addRow: lyAddRow, prune: lyPrune, migrate: lyMigrate, exportText: lyExportText, importText: lyImportText, Store: LyStore,
};

// ── the engine: the DOM ───────────────────────────────────────────────────────────────────────────

const lyEsc = (v) => String(v == null ? '' : v).replace(/[&<>"']/g,
  (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

/** The edit bar on every tile: the grip (drag it; arrow keys move it), ←↑↓→ for touch, the size, hide. A hidden
 *  tile shows the grip and `show` alone (layout.css hides the rest in the tray). */
function lyBarHtml(id, label, unknown) {
  const act = (name, text, title) => `<button type="button" class="ly-act" data-act="${name}" data-id="${lyEsc(id)}" title="${lyEsc(title)}" aria-label="${lyEsc(`${title}: ${label}`)}">${text}</button>`;
  return `<button type="button" class="ly-grip" data-id="${lyEsc(id)}" draggable="true"
      title="drag to move · arrow keys move it up, down, left, right" aria-label="${lyEsc(`move ${label}: drag, or arrow keys`)}">
      <i aria-hidden="true">⠿</i>${lyEsc(label)}${unknown ? ' <s>unknown</s>' : ''}</button>
    <span class="ly-acts">${act('left', '←', 'move left')}${act('up', '↑', 'move up')}${act('down', '↓', 'move down')}${act('right', '→', 'move right')}`
    + `${act('size', 'full', 'full or half of its column')}${act('hide', 'hide', 'hide')}${act('show', 'show', 'show')}</span>`;
}

class LayoutEngine {
  constructor(doc, win, store, fetchFn) {
    this.doc = doc; this.win = win; this.store = store; this.fetch = fetchFn;
    this.root = doc.getElementById('layout');
    this.tray = doc.getElementById('layout-hidden');
    this.registry = new Map();          // id → { host, label, slot }
    this.unknown = new Map();           // id → placeholder slot
    this.state = null;                  // { schema, template, layout, edited }
    this.editing = false;
    this.templates = null;              // [{name, label, what}] once the menu asked
    this.notes = new Map();             // key → notice row
    this.dragging = null;
  }

  // ── boot ──────────────────────────────────────────────────────────────────────────────────────

  boot() {
    if (!this.root || !this.tray) return;
    this.root.querySelectorAll('[data-ly]').forEach((host) => {
      const id = host.dataset.ly, label = host.dataset.lyLabel || id;
      const slot = this.doc.createElement('div');
      slot.className = 'ly-slot';
      slot.dataset.ly = id;
      slot.innerHTML = `<div class="ly-bar">${lyBarHtml(id, label, false)}</div>`;
      host.parentNode.insertBefore(slot, host);
      slot.appendChild(host);
      this.registry.set(id, { host, label, slot });
    });
    // A host gateway.js hides (`hidden`), or empties (`.servers:empty`), leaves its column blank: the column collapses.
    if (typeof MutationObserver === 'function') {
      const mo = new MutationObserver(() => this.scheduleRefresh());
      this.registry.forEach(({ host }) => mo.observe(host, { attributes: true, attributeFilter: ['hidden', 'class'], childList: true }));
    }
    this.wire();
    if (!this.store.available) this.say('store', `layout not kept in this browser: ${this.store.reason}`, []);
    let stored = null;
    try { stored = this.store.load(); } catch (e) {
      this.say('stored', `the stored layout cannot be read: ${e.message}`, [['Reset to default', () => this.pickTemplate(null)]]);
    }
    if (stored) this.applyState(stored, { save: false });
    else this.pickTemplate(null, { save: false });
  }

  ids() { return [...this.registry.keys()]; }

  // ── state ─────────────────────────────────────────────────────────────────────────────────────

  /** Take a state as this panel's: unknown ids drawn as notice tiles and said; new elements appended and said;
   *  a duplicate refused (two tiles for one node is not an arrangement) with Reset offered. */
  applyState(state, { save = true } = {}) {
    const v = lyValidate(state.layout, this.ids());
    if (v.duplicate.length) {
      this.say('dup', `the layout names ${v.duplicate.join(', ')} twice and cannot be drawn`, [['Reset to template', () => this.reset()]]);
      if (!this.state) this.pickTemplate(null, { save: false });
      return false;
    }
    let layout = state.layout;
    if (v.missing.length) {
      v.missing.forEach((id) => { layout = lyShow(layout, id); });
      this.say('missing', `new on this panel, shown at the end: ${v.missing.join(', ')}`, [['ok', () => this.unsay('missing')]]);
    } else this.unsay('missing');
    if (v.unknown.length) {
      this.say('unknown', `unknown element${v.unknown.length > 1 ? 's' : ''} in this layout: ${v.unknown.join(', ')} — not on this panel`,
        [['Reset to template', () => this.reset()]]);
    } else this.unsay('unknown');
    this.unsay('dup');
    this.state = { schema: LY_SCHEMA, template: state.template, layout, edited: !!state.edited };
    this.render();
    // Kept: what was stored and unreadable is overwritten now, so that notice has nothing left to say.
    if (save) { this.save(); this.unsay('stored'); }
    return true;
  }

  save() {
    if (!this.state) return;
    if (!this.store.save(this.state)) this.say('store', `layout not kept in this browser: ${this.store.reason}`, []);
  }

  /** A change from the edit bar: the reducer's new layout, drawn and kept; the focus stays on the same tile. */
  change(next, focus) {
    if (!this.state || next === this.state.layout) return;
    this.state = { ...this.state, layout: next, edited: true };
    this.render();
    this.save();
    if (focus) {
      const slot = this.slotOf(focus.id);
      const btn = slot && (slot.querySelector(`[data-act="${focus.act}"]`) || slot.querySelector('.ly-grip'));
      if (btn && btn.focus) btn.focus();
    }
  }

  /** The template library, once: /layouts/index.json then each file's facts, in menu order. */
  async loadTemplates() {
    if (this.templates) return this.templates;
    const index = await this.fetchJson('/layouts/index.json');
    if (!index || !Array.isArray(index.templates)) lyFail('/layouts/index.json names no templates');
    const metas = [];
    for (const name of index.templates) metas.push(lyTemplateMeta(await this.fetchJson(`/layouts/${encodeURIComponent(name)}.json`)));
    this.templates = metas;
    return metas;
  }

  async fetchJson(url) {
    const r = await this.fetch(url);
    if (!r.ok) lyFail(`${url}: HTTP ${r.status}`);
    return r.json();
  }

  /** The named template — or the library's first for null — read, checked and taken. A failure is a notice with
   *  the reason and the page keeps what it shows. */
  async pickTemplate(name, { save = true } = {}) {
    try {
      let want = name;
      if (want == null) {
        const index = await this.fetchJson('/layouts/index.json');
        want = Array.isArray(index.templates) ? index.templates[0] : null;
        if (!want) lyFail('/layouts/index.json names no templates');
      }
      const t = await this.fetchJson(`/layouts/${encodeURIComponent(want)}.json`);
      const meta = lyTemplateMeta(t);
      this.unsay('template');
      return this.applyState({ schema: LY_SCHEMA, template: meta.name, layout: lyNormalize(t), edited: false }, { save });
    } catch (e) {
      this.say('template', `template ${name == null ? '(default)' : JSON.stringify(name)} not loaded: ${e.message}`, [['retry', () => this.pickTemplate(name, { save })]]);
      return false;
    }
  }

  reset() { return this.pickTemplate(this.state ? this.state.template : null); }

  // ── drawing ───────────────────────────────────────────────────────────────────────────────────

  slotOf(id) {
    const known = this.registry.get(id);
    if (known) return known.slot;
    if (!this.unknown.has(id)) {
      const slot = this.doc.createElement('div');
      slot.className = 'ly-slot ly-unknown';
      slot.dataset.ly = id;
      slot.innerHTML = `<div class="ly-bar">${lyBarHtml(id, id, true)}</div>`
        + `<div class="ly-unknown-body" role="note">unknown element <code>${lyEsc(id)}</code> · not on this panel · Layout → Reset to template removes it</div>`;
      this.unknown.set(id, slot);
    }
    return this.unknown.get(id);
  }

  /** The rows and columns from the state, the slots moved into place (never rebuilt), the tray filled. */
  render() {
    if (!this.state) return;
    const { layout } = this.state;
    const keep = new Set();
    const rows = layout.rows.map((row, r) => {
      const rowEl = this.doc.createElement('div');
      rowEl.className = 'ly-row';
      rowEl.dataset.row = String(r);
      row.cols.forEach((col, c) => {
        const colEl = this.doc.createElement('div');
        colEl.className = 'ly-col';
        colEl.dataset.width = col.width;
        colEl.dataset.row = String(r); colEl.dataset.col = String(c);
        if (this.editing) {
          colEl.innerHTML = `<div class="ly-colbar"><button type="button" class="ly-act" data-act="width" data-r="${r}" data-c="${c}" `
            + `title="column width: side, 1, 2 or 3 shares">width ${lyEsc(col.width)}</button>`
            + `<span class="ly-colnum">row ${r + 1} · col ${c + 1}</span></div>`;
        }
        col.items.forEach((it) => {
          const slot = this.slotOf(it.id);
          slot.dataset.size = it.size;
          const size = slot.querySelector('[data-act="size"]');
          if (size) size.textContent = it.size;
          colEl.appendChild(slot);
          keep.add(it.id);
        });
        rowEl.appendChild(colEl);
      });
      if (this.editing && row.cols.length < LY_MAX_COLS) {
        rowEl.insertAdjacentHTML('beforeend', `<button type="button" class="ly-act ly-addcol" data-act="addcol" data-r="${r}" title="add a column to this row">+ col</button>`);
      }
      return rowEl;
    });
    this.root.querySelectorAll(':scope > .ly-row').forEach((el) => el.remove());
    rows.forEach((el) => this.root.insertBefore(el, this.tray));
    layout.hidden.forEach((id) => { const slot = this.slotOf(id); slot.dataset.size = 'full'; this.tray.appendChild(slot); keep.add(id); });
    // An unknown placeholder no layout names any more has nothing to stand for.
    [...this.unknown.keys()].forEach((id) => { if (!keep.has(id)) { this.unknown.get(id).remove(); this.unknown.delete(id); } });
    this.tray.hidden = !this.editing;
    this.root.querySelectorAll('.ly-grip').forEach((g) => { g.draggable = this.editing; });
    this.paintHead();
    this.refreshEmpty();
  }

  /** A column whose every tile is blank right now (its host hidden by gateway.js, or empty) takes no room; a row
   *  whose every column is blank neither — the old `.fleet.no-cluster` rule, for any column. Edit mode shows all. */
  refreshEmpty() {
    const blank = (slot) => {
      const host = slot.firstElementChild && slot.firstElementChild.classList.contains('ly-bar') ? slot.children[1] : slot.firstElementChild;
      if (!host) return true;
      if (slot.classList.contains('ly-unknown')) return false;
      if (host.hidden) return true;
      const cs = this.win.getComputedStyle ? this.win.getComputedStyle(host) : null;
      return !!cs && cs.display === 'none';
    };
    this.root.querySelectorAll(':scope > .ly-row').forEach((rowEl) => {
      let rowBlank = true;
      rowEl.querySelectorAll(':scope > .ly-col').forEach((colEl) => {
        const slots = [...colEl.querySelectorAll(':scope > .ly-slot')];
        const empty = slots.every(blank);
        colEl.classList.toggle('ly-empty', empty);
        if (!empty) rowBlank = false;
      });
      rowEl.classList.toggle('ly-empty', rowBlank);
    });
  }

  scheduleRefresh() {
    if (this.refreshing) return;
    this.refreshing = true;
    const run = () => { this.refreshing = false; this.refreshEmpty(); };
    if (this.win.requestAnimationFrame) this.win.requestAnimationFrame(run); else run();
  }

  paintHead() {
    const name = this.doc.getElementById('layout-name');
    if (name && this.state) name.textContent = `${this.state.template}${this.state.edited ? ' · edited' : ''}`;
    const tools = this.doc.getElementById('layout-tools-name');
    if (tools && this.state) tools.textContent = this.state.template;
    const btn = this.doc.getElementById('layout-button');
    if (btn && this.state) btn.title = `layout: template ${this.state.template}${this.state.edited ? ', arranged in this browser' : ''}`
      + ` · ${this.store.available ? 'kept in this browser' : `not kept: ${this.store.reason}`}`;
    // THE MENU FOLLOWS THE STATE (operator 2026-10-30: Wall picked with the menu open, Default still marked, the export
    // saying "wall"): the pressed template and an open export are redrawn on every change, not only when the menu opens.
    const dlg = this.menu();
    if (dlg && this.state) {
      dlg.querySelectorAll('.ly-template').forEach((b) => b.setAttribute('aria-pressed',
        b.getAttribute('data-template') === this.state.template ? 'true' : 'false'));
      const io = dlg.querySelector('.ly-io'), text = dlg.querySelector('.ly-io-text');
      if (io && text && !io.hidden && io.dataset.mode === 'export') {
        text.value = lyExportText(this.state);
        const a = dlg.querySelector('.ly-download');
        try {
          if (a && a.dataset.url) URL.revokeObjectURL(a.dataset.url);
          if (a) a.href = a.dataset.url = URL.createObjectURL(new Blob([text.value], { type: 'application/json' }));
        } catch (e) { /* no Blob URLs here: the text is still current */ }
      }
    }
  }

  // ── notices ───────────────────────────────────────────────────────────────────────────────────

  /** One line per matter, keyed so it is replaced, not repeated: the text and its actions. */
  say(key, text, actions) {
    const box = this.doc.getElementById('layout-notice');
    if (!box) return;
    this.unsay(key);
    const row = this.doc.createElement('div');
    row.className = 'ly-note';
    row.dataset.key = key;
    row.innerHTML = `<span>${lyEsc(text)}</span>`;
    actions.forEach(([label, fn]) => {
      const b = this.doc.createElement('button');
      b.type = 'button'; b.className = 'ly-act'; b.textContent = label;
      b.addEventListener('click', fn);
      row.appendChild(b);
    });
    box.appendChild(row);
    this.notes.set(key, row);
    box.hidden = false;
  }

  unsay(key) {
    const row = this.notes.get(key);
    if (row) { row.remove(); this.notes.delete(key); }
    const box = this.doc.getElementById('layout-notice');
    if (box) box.hidden = this.notes.size === 0;
  }

  // ── edit mode ─────────────────────────────────────────────────────────────────────────────────

  enterEdit() {
    if (!this.state) return;
    this.editing = true;
    if (this.doc.body) this.doc.body.classList.add('ly-editing');
    const tools = this.doc.getElementById('layout-tools');
    if (tools) tools.hidden = false;
    this.closeMenu();
    this.render();
    const first = this.root.querySelector('.ly-grip');
    if (first && first.focus) first.focus();
  }

  /** Done: the blank room pruned, the state kept. */
  exitEdit() {
    this.editing = false;
    if (this.doc.body) this.doc.body.classList.remove('ly-editing');
    const tools = this.doc.getElementById('layout-tools');
    if (tools) tools.hidden = true;
    if (this.state) this.state = { ...this.state, layout: lyPrune(this.state.layout) };
    this.render();
    this.save();
    const btn = this.doc.getElementById('layout-button');
    if (btn && btn.focus) btn.focus();
  }

  /** An edit-bar button, by `data-act`. */
  act(btn) {
    if (!this.state) return;
    const L = this.state.layout, act = btn.dataset.act, id = btn.dataset.id;
    const r = Number(btn.dataset.r), c = Number(btn.dataset.c);
    if (LY_DIRS.includes(act)) return this.change(lyMove(L, id, act), { id, act });
    if (act === 'hide') return this.change(lyHide(L, id), null);
    if (act === 'show') return this.change(lyShow(L, id), { id, act: 'hide' });
    if (act === 'size') { const it = lyItemOf(L, id); return it ? this.change(lyResize(L, id, it.size === 'full' ? 'half' : 'full'), { id, act }) : undefined; }
    if (act === 'width') {
      const cur = L.rows[r].cols[c].width;
      const next = LY_WIDTHS[(LY_WIDTHS.indexOf(cur) + 1) % LY_WIDTHS.length];
      this.change(lySetWidth(L, r, c, next), null);
      const again = this.root.querySelector(`[data-act="width"][data-r="${r}"][data-c="${c}"]`);
      if (again && again.focus) again.focus();
      return undefined;
    }
    if (act === 'addcol') return this.change(lyAddCol(L, r), null);
    return undefined;
  }

  // ── drag (pointer) ────────────────────────────────────────────────────────────────────────────

  /** The index a drop at `y` lands on among a column's tiles: before the first tile whose middle is below it. */
  dropIndex(colEl, y) {
    const slots = [...colEl.querySelectorAll(':scope > .ly-slot')];
    let i = 0;
    for (; i < slots.length; i++) {
      const b = slots[i].getBoundingClientRect();
      if (y < b.top + b.height / 2) break;
    }
    return { i, slots };
  }

  markDrop(colEl, i, slots) {
    this.root.querySelectorAll('.ly-drop-before, .ly-drop-end').forEach((el) => el.classList.remove('ly-drop-before', 'ly-drop-end'));
    if (i < slots.length) slots[i].classList.add('ly-drop-before'); else colEl.classList.add('ly-drop-end');
  }

  // ── the menu ──────────────────────────────────────────────────────────────────────────────────

  menu() { return this.doc.getElementById('layout-menu'); }

  /** Open as a <dialog> knows it (`open`); a page without <dialog> support keeps `hidden`. */
  menuOpen() {
    const dlg = this.menu();
    if (!dlg || !dlg.dataset.built) return false;
    return typeof dlg.show === 'function' ? !!dlg.open : !dlg.hidden;
  }

  async openMenu() {
    const dlg = this.menu();
    if (!dlg) return;
    if (!dlg.dataset.built) {
      dlg.innerHTML = `<div class="ly-menu-head"><b>Layout</b><button type="button" class="ly-act" data-menu="close" aria-label="close">×</button></div>
        <div class="ly-templates" role="group" aria-label="template"><span class="ly-dim">templates…</span></div>
        <div class="ly-menu-acts">
          <button type="button" class="ly-act ly-primary" data-menu="arrange">Arrange</button>
          <button type="button" class="ly-act" data-menu="reset">Reset to template</button>
          <button type="button" class="ly-act" data-menu="export">Export</button>
          <button type="button" class="ly-act" data-menu="import">Import</button>
        </div>
        <div class="ly-io" hidden>
          <textarea class="ly-io-text" rows="8" spellcheck="false" aria-label="layout as JSON"></textarea>
          <div class="ly-io-acts">
            <button type="button" class="ly-act ly-primary" data-menu="apply" hidden>Apply</button>
            <a class="ly-act ly-download" download="fleet-layout.json" hidden>Download</a>
            <button type="button" class="ly-act" data-menu="copy" hidden>Copy</button>
          </div>
          <p class="ly-io-err" role="alert" hidden></p>
        </div>
        <p class="ly-store ly-dim"></p>`;
      dlg.dataset.built = '1';
      dlg.addEventListener('click', (e) => {
        const b = e.target && e.target.closest ? e.target.closest('[data-menu], [data-template]') : null;
        if (!b) return;
        if (b.dataset.template) { this.pickTemplate(b.dataset.template); this.closeMenu(); return; }
        this.menuAct(b.dataset.menu);
      });
      dlg.addEventListener('keydown', (e) => { if (e.key === 'Escape') { e.preventDefault(); this.closeMenu(); } });
    }
    const store = dlg.querySelector('.ly-store');
    if (store) store.textContent = this.store.available ? `kept in this browser only (localStorage ${LY_KEY}) · the server stores no layout`
      : `not kept: ${this.store.reason} · the arrangement lasts this visit`;
    const btn = this.doc.getElementById('layout-button');
    if (btn) btn.setAttribute('aria-expanded', 'true');
    if (dlg.show && !dlg.open) dlg.show(); else dlg.hidden = false;
    try {
      const list = dlg.querySelector('.ly-templates');
      const metas = await this.loadTemplates();
      list.innerHTML = metas.map((m) => `<button type="button" class="ly-template" data-template="${lyEsc(m.name)}"
          aria-pressed="${this.state && this.state.template === m.name ? 'true' : 'false'}"><b>${lyEsc(m.label)}</b><span>${lyEsc(m.what)}</span></button>`).join('');
    } catch (e) {
      const list = dlg.querySelector('.ly-templates');
      if (list) list.innerHTML = `<span class="ly-io-err">templates not loaded: ${lyEsc(e.message)}</span>`;
    }
  }

  closeMenu() {
    const dlg = this.menu();
    if (!dlg) return;
    if (dlg.close && dlg.open) dlg.close(); else dlg.hidden = true;
    const btn = this.doc.getElementById('layout-button');
    if (btn) btn.setAttribute('aria-expanded', 'false');
  }

  menuAct(what) {
    const dlg = this.menu();
    const io = dlg.querySelector('.ly-io'), text = dlg.querySelector('.ly-io-text'), err = dlg.querySelector('.ly-io-err');
    const show = (sel, on) => { const el = dlg.querySelector(sel); if (el) el.hidden = !on; };
    if (what === 'close') return this.closeMenu();
    if (what === 'arrange') return this.enterEdit();
    if (what === 'reset') { this.reset(); return this.closeMenu(); }
    if (what === 'export') {
      if (!this.state) return undefined;
      const out = lyExportText(this.state);
      io.hidden = false; io.dataset.mode = 'export'; text.value = out; text.readOnly = true; err.hidden = true;
      show('[data-menu="apply"]', false); show('[data-menu="copy"]', true); show('.ly-download', true);
      const a = dlg.querySelector('.ly-download');
      try {
        if (a.dataset.url) URL.revokeObjectURL(a.dataset.url);
        a.href = a.dataset.url = URL.createObjectURL(new Blob([out], { type: 'application/json' }));
      } catch (e) { a.hidden = true; }
      text.focus(); text.select();
      return undefined;
    }
    if (what === 'import') {
      io.hidden = false; io.dataset.mode = 'import'; text.value = ''; text.readOnly = false; err.hidden = true;
      text.placeholder = 'paste a layout exported from another browser';
      show('[data-menu="apply"]', true); show('[data-menu="copy"]', false); show('.ly-download', false);
      text.focus();
      return undefined;
    }
    if (what === 'apply') {
      try {
        const state = lyImportText(text.value);
        err.hidden = true;
        if (this.applyState(state)) { io.hidden = true; this.closeMenu(); }
      } catch (e) { err.textContent = e.message; err.hidden = false; }
      return undefined;
    }
    if (what === 'copy') {
      text.focus(); text.select();
      try { if (navigator.clipboard) navigator.clipboard.writeText(text.value); } catch (e) { /* selected: ctrl-c */ }
      return undefined;
    }
    return undefined;
  }

  // ── wiring ────────────────────────────────────────────────────────────────────────────────────

  wire() {
    const btn = this.doc.getElementById('layout-button');
    if (btn) btn.addEventListener('click', () => { if (this.menuOpen()) this.closeMenu(); else this.openMenu(); });
    const tools = this.doc.getElementById('layout-tools');
    if (tools) tools.addEventListener('click', (e) => {
      const b = e.target && e.target.closest ? e.target.closest('[data-tools]') : null;
      if (!b) return;
      if (b.dataset.tools === 'done') this.exitEdit();
      else if (b.dataset.tools === 'reset') this.reset();
      else if (b.dataset.tools === 'addrow') this.change(lyAddRow(this.state.layout), null);
    });
    // The edit bars: the buttons, the grip's arrow keys.
    const onAct = (e) => {
      const b = e.target && e.target.closest ? e.target.closest('.ly-act[data-act]') : null;
      if (b && this.editing) this.act(b);
    };
    this.root.addEventListener('click', onAct);
    this.tray.addEventListener('click', onAct);
    const onKey = (e) => {
      const g = e.target && e.target.closest ? e.target.closest('.ly-grip') : null;
      if (!g || !this.editing) return;
      const dir = { ArrowUp: 'up', ArrowDown: 'down', ArrowLeft: 'left', ArrowRight: 'right' }[e.key];
      if (!dir) return;
      e.preventDefault();
      const id = g.dataset.id;
      if (this.state.layout.hidden.includes(id)) return;
      this.change(lyMove(this.state.layout, id, dir), { id, act: 'grip' });
    };
    this.root.addEventListener('keydown', onKey);
    this.tray.addEventListener('keydown', onKey);
    // Pointer drag (HTML5): the grip starts it, a column takes it at the index under the pointer.
    this.root.addEventListener('dragstart', (e) => {
      const g = e.target && e.target.closest ? e.target.closest('.ly-grip') : null;
      if (!g || !this.editing) { if (e.preventDefault) e.preventDefault(); return; }
      this.dragging = g.dataset.id;
      const slot = this.slotOf(this.dragging);
      slot.classList.add('ly-dragging');
      if (e.dataTransfer) { e.dataTransfer.effectAllowed = 'move'; try { e.dataTransfer.setData('text/plain', this.dragging); } catch (x) { /* some engines refuse */ } }
    });
    this.root.addEventListener('dragover', (e) => {
      if (!this.dragging) return;
      const colEl = e.target && e.target.closest ? e.target.closest('.ly-col') : null;
      if (!colEl) return;
      e.preventDefault();
      if (e.dataTransfer) e.dataTransfer.dropEffect = 'move';
      const { i, slots } = this.dropIndex(colEl, e.clientY);
      this.markDrop(colEl, i, slots);
    });
    this.root.addEventListener('drop', (e) => {
      if (!this.dragging) return;
      const colEl = e.target && e.target.closest ? e.target.closest('.ly-col') : null;
      if (!colEl) return;
      e.preventDefault();
      const { i, slots } = this.dropIndex(colEl, e.clientY);
      // The dragged tile's own place is not a gap: the index is counted among the others (`lyPlace` does the same).
      const id = this.dragging, r = Number(colEl.dataset.row), c = Number(colEl.dataset.col);
      const self = slots.findIndex((s) => s.dataset.ly === id);
      this.endDrag();
      this.change(lyPlace(this.state.layout, id, r, c, self >= 0 && self < i ? i - 1 : i), { id, act: 'grip' });
    });
    this.root.addEventListener('dragend', () => this.endDrag());
    this.doc.addEventListener('keydown', (e) => { if (e.key === 'Escape' && this.editing && !this.menuOpen()) this.exitEdit(); });
    // The old `.fleet` grid broke at a desk's width; the columns do the same (layout.css): a resize re-reads blankness.
    this.win.addEventListener('resize', () => this.scheduleRefresh());
  }

  endDrag() {
    this.root.querySelectorAll('.ly-dragging, .ly-drop-before, .ly-drop-end').forEach((el) => el.classList.remove('ly-dragging', 'ly-drop-before', 'ly-drop-end'));
    this.dragging = null;
  }
}

// ── boot ──────────────────────────────────────────────────────────────────────────────────────────
// Only on a page with the layout root (index.html); the node harness hands a document that has none.
if (typeof document !== 'undefined' && document && document.getElementById && document.getElementById('layout')) {
  let storage;
  try { storage = window.localStorage; } catch (e) { storage = { setItem() { throw e; }, getItem() { throw e; }, removeItem() {} }; }
  const engine = new LayoutEngine(document, window, new LyStore(storage), (url, init) => fetch(url, init));
  engine.boot();
  window.layoutEngine = engine;
}
