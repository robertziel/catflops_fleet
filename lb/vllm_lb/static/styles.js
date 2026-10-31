/* THE PANEL'S VISUAL STYLES (operator, 2026-10-30: "Add switch between styles, current is 'Vibe coded'").
 *
 *   A STYLE        a set of values for gateway.css's tokens (and, sparingly, rules a token cannot reach). The BASE,
 *                  "Vibe coded" (key `vibe`), IS gateway.css's :root block — it needs no file. Every other style is
 *                  /styles/<key>.css, scoped to :root[data-style="<key>"], loaded on demand by a <link> when picked.
 *   THE REGISTRY   /styles/index.json: `{styles: [{key, label, file}]}` in menu order, the first `vibe` with
 *                  `file: null`. The build stamps each file with its content hash (lb/Dockerfile), so the hrefs this
 *                  script loads are cache-busted like every static reference.
 *   THE DEFAULT    the DEPLOYMENT'S (operator, 2026-10-31: "allow setup default in yml config"): the fleet file's
 *                  `panel.style`, which the LB stamps on <html> as `data-style-default` + `data-style-default-href`
 *                  (vllm_lb/panel_styles.py), so it is applied BEFORE FIRST PAINT with no request of its own. A page
 *                  with nothing stamped (an LB with no fleet file) opens in the base and SAYS so.
 *   PRECEDENCE     this browser's own choice > the deployment's default. Nothing is stored until the operator picks:
 *                  a browser that chose nothing follows the deployment when its default changes, and picking the
 *                  default (the menu tags it) forgets the browser's choice, so it follows again.
 *   STATELESS      the server keeps only the deployment's default: a choice is this browser's localStorage under
 *                  `lb.style`, as JSON `{key, href}` — the href so the choice can be applied BEFORE FIRST PAINT, from
 *                  <head>, without waiting for the registry (a stale href still names the file: the LB serves by
 *                  path). Every storage touch is wrapped; a browser that keeps nothing works for the visit and is told
 *                  so.
 *   LOUD, NEVER SILENT  no fallbacks (AGENTS.md): a stored key the registry does not list, a stored value that does
 *                  not parse, a sheet that does not load, a registry that will not fetch or a page with no deployment
 *                  default is a visible notice with its action ("style "x" not found — using Clean (this deployment's
 *                  default) [reset]"), never a quiet substitute.
 *   COLOURS IN JS  a switch fires `fleet:style` on document once the new sheet has applied: palette.js drops its
 *                  cached tokens, gateway.js and benchmarks.js redraw what they computed from them. So does a change of
 *                  the system's light / dark appearance, which a style may follow (tokens under prefers-color-scheme).
 *
 * Loaded in <head> of every gateway page (index, benchmarks, sessions), after the stylesheets: the first half runs at
 * once (`stEarly`: data-style and the stored sheet, render-blocking); the control and the notices mount at
 * DOMContentLoaded into #style-slot and #style-notice. The pure half (`ST`) has no DOM and is what
 * tests/styles.node.js exercises. Vanilla, no library: the LB ships no third-party script.
 */
'use strict';

const ST_KEY = 'lb.style';
/** The base style: gateway.css's own :root, first in the registry, no file. NOT the default — that is the
 *  deployment's (the fleet's panel.style, stamped on <html>). */
const ST_BASE = 'vibe';
const ST_INDEX_URL = '/styles/index.json';
const ST_KEY_RE = /^[a-z][a-z0-9-]{0,31}$/;
const ST_HREF_RE = /^\/styles\/[a-z][a-z0-9-]{0,31}\.css(\?v=[0-9a-f]{6,64})?$/;
const stFail = (msg) => { throw new Error(msg); };
const stEsc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);

// ── the pure half ─────────────────────────────────────────────────────────────────────────────────

/** The registry, every entry checked: the first is the default (`vibe`, no file), every other names its own sheet. */
function stParseIndex(data) {
  if (!data || typeof data !== 'object' || !Array.isArray(data.styles) || !data.styles.length) {
    stFail(`${ST_INDEX_URL} lists no styles`);
  }
  const seen = new Set();
  return data.styles.map((s, i) => {
    if (!s || typeof s !== 'object') stFail(`${ST_INDEX_URL}: entry ${i} is not an object`);
    Object.keys(s).forEach((k) => { if (!['key', 'label', 'file'].includes(k)) stFail(`${ST_INDEX_URL}: entry ${i}: unknown key "${k}"`); });
    if (typeof s.key !== 'string' || !ST_KEY_RE.test(s.key)) stFail(`${ST_INDEX_URL}: entry ${i}: bad key ${JSON.stringify(s.key)}`);
    if (seen.has(s.key)) stFail(`${ST_INDEX_URL}: "${s.key}" is listed twice`);
    seen.add(s.key);
    if (typeof s.label !== 'string' || !s.label.trim()) stFail(`${ST_INDEX_URL}: "${s.key}" has no label`);
    if (i === 0) {
      if (s.key !== ST_BASE || s.file !== null) stFail(`${ST_INDEX_URL}: the first style must be "${ST_BASE}" with file null (gateway.css)`);
    } else if (typeof s.file !== 'string' || !ST_HREF_RE.test(s.file) || !s.file.startsWith(`/styles/${s.key}.css`)) {
      stFail(`${ST_INDEX_URL}: "${s.key}" must name /styles/${s.key}.css, not ${JSON.stringify(s.file)}`);
    }
    return { key: s.key, label: s.label, file: s.file };
  });
}

/** What this browser stored: `{key, href}`, null when nothing is, a throw naming what is stored and wrong. */
function stReadStored(storage) {
  let text = null;
  try { text = storage ? storage.getItem(ST_KEY) : null; } catch (e) { return null; }
  if (text == null) return null;
  let v;
  try { v = JSON.parse(text); } catch (e) { stFail(`the stored style is not JSON: ${text.slice(0, 60)}`); }
  if (!v || typeof v !== 'object' || typeof v.key !== 'string' || !ST_KEY_RE.test(v.key)) {
    stFail(`the stored style names no valid key: ${text.slice(0, 60)}`);
  }
  if (v.href != null && (typeof v.href !== 'string' || !ST_HREF_RE.test(v.href))) {
    stFail(`the stored style's sheet is not a /styles/ file: ${String(v.href).slice(0, 60)}`);
  }
  return { key: v.key, href: v.href == null ? null : v.href };
}

/** What the LB stamped on <html>: the deployment's default `{key, href}` (href null for the base), or null when nothing
 *  is stamped. A stamp that is not a key is no stamp; an href that is not one of our /styles/ files is not loaded. */
function stReadDeployed(root) {
  const key = root.getAttribute('data-style-default') || '';
  if (!ST_KEY_RE.test(key)) return null;
  const href = root.getAttribute('data-style-default-href') || '';
  return { key, href: href && ST_HREF_RE.test(href) ? href : null };
}

/** The registry's entry this page shows: the stored choice's, else the deployment default's. `unknown` names a stored
 *  key the registry does not list (`entry` then the deployment default's); `deployedUnknown` a deployment default it
 *  does not list, `undeployed` a page with none stamped (`entry` then the base's). */
function stResolve(styles, stored, deployed) {
  const base = styles[0];
  const dflt = deployed ? styles.find((s) => s.key === deployed.key) || null : null;
  const deployedUnknown = !!deployed && !dflt;
  if (stored) {
    const entry = styles.find((s) => s.key === stored.key);
    if (entry) return { entry };
    return { unknown: stored.key, entry: dflt || base, deployedUnknown, undeployed: !deployed };
  }
  return { entry: dflt || base, deployedUnknown, undeployed: !deployed };
}

/** This browser's keeping of the choice — or its honest absence (a private window, a blocked origin, a full quota).
 *  The probe at construction decides `available`; `reason` is the browser's own words. */
class StStore {
  constructor(storage) {
    this.storage = storage;
    this.available = false;
    this.reason = '';
    if (!storage || typeof storage.setItem !== 'function') { this.reason = 'no localStorage in this browser'; return; }
    try {
      storage.setItem(`${ST_KEY}.probe`, '1');
      if (storage.getItem(`${ST_KEY}.probe`) !== '1') throw new Error('a written value did not read back');
      storage.removeItem(`${ST_KEY}.probe`);
      this.available = true;
    } catch (e) { this.reason = String((e && e.message) || e); }
  }
  save(key, href) {
    if (!this.available) return false;
    try { this.storage.setItem(ST_KEY, JSON.stringify({ key, href })); return true; } catch (e) {
      this.available = false; this.reason = String((e && e.message) || e); return false;
    }
  }
  clear() {
    if (!this.available) return;
    try { this.storage.removeItem(ST_KEY); } catch (e) { /* nothing kept: nothing to clear */ }
  }
}

/** BEFORE FIRST PAINT, from <head>: this browser's stored style — else the deployment's default — its key on <html>
 *  and its sheet linked, render-blocking, so a page opens in the style it will show, never another first. A stored
 *  value that does not parse is left for the notice at boot, and the page opens in the deployment's default. */
function stEarly(doc, storage) {
  const root = doc.documentElement;
  const deployed = stReadDeployed(root);
  let stored = null, error = null;
  try { stored = stReadStored(storage); } catch (e) { error = e.message; }
  const shown = stored || deployed;
  root.setAttribute('data-style', shown ? shown.key : ST_BASE);
  let link = null;
  if (shown && shown.key !== ST_BASE && shown.href) {
    link = doc.createElement('link');
    link.rel = 'stylesheet';
    link.id = 'style-sheet';
    link.setAttribute('blocking', 'render');
    link.href = shown.href;
    doc.head.appendChild(link);
  }
  return { stored, error, link, deployed };
}

const ST = { KEY: ST_KEY, BASE: ST_BASE, INDEX_URL: ST_INDEX_URL, parseIndex: stParseIndex, readStored: stReadStored,
  readDeployed: stReadDeployed, resolve: stResolve, Store: StStore, early: stEarly };

// ── the DOM half: the header's Style control, the notices, the switch ─────────────────────────────

class StyleSwitch {
  constructor(doc, win, store, fetchJson, early) {
    this.doc = doc; this.win = win; this.store = store; this.fetchJson = fetchJson; this.early = early;
    this.styles = null;
    this.deployed = early.deployed ? early.deployed.key : null;
    this.current = early.stored ? early.stored.key : (this.deployed || ST_BASE);
    this.notes = new Map();
  }

  label(key) {
    const s = (this.styles || []).find((x) => x.key === key);
    return s ? s.label : key;
  }

  /** The style is applied: <html data-style>, everyone told (palette.js, the redraws). */
  applied(key) {
    this.current = key;
    this.doc.documentElement.setAttribute('data-style', key);
    const name = this.doc.getElementById('style-name');
    if (name) name.textContent = this.label(key);
    this.doc.dispatchEvent(new this.win.Event('fleet:style'));
    this.renderMenu();
  }

  /** Switch to `entry`: the default drops the sheet; another loads its sheet first and applies once it has loaded, so
   *  the page never shows half a style. A sheet that fails is a notice and the page keeps the style it had. */
  apply(entry, { save = true } = {}) {
    const old = this.doc.getElementById('style-sheet');
    const kept = () => {
      if (!save) return;
      if (!this.store.save(entry.key, entry.file)) {
        this.say('store', `style not kept in this browser: ${this.store.reason} — it lasts this visit`, []);
      }
    };
    if (!entry.file) {
      if (old) old.remove();
      this.unsay('load');
      this.applied(entry.key);
      kept();
      return;
    }
    if (old && old.getAttribute('href') === entry.file && old.sheet) {
      this.applied(entry.key);
      kept();
      return;
    }
    const link = this.doc.createElement('link');
    link.rel = 'stylesheet';
    link.addEventListener('load', () => {
      if (old && old !== link) old.remove();
      link.id = 'style-sheet';
      this.unsay('load');
      this.applied(entry.key);
      kept();
    });
    link.addEventListener('error', () => {
      link.remove();
      this.say('load', `style "${entry.label}" did not load (${entry.file}) — still ${this.label(this.current)}`,
        [['retry', () => this.apply(entry, { save })]]);
    });
    link.href = entry.file;
    this.doc.head.appendChild(link);
  }

  async boot() {
    this.mount();
    // The sheet linked from <head> may finish after gateway.js first drew: tell palette.js once it has.
    const early = this.early.link;
    if (early) {
      if (early.sheet) this.doc.dispatchEvent(new this.win.Event('fleet:style'));
      else early.addEventListener('load', () => this.doc.dispatchEvent(new this.win.Event('fleet:style')));
    }
    // THE SYSTEM APPEARANCE: a style may follow it ("Glass": its tokens re-valued under prefers-color-scheme), and then
    // a light ↔ dark change re-values them as a style change does — palette.js drops its cache, the panel redraws.
    this.win.matchMedia('(prefers-color-scheme: dark)')
      .addEventListener('change', () => this.doc.dispatchEvent(new this.win.Event('fleet:style')));
    await this.loadIndex();
    if (this.early.error) {
      this.say('stored', `the stored style cannot be read (${this.early.error}) — using ${this.labelDefault()}`,
        [['reset', () => { this.store.clear(); this.unsay('stored'); }]]);
    }
  }

  /** What a page with no usable choice of its own shows, as a notice names it: the deployment's default, or the base. */
  labelDefault() {
    const dflt = this.deployed && (!this.styles || this.styles.some((s) => s.key === this.deployed));
    return dflt ? `${this.label(this.deployed)} (this deployment's default)` : this.label(ST_BASE);
  }

  async loadIndex() {
    try {
      this.styles = stParseIndex(await this.fetchJson(ST_INDEX_URL));
    } catch (e) {
      this.say('index', `styles not loaded: ${e.message}`, [['retry', () => this.loadIndex()]]);
      return;
    }
    this.unsay('index');
    const r = stResolve(this.styles, this.early.stored, this.early.deployed);
    // NOT a silent substitute: whatever is shown in place of what was asked for is applied AND said.
    const base = this.label(ST_BASE);
    if (r.undeployed) this.say('undeployed', `this gateway names no default style (its fleet file's panel.style) — showing ${base}`, []);
    if (r.deployedUnknown) {
      this.say('deployed', `this deployment's default style "${this.deployed}" is not in this page's styles — showing ${base}`,
        [['reload', () => this.win.location.reload()]]);
    }
    if (r.unknown) {
      this.say('unknown', `style "${r.unknown}" not found — using ${this.labelDefault()}`,
        [['reset', () => { this.store.clear(); this.unsay('unknown'); }]]);
    }
    // Only a choice that IS this browser's is re-saved (a stored href from an older build: the registry's stamped one
    // is the truth); the deployment's default is followed, never stored.
    const save = !!this.early.stored && !r.unknown;
    const link = this.doc.getElementById('style-sheet');
    if (!r.entry.file) {
      if (link) link.remove();
      this.applied(r.entry.key);
    } else if (!link || link.getAttribute('href') !== r.entry.file) {
      this.apply(r.entry, { save });
    } else {
      this.applied(r.entry.key);
    }
  }

  // ── the control ──────────────────────────────────────────────────────────────────────────────

  mount() {
    const slot = this.doc.getElementById('style-slot');
    if (!slot || slot.dataset.built) return;
    slot.dataset.built = '1';
    slot.innerHTML = `<button id="style-button" class="st-open" type="button" aria-haspopup="menu" aria-controls="style-menu"
        aria-expanded="false"><span class="ly-word">Style &middot; </span><b id="style-name">${stEsc(this.current)}</b></button>
      <div id="style-menu" class="st-menu" role="menu" aria-label="style" hidden></div>`;
    const btn = this.doc.getElementById('style-button');
    const menu = this.doc.getElementById('style-menu');
    btn.addEventListener('click', () => (menu.hidden ? this.openMenu() : this.closeMenu()));
    btn.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowDown' || e.key === 'ArrowUp') { e.preventDefault(); this.openMenu(e.key === 'ArrowUp' ? 'last' : 'checked'); }
    });
    menu.addEventListener('click', (e) => {
      const item = e.target && e.target.closest ? e.target.closest('[data-style-key]') : null;
      if (!item) return;
      this.pick(item.dataset.styleKey);
    });
    menu.addEventListener('keydown', (e) => this.menuKey(e));
    this.doc.addEventListener('click', (e) => {
      if (!menu.hidden && !slot.contains(e.target)) this.closeMenu(false);
    });
  }

  renderMenu() {
    const menu = this.doc.getElementById('style-menu');
    if (!menu) return;
    const items = (this.styles || []).map((s) => `<button type="button" role="menuitemradio" class="st-item" tabindex="-1"
        data-style-key="${stEsc(s.key)}" aria-checked="${s.key === this.current}">${stEsc(s.label)}${
        s.key === this.deployed ? ' <span class="st-tag">default</span>' : ''}</button>`).join('');
    const store = (this.store.available ? `kept in this browser only (localStorage ${ST_KEY})`
      : `not kept: ${this.store.reason} · lasts this visit`) + " · default: this deployment's (fleet file panel.style)";
    menu.innerHTML = `${items || '<span class="st-dim">styles…</span>'}<p class="st-store">${stEsc(store)}</p>`;
  }

  items() { return [...this.doc.querySelectorAll('#style-menu [role="menuitemradio"]')]; }

  openMenu(focus = 'checked') {
    const menu = this.doc.getElementById('style-menu');
    this.renderMenu();
    menu.hidden = false;
    this.doc.getElementById('style-button').setAttribute('aria-expanded', 'true');
    const items = this.items();
    const target = focus === 'last' ? items[items.length - 1] : (items.find((i) => i.getAttribute('aria-checked') === 'true') || items[0]);
    if (target) target.focus();
  }

  closeMenu(refocus = true) {
    const menu = this.doc.getElementById('style-menu');
    if (!menu || menu.hidden) return;
    menu.hidden = true;
    const btn = this.doc.getElementById('style-button');
    btn.setAttribute('aria-expanded', 'false');
    if (refocus) btn.focus();
  }

  menuKey(e) {
    const items = this.items();
    const i = items.indexOf(this.doc.activeElement);
    const go = (n) => { e.preventDefault(); if (items.length) items[(n + items.length) % items.length].focus(); };
    if (e.key === 'ArrowDown') go(i + 1);
    else if (e.key === 'ArrowUp') go(i - 1);
    else if (e.key === 'Home') go(0);
    else if (e.key === 'End') go(items.length - 1);
    else if (e.key === 'Escape') { e.preventDefault(); this.closeMenu(); }
    else if (e.key === 'Tab') this.closeMenu(false);
  }

  pick(key) {
    const entry = (this.styles || []).find((s) => s.key === key);
    if (!entry) return;
    this.unsay('unknown'); this.unsay('stored');
    if (entry.key === this.deployed) {
      // The deployment's default: this browser's own choice is forgotten, so it follows the deployment again.
      this.store.clear();
      this.apply(entry, { save: false });
    } else {
      this.apply(entry);
    }
    this.closeMenu();
  }

  // ── notices ────────────────────────────────────────────────────────────────────────────────

  /** One line per matter, keyed so it is replaced, not repeated. */
  say(key, text, actions) {
    const box = this.doc.getElementById('style-notice');
    if (!box) return;
    this.unsay(key);
    const row = this.doc.createElement('div');
    row.className = 'st-note';
    row.dataset.key = key;
    row.innerHTML = `<span>${stEsc(text)}</span>`;
    actions.forEach(([label, fn]) => {
      const b = this.doc.createElement('button');
      b.type = 'button'; b.className = 'st-act'; b.textContent = label;
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
    const box = this.doc.getElementById('style-notice');
    if (box) box.hidden = this.notes.size === 0;
  }
}

// ── boot ──────────────────────────────────────────────────────────────────────────────────────────
// At once in <head> (the node harness hands a document without a head, and runs nothing).
if (typeof document !== 'undefined' && document && document.head && document.documentElement) {
  let storage;
  try { storage = window.localStorage; } catch (e) { storage = { setItem() { throw e; }, getItem() { throw e; }, removeItem() {} }; }
  const early = stEarly(document, storage);
  const boot = () => {
    const sw = new StyleSwitch(document, window, new StStore(storage), async (url) => {
      const r = await fetch(url, { cache: 'no-cache' });
      if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
      return r.json();
    }, early);
    window.styleSwitch = sw;
    sw.boot();
  };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot); else boot();
}
