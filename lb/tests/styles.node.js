/* THE STYLE SWITCH (vllm_lb/static/styles.js) under node: the registry it accepts, what a browser's stored choice may
 * be, the early apply that runs before first paint, the switch itself (a sheet loaded, THEN applied; the choice kept;
 * palette.js told), and the loud cases — a stored key the registry does not list, a stored value that does not parse,
 * a sheet that does not load, a browser that keeps nothing — each a visible notice with its action, never a silent
 * substitute (AGENTS.md: no fallbacks). A small fake document stands in for the page; the menu's markup is the
 * Chromium audit's (worklog).
 *
 *   node tests/styles.node.js        (tests/test_styles.py runs this when node exists)
 */
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const STATIC = path.join(__dirname, '..', 'vllm_lb', 'static');
const src = fs.readFileSync(path.join(STATIC, 'styles.js'), 'utf8');
// No head: the module's own boot does nothing under node.
const api = new Function('document', 'window', 'fetch', `${src}\n;return { ST, StyleSwitch };`)(
  { documentElement: null, head: null }, {}, () => new Promise(() => {}));
const { ST, StyleSwitch } = api;
const INDEX = JSON.parse(fs.readFileSync(path.join(STATIC, 'styles', 'index.json'), 'utf8'));
const throws = (fn, re) => { let e = null; try { fn(); } catch (x) { e = x; } assert.ok(e, 'expected a throw'); assert.ok(re.test(e.message), `${e.message} !~ ${re}`); };
let checks = 0;

// ── a fake page ────────────────────────────────────────────────────────────────────────────────────
// `deployed`: what the LB stamped on <html> (panel_styles.py) — the fleet's panel.style and its sheet; null = nothing
// stamped (an LB with no fleet file).
function fakeDoc(deployed = { key: 'vibe', href: '' }) {
  const byId = new Map();
  const listeners = new Map();
  const events = [];
  const el = (tag) => {
    const e = {
      tag, attrs: {}, children: [], listeners: {}, dataset: {}, hidden: false, innerHTML: '', textContent: '', sheet: null,
      parent: null,
      setAttribute(k, v) { this.attrs[k] = String(v); if (k === 'id') byId.set(String(v), this); },
      getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; },
      addEventListener(t, fn) { (this.listeners[t] = this.listeners[t] || []).push(fn); },
      fire(t) { (this.listeners[t] || []).forEach((fn) => fn({ type: t })); },
      appendChild(c) { c.parent = this; this.children.push(c); if (c.attrs.id) byId.set(c.attrs.id, c); return c; },
      remove() { if (this.parent) this.parent.children = this.parent.children.filter((x) => x !== this); if (this.attrs.id) byId.delete(this.attrs.id); this.parent = null; },
      focus() {},
    };
    for (const k of ['id', 'rel', 'href', 'className', 'type']) {
      Object.defineProperty(e, k, { get() { return this.attrs[k === 'className' ? 'class' : k] || ''; },
        set(v) { this.setAttribute(k === 'className' ? 'class' : k, v); } });
    }
    return e;
  };
  const doc = {
    documentElement: el('html'), head: el('head'), events,
    createElement: el,
    getElementById: (id) => byId.get(id) || null,
    querySelectorAll: () => [],
    addEventListener(t, fn) { (listeners.get(t) || listeners.set(t, []).get(t)).push(fn); },
    dispatchEvent(ev) { events.push(ev.type); (listeners.get(ev.type) || []).forEach((fn) => fn(ev)); },
  };
  const notice = el('div'); notice.setAttribute('id', 'style-notice'); notice.hidden = true;
  const menu = el('div'); menu.setAttribute('id', 'style-menu'); menu.hidden = true;
  doc.documentElement.setAttribute('data-style-default', deployed ? deployed.key : '');
  doc.documentElement.setAttribute('data-style-default-href', deployed ? deployed.href : '');
  return doc;
}
// The window: Event, and matchMedia's one query the switch watches — the system appearance (light / dark).
function fakeWin() {
  const queries = [];
  return {
    Event: class { constructor(type) { this.type = type; } },
    queries,
    matchMedia(q) {
      const mq = { media: q, matches: false, listeners: [], addEventListener(t, fn) { if (t === 'change') this.listeners.push(fn); } };
      queries.push(mq);
      return mq;
    },
  };
}
const win = fakeWin();
function memStorage(init = {}) {
  const m = new Map(Object.entries(init));
  return { getItem: (k) => (m.has(k) ? m.get(k) : null), setItem: (k, v) => m.set(k, String(v)), removeItem: (k) => m.delete(k), m };
}
const deadStorage = () => { const e = new Error('SecurityError: the operation is insecure'); return { getItem() { throw e; }, setItem() { throw e; }, removeItem() {} }; };
const notes = (doc) => doc.getElementById('style-notice').children.map((r) => ({ key: r.dataset.key, text: r.innerHTML, acts: r.children.map((b) => b.textContent) }));
const press = (doc, key, label) => {
  const row = doc.getElementById('style-notice').children.find((r) => r.dataset.key === key);
  row.children.find((b) => b.textContent === label).fire('click');
};
const sw = (doc, storage, early) => new StyleSwitch(doc, win, new ST.Store(storage), async () => JSON.parse(JSON.stringify(INDEX)), early);

// 1. THE SHIPPED REGISTRY parses, "Vibe coded" first and fileless; a malformed one is refused with where it is wrong.
{
  const styles = ST.parseIndex(INDEX);
  assert.deepStrictEqual(styles[0], { key: 'vibe', label: 'Vibe coded', file: null });
  assert.ok(styles.length >= 2 && styles.slice(1).every((s) => s.file === `/styles/${s.key}.css`));
  throws(() => ST.parseIndex({ styles: [] }), /lists no styles/);
  throws(() => ST.parseIndex({ styles: [{ key: 'clean', label: 'C', file: '/styles/clean.css' }] }), /first style must be "vibe"/);
  throws(() => ST.parseIndex({ styles: [INDEX.styles[0], INDEX.styles[0]] }), /listed twice/);
  throws(() => ST.parseIndex({ styles: [INDEX.styles[0], { key: 'x', label: 'X', file: 'https://cdn.example/x.css' }] }), /must name \/styles\/x.css/);
  throws(() => ST.parseIndex({ styles: [INDEX.styles[0], { key: 'x', label: 'X', file: '/styles/x.css', colour: 1 }] }), /unknown key "colour"/);
  // stamped by the build, still accepted
  assert.strictEqual(ST.parseIndex({ styles: [INDEX.styles[0], { key: 'x', label: 'X', file: '/styles/x.css?v=0123456789ab' }] })[1].key, 'x');
  checks += 8;
}

// 2. WHAT A BROWSER STORED: `{key, href}`; nothing is null; anything else throws, naming it — and an href that is not
//    one of our /styles/ files is never loaded from storage.
{
  assert.strictEqual(ST.readStored(memStorage()), null);
  assert.deepStrictEqual(ST.readStored(memStorage({ 'lb.style': '{"key":"clean","href":"/styles/clean.css?v=abcdef012345"}' })),
    { key: 'clean', href: '/styles/clean.css?v=abcdef012345' });
  assert.deepStrictEqual(ST.readStored(memStorage({ 'lb.style': '{"key":"clean"}' })), { key: 'clean', href: null });
  throws(() => ST.readStored(memStorage({ 'lb.style': 'clean' })), /not JSON/);
  throws(() => ST.readStored(memStorage({ 'lb.style': '{"key":"Bad Key"}' })), /no valid key/);
  throws(() => ST.readStored(memStorage({ 'lb.style': '{"key":"x","href":"https://evil.example/x.css"}' })), /not a \/styles\/ file/);
  assert.strictEqual(ST.readStored(deadStorage()), null, 'a storage that throws keeps nothing');
  checks += 7;
}

// 3. EARLY, BEFORE FIRST PAINT: the stored key on <html>, its sheet linked render-blocking; nothing stored → vibe and
//    no sheet; a stored value that does not parse → the default now, its reason kept for the notice.
{
  let doc = fakeDoc();
  let e = ST.early(doc, memStorage({ 'lb.style': '{"key":"clean","href":"/styles/clean.css?v=abcdef012345"}' }));
  assert.strictEqual(doc.documentElement.getAttribute('data-style'), 'clean');
  assert.ok(e.link && e.link.getAttribute('blocking') === 'render' && e.link.href === '/styles/clean.css?v=abcdef012345');
  assert.strictEqual(doc.head.children.length, 1);
  doc = fakeDoc();
  e = ST.early(doc, memStorage());
  assert.strictEqual(doc.documentElement.getAttribute('data-style'), 'vibe');
  assert.strictEqual(e.link, null);
  assert.strictEqual(doc.head.children.length, 0);
  doc = fakeDoc();
  e = ST.early(doc, memStorage({ 'lb.style': '{{' }));
  assert.strictEqual(doc.documentElement.getAttribute('data-style'), 'vibe');
  assert.ok(/not JSON/.test(e.error));
  checks += 8;
}

(async () => {
  // 4. THE SWITCH: picking a style loads its sheet and applies it only once loaded; the choice is stored with the
  //    registry's href; palette.js is told (`fleet:style`). Back to the default (vibe here) drops the sheet.
  {
    const doc = fakeDoc();
    const storage = memStorage();
    const s = sw(doc, storage, ST.early(doc, storage));
    await s.boot();
    assert.strictEqual(doc.documentElement.getAttribute('data-style'), 'vibe');
    s.pick('clean');
    const link = doc.head.children[0];
    assert.ok(link && link.href === '/styles/clean.css', 'the sheet is linked');
    assert.strictEqual(doc.documentElement.getAttribute('data-style'), 'vibe', 'not applied before it has loaded');
    const before = doc.events.length;
    link.sheet = {}; link.fire('load');
    assert.strictEqual(doc.documentElement.getAttribute('data-style'), 'clean');
    assert.strictEqual(doc.events.length, before + 1, 'palette.js is told once');
    assert.deepStrictEqual(JSON.parse(storage.m.get('lb.style')), { key: 'clean', href: '/styles/clean.css' });
    s.pick('vibe');
    assert.strictEqual(doc.head.children.length, 0, 'the default needs no sheet');
    assert.strictEqual(doc.documentElement.getAttribute('data-style'), 'vibe');
    assert.ok(!storage.m.has('lb.style'), 'back to the deployment default (here vibe) forgets the choice');
    assert.deepStrictEqual(notes(doc), []);
    checks += 9;
  }

  // 5. A STORED KEY THE REGISTRY DOES NOT LIST: the default is applied AND SAID — "style "x" not found — using Vibe
  //    coded" with [reset]; the stored value stays until the operator resets it.
  {
    const doc = fakeDoc();
    const storage = memStorage({ 'lb.style': '{"key":"retired","href":"/styles/retired.css?v=abcdef012345"}' });
    const s = sw(doc, storage, ST.early(doc, storage));
    await s.boot();
    assert.strictEqual(doc.documentElement.getAttribute('data-style'), 'vibe');
    assert.strictEqual(doc.head.children.length, 0, 'the unknown sheet is dropped');
    const n = notes(doc);
    assert.strictEqual(n.length, 1);
    assert.strictEqual(n[0].text, '<span>style &quot;retired&quot; not found — using Vibe coded (this deployment&#39;s default)</span>');
    assert.deepStrictEqual(n[0].acts, ['reset']);
    assert.ok(storage.m.has('lb.style'), 'nothing is changed behind the operator\'s back');
    press(doc, 'unknown', 'reset');
    assert.ok(!storage.m.has('lb.style') && notes(doc).length === 0 && doc.getElementById('style-notice').hidden);
    checks += 7;
  }

  // 6. A STORED VALUE THAT DOES NOT PARSE: said, with [reset].
  {
    const doc = fakeDoc();
    const storage = memStorage({ 'lb.style': 'not json' });
    const s = sw(doc, storage, ST.early(doc, storage));
    await s.boot();
    const n = notes(doc);
    assert.ok(n.length === 1 && /the stored style cannot be read .*not JSON.* — using Vibe coded/.test(n[0].text), JSON.stringify(n));
    press(doc, 'stored', 'reset');
    assert.ok(!storage.m.has('lb.style'));
    checks += 2;
  }

  // 7. A SHEET THAT DOES NOT LOAD: said, with [retry]; the page keeps the style it had.
  {
    const doc = fakeDoc();
    const storage = memStorage();
    const s = sw(doc, storage, ST.early(doc, storage));
    await s.boot();
    s.pick('clean');
    doc.head.children[0].fire('error');
    assert.strictEqual(doc.documentElement.getAttribute('data-style'), 'vibe');
    assert.strictEqual(doc.head.children.length, 0);
    const n = notes(doc);
    assert.ok(n.length === 1 && /style &quot;Clean&quot; did not load \(\/styles\/clean.css\) — still Vibe coded/.test(n[0].text), JSON.stringify(n));
    assert.deepStrictEqual(n[0].acts, ['retry']);
    assert.ok(!storage.m.has('lb.style'), 'a style that never applied is not stored');
    checks += 5;
  }

  // 8. A BROWSER THAT KEEPS NOTHING: the switch works for the visit, and says it is not kept.
  {
    const doc = fakeDoc();
    const storage = deadStorage();
    const s = sw(doc, storage, ST.early(doc, storage));
    await s.boot();
    s.pick('clean');
    const link = doc.head.children[0]; link.sheet = {}; link.fire('load');
    assert.strictEqual(doc.documentElement.getAttribute('data-style'), 'clean');
    const n = notes(doc);
    assert.ok(n.length === 1 && /style not kept in this browser: SecurityError.* — it lasts this visit/.test(n[0].text), JSON.stringify(n));
    checks += 2;
  }

  // 9. AN EARLY SHEET FROM AN OLDER BUILD: the registry's stamped href replaces it, and the store follows.
  {
    const doc = fakeDoc();
    const storage = memStorage({ 'lb.style': '{"key":"clean","href":"/styles/clean.css?v=000000000000"}' });
    const early = ST.early(doc, storage);
    early.link.sheet = {};
    const s = sw(doc, storage, early);
    await s.boot();
    const fresh = doc.head.children.find((l) => l.href === '/styles/clean.css');
    assert.ok(fresh, 'the registry\'s href is linked');
    fresh.sheet = {}; fresh.fire('load');
    assert.deepStrictEqual(doc.head.children.map((l) => l.href), ['/styles/clean.css'], 'the old link is gone');
    assert.strictEqual(JSON.parse(storage.m.get('lb.style')).href, '/styles/clean.css');
    checks += 3;
  }

  // 10. A REGISTRY THAT WILL NOT LOAD: said, with [retry]; the early-applied style stays (it is the operator's).
  {
    const doc = fakeDoc();
    const storage = memStorage({ 'lb.style': '{"key":"clean","href":"/styles/clean.css"}' });
    const early = ST.early(doc, storage);
    const s = new StyleSwitch(doc, win, new ST.Store(storage), async () => { throw new Error('/styles/index.json: HTTP 503'); }, early);
    await s.boot();
    assert.strictEqual(doc.documentElement.getAttribute('data-style'), 'clean');
    const n = notes(doc);
    assert.ok(n.length === 1 && n[0].text === '<span>styles not loaded: /styles/index.json: HTTP 503</span>' && n[0].acts[0] === 'retry');
    checks += 2;
  }

  // 11. THE SYSTEM APPEARANCE CHANGES (light ↔ dark): a style that follows it ("Glass") re-values its tokens under
  //     `prefers-color-scheme`, so palette.js's cached colours are stale — the switch tells it, as on a style change.
  {
    const doc = fakeDoc();
    const w = fakeWin();
    const storage = memStorage();
    const s = new StyleSwitch(doc, w, new ST.Store(storage), async () => JSON.parse(JSON.stringify(INDEX)), ST.early(doc, storage));
    await s.boot();
    const mq = w.queries.filter((q) => q.media === '(prefers-color-scheme: dark)');
    assert.strictEqual(mq.length, 1, 'the appearance is watched once');
    assert.strictEqual(mq[0].listeners.length, 1);
    const before = doc.events.filter((t) => t === 'fleet:style').length;
    mq[0].matches = true;
    mq[0].listeners[0]({ type: 'change', matches: true });
    assert.strictEqual(doc.events.filter((t) => t === 'fleet:style').length, before + 1, 'palette.js is told');
    assert.strictEqual(doc.documentElement.getAttribute('data-style'), 'vibe', 'the style itself is unchanged');
    checks += 4;
  }

  // 12. THE DEPLOYMENT'S DEFAULT (the fleet's panel.style, stamped on <html> by the LB): with nothing stored, the page
  //     opens in it BEFORE FIRST PAINT — its key on <html>, its sheet linked render-blocking — and nothing is stored, so
  //     this browser follows the deployment when the operator changes it.
  {
    const doc = fakeDoc({ key: 'glass', href: '/styles/glass.css?v=0123456789ab' });
    const storage = memStorage();
    const early = ST.early(doc, storage);
    assert.strictEqual(doc.documentElement.getAttribute('data-style'), 'glass');
    assert.ok(early.link && early.link.getAttribute('blocking') === 'render' && early.link.href === '/styles/glass.css?v=0123456789ab');
    assert.deepStrictEqual(early.deployed, { key: 'glass', href: '/styles/glass.css?v=0123456789ab' });
    early.link.sheet = {};
    const s = sw(doc, storage, early);
    await s.boot();
    // the registry (unstamped here) names another href: the registry's is linked, and STILL nothing is stored
    const fresh = doc.head.children.find((l) => l.href === '/styles/glass.css');
    assert.ok(fresh, 'the registry\'s href is linked');
    fresh.sheet = {}; fresh.fire('load');
    assert.strictEqual(doc.documentElement.getAttribute('data-style'), 'glass');
    assert.ok(!storage.m.has('lb.style'), 'following the deployment default stores nothing');
    assert.deepStrictEqual(notes(doc), []);
    checks += 6;
  }

  // 13. A BROWSER'S OWN CHOICE BEATS THE DEPLOYMENT'S DEFAULT.
  {
    const doc = fakeDoc({ key: 'glass', href: '/styles/glass.css' });
    const storage = memStorage({ 'lb.style': '{"key":"mono","href":"/styles/mono.css"}' });
    const early = ST.early(doc, storage);
    assert.strictEqual(doc.documentElement.getAttribute('data-style'), 'mono');
    assert.deepStrictEqual(doc.head.children.map((l) => l.href), ['/styles/mono.css'], 'only the chosen sheet');
    early.link.sheet = {};
    const s = sw(doc, storage, early);
    await s.boot();
    assert.strictEqual(doc.documentElement.getAttribute('data-style'), 'mono');
    checks += 3;
  }

  // 14. THE MENU MARKS THE DEPLOYMENT'S DEFAULT; picking it FORGETS this browser's choice (it follows the deployment
  //     again), picking another keeps it.
  {
    const doc = fakeDoc({ key: 'clean', href: '/styles/clean.css' });
    const storage = memStorage({ 'lb.style': '{"key":"mono","href":"/styles/mono.css"}' });
    const early = ST.early(doc, storage);
    early.link.sheet = {};
    const s = sw(doc, storage, early);
    await s.boot();
    const menu = doc.getElementById('style-menu').innerHTML;
    const item = (k) => (menu.match(new RegExp(`<button[^>]*data-style-key="${k}"[^>]*>(.*?)</button>`, 's')) || [])[1];
    assert.ok(/st-tag[^>]*>default</.test(item('clean')), `the default is tagged: ${item('clean')}`);
    assert.ok(!/st-tag/.test(item('mono')) && !/st-tag/.test(item('vibe')), 'only the default is tagged');
    assert.ok(/default: this deployment&#39;s \(fleet file panel.style\)/.test(menu), menu);
    s.pick('clean');
    const link = doc.head.children.find((l) => l.href === '/styles/clean.css');
    link.sheet = {}; link.fire('load');
    assert.strictEqual(doc.documentElement.getAttribute('data-style'), 'clean');
    assert.ok(!storage.m.has('lb.style'), 'choosing the default forgets the browser\'s own choice');
    s.pick('vibe');
    assert.deepStrictEqual(JSON.parse(storage.m.get('lb.style')), { key: 'vibe', href: null }, 'another is kept');
    checks += 6;
  }

  // 15. A STORED KEY THE REGISTRY DOES NOT LIST, on a deployment whose default is not the base: the DEPLOYMENT'S default
  //     is applied and said.
  {
    const doc = fakeDoc({ key: 'clean', href: '/styles/clean.css' });
    const storage = memStorage({ 'lb.style': '{"key":"retired","href":"/styles/retired.css"}' });
    const early = ST.early(doc, storage);
    const s = sw(doc, storage, early);
    await s.boot();
    const link = doc.head.children.find((l) => l.href === '/styles/clean.css');
    assert.ok(link, 'the deployment default\'s sheet is linked');
    link.sheet = {}; link.fire('load');
    assert.strictEqual(doc.documentElement.getAttribute('data-style'), 'clean');
    assert.deepStrictEqual(doc.head.children.map((l) => l.href), ['/styles/clean.css'], 'the unknown sheet is dropped');
    const n = notes(doc);
    assert.ok(n.length === 1 && n[0].text === '<span>style &quot;retired&quot; not found — using Clean (this deployment&#39;s default)</span>', JSON.stringify(n));
    assert.ok(storage.m.has('lb.style'), 'kept until the operator resets it');
    checks += 5;
  }

  // 16. AN LB THAT STAMPED NO DEFAULT (no fleet file): the page opens in gateway.css's base and SAYS so — never a
  //     silent stand-in for the deployment's choice.
  {
    const doc = fakeDoc(null);
    const storage = memStorage();
    const early = ST.early(doc, storage);
    assert.strictEqual(early.deployed, null);
    assert.strictEqual(doc.documentElement.getAttribute('data-style'), 'vibe');
    const s = sw(doc, storage, early);
    await s.boot();
    const n = notes(doc);
    assert.ok(n.length === 1 && /this gateway names no default style \(its fleet file&#39;s panel.style\) — showing Vibe coded/.test(n[0].text), JSON.stringify(n));
    checks += 3;
  }

  // 17. A STAMPED KEY THIS PAGE'S REGISTRY DOES NOT LIST (a page from an older build): said, never silently swapped.
  {
    const doc = fakeDoc({ key: 'neon', href: '/styles/neon.css' });
    const storage = memStorage();
    const early = ST.early(doc, storage);
    const s = sw(doc, storage, early);
    await s.boot();
    assert.strictEqual(doc.documentElement.getAttribute('data-style'), 'vibe');
    const n = notes(doc);
    assert.ok(n.some((x) => /this deployment&#39;s default style &quot;neon&quot; is not in this page&#39;s styles — showing Vibe coded/.test(x.text)), JSON.stringify(n));
    checks += 2;
  }

  console.log(`ok — ${checks} style-switch checks`);
})().catch((e) => { console.error(e); process.exit(1); });
