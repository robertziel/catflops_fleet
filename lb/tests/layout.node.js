/* THE LAYOUT ENGINE'S PURE HALF (vllm_lb/static/layout.js `LY`), under node: what a template or a stored layout must
 * look like, how an arrangement changes under a move / hide / show / resize, how a browser's stored state is
 * versioned and migrated, what happens to an id the registry no longer has, what a browser with no storage gets,
 * and that an export imports back to the same arrangement. The DOM half — slots moved between columns, the drag,
 * the menu — is the Chromium audit's (worklog); nothing here needs a DOM, which is the point of keeping it pure.
 *
 *   node tests/layout.node.js        (tests/test_layouts.py runs this when node exists)
 */
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const STATIC = path.join(__dirname, '..', 'vllm_lb', 'static');
const src = fs.readFileSync(path.join(STATIC, 'layout.js'), 'utf8');
// No DOM: `getElementById` answers null for everything, so the module's boot (which looks for #layout) does nothing.
const document = { getElementById: () => null, querySelectorAll: () => [], addEventListener() {}, body: null };
const window = { addEventListener() {}, matchMedia: () => ({ matches: false }) };
const LY = new Function('document', 'window', 'fetch', 'location', `${src}\n;return LY;`)(
  document, window, () => new Promise(() => {}), { search: '', origin: '' });

// THE REGISTRY IS THE MARKUP: the same ids tests/test_layouts.py reads, read the same way.
const html = fs.readFileSync(path.join(STATIC, 'index.html'), 'utf8');
const REGISTRY = [...html.matchAll(/data-ly="([^"]+)"/g)].map((m) => m[1]);
const template = (name) => JSON.parse(fs.readFileSync(path.join(STATIC, 'layouts', `${name}.json`), 'utf8'));
const INDEX = JSON.parse(fs.readFileSync(path.join(STATIC, 'layouts', 'index.json'), 'utf8')).templates;
const ids = (layout) => LY.placedIds(layout);
const throws = (fn, re) => { let e = null; try { fn(); } catch (x) { e = x; } assert.ok(e, 'expected a throw'); assert.ok(re.test(e.message), `${e.message} !~ ${re}`); };
let checks = 0;

// 1. EVERY SHIPPED TEMPLATE NORMALISES AND IS COMPLETE AGAINST THE REGISTRY — and the engine's tokens are the
//    python test's (test_layouts.py WIDTHS / SIZES / MAX_COLS).
{
  assert.deepStrictEqual(LY.WIDTHS, ['side', '1', '2', '3']);
  assert.deepStrictEqual(LY.SIZES, ['full', 'half']);
  assert.strictEqual(LY.MAX_COLS, 3);
  assert.strictEqual(LY.SCHEMA, 1);
  assert.ok(REGISTRY.length >= 7 && INDEX[0] === 'default');
  INDEX.forEach((name) => {
    const t = template(name);
    const layout = LY.normalize(t);
    const v = LY.validate(layout, REGISTRY);
    assert.deepStrictEqual(v, { unknown: [], missing: [], duplicate: [] }, `${name}: ${JSON.stringify(v)}`);
    assert.deepStrictEqual([...ids(layout), ...layout.hidden].sort(), [...REGISTRY].sort(), name);
  });
  // `default` is the panel as it was: every element, in document order, none hidden, every item full width.
  const d = LY.normalize(template('default'));
  assert.deepStrictEqual(ids(d), REGISTRY);
  assert.deepStrictEqual(d.hidden, []);
  d.rows.forEach((r) => r.cols.forEach((c) => c.items.forEach((it) => assert.strictEqual(it.size, 'full'))));
  checks += 6 + INDEX.length * 2;
}

// 2. NORMALISE IS STRICT: a template is refused with the reason, never patched — a misspelt key, an unknown width
//    or size, a fourth column, no rows, an item that is not an id.
{
  const one = (rows, hidden) => ({ rows, hidden });
  const col = (items, width = '1') => ({ width, items });
  throws(() => LY.normalize({}), /rows/);
  throws(() => LY.normalize(one([])), /rows.*empty/);
  throws(() => LY.normalize(one([{ cols: [] }])), /row 1.*cols.*empty/);
  throws(() => LY.normalize(one([{ cols: [col(['a'], 'wide')] }])), /width "wide".*side, 1, 2, 3/);
  throws(() => LY.normalize(one([{ cols: [col([{ id: 'a', size: 'third' }])] }])), /size "third".*full, half/);
  throws(() => LY.normalize(one([{ cols: [col(['a']), col(['b']), col(['c']), col(['d'])] }])), /4 columns.*at most 3/);
  throws(() => LY.normalize(one([{ cols: [col([7])] }])), /item.*not an id/);
  throws(() => LY.normalize(one([{ cols: [col(['a'])], extra: 1 }])), /row 1.*unknown key "extra"/);
  throws(() => LY.normalize(one([{ cols: [{ witdh: '1', items: [] }] }])), /column 1.*unknown key "witdh"/);
  throws(() => LY.normalize(one([{ cols: [col([{ id: 'a', span: 2 }])] }])), /unknown key "span"/);
  throws(() => LY.normalize(one([{ cols: [col(['a'])] }], 'b')), /hidden.*list/);
  // A string item is a full-width item; `hidden` may be left out.
  const n = LY.normalize({ rows: [{ cols: [col(['a', { id: 'b', size: 'half' }])] }] });
  assert.deepStrictEqual(n, { rows: [{ cols: [{ width: '1', items: [{ id: 'a', size: 'full' }, { id: 'b', size: 'half' }] }] }], hidden: [] });
  checks += 12;
}

// 3. VALIDATE CLASSIFIES, IT NEVER DROPS: an id the registry lacks is `unknown` (the engine draws a notice tile for
//    it), a registry id the layout does not name is `missing` (the engine appends it and says so), an id named
//    twice is `duplicate`.
{
  const layout = LY.normalize({ rows: [{ cols: [{ width: '1', items: ['server', 'gone', 'server'] }] }], hidden: ['tabs'] });
  assert.deepStrictEqual(LY.validate(layout, ['server', 'tabs', 'new']), { unknown: ['gone'], missing: ['new'], duplicate: ['server'] });
  checks += 1;
}

// 4. THE REDUCER: every change is a new object (the old one is untouched), and a move past an edge is a no-op.
{
  const L = LY.normalize({ rows: [
    { cols: [{ width: '1', items: ['queued', 'cluster', 'tabs'] }] },
    { cols: [{ width: 'side', items: ['server'] }, { width: '1', items: ['engine', 'requests', 'saturation'] }] }] });
  const frozen = JSON.stringify(L);
  const at = (l, id) => LY.locate(l, id);
  assert.deepStrictEqual(at(L, 'requests'), { r: 1, c: 1, i: 1 });
  assert.strictEqual(at(L, 'nope'), null);
  // up / down inside a column
  assert.deepStrictEqual(ids(LY.move(L, 'requests', 'up')).slice(4), ['requests', 'engine', 'saturation']);
  assert.deepStrictEqual(ids(LY.move(L, 'requests', 'down')).slice(4), ['engine', 'saturation', 'requests']);
  // up past the top of a column: to the END of the row above (same column index, or its last column)
  const up = LY.move(L, 'engine', 'up');
  assert.deepStrictEqual(ids(up), ['queued', 'cluster', 'tabs', 'engine', 'server', 'requests', 'saturation']);
  assert.deepStrictEqual(at(up, 'engine'), { r: 0, c: 0, i: 3 });
  // down past the bottom: to the START of the row below
  const down = LY.move(L, 'tabs', 'down');
  assert.deepStrictEqual(at(down, 'tabs'), { r: 1, c: 0, i: 0 });
  // left / right across the row's columns, keeping the index where the column has it
  const left = LY.move(L, 'requests', 'left');
  assert.deepStrictEqual(at(left, 'requests'), { r: 1, c: 0, i: 1 });
  const right = LY.move(L, 'server', 'right');
  assert.deepStrictEqual(at(right, 'server'), { r: 1, c: 1, i: 0 });
  // the edges: nothing moves, the same arrangement comes back
  assert.deepStrictEqual(LY.move(L, 'queued', 'up'), L);
  assert.deepStrictEqual(LY.move(L, 'saturation', 'down'), L);
  assert.deepStrictEqual(LY.move(L, 'server', 'left'), L);
  assert.deepStrictEqual(LY.move(L, 'requests', 'right'), L);
  assert.deepStrictEqual(LY.move(L, 'ghost', 'up'), L);
  throws(() => LY.move(L, 'server', 'sideways'), /direction "sideways"/);
  // place (a drop): anywhere, by row / column / index; past the end clamps
  const dropped = LY.place(L, 'cluster', 1, 1, 1);
  assert.deepStrictEqual(ids(dropped), ['queued', 'tabs', 'server', 'engine', 'cluster', 'requests', 'saturation']);
  assert.deepStrictEqual(at(LY.place(L, 'cluster', 1, 1, 99), 'cluster'), { r: 1, c: 1, i: 3 });
  throws(() => LY.place(L, 'cluster', 5, 0, 0), /row 6/);
  // hide parks it; show appends to the last column of the last row; both leave the registry's set intact
  const hid = LY.hide(L, 'requests');
  assert.deepStrictEqual(hid.hidden, ['requests']);
  assert.ok(!ids(hid).includes('requests'));
  assert.deepStrictEqual(LY.hide(hid, 'requests'), hid);           // hiding a hidden element: nothing
  const shown = LY.show(hid, 'requests');
  assert.deepStrictEqual(shown.hidden, []);
  assert.deepStrictEqual(at(shown, 'requests'), { r: 1, c: 1, i: 2 });
  // resize: full ↔ half, on the item alone
  const half = LY.resize(L, 'requests', 'half');
  assert.strictEqual(half.rows[1].cols[1].items[1].size, 'half');
  assert.strictEqual(half.rows[1].cols[1].items[0].size, 'full');
  throws(() => LY.resize(L, 'requests', 'third'), /size "third"/);
  // a column's width, a new column (at most three), a new row
  assert.strictEqual(LY.setWidth(L, 1, 0, '2').rows[1].cols[0].width, '2');
  throws(() => LY.setWidth(L, 1, 0, 'wide'), /width "wide"/);
  const col3 = LY.addCol(L, 1);
  assert.strictEqual(col3.rows[1].cols.length, 3);
  assert.deepStrictEqual(col3.rows[1].cols[2], { width: '1', items: [] });
  throws(() => LY.addCol(col3, 1), /at most 3/);
  const row3 = LY.addRow(L);
  assert.strictEqual(row3.rows.length, 3);
  // prune drops empty columns and rows — never the last row — so a saved layout has no blank space
  const pruned = LY.prune(LY.addRow(LY.addCol(L, 1)));
  assert.deepStrictEqual(pruned, L);
  const emptied = LY.prune(LY.normalize({ rows: [{ cols: [{ width: '1', items: [] }] }], hidden: ['server'] }));
  assert.deepStrictEqual(emptied.rows, [{ cols: [{ width: '1', items: [] }] }]);
  assert.strictEqual(JSON.stringify(L), frozen, 'the reducer never writes into its input');
  checks += 34;
}

// 5. THE STORED STATE IS VERSIONED: `schema` is read first; a newer panel's state is refused by name, an older one
//    walks the migration table one step at a time, a missing step is a loud stop — never a silent re-read.
{
  const layout = LY.normalize(template('default'));
  const good = { schema: 1, template: 'default', layout, edited: true };
  assert.deepStrictEqual(LY.migrate(good), { schema: 1, template: 'default', layout, edited: true });
  throws(() => LY.migrate({ schema: 2, template: 'default', layout }), /schema 2.*this panel reads 1/);
  throws(() => LY.migrate({ template: 'default', layout }), /no schema/);
  throws(() => LY.migrate('{}'), /not an object/);
  throws(() => LY.migrate({ schema: 1, layout }), /template/);
  throws(() => LY.migrate({ schema: 1, template: 'default' }), /layout/);
  throws(() => LY.migrate({ schema: 1, template: 'default', layout: { rows: [] } }), /rows.*empty/);
  // a migration path: schema 0 → 1 through an injected table (schema 1 is the first that ever shipped, so the
  // shipped table is empty; the loop is what is proven here)
  const legacy = { schema: 0, name: 'default', arrangement: { rows: layout.rows } };
  throws(() => LY.migrate(legacy), /no migration from schema 0/);
  const migrated = LY.migrate(legacy, { 0: (s) => ({ schema: 1, template: s.name, layout: s.arrangement, edited: false }) });
  assert.deepStrictEqual(migrated, { schema: 1, template: 'default', layout, edited: false });
  assert.deepStrictEqual(LY.MIGRATIONS, {});
  checks += 10;
}

// 6. THE STORE: a browser that keeps nothing is SAID, with the reason — the panel works for the visit. A store that
//    holds junk is refused with the reason (and the engine then offers a reset), never read as something.
{
  const mem = new Map();
  const storage = { getItem: (k) => (mem.has(k) ? mem.get(k) : null), setItem: (k, v) => mem.set(k, String(v)), removeItem: (k) => mem.delete(k) };
  const store = new LY.Store(storage);
  assert.strictEqual(store.available, true);
  assert.strictEqual(store.load(), null);
  const layout = LY.normalize(template('default'));
  store.save({ schema: 1, template: 'default', layout, edited: true });
  assert.deepStrictEqual(store.load(), { schema: 1, template: 'default', layout, edited: true });
  assert.ok(mem.has(LY.KEY) && !mem.has(`${LY.KEY}.probe`), 'the probe key is removed after the probe');
  store.clear();
  assert.strictEqual(store.load(), null);
  mem.set(LY.KEY, '{not json');
  throws(() => store.load(), /stored layout.*JSON/);
  mem.set(LY.KEY, JSON.stringify({ schema: 9 }));
  throws(() => store.load(), /schema 9/);

  // no storage at all (a browser that blocks it, a private window): nothing throws out of the engine, the reason is kept
  const broken = { getItem() { throw new Error('SecurityError: blocked'); }, setItem() { throw new Error('SecurityError: blocked'); }, removeItem() {} };
  const none = new LY.Store(broken);
  assert.strictEqual(none.available, false);
  assert.ok(/blocked/.test(none.reason), none.reason);
  assert.strictEqual(none.load(), null);
  assert.strictEqual(none.save({ schema: 1, template: 'default', layout, edited: true }), false);
  const absent = new LY.Store(undefined);
  assert.strictEqual(absent.available, false);
  assert.ok(/no localStorage/.test(absent.reason), absent.reason);
  checks += 12;
}

// 7. EXPORT / IMPORT ROUND-TRIP: the text a browser downloads is the whole state, and pasted into another browser it
//    is the same arrangement — validated like a template (unknown ids classified, never dropped), refused with the
//    reason when it is not a layout at all.
{
  const layout = LY.resize(LY.hide(LY.normalize(template('default')), 'saturation'), 'requests', 'half');
  const state = { schema: 1, template: 'default', layout, edited: true };
  const text = LY.exportText(state);
  assert.ok(text.startsWith('{\n') && text.includes('"schema": 1'), 'readable JSON');
  const back = LY.importText(text);
  assert.deepStrictEqual(back, state);
  assert.deepStrictEqual(LY.validate(back.layout, REGISTRY), { unknown: [], missing: [], duplicate: [] });
  throws(() => LY.importText('not json'), /JSON/);
  throws(() => LY.importText('[]'), /not an object/);
  // an import naming an element this panel does not have is classified, not dropped: the engine shows the notice
  const foreign = LY.importText(JSON.stringify({ schema: 1, template: 'x', layout: { rows: [{ cols: [{ width: '1', items: ['server', 'widget'] }] }], hidden: [] } }));
  assert.deepStrictEqual(LY.validate(foreign.layout, REGISTRY).unknown, ['widget']);
  checks += 6;
}

// 8. A TEMPLATE FILE'S METADATA is what the menu shows: name, label, what; the engine reads a template through the
//    same `normalize` the stored state goes through, so there is one reader.
{
  INDEX.forEach((name) => {
    const t = template(name);
    assert.strictEqual(t.name, name);
    assert.ok(t.label && t.what, name);
    const m = LY.templateMeta(t);
    assert.deepStrictEqual(m, { name, label: t.label, what: t.what });
  });
  throws(() => LY.templateMeta({ name: 'x' }), /label/);
  throws(() => LY.templateMeta({ label: 'x', what: 'y' }), /name/);
  checks += INDEX.length * 3 + 2;
}

console.log(`ok · ${checks} checks`);
