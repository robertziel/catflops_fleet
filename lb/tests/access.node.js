/* WHO IS WHAT, on the panel (vllm_lb/static/access.js): the admin's Access dialog over the gate's
 * /auth/access, and a member's page told it is one. Under node with a stub DOM and a fetch answered per URL.
 *
 * Pinned: the top bar gets `Access` ONLY when /auth/session says role admin (a member, an open gate, the LB's own 404
 * get none); the models offered are the catalog's served names and every served model, sorted, once; a person's row
 * shows their role and, for a member, a box per model ticked as granted (an admin has every model: no boxes); Save
 * POSTs {email, role, models} as JSON, same-origin, and shows the gate's refusal in its words; a member's page gets
 * `data-role="member"` on <html> and a line naming their models (none: says so).
 *
 *   docker run --rm -v "$PWD/lb":/lb:ro -w /lb node:22-slim node tests/access.node.js
 */
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const STATIC = path.join(__dirname, '..', 'vllm_lb', 'static');
const src = fs.readFileSync(path.join(STATIC, 'access.js'), 'utf8');
const html = fs.readFileSync(path.join(STATIC, 'index.html'), 'utf8');
let checks = 0;
const ok = (c, m) => { assert.ok(c, m); checks += 1; };
const eq = (a, b, m) => { assert.deepStrictEqual(a, b, m); checks += 1; };

class El {
  constructor(tag) {
    this.tagName = tag.toUpperCase(); this.children = []; this.attrs = {}; this.dataset = {}; this.on = {};
    this.textContent = ''; this.value = ''; this.checked = false; this.disabled = false; this.hidden = false;
    this.className = ''; this.type = ''; this.open = false;
  }
  append(...xs) { for (const x of xs) this.children.push(typeof x === 'string' ? Object.assign(new El('#text'), { textContent: x }) : x); }
  appendChild(x) { this.append(x); return x; }
  insertBefore(x) { this.children.unshift(x); return x; }
  replaceChildren(...xs) { this.children = []; this.append(...xs); }
  setAttribute(k, v) { this.attrs[k] = String(v); }
  getAttribute(k) { return this.attrs[k]; }
  addEventListener(k, f) { this.on[k] = f; }
  showModal() { this.open = true; }
  focus() { this.focused = true; }
  close() { this.open = false; }
  all() { return this.children.flatMap((c) => [c, ...c.all()]); }
  text() { return this.textContent + this.children.map((c) => c.text()).join(''); }
  find(pred) { return this.all().find(pred); }
}

function world(answers) {
  const byId = { 'top-right': new El('div'), access: new El('dialog'), 'connection-status': new El('div') };
  const root = new El('html');
  const document = {
    documentElement: root, readyState: 'complete',
    getElementById: (id) => byId[id] || null,
    createElement: (t) => new El(t),
    querySelector: (q) => (q === '.top-right' ? byId['top-right'] : null),
    addEventListener() {},
  };
  const sent = [];
  const fetch = async (url, opts = {}) => {
    sent.push({ url, opts });
    const a = typeof answers[url] === 'function' ? answers[url](opts) : answers[url];
    if (!a) return { status: 404, json: async () => ({}) };
    return { status: a[0], ok: a[0] >= 200 && a[0] < 300, json: async () => a[1] };
  };
  const api = new Function('document', 'fetch', `${src}\n;return { accessModels, accessRows, accessStart, accessOpen, accessSave, accessClose };`)(
    document, fetch);
  return { api, byId, root, sent };
}

const STATUS = {
  models: { cluster: [{ served_name: 'b/m' }, { served_name: 'a/m' }], classic: [{ served_name: 'a/m' }] },
  backends: [{ served_models: ['c/m', 'a/m'] }],
};
const PEOPLE = { roles: ['admin', 'member'], people: [
  { email: 'boss@example.com', role: 'admin', models: [] },
  { email: 'guest@example.com', role: 'member', models: ['a/m'] }] };

(async () => {
  // the models an admin may grant: catalog served names + what is served now, sorted, once
  const w0 = world({});
  eq(w0.api.accessModels(STATUS), ['a/m', 'b/m', 'c/m'], 'catalog + served, sorted, once');
  eq(w0.api.accessModels({}), [], 'nothing known: nothing offered (never a default list)');

  // rows: a member's boxes ticked as granted; an admin's row has none
  const rows = w0.api.accessRows(PEOPLE.people, ['a/m', 'b/m']);
  eq(rows[0].boxes, [], 'an admin has every model: no boxes');
  eq(rows[1].boxes, [{ model: 'a/m', checked: true }, { model: 'b/m', checked: false }], "the member's grants ticked");
  eq(w0.api.accessRows([{ email: 'x@example.com', role: 'member', models: ['gone/m'] }], ['a/m'])[0].boxes,
     [{ model: 'a/m', checked: false }, { model: 'gone/m', checked: true }], 'a granted model no longer offered still shows, ticked');

  // the button: an admin only
  for (const [session, want] of [[[200, { email: 'boss@example.com', role: 'admin' }], true],
                                  [[200, { email: 'guest@example.com', role: 'member', models: [] }], false],
                                  [[200, { email: 'anonymous', auth: 'false', role: 'admin' }], false],
                                  [null, false]]) {
    const w = world({ '/auth/session': session });
    await w.api.accessStart();
    eq(Boolean(w.byId['top-right'].find((e) => e.tagName === 'BUTTON' && e.text() === 'Access')), want,
       `Access button for ${JSON.stringify(session)}`);
  }

  // a member's page says so
  const wm = world({ '/auth/session': [200, { email: 'guest@example.com', role: 'member', models: ['a/m', 'b/m'] }] });
  await wm.api.accessStart();
  eq(wm.root.dataset.role, 'member', 'data-role on <html>');
  ok(wm.byId['top-right'].text().includes('member · a/m, b/m'), 'the line names their models');
  const wn = world({ '/auth/session': [200, { email: 'guest@example.com', role: 'member', models: [] }] });
  await wn.api.accessStart();
  ok(wn.byId['top-right'].text().includes('member · no models yet') && wn.byId['top-right'].find((e) => e.getAttribute && e.getAttribute('aria-label') === 'member — no models granted yet: ask an admin'), 'none: said so, in the text and the name');

  // the dialog: one row per person, Save POSTs JSON same-origin, the refusal in the gate's words
  let posted = null;
  const wa = world({
    '/auth/session': [200, { email: 'boss@example.com', role: 'admin' }],
    '/auth/access': (opts) => {
      if (opts.method === 'POST') { posted = JSON.parse(opts.body); return [400, { error: 'access_refused', detail: 'the last admin cannot be demoted: promote someone else first' }]; }
      return [200, PEOPLE];
    },
    '/lb/status': [200, STATUS],
  });
  await wa.api.accessStart();
  await wa.api.accessOpen();
  const dlg = wa.byId.access;
  ok(dlg.open, 'opens as a modal');
  ok(dlg.text().includes('boss@example.com') && dlg.text().includes('guest@example.com'), 'everyone listed');
  ok(dlg.text().includes('2 people · 1 admin'), 'the tally beside the title');
  const guestRow = dlg.find((e) => e.dataset && e.dataset.email === 'guest@example.com');
  const chips = guestRow.all().filter((e) => e.dataset && e.dataset.model);
  eq(chips.map((c) => [c.dataset.model, c.getAttribute('aria-pressed')]),
     [['a/m', 'true'], ['b/m', 'false'], ['c/m', 'false']], "the member's chips, lit as granted");
  const save = guestRow.find((e) => e.className === 'ac-save');
  ok(save.disabled && guestRow.parts.foot.hidden, 'a clean card: Save not live, its row hidden');
  chips[1].on.click();
  ok(!save.disabled && !guestRow.parts.foot.hidden && guestRow.dataset.dirty === 'true', 'a lit chip makes the card dirty');
  ok(guestRow.text().includes('2 of 3'), 'the count follows');
  const roles = guestRow.all().filter((e) => e.dataset && e.dataset.role);
  roles.find((r) => r.dataset.role === 'admin').on.click();
  ok(guestRow.parts.modelsWrap.hidden && !guestRow.parts.every.hidden, 'admin: no chips, every model');
  roles.find((r) => r.dataset.role === 'member').on.click();
  const bossRow = dlg.find((e) => e.dataset && e.dataset.email === 'boss@example.com');
  ok(bossRow.text().includes('every model') && bossRow.parts.modelsWrap.hidden, "an admin's card: every model, no chips");
  const res = await wa.api.accessSave(guestRow);
  eq(posted, { email: 'guest@example.com', role: 'member', models: ['a/m', 'b/m'] }, 'what Save sends');
  const post = wa.sent.find((s) => s.opts.method === 'POST');
  eq([post.opts.credentials, post.opts.headers['Content-Type']], ['same-origin', 'application/json'], 'JSON, same-origin');
  ok(!res && guestRow.text().includes('the last admin cannot be demoted'), "the gate's words");

  // a 200 that carries no `people` list is said, never a blank dialog
  const wb = world({ '/auth/session': [200, { email: 'boss@example.com', role: 'admin' }],
                     '/auth/access': [200, null], '/lb/status': [200, STATUS] });
  await wb.api.accessStart();
  await wb.api.accessOpen();
  ok(wb.byId.access.text().includes('/auth/access: the answer carries no list of people'), 'an unreadable 200 is said');

  // a member's page draws no admin action (gateway.css): press-and-hold controls, next-start settings, activate
  const css = fs.readFileSync(path.join(STATIC, 'gateway.css'), 'utf8');
  ok(/html\[data-role="member"\] \.hold, html\[data-role="member"\] #cluster-settings,\s*html\[data-role="member"\] \.cl-act \{ display: none !important; \}/.test(css),
     "a member's page hides .hold, #cluster-settings and .cl-act");

  // ── behaviour, not structure ──

  // a member's line is a BUTTON whose name carries role and models (a phone shows the mark and the count only), and
  // a tap opens their own access, read-only
  const wp = world({ '/auth/session': [200, { email: 'guest@example.com', role: 'member', models: ['a/m', 'b/m'] }] });
  await wp.api.accessStart();
  const mline = wp.byId['top-right'].find((e) => e.tagName === 'BUTTON' && /ac-member/.test(e.className));
  ok(mline, "the member's line is a button");
  eq(mline.getAttribute('aria-label'), 'member — you may use a/m, b/m', 'its accessible name: role and models');
  ok(mline.find((e) => e.className === 'ac-n' && e.textContent === '2'), "the count a phone shows");
  mline.on.click();
  ok(wp.byId.access.open && wp.byId.access.text().includes('your models') && wp.byId.access.text().includes('a/m')
     && wp.byId.access.text().includes('b/m'), 'a tap opens their models, read-only');
  ok(!wp.byId.access.find((e) => e.dataset && e.dataset.model), 'nothing in it to toggle');
  ok(wp.byId.access.find((e) => e.className === 'ac-lead').hidden, "the admin's explanation is not shown to a member");

  // refusal -> an edit clears the red line; a Save that lands makes the card clean and updates who changed it
  let answer = [400, { error: 'access_refused', detail: 'models must be a list of served model names' }];
  const wr = world({
    '/auth/session': [200, { email: 'boss@example.com', role: 'admin' }],
    '/auth/access': (opts) => (opts.method === 'POST' ? answer : [200, PEOPLE]),
    '/lb/status': [200, STATUS],
  });
  await wr.api.accessStart();
  await wr.api.accessOpen();
  const g = wr.byId.access.find((e) => e.dataset && e.dataset.email === 'guest@example.com');
  const gchips = g.all().filter((e) => e.dataset && e.dataset.model);
  gchips[1].on.click();
  ok(!(await wr.api.accessSave(g)) && g.parts.said.textContent.includes('served model names'), 'the refusal shown');
  eq(g.parts.said.dataset.kind, 'bad', 'in the refusal colour');
  gchips[2].on.click();
  eq(g.parts.said.textContent, '', 'an edit after a refusal clears it: the line speaks of the last Save only');
  answer = [200, { person: { email: 'guest@example.com', role: 'member', models: ['a/m', 'b/m', 'c/m'],
                             signed_in_at: 1, updated_by: 'boss@example.com', updated_at: 2 } }];
  ok(await wr.api.accessSave(g), 'a Save that lands');
  ok(g.parts.save.disabled && g.dataset.dirty === 'false', 'the card is clean again');
  ok(g.parts.meta.textContent.includes('changed by boss@example.com'), 'and says who changed it');
  ok(g.parts.said.textContent === 'saved' && !g.parts.foot.hidden, '"saved" stays until the next edit');

  // closing over unsaved edits says so once; a second close discards them
  gchips[0].on.click();
  ok(!wr.api.accessClose() && wr.byId.access.open, 'first close with a changed card stays open');
  ok(wr.byId.access.text().includes('Unsaved changes for guest@example.com'), 'and names whose');
  ok(wr.api.accessClose() && !wr.byId.access.open, 'a second close discards');
  // warn -> save -> edit ANOTHER card -> close: those edits are named too, never dropped on the old warning
  await wr.api.accessOpen();
  const g2 = wr.byId.access.find((e) => e.dataset && e.dataset.email === 'guest@example.com');
  const b2 = wr.byId.access.find((e) => e.dataset && e.dataset.email === 'boss@example.com');
  g2.all().filter((e) => e.dataset && e.dataset.model)[0].on.click();
  ok(!wr.api.accessClose(), 'warned for guest');
  ok(await wr.api.accessSave(g2), 'guest saved: no card is dirty now');
  b2.all().filter((e) => e.dataset && e.dataset.role === 'member')[0].on.click();
  ok(!wr.api.accessClose() && wr.byId.access.open, 'a new unsaved change warns again');
  ok(wr.byId.access.text().includes('Unsaved changes for boss@example.com'), 'naming the new card');

  // a hostile address is text, never markup (no innerHTML anywhere in access.js)
  const evil = '<img src=x onerror=alert(1)>@example.com';
  const wx = world({ '/auth/session': [200, { email: 'boss@example.com', role: 'admin' }],
                     '/auth/access': [200, { people: [{ email: evil, role: 'member', models: [] }] }],
                     '/lb/status': [200, STATUS] });
  await wx.api.accessStart();
  await wx.api.accessOpen();
  ok(wx.byId.access.find((e) => e.className === 'ac-email' && e.textContent === evil), 'the address as plain text');
  ok(!/innerHTML|insertAdjacentHTML|outerHTML/.test(src), 'access.js writes no markup');

  // nobody yet AND /lb/status down: both said, neither overwrites the other
  const wz = world({ '/auth/session': [200, { email: 'boss@example.com', role: 'admin' }],
                     '/auth/access': [200, { people: [] }], '/lb/status': [503, {}] });
  await wz.api.accessStart();
  await wz.api.accessOpen();
  ok(wz.byId.access.text().includes('Nobody has signed in yet.')
     && wz.byId.access.text().includes('/lb/status did not answer'), 'both notes kept');

  // the stale reason on a phone: renderFreshness writes the same words to #lb-stale (gateway.js), shown only where the
  // header hid #status-text for Access (gateway.css)
  const gsrc = fs.readFileSync(path.join(STATIC, 'gateway.js'), 'utf8');
  const fn = gsrc.slice(gsrc.indexOf('function renderFreshness()'), gsrc.indexOf('\n}\n', gsrc.indexOf('function renderFreshness()')) + 3);
  const els = { 'connection-status': { className: '' }, 'status-text': { textContent: '' }, 'lb-stale': { hidden: true, textContent: '' } };
  // stale after STALE_TICKS of the LB's tick (/lb/stream speaks at least every two): 3 × 3 s = 9 s here
  const run = (lastOk, lastError) => new Function('$', 'lastOk', 'lastError', 'loadedAt', 'lastStatus', 'STALE_TICKS',
    'CONNECT_WAIT_MS', `${fn}; renderFreshness();`)((id) => els[id], lastOk, lastError, Date.now(),
    { config: { tick_s: 3 } }, 3, 6000);
  run(Date.now() - 60000, 'HTTP 502');
  ok(!els['lb-stale'].hidden && els['lb-stale'].textContent === els['status-text'].textContent
     && els['lb-stale'].textContent.includes('HTTP 502'), 'stale: #lb-stale carries the reason');
  run(Date.now(), '');
  ok(els['lb-stale'].hidden, 'fresh again: #lb-stale hidden');
  // a member's backend rows carry no engine address (roles.member_status): the card names none, never "undefined"
  const hfn = gsrc.slice(gsrc.indexOf('function engineHost('), gsrc.indexOf('\n}\n', gsrc.indexOf('function engineHost(')) + 3);
  const engineHost = new Function(`${hfn}; return engineHost;`)();
  ok(engineHost({ url: 'http://10.0.0.7:8000' }) === '10.0.0.7:8000' && engineHost({}) === ''
     && engineHost({ url: 'not a url' }) === 'not a url', 'engineHost: the address, or nothing for a member');
  ok(gsrc.includes('[engineHost(b), c.hardware]') && gsrc.includes("srv.url ? ` at ${srv.url}` : ''")
     && !/url \? url\.host : b\.url/.test(gsrc), 'card meta uses engineHost; the connect title names an address only when it has one');
  const css2 = fs.readFileSync(path.join(STATIC, 'gateway.css'), 'utf8');
  ok(/\.lb-stale \{ display: none;/.test(css2) && /body:has\(\.ac-open, \.ac-member\) \.lb-stale:not\(\[hidden\]\) \{ display: block; \}/.test(css2),
     '#lb-stale shows only where the header lent its words to Access');
  ok(/<div id="lb-stale" class="alert lb-stale" role="status" hidden><\/div>/.test(html), 'index.html has #lb-stale');

  // the page carries the dialog and loads the script
  ok(/<dialog id="access"[^>]*aria-labelledby="access-title"/.test(html), 'index.html has the dialog');
  ok(html.includes('<script src="/access.js"></script>'), 'index.html loads access.js');

  console.log(`ok ${checks} checks`);
})().catch((e) => { console.error(e); process.exit(1); });
