/* WHO IS WHAT: an admin has everything; a member cannot shut a box down and uses the models granted to
 * them, kept in a database an admin edits here.
 *
 * The rows live in the front proxy's sign-in service: GET /auth/access lists everyone who has signed in, POST
 * /auth/access {email, role, models} changes one — the admin's alone, same-origin. This script is the panel's face of it,
 * drawn in the panel's own language (the Connect dialog's box, head and close; its pressed-tab look for choices):
 *
 *   ADMIN    `Access` beside Style and Layout opens a dialog: one card per person — who, when they last signed in and
 *            who changed them; a two-way switch admin | member; for a member, a chip per model, lit when granted
 *            (an admin has every model: no chips). The models offered are the catalog's served names and whatever a
 *            server serves now (/lb/status), sorted, once; a granted model no longer offered still shows, lit, so a
 *            Save never drops it unseen. Save is live only when the card differs from what the gate holds; the gate's
 *            refusal is shown in its words (the last admin, an unknown model name, …).
 *   MEMBER   no button; <html data-role="member"> and a line in the bar naming their models (none: said so). The LB
 *            already narrows what they see and refuses its controls (lb/vllm_lb/roles.py): this only says so.
 *   NEITHER  an open gate (auth "false": its one user is the operator) and the LB's own port (no /auth/: 404) add nothing.
 */
'use strict';

const ACCESS = { roles: ['admin', 'member'] };
const ac = { parts: null, models: [], me: null };

function acEl(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

function acButton(cls, text, label) {
  const b = acEl('button', cls, text);
  b.type = 'button';
  if (label) b.setAttribute('aria-label', label);
  return b;
}

async function acAsk(url, opts) {
  try {
    const r = await fetch(url, { cache: 'no-store', credentials: 'same-origin', ...(opts || {}) });
    let body = null;
    try { body = await r.json(); } catch (_) { body = null; }
    return { status: r.status, body };
  } catch (e) {
    return { status: 0, body: null, error: e.message };
  }
}

/** "just now", "12 min ago", "3 h ago", "4 d ago" — from a unix time; "" when there is none. */
function acAgo(t, now) {
  if (!Number.isFinite(t)) return '';
  const s = Math.max(0, (now === undefined ? Date.now() / 1000 : now) - t);
  if (s < 60) return 'just now';
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  return `${Math.floor(s / 86400)} d ago`;
}

/** The models an admin may grant: the catalog's served names and every model a server serves now, sorted, once. */
function accessModels(status) {
  const s = status || {};
  const m = s.models || {};
  const names = new Set();
  for (const row of [...(m.cluster || []), ...(m.classic || [])]) if (row && typeof row.served_name === 'string') names.add(row.served_name);
  for (const b of s.backends || []) for (const n of b.served_models || []) if (typeof n === 'string') names.add(n);
  return [...names].sort();
}

/** One row per person: an admin has every model (no chips lit); a member a chip per model, lit as granted. */
function accessRows(people, models) {
  return (people || []).map((p) => {
    const granted = new Set(p.models || []);
    const offered = [...new Set([...models, ...granted])].sort();
    return { email: p.email, role: p.role, signed_in_at: p.signed_in_at, updated_by: p.updated_by,
             updated_at: p.updated_at,
             boxes: p.role === 'admin' ? [] : offered.map((model) => ({ model, checked: granted.has(model) })) };
  });
}

/** What a card says now: its role and, for a member, the lit chips. */
function acState(card) {
  const p = card.parts;
  const role = p.roleBtns.find((b) => b.getAttribute('aria-pressed') === 'true').dataset.role;
  const models = role === 'admin' ? [] : p.chips.filter((c) => c.getAttribute('aria-pressed') === 'true')
    .map((c) => c.dataset.model).sort();
  return { role, models };
}

/** Re-draw one card from its state: which half of the switch is lit, chips or "every model", Save live when dirty. */
function acPaint(card) {
  const p = card.parts;
  const now = acState(card);
  const member = now.role === 'member';
  p.modelsWrap.hidden = !member;
  p.every.hidden = member;
  p.count.textContent = member ? `${now.models.length} of ${p.chips.length}` : '';
  const dirty = now.role !== card.saved.role || now.models.join('\n') !== card.saved.models.join('\n');
  p.save.disabled = !dirty;
  card.dataset.dirty = String(dirty);
  // the line under a card speaks of the LAST Save only: an edit since clears it, saved or refused
  if (card.edited && p.said.dataset.kind) { p.said.textContent = ''; p.said.dataset.kind = ''; }
  card.edited = false;
  // the Save row only while there is something to save or to say: a clean card is who + role + models, no more
  p.foot.hidden = !dirty && !p.said.textContent;
}

function acCard(row) {
  const card = acEl('li', 'ac-person');
  card.dataset.email = row.email;
  card.saved = { role: row.role, models: row.boxes.filter((b) => b.checked).map((b) => b.model).sort() };

  const top = acEl('div', 'ac-top');
  const who = acEl('div', 'ac-who');
  const mark = acEl('span', 'ac-mark', (row.email || '?').charAt(0).toUpperCase());
  mark.setAttribute('aria-hidden', 'true');
  who.append(mark, acEl('span', 'ac-email', row.email));
  if (ac.me && row.email && ac.me.toLowerCase() === row.email.toLowerCase()) who.append(acEl('span', 'ac-you', 'you'));
  const sw = acEl('div', 'ac-switch');
  sw.setAttribute('role', 'group');
  sw.setAttribute('aria-label', `Role of ${row.email}`);
  const roleBtns = ACCESS.roles.map((r) => {
    const b = acButton('ac-role', r);
    b.dataset.role = r;
    b.setAttribute('aria-pressed', String(r === row.role));
    return b;
  });
  for (const b of roleBtns) {
    b.addEventListener('click', () => {
      for (const o of roleBtns) o.setAttribute('aria-pressed', String(o === b));
      card.edited = true;
      acPaint(card);
    });
  }
  sw.append(...roleBtns);
  top.append(who, sw);

  const meta = acEl('p', 'ac-meta');
  const bits = [];
  if (row.signed_in_at) bits.push(`signed in ${acAgo(row.signed_in_at)}`);
  if (row.updated_by) bits.push(`changed by ${row.updated_by}${row.updated_at ? ` ${acAgo(row.updated_at)}` : ''}`);
  meta.textContent = bits.join(' · ');

  const every = acEl('p', 'ac-every', 'every model · Start / Stop · power · benchmarks');
  const modelsWrap = acEl('div', 'ac-models');
  const head = acEl('div', 'ac-models-head');
  const count = acEl('span', 'ac-count');
  head.append(acEl('span', 'ac-label', 'models'), count);
  const chips = acEl('div', 'ac-chips');
  chips.setAttribute('role', 'group');
  chips.setAttribute('aria-label', `Models ${row.email} may use`);
  const chipBtns = [];
  // an admin's card still carries the chips, unlit, so switching it to member offers the models at once
  const offered = row.role === 'admin' ? ac.models.map((model) => ({ model, checked: false })) : row.boxes;
  for (const b of offered) {
    const c = acButton('ac-chip', b.model);
    c.dataset.model = b.model;
    c.setAttribute('aria-pressed', String(b.checked));
    c.addEventListener('click', () => {
      c.setAttribute('aria-pressed', String(c.getAttribute('aria-pressed') !== 'true'));
      card.edited = true;
      acPaint(card);
    });
    chipBtns.push(c);
    chips.append(c);
  }
  if (!chipBtns.length) chips.append(acEl('span', 'ac-none', 'no model on offer: the catalog and the servers list none'));
  modelsWrap.append(head, chips);

  const foot = acEl('div', 'ac-foot');
  const said = acEl('span', 'ac-said');
  said.setAttribute('role', 'status');
  said.setAttribute('tabindex', '-1');      // where focus goes when a Save answers (its button may be disabled)
  const save = acButton('ac-save', 'Save', `Save ${row.email}`);
  save.addEventListener('click', () => accessSave(card));
  foot.append(said, save);

  card.append(top, meta, every, modelsWrap, foot);
  card.parts = { roleBtns, chips: chipBtns, modelsWrap, every, count, said, save, foot, meta };
  acPaint(card);
  return card;
}

/** Save one card: POST /auth/access {email, role, models}. True when the gate took it; its words when it did not. */
async function accessSave(card) {
  const p = card.parts;
  const now = acState(card);
  p.save.disabled = true;
  p.said.dataset.kind = '';
  p.said.textContent = 'saving…';
  const a = await acAsk('/auth/access', { method: 'POST', headers: { 'Content-Type': 'application/json' },
                                         body: JSON.stringify({ email: card.dataset.email, ...now }) });
  if (a.status === 200) {
    card.saved = now;
    const who = a.body && a.body.person;
    if (who && who.updated_by) {
      // the card's line says who changed it last, from the gate's own answer
      const bits = [];
      if (who.signed_in_at) bits.push(`signed in ${acAgo(who.signed_in_at)}`);
      bits.push(`changed by ${who.updated_by}${who.updated_at ? ` ${acAgo(who.updated_at)}` : ''}`);
      p.meta.textContent = bits.join(' · ');
    }
    acPaint(card);
    p.said.dataset.kind = 'ok';
    p.said.textContent = 'saved';
    acPaint(card);
    if (typeof p.said.focus === 'function') p.said.focus();
    return true;
  }
  const why = a.body && (a.body.detail || a.body.error);
  p.said.dataset.kind = 'bad';
  p.said.textContent = why ? String(why) : (a.status ? `HTTP ${a.status}` : `no answer (${a.error})`);
  acPaint(card);
  if (typeof p.said.focus === 'function') p.said.focus();
  return false;
}

function acBuild() {
  const dlg = document.getElementById('access');
  const box = acEl('div', 'cn-box ac-box');
  const head = acEl('div', 'cn-head');
  const titles = acEl('div', 'ac-titles');
  const title = acEl('h2', 'cn-title', 'Access');
  title.id = 'access-title';
  const tally = acEl('span', 'ac-tally');
  titles.append(title, tally);
  const x = acButton('cn-x', '×', 'Close');
  x.addEventListener('click', () => accessClose());
  head.append(titles, x);
  const lead = acEl('p', 'ac-lead', 'Admins have everything. A member uses only the models lit here — no Start / Stop, '
    + 'no power, no benchmarks. Someone appears here after their first sign-in.');
  const said = acEl('p', 'ac-note');
  said.setAttribute('role', 'status');
  const list = acEl('ul', 'ac-list');
  box.append(head, lead, said, list);
  dlg.append(box);
  dlg.addEventListener('click', (e) => { if (e.target === dlg) accessClose(); });
  dlg.addEventListener('cancel', (e) => { e.preventDefault(); accessClose(); });   // Escape
  ac.parts = { dlg, list, said, tally, x, lead };
}

/** Close — but not over unsaved edits without saying so: a close with changed cards names them and stays open; a
 *  second close over the SAME changed cards discards them (reopening reads the gate again). The warning remembers which
 *  cards it named (`ac.warned`), so a card changed after it — or after the named ones were saved — is named again. */
function accessClose() {
  const p = ac.parts;
  const dirty = [...(p.list.children || [])].filter((c) => c.dataset && c.dataset.dirty === 'true')
    .map((c) => c.dataset.email);
  const key = dirty.join('\n');
  if (dirty.length && ac.warned !== key) {
    ac.warned = key;
    p.said.textContent = `Unsaved changes for ${dirty.join(', ')}: Save them, or close again to discard.`;
    return false;
  }
  ac.warned = null;
  p.dlg.close();
  return true;
}

/** Open the dialog: everyone (GET /auth/access) and the models on offer (/lb/status), then one card each. */
async function accessOpen() {
  if (!ac.parts) acBuild();
  const p = ac.parts;
  if (!p.dlg.open) p.dlg.showModal();
  ac.warned = null;
  p.lead.hidden = false;
  p.said.textContent = 'reading /auth/access…';
  p.tally.textContent = '';
  p.list.replaceChildren();
  const [people, status] = await Promise.all([acAsk('/auth/access'), acAsk('/lb/status')]);
  if (people.status !== 200) {
    const why = people.body && (people.body.detail || people.body.error);
    p.said.textContent = `/auth/access: ${why || (people.status ? `HTTP ${people.status}` : 'no answer')}`;
    return;
  }
  if (!people.body || !Array.isArray(people.body.people)) {
    p.said.textContent = '/auth/access: the answer carries no list of people';
    return;
  }
  ac.models = status.status === 200 ? accessModels(status.body) : [];
  const rows = accessRows(people.body.people, ac.models);
  const admins = rows.filter((r) => r.role === 'admin').length;
  p.tally.textContent = `${rows.length} ${rows.length === 1 ? 'person' : 'people'} · ${admins} admin${admins === 1 ? '' : 's'}`;
  p.said.textContent = [rows.length ? '' : 'Nobody has signed in yet.',
                        status.status === 200 ? '' : '/lb/status did not answer: only the models already granted are shown']
    .filter(Boolean).join(' ');
  for (const row of rows) p.list.append(acCard(row));
}

/** A member's own access, read-only: their role and the models they may use (from /auth/session), what that means. */
function accessMine(models) {
  if (!ac.parts) acBuild();
  const p = ac.parts;
  p.tally.textContent = 'member';
  p.lead.hidden = true;                     // the admin's explanation of the dialog is not a member's
  p.list.replaceChildren();
  p.said.textContent = '';
  const card = acEl('li', 'ac-person');
  const head = acEl('div', 'ac-models-head');
  head.append(acEl('span', 'ac-label', 'your models'), acEl('span', 'ac-count', String(models.length)));
  const chips = acEl('div', 'ac-chips');
  for (const m of models) chips.append(acEl('span', 'ac-chip ac-chip-ro', m));
  if (!models.length) chips.append(acEl('span', 'ac-none', 'none yet: an admin grants models in their Access dialog'));
  card.append(head, chips, acEl('p', 'ac-every', 'A member uses these models only: Start / Stop, power and benchmarks '
    + 'are an admin\'s.'));
  p.list.append(card);
  if (!p.dlg.open) p.dlg.showModal();
}

/** The top bar: an admin's Access button, a member's line, nothing for an open gate or the LB's own port. */
async function accessStart() {
  const a = await acAsk('/auth/session');
  if (a.status !== 200 || !a.body || a.body.auth === 'false') return;
  const bar = document.querySelector('.top-right');
  if (!bar) return;
  ac.me = typeof a.body.email === 'string' ? a.body.email : null;
  const before = document.getElementById('connection-status');
  if (a.body.role === 'admin') {
    // the word, and a mark a phone's header shows alone (gateway.css: the header has no room for one more word)
    const b = acButton('ly-open ac-open', undefined, 'Access');
    b.append(acEl('span', 'ac-glyph'), acEl('span', 'ac-word', 'Access'));
    b.setAttribute('aria-haspopup', 'dialog');
    b.setAttribute('aria-controls', 'access');
    b.addEventListener('click', accessOpen);
    bar.insertBefore(b, before);
  } else if (a.body.role === 'member') {
    document.documentElement.dataset.role = 'member';
    const models = a.body.models || [];
    const full = models.length ? `member — you may use ${models.join(', ')}` : 'member — no models granted yet: ask an admin';
    // a BUTTON: a phone shows the mark and the count only, so the whole of it is a tap away ("your access") and in
    // its accessible name — never on hover alone
    const line = acButton('ly-open ac-member', undefined, full);
    line.title = full;
    line.append(acEl('span', 'ac-glyph'), acEl('b', null, 'member'),
                acEl('span', 'ac-word', models.length ? ` · ${models.join(', ')}` : ' · no models yet'),
                acEl('span', 'ac-n', String(models.length)));
    line.setAttribute('aria-haspopup', 'dialog');
    line.setAttribute('aria-controls', 'access');
    line.addEventListener('click', () => accessMine(models));
    bar.insertBefore(line, before);
  }
}

if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', accessStart);
else accessStart();
