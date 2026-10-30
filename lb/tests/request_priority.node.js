/* THE PRIORITY SQUARE, checked where the page builds it: gateway.js `newRequestRow`, `paintRequest`, `togglePriority`
 * and `renderRequests`, under node with the kill test's stub DOM, clock and recording fetch.
 *
 * Pinned (operator, 2026-10-28: "small square priority column, similar to kill task, green when active, user in GUI can
 * activate too or deactivate"): every live row has a square after the kill ×, in the same state cell (no column of its
 * own); green (`on`, aria-pressed) while the row's priority is sooner than none (< 0); a click posts ONCE
 * `POST /lb/requests/<id>/priority` with JSON {"active": <the opposite>}; locked (disabled, its why on hover) where the
 * LB says so (`priority_locked`), green still when a locked row is prioritised.
 *
 *   node tests/request_priority.node.js      (tests/test_request_priority_page.py runs this when node exists)
 */
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const STATIC = path.join(__dirname, '..', 'vllm_lb', 'static');
const src = fs.readFileSync(path.join(STATIC, 'gateway.js'), 'utf8');
const html = fs.readFileSync(path.join(STATIC, 'index.html'), 'utf8');
const css = fs.readFileSync(path.join(STATIC, 'gateway.css'), 'utf8');
let checks = 0;

// ── a DOM that dispatches: listeners, classes, attributes, children ────────────────────────────────────────────
class El {
  constructor(tag) {
    this.tagName = String(tag || 'div').toUpperCase(); this.children = []; this.parentNode = null;
    this.dataset = {}; this.style = {}; this.hidden = false; this.disabled = false; this.className = '';
    this.innerHTML = ''; this.textContent = ''; this.title = ''; this.attrs = {}; this.on = {};
    const self = this;
    this.classList = {
      add(c) { const s = new Set(self.className.split(/\s+/).filter(Boolean)); s.add(c); self.className = [...s].join(' '); },
      remove(c) { self.className = self.className.split(/\s+/).filter((x) => x && x !== c).join(' '); },
      toggle() {}, contains: (c) => self.className.split(/\s+/).includes(c),
    };
  }
  appendChild(c) { return this.insertBefore(c, null); }
  insertBefore(c, ref) {
    if (c.parentNode) c.parentNode.children.splice(c.parentNode.children.indexOf(c), 1);
    const i = ref ? this.children.indexOf(ref) : -1;
    if (i < 0) this.children.push(c); else this.children.splice(i, 0, c);
    c.parentNode = this; return c;
  }
  remove() { if (this.parentNode) this.parentNode.children.splice(this.parentNode.children.indexOf(this), 1); this.parentNode = null; }
  addEventListener(type, fn) { (this.on[type] = this.on[type] || []).push(fn); }
  setAttribute(k, v) { this.attrs[k] = String(v); }
  getAttribute(k) { return this.attrs[k]; }
  removeAttribute(k) { delete this.attrs[k]; }
  append() {} focus() {} blur() {} observe() {}
  querySelectorAll() { return []; } closest() { return null; }
  /** A button's parts (its `.hold-fill`) are stable stubs, so the fill the hold moves can be read; nothing else has any. */
  querySelector(sel) { if (this.tagName !== 'BUTTON') return null; this.parts = this.parts || {}; return this.parts[sel] || (this.parts[sel] = new El('span')); }
  getBoundingClientRect() { return {}; }
}
function fire(el, type, extra = {}) {
  let prevented = false;
  const ev = { type, button: 0, key: '', repeat: false, target: el, preventDefault() { prevented = true; }, ...extra };
  (el.on[type] || []).forEach((fn) => fn(ev));
  return prevented;
}

const ids = new Map();
const byId = (id) => { if (!ids.has(id)) ids.set(id, new El('div')); return ids.get(id); };
const document = { getElementById: byId, createElementNS: (ns, t) => new El(t), createElement: (t) => new El(t),
  addEventListener() {}, querySelectorAll: () => [], querySelector: () => null, activeElement: null,
  body: { style: {}, classList: { toggle() {} } }, documentElement: { style: {}, classList: { toggle() {} } } };
let reduced = false;                     // prefers-reduced-motion, as the test sets it
const window = { addEventListener() {}, innerWidth: 1280, innerHeight: 800, matchMedia: (q) => ({
  matches: reduced && /reduced-motion:\s*reduce/.test(q), addEventListener() {} }), localStorage: { getItem: () => null, setItem() {} } };

// ── the clock, the frames, the network ─────────────────────────────────────────────────────────────────────────
let clock = 1_000_000;
const performance = { now: () => clock };
class FakeDate extends Date { static now() { return clock; } }
let rafs = [];
const requestAnimationFrame = (fn) => { rafs.push(fn); return rafs.length; };
const cancelAnimationFrame = () => { rafs = []; };
/** Advance the clock to `ms` past now, running the animation frames on the way (every 16 ms). */
function hold(ms) {
  const end = clock + ms;
  while (clock < end) {
    clock = Math.min(end, clock + 16);
    const run = rafs; rafs = [];
    run.forEach((fn) => fn(clock));
  }
}
const posts = [];
let answer = null;                       // what the next kill POST resolves to
const pending = [];
const fetch = (url, opts = {}) => {
  if (url === '/lb/status') return new Promise(() => {});     // the re-poll: never answers here
  posts.push({ url, ...opts });
  return new Promise((resolve) => pending.push(() => resolve(answer)));
};
const respond = (status, body) => ({ status, ok: status >= 200 && status < 300, json: async () => body });
const settle = async () => { while (pending.length) pending.shift()(); for (let i = 0; i < 10; i += 1) await null; };

const api = new Function('document', 'window', 'fetch', 'location', 'setInterval', 'setTimeout', 'clearTimeout',
  'getComputedStyle', 'ResizeObserver', 'localStorage', 'navigator', 'requestAnimationFrame', 'cancelAnimationFrame',
  'performance', 'Date',
  `${src}\n;return { renderRequests, requestRow, recentRow, REQUEST_STATES,
     KILL_HOLD_MS: typeof KILL_HOLD_MS === 'undefined' ? null : KILL_HOLD_MS };`)(
  document, window, fetch, { search: '', href: '' }, () => 0, () => 0, () => 0, () => ({}), undefined,
  window.localStorage, { userAgent: 'node' }, requestAnimationFrame, cancelAnimationFrame, performance, FakeDate);

const ROW = { id: '1a', started_at: 1791016000, age_s: 135, state: 'streaming', phase_s: 41, inferred: null,
  client: 'claude', client_detail: 'cli', session: 'c35e5e00', model: 'qwen3.8:latest', messages: 212, tools: 23,
  box: 'box-a + box-b', backend: 'http://head', path: '/v1/messages', prompt: 'say hi', prompt_chars: 6,
  prompt_tokens: 48213, cached_tokens: 44000, prompt_from: 'message_start', cached_share: 0.913, prefill_s: 31.4,
  prefill_tok_s: null, decode_s: 41.0, delta_rate: 7.8, deltas: 318, thinking: 200, text: 100, tool: 18, tool_uses: 2,
  output_tokens: null, stop_reason: null, bytes: 51234, last_byte_s: 2.4, keepalives: 3 };
const counts = { streaming: 2, prefill: 0, queued: 0, sent: 0, held: 0, recent: 0 };
const status = (inflight, recent = []) => ({ queue: { inflight, inflight_total: inflight.length, counts,
  engine: null, recent, backends: {} } });
const live = byId('requests-live');
const rowOf = (id) => live.children.find((c) => c.dataset.id === id);
const stateText = (row) => row.stateBody.innerHTML.replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim();

const bodyOf = (p) => JSON.parse(p.body);

(async () => {
  const QUEUED = { ...ROW, id: '4d', state: 'queued', path: '/v1/chat/completions', priority: null, priority_locked: null };
  const ON = { ...QUEUED, id: '5e', priority: -1 };
  const LOCKED = { ...ROW, id: '6f', state: 'streaming', path: '/v1/chat/completions', priority: -5,
    priority_locked: 'streaming: the engine is answering it; a running request\'s priority is fixed' };
  api.renderRequests(status([QUEUED, ON, LOCKED]));

  // ── 1. a square after the ×, in the state cell; green while on ───────────────────────────────────────────────
  {
    const row = rowOf('4d');
    const sq = row.prioBtn;
    assert.ok(sq && sq.tagName === 'BUTTON', 'a live row has a priority square');
    assert.strictEqual(sq.parentNode, row.stateCell, 'in the state cell: no column of its own');
    assert.strictEqual(row.stateCell.children.indexOf(sq), row.stateCell.children.indexOf(row.killBtn) + 1, 'right after the ×');
    assert.ok(!/\bon\b/.test(sq.className) && sq.getAttribute('aria-pressed') === 'false', `off: ${sq.className}`);
    assert.ok(!sq.disabled, 'a queued row can be prioritised');
    const on = rowOf('5e').prioBtn;
    assert.ok(/\bon\b/.test(on.className) && on.getAttribute('aria-pressed') === 'true', `green when on: ${on.className}`);
    checks += 6;
  }
  // The square is the ×'s size, and green is the page's ok token.
  assert.ok(/\.rq-prio\s*\{[^}]*width:\s*1\.5rem[^}]*height:\s*1\.5rem/.test(css), 'a 24 px square, as the ×');
  assert.ok(/\.rq-prio\.on\s*\{[^}]*var\(--ok\)/.test(css), 'green when active');
  checks += 2;

  // ── 2. a click posts once: on, then off ─────────────────────────────────────────────────────────────────────
  {
    const sq = rowOf('4d').prioBtn;
    fire(sq, 'click');
    assert.strictEqual(posts.length, 1, 'one POST');
    const p = posts[0];
    assert.strictEqual(p.url, '/lb/requests/4d/priority');
    assert.strictEqual(p.method, 'POST');
    assert.strictEqual(p.headers['Content-Type'], 'application/json');
    assert.deepStrictEqual(bodyOf(p), { active: true });
    assert.ok(sq.disabled, 'no second toggle while one is out');
    fire(sq, 'click');
    assert.strictEqual(posts.length, 1, 'still one');
    answer = respond(200, { priority: { id: '4d', active: true, value: -1, applied: 'requeued' } });
    await settle();
    assert.ok(/4d/.test(byId('requests-kill-say').textContent) && /priority on/.test(byId('requests-kill-say').textContent),
      `said: ${byId('requests-kill-say').textContent}`);
    api.renderRequests(status([{ ...QUEUED, priority: -1 }, ON, LOCKED]));
    assert.ok(/\bon\b/.test(sq.className) && !sq.disabled, 'the next poll paints it green');
    fire(sq, 'click');
    assert.deepStrictEqual(bodyOf(posts[1]), { active: false }, 'an on square turns it off');
    answer = respond(409, { error: { type: 'priority_locked', message: 'request 4d: streaming: fixed' } });
    await settle();
    assert.ok(/fixed/.test(byId('requests-kill-say').textContent), 'a refusal is said in the LB\'s words');
    checks += 11;
  }

  // ── 3. locked: disabled, its why on hover, still green when prioritised ────────────────────────────────────────
  {
    const sq = rowOf('6f').prioBtn;
    assert.ok(sq.disabled, 'locked');
    assert.ok(sq.title.includes('streaming'), `its why on hover: ${sq.title}`);
    assert.ok(/\bon\b/.test(sq.className), 'a locked row prioritised by its header is still green');
    const sent = posts.length;
    fire(sq, 'click');
    assert.strictEqual(posts.length, sent, 'a locked square posts nothing');
    checks += 4;
  }
  console.log(`ok — ${checks} checks`);
})().catch((e) => { console.error(e); process.exit(1); });
