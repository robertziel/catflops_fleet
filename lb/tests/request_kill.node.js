/* THE KILL ×, checked where the page builds it: gateway.js `newRequestRow`, `holdButton` (the page's ONE hold),
 * `killRequest` and `renderRequests`, under node with a stub DOM that dispatches events, a hand-driven clock and
 * animation frames, and a recording fetch.
 *
 * Pinned (operator, 2026-10-12: "implement KILL red x button, hold it for 1s to work"): every live row starts with a red
 * × — `hold kill`, aria-label "Kill request <id> (hold 1 s)" — in its state cell, no column of its own; a hold of
 * less than 1 s (pointer or keyboard) posts nothing and a click is not a hold; a hold of 1 s posts ONCE
 * `POST /lb/requests/<id>/kill`, JSON `{}`; the row says "killing…", then the LB's answer — "killed", or the refusal in
 * its words in the row's text cell; the × is the SAME element across a poll's repaint, so a hold survives it; a killed
 * row the poll stops listing stays a moment in its place, then goes; the finished list has a KILLED outcome.
 *
 *   node tests/request_kill.node.js      (tests/test_request_kill_page.py runs this when node exists)
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

// ── 1. every live row starts with the × — the page's hold, labelled, in the state cell ─────────────────────────
assert.strictEqual(api.KILL_HOLD_MS, 1000, 'the kill is a 1 s hold');
api.renderRequests(status([ROW, { ...ROW, id: '2b', state: 'queued' }]));
{
  const row = rowOf('1a');
  assert.ok(row && row.killBtn, 'a live row has a kill button');
  const btn = row.killBtn;
  assert.strictEqual(btn.tagName, 'BUTTON');
  assert.ok(/\bhold\b/.test(btn.className) && /\bkill\b/.test(btn.className), `the page's hold, red: ${btn.className}`);
  assert.strictEqual(btn.getAttribute('aria-label'), 'Kill request 1a (hold 1 s)');
  assert.ok(btn.innerHTML.includes('hold-fill') && btn.innerHTML.includes('×'), 'the hold fill behind a ×');
  assert.strictEqual(btn.parentNode, row.stateCell, 'in the state cell: no column of its own');
  assert.strictEqual(row.stateCell.children[0], btn, 'at the row start');
  assert.ok(stateText(row).includes('STREAMING'), stateText(row));
  checks += 7;
}
// No new column: the live head keeps its columns, and the CSS keeps the × inside the pinned state cell.
{
  const head = html.match(/<div id="requests-live"[\s\S]*?<div class="rq-row rq-head"[^>]*>([\s\S]*?)<\/div>/)[1];
  assert.strictEqual(head.match(/<span/g).length, 12, 'twelve columns, as before');
  assert.ok(/\.hold\.kill\s*\{[^}]*width:\s*1\.5rem/.test(css), 'a 24 px square, not a full-width hold');
  assert.ok(/\.rq-cells[^{]*\{[^}]*display:\s*contents/.test(css), 'the painted cells stay the table\'s own');
  checks += 3;
}

// ── 2. less than 1 s posts nothing; a click is not a hold; the keyboard holds too ───────────────────────────────
{
  const btn = rowOf('1a').killBtn;
  fire(btn, 'pointerdown');
  hold(900);
  assert.ok(btn.holding(), 'held 0.9 s: still holding');
  fire(btn, 'pointerup');
  hold(500);
  assert.strictEqual(posts.length, 0, 'let go at 0.9 s: nothing sent');
  assert.ok(!btn.holding(), 'and the hold is over');
  fire(btn, 'click');
  hold(1500);
  assert.strictEqual(posts.length, 0, 'a click is not a hold');
  fire(btn, 'keydown', { key: ' ' });
  hold(600);
  fire(btn, 'keyup', { key: ' ' });
  hold(800);
  assert.strictEqual(posts.length, 0, 'Space held 0.6 s: nothing sent');
  fire(btn, 'pointerdown');
  hold(700);
  fire(btn, 'pointerleave');
  hold(800);
  assert.strictEqual(posts.length, 0, 'sliding off cancels');
  checks += 6;
}

// ── 3. a hold that SURVIVES a poll's repaint, then 1 s posts once ───────────────────────────────────────────────
(async () => {
  const row = rowOf('1a');
  const btn = row.killBtn;
  fire(btn, 'pointerdown');
  hold(500);
  api.renderRequests(status([{ ...ROW, phase_s: 44, age_s: 138 }, { ...ROW, id: '2b', state: 'queued' }]));
  assert.strictEqual(rowOf('1a'), row, 'the same row element');
  assert.strictEqual(rowOf('1a').killBtn, btn, 'the same button: a hold under way is not torn down by a poll');
  assert.ok(btn.holding(), 'still holding across the repaint');
  hold(520);
  assert.strictEqual(posts.length, 1, 'held 1 s: one POST');
  const p = posts[0];
  assert.strictEqual(p.url, '/lb/requests/1a/kill');
  assert.strictEqual(p.method, 'POST');
  assert.strictEqual(p.headers['Content-Type'], 'application/json');
  assert.strictEqual(p.body, '{}');
  assert.ok(/killing…/.test(stateText(row)), `the row says killing…: ${stateText(row)}`);
  assert.ok(btn.disabled, 'no second kill while one is out');
  fire(btn, 'pointerdown');
  hold(1200);
  assert.strictEqual(posts.length, 1, 'still one POST');
  checks += 11;

  // The LB answers only once the turn has ended as killed.
  answer = respond(200, { killed: { id: '1a', outcome: 'killed', message: 'killed' } });
  await settle();
  assert.ok(/\bkilled\b/.test(stateText(row)) && !/killing/.test(stateText(row)), `then "killed": ${stateText(row)}`);
  assert.ok(byId('requests-kill-say').textContent.includes('1a') && /killed/.test(byId('requests-kill-say').textContent),
    'said to a screen reader');
  checks += 2;

  // ── 4. the next poll no longer lists it: it stays a moment, said "killed", then goes ───────────────────────────
  api.renderRequests(status([{ ...ROW, id: '2b', state: 'queued' }], [{ ...ROW, outcome: 'killed',
    why: 'by someone@example.com · streaming 41 s', status: 200, duration_s: 41 }]));
  assert.ok(rowOf('1a'), 'a killed row stays after the poll drops it');
  assert.ok(/\bgone\b/.test(rowOf('1a').className) && /\bkilled\b/.test(stateText(rowOf('1a'))));
  assert.strictEqual(live.children[0].dataset.id, '1a', 'where it was held, not moved');
  const finished = byId('requests-recent-list').innerHTML;
  assert.ok(finished.includes('>KILLED<') && finished.includes('someone@example.com'), 'the finished list: KILLED + why');
  clock += 7000;
  api.renderRequests(status([{ ...ROW, id: '2b', state: 'queued' }]));
  assert.ok(!rowOf('1a'), 'and goes once its moment is over');
  checks += 5;

  // ── 5. a refusal: the LB's words, in the row ──────────────────────────────────────────────────────────────────
  const q = rowOf('2b');
  fire(q.killBtn, 'pointerdown');
  hold(1010);
  assert.strictEqual(posts.length, 2);
  assert.strictEqual(posts[1].url, '/lb/requests/2b/kill');
  answer = respond(404, { error: { type: 'request_finished', message: 'request 2b already finished: done, HTTP 200' } });
  await settle();
  assert.ok(/not killed/.test(stateText(q)), `refused: ${stateText(q)}`);
  const why = q.cellsBody.innerHTML.match(/<span class="rq-text rq-kill-why" title="([^"]*)">([^<]*)<\/span>/);
  assert.ok(why && why[2].includes('already finished: done') && why[2].includes('404'), `its words: ${q.cellsBody.innerHTML}`);
  assert.ok(!q.killBtn.disabled, 'a refused kill may be held again');
  api.renderRequests(status([{ ...ROW, id: '2b', state: 'queued' }]));
  assert.ok(/not killed/.test(stateText(q)), 'said for a moment across a poll');
  clock += 7000;
  api.renderRequests(status([{ ...ROW, id: '2b', state: 'queued' }]));
  assert.ok(/QUEUED/.test(stateText(q)) && !q.cellsBody.innerHTML.includes('rq-kill-why'),
    `a turn still live shows its own state again: ${stateText(q)}`);
  checks += 7;

  // ── 6. the states: KILLED is a finished outcome ───────────────────────────────────────────────────────────────
  assert.strictEqual(api.REQUEST_STATES.killed.label, 'KILLED');
  assert.strictEqual(api.REQUEST_STATES.killed.tone, 'bad');
  checks += 2;

  // ── 7. THE × AT REST, HOVERED, HELD (operator, 2026-10-12: "Kill button may grey, on hover make red, then ensure
  //       animated on 1s click"): grey at rest in the page's muted token, red on hover / focus / hold, a red fill
  //       that sweeps the hold's 1,000 ms (the hold's own progress) and empties on an early release; a touch is held
  //       the same way; under prefers-reduced-motion a static step instead of the sweep. ──────────────────────────
  /** The declarations of the rule whose WHOLE selector is `sel`. */
  const rule = (sel) => {
    const m = css.match(new RegExp(`(?:^|\\n)\\s*${sel.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\s*\\{([^}]*)\\}`));
    return m ? m[1] : '';
  };
  const rest = rule('.hold.kill');
  assert.ok(/color:\s*var\(--ink-faint\)/.test(rest), `at rest the × is the muted token: ${rest}`);
  assert.ok(/border-color:\s*var\(--line\)/.test(rest) && !/--bad|248, 81, 73/.test(rest), `and nothing red: ${rest}`);
  assert.ok(!/\.hold\.kill \.rq-x\s*\{[^}]*(--tint|--bad)/.test(css), 'the × glyph takes the button colour, not red');
  const lit = css.match(/([^{}]*\.hold\.kill:hover[^{}]*)\{([^}]*)\}/);
  assert.ok(lit && /:focus-visible/.test(lit[1]) && /\.holding/.test(lit[1]), `one rule lights hover, focus and the hold: ${lit && lit[1]}`);
  assert.ok(/color:\s*var\(--bad\)/.test(lit[2]) && /border-color:\s*var\(--bad\)/.test(lit[2]), `red when lit: ${lit[2]}`);
  assert.ok(/\.hold\.kill \.hold-fill\s*\{[^}]*background:\s*var\(--bad\)/.test(css), 'the fill is the red one');
  assert.ok(/@media \(prefers-reduced-motion: reduce\)\s*\{[^}]*\.hold\.kill[^}]*transition:\s*none/.test(css),
    'no colour transition under reduced motion');
  checks += 7;

  api.renderRequests(status([{ ...ROW, id: '3c' }]));
  const k3 = rowOf('3c').killBtn;
  const fill = k3.querySelector('.hold-fill');
  const scale = () => Number(((fill.style.transform || '').match(/scaleX\(([\d.]+)\)/) || [0, NaN])[1]);
  assert.ok(!k3.classList.contains('holding') && !fill.style.transform, 'at rest: no progress, not held');
  const sent = posts.length;
  fire(k3, 'pointerdown', { pointerType: 'touch' });
  assert.ok(k3.classList.contains('holding'), 'a TOUCH holds like a mouse');
  hold(250);
  const quarter = scale();
  hold(250);
  const half = scale();
  assert.ok(quarter > 0.2 && quarter < 0.3 && half > 0.45 && half < 0.55, `the fill sweeps the 1,000 ms: ${quarter} → ${half}`);
  fire(k3, 'pointerup', { pointerType: 'touch' });
  assert.ok(scale() === 0 && !k3.classList.contains('holding'), 'let go early: the fill empties');
  hold(800);
  assert.strictEqual(posts.length, sent, 'and nothing is sent');
  reduced = true;
  fire(k3, 'pointerdown', { pointerType: 'touch' });
  hold(100);
  const step1 = scale();
  hold(500);
  const step2 = scale();
  assert.ok(step1 > 0 && step1 === step2, `reduced motion: one static step, no sweep (${step1}, ${step2})`);
  fire(k3, 'pointerup', { pointerType: 'touch' });
  assert.strictEqual(scale(), 0, 'and it empties on release');
  fire(k3, 'pointerdown', { pointerType: 'touch' });
  hold(1010);
  assert.strictEqual(posts.length, sent + 1, 'a full touch hold under reduced motion still kills, once');
  assert.strictEqual(posts[posts.length - 1].url, '/lb/requests/3c/kill');
  reduced = false;
  checks += 8;
  console.log(`ok — ${checks} checks`);
})().catch((e) => { console.error(e); process.exit(1); });
