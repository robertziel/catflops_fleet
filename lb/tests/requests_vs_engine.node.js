/* THE GATEWAY'S COUNT BESIDE THE ENGINE'S, checked where the page builds them: gateway.js `engineCount`, `requestCounts`,
 * `engineGap`, `queueTile`, `lingerOf` and `renderRequests`, under node with a stub DOM.
 *
 * Operator, 2026-10-30: "why shows only one request if queue tells 2?". The QUEUE tile is the ENGINE's running +
 * waiting off its last /metrics read; the REQUESTS card lists this gateway's relays now. ~1 s turns make them disagree
 * by one or two, so the page says which is which and by how much:
 *   - the head: "gateway N · engine M running K waiting · …" — the engine figures from the SAME backend fields the
 *     QUEUE tile reads (`engineCount`, one definition);
 *   - a muted line when they differ: "+1 engine only — …" / "+1 gateway only — …", with "engine read 3 s ago" off the
 *     backend's `depth_read_at`; nothing when they agree or the engine was not read;
 *   - the QUEUE tile's subline says whose count it is: "engine: 2 running · 0 queued";
 *   - a turn that ENDED within LINGER_S stays in the live list as "done · 1.2 s" (muted), the same element, then
 *     only the finished list has it.
 *
 *   node tests/requests_vs_engine.node.js      (tests/test_requests_vs_engine_page.py runs this when node exists)
 */
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');

// palette.js first: the colours this script computes are read from the style's tokens (tests/palette_env.js).
const src = require('./palette_env').withPalette(fs.readFileSync(path.join(__dirname, '..', 'vllm_lb', 'static', 'gateway.js'), 'utf8'));

class El {
  constructor(tag) {
    this.tagName = String(tag || 'div').toUpperCase(); this.children = []; this.parentNode = null;
    this.dataset = {}; this.style = {}; this.hidden = false; this.open = false; this.className = '';
    this.innerHTML = ''; this.textContent = '';
    this.classList = { toggle() {}, add() {}, remove() {} };
  }
  appendChild(c) { return this.insertBefore(c, null); }
  insertBefore(c, ref) {
    if (c.parentNode) c.parentNode.children.splice(c.parentNode.children.indexOf(c), 1);
    const i = ref ? this.children.indexOf(ref) : -1;
    if (i < 0) this.children.push(c); else this.children.splice(i, 0, c);
    c.parentNode = this; return c;
  }
  remove() { if (this.parentNode) this.parentNode.children.splice(this.parentNode.children.indexOf(this), 1); this.parentNode = null; }
  addEventListener() {} removeAttribute() {} setAttribute() {} append() {} focus() {} observe() {}
  querySelectorAll() { return []; } querySelector() { return null; } closest() { return null; }
  getBoundingClientRect() { return {}; }
}
const ids = new Map();
const byId = (id) => { if (!ids.has(id)) ids.set(id, new El('div')); return ids.get(id); };
const document = { getElementById: byId, createElementNS: (ns, t) => new El(t), createElement: (t) => new El(t),
  addEventListener() {}, querySelectorAll: () => [], querySelector: () => null, activeElement: null,
  body: { style: {}, classList: { toggle() {} } }, documentElement: { style: {}, classList: { toggle() {} } } };
const window = { addEventListener() {}, innerWidth: 1280, innerHeight: 800, matchMedia: () => ({
  matches: false, addEventListener() {} }), localStorage: { getItem: () => null, setItem() {} } };

const api = new Function('document', 'window', 'fetch', 'location', 'setInterval', 'setTimeout',
  'getComputedStyle', 'ResizeObserver', 'localStorage', 'navigator',
  `${src}\n;return { engineCount, requestCounts, engineGap, queueTile, lingerOf, LINGER_S, renderRequests, scopeQueue };`)(
  document, window, () => new Promise(() => {}), { search: '', href: '' }, () => 0, () => 0,
  () => ({}), undefined, window.localStorage, { userAgent: 'node' });

const strip = (h) => h.replace(/<[^>]*>/g, ' ').replace(/&middot;/g, '·').replace(/&nbsp;/g, ' ').replace(/&amp;/g, '&')
  .replace(/&mdash;/g, '—').replace(/\s+/g, ' ').trim();
let checks = 0;

const NOW = 1791016000;
const URL = 'http://engine:8100';
const HEAD = { kind: 'cluster', cluster: 'pair', url: URL, live: true, depth_known: true, running: 2, waiting: 0,
  inflight: 1, depth_read_at: NOW - 3.2, history: { depth: [] } };

// ── 1. ONE DEFINITION: the head's engine figures are the QUEUE tile's own fields ───────────────────────
{
  const e = api.engineCount([HEAD], NOW);
  assert.ok(e && e.running === 2 && e.waiting === 0 && e.backends === 1, JSON.stringify(e));
  assert.strictEqual(Math.round(e.read_s), 3, 'its age off depth_read_at');
  assert.strictEqual(api.engineCount([{ ...HEAD, depth_known: false }], NOW), null, 'not read: no figure');
  assert.strictEqual(api.engineCount([{ ...HEAD, live: false }], NOW), null, 'not live: no figure');
  assert.strictEqual(api.engineCount([{ ...HEAD, depth_read_at: null }], NOW).read_s, null, 'no read time: no age');
  const two = api.engineCount([HEAD, { ...HEAD, url: 'x', running: 1, waiting: 3, depth_read_at: NOW - 9 }], NOW);
  assert.ok(two.running === 3 && two.waiting === 3 && two.backends === 2 && Math.round(two.read_s) === 9,
    `summed; the oldest read: ${JSON.stringify(two)}`);
  checks += 6;
}

// ── 2. THE HEAD: "gateway N · engine M running K waiting" side by side, then the gateway's states ──────
const Q = (n, extra) => ({ inflight: Array.from({ length: n }, (_, i) => ({ id: `r${i}`, backend: URL, state: 'streaming' })),
  inflight_total: n, counts: { streaming: n, prefill: 0, queued: 0, sent: 0, held: 0, recent: 0 }, recent: [],
  engine: { running: 2, waiting: 0, backends: 1, read_s: 3.2 }, ...extra });
{
  const h = strip(api.requestCounts(Q(1)));
  assert.ok(h.startsWith('gateway 1 · engine 2 running 0 waiting · 1 streaming'), h);
  const none = strip(api.requestCounts(Q(0, { engine: null })));
  assert.ok(none.startsWith('gateway 0 · engine —'), `not read: a dash, never 0: ${none}`);
  checks += 2;
}

// ── 3. THE NOTE: the difference and why, or nothing ─────────────────────────────────────────────────────
{
  assert.strictEqual(strip(api.engineGap(Q(1))),
    '+1 engine only — just finished, not yet relayed, or sent past this gateway · engine read 3 s ago');
  assert.strictEqual(strip(api.engineGap(Q(4))),
    '+2 gateway only — not yet admitted by the engine · engine read 3 s ago');
  assert.strictEqual(api.engineGap(Q(2)), '', 'they agree: nothing');
  assert.strictEqual(api.engineGap(Q(1, { engine: null })), '', 'no engine figure: no difference claimed');
  assert.strictEqual(strip(api.engineGap(Q(1, { engine: { running: 2, waiting: 0, read_s: null } }))),
    '+1 engine only — just finished, not yet relayed, or sent past this gateway', 'no read time: no age');
  checks += 5;
}

// ── 4. THE QUEUE TILE says whose count it is ────────────────────────────────────────────────────────────
{
  const subline = (t) => strip((t.match(/class="t-sub">([\s\S]*)<\/span>/) || [, ''])[1]);
  const eng = { wait: null, rows: { streaming: 3, prefill: 0, queued: 0, sent: 0 } };
  assert.strictEqual(subline(api.queueTile(HEAD, { depth: [] }, { concurrency: 16, engine: eng })),
    'engine: 2 running · 0 queued');
  checks += 1;
}

// ── 5. THE LINGER: a turn ended within LINGER_S stays a muted live row, then moves to the finished ──────
const ENDED = { id: 'z9', backend: URL, started_at: NOW - 3.2, ended_at: NOW - 2, duration_s: 1.23, outcome: 'done',
  client: 'ExampleClient', status: 200, stop_reason: 'end_turn', output_tokens: 12 };
{
  assert.ok(api.LINGER_S >= 5 && api.LINGER_S <= 15, `~10 s: ${api.LINGER_S}`);
  assert.deepStrictEqual(api.lingerOf([ENDED, { ...ENDED, id: 'old', ended_at: NOW - 60 }], NOW, new Set()).map((r) => r.id),
    ['z9'], 'only the just-ended');
  assert.deepStrictEqual(api.lingerOf([ENDED], NOW, new Set(['z9'])), [], 'still live: a live row, not a linger');
  assert.deepStrictEqual(api.lingerOf([ENDED], null, new Set()), [], 'no server clock: no linger guessed');
  const s = (now, inflight, recent) => ({ now, backends: [HEAD], clusters: {},
    queue: { inflight, inflight_total: inflight.length,
      counts: { streaming: inflight.length, prefill: 0, queued: 0, sent: 0, held: 0, recent: recent.length },
      engine: { running: 2, waiting: 0, backends: 1 }, recent,
      backends: { [URL]: { rows: { streaming: inflight.length, prefill: 0, queued: 0, sent: 0 }, running: 2, waiting: 0 } } } });
  const live = byId('requests-live');
  const LIVE = { id: 'z9', backend: URL, state: 'streaming', age_s: 1, phase_s: 1, client: 'ExampleClient' };
  api.renderRequests(s(NOW - 2.5, [LIVE], []));
  const el = live.children.find((c) => c.dataset.id === 'z9');
  assert.ok(el, 'live first');
  api.renderRequests(s(NOW, [], [ENDED]));
  assert.strictEqual(live.children.find((c) => c.dataset.id === 'z9'), el, 'the same element: no jump');
  assert.ok(/rq-linger/.test(el.className), el.className);
  assert.strictEqual(strip(el.stateBody.innerHTML), 'done · 1.2 s');
  assert.ok(el.killBtn.disabled && el.prioBtn.disabled, 'nothing to kill or prioritise');
  assert.ok(!/z9|ExampleClient/.test(byId('requests-recent-list').innerHTML), 'not twice: the finished list later');
  assert.ok(strip(byId('requests-counts').innerHTML).startsWith('gateway 0 · engine 2 running 0 waiting'),
    `a lingering row is no in-flight relay: ${strip(byId('requests-counts').innerHTML)}`);
  assert.ok(strip(byId('requests-gap').innerHTML).startsWith('+2 engine only'), byId('requests-gap').innerHTML);
  api.renderRequests(s(NOW + api.LINGER_S + 1, [], [ENDED]));
  assert.ok(!live.children.some((c) => c.dataset.id === 'z9'), 'past LINGER_S: gone from the live list');
  assert.ok(/ExampleClient/.test(byId('requests-recent-list').innerHTML), 'the finished list has it');
  checks += 12;
}
console.log(`ok · ${checks} checks`);
