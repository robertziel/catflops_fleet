/* WHY A TURN WAITS, and the queue tile's breakdown — gateway.js `requestRow`, `recentRow`, `requestCounts` and
 * `queueTile`, under node.
 *
 * The operator (2026-10-12, a one-GPU node at 16 seats, 2 rows STREAMING over 11 QUEUED): "why only one is
 * streaming, missing context space? Add such info column, ensure all details work in that table", and of the
 * queue tile, "17 /16 · fits 1": "tell how many are streaming, filling, queued plus concurrencies set on the server".
 *
 * Pinned: a WAIT column beside the state — on a queued row the server's verdict (request_queue.admission: "kv full
 * 99%", "seats 16/16", "deferred 4"), "—" when the figures prove none, the rule and the figures in its title; on a
 * running row nothing. The time-to-first-byte column is called what it measures (TTFT: queue + prefill, send →
 * message_start or the first byte) and prints no prompt ÷ that as a "prefill" rate. A prompt figure not reported
 * yet says why. The head carries the engine's own waiting reasons. The queue tile's parts add to its headline:
 * streaming + filling = the engine's running, queued = its waiting, over the server's seats.
 *
 *   node tests/queue_wait.node.js      (tests/test_queue_wait_page.py runs this when node exists)
 */
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const src = fs.readFileSync(path.join(__dirname, '..', 'vllm_lb', 'static', 'gateway.js'), 'utf8');
const html = fs.readFileSync(path.join(__dirname, '..', 'vllm_lb', 'static', 'index.html'), 'utf8');
const el = () => ({ addEventListener() {}, removeAttribute() {}, setAttribute() {}, append() {},
  classList: { toggle() {}, add() {}, remove() {} }, style: {}, hidden: false, innerHTML: '',
  textContent: '', value: '', dataset: {}, children: [], querySelectorAll: () => [], querySelector: () => null,
  focus() {}, observe() {}, closest: () => null, getBoundingClientRect: () => ({}) });
const document = { getElementById: el, createElementNS: el, createElement: el, addEventListener() {},
  querySelectorAll: () => [], querySelector: () => null, body: { style: {}, classList: { toggle() {} } },
  documentElement: { style: {}, classList: { toggle() {} } } };
const window = { addEventListener() {}, innerWidth: 1280, innerHeight: 800, matchMedia: () => ({
  matches: false, addEventListener() {} }), localStorage: { getItem: () => null, setItem() {} } };
const api = new Function('document', 'window', 'fetch', 'location', 'setInterval', 'setTimeout',
  'getComputedStyle', 'ResizeObserver', 'localStorage', 'navigator',
  `${src}\n;return { requestRow, recentRow, requestCounts, queueTile, RECENT_HEAD };`)(
  document, window, () => new Promise(() => {}), { search: '', href: '' }, () => 0, () => 0,
  () => ({}), undefined, window.localStorage, { userAgent: 'node' });

const strip = (h) => h.replace(/<[^>]*>/g, ' ').replace(/&middot;/g, '·').replace(/&nbsp;/g, ' ').replace(/&amp;/g, '&')
  .replace(/\s+/g, ' ').trim();
const cell = (row, cls) => {
  const m = row.match(new RegExp(`<span class="${cls}"( title="([^"]*)")?>([\\s\\S]*?)</span>`));
  return m ? { title: m[2] || '', text: strip(m[3]) } : null;
};
let checks = 0;

/** The captured read's verdict (request_queue.admission over vllm-2eb491e-kv-full-queue.metrics). */
const KV = { wait: 'kv', why: '2 of 164 KV blocks free, a request takes ≥ 7: no request fits · 3 of 16 seats; engine: 10 capacity · 4 deferred',
  running: 3, waiting: 14, waiting_capacity: 10, waiting_deferred: 4, seats: 16, kv_pct: 98.78, kv_usable_blocks: 164,
  kv_free_blocks: 2, kv_free_tokens: 3136, kv_fixed_blocks: 6, kv_least_blocks: 7, mean_context: 75264,
  rows: { streaming: 2, prefill: 1, queued: 11, sent: 0 } };
const BASE = { id: '9', started_at: 1, age_s: 72.4, phase_s: 72.4, client: 'OpenAI', client_detail: null, session: null,
  model: 'Qwen/Qwen3.8-27B', messages: 16, tools: 22, box: 'node-a', backend: 'http://engine:8000',
  path: '/v1/chat/completions', prompt: 'Write the article', prompt_chars: 971, prompt_tokens: null,
  cached_tokens: null, prompt_from: null, cached_share: null, prefill_s: null, prefill_tok_s: null, decode_s: null,
  delta_rate: null, deltas: 0, thinking: 0, text: 0, tool: 0, tool_uses: 0, output_tokens: null, stop_reason: null,
  bytes: 0, last_byte_s: null, keepalives: 0 };
const QUEUED = { ...BASE, state: 'queued', inferred: { no_bytes: 12, waiting: 11 } };
const STREAMING = { ...BASE, state: 'streaming', inferred: null, phase_s: 4, prefill_s: 166.2, decode_s: 4,
  deltas: 180, thinking: 150, text: 30, delta_rate: 45, bytes: 9000, last_byte_s: 0.1 };

// ── 1. THE WAIT COLUMN: the server's verdict on each queued row, its rule on hover ──────────────────────
{
  const q = api.requestRow(QUEUED, KV);
  const w = cell(q, 'rq-num rq-wait');
  assert.ok(w, `a queued row has a wait cell: ${q}`);
  assert.strictEqual(w.text, 'kv full 99%');
  assert.ok(w.title.includes('2 of 164') && w.title.includes('≥ 7'), `the rule and its figures on hover: ${w.title}`);
  const seats = api.requestRow(QUEUED, { ...KV, wait: 'seats', running: 16, why: '16 of 16 seats: a turn starts when one ends' });
  assert.strictEqual(cell(seats, 'rq-num rq-wait').text, 'seats 16/16');
  const deferred = api.requestRow(QUEUED, { ...KV, wait: 'deferred', waiting: 4, waiting_deferred: 4 });
  assert.strictEqual(cell(deferred, 'rq-num rq-wait').text, 'deferred 4');
  const unknown = api.requestRow(QUEUED, { ...KV, wait: null, why: '2 of 16 seats · 52 of 164 KV blocks free: neither full' });
  const u = cell(unknown, 'rq-num rq-wait');
  assert.ok(u.text === '—' && u.title.includes('52 of 164'), `not proven: "—" with the figures: ${JSON.stringify(u)}`);
  const none = cell(api.requestRow(QUEUED, undefined), 'rq-num rq-wait');
  assert.ok(none.text === '—' && /not read/.test(none.title), `no facts for its server: says so: ${JSON.stringify(none)}`);
  const sent = cell(api.requestRow({ ...QUEUED, state: 'sent', inferred: null }, KV), 'rq-num rq-wait');
  assert.strictEqual(sent.text, '—', 'no gauge: queued or prefilling unknown, so no reason claimed');
  for (const st of ['streaming', 'prefill']) {
    const r = cell(api.requestRow({ ...STREAMING, state: st }, KV), 'rq-num rq-wait');
    assert.ok(r && r.text === '', `${st}: admitted — the wait cell is empty, never "—": ${JSON.stringify(r)}`);
  }
  assert.ok(!api.recentRow({ ...BASE, outcome: 'done', duration_s: 9 }).includes('rq-wait'), 'a finished row has none');
  const liveHead = html.match(/<div id="requests-live"[\s\S]*?<div class="rq-row rq-head"[^>]*>([\s\S]*?)<\/div>/)[1];
  const heads = liveHead.match(/<span[^>]*>[^<]*<\/span>/g).map(strip);
  assert.deepStrictEqual(heads.slice(0, 2), ['state', 'wait'], `the live head: ${heads}`);
  const liveCells = (api.requestRow(QUEUED, KV).match(/<span class="rq-(?!phase|sess)[^"]*"/g) || []).length;
  assert.strictEqual(liveCells, heads.length, `one cell per head: ${liveCells} vs ${heads}`);
  const recentHeads = api.RECENT_HEAD.match(/<span[^>]*>[^<]*<\/span>/g).map(strip);
  assert.ok(!recentHeads.includes('wait'), 'the finished table has no wait column');
  checks += 13;
}

// ── 2. TTFT, NOT "PREFILL": send → first byte, the queue included; no prompt ÷ it ───────────────────────
{
  const liveHead = html.match(/<div id="requests-live"[\s\S]*?<div class="rq-row rq-head"[^>]*>([\s\S]*?)<\/div>/)[1];
  assert.ok(/>ttft</.test(liveHead) && !/>prefill</.test(liveHead), `the live head says ttft: ${liveHead}`);
  assert.ok(/>ttft</.test(api.RECENT_HEAD) && !/>prefill</.test(api.RECENT_HEAD), 'the finished head too');
  const r = api.requestRow({ ...STREAMING, prompt_tokens: 48213, cached_share: 0.9, prefill_tok_s: 290 }, KV);
  const t = cell(r, 'rq-num rq-ttft');
  assert.ok(t && t.text === '2m 46s', `TTFT as measured: ${JSON.stringify(t)}`);
  assert.ok(/queue/.test(t.title), `its title says the queue is in it: ${t.title}`);
  assert.ok(!r.includes('290/s'), 'prompt ÷ (queue + prefill) is not a prefill speed: not printed');
  const done = api.recentRow({ ...STREAMING, outcome: 'done', duration_s: 200, prefill_s: 2.0, prefill_tok_s: 450,
    output_tokens: 77, decode_tok_s: 31.4, stop_reason: 'stop' });
  assert.ok(!done.includes('450/s') && done.includes('2.0 s'), 'the finished row: TTFT, no rate');
  assert.ok(done.includes('>stop<'), "an OpenAI turn's finish_reason is its stop");
  checks += 7;
}

// ── 3. A PROMPT FIGURE NOT REPORTED YET SAYS WHY ────────────────────────────────────────────────────────
{
  const p = cell(api.requestRow(STREAMING, KV), 'rq-num rq-prompt');
  assert.ok(p.text === '—' && /final usage/.test(p.title), `an OpenAI stream's prompt: ${JSON.stringify(p)}`);
  const a = cell(api.requestRow({ ...STREAMING, client: 'claude', path: '/v1/messages' }, KV), 'rq-num rq-prompt');
  assert.ok(a.text === '—' && /message_start/.test(a.title), `an Anthropic stream with none: ${JSON.stringify(a)}`);
  const q = cell(api.requestRow(QUEUED, KV), 'rq-num rq-prompt');
  assert.ok(q.text === '—' && /no byte/.test(q.title), `a queued turn: nothing back yet: ${JSON.stringify(q)}`);
  checks += 3;
}

// ── 4. THE HEAD: the engine's own reasons beside its running / waiting ──────────────────────────────────
{
  const h = strip(api.requestCounts({ inflight: [], inflight_total: 14,
    counts: { streaming: 2, prefill: 1, queued: 11, sent: 0, held: 0, recent: 0 },
    engine: { running: 3, waiting: 14, backends: 1, waiting_capacity: 10, waiting_deferred: 4 }, recent: [] }));
  assert.ok(/10 capacity/.test(h) && /4 deferred/.test(h), `the engine's reasons: ${h}`);
  const old = strip(api.requestCounts({ inflight: [], inflight_total: 0, counts: {}, recent: [],
    engine: { running: 3, waiting: 14, backends: 1, waiting_capacity: null, waiting_deferred: null } }));
  assert.ok(!/capacity|deferred|null/.test(old), `not published: not drawn: ${old}`);
  checks += 2;
}

// ── 5. THE QUEUE TILE: streaming · filling · queued, adding to its headline, over the server's seats ────
{
  const headline = (t) => strip((t.match(/<b class="t-val"[^>]*>([\s\S]*?)<\/b>/) || [, ''])[1]);
  const subline = (t) => strip((t.match(/class="t-sub">([\s\S]*)<\/span>/) || [, ''])[1]);
  const b = { kind: 'cluster', url: 'http://engine:8000', running: 3, waiting: 14, inflight: 17, depth_known: true };
  const t = api.queueTile(b, { depth: [] }, { kind: 'queue', concurrency: 16, fits: 1, engine: KV });
  assert.strictEqual(headline(t), '17 /16 seats · fits 1');
  const sub = subline(t);
  assert.strictEqual(sub, 'engine: 2 stream · 1 fill · 14 queued · kv full');
  const parts = (sub.match(/\d+/g) || []).map(Number);
  assert.strictEqual(parts.reduce((x, y) => x + y, 0), 17, 'the parts add to the headline');
  // The LB's streams outnumber the engine's running (a probe older than the streams): no split is guessed.
  const stale = api.queueTile({ ...b, running: 1 }, { depth: [] }, { kind: 'queue', concurrency: 16, fits: 1, engine: KV });
  assert.strictEqual(subline(stale), 'engine: 1 running · 14 queued · kv full');
  const noWait = subline(api.queueTile(b, { depth: [] }, { kind: 'queue', concurrency: 16, engine: { ...KV, wait: null } }));
  assert.strictEqual(noWait, 'engine: 2 stream · 1 fill · 14 queued');
  const bare = subline(api.queueTile(b, { depth: [] }, {}));
  assert.strictEqual(bare, '', 'no rows for this server: nothing under the graph, as before');
  checks += 6;
}
console.log(`ok · ${checks} checks`);
