/* THE REQUESTS CARD, checked where the page builds it: gateway.js `requestRow`, `recentRow`, `requestCounts` and
 * `renderRequests`, under node with a stub DOM.
 *
 * /lb/status `queue` (request_queue.py): one row per turn the ENGINE is working on — streaming, prefill, queued,
 * or sent with the split unknown — and the finished folded away; held turns are a count. Pinned: each state's badge
 * word and tone (the page's own badge); the inferred prefill/queued split says so on hover; the row's figures —
 * prompt tokens and cached share, TTFT seconds (queue + prefill, no rate), decode ≈ tok/s (derived: the engine's window apportioned by events) and Δ events (not tokens), the decode sum
 * beside the engine's figure, the
 * thinking share and tool_use blocks, the last byte and keepalives; the prompt ONE line of at most 100 chars, the
 * capped whole only in its title, always ESCAPED; rows kept as the SAME elements across polls; the empty state a
 * figure, not a sentence.
 *
 *   node tests/request_queue.node.js      (tests/test_request_queue_page.py runs this when node exists)
 */
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const src = fs.readFileSync(path.join(__dirname, '..', 'vllm_lb', 'static', 'gateway.js'), 'utf8');

// ── a DOM just big enough for keyed rows ─────────────────────────────────────────────────────────────
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
  `${src}\n;return { requestRow, recentRow, requestCounts, renderRequests, REQUEST_STATES, PROMPT_LINE };`)(
  document, window, () => new Promise(() => {}), { search: '', href: '' }, () => 0, () => 0,
  () => ({}), undefined, window.localStorage, { userAgent: 'node' });

/** The page's OWN words: every cell apart, the prompt cell (the operator's traffic, not the page's prose) out. */
const text = (html) => html.replace(/<span class="rq-text"[^>]*>[^<]*<\/span>/g, '|')
  .replace(/<[^>]*>/g, '|').replace(/&middot;|·|→|\//g, '|');
/** No sentence: no run of four or more words between figures and separators. */
const noSentence = (html, what) => text(html).split('|').forEach((seg) => {
  const words = seg.trim().split(/\s+/).filter((w) => /[a-z]{2,}/i.test(w));
  assert.ok(words.length < 4, `${what} reads as a sentence: "${seg.trim()}"`);
});
const unesc = (h) => h.replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&quot;/g, '"').replace(/&#39;/g, "'")
  .replace(/&amp;/g, '&');

const PROMPT = 'summarise <script>alert(1)</script> & "quote" ' + 'y'.repeat(400);
const ROW = { id: '1a', started_at: 1791016000, age_s: 135, state: 'streaming', phase_s: 41, inferred: null,
  client: 'claude', client_detail: 'sdk-cli', session: 'c35e5e00', model: 'qwen3.8:latest', messages: 212, tools: 23,
  box: 'box-a + box-b', backend: 'http://192.0.2.2:8000', path: '/v1/messages', prompt: PROMPT,
  prompt_chars: PROMPT.length, prompt_tokens: 48213, cached_tokens: 44000, prompt_from: 'message_start',
  cached_share: 0.913, prefill_s: 31.4, prefill_tok_s: 1535.4, decode_s: 41.0, delta_rate: 7.8, deltas: 318,
  thinking: 200, text: 100, tool: 18, tool_uses: 2, output_tokens: null, stop_reason: null,
  bytes: 51234, last_byte_s: 2.4, keepalives: 3, tok_s: 61.2, window_events: 40, tok_s_why: null };

// ── the states: a badge word and a tone each, the page's own badge ────────────────────────────────────
const S = api.REQUEST_STATES;
assert.deepStrictEqual(Object.keys(S).sort(),
  ['client_gone', 'done', 'failed', 'killed', 'prefill', 'queued', 'sent', 'streaming']);   // killed: request_kill.node.js
assert.ok(!('held' in S), 'a held turn is a count, never a row');
assert.deepStrictEqual([S.streaming.label, S.prefill.label, S.queued.label, S.sent.label],
  ['STREAMING', 'PREFILL', 'QUEUED', 'NO BYTES']);
assert.strictEqual(S.streaming.tone, 'ok');
assert.strictEqual(S.failed.tone, 'bad');
for (const [state, { label, tone }] of Object.entries(S)) {
  const html = api.requestRow({ ...ROW, state, outcome: state });
  assert.ok(html.includes(`<span class="badge ${tone}"`) && html.includes(`>${label}</span>`), `${state} badge: ${html}`);
}

// ── a streaming row: one line of figures ──────────────────────────────────────────────────────────────
{
  const html = api.requestRow(ROW);
  assert.ok(html.includes('41 s'), 'how long in this state');
  assert.ok(html.includes('claude -p') && html.includes('c35e5e00'), 'client · session');
  assert.ok(html.includes('qwen3.8') && !html.includes('>qwen3.8:latest<'), 'the model, short');
  assert.ok(html.includes('212') && html.includes('23'), 'messages and tools');
  assert.ok(html.includes('48.2k') && html.includes('91%'), 'prompt tokens and the cached share');
  // TTFT (send → message_start: queue + prefill); prompt ÷ it is no prefill speed, so none is printed
  // (tests/queue_wait.node.js, 2026-10-12).
  assert.ok(html.includes('31.4 s') && !html.includes('1,535/s'), 'TTFT seconds, no rate');
  // DECODE is the engine's windowed tok/s apportioned by this turn's events in that window — derived, marked ≈ —
  // and the events since message_start beside it, called events (operator 2026-10-16: the rows must add up).
  assert.ok(html.includes('≈61.2 tok/s') && html.includes('Δ318 ev'), `derived tok/s and events: ${html}`);
  assert.ok(!html.includes('7.8/s'), 'no lifetime events-per-second beside a tok/s');
  const dec = html.match(/<span class="rq-num rq-decode" title="([^"]*)">/);
  assert.ok(dec && /derived/.test(dec[1]) && /40 events/.test(dec[1]), `its method on hover: ${dec && dec[1]}`);
  const none = api.requestRow({ ...ROW, tok_s: null, tok_s_why: 'first sample' });
  assert.ok(!/tok\/s/.test(none.replace(/title="[^"]*"/g, '')) && none.includes('first sample'),
    `no total, no number: the why: ${none}`);
  assert.ok(html.includes('think 63%') && html.includes('2 tool'), 'the thinking share and tool_use blocks');
  assert.ok(html.includes('2 s') && html.includes('3 ka'), 'last byte and keepalives');
  assert.ok(html.includes('box-a + box-b'), 'the box');
  assert.ok(!html.includes('<script>') && html.includes('&lt;script&gt;'), 'the prompt is escaped');
  const line = html.match(/<span class="rq-text" title="([^"]*)">([^<]*)<\/span>/);
  assert.ok(line, `the prompt's one line: ${html}`);
  const shown = unesc(line[2]);
  assert.ok(api.PROMPT_LINE <= 100 && shown.length <= api.PROMPT_LINE + 1 && shown.endsWith('…'),
    `one narrow line: ${shown.length}`);
  assert.ok(unesc(line[1]) === PROMPT, 'the capped whole is the title, and only there');
  assert.ok(!/<details|rq-more/.test(html), 'no disclosure on a row');
  noSentence(html, 'a row');
}
{
  const q = api.requestRow({ ...ROW, state: 'queued', phase_s: 12, inferred: { no_bytes: 9, waiting: 7 },
    prompt_tokens: null, cached_share: null, prefill_s: null, prefill_tok_s: null, decode_s: null, delta_rate: null,
    deltas: 0, thinking: 0, text: 0, tool: 0, tool_uses: 0, bytes: 0, last_byte_s: null, keepalives: 0 });
  assert.ok(q.includes('>QUEUED<') && q.includes('12 s') && q.includes('—'), q);
  const badge = q.match(/<span class="badge park" title="([^"]*)">/);
  assert.ok(badge && /inferred/.test(badge[1]) && /9/.test(badge[1]) && /7/.test(badge[1]),
    'the inferred split says so, with its evidence, on hover');
  const bare = api.requestRow({ ...ROW, client: 'no ua', session: null, model: null, messages: null, tools: null,
    prompt: null, box: null });
  assert.ok(bare.includes('no ua') && !bare.includes('null') && !bare.includes('undefined'), bare);
  const counted = api.requestRow({ ...ROW, output_tokens: 1200 });
  assert.ok(counted.includes('Δ318'), 'live: events until the stream is over');
}

// ── the finished ───────────────────────────────────────────────────────────────────────────────────────
{
  const r = { id: '19', started_at: 1791015900, ended_at: 1791016050, duration_s: 150, outcome: 'failed',
    why: 'no message_stop', status: 200, client: 'browser', client_detail: null, session: null, model: 'qwen3.8:latest',
    messages: 3, tools: 0, box: 'box-a + box-b', prompt: '<img src=x onerror=alert(1)>' + 'z'.repeat(140),
    prompt_tokens: 900, cached_share: 0.5, prefill_s: 2.0, prefill_tok_s: 450, output_tokens: 77, decode_tok_s: 31.4,
    deltas: 30, thinking: 0, text: 30, tool: 0, tool_uses: 0, stop_reason: null, bytes: 9000, keepalives: 0 };
  const html = api.recentRow(r);
  assert.ok(html.includes('>FAILED<') && html.includes('no message_stop'), 'the outcome and its why');
  assert.ok(html.includes('2m 30s') && html.includes('77') && html.includes('31.4 tok/s'), 'duration, output, tok/s');
  assert.ok(html.includes('2.0 s') && !html.includes('450/s'), 'TTFT, no rate');
  assert.ok(!html.includes('<img') && html.includes('&lt;img'), 'escaped');
  const ok = api.recentRow({ ...r, outcome: 'done', why: null, stop_reason: 'end_turn' });
  assert.ok(ok.includes('>DONE<') && ok.includes('end_turn'), 'the stop reason');
  noSentence(html, 'a finished row');
}

// ── the counts: held and recent are counts, beside the engine's own gauges ─────────────────────────────
{
  const q = { inflight: [ROW], inflight_total: 3, counts: { streaming: 1, prefill: 1, queued: 1, sent: 0, held: 4,
    recent: 30 }, engine: { running: 30, waiting: 6 }, recent: [] };
  const html = api.requestCounts(q);
  for (const n of ['>1<', '>4<', '>30<', '>6<']) assert.ok(html.includes(n), `${n} in ${html}`);
  // THE SUM BESIDE THE TOTAL: Σ rows + ended (+ unlisted) = the engine's GENERATED, the same window.
  const dq = { ...q, decode: { engine_tok_s: 180.0, rows_tok_s: 165.3, ended_tok_s: 14.7, unlisted_tok_s: 0,
    window_s: 5.0, unmeasured: 0, why: null, derived: 'derived: …' } };
  const sum = api.requestCounts(dq);
  assert.ok(sum.includes('165') && sum.includes('14.7') && sum.includes('180'), `rows, ended, engine: ${sum}`);
  assert.ok(/rows/.test(sum) && /engine/.test(sum), sum);
  noSentence(sum, 'the decode sum');
  const unm = api.requestCounts({ ...q, decode: { engine_tok_s: null, rows_tok_s: null, ended_tok_s: null,
    unlisted_tok_s: null, window_s: null, unmeasured: 1, why: 'first sample', derived: 'derived: …' } });
  assert.ok(unm.includes('first sample') && !/≈|Σ\s*\d/.test(unm.replace(/title="[^"]*"/g, '')), `no total: why: ${unm}`);
  for (const w of ['streaming', 'prefill', 'queued', 'held', 'running', 'waiting']) assert.ok(html.includes(w), w);
  assert.ok(!/no bytes/.test(html), 'a state with none is not drawn');
  noSentence(html, 'the counts');
  const empty = api.requestCounts({ inflight: [], inflight_total: 0,
    counts: { streaming: 0, prefill: 0, queued: 0, sent: 0, held: 0, recent: 0 }, engine: null, recent: [] });
  assert.ok(/<b[^>]*>0<\/b>/.test(empty), `the empty state is a figure: ${empty}`);
  assert.ok(!/no requests/i.test(empty));
  noSentence(empty, 'the empty state');
}

// ── rendering: keyed rows in the server's order, updated in place ─────────────────────────────────────
{
  const r = (id, extra) => ({ ...ROW, id, ...extra });
  const counts = { streaming: 3, prefill: 0, queued: 0, sent: 0, held: 0, recent: 0 };
  api.renderRequests({ queue: { inflight: [r('1'), r('2'), r('3')], inflight_total: 3, counts, engine: null,
    recent: [] } });
  const live = byId('requests-live');
  assert.deepStrictEqual(live.children.map((c) => c.dataset.id), ['1', '2', '3']);
  const kept = live.children[1];
  api.renderRequests({ queue: { inflight: [r('2'), r('4', { state: 'prefill' })], inflight_total: 2,
    counts: { ...counts, streaming: 1, prefill: 1 }, engine: null, recent: [] } });
  assert.deepStrictEqual(live.children.map((c) => c.dataset.id), ['2', '4']);
  assert.strictEqual(live.children[0], kept, 'the same element, updated in place');
  // A live row keeps its state cell (and its kill ×) across polls and paints the badge into it (request_kill.node.js).
  assert.ok(live.children[1].stateBody.innerHTML.includes('PREFILL'), 'a new row drawn');
  assert.ok(byId('lb-requests').hidden === false);
  const done = { ...ROW, outcome: 'done', duration_s: 9, decode_tok_s: 20, output_tokens: 50, stop_reason: 'end_turn' };
  api.renderRequests({ queue: { inflight: [], inflight_total: 0, counts: { ...counts, streaming: 0, recent: 1 },
    engine: null, recent: [done] } });
  assert.strictEqual(live.children.length, 0);
  assert.ok(byId('requests-recent-list').innerHTML.includes('>DONE<'), 'the finished list');
  assert.ok(/<b>1<\/b>/.test(byId('requests-recent-sum').innerHTML), 'opened from its count');
  api.renderRequests({});
  assert.ok(byId('lb-requests').hidden, 'an LB with no queue draws no card');
}
console.log('ok');
