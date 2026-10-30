/* THE SESSION HISTORY PAGE (/sessions, sessions.js) — checked where the page builds it, under node.
 *
 * The page reads one payload (/lb/sessions, perflog.sessions) and draws three things: the totals, a
 * strip per model, and the table. What is checked here is what the operator would otherwise have to
 * check by eye on every deploy: that a figure nobody measured is a dash and never a 0, that a session
 * which served nothing is still drawn and still listed, that the per-model strips share ONE token
 * scale (so two models are comparable by bar height), and that every bar carries its own numbers.
 *
 *   node tests/sessions_page.node.js   (tests/test_sessions_page.py runs this when node exists)
 */
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');

// the page loads server-scope.js first (its `serverScope`): so does this harness
const src = fs.readFileSync(path.join(__dirname, '..', 'vllm_lb', 'static', 'server-scope.js'), 'utf8') + '\n'
  + fs.readFileSync(path.join(__dirname, '..', 'vllm_lb', 'static', 'sessions.js'), 'utf8');

/* A DOM small enough to be read, big enough for what this page does: the page only ever sets
   textContent / innerHTML / hidden, appends SVG nodes, and reads back what it built. */
function node(tag) {
  const n = {
    tag, children: [], attrs: {}, dataset: {}, _text: '', _html: '', hidden: false, className: '',
    setAttribute(k, v) { this.attrs[k] = String(v); },
    getAttribute(k) { return this.attrs[k]; },
    append(...kids) { this.children.push(...kids); },
    replaceChildren(...kids) { this.children = [...kids]; },
    addEventListener() {},
    querySelector() { return node('input'); },
    get textContent() { return this._text; },
    set textContent(v) { this._text = String(v); this.children = []; },
    get innerHTML() { return this._html; },
    set innerHTML(v) { this._html = String(v); },
  };
  return n;
}
function load(payload) {
  const byId = {};
  const document = {
    getElementById: (id) => (byId[id] || (byId[id] = node('div'))),
    createElement: (t) => node(t),
    createElementNS: (ns, t) => node(t),
  };
  const calls = [];
  const fetchFn = (url) => {
    calls.push(url);
    return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(payload) });
  };
  new Function('document', 'window', 'fetch', 'setInterval', 'Node',
    `${src}\n;window.__api = { load, tokens, dur, energy, rate, pct, when, ceilNice, usd };`)(
    document, globalThis.window = {}, fetchFn, () => 0, function Node() {});
  return { byId, calls, api: globalThis.window.__api };
}

/* The payload /lb/sessions really returns, trimmed to what the page reads. */
const FLASH = 'qwen3.8-flash-next-hibrid48';
const SGLANG = 'qwen3.8-27b-sglang-dflash2';
const T0 = 1790740020;
const session = (over) => ({
  v: 1, generation: 52, model_choice: FLASH, label: 'Qwen3.8 Flash-Next', engine: 'vllm', known: true,
  started_at: T0, ended_at: T0 + 3600, wall_s: 3600, active_s: 2400, windows: 40,
  config: { max_num_seqs: 6, max_num_batched_tokens: 4096, num_speculative_tokens: 4 },
  config_variants: 1, gen_tokens: 240000, prompt_tokens: 4800000, requests: 400, gen_tok_s: 100,
  peak_gen_tok_s: 150, prefix_hit_rate: 0.78, tokens_per_request: 600, ttft_mean_s: 2.5,
  itl_mean_s: 0.04, acceptance: 0.56, boot_s: 120, outcome: 'serving', failure: '',
  watts_mean: 70, wh: 46.7, tok_per_kwh: 5142857, ...over });
const PAYLOAD = {
  now: T0 + 7200, days: null, error: '', windows_truncated: 0, malformed: 0, boots_malformed: 0,
  recorder: { enabled: true }, cluster_configured: true,
  sessions: [
    session({ generation: 54, model_choice: SGLANG, label: 'SGLang DFlash2', engine: 'sglang',
              started_at: T0 + 7200, ended_at: T0 + 10800, gen_tokens: 480000, gen_tok_s: 200 }),
    // A GENERATION THAT SERVED NOTHING: it loaded, it failed, and it is part of the history.
    session({ generation: 53, gen_tokens: 0, requests: 0, active_s: 0, gen_tok_s: null,
              peak_gen_tok_s: null, watts_mean: null, wh: null, boot_s: null, ttft_mean_s: null,
              outcome: 'failed', failure: 'rank on box-b exited (code 1)' }),
    session({ generation: 52 }),
  ],
  totals: { sessions: 3, models: 2, gen_tokens: 720000, prompt_tokens: 9600000, requests: 800,
            active_s: 4800, wall_s: 14400, gen_tok_s: 150, wh: 93.4, first_at: T0, last_at: T0 + 10800,
            efficiency: { tok_per_kwh: 4800000, measured: 2, of: 3 } },
  by_model: [
    { model_choice: SGLANG, label: 'SGLang DFlash2', engine: 'sglang', sessions: 1, gen_tokens: 480000,
      prompt_tokens: 4800000, requests: 400, active_s: 2400, wh: 46.7, gen_tok_s: 200,
      efficiency: { tok_per_kwh: 10280000, measured: 1, of: 1 } },
    { model_choice: FLASH, label: 'Qwen3.8 Flash-Next', engine: 'vllm', sessions: 2, gen_tokens: 240000,
      prompt_tokens: 4800000, requests: 400, active_s: 2400, wh: 46.7, gen_tok_s: 100 },
  ],
};

//: the table's columns in order, as sessions.js declares them — the checks index by name.
const COLUMN_NAMES = ['gen', 'model', 'started', 'busy', 'standing', 'tokens', 'tok/s', 'peak',
  'requests', 'tok/req', 'prompt', 'cache hit', 'TTFT', 'ITL', 'seats', 'budget', 'spec', 'accept',
  'boot', 'W', 'Wh', 'tok/kWh'];
const flat = (n, out = []) => (out.push(n), n.children.forEach((k) => flat(k, out)), out);
const rects = (n) => flat(n).filter((k) => k.tag === 'rect');
const titleOf = (n) => (n.children.find((k) => k.tag === 'title') || {})._text || '';

(async () => {
  const { byId, calls, api } = load(PAYLOAD);
  await new Promise((r) => setTimeout(r, 0));

  // THE WHOLE HISTORY by default: this page is not a live view.
  assert.ok(calls[0].includes('days=all'), calls[0]);

  // THE TILES: a number and what it counts, in the units they are read in.
  const tiles = byId['s-tiles'].children.map((t) => t.innerHTML).join(' | ');
  assert.ok(tiles.includes('720,000') || tiles.includes('720.00 k') || tiles.includes('720 k'), tiles);
  assert.ok(/1\.3 h|4800|1\.3/.test(tiles), tiles);
  assert.ok(tiles.includes('150.0'), `the mean rate: ${tiles}`);
  assert.ok(tiles.includes('93 Wh') || tiles.includes('93.4'), tiles);
  assert.strictEqual(byId['s-tiles'].hidden, false);

  // ONE STRIP PER MODEL, EACH ON ITS OWN CEILING: with one scale across them a single huge session
  // flattened every other bar on the real log (88 sessions, one of 7.9 M). Each strip's tallest bar
  // therefore reaches the same height, and the ceiling it was drawn to is printed on it.
  const facets = byId['s-facets'].children;
  assert.strictEqual(facets.length, 2, 'one facet per model that has sessions');
  assert.ok(facets[0].innerHTML.includes('SGLang DFlash2'), facets[0].innerHTML);
  const heights = facets.map((f) => rects(f).map((r) => Number(r.attrs.height)));
  assert.strictEqual(heights[0].length, 1);
  assert.strictEqual(heights[1].length, 2);
  const scale = (tok, h) => h / tok;
  // facet 1 is oldest-first: generation 52's 240k bar, then generation 53's nothing.
  assert.ok(Math.abs(heights[0][0] - heights[1][0]) < 1e-9,
    `each strip fills its own ceiling: ${JSON.stringify(heights)}`);
  const tops = facets.map((f) => flat(f).filter((n) => n.tag === 'text').map((n) => n._text)[0]);
  assert.deepStrictEqual(tops, ['500 k', '250 k'], `and prints the ceiling it used: ${tops}`);

  // A SESSION THAT GENERATED NOTHING has no bar, and says so in its own class — not a 1px bar that
  // reads as "a little".
  const none = rects(facets[1]).find((r) => r.attrs.class.includes('s-bar-none'));
  assert.ok(none && Number(none.attrs.height) === 0, JSON.stringify(none && none.attrs));

  // EVERY BAR CARRIES ITS OWN NUMBERS: a height is a comparison, the figure behind it is the fact.
  const bar = rects(facets[0])[0];
  const t = titleOf(bar);
  assert.ok(t.includes('generation 54') && t.includes('480,000 tokens') && t.includes('200.0 tok/s'), t);
  assert.ok(t.includes('6 seats') && t.includes('budget 4,096'), t);

  // THE CUMULATIVE LINE: one series, one dot per session that ended, each naming the running total.
  const dots = flat(byId['s-cum-chart']).filter((n) => n.tag === 'circle');
  assert.strictEqual(dots.length, 3);
  assert.ok(titleOf(dots[2]).includes('720,000 in total'), titleOf(dots[2]));
  assert.strictEqual(byId['s-cumulative'].hidden, false);

  // THE TABLE: newest first, and what nobody measured is a dash — never a zero.
  const html = byId['s-table'].innerHTML;
  const rows = html.split('<tr').filter((r) => r.includes('<td'));   // the header row has none
  assert.ok(rows[0].includes('>54<'), rows[0]);
  assert.ok(rows[1].includes('s-row-failed') && rows[1].includes('s-row-idle'), rows[1]);
  assert.ok(rows[1].includes('rank on box-b exited'), 'the failure is the row\'s own tooltip');
  const cells = rows[1].split('<td').slice(1).map((c) => c.replace(/^[^>]*>/, '').replace('</td>', ''));
  assert.ok(cells.includes('—'), `an unmeasured figure is a dash: ${cells.join('|')}`);
  assert.ok(!cells.includes('0.0'), `and never a zero: ${cells.join('|')}`);
  assert.ok(byId['s-table-sub']._text.includes('1 loaded and served nothing'), byId['s-table-sub']._text);

  // MONEY: a session that did nothing priced at exactly zero is "$0", never "$0.000"; a cost nobody could
  // compute is a dash — the two must never look alike (priced-at-nothing vs unpriced).
  assert.strictEqual(api.usd(0), '$0');
  assert.strictEqual(api.usd(null), '—');
  assert.strictEqual(api.usd(0.44), '$0.44');
  assert.strictEqual(api.usd(0.0104), '$0.010');
  assert.strictEqual(api.usd(12.3456), '$12.35');
  assert.strictEqual(api.usd(1716.14), '$1,716');

  // THE UNITS, where the page writes them itself.
  assert.strictEqual(api.tokens(20470391), '20.47 M');
  assert.strictEqual(api.tokens(174163), '174 k');
  assert.strictEqual(api.tokens(912), '912');
  assert.strictEqual(api.tokens(null), '—');
  assert.strictEqual(api.dur(227160), '63 h');
  assert.strictEqual(api.dur(90), '2 min');
  assert.strictEqual(api.energy(5169.2), '5.17 kWh');
  assert.strictEqual(api.pct(0.9863), '98.6%');
  assert.strictEqual(api.ceilNice(480000), 500000);

  // TROUBLE IS SAID, above the numbers it would otherwise leave unexplained.
  const bad = load({ ...PAYLOAD, windows_truncated: 1200, recorder: { enabled: false } });
  await new Promise((r) => setTimeout(r, 0));
  const alert = bad.byId['s-alert'];
  assert.strictEqual(alert.hidden, false);
  assert.ok(alert.innerHTML.includes('1,200') && alert.innerHTML.includes('recorder is OFF'), alert.innerHTML);

  // NOTHING RECORDED is not a blank page: it says why there is nothing.
  const empty = load({ ...PAYLOAD, sessions: [], by_model: [],
                       totals: { sessions: 0, models: 0, gen_tokens: 0, active_s: 0 } });
  await new Promise((r) => setTimeout(r, 0));
  assert.strictEqual(empty.byId['s-models'].hidden, true);
  assert.strictEqual(empty.byId['s-table-card'].hidden, true);
  assert.ok(empty.byId['s-loading']._text.includes('No session in this range'), empty.byId['s-loading']._text);

  // ── A ONE-HOST NODE (operator 2026-10-12, 'site-b': 310 recorded minutes, `sessions: []`) ──────────
  // Minutes recorded BEFORE sessions were (no intent, no engine start in them) are said, with their count —
  // never "nothing recorded", never folded into a guessed session.
  const before = load({ ...PAYLOAD, sessions: [], by_model: [], windows_without_a_session: 310,
                        windows_before_sessions: 310, totals: { sessions: 0, models: 0, gen_tokens: 0, active_s: 0 } });
  await new Promise((r) => setTimeout(r, 0));
  assert.ok(before.byId['s-loading']._text.includes('310 minutes from before sessions were recorded'),
    before.byId['s-loading']._text);
  const some = load({ ...PAYLOAD, windows_without_a_session: 12, windows_before_sessions: 10 });
  await new Promise((r) => setTimeout(r, 0));
  const sub = some.byId['s-table-sub']._text;
  assert.ok(sub.includes('10 minutes from before sessions were recorded') && sub.includes('2 minutes in no session'), sub);
  // A NODE THE LB ONLY OBSERVES: its session is one ENGINE PROCESS, known by the engine's own start — no
  // generation, so the gen cell is a dash and the bar and the now-tiles name the engine's start instead.
  const ENGINE = T0 - 600;
  const observed = load({ ...PAYLOAD, sessions: [session({ generation: null, engine_started_at: ENGINE })],
    current: { model_choice: FLASH, generation: null, engine_started_at: ENGINE, state: 'running', label: 'x' } });
  await new Promise((r) => setTimeout(r, 0));
  const oRow = observed.byId['s-table'].innerHTML.split('<tr').filter((r) => r.includes('<td'))[0];
  const gen = oRow.split('<td').slice(1)[0];
  assert.ok(/>—<\/td>/.test(gen) && gen.includes('engine up since 2026-09-30 03:37Z'), gen);
  const oBar = rects(observed.byId['s-facets'].children[0])[0];
  assert.ok(titleOf(oBar).startsWith('engine up since 2026-09-30 03:37Z'), titleOf(oBar));
  const oNow = observed.byId['s-tiles'].children.filter((t) => (t.innerHTML || '').includes('now · '));
  assert.ok(oNow[0].innerHTML.includes('engine up since 2026-09-30 03:37Z')
    && oNow[0].innerHTML.includes('40 min busy'), `the running engine's own row: ${oNow[0].innerHTML}`);

  {
  // ── COLOUR: one ramp, and it means "how good INSIDE ITS OWN COMPARABLE GROUP" ─────────────
  // A rate is washed against the model's OWN sessions (two models are different machines doing
  // different work), a volume against the whole table (a token is a token), a SETTING never.
  const PAY = { ...PAYLOAD, sessions: [
    session({ generation: 61, model_choice: SGLANG, label: 'SGLang', gen_tok_s: 60,
              gen_tokens: 100, requests: 10, ttft_mean_s: 9, config: { max_num_seqs: 32 } }),
    session({ generation: 60, model_choice: SGLANG, label: 'SGLang', gen_tok_s: 200,
              gen_tokens: 900, requests: 90, ttft_mean_s: 1, config: { max_num_seqs: 32 } }),
    session({ generation: 59, model_choice: FLASH, gen_tok_s: 61, gen_tokens: 500,
              requests: 50, ttft_mean_s: 5, config: { max_num_seqs: 6 } }),
  ] };
  const { byId } = load(PAY);
  await new Promise((r) => setTimeout(r, 0));
  const rows = byId['s-table'].innerHTML.split('<tr').filter((r) => r.includes('<td'));
  const cell = (row, i) => (row.split('<td').slice(1)[i] || '');
  const tierOf = (c) => Number((c.match(/--heat-(\d)/) || [])[1] || 0);
  const col = (name) => {
    const i = COLUMN_NAMES.indexOf(name);
    assert.ok(i >= 0, `no column named ${name}`);
    return i;
  };
  // THE LIST ABOVE IS NOT A SECOND TRUTH: it must be the header the page really drew, or every
  // index below is off by one and the checks pass against the wrong cells.
  const drawn = [...byId['s-table'].innerHTML.matchAll(/<th class="[^"]*">([^<]*)/g)].map(([, t]) => t);
  assert.deepStrictEqual(drawn, COLUMN_NAMES, 'the checks index the columns the page drew');

  // tok/s is washed PER MODEL: SGLang's 60 is its own worst, Flash-Next's 61 its only one.
  assert.strictEqual(tierOf(cell(rows[0], col('tok/s'))), 1, 'SGLang 60 tok/s is its model\'s weakest');
  assert.strictEqual(tierOf(cell(rows[1], col('tok/s'))), 4, 'SGLang 200 tok/s is its model\'s best');
  assert.strictEqual(tierOf(cell(rows[2], col('tok/s'))), 0,
    'one session of a model has nothing to be compared against');
  // tokens is a VOLUME: washed across the whole table, 900 > 500 > 100.
  const vol = rows.map((r) => tierOf(cell(r, col('tokens'))));
  assert.ok(vol[1] > vol[2] && vol[2] > vol[0], `volume washes across the table: ${vol}`);
  // TTFT is WAITING: less is better, so the 1 s row is the strong one, not the 9 s row.
  assert.ok(tierOf(cell(rows[1], col('TTFT'))) > tierOf(cell(rows[0], col('TTFT'))),
    'a lower TTFT washes stronger');
  // a SETTING is never washed.
  for (const name of ['seats', 'budget', 'spec']) {
    assert.ok(rows.every((r) => tierOf(cell(r, col(name))) === 0), `${name} is a setting, never washed`);
  }
  // the header says which way is better, so the ramp is never left to be inferred
  const head = byId['s-table'].innerHTML.split('<tbody')[0];
  assert.ok(head.includes('\u25b2') && head.includes('\u25bc'), 'both directions are declared');
  // and the bars wear the same ramp
  const bars = flat(byId['s-facets']).filter((n) => n.tag === 'rect' && Number(n.attrs.height) > 0);
  assert.ok(bars.some((b) => /s-t[1-4]/.test(b.attrs.class)), 'a bar carries its rate\'s tier');
}
{
  // ── PRICES: what the tokens would have cost, side by side, dated, and honest about coverage ──────
  // Flash-Next has no price on its own vendor's list, so its OWN cost is a dash — never $0 — and the
  // own-vendor tile says it covers fewer sessions than the history, instead of presenting a part as the
  // whole. The reference list (Claude) prices every session.
  const OPUS = 'anthropic-claude-opus-5-5', QWEN = 'alibaba-qwen3.8-27b-sg';
  const LISTS = {
    [OPUS]: { id: OPUS, vendor: 'Anthropic', label: 'Claude Opus 5.5', as_of: '2026-10-05',
      source: 'https://platform.claude.com/docs/en/about-claude/pricing', basis: 'base $4, out $20, hit 0.05x',
      excludes: 'the cache-write premium', tokenizer: 'token counts are Qwen\'s own',
      usd_per_mtok: { input: 4, output: 20, cache_read: 0.2 } },
    [QWEN]: { id: QWEN, vendor: 'Alibaba Cloud Model Studio · Singapore (International)', label: 'Qwen3.8-27B',
      as_of: '2026-10-03', source: 'https://www.alibabacloud.com/help/en/model-studio/model-pricing',
      basis: 'implicit cache typically 20%', excludes: '', tokenizer: '',
      usd_per_mtok: { input: 0.5, output: 3, cache_read: 0.1 } },
  };
  const WHY = 'Qwen3.8-Flash-Next is not on Alibaba\'s published price list';
  const priced = {
    ...PAYLOAD,
    sessions: [
      session({ generation: 61, model_choice: SGLANG, label: 'SGLang', gen_tokens: 100000,
                cost: { [QWEN]: 0.44, [OPUS]: 2.58 }, own_price: { list: QWEN, why: '' } }),
      session({ generation: 60, model_choice: FLASH, gen_tokens: 100000,
                cost: { [OPUS]: 1716.14 }, own_price: { list: null, why: WHY } }),
    ],
    totals: { ...PAYLOAD.totals, cost: {
      own: { usd: 0.44, priced: 1, of: 2 }, [OPUS]: { usd: 1718.72, priced: 2, of: 2 } } },
    by_model: [
      { ...PAYLOAD.by_model[0], model_choice: SGLANG, cost: { own: { usd: 0.44, priced: 1, of: 1 },
        [OPUS]: { usd: 2.58, priced: 1, of: 1 } } },
      { ...PAYLOAD.by_model[1], model_choice: FLASH, cost: { own: { usd: null, priced: 0, of: 1 },
        [OPUS]: { usd: 1716.14, priced: 1, of: 1 } } },
    ],
    price_lists: { own: [QWEN], references: [OPUS], lists: LISTS, unpriced: { [FLASH]: WHY }, error: '' },
  };
  const { byId } = load(priced);
  await new Promise((r) => setTimeout(r, 0));
  const tiles = byId['s-tiles'].children;
  const saved = tiles.filter((t) => /saved vs/.test(t.innerHTML));
  assert.strictEqual(saved.length, 2, 'one tile per comparison');
  const ownTile = saved.find((t) => t.innerHTML.includes('Qwen3.8-27B'));
  const refTile = saved.find((t) => t.innerHTML.includes('Claude Opus 5.5'));
  assert.ok(ownTile.innerHTML.includes('$0.44') && ownTile.innerHTML.includes('1 of 2 sessions'),
    `the own-vendor tile says it covers fewer sessions than the history: ${ownTile.innerHTML}`);
  assert.ok(refTile.innerHTML.includes('$1,719') && refTile.innerHTML.includes('2 of 2 sessions'), refTile.innerHTML);
  // WHERE THE NUMBERS CAME FROM travels with them: the date, the source, and what is not counted.
  assert.ok(refTile.title.includes('2026-10-05') && refTile.title.includes('platform.claude.com')
    && refTile.title.includes('cache-write premium') && refTile.title.includes('Qwen\'s own'), refTile.title);
  assert.ok(ownTile.title.includes('Qwen3.8-Flash-Next is not on Alibaba'), 'and the unpriced model says why');
  // the dated price line is PRINTED, not tucked in a tooltip
  const strip = byId['s-prices'].innerHTML;
  assert.strictEqual(byId['s-prices'].hidden, false);
  assert.ok(strip.includes('$4 / $20 / $0.2') && strip.includes('2026-10-05') && strip.includes('2026-10-03'), strip);
  // the table: a column per comparison; an unpriced cell is a DASH with the reason, never $0
  const head = byId['s-table'].innerHTML.split('<tbody')[0];
  assert.ok(head.includes('$ own') && head.includes('$ opus 5.5'), head);
  const rows = byId['s-table'].innerHTML.split('<tr').filter((r) => r.includes('<td'));
  const cells = (r) => r.split('<td').slice(1);
  const flashRow = cells(rows[1]);
  const ownCell = flashRow[flashRow.length - 2];
  assert.ok(/>—<\/td>/.test(ownCell) && ownCell.includes('not on Alibaba'),
    `an unpriced cell is a dash that says why: ${ownCell}`);
  assert.ok(!/\$0(\.0+)?</.test(ownCell), 'and never $0');
  assert.ok(cells(rows[1])[flashRow.length - 1].includes('$1,716'), 'while the reference prices it');
  // a facet says what its model saved, where there is a price, and leaves an unpriced comparison out
  const sg = byId['s-facets'].children.map((f) => f.innerHTML).find((h) => h.includes('SGLang'));
  assert.ok(sg.includes('$0.44 vs own list') && sg.includes('$2.58 vs opus 5 5'), sg);
  const fl = byId['s-facets'].children.map((f) => f.innerHTML).find((h) => h.includes('Flash-Next'));
  assert.ok(!fl.includes('vs own list'), 'an unpriced comparison is left out of the line, not drawn as $0');
  // AND THE RATES ITS DOLLARS WERE MADE AT, printed on the strip. The two lists bill differently
  // (qwen $0.50/$3/$0.10, opus $4/$20/$0.20), so a per-model dollar with no rate beside it is
  // unreadable next to another model's. The chip is the model's OWN list — its three numbers, with
  // the listTitle tooltip (date + source) on it — and an unpriced model gets NO chip: never $0,
  // never the reference list's numbers standing in for a list that model does not have.
  assert.ok(sg.includes('$0.50 fill') && sg.includes('$3.00 decode') && sg.includes('$0.10 cache'),
    `a priced strip prints the three rates of its own list: ${sg}`);
  assert.ok(!/\$\d[\d.]* (fill|decode|cache)/.test(fl),
    `an unpriced model prints no rates at all: ${fl}`);
  const chip = sg.match(/class="s-facet-rates" title="([\s\S]*?)"/);
  assert.ok(chip && chip[1].includes('as of 2026-10-03') && chip[1].includes('alibabacloud.com'),
    `the chip carries its date and source: ${chip && chip[1]}`);
  assert.ok(!/\$4\.00 (fill|decode)/.test(sg), 'its own numbers, never the other list\'s');
  // NOW (operator, 2026-10-06: "on top show current session running model costs and Opus, below total
  // all models as is shown now"): the FIRST tiles of the page are the ONE session the serving pair owns
  // — keyed by its model id + generation, the same tag the log stamps minutes with — and the totals
  // below stay the whole history, unchanged.
  const rNow = load({ ...priced,
    current: { model_choice: SGLANG, generation: 61, state: 'running', label: 'SGLang' } });
  await new Promise((r) => setTimeout(r, 0));
  const isNow = (t) => (t.innerHTML || '').includes('now · ');
  const nt = rNow.byId['s-tiles'].children;
  assert.ok(isNow(nt[0]) && isNow(nt[1]), 'the two first tiles are the running session');
  assert.ok(nt[0].innerHTML.includes('Qwen3.8-27B') && nt[0].innerHTML.includes('$0.44')
    && nt[0].innerHTML.includes('gen 61'), `own vendor's dollars for THIS generation: ${nt[0].innerHTML}`);
  assert.ok(nt[1].innerHTML.includes('vs Claude Opus 5.5') && nt[1].innerHTML.includes('$2.58'),
    nt[1].innerHTML);
  assert.ok(!isNow(nt[2]), 'then the all-model totals, as before');
  assert.ok(nt.filter((t) => /saved vs/.test(t.innerHTML || '')).length === 2,
    'the whole-history savings tiles are still drawn below');
  // an unpriced RUNNING model (Flash-Next on its own vendor) is a dash with the reason, never $0
  const rUnp = load({ ...priced,
    current: { model_choice: FLASH, generation: 60, state: 'running', label: 'Flash-Next' } });
  await new Promise((r) => setTimeout(r, 0));
  const unp = rUnp.byId['s-tiles'].children.filter(isNow);
  assert.ok(/>—<\/span>/.test(unp[0].innerHTML) && unp[0].title.includes('not on Alibaba'),
    `no published price, said, not $0: ${unp[0].innerHTML} | ${unp[0].title}`);
  assert.ok(unp[1].innerHTML.includes('$1,716'), 'while the reference still prices it');
  // a pair serving a generation with no recorded minute in range: dashes that say so, never an empty tile
  const rNoRec = load({ ...priced,
    current: { model_choice: SGLANG, generation: 99, state: 'running', label: 'SGLang' } });
  await new Promise((r) => setTimeout(r, 0));
  const noRec = rNoRec.byId['s-tiles'].children.filter(isNow);
  assert.strictEqual(noRec.length, 2);
  assert.ok(/>—<\/span>/.test(noRec[0].innerHTML) && noRec[0].title.includes('no recorded minute'), noRec[0].title);
  // no `current` at all (no pair serving — an LB without a cluster loop, or a stopped pair): no tiles.
  const rNone = load({ ...priced, current: null });
  await new Promise((r) => setTimeout(r, 0));
  assert.ok(!rNone.byId['s-tiles'].children.some(isNow), 'no pair, no tile — never a stand-in session');
}
{
  // A server that publishes NO costs (an LB older than this, or a price file it could not read) draws no
  // price tile, column or strip — and nothing else about the page changes.
  const { byId } = load(PAYLOAD);
  await new Promise((r) => setTimeout(r, 0));
  assert.ok(!byId['s-tiles'].children.some((t) => /saved vs/.test(t.innerHTML)), 'no tile without costs');
  assert.ok(!byId['s-table'].innerHTML.includes('$ own'), 'no column without costs');
  assert.strictEqual(byId['s-prices'].hidden, true, 'no price strip without prices');
  assert.ok(!byId['s-facets'].children.some((f) => f.innerHTML.includes('s-facet-rates')),
    'no rate chip on a server that publishes no prices');
  const broken = load({ ...PAYLOAD, price_lists: { own: [], references: [], lists: {}, unpriced: {},
    error: '/app/vllm_lb/pricing.json: ValueError: nope' } });
  await new Promise((r) => setTimeout(r, 0));
  assert.ok(broken.byId['s-alert'].innerHTML.includes('no prices'), 'an unreadable price file is an alert');
}
{
  // ── TOKENS PER kWh: a column, a total, a facet figure — and a dash where either side is missing ──
  const PAY = { ...PAYLOAD, sessions: [
    session({ generation: 71, model_choice: SGLANG, label: 'SGLang', tok_per_kwh: 4000000 }),
    session({ generation: 70, model_choice: SGLANG, label: 'SGLang', tok_per_kwh: 9000000 }),
    // no power reading: the ratio cannot be computed, so it is a dash — not 0, not infinite
    session({ generation: 69, model_choice: FLASH, tok_per_kwh: null, wh: null, watts_mean: null }),
  ] };
  const { byId } = load(PAY);
  await new Promise((r) => setTimeout(r, 0));
  const t = byId['s-tiles'].children.find((x) => /tokens per kWh/.test(x.innerHTML));
  assert.ok(t, 'the totals carry a tokens-per-kWh tile');
  assert.ok(t.innerHTML.includes('4.80 M') && t.innerHTML.includes('2 of 3 sessions'),
    `the tile says how many sessions it could be computed for: ${t.innerHTML}`);
  assert.ok(/ratio of sums/.test(t.title) && /Idle time/.test(t.title),
    'and says what the power does and does not include');
  const rows = byId['s-table'].innerHTML.split('<tr').filter((r) => r.includes('<td'));
  const kwh = (row) => (row.split('<td').slice(1).pop() || '');
  assert.ok(kwh(rows[0]).includes('4.00 M') && kwh(rows[1]).includes('9.00 M'), 'the column shows each session\'s');
  assert.ok(/>—<\/td>/.test(kwh(rows[2])) && !/>0</.test(kwh(rows[2])),
    `no power reading is a dash, never 0 tok/kWh: ${kwh(rows[2])}`);
  // washed against the MODEL's own sessions, more = better: SGLang's 9 M beats its 4 M
  const tierOf = (c) => Number((c.match(/--heat-(\d)/) || [])[1] || 0);
  assert.ok(tierOf(kwh(rows[1])) > tierOf(kwh(rows[0])), 'more tokens per kWh washes stronger');
  assert.strictEqual(tierOf(kwh(rows[2])), 0, 'an unmeasured cell is not washed');
  const sg = byId['s-facets'].children.map((f) => f.innerHTML).find((h) => h.includes('SGLang'));
  assert.ok(/10\.28 M tok\/kWh/.test(sg), `a facet says its model's efficiency: ${sg}`);
  // an LB that publishes no efficiency draws a dash tile rather than inventing a figure
  const old = { ...PAYLOAD, totals: { ...PAYLOAD.totals } };
  delete old.totals.efficiency;
  const o = load(old);
  await new Promise((r) => setTimeout(r, 0));
  const ot = o.byId['s-tiles'].children.find((x) => /tokens per kWh/.test(x.innerHTML));
  assert.ok(ot && ot.innerHTML.includes('—'), 'no efficiency published: a dash, not a made-up number');
}
console.log('ok — the session history draws its totals, one strip per model on its own scale, and its table');
})().catch((e) => { console.error(e); process.exit(1); });
