/* THE FIDELITY SECTION as drawn (static/fidelity.js), under node:
 *
 *   1. the Run button carries its state: "Run", "starting…" while the POST is in flight, "running 7/15" while a run
 *      runs (disabled both times), disabled with no corpus;
 *   2. a finished run is one row: model (engine × accelerator × TP under it) · quant · KV dtype · mean NLL · perplexity ·
 *      tokens · when, with a checkbox; an unread figure is "—"; a failed run is ONE line — its status and the error's
 *      first clause ("failed · returned no token logprobs") — opening (<details>, title) to the server's whole words,
 *      no figure, and no checkbox; every server string is escaped;
 *   3. ticking keeps at most two (the oldest tick drops); the OLDER run is A whatever the tick order;
 *   4. the comparison prints ΔNLL with its 95 % interval and "same quality" when the interval spans 0, "B worse" /
 *      "B better" when it does not, the reason when the runs cannot be paired;
 *   5. nothing POSTs at load: only the button's click does.
 *
 *   node tests/fidelity.node.js      (tests/test_fidelity_node.py runs this when node exists)
 */
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const src = fs.readFileSync(path.join(__dirname, '..', 'vllm_lb', 'static', 'fidelity.js'), 'utf8');
const posts = [];
const fetch = (url, opts) => { if (opts && opts.method === 'POST') posts.push(url); return new Promise(() => {}); };
const listeners = [];
const root = { innerHTML: '', addEventListener: (k, fn) => listeners.push(k) };
const document = { readyState: 'complete', addEventListener() {}, getElementById: (id) => (id === 'fid-root' ? root : null) };
const api = new Function('document', 'fetch', 'setTimeout', 'clearTimeout',
  `${src}\n;return { fidButton, fidRowsHtml, fidCompareHtml, fidSectionHtml, fidToggle, fidPair, fidCorpusLine, fidErrLine };`)(
  document, fetch, () => 0, () => {});

// ── 5. loading the page starts nothing ───────────────────────────────────────────────────────
assert.deepStrictEqual(posts, [], 'no POST at load');
assert.ok(listeners.includes('click') && listeners.includes('change'));

// ── 1. the button ────────────────────────────────────────────────────────────────────────────
const corpus = { sha256: '830241b5b7b2db342012a74dff68ff22', docs: 15, bytes: 9812, file: 'fidelity_corpus.txt' };
assert.deepStrictEqual(api.fidButton({ corpus, active: null }, false), { label: 'Run', disabled: false });
assert.deepStrictEqual(api.fidButton({ corpus, active: null }, true), { label: 'starting…', disabled: true });
assert.deepStrictEqual(api.fidButton({ corpus, active: { done_docs: 7, total_docs: 15 } }, false),
  { label: 'running 7/15', disabled: true });
assert.strictEqual(api.fidButton({ corpus: null, active: null }, false).disabled, true);
assert.strictEqual(api.fidCorpusLine({ corpus }), '15 docs · 9,812 bytes · sha 830241b5b7b2');
assert.ok(api.fidCorpusLine({ corpus: null, corpus_error: 'sha mismatch' }).includes('sha mismatch'));

// ── 2. the rows ──────────────────────────────────────────────────────────────────────────────
const NOW = 1_795_000_000;
const meta = { served_name: 'm', model_label: 'Qwen <X>', engine: 'sglang', accelerator: 'acc-a', tp: 2,
  quant: 'modelopt', kv_cache_dtype: 'fp8_e4m3', why: {} };
const a = { id: 'fid-20261127T100000Z-aaaa', status: 'done', started_at: NOW - 3600, meta, mean_nll: 1.23456, ppl: 3.4367,
  tokens: 2345, error: null };
const b = { id: 'fid-20261127T110000Z-bbbb', status: 'done', started_at: NOW - 60,
  meta: { ...meta, kv_cache_dtype: null, accelerator: null, tp: null }, mean_nll: 1.25, ppl: 3.49, tokens: 2345, error: null };
const f = { id: 'fid-20261127T120000Z-cccc', status: 'failed', started_at: NOW - 30, meta: { ...meta, engine: 'tensorfold' },
  mean_nll: null, ppl: null, tokens: 0, error: 'tensorfold (m), document genesis: returned no token logprobs <x>' };
let html = api.fidRowsHtml([f, b, a], [], NOW);
assert.strictEqual((html.match(/<tr /g) || []).length, 3);
assert.ok(html.includes('1.2346') && html.includes('3.437') && html.includes('2,345'), 'NLL · ppl · tokens');
assert.ok(html.includes('modelopt') && html.includes('fp8_e4m3'), 'quant · KV');
assert.ok(html.includes('sglang × acc-a × TP 2'), 'engine × accelerator × TP');
assert.ok(html.includes('sglang × — × TP —'), 'an unread figure is a dash');
assert.ok(html.includes('60 min ago') && html.includes('60 s ago'));
assert.ok(html.includes('Qwen &lt;X&gt;') && !html.includes('Qwen <X>'), 'escaped');
assert.ok(html.includes('no token logprobs &lt;x&gt;') && html.includes('fid-failed'), 'the failure in its words');
// …as one line: status · the error's first clause, without its subject (engine, document); the whole error opens on
// tap / keyboard (<details>) and hovers (title).
const fe = { ...f, error: 'tensorfold (m), document genesis: returned no token logprobs (choices[0].logprobs.token_logprobs'
  + ' is absent) — this engine does not score <the> prompt' };
const fh = api.fidRowsHtml([fe], [], NOW);
assert.ok(fh.includes('<details class="fid-errd"><summary title="tensorfold (m), document genesis: returned no token'), fh);
assert.ok(fh.includes('<span class="fid-status">failed</span> · returned no token logprobs</summary>'), fh);
assert.ok(fh.includes('does not score &lt;the&gt; prompt</p></details>'), 'the whole error inside, escaped');
assert.strictEqual(api.fidErrLine('the gateway restarted mid-run: not resumed'), 'not resumed');
assert.strictEqual(api.fidErrLine('HTTP 503 from the gateway for document d1: engine overloaded; retry'),
  'engine overloaded');
assert.strictEqual(api.fidErrLine('no colon here at all'), 'no colon here at all');
assert.ok(api.fidErrLine(`x: ${'a'.repeat(100)}`).length <= 48 && api.fidErrLine(`x: ${'a'.repeat(100)}`).endsWith('…'));
const iw = api.fidRowsHtml([{ ...f, status: 'interrupted', error: null }], [], NOW);
assert.ok(iw.includes('<span class="fid-status">interrupted</span>') && !iw.includes('fid-errd'), 'no error: no details');
assert.strictEqual((html.match(/type="checkbox"/g) || []).length, 2, 'a failed run cannot be ticked');
// A target the LB could not resolve: the dashes say WHAT is unread, the run's `meta.why` says WHY — a short figure-style
// label that opens (tap / keyboard: a <details>) or hovers (title) to the reason, the same reason grouped once.
const why = 'Qwen 27B\'s cluster launch did not resolve to a target on this fleet';
const unres = { ...a, id: 'fid-20261127T090000Z-dddd', meta: { ...meta, accelerator: null, tp: null,
  why: { accelerator: why, tp: why } } };
const uh = api.fidRowsHtml([unres], [], NOW);
assert.ok(uh.includes('sglang × — × TP —'), 'the dashes stay');
assert.ok(uh.includes('<details class="fid-why" data-fid-why="fid-20261127T090000Z-dddd">') && uh.includes('<summary title="accelerator, tp: Qwen 27B&#39;s cluster'), uh);
assert.ok(uh.includes('>2 unread</summary>'), 'the label is a count, not a sentence');
assert.ok(uh.includes('<li><b>accelerator, tp</b> · Qwen 27B&#39;s cluster launch did not resolve to a target on this fleet</li>'),
  'one line per distinct reason, naming the figures it covers');
assert.ok(!api.fidRowsHtml([a], [], NOW).includes('fid-why'), 'a run with nothing unread has no why');
const uf = api.fidRowsHtml([{ ...f, meta: { ...f.meta, why: { quant: 'no quant <q>' } } }], [], NOW);
assert.ok(uf.includes('>1 unread</summary>') && uf.includes('no quant &lt;q&gt;'), 'a failed row shows it too, escaped');

// A phone stacks each row into a card (fidelity.css): every figure cell names its column, so the card can label it.
for (const label of ['quant', 'KV dtype', 'mean NLL', 'perplexity', 'tokens', 'when']) {
  assert.ok(html.includes(`data-label="${label}"`), `a cell labelled ${label}`);
}
assert.ok(/<td class="fid-pick">/.test(html) && /<td class="fid-m">/.test(html), 'the tick and model cells are named');
assert.ok(api.fidRowsHtml([], [], NOW).includes('0 runs'));
html = api.fidRowsHtml([b, a], [a.id], NOW);
assert.strictEqual((html.match(/ checked/g) || []).length, 1);
assert.strictEqual((html.match(/aria-selected="true"/g) || []).length, 1);
html = api.fidRowsHtml([{ id: 'x', status: 'running', started_at: NOW, meta, done_docs: 3, total_docs: 15 }], [], NOW);
assert.ok(html.includes('running 3/15'));

// ── 3. ticking ───────────────────────────────────────────────────────────────────────────────
let sel = api.fidToggle([], b.id, true);
sel = api.fidToggle(sel, a.id, true);
assert.deepStrictEqual(api.fidPair(sel, [b, a]), [a.id, b.id], 'the older run is A');
sel = api.fidToggle(sel, 'third', true);
assert.deepStrictEqual(sel, [a.id, 'third'], 'at most two, the oldest tick drops');
assert.deepStrictEqual(api.fidToggle(sel, a.id, false), ['third']);

// ── 4. the comparison ────────────────────────────────────────────────────────────────────────
assert.ok(api.fidCompareHtml(null, [a.id]).includes('1/2'));
assert.ok(api.fidCompareHtml(null, [a.id, b.id]).includes('comparing'));
const same = { comparable: true, d_nll: 0.0154, ci95: [-0.0021, 0.0301], same_quality: true, nll_a: 1.2346, nll_b: 1.25,
  ppl_a: 3.437, ppl_b: 3.49, matched_docs: 15, mismatched_docs: 0, tokens: 2345 };
html = api.fidCompareHtml(same, [a.id, b.id]);
assert.ok(html.includes('ΔNLL +0.0154') && html.includes('[−0.0021, +0.0301]') && html.includes('same quality'), html);
assert.ok(html.includes('15 docs') && html.includes('2,345 tokens') && !html.includes('tokenised differently'));
html = api.fidCompareHtml({ ...same, ci95: [0.01, 0.02], same_quality: false, mismatched_docs: 2 }, [a.id, b.id]);
assert.ok(html.includes('B worse') && !html.includes('same quality') && html.includes('2 docs tokenised differently'));
html = api.fidCompareHtml({ ...same, d_nll: -0.015, ci95: [-0.02, -0.01], same_quality: false }, [a.id, b.id]);
assert.ok(html.includes('ΔNLL −0.0150') && html.includes('B better'));
html = api.fidCompareHtml({ comparable: false, reason: 'the runs scored a <different> corpus' }, [a.id, b.id]);
assert.ok(html.includes('not comparable') && html.includes('&lt;different&gt;'));
assert.ok(api.fidCompareHtml({ error: 'HTTP 404' }, [a.id, b.id]).includes('HTTP 404'));

// ── the section ──────────────────────────────────────────────────────────────────────────────
html = api.fidSectionHtml({ view: { corpus, active: null, runs: [b, a] }, sel: [], cmp: null, posting: false,
  error: 'run fid-x is running', now: NOW });
for (const h of ['model', 'quant', 'KV dtype', 'mean NLL', 'perplexity', 'tokens', 'when']) assert.ok(html.includes(`>${h}</th>`), h);
assert.ok(html.includes('>Fidelity</h2>') && html.includes('data-fid="run"') && html.includes('role="alert"'));

console.log('ok');
