/* The batch-size curve's spread, checked where it is computed: in benchmarks.js, under node.
 *
 * The page pools every window's per-level TTFT histogram (perflog._Bucket.view `ttft_buckets`) in the
 * browser and reads p10 / median / p90 and the distribution off the pooled counts — the same arithmetic
 * as perflog.quantile, so the numbers here mirror tests/test_perflog.py's. No DOM: the file is loaded
 * with a stub `document` (its top level only registers listeners) and a `fetch` that never answers, so
 * `boot()` waits forever and nothing is drawn.
 *
 *   node tests/curve_spread.node.js          (tests/test_curve_spread.py runs this when node exists)
 */
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');

// the page loads server-scope.js first (its `serverScope`): so does this harness
const src = fs.readFileSync(path.join(__dirname, '..', 'vllm_lb', 'static', 'server-scope.js'), 'utf8') + '\n'
  + fs.readFileSync(path.join(__dirname, '..', 'vllm_lb', 'static', 'benchmarks.js'), 'utf8');
const el = () => ({ addEventListener() {}, setAttribute() {}, append() {}, classList: { toggle() {}, add() {} },
  style: {}, hidden: false, innerHTML: '', value: '', dataset: {}, querySelectorAll: () => [], observe() {} });
const document = { getElementById: el, createElementNS: el, createElement: el, addEventListener() {},
  querySelectorAll: () => [], body: { style: {} } };
const window = { addEventListener() {}, innerWidth: 1280, innerHeight: 800 };
const api = new Function('document', 'window', 'fetch', 'location', 'setInterval', 'getComputedStyle', 'ResizeObserver',
  `${src}\n;return { bucketSums, levelFigures, pooledQuantile, histRows, CURVE, MIN_SPREAD_N, pooledRows, aggregate, tailNote, sampleQuantile, minuteBand, MIN_SPREAD_MINUTES, meanVsMedianNote };`)(
  document, window, () => new Promise(() => {}), { search: '' }, () => 0, () => ({}), undefined);

const near = (a, b, eps = 1e-9) => assert.ok(Math.abs(a - b) <= eps, `${a} != ${b}`);
const cfg = { model_choice: 'm', max_num_seqs: 16, max_num_batched_tokens: 8192, num_speculative_tokens: 4, max_model_len: 262144 };
const win = (t, lv, b) => ({ t_start: t, t_end: t + 60, span_s: 60, config: cfg, config_source: 'effective', by_running: { [lv]: b } });
const level = (ws, lv) => api.levelFigures([...api.bucketSums(ws).values()][0].levels.get(lv));
let checks = 0;

// 1. Two minutes at one level: the histograms are summed per bucket, then read exactly.
{
  const f = level([
    win(0, '13-16', { pairs: 10, span_s: 60, ttft_n: 10, ttft_sum: 6, ttft_buckets: [[0.5, 5], [1, 3], [null, 2]] }),
    win(60, '13-16', { pairs: 10, span_s: 60, ttft_n: 5, ttft_sum: 2, ttft_buckets: [[0.5, 5]] }),
  ], '13-16');
  assert.strictEqual(f.ttft_n, 15);
  near(f.ttft_mean, 8 / 15);
  near(f.ttft_p10, 0.075);                       // rank 1.5 inside the first bucket (0, 0.5], 10 there
  near(f.ttft_p50, 0.375);
  assert.strictEqual(f.ttft_p90, 1.0);           // rank 13.5 lands in +Inf: the last finite bound, never further
  assert.deepStrictEqual(f.ttft_rows.map((r) => [r.lo, r.le, r.n]), [[0, 0.5, 10], [0.5, 1, 3], [1, Infinity, 2]]);
  near(f.ttft_rows[0].share, 10 / 15); near(f.ttft_rows[1].cum, 13 / 15); near(f.ttft_rows[2].cum, 1);
  checks += 8;
}
// 2. A level whose engine exposed no TTFT histogram has no spread and no distribution — never a zero.
{
  const f = level([win(0, '2', { pairs: 3, span_s: 18, gen_tokens: 40 })], '2');
  assert.strictEqual(f.ttft_p10, null); assert.strictEqual(f.ttft_p90, null); assert.strictEqual(f.ttft_n, 0);
  assert.deepStrictEqual(f.ttft_rows, []);
  checks += 4;
}
// 3. The quantile is Prometheus' histogram_quantile — the numbers of test_perflog.py, per-bucket form.
{
  const h = (buckets) => [{ h: { buckets } }];
  assert.strictEqual(api.pooledQuantile(h([[1, 10], [2, 10], [null, 0]]), 'h', 0.5), 1.0);
  near(api.pooledQuantile(h([[1, 10], [2, 10], [null, 0]]), 'h', 0.9), 1.8);
  assert.strictEqual(api.pooledQuantile(h([[1, 1], [null, 9]]), 'h', 0.9), 1.0);
  assert.strictEqual(api.pooledQuantile(h([[1, 0], [null, 0]]), 'h', 0.9), null);
  assert.deepStrictEqual(api.histRows(new Map()), []);
  checks += 5;
}
// 4. ONE measure carries a per-REQUEST spread: time to first token, the only one with the engine's own
//    histogram behind it. Every other measure's spread is across the RECORDED MINUTES — a different
//    quantity — and its note must say so in words, because a reader has to be able to tell "10% of the
//    requests" from "10% of the minutes" without opening this file.
{
  const by = Object.fromEntries(api.CURVE.map((c) => [c.key, c]));
  assert.strictEqual(by.ttft_mean.spread, 'ttft');
  assert.deepStrictEqual(api.CURVE.filter((c) => c.spread === 'ttft').map((c) => c.key), ['ttft_mean']);
  const perMinute = api.CURVE.filter((c) => c.key !== 'ttft_mean');
  assert.deepStrictEqual(perMinute.map((c) => c.spread), perMinute.map(() => 'minutes'));
  for (const c of perMinute) {
    assert.ok(typeof c.note === 'string' && /minutes/.test(c.note), `${c.key}: its note never names the minutes`);
    assert.ok(/not across the requests/.test(c.note), `${c.key}: its note does not deny a per-request spread`);
  }
  assert.ok(Number.isInteger(api.MIN_SPREAD_N) && api.MIN_SPREAD_N >= 10);
  checks += 4 + perMinute.length * 2;
}
// 5. A MEAN ABOVE A P90 is not a bug, and the page must say why. Tonight's real inter-token buckets
//    (2026-09-30, one minute of the 8-seqs run): 2085 of 2329 gaps in one bucket at ~150 ms and a thin tail
//    of multi-second stalls, so the arithmetic mean (202.7 ms) sits above the p90 (150 ms). The p99 reaches
//    the tail (2.38 s, inside the 0.75–2.5 s bucket); the pooled rows are the distribution behind both; and
//    tailNote is the sentence printed beside the two numbers — empty in the usual case.
{
  const itl = { n: 2329, mean: 0.2027, p90: 0.15, buckets: [[0.1, 12], [0.15, 2085], [0.2, 94], [0.3, 37], [0.4, 12], [0.75, 6], [2.5, 64], [5.0, 12], [7.5, 7]] };
  const w = { t_start: 0, t_end: 60, span_s: 60, config: cfg, config_source: 'effective', itl, ttft: { n: 0 } };
  near(api.pooledQuantile([w], 'itl', 0.9), 0.14998, 1e-4);
  near(api.pooledQuantile([w], 'itl', 0.99), 2.3827, 1e-3);
  const a = api.aggregate([w]);
  near(a.itl_mean, 202.7, 1e-6); near(a.itl_p90, 149.98, 0.1); near(a.itl_p99, 2382.7, 1);
  assert.strictEqual(a.ttft_p99, null);                     // no first-token histogram in this minute: null, never 0
  const rows = api.pooledRows([w], 'itl');
  assert.strictEqual(rows.length, 9);
  near(rows[1].share, 2085 / 2329); near(rows[rows.length - 1].cum, 1);
  assert.ok(/above the p90/.test(api.tailNote(a.itl_mean, a.itl_p90, 'gaps')));
  assert.strictEqual(api.tailNote(100, 150, 'gaps'), '');
  assert.strictEqual(api.tailNote(null, 150, 'gaps'), '');
  checks += 11;
}
// 6. THE SAMPLE NOBODY HAD USED. A rate at a level keeps no per-request sample — but every recorded
//    MINUTE is one observation of it (perflog keeps `by_running` per window, and this page already draws
//    the windows one by one), so a percentile ACROSS THE MINUTES is real. Ten minutes at one level,
//    10 .. 100 tok/s, a clean 60 s each:
{
  const ws = [];
  for (let i = 0; i < 10; i++) ws.push(win(i * 60, '13-16', { pairs: 10, span_s: 60, gen_tokens: (i + 1) * 10 * 60 }));
  const f = level(ws, '13-16');
  near(f.agg_tok_s, 55);                                  // the BAR: tokens ÷ seconds over every minute
  assert.deepStrictEqual(f.minute.agg_tok_s, [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]);
  assert.strictEqual(f.minute_n, 10);
  const b = api.minuteBand(f, 'agg_tok_s');
  assert.strictEqual(b.kind, 'minutes'); assert.strictEqual(b.n, 10);
  near(b.lo, 20); near(b.mid, 60); near(b.hi, 90);        // nearest rank, the table's own arithmetic
  checks += 8;
}
// A minute's own figure is the CELL'S formula applied to one minute, so a short minute counts once however
// short it is — and the bar, being seconds-weighted, is then not the middle of the minutes. Both numbers
// are right and the page prints both: their gap is the cell's skew.
{
  const f = level([win(0, '5-8', { pairs: 5, span_s: 60, gen_tokens: 6000 }),
    win(60, '5-8', { pairs: 1, span_s: 6, gen_tokens: 60 })], '5-8');
  near(f.agg_tok_s, 6060 / 66);                           // 91.8 tok/s: the 60 s minute weighs ten times the 6 s one
  assert.deepStrictEqual(f.minute.agg_tok_s, [10, 100]);
  assert.strictEqual(api.minuteBand(f, 'agg_tok_s'), null);          // two minutes is not a tenth of anything
  assert.ok(/above the median/.test(api.meanVsMedianNote(6060 / 66, 55)), 'no sentence for a mean above the median');
  assert.ok(/below the median/.test(api.meanVsMedianNote(55, 6060 / 66)), 'no sentence for a mean below the median');
  assert.strictEqual(api.meanVsMedianNote(100, 101), '');            // 1% apart: nothing to explain
  checks += 6;
}
// THE THRESHOLD, and why it is its own number rather than MIN_SPREAD_N: the slowest 10% of the minutes has
// to BE at least one whole recorded minute. Nine minutes is below that line, ten is on it. (Tonight's real
// log: the biggest (configuration, level) cell the gateway serves holds 12 minutes, so MIN_SPREAD_N's 20
// requests — a low bar for a histogram of hundreds — would have drawn this band on nothing at all.)
{
  const at = (k) => {
    const ws = [];
    for (let i = 0; i < k; i++) ws.push(win(i * 60, '9-12', { pairs: 10, span_s: 60, gen_tokens: (i + 1) * 60 }));
    return api.minuteBand(level(ws, '9-12'), 'agg_tok_s');
  };
  assert.strictEqual(api.MIN_SPREAD_MINUTES, 10);
  assert.strictEqual(at(9), null);
  assert.ok(at(10) && at(10).n === 10);
  checks += 3;
}
// WHY A LIST OF MINUTES CANNOT GO THROUGH pooledQuantile. That one reads a HISTOGRAM, whose first bucket
// really does begin at 0, so it interpolates from 0 up to the lowest bound — over a list of recorded
// minutes it would report a p10 BELOW the slowest minute anybody recorded. Nearest rank only ever returns
// a figure some minute actually measured.
{
  assert.strictEqual(api.sampleQuantile([50, 100], 0.1), 50);
  assert.strictEqual(api.sampleQuantile([], 0.5), null);
  assert.ok(api.pooledQuantile([{ h: { buckets: [[50, 1], [100, 1]] } }], 'h', 0.1) < 50);
  checks += 3;
}
// Time to first token keeps the per-REQUEST spread it has and gains no per-minute band: one measure, one
// spread, and the two quantities are never mixed.
{
  const f = level([win(0, '2', { pairs: 10, span_s: 60, ttft_n: 10, ttft_sum: 6, ttft_buckets: [[0.5, 10]] })], '2');
  assert.strictEqual(api.minuteBand(f, 'ttft_mean'), null);
  assert.ok(!('ttft_mean' in f.minute));
  checks += 2;
}
console.log(`ok — ${checks} checks`);
