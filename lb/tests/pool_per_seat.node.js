/* A KV POOL RESERVED ONE WINDOW A SEAT (TensorFold) — checked where gateway.js draws the pool.
 *
 * Operator, 2026-10-29: "still not all hardware stats are shown" on #pair. Generation 281 (TensorFold Flash-Next,
 * 16 seats x 131,072) read "KV pool 28 GB · 1,777,890 tokens · fits 13 @ 131,072" — the gateway's configured 28 GB
 * pin, which TensorFold never takes — and its KV tile had no token figure. Its pool IS seats x context at the
 * model's bytes a token (`pool_per_seat`), so:
 *
 *   * the pool is 16 x 131,072 = 2,097,152 tokens, 16 x 131,072 x 15,749 B, and every seat fits;
 *   * the KV tile's share is of THAT pool (resident / pool, in millions of tokens);
 *   * a shared pool (vLLM, SGLang) is still its pin's, as before.
 *
 *   node tests/pool_per_seat.node.js   (tests/test_pool_per_seat.py runs this when node exists)
 */
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const src = fs.readFileSync(path.join(__dirname, '..', 'vllm_lb', 'static', 'gateway.js'), 'utf8');
const el = () => ({ addEventListener() {}, removeAttribute() {}, setAttribute() {}, append() {},
  classList: { toggle() {}, add() {}, remove() {} }, style: {}, hidden: false, innerHTML: '',
  textContent: '', value: '', dataset: {}, querySelectorAll: () => [], querySelector: () => null,
  focus() {}, observe() {}, closest: () => null, getBoundingClientRect: () => ({}) });
const document = { getElementById: el, createElementNS: el, createElement: el, addEventListener() {},
  querySelectorAll: () => [], querySelector: () => null, body: { style: {}, classList: { toggle() {} } },
  documentElement: { style: {}, classList: { toggle() {} } } };
const window = { addEventListener() {}, innerWidth: 1280, innerHeight: 800, matchMedia: () => ({
  matches: false, addEventListener() {} }), localStorage: { getItem: () => null, setItem() {} } };
const api = new Function('document', 'window', 'fetch', 'location', 'setInterval', 'setTimeout',
  'getComputedStyle', 'ResizeObserver', 'localStorage', 'navigator',
  `${src}\n;return { poolOf, poolLine, kvTile };`)(
  document, window, () => new Promise(() => {}), { search: '', href: '' }, () => 0, () => 0,
  () => ({}), undefined, window.localStorage, { userAgent: 'node' });

const TF = 'qwen3.8-flash-next-tensorfold-nvfp4';
const VLLM = 'qwen3.8-flash-next-hibrid48';
const models = [
  { id: TF, kv_bytes_per_token: 15749, launch: { cluster: { engine: 'tensorfold', kv_cache_memory: null,
    pool_per_seat: true } } },
  { id: VLLM, kv_bytes_per_token: 16392, launch: { cluster: { engine: 'vllm', kv_cache_memory: 28e9,
    pool_per_seat: false } } },
];
const cluster = (choice) => ({ models, defaults: { kv_cache_memory: 28e9 },
  effective: { model_choice: choice, max_num_seqs: 16, max_model_len: 131072, kv_cache_memory: null } });

// The pool of a per-seat engine is its seats x its context — never the gateway's pin.
const tf = api.poolOf(cluster(TF), TF, 28e9, 16, 131072);
assert.strictEqual(tf.tokens, 16 * 131072);
assert.strictEqual(tf.bytes, 16 * 131072 * 15749);
const line = api.poolLine(tf.bytes, tf.tokens, 131072, 16);
assert.ok(line.startsWith('KV pool 33 GB · 2,097,152 tokens · fits 16 @ 131,072'), line);
assert.ok(!/seats >/.test(line), `every seat fits: ${line}`);
// Without the model's bytes a token: its tokens still, and no GB figure from another model.
const bare = api.poolOf({ models: [{ id: TF, launch: { cluster: { pool_per_seat: true } } }] }, TF, 28e9, 16, 131072);
assert.strictEqual(bare.tokens, 16 * 131072);
assert.strictEqual(bare.bytes, null);

// A shared pool is still its pin's, at the model's bytes a token.
const v = api.poolOf(cluster(VLLM), VLLM, 28e9, 16, 131072);
assert.strictEqual(v.bytes, 28e9);
assert.strictEqual(v.tokens, Math.floor(28e9 / 16392));

// The KV tile: the share of THAT pool, with its tokens.
const s = { clusters: { pair: cluster(TF) } };
const b = { kind: 'cluster', cluster: 'pair', live: true, throughput: {} };
const html = api.kvTile(s, b, { kv_pct: [10, 25] });
assert.ok(html.includes('0.52M / 2.10M'), html);

console.log('ok');
