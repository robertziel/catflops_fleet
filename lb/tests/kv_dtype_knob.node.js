/* THE KV TYPE (kv_cache_dtype) ON THE PANEL — operator 2026-10-22, at the KV pool row: "can't we add nvfp4 and fp8
 * setting for KV here?". One row under the KV pool, for both topologies:
 *
 *   PAIR   the first option is the profile's own (null on the wire), labelled with what it resolves to ("profile
 *          (fp8_e4m3)"); then each dtype the LB publishes for the PICKER model on the pair's cards
 *          (`cluster.kv_cache_dtypes[model]`). A refused one is listed DISABLED with the switch's words as its title.
 *          A change is listed and sent like the other knobs (Apply & restart / Save; null for the default), a stored
 *          one runs at the next start ("running fp8_e4m3 · pending nvfp4"), and a refused pick is refused by name.
 *   LOCAL  the same row for a node the LB requests: "engine auto" (null: vLLM's own) then the node image's dtypes on
 *          its card (`cluster.kv_cache_dtypes[ref]`); it travels in the intent's launch like the seats.
 *   OLDER  an LB that publishes no choices gets no row and sends no field (it would ignore or refuse it).
 *
 * Fixtures are the library's own entries (vllm_lb/kv_cache_dtypes.json via kv_dtype.library().choices): the SGLang pair
 * image accepts nvfp4/fp4_mx_block16 on GB10, the Flash-Next vLLM image refuses nvfp4 on sm_121, the 5090 image accepts
 * nvfp4 on sm_120, the ROCm image refuses nvfp4. Sources are shortened; refusals are verbatim.
 *
 *   node tests/kv_dtype_knob.node.js   (tests/test_kv_dtype_knob_page.py runs it when node exists)
 */
'use strict';
const ENGINE_DECLS = JSON.parse(require('fs').readFileSync(require('path').join(__dirname, 'fixtures', 'engine-declarations.json'), 'utf8'));
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const src = fs.readFileSync(path.join(__dirname, '..', 'vllm_lb', 'static', 'gateway.js'), 'utf8');
const els = new Map();
const el = () => ({ addEventListener() {}, removeAttribute() {}, setAttribute() {}, append() {},
  classList: { _s: new Set(), toggle(c, on) { if (on === undefined ? !this._s.has(c) : on) this._s.add(c); else this._s.delete(c); },
    add(c) { this._s.add(c); }, remove(c) { this._s.delete(c); }, contains(c) { return this._s.has(c); } },
  style: {}, hidden: false, innerHTML: '', textContent: '', value: '', dataset: {}, disabled: false, open: false,
  title: '', querySelectorAll: () => [], querySelector: () => null, focus() {}, observe() {}, closest: () => null,
  contains: () => false, getBoundingClientRect: () => ({}) });
const byId = (id) => { if (!els.has(id)) els.set(id, el()); return els.get(id); };
const document = { getElementById: byId, createElementNS: el, createElement: el, addEventListener() {},
  querySelectorAll: () => [], querySelector: () => null, activeElement: null,
  body: { style: {}, classList: { toggle() {} } }, documentElement: { style: {}, classList: { toggle() {} } } };
const window = { addEventListener() {}, innerWidth: 1280, innerHeight: 800, matchMedia: () => ({
  matches: false, addEventListener() {} }), localStorage: { getItem: () => null, setItem() {} } };
const api = new Function('document', 'window', 'fetch', 'location', 'setInterval', 'setTimeout',
  'getComputedStyle', 'ResizeObserver', 'localStorage', 'navigator',
  `${src}\n;return { renderKnobs, clusterChanges, changeList, fieldNote, selectionProblem, takesKnob, renderCluster,
    localChanges, localBody };`)(
  document, window, () => new Promise(() => {}), { search: '', href: '' }, () => 0, () => 0,
  () => ({}), undefined, window.localStorage, { userAgent: 'node' });
const html = (id) => byId(id).innerHTML;
const pick = (id, v) => { byId(id).value = v; byId(id).dataset.dirty = '1'; };
const reset = () => els.clear();
let checks = 0;

// ── the library's entries ─────────────────────────────────────────────────────────────────────────────────────
const ok = (dtype, source, ab) => ({ dtype, refused: '', source, attention_backends: ab || null });
const no = (dtype, refused) => ({ dtype, refused, source: '', attention_backends: null });
const SGL_IMG = 'myllmbox/sglang-27b-tp2:708f51e-fleet.3';
/** The SGLang pair image on GB10 (sm_121 x2). */
const SGLANG_GB10 = [
  ok('bfloat16', 'fields/model.py:213-214 → torch.bfloat16 (as -fleet.2)'),
  no('float16', `${SGL_IMG} (SGLang 708f51e44bc6 (2026-09-09), image b40df1481b02) does not accept kv_cache_dtype float16: it takes bfloat16, fp8_e4m3, fp8_e5m2, nvfp4, fp4_mx_block16 (md5-identical to -fleet.2 …)`),
  ok('fp8_e4m3', 'fields/model.py:211 → torch.float8_e4m3fn (as -fleet.2)'),
  ok('fp8_e5m2', 'fields/model.py:210 → torch.float8_e5m2 (as -fleet.2); not measured on this fleet'),
  ok('nvfp4', "fields/model.py:215 'nvfp4' → torch.float4_e2m1fn_x2 … NOT MEASURED ON THIS HARDWARE",
    { prefill: 'flashinfer', decode: 'trtllm_mha', draft: 'trtllm_mha' }),
  ok('fp4_mx_block16', "fields/model.py:216 'fp4_mx_block16' → torch.float4_e2m1fn_x2 … NOT MEASURED ON THIS HARDWARE",
    { prefill: 'triton', decode: 'triton' }),
];
/** The Flash-Next vLLM pair image on GB10: nvfp4 refused by the engine's own rule. */
const NEXT_NVFP4 = "nvfp4 on myllmbox/qwen38-flash-next-cluster-vllm:v5 runs only on a card whose arch is sm_10[0-9] (config/cache.py:55; v1/attention/backends/flashinfer.py:491-498 supports_kv_cache_dtype: nvfp4 only on device capability family 100 with trtllm-gen attention, and :765-776 raises '--kv-cache-dtype nvfp4 requires the SM100 trtllm-gen FlashInfer path' — so a 12.x card (GB10 sm_121) has no backend for it. Not measured on any card); this server's is sm_121";
const VLLM_GB10 = [
  ok('bfloat16', 'config/cache.py:42 CacheDType; what `auto` resolved to on the pair'),
  ok('float16', 'config/cache.py:41 CacheDType; not measured on this fleet'),
  ok('fp8_e4m3', 'config/cache.py:44 CacheDType; not measured with this image'),
  ok('fp8_e5m2', 'config/cache.py:45 CacheDType; not measured on this fleet'),
  no('nvfp4', NEXT_NVFP4),
  no('fp4_mx_block16', 'myllmbox/qwen38-flash-next-cluster-vllm:v5 (vLLM 0.29.0 (98dff2a81), image d9b79b5efa9f) does not accept kv_cache_dtype fp4_mx_block16: it takes bfloat16, float16, fp8_e4m3, fp8_e5m2, nvfp4 (…)'),
];
const FIVE = 'vllm/vllm-openai@sha256:514cd42702d29a7b8efe20b0a916523ac473ecebdec8d53b1a0d0f5439a48634';
/** The 5090 node's image on sm_120: nvfp4 taken. */
const VLLM_5090 = [
  ok('bfloat16', 'config/cache.py:42 CacheDType; not measured on this fleet'),
  ok('float16', 'config/cache.py:41 CacheDType; not measured on this fleet'),
  ok('fp8_e4m3', 'config/cache.py:44; what `auto` resolved to on the 5090 node'),
  ok('fp8_e5m2', 'config/cache.py:45 CacheDType; not measured on this fleet'),
  ok('nvfp4', 'config/cache.py:57; v1/attention/backends/flashinfer.py:110-111 …'),
  no('fp4_mx_block16', `${FIVE} (vLLM 0.31.1rc1.dev273 (2eb491e18), image 578a7e140507) does not accept kv_cache_dtype fp4_mx_block16: it takes bfloat16, float16, fp8_e4m3, fp8_e5m2, nvfp4 (…)`),
];
const ROCM_NVFP4 = 'nvfp4 on rocm/vllm@sha256:394194d36edcf9b36bcb563e143b21b80e64e7d04f33a447b448c0c0c00c04a8: this ROCm vLLM has no FP4 KV cache: config/cache.py:14-23 CacheDType ends at fp8_ds_mla';

// ── the PAIR ──────────────────────────────────────────────────────────────────────────────────────────────────
const SGLANG = 'qwen3.8-27b-sglang-dflash2';
const FLASH = 'qwen3.8-flash-next-hibrid48';
const MODELS = [
  { id: SGLANG, label: 'Qwen3.8 27B · SGLang DFlash2', limits: { max_model_len: 262144, max_num_seqs: [1, 48] },
    kv_bytes_per_token: 21502, calibrated: true,
    launch: { cluster: { engine: 'sglang', speculative: true, num_speculative_tokens: 4, kv_cache_memory: 73435823273,
      ple_replicate: null, refuses: {}, kv_cache_dtype: 'fp8_e4m3' } } },
  { id: FLASH, label: 'Qwen3.8 Flash-Next', limits: { max_model_len: 262144, max_num_seqs: [1, 48] },
    kv_bytes_per_token: 16392, calibrated: true,
    launch: { cluster: { engine: 'vllm', speculative: true, num_speculative_tokens: 4, kv_cache_memory: 28000000000,
      ple_replicate: true, refuses: {}, kv_cache_dtype: null } } },
];
const PAIR = {
  configured: true, state: 'running', held: true, models: MODELS,
  desired: { mode: 'cluster', generation: 214, model_choice: SGLANG, max_num_seqs: 32, max_model_len: 262144,
    max_num_batched_tokens: null, num_speculative_tokens: null, ple_replicate: null, kv_cache_memory: null,
    kv_cache_auto: false, kv_offload_gib: 0, long_prefill_token_threshold: null, kv_cache_dtype: null },
  limits: { max_num_seqs: [1, 48], max_model_len: [1024, 262144], max_num_batched_tokens: [2048, 65536],
    num_speculative_tokens: [0, 8], kv_cache_memory: [8000000000, 96000000000], kv_offload_gib: [0, 512],
    long_prefill_token_threshold: [0, 65536], ple_replicate: [0, 1] },
  defaults: { max_num_seqs: 32, max_model_len: 262144, max_num_batched_tokens: 8192, num_speculative_tokens: 4,
    kv_cache_memory: 73435823273, kv_offload_gib: 0, ple_replicate: null },
  effective: { model_choice: SGLANG, generation: 214, max_num_seqs: 32, max_num_batched_tokens: 8192,
    num_speculative_tokens: 4, ple_replicate: null, kv_cache_memory: 73435823273, kv_offload_gib: 0,
    max_model_len: 262144, long_prefill_token_threshold: 0, kv_cache_dtype: 'fp8_e4m3' },
  kv_cache_dtypes: { [SGLANG]: SGLANG_GB10, [FLASH]: VLLM_GB10 },
  boxes: {},
};
const pair = (desired, over) => Object.assign({}, PAIR, { desired: Object.assign({}, PAIR.desired, desired || {}) },
  over || {});

// 1. The row, at the profile's own: "profile (fp8_e4m3)" first, every dtype after it, a refused one DISABLED with the
//    image's words, an accepted one's source and attention backends in its title. Nothing to send.
{
  reset();
  byId('cluster-model').value = SGLANG;
  api.renderKnobs(PAIR, true, SGLANG);
  const h = html('cluster-dtype');
  assert.strictEqual(byId('cluster-dtype-row').hidden, false, 'the LB publishes choices: the row shows');
  assert.ok(h.startsWith('<option value="fp8_e4m3">profile (fp8_e4m3)</option>'), h);
  assert.strictEqual((h.match(/value="fp8_e4m3"/g) || []).length, 1, 'the default is not listed twice');
  assert.ok(/<option value="float16" disabled title="[^"]*does not accept kv_cache_dtype float16/.test(h), h);
  assert.ok(/<option value="nvfp4" title="[^"]*NOT MEASURED[^"]*prefill flashinfer, decode trtllm_mha, draft trtllm_mha">nvfp4</.test(h), h);
  assert.ok(/<option value="fp4_mx_block16" title="[^"]*prefill triton, decode triton">/.test(h), h);
  assert.strictEqual(byId('cluster-dtype').value, 'fp8_e4m3');
  assert.strictEqual(byId('cluster-dtype').disabled, false);
  assert.ok(!api.clusterChanges(PAIR).some((ch) => ch.key === 'kv_cache_dtype'), 'nothing changed');
  checks += 9;
}

// 2. Picked nvfp4: listed and sent as itself; back at the profile's own, sent as null.
{
  reset();
  byId('cluster-model').value = SGLANG;
  pick('cluster-dtype', 'nvfp4');
  api.renderKnobs(PAIR, true, SGLANG);
  const ch = api.clusterChanges(PAIR).find((x) => x.key === 'kv_cache_dtype');
  assert.ok(ch && ch.from === 'fp8_e4m3' && ch.to === 'nvfp4' && ch.send === 'nvfp4' && !ch.reset, JSON.stringify(ch));
  assert.strictEqual(api.changeList(PAIR, [ch]), 'KV type fp8_e4m3 → nvfp4');
  const p = api.selectionProblem(PAIR, [ch]);   // (the harness's seats field is empty: only the KV type is asked here)
  assert.ok(!p || p.key !== 'kv_cache_dtype', `nvfp4 runs on GB10 with this image: ${JSON.stringify(p)}`);
  const stored = pair({ kv_cache_dtype: 'nvfp4' });
  pick('cluster-dtype', 'fp8_e4m3');
  const back = api.clusterChanges(stored).find((x) => x.key === 'kv_cache_dtype');
  assert.ok(back && back.reset && back.send === null, JSON.stringify(back));
  assert.strictEqual(api.changeList(stored, [back]), 'KV type nvfp4 → fp8_e4m3 (default)');
  checks += 5;
}

// 3. Stored nvfp4, the pair still runs fp8_e4m3: the field shows the stored pick, the note what runs and what is pending.
{
  reset();
  byId('cluster-model').value = SGLANG;
  const stored = pair({ kv_cache_dtype: 'nvfp4' });
  api.renderKnobs(stored, true, SGLANG);
  assert.strictEqual(byId('cluster-dtype').value, 'nvfp4');
  const note = api.fieldNote(stored, 'kv_cache_dtype', SGLANG, true);
  assert.ok(note.includes('running fp8_e4m3') && note.includes('pending nvfp4 after restart'), note);
  assert.ok(!api.clusterChanges(stored).some((x) => x.key === 'kv_cache_dtype'), 'stored is not a change');
  checks += 3;
}

// 4. Flash-Next (vLLM) in the picker on GB10: its default is the engine's auto, and nvfp4 is there, DISABLED, with the
//    engine's own rule — never hidden. A pick of it is refused by name before anything is sent.
{
  reset();
  byId('cluster-model').value = FLASH;
  api.renderKnobs(PAIR, true, FLASH);
  const h = html('cluster-dtype');
  assert.ok(h.startsWith('<option value="auto">profile (auto)</option>'), h);
  assert.ok(h.includes(`<option value="nvfp4" disabled title="${NEXT_NVFP4.replace(/'/g, '&#39;')}">nvfp4`), h);
  const ch = { key: 'kv_cache_dtype', id: 'cluster-dtype', name: 'KV type', from: 'auto', to: 'nvfp4', send: 'nvfp4', reset: false };
  const bad = api.selectionProblem(pair({ model_choice: FLASH }), [ch]);
  assert.ok(bad && bad.key === 'kv_cache_dtype' && bad.why.includes('runs only on a card whose arch is sm_10[0-9]'), JSON.stringify(bad));
  checks += 3;
}

// 5. An LB that publishes no choices (older, or `{}`): no row, nothing sent.
{
  for (const kv of [undefined, {}]) {
    reset();
    byId('cluster-model').value = SGLANG;
    const old = Object.assign({}, PAIR, { kv_cache_dtypes: kv });
    api.renderKnobs(old, true, SGLANG);
    assert.strictEqual(byId('cluster-dtype-row').hidden, true, `no choices (${JSON.stringify(kv)}): no row`);
    assert.strictEqual(api.takesKnob(old, 'kv_cache_dtype'), false);
    assert.ok(!api.clusterChanges(old).some((x) => x.key === 'kv_cache_dtype'));
    checks += 3;
  }
}

// ── a LOCAL node (the site-b RTX 5090) ───────────────────────────────────────────────────────────────────────
const TWIN = 'qwen3.8-27b-nvidia-nvfp4@uncensored', ORIGINAL = 'qwen3.8-27b-nvidia-nvfp4';
const ROW = { id: ORIGINAL, label: 'Qwen3.8 27B · dense NVFP4', served_name: 'Qwen/Qwen3.8-27B', size_gib: 20.4,
  limits: { max_model_len: 262144, max_num_seqs: [1, 48] }, launch: { cluster: { engine: 'vllm' } }, unsupported: '' };
const choice = (ref, uncensored) => ({ ref, model_choice: ORIGINAL, uncensored, label: ROW.label, checkpoint: ref,
  size_gib: 20.4, limits: { max_model_len: 262144, max_num_seqs: [1, 48] }, overhead_chunk: 2048, kv_pin_max: 8e9, engine: 'vllm' });
const DESIRED = { mode: 'cluster', id: 29, force: false, model: TWIN, model_choice: ORIGINAL, uncensored: true,
  max_model_len: 131072, max_num_seqs: 4, gpu_memory_utilization: 0.9, max_num_batched_tokens: null,
  kv_cache_memory: null, kv_offload_ram_gib: null, kv_cache_dtype: null };
const local = (over) => Object.assign({ configured: true,
  topology: { tp: 1, boxes: ['node-a'], control: 'local', why: '', steps: ['load', 'warm up', 'serve'] },
  state: 'running', held: true, detail: '', desired: DESIRED,
  effective: { model_choice: ORIGINAL, uncensored: true, max_num_seqs: 4, max_model_len: 131072, engine: 'vllm',
    gpu_memory_utilization: 0.9, kv_cache_dtype: 'fp8_e4m3' },
  boxes: { 'node-a': { label: 'node-a-gpu', rank: { present: true, running: true }, hold: { blockers: [] } } },
  loaded_model: null, models: [], limits: { max_model_len: 262144, max_num_seqs: [1, 48] }, since: 1, up_since: 1,
  declared: null, applier: { listening: true, heartbeat_age_s: 3, host: 'node-a-host', applying: null, applied: null,
    container: null, error: '', log_tail: null },
  choices: [choice(TWIN, true), choice(ORIGINAL, false)], fit: { fit: null, refused: '' }, loading: {},
  store_error: '', ram_tier: { max_gib: null, refused: 'no bound' },
  kv_cache_dtypes: { [TWIN]: VLLM_5090, [ORIGINAL]: VLLM_5090 }, engines: { vllm: ENGINE_DECLS['vllm/chat'] } }, over || {});
const statusOf = (cluster) => ({ now: 2, cluster, backends: [], leases: [], hardware: {}, broker: { enabled: false },
  config: {}, models: { cluster: [ROW], classic: [] } });

// 6. The 5090 node: "engine auto" first, nvfp4 OPEN on sm_120, fp4_mx_block16 disabled; the note says what the node runs.
{
  reset();
  api.renderCluster(statusOf(local()));
  const h = html('cluster-dtype');
  assert.strictEqual(byId('cluster-dtype-row').hidden, false);
  assert.ok(h.startsWith('<option value="">engine auto</option>'), h);
  assert.ok(/<option value="nvfp4" title="config\/cache.py:57[^"]*">nvfp4</.test(h), h);
  assert.ok(/<option value="fp4_mx_block16" disabled title="[^"]*does not accept kv_cache_dtype fp4_mx_block16/.test(h), h);
  assert.strictEqual(byId('cluster-dtype').value, '');
  assert.strictEqual(byId('cluster-dtype').disabled, false);
  assert.ok(byId('cluster-dtype-note').innerHTML.includes('node runs fp8_e4m3'), byId('cluster-dtype-note').innerHTML);
  assert.ok(!api.localChanges(local()).some((ch) => ch.key === 'kv_cache_dtype'));
  checks += 7;
}

// 7. Picked nvfp4: listed, and it travels in the intent's launch like the seats; a Start sends the selection.
{
  reset();
  const c = local();
  pick('cluster-dtype', 'nvfp4');
  api.renderCluster(statusOf(c));
  assert.deepStrictEqual(api.localChanges(c).map((ch) => [ch.key, ch.from, ch.to]), [['kv_cache_dtype', null, 'nvfp4']]);
  assert.ok(html('cluster-changes').includes('KV type engine auto → nvfp4'), html('cluster-changes'));
  assert.deepStrictEqual(api.localBody(c, 'apply'), { enabled: true, kv_cache_dtype: 'nvfp4' });
  assert.strictEqual(api.localBody(c, 'start').kv_cache_dtype, 'nvfp4');
  checks += 4;
}

// 8. The ROCm node: nvfp4 listed DISABLED with the image's reason. An LB without choices: no row, no field in a Start.
{
  reset();
  const rocm = VLLM_5090.map((x) => (x.dtype === 'nvfp4' ? no('nvfp4', ROCM_NVFP4) : x));
  api.renderCluster(statusOf(local({ kv_cache_dtypes: { [TWIN]: rocm, [ORIGINAL]: rocm } })));
  assert.ok(html('cluster-dtype').includes(`<option value="nvfp4" disabled title="${ROCM_NVFP4}">nvfp4`), html('cluster-dtype'));
  reset();
  const old = local({ kv_cache_dtypes: undefined, desired: Object.assign({}, DESIRED, { mode: 'off' }) });
  delete old.desired.kv_cache_dtype;
  api.renderCluster(statusOf(old));
  assert.strictEqual(byId('cluster-dtype-row').hidden, true);
  assert.ok(!('kv_cache_dtype' in api.localBody(old, 'start')), JSON.stringify(api.localBody(old, 'start')));
  checks += 3;
}

// 9. FP4 WITH A DRAFTER (generation 221, 2026-10-22): the LB publishes each dtype's refusal at depth > 0 beside its own
//    (`refused_with_drafter`). At the drafter's depth (the profile's 4) nvfp4 is DISABLED with the image's words and a
//    pick of it refused by name; at depth 0 it is open. A depth turned back on under a stored nvfp4 is refused too.
{
  const WITH = 'nvfp4 on myllmbox/sglang-27b-tp2:708f51e-fleet.3: refused with a drafter on this image: generations 214, 215 and 217 … 219 … — set num_speculative_tokens 0';
  const drafting = SGLANG_GB10.map((x) => Object.assign({}, x, {
    refused_with_drafter: /fp4/.test(x.dtype) ? WITH.replace(/^nvfp4/, x.dtype) : x.refused }));
  const fp4 = (desired) => pair(desired, { kv_cache_dtypes: { [SGLANG]: drafting, [FLASH]: VLLM_GB10 } });
  const ch = { key: 'kv_cache_dtype', id: 'cluster-dtype', name: 'KV type', from: 'fp8_e4m3', to: 'nvfp4', send: 'nvfp4', reset: false };
  reset();
  byId('cluster-model').value = SGLANG;
  api.renderKnobs(fp4(), true, SGLANG);                       // the depth at the profile's own: 4
  assert.ok(html('cluster-dtype').includes(`<option value="nvfp4" disabled title="${WITH}">nvfp4 · refused`), html('cluster-dtype'));
  assert.ok(/<option value="fp4_mx_block16" disabled /.test(html('cluster-dtype')), html('cluster-dtype'));
  const bad = api.selectionProblem(fp4(), [ch]);
  assert.ok(bad && bad.key === 'kv_cache_dtype' && bad.why === `KV type nvfp4: ${WITH}`, JSON.stringify(bad));
  reset();
  byId('cluster-model').value = SGLANG;
  pick('cluster-spec', '0');                                  // the drafter off: nvfp4 is open, and goes
  api.renderKnobs(fp4(), true, SGLANG);
  assert.ok(/<option value="nvfp4" title="[^"]*NOT MEASURED/.test(html('cluster-dtype')), html('cluster-dtype'));
  const p = api.selectionProblem(fp4(), [ch]);
  assert.ok(!p || p.key !== 'kv_cache_dtype', JSON.stringify(p));
  reset();
  byId('cluster-model').value = SGLANG;
  pick('cluster-spec', '4');                                  // stored nvfp4 at depth 0, the depth turned back on
  const stored = fp4({ kv_cache_dtype: 'nvfp4', num_speculative_tokens: 0 });
  api.renderKnobs(stored, true, SGLANG);
  const depth = api.clusterChanges(stored).filter((x) => x.key === 'num_speculative_tokens');   // (seats: empty here)
  assert.deepStrictEqual(depth.map((x) => [x.from, x.to]), [[0, 4]]);
  const back = api.selectionProblem(stored, depth);
  assert.ok(back && back.key === 'kv_cache_dtype' && back.why.includes('set num_speculative_tokens 0'), JSON.stringify(back));
  checks += 7;
}

console.log(`ok ${checks} checks`);
