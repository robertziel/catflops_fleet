/* A NODE THE LB REQUESTS (fleet control `local`) on the page: load, unload, select the model, set its parameters —
 * the operator's words — with what it needs on the card and who on the host applies it (vllm_lb/node_control.py).
 *
 *   1. Start/Stop is the hold button, its words the node's; the settings are EDITABLE: the model among the fleet's
 *      choices (the picker, one row per choice), seats, context and the GPU memory fraction (a field only this
 *      topology shows); the fit line is the fleet's numbers ("20.4 GiB weights + 4.3 GiB KV ≤ 28.8 GiB");
 *   2. an edit is listed and makes Apply & restart available; Start sends the selection on screen, Stop only stops;
 *   3. a refusal of the selection is shown VERBATIM, red, and holds back Apply and Start;
 *   4. no applier listening: Start/Stop is held back and the applier line says so, red;
 *   5. a failure is the applier's own words: the command's exit and stderr, the node's last log lines;
 *   6. finishing: a "cut them now" button sends the same wish with force;
 *   7. the pair renders none of it;
 *   8. the PREFILL BUDGET (vLLM's --max-num-batched-tokens) is editable like the others: "vLLM default" (nothing set —
 *      vLLM's own, never a number of ours) or a step up to the chunk the catalog measured the engine at (else the pair's
 *      list); a smaller chunk is priced at the measured overhead and the fit line says so;
 *   9. the GPU memory field offers 0.97, 0.98, 0.99 and 1.0 (operator, 2026-10-11) and the fit line prices each;
 *  10. the KV POOL may be pinned (vLLM --kv-cache-memory-bytes: "% of card" = not pinned, else a size, priced on
 *      the whole card) and KV OFFLOAD — a tier in the host's RAM (vLLM --kv-offloading-size) — is editable where the
 *      fleet bounds it on the box (Off / 8 / 16 / 32 / 48 GiB up to the bound), locked with the fleet's reason where it
 *      does not (no bound, or unified memory); the fit line carries "+ 32 GiB RAM tier", never in the card's sum.
 *
 *   node tests/cluster_local.node.js   (tests/test_cluster_local_page.py runs it)
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

/** What the page's fetch answers: nothing, unless a check says otherwise (an estimate, asked as a field is picked). */
let fetchImpl = () => new Promise(() => {});
const api = new Function('document', 'window', 'fetch', 'location', 'setInterval', 'setTimeout',
  'getComputedStyle', 'ResizeObserver', 'localStorage', 'navigator',
  `${src}\n;return { renderCluster, localBody, localChanges, askLocalFit, toggleLocal };`)(
  document, window, (...a) => fetchImpl(...a), { search: '', href: '' }, () => 0, () => 0,
  () => ({}), undefined, window.localStorage, { userAgent: 'node' });

const TWIN = 'qwen3.8-27b-nvidia-nvfp4@uncensored', ORIGINAL = 'qwen3.8-27b-nvidia-nvfp4';
const ROW = { id: ORIGINAL, label: 'Qwen3.8 27B · dense NVFP4', served_name: 'Qwen/Qwen3.8-27B', size_gib: 20.4,
  limits: { max_model_len: 262144, max_num_seqs: [1, 48] }, launch: { cluster: { engine: 'sglang', speculative_method: 'mtp', num_speculative_tokens: 4 } },
  uncensored: { path: 'Qwen3.8-27B-NVFP4_uncensored', default: true, calibration: 'c' }, unsupported: '' };
const choice = (ref, uncensored, checkpoint) => ({ ref, model_choice: ORIGINAL, uncensored, label: ROW.label, checkpoint,
  size_gib: 20.4, limits: { max_model_len: 262144, max_num_seqs: [1, 48] }, overhead_chunk: 2048, engine: 'vllm' });
const CHOICES = [choice(TWIN, true, 'Qwen3.8-27B-NVFP4_uncensored'), choice(ORIGINAL, false, 'Qwen3.8-27B-NVFP4')];
const FIT = { fit: { weights_gib: 20.4, overhead_gib: 3.4, kv_gib: 4.3, why: 'measured', budget_gib: 28.8 }, refused: '' };
const APPLIER = { listening: true, heartbeat_age_s: 12, host: 'node-a-host', applying: null,
  applied: { id: 3, at: 1, state: 'up', force: false, launch: {}, done_at: 2, exit: 0, stderr: '', refused: '' },
  container: { status: 'running', exit_code: 0, restarts: 0 }, error: '', log_tail: null };
const DESIRED = { mode: 'cluster', id: 3, force: false, model: TWIN, model_choice: ORIGINAL, uncensored: true,
  max_model_len: 131072, max_num_seqs: 4, gpu_memory_utilization: 0.9, max_num_batched_tokens: null,
  kv_cache_memory: null, kv_offload_ram_gib: null };
const TOPO = { tp: 1, boxes: ['node-a'], control: 'local', why: '', steps: ['load', 'warm up', 'serve'] };
const clusterOf = (over) => Object.assign({ configured: true, topology: TOPO, state: 'running', held: true,
  detail: 'TP=1 up on node-a (intent #3 (up))', desired: DESIRED, effective: { model_choice: ORIGINAL, uncensored: true,
    max_num_seqs: 4, max_model_len: 131072, num_speculative_tokens: 0, engine: 'vllm', served_name: 'Qwen/Qwen3.8-27B',
    gpu_memory_utilization: 0.9, kv_cache_dtype: 'fp8_e4m3', prefix_caching: true },
  boxes: { 'node-a': { label: 'node-a-gpu', rank: { present: true, running: true }, hold: { blockers: [] } } },
  loaded_model: { served_name: 'Qwen/Qwen3.8-27B', spec: ORIGINAL, label: ROW.label, max_model_len: 131072 },
  models: [], limits: { max_model_len: 262144, max_num_seqs: [1, 48] }, since: 1791741000, up_since: 1791741000,
  declared: null, applier: APPLIER, choices: CHOICES, fit: FIT, loading: {}, store_error: '',
  ram_tier: { max_gib: 32, refused: '' }, engines: { vllm: ENGINE_DECLS['vllm/chat'] } }, over || {});
const node = { kind: 'cluster', url: 'http://node-a:8000', live: true, running: 0, waiting: 0, inflight: 0,
  depth_known: true, served_models: ['Qwen/Qwen3.8-27B'], max_model_len: 131072, history: {},
  throughput: { source: 'metrics', generated: { tok_s: 0 }, prompt: {}, engine_totals: {},
    kv_pool: { tokens: 168937, bytes: null, blocks: 116, block_tokens: 1568 } } };
const statusOf = (cluster) => ({ now: 1791742000, cluster, backends: [node], leases: [], hardware: {},
  broker: { enabled: false }, config: {}, models: { cluster: [ROW], classic: [] } });
const html = (id) => byId(id).innerHTML;
const esc = (t) => String(t).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;')
  .replace(/'/g, '&#39;');
let checks = 0;

// 1. Running: Stop server, editable fields, the model picker over the choices, the GPU memory field, the fit line.
{
  const s = statusOf(clusterOf());
  api.renderCluster(s);
  assert.strictEqual(byId('cluster-toggle').hidden, false);
  assert.strictEqual(byId('cluster-toggle').dataset.label, 'Stop server');
  assert.strictEqual(byId('cluster-toggle').disabled, false);
  assert.strictEqual(byId('cluster-settings').hidden, false);
  for (const id of ['cluster-seqs', 'cluster-context', 'cluster-util', 'cluster-batched']) {
    assert.strictEqual(byId(id).disabled, false, `${id} is editable`);
  }
  assert.strictEqual(byId('cluster-util-row').hidden, false, 'the GPU memory field shows for a local node');
  assert.ok(html('cluster-util').includes('value="0.9" selected'), html('cluster-util'));
  assert.strictEqual(byId('cluster-seqs').value, '4');
  assert.strictEqual(byId('cluster-model').value, TWIN);
  assert.strictEqual(byId('cluster-model-button').disabled, false, 'the model is selectable');
  assert.ok(html('cluster-model-list').includes(`data-value="${ORIGINAL}"`) && html('cluster-model-list').includes(`data-value="${TWIN}"`),
    'one row per fleet choice');
  assert.ok(html('cluster-model-button').includes('vLLM') && !html('cluster-model-button').includes('SGLang'), 'the node\'s engine');
  assert.strictEqual(byId('cluster-fit').textContent, '20.4 GiB weights + 3.4 GiB engine + 4.3 GiB KV ≤ 28.8 GiB');
  assert.strictEqual(byId('cluster-fit').hidden, false);
  assert.ok(html('cluster-applier').includes('node-a-host') && html('cluster-applier').includes('12 s'), html('cluster-applier'));
  assert.strictEqual(byId('cluster-apply').hidden, false);
  assert.strictEqual(byId('cluster-apply').textContent, 'Apply & restart');
  assert.strictEqual(byId('cluster-apply').disabled, true, 'nothing changed');
  checks += 18;
}

// 2. An edit is listed; Start sends the selection on screen, Stop only stops, Apply only what changed.
{
  const c = clusterOf();
  byId('cluster-seqs').value = '2';
  byId('cluster-seqs').dataset.dirty = '1';
  byId('cluster-model').value = ORIGINAL;
  byId('cluster-model').dataset.dirty = '1';
  api.renderCluster(statusOf(c));
  assert.deepStrictEqual(api.localChanges(c).map((ch) => [ch.key, ch.from, ch.to]),
    [['model', TWIN, ORIGINAL], ['max_num_seqs', 4, 2]]);
  assert.ok(byId('cluster-changes').innerHTML.includes('seats 4 → 2'), byId('cluster-changes').innerHTML);
  assert.strictEqual(byId('cluster-apply').disabled, false);
  assert.deepStrictEqual(api.localBody(c, 'apply'), { enabled: true, model: ORIGINAL, max_num_seqs: 2 });
  assert.deepStrictEqual(api.localBody(c, 'stop'), { enabled: false });
  assert.deepStrictEqual(api.localBody(clusterOf({ desired: Object.assign({}, DESIRED, { mode: 'off' }) }), 'start'),
    { enabled: true, model: ORIGINAL, max_num_seqs: 2, max_model_len: 131072, gpu_memory_utilization: 0.9,
      max_num_batched_tokens: null, kv_cache_memory: null, kv_offload_ram_gib: null });
  ['cluster-seqs', 'cluster-model'].forEach((id) => { delete byId(id).dataset.dirty; });
  checks += 6;
}

// 3. A refusal of the selection, verbatim and red; Apply and Start held back.
{
  const words = 'the intent for clusters.node.launch: qwen3.8-27b-nvidia-nvfp4@uncensored: 20.4 GiB of weights > 0.50 × 32 GiB on rtx5090';
  const c = clusterOf({ desired: Object.assign({}, DESIRED, { mode: 'off' }), state: 'off', held: false,
    fit: { fit: { weights_gib: 20.4, overhead_gib: 3.4, kv_gib: 4.3, why: 'm', budget_gib: 16 }, refused: words } });
  api.renderCluster(statusOf(c));
  assert.strictEqual(byId('cluster-fit').textContent, words);
  assert.ok(byId('cluster-fit').classList.contains('bad'));
  assert.strictEqual(byId('cluster-toggle').dataset.label, 'Start server');
  assert.strictEqual(byId('cluster-toggle').disabled, true, 'Start is held back');
  assert.strictEqual(byId('cluster-caption').textContent, words);
  checks += 5;
}

// 4. No applier listening.
{
  const gone = 'no applier listening: its last heartbeat was 400 s ago';
  api.renderCluster(statusOf(clusterOf({ applier: Object.assign({}, APPLIER, { listening: false, heartbeat_age_s: 400, error: gone }) })));
  assert.strictEqual(byId('cluster-toggle').disabled, true);
  assert.strictEqual(byId('cluster-caption').textContent, gone);
  assert.ok(html('cluster-applier').includes(gone) && byId('cluster-applier').classList.contains('bad'));
  checks += 3;
}

// 5. A failure in the applier's words, the node's last log lines under it.
{
  const failed = Object.assign({}, APPLIER, {
    applied: Object.assign({}, APPLIER.applied, { id: 4, exit: 1, stderr: 'pulling\nError response from daemon: could not select device driver' }),
    container: { status: 'restarting', exit_code: 1, restarts: 6 },
    log_tail: ['torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB'] });
  api.renderCluster(statusOf(clusterOf({ state: 'failed', held: false, applier: failed })));
  const a = html('cluster-applier');
  for (const want of ['#4', 'exit 1', 'could not select device driver', 'restarting', '6 restarts', 'CUDA out of memory']) {
    assert.ok(a.includes(want), `applier line carries ${want}: ${a}`);
  }
  checks += 6;
}

// 5b. A crash loop after a clean apply (exit 0): `running` between attempts, its restarts grown — red, counted; the
//     strip fails at load, never at the pair's reserve with "not reserved".
{
  const looping = Object.assign({}, APPLIER, {
    applied: Object.assign({}, APPLIER.applied, { id: 3, exit: 0, restarts_at_apply: 0 }),
    container: { status: 'running', exit_code: 0, restarts: 2 },
    log_tail: ["ValueError: To serve at least one request with the model's max seq len (262144)"] });
  api.renderCluster(statusOf(clusterOf({ state: 'failed', held: false, applier: looping,
    detail: 'intent #3 (up): the node container restarted 2 time(s): ValueError: To serve …' })));
  const a = html('cluster-applier');
  assert.ok(byId('cluster-applier').classList.contains('bad') && a.includes('restarted 2'), `crash loop is red: ${a}`);
  const steps = html('cluster-state');
  assert.ok(/class="step failed"><i><\/i><span>load</.test(steps), `failed at load: ${steps.slice(0, 400)}`);
  assert.ok(!steps.includes('not reserved'), 'no pair reservation on a one-host node');
  // Not answering: the model line is the wish's, with no drafter from the catalog row's PAIR profile.
  api.renderCluster(statusOf(clusterOf({ state: 'failed', held: false, applier: looping, effective: null,
    loaded_model: null })));
  assert.ok(!/MTP|SGLang/.test(html('cluster-model-line')) && html('cluster-model-line').includes('4 seats'),
    html('cluster-model-line'));
  checks += 4;
}

// 5c. RECOVERED: a restart since the apply, but the node answers now (`running`) — said, never red (2026-10-24: one
//     CUDA OOM at load on a busy unified box, then served for hours under a red line).
{
  const recovered = Object.assign({}, APPLIER, {
    applied: Object.assign({}, APPLIER.applied, { id: 1, exit: 0, restarts_at_apply: 0 }),
    container: { status: 'running', exit_code: 0, restarts: 1 } });
  api.renderCluster(statusOf(clusterOf({ state: 'running', held: true, applier: recovered })));
  const a = html('cluster-applier');
  assert.ok(!byId('cluster-applier').classList.contains('bad'), `a recovered node is not red: ${a}`);
  assert.ok(a.includes('recovered after 1 restart'), `the restart is still said: ${a}`);
  checks += 2;
}

// 5d. STOPPED ON PURPOSE: the down intent applied, the container exited 0 — what was asked, never red.
{
  const stopped = Object.assign({}, APPLIER, {
    applied: Object.assign({}, APPLIER.applied, { id: 4, state: 'down', exit: 0, restarts_at_apply: 0 }),
    container: { status: 'exited', exit_code: 0, restarts: 0 } });
  api.renderCluster(statusOf(clusterOf({ state: 'off', held: false, applier: stopped })));
  const a = html('cluster-applier');
  assert.ok(!byId('cluster-applier').classList.contains('bad'), `a stopped node is not red: ${a}`);
  assert.ok(a.includes('container stopped'), a);
  checks += 2;
}

// 5e. A RAM ALERT beside a running pair (operator 2026-10-24: "just estimate ram usage and show alert, but allow
//     proceeding"): said above Start, amber, and Start stays offered.
{
  const off = clusterOf({ name: 'speech', state: 'off', held: false, desired: Object.assign({}, DESIRED, { mode: 'classic' }) });
  const s = Object.assign(statusOf(off), { declared: [{ name: 'speech', alerts: [
    'box node-a is held by cluster pair: 9.0 GiB available, speech needs 8.41 GiB: it would leave 0.6 GiB, under the 12 GiB floor'] }] });
  api.renderCluster(s);
  const al = byId('cluster-alert');
  assert.ok(!al.hidden && al.textContent.includes('0.6 GiB') && al.textContent.includes('Start proceeds'), al.textContent);
  assert.ok(!byId('cluster-toggle').disabled, 'Start is still offered');
  api.renderCluster(statusOf(clusterOf({ name: 'speech' })));
  assert.ok(byId('cluster-alert').hidden, 'no alert, no line');
  checks += 3;
}

// 6. Finishing: the cut-now button.
{
  api.renderCluster(statusOf(clusterOf({ state: 'finishing', held: false,
    detail: 'finishing 3 request(s) before the node is restarted — force cuts them' })));
  assert.strictEqual(byId('cluster-force').hidden, false);
  checks += 1;
}

// 7. The pair: none of it.
{
  const PAIR = { tp: 2, boxes: ['pair-a', 'pair-b'], control: 'ssh', why: '', steps: ['free', 'load', 'warm up', 'serve'] };
  const rank = { present: true, running: true, gen: 189 };
  const pairBox = (label) => ({ label, reservation: null, running_jobs: 0, rank, mem_available_gib: 20, hold: { blockers: [] } });
  api.renderCluster(statusOf({ configured: true, topology: PAIR, state: 'running', held: true, detail: 'serving',
    desired: { mode: 'cluster', model_choice: ORIGINAL, max_num_seqs: 8 },
    effective: { generation: 189, model_choice: ORIGINAL, max_num_seqs: 8, max_model_len: 262144 },
    defaults: {}, setups: {}, boxes: { 'pair-a': pairBox('a'), 'pair-b': pairBox('b') }, loaded_model: null,
    models: [ROW], limits: {}, since: 1791741000, up_since: 1791741000, declared: null, applier: null }));
  for (const id of ['cluster-util-row', 'cluster-fit', 'cluster-applier', 'cluster-force']) {
    assert.strictEqual(byId(id).hidden, true, `${id} is a local node's only`);
  }
  assert.strictEqual(byId('cluster-toggle').dataset.label, 'Stop server');
  checks += 5;
}

// 8. The prefill budget: vLLM's own default, or a step up to the chunk the engine was measured at; priced on the card.
(async () => {
  const asked = [];
  const answer = (body) => (url) => { asked.push(url); return Promise.resolve({ ok: true, json: () => Promise.resolve(body) }); };
  {
    api.renderCluster(statusOf(clusterOf()));
    const b = byId('cluster-batched');
    assert.strictEqual(b.disabled, false, 'the prefill budget is editable');
    assert.ok(html('cluster-batched').includes('<option value="" selected>vLLM default</option>'), html('cluster-batched'));
    assert.strictEqual(b.value, '', 'nothing set: vLLM\'s own default');
    for (const v of [512, 1024, 2048]) assert.ok(html('cluster-batched').includes(`value="${v}"`), `offers ${v}`);
    assert.ok(!html('cluster-batched').includes('value="4096"'), 'nothing past the chunk the overhead was measured at');
    assert.strictEqual(byId('cluster-batched-note').innerHTML, 'intent #3', 'provenance, like the others');
    checks += 7;
  }
  {
    const c = clusterOf();
    byId('cluster-batched').value = '1024';
    byId('cluster-batched').dataset.dirty = '1';
    api.renderCluster(statusOf(c));
    assert.deepStrictEqual(api.localChanges(c).map((ch) => [ch.key, ch.from, ch.to]), [['max_num_batched_tokens', null, 1024]]);
    assert.ok(byId('cluster-changes').innerHTML.includes('prefill budget vLLM default → 1,024'), byId('cluster-changes').innerHTML);
    assert.deepStrictEqual(api.localBody(c, 'apply'), { enabled: true, max_num_batched_tokens: 1024 });
    const note = 'engine overhead as measured at a 2,048-token chunk: the 1,024-token chunk\'s smaller activation is not credited';
    fetchImpl = answer({ fit: { weights_gib: 20.4, overhead_gib: 3.559, kv_gib: 4.3, why: '', budget_gib: 28.8,
      overhead_chunk: 2048, note }, refused: '' });
    await api.askLocalFit(c);
    assert.ok(asked[asked.length - 1].includes('max_num_batched_tokens=1024'), asked[asked.length - 1]);
    api.renderCluster(statusOf(c));
    assert.strictEqual(byId('cluster-fit').textContent,
      '20.4 GiB weights + 3.6 GiB engine (measured @ 2,048 chunk) + 4.3 GiB KV ≤ 28.8 GiB');
    assert.strictEqual(byId('cluster-fit').title, note, 'the words, on the line');
    delete byId('cluster-batched').dataset.dirty;
    checks += 6;
  }
  {
    // Set, then given back to vLLM: sent as null, asked blank — vLLM's own default, never a number of ours.
    const c = clusterOf({ desired: Object.assign({}, DESIRED, { max_num_batched_tokens: 1024 }) });
    byId('cluster-batched').value = '';
    byId('cluster-batched').dataset.dirty = '1';
    api.renderCluster(statusOf(c));
    assert.deepStrictEqual(api.localBody(c, 'apply'), { enabled: true, max_num_batched_tokens: null });
    assert.ok(byId('cluster-changes').innerHTML.includes('prefill budget 1,024 → vLLM default'), byId('cluster-changes').innerHTML);
    fetchImpl = answer({ fit: FIT.fit, refused: '' });
    await api.askLocalFit(c);
    assert.ok(/max_num_batched_tokens=(&|$)/.test(asked[asked.length - 1]), asked[asked.length - 1]);
    delete byId('cluster-batched').dataset.dirty;
    checks += 3;
  }
  {
    // A row whose overhead names no measured chunk: the pair's list, which the fleet refuses in its words if picked.
    const bare = CHOICES.map((ch) => Object.assign({}, ch, { overhead_chunk: null }));
    api.renderCluster(statusOf(clusterOf({ choices: bare })));
    for (const v of [2048, 4096, 65536]) assert.ok(html('cluster-batched').includes(`value="${v}"`), `the pair's ${v}`);
    assert.ok(!html('cluster-batched').includes('value="512"'), 'the pair\'s list');
    checks += 4;
  }
  // 9. GPU memory 0.97, 0.98, 0.99 and 1.0 (operator, 2026-10-11): offered, and each priced as picked (on the card's
  //    32 GiB, as the fleet's fit check prices it).
  for (const [util, text, budget] of [['0.97', '0.97', 31.04], ['0.98', '0.98', 31.36], ['0.99', '0.99', 31.68],
    ['1', '1.00', 32]]) {
    const c = clusterOf();
    byId('cluster-util').value = '0.9';
    api.renderCluster(statusOf(c));
    assert.ok(html('cluster-util').includes(`<option value="${util}">${text}</option>`), `offers ${text}: ${html('cluster-util')}`);
    byId('cluster-util').value = util;
    byId('cluster-util').dataset.dirty = '1';
    fetchImpl = answer({ fit: { weights_gib: 20.4, overhead_gib: 3.559, kv_gib: 4.3, why: '', budget_gib: budget,
      overhead_chunk: 2048, note: '' }, refused: '' });
    await api.askLocalFit(c);
    assert.ok(new RegExp(`gpu_memory_utilization=${util.replace('.', '\\.')}(&|$)`).test(asked[asked.length - 1]),
      asked[asked.length - 1]);
    api.renderCluster(statusOf(c));
    assert.strictEqual(byId('cluster-fit').textContent,
      `20.4 GiB weights + 3.6 GiB engine + 4.3 GiB KV ≤ ${budget.toFixed(1)} GiB`);
    assert.deepStrictEqual(api.localBody(c, 'apply'), { enabled: true, gpu_memory_utilization: Number(util) });
    delete byId('cluster-util').dataset.dirty;
    checks += 4;
  }
  // 10. The KV pool pinned and KV offload, a tier in the host's RAM.
  {
    api.renderCluster(statusOf(clusterOf()));
    const off = byId('cluster-offload');
    assert.strictEqual(off.disabled, false, 'KV offload is editable where the fleet bounds it');
    assert.ok(html('cluster-offload').includes('<option value="" selected>off</option>'), html('cluster-offload'));
    for (const v of [8, 16, 32]) assert.ok(html('cluster-offload').includes(`value="${v}">${v} GiB RAM<`), `offers ${v}`);
    assert.ok(!html('cluster-offload').includes('value="48"'), 'nothing past the fleet\'s bound');
    assert.strictEqual(byId('cluster-offload-note').innerHTML, 'intent #3 · fleet bound 32 GiB');
    const kv = byId('cluster-kv');
    assert.strictEqual(kv.disabled, false, 'the KV pool may be pinned');
    assert.ok(html('cluster-kv').includes('<option value="" selected>% of card</option>'), html('cluster-kv'));
    assert.ok(byId('cluster-kv-note').innerHTML.startsWith('intent #3'), byId('cluster-kv-note').innerHTML);
    checks += 8;
  }
  {
    // The relaunch: the card's whole KV and a 32 GiB tier — listed, sent, priced on the whole card, the tier beside it.
    const c = clusterOf({ desired: Object.assign({}, DESIRED, { kv_cache_memory: 8501810176 }) });
    api.renderCluster(statusOf(c));
    assert.ok(html('cluster-kv').includes('<option value="8501810176" selected>7.92 GiB</option>'), html('cluster-kv'));
    byId('cluster-offload').value = '32';
    byId('cluster-offload').dataset.dirty = '1';
    api.renderCluster(statusOf(c));
    assert.deepStrictEqual(api.localChanges(c).map((ch) => [ch.key, ch.from, ch.to]), [['kv_offload_ram_gib', null, 32]]);
    assert.ok(byId('cluster-changes').innerHTML.includes('KV offload off → 32 GiB RAM'), byId('cluster-changes').innerHTML);
    assert.deepStrictEqual(api.localBody(c, 'apply'), { enabled: true, kv_offload_ram_gib: 32 });
    fetchImpl = answer({ fit: { weights_gib: 20.4, overhead_gib: 3.559, kv_gib: 4.3, pool_gib: 7.918, ram_gib: 32,
      why: '', budget_gib: 32, overhead_chunk: 2048, note: '' }, refused: '' });
    await api.askLocalFit(c);
    const q = asked[asked.length - 1];
    assert.ok(q.includes('kv_offload_ram_gib=32') && q.includes('kv_cache_memory=8501810176'), q);
    api.renderCluster(statusOf(c));
    assert.strictEqual(byId('cluster-fit').textContent,
      '20.4 GiB weights + 3.6 GiB engine + 7.9 GiB KV pool (pinned) ≤ 32.0 GiB whole card · + 32 GiB RAM tier');
    // Given back: off is null, the pool unpinned asked blank.
    byId('cluster-offload').value = '';
    byId('cluster-kv').value = '';
    byId('cluster-kv').dataset.dirty = '1';
    api.renderCluster(statusOf(c));
    assert.deepStrictEqual(api.localBody(c, 'apply'), { enabled: true, kv_cache_memory: null });
    assert.ok(byId('cluster-changes').innerHTML.includes('KV pool 7.92 GiB → % of card'), byId('cluster-changes').innerHTML);
    ['cluster-offload', 'cluster-kv'].forEach((id) => { delete byId(id).dataset.dirty; });
    checks += 7;
  }
  {
    // No bound on the box, or unified memory: locked at off, the fleet's reason on the line.
    const why = "unified memory: the host RAM is the GPU's own pool";
    api.renderCluster(statusOf(clusterOf({ ram_tier: { max_gib: null, refused: why } })));
    assert.strictEqual(byId('cluster-offload').disabled, true);
    assert.ok(html('cluster-offload').includes('>off<'), html('cluster-offload'));
    assert.strictEqual(byId('cluster-offload-note').innerHTML, esc(why));
    checks += 3;
  }
  {
    // 11. DOES EACH CHOICE START (operator 2026-10-12: the chooser's start bars were empty on site-b): each fleet choice
    //     carries its own record off the boot log — `start_rates.rows` under the row's key, `<id>` / `<id>@uncensored`,
    //     the same key the pair's picker reads — drawn beside it as the pair's rows draw theirs.
    const days = ['2026-10-06', '2026-10-07', '2026-10-08', '2026-10-09', '2026-10-10', '2026-10-11', '2026-10-12'];
    const rates = { tz: 'UTC', days, unread: 0, malformed: 0, error: '',
      rows: { [TWIN]: { ok: 1, n: 2, stopped: 0, refused: 0, days: [null, null, null, null, null, [0, 1], [1, 1]] } } };
    api.renderCluster(Object.assign(statusOf(clusterOf()), { start_rates: rates }));
    const list = html('cluster-model-list');
    const rowOf = (ref) => list.split('data-value=').find((r) => r.startsWith(`"${ref}"`)) || '';
    assert.ok(rowOf(TWIN).includes('mp-starts warn') && rowOf(TWIN).includes('1/2 · 50 %'), rowOf(TWIN));
    assert.ok(rowOf(TWIN).includes('data-days="-,-,-,-,-,0/1,1/1"'), 'its days, as bars');
    assert.ok(rowOf(ORIGINAL).includes('no starts in 7 d'), `a choice never launched says so: ${rowOf(ORIGINAL)}`);
    checks += 3;
  }
  // 3b. Declared `run_at_boot: false` and never started (live 2026-10-17, the box-c gui): the view's desired mode is
  //     the fleet's run_at_boot — off (topology.local_desired; tests/test_cluster_local.py) — so the button is Start and
  //     holding it POSTS a start, never a Stop of a node that is not running.
  {
    const c = clusterOf({ desired: Object.assign({}, DESIRED, { mode: 'off', id: null }), state: 'off', held: false,
      effective: null, loaded_model: null, detail: '' });
    api.renderCluster(statusOf(c));
    assert.strictEqual(byId('cluster-toggle').dataset.label, 'Start server');
    assert.strictEqual(byId('cluster-toggle').dataset.kind, 'start');
    const posted = [];
    // the POST is recorded; the re-poll after it is never answered (only what was posted is checked)
    fetchImpl = (url, init) => {
      if (init && init.method === 'POST') { posted.push(JSON.parse(init.body)); return Promise.resolve({ ok: true }); }
      return new Promise(() => {});
    };
    const settle = () => new Promise((r) => setImmediate(r));
    api.toggleLocal(c);
    await settle();
    assert.strictEqual(posted.length, 1);
    assert.strictEqual(posted[0].enabled, true, `holding it starts the node: ${JSON.stringify(posted[0])}`);
    // and only a node whose desired mode is on is offered — and posts — a Stop
    posted.length = 0;
    api.toggleLocal(clusterOf());
    await settle();
    assert.deepStrictEqual(posted, [{ enabled: false }]);
    fetchImpl = () => new Promise(() => {});
    checks += 5;
  }
  console.log(`ok — ${checks} checks`);
})().catch((e) => { console.error(e); process.exit(1); });
