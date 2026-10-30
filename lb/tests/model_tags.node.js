/* THE MODEL'S TAGS — one colour-coded family wherever a model is shown, checked where the page builds them.
 *
 * Operator, 2026-10-10, on the live panel: "UX UI is terrible, need to read text to know if uncensored, etc.
 * Would be better to add multicolor tags that will make it easier to read." So every place that names a model
 * — the picker's trigger and rows, the cluster card's model line, a server card's badge — wears the SAME tags,
 * from ONE facts function (`modelTagList`): the engine (vLLM and SGLang in two hues), UNCENSORED loud and filled
 * when the checkpoint shown is the twin, the drafter in its own hue with its depth ("NEXTN 4"), the build and the
 * size neutral, the figures as neutral chips, a target rate in the positive colour, "other setup" muted.
 *
 * WHICH CHECKPOINT THE MODEL LINE SHOWS is read, never guessed: the generation's launch record
 * (`cluster.effective.uncensored`) or, before one is recorded, the stored switch (`desired.uncensored`); an LB
 * that publishes neither gets no tag either way.
 *
 *   node tests/model_tags.node.js      (tests/test_model_tags_page.py runs this when node exists)
 */
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const STATIC = path.join(__dirname, '..', 'vllm_lb', 'static');
const src = fs.readFileSync(path.join(STATIC, 'gateway.js'), 'utf8');
const css = fs.readFileSync(path.join(STATIC, 'gateway.css'), 'utf8');

function load() {
  const byId = {};
  const el = () => ({ addEventListener() {}, removeAttribute() {}, setAttribute() {}, append() {},
    classList: { toggle() {}, add() {}, remove() {}, contains: () => false }, style: {}, hidden: false, innerHTML: '',
    textContent: '', value: '', dataset: {}, querySelectorAll: () => [], querySelector: () => null,
    focus() {}, observe() {}, closest: () => null, getBoundingClientRect: () => ({}), options: [], contains: () => false });
  const document = {
    getElementById: (id) => (byId[id] || (byId[id] = el())),
    createElementNS: el, createElement: el, addEventListener() {},
    querySelectorAll: () => [], querySelector: () => null, activeElement: null,
    body: { style: {}, classList: { toggle() {} } },
    documentElement: { style: {}, classList: { toggle() {} } } };
  const window = { addEventListener() {}, innerWidth: 1280, innerHeight: 800, matchMedia: () => ({
    matches: false, addEventListener() {} }), localStorage: { getItem: () => null, setItem() {} } };
  const api = new Function('document', 'window', 'fetch', 'location', 'setInterval', 'setTimeout',
    'getComputedStyle', 'ResizeObserver', 'localStorage', 'navigator',
    `${src}\n;return { renderClusterModel, clusterModelInfo, modelInfoOf, modelBadge, modelTagList, pickerRowFacts,
      renderModelChoice };`)(
    document, window, () => new Promise(() => {}), { search: '', href: '' }, () => 0, () => 0,
    () => ({}), undefined, window.localStorage, { userAgent: 'node' });
  return { api, byId };
}

const FLASH = 'qwen3.8-flash-next-hibrid48';
const FLASH_SGLANG = 'qwen3.8-flash-next-sglang';
const DENSE = 'qwen3.8-27b-nvidia-nvfp4';
const SGLANG = 'qwen3.8-27b-sglang-dflash2';
const TWIN = { path: 'x_uncensored', default: true, calibration: 'ARCH-EXACT' };
const PROFILE = { engine: 'vllm', speculative: true, num_speculative_tokens: 4, speculative_method: 'mtp',
  kv_cache_memory: 28000000000, ple_replicate: true, tuned: null, refuses: {} };
const LIMITS = { max_model_len: 262144, max_num_seqs: [1, 48] };
const MODELS = [
  { id: FLASH, served_name: 'Qwen3.8-Flash-Next', label: 'Qwen3.8 Flash-Next hibrid48 · MoE NVFP4', size_gib: 98.1,
    limits: LIMITS, uncensored: TWIN, launch: { cluster: PROFILE } },
  { id: FLASH_SGLANG, served_name: 'Qwen3.8-Flash-Next-SGLang', label: 'Qwen3.8 Flash-Next hibrid48 · MoE NVFP4 · SGLang NEXTN',
    size_gib: 98.1, limits: LIMITS, uncensored: TWIN, launch: { cluster: { ...PROFILE, engine: 'sglang', speculative_method: 'nextn' } } },
  { id: DENSE, served_name: 'Qwen3.8-27B', label: 'Qwen3.8 27B · dense NVFP4', size_gib: 20.4, limits: LIMITS,
    uncensored: TWIN, launch: { cluster: PROFILE } },
  { id: SGLANG, served_name: 'Qwen3.8-27B-SGLang', label: 'Qwen3.8 27B · dense NVFP4 · SGLang DFlash2', size_gib: 20.4,
    limits: LIMITS, uncensored: TWIN, launch: { cluster: { ...PROFILE, engine: 'sglang', speculative_method: 'dflash' } } },
];
//: Generation 186 as /lb/status published it on 2026-10-10: the Flash-Next SGLang row, from its twin, starting.
const GEN = { model_choice: FLASH_SGLANG, generation: 186, max_num_seqs: 16, max_num_batched_tokens: 4096,
  num_speculative_tokens: 4, ple_replicate: false, kv_cache_memory: 6993411168, kv_offload_gib: 0,
  max_model_len: 262144, long_prefill_token_threshold: 0, uncensored: true };
const statusOf = (over) => {
  const c = { configured: true, state: 'starting', held: true, models: MODELS,
    limits: { max_num_seqs: [1, 48], max_model_len: [1024, 262144], uncensored: [0, 1] },
    desired: { mode: 'cluster', generation: 186, model_choice: FLASH_SGLANG, max_num_seqs: 16, max_model_len: 262144, uncensored: true },
    effective: { ...GEN }, loaded_model: null, loading: { phase: 'API server up', steps: [] },
    boxes: { 'box-a': {}, 'box-b': {} }, ...(over || {}) };
  return { now: 0, cluster: c, models: { cluster: MODELS, classic: [] }, backends: [] };
};
/* Every tag in an html string: "kind words:text" — the class after `tag `, then the text. */
const tags = (html) => [...String(html).matchAll(/<span class="tag ([^"]*)">([^<]*)<\/span>/g)].map((m) => `${m[1]}:${m[2]}`);
const kinds = (html) => tags(html).map((t) => t.split(/[ :]/)[0]);
let checks = 0;

{ // THE CLUSTER CARD'S MODEL LINE, generation 186 starting: engine, UNCENSORED, drafter, the figures — as tags.
  const { api, byId } = load();
  const s = statusOf();
  api.renderClusterModel(s.cluster, s);
  const html = byId['cluster-model-line'].innerHTML;
  assert.strictEqual(byId['cluster-model-line'].hidden, false);
  const got = tags(html);
  for (const want of ['arch mb-build:MoE NVFP4', 'engine sglang:SGLang', 'twin:uncensored', 'drafter mb-launch:NEXTN 4',
    'fig:256k ctx', 'fig:16 seats', 'fig mb-launch:TP=2']) {
    assert.ok(got.includes(want), `${want} missing from ${JSON.stringify(got)}\n${html}`);
  }
  // Decisive kinds first, figures after: the eye meets engine · checkpoint · drafter before any number.
  const order = kinds(html);
  assert.ok(order.indexOf('engine') < order.indexOf('fig') && order.indexOf('twin') < order.indexOf('fig')
    && order.indexOf('drafter') < order.indexOf('fig'), JSON.stringify(order));
  // The one line still names the model and carries the whole identity in its title.
  assert.ok(/<b class="mb-long">Qwen3.8 Flash-Next hibrid48<\/b>/.test(html), html);
  const title = (html.match(/title="([^"]*)"/) || [])[1] || '';
  assert.ok(title.includes('uncensored') && title.includes('Qwen3.8-Flash-Next-SGLang') && title.includes('NEXTN 4'), title);
  // No engine word, no SGLang/vLLM text, is left as plain text: the engine is the tag and nothing else.
  assert.ok(!/<span>[^<]*SGLang/.test(html), 'the engine is written once, as a tag');
  checks += 11;
}

{ // THE CHECKPOINT IS READ, NEVER GUESSED: the launch record first, the stored switch before one exists, nothing else.
  const line = (over) => {
    const { api, byId } = load();
    const s = statusOf(over);
    api.renderClusterModel(s.cluster, s);
    return tags(byId['cluster-model-line'].innerHTML);
  };
  assert.ok(!line({ effective: { ...GEN, uncensored: false } }).includes('twin:uncensored'), 'the original: no tag');
  assert.ok(line({ effective: { ...GEN, uncensored: false }, desired: { mode: 'cluster', model_choice: FLASH_SGLANG, uncensored: true } })
    .every((t) => !t.startsWith('twin')), 'the launch record outranks the switch: generation 186 launched the original');
  assert.ok(line({ effective: null }).includes('twin:uncensored'), 'no launch record yet: the stored switch says the twin');
  assert.ok(!line({ effective: null, desired: { mode: 'cluster', model_choice: FLASH_SGLANG, uncensored: null } }).includes('twin:uncensored'),
    'a switch from before the knob: no tag, no guess');
  const older = { ...GEN };
  delete older.uncensored;
  assert.ok(!line({ effective: older, desired: { mode: 'cluster', model_choice: FLASH_SGLANG } }).includes('twin:uncensored'),
    'an LB that publishes no checkpoint: no tag');
  // Serving: the loaded model is named by the head, the checkpoint by the generation's record.
  const serving = line({ state: 'running', loaded_model: { served_name: 'Qwen3.8-Flash-Next-SGLang', spec: FLASH_SGLANG,
    label: MODELS[1].label, max_model_len: 262144 } });
  assert.ok(serving.includes('twin:uncensored') && serving.includes('engine sglang:SGLang'), JSON.stringify(serving));
  checks += 6;
}

{ // A SERVER CARD'S BADGE (`modelInfoOf` → `modelBadge`) wears the same tags for the pair it names.
  const { api } = load();
  const s = statusOf({ state: 'running', loaded_model: { served_name: 'Qwen3.8-Flash-Next-SGLang', spec: FLASH_SGLANG,
    label: MODELS[1].label, max_model_len: 262144 } });
  const b = { kind: 'cluster', live: true, served_models: ['Qwen3.8-Flash-Next-SGLang'], loaded: { state: 'serving', models: [] },
    url: 'http://box-a:8000', box: 'box-a' };
  const info = api.modelInfoOf(s, b);
  assert.strictEqual(info.uncensored, true);
  assert.strictEqual(info.engine, 'sglang');
  const got = tags(api.modelBadge(info));
  for (const want of ['engine sglang:SGLang', 'twin:uncensored', 'drafter mb-launch:NEXTN 4', 'fig:256k ctx', 'fig mb-launch:TP=2']) {
    assert.ok(got.includes(want), `${want} missing from ${JSON.stringify(got)}`);
  }
  // A classic box server: the catalog's vLLM profile, no checkpoint known — no twin tag, no guess.
  const classic = { ...s, models: { cluster: MODELS, classic: [{ id: 'qwen3.8-27b-nvfp4', served_name: 'Qwen3.8-27B-classic',
    label: 'Qwen3.8 27B · dense NVFP4', launch: { classic: { engine: 'vllm', speculative: true, num_speculative_tokens: 4, speculative_method: 'mtp' } } }] } };
  const box = { kind: 'vllm', live: true, served_models: ['Qwen3.8-27B-classic'], loaded: { state: 'serving', models: [] }, box: 'box-b', url: 'http://box-b:8000' };
  const boxTags = tags(api.modelBadge(api.modelInfoOf(classic, box)));
  assert.ok(boxTags.includes('engine vllm:vLLM') && boxTags.includes('drafter mb-launch:MTP 4') && !boxTags.some((t) => t.startsWith('twin')),
    JSON.stringify(boxTags));
  checks += 9;
}

{ // THE PICKER: every row and the trigger from the same list — the drafter is a tag on line 1 now, not a chip.
  const { api, byId } = load();
  const setups = { [SGLANG]: { max_num_seqs: 32, max_model_len: 262144, max_num_batched_tokens: 2048, num_speculative_tokens: 4,
    kv_cache_memory: 73435823273 }, [`${SGLANG}@uncensored`]: { max_num_seqs: 32, max_model_len: 262144, max_num_batched_tokens: 2048,
    num_speculative_tokens: 0, kv_cache_memory: 73435823273 } };
  const c = { ...statusOf().cluster, state: 'off', desired: { mode: 'classic', model_choice: SGLANG, uncensored: true }, effective: null, setups,
    boxes: { 'box-a': {}, 'box-b': { availability: { [`${FLASH}@uncensored`]: { present: false } } } } };
  api.renderModelChoice(c, { setup_targets: { [SGLANG]: { gen_tok_s: 177, basis: 'setup', windows: 2134 },
    [`${SGLANG}@uncensored`]: { gen_tok_s: 149.6, basis: 'model', windows: 843 } } }, true);
  const list = byId['cluster-model-list'].innerHTML;
  const rowHtml = (value) => {
    const a = list.indexOf(`data-value="${value}"`);
    const b = list.indexOf('<div class="mp-option', a);
    return list.slice(a, b < 0 ? list.length : b);
  };
  assert.deepStrictEqual(tags(rowHtml(SGLANG)), ['engine sglang:SGLang', 'drafter mb-launch:DFlash2 4', 'size:20.4 GiB', 'fig:32 seqs',
    'fig:256k ctx', 'fig:prefill 2,048', 'fig:KV 73.4 GB', 'rate:target 177 tok/s']);
  assert.deepStrictEqual(tags(rowHtml(`${SGLANG}@uncensored`)), ['engine sglang:SGLang', 'twin:uncensored', 'drafter off mb-launch:no drafter',
    'size:20.4 GiB', 'fig:32 seqs', 'fig:256k ctx', 'fig:prefill 2,048', 'fig:KV 73.4 GB', 'rate other:150 tok/s other setup']);
  assert.deepStrictEqual(tags(rowHtml(`${FLASH}@uncensored`)), ['engine vllm:vLLM', 'twin:uncensored', 'size:98.1 GiB', 'gone:missing on box-b']);
  // The trigger: the family in the ink, then the build as a neutral tag, then the decisive ones, then the figures.
  const btn = byId['cluster-model-button'].innerHTML;
  assert.ok(btn.includes('<b>Qwen3.8 27B</b>'), btn);
  assert.deepStrictEqual(tags(btn), ['arch mb-build:dense NVFP4', 'engine sglang:SGLang', 'twin:uncensored', 'drafter off mb-launch:no drafter',
    'size:20.4 GiB', 'fig:32 seqs', 'fig:256k ctx', 'fig:prefill 2,048', 'fig:KV 73.4 GB', 'rate other:150 tok/s other setup']);
  checks += 5;
}

{ // ONE SOURCE: the picker's facts and the badge's facts are the same shape, so one tag list serves both.
  const { api } = load();
  const f = api.pickerRowFacts(MODELS[1], true, { max_num_seqs: 16, max_model_len: 262144, num_speculative_tokens: 4 }, null, []);
  const list = api.modelTagList(f, { running: true });
  assert.deepStrictEqual(list.map((t) => t.kind), ['arch', 'engine', 'twin', 'running', 'drafter', 'size', 'fig', 'fig']);
  assert.ok(list.every((t) => /^<span class="tag [a-z]/.test(t.html)), JSON.stringify(list));
  checks += 2;
}

{ // THE CSS: one family, the kinds in their hues from the page's tokens, readable on the dark ground.
  assert.ok(/--tag-vllm:\s*\d+,\s*\d+,\s*\d+/.test(css) && /--tag-sglang:\s*\d+,\s*\d+,\s*\d+/.test(css) && /--tag-drafter:\s*\d+,\s*\d+,\s*\d+/.test(css),
    'the three hues are tokens on :root');
  for (const rule of ['.tag.engine.vllm', '.tag.engine.sglang', '.tag.twin', '.tag.drafter', '.tag.rate', '.tag.rate.other', '.tag.gone', '.tag.running']) {
    assert.ok(css.includes(rule), `${rule} has no rule`);
  }
  const vllm = css.match(/\.tag\.engine\.vllm\s*\{([^}]*)\}/)[1];
  const sglang = css.match(/\.tag\.engine\.sglang\s*\{([^}]*)\}/)[1];
  assert.ok(vllm.includes('--tag-vllm') && sglang.includes('--tag-sglang'), 'the two engines take two hues');
  assert.ok(/\.tag\.twin\s*\{[^}]*background:\s*var\(--warn\)/.test(css), 'UNCENSORED is filled in the warning amber');
  assert.ok(!/\.mp-badge\b/.test(css) && !/\.mp-chip\b/.test(css), 'the picker has no second tag family');
  // The model line wraps now: a tag is never hidden by a one-line clamp.
  assert.ok(!/\.model-badge > span \{[^}]*height: 1\.5em/.test(css), 'the badge is clamped to one line');
  assert.ok(!/@container \(max-width: 600px\) \{ \.mb-build \{ display: none; \} \}/.test(css), 'the build tag is dropped at 600 px');
  assert.ok(!/\.cp-model \.mb-launch \{ display: none; \}/.test(css), 'the cluster card drops its drafter tag on a phone');
  checks += 14;
}

console.log(`ok — ${checks} checks`);
