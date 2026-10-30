/* WHICH MODEL THE CARD NAMES — checked where the page builds the name, against the SHIPPED catalog.
 *
 * The operator reported three cluster cards that all read "Qwen3.8 27B" and no way to tell which model
 * was loading or serving. `modelBadge` (vllm_lb/static/gateway.js) prints the label's FIRST segment as
 * the name and everything after it as `mb-build`, and vllm_lb/library gives three models the same
 * first segment:
 *
 *   "Qwen3.8 27B · dense NVFP4"                        qwen3.8-27b-nvidia-nvfp4     cluster
 *   "Qwen3.8 27B · dense NVFP4, BF16 head · DFlash2"   qwen3.8-27b-dflash2          cluster
 *   "Qwen3.8 27B · dense NVFP4 · SGLang DFlash2"       qwen3.8-27b-sglang-dflash2   cluster
 *   "Qwen3.8 27B · dense NVFP4"                        qwen3.8-27b-nvfp4            classic
 *
 * `mb-build` — the only part that told them apart — is hidden at a 600 px container (gateway.css), and
 * the cluster panel measures 440 px at a 1440 viewport and 371 px at 390, so it was drawn on no screen.
 * So the badge now prints the SERVED ID for a name the catalog shares (`.ambiguous`), which is unique by
 * construction: one engine, one id, one address.
 *
 * Also the drafter: catalog.json states `speculative_config.method` — "mtp" for Flash-Next and the NVIDIA
 * 27B, "dflash" for both DFlash2 builds — and since ef00125 `LaunchProfile.view()` publishes it as
 * `speculative_method`, so the page is told which and prints it ("MTP 4", "DFlash2 4"). It used to say
 * "MTP" over every drafter. The method here is read out of catalog.json and the printed word must CARRY
 * it — no map in this file, so the page may choose its own words but not another model's method. An
 * LB that publishes no method (anything before ef00125) must still print the depth alone.
 *
 * The MARKUP contract is here; the drawn pixels at 1440 and 390 are worklog/gateway_ui_states.py
 * (`check_badge_tells_the_models_apart`), which renders the real `ModelSpec.view()` in Chromium. The view
 * built here carries only the fields the badge reads, which that harness checks against the real one.
 *
 *   node tests/model_badge.node.js      (tests/test_model_badge.py runs this when node exists)
 */
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const STATIC = path.join(__dirname, '..', 'vllm_lb', 'static');
const src = fs.readFileSync(path.join(STATIC, 'gateway.js'), 'utf8');
const css = fs.readFileSync(path.join(STATIC, 'gateway.css'), 'utf8').replace(/\s+/g, ' ');
const raw = { models: require('./library_rows.js')() };     // the shipped model library, as rows

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
  // `typeof` never throws on a name the file does not declare: an older gateway.js must fail on the
  // CONTRACT below, not on loading.
  `${src}\n;return { modelBadge, clusterModelInfo, modelInfoOf, shortModel,
    clusterFigures: typeof clusterFigures === 'function' ? clusterFigures : null,
    sharedModelName: typeof sharedModelName === 'function' ? sharedModelName : null };`)(
  document, window, () => new Promise(() => {}), { search: '', href: '' }, () => 0, () => 0,
  () => ({}), undefined, window.localStorage, { userAgent: 'node' });

/* The fields of `ModelSpec.view()` the badge reads, for one row of the model library — `speculative_method`
 * and `engine` since ef00125. */
const viewOf = (m, mode) => ({ id: m.id, label: m.label, served_name: m.served_name,
  launch: { [mode]: { speculative: !!(m.launch[mode] || {}).speculative_config,
    num_speculative_tokens: (((m.launch[mode] || {}).speculative_config) || {}).num_speculative_tokens || 0,
    speculative_method: (((m.launch[mode] || {}).speculative_config) || {}).method || null,
    engine: (m.launch[mode] || {}).engine || 'vllm' } } });
const CLUSTER = raw.models.filter((m) => m.launch.cluster);
const CLASSIC = raw.models.filter((m) => m.launch.classic);
assert.ok(CLUSTER.length >= 3, `the shipped catalog ships ${CLUSTER.length} cluster models`);

/* /lb/status as the page reads it: every cluster spec published TWICE (proxy.py `_models_block` and
 * cluster_loop.py's `cluster.models`), which is one model, not two. */
const statusFor = (m, state) => {
  const cluster = CLUSTER.map((x) => viewOf(x, 'cluster'));
  const s = { models: { cluster, classic: CLASSIC.map((x) => viewOf(x, 'classic')) },
    cluster: { configured: true, models: cluster, boxes: { 'box-a': {}, 'box-b': {} },
      desired: { mode: 'cluster', model_choice: m.id, max_num_seqs: 8, max_model_len: 262144 },
      effective: null, loaded_model: null } };
  if (state === 'serving') {
    s.cluster.loaded_model = { served_name: m.served_name, spec: m.id, label: m.label, max_model_len: 262144 };
    s.cluster.effective = { mode: 'cluster', model_choice: m.id, max_num_seqs: 8, max_model_len: 262144 };
  } else {
    s.cluster.loading = { phase: 'loading model weights', percent: 42 };
  }
  return s;
};

/* THE NAME THE CSS KEEPS. `.mb-short` is `display: none` until the badge is `.ambiguous` (or the card
 * gets narrow); then it is the name and `.mb-long` is the one that goes. A badge whose name needs no
 * second spelling prints one plain `<b>`. */
const segments = (html) => [...html.matchAll(/<(b|span)(?: class="([^"]*)")?>([^<]*)<\/\1>/g)]
  .map((m) => ({ tag: m[1], cls: m[2] || '', text: m[3].replace(/&middot;/g, '·').trim() }));
function shownName(html) {
  const ambiguous = /class="model-badge[^"]*\bambiguous\b/.test(html);
  const segs = segments(html).filter((x) => x.tag === 'b');
  const keep = segs.find((x) => x.cls === (ambiguous ? 'mb-short' : 'mb-long')) || segs[0];
  assert.ok(keep && keep.text, `no name in ${html}`);
  return keep.text;
}
const titleOf = (html) => (html.match(/title="([^"]*)"/) || [])[1] || '';
// Since 2026-10-10 a launch fact is a TAG (`<span class="tag drafter mb-launch">`, model_tags.node.js): `mb-launch`
// is one of its classes, and the text carries no separator of its own.
const launchFacts = (html) => segments(html).filter((x) => x.cls.split(' ').includes('mb-launch')).map((x) => x.text);
let checks = 0;

// 1. THE REPORT: one distinct name per cluster model, in both the state the pair serves one and the state
//    a Start is bringing one up — and the full identity in the title either way.
for (const state of ['serving', 'pending']) {
  const names = CLUSTER.map((m) => {
    const html = api.modelBadge(api.clusterModelInfo(statusFor(m, state)), ['8 seqs']);
    const name = shownName(html);
    assert.ok(titleOf(html).includes(m.label), `${m.id}: the title drops the label: ${titleOf(html)}`);
    assert.ok(titleOf(html).includes(m.served_name), `${m.id}: the title drops the served id`);
    return [m.id, name];
  });
  const seen = new Set(names.map(([, name]) => name));
  assert.strictEqual(seen.size, names.length,
    `${state}: ${names.length} cluster models, ${seen.size} names: ${JSON.stringify(names)}`);
  checks += 1 + 2 * CLUSTER.length;
}

// 2. EVERY FAMILY THE CATALOG SHARES A NAME WITH falls back to its served id. This used to read "the
//    three 27Bs, and the one model nothing shares a name with keeps the catalog's words" — which quietly
//    assumed exactly one cluster model had a unique first segment. Adding the SGLang profile of
//    Flash-Next made that two, and the assertion failed on behaviour that was correct: both Flash-Next
//    entries now disambiguate by served id exactly as the 27Bs do. The rule is per FAMILY, so the test
//    is too, and it no longer depends on the catalog happening to contain a lone model.
{
  const families = new Map();
  for (const m of CLUSTER) {
    const first = m.label.split(' · ')[0].trim();
    families.set(first, (families.get(first) || []).concat([m]));
  }
  assert.ok(families.size >= 2, `cluster models fall into ${families.size} name families`);
  for (const [first, members] of families) {
    for (const m of members) {
      const name = shownName(api.modelBadge(api.clusterModelInfo(statusFor(m, 'serving')), ['8 seqs']));
      if (members.length > 1) {
        assert.strictEqual(name, api.shortModel(m.served_name), `${m.id} is named ${name}`);
        assert.notStrictEqual(name, first, `${m.id} is named for its family "${first}"`);
      } else {
        assert.strictEqual(name, first, `${m.id} lost its own name: ${name}`);
      }
      checks += members.length > 1 ? 2 : 1;
    }
  }
  checks += 1;
}

// 3. A name the catalog shares is shared ACROSS modes too: the classic "Qwen3.8 27B · dense NVFP4" is a
//    different checkpoint from the cluster model of the same label, and a card must not read the same.
{
  assert.strictEqual(typeof api.sharedModelName, 'function',
    'gateway.js does not ask the catalog whether a name is shared');
  const classicDense = CLASSIC.find((m) => m.label.startsWith('Qwen3.8 27B'));
  assert.ok(classicDense, 'the shipped catalog has no classic 27B');
  const s = statusFor(CLUSTER.find((m) => m.label.startsWith('Qwen3.8 27B')), 'serving');
  assert.strictEqual(api.sharedModelName(s, classicDense.label), true);
  // A name NO shipped model carries. This used to name the Flash-Next label, which stopped being
  // unique the moment that checkpoint gained a second profile — the rule was fine, the fixture was
  // coupled to what the catalog happened to hold. A synthetic name cannot go stale that way.
  assert.strictEqual(api.sharedModelName(s, 'Not A Shipped Model · some build'), false,
    'a name no other model carries must stay the name');
  assert.strictEqual(api.sharedModelName(s, ''), false);
  assert.strictEqual(api.sharedModelName({}, 'Qwen3.8 27B · dense NVFP4'), false,
    'with no catalog on the wire nothing is known to be shared');
  checks += 5;
}

// 4. THE DRAFTER the wire publishes, named and counted — and no OTHER model's method.
{
  // dflash (the dense 27B's own checkpoint), mtp (Flash-Next on vLLM) and nextn (Flash-Next on
  // SGLang, which reads the same `mtp.` tensors through SGLang's own EAGLE path). The point of
  // pinning the set is that the badge must name the profile's OWN method and never another's, so a
  // new method belongs here the moment a profile ships one.
  // Only the profiles the LB shapes a drafter for: TensorFold drafts by its own rule and names none (no
  // speculative_config, catalog.ENGINE_REFUSES num_speculative_tokens).
  const DRAFTED = CLUSTER.filter((m) => m.launch.cluster.speculative_config);
  const methods = new Set(DRAFTED.map((m) => m.launch.cluster.speculative_config.method));
  // dspark: DeepSeek-V4-Flash's own drafter (2026-10-23).
  assert.deepStrictEqual([...methods].sort(), ['dflash', 'dspark', 'mtp', 'nextn'],
    `the cluster's drafter methods are ${[...methods]}`);
  for (const m of DRAFTED) {
    const sc = m.launch.cluster.speculative_config;
    const facts = launchFacts(api.modelBadge(api.clusterModelInfo(statusFor(m, 'serving')), ['8 seqs']))
      .filter((t) => !t.startsWith('TP='));
    const read = JSON.stringify(facts);
    assert.ok(facts.some((t) => t.toLowerCase().includes(sc.method)),
      `${m.id} drafts with ${sc.method}; the launch facts read ${read}`);
    assert.ok(facts.some((t) => t.includes(String(sc.num_speculative_tokens))),
      `${m.id} drafts ${sc.num_speculative_tokens} per step; the launch facts read ${read}`);
    for (const other of [...methods].filter((x) => x !== sc.method)) {
      assert.ok(!facts.some((t) => t.toLowerCase().includes(other)),
        `${m.id} drafts with ${sc.method} and the launch facts say ${other}: ${read}`);
    }
    checks += 3;
  }
  // AN LB THAT DOES NOT SAY (anything before ef00125, or a profile whose method is null): the depth
  // alone, never a method the page made up.
  {
    const m = CLUSTER[0];
    const s = statusFor(m, 'serving');
    for (const list of [s.models.cluster, s.cluster.models]) {
      list.forEach((v) => { if (v.launch.cluster) v.launch.cluster.speculative_method = null; });
    }
    const facts = launchFacts(api.modelBadge(api.clusterModelInfo(s), ['8 seqs']))
      .filter((t) => !t.startsWith('TP='));
    const words = new RegExp(`\\b(${[...methods, 'eagle', 'medusa', 'dflash2'].join('|')})\\b`, 'i');
    assert.deepStrictEqual(facts.filter((t) => words.test(t)), [],
      `with no method on the wire the launch facts read ${JSON.stringify(facts)}`);
    assert.ok(facts.includes(`draft ${m.launch.cluster.speculative_config.num_speculative_tokens}`),
      `with no method on the wire the launch facts read ${JSON.stringify(facts)}`);
    checks += 2;
  }
  checks += 1;
}

// 6. The CSS keeps the served id for a shared name — the markup above means nothing without these two
//    rules, and they must hold at EVERY width (no @container around them).
{
  assert.ok(css.includes('.model-badge.ambiguous .mb-long { display: none; }'), 'no .ambiguous rule');
  assert.ok(css.includes('.model-badge.ambiguous .mb-short { display: inline; }'), 'no .ambiguous rule');
  const guarded = /@container[^{]*\{[^{}]*\.model-badge\.ambiguous/.test(css);
  assert.ok(!guarded, 'the .ambiguous rules are inside a container query');
  // And the ≤400 swap is EVERY container's: the cluster panel is 371 px at a 390 viewport and carries
  // no `.card`, so a `.card`-scoped rule left its line reading the full label there.
  const narrow = css.match(/@container \(max-width: 400px\) \{([^}]*\})*[^}]*\}/);
  assert.ok(narrow && /\.model-badge \.mb-long \{ display: none; \}/.test(narrow[0]),
    `the 400 px rule is ${narrow && narrow[0]}`);
  assert.ok(!/\.card \.model-badge \.mb-long/.test(css),
    'the name swap is still scoped to .card, which the cluster panel is not');
  checks += 5;
}

// THE KV PRECISION ON THE CARD (operator 2026-10-26, at a running pair: "Why no KV precision data?"): the chips name
// the KV type the pair RUNS (`cluster.effective.kv_cache_dtype`, the launch record) beside its seats; none before one.
assert.ok(api.clusterFigures, 'gateway.js has clusterFigures');
assert.deepStrictEqual(api.clusterFigures({ effective: { max_num_seqs: 16, kv_cache_dtype: 'bfloat16' } }),
  ['16 seats', 'KV BF16']);
assert.deepStrictEqual(api.clusterFigures({ effective: { max_num_seqs: 8, kv_cache_dtype: 'fp8_e4m3' } }),
  ['8 seats', 'KV FP8 E4M3']);
assert.deepStrictEqual(api.clusterFigures({ effective: { max_num_seqs: 8, kv_cache_dtype: 'nvfp4' } }), ['8 seats', 'KV NVFP4']);
assert.deepStrictEqual(api.clusterFigures({ effective: { max_num_seqs: 8, kv_cache_dtype: 'auto' } }), ['8 seats', 'KV auto']);
assert.deepStrictEqual(api.clusterFigures({ desired: { mode: 'cluster', max_num_seqs: 6 } }), ['6 seats'],
  'starting with no launch record yet: no KV type to state');
assert.deepStrictEqual(api.clusterFigures({ desired: { mode: 'classic', max_num_seqs: 6 } }), []);
checks += 6;

console.log(`ok — ${checks} checks`);
