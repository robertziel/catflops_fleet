/* THE MODEL PICKER IS A COMBOBOX, NOT A <select> — checked where the page builds and drives it.
 *
 * Operator, 2026-10-10: "Let's not use standard select, use UX UI". A native <select> holds one line of text per
 * row, so nine checkpoints read as nine 120-character lines. The picker is now a WAI-ARIA select-only combobox
 * (a trigger showing the picked row, a listbox of rows grouped by model family, each row its name, badges and
 * figure chips) DRIVING the same <select id="cluster-model">, which stays in the DOM as the state holder: its
 * value is still `<id>` or `<id>@uncensored`, it still gets the 'change' event, so `pickedRow`, dirty tracking,
 * `clusterChanges` and the Start body are untouched (uncensored_knob.node.js keeps proving those).
 *
 * What is checked here: the rows, groups, badges and chips drawn from one status; the ✓ row and the RUNNING row;
 * a row missing on a box disabled with "missing on <box>"; the keyboard (open, arrows skipping disabled rows,
 * Home/End, Enter commits + fires change, Esc closes without a change and returns focus, type-ahead); a disabled
 * row that cannot be chosen; and the Start body still carrying model_choice + uncensored after a keyboard pick.
 *
 *   node tests/model_picker_ui.node.js   (tests/test_model_picker_ui_page.py runs this when node exists)
 */
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const STATIC = path.join(__dirname, '..', 'vllm_lb', 'static');
const src = fs.readFileSync(path.join(STATIC, 'gateway.js'), 'utf8');
const html = fs.readFileSync(path.join(STATIC, 'index.html'), 'utf8');
const css = fs.readFileSync(path.join(STATIC, 'gateway.css'), 'utf8');

/* A DOM stub that can hold listeners, attributes, classes and focus — enough to drive the picker. */
function load(fetchImpl) {
  const byId = {};
  const document = {};
  const el = (id) => {
    const classes = new Set();
    const e = {
      id, attrs: {}, listeners: {}, events: [], focused: 0, style: {}, hidden: false, innerHTML: '', textContent: '',
      value: '', dataset: {}, disabled: false, options: [],
      addEventListener(type, fn) { (this.listeners[type] = this.listeners[type] || []).push(fn); },
      dispatchEvent(ev) { this.events.push(ev.type); (this.listeners[ev.type] || []).forEach((fn) => fn(ev)); return true; },
      setAttribute(k, v) { this.attrs[k] = String(v); },
      removeAttribute(k) { delete this.attrs[k]; },
      getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; },
      classList: { add: (c) => classes.add(c), remove: (c) => classes.delete(c), contains: (c) => classes.has(c),
        toggle(c, on) { const want = on === undefined ? !classes.has(c) : !!on; if (want) classes.add(c); else classes.delete(c); return want; } },
      classes,
      append() {}, querySelectorAll: () => [], querySelector: () => null, observe() {}, closest: () => null,
      getBoundingClientRect: () => ({}), contains: () => false, scrollIntoView() {},
      focus() { document.activeElement = this; this.focused += 1; },
      blur() { if (document.activeElement === this) document.activeElement = null; },
    };
    return e;
  };
  Object.assign(document, {
    getElementById: (id) => (byId[id] || (byId[id] = el(id))),
    createElementNS: () => el(''), createElement: () => el(''), addEventListener() {},
    querySelectorAll: () => [], querySelector: () => null, activeElement: null,
    body: { style: {}, classList: { toggle() {} } },
    documentElement: { style: {}, classList: { toggle() {} } } });
  const window = { addEventListener() {}, innerWidth: 1280, innerHeight: 800, matchMedia: () => ({
    matches: false, addEventListener() {} }), localStorage: { getItem: () => null, setItem() {} },
    requestAnimationFrame: () => 0 };
  const api = new Function('document', 'window', 'fetch', 'location', 'setInterval', 'setTimeout', 'clearTimeout',
    'getComputedStyle', 'ResizeObserver', 'localStorage', 'navigator', 'requestAnimationFrame',
    `${src}\n;return { renderModelChoice, pickedRow, clusterChanges, pickerRowText, pickerRowFacts, pickerKeydown, shownOf,
      pickerState, pickerOpen, pickerClose, pickerChoose, toggleCluster, setLastStatus: (s) => { lastStatus = s; } };`)(
    document, window, fetchImpl || (() => new Promise(() => {})), { search: '', href: '' }, () => 0, () => 0, () => 0,
    () => ({}), undefined, window.localStorage, { userAgent: 'node' }, () => 0);
  return { api, byId, document };
}

const FLASH = 'qwen3.8-flash-next-hibrid48';
const FLASH_SGLANG = 'qwen3.8-flash-next-sglang';
const DENSE = 'qwen3.8-27b-nvidia-nvfp4';
const DFLASH = 'qwen3.8-27b-dflash2';
const SGLANG = 'qwen3.8-27b-sglang-dflash2';
const NOTE = 'ARCH-EXACT, NOT CALIBRATION-VERIFIED (2026-10-07): the memory block is the original\'s.';
const FLASH_TWIN = { path: 'qwen38-flash-next-cluster/Qwen3.8-Flash-Next-hibrid48_uncensored', default: true, calibration: NOTE };
const DENSE_TWIN = { path: 'Qwen3.8-27B-NVFP4_uncensored', default: true, calibration: NOTE };
const PROFILE = { engine: 'vllm', speculative: true, num_speculative_tokens: 4, speculative_method: 'mtp',
  kv_cache_memory: 28000000000, ple_replicate: true, tuned: null, refuses: {} };
const LIMITS = { max_model_len: 262144, max_num_seqs: [1, 48] };
//: The five cluster entries as /lb/status publishes them, 2026-10-10.
const MODELS = [
  { id: FLASH, label: 'Qwen3.8 Flash-Next hibrid48 · MoE NVFP4', size_gib: 98.1, limits: LIMITS, calibrated: true,
    uncensored: FLASH_TWIN, launch: { cluster: PROFILE } },
  { id: FLASH_SGLANG, label: 'Qwen3.8 Flash-Next hibrid48 · MoE NVFP4 · SGLang NEXTN', size_gib: 98.1, limits: LIMITS,
    calibrated: true, uncensored: null, launch: { cluster: { ...PROFILE, engine: 'sglang', speculative_method: 'nextn' } } },
  { id: DENSE, label: 'Qwen3.8 27B · dense NVFP4', size_gib: 20.4, limits: LIMITS, calibrated: true,
    uncensored: DENSE_TWIN, launch: { cluster: PROFILE } },
  { id: DFLASH, label: 'Qwen3.8 27B · dense NVFP4, BF16 head · DFlash2', size_gib: 22.1, limits: LIMITS,
    calibrated: true, uncensored: null, launch: { cluster: { ...PROFILE, ple_replicate: null, speculative_method: 'dflash' } } },
  { id: SGLANG, label: 'Qwen3.8 27B · dense NVFP4 · SGLang DFlash2', size_gib: 20.4, limits: LIMITS, calibrated: true,
    uncensored: DENSE_TWIN, launch: { cluster: { ...PROFILE, engine: 'sglang', speculative_method: 'dflash' } } },
];
const ROWS = [FLASH, `${FLASH}@uncensored`, FLASH_SGLANG, DENSE, `${DENSE}@uncensored`, DFLASH, SGLANG, `${SGLANG}@uncensored`];
const setupOf = (seqs, batched, kv) => ({ max_num_seqs: seqs, max_model_len: 262144, max_num_batched_tokens: batched,
  num_speculative_tokens: 4, kv_cache_memory: kv });
const SETUPS = {
  [FLASH]: setupOf(6, 4096, 28000000000), [`${FLASH}@uncensored`]: setupOf(6, 4096, 28000000000),
  [FLASH_SGLANG]: setupOf(16, 4096, 6993411168),
  [DENSE]: setupOf(8, 8192, 48000000000), [`${DENSE}@uncensored`]: setupOf(8, 8192, 48000000000),
  [DFLASH]: setupOf(8, 8192, 48000000000),
  [SGLANG]: setupOf(32, 2048, 73435823273), [`${SGLANG}@uncensored`]: setupOf(32, 2048, 73435823273),
};
const S = { setup_targets: {
  [FLASH_SGLANG]: { gen_tok_s: 155.3, basis: 'setup', windows: 2363 },
  [SGLANG]: { gen_tok_s: 177.0, basis: 'setup', windows: 2134 },
  [`${SGLANG}@uncensored`]: { gen_tok_s: 149.6, basis: 'model', windows: 843 },
} };
// Only the limits the picker test needs: no engine knob, so clusterChanges lists none of them (takesKnob).
const CLUSTER_LIMITS = { max_num_seqs: [1, 48], max_model_len: [1024, 262144], uncensored: [0, 1] };
//: Switched off, the SGLang DFlash2 twin stored; the Flash-Next twin is still downloading on box-b.
const C = {
  configured: true, state: 'off', held: false, models: MODELS, limits: CLUSTER_LIMITS, setups: SETUPS,
  desired: { mode: 'classic', generation: 185, model_choice: SGLANG, max_num_seqs: 16, max_model_len: 262144, uncensored: true },
  defaults: { max_num_seqs: 8, max_model_len: 262144, uncensored: true },
  effective: null,
  boxes: { 'box-a': { availability: {} }, 'box-b': { availability: { [`${FLASH}@uncensored`]: { present: false } } } },
};
//: The same, switched on and serving the Flash-Next SGLang row.
const C_RUN = { ...C, state: 'running', desired: { ...C.desired, mode: 'cluster', model_choice: FLASH_SGLANG, uncensored: false },
  effective: { model_choice: FLASH_SGLANG, uncensored: false, max_num_seqs: 16, max_model_len: 262144 } };

/* The listbox's rows, parsed from what the page wrote. */
const OPTION = /<div class="mp-option([^"]*)" id="cluster-model-opt-(\d+)" role="option" aria-selected="(true|false)"( aria-disabled="true")? data-value="([^"]*)" data-index="(\d+)" title="([^"]*)">/g;
function optionsOf(list) {
  const out = [];
  const ms = [...list.innerHTML.matchAll(OPTION)];
  ms.forEach((m, i) => {
    const end = i + 1 < ms.length ? ms[i + 1].index : list.innerHTML.length;
    out.push({ classes: m[1].trim(), index: Number(m[2]), selected: m[3] === 'true', disabled: !!m[4], value: m[5],
      title: m[7], html: list.innerHTML.slice(m.index, end) });
  });
  return out;
}
/* The tags (gateway.css `.tag`, one family everywhere — model_tags.node.js): line 1's decisive kinds, line 2's figures. */
const tagsOf = (html, kinds) => [...html.matchAll(/<span class="tag ([^"]*)">([^<]*)<\/span>/g)]
  .filter((m) => kinds.includes(m[1].split(' ')[0])).map((m) => `${m[1]}:${m[2]}`);
const badges = (html) => tagsOf(html, ['engine', 'twin', 'running', 'drafter']);
const chips = (html) => tagsOf(html, ['size', 'fig', 'rate', 'gone']).map((x) => x.slice(x.indexOf(':') + 1));
const key = (k, extra) => ({ key: k, altKey: false, ctrlKey: false, metaKey: false, prevented: 0,
  preventDefault() { this.prevented += 1; }, stopPropagation() {}, ...(extra || {}) });

let checks = 0;

{ // THE MARKUP: one accessible control — the combobox — over a hidden <select> that still holds the value.
  assert.ok(/<select id="cluster-model"[^>]*class="vh"[^>]*aria-hidden="true"[^>]*tabindex="-1"/.test(html)
    || /<select id="cluster-model"[^>]*tabindex="-1"[^>]*aria-hidden="true"/.test(html), 'the select is the hidden state holder');
  const button = html.match(/<button id="cluster-model-button"[^>]*>/);
  assert.ok(button, 'the trigger button');
  for (const attr of ['role="combobox"', 'aria-haspopup="listbox"', 'aria-expanded="false"', 'aria-controls="cluster-model-list"',
    'aria-labelledby="cluster-model-label"', 'aria-describedby="cluster-model-note"', 'type="button"']) {
    assert.ok(button[0].includes(attr), `${attr} missing from ${button[0]}`);
  }
  assert.ok(/<label id="cluster-model-label" for="cluster-model-button">Model<\/label>/.test(html));
  assert.ok(/<div id="cluster-model-list" class="mp-list" role="listbox" aria-labelledby="cluster-model-label"[^>]*hidden/.test(html));
  // The build's stamp loop is untouched: no new file to stamp (lb/Dockerfile).
  assert.ok(css.includes('.mp-option') && css.includes('.mp-button') && css.includes('.mp-list'), 'the picker is styled in gateway.css');
  // Only the page's own tokens colour it (gateway.css :root): the picker follows the page's one scheme.
  const picker = css.slice(css.indexOf('.mp-button'), css.indexOf('.mp-list') + 4000);
  assert.ok(!/#[0-9a-f]{3,8}\b/i.test(picker.replace(/\/\*[\s\S]*?\*\//g, '')), 'no literal colour in the picker\'s rules');
  checks += 4;
}

{ // ROWS, GROUPS, BADGES, CHIPS: from one status, in the select's order, grouped by the label's first segment.
  const { api, byId } = load();
  api.renderModelChoice(C, S, true);
  const list = byId['cluster-model-list'];
  const groups = [...list.innerHTML.matchAll(/<div class="mp-group" role="group" aria-labelledby="(cluster-model-g\d+)"><div class="mp-group-name" id="\1">([^<]*)<\/div>/g)]
    .map((m) => m[2]);
  assert.deepStrictEqual(groups, ['Qwen3.8 Flash-Next hibrid48', 'Qwen3.8 27B'], list.innerHTML);
  const opts = optionsOf(list);
  assert.deepStrictEqual(opts.map((o) => o.value), ROWS, 'one row per checkpoint, twins under their original');
  assert.deepStrictEqual(opts.map((o) => o.index), [0, 1, 2, 3, 4, 5, 6, 7]);
  // The select is still filled, row for row — the state holder and the listbox never disagree.
  const selectRows = [...byId['cluster-model'].innerHTML.matchAll(/<option value="([^"]*)"/g)].map((m) => m[1]);
  assert.deepStrictEqual(selectRows, ROWS);
  // Each row's title is the one-line text the select shows: one source for the figures (pickerRowText).
  assert.strictEqual(opts[6].title, api.pickerRowText(MODELS[4], false, SETUPS[SGLANG], S.setup_targets[SGLANG], false));
  // Line 1: the name is the label without its family and without its engine segment; the engine, the twin and
  // the drafter are tags (model_tags.node.js has the whole family).
  assert.ok(opts[6].html.includes('<span class="mp-name">dense NVFP4</span>'), opts[6].html);
  assert.deepStrictEqual(badges(opts[6].html), ['engine sglang:SGLang', 'drafter mb-launch:DFlash2 4']);
  assert.deepStrictEqual(badges(opts[7].html), ['engine sglang:SGLang', 'twin:uncensored', 'drafter mb-launch:DFlash2 4']);
  assert.deepStrictEqual(badges(opts[0].html), ['engine vllm:vLLM', 'drafter mb-launch:MTP 4']);
  assert.ok(opts[5].html.includes('<span class="mp-name">dense NVFP4, BF16 head · DFlash2</span>'), opts[5].html);
  assert.ok(!opts[2].html.includes('mp-name">MoE NVFP4 · SGLang'), 'the engine segment is not repeated in the name');
  // Line 2: the figures as chips — the same words pickerRowText prints, one per chip (the drafter is on line 1).
  assert.deepStrictEqual(chips(opts[6].html),
    ['20.4 GiB', '32 seqs', '256k ctx', 'prefill 2,048', 'KV 73.4 GB', 'target 177 tok/s']);
  assert.deepStrictEqual(chips(opts[7].html),
    ['20.4 GiB', '32 seqs', '256k ctx', 'prefill 2,048', 'KV 73.4 GB', '150 tok/s other setup']);
  assert.ok(opts[7].html.includes('<span class="tag rate other">150 tok/s other setup</span>'), opts[7].html);
  assert.ok(opts[6].html.includes('<span class="tag rate">target 177 tok/s</span>'), opts[6].html);
  assert.deepStrictEqual(chips(opts[2].html), ['98.1 GiB', '16 seqs', '256k ctx', 'prefill 4,096', 'KV 7.0 GB', 'target 155 tok/s']);
  // A row missing on a box: disabled, and the box is named — in the chip, the title and the select alike.
  assert.deepStrictEqual(opts.map((o) => o.disabled), [false, true, false, false, false, false, false, false]);
  assert.ok(opts[1].html.includes('<span class="tag gone">missing on box-b</span>'), opts[1].html);
  assert.ok(opts[1].title.endsWith(' · missing on box-b'), opts[1].title);
  assert.ok(byId['cluster-model'].innerHTML.includes('missing on box-b</option>'));
  // The ✓: the stored pair's row, and only that one; nothing is running.
  assert.deepStrictEqual(opts.map((o) => o.selected), [false, false, false, false, false, false, false, true]);
  assert.ok(!list.innerHTML.includes('mp-running'), 'no pair runs: no running mark');
  assert.strictEqual(byId['cluster-model'].value, `${SGLANG}@uncensored`);
  // The trigger shows the picked row: the family, the build as a neutral tag, engine, twin and drafter tags, the figures.
  const btn = byId['cluster-model-button'];
  assert.ok(btn.innerHTML.includes('<b>Qwen3.8 27B</b>') && btn.innerHTML.includes('<span class="tag arch mb-build">dense NVFP4</span>'), btn.innerHTML);
  assert.deepStrictEqual(badges(btn.innerHTML), ['engine sglang:SGLang', 'twin:uncensored', 'drafter mb-launch:DFlash2 4']);
  assert.ok(chips(btn.innerHTML).includes('32 seqs') && chips(btn.innerHTML).includes('150 tok/s other setup'), btn.innerHTML);
  assert.strictEqual(btn.disabled, false);
  assert.strictEqual(btn.title, opts[7].title, 'the whole line is the trigger\'s title');
  // Locked while the cluster cannot take a change: the trigger is disabled like the select.
  api.renderModelChoice(C, S, false);
  assert.strictEqual(btn.disabled, true);
  assert.strictEqual(byId['cluster-model'].disabled, true);
  checks += 24;
}

{ // THE RUNNING ROW is marked, distinct from the ✓ — here they coincide on the trigger and differ after a pick.
  const { api, byId } = load();
  api.renderModelChoice(C_RUN, S, true);
  const opts = optionsOf(byId['cluster-model-list']);
  assert.deepStrictEqual(opts.map((o) => o.html.includes('tag running')), [false, false, true, false, false, false, false, false]);
  assert.ok(opts[2].html.includes('<span class="tag running">running</span>'), opts[2].html);
  assert.ok(opts[2].classes.includes('running') && opts[2].selected, opts[2].classes);
  assert.ok(byId['cluster-model-button'].innerHTML.includes('tag running'), 'the trigger says the picked row runs');
  // Pick another row: the running mark stays where the pair is, the ✓ moves.
  byId['cluster-model'].value = DENSE;
  byId['cluster-model'].dataset.dirty = '1';
  api.renderModelChoice(C_RUN, S, true);
  const again = optionsOf(byId['cluster-model-list']);
  assert.deepStrictEqual(again.map((o) => o.selected), [false, false, false, true, false, false, false, false]);
  assert.deepStrictEqual(again.map((o) => o.html.includes('tag running')), [false, false, true, false, false, false, false, false]);
  assert.ok(!byId['cluster-model-button'].innerHTML.includes('tag running'), 'the trigger shows a row that is not running');
  // While the pair is still coming up (generation 186, 2026-10-10) the launch record's row says so in the state's word.
  const { api: api2, byId: byId2 } = load();
  api2.renderModelChoice({ ...C_RUN, state: 'starting' }, S, true);
  const starting = optionsOf(byId2['cluster-model-list']);
  assert.ok(starting[2].html.includes('<span class="tag running starting">starting</span>'), starting[2].html);
  assert.ok(!byId2['cluster-model-list'].innerHTML.includes('>running<'), 'nothing says running over a pair that is starting');
  checks += 9;
}

{ // THE KEYBOARD: arrows open and move over enabled rows only, Enter commits through the select's change event.
  const { api, byId, document } = load();
  api.renderModelChoice(C, S, true);
  const btn = byId['cluster-model-button'];
  const sel = byId['cluster-model'];
  btn.focus();
  assert.deepStrictEqual(api.pickerState().open, false);
  const down = key('ArrowDown');
  api.pickerKeydown(down);
  assert.strictEqual(down.prevented, 1, 'a handled key does not scroll the page');
  assert.strictEqual(api.pickerState().open, true);
  assert.strictEqual(btn.attrs['aria-expanded'], 'true');
  assert.strictEqual(byId['cluster-model-list'].hidden, false);
  assert.strictEqual(api.pickerState().active, 7, 'opens on the ✓ row');
  assert.strictEqual(btn.attrs['aria-activedescendant'], 'cluster-model-opt-7');
  api.pickerKeydown(key('ArrowDown'));
  assert.strictEqual(api.pickerState().active, 7, 'the last row: no wrap');
  api.pickerKeydown(key('Home'));
  assert.strictEqual(api.pickerState().active, 0);
  api.pickerKeydown(key('ArrowDown'));
  assert.strictEqual(api.pickerState().active, 2, 'row 1 is disabled (missing on box-b): skipped');
  api.pickerKeydown(key('ArrowUp'));
  assert.strictEqual(api.pickerState().active, 0, '…in both directions');
  api.pickerKeydown(key('End'));
  assert.strictEqual(api.pickerState().active, 7);
  assert.ok(byId['cluster-model-opt-7'].classes.has('active') && !byId['cluster-model-opt-0'].classes.has('active'), 'the active row is marked');
  // Nothing has changed yet: moving is not choosing.
  assert.strictEqual(sel.value, `${SGLANG}@uncensored`);
  assert.deepStrictEqual(sel.events, []);
  // Enter on row 4 (the dense twin): the select takes the value and fires change — the page's own listener marks
  // it dirty — and the picker closes with focus back on the trigger.
  api.pickerKeydown(key('ArrowUp')); api.pickerKeydown(key('ArrowUp')); api.pickerKeydown(key('ArrowUp'));
  assert.strictEqual(api.pickerState().active, 4);
  const enter = key('Enter');
  api.pickerKeydown(enter);
  assert.strictEqual(enter.prevented, 1);
  assert.strictEqual(sel.value, `${DENSE}@uncensored`);
  assert.deepStrictEqual(sel.events, ['change']);
  assert.strictEqual(sel.dataset.dirty, '1', 'the change listener ran: the pick is the operator\'s until sent');
  assert.deepStrictEqual(api.pickedRow(), { model: DENSE, uncensored: true });
  assert.ok(api.clusterChanges(C).some((ch) => ch.key === 'model_choice' && ch.to === DENSE), JSON.stringify(api.clusterChanges(C)));
  assert.strictEqual(api.pickerState().open, false);
  assert.strictEqual(btn.attrs['aria-expanded'], 'false');
  assert.strictEqual(document.activeElement, btn);
  // The trigger and the ✓ follow the pick on the next render, through the same pristine rule as the select.
  api.renderModelChoice(C, S, true);
  assert.deepStrictEqual(optionsOf(byId['cluster-model-list']).map((o) => o.selected), [false, false, false, false, true, false, false, false]);
  assert.deepStrictEqual(badges(btn.innerHTML), ['engine vllm:vLLM', 'twin:uncensored', 'drafter mb-launch:MTP 4']);
  // Space commits too; Alt+ArrowUp as well (APG); Escape does not.
  api.pickerKeydown(key('ArrowDown'));          // opens on the ✓ (row 4)
  api.pickerKeydown(key('ArrowUp'));            // row 3
  api.pickerKeydown(key(' '));
  assert.strictEqual(sel.value, DENSE);
  assert.strictEqual(sel.events.length, 2);
  checks += 30;
}

{ // ESCAPE closes without a change and returns focus; a click outside closes the same way.
  const { api, byId, document } = load();
  api.renderModelChoice(C, S, true);
  const btn = byId['cluster-model-button'];
  const sel = byId['cluster-model'];
  api.pickerOpen();
  api.pickerKeydown(key('Home'));
  assert.strictEqual(api.pickerState().active, 0);
  btn.focused = 0;
  document.activeElement = null;
  const esc = key('Escape');
  api.pickerKeydown(esc);
  assert.strictEqual(esc.prevented, 1);
  assert.strictEqual(api.pickerState().open, false);
  assert.strictEqual(byId['cluster-model-list'].hidden, true);
  assert.strictEqual(sel.value, `${SGLANG}@uncensored`, 'Escape chose nothing');
  assert.deepStrictEqual(sel.events, []);
  assert.strictEqual(btn.focused, 1, 'focus returns to the trigger');
  // An Escape while closed is not the picker's: the page's confirm box reads it.
  const esc2 = key('Escape');
  api.pickerKeydown(esc2);
  assert.strictEqual(esc2.prevented, 0);
  // Tab commits the active row (APG) and lets focus move on.
  api.pickerOpen();
  api.pickerKeydown(key('Home'));
  const tab = key('Tab');
  api.pickerKeydown(tab);
  assert.strictEqual(tab.prevented, 0);
  assert.strictEqual(api.pickerState().open, false);
  assert.strictEqual(sel.value, FLASH);
  checks += 11;
}

{ // A DISABLED ROW cannot be chosen — by keyboard (never active) or by click (pickerChoose refuses).
  const { api, byId } = load();
  api.renderModelChoice(C, S, true);
  const sel = byId['cluster-model'];
  api.pickerOpen();
  assert.strictEqual(api.pickerChoose(1), false);
  assert.strictEqual(sel.value, `${SGLANG}@uncensored`);
  assert.deepStrictEqual(sel.events, []);
  assert.strictEqual(api.pickerState().open, true, 'a refused click leaves the list open');
  assert.strictEqual(api.pickerChoose(2), true);
  assert.strictEqual(sel.value, FLASH_SGLANG);
  assert.deepStrictEqual(sel.events, ['change']);
  assert.strictEqual(api.pickerState().open, false);
  // Choosing the ✓ row again is a close, not a change.
  api.pickerOpen();
  assert.strictEqual(api.pickerChoose(2), true);
  assert.deepStrictEqual(sel.events, ['change']);
  assert.strictEqual(api.pickerState().open, false);
  checks += 9;
}

{ // TYPE-AHEAD: a printable key opens and jumps to the next enabled row whose label starts with what was typed.
  const { api, byId } = load();
  api.renderModelChoice(C, S, true);
  api.pickerKeydown(key('q'));
  assert.strictEqual(api.pickerState().open, true);
  // Every label starts with "Qwen3.8": the same letter again walks the rows, skipping the disabled one.
  assert.strictEqual(api.pickerState().active, 0, 'from the ✓ (row 7) the next q-row wraps to 0');
  api.pickerKeydown(key('q'));
  assert.strictEqual(api.pickerState().active, 2);
  api.pickerState().typed = '';
  api.pickerKeydown(key('Q'));
  api.pickerKeydown(key('w'));
  api.pickerKeydown(key('e'));
  api.pickerKeydown(key('n'));
  api.pickerKeydown(key('3'));
  api.pickerKeydown(key('.'));
  api.pickerKeydown(key('8'));
  api.pickerKeydown(key(' '));
  api.pickerKeydown(key('2'));
  assert.strictEqual(api.pickerState().active, 3, '"qwen3.8 2" is the first 27B row');
  assert.strictEqual(byId['cluster-model'].value, `${SGLANG}@uncensored`, 'type-ahead moves, it does not choose');
  checks += 4;
}

{ // A RE-RENDER NEVER SCROLLS THE OPEN LIST (operator 2026-10-26, on a phone: "select list keeps scrolling down so I can't
  // choose models from top"): every status poll repaints the picker, and marking the active row scrolled it into view
  // each time, pulling a list the operator had scrolled back to that row. The keyboard still scrolls to the row it moves.
  const { api, byId } = load();
  api.renderModelChoice(C, S, true);
  api.pickerOpen();
  let scrolls = 0;
  ROWS.forEach((_, i) => { byId[`cluster-model-opt-${i}`].scrollIntoView = () => { scrolls += 1; }; });
  api.renderModelChoice({ ...C, setups: { ...SETUPS, [FLASH]: setupOf(7, 4096, 28000000000) } }, S, true);
  api.renderModelChoice({ ...C, state: 'starting' }, S, true);
  assert.strictEqual(scrolls, 0, 'a poll scrolled the open list');
  api.pickerKeydown(key('ArrowDown'));
  assert.ok(scrolls >= 1, 'the keyboard moved the active row and must still bring it into view');
  checks += 2;
}

{ // A PICK THE DEPLOYMENT NO LONGER OFFERS (its row gone from `cluster.offers`) is not left on the select, where the
  // trigger read "no model" over a running pair: the stored row is shown instead, as on a fresh page.
  const { api, byId } = load();
  api.renderModelChoice(C, S, true);
  const sel = byId['cluster-model'];
  sel.value = FLASH_SGLANG; sel.dataset.dirty = '1';        // picked earlier; now taken off the offers
  api.renderModelChoice({ ...C, offers: [`${SGLANG}@uncensored`, DENSE] }, S, true);
  assert.strictEqual(sel.value, `${SGLANG}@uncensored`, 'the stored row');
  assert.ok(!byId['cluster-model-button'].innerHTML.includes('no model'), byId['cluster-model-button'].innerHTML);
  checks += 2;
}

(async () => { // START still posts the row's model_choice + uncensored after a keyboard pick: the body is built from pickedRow().
  const posts = [];
  const fetchImpl = (url, init) => {
    if (init && init.method === 'POST') {
      posts.push({ url, body: JSON.parse(init.body) });
      return Promise.resolve({ ok: true, json: async () => ({}) });
    }
    return new Promise(() => {});                 // /lb/status and the estimate never answer here
  };
  const { api, byId } = load(fetchImpl);
  const status = { cluster: C, setup_targets: S.setup_targets, now: 0 };
  api.setLastStatus(status);
  api.renderModelChoice(C, S, true);
  byId['cluster-seqs'].value = '16';
  byId['cluster-context'].value = '262144';
  api.pickerOpen();
  api.pickerKeydown(key('Home'));
  api.pickerKeydown(key('ArrowDown'));           // row 2: the Flash-Next SGLang original
  api.pickerKeydown(key('Enter'));
  assert.deepStrictEqual(api.pickedRow(), { model: FLASH_SGLANG, uncensored: false });
  api.toggleCluster();                           // never settles here: the re-poll it ends with hangs on purpose
  await new Promise((resolve) => setImmediate(resolve));
  assert.strictEqual(posts.length, 1, JSON.stringify(posts));
  assert.strictEqual(posts[0].url, '/lb/cluster');
  const b = posts[0].body;
  assert.strictEqual(b.enabled, true);
  assert.strictEqual(b.model_choice, FLASH_SGLANG);
  assert.strictEqual(b.uncensored, false);
  // The concurrency ON SCREEN — the picked model's own default after a model change, as today.
  assert.strictEqual(b.max_num_seqs, api.shownOf('max_num_seqs'));
  checks += 6;
  // THE RUNNING ROW'S CHIPS ARE ITS RUNNING SETUP (operator 2026-10-30: "I selected other setup than default, why still
  // shows default in select?"): a pair up at 12 seats × 131,072 shows "12 seqs · 128k ctx" on its row, not the
  // catalog default; every other row keeps its default.
  {
    const { api, byId } = load();
    const run = { ...C_RUN, effective: { ...C_RUN.effective, max_num_seqs: 12, max_model_len: 131072 } };
    api.renderModelChoice(run, S, true);
    const opts = optionsOf(byId['cluster-model-list']);
    const running = opts.find((o) => o.html.includes('tag running'));
    assert.ok(running && running.html.includes('12 seqs') && running.html.includes('128k ctx'), running && running.html);
    const def = SETUPS[DENSE];
    const dense = opts.find((o) => o.value === DENSE);
    assert.ok(!def || dense.html.includes(`${def.max_num_seqs} seqs`), `the default row lost its default: ${dense.html}`);
    checks += 2;
  }
  console.log(`ok — ${checks} checks`);
})().catch((e) => { console.error(e); process.exit(1); });
