/* WHAT A CONFIGURATION IS on /benchmarks, and how it is said — benchmarks.js `settingsKey`, `variantKey`,
 * `cfgText`, `cfgStyle`, `groupsOf`, `renderPerfMatrix`, `inventory`, `contextRows`, `renderContext`.
 *
 * Operator 2026-10-12, a one-host node's page: its configurations read "4 seqs · spec 0 · batched ?" and
 * "16 seqs · spec 0 · batched ?" — 190 minutes under a 7.92 GiB KV pin and 77 without one POOLED into one
 * "16 seqs" row, because the page's identity stopped at (model, seqs, batched, spec, context). The rules:
 *   1. a configuration is every launch setting a minute is tagged with that changes what the engine does —
 *      the GPU memory fraction, the KV pin and the KV tier in the host's RAM included;
 *   2. ABSENT (recorded before the key existed) is "not recorded", null is the launch leaving it to the
 *      engine ("batched vLLM default", "KV not pinned", "no RAM tier"), a value is itself — three things,
 *      never folded into one another and never a bare "?";
 *   3. a label names a memory setting when it DIFFERS among that model's configurations, and the encoding
 *      stays the page's one: cap = blue step, a second configuration at the cap = hatch;
 *   4. the matrix draws only models with a recorded minute (a model that only booted has no grid, as it has
 *      no chip), each grid captioned with its name from /lb/perf `models`;
 *   5. a minute's context is derived only from the KV pool THAT MINUTE recorded — no model's pool is assumed
 *      for another minute, and no model id is written in the page.
 * A memoised stub DOM: `$` hands back the same element for an id, so what a renderer wrote can be read.
 *
 *   node tests/config_identity.node.js      (tests/test_config_identity.py runs this when node exists)
 */
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const STATIC = path.join(__dirname, '..', 'vllm_lb', 'static');
// the page loads server-scope.js first (its `serverScope`): so does this harness
const src = fs.readFileSync(path.join(STATIC, 'server-scope.js'), 'utf8') + '\n' + fs.readFileSync(path.join(STATIC, 'benchmarks.js'), 'utf8');
const html = fs.readFileSync(path.join(STATIC, 'benchmarks.html'), 'utf8');
const els = new Map();
const mk = () => ({ addEventListener() {}, setAttribute() {}, append() {}, prepend() {}, remove() {},
  classList: { toggle() {}, add() {}, remove() {}, contains: () => false }, style: {}, hidden: false, innerHTML: '',
  textContent: '', value: '', dataset: {}, querySelectorAll: () => [], querySelector: () => null, observe() {},
  contains: () => false, parentNode: null });
const document = { getElementById: (id) => els.get(id) || els.set(id, mk()).get(id), createElementNS: mk, createElement: mk,
  addEventListener() {}, querySelectorAll: () => [], body: mk(), activeElement: null };
const window = { addEventListener() {}, innerWidth: 1280, innerHeight: 800, scrollTo() {},
  localStorage: { getItem: () => null, setItem() {} } };
const api = new Function('document', 'window', 'fetch', 'location', 'setInterval', 'getComputedStyle', 'ResizeObserver',
  `${src}\n;return { perf, state, pm, recomputeDomains, settingsKey, variantKey, cfgText, cfgStyle, capColor, groupsOf, timed,
    renderPerfMatrix, contextRows, renderContext, pmSelect, gcfg, inventory: typeof inventory === 'function' ? inventory : null,
    winAsRun, seriesIn, capsOf, perfTwinHTML };`)(
  document, window, () => new Promise(() => {}), { search: '' }, () => 0, () => ({}), undefined);

const M = 'a-dense-model', OTHER = 'another-model', BOOTED = 'a-model-that-only-booted';
const PIN = 8501810176;                                  // 7.92 GiB: the pin the node's minutes carry
/** A minute's config exactly as a one-host node's effective was recorded before 18e23aa (no fraction, no tier
 *  key) — `more` adds or overrides keys, `drop` removes them. */
const cfg = (seqs, more, drop) => {
  const c = { model_choice: M, generation: 1, max_num_seqs: seqs, max_num_batched_tokens: null, num_speculative_tokens: 0,
    max_model_len: 131072, kv_cache_memory: null, kv_offload_gib: null, ple_replicate: null,
    long_prefill_token_threshold: null, uncensored: true, ...(more || {}) };
  for (const k of drop || []) delete c[k];
  return c;
};
const win = (c, t, more) => ({ t_start: t, t_end: t + 60, span_s: 60, config: c, config_source: 'effective', gen_tokens: 600, ...(more || {}) });
const rec = (id, label, engine, ws) => ({ id, label, known: !!engine, engine: engine ? engine.toLowerCase() : null,
  engine_name: engine, minutes: ws.length, first: ws[0].t_start, last: ws[ws.length - 1].t_end });
const load = (windows, models, boots) => {
  api.perf.data = { windows, boots: boots || [], models }; api.perf.model = null; api.state.index = null;
  api.recomputeDomains();
};
let checks = 0;

// 1. IDENTITY: the pin, the fraction and the RAM tier each make a different configuration, and absent, null
//    and a value are three different tokens.
{
  const k = (c) => api.settingsKey(win(c, 0));
  const keys = [k(cfg(16)), k(cfg(16, { kv_cache_memory: PIN })), k(cfg(16, { gpu_memory_utilization: 0.95 })),
    k(cfg(16, { gpu_memory_utilization: 0.9 })), k(cfg(16, { kv_offload_ram_gib: null })), k(cfg(16, { kv_offload_ram_gib: 32 })),
    k(cfg(16, {}, ['kv_cache_memory'])), k(cfg(16, { max_num_batched_tokens: 8192 })), k(cfg(16, {}, ['max_num_batched_tokens']))];
  assert.strictEqual(new Set(keys).size, keys.length, `two of these share a key:\n${keys.join('\n')}`);
  assert.strictEqual(keys[0].split(' | ')[0], M, 'the model stays the first field of the key');
  const ws = [win(cfg(16), 0), win(cfg(16, { kv_cache_memory: PIN }), 60), win(cfg(16, { kv_cache_memory: PIN }), 120)];
  assert.strictEqual(api.groupsOf(ws, []).length, 2, '190 pinned minutes and 77 unpinned are two configurations, never one');
  checks += 3;
}

// 2. LABELS: the node's three configurations as recorded — no "?", the engine's default said by name, and
//    the pin named because it differs among this model's configurations.
const today = [win(cfg(4), 0), win(cfg(16), 60), win(cfg(16, { kv_cache_memory: PIN }), 120), win(cfg(16, { kv_cache_memory: PIN }), 180)];
{
  load(today, [rec(M, 'Dense', 'vLLM', today)]);
  const labels = [...new Set(today.map((w) => api.cfgText(w.config)))];
  assert.deepStrictEqual(labels, ['4 seqs · spec 0 · batched vLLM default · KV not pinned',
    '16 seqs · spec 0 · batched vLLM default · KV not pinned', '16 seqs · spec 0 · batched vLLM default · KV 7.92 GiB']);
  assert.ok(labels.every((l) => !l.includes('?')), labels.join(' / '));
  checks += 2;
}
// ...and after 18e23aa the node's minutes carry the fraction and the tier: the older minutes say "not
// recorded" for both, never read as the newer values.
{
  const now = [...today, win(cfg(16, { gpu_memory_utilization: 0.95, kv_cache_memory: PIN, kv_offload_ram_gib: 32 }), 240),
    win(cfg(16, { gpu_memory_utilization: 0.95, kv_offload_ram_gib: null }), 300)];
  load(now, [rec(M, 'Dense', 'vLLM', now)]);
  assert.deepStrictEqual(now.slice(2).map((w) => api.cfgText(w.config)), [
    '16 seqs · spec 0 · batched vLLM default · GPU mem not recorded · KV 7.92 GiB · RAM tier not recorded',
    '16 seqs · spec 0 · batched vLLM default · GPU mem not recorded · KV 7.92 GiB · RAM tier not recorded',
    '16 seqs · spec 0 · batched vLLM default · GPU mem 0.95 · KV 7.92 GiB · RAM 32 GiB',
    '16 seqs · spec 0 · batched vLLM default · GPU mem 0.95 · KV not pinned · no RAM tier']);
  checks += 1;
}
// ...a model the catalog does not know has no engine to name; a minute recorded before the budget was a key
// says so; a model whose memory settings never differ (a pair: pinned once, no fraction, no tier) prints none.
{
  const ws = [win(cfg(8, { model_choice: OTHER }), 0), win(cfg(8, { model_choice: OTHER }, ['max_num_batched_tokens']), 60)];
  load(ws, [rec(OTHER, OTHER, null, ws)]);
  assert.deepStrictEqual(ws.map((w) => api.cfgText(w.config)), ['8 seqs · spec 0 · batched engine default',
    '8 seqs · spec 0 · batched not recorded']);
  const pair = [win(cfg(8, { max_num_batched_tokens: 8192, num_speculative_tokens: 4, kv_cache_memory: 28e9 }), 0),
    win(cfg(16, { max_num_batched_tokens: 8192, num_speculative_tokens: 4, kv_cache_memory: 28e9 }), 60)];
  load(pair, [rec(M, 'Dense', 'vLLM', pair)]);
  assert.deepStrictEqual(pair.map((w) => api.cfgText(w.config)), ['8 seqs · spec 4 · batched 8,192', '16 seqs · spec 4 · batched 8,192']);
  checks += 2;
}

// 3. THE ONE ENCODING: the pinned and the unpinned 16 are the same blue (the cap's step), the second hatched.
{
  load(today, [rec(M, 'Dense', 'vLLM', today)]);
  const free = api.cfgStyle(cfg(16)), pinned = api.cfgStyle(cfg(16, { kv_cache_memory: PIN })), four = api.cfgStyle(cfg(4));
  assert.strictEqual(free.color, pinned.color); assert.notStrictEqual(four.color, free.color);
  assert.deepStrictEqual([free.variant, pinned.variant, four.variant], ['solid', 'hatch', 'solid']);
  assert.strictEqual(api.perf.variants.get(16).length, 2);
  checks += 4;
}

// 4. THE MATRIX holds a square per configuration (none drawn over another in one cell), its columns name the
//    pin where it differs, and no header is a bare "?".
{
  load(today, [rec(M, 'Dense', 'vLLM', today)]);
  api.renderPerfMatrix(api.groupsOf(api.timed(today), []));
  const grid = document.getElementById('pm-grids').innerHTML;
  assert.strictEqual((grid.match(/<button type="button" class="pm-sq/g) || []).length, 3, 'one square per configuration');
  assert.ok(grid.includes('KV 7.92 GiB') && grid.includes('KV not pinned') && grid.includes('vLLM default'), grid.slice(0, 900));
  assert.ok(!/>\?</.test(grid) && !/batched \?/.test(grid), 'a header or title still says "?"');
  checks += 3;
}

// 5. A MODEL THAT ONLY BOOTED has no grid, like it has no chip; every grid is captioned by its recorded name.
{
  assert.ok(api.inventory, 'benchmarks.js has no inventory(): the matrix still takes every boot in the payload');
  const mine = today, theirs = [win(cfg(8, { model_choice: OTHER }), 400)];
  const boots = [{ t_start: 500, t_end: 590, config: cfg(16, { model_choice: BOOTED }), outcome: 'serving' },
    { t_start: 600, t_end: 690, config: cfg(32), outcome: 'failed' }];
  const windows = [...mine, ...theirs];
  load(windows, [rec(M, 'Dense', 'vLLM', mine), rec(OTHER, 'Another', 'SGLang', theirs)], boots);
  const inv = api.inventory(api.perf.data);
  assert.deepStrictEqual(inv.boots.map((b) => b.config.model_choice), [M], "a boot of a model with no minute is not the inventory's");
  assert.strictEqual(inv.ws.length, windows.length);
  api.renderPerfMatrix(api.groupsOf(inv.ws, inv.boots));
  const grid = document.getElementById('pm-grids').innerHTML;
  const captions = [...grid.matchAll(/<span class="pm-who">([^<]*)<\/span>/g)].map((m) => m[1]);
  assert.deepStrictEqual(captions, ['Dense', 'Another'], 'each grid named as /lb/perf `models` names it, never by its id');
  assert.ok(!grid.includes(BOOTED), 'the boot-only model still has a grid');
  checks += 4;
}

// 6. CONTEXT is derived from the minute's OWN recorded pool, whatever the model: a minute without one is not
//    banded, it is counted and the page says why — no pool is assumed, and no model id lives in the page.
{
  const busy = { kv: { mean: 0.5 }, running: { mean: 4 }, prompt_computed_tokens: 600 };
  const none = [win(cfg(16, { model_choice: 'qwen3.8-flash-next-hibrid48' }), 0, busy), win(cfg(16), 60, busy)];
  const out = api.contextRows(none);
  assert.deepStrictEqual([out.rows.length, out.unpooled], [0, 2], JSON.stringify(out));
  const own = api.contextRows([win(cfg(16), 0, { ...busy, kv_pool_tokens: 400000 }), ...none]);
  assert.deepStrictEqual([own.rows.length, own.rows[0].ctx, own.rows[0].minutes, own.unpooled], [1, 50000, 1, 2]);
  api.renderContext(none);
  const section = document.getElementById('perf-context'), panel = document.getElementById('perf-context-panel').innerHTML;
  assert.strictEqual(section.hidden, false, 'minutes with no recorded pool must be SAID, not hidden with the section');
  assert.ok(/2 min/.test(panel) && /KV pool not recorded/.test(panel), panel);
  const catalog = { models: require('./library_rows.js')() };
  const named = catalog.models.map((m) => m.id).filter((id) => src.includes(`'${id}'`) || src.includes(`"${id}"`));
  assert.deepStrictEqual(named, [], 'a catalog model id is written in benchmarks.js');
  assert.ok(!/KV_TOKENS_FALLBACK|KV_FALLBACK_MODEL/.test(src), 'the assumed KV pool is still in the page');
  assert.ok(!/id="ctx-hybrid"/.test(html), 'a note is still switched on by one model id');
  checks += 7;
}
// 7. THE POOLED PANEL ("by max_num_seqs") is a configuration per row too: its series were (spec, batched)
//    pairs, so the 16-seat minutes with and without the pin were one bar and one table row there as well.
{
  load(today, [rec(M, 'Dense', 'vLLM', today)]);
  const runs = today.map(api.winAsRun);
  const series = api.seriesIn(runs);
  assert.strictEqual(series.length, 2, 'the pinned and the unpinned minutes are two series');
  const twin = api.perfTwinHTML(runs, api.capsOf(runs), series, [], (s, cap) => api.cfgStyle({ ...s.cfg, max_num_seqs: cap }));
  const rows = (twin.match(/<tr><td class="num">/g) || []).length;
  assert.strictEqual(rows, 3, 'one table row per (cap, configuration)');
  assert.ok(twin.includes('KV 7.92 GiB') && twin.includes('batched vLLM default') && !twin.includes('batched ?'), twin.slice(0, 600));
  const styles = series.map((s) => api.cfgStyle({ ...s.cfg, max_num_seqs: 16 }).variant);
  assert.deepStrictEqual(styles, ['solid', 'hatch'], 'the panel wears the section\'s encoding: the cap\'s blue, the second hatched');
  checks += 4;
}
// 8. THE CONTEXT TABLE SCOPES TO THE SELECTED CONFIGURATION (operator 2026-10-22: the aggregate read "wedged"
//    because it was not per context — the band table pooled a model's every configuration together. Picking a
//    square now bands THAT configuration's minutes alone, so its per-context throughput is read apart from the
//    others; nothing selected is the whole model, as before. Reaggregated in the browser from the raw per-minute
//    log, so every historical minute re-bands on load — nothing stored is cleared).
{
  const pin = { kv_cache_memory: 28e9 }, pool = 400000;           // ctx/req = kv.mean × pool ÷ running.mean
  const short = { kv: { mean: 0.1 }, running: { mean: 50 }, kv_pool_tokens: pool };   //    800 ctx → band 0–8k
  const long = { kv: { mean: 0.8 }, running: { mean: 4 }, kv_pool_tokens: pool };     // 80,000 ctx → band 80k+
  const wsA = [win(cfg(8, pin), 0, short), win(cfg(8, pin), 60, short)];
  const wsB = [win(cfg(16, pin), 120, long), win(cfg(16, pin), 180, long)];
  const all = [...wsA, ...wsB];
  load(all, [rec(M, 'Dense', 'vLLM', all)]);
  api.perf.ctxWs = all;
  assert.strictEqual(api.contextRows(all).rows.length, 2, 'the model spans two context bands');
  api.renderPerfMatrix(api.groupsOf(api.timed(all), []));
  const rowheads = (h) => (h.match(/role="rowheader"/g) || []).length;
  api.pmSelect(null);                                              // nothing selected: the whole model, both bands
  assert.strictEqual(rowheads(document.getElementById('perf-context-panel').innerHTML), 2, 'unscoped: every band of the model');
  const gi = api.pm.groups.findIndex((g) => api.gcfg(g).max_num_seqs === 8);   // select the 8-seat configuration
  api.pmSelect(gi);
  const panel = document.getElementById('perf-context-panel').innerHTML;
  assert.strictEqual(rowheads(panel), 1, 'scoped to the selected configuration: only its context bands (one here)');
  assert.ok(/8 seqs/.test(panel), `the scoped context table names the configuration:\n${panel.slice(0, 300)}`);
  checks += 4;
}
console.log(`ok — ${checks} checks`);
