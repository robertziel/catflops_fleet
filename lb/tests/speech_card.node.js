/* THE SPEECH CARD: a transcription server's card is drawn from its cluster's declared tile groups (`cluster.tiles`,
 * fleet.py NodeEngine) — in flight, audio s/s, × real time, answers/min and its own stats — and NONE of the LLM tiles
 * (generated, prefill, prompt, KV) nor the saturation ceilings; a chat cluster keeps them. Every figure is the engine's
 * own (`throughput.speech`) or "—" with its reason. The card names the card the fleet declares, never the engine's
 * device string.
 *
 *   node tests/speech_card.node.js      (tests/test_speech_card.py runs this when node exists)
 */
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');

// palette.js first: the colours this script computes are read from the style's tokens (tests/palette_env.js).
const src = require('./palette_env').withPalette(fs.readFileSync(path.join(__dirname, '..', 'vllm_lb', 'static', 'gateway.js'), 'utf8'));
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
  `${src}\n;return { serverCard };`)(
  document, window, () => new Promise(() => {}), { search: '', href: '' }, () => 0, () => 0,
  () => ({}), undefined, window.localStorage, { userAgent: 'node' });

const strip = (h) => h.replace(/<[^>]*>/g, ' ').replace(/&middot;/g, '·').replace(/&nbsp;/g, ' ')
  .replace(/\s+/g, ' ').trim();
const labels = (html) => [...html.matchAll(/<span class="t-label">([^<]*)<\/span>/g)].map((m) => m[1]);

const speech = (over) => Object.assign({
  audio_s_per_s: 137.2, audio_why: null, speed_x: 28.4, latency_s: 4.8, speed_why: null,
  per_min: { finished: 12, silence: 1, failed: 0, rejected: 2 }, window_s: 300, why: null,
  seats: 4, queue_capacity: 10, running: 1, waiting: 0, gpu_memory_bytes: 2828525568 + 1417216,
  totals: { audio_seconds: 3464.6, finished: 12, silence: 1, failed: 0, rejected: 2 } }, over || {});
const backend = (sp) => ({ kind: 'cluster', cluster: 'stt', url: 'http://stt-node:8000', live: true, running: 1,
  waiting: 0, inflight: 1, depth_known: true, history: { depth: [0, 1] }, throughput: { speech: sp } });
const view = (tiles, engine) => ({ configured: true, state: 'running', topology: { tp: 1, boxes: ['apu'], control: 'local' },
  tiles, engine, hardware: 'AMD Radeon 8060S (Ryzen AI Max+ 395) · rocm', desired: { mode: 'cluster', max_num_seqs: 4 } });
const status = (c) => ({ now: 1000, clusters: { stt: c }, models: {}, hardware: {} });

const FW = { name: 'faster-whisper', label: 'faster-whisper (CTranslate2)', tiles: ['speech'] };
let html = api.serverCard(status(view(['speech'], FW)), backend(speech()));
assert.deepStrictEqual(labels(html), ['queue', 'audio', 'speed', 'answers']);
const text = strip(html);
for (const want of ['audio 137 s/s', 'speed 28.4 × real time', 'answers 12.0 /min', '1.0 silence · 0 failed · 2.0 rejected',
  'latency 4.8 s', 'GPU memory 2.64 GiB', 'seats 4', 'queue 10', 'faster-whisper (CTranslate2)',
  'AMD Radeon 8060S (Ryzen AI Max+ 395) · rocm']) {
  assert.ok(text.includes(want), `${want} in: ${text}`);
}
for (const llm of ['generated', 'prefill', 'prompt', 'kv']) assert.ok(!labels(html).includes(llm), llm);
assert.ok(!/cuda/i.test(text), 'never the engine\'s own device string');

// An engine that counts no audio (vLLM whisper): the tile says so, never a zero; latency still drawn.
html = api.serverCard(status(view(['speech'], { name: 'vllm', label: 'vLLM', tiles: ['speech'] })),
  backend(speech({ audio_s_per_s: null, audio_why: 'not exposed by this server', speed_x: null,
    speed_why: 'not exposed by this server', seats: null, queue_capacity: null, gpu_memory_bytes: null,
    per_min: { finished: 3, silence: null, failed: null, rejected: null } })));
assert.ok(strip(html).includes('audio — not exposed by this server'), strip(html));
assert.ok(strip(html).includes('GPU memory —'), strip(html));

// A chat cluster keeps the LLM tiles.
html = api.serverCard(status(view(['generated', 'prefill', 'prompt', 'queue', 'kv', 'bandwidth', 'compute', 'saturation'],
  { name: 'vllm', label: 'vLLM', tiles: [] })), Object.assign(backend(speech()), { throughput: {} }));
assert.ok(labels(html).includes('generated') && !labels(html).includes('audio'), labels(html).join());
console.log('ok');
