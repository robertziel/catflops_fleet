/* Strings the panel did not write itself reach the page ESCAPED, in text and in attributes: gateway.js `tag` (a model
 * row's chips, whose `extra` class is the library's engine key) and `ceilingMeter` (a compute meter's note, whose
 * precision is the hardware library's). Both are library files a fleet can edit or extend; a quote or a tag in one
 * must stay text, never break out of its attribute or open an element.
 *
 *   docker run --rm -v "$PWD":/repo:ro -w /repo/lb node:22-slim node tests/escape_sinks.node.js
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
  `${src}\n;return { tag, ceilingMeter };`)(
  document, window, () => new Promise(() => {}), { search: '', href: '' }, () => 0, () => 0,
  () => ({}), undefined, window.localStorage, { userAgent: 'node' });

const HOSTILE = '"><img src=x onerror=alert(1)>';

// 1. a chip's class from the library: the quote cannot close the class attribute.
const chip = api.tag('engine', 'SGLang', HOSTILE);
assert.ok(!chip.includes('<img'), `a class value opened an element: ${chip}`);
assert.strictEqual((chip.match(/"/g) || []).length, 2, `a class value broke out of its attribute: ${chip}`);

// 2. a compute meter's precision from the hardware library: text, never markup.
const meter = api.ceilingMeter('COMPUTE', { rate: 10, pct: 5, peak: 200, peak_precision: HOSTILE }, 'TFLOP/s', 1,
  [], 'compute', null);
assert.ok(!meter.includes('<img'), `a precision opened an element: ${meter}`);
assert.ok(meter.includes('&lt;img'), 'the precision is still shown, as text');

console.log('escape_sinks: ok');
