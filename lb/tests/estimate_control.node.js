'use strict';
/* THE ESTIMATE CONTROL (docs/design/engines.md, "Unmeasured combination"): a model never measured on the cluster's
 * cards is REFUSED by default; the panel offers the operator ONE explicit choice to start it sized by the estimate.
 *   1. a measured model shows no control, and its picker row says nothing of an estimate;
 *   2. an unmeasured one shows the control UNCHECKED, in the server's own words, and its row says "estimate only";
 *   3. a new pick unchecks it — the choice is the operator's each time, never carried over;
 *   4. `estimatedChosen` is true only when the control is shown AND checked (what toggleCluster sends).
 * Run: docker run --rm -v "$PWD":/repo:ro -w /repo/lb node:22-slim node tests/estimate_control.node.js */
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const src = fs.readFileSync(path.join(__dirname, '..', 'vllm_lb', 'static', 'gateway.js'), 'utf8');
const html = fs.readFileSync(path.join(__dirname, '..', 'vllm_lb', 'static', 'index.html'), 'utf8');
assert.ok(/id="cluster-estimated-row"[^>]*hidden/.test(html), 'the control ships hidden');
assert.ok(/id="cluster-estimated" type="checkbox"/.test(html), 'the control is a checkbox');

const els = {};
const el = (id) => (els[id] = els[id] || { id, hidden: true, checked: false, textContent: '', dataset: {},
  addEventListener() {}, classList: { toggle() {}, add() {}, remove() {} }, style: {}, setAttribute() {},
  removeAttribute() {}, querySelectorAll: () => [], querySelector: () => null, innerHTML: '', value: '' });
const document = { getElementById: el, createElementNS: () => el('x'), createElement: () => el('y'), addEventListener() {},
  querySelectorAll: () => [], querySelector: () => null, body: { style: {}, classList: { toggle() {} } },
  documentElement: { style: {}, classList: { toggle() {} } } };
const window = { addEventListener() {}, innerWidth: 1280, innerHeight: 800, matchMedia: () => ({
  matches: false, addEventListener() {} }), localStorage: { getItem: () => null, setItem() {} } };
const api = new Function('document', 'window', 'fetch', 'location', 'setInterval', 'setTimeout',
  'getComputedStyle', 'ResizeObserver', 'localStorage', 'navigator',
  `${src}\n;return { renderEstimated, estimatedChosen, pickerRowText };`)(
  document, window, () => new Promise(() => {}), { search: '', href: '' }, () => 0, () => 0,
  () => ({}), undefined, window.localStorage, { userAgent: 'node' });

const res = (sizing) => ({ sizing, refused: sizing === 'unmeasured' ? 'not measured on acc-x at TP=4 with vllm' : '',
  estimate: sizing === 'unmeasured' ? 'ESTIMATED, NOT MEASURED: weights 24.5 GiB a rank' : '' });
const model = (id, sizing) => ({ id, label: id, launch: { cluster: { engine: 'vllm', resolution: res(sizing) } } });
const rows = [{ value: 'measured', m: model('measured', 'measured') }, { value: 'new', m: model('new', 'unmeasured') }];
let checks = 0;

// 1.
api.renderEstimated(rows, 'measured');
assert.strictEqual(els['cluster-estimated-row'].hidden, true);
assert.ok(!/estimate/.test(api.pickerRowText(rows[0].m, false, null, null, false)));
checks += 2;
// 2.
api.renderEstimated(rows, 'new');
assert.strictEqual(els['cluster-estimated-row'].hidden, false);
assert.strictEqual(els['cluster-estimated'].checked, false, 'never pre-checked: the LB chooses nothing for anyone');
assert.ok(/not measured on acc-x at TP=4/.test(els['cluster-estimated-why'].textContent));
assert.ok(/ESTIMATED, NOT MEASURED/.test(els['cluster-estimated-why'].textContent));
assert.ok(/not measured here · estimate only/.test(api.pickerRowText(rows[1].m, false, null, null, false)));
assert.strictEqual(api.estimatedChosen(), false);
checks += 6;
// 4.
els['cluster-estimated'].checked = true;
api.renderEstimated(rows, 'new');
assert.strictEqual(api.estimatedChosen(), true, 'shown and checked: the Start carries estimated: true');
checks += 1;
// 3.
api.renderEstimated(rows, 'measured');
assert.strictEqual(api.estimatedChosen(), false, 'hidden: nothing is sent');
api.renderEstimated(rows, 'new');
assert.strictEqual(els['cluster-estimated'].checked, false, 'a new pick unchecks it');
checks += 2;
console.log(`estimate_control: ${checks} checks passed`);
