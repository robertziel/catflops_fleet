/* The gateway page with broker_parrot OFF (LB_BROKER_ENABLED unset), checked where it is drawn: gateway.js.
 *
 * /lb/status then says `"broker": {"enabled": false}` and carries no lease state and no broker DB state
 * (proxy.py `BROKER_STATUS_KEYS`). The page must read that as what it is:
 *   - a held, idle server reads IDLE — not "LEASED · IDLE", there is no lease;
 *   - no db_ok is no broker alarm;
 * and with the broker ON every word stays what it was.
 *
 *   node tests/broker_off.node.js      (tests/test_broker_off_page.py runs this when node exists)
 */
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const src = fs.readFileSync(path.join(__dirname, '..', 'vllm_lb', 'static', 'gateway.js'), 'utf8');
const els = new Map();
const el = () => ({ addEventListener() {}, removeAttribute() {}, setAttribute() {}, append() {},
  classList: { toggle() {}, add() {}, remove() {} }, style: {}, hidden: false, innerHTML: '',
  textContent: '', value: '', dataset: {}, querySelectorAll: () => [], querySelector: () => null,
  focus() {}, observe() {}, closest: () => null, getBoundingClientRect: () => ({}) });
const byId = (id) => { if (!els.has(id)) els.set(id, el()); return els.get(id); };
const document = { getElementById: byId, createElementNS: el, createElement: el, addEventListener() {},
  querySelectorAll: () => [], querySelector: () => null, body: { style: {}, classList: { toggle() {} } },
  documentElement: { style: {}, classList: { toggle() {} } } };
const window = { addEventListener() {}, innerWidth: 1280, innerHeight: 800, matchMedia: () => ({
  matches: false, addEventListener() {} }), localStorage: { getItem: () => null, setItem() {} } };

const api = new Function('document', 'window', 'fetch', 'location', 'setInterval', 'setTimeout',
  'getComputedStyle', 'ResizeObserver', 'localStorage', 'navigator',
  `${src}\n;return { statusOf, renderAlert };`)(
  document, window, () => new Promise(() => {}), { search: '', href: '' }, () => 0, () => 0,
  () => ({}), undefined, window.localStorage, { userAgent: 'node' });

const pair = (over = {}) => ({ kind: 'cluster', cluster: 'pair', url: 'http://10.0.0.1:8000', live: true,
  waiting: 0, running: 0, inflight: 0, depth_known: true, penalised: false, history: {}, ...over });
const status = (broker, over = {}) => ({
  now: 1000, backends: [pair()], leases: [], hardware: {},
  clusters: { pair: { name: 'pair', configured: true, held: true, max_depth: 8, state: 'running' } },
  requests_total: 3, last_tick_error: '', broker: { enabled: broker },
  ...(broker ? { db_ok: true, db_error: '', desired_leases: 1, vllm_capacity: 2 } : {}), ...over });

// ── OFF: a held server is the gateway's own ─────────────────────────────────
{
  const s = status(false);
  const st = api.statusOf(s, s.backends[0]);
  assert.strictEqual(st.label, 'IDLE', `OFF: an idle server is IDLE, not "${st.label}"`);

  api.renderAlert(s);
  assert.strictEqual(byId('lb-alert').hidden, true, 'OFF: no db_ok is not a broker alarm');
}

// ── ON: every word as it was ─────────────────────────────────────────────────
{
  const s = status(true);
  assert.strictEqual(api.statusOf(s, s.backends[0]).label, 'LEASED · IDLE');

  api.renderAlert({ ...s, db_ok: false, db_error: 'engine unavailable' });
  assert.strictEqual(byId('lb-alert').hidden, false);
}

console.log('ok');
