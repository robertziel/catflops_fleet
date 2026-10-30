/* THREE LINES THE PAGE PUBLISHES BUT NEVER SHOWS — checked where the page builds them, under node.
 *
 * All three are in vllm_lb/static/gateway.js, the file :8010 serves (its md5 equals the bytes
 * `curl :8010/gateway.js` returns; dashboard/public/gateway.js is a 176-line app with no container
 * running and no part in this page).
 *
 *  1. `swStateLines`'s `sub()` opened `class="` and never closed it, so a browser's attribute value ran
 *     on past the `>`. With no title the whole LINE is swallowed; with one the title is lost. Every
 *     swarm sub-line, every load.
 *  2. `clusterFoot` read `boxes[*].parked_by`. The wire field is `park`
 *     (cluster_loop.py:661 `"park": o.parked_by`), so the read was `undefined` and "held by …" could
 *     never appear. Behind that, `reservedForCluster` tests `.status` on what is a STRING server-side
 *     (cluster.py:644). A park is the OPERATOR's — "`requested_by` of an OFF `worker_controls` row …
 *     READ for the words, never acted on" — so there is no cluster-park state to tell apart; what was
 *     missing is the operator's own words.
 *  3. `throughput.ours.uncounted` had one call site, inside `lastRequestTile`, which `throughputTiles`
 *     reaches only when `sourceOf(b) === 'proxy'`. The cluster's source is `'metrics'`, so for the only
 *     backend the fleet serves from, the count of turns the LB FAILED to account for had no path to the
 *     page at all.
 *
 * No DOM: gateway.js is loaded with a stub `document` and a `fetch` that never answers.
 *
 *   node tests/gateway_cluster_lines.node.js   (tests/test_gateway_cluster_lines.py runs this when node exists)
 */
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const src = fs.readFileSync(path.join(__dirname, '..', 'vllm_lb', 'static', 'gateway.js'), 'utf8');
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
  `${src}\n;return { swStateLines, clusterFoot, engineStats };`)(
  document, window, () => new Promise(() => {}), { search: '', href: '' }, () => 0, () => 0,
  () => ({}), undefined, window.localStorage, { userAgent: 'node' });

/* A BROWSER'S OWN READING of one start tag and the text after it: an attribute value opened with a
 * quote runs to the NEXT quote, `>` included — which is the whole bug in 1. Returns the tag's
 * attributes and the text up to the following `<`. */
function readTag(html) {
  assert.ok(html.startsWith('<'), `not a tag: ${html.slice(0, 40)}`);
  let i = html.indexOf(' ') + 1;
  const attrs = {};
  while (i < html.length && html[i] !== '>') {
    if (html[i] === ' ') { i++; continue; }
    let name = '';
    while (i < html.length && !'= >'.includes(html[i])) name += html[i++];
    if (html[i] !== '=') { attrs[name] = null; continue; }
    i++;
    if (html[i] === '"') {                       // runs to the next quote, whatever is in between
      const end = html.indexOf('"', ++i);
      attrs[name] = html.slice(i, end === -1 ? html.length : end);
      i = end === -1 ? html.length : end + 1;
    } else {
      let v = '';
      while (i < html.length && !' >'.includes(html[i])) v += html[i++];
      attrs[name] = v;
    }
  }
  const rest = html.slice(i + 1);
  return { attrs, text: rest.slice(0, rest.indexOf('<') === -1 ? rest.length : rest.indexOf('<')) };
}

// ── 1. the swarm sub-lines ───────────────────────────────────────────────────────────────────────
// /lb/status `swarm` as the live LB served it at 2026-10-02T20:48Z, session S-20260101-000002.
const NOW = 1790974856.75;
const LIVE = { runner: { listening: true, session: {
  alive: true, pid: 3025857, since: 1790974076.08, label: 'S-20260101-000002', phase: 'running',
  cap_min: 15.0, claude_pid: 3025861, started_at: 1790974076.11, ended_at: null, rc: null,
  detail: 'a deep-research swarm is under way on http://127.0.0.1:8010' } } };
const PENDING = { request_pending: true, request: { at: NOW - 780 }, runner: { listening: false } };

{
  const html = api.swStateLines(LIVE, NOW);
  assert.ok(html.startsWith('<span'), `no line at all: ${JSON.stringify(html)}`);
  const got = readTag(html);
  assert.strictEqual(got.attrs.class, 'sw-sub',
    `the class attribute swallowed the rest of the tag: ${JSON.stringify(got.attrs)}`);
  assert.strictEqual(got.attrs.title, LIVE.runner.session.detail,
    "the session's own sentence is the line's title");
  assert.ok(html.includes('S-20260101-000002'), 'the label is in the line');
}
{
  // No title: this is where the browser swallows the TEXT, not just the tooltip.
  const html = api.swStateLines(PENDING, NOW);
  const got = readTag(html);
  assert.strictEqual(got.attrs.class, 'sw-sub sw-warn', `class ran on: ${JSON.stringify(got.attrs)}`);
  assert.ok(got.text.includes('requested'), `the line's own text never rendered: ${JSON.stringify(got.text)}`);
  assert.ok(html.includes('no runner listening'), html);
}

// ── 2. who holds the pair, and whose park it is ──────────────────────────────────────────────────
// `boxes` as /lb/status carries them: `park` is a STRING (or null), `reservation` an object.
const HELD = { boxes: {
  'box-a': { park: 'someone', reservation: { job_id: 'a', status: 'running', claimed_by: 'box-a-gpu' } },
  'box-b': { park: null, reservation: { job_id: 'b', status: 'running', claimed_by: 'box-b-gpu' } } } };
const UNPARKED = { boxes: { 'box-a': { park: null, reservation: null } } };

{
  const foot = api.clusterFoot({ label: 'HELD' }, HELD);
  assert.ok(foot.includes('someone'), `the park's own words never reach the card: ${JSON.stringify(foot)}`);
  assert.strictEqual(api.clusterFoot({ label: 'STARTING' }, HELD), '', 'the badge says STARTING, not this');
  assert.strictEqual(api.clusterFoot({ label: 'HELD' }, UNPARKED), '', 'nothing to say when nothing is parked');
}

// ── 3. the LB's own bookkeeping, for the backend the fleet serves from ───────────────────────────
// backends[2] as the live LB served it: source `metrics`, so the proxy-only tile never drew `ours`.
const CLUSTER_BACKEND = { kind: 'cluster', live: true, running: 1, waiting: 0, depth_known: true,
  throughput: { source: 'metrics', ours: { counted: 61, uncounted: {
    client_gone: 2, no_usage_in_stream: 0, usage_malformed: 0, body_over_cap: 0, error_status: 0,
    error_event: 0, no_terminal_event: 1, parser_error: 0 } } } };

{
  const html = api.engineStats(CLUSTER_BACKEND, false, {});
  assert.ok(html.includes('uncounted'),
    'the turns the LB could not account for have no path to the page for a metrics backend');
  assert.ok(html.includes('>3<') || /uncounted<\/span><b>3/.test(html.replace(/\s+/g, '')),
    `3 turns went uncounted (2 client_gone + 1 no_terminal_event): ${html}`);
  assert.ok(html.includes('client gone') && html.includes('no terminal event'),
    'the closed set of reasons is what makes the number actionable');
}
{
  // Every turn counted is a MEASURED zero, and says so — not a blank.
  const clean = { ...CLUSTER_BACKEND, throughput: { source: 'metrics',
    ours: { counted: 61, uncounted: { client_gone: 0, no_usage_in_stream: 0 } } } };
  const html = api.engineStats(clean, false, {});
  assert.ok(html.includes('uncounted'), 'the figure is drawn when it is zero too');
}

console.log('ok  gateway_cluster_lines: swarm sub-lines parse, the pair names its park, uncounted turns show');
