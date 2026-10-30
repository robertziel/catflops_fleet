/* THE SERVER TABS on the page (vllm_lb/static/clusters.js; worklog/parallel-clusters.md, "Servers, not clusters").
 * Operator (2026-10-17): "Update that to servers not clusters … Servers tab should not open separate view. Should only
 * switch server view." — ONE TAB PER SERVER in this run (its name, a state dot, the default one marked) and "+"; a server
 * tab only SELECTS that server (gateway.js draws it: card, engine, requests); "+" is the one tab with its own panel.
 *
 *   1. a WAI-ARIA tablist: role=tab buttons, aria-selected, roving tabindex, arrow keys / Home / End, the hash;
 *   2. the tabs are KEYED by the run's names: a poll that changes nothing leaves the buttons (and the focus) alone;
 *      a changed state repaints the dot only;
 *   3. selecting a tab selects the server and asks gateway.js to draw it (`renderStatus`), on the page — never a panel;
 *   4. the + tab lists what the fleet declares: Activate / the blocker's words / Deactivate for an off one / cannot launch;
 *   5. words say "server", never "cluster", to the operator; "Cluster" is the hardware section (index.html);
 *   6. words are escaped; nothing names a box's kind (tests/test_hardware_guard.py).
 *
 *   node tests/server_tabs.node.js   (tests/test_server_tabs_page.py runs it)
 */
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const STATIC = path.join(__dirname, '..', 'vllm_lb', 'static');
const src = fs.readFileSync(path.join(STATIC, 'clusters.js'), 'utf8');
const els = new Map();
const el = (id) => {
  const attrs = {};
  return { id, hidden: false, innerHTML: '', textContent: '', className: '', dataset: {}, tabIndex: -1, disabled: false,
    focused: 0, listeners: {}, children: [],
    addEventListener(k, f) { this.listeners[k] = f; }, setAttribute(k, v) { attrs[k] = String(v); },
    getAttribute(k) { return attrs[k]; }, focus() { this.focused += 1; }, scrollIntoView() {},
    querySelectorAll() { return []; } };
};
const byId = (id) => { if (!els.has(id)) els.set(id, el(id)); return els.get(id); };
const posted = [];
let answer = { ok: true, status: 200, body: {} };
const fetchImpl = (url, opts) => { posted.push([url, (opts || {}).method, (opts || {}).body]);
  const a = answer; return Promise.resolve({ ok: a.ok, status: a.status, json: async () => a.body }); };
const location = { hash: '', pathname: '/' };
const history = { replaceState(_, __, url) { location.hash = url.startsWith('#') ? url : ''; } };
const document = { getElementById: byId, addEventListener() {}, body: { dataset: {} } };
// gateway.js, as clusters.js meets it on the page: the status it last drew, and the one entry that redraws the page.
let drawn = 0;
const lastStatus = { clusters: {} };
const renderStatus = () => { drawn += 1; };
const api = new Function('document', 'fetch', 'location', 'history', 'lastStatus', 'renderStatus',
  `${src}\n;return { renderClusters, declaredInner, selectTab, tabKey, postRunAction, renderAdd, tabIds: () => clTabIds.slice(),
    selected: () => selectedCluster };`)(document, fetchImpl, location, history, lastStatus, renderStatus);

let checks = 0;
const ok = (cond, msg) => { assert.ok(cond, msg); checks += 1; };

const pair = { name: 'pair', configured: true, state: 'running', default: true, held: true, detail: 'TP=2 pair up',
  topology: { tp: 2, boxes: ['box-a', 'box-b'], control: 'ssh', why: '', steps: [] },
  desired: { mode: 'cluster' }, loaded_model: { label: 'Model A', served_name: 'org/model-a' }, switch: '/lb/clusters/pair' };
const node = { name: 'node', configured: true, state: 'off', default: false, held: false, detail: 'off',
  topology: { tp: 1, boxes: ['box-c'], control: 'local', why: '', steps: [] },
  desired: { mode: 'off', model: 'model-b' }, loaded_model: null, switch: '/lb/clusters/node' };
const declared = [
  { name: 'pair', control: 'ssh', boxes: ['box-a', 'box-b'], tp: 2, model: null, boot: true, active: true, blocked_by: [], available: false, default: true },
  { name: 'node', control: 'local', boxes: ['box-c'], tp: 1, model: 'model-b', boot: false, active: true, blocked_by: [], available: false },
  { name: 'alt', control: 'ssh', boxes: ['box-a', 'box-b'], tp: 2, model: null, boot: false, active: false,
    blocked_by: [{ box: 'box-a', cluster: 'pair' }, { box: 'box-b', cluster: 'pair' }], available: false },
  { name: 'spare', control: 'local', boxes: ['box-e'], model: 'model-c', boot: false, active: false, blocked_by: [], available: true },
];
const status = { clusters: { pair, node }, declared, backends: [] };

// 1 + 2. one tab per server, keyed; the + tab last
{
  const list = byId('tablist');
  api.renderClusters(status);
  assert.deepStrictEqual(api.tabIds(), ['tab-btn-s-pair', 'tab-btn-s-node', 'tab-btn-add']); checks += 1;
  ok((list.innerHTML.match(/role="tab"/g) || []).length === 2 && !/tab-btn-fleet|tab-btn-clusters|Fleet|Clusters/.test(list.innerHTML),
    `two server tabs rendered into the list, no Fleet or Clusters tab: ${list.innerHTML}`);
  const tPair = byId('tab-btn-s-pair'), tNode = byId('tab-btn-s-node'), add = byId('tab-btn-add');
  ok(/class="diode ok"/.test(list.innerHTML) && /class="diode off"/.test(list.innerHTML), 'a state dot per tab: running ok, off off');
  ok(/<span class="cl-default"[^>]*>default<\/span>/.test(list.innerHTML) && (list.innerHTML.match(/cl-default/g) || []).length === 1,
    'the default server is marked, once');
  ok(/data-server="pair"/.test(list.innerHTML) && /aria-controls="cluster-panel"/.test(list.innerHTML), 'a server tab controls the one server panel');
  ok(tPair.getAttribute('aria-selected') === 'true' && tNode.getAttribute('aria-selected') === 'false' && add.getAttribute('aria-selected') === 'false',
    'the first server is selected at load (no hash)');
  ok(tPair.tabIndex === 0 && tNode.tabIndex === -1 && add.tabIndex === -1, 'roving tabindex');
  ok(api.selected() === 'pair' && document.body.dataset.tab === 'server', 'the selection is the first server; the page shows a server');
  const before = list.innerHTML;
  api.renderClusters(status);
  ok(list.innerHTML === before, 'a poll with the same names and states rewrites nothing');
  api.renderClusters({ ...status, clusters: { pair, node: { ...node, state: 'starting' } } });
  ok(list.innerHTML === before && /diode warn/.test(tNode.innerHTML) && tNode.dataset.state === 'starting',
    'a state change repaints that tab\'s dot, not the list');
  ok(byId('tab-n-add').textContent === '2' && /not in this run: 2/.test(byId('tab-btn-add').title),
    'the + tab counts the declared servers NOT in this run (alt, spare), and says so');
}

// 3. a server tab selects; the + tab is the one panel; keys and hash
{
  const tPair = byId('tab-btn-s-pair'), tNode = byId('tab-btn-s-node'), add = byId('tab-btn-add');
  const was = drawn;
  api.selectTab('node', false);
  ok(api.selected() === 'node' && tNode.getAttribute('aria-selected') === 'true' && tPair.getAttribute('aria-selected') === 'false', 'the node tab selects the node');
  ok(drawn === was + 1, 'selecting asks gateway.js to draw the page for it (renderStatus)');
  ok(document.body.dataset.tab === 'server' && byId('tab-add').hidden === true, 'no panel opened: the page is the server');
  ok(location.hash === '#node', 'the hash keeps the selection');
  api.selectTab('add', false);
  ok(add.getAttribute('aria-selected') === 'true' && byId('tab-add').hidden === false && document.body.dataset.tab === 'add', 'the + tab opens its panel');
  ok(api.selected() === 'node' && location.hash === '#add', 'the selection is kept behind the + tab; the hash names the + tab');
  api.tabKey({ key: 'ArrowRight', target: add, preventDefault() {} });
  ok(tPair.getAttribute('aria-selected') === 'true' && tPair.focused === 1, 'ArrowRight wraps to the first server and focuses it');
  api.tabKey({ key: 'End', target: tPair, preventDefault() {} });
  ok(add.getAttribute('aria-selected') === 'true', 'End: the + tab');
  api.tabKey({ key: 'ArrowLeft', target: add, preventDefault() {} });
  ok(tNode.getAttribute('aria-selected') === 'true' && api.selected() === 'node', 'ArrowLeft: the server before it');
  api.tabKey({ key: 'Home', target: tNode, preventDefault() {} });
  ok(tPair.getAttribute('aria-selected') === 'true' && api.selected() === 'pair' && location.hash === '#pair', 'Home: the first server');
  api.selectTab('nobody', false);
  ok(api.selected() === 'pair' && tPair.getAttribute('aria-selected') === 'true', 'an unknown name selects nothing new');
  // a server leaving the run: its tab goes, the selection falls back to the first
  api.selectTab('node', false);
  api.renderClusters({ ...status, clusters: { pair } });
  assert.deepStrictEqual(api.tabIds(), ['tab-btn-s-pair', 'tab-btn-add']); checks += 1;
  ok(byId('tab-btn-s-pair').getAttribute('aria-selected') === 'true', 'the one remaining server is the selected tab');
  api.renderClusters({ clusters: {}, declared: [] });
  assert.deepStrictEqual(api.tabIds(), ['tab-btn-add']); checks += 1;
  ok(byId('add-empty').hidden === false, 'no server declared: the + tab says so');
  api.renderClusters(status);
}

// 4. the + tab
{
  const panel = byId('add-panel');
  ok(panel.innerHTML.split('<article').length - 1 === 4, 'every declared server is listed');
  const alt = api.declaredInner(declared[2], status), spare = api.declaredInner(declared[3], status);
  const live = api.declaredInner(declared[0], status), offNode = api.declaredInner(declared[1], status);
  ok(/blocked · box-a \+ box-b → pair/.test(alt) && !/<button/.test(alt), 'a blocked one says which boxes, grouped by who holds them, no button');
  ok(/TP=2 on box-a \+ box-b/.test(live) && !/TP=/.test(spare), "TP is the API's own; one that states none prints none");
  ok(/<button class="cl-act join"[^>]*data-act="activate" data-name="spare"[^>]*>Activate<\/button>/.test(spare), 'an available one offers Activate');
  ok(/in this run · running/.test(live) && !/data-act="deactivate"/.test(live), 'a running active one: no Deactivate (the state says why)');
  ok(/data-act="deactivate" data-name="node"(?![^>]*disabled)/.test(offNode), 'an off active one may be deactivated');
  ok(/at boot/.test(live) && !/at boot/.test(spare), 'the boot servers are marked');
  ok(/class="cl-default"/.test(live) && !/cl-default/.test(spare), 'the default server wears the chip on its + row');
  const amd = api.declaredInner({ name: 'igpu', control: 'local', boxes: ['box-f'], tp: 1, model: 'm', boot: false,
    active: false, available: false, blocked_by: [], holding: false, unlaunchable: 'a type states no visible_gib' }, status);
  ok(/cl-blocked[^>]*>cannot launch · a type states no visible_gib/.test(amd) && !/<button/.test(amd), 'an unlaunchable one: the reason, no button');
  const stray = api.declaredInner(Object.assign({}, declared[3], { available: false, holding: true }), status);
  ok(/not in this run · its switch is on/.test(stray), 'one outside the run that still holds its boxes says so');
  answer = { ok: false, status: 409, body: { error: { type: 'cluster_blocked', message: 'cluster \'spare\' cannot run now: box box-e is held by cluster other' } } };
  return api.postRunAction('spare', 'activate').then(() => {
    assert.deepStrictEqual(posted[posted.length - 1], ['/lb/clusters/spare/activate', 'POST', '{}']); checks += 1;
    const after = api.declaredInner(declared[3], status);
    ok(/<p class="cl-error" role="status">cluster &#39;spare&#39; cannot run now: box box-e is held by cluster other<\/p>/.test(after),
      'the refusal is on the card, in the API\'s words, escaped');
    const card = byId('add-spare');
    const before = card.innerHTML;
    api.renderAdd(status);
    ok(card.innerHTML === before, 'unchanged data: the card is left alone');
    answer = { ok: true, status: 200, body: { active: ['pair', 'node', 'spare'] } };
    return api.postRunAction('spare', 'activate');
  }).then(() => {
    ok(!/cl-error" role="status">cluster/.test(api.declaredInner(declared[3], status)), 'the next action clears the refusal');
    // 5 + 6. the operator reads "server"; the hostile name is escaped; no hardware word
    const hostile = api.declaredInner(Object.assign({}, declared[3], { name: '<img src=x onerror=1>' }), status);
    ok(!/<img/.test(hostile) && /&lt;img/.test(hostile), 'escaped');
    const words = src.replace(/\/\*[\s\S]*?\*\/|\/\/.*$/gm, '');
    const shown = [...words.matchAll(/(?:aria-label|title)="([^"$]*)"|>([^<`$]*cluster[^<`$]*)</gi)].map((m) => m[1] || m[2]).filter(Boolean);
    const said = shown.filter((t) => /cluster/i.test(t));
    ok(!said.length, `the operator reads "server", never "cluster": ${said.join(' | ')}`);
    ok(!/unified|discrete|gb10|rtx|radeon|box-a/i.test(src), 'no hardware word in the tabs source');
    const names = (code) => new Set([...code.matchAll(/^(?:async )?(?:function|const|let|class) ([A-Za-z_$][\w$]*)/gm)].map((m) => m[1]));
    const gw = names(fs.readFileSync(path.join(STATIC, 'gateway.js'), 'utf8'));
    const clash = [...names(src)].filter((n) => gw.has(n));
    ok(!clash.length, `top-level names clusters.js shares with gateway.js: ${clash.join(', ')}`);
    console.log(`ok ${checks} checks`);
  });
}
