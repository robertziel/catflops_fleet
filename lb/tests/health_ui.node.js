/* THE HEALTH BAND AND THE ACTIVITY FEED (vllm_lb/static/health.js) — what the page says, from the shapes
 * /lb/status `health` (vllm_lb/health.py `status_view`) and GET /lb/events (vllm_lb/events.py) carry.
 *
 *   docker run --rm -v "$PWD":/app:ro -w /app node:22-slim node tests/health_ui.node.js
 */
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');
const STATIC = path.join(__dirname, '..', 'vllm_lb', 'static');
const h = require(path.join(STATIC, 'health.js'));

// ages: whole units, never negative
assert.strictEqual(h.healthAge(40), '40s');
assert.strictEqual(h.healthAge(185), '3m');
assert.strictEqual(h.healthAge(2 * 3600 + 5 * 60), '2h 5m');
assert.strictEqual(h.healthAge(3 * 86400 + 4 * 3600), '3d 4h');
assert.strictEqual(h.healthAge(-5), '0s');

const now = 10000;
const temp = { id: 'solo:gpu_temp', box: 'solo', kind: 'gpu_temp', label: 'GPU TEMP', unit: '°C', state: 'firing',
               since: now - 300, firing_since: now - 210, value: 3, threshold: 5,
               why: 'GPU 3 °C below its own throttle limit ≤ margin 5 °C' };
const ram = { id: 'solo:low_ram', box: 'solo', kind: 'low_ram', label: 'RAM AVAIL', unit: 'GiB', state: 'pending',
              since: now - 40, firing_since: null, value: 2.5, threshold: 4, why: '' };
const xid = { id: 'n1:xid', box: 'n1', kind: 'xid', label: 'XID', unit: 'new', state: 'firing', since: now - 60,
              firing_since: now - 60, value: 2, threshold: 0, why: '2 new' };
const down = { id: 'a-gpu:backend_down', box: 'a-gpu', kind: 'backend_down', label: 'SERVER', unit: '', state: 'pending',
               since: now - 10, firing_since: null, value: 'not live', threshold: 'live', why: '' };

// the figure against the threshold, in the finding's unit; the age from when it FIRED (pending: from its start)
const chips = h.healthChips({ findings: [temp, ram, xid, down], alerts: { channels: [], recent_failures: [] } }, now);
assert.deepStrictEqual(chips.map((c) => c.text), [
  'FIRING · solo · GPU TEMP 3 °C to limit ≤ 5 °C · 3m',
  'PENDING · solo · RAM AVAIL 2.5 GiB < 4 GiB · 40s',
  'FIRING · n1 · XID +2 · 1m',
  'PENDING · a-gpu · SERVER not live · 10s',
]);
assert.deepStrictEqual(chips.map((c) => c.cls), ['hl-firing', 'hl-pending', 'hl-firing', 'hl-pending']);
assert.strictEqual(chips[0].title, temp.why);
// A SERVER finding is keyed by the server's lane label (health.py: backend.host_label); with the status's backends
// the chip names its BOX, as every other chip does (the UX audit: "box-c-gpu" beside "box-c").
const named = h.healthChips({ findings: [down], alerts: { recent_failures: [] } }, now,
  [{ host_label: 'a-gpu', box: 'box-a', kind: 'vllm' }, { host_label: null, box: null, kind: 'cluster' }]);
assert.deepStrictEqual(named.map((c) => c.text), ['PENDING · box-a · SERVER not live · 10s']);

// GPU POWER: at or under the card type's idle draw + the fleet's margin (health.py low_power)
assert.strictEqual(h.findingFigure({ kind: 'low_power', label: 'GPU POWER', unit: 'W', value: 9.5, threshold: 10.9 }),
  'GPU POWER 9.5 W ≤ 10.9 W');

// nothing pending or firing, no recent failed send: no chip (the band hides)
assert.deepStrictEqual(h.healthChips({ findings: [], alerts: { recent_failures: [] } }, now), []);
assert.deepStrictEqual(h.healthChips(undefined, now), []);

// a failed send holds the band for an hour, then only the feed keeps it
const fail = { at: now - 120, channel: 'ntfy', why: 'HTTP 500 Internal Server Error', title: 'FIRING gpu_temp on solo' };
assert.deepStrictEqual(h.healthChips({ findings: [], alerts: { recent_failures: [fail] } }, now).map((c) => c.text),
  ['ALERT ntfy failed · HTTP 500 Internal Server Error · 2m']);
assert.deepStrictEqual(h.healthChips({ findings: [], alerts: { recent_failures: [{ ...fail, at: now - 4000 }] } }, now), []);

// the health line: figures only
assert.strictEqual(h.healthSummary({ enabled: false, why: 'no health: block' }), 'HEALTH off');
assert.strictEqual(h.healthSummary({ enabled: true, findings: [temp, ram], no_data: [{}, {}, {}],
                                     alerts: { channels: [{ name: 'ntfy' }, { name: 'webhook' }], sent: 1234, failed: 1 } }),
  'HEALTH 1 firing · 1 pending · 3 no data · ALERTS ntfy+webhook · 1,234 sent · 1 failed');

// an event row: every string the feed carries is ESCAPED (a summary is an engine's or a box's words)
const row = h.eventRowHtml({ t: 1800000000, kind: 'cluster.state', subject: '<b>pair</b>', state: 'failed',
                             summary: 'starting → failed: "rank" <script>' }, new Date(1800000000 * 1000));
assert.ok(!row.includes('<b>pair') && row.includes('&lt;b&gt;pair'));
assert.ok(!row.includes('<script>') && row.includes('&quot;rank&quot;'));
assert.ok(row.includes('ev-bad'), 'a failed cluster is drawn in the fault tone');
assert.ok(/\d\d:\d\d:\d\d/.test(row), 'today: the time of day');
assert.strictEqual(h.eventTone({ kind: 'cluster.state', state: 'running' }), 'ok');
assert.strictEqual(h.eventTone({ kind: 'finding.cleared' }), 'ok');
assert.strictEqual(h.eventTone({ kind: 'alert.failed' }), 'bad');
assert.strictEqual(h.eventTone({ kind: 'request.killed' }), 'warn');
assert.ok(/^[A-Z][a-z]{2} \d{1,2} \d\d:\d\d$/.test(h.eventTime(1800000000, new Date((1800000000 + 3 * 86400) * 1000))),
  'another day: month, day and time');

// the chips ask for the kinds of their families; none = every kind
assert.strictEqual(h.selectedKinds(new Set()), null);
assert.deepStrictEqual(h.selectedKinds(new Set(['finding', 'kill'])), ['request.killed', 'finding.firing', 'finding.cleared']);
// every kind the feed records has a chip (events.py KINDS)
const kinds = fs.readFileSync(path.join(__dirname, '..', 'vllm_lb', 'events.py'), 'utf8')
  .match(/KINDS = frozenset\(\(([\s\S]*?)\)\)/)[1].match(/"([a-z.]+)"/g).map((s) => s.slice(1, -1));
assert.deepStrictEqual(h.EVENT_GROUPS.flatMap(([, ks]) => ks).sort(), kinds.sort());

// the page wires it: the band host, the tile with its layout id, the script before gateway.js, the hook in renderStatus
const html = fs.readFileSync(path.join(STATIC, 'index.html'), 'utf8');
assert.ok(html.includes('id="lb-health"') && html.includes('data-ly="events"'));
assert.ok(html.indexOf('/health.js') < html.indexOf('/gateway.js') && html.includes('/health.css'));
assert.ok(/function renderStatus[\s\S]*?renderHealth\(s\)/.test(fs.readFileSync(path.join(STATIC, 'gateway.js'), 'utf8')));
// a phone: the summary takes its own full-width line
assert.ok(/@media \(max-width: 600px\)[\s\S]*\.ev-m \{ grid-column: 1 \/ -1/.test(fs.readFileSync(path.join(STATIC, 'health.css'), 'utf8')));

console.log('ok');
