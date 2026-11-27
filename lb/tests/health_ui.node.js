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

// THE BAND, one chip per BOX (eight one-finding chips took 36 % of a phone's first screen): firing boxes
// first, then pending ones; the chip says each finding's FIGURE, the threshold, the age and the why are its title.
const chips = h.healthChips({ findings: [temp, xid, ram, down], alerts: { channels: [], recent_failures: [] } }, now);
assert.deepStrictEqual(chips.map((c) => c.text), [
  'solo · GPU TEMP 3 °C to limit',
  'n1 · XID +2',
  'solo · RAM AVAIL 2.5 GiB',
  'a-gpu · SERVER not live',
]);
assert.deepStrictEqual(chips.map((c) => c.cls), ['hl-firing', 'hl-firing', 'hl-pending', 'hl-pending']);
assert.deepStrictEqual(chips.map((c) => c.box), ['solo', 'n1', 'solo', 'a-gpu']);
assert.deepStrictEqual(chips[0].kinds, ['gpu_temp']);
assert.strictEqual(chips[0].title, `FIRING GPU TEMP 3 °C to limit ≤ 5 °C · 3m · ${temp.why}`);
assert.strictEqual(chips[2].title, 'PENDING RAM AVAIL 2.5 GiB < 4 GiB · 40s');
assert.strictEqual(chips[0].label, 'firing · solo · GPU TEMP 3 °C to limit', 'the state is said, not only coloured');
// two findings of one box and state: ONE chip, both figures, both titles
const pw = { id: 'solo:low_power', box: 'solo', kind: 'low_power', label: 'GPU POWER', unit: 'W', state: 'firing',
             since: now - 29, firing_since: now - 29, value: 9.5, threshold: 10.9, why: '' };
const roce = { id: 'solo:roce', box: 'solo', kind: 'roce', label: 'RoCE ERR', unit: 'new', state: 'firing',
               since: now - 70, firing_since: now - 70, value: 5, threshold: 0, why: 'port_rcv_errors +5' };
const grouped = h.healthChips({ findings: [pw, roce], alerts: { recent_failures: [] } }, now);
assert.deepStrictEqual(grouped.map((c) => c.text), ['solo · GPU POWER 9.5 W · RoCE ERR +5']);
assert.deepStrictEqual(grouped[0].kinds, ['low_power', 'roce']);
assert.strictEqual(grouped[0].title, 'FIRING GPU POWER 9.5 W ≤ 10.9 W · 29s\nFIRING RoCE ERR +5 · 1m · port_rcv_errors +5');
// a pending finding listed before a firing one (health.py sorts firing first; a box's chip is ordered by state anyway)
assert.deepStrictEqual(h.healthChips({ findings: [ram, temp], alerts: { recent_failures: [] } }, now).map((c) => c.cls),
  ['hl-firing', 'hl-pending']);
// A SERVER finding is keyed by the server's lane label (health.py: backend.host_label); with the status's backends
// the chip names its BOX, as every other chip does (the UX audit: "box-c-gpu" beside "box-c").
const named = h.healthChips({ findings: [down], alerts: { recent_failures: [] } }, now,
  [{ host_label: 'a-gpu', box: 'box-a', kind: 'vllm' }, { host_label: null, box: null, kind: 'cluster' }]);
assert.deepStrictEqual(named.map((c) => c.text), ['box-a · SERVER not live']);
assert.strictEqual(named[0].box, 'box-a');

// the machine card's badge: firing findings per box (a SERVER finding counted on its box); pending ones are not
const counts = h.firingByBox({ findings: [temp, ram, xid, pw, { ...down, state: 'firing' }] },
  [{ host_label: 'a-gpu', box: 'box-a' }]);
assert.deepStrictEqual([...counts.entries()].sort(), [['box-a', 1], ['n1', 1], ['solo', 2]]);
assert.strictEqual(h.firingByBox(undefined).size, 0);

// a phone keeps 2 rows of chips: how many lead chips sit in the first N rows, from their offsetTop
assert.strictEqual(h.bandCut([0, 0, 0, 48, 48, 96, 96], 2), 5);
assert.strictEqual(h.bandCut([0, 1, 48], 2), 3, 'a pixel apart is one row');
assert.strictEqual(h.bandCut([0, 48, 96, 144], 2), 2);
assert.strictEqual(h.bandCut([], 2), 0);

// a chip's tap: the box's machine card scrolled to and flashed; its flight recorder opened only for a kind it details
// (recorder.js `recOpen`, optional: absent, the scroll and flash alone)
function fakeCard(box) {
  const cls = new Set(['card', 'machine']);
  return { dataset: { box }, scrolled: null, offsetWidth: 1,
           classList: { add: (c) => cls.add(c), remove: (c) => cls.delete(c), has: (c) => cls.has(c) },
           closest: () => ({ open: false }), scrollIntoView(o) { this.scrolled = o; } };
}
const cards = [fakeCard('n1'), fakeCard('solo')];
const fakeDoc = { querySelectorAll: (sel) => (sel === 'article.card.machine[data-box]' ? cards : []) };
const opened = [];
// `rec` is recorder.js's way in: `open(box)` (recOpen) and `whyNot(box)` (recWhyNot: '' when a strip is drawn).
const rec = (why = '') => ({ open: (b) => opened.push(b),
  whyNot: (b) => (why ? `no recorder strip for ${b}: ${why}` : '') });
// THE STRIP OPENS: ONE scroll, the strip's (recOpen centres its line: the card's flash above it and the strip both on
// screen) — the card does not scroll too (its block:'center' was cancelled by the strip's block:'start').
assert.deepStrictEqual(h.jumpToBox(fakeDoc, 'solo', ['low_power', 'roce'], rec(), () => 0),
  { card: true, strip: 'opened' });
assert.ok(cards[1].scrolled === null && cards[1].classList.has('hl-flash') && !cards[0].scrolled,
  'the strip opened: the card flashed, not scrolled (the strip\'s scroll is the one)');
assert.deepStrictEqual(opened, ['solo']);
assert.deepStrictEqual(h.jumpToBox(fakeDoc, 'n1', ['xid'], rec(), () => 0), { card: true, strip: null });
assert.deepStrictEqual(opened, ['solo'], 'XID is not the recorder strip\'s: not opened');
assert.deepStrictEqual(cards[0].scrolled, { behavior: 'smooth', block: 'center' },
  'no strip to open: the card scrolls');
// NO STRIP DRAWN for the box (recOpen threw, uncaught, in the click handler): the card's scroll and flash,
// and the reason back for the chip's title — never a throw, never a silent swallow.
cards[1].scrolled = null;
const throwing = { open: () => { throw new Error('recOpen must not be called'); },
  whyNot: rec('the box sends neither').whyNot };
assert.deepStrictEqual(h.jumpToBox(fakeDoc, 'solo', ['roce'], throwing, () => 0),
  { card: true, strip: 'no recorder strip for solo: the box sends neither' });
assert.ok(cards[1].scrolled && cards[1].scrolled.block === 'center' && cards[1].classList.has('hl-flash'));
assert.deepStrictEqual(h.jumpToBox(fakeDoc, 'solo', ['roce'], null, () => 0),
  { card: true, strip: 'no recorder strip for solo: recorder.js is not loaded' },
  'no recorder.js: said, scroll + flash');
assert.deepStrictEqual(h.jumpToBox(fakeDoc, 'gone', ['roce'], rec(), () => 0), { card: false, strip: null });
// the chip's title carries the reason, once however often it is tapped
const chipEl = { title: 'FIRING RoCE ERR +5' };
h.chipSays(chipEl, 'no recorder strip for solo: x');
h.chipSays(chipEl, 'no recorder strip for solo: x');
assert.strictEqual(chipEl.title, 'FIRING RoCE ERR +5\nno recorder strip for solo: x');
// THE REASON OUTLIVES A REDRAW: a tap's "no recorder strip" reason is kept per box (`bandSay`) and
// written into the chip's title by every renderHealthBand — the full redraw and the in-place title refresh alike —
// until that box has a strip again (recorder.js recWhyNot → '').
{
  const chipEls = [];
  const band = { hidden: false, innerHTML: '', addEventListener() {}, querySelector: () => null,
    querySelectorAll: (sel) => (sel === '.hl-chip' ? chipEls : []) };
  const g = globalThis;
  const saved = [g.document, g.window, g.recWhyNot];
  g.document = { getElementById: (id) => (id === 'lb-health' ? band : null) };
  g.window = { matchMedia: () => ({ matches: false }), addEventListener() {} };
  let why = 'no recorder strip for solo: its sender sends no recorder block and no box health';
  g.recWhyNot = () => why;
  const st = { now, health: { findings: [roce], alerts: { recent_failures: [] } }, backends: [] };
  h.bandSay('solo', why);
  h.renderHealthBand(st);
  assert.ok(band.innerHTML.includes(`title="FIRING RoCE ERR +5 · 1m · port_rcv_errors +5\n${why}"`), band.innerHTML);
  chipEls.push({ title: 'stale' });
  h.renderHealthBand({ ...st, now: now + 5 });              // same chips: the titles refreshed in place
  assert.ok(chipEls[0].title.endsWith(`\n${why}`) && chipEls[0].title.includes('· 1m'), chipEls[0].title);
  why = '';                                                  // the box has a strip now
  h.renderHealthBand({ ...st, now: now + 10 });
  assert.ok(!chipEls[0].title.includes('no recorder strip'), `cleared once the strip is drawn: ${chipEls[0].title}`);
  h.renderHealthBand({ ...st, now: now + 15 });
  assert.ok(!chipEls[0].title.includes('no recorder strip'), 'and stays cleared');
  [g.document, g.window, g.recWhyNot] = saved;
}
// the kinds the recorder strip details are health.py kinds (a kind it does not raise would never match)
const hkinds = fs.readFileSync(path.join(__dirname, '..', 'vllm_lb', 'health.py'), 'utf8')
  .match(/^KINDS = \(([^)]*)\)/m)[1].match(/"([a-z_]+)"/g).map((s) => s.slice(1, -1));
assert.ok(h.REC_KINDS.length && h.REC_KINDS.every((k) => hkinds.includes(k)), h.REC_KINDS.join(','));

// DISK (health.py `disk`): its fullest path's whole percent, at or over the fleet's disk_full_pct; a strip kind
assert.strictEqual(h.findingFigure({ kind: 'disk', label: 'DISK', unit: '%', value: 93, threshold: 90 }),
  'DISK 93 % ≥ 90 %');
assert.ok(h.REC_KINDS.includes('disk'), 'a DISK chip opens the strip, whose disk rows the finding flags');
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
// the delivery chip is the alert path's, not a box's fault: warn, and LAST in the band
const withFail = h.healthChips({ findings: [ram, temp], alerts: { recent_failures: [fail] } }, now);
assert.deepStrictEqual(withFail.map((c) => c.cls), ['hl-firing', 'hl-pending', 'hl-failed']);
assert.strictEqual(withFail[2].box, null);
assert.ok(/\.hl-failed \{[^}]*var\(--warn\)/.test(fs.readFileSync(path.join(STATIC, 'health.css'), 'utf8'))
  && !/\.hl-failed \{[^}]*var\(--bad\)/.test(fs.readFileSync(path.join(STATIC, 'health.css'), 'utf8')));

// the health line: figures only
assert.strictEqual(h.healthSummary({ enabled: false, why: 'no health: block' }), 'HEALTH off');
assert.strictEqual(h.healthSummary({ enabled: true, findings: [temp, ram], no_data: [{}, {}, {}],
                                     alerts: { channels: [{ name: 'ntfy' }, { name: 'webhook' }], sent: 1234, failed: 1 } }),
  'HEALTH 3 no data · ALERTS ntfy+webhook · 1,234 sent · 1 failed', 'firing / pending are the band\'s: not repeated');

// an event row: every string the feed carries is ESCAPED (a summary is an engine's or a box's words)
const row = h.eventRowHtml({ t: 1800000000, kind: 'cluster.state', subject: '<b>pair</b>', state: 'failed',
                             summary: 'starting → failed: "rank" <script>' }, new Date(1800000000 * 1000));
assert.ok(!row.includes('<b>pair') && row.includes('&lt;b&gt;pair'));
assert.ok(!row.includes('<script>') && row.includes('&quot;rank&quot;'));
assert.ok(row.includes('ev-bad'), 'a failed cluster is drawn in the fault tone');
assert.ok(/\d\d:\d\d:\d\d/.test(row), 'today: the time of day');
assert.strictEqual(h.eventTone({ kind: 'cluster.state', state: 'running' }), 'ok');
assert.strictEqual(h.eventTone({ kind: 'finding.cleared' }), 'ok');
assert.strictEqual(h.eventTone({ kind: 'alert.failed' }), 'warn', 'a failed send is the alert path\'s, as its band chip');
assert.strictEqual(h.eventTone({ kind: 'request.killed' }), 'warn');
assert.ok(/^[A-Z][a-z]{2} \d{1,2} \d\d:\d\d$/.test(h.eventTime(1800000000, new Date((1800000000 + 3 * 86400) * 1000))),
  'another day: month, day and time');

// ACTIVITY folds each alert send into the event it sent (alerts.py: `about` = the event's id or subject, `alerted` = its
// kind): a mark on that row, not a row of its own. Newest first; a re-fired finding's sends land on their own firing.
const evs = [
  { t: 108, kind: 'alert.failed', subject: 'solo', summary: 'webhook failed: … — HTTP 500', channel: 'webhook', about: 'solo:roce', alerted: 'finding.firing', why: 'HTTP 500' },
  { t: 107, kind: 'alert.sent', subject: 'solo', summary: 'ntfy: …', channel: 'ntfy', about: 'solo:roce', alerted: 'finding.firing' },
  { t: 106, kind: 'finding.firing', subject: 'solo', summary: 'RoCE ERR on solo', id: 'solo:roce', finding: 'roce' },
  { t: 105, kind: 'finding.cleared', subject: 'solo', summary: 'RoCE ERR cleared', id: 'solo:roce', finding: 'roce' },
  { t: 102, kind: 'alert.sent', subject: 'solo', summary: 'ntfy: …', channel: 'ntfy', about: 'solo:roce', alerted: 'finding.firing' },
  { t: 101, kind: 'finding.firing', subject: 'solo', summary: 'RoCE ERR on solo', id: 'solo:roce', finding: 'roce' },
  { t: 100, kind: 'alert.failed', subject: 'pair', summary: 'webhook failed', channel: 'webhook', about: 'pair', alerted: 'cluster.state', why: 'HTTP 502' },
];
const folded = h.foldAlerts(evs);
assert.deepStrictEqual(folded.map((e) => e.kind), ['finding.firing', 'finding.cleared', 'finding.firing', 'alert.failed'],
  'the sends that found their event are gone as rows; one whose event is outside the page stays a row');
assert.deepStrictEqual(folded[0].alerts.map((a) => [a.channel, a.ok]), [['webhook', false], ['ntfy', true]]);
assert.deepStrictEqual(folded[2].alerts.map((a) => [a.channel, a.ok]), [['ntfy', true]]);
assert.deepStrictEqual(folded[1].alerts, []);
assert.strictEqual(evs[2].alerts, undefined, 'the feed\'s own records are not written to');
const markRow = h.eventRowHtml(folded[0], new Date(106 * 1000));
assert.ok(markRow.includes('✗ webhook') && markRow.includes('✓ ntfy') && markRow.includes('title="webhook failed · HTTP 500"'));
assert.ok(/class="ev-al ev-warn"[^>]*>✗ webhook/.test(markRow) && /class="ev-al ev-ok"[^>]*>✓ ntfy/.test(markRow));
// THE COUNT'S "+" (page cut): judged on the RAW page the server returned against its stated `limit` (events.py
// READ_LIMIT), never on the folded rows — a full page of 500 whose sends fold into 300 rows is still cut.
assert.strictEqual(h.activityCount({ events: new Array(500).fill({}), limit: 500 }, 300), '300+');
assert.strictEqual(h.activityCount({ events: new Array(499).fill({}), limit: 500 }, 499), '499',
  'under the page: whole');
assert.strictEqual(h.activityCount({ events: new Array(2000).fill({}), limit: 2000 }, 1234), '1,234+');
assert.strictEqual(h.activityCount({ events: [] }, 0), '0', 'no stated limit: no cut is claimed');
// collapsed, the 3 newest rows; open, a page; the expander names what it hides
assert.deepStrictEqual(h.activityView(10, true, 60), { shown: 3, rest: 7 });
assert.deepStrictEqual(h.activityView(2, true, 60), { shown: 2, rest: 0 });
assert.deepStrictEqual(h.activityView(100, false, 60), { shown: 60, rest: 0 });
// the open / collapsed choice is this browser's (localStorage `lb.activity`), every touch wrapped
const mem = new Map();
const okStore = { getItem: (k) => (mem.has(k) ? mem.get(k) : null), setItem: (k, v) => mem.set(k, v) };
assert.strictEqual(h.actOpenGet(okStore), false, 'a fresh browser starts collapsed');
h.actOpenSet(okStore, true);
assert.strictEqual(mem.get('lb.activity'), '{"open":true}');
assert.strictEqual(h.actOpenGet(okStore), true);
const deadStore = { getItem() { throw new Error('denied'); }, setItem() { throw new Error('denied'); } };
assert.strictEqual(h.actOpenGet(deadStore), false);
assert.doesNotThrow(() => h.actOpenSet(deadStore, true));
assert.strictEqual(h.actOpenGet(null), false);
mem.set('lb.activity', 'not json');
assert.strictEqual(h.actOpenGet(okStore), false);

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
// the Activity tile sits directly under the band: the first element of the layout after the queued leases
const lyIds = html.match(/<main id="layout"[\s\S]*?<\/main>/)[0].match(/data-ly="([^"]+)"/g);
assert.deepStrictEqual(lyIds.slice(0, 3), ['data-ly="queued"', 'data-ly="events"', 'data-ly="cluster"']);
assert.ok(html.includes('id="events-expand"') && html.includes('aria-controls="events-list"'));
// a phone: every control of the band and the tile is a finger's height (44 px)
const css = fs.readFileSync(path.join(STATIC, 'health.css'), 'utf8');
const phone = css.match(/@media \(max-width: 600px\) \{([\s\S]*)\}\s*$/)[1];
for (const sel of ['.hl-chip', '.hl-more', '.ev-chip', '.ev-more', '.ev-expand']) {
  assert.ok(new RegExp(`\\${sel}[^{]*\\{[^}]*min-height: 44px`).test(phone), `${sel} is 44 px on a phone`);
}
// a phone: the summary takes its own full-width line
assert.ok(/@media \(max-width: 600px\)[\s\S]*\.ev-m \{ grid-column: 1 \/ -1/.test(fs.readFileSync(path.join(STATIC, 'health.css'), 'utf8')));

console.log('ok');
