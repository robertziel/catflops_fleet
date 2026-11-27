/* BOX HEALTH (static/hwextra.js), under node:
 * what gateway.js `machineBody` and the recorder strip (recorder.js) draw from /lb/status `hardware.<box>` `rdma`,
 * `kernel`, `disks`, `models` (vllm_lb/box_health.py).
 *
 * What it pins:
 *   1. the kernel's counts are two figures beside MEMORY on the CARD — XID and OOM — numbers only; "≥n" when the Xid
 *      count is not the whole boot; "—" when unread, never 0; the RoCE / disk / models sections are NOT on the card
 *      (they are the recorder strip's, so the card keeps its height);
 *   2. a RoCE line per port: port · state · rate · MTU, then ONLY the counters that are non-zero or rising; a counter
 *      that MOVED shows "+r/s" and is flagged (`.up`), a port not ACTIVE is flagged (`.down`), and the head counts
 *      both; the rest of the counters and the QoS sit in the port's own <details>, open when the port is down or a
 *      counter rises — each counter drawn once;
 *   3. a counter the port does not expose is "—" with its reason in the title — never 0;
 *   4. a disk bar per path: "used / total GB · %", width = the share; an unreadable path is "—" with why;
 *   5. a models root: its total, the walk's age off `s.now`, the folders holding fleet checkpoints with their count,
 *      the rest summed as "other n";
 *   6. a box without the blocks (an old sender, a stale box) draws none of it, and a name is text (`<script>` escaped);
 *   9. the strip's figures (`hwExtraStrip`): "RoCE up/total up" (+ "· n rising"), "disk <fullest %>", the fault flag
 *      ONLY on a fault (a port not ACTIVE, a counter rising, a disk path the box's pending / firing `disk` finding
 *      names full — health.py judges it against the fleet's `health.disk_full_pct`, the ONE definition); nothing for a
 *      box with no RDMA port (the server's why is a note for the strip's title) or no disk block — never a 0;
 *  10. NO THRESHOLD, NO FULL DISK: health off (or a status that carries no `disk_full_pct`) draws the disk's figure,
 *      never in the fault colour however full, and its title says no threshold is configured and why — the panel
 *      owns no number of its own.
 *
 *   node tests/hwextra.node.js      (tests/test_hwextra_node.py runs this when node exists)
 */
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const STATIC = path.join(__dirname, '..', 'vllm_lb', 'static');
const src = require('./palette_env').withPalette(fs.readFileSync(path.join(STATIC, 'gateway.js'), 'utf8') + '\n'
  + fs.readFileSync(path.join(STATIC, 'hwextra.js'), 'utf8'));
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
  'getComputedStyle', 'ResizeObserver', 'localStorage', 'navigator', 'performance', 'module',
  `${src}\n;return { machineBody, hwExtraFigs, hwExtraHtml, hwExtraStrip, loadColour };`)(
  document, window, () => new Promise(() => {}), { search: '', href: '' }, () => 0, () => 0,
  () => ({}), undefined, window.localStorage, { userAgent: 'node' }, { now: () => 4321 }, undefined);

const base = { host: 'h', age_s: 1, stale: false, cpu_percent: 10, ram_percent: 20, ram_used_mb: 20480,
  ram_total_mb: 102400, rows: [], history: {} };
const port = (over) => ({ dev: 'mlx5_0', port: 1, state: 'ACTIVE', phys_state: 'LinkUp', rate_gbps: 200,
  link_layer: 'Ethernet', netdev: 'enp1s0f0np0', mtu: 9000, why: {}, rates_why: {},
  counters: { out_of_buffer: 1234, local_ack_timeout_err: 7, port_rcv_errors: 0 },
  rates: { out_of_buffer: 20, local_ack_timeout_err: 0, port_rcv_errors: 0 }, ...over });
const h = {
  ...base,
  kernel: { xid_total: 2, oom_kills_total: 10, last_xid: { code: 79, at: 1 }, xid_since: 0, xid_complete: true, why: null },
  rdma: { ports: [port({}), port({ dev: 'mlx5_1', state: 'DOWN', rates: {}, counters: { port_rcv_errors: 3 },
    why: { out_of_buffer: 'no readable hw_counters/out_of_buffer on this port' } })], why: null },
  disks: [{ path: '/', total_bytes: 2e12, used_bytes: 5e11, free_bytes: 1.4e12 },
    { path: '/gone', total_bytes: null, used_bytes: null, free_bytes: null, why: 'statvfs /gone: No such file' }],
  disks_why: null,
  models: [{ root: '/mnt/models', measured_at: 1000, took_s: 2.5, more_count: 2, more_bytes: 3e9, unreadable: [],
    why: null, folders: [{ name: 'qwen-cluster', bytes: 4.12e11, checkpoints: ['a', 'a (uncensored)'] },
      { name: '<script>x</script>', bytes: 5e10, checkpoints: ['b'] }, { name: 'junk', bytes: 1e9, checkpoints: [] }] }],
  models_why: null,
};
const s = { now: 1240, hardware: { box: h } };

// 1. the kernel's figures
assert.deepStrictEqual(api.hwExtraFigs(h), [['xid', '2'], ['oom kills', '10']]);
assert.deepStrictEqual(api.hwExtraFigs({ kernel: { ...h.kernel, xid_complete: false } })[0], ['xid', '≥2']);
assert.deepStrictEqual(api.hwExtraFigs({ kernel: { xid_total: null, oom_kills_total: 3, why: 'no /dev/kmsg' } }),
  [['xid', '—'], ['oom kills', '3']]);
const body = api.machineBody(s, 'box');
assert.ok(body.includes('<span>xid</span><b>2</b>') && body.includes('<span>oom kills</span><b>10</b>'), 'figures on the tile');

// 2. RoCE lines
const html = api.hwExtraHtml(h, s);
assert.ok(!body.includes('class="hx"') && !body.includes('hx-sec') && !body.includes('RoCE'),
  'the card carries no box-health section: it is the recorder strip\'s');
assert.ok(html.includes('RoCE · 2 ports') && html.includes('1 down') && html.includes('1 rising'), html);
assert.ok(html.includes('mlx5_0/1 · <b>ACTIVE</b> · 200 Gb/s · MTU 9000'));
assert.ok(/class="hx-c up"[^>]*>OOB <b>1,234<\/b> <i>\+20\/s<\/i>/.test(html), 'a rising counter is flagged with its rate');
assert.ok(/class="hx-c"[^>]*>ACK TO <b>7<\/b><\/span>/.test(html), 'a still counter: total only, not flagged');
assert.ok(html.includes('class="hx-port down"'), 'a port not ACTIVE is flagged');
// 3. absent counter
assert.ok(/class="hx-c none" title="out_of_buffer: no readable hw_counters\/out_of_buffer on this port">OOB —/.test(html));
// 4. disks
assert.ok(html.includes('500 / 2,000 GB · 25%') && html.includes('width:25.0%'), 'used / total GB · %');
assert.ok(/hx-disk none" title="\/gone: statvfs \/gone: No such file"/.test(html));
// 5. models
assert.ok(html.includes('MODELS /mnt/models · 466 GB') && html.includes('4 min ago'), html);
assert.ok(html.includes('<span>qwen-cluster</span> <b>412 GB · 2 ckpt</b>'));
assert.ok(html.includes('<span>other 3</span> <b>4.0 GB</b>'), 'one unnamed folder + two not sent');
// 6. escaped, and nothing for a box without the blocks or a stale one
assert.ok(!html.includes('<script>') && html.includes('&lt;script&gt;'));
// 7. THE CARD AGREES WITH THE BAND (status.health): a pending / firing `roce` finding flags the counters it names
//    (`rose`), even when this sample's rate is 0; an `xid` / `oom` finding flags its figure with the window's rise.
const still = { ...h, rdma: { ...h.rdma, ports: [port({ rates: { out_of_buffer: 0, local_ack_timeout_err: 0, port_rcv_errors: 0 } }),
  port({ dev: 'mlx5_1', state: 'DOWN', rates: {}, counters: { port_rcv_errors: 3, link_downed: 4 } })] } };
const health = { findings: [
  { id: 'box:roce', box: 'box', kind: 'roce', state: 'firing', value: 1, why: 'mlx5_1/1 link_downed +1 in the last 3600 s',
    rose: [{ dev: 'mlx5_1', port: '1', counter: 'link_downed', rise: 1 }] },
  { id: 'box:xid', box: 'box', kind: 'xid', state: 'pending', value: 2, why: '2 new in the last 3600 s' },
  { id: 'other:oom', box: 'other', kind: 'oom', state: 'firing', value: 5, why: '5 new' }] };
const sh = { now: 1240, hardware: { box: still }, health };
const flagged = api.hwExtraHtml(still, sh, 'box');
assert.ok(flagged.includes('1 down') && flagged.includes('1 rising'), `the head counts the finding's port: ${flagged}`);
assert.ok(/class="hx-c up"[^>]*>LINK DN <b>4<\/b> <i>\+1<\/i>/.test(flagged), 'the named counter: flagged with its window rise');
assert.ok(/class="hx-c"[^>]*>OOB <b>1,234<\/b><\/span>/.test(flagged), 'a counter the finding does not name, rate 0: not flagged');
assert.ok(!api.hwExtraHtml(still, { now: 1240 }, 'box').includes('rising'), 'no finding and no rate: nothing rising');
assert.ok(!api.hwExtraHtml(still, sh, 'other').includes('rising'), "another box's finding is not this box's");
assert.deepStrictEqual(api.hwExtraFigs(h, sh, 'box'), [['xid', '2 · +2', 'bad'], ['oom kills', '10']],
  'a pending / firing xid finding flags XID with its rise; OOM has no finding on this box');
assert.deepStrictEqual(api.hwExtraFigs(h, sh, 'other')[1], ['oom kills', '10 · +5', 'bad']);
const flaggedBody = api.machineBody(sh, 'box');
assert.ok(flaggedBody.includes('<div class="fig bad"><span>xid</span><b>2 · +2</b></div>'), 'machineBody draws the flag');
assert.ok(!flaggedBody.includes('LINK DN'), 'the RoCE lines are not on the card');
// 7. the PHY counters (ethtool -S) and the QoS (dcb) on a port line; a pause counter's rate is shown, not flagged
const qosPort = port({ counters: { rx_discards_phy: 4242, rx_pause_ctrl_phy: 17 }, rates: { rx_discards_phy: 5,
  rx_pause_ctrl_phy: 3 }, pfc_prios: [3], trust: null, dscp_map: 4, qos_at: 1180,
  why: { trust: 'dcb apptrust: exit 1: Attribute read: Operation not supported' } });
const q = api.hwExtraHtml({ ...base, rdma: { ports: [qosPort], why: null } }, { now: 1240 }, 'box');
assert.ok(/class="hx-c up"[^>]*>RX DISC PHY <b>4,242<\/b> <i>\+5\.0\/s<\/i>/.test(q), `a rising PHY drop is flagged: ${q}`);
assert.ok(/class="hx-c"[^>]*>RX PAUSE <b>17<\/b> <i>\+3\.0\/s<\/i>/.test(q), 'pause frames: rate shown, not flagged');
assert.ok(q.includes('PFC <b>3</b>') && q.includes('DSCP map <b>4</b>') && q.includes('dcb read 60 s ago'), q);
assert.ok(/hx-c none" title="trust: dcb apptrust: exit 1: Attribute read: Operation not supported">trust —/.test(q),
  'trust absent with the tool\'s words');
assert.ok(q.includes('RoCE · 1 port</div>') || q.includes('1 rising'), 'the PHY drop counts the port as rising');
const refused = api.hwExtraHtml({ ...base, rdma: { ports: [port({ counters: {}, rates: {},
  why: { ethtool: 'ethtool -S: exit 94: no stats available', qos: 'no netdev in this namespace (see mtu)' } })],
  why: null } }, { now: 1240 }, 'box');
assert.ok(/title="rx_discards_phy: ethtool -S: exit 94: no stats available">RX DISC PHY —/.test(refused),
  'a refused ethtool: every PHY counter absent with its one reason');
assert.ok(/title="pfc: no netdev in this namespace \(see mtu\)">PFC —/.test(refused));
assert.ok(api.hwExtraHtml({ ...base, rdma: { ports: [port({ pfc_prios: [], trust: ['dscp', 'pcp'] })], why: null } },
  { now: 1240 }, 'box').includes('PFC <b>off</b>'), 'PFC off on every priority is a reading: "off", not "—"');
// 8. models: more_count / more_bytes not sent → "other —" and the root's total "—", never a sum that leaves them out
const noMore = api.hwExtraHtml({ ...base, models: [{ ...h.models[0], more_count: null, more_bytes: null,
  more_why: 'the sender sent no readable more_count / more_bytes' }] }, s);
assert.ok(noMore.includes('MODELS /mnt/models · —') && /title="the sender sent no readable more_count[^"]*"><span>other —/.test(noMore),
  noMore);
// 2b. INLINE, ONLY WHAT MOVED: a port line carries its non-zero / rising counters; the rest (zeros, absent) and the
//     QoS are behind the port's own <details> — open when the port is down or a counter rises; each counter once.
const p0 = html.slice(html.indexOf('mlx5_0/1'), html.indexOf('mlx5_1/1'));
const p1 = html.slice(html.indexOf('mlx5_1/1'));
const inline0 = p0.split('<details')[0];
assert.ok(inline0.includes('OOB <b>1,234</b>') && inline0.includes('ACK TO <b>7</b>') && !inline0.includes('RX ERR'),
  `inline: the non-zero counters only: ${inline0}`);
assert.ok(/<details class="hx-more" data-port="mlx5_0\/1" data-drawn="1" open><summary>1 at 0 · 13 — · QoS<\/summary>/.test(p0),
  `a rising port: open, keyed by dev/port, drawn open: ${p0}`);
assert.ok(/<details class="hx-more"[^>]*>[\s\S]*RX ERR <b>0<\/b>/.test(p0), 'the zero counter is in the details');
assert.ok(/<details class="hx-more" data-port="mlx5_1\/1" data-drawn="1" open><summary>15 — · QoS<\/summary>/.test(p1),
  `a down port: open: ${p1}`);
assert.strictEqual((html.match(/OOB /g) || []).length, 2, 'OOB once per port (inline on one, "—" in the other\'s details)');
assert.strictEqual((p0.match(/PFC /g) || []).length, 1, 'the QoS once, in the details');
const calmPort = api.hwExtraHtml({ ...base, rdma: { ports: [port({ rates: {} })], why: null } }, { now: 1240 }, 'box');
assert.ok(/<details class="hx-more" data-port="mlx5_0\/1" data-drawn="0"><summary>1 at 0 · 13 — · QoS<\/summary>/.test(calmPort), `up and still: closed: ${calmPort}`);
// 2c. THE USER'S TOGGLE WINS: `hwExtraHtml`'s fourth argument is the user's choice per "dev/port" (recorder.js
//     keeps it); a chosen port is drawn as chosen, an unchosen one by the fault default, and a choice names one port only.
const isOpen = (x, key) => {
  const m = x.match(new RegExp(`<details class="hx-more" data-port="${key}" data-drawn="([01])"( open)?>`));
  assert.ok(m, `no hx-more for ${key}: ${x}`);
  assert.strictEqual(m[1] === '1', !!m[2], `data-drawn says what was drawn: ${m[0]}`);
  return !!m[2];
};
assert.strictEqual(isOpen(api.hwExtraHtml({ ...base, rdma: { ports: [port({ rates: {} })], why: null } }, { now: 1240 }, 'box',
  { 'mlx5_0/1': true }), 'mlx5_0/1'), true, 'a healthy port the user opened: drawn open');
const closedRising = api.hwExtraHtml(h, s, 'box', { 'mlx5_0/1': false });
assert.strictEqual(isOpen(closedRising, 'mlx5_0/1'), false, 'a rising port the user closed: drawn closed');
assert.strictEqual(isOpen(closedRising, 'mlx5_1/1'), true, 'the other port, unchosen: the fault default (down: open)');
assert.strictEqual(isOpen(api.hwExtraHtml(h, s, 'box', {}), 'mlx5_0/1'), true, 'no choice: the fault default');
assert.strictEqual(isOpen(api.hwExtraHtml(h, s, 'box', { 'mlx5_0/2': false, mlx5_0: false }), 'mlx5_0/1'), true,
  'a choice for another port (or a bare device) is not this port\'s');
// 9. THE STRIP'S FIGURES: RoCE up/total (+ rising), the fullest disk; the fault flag only on a fault.
assert.ok(!/HX_DISK_FULL_PCT/.test(fs.readFileSync(path.join(STATIC, 'hwextra.js'), 'utf8')),
  'no full-disk number of the panel\'s own: the fleet\'s health.disk_full_pct is the one definition');
/** A status with health on at `pct`, and the box's `disk` finding naming `full` paths ([] = no finding). */
const on = (pct, full = []) => ({ now: 1240, health: { enabled: true, no_data: [], hold_s: 20, disk_full_pct: pct,
  findings: full.length ? [{ id: 'box:disk', box: 'box', kind: 'disk', state: 'pending', value: 90, threshold: pct,
    unit: '%', full: full.map((p) => ({ path: p, pct: 90 })) }] : [] } });
const st = api.hwExtraStrip(h, s, 'box');
assert.deepStrictEqual(st.parts.map((p) => [p.text, !!p.bad]), [['RoCE 1/2 up · 1 rising', true], ['disk 25%', false]],
  JSON.stringify(st));
assert.ok(st.parts[0].title.includes('mlx5_1/1 DOWN') && st.parts[0].title.includes('mlx5_0/1 rising OOB'), st.parts[0].title);
assert.ok(st.parts[1].title.includes('/ 500 / 2,000 GB · 25%') && st.parts[1].title.includes('/gone: statvfs /gone: No such file'),
  st.parts[1].title);
const disk = (path, pct) => ({ path, total_bytes: 1e12, used_bytes: pct * 1e10, free_bytes: (100 - pct) * 1e10 });
const calm = { ...base, rdma: { ports: [port({ rates: {} }), port({ dev: 'mlx5_1', rates: {} })], why: null },
  disks: [disk('/', 40), disk('/data', 90)], disks_why: null };
assert.deepStrictEqual(api.hwExtraStrip(calm, on(90, ['/data']), 'box').parts.map((p) => [p.text, !!p.bad]),
  [['RoCE 2/2 up', false], ['disk 90%', true]], 'every port up and still: plain; the disk the finding names: flagged');
const fullRows = api.hwExtraHtml(calm, on(90, ['/data']), 'box');
assert.ok(/class="hx-disk full" title="\/data /.test(fullRows) && !/class="hx-disk full" title="\/ /.test(fullRows),
  'the named path\'s row is flagged, the other not');
// the strip judges NOTHING itself: the finding (health.py, against the configured disk_full_pct) is the one definition
assert.ok(!api.hwExtraStrip(calm, on(90), 'box').parts[1].bad, 'no disk finding: no red, whatever the percent');
assert.ok(!api.hwExtraHtml(calm, on(90), 'box').includes('hx-disk full'), 'nor any row');
const elsewhere = on(90, ['/data']);
elsewhere.health.findings[0].box = 'other';
assert.ok(!api.hwExtraStrip(calm, elsewhere, 'box').parts[1].bad, 'another box\'s finding: not ours');
assert.ok(api.hwExtraStrip(calm, on(90, ['/data']), 'box').parts[1].title.includes('full at ≥ 90 %'),
  'the configured line is in the title');
// every port ACTIVE and one fault counter moving: rising alone is a fault (`bad: downs > 0` must not pass)
const risingUp = { ...calm, rdma: { ports: [port({}), port({ dev: 'mlx5_1', rates: {} })], why: null } };
assert.deepStrictEqual(api.hwExtraStrip(risingUp, on(90), 'box').parts[0], {
  text: 'RoCE 2/2 up · 1 rising', title: 'mlx5_0/1 rising OOB', bad: true }, 'all up, one rising: flagged');
// only a PAUSE counter moving (rx_pause_ctrl_phy, HX_INFO: PFC at work): not rising, not a fault
const pausing = { ...calm, rdma: { why: null, ports: [
  port({ counters: { rx_pause_ctrl_phy: 900 }, rates: { rx_pause_ctrl_phy: 40 } }),
  port({ dev: 'mlx5_1', rates: {} })] } };
assert.deepStrictEqual(api.hwExtraStrip(pausing, on(90), 'box').parts[0].text, 'RoCE 2/2 up');
assert.strictEqual(api.hwExtraStrip(pausing, on(90), 'box').parts[0].bad, false, 'pause frames alone: plain');
// AN UNREAD PORT STATE is missing data, not a fault: "—" with its reason, neither down nor up — counted
// as unread — and never the fault colour on its own.
const unread = { ...calm, rdma: { why: null,
  ports: [port({ rates: {} }), port({ dev: 'mlx5_1', state: null, rates: {} })] } };
const ur = api.hwExtraStrip(unread, on(90), 'box').parts[0];
assert.deepStrictEqual([ur.text, ur.bad], ['RoCE 1/2 up · 1 unread', false], JSON.stringify(ur));
assert.ok(ur.title.includes('mlx5_1/1 state unread: the port\'s sysfs state file was not read'), ur.title);
const urHtml = api.hwExtraHtml(unread, on(90), 'box');
assert.ok(!urHtml.includes('hx-port down') && !urHtml.includes('down</b>'), `not drawn down: ${urHtml}`);
assert.ok(urHtml.includes('RoCE · 2 ports · 1 unread') && /mlx5_1\/1 · <b[^>]*>—<\/b>/.test(urHtml), urHtml);
const urWhy = api.hwExtraStrip({ ...unread, rdma: { ports: [port({ dev: 'mlx5_1', state: undefined, rates: {},
  why: { state: 'cannot read /sys/class/infiniband/mlx5_1/ports/1/state' } })], why: null } }, on(90), 'box').parts[0];
assert.ok(urWhy.text === 'RoCE 0/1 up · 1 unread' && !urWhy.bad && urWhy.title.includes('cannot read /sys'),
  JSON.stringify(urWhy));
// THE TEXT, NOT ONLY THE LAYOUT, PARTS EVERY FIGURE (the audit read "ungrouped 98015" in box-b's RoCE
// section): a port whose one inline counter is OOB 980 and whose details summary starts with its absent count "15 —"
// rendered as two flex items — but its textContent (what a detached node's innerText, a copy or a screen reader gets)
// glued them into "98015". Every figure is parted by whitespace in the markup itself.
function unreadHtmlBox() {
  return { ...base, rdma: { why: null, ports: [port({}), port({ dev: 'mlx5_1', state: null, rates: {} })] },
    disks: [{ path: '/', total_bytes: 1e12, used_bytes: 9e11, free_bytes: 1e11 }], disks_why: null };
}
const gluedPort = port({ dev: 'mlx5_0', counters: { out_of_buffer: 980 }, rates: {} });
const flat = api.hwExtraHtml({ ...base, rdma: { ports: [gluedPort], why: null } }, { now: 1240 }, 'box')
  .replace(/<[^>]+>/g, '');
assert.ok(!/\d{5,}/.test(flat) && /OOB 980\s+15 —/.test(flat), `no two figures glued in the text: ${flat}`);
// and in general: the words of the text with the tags dropped are the words with every tag a space — no element
// boundary is the only thing parting two figures (the whole open view, every section)
const words = (x) => x.replace(/&[a-z#0-9]+;/g, 'x').trim().split(/\s+/);
const wholeHtml = `${api.hwExtraHtml(h, s, 'box')} ${api.hwExtraHtml(unreadHtmlBox(), on(90), 'box')}`;
assert.deepStrictEqual(words(wholeHtml.replace(/<[^>]+>/g, '')), words(wholeHtml.replace(/<[^>]+>/g, ' ')),
  'two words glued across a tag');
// 10. NO THRESHOLD: health off, or a status without the figure — the figure, never the fault colour, and why.
const off = { now: 1240,
  health: { enabled: false, why: 'health findings are off: no `health:` block', findings: [], no_data: [] } };
const brim = { ...calm, disks: [disk('/', 99)] };
[[off, 'health findings are off: no `health:` block'], [{ now: 1240 }, 'the status carries no health.disk_full_pct']]
  .forEach(([st0, why]) => {
    const part = api.hwExtraStrip(brim, st0, 'box').parts[1];
    assert.ok(part.text === 'disk 99%' && !part.bad,
      `no threshold: the figure, never flagged: ${JSON.stringify(part)}`);
    assert.ok(part.title.includes(`no full-disk threshold configured: ${why}`), part.title);
    const row = api.hwExtraHtml(brim, st0, 'box');
    assert.ok(!row.includes('hx-disk full') && row.includes('99%'), row);
    assert.ok(row.includes(`no full-disk threshold configured: ${why}`), `the row's title says why too: ${row}`);
  });
assert.deepStrictEqual(api.hwExtraStrip(still, sh, 'box').parts[0].text, 'RoCE 1/2 up · 1 rising',
  'the finding\'s rise counts its port as rising (the band and the strip agree)');
const bare = api.hwExtraStrip({ ...base, rdma: { ports: [], why: 'no /sys/class/infiniband' }, disks: [],
  disks_why: 'the sender sends no disks block (it predates box health)' }, { now: 1240 }, 'box');
assert.deepStrictEqual(bare.parts, [], 'no RDMA port, no disk block: no figure, never a 0');
assert.deepStrictEqual(bare.notes, ['RoCE: no /sys/class/infiniband', 'disk: the sender sends no disks block (it predates box health)']);
assert.deepStrictEqual(api.hwExtraStrip(base, s, 'box'), { parts: [], notes: [] }, 'a box without the blocks: nothing');
const gone = api.hwExtraStrip({ ...base, disks: [h.disks[1]] }, s, 'box').parts[0];
assert.ok(gone.text === 'disk —' && !gone.bad && gone.title.includes('statvfs /gone'), 'no readable disk: "—" with why');
assert.strictEqual(api.hwExtraHtml(base, s), '');
assert.deepStrictEqual(api.hwExtraFigs(base), []);
assert.strictEqual(api.hwExtraHtml({ ...base, rdma: null, kernel: null, disks: null, models: null }, s), '');
assert.strictEqual(api.hwExtraHtml({ ...base, rdma: { ports: [], why: 'no /sys/class/infiniband' } }, s), '');
console.log('ok');
