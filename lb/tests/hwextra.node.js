/* BOX HEALTH on a machine card (static/hwextra.js), under node: what gateway.js `machineBody` draws
 * from /lb/status `hardware.<box>` `rdma`, `kernel`, `disks`, `models` (vllm_lb/box_health.py).
 *
 * What it pins:
 *   1. the kernel's counts are two figures beside MEMORY — XID and OOM — numbers only; "≥n" when the Xid count is not
 *      the whole boot; "—" when unread, never 0;
 *   2. a RoCE line per port: port · state · rate · MTU, each counter's total; a counter that MOVED shows "+r/s" and
 *      is flagged (`.up`), a port not ACTIVE is flagged (`.down`), and the head counts both;
 *   3. a counter the port does not expose is "—" with its reason in the title — never 0;
 *   4. a disk bar per path: "used / total GB · %", width = the share; an unreadable path is "—" with why;
 *   5. a models root: its total, the walk's age off `s.now`, the folders holding fleet checkpoints with their count,
 *      the rest summed as "other n";
 *   6. a box without the blocks (an old sender, a stale box) draws none of it, and a name is text (`<script>` escaped).
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
  `${src}\n;return { machineBody, hwExtraFigs, hwExtraHtml, loadColour };`)(
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
assert.ok(body.includes(html), 'machineBody carries the section');
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
assert.ok(html.includes('<span>qwen-cluster</span><b>412 GB · 2 ckpt</b>'));
assert.ok(html.includes('<span>other 3</span><b>4.0 GB</b>'), 'one unnamed folder + two not sent');
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
assert.ok(flaggedBody.includes('LINK DN <b>4</b> <i>+1</i>'), 'machineBody passes the box to the RoCE lines');
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
assert.strictEqual(api.hwExtraHtml(base, s), '');
assert.deepStrictEqual(api.hwExtraFigs(base), []);
assert.strictEqual(api.hwExtraHtml({ ...base, rdma: null, kernel: null, disks: null, models: null }, s), '');
assert.strictEqual(api.hwExtraHtml({ ...base, rdma: { ports: [], why: 'no /sys/class/infiniband' } }, s), '');
console.log('ok');
