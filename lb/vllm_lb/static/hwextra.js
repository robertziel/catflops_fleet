/* BOX HEALTH — drawn from /lb/status `hardware.<box>` (vllm_lb/box_health.py); where it sits — in a
 * dropdown that already exists, so it takes no more space and stays one tap away:
 *
 *   hwExtraFigs(h, s, box)   ON THE MACHINE CARD: the kernel's counts as two more figures beside MEMORY: XID and OOM.
 *                            Numbers only; "≥n" when the Xid count is not the whole boot (`xid_complete` false: the
 *                            ring buffer no longer held its first record), "—" when unread.
 *   hwExtraStrip(h, s, box)  ON THE FLIGHT RECORDER'S STRIP (recorder.js), closed: `{parts, notes}` — "RoCE 2/2 up"
 *                            (+ "· 1 rising") and "disk 93%" (the fullest configured path), each {text, title, bad};
 *                            `notes` are the reasons a block draws no figure (a box with no RDMA port: the server's
 *                            `rdma.why`), for the strip's title. No figure for a block the sender does not send, never 0.
 *   hwExtraHtml(h, s, box, chosen)
 *                            THE OPEN STRIP, after the recorder's own content: one RoCE line per port (port · state ·
 *                            rate · MTU, then ONLY the counters that are non-zero or rising; every other counter and
 *                            the port's QoS behind its own <details>, open when the port is down or a counter rises —
 *                            each counter drawn once), one bar per configured disk ("used / total GB · %"), and each
 *                            models root's folders by size, named by the fleet checkpoints they hold, with the walk's age.
 *                            `chosen` = the user's own toggle per port, {"mlx5_0/1": true} (recorder.js keeps it).
 *
 * THE USER'S TOGGLE WINS. The open strip is repainted on every status patch (~1 s), so a port's <details>
 * drawn from the fault default alone undid the user's tap: a healthy port's "12 at 0 · QoS" opened, then shut on the
 * next patch; a faulty port's closed, then reopened. A port the user toggled is drawn as they left it (`chosen`); one
 * they did not is drawn by the fault default. Each details carries its key (`data-port`, "dev/port") and what was
 * drawn (`data-drawn`), so recorder.js can tell the user's toggle from the browser's own toggle event for a details
 * painted open. This file still holds no state: the choices are recorder.js's, handed in.
 *
 * WHY NOT ON THE CARD. The sections made every card taller by the box's ports × counters; the strip is collapsed by
 * default, sits in its own subgrid row (recorder.css `.machine-slot`) and remembers per browser whether it is open —
 * the figures that say "look here" (a port down, a counter rising, a disk full) are on its closed line in the fault
 * colour, the rest one click away.
 *
 * WHAT IS A FAULT, ONCE. `hxDown` (a port whose READ state is not ACTIVE — an unread state is "—", counted "unread",
 * never a fault), `hxRising` (a non-pause counter with a rate > 0, or one a finding names) and `hxFull` (a disk path
 * the box's pending / firing `disk` finding names in its `full` — health.py judges it against the fleet's
 * `health.disk_full_pct`, the band's chip and the alert are that same finding) are the only definitions: the strip's
 * flag, the section head's counts, the port line's and the disk row's colour all read them. The fault colour is for
 * those and nothing else. Health off, or a status without the figure: there is NO threshold — the disk's figure is
 * drawn unflagged however full, and its title says no threshold is configured and why (never a number of
 * the panel's own).
 *
 * THE STRIP AGREES WITH THE BAND. A box's pending / firing findings (/lb/status `health.findings`, vllm_lb/health.py)
 * flag it too: a `roce` finding marks every counter its `rose` names ("+1": the window's rise, even when this
 * sample's rate is 0) and counts its port as rising; an `xid` / `oom` finding flags that card figure with its rise
 * ("2 · +2"). The rise is the finding's, read from the status — never a second rise computation here.
 *
 * Figures, not sentences (operator rule): every label carries a number; what is absent is "—" with its reason in the
 * title, never a 0. A box whose sender predates these blocks, or a stale box (hw.py withdraws them), draws nothing.
 * gateway.js `machineBody` calls hwExtraFigs and `renderMachines` hands the other two to `recorderPaint` (hwExtraHtml
 * as a builder of the user's port choices); this file holds no state and reads no clock but `s.now`.
 */
'use strict';

/** The port counters' short labels (readers/rdma.py COUNTERS, then PHY_COUNTERS off `ethtool -S`), in the order a
 *  port line shows them, and the `why` key that holds the reason for a whole group (a missing hw_counters/ directory,
 *  a refused ethtool). A pause counter (`info`) rising is PFC at work, not a fault: it shows its rate unflagged. */
const HX_COUNTERS = [
  ['out_of_buffer', 'OOB', 'hw_counters'], ['local_ack_timeout_err', 'ACK TO', 'hw_counters'],
  ['rnr_nak_retry_err', 'RNR', 'hw_counters'], ['packet_seq_err', 'SEQ', 'hw_counters'],
  ['implied_nak_seq_err', 'NAK SEQ', 'hw_counters'], ['port_rcv_errors', 'RX ERR', 'counters'],
  ['port_xmit_discards', 'TX DISC', 'counters'], ['port_rcv_remote_physical_errors', 'RX PHY', 'counters'],
  ['symbol_error', 'SYM', 'counters'], ['link_downed', 'LINK DN', 'counters'],
  ['rx_discards_phy', 'RX DISC PHY', 'ethtool'], ['tx_discards_phy', 'TX DISC PHY', 'ethtool'],
  ['rx_crc_errors_phy', 'CRC', 'ethtool'], ['rx_symbol_err_phy', 'SYM PHY', 'ethtool'],
  ['rx_pause_ctrl_phy', 'RX PAUSE', 'ethtool', 'info'], ['tx_pause_ctrl_phy', 'TX PAUSE', 'ethtool', 'info'],
];

/** The port's QoS (iproute2 dcb, `qos_at`): `PFC 3` · `trust dscp>pcp` · `DSCP map 4`; "—" with the tool's words. */
function hxQos(p, why, now) {
  const fig = (label, v, key) => (v === null || v === undefined
    ? `<span class="hx-c none" title="${hxEsc(`${key}: ${why[key] || why.qos || 'not sent'}`)}">${label} —</span>`
    : `<span class="hx-c">${label} <b>${hxEsc(v)}</b></span>`);
  const pfc = Array.isArray(p.pfc_prios) ? (p.pfc_prios.length ? p.pfc_prios.join(',') : 'off') : null;
  const trust = Array.isArray(p.trust) ? p.trust.join('>') : null;
  const age = hxNum(p.qos_at) && hxNum(now) ? `dcb read ${hxAge(Math.max(0, now - p.qos_at))} ago` : 'dcb: no reading';
  return `<span class="hx-cs hx-qos" title="${hxEsc(age)}">${fig('PFC', pfc, 'pfc')} ${fig('trust', trust, 'trust')} `
    + `${fig('DSCP map', hxNum(p.dscp_map) ? p.dscp_map : null, 'dscp_map')}</span>`;
}

const hxEsc = (v) => String(v == null ? '' : v).replace(/[&<>"']/g, (c) =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const hxNum = (v) => typeof v === 'number' && Number.isFinite(v);
const hxInt = (v) => (hxNum(v) ? Math.round(v).toLocaleString('en-US') : '—');
/** Bytes as decimal GB (what `df -H` and a disk's label say), whole above 100. */
const hxGB = (b) => (hxNum(b) ? (b / 1e9 >= 100 ? Math.round(b / 1e9).toLocaleString('en-US') : (b / 1e9).toFixed(1)) : '—');
/** A rate with the precision its size needs (recorder.js `recFmt` "rate"): ≥10 whole, 1–10 one decimal, <1 two. */
const hxRate = (r) => (r >= 10 ? Math.round(r).toLocaleString('en-US') : r >= 1 ? r.toFixed(1) : r.toFixed(2));
const hxAge = (s) => (s < 90 ? `${Math.round(s)} s` : s < 5400 ? `${Math.round(s / 60)} min` : `${(s / 3600).toFixed(1)} h`);

/** THE BOX'S FULL DISKS, from the status: {paths, note} — `paths` the set its pending / firing `disk` finding names
 *  (health.py: each configured path at or over the fleet's `disk_full_pct`), `note` the title's word on the line:
 *  "full at ≥ 90 %", or why there is none (health off: the server's own why). This file judges no disk itself. */
function hxFullAt(s, box) {
  const hl = (s && s.health) || null;
  if (!hl || !hxNum(hl.disk_full_pct)) {
    return { paths: new Set(), note: `no full-disk threshold configured: ${hl && hl.enabled === false
      && hl.why ? hl.why : 'the status carries no health.disk_full_pct'}` };
  }
  const f = hxFinding(s, box, 'disk');
  return { paths: new Set(((f && f.full) || []).map((x) => x && x.path)),
    note: `full at ≥ ${hl.disk_full_pct} %` };
}
/** A port whose state was not read: missing data, not a fault — "—" with its reason, counted neither
 *  up nor down. */
const hxUnread = (p) => typeof p.state !== 'string' || !p.state;
/** Why a port's state is unread: the sender's `why.state`, else the plain fact. */
const hxStateWhy = (p) => ((p.why || {}).state) || 'the port\'s sysfs state file was not read';
/** A port is down when its state was READ and is not ACTIVE. */
const hxDown = (p) => !hxUnread(p) && p.state !== 'ACTIVE';
/** The counters that are information, not faults: a pause counter rising is PFC at work. */
const HX_INFO = new Set(HX_COUNTERS.filter((c) => c[3]).map((c) => c[0]));
/** A port is rising when a fault counter moved this sample, or a pending / firing finding names one of its counters. */
const hxRising = (p, rose) => Object.entries(p.rates || {}).some(([k, v]) => hxNum(v) && v > 0 && !HX_INFO.has(k))
  || Object.keys(rose || {}).length > 0;
/** One disk's share used (0-100), or null when it was not read. */
const hxPct = (d) => (hxNum(d.used_bytes) && hxNum(d.total_bytes) && d.total_bytes > 0 ? (100 * d.used_bytes) / d.total_bytes : null);
/** A disk is FULL when the box's `disk` finding names its path (`at` = hxFullAt); never with no threshold. */
const hxFull = (d, at) => at.paths.has(d.path);

/** The box's pending / firing finding of `kind` (status.health), or null. */
function hxFinding(s, box, kind) {
  const fs = (s && s.health && Array.isArray(s.health.findings)) ? s.health.findings : [];
  return fs.find((f) => f && f.box === box && f.kind === kind && (f.state === 'pending' || f.state === 'firing')) || null;
}

/** [label, value] per figure, and a third 'bad' when a finding flags it: [label, "n · +rise", 'bad']. */
function hwExtraFigs(h, s, box) {
  const k = (h || {}).kernel;
  if (!k) return [];
  const xid = hxNum(k.xid_total) ? `${k.xid_complete === false ? '≥' : ''}${hxInt(k.xid_total)}` : '—';
  const flag = (label, value, kind) => {
    const f = hxFinding(s, box, kind);
    return f ? [label, `${value} · +${hxInt(f.value)}`, 'bad'] : [label, value];
  };
  return [flag('xid', xid, 'xid'), flag('oom kills', hxInt(k.oom_kills_total), 'oom')];
}

/** The counters a `roce` finding names on port `p`: counter → its rise in the window. */
function hxRose(p, finding) {
  const out = {};
  ((finding && finding.rose) || []).forEach((r) => {
    if (r && r.dev === p.dev && String(r.port) === String(p.port)) out[r.counter] = r.rise;
  });
  return out;
}

/** A port's key for the user's toggle: "mlx5_0/1" — the device and port the line is headed by, unique on a box. */
const hxPortKey = (p) => `${p.dev}/${p.port}`;

/** One port: `mlx5_0/1 · ACTIVE · 200 Gb/s · MTU 9000`, its non-zero / rising counters inline, and a <details> with
 *  the rest ("1 at 0 · 13 — · QoS") — as the user last toggled it (`chosen[key]`, a boolean), else open when the port
 *  is down or rising. `rose` = the finding's counters here. */
function hxPort(p, rose, now, chosen) {
  const rates = p.rates || {};
  const why = { ...(p.why || {}), ...(p.rates_why || {}) };
  const down = hxDown(p);
  const head = [`${p.dev}/${p.port}`, hxUnread(p) ? '—' : p.state, hxNum(p.rate_gbps) ? `${p.rate_gbps} Gb/s` : '—',
    hxNum(p.mtu) ? `MTU ${p.mtu}` : 'MTU —'];
  const title = [hxUnread(p) ? `state: ${hxStateWhy(p)}` : '', p.netdev ? `netdev ${p.netdev}` : '',
    p.phys_state ? `phys ${p.phys_state}` : '',
    p.link_layer || '', why.mtu ? `mtu: ${why.mtu}` : '', why.rate_gbps ? `rate: ${why.rate_gbps}` : '']
    .filter(Boolean).join(' · ');
  const inline = [], zero = [], absent = [];
  HX_COUNTERS.forEach(([key, label, group, info]) => {
    const total = (p.counters || {})[key];
    if (!hxNum(total)) {
      absent.push(`<span class="hx-c none" title="${hxEsc(`${key}: ${why[key] || why[group] || 'not sent'}`)}">${label} —</span>`);
      return;
    }
    const r = rates[key];
    const moving = hxNum(r) && r > 0 && !info;
    const named = hxNum((rose || {})[key]);
    const t = `${key} · ${total.toLocaleString('en-US')} total`
      + (hxNum(r) ? ` · ${hxRate(r)}/s` : why[key] ? ` · rate: ${why[key]}` : '')
      + (named ? ` · +${rose[key]} in the finding's window` : '');
    const mark = moving || (info && hxNum(r) && r > 0) ? ` <i>+${hxRate(r)}/s</i>` : named ? ` <i>+${hxInt(rose[key])}</i>` : '';
    const chip = `<span class="hx-c${moving || named ? ' up' : ''}" title="${hxEsc(t)}">${label} <b>${hxInt(total)}</b>${mark}</span>`;
    (total > 0 || mark ? inline : zero).push(chip);
  });
  const summary = [zero.length ? `${zero.length} at 0` : '', absent.length ? `${absent.length} —` : '', 'QoS']
    .filter(Boolean).join(' · ');
  const key = hxPortKey(p);
  const choice = (chosen || {})[key];
  const open = typeof choice === 'boolean' ? choice : down || hxRising(p, rose);
  return `<div class="hx-port${down ? ' down' : ''}"><span class="hx-ph" title="${hxEsc(title)}">`
    + `${head.map((x, i) => (i === 1 ? `<b>${hxEsc(x)}</b>` : hxEsc(x))).join(' · ')}</span> `
    + (inline.length ? `<span class="hx-cs">${inline.join(' ')}</span> ` : '')
    + `<details class="hx-more" data-port="${hxEsc(key)}" data-drawn="${open ? 1 : 0}"${open ? ' open' : ''}>`
    + `<summary>${summary}</summary> `
    + `<span class="hx-cs">${[...zero, ...absent].join(' ')}</span> ${hxQos(p, why, now)}</details></div>`;
}

/** The ports' counts, from the one definition of down, unread and rising: {ports, roses, up, downs, unread, risers}.
 *  `up` counts only a port that SAID ACTIVE. */
function hxRdmaCounts(r, finding) {
  const ports = r && Array.isArray(r.ports) ? r.ports : [];
  const roses = ports.map((p) => hxRose(p, finding));
  const downs = ports.filter(hxDown);
  const unread = ports.filter(hxUnread);
  const risers = ports.map((p, i) => [p, roses[i]]).filter(([p, rose]) => hxRising(p, rose));
  return { ports, roses, up: ports.length - downs.length - unread.length, downs, unread, risers };
}

function hxRdma(r, finding, now, chosen) {
  const c = hxRdmaCounts(r, finding);
  if (!c.ports.length) return '';            // a box without RDMA: nothing to draw (the reason is in rdma.why)
  return `<div class="hx-sec"><div class="hx-h">RoCE · ${c.ports.length} port${c.ports.length === 1 ? '' : 's'}`
    + `${c.downs.length ? ` · <b class="hx-bad">${c.downs.length} down</b>` : ''}`
    + `${c.unread.length ? ` · ${c.unread.length} unread` : ''}`
    + `${c.risers.length ? ` · <b class="hx-bad">${c.risers.length} rising</b>` : ''}</div> `
    + `${c.ports.map((p, i) => hxPort(p, c.roses[i], now, chosen)).join(' ')}</div>`;
}

/** "500 / 2,000 GB · 25%": a read disk's figure, the row's and the strip title's one wording. */
const hxDiskFig = (d, pct) => `${hxGB(d.used_bytes)} / ${hxGB(d.total_bytes)} GB · ${Math.round(pct)}%`;

function hxDisks(disks, why, at) {
  if (!Array.isArray(disks)) return '';
  if (!disks.length) return why ? `<div class="hx-sec"><div class="hx-h" title="${hxEsc(why)}">DISKS · —</div></div>` : '';
  const rows = disks.map((d) => {
    const pct = hxPct(d);
    const ok = pct !== null;
    const colour = ok && typeof loadColour === 'function' ? loadColour(pct) : null;
    const fig = ok ? hxDiskFig(d, pct) : '—';
    const title = ok ? `${d.path} · ${hxGB(d.free_bytes)} GB free · ${at.note}` : `${d.path}: ${d.why || 'not read'}`;
    return `<div class="hx-disk${ok ? (hxFull(d, at) ? ' full' : '') : ' none'}" title="${hxEsc(title)}">`
      + `<span class="hx-dp">${hxEsc(d.path)}</span> `
      + `<span class="hx-bar"><i style="width:${ok ? Math.min(100, pct).toFixed(1) : 0}%${colour ? `;background:${colour}` : ''}"></i></span> `
      + `<b>${fig}</b></div>`;
  }).join(' ');
  return `<div class="hx-sec"><div class="hx-h">DISKS · ${disks.length}</div> ${rows}</div>`;
}

function hxModels(models, why, now) {
  if (!Array.isArray(models) || !models.length) return '';
  return models.map((m) => {
    const folders = Array.isArray(m.folders) ? m.folders : [];
    const held = folders.filter((f) => (f.checkpoints || []).length);
    const other = folders.filter((f) => !(f.checkpoints || []).length);
    // `more_*` null: the folders past the named ones were not counted (`more_why`) — the totals are then "—", not a sum
    // that leaves them out.
    const known = hxNum(m.more_count) && hxNum(m.more_bytes);
    const otherN = known ? other.length + m.more_count : null;
    const otherB = known ? other.reduce((a, f) => a + f.bytes, 0) + m.more_bytes : null;
    const total = known ? folders.reduce((a, f) => a + f.bytes, 0) + m.more_bytes : null;
    const age = hxNum(m.measured_at) && hxNum(now) ? `${hxAge(Math.max(0, now - m.measured_at))} ago` : null;
    const head = `MODELS ${hxEsc(m.root)} · ${m.why || total === null ? '—' : `${hxGB(total)} GB`}`
      + (age ? ` · <span title="walk took ${hxEsc(m.took_s)} s">${age}</span>` : '');
    const rows = held.map((f) => `<div class="hx-f" title="${hxEsc(f.checkpoints.join('\n'))}">`
      + `<span>${hxEsc(f.name)}</span> <b>${hxGB(f.bytes)} GB · ${f.checkpoints.length} ckpt</b></div> `).join('')
      + (!known ? `<div class="hx-f other none" title="${hxEsc(m.more_why || 'more_count not sent')}"><span>other —</span> <b>—</b></div>`
        : otherN ? `<div class="hx-f other"><span>other ${otherN}</span> <b>${hxGB(otherB)} GB</b></div>` : '')
      + ((m.unreadable || []).length ? ` <div class="hx-f none" title="${hxEsc(m.unreadable.join('\n'))}">`
        + `<span>unreadable ${m.unreadable.length}</span> <b>—</b></div>` : '');
    const title = [m.why, m.names_why].filter(Boolean).join(' · ');
    return `<div class="hx-sec"><div class="hx-h"${title ? ` title="${hxEsc(title)}"` : ''}>${head}</div> `
      + `${rows}</div>`;
  }).join(' ');
}

/** THE STRIP'S FIGURES: {parts: [{text, title, bad}], notes: [why…]}. RoCE first, then the disk — the
 *  order the open view's sections follow. `bad` only on a fault (hxDown, hxRising, hxFull); a block with nothing to
 *  count draws no part, and its reason, when the server gave one, is a note for the strip's title. */
function hwExtraStrip(h, s, box) {
  const parts = [], notes = [];
  if (!h) return { parts, notes };
  if (h.rdma) {
    const c = hxRdmaCounts(h.rdma, hxFinding(s, box, 'roce'));
    if (c.ports.length) {
      const name = (p) => `${p.dev}/${p.port}`;
      const faults = HX_COUNTERS.map((x) => x[0]).filter((k) => !HX_INFO.has(k));
      const title = [...c.downs.map((p) => `${name(p)} ${p.state}`),
        ...c.unread.map((p) => `${name(p)} state unread: ${hxStateWhy(p)}`),
        ...c.risers.map(([p, rose]) => `${name(p)} rising ${HX_COUNTERS.filter(([k]) => faults.includes(k)
          && ((hxNum((p.rates || {})[k]) && p.rates[k] > 0) || hxNum(rose[k]))).map((x) => x[1]).join(', ')}`)];
      parts.push({ text: `RoCE ${c.up}/${c.ports.length} up${c.unread.length ? ` · ${c.unread.length} unread` : ''}`
        + `${c.risers.length ? ` · ${c.risers.length} rising` : ''}`,
        title: title.join(' · '), bad: c.downs.length > 0 || c.risers.length > 0 });
    } else if (h.rdma.why) notes.push(`RoCE: ${h.rdma.why}`);
  }
  if (Array.isArray(h.disks) && h.disks.length) {
    const read = h.disks.map((d) => [d, hxPct(d)]).filter(([, pct]) => pct !== null);
    const fullest = read.reduce((a, b) => (!a || b[1] > a[1] ? b : a), null);
    const lines = h.disks.map((d) => {
      const pct = hxPct(d);
      return pct === null ? `${d.path}: ${d.why || 'not read'}` : `${d.path} ${hxDiskFig(d, pct)}`;
    });
    const at = hxFullAt(s, box);
    if (fullest) lines.push(at.note);
    parts.push(fullest
      ? { text: `disk ${Math.round(fullest[1])}%`, title: lines.join(' · '), bad: read.some(([d]) => hxFull(d, at)) }
      : { text: 'disk —', title: lines.join(' · '), bad: false });
  } else if (Array.isArray(h.disks) && h.disks_why) notes.push(`disk: ${h.disks_why}`);
  return { parts, notes };
}

function hwExtraHtml(h, s, box, chosen) {
  if (!h) return '';
  // Sections, rows and chips are parted by whitespace in the markup, not only by the layout: the text alone (a copy, a
  // screen reader, a detached node's innerText) never glues two figures ("OOB 980" + "15 —" read "98015").
  const body = [hxRdma(h.rdma, hxFinding(s, box, 'roce'), (s || {}).now, chosen),
    hxDisks(h.disks, h.disks_why, hxFullAt(s, box)), hxModels(h.models, h.models_why, (s || {}).now)]
    .filter(Boolean).join(' ');
  return body ? `<div class="hx">${body}</div>` : '';
}

if (typeof module !== 'undefined') module.exports = { hwExtraFigs, hwExtraHtml, hwExtraStrip };
