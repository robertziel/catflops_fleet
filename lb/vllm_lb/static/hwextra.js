/* BOX HEALTH on a machine card — drawn from /lb/status `hardware.<box>` (vllm_lb/box_health.py):
 *
 *   hwExtraFigs(h)     the kernel's counts as two more figures beside MEMORY: XID and OOM. Numbers only; "≥n" when the
 *                      Xid count is not the whole boot (`xid_complete` false: the ring buffer no longer held its first
 *                      record), "—" when unread (the reason is the section's title below).
 *   hwExtraHtml(h, s)  under the box's graphs: one RoCE line per port (port · state · rate · MTU, then each error
 *                      counter's total and, when it moved, its per-second rate — a counter that is RISING is flagged in
 *                      the fault colour, a port not ACTIVE too), one bar per configured disk ("used / total GB · %"),
 *                      and each models root's folders by size, named by the fleet checkpoints they hold, with the walk's
 *                      age.
 *
 * THE CARD AGREES WITH THE BAND. A box's pending / firing findings (/lb/status `health.findings`, vllm_lb/health.py)
 * flag the card too: a `roce` finding marks every counter its `rose` names ("+1": the window's rise, even when this
 * sample's rate is 0) and counts its port as rising; an `xid` / `oom` finding flags that figure with its rise
 * ("2 · +2"). The rise is the finding's, read from the status — never a second rise computation here.
 *
 * Figures, not sentences (operator rule): every label carries a number; what is absent is "—" with its reason in the
 * title, never a 0. A box whose sender predates these blocks, or a stale box (hw.py withdraws them), draws nothing.
 * gateway.js `machineBody` calls both with (h, s, box) (the one hook); this file holds no state and reads no clock but
 * `s.now`.
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
  return `<span class="hx-cs hx-qos" title="${hxEsc(age)}">${fig('PFC', pfc, 'pfc')}${fig('trust', trust, 'trust')}`
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

/** One port: `mlx5_0/1 · ACTIVE · 200 Gb/s · MTU 9000`, then its counters; `rose` = the finding's counters here. */
function hxPort(p, rose, now) {
  const rates = p.rates || {};
  const why = { ...(p.why || {}), ...(p.rates_why || {}) };
  const down = p.state && p.state !== 'ACTIVE';
  const head = [`${p.dev}/${p.port}`, p.state || '—', hxNum(p.rate_gbps) ? `${p.rate_gbps} Gb/s` : '—',
    hxNum(p.mtu) ? `MTU ${p.mtu}` : 'MTU —'];
  const title = [p.netdev ? `netdev ${p.netdev}` : '', p.phys_state ? `phys ${p.phys_state}` : '',
    p.link_layer || '', why.mtu ? `mtu: ${why.mtu}` : '', why.rate_gbps ? `rate: ${why.rate_gbps}` : '']
    .filter(Boolean).join(' · ');
  const chips = HX_COUNTERS.map(([key, label, group, info]) => {
    const total = (p.counters || {})[key];
    if (!hxNum(total)) {
      return `<span class="hx-c none" title="${hxEsc(`${key}: ${why[key] || why[group] || 'not sent'}`)}">${label} —</span>`;
    }
    const r = rates[key];
    const moving = hxNum(r) && r > 0 && !info;
    const named = hxNum((rose || {})[key]);
    const t = `${key} · ${total.toLocaleString('en-US')} total`
      + (hxNum(r) ? ` · ${hxRate(r)}/s` : why[key] ? ` · rate: ${why[key]}` : '')
      + (named ? ` · +${rose[key]} in the finding's window` : '');
    const mark = moving || (info && hxNum(r) && r > 0) ? ` <i>+${hxRate(r)}/s</i>` : named ? ` <i>+${hxInt(rose[key])}</i>` : '';
    return `<span class="hx-c${moving || named ? ' up' : ''}" title="${hxEsc(t)}">${label} <b>${hxInt(total)}</b>${mark}</span>`;
  }).join('');
  return `<div class="hx-port${down ? ' down' : ''}"><span class="hx-ph" title="${hxEsc(title)}">`
    + `${head.map((x, i) => (i === 1 ? `<b>${hxEsc(x)}</b>` : hxEsc(x))).join(' · ')}</span>`
    + `<span class="hx-cs">${chips}</span>${hxQos(p, why, now)}</div>`;
}

function hxRdma(r, finding, now) {
  if (!r) return '';
  const ports = Array.isArray(r.ports) ? r.ports : [];
  if (!ports.length) return '';            // a box without RDMA: nothing to draw (the reason is in rdma.why)
  const roses = ports.map((p) => hxRose(p, finding));
  const info = new Set(HX_COUNTERS.filter((c) => c[3]).map((c) => c[0]));
  const rising = ports.filter((p, i) => Object.entries(p.rates || {}).some(([k, v]) => v > 0 && !info.has(k))
    || Object.keys(roses[i]).length).length;
  const down = ports.filter((p) => p.state && p.state !== 'ACTIVE').length;
  return `<div class="hx-sec"><div class="hx-h">RoCE · ${ports.length} port${ports.length === 1 ? '' : 's'}`
    + `${down ? ` · <b class="hx-bad">${down} down</b>` : ''}${rising ? ` · <b class="hx-bad">${rising} rising</b>` : ''}</div>`
    + `${ports.map((p, i) => hxPort(p, roses[i], now)).join('')}</div>`;
}

function hxDisks(disks, why) {
  if (!Array.isArray(disks)) return '';
  if (!disks.length) return why ? `<div class="hx-sec"><div class="hx-h" title="${hxEsc(why)}">DISKS · —</div></div>` : '';
  const rows = disks.map((d) => {
    const ok = hxNum(d.used_bytes) && hxNum(d.total_bytes) && d.total_bytes > 0;
    const pct = ok ? (100 * d.used_bytes) / d.total_bytes : null;
    const colour = ok && typeof loadColour === 'function' ? loadColour(pct) : null;
    const fig = ok ? `${hxGB(d.used_bytes)} / ${hxGB(d.total_bytes)} GB · ${Math.round(pct)}%` : '—';
    const title = ok ? `${d.path} · ${hxGB(d.free_bytes)} GB free` : `${d.path}: ${d.why || 'not read'}`;
    return `<div class="hx-disk${ok ? '' : ' none'}" title="${hxEsc(title)}"><span class="hx-dp">${hxEsc(d.path)}</span>`
      + `<span class="hx-bar"><i style="width:${ok ? Math.min(100, pct).toFixed(1) : 0}%${colour ? `;background:${colour}` : ''}"></i></span>`
      + `<b>${fig}</b></div>`;
  }).join('');
  return `<div class="hx-sec"><div class="hx-h">DISKS · ${disks.length}</div>${rows}</div>`;
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
      + `<span>${hxEsc(f.name)}</span><b>${hxGB(f.bytes)} GB · ${f.checkpoints.length} ckpt</b></div>`).join('')
      + (!known ? `<div class="hx-f other none" title="${hxEsc(m.more_why || 'more_count not sent')}"><span>other —</span><b>—</b></div>`
        : otherN ? `<div class="hx-f other"><span>other ${otherN}</span><b>${hxGB(otherB)} GB</b></div>` : '')
      + ((m.unreadable || []).length ? `<div class="hx-f none" title="${hxEsc(m.unreadable.join('\n'))}">`
        + `<span>unreadable ${m.unreadable.length}</span><b>—</b></div>` : '');
    const title = [m.why, m.names_why].filter(Boolean).join(' · ');
    return `<div class="hx-sec"><div class="hx-h"${title ? ` title="${hxEsc(title)}"` : ''}>${head}</div>${rows}</div>`;
  }).join('');
}

function hwExtraHtml(h, s, box) {
  if (!h) return '';
  const body = hxRdma(h.rdma, hxFinding(s, box, 'roce'), (s || {}).now) + hxDisks(h.disks, h.disks_why) + hxModels(h.models, h.models_why, (s || {}).now);
  return body ? `<div class="hx">${body}</div>` : '';
}

if (typeof module !== 'undefined') module.exports = { hwExtraFigs, hwExtraHtml };
