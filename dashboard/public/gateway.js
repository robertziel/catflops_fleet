/* The gateway view: live LB state, and a chat that streams through the same origin.
 *
 * Nothing here dials the LB directly — every call goes to this app, which proxies. That is
 * what lets one tunnelled hostname behind one authenticated nginx vhost be sufficient, and
 * keeps :8010 off the public internet.
 */
'use strict';

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

// ── status ───────────────────────────────────────────────────────────────────

function depthOf(b) { return b.waiting + Math.max(b.running, b.inflight); }

function renderSummary(s) {
  const maxDepth = (s.config && s.config.vllm_max_depth);
  const kv = [
    ['backends live', s.backends.filter((b) => b.live).length + ' / ' + s.backends.length],
    ['vllm capacity', s.vllm_capacity === null ? '?' : s.vllm_capacity],
    ['leases', s.leases.length + ' (want ' + s.desired_leases + ')'],
    ['requests', s.requests_total],
    ['max depth', maxDepth === undefined ? '?' : maxDepth],
    ['broker db', s.db_ok ? 'ok' : 'FAIL'],
  ];
  $('lb-summary').innerHTML = kv.map(([k, v]) =>
    `<div class="kv"><b>${esc(v)}</b><span>${esc(k)}</span></div>`).join('');
}

function renderBackends(s) {
  const maxDepth = (s.config && s.config.vllm_max_depth);
  $('lb-backends').innerHTML = s.backends.map((b) => {
    const d = depthOf(b);
    const demoted = b.kind === 'vllm' && b.live &&
      (b.depth_known === false || (maxDepth !== undefined && d > maxDepth));
    const stalled = b.live && b.tokens_per_s === 0 && d > 0;
    const cls = !b.live ? 'down' : (stalled ? 'wedged' : (demoted ? 'demoted' : 'live'));
    const rate = b.tokens_per_s;
    const bits = [
      b.kind,
      b.live ? 'live' : 'down',
      'depth ' + d + ' (w' + b.waiting + '/r' + b.running + '/f' + b.inflight + ')',
      // MEASURED across probes. null = not yet known (one sample, or the counter reset because
      // the sidecar restarted). 0 with a queue behind it is the WEDGED signature.
      rate === null || rate === undefined ? 'tok/s ?' : rate.toFixed(1) + ' tok/s',
    ];
    if (b.gen_tokens_total) bits.push('gen ' + b.gen_tokens_total.toLocaleString());
    const wedged = b.live && rate === 0 && d > 0;
    if (wedged) bits.push('WEDGED — queued but generating nothing');
    if (b.host_label) bits.push(b.host_label);
    if (b.depth_known === false) bits.push('depth unknown');
    if (demoted) bits.push('DEMOTED');
    if (b.served_models && b.served_models.length) bits.push(b.served_models.join(','));
    return `<div class="backend ${cls}">
      <div class="url">${esc(b.url)}</div>
      <div class="meta">${esc(bits.join(' · '))}</div>
      ${b.last_error ? `<div class="err">${esc(b.last_error)}</div>` : ''}
    </div>`;
  }).join('');

  renderLeases(s);
  const live = s.backends.filter((b) => b.live);
  const pick = live.find((b) => b.kind === 'vllm' && depthOf(b) <= (maxDepth === undefined ? 1 : maxDepth))
            || live.find((b) => b.kind === 'ollama') || live[0];
  $('chat-route').textContent = pick ? '→ ' + pick.url : '→ no backend is live';
}

function renderLeases(s) {
  const el = document.getElementById('lb-leases');
  if (!el) return;
  if (!s.leases.length) { el.innerHTML = '<div class="none">no node_job holds a server</div>'; return; }
  const byUrl = Object.fromEntries(s.backends.map((b) => [b.url, b]));
  el.innerHTML = s.leases.map((l) => {
    const b = l.backend ? byUrl[l.backend] : null;
    const age = Math.round((s.now - l.created_at) / 60);
    const bits = ['node_job ' + (l.job_id || '').slice(0, 8), l.status,
                  l.box ? 'on ' + l.box : 'unplaced', age + 'm old'];
    if (b) {
      bits.push(b.live ? 'server live' : 'server down');
      bits.push(b.tokens_per_s === null || b.tokens_per_s === undefined
        ? 'tok/s ?' : b.tokens_per_s.toFixed(1) + ' tok/s');
    }
    return `<div class="lease ${esc(l.status)}">
      <div class="url">${esc(l.backend || '(no server yet)')}</div>
      <div class="meta">${esc(bits.join(' · '))}</div></div>`;
  }).join('');
}

async function poll() {
  try {
    const r = await fetch('/api/lb');
    const s = await r.json();
    if (!r.ok) throw new Error(s.error || ('HTTP ' + r.status));
    renderSummary(s);
    renderBackends(s);
    $('status-text').textContent = 'connected';
    $('connection-status').classList.remove('bad');
  } catch (e) {
    $('status-text').textContent = e.message;
    $('connection-status').classList.add('bad');
  }
}

// ── chat ─────────────────────────────────────────────────────────────────────

const history = [];

function bubble(who, cls) {
  const el = document.createElement('div');
  el.className = 'msg ' + (cls || '');
  el.innerHTML = `<span class="who">${esc(who)}</span><span class="body"></span>`;
  $('chat-log').appendChild(el);
  $('chat-log').scrollTop = $('chat-log').scrollHeight;
  return el.querySelector('.body');
}

async function send(ev) {
  ev.preventDefault();
  const text = $('chat-input').value.trim();
  if (!text) return;
  $('chat-input').value = '';
  $('chat-send').disabled = true;
  bubble('you', 'user').textContent = text;
  history.push({ role: 'user', content: text });

  const out = bubble('assistant');
  const think = document.createElement('span');
  think.className = 'think';
  out.parentElement.appendChild(think);
  let answer = '';

  try {
    const r = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages: history, effort: $('chat-effort').value }),
    });
    if (!r.ok || !r.body) throw new Error('HTTP ' + r.status);
    const reader = r.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const parts = buf.split('\n\n');
      buf = parts.pop();
      for (const p of parts) {
        const line = p.split('\n').find((l) => l.startsWith('data:'));
        if (!line) continue;
        let d;
        try { d = JSON.parse(line.slice(5).trim()); } catch (_) { continue; }
        if (d.error) { out.parentElement.classList.add('err'); out.textContent = d.error.message || d.error; continue; }
        const delta = d.delta || {};
        if (delta.type === 'thinking_delta' && delta.thinking) think.textContent += delta.thinking;
        if (delta.type === 'text_delta' && delta.text) { answer += delta.text; out.textContent = answer; }
        $('chat-log').scrollTop = $('chat-log').scrollHeight;
      }
    }
    if (!answer && !think.textContent) out.textContent = '(no output)';
    if (answer) history.push({ role: 'assistant', content: answer });
  } catch (e) {
    out.parentElement.classList.add('err');
    out.textContent = e.message;
  } finally {
    $('chat-send').disabled = false;
  }
}

$('chat-form').addEventListener('submit', send);
$('chat-input').addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) send(e);
});
poll();
setInterval(poll, 3000);   // 'real time every few seconds'
