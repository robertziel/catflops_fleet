/* vllm-power — the fleet's power controls, added to the vLLM Gateway panel.
 *
 * the edge's nginx injects `<script src="/power/button.js">` into the panel's HTML (only the page at
 * `/`); the panel's own code (vllm-lb) is not changed. This script:
 *   - puts a ⏻ control on the head of each MACHINE card (box-a, box-b, box-c). The panel
 *     rebuilds its cards every 3 s, so a MutationObserver puts the control back each time;
 *   - next to it, on a box with a GPU, a clock pill: the GPU clock cap that used to be a
 *     systemd unit on every boot, now applied (and released) by a click. One option per cap the
 *     service offers (300-2200, 300-2300) plus the release; every action, its title, its MHz and
 *     its consequence come from the service — nothing here names or chooses a frequency;
 *   - polls /power/api/state for each box's live state (on / off / powering off / waking);
 *   - opens a confirm dialog: the consequences, the cluster state (read-only /lb/status),
 *     an explicit "the cluster will drop" tick when it runs, and a press-and-hold button: hold it
 *     down for 5 s (a bar fills across it) and the action runs; let go earlier and nothing does.
 *     The service enforces the wait on its side too (it refuses a confirmation younger than the
 *     hold), so this is an accident guard exactly like the typed box name it replaced.
 * All text from the server goes in by textContent. Nothing here calls the gateway's control API.
 */
(() => {
  'use strict';
  if (window.__vllmPower) return;
  window.__vllmPower = true;

  const API = '/power/api';
  const POLL_MS = 5000;
  let state = null;          // last /power/api/state
  let stateError = '';
  let focusBox = null;       // the box whose ⏻ had keyboard focus before a card rebuild

  const PHASE = {
    on: ['on', 'ok'], off: ['off', 'off'], powering_off: ['powering off…', 'warn'],
    waking: ['waking…', 'warn'], wake_timeout: ['did not wake', 'bad'],
  };

  // ── style: the panel's own tokens (gateway.css :root), nothing restyled ─────────────────
  const css = `
.pw-ctl { display: inline-flex; align-items: center; gap: .4rem; margin-left: auto; flex: none; }
.pw-phase { font: 600 var(--small, .7rem)/1 var(--mono, monospace); letter-spacing: .04em;
  text-transform: uppercase; color: var(--ink-faint, #818a94); }
.pw-phase.ok { color: var(--ok, #3fb950); } .pw-phase.warn { color: var(--warn, #d29922); }
.pw-phase.bad { color: var(--bad, #f85149); }
.pw-btn { width: 1.9rem; height: 1.9rem; border-radius: 50%; border: 1px solid var(--line, #2b313c);
  background: var(--panel-2, #1b2129); color: var(--ink-dim, #8b949e); font-size: 1rem; line-height: 1;
  cursor: pointer; display: inline-grid; place-items: center; padding: 0; }
.pw-btn:hover { color: var(--bad, #f85149); border-color: rgba(248, 81, 73, .6); }
.pw-btn.on-action:hover { color: var(--ok, #3fb950); border-color: rgba(63, 185, 80, .6); }
.pw-btn:focus-visible { outline: 2px solid var(--accent, #58a6ff); outline-offset: 2px; }
.pw-btn[disabled] { opacity: .45; cursor: default; }
.pw-clk { height: 1.9rem; padding: 0 .5rem; border-radius: .95rem; border: 1px solid var(--line, #2b313c);
  background: var(--panel-2, #1b2129); color: var(--ink-dim, #8b949e); cursor: pointer;
  font: 600 var(--small, .7rem)/1 var(--mono, monospace); letter-spacing: .03em; white-space: nowrap; }
.pw-clk:hover { color: var(--accent, #58a6ff); border-color: rgba(88, 166, 255, .6); }
.pw-clk.limited { color: var(--warn, #d29922); border-color: rgba(210, 153, 34, .5); }
.pw-clk:focus-visible { outline: 2px solid var(--accent, #58a6ff); outline-offset: 2px; }
.pw-clk[disabled] { opacity: .45; cursor: default; }
.pw-dialog .pw-opt { display: flex; flex-direction: column; align-items: stretch; gap: .6rem;
  margin: .7rem 0; border: 1px solid var(--line, #2b313c); border-radius: 8px; padding: .6rem .7rem; }
.pw-dialog .pw-opt button { flex: none; border: 0; border-radius: 6px; padding: .45rem .7rem;
  font-weight: 600; cursor: pointer; background: #21262d; color: var(--ink, #e6edf3); }
.pw-dialog .pw-opt button[disabled] { background: #30363d; color: var(--ink-faint, #818a94); cursor: default; }
.pw-dialog .pw-opt p { margin: 0; }
.pw-dialog .pw-now { color: var(--ink, #e6edf3); }
.pw-dialog { background: var(--panel, #161b22); color: var(--ink, #e6edf3); border: 1px solid var(--line, #2b313c);
  border-radius: 10px; padding: 1.1rem 1.25rem; width: min(30rem, calc(100vw - 2rem));
  max-height: calc(100vh - 2rem); overflow: auto; }
.pw-dialog::backdrop { background: rgba(1, 4, 9, .7); }
.pw-dialog h2 { margin: 0 0 .6rem; font-size: 1.05rem; }
.pw-dialog p { margin: .45rem 0; color: var(--ink-dim, #8b949e); }
.pw-dialog .pw-cons { color: var(--ink, #e6edf3); }
.pw-dialog .pw-warn { color: var(--warn, #d29922); }
.pw-dialog .pw-bad { color: var(--bad, #f85149); }
.pw-dialog .pw-dry { font: 600 var(--small, .7rem)/1.4 var(--mono, monospace); color: var(--park, #6ea8fe);
  border: 1px dashed var(--park, #6ea8fe); border-radius: 6px; padding: .3rem .5rem; }
.pw-dialog label { display: flex; gap: .5rem; align-items: flex-start; margin: .6rem 0; }
.pw-hold { --pw-c: #b62324; position: relative; overflow: hidden; display: block; width: 100%;
  margin-top: .9rem; border: 1px solid var(--pw-c); border-radius: 8px; padding: .8rem 1rem;
  background: rgba(182, 35, 36, .16); color: var(--ink, #e6edf3); font-weight: 700; cursor: pointer;
  touch-action: none; user-select: none; -webkit-user-select: none; -webkit-touch-callout: none; }
.pw-hold.on-action { --pw-c: #238636; background: rgba(35, 134, 54, .16); }
.pw-hold .pw-fill { position: absolute; inset: 0; background: var(--pw-c); transform-origin: left center;
  transform: scaleX(0); pointer-events: none; }
.pw-hold .pw-lbl { position: relative; font-variant-numeric: tabular-nums; }
.pw-hold.holding { box-shadow: 0 0 0 3px rgba(255, 255, 255, .18); }
.pw-hold:focus-visible { outline: 2px solid var(--accent, #58a6ff); outline-offset: 2px; }
.pw-hold[disabled] { border-color: var(--line, #2b313c); background: #30363d; color: var(--ink-faint, #818a94);
  cursor: default; }
.pw-hold[disabled] .pw-fill { background: transparent; }
.pw-row { display: flex; justify-content: flex-end; gap: .5rem; margin-top: .9rem; }
.pw-row button { border: 0; border-radius: 6px; padding: .5rem .9rem; font-weight: 600; cursor: pointer;
  background: #21262d; color: var(--ink, #e6edf3); }
.pw-out { font: var(--small, .7rem)/1.45 var(--mono, monospace); white-space: pre-wrap; overflow-wrap: anywhere;
  background: var(--bg, #0d1117); border-radius: 6px; padding: .4rem .55rem; max-height: 9rem; overflow: auto; }
`;
  const style = document.createElement('style');
  style.textContent = css;
  document.head.appendChild(style);

  const el = (tag, props = {}, ...kids) => {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(props)) {
      if (k === 'class') n.className = v;
      else if (k === 'text') n.textContent = v;
      else if (k.startsWith('on')) n.addEventListener(k.slice(2), v);
      else n.setAttribute(k, v);
    }
    for (const kid of kids) if (kid) n.append(kid);
    return n;
  };

  async function api(path, body) {
    const opt = body === undefined ? { credentials: 'same-origin', cache: 'no-store' } : {
      method: 'POST', credentials: 'same-origin', cache: 'no-store',
      headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    };
    const r = await fetch(API + path, opt);
    let j = null;
    try { j = await r.json(); } catch (_) { j = null; }
    return { status: r.status, body: j || {} };
  }

  // ── the per-card control ─────────────────────────────────────────────────────────────────
  function boxOfCard(card) {
    const name = card.querySelector('.box-name');
    if (!name || !name.firstChild) return null;
    const t = (name.firstChild.nodeType === 3 ? name.firstChild.nodeValue : name.textContent).trim();
    return t;
  }

  function actionFor(b) {
    if (!b) return null;
    if (b.phase === 'on') return 'poweroff';
    if ((b.phase === 'off' || b.phase === 'wake_timeout') && b.can_power_on) return 'poweron';
    return null;
  }

  // What the pill says, from what the service recorded: which cap, not just "capped".
  const CLK_LABEL = { limited: (c) => `${c.applied_mhz} MHz`, released: () => 'no limit' };

  function paintClk(clk, box) {
    const b = state && state.boxes && state.boxes[box];
    const c = b && b.clock;
    if (!c) return;
    const applied = c.applied;
    clk.textContent = (CLK_LABEL[applied] || (() => 'GPU clk'))(c);
    clk.classList.toggle('limited', applied === 'limited');
    clk.disabled = !b || b.phase !== 'on';
    const caps = (c.limits_mhz || []).map((m) => `${c.min_mhz}-${m}`).join(' / ');
    const what = applied
      ? `GPU clock — this panel last set: ${clk.textContent}. Click to change.`
      : `GPU clock limit for ${box} (${caps} MHz, or no limit)…`;
    clk.title = what;
    clk.setAttribute('aria-label', what);
  }

  function paintCtl(ctl, box) {
    const b = state && state.boxes && state.boxes[box];
    const phase = ctl.querySelector('.pw-phase');
    const btn = ctl.querySelector('.pw-btn');
    const clk = ctl.querySelector('.pw-clk');
    if (clk) paintClk(clk, box);
    const [label, tone] = b ? (PHASE[b.phase] || [b.phase, 'off']) : ['power ?', 'off'];
    phase.textContent = b ? label : (stateError ? 'power n/a' : '');
    phase.className = 'pw-phase ' + tone;
    const act = actionFor(b);
    btn.disabled = !act;
    btn.classList.toggle('on-action', act === 'poweron');
    const what = act === 'poweroff' ? `Power off ${box}…`
      : act === 'poweron' ? `Power on ${box} (Wake-on-LAN)…`
        : b ? `${box}: ${label}${b.can_power_on ? '' : ' — ' + b.power_on_note}` : `${box}: power state unknown`;
    btn.title = what;
    btn.setAttribute('aria-label', what);
  }

  function decorate() {
    const cards = document.querySelectorAll('#lb-backends article.card');
    for (const card of cards) {
      const box = boxOfCard(card);
      if (!box || !state || !state.boxes || !(box in state.boxes)) continue;
      const head = card.querySelector('.card-head');
      if (!head) continue;
      let ctl = head.querySelector('.pw-ctl');
      if (!ctl) {
        const b0 = state.boxes[box];
        ctl = el('span', { class: 'pw-ctl', 'data-box': box },
          el('span', { class: 'pw-phase' }),
          b0 && b0.clock ? el('button', { class: 'pw-clk', type: 'button', text: 'GPU clk',
            onclick: () => openClockDialog(box) }) : null,
          el('button', { class: 'pw-btn', type: 'button', text: '⏻',
            onclick: () => openDialog(box), onfocus: () => { focusBox = box; },
            // Moving away clears it; the card rebuild removing the button does not.
            onblur: (e) => { const t = e.target; setTimeout(() => { if (t.isConnected) focusBox = null; }, 0); } }));
        head.appendChild(ctl);
        if (focusBox === box && document.activeElement === document.body) ctl.querySelector('.pw-btn').focus();
      }
      paintCtl(ctl, box);
    }
  }

  async function poll() {
    try {
      const r = await api('/state');
      if (r.status === 200 && r.body.boxes) { state = r.body; stateError = ''; }
      else stateError = (r.body.error && r.body.error.message) || `HTTP ${r.status}`;
    } catch (e) { stateError = String(e); }
    decorate();
  }

  // ── the dialog ───────────────────────────────────────────────────────────────────────────
  const dlg = el('dialog', { class: 'pw-dialog', 'aria-labelledby': 'pw-title' });
  document.body.appendChild(dlg);

  async function clusterLine() {
    try {
      const r = await fetch('/lb/status', { cache: 'no-store' });
      const s = await r.json();
      const c = s.cluster || {};
      return c.configured ? `${c.state || 'unknown'}${c.detail ? ' — ' + c.detail : ''}` : 'not configured';
    } catch (_) { return 'unknown (the gateway status did not answer)'; }
  }

  // ── press and hold ───────────────────────────────────────────────────────────────────────
  // `onDone(heldMs)` fires once the button has been held down for `ms` without a let-go; letting
  // go earlier (or sliding off it, or losing focus) cancels and empties the bar. Pointer (mouse,
  // pen, touch) and keyboard (Space / Enter held down; auto-repeat ignored) both work. A plain
  // click does nothing. The service checks the wait itself, too: see app.py HOLD_S.
  function holdButton({ ms, label, on, onDone }) {
    const secs = ms / 1000;
    const fill = el('span', { class: 'pw-fill' });
    const lbl = el('span', { class: 'pw-lbl', text: label });
    const btn = el('button', { type: 'button', class: 'pw-hold' + (on ? ' on-action' : ''),
      'aria-label': `${label}: press and hold for ${secs} seconds` }, fill, lbl);
    let t0 = 0, raf = 0, done = false;
    const paint = (p) => { fill.style.transform = `scaleX(${p})`; };
    const stop = () => { cancelAnimationFrame(raf); raf = 0; t0 = 0; btn.classList.remove('holding'); };
    function frame() {
      const held = performance.now() - t0;
      paint(Math.min(1, held / ms));
      if (held >= ms) {
        stop(); done = true;
        onDone(Math.round(held));
        return;
      }
      lbl.textContent = `Keep holding… ${Math.ceil((ms - held) / 1000)}`;
      raf = requestAnimationFrame(frame);
    }
    function start(e) {
      e.preventDefault();
      if (btn.disabled || done || t0) return;
      t0 = performance.now();
      btn.classList.add('holding');
      raf = requestAnimationFrame(frame);
    }
    function cancel() {
      if (!t0 || done) return;
      stop(); paint(0);
      lbl.textContent = `Let go too early — hold for the full ${secs} s`;
    }
    const isKey = (e) => e.key === ' ' || e.key === 'Enter';
    btn.addEventListener('pointerdown', (e) => { if (e.button === 0) start(e); });
    for (const ev of ['pointerup', 'pointercancel', 'pointerleave', 'blur']) btn.addEventListener(ev, cancel);
    btn.addEventListener('keydown', (e) => { if (isKey(e)) { if (e.repeat) e.preventDefault(); else start(e); } });
    btn.addEventListener('keyup', (e) => { if (isKey(e)) cancel(); });
    btn.addEventListener('click', (e) => e.preventDefault());          // a click is not a hold
    btn.addEventListener('contextmenu', (e) => e.preventDefault());    // a long touch is not a menu
    btn.cancelHold = cancel;
    btn.setLabel = (t) => { lbl.textContent = t; };
    btn.rearm = () => { done = false; paint(0); lbl.textContent = label; };
    return btn;
  }
  dlg.addEventListener('close', () => { const h = dlg.querySelector('.pw-hold'); if (h) h.cancelHold(); });

  async function openDialog(box) {
    const b = state && state.boxes && state.boxes[box];
    const action = actionFor(b);
    if (!action) return;
    const on = action === 'poweron';
    dlg.replaceChildren(el('h2', { id: 'pw-title', text: on ? `Power on ${box}?` : `Power off ${box}?` }),
      el('p', { text: 'Asking the power service…' }));
    if (!dlg.open) dlg.showModal();

    const [conf, cl] = await Promise.all([api('/confirm', { box, action }), b.in_cluster ? clusterLine() : null]);
    if (conf.status !== 200) {
      dlg.replaceChildren(el('h2', { id: 'pw-title', text: `${box}: not now` }),
        el('p', { class: 'pw-bad', text: (conf.body.error && conf.body.error.message) || `HTTP ${conf.status}` }),
        el('div', { class: 'pw-row' }, el('button', { type: 'button', text: 'Close', onclick: () => dlg.close() })));
      return;
    }
    const c = conf.body;
    const ms = c.hold_ms || 5000;
    const ack = c.needs_cluster_ack ? el('input', { type: 'checkbox' }) : null;
    const out = el('div', { class: 'pw-out', hidden: '' });
    const cancel = el('button', { type: 'button', text: 'Cancel', onclick: () => dlg.close() });
    const hold = holdButton({
      ms, on, label: on ? `Hold to power on ${box}` : `Hold to power off ${box}`,
      onDone: async (heldMs) => {
        if (!dlg.open) return;
        hold.disabled = true; if (ack) ack.disabled = true;
        hold.setLabel(on ? 'Waking…' : 'Powering off…');
        const r = await api('/execute', { token: c.token, held_ms: heldMs, ack_cluster: ack ? ack.checked : false });
        out.hidden = false;
        const msg = r.body.error ? r.body.error.message : (r.body.output || r.body.result || '');
        out.textContent = `${r.status === 200 ? (r.body.dry_run ? 'DRY RUN — ' : '') + r.body.result : 'failed'}: ${msg}`;
        out.className = 'pw-out ' + (r.status === 200 ? '' : 'pw-bad');
        const again = r.body.error && ['hold_too_short', 'too_soon'].includes(r.body.error.type);
        if (again) {             // the same confirmation is still good: hold again
          hold.rearm(); hold.disabled = false; if (ack) ack.disabled = false;
        } else {
          hold.setLabel(r.status === 200 ? 'Done' : 'Failed');
          cancel.textContent = 'Close';
        }
        poll();
      },
    });
    const refresh = () => { hold.disabled = !!(ack && !ack.checked); };
    if (ack) ack.addEventListener('change', refresh);
    refresh();

    dlg.replaceChildren(...[   // replaceChildren would print a null as the text "null"
      el('h2', { id: 'pw-title', text: on ? `Power on ${box}?` : `Power off ${box}?` }),
      c.dry_run ? el('p', { class: 'pw-dry', text: 'DRY RUN — the service uses its test key: the box runs a harmless check instead of powering off.' }) : null,
      el('p', { class: 'pw-cons', text: on ? c.power_on_note : c.consequence }),
      !on && b.can_power_on ? el('p', { text: `Power-on afterwards: ${c.power_on_note}` }) : null,
      !on && !b.can_power_on ? el('p', { class: 'pw-warn', text: `Power-on afterwards: ${c.power_on_note}` }) : null,
      cl !== null ? el('p', { class: c.needs_cluster_ack ? 'pw-warn' : '', text: `Model cluster: ${cl}` }) : null,
      ack ? el('p', { class: 'pw-warn', text: 'Best: stop the cluster first with "Stop cluster" above.' }) : null,
      ack ? el('label', {}, ack, el('span', { text: 'I know the model cluster will drop' })) : null,
      el('p', { class: 'pw-hint', text: `Press and hold the button for ${ms / 1000} s: the bar fills, then it runs. Let go earlier and nothing happens.` }),
      hold,
      out,
      el('div', { class: 'pw-row' }, cancel)].filter(Boolean));
    (ack || hold).focus();
  }

  // ── the GPU clock dialog ─────────────────────────────────────────────────────────────────
  // No press-and-hold here: a clock cap is reversible, applies in place and drops no request.
  // The actions, their titles and their consequences all come from /power/api/state — one option
  // per cap the service offers, in its order, plus the release.
  function ago(state_now, at) {
    const s = Math.max(0, Math.round(state_now - at));
    if (s < 60) return `${s}s ago`;
    if (s < 3600) return `${Math.round(s / 60)} min ago`;
    return `${Math.round(s / 3600)} h ago`;
  }

  async function openClockDialog(box) {
    const b = state && state.boxes && state.boxes[box];
    const c = b && b.clock;
    if (!c || b.phase !== 'on') return;
    const out = el('div', { class: 'pw-out', hidden: '' });
    const opts = Object.entries(c.actions).map(([action, a]) => {
      const go = el('button', { type: 'button', text: a.title });
      go.addEventListener('click', () => apply(action, go, a.title));
      return el('div', { class: 'pw-opt' }, el('p', { text: a.consequence }), go);
    });
    const buttons = () => opts.map((o) => o.querySelector('button'));

    async function apply(action, go, title) {
      buttons().forEach((x) => { x.disabled = true; });
      go.textContent = 'Applying…';
      let r = await api('/confirm', { box, action });
      if (r.status === 200 && r.body.token) r = await api('/execute', { token: r.body.token });
      const okay = r.status === 200 && !r.body.error;
      const msg = r.body.error ? r.body.error.message : (r.body.output || r.body.result || '');
      out.hidden = false;
      out.textContent = `${okay ? (r.body.dry_run ? 'DRY RUN — ' : '') + r.body.result : 'failed'}: ${msg}`;
      out.className = 'pw-out ' + (okay ? '' : 'pw-bad');
      go.textContent = okay ? 'Done' : title;
      if (!okay) buttons().forEach((x) => { x.disabled = false; });
      poll();
    }

    const now = (state && state.now) || 0;
    const applied = c.applied
      ? `This panel last set: ${(CLK_LABEL[c.applied] || (() => c.applied))(c)}`
        + (c.applied_at ? ` (${ago(now, c.applied_at)}` + (c.applied_by ? `, by ${c.applied_by})` : ')') : '')
      : `This panel has set no clock on ${box} since the service started.`;
    dlg.replaceChildren(...[
      el('h2', { id: 'pw-title', text: `GPU clock — ${box}` }),
      state.dry_run ? el('p', { class: 'pw-dry', text: 'DRY RUN — the service uses its test key: the box runs a harmless check instead of setting a clock.' }) : null,
      el('p', { class: 'pw-now', text: applied }),
      el('p', { text: c.applies_note }),
      el('p', { text: c.note }),
      ...opts,
      out,
      el('div', { class: 'pw-row' }, el('button', { type: 'button', text: 'Close', onclick: () => dlg.close() }))].filter(Boolean));
    if (!dlg.open) dlg.showModal();
    buttons()[0].focus();
  }

  const target = document.getElementById('lb-backends');
  if (target) new MutationObserver(decorate).observe(target, { childList: true });
  poll();
  setInterval(poll, POLL_MS);
})();
