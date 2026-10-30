/* THE CONNECT DIALOG, checked where gateway.js builds it (operator, 2026-10-12: "Can we include somewhere connect
 * button, top navbar? Will show how to connect agent to this vllm"), under node with a stub DOM that keeps a real
 * tree (children, parents, focus, events) and a fetch answered per URL.
 *
 * Pinned: the top bar's `connect` opens a modal dialog; the base URL is the page's own `location.origin` (or the
 * SSH forward the deployment declares), never a literal; the model and its window are the LB's live /v1/models;
 * the door is what /auth/session answers (404 = the LB itself, `auth: "false"` = a gate without sign-in, 401 or a
 * session = the sign-in gate); /lb/status `connect` adds the tunnel, Basic and the LB's own timeouts; a door an API
 * client cannot pass SAYS so and offers no snippet for it; the Claude Code lines are fullclaudevllm's (window, 80 %
 * compact, output cap, timeouts from the LB, idle watchdog ≤ 30 min); no credential is ever a value; Esc closes and
 * gives focus back; Tab is trapped; copy writes the snippet; the tabs switch by click and arrow.
 *
 * PER-USER API KEYS (the sign-in gate's key store; operator, 2026-10-12): where the sign-in gate's /auth/session says
 * `api_keys`, this address is an API route too — Claude Code / curl / the SDKs with `<your key>`, never a value — and
 * the dialog creates a key (name, a lifetime the gate offers, its default preselected), shows it ONCE with copy,
 * lists the person's own keys (name, prefix + first 4…, created, expires, last used, never the key) and revokes one on a
 * second press. Closing the dialog forgets the shown key.
 *
 *   node tests/connect.node.js      (tests/test_connect_page.py runs this when node exists)
 */
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const STATIC = path.join(__dirname, '..', 'vllm_lb', 'static');
const src = fs.readFileSync(path.join(STATIC, 'gateway.js'), 'utf8');
const html = fs.readFileSync(path.join(STATIC, 'index.html'), 'utf8');
const css = fs.readFileSync(path.join(STATIC, 'gateway.css'), 'utf8');
let checks = 0;
const ok = (cond, msg) => { assert.ok(cond, msg); checks += 1; };
const eq = (a, b, msg) => { assert.strictEqual(a, b, msg); checks += 1; };

// ── a DOM that keeps a tree ──────────────────────────────────────────────────────────────────────────────────────
let document;
class El {
  constructor(tag) {
    this.tagName = String(tag || 'div').toUpperCase(); this.children = []; this.parentNode = null; this.attrs = {};
    this.dataset = {}; this.style = {}; this.hidden = false; this.disabled = false; this.className = ''; this.on = {};
    this._text = ''; this.value = ''; this.title = ''; this.type = ''; this.open = false;
    const self = this;
    this.classList = {
      add(c) { const s = new Set(self.className.split(/\s+/).filter(Boolean)); s.add(c); self.className = [...s].join(' '); },
      remove(c) { self.className = self.className.split(/\s+/).filter((x) => x && x !== c).join(' '); },
      toggle(c, on) { const has = self.classList.contains(c); const want = on === undefined ? !has : !!on;
        if (want) self.classList.add(c); else self.classList.remove(c); return want; },
      contains: (c) => self.className.split(/\s+/).includes(c),
    };
  }
  get textContent() { return this.children.length ? this.children.map((c) => c.textContent).join('') : this._text; }
  set textContent(v) { this.children.forEach((c) => { c.parentNode = null; }); this.children = []; this._text = String(v); }
  set innerHTML(v) { if (v !== '') throw new Error(`the connect dialog writes no HTML (got ${String(v).slice(0, 60)})`); this.textContent = ''; }
  get innerHTML() { return ''; }
  appendChild(c) { if (c.parentNode) c.parentNode.children.splice(c.parentNode.children.indexOf(c), 1);
    this.children.push(c); c.parentNode = this; return c; }
  append(...cs) { cs.forEach((c) => this.appendChild(typeof c === 'string' ? Object.assign(new El('#text'), { _text: c }) : c)); }
  replaceChildren(...cs) { this.textContent = ''; this.append(...cs); }
  remove() { if (this.parentNode) this.parentNode.children.splice(this.parentNode.children.indexOf(this), 1); this.parentNode = null; }
  addEventListener(type, fn) { (this.on[type] = this.on[type] || []).push(fn); }
  setAttribute(k, v) { this.attrs[k] = String(v); if (k === 'id') this.id = String(v); }
  getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; }
  removeAttribute(k) { delete this.attrs[k]; }
  hasAttribute(k) { return k in this.attrs; }
  focus() { document.activeElement = this; }
  blur() {}
  showModal() { this.open = true; }
  close() { this.open = false; }
  contains(x) { for (let n = x; n; n = n.parentNode) if (n === this) return true; return false; }
  closest(sel) {
    for (let n = this; n; n = n.parentNode) {
      if (sel === '[hidden]' && n.hidden) return n;
      if (sel.startsWith('#') && n.id === sel.slice(1)) return n;
    }
    return null;
  }
  *walk() { for (const c of this.children) { yield c; yield* c.walk(); } }
  /** The selectors the page asks of the dialog: tags, and `[data-x]` / `.class` — nothing else is understood. */
  querySelectorAll(sel) {
    const parts = sel.split(',').map((s) => s.trim());
    const hit = (n) => parts.some((p) => {
      if (/^[a-z]+$/.test(p)) return n.tagName === p.toUpperCase();
      if (/^[a-z]+\[href\]$/.test(p)) return n.tagName === p.split('[')[0].toUpperCase() && 'href' in n.attrs;
      if (p.startsWith('[data-')) return p.slice(6, -1).replace(/-([a-z])/g, (_, c) => c.toUpperCase()) in n.dataset;
      if (p.startsWith('.')) return n.classList.contains(p.slice(1));
      throw new Error(`stub querySelectorAll: ${p}`);
    });
    return [...this.walk()].filter(hit);
  }
  querySelector(sel) { return this.querySelectorAll(sel)[0] || null; }
  getBoundingClientRect() { return {}; }
  observe() {}
}
const textOf = (el) => (el ? el.textContent : '');
const visible = (el) => !el.closest('[hidden]');
function fire(el, type, extra = {}) {
  let prevented = false;
  const ev = { type, key: '', shiftKey: false, target: el, preventDefault() { prevented = true; }, stopPropagation() {}, ...extra };
  for (let n = el; n; n = n.parentNode) (n.on[type] || []).forEach((fn) => fn(ev));
  return prevented;
}

function load({ origin = 'https://gw.example.test', door = { status: 404, body: { error: { type: 'not_found' } } },
  models = { status: 200, body: { object: 'list', data: [{ id: 'Qwen/Qwen3.8-27B', object: 'model', max_model_len: 196608 }] } },
  connect = { tunnel: '', basic: false, first_byte_timeout_s: 3600, stall_timeout_s: 1800 }, clipboard = 'ok', stored = {},
  keysApi = null } = {}) {
  const ids = new Map();
  const byId = (id) => { if (!ids.has(id)) { const e = new El('div'); e.id = id; ids.set(id, e); } return ids.get(id); };
  document = { getElementById: byId, createElement: (t) => new El(t), createElementNS: (ns, t) => new El(t),
    addEventListener() {}, querySelectorAll: () => [], querySelector: () => null, activeElement: null,
    body: { style: {}, classList: { toggle() {} } }, documentElement: { style: {}, classList: { toggle() {} } } };
  const dlg = new El('dialog'); dlg.id = 'connect'; ids.set('connect', dlg);
  const opener = new El('button'); opener.id = 'cluster-connect'; ids.set('cluster-connect', opener);
  const store = { ...stored };
  const localStorage = { getItem: (k) => (k in store ? store[k] : null), setItem: (k, v) => { store[k] = String(v); } };
  const window = { addEventListener() {}, innerWidth: 1280, innerHeight: 800, localStorage,
    matchMedia: () => ({ matches: false, addEventListener() {} }) };
  const asked = [];
  const copied = [];
  const respond = (a) => Promise.resolve({ status: a.status, ok: a.status >= 200 && a.status < 300, json: async () => a.body });
  const fetch = (url, opts = {}) => {
    asked.push({ url, ...opts });
    if (url === '/lb/status') return respond({ status: 200, body: { connect, queue: null } });
    if (url === '/v1/models') return respond(models);
    if (url === '/auth/session') return respond(door);
    if (keysApi && (url === '/auth/keys' || url === '/auth/keys/revoke')) return respond(keysApi(url, opts));
    return new Promise(() => {});
  };
  const navigator = { userAgent: 'node', clipboard: { writeText: (t) => { copied.push(t);
    return clipboard === 'ok' ? Promise.resolve() : Promise.reject(new Error(clipboard)); } } };
  const api = new Function('document', 'window', 'fetch', 'location', 'setInterval', 'setTimeout', 'clearTimeout',
    'getComputedStyle', 'ResizeObserver', 'localStorage', 'navigator', 'requestAnimationFrame', 'cancelAnimationFrame',
    `${src}\n;return { openConnect, closeConnect, connectDoor, connectRoutes, connectSnippets,
       setLastStatus: (s) => { lastStatus = s; } };`)(
    document, window, fetch, { search: '', href: origin + '/', origin }, () => 0, () => 0, () => 0,
    () => ({}), undefined, localStorage, navigator, () => 0, () => {});
  api.setLastStatus({ connect });
  return { api, dlg, opener, asked, copied, store, byId };
}
const settle = async () => { for (let i = 0; i < 20; i += 1) await null; };
const find = (dlg, pred) => [...dlg.walk()].find(pred);
const findAll = (dlg, pred) => [...dlg.walk()].filter(pred);
const tabs = (dlg) => findAll(dlg, (n) => n.getAttribute('role') === 'tab');
const snippets = (dlg) => findAll(dlg, (n) => n.tagName === 'PRE' && visible(n) && n.dataset.snippet !== undefined)
  .map((n) => n.textContent);
const allText = (dlg) => textOf(dlg);

(async () => {
  // ── 0. the button is the SERVER card's (its pages row — connect is per server, 2026-10-17); nothing in the top bar
  const top = html.split('<header class="top">')[1].split('</header>')[0];
  ok(!/connect-open|href="\/benchmarks"|href="\/sessions"/.test(top), 'the top bar carries no server page or connect');
  const card = html.split('id="cluster-links"')[1].split('</nav>')[0];
  ok(/<button[^>]*id="cluster-connect"[^>]*aria-haspopup="dialog"/.test(card), "the connect button is in the server card's row and says it opens a dialog");
  ok(/<dialog[^>]*id="connect"[^>]*aria-labelledby="connect-title"/.test(html), 'the dialog is a <dialog>, labelled');
  ok(/\.cn-box[^{]*\{[^}]*max-width/.test(css), 'the dialog has a width bound for a phone');
  ok(/@media \(max-width: ?\d+px\)[^@]*\.cn-/.test(css), 'and a phone layout');

  // ── 1. the LB itself: base = origin, model + window live, fullclaudevllm's lines ────────────────────────────────
  {
    const { api, dlg, opener, asked } = load();
    opener.focus();
    await api.openConnect(); await settle();
    ok(dlg.open, 'the dialog opened (modal)');
    ok(asked.some((a) => a.url === '/v1/models'), 'the model is asked of the LB, live');
    ok(asked.some((a) => a.url === '/auth/session'), 'the door is asked at this origin');
    ok(asked.every((a) => a.url.startsWith('/')), 'every fetch is same-origin, relative');
    const t = allText(dlg);
    ok(t.includes('Qwen/Qwen3.8-27B') && t.includes('196,608'), `model + window: ${t.slice(0, 200)}`);
    ok(/the LB itself/.test(t) && /no credential/.test(t), 'the door says what it is');
    const [claude] = snippets(dlg);
    const want = ['ANTHROPIC_BASE_URL=https://gw.example.test', 'ANTHROPIC_MODEL=Qwen/Qwen3.8-27B',
      'ANTHROPIC_AUTH_TOKEN=unused', 'CLAUDE_CODE_MAX_CONTEXT_TOKENS=196608', 'CLAUDE_CODE_AUTO_COMPACT_WINDOW=157286',
      'CLAUDE_CODE_MAX_OUTPUT_TOKENS=16384', 'API_TIMEOUT_MS=3600000', 'CLAUDE_STREAM_FIRST_BYTE_TIMEOUT_MS=3600000',
      'CLAUDE_STREAM_IDLE_TIMEOUT_MS=1800000', 'API_FORCE_IDLE_TIMEOUT=0', 'CLAUDE_CODE_MAX_RETRIES=10',
      'CLAUDE_CODE_ATTRIBUTION_HEADER=0'];
    want.forEach((w) => ok(claude.split(' \\\n').includes(w), `Claude Code line ${w} in:\n${claude}`));
    ok(/\\\nclaude$/.test(claude), 'and ends by running claude');
    ok(!/ANTHROPIC_API_KEY/.test(claude), 'no API key line where none is asked (the token skips the key prompt)');
    // the OpenAI tab: curl + the python SDK; the Anthropic tab: curl
    const [t0, t1, t2] = tabs(dlg);
    eq(t0.getAttribute('aria-selected'), 'true', 'Claude Code is the first tab, selected');
    fire(t1, 'click');
    const [curl, py] = snippets(dlg);
    ok(curl.startsWith('curl https://gw.example.test/v1/chat/completions'), curl);
    ok(curl.includes(`"model":"Qwen/Qwen3.8-27B"`) && curl.includes('Reply with 7'), curl);
    ok(py.includes('base_url="https://gw.example.test/v1"') && py.includes('api_key="unused"'), py);
    ok(py.includes('timeout=3600') && py.includes('model="Qwen/Qwen3.8-27B"'), py);
    eq(t1.getAttribute('aria-selected'), 'true', 'the clicked tab is selected');
    eq(t0.getAttribute('aria-selected'), 'false', 'and only it');
    fire(t1, 'keydown', { key: 'ArrowRight' });
    eq(t2.getAttribute('aria-selected'), 'true', 'ArrowRight moves to the next tab');
    eq(document.activeElement, t2, 'and focuses it');
    const [msg] = snippets(dlg);
    ok(msg.startsWith('curl https://gw.example.test/v1/messages'), msg);
    ok(msg.includes("-H 'anthropic-version: 2023-06-01'") && msg.includes('"max_tokens":1024'), msg);
    fire(t2, 'keydown', { key: 'ArrowRight' });
    eq(t0.getAttribute('aria-selected'), 'true', 'and wraps');
  }
  {   // copy: the snippet shown, exactly
    const { api, dlg, copied } = load();
    await api.openConnect(); await settle();
    const copy = find(dlg, (n) => n.tagName === 'BUTTON' && n.dataset.copy !== undefined && visible(n));
    fire(copy, 'click'); await settle();
    eq(copied.length, 1, 'one copy');
    eq(copied[0], snippets(dlg)[0], 'the snippet shown is the text copied');
    ok(/copied/.test(copy.textContent), `the button says so: ${copy.textContent}`);
  }
  {
    const { api, dlg } = load({ clipboard: 'Write permission denied.' });
    await api.openConnect(); await settle();
    const copy = find(dlg, (n) => n.tagName === 'BUTTON' && n.dataset.copy !== undefined && visible(n));
    fire(copy, 'click'); await settle();
    ok(/fail/.test(copy.textContent) && /Write permission denied/.test(copy.title), `a refused copy says why: ${copy.textContent} / ${copy.title}`);
  }

  // ── 2. another origin, other LB budgets: everything follows, nothing is a literal ───────────────────────────────
  {
    const { api, dlg } = load({ origin: 'http://127.0.0.1:18031',
      connect: { tunnel: '', basic: false, first_byte_timeout_s: 2400, stall_timeout_s: 900 } });
    await api.openConnect(); await settle();
    const [claude] = snippets(dlg);
    ok(claude.includes('ANTHROPIC_BASE_URL=http://127.0.0.1:18031'), 'the base is the origin the page was opened at');
    ok(claude.includes('API_TIMEOUT_MS=2400000') && claude.includes('CLAUDE_STREAM_FIRST_BYTE_TIMEOUT_MS=2400000'),
      'the client waits as long as the LB does for a first byte');
    ok(claude.includes('CLAUDE_STREAM_IDLE_TIMEOUT_MS=900000'), 'and for silence');
  }
  {
    const { api, dlg } = load({ connect: { tunnel: '', basic: false, first_byte_timeout_s: 3600, stall_timeout_s: 7200 } });
    await api.openConnect(); await settle();
    ok(snippets(dlg)[0].includes('CLAUDE_STREAM_IDLE_TIMEOUT_MS=1800000'), 'the idle watchdog stops at its 30-min maximum');
  }
  const section = src.split('// ── connect an agent')[1].split('\n// ── ')[0];
  ok(section.length > 2000, 'the connect section is found');
  ok(section.includes('location.origin'), 'the base is location.origin');
  const literals = section.match(/https?:\/\/[^\s'"`$<]*/g) || [];
  ok(literals.every((u) => u === 'http://127.0.0.1:'), `no host but the client's own loopback is written: ${literals}`);
  ['8010', '8018', '18010'].forEach((w) =>
    ok(!section.toLowerCase().includes(w), `no deployment port ${w} in the connect section`));
  // The deployment's own words are tests/publishable_words.json's digests (one definition with the python guards):
  // every word and hyphenated word of the section, lowercase, hashed — none may be listed.
  const words = JSON.parse(fs.readFileSync(path.join(__dirname, 'publishable_words.json'), 'utf8'));
  const listed = new Set(words.digests);
  const sha = (t) => require('crypto').createHash('sha256').update(t).digest('hex');
  const lower = section.toLowerCase();
  const tokens = new Set([...(lower.match(/\w+/g) || []), ...(lower.match(/[\w-]+/g) || [])]);
  const hits = [...tokens].filter((t) => listed.has(sha(t)));
  ok(hits.length === 0, `no deployment word in the connect section: ${hits}`);
  ok(listed.has(sha(words.canary)) && [words.canary, 'x'].filter((t) => listed.has(sha(t))).length === 1,
    'the guard bites the planted canary and nothing else');

  // ── 3. the sign-in gate, nothing declared: says so, no snippet ─────────────────────────────────────────────────
  {
    const { api, dlg } = load({ door: { status: 401, body: { error: 'sign_in_required' } } });
    await api.openConnect(); await settle();
    const t = allText(dlg);
    ok(/sign-in/.test(t) && /401/.test(t), `the door says an API client gets 401: ${t}`);
    ok(/LB_CONNECT_TUNNEL/.test(t), 'and names what would declare a route');
    eq(snippets(dlg).length, 0, 'no snippet for a door an API client cannot pass');
    ok(!/ANTHROPIC_BASE_URL/.test(t), 'none anywhere');
  }
  {   // a signed-in browser reads the same door
    const { api, dlg } = load({ door: { status: 200, body: { email: 'a@b.test', expires_in: 300, renew_in: 120 } } });
    await api.openConnect(); await settle();
    ok(/sign-in/.test(allText(dlg)), 'a session is the sign-in door too');
    eq(snippets(dlg).length, 0, 'and still no snippet');
  }

  // ── 4. the SSH forward the deployment declares ─────────────────────────────────────────────────────────────────
  {
    const { api, dlg, store } = load({ door: { status: 401, body: {} },
      connect: { tunnel: '127.0.0.1:8010', basic: false, first_byte_timeout_s: 3600, stall_timeout_s: 1800 } });
    await api.openConnect(); await settle();
    const cmd = find(dlg, (n) => n.tagName === 'PRE' && n.dataset.ssh !== undefined);
    ok(cmd && visible(cmd), 'the forward is shown');
    eq(cmd.textContent, 'ssh -N -L 127.0.0.1:8010:127.0.0.1:8010 <server>', 'its command: the LB port by default');
    ok(snippets(dlg)[0].includes('ANTHROPIC_BASE_URL=http://127.0.0.1:8010'), 'the snippets use the forward');
    const host = find(dlg, (n) => n.tagName === 'INPUT' && n.dataset.field === 'host');
    const port = find(dlg, (n) => n.tagName === 'INPUT' && n.dataset.field === 'port');
    host.value = 'gpu-host'; fire(host, 'input');
    port.value = '18031'; fire(port, 'input');
    eq(cmd.textContent, 'ssh -N -L 127.0.0.1:18031:127.0.0.1:8010 gpu-host', 'typed host and port');
    ok(snippets(dlg)[0].includes('ANTHROPIC_BASE_URL=http://127.0.0.1:18031'), 'the snippets follow the local port');
    eq(store['vllm-connect-host'], 'gpu-host', 'the host is remembered for this viewer');
    eq(store['vllm-connect-port'], '18031', 'and the port');
    port.value = '99999'; fire(port, 'input');
    ok(/1.65535/.test(allText(dlg)) && port.getAttribute('aria-invalid') === 'true', 'a port out of range is refused, said');
    eq(snippets(dlg).length, 0, 'and no snippet is built on it');
  }
  {
    const { api, dlg } = load({ door: { status: 401, body: {} }, stored: { 'vllm-connect-host': 'box-a', 'vllm-connect-port': '18044' },
      connect: { tunnel: '127.0.0.1:8010', basic: false, first_byte_timeout_s: 3600, stall_timeout_s: 1800 } });
    await api.openConnect(); await settle();
    eq(find(dlg, (n) => n.dataset.ssh !== undefined).textContent, 'ssh -N -L 127.0.0.1:18044:127.0.0.1:8010 box-a',
      'a remembered host and port come back');
  }

  // ── 5. the sign-in gate that also takes Basic ──────────────────────────────────────────────────────────────────
  {
    const { api, dlg } = load({ door: { status: 401, body: {} },
      connect: { tunnel: '127.0.0.1:8010', basic: true, first_byte_timeout_s: 3600, stall_timeout_s: 1800 } });
    await api.openConnect(); await settle();
    const routes = findAll(dlg, (n) => n.tagName === 'BUTTON' && n.dataset.route !== undefined);
    eq(routes.map((r) => r.dataset.route).join(','), 'basic,ssh', 'Basic at this address first, then the forward');
    eq(routes[0].getAttribute('aria-pressed'), 'true', 'the first route is chosen');
    const [claude] = snippets(dlg);
    ok(claude.includes('ANTHROPIC_BASE_URL=https://gw.example.test'), 'Basic is at this address');
    ok(claude.includes('ANTHROPIC_API_KEY=unused'), 'an API key, so the custom header is not overridden by a bearer');
    ok(!claude.includes('ANTHROPIC_AUTH_TOKEN'), 'never a token beside Basic (Claude Code sends the bearer instead)');
    ok(claude.includes(`ANTHROPIC_CUSTOM_HEADERS="Authorization: Basic $(printf %s '<user>:<password>' | base64 | tr -d '\\n')"`), claude);
    fire(tabs(dlg)[1], 'click');
    const [curl, py] = snippets(dlg);
    ok(curl.includes("-u '<user>:<password>'"), curl);
    ok(py.includes('default_headers={"Authorization": "Basic " + base64.b64encode(b"<user>:<password>").decode()}'), py);
    fire(routes[1], 'click');
    eq(routes[1].getAttribute('aria-pressed'), 'true', 'a route is chosen by click');
    const [curl2] = snippets(dlg);
    ok(curl2.startsWith('curl http://127.0.0.1:8010/') && !curl2.includes('-u '), 'the forward needs no credential');
  }

  // ── 6. the gate without sign-in ────────────────────────────────────────────────────────────────────────────────
  {
    const { api, dlg } = load({ door: { status: 200, body: { email: 'anonymous', auth: 'false' } } });
    await api.openConnect(); await settle();
    const t = allText(dlg);
    ok(/Origin: https:\/\/gw\.example\.test/.test(t) && /403/.test(t), `the open gate says a POST needs Origin: ${t}`);
    eq(snippets(dlg).length, 0, 'no snippet when no route is declared');
  }

  // ── 7. what /v1/models did not say, said ───────────────────────────────────────────────────────────────────────
  {
    const { api, dlg } = load({ models: { status: 503, body: { error: 'no_backend' } } });
    await api.openConnect(); await settle();
    ok(/\/v1\/models/.test(allText(dlg)) && /503/.test(allText(dlg)), 'a model list that failed says its status');
    eq(snippets(dlg).length, 0, 'and no snippet names a model nobody served');
  }
  {
    const { api, dlg } = load({ models: { status: 200, body: { data: [{ id: 'x' }] } } });
    await api.openConnect(); await settle();
    ok(/max_model_len/.test(allText(dlg)), 'a model without a window says so');
    eq(snippets(dlg).length, 0, 'no window, no Claude Code lines guessed');
  }

  // ── 8. the keyboard: Esc closes and returns focus; Tab is trapped ──────────────────────────────────────────────
  {
    const { api, dlg, opener } = load();
    opener.focus();
    await api.openConnect(); await settle();
    ok(dlg.contains(document.activeElement), 'focus moves into the dialog');
    const focusables = dlg.querySelectorAll('button, input, a[href]').filter((n) => visible(n) && !n.disabled);
    const first = focusables[0], last = focusables[focusables.length - 1];
    last.focus();
    ok(fire(last, 'keydown', { key: 'Tab' }), 'Tab on the last is taken');
    eq(document.activeElement, first, 'and goes to the first');
    ok(fire(first, 'keydown', { key: 'Tab', shiftKey: true }), 'Shift+Tab on the first is taken');
    eq(document.activeElement, last, 'and goes to the last');
    ok(fire(first, 'keydown', { key: 'Escape' }), 'Escape is taken');
    ok(!dlg.open, 'Escape closes');
    eq(document.activeElement, opener, 'focus returns to the connect button');
    await api.openConnect(); await settle();
    ok(fire(dlg, 'cancel'), "the browser's own cancel is taken too");
    ok(!dlg.open && document.activeElement === opener, 'and closes the same way');
    await api.openConnect(); await settle();
    const x = find(dlg, (n) => n.tagName === 'BUTTON' && n.getAttribute('aria-label') === 'Close');
    fire(x, 'click');
    ok(!dlg.open, 'the × closes');
    // the backdrop: a press that starts on it closes; a drag out of a snippet (a click on the dialog) does not
    await api.openConnect(); await settle();
    const pre = find(dlg, (n) => n.tagName === 'PRE' && n.dataset.snippet !== undefined);
    fire(pre, 'pointerdown'); fire(dlg, 'click');
    ok(dlg.open, 'selecting a snippet past the box does not close');
    fire(dlg, 'pointerdown'); fire(dlg, 'click');
    ok(!dlg.open, 'a press on the backdrop closes');
    // every copy button says which snippet it copies
    await api.openConnect(); await settle();
    const label = find(dlg, (n) => n.dataset.copy !== undefined && visible(n)).getAttribute('aria-label');
    eq(label, 'Copy the Claude Code snippet (shell)', 'a copy button names its snippet');
  }

  // ── 10. the sign-in gate that takes per-user API keys ──────────────────────────────────────────────────────────
  const KEYS_ON = { status: 200, body: { email: 'a@b.test', expires_in: 300, renew_in: 120,
    api_keys: { days: [7, 30, 90], default_days: 30, owner_days: 30 } } };
  /** The gate's key routes, in memory: what the dialog asked, what it minted, and canned refusals. */
  function fakeGate({ create = null, list = null, revoke = null } = {}) {
    const now = Math.floor(Date.now() / 1000);
    const g = { keys: [], minted: [], ids: [] };
    g.answer = (url, opts) => {
      const method = (opts.method || 'GET').toUpperCase();
      if (url === '/auth/keys' && method === 'GET') {
        return list || { status: 200, body: { keys: g.keys.slice(), days: [7, 30, 90], default_days: 30, owner_days: 30 } };
      }
      if (url === '/auth/keys' && method === 'POST') {
        if (create) return create;
        const b = JSON.parse(opts.body);
        const key = `vgk_wxyz${'k'.repeat(39)}`;
        const id = `id${g.ids.length}`;
        const row = { id, name: b.name, hint: 'vgk_wxyz…', created_at: now, expires_at: now + b.days * 86400,
          last_used_at: g.ids.length ? now - 180 : null, state: 'active' };
        g.keys.unshift(row); g.minted.push(key); g.ids.push(id);
        return { status: 201, body: { key, ...row } };
      }
      if (url === '/auth/keys/revoke') {
        if (revoke) return revoke;
        const b = JSON.parse(opts.body);
        g.keys = g.keys.filter((k) => k.id !== b.id);
        return { status: 200, body: { ok: true, id: b.id } };
      }
      return { status: 404, body: {} };
    };
    return g;
  }
  const keySection = (dlg) => find(dlg, (n) => n.dataset.keys !== undefined);
  const keyRows = (dlg) => findAll(dlg, (n) => n.dataset.keyId !== undefined && visible(n));
  {
    const gate = fakeGate();
    const { api, dlg, asked, copied } = load({ door: KEYS_ON, keysApi: gate.answer });
    await api.openConnect(); await settle();
    const t = allText(dlg);
    ok(/sign-in gate/.test(t) && /API key on GET \/v1\/models, POST \/v1\/\*/.test(t), `the door says it takes keys: ${t.slice(0, 300)}`);
    const routes = findAll(dlg, (n) => n.tagName === 'BUTTON' && n.dataset.route !== undefined);
    eq(routes.map((r) => r.dataset.route).join(','), 'apikey', 'this address with an API key is the route');
    eq(routes[0].getAttribute('aria-pressed'), 'true', 'and it is chosen');
    const [claude] = snippets(dlg);
    ok(claude.includes('ANTHROPIC_BASE_URL=https://gw.example.test'), claude);
    ok(claude.split(' \\\n').includes("ANTHROPIC_AUTH_TOKEN='<your key>'"),
      'Claude Code sends the key as its bearer (captured 2.1.296: AUTH_TOKEN → Authorization: Bearer, no approval prompt)');
    ok(!claude.includes('ANTHROPIC_API_KEY') && !claude.includes('ANTHROPIC_CUSTOM_HEADERS'), claude);
    ok(claude.includes('CLAUDE_CODE_MAX_CONTEXT_TOKENS=196608'), 'the rest of fullclaudevllm\'s lines as on every route');
    fire(tabs(dlg)[1], 'click');
    const [curl, py] = snippets(dlg);
    ok(curl.startsWith('curl https://gw.example.test/v1/chat/completions'), curl);
    ok(curl.includes("  -H 'Authorization: Bearer <your key>'"), curl);
    ok(py.includes('base_url="https://gw.example.test/v1"') && py.includes('api_key="<your key>"'), py);
    fire(tabs(dlg)[2], 'click');
    const [msg] = snippets(dlg);
    ok(msg.includes("  -H 'x-api-key: <your key>'") && msg.includes("anthropic-version"), msg);
    fire(tabs(dlg)[0], 'click');

    // the keys: the section, the form with the gate's lifetimes, the default preselected
    const sec = keySection(dlg);
    ok(sec && visible(sec), 'the key section shows on the key route');
    ok(asked.some((a) => a.url === '/auth/keys' && !a.method), 'the person\'s keys are read');
    ok(/no keys yet/.test(textOf(sec)), `an empty list says so: ${textOf(sec)}`);
    ok(/30 d/.test(textOf(sec)), 'the owner\'s standing is stated as the gate states it');
    ok(/stops working after 30 d without a sign-in here/.test(textOf(sec)), `in plain words: ${textOf(sec)}`);
    const name = find(dlg, (n) => n.tagName === 'INPUT' && n.dataset.keyField === 'name');
    const days = find(dlg, (n) => n.tagName === 'SELECT' && n.dataset.keyField === 'days');
    ok(name && days, 'a name and a lifetime');
    eq(days.children.map((o) => o.value).join(','), '7,30,90', 'the lifetimes are the gate\'s, no "never"');
    eq(days.value, '30', 'its default preselected');
    const create = find(dlg, (n) => n.tagName === 'BUTTON' && n.dataset.keyCreate !== undefined);
    eq(create.textContent, 'Create API key', 'the create control says what it does');
    name.value = 'laptop'; fire(name, 'input');
    days.value = '90'; fire(days, 'change');
    fire(create, 'click'); await settle();
    const post = asked.find((a) => a.url === '/auth/keys' && a.method === 'POST');
    ok(post, 'created with a POST');
    eq(post.body, JSON.stringify({ name: 'laptop', days: 90 }), 'the name and the lifetime picked');
    eq(post.credentials, 'same-origin', 'with the session');
    eq((post.headers || {})['Content-Type'], 'application/json', 'as JSON');
    const shown = find(dlg, (n) => n.tagName === 'PRE' && n.dataset.newKey !== undefined);
    ok(shown && visible(shown), 'the new key is shown');
    eq(shown.textContent, gate.minted[0], 'exactly the key the gate minted');
    ok(/shown once/.test(textOf(sec)), 'and it says it is shown once');
    // the one-time warning (UX audit 2026-10-15): announced, unmissable, and it says what is left afterwards
    const once = find(dlg, (n) => n.dataset.keyOnce !== undefined);
    ok(once && visible(once) && once.getAttribute('role') === 'alert', 'the shown-once warning is an alert');
    ok(/never be shown again/.test(textOf(once)), `it says the key will never be shown again: ${textOf(once)}`);
    ok(textOf(once).includes('vgk_wxyz…'), 'and names what stays listed: the prefix + first 4');
    ok(/\.cn-key-once\s*\{[^}]*(--warn|font-weight)/.test(css), 'and it is set apart in the stylesheet');
    ok(/\.cn-key-new \.cn-pre\s*\{[^}]*user-select: all/.test(css), 'a click on the key selects the whole of it');
    eq(document.activeElement && document.activeElement.getAttribute('aria-label'), 'Copy the new API key',
      'the focus lands on its copy button');
    const copyKey = find(dlg, (n) => n.tagName === 'BUTTON' && n.dataset.copy !== undefined && visible(n)
      && n.getAttribute('aria-label') === 'Copy the new API key');
    fire(copyKey, 'click'); await settle();
    eq(copied[copied.length - 1], gate.minted[0], 'its copy button copies the key');
    eq(name.value, '', 'the name field is emptied for the next one');
    let rows = keyRows(dlg);
    eq(rows.length, 1, 'the list now holds the new key');
    const row = textOf(rows[0]);
    ok(row.includes('laptop') && row.includes('vgk_wxyz…'), `name and prefix + first 4: ${row}`);
    ok(/created \d{4}-\d\d-\d\d/.test(row) && /expires \d{4}-\d\d-\d\d/.test(row) && /used never/.test(row), row);
    ok(!row.includes(gate.minted[0]), 'a row never shows the key');
    // a second key: the list is newest first, last used said
    name.value = 'ci'; fire(name, 'keydown', { key: 'Enter' }); await settle();
    rows = keyRows(dlg);
    eq(rows.length, 2, 'Enter in the name creates too');
    ok(/used 3 min ago/.test(textOf(rows[0])), `last used, relative: ${textOf(rows[0])}`);
    // revoke: one press arms, the second revokes
    const revoke = find(rows[1], (n) => n.tagName === 'BUTTON' && n.dataset.revoke !== undefined);
    eq(revoke.getAttribute('aria-label'), 'Revoke API key laptop (vgk_wxyz…)', 'a revoke button names its key');
    fire(revoke, 'click'); await settle();
    ok(!asked.some((a) => a.url === '/auth/keys/revoke'), 'one press only arms it');
    ok(/confirm/.test(revoke.textContent), `and says so: ${revoke.textContent}`);
    ok(/confirm/i.test(revoke.getAttribute('aria-label')) && /laptop/.test(revoke.getAttribute('aria-label')),
      `an armed button says so to a screen reader too: ${revoke.getAttribute('aria-label')}`);
    const status = find(dlg, (n) => n.getAttribute('role') === 'status' && n.dataset.keyMsg !== undefined);
    ok(status && visible(status) && /laptop/.test(textOf(status)) && /again/.test(textOf(status)),
      `the status line says which key is about to go and what to press: ${textOf(status)}`);
    fire(revoke, 'click'); await settle();
    const rv = asked.find((a) => a.url === '/auth/keys/revoke');
    ok(rv && rv.method === 'POST' && rv.credentials === 'same-origin', 'revoked with a POST, with the session');
    eq(rv.body, JSON.stringify({ id: gate.ids[0] }), 'the key pressed, by its id');
    eq(keyRows(dlg).length, 1, 'the list reloads without it');
    ok(visible(status) && /revoked laptop/.test(textOf(status)) && /vgk_wxyz…/.test(textOf(status)),
      `and the status line confirms it by name and hint: ${textOf(status)}`);
    ok(!status.classList.contains('bad'), 'as a plain notice, not a fault');
    eq(find(dlg, (n) => n.tagName === 'UL' && n.classList.contains('cn-key-list')).getAttribute('aria-label'),
      'Your API keys', 'the list is named for a screen reader');
    // the focus trap reaches the new controls
    const focusables = dlg.querySelectorAll('button, input, select, a[href]').filter((n) => visible(n) && !n.disabled);
    ok(focusables.includes(days) && focusables.includes(name), "the lifetime and the name are in the Tab order");
    eq(document.activeElement, name, "after a revoke the focus stays in the dialog, on the name");
    // closing forgets the shown key
    api.closeConnect();
    await api.openConnect(); await settle();
    const again = find(dlg, (n) => n.tagName === 'PRE' && n.dataset.newKey !== undefined);
    ok(!again || !visible(again) || again.textContent === '', 'reopened, the key is gone for good');
    ok(!allText(dlg).includes(gate.minted[0]) && !allText(dlg).includes(gate.minted[1]), 'nowhere in the dialog');
  }
  {   // the gate's refusals are said in its words
    const gate = fakeGate({ create: { status: 400, body: { error: 'bad_name', detail: 'a name of 1..64 printable characters' } } });
    const { api, dlg } = load({ door: KEYS_ON, keysApi: gate.answer });
    await api.openConnect(); await settle();
    fire(find(dlg, (n) => n.dataset.keyCreate !== undefined), 'click'); await settle();
    ok(/bad_name: a name of 1\.\.64 printable characters/.test(textOf(keySection(dlg))), textOf(keySection(dlg)));
    ok(!find(dlg, (n) => n.tagName === 'PRE' && n.dataset.newKey !== undefined && visible(n) && n.textContent),
      'no key shown on a refusal');
    const status = find(dlg, (n) => n.getAttribute('role') === 'status' && n.dataset.keyMsg !== undefined);
    ok(status && status.classList.contains('bad'), 'a refusal is a fault');
    // the refused name: marked, focused, unmarked as soon as it is typed into (the create button was disabled
    // under the pointer, so without this the focus fell out of the dialog — UX audit 2026-10-15)
    const name = find(dlg, (n) => n.tagName === 'INPUT' && n.dataset.keyField === 'name');
    eq(name.getAttribute('aria-invalid'), 'true', 'the refused name is marked');
    eq(document.activeElement, name, 'and focused to be fixed');
    name.value = 'x'; fire(name, 'input');
    ok(name.getAttribute('aria-invalid') !== 'true', 'typing clears the mark');
  }
  {   // a create the gate refused still re-reads the list: what it refused for (20 live keys) is then in view
    const gate = fakeGate();
    const { api, dlg, asked } = load({ door: KEYS_ON, keysApi: gate.answer });
    await api.openConnect(); await settle();
    gate.keys.push({ id: 'elsewhere', name: 'other tab', hint: 'vgk_abcd…', created_at: 1, expires_at: 2 ** 31,
      last_used_at: null, state: 'active' });
    gate.answer = ((inner) => (url, opts) => ((opts.method || 'GET') === 'POST' && url === '/auth/keys'
      ? { status: 409, body: { error: 'too_many_keys', detail: '20 live keys; revoke one first' } } : inner(url, opts)))(gate.answer);
    const reads = asked.filter((a) => a.url === '/auth/keys' && !a.method).length;
    fire(find(dlg, (n) => n.dataset.keyCreate !== undefined), 'click'); await settle();
    eq(asked.filter((a) => a.url === '/auth/keys' && !a.method).length, reads + 1, 'the list is re-read after a refusal');
    ok(keyRows(dlg).some((r) => /other tab/.test(textOf(r))), 'and a key made elsewhere is now listed');
    eq(find(dlg, (n) => n.tagName === 'INPUT' && n.dataset.keyField === 'name').getAttribute('aria-invalid'), null,
      'a refusal that is not about the name does not mark it');
  }
  {
    const gate = fakeGate({ create: { status: 409, body: { error: 'too_many_keys', detail: '20 live keys; revoke one first' } },
      list: { status: 401, body: { error: 'sign_in_required' } } });
    const { api, dlg } = load({ door: KEYS_ON, keysApi: gate.answer });
    await api.openConnect(); await settle();
    ok(/\/auth\/keys: signed out/.test(textOf(keySection(dlg))), `a list the gate refused says why: ${textOf(keySection(dlg))}`);
    fire(find(dlg, (n) => n.dataset.keyCreate !== undefined), 'click'); await settle();
    ok(/too_many_keys: 20 live keys; revoke one first/.test(textOf(keySection(dlg))), textOf(keySection(dlg)));
  }
  {   // keys off, the open gate, the LB itself: no key route, no key section
    for (const door of [{ status: 200, body: { email: 'a@b.test', expires_in: 300, renew_in: 120 } },
      { status: 200, body: { email: 'anonymous', auth: 'false' } }, { status: 404, body: {} }]) {
      const gate = fakeGate();
      const { api, dlg, asked } = load({ door, keysApi: gate.answer });
      await api.openConnect(); await settle();
      const sec = keySection(dlg);
      ok(!sec || !visible(sec), `no key section on door ${JSON.stringify(door.body)}`);
      ok(!asked.some((a) => a.url.startsWith('/auth/keys')), 'and the gate is not asked for keys');
      ok(!findAll(dlg, (n) => n.dataset.route === 'apikey').length, 'and no key route');
    }
  }

  // ── 9. no secret, ever ─────────────────────────────────────────────────────────────────────────────────────────
  ok(!/Basic [A-Za-z0-9+/]{8,}={0,2}/.test(section), 'no Basic credential is written in the page');
  ok(!/Bearer [A-Za-z0-9._-]{8,}/.test(section), 'nor a bearer');

  console.log(`ok — ${checks} checks`);
})().catch((e) => { console.error(e); process.exit(1); });
