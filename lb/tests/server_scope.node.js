/* `?server=<name>` ON THE PERFORMANCE AND SESSIONS PAGES (vllm_lb/static/server-scope.js, ONE helper both load).
 * Every perf window and boot line names its server, and /lb/perf and /lb/sessions take `server=`: the page names the
 * server and asks for its minutes alone — whether one server runs or several. Unscoped, nothing changes.
 *
 *   docker run --rm -v "$PWD/lb":/lb:ro -w /lb node:22-slim node tests/server_scope.node.js
 */
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const STATIC = path.join(__dirname, '..', 'vllm_lb', 'static');
const src = fs.readFileSync(path.join(STATIC, 'server-scope.js'), 'utf8');
let checks = 0;

function page(search) {
  const els = { 'server-only': { hidden: true, innerHTML: '' } };
  const document = { title: 'x', getElementById: (id) => els[id] || null };
  const api = new Function('document', 'location', `${src}\n;return { serverScope, serverQuery };`)(
    document, { search, pathname: '/benchmarks' });
  return { els, document, api };
}

// unscoped: nothing named, nothing appended
let p = page('');
assert.equal(p.api.serverScope('Performance'), null); checks++;
assert.equal(p.api.serverQuery(), ''); checks++;
assert.equal(p.els['server-only'].hidden, true); checks++;

// scoped: the title and the line name it, every query asks for its minutes alone
p = page('?server=gui');
assert.equal(p.api.serverScope('Performance'), 'gui'); checks++;
assert.equal(p.document.title, 'Performance · gui'); checks++;
assert.equal(p.api.serverQuery(), '&server=gui'); checks++;
assert.match(p.els['server-only'].innerHTML, /<b>gui<\/b> · this server's minutes · <a href="\/benchmarks">every server<\/a>/); checks++;

// a name is escaped wherever it is written
p = page('?server=a%26b%3Cx');
p.api.serverScope('Sessions');
assert.equal(p.api.serverQuery(), '&server=a%26b%3Cx'); checks++;
assert.match(p.els['server-only'].innerHTML, /<b>a&amp;b&lt;x<\/b>/); checks++;

// both pages fetch with it
for (const f of ['benchmarks.js', 'sessions.js']) {
  assert.match(fs.readFileSync(path.join(STATIC, f), 'utf8'), /\$\{serverQuery\(\)\}`/, f); checks++;
}

console.log(`server_scope: ${checks} checks ok`);
