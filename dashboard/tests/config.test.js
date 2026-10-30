// The dashboard's server list comes from VLLM_SERVERS and from nowhere else.
// Run (no npm install needed: config.js has no dependencies):
//   docker run --rm -v "$PWD/dashboard":/d:ro -w /d node:22-slim node --test 'tests/*.test.js'
const test = require('node:test');
const assert = require('node:assert');
const { parseServers, requireEnv } = require('../config');

test('the servers are the comma-separated VLLM_SERVERS URLs, trimmed and in order', () => {
  assert.deepStrictEqual(
    parseServers('http://192.0.2.2:8000, http://198.51.100.2:8000 ,'),
    ['http://192.0.2.2:8000', 'http://198.51.100.2:8000']);
});

test('unset or blank refuses, and names the variable: there is no localhost stand-in', () => {
  for (const value of [undefined, '', '   ', ' , ']) {
    assert.throws(() => parseServers(value), /VLLM_SERVERS/, `value ${JSON.stringify(value)}`);
  }
});

test('the LB shown and the model labelled refuse unset or blank, naming the variable: no stand-in', () => {
  assert.strictEqual(requireEnv('LB_URL', ' http://192.0.2.2:8010 ', 'hint'), 'http://192.0.2.2:8010');
  for (const value of [undefined, '', '   ']) {
    assert.throws(() => requireEnv('CHAT_MODEL', value, 'the model name'), /CHAT_MODEL is not set: the model name/);
  }
});
