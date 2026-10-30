// The dashboard's compose file names no operator box and requires no operator secret to parse.
// Run with the config test: node --test tests/config.test.js tests/compose.test.js
const test = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const path = require('node:path');

const compose = fs.readFileSync(path.join(__dirname, '..', 'docker-compose.yml'), 'utf8');

test('VLLM_SERVERS comes from the environment, with no literal box address', () => {
  assert.match(compose, /^\s*-\s*VLLM_SERVERS=\$\{VLLM_SERVERS:-\}\s*(#.*)?$/m);
  assert.doesNotMatch(compose, /10\.10\./);
});

test('the LB shown and the model labelled are required, with no literal address or model name as a default', () => {
  assert.match(compose, /^\s*-\s*LB_URL=\$\{LB_URL:\?[^}]+\}\s*$/m);
  assert.match(compose, /^\s*-\s*CHAT_MODEL=\$\{CHAT_MODEL:\?[^}]+\}\s*$/m);
  assert.doesNotMatch(compose, /LB_URL:-|CHAT_MODEL:-/);
});

test('no interpolation requires FRP_TOKEN: a missing token must not fail the dashboard project', () => {
  assert.doesNotMatch(compose, /\$\{FRP_TOKEN:\?/);
  assert.match(compose, /FRP_TOKEN=\$\{FRP_TOKEN:-\}/);
});
