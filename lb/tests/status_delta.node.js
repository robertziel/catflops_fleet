/* The page's applier of /lb/stream's patches (static/status_delta.js), held to the SAME wire contract the LB's
 * writer is (tests/fixtures/status_delta_cases.json; tests/test_status_delta.py checks the Python side): every
 * case's patch applied to its `a` is exactly its `b`, and `a` is left as it was.
 *
 *   node tests/status_delta.node.js   (tests/test_status_delta_node.py runs this when node exists)
 */
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const statusDelta = require(path.join(__dirname, '..', 'vllm_lb', 'static', 'status_delta.js'));
const { cases } = JSON.parse(fs.readFileSync(path.join(__dirname, 'fixtures', 'status_delta_cases.json'), 'utf8'));

let checks = 0;
for (const c of cases) {
  const before = JSON.stringify(c.a);
  const got = statusDelta.apply(c.patch, c.a);
  assert.deepStrictEqual(got, c.b, c.name);
  assert.strictEqual(JSON.stringify(c.a), before, `${c.name}: the document it was given is unchanged`);
  checks++;
}
assert.throws(() => statusDelta.apply({ '?': 1 }, {}), /not a status patch node/);
checks++;
console.log(`ok — ${checks} checks`);
