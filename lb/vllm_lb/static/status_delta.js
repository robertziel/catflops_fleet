/* THE STREAM'S PATCH, APPLIED — /lb/stream sends the status whole once, then after every control tick
 * only what changed (vllm_lb/status_delta.py writes the patch; this is its `apply`, line for line). The format, the
 * wire contract both sides' tests read (tests/fixtures/status_delta_cases.json):
 *
 *   {"=": v}                        the value is now v
 *   {"o": {k: node}, "x": [k, …]}   an object: these keys changed, these are gone
 *   {"i": {"<index>": node}}        a list of the same length: these items changed
 *   {"r": [drop, tail]}             a ring: the old list without its first `drop` items, then `tail`
 *   {"u": [drop, head]}             a newest-first list: `head`, then the old list without its last `drop` items
 *   null                            nothing changed
 *
 * A NEW DOCUMENT, never the old one changed in place: the page keeps the status it drew (`lastStatus`) and a style
 * switch draws it again, so a patch must not reach into it. A node this file does not know THROWS — the stream then
 * drops the connection and the page is sent the document whole again; it never draws a guess.
 */
'use strict';

const statusDelta = (() => {
  function apply(patch, doc) {
    if (patch === null || patch === undefined) return doc;
    if (Object.prototype.hasOwnProperty.call(patch, '=')) return patch['='];
    if (patch.o) {
      const out = Object.assign({}, doc);
      for (const [k, node] of Object.entries(patch.o)) out[k] = apply(node, doc == null ? undefined : doc[k]);
      for (const k of patch.x || []) delete out[k];
      return out;
    }
    if (patch.i) {
      const out = doc.slice();
      for (const [idx, node] of Object.entries(patch.i)) out[Number(idx)] = apply(node, doc[Number(idx)]);
      return out;
    }
    if (patch.r) {
      const [drop, tail] = patch.r;
      return doc.slice(drop).concat(tail);
    }
    if (patch.u) {
      const [drop, head] = patch.u;
      return head.concat(doc.slice(0, doc.length - drop));
    }
    throw new Error(`not a status patch node: ${Object.keys(patch).sort().join(', ')}`);
  }
  return { apply };
})();

if (typeof module !== 'undefined') module.exports = statusDelta;
