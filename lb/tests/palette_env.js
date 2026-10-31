/* THE STYLE TOKENS UNDER NODE: what a browser's getComputedStyle(:root) would answer, read from the real sheets.
 *
 * gateway.js and benchmarks.js compute colours through palette.js, which reads the active style's tokens from the
 * page. A node harness has no page, so this file parses gateway.css's `:root` block (the default style, "Vibe
 * coded") — and, for a named style, its sheet's `:root[data-style="<key>"]` block on top — resolving var() the way a
 * browser does for a custom property. The harnesses read the SHIPPED tokens: a test never carries its own copy of a
 * colour.
 *
 *   const { withPalette } = require('./palette_env');
 *   const src = withPalette(fs.readFileSync(.../gateway.js, 'utf8'));   // palette.js, then the script
 */
'use strict';
const fs = require('fs');
const path = require('path');

const STATIC = path.join(__dirname, '..', 'vllm_lb', 'static');
const PALETTE_SRC = fs.readFileSync(path.join(STATIC, 'palette.js'), 'utf8');
const stripComments = (css) => css.replace(/\/\*[\s\S]*?\*\//g, '');

/** The top-level rules of a sheet: [{prelude, body}] — a body may hold nested rules (CSS nesting, @media). */
function topRules(css) {
  const text = stripComments(css);
  const out = [];
  let depth = 0, start = 0, open = -1;
  for (let i = 0; i < text.length; i++) {
    if (text[i] === '{') { if (depth === 0) open = i; depth++; }
    else if (text[i] === '}') {
      depth--;
      if (depth === 0) { out.push({ prelude: text.slice(start, open).trim(), body: text.slice(open + 1, i) }); start = i + 1; }
    } else if (text[i] === ';' && depth === 0) start = i + 1;   // a top-level statement (@import …)
  }
  return out;
}

/** A rule body's own declarations — depth 0 only, nested rules skipped — as [name, value]. */
function ownDeclarations(body) {
  const out = [];
  let depth = 0, cur = '';
  for (const ch of body) {
    if (ch === '{') { depth++; cur = ''; continue; }
    if (ch === '}') { depth--; cur = ''; continue; }
    if (depth) continue;
    if (ch === ';') { out.push(cur); cur = ''; } else cur += ch;
  }
  out.push(cur);
  return out.map((d) => { const i = d.indexOf(':'); return i < 0 ? null : [d.slice(0, i).trim(), d.slice(i + 1).trim()]; })
    .filter((p) => p && p[0]);
}

/** Every custom property declared, at its top level, by the rules whose selector is exactly `selector`. */
function declarations(css, selector) {
  const out = {};
  for (const r of topRules(css)) {
    if (r.prelude !== selector) continue;
    for (const [name, value] of ownDeclarations(r.body)) if (name.startsWith('--')) out[name] = value;
  }
  return out;
}

const APPEARANCES = { light: null, dark: '@media (prefers-color-scheme: dark)' };

/** The vibe tokens, then a named style's on top (its sheet under styles/) — in the system `appearance` a browser
 *  reports: 'dark' adds what the style's nested `@media (prefers-color-scheme: dark)` block re-values ("Glass"). */
function tokensOf(style, appearance = 'light') {
  if (!(appearance in APPEARANCES)) throw new Error(`appearance "${appearance}": want light or dark`);
  const t = declarations(fs.readFileSync(path.join(STATIC, 'gateway.css'), 'utf8'), ':root');
  if (style && style !== 'vibe') {
    const css = fs.readFileSync(path.join(STATIC, 'styles', `${style}.css`), 'utf8');
    const scope = `:root[data-style="${style}"]`;
    Object.assign(t, declarations(css, scope));
    const media = APPEARANCES[appearance];
    if (media) {
      for (const r of topRules(css)) {
        if (r.prelude !== scope) continue;
        for (const inner of topRules(r.body)) {
          if (inner.prelude !== media) continue;
          for (const [name, value] of ownDeclarations(inner.body)) if (name.startsWith('--')) t[name] = value;
        }
      }
    }
  }
  return t;
}

/** var(--x) substituted, as a browser computes a custom property; an undefined reference is the empty string. */
function resolve(tokens, value, depth = 0) {
  if (depth > 20) throw new Error(`var() cycle at ${value}`);
  return value.replace(/var\((--[\w-]+)\)/g, (_, n) => (n in tokens ? resolve(tokens, tokens[n], depth + 1) : ''));
}

function computedStyleOf(style, appearance) {
  const tokens = tokensOf(style, appearance);
  return () => ({ getPropertyValue: (n) => (n in tokens ? resolve(tokens, tokens[n]) : '') });
}

/** palette.js wrapped to read `style`'s tokens, prepended to `src`: `palette` is in scope for the script. */
function withPalette(src, style = 'vibe', appearance = 'light') {
  const key = `__palette_cs_${String(style).replace(/\W/g, '_')}_${appearance}`;
  globalThis[key] = computedStyleOf(style, appearance);
  return `const palette = (function (getComputedStyle, document) {\n${PALETTE_SRC}\n;return palette;\n})(`
    + `globalThis.${key}, { documentElement: {}, addEventListener() {} });\n${src}`;
}

module.exports = { STATIC, PALETTE_SRC, topRules, ownDeclarations, declarations, tokensOf, computedStyleOf, withPalette };
