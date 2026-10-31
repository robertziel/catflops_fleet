/* THE PALETTE (vllm_lb/static/palette.js) under node: the colours scripts compute come from the active style's tokens,
 * and the default style's tokens reproduce the panel's colours as they were before styles existed.
 *
 * THE REFERENCE below is the arithmetic gateway.js carried until 2026-10-30 (scaleColor, rateColor, queueColor), kept
 * here VERBATIM as the oracle: "Vibe coded" must stay pixel-identical, so the token-driven scales must land on the
 * same colours at every share. And a style that leaves a scale undefined is a loud throw, never a stand-in colour.
 *
 *   node tests/palette.node.js        (tests/test_styles.py runs this when node exists)
 */
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');
const { withPalette, PALETTE_SRC } = require('./palette_env');

let checks = 0;
const load = (style) => new Function(`${withPalette('', style)}\n;return palette;`)();
const P = load('vibe');

// ── the oracle: gateway.js before styles ─────────────────────────────────────────────────────────────────────────
function oldScale(t) {
  const x = Math.min(Math.max(Number(t) || 0, 0), 1);
  if (x < 0.12) {
    const u = x / 0.12;
    const h = 360 * (0.52 - 0.16 * u), sat = 0.40 + 0.45 * u, v = 1.0 - 0.05 * u;
    const l = v * (1 - sat / 2);
    const sl = l === 0 || l === 1 ? 0 : (v - l) / Math.min(l, 1 - l);
    return [h, 100 * sl, 100 * l, 0.80 + 0.20 * u];
  }
  const hue = x < 0.60 ? 0.36 - 0.21 * ((x - 0.12) / 0.48) : 0.15 - 0.15 * ((x - 0.60) / 0.40);
  return [hue * 360, 89, 54.6, 1];
}
function oldRate(t) {
  const x = Math.min(Math.max(Number(t) || 0, 0), 1);
  let h, sat, v;
  if (x < 0.4) { const u = x / 0.4; h = 0.60; sat = 0.05 + 0.25 * u; v = 0.50 + 0.12 * u; }
  else if (x < 0.8) { const u = (x - 0.4) / 0.4; h = 0.60 - 0.02 * u; sat = 0.30 + 0.45 * u; v = 0.62 + 0.23 * u; }
  else { const u = (x - 0.8) / 0.2; h = 0.58 - 0.04 * u; sat = 0.75 + 0.20 * u; v = 0.85 + 0.15 * u; }
  const l = v * (1 - sat / 2);
  const sl = l === 0 || l === 1 ? 0 : (v - l) / Math.min(l, 1 - l);
  return [360 * h, 100 * sl, 100 * l, 1];
}
function oldQueue(t) {
  const hue = t < 1 ? 0.36 * t : (t < 3 ? 0.36 * (1 - (t - 1) / 2) : 0);
  return [hue * 360, 89, 54.6, 1];
}
const parse = (css) => {
  const m = /^hsla?\(([\d.]+),([\d.]+)%,([\d.]+)%(?:,([\d.]+))?\)$/.exec(css);
  assert.ok(m, `not an hsl colour: ${css}`);
  return [Number(m[1]), Number(m[2]), Number(m[3]), m[4] == null ? 1 : Number(m[4])];
};
// Within the 0.1 rounding every figure is printed to (and the old upper segment's literal 89 % / 54.6 %): the same
// colour to the eye and, at 8 bits a channel, to the pixel.
const near = (got, want, what) => {
  const tol = [0.11, 0.06, 0.06, 0.006];
  got.forEach((g, i) => assert.ok(Math.abs(g - want[i]) <= tol[i], `${what}: ${got} vs ${want}`));
  checks++;
};

// 1. THE VIBE TOKENS ARE THE OLD SCALES, at every thousandth of the domain.
for (let i = 0; i <= 1000; i++) {
  const t = i / 1000;
  near(parse(P.css(P.hsl('--scale-load', t))), oldScale(t), `load ${t}`);
  near(parse(P.css(P.hsl('--scale-rate', t))), oldRate(t), `rate ${t}`);
  near(parse(P.css(P.hsl('--scale-queue', 4 * t))), oldQueue(4 * t), `queue ${4 * t}`);
}
// …and out of range: clamped, as before (a share over 1, a negative, a NaN, an infinite queue).
near(parse(P.css(P.hsl('--scale-load', 1.7))), oldScale(1.7), 'load over 1');
near(parse(P.css(P.hsl('--scale-load', -3))), oldScale(-3), 'load negative');
near(parse(P.css(P.hsl('--scale-load', NaN))), oldScale(NaN), 'load NaN');
near(parse(P.css(P.hsl('--scale-queue', Infinity))), oldQueue(Infinity), 'queue infinite');
// The idle end is translucent (hsla), the rest opaque (hsl) — as gateway.js drew them.
assert.ok(P.css(P.hsl('--scale-load', 0)).startsWith('hsla(')); checks++;
assert.ok(P.css(P.hsl('--scale-load', 0.5)).startsWith('hsl(')); checks++;

// 2. THE IDLE CORE'S GREY AND THE CARD SURFACE come from tokens too: --core-idle resolves through var() to the
//    neutral triplet, and --panel-2's luminance is the 0.0148 benchmarks.js hard-coded.
assert.strictEqual(P.rgb('--core-idle'), '139,148,158'); checks++;
assert.ok(Math.abs(P.luminance('--panel-2') - 0.0148) < 0.0005, String(P.luminance('--panel-2'))); checks++;
assert.strictEqual(P.space(), 'hsb'); checks++;

// 3. NO FALLBACK: a style missing a scale, or writing one wrong, throws with the token's name.
const stub = (tokens, onListen) => new Function('getComputedStyle', 'document', `${PALETTE_SRC}\n;return palette;`)(
  () => ({ getPropertyValue: (n) => tokens()[n] || '' }),
  { documentElement: {}, addEventListener: (type, fn) => { if (onListen) onListen(type, fn); } });
{
  const bare = stub(() => ({}));
  assert.throws(() => bare.hsl('--scale-load', 0.5), /--scale-load is not defined by the active style/); checks++;
  const bad = stub(() => ({ '--scale-space': 'hsb', '--a': '0 #000', '--b': '0 .5 .5 .5, 1 #fff', '--x': '1, 2',
    '--y': '0 #000, 0 #fff', '--z': 'red, blue' }));
  assert.throws(() => bad.hsl('--a', 0.5), /at least two stops/); checks++;
  assert.throws(() => bad.hsl('--b', 0.5), /is not a colour palette.js reads/); checks++;
  assert.throws(() => bad.rgb('--x'), /not an RGB triplet/); checks++;
  assert.throws(() => bad.hsl('--y', 0.5), /must ascend/); checks++;
  assert.throws(() => bad.hsl('--z', 0.5), /want stops of "<t> <colour>"/); checks++;
  const space = stub(() => ({ '--scale-space': 'lab', '--s': '0 #000, 1 #fff' }));
  assert.throws(() => space.hsl('--s', 0.5), /--scale-space is "lab": want hsb or oklch/); checks++;
}

// 4. A STYLE CHANGE IS SEEN: the cache holds until `fleet:style`, then the new style's tokens are read.
{
  let tokens = { '--scale-space': 'hsb', '--scale-load': '0 hwb(0 0% 0%), 1 hwb(0 0% 0%)' };
  let handler = null;
  const live = stub(() => tokens, (type, fn) => { if (type === 'fleet:style') handler = fn; });
  assert.strictEqual(live.css(live.hsl('--scale-load', 0.5)), 'hsl(0.0,100.0%,50.0%)');
  tokens = { '--scale-space': 'hsb', '--scale-load': '0 hwb(180 0% 0%), 1 hwb(180 0% 0%)' };
  assert.strictEqual(live.css(live.hsl('--scale-load', 0.5)), 'hsl(0.0,100.0%,50.0%)', 'cached until the style changes');
  assert.ok(handler, 'palette.js listens for fleet:style');
  handler();
  assert.strictEqual(live.css(live.hsl('--scale-load', 0.5)), 'hsl(180.0,100.0%,50.0%)', 'read again after the switch');
  checks += 3;
}

// 5. EVERY SHIPPED STYLE defines what palette.js reads, or inherits the vibe value: each scale parses and every
//    computed colour is a valid one.
const index = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'vllm_lb', 'static', 'styles', 'index.json'), 'utf8'));
for (const s of index.styles) {
  const S = load(s.key);
  ['--scale-load', '--scale-rate', '--scale-queue'].forEach((n) => {
    assert.ok(S.stops(n).length >= 2, `${s.key} ${n}`);
    for (let i = 0; i <= 20; i++) parse(S.css(S.hsl(n, (n === '--scale-queue' ? 3 : 1) * i / 20)));
    checks++;
  });
  assert.ok(/^\d+,\d+,\d+$/.test(S.rgb('--core-idle')), s.key); checks++;
  assert.ok(S.luminance('--panel-2') >= 0, s.key); checks++;
}

// 6. "CLEAN" mixes in oklch — CSS color-mix(in oklch, …), the reference values pinned here computed independently (a
//    python oklch mix, shorter hue arc): at a stop
//    the scale IS the stop's colour; between two stops it is color-mix(in oklch, …). Its rate reaches the target at 80 %.
{
  const O = load('clean');
  assert.strictEqual(O.space(), 'oklch'); checks++;
  const hex = (c) => {   // {h, s, l} → #rrggbb, as a browser rounds it
    const a = c.s * Math.min(c.l, 1 - c.l);
    const f = (n) => { const k = (n + c.h / 30) % 12; return c.l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1)); };
    return '#' + [f(0), f(8), f(4)].map((v) => Math.round(v * 255).toString(16).padStart(2, '0')).join('');
  };
  const close = (got, want, what) => {
    const d = [1, 3, 5].map((i) => Math.abs(parseInt(got.slice(i, i + 2), 16) - parseInt(want.slice(i, i + 2), 16)));
    assert.ok(Math.max(...d) <= 1, `${what}: ${got} vs ${want}`); checks++;
  };
  close(hex(O.hsl('--scale-load', 0.12)), '#15803d', 'load at 12 % is --load-ok');
  close(hex(O.hsl('--scale-load', 1)), '#dc2626', 'load at 100 % is --load-bad');
  close(hex(O.hsl('--scale-load', 0.36)), '#846f00', 'load half way ok → warn, mixed in oklch');
  close(hex(O.hsl('--scale-load', 0.06)), '#187b87', 'load half way idle → ok, mixed in oklch');
  close(hex(O.hsl('--scale-rate', 0.2)), '#4a6db9', 'rate half way rest → mid, mixed in oklch');
  close(hex(O.hsl('--scale-rate', 0.8)), '#1d4ed8', 'rate at 80 % is the target');
  close(hex(O.hsl('--scale-rate', 0.9)), '#1d4ed8', '…and stays there');
  assert.strictEqual(O.rgb('--core-idle'), '152,162,179'); checks++;
}

// THE AA CHECK a style's colours pass (sections 7 and 8): every computed figure colour — every load and rate share,
// every queue depth — and every text token reads at WCAG AA (≥ 4.5:1) on a card (--panel) and on a band inside it
// (--panel-2); a chip's hue on its 12 % tint too. One definition, every style that pins it.
const lin = (c) => (c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4));
const rgbOfHsl = (c) => {
  const a = c.s * Math.min(c.l, 1 - c.l);
  const f = (n) => { const k = (n + c.h / 30) % 12; return c.l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1)); };
  return [f(0), f(8), f(4)];
};
const hexOfHsl = (c) => '#' + rgbOfHsl(c).map((v) => Math.round(v * 255).toString(16).padStart(2, '0')).join('');
const lumOf = (rgb) => 0.2126 * lin(rgb[0]) + 0.7152 * lin(rgb[1]) + 0.0722 * lin(rgb[2]);
const ratio = (a, b) => (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05);
const hexRgb = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16) / 255);
const tripletRgb = (t) => t.split(',').map((x) => Number(x) / 255);
function readsAtAA(G, label) {
  assert.strictEqual(G.space(), 'oklch'); checks++;
  const grounds = { '--panel': G.luminance('--panel'), '--panel-2': G.luminance('--panel-2') };
  const aa = (lum, what) => {
    for (const [g, lg] of Object.entries(grounds)) {
      const r = ratio(lum, lg);
      assert.ok(r >= 4.5, `${label}: ${what} on ${g} is ${r.toFixed(2)}:1`); checks++;
    }
  };
  for (let i = 0; i <= 200; i++) {
    const t = i / 200;
    aa(lumOf(rgbOfHsl(G.hsl('--scale-load', t))), `load ${t}`);
    aa(lumOf(rgbOfHsl(G.hsl('--scale-rate', t))), `rate ${t}`);
    aa(lumOf(rgbOfHsl(G.hsl('--scale-queue', 3 * t))), `queue ${3 * t}`);
  }
  ['--ink', '--ink-dim', '--ink-faint', '--accent', '--ok', '--warn', '--bad', '--park']
    .forEach((n) => aa(G.luminance(n), n));
  const panel = hexRgb(G.token('--panel'));
  for (const n of ['--tag-vllm', '--tag-sglang', '--tag-tensorfold', '--tag-drafter', '--ok-rgb', '--warn-rgb', '--bad-rgb', '--accent-rgb']) {
    const ink = tripletRgb(G.rgb(n));
    const tint = ink.map((c, k) => 0.12 * c + 0.88 * panel[k]);
    const r = ratio(lumOf(ink), lumOf(tint));
    assert.ok(r >= 4.5, `${label}: ${n} on its 12 % tint is ${r.toFixed(2)}:1`); checks++;
  }
}

// 7. "GLASS" follows the system appearance: a light and a dark token set in one sheet, both mixed in oklch, each AA.
for (const appearance of ['light', 'dark']) {
  readsAtAA(new Function(`${withPalette('', 'glass', appearance)}\n;return palette;`)(), `glass ${appearance}`);
}

// 8. "TERMINAL" is one dark appearance in green phosphor, AA throughout — its dimmest green included. Its scales,
//    pinned (reference values from an independent python oklch mix, shorter hue arc): load dim green → bright green
//    at 12 % → amber at 60 % → red at 100 %; rate dim → bright at 80 % → a white-hot green at the target.
{
  const T = load('terminal');
  readsAtAA(T, 'terminal');
  const close = (got, want, what) => {
    const d = [1, 3, 5].map((i) => Math.abs(parseInt(got.slice(i, i + 2), 16) - parseInt(want.slice(i, i + 2), 16)));
    assert.ok(Math.max(...d) <= 1, `terminal ${what}: ${got} vs ${want}`); checks++;
  };
  const pins = {
    '--scale-load': [[0, '#1fa344'], [0.06, '#28d055'], [0.12, '#33ff66'], [0.36, '#cbe500'], [0.6, '#ffcc33'],
      [0.8, '#ff9320'], [1, '#ff5c5c']],
    '--scale-rate': [[0, '#1fa344'], [0.2, '#24bb4d'], [0.6, '#2ee95f'], [0.8, '#33ff66'], [0.9, '#81ffa0'],
      [1, '#b8ffcb']],
  };
  for (const [scale, at] of Object.entries(pins)) at.forEach(([t, want]) => close(hexOfHsl(T.hsl(scale, t)), want, `${scale} ${t}`));
  assert.strictEqual(T.token('--glow'), '1'); checks++;
  assert.strictEqual(T.rgb('--core-idle'), '40,150,72'); checks++;
}

console.log(`ok — ${checks} palette checks`);
