/* THE PALETTE AS SCRIPTS SEE IT (operator, 2026-10-30: "Add switch between styles"). A colour that is COMPUTED — a
 * figure on the load scale, a rate on the rate scale, a queue against its concurrency, an idle core's grey, a
 * figure's contrast against its card — is computed from the ACTIVE STYLE's tokens (gateway.css `:root`, re-valued by
 * styles/<key>.css under :root[data-style="<key>"]), read through getComputedStyle and cached until the style
 * changes (styles.js fires `fleet:style` on document). No script holds a colour: gateway.js and benchmarks.js used
 * to carry two hand-kept copies of the same HSB arithmetic and its numbers; both now ask here.
 *
 * A SCALE TOKEN is a list of stops "<t> <colour>" (comma-separated, ascending t; gateway.css --scale-load etc.),
 * clamped at its ends and interpolated in the style's --scale-space:
 *   hsb    hue, saturation, brightness and alpha each linear between the two stops — the Fleet app's scaleColor /
 *          rateColor arithmetic, exactly as gateway.js computed it before styles (a dark style's stops are hwb(),
 *          HSB written without rounding);
 *   oklch  lightness, chroma and alpha linear, hue the shorter way round — what CSS color-mix(in oklch, …) does, the
 *          designer's choice for a light style, where a mix through HSB would muddy.
 * The answer is always a number triple ({h, s, l, a}, CSS hsl/hsla via `css`): benchmarks.js measures a figure's
 * contrast from it, which a color-mix() string would not allow. A missing or malformed token THROWS, naming it: a
 * style that forgets a scale draws nothing in a stand-in colour (AGENTS.md: no fallbacks).
 *
 * Loaded before gateway.js (index.html) and benchmarks.js (benchmarks.html). Node harnesses load it through
 * tests/palette_env.js, which hands it the shipped tokens parsed from the stylesheets.
 */
'use strict';

const palette = (() => {
  const cache = new Map();
  const fail = (msg) => { throw new Error(msg); };
  const clamp01 = (x) => Math.min(Math.max(x, 0), 1);

  /** A token's value in the active style, trimmed; a throw when the style does not define it. */
  function token(name) {
    if (!cache.has(name)) {
      const v = String(getComputedStyle(document.documentElement).getPropertyValue(name) || '').trim();
      if (!v) fail(`style token ${name} is not defined by the active style`);
      cache.set(name, v);
    }
    return cache.get(name);
  }

  /** Split at the commas that are not inside parentheses. */
  function topLevel(text) {
    const out = [];
    let depth = 0, cur = '';
    for (const ch of text) {
      if (ch === '(') depth++;
      if (ch === ')') depth--;
      if (ch === ',' && depth === 0) { out.push(cur); cur = ''; } else cur += ch;
    }
    out.push(cur);
    return out.map((s) => s.trim());
  }

  // ── colour parsing: hwb(), hsl()/hsla(), rgb()/rgba(), #hex → sRGB 0…1 or HSB ─────────────────────────────────────
  const num = (s, pct) => {
    const n = parseFloat(s);
    if (!Number.isFinite(n)) return NaN;
    return /%$/.test(s) ? n / 100 : (pct ? n / 100 : n);
  };
  /** The arguments of a CSS colour function, comma or space syntax, with the alpha after `/` or as the fourth. */
  function args(body) {
    const [main, slash] = body.split('/');
    const parts = main.includes(',') ? main.split(',') : main.trim().split(/\s+/);
    const p = parts.map((s) => s.trim()).filter(Boolean);
    const alpha = slash != null ? slash.trim() : p[3];
    return { p: p.slice(0, 3), a: alpha == null ? 1 : num(alpha, false) };
  }
  /** {h 0…1, s, v, a}: a colour token read as HSB. */
  function hsbOf(text, name) {
    const m = /^(hwb|hsla?|rgba?)\((.*)\)$/i.exec(text.trim());
    let out = null;
    if (m) {
      const fn = m[1].toLowerCase(), { p, a } = args(m[2]);
      if (fn === 'hwb') {
        let w = num(p[1], true), b = num(p[2], true);
        if (w + b > 1) { const k = w + b; w /= k; b /= k; }
        const v = 1 - b;
        out = { h: parseFloat(p[0]) / 360, s: v === 0 ? 0 : 1 - w / v, v, a };
      } else if (fn.startsWith('hsl')) {
        const s = num(p[1], true), l = num(p[2], true);
        const v = l + s * Math.min(l, 1 - l);
        out = { h: parseFloat(p[0]) / 360, s: v === 0 ? 0 : 2 * (1 - l / v), v, a };
      } else {
        out = { ...hsbFromRgb(p.map((c) => (/%$/.test(c) ? num(c) : parseFloat(c) / 255))), a };
      }
    } else {
      const hex = /^#([0-9a-f]{3,4}|[0-9a-f]{6}|[0-9a-f]{8})$/i.exec(text.trim());
      if (hex) {
        const h = hex[1].length <= 4 ? hex[1].replace(/./g, '$&$&') : hex[1];
        const c = [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16) / 255);
        out = { ...hsbFromRgb(c), a: h.length === 8 ? parseInt(h.slice(6, 8), 16) / 255 : 1 };
      }
    }
    if (!out || [out.h, out.s, out.v, out.a].some((x) => !Number.isFinite(x))) {
      fail(`style token ${name}: "${text}" is not a colour palette.js reads (hwb, hsl, rgb or #hex)`);
    }
    out.h = ((out.h % 1) + 1) % 1;
    return out;
  }
  function hsbFromRgb([r, g, b]) {
    const mx = Math.max(r, g, b), mn = Math.min(r, g, b), d = mx - mn;
    let h = 0;
    if (d) h = mx === r ? ((g - b) / d) % 6 : mx === g ? (b - r) / d + 2 : (r - g) / d + 4;
    return { h: ((h / 6) + 1) % 1, s: mx === 0 ? 0 : d / mx, v: mx };
  }
  function rgbFromHsb({ h, s, v }) {
    const f = (n) => { const k = (n + h * 6) % 6; return v - v * s * Math.max(0, Math.min(k, 4 - k, 1)); };
    return [f(5), f(3), f(1)];
  }
  const toLin = (c) => (c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4));
  const toGam = (c) => (c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055);
  function oklchFromRgb(rgb) {
    const [r, g, b] = rgb.map(toLin);
    const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
    const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
    const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
    const L = 0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s;
    const A = 1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s;
    const B = 0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s;
    return { L, C: Math.hypot(A, B), H: (Math.atan2(B, A) * 180) / Math.PI };
  }
  function rgbFromOklch({ L, C, H }) {
    const A = C * Math.cos((H * Math.PI) / 180), B = C * Math.sin((H * Math.PI) / 180);
    const l = (L + 0.3963377774 * A + 0.2158037573 * B) ** 3;
    const m = (L - 0.1055613458 * A - 0.0638541728 * B) ** 3;
    const s = (L - 0.0894841775 * A - 1.2914855480 * B) ** 3;
    return [4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
      -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
      -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s].map((c) => clamp01(toGam(c)));
  }
  /** HSB → the {h degrees, s, l, a} that `css` prints. */
  function hslFromHsb({ h, s, v, a }) {
    const l = v * (1 - s / 2);
    return { h: 360 * h, s: l === 0 || l === 1 ? 0 : (v - l) / Math.min(l, 1 - l), l, a };
  }

  /** A scale token's stops, ascending by t: [{t, text, hsb}]. */
  function stops(name) {
    const key = `${name}::stops`;
    if (!cache.has(key)) {
      const list = topLevel(token(name)).map((part) => {
        const m = /^(-?[\d.]+)\s+(.+)$/.exec(part);
        if (!m) fail(`style token ${name} is not a scale: want stops of "<t> <colour>", got "${part}"`);
        return { t: Number(m[1]), text: m[2], hsb: hsbOf(m[2], name) };
      });
      if (list.length < 2) fail(`style token ${name} is not a scale: want at least two stops, got "${token(name)}"`);
      list.forEach((p, i) => { if (i && p.t <= list[i - 1].t) fail(`style token ${name}: stops must ascend in t`); });
      cache.set(key, list);
    }
    return cache.get(key);
  }

  /** The active style's interpolation space for every scale. */
  function space() {
    const s = token('--scale-space');
    if (s !== 'hsb' && s !== 'oklch') fail(`style token --scale-space is "${s}": want hsb or oklch`);
    return s;
  }

  /** The colour at `t` on scale `name`: {h (degrees), s, l (0…1), a}. */
  function hsl(name, t) {
    const st = stops(name);
    const x = Math.min(Math.max(Number(t) || 0, st[0].t), st[st.length - 1].t);
    let i = 0;
    while (i < st.length - 2 && x >= st[i + 1].t) i++;
    const p = st[i], q = st[i + 1];
    const u = (x - p.t) / (q.t - p.t);
    const lerp = (a, b) => a + (b - a) * u;
    if (space() === 'hsb') {
      const P = p.hsb, Q = q.hsb;
      return hslFromHsb({ h: lerp(P.h, Q.h), s: lerp(P.s, Q.s), v: lerp(P.v, Q.v), a: lerp(P.a, Q.a) });
    }
    const P = oklchFromRgb(rgbFromHsb(p.hsb)), Q = oklchFromRgb(rgbFromHsb(q.hsb));
    // An achromatic end has no hue of its own: it takes the other end's, as color-mix() does (CSS "powerless" hue).
    const eps = 1e-4;
    const hP = P.C < eps ? Q.H : P.H, hQ = Q.C < eps ? P.H : Q.H;
    let dh = hQ - hP;
    if (dh > 180) dh -= 360; else if (dh < -180) dh += 360;
    const rgb = rgbFromOklch({ L: lerp(P.L, Q.L), C: lerp(P.C, Q.C), H: hP + dh * u });
    return hslFromHsb({ ...hsbFromRgb(rgb), a: lerp(p.hsb.a, q.hsb.a) });
  }

  /** That colour as CSS: hsl() when opaque, hsla() when not. */
  function css(c) {
    const body = `${c.h.toFixed(1)},${(100 * c.s).toFixed(1)}%,${(100 * c.l).toFixed(1)}%`;
    return c.a < 1 ? `hsla(${body},${c.a.toFixed(2)})` : `hsl(${body})`;
  }

  /** An RGB-triplet token ("139, 148, 158") as "139,148,158", for rgba(). */
  function rgb(name) {
    const v = token(name).replace(/\s+/g, '');
    if (!/^\d{1,3},\d{1,3},\d{1,3}$/.test(v)) fail(`style token ${name} is not an RGB triplet: "${token(name)}"`);
    return v;
  }

  /** WCAG relative luminance of a colour token (any form `hsbOf` reads). */
  function luminance(name) {
    const lin = rgbFromHsb(hsbOf(token(name), name)).map((c) => (c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4)));
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2];
  }

  function reset() { cache.clear(); }
  document.addEventListener('fleet:style', reset);
  return Object.freeze({ token, stops, space, hsl, css, rgb, luminance, reset });
})();
