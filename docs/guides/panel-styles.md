# Panel styles

The gateway panel (`/`, `/benchmarks`, `/sessions`) has several looks. A style only re-values CSS tokens; layout,
data and behaviour are the same in every style.

## The styles

| key | label | look | fonts | light / dark |
|---|---|---|---|---|
| `vibe` | Vibe coded | the base: gateway.css's own `:root`, no extra sheet | system UI + system mono | dark |
| `clean` | Clean | light, quiet, oklch scales | Inter, Fragment Mono | light |
| `mono` | Mono | monospace, terminal-like | JetBrains Mono | dark |
| `glass` | Glass | translucent, rounded, system UI face | system (Inter where none) | follows `prefers-color-scheme` |
| `terminal` | Terminal | green-phosphor CRT terminal: glow, faint scanlines, blinking cursor (solid under reduced motion) | JetBrains Mono | dark only |

The registry is `lb/vllm_lb/static/styles/index.json`, in menu order. The panel's menu, the fleet loader
(`vllm_lb/panel_styles.py`) and the setup wizard (`render.py` source `panel_styles`) all read this one file.

## Who picks

| layer | where it lives | wins over |
|---|---|---|
| this browser's choice | `localStorage["lb.style"]` = `{key, href}` | the deployment's default |
| the deployment's default | `lb/fleet.yml` `panel.style` | nothing (required) |

- **Style menu** (header, "Style · …"): one item per style. The deployment's default is tagged `default`.
- Picking a style stores it in this browser only. The server stores nothing per browser.
- Picking the item tagged `default` clears the stored choice, so the browser follows the deployment again.
- A browser that never picked anything follows the deployment's default, including after the operator changes it.
- Before first paint: the LB writes the default onto `<html data-style-default="…" data-style-default-href="…">`
  of every page it serves. `styles.js` runs in `<head>` and links the right sheet render-blocking, so the page
  never shows another style first and makes no extra request.
- Errors are shown, never swapped in silently. Each shows as a notice with an action:

| case | notice | action |
|---|---|---|
| stored key not in the registry | `style "x" not found — using <default> (this deployment's default)` | reset |
| stored value unreadable | `the stored style cannot be read (…)` | reset |
| sheet fails to load | `style "X" did not load (…) — still <current>` | retry |
| registry fails to load | `styles not loaded: …` | retry |
| page has no stamped default (LB with no fleet file) | `this gateway names no default style … — showing Vibe coded` | none |
| browser keeps no storage | `style not kept in this browser: … — it lasts this visit` | none |

## The fleet key

```yaml
# lb/fleet.yml
panel:
  style: clean        # vibe | clean | mono | glass | terminal — a key of lb/vllm_lb/static/styles/index.json
```

- **Required.** If it is missing, the LB does not start:
  `fleet.yml names no panel.style — one of vibe, clean, mono, glass, terminal: … (vibe keeps the original look). Add: panel:\n  style: glass`.
- **Unknown key:** `fleet.yml: panel.style 'neon' is no style of the panel — one of vibe, clean, mono, glass, terminal`.
- `panel` takes only `style`. Any other key is refused.
- It is a fleet value, so it is never an env var too (one place per value).
- To change it, edit `lb/fleet.yml` and restart the LB. Browsers with no stored choice pick up the new default on
  their next page load.
- Setup wizard: question `panel.style` (section `panel`) writes it.

## Adding a style

| # | step | file |
|---|---|---|
| 1 | Pick a key `[a-z][a-z0-9-]{0,31}` and name it for how it **looks**, never for where the look came from (no site, company, product or typeface name: `BANNED_NAMES` in `lb/tests/test_styles.py` scans the whole static tree) | — |
| 2 | Write one sheet, `lb/vllm_lb/static/styles/<key>.css`. Every top-level rule is `:root[data-style="<key>"]` (CSS nesting inside it), `@font-face`, or `@keyframes <key>-…`. Any `@media` holds only scoped rules | `styles/<key>.css` |
| 3 | Re-value the tokens (inventory below). Write rules only where a token cannot reach | same |
| 4 | Fonts: vendor them as woff2 in `styles/fonts/<Family>-<weight>.woff2`, add `<Family>-OFL.txt` beside them, and add a line to the root `NOTICE` (§4, fonts). Reference them as `url("fonts/…")`. The build hashes every font and sheet, and fails if one ships unstamped | `styles/fonts/`, `NOTICE` |
| 5 | Light and dark: to follow the system, declare the light tokens plus `color-scheme: light` in the scope block, and nest **one** `@media (prefers-color-scheme: dark)` block that re-values every surface, ink and meaning token with `color-scheme: dark` (as `glass.css` does). `styles.js` tells palette.js when the appearance changes | same |
| 6 | Register it: append `{ "key": "<key>", "label": "<Label>", "file": "/styles/<key>.css" }` to `index.json`. The first entry stays `vibe` with `file: null` | `styles/index.json` |
| 7 | Pin it in tests (below), then rebuild the image (`docker build -t vllm-lb:dev lb`). The setup wizard and the fleet loader accept the new key with no further change | `lb/tests/` |

### Token inventory (gateway.css `:root`)

| group | tokens |
|---|---|
| surfaces / ink | `--bg --panel --panel-2 --surface-3 --line --line-strong --ink --ink-dim --ink-faint --ink-faint-rgb --muted --neutral-rgb --on-fill --shade-rgb --scrim --scrim-sheet --track --track-bar --track-progress` |
| meaning | `--ok --ok-rgb --ok-hi --warn --warn-rgb --bad --bad-rgb --accent --accent-rgb --park --park-rgb --step-current --step-current-rgb --hold-pill-ok --hold-pill-stop --hold-pill-swarm` |
| engine tags / cores | `--tag-vllm --tag-sglang --tag-tensorfold --tag-drafter --core-mul --core-mem --core-other --core-idle --glyph-blade` |
| scale stops | `--load-idle --load-ok --load-warn --load-bad --rate-rest --rate-mid --rate-high --rate-target` |
| scales (palette.js) | `--scale-load --scale-rate --scale-queue` (`"<t> <colour>, …"`, ascending t) · `--scale-space` (`hsb` \| `oklch`) |
| effects | `--glow` (`0` turns it off) · `--area-opacity --diode-size` · `--shadow-menu --shadow-pop --shadow-sheet` · `--rule` |
| brand mark | `--brand-mark --brand-mark-pkg --brand-mark-line --brand-mark-idle --brand-mark-a --brand-mark-b --wm-cat --wm-flops --wm-fleet` |
| type | `--sans --sans-ui --serif --mono --body-size --small --fs-62 --fs-80 --fs-100 --fs-svg-tiny --fs-svg-figure` |
| shape / space | `--radius-card --radius-menu --radius-band --radius-box --radius-ctl --radius-ctl-sm --radius-badge --radius-chip --radius-3 --radius-pill --radius-tally --gutter --gap` |

`*-rgb` tokens are bare `r, g, b` triplets, used as `rgba(var(--ok-rgb), .15)`. A token a style leaves out keeps
the `vibe` value. A scale that is missing or malformed makes palette.js **throw** and name the token. It never
draws in a stand-in colour.

### Tests that gate a style

| test | holds |
|---|---|
| `tests/test_styles.py` | registry shape; every sheet listed; scoping; fonts vendored + licence; served types; build stamps; banned names; a style that follows light/dark re-values every token |
| `tests/test_style_tokens.py` | no colour literal outside a `:root` token block; type and radius are tokens; `vibe` defines every token palette.js reads |
| `tests/palette.node.js` | `vibe` scales = the pre-styles colours; every shipped style's scales parse; `clean`'s oklch reference colours pinned; `glass` and `terminal`: WCAG AA (≥ 4.5:1) text on their surfaces and tints, in each appearance |
| `tests/styles.node.js` | the switch: precedence, before-first-paint apply, notices |
| `tests/test_panel_style.py` | `panel.style` required and validated against the registry; the stamp on every page |

Add the new style's pins to `palette.node.js` (its scale reference values and an AA contrast set like `glass`'s) and to the
registry list asserted in `test_the_registry_starts_with_vibe_coded_and_lists_every_sheet`.
