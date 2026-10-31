# Panel layout templates

A layout template places the gateway panel's elements in rows and columns. The **Layout** menu in the panel header
lists them; each browser keeps its pick and any edits in its own `localStorage` (`lb.layout`). The server stores none
of it, and no env or fleet key names a layout.

| Template | What it is |
|---|---|
| `default` | queued · cluster · tabs on top; the server card in a side column; throughput, requests, saturation beside it |
| `compact` | tabs first; the server card beside throughput + saturation; requests and the cluster share a row |
| `wall` | a monitoring wall: boxes on top, throughput + saturation beside requests, no controls; tabs at the foot |
| `column` | one column, a phone's order |

The fastest custom layout needs no file: **Layout → Edit**, drag tiles, **Done**, then **Export** — the JSON it
downloads is a template body. The rest of this guide is for shipping one in the repo.

## The format

`lb/vllm_lb/static/layouts/<name>.json`:

```json
{
  "schema": 1,
  "name": "ops",
  "label": "Ops",
  "what": "requests first and wide; throughput + saturation in a narrow column; the cluster below",
  "rows": [
    { "cols": [
      { "width": "1", "items": ["queued", "tabs"] }
    ] },
    { "cols": [
      { "width": "2", "items": ["requests"] },
      { "width": "1", "items": ["engine", "saturation"] }
    ] },
    { "cols": [
      { "width": "1", "items": [{ "id": "server", "size": "half" }, { "id": "cluster", "size": "half" }] }
    ] }
  ],
  "hidden": []
}
```

| Key | Rule |
|---|---|
| `schema` | `1` |
| `name` | the file's stem; lowercase |
| `label` | what the menu shows |
| `what` | one line: what the arrangement is for |
| `rows[].cols` | 1–3 columns a row |
| `cols[].width` | `side` (the server card's fixed-ish column) or a share: `1`, `2`, `3` |
| `cols[].items` | element ids, top to bottom; an id alone is full width, `{ "id", "size": "half" }` is half of its column |
| `hidden` | ids not drawn |

**The elements** (the registry is the markup: `data-ly` hosts in `lb/vllm_lb/static/index.html`):

| id | Element |
|---|---|
| `queued` | Queued leases |
| `cluster` | Cluster (the boxes: CPU, GPU cards, memory, flight recorder) |
| `tabs` | Server tabs |
| `server` | Server card (Start / Stop, settings, swarm) |
| `engine` | Throughput (generated, prefill, prompt, queue, KV, bandwidth, compute) |
| `requests` | Requests |
| `saturation` | Saturation |

Every id appears **exactly once** — in a column or in `hidden`. Below a desk's width every row collapses to one
column in item order, so order the items for a phone too.

Register it: append the name to `layouts/index.json` `templates`. The **first** entry is what a fresh browser opens
in; leave `default` first unless that is the change you mean.

## Vibe coding your own template

Hand an agent this section with your intent ("requests on the left, two-thirds wide; hide saturation"). Rules it
must follow:

1. **Only data.** Write `layouts/<name>.json` and one line in `layouts/index.json`. Nothing else changes: no CSS, no
   JS, no HTML, no server code. A width or size the format lacks is not added for one template.
2. **Only registry ids.** Use the seven ids above, each exactly once (placed or `hidden`). An unknown id draws an
   "unknown element" notice; a missing one is appended at the end — both fail the tests.
3. **Format limits.** At most 3 columns a row; widths `side | 1 | 2 | 3`; sizes `full | half`. `side` is meant for the
   `server` card.
4. **Name it by purpose**, not by person or deployment (`ops`, `wall`, `triage`; not a host or team name). `what` says
   what it is for in one line.
5. **Prove it** before calling it done:
   ```bash
   cd lb && docker build -t vllm-lb:dev . \
     && docker run --rm -v "$PWD":/app -w /app vllm-lb:dev python -m pytest -q tests/test_layouts.py
   docker run --rm -v "$PWD":/repo:ro -w /repo/lb node:22-slim node tests/layout.node.js
   ```
   then open the panel, pick it in **Layout**, and look at it at a desk width and at a phone width (a headless
   Chromium screenshot is enough). No notice line may appear under the header.
6. **Show it.** Add a screenshot to the README gallery only when the template ships in the repo; a template for
   one deployment stays an exported JSON that browser imports (**Layout → Import**).

What the tests pin (`lb/tests/test_layouts.py`, `lb/tests/layout.node.js`): `index.json` and the files on disk
agree, `default` is first and reproduces the original panel, every shipped template is complete and names only
registry ids, templates are served as JSON, and the server stores no layout.
