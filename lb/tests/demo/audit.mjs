/* THE PANEL UX AUDIT (Playwright, run by audit.sh inside the pinned Playwright image, in
 * demo-lb's network namespace).
 *
 * For every screen — desktop 1440×1000 and mobile 390×844, in the dark default style (vibe) and one light style
 * (clean) — it opens the panel, shoots the whole page (ux-page-<screen>.png), finds each change, screenshots it as
 * ux-<n>-<desktop|mobile>-<style>.png and asserts:
 *   console     no console error and no uncaught page error (each listed with its URL when it is a failed load)
 *   overflow    no horizontal page overflow (scrollWidth ≤ the viewport)
 *   clipped     no text cut by its box (scrollWidth > clientWidth on an element that hides overflow), and no box that
 *               scrolls sideways (what is past its edge is off screen)
 *   overlap     no two text leaves of the feature intersect
 *   numbers     the figures shown equal the harness's INPUTS (sender.py's inputs.json, seed.py's expected.json) — never
 *               the gateway's own status, which would check the gateway against itself
 *   figures     no label is prose: a text leaf of 7+ words with no digit fails
 *   tap         at 390 px every control of the feature is ≥ 44 × 44 px (a finger's height)
 *   dialogs     a dialog closes on × and on Escape
 *   tokens      nothing shows `undefined`, `NaN` or `null`, and no count of 5+ digits is printed ungrouped
 *
 *
 * WHERE THE PANEL MOVED THINGS, and what is asserted there:
 *   band        the findings band is ONE chip per box and state (firing boxes, then pending, then the delivery chip,
 *               last, in the warn tone); each finding's figure on the chip, its threshold in the chip's title; a phone
 *               shows 2 rows and a "+n" (aria-expanded) that shows the rest.
 *   strip       RoCE, DISKS and MODELS are in each box's flight-recorder strip (`.machine-slot .rec`), opened per box;
 *               a port line shows its non-zero / rising counters, the rest behind `details.hx-more` (open when the port
 *               is down or rising).
 *   activity    Activity sits under the band, collapsed to 3 rows, "<n> ›" shows the rest; alert sends are marks on the
 *               row they sent ("✗ webhook"); the health line no longer repeats the band's firing / pending counts.
 *   strip line  the CLOSED strip line: "RoCE <up>/<total> up · <n> rising" and "disk <max %>" equal the inputs, in the
 *               fault colour only on a fault; no box-health section is left inside a machine card.
 *   chip tap    a band chip's tap scrolls to the box's card and flashes it; a RoCE chip opens the box's strip.
 *
 * Usage: node audit.mjs <out dir>   (env AUDIT_URL, default http://127.0.0.1:8010; AUDIT_DATA, default /data)
 * Writes <out dir>/audit.json and prints one table per change; exits 1 when any row fails.
 */
import { chromium } from '/pw/node_modules/playwright/index.mjs';
import fs from 'node:fs';

const OUT = process.argv[2] || '/out';
const URL = process.env.AUDIT_URL || 'http://127.0.0.1:8010';
const DATA = process.env.AUDIT_DATA || '/data';
/** The fleet file the demo LB runs (audit.sh mounts lb/tests/demo at /demo): the policy the page is judged against. */
const FLEET = process.env.AUDIT_FLEET || '/demo/fleet.demo.yml';
const SCREENS = [['desktop', { width: 1440, height: 1000 }], ['mobile', { width: 390, height: 844 }]];
const STYLES = ['vibe', 'clean'];
const TAP_MIN = 44;
/** Ports the sender makes RISE every send (sender.py: box-a mlx5_0/1's out_of_buffer +3, local_ack_timeout_err +1,
 *  rx_discards_phy +2 per send) or that a firing RoCE finding names (box-b mlx5_1/1 link_downed, every 10th send):
 *  one per RDMA box. */
const RISING = { 'box-a': 1, 'box-b': 1 };

const inputs = () => JSON.parse(fs.readFileSync(`${DATA}/inputs.json`, 'utf8'));
const expected = JSON.parse(fs.readFileSync(`${DATA}/expected.json`, 'utf8'));
const en = (n) => Math.round(n).toLocaleString('en-US');
const fig = (v) => { const a = Math.abs(v); return a >= 100 ? Math.round(v).toLocaleString('en-US') : a >= 10 ? String(Math.round(v * 10) / 10) : String(Math.round(v * 100) / 100); };
/** The last `n` sends' value of a counter the sender derives from its send number k. */
const recent = (f, n = 4) => { const k = inputs().k; return Array.from({ length: n }, (_, i) => f(k - i)).filter((v) => v >= 0); };

/** THE PANEL REPAINTS UNDER THE AUDIT (three races — the disks clip's null box, the "ungrouped 98015" read off a replaced
 *  section, the delivery chip's colour probe on a detached chip): the band and the cards are redrawn on every status patch, so an
 *  element found by one step can be gone by the next. Every measuring step on an element goes through these: the
 *  locator is re-resolved on each try, the function runs on the element as it is NOW and reports whether it (each of
 *  them, for `measureAll`) was still in the page when read; a throw or a detached read is tried again, a few times,
 *  then the step FAILS LOUDLY naming itself and the last reason — never a result read off a node the page dropped. */
const TRIES = 20;
const RETRY_MS = 150;
function inPage(fn, all) {
  // Built here and serialised by Playwright (Runtime.callFunctionOn: not the page's code, so no CSP applies).
  return new Function('e', 'a', `const r = (${fn.toString()})(e, a.arg);
    return { r, connected: ${all ? 'e.every((x) => x.isConnected)' : 'e.isConnected'} };`);
}
async function measureWith(loc, what, fn, arg, all) {
  const run = inPage(fn, all);
  let why = '';
  for (let i = 0; i < TRIES; i++) {
    try {
      const got = await (all ? loc.evaluateAll(run, { arg }) : loc.evaluate(run, { arg }, { timeout: 5000 }));
      if (got.connected) return got.r;
      why = 'the element left the page while it was read';
    } catch (e) {
      why = String(e.message || e).split('\n')[0];
    }
    await loc.page().waitForTimeout(RETRY_MS);
  }
  throw new Error(`${what}: no stable read after ${TRIES} tries (${loc}): ${why}`);
}
/** `fn(element, arg)` on the one element `loc` names, read while it is in the page. */
const measure = (loc, what, fn, arg) => measureWith(loc, what, fn, arg, false);
/** `fn(elements, arg)` on every element `loc` names, read while every one is in the page. */
const measureAll = (loc, what, fn, arg) => measureWith(loc, what, fn, arg, true);

/** The generic UX checks over one root element (in the page). */
async function ux(page, root, mobile) {
  return measure(root, `ux checks on ${root}`, (el, { mobile, TAP_MIN }) => {
    const out = { clipped: [], overlap: [], prose: [], tap: [], tokens: [] };
    // Rendered for a reader: not inside a closed <details>, not display:none / visibility:hidden / content-visibility.
    const vis = (e) => { const r = e.getBoundingClientRect(); return r.width > 0 && r.height > 0 && e.checkVisibility({ checkVisibilityCSS: true, contentVisibilityAuto: true }); };
    /** A rect cut to what its scrolling / clipping ancestors show: a row scrolled out of a list is behind the list's
     *  edge, not over the line under it. */
    const shown = (e, b) => {
      let [l, t, r, btm] = [b.left, b.top, b.right, b.bottom];
      for (let a = e.parentElement; a; a = a.parentElement) {
        const cs = getComputedStyle(a);
        if (cs.overflowX !== 'visible' || cs.overflowY !== 'visible') {
          const c = a.getBoundingClientRect();
          l = Math.max(l, c.left); t = Math.max(t, c.top); r = Math.min(r, c.right); btm = Math.min(btm, c.bottom);
        }
      }
      return r - l > 0 && btm - t > 0 ? { left: l, top: t, right: r, bottom: btm } : null;
    };
    const all = [el, ...el.querySelectorAll('*')].filter(vis);
    for (const e of all) {
      const cs = getComputedStyle(e);
      const hides = ['hidden', 'clip'].includes(cs.overflowX) || cs.textOverflow === 'ellipsis';
      if (hides && e.scrollWidth > e.clientWidth + 1 && e.textContent.trim() && !e.matches('svg, svg *')) {
        out.clipped.push(`${e.className || e.tagName}: "${e.textContent.trim().slice(0, 60)}" ${e.scrollWidth}>${e.clientWidth}`);
      }
    }
    const leaves = all.filter((e) => !e.matches('svg, svg *') && [...e.childNodes].some((n) => n.nodeType === 3 && n.textContent.trim()));
    const boxes = leaves.slice(0, 500).map((e) => {
      // the TEXT's own boxes (a range over the element's text nodes), not the element's: an inline-block's padding is not text
      const rects = [];
      for (const n of e.childNodes) {
        if (n.nodeType !== 3 || !n.textContent.trim()) continue;
        const r = document.createRange(); r.selectNodeContents(n);
        for (const b of r.getClientRects()) { const v = b.width > 0 && b.height > 0 && shown(e, b); if (v) rects.push(v); }
      }
      return { e, rects };
    });
    for (let i = 0; i < boxes.length; i++) {
      for (let j = i + 1; j < boxes.length; j++) {
        const a = boxes[i]; const b = boxes[j];
        if (a.e.contains(b.e) || b.e.contains(a.e)) continue;
        const hit = a.rects.some((x) => b.rects.some((y) => {
          const w = Math.min(x.right, y.right) - Math.max(x.left, y.left);
          const h = Math.min(x.bottom, y.bottom) - Math.max(x.top, y.top);
          return w > 1.5 && h > 1.5;
        }));
        if (hit) out.overlap.push(`"${a.e.textContent.trim().slice(0, 30)}" × "${b.e.textContent.trim().slice(0, 30)}"`);
      }
    }
    for (const e of leaves) {
      const t = [...e.childNodes].filter((n) => n.nodeType === 3).map((n) => n.textContent).join(' ').trim();
      if (t.split(/\s+/).length >= 7 && !/\d/.test(t)) out.prose.push(t.slice(0, 80));
    }
    // A box that scrolls sideways hides what is past its edge: at 390 px a figure must be on screen, not off it.
    for (const e of all) {
      const cs = getComputedStyle(e);
      if (['auto', 'scroll'].includes(cs.overflowX) && e.scrollWidth > e.clientWidth + 1) {
        out.clipped.push(`${e.className || e.tagName} scrolls sideways ${e.scrollWidth}>${e.clientWidth}`);
      }
    }
    if (mobile) {
      for (const c of all.filter((e) => e.matches('button, a[href], summary, [role=button], input, select'))) {
        const r = c.getBoundingClientRect();
        if (r.height < TAP_MIN || r.width < TAP_MIN) out.tap.push(`${c.textContent.trim().slice(0, 20) || c.getAttribute('aria-label')} ${Math.round(r.width)}×${Math.round(r.height)}`);
      }
    }
    const text = el.innerText || '';
    // A count of 5+ digits printed ungrouped ("68999 MiB") — the panel groups every count ("1,708,150").
    // Each match is reported with ~30 characters of its context ("ungrouped 98015" alone did not say
    // where it was), and whether the element was still in the page when read — a detached node's innerText is its
    // textContent, which has no line breaks between blocks.
    const around = (i, n) => JSON.stringify(text.slice(Math.max(0, i - 30), i + n + 30));
    for (const m of text.matchAll(/(?<![\w.,:\/-])\d{5,}(?![\w.,:\/-])/g)) {
      out.tokens.push(`ungrouped ${m[0]} in ${around(m.index, m[0].length)}${el.isConnected ? '' : ' (element detached)'}`);
    }
    for (const w of ['undefined', 'NaN', 'null']) if (new RegExp(`\\b${w}\\b`).test(text)) out.tokens.push(w);
    return out;
  }, { mobile, TAP_MIN });
}

const results = {};   // change → [{screen, pass, issues}]
function record(n, screen, issues) {
  (results[n] ||= []).push({ screen, pass: !issues.length, issues });
}
/** The panel's access.js and the connect dialog ask /auth/session; on the LB's own port (no gate in front, as here) the
 *  LB answers it itself (connect.py LB_SESSION, 200 `door: "lb"`), so no load fails. Should a 404 come back
 *  (a regression), it is counted on its own row (`page`), never dropped. Each such 404 brings one "Failed to load
 *  resource … 404" console line with no URL: exactly that many are its. */
function splitGate(errors) {
  const fails = []; const gated = [];
  let owed = errors.filter((e) => e === 'HTTP 404 /auth/session').length;
  for (const e of errors) {
    if (e === 'HTTP 404 /auth/session') gated.push(e);
    else if (owed > 0 && e.startsWith('Failed to load resource') && e.includes('404')) { gated.push(e); owed--; }
    else fails.push(e);
  }
  return { fails, gated };
}
function uxIssues(u) {
  return [...u.clipped.map((x) => `clipped ${x}`), ...u.overlap.map((x) => `overlap ${x}`),
    ...u.prose.map((x) => `prose "${x}"`), ...u.tap.map((x) => `tap ${x}`), ...u.tokens.map((x) => `shows ${x}`)];
}
const card = (page, box) => page.locator(`article.card.machine[data-box="${box}"]`);
/** The box's flight-recorder slot (recorder.js): the strip button and, open, the box health sections. */
const slot = (page, box) => page.locator(`.machine-slot[data-box="${box}"] .rec`);
const strip = (page, box) => slot(page, box).locator('.rec-strip');
const sec = (page, box, head) => slot(page, box).locator('.hx-sec', { has: page.locator('.hx-h', { hasText: head }) }).first();
/** Open the box's strip the way a reader does (a click on the strip), unless it is open already. */
async function openStrip(page, box) {
  if ((await strip(page, box).getAttribute('aria-expanded')) !== 'true') await strip(page, box).click();
  await slot(page, box).locator('.hx-sec').first().waitFor({ timeout: 10000 });
}
/** Two CSS colours compared as the browser resolves them (a token may be written as hex, the computed one is rgb). */
async function sameColour(loc, token) {
  return measure(loc, `colour probe ${token}`, (e, t) => {
    const probe = document.createElement('span');
    probe.style.color = `var(${t})`;
    e.parentElement.appendChild(probe);
    const want = getComputedStyle(probe).color;
    probe.remove();
    return getComputedStyle(e).color === want;
  }, token);
}
/** The element's box, shot from the full page: the panel repaints its cards on every status patch, so an element
 *  handle can be replaced between "find" and "shoot" — a clip of its box is not. */
/** An element's box. A repaint can land between the lookup and the box (null while the element is replaced): look
 *  again, a few times — then fail loudly, naming the element; never a skipped shot. */
async function boxOf(loc, what) {
  let b = null;
  for (let i = 0; i < TRIES && !b; i++) { b = await loc.boundingBox().catch(() => null); if (!b) await loc.page().waitForTimeout(RETRY_MS); }
  if (!b) throw new Error(`${what}: the element never had a box (${loc})`);
  return b;
}

async function shoot(loc, file) {
  const b = await boxOf(loc, file);
  const sy = await loc.page().evaluate(() => scrollY);
  await loc.page().screenshot({ path: `${OUT}/${file}`, fullPage: true,
    clip: { x: Math.max(0, b.x - 4), y: Math.max(0, b.y + sy - 4), width: b.width + 8, height: b.height + 8 } });
}

async function dialogCloses(page) {
  const issues = [];
  const d = page.locator('#obs-dialog');
  await d.locator('.obs-x').click();
  if (await measure(d, 'dialog open after ×', (x) => x.open)) issues.push('× did not close the dialog');
  return issues;
}
async function escapeCloses(page, open) {
  await open();
  await page.keyboard.press('Escape');
  return (await measure(page.locator('#obs-dialog'), 'dialog open after Escape', (x) => x.open))
    ? ['Escape did not close the dialog'] : [];
}

const browser = await chromium.launch();
for (const [scr, viewport] of SCREENS) {
  for (const style of STYLES) {
    const screen = `${scr}-${style}`;
    const mobile = scr === 'mobile';
    const ctx = await browser.newContext({ viewport, deviceScaleFactor: 2 });
    // styles.js's stored choice is JSON {key, href} (href null for the base): a bare key is refused with a notice and
    // the page opens in the default — the style under audit would not be the one shot.
    await ctx.addInitScript((s) => { try { localStorage.setItem('lb.style', JSON.stringify({ key: s, href: s === 'vibe' ? null : `/styles/${s}.css` })); } catch (e) { /* recorded by the page */ } }, style);
    const page = await ctx.newPage();
    const errors = [];
    page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
    page.on('pageerror', (e) => errors.push(`pageerror ${e.message}`));
    page.on('response', (r) => { if (r.status() >= 400) errors.push(`HTTP ${r.status()} ${new globalThis.URL(r.url()).pathname}`); });
    await page.goto(`${URL}/`);
    // up.sh's states take ~70 s to settle: XID fires 30 s + hold after the first send, box-b's link_downed 20 s + hold,
    // box-c's engine stops at 45 s. The band is one chip per box and state: the box's firing chip names
    // each of its findings' figures.
    await page.waitForFunction(() => {
      const t = [...document.querySelectorAll('#lb-health .hl-chip')].map((c) => [c.className, c.textContent]);
      const firing = (box, what) => t.some(([cls, x]) => cls.includes('hl-firing') && x.startsWith(`${box} · `) && x.includes(what));
      return firing('box-a', 'XID +') && firing('box-b', 'RoCE ERR +') && t.some(([, x]) => x.includes('SERVER not live'))
        && document.querySelector('#lb-health .hl-failed');
    }, null, { timeout: 150000 });
    await page.waitForSelector('.machine-slot[data-box="box-a"] .rec-strip', { timeout: 30000 });
    await page.waitForSelector('#events-list .ev-row', { timeout: 30000 });
    await page.waitForTimeout(800);
    await page.screenshot({ path: `${OUT}/ux-page-${screen}.png`, fullPage: true });

    // ── findings band (one chip per box), and the kernel figures on the machine cards ──
    {
      const band = page.locator('#lb-health');
      const issues = uxIssues(await ux(page, band, mobile));
      // Every chip, shown or folded behind a phone's "+n": its class, text, title and box.
      const chipsAll = await measureAll(band.locator('.hl-chip'), 'the band\'s chips', (cs) => cs.map((c) => ({
        cls: c.className, text: c.textContent, title: c.title || '', box: c.dataset.box || null, hidden: c.hidden })));
      const chips = chipsAll.map((c) => c.text);
      const inp = inputs().boxes;
      const c = inp['box-c'].deep.gpu[0].temp_headroom_c;
      const ramB = (inp['box-b'].ram_total_mb - inp['box-b'].ram_used_mb) / 1024;
      // health.py: the floor is the fleet's share (fleet.demo.yml low_ram_fraction 0.10) of box-b's OWN RAM, rounded as
      // the gateway rounds it; GPU POWER's ceiling is the GB10 type's idle draw (8.9 W) + the demo's 2 W margin.
      const ramFloor = Math.round(0.10 * inp['box-b'].ram_total_mb / 1024 * 10) / 10;
      const wB = inp['box-b'].deep.gpu[0].power_w;
      // [state, box, the chip's figure, the title's threshold line] — the figure on the chip, the threshold in its title
      const want = [['firing', 'box-c', `GPU TEMP ${c} °C to limit`, `GPU TEMP ${c} °C to limit ≤ 5 °C`],
        ['firing', 'box-b', 'RoCE ERR +', 'FIRING RoCE ERR +'],
        ['firing', 'box-b', `RAM AVAIL ${fig(ramB)} GiB`, `RAM AVAIL ${fig(ramB)} GiB < ${fig(ramFloor)} GiB`],
        ['firing', 'box-b', `GPU POWER ${fig(wB)} W`, `GPU POWER ${fig(wB)} W ≤ 10.9 W`],
        ['firing', 'box-a', 'RoCE ERR +', 'FIRING RoCE ERR +'], ['firing', 'box-a', 'XID +', 'FIRING XID +'],
        ['pending', 'box-c', 'SERVER not live', 'PENDING SERVER not live · ']];
      for (const [st, box, figure, line] of want) {
        const ch = chipsAll.find((x) => x.cls.includes(`hl-${st}`) && x.box === box);
        if (!ch) { issues.push(`no ${st} chip for ${box} in [${chips.join(' | ')}]`); continue; }
        if (!ch.text.includes(figure)) issues.push(`${st} ${box} chip "${ch.text}" lacks "${figure}"`);
        if (!ch.title.includes(line)) issues.push(`${st} ${box} chip title lacks "${line}": "${ch.title}"`);
        if (/[≤<]/.test(ch.text)) issues.push(`${st} ${box} chip "${ch.text}" carries a threshold (its title's)`);
      }
      // ONE chip per box and state; every firing chip before every pending one; the delivery chip last
      const keys = chipsAll.filter((x) => x.box).map((x) => `${x.cls.includes('hl-firing') ? 'firing' : 'pending'}|${x.box}`);
      if (new Set(keys).size !== keys.length) issues.push(`a box has two chips of one state: ${keys.join(', ')}`);
      const rank = chipsAll.map((x) => (x.cls.includes('hl-firing') ? 0 : x.cls.includes('hl-pending') ? 1 : 2));
      if (rank.some((r, i) => i && r < rank[i - 1])) issues.push(`chip order ${rank.join(',')}: firing, then pending, then delivery`);
      if (!chipsAll.length || !chipsAll[chipsAll.length - 1].cls.includes('hl-failed')) issues.push('the delivery chip is not last');
      if (chips.some((t) => t.includes('GPU TEMP') && t.startsWith('box-d'))) issues.push('box-d GPU TEMP chip: its headroom is absent, so no data');
      // A PHONE: at most 2 rows, the rest behind "+n" (aria-expanded), and all of them once it is opened
      const more = band.locator('.hl-more');
      const rowsOf = () => measureAll(band.locator('.hl-chip, .hl-more'), 'the band\'s rows', (es) => new Set(es.filter((e) => !e.hidden)
        .map((e) => Math.round(e.getBoundingClientRect().top))).size);
      const folded = chipsAll.filter((x) => x.hidden).length;
      if (mobile) {
        if ((await rowsOf()) > 2) issues.push(`phone band: ${await rowsOf()} rows, at most 2`);
        if (folded) {
          if (!(await more.isVisible()) || (await more.innerText()).trim() !== `+${folded}` || (await more.getAttribute('aria-expanded')) !== 'false') {
            issues.push(`phone band: ${folded} chips folded, expander "${await more.innerText()}" expanded=${await more.getAttribute('aria-expanded')}`);
          }
          await more.click();
          const open = await measureAll(band.locator('.hl-chip'), 'chips shown after "+n"', (cs) => cs.filter((x) => !x.hidden).length);
          if (open !== chipsAll.length || (await more.getAttribute('aria-expanded')) !== 'true') issues.push(`"+${folded}" showed ${open} of ${chipsAll.length}`);
          issues.push(...uxIssues(await ux(page, band, mobile)).map((x) => `open band ${x}`));
          await shoot(band, `ux-2-open-${screen}.png`);
          await more.click();
          if ((await measureAll(band.locator('.hl-chip'), 'chips folded again', (cs) => cs.filter((x) => x.hidden).length)) !== folded) issues.push('the expander did not fold the band again');
        }
      } else if (folded || await more.isVisible()) issues.push(`desktop band folds ${folded} chips`);
      // box-d's headroom is absent: GPU TEMP is no data, with the sender's own why, under the Activity tile's health line
      const nd = await page.evaluate(() => [...document.querySelectorAll('#events-health .ev-nd li')].map((l) => l.textContent));
      const amd = inp['box-d'].deep.gpu[0].temp_headroom_why;
      if (!nd.some((t) => t.startsWith('box-d · gpu_temp · ') && t.includes(amd))) issues.push(`no "box-d · gpu_temp · ${amd}" in no-data [${nd.join(' | ')}]`);
      const figs = async (box) => Object.fromEntries((await measureAll(card(page, box).locator('.figs .fig'), `${box} card figures`, (fs) =>
        fs.map((f) => [f.querySelector('span').textContent.trim(), f.querySelector('b').textContent.trim()]))));
      const a = await figs('box-a'); const b = await figs('box-b'); const cc = await figs('box-c');
      const xids = recent((k) => 2 + Math.floor(k / 15)).map(String);
      // box-a's XID finding is firing (the band says so): the card flags the figure with the finding's rise.
      const [axid, arise] = a.xid.split(' · +');
      if (!xids.includes(axid)) issues.push(`box-a XID ${a.xid}, sent ${xids.join('/')}`);
      const aChip = chipsAll.find((x) => x.box === 'box-a' && x.cls.includes('hl-firing'));
      const chipRise = ((aChip ? aChip.text : '').match(/XID \+(\d+)/) || [])[1];
      if (!arise || arise !== chipRise) issues.push(`box-a XID figure "${a.xid}" not flagged with the band's +${chipRise}`);
      if (!(await card(page, 'box-a').locator('.figs .fig.bad', { hasText: 'xid' }).count())) issues.push('box-a XID figure not in the fault colour');
      if (await card(page, 'box-b').locator('.figs .fig.bad').count()) issues.push('box-b has no xid/oom finding, yet a figure is flagged');
      if (a['oom kills'] !== '0') issues.push(`box-a OOM ${a['oom kills']}, sent 0`);
      if (b.xid !== '0' || b['oom kills'] !== '3') issues.push(`box-b XID/OOM ${b.xid}/${b['oom kills']}, sent 0/3`);
      if (cc.xid !== '—' || cc['oom kills'] !== '1') issues.push(`box-c XID/OOM ${cc.xid}/${cc['oom kills']}, sent unread/1`);
      for (const box of ['box-a', 'box-b', 'box-c']) issues.push(...uxIssues(await ux(page, card(page, box).locator('.figs'), mobile)).map((x) => `${box} figs ${x}`));
      // the machine card's badge: its box's FIRING findings (the band's), in the fault colour only above 0
      for (const box of ['box-a', 'box-b', 'box-c', 'box-d']) {
        const n = chipsAll.filter((x) => x.box === box && x.cls.includes('hl-firing'))
          .reduce((acc, x) => acc + x.title.split('\n').filter((l) => l.startsWith('FIRING ')).length, 0);
        const badge = card(page, box).locator('.card-head .hl-count');
        const [t, bad] = (await badge.count()) ? await measure(badge, `${box} badge`, (e) => [e.textContent, e.classList.contains('hl-count-bad')]) : [null, null];
        if (t !== String(n) || bad !== (n > 0)) issues.push(`${box} badge "${t}" bad=${bad}, band: ${n} firing`);
      }
      await shoot(band, `ux-2-${screen}.png`);
      await shoot(card(page, 'box-a').locator('.rings'), `ux-2-kernel-${screen}.png`);
      record(2, screen, issues);
    }
    // ── alert delivery chip: the alert path's trouble, warn (not the fault red), last in the band ──
    {
      const chip = page.locator('#lb-health .hl-failed');
      if (mobile && !(await chip.isVisible())) { await page.locator('#lb-health .hl-more').click(); }
      const issues = uxIssues(await ux(page, chip, mobile));
      const t = await chip.innerText();
      if (!/^ALERT webhook failed · HTTP 500 Internal Server Error · \d/.test(t)) issues.push(`alert line "${t}"`);
      if (!(await sameColour(chip, '--warn'))) issues.push('delivery chip not in the warn colour');
      if (await sameColour(chip, '--bad')) issues.push('delivery chip in the fault colour');
      await shoot(chip, `ux-3-${screen}.png`);
      if (mobile && (await page.locator('#lb-health .hl-more').getAttribute('aria-expanded')) === 'true') await page.locator('#lb-health .hl-more').click();
      record(3, screen, issues);
    }
    // ── the CLOSED flight-recorder strip: RoCE and disk figures from the inputs; no box health left in a card ──
    {
      const issues = [];
      const inp = inputs().boxes;
      // The full-disk line is the fleet's (fleet.demo.yml health.disk_full_pct, read here from the file the demo LB
      // mounts): the status must carry that value, and the strip is judged against the CONFIG, not the page's copy.
      const full = Number((/^health:\n(?:  .*\n)*?  disk_full_pct: *([0-9.]+)/m.exec(
        fs.readFileSync(FLEET, 'utf8')) || [])[1]);
      if (!Number.isFinite(full)) issues.push(`${FLEET} states no health.disk_full_pct`);
      const carried = await page.evaluate(() => fetch('/lb/status').then((r) => r.json())
        .then((j) => (j.health || {}).disk_full_pct));
      if (carried !== full) issues.push(`/lb/status health.disk_full_pct ${carried}, fleet.demo.yml ${full}`);
      for (const box of ['box-a', 'box-b', 'box-c', 'box-d']) {
        const st = strip(page, box);
        if ((await st.getAttribute('aria-expanded')) !== 'false') issues.push(`${box} strip not collapsed by default`);
        const parts = await measureAll(st.locator('span:not(.rec-dot)'), `${box} strip parts`,
          (ss) => ss.map((x) => [x.textContent, x.classList.contains('rec-bad')]));
        const title = (await st.getAttribute('title')) || '';
        const ports = (inp[box].rdma || {}).ports || [];
        const roce = parts.find(([t]) => t.startsWith('RoCE'));
        if (ports.length) {
          const up = ports.filter((p) => p.state === 'ACTIVE').length;
          const rising = RISING[box] || 0;
          const want = `RoCE ${up}/${ports.length} up${rising ? ` · ${rising} rising` : ''}`;
          const fault = up < ports.length || rising > 0;
          if (!roce || roce[0] !== want) issues.push(`${box} strip "${roce ? roce[0] : '(no RoCE)'}", inputs: "${want}"`);
          else if (roce[1] !== fault) issues.push(`${box} "${roce[0]}" fault colour ${roce[1]}, a fault: ${fault}`);
        } else {
          if (roce) issues.push(`${box} has no RDMA port, yet the strip says "${roce[0]}"`);
          if (!title.includes(inp[box].rdma.why)) issues.push(`${box} strip title lacks the no-RDMA why "${inp[box].rdma.why}"`);
        }
        const pcts = inp[box].disks.filter((d) => d.total_bytes > 0).map((d) => 100 * d.used_bytes / d.total_bytes);
        const top = Math.round(Math.max(...pcts));
        const disk = parts.find(([t]) => t.startsWith('disk'));
        if (!disk || disk[0] !== `disk ${top}%`) issues.push(`${box} strip "${disk ? disk[0] : '(no disk)'}", inputs: fullest "disk ${top}%"`);
        else if (disk[1] !== (top >= full)) issues.push(`${box} "${disk[0]}" fault colour ${disk[1]} (full at ${full} %)`);
        issues.push(...uxIssues(await ux(page, st, mobile)).map((x) => `${box} strip ${x}`));
      }
      const inCard = await page.locator('article.card.machine .hx-sec, article.card.machine .hx').count();
      if (inCard) issues.push(`${inCard} box-health sections still inside a machine card`);
      await shoot(page.locator('.machine-slot[data-box="box-a"]'), `ux-11-${screen}.png`);
      record(11, screen, issues);
    }
    // ── a band chip's tap: the box's card scrolled to and flashed; a RoCE chip opens that box's strip ──
    {
      const issues = [];
      const band = page.locator('#lb-health');
      const tap = async (box) => {
        await page.evaluate(() => scrollTo(0, 0));
        const ch = band.locator(`.hl-chip.hl-firing[data-box="${box}"]`);
        if (mobile && !(await ch.isVisible())) await band.locator('.hl-more').click();
        await ch.click();
        await page.waitForTimeout(900);              // the smooth scroll
        return page.evaluate((b) => {
          const c = document.querySelector(`article.card.machine[data-box="${b}"]`);
          const r = c.getBoundingClientRect();
          const s = document.querySelector(`.machine-slot[data-box="${b}"] .rec-strip`).getBoundingClientRect();
          return { flash: c.classList.contains('hl-flash'), cardSeen: r.bottom > 0 && r.top < innerHeight,
                   stripSeen: s.bottom > 0 && s.top < innerHeight };
        }, box);
      };
      // box-c's chip is GPU TEMP: scroll + flash, its strip stays closed
      const c = await tap('box-c');
      if (!c.flash || !c.cardSeen) issues.push(`box-c chip: flash=${c.flash} on screen=${c.cardSeen}`);
      if ((await strip(page, 'box-c').getAttribute('aria-expanded')) !== 'false') issues.push('a GPU TEMP chip opened box-c\'s strip');
      // box-a's chip names RoCE ERR: scroll + flash, and its strip opens (recorder.js recOpen) and is on screen
      const a = await tap('box-a');
      if (!a.flash || !a.cardSeen && !a.stripSeen) issues.push(`box-a chip: flash=${a.flash} card on screen=${a.cardSeen}`);
      if ((await strip(page, 'box-a').getAttribute('aria-expanded')) !== 'true' || !a.stripSeen) issues.push(`box-a RoCE chip: strip open=${await strip(page, 'box-a').getAttribute('aria-expanded')} on screen=${a.stripSeen}`);
      await shoot(page.locator('.machine-slot[data-box="box-a"]'), `ux-12-${screen}.png`);
      if (mobile && (await band.locator('.hl-more').getAttribute('aria-expanded')) === 'true') await band.locator('.hl-more').click();
      record(12, screen, issues);
    }
    // ── RoCE per-port lines, in the open strip ──
    {
      await openStrip(page, 'box-a'); await openStrip(page, 'box-b');
      const ra = sec(page, 'box-a', 'RoCE'); const rb = sec(page, 'box-b', 'RoCE');
      const issues = [...uxIssues(await ux(page, ra, mobile)), ...uxIssues(await ux(page, rb, mobile))];
      // A port's further counters and QoS sit behind its own <details>, OPEN when it is down or rising: box-a mlx5_0/1
      // rises, box-b mlx5_1/1 is down — both open; box-a mlx5_1/1 and box-b mlx5_0/1 are neither — closed.
      const openOf = async (s, name) => measure(s.locator('.hx-port', { hasText: name }).first().locator('details.hx-more'),
        `${name} details open`, (d) => d.open);
      for (const [s, box, name, want] of [[ra, 'box-a', 'mlx5_0/1', true], [ra, 'box-a', 'mlx5_1/1', false],
        [rb, 'box-b', 'mlx5_1/1', true], [rb, 'box-b', 'mlx5_0/1', false]]) {
        if ((await openOf(s, name)) !== want) issues.push(`${box} ${name} counters ${want ? 'closed' : 'open'}, ${want ? 'down or rising' : 'neither down nor rising'}`);
      }
      const ports = await ra.locator('.hx-port').allInnerTexts();
      const p0 = ports.find((t) => t.startsWith('mlx5_0/1')) || '';
      if (!p0.startsWith('mlx5_0/1 · ACTIVE · 200 Gb/s · MTU 9000')) issues.push(`box-a port line "${p0.split('\n')[0]}"`);
      // the line shows ONLY the non-zero / rising counters inline; a counter at 0 is behind the details
      const inline = await ra.locator('.hx-port').first().locator(':scope > .hx-cs .hx-c').allInnerTexts();
      if (inline.some((t) => /^[A-Z ]+ 0$/.test(t.trim()))) issues.push(`box-a mlx5_0/1 shows a zero counter inline: ${inline.join(' | ')}`);
      const oob = (p0.match(/OOB ([\d,]+)/) || [])[1];
      const oobs = recent((k) => 1200 + 3 * k).map(en);
      if (!oobs.includes(oob)) issues.push(`OOB ${oob}, sent ${oobs.join('/')}`);
      const ack = (p0.match(/ACK TO ([\d,]+)/) || [])[1];
      if (!recent((k) => 40 + k).map(en).includes(ack)) issues.push(`ACK TO ${ack}`);
      if (!/OOB [\d,]+ \+1\.5\/s/.test(p0)) issues.push(`OOB rate not +1.5/s (3 per ${inputs().every_s} s send): "${p0}"`);
      // ethtool's PHY counters (a discard rising is a fault; pause frames are PFC at work: a rate, unflagged) and dcb QoS
      const rdp = (p0.match(/RX DISC PHY ([\d,]+)/) || [])[1];
      if (!recent((k) => 310 + 2 * k).map(en).includes(rdp)) issues.push(`RX DISC PHY ${rdp}, sent ${recent((k) => 310 + 2 * k).map(en).join('/')}`);
      if (!(await ra.locator('.hx-port').first().locator('.hx-c.up', { hasText: 'RX DISC PHY' }).count())) issues.push('RX DISC PHY rising, not flagged');
      if (await ra.locator('.hx-port').first().locator('.hx-c.up', { hasText: 'RX PAUSE' }).count()) issues.push('RX PAUSE flagged as a fault (it is PFC at work)');
      for (const w of ['PFC 3', 'trust —', 'DSCP map 8']) if (!p0.includes(w)) issues.push(`box-a mlx5_0/1: no "${w}" in "${p0}"`);
      const trustWhy = await ra.locator('.hx-port').first().locator('.hx-c.none', { hasText: 'trust' }).getAttribute('title');
      if (!/Operation not supported/.test(trustWhy || '')) issues.push(`trust — without mlx5's words: "${trustWhy}"`);
      if (!/RoCE · 2 ports · 1 rising/i.test(await ra.locator('.hx-h').innerText())) issues.push('box-a head not "2 ports · 1 rising"');
      const pb = await rb.locator('.hx-port').allInnerTexts();
      const down = pb.find((t) => t.startsWith('mlx5_1/1 · DOWN')) || '';
      // box-b mlx5_0/1's QoS is behind its closed details (neither down nor rising): read from the markup, then opened
      const pb0 = rb.locator('.hx-port', { hasText: 'mlx5_0/1' }).first();
      if (!(await pb0.textContent()).includes('PFC off')) issues.push('box-b mlx5_0/1 not "PFC off"');
      await pb0.locator('details.hx-more > summary').click();
      if (!(await pb0.innerText()).includes('PFC off')) issues.push('box-b mlx5_0/1 "PFC off" not shown on tap');
      issues.push(...uxIssues(await ux(page, rb, mobile)).map((x) => `opened ${x}`));
      if (!/RX DISC PHY —/.test(down)) issues.push(`box-b mlx5_1/1: ethtool refused, yet PHY counters shown: "${down}"`);
      if (!down) issues.push('box-b mlx5_1/1 not DOWN');
      const ld = (down.match(/LINK DN ([\d,]+)/) || [])[1];
      const lds = recent((k) => 2 + Math.floor(k / 10), 6).map(en);
      if (!lds.includes(ld)) issues.push(`box-b LINK DN ${ld}, sent ${[...new Set(lds)].join('/')}`);
      // The strip agrees with the band: box-b's RoCE ERR finding names mlx5_1/1 link_downed, so the head counts the port
      // as rising and the LINK DN chip is flagged with the finding's window rise — even when this sample's rate is 0.
      if (!/1 down · 1 rising/i.test(await rb.locator('.hx-h').innerText())) issues.push(`box-b head "${await rb.locator('.hx-h').innerText()}" not "1 down · 1 rising"`);
      if (!/LINK DN [\d,]+ \+\d/.test(down)) issues.push(`box-b LINK DN not flagged with its rise: "${down}"`);
      if (!(await rb.locator('.hx-port.down .hx-c.up', { hasText: 'LINK DN' }).count())) issues.push('box-b LINK DN chip not in the fault colour');
      await shoot(ra, `ux-1-${screen}.png`);
      await shoot(rb, `ux-1-boxb-${screen}.png`);
      record(1, screen, issues);
    }
    // ── disks + models folders, in the open strip ──
    {
      const disks = sec(page, 'box-a', 'DISKS'); const models = sec(page, 'box-a', 'MODELS');
      const issues = [...uxIssues(await ux(page, disks, mobile)), ...uxIssues(await ux(page, models, mobile))];
      const d = inputs().boxes['box-a'].disks.find((x) => x.path === '/mnt/models');
      const want = `${en(d.used_bytes / 1e9)} / ${en(d.total_bytes / 1e9)} GB · ${Math.round(100 * d.used_bytes / d.total_bytes)}%`;
      const rows = await disks.locator('.hx-disk').allInnerTexts();
      if (!rows.some((t) => t.includes('/mnt/models') && t.includes(want))) issues.push(`no "/mnt/models ${want}" in [${rows.join(' | ')}]`);
      const m = inputs().boxes['box-a'].models[0];
      const total = Object.values(m.dirs).reduce((a, b) => a + b, 0) + m.more_bytes;
      const head = await models.locator('.hx-h').innerText();
      if (!head.includes(`${Math.round(total / 1e9)} GB`)) issues.push(`models head "${head}", sent ${Math.round(total / 1e9)} GB`);
      const f = await models.locator('.hx-f').allInnerTexts();
      const q = (m.dirs['Qwen3.8-27B-NVFP4'] / 1e9).toFixed(1);
      if (!f.some((t) => t.startsWith('Qwen3.8-27B-NVFP4') && t.includes(`${q} GB · 1 ckpt`))) issues.push(`no Qwen3.8-27B-NVFP4 ${q} GB row`);
      if (!f.some((t) => t.startsWith('dflash2-27b-tp2') && t.includes('2 ckpt'))) issues.push('dflash2 folder not "2 ckpt"');
      const box = await boxOf(disks, `ux-5-${screen}.png: box-a DISKS section`);
      const box2 = await boxOf(models, `ux-5-${screen}.png: box-a MODELS section`);
      const sy = await page.evaluate(() => scrollY);
      await page.screenshot({ path: `${OUT}/ux-5-${screen}.png`, fullPage: true, clip: { x: box.x - 4, y: box.y + sy - 4,
        width: Math.max(box.width, box2.width) + 8, height: box2.y + box2.height - box.y + 8 } });
      record(5, screen, issues);
    }
    // ── Activity tile: under the band, 3 rows collapsed, "<n> ›" for the rest, alert sends folded into their row ──
    {
      const ev = page.locator('#events-section');
      await page.evaluate(() => scrollTo(0, 0));
      const issues = uxIssues(await ux(page, ev, mobile));
      // where it stands: directly under the band, over the Cluster
      const [bandBottom, evTop, clusterTop] = await page.evaluate(() => [document.getElementById('lb-health').getBoundingClientRect().bottom,
        document.getElementById('events-section').getBoundingClientRect().top, document.getElementById('cluster-section').getBoundingClientRect().top]);
      if (!(evTop >= bandBottom && evTop < clusterTop)) issues.push(`Activity at ${Math.round(evTop)} px: band ends ${Math.round(bandBottom)}, Cluster at ${Math.round(clusterTop)}`);
      const exp = page.locator('#events-expand');
      const collapsed = await page.locator('#events-list .ev-row').count();
      if (collapsed !== 3) issues.push(`collapsed Activity shows ${collapsed} rows, 3`);
      await shoot(ev, `ux-8-${screen}.png`);
      const restTxt = (await exp.innerText()).trim();
      if ((await exp.getAttribute('aria-expanded')) !== 'false' || !/^[\d,]+ ›$/.test(restTxt)) issues.push(`expander "${restTxt}" expanded=${await exp.getAttribute('aria-expanded')}`);
      await exp.click();
      await page.waitForFunction(() => document.getElementById('events-expand').getAttribute('aria-expanded') === 'true');
      const listed = await page.locator('#events-list .ev-row').count();
      const rest = Number(restTxt.replace(/\D/g, ''));
      if (listed !== 3 + rest) issues.push(`"${restTxt}" opened ${listed} rows, 3 + ${rest}`);
      const kept = await page.evaluate(() => { try { return localStorage.getItem('lb.activity'); } catch (e) { return `unreadable: ${e.message}`; } });
      if (kept !== '{"open":true}') issues.push(`the open choice is not kept in this browser: lb.activity=${kept}`);
      issues.push(...uxIssues(await ux(page, ev, mobile)).map((x) => `open ${x}`));
      const rows = await ev.locator('.ev-row').allInnerTexts();
      if (!rows.some((t) => t.includes('clock box-c: ok by admin'))) issues.push('seeded power.action event missing');
      if (!rows.some((t) => t.includes('clock box-d: ok by admin'))) issues.push('POSTed power.action event (up.sh) missing');
      // ALERT SENDS FOLDED: every webhook send the receiver refused is a "✗ webhook" mark (warn) on the finding row it
      // sent. The tile's count is its rows; the feed's records (GET /lb/events, the seeded + POSTed + live ones) are
      // the rows + the marks — no send dropped, none drawn twice.
      const marks = await measureAll(ev.locator('.ev-row .ev-al'), 'alert marks', (ms) => ms.map((m) => [m.textContent, m.className, m.title]));
      const count = Number((await page.locator('#events-count').innerText()).replace(/\D/g, ''));
      const records = await page.evaluate(async () => (await (await fetch('/lb/events', { cache: 'no-store' })).json()).events.length);
      if (count !== listed) issues.push(`count ${count}, rows ${listed}`);
      if (records < expected.events + 1 || records !== listed + marks.length) issues.push(`${records} records, rows ${listed} + marks ${marks.length}, seeded ${expected.events} + 1 POSTed`);
      if (!marks.length) issues.push('no alert send folded into a finding row (the receiver refuses every alert)');
      for (const [t, cls, title] of marks) {
        if (t !== '✗ webhook' || !cls.includes('ev-warn') || !/^webhook failed · HTTP 500/.test(title)) issues.push(`mark "${t}" ${cls} "${title}"`);
      }
      const markRow = ev.locator('.ev-row', { has: page.locator('.ev-al') }).first();
      if (marks.length && !/finding\.firing/.test(await markRow.innerText())) issues.push(`a mark on "${(await markRow.innerText()).slice(0, 60)}", not a finding row`);
      // the health line: what the band does not say — no firing / pending counts
      const line = (await page.locator('#events-health > span').first().innerText()).trim();
      const nd = await page.locator('#events-health .ev-nd li').count();
      if (!new RegExp(`^HEALTH ${nd} no data · ALERTS webhook · 0 sent · [\\d,]+ failed$`).test(line)) issues.push(`health line "${line}", ${nd} no data, the receiver refuses every send`);
      if (/firing|pending/.test(line)) issues.push(`health line repeats the band: "${line}"`);
      await shoot(ev, `ux-8-open-${screen}.png`);
      await ev.locator('button.ev-chip[data-g="power"]').click();
      await page.waitForFunction(() => [...document.querySelectorAll('#events-list .ev-row .ev-k')].every((k) => k.textContent === 'power.action'));
      const pw = await page.locator('#events-list .ev-row').count();
      if (pw !== 2) issues.push(`power filter shows ${pw} rows: 1 seeded + 1 POSTed`);
      await ev.locator('button.ev-chip[data-g=""]').click();
      await page.waitForFunction((n) => document.querySelectorAll('#events-list .ev-row').length >= n, listed);
      await exp.click();
      if ((await page.locator('#events-list .ev-row').count()) !== 3) issues.push('"‹" did not collapse Activity to 3 rows');
      record(8, screen, issues);
    }
    // ── daily rollup dialog ──
    {
      const open = async () => { await page.locator('.obs-daily').click(); await page.waitForSelector('#obs-dialog .obs-model'); };
      const btn = uxIssues(await ux(page, page.locator('#cluster-links'), mobile)).filter((x) => x.includes('daily')).map((x) => `daily button ${x}`);
      await open();
      const box = page.locator('#obs-dialog .obs-box');
      const issues = [...btn, ...uxIssues(await ux(page, box, mobile))];
      const sum = await box.locator('.obs-sum').innerText();
      if (!sum.includes(`24 h ${en(expected.perf.total_24h_gen)} tok`)) issues.push(`summary "${sum}", seeded 24 h ${en(expected.perf.total_24h_gen)}`);
      for (const [id, m] of Object.entries(expected.perf.models)) {
        const sec7 = box.locator(`.obs-model[data-model="${id}"]`);
        if (!(await sec7.count())) { issues.push(`no section for ${id}`); continue; }
        const t = await sec7.innerText();
        if (!t.includes(`24 h ${en(m.last_24h_gen)} tok`)) issues.push(`${id} 24 h: "${t.split('\n')[1]}", seeded ${en(m.last_24h_gen)}`);
        if (!t.includes(`peak min ${fig(m.peak_tok_s)} tok/s`)) issues.push(`${id} peak, seeded ${fig(m.peak_tok_s)}`);
        // no tok/s bar row (a day's pooled rate drew nearly the tokens' shape); the tokens/day row stays
        const bars = await sec7.locator('.obs-bl').allInnerTexts();
        if (bars.some((x) => /tok\/s/.test(x)) || !bars.some((x) => x.startsWith('tokens/day'))) issues.push(`${id} bar rows [${bars.join(' | ')}]`);
      }
      const best = Math.max(...Object.values(expected.perf.models).map((m) => m.peak_tok_s));
      if (!sum.includes(`peak min ${fig(best)} tok/s`)) issues.push(`summary "${sum}" lacks the best peak min ${fig(best)} tok/s`);
      await page.waitForTimeout(300);
      await shoot(box, `ux-7-${screen}.png`);
      issues.push(...await dialogCloses(page), ...await escapeCloses(page, open));
      record(7, screen, issues);
    }
    // ── Check dialog ──
    {
      const btn = card(page, 'box-c').locator('.obs-btn[data-obs="check"]');
      const open = async () => { await btn.click(); await page.waitForSelector('#obs-dialog .obs-row'); };
      const issues = uxIssues(await ux(page, card(page, 'box-c').locator('.obs-tools'), mobile)).map((x) => `card buttons ${x}`);
      await open();
      const box = page.locator('#obs-dialog .obs-box');
      issues.push(...uxIssues(await ux(page, box, mobile)));
      const rows = await measureAll(box.locator('.obs-row'), 'check rows', (rs) => rs.map((r) => ({ ok: r.classList.contains('ok'), cap: r.querySelector('.obs-cap').textContent, text: r.innerText })));
      const title = await box.locator('.cn-title').innerText();
      const pass = rows.filter((r) => r.ok).length;
      if (!title.includes(`${pass}/${rows.length} pass`)) issues.push(`title "${title}" vs ${pass}/${rows.length}`);
      if (!rows.some((r) => r.ok && /telemetry/.test(r.cap))) issues.push('no passing telemetry row (the sender is up)');
      if (!rows.some((r) => !r.ok && /engine|server|backend/.test(r.cap))) issues.push('no failing engine row (box-c engine is down)');
      // a failed row prints its SHORT hint, the whole recovery in its title
      const hints = await measureAll(box.locator('.obs-row .obs-hint'), 'check hints', (hs) => hs.map((h) => [h.textContent.trim(), h.title || '']));
      if (!hints.length) issues.push('the failing engine row has no hint');
      for (const [t, title] of hints) if (!t || !title || t === title || title.length <= t.length) issues.push(`hint "${t}" / title "${title}": short on the row, the whole in the title`);
      // box-c has no ssh address (fleet.demo.yml): no SSH row, and "ssh —" opening (tap) to the gateway's reason.
      const sshSum = box.locator('.obs-why summary');
      if ((await sshSum.count()) !== 1 || (await sshSum.innerText()).trim().toLowerCase() !== 'ssh —') issues.push('no "ssh —" for box-c');
      else {
        await sshSum.click();
        const r = await box.locator('.obs-why-r').innerText();
        if (!r.includes('boxes.box-c.ssh') || !(await box.locator('.obs-why-r').isVisible())) issues.push(`ssh reason "${r}" not shown on tap`);
      }
      await page.waitForTimeout(200);
      await shoot(box, `ux-9-${screen}.png`);
      issues.push(...await dialogCloses(page), ...await escapeCloses(page, open));
      record(9, screen, issues);
    }
    // ── history dialog ──
    {
      const btn = card(page, 'box-c').locator('.obs-btn[data-obs="history"]');
      const open = async () => { await btn.click(); await page.waitForSelector('#obs-dialog .obs-tile'); };
      await open();
      const box = page.locator('#obs-dialog .obs-box');
      const issues = uxIssues(await ux(page, box, mobile));
      const tile = async () => box.locator('.obs-tile[data-metric="gpu_temp_c"] .obs-th b').innerText();
      const want = `max ${fig(expected.max_gpu_temp_c['box-c'])} °C`;
      const t48 = await tile();
      if (!t48.includes(want)) issues.push(`48 h GPU TEMP "${t48}", seeded ${want}`);
      // every chart carries its start / end ticks, from its samples' own timestamps ("Nov 25 14:00" … "now")
      const ticks = await measureAll(box.locator('.obs-tile'), 'history ticks', (ts) => ts.map((t) => [t.dataset.metric, (t.querySelector('.obs-tt') || {}).textContent || '']));
      for (const [k, tt] of ticks) if (!/[A-Z][a-z]{2} \d{1,2} \d\d:\d\d/.test(tt)) issues.push(`${k} chart ticks "${tt}"`);
      await page.waitForTimeout(200);
      await shoot(box, `ux-10-${screen}.png`);
      await box.locator('.obs-tier[data-tier="30d"]').click();
      await page.waitForFunction(() => document.querySelector('#obs-dialog .obs-tier[data-tier="30d"]')?.getAttribute('aria-pressed') === 'true' && document.querySelector('#obs-dialog .obs-tile'));
      const t30 = await tile();
      if (!t30.includes(want)) issues.push(`30 d GPU TEMP "${t30}", seeded ${want}`);
      issues.push(...uxIssues(await ux(page, box, mobile)).map((x) => `30d ${x}`));
      await shoot(box, `ux-10-30d-${screen}.png`);
      await box.locator('.obs-tier[data-tier="48h"]').click();
      issues.push(...await dialogCloses(page), ...await escapeCloses(page, open));
      record(10, screen, issues);
    }
    // ── the page as a whole: overflow + console, recorded on every change of this screen ──
    {
      const [sw, iw] = await page.evaluate(() => [document.documentElement.scrollWidth, innerWidth]);
      const { fails, gated } = splitGate(errors);
      record('page', screen, gated.map((e) => `pre-existing, not under audit: ${e}`));
      // the style under audit is the one applied, with no notice (a refused stored choice opens the default)
      const [applied, notice] = await page.evaluate(() => [document.documentElement.getAttribute('data-style'),
        document.getElementById('style-notice').hidden ? '' : document.getElementById('style-notice').innerText.trim()]);
      const page_issues = [...(sw > iw ? [`page overflows ${sw} > ${iw}`] : []), ...fails.map((e) => `console: ${e}`),
        ...(applied !== style ? [`style "${applied}" applied, not "${style}"`] : []), ...(notice ? [`style notice: ${notice}`] : [])];
      for (const n of [1, 2, 3, 5, 7, 8, 9, 10, 11, 12]) {
        const r = results[n].find((x) => x.screen === screen);
        r.issues.push(...page_issues); r.pass = !r.issues.length;
      }
    }
    // ── fidelity on /benchmarks: the runs table, a ticked A/B compare (ΔNLL + CI), the Run button ──
    {
      const p = await ctx.newPage();
      const errs = [];
      p.on('console', (m) => { if (m.type() === 'error') errs.push(m.text()); });
      p.on('pageerror', (e) => errs.push(`pageerror ${e.message}`));
      p.on('response', (r) => { if (r.status() >= 400) errs.push(`HTTP ${r.status()} ${new globalThis.URL(r.url()).pathname}`); });
      await p.goto(`${URL}/benchmarks#fidelity`);
      await p.waitForSelector('#fid-root .fid-row', { timeout: 30000 });
      const fx = expected.fidelity;
      const sec4 = p.locator('#fidelity');
      const issues = [];
      const run = p.locator('#fid-root button.fid-run');
      if ((await run.innerText()).trim() !== 'Run' || await run.isDisabled()) issues.push(`Run button "${await run.innerText()}" disabled=${await run.isDisabled()}`);
      const rows = await p.locator('#fid-root tr.fid-row').count();
      if (rows !== fx.runs.length) issues.push(`${rows} rows, seeded ${fx.runs.length}`);
      for (const r of fx.runs) {
        const tr = p.locator('#fid-root tr.fid-row', { has: p.locator(`td[title="${r.id}"]`) });
        const t = (await tr.innerText()).replace(/\s+/g, ' ');
        if (r.status === 'done') {
          for (const want of [r.mean_nll.toFixed(4), r.ppl.toFixed(3), en(r.tokens), r.kv]) if (!t.includes(want)) issues.push(`${r.kv} run: no "${want}" in "${t}"`);
        } else {
          if (!/failed/.test(t) || !/no token logprobs/.test(t)) issues.push(`failed run row "${t}"`);
          // ONE line (status · the error's first clause) opening to the server's whole words
          const d = tr.locator('details.fid-errd');
          if (!(await d.count())) issues.push('failed run: no one-line <details>');
          else {
            const sum = d.locator('summary');
            const [h, lh] = await measure(sum, 'fidelity failed-run summary height',
              (e) => [e.getBoundingClientRect().height, parseFloat(getComputedStyle(e).lineHeight) || 20]);
            const whole = (await sum.getAttribute('title')) || '';
            if (await measure(d, 'fidelity failed-run details open', (x) => x.open)) issues.push('failed run error open by default');
            if (h > Math.max(44, lh * 1.6)) issues.push(`failed run summary ${Math.round(h)} px tall: more than one line`);
            await sum.click();
            const shown = await d.locator('.fid-err').innerText();
            if (!(await d.locator('.fid-err').isVisible()) || shown !== whole || !/no token logprobs/.test(shown)) issues.push(`failed run error "${shown}" on tap, title "${whole}"`);
            issues.push(...uxIssues(await ux(p, tr, mobile)).map((x) => `opened error ${x}`));
            await sum.click();
          }
        }
        const n = Object.keys(r.why || {}).length;
        if (n) {
          // A target the LB could not resolve: dashes, then "<n> unread" opening (tap) to the seeded reason.
          if (!t.includes('vllm × — × TP —')) issues.push(`unresolved run: no "vllm × — × TP —" in "${t}"`);
          const sum = tr.locator('.fid-why summary');
          if ((await sum.innerText()).trim() !== `${n} unread`) issues.push(`unresolved run label "${await sum.innerText()}", ${n} unread`);
          await sum.click();
          const li = (await tr.locator('.fid-why li').allInnerTexts()).join(' | ');
          const reason = Object.values(r.why)[0];
          if (!li.includes(`${Object.keys(r.why).join(', ')} · ${reason}`)) issues.push(`unresolved run reasons "${li}", seeded "${reason}"`);
          if (!(await tr.locator('.fid-why li').first().isVisible())) issues.push('the reason did not open on tap');
        } else if (await tr.locator('.fid-why').count()) issues.push(`${r.kv} run: nothing unread, yet a why`);
      }
      // Tick B first, then A: the page must still make the OLDER run A.
      await p.locator(`input[data-fid-run="${fx.ab.b}"]`).check();
      await p.locator(`input[data-fid-run="${fx.ab.a}"]`).check();
      await p.waitForSelector('#fid-root .fid-cmp b', { timeout: 30000 });
      const cmp = (await p.locator('#fid-root .fid-cmp-box').innerText()).replace(/\s+/g, ' ');
      if (fx.runs.some((r) => Object.keys(r.why || {}).length) && !(await p.locator('#fid-root details.fid-why[open]').count())) {
        issues.push('the opened reasons closed when the compare repainted the section');
      }
      const sgn = (v) => `${v > 0 ? '+' : v < 0 ? '−' : '±'}${Math.abs(v).toFixed(4)}`;
      if (!cmp.includes(`ΔNLL ${sgn(fx.ab.d_nll)}`)) issues.push(`compare "${cmp}", seeded ΔNLL ${sgn(fx.ab.d_nll)}`);
      const ci = cmp.match(/CI \[([−+±])([\d.]+), ([−+±])([\d.]+)\]/);
      const num = (sg, v) => (sg === '−' ? -1 : 1) * Number(v);
      if (!ci) issues.push(`no CI in "${cmp}"`);
      else if (!(num(ci[1], ci[2]) <= fx.ab.d_nll && fx.ab.d_nll <= num(ci[3], ci[4]))) issues.push(`CI [${ci[1]}${ci[2]}, ${ci[3]}${ci[4]}] does not hold ΔNLL`);
      if (!cmp.includes(`A ${fx.ab.nll_a.toFixed(4)}`) || !cmp.includes(`B ${fx.ab.nll_b.toFixed(4)}`)) issues.push(`A/B NLL in "${cmp}", seeded ${fx.ab.nll_a}/${fx.ab.nll_b}`);
      if (!cmp.includes(`${fx.ab.docs} docs`) || !cmp.includes(`${en(fx.ab.tokens)} tokens`)) issues.push(`docs/tokens in "${cmp}"`);
      if (!/B worse/.test(cmp)) issues.push(`verdict in "${cmp}" (NVFP4 KV seeded 1.5 % worse)`);
      issues.push(...uxIssues(await ux(p, sec4, mobile)));
      const [sw, iw] = await p.evaluate(() => [document.documentElement.scrollWidth, innerWidth]);
      if (sw > iw) issues.push(`page overflows ${sw} > ${iw}`);
      const { fails, gated } = splitGate(errs);
      issues.push(...fails.map((e) => `console: ${e}`));
      const pg = results.page.find((x) => x.screen === screen);
      pg.issues.push(...gated.map((e) => `pre-existing, not under audit (/benchmarks): ${e}`)); pg.pass = !pg.issues.length;
      await shoot(sec4, `ux-4-${screen}.png`);
      record(4, screen, issues);
      await p.close();
    }
    // ── /metrics as the browser renders it ──
    {
      const p = await ctx.newPage();
      const errs = [];
      p.on('console', (m) => { if (m.type() === 'error') errs.push(m.text()); });
      const r = await p.goto(`${URL}/metrics`);
      const text = await p.evaluate(() => document.body.innerText);
      const issues = errs.map((e) => `console: ${e}`);
      if (r.status() !== 200) issues.push(`HTTP ${r.status()}`);
      const inp = inputs().boxes;
      const want = [`fleet_box_cpu_percent{box="box-c"} ${inp['box-c'].cpu_percent}`,
        `fleet_box_ram_percent{box="box-b"} ${inp['box-b'].ram_percent}`,
        `fleet_box_gpu_utilization_percent{box="box-c",card="0"} ${inp['box-c'].gpus[0].use_pct}`, 'collector_last_success_timestamp_seconds{box="box-a"}'];
      for (const w of want) if (!text.includes(w)) issues.push(`no line "${w}"`);
      for (const w of ['undefined', 'NaN']) if (new RegExp(`\\b${w}\\b`).test(text)) issues.push(`shows ${w}`);
      const [sw, iw] = await p.evaluate(() => [document.documentElement.scrollWidth, innerWidth]);
      if (sw > iw) issues.push(`overflows ${sw} > ${iw}`);
      await p.screenshot({ path: `${OUT}/ux-6-${screen}.png` });
      record(6, screen, issues);
      await p.close();
    }
    await ctx.close();
  }
}
await browser.close();

fs.writeFileSync(`${OUT}/audit.json`, JSON.stringify(results, null, 1));
let failed = 0;
for (const n of Object.keys(results).sort((a, b) => (Number(a) || 99) - (Number(b) || 99))) {
  console.log(`\n${Number(n) ? '#' : ''}${n}\n| screen | result | issue |\n|---|---|---|`);
  for (const r of results[n]) {
    if (!r.pass) failed++;
    console.log(`| ${r.screen} | ${r.pass ? 'PASS' : 'FAIL'} | ${r.issues.join('; ').replace(/\|/g, '\\|') || '—'} |`);
  }
}
console.log(`\n${failed ? `${failed} FAIL` : 'all PASS'}`);
process.exit(failed ? 1 : 0);
