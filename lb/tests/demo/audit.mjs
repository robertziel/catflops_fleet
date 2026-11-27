/* THE UX AUDIT (Playwright, run by audit.sh inside the pinned Playwright image, in demo-lb's network namespace).
 *
 * For every screen — desktop 1440×1000 and mobile 390×844, in the dark default style (vibe) and one light style
 * (clean) — it opens the panel, finds each change, screenshots it as ux-<n>-<desktop|mobile>-<style>.png
 * and asserts:
 *   console     no console error and no uncaught page error (each listed with its URL when it is a failed load)
 *   overflow    no horizontal page overflow (scrollWidth ≤ the viewport)
 *   clipped     no text cut by its box (scrollWidth > clientWidth on an element that hides overflow), and no box that
 *               scrolls sideways (what is past its edge is off screen)
 *   overlap     no two text leaves of the feature intersect
 *   numbers     the figures shown equal the harness's INPUTS (sender.py's inputs.json, seed.py's expected.json) — never
 *               the gateway's own status, which would check the gateway against itself
 *   figures     no label is prose: a text leaf of 7+ words with no digit fails
 *   tap         at 390 px every control of the feature is ≥ 32 × 32 px
 *   dialogs     a dialog closes on × and on Escape
 *   tokens      nothing shows `undefined`, `NaN` or `null`, and no count of 5+ digits is printed ungrouped
 *
 * Usage: node audit.mjs <out dir>   (env AUDIT_URL, default http://127.0.0.1:8010; AUDIT_DATA, default /data)
 * Writes <out dir>/audit.json and prints one table per change; exits 1 when any row fails.
 */
import { chromium } from '/pw/node_modules/playwright/index.mjs';
import fs from 'node:fs';

const OUT = process.argv[2] || '/out';
const URL = process.env.AUDIT_URL || 'http://127.0.0.1:8010';
const DATA = process.env.AUDIT_DATA || '/data';
const SCREENS = [['desktop', { width: 1440, height: 1000 }], ['mobile', { width: 390, height: 844 }]];
const STYLES = ['vibe', 'clean'];
const TAP_MIN = 32;

const inputs = () => JSON.parse(fs.readFileSync(`${DATA}/inputs.json`, 'utf8'));
const expected = JSON.parse(fs.readFileSync(`${DATA}/expected.json`, 'utf8'));
const en = (n) => Math.round(n).toLocaleString('en-US');
const fig = (v) => { const a = Math.abs(v); return a >= 100 ? Math.round(v).toLocaleString('en-US') : a >= 10 ? String(Math.round(v * 10) / 10) : String(Math.round(v * 100) / 100); };
/** The last `n` sends' value of a counter the sender derives from its send number k. */
const recent = (f, n = 4) => { const k = inputs().k; return Array.from({ length: n }, (_, i) => f(k - i)).filter((v) => v >= 0); };

/** The generic UX checks over one root element (in the page). */
async function ux(page, root, mobile) {
  return root.evaluate((el, { mobile, TAP_MIN }) => {
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
    for (const m of text.matchAll(/(?<![\w.,:\/-])\d{5,}(?![\w.,:\/-])/g)) out.tokens.push(`ungrouped ${m[0]}`);
    for (const w of ['undefined', 'NaN', 'null']) if (new RegExp(`\\b${w}\\b`).test(text)) out.tokens.push(w);
    return out;
  }, { mobile, TAP_MIN });
}

const results = {};   // change → [{screen, pass, issues}]
function record(n, screen, issues) {
  (results[n] ||= []).push({ screen, pass: !issues.length, issues });
}
/** The panel's access.js asks the sign-in gate for /auth/session; on the LB's own port (no gate in front, as here)
 *  that is a 404 by design (access.js: "nothing for … the LB's own port") and the browser logs the failed load. It is
 *  not a change under audit: it is counted on its own row (`page`), never dropped. Each gate 404 brings one "Failed to load
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
const sec = (page, box, head) => card(page, box).locator('.hx-sec', { has: page.locator('.hx-h', { hasText: head }) }).first();
/** The element's box, shot from the full page: the panel repaints its cards on every status patch, so an element
 *  handle can be replaced between "find" and "shoot" — a clip of its box is not. */
async function shoot(loc, file) {
  // A repaint can land between the lookup and the box (null while the element is replaced): look again, a few times.
  let b = null;
  for (let i = 0; i < 20 && !b; i++) { b = await loc.boundingBox().catch(() => null); if (!b) await loc.page().waitForTimeout(150); }
  if (!b) throw new Error(`${file}: the element never had a box`);
  const sy = await loc.page().evaluate(() => scrollY);
  await loc.page().screenshot({ path: `${OUT}/${file}`, fullPage: true,
    clip: { x: Math.max(0, b.x - 4), y: Math.max(0, b.y + sy - 4), width: b.width + 8, height: b.height + 8 } });
}

async function dialogCloses(page) {
  const issues = [];
  const d = page.locator('#obs-dialog');
  await d.locator('.obs-x').click();
  if (await d.evaluate((x) => x.open)) issues.push('× did not close the dialog');
  return issues;
}
async function escapeCloses(page, open) {
  await open();
  await page.keyboard.press('Escape');
  return (await page.locator('#obs-dialog').evaluate((x) => x.open)) ? ['Escape did not close the dialog'] : [];
}

const browser = await chromium.launch();
for (const [scr, viewport] of SCREENS) {
  for (const style of STYLES) {
    const screen = `${scr}-${style}`;
    const mobile = scr === 'mobile';
    const ctx = await browser.newContext({ viewport, deviceScaleFactor: 2 });
    await ctx.addInitScript((s) => { try { localStorage.setItem('lb.style', s); } catch (e) { /* recorded by the page */ } }, style);
    const page = await ctx.newPage();
    const errors = [];
    page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
    page.on('pageerror', (e) => errors.push(`pageerror ${e.message}`));
    page.on('response', (r) => { if (r.status() >= 400) errors.push(`HTTP ${r.status()} ${new globalThis.URL(r.url()).pathname}`); });
    await page.goto(`${URL}/`);
    // up.sh's states take ~70 s to settle: XID fires 30 s + hold after the first send, box-b's link_downed 20 s + hold,
    // box-c's engine stops at 45 s.
    await page.waitForFunction(() => {
      const t = [...document.querySelectorAll('#lb-health .hl-chip')].map((c) => c.textContent);
      return t.some((x) => x.startsWith('FIRING · box-a · XID')) && t.some((x) => x.startsWith('FIRING · box-b · RoCE ERR'))
        && t.some((x) => x.includes('SERVER not live')) && document.querySelector('#lb-health .hl-failed');
    }, null, { timeout: 150000 });
    await page.waitForSelector('article.card.machine[data-box="box-a"] .hx-port', { timeout: 30000 });
    await page.waitForSelector('#events-list .ev-row', { timeout: 30000 });
    await page.waitForTimeout(800);

    // ── findings band, and the kernel figures on the machine cards ──
    {
      const band = page.locator('#lb-health');
      const issues = uxIssues(await ux(page, band, mobile));
      const chips = await band.locator('.hl-chip').allInnerTexts();
      const inp = inputs().boxes;
      const c = inp['box-c'].deep.gpu[0].temp_headroom_c;
      const ramB = (inp['box-b'].ram_total_mb - inp['box-b'].ram_used_mb) / 1024;
      // health.py: the floor is the fleet's share (fleet.demo.yml low_ram_fraction 0.10) of box-b's OWN RAM, rounded as
      // the gateway rounds it; GPU POWER's ceiling is the GB10 type's idle draw (8.9 W) + the demo's 2 W margin.
      const ramFloor = Math.round(0.10 * inp['box-b'].ram_total_mb / 1024 * 10) / 10;
      const wB = inp['box-b'].deep.gpu[0].power_w;
      const want = [`FIRING · box-c · GPU TEMP ${c} °C to limit ≤ 5 °C`, 'FIRING · box-b · RoCE ERR +',
        `FIRING · box-b · RAM AVAIL ${fig(ramB)} GiB < ${fig(ramFloor)} GiB`, `FIRING · box-b · GPU POWER ${fig(wB)} W ≤ 10.9 W`,
        'FIRING · box-a · RoCE ERR +', 'FIRING · box-a · XID +', 'PENDING · box-c · SERVER not live · '];
      for (const w of want) if (!chips.some((t) => t.startsWith(w))) issues.push(`no chip "${w}…" in [${chips.join(' | ')}]`);
      if (chips.some((t) => t.includes('box-d · GPU TEMP'))) issues.push('box-d GPU TEMP chip: its headroom is absent, so no data');
      // box-d's headroom is absent: GPU TEMP is no data, with the sender's own why, under the Activity tile's health line
      const nd = await page.evaluate(() => [...document.querySelectorAll('#events-health .ev-nd li')].map((l) => l.textContent));
      const amd = inp['box-d'].deep.gpu[0].temp_headroom_why;
      if (!nd.some((t) => t.startsWith('box-d · gpu_temp · ') && t.includes(amd))) issues.push(`no "box-d · gpu_temp · ${amd}" in no-data [${nd.join(' | ')}]`);
      const figs = async (box) => Object.fromEntries((await card(page, box).locator('.figs .fig').evaluateAll((fs) =>
        fs.map((f) => [f.querySelector('span').textContent.trim(), f.querySelector('b').textContent.trim()]))));
      const a = await figs('box-a'); const b = await figs('box-b'); const cc = await figs('box-c');
      const xids = recent((k) => 2 + Math.floor(k / 15)).map(String);
      // box-a's XID finding is firing (the band says so): the card flags the figure with the finding's rise.
      const [axid, arise] = a.xid.split(' · +');
      if (!xids.includes(axid)) issues.push(`box-a XID ${a.xid}, sent ${xids.join('/')}`);
      const xidChip = chips.find((t) => t.startsWith('FIRING · box-a · XID +')) || '';
      const chipRise = (xidChip.match(/XID \+(\d+)/) || [])[1];
      if (!arise || arise !== chipRise) issues.push(`box-a XID figure "${a.xid}" not flagged with the band's +${chipRise}`);
      if (!(await card(page, 'box-a').locator('.figs .fig.bad', { hasText: 'xid' }).count())) issues.push('box-a XID figure not in the fault colour');
      if (await card(page, 'box-b').locator('.figs .fig.bad').count()) issues.push('box-b has no xid/oom finding, yet a figure is flagged');
      if (a['oom kills'] !== '0') issues.push(`box-a OOM ${a['oom kills']}, sent 0`);
      if (b.xid !== '0' || b['oom kills'] !== '3') issues.push(`box-b XID/OOM ${b.xid}/${b['oom kills']}, sent 0/3`);
      if (cc.xid !== '—' || cc['oom kills'] !== '1') issues.push(`box-c XID/OOM ${cc.xid}/${cc['oom kills']}, sent unread/1`);
      for (const box of ['box-a', 'box-b', 'box-c']) issues.push(...uxIssues(await ux(page, card(page, box).locator('.figs'), mobile)).map((x) => `${box} figs ${x}`));
      await shoot(band, `ux-2-${screen}.png`);
      await shoot(card(page, 'box-a').locator('.rings'), `ux-2-kernel-${screen}.png`);
      record(2, screen, issues);
    }
    // ── alert status line ──
    {
      const chip = page.locator('#lb-health .hl-failed');
      const issues = uxIssues(await ux(page, chip, mobile));
      const t = await chip.innerText();
      if (!/^ALERT webhook failed · HTTP 500 Internal Server Error · \d/.test(t)) issues.push(`alert line "${t}"`);
      await shoot(chip, `ux-3-${screen}.png`);
      record(3, screen, issues);
    }
    // ── RoCE per-port lines ──
    {
      const ra = sec(page, 'box-a', 'RoCE'); const rb = sec(page, 'box-b', 'RoCE');
      const issues = [...uxIssues(await ux(page, ra, mobile)), ...uxIssues(await ux(page, rb, mobile))];
      const ports = await ra.locator('.hx-port').allInnerTexts();
      const p0 = ports.find((t) => t.startsWith('mlx5_0/1')) || '';
      if (!p0.startsWith('mlx5_0/1 · ACTIVE · 200 Gb/s · MTU 9000')) issues.push(`box-a port line "${p0.split('\n')[0]}"`);
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
      if (!(pb.find((t) => t.startsWith('mlx5_0/1')) || '').includes('PFC off')) issues.push('box-b mlx5_0/1 not "PFC off"');
      if (!/RX DISC PHY —/.test(down)) issues.push(`box-b mlx5_1/1: ethtool refused, yet PHY counters shown: "${down}"`);
      if (!down) issues.push('box-b mlx5_1/1 not DOWN');
      const ld = (down.match(/LINK DN ([\d,]+)/) || [])[1];
      const lds = recent((k) => 2 + Math.floor(k / 10), 6).map(en);
      if (!lds.includes(ld)) issues.push(`box-b LINK DN ${ld}, sent ${[...new Set(lds)].join('/')}`);
      // The card agrees with the band: box-b's RoCE ERR finding names mlx5_1/1 link_downed, so the head counts the port
      // as rising and the LINK DN chip is flagged with the finding's window rise — even when this sample's rate is 0.
      if (!/1 down · 1 rising/i.test(await rb.locator('.hx-h').innerText())) issues.push(`box-b head "${await rb.locator('.hx-h').innerText()}" not "1 down · 1 rising"`);
      if (!/LINK DN [\d,]+ \+\d/.test(down)) issues.push(`box-b LINK DN not flagged with its rise: "${down}"`);
      if (!(await rb.locator('.hx-port.down .hx-c.up', { hasText: 'LINK DN' }).count())) issues.push('box-b LINK DN chip not in the fault colour');
      await shoot(ra, `ux-1-${screen}.png`);
      await shoot(rb, `ux-1-boxb-${screen}.png`);
      record(1, screen, issues);
    }
    // ── disks + models folders ──
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
      const box = await disks.boundingBox(); const box2 = await models.boundingBox();
      const sy = await page.evaluate(() => scrollY);
      await page.screenshot({ path: `${OUT}/ux-5-${screen}.png`, fullPage: true, clip: { x: box.x - 4, y: box.y + sy - 4,
        width: Math.max(box.width, box2.width) + 8, height: box2.y + box2.height - box.y + 8 } });
      record(5, screen, issues);
    }
    // ── Activity tile ──
    {
      const ev = page.locator('#events-section');
      await ev.scrollIntoViewIfNeeded();
      const issues = uxIssues(await ux(page, ev, mobile));
      const rows = await ev.locator('.ev-row').allInnerTexts();
      if (!rows.some((t) => t.includes('clock box-c: ok by admin'))) issues.push('seeded power.action event missing');
      if (!rows.some((t) => t.includes('clock box-d: ok by admin'))) issues.push('POSTed power.action event (up.sh) missing');
      const count = Number((await page.locator('#events-count').innerText()).replace(/\D/g, ''));
      const listed = await page.locator('#events-list .ev-row').count();
      if (count < expected.events + 1 || count !== listed) issues.push(`count ${count}, rows ${listed}, seeded ${expected.events}`);
      await ev.locator('button.ev-chip[data-g="power"]').click();
      await page.waitForFunction(() => [...document.querySelectorAll('#events-list .ev-row .ev-k')].every((k) => k.textContent === 'power.action'));
      const pw = await page.locator('#events-list .ev-row').count();
      if (pw !== 2) issues.push(`power filter shows ${pw} rows: 1 seeded + 1 POSTed`);
      await ev.locator('button.ev-chip[data-g=""]').click();
      await page.waitForFunction((n) => document.querySelectorAll('#events-list .ev-row').length >= n, listed);
      await shoot(ev, `ux-8-${screen}.png`);
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
      }
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
      const rows = await box.locator('.obs-row').evaluateAll((rs) => rs.map((r) => ({ ok: r.classList.contains('ok'), cap: r.querySelector('.obs-cap').textContent, text: r.innerText })));
      const title = await box.locator('.cn-title').innerText();
      const pass = rows.filter((r) => r.ok).length;
      if (!title.includes(`${pass}/${rows.length} pass`)) issues.push(`title "${title}" vs ${pass}/${rows.length}`);
      if (!rows.some((r) => r.ok && /telemetry/.test(r.cap))) issues.push('no passing telemetry row (the sender is up)');
      if (!rows.some((r) => !r.ok && /engine|server|backend/.test(r.cap))) issues.push('no failing engine row (box-c engine is down)');
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
      const page_issues = [...(sw > iw ? [`page overflows ${sw} > ${iw}`] : []), ...fails.map((e) => `console: ${e}`)];
      for (const n of [1, 2, 3, 5, 7, 8, 9, 10]) {
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
        } else if (!/failed/.test(t) || !/no token logprobs/.test(t)) issues.push(`failed run row "${t}"`);
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
