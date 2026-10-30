#!/usr/bin/env python3
"""The power buttons as a real Chromium sees them on the UNCHANGED gateway panel, behind an nginx
front that mirrors the edge vhost (sub_filter + /power/), with the REAL vllm-power in DRY-RUN mode
and a fake Ubuntu box built by the REAL install/host-setup.sh. Prints PASS/FAIL per check,
writes PNGs to --out, exits 1 on any failure.

  docker run --rm --network fp-e2e -v "$PWD":/src:ro -v <out>:/out chat-audit:1.55.0 \
      python3 /src/tests/e2e/browser.py --base http://front:8080 --out /out
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import urllib.error
import urllib.request

from playwright.sync_api import sync_playwright

results: list[tuple[bool, str]] = []


def check(ok: bool, what: str, detail: str = ""):
    results.append((bool(ok), what))
    print(("PASS " if ok else "FAIL ") + what + (f"  [{detail}]" if detail and not ok else ""), flush=True)


def http(base, path, *, method="GET", body=None, headers=None, auth=True):
    h = dict(headers or {})
    if auth:
        h["Authorization"] = "Basic " + base64.b64encode(b"operator:e2e-pass").decode()
    req = urllib.request.Request(base + path, data=body, method=method, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://front:8080")
    ap.add_argument("--out", default="/out")
    a = ap.parse_args()
    base, out = a.base, a.out

    # ── the door, over HTTP through the nginx front ─────────────────────────────────────────
    check(http(base, "/", auth=False)[0] == 401, "panel without credentials -> 401")
    check(http(base, "/power/api/state", auth=False)[0] == 401, "/power/ without credentials -> 401")
    st, body = http(base, "/")
    check(st == 200 and body.count(b'<script src="/power/button.js" defer></script></body>') == 1,
          "panel HTML carries exactly one injected button.js before </body>")
    st, body = http(base, "/gateway.js")
    check(st == 200 and b"/power/button.js" not in body, "other assets pass through without injection")
    st, body = http(base, "/lb/status")
    check(st == 200 and b"button.js" not in body, "/lb/status (JSON) untouched")
    st, _ = http(base, "/power/api/confirm", method="POST", body=b'{"box":"box-b","action":"poweroff"}',
                 headers={"Content-Type": "text/plain", "Origin": base})
    check(st == 415, "confirm with text/plain through nginx -> 415", str(st))
    st, _ = http(base, "/power/api/confirm", method="POST", body=b'{"box":"box-b","action":"poweroff"}',
                 headers={"Content-Type": "application/json", "Origin": "https://evil.example"})
    check(st == 403, "confirm from another Origin through nginx -> 403", str(st))
    st, _ = http(base, "/power/api/confirm", method="POST", body=b'{"box":"edge","action":"poweroff"}',
                 headers={"Content-Type": "application/json", "Origin": base})
    check(st == 404, "confirm for an unknown box -> 404", str(st))
    st, _ = http(base, "/power/api/confirm", method="POST", body=b'{"box":"box-b","action":"poweroff"}',
                 headers={"Content-Type": "application/json", "Origin": base,
                          "X-Power-User": "mallory", "X-Power-Gate": "forged"})
    check(st == 200, "client-sent X-Power-User/X-Power-Gate are replaced by nginx's (audit must say operator)", str(st))
    st, _ = http("http://power:8016", "/power/api/state", auth=False)
    check(st == 403, "the service reached directly (no nginx gate secret) -> 403", str(st))

    with sync_playwright() as p:
        b = p.chromium.launch()
        ctx = b.new_context(viewport={"width": 1440, "height": 1000},
                            http_credentials={"username": "operator", "password": "e2e-pass"})
        page = ctx.new_page()
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(base + "/")
        page.wait_for_selector("#lb-backends article.card .pw-ctl", timeout=15000)
        page.wait_for_timeout(1500)

        def ctl_boxes():
            return page.eval_on_selector_all("#lb-backends .pw-ctl", "els => els.map(e => e.dataset.box)")

        # Which machine cards the panel actually draws is the LB's business, not this service's:
        # it draws one per backend, and box-c stopped being a backend when its ollama model was
        # removed (2026-09-25). So expect a control on every machine card that IS there, and say
        # out loud when box-c is not among them — its ⏻ is then unreachable from the panel.
        names = page.eval_on_selector_all(
            "#lb-backends article.card .box-name",
            "els => els.map(e => ((e.firstChild && e.firstChild.nodeValue) || e.textContent).trim())")
        machines = [n for n in names if n in ("box-a", "box-b", "box-c")]
        boxes = ctl_boxes()
        check("box-a" in machines and "box-b" in machines,      # else the check below is vacuous
              "the panel draws both GB10 box machine cards", str(machines))
        check(sorted(boxes) == sorted(machines), "a ⏻ control on every machine card the panel draws",
              f"controls={boxes} cards={machines}")
        if "box-c" not in machines:
            print("NOTE box-c has no machine card in #lb-backends (the LB lists box-a, box-b and "
                  "the cluster as backends): its ⏻ cannot be reached or exercised from the panel.",
                  flush=True)
        st, body = http(base, "/power/api/state")
        check(st == 200 and b'"box-c"' in body, "the service still offers box-c in /api/state", str(st))
        cluster_ctl = page.evaluate("""() => [...document.querySelectorAll('#lb-backends article.card')]
            .filter(c => (c.querySelector('.box-name')||{}).textContent?.startsWith('box-a + box-b'))
            .map(c => !!c.querySelector('.pw-ctl'))""")
        check(cluster_ctl == [False], "no control on the cluster card", str(cluster_ctl))
        phases = page.evaluate("""() => Object.fromEntries([...document.querySelectorAll('#lb-backends .pw-ctl')]
            .map(e => [e.dataset.box, e.querySelector('.pw-phase').textContent]))""")
        expect = {"box-a": "off", "box-b": "on", "box-c": "on"}
        check(phases == {m: expect[m] for m in machines}, "live phases shown", str(phases))
        page.wait_for_timeout(7000)                       # two of the panel's 3 s rebuilds
        check(sorted(ctl_boxes()) == sorted(machines), "controls survive the panel's card rebuilds")
        page.screenshot(path=f"{out}/panel-desktop.png", full_page=True)
        page.locator('.pw-ctl[data-box="box-b"]').screenshot(path=f"{out}/control-box-b.png")

        # ── the press-and-hold: a mouse held down on the button for `ms`, then released ─────────
        def fill():
            return page.evaluate("""() => { const f = document.querySelector('dialog.pw-dialog .pw-fill');
                return f ? new DOMMatrix(getComputedStyle(f).transform).a : null; }""")

        def hold_for(loc, ms, mid=None):
            bb = loc.bounding_box()
            page.mouse.move(bb["x"] + bb["width"] / 2, bb["y"] + bb["height"] / 2)
            page.mouse.down()
            if mid:
                page.wait_for_timeout(2500)
                mid()
                page.wait_for_timeout(ms - 2500)
            else:
                page.wait_for_timeout(ms)
            page.mouse.up()

        # ── box-b: power off, dry run, under a running cluster ───────────────────────────────
        page.click('.pw-ctl[data-box="box-b"] .pw-btn')
        dlg = page.locator("dialog.pw-dialog")
        dlg.locator("button.pw-hold").wait_for(timeout=10000)
        text = dlg.inner_text()
        check("Power off box-b?" in text, "dialog names the box and action")
        check("DRY RUN" in text, "dialog says DRY RUN")
        check("chat falls back" in text and "needs Start" in text, "dialog states the GB10 box consequences")
        # Whatever the gateway says right now — the live cluster is "running" between benchmark
        # configurations and "starting"/"off" while one is being brought up. The dialog must show
        # the state the gateway reports, not a state this test wishes for.
        cl_state = json.loads(http(base, "/lb/status")[1])["cluster"]["state"]
        check(f"Model cluster: {cl_state}" in text, "dialog shows the cluster state the gateway reports",
              f"expected '{cl_state}' in: " + text.replace("\n", " | "))
        check("null" not in text and "undefined" not in text, "no stray null/undefined in the dialog", text)
        check(dlg.locator("input[type=text]").count() == 0, "no box name to type any more")
        check("Press and hold the button for 5 s" in text, "the dialog says how to confirm: a 5 s hold")
        hold = dlg.locator("button.pw-hold")
        check(hold.is_disabled(), "the hold button is disabled until the cluster-drop box is ticked")
        page.screenshot(path=f"{out}/dialog-box-b-poweroff.png")
        dlg.locator("input[type=checkbox]").check()
        check(hold.is_enabled(), "enabled once ticked")
        hold.click()
        page.wait_for_timeout(800)
        check(dlg.locator(".pw-out").is_hidden() and fill() == 0, "a plain click runs nothing")
        hold_for(hold, 2000)
        page.wait_for_timeout(300)
        check(dlg.locator(".pw-out").is_hidden(), "letting go after 2 s runs nothing")
        check(fill() == 0 and "too early" in hold.inner_text(), "... and empties the bar and says so",
              f"fill={fill()} label={hold.inner_text()}")
        # A script that skips the hold and just CLAIMS it held for 5 s: the service itself refuses
        # a confirmation younger than the hold (same page, same origin, same login as the button).
        r = page.evaluate("""async () => {
            const post = (p, b) => fetch('/power/api/' + p, { method: 'POST', credentials: 'same-origin',
                headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(b) })
              .then(async (r) => [r.status, await r.json()]);
            const [edge, c] = await post('confirm', { box: 'box-b', action: 'poweroff' });
            const [s2, e] = await post('execute', { token: c.token, held_ms: 5000, ack_cluster: true });
            return [edge, s2, e.error ? e.error.type : null];
        }""")
        check(r == [200, 400, "too_soon"], "a script claiming held_ms=5000 at once is refused: too_soon", str(r))
        mid: dict = {}

        def at_half():
            mid["fill"] = fill()
            mid["label"] = hold.inner_text()
            page.screenshot(path=f"{out}/dialog-box-b-holding.png")
        hold_for(hold, 5500, mid=at_half)
        check(0.3 < (mid.get("fill") or 0) < 0.75 and "Keep holding" in mid.get("label", ""),
              "halfway through the hold the bar is about half full and counts down", str(mid))
        dlg.locator(".pw-out").wait_for(state="visible", timeout=30000)
        res = dlg.locator(".pw-out").inner_text()
        check(res.startswith("DRY RUN — sent") and "/usr/bin/systemctl poweroff" in res,
              "dry-run poweroff went hold → service → ssh → forced command", res)
        page.screenshot(path=f"{out}/dialog-box-b-dryrun-result.png")
        dlg.locator("button", has_text="Close").click()

        # ── box-c: its own consequences, no power-on ────────────────────────────────────────
        if "box-c" in machines:
            page.click('.pw-ctl[data-box="box-c"] .pw-btn')
            dlg.locator("button.pw-hold").wait_for(timeout=10000)
            text = dlg.inner_text()
            check("smart plug" in text and "100+" in text, "box-c dialog: services go down, power-on is its smart plug")
            check(dlg.locator("input[type=checkbox]").count() == 0, "box-c needs no cluster tick")
            page.screenshot(path=f"{out}/dialog-box-c-poweroff.png")
            hold_for(dlg.locator("button.pw-hold"), 5500)
            dlg.locator(".pw-out").wait_for(state="visible", timeout=30000)
            res = dlg.locator(".pw-out").inner_text()
            check(res.startswith("DRY RUN — sent"), "box-c dry run sent", res)
            dlg.locator("button", has_text="Close").click()
        else:
            print("SKIP the box-c dialog: no box-c card on the page to open it from.", flush=True)

        # ── box-a (off): power on via Wake-on-LAN; in the test net the cable route is absent ─────
        # (held with the KEYBOARD this time: Space down for 5.5 s on the focused hold button)
        page.click('.pw-ctl[data-box="box-a"] .pw-btn')
        dlg.locator("button.pw-hold").wait_for(timeout=10000)
        text = dlg.inner_text()
        check("Power on box-a?" in text and "Wake-on-LAN" in text, "an off GB10 box offers power-on (WoL)")
        dlg.locator("button.pw-hold").focus()
        page.keyboard.down(" ")
        page.wait_for_timeout(5500)
        page.keyboard.up(" ")
        dlg.locator(".pw-out").wait_for(state="visible", timeout=30000)
        res = dlg.locator(".pw-out").inner_text()
        check(res.startswith("failed") and "eth-box-a" in res,
              "no cable route here -> honest failure, nothing sent", res)
        page.screenshot(path=f"{out}/dialog-box-a-poweron-no-route.png")
        dlg.locator("button", has_text="Close").click()

        # ── the GPU clock pill: on the GB10 boxes only, and a DRY-RUN round trip on box-b ────────
        clks = page.eval_on_selector_all("#lb-backends .pw-ctl",
                                         "els => els.map(e => [e.dataset.box, !!e.querySelector('.pw-clk')])")
        check(sorted(clks) == sorted([[m, m != "box-c"] for m in machines]),
              "a clock pill on each GB10 box card the panel draws, none on box-c", str(clks))
        disabled = page.eval_on_selector_all("#lb-backends .pw-clk",
                                             "els => els.map(e => [e.closest('.pw-ctl').dataset.box, e.disabled])")
        check(sorted(disabled) == [["box-a", True], ["box-b", False]],
              "the pill is disabled on a box that is off", str(disabled))
        label = page.inner_text('.pw-ctl[data-box="box-b"] .pw-clk')
        check(label == "GPU clk", "with nothing applied the pill does not claim a state", label)

        page.click('.pw-ctl[data-box="box-b"] .pw-clk')
        dlg.locator(".pw-opt button").first.wait_for(timeout=10000)
        text = dlg.inner_text()
        check("GPU clock — box-b" in text, "the clock dialog names the box", text[:120])
        check("DRY RUN" in text, "the clock dialog says DRY RUN")
        check("has set no clock on box-b" in text, "it states that nothing was applied yet", text[:400])
        check("no lock state" in text, "it does not pretend to read the cap back from the driver")
        check("2200 MHz" in text and "2300 MHz" in text and "3003" in text,
              "it names every cap and the GPU's own maximum", text[:600])
        check("keeps serving" in text, "it says the cluster keeps serving")
        check(dlg.locator("input[type=text]").count() == 0, "no box name to type for a clock change")
        titles = dlg.locator(".pw-opt button").all_inner_texts()
        check(titles == ["Limit the GPU to 2200 MHz", "Limit the GPU to 2300 MHz",
                         "Release the clock limit"],
              "every action offered, named by the service", str(titles))
        check("null" not in text and "undefined" not in text, "no stray null/undefined", text)
        page.screenshot(path=f"{out}/dialog-box-b-clock.png")
        dlg.locator(".pw-opt button").first.click()
        dlg.locator(".pw-out").wait_for(state="visible", timeout=30000)
        res = dlg.locator(".pw-out").inner_text()
        check(res.startswith("DRY RUN — sent") and "/usr/bin/nvidia-smi -lgc 300,2200" in res,
              "dry-run cap went button → service → ssh → the key's own forced command", res)
        page.screenshot(path=f"{out}/dialog-box-b-clock-dryrun-result.png")
        dlg.locator("button", has_text="Close").click()
        page.wait_for_timeout(6000)
        check(page.inner_text('.pw-ctl[data-box="box-b"] .pw-clk') == "GPU clk",
              "a DRY RUN leaves the pill claiming nothing was applied")
        page.screenshot(path=f"{out}/panel-desktop-clock.png", full_page=True)

        # ── a phone ───────────────────────────────────────────────────────────────────────────
        page.set_viewport_size({"width": 390, "height": 844})
        page.wait_for_timeout(3500)
        page.locator("#lb-backends").screenshot(path=f"{out}/panel-phone-cards.png")
        overflow = page.evaluate("() => document.documentElement.scrollWidth - document.documentElement.clientWidth")
        check(overflow <= 0, "no horizontal scroll at 390 px", str(overflow))
        page.click('.pw-ctl[data-box="box-b"] .pw-btn')
        dlg.locator("button.pw-hold").wait_for(timeout=10000)
        page.screenshot(path=f"{out}/dialog-phone.png")
        overflow = page.evaluate("() => document.documentElement.scrollWidth - document.documentElement.clientWidth")
        check(overflow <= 0, "the power dialog with its hold button does not overflow 390 px", str(overflow))
        dlg.locator("button", has_text="Cancel").click()
        page.click('.pw-ctl[data-box="box-b"] .pw-clk')
        dlg.locator(".pw-opt button").first.wait_for(timeout=10000)
        page.screenshot(path=f"{out}/dialog-phone-clock.png")
        dh = page.evaluate("() => document.querySelector('dialog.pw-dialog').getBoundingClientRect().height")
        check(dh <= 844, "the clock dialog fits a phone screen (nothing clipped off the bottom)", str(dh))
        overflow = page.evaluate("() => document.documentElement.scrollWidth - document.documentElement.clientWidth")
        check(overflow <= 0, "the clock dialog does not overflow 390 px", str(overflow))
        dlg.locator("button", has_text="Close").click()
        check(not errors, "no page errors", "; ".join(errors))
        b.close()

    failed = [w for ok, w in results if not ok]
    json.dump({"passed": len(results) - len(failed), "failed": failed}, open(f"{out}/report.json", "w"), indent=1)
    print(f"\n{len(results) - len(failed)}/{len(results)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
