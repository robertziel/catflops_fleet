# vllm-power: power and GPU-clock buttons on the vLLM Gateway page

A ⏻ control on each machine card of https://vllm.example.com/ (box-a, box-b, box-c),
and a clock pill next to it on the GB10 boxes:

- **power off**: `systemctl poweroff` on the box, through a forced-command ssh key.
- **power on**: Wake-on-LAN, for box-a and box-b only. The packet goes from box-c down each
  GB10 box's direct cable. It stays **untested** until the first real power-off.
- box-c has no power-on button. It runs this service and the page, so when it is off nothing
  here exists. Power it on with its smart plug.
- **GPU clock** (box-a, box-b): *Limit the GPU to 2200 MHz* → `nvidia-smi -lgc 300,2200`,
  *Limit the GPU to 2300 MHz* → `nvidia-smi -lgc 300,2300` (added 2026-10-07), and *Release the
  clock limit* → `nvidia-smi -rgc`, each through its **own** forced-command key — one key per cap,
  so no key can set a frequency other than its own and no number a client sends reaches root. This
  cap used to be `nvidia-clock-lock.service` on each box (`install/remove-clock-lock.sh` took it
  off both GB10 boxes on 2026-09-30). A clock change drops no request: the cluster keeps serving at the
  capped speed. The driver reports no lock state, so the panel shows which cap it last applied,
  never a read-back.
- **Configured clock** (2026-10-23): each GPU box's `default_gpu_clock_mhz` in `POWER_BOXES` (one
  of the caps above; absent = released) is applied by the service itself, through the same keys.
  A lock dies with a reboot and nothing on the box re-applies it, so the service probes each GPU
  box's sshd every 10 s (`CLOCK_RECONCILE_S`): a box seen down, or not yet handled since the
  service started, gets its configured clock as soon as sshd answers, retried every pass until
  `nvidia-smi` succeeds. A **manual** clock click holds only while the box stays up: the next time
  the box is seen down (reboot, power cut) or the service restarts, the config wins again — so a
  reboot can never leave a capped box uncapped. Each apply/release is an audit line with
  `event: "reconcile"`, `user: "config"`. box-a (GIGABYTE GB10) is configured to 2200: uncapped
  (2200–2450 MHz, 14→112 W transients) it power-cut itself twice on 2026-10-23. Adding a cap = one more key + forced command + sudoers argv on the box
  (`CLOCK_CAPS_MHZ` in `fleet_power/app.py`, `CLOCK_MAXn` in `install/host-setup.sh`).

> box-c currently has **no machine card** on the panel — vllm-lb draws one card per backend and
> box-c stopped being one when its ollama model was removed (2026-09-25). Its ⏻ therefore cannot
> be reached from the page, though the service still offers it at `/power/api/state`.

The gateway's own code (vllm-lb, `lb/`) is not changed. The edge's nginx injects one
`<script src="/power/button.js">` into the panel page, and only into the page at `/`.

## How a click travels

```
browser ──https──▶ edge nginx  vllm.example.com
                    ├─ location = /        your proxy's sign-in (auth_request to it) → frps → vllm-lb
                    │                       + sub_filter: one <script src="/power/button.js"> before </body>
                    ├─ location /power/    the session (X-Power-User = its email) or, for a Fleet app on its
                    │                       listed paths, Basic .htpasswd_gateway (X-Power-User=$remote_user); sets
                    │                       X-Power-Gate=<secret>, drops Authorization → frps
                    └─ location /          the same two gates (API, SSE, assets) → frps → vllm-lb
frps (edge, 127.0.0.1:8082) ── routes Host=vllm + path /power/ ──▶ frpc "vllm-power" (container vllm-power-frpc)
                                                              ──▶ vllm-power 127.0.0.1:8016 on box-c
vllm-power ── ssh -i fleet-power key ──▶ fleet-power@192.0.2.2 / 192.0.2.12 / 127.0.0.1
             forced command: sudo -n /usr/bin/systemctl poweroff  (sudoers: that command only)
vllm-power ── ssh -i fleet-clock-limit / -limit-2300 / -release key ──▶ fleet-power@192.0.2.2 / 192.0.2.12
             forced command: sudo -n /usr/bin/nvidia-smi -lgc 300,2200  |  … 300,2300  |  … -rgc
             The MHz live in the forced command, one key per cap: no number a client sends can
             reach root, and no key reaches another cap.
vllm-power ── UDP broadcast, magic packet ──▶ 192.0.2.3 / 192.0.2.13 port 9 (the direct cables)
```

The API (all under `/power/`):

| | |
|---|---|
| `GET /power/button.js` | the control script (`application/javascript`, `nosniff`) |
| `GET /power/api/state` | each box: `on` / `off` / `powering_off` / `waking` / `wake_timeout` (a TCP probe of its sshd) |
| `POST /power/api/confirm` `{box, action}` | a single-use token, bound to user + box + action, that lasts 120 s. Also returns the consequences, the cluster state, `needs_cluster_ack`, and `needs_hold` / `hold_ms` (5000 for a power action, `null` for a clock one) |
| `POST /power/api/execute` `{token, held_ms, ack_cluster}` | runs the action. **For a power action** the button must have been pressed and held for 5 s: `held_ms` ≥ 5000 (the client's own measurement), AND the token must be at least 5 s old when execute arrives — the server's own check, so a script cannot skip the wait. Either refusal (400 `hold_too_short` / `too_soon`) leaves the token usable. `ack_cluster: true` is required to power off a GB10 box while the cluster runs, or while its state is unknown |

The actions (`action` above), each with its own key and forced command on the box:

| action | box | 5 s hold | what runs there |
|---|---|---|---|
| `poweroff` | all | yes | `systemctl poweroff` |
| `poweron` | box-a, box-b | yes | Wake-on-LAN from box-c; no ssh at all |
| `clock_limit_2200` | box-a, box-b | no | `nvidia-smi -lgc 300,2200` |
| `clock_limit_2300` | box-a, box-b | no | `nvidia-smi -lgc 300,2300` |
| `clock_release` | box-a, box-b | no | `nvidia-smi -rgc` |

A power action is confirmed by **pressing and holding** its button for 5 s: a bar fills across the
button, letting go earlier cancels and empties it, and only a completed hold sends `execute`. (It
replaced typing the box name, 2026-09-30.) The token is fetched when the dialog opens — before the
hold can start — so an honest hold always clears the server's 5 s token-age check. A clock change
needs no hold: it is reversible, applies in place and drops no request — the confirm token, the
Origin rules and a 10 s per-box window are its guards. `/power/api/state` carries, per GB10 box,
`clock.applied` and `clock.applied_mhz` (which cap this panel last set, `null` after a restart),
`clock.limits_mhz` (every cap offered), `clock.configured_mhz` (`default_gpu_clock_mhz`, `null` =
released) and each action's title, `limit_mhz` and consequence.

## Threat model and trust boundaries

**Assets:** the three boxes' uptime. box-c also carries the gateway and 100+ other
services.

**Adversaries considered:** someone on the internet without the basic-auth password; a
malicious web page open in the operator's browser (CSRF); a container or process on box-c that
is not the operator's; someone who steals the service's ssh key; a replayed or duplicated click.

| # | Boundary | What is trusted across it | Enforced by |
|---|---|---|---|
| B1 | internet → edge nginx | nothing without your sign-in proxy's session for `vllm` (browsers) or, on the Fleet apps' listed method+path pairs only, the `.htpasswd_gateway` password | `auth_request` to your sign-in service on every gated location; Basic only via the internal `/_machine/` locations (your edge vhost for the gateway). Without either, every path answers 401/302 (verified live 2026-10-07) |
| B2 | edge nginx → frps → frpc → service | the `X-Power-User` and `X-Power-Gate` headers | nginx **sets** both, replacing any client-sent value (e2e: a forged `X-Power-User: mallory` never reaches the audit log). frps `vhost_http_port` 8082 is bound to 127.0.0.1 on the edge, and the frp token guards registration |
| B3 | other processes on box-c → service at 127.0.0.1:8016 | nothing | the service refuses every request without the gate secret (a 32-byte random value, checked with `hmac.compare_digest`). An empty secret file makes it refuse everything. Only host-network containers can reach 127.0.0.1 at all |
| B4 | browser (other origins) → service | nothing | POST only; `Content-Type: application/json` or 415 (no CORS-simple form post can send it); `Origin` must be `https://vllm.example.com`; `Sec-Fetch-Site`, when present, must be `same-origin`; body ≤ 4 KiB. These are CSRF rules for *browsers*: a native client holding the password (the Fleet iPhone app) sets `Origin` itself and sends no `Sec-Fetch-Site`, which is what any password holder could do anyway (B1, residual risks) |
| B5 | one hold → one action | the confirm token; the hold | a token lasts 120 s, is single-use and is bound to user + box + action. A power action needs a 5 s hold: `held_ms` ≥ 5000 (reported by the client, so only an attestation) AND a token at least 5 s old (measured by the server, from issue to execute — the part a client cannot shorten). A too-short hold does not burn the token. Rate limits: 10 confirms per minute, 1 execute per box per minute |
| B6 | service → box (ssh) | the box's host key; the service's key | `StrictHostKeyChecking=yes` with a pinned `known_hosts` (fingerprints matched against `~/.ssh/known_hosts`); `-F none`, `IdentitiesOnly`, no agent, no forwarding, `BatchMode` |
| B7 | ssh key → root on the box | **only** `systemctl poweroff`, and on a GB10 box `nvidia-smi -lgc 300,2200` / `-lgc 300,2300` / `-rgc` | `authorized_keys`: one line per action, `restrict,from="<box-c's cable address>",command="sudo -n <that one command>"` — the action IS the key, so a client chooses neither the command nor the frequency, and one cap's key cannot set the other's. `sudoers.d/fleet-power` lists exactly those argv (the comma escaped: `-lgc 300\,2200`). The `fleet-power` user has no password (`*`). Proven on a fake box built by the real `host-setup.sh`: a hostile command still runs only the forced one; `-W` forwarding, a pty, `reboot`, `poweroff --force`, `/bin/sh`, `nvidia-smi -lgc 300,3000`, `-lgc 300,2250`, `nvidia-smi -i 0 -lgc 300,2200`, `nvidia-smi -r`, a bare `nvidia-smi`, a different source address and another key are all refused |
| B8 | the service container | its own files only | uid 1000, `read_only`, `cap_drop: ALL`, `no-new-privileges`. WoL is an unprivileged UDP broadcast. The keys are mounted read-only from `~/.config/fleet-power/keys` (700/600, not in git) |
| B9 | the service ← vllm-lb | the cluster state (read-only `GET /lb/status`) | an unreadable state counts as "running" (the tick is required). The service never calls the LB's control API |

**Residual risks (accepted, stated):**
- Anyone with the gateway's basic-auth password can power the boxes off, just as they can already
  stop the cluster. The 5 s hold and the token defend against accidents and CSRF, not against
  the password holder: a script with the password can confirm, wait 5 s and execute with
  `held_ms: 5000`. The server's token-age check only guarantees that *someone waited* 5 s between
  asking and acting — which is exactly what the typed name guaranteed too (a slip cannot power a
  box off), no more.
- `user` is in the `docker` group on all three boxes, which is root-equivalent. A process
  running as user on box-c can read the keys, but it could already do far more.
- A stolen power key can power off a box, and only from box-c's cable address. The dry-run key
  can only ask sudo a question.
- The Fleet iPhone app (`FleetStats`, "Power" rows in its box sections) is a second client of the
  same API with the same basic-auth login baked into its build (`Secrets.xcconfig`). A lost,
  unlocked phone is a password holder. It shows the same consequences and the same 5 s hold; a
  running cluster is acknowledged by the hold itself after a warning shown above the button.
- The 401 page nginx returns for `/` without credentials also gets the injected `<script>` tag,
  because sub_filter doesn't look at the status. The script's own request is then refused with a
  401. The page is a browser page, not an API; `/v1/*`, `/lb/*` and every other path are
  byte-identical (verified).

## Configuration (power/.env — see power/.env.example)

Everything that is yours lives in `power/.env` (gitignored), and every variable refuses when unset:

| variable | unset means | what it is |
|---|---|---|
| `POWER_ORIGINS` | every request is refused | the page's public origin (the CSRF check) |
| `POWER_ALLOWED_HOSTS` | the service does not start, naming it | the Host names it answers to beyond its loopback names: the name the edge's nginx presents (no port behind `proxy_set_header Host $host`), or `loopback`; any other Host is 421 (`fleet_power/host_guard.py`, the DNS-rebinding guard) |
| `POWER_BOXES` | no box may be acted on; a malformed value stops the service at boot | JSON array: the machines, their cable addresses, Wake-on-LAN, the panel's wording, and each GPU box's `default_gpu_clock_mhz` (2200 / 2300, or absent = released; any other value, or one on a box without `gpu_clock: true`, stops the service at boot) |
| `POWER_DRY_RUN` | `1` (dry run) | `0` = live: the forced command really powers off |

The public inventory (`POWER_BOXES`) is read by `fleet_power/boxes.py`; the keys the boxes trust are read
from the directory `install/host-setup.sh` is given (`FLEET_POWER_KEYS`), never from the repo.

## Install / uninstall

1. **Keys and secret** (on box-c, no privilege; done 2026-09-23, the clock keys 2026-09-30):
   `~/.config/fleet-power/{keys/fleet-power,keys/fleet-power-dryrun,keys/known_hosts,gate_secret,audit/}`
   plus `keys/fleet-clock-{limit,limit-2300,release,dryrun}` (the 2300 one 2026-10-07) —
   `ssh-keygen -t ed25519 -N "" -f <name>`. The
   compose file mounts the whole `keys` directory, so a new key needs no compose change.
2. **Each box** (the operator runs this; sudo asks for a password). The script ships no key and no
   address: it installs the PUBLIC keys in the directory you name (`FLEET_POWER_KEYS`), pinned to the
   source address in `from-<box>` (the host's end of that box's cable: one IPv4 address or CIDR). Add
   `from-box-a` and `from-box-b` to `~/.config/fleet-power/keys` first. From `~/vllm-docker` on box-c:
   ```
   FLEET_POWER_PRINT=1 FLEET_POWER_KEYS=~/.config/fleet-power/keys sh power/install/host-setup.sh box-a --gpu  # what it would install
   sudo FLEET_POWER_KEYS=~/.config/fleet-power/keys sh power/install/host-setup.sh box-c --local   # the box that runs vllm-power
   ```
   On a GB10 box, send it only the public keys and that box's source file (never a private key):
   ```
   ssh box-a-fast 'mkdir -p /tmp/fp-keys' && scp ~/.config/fleet-power/keys/*.pub ~/.config/fleet-power/keys/from-box-a box-a-fast:/tmp/fp-keys/
   scp power/install/host-setup.sh box-a-fast:/tmp/fleet-power-setup.sh && ssh -t box-a-fast 'sudo FLEET_POWER_KEYS=/tmp/fp-keys sh /tmp/fleet-power-setup.sh box-a --gpu; rm -rf /tmp/fleet-power-setup.sh /tmp/fp-keys'
   ```
   (box-b the same, with `from-box-b` and `box-b-fast`.)
   To undo: `sudo sh power/install/host-uninstall.sh` on each box.

   On a box whose sudo wants a password, the `docker` group is the way in (it is root-equivalent —
   the same path `deploy/install-nfs-healer.sh` uses):
   ```
   cat power/install/host-setup.sh | ssh box-a 'cat > /tmp/.fp.sh && docker run --rm --privileged \
     --pid=host --net=host alpine:latest nsenter -t 1 -m -u -i -n -p -- sh /tmp/.fp.sh box-a --gpu; rm -f /tmp/.fp.sh'
   ```
2b. **Take the boot-time clock cap off a GB10 box** (once, 2026-09-30): the same way, with
   `install/remove-clock-lock.sh`. It disables and deletes `nvidia-clock-lock.service` and
   deliberately does **not** `systemctl stop` it — that unit's `ExecStop` runs `nvidia-smi -rgc`,
   which would change the clocks under whatever the box is running.
3. **Service + frp client** (on box-c): `cd power && docker compose up -d --build`. This starts in
   DRY RUN. Going live needs the operator's go: `POWER_DRY_RUN=0 docker compose up -d power`.
   To undo: `docker compose down`.
4. **edge nginx**: copy `deploy/nginx-edge-power.example.conf`, `install/edge-apply.sh` and a
   `fleet-power-gate.conf` (`proxy_set_header X-Power-Gate "<gate_secret>";`) to `/root/fleet-power/`,
   then `bash /root/fleet-power/edge-apply.sh`. The script backs up the vhost, inserts the block,
   runs `nginx -t`, restores the backup if the test fails, checks that `location /` is unchanged,
   and reloads. To undo: copy the printed backup back, then `nginx -t && systemctl reload nginx`.

The audit log is `~/.config/fleet-power/audit/audit.jsonl`: one JSON line per confirm or execute,
including refusals, and one per automatic clock apply/release (`event: "reconcile"`). Each line has `ts` (UTC), `user` (the signed-in email, or the basic-auth user for a Fleet app), `ip`, `box`, `action`,
`result`, `rc`, `output`, `cluster` and `dry_run`. Tokens are never logged.

## Tests

```
docker build -t vllm-power:test power
docker run --rm vllm-power:test pytest                                  # 131 tests (unit + HTTP wrapper)
docker run --rm -v "$PWD/power":/src:ro vllm-power:test sh /src/tests/mutations.sh   # each guard broken → its test fails
cd power && bash tests/e2e/fakebox_proof.sh                             # authorized_keys + sudoers semantics
cd power && bash tests/e2e/run_e2e.sh <out-dir>                         # 52 checks, Chromium on the real panel files, DRY RUN
bash power/install/edge-mirror-test.sh <old-vhost> <new-vhost> <gate-include>   # on the edge: OLD vs NEW nginx, real upstream
```
