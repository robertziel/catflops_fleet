#!/bin/bash
# The dry-run end-to-end proof, on an isolated (--internal) docker network; nothing here touches
# vllm-lb (its static files are mounted read-only, /lb/status is one captured GET) or any real box.
#   button (Chromium) → nginx front (the edge's locations) → vllm-power (DRY RUN) → ssh → fake box
#   (built by the REAL host-setup.sh) → forced command `sudo -n -l …` → audit log.
# Run from power/:  bash tests/e2e/run_e2e.sh <out-dir>
set -euo pipefail
OUT="$(realpath "${1:?out dir}")"; mkdir -p "$OUT"
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
POWER="$REPO/power"
KEYS="${FLEET_POWER_KEYS:-$HOME/.config/fleet-power/keys}"
W="$(mktemp -d)"; chmod 755 "$W"
NET=fp-e2e
cleanup() { docker rm -f fp-box fp-panel fp-power fp-front >/dev/null 2>&1 || true; }
trap cleanup EXIT
cleanup

docker build -q -t vllm-power:test "$POWER" >/dev/null
docker build -q -t fleet-power-fakebox:test -f "$POWER/tests/e2e/fakebox/Dockerfile" "$POWER" >/dev/null
docker network inspect $NET >/dev/null 2>&1 || docker network create --internal --subnet 10.231.99.0/24 $NET >/dev/null

curl -fsS http://127.0.0.1:8010/lb/status > "$W/status.json"         # one read-only GET
echo 'proxy_set_header X-Power-Gate "e2e-gate-secret-0123456789";' > "$W/gate.conf"
echo -n 'e2e-gate-secret-0123456789' > "$W/gate_secret"
python3 -c 'import hashlib,base64;print("operator:{SHA}"+base64.b64encode(hashlib.sha1(b"e2e-pass").digest()).decode())' > "$W/htpasswd"
mkdir -p "$W/audit"; chmod 777 "$W/audit"; chmod 644 "$W"/*.conf "$W/htpasswd" "$W/status.json"; chmod 644 "$W/gate_secret"

# The fake box installs the REAL host-setup.sh from PUBLIC keys only (copied here) and the test network's source.
mkdir -p "$W/fpkeys"; cp "$KEYS"/*.pub "$W/fpkeys/"; echo "10.231.99.0/24" > "$W/fpkeys/from-box-b"
chmod 755 "$W/fpkeys"; chmod 644 "$W/fpkeys"/*
docker run -d --name fp-box --network $NET --ip 10.231.99.10 -v "$W/fpkeys":/fp-keys:ro fleet-power-fakebox:test >/dev/null
docker run -d --name fp-panel --network $NET --network-alias panel -v "$REPO/lb/vllm_lb/static":/static:ro \
  -v "$W/status.json":/status.json:ro -v "$POWER/tests/e2e":/e2e:ro python:3.12-slim \
  python3 /e2e/panel_fixture.py /static /status.json >/dev/null
sleep 1
docker run --rm --network $NET vllm-power:test sh -c 'ssh-keyscan -t ed25519 10.231.99.10 2>/dev/null' > "$W/known_hosts"
chmod 644 "$W/known_hosts"
docker run -d --name fp-power --network $NET --network-alias power --user 1000:1000 --read-only \
  --cap-drop ALL --security-opt no-new-privileges --tmpfs /tmp \
  -v "$KEYS":/keys:ro -v "$W/known_hosts":/e2e/known_hosts:ro -v "$W/gate_secret":/run/secrets/gate:ro \
  -v "$W/audit":/audit -v "$POWER/tests/e2e/run_power.py":/app/run_power.py:ro \
  -e POWER_BIND=0.0.0.0:8016 -e POWER_ALLOWED_HOSTS=front -e POWER_ORIGINS=http://front:8080 -e POWER_DRY_RUN=1 \
  -e POWER_KNOWN_HOSTS=/e2e/known_hosts -e POWER_LB_STATUS=http://panel:8000/lb/status \
  -e E2E_HOSTS=box-a=10.231.99.12,box-b=10.231.99.10,box-c=10.231.99.10 \
  vllm-power:test python /app/run_power.py >/dev/null
docker run -d --name fp-front --network $NET --network-alias front \
  -v "$POWER/tests/e2e/nginx.conf":/etc/nginx/conf.d/default.conf:ro \
  -v "$W/htpasswd":/etc/nginx/htpasswd:ro -v "$W/gate.conf":/etc/nginx/fleet-power-gate.conf:ro nginx:alpine >/dev/null
sleep 2

set +e
docker run --rm --network $NET -v "$REPO":/src:ro -v "$OUT":/out chat-audit:1.55.0 \
  python3 /src/power/tests/e2e/browser.py --base http://front:8080 --out /out
rc=$?
set -e
echo; echo "## audit log written by the service:"; cat "$W/audit/audit.jsonl" | tee "$OUT/audit.jsonl"
if grep -q mallory "$W/audit/audit.jsonl"; then echo "FAIL a client-sent X-Power-User reached the audit log"; rc=1; else echo "PASS no client-sent user name in the audit log"; fi
echo; echo "## the fake box's stub systemctl (must be EMPTY: the dry-run key only asks sudo):"
docker exec fp-box sh -c 'cat /tmp/systemctl.log 2>/dev/null || echo "(no calls)"'
echo "## control POSTs the panel fixture received (must be none):"; docker logs fp-panel 2>&1 | grep -c PANEL-CONTROL-POST || true
docker logs fp-power > "$OUT/power.log" 2>&1
exit $rc
