#!/bin/bash
# Apply the power snippet (deploy/nginx-edge-power.conf, your filled-in copy of its .example) to the edge's gateway
# vhost. Run ON the edge as root, with the snippet and the gate include already copied to /root/fleet-power/:
#   /root/fleet-power/nginx-edge-power.conf    (this repo's deploy/nginx-edge-power.conf)
#   /root/fleet-power/fleet-power-gate.conf    (proxy_set_header X-Power-Gate "<secret>";)
#   VHOST=/etc/nginx/sites-available/<the gateway's hostname> bash edge-apply.sh
set -euo pipefail
V=${VHOST:?VHOST is not set — the gateway vhost file under /etc/nginx/sites-available on this edge}
SNIP=/root/fleet-power/nginx-edge-power.conf
GATE_SRC=/root/fleet-power/fleet-power-gate.conf
TS=$(date -u +%Y%m%dT%H%M%SZ)
BAK=/etc/nginx/backups/$(basename "$V").bak-$TS

grep -q 'location /power/' "$V" && { echo "already applied"; exit 0; }
[ "$(grep -c '^    location / {$' "$V")" = 1 ] || { echo "expected exactly one '    location / {'"; exit 1; }
grep -qE '^proxy_set_header X-Power-Gate "[A-Za-z0-9_-]{32,}";$' "$GATE_SRC" || { echo "bad gate file"; exit 1; }

block_of_root() { sed -n '/^    location \/ {$/,/^    }$/p' "$1"; }
BEFORE_ROOT=$(block_of_root "$V" | sha256sum)
BEFORE_AUTH=$(grep -n 'auth_basic_user_file' "$V")

install -d -m 700 /etc/nginx/backups
cp -a "$V" "$BAK"
install -m 600 -o root -g root "$GATE_SRC" /etc/nginx/fleet-power-gate.conf

python3 - "$V" "$SNIP" <<'PY'
import sys
v, snip = sys.argv[1], sys.argv[2]
text = open(v).read()
block = open(snip).read()
start = block.index("    # ── fleet power")        # the file's own header comments stay out of the vhost
block = block[start:]
marker = "\n    location / {\n"
assert text.count(marker) == 1
text = text.replace(marker, "\n" + block.rstrip("\n") + "\n\n    location / {\n", 1)
open(v, "w").write(text)
PY

if ! nginx -t; then
  echo "nginx -t FAILED: restoring $BAK"; cp -a "$BAK" "$V"; nginx -t; exit 1
fi
AFTER_ROOT=$(block_of_root "$V" | sha256sum)
[ "$BEFORE_ROOT" = "$AFTER_ROOT" ] || { echo "location / CHANGED: restoring"; cp -a "$BAK" "$V"; nginx -t; exit 1; }
systemctl reload nginx
echo "applied. backup: $BAK"
echo "location / sha256 before=after: $AFTER_ROOT"
echo "auth_basic_user_file lines before:"; echo "$BEFORE_AUTH"
echo "auth_basic_user_file lines after:";  grep -n 'auth_basic_user_file' "$V"
