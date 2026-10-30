#!/bin/bash
# The vhost change, proven against the REAL upstream (frps → vllm-lb) BEFORE it touches the live
# config: the edge's own nginx binary runs two throwaway instances on 127.0.0.1 — OLD = the live vhost
# as it is, NEW = the same file with deploy/nginx-edge-power.example.conf inserted — each with a temporary
# htpasswd (the live password is not available to the agent) and private log/temp paths. Then
# every response API clients see is compared OLD vs NEW. Run ON the edge as root:
#   EDGE_HOST=<the gateway's hostname> bash edge-mirror-test.sh <old-vhost> <new-vhost> <gate-include>
set -uo pipefail
OLD=$1; NEW=$2; GATE=$3
H=${EDGE_HOST:?EDGE_HOST is not set — the gateway's hostname, as the vhost's server_name states it}
D=/tmp/fpm; rm -rf $D; mkdir -p $D/{old,new}; chown -R www-data $D
PASS=$(head -c 18 /dev/urandom | base64 | tr -dc A-Za-z0-9)
printf 'mirror:{SHA}%s\n' "$(printf %s "$PASS" | openssl dgst -sha1 -binary | base64)" > $D/htpasswd
chmod 644 $D/htpasswd
prep() {  # prep <src> <name> <port>
  python3 - "$1" "$D/$2/vhost.conf" "$3" "$D" "$GATE" <<'PY'
import re, sys
src, dst, port, d, gate = sys.argv[1:]
t = open(src).read()
t = re.sub(r"server \{\n    listen 80;.*?\n\}\n", "", t, count=1, flags=re.S)   # drop the :80 redirect
t = t.replace("listen 443 ssl http2;", f"listen 127.0.0.1:{port} ssl;")
t = re.sub(r"auth_basic_user_file [^;]+;", f"auth_basic_user_file {d}/htpasswd;", t)
t = re.sub(r"access_log [^;]+;", f"access_log {d}/access-{port}.log;", t)
t = re.sub(r"error_log [^;]+;", f"error_log {d}/error-{port}.log notice;", t)
t = t.replace("include /etc/nginx/fleet-power-gate.conf;", f"include {gate};")
open(dst, "w").write(t)
PY
  cat > "$D/$2/nginx.conf" <<EOF
user www-data;
worker_processes 1;
pid $D/$2/nginx.pid;
error_log $D/$2/error.log notice;
events { worker_connections 64; }
http {
  include /etc/nginx/mime.types;
  client_body_temp_path $D/$2/cb; proxy_temp_path $D/$2/pt; fastcgi_temp_path $D/$2/ft;
  uwsgi_temp_path $D/$2/ut; scgi_temp_path $D/$2/st;
  include $D/$2/vhost.conf;
}
EOF
  nginx -t -c "$D/$2/nginx.conf" 2>&1 | sed "s/^/  [$2] /"
  nginx -c "$D/$2/nginx.conf"
}
prep "$OLD" old 18441
prep "$NEW" new 18442
sleep 0.5
c() {  # c <port> <path> [curl args...]
  local port=$1 path=$2; shift 2
  curl -sk --resolve $H:$port:127.0.0.1 "https://$H:$port$path" "$@"
}
fail=0
ok() { echo "PASS $1"; }
bad() { echo "FAIL $1"; fail=1; }

echo "## without credentials"
for p in / /v1/models /lb/status /power/button.js /power/api/state; do
  o=$(c 18441 $p -o /dev/null -w '%{http_code}'); n=$(c 18442 $p -o /dev/null -w '%{http_code}')
  [ "$o" = 401 ] && [ "$n" = 401 ] && ok "$p -> 401 old/new" || bad "$p old=$o new=$n"
done

echo "## bodies API clients see: OLD vs NEW, byte for byte"
for p in /v1/models /gateway.js /gateway.css /index.html /healthz /nope; do
  o=$(c 18441 $p -u mirror:$PASS | sha256sum); n=$(c 18442 $p -u mirror:$PASS | sha256sum)
  [ "$o" = "$n" ] && ok "$p identical (${o:0:12})" || bad "$p differs"
done
hdr() { c $1 "$2" -u mirror:$PASS -D - -o /dev/null | tr -d '\r' | grep -viE '^(date|content-length):' | sort; }
for p in /v1/models /lb/status /gateway.js; do
  if diff <(hdr 18441 $p) <(hdr 18442 $p) >/dev/null; then ok "$p headers identical (minus Date/Content-Length)"; else bad "$p headers differ"; diff <(hdr 18441 $p) <(hdr 18442 $p); fi
done
ko=$(c 18441 /lb/status -u mirror:$PASS | python3 -c 'import json,sys;print(sorted(json.load(sys.stdin)))')
kn=$(c 18442 /lb/status -u mirror:$PASS | python3 -c 'import json,sys;print(sorted(json.load(sys.stdin)))')
[ "$ko" = "$kn" ] && ok "/lb/status same JSON keys (values are live)" || bad "/lb/status keys differ"
if c 18442 /lb/status -u mirror:$PASS | grep -q 'button.js'; then bad "/lb/status got injected"; else ok "/lb/status not injected"; fi

echo "## the panel page: NEW = OLD + exactly the one script tag"
c 18441 / -u mirror:$PASS > $D/old.html; c 18442 / -u mirror:$PASS > $D/new.html
diff $D/old.html $D/new.html; n=$(grep -c '<script src="/power/button.js" defer></script></body>' $D/new.html)
python3 - $D/old.html $D/new.html <<'PY' && ok "panel: new == old with one <script> before </body> (count=$n)" || bad "panel injection wrong"
import sys
o, n = open(sys.argv[1]).read(), open(sys.argv[2]).read()
assert n == o.replace("</body>", '<script src="/power/button.js" defer></script></body>', 1), "mismatch"
PY

echo "## SSE through NEW (a real 8-token streamed completion): events arrive as a stream"
body='{"model":"Qwen/Qwen3.8-Flash-Next","stream":true,"max_tokens":8,"messages":[{"role":"user","content":"Say hi."}]}'
for port in 18441 18442; do
  c $port /v1/chat/completions -u mirror:$PASS -N -H 'Content-Type: application/json' -d "$body" \
    | python3 -c '
import sys, time
t0 = time.time(); stamps = []
for line in sys.stdin:
    if line.startswith("data:"): stamps.append(round(time.time() - t0, 3))
print(len(stamps), "data events; first at", stamps[:1], "last at", stamps[-1:])' | sed "s/^/  [$port] /"
done
echo "## transfer-encoding of the SSE response (streamed, not buffered into a Content-Length)"
for port in 18441 18442; do
  c $port /v1/chat/completions -u mirror:$PASS -N -D - -o /dev/null -H 'Content-Type: application/json' -d "$body" \
    | tr -d '\r' | grep -iE '^(content-type|transfer-encoding|content-length):' | sed "s/^/  [$port] /"
done

echo "## /power/ through NEW (frps must route it; before vllm-power-frpc runs it reaches the LB)"
echo "  /power/api/state -> $(c 18442 /power/api/state -u mirror:$PASS -o /dev/null -w '%{http_code}')"

nginx -c $D/old/nginx.conf -s quit; nginx -c $D/new/nginx.conf -s quit; sleep 0.5
rm -rf $D
echo "## the live vhost was not touched: $(sha256sum "$OLD")"
exit $fail
