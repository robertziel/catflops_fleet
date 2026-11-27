#!/usr/bin/env bash
# THE DEMO HARNESS: the gateway on a local docker network with three synthetic boxes, two stand-in engines,
# an alert receiver that refuses every alert, and seeded history — for UX audits (audit.sh) and for looking at the
# panel's box-health, findings, alerts, /metrics, daily, Activity, Check and History views without a fleet.
#
#   lb/tests/demo/up.sh          build the image from this checkout, seed, start; prints the URL
#   lb/tests/demo/down.sh        stop and remove every demo container, the network and the state volume
#
# Wiring (env, all optional): DEMO_PORT (loopback port the panel is published on; default 28710), DEMO_ID (the
# instance; default p<DEMO_PORT>), DEMO_IMAGE (default fleet-lb:demo), DEMO_ENGINE_DOWN_AFTER_S (box-c's classic server
# stops after this; default 45). See demo-env.sh.
# SIDE BY SIDE: every container (demo-<role>-<id>), the state volume and the labels carry the instance's id, so two
# instances on two ports run at once; down.sh (same DEMO_PORT / DEMO_ID) removes only its own, nothing is left behind.
#
# ONE NETWORK NAMESPACE, NO DOCKER NETWORK. The sidecars join demo-lb's namespace (--network container:demo-lb), so
# the gateway reaches its engines, its alert receiver and the sender on loopback and the demo creates no docker network
# (a host whose address pools are spent cannot create one). Ports inside: 8010 gateway, 48291/udp telemetry, 8100 and
# 8001 engines, 8080 alert receiver. The audit joins the same namespace and opens http://127.0.0.1:8010/.
# (demo-lb is demo-lb-<id> and so on: demo-env.sh.)
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
lb="$(cd "$here/../.." && pwd)"
image="${DEMO_IMAGE:-fleet-lb:demo}"
. "$here/demo-env.sh"

"$here/down.sh" >/dev/null
docker build -q -t "$image" "$lb" >/dev/null
docker volume create --label "$label" --label "$ilabel" "$vol" >/dev/null

run() { docker run -d --label "$label" --label "$ilabel" "$@" >/dev/null; }
side="--network container:$lbname"
code="-v $here:/demo:ro"

docker run --rm --label "$label" --label "$ilabel" -v "$vol":/data $code "$image" python -I /demo/seed.py /data
run --name "$lbname" -p "127.0.0.1:$port:8010" -v "$vol":/data -v "$here/fleet.demo.yml":/etc/vllm-lb/fleet.yml:ro \
  -e LB_FLEET=/etc/vllm-lb/fleet.yml -e MODELS_SEARCH_DIRS=/mnt/models -e LB_ALLOWED_HOSTS="127.0.0.1:$port,localhost:$port" -e LB_METRICS=1 -e LB_SEARCH_URL= \
  -e LB_ALERT_WEBHOOK_URL=http://127.0.0.1:8080/hook -e LB_HW_TIERS_FILE=/data/hw-tiers.json \
  "$image"
run --name "demo-alerts-$id" $side $code "$image" python -I /demo/alert_sink.py
run --name "demo-engine-$id" $side -e DEMO_ENGINE_DOWN_AFTER_S="${DEMO_ENGINE_DOWN_AFTER_S:-45}" $code "$image" \
  python -I /demo/engine.py
run --name "demo-sender-$id" $side -v "$vol":/data -e DEMO_LB_HOST=127.0.0.1 -e DEMO_LB_HW_PORT=48291 \
  -e DEMO_INPUTS=/data/inputs.json $code "$image" python -I /demo/sender.py

for _ in $(seq 60); do
  if curl -fsS "http://127.0.0.1:$port/healthz" >/dev/null 2>&1; then
    # One power.action through the real route, as fleet-power posts it (power/fleet_power/app.py `_notify`).
    curl -fsS -X POST -H 'Content-Type: application/json' "http://127.0.0.1:$port/lb/events" \
      -d '{"kind":"power.action","subject":"box-d","summary":"clock box-d: ok by admin","action":"clock","result":"ok","user":"admin","dry_run":false,"rc":0}' >/dev/null
    echo "demo $id up: http://127.0.0.1:$port/ · volume $vol"
    exit 0
  fi
  sleep 1
done
echo "$lbname did not answer /healthz in 60 s:" >&2
docker logs --tail 40 "$lbname" >&2
exit 1
