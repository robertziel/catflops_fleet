#!/bin/bash
# After host-setup.sh ran on the boxes: prove the REAL chain with the DRY-RUN key only, and power nothing off.
# The service must be running in DRY RUN. For each box this script:
#   1. sshes with the dry-run key from inside the vllm-power container (same argv the service uses);
#   2. calls the service API the way nginx would (gate + user + Origin) → confirm → execute.
# Each box must answer "/usr/bin/systemctl poweroff" (sudo -l: "allowed"), and the audit log must
# have the lines. Run on the LB's box from the repo root, naming this deployment's origin and boxes:
#   POWER_ORIGIN=https://<gateway host> POWER_VERIFY_BOXES="box-a=<ip> box-b=<ip> box-c=127.0.0.1" \
#     bash power/install/verify-dryrun.sh
# (POWER_VERIFY_BOXES: name=address pairs as the service's inventory names them, space-separated.)
set -uo pipefail
G=$(cat "$HOME/.config/fleet-power/gate_secret")
ORIGIN=${POWER_ORIGIN:?POWER_ORIGIN is not set — the gateway origin the service allows (POWER_ORIGINS)}
: "${POWER_VERIFY_BOXES:?POWER_VERIFY_BOXES is not set — name=address pairs, space-separated}"
fail=0
[ "$(docker inspect vllm-power --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -c '^POWER_DRY_RUN=1$')" = 1 ] \
  || { echo "REFUSING: vllm-power is not in DRY RUN"; exit 2; }
api() { curl -s -H "X-Power-Gate: $G" -H 'X-Power-User: verify-dryrun' -H "Origin: $ORIGIN" \
             -H 'Content-Type: application/json' "http://127.0.0.1:8016/power/api/$1" -d "$2"; }
for spec in $POWER_VERIFY_BOXES; do
  box=${spec%%=*}; host=${spec#*=}
  out=$(docker exec vllm-power ssh -i /keys/fleet-power-dryrun -F none -o IdentitiesOnly=yes \
        -o IdentityAgent=none -o BatchMode=yes -o StrictHostKeyChecking=yes \
        -o UserKnownHostsFile=/keys/known_hosts -o ConnectTimeout=5 -T fleet-power@$host poweroff 2>&1)
  rc=$?
  if [ $rc = 0 ] && [ "$out" = "/usr/bin/systemctl poweroff" ]; then echo "PASS $box ssh dry-run key -> '$out'"
  else echo "FAIL $box ssh dry-run key rc=$rc: $out"; fail=1; fi
  tok=$(api confirm "{\"box\":\"$box\",\"action\":\"poweroff\"}" | python3 -c 'import json,sys;print(json.load(sys.stdin).get("token",""))')
  res=$(api execute "{\"token\":\"$tok\",\"typed\":\"$box\",\"ack_cluster\":true}")
  if echo "$res" | python3 -c 'import json,sys;d=json.load(sys.stdin);sys.exit(0 if d.get("result")=="sent" and d.get("dry_run") is True and "/usr/bin/systemctl poweroff" in d.get("output","") else 1)'
  then echo "PASS $box service API (DRY RUN) -> $res"; else echo "FAIL $box service API -> $res"; fail=1; fi
done
echo "## audit log (last 6 lines)"; tail -6 "$HOME/.config/fleet-power/audit/audit.jsonl"
echo "## uptime (nothing went down)"; uptime -s
for spec in $POWER_VERIFY_BOXES; do host=${spec#*=}; [ "$host" = 127.0.0.1 ] || ssh "$host" uptime -s; done
exit $fail
