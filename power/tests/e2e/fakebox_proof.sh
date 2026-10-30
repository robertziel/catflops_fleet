#!/bin/bash
# What the installed authorized_keys + sudoers allow, proven on a fake Ubuntu 24.04 box built with
# the REAL install/host-setup.sh (tests/e2e/fakebox). Its systemctl is a stub: nothing powers off.
# Run from power/:  bash tests/e2e/fakebox_proof.sh
set -u
K=${FLEET_POWER_KEYS:-$HOME/.config/fleet-power/keys}
docker build -q -t fleet-power-fakebox:test -f tests/e2e/fakebox/Dockerfile . >/dev/null
docker network inspect fp-e2e >/dev/null 2>&1 || docker network create --internal --subnet 10.231.99.0/24 fp-e2e >/dev/null
docker network inspect fp-other >/dev/null 2>&1 || docker network create --internal --subnet 10.231.98.0/24 fp-other >/dev/null
docker rm -f fp-box >/dev/null 2>&1
# The fake box installs the REAL host-setup.sh from PUBLIC keys only (copied here), from-box-b = the test network.
FPK="$(mktemp -d)"; cp "$K"/*.pub "$FPK/"; echo "10.231.99.0/24" > "$FPK/from-box-b"; chmod 755 "$FPK"; chmod 644 "$FPK"/*
docker run -d --name fp-box --network fp-e2e --ip 10.231.99.10 -v "$FPK":/fp-keys:ro fleet-power-fakebox:test >/dev/null
docker network connect --ip 10.231.98.10 fp-other fp-box
sleep 1
S="ssh -F none -o IdentitiesOnly=yes -o IdentityAgent=none -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"
run() { docker run --rm --network "${NET:-fp-e2e}" --user 1000:1000 -v "$K":/keys:ro vllm-power:test sh -c "$1" 2>&1 | sed 's/^/    /'; echo "    -> rc=${PIPESTATUS[0]}"; }
echo "## 1 dry-run key: sudo only ANSWERS whether poweroff is allowed"; run "$S -T -i /keys/fleet-power-dryrun fleet-power@10.231.99.10 poweroff"
echo "## 2 power key with a hostile command: sshd runs the forced one (stub systemctl, as root)"; run "$S -T -i /keys/fleet-power fleet-power@10.231.99.10 'rm -rf / ; id'"
echo "## 3 dry-run key asking for a file: ignored, forced command again"; run "$S -T -i /keys/fleet-power-dryrun fleet-power@10.231.99.10 'cat /etc/shadow'"
echo "## 4 port forwarding (a -W tunnel to the box's own sshd) refused"; run "timeout 8 $S -W 127.0.0.1:22 -i /keys/fleet-power-dryrun fleet-power@10.231.99.10 </dev/null"
echo "## 5 a pty is refused"; run "$S -tt -i /keys/fleet-power-dryrun fleet-power@10.231.99.10"
echo "## 6 sudo inside the box refuses every systemctl but poweroff"
docker exec fp-box su -s /bin/bash fleet-power -c 'for c in "/usr/bin/systemctl poweroff" "/usr/bin/systemctl reboot" "/usr/bin/systemctl poweroff --force" "/bin/sh -c id"; do out=$(sudo -n -l $c 2>&1); echo "    sudo -l $c -> rc=$? ${out}"; done' 
echo "## 7 from=: the same key from a different address is refused"; NET=fp-other run "$S -T -i /keys/fleet-power-dryrun fleet-power@10.231.98.10 poweroff"
echo "## 8 any other key (a throwaway one, not the operator's) is refused for fleet-power"; docker run --rm --network fp-e2e vllm-power:test sh -c "ssh-keygen -q -t ed25519 -N '' -f /tmp/stranger && $S -T -i /tmp/stranger fleet-power@10.231.99.10 true" 2>&1 | sed 's/^/    /'
echo "## 9 the 2200 key with a hostile command: only -lgc 300,2200 runs (stub nvidia-smi, as root)"
run "$S -T -i /keys/fleet-clock-limit fleet-power@10.231.99.10 '-lgc 300,3000 ; id'"
echo "## 10 the 2300 key: only -lgc 300,2300 — one key per cap, neither can set the other's"
run "$S -T -i /keys/fleet-clock-limit-2300 fleet-power@10.231.99.10 '-lgc 300,2200'"
echo "## 11 the clock-release key: only -rgc"; run "$S -T -i /keys/fleet-clock-release fleet-power@10.231.99.10 '-lgc 300,3000'"
echo "## 12 the clock dry-run key: sudo only ANSWERS whether the cap would be allowed"
run "$S -T -i /keys/fleet-clock-dryrun fleet-power@10.231.99.10 anything"
echo "## 13 sudo allows exactly the three nvidia-smi argv and nothing else"
docker exec fp-box su -s /bin/bash fleet-power -c 'for c in "/usr/bin/nvidia-smi -lgc 300,2200" "/usr/bin/nvidia-smi -lgc 300,2300" "/usr/bin/nvidia-smi -rgc" "/usr/bin/nvidia-smi -lgc 300,3000" "/usr/bin/nvidia-smi -lgc 300,2250" "/usr/bin/nvidia-smi -lgc 2600" "/usr/bin/nvidia-smi -i 0 -lgc 300,2200" "/usr/bin/nvidia-smi -r" "/usr/bin/nvidia-smi" "/usr/bin/nvidia-smi -lgc 300,2200 ; id"; do out=$(sudo -n -l $c 2>&1); echo "    sudo -l $c -> rc=$? ${out}"; done'
echo "## what the stub nvidia-smi recorded (only the keys' own forced commands, as root):"; docker exec fp-box sh -c 'cat /tmp/nvidia-smi.log 2>/dev/null || echo "(no calls)"' | sed 's/^/    /'
echo "## what the stub systemctl recorded (only the power key's call, once):"; docker exec fp-box cat /tmp/systemctl.log | sed 's/^/    /'
echo "## the installed files:"; docker exec fp-box sh -c 'getent shadow fleet-power | cut -d: -f1,2; stat -c "%A %U:%G %n" /var/lib/fleet-power/.ssh /var/lib/fleet-power/.ssh/authorized_keys /etc/sudoers.d/fleet-power; cat /etc/sudoers.d/fleet-power; cut -c1-120 /var/lib/fleet-power/.ssh/authorized_keys' | sed 's/^/    /'
