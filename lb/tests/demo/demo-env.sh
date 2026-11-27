# Sourced by up.sh, down.sh and audit.sh: WHICH demo instance this is. Two instances run side by side when they differ
# in DEMO_PORT (or DEMO_ID): every container name, label and volume carries the id, and down.sh removes only its own.
#   DEMO_PORT  the loopback port the panel is published on (default 28710)
#   DEMO_ID    the instance's name (default p<DEMO_PORT>): letters, digits, - and _
port="${DEMO_PORT:-28710}"
id="${DEMO_ID:-p$port}"
case "$id" in (*[!A-Za-z0-9_-]*|'') echo "DEMO_ID=$id: letters, digits, - and _ only" >&2; exit 2;; esac
label=fleet-demo=ux                       # every demo resource, any instance
ilabel="fleet-demo-id=$id"                # this instance's
lbname="demo-lb-$id"
vol="fleet-demo-state-$id"
pwvol="fleet-demo-pw-$id"
