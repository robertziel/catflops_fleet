#!/usr/bin/env bash
# Fetch a pinned Hugging Face checkpoint and stage it where the TP=2 cluster reads its models — ONE script for every
# checkpoint (operator 2026-10-20: "keep code DRY"). A per-model wrapper (deploy/stage-<model>.sh) is its pin alone:
#
#   STAGE_REPO   the Hugging Face repo, e.g. nvidia/Qwen3.8-27B-NVFP4
#   STAGE_REV    the commit it is pinned to: 40 hex digits, never a branch (a branch moves under a verified copy)
#   STAGE_NAME   its folder under the models roots — the catalog row's `path`, group folder included
#
#   <wrapper> fetch   # box-c: download into MODELS_DIR, every file verified against the repo's own manifest
#   <wrapper> stage   # box-a: copy MODELS_DIR -> its own NVMe (the first MODELS_SEARCH_DIRS root), verified again
#   <wrapper> check   # report every root the cluster searches, change nothing
#
# WHERE IT GOES, and why both: the catalog's `cluster_roots` are searched best first —
#   the first root     box-a's NVMe: the copy both ranks load (box-b reads box-a's export over NFS)
#   MODELS_DIR         box-c's HDD: the copy this script downloads to, so the slow internet download never
#                      loads a GB10 box
# A copy lands as <name>.partial and is renamed only once every file matched the manifest, so the final
# name means a whole checkpoint (tp-rank's own rule, lb/cluster/tp-rank).
#
# THE HEAD COPY USES DIRECT I/O, both for the copy and for the re-hash: a GB10 box serving the cluster has
# ~5 GiB of headroom in ONE unified pool, and a checkpoint through the page cache is pressure it does not need.
set -euo pipefail

REPO=${STAGE_REPO:?STAGE_REPO is not set — the Hugging Face repo to fetch (a deploy/stage-<model>.sh wrapper sets it)}
REV=${STAGE_REV:?STAGE_REV is not set — the commit the checkpoint is pinned to}
NAME=${STAGE_NAME:?STAGE_NAME is not set — its folder under the models roots, as the catalog row path names it}
[[ $REV =~ ^[0-9a-f]{40}$ ]] || { echo "STAGE_REV must be a 40-hex commit, never a branch: '$REV'" >&2; exit 2; }
[[ $NAME =~ ^[A-Za-z0-9_.-]+(/[A-Za-z0-9_.-]+)?$ && $NAME != *..* ]] \
  || { echo "STAGE_NAME must be <folder> or <group>/<folder> of [A-Za-z0-9_.-]: '$NAME'" >&2; exit 2; }
# shellcheck source=/dev/null
. "$(dirname "$0")/models-env.sh"
HDD_ROOT=$MODELS_DIR          # lb/.env — the HDD checkpoints are downloaded to
STAGE_HEAD=${STAGE_HEAD:?STAGE_HEAD is not set — the ssh host whose fast disk the copy lands on (the first MODELS_SEARCH_DIRS root)}
STAGE_WORKER=${STAGE_WORKER:?STAGE_WORKER is not set — the ssh host that reads the head export (the pair other box)}
MANIFEST=.hf-manifest.json          # a dotfile: tp-rank's drafter view and vLLM both skip it
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=8)

# Verify the directory $1 against its own manifest: every file present, the right size, and the right
# content — sha256 for an LFS file, the git blob sha1 for a plain one. `--sizes` checks sizes only
# (instant); `--direct` reads through O_DIRECT so nothing lands in the page cache. One line per problem,
# exit 1 on any. Sent to a box as it stands, so it needs nothing but python3's standard library.
VERIFY_PY=$(cat <<'PY'
import hashlib, json, mmap, os, sys

args = [a for a in sys.argv[1:] if not a.startswith("--")]
flags = {a for a in sys.argv[1:] if a.startswith("--")}
root = args[0]
CHUNK = 16 << 20

def digest(path, algo, size, direct):
    h = hashlib.new(algo)
    if algo == "sha1":
        h.update(b"blob %d\0" % size)
    if direct and size >= CHUNK:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
        buf = mmap.mmap(-1, CHUNK)          # page-aligned, as O_DIRECT requires
        try:
            while True:
                n = os.readv(fd, [buf])
                if n <= 0:
                    break
                h.update(buf[:n])
        finally:
            os.close(fd)
            buf.close()
    else:
        with open(path, "rb") as f:
            while True:
                b = f.read(CHUNK)
                if not b:
                    break
                h.update(b)
    return h.hexdigest()

try:
    with open(os.path.join(root, ".hf-manifest.json"), encoding="utf-8") as f:
        files = [e for e in json.load(f) if e.get("type") == "file"]
except Exception as exc:
    print("absent" if isinstance(exc, FileNotFoundError) else f"no readable manifest in {root}: {type(exc).__name__}: {exc}")
    sys.exit(1)
bad = total = 0
for e in files:
    path = os.path.join(root, e["path"])
    lfs = e.get("lfs")
    size = (lfs or e)["size"]
    total += size
    if not os.path.isfile(path):
        print(f"MISSING {e['path']}"); bad += 1; continue
    have = os.path.getsize(path)
    if have != size:
        print(f"SIZE    {e['path']}: {have} of {size} bytes"); bad += 1; continue
    if "--sizes" in flags:
        continue
    got = digest(path, "sha256" if lfs else "sha1", size, "--direct" in flags)
    want = lfs["oid"] if lfs else e["oid"]
    if got != want:
        print(f"HASH    {e['path']}: {got} is not {want}"); bad += 1
print(f"{'FAILED' if bad else 'whole'}: {len(files)} files, {total} bytes, {bad} problems"
      + (" (sizes only)" if "--sizes" in flags else ""))
sys.exit(1 if bad else 0)
PY
)

verify() { python3 -I -c "$VERIFY_PY" "$@"; }

# path<TAB>size for every file of the manifest $1, the big ones last.
files_of() {
  python3 -I - "$1" <<'PY'
import json, sys
files = [e for e in json.load(open(sys.argv[1], encoding="utf-8")) if e.get("type") == "file"]
for e in sorted(files, key=lambda e: (e.get("lfs") or e)["size"]):
    print(f"{e['path']}\t{(e.get('lfs') or e)['size']}")
PY
}

cmd_fetch() {
  local dest=$HDD_ROOT/$NAME part=$HDD_ROOT/$NAME.partial path size have pids=()
  if [[ -f $dest/config.json ]]; then
    echo "$dest is already there; verifying"
    verify "$dest"
    return
  fi
  mkdir -p "$part"
  curl -fsS --retry 5 --retry-delay 3 -o "$part/$MANIFEST" "https://huggingface.co/api/models/$REPO/tree/$REV"
  while IFS=$'\t' read -r path size; do
    [[ $path =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "refusing a file named '$path'" >&2; exit 1; }
    have=$(stat -c %s "$part/$path" 2>/dev/null || echo 0)
    [[ $have == "$size" ]] && continue                    # whole already: a resumed fetch skips it
    # -C -: resume a partial file. Every file runs side by side; a checkpoint's shards are its bulk.
    curl -fL -sS --retry 8 --retry-delay 5 -C - -o "$part/$path" \
      "https://huggingface.co/$REPO/resolve/$REV/$path" &
    pids+=($!)
  done < <(files_of "$part/$MANIFEST")
  local rc=0 pid
  for pid in ${pids[@]+"${pids[@]}"}; do wait "$pid" || rc=1; done
  (( rc == 0 )) || { echo "a download failed; run fetch again to resume" >&2; exit 1; }
  verify "$part"
  # 755 / 644: box-b's container reads box-a's copy as `nobody` (NFS root_squash).
  chmod 755 "$part"
  find "$part" -maxdepth 1 -type f -exec chmod 644 {} +
  mv -T "$part" "$dest"
  echo "fetched $REPO@${REV:0:12} into $dest"
}

# Runs ON box-a (stdin of `bash -s`): copy $1 (the HDD copy, over NFS) to $2.partial with direct I/O,
# verify it there, rename. The verifier arrives as $3.
STAGE_SH=$(cat <<'SH'
set -euo pipefail
src=$1 dest=$2 verify=$3
if [[ -f $dest/config.json ]]; then
  echo "$dest is already there; verifying"
  python3 -I -c "$verify" "$dest" --direct
  exit
fi
[[ -f $src/config.json ]] || { echo "no $src/config.json on $(hostname): fetch first, and check the HDD mount" >&2; exit 1; }
part=$dest.partial
mkdir -p "$part"
for path in "$src"/* "$src"/.hf-manifest.json "$src"/.[!.]*; do
  [[ -f $path ]] || continue
  file=${path##*/}
  size=$(stat -c %s "$path")
  if [[ $(stat -c %s "$part/$file" 2>/dev/null || echo 0) == "$size" ]]; then
    continue
  fi
  if (( size >= 67108864 )); then
    dd if="$path" of="$part/$file" bs=16M iflag=direct oflag=direct status=none
  else
    cp -- "$path" "$part/$file"
  fi
done
python3 -I -c "$verify" "$part" --direct
chmod 755 "$part"
find "$part" -maxdepth 1 -type f -exec chmod 644 {} +
mv -T "$part" "$dest"
echo "staged into $dest on $(hostname)"
SH
)

cmd_stage() {
  # shellcheck disable=SC2029  # the remote shell expands ~ on purpose: only the box knows its HOME
  "${SSH[@]}" "$STAGE_HEAD" bash -s -- "$HDD_ROOT/$NAME" "$(remote_dir "${MODELS_SEARCH_DIRS%%:*}")/$NAME" "$(printf %q "$VERIFY_PY")" <<<"$STAGE_SH"
}


cmd_check() {
  echo "box-c  $HDD_ROOT/$NAME: $(verify "$HDD_ROOT/$NAME" --sizes 2>&1 | tail -1)"
  local box dir
  for box in "$STAGE_HEAD" "$STAGE_WORKER"; do
    local search; IFS=: read -ra search <<<"$MODELS_SEARCH_DIRS"
    for dir in "${search[@]}"; do
      dir=$(remote_dir "$dir")
      printf '%-8s %s: ' "$box" "$dir/$NAME"
      "${SSH[@]}" "$box" "python3 -I -c $(printf %q "$VERIFY_PY") $dir/$NAME --sizes" 2>&1 | tail -1 || true
    done
  done
}

case ${1:-} in
  fetch) cmd_fetch ;;
  stage) cmd_stage ;;
  check) cmd_check ;;
  *) sed -n '2,12p' "$0"; exit 2 ;;
esac
