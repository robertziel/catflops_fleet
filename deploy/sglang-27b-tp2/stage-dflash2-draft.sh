#!/usr/bin/env bash
# Fetch the DFlash2 block-diffusion drafter and stage it beside the target checkpoint the
# SGLang TP=2 pair loads.
#
#   deploy/sglang-27b-tp2/stage-dflash2-draft.sh fetch   # box-c: download into /srv/models, verified
#   deploy/sglang-27b-tp2/stage-dflash2-draft.sh stage   # box-a:   copy HDD -> its own NVMe (~/models), verified
#   deploy/sglang-27b-tp2/stage-dflash2-draft.sh check   # report every root, change nothing
#
# WHAT IT IS: https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2 at the revision pinned below — the draft
# upstream's start-dflash.sh pins (DRAFT_MODEL/DRAFT_REVISION), 3.58 GiB in one safetensors file.
# incoai/Qwen3.8-27B-DFlash2 mirrors the same weights.
#
# WHY IT GOES BESIDE THE TARGET, not in an HF cache: upstream's single-box script lets the container
# download it into ./.cache/huggingface. A TP=2 pair cannot do that — BOTH ranks must read the SAME
# drafter, and rank 1 lives on the other box. So it is staged into box-a's NVMe `~/models`, which box-b
# already reads over pair-link as /mnt/box-a-models (the same export the vLLM cluster's
# checkpoint uses), and both ranks are pointed at /models/<name> inside the container with
# HF_HUB_OFFLINE=1. Nothing is downloaded at launch, and no rank can launch a different drafter from
# its peer's.
#
# The verify-then-rename pattern is deploy/stage-qwen38-27b-nvfp4.sh's: a copy lands as <name>.partial
# and is renamed only once every file matched the manifest, so the final name means a whole checkpoint.
set -euo pipefail

REPO=z-lab/Qwen3.8-27B-DFlash2
REV=50307d4c4cde6860d4eee73e2547cd786fe8e8a4   # upstream start-dflash.sh's DRAFT_REVISION
NAME=Qwen3.8-27B-DFlash2
# shellcheck source=/dev/null
. "$(dirname "$0")/../models-env.sh"
HDD_ROOT=$MODELS_DIR          # lb/.env — the HDD checkpoints are downloaded to
STAGE_HEAD=${STAGE_HEAD:?STAGE_HEAD is not set — the ssh host whose fast disk the copy lands on (the first MODELS_SEARCH_DIRS root)}
STAGE_WORKER=${STAGE_WORKER:?STAGE_WORKER is not set — the ssh host that reads the head export (the pair other box)}
MANIFEST=.hf-manifest.json
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=8)

# Verify directory $1 against its own manifest: every file present, right size, right content
# (sha256 for an LFS file, the git blob sha1 for a plain one). --sizes checks sizes only; --direct
# reads through O_DIRECT so a 3.6 GiB re-hash does not evict a serving GB10 box's page cache.
# Sent to a box as it stands, so it uses nothing but python3's standard library.
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
        buf = mmap.mmap(-1, CHUNK)
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

# path<TAB>size for every top-level file of the manifest $1, the big ones last. The repo's `assets/`
# directory holds only the model card's images: the engine never reads it, so it is not staged.
files_of() {
  python3 -I - "$1" <<'PY'
import json, sys
files = [e for e in json.load(open(sys.argv[1], encoding="utf-8")) if e.get("type") == "file"]
for e in sorted(files, key=lambda e: (e.get("lfs") or e)["size"]):
    print(f"{e['path']}\t{(e.get('lfs') or e)['size']}")
PY
}

cmd_fetch() {
  local dest=$HDD_ROOT/$NAME part=$HDD_ROOT/$NAME.partial path size have
  if [[ -f $dest/config.json ]]; then
    echo "$dest is already there; verifying"
    verify "$dest"
    return
  fi
  mkdir -p "$part"
  curl -fsS --retry 5 --retry-delay 3 -o "$part/$MANIFEST" "https://huggingface.co/api/models/$REPO/tree/$REV"
  local try
  while IFS=$'\t' read -r path size; do
    [[ $path =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "refusing a file named '$path'" >&2; exit 1; }
    # Three attempts. Each one DELETES whatever is there and fetches the file whole, and the size is
    # re-checked after every attempt — no resume, and no curl-level retry.
    #
    # Both of the obvious shortcuts corrupt the file here, measured on this draft on 2026-10-02:
    #   * `curl -C -` across Hugging Face's CDN redirect: a retry mid-transfer re-requests the
    #     object, the CDN answers 200 with the WHOLE body instead of 206 with the tail, and curl
    #     appends it to what is already on disk — 4,346,768,616 bytes of a 3,848,817,896-byte file.
    #   * `curl --retry` with `-o`: on a retry curl does not truncate the output file, so a second
    #     attempt concatenates onto the first — 3,849,568,256 then 3,948,575,976 bytes of the same
    #     file, i.e. an overshoot of only 750 KB that a size check catches and an eyeball does not.
    # Both produce a file that is plausible, complete-looking and WRONG. A fetch whose only recovery
    # is "start over" cannot do either. The file is 3.6 GiB on a link that carries ~18 MB/s, so a
    # clean restart costs ~3.5 minutes and buys a checkpoint that matched its manifest.
    for try in 1 2 3; do
      have=$(stat -c %s "$part/$path" 2>/dev/null || echo 0)
      if [[ $have == "$size" ]]; then
        break                                             # whole: this attempt and the rest are skipped
      fi
      (( have > 0 )) && echo "discarding $have bytes of $path (expected $size)"
      rm -f -- "$part/$path"
      echo "fetching $path ($size bytes, attempt $try)"
      curl -fL -sS -o "$part/$path" "https://huggingface.co/$REPO/resolve/$REV/$path" || true
    done
    have=$(stat -c %s "$part/$path" 2>/dev/null || echo 0)
    [[ $have == "$size" ]] || { echo "could not fetch $path whole: $have of $size bytes" >&2; exit 1; }
  done < <(files_of "$part/$MANIFEST")
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
  *) sed -n '2,6p' "$0"; exit 2 ;;
esac
