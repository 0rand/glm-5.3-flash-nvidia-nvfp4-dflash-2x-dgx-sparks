#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# Fetch/prep everything this stack needs, on BOTH nodes.
#
#   1. docker image        : pull ONCE on the head, ship to the worker
#                            (docker save | ssh docker load over RoCE) — never
#                            pull the same image twice over the slow WAN.
#   2. NVIDIA NVFP4 weights: hf download into the head's HF cache, then rsync
#                            the cache entry to the worker (RoCE).
#   3. DFlash2 draft model : incoai original -> DRAFT_MOUNT_DIR/<DRAFT_NAME>,
#                            synced to the worker at the SAME absolute path
#                            (both ranks bind-mount it as /workspace/models).
#   4. verify              : image IDs match; both nodes see weights + draft.
#
#   ./download.sh
# ═══════════════════════════════════════════════════════════════════════════════
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
[ -f .env ] || { echo "[glm53f] missing .env — run: cp .env.sample .env"; exit 1; }
set -a; . ./.env; set +a

: "${IMAGE:=pilcothink/vllm_spark_glm53:0.28}"
: "${HF_CACHE:=$HOME/.cache/huggingface}"
: "${DRAFT_MOUNT_DIR:=$HOME/models}"
: "${DRAFT_NAME:=glm53-dflash2-orig}"
: "${DRAFT_REPO:=incoai/GLM-5.3-Flash-DFlash2}"
: "${WORKER_SSH_TARGET:=${WORKER_IP:-}}"
: "${WORKER_ROCE_SSH_TARGET:=$WORKER_SSH_TARGET}"
: "${MODEL_REPO:=nvidia/GLM-5.3-Flash-NVFP4}"
: "${MODEL_REVISION:=main}"

CACHE_ENTRY="models--${MODEL_REPO//\//--}"
WORKER_WORKDIR="${WORKER_DIR:-$HERE}"

SNAPSHOT="$HF_CACHE/hub/$CACHE_ENTRY/snapshots/$MODEL_REVISION"

say() { echo; echo "════ $* ════"; }
wssh() { ssh -o BatchMode=yes "$WORKER_ROCE_SSH_TARGET" "$@"; }

# "relative-path size" for every regular file under $1, following symlinks.
# HF snapshots are symlinks into blobs/: a missing blob drops out of the list
# (broken link), a truncated one shows the wrong size.
manifest_cmd() { printf "find -L %q -type f -printf '%%P %%s\\\\n' 2>/dev/null | LC_ALL=C sort" "$1"; }

# Head snapshot is complete when every shard named in the index resolves to a
# non-empty file (hf only links a shard into snapshots/ once it has finished).
head_weights_complete() {
  local idx="$SNAPSHOT/model.safetensors.index.json" shard
  [ -f "$SNAPSHOT/config.json" ] && [ -s "$idx" ] || return 1
  for shard in $(grep -o '"[^"]*\.safetensors"' "$idx" | tr -d '"' | sort -u); do
    [ -s "$SNAPSHOT/$shard" ] || return 1
  done
}

# Fail unless the worker's copy of $1 matches the head's, file for file.
verify_same() {
  local label="$1" path="$2" diffs
  diffs=$(diff <(eval "$(manifest_cmd "$path")") <(wssh "$(manifest_cmd "$path")"))
  if [ -z "$diffs" ]; then
    echo "  $label: head and worker match ($(eval "$(manifest_cmd "$path")" | wc -l) files)"
  else
    echo "  ERROR: $label differs on the worker (< head, > worker):"
    echo "$diffs" | head -10 | sed 's/^/    /'
    return 1
  fi
}

# Fail before rsync if the worker filesystem can't take what's still missing.
check_worker_space() {
  local src="$1" dst_parent="$2" name need have avail largest
  name=$(basename "$src")
  need=$(du -sbL "$src" | cut -f1)                 # -L: blobs may be links into hub/blobs/
  have=$(wssh "du -sbL '$dst_parent/$name' 2>/dev/null | cut -f1" || true)
  avail=$(wssh "df -B1 --output=avail '$dst_parent' | tail -1" | tr -d ' ')
  largest=$(find -L "$src" -type f -printf '%s\n' | sort -n | tail -1)
  local missing=$(( need - ${have:-0} + ${largest:-0} ))   # + one file of rsync temp headroom
  if [ -z "$avail" ] || [ "$missing" -gt "$avail" ]; then
    echo "  ERROR: not enough disk on the worker for $name"
    echo "         needs ~$(( missing / 1024**3 )) GiB more under $dst_parent, has $(( ${avail:-0} / 1024**3 )) GiB free"
    exit 1
  fi
}

[ -n "$WORKER_SSH_TARGET" ] || { echo "ERROR: WORKER_SSH_TARGET (or WORKER_IP) must be set in .env"; exit 1; }

say "STEP 1 — docker image on the head"
if docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "  $IMAGE present (skip pull)"
else
  echo "  pulling $IMAGE …"
  docker pull "$IMAGE" || { echo "  ERROR: pull failed"; exit 1; }
fi

say "STEP 1b — ship the image to the worker (RoCE), verified by ID"
HEAD_ID=$(docker image inspect "$IMAGE" --format '{{.Id}}')
WORKER_ID=$(ssh -o BatchMode=yes "$WORKER_ROCE_SSH_TARGET" "docker image inspect $IMAGE --format '{{.Id}}' 2>/dev/null" || true)
if [ "$HEAD_ID" = "$WORKER_ID" ] && [ -n "$HEAD_ID" ]; then
  echo "  worker already has the same image ($HEAD_ID)"
else
  echo "  transferring over the fabric (docker save | ssh docker load) …"
  docker save "$IMAGE" | ssh -o BatchMode=yes "$WORKER_ROCE_SSH_TARGET" "docker load" | tail -2
  WORKER_ID=$(ssh -o BatchMode=yes "$WORKER_ROCE_SSH_TARGET" "docker image inspect $IMAGE --format '{{.Id}}'" || true)
  [ "$HEAD_ID" = "$WORKER_ID" ] && echo "  IDs match: $HEAD_ID" || echo "  WARNING: IDs differ (head=$HEAD_ID worker=$WORKER_ID)"
fi

say "STEP 2 — NVIDIA NVFP4 weights into the head's HF cache"
if head_weights_complete; then
  echo "  $CACHE_ENTRY @ $MODEL_REVISION complete (skip download)"
else
  if command -v hf >/dev/null 2>&1; then
    hf download "$MODEL_REPO" --revision "$MODEL_REVISION" || { echo "  ERROR: hf download failed"; exit 1; }
  else
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL_REPO', revision='$MODEL_REVISION')" \
      || { echo "  ERROR: download failed (install huggingface_hub or the hf CLI)"; exit 1; }
  fi
  head_weights_complete || { echo "  ERROR: $SNAPSHOT is incomplete after download"; exit 1; }
fi

say "STEP 2b — sync the weights to the worker (RoCE)"
wssh "mkdir -p '$HF_CACHE/hub'" || { echo "  ERROR: worker unreachable"; exit 1; }
check_worker_space "$HF_CACHE/hub/$CACHE_ENTRY" "$HF_CACHE/hub"
# Newer hf keeps the data in a shared hub/blobs/ store and makes the model's
# blobs/ symlinks into it. --copy-unsafe-links turns links that leave the
# model dir into real files; snapshot -> blobs links stay links (no 2x copy).
# Trailing slashes matter: "unsafe" is judged against the transfer root, so
# the root must be the model dir itself, not hub/.
rsync -a --copy-unsafe-links --info=progress2 "$HF_CACHE/hub/$CACHE_ENTRY/" "$WORKER_ROCE_SSH_TARGET:$HF_CACHE/hub/$CACHE_ENTRY/" \
  || { echo "  ERROR: rsync of $CACHE_ENTRY failed"; exit 1; }
echo "  synced $CACHE_ENTRY"

say "STEP 3 — DFlash2 draft model"
if [ -n "$(ls -A "$DRAFT_MOUNT_DIR/$DRAFT_NAME" 2>/dev/null)" ]; then
  echo "  $DRAFT_MOUNT_DIR/$DRAFT_NAME present (skip download)"
else
  echo "  fetching $DRAFT_REPO -> $DRAFT_MOUNT_DIR/$DRAFT_NAME"
  mkdir -p "$DRAFT_MOUNT_DIR/$DRAFT_NAME"
  if command -v hf >/dev/null 2>&1; then
    hf download "$DRAFT_REPO" --local-dir "$DRAFT_MOUNT_DIR/$DRAFT_NAME" || { echo "  ERROR: draft download failed"; exit 1; }
  else
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download('$DRAFT_REPO', local_dir='$DRAFT_MOUNT_DIR/$DRAFT_NAME')" \
      || { echo "  ERROR: draft download failed"; exit 1; }
  fi
fi

say "STEP 3b — sync the draft at the SAME absolute path"
wssh "mkdir -p '$DRAFT_MOUNT_DIR'" || { echo "  ERROR: worker unreachable"; exit 1; }
check_worker_space "$DRAFT_MOUNT_DIR/$DRAFT_NAME" "$DRAFT_MOUNT_DIR"
rsync -a --copy-unsafe-links --info=progress2 "$DRAFT_MOUNT_DIR/$DRAFT_NAME/" "$WORKER_ROCE_SSH_TARGET:$DRAFT_MOUNT_DIR/$DRAFT_NAME/" \
  || { echo "  ERROR: draft rsync failed"; exit 1; }
echo "  synced $DRAFT_NAME"

say "STEP 4 — verify both nodes"
OK=1
head_weights_complete && echo "  head: all index shards present in $SNAPSHOT" \
  || { echo "  ERROR: head snapshot incomplete: $SNAPSHOT"; OK=0; }
verify_same "weights" "$SNAPSHOT" || OK=0
verify_same "draft"   "$DRAFT_MOUNT_DIR/$DRAFT_NAME" || OK=0

echo
[ "$OK" = 1 ] || { echo "[glm53f] verification FAILED — do not ./start.sh yet"; exit 1; }
echo "[glm53f] download/prep complete — next: ./start.sh"
