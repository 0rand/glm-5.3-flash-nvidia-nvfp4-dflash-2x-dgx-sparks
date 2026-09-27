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
SNAPSHOT="$HF_CACHE/hub/$CACHE_ENTRY/snapshots/$MODEL_REVISION"
WORKER_WORKDIR="${WORKER_DIR:-$HERE}"

# hf/huggingface_hub write to $HF_HOME/hub, not to $HF_CACHE — without this the
# weights land in ~/.cache/huggingface while every later step looks in $HF_CACHE.
export HF_HOME="$HF_CACHE"

say() { echo; echo "════ $* ════"; }

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
if [ -f "$HF_CACHE/hub/$CACHE_ENTRY/snapshots/$MODEL_REVISION/config.json" ]; then
  echo "  $CACHE_ENTRY @ $MODEL_REVISION present (skip download)"
else
  if command -v hf >/dev/null 2>&1; then
    hf download "$MODEL_REPO" --revision "$MODEL_REVISION" || { echo "  ERROR: hf download failed"; exit 1; }
  else
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL_REPO', revision='$MODEL_REVISION')" \
      || { echo "  ERROR: download failed (install huggingface_hub or the hf CLI)"; exit 1; }
  fi
fi

say "STEP 2b — sync the weights to the worker (RoCE)"
# huggingface_hub >= 1.x keeps the bytes in a SHARED content store (hub/blobs/xx/<sha>)
# and the repo's own blobs/ holds only relative symlinks into it. Syncing just
# $CACHE_ENTRY ships ~1 MB of dangling links. Ship the entry plus exactly the shared
# blobs it references (not the whole hub/ — the cache may hold unrelated models).
ssh -o BatchMode=yes "$WORKER_ROCE_SSH_TARGET" "mkdir -p '$HF_CACHE/hub'"
HUB_REAL=$(cd "$HF_CACHE/hub" && pwd -P)
SYNC_LIST=$(mktemp)
{
  echo "$CACHE_ENTRY"
  find "$HF_CACHE/hub/$CACHE_ENTRY" -type l -exec readlink -f {} + \
    | grep "^$HUB_REAL/blobs/" | sed "s|^$HUB_REAL/||" | sort -u
} > "$SYNC_LIST"
echo "  $(($(wc -l < "$SYNC_LIST") - 1)) shared blobs referenced outside $CACHE_ENTRY"
rsync -a -r --info=progress2 --files-from="$SYNC_LIST" "$HF_CACHE/hub/" "$WORKER_ROCE_SSH_TARGET:$HF_CACHE/hub/" \
  && echo "  synced $CACHE_ENTRY" || echo "  ERROR: rsync failed"
rm -f "$SYNC_LIST"

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
ssh -o BatchMode=yes "$WORKER_ROCE_SSH_TARGET" "mkdir -p '$DRAFT_MOUNT_DIR'"
rsync -a --info=progress2 "$DRAFT_MOUNT_DIR/$DRAFT_NAME" "$WORKER_ROCE_SSH_TARGET:$DRAFT_MOUNT_DIR/" \
  && echo "  synced $DRAFT_NAME" || echo "  ERROR: draft rsync failed"

say "STEP 4 — verify both nodes"
# Checking that config.json exists is not enough: the snapshot dir holds only
# symlinks, so a sync that copied the links but not the blobs behind them still
# has config.json while every weight shard points at nothing. Instead, measure
# what is actually there on each node and require the two to match.
#
# VERIFY is a small script run once on the head and once on the worker (over ssh),
# so both sides are measured the same way. Args: $1 = weights snapshot dir,
# $2 = draft dir. It prints one line: "<weights bytes> <broken links> <draft bytes>"
#   b: size of the weights, following symlinks (du -L) so it counts the real
#      blob data, not the size of the links themselves
#   x: number of symlinks whose target is missing (find -xtype l)
#   d: size of the draft dir (a plain copy, no symlinks to follow)
# A missing dir yields 0 bytes rather than an empty field.
VERIFY='S="$1"; D="$2"
  b=$(du -sbL "$S" 2>/dev/null | cut -f1); x=$(find "$S" -xtype l 2>/dev/null | wc -l)
  d=$(du -sb "$D" 2>/dev/null | cut -f1); echo "${b:-0} $x ${d:-0}"'
# Head: run VERIFY locally. "_" fills $0 so the paths land in $1 and $2.
# Results: H_W = weights bytes, H_X = broken links, H_D = draft bytes.
read -r H_W H_X H_D <<< "$(bash -c "$VERIFY" _ "$SNAPSHOT" "$DRAFT_MOUNT_DIR/$DRAFT_NAME")"
# Worker: send the same script over ssh (printf %q quotes it so it survives the
# remote shell). Results go into W_W / W_X / W_D. If ssh fails, the fallback
# "0 unreachable 0" guarantees the comparison below fails.
read -r W_W W_X W_D <<< "$(ssh -o BatchMode=yes "$WORKER_ROCE_SSH_TARGET" \
  "bash -c $(printf '%q' "$VERIFY") _ '$SNAPSHOT' '$DRAFT_MOUNT_DIR/$DRAFT_NAME'" 2>/dev/null || echo "0 unreachable 0")"
echo "  head:   weights=$H_W bytes (broken links: $H_X)  draft=$H_D bytes"
echo "  worker: weights=$W_W bytes (broken links: $W_X)  draft=$W_D bytes"
# Pass only if: the head has weights, the worker has the same number of bytes,
# neither side has a broken link, and the draft is present and the same size on both.
if [ "$H_W" -gt 0 ] && [ "$H_W" = "$W_W" ] && [ "$H_X" = 0 ] && [ "$W_X" = 0 ] \
   && [ "$H_D" -gt 0 ] && [ "$H_D" = "$W_D" ]; then
  echo "  OK: both nodes hold identical, fully-resolved weights and draft"
else
  echo "  ERROR: nodes differ or links dangle — do NOT start; re-run ./download.sh"
  exit 1
fi

echo
echo "[glm53f] download/prep complete — next: ./start.sh"
