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
ssh -o BatchMode=yes "$WORKER_ROCE_SSH_TARGET" "mkdir -p '$HF_CACHE/hub'"
rsync -a --info=progress2 "$HF_CACHE/hub/$CACHE_ENTRY" "$WORKER_ROCE_SSH_TARGET:$HF_CACHE/hub/" \
  && echo "  synced $CACHE_ENTRY" || echo "  ERROR: rsync failed"

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
echo "  head:  weights=$( [ -f "$HF_CACHE/hub/$CACHE_ENTRY/snapshots/$MODEL_REVISION/config.json" ] && echo yes || echo NO )  draft=$( [ -n "$(ls -A "$DRAFT_MOUNT_DIR/$DRAFT_NAME" 2>/dev/null)" ] && echo yes || echo NO )"
ssh -o BatchMode=yes "$WORKER_ROCE_SSH_TARGET" "
  echo \"  worker: weights=\$( [ -f '$HF_CACHE/hub/$CACHE_ENTRY/snapshots/$MODEL_REVISION/config.json' ] && echo yes || echo NO )  draft=\$( [ -n \"\$(ls -A '$DRAFT_MOUNT_DIR/$DRAFT_NAME' 2>/dev/null)\" ] && echo yes || echo NO )\"
" 2>/dev/null || echo "  worker: unreachable"

echo
echo "[glm53f] download/prep complete — next: ./start.sh"
