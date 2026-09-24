#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# Boot the EXPERIMENTAL display-KV variant of the proven GLM-5.3-Flash
# NVIDIA-NVFP4 + DFlash2 k=5 stack on 2x DGX Spark, TP=2.
#
# This file is copied from start.sh. The production launcher is untouched.
# It preserves the 11 GB pin's ordinary-RAM use and 1 Mi-token model length,
# adding only the 1.75 GiB display-reserve suffix and fail-closed allocator hook.
#
#   ./start-display-kv.sh preflight   # no restart / no DRM reload
#   ./start-display-kv.sh             # experimental launch; prompts for head sudo
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
[ -f .env ] || { echo "[glm53f] missing .env — run: cp .env.sample .env"; exit 1; }
set -a; . ./.env; set +a

# --- optional keys / defaults -------------------------------------------------
: "${SERVED_MODEL_NAME:=}"; : "${ATTN_BACKEND:=}"; : "${KV_CACHE_MEMORY:=}"
: "${RECIPE_SPEC_EXTRAS:=1}"; : "${ENFORCE_EAGER:=0}"; : "${CAPTURE_SIZE:=16}"
: "${ASYNC_SCHED:=1}"; : "${ESTIMATE_CUDAGRAPHS:=0}"
: "${K:=5}"; : "${CTX:=900096}"; : "${GMU:=0.88}"; : "${SEQS:=4}"; : "${BATCH:=1024}"; : "${PORT:=8100}"
: "${MASTER_PORT:=29503}"; : "${CONTAINER:=glm53f-nvidia}"; : "${TP_SIZE:=2}"
: "${WORKER_SSH_TARGET:=${WORKER_IP:-}}"; : "${DRAFT_MOUNT_DIR:=}"; : "${HF_CACHE:=}"
: "${MODEL_REVISION:=main}"

# --- display-KV experiment invariants and preflight --------------------------
ACTION="${1:-start}"
PRODUCTION_KV_MEMORY=10240000000
DISPLAY_KV_BYTES=1879048192
DISPLAY_KV_MEMORY=12119048192
DISPLAY_KV_RUNTIME="$HERE/display-kv"
DISPLAY_KV_TOOLKIT="${DISPLAY_KV_TOOLKIT:-$HERE/display-kv/toolkit}"
WORKER_HOST="${WORKER_HOST:-$WORKER_SSH_TARGET}"
# The NVIDIA card's /dev/dri/cardN number differs per node (a firmware
# simple-framebuffer takes card0 on some boots). Address it by PCI path instead,
# which is identical on every GB10; containers get the host's /dev/dri for it.
: "${DISPLAY_KV_DRM_CARD:=/dev/dri/by-path/pci-000f:01:00.0-card}"

[ "$ACTION" = start ] || [ "$ACTION" = preflight ] || {
  echo "usage: $0 [start|preflight]" >&2; exit 2;
}
[ "$KV_CACHE_MEMORY" = "$PRODUCTION_KV_MEMORY" ] || {
  echo "ERROR: expected production 11 GB pin $PRODUCTION_KV_MEMORY, got $KV_CACHE_MEMORY" >&2
  exit 1
}
[ "$CTX" = 1048576 ] || {
  echo "ERROR: display-KV experiment requires production CTX=1048576, got $CTX" >&2
  exit 1
}
[ "$K" = 5 ] || {
  echo "ERROR: display-KV experiment requires production DFlash2 K=5, got $K" >&2
  exit 1
}
[ "$DISPLAY_KV_MEMORY" -eq $((PRODUCTION_KV_MEMORY + DISPLAY_KV_BYTES)) ] || {
  echo "ERROR: display-KV budget arithmetic mismatch" >&2; exit 1
}
for f in display_kv_glm.py sitecustomize.py libdisplay_kv_glm.so; do
  [ -s "$DISPLAY_KV_RUNTIME/$f" ] || {
    echo "ERROR: missing display-KV runtime file: $f" >&2; exit 1
  }
done
for f in display-drm-mode.sh prepare-display-drm-both.sh verify-display-kv.sh build.sh; do
  [ -s "$DISPLAY_KV_TOOLKIT/$f" ] || {
    echo "ERROR: missing vendored toolkit file: $DISPLAY_KV_TOOLKIT/$f" >&2; exit 1
  }
done
[ -s "$DISPLAY_KV_TOOLKIT/repo/release/runtime/sources/display_kv.c" ] || {
  echo "ERROR: vendored AGPL allocator source missing" >&2; exit 1
}
bash -n "$0" "$DISPLAY_KV_TOOLKIT/display-drm-mode.sh" \
  "$DISPLAY_KV_TOOLKIT/prepare-display-drm-both.sh"
python3 "$DISPLAY_KV_TOOLKIT/tests/test_display_kv_glm.py"
ssh -o BatchMode=yes -o ConnectTimeout=8 "$WORKER_SSH_TARGET" \
  'sudo -n true' \
  || { echo "ERROR: worker needs passwordless sudo for the DRM reload" >&2; exit 1; }

echo "[display-kv] preflight PASS: pin=$KV_CACHE_MEMORY ctx=$CTX k=$K"
echo "[display-kv] experimental KV budget: $DISPLAY_KV_MEMORY (+$DISPLAY_KV_BYTES display)"
if [ "$ACTION" = preflight ]; then
  exit 0
fi

# Sync the tiny runtime + vendored DRM toolkit directly over SSH. No NFS.
# Same absolute path on the worker: both ranks bind-mount it as /opt/display-kv.
ssh -o BatchMode=yes "$WORKER_SSH_TARGET" "mkdir -p '$DISPLAY_KV_RUNTIME'" \
  || { echo "ERROR: cannot create $DISPLAY_KV_RUNTIME on the worker" >&2; exit 1; }
rsync -a --delete "$DISPLAY_KV_RUNTIME/" \
  "$WORKER_SSH_TARGET:$DISPLAY_KV_RUNTIME/" \
  || { echo "ERROR: display-kv runtime sync to the worker failed" >&2; exit 1; }

# Runtime-only DRM reload. Local node may prompt for sudo once; the worker needs
# passwordless sudo for the launcher to apply its reload remotely.
WORKER_HOST="$WORKER_HOST" \
  "$DISPLAY_KV_TOOLKIT/prepare-display-drm-both.sh" on

# Card existence is checked on both nodes after the DRM reload below.
[ -e "$DISPLAY_KV_DRM_CARD" ] \
  || { echo "ERROR: head: $DISPLAY_KV_DRM_CARD missing after reload" >&2; exit 1; }
ssh -o BatchMode=yes "$WORKER_SSH_TARGET" "[ -e '$DISPLAY_KV_DRM_CARD' ]" \
  || { echo "ERROR: worker: $DISPLAY_KV_DRM_CARD missing after reload" >&2; exit 1; }
echo "[display-kv] DRM card: $DISPLAY_KV_DRM_CARD (head -> $(readlink -f "$DISPLAY_KV_DRM_CARD"), worker -> $(ssh -o BatchMode=yes "$WORKER_SSH_TARGET" "readlink -f '$DISPLAY_KV_DRM_CARD'"))"

# Preserve the production ordinary allocation and add only the display suffix.
KV_CACHE_MEMORY="$DISPLAY_KV_MEMORY"

mkdir -p logs

# --- stop anything already holding the GPU, then give RAM a moment to return ---
# NO static memory preflight here, by design (ruling 2026-09-13):
#   a hardcoded CUDA_FREE_REF cannot know what is actually free at engine init, so a
#   static guard only produces false aborts -- and it blocked boots that the engine
#   itself would have accepted. vLLM performs the authoritative check, and when KV is
#   short it names the largest context that does fit. Let the engine decide.
echo "[glm53f] stopping existing $CONTAINER (both nodes)"
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
[ -n "$WORKER_SSH_TARGET" ] && ssh -o BatchMode=yes "$WORKER_SSH_TARGET" "docker rm -f $CONTAINER" >/dev/null 2>&1 || true
sleep 10

# --- RoCE v2 IPv4 GID index (renumbers across reboots) -----------------------
FIRST_HCA="${HCAS%%,*}"
GID=""
for i in $(seq 0 15); do
  t=$(cat "/sys/class/infiniband/$FIRST_HCA/ports/1/gid_attrs/types/$i" 2>/dev/null || true)
  g=$(cat "/sys/class/infiniband/$FIRST_HCA/ports/1/gids/$i" 2>/dev/null || true)
  case "$t" in
    *"RoCE v2"*) case "$g" in *0000:0000:0000:0000:ffff:*) GID="$i"; break;; esac ;;
  esac
done
[ -n "$GID" ] || { echo "ERROR: no RoCE v2 IPv4 GID found on $FIRST_HCA"; exit 1; }

# --- speculative config as REAL JSON (validated before it reaches the engine) --
SPEC_CONFIG=$(python3 - "$DRAFT_NAME" "$K" "$RECIPE_SPEC_EXTRAS" <<'JSONEOF'
import json, sys
draft, k, extras = sys.argv[1], int(sys.argv[2]), sys.argv[3] == "1"
cfg = {"method": "dflash", "model": f"/workspace/models/{draft}",
       "num_speculative_tokens": k, "attention_backend": "TRITON_ATTN", "kv_cache_dtype": "auto"}
if extras:
    cfg.update({"draft_sample_method": "probabilistic", "rejection_sample_method": "standard",
                "enable_adaptive_verification": False, "disable_eagle_block_drop": False})
out = json.dumps(cfg)
assert json.loads(out) == cfg, "spec config does not round-trip"
print(out)
JSONEOF
)

# --- graph / eager arguments -------------------------------------------------
# Use an explicit 1/0 test, NOT ${VAR:+...}: ":+" expands for ANY set non-empty
# value, so ENFORCE_EAGER=0 used to pass --enforce-eager and silently cancel
# cudagraphs while the launcher still reported "graphs ON".
EAGER_ARG=""; CUDAGRAPH_ARGS=""
if [ "$ENFORCE_EAGER" = "1" ]; then
  EAGER_ARG="--enforce-eager"
else
  CUDAGRAPH_ARGS="--max-cudagraph-capture-size $CAPTURE_SIZE"
fi
ASYNC_ARG=""; [ "$ASYNC_SCHED" = "1" ] && ASYNC_ARG="--async-scheduling"
SERVED_ARG=""; [ -n "$SERVED_MODEL_NAME" ] && SERVED_ARG="--served-model-name $SERVED_MODEL_NAME"

BLOG="$HERE/logs/boot-$(date +%Y%m%d-%H%M%S).log"

echo "[glm53f] launching $CONTAINER"
{
  echo "── GLM-5.3-Flash NVIDIA-NVFP4 + DFlash2"
  echo "   image    : $IMAGE"
  echo "   weights  : $MODEL_REPO @ $MODEL_REVISION"
  echo "   draft    : $DRAFT_MOUNT_DIR/$DRAFT_NAME  (k=$K, method=dflash)"
  echo "   fabric   : $MN_IF / $HCAS / GID $GID   head=$HEAD_IP worker=$WORKER_IP"
  echo "   serving  : ctx $CTX, GMU $GMU, seqs $SEQS, batch $BATCH, port $PORT"
  echo "   kv       : ${KV_CACHE_MEMORY:-GMU-driven (no pin)}"
  echo "   graphs   : $([ "$ENFORCE_EAGER" = "1" ] && echo 'enforce-eager (no cudagraphs)' || echo "cudagraphs <=$CAPTURE_SIZE")"
  echo "   graphmem : $([ "$ESTIMATE_CUDAGRAPHS" = "1" ] && echo 'profiler estimate ON (~2 GiB of KV reserved)' || echo 'profiler estimate OFF — KV keeps the budget')"
  echo "   async    : $([ "$ASYNC_SCHED" = "1" ] && echo 'ON' || echo 'OFF')"
  echo "   spec     : $SPEC_CONFIG"
} | tee "$BLOG"

IMAGE="$IMAGE" \
MODEL_REPO="$MODEL_REPO" MODEL_REVISION="$MODEL_REVISION" \
DRAFT_MOUNT_DIR="$DRAFT_MOUNT_DIR" DRAFT_NAME="$DRAFT_NAME" \
HF_CACHE_DIR="$HF_CACHE" HEAD_IP="$HEAD_IP" WORKER_IP="$WORKER_IP" \
MN_IF="$MN_IF" HCAS="$HCAS" PORT="$PORT" MASTER_PORT="$MASTER_PORT" \
GMU="$GMU" CTX="$CTX" SEQS="$SEQS" BATCH="$BATCH" K="$K" \
KV_CACHE_MEMORY="$KV_CACHE_MEMORY" CONTAINER="$CONTAINER" \
TP_SIZE="$TP_SIZE" ENFORCE_EAGER="$ENFORCE_EAGER" CAPTURE_SIZE="$CAPTURE_SIZE" \
ASYNC_SCHED="$ASYNC_SCHED" ESTIMATE_CUDAGRAPHS="$ESTIMATE_CUDAGRAPHS" \
ATTN_BACKEND="$ATTN_BACKEND" RECIPE_SPEC_EXTRAS="$RECIPE_SPEC_EXTRAS" \
SERVED_MODEL_NAME="$SERVED_MODEL_NAME" \
SPEC_CONFIG="$SPEC_CONFIG" GID="$GID" SERVED_ARG="$SERVED_ARG" \
ASYNC_ARG="$ASYNC_ARG" EAGER_ARG="$EAGER_ARG" CUDAGRAPH_ARGS="$CUDAGRAPH_ARGS" \
  bash "$HERE/vendor/run_cluster_dual.sh" "$IMAGE" "$HEAD_IP" \
    --head "$HF_CACHE" --no-ray --workers "$WORKER_IP" \
    --master-port "$MASTER_PORT" --container-name "$CONTAINER" \
    --ipc=host --privileged --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "$DISPLAY_KV_RUNTIME:/opt/display-kv:ro" \
    -v /dev/dri:/dev/dri \
    -e PYTHONPATH=/opt/display-kv \
    -e DISPLAY_KV_ENABLED=1 \
    -e DISPLAY_KV_MIN_TOTAL_BYTES=$((PRODUCTION_KV_MEMORY + 1)) \
    -e DISPLAY_KV_MAX_ORDINARY_BYTES="$PRODUCTION_KV_MEMORY" \
    -e DISPLAY_KV_LIBRARY=/opt/display-kv/libdisplay_kv_glm.so \
    -e DISPLAY_KV_RECEIPT_DIR=/tmp/display-kv-receipts \
    -e DS41_DRM_CARD="$DISPLAY_KV_DRM_CARD" \
    -e DS41_ORDINARY_CAP_GIB=16 \
    -e NCCL_SOCKET_IFNAME="$MN_IF" \
    -e NCCL_IB_HCA="$HCAS" \
    -e NCCL_IB_GID_INDEX="$GID" \
    -e NCCL_IGNORE_CPU_AFFINITY=1 \
    -e GLOO_SOCKET_IFNAME="$MN_IF" \
    -e VLLM_HOST_IP="$HEAD_IP" \
    -e HF_HUB_OFFLINE=1 \
    -e TRANSFORMERS_OFFLINE=1 \
    -e CUTE_DSL_ARCH=sm_121a \
    -e TORCH_USE_RTLD_GLOBAL=1 \
    -e TORCHINDUCTOR_COMPILE_THREADS=1 \
    -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
    -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    -e VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE=auto \
    -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS="$ESTIMATE_CUDAGRAPHS" \
    -v "$DRAFT_MOUNT_DIR:/workspace/models" \
    -- vllm serve "$MODEL_REPO" \
      --revision "$MODEL_REVISION" \
      --host 0.0.0.0 --port "$PORT" \
      --distributed-executor-backend mp \
      --tensor-parallel-size "$TP_SIZE" \
      --dtype bfloat16 --trust-remote-code \
      --moe-backend b12x --linear-backend b12x \
      ${ATTN_BACKEND:+--attention-backend "$ATTN_BACKEND"} \
      --gpu-memory-utilization "$GMU" \
      --no-enable-flashinfer-autotune \
      --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice \
      --mamba-cache-mode align \
      --enable-prefix-caching --enable-chunked-prefill \
      ${ASYNC_ARG} \
      --max-num-batched-tokens "$BATCH" \
      --max-model-len "$CTX" \
      ${KV_CACHE_MEMORY:+--kv-cache-memory "$KV_CACHE_MEMORY"} \
      --kv-cache-dtype fp8 --block-size 256 \
      --max-num-seqs "$SEQS" \
      ${EAGER_ARG} ${CUDAGRAPH_ARGS:-} \
      --skip-mm-profiling \
      ${SERVED_ARG} \
      --speculative-config "$SPEC_CONFIG" \
    >> "$BLOG" 2>&1 &
BOOT_PID=$!
echo "[glm53f] launcher pid $BOOT_PID — log: $BLOG"

BASE="http://localhost:$PORT"
for i in $(seq 1 120); do
  curl -s --max-time 4 "$BASE/health" >/dev/null 2>&1 && { echo "  serving after ~$((i*20))s"; break; }
  if [ "$i" -ge 4 ] && ! docker inspect -f '{{.State.Running}}' "$CONTAINER" >/dev/null 2>&1; then
    echo "  container $CONTAINER gone after ~$((i*20))s — boot failed"
    break
  fi
  sleep 20
done

curl -s --max-time 8 "$BASE/health" >/dev/null 2>&1 || {
  echo "ABORT: boot failed — $BLOG"
  grep -aiE 'ValueError|RuntimeError|out of memory|not enough|less than desired GPU memory utilization' "$BLOG" | tail -8
  exit 1
}

echo "[glm53f] UP"
grep -aE 'Available KV cache memory|CUDA graph memory profiling|Graph capturing finished|GPU KV cache size|Maximum concurrency|Loading weights took|Application startup complete' "$BLOG" | sed 's/^/  /' | tail -8
