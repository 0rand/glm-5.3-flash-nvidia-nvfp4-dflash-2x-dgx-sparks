#!/usr/bin/env bash
# Stop GLM-5.3-Flash NVIDIA-NVFP4 + DFlash2 on both nodes.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
[ -f .env ] || { echo "[glm53f] missing .env"; exit 1; }
set -a; . ./.env; set +a
: "${CONTAINER:=glm53f-nvidia}"; : "${WORKER_SSH_TARGET:=${WORKER_IP:-}}"

# stop the tailer if it is running
if [ -f logs/tail.pid ]; then
  kill "$(cat logs/tail.pid)" 2>/dev/null || true
  rm -f logs/tail.pid
fi

echo "[glm53f] stopping $CONTAINER"
[ -n "$WORKER_SSH_TARGET" ] && ssh -o BatchMode=yes "$WORKER_SSH_TARGET" "docker rm -f $CONTAINER" >/dev/null 2>&1 && echo "  worker: stopped" || echo "  worker: nothing to stop (or unreachable)"
docker rm -f "$CONTAINER" >/dev/null 2>&1 && echo "  head:   stopped" || echo "  head:   nothing to stop"
echo "[glm53f] done"
