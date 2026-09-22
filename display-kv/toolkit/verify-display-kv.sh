#!/bin/bash
# Verify display-reserve KV unlock state on this node (run on the head or worker).
# Checks loaded module/card identity, then proves capability with real allocations.
set -euo pipefail
cd "$(dirname "$0")"

echo "== DRM module =="
[ -d /sys/module/nvidia_drm ] || {
  echo "!! nvidia_drm is not loaded" >&2; exit 1
}
m=$(cat /sys/module/nvidia_drm/parameters/modeset 2>/dev/null || true)
f=$(cat /sys/module/nvidia_drm/parameters/fbdev 2>/dev/null || true)
echo "nvidia_drm loaded; parameter files: modeset=${m:-unreported} fbdev=${f:-unreported}"
echo "(580.x reports these files empty; the allocation probe below is authoritative)"

echo "== NVIDIA card (want 000f:01:00.0) =="
NVIDIA_CARD=""
for c in card0 card1; do
  d=$(readlink -f /sys/class/drm/$c/device | sed 's|.*/||')
  echo "$c: $d"
  [ "$d" = "000f:01:00.0" ] && NVIDIA_CARD=/dev/dri/$c
done
[ -z "$NVIDIA_CARD" ] && echo "!! NVIDIA DRM card not found" && exit 1
echo "using $NVIDIA_CARD"

echo "== compute stack state (informational) =="
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader || true

echo "== integrity probe =="
DS41_DRM_CARD=$NVIDIA_CARD ./probe_display_kv
echo "== SM bandwidth probe =="
DS41_DRM_CARD=$NVIDIA_CARD ./sm_bandwidth_all
echo "== VERIFY DONE =="
