#!/bin/bash
# Runtime-only DRM display-reserve preparation for headless nodes.
# Never writes /etc/modprobe.d or initramfs. A reboot restores boot defaults.
# Uses display-drm-mode.sh for the local node (interactive sudo once) and SSH +
# passwordless sudo for the worker node. Override HOSTS/SSH targets via env.
set -euo pipefail

ACTION="${1:-on}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER_HOST="${WORKER_HOST:-}"
HEAD_SSH_TARGET="${HEAD_SSH_TARGET:-}"

worker_action() {
  local node_ssh="$1" action="$2"
  ssh -o BatchMode=yes -o ConnectTimeout=8 "$node_ssh" \
    DISPLAY_KV_ACTION="$action" DISPLAY_KV_DIR="$(basename "$PWD")" bash -s <<'REMOTE'
set -euo pipefail
action="${DISPLAY_KV_ACTION:-on}"
if sudo -n grep -l "libds41_display_kv\|libdisplay_kv" \
    /proc/[0-9]*/maps >/dev/null 2>&1; then
  echo "REFUSING: a display-KV pool is mapped on this node" >&2
  exit 1
fi
sudo -n rmmod nvidia_drm
case "$action" in
  on)  sudo -n modprobe nvidia_drm modeset=1 fbdev=0 ;;
  off) sudo -n modprobe nvidia_drm ;;
  *) echo "unknown action $action" >&2; exit 2 ;;
esac
[ -d /sys/module/nvidia_drm ] || { echo "nvidia_drm not loaded" >&2; exit 1; }
echo "remote node: runtime reload done ($action)"
REMOTE
}

case "$ACTION" in
  on)
    echo "[display-drm] local node: runtime reload modeset=1 fbdev=0"
    "$DIR/display-drm-mode.sh" on
    if [ -n "$WORKER_HOST" ]; then
      echo "[display-drm] worker node ($WORKER_HOST): runtime reload modeset=1 fbdev=0"
      worker_action "$WORKER_HOST" on
    fi
    ;;
  off)
    echo "[display-drm] local node: restoring boot-default module parameters"
    "$DIR/display-drm-mode.sh" off
    if [ -n "$WORKER_HOST" ]; then
      echo "[display-drm] worker node ($WORKER_HOST): restoring boot-default module parameters"
      worker_action "$WORKER_HOST" off
    fi
    ;;
  status)
    "$DIR/display-drm-mode.sh" status
    if [ -n "$WORKER_HOST" ]; then
      echo "[display-drm] worker node ($WORKER_HOST):"
      ssh -o BatchMode=yes -o ConnectTimeout=8 "$WORKER_HOST" \
        'hostname; [ -d /sys/module/nvidia_drm ] && echo nvidia_drm_loaded || echo nvidia_drm_unloaded'
    fi
    ;;
  *) echo "usage: $0 {on|off|status}  (env: WORKER_HOST=<user>@<ip>)" >&2; exit 2;;
esac
