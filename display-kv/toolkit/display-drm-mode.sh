#!/bin/bash
# display-drm-mode.sh — runtime-only display-reserve unlock for GB10 headless nodes.
#
# NO system persistence: does not touch /etc/modprobe.d, does not rebuild initramfs.
# Boot defaults (incl. zz-nvidia-drm-override.conf modeset=0) are untouched — a
# plain reboot restores them. This script IS the persistence mechanism: call it
# as a pre-step from an experimental stack launcher, never the production one.
#
# Usage:
#   display-drm-mode.sh status   # report current mode
#   display-drm-mode.sh on       # reload nvidia_drm with modeset=1 fbdev=0 (needs sudo)
#   display-drm-mode.sh off      # reload with boot-default params (restores stock)
#
# Safe to run while a compute stack (vLLM etc.) is up: nvidia_drm is independent
# of the compute driver and has 0 users on headless nodes. NOT safe while a
# display-KV pool is mapped — stop any stack using the pool first (the script
# checks for our display_kv library in /proc/*/maps).
set -euo pipefail
MODE_PARAM=/sys/module/nvidia_drm/parameters/modeset
FB_PARAM=/sys/module/nvidia_drm/parameters/fbdev
SUDO=()

require_sudo() {
  if [ "$EUID" -eq 0 ]; then
    SUDO=()
    return
  fi
  # The head node intentionally has no passwordless sudo: the operator
  # authenticates once, then all later privileged calls are non-interactive
  # and individually checked.
  sudo -v || { echo "ERROR: sudo authorization failed" >&2; exit 1; }
  SUDO=(sudo -n)
}

current_mode() {
  if [ ! -d /sys/module/nvidia_drm ]; then echo "unloaded"; return; fi
  m=$(cat $MODE_PARAM 2>/dev/null || echo "?"); f=$(cat $FB_PARAM 2>/dev/null || echo "?")
  echo "modeset=$m fbdev=$f"
}

pool_in_use() {
  # Read as root so root-owned container workers cannot hide their mappings.
  "${SUDO[@]}" grep -l "libds41_display_kv\|libdisplay_kv" \
    /proc/[0-9]*/maps 2>/dev/null | head -1
}

case "${1:-status}" in
  status)
    echo "nvidia_drm: $(current_mode)"
    ls /dev/dri/ 2>/dev/null | tr '\n' ' '; echo
    echo "GPU cards:"; for c in /dev/dri/card*; do
      d=$(readlink -f /sys/class/drm/$(basename $c)/device | sed 's|.*/||')
      echo "  $c -> $d"
    done
    ;;
  on)
    # Always reload: this driver's parameter files are empty even when args were
    # applied, and the experiment requires reallocation on every stack boot.
    require_sudo
    if [ -n "$(pool_in_use || true)" ]; then
      echo "REFUSING: a display-KV pool is mapped by a live process. Stop it first." >&2
      exit 1
    fi
    echo "system configs untouched; applying runtime-only reload..."
    if ! "${SUDO[@]}" rmmod nvidia_drm; then
      echo "ERROR: failed to unload nvidia_drm" >&2
      exit 1
    fi
    if ! "${SUDO[@]}" modprobe nvidia_drm modeset=1 fbdev=0; then
      echo "ERROR: failed to reload nvidia_drm in display mode; attempting stock restore" >&2
      "${SUDO[@]}" modprobe nvidia_drm || true
      exit 1
    fi
    sleep 1
    # Do not use `lsmod | grep -q` under pipefail: grep exits after its first
    # match, lsmod can receive SIGPIPE, and the successful load is reported as
    # a failure. /sys/module is the authoritative loaded-module state.
    [ -d /sys/module/nvidia_drm ] || {
      echo "ERROR: nvidia_drm is not loaded after modprobe" >&2; exit 1;
    }
    # The NVIDIA card is card0 or card1 depending on whether a firmware
    # simple-framebuffer grabbed card0 first; find it by driver, not number.
    nv_card=""
    for c in /sys/class/drm/card[0-9]*; do
      [ "$(basename "$(readlink -f "$c/device/driver" 2>/dev/null)")" = nvidia ] \
        && [ -e "/dev/dri/$(basename "$c")" ] && { nv_card=/dev/dri/$(basename "$c"); break; }
    done
    if [ -z "$nv_card" ]; then
      echo "ERROR: no nvidia DRM card after reload" >&2; exit 1
    fi
    echo "nvidia DRM card: $nv_card"
    echo "display mode ON (runtime only; reboot restores boot defaults)"
    ;;
  off)
    require_sudo
    if [ -n "$(pool_in_use || true)" ]; then
      echo "REFUSING: a display-KV pool is mapped by a live process. Stop it first." >&2
      exit 1
    fi
    if ! "${SUDO[@]}" rmmod nvidia_drm; then
      echo "ERROR: failed to unload nvidia_drm" >&2; exit 1
    fi
    if ! "${SUDO[@]}" modprobe nvidia_drm; then
      echo "ERROR: failed to restore nvidia_drm boot defaults" >&2; exit 1
    fi
    sleep 1
    echo "restored boot defaults: $(current_mode)"
    ;;
  *) echo "usage: $0 {status|on|off}"; exit 2;;
esac
