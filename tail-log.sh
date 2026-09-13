#!/usr/bin/env bash
# Follow the boot log, or the live container log.
#
#   ./tail-log.sh              # follow the newest logs/boot-*.log
#   ./tail-log.sh --list       # list boot logs
#   ./tail-log.sh --docker     # follow docker logs of the head container
#   ./tail-log.sh -n 100       # newest boot log, last 100 lines then follow
#   ./tail-log.sh <file>       # a specific file
#
# Ctrl-C stops following.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
[ -f .env ] || { echo "[glm53f] missing .env"; exit 1; }
set -a; . ./.env; set +a
: "${CONTAINER:=glm53f-nvidia}"

LOG_DIR="$HERE/logs"
COUNT=""; LOGFILE=""

case "${1:-}" in
  --list|-l)
    ls -1t "$LOG_DIR"/boot-*.log 2>/dev/null | while read -r f; do
      echo "  $f  ($(du -h "$f" | cut -f1))"
    done
    exit 0 ;;
  --docker|-d)
    exec docker logs -f --tail "${2:-200}" "$CONTAINER" ;;
  -n)
    COUNT="${2:?usage: $0 -n <lines>}"
    LOGFILE=$(ls -1t "$LOG_DIR"/boot-*.log 2>/dev/null | head -1) ;;
  "")
    LOGFILE=$(ls -1t "$LOG_DIR"/boot-*.log 2>/dev/null | head -1) ;;
  *)
    if [ -f "$1" ]; then LOGFILE="$1"
    elif [ -f "$LOG_DIR/$1" ]; then LOGFILE="$LOG_DIR/$1"
    else echo "[tail-log] no such log: $1"; exit 1; fi ;;
esac

[ -n "$LOGFILE" ] || { echo "[tail-log] no boot logs in $LOG_DIR — run ./start.sh first"; exit 1; }
echo "[tail-log] following: $LOGFILE"
if [ -n "$COUNT" ]; then tail -n "$COUNT" -f "$LOGFILE"; else tail -f "$LOGFILE"; fi
