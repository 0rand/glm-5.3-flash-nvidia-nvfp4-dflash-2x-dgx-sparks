#!/usr/bin/env bash
# Status: containers, health, served model, and the KV pool / boot markers.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
[ -f .env ] || { echo "[glm53f] missing .env"; exit 1; }
set -a; . ./.env; set +a
: "${CONTAINER:=glm53f-nvidia}"; : "${PORT:=8100}"; : "${WORKER_SSH_TARGET:=${WORKER_IP:-}}"

echo "=== Containers ==="
docker ps --format '  head   {{.Names}}  {{.Status}}' | grep -F "$CONTAINER" || echo "  head:   not running"
[ -n "$WORKER_SSH_TARGET" ] && { ssh -o BatchMode=yes -o ConnectTimeout=6 "$WORKER_SSH_TARGET" "docker ps --format '  worker {{.Names}}  {{.Status}}'" 2>/dev/null | grep -F "$CONTAINER" || echo "  worker: not running"; }

echo
echo "=== Health (head :$PORT) ==="
code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "http://localhost:${PORT}/health" 2>/dev/null || echo 000)
echo "  /health -> $code"
if [ "$code" = "200" ]; then
  curl -s --max-time 6 "http://localhost:${PORT}/v1/models" \
    | python3 -c "import sys,json
for m in json.load(sys.stdin)['data']:
    print('   ', m['id'], 'max_model_len=', m.get('max_model_len'))" 2>/dev/null
fi

echo
echo "=== Boot markers / KV pool ==="
docker logs "$CONTAINER" 2>&1 \
  | grep -aE 'Available KV cache memory|CUDA graph memory profiling|Graph capturing finished|GPU KV cache size|Maximum concurrency|Application startup complete' \
  | tail -6 | sed 's/^/  /' || echo "  (no markers yet)"

echo
echo "=== Memory (head) ==="
awk '/^(MemTotal|MemFree|MemAvailable|Cached):/{printf "  %-14s %.2f GiB\n",$1,$2/1048576}' /proc/meminfo
