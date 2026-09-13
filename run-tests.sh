#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# The full measurement set for a running stack. Boot first with ./start.sh, or
# pass --boot to do both.
#
#   ./run-tests.sh            # against an already-serving stack
#   ./run-tests.sh --boot     # ./start.sh, then the whole set
#
# Sequence (order matters):
#   1. spec-bench warm-up #1   — JIT/warmup evidence, NOT the headline number
#   2. spec-bench warm-up #2   — the warmed per-workload result
#   3. throughput diagnostic   — pp1024/tg512, c1, depth 0/2048/8192
#   4. hardmode quality        — 88 scenarios, parallel 4, high effort (e2e timed)
#
# Every step prints its own raw log path. Quote the bench build with any score:
# a bench bump has moved scores by whole points before now.
# ═══════════════════════════════════════════════════════════════════════════════
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
[ -f .env ] || { echo "[glm53f] missing .env"; exit 1; }
set -a; . ./.env; set +a

: "${PORT:=8100}"; : "${MODEL_REPO:=nvidia/GLM-5.3-Flash-NVFP4}"; : "${MODEL_REVISION:=main}"
: "${HF_CACHE:=$HOME/.cache/huggingface}"; : "${TEST_PARALLEL:=4}"; : "${TEST_SEED:=42}"
: "${TEST_MAX_TURNS:=32}"; : "${TEST_TIMEOUT:=360}"; : "${TEST_DEPTHS:=0,2048,8192}"

BASE="http://localhost:$PORT"
MODEL="${SERVED_MODEL_NAME:-$MODEL_REPO}"
TOKENIZER="${TEST_TOKENIZER:-$HF_CACHE/hub/models--${MODEL_REPO//\//--}/snapshots/$MODEL_REVISION}"
BENCH="$(tool-eval-bench --version 2>/dev/null | awk '{print $2}')"
LOGDIR="$HERE/logs"; mkdir -p "$LOGDIR"

if [ "${1:-}" = "--boot" ]; then
  echo "[glm53f] booting first via ./start.sh"
  bash "$HERE/start.sh" || { echo "[glm53f] boot failed — not running tests"; exit 1; }
fi

say() { echo; echo "════ $* ════"; }

say "preflight"
curl -s --max-time 8 "$BASE/v1/models" >/dev/null 2>&1 || { echo "  ERROR: nothing serving at $BASE — run ./start.sh"; exit 1; }
SERVED=$(curl -s --max-time 8 "$BASE/v1/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
echo "  endpoint : $BASE"
echo "  model    : $SERVED"
echo "  bench    : $BENCH   <- quote this with any score"
[ -d "$TOKENIZER" ] || echo "  WARNING: tokenizer dir not found ($TOKENIZER) — set TEST_TOKENIZER in .env"

say "STEP 1 — spec-bench warm-up #1 (JIT/warmup evidence, not the headline)"
S1="$LOGDIR/spec-warm1-$(date +%Y%m%d-%H%M%S).log"; SECONDS=0
tool-eval-bench bench --spec-bench --depth "$TEST_DEPTHS" --base-url "$BASE" --model "$SERVED" --no-live > "$S1" 2>&1
echo "  elapsed: ${SECONDS}s  log: $S1"
grep -E 'filler|code|structured|Highest acceptance' "$S1" | tail -12 | sed 's/^/  /'

say "STEP 2 — spec-bench warm-up #2 (warmed result)"
S2="$LOGDIR/spec-warm2-$(date +%Y%m%d-%H%M%S).log"; SECONDS=0
tool-eval-bench bench --spec-bench --depth "$TEST_DEPTHS" --base-url "$BASE" --model "$SERVED" --no-live > "$S2" 2>&1
echo "  elapsed: ${SECONDS}s  log: $S2"
grep -E 'filler|code|structured|Highest acceptance' "$S2" | tail -12 | sed 's/^/  /'

say "STEP 3 — throughput diagnostic (pp1024/tg512, c1, depth $TEST_DEPTHS)"
P1="$LOGDIR/throughput-$(date +%Y%m%d-%H%M%S).log"
tool-eval-bench --perf-only --benchy-runs 1 --pp 1024 --tg 512 \
  --depth "$TEST_DEPTHS" --concurrency 1 --base-url "$BASE" --tokenizer "$TOKENIZER" > "$P1" 2>&1
echo "  log: $P1"
grep -E 'pp1024' "$P1" | sed 's/[│┃]/|/g' | sed 's/^/  /'

say "STEP 4 — hardmode quality (88 scenarios, parallel $TEST_PARALLEL, high effort)"
MODEL="$SERVED" BASE="$BASE" TOKENIZER="$TOKENIZER" \
  TEST_PARALLEL="$TEST_PARALLEL" TEST_SEED="$TEST_SEED" \
  TEST_MAX_TURNS="$TEST_MAX_TURNS" TEST_TIMEOUT="$TEST_TIMEOUT" \
  bash "$HERE/replicate-hardmode.sh"

say "DONE — bench $BENCH"
echo "  logs in: $LOGDIR"
