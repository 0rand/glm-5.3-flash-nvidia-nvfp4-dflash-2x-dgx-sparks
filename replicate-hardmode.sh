#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# REPLICATE the e2e hardmode measurement exactly — the decisive metric.
#
# Invocation is byte-matched to the recorded Run Context + Extra Params:
#   temperature 0.0 · seed 42 · max-turns 32 · timeout 360s · all 88 scenarios
#   parallel 4 · thinking enabled
#   backend-kwargs: {"chat_template_kwargs": {"thinking": true,
#                     "reasoning_effort": "high"}, "temperature": 1.0, "top_p": 1}
#
# The wall clock is measured HERE (not inferred) so the e2e number is trustworthy.
# EVERY run prints its bench build — the evaluator moves between builds, so a
# score without its build tag is meaningless.
#
#   ./replicate-hardmode.sh
#   MODEL=nvidia/GLM-5.3-Flash-NVFP4 TOKENIZER=/path ./replicate-hardmode.sh
# ═══════════════════════════════════════════════════════════════════════════════
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
[ -f "$HERE/.env" ] && { set -a; . "$HERE/.env"; set +a; }

: "${PORT:=8100}"; : "${MODEL_REPO:=nvidia/GLM-5.3-Flash-NVFP4}"; : "${MODEL_REVISION:=main}"
: "${HF_CACHE:=$HOME/.cache/huggingface}"; : "${TEST_PARALLEL:=4}"; : "${TEST_SEED:=42}"
: "${TEST_MAX_TURNS:=32}"; : "${TEST_TIMEOUT:=360}"

MODEL="${MODEL:-${SERVED_MODEL_NAME:-$MODEL_REPO}}"
BASE="${BASE:-http://localhost:$PORT}"
TOKENIZER="${TOKENIZER:-${TEST_TOKENIZER:-$HF_CACHE/hub/models--${MODEL_REPO//\//--}/snapshots/$MODEL_REVISION}}"
BACKEND_KWARGS='{"chat_template_kwargs": {"thinking": true, "reasoning_effort": "high"}, "temperature": 1.0, "top_p": 1}'

BENCH="$(tool-eval-bench --version 2>/dev/null | awk '{print $2}')"
LOG="/var/tmp/hardmode-repl-$(date +%Y%m%d-%H%M%S).log"

echo "── e2e hardmode replication"
echo "   model    : $MODEL"
echo "   endpoint : $BASE"
echo "   bench    : $BENCH      <- quote this with any score"
echo "   log      : $LOG"
echo "   parallel : $TEST_PARALLEL   temp 0   seed $TEST_SEED   max-turns $TEST_MAX_TURNS   timeout $TEST_TIMEOUT   --hardmode"

# refuse to measure against a dead server
curl -s --max-time 8 "$BASE/v1/models" >/dev/null 2>&1 || { echo "   ERROR: $BASE not serving"; exit 1; }

START=$(date +%s)
tool-eval-bench --base-url "$BASE" --tokenizer "$TOKENIZER" \
  --parallel "$TEST_PARALLEL" --temperature 0 --seed "$TEST_SEED" \
  --max-turns "$TEST_MAX_TURNS" --timeout "$TEST_TIMEOUT" --hardmode \
  --backend-kwargs "$BACKEND_KWARGS" > "$LOG" 2>&1
RC=$?
END=$(date +%s)

echo
echo "── RESULT"
echo "   exit     : $RC"
echo "   e2e time : $((END-START))s  ($(( (END-START)/60 ))m $(( (END-START)%60 ))s)"
grep -E 'Final Score|Total Points|Rating|Quality:|Responsiveness:|Deployability' "$LOG" | sed 's/^/   /'
echo "   bench    : $BENCH"
echo
echo "   raw log  : $LOG"
