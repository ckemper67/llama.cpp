#!/usr/bin/env bash
# Launch straight llama-server with one speculative config for A/B benchmarking.
# Usage:  ./serve.sh <dflash|mtp|none> <n_max> [p_min]
# Example: ./serve.sh dflash 8
#          ./serve.sh dflash 12 0.4
#          ./serve.sh mtp 4
#          ./serve.sh none 0
#
# Logs to ./server.log (the draft-acceptance stats are printed there per request).
# Stop with: ./stop.sh   (or Ctrl-C if run in foreground)

set -euo pipefail
cd "$(dirname "$0")"

SERVER=~/work/llama.cpp/build/bin/llama-server
# Override BASE/DFLASH via env to benchmark a different model pair.
BASE=${BASE:-/Users/ckemper/Models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-IQ4_NL.gguf}
DFLASH=${DFLASH:-/Users/ckemper/Models/dflash/Qwen3.6-35B-A3B-DFlash-Q8_0.gguf}
PORT=8080

MODE="${1:?mode required: dflash|mtp|none}"
NMAX="${2:-8}"
PMIN="${3:-}"

# Common flags mirror the [*] + qwen sampling block from models.ini.
# ctx-size trimmed to 16384 for fast load/iteration (decode t/s is ctx-insensitive).
COMMON=(
  -m "$BASE"
  --host 127.0.0.1 --port "$PORT"
  -ngl 99
  -fa on
  --cache-type-k q8_0 --cache-type-v q8_0
  --cache-reuse 256
  -kvu
  --ctx-size 16384
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 1.5
  --reasoning-format none   # avoid content-parser 500s on channel/reasoning tokens
  ${EXTRA_ARGS:-}
)

SPEC=()
case "$MODE" in
  dflash)
    SPEC=( --spec-draft-model "$DFLASH" --spec-type draft-dflash
           --spec-draft-n-max "$NMAX" --spec-draft-ngl all )
    [ -n "$PMIN" ] && SPEC+=( --spec-draft-p-min "$PMIN" )
    ;;
  mtp)
    SPEC=( --spec-type draft-mtp --spec-draft-n-max "$NMAX" )
    ;;
  none)
    SPEC=()
    ;;
  *) echo "unknown mode: $MODE" >&2; exit 2;;
esac

echo "=== serving mode=$MODE n_max=$NMAX p_min=${PMIN:-default} ===" > server.log
# Foreground exec: the caller runs this as a tracked background process.
exec "$SERVER" "${COMMON[@]}" "${SPEC[@]}" >>server.log 2>&1
