#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REDUCE_DIR="$ROOT_DIR/learning/01-fundamentals/reduce_sum"

N="${1:-16777216}"

echo "[compare_reduce] n=$N"

cd "$REDUCE_DIR"
make

BASE_OUT=$(./reduce_sum "$N")
OPT_OUT=$(./reduce_sum_opt "$N")

echo "===== BASELINE ====="
echo "$BASE_OUT"

echo "===== OPTIMIZED ====="
echo "$OPT_OUT"

BASE_MS=$(echo "$BASE_OUT" | awk -F': ' '/gpu_ms\(avg\)/ {print $2}')
OPT_MS=$(echo "$OPT_OUT" | awk -F': ' '/gpu_ms\(avg\)/ {print $2}')

if [[ -z "$BASE_MS" || -z "$OPT_MS" ]]; then
  echo "failed to parse gpu_ms from outputs"
  exit 1
fi

python3 - <<PY
base_ms = float("$BASE_MS")
opt_ms = float("$OPT_MS")
speedup = base_ms / opt_ms if opt_ms > 0 else float('inf')
print("===== SPEEDUP =====")
print(f"baseline_gpu_ms: {base_ms:.6f}")
print(f"optimized_gpu_ms: {opt_ms:.6f}")
print(f"speedup: {speedup:.3f}x")
PY
