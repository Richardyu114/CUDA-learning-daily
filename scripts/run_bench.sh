#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPORT_DIR="$ROOT_DIR/benchmarks/raw"
mkdir -p "$REPORT_DIR"

TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$REPORT_DIR/bench_$TS.txt"

echo "[bench] output -> $OUT"

echo "===== ENV =====" | tee -a "$OUT"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi | tee -a "$OUT"
else
  echo "nvidia-smi not found" | tee -a "$OUT"
fi

echo "\n===== VECTOR_ADD =====" | tee -a "$OUT"
(
  cd "$ROOT_DIR/learning/01-fundamentals/vector_add"
  make
  ./vector_add
) | tee -a "$OUT"

echo "\n===== SAXPY =====" | tee -a "$OUT"
(
  cd "$ROOT_DIR/learning/01-fundamentals/saxpy"
  make
  ./saxpy
) | tee -a "$OUT"

echo "\n===== REDUCE_SUM (BASE + OPT) =====" | tee -a "$OUT"
(
  cd "$ROOT_DIR/learning/01-fundamentals/reduce_sum"
  make
  ./reduce_sum
  ./reduce_sum_opt
) | tee -a "$OUT"

echo "\n===== REDUCE SPEEDUP =====" | tee -a "$OUT"
(
  bash "$ROOT_DIR/scripts/compare_reduce.sh"
) | tee -a "$OUT"

echo "\nDone. Raw bench saved to: $OUT"
