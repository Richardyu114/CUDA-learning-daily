#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "[stage1] build gemm kernels"
(
  cd "$ROOT/kernels/gemm"
  make
)

echo "[stage1] build attention kernel"
(
  cd "$ROOT/kernels/attention"
  make
)

echo "[stage1] run bf16_gemm_naive"
"$ROOT/kernels/gemm/bf16_gemm_naive" 128 128 128 || true

echo "[stage1] run bf16_gemm_tiled"
"$ROOT/kernels/gemm/bf16_gemm_tiled" 128 128 128 || true

echo "[stage1] run flash_attn_fwd_baseline"
"$ROOT/kernels/attention/flash_attn_fwd_baseline" 1 2 32 64 || true

echo "[stage1] run pytorch reference (if available)"
python3 "$ROOT/tests/numeric/flash_attn_ref.py" --b 1 --h 2 --s 32 --d 64 --causal || true

echo "done"
