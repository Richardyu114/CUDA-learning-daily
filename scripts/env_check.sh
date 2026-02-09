#!/usr/bin/env bash
set -euo pipefail

echo "[1/4] nvidia-smi"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo "nvidia-smi not found"
fi

echo "\n[2/4] nvcc --version"
if command -v nvcc >/dev/null 2>&1; then
  nvcc --version
else
  echo "nvcc not found"
fi

echo "\n[3/4] gcc --version"
if command -v gcc >/dev/null 2>&1; then
  gcc --version | head -n 1
else
  echo "gcc not found"
fi

echo "\n[4/4] CUDA sample compile test (optional)"
echo "Tip: add a tiny vector_add.cu under learning/01-fundamentals and compile with:"
echo "  nvcc -O2 vector_add.cu -o vector_add"
