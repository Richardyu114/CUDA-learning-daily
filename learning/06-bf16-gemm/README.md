# 06-bf16-gemm

目标：从 naive GEMM 进阶到 Tensor Core BF16 GEMM。

建议子任务：
1. `gemm_naive` ✅（已创建：`learning/06-bf16-gemm/gemm_naive`）
2. `gemm_tiled_shared`
3. `gemm_vectorized`
4. `gemm_bf16_tensorcore`
5. `gemm_bf16_epilogue_fusion`（bias / activation）

## Lesson 1 快速开始
```bash
cd learning/06-bf16-gemm/gemm_naive
make
./naive_gemm 512 512 512 50
```

