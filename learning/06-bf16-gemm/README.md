# 06-bf16-gemm

目标：从 naive GEMM 进阶到 Tensor Core BF16 GEMM。

建议子任务：
1. `gemm_naive`
2. `gemm_tiled_shared`
3. `gemm_vectorized`
4. `gemm_bf16_tensorcore`
5. `gemm_bf16_epilogue_fusion`（bias / activation）
