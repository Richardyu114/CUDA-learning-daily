# 06-bf16-gemm

目标：从 naive GEMM 进阶到 Tensor Core BF16 GEMM。

建议子任务：
1. `gemm_naive` ✅（已创建：`learning/06-bf16-gemm/gemm_naive`）
2. `gemm_coalesced` ✅（已创建：`learning/06-bf16-gemm/gemm_coalesced`）
3. `gemm_tiled_shared` ✅（已创建：`learning/06-bf16-gemm/gemm_tiled_shared`）
4. `gemm_vectorized`
5. `gemm_bf16_tensorcore`
6. `gemm_bf16_epilogue_fusion`（bias / activation）

## Lesson 1 快速开始
```bash
cd learning/06-bf16-gemm/gemm_naive
make
./naive_gemm 512 512 512 50
```

## Lesson 2 快速开始
```bash
cd learning/06-bf16-gemm/gemm_coalesced
make
./lesson2_gemm_coalesced 1024 1024 1024 50
```

## Lesson 3 快速开始
```bash
cd learning/06-bf16-gemm/gemm_tiled_shared
make
./lesson3_gemm_tiled_shared 1024 1024 1024 50
```

