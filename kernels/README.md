# Kernels

统一管理可复用 kernel 实现，避免学习目录与实战代码混杂。

- `gemm/`
  - `bf16_gemm_naive.cu`
  - `bf16_gemm_tiled.cu`
- `attention/`
  - `flash_attn_fwd_baseline.cu`
- `quant/`

快速试跑：
```bash
bash scripts/run_kernels_stage1.sh
```
