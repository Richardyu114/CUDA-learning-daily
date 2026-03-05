# Lesson 2 - GEMM Global Memory Coalescing

这一节目标：在不引入 shared memory tiling 的前提下，先把 global memory 访问模式调顺。

## 核心思路

- Baseline：`block(16,16)`（Lesson1 同风格）
- Coalesced：`block(32,8)`（warp 对齐）

为什么 `32x8`？
- 一个 warp 恰好覆盖 32 个连续列（`threadIdx.x=0..31`）
- 对 `B[k, col]` 的读取在 warp 内连续，满足 coalescing
- `A[row, k]` 在同一行可被线程复用，缓存友好

## 文件
- `lesson2_gemm_coalesced.cu`：包含 baseline 和 coalesced 两个 kernel + 对比 benchmark
- `Makefile`

## 编译与运行
```bash
cd learning/06-bf16-gemm/gemm_coalesced
make
./lesson2_gemm_coalesced 1024 1024 1024 50
```

## 输出解读
- correctness：`max_abs_err`
- baseline vs coalesced：各自 `avg ms` + `GFLOPS`
- speedup：`baseline/coalesced`

> 注意：不同 GPU 架构上提升幅度会不一样。Lesson2 的重点是“理解内存访问映射”，不是追极限性能。

## 下一步（Lesson 3）
- 在 coalesced 基础上加 shared memory tile，减少重复 global load。
