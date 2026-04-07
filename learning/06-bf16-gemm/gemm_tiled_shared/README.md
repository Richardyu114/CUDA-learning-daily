# Lesson 3 - GEMM Shared Memory Tiling

这一节目标：在 Lesson 2 的 coalesced global-memory 访问基础上，再进一步减少对 global memory 的重复读取。

## 核心思路

- Lesson 2：主要解决 **读 B 时的 warp 内连续访问**
- Lesson 3：进一步把一个 `K` 分块搬进 shared memory，让一个 block 内的线程复用这块数据

为什么会更快？
- 旧路径里，同一个 `A[row, k]` / `B[k, col]` 会被很多线程反复从 global memory 读取
- tiled shared 先把 `A tile` 和 `B tile` 搬到 shared memory
- block 内线程随后在 shared memory 上做多次乘加，减少 global load 压力

## 本节对比

本文件内置两条路径：
- `gemmCoalesced32x8`：Lesson 2 风格，对照组
- `gemmTiledShared16x16`：Lesson 3，共享内存 tile 版本

## 文件
- `lesson3_gemm_tiled_shared.cu`：包含 correctness + benchmark + Lesson2/3 对比
- `Makefile`

## 编译与运行
```bash
cd learning/06-bf16-gemm/gemm_tiled_shared
make
./lesson3_gemm_tiled_shared 1024 1024 1024 50
```

## 输出解读
- correctness：`max_abs_err`
- Lesson2 vs Lesson3：各自 `avg ms` + `GFLOPS`
- speedup：`Lesson2 / Lesson3`

## 这一版你要真正看懂什么

1. `K` 维为什么要按 tile 分块
2. 为什么 `__syncthreads()` 要出现两次
3. shared memory 复用，具体帮我们省掉了哪些 global load
4. 为什么这里先选 `16x16`，而不是一上来追更大 tile

## 下一步（Lesson 4）
- 做 vectorized load/store（例如 `float4`）
- 看 shared-memory tiled 版本里，global load 是否还能再压缩
- 为后续 BF16 / Tensor Core 版本铺路
