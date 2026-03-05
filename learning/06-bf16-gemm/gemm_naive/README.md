# Lesson 1 - Naive GEMM Kernel (CUDA)

这个 lesson 的目标不是追求最快，而是建立三件事：
1. 能跑通 GEMM kernel
2. 能做 correctness 验证
3. 能拿到 baseline 性能数字（后续优化对照用）

## 文件说明
- `naive_gemm.cu`：每线程计算一个 `C[row, col]`
- `Makefile`：编译与运行

## 编译与运行
```bash
cd learning/06-bf16-gemm/gemm_naive
make
./naive_gemm 512 512 512 50
```
参数：
- `M N K`：矩阵尺寸
- `iters`：计时迭代次数

## 输出解读
- `[correctness] max_abs_err`：和 CPU reference 的最大绝对误差
- `[perf] avg_kernel_ms / GFLOPS`：当前 naive baseline 性能

## 下一步（Lesson 2）
- 优化 global memory 访问（coalescing）
- 目标：在保持正确性的前提下，提高 GFLOPS
