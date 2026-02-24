# CUDA 核心概念速览

> 阅读时间：约 15-20 分钟
> 目标：建立最小必要的概念框架，让后续写 kernel 时不会懵
> 
> 类比原则：你已经懂 vLLM 的框架层，这里会不断把 GPU 概念映射到你熟悉的东西上。

---

## 1. GPU 硬件结构：一个"大工厂"

把 GPU 想象成一个工厂：

```
GPU（工厂）
├── SM 0（车间）  — Streaming Multiprocessor
│   ├── 128 CUDA Cores（工人）
│   ├── Shared Memory（车间内共享工具柜，~100KB）
│   ├── L1 Cache
│   ├── Register File（每个工人自己的工具箱）
│   └── Warp Schedulers（车间主任，4 个）
├── SM 1（车间）
├── ...
├── SM N（车间）
├── L2 Cache（工厂级仓库，~40MB）
└── Global Memory / HBM（工厂外的大仓库，~80GB，但搬运慢）
```

### 关键数字感受（以 A100 为例）

| 层级 | 大小 | 带宽 | 类比 |
|------|------|------|------|
| Register | 256KB / SM | ~19 TB/s | 手边的工具 |
| Shared Memory | 100KB / SM | ~19 TB/s | 车间内共享柜 |
| L1 Cache | 128KB / SM | — | 车间小缓存 |
| L2 Cache | 40MB | ~5 TB/s | 工厂仓库 |
| HBM (Global) | 80GB | ~2 TB/s | 外部大仓库 |

**核心认知：离计算单元越近的存储越快越小，越远越慢越大。**

这就是为什么 vLLM 的 KV cache 管理那么重要——本质上是在管理 GPU 的"仓库系统"。

---

## 2. 线程层次：Grid → Block → Thread → Warp

你在 CPU 上写代码是单线程思维（或少量多线程）。GPU 是**大规模并行**——同时启动成千上万个线程。

### 层次结构

```
Kernel Launch（一次 GPU 函数调用）
│
└── Grid（所有 thread 的集合）
    ├── Block 0（一组 thread，最多 1024 个）
    │   ├── Warp 0 (Thread 0-31)    ← 32 个 thread 锁步执行
    │   ├── Warp 1 (Thread 32-63)
    │   └── ...
    ├── Block 1
    └── Block N
```

### 每一层的关键特性

| 层级 | 调度单位 | 共享什么 | 大小限制 |
|------|---------|---------|---------|
| Grid | 整个 kernel | Global Memory | blocks 数量几乎无限 |
| Block | 分配到 SM 上 | Shared Memory | 最多 1024 threads |
| Warp | **实际执行单位** | 指令流（SIMT） | 固定 32 threads |
| Thread | 逻辑最小单位 | 自己的 Register | — |

### 最重要的一点：Warp

**Warp 是 GPU 执行的基本单位。** 32 个 thread 组成一个 warp，它们在同一时刻执行相同的指令（SIMT: Single Instruction, Multiple Threads）。

这意味着：
- 如果 warp 中 16 个 thread 走 if 分支，16 个走 else 分支 → **warp divergence**，两个分支串行执行，效率减半
- 如果 warp 中的 32 个 thread 访问的内存地址是连续的 → **coalesced access**，一次事务搞定
- 如果地址不连续 → 多次事务，带宽浪费

**类比 vLLM**：continuous batching 的核心也是类似的——把不同请求的 token 紧凑地排列，让 GPU 的每次计算都是"满的"，不浪费。

---

## 3. 内存层次：速度 vs 容量的权衡

### 每种内存的使用场景

```
Register（最快）
  └─ 每个 thread 私有的局部变量
  └─ 自动分配，不需要手动管理
  └─ 数量有限，用多了会"溢出"到 local memory（很慢）

Shared Memory（很快）
  └─ 同一 Block 内的 thread 共享
  └─ 需要手动管理（__shared__ 声明）
  └─ 典型用途：tile-based 计算时暂存数据
  └─ 类比：FlashAttention 中，把 Q/K/V 的 tile 加载到 shared memory 再计算

Global Memory / HBM（大但慢）
  └─ 所有 thread 都能访问
  └─ cudaMalloc 分配的就是这个
  └─ 类比：vLLM 中 KV cache 的主存储
```

### Coalesced Memory Access（合并访存）

这是 GPU 编程中**最重要的优化原则之一**。

```
✅ 好（Coalesced）:
Thread 0 → addr[0]
Thread 1 → addr[1]
Thread 2 → addr[2]
...
→ 32 个 thread 的请求合并成 1 次 128-byte 事务

❌ 坏（Strided）:
Thread 0 → addr[0]
Thread 1 → addr[128]
Thread 2 → addr[256]
...
→ 32 个 thread 需要 32 次独立事务，带宽浪费 32x
```

**这就是矩阵转置要用 shared memory 的原因**——naive 版本的写入是 strided 的，用 shared memory 做一次中转就能变成 coalesced。

---

## 4. Kernel 执行模型

### 一个 kernel 从启动到完成的过程

```
1. CPU 调用 kernel<<<grid, block>>>(args)
   → CPU 把 kernel 参数和配置发给 GPU
   → CPU 不等待，立刻返回（异步！）

2. GPU 调度器把 blocks 分配到各个 SM
   → 每个 SM 可以同时运行多个 block（取决于资源）
   → block 之间没有执行顺序保证

3. SM 内部，warp scheduler 调度 warp 执行
   → 当一个 warp 等待数据（内存延迟）时，切换到另一个 warp
   → 这就是 GPU 隐藏延迟的方式（类似 CPU 超线程，但规模大得多）

4. 所有 thread 完成 → kernel 结束
   → CPU 需要 cudaDeviceSynchronize() 或 event 来确认完成
```

### Occupancy（占用率）

Occupancy = 一个 SM 上实际活跃的 warp 数 / 最大支持的 warp 数

高 occupancy → 更多 warp 可以互相掩盖延迟 → 通常性能更好
低 occupancy → warp 等待时没别的 warp 可切换 → 性能差

影响 occupancy 的因素：
- 每个 thread 用了多少 register
- 每个 block 用了多少 shared memory
- block 里有多少 thread

**不需要死记**，后面写 kernel 的时候 nvcc 和 profiler 会告诉你。

---

## 5. 性能瓶颈分类

### Memory-bound（内存受限）

```
特征：计算很少，但数据搬运很多
例子：vector add、element-wise 操作、layer norm
瓶颈：HBM 带宽
优化方向：减少内存访问次数、kernel fusion
```

### Compute-bound（计算受限）

```
特征：大量数学运算
例子：矩阵乘法（GEMM）、卷积
瓶颈：SM 的计算吞吐
优化方向：用 Tensor Core、优化算法复杂度
```

### Latency-bound（延迟受限）

```
特征：kernel 太小，GPU 都没被填满
例子：很小的 kernel 反复启动
瓶颈：kernel launch 开销、同步开销
优化方向：kernel fusion、CUDA Graph（vLLM 中大量使用！）
```

**vLLM 中的例子：**
- Attention → 曾经是 memory-bound，FlashAttention 通过 tiling 把它变成 compute-bound
- Sampling → latency-bound，所以 vLLM 用 CUDA Graph 来减少 launch 开销
- GEMM → compute-bound，交给 cuBLAS / Tensor Core

---

## 6. 概念 → 代码 映射表

| 概念 | 代码中怎么体现 | vLLM 中在哪 |
|------|---------------|------------|
| Grid/Block/Thread | `<<<blocks, threads>>>` | 每个 kernel launch |
| Thread index | `blockIdx.x * blockDim.x + threadIdx.x` | 几乎所有 kernel |
| Grid-stride loop | `for (i = idx; i < n; i += stride)` | 处理 variable-length 输入 |
| Global Memory | `cudaMalloc` / `cudaMemcpy` | KV cache 管理 |
| Shared Memory | `__shared__ float tile[SIZE]` | Attention tile 计算 |
| Warp primitives | `__shfl_sync`, `__ballot_sync` | Reduction, sampling |
| CUDA Graph | `cudaGraphLaunch` | decode 阶段批量执行 |
| Async copy | `cudaMemcpyAsync` + streams | KV cache prefetch |

---

## 接下来

概念有了，回去看 `kernels/01_vector_add/01_vector_add.cu`，应该会清晰很多。

跑完 Lesson 01 后，Lesson 02（矩阵转置）会把 **Shared Memory** 和 **Coalesced Access** 这两个概念从"知道"变成"体感"。

**记住：不用一次全记住。** 这份文档是参考手册，写 kernel 时遇到问题随时回来翻。

---

*Camus × Richard — 2026-02-24*
