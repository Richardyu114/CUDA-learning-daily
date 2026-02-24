# Lesson 01: Vector Add — 理解 GPU 执行模型

## 为什么从这里开始

Vector add 是 GPU 编程的 "Hello World"，但别小看它——它能引出 GPU 编程中最重要的几个概念：

1. **Host vs Device** — CPU 和 GPU 是两个独立的处理器，各有各的内存
2. **Thread 层次结构** — grid → block → thread，GPU 的并行模型
3. **Memory 搬运** — cudaMemcpy 是昂贵的，减少搬运是优化的核心
4. **Memory-bound vs Compute-bound** — vector add 是典型的 memory-bound

## 代码结构

```
01_vector_add.cu
├── vector_add_v1  — 最基础版本：一个 thread 一个元素
├── vector_add_v2  — grid-stride loop：更少的 thread，每个做更多工作
├── vector_add_cpu — CPU 参考实现（用于验证正确性）
└── main           — 计时 + 验证 + 思考题
```

## 编译 & 运行

```bash
cd kernels/01_vector_add
nvcc -o 01_vector_add 01_vector_add.cu
./01_vector_add
```

## 核心概念

### Thread 层次

```
Grid (整个 GPU 任务)
├── Block 0
│   ├── Thread 0
│   ├── Thread 1
│   └── ...Thread 255
├── Block 1
│   ├── Thread 0
│   └── ...
└── Block N
    └── ...
```

每个 thread 通过 `blockIdx.x * blockDim.x + threadIdx.x` 算出自己负责的全局 index。

### Grid-stride loop（V2）

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
int stride = blockDim.x * gridDim.x;
for (int i = idx; i < n; i += stride) {
    c[i] = a[i] + b[i];
}
```

这个模式在 vLLM kernel 中非常常见。好处：
- 不需要精确计算 grid 大小
- 一个 kernel 适配任意大小的输入
- 更好的 load balancing

### 与 vLLM 的关联

- `cudaMalloc` / `cudaMemcpy` → vLLM 中 KV cache 的显存管理和搬运
- grid-stride loop → vLLM kernel 中处理 variable-length 序列的标准模式
- CUDA event timing → vLLM benchmark 中的计时方式

## 思考题

1. V1 和 V2 性能差多少？为什么？
2. 改 `threads_per_block` 为 64/128/512/1024，哪个最快？
3. 增大 N 到 256M，观察 memcpy 时间占比的变化

## 下一课预告

**Lesson 02: Matrix Transpose — 理解 Shared Memory 和 Bank Conflict**

矩阵转置看起来简单，但 naive 实现会有严重的 uncoalesced memory access。我们会用 shared memory 来优化它，这也是 attention kernel 中 tile-based 计算的基础。
