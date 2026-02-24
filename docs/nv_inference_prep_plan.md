# NV Inference 岗准备计划

> 为 Richard 定制 — 基于 vLLM 框架工程师背景，目标 NVIDIA Inference/Serving 方向

---

## 一、目标岗位分析

NV inference 相关团队主要有：

| 团队 | 方向 | 你的匹配度 |
|------|------|-----------|
| TensorRT-LLM | LLM 推理引擎（C++/CUDA） | ⭐⭐⭐⭐ 框架设计经验直接对口 |
| Triton Inference Server | 模型 serving 框架 | ⭐⭐⭐⭐⭐ 最对口，调度/batching/分布式 |
| CUDA Inference Libraries | cuDNN/cuBLAS 推理路径优化 | ⭐⭐⭐ 需要更多 kernel 功底 |
| AI Frameworks (PyTorch/JAX) | 框架集成与优化 | ⭐⭐⭐⭐ 系统视角强 |
| Solution Architect | 客户侧推理方案落地 | ⭐⭐⭐⭐ 偏应用，技术深度要求稍低 |

**建议主攻**：TensorRT-LLM 或 Triton Inference Server 团队，这两个和你 vLLM 经验最重合。

---

## 二、能力补强计划

### 2.1 CUDA / Kernel 基础（4-6 周）

**目标**：不需要成为 kernel 专家，但要能读懂、能改、能讲清楚性能瓶颈。

**第 1-2 周：概念建立**
- [ ] 读 PMPP（Programming Massively Parallel Processors）第 1-7 章
  - 重点：线程层次结构、memory hierarchy、warp 执行模型
- [ ] NVIDIA 官方 CUDA Programming Guide 的 "Programming Model" 和 "Hardware Implementation" 两节
- [ ] 理解关键概念：coalesced memory access、shared memory bank conflict、occupancy、warp divergence

**第 3-4 周：动手写**
- [ ] 完成 3-5 个小 kernel 练习：
  - vector add（入门）
  - matrix transpose（shared memory）
  - reduction（warp-level primitives）
  - softmax（numerically stable）
  - simple attention score computation
- [ ] 用 Nsight Compute profile 自己写的 kernel，学会看 profiling 报告

**第 5-6 周：读真实 kernel**
- [ ] 精读 vLLM 中 2-3 个你每天看 PR 但没深入过的 kernel：
  - `csrc/attention/` 下的 attention kernel
  - `csrc/quantization/` 下的量化 kernel
  - 一个 Triton 写的 kernel（如 MLA decode attention）
- [ ] 尝试对其中一个做小改动并跑 benchmark

### 2.2 Triton（2-3 周，可与上面并行）

**为什么重要**：NV 自己在推 Triton，vLLM 也在大量使用，面试加分项。

- [ ] 读 Triton 官方 tutorial（vector add → fused softmax → matmul → flash attention）
- [ ] 把一个简单的 CUDA kernel 用 Triton 重写，对比性能
- [ ] 理解 Triton 的 auto-tuning 机制和 tile-based programming 模型

### 2.3 TensorRT-LLM 了解（2 周）

- [ ] 跑通 TensorRT-LLM 的 quick start（一个模型从 HuggingFace → TRT-LLM → deploy）
- [ ] 读 TensorRT-LLM 的架构文档，理解它和 vLLM 的异同
  - 关注：in-flight batching、KV cache 管理、weight streaming、quantization pipeline
- [ ] 准备一个"vLLM vs TensorRT-LLM"的对比分析（面试必问）

### 2.4 系统设计（持续）

你的强项，但要准备好用 NV 的语境来讲。

- [ ] 能画出完整的 LLM serving pipeline（从请求到 token 输出）
- [ ] 能讲清楚 continuous batching、paged attention、speculative decoding 的设计权衡
- [ ] 准备 2-3 个你在 vLLM 中做的有深度的项目案例
- [ ] 分布式推理：TP/PP/EP 的权衡，什么场景用什么策略

---

## 三、面试准备

### 3.1 常见面试环节

| 环节 | 内容 | 准备策略 |
|------|------|---------|
| Coding | C++/Python，偶尔 CUDA | LeetCode medium 够用，重点练 C++ |
| System Design | LLM serving 系统设计 | 你的主场，准备 2-3 个深度方案 |
| Domain Knowledge | GPU 架构、推理优化、量化 | 补 kernel 知识 + 读 NV 近期博客 |
| Behavioral | 项目经验、团队协作 | STAR 格式准备 3-5 个故事 |

### 3.2 高频面试题（基于公开信息）

**系统设计类**：
1. 设计一个支持 100B 参数模型的推理 serving 系统
2. 如何优化 LLM 的 first token latency？
3. 设计一个 KV cache 管理系统（paged attention 变体）
4. Multi-node inference 的通信优化

**GPU/Kernel 类**：
1. 解释 GPU memory hierarchy，如何优化 memory-bound kernel？
2. Flash Attention 的核心思想是什么？为什么比 naive attention 快？
3. 什么是 quantization？FP8 vs INT8 vs INT4 的权衡？
4. 解释 tensor core 的工作原理

**对比分析类**：
1. vLLM vs TensorRT-LLM 各自的优劣？
2. Continuous batching 的实现挑战？
3. Speculative decoding 的适用场景和局限？

### 3.3 你的独特卖点（面试中要突出）

1. **日均追踪 30-70 条 vLLM PR** — 对整个推理生态的演进方向有全局视角
2. **框架层实战经验** — engine/scheduler/model runner/分布式，不是纸上谈兵
3. **对中国模型架构（MLA/MoE/GDN）的理解** — NV 非常重视中国市场和模型适配
4. **跨层理解能力** — 能把 kernel 层的优化和框架层的设计决策联系起来

---

## 四、时间线建议

```
Week 1-2:  CUDA 概念 + PMPP 精读 + C++ 刷题开始
Week 3-4:  动手写 kernel + Triton tutorial
Week 5-6:  读真实 kernel + TensorRT-LLM 上手
Week 7-8:  系统设计准备 + 模拟面试 + 投简历
```

**总计约 2 个月**，每天投入 1-2 小时，周末多一些。

---

## 五、资源清单

### 书籍
- 《Programming Massively Parallel Processors》4th Edition
- 《CUDA by Example》（更入门，可选）

### 课程
- NVIDIA DLI（Deep Learning Institute）— CUDA/Triton 课程
- Caltech CS179（GPU Computing）— 公开课，作业质量好

### 博客/文档
- NVIDIA Technical Blog（搜 inference、TensorRT-LLM）
- Lei Mao's Blog（有很多 CUDA 和推理优化的中文技术文章）
- FlashAttention 论文（Tri Dao）
- vLLM 论文（PagedAttention）

### 代码
- vLLM `csrc/` 目录 — 你最熟悉的真实 kernel 代码库
- TensorRT-LLM GitHub — 读 `cpp/tensorrt_llm/kernels/`
- Triton tutorials — `triton-lang/triton` repo

---

*由 Camus 整理 — 2026-02-24*
*祝 Richard 拿到心仪 offer 🚀*
