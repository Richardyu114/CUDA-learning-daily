# CUDA Learning Daily

> 目标导向版 CUDA Kernel 学习仓库：最终能独立实现 **Flash-Attention / BF16 GEMM / Quant Attn+GEMM**，并能针对 **Hopper -> Blackwell -> Rubin(后续)** 做架构化优化。

---

## North Star（最终能力）

你最终要达到的不是“会写几个 demo kernel”，而是：

1. 能从数学与访存角度设计 kernel（不是只会改参数）
2. 能做端到端验证：正确性、吞吐、延迟、稳定性、可复现
3. 能按架构特性做实现分层（sm80/sm90/sm100+）
4. 能把优化方法沉淀成模板，迁移到新算子

---

## 12~20 周学习地图（Refined）

> 时间可伸缩：每周 5~10h 偏 20 周；每周 15~20h 可压到 12~14 周。

### Stage A — 基础打牢（Week 1~3）
- CUDA execution model、memory hierarchy、同步语义
- 建立 benchmark/verify 工具链（你已经开始了）
- 完成基础 kernel：vector_add / saxpy / reduce（base + optimized）

**退出条件**
- [ ] 能稳定跑 benchmark 并记录结果
- [ ] 能解释至少 3 类常见性能瓶颈（带宽、占用、分歧）

### Stage B — GEMM 路线（Week 4~8）
- naive GEMM -> tiled shared GEMM -> vectorized load/store
- Tensor Core 路线：BF16 输入、FP32 accumulate
- 尝试 epilogue fusion（bias/activation）

**交付物**
- [ ] `gemm_naive`
- [ ] `gemm_tiled`
- [ ] `gemm_bf16_tensorcore`
- [ ] 与 cuBLAS 的基线对比（至少 2~3 个 shape）

### Stage C — Flash-Attention 路线（Week 9~13）
- online softmax
- block-wise QK^T + softmax + PV
- causal/non-causal 两模式
- 混合精度（fp16/bf16 in, fp32 acc）

**交付物**
- [ ] Flash-Attention forward kernel（先 correctness）
- [ ] 性能对比（PyTorch SDPA / flash-attn baseline）
- [ ] 至少 1 版优化迭代说明（profile 证据）

### Stage D — Quantized Kernel 路线（Week 14~17）
- int8/fp8（按硬件能力）量化路径
- scale/zero-point 策略（per-tensor/per-channel）
- dequant + matmul / attention 融合

**交付物**
- [ ] quant GEMM baseline
- [ ] quant attention baseline
- [ ] 准确性与吞吐权衡报告

### Stage E — 架构专项（Week 18+）
- Hopper 特性优先：WGMMA / TMA / async pipeline
- Blackwell 特性跟进：迁移与重调优
- Rubin（后续公开信息）持续跟踪

**交付物**
- [ ] Hopper-specific kernel 版本
- [ ] architecture tuning checklist
- [ ] 同一 workload 的跨架构性能对比表

---

## 你要做的 3 条主线（并行推进）

### 主线 1：Kernel 实现线
按 `baseline -> optimized -> architecture-specific` 推进。

### 主线 2：验证与基准线
每个 kernel 必带：
- 正确性对照（CPU/reference）
- 固定 shape 套件
- profile 证据（ncu/nsys）

### 主线 3：架构与文档线
每次优化写清：
- 为什么有效（访存/指令/并发）
- 在哪类 shape 最有效
- 在哪个架构收益最大

---

## 仓库结构（升级版）

```text
CUDA-learning-daily/
├── README.md
├── learning/
│   ├── 00-setup/
│   ├── 01-fundamentals/
│   ├── 02-memory-optimization/
│   ├── 03-parallel-patterns/
│   ├── 04-profiling/
│   ├── 05-projects/
│   ├── 06-bf16-gemm/             # BF16 / Tensor Core GEMM 路线
│   ├── 07-flash-attention/       # Flash-Attention 路线
│   ├── 08-quant-kernels/         # Quant GEMM/Attn 路线
│   ├── 09-arch-hopper/           # Hopper 特性专项
│   └── 10-arch-blackwell-rubin/  # Blackwell + 后续架构专项
├── kernels/
│   ├── gemm/                     # 统一放置 GEMM kernels
│   ├── attention/                # 统一放置 attention kernels
│   └── quant/                    # 统一放置 quant kernels
├── tests/
│   ├── unit/                     # 小规模功能正确性
│   ├── numeric/                  # 数值精度/误差边界
│   └── perf/                     # 固定shape性能回归
├── benchmarks/
│   ├── raw/                      # 原始 benchmark 输出
│   └── reports/                  # 可读报告
├── profiler/
│   ├── ncu/                      # Nsight Compute capture/notes
│   └── nsys/                     # Nsight Systems capture/notes
├── docs/
│   ├── architecture-notes/       # 架构特性笔记
│   └── playbooks/                # 调优套路沉淀
├── notes/
│   ├── daily/
│   └── weekly/
└── scripts/
    ├── env_check.sh
    ├── run_bench.sh
    └── compare_reduce.sh
```

---

## 每周输出要求（强约束）

每周至少交付：
1. 1 个可运行 kernel 增量
2. 1 份 benchmark 记录（raw + report）
3. 1 条 profile 证据（截图或关键 metric）
4. 1 条“失败尝试与教训”

---

## 关键指标（KPI）

- **Correctness**：误差是否在预期范围（不同 dtype 分开记录）
- **Performance**：kernel time / TFLOPS / GBps / speedup
- **Stability**：多次运行抖动（P50/P90）
- **Portability**：跨架构迁移后的性能保留率

---

## 里程碑（面向你的最终目标）

### Milestone M1（基础完成）
- [ ] Fundamentals 全部可复现
- [ ] 有统一 benchmark 与报告格式

### Milestone M2（BF16 GEMM 可用）
- [ ] 有 BF16 Tensor Core GEMM 可运行版本
- [ ] 对比 cuBLAS 有明确差距认知与优化方向

### Milestone M3（Flash-Attention 可用）
- [ ] Forward kernel 正确 + 有性能数据
- [ ] 至少一轮优化并有 profile 支撑

### Milestone M4（Quant Kernel 初版）
- [ ] Quant GEMM / Quant Attention 都有 baseline
- [ ] 有精度-性能 tradeoff 报告

### Milestone M5（架构专项）
- [ ] Hopper 特性落地至少 1 个案例
- [ ] Blackwell/Rubin 跟进机制建立（feature matrix + tuning notes）

---

## 推荐执行顺序（今天就能继续）

1. 继续完善 `01-fundamentals` 的 benchmark 稳定性
2. 开始 `06-bf16-gemm`：先做 naive/tiled，再切 Tensor Core
3. 同时在 `docs/architecture-notes/` 建立 Hopper 特性笔记
4. 等 GEMM 路线稳定后推进 `07-flash-attention`

---

## License

MIT
