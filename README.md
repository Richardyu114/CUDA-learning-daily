# CUDA Learning Daily

> 从零系统学习 CUDA Kernel 的日更仓库（Plan + Notes + Code + Bench）。

## 目标（8 周）

- 从基础 CUDA 编程模型走到可读写高性能 kernel
- 建立“**实现 -> 验证 -> benchmark -> 复盘**”的固定训练闭环
- 完成一组可展示的 mini kernels（含性能对比与优化记录）

---

## 学习路线（Refined Plan）

### Phase 0 — 环境与方法（Day 1~2）
- [ ] 配置 CUDA 开发环境（驱动、nvcc、nvidia-smi）
- [ ] 跑通第一个 kernel（vector add）
- [ ] 建立 benchmark 基线（包含 CPU 对照）
- [ ] 固定记录模板（问题、假设、实验、结论）

### Phase 1 — CUDA 基础与正确性（Week 1~2）
- [ ] thread/block/grid 索引与边界处理
- [ ] host/device memory 管理与 memcpy
- [ ] kernel launch 配置与 occupancy 初步理解
- [ ] error check 与单元测试习惯

**交付物**
- [ ] `vector_add`
- [ ] `saxpy`
- [ ] `reduce_sum`（基础版）

### Phase 2 — Memory-Centric 优化（Week 3~4）
- [ ] global memory coalescing
- [ ] shared memory 基本模式（tiling）
- [ ] bank conflict 识别与规避
- [ ] pinned memory / async copy / stream 基础

**交付物**
- [ ] `matmul_naive` vs `matmul_tiled`
- [ ] `transpose_naive` vs `transpose_optimized`

### Phase 3 — 并行模式与性能分析（Week 5~6）
- [ ] reduction 优化套路（unroll、warp-level）
- [ ] prefix sum / scan（可选）
- [ ] warp divergence 分析
- [ ] 用 Nsight Systems / Nsight Compute 做瓶颈定位

**交付物**
- [ ] `reduce_sum_optimized`
- [ ] 至少 2 份 profile 报告 + 复盘文档

### Phase 4 — 实战与可迁移能力（Week 7~8）
- [ ] 选 1~2 个真实场景 kernel（如 softmax / layernorm 简化版）
- [ ] 写出可复现 benchmark（输入规模、GPU 型号、配置）
- [ ] 总结优化 checklist（下次新 kernel 可复用）

**交付物**
- [ ] 1 个“从 baseline 到优化版”的完整案例
- [ ] Final report（思路、曲线、教训、下一步）

---

## 每日最小闭环（Daily Loop）

每天至少完成以下 4 件事：

1. **实现**：写一个最小可运行 kernel / 优化点
2. **验证**：和 CPU 或 reference 对比正确性
3. **测试**：记录关键指标（时间、吞吐、占用）
4. **复盘**：写下今天有效/无效的假设与原因

> 建议：每天保证一个小而完整的“可提交增量”，避免只看资料不落地。

---

## 仓库结构

```text
CUDA-learning-daily/
├── README.md
├── learning/
│   ├── 00-setup/               # 环境配置与工具检查
│   ├── 01-fundamentals/        # thread/block/grid, memory basics
│   ├── 02-memory-optimization/ # coalescing/shared-memory/bank-conflict
│   ├── 03-parallel-patterns/   # reduction/scan/warp-level
│   ├── 04-profiling/           # Nsight 结果与分析
│   └── 05-projects/            # 实战 mini project
├── notes/
│   ├── daily/                  # 日更学习日志
│   └── weekly/                 # 周复盘
├── benchmarks/
│   ├── raw/                    # 原始结果数据
│   └── reports/                # 汇总图表/分析
└── scripts/
    └── env_check.sh            # 基础环境检查脚本
```

---

## 建议里程碑（可打勾）

### Milestone A（第 2 周末）
- [ ] 完成 3 个基础 kernel
- [ ] 有统一 benchmark 框架
- [ ] 能解释 block size 对性能的影响（初步）

### Milestone B（第 4 周末）
- [ ] 完成 matmul/transpose 的 naive vs optimized 对比
- [ ] 能清楚说明 coalescing 与 shared memory 的收益来源

### Milestone C（第 6 周末）
- [ ] reduction 达到明显加速（相对 baseline）
- [ ] 掌握至少一种 profile 工具的核心视图

### Milestone D（第 8 周末）
- [ ] 形成 1 个可展示案例（含图和结论）
- [ ] 输出个人 CUDA kernel 优化 checklist

---

## Daily Note 模板（建议）

可复制到 `notes/daily/YYYY-MM-DD.md`：

```md
# YYYY-MM-DD

## 今日目标
- 

## 实现内容
- 

## 正确性验证
- 输入：
- 对照：
- 结论：

## 性能结果
- GPU：
- 问题规模：
- baseline：
- current：
- speedup：

## 关键观察
- 

## 明日计划
- 
```

---

## 使用建议

- 先追求正确，再追求快
- 一次只改一个变量（如 block size / memory layout）
- 所有“感觉更快”的结论都要有数据
- 失败实验也写下来，价值很高

---

## License

MIT
