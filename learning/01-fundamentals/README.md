# 01-fundamentals

基础 kernel 练习：
- vector add（已初始化：`vector_add/`）
- SAXPY（已初始化：`saxpy/`）
- 基础 reduce（已初始化：`reduce_sum/`）

每个练习至少包含：
- 代码
- 正确性对照
- 简单 benchmark

快速开始：
```bash
cd learning/01-fundamentals/vector_add && make && ./vector_add
cd learning/01-fundamentals/saxpy && make && ./saxpy
cd learning/01-fundamentals/reduce_sum && make && ./reduce_sum
```

批量跑 benchmark：
```bash
bash scripts/run_bench.sh
```

对比 reduce baseline vs optimized（含 speedup）：
```bash
bash scripts/compare_reduce.sh
```
