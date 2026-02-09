# 10-arch-blackwell-rubin

目标：跟进 Blackwell 以及后续架构（如 Rubin）特性，将已有 kernel 迁移/重调优。

建议方式：
- 维护 feature matrix（架构 -> 可用特性）
- 每个 kernel 保留 architecture-specific tuning notes
- 对同一 workload 记录跨架构对比
