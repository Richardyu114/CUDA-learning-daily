# 09-arch-hopper

目标：利用 Hopper 架构特性提升 kernel 表现。

重点方向：
- WGMMA（warpgroup MMA）
- TMA（Tensor Memory Accelerator）
- 异步 pipeline（cp.async / mbarrier 等）
- persistent kernel / cluster 相关策略
