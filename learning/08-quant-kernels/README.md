# 08-quant-kernels

目标：实现 quantized attention / GEMM 的关键路径。

建议子任务：
1. per-tensor / per-channel scale
2. int8/fp8 路径（按硬件能力）
3. dequant + matmul 融合
4. quantized attention 核心算子拆解与融合
