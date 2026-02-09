# 07-flash-attention

目标：手写并验证 Flash-Attention 风格 kernel（先 fwd，再 bwd）。

建议子任务：
1. online softmax 基础
2. block-wise QK^T + softmax + PV
3. 支持 causal mask
4. 支持 bf16/fp16 输入 + fp32 accumulate
5. 与 PyTorch / reference 实现做数值与性能对比
