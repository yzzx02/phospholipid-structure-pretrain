# ROCm Math attention：已完成的 GPU 工程验证（2026-10-19）

本轮已经实测 Math SDPA + 关闭 MHA fastpath 的可运行路径；不能据此宣称已彻底修复驱动，或硬件一定没有问题。原始失败可能涉及 fused attention 路径，尚未独立证明唯一根因。

## 环境与代码

RX 9070 XT，Windows driver 32.0.31041.1004，Ubuntu 24.04.4 / WSL kernel 6.18.33.1，ROCm 7.2.1 / ROCDXG 1.2.0，Python 3.12.13，Torch 2.9.1 ROCm。核心版本未替换。

`utils/attention.py` 在作用域内选择 Math SDPA 并关闭 MHA fastpath；退出时恢复原状态。`scripts/run_rocm.sh` 固定 ROCm 解释器并设置 DXG 检测。GPU RNG 恢复使用 CPU ByteTensor。Checkpoint 核验将权重/buffer 严格一致与 GPU 输出容差分开，FP32 使用 PyTorch 默认容差，CPU 保持严格相等。

## 已实测结果

|指标|Morgan MLP|GINE|
|---|---:|---:|
|32 个真实 TRAIN 结构过拟合 steps|200|200|
|过拟合 EOS|100%|100%|
|过拟合合法 Polar / Polar exact|100% / 100%|93.75% / 93.75%|
|过拟合 chain set exact|100%|100%|
|GPU smoke optimizer steps|1,000 + 实际恢复 1 步|1,000 + 实际恢复 1 步|
|训练池 / validation pool|20,000 / 2,000|20,000 / 2,000|
|实际已消费的不同训练目标|16,016|16,016|
|Validation token accuracy (2,000)|69.85%|68.91%|
|Validation teacher-forced chain exact (2,000)|3.60%|3.70%|
|Free generation EOS (128)|96.09%|96.09%|
|合法 Polar / assembly (128)|26.56%|20.31%|
|Polar exact / Top-5 (128)|0% / 0%|0% / 0%|
|Chain set exact (128)|17.97%|17.19%|
|Full formula match / full graph exact (128)|0% / 0%|0% / 0%|
|续训进程 allocated VRAM 峰值|0.692 GiB|0.660 GiB|

276 项回归、Ruff、数据冻结文件哈希核验通过。原始默认 attention 过拟合时出现黑屏和 Windows LiveKernelEvent 141 / watchdog 记录并停止。后续受监控 Math 运行没有新增 GPU 转储；这只描述已执行的窗口。原始 FP 的 CPU 头基规则装载被无日志监控误停，随后分批装载并恢复完成。原始 GINE 的输出比较因约 1e-6 的舍入差停止，随后增加严格权重核验并采用官方 FP32 输出容差，实际续训通过。所有中断记录在本地保留。

## 尚未证明与下一步

32 条过拟合只证明可记忆并生成 EOS，不证明未见头基泛化。原始 128 条自由生成仍无 polar/full exact。尚未执行全量多轮结构预训练、MS/MS 微调或 Math 路径 AMP 中程验证。原始资源文件仅覆盖续训进程，不能用一优化步的速率代表完整 1,000 步吞吐。

下一轮先冻结原 128 条及分层诊断验证集，补充化学失败类别、teacher-forced EOS、生成长度及模板 H/C/HC 关系。保留现有模型、训练池、公式统计、种子和 50/50 模板模式；从 1,001 步继续至 1,500、2,000，根据固定指标和 GPU 稳定性决定是否延长到 3,000/5,000。不会自动启动完整预训练。

完整聚合数据见 [JSON](../results/ROCM_MATH_ATTENTION_20261019.json)，方法和本地历史见 [执行记录](../ROCM_VALIDATION_2026-10-19.md)。本报告不包含样本 ID、分子、逐目标检索、checkpoint、私有日志、转储或本机用户路径。
