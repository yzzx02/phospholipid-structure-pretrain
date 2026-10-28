# 第二轮甘油脂核心模型：学习率调度对照

本轮沿用第二轮甘油＋头基模型及冻结的 8,000 模板库，不改变数据、表示或网络。TRAIN 170,872、validation 9,493、test 9,499；本轮不使用测试集调参或生成。模板均来自 TRAIN，也继续作为 TRAIN 的预测目标；屏蔽自身及等价模板。

用户在 2026-10-28 将“固定训练 48 轮”修订为图表统一显示 Epoch 1–50、后续仍按早停选权重。本轮据此采用：**至少完成 50 轮，之后连续 5 轮完整验证 CE 没有严格下降则停止，最多 80 轮**。patience 从第一轮累计，到第 50 轮不重置。最佳权重始终是全程最低完整 Validation Token CE，可以出现在第 50 轮之前或之后。第 50 轮是共同绘图窗口，不是强制终点，也不是候选权重范围的起点。

| 实验 | 调度 | 初始状态 |
|---|---|---|
| E1 | 固定 LR 3e-4 | 共享 seed42 初始参数和 RNG 快照 |
| E2 | 无 warmup 的 Cosine，3e-4 → 1e-5 | 同一共享快照 |
| E3 | Warmup 4,000 + Cosine | 独立目录恢复原 epoch48 完整状态 |
| E4 | Warmup 10,000 + Cosine，峰值 3e-4、最低 1e-5 | 同一共享快照 |

所有 Cosine 的计划周期仍是 80 轮、427,200 次更新，每轮 5,340 次更新。E2 的第一次实际更新使用峰值，计划最后一次使用最低值，公式相位为 `(u-1)/(T-1)`；带 warmup 的 E3/E4 继续使用原有 `(u-w)/(T-w)`。Batch32、AdamW、weight decay 0.01、clip 1、seed42、K3、16,479,940 参数和 FP32/Math Attention/TF32off 保持一致。

原 E3 在第 48 轮早停，最佳 epoch43，CE 0.09594137142809196。原训练目录、报告、全部保留权重均不修改。新目录接续 epoch48 的模型、AdamW 状态、更新时钟、RNG、patience 和验证历史，获得真实 epoch49+；不会重启 warmup。新 E3 选择最低 CE：若没有改善，仍指向原 epoch43；若改善，只在新目录记录新最佳。报告同时保留原 E3 的固定基线值，明确标识独立接续。历史 E3 没有保存初始参数快照，因此只依据冻结 seed/构造代码保持初始化方案一致，不声称已直接比对历史初始张量。

四组按 CE 选定的最佳权重在同一完整 validation9,493 上使用 Beam32、最大长度96、元素＋基础 SMILES 语法约束生成。保留原词表；在线约束重原子及两个甘油端口，最终由 RDKit/组装检查全部元素和 H。无强制修复、分子式变更或可行集概率重新归一化。在线约束的原始排序和后验公式过滤、图去重后的指标分开报告。空结果仍保留在分母。

主要生成指标为 Exact@10、Similarity@10，同时保留 Exact@1/5、分子式覆盖、结构有效率、官能团 F1、相对最佳输入模板的收益、P/nonP 分层、时间和显存。模板检索依赖目标结构，因此属于结构预训练对照，不代表真实谱图识别性能。单 seed 的调度对照也不能作为跨随机种子的稳定性证据。

PNG/SVG 展示实际、无平滑的训练 CE、验证 CE 和实际学习率，共同窗口 1–50；另给完整验证历史图，避免遗漏第 50 轮之后的最低点。运行中未发生的轮次留空，不补值。E3 在 epoch48 用虚线标识新目录接续点。

所有新运行使用独立目录 `outputs/core_lr_schedule_study_20261028`，只保留每组 best＋latest2 全状态权重。CPU/CUDA 均验证真实磁盘断点恢复，CUDA 后向和 Beam32 探针通过后再由 Windows GPU 监督器串行执行。保留全部旧停止锁，无故障自动重试。结构、原始候选、权重和缓存留在 G；完成后自动发布聚合 Markdown/JSON 和 PNG/SVG，公开代码可复现实验。

入口：`scripts/prepare_core_lr_study.py`、`scripts/gate_core_lr_study.py`、`scripts/supervise_core_lr_study.py`、`scripts/run_core_lr_study.py`。聚合报告：`docs/results/CORE_LR_SCHEDULE_STUDY_20261028.{md,json}`。`PUBLICATION_DONE.json` 给出最终提交 SHA；失败保留 `PUBLICATION_PENDING.json`，不会触发 GPU 重跑。
