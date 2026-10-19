# v0.4 Polar-only：本机实测结果

生成时间（UTC）：2026-10-19T16:49:17.392650+00:00。需求：[Issue #1](https://github.com/yzzx02/phospholipid-structure-pretrain/issues/1)。
[方法与完整复现](../experiments/POLAR_ONLY_V04.md) · [聚合 JSON／配置与数据 SHA256](POLAR_ONLY_V04_RESULTS.json)

## 结论

v0.4 Polar-only改造、全库化学/检索审计、真实CPU测试、32头基GPU过拟合与checkpoint续训均已完成。FP/GINE均从新随机初始化完成20k池的500/1000/2000/3000步实测。本轮有界实验在3000步结束：最终greedy头基和完整图exact均为0，Top5仅3/512与1/512，原始公式匹配19/512与5/512。暂缓完整201.5万库预训练；5000步与50k池未执行，不据此断言它们无效。

当前是已知目标链、可信中性式及离线结构 oracle 检索下的 Polar 生成。不是 MS/MS 鉴定准确率，也不能把库中已有验证头基的命中称为新化合物发现。

## 已实现且通过测试

- 独立 v0.4 模型、dataset、训练入口、配置及 checkpoint 契约；无 Chain Head／Chain Loss。 311 项 pytest、Ruff、pip check 通过。
- FP/GINE 真实数据 CPU forward/backward、参数更新、无目标 Polar 标签的 greedy、worker加载通过。
- 201.5万行 attached-fragment 公式恒等式全部通过；负数／不匹配0。671条原审计（全部71特殊结构）及额外10,000条 TRAIN 跨家族／连接类型重建零失败。
- 全部390,612个无链 head/occupancy 变体的 canonical、dummy、glycerol–O–P、元素组成审计通过；独立重算512个图／指纹并核对所有 Arrow shard哈希。
- TCLS＋全部模板 `[B,K+1,512]` 保留，K3/5/10 padding／置换测试通过；多链候选按 B×N 独立公式及 memory，未创建假谱图。K5另用真实数据拟合TRAIN-only统计并通过FP/GINE CPU前向、反向和greedy；K5 GPU未执行。

## 待本地 GPU 运行

要求的两模型初始1000-step实验均已实测完成。
本轮还完成了两模型3000步扩展；5000步与50k池未执行，不自动启动完整预训练或MS/MS微调。

## 数据与检索审计

冻结 split：TRAIN1,813,451／validation100,792／test100,757；canonical Polar graph交集0。所有存储参考逐条验证 TRAIN-only、异头基、H/HC链条件与新指纹 Tanimoto。
阈值0.2，K3（缓存至10）；不足K保留真实数量。H／HC各64个穷举对照的最优top1分数命中率均100%。
下表为阈值后的覆盖；阈值前的可用参考数与全部相似度分位数见JSON。

| split | 模式 | ≥1 | ≥3 | ≥5 | zero |
|---|---|---:|---:|---:|---:|
| train | H | 1813353 | 1812402 | 1810467 | 98 |
| train | HC | 1813451 | 1813451 | 1813451 | 0 |
| validation | H | 100787 | 100731 | 100609 | 5 |
| validation | HC | 100792 | 100792 | 100792 | 0 |
| test | H | 100750 | 100702 | 100602 | 7 |
| test | HC | 100757 | 100757 | 100757 | 0 |

TRAIN／validation／test的显式fallback目标数分别为85／4／7；fallback后zero均为0。TRAIN实际H目标80.0013%、同链模板对79.9977%，所有模板对异头基。
初版CPU准备在常见lyso链上频繁穷举，已停止该CPU任务，改建排除该链的合法TRAIN候选索引；保留初版日志，未将该准备阶段算作GPU训练。

## GPU实测结果：32头基过拟合

32个不同TRAIN头基、相同已知链组合、全部H参考；随机初始化，dropout0、batch8、lr3e-4。
| 编码器 | step | EOS | 合法Polar | greedy Polar exact | Top5 exact |
|---|---:|---:|---:|---:|---:|
| fingerprint | 200 | 100.00% | 100.00% | 100.00% | 100.00% |
| graph | 200 | 100.00% | 100.00% | 93.75% | 100.00% |

两模型均真实从200步checkpoint恢复模型、优化器、RNG和消费样本游标，继续到201步。检查点reload logits一致（容差内），优化器step与目标step一致。这是TRAIN记忆诊断，不是未见头基准确率。

## GPU实测结果：有界真实 smoke

匹配20,000训练目标（18,112个头基）、2,000验证目标及固定512个不同验证头基。FP与GINE从新随机初始化开始，非v0.3权重；有效batch16，K3，lr1e-4、Math SDPA／FP32。
greedy从BOS开始；不强制EOS、不修复化学结构、不预先按公式筛选。

| 编码器 | step | EOS | 合法Polar | Polar exact | Polar式exact | 完整式exact（给定链） | Top5 Polar | TF-EOS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fingerprint | 500 | 99.80% | 25.00% | 0.00% | 0.00% | 0.00% | 0.20% | 77.40% |
| graph | 500 | 95.51% | 19.14% | 0.00% | 0.20% | 0.20% | 0.00% | 83.90% |
| fingerprint | 1000 | 97.27% | 28.12% | 0.00% | 0.00% | 0.00% | 0.00% | 88.60% |
| graph | 1000 | 98.05% | 26.95% | 0.00% | 0.00% | 0.00% | 0.00% | 90.40% |
| fingerprint | 2000 | 99.80% | 48.44% | 0.20% | 0.98% | 0.98% | 0.20% | 96.75% |
| graph | 2000 | 99.61% | 38.67% | 0.00% | 0.39% | 0.39% | 0.00% | 93.85% |
| fingerprint | 3000 | 99.61% | 42.38% | 0.00% | 3.71% | 3.71% | 0.59% | 97.75% |
| graph | 3000 | 99.80% | 37.11% | 0.00% | 0.98% | 0.98% | 0.20% | 95.90% |

完整图exact、模板复制率、RDKit合法率、TF token／CE、生成长度、检索top1/topK、目标相似度分位数，以及family／length／retrieval similarity分层均在聚合JSON中保留。

### 可选Top5完整公式过滤（给定链）

| 编码器 | step | 选中/512 | 覆盖率 | 完整命中数 | 总体exact | 选中内exact |
|---|---:|---:|---:|---:|---:|---:|
| fingerprint | 500 | 7 | 1.37% | 1 | 0.20% | 14.29% |
| graph | 500 | 6 | 1.17% | 0 | 0.00% | 0.00% |
| fingerprint | 1000 | 8 | 1.56% | 0 | 0.00% | 0.00% |
| graph | 1000 | 7 | 1.37% | 0 | 0.00% | 0.00% |
| fingerprint | 2000 | 29 | 5.66% | 1 | 0.20% | 3.45% |
| graph | 2000 | 28 | 5.47% | 0 | 0.00% | 0.00% |
| fingerprint | 3000 | 60 | 11.72% | 3 | 0.59% | 5.00% |
| graph | 3000 | 48 | 9.38% | 1 | 0.20% | 2.08% |

筛选只在同一组原始beam5中选首个完整式匹配者，未补充或修复候选。选中内准确率必须与覆盖率、总体准确率一起看。

## 失败原因与解释

- fingerprint最后实测节点3000步，512目标的raw失败分类：{"formula_mismatch": 198, "polar_or_builder_chemistry": 41, "smiles_syntax": 247, "formula_matches_wrong_structure": 19, "no_eos": 2, "attachment": 5}。
- graph最后实测节点3000步，512目标的raw失败分类：{"polar_or_builder_chemistry": 48, "smiles_syntax": 266, "formula_mismatch": 185, "attachment": 7, "no_eos": 1, "formula_matches_wrong_structure": 5}。

公式条件输入不等于严格计数约束；语法无效与合法但式不匹配需分别分析。32样本过拟合通过说明接线与优化能工作，不证明大规模泛化。所有结果只覆盖固定验证诊断集，不能外推至全库或实际谱图。

## v0.3参照与不可混比的条件

v0.3的5000步、同一512目标：FP/GINE合法Polar43.95%／48.44%，greedy Polar及完整图exact均0，Top5为4/512／0/512；完整式3/512／4/512。FP可选过滤后总体完整命中2/512。
v0.3需要预测链，v0.4已给定真实链；同时改变模板表示、公式和检索。因此完整式／完整图指标是不同条件下的描述性参照，不能归因为单一编码器改进。v0.3原始报告与checkpoint原样保留。

## 资源、故障监控与资产保留

本轮已记录22个GPU子任务：新增GPU dump 0，显示驱动重置 0，watchdog停止 0。未改变TDR、驱动或核心依赖。

- fingerprint训练任务峰值VRAM allocated：0.624GiB；峰值进程RAM 1.909GiB。
- graph训练任务峰值VRAM allocated：0.593GiB；峰值进程RAM 1.936GiB。

v0.3保留审计：passed；589个既有代码/输出、38个旧checkpoints及320个冻结文件（4,334,776,708 bytes）无内容变化。所有新大数组、检索索引、逐目标预测与模型权重均在WSL Linux文件系统，未上传。
Math路径的本次有界稳定运行不证明此前默认attention黑屏已经永久修复。触发GPU dump、显示重置、黑屏通知或stall后应停止，不自动重试。

## 配置与复现

[FP配置](../../configs/polar_only_v04_fp.yaml) · [GINE配置](../../configs/polar_only_v04_gine.yaml) · [训练入口](../../train_polar_only_v04.py) · [Windows watchdog](../../scripts/supervise_polar_only_v04.ps1)
具体命令见方法；所有实际训练／评价节点、checkpoint SHA、数据/检索/标准化/代码SHA、软件版本在JSON中。Test仅作固定数据/覆盖审计，未用于训练、生成评价或选择。未执行CL四链、完整201.5万预训练或MS/MS微调。
