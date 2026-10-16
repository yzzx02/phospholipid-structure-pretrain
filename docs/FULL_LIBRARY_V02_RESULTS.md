# 2M 正式完整结构库与检索审计结果

2026-10-16。正式完整结构库及其独立质量审计已经完成。
正式 Morgan benchmark 已通过，全库索引与检索指标正在运行。
方法与复现入口见 [FULL_LIBRARY_V02.md](FULL_LIBRARY_V02.md)。
本轮未训练模型，未创建训练/验证/测试划分。

## 用户修订后的 P 规则与预算

按用户后续明确指示，取消 61 个 P token 全覆盖及每种至少三次的要求，
不增加 P/A mandatory pairs，不为 P 覆盖调整主体采样。
P/A 只保留为 exact PE 的少量特殊分支；非 PE 使用既有 O 链表示及 O 链池。
冻结 A177/O84/P61 表与代表双键位置保持不变。

重新枚举的 2M 配对 schedule SHA-256 与上一轮原始 dry-run 完全一致：
`865dfc9b3a37d48f562ea7b74da2be775beae11d25a6383bac8f3acd125da112`。
每头基三组共同 A/A anchors、五个特殊头基、generated keto-inositol 及 2:1:1 family extras 权重均保留。

## 正式组装与全局去重

| 指标 | 实测 |
|---|---:|
| Planned head/chain pairs | 2,000,000 |
| Successfully assembled | 2,000,000 |
| Sanitize rejected | 0 |
| Formula rejected | 0 |
| Other rejected | 0 |
| Canonical duplicates | 0 |
| **Final unique full structures** | **2,000,000** |

这里的 2,000,000 已是生成分子图、sanitize、元素/中性分子式校验和全局 canonical 去重后的实际数。
没有因为最终数减少而补库，也没有删除 phosphonate。
候选组装及全局导出约 3,481 秒，使用四个 CPU worker；计时不含输入预检查。

| Chemistry family | 2M dry-run | 最终 unique | 最终占比 |
|---|---:|---:|---:|
| phosphate | 1,192,847 | 1,192,847 | 59.64235% |
| phosphonate | 552,565 | 552,565 | 27.62825% |
| polyphosphate | 254,588 | 254,588 | 12.72940% |

| 链架构 | 最终完整结构数 |
|---|---:|
| AA | 1,232,779 |
| OA | 567,218 |
| PE-P/A | 3 |
| Lyso | 200,000 |

## 头基、链与特殊结构的最终真实覆盖

195,305 个 expanded heads 加一个既有 generated keto-inositol，
**195,306 / 195,306 个头基全部覆盖**。
每头基完整结构数 min=7、median=8、max=15；共同 A/A anchors 最少仍为 3。

| 冻结链类型 | 实际覆盖 | 每 token 完整结构数范围 | 是否为覆盖硬门槛 |
|---|---:|---:|---|
| A | 177 / 177 | 6,406–599,824 | 是 |
| O | 84 / 84 | 5,255–8,778 | 是 |
| P | 3 / 61 | 0–1 | 否，遵循最新用户要求 |

P-22:2、P-22:4、P-24:4 各出现一次，其余 58 个 P token 为零。
这三个图均属于 exact PE-P/A。
全部 322 个 token 的逐项计数保存在本地 `chain_coverage_report.json`，未把 P 未覆盖视为审计失败。

| 必选特殊头基 ID | 最终完整结构数 |
|---|---:|
| PCP_43385b3802b3fc6a | 14 |
| PCP_4ff7b8724e2b702c | 14 |
| PCP_9bc81235ad9408a5 | 15 |
| PCP_9fa3623f8e27fc53 | 15 |
| special_7c3f6de8af58648723a1 | 13 |
| **合计** | **71** |

这些完整结构为程序组装代表图，不作为数据库或文献已报道 exact full lipid 的证据。

## 来源与 plausibility tier

| 来源 | 最终完整结构数 |
|---|---:|
| PubChem baseline polar | 1,011,109 |
| PubChem multi-attachment rescued polar | 348,909 |
| Generated phosphate analog polar | 639,969 |
| Generated keto-inositol polar | 13 |

CORE 793,200；EXTENDED 1,206,800。
生成 analog 的 direct source CIDs 为空，父分子 CID 单独保存；来源不参与标签或权重。
有真实 PubChem 极性片段来源不代表对应完整脂质已在实验中发现。
沿用此前采集快照，本轮未继续联网补齐 PubChem。

## 独立质量审计

`verification_audit.json` 为 passed。
全部 2,000,000 条最终记录逐条检查 formula 元素向量、structure_id 哈希、头基关联、
来源标记和 P 仅用于 exact PE；重新统计的 family、tier、origin、头基、链与特殊结构计数均匹配正式报告。

随机 300 个图，加上全部 71 个 special 和 3 个 PE-P/A，
共 **374 个完整图**从冻结片段独立重新组装并核对 canonical SMILES、formula 和重新计算的 Morgan 指纹。
这是对全部最终记录的字段审计加 374 个图的独立重建，不将其称为二次重建整个 2M 库。

`structures.parquet`：246,622,744 bytes。
SHA-256：`48ee6cca87cef1fce49cd92407f540eee7ed6dd254a0d1f5edb6141913290db4`。
组装实现提交：`0e8fece192f9b1f6ee8f4629e71c4924d5439168`。
执行参数与源码哈希保存在本地 `execution_provenance.json`。

## Morgan 检索审计

正式流程先做随机 10,000 targets / 20,000 reference 的 exact benchmark，
通过后才构建全库索引并统计所有 target 的模板可用性。
开发 smoke 的结果不用于本节的正式指标。

正式 reference subset 含 19,912 个不同指纹；执行 200,000,000 次 exact Tanimoto 比较，
exact 阶段实测约 91.46 秒。

| Benchmark 指标 | 实测 |
|---|---:|
| Strict ID recall@10 | 99.7770% |
| Tie-aware recall@10 | 99.7980% |
| Top-1 score recall | 99.5900% |
| Mean score regret@10 | 0.000153351 |

USearch connectivity=32、expansion_add=256。
按预设逐步提高搜索深度后，选择 expansion_search=1024、native candidate groups=512、
RDKit 精确重排 16 个最近指纹组。较小设置的完整结果保存在本地 `exact_benchmark.json`，
不通过降低验收阈值选择设置。
以上是随机 reference subset 上的实测召回，不宣称全库检索为严格 exact。

全库 2,000,000 个 canonical 图对应 **1,404,613 个不同 Morgan 指纹**。
相同指纹下的不同结构全部保留为可用模板；本轮不将指纹相同视为结构相同。

## 本地输出与测试

完整库：`data/structure_library_v0.2/`。
指纹缓存、索引、benchmark 和逐目标检索指标：`data/private/morgan_readiness_v0.1/`。
输入、原始来源库、完整结构与批量指标不上传 GitHub；只同步代码、方法和聚合报告。

七组独立离线测试共 **217 项通过**，Ruff lint/format 通过。
本轮未训练、未添加 MS/MS、未随意 random split。
