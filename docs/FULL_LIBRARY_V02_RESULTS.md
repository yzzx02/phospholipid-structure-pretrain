# 2M 正式完整结构库与检索审计结果

2026-10-16。正式完整结构库及其独立质量审计已经完成。
正式 Morgan benchmark、全库检索可用性统计和查询 API 抽查均已完成。
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


### 全部 2M 目标的模板可用性

| 可用模板条件 | 目标数 | 比例 |
|---|---:|---:|
| 至少 1 个 | 2,000,000 | 100.000000% |
| 至少 5 个 | 1,999,998 | 99.999900% |
| 10 个 | 1,999,995 | 99.999750% |
| 零模板 | 0 | 0.000000% |

五个未满十个模板的目标分别获得 **3、4、6、7、9** 个模板。
它们均来自 PubChem baseline 的 phosphate / EXTENDED，单独保存在本地
`targets_with_fewer_than_10_templates.parquet`。没有零模板目标，因此无需追加零模板 exact 扫描。

下列比例来自全库逐组查询后映射到每个 canonical 图，未以 benchmark 比例外推。
所有返回分数均经 RDKit Tanimoto 重排，排除了相同 canonical 目标。
不同图共享 Morgan 指纹时 similarity=1，仍是不同的合法模板。

| 目标分组 | 数量 | ≥1 个 | ≥5 个 | 10 个 | 零模板 |
|---|---:|---:|---:|---:|---:|
| family: phosphate | 1,192,847 | 100.000000% | 99.999832% | 99.999581% | 0 |
| family: phosphonate | 552,565 | 100.000000% | 100.000000% | 100.000000% | 0 |
| family: polyphosphate | 254,588 | 100.000000% | 100.000000% | 100.000000% | 0 |
| tier: CORE | 793,200 | 100.000000% | 100.000000% | 100.000000% | 0 |
| tier: EXTENDED | 1,206,800 | 100.000000% | 99.999834% | 99.999586% | 0 |
| source: PubChem-derived | 1,360,018 | 100.000000% | 99.999853% | 99.999632% | 0 |
| source: generated_phosphate_analog | 639,969 | 100.000000% | 100.000000% | 100.000000% | 0 |
| source: generated_keto_inositol | 13 | 100.000000% | 100.000000% | 100.000000% | 0 |
| origin: pubchem_baseline | 1,011,109 | 100.000000% | 99.999802% | 99.999505% | 0 |
| origin: pubchem_multi_attachment_split | 348,909 | 100.000000% | 100.000000% | 100.000000% | 0 |
| origin: generated_phosphate_analog | 639,969 | 100.000000% | 100.000000% | 100.000000% | 0 |
| origin: generated_keto_inositol | 13 | 100.000000% | 100.000000% | 100.000000% | 0 |

### 相似度分布

最大相似度的总体均值为 **0.965559**、中位数为 **1.000000**；
top-10 mean similarity 的总体均值为 **0.888487**、中位数为 **0.898689**。
不足十个时取实际可用模板的均值，不补零。

| 分位数 | Max similarity | Top-10 mean similarity |
|---|---:|---:|
| 0% | 0.424242 | 0.408838 |
| 1% | 0.793651 | 0.729686 |
| 5% | 0.853659 | 0.784784 |
| 25% | 0.948718 | 0.863441 |
| 50% | 1.000000 | 0.898689 |
| 75% | 1.000000 | 0.925165 |
| 95% | 1.000000 | 0.952650 |
| 99% | 1.000000 | 0.981928 |
| 100% | 1.000000 | 1.000000 |

| 目标分组 | Max 相似度均值 | Max 中位数 | Top-10 mean 均值 | Top-10 mean 中位数 |
|---|---:|---:|---:|---:|
| family: phosphate | 0.968443 | 1.000000 | 0.898017 | 0.909469 |
| family: phosphonate | 0.957892 | 0.987952 | 0.864311 | 0.873800 |
| family: polyphosphate | 0.968692 | 0.989011 | 0.896311 | 0.906252 |
| tier: CORE | 0.960146 | 1.000000 | 0.874287 | 0.884098 |
| tier: EXTENDED | 0.969117 | 1.000000 | 0.897821 | 0.907847 |
| source: PubChem-derived | 0.964590 | 1.000000 | 0.885427 | 0.894305 |
| source: generated_phosphate_analog | 0.967620 | 1.000000 | 0.894991 | 0.906948 |
| source: generated_keto_inositol | 0.947170 | 1.000000 | 0.865907 | 0.861070 |
| origin: pubchem_baseline | 0.963332 | 1.000000 | 0.882284 | 0.891182 |
| origin: pubchem_multi_attachment_split | 0.968235 | 1.000000 | 0.894536 | 0.904108 |
| origin: generated_phosphate_analog | 0.967620 | 1.000000 | 0.894991 | 0.906948 |
| origin: generated_keto_inositol | 0.947170 | 1.000000 | 0.865907 | 0.861070 |

这些结果衡量当前完整库的结构模板可用性；模型尚未训练。
benchmark 召回是 reference subset 上的实测，全库 Top-K 为经基准检验的近似索引结果，
未执行全库两两 Tanimoto。全库找到的每个可用模板均满足实际精确分数 ≥0.4。

### 查询 API 复核

独立调用 `IndexedMorganRetriever.retrieve_smiles()` 核对 **105 个目标**，
包括 100 个随机目标、全部五个未满十个模板的目标，以及最低相似度目标。
API 返回的模板数、最大分数和均值与全库审计一致；返回 ID 不重复、均排除目标自身，
所有分数再次以 RDKit 逐对独立核对通过。结果保存在 `query_api_verification.json`。

索引文件大小为 747,767,440 bytes，构建约 1,357.7 秒。
模型训练和真实训练数据适配、划分及模板泄漏规则仍留待后续讨论。

## 本地输出与测试

完整库：`data/structure_library_v0.2/`。
指纹缓存、索引、benchmark 和逐目标检索指标：`data/private/morgan_readiness_v0.1/`。
输入、原始来源库、完整结构与批量指标不上传 GitHub；只同步代码、方法和聚合报告。

七组独立离线测试共 **217 项通过**，Ruff lint/format 通过。
本轮未训练、未添加 MS/MS、未随意 random split。
