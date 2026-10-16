# 第二层筛选与特殊结构核验：2026-10-16

本轮已生成正式 CORE + EXTENDED polar 表和 v0.2 完整结构组装程序。
没有训练，没有运行最终完整脂质库；只组装了 14 个特殊结构代表例。
冻结链表仍为 A177/O84/P61，总数 322，SHA-256 与上一阶段完全一致。

## 第二层筛选

已有 APPROVABLE 输入 107,803；不等待缺失的 399,321 CID，不继续采集。

| 原化学 family | CORE | EXTENDED | HOLD | REJECT |
|---|---:|---:|---:|---:|
| phosphate | 15,288 | 24,303 | 1,562 | 1,016 |
| phosphonate | 31,759 | 31,108 | 844 | 933 |
| polyphosphate | 278 | 661 | 8 | 43 |
| 合计 | **47,325** | **56,072** | **2,414** | **1,992** |

正式可用 **103,397** 个头基，逐条通过未修改的 v0.1 builder importer；
九类正对照全部为 CORE。只有 exact PE 开放 P/A。未按头基数量裁剪，没有 family 降级超过 90%。

反应性拒绝涉及 O–O 1,568、O–halogen 324、acyl halide 89、sulfonyl halide 18；
标记可能重叠，去重后 1,992 个。规则异常为 **0**。
HOLD 原因：卤素数超过 4 共 1,501、hetero fraction 低于 0.10 共 40，
另有 878 个化学阈值通过但触发现有 builder 完整酰基甘油磷脂子结构限制。
最后一类明确记录为兼容性 HOLD，没有冒充反应性拒绝。

| Profile | CORE | EXTENDED | HOLD | REJECT | 正式可用 |
|---|---:|---:|---:|---:|---:|
| conservative | 32,984 | 70,413 | 2,414 | 1,992 | 103,397 |
| default | 47,325 | 56,072 | 2,414 | 1,992 | 103,397 |
| broad | 66,804 | 36,593 | 2,414 | 1,992 | 103,397 |

三套 profile 只调整 CORE 的 MW/C 边界，EXTENDED 边界相同，所以正式总数相同。
Scaffold 共 37,723；最大 family 13,272 个片段。Scaffold 只用于统计，没有删结构。
私有输出含每层 20 个代表例、另 10 个 phosphonate 和 10 个杂环/多元醇/S 例，以及分层 spot-check。
逐项跨表核验 107,803 行的 ID、SMILES、source CID、分区及 approved TSV 哈希全部通过。

## 腈基定向核验

| motif | 所有状态片段 | 原 APPROVABLE | 第二层正式可用 | raw CID 命中 |
|---|---:|---:|---:|---:|
| P–O–CH2–CN | 13 | 9 | **7** | 24 |
| P–O–CH2–CH2–CN | 246 | 86 | **85** | 441 |

片段 Parquet 与 review TSV 的 motif 命中集合完全一致。raw 搜索扫描 583,937 个 CID；
其中 6 个含三键候选不能被 RDKit 解析，单独保存搜索拒绝记录，没有作为证据。

选择的两个真实片段：

| motif | 代表片段 | source CIDs |
|---|---|---|
| cyanomethyl | `N#CCOP(=O)(O)[*:1]` | 19077309, 21106896, 21916515, 23715241, 88413876 |
| 2-cyanoethyl | `N#CCCOP(=O)(O)[*:1]` | 78715, 4064087, 23704063, 88072783, 165177524 |

每个片段分别用 A-16:0/A-18:1、A-18:0/A-20:4、O-18:0/A-20:4 组装，共六个代表结构。
它们的 head 来源真实，具体完整图为程序组装；本轮没有给这些 exact full graphs 分配数据库或文献已报道标记。

进一步要求 nitrile 与长链 A/A 或 O/A glycerol 骨架连接到**同一个 P**，
在 raw 中找到 **16 个 CID**，均为 2-cyanoethyl 相关磷酸三酯候选，包含盐/多组分记录。
未找到本轮查询范围内对应的游离酸 phosphatidyl nitrile diester。
三酯候选需要核对保护基/最终产物状态，不能仅凭子结构匹配声称天然脂质，
也不能把它们等同于上述六个程序组装图。全部原始 SMILES 和 CID 保留在私有核验表。

定向文献核验中，[PG 合成路线](https://pmc.ncbi.nlm.nih.gov/articles/PMC7484214/)
使用 cyanoethyl phosphoramidite；
[α-cholesteryl phosphatidyl glucoside 合成](https://pmc.ncbi.nlm.nih.gov/articles/PMC4319363/)
明确在得到最终目标物之前移去 cyanoethyl 基。
这支持“保护基与最终头基必须分开”，不能证明上述 exact phosphatidyl nitrile diester 已报道。

## PI 头基氧化

五个非 phosphate-linked OH 位点生成五个单氧化图，去立体规范化后为 **三个 unique heads**：

| 距 phosphate-linked 环碳的图距离 | 片段 SMILES | 本地真实头基证据 |
|---:|---|---|
| 1 | `O=C1C(O)C(O)C(O)C(O)C1OP(=O)(O)[*:1]` | CID 10038111, 85100383, 101987983 |
| 2 | `O=C1C(O)C(O)C(O)C(OP(=O)(O)[*:1])C1O` | 当前快照无 exact head；由标准 PI 程序生成 |
| 3 | `O=C1C(O)C(O)C(OP(=O)(O)[*:1])C(O)C1O` | CID 91799223 |

图距离不是生化位号，也不保留 myo/scyllo 等立体命名。
两个候选本来已在正式表中；另一个允许作为可追溯生成补充，并通过同一套化学规则。
所有修改均保持 phosphate attachment 和环连接，头基及完整结构均严格满足仅 H−2。
标准 PI + 三个候选分别组装 A-18:0/A-20:4 与 A-16:0/A-18:1，共八个代表结构。

[myo-2-inosose 1-phosphate 研究](https://pubs.acs.org/doi/abs/10.1021/ja952570i)
和 [MI1P synthase 机制讨论](https://www.nature.com/articles/s42003-020-0814-7)
提供头基层面证据。
[Agranoff 等的早期原文索引](https://deepblue.lib.umich.edu/bitstream/2027.42/73945/1/j.1749-6632.1970.tb56440.x.pdf)
还提到 epi-/scyllo-inosose 与脑制备物的孵育和 phosphatidylinosose 验证实验，
同时说明回收率低及可能不稳定。它是值得保留的完整类别实验线索，不能直接确定本轮三个图的
氧化位点或具体链组成。直接 PDF 下载受到访问限制，检索索引可读取相关原文段落。
本轮未独立核实八个代表图中的任一 exact full phosphatidyl graph。

这与“脂肪链被氧化”的 oxidized PI 证据不同，后者未用于本问题。

## 完整库预算与程序状态

103,397 个真实 filtered heads，加一个生成的 keto-inositol 头基，未来输入共 **103,398**。
真实输入预检查通过：322 条链、41,679 个链组合；普通头基 A/A、O/A，只有 PE 开放 P/A。
七个共享 anchors 加必须保留的特殊 O/A 组合，预留 **723,788** 个结构位置，
剩余 **776,212**。若最终填满 1,500,000，上限对应平均约 **14.51** 个链背景/头基；
这是预算值，实际完整库结构数和实际覆盖仍待正式运行。

当前 14 个代表结构全部通过重新组装、canonical SMILES、分子式及来源字段核验，
5 个 special heads 已写入 must-include 表。来源字段不作为采样分层、标签或 loss 权重。
完整库入口默认只做预检查，需显式 `--execute` 才组装；本轮未运行。

完整方法见 [POLAR_PLAUSIBILITY.md](POLAR_PLAUSIBILITY.md)。私有数据路径：

- `data/private/pubchem_polar_plausibility_v0.1/`：正式头基、四层结果、敏感性、scaffold、示例与核验。
- `data/private/special_cases_v0.2/nitrile_special_cases.tsv`：六个完整腈基代表 SMILES 和全部来源字段。
- `data/private/special_cases_v0.2/keto_pi_special_cases.tsv`：八个 PI/keto-PI 代表 SMILES 和全部来源字段。
- `data/private/special_cases_v0.2/nitrile_raw_full_lipid_candidates.tsv`：16 个三酯候选 CID 的真实原始图。
- `configs/special_case_evidence_v02.tsv`：定向文献核验，未设 LEVEL；检索未覆盖数据库全库。

初次调试的输出单独归档为 draft，不是正式结果。原 PubChem review、HOLD、源库和 SSD 缓存均保留。

本地离线测试 **179 项通过**，Ruff 检查通过；筛选表和 14 个代表结构均完成独立跨表核验。
新增 CI 只运行化学模块与微型 fixture，不触发模型训练工作流。
