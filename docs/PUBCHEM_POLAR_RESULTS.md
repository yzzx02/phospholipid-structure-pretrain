# PubChem 极性片段扩展：实际执行结果

**全量属性采集尚未完成：目标 983,258 CID，已取得 583,937，剩余 399,321。**
PubChem 请求链路持续返回 HTTP 429；冷却后的单 CID 请求仍被限流。以下为缓存快照，不宣称全量建库完成。
审核快照生成时间：2026-10-15T08:39:37Z；RDKit 2026.03.3。
结果范围是配置中的三条 SMARTS 查询，不能等同于所有含磷化学空间。
本轮未训练模型、未生成最终 150 万完整脂质库、未生成 approved polar 表。

## MS-DIAL 2025 冻结链表

只使用既有 2025 证据，保留单一位置版本：A 177 / O 84 / P 61，共 **322**。
排除 A-24:1、A-26:1、A-28:1、A-30:1、A-32:1、A-34:1。
原有双键位置全部保留，322 条均通过现有 builder 校验；未重新处理 2022。

## 分阶段实测

前两阶段是 API 截取样本，并非随机抽样；full 实际结果均低于一百万上限，未触发拆分。

| 阶段 | 每 query 上限 | 目标去重 CID | 已取得属性 CID | 唯一片段 | APPROVABLE | 属性完整 |
|---|---:|---:|---:|---:|---:|---|
| stage1 | 1,000 | 2,794 | 2,794 | 3,397 | 770 | True |
| stage2 | 10,000 | 29,134 | 29,134 | 40,414 | 6,235 | True |
| full | 1,000,000 | 983,258 | 583,937 | 490,714 | 107,803 | False |

## 查询与重叠

| Query | SMARTS | full CID | 状态 |
|---|---|---:|---|
| phosphate_acid | `[P;X4](=[O])([O;H1,-1])([O])[O]` | 789,897 | COMPLETE |
| phosphonate_acid | `[P;X4](=[O])([O;H1,-1])([O])[#6]` | 205,237 | COMPLETE |
| polyphosphate_acid | `[P;X4](=[O])([O;H1,-1])([O])[O][P]` | 105,835 | COMPLETE |

查询并集 **983,258 CID**；已取得属性 **583,937**，缺失 **399,321**。
缺失 CID 单独列在报告 missing_property_cids，不作为化学拒绝。

- `phosphate_acid|phosphonate_acid`：11,876 CID。
- `phosphate_acid|polyphosphate_acid`：105,835 CID。
- `phosphonate_acid|polyphosphate_acid`：6,365 CID。

配置版本 `pubchem-polar-v0.1-neutral-coverage`；SHA-256 `3b99cff6dd5d68f3098ffd2f2814966e65b5bee66f710bbeb333f4baf9f7d9d8`。

## 片段与审核状态

表中总数包含全部审核状态，不是最终批准头基数。

| 化学 family | 唯一审核片段 | APPROVABLE 子集 |
|---|---:|---:|
| phosphate | 317,776 | 42,169 |
| phosphonate | 97,035 | 64,644 |
| polyphosphate | 75,903 | 990 |

唯一审核片段共 **490,714**；原始 attachment candidates 943,377，`(CID, distinct fragment)` 对数 673,562。

位点候选→片段去重率 **47.98%**；CID-fragment 对→片段去重率 **27.15%**。
一个 CID 可产生多个不等价连接结果，因此不能简单把片段数除以 CID 数解释成去重率。

| 主状态 | 片段数 |
|---|---:|
| APPROVABLE | 107,803 |
| MULTIPLE_ATTACHMENT_AMBIGUITY | 42,688 |
| CHARGE_AMBIGUOUS | 80,247 |
| FILTER_REVIEW | 259,976 |

电荷审核标记包括两种情况：代表形式净电荷非零 71,574 个；净电荷为零但有不等价电荷位置 8,673 个。
其中 76,360 个还同时超出软阈值，67,913 个同时带长链标记。
所以不能把全部电荷审核条目解释成小头基的质子化歧义；完整脂质/大结构也会被宽查询命中。
永久正电未被强行抹去；这些重叠标记保留供后续审核。

APPROVABLE 只表示自动检查通过，仍待人工审核。电荷、超限和连接歧义的其他标记也保留。
宽松查询还会保留满足价态但需要反应性审核的记录，例如 O–O 环片段（CID 15920549）、
O–F（CID 3014593）和 O–Cl（CID 18539471）。本轮未暗中增加排除条件；
这些记录不是最终批准头基，下一步审核规则需要明确是否排除这类反应性键。
片段默认 A/A、O/A 开放，P/A 关闭；仅精确 PE control 可允许 P/A。
审核 family 尚未映射成 builder 的旧枚举，不能直接把 review.tsv 当 approved 表输入。

## 组分与拒绝

通过组分硬检查的单一含 P 组分记录：506,065；可生成 attachment 的 CID：491,278。
移除简单 counterion 的记录：23,729；离子数：50,733。

| 拒绝原因 | CID 数 |
|---|---:|
| unsupported_cocomponent | 41,717 |
| multiple_p_components | 18,203 |
| unsupported_phosphorus_chemistry | 14,776 |
| unsupported_elements | 13,295 |
| isotope_labels | 4,022 |
| radicals | 573 |
| parse_or_sanitize_failure | 62 |
| no_acidic_attachment | 11 |

软阈值超限保留为审核，而非直接批准；各标记可重叠：

- `molecular_weight`：315,572 片段。
- `carbon_count`：276,594 片段。
- `long_aliphatic_carbon_run`：234,786 片段。
- `heavy_atoms`：210,355 片段。
- `phosphorus_count`：42,314 片段。

## 描述符

MW/分子式属于开放连接片段，不包含未来 glycerol-O；软阈值判断使用选中的酸前体组分。

| 描述符，全部审核片段 | min | median | p95 | max |
|---|---:|---:|---:|---:|
| molecular_weight | 76.999 | 782.893 | 1934.545 | 8983.410 |
| carbon_count | 0.000 | 38.000 | 90.000 | 307.000 |
| phosphorus_count | 1.000 | 1.000 | 6.000 | 50.000 |

含 N：309,948；含 S：48,789；含环：250,380。这些集合可重叠。

## 对照与验证

14 个本地化学/SMARTS 对照全部通过：9 正、5 负。13 个真实 PubChem 名称对照全部通过；
含铜金属负对照使用明确标注的本地构造记录，不虚构 CID。
9 类正对照在各阶段实际检索结果中全部找回：PA、PE、PC、PS、PG、PI、
aminoethylphosphonate、phosphonoacetate、pyrophosphate。

指定正对照 CID 也在 PubChem 服务端请求中直接找回，不能仅靠盐形式的同片段回收代替覆盖校验。
49 项 PubChem 离线测试、48 项 builder 回归和 39 项链提取测试通过；Ruff 检查通过。
各阶段已取得属性的产物逐条跨表核验通过，详见各目录 verification_audit.json；
核对 source CID 外键、原始/候选/拒绝分区、位点/来源聚合数、全部 fragment P/dummy 边界。
独立 GitHub CI 只运行离线验证，不触发训练。

## 20 个对照之外的代表扩展片段

均来自本次真实 PubChem CID，且为 APPROVABLE 候选；“扩展”不表示文献新颖性。
展示按 family 轮转，优先含碳、无上述异常氧单键/三四元环的例子；仅影响展示，不删审核数据。
这些是连接片段，不能作为独立完整分子直接解释分子式或电荷。

| # | family | fragment SMILES | 来源 CID 示例 | 合并 CID 数 |
|---:|---|---|---|---:|
| 1 | phosphonate | `CP(=O)(O)[*:1]` | [13818](https://pubchem.ncbi.nlm.nih.gov/compound/13818), [14671056](https://pubchem.ncbi.nlm.nih.gov/compound/14671056), [14959140](https://pubchem.ncbi.nlm.nih.gov/compound/14959140) | 21 |
| 2 | phosphate | `COP(=O)(O)[*:1]` | [13130](https://pubchem.ncbi.nlm.nih.gov/compound/13130), [11970392](https://pubchem.ncbi.nlm.nih.gov/compound/11970392), [12691389](https://pubchem.ncbi.nlm.nih.gov/compound/12691389) | 17 |
| 3 | polyphosphate | `CP(=O)(O)OP(C)(=O)[*:1]` | [431093](https://pubchem.ncbi.nlm.nih.gov/compound/431093), [140742682](https://pubchem.ncbi.nlm.nih.gov/compound/140742682), [164703365](https://pubchem.ncbi.nlm.nih.gov/compound/164703365) | 3 |
| 4 | phosphonate | `O=P(O)(CO)[*:1]` | [75795](https://pubchem.ncbi.nlm.nih.gov/compound/75795), [21680357](https://pubchem.ncbi.nlm.nih.gov/compound/21680357), [23280505](https://pubchem.ncbi.nlm.nih.gov/compound/23280505) | 8 |
| 5 | phosphate | `O=P1([*:1])OC=CO1` | [57652994](https://pubchem.ncbi.nlm.nih.gov/compound/57652994) | 1 |
| 6 | polyphosphate | `COP(C)(=O)OP(=O)(O)[*:1]` | [117627781](https://pubchem.ncbi.nlm.nih.gov/compound/117627781) | 1 |
| 7 | phosphonate | `C#CP(=O)(O)[*:1]` | [18694015](https://pubchem.ncbi.nlm.nih.gov/compound/18694015), [18694016](https://pubchem.ncbi.nlm.nih.gov/compound/18694016) | 2 |
| 8 | phosphate | `N=COP(=O)(O)[*:1]` | [53801550](https://pubchem.ncbi.nlm.nih.gov/compound/53801550), [122198540](https://pubchem.ncbi.nlm.nih.gov/compound/122198540) | 2 |
| 9 | polyphosphate | `O=P1(O)CCCP(=O)([*:1])O1` | [53991066](https://pubchem.ncbi.nlm.nih.gov/compound/53991066) | 1 |
| 10 | phosphonate | `CCP(=O)(O)[*:1]` | [204482](https://pubchem.ncbi.nlm.nih.gov/compound/204482), [7204818](https://pubchem.ncbi.nlm.nih.gov/compound/7204818), [20480863](https://pubchem.ncbi.nlm.nih.gov/compound/20480863) | 12 |
| 11 | phosphate | `O=P1([*:1])OCCO1` | [13168503](https://pubchem.ncbi.nlm.nih.gov/compound/13168503), [23416182](https://pubchem.ncbi.nlm.nih.gov/compound/23416182), [23692348](https://pubchem.ncbi.nlm.nih.gov/compound/23692348) | 5 |
| 12 | polyphosphate | `O=P1(O)OCOP(=O)([*:1])O1` | [18459455](https://pubchem.ncbi.nlm.nih.gov/compound/18459455), [87770385](https://pubchem.ncbi.nlm.nih.gov/compound/87770385) | 2 |
| 13 | phosphonate | `O=P(O)(CS)[*:1]` | [445446](https://pubchem.ncbi.nlm.nih.gov/compound/445446), [5288817](https://pubchem.ncbi.nlm.nih.gov/compound/5288817), [163405750](https://pubchem.ncbi.nlm.nih.gov/compound/163405750) | 3 |
| 14 | phosphate | `O=P(O)(OCBr)[*:1]` | [130453849](https://pubchem.ncbi.nlm.nih.gov/compound/130453849) | 1 |
| 15 | polyphosphate | `COP(C)(=O)OP(C)(=O)[*:1]` | [53717652](https://pubchem.ncbi.nlm.nih.gov/compound/53717652) | 1 |
| 16 | phosphonate | `O=P(O)(CI)[*:1]` | [11736185](https://pubchem.ncbi.nlm.nih.gov/compound/11736185) | 1 |
| 17 | phosphate | `O=P(O)(OCO)[*:1]` | [19914290](https://pubchem.ncbi.nlm.nih.gov/compound/19914290), [20754355](https://pubchem.ncbi.nlm.nih.gov/compound/20754355), [21275652](https://pubchem.ncbi.nlm.nih.gov/compound/21275652) | 6 |
| 18 | polyphosphate | `N#CP(=O)(O)OP(=O)(C#N)[*:1]` | [88162852](https://pubchem.ncbi.nlm.nih.gov/compound/88162852) | 1 |
| 19 | phosphonate | `O=P(O)(CF)[*:1]` | [13736774](https://pubchem.ncbi.nlm.nih.gov/compound/13736774), [59533630](https://pubchem.ncbi.nlm.nih.gov/compound/59533630) | 2 |
| 20 | phosphate | `O=P(O)(OCF)[*:1]` | [20614582](https://pubchem.ncbi.nlm.nih.gov/compound/20614582), [59669328](https://pubchem.ncbi.nlm.nih.gov/compound/59669328), [140799717](https://pubchem.ncbi.nlm.nih.gov/compound/140799717) | 4 |

## 代码与本地输出

- `src/structure_pretrain/pubchem_polar/`：化学、API/cache、阶段流水线和链冻结。
- `scripts/harvest_pubchem_polar.py`、`scripts/audit_pubchem_polar.py`。
- `configs/pubchem_polar_queries.yaml`、`configs/known_head_controls.tsv`。
- `tests/pubchem_polar/`、`.github/workflows/pubchem_polar.yml`。
- 方法与复现：[PUBCHEM_POLAR.md](PUBCHEM_POLAR.md)。

审核包在 `data/private/pubchem_polar_review/`，包含需求的全部 8 类输出、对照、checkpoint 和核验。
冻结链表在 `data/private/msdial_chain_review/msdial_derived_chains_v0.1.tsv`。
完整原始数据与 SSD 缓存保留本地，不上传 GitHub；现有模型与 builder 代码未修改。

## 未完成范围与续传

尚缺 399,321 个 CID 的属性，名单在报告 `missing_property_cids`；接口限流证据在本地 `service_status.json`。
两次冷却后的单 CID 请求（16:38、16:47 本地时间）仍返回 429，均无 Retry-After。
本轮已停止在线请求；当前审查表仅包含属性已经入库的记录。

服务恢复后，在同一项目/缓存目录执行：

```bash
python scripts/harvest_pubchem_polar.py --stages full
```

已有属性和化学判定会复用；新请求仍受全局限速和持续限流暂停保护。
离线运行的网络请求统计为零，历史请求的 URL/时间保存在 SSD 响应缓存中。
正式 approved 导出和完整脂质组合仍待后续规则确认，本轮没有生成。
